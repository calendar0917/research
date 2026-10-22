"""Merge frozen-model predictions with support results and write per-molecule diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
GROUP_ORDER = ("seen_ordinary", "seen_tail", "unseen_ordinary", "unseen_tail")


def read_support() -> list[dict]:
    rows = list(csv.DictReader(open(ANALYSIS / "support_per_molecule.csv",
                                    newline="", encoding="utf-8")))
    if len(rows) != 1000:
        raise RuntimeError("support CSV must have 1000 rows")
    return rows


def as_bool(x) -> bool:
    return str(x).lower() in ("true", "1")


def metrics(a: np.ndarray, n: np.ndarray) -> dict[str, np.ndarray]:
    """Per-molecule prediction diagnostics over the legal representations."""
    a = np.asarray(a, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)
    valid = np.arange(a.shape[1])[None, :] < n[:, None]
    orig = a[:, 0]
    alt = valid & (np.arange(a.shape[1])[None, :] >= 1)
    with np.errstate(all="ignore"):
        mean_all = np.nansum(np.where(valid, a, np.nan), axis=1) / n
        mean_alt = np.where(n > 1,
                            np.nansum(np.where(alt, a, np.nan), axis=1) / np.maximum(n - 1, 1),
                            orig)
        lo = np.nanmin(np.where(valid, a, np.nan), axis=1)
        hi = np.nanmax(np.where(valid, a, np.nan), axis=1)
        abs_alt = np.abs(np.where(alt, a, np.nan) - orig[:, None])
        mac = np.nanmax(abs_alt, axis=1)
        mma = np.nanmean(abs_alt, axis=1)
        std = np.nanstd(np.where(valid, a, np.nan), axis=1)
    return {
        "orig": orig,
        "mean_all": mean_all,
        "mean_alt": mean_alt,
        "min": lo,
        "max": hi,
        "range": np.where(n > 1, hi - lo, 0.0),
        "std": np.where(n > 1, std, 0.0),
        "max_abs_change": np.where(n > 1, mac, 0.0),
        "mean_abs_change_alt": np.where(n > 1, mma, 0.0),
        "n_alt": np.maximum(n - 1, 0),
    }


def block(rows: Sequence[dict], y: np.ndarray, m: dict[str, dict[str, np.ndarray]],
          idx: np.ndarray, model: str) -> dict:
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if n == 0:
        return {"n": 0}
    d = m[model]
    err_orig = np.abs(d["orig"][idx] - y[idx])
    err_mean = np.abs(d["mean_all"][idx] - y[idx])
    mac = d["max_abs_change"][idx]
    return {
        "n_molecules": int(n),
        "n_with_alternative": int((d["n_alt"][idx] > 0).sum()),
        "n_no_alternative": int((d["n_alt"][idx] == 0).sum()),
        "mean_pred_original": float(np.mean(d["orig"][idx])),
        "mean_pred_all_rep_mean": float(np.mean(d["mean_all"][idx])),
        "mean_max_abs_change": float(np.mean(mac)),
        "median_max_abs_change": float(np.median(mac)),
        "p90_max_abs_change": float(np.quantile(mac, 0.9)),
        "max_max_abs_change": float(np.max(mac)),
        "mean_mean_abs_change_alt": float(np.mean(d["mean_abs_change_alt"][idx])),
        "mean_pred_range": float(np.mean(d["range"][idx])),
        "mean_pred_std": float(np.mean(d["std"][idx])),
        "n_max_change_gt_0.01": int((mac > 0.01).sum()),
        "n_max_change_gt_0.05": int((mac > 0.05).sum()),
        "n_max_change_gt_0.10": int((mac > 0.10).sum()),
        "mae_original": float(np.mean(err_orig)),
        "mae_all_rep_mean_fixed_diagnostic": float(np.mean(err_mean)),
        "delta_mae_all_rep_mean_minus_original": float(np.mean(err_mean - err_orig)),
        "n_abs_error_improved_by_rep_mean": int((err_mean < err_orig).sum()),
        "n_abs_error_worsened_by_rep_mean": int((err_mean > err_orig).sum()),
    }


def subgroup_indices(rows: Sequence[dict]) -> dict[str, list[int]]:
    def flags(i):
        r = rows[i]
        return (as_bool(r["full_orig"]), as_bool(r["full_rep"]),
                as_bool(r["full_arom"]), as_bool(r["full_union"]),
                int(r["rescued_union"]))

    out = {
        "unseen_ordinary_all": [i for i, r in enumerate(rows)
                                if r["old_group"] == "unseen_ordinary"],
        "unseen_ordinary_rescued_any": [i for i, r in enumerate(rows)
                                        if r["old_group"] == "unseen_ordinary"
                                        and int(r["rescued_union"]) > 0],
        "unseen_ordinary_no_rescue": [i for i, r in enumerate(rows)
                                      if r["old_group"] == "unseen_ordinary"
                                      and int(r["rescued_union"]) == 0],
        "union_full_support_gain": [i for i, r in enumerate(rows)
                                    if as_bool(r["full_union"]) and not as_bool(r["full_orig"])],
        "rep_full_support_gain": [i for i, r in enumerate(rows)
                                  if as_bool(r["full_rep"]) and not as_bool(r["full_orig"])],
        "arom_full_support_gain": [i for i, r in enumerate(rows)
                                   if as_bool(r["full_arom"]) and not as_bool(r["full_orig"])],
    }
    return out


def relation_stats(idx: np.ndarray, y: np.ndarray,
                   d: dict[str, np.ndarray]) -> dict:
    idx = np.asarray(idx, dtype=np.int64)
    err = np.abs(d["orig"][idx] - y[idx])
    mac = d["max_abs_change"][idx]
    top = idx[np.argsort(-err)[:20]]
    return {
        "n": int(len(idx)),
        "mean_abs_error_original": float(err.mean()),
        "mean_max_abs_change": float(mac.mean()),
        "ratio_sensitivity_to_error": float(mac.mean() / err.mean()) if err.mean() else None,
        "pearson_error_vs_sensitivity": float(np.corrcoef(err, mac)[0, 1])
        if len(idx) > 2 and np.std(err) > 0 and np.std(mac) > 0 else None,
        "spearman_error_vs_sensitivity": float(np.corrcoef(
            np.argsort(np.argsort(err)), np.argsort(np.argsort(mac)))[0, 1])
        if len(idx) > 2 and np.std(mac) > 0 else None,
        "n_sensitivity_gt_error": int((mac > err).sum()),
        "n_sensitivity_gt_half_error": int((mac > 0.5 * err).sum()),
        "top20_error_mean_sensitivity": float(np.mean(d["max_abs_change"][top]))
        if len(top) else None,
        "top20_error_median_sensitivity": float(np.median(d["max_abs_change"][top]))
        if len(top) else None,
    }


def pred_example(i: int, rows: Sequence[dict], y: np.ndarray,
                 m: dict[str, dict[str, np.ndarray]], a: dict[str, np.ndarray]) -> dict:
    r = rows[i]
    out = {
        "row": int(i),
        "old_group": r["old_group"],
        "y": float(y[i]),
        "n_atoms": int(r["n_atoms"]),
        "n_reps": int(r["n_reps"]),
        "status": r["status"],
        "truncated": as_bool(r["truncated"]),
        "support": {
            "unseen_orig": int(r["unseen_orig"]),
            "unseen_rep": int(r["unseen_rep"]),
            "unseen_arom": int(r["unseen_arom"]),
            "unseen_union": int(r["unseen_union"]),
            "full_orig": as_bool(r["full_orig"]),
            "full_rep": as_bool(r["full_rep"]),
            "full_arom": as_bool(r["full_arom"]),
            "full_union": as_bool(r["full_union"]),
            "rescued_union": int(r["rescued_union"]),
        },
        "models": {},
    }
    n = int(r["n_reps"])
    for model in ("p3b", "t1"):
        d = m[model]
        out["models"][model] = {
            "predictions": [float(x) for x in a[model][i, :n]],
            "original": float(d["orig"][i]),
            "mean_all": float(d["mean_all"][i]),
            "mean_alt": float(d["mean_alt"][i]),
            "range": float(d["range"][i]),
            "std": float(d["std"][i]),
            "max_abs_change": float(d["max_abs_change"][i]),
            "mean_abs_change_alt": float(d["mean_abs_change_alt"][i]),
            "abs_error_original": float(abs(d["orig"][i] - y[i])),
            "abs_error_all_rep_mean": float(abs(d["mean_all"][i] - y[i])),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", default=str(ANALYSIS / "preds_valid.npz"))
    p.add_argument("--checks", default=str(ANALYSIS / "infer_checks.json"))
    p.add_argument("--out-json", default=str(ANALYSIS / "pred_sensitivity_summary.json"))
    p.add_argument("--out-csv", default=str(ANALYSIS / "pred_per_molecule.csv"))
    args = p.parse_args(argv)

    rows = read_support()
    y = np.asarray([float(r["y"]) for r in rows])
    group = np.asarray([r["old_group"] for r in rows])
    preds = np.load(args.preds)
    n = np.asarray(preds["n_reps"], dtype=np.int64)
    a = {"p3b": np.asarray(preds["p3b"]), "t1": np.asarray(preds["t1"])}
    m = {k: metrics(v, n) for k, v in a.items()}

    idx_all = np.arange(1000)
    groups = {"all": idx_all}
    for g in GROUP_ORDER:
        groups[g] = np.where(group == g)[0]

    summary: dict[str, Any] = {
        "protocol": "zinc-kekule-equiv-v1",
        "official_test_loaded": False,
        "inputs": {
            "preds": str(Path(args.preds).relative_to(KG.REPO))
            if Path(args.preds).is_relative_to(KG.REPO) else str(args.preds),
            "support": "tracks/structural-sufficiency/zinc_kekule_equiv_v1/analysis/support_per_molecule.csv",
            "checkpoints": "P3B seed0 (xBssl_P3N_s0) and T seed1 (SF_T_s1), fixed",
        },
        "groups": {},
        "subgroups": {},
        "relation": {},
        "examples": {},
        "definitions": {
            "max_abs_change": "max_j |pred_j - pred_orig| over legal representations j, 0 if n_reps=1",
            "mean_abs_change_alt": "mean |pred_j - pred_orig| over alternative representations only",
            "range": "max_j pred_j - min_j pred_j over legal representations",
            "std": "population std over legal representations",
            "mean_all": "mean prediction over all legal representations including the original (fixed-model diagnostic, not a score)",
            "mae_all_rep_mean_fixed_diagnostic": "MAE of the above fixed mean; labels are used only for evaluation, never to select a representation",
        },
    }

    for g, idx in groups.items():
        summary["groups"][g] = {
            "n": int(len(idx)),
            **{model: block(rows, y, m, idx, model) for model in ("p3b", "t1")},
        }

    sg = subgroup_indices(rows)
    # rows rescued by support but with negligible prediction change
    sg["rescued_but_insensitive"] = [
        i for i, r in enumerate(rows)
        if int(r["rescued_union"]) > 0
        and max(m["p3b"]["max_abs_change"][i], m["t1"]["max_abs_change"][i]) < 0.01
    ]
    for name, idx in sg.items():
        summary["subgroups"][name] = {
            "rows": [int(x) for x in idx],
            "n": int(len(idx)),
            **{model: block(rows, y, m, np.asarray(idx), model) for model in ("p3b", "t1")},
        }

    for model in ("p3b", "t1"):
        summary["relation"][model] = {
            "all": relation_stats(idx_all, y, m[model]),
            "unseen_ordinary": relation_stats(groups["unseen_ordinary"], y, m[model]),
        }

    # examples: selected by prediction/support diagnostics, never by label.
    for model in ("p3b", "t1"):
        uidx = groups["unseen_ordinary"]
        order = uidx[np.argsort(-m[model]["max_abs_change"][uidx])][:5]
        sidx = groups["seen_ordinary"]
        sorder = sidx[np.argsort(-m[model]["max_abs_change"][sidx])][:3]
        summary["examples"][model] = {
            "top_unseen_ordinary": [pred_example(int(i), rows, y, m, a) for i in order],
            "top_seen_ordinary": [pred_example(int(i), rows, y, m, a) for i in sorder],
            "union_gain": [pred_example(int(i), rows, y, m, a)
                           for i in sg["union_full_support_gain"][:6]],
            "rescued_but_insensitive": [pred_example(int(i), rows, y, m, a)
                                        for i in sg["rescued_but_insensitive"][:4]],
        }
    # one explicit outlier note, if present
    j = int(np.argmax(np.maximum(m["p3b"]["max_abs_change"], m["t1"]["max_abs_change"])))
    summary["examples"]["largest_change_overall"] = pred_example(j, rows, y, m, a)

    if Path(args.checks).exists():
        checks = json.loads(Path(args.checks).read_text(encoding="utf-8"))
        summary["inference_checks"] = {
            "original_prediction_replay": checks.get("original_prediction_replay"),
            "permutation_check": checks.get("permutation_check"),
            "feature_rebuild": checks.get("feature_rebuild"),
            "batch_size": checks.get("batch_size"),
            "device": checks.get("device"),
            "torch": checks.get("torch"),
            "elapsed_seconds": checks.get("elapsed_seconds"),
        }

    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")

    # per-molecule CSV, including every legal prediction as JSON
    fields = ["row", "old_group", "y", "n_atoms", "n_reps", "status", "truncated",
              "n_aromatic_edges", "n_inferred_h",
              "unseen_orig", "unseen_rep", "unseen_arom", "unseen_union",
              "full_orig", "full_rep", "full_arom", "full_union",
              "rescued_union", "lost_rep", "lost_arom"]
    for model in ("p3b", "t1"):
        fields += [f"{model}_{k}" for k in (
            "orig", "mean_all", "mean_alt", "min", "max", "range", "std",
            "max_abs_change", "mean_abs_change_alt", "err_orig", "err_mean_all",
            "reps_json")]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            out = {k: r[k] for k in fields if k in r}
            out["row"] = int(i)
            out["y"] = float(y[i])
            k = int(n[i])
            for model in ("p3b", "t1"):
                d = m[model]
                vals = a[model][i, :k]
                out[f"{model}_orig"] = float(d["orig"][i])
                out[f"{model}_mean_all"] = float(d["mean_all"][i])
                out[f"{model}_mean_alt"] = float(d["mean_alt"][i])
                out[f"{model}_min"] = float(d["min"][i])
                out[f"{model}_max"] = float(d["max"][i])
                out[f"{model}_range"] = float(d["range"][i])
                out[f"{model}_std"] = float(d["std"][i])
                out[f"{model}_max_abs_change"] = float(d["max_abs_change"][i])
                out[f"{model}_mean_abs_change_alt"] = float(d["mean_abs_change_alt"][i])
                out[f"{model}_err_orig"] = float(abs(d["orig"][i] - y[i]))
                out[f"{model}_err_mean_all"] = float(abs(d["mean_all"][i] - y[i]))
                out[f"{model}_reps_json"] = json.dumps(
                    [round(float(x), 8) for x in vals], separators=(",", ":"))
            w.writerow(out)

    write_tables(summary)
    print(json.dumps({
        "groups": {g: {model: {
            "n_with_alternative": summary["groups"][g][model]["n_with_alternative"],
            "mean_max_abs_change": summary["groups"][g][model]["mean_max_abs_change"],
            "median_max_abs_change": summary["groups"][g][model]["median_max_abs_change"],
            "mae_original": summary["groups"][g][model]["mae_original"],
            "delta_mae_mean": summary["groups"][g][model]["delta_mae_all_rep_mean_minus_original"],
        } for model in ("p3b", "t1")} for g in ("all",) + GROUP_ORDER},
        "union_gain": {model: {
            "n": summary["subgroups"]["union_full_support_gain"][model]["n_molecules"],
            "mean_max_abs_change": summary["subgroups"]["union_full_support_gain"][model]["mean_max_abs_change"],
            "delta_mae_mean": summary["subgroups"]["union_full_support_gain"][model]["delta_mae_all_rep_mean_minus_original"],
        } for model in ("p3b", "t1")},
    }, indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    return 0


def write_tables(summary: dict) -> None:
    lines = ["# Frozen-model sensitivity tables", ""]
    lines += ["Per-molecule changes are in `pred_per_molecule.csv`; every legal prediction "
              "is included as JSON in the `*_reps_json` columns.", ""]
    lines += ["## Group summary", "",
              "| group | n | n alt | model | mean max |Δ| | median max |Δ| | mean range | MAE orig | MAE all-rep mean | ΔMAE |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in ("all",) + GROUP_ORDER:
        b = summary["groups"][g]
        for model in ("p3b", "t1"):
            d = b[model]
            lines.append(
                f"| {g} | {d['n_molecules']} | {d['n_with_alternative']} | {model} | "
                f"{d['mean_max_abs_change']:.6f} | {d['median_max_abs_change']:.6f} | "
                f"{d['mean_pred_range']:.6f} | {d['mae_original']:.6f} | "
                f"{d['mae_all_rep_mean_fixed_diagnostic']:.6f} | "
                f"{d['delta_mae_all_rep_mean_minus_original']:+.6f} |")
    lines += ["", "## Support-gain subgroups", "",
              "| subgroup | n | model | mean max |Δ| | median max |Δ| | MAE orig | ΔMAE (all-rep mean) |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for name, b in summary["subgroups"].items():
        if not b["n"]:
            continue
        for model in ("p3b", "t1"):
            d = b[model]
            lines.append(
                f"| {name} | {d['n_molecules']} | {model} | "
                f"{d['mean_max_abs_change']:.6f} | {d['median_max_abs_change']:.6f} | "
                f"{d['mae_original']:.6f} | "
                f"{d['delta_mae_all_rep_mean_minus_original']:+.6f} |")
    lines += ["", "## Error / sensitivity relation", "",
              "| model | subset | n | mean |error| | mean max |Δ| | ratio | Pearson | n |Δ|>|err| |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("p3b", "t1"):
        for subset in ("all", "unseen_ordinary"):
            r = summary["relation"][model][subset]
            lines.append(
                f"| {model} | {subset} | {r['n']} | {r['mean_abs_error_original']:.6f} | "
                f"{r['mean_max_abs_change']:.6f} | {r['ratio_sensitivity_to_error']:.3f} | "
                f"{r['pearson_error_vs_sensitivity']:.3f} | {r['n_sensitivity_gt_error']} |")
    lines.append("")
    (ANALYSIS / "pred_tables.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
