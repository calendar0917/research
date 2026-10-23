"""Aggregate valid-split evidence for zinc-performance-search-v1.

Reads only run artifacts that already exist on disk.  Every selector works on
the official valid split; the official test split is never touched here.

A model record is (arm, recipe, policy, seed):
  A/B  -> this round's ``zps_runner`` runs, plus the two declared reused
          seed-0 anchors (A-R0 = ``xBssl_P3N_s0``, B-R0 = ``L_s0``)
  C    -> this round's dense-GRIT runs.
Policy ``online`` is the best-valid checkpoint of the online weights; policy
``ema`` is the best-valid checkpoint of the single-trajectory EMA state.  They
are separate checkpoints and are never averaged at prediction time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
ROOT = _HERE.parents[2]
PERMOL = _SS / "zinc_generalization_audit_v1" / "analysis" / "per_molecule.csv"

# frozen seed-0 anchors reused from their owning rounds (identical R0 recipe)
EXTERNAL = {
    ("A", "R0", "online", 0): {
        "run": _SS / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0",
        "source": "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0",
        "source_round": "zinc-recipe-trunk-v2",
    },
    ("B", "R0", "online", 0): {
        "run": _SS / "zinc_dual_channel_composer_v1" / "runs" / "L_s0",
        "source": "zinc_dual_channel_composer_v1/runs/L_s0",
        "source_round": "zinc-dual-channel-composer-v1",
    },
}

GROUPS = {
    "all_1000": lambda r: np.ones(len(r["y_raw"]), dtype=bool),
    "seen_749": lambda r: r["r2_unseen_atom_frac"] <= 0,
    "unseen_251": lambda r: r["r2_unseen_atom_frac"] > 0,
    "ordinary_935": lambda r: r["y_raw"] > -3.0,
    "tail_65": lambda r: r["y_raw"] <= -3.0,
    "unseen_ordinary_225": lambda r: (r["r2_unseen_atom_frac"] > 0) & (r["y_raw"] > -3.0),
    "seen_ordinary_710": lambda r: (r["r2_unseen_atom_frac"] <= 0) & (r["y_raw"] > -3.0),
    "unseen_tail_26": lambda r: (r["r2_unseen_atom_frac"] > 0) & (r["y_raw"] <= -3.0),
    "rings_ge4_213": lambda r: r["n_rings"] >= 4,
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_name(name: str):
    """``A_R1_s0`` / ``A_R1_ema_s0`` / ``gritd_C0_s0``."""
    parts = name.split("_")
    if parts[0] == "gritd":
        parts = parts[1:]
    if parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    ema = parts[-1] == "ema"
    if ema:
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    return parts[0], parts[1], ("ema" if ema else "online"), seed


def collect() -> list[dict]:
    recs = []
    for key, meta in EXTERNAL.items():
        arm, recipe, policy, seed = key
        train_json = meta["run"] / "train.json"
        if not train_json.exists():
            continue
        recs.append(_record(arm, recipe, policy, seed, meta["run"], meta["source"],
                            meta.get("source_round"), reused=True))
    for tj in sorted(RUNS.rglob("train.json")) if RUNS.exists() else []:
        d = tj.parent
        info = json.loads(tj.read_text())
        if str(info.get("variant", "")).startswith("DBT"):
            arm, recipe, policy, seed = "C", "DBT600", "online", int(info.get("seed", 0))
        else:
            parsed = parse_name(d.name)
            if parsed is None:
                continue
            arm, recipe, policy, seed = parsed
            if policy == "ema":
                # this round's EMA branch is invalid (copy-direction bug); the
                # run itself is a legitimate online run and is kept as such
                policy = "online"
        recs.append(_record(arm, recipe, policy, seed, d,
                            "runs/" + str(d.relative_to(RUNS)), None, reused=False))
    return recs


def _curve_metrics(curve, key="valid_mae"):
    vals = [float(e[key]) for e in curve if e.get(key) is not None]
    if not vals:
        return None, None
    w20 = None
    if len(vals) >= 20:
        windows = [float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19)]
        w20 = float(min(windows))
    return w20, float(vals[-1])


def _record(arm, recipe, policy, seed, run: Path, source, source_round, reused) -> dict:
    info = json.loads((run / "train.json").read_text())
    curve = info["curve"]
    w20, final = _curve_metrics(curve, "valid_mae")
    if policy == "ema":
        best = info.get("best_valid_ema")
        w20, final = _curve_metrics(curve, "ema_valid_mae")
    else:
        best = info["best_valid"]
    pred_name = "valid_best.npz" if policy == "online" else "valid_best_ema.npz"
    pred_path = run / "predictions" / pred_name
    ckpt = run / ("best.pt" if policy == "online" else "ema_best.pt")
    return {
        "arm": arm,
        "recipe": recipe,
        "policy": policy,
        "seed": int(seed),
        "run": str(run.relative_to(ROOT)),
        "source": source,
        "source_round": source_round,
        "reused": bool(reused),
        "params": int(info.get("parameter_count")
                      or info.get("declared_total_parameters")
                      or (info.get("model_info") or {}).get("parameters")),
        "epochs": int(info["recipe"].get("epochs", len(curve))),
        "updates": int(info["recipe"].get("updates", len(curve) * info["recipe"].get("steps_per_epoch", 0))),
        "best_valid": None if best is None else float(best["valid_mae"]),
        "best_epoch": None if best is None else int(best["epoch"]),
        "w20": w20,
        "final_valid": final,
        "train_mae_at_best": info["fixed_objects"]["best"]["train_mae_eval_mode"],
        "train_mae_at_final": info["fixed_objects"]["final"]["train_mae_eval_mode"],
        "gpu_hours": info.get("gpu_hours"),
        "ema_note": ("invalid" if info.get("best_valid_ema")
                     and float(info["best_valid_ema"]["valid_mae"]) > 0.5 else None),
        "init": info.get("model_info", {}).get("initial_state_sha256"),
        "ssl": (info.get("model_info", {}) or {}).get("ssl"),
        "loss": info["recipe"].get("loss"),
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.exists() else None,
        "checkpoint_sha256": sha256(ckpt),
        "predictions": str(pred_path.relative_to(ROOT)) if pred_path.exists() else None,
        "git_commit": info.get("git_commit"),
    }


def permol_labels():
    rows = list(csv.DictReader(open(PERMOL)))
    assert len(rows) == 1000
    order = [int(r["row_id"]) for r in rows]
    assert order == list(range(1000)), "per_molecule.csv must be in official valid order"
    return {
        "y_raw": np.asarray([float(r["y_raw"]) for r in rows]),
        "r2_unseen_atom_frac": np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows]),
        "n_rings": np.asarray([int(r["n_rings"]) for r in rows]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args(argv)
    recs = collect()
    lab = permol_labels()
    y = lab["y_raw"]
    rng = np.random.default_rng(20261023)
    boot = rng.integers(0, len(y), size=(args.bootstrap, len(y)))

    preds, aerr = {}, {}
    missing = []
    for r in recs:
        key = f"{r['arm']}-{r['recipe']}-{r['policy']}-s{r['seed']}"
        r["name"] = key
        if not r["predictions"]:
            missing.append(key)
            continue
        npz = np.load(ROOT / r["predictions"])
        pred = npz["pred"].astype(np.float64)
        if pred.shape != (1000,):
            raise RuntimeError(f"{key}: predictions shape {pred.shape}")
        preds[key] = pred
        r["valid_mae_recomputed"] = float(np.abs(pred - y).mean())
        r["valid_mae_reported"] = r["best_valid"]
        r["mae_recompute_absdiff"] = abs(r["valid_mae_recomputed"] - r["best_valid"])
        aerr[key] = np.abs(pred - y)

    rows = []
    for r in recs:
        key = r["name"]
        row = dict(r)
        if key in aerr:
            ae = aerr[key]
            # sensitivity: drop row 172 (the single extreme-label molecule used
            # in earlier rounds) and drop this model's own top-10 AE rows
            keep172 = np.ones(len(y), dtype=bool)
            keep172[172] = False
            top10 = np.argsort(-ae)[:10]
            keep10 = np.ones(len(y), dtype=bool)
            keep10[top10] = False
            row["mae_drop_row172"] = float(ae[keep172].mean())
            row["mae_drop_top10"] = float(ae[keep10].mean())
            row["groups"] = {}
            for gname, fn in GROUPS.items():
                m = fn(lab)
                row["groups"][gname] = {"n": int(m.sum()),
                                       "mae": float(ae[m].mean())}
        rows.append(row)

    # paired bootstrap contrasts (descriptive; exposed valid, fixed checkpoints)
    keys = sorted(aerr)
    contrasts = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = aerr[a] - aerr[b]
            mean_d = float(d.mean())
            bs = d[boot].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            contrasts[f"{a} minus {b}"] = {
                "mean_delta_mae": mean_d,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "positive_share": float((d > 0).mean()),
            }

    # signed-error correlations and pairwise wins
    signed = {k: preds[k] - y for k in keys}
    corr = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            corr[f"{a} vs {b}"] = float(np.corrcoef(signed[a], signed[b])[0, 1])
    wins = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            da, db = aerr[a], aerr[b]
            wins[f"{a} vs {b}"] = {"a_better": int((da < db).sum()),
                                   "b_better": int((db < da).sum()),
                                   "ties": int((da == db).sum())}

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "main_table.json").write_text(json.dumps(
        {"records": rows, "missing_predictions": missing,
         "groups_definition": {
             "source": "tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/per_molecule.csv",
             "unseen": "r2_unseen_atom_frac > 0 (251)", "ordinary": "y > -3 (935)",
             "unseen_ordinary": "225", "tail": "y <= -3 (65)"},
         "w20_definition": "min over contiguous 20-epoch windows of the window median best-valid MAE",
         "bootstrap": {"n": args.bootstrap, "seed": 20261023,
                       "note": "row bootstrap on exposed valid; describes sample uncertainty, not training noise"}},
        indent=1))
    (ANALYSIS / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    (ANALYSIS / "error_relations.json").write_text(json.dumps(
        {"signed_error_correlation": corr, "pairwise_ae_wins": wins}, indent=1))

    # per-molecule table for every collected model
    fields = ["row_id", "y_raw", "r2_unseen_atom_frac", "n_rings"]
    for k in keys:
        fields += [f"pred_{k}", f"ae_{k}"]
    with open(ANALYSIS / "per_molecule_valid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i in range(len(y)):
            line = [i, f"{y[i]:.9f}", f"{lab['r2_unseen_atom_frac'][i]:.9f}",
                    int(lab["n_rings"][i])]
            for k in keys:
                line += [f"{preds[k][i]:.9f}", f"{aerr[k][i]:.9f}"]
            w.writerow(line)

    # markdown summary
    lines = ["# valid table (official valid-1000, raw-y MAE)", "",
             "| model | policy | seed | params | epochs | best-valid | ep | W20 | final | train@best | d(row172) | d(top10) | gpu-h | source |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["arm"], r["recipe"], r["policy"], r["seed"])):
        def f(x, fmt="{:.6f}"):
            return "-" if x is None else fmt.format(x)
        lines.append(
            f"| {r['arm']}-{r['recipe']} | {r['policy']} | {r['seed']} | {r['params']} | "
            f"{r['epochs']}/{r['updates']} | {f(r['best_valid'])} | {r['best_epoch']} | "
            f"{f(r['w20'])} | {f(r['final_valid'])} | {f(r['train_mae_at_best'])} | "
            f"{f(r.get('mae_drop_row172'))} | {f(r.get('mae_drop_top10'))} | "
            f"{f(r['gpu_hours'], '{:.3f}')} | {r['source']} |")
    # per-config seed summary (this is the discovery ranking unit)
    fam = {}
    for r in rows:
        if r["policy"] != "online":
            continue
        key = f"{r['arm']}-{r['recipe']}"
        fam.setdefault(key, []).append(r)
    lines += ["", "## Config summary over seeds (online branch)", "",
              "| config | seeds | best-valid mean | sd | range | W20 mean | mean ep | mean gpu-h |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for key, rs in sorted(fam.items()):
        vals = [r["best_valid"] for r in rs]
        w20s = [r["w20"] for r in rs if r["w20"] is not None]
        eps = [r["best_epoch"] for r in rs]
        gpus = [r["gpu_hours"] for r in rs if r["gpu_hours"]]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        summary[key] = {"n": len(vals), "mean": float(np.mean(vals)), "sd": sd,
                        "range": [float(min(vals)), float(max(vals))],
                        "seeds": [r["seed"] for r in rs],
                        "w20_mean": float(np.mean(w20s)) if w20s else None,
                        "mean_epoch": float(np.mean(eps)),
                        "gpu_hours": float(np.sum(gpus)) if gpus else None}
        lines.append(
            f"| {key} | {len(vals)} ({','.join(str(r['seed']) for r in rs)}) | "
            f"{np.mean(vals):.6f} | {'-' if sd is None else f'{sd:.6f}'} | "
            f"[{min(vals):.6f}, {max(vals):.6f}] | "
            f"{'-' if not w20s else f'{np.mean(w20s):.6f}'} | {np.mean(eps):.0f} | "
            f"{'-' if not gpus else f'{np.sum(gpus):.3f}'} |")
    (ANALYSIS / "config_summary.json").write_text(json.dumps(summary, indent=1))
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if missing:
        print("\nMISSING predictions:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
