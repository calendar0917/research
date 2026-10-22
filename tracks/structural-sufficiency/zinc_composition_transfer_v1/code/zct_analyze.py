"""Analysis for the composition-transfer round.

Reads only frozen artifacts written by the formal runs (and, for the terminal
test, the separately frozen test predictions).  Produces ``analysis/summary.json``,
``analysis/main_table.md``, ``analysis/groups.md`` and ``analysis/support.md``.

Definitions fixed before the runs:

* ``best_valid`` / ``best_epoch`` / ``final_valid`` from ``train.json``;
* ``eval_train_at_best`` from the fixed-object measurement in ``train.json``;
* ``W20`` = minimum over all 20-consecutive-epoch windows of that window's
  **median** valid MAE;
* ``T100`` = median valid MAE over the last 100 epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

RUNS = _TASK / "runs"
ANALYSIS = _TASK / "analysis"
CACHE = _TASK / "cache"
OUT = CACHE / "artifacts"
WINDOW = 20


def load_run(name: str) -> dict:
    d = RUNS / name
    info = C.read_json(d / "train.json")
    pred = np.load(d / "predictions" / "valid_best.npz")
    info["_pred"] = np.asarray(pred["pred"], dtype=np.float64)
    info["_row"] = np.asarray(pred["row"], dtype=np.int64)
    info["_dir"] = str(d)
    return info


def curve_metrics(curve: list[dict]) -> dict:
    valid = np.asarray([e["valid_mae"] for e in curve], dtype=np.float64)
    epochs = [int(e["epoch"]) for e in curve]
    n = len(valid)
    w20, w20_start = float("inf"), None
    if n >= WINDOW:
        for s in range(0, n - WINDOW + 1):
            med = float(np.median(valid[s:s + WINDOW]))
            if med < w20:
                w20, w20_start = med, epochs[s]
    t100 = float(np.median(valid[-100:])) if n >= 100 else float(np.median(valid))
    return {
        "W20": w20 if math.isfinite(w20) else float(np.median(valid)),
        "W20_window_start_epoch": w20_start,
        "T100": t100,
        "valid_min": float(valid.min()),
        "valid_last": float(valid[-1]),
        "n_epochs": int(n),
        "valid_curve": [float(x) for x in valid],
        "epochs": epochs,
    }


def old_groups() -> dict:
    path = _SSUF / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    unseen = np.asarray([int(r["signature_unseen"]) for r in rows], dtype=np.int64)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows], dtype=np.int64)
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    frac = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64)
    names = {(0, 1): "seen_ordinary", (0, 0): "seen_tail", (1, 1): "unseen_ordinary",
             (1, 0): "unseen_tail"}
    groups = {names[(int(u), int(o))]: np.flatnonzero((unseen == u) & (ordinary == o))
              for (u, o) in names}
    return {"y": y, "unseen": unseen, "ordinary": ordinary, "frac": frac, "groups": groups,
            "source": str(path)}


def export_preds(name: str, which: str = "final") -> np.ndarray | None:
    path = OUT / f"export_{name}_{which}.npz"
    if not path.exists():
        return None
    return np.asarray(np.load(path)["pred"], dtype=np.float64)


def group_table(pred: np.ndarray, g: dict) -> dict:
    y = g["y"]
    ae = np.abs(pred - y)
    out = {}
    total = float(ae.sum())
    for name, idx in g["groups"].items():
        out[name] = {"n": int(len(idx)), "mae": float(ae[idx].mean()),
                     "sum_ae": float(ae[idx].sum()),
                     "share_of_total_ae": float(ae[idx].sum() / max(total, 1e-12))}
    order = np.argsort(-ae)
    top10 = order[:10]
    out["all"] = {"n": int(len(ae)), "mae": float(ae.mean()), "sum_ae": total}
    out["drop_row172"] = {"n": int(len(ae) - 1),
                          "mae": float(np.delete(ae, 172).mean())}
    out["median_ae"] = {"all": float(np.median(ae)),
                        **{name: float(np.median(ae[idx])) for name, idx in g["groups"].items()}}
    top = order[: max(1, len(ae) // 100)]
    keep = np.setdiff1d(np.arange(len(ae)), top)
    out["trimmed_top1pct"] = {"n": int(len(keep)), "mae": float(ae[keep].mean())}
    out["top10_molecules"] = [{"row": int(i), "y": float(y[i]), "ae": float(ae[i]),
                               "group": _group_name(i, g)} for i in top10]
    out["top1pct_share_of_ae"] = float(ae[order[: max(1, len(ae) // 100)]].sum() / max(total, 1e-12))
    out["top5pct_share_of_ae"] = float(ae[order[: max(1, len(ae) // 20)]].sum() / max(total, 1e-12))
    out["single_molecule_max_share"] = float(ae.max() / max(total, 1e-12))
    out["single_molecule_max_row"] = int(order[0])
    return out


def _group_name(row: int, g: dict) -> str:
    for name, idx in g["groups"].items():
        if row in set(idx.tolist()):
            return name
    return "unknown"


def _sign_test_p(k: int, n: int) -> float:
    """Exact two-sided binomial sign test on k successes out of n (p=0.5)."""
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / float(2 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, reps: int = 5000,
                     seed: int = 12345) -> dict:
    rng = np.random.default_rng(seed)
    d = ae_a - ae_b
    n = len(d)
    idx = rng.integers(0, n, size=(reps, n))
    means = d[idx].mean(1)
    better = int((d < 0).sum())
    d172 = np.delete(d, 172) if n > 172 else d
    return {"delta_mean": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "delta_median": float(np.median(d)),
            "frac_molecules_a_worse": float((d > 0).mean()),
            "frac_molecules_a_better": float((d < 0).mean()),
            "sign_test_p": _sign_test_p(better, n),
            "delta_mean_excluding_row172": float(d172.mean()),
            "note": "paired over the 1000 valid molecules; descriptive only"}


def verify_contract(runs: dict) -> dict:
    checks = {}
    for seed in (0, 1):
        arms = [runs[f"{a}_s{seed}"] for a in ("E", "R", "S") if f"{a}_s{seed}" in runs]
        if len(arms) < 2:
            continue
        init = {a["arm"]: a["init_state_sha256"] for a in arms}
        recipes = {a["arm"]: json.dumps(a["recipe"], sort_keys=True, default=str) for a in arms}
        meta_kappas = {a["arm"]: a["kappa"] for a in arms if a["arm"] in ("R", "S")}
        arms_len = {a["arm"]: len(a["curve"]) for a in arms}
        hashes = {}
        n = min(arms_len.values())
        sofar = None
        same = True
        for i in range(n):
            h = tuple(a["curve"][i]["batch_union_sha256"] for a in arms)
            if len(set(h)) != 1:
                same = False
                break
        for a in arms:
            hashes[a["arm"]] = a["curve"][-1]["batch_union_sha256"]
        checks[f"seed{seed}"] = {
            "init_hashes_identical": len(set(init.values())) == 1,
            "recipe_identical": len(set(recipes.values())) == 1,
            "kappa_meta_arms": meta_kappas,
            "kappa_identical": len(set(meta_kappas.values())) <= 1,
            "curve_lengths": arms_len,
            "per_epoch_batch_union_hashes_identical_over_common_length": bool(same),
            "n_common_epochs": int(n),
            "final_union_hash": hashes,
        }
    return checks


def _final_epoch_contrast(arm: str, seed: int, ref: str, g: dict) -> dict:
    """Unselected epoch-600 per-molecule contrast against the reference arm."""
    a = export_preds(f"{arm}_s{seed}", "final")
    b = export_preds(f"{ref}_s{seed}", "final")
    if a is None or b is None:
        return {}
    ae_a = np.abs(a - g["y"])
    ae_b = np.abs(b - g["y"])
    return {"final_epoch_molecule": paired_bootstrap(ae_a, ae_b)}


def budget_ledger(runs: dict, extra: dict | None = None) -> dict:
    rows = []
    for name, r in runs.items():
        rows.append({"run": name, "arm": r["arm"], "seed": r["seed"],
                     "gpu_hours_runtime": r.get("gpu_hours"),
                     "wall_seconds": r.get("wall_seconds"),
                     "device": r.get("device"),
                     "epochs": r["recipe"]["epochs"]})
    total = float(sum(x["gpu_hours_runtime"] or 0.0 for x in rows))
    out = {"runs": rows, "gpu_hours_sum_runtime": total}
    if extra:
        out.update(extra)
    out["cap_hours"] = 8.0
    return out


def core(extra_ledger: dict | None = None) -> dict:
    names = [f"{a}_s{s}" for s in (0, 1) for a in ("E", "R", "S")]
    runs = {n: load_run(n) for n in names if (RUNS / n / "train.json").exists()}
    g = old_groups()
    summary = {"round": "zinc-composition-transfer-v1", "runs": {}, "comparisons": {}}
    for name, r in runs.items():
        cm = curve_metrics(r["curve"])
        gt = group_table(r["_pred"], g)
        summary["runs"][name] = {
            "arm": r["arm"], "seed": r["seed"], "kappa": r["kappa"],
            "parameters": r["parameter_count"],
            "best_valid": r["best_valid"]["valid_mae"], "best_epoch": r["best_valid"]["epoch"],
            "final_valid": r["fixed_objects"]["final"]["valid_mae"],
            "eval_train_at_best": r["fixed_objects"]["best"]["train_mae_eval_mode"],
            "eval_train_final": r["fixed_objects"]["final"]["train_mae_eval_mode"],
            "W20": cm["W20"], "W20_window_start_epoch": cm["W20_window_start_epoch"],
            "T100": cm["T100"],
            "valid_last": cm["valid_last"],
            "init_state_sha256": r["init_state_sha256"],
            "ssl_sha256": r["model_info"]["sha256"],
            "curve": {"epochs": cm["epochs"], "valid": cm["valid_curve"]},
            "groups_valid": gt,
            "probe_epochs": r.get("probe_epochs") or [],
            "meta_objective": r.get("meta_objective"),
            "gpu_hours": r.get("gpu_hours"),
            "wall_seconds": r.get("wall_seconds"),
            "provenance": {"git_commit": r.get("git_commit"),
                           "host": (r.get("environment") or {}).get("gpu_names"),
                           "torch": (r.get("environment") or {}).get("torch")},
        }
        # last-100-epoch eval-train median (used by the pre-registered repair rule)
        tr = [e.get("train_mae_eval_mode") for e in r["curve"]
              if e.get("train_mae_eval_mode") is not None]
        summary["runs"][name]["eval_train_last100_median"] = (
            float(np.median(tr[-100:])) if tr else None)
        for which in ("best", "final"):
            pr = export_preds(name, which)
            if pr is None:
                continue
            ae = np.abs(pr - g["y"])
            summary["runs"][name].setdefault("molecules", {})[which] = {
                "mae": float(ae.mean()), "median_ae": float(np.median(ae)),
                "ae_row172": float(ae[172]),
                "mae_excluding_row172": float(np.delete(ae, 172).mean()),
                "trimmed_top1pct_mae": float(
                    np.sort(ae)[: len(ae) - max(1, len(ae) // 100)].mean()),
                "groups": {k: float(ae[idx].mean()) for k, idx in g["groups"].items()},
            }

    for seed in (0, 1):
        base = runs.get(f"E_s{seed}")
        if base is None:
            continue
        for arm in ("R", "S"):
            r = runs.get(f"{arm}_s{seed}")
            if r is None:
                continue
            cm_b = curve_metrics(base["curve"])
            cm_r = curve_metrics(r["curve"])
            ae_b = np.abs(base["_pred"] - g["y"])
            ae_r = np.abs(r["_pred"] - g["y"])
            c = {
                "best_valid_delta": r["best_valid"]["valid_mae"] - base["best_valid"]["valid_mae"],
                "W20_delta": cm_r["W20"] - cm_b["W20"],
                "T100_delta": cm_r["T100"] - cm_b["T100"],
                "final_valid_delta": r["fixed_objects"]["final"]["valid_mae"]
                - base["fixed_objects"]["final"]["valid_mae"],
                "molecule": paired_bootstrap(ae_r, ae_b),
                "groups": {
                    name: float(ae_r[idx].mean() - ae_b[idx].mean())
                    for name, idx in g["groups"].items()
                },
            }
            c.update(_final_epoch_contrast(arm, seed, "E", g))
            summary["comparisons"][f"{arm}_minus_E_s{seed}"] = c
        r, s = runs.get(f"R_s{seed}"), runs.get(f"S_s{seed}")
        if r is not None and s is not None:
            cm_r, cm_s = curve_metrics(r["curve"]), curve_metrics(s["curve"])
            ae_r = np.abs(r["_pred"] - g["y"])
            ae_s = np.abs(s["_pred"] - g["y"])
            c = {
                "best_valid_delta": s["best_valid"]["valid_mae"] - r["best_valid"]["valid_mae"],
                "W20_delta": cm_s["W20"] - cm_r["W20"],
                "T100_delta": cm_s["T100"] - cm_r["T100"],
                "final_valid_delta": s["fixed_objects"]["final"]["valid_mae"]
                - r["fixed_objects"]["final"]["valid_mae"],
                "molecule": paired_bootstrap(ae_s, ae_r),
                "groups": {name: float(ae_s[idx].mean() - ae_r[idx].mean())
                           for name, idx in g["groups"].items()},
            }
            c.update(_final_epoch_contrast("S", seed, "R", g))
            summary["comparisons"][f"S_minus_R_s{seed}"] = c
    # seed means
    def seed_mean(arm, key):
        vals = [summary["runs"][f"{arm}_s{s}"][key] for s in (0, 1)
                if f"{arm}_s{s}" in summary["runs"]]
        return float(np.mean(vals)) if vals else None

    summary["seed_means"] = {
        arm: {k: seed_mean(arm, k) for k in ("best_valid", "W20", "T100", "final_valid")}
        for arm in ("E", "R", "S")
    }
    summary["verification"] = verify_contract(runs)
    summary["budget"] = budget_ledger(runs, extra_ledger)
    summary["definitions"] = {
        "W20": "min over 20-consecutive-epoch windows of the window median valid MAE",
        "T100": "median valid MAE over the last 100 epochs",
        "groups": "frozen old four-group valid split (710/225/39/26), reproduced from "
                  "zinc_restricted_adaptation_v1/analysis/reference_preds_valid.csv",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT / "summary.json", summary)
    (OUT / "main_table.md").write_text(_main_table_md(summary), encoding="utf-8")
    (OUT / "groups.md").write_text(_groups_md(summary), encoding="utf-8")
    return summary


def _fmt(x, nd=6):
    return "—" if x is None else f"{x:.{nd}f}"


def _main_table_md(s: dict) -> str:
    rows = ["# Main table (official ZINC subset, raw-y L1, official valid)",
            "",
            "| run | params | best valid | epoch | final valid | eval-train@best | W20 | W20 start | T100 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(s["runs"]):
        r = s["runs"][name]
        rows.append(f"| {name} | {r['parameters']} | {_fmt(r['best_valid'])} | {r['best_epoch']} | "
                    f"{_fmt(r['final_valid'])} | {_fmt(r['eval_train_at_best'])} | {_fmt(r['W20'])} | "
                    f"{r['W20_window_start_epoch']} | {_fmt(r['T100'])} |")
    rows += ["", "## Seed means", "",
             "| arm | best valid | W20 | T100 | final valid |", "|---|---:|---:|---:|---:|"]
    for arm in ("E", "R", "S"):
        sm = s["seed_means"].get(arm) or {}
        rows.append(f"| {arm} | {_fmt(sm.get('best_valid'))} | {_fmt(sm.get('W20'))} | "
                    f"{_fmt(sm.get('T100'))} | {_fmt(sm.get('final_valid'))} |")
    rows += ["", "## Paired contrasts (arm − reference; positive = arm worse)", "",
             "| contrast | best valid | W20 | T100 | final valid | molecule ΔAE [95% CI] |",
             "|---|---:|---:|---:|---:|---|"]
    for k, c in s["comparisons"].items():
        rows.append(f"| {k} | {c['best_valid_delta']:+.6f} | {c['W20_delta']:+.6f} | "
                    f"{c['T100_delta']:+.6f} | {c.get('final_valid_delta', float('nan')):+.6f} | "
                    f"{c['molecule']['delta_mean']:+.5f} "
                    f"[{c['molecule']['ci95'][0]:+.5f}, {c['molecule']['ci95'][1]:+.5f}] |")
    rows += ["", "## Contract checks", ""]
    for seed, v in s["verification"].items():
        rows.append(f"* **{seed}**: init identical = {v['init_hashes_identical']}, recipe identical = "
                    f"{v['recipe_identical']}, kappa identical across meta arms = {v['kappa_identical']} "
                    f"({v['kappa_meta_arms']}), per-epoch batch-union hashes identical over "
                    f"{v['n_common_epochs']} epochs = "
                    f"{v['per_epoch_batch_union_hashes_identical_over_common_length']}")
    rows += ["", "## Budget", "", f"* GPU hours (trainer-measured runtime): "
             f"{s['budget']['gpu_hours_sum_runtime']:.4f} of {s['budget']['cap_hours']} cap"]
    for r in s["budget"]["runs"]:
        rows.append(f"  * {r['run']}: {r['gpu_hours_runtime']:.4f} h, {r['epochs']} epochs, "
                    f"{r['device']}")
    return "\n".join(rows) + "\n"


def _groups_md(s: dict) -> str:
    rows = ["# Fixed four-group valid errors (710 seen+ordinary / 225 unseen+ordinary / "
            "39 seen+tail / 26 unseen+tail)", "",
            "| run | all | seen_ordinary | unseen_ordinary | seen_tail | unseen_tail | "
            "drop row172 | max single-molecule share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(s["runs"]):
        g = s["runs"][name]["groups_valid"]
        rows.append(
            f"| {name} | {g['all']['mae']:.5f} | {g['seen_ordinary']['mae']:.5f} | "
            f"{g['unseen_ordinary']['mae']:.5f} | {g['seen_tail']['mae']:.5f} | "
            f"{g['unseen_tail']['mae']:.5f} | {g['drop_row172']['mae']:.5f} | "
            f"{g['single_molecule_max_share'] * 100:.2f}% (row {g['single_molecule_max_row']}) |")
    rows += ["", "The last column is the largest single-molecule share of total absolute error.",
             "",
             "Top-10 molecule contributions per run (row, group, y, |AE|):", ""]
    for name in sorted(s["runs"]):
        g = s["runs"][name]["groups_valid"]
        top = ", ".join(f"{m['row']}({m['group']}, y={m['y']:.2f}, {m['ae']:.3f})"
                        for m in g["top10_molecules"][:5])
        rows.append(f"* **{name}**: {top}")
    rows += ["", "Row-172 sensitivity is kept in every table; no number is reported without it."]
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# support intervention
# ---------------------------------------------------------------------------


def support() -> dict:
    manifest = C.read_json(CACHE / "support_manifest.json")
    sets = dict(np.load(CACHE / "support_sets.npz"))
    train_cache_labels = None
    import ad_data as D

    y = np.asarray(D.load_cache("train")["labels"], dtype=np.float64)
    runs = {}
    for fit in ("plus", "minus"):
        for seed in (0, 1):
            d = RUNS / f"sup_{fit}_s{seed}"
            if not (d / "train.json").exists():
                continue
            info = C.read_json(d / "train.json")
            runs[f"{fit}_s{seed}"] = info
    frozen = C.read_json(OUT / "support_predictions.json") if \
        (OUT / "support_predictions.json").exists() else None
    q_t = sets["q_target"]
    q_b = sets["q_background"]
    out = {"manifest_sizes": manifest["sizes"], "n_target_keys": manifest["n_target_keys"],
           "runs": {}, "comparisons": {}}
    keys = manifest["target_keys"]
    key_support = np.asarray([manifest["key_support_train"][str(k)] for k in keys])
    parent_support = np.asarray([manifest["key_parent_support_train"][str(k)] for k in keys])
    # per-key query molecules
    kmer = {}
    koff = np.asarray(np.load(CACHE / "keys.npz")["mol_k2_off"])
    kmol = np.asarray(np.load(CACHE / "keys.npz")["mol_k2"])
    inv = {}
    for m in range(len(koff) - 1):
        for k in kmol[int(koff[m]):int(koff[m + 1])].tolist():
            inv.setdefault(int(k), []).append(m)
    for k in keys:
        kmer[str(int(k))] = sorted(set(inv.get(int(k), [])) & set(q_t.tolist()))

    if frozen is None:
        return {"status": "no frozen predictions yet"}

    rows = []
    for k, rec in frozen.get("mae_by_run", {}).items():
        for which, vals in rec.items():
            rows.append({"run": k, "checkpoint": which, **vals})
    out["mae_by_run"] = rows
    for name, info in runs.items():
        out["runs"][name] = {
            "best_monitor": info["best_monitor"], "rows": info["rows"],
            "steps_per_epoch": info["steps_per_epoch"], "updates": info["updates"],
            "gpu_hours": info.get("gpu_hours"), "wall_seconds": info.get("wall_seconds"),
            "init_state_sha256": info["init_state_sha256"],
            "final_monitor": info["curve"][-1]["monitor_mae"],
        }
    # load per-molecule predictions written by zct_support.freeze_predictions
    def preds(fit, seed, which):
        p = OUT / f"support_predictions_{fit}_s{seed}_{which}.npz"
        if not p.exists():
            return None
        return dict(np.load(p))

    for seed in (0, 1):
        for which in ("best_monitor", "final"):
            pp = preds("plus", seed, which)
            pm = preds("minus", seed, which)
            if pp is None or pm is None:
                continue
            a_plus_t = np.abs(pp["q_target"] - y[q_t])
            a_minus_t = np.abs(pm["q_target"] - y[q_t])
            a_plus_b = np.abs(pp["q_background"] - y[q_b])
            a_minus_b = np.abs(pm["q_background"] - y[q_b])
            # per-molecule paired difference, plus_k better when negative
            d_t = a_plus_t - a_minus_t
            d_b = a_plus_b - a_minus_b
            rng = np.random.default_rng(1000 + seed)
            idx = rng.integers(0, len(d_t), size=(5000, len(d_t)))
            boot_t = d_t[idx].mean(1)
            idx_b = rng.integers(0, len(d_b), size=(5000, len(d_b)))
            boot_b = d_b[idx_b].mean(1)
            # key-level aggregation: mean AE per target key over its query molecules
            per_key = []
            for k in keys:
                ms = kmer[str(int(k))]
                if not ms:
                    continue
                pos = {int(m): i for i, m in enumerate(q_t.tolist())}
                ii = [pos[m] for m in ms if m in pos]
                if not ii:
                    continue
                per_key.append((float(a_plus_t[ii].mean()), float(a_minus_t[ii].mean())))
            per_key = np.asarray(per_key, dtype=np.float64)
            dk = per_key[:, 0] - per_key[:, 1] if len(per_key) else np.zeros(0)
            kidx = rng.integers(0, len(dk), size=(5000, len(dk))) if len(dk) else None
            boot_k = dk[kidx].mean(1) if kidx is not None else np.zeros(1)
            out["comparisons"][f"plus_minus_{which}_s{seed}"] = {
                "q_target_mae_plus": float(a_plus_t.mean()),
                "q_target_mae_minus": float(a_minus_t.mean()),
                "q_target_delta": float(d_t.mean()),
                "q_target_ci95": [float(np.percentile(boot_t, 2.5)),
                                  float(np.percentile(boot_t, 97.5))],
                "q_background_mae_plus": float(a_plus_b.mean()),
                "q_background_mae_minus": float(a_minus_b.mean()),
                "q_background_delta": float(d_b.mean()),
                "q_background_ci95": [float(np.percentile(boot_b, 2.5)),
                                      float(np.percentile(boot_b, 97.5))],
                "difference_in_differences": float(d_t.mean() - d_b.mean()),
                "q_target_frac_plus_better": float((d_t < 0).mean()),
                "key_level_delta": float(dk.mean()) if len(dk) else None,
                "key_level_ci95": [float(np.percentile(boot_k, 2.5)),
                                   float(np.percentile(boot_k, 97.5))] if len(dk) else None,
                "n_keys": int(len(per_key)),
                "effective_independent_groups": "target keys (shared across molecules); "
                                                "bootstrap resamples keys, not molecules",
            }
            # strata by key training support
            for lo, hi, label in ((4, 7, "support_4_7"), (8, 15, "support_8_15"),
                                  (16, 25, "support_16_25")):
                sel = [str(int(k)) for k, s_ in zip(keys, key_support) if lo <= s_ <= hi]
                ii = []
                pos = {int(m): i for i, m in enumerate(q_t.tolist())}
                for k in sel:
                    ii.extend([pos[m] for m in kmer[k] if m in pos])
                ii = sorted(set(ii))
                if ii:
                    out["comparisons"][f"plus_minus_{which}_s{seed}"].setdefault("strata", {})[
                        label] = {"n_query_molecules": len(ii),
                                  "delta": float(d_t[ii].mean())}
    C.write_json(OUT / "support_summary.json", out)
    (OUT / "support.md").write_text(_support_md(out), encoding="utf-8")
    return out


def _support_md(s: dict) -> str:
    if s.get("status"):
        return s["status"]
    rows = ["# Supervision-support intervention (official train only)", "",
            f"* target keys: {s['n_target_keys']}", f"* sizes: {s['manifest_sizes']}", "",
            "| comparison | Q_target MAE plus / minus | Δ (plus−minus) [95% CI] | "
            "Q_background Δ | difference-in-differences | key-level Δ [95% CI] |",
            "|---|---:|---|---:|---:|---|"]
    for k, v in s["comparisons"].items():
        rows.append(
            f"| {k} | {v['q_target_mae_plus']:.5f} / {v['q_target_mae_minus']:.5f} | "
            f"{v['q_target_delta']:+.5f} [{v['q_target_ci95'][0]:+.5f}, "
            f"{v['q_target_ci95'][1]:+.5f}] | {v['q_background_delta']:+.5f} | "
            f"{v['difference_in_differences']:+.5f} | "
            f"{('%.5f [%.5f, %.5f]' % (v['key_level_delta'], v['key_level_ci95'][0],
                                        v['key_level_ci95'][1])) if v['key_level_delta'] is not None else '—'} |")
    rows += ["", "Negative Δ = the plus (target-environment donor) fit has lower error.", "",
             "Strata by target-key training support:", ""]
    for k, v in s["comparisons"].items():
        if "strata" in v:
            parts = ", ".join(f"{a}: Δ={b['delta']:+.4f} (n={b['n_query_molecules']})"
                              for a, b in v["strata"].items())
            rows.append(f"* **{k}**: {parts}")
    return "\n".join(rows) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=["core", "support", "all"], nargs="?", default="all")
    args = p.parse_args(argv)
    if args.what in ("core", "all"):
        s = core()
        C.log({"runs": sorted(s["runs"]), "comparisons": sorted(s["comparisons"])})
    if args.what in ("support", "all"):
        try:
            out = support()
            C.log(list(out.get("comparisons", {})))
        except FileNotFoundError as exc:
            C.log(f"support analysis skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
