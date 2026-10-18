"""Phase 1b analysis: absolute table, pre-registered gains, paired bootstrap.

Reads the per-run JSONs under ``runs/phase1b/<dataset>/`` and writes a compact
summary.  No label outside Fit/Monitor is read: the run JSONs contain only
Monitor predictions and truths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ssrc_data import LIPO_SPLIT  # noqa: E402

MODELS = ["S0", "R1", "M1", "M2", "G_ref"]
EXTRA_MODELS = {"R1_rewire": "R1_seed0_rewire.json", "M1_long": "M1_seed0_long.json"}
BANDS = {"zinc": {"negligible": 0.01, "meaningful": 0.02}, "lipo": {"negligible": 0.02, "meaningful": 0.04}}


def load_runs(root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    out = {}
    for model in MODELS:
        path = root / dataset / f"{model}_seed0.json"
        if path.exists():
            out[model] = json.loads(path.read_text())
    for model, fname in EXTRA_MODELS.items():
        path = root / dataset / fname
        if path.exists():
            out[model] = json.loads(path.read_text())
    return out


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _lipo_groups(runs: Mapping[str, Any]) -> dict[int, str]:
    from phase1a_data import scaffold_groups_from_split

    split = json.loads(LIPO_SPLIT.read_text())
    class S:
        payload = split
    groups = scaffold_groups_from_split(S())  # row -> scaffold hash
    return groups


def paired_bootstrap(
    err_a: np.ndarray,
    err_b: np.ndarray,
    group_ids: np.ndarray | None = None,
    n_boot: int = 5000,
    seed: int = 20261019,
) -> dict[str, Any]:
    """Bootstrap CI for mean|err_a| - mean|err_b| (positive: b better)."""
    a = np.abs(np.asarray(err_a, dtype=np.float64))
    b = np.abs(np.asarray(err_b, dtype=np.float64))
    if group_ids is None:
        group_ids = np.arange(a.size)
    unique = np.unique(group_ids)
    members = {g: np.flatnonzero(group_ids == g) for g in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([members[g] for g in sampled])
        deltas[i] = a[idx].mean() - b[idx].mean()
    point = float(a.mean() - b.mean())
    return {
        "delta": point,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "n_boot": int(n_boot),
        "n_groups": int(unique.size),
        "positive_frac": float(np.mean(deltas > 0)),
    }


def summarize_dataset(dataset: str, runs: Mapping[str, dict[str, Any]], n_boot: int = 5000) -> dict[str, Any]:
    if not runs:
        return {"dataset": dataset, "models": {}, "note": "no runs"}
    labels = np.asarray(runs[next(iter(runs))]["monitor_y"], dtype=np.float64)
    preds = {m: np.asarray(r["monitor_pred_best"], dtype=np.float64) for m, r in runs.items()}
    errs = {m: preds[m] - labels for m in runs}
    groups = None
    if dataset == "lipo":
        try:
            gmap = _lipo_groups(runs)
            row_index = [int(r) for r in runs[next(iter(runs))]["monitor_mol_index"]]
            groups = np.asarray([gmap[i] for i in row_index])
        except Exception:
            groups = None

    table = {}
    for m, r in runs.items():
        hist = r.get("curves", [])
        last_mon = hist[-1]["monitor_mae"] if hist else None
        table[m] = {
            "params": r["parameters"],
            "flops_forward": r.get("flops", {}).get("flops_forward"),
            "step_time_ms": r.get("step_time_ms_median"),
            "train_seconds": r.get("train_seconds"),
            "gpu_seconds": r.get("gpu_seconds"),
            "peak_gpu_mb": r.get("peak_gpu_memory_mb_allocated"),
            "best_epoch": r.get("best_epoch"),
            "epochs_run": len(hist),
            "monitor_mae": mae(preds[m], labels),
            "monitor_rmse": rmse(preds[m], labels),
            "monitor_mae_last": last_mon,
            "train_mae": r["best_metrics"]["train"]["mae"],
            "train_monitor_gap": r["best_metrics"]["monitor"]["mae"] - r["best_metrics"]["train"]["mae"],
        }

    constant_mae = float(np.mean(np.abs(labels - labels.mean())))
    comparisons = {}
    pairs = {
        "relation_S0_minus_R1": ("S0", "R1"),
        "update_R1_minus_M1": ("R1", "M1"),
        "iteration_M1_minus_M2": ("M1", "M2"),
        "gap_S0_minus_Gref": ("S0", "G_ref"),
        "gap_R1_minus_Gref": ("R1", "G_ref"),
        "gap_M1_minus_Gref": ("M1", "G_ref"),
        "gap_M2_minus_Gref": ("M2", "G_ref"),
        # pre-registered controls
        "control_broken_adjacency_R1rewire_minus_R1": ("R1_rewire", "R1"),
        "control_matched_compute_M1long_minus_M2": ("M1_long", "M2"),
    }
    for name, (a, b) in pairs.items():
        if a in errs and b in errs:
            comparisons[name] = paired_bootstrap(errs[a], errs[b], groups, n_boot=n_boot)
    return {
        "dataset": dataset,
        "n_monitor": int(labels.size),
        "constant_baseline_mae": constant_mae,
        "models": table,
        "comparisons": comparisons,
        "bands": BANDS[dataset],
        "bootstrap_grouping": "scaffold_cluster" if groups is not None else "molecule",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "tracks/structural-sufficiency/runs/phase1b"))
    ap.add_argument("--out", default=str(REPO / "tracks/structural-sufficiency/results/phase1b/analysis_phase1b.json"))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    root = Path(args.runs)
    out: dict[str, Any] = {"protocol_id": "ssrc-phase1b-analysis", "datasets": {}}
    for dataset in ["zinc", "lipo"]:
        runs = load_runs(root, dataset)
        out["datasets"][dataset] = summarize_dataset(dataset, runs, n_boot=args.n_boot)
        # Phase D: paired seed replication for the gate-passing relation candidate.
        rep = {}
        for seed in (0, 1):
            s0 = root / dataset / f"S0_seed{seed}.json"
            r1 = root / dataset / f"R1_seed{seed}.json"
            if s0.exists() and r1.exists():
                a = json.loads(s0.read_text())
                b = json.loads(r1.read_text())
                ma = a["best_metrics"]["monitor"]["mae"]
                mb = b["best_metrics"]["monitor"]["mae"]
                rep[f"seed{seed}"] = {
                    "S0_monitor_mae": ma,
                    "R1_monitor_mae": mb,
                    "gain_relation": ma - mb,
                    "S0_best_epoch": a["best_epoch"],
                    "R1_best_epoch": b["best_epoch"],
                }
        if rep:
            gains = [v["gain_relation"] for v in rep.values()]
            rep["pooled"] = {
                "n_seeds": len(gains),
                "mean_gain_relation": float(np.mean(gains)),
                "min_gain_relation": float(np.min(gains)),
                "max_gain_relation": float(np.max(gains)),
                "sign_consistent": bool(all(g > 0 for g in gains) or all(g < 0 for g in gains)),
                "s0_run_to_run_abs_diff": abs(rep["seed0"]["S0_monitor_mae"] - rep["seed1"]["S0_monitor_mae"]) if len(gains) > 1 else None,
                "r1_run_to_run_abs_diff": abs(rep["seed0"]["R1_monitor_mae"] - rep["seed1"]["R1_monitor_mae"]) if len(gains) > 1 else None,
            }
        out["datasets"][dataset]["seed_replication"] = rep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    for dataset, d in out["datasets"].items():
        if not d.get("models"):
            print(f"\n{dataset}: no runs")
            continue
        print(f"\n== {dataset} (n_monitor={d['n_monitor']}, constant MAE={d['constant_baseline_mae']:.4f}) ==")
        print(f"{'model':8s} {'params':>9s} {'MAE':>8s} {'RMSE':>8s} {'train':>8s} {'best_ep':>7s} {'epochs':>6s} {'step_ms':>8s}")
        for m in MODELS:
            if m not in d["models"]:
                continue
            v = d["models"][m]
            print(f"{m:8s} {v['params']:9d} {v['monitor_mae']:8.4f} {v['monitor_rmse']:8.4f} "
                  f"{v['train_mae']:8.4f} {v['best_epoch']:7d} {v['epochs_run']:6d} {v['step_time_ms']:8.2f}")
        for k, v in d["comparisons"].items():
            print(f"  {k:26s} {v['delta']:+.4f}  CI95 [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  pos={v['positive_frac']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
