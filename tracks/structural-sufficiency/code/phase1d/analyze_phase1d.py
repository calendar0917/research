"""Phase 1d GPU analysis: incidence control and bounded capacity scaling.

Reads only the Phase 1d run JSONs plus the frozen Phase 1b/1c Monitor anchors.
No Reserved or official valid/test label is read.
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
PHASE1B = HERE.parent / "phase1b"
for p in (HERE, PHASE1B):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_phase1b import paired_bootstrap  # noqa: E402
from analyze_phase1d_cpu import _align_run, comparison_stats, metrics  # noqa: E402

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1C = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1D = REPO / "tracks/structural-sufficiency/results/phase1d"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
RUNS_1C = REPO / "tracks/structural-sufficiency/runs/phase1c"
RUNS_1D = REPO / "tracks/structural-sufficiency/runs/phase1d"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
FEATURES_PATH = RESULTS_1D / "structural_features.npz"
OUT = RESULTS_1D / "analysis_phase1d.json"
VALIDATION = RESULTS_1D / "run_validation.json"

FROZEN = {
    "R1_size_seed0": RUNS_1C / "zinc/R1_size_seed0.json",
    "R1_size_seed1": RUNS_1C / "zinc/R1_size_seed1.json",
    "G_ref_seed0": RUNS_1B / "zinc/G_ref_seed0.json",
}
BANDS = {"negligible": 0.01, "meaningful": 0.02}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return {"mean": None, "ci95": [None, None]}
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draws[i] = values[rng.integers(0, n, size=n)].mean()
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def _subgroup(
    names: Mapping[str, np.ndarray],
    mask: np.ndarray,
    d: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(mask.sum()), "fraction": float(mask.mean())}
    for label, m in [("negative", ~mask), ("positive", mask)]:
        out[label] = {
            "n": int(m.sum()),
            "fraction": float(m.mean()),
            "model_mae": {k: float(v[m].mean()) for k, v in names.items()},
            "d_mean": float(d[m].mean()) if m.any() else None,
            "d_ci95": _bootstrap_mean(d[m], seed=seed + (1 if label == "positive" else 0))["ci95"],
            "d_weighted_contribution": float(m.mean() * d[m].mean()) if m.any() else None,
            "win_rate_second_better": float(np.mean(d[m] > 0)) if m.any() else None,
        }
    return out


def _decision_b(gain: float, ci: Sequence[float], rmse_agrees: bool) -> dict[str, Any]:
    lo, hi = float(ci[0]), float(ci[1])
    labels: list[str] = []
    if gain >= BANDS["meaningful"] and lo > 0 and rmse_agrees:
        labels.append("PAIRWISE_INTERACTION_PROMISING")
        confidence = "screening_positive"
    elif gain <= -BANDS["meaningful"] and hi < 0 and rmse_agrees:
        labels.append("INCIDENCE_STATISTICS_SUFFICIENT_FOR_CURRENT_GAIN")
        confidence = "screening_incidence_better"
    elif abs(gain) < BANDS["meaningful"]:
        if BANDS["negligible"] <= gain < BANDS["meaningful"] and lo > 0 and rmse_agrees:
            labels.append("INSUFFICIENT_EVIDENCE")
            confidence = "weak_r1_advantage"
        else:
            labels.append("INCIDENCE_STATISTICS_SUFFICIENT_FOR_CURRENT_GAIN")
            confidence = "screening_close_or_incidence"
    else:
        labels.append("INSUFFICIENT_EVIDENCE")
        confidence = "ci_or_rmse_unclear"
    return {
        "labels": labels,
        "confidence": confidence,
        "gain_joint": float(gain),
        "ci95": [lo, hi],
        "rmse_agrees": bool(rmse_agrees),
        "bands": BANDS,
    }


def _decision_c(
    gain: float,
    ci: Sequence[float],
    rmse_agrees: bool,
    gap: float,
    converged: bool,
    efficiency_obstacle: bool,
) -> dict[str, Any]:
    lo, hi = float(ci[0]), float(ci[1])
    labels: list[str] = []
    if not converged:
        labels.append("OPTIMIZATION_UNRESOLVED")
        confidence = "not_converged"
    elif gain >= BANDS["meaningful"] and lo > 0 and rmse_agrees:
        labels.append("CAPACITY_SCALING_PROMISING")
        confidence = "screening_positive"
        if gap >= BANDS["meaningful"]:
            labels.append("REPRESENTATION_OR_INDUCTIVE_BIAS_GAP")
    elif abs(gain) < BANDS["negligible"] or (lo <= 0 <= hi):
        labels.append("REPRESENTATION_OR_INDUCTIVE_BIAS_GAP")
        confidence = "capacity_not_main_bottleneck"
    else:
        labels.append("INSUFFICIENT_EVIDENCE")
        confidence = "weak_or_unclear"
    return {
        "labels": labels,
        "confidence": confidence,
        "gain_capacity": float(gain),
        "ci95": [lo, hi],
        "rmse_agrees": bool(rmse_agrees),
        "gap_to_gref": float(gap),
        "converged": bool(converged),
        "efficiency_obstacle": bool(efficiency_obstacle),
        "bands": BANDS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    split = json.loads(SPLIT_PATH.read_text())
    monitor_rows = [int(x) for x in split["rows"]["monitor"]]
    runs: dict[str, dict[str, Any]] = {}
    for name, path in FROZEN.items():
        if not path.exists():
            raise SystemExit(f"missing frozen run {path}")
        r = _align_run(path, monitor_rows)
        r["name"] = name
        runs[name] = r
    phase1d_files = sorted((RUNS_1D / "zinc").glob("*.json"))
    if not phase1d_files:
        raise SystemExit(f"no Phase 1d run JSONs in {RUNS_1D / 'zinc'}")
    for path in phase1d_files:
        if path.name.endswith("_smoke.json"):
            continue
        r = _align_run(path, monitor_rows)
        r["name"] = path.stem
        runs[r["name"]] = r
    labels = next(iter(runs.values()))["y"]
    for name, r in runs.items():
        if not np.array_equal(r["y"], labels):
            raise SystemExit(f"label mismatch in {name}")

    model_abs = {name: np.abs(r["pred"] - r["y"]) for name, r in runs.items()}
    model_table = {
        name: {
            "model": r["model"],
            "seed": r["seed"],
            "params": r["parameters"],
            "flops_forward": r["flops_forward"],
            "step_time_ms_median": r["step_time_ms_median"],
            "gpu_seconds": r["gpu_seconds"],
            "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "epochs_planned": r["epochs_planned"],
            "train_mae": r["train_mae"],
            "monitor": metrics(r["pred"], r["y"]),
            "path": r["path"],
            "sha256": r["sha256"],
        }
        for name, r in runs.items()
    }

    # Phase B
    incidence_name = next((n for n in runs if n.startswith("R_incidence_seed0") and "cap" not in n), None)
    b: dict[str, Any] = {"present": incidence_name is not None}
    if incidence_name:
        comp = comparison_stats(runs[incidence_name], runs["R1_size_seed0"], args.n_boot)
        gain = comp["delta_mean_d"]
        b.update(
            {
                "incidence": incidence_name,
                "baseline": "R1_size_seed0",
                "comparison": comp,
                "decision": _decision_b(gain, comp["bootstrap_delta"]["ci95"], comp["rmse_sign_agrees"]),
            }
        )
        features = np.load(FEATURES_PATH, allow_pickle=False)
        feat_idx = {int(r): i for i, r in enumerate(features["rows"].tolist())}
        large = np.asarray(
            [features["large_cycles_gt6"][feat_idx[int(m)]] >= 1 for m in monitor_rows],
            dtype=bool,
        )
        d = model_abs[incidence_name] - model_abs["R1_size_seed0"]
        b["large_ring_subgroup"] = _subgroup(
            {
                incidence_name: model_abs[incidence_name],
                "R1_size_seed0": model_abs["R1_size_seed0"],
            },
            large,
            d,
            seed=41,
        )

    # Phase C
    c: dict[str, Any] = {"present": False}
    cap_name = next((n for n in runs if n.endswith("_cap_seed0") or "_cap_seed0" in n), None)
    if cap_name:
        if cap_name.startswith("R1_size_cap"):
            base_name = "R1_size_seed0"
        elif cap_name.startswith("R_incidence_cap"):
            base_name = incidence_name
        else:
            raise SystemExit(f"cannot map capacity run {cap_name}")
        if base_name is None or base_name not in runs:
            raise SystemExit(f"missing base run for {cap_name}")
        comp_cap = comparison_stats(runs[base_name], runs[cap_name], args.n_boot)
        comp_gap = comparison_stats(runs[cap_name], runs["G_ref_seed0"], args.n_boot)
        cap = runs[cap_name]
        gref = runs["G_ref_seed0"]
        gain = comp_cap["delta_mean_d"]
        gap = comp_gap["delta_mean_d"]
        converged = bool(cap["best_epoch"] < cap["epochs_run"] - 1 and cap["epochs_run"] < cap["epochs_planned"])
        flops_ratio = (cap["flops_forward"] / gref["flops_forward"]) if cap["flops_forward"] and gref["flops_forward"] else None
        step_ratio = (cap["step_time_ms_median"] / gref["step_time_ms_median"]) if cap["step_time_ms_median"] and gref["step_time_ms_median"] else None
        gpu_ratio = (cap["gpu_seconds"] / gref["gpu_seconds"]) if cap["gpu_seconds"] and gref["gpu_seconds"] else None
        efficiency_obstacle = bool(
            (flops_ratio is not None and flops_ratio > 2.0) or (step_ratio is not None and step_ratio > 2.0)
        )
        features = np.load(FEATURES_PATH, allow_pickle=False)
        feat_idx = {int(r): i for i, r in enumerate(features["rows"].tolist())}
        large = np.asarray(
            [features["large_cycles_gt6"][feat_idx[int(m)]] >= 1 for m in monitor_rows],
            dtype=bool,
        )
        d_cap = model_abs[base_name] - model_abs[cap_name]
        c.update(
            {
                "present": True,
                "capacity_run": cap_name,
                "base_run": base_name,
                "comparison_capacity": comp_cap,
                "comparison_gap_to_gref": comp_gap,
                "decision": _decision_c(gain, comp_cap["bootstrap_delta"]["ci95"], comp_cap["rmse_sign_agrees"], gap, converged, efficiency_obstacle),
                "convergence": {
                    "best_epoch": cap["best_epoch"],
                    "epochs_run": cap["epochs_run"],
                    "epochs_planned": cap["epochs_planned"],
                    "early_stop_fired": converged,
                    "train_mae": cap["train_mae"],
                    "train_monitor_gap": cap["train_mae"] - cap["monitor_mae"],
                },
                "cost": {
                    "params": cap["parameters"],
                    "params_ratio_vs_gref": cap["parameters"] / gref["parameters"],
                    "flops_forward": cap["flops_forward"],
                    "flops_ratio_vs_gref": flops_ratio,
                    "step_time_ms": cap["step_time_ms_median"],
                    "step_time_ratio_vs_gref": step_ratio,
                    "gpu_seconds": cap["gpu_seconds"],
                    "gpu_seconds_ratio_vs_gref": gpu_ratio,
                },
                "large_ring_subgroup": _subgroup(
                    {base_name: model_abs[base_name], cap_name: model_abs[cap_name], "G_ref_seed0": model_abs["G_ref_seed0"]},
                    large,
                    d_cap,
                    seed=53,
                ),
            }
        )
    validation = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else None
    payload = {
        "protocol_id": "ssrc-phase1d-analysis",
        "scope": {
            "labels_used": "Fit/Monitor only; Monitor predictions in run JSONs",
            "official_valid_test_read": False,
            "internal_reserved_read": False,
            "model_selection": "best Monitor MAE inside the frozen recipe; no post-result tuning",
        },
        "run_validation": validation,
        "models": model_table,
        "phase_b": b,
        "phase_c": c,
        "bands": BANDS,
        "notes": [
            "Single seed; molecule bootstrap is not training-seed replication.",
            "The large-ring subgroup was identified in Phase A and is secondary/descriptive.",
            "No official ZINC test or internal Reserved label was read.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "phase_b": b.get("decision"), "phase_c": c.get("decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
