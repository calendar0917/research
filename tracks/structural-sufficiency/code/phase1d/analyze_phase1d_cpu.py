"""Phase 1d CPU audit: where does the frozen Monitor MAE difference live?

The script reads only the frozen Phase 1b/1c run JSONs (Monitor predictions and
truths) and the structure-only Phase 1d feature table.  It never reads the
internal Reserved split or official ZINC valid/test.

The analysis is descriptive.  It does not select a model, does not fit a new
predictor, and does not turn a structural correlation into a causal mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
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
from ssrc_data import LabelVault, sha256_file  # noqa: E402

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1C = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1D = REPO / "tracks/structural-sufficiency/results/phase1d"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
RUNS_1C = REPO / "tracks/structural-sufficiency/runs/phase1c"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
LABELS_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
FEATURES_PATH = RESULTS_1D / "structural_features.npz"
FEATURES_MANIFEST = RESULTS_1D / "structural_features_manifest.json"
OUT = RESULTS_1D / "analysis_phase1d_cpu.json"

N_BOOT = 5000
GROUP_BOOT = 1000
SPEARMAN_BOOT = 1000

RUN_FILES = {
    "S0_seed0": RUNS_1B / "zinc/S0_seed0.json",
    "S0_seed1": RUNS_1B / "zinc/S0_seed1.json",
    "R1_seed0": RUNS_1B / "zinc/R1_seed0.json",
    "R1_seed1": RUNS_1B / "zinc/R1_seed1.json",
    "G_ref_seed0": RUNS_1B / "zinc/G_ref_seed0.json",
    "S_cap_size_seed0": RUNS_1C / "zinc/S_cap_size_seed0.json",
    "R1_size_seed0": RUNS_1C / "zinc/R1_size_seed0.json",
    "R1_size_seed1": RUNS_1C / "zinc/R1_size_seed1.json",
    "R1_marginal_size_seed0": RUNS_1C / "zinc/R1_marginal_size_seed0.json",
    "R1_marginal_size_seed1": RUNS_1C / "zinc/R1_marginal_size_seed1.json",
    "M_keep_edge_seed0": RUNS_1C / "zinc/M_keep_edge_seed0.json",
}

# Pre-registered primary grouping variables and fixed bins.  The integer edges
# were chosen from the *Fit* structure-only distributions before any Phase 1d
# error analysis (see notes/phase1d_protocol.md); they are not tuned on Monitor.
GROUPS: dict[str, list[float]] = {
    "n_atoms": [0, 21, 24, 27, 1000],
    "rings_sssr": [0, 2, 3, 4, 1000],
    "max_ring_size": [0, 6, 7, 1000],
    "large_cycles_gt6": [0, 1, 2, 1000],
    "kekule6_ring_count": [0, 1, 2, 1000],
    "ring_fusion_bonds": [0, 1, 3, 1000],
    "ring_shared_atoms": [0, 1, 3, 1000],
    "diameter": [0, 11, 13, 15, 1000],
    "patch_size_mean": [0, 5.94, 6.10, 6.27, 1000],
    "patch_dup_rate": [0, 0.05, 0.11, 0.17, 1000],
    "root_vs_patch_tv_mean": [0, 0.33, 0.39, 0.44, 1000],
    "frac_pairs_distance_gt4": [0, 0.459, 0.523, 0.569, 1000],
}

PRIMARY_CORRELATION_VARS = list(GROUPS)
EXTRA_CORRELATION_VARS = [
    "n_bonds",
    "rings_cyclomatic",
    "ring_bond_fraction",
    "frac_ring_atoms",
    "mean_degree",
    "max_degree",
    "min_degree",
    "frac_degree1",
    "single_bonds",
    "double_bonds",
    "triple_bonds",
    "double_bond_fraction",
    "mean_pair_distance",
    "atom_type_entropy",
    "bond_type_entropy",
    "patch_size_max",
    "patch_size_std",
    "patch_whole_fraction",
    "patch_key_entropy",
    "patch_max_multiplicity",
    "patch_unique_count",
    "mean_pool_bits",
    "patch_atom_entropy_mean",
    "first_shell_mean",
    "first_shell_max",
    "second_shell_mean",
]

COMPARISONS = {
    "d_pair_seed0": ("R1_marginal_size_seed0", "R1_size_seed0"),
    "d_pair_seed1": ("R1_marginal_size_seed1", "R1_size_seed1"),
    "d_capacity_seed0": ("S_cap_size_seed0", "R1_size_seed0"),
    "d_gap_seed0": ("R1_size_seed0", "G_ref_seed0"),
    "d_gap_seed1": ("R1_size_seed1", "G_ref_seed0"),
    "d_gap_marginal_seed0": ("R1_marginal_size_seed0", "G_ref_seed0"),
    "d_gap_scap_seed0": ("S_cap_size_seed0", "G_ref_seed0"),
    "d_same_model_R1_size_s0_vs_s1": ("R1_size_seed0", "R1_size_seed1"),
    "d_same_model_marginal_s0_vs_s1": ("R1_marginal_size_seed0", "R1_marginal_size_seed1"),
}


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _align_run(path: Path, monitor_rows: Sequence[int]) -> dict[str, Any]:
    payload = _load_json(path)
    idx = {int(m): i for i, m in enumerate(payload["monitor_mol_index"])}
    if len(idx) != len(payload["monitor_mol_index"]):
        raise ValueError(f"duplicate monitor rows in {path}")
    missing = [m for m in monitor_rows if m not in idx]
    if missing:
        raise ValueError(f"{path} missing {len(missing)} monitor rows")
    pred = np.asarray([payload["monitor_pred_best"][idx[int(m)]] for m in monitor_rows], dtype=np.float64)
    y = np.asarray([payload["monitor_y"][idx[int(m)]] for m in monitor_rows], dtype=np.float64)
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": sha256_file(path),
        "model": payload.get("model"),
        "seed": int(payload["seed"]),
        "parameters": int(payload["parameters"]),
        "flops_forward": payload.get("flops", {}).get("flops_forward"),
        "step_time_ms_median": payload.get("step_time_ms_median"),
        "gpu_seconds": payload.get("gpu_seconds"),
        "best_epoch": int(payload["best_epoch"]),
        "epochs_run": len(payload.get("curves", [])),
        "epochs_planned": int(payload.get("config", {}).get("epochs", -1)),
        "train_mae": float(payload["best_metrics"]["train"]["mae"]),
        "monitor_mae": float(np.mean(np.abs(pred - y))),
        "monitor_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pred": pred,
        "y": y,
    }


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_abs_error": float(np.median(abs_err)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "mean_signed_error": float(err.mean()),
        "n": int(y.size),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {f"p{int(q*100):02d}": float(np.quantile(values, q)) for q in qs}


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = values.size
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n_boot": n_boot}
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        draws[i] = values[rng.integers(0, n, size=n)].mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_boot": int(n_boot),
    }


def _top_q(d: np.ndarray, q: float) -> dict[str, Any]:
    n = d.size
    k = max(1, int(np.ceil(q * n)))
    order = np.argsort(-np.abs(d), kind="stable")
    top = order[:k]
    rest = order[k:]
    total = float(d.sum())
    abs_total = float(np.abs(d).sum())
    sum_top = float(d[top].sum())
    abs_sum_top = float(np.abs(d[top]).sum())
    # diagnostic-only truncations; never report the truncated sample MAE as a
    # model-performance estimate
    abs_threshold = float(np.quantile(np.abs(d), 1.0 - q))
    abs_clipped = np.sign(d) * np.minimum(np.abs(d), abs_threshold)
    lo, hi = float(np.quantile(d, q)), float(np.quantile(d, 1.0 - q))
    signed_clipped = np.clip(d, lo, hi)
    return {
        "q": float(q),
        "n_top": int(k),
        "top_fraction": float(k / n),
        "sum_d_top": sum_top,
        "mean_d_contribution_top": float(sum_top / n),
        "share_of_total_signed_delta": float(sum_top / total) if abs(total) > 1e-15 else None,
        "share_of_total_abs_deviation": float(abs_sum_top / abs_total) if abs_total > 1e-15 else None,
        "delta_after_dropping_top_abs": float(d[rest].mean()) if rest.size else None,
        "abs_clip_threshold": abs_threshold,
        "delta_after_abs_clip": float(abs_clipped.mean()),
        "signed_clip_bounds": [lo, hi],
        "delta_after_signed_clip": float(signed_clipped.mean()),
    }


def _top_direction(d: np.ndarray, positive: bool, q: float) -> dict[str, Any]:
    n = d.size
    k = max(1, int(np.ceil(q * n)))
    order = np.argsort(d if positive else -d, kind="stable")[:k]
    sum_top = float(d[order].sum())
    return {
        "q": float(q),
        "n_top": int(k),
        "sum_d": sum_top,
        "mean_d_contribution": float(sum_top / n),
        "top_values": [float(x) for x in d[order][:10]],
    }


def comparison_stats(a: Mapping[str, Any], b: Mapping[str, Any], n_boot: int) -> dict[str, Any]:
    ya, yb = np.asarray(a["y"]), np.asarray(b["y"])
    if not np.allclose(ya, yb, atol=0, rtol=0):
        raise ValueError("label vectors differ")
    pa, pb = np.asarray(a["pred"]), np.asarray(b["pred"])
    ea, eb = pa - ya, pb - yb
    aa, ab = np.abs(ea), np.abs(eb)
    d = aa - ab
    boot = paired_bootstrap(ea, eb, n_boot=n_boot)
    rmse_a = float(np.sqrt(np.mean(ea**2)))
    rmse_b = float(np.sqrt(np.mean(eb**2)))
    pos = d > 1e-12
    neg = d < -1e-12
    zero = ~(pos | neg)
    top_q = {f"{q:.2f}": _top_q(d, q) for q in (0.01, 0.05, 0.10)}
    return {
        "a": {
            "name": a.get("name"),
            "model": a.get("model"),
            "seed": a.get("seed"),
            "monitor": metrics(pa, ya),
        },
        "b": {
            "name": b.get("name"),
            "model": b.get("model"),
            "seed": b.get("seed"),
            "monitor": metrics(pb, yb),
        },
        "definition": "d = |err_a| - |err_b|; positive => b has lower absolute error",
        "delta_mean_d": float(d.mean()),
        "delta_median_d": float(np.median(d)),
        "delta_std_d": float(np.std(d)),
        "delta_quantiles": _quantiles(d),
        "bootstrap_delta": boot,
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "rmse_delta_a_minus_b": float(rmse_a - rmse_b),
        "rmse_sign_agrees": bool(np.sign(rmse_a - rmse_b) == np.sign(d.mean())),
        "win_rate_b_better": float(np.mean(ab < aa)),
        "win_rate_a_better": float(np.mean(aa < ab)),
        "tie_rate": float(np.mean(aa == ab)),
        "positive_d_fraction": float(pos.mean()),
        "negative_d_fraction": float(neg.mean()),
        "zero_d_fraction": float(zero.mean()),
        "top_abs_d": top_q,
        "top_positive_d": _top_direction(d, True, 0.01),
        "top_negative_d": _top_direction(d, False, 0.01),
        "delta_after_dropping_top1pct": top_q["0.01"]["delta_after_dropping_top_abs"],
        "delta_after_dropping_top5pct": top_q["0.05"]["delta_after_dropping_top_abs"],
        "delta_after_dropping_top10pct": top_q["0.10"]["delta_after_dropping_top_abs"],
        "delta_after_abs_clip_p99": top_q["0.01"]["delta_after_abs_clip"],
        "delta_after_abs_clip_p95": top_q["0.05"]["delta_after_abs_clip"],
        "delta_after_signed_clip_p99": top_q["0.01"]["delta_after_signed_clip"],
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    point = _spearman(a, b)
    if point is None:
        return {"spearman": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    n = a.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = _spearman(a[idx], b[idx])
        if val is not None:
            draws.append(val)
    return {
        "spearman": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))] if draws else [None, None],
        "n_boot": int(n_boot),
    }


def group_stats(
    values: np.ndarray,
    edges: Sequence[float],
    d_map: Mapping[str, np.ndarray],
    model_abs: Mapping[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = values.size
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (values >= lo) & (values < hi)
        if not mask.any():
            continue
        row: dict[str, Any] = {
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "model_mae": {k: float(v[mask].mean()) for k, v in model_abs.items()},
        }
        for name, d in d_map.items():
            vals = d[mask]
            ci = _bootstrap_mean_ci(vals, n_boot, seed + 1000 * i + len(name))
            row[f"{name}_mean"] = ci["mean"]
            row[f"{name}_ci95"] = ci["ci95"]
            row[f"{name}_weighted_contribution"] = float(mask.mean() * vals.mean()) if vals.size else 0.0
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    split = _load_json(SPLIT_PATH)
    monitor_rows = [int(x) for x in split["rows"]["monitor"]]
    fit_rows = [int(x) for x in split["rows"]["fit"]]
    reserved_rows = [int(x) for x in split["rows"]["reserved"]]
    vault = LabelVault(LABELS_PATH, fit_rows, monitor_rows, reserved_rows)
    vault.load()

    runs: dict[str, dict[str, Any]] = {}
    for name, path in RUN_FILES.items():
        if not path.exists():
            raise SystemExit(f"missing frozen run {path}")
        item = _align_run(path, monitor_rows)
        item["name"] = name
        runs[name] = item
    labels = next(iter(runs.values()))["y"]
    for name, r in runs.items():
        if not np.array_equal(r["y"], labels):
            raise SystemExit(f"label mismatch in {name}")

    features = np.load(FEATURES_PATH, allow_pickle=False)
    feat_rows = {int(r): i for i, r in enumerate(features["rows"].tolist())}
    missing = [m for m in monitor_rows if m not in feat_rows]
    if missing:
        raise SystemExit(f"missing structural features for {len(missing)} Monitor rows")
    fvals = {k: np.asarray(features[k], dtype=np.float64)[[feat_rows[m] for m in monitor_rows]] for k in features.files if k != "rows"}

    model_abs = {name: np.abs(r["pred"] - r["y"]) for name, r in runs.items()}
    d_map = {
        name: model_abs[a] - model_abs[b]
        for name, (a, b) in COMPARISONS.items()
    }

    comparisons: dict[str, Any] = {}
    for name, (a_name, b_name) in COMPARISONS.items():
        comparisons[name] = comparison_stats(runs[a_name], runs[b_name], args.n_boot)

    model_table = {
        name: {
            "params": r["parameters"],
            "flops_forward": r["flops_forward"],
            "step_time_ms_median": r["step_time_ms_median"],
            "gpu_seconds": r["gpu_seconds"],
            "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"],
            "train_mae": r["train_mae"],
            "monitor": metrics(r["pred"], r["y"]),
            "path": r["path"],
            "sha256": r["sha256"],
        }
        for name, r in runs.items()
    }

    # Seed consistency of the per-molecule differences.
    seed_consistency: dict[str, Any] = {}
    for label, a, b in [
        ("d_pair", "d_pair_seed0", "d_pair_seed1"),
        ("d_gap_R1size_vs_Gref", "d_gap_seed0", "d_gap_seed1"),
        ("abs_err_R1_size", "R1_size_seed0", "R1_size_seed1"),
        ("abs_err_marginal", "R1_marginal_size_seed0", "R1_marginal_size_seed1"),
    ]:
        if label.startswith("abs_err_"):
            a_name, b_name = a, b
            va, vb = model_abs[a_name], model_abs[b_name]
        else:
            va, vb = d_map[a], d_map[b]
        top_a = set(np.argsort(-np.abs(va), kind="stable")[: max(1, int(np.ceil(0.10 * va.size)))].tolist())
        top_b = set(np.argsort(-np.abs(vb), kind="stable")[: max(1, int(np.ceil(0.10 * vb.size)))].tolist())
        seed_consistency[label] = {
            "seed0": a,
            "seed1": b,
            "pearson": _pearson(va, vb),
            "spearman": _spearman(va, vb),
            "sign_agreement": float(np.mean(np.sign(va) == np.sign(vb))),
            "top10pct_abs_overlap_count": int(len(top_a & top_b)),
            "top10pct_abs_overlap_jaccard": float(len(top_a & top_b) / len(top_a | top_b)),
            "seed0_mean": float(va.mean()),
            "seed1_mean": float(vb.mean()),
            "seed0_win_rate_positive": float(np.mean(va > 0)),
            "seed1_win_rate_positive": float(np.mean(vb > 0)),
        }

    # Binary large-ring subgroup identified in Phase A (structure-only, not a
    # model-selection rule).  The target contains -cycles with cycles >6 atoms,
    # so this subgroup is also a label-mechanism diagnostic.
    subgroup_tables: dict[str, Any] = {}
    for sg_name, sg_mask in [
        ("large_cycles_gt6_ge1", fvals["large_cycles_gt6"] >= 1),
        ("max_ring_size_ge7", fvals["max_ring_size"] >= 7),
    ]:
        rows = []
        for label, mask in [("negative", ~sg_mask), ("positive", sg_mask)]:
            row: dict[str, Any] = {
                "group": label,
                "n": int(mask.sum()),
                "fraction": float(mask.mean()),
                "model_mae": {k: float(v[mask].mean()) for k, v in model_abs.items()},
            }
            for d_name, d in d_map.items():
                vals = d[mask]
                ci = _bootstrap_mean_ci(vals, GROUP_BOOT, seed=101 + len(sg_name) + len(d_name) + (0 if label == "negative" else 1))
                row[f"{d_name}_mean"] = ci["mean"]
                row[f"{d_name}_ci95"] = ci["ci95"]
                row[f"{d_name}_weighted_contribution"] = float(mask.mean() * vals.mean()) if vals.size else 0.0
            rows.append(row)
        subgroup_tables[sg_name] = {
            "definition": "structure-only binary split; positive group is the rare large-ring tail",
            "rows": rows,
        }

    # Catastrophic-error counts are descriptive: they show whether the mean
    # difference is a broad shift or a tail trade-off.
    catastrophic: dict[str, Any] = {}
    for name, ae in model_abs.items():
        catastrophic[name] = {
            "n_abs_err_gt_1": int((ae > 1.0).sum()),
            "n_abs_err_gt_2": int((ae > 2.0).sum()),
            "n_abs_err_gt_3": int((ae > 3.0).sum()),
            "n_large_ring_with_abs_err_gt_1": int(((ae > 1.0) & (fvals["large_cycles_gt6"] >= 1)).sum()),
            "n_large_ring": int((fvals["large_cycles_gt6"] >= 1).sum()),
        }
    for a, b in [
        ("R1_size_seed0", "R1_size_seed1"),
        ("R1_marginal_size_seed0", "R1_marginal_size_seed1"),
        ("R1_size_seed0", "S_cap_size_seed0"),
        ("R1_size_seed0", "G_ref_seed0"),
    ]:
        ma = model_abs[a] > 2.0
        mb = model_abs[b] > 2.0
        catastrophic[f"overlap_gt2_{a}_vs_{b}"] = {
            "both": int((ma & mb).sum()),
            "only_a": int((ma & ~mb).sum()),
            "only_b": int((~ma & mb).sum()),
            "jaccard": float((ma & mb).sum() / max((ma | mb).sum(), 1)),
        }

    # Variable correlations and fixed-bin decompositions.
    variable_correlations: dict[str, Any] = {}
    group_tables: dict[str, Any] = {}
    for var in PRIMARY_CORRELATION_VARS + EXTRA_CORRELATION_VARS:
        if var not in fvals:
            continue
        x = fvals[var]
        variable_correlations[var] = {}
        for d_name in ["d_pair_seed0", "d_gap_seed0", "d_capacity_seed0", "d_gap_seed1"]:
            y = d_map[d_name]
            sp = _spearman_ci(x, y, SPEARMAN_BOOT, seed=17) if d_name in {"d_pair_seed0", "d_gap_seed0"} else {"spearman": _spearman(x, y), "ci95": [None, None]}
            variable_correlations[var][d_name] = {
                "pearson": _pearson(x, y),
                "spearman": sp["spearman"],
                "ci95": sp["ci95"],
            }
    for var, edges in GROUPS.items():
        if var not in fvals:
            continue
        rows = group_stats(
            fvals[var],
            edges,
            {k: d_map[k] for k in ["d_pair_seed0", "d_capacity_seed0", "d_gap_seed0", "d_gap_seed1"]},
            {
                "R1_size_seed0": model_abs["R1_size_seed0"],
                "R1_marginal_size_seed0": model_abs["R1_marginal_size_seed0"],
                "S_cap_size_seed0": model_abs["S_cap_size_seed0"],
                "G_ref_seed0": model_abs["G_ref_seed0"],
            },
            GROUP_BOOT,
            seed=23 + len(var),
        )
        contrib = {
            d_name: float(sum(abs(r[f"{d_name}_weighted_contribution"]) for r in rows))
            for d_name in ["d_pair_seed0", "d_capacity_seed0", "d_gap_seed0", "d_gap_seed1"]
        }
        group_tables[var] = {
            "edges": [float(x) for x in edges],
            "rows": rows,
            "sum_abs_weighted_contribution": contrib,
            "top_abs_bin_contribution_share": {
                d_name: (
                    float(max(abs(r[f"{d_name}_weighted_contribution"]) for r in rows) / contrib[d_name])
                    if contrib[d_name] > 1e-15 else None
                )
                for d_name in contrib
            },
        }

    # Label-mechanism diagnostic only: target correlations, no model use.
    label_diag = {}
    for var in ["n_atoms", "rings_sssr", "large_cycles_gt6", "kekule6_ring_count", "diameter", "patch_dup_rate"]:
        if var in fvals:
            label_diag[var] = {
                "pearson_y": _pearson(fvals[var], labels),
                "spearman_y": _spearman(fvals[var], labels),
            }
    label_diag["target_definition"] = "PyG ZINC subset target = normalized(logP - SAS - cycles), cycles = rings with >6 atoms"

    # Compact top-sample tables (Monitor rows only; labels are allowed here).
    top_tables = {}
    for d_name in ["d_pair_seed0", "d_capacity_seed0", "d_gap_seed0", "d_gap_seed1"]:
        d = d_map[d_name]
        order_pos = np.argsort(-d, kind="stable")[:10]
        order_neg = np.argsort(d, kind="stable")[:10]
        def row(i: int) -> dict[str, Any]:
            return {
                "mol_index": int(monitor_rows[i]),
                "y": float(labels[i]),
                "d": float(d[i]),
                "abs_err_R1_size_seed0": float(model_abs["R1_size_seed0"][i]),
                "abs_err_R1_marginal_size_seed0": float(model_abs["R1_marginal_size_seed0"][i]),
                "abs_err_S_cap_size_seed0": float(model_abs["S_cap_size_seed0"][i]),
                "abs_err_G_ref_seed0": float(model_abs["G_ref_seed0"][i]),
                **{k: float(fvals[k][i]) for k in PRIMARY_CORRELATION_VARS if k in fvals},
            }
        top_tables[d_name] = {
            "top10_positive_d": [row(int(i)) for i in order_pos],
            "top10_negative_d": [row(int(i)) for i in order_neg],
        }

    payload = {
        "protocol_id": "ssrc-phase1d-cpu-analysis",
        "scope": {
            "labels_used": "Fit/Monitor only; Monitor labels in this audit",
            "official_valid_test_read": False,
            "internal_reserved_read": False,
            "model_selection": False,
            "purpose": "descriptive audit; no new model input and no automatic architecture choice",
        },
        "inputs": {
            "split": {"path": str(SPLIT_PATH.relative_to(REPO)), "sha256": sha256_file(SPLIT_PATH)},
            "labels": {"path": str(LABELS_PATH.relative_to(REPO)), "sha256": sha256_file(LABELS_PATH)},
            "features": {
                "path": str(FEATURES_PATH.relative_to(REPO)),
                "sha256": sha256_file(FEATURES_PATH),
                "manifest_sha256": sha256_file(FEATURES_MANIFEST) if FEATURES_MANIFEST.exists() else None,
            },
            "label_vault": vault.audit(),
            "runs": {name: {"path": r["path"], "sha256": r["sha256"]} for name, r in runs.items()},
        },
        "models": model_table,
        "comparisons": comparisons,
        "seed_consistency": seed_consistency,
        "variable_correlations": variable_correlations,
        "groups": group_tables,
        "binary_subgroups": subgroup_tables,
        "catastrophic_error_diagnostic": catastrophic,
        "label_mechanism_diagnostic": label_diag,
        "top_samples": top_tables,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
