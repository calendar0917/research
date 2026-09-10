"""Compact-v5 multi-quantile stage analysis (validation-only diagnostics).

Reads compact-v5 run artifacts (validation per-molecule predictions produced by
the frozen-objective runs) and the frozen audit feature tables, then produces
the pre-registered Stage-1/Stage-2 diagnostics and decision record:

* point-prediction comparison (valid q50 MAE vs compact-v4-hinge canonical runs)
* width-error Spearman, width quintile ladder, coverage, width distribution
* rarity -> width mechanism, topology-subgroup / rarity-group MAE tables
* v4 seed-disagreement and frozen OOF difficulty-predictor correlations
* figures 1-6 + per-molecule validation CSVs under
  ``results/compact_v5_multi_quantile/``

The module never loads the official test split and never fits anything on
validation (rarity/topology/component context comes from frozen audit tables).

Usage:
    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v5_multi_quantile_analysis \
        <run_map.json> [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = (
    REPO_ROOT / "tracks/ksvd/results/compact_v5_multi_quantile"
)
MASTER_TABLE = (
    REPO_ROOT
    / "tracks/ksvd/results/post_v4_residual_audit/validation_master_table.csv"
)
OOF_TABLE = (
    REPO_ROOT / "tracks/ksvd/results/oof_difficulty_audit/oof_per_molecule.csv"
)

# canonical compact-v4-hinge runs (primary benchmark protocol, seeds 0-3)
V4_CANONICAL_RUNS = {
    0: "20260909-194445-182c7021",
    1: "20260909-200320-34b347bf",
    2: "20260909-201918-45fbe48d",
    3: "20260909-203516-04e62a28",
}

RARITY_BINS = [
    (0.0, "0", "none"),
    (0.01, "(0.01, 0.05]", "(0, 0.05]"),
    (0.05, "(0.05, 0.1]", "(0.05, 0.1]"),
    (0.1, "(0.1, 0.2]", "(0.1, 0.2]"),
    (0.2, "> 0.2", "> 0.2"),
]

# fixed rarity groups used in the subgroup table (pre-registered bins)
RARITY_GROUP_BINS = [
    ("easy/common", lambda r: r == 0.0),
    ("medium", lambda r: r > 0.0 and r <= 0.1),
    ("rare", lambda r: r > 0.1),
]


def _locate_run_dir(run_id: str) -> Path:
    for manifest in (REPO_ROOT / "tracks/ksvd/runs").rglob(
        f"{run_id}/manifest.json"
    ):
        return manifest.parent
    raise FileNotFoundError(f"run {run_id} not found under runs/")


def _read_run_artifact(run_id: str, name: str) -> dict[str, Any]:
    run_dir = _locate_run_dir(run_id)
    artifact = run_dir / "artifacts" / name
    if not artifact.exists():
        raise FileNotFoundError(f"{artifact} missing for run {run_id}")
    return json.loads(artifact.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(scipy_stats.spearmanr(x, y).statistic)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _qcut_quintile_labels(values: np.ndarray, q: int = 5) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first").to_numpy()
    labels = pd.qcut(ranks, q, labels=False)
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    return np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# run artifact -> per-molecule validation frame (valid:NNNN order)
# ---------------------------------------------------------------------------

def _variant_frame(
    run_id: str,
    quantile: bool,
) -> pd.DataFrame:
    payload = _read_run_artifact(run_id, "legacy_full_result.json")
    valid = payload["evaluation"]["valid"]
    targets = np.asarray(valid["targets"], dtype=np.float64)
    predictions = np.asarray(valid["predictions"], dtype=np.float64)
    n = int(targets.shape[0])
    frame = pd.DataFrame(
        {
            "molecule_id": [f"valid:{i:04d}" for i in range(n)],
            "target": targets,
            "q50": predictions,
        }
    )
    frame["abs_error"] = np.abs(frame["target"].to_numpy() - predictions)
    frame["signed_residual"] = frame["target"].to_numpy() - predictions
    if quantile:
        q = valid["quantile_predictions"]
        frame["q10"] = np.asarray(q["q10"], dtype=np.float64)
        frame["q90"] = np.asarray(q["q90"], dtype=np.float64)
        width = frame["q90"].to_numpy() - frame["q10"].to_numpy()
        frame["width80"] = width
        frame["lower_width"] = predictions - frame["q10"].to_numpy()
        frame["upper_width"] = frame["q90"].to_numpy() - predictions
    return frame


def _merge_master(frame: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(MASTER_TABLE)
    keep = [
        "molecule_id",
        "subgroup",
        "z_logP",
        "z_SA",
        "z_cycle",
        "rare_le1_ratio",
        "rare_le2_ratio",
        "rare_le5_ratio",
        "rare_le10_ratio",
        "oov_patch_ratio",
        "min_train_frequency",
        "mean_log1p_train_frequency",
        "residual_std_across_seeds",
        "prediction_seed0",
        "num_patches",
        "num_nodes",
        "mean_abs_error",
    ]
    keep = [c for c in keep if c in master.columns]
    return frame.merge(master[keep], on="molecule_id", how="left", validate="one_to_one")


def _run_meta(run_id: str) -> dict[str, Any]:
    payload = _read_run_artifact(run_id, "legacy_full_result.json")
    valid = payload["evaluation"]["valid"]
    return {
        "run_id": run_id,
        "candidate": payload.get("protocol_id"),
        "valid_q50_mae": float(valid["mae"]),
        "best_mae": float(valid["best_mae"]),
        "selected_epoch": int(valid["selected_epoch"]),
        "epochs_run": int(valid["epochs_run"]),
        "parameters": int(payload["evaluation"]["parameters"]),
        "loss": payload["training"].get("loss"),
        "quantile_mode": payload["training"].get("quantile_mode"),
        "quantile_lambda": payload["training"].get("quantile_lambda"),
    }


# ---------------------------------------------------------------------------
# diagnostics over one mq frame
# ---------------------------------------------------------------------------

def _quantile_diagnostics(
    frame: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"variant": label, "n": int(len(frame))}
    width = frame["width80"].to_numpy()
    abs_err = frame["abs_error"].to_numpy()
    signed = frame["signed_residual"].to_numpy()
    target = frame["target"].to_numpy()
    q10 = frame["q10"].to_numpy()
    q50 = frame["q50"].to_numpy()
    q90 = frame["q90"].to_numpy()
    out["q50_mae"] = float(np.mean(abs_err))
    out["spearman_width_abs_error"] = _spearman(width, abs_err)
    out["pearson_width_abs_error"] = _pearson(width, abs_err)
    out["spearman_width_signed"] = _spearman(width, signed)
    out["pearson_width_signed"] = _pearson(width, signed)
    out["coverage_q10"] = float(np.mean(target <= q10))
    out["coverage_q90"] = float(np.mean(target <= q90))
    out["central_80_coverage"] = float(np.mean((target >= q10) & (target <= q90)))
    out["width_mean"] = float(np.mean(width))
    out["width_median"] = float(np.median(width))
    out["width_std"] = float(np.std(width))
    out["width_p10"] = float(np.percentile(width, 10))
    out["width_p50"] = float(np.percentile(width, 50))
    out["width_p90"] = float(np.percentile(width, 90))
    out["width_min"] = float(np.min(width))
    out["width_max"] = float(np.max(width))
    out["lower_width_mean"] = float(np.mean(frame["lower_width"]))
    out["upper_width_mean"] = float(np.mean(frame["upper_width"]))
    out["asymmetry_mean"] = float(
        np.mean(frame["upper_width"] - frame["lower_width"])
    )
    out["asymmetry_median"] = float(
        np.median(frame["upper_width"] - frame["lower_width"])
    )
    out["q90_max"] = float(np.max(q90))
    out["q10_min"] = float(np.min(q10))
    # rarity and component axes
    out["spearman_width_rare_le5"] = _spearman(
        width, frame["rare_le5_ratio"].to_numpy()
    )
    out["spearman_width_oov"] = _spearman(
        width, frame["oov_patch_ratio"].to_numpy()
    )
    out["spearman_width_zSA"] = _spearman(width, frame["z_SA"].to_numpy())
    out["spearman_width_zlogP"] = _spearman(width, frame["z_logP"].to_numpy())
    out["spearman_width_v4_seed_disagreement"] = _spearman(
        width, frame["residual_std_across_seeds"].to_numpy()
    )
    # width quintile ladder (actual MAE per predicted-width quintile)
    quintile = _qcut_quintile_labels(width)
    ladder: list[dict[str, Any]] = []
    for b in range(5):
        mask = quintile == b
        ladder.append(
            {
                "width_quintile": f"Q{b+1}",
                "n": int(mask.sum()),
                "actual_mae": float(np.mean(abs_err[mask])),
                "median_abs_error": float(np.median(abs_err[mask])),
                "mean_width": float(np.mean(width[mask])),
                "rare_le5_ratio_mean": float(
                    np.mean(frame["rare_le5_ratio"].to_numpy()[mask])
                ),
            }
        )
    out["width_quintiles"] = ladder
    # rarity bin table (fixed half-open bins on rare<=5 ratio)
    bins: list[dict[str, Any]] = []
    rare = frame["rare_le5_ratio"].to_numpy()
    for _, label_bin, _ in RARITY_BINS:
        pass
    edges = [0.0, 0.01, 0.05, 0.1, 0.2, np.inf]
    names = ["0", "(0.01, 0.05]", "(0.05, 0.1]", "(0.1, 0.2]", "> 0.2"]
    for i in range(len(edges) - 1):
        if i == 0:
            mask = rare == 0.0
        else:
            lo, hi = edges[i], edges[i + 1]
            mask = (rare > lo) & (rare <= hi)
        bins.append(
            {
                "rarity_bin": names[i],
                "n": int(mask.sum()),
                "q50_mae": float(np.mean(abs_err[mask])) if mask.sum() else float("nan"),
                "mean_width80": (
                    float(np.mean(width[mask])) if mask.sum() else float("nan")
                ),
                "mean_abs_error": (
                    float(np.mean(abs_err[mask])) if mask.sum() else float("nan")
                ),
                "mean_rare_le5": (
                    float(np.mean(rare[mask])) if mask.sum() else float("nan")
                ),
            }
        )
    out["rarity_bins"] = bins
    return out


# ---------------------------------------------------------------------------
# frozen OOF difficulty predictor (static model-visible features)
# ---------------------------------------------------------------------------

STATIC_BLOCKS = {
    "rarity": [
        "oov_patch_ratio",
        "rare_le1_ratio",
        "rare_le2_ratio",
        "rare_le5_ratio",
        "rare_le10_ratio",
        "min_train_frequency",
        "mean_train_frequency",
        "median_train_frequency",
        "mean_log1p_train_frequency",
        "mean_inverse_of_1p_frequency",
        "num_unique_patch_tokens",
        "unique_patch_ratio",
    ],
    "structure": [
        "num_nodes",
        "num_edges",
        "num_patches",
        "num_pairs",
        "diameter",
        "mean_shortest_path_distance",
        "max_shortest_path_distance",
        "average_degree",
        "max_degree",
        "num_branching_nodes",
        "branching_node_ratio",
        "cycle_rank",
        "num_components",
        "density",
        "fraction_nodes_in_cycles",
        "num_simple_cycles_bounded",
        "cycle_length_mean",
        "num_nodes_in_cycles",
        "frac_dist_gt_2",
    ],
}


def _oof_static_predictor_summary() -> dict[str, Any]:
    """Replicate the OOF audit's model-visible difficulty predictor restricted
    to static features that exist identically on the validation master table,
    then apply the frozen fit to validation and correlate with width80."""
    oof = pd.read_csv(OOF_TABLE)
    master = pd.read_csv(MASTER_TABLE)
    common_candidates = [c for c in oof.columns if c.startswith("topo_")]
    candidates = STATIC_BLOCKS["rarity"] + STATIC_BLOCKS["structure"] + common_candidates
    # keep only columns present identically in the OOF-train and validation
    # tables so the frozen fit transfers without any re-fitting on validation
    feature_cols = [c for c in candidates if c in oof.columns and c in master.columns]
    error = oof["mean_absolute_error"].to_numpy(dtype=np.float64)
    folds = oof["outer_fold"].to_numpy(dtype=np.int64)
    log_target = np.log(error + 1e-3)
    X = oof[feature_cols].to_numpy(dtype=np.float64)
    # nested cross-fit (same scheme as the OOF audit) for the train-side number
    pred_log = np.full(len(oof), np.nan, dtype=np.float64)
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=1.0).fit(
            scaler.transform(X[tr]), log_target[tr]
        )
        pred_log[te] = model.predict(scaler.transform(X[te]))
    pred_exp = np.clip(np.exp(pred_log) - 1e-3, 0.0, None)
    oof_spearman = _spearman(pred_exp, error)
    # frozen fit on the full OOF train, applied to validation
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), log_target)
    summary: dict[str, Any] = {
        "n_features": len(feature_cols),
        "oof_train_spearman_static": oof_spearman,
        "frozen_fit_scope": "full OOF-train (10k), StandardScaler+Ridge on log(error+1e-3)",
    }
    return summary, model, scaler, feature_cols


def _frozen_difficulty_on_validation(model, scaler, feature_cols) -> np.ndarray:
    master = pd.read_csv(MASTER_TABLE)
    cols = [c for c in feature_cols if c in master.columns]
    X = master[cols].to_numpy(dtype=np.float64)
    pred_log = model.predict(scaler.transform(X))
    return np.clip(np.exp(pred_log) - 1e-3, 0.0, None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _multiseed_table(
    seed_map: Mapping[str, str],
    v4_seed_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Paired validation comparison (v5 q50 vs canonical v4) across seeds."""
    v4_seed_map = v4_seed_map or V4_CANONICAL_RUNS
    rows: list[dict[str, Any]] = []
    for seed_key, run_id in sorted(seed_map.items(), key=lambda kv: int(kv[0])):
        seed = int(seed_key)
        v5_payload = _read_run_artifact(run_id, "legacy_full_result.json")
        valid = v5_payload["evaluation"]["valid"]
        q50 = np.asarray(valid["predictions"], dtype=np.float64)
        target = np.asarray(valid["targets"], dtype=np.float64)
        q10 = np.asarray(valid["quantile_predictions"]["q10"], dtype=np.float64)
        q90 = np.asarray(valid["quantile_predictions"]["q90"], dtype=np.float64)
        width = q90 - q10
        abs_err = np.abs(target - q50)
        v4_id = v4_seed_map.get(seed)
        v4_valid = (
            float(
                _read_run_artifact(v4_id, "legacy_full_result.json")["evaluation"][
                    "valid"
                ]["best_mae"]
            )
            if v4_id is not None
            else float("nan")
        )
        rows.append(
            {
                "seed": seed,
                "v4_run": v4_id,
                "v4_valid_best_mae": v4_valid,
                "v5_run": run_id,
                "v5_valid_q50_mae": float(valid["best_mae"]),
                "delta_v4_minus_v5": float(v4_valid - valid["best_mae"]),
                "width_error_spearman": _spearman(width, abs_err),
                "central_80_coverage": float(np.mean((target >= q10) & (target <= q90))),
                "coverage_q10": float(np.mean(target <= q10)),
                "coverage_q90": float(np.mean(target <= q90)),
                "selected_epoch": int(valid["selected_epoch"]),
            }
        )
    deltas = np.array([r["delta_v4_minus_v5"] for r in rows], dtype=np.float64)
    width_spearman = np.array(
        [r["width_error_spearman"] for r in rows], dtype=np.float64
    )
    return {
        "rows": rows,
        "paired_mean_delta": float(np.mean(deltas)),
        "paired_std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
        "paired_median_delta": float(np.median(deltas)),
        "paired_min_delta": float(np.min(deltas)),
        "paired_max_delta": float(np.max(deltas)),
        "seeds_v5_better": int(np.sum(deltas > 0)),
        "n_seeds": int(len(deltas)),
        "width_spearman_mean": float(np.mean(width_spearman)),
        "width_spearman_std": float(np.std(width_spearman, ddof=1)) if len(width_spearman) > 1 else float("nan"),
        "width_spearman_min": float(np.min(width_spearman)),
        "seeds_width_positive": int(np.sum(width_spearman > 0)),
    }


def classify_stage1(
    point_improvement: float,
    width_spearman: float,
) -> dict[str, Any]:
    """Stage-1 (seed 0) pre-registered bands (§16, §18) and Case (§34)."""
    if point_improvement >= 0.008:
        point_band = "Strong GO"
    elif point_improvement >= 0.005:
        point_band = "GO"
    elif point_improvement >= 0.003:
        point_band = "Mild"
    else:
        point_band = "NO-GO"
    if width_spearman >= 0.40:
        width_band = "Strong uncertainty signal"
    elif width_spearman >= 0.25:
        width_band = "GO"
    elif width_spearman >= 0.15:
        width_band = "Mild"
    else:
        width_band = "NO-GO"
    point_go = point_band in ("Strong GO", "GO")
    width_go = width_band in ("Strong uncertainty signal", "GO")
    if point_go and width_go:
        case = "A"
    elif (not point_go) and point_band == "NO-GO" and width_go:
        case = "B"
    elif point_go and (not width_go):
        case = "C"
    elif point_band == "NO-GO" and width_band == "NO-GO":
        case = "D"
    else:
        case = "unresolved"
    return {
        "point_prediction_band": point_band,
        "uncertainty_band": width_band,
        "case": case,
    }


def classify_case(
    point_improvement: float,
    point_seeds_better: int,
    n_seeds: int,
    width_spearman_mean: float,
    seeds_width_positive: int,
) -> dict[str, Any]:
    """Pre-registered gates (§16, §18, §38, §39, §56)."""
    if point_improvement >= 0.008 and point_seeds_better >= 3:
        point_band = "Strong GO"
    elif point_improvement >= 0.005 and point_seeds_better >= 3:
        point_band = "GO"
    elif point_improvement >= 0.003 and point_seeds_better * 2 > n_seeds:
        point_band = "Mild"
    else:
        point_band = "NO-GO"
    if width_spearman_mean >= 0.40:
        width_band = "Strong uncertainty signal"
    elif width_spearman_mean >= 0.25 and seeds_width_positive >= 3:
        width_band = "GO"
    elif width_spearman_mean >= 0.15:
        width_band = "Mild"
    else:
        width_band = "NO-GO"
    point_go = point_band in ("Strong GO", "GO")
    width_go = width_band in ("Strong uncertainty signal", "GO")
    if point_go and width_go:
        case = "A"
    elif (not point_go) and point_band == "NO-GO" and width_go:
        case = "B"
    elif point_go and (not width_go):
        case = "C"
    elif point_band == "NO-GO" and width_band == "NO-GO":
        case = "D"
    else:
        case = "unresolved (neither GO nor NO-GO)"
    return {
        "point_prediction_band": point_band,
        "uncertainty_band": width_band,
        "case": case,
    }


def _test_table(
    test_map: Mapping[str, str],
    v4_seed_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Frozen one-time benchmark-protocol test comparison vs canonical v4."""
    v4_seed_map = v4_seed_map or V4_CANONICAL_RUNS
    rows: list[dict[str, Any]] = []
    for seed_key, run_id in sorted(test_map.items(), key=lambda kv: int(kv[0])):
        seed = int(seed_key)
        v5_payload = _read_run_artifact(run_id, "legacy_full_result.json")
        v5_mae = v5_payload["evaluation"].get("test_with_selection_checkpoint", {}).get("mae")
        v4_id = v4_seed_map.get(seed)
        v4_mae = (
            _read_run_artifact(v4_id, "legacy_full_result.json")["evaluation"][
                "test_with_selection_checkpoint"
            ]["mae"]
            if v4_id is not None
            else float("nan")
        )
        rows.append(
            {
                "seed": seed,
                "v4_run": v4_id,
                "v4_test_mae": float(v4_mae),
                "v5_run": run_id,
                "v5_test_mae": float(v5_mae),
                "delta_v4_minus_v5": float(v4_mae - v5_mae),
            }
        )
    v4 = np.array([r["v4_test_mae"] for r in rows], dtype=np.float64)
    v5 = np.array([r["v5_test_mae"] for r in rows], dtype=np.float64)
    deltas = v4 - v5
    return {
        "rows": rows,
        "v4_mean": float(np.mean(v4)),
        "v4_std": float(np.std(v4, ddof=1)),
        "v5_mean": float(np.mean(v5)),
        "v5_std": float(np.std(v5, ddof=1)),
        "paired_mean_delta": float(np.mean(deltas)),
        "paired_std_delta": float(np.std(deltas, ddof=1)),
        "paired_median_delta": float(np.median(deltas)),
        "seeds_v5_better": int(np.sum(deltas > 0)),
        "n_seeds": int(len(deltas)),
    }


def _load_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _subgroup_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for group in ("A", "B", "C"):
        mask = frame["subgroup"] == group
        if mask.sum() == 0:
            continue
        sub = frame[mask]
        rows.append(
            {
                "group": group,
                "n": int(len(sub)),
                "v4_q50_mae": float(np.mean(sub["v4_abs_error"])),
                "v5_q50_mae": float(np.mean(sub["abs_error"])),
                "delta_v4_minus_v5": float(
                    np.mean(sub["v4_abs_error"]) - np.mean(sub["abs_error"])
                ),
                "mean_width80": (
                    float(np.mean(sub["width80"])) if "width80" in sub else None
                ),
            }
        )
    return rows


def _rarity_group_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rare = frame["rare_le5_ratio"].to_numpy()
    rows = []
    for name, predicate in RARITY_GROUP_BINS:
        mask = np.array([predicate(v) for v in rare])
        if mask.sum() == 0:
            continue
        sub = frame[mask]
        rows.append(
            {
                "rarity_group": name,
                "n": int(len(sub)),
                "v4_q50_mae": float(np.mean(sub["v4_abs_error"])),
                "v5_q50_mae": float(np.mean(sub["abs_error"])),
                "delta_v4_minus_v5": float(
                    np.mean(sub["v4_abs_error"]) - np.mean(sub["abs_error"])
                ),
                "mean_width80": (
                    float(np.mean(sub["width80"])) if "width80" in sub else None
                ),
            }
        )
    return rows


def _top_widest(frame: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    cols = [
        "molecule_id",
        "target",
        "q10",
        "q50",
        "q90",
        "width80",
        "abs_error",
        "rare_le5_ratio",
        "oov_patch_ratio",
        "subgroup",
        "z_SA",
    ]
    cols = [c for c in cols if c in frame.columns]
    return frame.sort_values("width80", ascending=False).head(k)[cols]


def _figures(frame: pd.DataFrame, label: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    width = frame["width80"].to_numpy()
    abs_err = frame["abs_error"].to_numpy()
    rare = frame["rare_le5_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(width, abs_err, s=6, alpha=0.3, rasterized=True)
    order = np.argsort(width)
    edges = np.percentile(width, np.linspace(0, 100, 11))
    bin_means = []
    for i in range(len(edges) - 1):
        mask = (width >= edges[i]) & (width <= edges[i + 1]) if i == len(edges) - 2 else (
            (width >= edges[i]) & (width < edges[i + 1])
        )
        bin_means.append(np.mean(abs_err[mask]))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, bin_means, "o-", color="tab:red")
    ax.set_xlabel("width80 (q90-q10)")
    ax.set_ylabel("actual |error|")
    ax.set_title(f"{label}: width80 vs |error|")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig1_width_vs_error_{label}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    edges = [0.0, 0.01, 0.05, 0.1, 0.2, 1.0]
    names = ["0", "(0,0.05]", "(0.05,0.1]", "(0.1,0.2]", ">0.2"]
    means = []
    for i in range(len(edges) - 1):
        if i == 0:
            mask = rare == 0.0
        else:
            mask = (rare > edges[i]) & (rare <= edges[i + 1])
        means.append(np.mean(width[mask]) if mask.sum() else np.nan)
    ax.bar(names, means)
    ax.set_xlabel("rare<=5 ratio bin")
    ax.set_ylabel("mean width80")
    ax.set_title(f"{label}: rarity vs width80")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig2_rarity_vs_width_{label}.png", dpi=120)
    plt.close(fig)

    quintile = _qcut_quintile_labels(width)
    fig, ax = plt.subplots(figsize=(6, 5))
    mae = [np.mean(abs_err[quintile == b]) for b in range(5)]
    med = [np.median(abs_err[quintile == b]) for b in range(5)]
    ax.plot(range(1, 6), mae, "o-", label="actual MAE")
    ax.plot(range(1, 6), med, "s--", label="median abs error")
    ax.set_xlabel("predicted-width quintile")
    ax.set_ylabel("actual error")
    ax.set_title(f"{label}: width-quintile vs actual MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig3_width_quintile_mae_{label}.png", dpi=120)
    plt.close(fig)

    # representative easy vs hard molecules by predicted width
    easy = frame.nsmallest(5, "width80")
    hard = frame.nlargest(5, "width80")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, subset, title in ((axes[0], easy, "easiest width"), (axes[1], hard, "hardest width")):
        x = np.arange(len(subset))
        q10 = subset["q10"].to_numpy()
        q50 = subset["q50"].to_numpy()
        q90 = subset["q90"].to_numpy()
        ax.vlines(x, q10, q90, color="tab:blue", alpha=0.5)
        ax.plot(x, q50, "o", color="tab:blue")
        ax.plot(x, subset["target"].to_numpy(), "x", color="tab:red")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["molecule_id"], rotation=90, fontsize=6)
        ax.axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("target / quantiles")
    fig.suptitle(f"{label}: q10/q50/q90 for representative molecules")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig4_easy_hard_quantiles_{label}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    disagreement = frame["residual_std_across_seeds"].to_numpy()
    ax.scatter(disagreement, width, s=6, alpha=0.3, rasterized=True)
    ax.set_xlabel("v4 4-seed prediction std (disagreement)")
    ax.set_ylabel("width80")
    ax.set_title(f"{label}: width vs v4 seed disagreement")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig5_width_vs_disagreement_{label}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ["easy/common", "medium", "rare"]
    data = []
    for _, predicate in RARITY_GROUP_BINS:
        mask = np.array([predicate(v) for v in rare])
        data.append(width[mask])
    ax.boxplot(data, tick_labels=groups, showfliers=False)
    ax.set_ylabel("width80")
    ax.set_title(f"{label}: width distribution by rarity group")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig6_width_by_rarity_group_{label}.png", dpi=120)
    plt.close(fig)


def run(map_path: Path, force: bool = False) -> dict[str, Any]:
    run_map = _load_map(map_path)
    fig_dir = RESULTS_ROOT / "figures"
    (RESULTS_ROOT / "seed0").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # master features + canonical v4 seed-0 predictions
    master = pd.read_csv(MASTER_TABLE)
    v4_meta = _run_meta(V4_CANONICAL_RUNS[0])

    variant_order = ["v4_none", "median_only", "lambda010", "lambda025", "lambda050"]
    summary: dict[str, Any] = {
        "stage": "compact-v5 multi-quantile",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v4_canonical_seed0": v4_meta,
        "runs": {},
        "variants": {},
        "tables": {},
    }
    frames: dict[str, pd.DataFrame] = {}
    for variant in variant_order:
        if variant not in run_map.get("seed0", {}):
            continue
        run_id = run_map["seed0"][variant]
        quantile = variant.startswith("lambda")
        frame = _variant_frame(run_id, quantile)
        frame = _merge_master(frame)
        # canonical v4 seed-0 per-molecule abs error for paired subgroup views
        frame["v4_abs_error"] = np.abs(
            frame["target"].to_numpy() - frame["prediction_seed0"].to_numpy()
        )
        frames[variant] = frame
        meta = _run_meta(run_id)
        summary["runs"][variant] = meta
        # per-molecule CSV (valid outputs + rarity + subgroup)
        csv_cols = [
            "molecule_id",
            "target",
            "q10",
            "q50",
            "q90",
            "width80",
            "lower_width",
            "upper_width",
            "abs_error",
            "signed_residual",
            "rare_le1_ratio",
            "rare_le2_ratio",
            "rare_le5_ratio",
            "rare_le10_ratio",
            "oov_patch_ratio",
            "min_train_frequency",
            "mean_log1p_train_frequency",
            "subgroup",
        ]
        csv_cols = [c for c in csv_cols if c in frame.columns]
        frame[csv_cols].to_csv(
            RESULTS_ROOT / "seed0" / f"validation_predictions_{variant}.csv",
            index=False,
        )
        entry: dict[str, Any] = {"meta": meta}
        if quantile:
            entry["diagnostics"] = _quantile_diagnostics(frame, variant)
            entry["subgroup_table"] = _subgroup_table(frame)
            entry["rarity_group_table"] = _rarity_group_table(frame)
            top = _top_widest(frame)
            top.to_csv(
                RESULTS_ROOT / "seed0" / f"top_widest_{variant}.csv", index=False
            )
            entry["top_widest_ids"] = top["molecule_id"].tolist()
            _figures(frame, variant, fig_dir)
        summary["variants"][variant] = entry

    # frozen difficulty predictor diagnostic on validation width
    try:
        pred_summary, model, scaler, feature_cols = _oof_static_predictor_summary()
        frozen_valid = _frozen_difficulty_on_validation(model, scaler, feature_cols)
        pred_summary["validation_n"] = int(len(frozen_valid))
        for variant, frame in frames.items():
            if not variant.startswith("lambda"):
                continue
            width = frame["width80"].to_numpy()
            pred_summary[f"spearman_width_pred_{variant}"] = _spearman(width, frozen_valid)
            pred_summary[f"spearman_abserr_pred_{variant}"] = _spearman(
                frozen_valid, frame["abs_error"].to_numpy()
            )
        summary["oof_difficulty_predictor"] = pred_summary
    except Exception as exc:  # diagnostic only
        summary["oof_difficulty_predictor"] = {"error": str(exc)}

    # primary table (all variants incl. non-quantile)
    primary_rows: list[dict[str, Any]] = []
    v4_best = v4_meta["best_mae"]
    for variant in variant_order:
        if variant not in summary["variants"]:
            continue
        entry = summary["variants"][variant]
        meta = entry["meta"]
        row = {
            "variant": variant,
            "run_id": meta["run_id"],
            "params": meta["parameters"],
            "quantile_mode": meta["quantile_mode"],
            "lambda": meta["quantile_lambda"],
            "valid_q50_mae": meta["best_mae"],
            "delta_vs_v4": float(v4_best - meta["best_mae"]),
        }
        if "diagnostics" in entry:
            d = entry["diagnostics"]
            row["width_error_spearman"] = d["spearman_width_abs_error"]
            row["central_80_coverage"] = d["central_80_coverage"]
            row["coverage_q10"] = d["coverage_q10"]
            row["coverage_q90"] = d["coverage_q90"]
        primary_rows.append(row)
    summary["tables"]["primary"] = primary_rows

    # --- multi-seed confirmation + pre-registered decision classification ---
    selected_variant = run_map.get("selected_variant")
    if selected_variant and "multiseed" in run_map:
        multiseed = _multiseed_table(run_map["multiseed"])
        summary["tables"]["multiseed"] = multiseed
        selected_entry = summary["variants"].get(selected_variant, {})
        # seed0 stage-1 classification (point + width, single seed)
        s0_point = 0.0
        s0_width = float("nan")
        if selected_entry:
            s0_point = float(
                v4_best - selected_entry["meta"]["best_mae"]
            )
            s0_width = selected_entry.get("diagnostics", {}).get(
                "spearman_width_abs_error", float("nan")
            )
        stage1_case = classify_stage1(s0_point, s0_width)
        final_case = classify_case(
            multiseed["paired_mean_delta"],
            multiseed["seeds_v5_better"],
            multiseed["n_seeds"],
            multiseed["width_spearman_mean"],
            multiseed["seeds_width_positive"],
        )
        summary["decision"] = {
            "selected_variant": selected_variant,
            "selected_lambda": summary["variants"][selected_variant]["meta"][
                "quantile_lambda"
            ],
            "stage1_seed0_point_improvement": s0_point,
            "stage1_seed0_width_error_spearman": s0_width,
            "stage1_case": stage1_case,
            "multiseed_point_band": final_case["point_prediction_band"],
            "multiseed_uncertainty_band": final_case["uncertainty_band"],
            "final_case": final_case["case"],
            "multiseed_paired_mean_delta": multiseed["paired_mean_delta"],
            "multiseed_seeds_v5_better": multiseed["seeds_v5_better"],
            "multiseed_width_spearman_mean": multiseed["width_spearman_mean"],
        }
    if "test" in run_map:
        summary["tables"]["benchmark_test"] = _test_table(run_map["test"])

    # --- materialise the pre-registered tables as CSVs ---
    seed0_dir = RESULTS_ROOT / "seed0"
    primary_df = pd.DataFrame(summary["tables"].get("primary", []))
    if len(primary_df):
        primary_df.to_csv(seed0_dir / "primary_table.csv", index=False)
    if "multiseed" in summary["tables"]:
        pd.DataFrame(summary["tables"]["multiseed"]["rows"]).to_csv(
            seed0_dir / "multiseed_table.csv", index=False
        )
    if "benchmark_test" in summary["tables"]:
        pd.DataFrame(summary["tables"]["benchmark_test"]["rows"]).to_csv(
            seed0_dir / "benchmark_test_table.csv", index=False
        )
    for variant, entry in summary["variants"].items():
        if "diagnostics" not in entry:
            continue
        d = entry["diagnostics"]
        if d.get("width_quintiles"):
            pd.DataFrame(d["width_quintiles"]).to_csv(
                seed0_dir / f"width_quintile_table_{variant}.csv", index=False
            )
        if d.get("rarity_bins"):
            pd.DataFrame(d["rarity_bins"]).to_csv(
                seed0_dir / f"rarity_bin_table_{variant}.csv", index=False
            )
        if entry.get("subgroup_table"):
            pd.DataFrame(entry["subgroup_table"]).to_csv(
                seed0_dir / f"subgroup_table_{variant}.csv", index=False
            )
        if entry.get("rarity_group_table"):
            pd.DataFrame(entry["rarity_group_table"]).to_csv(
                seed0_dir / f"rarity_group_table_{variant}.csv", index=False
            )
    _write_json(RESULTS_ROOT / "stage1_seed0_summary.json", summary)
    elapsed = time.perf_counter() - started
    summary["seconds"] = float(elapsed)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_map", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.run_map, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
