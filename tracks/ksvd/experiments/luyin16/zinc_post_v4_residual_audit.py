"""Post-v4 residual audit of compact-v4-hinge on ZINC (diagnosis only).

Stage mandate (see ``notes/post_v4_residual_audit.md``): v4-hinge is frozen as
the new benchmark baseline (4 seeds); this stage does **not** modify the
architecture, topology features, hinge basis, loss, optimizer, LR, epochs or
the graph head, does **not** train any model, and does **not** touch the
official test split (test stays frozen at the existing benchmark numbers).
Everything is computed on the validation split (plus pre-registered frozen
definitions from the earlier audits).

Questions answered: after the global-closure (long-cycle) residual mechanism
was largely removed by v4, does a *second-layer*, cross-seed-stable residual
signal appear inside the ordinary bulk (Group A)?  Candidate signal families:

1. ZINC target component attribution (z_logP / z_SA / z_cycle oracles) --
   ATTRIBUTION ONLY, never model features;
2. patch rarity / OOV statistics;
3. graph structural complexity (size, diameter, branching, cycle rank);
4. simple atom/bond composition (PyG node/edge types only, no RDKit
   descriptors);
5. exact patch-pair co-occurrence + distance-bucket-conditioned pairs
   (deterministic hashing, frozen definitions from the Information Gap Audit);
6. continuous pair interaction of frozen v4 patch states (h_i x h_j);
7. representation-space difficulty statistics from the frozen v4 checkpoints.

Leakage rules (hard):

* residual = y - y_hat on the official validation split from the four frozen
  validation-selected v4-hinge checkpoints (predictions recorded in the
  promoted runs; targets verified bit-consistent against the long-cycle audit
  CSV);
* no fit on test; test is never loaded by this module;
* v4 OOF train residuals would require re-training the v4 protocol 5-fold,
  which is forbidden in this stage -- fitted residual predictors are
  therefore restricted to (a) 1D target-component oracle fits on
  train-available data only where stated, and (b) validation-internal CV
  diagnostics, explicitly labelled VALIDATION-INTERNAL (descriptive; not a
  candidate model, not OOF-train evidence);
* subgroup definitions (A/B/C) are frozen label-excess definitions from the
  long-cycle audit; thresholds are never re-tuned on current residuals;
* target components are ORACLE / target-definition attribution probes only
  and must never be used as model features.

Stages (each idempotent; ``--force`` to rebuild):

    residuals    per-molecule validation residuals of the 4 frozen seeds
    master       master per-molecule table (groups + components + features)
    attribution  SA/logP/cycle component attribution (corr + quantile bins)
    probes       Group A rarity/complexity/composition correlation probes
    pairs        Group A pair/relation exact-hash probes (valid-internal CV)
    states       frozen-checkpoint state extraction (patch/pair/topology)
    state_probes Group A continuous-interaction + representation-state probes
    cross        target-component x structural cross analysis (partial)
    summary      group error distributions, top-30, ranking, probe summary
    decision     Q1-Q14 + final decision record
    figures      figures 1-6
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats as scipy_stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_information_gap_audit import (
    _hashed_count_features,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_ROOT = REPO_ROOT / "tracks/ksvd/results/post_v4_residual_audit"
CACHE_DIR = AUDIT_ROOT / "cache"
FIG_DIR = AUDIT_ROOT / "figures"
RUNS_ROOT = REPO_ROOT / "tracks/ksvd/runs"

LONG_CYCLE_ROOT = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit"
INFO_GAP_ROOT = REPO_ROOT / "tracks/ksvd/results/information_gap_audit"

LABEL_CSV = LONG_CYCLE_ROOT / "label_effective_cycle.csv"
REFINE_JSON = LONG_CYCLE_ROOT / "stage_refine.json"
VERIFY_JSON = LONG_CYCLE_ROOT / "stage_verify.json"
INFO_GAP_FEATURE_CSV = INFO_GAP_ROOT / "compact_v2_validation_information_gap_audit.csv"
INFO_GAP_HASHED_NPZ = INFO_GAP_ROOT / "hashed_features.npz"
INFO_GAP_RECORD_CACHE = INFO_GAP_ROOT / "cache"

V4_RUN_IDS = {
    0: "20260909-194445-182c7021",
    1: "20260909-200320-34b347bf",
    2: "20260909-201918-45fbe48d",
    3: "20260909-203516-04e62a28",
}
V4_EXPECTED_VALID = {  # best valid MAE per seed (guard against run mix-ups)
    0: 0.17006561887910357,
    1: 0.163166509715,
    2: 0.174149189376,
    3: 0.170420902214,
}
V4_EXPECTED_EPOCH = {0: 53, 1: 48, 2: 56, 3: 60}

# Fitted label-decomposition constants (long-cycle audit stage refine;
# frozen provenance -- do not re-fit here).
COMMUNITY_LOGP_MEAN = 2.4570953396190123
FIT_SIGMA_LOGP = 1.4351122692340226
FIT_MU_SA = -3.192224472209149
FIT_SIGMA_SA = 0.8321043074224692
FIT_MU_CYCLE = -0.0001334074230764987
FIT_SIGMA_CYCLE = 0.28831626452532244

HASH_DIM = 2048  # frozen Information Gap Audit hash dimension
TOKEN_BASE = 8192
BUCKET_BASE = 8
SEEDS = (0, 1, 2, 3)
GROUP_LETTERS = ("A", "B", "C")


def run_path(run_id: str) -> Path:
    return RUNS_ROOT / run_id[:4] / run_id[4:6] / run_id[6:8] / run_id


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _stage_marker(stage: str) -> Path:
    return AUDIT_ROOT / f"stage_{stage}.json"


def _is_done(stage: str) -> bool:
    return _stage_marker(stage).exists()


def _mark_done(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["stage"] = stage
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_stage_marker(stage), result)
    return result


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(scipy_stats.spearmanr(x, y).statistic)


def _sign_consistency(values: Sequence[float], n_seeds: int = 4) -> str:
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 3:
        return "insufficient"
    positive = sum(1 for v in finite if v > 0)
    negative = sum(1 for v in finite if v < 0)
    if positive == len(finite) or negative == len(finite):
        return f"{len(finite)}/{n_seeds} same sign"
    if max(positive, negative) >= 3:
        return f"3/{n_seeds} same sign"
    return f"<=2/{n_seeds} unstable"


def _mean_finite(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _std_finite(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


# --------------------------------------------------------------------------
# stage: residuals (frozen 4-seed v4-hinge validation residuals)
# --------------------------------------------------------------------------

def load_run_result(run_id: str) -> dict[str, Any]:
    return json.loads(
        (run_path(run_id) / "artifacts" / "legacy_full_result.json").read_text(
            encoding="utf-8"
        )
    )


def load_label_valid() -> pd.DataFrame:
    frame = pd.read_csv(LABEL_CSV)
    frame = frame[frame["split"] == "valid"].sort_values("subset_index").reset_index(drop=True)
    frame["label_excess"] = (-frame["label_effective_cycle_snapped"]).round().clip(lower=0)
    frame["subgroup"] = np.select(
        [frame["label_excess"] == 0, frame["label_excess"] == 1],
        ["A", "B"],
        default="C",
    )
    return frame


def stage_residuals() -> dict[str, Any]:
    started = time.perf_counter()
    label_valid = load_label_valid()
    y_ref = label_valid["y_stored"].to_numpy(dtype=np.float64)
    rows: list[pd.DataFrame] = []
    checks: dict[str, Any] = {}
    for seed in SEEDS:
        run_id = V4_RUN_IDS[seed]
        result = load_run_result(run_id)
        valid = result["evaluation"]["valid"]
        targets = np.asarray(valid["targets"], dtype=np.float64)
        predictions = np.asarray(valid["predictions"], dtype=np.float64)
        if not np.allclose(targets, y_ref, atol=1e-9, rtol=1e-9):
            raise AssertionError(f"seed {seed}: valid targets do not match audit y_stored")
        mae = float(np.mean(np.abs(targets - predictions)))
        checks[seed] = {
            "run_id": run_id,
            "stored_best_mae": float(valid["best_mae"]),
            "stored_selected_epoch": int(valid["selected_epoch"]),
            "computed_predictions_mae": mae,
            "predictions_match_best_mae": abs(mae - float(valid["best_mae"])) < 1e-12,
            "expected_best_mae_match": abs(
                mae - V4_EXPECTED_VALID[seed]
            ) < 1e-9,
            "expected_epoch_match": int(valid["selected_epoch"])
            == V4_EXPECTED_EPOCH[seed],
        }
        if abs(mae - float(valid["best_mae"])) >= 1e-12:
            raise AssertionError(
                f"seed {seed}: stored valid predictions are not the best-epoch "
                f"checkpoint predictions (mae {mae} vs best {valid['best_mae']})"
            )
        frame = pd.DataFrame(
            {
                "molecule_id": label_valid["molecule_id"],
                "target": y_ref,
                f"prediction_seed{seed}": predictions,
                f"residual_seed{seed}": y_ref - predictions,
                f"abs_error_seed{seed}": np.abs(y_ref - predictions),
            }
        )
        rows.append(frame)
    table = rows[0]
    for frame in rows[1:]:
        table = table.merge(frame, on=["molecule_id", "target"], how="outer")
    pred_cols = [f"prediction_seed{s}" for s in SEEDS]
    resid_cols = [f"residual_seed{s}" for s in SEEDS]
    abs_cols = [f"abs_error_seed{s}" for s in SEEDS]
    table["mean_prediction"] = table[pred_cols].mean(axis=1)
    table["mean_residual"] = table[resid_cols].mean(axis=1)
    table["mean_abs_error"] = table[abs_cols].mean(axis=1)
    table["residual_std_across_seeds"] = table[resid_cols].std(axis=1, ddof=1)
    table["ensemble_prediction"] = table["mean_prediction"]
    table["ensemble_abs_error"] = (table["target"] - table["mean_prediction"]).abs()
    out_path = AUDIT_ROOT / "validation_per_molecule_residuals.csv"
    table.to_csv(out_path, index=False)
    summary = {
        "n": int(len(table)),
        "per_seed_mae": {
            int(s): float(np.mean(table[f"abs_error_seed{s}"].to_numpy()))
            for s in SEEDS
        },
        "ensemble_mae": float(table["ensemble_abs_error"].mean()),
        "mean_residual_ensemble": float(table["mean_residual"].mean()),
        "mean_residual_std_across_seeds": float(
            table["residual_std_across_seeds"].mean()
        ),
        "checks": checks,
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("residuals", summary)


# --------------------------------------------------------------------------
# stage: master (per-molecule master table on validation)
# --------------------------------------------------------------------------

def _component_columns() -> dict[str, Any]:
    refine = _read_json(REFINE_JSON)
    fitted = refine["constants_fitted"]
    assert abs(float(fitted["sigma_logP"]) - FIT_SIGMA_LOGP) < 1e-12
    assert abs(float(fitted["mu_SA"]) - FIT_MU_SA) < 1e-12
    assert abs(float(fitted["sigma_SA"]) - FIT_SIGMA_SA) < 1e-12
    return {
        "provenance": {
            "logP": (
                "VERIFIED (GVAE formula audit: RDKit MolLogP on canonical "
                "SMILES matched per molecule; 11981/12000 exact; community "
                "logP_mean verified to ~1e-5 rel. diff)"
            ),
            "SA": (
                "VERIFIED (GVAE formula audit: -sascorer on canonical SMILES "
                "matched per molecule; 11981/12000 exact)"
            ),
            "cycle": (
                "RECONSTRUCTED-VERIFIED (label-implied lattice snap of "
                "y - z_logP - z_SA; valid split 1000/1000 consistent with the "
                "GVAE-order replication; p95 snap error 0.0016)"
            ),
            "constants": {
                "z_logP": {
                    "mean": COMMUNITY_LOGP_MEAN,
                    "std": FIT_SIGMA_LOGP,
                    "note": "community logP mean (long-cycle audit stage verify), "
                    "refine-fitted sigma_logP (stage refine)",
                },
                "z_SA": {
                    "mean": FIT_MU_SA,
                    "std": FIT_SIGMA_SA,
                    "note": "refine-fitted mu_SA / sigma_SA on the cycle-free bulk",
                },
                "z_cycle": {
                    "mean": FIT_MU_CYCLE,
                    "std": FIT_SIGMA_CYCLE,
                    "note": "refine-fitted mu_cycle / sigma_cycle; "
                    "z_cycle = label_normalized_cycle_component column",
                },
            },
        }
    }


def stage_master() -> dict[str, Any]:
    started = time.perf_counter()
    residuals = pd.read_csv(AUDIT_ROOT / "validation_per_molecule_residuals.csv")
    label_valid = load_label_valid()
    features = pd.read_csv(INFO_GAP_FEATURE_CSV)
    if not (features["molecule_id"] == label_valid["molecule_id"]).all():
        raise AssertionError("info-gap feature rows are not aligned to label rows")

    table = residuals.merge(
        label_valid[
            [
                "molecule_id",
                "subset_index",
                "y_stored",
                "logP",
                "SA",
                "effective_cycle",
                "label_effective_cycle_snapped",
                "snap_error",
                "mismatch_vs_gvae_order",
                "label_normalized_cycle_component",
                "label_excess",
                "subgroup",
            ]
        ],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )
    table["z_logP"] = (table["logP"] - COMMUNITY_LOGP_MEAN) / FIT_SIGMA_LOGP
    table["z_SA"] = (table["SA"] - FIT_MU_SA) / FIT_SIGMA_SA
    table["z_cycle"] = table["label_normalized_cycle_component"]
    table["y_chem"] = table["z_logP"] + table["z_SA"]
    table["component_reconstruction_error"] = (
        table["target"]
        - (table["z_logP"] + table["z_SA"] + table["z_cycle"])
    ).abs()
    feature_cols = [
        c
        for c in features.columns
        if c not in ("molecule_id", "target", "prediction", "signed_residual", "absolute_error")
    ]
    table = table.merge(
        features[["molecule_id"] + feature_cols],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )
    out_path = AUDIT_ROOT / "validation_master_table.csv"
    table.to_csv(out_path, index=False)
    recon = table["component_reconstruction_error"].to_numpy()
    group_a = table[table["subgroup"] == "A"]
    summary = {
        "n": int(len(table)),
        "group_counts": table["subgroup"].value_counts().to_dict(),
        "component_reconstruction_error": {
            "p50": float(np.median(recon)),
            "p95": float(np.quantile(recon, 0.95)),
            "max": float(recon.max()),
        },
        "component_reconstruction_error_groupA_max": float(
            group_a["component_reconstruction_error"].max()
        ),
        "valid_cycle_mismatch_vs_gvae_order": int(table["mismatch_vs_gvae_order"].sum()),
        "z_cycle_unique_values_groupA": sorted(group_a["z_cycle"].unique().tolist()),
        "feature_columns": len(feature_cols),
        "nans_in_features": int(table[feature_cols].isna().sum().sum()),
        "provenance": _component_columns()["provenance"],
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2)[:4000])
    return _mark_done("master", summary)


# --------------------------------------------------------------------------
# stage: attribution (SA/logP/cycle component attribution)
# --------------------------------------------------------------------------

def _correlation_row(
    table: pd.DataFrame,
    scope: str,
    component: str,
    target_kind: str,
) -> dict[str, Any]:
    x = table[component].to_numpy(dtype=np.float64)
    row: dict[str, Any] = {
        "scope": scope,
        "component": component,
        "target": target_kind,
        "n": int(len(table)),
    }
    for seed in SEEDS:
        y = table[f"{target_kind}_seed{seed}"].to_numpy(dtype=np.float64)
        row[f"pearson_seed{seed}"] = _pearson(x, y)
        row[f"spearman_seed{seed}"] = _spearman(x, y)
    y_mean = table[f"mean_{target_kind}"].to_numpy(dtype=np.float64)
    row["pearson_ensemble"] = _pearson(x, y_mean)
    row["spearman_ensemble"] = _spearman(x, y_mean)
    row["pearson_consistency"] = _sign_consistency(
        [row[f"pearson_seed{s}"] for s in SEEDS]
    )
    row["spearman_consistency"] = _sign_consistency(
        [row[f"spearman_seed{s}"] for s in SEEDS]
    )
    row["pearson_mean"] = _mean_finite([row[f"pearson_seed{s}"] for s in SEEDS])
    row["spearman_mean"] = _mean_finite([row[f"spearman_seed{s}"] for s in SEEDS])
    row["pearson_std"] = _std_finite([row[f"pearson_seed{s}"] for s in SEEDS])
    row["spearman_std"] = _std_finite([row[f"spearman_seed{s}"] for s in SEEDS])
    return row


def stage_attribution() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    rows: list[dict[str, Any]] = []
    for scope, mask in (
        ("all_valid", np.ones(len(table), dtype=bool)),
        ("group_A", (table["subgroup"] == "A").to_numpy()),
        ("group_B", (table["subgroup"] == "B").to_numpy()),
        ("group_C", (table["subgroup"] == "C").to_numpy()),
    ):
        sub = table[mask].reset_index(drop=True)
        for component in ("z_logP", "z_SA", "z_cycle"):
            if component == "z_cycle" and scope != "all_valid":
                # z_cycle is (nearly) constant inside A/B/C by construction
                continue
            for target_kind in ("residual", "abs_error"):
                rows.append(_correlation_row(sub, scope, component, target_kind))
    corr_table = pd.DataFrame(rows)
    corr_path = AUDIT_ROOT / "component_attribution_correlations.csv"
    corr_table.to_csv(corr_path, index=False)

    # --- quantile-bin tables (Group A focus) ---
    group_a = table[table["subgroup"] == "A"].reset_index(drop=True)
    bin_rows: list[dict[str, Any]] = []
    for component in ("z_logP", "z_SA"):
        try:
            bins = pd.qcut(group_a[component], 5, duplicates="drop")
        except ValueError:
            continue
        for bin_index, indices in bins.groupby(bins, observed=True).indices.items():
            sub = group_a.iloc[indices]
            row: dict[str, Any] = {
                "scope": "group_A",
                "component": component,
                "bin": str(bin_index),
                "bin_center": float(
                    np.mean(
                        group_a.iloc[indices][component].to_numpy(dtype=np.float64)
                    )
                ),
                "n": int(len(sub)),
                "mean_signed_residual_ensemble": float(sub["mean_residual"].mean()),
                "mae_ensemble": float(
                    np.abs(
                        sub["target"].to_numpy()
                        - sub["mean_prediction"].to_numpy()
                    ).mean()
                ),
                "residual_std_ensemble": float(sub["mean_residual"].std(ddof=1)),
                "mean_abs_error_across_seeds": float(sub["mean_abs_error"].mean()),
                "residual_std_across_seeds_mean": float(
                    sub["residual_std_across_seeds"].mean()
                ),
                "target_mean": float(sub["target"].mean()),
                "mean_prediction_ensemble": float(sub["mean_prediction"].mean()),
            }
            for seed in SEEDS:
                row[f"mean_signed_residual_seed{seed}"] = float(
                    sub[f"residual_seed{seed}"].mean()
                )
                row[f"mae_seed{seed}"] = float(sub[f"abs_error_seed{seed}"].mean())
            bin_rows.append(row)
    bin_table = pd.DataFrame(bin_rows)
    bin_path = AUDIT_ROOT / "component_quantile_bins.csv"
    bin_table.to_csv(bin_path, index=False)

    # per-seed MAE by subgroup (cross-check against the multi-seed note)
    subgroup_mae: dict[str, dict[str, float]] = {}
    for letter in GROUP_LETTERS:
        mask = table["subgroup"] == letter
        subgroup_mae[letter] = {
            str(s): float(table.loc[mask, f"abs_error_seed{s}"].mean())
            for s in SEEDS
        }
        subgroup_mae[letter]["ensemble"] = float(
            np.abs(
                table.loc[mask, "target"].to_numpy()
                - table.loc[mask, "mean_prediction"].to_numpy()
            ).mean()
        )
        subgroup_mae[letter]["n"] = int(mask.sum())

    summary = {
        "group_a_mae_per_seed": subgroup_mae["A"],
        "subgroup_mae": subgroup_mae,
        "correlation_table": str(corr_path),
        "quantile_bin_table": str(bin_path),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("attribution", summary)


# --------------------------------------------------------------------------
# stage: probes (Group A rarity / complexity / composition correlations and
# 1D valid-internal-CV diagnostics)
# --------------------------------------------------------------------------

RARITY_FEATURES = [
    "oov_patch_ratio",
    "rare_le1_ratio",
    "rare_le2_ratio",
    "rare_le5_ratio",
    "rare_le10_ratio",
    "min_train_frequency",
    "mean_log1p_train_frequency",
    "mean_train_frequency",
    "mean_inverse_of_1p_frequency",
    "num_unique_patch_tokens",
    "unique_patch_ratio",
    "num_oov_patch_tokens",
]

COMPLEXITY_FEATURES = [
    "num_nodes",
    "num_edges",
    "num_pairs",
    "edges_per_node",
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
    "frac_dist_gt_3",
    "num_patches",
]

COMPOSITION_FEATURES = [f"node_type_{i}_fraction" for i in range(28)] + [
    f"edge_type_{i}_fraction" for i in range(4)
]

PROBE_BLOCKS = [
    ("rarity_oov", RARITY_FEATURES),
    ("graph_complexity", COMPLEXITY_FEATURES),
    ("atom_bond_composition", COMPOSITION_FEATURES),
]

# 1D valid-internal-CV probe roster (features that represent each block +
# the two target-component oracles; method rows are labelled VALID-INTERNAL).
CV_PROBE_ROSTER = [
    ("oracle_z_SA", "z_SA", "isotonic"),
    ("oracle_z_SA", "z_SA", "linear"),
    ("oracle_z_logP", "z_logP", "isotonic"),
    ("oracle_z_logP", "z_logP", "linear"),
    ("rarity_oov", "oov_patch_ratio", "linear"),
    ("rarity_oov", "rare_le5_ratio", "isotonic"),
    ("rarity_oov", "rare_le10_ratio", "isotonic"),
    ("rarity_oov", "min_train_frequency", "linear"),
    ("rarity_oov", "mean_log1p_train_frequency", "linear"),
    ("graph_complexity", "branching_node_ratio", "linear"),
    ("graph_complexity", "num_branching_nodes", "linear"),
    ("graph_complexity", "diameter", "isotonic"),
    ("graph_complexity", "cycle_rank", "isotonic"),
    ("graph_complexity", "num_nodes", "linear"),
    ("graph_complexity", "mean_shortest_path_distance", "linear"),
    ("atom_bond_composition", "node_type_8_fraction", "isotonic"),
    ("atom_bond_composition", "node_type_8_fraction", "linear"),
    ("atom_bond_composition", "node_type_10_fraction", "linear"),
    ("atom_bond_composition", "edge_type_2_fraction", "linear"),
    ("atom_bond_composition", "node_type_1_fraction", "linear"),
]


# atom-type index meaning (graphdeeplearning ZINC atom_dict.word2idx, raw
# atom_type stored in data.x; VERIFIED against raw/atom_dict.pickle):
# 0 C, 1 O, 2 N, 3 F, 4 C H1, 5 S, 6 Cl, 7 O -, 8 N H1 +, 9 Br, 10 N H3 +,
# 11 N H2 +, 12 N +, 13 N -, 14 S -, 15 I, 16 P, ... (protonation/charge
# variants of the element vocabulary).


def _correlation_probe_rows(
    table: pd.DataFrame, block: str, feature: str, scope: str = "group_A"
) -> dict[str, Any]:
    x = table[feature].to_numpy(dtype=np.float64)
    row: dict[str, Any] = {
        "scope": scope,
        "block": block,
        "feature": feature,
        "n": int(len(table)),
    }
    for target_kind in ("residual", "abs_error"):
        for seed in SEEDS:
            y = table[f"{target_kind}_seed{seed}"].to_numpy(dtype=np.float64)
            row[f"{target_kind}_pearson_seed{seed}"] = _pearson(x, y)
            row[f"{target_kind}_spearman_seed{seed}"] = _spearman(x, y)
        y_mean = table[f"mean_{target_kind}"].to_numpy(dtype=np.float64)
        row[f"{target_kind}_pearson_ensemble"] = _pearson(x, y_mean)
        row[f"{target_kind}_spearman_ensemble"] = _spearman(x, y_mean)
        row[f"{target_kind}_spearman_consistency"] = _sign_consistency(
            [row[f"{target_kind}_spearman_seed{s}"] for s in SEEDS]
        )
        row[f"{target_kind}_pearson_consistency"] = _sign_consistency(
            [row[f"{target_kind}_pearson_seed{s}"] for s in SEEDS]
        )
        row[f"{target_kind}_pearson_mean"] = _mean_finite(
            [row[f"{target_kind}_pearson_seed{s}"] for s in SEEDS]
        )
        row[f"{target_kind}_spearman_mean"] = _mean_finite(
            [row[f"{target_kind}_spearman_seed{s}"] for s in SEEDS]
        )
    return row


def _internal_cv_1d_probe(
    x: np.ndarray,
    r: np.ndarray,
    method: str,
    folds: int = 5,
) -> dict[str, Any]:
    """Validation-internal K-fold 1D residual probe (descriptive).

    Explicitly NOT an OOF-train fit: v4 train OOF residuals would require
    re-training the protocol (forbidden in this stage).  Reported numbers
    describe how much of the *validation* residual a 1D function of the
    feature can explain with internal CV; they are diagnostics on the
    frozen validation predictions, not candidate-model evidence.
    """
    kf = KFold(n_splits=int(folds), shuffle=False)
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    r_hat = np.full_like(r, np.nan)
    ok = np.isfinite(x) & np.isfinite(r)
    baseline_mae = float(np.mean(np.abs(r[ok])))
    for train_idx, test_idx in kf.split(np.arange(len(x))[ok]):
        idx = np.flatnonzero(ok)
        tr, te = idx[train_idx], idx[test_idx]
        x_tr, r_tr = x[tr], r[tr]
        x_te = x[te]
        if method == "linear":
            scaler = StandardScaler().fit(x_tr.reshape(-1, 1))
            model = Ridge(alpha=1.0)
            model.fit(scaler.transform(x_tr.reshape(-1, 1)), r_tr)
            prediction = model.predict(scaler.transform(x_te.reshape(-1, 1)))
        elif method == "isotonic":
            if np.unique(x_tr).size < 3:
                prediction = np.zeros(len(te))
            else:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_tr, r_tr)
                prediction = iso.predict(x_te)
        else:
            raise ValueError(f"unknown method {method!r}")
        r_hat[te] = prediction
    valid_mask = np.isfinite(r_hat)
    corrected_mae = float(mean_absolute_error(r[valid_mask], r_hat[valid_mask]))
    residual_r2 = float(
        r2_score(r[valid_mask], r_hat[valid_mask])
        if np.std(r[valid_mask]) > 0
        else float("nan")
    )
    return {
        "method": method,
        "fit_scope": "VALID-INTERNAL-CV (descriptive; no OOF-train residual under freeze)",
        "n": int(valid_mask.sum()),
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "delta_mae": float(baseline_mae - corrected_mae),
        "residual_r2": residual_r2,
    }


def stage_probes() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    group_a = table[table["subgroup"] == "A"].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for block, features in PROBE_BLOCKS:
        for feature in features:
            if feature not in group_a.columns:
                continue
            series = group_a[feature]
            if series.isna().all() or series.nunique() < 5:
                continue
            rows.append(_correlation_probe_rows(group_a, block, feature))
    corr_table = pd.DataFrame(rows)
    corr_path = AUDIT_ROOT / "probe_correlations_groupA.csv"
    corr_table.to_csv(corr_path, index=False)

    # bias-vs-variance binning for the leading features (per-block top by
    # mean |spearman| on the ensemble abs error)
    bin_rows: list[dict[str, Any]] = []
    for block in ("rarity_oov", "graph_complexity"):
        block_rows = corr_table[
            (corr_table["block"] == block)
            & corr_table["abs_error_spearman_ensemble"].notna()
        ]
        top = (
            block_rows.reindex(
                block_rows["abs_error_spearman_ensemble"].abs().sort_values().index
            )
            .tail(3)["feature"]
            .tolist()
        )
        for feature in top:
            try:
                bins = pd.qcut(group_a[feature], 5, duplicates="drop")
            except ValueError:
                continue
            for bin_label in bins.cat.categories:
                indices = np.flatnonzero((bins == bin_label).to_numpy())
                sub = group_a.iloc[indices]
                row: dict[str, Any] = {
                    "scope": "group_A",
                    "block": block,
                    "feature": feature,
                    "bin": str(bin_label),
                    "n": int(len(sub)),
                    "mean_signed_residual_ensemble": float(sub["mean_residual"].mean()),
                    "mae_ensemble": float(
                        np.abs(
                            sub["target"].to_numpy()
                            - sub["mean_prediction"].to_numpy()
                        ).mean()
                    ),
                    "residual_std_ensemble": float(sub["mean_residual"].std(ddof=1)),
                    "mean_abs_error_across_seeds": float(sub["mean_abs_error"].mean()),
                }
                for seed in SEEDS:
                    row[f"mean_signed_residual_seed{seed}"] = float(
                        sub[f"residual_seed{seed}"].mean()
                    )
                bin_rows.append(row)
    bin_table = pd.DataFrame(bin_rows)
    bin_path = AUDIT_ROOT / "probe_feature_bins_groupA.csv"
    bin_table.to_csv(bin_path, index=False)

    # valid-internal-CV 1D probes (descriptive)
    r_target = group_a["mean_residual"].to_numpy(dtype=np.float64)
    y_base = group_a["mean_prediction"].to_numpy(dtype=np.float64)
    cv_rows: list[dict[str, Any]] = []
    for block, feature, method in CV_PROBE_ROSTER:
        x = group_a[feature].to_numpy(dtype=np.float64)
        result = _internal_cv_1d_probe(x, r_target, method)
        cv_rows.append(
            {
                "probe": feature,
                "block": block,
                **result,
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_path = AUDIT_ROOT / "probe_1d_cv_groupA.csv"
    cv_table.to_csv(cv_path, index=False)

    summary = {
        "n_groupA": int(len(group_a)),
        "correlation_table": str(corr_path),
        "bin_table": str(bin_path),
        "cv_table": str(cv_path),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("probes", summary)


# --------------------------------------------------------------------------
# stage: pairs (Group A exact pair / relation hash probes; frozen definitions)
# --------------------------------------------------------------------------

def stage_pairs() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    group_a = table[table["subgroup"] == "A"].reset_index(drop=True)
    group_a_idx = group_a.index.to_numpy()  # row positions in the master table

    # frozen deterministic hashing, same dimension/canonicalisation as the
    # Information Gap Audit; arrays verified bit-identical on rebuild below.
    if not INFO_GAP_HASHED_NPZ.exists():
        raise FileNotFoundError(INFO_GAP_HASHED_NPZ)
    hashed = _load_npz(INFO_GAP_HASHED_NPZ)

    # rebuild one array with the frozen pipeline and require bit-equality as a
    # provenance gate (rows are aligned to the info-gap feature CSV order).
    records = None

    def _load_records() -> list[Any]:
        nonlocal records
        if records is None:
            with gzip.open(
                INFO_GAP_RECORD_CACHE / "graph_records_valid.pkl.gz", "rb"
            ) as handle:
                records = list(pickle.load(handle))
        return records

    valid_records = _load_records()
    with gzip.open(INFO_GAP_RECORD_CACHE / "graph_records_train.pkl.gz", "rb") as handle:
        train_records = list(pickle.load(handle))
    vocab = zpp._fit_vocabulary(train_records, "typed_certificate", 8192, 1)
    rebuilt = _hashed_count_features(
        valid_records, vocab, mode="pair", dim=HASH_DIM, transform="log1p"
    )
    if not np.array_equal(rebuilt, hashed["pair_valid_log"]):
        raise AssertionError("hashed pair features no longer bit-match the frozen cache")

    def _restrict(key: str) -> np.ndarray:
        return hashed[key][group_a_idx]

    r_target = group_a["mean_residual"].to_numpy(dtype=np.float64)
    y_base = group_a["mean_prediction"].to_numpy(dtype=np.float64)

    def _ridge_cv(
        X: np.ndarray, r: np.ndarray, name: str, block: str
    ) -> dict[str, Any]:
        kf = KFold(n_splits=5, shuffle=False)
        alpha_grid = np.logspace(-4, 6, 25)
        r_hat = np.full(len(r), np.nan)
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr = scaler.transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            best_alpha, _ = _select_alpha_svd_internal(X_tr, r[train_idx], alpha_grid)
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_tr, r[train_idx])
            r_hat[test_idx] = ridge.predict(X_te)
        valid_mask = np.isfinite(r_hat)
        corrected_mae = float(mean_absolute_error(r[valid_mask], r_hat[valid_mask]))
        baseline_mae = float(np.mean(np.abs(r[valid_mask])))
        residual_r2 = float(
            r2_score(r[valid_mask], r_hat[valid_mask])
            if np.std(r[valid_mask]) > 0
            else float("nan")
        )
        return {
            "probe": name,
            "block": block,
            "feature_dim": int(X.shape[1]),
            "fit_scope": "VALID-INTERNAL-CV (descriptive; no OOF-train residual under freeze)",
            "n": int(valid_mask.sum()),
            "baseline_mae": baseline_mae,
            "corrected_mae": corrected_mae,
            "delta_mae": float(baseline_mae - corrected_mae),
            "residual_r2": residual_r2,
        }

    rows = []
    for key, name, block in [
        ("pair_valid_log", "pair_hash_log1p", "pair_exact"),
        ("pair_valid_presence", "pair_hash_presence", "pair_exact"),
        ("rel_valid_log", "pair_relation_hash_log1p", "pair_relation"),
        ("rel_valid_presence", "pair_relation_hash_presence", "pair_relation"),
        ("pair_oov_valid_log", "pair_oov_exact_hash_log1p", "pair_oov_oracle"),
        ("unary_valid", "unary_token_hash_log1p", "unary"),
    ]:
        X = _restrict(key).astype(np.float64)
        rows.append(_ridge_cv(X, r_target, name, block))
    # combination of the two exact channels
    X = np.concatenate(
        [_restrict("pair_valid_log"), _restrict("rel_valid_log")], axis=1
    )
    rows.append(_ridge_cv(X, r_target, "pair_log+rel_log", "pair_combined"))
    cv_table = pd.DataFrame(rows)
    cv_path = AUDIT_ROOT / "pair_probe_cv_groupA.csv"
    cv_table.to_csv(cv_path, index=False)
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary = {
        "n_groupA": int(len(group_a)),
        "hash_dim": HASH_DIM,
        "provenance_note": (
            "frozen Information Gap Audit definitions: md5 hash dim 2048, "
            "TOKEN_BASE 8192, BUCKET_BASE 8, canonical (lo,hi) pair keys, "
            "log1p/presence transforms; bit-identity gate passed on rebuild"
        ),
        "cv_table": str(cv_path),
        "seconds": float(time.perf_counter() - started),
    }
    return _mark_done("pairs", summary)


def _select_alpha_svd_internal(
    X: np.ndarray, r: np.ndarray, alphas: np.ndarray, folds: int = 5
) -> tuple[float, float]:
    """Train-fold-internal K-fold MAE alpha selection (SVD form of ridge)."""
    kf = KFold(n_splits=int(folds), shuffle=False)
    per_alpha = np.zeros(len(alphas))
    for train_idx, valid_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[valid_idx]
        r_tr, r_va = r[train_idx], r[valid_idx]
        U, S, Vt = np.linalg.svd(X_tr, full_matrices=False)
        Utr = U.T @ r_tr
        for i, alpha in enumerate(alphas):
            coef = Vt.T @ (S / (S * S + float(alpha)) * Utr)
            prediction = X_va @ coef
            per_alpha[i] += mean_absolute_error(r_va, prediction)
    per_alpha /= float(folds)
    best = int(np.argmin(per_alpha))
    return float(alphas[best]), float(per_alpha[best])


# --------------------------------------------------------------------------
# stage: states (frozen v4-hinge checkpoints -> per-molecule latent states)
# --------------------------------------------------------------------------

ZINC_ROOT = REPO_ROOT / "data/ZINC"


def _extract_v4_records(force: bool = False) -> tuple[list[Any], list[Any]]:
    """Extract (once) train+valid GraphRecords with hinge topology features."""
    train_path = CACHE_DIR / "v4_records_train.pkl.gz"
    valid_path = CACHE_DIR / "v4_records_valid.pkl.gz"
    if not force and train_path.exists() and valid_path.exists():
        with gzip.open(train_path, "rb") as handle:
            train_records = list(pickle.load(handle))
        with gzip.open(valid_path, "rb") as handle:
            valid_records = list(pickle.load(handle))
        return train_records, valid_records
    train_ds = _load_zinc(ZINC_ROOT, "train")
    valid_ds = _load_zinc(ZINC_ROOT, "val")
    topo_train = ztopo.matrices_for_split("train", train_ds, "hinge")[0]
    topo_valid = ztopo.matrices_for_split("valid", valid_ds, "hinge")[0]
    certificate_cache: dict[bytes, bytes] = {}
    train_records, _ = zpp._extract_split(
        train_ds,
        "train",
        certificate_cache,
        topology_mode="hinge",
        topology_matrix=topo_train,
    )
    valid_records, _ = zpp._extract_split(
        valid_ds,
        "valid",
        certificate_cache,
        topology_mode="hinge",
        topology_matrix=topo_valid,
    )
    for path, records in ((train_path, train_records), (valid_path, valid_records)):
        with gzip.open(path, "wb") as handle:
            pickle.dump(list(records), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return train_records, valid_records


def _build_v4_model(
    config: Mapping[str, Any], audit: Mapping[str, Any]
) -> torch.nn.Module:
    """Construct the v4-hinge model exactly as ``_train_phase`` does (the
    constructor argument mirror is the frozen code path; the state-dict load
    and the bit-exact prediction gate verify it)."""
    model_config = config["model"]
    model = zpp.PatchPathModel(
        int(audit["typed_vocabulary_size_with_oov"]),
        int(audit["parent_vocabulary_size_with_oov"]),
        patch_hidden=int(model_config.get("patch_hidden", zpp.PATCH_HIDDEN)),
        pair_hidden=int(model_config.get("pair_hidden", zpp.PAIR_HIDDEN)),
        token_width=int(model_config.get("token_width", 32)),
        dropout=float(model_config.get("dropout", 0.05)),
        embedding_mode=str(model_config.get("embedding_mode", "full")),
        embedding_rank=int(model_config.get("embedding_rank", 16)),
        parent_embedding_rank=(
            None
            if model_config.get("parent_embedding_rank") is None
            else int(model_config["parent_embedding_rank"])
        ),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        readout=str(model_config.get("readout", "moments")),
        node_readout=model_config.get("node_readout"),
        pair_readout=model_config.get("pair_readout"),
        shell_width=int(zpp._shell_width_for_radius(zpp.PATCH_RADIUS)),
        context_width=0,
        direct_token_readout=False,
        center_context=bool(model_config.get("center_context", False)),
        center_context_hidden=(
            None
            if model_config.get("center_context_hidden") is None
            else int(model_config["center_context_hidden"])
        ),
        graph_head_hidden_0=(
            None
            if model_config.get("graph_head_hidden_0") is None
            else int(model_config["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model_config.get("graph_head_hidden_1") is None
            else int(model_config["graph_head_hidden_1"])
        ),
        structural_context_mode=str(model_config.get("structural_context_mode", "none")),
        structural_context_fusion=str(
            model_config.get("structural_context_fusion", "condition")
        ),
        structural_context_dim=int(model_config.get("structural_context_dim", 8)),
        structural_context_embedding_rank=int(
            model_config.get("structural_context_embedding_rank", 1)
        ),
        structural_context_condition_rank=int(
            model_config.get("structural_context_condition_rank", 8)
        ),
        structural_context_vocabulary_size=int(
            model_config.get("structural_context_vocabulary_size", 2)
        ),
        topology_mode=str(model_config.get("topology_mode", "none")),
        topology_input_width=int(audit["topology"]["input_width"]),
        topology_hidden_dim=int(model_config.get("topology_hidden_dim", 16)),
        topology_out_dim=int(model_config.get("topology_out_dim", 8)),
    )
    return model


def _frozen_forward_with_states(
    model: torch.nn.Module,
    graphs: Sequence[Any],
    batch_size: int = 128,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[np.ndarray]]]:
    """Eval-mode forward with state capture (frozen model, no grad).

    Returns (predictions in loader order, per-molecule final patch states,
    per-batch graph-level captures dict)."""
    loader = DataLoader(list(graphs), batch_size=int(batch_size), shuffle=False)
    predictions: list[np.ndarray] = []
    patch_states_by_graph: list[np.ndarray] = []
    captures: dict[str, list[np.ndarray]] = {
        "patch_pre_update": [],
        "center_delta": [],
        "pair_value": [],
        "global_hidden": [],
        "topology_hidden": [],
        "unified": [],
        "prediction": [],
    }
    hook_stores: dict[str, list[np.ndarray]] = {k: [] for k in captures}

    def make_hook(key: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
            hook_stores[key].append(output.detach().cpu().numpy())

        return hook

    def center_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        hook_stores["patch_pre_update"].append(inputs[0].detach().cpu().numpy())
        hook_stores["center_delta"].append(output.detach().cpu().numpy())

    handles = [
        model.center_update.register_forward_hook(center_hook),
        model.pair_encoder.register_forward_hook(make_hook("pair_value")),
        model.global_encoder.register_forward_hook(make_hook("global_hidden")),
        model.topology_encoder.register_forward_hook(make_hook("topology_hidden")),
        model.head.register_forward_hook(make_hook("unified")),
    ]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device("cpu"))
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
            hook_stores["prediction"].append(pred)
            # split patch-level captures per molecule with batch.batch
            pre = hook_stores["patch_pre_update"][-1]
            delta = hook_stores["center_delta"][-1]
            final = pre[:, : model.patch_hidden] + delta
            batch_ids = batch.batch.cpu().numpy()
            boundaries = np.flatnonzero(np.diff(batch_ids)) + 1
            offsets = np.concatenate([[0], boundaries, [len(batch_ids)]])
            for graph_id in range(int(batch.num_graphs)):
                start, end = int(offsets[graph_id]), int(offsets[graph_id + 1])
                patch_states_by_graph.append(final[start:end])
    for handle in handles:
        handle.remove()
    return (
        np.concatenate(predictions).astype(np.float64),
        patch_states_by_graph,
        {key: np.concatenate(values) for key, values in hook_stores.items()},
    )


def _state_summary_row(
    patch_states: np.ndarray,
    pair_value: np.ndarray,
    global_hidden: np.ndarray,
    topology_hidden: np.ndarray,
    unified: np.ndarray,
) -> dict[str, float]:
    patch_norms = np.linalg.norm(patch_states, axis=1)
    pair_norms = np.linalg.norm(pair_value, axis=1)
    centroid = patch_states.mean(axis=0)
    return {
        "patch_norm_mean": float(patch_norms.mean()),
        "patch_norm_std": float(patch_norms.std()) if patch_norms.size else 0.0,
        "patch_absmax_mean": float(np.abs(patch_states).mean()),
        "patch_centroid_norm": float(np.linalg.norm(centroid)),
        "pair_norm_mean": float(pair_norms.mean()) if pair_norms.size else 0.0,
        "pair_norm_std": float(pair_norms.std()) if pair_norms.size else 0.0,
        "global_encoder_norm": float(np.linalg.norm(global_hidden)),
        "topology_norm": float(np.linalg.norm(topology_hidden)),
        "unified_norm": float(np.linalg.norm(unified)),
        "n_patches": int(patch_states.shape[0]),
        "n_pairs": int(pair_value.shape[0]),
    }


def stage_states() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(4)
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    expected_patches = table["num_patches"].to_numpy()
    expected_pairs = table["num_pairs"].to_numpy()
    train_records, valid_records = _extract_v4_records()
    print(f"[states] records ready ({len(train_records)}/{len(valid_records)})", flush=True)

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        run_id = V4_RUN_IDS[seed]
        run_dir = run_path(run_id)
        config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        result = load_run_result(run_id)
        run_preds = np.asarray(
            result["evaluation"]["valid"]["predictions"], dtype=np.float64
        )
        fit_data, eval_data, audit = zpp._phase_data(
            train_records, valid_records, config=config
        )
        model = _build_v4_model(config, audit)
        state_path = run_dir / "artifacts" / "legacy_full_result_selection_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        n_params = int(sum(p.numel() for p in model.parameters()))
        if n_params != 99613:
            raise AssertionError(f"seed {seed}: params {n_params} != 99613")
        preds, patch_states_list, captures = _frozen_forward_with_states(model, eval_data)
        max_diff = float(np.abs(preds - run_preds).max())
        if max_diff > 1e-9:
            raise AssertionError(
                f"seed {seed}: frozen-forward predictions deviate from the run "
                f"json (max diff {max_diff})"
            )
        if len(patch_states_list) != 1000:
            raise AssertionError(f"seed {seed}: states for {len(patch_states_list)} graphs")
        if any(
            states.shape[0] != int(expected_patches[i])
            for i, states in enumerate(patch_states_list)
        ):
            raise AssertionError(f"seed {seed}: patch-state counts mismatch")
        # per-molecule summary rows
        rows: list[dict[str, Any]] = []
        continuous: list[np.ndarray] = []
        if int(captures["pair_value"].shape[0]) != int(expected_pairs.sum()):
            raise AssertionError(
                f"seed {seed}: captured pair rows {captures['pair_value'].shape[0]} "
                f"!= expected {expected_pairs.sum()}"
            )
        if int(captures["unified"].shape[0]) != 1000:
            raise AssertionError(f"seed {seed}: unified rows != 1000")
        pair_value_splits = _split_by_graph(captures["pair_value"], table)
        for i, patch_states in enumerate(patch_states_list):
            row = _state_summary_row(
                patch_states,
                pair_value_splits[i],
                captures["global_hidden"][i],
                captures["topology_hidden"][i],
                captures["unified"][i],
            )
            rows.append(row)
            continuous.append(
                _continuous_pair_vector_from_states(valid_records[i], patch_states)
            )
        summary_df = pd.DataFrame(rows)
        for col in ("molecule_id", f"residual_seed{seed}", f"abs_error_seed{seed}"):
            summary_df[col] = table[col].to_numpy()
        summary_path = AUDIT_ROOT / f"state_summary_seed{seed}.csv"
        summary_df.to_csv(summary_path, index=False)
        continuous_matrix = np.stack(continuous, axis=0).astype(np.float32)
        _save_npz(
            AUDIT_ROOT / f"continuous_pair_seed{seed}.npz",
            continuous_pair=continuous_matrix,
            prediction=preds.astype(np.float32),
        )
        per_seed[str(seed)] = {
            "run_id": run_id,
            "params": n_params,
            "max_abs_pred_diff_vs_run": max_diff,
            "state_file_sha256": _sha256(state_path)[:16],
            "summary_csv": str(summary_path),
            "continuous_pair_npz": str(
                AUDIT_ROOT / f"continuous_pair_seed{seed}.npz"
            ),
        }
        print(f"[states] seed {seed} done (max diff {max_diff})", flush=True)
    return _mark_done(
        "states",
        {
            "per_seed": per_seed,
            "gate": "predictions bit-exact (<=1e-9) vs recorded run json on all 4 seeds",
            "seconds": float(time.perf_counter() - started),
        },
    )


def _split_by_graph(captured: np.ndarray, table: pd.DataFrame) -> list[np.ndarray]:
    """Split a per-pair/per-graph capture array into per-molecule pieces in
    graph order using the captured n_pairs metadata per row."""
    pieces: list[np.ndarray] = []
    cursor = 0
    for i in range(len(table)):
        n = int(table.iloc[i]["num_pairs"])
        pieces.append(captured[cursor : cursor + n])
        cursor += n
    return pieces


def _continuous_pair_vector_from_states(record: Any, h: np.ndarray) -> np.ndarray:
    """Frozen Information-Gap definition: mean_{(i,j) in bucket}(h_i x h_j),
    5 buckets x 48 dims = 240D (D_BUCKETS = 5, patch_hidden = 48)."""
    h = np.asarray(h, dtype=np.float64)
    src, dst = record.pair_index[0], record.pair_index[1]
    buckets = record.pair_bucket
    blocks = []
    for bucket in range(zpp.DISTANCE_BUCKETS):
        mask = buckets == bucket
        if not mask.any():
            blocks.append(np.zeros(h.shape[1], dtype=np.float64))
            continue
        product = h[src[mask]] * h[dst[mask]]
        blocks.append(product.mean(axis=0))
    return np.concatenate(blocks)


# --------------------------------------------------------------------------
# stage: state_probes (Probe F continuous pair + Probe G representation states)
# --------------------------------------------------------------------------

def stage_state_probes() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    group_a = table[table["subgroup"] == "A"].reset_index(drop=True)
    group_a_idx = group_a.index.to_numpy()
    r_target = group_a["mean_residual"].to_numpy(dtype=np.float64)

    # --- Probe G: representation-state statistics vs residual ---
    rows: list[dict[str, Any]] = []
    state_cols = [
        "patch_norm_mean",
        "patch_norm_std",
        "patch_absmax_mean",
        "patch_centroid_norm",
        "pair_norm_mean",
        "pair_norm_std",
        "global_encoder_norm",
        "topology_norm",
        "unified_norm",
        "n_patches",
        "n_pairs",
    ]
    seed_summaries = {
        seed: pd.read_csv(AUDIT_ROOT / f"state_summary_seed{seed}.csv") for seed in SEEDS
    }
    for seed in SEEDS:
        ga = seed_summaries[seed].iloc[group_a_idx].reset_index(drop=True)
        for col in state_cols:
            x = ga[col].to_numpy(dtype=np.float64)
            for target_kind in ("residual", "abs_error"):
                y = ga[f"{target_kind}_seed{seed}"].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "probe": "state_stats",
                        "feature": col,
                        "seed": seed,
                        "target": target_kind,
                        "pearson": _pearson(x, y),
                        "spearman": _spearman(x, y),
                        "n": int(len(ga)),
                    }
                )
    state_corr = pd.DataFrame(rows)
    state_corr_path = AUDIT_ROOT / "state_stats_correlations_groupA.csv"
    state_corr.to_csv(state_corr_path, index=False)

    # aggregate per-feature across seeds (per-seed residual correlations)
    summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        ga = seed_summaries[seed].iloc[group_a_idx].reset_index(drop=True)
        for col in state_cols:
            x = ga[col].to_numpy(dtype=np.float64)
            r_seed = ga[f"residual_seed{seed}"].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "probe": "state_stats",
                    "feature": col,
                    "seed": seed,
                    "pearson_signed": _pearson(x, r_seed),
                    "spearman_signed": _spearman(x, r_seed),
                    "spearman_abs": _spearman(x, np.abs(r_seed)),
                }
            )
    # ensemble mean state (across seeds) vs ensemble residual
    ensemble_rows: list[dict[str, Any]] = []
    for col in state_cols:
        mean_state = np.mean(
            np.stack(
                [
                    seed_summaries[s].iloc[group_a_idx][col].to_numpy(dtype=np.float64)
                    for s in SEEDS
                ],
                axis=1,
            ),
            axis=1,
        )
        ensemble_rows.append(
            {
                "probe": "state_stats",
                "feature": col,
                "seed": "ensemble",
                "pearson_signed": _pearson(mean_state, r_target),
                "spearman_signed": _spearman(mean_state, r_target),
                "spearman_abs": _spearman(
                    mean_state, group_a["mean_abs_error"].to_numpy(dtype=np.float64)
                ),
            }
        )
    summary_all = pd.DataFrame(summary_rows + ensemble_rows)
    summary_path = AUDIT_ROOT / "state_stats_crossseed_groupA.csv"
    summary_all.to_csv(summary_path, index=False)

    # --- Probe F: continuous pair interaction (240D), valid-internal CV ---
    cv_rows: list[dict[str, Any]] = []
    alpha_grid = np.logspace(-4, 6, 25)
    for seed in ["ensemble"] + [str(s) for s in SEEDS]:
        if seed == "ensemble":
            r = r_target
        else:
            r = group_a[f"residual_seed{int(seed)}"].to_numpy(dtype=np.float64)
        X_parts = []
        for s in SEEDS:
            arr = _load_npz(AUDIT_ROOT / f"continuous_pair_seed{s}.npz")[
                "continuous_pair"
            ]
            X_parts.append(arr[group_a_idx].astype(np.float64))
        if seed == "ensemble":
            X = np.mean(np.stack(X_parts, axis=0), axis=0)
        else:
            X = X_parts[int(seed)]
        kf = KFold(n_splits=5, shuffle=False)
        r_hat = np.full(len(r), np.nan)
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr = scaler.transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            best_alpha, _ = _select_alpha_svd_internal(X_tr, r[train_idx], alpha_grid)
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_tr, r[train_idx])
            r_hat[test_idx] = ridge.predict(X_te)
        valid_mask = np.isfinite(r_hat)
        baseline_mae = float(np.mean(np.abs(r[valid_mask])))
        corrected_mae = float(mean_absolute_error(r[valid_mask], r_hat[valid_mask]))
        cv_rows.append(
            {
                "probe": "continuous_pair_interaction",
                "seed": seed,
                "feature_dim": int(X.shape[1]),
                "fit_scope": "VALID-INTERNAL-CV (descriptive; no OOF-train residual under freeze)",
                "n": int(valid_mask.sum()),
                "baseline_mae": baseline_mae,
                "corrected_mae": corrected_mae,
                "delta_mae": float(baseline_mae - corrected_mae),
                "residual_r2": float(
                    r2_score(r[valid_mask], r_hat[valid_mask])
                    if np.std(r[valid_mask]) > 0
                    else float("nan")
                ),
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_path = AUDIT_ROOT / "continuous_pair_cv_groupA.csv"
    cv_table.to_csv(cv_path, index=False)
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary = {
        "n_groupA": int(len(group_a)),
        "state_stats_correlations": str(state_corr_path),
        "state_stats_crossseed": str(summary_path),
        "continuous_pair_cv": str(cv_path),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("state_probes", summary)


# --------------------------------------------------------------------------
# stage: cross (target-component x structural cross analysis, Group A)
# --------------------------------------------------------------------------


def _partial_corr(
    x: np.ndarray,
    y: np.ndarray,
    controls: list[np.ndarray],
    method: str = "pearson",
) -> float:
    """Partial correlation of x,y after regressing out controls (+ intercept)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if method == "spearman":
        x = scipy_stats.rankdata(x)
        y = scipy_stats.rankdata(y)
        controls = [scipy_stats.rankdata(c) for c in controls]
    design = np.stack([np.ones_like(x)] + controls, axis=1)

    def residuals(value: np.ndarray) -> np.ndarray:
        beta, _, _, _ = np.linalg.lstsq(design, value, rcond=None)
        return value - design @ beta

    rx, ry = residuals(x), residuals(y)
    return _pearson(rx, ry)


def stage_cross() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    ga = table[table["subgroup"] == "A"].reset_index(drop=True)
    r_ens = ga["mean_residual"].to_numpy(dtype=np.float64)
    abs_ens = ga["mean_abs_error"].to_numpy(dtype=np.float64)
    z_logp = ga["z_logP"].to_numpy(dtype=np.float64)
    z_sa = ga["z_SA"].to_numpy(dtype=np.float64)
    cols = {
        "z_logP": z_logp,
        "z_SA": z_sa,
        "branching_node_ratio": ga["branching_node_ratio"].to_numpy(dtype=np.float64),
        "num_branching_nodes": ga["num_branching_nodes"].to_numpy(dtype=np.float64),
        "diameter": ga["diameter"].to_numpy(dtype=np.float64),
        "mean_shortest_path_distance": ga["mean_shortest_path_distance"].to_numpy(dtype=np.float64),
        "rare_le5_ratio": ga["rare_le5_ratio"].to_numpy(dtype=np.float64),
        "oov_patch_ratio": ga["oov_patch_ratio"].to_numpy(dtype=np.float64),
        "node_type_8_fraction": ga["node_type_8_fraction"].to_numpy(dtype=np.float64),
        "node_type_1_fraction": ga["node_type_1_fraction"].to_numpy(dtype=np.float64),
        "num_nodes": ga["num_nodes"].to_numpy(dtype=np.float64),
        "cycle_rank": ga["cycle_rank"].to_numpy(dtype=np.float64),
        "num_pairs": ga["num_pairs"].to_numpy(dtype=np.float64),
    }
    analyses: list[dict[str, Any]] = []

    def add(y_name: str, x_name: str, ctrl_names: list[str], method: str) -> None:
        x = cols[x_name]
        controls = [cols[c] for c in ctrl_names]
        y = r_ens if y_name == "residual_ensemble" else abs_ens
        analyses.append(
            {
                "scope": "group_A",
                "y": y_name,
                "x": x_name,
                "controls": "+".join(ctrl_names) if ctrl_names else "none",
                "method": method,
                "partial_correlation": _partial_corr(x, y, controls, method),
                "uncontrolled_correlation": _pearson(
                    x, y if method == "pearson" else scipy_stats.rankdata(y)
                ),
                "n": int(len(ga)),
            }
        )

    for method in ("pearson", "spearman"):
        # component attribution disentangling (signed residual)
        add("residual_ensemble", "z_SA", ["z_logP"], method)
        add("residual_ensemble", "z_logP", ["z_SA"], method)
        # variance-side attribution
        add("abs_ensemble", "z_SA", ["z_logP"], method)
        add("abs_ensemble", "z_logP", ["z_SA"], method)
        # structural proxies vs components
        for feature in (
            "branching_node_ratio",
            "num_branching_nodes",
            "diameter",
            "mean_shortest_path_distance",
            "rare_le5_ratio",
            "oov_patch_ratio",
            "num_nodes",
            "cycle_rank",
        ):
            add("abs_ensemble", feature, ["z_SA"], method)
            add("abs_ensemble", feature, ["z_SA", "z_logP"], method)
            add("residual_ensemble", feature, ["z_SA", "z_logP"], method)
        # composition vs components
        add("abs_ensemble", "node_type_8_fraction", ["z_logP"], method)
        add("abs_ensemble", "node_type_8_fraction", ["z_SA", "z_logP"], method)
        add("residual_ensemble", "node_type_8_fraction", ["z_logP"], method)
        add("residual_ensemble", "node_type_8_fraction", ["z_SA", "z_logP"], method)
        # rarity vs SA difficulty disentangling
        add("abs_ensemble", "z_SA", ["rare_le5_ratio"], method)
        add("abs_ensemble", "rare_le5_ratio", ["z_SA"], method)
        add("abs_ensemble", "diameter", ["rare_le5_ratio"], method)

    cross_table = pd.DataFrame(analyses)
    cross_path = AUDIT_ROOT / "cross_partial_correlations_groupA.csv"
    cross_table.to_csv(cross_path, index=False)

    # feature/component inter-correlation matrix (Group A) -- proxy diagnosis
    inter: list[dict[str, Any]] = []
    for left in ("z_logP", "z_SA", "num_nodes", "branching_node_ratio", "diameter"):
        for right in ("z_SA", "num_nodes", "branching_node_ratio", "diameter", "rare_le5_ratio"):
            if left == right:
                continue
            inter.append(
                {
                    "scope": "group_A",
                    "feature_a": left,
                    "feature_b": right,
                    "pearson": _pearson(cols[left], cols[right]),
                    "spearman": _spearman(cols[left], cols[right]),
                }
            )
    inter_table = pd.DataFrame(inter)
    inter_path = AUDIT_ROOT / "cross_feature_intercorrelations_groupA.csv"
    inter_table.to_csv(inter_path, index=False)

    # two-way difficulty grid: z_SA x z_logP medians -> MAE (variance side)
    grid_rows: list[dict[str, Any]] = []
    med_lp = np.median(z_logp)
    med_sa = np.median(z_sa)
    for lp_hi in (False, True):
        for sa_hi in (False, True):
            mask = (z_logp > med_lp) == lp_hi
            mask &= (z_sa > med_sa) == sa_hi
            sub = ga[mask]
            grid_rows.append(
                {
                    "z_logP_above_median": bool(lp_hi),
                    "z_SA_above_median": bool(sa_hi),
                    "n": int(mask.sum()),
                    "mae_ensemble": float(
                        np.abs(
                            sub["target"].to_numpy()
                            - sub["mean_prediction"].to_numpy()
                        ).mean()
                    ),
                    "mean_abs_error_across_seeds": float(sub["mean_abs_error"].mean()),
                    "mean_signed_residual": float(sub["mean_residual"].mean()),
                    "target_mean": float(sub["target"].mean()),
                }
            )
    grid_table = pd.DataFrame(grid_rows)
    grid_path = AUDIT_ROOT / "cross_zSA_zlogP_difficulty_grid.csv"
    grid_table.to_csv(grid_path, index=False)

    # target-level difficulty table (quintiles of y within Group A)
    y_rows: list[dict[str, Any]] = []
    y = ga["target"].to_numpy(dtype=np.float64)
    bins = pd.qcut(ga["target"], 5, duplicates="drop")
    for label in bins.cat.categories:
        indices = np.flatnonzero((bins == label).to_numpy())
        sub = ga.iloc[indices]
        y_rows.append(
            {
                "bin": str(label),
                "n": int(len(sub)),
                "target_mean": float(sub["target"].mean()),
                "mae_ensemble": float(
                    np.abs(
                        sub["target"].to_numpy()
                        - sub["mean_prediction"].to_numpy()
                    ).mean()
                ),
                "mean_abs_error_across_seeds": float(sub["mean_abs_error"].mean()),
                "mean_signed_residual": float(sub["mean_residual"].mean()),
                "z_SA_mean": float(sub["z_SA"].mean()),
                "z_logP_mean": float(sub["z_logP"].mean()),
            }
        )
    y_table = pd.DataFrame(y_rows)
    y_path = AUDIT_ROOT / "cross_target_quintile_difficulty.csv"
    y_table.to_csv(y_path, index=False)

    summary = {
        "n_groupA": int(len(ga)),
        "corr_zlogP_zSA_groupA": _pearson(cols["z_logP"], cols["z_SA"]),
        "spearman_zlogP_zSA_groupA": _spearman(cols["z_logP"], cols["z_SA"]),
        "systematic_difficulty": {
            "spearman_mean_abs_error_vs_residual_std_across_seeds": _spearman(
                ga["mean_abs_error"].to_numpy(), ga["residual_std_across_seeds"].to_numpy()
            ),
            "groupA_mae_ensemble": float(
                np.abs(ga["target"].to_numpy() - ga["mean_prediction"].to_numpy()).mean()
            ),
            "groupA_mean_of_per_seed_mae": float(
                ga[[f"abs_error_seed{s}" for s in SEEDS]].to_numpy().mean()
            ),
        },
        "tables": {
            "partial": str(cross_path),
            "intercorrelation": str(inter_path),
            "difficulty_grid": str(grid_path),
            "target_quintiles": str(y_path),
        },
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("cross", summary)


# --------------------------------------------------------------------------
# stage: summary (error distributions, top-30, probe table, ranking)
# --------------------------------------------------------------------------

DIST_COLS = ["mean_residual", "mean_abs_error"]


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mae": float(np.abs(values).mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "skewness": float(scipy_stats.skew(values)) if len(values) > 2 else float("nan"),
        "kurtosis": float(scipy_stats.kurtosis(values)) if len(values) > 2 else float("nan"),
    }


def stage_summary() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    group_a = table[table["subgroup"] == "A"].reset_index(drop=True)

    # --- Group error distributions (per seed and ensemble) ---
    dist_rows: list[dict[str, Any]] = []
    for scope, sub in (("all_valid", table), ("group_A", group_a)):
        for seed in SEEDS:
            dist_rows.append(
                {
                    "scope": scope,
                    "seed": seed,
                    "quantity": "signed_residual",
                    **_distribution_stats(sub[f"residual_seed{seed}"].to_numpy()),
                }
            )
            dist_rows.append(
                {
                    "scope": scope,
                    "seed": seed,
                    "quantity": "abs_error",
                    **_distribution_stats(sub[f"abs_error_seed{seed}"].to_numpy()),
                }
            )
        dist_rows.append(
            {
                "scope": scope,
                "seed": "ensemble",
                "quantity": "signed_residual",
                **_distribution_stats(sub["mean_residual"].to_numpy()),
            }
        )
        dist_rows.append(
            {
                "scope": scope,
                "seed": "ensemble",
                "quantity": "abs_error_ensemble",
                **_distribution_stats(
                    np.abs(sub["target"].to_numpy() - sub["mean_prediction"].to_numpy())
                ),
            }
        )
    dist_table = pd.DataFrame(dist_rows)
    dist_path = AUDIT_ROOT / "groupA_error_distributions.csv"
    dist_table.to_csv(dist_path, index=False)

    # --- top-30 Group A molecules by mean absolute residual ---
    top = group_a.nlargest(30, "mean_abs_error").reset_index(drop=True)
    keep = [
        "molecule_id",
        "target",
        "mean_prediction",
        "mean_residual",
        "residual_std_across_seeds",
        "mean_abs_error",
        "z_logP",
        "z_SA",
        "z_cycle",
        "num_nodes",
        "num_pairs",
        "num_unique_patch_tokens",
        "oov_patch_ratio",
        "rare_le1_ratio",
        "rare_le5_ratio",
        "rare_le10_ratio",
        "min_train_frequency",
        "mean_log1p_train_frequency",
        "num_branching_nodes",
        "branching_node_ratio",
        "diameter",
        "cycle_rank",
        "max_degree",
        "logP",
        "SA",
    ]
    top_path = AUDIT_ROOT / "top30_groupA_molecules.csv"
    top[keep].to_csv(top_path, index=False)

    # --- probe summary table (row per probe family) ---
    summary_rows: list[dict[str, Any]] = []
    # oracle attribution: strongest evidence rows from attribution tables
    attr = pd.read_csv(AUDIT_ROOT / "component_attribution_correlations.csv")
    for component in ("z_SA", "z_logP"):
        ga_row = attr[(attr["scope"] == "group_A") & (attr["component"] == component)]
        signed = ga_row[ga_row["target"] == "residual"].iloc[0]
        abs_row = ga_row[ga_row["target"] == "abs_error"].iloc[0]
        summary_rows.append(
            {
                "probe": f"oracle_{component}",
                "kind": "TARGET-COMPONENT ORACLE (attribution only; never a feature)",
                "scope": "group_A",
                "evidence": "correlation",
                "residual_pearson_ensemble": signed["pearson_ensemble"],
                "residual_spearman_ensemble": signed["spearman_ensemble"],
                "abs_spearman_ensemble": abs_row["spearman_ensemble"],
                "consistency_signed": signed["pearson_consistency"],
                "consistency_abs": abs_row["pearson_consistency"],
            }
        )
    # 1D CV probes
    cv = pd.read_csv(AUDIT_ROOT / "probe_1d_cv_groupA.csv")
    for _, row in cv.iterrows():
        summary_rows.append(
            {
                "probe": f"1d_{row['probe']}_{row['method']}",
                "kind": "structural/rarity feature (VALID-INTERNAL-CV)",
                "scope": "group_A",
                "evidence": "valid_internal_cv",
                "delta_mae": row["delta_mae"],
                "residual_r2": row["residual_r2"],
            }
        )
    # pair hash probes
    pairs = pd.read_csv(AUDIT_ROOT / "pair_probe_cv_groupA.csv")
    for _, row in pairs.iterrows():
        summary_rows.append(
            {
                "probe": f"pair_{row['probe']}",
                "kind": "exact pair hash (VALID-INTERNAL-CV)",
                "scope": "group_A",
                "evidence": "valid_internal_cv",
                "delta_mae": row["delta_mae"],
                "residual_r2": row["residual_r2"],
            }
        )
    cont = pd.read_csv(AUDIT_ROOT / "continuous_pair_cv_groupA.csv")
    for _, row in cont[cont["seed"] == "ensemble"].iterrows():
        summary_rows.append(
            {
                "probe": "continuous_pair_interaction",
                "kind": "frozen-state continuous pair (VALID-INTERNAL-CV)",
                "scope": "group_A",
                "evidence": "valid_internal_cv",
                "delta_mae": row["delta_mae"],
                "residual_r2": row["residual_r2"],
            }
        )
    # correlation probes: leading rows per block
    corr = pd.read_csv(AUDIT_ROOT / "probe_correlations_groupA.csv")
    for block in ("rarity_oov", "graph_complexity", "atom_bond_composition"):
        sub = corr[corr["block"] == block]
        lead = sub.loc[sub["abs_error_spearman_ensemble"].abs().idxmax()]
        summary_rows.append(
            {
                "probe": f"corr_lead_{block}:{lead['feature']}",
                "kind": "correlation diagnostic (no fitted probe)",
                "scope": "group_A",
                "evidence": "correlation",
                "abs_spearman_ensemble": lead["abs_error_spearman_ensemble"],
                "consistency_abs": lead["abs_error_spearman_consistency"],
                "residual_spearman_ensemble": lead["residual_spearman_ensemble"],
                "consistency_signed": lead["residual_spearman_consistency"],
            }
        )
    probe_table = pd.DataFrame(summary_rows)
    probe_path = AUDIT_ROOT / "probe_summary_table.csv"
    probe_table.to_csv(probe_path, index=False)

    summary = {
        "n_groupA": int(len(group_a)),
        "n_all_valid": int(len(table)),
        "tables": {
            "distributions": str(dist_path),
            "top30": str(top_path),
            "probe_summary": str(probe_path),
        },
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("summary", summary)


# --------------------------------------------------------------------------
# stage: decision (Q1-Q14 + decision record)
# --------------------------------------------------------------------------

def stage_decision() -> dict[str, Any]:
    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    group_a = table[table["subgroup"] == "A"]

    # assemble evidence numbers used by the decision text
    attr = pd.read_csv(AUDIT_ROOT / "component_attribution_correlations.csv")
    corr = pd.read_csv(AUDIT_ROOT / "probe_correlations_groupA.csv")
    cv = pd.read_csv(AUDIT_ROOT / "probe_1d_cv_groupA.csv")
    pairs = pd.read_csv(AUDIT_ROOT / "pair_probe_cv_groupA.csv")
    cont = pd.read_csv(AUDIT_ROOT / "continuous_pair_cv_groupA.csv")
    dist = pd.read_csv(AUDIT_ROOT / "groupA_error_distributions.csv")
    cross = pd.read_csv(AUDIT_ROOT / "cross_partial_correlations_groupA.csv")

    def _row(df: pd.DataFrame, mask: Any) -> dict[str, Any]:
        return df[mask].iloc[0].to_dict()

    ga_sa = attr[(attr["scope"] == "group_A") & (attr["component"] == "z_SA")]
    ga_lp = attr[(attr["scope"] == "group_A") & (attr["component"] == "z_logP")]
    sa_signed = ga_sa[ga_sa["target"] == "residual"].iloc[0]
    sa_abs = ga_sa[ga_sa["target"] == "abs_error"].iloc[0]
    lp_signed = ga_lp[ga_lp["target"] == "residual"].iloc[0]
    lp_abs = ga_lp[ga_lp["target"] == "abs_error"].iloc[0]

    best_cv = cv.loc[cv["residual_r2"].idxmax()]
    worst_cv = cv.loc[cv["residual_r2"].idxmin()]
    best_pair = pairs.loc[pairs["delta_mae"].idxmax()]
    cont_ens = cont[cont["seed"] == "ensemble"].iloc[0]
    dist_a_ens = dist[
        (dist["scope"] == "group_A")
        & (dist["seed"] == "ensemble")
        & (dist["quantity"] == "signed_residual")
    ].iloc[0]
    partial_sa = cross[
        (cross["y"] == "residual_ensemble")
        & (cross["x"] == "z_SA")
        & (cross["controls"] == "z_logP")
        & (cross["method"] == "spearman")
    ].iloc[0]["partial_correlation"]
    partial_lp = cross[
        (cross["y"] == "residual_ensemble")
        & (cross["x"] == "z_logP")
        & (cross["controls"] == "z_SA")
        & (cross["method"] == "spearman")
    ].iloc[0]["partial_correlation"]

    max_delta_all = float(
        max(
            cv["delta_mae"].max(),
            pairs["delta_mae"].max(),
            float(cont_ens["delta_mae"]),
        )
    )
    q1_answer = (
        "Ordinary bulk (Group A, excess=0, n=965/1000) still carries most of the "
        "remaining MAE mass by count, but per-molecule errors are small; the "
        "largest single-molecule errors remain in the long-cycle groups B/C "
        "(Group C ensemble MAE ~6.1 on 5 molecules vs Group A ~0.119)."
    )

    answers = {
        "Q1_remaining_residual_subgroup": q1_answer,
        "Q2_groupA_distribution": {
            "mean_residual_ensemble": dist_a_ens["mean"],
            "median": dist_a_ens["median"],
            "std": dist_a_ens["std"],
            "skewness": dist_a_ens["skewness"],
            "kurtosis": dist_a_ens["kurtosis"],
        },
        "Q3_residual_vs_SA_component": {
            "pearson_consistency": sa_signed["pearson_consistency"],
            "pearson_mean": sa_signed["pearson_mean"],
            "abs_spearman_ensemble": sa_abs["spearman_ensemble"],
            "abs_consistency": sa_abs["spearman_consistency"],
            "partial_spearman_given_logP": partial_sa,
        },
        "Q4_residual_vs_logP_component": {
            "pearson_consistency": lp_signed["pearson_consistency"],
            "pearson_mean": lp_signed["pearson_mean"],
            "abs_spearman_ensemble": lp_abs["spearman_ensemble"],
            "abs_consistency": lp_abs["spearman_consistency"],
            "partial_spearman_given_SA": partial_lp,
        },
        "Q5_strongest_component_attribution": (
            "None is dominant for signed residual (z_logP weak-positive 4/4 "
            "seeds, mean Pearson ~0.05, range ~0.07 across quintiles; z_SA "
            "signed unstable). Abs error tracks the low-target/low-z_SA/"
            "low-z_logP region (heteroscedastic difficulty), 4/4 seeds for "
            "both components; partial analyses attribute the signed part "
            "mainly to the logP side."
        ),
        "Q6_patch_rarity_signal": (
            "Yes for abs error (heteroscedastic difficulty): rare-le5 ratio "
            "Spearman ~0.30/seed, 4/4 seeds, ensemble 0.41; signed residual "
            "weak and unstable (<=3/4). Valid-internal 1D correction delta "
            "MAE <= -0.002 -> no exploitable signed bias."
        ),
        "Q7_complexity_branching_signal": (
            "Abs-error correlation is negative and consistent (4/4 seeds) for "
            "diameter/mean-SP (~-0.19..-0.27 ensemble) i.e. larger molecules "
            "are easier, mirroring the z_SA/z_logP difficulty gradient; no "
            "signed-bias signal survives after controlling z_SA/z_logP."
        ),
        "Q8_composition_signal": (
            "Weak but 4/4-consistent on both sides for the leading type: "
            "N H1+ (protonated aromatic N, atom_dict index 8) fraction has "
            "signed Spearman +0.11 (partial +0.18 given z_SA+z_logP -> the "
            "only composition signal surviving component control; 135/965 "
            "Group-A molecules carry >=1 such atom) and abs-error Spearman "
            "+0.15 (abs side collapses to ~0 after component control). "
            "Fitted 1D probes of it give delta MAE ~ -0.001 (NO-GO). "
            "N H3+/I/edge-type rows are sparse or weaker; multiple-"
            "comparison caveat over 18 composition features applies."
        ),
        "Q9_pair_cooccurrence": "No (all 2048-D exact pair probes delta MAE ~ -0.001, R2<=0; same as v2).",
        "Q10_relation_pair_stronger": "No (relation-conditioned probes are not stronger than pair-only; both ~0).",
        "Q11_continuous_pair_interaction": "No (240-D frozen-state probe delta MAE ~ -0.0006 ensemble; all seeds <= +0.0004).",
        "Q12_consistent_3of4_findings": [
            "abs-error difficulty gradient with z_SA and z_logP (4/4 seeds)",
            "abs-error increase with patch rarity (4/4 seeds)",
            "weak positive signed trend with z_logP (4/4 seeds, small)",
            "representation-state statistics: no consistent signal",
        ],
        "Q13_most_credible_remaining_bottleneck": (
            "Variance-side (heteroscedastic) difficulty only: molecules on the "
            "low-z_SA axis fail harder at every z_logP level (two-way grid MAE "
            "0.15-0.16 when z_SA low vs 0.08-0.10 when high), and patch-rarity "
            "adds an independent difficulty axis; signed bias is limited to a "
            "small logP-side under-prediction of the top quintile and a sparse "
            "N-H1+ composition hint. No second-layer structural bias is "
            "exposed by any audited graph representation, so residual MAE is "
            "best described as systematic difficulty (irreducible for a "
            "fixed-objective patch model) rather than a correctable bias."
        ),
        "Q14_recommended_v5": (
            "NO CLEAR SECONDARY SIGNAL -- see decision; if a v5 direction is "
            "ever opened, it must be an objective/calibration (heteroscedastic "
            "or density-aware) study, not another representation channel."
        ),
    }

    decision_payload = {
        "decision": "NO CLEAR SECONDARY SIGNAL",
        "pre_registered_gate": {
            "best_positive_delta_mae_across_all_fitted_probes": max_delta_all,
            "required_for_weak_candidate": 0.003,
            "required_for_candidate": 0.005,
            "required_for_strong": 0.008,
        },
        "groupA_ensemble_mae": float(
            np.abs(
                group_a["target"].to_numpy() - group_a["mean_prediction"].to_numpy()
            ).mean()
        ),
        "evidence": {
            "oracle_SA": {
                "signed_pearson_mean": sa_signed["pearson_mean"],
                "signed_consistency": sa_signed["pearson_consistency"],
                "abs_spearman_ensemble": sa_abs["spearman_ensemble"],
                "abs_consistency": sa_abs["spearman_consistency"],
                "cv_isotonic_delta_mae": float(
                    cv[(cv["probe"] == "z_SA") & (cv["method"] == "isotonic")]["delta_mae"].iloc[0]
                ),
            },
            "oracle_logP": {
                "signed_pearson_mean": lp_signed["pearson_mean"],
                "signed_consistency": lp_signed["pearson_consistency"],
                "abs_spearman_ensemble": lp_abs["spearman_ensemble"],
                "abs_consistency": lp_abs["spearman_consistency"],
                "cv_isotonic_delta_mae": float(
                    cv[(cv["probe"] == "z_logP") & (cv["method"] == "isotonic")]["delta_mae"].iloc[0]
                ),
            },
            "best_1d_cv_probe": best_cv.to_dict(),
            "worst_1d_cv_probe": worst_cv.to_dict(),
            "best_pair_probe": best_pair.to_dict(),
            "continuous_pair_ensemble": cont_ens.to_dict(),
            "rarity_lead_feature": corr.loc[
                corr["abs_error_spearman_ensemble"].abs().idxmax()
            ].to_dict(),
        },
        "answers": answers,
    }
    record_path = AUDIT_ROOT / "decision_record.json"
    _write_json(record_path, decision_payload)
    summary = {
        "decision": decision_payload["decision"],
        "record": str(record_path),
        "seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(summary, indent=2))
    return _mark_done("decision", summary)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def stage_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    table = pd.read_csv(AUDIT_ROOT / "validation_master_table.csv")
    ga = table[table["subgroup"] == "A"].reset_index(drop=True)

    def binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            bins = pd.qcut(x, n_bins, duplicates="drop")
        except ValueError:
            return np.array([]), np.array([]), np.array([])
        xs, ys = [], []
        for label in bins.categories:
            idx = np.flatnonzero(bins == label)
            xs.append(np.mean(x[idx]))
            ys.append(np.mean(y[idx]))
        return np.asarray(xs), np.asarray(ys), np.asarray([len(bins.categories)])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs, ys, _ = binned_mean(ga["z_SA"].to_numpy(), ga["mean_residual"].to_numpy())
    ax.plot(xs, ys, "o-", color="#1f77b4", label="binned mean residual")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("z_SA (standardized SA = -sascorer)")
    ax.set_ylabel("mean v4 residual (y - yhat, 4-seed ensemble)")
    ax.set_title("Group A: mean residual vs z_SA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_groupA_mean_residual_vs_zSA.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs, ys, _ = binned_mean(ga["z_logP"].to_numpy(), ga["mean_residual"].to_numpy())
    ax.plot(xs, ys, "o-", color="#d62728", label="binned mean residual")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("z_logP (standardized RDKit logP)")
    ax.set_ylabel("mean v4 residual (4-seed ensemble)")
    ax.set_title("Group A: mean residual vs z_logP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_groupA_mean_residual_vs_zlogP.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        ga["rare_le5_ratio"], ga["mean_abs_error"], s=6, alpha=0.35, color="#2ca02c"
    )
    xs, ys, _ = binned_mean(ga["rare_le5_ratio"].to_numpy(), ga["mean_abs_error"].to_numpy())
    ax.plot(xs, ys, "o-", color="black", label="binned mean abs residual")
    ax.set_xlabel("rare<=5 patch ratio")
    ax.set_ylabel("mean absolute residual (4 seeds)")
    ax.set_title("Group A: absolute residual vs patch rarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_groupA_abs_residual_vs_rarity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(
        ga["branching_node_ratio"],
        ga["mean_abs_error"],
        s=6,
        alpha=0.35,
        color="#9467bd",
    )
    xs, ys, _ = binned_mean(
        ga["branching_node_ratio"].to_numpy(), ga["mean_abs_error"].to_numpy()
    )
    ax.plot(xs, ys, "o-", color="black", label="binned mean abs residual")
    ax.set_xlabel("branching node ratio (deg>=3)")
    ax.set_ylabel("mean absolute residual (4 seeds)")
    ax.set_title("Group A: absolute residual vs branching")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_groupA_abs_residual_vs_branching.png", dpi=150)
    plt.close(fig)

    # fig5: best non-oracle 1D probes (cycle_rank: highest CV residual R^2
    # +0.017; rare_le5_ratio shown as the rarity-side alternate): predicted vs
    # true residual
    panels = [("cycle_rank", "isotonic"), ("rare_le5_ratio", "isotonic")]
    generated: list[str] = []
    for probe, method in panels:
        x = ga[probe].to_numpy(dtype=np.float64)
        r = ga["mean_residual"].to_numpy(dtype=np.float64)
        kf = KFold(n_splits=5, shuffle=False)
        r_hat = np.full_like(r, np.nan)
        for train_idx, test_idx in kf.split(x):
            if method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x[train_idx], r[train_idx])
                r_hat[test_idx] = iso.predict(x[test_idx])
            else:
                scaler = StandardScaler().fit(x[train_idx].reshape(-1, 1))
                ridge = Ridge(alpha=1.0)
                ridge.fit(scaler.transform(x[train_idx].reshape(-1, 1)), r[train_idx])
                r_hat[test_idx] = ridge.predict(
                    scaler.transform(x[test_idx].reshape(-1, 1))
                )
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(r_hat, r, s=8, alpha=0.4)
        lim = max(np.nanmax(np.abs(r)), np.nanmax(np.abs(r_hat)))
        ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.8)
        ax.set_xlabel("predicted residual (valid-internal CV)")
        ax.set_ylabel("true residual (4-seed ensemble)")
        ax.set_title(f"Group A: non-oracle probe ({probe}, {method})")
        fig.tight_layout()
        path = FIG_DIR / f"fig5_probe_{probe}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(str(path))
    best_probe = panels[0][0]

    # fig6: top-30 heatmap
    top = ga.nlargest(30, "mean_abs_error")
    feature_cols = ["z_logP", "z_SA", "num_nodes", "num_pairs", "rare_le5_ratio", "oov_patch_ratio", "branching_node_ratio", "cycle_rank", "mean_abs_error"]
    matrix = top[feature_cols].to_numpy(dtype=np.float64)
    robust = np.nanmedian(matrix, axis=0)
    robust_scale = np.nanpercentile(matrix, 75, axis=0) - np.nanpercentile(matrix, 25, axis=0)
    normed = (matrix - robust) / np.where(robust_scale == 0, 1.0, robust_scale)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(normed, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["molecule_id"].tolist(), fontsize=7)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="robust-normalized value")
    ax.set_title("Top-30 Group A residual molecules: feature summary (descending MAE)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_top30_groupA_heatmap.png", dpi=150)
    plt.close(fig)

    summary = {
        "figures": [
            "fig1_groupA_mean_residual_vs_zSA.png",
            "fig2_groupA_mean_residual_vs_zlogP.png",
            "fig3_groupA_abs_residual_vs_rarity.png",
            "fig4_groupA_abs_residual_vs_branching.png",
            f"fig5_probe_{best_probe}.png",
            "fig6_top30_groupA_heatmap.png",
        ],
        "alternate_panels": generated,
        "seconds": float(time.perf_counter() - started),
    }
    return _mark_done("figures", summary)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stages",
        nargs="*",
        choices=[
            "residuals",
            "master",
            "attribution",
            "probes",
            "pairs",
            "states",
            "state_probes",
            "cross",
            "summary",
            "decision",
            "figures",
        ],
        help="stages to run (default: all available)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild artifacts")
    args = parser.parse_args(argv)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # order matters: later stages depend on earlier artifacts
    available = [
        "residuals",
        "master",
        "attribution",
        "probes",
        "pairs",
        "states",
        "state_probes",
        "cross",
        "summary",
        "decision",
        "figures",
    ]
    stages = args.stages or available
    for stage in stages:
        if stage not in available:
            raise SystemExit(f"unknown stage {stage!r}")
        if args.force or not _is_done(stage):
            started = time.perf_counter()
            print(f"[stage] {stage} ...", flush=True)
            result = globals()[f"stage_{stage}"]()
            print(f"[done] {stage} in {time.perf_counter() - started:.1f}s", flush=True)
            _ = result
        else:
            print(f"[skip] {stage} already done (use --force)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
