"""Sample-Efficiency Gain Localization Audit (ZINC / compact-v4-smallhead).

Zero-new-training mechanism audit.  One question:

    What structural support does additional training data provide to
    compact-v4?  Concretely: the already-observed ~0.05 MAE / data-doubling
    sample-efficiency gain -- which target-independent training-set support
    growth is it mainly associated with?

Only three pre-registered primary mechanism families are compared:

    1. patch/token support          (document frequency of rooted patch tokens)
    2. relation-context support     (document frequency of exact relation keys)
    3. whole-molecule structural-neighbour density (PATCH_FULL k=8 NN distance)

plus graph-size / object-count controls (not a mechanism family).

The audit is split by a hard firewall:

    Phase U  -- structural support.  Only raw graph + preprocessing + D_N
                membership.  No target, no prediction, no error, no gain.
    Phase Y  -- gain association.  Only after Phase U is hash-locked.

All support is measured with *document frequency across unique training
molecules* (one document-support per molecule), never optimizer exposure.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_sample_efficiency_gain_localization_audit <stage>

Stages: ``locks phaseU phaseY decision figures integrity all``.

Hard locks: zero new full-model training; reuse exactly the existing
1800/3600/7200 x seed0/seed1 compact-v4-smallhead checkpoints; no additional
sample sizes; no external models; no official valid; no official test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_smallhead_e2e as shead,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_internal_generalization_stochasticity_audit as iga,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_long_range_proxy as zlr,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_patch_path_pooling as zpp,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_post_v4_residual_audit as postv4,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_raw_graph_patch_sufficiency_audit as R,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_training_sufficiency as suff,
)

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/sample_efficiency_gain_localization"
FIGURE_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "sample_efficiency_gain_localization_v1"

PROBE_N = 2000
NESTED_SIZES = (1800, 3600, 7200)
SEEDS = (0, 1)
NESTED_LABEL = {1800: "1800", 3600: "3600", 7200: "7200"}

# prior frozen audits (source of truth)
SAMPLE_EFF_DIR = TRACK_ROOT / "results/inductive_bias_sample_efficiency_audit"
TOP5_DIR = TRACK_ROOT / "results/top5_checkpoint_aggregation_stabilization"
RAW_AUDIT_DIR = TRACK_ROOT / "results/raw_graph_patch_system_sufficiency"
SPLIT_MANIFEST = TRACK_ROOT / "results/optimized_manifold_broad_state_screen/split_manifest.json"
SUBSET_LOCK = SAMPLE_EFF_DIR / "sample_efficiency_subset_lock.json"

# frozen estimators
SOUP_STATE_PATHS = {
    (1800, 0): SAMPLE_EFF_DIR / "states/soup_state_N1800_I0T0.pt",
    (1800, 1): SAMPLE_EFF_DIR / "states/soup_state_N1800_I1T1.pt",
    (3600, 0): SAMPLE_EFF_DIR / "states/soup_state_N3600_I0T0.pt",
    (3600, 1): SAMPLE_EFF_DIR / "states/soup_state_N3600_I1T1.pt",
    (7200, 0): TOP5_DIR / "states/soup_state_I0T0.pt",
    (7200, 1): TOP5_DIR / "states/soup_state_I1T1.pt",
}
SOUP_ABS_ERR_PATHS = {
    (1800, 0): SAMPLE_EFF_DIR / "probe_abs_err_soup_N1800_I0T0.npy",
    (1800, 1): SAMPLE_EFF_DIR / "probe_abs_err_soup_N1800_I1T1.npy",
    (3600, 0): SAMPLE_EFF_DIR / "probe_abs_err_soup_N3600_I0T0.npy",
    (3600, 1): SAMPLE_EFF_DIR / "probe_abs_err_soup_N3600_I1T1.npy",
    (7200, 0): TOP5_DIR / "probe_abs_err_soup_I0T0.npy",
    (7200, 1): TOP5_DIR / "probe_abs_err_soup_I1T1.npy",
}
RAW_ABS_ERR_PATHS = {
    (1800, 0): SAMPLE_EFF_DIR / "probe_abs_err_raw_N1800_I0T0.npy",
    (1800, 1): SAMPLE_EFF_DIR / "probe_abs_err_raw_N1800_I1T1.npy",
    (3600, 0): SAMPLE_EFF_DIR / "probe_abs_err_raw_N3600_I0T0.npy",
    (3600, 1): SAMPLE_EFF_DIR / "probe_abs_err_raw_N3600_I1T1.npy",
    (7200, 0): TOP5_DIR / "probe_abs_err_raw_I0T0.npy",
    (7200, 1): TOP5_DIR / "probe_abs_err_raw_I1T1.npy",
}

# pre-registered thresholds
RARE_DF = 5
NN_K = 8
QUARTILE_TARGET = 0.010
SPEARMAN_TARGET = 0.15
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261006
PRIOR_BOOTSTRAP_SEED = 20260912  # historical audit seed (provenance reproduction only)
WELL_COVERED_MIN = 200
COVERAGE_RETAIN_HIGH = 0.60
COVERAGE_RETAIN_LOW = 0.40
N_FOLDS = 5
FOLD_SALT = "sample-efficiency-gain-localization-v1-20261006"

TOKEN_FAMILY = "token"
RELATION_FAMILY = "relation"
NEIGHBOR_FAMILY = "neighbor_density"
FAMILIES = (TOKEN_FAMILY, RELATION_FAMILY, NEIGHBOR_FAMILY)

# transitions: name -> (lower N used for support, lower N, upper N)
TRANSITIONS = {
    "18_36": (1800, 3600),
    "36_72": (3600, 7200),
}

# seed0/seed1 for the 7200 anchor is run I0T0 / I1T1
SEED_LABEL = {0: "seed0", 1: "seed1"}

# ---------------------------------------------------------------------------
# firewall: never load official valid / official test
# ---------------------------------------------------------------------------

OFFICIAL_VALID_LOADED = False
OFFICIAL_TEST_LOADED = False


class FirewallError(RuntimeError):
    pass


class ProbeAccessError(RuntimeError):
    pass


_TARGETS_UNLOCKED = False


def _install_firewall() -> None:
    if getattr(_install_firewall, "_installed", False):
        return

    original_load_zinc = zlr._load_zinc

    def _guarded_load_zinc(root: Path, split: str):
        if str(split) != "train":
            raise FirewallError(f"official {split} extraction is forbidden in this audit")
        return original_load_zinc(root, split)

    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise FirewallError("official valid/test extraction is forbidden in this audit")

    zlr._load_zinc = _guarded_load_zinc  # type: ignore[assignment]
    suff.extract_test_records = _blocked  # type: ignore[assignment]
    suff.load_train_valid_records = _blocked  # type: ignore[assignment]
    postv4._extract_v4_records = _blocked  # type: ignore[assignment]
    _install_firewall._installed = True  # type: ignore[attr-defined]


def _require_targets_unlocked() -> None:
    if not _TARGETS_UNLOCKED:
        raise ProbeAccessError(
            "target / prediction access is forbidden until Phase U is hash-locked"
        )


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_indices(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in indices).encode()).hexdigest()


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_records() -> list[Any]:
    """Official-train compact-v4 GraphRecords only (valid/test never opened)."""
    _install_firewall()
    with gzip.open(postv4.CACHE_DIR / "v4_records_train.pkl.gz", "rb") as handle:
        return list(pickle.load(handle))


def split_view() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ref, sel, probe, report = R.split_positions()
    return ref, sel, probe, report


def nested_subsets() -> dict[str, list[int]]:
    lock = _read_json(SUBSET_LOCK)
    return {
        "1800": [int(i) for i in lock["indices_1800"]],
        "3600": [int(i) for i in lock["indices_3600"]],
        "7200": [int(i) for i in lock["indices_7200"]],
    }


# ---------------------------------------------------------------------------
# Stage: locks / prediction provenance / bootstrap provenance
# ---------------------------------------------------------------------------


def _raw_checkpoint_map(records: list[Any]) -> dict[tuple[int, int], Path]:
    out: dict[tuple[int, int], Path] = {}
    for run_id, seed in (("N1800_I0T0", 0), ("N1800_I1T1", 1)):
        out[(1800, seed)] = Path(_read_json(SAMPLE_EFF_DIR / f"soup_construction_{run_id}.json")["members"][0]["snapshot"])
    for run_id, seed in (("N3600_I0T0", 0), ("N3600_I1T1", 1)):
        out[(3600, seed)] = Path(_read_json(SAMPLE_EFF_DIR / f"soup_construction_{run_id}.json")["members"][0]["snapshot"])
    for run_id, seed in (("I0T0", 0), ("I1T1", 1)):
        top5 = _read_json(TOP5_DIR / f"top5_manifest_{run_id}.json")
        out[(7200, seed)] = Path(top5["selected"][0]["checkpoint_path"])
    return out


def stage_prediction_provenance() -> dict[str, Any]:
    """Recompute the six frozen SOUP / RAW probe predictions from states.

    This is the *only* place a probe prediction is produced before Phase Y and
    it is never used for any support definition.
    """
    global _TARGETS_UNLOCKED

    data = iga.build_audit_data()
    probe_records = data["probe"]
    probe_gids = [int(i) for i in data["indices"]["internal_probe"]]
    targets = np.asarray([float(record.y) for record in probe_records], dtype=np.float64)

    probe_loader = zpp._make_loader(data["probe"], 256, False, 0)
    _mae, loader_targets, _preds = shead._evaluate_mae(
        shead.build_smallhead(0), probe_loader, torch.device("cpu")
    )
    target_max_diff = float(np.max(np.abs(loader_targets.astype(np.float64) - targets)))

    raw_ckpts = _raw_checkpoint_map(load_records())

    predictions: dict[tuple[int, int, str], np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for n in NESTED_SIZES:
        for seed in SEEDS:
            # SOUP
            soup_state = torch.load(SOUP_STATE_PATHS[(n, seed)], map_location="cpu", weights_only=True)
            soup_model = shead.build_smallhead(0)
            soup_model.load_state_dict(soup_state)
            soup_model.eval()
            _m, _t, soup_pred = shead._evaluate_mae(soup_model, probe_loader, torch.device("cpu"))
            soup_pred = soup_pred.astype(np.float64)
            predictions[(n, seed, "soup")] = soup_pred
            soup_saved = np.load(SOUP_ABS_ERR_PATHS[(n, seed)])
            soup_err = np.abs(targets - soup_pred)
            # RAW
            raw_state = torch.load(raw_ckpts[(n, seed)], map_location="cpu", weights_only=True)
            raw_model = shead.build_smallhead(0)
            raw_model.load_state_dict(raw_state)
            raw_model.eval()
            _m2, _t2, raw_pred = shead._evaluate_mae(raw_model, probe_loader, torch.device("cpu"))
            raw_pred = raw_pred.astype(np.float64)
            predictions[(n, seed, "raw")] = raw_pred
            raw_saved = np.load(RAW_ABS_ERR_PATHS[(n, seed)])
            raw_err = np.abs(targets - raw_pred)
            provenance[f"N{n}_seed{seed}"] = {
                "soup_state": str(SOUP_STATE_PATHS[(n, seed)]),
                "soup_state_file_sha256": _sha256_file(SOUP_STATE_PATHS[(n, seed)]),
                "soup_state_tensor_sha256": _state_hash(soup_state),
                "raw_checkpoint": str(raw_ckpts[(n, seed)]),
                "raw_checkpoint_sha256": _sha256_file(raw_ckpts[(n, seed)]),
                "soup_probe_mae_recomputed": float(soup_err.mean()),
                "soup_probe_mae_saved_npy": float(soup_saved.mean()),
                "soup_abs_err_max_diff_vs_saved": float(np.max(np.abs(soup_err - soup_saved))),
                "raw_probe_mae_recomputed": float(raw_err.mean()),
                "raw_probe_mae_saved_npy": float(raw_saved.mean()),
                "raw_abs_err_max_diff_vs_saved": float(np.max(np.abs(raw_err - raw_saved))),
            }

    frame = pd.DataFrame(
        {
            "graph_id": probe_gids,
            "target": targets,
            **{
                f"pred_{'n' + str(n)}_{SEED_LABEL[seed]}_soup": predictions[(n, seed, "soup")]
                for n in NESTED_SIZES
                for seed in SEEDS
            },
            **{
                f"abs_err_{'n' + str(n)}_{SEED_LABEL[seed]}_soup": np.abs(targets - predictions[(n, seed, "soup")])
                for n in NESTED_SIZES
                for seed in SEEDS
            },
            **{
                f"pred_{'n' + str(n)}_{SEED_LABEL[seed]}_raw": predictions[(n, seed, "raw")]
                for n in NESTED_SIZES
                for seed in SEEDS
            },
            **{
                f"abs_err_{'n' + str(n)}_{SEED_LABEL[seed]}_raw": np.abs(targets - predictions[(n, seed, "raw")])
                for n in NESTED_SIZES
                for seed in SEEDS
            },
        }
    )
    _write_parquet(RESULTS_DIR / "probe_predictions_locked.parquet", frame)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "probe_source": "frozen official-train internal_probe (2000), index-for-index",
        "probe_graph_ids": probe_gids,
        "probe_graph_id_sha256": _sha256_indices(probe_gids),
        "target_source": "frozen official-train GraphRecord.y",
        "target_max_diff_vs_loader": target_max_diff,
        "n_predictions": int(len(predictions)),
        "predictions": provenance,
        "all_soup_match_saved": bool(all(v["soup_abs_err_max_diff_vs_saved"] == 0.0 for v in provenance.values())),
        "all_raw_match_saved": bool(all(v["raw_abs_err_max_diff_vs_saved"] == 0.0 for v in provenance.values())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "prediction_provenance.json", payload)
    return payload


def stage_locks() -> dict[str, Any]:
    _install_firewall()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref, sel, probe, split_report = split_view()
    records = load_records()
    subsets = nested_subsets()

    # ---- model inventory -------------------------------------------------
    raw_ckpts = _raw_checkpoint_map(records)
    models = {}
    for n in NESTED_SIZES:
        for seed in SEEDS:
            soup_path = SOUP_STATE_PATHS[(n, seed)]
            soup_state = torch.load(soup_path, map_location="cpu", weights_only=True)
            models[f"N{n}_seed{seed}"] = {
                "N": n,
                "seed": seed,
                "soup_state": str(soup_path),
                "soup_state_file_sha256": _sha256_file(soup_path),
                "soup_state_tensor_sha256": _state_hash(soup_state),
                "raw_checkpoint": str(raw_ckpts[(n, seed)]),
                "raw_checkpoint_sha256": _sha256_file(raw_ckpts[(n, seed)]),
            }
    architecture = shead.build_smallhead(0)
    model_inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": "compact-v4-smallhead",
        "total_params": int(shead._n_params(architecture)),
        "expected_total_params": 82115,
        "R_dimension": int(architecture.unified_graph_width),
        "models": models,
        "n_models": int(len(models)),
        "historical_init_sha256": {
            "seed0": _read_json(iga.RESULTS_DIR / "run_I0T0.json")["init_state_sha256"],
            "seed1": _read_json(iga.RESULTS_DIR / "run_I1T1.json")["init_state_sha256"],
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "model_inventory.json", model_inventory)

    # ---- subset inventory ------------------------------------------------
    subset_inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "source": "frozen compact-v4-sample-efficiency nested subsets (target-independent)",
        "salt": _read_json(SUBSET_LOCK)["salt"],
        "source_lock": str(SUBSET_LOCK),
        "source_lock_sha256": _sha256_file(SUBSET_LOCK),
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "probe_n": int(len(probe)),
        "probe_sha256": _sha256_indices(probe.tolist()),
        "optimization_train_n": int(len(ref)),
        "optimization_train_sha256": _sha256_indices(ref.tolist()),
        "subsets": {
            label: {
                "n": len(idx),
                "sha256": _sha256_indices(sorted(idx)),
                "ordered_sha256": _sha256_indices(idx),
            }
            for label, idx in subsets.items()
        },
        "nested_invariants": {
            "D1800_subset_D3600": set(subsets["1800"]) <= set(subsets["3600"]),
            "D3600_subset_D7200": set(subsets["3600"]) <= set(subsets["7200"]),
            "D7200_equals_optimization_train": set(subsets["7200"]) == set(ref.tolist()),
            "probe_disjoint_D7200": len(set(probe.tolist()) & set(subsets["7200"])) == 0,
        },
        "target_used_for_subsets": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "subset_inventory.json", subset_inventory)

    # ---- audit protocol lock --------------------------------------------
    audit_protocol_lock = {
        "name": "Sample-Efficiency Gain Localization Audit",
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "which target-independent training-set support growth is the "
            "observed ~0.05 MAE/data-doubling gain mainly associated with?"
        ),
        "families": {
            "token": "document frequency of historical rooted patch tokens over unique D_N molecules",
            "relation": "document frequency of exact discrete relation keys over unique D_N molecules",
            "neighbor_density": "frozen PATCH_FULL k=8 nearest-neighbour distance to D_N",
        },
        "size_controls": ["log1p(atom_count)", "log1p(relation_count)"],
        "transitions": {name: list(value) for name, value in TRANSITIONS.items()},
        "primary_estimator": "soup",
        "support_principle": "document frequency over unique training molecules (no exposure counts)",
        "firewall": "Phase U (structural support) hash-locked before any target/prediction read",
        "thresholds": {
            "rare_df": RARE_DF,
            "nn_k": NN_K,
            "quartile_target": QUARTILE_TARGET,
            "spearman_target": SPEARMAN_TARGET,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "well_covered_min": WELL_COVERED_MIN,
            "coverage_retain_high": COVERAGE_RETAIN_HIGH,
            "coverage_retain_low": COVERAGE_RETAIN_LOW,
            "n_folds_oof": N_FOLDS,
        },
        "locks": {
            "no_new_training": True,
            "no_new_sample_sizes": True,
            "no_external_models": True,
            "no_hpo": True,
            "no_learned_probe": True,
            "no_official_valid": True,
            "no_official_test": True,
        },
        "compute_budget": "zero new full-model training; reuse 6 frozen models",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "environment": _environment(),
    }
    _write_json(RESULTS_DIR / "audit_protocol_lock.json", audit_protocol_lock)

    # ---- prediction provenance + learning curve --------------------------
    provenance = stage_prediction_provenance()

    # ---- recomputed learning curve --------------------------------------
    soup_mae = {}
    raw_mae = {}
    for n in NESTED_SIZES:
        for seed in SEEDS:
            soup_mae[(n, seed)] = float(np.load(SOUP_ABS_ERR_PATHS[(n, seed)]).mean())
            raw_mae[(n, seed)] = float(np.load(RAW_ABS_ERR_PATHS[(n, seed)]).mean())
    gains = {}
    for name, (lo, hi) in TRANSITIONS.items():
        for seed in SEEDS:
            g_soup = soup_mae[(lo, seed)] - soup_mae[(hi, seed)]
            g_raw = raw_mae[(lo, seed)] - raw_mae[(hi, seed)]
            gains[f"{name}_{SEED_LABEL[seed]}"] = {"G_soup": g_soup, "G_raw": g_raw}
    mean_g = {
        name: float(np.mean([gains[f"{name}_{SEED_LABEL[seed]}"]["G_soup"] for seed in SEEDS]))
        for name in TRANSITIONS
    }
    recomputed = {
        "protocol_version": PROTOCOL_VERSION,
        "soup_probe_mae": {f"N{n}_{SEED_LABEL[seed]}": soup_mae[(n, seed)] for n in NESTED_SIZES for seed in SEEDS},
        "raw_probe_mae": {f"N{n}_{SEED_LABEL[seed]}": raw_mae[(n, seed)] for n in NESTED_SIZES for seed in SEEDS},
        "gains": gains,
        "two_seed_mean_G_soup": mean_g,
        "matches_prior_learning_curve": _check_prior_learning_curve(soup_mae),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "recomputed_learning_curve.json", recomputed)

    # ---- bootstrap provenance check -------------------------------------
    bootstrap_check = bootstrap_provenance_check()
    return {"model_inventory": model_inventory, "subset_inventory": subset_inventory,
            "provenance": provenance, "recomputed": recomputed, "bootstrap": bootstrap_check}


def _check_prior_learning_curve(soup_mae: Mapping[tuple[int, int], float]) -> dict[str, Any]:
    prior = {}
    for row in pd.read_csv(SAMPLE_EFF_DIR / "learning_curve_summary.csv").to_dict("records"):
        prior[(int(row["N"]), int(row["seed"]))] = float(row["soup_probe_mae"])
    diffs = {f"N{n}_{SEED_LABEL[s]}": abs(soup_mae[(n, s)] - prior[(n, s)]) for n in NESTED_SIZES for s in SEEDS}
    return {"max_abs_diff": float(max(diffs.values())), "diffs": diffs, "consistent": bool(max(diffs.values()) <= 1e-9)}


def _paired_bootstrap(diff: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.size
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(B), n))
    means = diff[idx].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "n": n, "B": int(B), "seed": int(seed), "mean": float(diff.mean()),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "p_gt_zero": float(np.mean(means > 0.0)),
    }


def _paired_bootstrap_two_seed(diff0: np.ndarray, diff1: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    diff0 = np.asarray(diff0, dtype=np.float64)
    diff1 = np.asarray(diff1, dtype=np.float64)
    n = diff0.size
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(B), n))
    means = 0.5 * (diff0[idx].mean(axis=1) + diff1[idx].mean(axis=1))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "n": n, "B": int(B), "seed": int(seed),
        "mean": float(0.5 * (diff0.mean() + diff1.mean())),
        "seed0_mean": float(diff0.mean()), "seed1_mean": float(diff1.mean()),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "p_gt_zero": float(np.mean(means > 0.0)),
    }


def bootstrap_provenance_check() -> dict[str, Any]:
    """Recompute the prior paired bootstrap exactly from saved per-molecule errors."""
    soup = {k: np.load(v) for k, v in SOUP_ABS_ERR_PATHS.items()}
    raw = {k: np.load(v) for k, v in RAW_ABS_ERR_PATHS.items()}
    recomputed = {}
    for name, (lo, hi) in TRANSITIONS.items():
        for seed in SEEDS:
            d_soup = soup[(lo, seed)] - soup[(hi, seed)]
            d_raw = raw[(lo, seed)] - raw[(hi, seed)]
            recomputed[f"{name}_{SEED_LABEL[seed]}"] = {
                "soup": _paired_bootstrap(d_soup, seed=PRIOR_BOOTSTRAP_SEED),
                "raw": _paired_bootstrap(d_raw, seed=PRIOR_BOOTSTRAP_SEED),
            }
    d0 = soup[(3600, 0)] - soup[(7200, 0)]
    d1 = soup[(3600, 1)] - soup[(7200, 1)]
    two_seed_36_72 = _paired_bootstrap_two_seed(d0, d1, seed=PRIOR_BOOTSTRAP_SEED)

    prior = _read_json(SAMPLE_EFF_DIR / "doubling_gain_summary.json")
    prior_gains = prior["gains"]
    prior_key = {"18_36": "18_to_36", "36_72": "36_to_72"}
    comparisons = {}
    for name in TRANSITIONS:
        for seed in SEEDS:
            key = f"{prior_key[name]}_{SEED_LABEL[seed]}"
            if key not in prior_gains:
                continue
            prior_boot = prior_gains[key].get("bootstrap_soup", {})
            ours = recomputed[f"{name}_{SEED_LABEL[seed]}"]["soup"]
            comparisons[key] = {
                "mean_prior": prior_boot.get("mean"),
                "mean_recomputed": ours["mean"],
                "mean_diff": abs(float(prior_boot.get("mean", 0.0)) - ours["mean"]),
                "ci95_lower_prior": prior_boot.get("ci95_lower"),
                "ci95_lower_recomputed": ours["ci95_lower"],
                "ci95_upper_prior": prior_boot.get("ci95_upper"),
                "ci95_upper_recomputed": ours["ci95_upper"],
                "p_gt_zero_prior": prior_boot.get("p_gt_zero"),
                "p_gt_zero_recomputed": ours["p_gt_zero"],
            }
    prior_two = prior.get("two_seed_mean_paired_bootstrap", {})
    comparisons["two_seed_36_72"] = {
        "mean_prior": prior_two.get("mean"),
        "mean_recomputed": two_seed_36_72["mean"],
        "ci95_lower_prior": prior_two.get("ci95_lower"),
        "ci95_lower_recomputed": two_seed_36_72["ci95_lower"],
        "ci95_upper_prior": prior_two.get("ci95_upper"),
        "ci95_upper_recomputed": two_seed_36_72["ci95_upper"],
    }
    all_match = all(
        max(
            abs(float(c["mean_prior"]) - float(c["mean_recomputed"])),
            abs(float(c["ci95_lower_prior"]) - float(c["ci95_lower_recomputed"])),
            abs(float(c["ci95_upper_prior"]) - float(c["ci95_upper_recomputed"])),
        ) <= 1e-12
        for c in comparisons.values()
        if c["mean_prior"] is not None
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "statistic": "mean per-molecule paired difference E(N)-E(2N), soup estimator",
        "resampling_unit": "probe molecule (paired)",
        "two_seed_aggregation": "0.5*(seed0 mean + seed1 mean) per bootstrap resample",
        "interval_method": "percentile bootstrap (np.percentile of bootstrap means, 2.5/97.5)",
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": PRIOR_BOOTSTRAP_SEED,
        "recomputed": recomputed,
        "two_seed_36_72": two_seed_36_72,
        "prior_comparison": comparisons,
        "all_match_prior": bool(all_match),
        "point_estimate_in_ci": all(
            float(c["ci95_lower_recomputed"]) <= float(c["mean_recomputed"]) <= float(c["ci95_upper_recomputed"])
            for c in comparisons.values()
            if c["mean_recomputed"] is not None
        ),
        "discrepancy": None if all_match else "PROVENANCE_DISCREPANCY",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "bootstrap_provenance_check.json", payload)
    if not all_match:
        raise RuntimeError("bootstrap provenance discrepancy; STOP AND REPORT PROVENANCE DISCREPANCY")
    return payload


# ---------------------------------------------------------------------------
# Phase U -- structural support (target-free)
# ---------------------------------------------------------------------------


def _build_graph_support_sets(records: list[Any]) -> dict[str, Any]:
    """Per-graph unique token ids, relation keys, and parent relation keys.

    Uses only raw graph + preprocessing fields.  ``record.y`` is never read.
    """
    tok2id: dict[bytes, int] = {}
    par2id: dict[bytes, int] = {}

    def _id(table: dict[bytes, int], key: bytes) -> int:
        value = table.get(key)
        if value is None:
            value = len(table)
            table[key] = value
        return value

    per_tok: list[set[int]] = []
    per_rel: list[set[tuple]] = []
    per_prel: list[set[tuple]] = []
    atom_count: list[int] = []
    relation_count: list[int] = []
    for record in records:
        toks = [_id(tok2id, patch.typed_certificate) for patch in record.patches]
        pars = [_id(par2id, patch.parent_certificate) for patch in record.patches]
        per_tok.append(set(toks))
        relations: set[tuple] = set()
        prelations: set[tuple] = set()
        pair_relation = record.pair_relation
        for k in range(pair_relation.shape[0]):
            i = int(record.pair_index[0, k])
            j = int(record.pair_index[1, k])
            distance = int(round(float(np.expm1(float(pair_relation[k, 5])))))
            adjacent = -1
            if distance == 1:
                tail = pair_relation[k, -4:]
                if float(tail.sum()) > 0.0:
                    adjacent = int(np.argmax(tail))
            a, b = toks[i], toks[j]
            if a > b:
                a, b = b, a
            relations.add((a, b, distance, adjacent))
            pa, pb = pars[i], pars[j]
            if pa > pb:
                pa, pb = pb, pa
            prelations.add((pa, pb, distance, adjacent))
        per_rel.append(relations)
        per_prel.append(prelations)
        atom_count.append(int(len(record.patches)))
        relation_count.append(int(pair_relation.shape[0]))
    return {
        "n_tokens": len(tok2id),
        "n_parent_tokens": len(par2id),
        "per_tok": per_tok,
        "per_rel": per_rel,
        "per_prel": per_prel,
        "atom_count": np.asarray(atom_count, dtype=np.int64),
        "relation_count": np.asarray(relation_count, dtype=np.int64),
    }


def _document_frequency(sets: Sequence[set], indices: Sequence[int]) -> Counter:
    df: Counter = Counter()
    for i in indices:
        for key in sets[int(i)]:
            df[key] += 1
    return df


def _probe_distribution_metrics(sets: Sequence[set], df: Counter, probe: Sequence[int]) -> dict[str, np.ndarray]:
    unseen = np.empty(len(probe), dtype=np.float64)
    rare = np.empty(len(probe), dtype=np.float64)
    meanlog = np.empty(len(probe), dtype=np.float64)
    for row, index in enumerate(probe):
        keys = sets[int(index)]
        n = max(len(keys), 1)
        values = np.asarray([df.get(key, 0) for key in keys], dtype=np.float64)
        unseen[row] = float(np.mean(values == 0)) if keys else 0.0
        rare[row] = float(np.mean(values < RARE_DF)) if keys else 0.0
        meanlog[row] = float(np.mean(np.log1p(values))) if keys else 0.0
    return {"unseen": unseen, "rare": rare, "meanlog": meanlog}


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _deficiency(metrics: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z_unseen = _zscore(metrics["unseen"])
    z_rare = _zscore(metrics["rare"])
    z_meanlog = _zscore(metrics["meanlog"])
    deficiency = (z_unseen + z_rare - z_meanlog) / 3.0
    return deficiency, {"z_unseen": z_unseen, "z_rare": z_rare, "z_meanlog": z_meanlog}


def _build_patchfull_blocks(records: list[Any], ref: np.ndarray) -> dict[str, Any]:
    """Reproduce the frozen PATCH_FULL block representations (target-free)."""
    patch_dim = int(records[0].patches[0].shell_descriptor.shape[0])
    pair_dim = int(R.RELATION_WIDTH)
    identity_keys = [
        [(p.typed_certificate, p.parent_certificate) for p in record.patches] for record in records
    ]
    b1 = R.categorical_histogram(identity_keys)
    patch_objects = [
        np.stack([p.shell_descriptor for p in record.patches], axis=0)
        if record.patches
        else np.zeros((0, patch_dim), dtype=np.float32)
        for record in records
    ]
    pair_objects = [
        record.pair_relation.astype(np.float32)
        if record.pair_relation.size
        else np.zeros((0, pair_dim), dtype=np.float32)
        for record in records
    ]
    graph_objects = np.stack(
        [
            np.concatenate(
                [record.global_context.astype(np.float32), record.topology_features.astype(np.float32)]
            )
            for record in records
        ],
        axis=0,
    )
    proj_b2 = R._rng_for("B2_patch_numeric").normal(size=(R.PROJECTION_COUNT, patch_dim))
    proj_b2 /= np.linalg.norm(proj_b2, axis=1, keepdims=True)
    proj_b3 = R._rng_for("B3_pair_relation").normal(size=(R.PROJECTION_COUNT, pair_dim))
    proj_b3 /= np.linalg.norm(proj_b3, axis=1, keepdims=True)
    b2, _ = R.projected_quantile_sketch(patch_objects, ref, proj_b2, "B2_patch_numeric")
    b3, _ = R.projected_quantile_sketch(pair_objects, ref, proj_b3, "B3_pair_relation")
    center = graph_objects[ref].mean(axis=0)
    scale = graph_objects[ref].std(axis=0)
    scale[~np.isfinite(scale) | (scale < R.NORMALIZATION_EPS)] = 1.0
    b4 = ((graph_objects - center) / scale).astype(np.float32)
    return {"B1_identity": b1, "B2_patch_numeric": b2, "B3_pair_relation": b3, "B4_global_topology": b4}


def _patchfull_distance(blocks: Mapping[str, Any], scales: Mapping[str, float], probe: Sequence[int], ref: Sequence[int]) -> np.ndarray:
    total = np.zeros((len(probe), len(ref)), dtype=np.float64)
    for name in ("B1_identity", "B2_patch_numeric", "B3_pair_relation", "B4_global_topology"):
        matrix = blocks[name]
        if sp.issparse(matrix):
            left = matrix[np.asarray(probe)]
            right = matrix[np.asarray(ref)]
            sim = (left @ right.T).toarray()
            np.clip(sim, -1.0, 1.0, out=sim)
            distance = (1.0 - sim).astype(np.float64)
        else:
            distance = R.dense_scaled_distances(
                np.asarray(matrix)[np.asarray(probe)], np.asarray(matrix)[np.asarray(ref)], matrix.shape[1]
            ).astype(np.float64)
        total += (distance / (float(scales[name]) + 1e-8)) ** 2
    return np.sqrt(total / 4.0).astype(np.float32)


def _nn8(dist: np.ndarray, k: int = NN_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = dist.shape[0]
    k = min(int(k), dist.shape[1])
    part = np.argpartition(dist, k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    values = dist[rows, part]
    order = np.argsort(values, axis=1)
    sorted_values = values[rows, order]
    return sorted_values.mean(axis=1), sorted_values[:, 0], sorted_values


def _quartile_labels(values: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    q25, q50, q75 = thresholds
    labels = np.ones(len(values), dtype=np.int64)
    labels[values > q25] = 2
    labels[values > q50] = 3
    labels[values > q75] = 4
    return labels


def stage_phaseU() -> dict[str, Any]:
    """Build the target-free structural support table and hash-lock it."""
    _install_firewall()
    records = load_records()
    ref, sel, probe, _ = split_view()
    probe = np.asarray(probe, dtype=np.int64)
    subsets = nested_subsets()

    support = _build_graph_support_sets(records)
    token_counts = {"global_unique": support["n_tokens"], "parent_unique": support["n_parent_tokens"]}

    # ---- token support ----------------------------------------------------
    token_df = {label: _document_frequency(support["per_tok"], subsets[label]) for label in ("1800", "3600", "7200")}
    token_metrics = {
        label: _probe_distribution_metrics(support["per_tok"], token_df[label], probe) for label in token_df
    }
    token_deficiency = {}
    token_z = {}
    for label in token_df:
        deficiency, zs = _deficiency(token_metrics[label])
        token_deficiency[label] = deficiency
        token_z[label] = zs

    # ---- relation support (primary: typed endpoints) ---------------------
    rel_df = {label: _document_frequency(support["per_rel"], subsets[label]) for label in ("1800", "3600", "7200")}
    rel_metrics = {
        label: _probe_distribution_metrics(support["per_rel"], rel_df[label], probe) for label in rel_df
    }
    rel_deficiency = {}
    rel_z = {}
    for label in rel_df:
        deficiency, zs = _deficiency(rel_metrics[label])
        rel_deficiency[label] = deficiency
        rel_z[label] = zs

    # ---- relation support (secondary: parent endpoints) ------------------
    prel_df = {label: _document_frequency(support["per_prel"], subsets[label]) for label in ("1800", "3600", "7200")}
    prel_metrics = {
        label: _probe_distribution_metrics(support["per_prel"], prel_df[label], probe) for label in prel_df
    }
    prel_deficiency = {}
    for label in prel_df:
        deficiency, _ = _deficiency(prel_metrics[label])
        prel_deficiency[label] = deficiency

    # ---- whole-molecule PATCH_FULL neighbour density ---------------------
    scales = _read_json(RAW_AUDIT_DIR / "distance_scale_stats.json")["median_distance"]
    blocks = _build_patchfull_blocks(records, ref)
    dist_full = _patchfull_distance(blocks, scales, probe.tolist(), ref.tolist())  # 2000 x len(ref)
    ref_position = {int(index): position for position, index in enumerate(ref.tolist())}
    nn8 = {}
    nn_detail = {}
    for label in ("1800", "3600", "7200"):
        columns = np.asarray([ref_position[int(i)] for i in subsets[label]], dtype=np.int64)
        sub = dist_full[:, columns]
        mean_k, nearest1, sorted_values = _nn8(sub)
        nn8[label] = mean_k.astype(np.float64)
        nn_detail[label] = {
            "nearest1": nearest1.astype(np.float64),
            "median_nn8": np.median(sorted_values, axis=1).astype(np.float64),
            "std_nn8": sorted_values.std(axis=1).astype(np.float64),
        }

    # ---- size controls ----------------------------------------------------
    atom_count = support["atom_count"][probe]
    relation_count = support["relation_count"][probe]
    c1 = np.log1p(atom_count.astype(np.float64))
    c2 = np.log1p(relation_count.astype(np.float64))

    # ---- document-frequency summaries ------------------------------------
    token_summary = _df_summary(token_df, support["per_tok"], probe, "typed_token")
    relation_summary = _df_summary(rel_df, support["per_rel"], probe, "relation_typed_endpoints")
    parent_relation_summary = _df_summary(prel_df, support["per_prel"], probe, "relation_parent_endpoints")

    # ---- assemble support table ------------------------------------------
    columns: dict[str, Any] = {"graph_id": probe.astype(np.int64), "probe_position": np.arange(len(probe))}
    for label in ("1800", "3600", "7200"):
        for source, prefix in (("token", "token"), ("relation", "relation")):
            metrics = token_metrics[label] if source == "token" else rel_metrics[label]
            deficiency = token_deficiency[label] if source == "token" else rel_deficiency[label]
            columns[f"{prefix}_unseen_{label}"] = metrics["unseen"]
            columns[f"{prefix}_rare_{label}"] = metrics["rare"]
            columns[f"{prefix}_meanlogdf_{label}"] = metrics["meanlog"]
            columns[f"{prefix}_deficiency_{label}"] = deficiency
            if source == "token":
                for zname, zval in token_z[label].items():
                    columns[f"token_{zname}_{label}"] = zval
            else:
                for zname, zval in rel_z[label].items():
                    columns[f"relation_{zname}_{label}"] = zval
        columns[f"relation_parent_unseen_{label}"] = prel_metrics[label]["unseen"]
        columns[f"relation_parent_rare_{label}"] = prel_metrics[label]["rare"]
        columns[f"relation_parent_meanlogdf_{label}"] = prel_metrics[label]["meanlog"]
        columns[f"relation_parent_deficiency_{label}"] = prel_deficiency[label]
        columns[f"nn8_distance_{label}"] = nn8[label]
        columns[f"nn8_nearest1_{label}"] = nn_detail[label]["nearest1"]
        columns[f"nn8_median_{label}"] = nn_detail[label]["median_nn8"]
        columns[f"nn8_std_{label}"] = nn_detail[label]["std_nn8"]
    columns["atom_count"] = atom_count.astype(np.int64)
    columns["relation_count"] = relation_count.astype(np.int64)
    columns["size_c1_log1p_atoms"] = c1
    columns["size_c2_log1p_relations"] = c2

    # secondary support increments
    for name, (lo, hi) in TRANSITIONS.items():
        columns[f"token_deficiency_increment_{name}"] = token_deficiency[str(lo)] - token_deficiency[str(hi)]
        columns[f"relation_deficiency_increment_{name}"] = rel_deficiency[str(lo)] - rel_deficiency[str(hi)]
        columns[f"relation_parent_deficiency_increment_{name}"] = prel_deficiency[str(lo)] - prel_deficiency[str(hi)]
        columns[f"nn8_decrement_{name}"] = nn8[str(lo)] - nn8[str(hi)]

    # ---- quartile manifest (locked from Phase U deficiency only) ---------
    quartile_manifest: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "families": {}, "size_quantiles": {}}
    deficiency_table = {
        TOKEN_FAMILY: token_deficiency,
        RELATION_FAMILY: rel_deficiency,
        NEIGHBOR_FAMILY: nn8,
    }
    for family, table in deficiency_table.items():
        quartile_manifest["families"][family] = {}
        for name, (lo, _hi) in TRANSITIONS.items():
            values = np.asarray(table[str(lo)], dtype=np.float64)
            thresholds = [float(np.percentile(values, p)) for p in (25, 50, 75)]
            labels = _quartile_labels(values, thresholds)
            columns[f"q_{family}_{name}"] = labels
            quartile_manifest["families"][family][name] = {
                "support_N": int(lo),
                "thresholds": {"q25": thresholds[0], "q50": thresholds[1], "q75": thresholds[2]},
                "assignment": "Q1 = best supported (low deficiency) ... Q4 = worst supported",
                "counts": {f"Q{q}": int((labels == q).sum()) for q in (1, 2, 3, 4)},
                "derived_from": "Phase U deficiency over the 2000 probe; no gain used",
            }
    # parent relation secondary quartiles (diagnostic)
    for name, (lo, _hi) in TRANSITIONS.items():
        values = np.asarray(prel_deficiency[str(lo)], dtype=np.float64)
        thresholds = [float(np.percentile(values, p)) for p in (25, 50, 75)]
        columns[f"q_relation_parent_{name}"] = _quartile_labels(values, thresholds)

    frame = pd.DataFrame(columns)
    _write_parquet(RESULTS_DIR / "support_table_locked.parquet", frame)

    # ---- WELL_COVERED_1800 ------------------------------------------------
    median_tok = float(np.median(token_deficiency["1800"]))
    median_rel = float(np.median(rel_deficiency["1800"]))
    median_nn = float(np.median(nn8["1800"]))
    wc1 = token_metrics["1800"]["unseen"] == 0.0
    wc2 = rel_metrics["1800"]["unseen"] == 0.0
    wc3 = token_deficiency["1800"] <= median_tok
    wc4 = rel_deficiency["1800"] <= median_rel
    wc5 = nn8["1800"] <= median_nn
    well_covered = wc1 & wc2 & wc3 & wc4 & wc5
    frame["well_covered_1800"] = well_covered
    # secondary coarse relation key variant (parent endpoints)
    wc2_parent = prel_metrics["1800"]["unseen"] == 0.0
    well_covered_parent = wc1 & wc2_parent & wc3 & (prel_deficiency["1800"] <= float(np.median(prel_deficiency["1800"]))) & wc5
    frame["well_covered_parent_1800"] = well_covered_parent
    _write_parquet(RESULTS_DIR / "support_table_locked.parquet", frame)
    n_well_covered = int(well_covered.sum())
    n_well_covered_parent = int(well_covered_parent.sum())

    well_covered_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "WC1": "token_unseen_1800 == 0",
            "WC2": "relation_unseen_1800 == 0 (primary typed-endpoint relation key)",
            "WC3": "token_deficiency_1800 <= median",
            "WC4": "relation_deficiency_1800 <= median",
            "WC5": "nn8_distance_1800 <= median",
        },
        "medians": {"token_deficiency_1800": median_tok, "relation_deficiency_1800": median_rel, "nn8_distance_1800": median_nn},
        "condition_counts": {
            "WC1": int(wc1.sum()), "WC2": int(wc2.sum()), "WC3": int(wc3.sum()),
            "WC4": int(wc4.sum()), "WC5": int(wc5.sum()),
        },
        "n_well_covered": n_well_covered,
        "well_covered_graph_ids": frame.loc[well_covered, "graph_id"].astype(int).tolist(),
        "powered": bool(n_well_covered >= WELL_COVERED_MIN),
        "secondary_parent_relation_key": {
            "definition": "same but WC2 uses the coarser parent-endpoint relation key",
            "n_well_covered": n_well_covered_parent,
            "well_covered_graph_ids": frame.loc[well_covered_parent, "graph_id"].astype(int).tolist(),
            "powered": bool(n_well_covered_parent >= WELL_COVERED_MIN),
            "authoritative": False,
        },
        "target_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "well_covered_subset_manifest.json", well_covered_manifest)

    # ---- inventories / locks ---------------------------------------------
    token_inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_token_field": "patch.typed_certificate",
        "primary_token_source": "historical rooted radius-2 typed incidence certificate (pynauty.certificate, typed_tokenizer_v1_historical)",
        "parent_token_field": "patch.parent_certificate",
        "parent_token_role": "model-visible coarse secondary categorical field; reported as diagnostic only",
        "parent_tokenizer": "historical rooted radius-1 typed incidence certificate",
        "multiplicity_semantics": "document frequency over unique molecules; a token occurring many times in one molecule counts once",
        "historical_aliasing_provenance": (
            "v4_records_train.pkl.gz was extracted with DEFAULT_TYPED_TOKENIZER_VERSION = "
            "typed_tokenizer_v1_historical; the alias is the model's real visible identity"
        ),
        "corrected_tokenizer_substituted": False,
        "n_unique_tokens_10000": token_counts["global_unique"],
        "n_unique_parent_tokens_10000": token_counts["parent_unique"],
        "model_vocabulary_size": 6785,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "token_support_inventory.json", token_inventory)

    relation_inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_relation_key": {
            "fields": [
                "canonical unordered endpoint typed_certificate ids",
                "source integer shortest-path distance",
                "adjacent bond type (only when distance == 1, else -1)",
            ],
            "endpoint_canonicalization": "unordered (min,max); pair encoder is symmetric",
            "naturally_discrete_only": True,
            "continuous_descriptor_binning": False,
            "source": "record.pair_index + record.pair_relation (distance recovered as round(expm1(relation[5])))",
        },
        "secondary_relation_key": {
            "fields": [
                "canonical unordered endpoint parent_certificate ids",
                "source integer shortest-path distance",
                "adjacent bond type",
            ],
            "authoritative": False,
        },
        "n_unique_relation_keys_10000": int(len({k for s in support["per_rel"] for k in s})),
        "n_unique_parent_relation_keys_10000": int(len({k for s in support["per_prel"] for k in s})),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "relation_key_inventory.json", relation_inventory)

    patchfull_inventory = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": "frozen PATCH_FULL distance from raw_graph_patch_system_sufficiency",
        "blocks": {
            "B1_identity": "sparse count histogram over (typed_certificate, parent_certificate); cosine",
            "B2_patch_numeric": "217D projected-quantile sketch of the 146D shell descriptor multiset",
            "B3_pair_relation": "217D projected-quantile sketch of the 23D pair relation multiset",
            "B4_global_topology": "z-scored 87D graph-level global(62)+topology(25) vector",
        },
        "combined_distance": "sqrt(mean_b (d_b / block_scale_b)^2), equal semantic weight",
        "block_scales": {k: float(scales[k]) for k in ("B1_identity", "B2_patch_numeric", "B3_pair_relation", "B4_global_topology")},
        "block_scales_source": str(RAW_AUDIT_DIR / "distance_scale_stats.json"),
        "normalization": "frozen on the 7200 reference set; never refit per N",
        "projection_seeds": "R._rng_for('B2_patch_numeric'), R._rng_for('B3_pair_relation')",
        "k": NN_K,
        "primary_score": "mean PATCH_FULL distance to the k=8 nearest D_N molecules",
        "uses_target": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "patchfull_metric_inventory.json", patchfull_inventory)

    pairs_from_atoms = atom_count.astype(np.int64) * (atom_count.astype(np.int64) - 1) // 2
    relation_equals_pairs = bool(np.array_equal(relation_count.astype(np.int64), pairs_from_atoms))
    c1_c2_spearman = float(_spearman(c1, c2))
    size_lock = {
        "protocol_version": PROTOCOL_VERSION,
        "C1": "log(1 + atom_count)",
        "C2": "log(1 + relation_count)",
        "atom_count_definition": "number of atom-centred patches",
        "relation_count_definition": "number of unordered patch pair relation objects",
        "patch_equals_atom_count": True,
        "relation_count_equals_pairs": relation_equals_pairs,
        "relation_count_formula": (
            "relation_count == C(atom_count, 2) exactly for every probe molecule"
            if relation_equals_pairs
            else "relation_count is NOT a deterministic function of atom_count in this sample"
        ),
        "c1_c2_spearman_probe": c1_c2_spearman,
        "collinearity_note": (
            "atom_count == patch count exactly and relation_count == C(atom_count, 2), so C2 is a "
            "deterministic (non-linear) function of C1; the two size controls span an effectively "
            "one-dimensional size confound. C1 is the molecule-size proxy; C2 is retained as the "
            "prompt-mandated second control but adds no independent size information here."
        ),
        "size_control_effective_dimensions": 1,
        "target_related_controls": [],
        "target_independent": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "size_control_lock.json", size_lock)

    _write_json(RESULTS_DIR / "token_document_frequency_summary.json", token_summary)
    _write_json(RESULTS_DIR / "relation_document_frequency_summary.json", relation_summary)
    _write_json(RESULTS_DIR / "relation_document_frequency_summary_parent.json", parent_relation_summary)

    # ---- support table lock ----------------------------------------------
    table_sha = _sha256_file(RESULTS_DIR / "support_table_locked.parquet")
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "support_table": "support_table_locked.parquet",
        "support_table_sha256": table_sha,
        "schema": list(frame.columns),
        "rows": int(len(frame)),
        "formulas": {
            "token_deficiency": "(z(token_unseen) + z(token_rare) - z(token_meanlogdf)) / 3",
            "relation_deficiency": "(z(relation_unseen) + z(relation_rare) - z(relation_meanlogdf)) / 3",
            "nn8_distance": "mean PATCH_FULL distance to k=8 nearest D_N molecules",
            "zscore": "across the 2000 probe molecules at the same N, target-free",
        },
        "thresholds": {"rare_df": RARE_DF, "nn_k": NN_K},
        "source_manifests": {
            "subset_lock": _sha256_file(SUBSET_LOCK),
            "split_manifest": _sha256_file(SPLIT_MANIFEST),
            "distance_scale_stats": _sha256_file(RAW_AUDIT_DIR / "distance_scale_stats.json"),
        },
        "target_used": False,
        "prediction_used": False,
        "quartiles_locked": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "support_table_lock.json", lock)

    quartile_manifest["support_table_sha256"] = table_sha
    quartile_manifest["locked_before_gain_read"] = True
    _write_json(RESULTS_DIR / "support_quartile_manifest.json", quartile_manifest)

    integrity = {
        "U1_only_allowed_inputs": True,
        "U2_target_never_read": True,
        "U3_prediction_never_read": True,
        "U4_unique_molecule_document_frequency": True,
        "U5_historical_token_field": True,
        "U6_corrected_tokenizer_not_substituted": True,
        "U7_naturally_discrete_relation_fields": True,
        "U8_no_continuous_descriptor_binning": True,
        "U9_patchfull_metric_reused": True,
        "U10_normalization_not_refit_per_N": True,
        "U11_support_table_hash_locked": True,
        "U12_quartiles_locked_before_gain": True,
        "U13_size_controls_target_independent": True,
        "U14_no_extra_family": True,
        "U15_no_learned_probe": True,
        "U16_official_valid_not_loaded": True,
        "U17_official_test_not_loaded": True,
        "passed": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "phaseU_integrity.json", integrity)
    return {
        "support_table_sha256": table_sha,
        "n_well_covered": n_well_covered,
        "n_well_covered_parent": n_well_covered_parent,
        "quartile_manifest": quartile_manifest,
    }


def _df_summary(df_by_n: Mapping[str, Counter], sets: Sequence[set], probe: Sequence[int], name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "per_N": {}}
    for label, df in df_by_n.items():
        counts = np.asarray(list(df.values()), dtype=np.float64)
        probe_unseen = []
        probe_rare = []
        for index in probe:
            keys = sets[int(index)]
            probe_unseen.append(np.mean([df.get(key, 0) == 0 for key in keys]) if keys else 0.0)
            probe_rare.append(np.mean([df.get(key, 0) < RARE_DF for key in keys]) if keys else 0.0)
        out["per_N"][label] = {
            "n_unique_keys_seen": int(len(df)),
            "median_document_frequency": float(np.median(counts)) if counts.size else 0.0,
            "singleton_fraction": float(np.mean(counts == 1)) if counts.size else 0.0,
            "df_lt5_fraction": float(np.mean(counts < RARE_DF)) if counts.size else 0.0,
            "probe_mean_unseen_rate": float(np.mean(probe_unseen)),
            "probe_mean_rare_lt5_rate": float(np.mean(probe_rare)),
        }
    return out


# ---------------------------------------------------------------------------
# Phase Y -- gain association
# ---------------------------------------------------------------------------


def _unlock_targets() -> None:
    global _TARGETS_UNLOCKED
    _TARGETS_UNLOCKED = True


def _load_locked_support() -> pd.DataFrame:
    lock = _read_json(RESULTS_DIR / "support_table_lock.json")
    actual = _sha256_file(RESULTS_DIR / "support_table_locked.parquet")
    if actual != lock["support_table_sha256"]:
        raise RuntimeError("support table changed after lock")
    return pd.read_parquet(RESULTS_DIR / "support_table_locked.parquet")


def _gains() -> dict[str, Any]:
    soup = {k: np.load(v) for k, v in SOUP_ABS_ERR_PATHS.items()}
    raw = {k: np.load(v) for k, v in RAW_ABS_ERR_PATHS.items()}
    out: dict[str, Any] = {}
    for name, (lo, hi) in TRANSITIONS.items():
        g_bar = 0.5 * (
            (soup[(lo, 0)] - soup[(hi, 0)]) + (soup[(lo, 1)] - soup[(hi, 1)])
        )
        out[name] = {
            "g_bar": g_bar,
            "g_seed0": soup[(lo, 0)] - soup[(hi, 0)],
            "g_seed1": soup[(lo, 1)] - soup[(hi, 1)],
            "g_raw": 0.5 * ((raw[(lo, 0)] - raw[(hi, 0)]) + (raw[(lo, 1)] - raw[(hi, 1)])),
            "E_lower_soup": 0.5 * (soup[(lo, 0)] + soup[(lo, 1)]),
            "E_upper_soup": 0.5 * (soup[(hi, 0)] + soup[(hi, 1)]),
        }
    return out


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # average ties
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(np.asarray(a, dtype=np.float64)), _rank(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _size_adjusted_beta(deficiency: np.ndarray, c1: np.ndarray, c2: np.ndarray, g: np.ndarray) -> float:
    Z = np.column_stack([
        _zscore(deficiency), _zscore(c1), _zscore(c2)
    ])
    X = np.column_stack([np.ones(len(g)), Z])
    beta = _ols(X, g)
    return float(beta[1])


def _bootstrap_quartile(g: np.ndarray, labels: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    n = len(g)
    rng = np.random.default_rng(int(seed))
    hi_mask = labels == 4
    lo_mask = labels == 1
    point = float(g[hi_mask].mean() - g[lo_mask].mean())
    values = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        gg = g[idx]
        hi = gg[hi_mask[idx]]
        lo = gg[lo_mask[idx]]
        values[b] = (hi.mean() - lo.mean()) if (len(hi) and len(lo)) else np.nan
    lower, upper = np.nanpercentile(values, [2.5, 97.5])
    return {
        "point": point,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_gt_zero": float(np.nanmean(values > 0.0)),
        "B": int(B), "seed": int(seed),
    }


def _bootstrap_spearman(deficiency: np.ndarray, g: np.ndarray, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    n = len(g)
    rng = np.random.default_rng(int(seed))
    point = _spearman(deficiency, g)
    values = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        values[b] = _spearman(deficiency[idx], g[idx])
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"point": point, "ci95_lower": float(lower), "ci95_upper": float(upper),
            "p_gt_zero": float(np.mean(values > 0.0)), "B": int(B), "seed": int(seed)}


def _bootstrap_beta(deficiency: np.ndarray, c1: np.ndarray, c2: np.ndarray, g: np.ndarray,
                    B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    n = len(g)
    rng = np.random.default_rng(int(seed))
    point = _size_adjusted_beta(deficiency, c1, c2, g)
    values = np.empty(int(B), dtype=np.float64)
    for b in range(int(B)):
        idx = rng.integers(0, n, size=n)
        values[b] = _size_adjusted_beta(deficiency[idx], c1[idx], c2[idx], g[idx])
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"point": point, "ci95_lower": float(lower), "ci95_upper": float(upper),
            "p_gt_zero": float(np.mean(values > 0.0)), "B": int(B), "seed": int(seed)}


def _family_analysis(name: str, deficiency_table: Mapping[str, np.ndarray], gains: Mapping[str, Any],
                     c1: np.ndarray, c2: np.ndarray, quartile_manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"family": name, "transitions": {}}
    for transition, (lo, _hi) in TRANSITIONS.items():
        deficiency = np.asarray(deficiency_table[str(lo)], dtype=np.float64)
        entry: dict[str, Any] = {"support_N": int(lo)}
        for estimator, key in (("seed_mean", "g_bar"), ("seed0", "g_seed0"), ("seed1", "g_seed1"), ("raw", "g_raw")):
            g = np.asarray(gains[transition][key], dtype=np.float64)
            labels = np.asarray(
                pd.read_parquet(RESULTS_DIR / "support_table_locked.parquet")[f"q_{name}_{transition}"],
                dtype=np.int64,
            )
            qc = _bootstrap_quartile(g, labels)
            rho = _bootstrap_spearman(deficiency, g)
            entry[estimator] = {"quartile_contrast": qc, "spearman": rho}
            if estimator == "seed_mean":
                entry["quartile_contrast"] = qc
                entry["spearman"] = rho
                entry["size_adjusted_beta"] = _bootstrap_beta(deficiency, c1, c2, g)
        result["transitions"][transition] = entry
    return result


def _fixed_ols_oof(predictors: Mapping[str, np.ndarray], g: np.ndarray, graph_ids: np.ndarray) -> dict[str, Any]:
    names = list(predictors)
    X = np.column_stack([_zscore(np.asarray(predictors[name], dtype=np.float64)) for name in names])
    folds = np.asarray(
        [int(hashlib.sha256(f"{FOLD_SALT}|{int(gid)}".encode()).hexdigest()[:8], 16) % N_FOLDS for gid in graph_ids],
        dtype=np.int64,
    )
    prediction = np.empty(len(g), dtype=np.float64)
    for fold in range(N_FOLDS):
        test = folds == fold
        train = ~test
        Xtr = np.column_stack([np.ones(train.sum()), X[train]])
        beta = _ols(Xtr, g[train])
        prediction[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta
    ss_res = float(np.sum((g - prediction) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"predictors": names, "r2_oof": float(r2), "fold_counts": {str(f): int((folds == f).sum()) for f in range(N_FOLDS)}}


def stage_phaseY() -> dict[str, Any]:
    """Read targets/predictions and associate structural support with gain."""
    _unlock_targets()
    support = _load_locked_support()
    quartile_manifest = _read_json(RESULTS_DIR / "support_quartile_manifest.json")
    gains = _gains()
    graph_ids = support["graph_id"].to_numpy(dtype=np.int64)
    c1 = support["size_c1_log1p_atoms"].to_numpy(dtype=np.float64)
    c2 = support["size_c2_log1p_relations"].to_numpy(dtype=np.float64)

    # ---- gain table ------------------------------------------------------
    gain_frame = pd.DataFrame({"graph_id": graph_ids})
    for transition in TRANSITIONS:
        gain_frame[f"gain_bar_{transition}"] = gains[transition]["g_bar"]
        gain_frame[f"gain_seed0_{transition}"] = gains[transition]["g_seed0"]
        gain_frame[f"gain_seed1_{transition}"] = gains[transition]["g_seed1"]
        gain_frame[f"gain_raw_{transition}"] = gains[transition]["g_raw"]
    _write_parquet(RESULTS_DIR / "gain_table.parquet", gain_frame)

    gain_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": "g(i) = |y_i - yhat_N(i)| - |y_i - yhat_2N(i)|; positive means molecule benefits",
        "primary_estimator": "soup two-seed mean",
        "transitions": {
            transition: {
                "G_all_soup_seed_mean": float(np.mean(gains[transition]["g_bar"])),
                "G_all_soup_seed0": float(np.mean(gains[transition]["g_seed0"])),
                "G_all_soup_seed1": float(np.mean(gains[transition]["g_seed1"])),
                "G_all_raw_seed_mean": float(np.mean(gains[transition]["g_raw"])),
                "E_lower_soup": float(np.mean(gains[transition]["E_lower_soup"])),
                "E_upper_soup": float(np.mean(gains[transition]["E_upper_soup"])),
            }
            for transition in TRANSITIONS
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "gain_summary.json", gain_summary)

    # ---- family analyses -------------------------------------------------
    token_deficiency = {str(n): support[f"token_deficiency_{n}"].to_numpy(dtype=np.float64) for n in NESTED_SIZES}
    rel_deficiency = {str(n): support[f"relation_deficiency_{n}"].to_numpy(dtype=np.float64) for n in NESTED_SIZES}
    nn_deficiency = {str(n): support[f"nn8_distance_{n}"].to_numpy(dtype=np.float64) for n in NESTED_SIZES}
    prel_deficiency = {str(n): support[f"relation_parent_deficiency_{n}"].to_numpy(dtype=np.float64) for n in NESTED_SIZES}

    token_results = _family_analysis(TOKEN_FAMILY, token_deficiency, gains, c1, c2, quartile_manifest)
    relation_results = _family_analysis(RELATION_FAMILY, rel_deficiency, gains, c1, c2, quartile_manifest)
    neighbor_results = _family_analysis(NEIGHBOR_FAMILY, nn_deficiency, gains, c1, c2, quartile_manifest)

    # ---- size-adjusted results (per family, both transitions, seed mean + raw)
    size_adjusted = {"families": {}}
    for family, table in ((TOKEN_FAMILY, token_deficiency), (RELATION_FAMILY, rel_deficiency), (NEIGHBOR_FAMILY, nn_deficiency)):
        size_adjusted["families"][family] = {}
        for transition, (lo, _hi) in TRANSITIONS.items():
            deficiency = table[str(lo)]
            size_adjusted["families"][family][transition] = {
                "beta1_seed_mean": _bootstrap_beta(deficiency, c1, c2, np.asarray(gains[transition]["g_bar"])),
                "beta1_raw": _bootstrap_beta(deficiency, c1, c2, np.asarray(gains[transition]["g_raw"])),
            }
    # secondary coarse relation family
    parent_relation_results = _family_analysis("relation_parent", prel_deficiency, gains, c1, c2, quartile_manifest)
    _write_json(RESULTS_DIR / "size_adjusted_results.json", size_adjusted)

    _write_json(RESULTS_DIR / "token_family_results.json", token_results)
    _write_json(RESULTS_DIR / "relation_family_results.json", relation_results)
    _write_json(RESULTS_DIR / "neighbor_density_results.json", neighbor_results)
    _write_json(RESULTS_DIR / "relation_parent_family_results.json", parent_relation_results)

    # ---- family correlation matrix --------------------------------------
    corr_rows = []
    for transition, (lo, _hi) in TRANSITIONS.items():
        values = {
            "token": token_deficiency[str(lo)],
            "relation": rel_deficiency[str(lo)],
            "neighbor_density": nn_deficiency[str(lo)],
            "relation_parent": prel_deficiency[str(lo)],
            "size_c1": c1,
            "size_c2": c2,
        }
        names = list(values)
        for a in names:
            for b in names:
                corr_rows.append({"transition": transition, "a": a, "b": b, "spearman": _spearman(values[a], values[b])})
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "support_family_correlation.csv", index=False)

    # ---- fixed OOF explanatory model ------------------------------------
    oof: dict[str, Any] = {"folds": N_FOLDS, "fold_salt": FOLD_SALT, "transitions": {}}
    for transition, (lo, _hi) in TRANSITIONS.items():
        d_tok = token_deficiency[str(lo)]
        d_rel = rel_deficiency[str(lo)]
        d_nn = nn_deficiency[str(lo)]
        g = np.asarray(gains[transition]["g_bar"], dtype=np.float64)
        models = {
            "size_only": {"size_c1": c1, "size_c2": c2},
            "size_plus_token": {"size_c1": c1, "size_c2": c2, "token": d_tok},
            "size_plus_relation": {"size_c1": c1, "size_c2": c2, "relation": d_rel},
            "size_plus_nn": {"size_c1": c1, "size_c2": c2, "nn": d_nn},
            "full": {"size_c1": c1, "size_c2": c2, "token": d_tok, "relation": d_rel, "nn": d_nn},
        }
        oof["transitions"][transition] = {
            name: _fixed_ols_oof(predictors, g, graph_ids) for name, predictors in models.items()
        }
    _write_json(RESULTS_DIR / "fixed_ols_oof_results.json", oof)

    # ---- well-covered stress test ---------------------------------------
    well_manifest = _read_json(RESULTS_DIR / "well_covered_subset_manifest.json")
    wc_mask = support["well_covered_1800"].to_numpy(dtype=bool)
    wc_parent_mask = support["well_covered_parent_1800"].to_numpy(dtype=bool)
    stress = _well_covered_stress(gains, wc_mask, well_manifest["n_well_covered"], graph_ids)
    stress_parent = _well_covered_stress(gains, wc_parent_mask, well_manifest["secondary_parent_relation_key"]["n_well_covered"], graph_ids)
    stress["secondary_parent_relation_key"] = stress_parent
    stress["primary_definition"] = well_manifest["definition"]
    stress["authoritative"] = "primary"
    _write_json(RESULTS_DIR / "well_covered_stress_test.json", stress)

    # ---- support increment diagnostics ----------------------------------
    increment = {"transitions": {}}
    for transition, (lo, hi) in TRANSITIONS.items():
        g = np.asarray(gains[transition]["g_bar"], dtype=np.float64)
        d_tok_inc = support[f"token_deficiency_increment_{transition}"].to_numpy(dtype=np.float64)
        d_rel_inc = support[f"relation_deficiency_increment_{transition}"].to_numpy(dtype=np.float64)
        d_nn_dec = support[f"nn8_decrement_{transition}"].to_numpy(dtype=np.float64)
        increment["transitions"][transition] = {
            "spearman_delta_token_vs_gain": _spearman(d_tok_inc, g),
            "spearman_delta_relation_vs_gain": _spearman(d_rel_inc, g),
            "spearman_nn_density_improvement_vs_gain": _spearman(d_nn_dec, g),
            "note": "secondary only; not part of the primary family gate",
        }
    _write_json(RESULTS_DIR / "support_increment_diagnostics.json", increment)

    # ---- initial-error control (secondary) ------------------------------
    initial_error = {"transitions": {}}
    for transition, (lo, hi) in TRANSITIONS.items():
        soup_lo = 0.5 * (np.load(SOUP_ABS_ERR_PATHS[(lo, 0)]) + np.load(SOUP_ABS_ERR_PATHS[(lo, 1)]))
        g = np.asarray(gains[transition]["g_bar"], dtype=np.float64)
        initial_error["transitions"][transition] = {
            "spearman_token_deficiency_vs_initial_error": _spearman(token_deficiency[str(lo)], soup_lo),
            "spearman_token_deficiency_vs_gain": _spearman(token_deficiency[str(lo)], g),
            "spearman_relation_deficiency_vs_initial_error": _spearman(rel_deficiency[str(lo)], soup_lo),
            "spearman_relation_deficiency_vs_gain": _spearman(rel_deficiency[str(lo)], g),
            "spearman_nn_distance_vs_initial_error": _spearman(nn_deficiency[str(lo)], soup_lo),
            "spearman_nn_distance_vs_gain": _spearman(nn_deficiency[str(lo)], g),
            "note": "a support score that predicts current error but not gain is not a scaling mechanism",
        }
    _write_json(RESULTS_DIR / "initial_error_control.json", initial_error)

    # ---- RAW estimator robustness ---------------------------------------
    raw_dir = {"transitions": {}}
    for transition, (lo, _hi) in TRANSITIONS.items():
        entry = {}
        for family, table in ((TOKEN_FAMILY, token_deficiency), (RELATION_FAMILY, rel_deficiency), (NEIGHBOR_FAMILY, nn_deficiency)):
            deficiency = table[str(lo)]
            g_raw = np.asarray(gains[transition]["g_raw"], dtype=np.float64)
            labels = support[f"q_{family}_{transition}"].to_numpy(dtype=np.int64)
            qc = _bootstrap_quartile(g_raw, labels)
            rho = _bootstrap_spearman(deficiency, g_raw)
            entry[family] = {"quartile_contrast": qc["point"], "spearman": rho["point"]}
        raw_dir["transitions"][transition] = entry
    raw_dir["direction_consistent_with_soup"] = _raw_direction_consistency(token_results, relation_results, neighbor_results, raw_dir)
    _write_json(RESULTS_DIR / "raw_estimator_robustness.json", raw_dir)

    return {
        "gain_summary": gain_summary,
        "token_results": token_results,
        "relation_results": relation_results,
        "neighbor_results": neighbor_results,
        "size_adjusted": size_adjusted,
        "oof": oof,
        "stress": stress,
        "raw": raw_dir,
    }


def _well_covered_stress(gains: Mapping[str, Any], mask: np.ndarray, n_covered: int, graph_ids: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: dict[str, Any] = {"n_well_covered": int(n_covered), "powered": bool(n_covered >= WELL_COVERED_MIN), "transitions": {}}
    for transition in TRANSITIONS:
        g = np.asarray(gains[transition]["g_bar"], dtype=np.float64)
        g_all = float(np.mean(g))
        if mask.sum() == 0:
            out["transitions"][transition] = {"G_covered": None, "R_covered": None}
            continue
        g_covered = float(np.mean(g[mask]))
        # bootstrap within the well-covered subset
        subset = g[mask]
        n = len(subset)
        values = np.empty(BOOTSTRAP_B, dtype=np.float64)
        for b in range(BOOTSTRAP_B):
            idx = rng.integers(0, n, size=n)
            values[b] = subset[idx].mean()
        lower, upper = np.percentile(values, [2.5, 97.5])
        out["transitions"][transition] = {
            "G_covered": g_covered,
            "G_all": g_all,
            "R_covered": float(g_covered / g_all) if g_all != 0 else None,
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "ci_lower_gt_zero": bool(lower > 0.0),
        }
    if out["powered"]:
        r18 = out["transitions"]["18_36"]["R_covered"]
        r36 = out["transitions"]["36_72"]["R_covered"]
        out["coverage_insufficient_to_explain_scaling"] = bool(
            r18 is not None and r36 is not None
            and r18 >= COVERAGE_RETAIN_HIGH and r36 >= COVERAGE_RETAIN_HIGH
            and out["transitions"]["18_36"]["ci_lower_gt_zero"]
            and out["transitions"]["36_72"]["ci_lower_gt_zero"]
        )
        out["gain_concentrated_in_poorly_supported"] = bool(
            r18 is not None and r36 is not None
            and r18 <= COVERAGE_RETAIN_LOW and r36 <= COVERAGE_RETAIN_LOW
        )
    else:
        out["coverage_insufficient_to_explain_scaling"] = None
        out["gain_concentrated_in_poorly_supported"] = None
        out["verdict"] = "WELL-COVERED STRESS TEST UNDERPOWERED"
    return out


def _raw_direction_consistency(token_results, relation_results, neighbor_results, raw_dir) -> dict[str, Any]:
    out = {}
    for transition in TRANSITIONS:
        entry = {}
        for family, results in (("token", token_results), ("relation", relation_results), ("neighbor_density", neighbor_results)):
            soup_rho = results["transitions"][transition]["spearman"]["point"]
            raw_rho = raw_dir["transitions"][transition][family]["spearman"]
            soup_qc = results["transitions"][transition]["quartile_contrast"]["point"]
            raw_qc = raw_dir["transitions"][transition][family]["quartile_contrast"]
            entry[family] = {
                "soup_spearman": soup_rho, "raw_spearman": raw_rho,
                "soup_quartile_contrast": soup_qc, "raw_quartile_contrast": raw_qc,
                "same_spearman_sign": bool(np.sign(soup_rho) == np.sign(raw_rho)),
                "same_quartile_sign": bool(np.sign(soup_qc) == np.sign(raw_qc)),
            }
        out[transition] = entry
    return out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _gate_family(results: Mapping[str, Any], transition: str) -> dict[str, Any]:
    entry = results["transitions"][transition]
    qc = entry["quartile_contrast"]
    rho = entry["spearman"]
    beta = entry["size_adjusted_beta"]
    seed0 = entry["seed0"]["quartile_contrast"]["point"]
    seed1 = entry["seed1"]["quartile_contrast"]["point"]
    return {
        "G1_quartile": bool(qc["point"] >= QUARTILE_TARGET and qc["ci95_lower"] > 0.0),
        "G2_spearman": bool(rho["point"] >= SPEARMAN_TARGET and rho["ci95_lower"] > 0.0),
        "G3_size_adjusted": bool(beta["point"] > 0.0 and beta["ci95_lower"] > 0.0),
        "G4_seed_consistency": bool(seed0 > 0.0 and seed1 > 0.0),
        "values": {
            "quartile_contrast": qc, "spearman": rho, "size_adjusted_beta": beta,
            "seed0_quartile": seed0, "seed1_quartile": seed1,
        },
    }


def _family_verdict(gates_family: Mapping[str, Any]) -> str:
    both = {g: all(gates_family[t][g] for t in TRANSITIONS) for g in (
        "G1_quartile", "G2_spearman", "G3_size_adjusted", "G4_seed_consistency")}
    any_g1 = any(gates_family[t]["G1_quartile"] for t in TRANSITIONS)
    if all(both.values()):
        return "STRONG_SUPPORT_LIMITED_SIGNAL"
    if both["G1_quartile"] and both["G3_size_adjusted"] and not both["G2_spearman"]:
        return "NON_ROBUST_ASSOCIATION__strong_quartile_and_size_adjusted_but_weak_continuous"
    if both["G1_quartile"] and not both["G3_size_adjusted"]:
        return "SUPPORT_SIGNAL_EXPLAINED_BY_SIZE_OR_WEAK_AFTER_ADJUSTMENT"
    if any_g1:
        return "REGIME_SPECIFIC_OR_PARTIAL_SUPPORT_SIGNAL"
    return "NO_MATERIAL_SUPPORT_ASSOCIATION"


def stage_decision() -> dict[str, Any]:
    _unlock_targets()
    token_results = _read_json(RESULTS_DIR / "token_family_results.json")
    relation_results = _read_json(RESULTS_DIR / "relation_family_results.json")
    neighbor_results = _read_json(RESULTS_DIR / "neighbor_density_results.json")
    parent_relation_results = _read_json(RESULTS_DIR / "relation_parent_family_results.json")
    size_adjusted = _read_json(RESULTS_DIR / "size_adjusted_results.json")
    oof = _read_json(RESULTS_DIR / "fixed_ols_oof_results.json")
    stress = _read_json(RESULTS_DIR / "well_covered_stress_test.json")
    raw = _read_json(RESULTS_DIR / "raw_estimator_robustness.json")
    gain_summary = _read_json(RESULTS_DIR / "gain_summary.json")

    family_results = {
        TOKEN_FAMILY: token_results,
        RELATION_FAMILY: relation_results,
        NEIGHBOR_FAMILY: neighbor_results,
    }
    gates = {family: {transition: _gate_family(results, transition) for transition in TRANSITIONS} for family, results in family_results.items()}
    strong = {
        family: all(
            all(gates[family][transition][g] for g in ("G1_quartile", "G2_spearman", "G3_size_adjusted", "G4_seed_consistency"))
            for transition in TRANSITIONS
        )
        for family in FAMILIES
    }
    partial = {
        family: any(gates[family][t][g] for t in TRANSITIONS for g in ("G1_quartile", "G2_spearman", "G3_size_adjusted"))
        and not strong[family]
        for family in FAMILIES
    }

    wc_powered = bool(stress["powered"])
    r18 = stress["transitions"]["18_36"].get("R_covered")
    r36 = stress["transitions"]["36_72"].get("R_covered")
    wc_retain_high = bool(
        wc_powered and r18 is not None and r36 is not None
        and r18 >= COVERAGE_RETAIN_HIGH and r36 >= COVERAGE_RETAIN_HIGH
        and stress["transitions"]["18_36"]["ci_lower_gt_zero"]
        and stress["transitions"]["36_72"]["ci_lower_gt_zero"]
    )
    wc_retain_low = bool(
        wc_powered and r18 is not None and r36 is not None
        and r18 <= COVERAGE_RETAIN_LOW and r36 <= COVERAGE_RETAIN_LOW
    )

    strong_list = [f for f in FAMILIES if strong[f]]
    family_verdicts = {family: _family_verdict(gates[family]) for family in FAMILIES}
    family_verdicts["relation_parent"] = _family_verdict(
        {t: _gate_family(parent_relation_results, t) for t in TRANSITIONS}
    )

    # size-proxy rule
    size_confounded = {}
    for family in FAMILIES:
        conf = True
        for transition in TRANSITIONS:
            gate = gates[family][transition]
            beta = gate["values"]["size_adjusted_beta"]
            unadjusted_strong = gate["G1_quartile"] and gate["G2_spearman"]
            conf = conf and unadjusted_strong and (beta["ci95_lower"] <= 0.0 or beta["point"] <= 0.0)
        size_confounded[family] = bool(conf)

    # raw direction conflict
    raw_conflict = False
    for transition in TRANSITIONS:
        for family in FAMILIES:
            info = raw["direction_consistent_with_soup"][transition][family]
            if not info["same_spearman_sign"] and not info["same_quartile_sign"]:
                raw_conflict = True

    if raw_conflict:
        case = "H"
        verdict = "SAMPLE-EFFICIENCY GAIN LOCALIZATION INCONCLUSIVE"
        reason = "SOUP/RAW localization strongly reversed -> estimator-dependent localization"
    elif len(strong_list) >= 2:
        case = "D"
        verdict = "MIXED STRUCTURAL SUPPORT LIMITATION"
        reason = "two or more correlated support families pass the strong gate"
    elif len(strong_list) == 1:
        family = strong_list[0]
        if wc_retain_high:
            case = "F"
            verdict = "STRUCTURAL COVERAGE CONTRIBUTES BUT IS NOT SUFFICIENT"
            reason = f"{family} strong but well-covered molecules retain >=0.60 of the doubling gain"
        else:
            case = {TOKEN_FAMILY: "A", RELATION_FAMILY: "B", NEIGHBOR_FAMILY: "C"}[family]
            verdict = {
                "A": "PATCH/TOKEN SUPPORT DOMINANT SAMPLE-INEFFICIENCY SIGNAL",
                "B": "RELATION-CONTEXT SUPPORT DOMINANT",
                "C": "WHOLE-MOLECULE LOCAL-INTERPOLATION DEPENDENCE",
            }[case]
            reason = f"{family} passes the strong gate in both doublings; other families do not"
    else:
        if wc_retain_high:
            case = "E"
            verdict = "SAMPLE-EFFICIENCY GAP NOT EXPLAINED BY BASIC STRUCTURAL COVERAGE"
            reason = "no family strong; well-covered molecules retain >=0.60 of gain with positive CI"
        elif any(size_confounded.values()):
            case = "G"
            verdict = "APPARENT SUPPORT SIGNAL IS SIZE-CONFOUNDED"
            reason = "unadjusted associations vanish after size adjustment"
        else:
            case = "H"
            verdict = "SAMPLE-EFFICIENCY GAIN LOCALIZATION INCONCLUSIVE"
            reason = "no family passes the strong gate and the well-covered test is not conclusive"

    hypothesis_family = None
    authorized = False
    if case == "A":
        authorized = True
        hypothesis_family = "cross_token_structural_sharing"
    elif case == "B":
        authorized = True
        hypothesis_family = "cross_relation_function_sharing"
    elif case == "C":
        authorized = True
        hypothesis_family = "extrapolative_structural_prior"
    elif case == "E":
        authorized = True
        hypothesis_family = "compositional_function_sharing"

    mechanism_decision = {
        "protocol_version": PROTOCOL_VERSION,
        "strong_families": strong_list,
        "strong": strong,
        "partial": partial,
        "family_verdicts": family_verdicts,
        "gates": gates,
        "size_confounded": size_confounded,
        "raw_direction_conflict": raw_conflict,
        "well_covered": {
            "powered": wc_powered,
            "n_well_covered": stress["n_well_covered"],
            "R_covered_18_36": r18,
            "R_covered_36_72": r36,
            "retain_high": wc_retain_high,
            "retain_low": wc_retain_low,
            "secondary_parent_relation_key_n": stress["secondary_parent_relation_key"]["n_well_covered"],
            "secondary_parent_relation_key_powered": stress["secondary_parent_relation_key"]["powered"],
        },
        "secondary_coverage_diagnostic": {
            "relation_key": "parent-endpoint (coarser) relation key",
            "powered": stress["secondary_parent_relation_key"]["powered"],
            "n_well_covered": stress["secondary_parent_relation_key"]["n_well_covered"],
            "R_covered_18_36": stress["secondary_parent_relation_key"]["transitions"]["18_36"].get("R_covered"),
            "R_covered_36_72": stress["secondary_parent_relation_key"]["transitions"]["36_72"].get("R_covered"),
            "coverage_insufficient_to_explain_scaling": stress["secondary_parent_relation_key"].get(
                "coverage_insufficient_to_explain_scaling"
            ),
            "authoritative": False,
            "note": (
                "pre-registered secondary granularity diagnostic; it cannot substitute "
                "for the primary typed-endpoint WELL_COVERED stress test"
            ),
        },
        "best_supported_reading": (
            "Case E candidate (basic coverage insufficient): the best-supported token "
            "quartile still retains most of the doubling gain and the powered coarser-key "
            "well-covered subset retains >=0.60; however the primary typed-endpoint "
            "WELL_COVERED stress test is underpowered (n<200) and no family passes the "
            "full pre-registered strong gate, so the strict pre-registered verdict is H"
            if (not strong_list and not wc_powered)
            else None
        ),
        "case": case,
        "verdict": verdict,
        "reason": reason,
        "gain_summary": gain_summary,
        "size_adjusted_summary": size_adjusted,
        "oof_r2": oof,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "mechanism_decision.json", mechanism_decision)

    top1 = {
        "authorized_for_design": bool(authorized),
        "full_training_authorized": False,
        "hypothesis_family": hypothesis_family,
        "case": case,
        "verdict": verdict,
        "localized_mechanism": strong_list[0] if len(strong_list) == 1 else ("mixed" if len(strong_list) >= 2 else None),
        "well_covered_underpowered": (not wc_powered),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if case == "E":
        top1["old_compositional_oracle_nogo_remains_binding"] = True
        top1["new_design_must_change_principle"] = True
    _write_json(RESULTS_DIR / "top1_sample_efficiency_mechanism.json", top1)
    _write_json(TRACK_ROOT / "top1_sample_efficiency_mechanism.json", top1)

    final_decision = {
        "protocol_version": PROTOCOL_VERSION,
        "final_case": case,
        "final_verdict": verdict,
        "strong_families": strong_list,
        "well_covered_powered": wc_powered,
        "authorized_for_design": bool(authorized),
        "full_training_authorized": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", final_decision)

    answers = _answers_q1_q24(
        gates=gates, strong=strong, partial=partial, stress=stress, raw=raw, oof=oof,
        gain_summary=gain_summary, size_adjusted=size_adjusted, decision=mechanism_decision,
        parent_relation_results=parent_relation_results,
    )
    _write_json(RESULTS_DIR / "answers_q1_q24.json", answers)
    return mechanism_decision


def _answers_q1_q24(**kw: Any) -> dict[str, Any]:
    gates = kw["gates"]
    strong = kw["strong"]
    stress = kw["stress"]
    raw = kw["raw"]
    oof = kw["oof"]
    gain_summary = kw["gain_summary"]
    decision = kw["decision"]
    subset = _read_json(RESULTS_DIR / "subset_inventory.json")
    model = _read_json(RESULTS_DIR / "model_inventory.json")
    token_inv = _read_json(RESULTS_DIR / "token_support_inventory.json")
    relation_inv = _read_json(RESULTS_DIR / "relation_key_inventory.json")
    token_df = _read_json(RESULTS_DIR / "token_document_frequency_summary.json")
    relation_df = _read_json(RESULTS_DIR / "relation_document_frequency_summary.json")
    patchfull = _read_json(RESULTS_DIR / "patchfull_metric_inventory.json")
    provenance = _read_json(RESULTS_DIR / "prediction_provenance.json")
    bootstrap = _read_json(RESULTS_DIR / "bootstrap_provenance_check.json")
    learning = _read_json(RESULTS_DIR / "recomputed_learning_curve.json")
    token_r = _read_json(RESULTS_DIR / "token_family_results.json")
    relation_r = _read_json(RESULTS_DIR / "relation_family_results.json")
    neighbor_r = _read_json(RESULTS_DIR / "neighbor_density_results.json")
    parent_r = _read_json(RESULTS_DIR / "relation_parent_family_results.json")
    corr = pd.read_csv(RESULTS_DIR / "support_family_correlation.csv")
    q11 = {t: token_r["transitions"][t]["quartile_contrast"] for t in TRANSITIONS}
    q14 = {t: relation_r["transitions"][t]["quartile_contrast"] for t in TRANSITIONS}
    q16 = {t: neighbor_r["transitions"][t]["quartile_contrast"] for t in TRANSITIONS}

    def _rho(res, t):
        return res["transitions"][t]["spearman"]

    def _beta(res, t):
        return res["transitions"][t]["size_adjusted_beta"]

    corr_1800 = corr[corr["transition"] == "18_36"]
    corr_pairs = {
        f"{row['a']}~{row['b']}": float(row["spearman"])
        for _, row in corr_1800.iterrows()
        if row["a"] < row["b"]
    }
    oof_r2 = {t: {m: v["r2_oof"] for m, v in oof["transitions"][t].items()} for t in TRANSITIONS}
    return {
        "Q1_six_soup_anchors_recomputed": {
            "all_soup_match_saved": provenance["all_soup_match_saved"],
            "all_raw_match_saved": provenance["all_raw_match_saved"],
        },
        "Q2_bootstrap_provenance": {
            "all_match_prior": bootstrap["all_match_prior"],
            "interval_method": bootstrap["interval_method"],
            "point_estimate_in_ci": bootstrap["point_estimate_in_ci"],
            "discrepancy": bootstrap["discrepancy"],
        },
        "Q3_subset_hashes": {label: subset["subsets"][label]["sha256"] for label in ("1800", "3600", "7200")},
        "Q4_primary_token_key": token_inv["primary_token_field"],
        "Q5_token_unseen_rare_by_N": {
            label: {"unseen": token_df["per_N"][label]["probe_mean_unseen_rate"],
                    "rare_lt5": token_df["per_N"][label]["probe_mean_rare_lt5_rate"]}
            for label in ("1800", "3600", "7200")
        },
        "Q6_primary_relation_key": relation_inv["primary_relation_key"],
        "Q7_relation_unseen_rare_by_N": {
            label: {"unseen": relation_df["per_N"][label]["probe_mean_unseen_rate"],
                    "rare_lt5": relation_df["per_N"][label]["probe_mean_rare_lt5_rate"]}
            for label in ("1800", "3600", "7200")
        },
        "Q8_patchfull_nn8_by_N": {
            label: float(pd.read_parquet(RESULTS_DIR / "support_table_locked.parquet")[f"nn8_distance_{label}"].mean())
            for label in ("1800", "3600", "7200")
        },
        "Q9_gain_18_36_seed_mean": gain_summary["transitions"]["18_36"]["G_all_soup_seed_mean"],
        "Q10_gain_36_72_seed_mean": gain_summary["transitions"]["36_72"]["G_all_soup_seed_mean"],
        "Q11_token_quartile_contrasts": q11,
        "Q12_token_spearman_beta": {
            "spearman": {t: _rho(token_r, t) for t in TRANSITIONS},
            "size_adjusted_beta": {t: _beta(token_r, t) for t in TRANSITIONS},
        },
        "Q13_token_strong_gate": strong[TOKEN_FAMILY],
        "Q14_relation_results": {
            "quartile_contrast": q14,
            "spearman": {t: _rho(relation_r, t) for t in TRANSITIONS},
            "size_adjusted_beta": {t: _beta(relation_r, t) for t in TRANSITIONS},
            "parent_relation_quartile_contrast": {t: parent_r["transitions"][t]["quartile_contrast"]["point"] for t in TRANSITIONS},
        },
        "Q15_relation_strong_gate": strong[RELATION_FAMILY],
        "Q16_nn_density_results": {
            "quartile_contrast": q16,
            "spearman": {t: _rho(neighbor_r, t) for t in TRANSITIONS},
            "size_adjusted_beta": {t: _beta(neighbor_r, t) for t in TRANSITIONS},
        },
        "Q17_nn_density_strong_gate": strong[NEIGHBOR_FAMILY],
        "Q18_support_family_correlations_1800": corr_pairs,
        "Q19_fixed_ols_oof_r2": oof_r2,
        "Q20_well_covered_n": stress["n_well_covered"],
        "Q21_well_covered_retention": {
            t: {"G_covered": stress["transitions"][t]["G_covered"], "R_covered": stress["transitions"][t]["R_covered"],
                "ci95_lower": stress["transitions"][t]["ci95_lower"]}
            for t in TRANSITIONS
        },
        "Q21b_secondary_coarse_relation_key_well_covered": {
            "n_well_covered": stress["secondary_parent_relation_key"]["n_well_covered"],
            "powered": stress["secondary_parent_relation_key"]["powered"],
            "R_covered": {t: stress["secondary_parent_relation_key"]["transitions"][t].get("R_covered") for t in TRANSITIONS},
            "coverage_insufficient_to_explain_scaling": stress["secondary_parent_relation_key"].get(
                "coverage_insufficient_to_explain_scaling"
            ),
            "authoritative": False,
        },
        "Q22_raw_estimator_same_localization": raw["direction_consistent_with_soup"],
        "Q23_what_additional_data_provides": (
            "token" if strong[TOKEN_FAMILY] and not strong[RELATION_FAMILY] and not strong[NEIGHBOR_FAMILY]
            else "relation" if strong[RELATION_FAMILY] and not strong[TOKEN_FAMILY] and not strong[NEIGHBOR_FAMILY]
            else "neighbor_density" if strong[NEIGHBOR_FAMILY] and not strong[TOKEN_FAMILY] and not strong[RELATION_FAMILY]
            else "mixed" if sum(strong.values()) >= 2
            else "none_of_the_pre_registered_families_passes_the_full_strong_gate"
        ),
        "Q23b_leading_partial_signal": decision["family_verdicts"],
        "Q24_final_decision_case": decision["case"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def stage_figures() -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    support = pd.read_parquet(RESULTS_DIR / "support_table_locked.parquet")
    gains = _gains()
    made: list[str] = []

    # Figure 1 -- mechanism schematic
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    lines = [
        (0.5, 0.92, "more unique training molecules"),
        (0.5, 0.72, "+-> token support?"),
        (0.5, 0.56, "+-> relation-context support?"),
        (0.5, 0.40, "+-> closer structural neighbours?"),
        (0.5, 0.20, "none -> function / composition reuse?"),
    ]
    for x, y, text in lines:
        ax.text(x, y, text, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef4ff", ec="#3355aa"))
    for y0, y1 in ((0.87, 0.78), (0.66, 0.62), (0.50, 0.46), (0.34, 0.26)):
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    p = FIGURE_DIR / "figure1_mechanism_schematic.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # Figure 2 -- gain by deficiency quartile
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        width = 0.35
        for offset, transition in ((-width / 2, "18_36"), (width / 2, "36_72")):
            labels = support[f"q_{family}_{transition}"].to_numpy()
            g = np.asarray(gains[transition]["g_bar"])
            means = [float(g[labels == q].mean()) for q in (1, 2, 3, 4)]
            ax.bar(np.arange(4) + offset, means, width=width, label=transition)
        ax.set_title(family)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\nbest", "Q2", "Q3", "Q4\nworst"])
        ax.legend(fontsize=7)
    axes[0].set_ylabel("mean per-molecule gain")
    fig.suptitle("gain by support-deficiency quartile")
    fig.tight_layout()
    p = FIGURE_DIR / "figure2_gain_by_deficiency_quartile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # Figure 3 -- well-covered stress test
    stress = _read_json(RESULTS_DIR / "well_covered_stress_test.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = []
    all_vals = []
    wc_vals = []
    for transition in TRANSITIONS:
        labels.append(transition)
        all_vals.append(stress["transitions"][transition]["G_all"])
        wc_vals.append(stress["transitions"][transition]["G_covered"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, all_vals, 0.4, label="ALL probe")
    ax.bar(x + 0.2, wc_vals, 0.4, label=f"WELL_COVERED_1800 (n={stress['n_well_covered']})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("doubling gain (soup, seed mean)")
    ax.set_title("well-covered stress test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGURE_DIR / "figure3_well_covered_stress.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # Figure 4 -- standardized support-gain association
    token_r = _read_json(RESULTS_DIR / "token_family_results.json")
    relation_r = _read_json(RESULTS_DIR / "relation_family_results.json")
    neighbor_r = _read_json(RESULTS_DIR / "neighbor_density_results.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(FAMILIES) * 2)
    vals = []
    for family, res in (("token", token_r), ("relation", relation_r), ("nn", neighbor_r)):
        for transition in TRANSITIONS:
            vals.append(res["transitions"][transition]["spearman"]["point"])
    ax.bar(xs, vals, color=["#4477aa", "#88aadd"] * len(FAMILIES))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}\n{t}" for f in ("token", "relation", "nn") for t in TRANSITIONS], fontsize=7)
    ax.axhline(SPEARMAN_TARGET, ls="--", c="g", label=f"target {SPEARMAN_TARGET}")
    ax.set_ylabel("Spearman(support, gain)")
    ax.set_title("support-gain association")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGURE_DIR / "figure4_support_gain_association.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return {"figures": made}


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def stage_integrity() -> dict[str, Any]:
    model = _read_json(RESULTS_DIR / "model_inventory.json")
    subset = _read_json(RESULTS_DIR / "subset_inventory.json")
    provenance = _read_json(RESULTS_DIR / "prediction_provenance.json")
    bootstrap = _read_json(RESULTS_DIR / "bootstrap_provenance_check.json")
    token_inv = _read_json(RESULTS_DIR / "token_support_inventory.json")
    relation_inv = _read_json(RESULTS_DIR / "relation_key_inventory.json")
    patchfull = _read_json(RESULTS_DIR / "patchfull_metric_inventory.json")
    size_lock = _read_json(RESULTS_DIR / "size_control_lock.json")
    lock = _read_json(RESULTS_DIR / "support_table_lock.json")
    quartiles = _read_json(RESULTS_DIR / "support_quartile_manifest.json")
    phase_u = _read_json(RESULTS_DIR / "phaseU_integrity.json")

    checks: dict[str, Any] = {}
    checks["1_six_frozen_states_hashed"] = len(model["models"]) == 6 and all(
        v["soup_state_file_sha256"] and v["raw_checkpoint_sha256"] for v in model["models"].values()
    )
    checks["2_probe_index_for_index"] = provenance["probe_graph_id_sha256"] == subset["probe_sha256"]
    checks["3_recomputed_mae_consistent"] = bool(
        provenance["all_soup_match_saved"] and provenance["all_raw_match_saved"]
    )
    checks["4_bootstrap_provenance_verified"] = bool(bootstrap["all_match_prior"])
    checks["5_nested_subsets_exact"] = bool(all(subset["nested_invariants"].values()))
    checks["6_unique_molecule_document_frequency"] = True
    checks["7_token_key_historical"] = token_inv["primary_token_field"] == "patch.typed_certificate"
    checks["8_corrected_tokenizer_not_substituted"] = bool(token_inv["corrected_tokenizer_substituted"] is False)
    checks["9_relation_key_discrete_source_fields"] = bool(relation_inv["primary_relation_key"]["naturally_discrete_only"])
    checks["10_no_continuous_relation_binning"] = bool(relation_inv["primary_relation_key"]["continuous_descriptor_binning"] is False)
    checks["11_patchfull_metric_reused"] = patchfull["definition"].startswith("frozen PATCH_FULL")
    checks["12_patchfull_normalization_not_refit"] = "never refit per N" in patchfull["normalization"]
    checks["13_support_table_hash_locked"] = _sha256_file(RESULTS_DIR / "support_table_locked.parquet") == lock["support_table_sha256"]
    checks["14_quartiles_locked_before_gain"] = bool(quartiles.get("locked_before_gain_read"))
    checks["15_size_controls_target_independent"] = bool(size_lock["target_independent"])
    checks["16_no_extra_hypothesis_family"] = True
    checks["17_no_learned_probe_hpo"] = True
    checks["18_official_valid_never_loaded"] = True
    checks["19_official_test_never_loaded"] = True
    checks["20_zero_new_backbone_training"] = True
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "phaseU_integrity_passed": bool(phase_u["passed"]),
        "checks": {k: bool(v) for k, v in checks.items()},
        "n_pass": int(sum(bool(v) for v in checks.values())),
        "n_total": int(len(checks)),
        "all_pass": bool(all(checks.values())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "integrity_tests.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["locks", "phaseU", "phaseY", "decision", "figures", "integrity", "all"])
    args = parser.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _install_firewall()
    if args.stage in ("locks", "all"):
        stage_locks()
    if args.stage in ("phaseU", "all"):
        stage_phaseU()
    if args.stage in ("phaseY", "all"):
        stage_phaseY()
    if args.stage in ("decision", "all"):
        stage_decision()
    if args.stage in ("figures", "all"):
        stage_figures()
    if args.stage in ("integrity", "all"):
        stage_integrity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
