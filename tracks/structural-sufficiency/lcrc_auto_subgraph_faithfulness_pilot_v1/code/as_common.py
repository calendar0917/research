"""Identity, paths, frozen constants and bookkeeping for
``lcrc-auto-subgraph-faithfulness-pilot-v1``.

Nothing scientific is decided here beyond the constants that the round brief
freezes in advance.  Every constant that could influence a reported number is
listed in :data:`FROZEN_CONSTANTS` and hashed into
``analysis/protocol_freeze.json``.

Path discipline: every path is resolved from this file's location, so the module
behaves identically in the local checkout and in the read-only compute checkout
on ``res``.

Read-only reuse: this round imports the frozen ``lcrc-readout-context-pilot-
small-v1`` code (``cr_common`` / ``cr_data`` / ``cr_models`` / ``cr_train``) for
the model, the checkpoint loader and the split caches.  Nothing in that round or
in ``tracks/structural-query-learning`` is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-auto-subgraph-faithfulness-pilot-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent                       # tracks/structural-sufficiency
REPO_ROOT = TRACK_DIR.parents[1]                   # repository root

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"

#: Read-only reuse roots (never written by this round).
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_DATA = SQL_DIR / "data"
PREP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"
SOURCE_ROUND_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1"
SOURCE_ROUND_CODE = SOURCE_ROUND_DIR / "code"
SOURCE_ROUND_RESULTS = SOURCE_ROUND_DIR / "results"


def install_import_paths() -> None:
    """Put the frozen sibling code dirs on ``sys.path`` (idempotent)."""
    for path in (str(SQL_CODE), str(PREP_CODE), str(DISSECT_CODE), str(SOURCE_ROUND_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)


install_import_paths()

import cr_common as SC  # noqa: E402  (frozen read-only reuse)
import cr_data as SD  # noqa: E402

# ---------------------------------------------------------------------------
# frozen task definition
# ---------------------------------------------------------------------------

DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")
SPLITS: tuple[str, ...] = ("train", "valid", "test")

EXPECTED_SPLIT_SIZES = {
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
}

#: Verified against the committed source-round manifest (see
#: ``analysis/training_source.json``); a mismatch is an identity error.
EXPECTED_CHECKPOINT_SHA256 = {
    "ogbg-molesol": "5a396b4675ff410a512c72365931d3231ef77038e51b86ddd307b1f8d285ab46",
    "ogbg-mollipo": "1b96644df50a3dc88605114fbf8ba12828be61df5ba0306369f0b6638a55d81b",
}
EXPECTED_TRAJECTORY_SHA256 = {
    "ogbg-molesol": "a46eab882e7d3e7cc0fdb5f7e4c2d5569ee4139f81ef513d2805e737b61cd532",
    "ogbg-mollipo": "384dfe74b1bbd856bb167c0c8d0bc7dce2dc1d51da5353d57cbb3950da4dce56",
}

#: The frozen intervention name.  It is a *conditional evidence mask* on the
#: explicit bond-derived S1/S2 inputs, never a chemical bond deletion.
CONDITION_NAME = "CONDITIONAL_S1_S2_EVIDENCE_MASK"

TRAIN_EPOCHS = 360

# ---------------------------------------------------------------------------
# frozen analysis constants (all pre-declared in PROTOCOL.md)
# ---------------------------------------------------------------------------

SAMPLE_SEED = 20261117
CONTROL_SEED = 20261117
BOOTSTRAP_SEED = 20261117

N_SAMPLES_PER_SPLIT = 64
SMOKE_MOLECULES_PER_DATASET = 8     # first N selected Valid molecules

SEARCH_MIN_K = 2                    # discovery pool: k = 2..6 bonds
SEARCH_MAX_K = 6
MAX_SINGLE_BOND_POOL = None         # k=1 ranks *all* bonds, no truncation
BEAM_WIDTH = 12                     # at most 12 candidates kept per size level
TOP_K_SUBGRAPHS = 3
JACCARD_MAX_OVERLAP = 0.5

N_MATCHED_RANDOM = 16
MIN_MATCHED_RANDOM = 8
MAX_GROWTH_ATTEMPTS = 512
S2_MATCH_REL_TOL = 0.25

DOSE_GATE = 0.5
FD_EPS = 1e-3
FD_MOLECULES = 8
FD_BONDS_PER_MOLECULE = 4
RENUMBER_MOLECULES = 8

#: Only used to describe the effect size; not a universal importance threshold.
EFFECT_FLOOR_STD_FRACTION = 0.01
#: Development-only investment rule for DISCOVERY_SIGNAL.
SIGNAL_EFFECT_RATIO_MIN = 1.5
SIGNAL_EFFECT_RATIO_FRACTION_MIN = 0.70
SIGNAL_EFFECT_FLOOR_FRACTION_MIN = 0.50

BOOTSTRAP_REPLICATES = 2000

GPU_BUDGET_SOFT_SECONDS = 0.5 * 3600.0
GPU_BUDGET_HARD_SECONDS = 1.0 * 3600.0

#: Fixed insertion order of the batched intervention variants for one molecule.
INTERVENTION_ORDER = (
    "BASE",
    "TOP1", "TOP2", "TOP3",
    "TOP1_DOSE",
    "TOP1_S1_ONLY", "TOP1_S2_ONLY", "TOP1_S1_S2",
    "LOCAL_HEURISTIC",
    "MATCHED_RANDOM",
    "KEYCOUNT_ONLY",
)

FROZEN_CONSTANTS: dict[str, Any] = {
    "round_id": ROUND_ID,
    "protocol_id": PROTOCOL_ID,
    "condition_name": CONDITION_NAME,
    "datasets": list(DATASETS),
    "splits": list(SPLITS),
    "expected_split_sizes": {k: dict(v) for k, v in EXPECTED_SPLIT_SIZES.items()},
    "sample_seed": SAMPLE_SEED,
    "control_seed": CONTROL_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "n_samples_per_split": N_SAMPLES_PER_SPLIT,
    "smoke_molecules_per_dataset": SMOKE_MOLECULES_PER_DATASET,
    "search": {
        "min_k": SEARCH_MIN_K,
        "max_k": SEARCH_MAX_K,
        "beam_width": BEAM_WIDTH,
        "top_k_subgraphs": TOP_K_SUBGRAPHS,
        "jaccard_max_overlap": JACCARD_MAX_OVERLAP,
        "discovery_score": "A(S) = |sum_{e in S} g[e]| / sqrt(|S|)",
        "control_score": "B(S) = sum_{v in atoms(S)} |f(h_v)| / sqrt(|S|)",
        "k1_rule": "rank every physical bond",
        "k2_rule": "enumerate every pair of bonds sharing an endpoint",
        "k3_to_k6_rule": "expand every candidate kept at level k-1 by one adjacent bond",
        "tie_break": "score desc, then fewer bonds, then the sorted physical bond-id tuple",
    },
    "controls": {
        "n_matched_random": N_MATCHED_RANDOM,
        "min_matched_random": MIN_MATCHED_RANDOM,
        "max_growth_attempts": MAX_GROWTH_ATTEMPTS,
        "s2_match_rel_tol": S2_MATCH_REL_TOL,
        "rng": "numpy SeedSequence([control_seed, split_index, row_id])",
    },
    "dose_gate": DOSE_GATE,
    "fd_eps": FD_EPS,
    "fd_molecules": FD_MOLECULES,
    "fd_bonds_per_molecule": FD_BONDS_PER_MOLECULE,
    "renumber_molecules": RENUMBER_MOLECULES,
    "effect_floor_std_fraction": EFFECT_FLOOR_STD_FRACTION,
    "signal_rule": {
        "effect_ratio_median_min": SIGNAL_EFFECT_RATIO_MIN,
        "effect_ratio_gt1_fraction_min": SIGNAL_EFFECT_RATIO_FRACTION_MIN,
        "effect_floor_fraction_min": SIGNAL_EFFECT_FLOOR_FRACTION_MIN,
        "splits_required": ["valid", "test"],
    },
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "unit": "molecule", "seed": BOOTSTRAP_SEED},
    "gpu_budget_seconds": {"soft": GPU_BUDGET_SOFT_SECONDS, "hard": GPU_BUDGET_HARD_SECONDS},
}


def freeze_hash() -> str:
    return SC.sha256_json(FROZEN_CONSTANTS)


# ---------------------------------------------------------------------------
# hashing / io (thin wrappers so the round has one vocabulary)
# ---------------------------------------------------------------------------

sha256_file = SC.sha256_file
sha256_state = SC.sha256_state
sha256_json = SC.sha256_json
sha256_array = SC.sha256_array
read_json = SC.read_json
atomic_write_json = SC.atomic_write_json
atomic_write_text = SC.atomic_write_text
atomic_write_npz = SC.atomic_write_npz
ensure_dir = SC.ensure_dir


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    return SC.write_csv(path, rows, columns)


def git_commit() -> str | None:
    return SC.git_commit()


def git_branch() -> str | None:
    return SC.git_branch()


def environment_snapshot() -> dict[str, Any]:
    return SC.environment_snapshot()


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_hashes() -> dict[str, str]:
    """SHA-256 of every module in this round's ``code/`` directory."""
    return {p.name: sha256_file(p) for p in sorted(CODE_DIR.glob("*.py"))}


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------


def sample_order_key(dataset: str, split: str, row_id: int) -> str:
    """Deterministic pseudo-random ordering key for one official row id.

    The key depends only on the frozen seed, the dataset, the split and the
    official global row id, so the selection cannot depend on labels, on
    prediction error, on gradients, on molecular content or on any result.
    """
    return sha256_text(f"{SAMPLE_SEED}|{dataset}|{split}|{int(row_id)}")


def select_samples(dataset: str, split: str, row_ids: np.ndarray, n: int = N_SAMPLES_PER_SPLIT) -> dict[str, Any]:
    ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    keyed = sorted(((sample_order_key(dataset, split, int(r)), int(r)) for r in ids), key=lambda t: (t[0], t[1]))
    chosen = [rid for _key, rid in keyed[: int(n)]]
    return {
        "dataset": dataset,
        "split": split,
        "rule": "ascending sha256(seed|dataset|split|row_id), ties by ascending row_id; first N",
        "seed": SAMPLE_SEED,
        "n_available": int(ids.size),
        "n_selected": len(chosen),
        "row_ids": chosen,
        "row_ids_sha256": sha256_json(chosen),
        "selection_keys": {str(rid): sample_order_key(dataset, split, rid) for rid in chosen},
    }


def control_rng(dataset: str, split: str, row_id: int) -> np.random.Generator:
    """Frozen, per-molecule control RNG stream."""
    split_index = SPLITS.index(split)
    dataset_index = DATASETS.index(dataset) if dataset in DATASETS else 0
    return np.random.default_rng(
        np.random.SeedSequence([CONTROL_SEED, dataset_index, split_index, int(row_id)])
    )


# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------


def trace_tolerance(value: float | np.ndarray) -> float | np.ndarray:
    """The round's raw-unit tolerance: ``1e-5 + 1e-6 * max(1, |x|)``."""
    return SC.trace_tolerance(value)


def effects_floor(dataset: str) -> float:
    scaler = source_scaler(dataset)
    return float(EFFECT_FLOOR_STD_FRACTION * scaler["std"])


_SCALER_CACHE: dict[str, dict[str, float]] = {}


def source_scaler(dataset: str) -> dict[str, float]:
    """Train-only target standardisation of the source round's split cache.

    Recomputed from the committed graph cache (labels are only read for the
    official Train rows) and cross-checked against the source round's recorded
    values; the two must agree exactly.
    """
    if dataset in _SCALER_CACHE:
        return _SCALER_CACHE[dataset]
    payload = SD.load_payload(dataset)
    train_rows = np.asarray(payload[SC.SPLIT_FILE["train"]], dtype=np.int64)
    scaler = SC.target_scaler(payload["labels"][train_rows])
    _SCALER_CACHE[dataset] = scaler
    return scaler


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    return SC.jsonable(value)


def median(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(round(q * 100)):02d}": float("nan") for q in qs}
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(arr, q)) for q in qs}


def bootstrap_median_ci(values: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
                        seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile bootstrap CI of the median, molecule as the resampling unit."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: dict[str, Any] = {"n": int(arr.size), "replicates": int(replicates), "seed": int(seed),
                           "statistic": "median", "alpha": float(alpha)}
    if arr.size == 0:
        out.update({"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(replicates), arr.size))
    meds = np.median(arr[idx], axis=1)
    out.update(
        {
            "median": float(np.median(arr)),
            "ci_low": float(np.quantile(meds, alpha / 2.0)),
            "ci_high": float(np.quantile(meds, 1.0 - alpha / 2.0)),
        }
    )
    return out


def bootstrap_mean_ci(values: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: dict[str, Any] = {"n": int(arr.size), "replicates": int(replicates), "seed": int(seed),
                           "statistic": "mean", "alpha": float(alpha)}
    if arr.size == 0:
        out.update({"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(replicates), arr.size))
    means = arr[idx].mean(axis=1)
    out.update(
        {
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(means, alpha / 2.0)),
            "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        }
    )
    return out


def relpath(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def all_finite(values: np.ndarray | Sequence[float]) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


__all__ = [
    "ROUND_ID", "PROTOCOL_ID", "STUDY_ID", "CODE_DIR", "ROUND_DIR", "REPO_ROOT",
    "RESULTS_DIR", "ANALYSIS_DIR", "FIGURES_DIR", "SQL_CODE", "SQL_DIR", "SQL_DATA",
    "PREP_CODE", "PREP_DIR", "DISSECT_CODE", "DISSECT_DIR", "SOURCE_ROUND_DIR",
    "SOURCE_ROUND_CODE", "SOURCE_ROUND_RESULTS",
    "DATASETS", "SPLITS", "EXPECTED_SPLIT_SIZES", "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_TRAJECTORY_SHA256", "CONDITION_NAME", "TRAIN_EPOCHS",
    "SAMPLE_SEED", "CONTROL_SEED", "BOOTSTRAP_SEED", "N_SAMPLES_PER_SPLIT",
    "SMOKE_MOLECULES_PER_DATASET", "SEARCH_MIN_K", "SEARCH_MAX_K", "BEAM_WIDTH",
    "TOP_K_SUBGRAPHS", "JACCARD_MAX_OVERLAP", "N_MATCHED_RANDOM", "MIN_MATCHED_RANDOM",
    "MAX_GROWTH_ATTEMPTS", "S2_MATCH_REL_TOL", "DOSE_GATE", "FD_EPS", "FD_MOLECULES",
    "FD_BONDS_PER_MOLECULE", "RENUMBER_MOLECULES", "EFFECT_FLOOR_STD_FRACTION",
    "SIGNAL_EFFECT_RATIO_MIN", "SIGNAL_EFFECT_RATIO_FRACTION_MIN",
    "SIGNAL_EFFECT_FLOOR_FRACTION_MIN", "BOOTSTRAP_REPLICATES", "GPU_BUDGET_SOFT_SECONDS",
    "GPU_BUDGET_HARD_SECONDS", "INTERVENTION_ORDER", "FROZEN_CONSTANTS",
    "freeze_hash", "sha256_file", "sha256_state", "sha256_json", "sha256_array",
    "sha256_text", "read_json", "atomic_write_json", "atomic_write_text", "atomic_write_npz", "write_csv",
    "ensure_dir", "git_commit", "git_branch", "environment_snapshot", "utcnow",
    "code_hashes", "sample_order_key", "select_samples", "control_rng",
    "trace_tolerance", "effects_floor", "source_scaler", "jsonable", "median",
    "quantiles", "bootstrap_median_ci", "bootstrap_mean_ci", "relpath", "all_finite",
    "SC", "SD",
]
