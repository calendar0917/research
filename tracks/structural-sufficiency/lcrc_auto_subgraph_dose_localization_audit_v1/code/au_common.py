"""Identity, paths, frozen constants and bookkeeping for
``lcrc-auto-subgraph-dose-localization-audit-v1``.

This round is a **read-only audit** of the already completed
``lcrc-auto-subgraph-faithfulness-pilot-v1``.  It re-uses, without modification:

* the two official OGB datasets and their frozen Train/Valid/Test caches;
* the exact 64 official row ids per split selected by the pilot;
* the frozen Arm A (LCRC-C1) checkpoint of ``lcrc_readout_context_pilot_small_v1``;
* the pilot's already selected Top-1 physical bond sets;
* the pilot's stored bond gradients, discovery scores and search log.

It re-derives nothing that the pilot selected (no new Top-1, no retraining, no new
seed, no architecture change, no wider search space).  Its two new measurements
are the **correct S2 dose accounting** (``n_s2_affected`` instead of
``n_s2_internal``) and the **root-node localisation** of the masked response.

Path discipline: every path is resolved from this file's location.  Upstream
rounds are imported read-only; every artefact this round writes lives under
``ROUND_DIR`` and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-auto-subgraph-dose-localization-audit-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID
UPSTREAM_ROUND_ID = "lcrc-auto-subgraph-faithfulness-pilot-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent                       # tracks/structural-sufficiency
REPO_ROOT = TRACK_DIR.parents[1]                   # repository root

RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

#: Read-only upstream rounds and frozen input roots (never written here).
#: Round ids use hyphens; repository directories use underscores.
UPSTREAM_ROUND_DIR = TRACK_DIR / UPSTREAM_ROUND_ID.replace("-", "_")
UPSTREAM_RESULTS = UPSTREAM_ROUND_DIR / "results"
UPSTREAM_ANALYSIS = UPSTREAM_ROUND_DIR / "analysis"
UPSTREAM_CODE = UPSTREAM_ROUND_DIR / "code"
SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_DATA = SQL_DIR / "data"
PREP_DIR = TRACK_DIR / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
DISSECT_DIR = TRACK_DIR / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"
SOURCE_ROUND_DIR = TRACK_DIR / "lcrc_readout_context_pilot_small_v1"
SOURCE_ROUND_CODE = SOURCE_ROUND_DIR / "code"
SOURCE_ROUND_RESULTS = SOURCE_ROUND_DIR / "results"


def install_import_paths() -> None:
    """Put the upstream read-only code dirs on ``sys.path`` (idempotent)."""
    for path in (
        str(CODE_DIR),
        str(UPSTREAM_CODE),
        str(SQL_CODE),
        str(PREP_CODE),
        str(DISSECT_CODE),
        str(SOURCE_ROUND_CODE),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


install_import_paths()

import as_common as UP  # noqa: E402  (frozen read-only reuse of the pilot round)
import cr_data as SD  # noqa: E402  (frozen read-only reuse)
import cr_common as SC  # noqa: E402

# ---------------------------------------------------------------------------
# frozen task definition (identical to the pilot)
# ---------------------------------------------------------------------------

DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")
SPLITS: tuple[str, ...] = ("train", "valid", "test")

EXPECTED_SPLIT_SIZES = {
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
}

#: Verified against the pilot's committed manifest and re-hashed by the preflight.
EXPECTED_CHECKPOINT_SHA256 = dict(UP.EXPECTED_CHECKPOINT_SHA256)
EXPECTED_TRAJECTORY_SHA256 = dict(UP.EXPECTED_TRAJECTORY_SHA256)

CONDITION_NAME = UP.CONDITION_NAME                  # CONDITIONAL_S1_S2_EVIDENCE_MASK

SAMPLE_SEED = 20261117
CONTROL_SEED = 20261117
BOOTSTRAP_SEED = 20261117
N_SAMPLES_PER_SPLIT = 64

#: Corrected null: same molecule, same physical bond count, ``n_s2_affected``
#: matched within +-25% of the Top-1 value.
N_MATCHED_RANDOM = 16
MIN_MATCHED_RANDOM = 8
MAX_GROWTH_ATTEMPTS = 512
S2_AFFECTED_REL_TOL = 0.25

#: Descriptive effect floor reused from the pilot (1% of Train label std).
EFFECT_FLOOR_STD_FRACTION = 0.01

#: The pilot's frozen discovery-signal thresholds, reused unchanged so that the
#: corrected numbers are directly comparable with the pilot's own verdict rule.
SIGNAL_EFFECT_RATIO_MIN = UP.SIGNAL_EFFECT_RATIO_MIN            # 1.5
SIGNAL_EFFECT_RATIO_FRACTION_MIN = UP.SIGNAL_EFFECT_RATIO_FRACTION_MIN  # 0.70
SIGNAL_EFFECT_FLOOR_FRACTION_MIN = UP.SIGNAL_EFFECT_FLOOR_FRACTION_MIN  # 0.50
MIN_MATCHED_COVERAGE_FRACTION = 0.80

#: Pre-declared localisation bands on the molecule-level ``outside_fraction``.
LOCALIZATION_CONCENTRATED_MAX = 0.25
LOCALIZATION_SPREAD_MIN = 0.50

#: Pre-declared dose-defect grade for the pilot's null matching.
DOSE_DEFECT_SUBSTANTIVE_CONTROL_FRACTION = 0.20

BOOTSTRAP_REPLICATES = 2000

#: Implementation guard (memory), not a research rule.  One forward pass never
#: sees more than this many replicas of a single molecule.
MAX_VARIANTS_PER_FORWARD = 8
MAX_FORWARD_ATOMS = 200_000
MAX_FORWARD_BONDS = 200_000

#: Localisation agreement tolerance: the round reuses the pilot's raw-unit
#: tolerance ``1e-5 + 1e-6*max(1,|x|)`` for every closure identity, and a slightly
#: looser (still raw-unit) tolerance to re-identify a stored number across the
#: GPU->CPU replay of the same frozen function.
REPLAY_TOLERANCE_ABS = 1e-5
REPLAY_TOLERANCE_REL = 1e-5

INTERVENTION_ORDER = ("BASE", "TOP1", "MATCHED_RANDOM")

FROZEN_CONSTANTS: dict[str, Any] = {
    "round_id": ROUND_ID,
    "upstream_round_id": UPSTREAM_ROUND_ID,
    "protocol_id": PROTOCOL_ID,
    "condition_name": CONDITION_NAME,
    "datasets": list(DATASETS),
    "splits": list(SPLITS),
    "expected_split_sizes": {k: dict(v) for k, v in EXPECTED_SPLIT_SIZES.items()},
    "sample_seed": SAMPLE_SEED,
    "control_seed": CONTROL_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "n_samples_per_split": N_SAMPLES_PER_SPLIT,
    "reused": {
        "top1": "pilot results/<dataset>/<split>/molecules.json topk[0].bonds (never re-selected)",
        "gradient": "pilot results/<dataset>/<split>/molecules.json gradient.gradient_raw",
        "discovery_score": "pilot topk[*].score_A",
        "search_log": "pilot results/<dataset>/candidates.npz",
        "checkpoint": "lcrc_readout_context_pilot_small_v1/results/<dataset>/A/checkpoint_best_val.pt",
        "heuristic": "pilot local_heuristic.effect (reused verbatim, no re-search)",
    },
    "s2_dose_accounting": {
        "n_s2_internal": "e1 in S AND e2 in S (pilot's matching quantity; NOT the affected dose)",
        "n_s2_boundary": "exactly one of e1, e2 in S",
        "n_s2_affected": "e1 in S OR e2 in S (the occurrences the m[e1]*m[e2] gate actually removes)",
        "identity": "n_s2_affected = n_s2_internal + n_s2_boundary",
    },
    "controls": {
        "match_quantity": "n_s2_affected",
        "same_molecule": True,
        "same_bond_count": True,
        "exclude_identical_to_top1": True,
        "s2_affected_rel_tol": S2_AFFECTED_REL_TOL,
        "denominator": "max(1, n_s2_affected(top1))",
        "n_matched_random": N_MATCHED_RANDOM,
        "min_matched_random": MIN_MATCHED_RANDOM,
        "max_growth_attempts": MAX_GROWTH_ATTEMPTS,
        "growth": "identical to the pilot's grow_connected_subgraph (same RNG stream, only the acceptance criterion changes)",
        "rng": "numpy SeedSequence([20261117, dataset_index, split_index, row_id])",
        "shortfall_label": "INSUFFICIENT_MATCHED_NULL",
    },
    "localization": {
        "root_value": "c_v = f(h_v) for every atom root v, evaluated before and after the mask",
        "inside": "atoms incident to at least one masked physical bond of the Top-1 subgraph",
        "outside": "every other atom root of the same molecule",
        "delta_c_v": "c_v(original) - c_v(masked)",
        "outside_fraction": "sum_outside |delta_c_v| / (sum_inside + sum_outside) |delta_c_v|",
        "closure_checks": ["sum_v delta_c_v == delta_L", "delta_L + delta_rho == delta_total"],
        "concentrated_max": LOCALIZATION_CONCENTRATED_MAX,
        "spread_min": LOCALIZATION_SPREAD_MIN,
    },
    "signal_rule": {
        "effect_ratio_median_min": SIGNAL_EFFECT_RATIO_MIN,
        "effect_ratio_gt1_fraction_min": SIGNAL_EFFECT_RATIO_FRACTION_MIN,
        "effect_floor_fraction_min": SIGNAL_EFFECT_FLOOR_FRACTION_MIN,
        "matched_coverage_fraction_min": MIN_MATCHED_COVERAGE_FRACTION,
        "splits_required": ["valid", "test"],
    },
    "dose_defect_rule": {
        "substantive_if_control_fraction_outside_tol_at_least": DOSE_DEFECT_SUBSTANTIVE_CONTROL_FRACTION,
        "tolerance": S2_AFFECTED_REL_TOL,
    },
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "unit": "molecule", "seed": BOOTSTRAP_SEED},
    "replay_tolerance": {"abs": REPLAY_TOLERANCE_ABS, "rel": REPLAY_TOLERANCE_REL},
    "memory_guard": {
        "max_variants_per_forward": MAX_VARIANTS_PER_FORWARD,
        "max_forward_atoms": MAX_FORWARD_ATOMS,
        "max_forward_bonds": MAX_FORWARD_BONDS,
    },
}


def freeze_hash() -> str:
    return SC.sha256_json(FROZEN_CONSTANTS)


# ---------------------------------------------------------------------------
# hashing / io
# ---------------------------------------------------------------------------

sha256_file = SC.sha256_file
sha256_state = SC.sha256_state
sha256_json = SC.sha256_json
sha256_array = SC.sha256_array
read_json = SC.read_json
read_csv = SC.read_csv
atomic_write_json = SC.atomic_write_json
atomic_write_text = SC.atomic_write_text
atomic_write_npz = SC.atomic_write_npz
write_csv = SC.write_csv
ensure_dir = SC.ensure_dir
trace_tolerance = SC.trace_tolerance
environment_snapshot = SC.environment_snapshot
git_commit = SC.git_commit
git_branch = SC.git_branch
git_dirty_tracked = SC.git_dirty_tracked


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def worktree_dirty() -> bool | None:
    try:
        return bool(SC.git_dirty_tracked())
    except Exception:                              # noqa: BLE001 - provenance only
        return None


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_hashes() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(CODE_DIR.glob("*.py"))}


def relpath(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def jsonable(value: Any) -> Any:
    return SC.jsonable(value)


def write_csv_file(path: Path | str, rows: Sequence[Mapping[str, Any]],
                   columns: Sequence[str] | None = None) -> Path:
    ensure_dir(Path(path).parent)
    return SC.write_csv(path, rows, columns)


# ---------------------------------------------------------------------------
# upstream identity helpers
# ---------------------------------------------------------------------------


def upstream_paths(dataset: str) -> dict[str, Path]:
    base = UPSTREAM_RESULTS / dataset
    return {
        "samples": base / "samples.json",
        "preflight": base / "preflight.json",
        "run_summary": base / "run_summary.json",
        "candidates": base / "candidates.npz",
        "train": base / "train" / "molecules.json",
        "valid": base / "valid" / "molecules.json",
        "test": base / "test" / "molecules.json",
        "checkpoint": SOURCE_ROUND_RESULTS / dataset / "A" / "checkpoint_best_val.pt",
        "trajectory": SOURCE_ROUND_RESULTS / dataset / "A" / "trajectory.json",
        "source_manifest": SOURCE_ROUND_DIR / "analysis" / "run_manifest.json",
    }


def upstream_molecules(dataset: str, split: str) -> list[dict[str, Any]]:
    path = upstream_paths(dataset)[split]
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"unexpected pilot molecule payload type at {relpath(path)}")
    return payload


def load_bundle(dataset: str):
    return SD.build_bundle(dataset)


def train_std(dataset: str) -> float:
    return float(UP.source_scaler(dataset)["std"])


def source_scaler(dataset: str) -> dict[str, float]:
    return UP.source_scaler(dataset)


def select_samples(dataset: str, split: str, row_ids: np.ndarray) -> dict[str, Any]:
    return UP.select_samples(dataset, split, row_ids, n=N_SAMPLES_PER_SPLIT)


def control_rng(dataset: str, split: str, row_id: int) -> np.random.Generator:
    """Frozen per-molecule control RNG stream (identical derivation to the pilot)."""
    return UP.control_rng(dataset, split, int(row_id))


# ---------------------------------------------------------------------------
# small deterministic statistics
# ---------------------------------------------------------------------------


def median(values: Sequence[float]) -> float:
    return UP.median(values)


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95)) -> dict[str, float]:
    return UP.quantiles(values, qs=qs)


def bootstrap_median_ci(values: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
                        seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict[str, Any]:
    return UP.bootstrap_median_ci(values, replicates=replicates, seed=seed, alpha=alpha)


def bootstrap_mean_ci(values: Sequence[float], *, replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict[str, Any]:
    return UP.bootstrap_mean_ci(values, replicates=replicates, seed=seed, alpha=alpha)


def spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Spearman rank correlation with a NaN-aware pairwise complete sample."""
    xa = np.asarray([float(v) for v in x], dtype=np.float64)
    ya = np.asarray([float(v) for v in y], dtype=np.float64)
    if xa.size != ya.size:
        raise ValueError("spearman requires equal-length inputs")
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[keep], ya[keep]
    n = int(xa.size)
    out: dict[str, Any] = {"n": n}
    if n < 3:
        out.update({"rho": float("nan")})
        return out

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        # average ties
        sorted_a = a[order]
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return ranks

    rx, ry = _rank(xa), _rank(ya)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        out.update({"rho": float("nan")})
        return out
    out["rho"] = float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))
    return out


def replay_tolerance(value: float | np.ndarray) -> float | np.ndarray:
    """Raw-unit tolerance used to re-identify a pilot number on the CPU replay."""
    arr = np.abs(np.asarray(value, dtype=np.float64))
    return REPLAY_TOLERANCE_ABS + REPLAY_TOLERANCE_REL * np.maximum(1.0, arr)


__all__ = [
    "ROUND_ID", "PROTOCOL_ID", "STUDY_ID", "UPSTREAM_ROUND_ID", "CODE_DIR", "ROUND_DIR",
    "TRACK_DIR", "REPO_ROOT", "RESULTS_DIR", "ANALYSIS_DIR", "UPSTREAM_ROUND_DIR",
    "UPSTREAM_RESULTS", "UPSTREAM_ANALYSIS", "UPSTREAM_CODE", "SQL_DIR", "SQL_CODE",
    "SQL_DATA", "PREP_DIR", "PREP_CODE", "DISSECT_DIR", "DISSECT_CODE", "SOURCE_ROUND_DIR",
    "SOURCE_ROUND_CODE", "SOURCE_ROUND_RESULTS", "DATASETS", "SPLITS",
    "EXPECTED_SPLIT_SIZES", "EXPECTED_CHECKPOINT_SHA256", "EXPECTED_TRAJECTORY_SHA256",
    "CONDITION_NAME", "SAMPLE_SEED", "CONTROL_SEED", "BOOTSTRAP_SEED",
    "N_SAMPLES_PER_SPLIT", "N_MATCHED_RANDOM", "MIN_MATCHED_RANDOM",
    "MAX_GROWTH_ATTEMPTS", "S2_AFFECTED_REL_TOL", "EFFECT_FLOOR_STD_FRACTION",
    "SIGNAL_EFFECT_RATIO_MIN", "SIGNAL_EFFECT_RATIO_FRACTION_MIN",
    "SIGNAL_EFFECT_FLOOR_FRACTION_MIN", "MIN_MATCHED_COVERAGE_FRACTION",
    "LOCALIZATION_CONCENTRATED_MAX", "LOCALIZATION_SPREAD_MIN",
    "DOSE_DEFECT_SUBSTANTIVE_CONTROL_FRACTION", "BOOTSTRAP_REPLICATES",
    "MAX_VARIANTS_PER_FORWARD", "MAX_FORWARD_ATOMS", "MAX_FORWARD_BONDS",
    "REPLAY_TOLERANCE_ABS", "REPLAY_TOLERANCE_REL", "INTERVENTION_ORDER",
    "FROZEN_CONSTANTS", "freeze_hash", "sha256_file", "sha256_state", "sha256_json",
    "sha256_array", "read_json", "read_csv", "atomic_write_json", "atomic_write_text",
    "atomic_write_npz", "write_csv", "ensure_dir", "trace_tolerance",
    "environment_snapshot", "git_commit", "git_branch", "git_dirty_tracked",
    "sha256_text", "worktree_dirty", "utcnow", "code_hashes", "relpath", "jsonable",
    "write_csv_file", "upstream_paths", "upstream_molecules", "load_bundle", "train_std",
    "source_scaler", "select_samples", "control_rng", "median", "quantiles",
    "bootstrap_median_ci", "bootstrap_mean_ci", "spearman", "replay_tolerance",
    "UP", "SC", "SD",
]
