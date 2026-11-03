"""Shared constants, paths and hashing for the sparse-identity round.

Round: ``lcrc-sparse-identity-v1``.

Everything scientific (arms, seeds, K, fold count, null-swap seeds, bootstrap
seed, decision thresholds, budget) is declared here before any formal run, and
is copied verbatim into ``analysis/provenance.json``.

Reused read-only (through the frozen PTB layer):

* ``lcrc_predictive_trace_benchmark_v1``: the A0 graph (``ptb_model``), the
  single training unit recipe (``ptb_train_mol``), the MolHIV data view, the
  metrics / official OGB evaluator (``mb2_metrics``) and the frozen R3 recipe
  (``dissect_common.RECIPE``).
* ``lcrc_molnet_benchmark_pilot_v2``: label-free structure caches.
* ``lcrc_cross_task_preparation_v1``: ``lipo_v1`` FeatureSpec, collate.

No upstream file is modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROUND_ID = "lcrc-sparse-identity-v1"
CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]

PTB = REPO / "tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1"
PTB_CODE = PTB / "code"
if str(PTB_CODE) not in sys.path:
    sys.path.insert(0, str(PTB_CODE))

import ptb_common as PC  # noqa: E402  (adds dissect / molnet_v2 / prep / tu paths)

# Re-exported for the other modules of this round (``__all__`` is dynamic below).
import lcrc_data as PREP_DATA  # noqa: E402,F401
import mb2_metrics as MET  # noqa: E402,F401
import ptb_data as PD  # noqa: E402,F401
import ptb_model as PM  # noqa: E402,F401
import ptb_train_mol as PT  # noqa: E402,F401

ANALYSIS = TRACK / "analysis"
PREDICTIONS = ANALYSIS / "predictions"
CONFIGS = TRACK / "configs"
RUNS = TRACK / "runs"
TESTS = TRACK / "tests"

DATASET = "ogbg-molhiv"
TASK = "binary"
N_TASKS = 1

#: frozen pattern budget (never tuned)
K = 512
#: frozen fold count / per-bucket size (FOLD64)
N_BUCKETS = 64
PER_BUCKET = K // N_BUCKETS
FOLD_SALT = "lcrc-sparse-identity-v1"

SEEDS = (0, 1, 2)
NULL_SEED = {"train": 20261103, "valid": 20261104, "test": 20261105}
NULL_SWAP_MULTIPLIER = 8

EPOCHS = 100
SPARSE_RANDOM_STATE = 20261103

BOOTSTRAP_SEED = 20261103
BOOTSTRAP_REPLICATES = 2000

MODEL_A0 = "A0"
MODEL_TRUE = "WIDE_TRUE"
MODEL_NULL = "WIDE_NULL"
MODEL_FOLD = "FOLD64"
MODEL_SPARSE = "SPARSE_ONLY"
NEURAL = (MODEL_A0, MODEL_TRUE, MODEL_NULL, MODEL_FOLD)
WIDE = (MODEL_TRUE, MODEL_NULL, MODEL_FOLD)
ALL_ARMS = NEURAL + (MODEL_SPARSE,)
#: dimension of the vector that feeds the new residual branch
BRANCH_DIM = {MODEL_A0: 0, MODEL_TRUE: K, MODEL_NULL: K, MODEL_FOLD: N_BUCKETS}

CONTRASTS = (
    (MODEL_TRUE, MODEL_A0),
    (MODEL_TRUE, MODEL_NULL),
    (MODEL_TRUE, MODEL_FOLD),
    (MODEL_NULL, MODEL_A0),
    (MODEL_FOLD, MODEL_A0),
)

#: pre-registered decision thresholds (immutable)
DECISION_MIN_TRUE_MINUS_A0 = 0.010
DECISION_MIN_TRUE_MINUS_NULL = 0.005
DECISION_MIN_SEEDS_BEATING_A0 = 2

GPU_SOFT_CAP_HOURS = 4.0
GPU_HARD_CAP_HOURS = 5.0

RESULT_LABEL = "EXPOSED_TEST_EXPLORATORY"
FROZEN_SPEC_HASH = "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee"
EXPECTED_SPLIT_SIZES = {"train": 32901, "valid": 4113, "test": 4113}

#: the 18-ring witness pair from the non-local round (documented, not a gate)
RING18_PAIR = {
    "A": {"smiles": "C1CCCCCNCCCCCOCCCCC1", "n_o_cyclic_distance": 6},
    "B": {"smiles": "C1CCCCNCCCCCCCOCCCC1", "n_o_cyclic_distance": 8},
}

REUSED_MODULES: dict[str, Path] = dict(PC.REUSED_MODULES)
REUSED_MODULES["ptb/ptb_common.py"] = PTB_CODE / "ptb_common.py"
REUSED_MODULES["ptb/ptb_data.py"] = PTB_CODE / "ptb_data.py"
REUSED_MODULES["ptb/ptb_model.py"] = PTB_CODE / "ptb_model.py"
REUSED_MODULES["ptb/ptb_train_mol.py"] = PTB_CODE / "ptb_train_mol.py"

# ---------------------------------------------------------------------------
# hashing / IO (delegated to the frozen PTB helpers where possible)
# ---------------------------------------------------------------------------

sha256_file = PC.sha256_file
sha256_bytes = PC.sha256_bytes
sha256_text = PC.sha256_text
stable_json = PC.stable_json
write_json = PC.write_json
read_json = PC.read_json
write_csv = PC.write_csv
track_rel = PC.track_rel
index_sha256 = PC.index_sha256
utcnow = PC.utcnow
env_info = PC.env_info
git_commit = PC.git_commit
git_dirty = PC.git_dirty


def log(msg: str) -> None:
    print(f"[{ROUND_ID}] {PC.utcnow()} {msg}", flush=True)


def array_hash_f32(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return sha256_bytes(a.tobytes() + str(a.shape).encode("utf-8"))


def array_hash_u8(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8))
    return sha256_bytes(a.tobytes() + str(a.shape).encode("utf-8"))


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(TESTS.glob("*.py")):
        out[f"tests/{p.name}"] = sha256_file(p)
    for p in sorted(TRACK.glob("*.md")):
        out[f"{p.name}"] = sha256_file(p)
    for name, path in sorted(REUSED_MODULES.items()):
        out[f"reused/{name}"] = sha256_file(path)
    return out


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------


def run_name(model: str, seed: int) -> str:
    return f"mol_{DATASET}_s{int(seed)}_{model}"


def run_dir(model: str, seed: int) -> Path:
    return RUNS / run_name(model, seed)


def smoke_dir(model: str, seed: int, tag: str) -> Path:
    return RUNS / "smoke" / f"{tag}_{run_name(model, seed)}"


def units() -> list[tuple[str, int]]:
    return [(m, int(s)) for m in NEURAL for s in SEEDS]


__all__ = [n for n in dir() if not n.startswith("_")]
