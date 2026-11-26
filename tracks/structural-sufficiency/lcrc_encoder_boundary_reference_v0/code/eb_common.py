"""Identity, frozen constants and read-only reuse for the encoder-boundary round.

Everything scientific (recipe, model A, data pipeline) is inherited read-only from
the CSC v0 round and its upstream modules.  This module only fixes *this* round's
constants: the GINE encoder shape rule, the parameter budget, the comparability
conditions, the bootstrap seed and the verdict thresholds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROUND_ID = "lcrc-encoder-boundary-reference-v0"
PROTOCOL_ID = "lcrc-encoder-boundary-reference-v0"
SCHEMA_RUN = "ebr-run-v1"
SCHEMA_CHECKPOINT = "ebr-checkpoint-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

CSC_DIR = TRACK_DIR / "context_conditioned_structural_computation_v0"
A_RUN = CSC_DIR / "runs/mollipo_A_seed0"

RUNS_DIR = ROUND_DIR / "runs"
ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
CONFIG_DIR = ROUND_DIR / "configs"

for _path in (str(CSC_DIR / "code"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cs_common as C  # noqa: E402  (read-only; calls C.setup_paths())

C.setup_paths()

DATASET = "ogbg-mollipo"
#: this round never builds, scores or reads a Test prediction
SPLITS_USED: tuple[str, ...] = ("train", "valid")

A_TOTAL_PARAMS = 294135
PARAM_TOL = 0.15
PARAM_LOW = int(A_TOTAL_PARAMS * (1.0 - PARAM_TOL))          # 250,014
PARAM_HIGH = int(A_TOTAL_PARAMS * (1.0 + PARAM_TOL))         # 338,255

#: GINE encoder, determined by the existing implementation + the budget rule
GINE_HIDDEN = int(128)
GINE_MLP_HIDDEN = int(2 * GINE_HIDDEN)
GINE_LAYERS = int(4)
GINE_DROPOUT = 0.10
OUT_WIDTH = int(C.WIDTH)                                     # 64, the shared post_local width

#: analysis constants
BOOTSTRAP_REPLICATES = int(C.BOOTSTRAP_REPLICATES)
BOOTSTRAP_SEED = int(C.BOOTSTRAP_SEED)
TOP_ERROR_FRACTION = 0.05
REL_GAIN_REFERENCE = 0.02

#: hard limits of this round
GPU_HARD_HOURS = 0.6
TRAIN_HARD_SECONDS = 1500.0
WALL_HARD_MINUTES = 60

A_CHECKPOINT = A_RUN / "checkpoint_best.pt"
A_PREDICTIONS = A_RUN / "predictions_best.npz"
A_RUN_JSON = A_RUN / "run.json"
A_CURVE = A_RUN / "train_curve.csv"

SQ_CACHE_NAME = "ogbg_mollipo_sq-pilot-cache-v1.npz"
SQ_CACHE = C.SQL_DIR / "data" / SQ_CACHE_NAME
SQ_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"

#: expected historical GINE vocabulary (data_audit.json of sq-pilot-small-v1)
EXPECTED_CATEGORIES = {
    "atom_0": 53, "atom_1": 3, "atom_2": 5, "atom_3": 7, "atom_4": 4,
    "atom_5": 1, "atom_6": 3, "atom_7": 2, "atom_8": 2,
    "bond_0": 4, "bond_1": 3, "bond_2": 2,
    "degree": 5, "ring": 10,
}

ATOM_COLUMN_NAMES = (
    "atomic_num", "chirality", "degree", "formal_charge", "num_hs",
    "num_radical_electrons", "hybridization", "is_aromatic", "is_in_ring",
)
BOND_COLUMN_NAMES = ("bond_type", "bond_stereo", "bond_is_in_ring")

POST_LOCAL_PREFIXES = ("f.", "rho.", "readout.")
POST_LOCAL_BUFFERS = ("global_mean", "global_scale")


def track_rel(path: Path | str) -> str:
    return C.track_rel(path)


def write_json(path: Path | str, payload: Any) -> Path:
    return C.write_json(path, payload)


def read_json(path: Path | str) -> Any:
    return C.read_json(path)


def sha256_file(path: Path | str) -> str:
    return C.sha256_file(path)


def state_fingerprint(state: Mapping[str, Any]) -> str:
    return C.state_fingerprint(state)


def head_keys(state: Mapping[str, Any]) -> list[str]:
    return sorted(
        k for k in state
        if k.startswith(POST_LOCAL_PREFIXES) or k in POST_LOCAL_BUFFERS
    )
