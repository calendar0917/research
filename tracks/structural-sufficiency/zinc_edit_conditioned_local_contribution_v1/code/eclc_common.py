"""Shared plumbing for ECLC-v1.

ECLC-v1 = Edit-Conditioned Local Contribution.  The round reuses, read-only,
the frozen LCRC-v1 backbone (atom/bond SSL encoders, LocalContent, f, rho and
the official train/valid caches).  Nothing in the frozen LCRC track is modified.

Only the official PyG ZINC subset train/valid splits are used.  The official
test split is never constructed, loaded, predicted or scored in this track.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROUND_ID = "zinc-edit-conditioned-local-contribution-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
RUNS = EXP_DIR / "runs"
NOTES = EXP_DIR / "notes"
TESTS = EXP_DIR / "tests"

# Read-only reuse of the frozen LCRC-v1 code.  lc_common/lc_data/lc_models are
# imported by every ECLC module; the import only defines paths, constants and
# architecture classes and does not touch labels or the test split.
_LCRC_CODE = REPO / "tracks/structural-sufficiency/zinc_local_relation_composition_v1/code"
if str(_LCRC_CODE) not in sys.path:
    sys.path.insert(0, str(_LCRC_CODE))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402
import lc_models as M  # noqa: E402

#: The frozen LCRC caches are the only data source for the supervised runs.
LCRC_TRAIN_CACHE = L.CACHE / "lc_train.pt"
LCRC_VALID_CACHE = L.CACHE / "lc_valid.pt"
LCRC_LABELS = {split: None for split in ("train", "valid")}  # documentation only

#: Base-head feature layout.  Dimensions are fixed before GPU.
BASE_CENTER_DIM = 21          # one-hot of the frozen 21-way atom type
BASE_R1_DIM = 63              # (neighbour type, bond type) counts
BASE_R2_PATH_DIM = 189        # (middle type, bond1, bond2) 2-path counts
BASE_R2_END_DIM = 63          # (end type, bond2) 2-path counts
BASE_R2_CLOSE_DIM = 1         # number of 2-paths whose end is adjacent to root
BASE_R1_TOTAL = BASE_CENTER_DIM + BASE_R1_DIM
BASE_INPUT_DIM = (
    BASE_CENTER_DIM + BASE_R1_DIM + BASE_R2_PATH_DIM + BASE_R2_END_DIM + BASE_R2_CLOSE_DIM
)

#: Frozen valid slices (same definitions as LCRC-v1 / ZPR-v1).
VALID_ANCHOR_CSV = (
    REPO
    / "tracks/structural-sufficiency/zinc_performance_search_v1/analysis/per_molecule_valid.csv"
)
FIXED_WORST10_JSON = (
    REPO
    / "tracks/structural-sufficiency/zinc_generalization_audit_v1/analysis/fixed_worst10.json"
)
ROW172 = 172

#: Primary arm names.
ARMS = ("REAL", "NULL")
SEEDS = (0, 1)

PAIR_LOSS_COEF = 0.2
PAIRS_PER_STEP = 8


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def require_file(path: Path | str, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing {what}: {p}")
    return p


def load_train_cache():
    return D.load_cache("train")


def load_valid_cache():
    return D.load_cache("valid")


def load_norm():
    return D.load_norm()


def sha256_file(path: Path | str) -> str:
    return L.C.sha256_file(path)


def write_json(path: Path | str, obj) -> None:
    L.write_json(path, obj)


def read_json(path: Path | str):
    return L.read_json(path)


def ensure_dir(path: Path | str) -> Path:
    return L.ensure_dir(path)


def log(msg: str) -> None:
    L.log(msg)


def now() -> str:
    return L.C.utcnow()


def git_commit() -> str:
    return L.C.git_commit()


__all__ = [
    "ROUND_ID", "CODE_DIR", "EXP_DIR", "REPO", "ANALYSIS", "CONFIGS", "RUNS",
    "NOTES", "TESTS", "L", "D", "M", "LCRC_TRAIN_CACHE", "LCRC_VALID_CACHE",
    "BASE_CENTER_DIM", "BASE_R1_DIM", "BASE_R2_PATH_DIM", "BASE_R2_END_DIM",
    "BASE_R2_CLOSE_DIM", "BASE_R1_TOTAL", "BASE_INPUT_DIM", "VALID_ANCHOR_CSV",
    "FIXED_WORST10_JSON", "ROW172", "ARMS", "SEEDS", "PAIR_LOSS_COEF",
    "PAIRS_PER_STEP", "track_rel", "require_file", "load_train_cache",
    "load_valid_cache", "load_norm", "sha256_file", "write_json", "read_json",
    "ensure_dir", "log", "now", "git_commit",
]
