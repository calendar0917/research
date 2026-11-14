"""Frozen constants, paths and arm registry for ``lcrc-lipo-context-small-s0-v1``.

Round: *LCRC small-data first: Lipophilicity performance and structural failure
diagnosis*.  One dataset (OGB ``ogbg-mollipo``, official scaffold split), one
seed (0), six formal arms (B, S, R1, R2, D2, G), development-evaluation access
to the official Test split (declared in ``protocols/``, never used for
selection).

Everything scientific (splits, rows, recipe, parameter counts, tolerances,
group definitions, bootstrap design) is frozen here *before* the first formal
run.  Anything that changes it is a new round.

Read-only reuse (never edited, hashed into every run record):

* ``lcrc_lipo_model_dissection_v1/code``  - the frozen C1 model / R3 recipe
* ``lcrc_cross_task_preparation_v1/code`` - feature spec, graphs, caches, collate
* ``lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py`` - ``MessageFFN``
* ``lcrc_pcba_dynamic_v1/code/dy_model.py`` - private-init / provenance idea
* ``lcrc_performance_route_8h_public122_s0_v1/code/lcpr_dmpnn.py`` (+ vendor)
* ``lcrc_bottleneck_diagnosis_public122_s0_v1/code`` - E4-corrected schedule
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# determinism must be configured before the first CUDA call
# ---------------------------------------------------------------------------

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROUND_ID = "lcrc-lipo-context-small-s0-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]
SS = REPO / "tracks/structural-sufficiency"

DISSECT = SS / "lcrc_lipo_model_dissection_v1"
PREP = SS / "lcrc_cross_task_preparation_v1"
PILOT = SS / "lcrc_molnet_benchmark_pilot_v1"
SGS = SS / "lcrc_pcba_structural_gain_sprint_v1"
DYN = SS / "lcrc_pcba_dynamic_v1"
PERF = SS / "lcrc_performance_route_8h_public122_s0_v1"
BOTTLE = SS / "lcrc_bottleneck_diagnosis_public122_s0_v1"

DISSECT_CODE = DISSECT / "code"
PREP_CODE = PREP / "code"
PILOT_CODE = PILOT / "code"
SGS_CODE = SGS / "code"
DYN_CODE = DYN / "code"
PERF_CODE = PERF / "code"
BOTTLE_CODE = BOTTLE / "code"

#: Order matters only for import side effects; each entry is inserted once.
for _p in (BOTTLE_CODE, PERF_CODE, DYN_CODE, SGS_CODE, PILOT_CODE, PREP_CODE, DISSECT_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: Frozen upstream modules consumed read-only (hashed into every run record).
REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_train.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
    "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code/sg_model.py",
    "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_model.py",
    "tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1/code/lcpr_dmpnn.py",
    "tracks/structural-sufficiency/lcrc_bottleneck_diagnosis_public122_s0_v1/code/lb_common.py",
    "tracks/structural-sufficiency/lcrc_bottleneck_diagnosis_public122_s0_v1/code/lb_train.py",
)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

DATA = TRACK / "data"
RAW_ARTIFACT = DATA / "raw_attributes_v1.json.gz"
LABELS = DATA / "labels_v1.npz"
SPLIT_JSON = DATA / "split_v1.json"
DMPNN_FEATURES = DATA / "dmpnn_features_v1.npz"
DATA_MANIFEST = DATA / "manifest.json"

ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
CACHE = TRACK / "cache"
RUNS = TRACK / "runs"
TESTS = TRACK / "tests"

RESULTS = ANALYSIS / "results"          # pulled per-arm run outputs
TABLES = ANALYSIS / "tables"
FIGURES = ANALYSIS / "figures"
GROUPS = ANALYSIS / "groups"
INTERVENTIONS = ANALYSIS / "interventions"

# ---------------------------------------------------------------------------
# dataset (OGB ogbg-mollipo, official scaffold split)
# ---------------------------------------------------------------------------

OGB_NAME = "ogbg-mollipo"
OGB_MASTER_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
OGB_MASTER_CSV_URL = (
    "https://raw.githubusercontent.com/snap-stanford/ogb/"
    f"{OGB_MASTER_COMMIT}/ogb/graphproppred/master.csv"
)
OGB_ZIP_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/lipophilicity.zip"
OGB_ZIP_SHA256 = "c58d285a55af5d383483a8c9167030e2beea94041ddbf69c39d2a193548a25d0"
OGB_DOWNLOAD_NAME = "lipophilicity"
OGB_TASK_TYPE = "regression"
OGB_EVAL_METRIC = "rmse"
OGB_SPLIT_METHOD = "scaffold"

DATASET = "ogbg-mollipo"
TASK = "regression"
TARGET_COLUMN = "exp"
UNITS = "logD (log10 octanol/water distribution coefficient)"
N_MOLECULES = 4200
SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
SPLITS = ("train", "valid", "test")

#: Frozen feature spec / cache geometry (inherited from ``lipo_v1``).
FEATURE_SPEC_HASH = "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee"
ATOM_FEAT_DIM = 51
BOND_FEAT_DIM = 13
ATOM_SCALAR_DIM = 9
GLOBAL_DIM = 42
LOCAL_DIM = 529
RHO_DIM = 171

TRAIN_SEED = 0
SEEDS = (0,)

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

ARMS = ("B", "S", "R1", "R2", "D2", "G")
#: Arms built on the frozen C1 base (every one of them shares the same init).
CONTEXT_ARMS = ("S", "R1", "R2", "D2")
BASE_ARM = "B"
DM_ARM = "G"

ARM_TITLES = {
    "B": "B: frozen C1 base (no context exchange)",
    "S": "S: same-parameter local computation (source = target itself)",
    "R1": "R1: repeat the original neighbours (both layers read h0_j)",
    "R2": "R2: fixed extended source c_j at layer 2 (h0 + 1-hop mean)",
    "D2": "D2: dynamic neighbour state (layer 2 reads h1[src])",
    "G": "G: vendored chemprop v2.2.1 D-MPNN reference (independent coordinate)",
}
ARM_SPECS: dict[str, dict[str, Any]] = {
    "B": {"kind": "base"},
    "S": {"kind": "context", "source": "same_node"},
    "R1": {"kind": "context", "source": "repeat_h0"},
    "R2": {"kind": "context", "source": "fixed_extended"},
    "D2": {"kind": "context", "source": "dynamic"},
    "G": {"kind": "dmpnn"},
}
#: Pre-registered trainable parameter counts (a mismatch is a bug, not a knob).
EXPECTED_PARAMS = {
    "B": 228_674,
    "S": 262_978,
    "R1": 262_978,
    "R2": 262_978,
    "D2": 262_978,
}
BASE_PARAMS = 228_674
CONTEXT_PARAMS = 262_978
NEW_PARAMS = CONTEXT_PARAMS - BASE_PARAMS          # 34,304
BOND_PROJ_PARAMS = 13 * 64 + 64                    # 896
MESSAGE_FFN_PARAMS = 16_704                        # LN64 + 64->128 + 128->64
N_CONTEXT_LAYERS = 2
WIDTH = 64
BLOCK_HIDDEN = 128
PRIVATE_INIT_SEED = 17

#: Max source hop relative to h0 that each arm can see at layer 2.
ARM_SOURCE_HOPS = {"B": 0, "S": 0, "R1": 1, "R2": 2, "D2": 2, "G": -1}  # G: not applicable (independent recipe)

# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

#: Frozen R3 recipe (identical to ``dissect_common.RECIPE``; asserted in code).
C1_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "dropout": 0.10,
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine",
    "shuffle_offset": 917_000,
    "soup": False,
    "ema": False,
    "swa": False,
    "ssl": False,
    "early_stopping": False,
    "hyperparameter_search": False,
}

#: Frozen D-MPNN reference recipe (chemprop 2.2.1 coordinate, ERRATA-E4 schedule).
DMPNN_RECIPE: dict[str, Any] = {
    "optimizer": "Adam",
    "batch_size": 64,
    "epochs": 100,
    "init_lr": 1.0e-4,
    "peak_lr": 1.0e-3,
    "min_lr": 1.0e-4,
    "weight_decay": 0.0,
    "gradient_clip": 5.0,
    "warmup_epochs": 2,
    "decay": "exponential",
    "optimizer_base_lr": 1.0e-4,   # == init_lr: LambdaLR base lr is the init lr
    "hidden": 300,
    "depth": 3,
}

EPOCHS = {"B": 360, "S": 360, "R1": 360, "R2": 360, "D2": 360, "G": 100}
#: Averaging window for the "late-phase mean" summary (G is a 100-epoch recipe).
LATE_WINDOW = {"B": 20, "S": 20, "R1": 20, "R2": 20, "D2": 20, "G": 10}

#: Evaluation during training: every completed epoch, all three full splits.
EVAL_EVERY_EPOCH = True
EVAL_BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# judgements (pre-registered thresholds, in raw logD units / relative)
# ---------------------------------------------------------------------------

PROMISE_REL_IMPROVEMENT = 0.02       # both Val and Test RMSE at the best-Val point
PROMISE_MAE_TOLERANCE = 0.01         # MAE may not worsen by more than this fraction
TAIL_TRIM_FRACTIONS = (0.01, 0.05)   # worst-row sensitivity sets
GROUP_GAP_THRESHOLD = 0.05           # raw logD, size-standardised MAE gap
GROUP_MIN_SIDE = 40                  # molecules on each side
GROUP_MIN_BIN_SIDE = 10              # molecules per side per size bin
GROUP_MIN_BINS = 2
GROUP_SUPPORT_MIN = 5                # H2: radius-2 support >= 5
MORGAN_RADIUS = 3
MORGAN_RADIUS_FINE = (2, 3)

BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_WEIGHT_DIST = "Exp(1)"

GPU_SOFT_CAP_HOURS = 1.5
GPU_HARD_CAP_HOURS = 2.5

REPRO_TOL = 1e-5
ZERO_DRIFT_TOL = 1e-6
NUMERICAL_ALARM = "NUMERICAL_PREFLIGHT_BLOCKED"

#: Row -> plan hash generator (shared epoch plan for all C1-family arms).
SHUFFLE_OFFSET = 917_000


# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))


def _json_default(obj: Any):
    import numpy as np

    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def row_sha256(rows: Iterable[int]) -> str:
    import numpy as np

    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def env_info() -> dict[str, Any]:
    import numpy as np
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    try:
        import rdkit

        info["rdkit"] = str(rdkit.__version__)
    except Exception:  # pragma: no cover
        info["rdkit"] = None
    return info


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for p in sorted(CONFIGS.glob("*")):
        if p.is_file():
            out[track_rel(p)] = sha256_file(p)
    for rel in REUSED_MODULES:
        target = REPO / rel
        if target.is_file():
            out[rel] = sha256_file(target)
    return out


def setup_determinism(torch_module=None, *, strict: bool = True) -> dict[str, Any]:
    """Deterministic CUDA configuration (TF32 off, benchmark off, CUBLAS ws)."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(strict))
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


# ---------------------------------------------------------------------------
# arm helpers
# ---------------------------------------------------------------------------


def arm_epochs(arm: str) -> int:
    return int(EPOCHS[arm])


def late_window(arm: str) -> int:
    return int(LATE_WINDOW[arm])


def is_context(arm: str) -> bool:
    return arm in CONTEXT_ARMS


def run_name(arm: str) -> str:
    return f"lipo_ctx_{arm}_s{TRAIN_SEED}"


def arm_payload(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "title": ARM_TITLES[arm],
        "spec": dict(ARM_SPECS[arm]),
        "epochs": arm_epochs(arm),
        "late_window": late_window(arm),
        "expected_parameters": EXPECTED_PARAMS.get(arm),
        "seed": TRAIN_SEED,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
