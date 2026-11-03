"""Shared constants, frozen recipes and IO helpers for the fast positioning round.

Round: ``lcrc_benchmark_positioning_fast_v1``
Result label: ``EXPLORATORY_FAST_BENCHMARK``

Nothing in this module is fitted.  Every number here was fixed before the first
formal benchmark task was launched.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-benchmark-positioning-fast-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABEL = "EXPLORATORY_FAST_BENCHMARK"

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
TUQ = REPO / "tracks/structural-sufficiency/lcrc_tu_quick_screen_v1"
TUQ_CODE = TUQ / "code"
MB2 = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2"
MB2_CODE = MB2 / "code"

for _p in (str(PREP_CODE), str(TUQ_CODE), str(MB2_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = REPO / "data"
TU_ROOT = DATA / "TUD"
OGB_ROOT = DATA / "ogb"
LRGB_ROOT = DATA / "LRGB"

ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "cache"
RUNS = ROUND / "runs"
RESULTS = ROUND / "results"
CONFIGS = ROUND / "configs"
TESTS = ROUND / "tests"

PROTOCOL_JSON = CONFIGS / "protocol.json"

# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

#: Official graph/class counts used as load-time audit anchors.  They are the
#: published TUDataset sizes, not values inferred from a model result.
TU_DATASETS: dict[str, dict[str, Any]] = {
    "IMDB-BINARY": dict(n_graphs=1000, n_classes=2, node_type="degree", has_node_labels=False),
    "IMDB-MULTI": dict(n_graphs=1500, n_classes=3, node_type="degree", has_node_labels=False),
    "COLLAB": dict(n_graphs=5000, n_classes=3, node_type="degree", has_node_labels=False),
    "REDDIT-BINARY": dict(n_graphs=2000, n_classes=2, node_type="degree", has_node_labels=False),
    "MUTAG": dict(n_graphs=188, n_classes=2, node_type="label", has_node_labels=True),
    "NCI1": dict(n_graphs=4110, n_classes=2, node_type="label", has_node_labels=True),
    "PROTEINS": dict(n_graphs=1113, n_classes=2, node_type="label", has_node_labels=True),
    "DD": dict(n_graphs=1178, n_classes=2, node_type="label", has_node_labels=True),
    "ENZYMES": dict(n_graphs=600, n_classes=6, node_type="label", has_node_labels=True),
}

TU_ORDER = (
    "IMDB-BINARY",
    "IMDB-MULTI",
    "NCI1",
    "PROTEINS",
    "MUTAG",
    "DD",
    "ENZYMES",
    "REDDIT-BINARY",
    "COLLAB",
)

#: Datasets that trigger the section-6 large-graph precheck.
LARGE_GRAPH_DATASETS = ("COLLAB", "REDDIT-BINARY", "DD", "PEPTIDES-FUNC")
# Above this exact-S2 total, the eager per-path cache is not materialised:
# the run is blocked before formal training instead of silently sampling.
EAGER_S2_LIMIT = 20_000_000

MOL_DATASETS = ("ogbg-molhiv", "ogbg-molbace", "ogbg-moltox21")
MOL_SEEDS = (0, 1, 2)
PEPTIDE_DATASET = "peptides-func"
PEPTIDE_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

TU_N_SPLITS = 10
TU_OUTER_SPLIT_SEED = 20261104
TU_INNER_VALID_SEED_BASE = 1000
TU_MODEL_SEED_BASE = 100

# ---------------------------------------------------------------------------
# frozen model
# ---------------------------------------------------------------------------

MODEL_CONFIG: dict[str, Any] = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
}

#: The only authorised LCRC variant this round.
MODEL_VARIANT = "C1"

# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------

TU_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "peak_lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "min_lr": 1.0e-5,
    "dropout": 0.10,
    "batch_size": 64,
    "loss": "CrossEntropyLoss (unweighted)",
    "max_epochs": 120,
    "scheduler": "per_step_cosine",
    "scheduler_horizon_epochs": 200,
    "min_epochs": 40,
    "patience": 25,
    "gradient_clip": 5.0,
    "selection": "max Valid accuracy; ties -> lower Valid CE; ties -> earlier epoch",
    "early_stop": "after min_epochs, stop when 25 consecutive epochs do not strictly improve best Valid accuracy",
    "test_policy": "terminal: one evaluation after checkpoint hash is recorded",
}

OGB_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "peak_lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "min_lr": 1.0e-5,
    "dropout": 0.10,
    "batch_size": 128,
    "gradient_clip": 5.0,
    "scheduler": "per_step_cosine",
    "scheduler_horizon_epochs": 360,
    "test_policy": "terminal: one evaluation with the OGB official Evaluator after checkpoint hash is recorded",
}

#: Dataset-specific stopping rules from the execution instruction.
OGB_STOPPING: dict[str, dict[str, int]] = {
    "ogbg-molhiv": {"max_epochs": 110, "min_epochs": 60, "patience": 40},
    "ogbg-molbace": {"max_epochs": 150, "min_epochs": 50, "patience": 35},
    "ogbg-moltox21": {"max_epochs": 120, "min_epochs": 40, "patience": 30},
}

PEPTIDE_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "peak_lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "min_lr": 1.0e-5,
    "dropout": 0.10,
    "batch_size": 64,
    "batch_size_fallbacks": (32, 16),
    "loss": "BCEWithLogitsLoss over 10 independent labels",
    "max_epochs": 120,
    "scheduler": "per_step_cosine",
    "scheduler_horizon_epochs": 120,
    "min_epochs": 40,
    "patience": 25,
    "gradient_clip": 5.0,
    "selection": "max Validation mean average precision (macro AP over 10 labels)",
    "early_stop": "after min_epochs, stop when 25 consecutive epochs do not strictly improve best Validation mean AP",
    "test_policy": "terminal: one evaluation after checkpoint hash is recorded",
}

# ---------------------------------------------------------------------------
# budgets (measured from the start of the formal GPU queue)
# ---------------------------------------------------------------------------

BUDGET: dict[str, Any] = {
    "wall_soft_hours": 6.0,
    "wall_hard_hours": 10.0,
    "gpu_total_hours": 18.0,
    "per_task_minutes": 25.0,
    "preprocess_per_dataset_minutes": 30.0,
}

# ---------------------------------------------------------------------------
# user-provided screenshot reference (protocol unverified)
# ---------------------------------------------------------------------------

SCREENSHOT_LABEL = "SCREENSHOT_REFERENCE_PROTOCOL_UNVERIFIED"
REFERENCE: dict[str, float] = {
    "IMDB-BINARY": 73.87,
    "IMDB-MULTI": 50.97,
    "COLLAB": 80.72,
    "REDDIT-BINARY": 91.14,
    "MUTAG": 89.20,
    "NCI1": 83.09,
    "PROTEINS": 77.85,
    "DD": 79.48,
    "ENZYMES": 69.73,
    "ogbg-molhiv": 81.47,
    "ogbg-molbace": 81.67,
    "ogbg-moltox21": 76.67,
    "peptides-func": 71.57,
}

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


def sha256_array(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def index_hash(rows: Iterable[int]) -> str:
    return sha256_array(np.asarray(list(rows), dtype=np.int64))


def _json_default(obj: Any):
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
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def write_json(path: Path | str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


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


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


def seed_everything(seed: int) -> dict[str, Any]:
    import random

    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {"seed": int(seed), "deterministic_algorithms": False}


def result_paths(kind: str, dataset: str, unit: int) -> Path:
    """Canonical run directory for one formal task."""
    if kind == "tu":
        tag = f"tu_{dataset.lower().replace('-', '_')}_fold{int(unit):02d}"
    elif kind == "mol":
        tag = f"mol_{dataset.replace('ogbg-', '')}_seed{int(unit)}"
    elif kind == "peptide":
        tag = f"peptide_{dataset}_seed{int(unit)}"
    else:
        raise KeyError(kind)
    return RUNS / tag


def task_id(kind: str, dataset: str, unit: int) -> str:
    return result_paths(kind, dataset, unit).name


def plan_entry(kind: str, dataset: str, unit: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "dataset": dataset,
        "unit": int(unit),
        "task_id": task_id(kind, dataset, unit),
        "cost_class": "large" if dataset in LARGE_GRAPH_DATASETS else "small",
    }
