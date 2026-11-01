"""Shared constants, paths and frozen declarations for the MoleculeNet pilot.

This round is a *consumer* of two frozen upstream artifacts:

* the ``lipo_v1`` FeatureSpec / molecular-graph / data pipeline of
  ``lcrc_cross_task_preparation_v1`` (imported read-only, never modified);
* the frozen ``LCRC-A`` model of the same round (imported read-only).

The only new biology/chemistry code in this round is (a) a vendored copy of the
public ``shenwanxiang/ChemBench`` MoleculeNet CSVs and published split indices,
(b) an RDKit SMILES -> frozen raw-attribute adapter that reproduces the exact
attribute conventions the ``lipo_v1`` schema was defined on, and (c) a
regression/classification training + evaluation driver.

Nothing here contains a learned parameter, a label-derived statistic or a
split decision that depends on observed performance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROUND_ID = "lcrc-molnet-benchmark-pilot-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

#: Frozen upstream pipeline (read-only import).  Every file that is imported is
#: hashed into ``analysis/protocol_freeze.json``.
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
REUSED_MODULES = (
    "feature_spec.py",
    "molecular_graph.py",
    "adapters.py",
    "lcrc_data.py",
    "lcrc_model.py",
    "lcrc_common.py",
)

#: The ``lipo_v1`` FeatureSpec produced by the frozen upstream audit.  The spec
#: is rebuilt in-process from ``feature_spec.lipo_v1_spec()``; this JSON is only
#: used as a cross-check of the rebuilt hash.
FROZEN_SPEC_JSON = PREP / "analysis/feature_spec_lipo.json"

if str(PREP_CODE) not in sys.path:
    sys.path.insert(0, str(PREP_CODE))

DATA_DIR = EXP_DIR / "analysis/source"
CHEMBENCH_DIR = DATA_DIR / "chembench"
SOURCE_MANIFEST = DATA_DIR / "chembench_source_manifest.json"
ANALYSIS = EXP_DIR / "analysis"
CACHE = EXP_DIR / "cache"
RUNS = ANALYSIS / "runs"
TESTS = EXP_DIR / "tests"

# ---------------------------------------------------------------------------
# frozen benchmark declarations (pre-registered, never chosen after seeing a
# score)
# ---------------------------------------------------------------------------

#: ChemBench publishes three split repeats per dataset (seed suffixes 122/123/124).
#: ``load_data`` globs ``*.pkl`` and therefore has *no* guaranteed order.  The
#: round therefore freezes "split repeat 0" as the lexicographically first split
#: file (suffix ``122``), and records the sha256 of every repeat file so the
#: choice can be audited.  This is fixed before any label is loaded.
SPLIT_REPEAT_INDEX = 0
SPLIT_REPEAT_FILE_INDEX = 122

#: Training seed.  Deliberately a different concept from the split repeat index.
TRAIN_SEED = 0
SHUFFLE_OFFSET = 917_000

#: Frozen R3 / ESOL-BASE recipe (lcrc_lipo_best_epoch_distance_fix_v1 onwards).
RECIPE = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "dropout": 0.10,
    "dropout_position": "final atom-level local representation h, before the per-atom contribution f and the global rho",
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine",
    "soup": False,
    "ema": False,
    "swa": False,
    "ssl": False,
    "early_stopping": False,
    "hyperparameter_search": False,
}

#: Dataset registry.  ``label_column`` is the *measured* target column; no
#: derived column (predicted solubility, pIC50, ...) is ever read.
DATASETS: dict[str, dict[str, Any]] = {
    "ESOL": {
        "key": "ESOL",
        "chembench_task": "ESOL",
        "label_column": "measured log solubility in mols per litre",
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "logS (log10 mol/L, measured)",
        "split_method": "random",
        "expected_rows": 1128,
        "expected_split_sizes": (902, 113, 113),
        "literature_scale_note": "MoleculeNet ESOL, 1128 molecules (Delaney)",
    },
    "Lipophilicity": {
        "key": "Lipophilicity",
        "chembench_task": "Lipop",
        "label_column": "exp",
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "logD (log10 octanol/water distribution coefficient, experimental)",
        "split_method": "random",
        "expected_rows": 4200,
        "expected_split_sizes": (3360, 420, 420),
        "literature_scale_note": "MoleculeNet Lipophilicity, 4200 molecules (ChEMBL)",
    },
    "FreeSolv": {
        "key": "FreeSolv",
        "chembench_task": "FreeSolv",
        "label_column": "expt",
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "kcal/mol (experimental hydration free energy)",
        "split_method": "random",
        "expected_rows": 642,
        "expected_split_sizes": (513, 64, 65),
        "literature_scale_note": (
            "MoleculeNet FreeSolv cleansed set, 642 molecules; the original "
            "FreeSolv database has 643 entries. No molecule was added or removed here."
        ),
    },
    "BACE": {
        "key": "BACE",
        "chembench_task": "BACE",
        "label_column": "Class",
        "task": "classification",
        "primary_metric": "roc_auc",
        "metric_direction": "higher_is_better",
        "units": "binary class (1 = active/potent BACE-1 inhibitor)",
        "split_method": "scaffold",
        "expected_rows": 1513,
        "expected_split_sizes": (1210, 151, 152),
        "literature_scale_note": (
            "MoleculeNet BACE cleansed set, 1513 molecules; the original "
            "MoleculeNet release quotes 1522. No molecule was added or removed here."
        ),
    },
}

DATASET_ORDER = ("ESOL", "Lipophilicity", "FreeSolv", "BACE")

#: Old project artifacts used only for row-level / protocol comparison.  These
#: are read-only and are never overwritten.
OLD_LIPO_CSV = REPO / "data/lipophilicity/Lipophilicity.csv"
OLD_LIPO_OFFICIAL_SPLIT = (
    REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1/split_lipo_v1.json"
)
OLD_ESOL_DIR = REPO / "tracks/structural-sufficiency/lcrc_esol_quickstart_v1"
OLD_DIRS_READ_ONLY = (
    REPO / "tracks/structural-sufficiency/lcrc_esol_quickstart_v1",
    REPO / "tracks/structural-sufficiency/lcrc_esol_physchem_s2_v1",
    REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2",
    REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1",
)

#: Hard budget for this round.
GPU_HARD_CAP_HOURS = 1.5


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


def row_sha256(rows: Iterable[int]) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _json_default(obj: Any):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
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


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - non-git checkout
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
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def rdkit_version() -> str:
    import rdkit

    return str(rdkit.__version__)


def public_env_info() -> dict[str, Any]:
    """Environment info that must not require CUDA (CPU P0 stage)."""
    import pandas
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdkit_version(),
        "hostname": platform.node(),
    }


def reused_module_hashes() -> dict[str, str]:
    return {name: sha256_file(PREP_CODE / name) for name in REUSED_MODULES}


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "EXP_DIR",
    "REPO",
    "PREP",
    "PREP_CODE",
    "REUSED_MODULES",
    "FROZEN_SPEC_JSON",
    "DATA_DIR",
    "CHEMBENCH_DIR",
    "SOURCE_MANIFEST",
    "ANALYSIS",
    "CACHE",
    "RUNS",
    "TESTS",
    "SPLIT_REPEAT_INDEX",
    "SPLIT_REPEAT_FILE_INDEX",
    "TRAIN_SEED",
    "SHUFFLE_OFFSET",
    "RECIPE",
    "DATASETS",
    "DATASET_ORDER",
    "GPU_HARD_CAP_HOURS",
    "utcnow",
    "sha256_file",
    "sha256_bytes",
    "sha256_json",
    "row_sha256",
    "stable_json",
    "write_json",
    "read_json",
    "track_rel",
    "ensure_dir",
    "log",
    "git_commit",
    "git_dirty",
    "env_info",
    "public_env_info",
    "reused_module_hashes",
]
