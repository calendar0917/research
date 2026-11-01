"""Shared constants, paths and frozen declarations for Sprint 02 (v2).

Round id: ``lcrc-molnet-benchmark-pilot-v2``.

This module is the single place where every *pre-registered* quantity of the
round lives.  Nothing here is chosen after seeing a score.

What is reused (read-only) from earlier rounds
---------------------------------------------
* ``lcrc_cross_task_preparation_v1/code``: the frozen ``lipo_v1`` FeatureSpec,
  the molecular-graph builder, the cache/collate layer and the frozen
  ``LCRC`` model.  Imported, never modified.
* ``lcrc_molnet_benchmark_pilot_v1``: the vendored ChemBench bytes of the four
  Sprint-01 datasets (read-only), and the *behaviour* of its RDKit raw-attribute
  adapter, which ``mb2_rdkit`` reproduces bit-exactly (asserted by a test
  against ``raw_attributes_v1.json.gz``).

What is new in v2
-----------------
* BBBP (scaffold) and Tox21 (random, 12 endpoints) from the same hash-pinned
  ChemBench commit, plus the official ``ogbg-molhiv`` scaffold split.
* A 12-output variant of the frozen model head (``f`` and ``rho`` only).
* Opt-in recipe overrides for the four Validation-only P3 candidates, guarded by
  a frozen allowlist (``PROBE_CANDIDATES``): a run that deviates from R3 in any
  other way is refused by ``mb2_train``.
* A matched ECFP4 + RDKit-descriptor + XGBoost baseline on the identical rows.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROUND_ID = "lcrc-molnet-benchmark-pilot-v2"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

# ---------------------------------------------------------------------------
# frozen upstream code (read-only imports)
# ---------------------------------------------------------------------------

V1 = REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1"
V1_CODE = V1 / "code"
PREP = REPO / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"

#: every imported upstream file is hashed into ``analysis/protocol_freeze.json``
REUSED_MODULES = {
    "prep/feature_spec.py": PREP_CODE / "feature_spec.py",
    "prep/molecular_graph.py": PREP_CODE / "molecular_graph.py",
    "prep/adapters.py": PREP_CODE / "adapters.py",
    "prep/lcrc_data.py": PREP_CODE / "lcrc_data.py",
    "prep/lcrc_model.py": PREP_CODE / "lcrc_model.py",
    "prep/lcrc_common.py": PREP_CODE / "lcrc_common.py",
    "v1/mb_rdkit.py": V1_CODE / "mb_rdkit.py",
    "v1/mb_source.py": V1_CODE / "mb_source.py",
    "v1/mb_literature.py": V1_CODE / "mb_literature.py",
}

for _p in (str(PREP_CODE), str(V1_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: the ``lipo_v1`` FeatureSpec produced by the frozen upstream audit
FROZEN_SPEC_JSON = PREP / "analysis/feature_spec_lipo.json"

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

ANALYSIS = EXP_DIR / "analysis"
SOURCE_DIR = ANALYSIS / "source"
CHEMBENCH_V2 = SOURCE_DIR / "chembench"
CHEMBENCH_V1 = V1 / "analysis/source/chembench"
OGB_SOURCE = SOURCE_DIR / "ogb/ogbg_molhiv"
CACHE = EXP_DIR / "cache"
RUNS = ANALYSIS / "runs"
BASELINE_DIR = ANALYSIS / "baseline"
TESTS = EXP_DIR / "tests"

SOURCE_MANIFEST = ANALYSIS / "source_manifest.json"
RAW_ARTIFACT = SOURCE_DIR / "raw_attributes_v2.json.gz"
RAW_ARTIFACT_MANIFEST = SOURCE_DIR / "raw_artifact_manifest.json"

#: big local-only inputs/outputs (never committed as data)
OGB_LOCAL_ROOT = REPO / "data/ogb/ogbg_molhiv"

# ---------------------------------------------------------------------------
# frozen benchmark declarations
# ---------------------------------------------------------------------------

#: ChemBench publishes three split repeats per dataset (seed suffixes 122/123/124).
#: ``chembench.load_data`` globs ``*.pkl`` with no guaranteed order, so "repeat 0"
#: is frozen as *the lexicographically first published split file* (suffix 122).
#: Fixed before any label is read.
SPLIT_REPEAT_INDEX = 0
SPLIT_REPEAT_FILE_INDEX = 122

#: Training seed.  A different concept from the split repeat index.
TRAIN_SEED = 0
SHUFFLE_OFFSET = 917_000

#: Frozen R3 / ESOL-BASE recipe (unchanged from Sprint 01).
RECIPE: dict[str, Any] = {
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

#: The four frozen P3 validation-only probes.  ``overrides`` is the *only* way a
#: run may deviate from ``RECIPE``; ``mb2_train`` refuses anything else.
PROBE_CANDIDATES: dict[str, dict[str, Any]] = {
    "fs_dropout00": {
        "dataset": "FreeSolv",
        "overrides": {"dropout": 0.0},
        "hypothesis": "dropout 0.10 -> 0.00",
    },
    "fs_dropout20": {
        "dataset": "FreeSolv",
        "overrides": {"dropout": 0.20},
        "hypothesis": "dropout 0.10 -> 0.20",
    },
    "bace_dropout20": {
        "dataset": "BACE",
        "overrides": {"dropout": 0.20},
        "hypothesis": "dropout 0.10 -> 0.20",
    },
    "bace_wd1e2": {
        "dataset": "BACE",
        "overrides": {"weight_decay": 1.0e-2},
        "hypothesis": "weight decay 1e-3 -> 1e-2",
    },
}
PROBE_ORDER = ("fs_dropout00", "fs_dropout20", "bace_dropout20", "bace_wd1e2")

#: Fixed Tox21 endpoint order = the published CSV column order.  Never reordered.
TOX21_TASKS: tuple[str, ...] = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)

#: Matched traditional baseline: one frozen configuration, no tuning at all.
BASELINE_CONFIG: dict[str, Any] = {
    "features": "Morgan ECFP4 (radius 2, 2048 bits) + RDKit 2D descriptors (frozen list)",
    "ecfp_radius": 2,
    "ecfp_n_bits": 2048,
    "descriptor_list": "frozen tuple RDKIT_DESCRIPTORS in mb2_baseline.py",
    "xgboost": {
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "min_child_weight": 1,
        "random_state": 0,
    },
    "imputation": "per-descriptor median fitted on the Train split only; constant columns dropped by Train-only variance",
    "scaling": "none (tree model; descriptors are not standardised)",
    "hyperparameter_search": False,
    "validation_used_for_any_baseline_choice": False,
}

#: Hard budget for this round.
GPU_HARD_CAP_HOURS = 2.0

#: Metric re-verification tolerance (CUDA reductions are not bit-reproducible).
REPRO_TOL = 1e-5

# ---------------------------------------------------------------------------
# dataset registry (7 tasks)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "ESOL": {
        "source": "chembench",
        "chembench_task": "ESOL",
        "vendor": "v1",
        "split_stem": "ESOL.random",
        "label_columns": ("measured log solubility in mols per litre",),
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "logS (log10 mol/L, measured)",
        "split_method": "random",
        "expected_rows": 1128,
        "expected_split_sizes": (902, 113, 113),
        "n_tasks": 1,
        "round": "S1 (Test exposed)",
        "literature_scale_note": "MoleculeNet ESOL, 1128 molecules (Delaney)",
    },
    "Lipophilicity": {
        "source": "chembench",
        "chembench_task": "Lipop",
        "vendor": "v1",
        "split_stem": "Lipop.random",
        "label_columns": ("exp",),
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "logD (log10 octanol/water distribution coefficient, experimental)",
        "split_method": "random",
        "expected_rows": 4200,
        "expected_split_sizes": (3360, 420, 420),
        "n_tasks": 1,
        "round": "S1 (Test exposed)",
        "literature_scale_note": "MoleculeNet Lipophilicity, 4200 molecules (ChEMBL)",
    },
    "FreeSolv": {
        "source": "chembench",
        "chembench_task": "FreeSolv",
        "vendor": "v1",
        "split_stem": "FreeSolv.random",
        "label_columns": ("expt",),
        "task": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower_is_better",
        "units": "kcal/mol (experimental hydration free energy)",
        "split_method": "random",
        "expected_rows": 642,
        "expected_split_sizes": (513, 64, 65),
        "n_tasks": 1,
        "round": "S1 (Test exposed)",
        "literature_scale_note": (
            "MoleculeNet FreeSolv cleansed set, 642 molecules; the FreeSolv database has 643. "
            "No molecule was added or removed here."
        ),
    },
    "BACE": {
        "source": "chembench",
        "chembench_task": "BACE",
        "vendor": "v1",
        "split_stem": "BACE.scaffold",
        "label_columns": ("Class",),
        "task": "binary",
        "primary_metric": "roc_auc",
        "metric_direction": "higher_is_better",
        "units": "binary class (1 = active BACE-1 inhibitor)",
        "split_method": "scaffold",
        "expected_rows": 1513,
        "expected_split_sizes": (1210, 151, 152),
        "n_tasks": 1,
        "round": "S1 (Test exposed)",
        "literature_scale_note": (
            "MoleculeNet BACE cleansed set, 1513 molecules; the MoleculeNet release quotes 1522. "
            "No molecule was added or removed here."
        ),
    },
    "BBBP": {
        "source": "chembench",
        "chembench_task": "BBBP",
        "vendor": "v2",
        "split_stem": "BBBP.scaffold",
        "label_columns": ("p_np",),
        "task": "binary",
        "primary_metric": "roc_auc",
        "metric_direction": "higher_is_better",
        "units": "binary class (1 = blood-brain barrier permeant)",
        "split_method": "scaffold",
        "expected_rows": 2039,
        "expected_split_sizes": (1631, 204, 204),
        "n_tasks": 1,
        "round": "S2 new (Test not yet read)",
        "literature_scale_note": "MoleculeNet BBBP, 2039 molecules (Martins et al.)",
    },
    "Tox21": {
        "source": "chembench",
        "chembench_task": "Tox21",
        "vendor": "v2",
        "split_stem": "Tox21.random",
        "label_columns": TOX21_TASKS,
        "task": "multitask",
        "primary_metric": "macro_roc_auc",
        "metric_direction": "higher_is_better",
        "units": "12 binary toxicity endpoints; NaN = not assayed (never 0)",
        "split_method": "random",
        "expected_rows": 7831,
        "expected_split_sizes": (6264, 783, 784),
        "n_tasks": 12,
        "round": "S2 new (Test not yet read)",
        "literature_scale_note": "MoleculeNet Tox21, 7831 molecules / 12 endpoints (NIH Tox21 challenge)",
    },
    "ogbg-molhiv": {
        "source": "ogb",
        "chembench_task": None,
        "vendor": "v2",
        "split_stem": None,
        "label_columns": ("HIV_active",),
        "task": "binary",
        "primary_metric": "roc_auc",
        "metric_direction": "higher_is_better",
        "units": "binary class (1 = HIV active)",
        "split_method": "scaffold (OGB official)",
        "expected_rows": 41127,
        "expected_split_sizes": (32901, 4113, 4113),
        "n_tasks": 1,
        "round": "S2 new (Test not yet read)",
        "literature_scale_note": "ogbg-molhiv, 41127 molecules (OGB release v1, May 2020)",
    },
}

DATASET_ORDER: tuple[str, ...] = (
    "ESOL",
    "Lipophilicity",
    "FreeSolv",
    "BACE",
    "BBBP",
    "Tox21",
    "ogbg-molhiv",
)

#: datasets whose Test labels are already public knowledge from Sprint 01
EXPOSED_TEST_DATASETS: tuple[str, ...] = ("ESOL", "Lipophilicity", "FreeSolv", "BACE")
#: datasets whose first Test read happens in this round
NEW_TEST_DATASETS: tuple[str, ...] = ("BBBP", "Tox21", "ogbg-molhiv")

#: read-only earlier outputs (never written by this round)
OLD_V1_RUNS = V1 / "analysis/runs"
OLD_V1_BENCHMARK_TABLE = V1 / "analysis/benchmark_table.csv"
OLD_DIRS_READ_ONLY = (
    V1,
    PREP,
    REPO / "tracks/structural-sufficiency/lcrc_esol_quickstart_v1",
    REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2",
)


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


def reused_module_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in REUSED_MODULES.items()}


def code_hashes() -> dict[str, str]:
    out = {f"v2/{p.name}": sha256_file(p) for p in sorted(CODE_DIR.glob("mb2_*.py"))}
    out.update(reused_module_hashes())
    return out


def public_env_info() -> dict[str, Any]:
    import pandas
    import sklearn
    import torch
    import xgboost

    import rdkit

    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "rdkit": str(rdkit.__version__),
        "hostname": platform.node(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        out["cuda_device_name"] = torch.cuda.get_device_name(0)
        out["cuda_device_count"] = torch.cuda.device_count()
    return out


def dataset_entry(key: str) -> dict[str, Any]:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; expected one of {DATASET_ORDER}")
    return DATASETS[key]


def recipe_for(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """R3 plus an allowlisted subset of overrides."""
    recipe = dict(RECIPE)
    if not overrides:
        return recipe
    allowed = {"dropout", "weight_decay"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"recipe overrides outside the frozen allowlist {sorted(allowed)}: {sorted(unknown)}")
    for k, v in overrides.items():
        recipe[k] = v
    return recipe


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "EXP_DIR",
    "REPO",
    "V1",
    "V1_CODE",
    "PREP",
    "PREP_CODE",
    "REUSED_MODULES",
    "FROZEN_SPEC_JSON",
    "ANALYSIS",
    "SOURCE_DIR",
    "CHEMBENCH_V2",
    "CHEMBENCH_V1",
    "OGB_SOURCE",
    "OGB_LOCAL_ROOT",
    "CACHE",
    "RUNS",
    "BASELINE_DIR",
    "TESTS",
    "SOURCE_MANIFEST",
    "RAW_ARTIFACT",
    "RAW_ARTIFACT_MANIFEST",
    "SPLIT_REPEAT_INDEX",
    "SPLIT_REPEAT_FILE_INDEX",
    "TRAIN_SEED",
    "SHUFFLE_OFFSET",
    "RECIPE",
    "PROBE_CANDIDATES",
    "PROBE_ORDER",
    "TOX21_TASKS",
    "BASELINE_CONFIG",
    "GPU_HARD_CAP_HOURS",
    "REPRO_TOL",
    "DATASETS",
    "DATASET_ORDER",
    "EXPOSED_TEST_DATASETS",
    "NEW_TEST_DATASETS",
    "OLD_V1_RUNS",
    "OLD_V1_BENCHMARK_TABLE",
    "OLD_DIRS_READ_ONLY",
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
    "reused_module_hashes",
    "code_hashes",
    "public_env_info",
    "dataset_entry",
    "recipe_for",
]
