"""Shared constants, paths and provenance for the LCRC performance long-run.

Round id: ``lcrc-performance-longrun-v1`` (Sprint 04).

The round asks which bottleneck limits LCRC-A on real molecular-property tasks:
local chemistry completeness, global-statistics compression, or the single
additive contribution readout.  Three single-change candidates are screened on
three anchors, then combined only if the screen supports it.

Everything imported from earlier rounds is read-only:

* ``lcrc_molnet_benchmark_pilot_v2/code``: the frozen ``lipo_v1`` data path,
  the seven-dataset registry, the R3 recipe, metrics and the frozen LCRC-A
  model.
* ``lcrc_taskaware_representation_pilot_v1/code``: the published Lipophilicity
  repeat loaders (``random123`` / ``random124``).
* ``lcrc_cross_task_preparation_v1/code``: the cache/collate/model layer.

The new code in this directory adds only the pre-registered model arms and the
8-channel local-chemistry extension.  It never edits an upstream file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-performance-longrun-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
REPO = ROUND.parents[2]

STRUCT = REPO / "tracks/structural-sufficiency"
V1 = STRUCT / "lcrc_molnet_benchmark_pilot_v1"
V2 = STRUCT / "lcrc_molnet_benchmark_pilot_v2"
PREP = STRUCT / "lcrc_cross_task_preparation_v1"
TASK03 = STRUCT / "lcrc_taskaware_representation_pilot_v1"
V1_CODE = V1 / "code"
V2_CODE = V2 / "code"
PREP_CODE = PREP / "code"
TASK03_CODE = TASK03 / "code"

for _p in (str(CODE_DIR), str(V2_CODE), str(TASK03_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mb2_common as V2C  # noqa: E402  (also puts PREP_CODE and V1_CODE on sys.path)

ANALYSIS = ROUND / "analysis"
RUNS = ANALYSIS / "runs"
CACHE = ROUND / "cache"
TESTS = ROUND / "tests"
SOURCE_CHEM8 = ANALYSIS / "source/chem8"

PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
CHECKPOINT_MANIFEST = ANALYSIS / "checkpoint_manifest.json"
TEST_LEDGER = ANALYSIS / "test_access_ledger.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"
RUN_REGISTRY = ANALYSIS / "rr_run_registry.json"
CHEM8_MANIFEST = ANALYSIS / "chem8_manifest.json"

SEED = 0
SHUFFLE_OFFSET = int(V2C.SHUFFLE_OFFSET)
REPRO_TOL = 1e-5

#: The common recipe.  HIV uses the pre-registered short protocol in §14 of the
#: task book; every other dataset uses 360 epochs.
RECIPE: dict[str, Any] = dict(V2C.RECIPE)

HIV_RECIPE: dict[str, Any] = {
    **RECIPE,
    "epochs": 160,
    "minimum_epochs": 60,
    "early_stop_patience": 30,
    "scheduler_horizon_epochs": 160,
}

#: Anchors (P1) and transfer datasets (P4).
ANCHORS: tuple[str, ...] = ("Lipophilicity", "ESOL", "BACE")
TRANSFER: tuple[str, ...] = (
    "Lipophilicity",  # random123 / random124
    "FreeSolv",
    "BBBP",
    "Tox21",
    "ogbg-molhiv",
)

#: Frozen arms.  A run key must name exactly one of these.
ARMS: tuple[str, ...] = (
    "base",
    "chem8",
    "rhoshape",
    "multicontrib",
    "combo_chem_contrib",
    "combo_chem_rho",
    "combo_rho_contrib",
)

ARM_LABEL = {
    "base": "LCRC-A (BASE)",
    "chem8": "LCRC-CHEMLOCAL8",
    "rhoshape": "LCRC-RHOSHAPE",
    "multicontrib": "LCRC-MULTICONTRIB8",
    "combo_chem_contrib": "LCRC-COMBO (CHEMLOCAL8 + MULTICONTRIB8)",
    "combo_chem_rho": "LCRC-COMBO (CHEMLOCAL8 + RHOSHAPE)",
    "combo_rho_contrib": "LCRC-COMBO (RHOSHAPE + MULTICONTRIB8)",
}

#: SCALE-M start point from the task book.  The effective parameter count is
#: checked by ``tests/test_models.py`` and only ``local_hidden`` may be nudged.
SCALE_M_CONFIG: dict[str, int] = {
    "width": 80,
    "stat_rank": 40,
    "local_hidden": 230,
    "atom_hidden": 144,
    "atom_out": 104,
    "bond_hidden": 176,
    "bond_out": 104,
    "rho_hidden": 80,
}

PLANNED_RUNS: tuple[dict[str, Any], ...] = (
    # P1 screening: 9 runs, seed 0.
    {"key": "p1_chem8_lipo", "stage": "P1", "dataset": "Lipophilicity", "split": "published", "arm": "chem8", "seed": 0},
    {"key": "p1_rho_lipo", "stage": "P1", "dataset": "Lipophilicity", "split": "published", "arm": "rhoshape", "seed": 0},
    {"key": "p1_contrib_lipo", "stage": "P1", "dataset": "Lipophilicity", "split": "published", "arm": "multicontrib", "seed": 0},
    {"key": "p1_chem8_esol", "stage": "P1", "dataset": "ESOL", "split": "published", "arm": "chem8", "seed": 0},
    {"key": "p1_rho_esol", "stage": "P1", "dataset": "ESOL", "split": "published", "arm": "rhoshape", "seed": 0},
    {"key": "p1_contrib_esol", "stage": "P1", "dataset": "ESOL", "split": "published", "arm": "multicontrib", "seed": 0},
    {"key": "p1_chem8_bace", "stage": "P1", "dataset": "BACE", "split": "published", "arm": "chem8", "seed": 0},
    {"key": "p1_rho_bace", "stage": "P1", "dataset": "BACE", "split": "published", "arm": "rhoshape", "seed": 0},
    {"key": "p1_contrib_bace", "stage": "P1", "dataset": "BACE", "split": "published", "arm": "multicontrib", "seed": 0},
    # P3 scaling: BASE and the frozen winner at two anchors, 4 runs.
    {"key": "p3_base_scalem_lipo", "stage": "P3", "dataset": "Lipophilicity", "split": "published", "arm": "base", "seed": 0, "scale": "M"},
    {"key": "p3_winner_scalem_lipo", "stage": "P3", "dataset": "Lipophilicity", "split": "published", "arm": "WINNER", "seed": 0, "scale": "M"},
    {"key": "p3_base_scalem_esol", "stage": "P3", "dataset": "ESOL", "split": "published", "arm": "base", "seed": 0, "scale": "M"},
    {"key": "p3_winner_scalem_esol", "stage": "P3", "dataset": "ESOL", "split": "published", "arm": "WINNER", "seed": 0, "scale": "M"},
    # P4 transfer: winner at seed 0.
    {"key": "p4_winner_lipo123", "stage": "P4", "dataset": "Lipophilicity", "split": "lipo-repeat:123", "arm": "WINNER", "seed": 0},
    {"key": "p4_winner_lipo124", "stage": "P4", "dataset": "Lipophilicity", "split": "lipo-repeat:124", "arm": "WINNER", "seed": 0},
    {"key": "p4_winner_freesolv", "stage": "P4", "dataset": "FreeSolv", "split": "published", "arm": "WINNER", "seed": 0},
    {"key": "p4_winner_bbbp", "stage": "P4", "dataset": "BBBP", "split": "published", "arm": "WINNER", "seed": 0},
    {"key": "p4_winner_tox21", "stage": "P4", "dataset": "Tox21", "split": "published", "arm": "WINNER", "seed": 0},
    {"key": "p4_winner_hiv", "stage": "P4", "dataset": "ogbg-molhiv", "split": "published", "arm": "WINNER", "seed": 0, "protocol": "hiv_short"},
    {"key": "p4_base_hiv", "stage": "P4", "dataset": "ogbg-molhiv", "split": "published", "arm": "base", "seed": 0, "protocol": "hiv_short"},
)

#: P5 is filled only after P1-P4; these are the maximum planned cells.
P5_TASKS: tuple[tuple[str, str], ...] = (
    ("Lipophilicity", "published"),
    ("ESOL", "published"),
    ("BACE", "published"),
)


def planned_run(key: str) -> dict[str, Any]:
    for item in PLANNED_RUNS:
        if item["key"] == key:
            return dict(item)
    raise KeyError(f"unknown planned run {key!r}")


def run_dir(key: str) -> Path:
    return RUNS / key


def track_rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_sha(rows: Sequence[int]) -> str:
    arr = np.asarray([int(x) for x in rows], dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message: str) -> None:
    print(message, flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - only on a non-git checkout
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


TRAINING_CODE_FILES = (
    "cl_common.py",
    "cl_chem8.py",
    "cl_data.py",
    "cl_model.py",
    "cl_train.py",
)


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRAINING_CODE_FILES:
        path = CODE_DIR / name
        out[track_rel(path)] = sha256_file(path)
    for label, path in (
        ("prep/lcrc_model.py", PREP_CODE / "lcrc_model.py"),
        ("prep/lcrc_data.py", PREP_CODE / "lcrc_data.py"),
        ("prep/feature_spec.py", PREP_CODE / "feature_spec.py"),
        ("v2/mb2_model.py", V2_CODE / "mb2_model.py"),
        ("v2/mb2_data.py", V2_CODE / "mb2_data.py"),
        ("v2/mb2_metrics.py", V2_CODE / "mb2_metrics.py"),
    ):
        out[label] = sha256_file(path)
    return out


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hostname": platform.node(),
        "cwd": str(REPO),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "generated_utc": utcnow(),
    }


def verify_protocol_freeze() -> dict[str, Any]:
    """Refuse a formal run whose training code no longer matches the freeze."""
    if not PROTOCOL_FREEZE.exists():
        raise FileNotFoundError(f"{track_rel(PROTOCOL_FREEZE)} is missing; write it before formal runs")
    freeze = read_json(PROTOCOL_FREEZE)
    recorded = freeze.get("code_hashes", {})
    actual = code_hashes()
    mismatched = {k: (recorded.get(k), v) for k, v in actual.items() if recorded.get(k) != v}
    if mismatched:
        raise RuntimeError(f"training code changed after the protocol freeze: {mismatched}")
    return {
        "protocol_freeze": track_rel(PROTOCOL_FREEZE),
        "protocol_freeze_sha256": sha256_file(PROTOCOL_FREEZE),
        "git_commit_at_freeze": freeze.get("git_commit"),
        "code_hashes_match": True,
    }


def dataset_entry(key: str) -> dict[str, Any]:
    return V2C.dataset_entry(key)


def recipe_for(protocol: str | None = None) -> dict[str, Any]:
    if protocol in (None, "default"):
        return dict(RECIPE)
    if protocol == "hiv_short":
        return dict(HIV_RECIPE)
    raise ValueError(f"unknown protocol {protocol!r}")


__all__ = [
    "ROUND_ID",
    "PROTOCOL_ID",
    "CODE_DIR",
    "ROUND",
    "REPO",
    "V1",
    "V2",
    "PREP",
    "TASK03",
    "ANALYSIS",
    "RUNS",
    "CACHE",
    "TESTS",
    "SOURCE_CHEM8",
    "PROTOCOL_FREEZE",
    "CHECKPOINT_MANIFEST",
    "TEST_LEDGER",
    "GPU_LEDGER",
    "RUN_REGISTRY",
    "CHEM8_MANIFEST",
    "SEED",
    "SHUFFLE_OFFSET",
    "REPRO_TOL",
    "RECIPE",
    "HIV_RECIPE",
    "ANCHORS",
    "TRANSFER",
    "ARMS",
    "ARM_LABEL",
    "SCALE_M_CONFIG",
    "PLANNED_RUNS",
    "P5_TASKS",
    "planned_run",
    "run_dir",
    "track_rel",
    "utcnow",
    "sha256_file",
    "row_sha",
    "write_json",
    "read_json",
    "log",
    "git_commit",
    "git_dirty",
    "code_hashes",
    "verify_protocol_freeze",
    "environment",
    "dataset_entry",
    "recipe_for",
]
