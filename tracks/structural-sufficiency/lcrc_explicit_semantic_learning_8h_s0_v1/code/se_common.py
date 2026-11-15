"""Frozen constants, codec, arm registry and reuse hashes for

``lcrc-explicit-semantic-learning-8h-s0-v1``.

The method body is still LCRC: the same explicit local objects, the same
atom/bond/path statistics, the same LocalMLP and the same readout.  The *only*
method change is that the six free embedding tables of the C1 local encoder are
generated from shared atom/bond composition components via small MLPs.

Nothing scientific is fitted here.  This module reuses the frozen
``lcrc_lipo_context_small_s0_v1`` data/normalisation/recipe machinery read-only
and adds only the semantic-table definition plus the paired control arms.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# reuse the frozen small-round machinery (read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]
SS = REPO / "tracks/structural-sufficiency"

HOME = SS / "lcrc_lipo_context_small_s0_v1"
HOME_CODE = HOME / "code"
DISSECT = SS / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
PREP = SS / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
BOTTLE = SS / "lcrc_bottleneck_diagnosis_public122_s0_v1"
BOTTLE_CODE = BOTTLE / "code"
PCBA = SS / "lcrc_pcba_backbone_scaling_v1"
PCBA_CODE = PCBA / "code"
DYN = SS / "lcrc_pcba_dynamic_v1"
DYN_CODE = DYN / "code"
EDIT = SS / "lcrc_lipo_edit_supervision_public122_s0_v1"

#: extra import paths (lp_common itself inserts its own reuse paths)
for _p in (str(HOME_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROUND_ID = "lcrc-explicit-semantic-learning-8h-s0-v1"
PROTOCOL_ID = ROUND_ID
STUDY = ROUND_ID

CONFIGS = TRACK / "configs"
PROTOCOLS = TRACK / "protocols"
STUDIES = TRACK / "studies"
ANALYSIS = TRACK / "analysis"
DATA = TRACK / "data"
CACHE = TRACK / "cache"
RUNS = TRACK / "runs"
TESTS = TRACK / "tests"

#: Frozen upstream modules consumed read-only (hashed into every run record).
REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_common.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_source.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_model.py",
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_common.py",
    "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code/bs_train.py",
    "tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_train.py",
)

# ---------------------------------------------------------------------------
# codec (frozen by lipo_v1 spec: 16 atom types, 5 bond types, 400/80 codes)
# ---------------------------------------------------------------------------

N_ATOM_TYPES = 16
N_BOND_TYPES = 5
N_PATH_CODES = N_ATOM_TYPES * N_BOND_TYPES * N_BOND_TYPES   # 400
N_NBR_CODES = N_ATOM_TYPES * N_BOND_TYPES                   # 80
RANK = 32
ATOM_EMB_DIM = 16
BOND_EMB_DIM = 8
PATH_MLP_HIDDEN = 197
BINARY_MLP_HIDDEN = 40
GATE_MLP_HIDDEN = 23
GENERATOR_SEED = 17
PERM_SEED = 20261114
STD_FLOOR = 1e-4

#: The six free tables that F replaces, with (n_codes, dim, kind).
TABLES: dict[str, tuple[int, int, str]] = {
    "path_emb": (N_PATH_CODES, RANK, "path"),
    "path_u": (N_PATH_CODES, RANK, "path"),
    "nbr_emb": (N_NBR_CODES, RANK, "binary"),
    "end_emb": (N_NBR_CODES, RANK, "binary"),
    "path_v": (N_NBR_CODES, RANK, "binary"),
    "path_gate": (N_PATH_CODES, 1, "path_gate"),
}
#: The five vector banks that get affine standardisation (gate excluded).
VECTOR_TABLES = ("path_emb", "path_u", "nbr_emb", "end_emb", "path_v")
FREE_TABLE_PARAMS = sum(n * d for n, d, _ in TABLES.values())   # 33,680
#: shared composition component tables
SHARED_PARAMS = N_ATOM_TYPES * ATOM_EMB_DIM + N_BOND_TYPES * BOND_EMB_DIM   # 296
PATH_MLP_PARAMS = (RANK * PATH_MLP_HIDDEN + PATH_MLP_HIDDEN) + (PATH_MLP_HIDDEN * RANK + RANK)
BINARY_MLP_PARAMS = ((ATOM_EMB_DIM + BOND_EMB_DIM) * BINARY_MLP_HIDDEN + BINARY_MLP_HIDDEN) + (
    BINARY_MLP_HIDDEN * RANK + RANK
)
GATE_MLP_PARAMS = (RANK * GATE_MLP_HIDDEN + GATE_MLP_HIDDEN) + (GATE_MLP_HIDDEN * 1 + 1)
GENERATOR_PARAMS = SHARED_PARAMS + 2 * PATH_MLP_PARAMS + 3 * BINARY_MLP_PARAMS + GATE_MLP_PARAMS

C1_PARAMS = 228_674
FS_PARAMS = C1_PARAMS - FREE_TABLE_PARAMS + GENERATOR_PARAMS

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
# family: "sem"/"perm" (generator), "ls"/"lp" (free tables, matched init)
# recipe: "R" (wd 1e-3), "W0" (wd 0), "Rwd" (wd 1e-2)
# dataset: "lipo", "esol"
# label budget: "N100", "N25", "N50"

SEM_FAMILIES = ("LS", "FS", "LP", "FP")
ARM_TABLE = {
    "LS": {"family": "ls", "perm": False},
    "FS": {"family": "fs", "perm": False},
    "LP": {"family": "lp", "perm": True},
    "FP": {"family": "fp", "perm": True},
}
RECIPES = {
    "R": {"weight_decay": 1.0e-3},
    "W0": {"weight_decay": 0.0},
    "Rwd": {"weight_decay": 1.0e-2},
}


def arm_name(family: str, recipe: str = "R", dataset: str = "lipo", budget: str = "N100") -> str:
    return f"{family}_{budget}_{dataset}_{recipe}"


def parse_arm(name: str) -> dict[str, Any]:
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"bad arm name {name!r}")
    family, budget, dataset, recipe = parts
    if family not in SEM_FAMILIES and family not in ("B",):
        raise ValueError(f"unknown family {family!r}")
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}")
    return {"family": family, "budget": budget, "dataset": dataset, "recipe": recipe, "name": name}


# ---------------------------------------------------------------------------
# time / budget bookkeeping
# ---------------------------------------------------------------------------

T0_ENV = "SE_T0_EPOCH"
WALL_TARGET_HOURS = 7.5
WALL_HARD_HOURS = 8.0
GPU_SOFT_HOURS = 14.0
GPU_HARD_HOURS = 16.0


def utcnow() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def addr2line() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for rel in REUSED_MODULES:
        target = REPO / rel
        if target.is_file():
            out[rel] = sha256_file(target)
    return out


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def track_rel(path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def write_json(path, obj) -> Path:
    import json

    payload = json.dumps(
        obj, indent=2, sort_keys=True,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else (float(o) if isinstance(o, (int, float)) else str(o)),
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload + "\n", encoding="utf-8")
    return p


def read_json(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [name for name in dir() if not name.startswith("_")]
