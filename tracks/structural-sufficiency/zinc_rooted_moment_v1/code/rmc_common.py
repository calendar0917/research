"""Shared constants / provenance helpers for ``zinc_rooted_moment_v1`` (RMC).

Protocol ``ssrc-zinc-rooted-moment-v1``.

Reuse policy
------------
Metrics, fixed valid group tables and the paired molecule bootstrap are the
*authoritative* implementations from ``zinc_native_relation_v1/nr_common.py``
(the exact code used by the same-round reference P).  This module only adds the
round's own paths, config and cache handling.  Nothing here reads ``y`` except
the trainer and the evaluators.

Boundary notes
--------------
* The rooted measure is built from the frozen P3 ``bond_index`` (each physical
  bond once; the PyG directed duplicate is not counted twice) plus an explicit
  walk-mass normalisation.  It is a *walk* measure with a stay term, not a
  shortest-path / simple-path / chemical-resonance object.
* ``F`` is computed from cached moments only: no other atom's trainable hidden
  state is read, no structural operator is updated, no neighbour-state recursion.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
REPO = TRACK.parents[2]
V1_CODE = REPO / "tracks/structural-sufficiency/zinc_arch_discovery_v1/code"
NR_CODE = REPO / "tracks/structural-sufficiency/zinc_native_relation_v1/code"
for _p in (HERE, V1_CODE, NR_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as AC  # noqa: E402  (repo-level paths / io / git)
import nr_common as NC  # noqa: E402  (authoritative metrics + fixed groups)

CONFIG_PATH = TRACK / "configs/rmc_v1.json"
CACHE_DIR = TRACK / "cache"
ANALYSIS_DIR = TRACK / "analysis"
RUNS_DIR = TRACK / "runs"
PROTOCOL_ID = "ssrc-zinc-rooted-moment-v1"
PROTOCOL_PATH = TRACK / "PROTOCOL.md"

N_TRAIN = 10000
N_VALID = 1000
ARMS = ("P", "M", "G", "U")
SEEDS = (0, 1)

# ---- fixed mathematical constants (frozen before any GPU run) -------------
SCALES = (0, 1, 2, 3, 4, 6, 8, 12)
N_SCALE = len(SCALES)
VIEWS = (0, 1)
N_VIEW = len(VIEWS)
PROJ_DIM = 12
R_PROJ = 12
R_SEEDS = (7301, 7302)
EPS = 0.05
ENDPOINT_DIM = 204  # 108 raw atom features + 96 frozen SSL atom encoding
N_PAIR = PROJ_DIM * (PROJ_DIM + 1) // 2  # 78
N_TRIPLE = PROJ_DIM * (PROJ_DIM + 1) * (PROJ_DIM + 2) // 6  # 364
C3_FULL_TERMS = PROJ_DIM * PROJ_DIM * (PROJ_DIM + 1) // 2  # 936
N_HEAD = 4
READOUT_IN = 96 + N_HEAD * N_SCALE * PROJ_DIM + N_HEAD * N_SCALE + N_SCALE + 33 + 3  # 556

# metric helpers are re-exported so downstream code has one import site
mae = NC.mae
metric_block = NC.metric_block
paired_bootstrap = NC.paired_bootstrap
window_metrics = NC.window_metrics
load_valid_groups = NC.load_valid_groups
group_table = NC.group_table
delta_stats = NC.delta_stats
tail_calibration = NC.tail_calibration
topk_removed_gain = NC.topk_removed_gain


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, payload: Mapping[str, Any] | Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_once_json(path: Path | str, payload: Mapping[str, Any] | Any) -> None:
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen evidence {path}")
    write_json(path, payload)


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def sha256_file(path: Path | str) -> str:
    return NC.sha256_file(path)


def sha256_array(arr: np.ndarray, dtype: str = "float32") -> str:
    return NC.sha256_array(arr, dtype=dtype)


def sha256_state_dict(state: Mapping[str, Any]) -> str:
    return NC.sha256_state_dict(state)


def sha256_tensor(t: "Any") -> str:
    import torch

    h = hashlib.sha256()
    arr = t.detach().cpu().contiguous()
    h.update(str(tuple(arr.shape)).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.numpy().tobytes())
    return h.hexdigest()


def git_commit() -> str:
    return AC.git_commit()


def environment() -> dict[str, Any]:
    return AC.environment()


# ---- cache paths ----------------------------------------------------------
def moments_path(split: str) -> Path:
    return CACHE_DIR / f"rmc_{split}_moments.pt"


def static_path(split: str) -> Path:
    return CACHE_DIR / f"rmc_{split}_static.pt"


def norm_path() -> Path:
    return CACHE_DIR / "rmc_norm.pt"


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)
