"""Shared constants, paths, IO and frozen reuse for the repair round.

Round: ``lcrc-semantic-analysis-repair-bw0-s0-v1``.

This round does **not** redefine the LCRC method, the data pipeline or the
training loop.  It reuses the frozen predecessors read-only:

* ``lcrc_explicit_semantic_learning_8h_s0_v1`` (the 20-run semantic round and
  its analysis/training code; the audit found post-processing/export bugs, not
  training-input bugs),
* ``lcrc_lipo_context_small_s0_v1`` (caches, feature spec, C1 recipe, the B arm),
* ``lcrc_cross_task_preparation_v1`` (``lipo_v1`` spec, graphs, ``collate``),
* ``lcrc_lipo_model_dissection_v1`` (the ``DissectLCRC`` C1 model and state maps).

Everything scientific (seeds, selection, tolerances, bootstrap design) is
frozen in :mod:`rp_protocol` / ``PROTOCOL.md`` before the first analysis.

The only new neural training in this round is ``B_N100_lipo_W0``; every other
number is a re-derivation from stored checkpoints / predictions / caches.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# paths (import the frozen predecessors read-only)
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
TRACK = CODE_DIR.parent
REPO = TRACK.parents[2]
SS = REPO / "tracks/structural-sufficiency"

OLD = SS / "lcrc_explicit_semantic_learning_8h_s0_v1"
OLD_CODE = OLD / "code"
HOME = SS / "lcrc_lipo_context_small_s0_v1"
HOME_CODE = HOME / "code"
PREP = SS / "lcrc_cross_task_preparation_v1"
PREP_CODE = PREP / "code"
DISSECT = SS / "lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT / "code"
EDIT = SS / "lcrc_lipo_edit_supervision_public122_s0_v1"

for _p in (str(PREP_CODE), str(HOME_CODE), str(DISSECT_CODE), str(OLD_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROUND_ID = "lcrc-semantic-analysis-repair-bw0-s0-v1"
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

#: frozen modules this round consumes read-only (hashed into every record)
REUSED_MODULES = (
    "tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/code/se_common.py",
    "tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/code/se_tables.py",
    "tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/code/se_model.py",
    "tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/code/se_data.py",
    "tracks/structural-sufficiency/lcrc_explicit_semantic_learning_8h_s0_v1/code/se_train.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_common.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_data.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_source.py",
    "tracks/structural-sufficiency/lcrc_lipo_context_small_s0_v1/code/lp_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py",
    "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_common.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_data.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/molecular_graph.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/feature_spec.py",
    "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/adapters.py",
)

# ---------------------------------------------------------------------------
# frozen design constants (mirror the predecessors; asserted, never re-derived)
# ---------------------------------------------------------------------------

SPLITS = ("train", "valid", "test")
N_ATOM_TYPES = 16
N_BOND_TYPES = 5
N_PATH_CODES = 400
N_NBR_CODES = 80
RANK = 32
C1_PARAMS = 228_674
GEN_PARAMS = 33_689
FREE_TABLE_PARAMS = 33_680
FS_PARAMS = 228_683

GEN_SEED = 17
PERM_SEED = 20261114
KERNEL_SEED = 20261115
BOOTSTRAP_SEED_OLD = 20261114
BOOTSTRAP_SEED = 20261115
BOOTSTRAP_REPLICATES = 500

REPRO_TOL = 1e-5        # replay / export raw prediction max|Δ|
KERNEL_TOL = 1e-6

#: low-support joint df threshold frozen by the old protocol
LOW_SUPPORT_DF = 5
#: LOW_SUPPORT cell rule is component count, not pair count
LOW_SUPPORT_MIN_COMPONENTS = 20
#: two-sided risk comparison requirements (frozen in the old protocol)
RISK_MIN_SIDE = 40
RISK_MIN_BIN_SIDE = 10
RISK_MIN_BINS = 2

#: old `last` window is the final 20 epochs (NOT "last 5%")
LATE_WINDOW = 20

# ---------------------------------------------------------------------------
# source runs (the 20 old formal runs + the HOME B arm + the ESOL runs)
# ---------------------------------------------------------------------------

OLD_RESULTS = OLD / "analysis/results"
HOME_RESULTS = HOME / "analysis/results"

#: arm name -> where its per-run artifacts live
LIPO_ARMS = {
    "LS_R": OLD_RESULTS / "LS_N100_lipo_R",
    "FS_R": OLD_RESULTS / "FS_N100_lipo_R",
    "LP_R": OLD_RESULTS / "LP_N100_lipo_R",
    "FP_R": OLD_RESULTS / "FP_N100_lipo_R",
    "LS_W0": OLD_RESULTS / "LS_N100_lipo_W0",
    "FS_W0": OLD_RESULTS / "FS_N100_lipo_W0",
    "LS_Rwd": OLD_RESULTS / "LS_N100_lipo_Rwd",
    "FS_Rwd": OLD_RESULTS / "FS_N100_lipo_Rwd",
    "LP_Rwd": OLD_RESULTS / "LP_N100_lipo_Rwd",
    "FP_Rwd": OLD_RESULTS / "FP_N100_lipo_Rwd",
    "B_Rwd": OLD_RESULTS / "B_N100_lipo_Rwd",
    "LS_N25": OLD_RESULTS / "LS_N25_lipo_R",
    "FS_N25": OLD_RESULTS / "FS_N25_lipo_R",
    "LS_N50": OLD_RESULTS / "LS_N50_lipo_R",
    "FS_N50": OLD_RESULTS / "FS_N50_lipo_R",
    # the original C1 base under the R recipe is the HOME round's arm B
    "B_R": HOME_RESULTS / "B",
}
ESOL_ARMS = {
    "B_esol": OLD_RESULTS / "B_N100_esol_R",
    "LS_esol": OLD_RESULTS / "LS_N100_esol_R",
    "FS_esol": OLD_RESULTS / "FS_N100_esol_R",
    "LP_esol": OLD_RESULTS / "LP_N100_esol_R",
    "FP_esol": OLD_RESULTS / "FP_N100_esol_R",
}
#: the new single training of this round
BW0_DIR = RUNS / "arm_B_N100_lipo_W0"

#: checkpoint directory for replay (pulled from res via `rr pull`)
CKPT_DIR = DATA / "checkpoints"

# ---------------------------------------------------------------------------
# hashing / IO
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_rel(path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def row_sha256(rows) -> str:
    arr = np.asarray(list(rows), dtype=np.int64)
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def state_hash(state: Mapping[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def module_paths() -> list[Path]:
    """``sys.path`` entries this round needs (own code + the frozen predecessors)."""
    paths = [CODE_DIR]
    seen = {str(CODE_DIR)}
    for rel in REUSED_MODULES:
        parent = (REPO / rel).parent
        if str(parent) not in seen:
            seen.add(str(parent))
            paths.append(parent)
    return paths


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(CODE_DIR.glob("*.py")):
        out[f"code/{p.name}"] = sha256_file(p)
    for rel in REUSED_MODULES:
        target = REPO / rel
        if target.is_file():
            out[rel] = sha256_file(target)
    return out


__all__ = [name for name in dir() if not name.startswith("_")]
