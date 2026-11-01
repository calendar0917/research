"""Shared constants, paths and provenance helpers -- Sprint 03.

Round id: ``lcrc-taskaware-representation-pilot-v1``.

This round is deliberately split into *one* audit (P0), *one* benchmark-robustness
check (P1/P2) and *two* task-objective probes (P3).  No representation is
modified and no new FeatureSpec is trained here.

What is reused read-only
------------------------
* ``lcrc_molnet_benchmark_pilot_v2/code``: the frozen ``lipo_v1`` data path
  (``mb2_raw_artifact`` -> ``mb2_rdkit`` -> ``adapters.lipo_row_to_graph`` ->
  ``lcrc_data.build_molecule``), the frozen LCRC-A model (``mb2_model``), the
  frozen R3 recipe, the seven-dataset registry, the metrics and the frozen
  ECFP4+RDKit+XGBoost baseline configuration.
* ``lcrc_molnet_benchmark_pilot_v1``: the vendored ChemBench bytes (the
  ``Lipop.random123/124`` split files live there, unmodified).

Nothing in those directories is edited.  Every quantity that this round chooses
(two extra Lipophilicity split repeats, three BBBP inner-CV folds, two loss
variants) is frozen in ``analysis/protocol_freeze.json`` *before* the first GPU
job of the round.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-taskaware-representation-pilot-v1"
PROTOCOL_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
REPO = ROUND.parents[2]

STRUCT = REPO / "tracks/structural-sufficiency"
V1 = STRUCT / "lcrc_molnet_benchmark_pilot_v1"
V2 = STRUCT / "lcrc_molnet_benchmark_pilot_v2"
PREP = STRUCT / "lcrc_cross_task_preparation_v1"
PHYSCHEM = STRUCT / "lcrc_esol_physchem_s2_v1"
V1_CODE = V1 / "code"
V2_CODE = V2 / "code"
PREP_CODE = PREP / "code"

for _p in (str(CODE_DIR), str(V2_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mb2_common as V2C  # noqa: E402  (also puts PREP_CODE and V1_CODE on sys.path)

ANALYSIS = ROUND / "analysis"
RUNS = ANALYSIS / "runs"
CACHE = ROUND / "cache"
TESTS = ROUND / "tests"
SOURCE_RAW = V2 / "analysis/source/raw/lcrc_raw_v2"

PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"
SPLIT_FREEZE = ANALYSIS / "split_freeze.json"
CHECKPOINT_MANIFEST = ANALYSIS / "checkpoint_manifest.json"
TEST_LEDGER = ANALYSIS / "test_access_ledger.json"
RUN_REGISTRY = ANALYSIS / "rr_run_registry.json"
GPU_LEDGER = ANALYSIS / "gpu_ledger.json"

DATASETS: tuple[str, ...] = tuple(V2C.DATASET_ORDER)

# ---------------------------------------------------------------------------
# P1: the two extra published Lipophilicity repeats
# ---------------------------------------------------------------------------

#: split-file suffix -> our repeat label.  ``122`` is the Sprint-01/02 split and
#: is *not* re-run; these two are the published repeats this round adds.
LIPO_EXTRA_REPEATS: dict[int, int] = {123: 1, 124: 2}
LIPO_BASELINE_REPEAT_SUFFIX = 122

# ---------------------------------------------------------------------------
# P2: BBBP inner, Train-only scaffold CV
# ---------------------------------------------------------------------------

BBBP_CV_SEED = 0
BBBP_CV_FOLDS = 3
BBBP_CV_SALT = "lcrc-taskaware-bbbp-inner-cv-v1"
#: split-file suffix of the published BBBP split whose *Train* rows are used.
BBBP_CV_SOURCE_SUFFIX = 122

# ---------------------------------------------------------------------------
# P3: frozen loss candidates
# ---------------------------------------------------------------------------

LOSS_FAMILIES = ("masked_cell", "equal_task", "sqrt_pos_weight")

TOX21_EQUAL_TASK = {
    "name": "tox21_equal_task_loss",
    "dataset": "Tox21",
    "definition": (
        "per-task mean BCE over that task's observed cells in the batch, then an "
        "unweighted mean over the tasks that have >=1 observed cell in this batch"
    ),
    "empty_task_policy": "excluded from the denominator; never treated as zero loss",
    "class_weights": None,
    "nan_fill": None,
}

HIV_SQRT_POS_WEIGHT = {
    "name": "hiv_sqrt_pos_weight",
    "dataset": "ogbg-molhiv",
    "definition": "BCEWithLogitsLoss(pos_weight=min(5.0, sqrt(n_negative/n_positive)))",
    "statistic_source": "official OGB Train split labels only",
    "cap": 5.0,
    "validation_or_test_used": False,
}

# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

#: Hard cap for this round (A100 GPU-hours).
GPU_HARD_CAP_HOURS = 1.6
#: Planned formal runs.  The order is also the completion priority order.
PLANNED_RUNS: tuple[dict[str, Any], ...] = (
    {"key": "lipo_r123", "workstream": "P1", "dataset": "Lipophilicity", "repeat": 1, "loss": "masked_cell"},
    {"key": "lipo_r124", "workstream": "P1", "dataset": "Lipophilicity", "repeat": 2, "loss": "masked_cell"},
    {"key": "bbbp_cv_f0", "workstream": "P2", "dataset": "BBBP", "fold": 0, "loss": "masked_cell"},
    {"key": "bbbp_cv_f1", "workstream": "P2", "dataset": "BBBP", "fold": 1, "loss": "masked_cell"},
    {"key": "bbbp_cv_f2", "workstream": "P2", "dataset": "BBBP", "fold": 2, "loss": "masked_cell"},
    {"key": "tox21_equal_task", "workstream": "P3", "dataset": "Tox21", "loss": "equal_task"},
    {"key": "hiv_sqrt_pos", "workstream": "P3", "dataset": "ogbg-molhiv", "loss": "sqrt_pos_weight"},
)
MAX_FORMAL_RUNS = 7

SEED = 0
REPRO_TOL = float(V2C.REPRO_TOL)


# ---------------------------------------------------------------------------
# small provenance helpers
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, obj: Any):
    return V2C.write_json(path, obj)


def read_json(path) -> Any:
    return V2C.read_json(path)


def sha256_file(path) -> str:
    return V2C.sha256_file(path)


def row_sha(rows: Sequence[int]) -> str:
    return V2C.row_sha256([int(x) for x in rows])


def track_rel(path) -> str:
    return V2C.track_rel(path)


def log(msg: str) -> None:
    V2C.log(msg)


def ensure_dir(path) -> Path:
    return V2C.ensure_dir(path)


def dataset_entry(key: str) -> dict[str, Any]:
    return V2C.dataset_entry(key)


def recipe() -> dict[str, Any]:
    """The frozen R3 recipe, byte-identical to Sprint 02."""
    return dict(V2C.RECIPE)


def git_commit() -> str:
    return V2C.git_commit()


def git_dirty() -> bool:
    return V2C.git_dirty()


def code_hashes() -> dict[str, str]:
    """This round's code + every reused upstream module, hashed for the freeze."""
    out = {f"ta/{p.name}": sha256_file(p) for p in sorted(CODE_DIR.glob("ta_*.py"))}
    out.update({f"v2/{p.name}": sha256_file(p) for p in sorted(V2_CODE.glob("mb2_*.py"))})
    out.update(V2C.reused_module_hashes())
    out["prep/feature_spec_lipo.json"] = sha256_file(PREP / "analysis/feature_spec_lipo.json")
    for suffix in ("122", "123", "124"):
        p = V1 / f"analysis/source/chembench/Lipop/Lipop.random{suffix}.ind.pkl"
        out[f"chembench/Lipop.random{suffix}.ind.pkl"] = sha256_file(p)
    out["chembench/BBBP.scaffold122.ind.pkl"] = sha256_file(
        V2 / "analysis/source/chembench/BBBP/BBBP.scaffold122.ind.pkl"
    )
    return out


def env_info() -> dict[str, Any]:
    import platform as _platform

    import torch

    out = {
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "hostname": _platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
    }
    try:
        out.update(V2C.public_env_info())
    except Exception as exc:  # pragma: no cover - xgboost/rdkit absent
        out["public_env_error"] = f"{type(exc).__name__}: {exc}"
    return out


def content_fingerprint(cache: Mapping[str, Any]) -> str:
    """Order-stable digest of one structure cache (structural tensors only)."""
    h = hashlib.sha256()
    for key in (
        "atom_ptr",
        "bond_ptr",
        "nbr_ptr",
        "path_ptr",
        "atom_feat",
        "atom_type",
        "atom_scalar",
        "bond_feat",
        "bond_type",
        "bond_index",
        "nbr_code",
        "path_cat",
        "path_end63",
        "path_end",
        "path_closure",
        "global",
    ):
        arr = np.ascontiguousarray(np.asarray(cache[key]))
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    h.update(b"|rows|")
    h.update(np.ascontiguousarray(np.asarray(cache["rows"], dtype=np.int64)).tobytes())
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return V2C.sha256_bytes(payload.encode("utf-8"))


def run_dir(key: str) -> Path:
    return RUNS / key


def committed_blob_sha256(rel_path: str | Path) -> str:
    """sha256 of ``HEAD:<path>`` as committed, or '' when the path is not in HEAD."""
    rel = str(rel_path).replace(str(REPO) + "/", "")
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True
    )
    if proc.returncode != 0:
        return ""
    return hashlib.sha256(proc.stdout).hexdigest()


def verify_protocol_freeze() -> dict[str, Any]:
    """A formal run may only start from the commit that contains the frozen protocol."""
    if not PROTOCOL_FREEZE.exists():
        raise SystemExit(f"{track_rel(PROTOCOL_FREEZE)} is missing; freeze the protocol first")
    local = sha256_file(PROTOCOL_FREEZE)
    committed = committed_blob_sha256(track_rel(PROTOCOL_FREEZE))
    if not committed:
        raise SystemExit("protocol_freeze.json is not committed; commit it before launching formal runs")
    if local != committed:
        raise SystemExit("protocol_freeze.json differs from the committed version")
    if git_dirty():
        raise SystemExit("tracked files are dirty; formal runs start from a clean commit")
    if CHECKPOINT_MANIFEST.exists():
        raise SystemExit("checkpoint_manifest.json exists: the round is already frozen and closed")
    return {
        "protocol_freeze": track_rel(PROTOCOL_FREEZE),
        "protocol_freeze_sha256": local,
        "protocol_freeze_committed_at": git_commit(),
        "worktree_clean": True,
    }


def hours(seconds: float) -> float:
    return float(seconds) / 3600.0


def git_show_remote_head(host: str = "res") -> str:  # pragma: no cover - ssh
    try:
        return subprocess.check_output(
            ["ssh", host, "git -C /home/hxy/cy/research rev-parse HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


__all__ = [name for name in dir() if not name.startswith("_")]
