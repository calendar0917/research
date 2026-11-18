"""Identity, frozen constants, paths, hashing and IO for ``sq-lipo-pair-rule-v1``.

This round is a *small* controlled comparison on one official OGB task:

    BASE    four whole-graph counts + the frozen B query bank + typed local
            two-bond paths + type-pair totals + distance marginals
    NULL    BASE + the same relation slots represented by their exchangeable
            (hypergeometric) marginal expectation
    REAL    BASE + the same relation slots represented by actual anchor-pair
            counts
    MLP     a two-hidden-layer MLP on exactly the REAL input
    C1      the repository's LCRC-C1 / NO_BOND_MEAN, retrained here

Only engineering constants live here; the scientific rules are frozen in
``PROTOCOL.md`` / ``protocols/sq-lipo-pair-rule-v1.yaml`` and hashed at prepare.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

SQL_CODE = TRACK_DIR / "code"
GR_CODE = REPO_ROOT / "tracks/structural-sufficiency/gnn_route_clarification_8h_v1/code"
for _path in (str(SQL_CODE), str(GR_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sq_common as SC  # noqa: E402  (path setup must come first)

ROUND_ID = "sq-lipo-pair-rule-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID
DESIGN_BASIS_COMMIT = "0d2ef5f4f72fbda3efbdbf590f2d0b02ec7da007"

DATASET = "ogbg-mollipo"
SPLITS: tuple[str, ...] = ("train", "valid", "test")
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420, "test": 420}
EXPECTED_TOTAL_ROWS = 4200
EXPECTED_CACHE_SHA256 = {
    "graph": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "lipo_raw": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}
CACHE_NAME = "ogbg_mollipo_sq-pilot-cache-v1.npz"
LIPO_RAW_NAME = "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz"

# Frozen training / evaluation constants.
TRAIN_SEED = 0
BOOTSTRAP_SEED = 1703
BOOTSTRAP_RESAMPLES = 2000
EPOCHS = 180
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
DROPOUT = 0.10
GRAD_CLIP = 5.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-5
C1_RECIPE = "Rf"
C1_EXPECTED_PARAMS = 228674

# Frozen rule-library constants.
BANK_CAP = 512
BANK_MIN_MOLECULES = 8
S2_MIN_MOLECULES = 16
S2_CAP = 256
RELATION_MIN_MOLECULES = 16
RELATION_CAP = 512
MAX_LOCAL_HOPS = 2
LAMBDA_FRACTION = 0.02
SCALE_FLOOR = 0.1
ZERO_WEIGHT_EPS = 1e-10
RAW_DIFF_EPS = 1e-8
NEG_CLAMP_EPS = 1e-10
RULE_REPLAY_TOL = 1e-6
MIN_VAL_COVERAGE = 84
SIGNAL_GAIN = 0.02
NEAR_C1_GAP = 0.04
OLD_C1_VAL = 0.7052
WEAK_REFERENCE_MARGIN = 0.04
INTERPRET_SCALE = 0.02

#: Relation extension distance bins (finite only; unreachable is never a slot).
EXTENSION_BINS: tuple[str, ...] = ("3", "4", "5", "6", "7", "8", "9+")
#: Whole-graph distance marginals, including the unreachable bucket.
DISTANCE_BINS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9+", "unreachable")
#: The four named whole-graph counts, in the frozen order.
NAMED_GRAPH_COUNTS: tuple[str, ...] = ("n_atoms", "n_bonds", "n_components", "cycle_rank")
READOUTS: tuple[str, ...] = ("FULL", "-32")
ADDITIVE_ARMS: tuple[str, ...] = ("BASE", "NULL", "REAL")
STATES: tuple[str, ...] = ("best", "epoch180")

# Role codes for the atom rule type.
ROLE_NONE = 0
ROLE_CARBOXYL_C = 1
ROLE_CARBONYL_O = 2
ROLE_NAMES = {
    ROLE_NONE: "none",
    ROLE_CARBOXYL_C: "C_double_bond_O",
    ROLE_CARBONYL_O: "O_double_bond_C",
}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def _resolve_override(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _default_runs_dir() -> Path:
    return ROUND_DIR / "runs"


def _default_analysis_dir() -> Path:
    return ROUND_DIR / "analysis"


RUNS_DIR = _resolve_override(os.environ.get("SQ_PAIR_RUNS_DIR")) or _default_runs_dir()
ANALYSIS_DIR = _resolve_override(os.environ.get("SQ_PAIR_ANALYSIS_DIR")) or _default_analysis_dir()
REPORT_PATH = ROUND_DIR / "REPORT.md"
CONFIG_PATH = ROUND_DIR / "configs" / f"{ROUND_ID}.yaml"
PROTOCOL_PATH = TRACK_DIR / "protocols" / f"{ROUND_ID}.yaml"
STUDY_PATH = TRACK_DIR / "studies" / f"{ROUND_ID}.yaml"


def stage_dir(stage: str) -> Path:
    return RUNS_DIR / stage


def ensure_dir(path: Path | str) -> Path:
    return SC.ensure_dir(path)


def sha256_file(path: Path | str) -> str:
    return SC.sha256_file(path)


def sha256_array(array: np.ndarray) -> str:
    return SC.sha256_array(array)


def sha256_json(payload: Any) -> str:
    return SC.sha256_json(payload)


def sha256_text(text: str) -> str:
    return SC.sha256_text(text)


def read_json(path: Path | str) -> Any:
    return SC.read_json(path)


def write_json(path: Path | str, payload: Any) -> Path:
    return SC.write_json(path, payload)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    return SC.write_csv(path, rows, columns=columns)


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def save_npz(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in payload.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    return SC.git_commit()


def git_state() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=20)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def protocol_hash() -> str:
    from ksvd_research.runtime.control import load_protocol, protocol_hash as _hash

    try:
        return _hash(load_protocol(PROTOCOL_ID))
    except Exception:  # noqa: BLE001 - aggregate may run without the control plane on path
        return "unavailable"


def code_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(CODE_DIR.glob("sq_pair_*.py")):
        out[f"code/{path.name}"] = sha256_file(path)
    for path in (CONFIG_PATH, PROTOCOL_PATH, STUDY_PATH, ROUND_DIR / "run_remote.sh"):
        if path.is_file():
            out[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    for name in ("sq_data.py", "sq_bank.py", "sq_ast.py", "sq_common.py"):
        path = SQL_CODE / name
        if path.is_file():
            out[f"reused/structural-query-learning/code/{name}"] = sha256_file(path)
    for name in (
        "gr_data.py",
        "gr_models.py",
        "gr_train.py",
        "gr_common.py",
        "gr_select.py",
    ):
        path = GR_CODE / name
        if path.is_file():
            out[f"reused/gnn_route_clarification_8h_v1/code/{name}"] = sha256_file(path)
    return out


def environment() -> dict[str, Any]:
    payload = SC.environment_snapshot()
    payload["git"] = git_state()
    payload["round_id"] = ROUND_ID
    payload["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    payload["threads"] = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    return payload


def run_provenance() -> dict[str, Any]:
    return {
        "rr_run_id": os.environ.get("RR_RUN_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "git_commit": git_commit(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": SC.time.strftime("%Y-%m-%dT%H:%M:%SZ", SC.time.gmtime()),
        "started_epoch": float(SC.time.time()),
    }


def finish_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    payload["finished_at"] = SC.time.strftime("%Y-%m-%dT%H:%M:%SZ", SC.time.gmtime())
    payload["finished_epoch"] = float(SC.time.time())
    payload["wall_seconds"] = payload["finished_epoch"] - float(payload.get("started_epoch", SC.time.time()))
    return payload


# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return SC.rmse(pred, target)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return SC.mae(pred, target)


def u_basis(counts: np.ndarray) -> np.ndarray:
    """``[min(n,1), max(n-1,0)]`` for integer/float counts, shape (..., 2)."""
    arr = np.asarray(counts, dtype=np.float64)
    return np.stack([np.minimum(arr, 1.0), np.maximum(arr - 1.0, 0.0)], axis=-1)


def column_scale(column: np.ndarray) -> float:
    arr = np.asarray(column, dtype=np.float64)
    if arr.size == 0:
        return SCALE_FLOOR
    value = float(np.sqrt(np.mean(arr * arr)))
    return max(value, SCALE_FLOOR)


def scales_for(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("scale computation expects a 2-D matrix")
    if arr.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([column_scale(arr[:, j]) for j in range(arr.shape[1])], dtype=np.float64)


def group_slices(n_groups: int, width: int = 2) -> list[slice]:
    return [slice(k * width, (k + 1) * width) for k in range(n_groups)]


def json_safe(value: Any) -> Any:
    return SC.jsonable(value)


def ledger(stage: str) -> SC.Ledger:
    return SC.Ledger(stage_dir(stage) / "ledger.jsonl")
