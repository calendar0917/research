"""Identity, paths and pure-helper reuse for round ``rd-pilot-small-v1``.

This track asks one question: on two official OGB molecular tasks, does a
*joint pairwise delta* learner that reads two static molecular inputs and
predicts their property difference beat (D) direct absolute regression,
(N) neighbour-label aggregation, and (R) a direct model with neighbour residual
correction?

Nothing scientific lives here.  The module only pins

* the round / protocol / study ids,
* the frozen data files and their SHA-256 (identical to the SQ round),
* the path of the committed label-free caches that are reused read-only,

and re-exports the *pure* helpers of the SQ track (hashing, JSON/CSV IO, the
resource ledger, RMSE/MAE, the Train-only target scaler, the label-free cache
loader and the ECFP design matrix).  The old runner is never imported and no
old module global is redirected: only ``sq_data`` / ``sq_common`` / ``sq_xgb``
functions are called.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROUND_ID = "rd-pilot-small-v1"
PROTOCOL_ID = ROUND_ID
STUDY_ID = ROUND_ID

CODE_DIR = Path(__file__).resolve().parent
TRACK_ROOT = CODE_DIR.parent
REPO_ROOT = TRACK_ROOT.parents[1]

SQ_TRACK_ROOT = REPO_ROOT / "tracks" / "structural-query-learning"
SQ_CODE_DIR = SQ_TRACK_ROOT / "code"
SQ_DATA_DIR = SQ_TRACK_ROOT / "data"

DATA_DIR = SQ_DATA_DIR                      # committed label-free caches (reuse)
CACHE_DIR = TRACK_ROOT / "cache"            # git-ignored scratch (memmaps, models)
RESULTS_DIR = TRACK_ROOT / "results" / ROUND_ID

if str(SQ_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SQ_CODE_DIR))

import sq_common as SQ  # noqa: E402
import sq_data as SQD  # noqa: E402
import sq_xgb as SQX  # noqa: E402,F401  (re-exported through C.SQX)

if Path(SQD.__file__).resolve() != (SQ_CODE_DIR / "sq_data.py"):  # pragma: no cover
    raise ImportError(f"sq_data resolved to {SQD.__file__}, expected {SQ_CODE_DIR / 'sq_data.py'}")

#: Pure helpers reused verbatim from the SQ round (no side effects, no writes).
sha256_file = SQ.sha256_file
sha256_array = SQ.sha256_array
sha256_json = SQ.sha256_json
write_json = SQ.write_json
read_json = SQ.read_json
write_csv = SQ.write_csv
rmse = SQ.rmse
mae = SQ.mae
train_target_scaler = SQ.train_target_scaler
Ledger = SQ.Ledger
stage_timer = SQ.stage_timer

#: The two official OGB tasks of this round, in pipeline order.
DATASETS: tuple[str, ...] = ("ogbg-molesol", "ogbg-mollipo")
SPLITS: tuple[str, ...] = ("train", "valid", "test")

#: Official split sizes of the two reused caches.
EXPECTED_SPLIT_SIZES: dict[str, dict[str, int]] = dict(SQ.EXPECTED_SPLIT_SIZES)

#: Frozen SHA-256 of the four reused cache files (identical to SQ's freeze).
EXPECTED_CACHE_SHA256: dict[str, str] = {
    "ogbg_molesol_sq-pilot-cache-v1.npz": (
        "3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a"
    ),
    "ogbg_mollipo_sq-pilot-cache-v1.npz": (
        "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
    ),
    "ogbg_molesol_ecfp4_sq-pilot-cache-v1.npz": (
        "653fb0190abc4af43822ec643bacbcbdbfdd7555f5e632574c9f61728d574f2e"
    ),
    "ogbg_mollipo_ecfp4_sq-pilot-cache-v1.npz": (
        "7990ce4c90a0fdb032897076e19221b62b1361fc98f36fea248cf097bb424ae4"
    ),
}

#: Seed semantics of this round (no multi-seed claim).
TRAIN_SEED = 0
PAIR_RNG_SEED = 1701
BOOTSTRAP_SEED = 1703

#: Fixed reference-retrieval grid.  ``CONFIGS`` order is the tie-break order.
KS: tuple[int, ...] = (4, 16, 64)
WEIGHT_MODES: tuple[str, ...] = ("uniform", "softmax_tau_0.1")
SOFTMAX_TAU = 0.1
MAX_REFERENCES = 64
CONFIGS: tuple[dict[str, Any], ...] = tuple(
    {"name": f"K{k}_{mode}", "K": k, "weights": mode} for k in KS for mode in WEIGHT_MODES
)

#: Frozen XGBoost presets / budgets (identical to the SQ recipe for D).
PRESETS: dict[str, dict[str, float]] = {
    "H1": {"max_depth": 3, "eta": 0.05},
    "H2": {"max_depth": 5, "eta": 0.03},
}
PRESET_ORDER: tuple[str, ...] = ("H1", "H2")
ROUNDS_D = 800
ROUNDS_C = 1600

#: Frozen pair-construction constants.
TOP_POOL = 64
N_TOP = 16
N_TAIL = 16
MAX_PAIRS = 32


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def results_dir(dataset: str) -> Path:
    return RESULTS_DIR / dataset


def environment_snapshot() -> dict[str, Any]:
    """Provenance: python, xgboost, numpy, host, CUDA visibility."""
    import importlib
    import os
    import platform

    versions: dict[str, Any] = {}
    for name in ("xgboost", "numpy", "scipy", "torch"):
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - provenance only
            versions[name] = f"MISSING ({type(exc).__name__})"
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cpu_count": os.cpu_count(),
        "env": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "RR_RUN_ID", "SLURM_JOB_ID")
            if os.environ.get(key) is not None
        },
    }
    try:  # provenance only: the commit the code was executed from
        import subprocess

        payload["git_commit"] = (
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
        payload["git_branch"] = (
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:  # noqa: BLE001 - a missing git must not fail a run
        payload["git_commit"] = None
        payload["git_branch"] = None
    try:
        import xgboost

        payload["xgboost"] = {
            "version": xgboost.__version__,
            "use_cuda": str(xgboost.build_info().get("USE_CUDA")),
        }
    except Exception as exc:  # noqa: BLE001
        payload["xgboost"] = {"error": repr(exc)}
    return payload


def peak_memory_mb() -> dict[str, float | None]:
    """Peak RSS of the worker process and the GPU memory high-water mark."""

    def _rss_mb() -> float | None:
        try:
            import resource

            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:  # noqa: BLE001
            return None

    return {"peak_rss_mb": _rss_mb()}


def label_array(payload: Mapping[str, Any], split: str) -> np.ndarray:
    """Raw labels of one split, float64, in the frozen SQ split order."""
    idx = SQD.split_indices(payload, split)
    return np.asarray(payload["labels"], dtype=np.float64)[idx]


def row_id_array(payload: Mapping[str, Any], split: str) -> np.ndarray:
    idx = SQD.split_indices(payload, split)
    return np.asarray(payload["row_ids"], dtype=np.int64)[idx]
