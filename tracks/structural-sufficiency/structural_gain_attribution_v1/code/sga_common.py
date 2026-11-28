"""Shared paths, hashing, IO and provenance helpers for the SGA-v1 round.

Round: ``tracks/structural-sufficiency/structural_gain_attribution_v1``
Protocol id: ``structural-gain-attribution-v1``

The round is a *progressive* study on top of
``molhiv_multigranular_structure_sharing_v0`` (``MSDS-v0``).  Frozen historical
artefacts (enumeration cache, vocabulary, design matrices, historical XGBoost
features and predictions) are read **read-only** from MSDS-v0; nothing in MSDS-v0
is rewritten here.

Only label-free structure caches live under ``data/`` (git-ignored by the
repo-wide ``**/data/**`` rule); JSON/CSV/NPZ deliverables that must travel to the
remote execution host live under ``bundle/`` and are committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROUND_ID = "structural-gain-attribution-v1"
SEED = 0

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
ANALYSIS = ROUND / "analysis"
CACHE = ROUND / "data"  # label-free local caches; git-ignored
BUNDLE = ROUND / "bundle"  # committed label-free inputs for the remote host
FIGURES = ANALYSIS / "figures"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
PREDICTIONS = ANALYSIS / "predictions"

REPO = ROUND.parents[2]
DATA_DIR = REPO / "data"
OGB_MOLHIV = DATA_DIR / "ogb" / "ogbg_molhiv"
ZINC_DIR = DATA_DIR / "ZINC"

#: predecessor round (read-only)
MSDS = REPO / "tracks" / "structural-sufficiency" / "molhiv_multigranular_structure_sharing_v0"
MSDS_CODE = MSDS / "code"
MSDS_DATA = MSDS / "data"
MSDS_ANALYSIS = MSDS / "analysis"

#: historical XGBoost baseline (read-only)
HIST = REPO / "tracks" / "structural-sufficiency" / "lcrc_molnet_benchmark_pilot_v2"
HIST_MANIFEST = HIST / "analysis" / "baseline_manifest.json"
HIST_FEATURES = HIST / "cache" / "baseline_features" / "ogbg-molhiv.npz"
HIST_PRED_DIR = HIST / "analysis" / "baseline"

#: frozen remote-side inputs shipped to the execution host
INPUT_BUNDLE = BUNDLE / "sav1_inputs.npz"
INPUT_BUNDLE_META = BUNDLE / "sav1_inputs_meta.json"


def add_prev_code_path() -> None:
    """Make the MSDS-v0 modules importable (they are reused, not copied)."""
    if str(MSDS_CODE) not in sys.path:
        sys.path.insert(0, str(MSDS_CODE))
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))


def ensure_dirs() -> None:
    for p in (ANALYSIS, CACHE, FIGURES, CONFIGS, RUNS, PREDICTIONS, BUNDLE):
        p.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def write_json(path: Path | str, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    path.write_text(text + "\n")
    return sha256_file(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def git_state() -> dict[str, Any]:
    import subprocess

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(status),
        "status_short": status.splitlines()[:20],
    }


def environment() -> dict[str, Any]:
    import platform

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for name in ("scipy", "sklearn", "pandas", "torch", "xgboost", "rdkit"):
        try:
            mod = __import__(name)
        except Exception:  # pragma: no cover - optional stack
            env[name] = None
            continue
        env[name] = getattr(mod, "__version__", "unknown")
    env["cpu_count"] = os.cpu_count()
    if env.get("torch"):
        try:
            import torch

            env["torch_cuda"] = torch.version.cuda
            env["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            pass
    return env


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = time.perf_counter()

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        print(f"[{self.label}] start", flush=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self.t0
        print(f"[{self.label}] done in {self.seconds:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# split / index identity
# ---------------------------------------------------------------------------


def molhiv_splits() -> dict[str, np.ndarray]:
    """Official ogbg-molhiv scaffold split row ids (read-only, from MSDS-v0)."""
    import msds_data as D

    molecules = D.load_molecules()
    return {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}


def split_fingerprint(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "sizes": {k: int(v.size) for k, v in splits.items()},
        "sha256": {k: sha256_array(v) for k, v in splits.items()},
        "union_size": int(np.unique(np.concatenate(list(splits.values()))).size),
        "min": {k: int(v.min()) for k, v in splits.items()},
        "max": {k: int(v.max()) for k, v in splits.items()},
    }


# ---------------------------------------------------------------------------
# frozen XGBoost recipe (Phase A of the predecessor round / historical baseline)
# ---------------------------------------------------------------------------

XGB_CLASSIFICATION_RECIPE: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "random_state": 0,
    "device": "cpu",
}

#: Phase C only changes the objective; nothing else may change.
XGB_REGRESSION_RECIPE: dict[str, Any] = {
    **XGB_CLASSIFICATION_RECIPE,
}
