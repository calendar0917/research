"""Shared plumbing for zinc-perf-protocol-v1.

No model or data code lives here; this module only provides provenance,
deterministic I/O, RNG isolation and small numeric helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

PROTOCOL_ID = "zinc-perf-protocol-v1"

CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
PREDICTIONS = EXP_DIR / "predictions"
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"

PROTOCOL_YAML = CONFIGS / "protocol.yaml"
ZSRT_YAML = CONFIGS / "zsrt_screen.yaml"
DATA_AUDIT = ANALYSIS / "data_audit.json"
PROTOCOL_FREEZE = ANALYSIS / "protocol_freeze.json"

ZINC_ROOT = REPO / "data" / "ZINC"

# Frozen P3 inputs (read-only; the old namespace is not modified).
P3_ROOT = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
P3_OBJECTS_TRAIN = P3_ROOT / "objects_train10k.pt"
P3_OBJECTS_VALID = P3_ROOT / "objects_valid.pt"
P3_PREP = P3_ROOT / "prep_train10k.json"
P3_SSL_S0 = P3_ROOT / "pretrain_encoder_s0.pt"
P3_SSL_S1 = P3_ROOT / "pretrain_encoder_s1.pt"
P3_ENCODED_TRAIN = REPO / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

P3_FROZEN_VALID_MAE = {
    0: 0.07910149778419873,
    1: 0.08058639967715135,
}
P3_HISTORICAL_VALID_PRED = {
    0: REPO / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/predictions/valid_R30.npz",
    1: REPO / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/predictions/valid_R31.npz",
}

REF_BUNDLE = REPO / "tracks/structural-sufficiency/zinc_grit_reference_preparation_v1"


def log(msg: str) -> None:
    print(msg, flush=True)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: Path | str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {y.shape}")
    return float(np.abs(pred - y).mean())


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def choose_best(curve: Sequence[Mapping[str, Any]], key: str = "valid_mae") -> dict:
    """Return the entry with the lowest key; ties keep the earlier entry."""
    best = None
    for entry in curve:
        if key not in entry or entry[key] is None:
            continue
        value = float(entry[key])
        if not math.isfinite(value):
            continue
        if best is None or value < float(best[key]):
            best = dict(entry)
    if best is None:
        raise ValueError(f"no finite {key} in curve")
    return best


@contextmanager
def isolated_rng():
    """Fork CPU + current-CUDA RNG state so eval cannot perturb training."""
    devices: list[int] = []
    if torch.cuda.is_available():
        try:
            devices = [torch.cuda.current_device()]
        except Exception:
            devices = []
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                          base_lr: float, min_lr: float) -> float:
    """Per-step linear warm-up then cosine decay (step is 1-based)."""
    step = int(step)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * float(step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return float(base_lr)
    frac = float(step - warmup_steps) / float(total_steps - warmup_steps)
    frac = min(max(frac, 0.0), 1.0)
    return float(min_lr) + 0.5 * (float(base_lr) - float(min_lr)) * (1.0 + math.cos(math.pi * frac))


def environment() -> dict:
    import torch_geometric

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "protocol_id": PROTOCOL_ID,
        "timestamp": utcnow(),
    }


def ensure_grit_runtime() -> None:
    """Put the audited GRIT runtime (pristine upstream + patch) on sys.path."""
    if str(REF_BUNDLE) not in sys.path:
        sys.path.insert(0, str(REF_BUNDLE))
    import reference_common as rc  # noqa: E402

    rc.setup_import_paths()


def ensure_p3_modules() -> None:
    for p in (
        REPO / "tracks/structure-contribution/code",
        REPO / "tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code",
    ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
