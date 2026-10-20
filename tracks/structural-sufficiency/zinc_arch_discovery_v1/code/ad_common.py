"""Shared plumbing for the ZINC architecture-discovery round.

This module deliberately contains no model or data logic.  It only provides
paths, deterministic I/O, RNG isolation, checkpoint helpers and provenance.
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROUND_ID = "zinc-arch-discovery-v1"
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent
REPO = EXP_DIR.parents[2]

CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
ANALYSIS = EXP_DIR / "analysis"
CONFIGS = EXP_DIR / "configs"
PREDICTIONS = EXP_DIR / "predictions"

ZINC_ROOT = REPO / "data" / "ZINC"

# Frozen P3 inputs.  They are read-only; this round never modifies the old
# namespace.  The objects are git-ignored but are present on the compute host
# from the previous P3 runs; ad_data.py can rebuild them from official PyG ZINC
# if they are absent.
P3_ROOT = REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
P3_OBJECTS_TRAIN = P3_ROOT / "objects_train10k.pt"
P3_OBJECTS_VALID = P3_ROOT / "objects_valid.pt"
P3_PREP = P3_ROOT / "prep_train10k.json"
P3_SSL_S0 = P3_ROOT / "pretrain_encoder_s0.pt"
P3_SSL_S1 = P3_ROOT / "pretrain_encoder_s1.pt"

P3_BEST_VALID = {
    0: {"valid_mae": 0.07910549778419873, "epoch": 272},
    1: {"valid_mae": 0.07666767603445052, "epoch": 267},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
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


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {y.shape}")
    return float(np.abs(pred - y).mean())


def choose_best(curve: Sequence[Mapping[str, Any]], key: str = "valid_mae") -> dict:
    """Lowest key; ties keep the earlier entry (strict <)."""
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
    """Fork CPU + current-CUDA RNG so eval cannot perturb training."""
    devices: list[int] = []
    if torch.cuda.is_available():
        try:
            devices = [torch.cuda.current_device()]
        except Exception:
            devices = []
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def cosine_with_warmup_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
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
        if torch.cuda.is_available()
        else [],
        "git_commit": git_commit(),
        "git_dirty_tracked": git_dirty(),
        "round_id": ROUND_ID,
        "timestamp": utcnow(),
    }


def ensure_p3_modules() -> None:
    for p in (
        REPO / "tracks/structure-contribution/code",
        REPO / "tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code",
    ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor], device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def optimizer_parameter_count(optimizer: torch.optim.Optimizer) -> int:
    return int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))


def write_curve_csv(path: Path | str, curve: Sequence[Mapping[str, Any]]) -> None:
    import csv

    keys: list[str] = []
    for entry in curve:
        for k in entry:
            if k not in keys:
                keys.append(k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for entry in curve:
            w.writerow(entry)


def atomic_torch_save(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
