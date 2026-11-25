"""Identity, paths, frozen constants, recipe and IO for CSC v0.

Nothing scientific is decided here at runtime: every number below is the frozen
protocol constant of ``PROTOCOL.md``.  Paths are resolved from this file so the
module is identical in the local checkout and in the compute checkout on ``res``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROUND_ID = "csc-v0"
PROTOCOL_ID = "context-conditioned-structural-computation-v0"
SCHEMA_RUN = "csc-run-v1"
SCHEMA_CHECKPOINT = "csc-checkpoint-v1"

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
RUNS_DIR = ROUND_DIR / "runs"
CONFIG_DIR = ROUND_DIR / "configs"

# ---------------------------------------------------------------------------
# read-only reuse roots
# ---------------------------------------------------------------------------

SQL_DIR = REPO_ROOT / "tracks/structural-query-learning"
SQL_CODE = SQL_DIR / "code"
SQL_RESULTS = SQL_DIR / "results/sq-pilot-small-v1"
BS_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1"
BS_CODE = BS_DIR / "code"
SG_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1"
SG_CODE = SG_DIR / "code"
NMP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1"
NMP_CODE = NMP_DIR / "code"
TU_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_tu_quick_screen_v1"
TU_CODE = TU_DIR / "code"
DISSECT_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
DISSECT_CODE = DISSECT_DIR / "code"
PREP_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1"
PREP_CODE = PREP_DIR / "code"
RC_DIR = REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1"
RC_CODE = RC_DIR / "code"
DSCR_DIR = REPO_ROOT / "tracks/structural-sufficiency/direct_structural_computation_recurrence_v0"


def setup_paths() -> None:
    """Put every read-only dependency on ``sys.path`` exactly once."""
    for path in (
        str(CODE_DIR),
        str(BS_CODE),
        str(SG_CODE),
        str(NMP_CODE),
        str(TU_CODE),
        str(RC_CODE),
        str(DISSECT_CODE),
        str(PREP_CODE),
        str(SQL_CODE),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# data / protocol constants
# ---------------------------------------------------------------------------

DATASETS: tuple[str, ...] = ("ogbg-mollipo", "ogbg-molesol")
MAIN_DATASET = "ogbg-mollipo"
SUPPLEMENT_DATASET = "ogbg-molesol"
SPLITS: tuple[str, ...] = ("train", "valid", "test")

EXPECTED_SPLIT_SIZES = {
    "ogbg-mollipo": {"train": 3360, "valid": 420, "test": 420},
    "ogbg-molesol": {"train": 902, "valid": 113, "test": 113},
}
EXPECTED_CACHE_SHA256 = {
    "ogbg_molesol_sq-pilot-cache-v1.npz": "3b7a8e16ca9c7ab240add84fbe26050dc81e22359bcc36f1c8a8111b77aec24a",
    "ogbg_mollipo_sq-pilot-cache-v1.npz": "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c",
    "ogbg_molesol_sq-pilot-lipo-raw-v1.npz": "a3ab9f7e22798a8f6fdc7f0913b1136b7cf59d4ffc7168faecc7f8b3094710b1",
    "ogbg_mollipo_sq-pilot-lipo-raw-v1.npz": "3005aa55e766d83a237b1659ad2cabb51b8702eef6d82c203fdecf9cb3c42cc3",
}

SEED = 0
TEST_ROLE = "DEVELOPMENT_TEST"

#: one arm per structural computation (section 6 of PROTOCOL.md)
ARMS: dict[str, dict[str, Any]] = {
    "A": {"name": "FIXED", "role": "original FIXED1 second stage, Q = 0", "statistics": False},
    "B": {"name": "ONE_READ", "role": "second stage reads LN2(h0[i]) on every real incoming bond, Q = 0", "statistics": False},
    "C": {"name": "MIXED", "role": "Gate on the same environment, correction content replaced by the count-preserving mixed grouping", "statistics": True},
    "D": {"name": "CSC", "role": "Gate on the same environment, correction content = real relation-grouped content", "statistics": True},
}
ARM_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
STAT_ARMS: tuple[str, ...] = ("C", "D")

QUEUE: dict[str, list[str]] = {
    "ogbg-mollipo": ["A", "D", "C", "B"],
    "ogbg-molesol": ["A", "C", "D"],
}

#: local structural relations, from the ``lipo_v1`` bond-type vocabulary
RELATION_NAMES: tuple[str, ...] = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER")
N_RELATIONS = len(RELATION_NAMES)
#: cache ``bond_type`` is one-based: 1..4 = SINGLE..AROMATIC, 5 = UNK -> OTHER
BOND_TYPE_TO_RELATION = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

# architecture constants (section 4.1 / 5)
WIDTH = 64           # d, the local hidden state
BLOCK_HIDDEN = 128   # q, the structural computation block width
GATE_HIDDEN = 16
DELTA_SCALE = 0.5
N_OUT = 1
CFG_KEY = 1          # BSModel scale-1 (1x) widths

#: frozen recipe (section 3.2)
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "peak_lr": 3.0e-4,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-3,
    "batch_size": 128,
    "eval_batch_size": 256,
    "gradient_clip": 5.0,
    "epochs": 360,
    "scheduler": "per_step_cosine_no_warmup",
    "amp": False,
    "ema": False,
    "swa": False,
    "soup": False,
    "early_stopping": False,
    "auxiliary_loss": False,
    "selection": "minimum full official Valid RMSE; strictly smaller replaces, ties keep the earlier epoch",
    "shuffle": "np.random.default_rng(0).permutation over Train split-local rows, one draw per epoch, shared by every arm",
}

#: attribution / explanation constants (sections 11-12)
QUADRATURE_NODES = 16
QUADRATURE_NODES_FALLBACK = 32
TRACE_ABS_TOL = 1e-4
TRACE_REL_TOL = 1e-3
GAMMA_EPS = 1e-6
TREE_MAX_DEPTH = 4
TREE_MIN_LEAF = 100
TREE_RANDOM_STATE = 0
TREE_TOP_CODES = 32
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261126

#: budget (section 7.2): GPU hours
GPU_TARGET_HOURS = 2.0
GPU_HARD_HOURS = 3.0


# ---------------------------------------------------------------------------
# hashing / json / io
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return sha256_bytes(arr.tobytes() + f"|{arr.dtype}|{arr.shape}".encode())


def state_fingerprint(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(arr.dtype).encode())
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def dumps_json(payload: Any) -> str:
    return json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps_json(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            out = {}
            for key in columns:
                value = row.get(key)
                if isinstance(value, np.generic):
                    value = value.item()
                out[key] = value
            writer.writerow(out)
    os.replace(tmp, path)
    return path


def atomic_torch_save(path: Path | str, payload: Mapping[str, Any]) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return path


def track_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def log(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# metrics / statistics
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def target_scaler(labels: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("invalid Train target statistics")
    return {"mean": mean, "std": std, "n": int(y.size), "ddof": 0, "fit_split": "train"}


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(values))
    half = 0.5 * cw[-1]
    idx = int(np.searchsorted(cw, half, side="left"))
    return float(v[min(idx, v.size - 1)])


def weighted_mae(values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * np.abs(np.asarray(values) - np.asarray(target))) / w.sum())


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_at(step: int, total_steps: int) -> float:
    """Per-step cosine from peak to min, no warmup (frozen recipe)."""
    total = max(1, int(total_steps))
    frac = min(1.0, max(0.0, float(step) / float(total)))
    return float(RECIPE["min_lr"]) + 0.5 * (float(RECIPE["peak_lr"]) - float(RECIPE["min_lr"])) * (
        1.0 + math.cos(math.pi * frac)
    )


def steps_per_epoch(dataset: str) -> int:
    return int(math.ceil(EXPECTED_SPLIT_SIZES[dataset]["train"] / int(RECIPE["batch_size"])))


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------


def _cmd(args: Sequence[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20, cwd=REPO_ROOT).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_commit() -> str | None:
    return _cmd(["git", "rev-parse", "HEAD"])


def git_dirty() -> bool | None:
    out = _cmd(["git", "status", "--porcelain"])
    return None if out is None else bool(out)


def environment() -> dict[str, Any]:
    import torch

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "cwd": str(Path.cwd()),
    }
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        env["cuda_version"] = torch.version.cuda
    return env


def set_seeds(seed: int) -> dict[str, Any]:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return {"seed": int(seed), "numpy": "legacy global seed (batch order uses its own generator)"}


def rng_hashes() -> dict[str, Any]:
    import torch

    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "torch_cpu": sha256_bytes(cpu.numpy().tobytes()),
        "torch_cuda": [sha256_bytes(t.numpy().tobytes()) for t in cuda],
    }


def capture_rng() -> dict[str, Any]:
    import torch

    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [t.clone() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_rng_state(payload["torch_cpu"])
    if payload.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    np.random.set_state(payload["numpy"])
    random.setstate(payload["python"])
    return rng_hashes()


def state_cpu(model) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def is_new_parameter(name: str) -> bool:
    return name.startswith(("bond_proj.", "blocks.", "gate."))


def recipe(dataset: str, arm: str) -> dict[str, Any]:
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in RECIPE.items()}
    out.update(
        {
            "round_id": ROUND_ID,
            "dataset": dataset,
            "arm": arm,
            "arm_name": ARMS[arm]["name"],
            "seed": SEED,
            "steps_per_epoch": steps_per_epoch(dataset),
            "total_steps": steps_per_epoch(dataset) * int(RECIPE["epochs"]),
            "width": WIDTH,
            "block_hidden": BLOCK_HIDDEN,
            "gate_hidden": GATE_HIDDEN,
            "delta_scale": DELTA_SCALE,
            "n_relations": N_RELATIONS,
            "relation_names": list(RELATION_NAMES),
            "lr_step0": lr_at(0, steps_per_epoch(dataset) * int(RECIPE["epochs"])),
            "lr_final_step": lr_at(steps_per_epoch(dataset) * int(RECIPE["epochs"]), steps_per_epoch(dataset) * int(RECIPE["epochs"])),
            "test_role": TEST_ROLE,
        }
    )
    return out
