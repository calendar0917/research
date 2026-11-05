"""Frozen definitions for the LCRC PCBA learning-rate probe.

Round: ``lcrc-pcba-lr-probe-v1``

This round is a *diagnostic*, not an architecture experiment.  It reuses,
read-only:

* ``lcrc_pcba_scale_pilot_v1`` -- feature cache, split, labels, frozen LCRC-C1,
  best epoch-29 checkpoint (0.23449277039313338 Validation AP),
* ``lcrc_pcba_pipeline_perf_v1`` -- the already-gated ``PinnedStager``.

Nothing here opens the official Test targets.  The label artifact has every
official Test row set to NaN and both the preflight and the trainer refuse to
index a Test row.

The two arms differ **only** in the fixed AdamW learning rate.  Both start from
the same historical epoch-29 weights, use the same seed, the same batch order,
and the same 12-epoch horizon.  AdamW moments are re-initialised in both arms;
the historical checkpoint contains no optimizer state.  This is recorded as a
limitation, not hidden as seamless continuation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# paths / inherited code
# ---------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]

PILOT = REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1"
PILOT_CODE = PILOT / "code"
PERF = REPO / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1"
PERF_CODE = PERF / "code"
for _p in (str(PILOT_CODE), str(PERF_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pc_common as PC  # noqa: E402  (puts bp/prep/tuq/mb2 code on sys.path)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-lr-probe-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABELS = (
    "LR_SIGNAL",
    "TRAINING_SIGNAL_NOT_LR_SPECIFIC",
    "NO_CLEAR_INCREMENT",
    "INCOMPLETE",
)

DATASET = "ogbg-molpcba"
N_TASKS = 128
N_MOLECULES = 437_929
SEED = 0
BATCH_SIZE = 128
TOTAL_EPOCHS = 12
EVAL_EPOCHS: tuple[int, ...] = (0, 3, 6, 9, 12)
STEPS_PER_EPOCH = (350_343 + BATCH_SIZE - 1) // BATCH_SIZE  # 2738

#: Historical reference (frozen; never overwritten by this round).
HIST = {
    "round": "lcrc-pcba-scale-pilot-v1",
    "checkpoint": PILOT / "runs" / "pcba_seed0" / "checkpoint.pt",
    "checkpoint_sha256": "b2affec75fdb947417052f11293922d1e504d490ca48e8b2a074355be6d8edf8",
    "run_json": PILOT / "runs" / "pcba_seed0" / "run.json",
    "best_epoch": 29,
    "best_valid_ap": 0.23449277039313338,
    "feature_spec_hash": "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee",
    "norm_fingerprint": "6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb",
    "split_partition_sha256": "c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f",
    "label_artifact_sha256": "5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01",
    "cache_sha256": "caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035",
    "cache_bytes": 4_740_359_231,
    "cache_files": 23,
    "n_atoms_total": 11_373_137,
    "n_bonds_total": 12_309_186,
    "n_s2_paths": 34_369_144,
}

CACHE_PATH = PILOT / "cache" / "pcba_v1"
LABELS_PATH = PILOT / "analysis" / "pcba_labels_trainvalid.npy"
SPEC_PATH = PILOT / "analysis" / "pcba_spec.json"
SPLIT_PATH = PILOT / "analysis" / "pcba_split.json"
CACHE_MANIFEST = PILOT / "analysis" / "cache_manifest.json"

#: The two arms.  ``assigned_pool`` is the protocol assignment; the actual
#: pool/device is read from the environment at run time and recorded, so a
#: documented deviation (for example a busy GPU0) cannot masquerade as the
#: planned assignment.
GROUPS: dict[str, dict[str, Any]] = {
    "H": {
        "name": "H_high_lr",
        "role": "control_high_lr",
        "learning_rate": 3.0e-4,
        "assigned_pool": "res-gpu0",
    },
    "L": {
        "name": "L_low_lr",
        "role": "experiment_low_lr",
        "learning_rate": 5.0e-5,
        "assigned_pool": "res-gpu1",
    },
}
GROUP_ORDER = ("H", "L")

#: Common recipe.  Only ``learning_rate`` differs between arms; no scheduler.
PROBE_RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "weight_decay": 1.0e-3,
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "scheduler": "none",
    "learning_rate": "fixed_per_arm (H=3e-4, L=5e-5)",
    "precision": "fp32",
    "epochs": TOTAL_EPOCHS,
    "eval_epochs": list(EVAL_EPOCHS),
    "early_stopping": False,
    "seed": SEED,
    "loss": "masked BCEWithLogitsLoss over observed entries only",
    "selection": "max Validation AP (OGB official Evaluator); ties -> earlier eval epoch",
    "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
}

#: Resource guards.  Budget is total for both arms *and* both preflights.
BUDGET: dict[str, Any] = {
    "gpu_hours_total": 1.5,
    "per_group_soft_seconds": 2050.0,
    "per_group_hard_seconds": 2200.0,
    "per_group_outer_kill_seconds": 2300.0,
    "preflight_smoke_steps": 20,
    "preflight_equivalence_steps": 3,
}

#: Decision thresholds fixed before the runs.
DECISION = {
    "improvement_floor": 0.005,
    "historical_ap_literal": 0.23449,
    "main_metric": "D_LR = AP_L(epoch12) - AP_H(epoch12)",
    "historical_metric": "D_old = AP_L(epoch12) - 0.23449",
    "rules": {
        "LR_SIGNAL": "D_LR >= 0.005 and D_old >= 0.005",
        "TRAINING_SIGNAL_NOT_LR_SPECIFIC": (
            "both epoch12 AP improved by >= 0.005 over the historical 0.23449, "
            "but D_LR < 0.005"
        ),
        "NO_CLEAR_INCREMENT": "complete paired run that meets neither threshold",
        "INCOMPLETE": "resource, environment or correctness failure prevented the paired run",
    },
    "statistical_claim": "none; these are resource-decision labels only",
}


# ---------------------------------------------------------------------------
# IO / hashing
# ---------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(obj: Any):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
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
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_default(x) for x in obj]
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def write_json(path: Path | str, obj: Any, *, compact: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if compact:
        text = json.dumps(obj, separators=(",", ":"), sort_keys=True, default=_json_default)
    else:
        text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default)
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_int64(values: Iterable[int]) -> str:
    arr = np.ascontiguousarray(np.asarray(list(values), dtype=np.int64))
    h = hashlib.sha256()
    h.update(struct.pack("<Q", int(arr.size)))
    h.update(arr.tobytes())
    return h.hexdigest()


def sha256_batches(batches: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(batches)))
    for batch in batches:
        arr = np.ascontiguousarray(np.asarray(batch, dtype=np.int64))
        h.update(struct.pack("<Q", int(arr.size)))
        h.update(arr.tobytes())
    return h.hexdigest()


def track_rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def log(msg: str) -> None:
    print(f"[{utcnow()}] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return ""


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def environment() -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "hostname": platform.node(),
        "cuda_available": bool(cuda),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
    }
    if cuda:
        out["cuda_device_index"] = int(torch.cuda.current_device())
        out["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
        out["cuda_device_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    return out


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, CPU torch and every visible CUDA device."""
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return {
        "seed": int(seed),
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": bool(torch.cuda.is_available()),
    }


def _tensor_sha256(value: Any) -> str:
    import torch

    if torch.is_tensor(value):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Full stochastic state, including per-device CUDA generators."""
    import torch

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
    return state


def rng_state_hashes(state: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "python_random_sha256": sha256_bytes(pickle.dumps(state["python_random"], protocol=5)),
        "numpy_sha256": sha256_bytes(pickle.dumps(state["numpy"], protocol=5)),
        "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
        "torch_cuda_sha256": [_tensor_sha256(x) for x in state.get("torch_cuda", [])],
    }
    joined = "|".join(
        [out["python_random_sha256"], out["numpy_sha256"], out["torch_cpu_sha256"]]
        + out["torch_cuda_sha256"]
    )
    out["combined_sha256"] = sha256_bytes(joined.encode())
    return out


# ---------------------------------------------------------------------------
# frozen inheritance / checkpoint metadata
# ---------------------------------------------------------------------------


def assert_inherited_constants() -> dict[str, Any]:
    """Fail hard if any inherited model/recipe constant drifted."""
    inherited = PC.assert_frozen_inheritance()
    if PC.MODEL_VARIANT != "C1":
        raise RuntimeError("this round only runs LCRC-C1")
    if PC.MODEL_CONFIG != PC.FROZEN_WIDTHS:
        raise RuntimeError(f"model config drifted: {PC.MODEL_CONFIG}")
    if float(PC.RECIPE["dropout"]) != 0.10:
        raise RuntimeError(f"inherited dropout {PC.RECIPE['dropout']} != 0.10")
    if tuple(PC.RECIPE["adam_betas"]) != (0.9, 0.999):
        raise RuntimeError(f"inherited betas {PC.RECIPE['adam_betas']} != (0.9, 0.999)")
    if float(PC.RECIPE["adam_eps"]) != 1.0e-8:
        raise RuntimeError(f"inherited eps {PC.RECIPE['adam_eps']} != 1e-8")
    if float(PC.RECIPE["weight_decay"]) != 1.0e-3:
        raise RuntimeError(f"inherited weight decay {PC.RECIPE['weight_decay']} != 1e-3")
    if float(PC.RECIPE["gradient_clip"]) != 5.0:
        raise RuntimeError(f"inherited gradient clip {PC.RECIPE['gradient_clip']} != 5.0")
    if int(PC.RECIPE["batch_size"]) != BATCH_SIZE:
        raise RuntimeError(f"inherited batch size {PC.RECIPE['batch_size']} != {BATCH_SIZE}")
    if int(PC.N_TASKS) != N_TASKS:
        raise RuntimeError(f"inherited task count {PC.N_TASKS} != {N_TASKS}")
    return {
        **inherited,
        "model_config": dict(PC.MODEL_CONFIG),
        "model_variant_definition": PC.MODEL_VARIANT_DEFINITION,
        "inherited_recipe": {
            key: PC.RECIPE[key]
            for key in (
                "optimizer",
                "weight_decay",
                "adam_betas",
                "adam_eps",
                "dropout",
                "batch_size",
                "gradient_clip",
            )
        },
        "n_tasks": N_TASKS,
    }


def verify_historical_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "round_id": HIST["round"],
        "dataset": DATASET,
        "variant": "C1",
        "seed": SEED,
        "n_outputs": N_TASKS,
        "best_epoch": HIST["best_epoch"],
        "feature_spec_hash": HIST["feature_spec_hash"],
        "norm_fingerprint": HIST["norm_fingerprint"],
    }
    mismatches = {k: {"expected": v, "found": checkpoint.get(k)} for k, v in expected.items() if checkpoint.get(k) != v}
    if abs(float(checkpoint.get("best_metric", float("nan"))) - HIST["best_valid_ap"]) > 1e-12:
        mismatches["best_metric"] = {
            "expected": HIST["best_valid_ap"],
            "found": checkpoint.get("best_metric"),
        }
    if checkpoint.get("result_label") != "SCALE_PILOT":
        mismatches["result_label"] = {"expected": "SCALE_PILOT", "found": checkpoint.get("result_label")}
    if checkpoint.get("metric") != "Validation AP (OGB official Evaluator)":
        mismatches["metric"] = {
            "expected": "Validation AP (OGB official Evaluator)",
            "found": checkpoint.get("metric"),
        }
    if mismatches:
        raise RuntimeError(f"historical checkpoint metadata mismatch: {mismatches}")
    sha = sha256_file(HIST["checkpoint"])
    if sha != HIST["checkpoint_sha256"]:
        raise RuntimeError(f"historical checkpoint sha256 {sha} != frozen {HIST['checkpoint_sha256']}")
    return {
        "path": track_rel(HIST["checkpoint"]),
        "sha256": sha,
        "metadata": {k: _json_default(v) for k, v in checkpoint.items() if k != "state_dict"},
    }


def recipe_for_group(group: str) -> dict[str, Any]:
    if group not in GROUPS:
        raise KeyError(f"unknown group {group!r}")
    recipe = dict(PROBE_RECIPE)
    recipe["learning_rate"] = float(GROUPS[group]["learning_rate"])
    recipe["group"] = group
    recipe["group_name"] = GROUPS[group]["name"]
    recipe["role"] = GROUPS[group]["role"]
    recipe["assigned_pool"] = GROUPS[group]["assigned_pool"]
    return recipe


def decide_label(
    h_ap12: float | None,
    l_ap12: float | None,
    h_status: str,
    l_status: str,
    old_ap: float = DECISION["historical_ap_literal"],
) -> dict[str, Any]:
    """Apply the pre-registered resource-decision rule verbatim."""
    floor = float(DECISION["improvement_floor"])
    if h_status != "COMPLETED" or l_status != "COMPLETED":
        label = "INCOMPLETE"
    elif h_ap12 is None or l_ap12 is None:
        label = "INCOMPLETE"
    else:
        d_lr = float(l_ap12) - float(h_ap12)
        d_old = float(l_ap12) - float(old_ap)
        both_improved = (float(h_ap12) - old_ap) >= floor and d_old >= floor
        if d_lr >= floor and d_old >= floor:
            label = "LR_SIGNAL"
        elif both_improved:
            label = "TRAINING_SIGNAL_NOT_LR_SPECIFIC"
        else:
            label = "NO_CLEAR_INCREMENT"
    d_lr = None if h_ap12 is None or l_ap12 is None else float(l_ap12) - float(h_ap12)
    d_old = None if l_ap12 is None else float(l_ap12) - float(old_ap)
    return {
        "label": label,
        "D_LR": d_lr,
        "D_old": d_old,
        "H_epoch12_ap": h_ap12,
        "L_epoch12_ap": l_ap12,
        "historical_best_ap": float(HIST["best_valid_ap"]),
        "historical_ap_used_for_decision": float(old_ap),
        "improvement_floor": floor,
        "h_status": h_status,
        "l_status": l_status,
        "rules": DECISION["rules"],
        "statistical_claim": DECISION["statistical_claim"],
    }


__all__ = [
    "ANALYSIS",
    "BATCH_SIZE",
    "BUDGET",
    "CACHE_MANIFEST",
    "CACHE_PATH",
    "CODE_DIR",
    "DATASET",
    "DECISION",
    "EVAL_EPOCHS",
    "GROUPS",
    "GROUP_ORDER",
    "HIST",
    "LABELS_PATH",
    "N_MOLECULES",
    "N_TASKS",
    "PERF",
    "PERF_CODE",
    "PILOT",
    "PILOT_CODE",
    "PROBE_RECIPE",
    "PROTOCOL_ID",
    "REPO",
    "RESULT_LABELS",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SEED",
    "SPEC_PATH",
    "SPLIT_PATH",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "assert_inherited_constants",
    "capture_rng_state",
    "decide_label",
    "environment",
    "git_commit",
    "git_dirty",
    "log",
    "read_json",
    "recipe_for_group",
    "rng_state_hashes",
    "seed_everything",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "track_rel",
    "utcnow",
    "verify_historical_checkpoint_metadata",
    "write_json",
]
