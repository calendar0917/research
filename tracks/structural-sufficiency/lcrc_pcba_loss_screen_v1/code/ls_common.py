"""Frozen definitions for the LCRC PCBA loss-screen round.

Round: ``lcrc-pcba-loss-screen-v1``

This round has two fixed phases:

* **A** -- a read-only Train/Validation diagnostic of the frozen epoch-9
  checkpoint from ``lcrc_pcba_lr_probe_v1/runs/L_low_lr/checkpoint_best.pt``.
* **B** -- a six-epoch paired screen of the historical masked BCE (group M)
  against a task-macro BCE (group T), starting from the same epoch-9 weights
  **and** the same AdamW optimizer state.

The official Test targets are unreachable: the label artifact has every Test
row set to NaN, and every loading/evaluation path in this round refuses a
finite Test label or a Test-row index.  No Test metric is produced here.

Everything in this module is frozen before the first GPU run.  Implementation
helpers live in ``ls_losses.py``, ``ls_checkpoint.py``, ``ls_diagnostic.py``,
``ls_preflight.py`` and ``ls_train.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
LR_PROBE = REPO / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1"
LR_PROBE_CODE = LR_PROBE / "code"
for _p in (str(PILOT_CODE), str(PERF_CODE), str(LR_PROBE_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANALYSIS = ROUND / "analysis"
RUNS = ROUND / "runs"
TESTS = ROUND / "tests"

ROUND_ID = "lcrc-pcba-loss-screen-v1"
PROTOCOL_ID = ROUND_ID
RESULT_LABELS = (
    "SCREEN_PROMISING",
    "SCREEN_MIXED",
    "SCREEN_NO_SIGNAL",
    "INCOMPLETE_OR_INVALID",
)

DATASET = "ogbg-molpcba"
N_TASKS = 128
N_MOLECULES = 437_929
N_TRAIN = 350_343
N_VALID = 43_793
N_TEST = 43_793
BATCH_SIZE = 128
SEED = 0
TOTAL_EPOCHS = 6
EVAL_EPOCHS: tuple[int, ...] = (0, 2, 4, 6)
STEPS_PER_EPOCH = (N_TRAIN + BATCH_SIZE - 1) // BATCH_SIZE  # 2738

#: Frozen source checkpoint for both phases.  The file is not copied or
#: rewritten; its SHA-256 is checked at load time.
SOURCE = {
    "round": "lcrc-pcba-lr-probe-v1",
    "group": "L",
    "group_name": "L_low_lr",
    "checkpoint": LR_PROBE / "runs" / "L_low_lr" / "checkpoint_best.pt",
    "checkpoint_sha256": "b158f1669f6d0ccbf5fc05f510050c69ef2fba9f6258cbe59d4d3d9a938eb6c5",
    "best_epoch": 9,
    "best_valid_ap": 0.2374902968831333,
    "optimizer_step_per_param": 9 * STEPS_PER_EPOCH,  # 24642
    "feature_spec_hash": "7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee",
    "norm_fingerprint": "6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb",
    "split_partition_sha256": "c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f",
    "label_artifact_sha256": "5c3b5fbe642d80113522832fe01090097b860010868c59f8696bc247d5027d01",
    "cache_sha256": "caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035",
    "n_parameters": 245_184,
}

CACHE_PATH = PILOT / "cache" / "pcba_v1"
LABELS_PATH = PILOT / "analysis" / "pcba_labels_trainvalid.npy"
SPEC_PATH = PILOT / "analysis" / "pcba_spec.json"
SPLIT_PATH = PILOT / "analysis" / "pcba_split.json"
CACHE_MANIFEST = PILOT / "analysis" / "cache_manifest.json"

#: Assigned GPU pools.  Actual CUDA_VISIBLE_DEVICES and GPU model are recorded
#: at run time; a documented co-tenant is an execution fact, not a model change.
GROUPS: dict[str, dict[str, Any]] = {
    "M": {
        "name": "M_micro",
        "role": "control_masked_bce",
        "loss": "masked_bce",
        "assigned_pool": "res-gpu0",
    },
    "T": {
        "name": "T_task_macro",
        "role": "experiment_task_macro_bce",
        "loss": "task_macro_bce",
        "assigned_pool": "res-gpu1",
    },
}
GROUP_ORDER = ("M", "T")

#: Common recipe.  Only the loss construction differs between arms.
RECIPE: dict[str, Any] = {
    "optimizer": "AdamW",
    "learning_rate": 5.0e-5,
    "weight_decay": 1.0e-3,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1.0e-8,
    "dropout": 0.10,
    "batch_size": BATCH_SIZE,
    "gradient_clip": 5.0,
    "scheduler": "none",
    "precision": "fp32",
    "epochs": TOTAL_EPOCHS,
    "eval_epochs": list(EVAL_EPOCHS),
    "early_stopping": False,
    "seed": SEED,
    "selection": "fixed epoch 6; best checkpoint is auxiliary only",
    "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
}

#: Resource guards.  The total budget covers phase A, both preflight records and
#: both formal six-epoch arms.
BUDGET: dict[str, Any] = {
    "gpu_hours_total": 1.0,
    "phase_a_soft_seconds": 600.0,
    "phase_a_hard_seconds": 720.0,
    "preflight_soft_seconds": 600.0,
    "per_group_soft_seconds": 1200.0,
    "per_group_hard_seconds": 1350.0,
    "per_group_outer_kill_seconds": 1500.0,
}

#: Decision rule fixed before any phase-B run.
DECISION = {
    "ap0": SOURCE["best_valid_ap"],
    "improvement_floor": 0.003,
    "task_majority": 64,
    "n_scorable_tasks_expected": 127,
    "rules": {
        "SCREEN_PROMISING": (
            "both arms complete 6 epochs; D6 = AP_T6 - AP_M6 >= +0.003; "
            "AP_T6 >= AP0; at least 64 of 127 scorable tasks have T > M"
        ),
        "SCREEN_MIXED": (
            "both arms complete 6 epochs; D6 >= +0.003; but not all "
            "SCREEN_PROMISING conditions hold"
        ),
        "SCREEN_NO_SIGNAL": "both arms complete 6 epochs; D6 < +0.003",
        "INCOMPLETE_OR_INVALID": (
            "either arm incomplete, or a correctness/data-isolation/checkpoint "
            "gate fails, or an unexplained configuration difference exists"
        ),
    },
    "statistical_claim": (
        "the +0.003 floor is a resource-decision threshold, not a significance "
        "threshold and not a guarantee of the next full-run gain"
    ),
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


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
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
    out: dict[str, Any] = {
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
        out["cuda_device_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
    return out


def gpu_occupancy() -> dict[str, Any]:
    """Record visible free memory and any co-tenant processes (execution fact)."""
    import torch

    out: dict[str, Any] = {"captured_utc": utcnow()}
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            out["visible_device_mem_get_info"] = {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "used_bytes": int(total - free),
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["visible_device_mem_get_info_error"] = f"{type(exc).__name__}: {exc}"
    for key, args in (
        ("nvidia_smi_gpus", ["--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        ("nvidia_smi_compute_apps", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
    ):
        try:
            out[key] = subprocess.check_output(
                ["nvidia-smi", *args], text=True, stderr=subprocess.DEVNULL, timeout=15
            ).strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[key] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> dict[str, Any]:
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
        "python_random_sha256": sha256_bytes(__import__("pickle").dumps(state["python_random"], protocol=5)),
        "numpy_sha256": sha256_bytes(__import__("pickle").dumps(state["numpy"], protocol=5)),
        "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
        "torch_cuda_sha256": [_tensor_sha256(x) for x in state.get("torch_cuda", [])],
    }
    joined = "|".join(
        [out["python_random_sha256"], out["numpy_sha256"], out["torch_cpu_sha256"]]
        + out["torch_cuda_sha256"]
    )
    out["combined_sha256"] = sha256_bytes(joined.encode())
    return out


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore Python/NumPy/CPU/CUDA generators from a checkpoint state.

    Both arms call this after loading weights and optimizer state, so dropout
    masks are comparable when the batch order is identical.
    """
    import torch

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state.get("torch_cuda", []))
    if torch.cuda.is_available():
        if not cuda_states:
            raise RuntimeError("checkpoint has no CUDA RNG state but CUDA is available")
        # Each pool exposes exactly one device; restore the saved generator on it.
        torch.cuda.set_rng_state(cuda_states[0], torch.cuda.current_device())
    return {
        "restored_python_random": True,
        "restored_numpy": True,
        "restored_torch_cpu": True,
        "restored_cuda_devices": [0] if torch.cuda.is_available() else [],
        "source_cuda_states": len(cuda_states),
    }


# ---------------------------------------------------------------------------
# paired batch order (new fixed six-epoch order; not the historical epoch-10+ order)
# ---------------------------------------------------------------------------


def make_order_plan(
    train_rows: Sequence[int],
    *,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    epochs: int = TOTAL_EPOCHS,
) -> list[dict[str, Any]]:
    """One CPU generator, ``torch.randperm`` per epoch, identical for both arms."""
    import torch

    rows = np.asarray(train_rows, dtype=np.int64)
    generator = torch.Generator().manual_seed(int(seed))
    plan: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(len(rows), generator=generator).numpy()
        shuffled = rows[perm]
        batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
        plan.append(
            {
                "epoch": int(epoch),
                "rows": shuffled,
                "batches": batches,
                "row_order_sha256": sha256_int64(shuffled),
                "batch_plan_sha256": sha256_batches(batches),
                "n_batches": len(batches),
                "first_row": int(shuffled[0]),
                "last_row": int(shuffled[-1]),
                "first_batch_rows": [int(x) for x in batches[0]],
                "last_batch_size": int(len(batches[-1])),
            }
        )
    generator_state = generator.get_state()
    plan_sha = sha256_bytes(
        json.dumps([p["row_order_sha256"] for p in plan], sort_keys=True).encode()
    )
    for record in plan:
        record["generator_state_sha256"] = sha256_bytes(generator_state.numpy().tobytes())
        record["global_plan_sha256"] = plan_sha
    return plan


def order_summary(plan: Sequence[Mapping[str, Any]], *, n_train_rows: int, seed: int = SEED) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "batch_size": BATCH_SIZE,
        "n_train_rows": int(n_train_rows),
        "steps_per_epoch": int((n_train_rows + BATCH_SIZE - 1) // BATCH_SIZE),
        "epochs": len(plan),
        "source": "torch.Generator().manual_seed(0) + torch.randperm(len(train_rows)) per epoch; new six-epoch fork",
        "not_historical_epoch_10_plus_order": True,
        "global_plan_sha256": plan[0]["global_plan_sha256"] if plan else "",
        "generator_state_sha256": plan[0]["generator_state_sha256"] if plan else "",
        "epochs_detail": [
            {
                "epoch": int(p["epoch"]),
                "n_batches": int(p["n_batches"]),
                "row_order_sha256": p["row_order_sha256"],
                "batch_plan_sha256": p["batch_plan_sha256"],
                "first_row": int(p["first_row"]),
                "last_row": int(p["last_row"]),
                "first_batch_rows": list(p["first_batch_rows"]),
                "last_batch_size": int(p["last_batch_size"]),
            }
            for p in plan
        ],
    }


def assert_order_isolation(
    plan: Sequence[Mapping[str, Any]],
    train_rows: Sequence[int],
    valid_rows: Sequence[int],
    test_rows: Sequence[int],
) -> dict[str, Any]:
    train_sorted = np.sort(np.asarray(train_rows, dtype=np.int64))
    n = N_MOLECULES
    valid_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    valid_mask[np.asarray(valid_rows, dtype=np.int64)] = True
    test_mask[np.asarray(test_rows, dtype=np.int64)] = True
    checks = []
    for record in plan:
        rows = np.asarray(record["rows"], dtype=np.int64)
        checks.append(
            {
                "epoch": int(record["epoch"]),
                "is_train_permutation": bool(np.array_equal(np.sort(rows), train_sorted)),
                "n_rows": int(rows.size),
                "n_unique_rows": int(np.unique(rows).size),
                "valid_rows_in_epoch": int(valid_mask[rows].sum()),
                "test_rows_in_epoch": int(test_mask[rows].sum()),
            }
        )
    passed = all(
        c["is_train_permutation"]
        and c["n_rows"] == int(train_sorted.size)
        and c["n_unique_rows"] == int(train_sorted.size)
        and c["valid_rows_in_epoch"] == 0
        and c["test_rows_in_epoch"] == 0
        for c in checks
    )
    return {
        "passed": bool(passed),
        "epochs_checked": len(checks),
        "train_rows": int(train_sorted.size),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# run helpers
# ---------------------------------------------------------------------------


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return sha256_file(path)


class Watchdog:
    """Arm a hard-stop flag; the training loop saves and exits gracefully."""

    def __init__(self, hard_seconds: float):
        import threading

        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self.fired_at: float | None = None
        self._t0 = time.perf_counter()
        self._timer = threading.Timer(self.hard_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired_at = time.perf_counter() - self._t0
        self.stop.set()

    def cancel(self) -> None:
        self._timer.cancel()


class Tracker:
    """Best Validation AP, ties -> earlier eval epoch (strict improvement only)."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_metric = -math.inf
        self.scores: list[dict[str, Any]] = []

    def consider(self, epoch: int, metric: float) -> bool:
        strict = self.best_epoch is None or float(metric) > float(self.best_metric)
        self.scores.append(
            {"epoch": int(epoch), "metric": float(metric), "strict_improve": bool(strict)}
        )
        if strict:
            self.best_epoch = int(epoch)
            self.best_metric = float(metric)
        return strict

    def tie_check(self) -> dict[str, Any]:
        best_epoch = None
        best_key = None
        for row in self.scores:
            key = (row["metric"], -row["epoch"])
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = row["epoch"]
        return {
            "argbest_epoch": best_epoch,
            "tracked_epoch": self.best_epoch,
            "consistent": best_epoch == self.best_epoch,
            "rule": "max Validation AP; ties resolved to the earlier eval epoch",
        }


def recipe_for_group(group: str) -> dict[str, Any]:
    if group not in GROUPS:
        raise KeyError(f"unknown group {group!r}")
    recipe = dict(RECIPE)
    recipe["group"] = group
    recipe["group_name"] = GROUPS[group]["name"]
    recipe["role"] = GROUPS[group]["role"]
    recipe["loss"] = GROUPS[group]["loss"]
    recipe["assigned_pool"] = GROUPS[group]["assigned_pool"]
    return recipe


def decide_screen_label(
    *,
    ap_m6: float | None,
    ap_t6: float | None,
    status_m: str,
    status_t: str,
    n_t_better: int | None,
    n_scored_both: int | None,
    config_valid: bool,
    ap0: float = DECISION["ap0"],
) -> dict[str, Any]:
    floor = float(DECISION["improvement_floor"])
    d6 = None if ap_m6 is None or ap_t6 is None else float(ap_t6) - float(ap_m6)
    conditions = {
        "both_completed": status_m == "COMPLETED" and status_t == "COMPLETED",
        "config_valid": bool(config_valid),
        "d6_ge_floor": bool(d6 is not None and d6 >= floor),
        "ap_t6_ge_ap0": bool(ap_t6 is not None and float(ap_t6) >= float(ap0)),
        "n_scored_both_is_127": bool(
            n_scored_both is not None
            and int(n_scored_both) == int(DECISION["n_scorable_tasks_expected"])
        ),
        "task_majority": bool(
            n_t_better is not None and int(n_t_better) >= int(DECISION["task_majority"])
        ),
    }
    if not conditions["both_completed"] or not conditions["config_valid"]:
        label = "INCOMPLETE_OR_INVALID"
    elif d6 is None:
        label = "INCOMPLETE_OR_INVALID"
    elif d6 < floor:
        label = "SCREEN_NO_SIGNAL"
    elif all(
        conditions[k]
        for k in (
            "both_completed",
            "config_valid",
            "d6_ge_floor",
            "ap_t6_ge_ap0",
            "n_scored_both_is_127",
            "task_majority",
        )
    ):
        label = "SCREEN_PROMISING"
    else:
        label = "SCREEN_MIXED"
    return {
        "label": label,
        "D6": d6,
        "AP_M6": ap_m6,
        "AP_T6": ap_t6,
        "AP0": float(ap0),
        "T6_minus_AP0": None if ap_t6 is None else float(ap_t6) - float(ap0),
        "M6_minus_AP0": None if ap_m6 is None else float(ap_m6) - float(ap0),
        "n_T_better": n_t_better,
        "n_scored_both": n_scored_both,
        "improvement_floor": floor,
        "task_majority": int(DECISION["task_majority"]),
        "conditions": conditions,
        "rules": DECISION["rules"],
        "statistical_claim": DECISION["statistical_claim"],
    }


def assert_inherited_constants() -> dict[str, Any]:
    """Fail hard if the inherited model/recipe constants drifted."""
    import pc_common as PC

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
    "LABELS_PATH",
    "LR_PROBE",
    "N_MOLECULES",
    "N_TASKS",
    "N_TRAIN",
    "N_VALID",
    "N_TEST",
    "PERF",
    "PILOT",
    "PROTOCOL_ID",
    "RECIPE",
    "REPO",
    "RESULT_LABELS",
    "ROUND",
    "ROUND_ID",
    "RUNS",
    "SEED",
    "SOURCE",
    "SPEC_PATH",
    "SPLIT_PATH",
    "STEPS_PER_EPOCH",
    "TESTS",
    "TOTAL_EPOCHS",
    "Tracker",
    "Watchdog",
    "assert_inherited_constants",
    "assert_order_isolation",
    "atomic_torch_save",
    "capture_rng_state",
    "decide_screen_label",
    "environment",
    "git_commit",
    "git_dirty",
    "gpu_occupancy",
    "log",
    "make_order_plan",
    "order_summary",
    "read_json",
    "recipe_for_group",
    "restore_rng_state",
    "rng_state_hashes",
    "seed_everything",
    "sha256_array",
    "sha256_batches",
    "sha256_bytes",
    "sha256_file",
    "sha256_int64",
    "track_rel",
    "utcnow",
    "write_json",
]
