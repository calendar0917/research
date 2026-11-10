"""Formal 40-epoch runs of the LCRC-W3 regularization diagnostic.

Two runs (``W3_WD10``, ``W3_DO20``), one seed, 40 epochs each, from random
initialisation, with the historical W3 recipe and the historical seed-0 batch
order.  After every epoch the **full** Train, Validation and Test splits are
inferred and scored (official OGB macro AP, masked BCE, per-task AP), then the
model returns to train mode.

The primary checkpoint is the maximum Validation AP (ties -> earlier epoch).
Test scores are recorded as ``EXPOSED_TEST_EXPLORATORY`` and never influence
selection.

Artifacts per run directory: ``epoch_metrics.csv`` (epoch x split),
``per_task_ap.csv`` (epoch x split x 128 tasks), ``epoch_log.csv`` (training
side), ``checkpoint_best_valid.pt``, ``checkpoint_epoch40.pt``,
``checkpoint_last.pt`` (resumable), ``rng_state.pt``, ``training_order.json``,
``run.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import rd_common as C  # noqa: E402
import rd_data as D  # noqa: E402
import rd_model as M  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

EPOCH_FIELDS = [
    "epoch",
    "split",
    "rows",
    "ap_official",
    "ap_sklearn",
    "ap_abs_delta",
    "bce",
    "n_tasks_scored",
    "n_tasks_with_observed_labels",
    "observed_entries",
    "positive_entries",
    "infer_seconds",
    "metric_seconds",
    "eval_seconds",
    "global_step",
    "learning_rate",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best_epoch",
    "logits_sha256",
    "probs_sha256",
    "ap_source",
]
TASK_FIELDS = [
    "epoch",
    "split",
    "task",
    "ap",
    "observed",
    "positive",
    "negative",
    "train_positive",
    "valid_positive",
    "stratum",
]
LOG_FIELDS = [
    "epoch",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce",
    "epoch_train_seconds",
    "epoch_eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "peak_rss_kb",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best_epoch",
    "eval_state_preserved",
    "checkpoint_seconds",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "rng_combined_sha256",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


class Watchdog:
    def __init__(self, hard_seconds: float):
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


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    os.replace(tmp, path)
    return C.sha256_file(path)


def read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path.name} is missing columns {sorted(missing)}")
        return [dict(row) for row in reader]


def peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    sd = optimizer.state_dict()
    h = hashlib.sha256()
    for pid in sorted(sd["state"]):
        state = sd["state"][pid]
        for key in sorted(state):
            value = state[key]
            h.update(f"{pid}:{key}".encode())
            if torch.is_tensor(value):
                h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
            else:
                h.update(repr(value).encode())
    for group in sd["param_groups"]:
        payload = sorted((k, v) for k, v in group.items() if k != "params")
        h.update(repr(payload).encode())
    return h.hexdigest()


def gradient_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.grad is not None:
            h.update(name.encode())
            h.update(np.ascontiguousarray(p.grad.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def state_snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, str]:
    return {
        "params_sha256": C.state_fingerprint(M.BM.state_cpu(model)),
        "optimizer_sha256": optimizer_fingerprint(optimizer),
        "grads_sha256": gradient_fingerprint(model),
        "rng_combined_sha256": C.rng_state_hashes(C.capture_rng_state())["combined_sha256"],
    }


def plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, order_seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(order_seed),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode()),
    }


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(),
    }


# ---------------------------------------------------------------------------
# per-epoch pieces
# ---------------------------------------------------------------------------


def run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stager: PinnedStager,
    task: Any,
    plan: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    global_step: int,
    clip: float,
    device: torch.device,
    watchdog: Watchdog,
) -> dict[str, Any]:
    """One full epoch of real training.  Returns the epoch's training record."""
    model.train()
    t_epoch = time.perf_counter()
    batches = plan[epoch - 1]["batches"]
    losses: list[float] = []
    step_times: list[float] = []
    grad_norms: list[float] = []
    clip_events = 0
    for rows in batches:
        if watchdog.stop.is_set():
            return {"aborted": True, "epoch": int(epoch), "steps": len(losses)}
        t0 = time.perf_counter()
        batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
        y = stager.labels(task.labels[rows])  # Test rows are NaN in this artifact
        out = C.BT.train_step(model, optimizer, batch, y, global_step=global_step, clip=clip)
        global_step += 1
        losses.append(float(out["loss"]))
        grad_norms.append(float(out["grad_norm"]))
        clip_events += int(out["clipped"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(float(time.perf_counter() - t0))
    arr = np.asarray(step_times, dtype=np.float64)
    train_seconds = float(time.perf_counter() - t_epoch)
    return {
        "aborted": False,
        "epoch": int(epoch),
        "steps": int(len(losses)),
        "global_step_end": int(global_step),
        "train_bce": float(np.mean(losses)) if losses else float("nan"),
        "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
        "gradient_clip_events": int(clip_events),
        "train_seconds": train_seconds,
        "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan"),
    }


def evaluate_all_splits(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Any,
    splits: Mapping[str, D.SplitData],
    device: torch.device,
    *,
    epoch: int,
    global_step: int,
    learning_rate: float,
    batch_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Full Train, then Validation, then Test inference + metrics.

    The model, optimizer, gradient and RNG state are fingerprinted before and
    after the whole sequence; any drift is an error (and is recorded).
    """
    batch_sizes = dict(C.EVAL_BATCH_SIZE if batch_sizes is None else batch_sizes)
    before = state_snapshot(model, optimizer)
    model.eval()
    rows_by_split: dict[str, dict[str, Any]] = {}
    t_all = time.perf_counter()
    with torch.inference_mode():
        for split_name in C.SPLITS:  # train, valid, test -- exactly this order
            split = splits[split_name]
            rows_by_split[split_name] = D.evaluate_split(
                model,
                task,
                split,
                device,
                batch_size=int(batch_sizes[split_name]),
                reason=f"epoch_{epoch}_{split_name}_full_split",
            )
    eval_seconds = float(time.perf_counter() - t_all)
    after = state_snapshot(model, optimizer)
    integrity = {
        "epoch": int(epoch),
        "params_unchanged": before["params_sha256"] == after["params_sha256"],
        "optimizer_unchanged": before["optimizer_sha256"] == after["optimizer_sha256"],
        "grads_unchanged": before["grads_sha256"] == after["grads_sha256"],
        "rng_unchanged": before["rng_combined_sha256"] == after["rng_combined_sha256"],
        "before": before,
        "after": after,
        "eval_seconds": eval_seconds,
        "split_order": list(C.SPLITS),
    }
    integrity["passed"] = bool(
        integrity["params_unchanged"]
        and integrity["optimizer_unchanged"]
        and integrity["grads_unchanged"]
        and integrity["rng_unchanged"]
    )
    if not integrity["passed"]:
        raise RuntimeError(f"evaluation changed training state at epoch {epoch}: {integrity}")
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "learning_rate": float(learning_rate),
        "splits": rows_by_split,
        "integrity": integrity,
        "eval_seconds": eval_seconds,
    }


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def checkpoint_payload(
    *,
    run_key: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    optimizer_audit: Mapping[str, Any],
    scores: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "run": run_key,
        "model_key": C.RUNS[run_key]["model_key"],
        "seed": int(C.SEED),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "parameters": int(model_audit["parameters_recount"]["total"]),
        "state_dict": M.BM.state_cpu(model),
        "optimizer_state_dict": M.BM.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": scheduler_state(global_step, optimizer),
        "rng_state": dict(rng_state),
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "optimizer_audit": dict(optimizer_audit),
        "recipe": C.recipe_for_run(run_key),
        "scores": None if scores is None else dict(scores),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "host": {"hostname": os.uname().nodename},
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_evaluated": True,
        "test_policy": "EXPOSED_TEST_EXPLORATORY: Test scored every epoch, never used for selection",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def restore_state(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    run_key: str,
    plan: Sequence[Mapping[str, Any]],
    order_seed: int,
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("run") != run_key:
        raise RuntimeError(f"checkpoint run {payload.get('run')} != requested {run_key}")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    if int(payload["global_step"]) > 0 and steps != [float(payload["global_step"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != {payload['global_step']}")
    if float(optimizer.param_groups[0]["weight_decay"]) != float(C.RUNS[run_key]["weight_decay"]):
        raise RuntimeError("restored optimizer weight_decay does not match the run")
    hashes = M.BM.restore_rng_state(payload["rng_state"])
    if hashes["combined_sha256"] != (payload.get("rng_state_hashes") or {}).get("combined_sha256"):
        raise RuntimeError("restored RNG state does not match the checkpoint hashes")
    saved = payload.get("batch_plan") or {}
    now = plan_meta(plan, int(payload["epoch"]), order_seed)
    if now["global_plan_sha256"] != saved.get("global_plan_sha256") or now["prefix_sha256"] != saved.get(
        "prefix_sha256"
    ):
        raise RuntimeError("resumed plan does not match the checkpoint plan")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "rng_state_hashes": hashes,
        "plan_prefix_verified": True,
    }


# ---------------------------------------------------------------------------
# CSV recovery (resume must not lose completed rows, nor retrain epochs)
# ---------------------------------------------------------------------------


def load_completed(
    run_dir: Path, *, upto_epoch: int, strata: Sequence[str]
) -> dict[str, Any]:
    """Read the CSVs written so far and verify every completed epoch is intact."""
    metrics = read_csv(run_dir / "epoch_metrics.csv", EPOCH_FIELDS)
    tasks = read_csv(run_dir / "per_task_ap.csv", TASK_FIELDS)
    logs = read_csv(run_dir / "epoch_log.csv", LOG_FIELDS)
    kept_metrics = [r for r in metrics if int(float(r["epoch"])) <= int(upto_epoch)]
    kept_tasks = [r for r in tasks if int(float(r["epoch"])) <= int(upto_epoch)]
    kept_logs = [r for r in logs if int(float(r["epoch"])) <= int(upto_epoch)]
    for epoch in range(1, int(upto_epoch) + 1):
        present = {r["split"] for r in kept_metrics if int(float(r["epoch"])) == epoch}
        if present != set(C.SPLITS):
            raise RuntimeError(f"epoch {epoch} is missing splits {sorted(set(C.SPLITS) - present)}")
        n_tasks = sum(1 for r in kept_tasks if int(float(r["epoch"])) == epoch)
        if n_tasks != len(C.SPLITS) * C.N_TASKS:
            raise RuntimeError(f"epoch {epoch} has {n_tasks} per-task rows, expected {len(C.SPLITS) * C.N_TASKS}")
        if not any(int(float(r["epoch"])) == epoch for r in kept_logs):
            raise RuntimeError(f"epoch {epoch} has no epoch_log row")
    for row in kept_tasks:
        if row["stratum"] != strata[int(float(row["task"]))]:
            raise RuntimeError(f"per-task stratum drift at epoch {row['epoch']} task {row['task']}")
    valid_aps = [
        (int(float(r["epoch"])), float(r["ap_official"])) for r in kept_metrics if r["split"] == "valid"
    ]
    best_epoch = None
    best_ap = -math.inf
    for epoch, ap in sorted(valid_aps):
        if ap > best_ap:
            best_epoch, best_ap = epoch, ap
    return {
        "metrics": kept_metrics,
        "tasks": kept_tasks,
        "logs": kept_logs,
        "best_epoch": best_epoch,
        "best_valid_ap": None if best_epoch is None else best_ap,
        "epochs_complete": [e for e, _ in sorted(valid_aps)],
        "dropped_rows": {
            "metrics": len(metrics) - len(kept_metrics),
            "tasks": len(tasks) - len(kept_tasks),
            "logs": len(logs) - len(kept_logs),
        },
    }


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------


def train(
    task: Any,
    splits: Mapping[str, D.SplitData],
    *,
    run_key: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    epochs: int = C.TOTAL_EPOCHS,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float = 1e9,
    hard_seconds: float = 1e9,
    eval_batch_sizes: Mapping[str, int] | None = None,
    require_frozen_plan: bool = True,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_key not in C.RUNS:
        raise KeyError(run_key)
    run_cfg = C.RUNS[run_key]
    order_seed = int(seed)
    formal = int(epochs) == int(C.TOTAL_EPOCHS)
    gpu_start = C.gpu_occupancy()

    model, model_audit = M.build_model(run_key, task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device)
    optimizer, opt_audit = M.make_optimizer(model, weight_decay=float(run_cfg["weight_decay"]))

    plan = C.SD.make_plan40(task, seed=order_seed)
    if require_frozen_plan and plan[0]["global_plan_sha256"] != C.SHARED_RECIPE["batch_plan_sha256"]:
        raise RuntimeError(
            f"order plan {plan[0]['global_plan_sha256']} != frozen W3 plan "
            f"{C.SHARED_RECIPE['batch_plan_sha256']}"
        )
    plan_audit = C.SD.plan_summary(plan, task)
    if require_frozen_plan and int(len(plan[0]["batches"])) != C.STEPS_PER_EPOCH:
        raise RuntimeError(f"steps per epoch {len(plan[0]['batches'])} != {C.STEPS_PER_EPOCH}")

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    strata = [
        "support_lt5"
        if splits["valid"].positive[t] < C.DECISION["support_5_19_min_positive"]
        else "support_5_19"
        if splits["valid"].positive[t] < C.DECISION["support_ge20_min_positive"]
        else "support_ge20"
        for t in range(C.N_TASKS)
    ]
    if resume is not None:
        resume_audit = restore_state(
            resume, model, optimizer, run_key=run_key, plan=plan, order_seed=order_seed
        )
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        completed = load_completed(out_dir, upto_epoch=int(resume["epoch"]), strata=strata)
        metric_rows = list(completed["metrics"])
        task_rows = list(completed["tasks"])
        log_rows = list(completed["logs"])
        best_epoch = completed["best_epoch"]
        best_ap = completed["best_valid_ap"]
    else:
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
        completed = None
        metric_rows = []
        task_rows = []
        log_rows = []
        best_epoch = None
        best_ap = None
    rng_start_hashes = dict(rng_hashes)
    if best_ap is None:
        best_ap = -math.inf

    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": plan_meta(plan, int(epochs), order_seed), **plan_audit})

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.BASE_RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    metrics_path = out_dir / "epoch_metrics.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    log_path = out_dir / "epoch_log.csv"
    best_path = out_dir / "checkpoint_best_valid.pt"
    final_path = out_dir / f"checkpoint_epoch{int(epochs)}.pt"
    last_path = out_dir / "checkpoint_last.pt"
    rng_path = out_dir / "rng_state.pt"

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    best_checkpoint_sha = ""
    final_checkpoint_sha = ""
    last_checkpoint_sha = ""
    epochs_run = int(start_epoch) - 1
    clip_events_total = 0
    stop_reason = "RUNNING"
    integrity_all_passed = True
    eval_epochs_completed: list[int] = list(completed["epochs_complete"]) if completed else []
    rng_epoch_hashes: list[dict[str, Any]] = []

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    for epoch in range(int(start_epoch), int(epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        log_row_before = len(log_rows)
        record = run_epoch(
            model,
            optimizer,
            stager,
            task,
            plan,
            epoch=epoch,
            global_step=global_steps_run,
            clip=clip,
            device=device,
            watchdog=watchdog,
        )
        if record["aborted"]:
            stop_reason = "HARD_WALL_LIMIT"
            del log_rows[log_row_before:]
            C.log(f"[{run_key}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        global_steps_run = int(record["global_step_end"])
        clip_events_total += int(record["gradient_clip_events"])

        lr_now = float(optimizer.param_groups[0]["lr"])
        t_eval = time.perf_counter()
        evaluation = evaluate_all_splits(
            model,
            optimizer,
            task,
            splits,
            device,
            epoch=epoch,
            global_step=global_steps_run,
            learning_rate=lr_now,
            batch_sizes=eval_batch_sizes,
        )
        eval_seconds = float(time.perf_counter() - t_eval)
        integrity_all_passed = bool(integrity_all_passed and evaluation["integrity"]["passed"])
        eval_epochs_completed.append(int(epoch))

        valid_ap = float(evaluation["splits"]["valid"]["ap"])
        is_best = bool(valid_ap > float(best_ap))
        if is_best:
            best_epoch = int(epoch)
            best_ap = valid_ap
        for split_name in C.SPLITS:
            rep = evaluation["splits"][split_name]
            ap_sklearn = float(
                np.mean([x for x in rep["per_task_ap"] if x is not None and np.isfinite(x)])
            )
            metric_rows.append(
                {
                    "epoch": int(epoch),
                    "split": split_name,
                    "rows": int(rep["rows"]),
                    "ap_official": float(rep["ap"]),
                    "ap_sklearn": ap_sklearn,
                    "ap_abs_delta": abs(float(rep["ap"]) - ap_sklearn),
                    "bce": float(rep["bce"]),
                    "n_tasks_scored": int(rep["n_tasks_scored"]),
                    "n_tasks_with_observed_labels": int(rep["n_tasks_with_observed_labels"]),
                    "observed_entries": int(rep["observed_entries"]),
                    "positive_entries": int(rep["positive_entries"]),
                    "infer_seconds": float(rep["infer_seconds"]),
                    "metric_seconds": float(rep["metric_seconds"]),
                    "eval_seconds": float(rep["seconds"]),
                    "global_step": int(global_steps_run),
                    "learning_rate": lr_now,
                    "best_epoch_so_far": int(best_epoch),
                    "best_valid_ap_so_far": float(best_ap),
                    "is_best_epoch": is_best,
                    "logits_sha256": rep["logits_sha256"],
                    "probs_sha256": rep["probs_sha256"],
                    "ap_source": rep["ap_source"],
                }
            )
            for t in range(C.N_TASKS):
                ap = rep["per_task_ap"][t]
                task_rows.append(
                    {
                        "epoch": int(epoch),
                        "split": split_name,
                        "task": t,
                        "ap": "" if ap is None else float(ap),
                        "observed": int(rep["observed"][t]),
                        "positive": int(rep["positive"][t]),
                        "negative": int(rep["negative"][t]),
                        "train_positive": int(splits["train"].positive[t]),
                        "valid_positive": int(splits["valid"].positive[t]),
                        "stratum": strata[t],
                    }
                )

        update_peak()
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        cumulative = float(time.perf_counter() - t_start)
        log_rows.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_steps_run),
                "steps_per_epoch": int(record["steps"]),
                "learning_rate": lr_now,
                "train_bce": float(record["train_bce"]),
                "epoch_train_seconds": float(record["train_seconds"]),
                "epoch_eval_seconds": eval_seconds,
                "epoch_seconds": float(record["train_seconds"]) + eval_seconds,
                "cumulative_seconds": cumulative,
                "step_ms_mean": float(record["step_ms_mean"]),
                "step_ms_p50": float(record["step_ms_p50"]),
                "step_ms_p90": float(record["step_ms_p90"]),
                "steps_per_second": float(record["steps_per_second"]),
                "grad_norm_max": record["grad_norm_max"],
                "grad_norm_mean": record["grad_norm_mean"],
                "gradient_clip_events": int(record["gradient_clip_events"]),
                "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
                "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
                "peak_rss_kb": peak_rss_kb(),
                "best_epoch_so_far": int(best_epoch),
                "best_valid_ap_so_far": float(best_ap),
                "is_best_epoch": is_best,
                "eval_state_preserved": bool(evaluation["integrity"]["passed"]),
                "checkpoint_seconds": None,
                "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
                "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
                "rng_combined_sha256": rng_hashes["combined_sha256"],
            }
        )

        C.log(
            f"[{run_key}] epoch {epoch}/{int(epochs)} train_bce {record['train_bce']:.6f} "
            f"AP train {evaluation['splits']['train']['ap']:.6f} "
            f"valid {valid_ap:.6f} test {evaluation['splits']['test']['ap']:.6f} "
            f"lr {lr_now:.3e} train {record['train_seconds']:.1f}s eval {eval_seconds:.1f}s "
            f"step_p50 {record['step_ms_p50']:.2f}ms{' *best*' if is_best else ''}"
        )

        scores = {
            "epoch": int(epoch),
            "train_ap": float(evaluation["splits"]["train"]["ap"]),
            "valid_ap": valid_ap,
            "test_ap": float(evaluation["splits"]["test"]["ap"]),
            "train_bce": float(record["train_bce"]),
            "valid_bce": float(evaluation["splits"]["valid"]["bce"]),
            "test_bce": float(evaluation["splits"]["test"]["bce"]),
        }
        if is_best:
            payload = checkpoint_payload(
                run_key=run_key,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                optimizer_audit=opt_audit,
                scores=scores,
                kind="best_valid_ap",
                extra={"selection": "maximum Validation AP, ties -> earlier epoch", "diagnostic_only": False},
            )
            best_checkpoint_sha = atomic_save(payload, best_path)
        if epoch == int(epochs):
            payload = checkpoint_payload(
                run_key=run_key,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                optimizer_audit=opt_audit,
                scores=scores,
                kind=f"epoch{int(epochs)}_final",
                extra={"resume_audit": resume_audit, "status_at_save": "COMPLETED"},
            )
            final_checkpoint_sha = atomic_save(payload, final_path)

        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            run_key=run_key,
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            plan=plan_meta(plan, int(epoch), order_seed),
            model_audit=model_audit,
            optimizer_audit=opt_audit,
            scores=None,
            kind="last_epoch_resumable",
            extra={"status_at_save": "RUNNING_OR_COMPLETED"},
        )
        last_checkpoint_sha = atomic_save(payload_last, last_path)
        log_rows[-1]["checkpoint_seconds"] = float(time.perf_counter() - t_ck)

        write_csv(metrics_path, EPOCH_FIELDS, metric_rows)
        write_csv(per_task_path, TASK_FIELDS, task_rows)
        write_csv(log_path, LOG_FIELDS, log_rows)
        epochs_run = int(epoch)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    metrics_sha = write_csv(metrics_path, EPOCH_FIELDS, metric_rows)
    per_task_sha = write_csv(per_task_path, TASK_FIELDS, task_rows)
    log_sha = write_csv(log_path, LOG_FIELDS, log_rows)

    expected_eval = set(range(1, int(epochs) + 1))
    completed_ok = (
        epochs_run >= int(epochs)
        and set(eval_epochs_completed) == expected_eval
        and bool(final_checkpoint_sha)
        and integrity_all_passed
    )
    status = "COMPLETED" if completed_ok else "INCOMPLETE"
    if completed_ok:
        stop_reason = "COMPLETED"
    elif not integrity_all_passed:
        stop_reason = "EVALUATION_STATE_DRIFT"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end_hashes = C.rng_state_hashes(C.capture_rng_state())
    torch.save(
        {
            "schema": "lcrc-pcba-w3-regularization-rng-v1",
            "round_id": C.ROUND_ID,
            "run": run_key,
            "seed": int(seed),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    total_seconds = float(time.perf_counter() - t_start)
    train_seconds = float(sum(float(r["epoch_train_seconds"]) for r in log_rows))
    eval_seconds_total = float(sum(float(r["epoch_eval_seconds"]) for r in log_rows))
    split_seconds = {
        s: {
            "infer_seconds": float(sum(float(r["infer_seconds"]) for r in metric_rows if r["split"] == s)),
            "metric_seconds": float(sum(float(r["metric_seconds"]) for r in metric_rows if r["split"] == s)),
            "eval_seconds": float(sum(float(r["eval_seconds"]) for r in metric_rows if r["split"] == s)),
        }
        for s in C.SPLITS
    }
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run_key,
        "role": run_cfg["role"],
        "model_key": run_cfg["model_key"],
        "scale": int(run_cfg["scale"]),
        "seed": int(seed),
        "order_seed": int(order_seed),
        "formal_run": bool(formal),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters_recount"]["total"]),
        "base_parameters": int(model_audit["parameters_recount"]["trunk"]),
        "readout_parameters": int(model_audit["parameters_recount"]["readout"]),
        "dropout": float(model.drop.p),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "epochs_requested": int(epochs),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_completed": sorted(eval_epochs_completed),
        "all_three_splits_every_epoch": bool(set(eval_epochs_completed) == expected_eval),
        "eval_state_integrity_all_epochs": bool(integrity_all_passed),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_ap),
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "final_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "final_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "epoch_metrics": C.track_rel(metrics_path),
        "epoch_metrics_sha256": metrics_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "epoch_log": C.track_rel(log_path),
        "epoch_log_sha256": log_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "resume_audit": resume_audit,
        "checkpoint_selection": "maximum Validation AP, ties resolved to the earlier epoch",
        "test_policy": "EXPOSED_TEST_EXPLORATORY: Test scored every epoch, never used for selection",
        "test_evaluated_every_epoch": True,
        "eval_batch_sizes": dict(C.EVAL_BATCH_SIZE if eval_batch_sizes is None else eval_batch_sizes),
        "model_audit": model_audit,
        "optimizer_audit": opt_audit,
        "batch_plan": plan_meta(plan, int(epochs_run), order_seed),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": C.sha256_file(order_path),
        "recipe": C.recipe_for_run(run_key),
        "data": {
            "feature_spec_hash": task.spec.hash(),
            "norm_fingerprint": task.norm["norm_fingerprint"],
            "split_partition_sha256": task.split_partition_sha256,
            "label_artifact_sha256": task.label_artifact_sha256,
            "label_artifact": C.track_rel(task.labels_path),
            "cache": task.cache_identity,
            "test_labels_source": "official OGB raw/graph-label.csv.gz",
            "test_labels_enter_training": False,
        },
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "precision": "fp32",
        "timing": {
            "total_seconds": total_seconds,
            "train_seconds": train_seconds,
            "eval_seconds_total": eval_seconds_total,
            "per_split": split_seconds,
            "gpu_hours": total_seconds / 3600.0,
            "epoch_seconds_mean": float(np.mean([float(r["epoch_seconds"]) for r in log_rows])) if log_rows else 0.0,
            "steps_per_second_mean": float(np.mean([float(r["steps_per_second"]) for r in log_rows]))
            if log_rows
            else 0.0,
        },
        "resources": {
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "peak_rss_kb": peak_rss_kb(),
            "soft_seconds": float(soft_seconds),
            "hard_seconds": float(hard_seconds),
            "watchdog_fired_at": watchdog.fired_at,
        },
        "scores_by_epoch": {
            str(int(r["epoch"])): {
                s: {
                    "ap": float(next(x["ap_official"] for x in metric_rows if int(x["epoch"]) == int(r["epoch"]) and x["split"] == s)),
                }
                for s in C.SPLITS
            }
            for r in log_rows
        },
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "module_sha256": C.module_inventory()["sha256"],
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done run={run_key} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_valid_ap={best_ap} gpu_hours={run['timing']['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(C.RUNS))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=C.TOTAL_EPOCHS)
    parser.add_argument("--resume", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "run": args.run,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        task = D.load_task()
        test = D.read_official_test_labels(task)
        splits = D.build_split_data(task, test["labels"])
        C.write_json(
            out_dir / "test_label_audit.json",
            {"schema": "lcrc-pcba-w3-regularization-test-labels-v1", **test["audit"], "captured_utc": C.utcnow()},
        )
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        run = train(
            task,
            splits,
            run_key=args.run,
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            epochs=int(args.epochs),
            resume=resume,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "run": args.run,
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED run={args.run}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
