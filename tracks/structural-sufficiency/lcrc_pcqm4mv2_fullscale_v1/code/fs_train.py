"""Formal full-Train training for the fullscale round.

One recipe, two widths (W2 = 939,079 / W3 = 2,038,167 parameters), 40 epochs on
the official 3,378,606-molecule Train split, batch 128, seed 0, FP32, E1
vectorised collation, full official Validation + 20k Train-monitor after every
epoch.  Checkpoints are epoch-boundary resumable (model, AdamW, scheduler,
Python/NumPy/Torch/CUDA RNG state) and the batch-order hashes are replayed on
resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fs_common as C  # noqa: E402
import fs_data as D  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

import torch  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_l1_z",
    "train_l1_z_molecule_weighted",
    "valid_mae_eV",
    "valid_mae_eV_numpy",
    "valid_mae_crosscheck_delta_eV",
    "valid_pred_mean_eV",
    "valid_pred_std_eV",
    "valid_n",
    "monitor_mae_eV",
    "monitor_n",
    "epoch_train_seconds",
    "epoch_valid_seconds",
    "epoch_monitor_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "train_molecules_per_second",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "peak_rss_mb",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "best_epoch_so_far",
    "best_valid_mae_eV_so_far",
    "is_best",
    "checkpoint_seconds",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "implementation",
]

CURVE_STATE = "curve_state.json"


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def capture_rng() -> dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [s.detach().cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda = state.get("torch_cuda")
    if cuda:
        torch.cuda.set_rng_state_all([s.cuda() if torch.is_tensor(s) else s for s in cuda])


def rng_hashes() -> dict[str, Any]:
    return C.rng_state_hashes(capture_rng())


# ---------------------------------------------------------------------------
# watch dog / IO
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


def atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# optimizer / checkpoint payload
# ---------------------------------------------------------------------------


def make_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "end_lr": float(C.RECIPE["end_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": (
            "per-step cosine with 1-epoch linear warmup, horizon = "
            f"{C.TOTAL_STEPS:,} steps (40 x {C.STEPS_PER_EPOCH:,})"
        ),
    }
    return optimizer, audit


def batch_plan(task: Mapping[str, Any], consumed: int) -> dict[str, Any]:
    orders = task["orders"]
    prefix = orders["epoch_permutation_sha256"][: int(consumed)]
    return {
        "order_seed": int(orders["order_seed"]),
        "n_rows": int(orders["n_rows"]),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(orders["epochs"]),
        "consumed_epochs": int(consumed),
        "combined_sha256": orders["combined_sha256"],
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode("utf-8")),
    }


def checkpoint_payload(
    *,
    task: Mapping[str, Any],
    run: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_audit: Mapping[str, Any],
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = capture_rng()
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": task["model"],
        "model_name": task["model_name"],
        "scale": int(task["scale"]),
        "seed": int(task["seed"]),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_rows": int(task["n_rows"]),
        "n_outputs": int(C.N_OUT),
        "parameters": int(model_audit["total_parameters"]),
        "state_dict": C.EW_MODEL.state_cpu(model),
        "optimizer_state_dict": C.EW_MODEL.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": {
            "kind": "scratch_warmup_cosine_fullscale",
            "global_step": int(global_step),
            "steps_per_epoch": int(task["steps_epoch"]),
            "steps_total": int(task["steps_total"]),
            "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        },
        "rng_state": state,
        "rng_state_hashes": C.rng_state_hashes(state),
        "batch_plan": batch_plan(task, int(epoch)),
        "model_audit": dict(model_audit),
        "feature_spec_hash": task["spec"].hash(),
        "feature_norm": {
            "fingerprint": task["feature_norm"]["norm_fingerprint"],
            "source": task["feature_norm"]["source"],
        },
        "label_norm": dict(task["label_norm"]),
        "cache": dict(task["cache_manifest"]),
        "rows_sha256": task["row_plan"]["rows_sha256"],
        "valid_mae_eV": None if valid_report is None else float(valid_report["mae_eV"]),
        "valid_report_summary": None
        if valid_report is None
        else {
            "mae_eV": float(valid_report["mae_eV"]),
            "mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "n": int(valid_report["n"]),
            "pred_sha256": valid_report["pred_sha256"],
            "pred_z_sha256": valid_report["pred_z_sha256"],
        },
        "recipe": C.recipe_for_run(run),
        "implementation": {"name": C.FORMAL_IMPLEMENTATION, "collate": "E1 vectorised", "fp32": True, "tf32": False},
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official test-dev / test-challenge labels are hidden and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _rng_hash_diffs(saved: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    diffs = [
        key
        for key in ("python_sha256", "numpy_sha256", "torch_cpu_sha256")
        if saved.get(key) != now.get(key)
    ]
    a = list(saved.get("torch_cuda_sha256") or [])
    b = list(now.get("torch_cuda_sha256") or [])
    if a[: min(len(a), len(b))] != b[: min(len(a), len(b))]:
        diffs.append("torch_cuda_sha256")
    return diffs


def restore(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    run: str,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("run") != run:
        raise RuntimeError(f"checkpoint run {payload.get('run')} != requested {run}")
    if int(payload.get("scale", -1)) != int(task["scale"]):
        raise RuntimeError("checkpoint scale differs from the requested run")
    if int(payload.get("n_rows", -1)) != int(task["n_rows"]):
        raise RuntimeError("checkpoint was trained on a different number of rows")
    if payload.get("rows_sha256") != task["row_plan"]["rows_sha256"]:
        raise RuntimeError("checkpoint row plan differs from the current full-Train plan")
    if payload.get("feature_spec_hash") != task["spec"].hash():
        raise RuntimeError("checkpoint feature spec differs")
    if (payload.get("label_norm") or {}).get("mean_eV") != task["label_norm"]["mean_eV"]:
        raise RuntimeError("checkpoint label normalisation differs")
    if (payload.get("cache") or {}).get("full_cache", {}).get("cache_sha256") != task["cache_manifest"][
        "full_cache"
    ]["cache_sha256"]:
        raise RuntimeError("checkpoint full-cache hash differs")

    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    if int(payload["global_step"]) > 0 and steps != [float(payload["global_step"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != checkpoint global_step {payload['global_step']}")
    restore_rng(payload["rng_state"])
    now = rng_hashes()
    diffs = _rng_hash_diffs(payload.get("rng_state_hashes") or {}, now)
    if diffs:
        raise RuntimeError(f"restored RNG state does not match checkpoint hashes: {diffs}")
    saved = payload.get("batch_plan") or {}
    want = batch_plan(task, int(payload["epoch"]))
    if saved.get("prefix_sha256") != want["prefix_sha256"] or int(saved.get("order_seed", -1)) != int(
        want["order_seed"]
    ):
        raise RuntimeError("resumed batch plan does not match the checkpoint")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "rng_state_hashes": now,
        "resumed_best_epoch": payload.get("best_epoch_so_far"),
        "resumed_best_mae": payload.get("best_valid_mae_so_far"),
        "checkpoint_implementation": payload.get("implementation"),
    }


def replay_epoch_orders(task: Mapping[str, Any], upto: int) -> list[str]:
    rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    seen: list[str] = []
    for epoch in range(1, int(upto) + 1):
        order = rng.permutation(int(task["n_rows"])).astype(np.int64)
        digest = C.sha256_bytes(order.tobytes())
        expected = task["orders"]["epoch_permutation_sha256"][epoch - 1]
        if digest != expected:
            raise RuntimeError(f"epoch {epoch} order hash {digest} != planned {expected}")
        seen.append(digest)
    return seen


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    task: Mapping[str, Any],
    *,
    run: str,
    epochs: int,
    out_dir: Path,
    device: torch.device,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
    eval_batch_size: int = 512,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    if int(epochs) != int(C.RUNS_TABLE[run]["epochs"]) and limit_batches is None:
        raise RuntimeError(f"formal run must use {C.RUNS_TABLE[run]['epochs']} epochs, got {epochs}")
    formal = bool(limit_batches is None and int(epochs) == int(C.RUNS_TABLE[run]["epochs"]))
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = int(task["n_rows"])
    steps_epoch = int(task["steps_epoch"]) if limit_batches is None else int(limit_batches)
    steps_total = int(task["steps_total"])

    gpu_start = C.gpu_occupancy()
    deterministic = C.apply_deterministic_settings()
    seed_info = C.seed_everything(int(task["seed"]))
    model, model_audit = C.EW_MODEL.build_model(
        task["spec"], task["norm"], scale=int(task["scale"]), seed=int(task["seed"]), device=device
    )
    if int(model_audit["total_parameters"]) != int(task["expected_parameters"]):
        raise RuntimeError(
            f"built {model_audit['total_parameters']} parameters, expected {task['expected_parameters']}"
        )
    optimizer, opt_audit = make_optimizer(model)

    resume_audit: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer, run=run, task=task)
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        best_epoch = resume.get("best_epoch_so_far")
        best_mae = float(resume.get("best_valid_mae_so_far") or math.inf)
        best_checkpoint_sha = resume.get("best_checkpoint_sha256") or ""
        curve_rows = list(resume.get("curve_rows") or [])
        verified_orders = replay_epoch_orders(task, start_epoch - 1)
        if resume_audit["resumed_best_epoch"] is not None:
            best_epoch = int(resume_audit["resumed_best_epoch"])
            best_mae = float(resume_audit["resumed_best_mae"])
    else:
        start_epoch = 1
        global_steps_run = 0
        best_epoch = None
        best_mae = math.inf
        best_checkpoint_sha = ""
        curve_rows = []
        verified_orders = []
    rng_start_hashes = rng_hashes()

    order_rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    for _ in range(1, start_epoch):
        _ = order_rng.permutation(n_rows)

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_path = out_dir / "train_curve.csv"
    curve_state_path = out_dir / CURVE_STATE
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best_valid.pt"
    final_path = out_dir / f"checkpoint_epoch{int(epochs)}.pt"
    rng_path = out_dir / "rng_state.pt"

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    final_checkpoint_sha = ""
    last_checkpoint_sha = ""
    eval_completed: list[int] = [int(r["epoch"]) for r in curve_rows if r.get("evaluated")]
    fixed_reports: dict[int, dict[str, Any]] = {
        int(r["epoch"]): {
            "mae_eV": float(r["valid_mae_eV"]),
            "mae_eV_numpy": float(r["valid_mae_eV_numpy"]),
            "crosscheck_delta_eV": float(r["valid_mae_crosscheck_delta_eV"]),
            "n": int(r["valid_n"]),
            "pred_mean_eV": float(r["valid_pred_mean_eV"]),
            "pred_std_eV": float(r["valid_pred_std_eV"]),
            "step": int(r["global_step"]),
        }
        for r in curve_rows
        if r.get("evaluated")
    }
    monitor_reports: dict[int, dict[str, Any]] = {
        int(r["epoch"]): {
            "mae_eV": float(r["monitor_mae_eV"]),
            "n": int(r["monitor_n"]),
        }
        for r in curve_rows
        if r.get("evaluated")
    }
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    clip_events_total = 0
    total_steps_measured = 0
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
        t_epoch = time.perf_counter()
        order = order_rng.permutation(n_rows).astype(np.int64)
        digest = C.sha256_bytes(order.tobytes())
        if digest != task["orders"]["epoch_permutation_sha256"][epoch - 1]:
            raise RuntimeError(f"epoch {epoch} order hash {digest} != planned order")
        verified_orders.append(digest)
        model.train()
        losses: list[float] = []
        weighted: list[tuple[float, int]] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        n_batches = 0
        molecules_epoch = 0
        for i in range(0, order.size, int(C.BATCH_SIZE)):
            if watchdog.stop.is_set():
                aborted = True
                break
            if limit_batches is not None and n_batches >= int(limit_batches):
                break
            rows = order[i : i + int(C.BATCH_SIZE)]
            t0 = time.perf_counter()
            raw = C.EW_COLLATE.collate_v2(task["train_cache"], rows.tolist())
            batch = stager.batch(raw)
            y = stager.labels(np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32))
            out = C.PQ_TRAIN.train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=global_steps_run,
                steps_epoch=steps_epoch,
                steps_total=steps_total,
                clip=clip,
            )
            global_steps_run += 1
            n_batches += 1
            molecules_epoch += int(rows.size)
            losses.append(out["loss"])
            weighted.append((out["loss"], int(out["n"])))
            grad_norms.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
            if n_batches % 2000 == 0:
                C.log(
                    f"[{run}] epoch {epoch} step {n_batches}/{steps_epoch} "
                    f"loss(z) {np.mean(losses[-2000:]):.5f} "
                    f"{1e3 * np.mean(step_times[-2000:]):.1f} ms/step"
                )
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{run}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        if limit_batches is None and n_batches != steps_epoch:
            raise RuntimeError(f"epoch {epoch} ran {n_batches} steps, expected {steps_epoch}")
        if not losses:
            raise RuntimeError(f"epoch {epoch} produced no loss values")
        total_steps_measured += int(n_batches)

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        t_eval = time.perf_counter()
        valid_report, _ = C.PQ_TRAIN.evaluate_split(
            model, task, device, which="valid", batch_size=eval_batch_size
        )
        valid_seconds = float(time.perf_counter() - t_eval)
        t_mon = time.perf_counter()
        monitor_report, _ = C.PQ_TRAIN.evaluate_split(
            model, task, device, which="monitor", batch_size=eval_batch_size
        )
        monitor_seconds = float(time.perf_counter() - t_mon)
        strict = best_epoch is None or float(valid_report["mae_eV"]) < float(best_mae)
        if strict:
            best_epoch = int(epoch)
            best_mae = float(valid_report["mae_eV"])
            payload = checkpoint_payload(
                task=task,
                run=run,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                model_audit=model_audit,
                valid_report=valid_report,
                kind="best_valid_mae",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "best_epoch_so_far": int(best_epoch),
                    "best_valid_mae_so_far": float(best_mae),
                },
            )
            best_checkpoint_sha = atomic_save(payload, best_path)
        eval_completed.append(int(epoch))
        fixed_reports[int(epoch)] = {
            "mae_eV": float(valid_report["mae_eV"]),
            "mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "crosscheck_delta_eV": float(valid_report["crosscheck_delta_eV"]),
            "n": int(valid_report["n"]),
            "pred_mean_eV": float(valid_report["pred_mean_eV"]),
            "pred_std_eV": float(valid_report["pred_std_eV"]),
            "pred_sha256": valid_report["pred_sha256"],
            "pred_z_sha256": valid_report["pred_z_sha256"],
            "inference_seconds": float(valid_report["inference_seconds"]),
            "step": int(global_steps_run),
        }
        monitor_reports[int(epoch)] = {
            "mae_eV": float(monitor_report["mae_eV"]),
            "mae_eV_numpy": float(monitor_report["mae_eV_numpy"]),
            "n": int(monitor_report["n"]),
            "pred_mean_eV": float(monitor_report["pred_mean_eV"]),
            "pred_std_eV": float(monitor_report["pred_std_eV"]),
            "pred_sha256": monitor_report["pred_sha256"],
            "inference_seconds": float(monitor_report["inference_seconds"]),
        }
        model.train()
        hashes = rng_hashes()
        rng_epoch_hashes.append({"epoch": int(epoch), **hashes})
        update_peak()
        epochs_run = int(epoch)
        cumulative = float(time.perf_counter() - t_start)
        mean_loss = float(np.mean(losses))
        total_n = sum(n for _loss, n in weighted)
        weighted_loss = float(sum(loss * n for loss, n in weighted) / total_n) if total_n else float("nan")
        row = {
            "epoch": int(epoch),
            "evaluated": True,
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(n_batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_l1_z": mean_loss,
            "train_l1_z_molecule_weighted": weighted_loss,
            "valid_mae_eV": float(valid_report["mae_eV"]),
            "valid_mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "valid_mae_crosscheck_delta_eV": float(valid_report["crosscheck_delta_eV"]),
            "valid_pred_mean_eV": float(valid_report["pred_mean_eV"]),
            "valid_pred_std_eV": float(valid_report["pred_std_eV"]),
            "valid_n": int(valid_report["n"]),
            "monitor_mae_eV": float(monitor_report["mae_eV"]),
            "monitor_n": int(monitor_report["n"]),
            "epoch_train_seconds": train_seconds,
            "epoch_valid_seconds": valid_seconds,
            "epoch_monitor_seconds": monitor_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "train_molecules_per_second": float(molecules_epoch / train_seconds) if train_seconds > 0 else float("nan"),
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
            "steps_per_second": float(n_batches / train_seconds) if train_seconds > 0 else float("nan"),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "peak_rss_mb": C.peak_rss_mb(),
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": int(best_epoch),
            "best_valid_mae_eV_so_far": float(best_mae),
            "is_best": bool(strict),
            "checkpoint_seconds": None,
            "rng_cpu_sha256": hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(hashes["torch_cuda_sha256"]),
            "implementation": C.FORMAL_IMPLEMENTATION,
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        C.log(
            f"[{run}] epoch {epoch}/{int(epochs)} trainL1(z) {mean_loss:.5f} "
            f"validMAE {row['valid_mae_eV']:.6f} eV monitorMAE {row['monitor_mae_eV']:.6f} eV "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.0f}s "
            f"({row['train_molecules_per_second']:.0f} mol/s) step_p50 {row['step_ms_p50']:.2f}ms "
            f"{'*best*' if strict else ''}"
        )
        if epoch == int(epochs):
            payload = checkpoint_payload(
                task=task,
                run=run,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                model_audit=model_audit,
                valid_report=valid_report,
                kind=f"epoch{int(epochs)}_primary",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "status_at_save": "COMPLETED",
                    "formal_run": bool(formal),
                    "best_epoch_so_far": int(best_epoch),
                    "best_valid_mae_so_far": float(best_mae),
                },
            )
            final_checkpoint_sha = atomic_save(payload, final_path)
        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            task=task,
            run=run,
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            model_audit=model_audit,
            valid_report=valid_report,
            kind="last_epoch_resumable",
            extra={
                "optimizer_state_audit": opt_audit,
                "verified_epoch_orders": len(verified_orders),
                "best_epoch_so_far": int(best_epoch),
                "best_valid_mae_so_far": float(best_mae),
                "best_checkpoint_sha256": best_checkpoint_sha,
                "curve_rows": curve_rows,
            },
        )
        last_checkpoint_sha = atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        with open(curve_state_path, "w", encoding="utf-8") as fh:
            json.dump(C.json_safe(curve_rows), fh)
        update_peak()
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    evaluated_through = max(eval_completed) if eval_completed else 0
    completed = epochs_run >= int(epochs) and evaluated_through >= int(epochs) and bool(final_checkpoint_sha)
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"

    rng_end_hashes = rng_hashes()
    torch.save(
        {
            "schema": "lcrc-pcqm4mv2-fullscale-rng-v1",
            "round_id": C.ROUND_ID,
            "run": run,
            "seed": int(task["seed"]),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)
    curve_sha = write_csv(curve_path, CURVE_FIELDS, curve_rows)
    total_seconds = float(time.perf_counter() - t_start)
    train_seconds_total = float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows))
    molecules_seen = int(sum(int(r["steps_per_epoch"]) for r in curve_rows) * int(C.BATCH_SIZE))
    run_payload = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": task["model"],
        "model_name": task["model_name"],
        "scale": int(task["scale"]),
        "seed": int(task["seed"]),
        "formal_run": bool(formal),
        "implementation": {
            "name": C.FORMAL_IMPLEMENTATION,
            "collate": "E1 vectorised (read-only inherited ew_collate.collate_v2)",
            "deferred_sync": False,
            "prefetch": False,
            "note": "frozen per-step synchronisation, frozen loss/clip/optimizer and batch order",
        },
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_OUT),
        "train_size": int(n_rows),
        "train_cache_rows": int(task["train_cache"]["n_molecules"]),
        "parameters": int(model_audit["total_parameters"]),
        "base_parameters": int(model_audit["base_parameters"]),
        "readout_parameters": int(model_audit["readout_parameters"]),
        "epochs": int(epochs),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "steps_per_epoch": int(steps_epoch),
        "steps_total": int(steps_total),
        "limit_batches": None if limit_batches is None else int(limit_batches),
        "eval_epochs_completed": sorted(int(x) for x in eval_completed),
        "evaluated_through_epoch": int(evaluated_through),
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "monitor_epoch_reports": {str(k): v for k, v in monitor_reports.items()},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "batch_plan": batch_plan(task, max(int(epochs_run), 0)),
        "epoch_order_hashes_verified": len(verified_orders),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_mae_eV": None if best_epoch is None else float(best_mae),
        "final_epoch_valid_mae_eV": None
        if int(epochs) not in fixed_reports
        else float(fixed_reports[int(epochs)]["mae_eV"]),
        "final_epoch_monitor_mae_eV": None
        if int(epochs) not in monitor_reports
        else float(monitor_reports[int(epochs)]["mae_eV"]),
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "primary_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "primary_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "spec_hash": task["spec"].hash(),
        "feature_norm": task["feature_norm"],
        "label_norm": task["label_norm"],
        "cache_manifest": task["cache_manifest"],
        "row_plan": task["row_plan"],
        "split_identity": {
            "split_dict_sha256": task["split_identity"]["split_dict"]["sha256"],
            "splits": task["split_identity"]["splits"],
        },
        "monitor_plan": task["monitor_plan"],
        "data_isolation": task["data_isolation"],
        "recipe": C.recipe_for_run(run),
        "model_dropout": float(model.drop.p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": train_seconds_total,
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "monitor_seconds": float(sum(float(r.get("epoch_monitor_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows])) if curve_rows else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows]))
        if curve_rows
        else 0.0,
        "throughput": {
            "measured_steps": int(total_steps_measured),
            "train_molecules": int(molecules_seen),
            "train_seconds": train_seconds_total,
            "molecules_per_second_train": float(molecules_seen / train_seconds_total)
            if train_seconds_total > 0
            else None,
            "molecules_per_second_total_wall": float(molecules_seen / total_seconds) if total_seconds > 0 else None,
            "step_ms_mean_all_epochs": float(np.mean([float(r["step_ms_mean"]) for r in curve_rows]))
            if curve_rows
            else None,
            "step_ms_p50_all_epochs": float(np.mean([float(r["step_ms_p50"]) for r in curve_rows]))
            if curve_rows
            else None,
            "step_timing_mode": "per-step (frozen)",
        },
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "peak_rss_mb": C.peak_rss_mb(),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "deterministic_settings": deterministic,
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official test-dev / test-challenge labels are hidden and are never indexed",
    }
    C.write_json(out_dir / "run.json", C.json_safe(run_payload))
    C.log(
        f"done run={run} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_valid_mae={best_mae:.6f} eV "
        f"gpu_hours={run_payload['gpu_hours']:.4f} "
        f"mol/s={run_payload['throughput']['molecules_per_second_train']:.1f}"
    )
    return run_payload


# ---------------------------------------------------------------------------
# throughput benchmark (preflight only; never writes a formal run)
# ---------------------------------------------------------------------------


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable", "Cached", "Dirty"):
                    out[key] = int(rest.strip().split()[0]) * 1024
    except OSError:
        pass
    return out


def _gpu_sample() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - preflight only
        return []
    out = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            out.append(
                {
                    "index": int(parts[0]),
                    "util_pct": float(parts[1]),
                    "mem_used_mib": float(parts[2]),
                    "mem_total_mib": float(parts[3]),
                }
            )
    return out


class HostSampler:
    """Background GPU / memory sampler for the throughput preflight."""

    def __init__(self, interval: float = 2.0):
        self.interval = float(interval)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.samples.append(
                {
                    "utc": time.time(),
                    "gpus": _gpu_sample(),
                    "meminfo": _meminfo(),
                    "loadavg": list(os.getloadavg()),
                }
            )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        gpus: dict[int, dict[str, Any]] = {}
        for sample in self.samples:
            for gpu in sample["gpus"]:
                rec = gpus.setdefault(
                    int(gpu["index"]), {"util": [], "mem_used_mib": [], "mem_total_mib": gpu["mem_total_mib"]}
                )
                rec["util"].append(float(gpu["util_pct"]))
                rec["mem_used_mib"].append(float(gpu["mem_used_mib"]))
        summary = {
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "gpus": {
                str(idx): {
                    "util_mean_pct": float(np.mean(rec["util"])) if rec["util"] else None,
                    "util_max_pct": float(np.max(rec["util"])) if rec["util"] else None,
                    "mem_used_mean_mib": float(np.mean(rec["mem_used_mib"])) if rec["mem_used_mib"] else None,
                    "mem_total_mib": float(rec["mem_total_mib"]),
                }
                for idx, rec in sorted(gpus.items())
            },
            "meminfo_start": self.samples[0]["meminfo"] if self.samples else _meminfo(),
            "meminfo_end": _meminfo(),
            "loadavg_end": list(os.getloadavg()),
        }
        return summary


def bench(task: Mapping[str, Any], *, run: str, device: torch.device, steps: int, warmup: int, out: Path) -> dict[str, Any]:
    C.apply_deterministic_settings()
    C.seed_everything(int(task["seed"]))
    model, audit = C.EW_MODEL.build_model(
        task["spec"], task["norm"], scale=int(task["scale"]), seed=int(task["seed"]), device=device
    )
    optimizer, _ = make_optimizer(model)
    stager = PinnedStager(device, depth=2)
    order = np.random.default_rng(12345).permutation(int(task["n_rows"])).astype(np.int64)
    total = min(int(warmup) + int(steps), int(order.size))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    import resource

    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    sampler = HostSampler(interval=2.0)
    sampler.start()
    t0 = time.perf_counter()
    times: list[float] = []
    losses: list[float] = []
    for n, i in enumerate(range(0, total * int(C.BATCH_SIZE), int(C.BATCH_SIZE))):
        rows = order[i : i + int(C.BATCH_SIZE)]
        t = time.perf_counter()
        raw = C.EW_COLLATE.collate_v2(task["train_cache"], rows.tolist())
        batch = stager.batch(raw)
        y = stager.labels(np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32))
        out_step = C.PQ_TRAIN.train_step(
            model,
            optimizer,
            batch,
            y,
            global_step=n,
            steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]),
            clip=float(C.RECIPE["gradient_clip"]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if n >= int(warmup):
            times.append(float(time.perf_counter() - t))
            losses.append(float(out_step["loss"]))
    wall = float(time.perf_counter() - t0)
    host = sampler.stop()
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    measured_steps = int(len(times))
    cpu_seconds = float(
        (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    )
    payload = {
        "schema": "lcrc-pcqm4mv2-fullscale-bench-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "run": run,
        "scale": int(task["scale"]),
        "parameters": int(audit["total_parameters"]),
        "device": str(device),
        "host": C.host_facts().get("hostname"),
        "steps_measured": measured_steps,
        "warmup_steps": int(warmup),
        "batch_size": int(C.BATCH_SIZE),
        "wall_seconds": wall,
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "molecules_per_second": float(measured_steps * int(C.BATCH_SIZE) / arr.sum()),
        "steps_per_second": float(measured_steps / arr.sum()),
        "loss_mean": float(np.mean(losses)),
        "loss_finite": bool(np.isfinite(losses).all()),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_rss_mb": C.peak_rss_mb(),
        "cpu_seconds_process": cpu_seconds,
        "cpu_cores_mean_process": float(cpu_seconds / wall) if wall > 0 else None,
        "host_sampler": host,
        "gpu": C.gpu_occupancy(),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
    }
    C.write_json(out, payload)
    C.log(
        f"[bench:{run}] {payload['molecules_per_second']:.0f} mol/s "
        f"({payload['step_ms_mean']:.2f} ms/step) {C.track_rel(out)}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=sorted(C.RUNS_TABLE))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=C.TOTAL_EPOCHS)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--soft-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--hard-seconds", type=float, default=13 * 3600.0)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--limit-batches", type=int, default=None, help="smoke only; marks the run non-formal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bench-steps", type=int, default=None)
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument("--bench-out", default=None)
    parser.add_argument("--no-verify-hashes", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    task = D.load_task(args.run, verify_hashes=not args.no_verify_hashes)
    if args.bench_steps is not None:
        out = Path(args.bench_out) if args.bench_out else C.ANALYSIS / f"bench_{args.run}.json"
        bench(task, run=args.run, device=device, steps=int(args.bench_steps), warmup=int(args.bench_warmup), out=out)
        return 0
    out_dir = Path(args.out_dir) if args.out_dir else C.RUNS / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_payload = None
    resume_mode = args.resume
    if resume_mode == "auto":
        candidate = out_dir / "checkpoint_last.pt"
        resume_mode = str(candidate) if candidate.is_file() else None
    if resume_mode:
        resume_payload = torch.load(resume_mode, map_location="cpu", weights_only=False)
        C.log(f"[{args.run}] resuming from {resume_mode} (epoch {resume_payload.get('epoch')})")
    elif (out_dir / "checkpoint_last.pt").exists() and not args.overwrite:
        raise RuntimeError(
            f"{out_dir}/checkpoint_last.pt exists; pass --resume auto to continue or --overwrite to start fresh"
        )
    try:
        payload = train(
            task,
            run=args.run,
            epochs=int(args.epochs),
            out_dir=out_dir,
            device=device,
            resume=resume_payload,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            eval_batch_size=int(args.eval_batch_size),
            limit_batches=args.limit_batches,
        )
    except Exception:  # noqa: BLE001 - keep the traceback in the job log
        traceback.print_exc()
        return 1
    return 0 if payload["status"] == "COMPLETED" or args.limit_batches is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
