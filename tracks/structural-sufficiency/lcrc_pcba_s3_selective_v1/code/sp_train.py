"""Formal from-scratch training for the LCRC-S3-SP round.

One arm (``SP``), one seed, 40 epochs, exactly one formal run.  The training
step is the *inherited* ``bs_train.train_step``, so collate / stager / model /
AdamW / clip / masked-BCE / LR schedule are the same code path as the historical
W2 and S3 runs.  The train batch order is the frozen W2 plan
(``06f43729…b6f6``); the S3 structural tensors come from the read-only
Train+Valid S3 cache built by ``lcrc_pcba_s3_v1``.

Model selection: official Validation macro AP at every epoch 1..40 and

    e* = argmax_{e in 1..40} AP_valid(e),   earliest epoch wins exact ties

``checkpoint_best.pt`` (selected), ``checkpoint_epoch40.pt`` (auxiliary fixed
endpoint) and ``checkpoint_last.pt`` (resumable) are written together with the
full Validation curve and per-task AP.  No official Test target is read, indexed
or scored; ``test_evaluated`` is always false in this file.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
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

import sp_common as C  # noqa: E402
import sp_model as M  # noqa: E402
import bs_train as BT  # noqa: E402  (inherited training step / IO helpers)
import bs_model as BSM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce",
    "valid_ap",
    "valid_bce",
    "n_tasks_scored",
    "n_tasks_with_observed_labels",
    "epoch_train_seconds",
    "epoch_valid_seconds",
    "epoch_diag_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "s3_param_grad_norm",
    "selective_param_grad_norm",
    "s3_module_norm",
    "selective_module_norm",
    "branch_residual_over_h_raw",
    "s3_residual_rms",
    "selective_residual_rms",
    "selective_delta_rms",
    "selective_over_residual_total",
    "selective_nonzero_fraction",
    "paths_per_atom",
    "attention_entropy_ratio",
    "attention_max_weight_mean",
    "attention_near_one_hot_fraction",
    "attention_weight_sum_error",
    "checkpoint_seconds",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
]
PER_TASK_FIELDS = [
    "epoch",
    "task",
    "ap",
    "observed",
    "positive",
    "negative",
    "train_observed",
    "train_positive",
    "stratum",
]

train_step = BT.train_step
Watchdog = BT.Watchdog
_write_csv = BT._write_csv
plan_meta = BT.plan_meta
scheduler_state = BT.scheduler_state
_atomic_save = BT._atomic_save


# ---------------------------------------------------------------------------
# checkpoint payload / restore
# ---------------------------------------------------------------------------


def checkpoint_payload(
    *,
    model_key: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    kind: str,
    progress: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    m = C.MODELS[model_key]
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": model_key,
        "model_name": m["name"],
        "base_model": C.BASE_MODEL_KEY,
        "seed": int(C.SEED),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "parameters": int(model_audit["total_parameters"]),
        "new_parameters": int(model_audit["new_parameters"]),
        "state_dict": BSM.state_cpu(model),
        "optimizer_state_dict": BSM.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": scheduler_state(global_step, optimizer),
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": dict(plan),
        "model_audit": dict(model_audit),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "cache_sha256_manifest": C.SOURCE["cache_sha256"],
        "s3_cache_sha256": C.s3_train_cache_identity(),
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "valid_report_summary": None
        if valid_report is None
        else {
            "ap": float(valid_report["ap"]),
            "valid_bce": float(valid_report["valid_bce"]),
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "recipe": C.recipe_for_model(model_key),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed here",
        "created_utc": C.utcnow(),
        "progress": None if progress is None else dict(progress),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_key: str,
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("model") != model_key:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != requested {model_key}")
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
    hashes = BSM.restore_rng_state(payload["rng_state"])
    if hashes["combined_sha256"] != (payload.get("rng_state_hashes") or {}).get("combined_sha256"):
        raise RuntimeError("restored RNG state does not match the checkpoint hashes")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "rng_state_hashes": hashes,
    }


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_validation(
    model: torch.nn.Module,
    task: Any,
    device: torch.device,
    *,
    reason: str,
    selective_zero: bool = False,
) -> dict[str, Any]:
    """Official Validation AP through the frozen evaluator, no weight update."""
    import pc_metrics as MET
    import pc_model as PC

    model.eval()
    logits = []
    rows = np.asarray(task.valid_rows, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, rows.size, 256):
            chunk = rows[start : start + 256]
            batch = C.collate_with_s3(task, task.s3, chunk.tolist())
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if selective_zero:
                out = model(batch, selective_zero=True)
            else:
                out = model(batch)
            logits.append(out.detach().cpu().numpy().astype(np.float32))
    logits_np = np.concatenate(logits, axis=0)
    y = np.asarray(task.labels[rows])
    probs = PC.sigmoid(logits_np)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": reason,
            "valid_bce": float(MET.masked_bce_numpy(logits_np, y)),
            "valid_logits_sha256": C.sha256_bytes(np.ascontiguousarray(logits_np, dtype=np.float32).tobytes()),
            "evaluated_rows": int(rows.size),
            "selective_zero_ablation": bool(selective_zero),
            "bce_and_ap_are_nan_masked": True,
        }
    )
    return report


def evaluate_train(model: torch.nn.Module, task: Any, device: torch.device) -> dict[str, Any]:
    import pc_metrics as MET
    import pc_model as PC

    t0 = time.perf_counter()
    model.eval()
    rows = np.asarray(task.train_rows, dtype=np.int64)
    chunks = []
    with torch.inference_mode():
        for start in range(0, rows.size, 512):
            chunk = rows[start : start + 512]
            batch = C.collate_with_s3(task, task.s3, chunk.tolist())
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(chunks, axis=0)
    y = np.asarray(task.labels[rows])
    probs = PC.sigmoid(logits)
    report = MET.validation_report(y, probs)
    report.update(
        {
            "reason": "fixed_final_epoch_train_eval",
            "train_bce": float(MET.masked_bce_numpy(logits, y)),
            "train_logits_sha256": C.sha256_bytes(np.ascontiguousarray(logits, dtype=np.float32).tobytes()),
            "evaluated_rows": int(rows.size),
            "seconds": float(time.perf_counter() - t0),
        }
    )
    return report


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    task: Any,
    *,
    model_key: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    eval_every: int,
    epochs: int = C.TOTAL_EPOCHS,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_key not in C.MODELS:
        raise KeyError(model_key)
    eval_epochs = tuple(e for e in range(1, int(epochs) + 1) if e % int(eval_every) == 0)
    formal = int(epochs) == int(C.TOTAL_EPOCHS) and int(eval_every) == 1
    order_seed = int(seed)
    gpu_start = C.BC.gpu_occupancy()

    model, model_audit = M.build_model(
        task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device
    )
    optimizer, opt_audit = BSM.make_optimizer(model)

    plan = C.BC.SD.make_plan40(task, seed=order_seed)
    plan_audit = C.BC.SD.plan_summary(plan, task)
    if plan[0]["global_plan_sha256"] != C.W2["global_plan_sha256"]:
        raise RuntimeError(
            f"order plan {plan[0]['global_plan_sha256']} != frozen W2 plan {C.W2['global_plan_sha256']}"
        )

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer, model_key=model_key)
        saved = resume.get("batch_plan") or {}
        now = plan_meta(plan, int(resume["epoch"]), order_seed)
        if now["global_plan_sha256"] != saved.get("global_plan_sha256") or now["prefix_sha256"] != saved.get("prefix_sha256"):
            raise RuntimeError("resumed plan does not match the checkpoint plan")
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": plan_meta(plan, int(epochs), order_seed), **plan_audit})

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    counts = C.BC.SD.task_counts(task)
    train_counts = counts["train"]
    valid_pos = counts["valid"]["positive"]
    strata = [
        "support_lt5" if valid_pos[t] < 5 else "support_5_19" if valid_pos[t] < 20 else "support_ge20"
        for t in range(C.N_TASKS)
    ]

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{int(epochs)}.pt"
    rng_path = out_dir / "rng_state.pt"

    diag_rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[:64]
    diag_batch = C.collate_with_s3(task, task.s3, diag_rows.tolist())
    diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in diag_batch.items()}

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    best_epoch: int | None = None
    best_ap = -math.inf
    best_checkpoint_sha = ""
    final_checkpoint_sha = ""
    last_checkpoint_sha = ""
    eval_completed: list[int] = []
    fixed_reports: dict[int, dict[str, Any]] = {}
    clip_events_total = 0
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    train_eval_summary: dict[str, Any] | None = None
    epoch0_report: dict[str, Any] | None = None

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    def record_eval(epoch: int, report: Mapping[str, Any]) -> None:
        eval_completed.append(int(epoch))
        fixed_reports[int(epoch)] = {
            "ap": float(report["ap"]),
            "valid_bce": float(report["valid_bce"]),
            "n_tasks_scored": int(report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
            "per_task_ap": report["per_task_ap"],
            "positive": report["positive"],
            "observed": report["observed"],
            "valid_logits_sha256": report.get("valid_logits_sha256"),
            "reason": report.get("reason", ""),
            "step": int(global_steps_run),
        }
        for t, ap in enumerate(report["per_task_ap"]):
            per_task_rows.append(
                {
                    "epoch": int(epoch),
                    "task": int(t),
                    "ap": "" if ap is None else float(ap),
                    "observed": int(report["observed"][t]),
                    "positive": int(report["positive"][t]),
                    "negative": int(report["negative"][t]),
                    "train_observed": int(train_counts["observed"][t]),
                    "train_positive": int(train_counts["positive"][t]),
                    "stratum": strata[t],
                }
            )
        update_peak()

    if resume is None:
        t_eval = time.perf_counter()
        report0 = evaluate_validation(model, task, device, reason="epoch_0_initialisation_diagnostic")
        epoch0_report = {
            "epoch": 0,
            "ap": float(report0["ap"]),
            "valid_bce": float(report0["valid_bce"]),
            "n_tasks_scored": int(report0["n_tasks_scored"]),
            "seconds": float(time.perf_counter() - t_eval),
            "global_step": 0,
            "valid_logits_sha256": report0.get("valid_logits_sha256"),
            "rng_after": C.rng_state_hashes(C.capture_rng_state()),
        }
        if epoch0_report["rng_after"]["combined_sha256"] != rng_start_hashes["combined_sha256"]:
            raise RuntimeError("epoch-0 diagnostic evaluation changed the RNG state")
        update_peak()
        C.log(f"[{model_key}] epoch0 diagnostic validAP {epoch0_report['ap']:.6f}")

    for epoch in range(int(start_epoch), int(epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            raw = C.collate_with_s3(task, task.s3, rows.tolist())
            batch = stager.batch(raw)
            y = stager.labels(task.labels[rows])
            out = train_step(model, optimizer, batch, y, global_step=global_steps_run, clip=clip)
            global_steps_run += 1
            losses.append(out["loss"])
            grad_norms.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{model_key}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        report: dict[str, Any] | None = None
        valid_seconds = 0.0
        diag_seconds = 0.0
        strict = False

        t_diag = time.perf_counter()
        branch_diag = M.branch_diagnostics(model, diag_batch)
        grad_snapshot = M.branch_grad_norm(model)
        new_norm = M.new_module_norms(model)
        diag_seconds += float(time.perf_counter() - t_diag)

        if epoch in eval_epochs:
            t_eval = time.perf_counter()
            report = evaluate_validation(model, task, device, reason=f"epoch_{epoch}_official_validation")
            valid_seconds = float(time.perf_counter() - t_eval)
            strict = C.better_candidate(float(report["ap"]), best_ap, int(epoch), best_epoch)
            if strict:
                best_epoch = int(epoch)
                best_ap = float(report["ap"])
                payload = checkpoint_payload(
                    model_key=model_key,
                    epoch=int(epoch),
                    global_step=int(global_steps_run),
                    model=model,
                    optimizer=optimizer,
                    rng_state=C.capture_rng_state(),
                    plan=plan_meta(plan, int(epoch), order_seed),
                    model_audit=model_audit,
                    valid_ap=best_ap,
                    valid_report=report,
                    kind="best_valid_ap_selected",
                    extra={
                        "optimizer_state_audit": opt_audit,
                        "selection_rule": "argmax AP_valid over epochs 1..40 (earliest wins ties)",
                    },
                )
                best_checkpoint_sha = _atomic_save(payload, best_path)
            record_eval(epoch, report)
            if epoch == int(epochs):
                t_tr = time.perf_counter()
                train_report = evaluate_train(model, task, device)
                train_eval_path = out_dir / f"train_eval_epoch{int(epoch)}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "model": model_key,
                        "seed": int(seed),
                        "ap": float(train_report["ap"]),
                        "bce": float(train_report["train_bce"]),
                        "per_task_ap": train_report["per_task_ap"],
                        "observed": train_report["observed"],
                        "positive": train_report["positive"],
                        "negative": train_report["negative"],
                        "n_tasks_scored": int(train_report["n_tasks_scored"]),
                        "evaluated_rows": int(train_report["evaluated_rows"]),
                        "seconds": float(train_report["seconds"]),
                        "logits_sha256": train_report["train_logits_sha256"],
                        "test_evaluated": False,
                    },
                )
                train_eval_summary = {
                    "epoch": int(epoch),
                    "ap": float(train_report["ap"]),
                    "bce": float(train_report["train_bce"]),
                    "n_tasks_scored": int(train_report["n_tasks_scored"]),
                    "evaluated_rows": int(train_report["evaluated_rows"]),
                    "seconds": float(train_report["seconds"]),
                    "eval_seconds": float(time.perf_counter() - t_tr),
                    "path": C.track_rel(train_eval_path),
                }
            model.train()
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        update_peak()
        epochs_run = int(epoch)
        cumulative = float(time.perf_counter() - t_start)
        attention = branch_diag["attention"]
        s3_grad = sum(v for k, v in grad_snapshot.items() if k.startswith("s3_branch."))
        sel_grad = sum(v for k, v in grad_snapshot.items() if k.startswith("selective."))
        s3_norm = sum(v["l2"] for k, v in new_norm.items() if k.startswith("s3_branch."))
        sel_norm = sum(v["l2"] for k, v in new_norm.items() if k.startswith("selective."))
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce": float(np.mean(losses)) if losses else float("nan"),
            "valid_ap": None if report is None else float(report["ap"]),
            "valid_bce": None if report is None else float(report["valid_bce"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": None
            if report is None
            else int(report["n_tasks_with_observed_labels"]),
            "epoch_train_seconds": train_seconds,
            "epoch_valid_seconds": valid_seconds,
            "epoch_diag_seconds": diag_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
            "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan"),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": None if best_epoch is None else int(best_epoch),
            "best_valid_ap_so_far": None if best_epoch is None else float(best_ap),
            "is_best": bool(strict),
            "s3_param_grad_norm": float(s3_grad),
            "selective_param_grad_norm": float(sel_grad),
            "s3_module_norm": float(s3_norm),
            "selective_module_norm": float(sel_norm),
            "branch_residual_over_h_raw": float(branch_diag["residual_over_h_raw"]),
            "s3_residual_rms": float(branch_diag["residual_original"]["rms"]),
            "selective_residual_rms": float(branch_diag["residual_selective"]["rms"]),
            "selective_delta_rms": float(branch_diag["selective_delta"]["rms"]),
            "selective_over_residual_total": float(branch_diag["selective_over_residual_total"]),
            "selective_nonzero_fraction": float(branch_diag["selective_nonzero_fraction"]),
            "paths_per_atom": float(branch_diag.get("paths_per_atom", 0.0)),
            "attention_entropy_ratio": float(attention.get("entropy_ratio_mean", 1.0)),
            "attention_max_weight_mean": float(attention.get("max_weight_mean", 1.0)),
            "attention_near_one_hot_fraction": float(attention.get("near_one_hot_fraction", 0.0)),
            "attention_weight_sum_error": float(attention.get("weight_sum_max_abs_error", 0.0)),
            "checkpoint_seconds": None,
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{model_key}] epoch {epoch}/{int(epochs)} bce {row['train_bce']:.6f} "
            f"validAP {valid_txt} LR {row['learning_rate']:.3e} "
            f"train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
            f"sel/h_raw {row['selective_over_residual_total']:.3f} "
            f"attn_entropy {row['attention_entropy_ratio']:.3f}"
            f"{' *best*' if strict else ''}"
        )

        if epoch == int(epochs):
            payload = checkpoint_payload(
                model_key=model_key,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"epoch{int(epochs)}_fixed_endpoint",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": None if train_eval_summary is None else dict(train_eval_summary),
                    "status_at_save": "COMPLETED",
                    "role": "auxiliary fixed-epoch endpoint (NOT the selected model)",
                    "selected_epoch": None if best_epoch is None else int(best_epoch),
                },
            )
            final_checkpoint_sha = _atomic_save(payload, final_path)

        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            model_key=model_key,
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            plan=plan_meta(plan, int(epoch), order_seed),
            model_audit=model_audit,
            valid_ap=None if report is None else float(report["ap"]),
            valid_report=None,
            kind="last_epoch_resumable",
            extra={"status_at_save": "RUNNING_OR_COMPLETED", "optimizer_state_audit": opt_audit},
        )
        last_checkpoint_sha = _atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    completed = (
        epochs_run >= int(epochs)
        and set(eval_completed) == set(eval_epochs)
        and bool(final_checkpoint_sha)
        and bool(best_checkpoint_sha)
    )
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end_hashes = C.rng_state_hashes(C.capture_rng_state())
    torch.save(
        {
            "schema": "lcrc-pcba-s3-selective-rng-v1",
            "round_id": C.ROUND_ID,
            "model": model_key,
            "seed": int(seed),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    curve_sha = _write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": model_key,
        "model_name": C.MODELS[model_key]["name"],
        "role": C.MODELS[model_key]["role"],
        "seed": int(seed),
        "order_seed": int(order_seed),
        "formal_run": bool(formal),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["total_parameters"]),
        "base_parameters": int(model_audit["base_parameters"]),
        "inherited_s3_parameters": int(model_audit["inherited_s3_parameters"]),
        "new_parameters": int(model_audit["new_parameters"]),
        "final_epoch": int(epochs),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_every": int(eval_every),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(set(eval_completed) == set(eval_epochs)),
        "epoch0_diagnostic": epoch0_report,
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "train_eval_final": train_eval_summary,
        "task_counts": counts,
        "task_strata": {str(t): strata[t] for t in range(C.N_TASKS)},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "batch_plan": plan_meta(plan, max(int(epochs_run), 0), order_seed),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": C.sha256_file(order_path),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_ap),
        "selection_rule": "argmax official Validation macro AP over epochs 1..40 (earliest epoch wins ties)",
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "epoch40_ap": None
        if int(epochs) not in fixed_reports
        else float(fixed_reports[int(epochs)]["ap"]),
        "epoch40_valid_bce": None
        if int(epochs) not in fixed_reports
        else float(fixed_reports[int(epochs)]["valid_bce"]),
        "epoch40_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "epoch40_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "recipe": C.recipe_for_model(model_key),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows])) if curve_rows else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows])) if curve_rows else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "host_facts": C.BC.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "s3_cache_identity": C.s3_train_cache_identity(),
        "test_cache_used": False,
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed here",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done model={model_key} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_ap={best_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="SP", choices=sorted(C.MODELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--s3-cache", default=str(C.S3_TRAIN_CACHE_DIR))
    parser.add_argument("--eval-every", type=int, default=1)
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
                "model": args.model,
                "seed": int(args.seed),
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        cache_path = Path(args.s3_cache)
        if not (cache_path / "meta.json").is_file():
            raise RuntimeError(
                f"the frozen Train+Valid S3 cache is missing at {cache_path}; it is read-only in this round"
            )
        task = C.load_task_gated(train_cache=cache_path)
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        run = train(
            task,
            model_key=args.model,
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            eval_every=int(args.eval_every),
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
                "model": args.model,
                "seed": int(args.seed),
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"FAILED model={args.model}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
