"""Continuation training: epoch 40 -> 60 (Stage 1) or 60 -> 80 (Stage 2).

The loop is the frozen structural-gain/no-MP training loop with one controlled
change: the learning rate for the next global optimizer step is a deterministic
function of the recipe.  Model architecture, dropout, batch size, loss,
gradient clipping, AdamW moments and the L/R-shared batch order are unchanged.
No official Test target is read or scored.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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

import cc_common as C  # noqa: E402
import cc_data as D  # noqa: E402
import cc_restore as R  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "steps_per_epoch",
    "lr_first",
    "lr_last",
    "lr_mean",
    "train_loss",
    "valid_ap",
    "valid_bce",
    "n_tasks_with_observed_labels",
    "n_tasks_scored",
    "train_seconds",
    "valid_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "train_molecules_per_second",
    "grad_norm_max_epoch",
    "grad_norm_mean_epoch",
    "clip_events_epoch",
    "rng_unchanged_by_eval",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "notes",
]

PER_TASK_FIELDS = [
    "epoch",
    "task",
    "ap",
    "observed",
    "positive",
    "negative",
    "valid_stratum",
    "train_observed",
    "train_positive",
    "train_stratum",
]


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


def _write_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in CURVE_FIELDS})
    return C.sha256_file(path)


def _write_per_task(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_TASK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in PER_TASK_FIELDS})
    return C.sha256_file(path)


def _stratum(positive: int) -> str:
    if int(positive) < 5:
        return "support_lt5"
    if int(positive) < 20:
        return "support_5_19"
    return "support_ge20"


def _check_resume(
    payload: Mapping[str, Any],
    *,
    model_id: str,
    recipe: str,
    start_epoch: int,
    plan: Sequence[Mapping[str, Any]],
) -> None:
    expected_step = (int(start_epoch) - 1) * C.STEPS_PER_EPOCH
    checks = {
        "model": payload.get("model", payload.get("arm")) == model_id,
        "seed": int(payload.get("seed", -1)) == C.SEED,
        "epoch": int(payload.get("epoch", -1)) == int(start_epoch) - 1,
        "global_step": int(payload.get("global_step", -1)) == expected_step,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"resume checkpoint identity mismatch: {failed} ({checks})")
    if start_epoch == C.EPOCH40 + 1:
        if payload.get("status_at_save") != "COMPLETED":
            raise RuntimeError("epoch-40 source checkpoint is not marked COMPLETED")
        if payload.get("schema") not in (
            "lcrc-pcba-no-mp-checkpoint-v1",
            "lcrc-pcba-structural-gain-checkpoint-v1",
        ):
            raise RuntimeError(f"unexpected source schema {payload.get('schema')}")
        # The source plan is the historical 40-epoch plan, not the new 41..80 one.
        saved = payload.get("batch_plan") or {}
        if int(saved.get("consumed_epochs", -1)) != 40:
            raise RuntimeError("epoch-40 source checkpoint does not record 40 consumed epochs")
    elif start_epoch == C.EPOCH60 + 1:
        if payload.get("schema") != C.SCHEMA_CHECKPOINT:
            raise RuntimeError(f"Stage-2 resume schema {payload.get('schema')} is not this round's")
        if payload.get("stage") != "stage1":
            raise RuntimeError("Stage-2 resume is not a Stage-1 checkpoint")
        if payload.get("recipe") != recipe:
            raise RuntimeError(
                f"Stage-2 recipe {recipe} != Stage-1 checkpoint recipe {payload.get('recipe')}"
            )
        if payload.get("status_at_save") != "COMPLETED":
            raise RuntimeError("Stage-1 checkpoint is not marked COMPLETED")
        saved = payload.get("batch_plan") or {}
        if saved.get("global_plan_sha256") != D.plan_global_sha256(plan):
            raise RuntimeError("Stage-1 checkpoint batch plan differs from the frozen 41..80 plan")
        if saved.get("prefix_sha256") != D.plan_prefix_sha256(plan, C.EPOCH60 - C.EPOCH40):
            raise RuntimeError("Stage-1 checkpoint consumed prefix differs from the frozen plan")
    else:
        raise RuntimeError(f"unsupported start epoch {start_epoch}")


def train(
    task: Any,
    *,
    model_id: str,
    recipe: str,
    start_epoch: int,
    final_epoch: int,
    out_dir: Path,
    device: torch.device,
    resume_payload: Mapping[str, Any],
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (start_epoch, final_epoch) not in ((C.EPOCH40 + 1, C.EPOCH60), (C.EPOCH60 + 1, C.EPOCH80)):
        raise ValueError(f"unsupported continuation {(start_epoch, final_epoch)}")
    stage = "stage1" if start_epoch == C.EPOCH40 + 1 else "stage2"
    eval_epochs = C.EVAL_EPOCHS_STAGE1 if stage == "stage1" else C.EVAL_EPOCHS_STAGE2
    if final_epoch not in eval_epochs:
        raise RuntimeError("final epoch is not a fixed evaluation epoch")

    plan = D.make_plan_41_80(task, seed=C.SEED)
    plan_summary = D.plan_summary(plan, task, seed=C.SEED)
    plan_path = out_dir / "training_order_epoch41_80.json"
    plan_file_sha = D.write_plan(plan_path, plan_summary)
    _check_resume(
        resume_payload, model_id=model_id, recipe=recipe, start_epoch=start_epoch, plan=plan
    )

    model, optimizer, restore_audit = R.restore_model_and_optimizer(
        model_id, resume_payload, task, device=device
    )
    rng_hashes = R.restore_rng_state(resume_payload["rng_state"])
    saved_hashes = resume_payload.get("rng_state_hashes") or {}
    if rng_hashes["combined_sha256"] != saved_hashes.get("combined_sha256"):
        raise RuntimeError("restored RNG state does not match the resume checkpoint hashes")
    rng_start_hashes = dict(rng_hashes)

    start_step = (int(start_epoch) - 1) * C.STEPS_PER_EPOCH
    global_steps_run = int(start_step)
    model.train()
    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.COMMON_RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    counts = D.task_counts(task)
    train_counts = counts["train"]
    valid_counts = counts["valid"]

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    final_path = out_dir / f"checkpoint_epoch{final_epoch}.pt"
    rng_path = out_dir / "rng_state_end.pt"
    fixed_reports: dict[str, dict[str, Any]] = {}
    eval_completed: list[int] = []
    epochs_run = int(start_epoch) - 1
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    non_finite_events = 0
    clip_events_total = 0
    stop_reason = "RUNNING"
    final_checkpoint_sha = ""
    train_eval_summary: dict[str, Any] | None = None
    first_step: dict[str, Any] | None = None
    eval_rng_checks: list[dict[str, Any]] = []

    for epoch in range(int(start_epoch), int(final_epoch) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - (C.EPOCH40 + 1)]["batches"]
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        lrs: list[float] = []
        clip_events_epoch = 0
        aborted = False
        for step, rows in enumerate(batches):
            if watchdog.stop.is_set():
                aborted = True
                break
            lr = C.scheduled_lr(recipe, global_steps_run)
            R.set_lr(optimizer, lr)
            lrs.append(float(lr))
            t0 = time.perf_counter()
            batch = stager.batch(BM.collate(task.cache, np.asarray(rows).tolist()))
            y = stager.labels(task.labels[rows])
            logits = model(batch)
            if tuple(logits.shape) != (len(rows), C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), C.N_TASKS)}")
            if not torch.isfinite(logits).all():
                non_finite_events += 1
                raise RuntimeError(f"non-finite logits at epoch {epoch} step {step}")
            loss = MET.masked_bce_loss(logits, y)
            if not torch.isfinite(loss):
                non_finite_events += 1
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            if not math.isfinite(gnorm):
                non_finite_events += 1
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} step {step}")
            if gnorm > clip:
                clip_events_epoch += 1
            grad_norms.append(gnorm)
            if first_step is None:
                first_step = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "global_step": int(global_steps_run),
                    "lr": float(lr),
                    "loss": float(loss.detach().cpu().item()),
                    "grad_norm": float(gnorm),
                    "logits_sha256": C.sha256_bytes(
                        np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32).tobytes()
                    ),
                    "batch_rows_sha256": C.sha256_int64(np.asarray(rows, dtype=np.int64)),
                }
            optimizer.step()
            global_steps_run += 1
            losses.append(float(loss.detach().cpu().item()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{model_id}-{recipe}] hard wall during epoch {epoch}; partial epoch discarded")
            break
        if len(lrs) != C.STEPS_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} ran {len(lrs)} steps, expected {C.STEPS_PER_EPOCH}")

        train_seconds = float(time.perf_counter() - t_epoch)
        steps_arr = np.asarray(step_times, dtype=np.float64)
        step_mean = float(steps_arr.mean()) if steps_arr.size else float("nan")
        step_p50 = float(np.percentile(steps_arr, 50)) if steps_arr.size else float("nan")
        step_p90 = float(np.percentile(steps_arr, 90)) if steps_arr.size else float("nan")
        steps_per_sec = float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan")
        rng_before_eval = C.capture_rng_state()
        epochs_run = int(epoch)
        report: dict[str, Any] | None = None
        strict = False
        valid_seconds = 0.0
        rng_eval_ok = True
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            report = D.evaluate_validation(
                model, task, device, reason=f"epoch_{epoch}_official_validation"
            )
            valid_seconds = float(time.perf_counter() - t_eval)
            rng_after_eval = C.capture_rng_state()
            rng_eval_ok = (
                C.rng_state_hashes(rng_before_eval)["combined_sha256"]
                == C.rng_state_hashes(rng_after_eval)["combined_sha256"]
            )
            eval_rng_checks.append(
                {
                    "epoch": int(epoch),
                    "rng_unchanged": bool(rng_eval_ok),
                    "before": C.rng_state_hashes(rng_before_eval),
                    "after": C.rng_state_hashes(rng_after_eval),
                }
            )
            fixed_reports[str(epoch)] = {
                "ap": float(report["ap"]),
                "valid_bce": float(report["valid_bce"]),
                "n_tasks_scored": int(report["n_tasks_scored"]),
                "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
                "per_task_ap": report["per_task_ap"],
                "observed": report["observed"],
                "positive": report["positive"],
                "negative": report["negative"],
                "valid_logits_sha256": report.get("valid_logits_sha256"),
                "valid_probs_sha256": report.get("valid_probs_sha256"),
                "reason": report.get("reason", ""),
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
                        "valid_stratum": _stratum(int(valid_counts["positive"][t])),
                        "train_observed": int(train_counts["observed"][t]),
                        "train_positive": int(train_counts["positive"][t]),
                        "train_stratum": _stratum(int(train_counts["positive"][t])),
                    }
                )
            if epoch == final_epoch:
                t_tr = time.perf_counter()
                train_report = D.evaluate_train(model, task, device)
                train_eval_path = out_dir / f"train_eval_epoch{epoch}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "model": model_id,
                        "recipe": recipe,
                        "stage": stage,
                        "seed": C.SEED,
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
                        "ap_source": train_report["ap_source"],
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
                    "eval_seconds_with_eval": float(time.perf_counter() - t_tr),
                    "path": C.track_rel(train_eval_path),
                }
            model.train()

        rng_end_epoch = C.capture_rng_state()
        rng_epoch_hashes.append({"epoch": int(epoch), **C.rng_state_hashes(rng_end_epoch)})
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        cumulative = float(time.perf_counter() - t_start)
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "steps_per_epoch": int(len(batches)),
            "lr_first": float(lrs[0]),
            "lr_last": float(lrs[-1]),
            "lr_mean": float(np.mean(lrs)),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_ap": None if report is None else float(report["ap"]),
            "valid_bce": None if report is None else float(report["valid_bce"]),
            "n_tasks_with_observed_labels": None
            if report is None
            else int(report["n_tasks_with_observed_labels"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "train_seconds": train_seconds,
            "valid_seconds": valid_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "step_ms_mean": float(step_mean * 1e3),
            "step_ms_p50": float(step_p50 * 1e3),
            "step_ms_p90": float(step_p90 * 1e3),
            "steps_per_second": float(steps_per_sec),
            "train_molecules_per_second": float(len(task.train_rows) / train_seconds)
            if train_seconds > 0
            else float("nan"),
            "grad_norm_max_epoch": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean_epoch": float(np.mean(grad_norms)) if grad_norms else None,
            "clip_events_epoch": int(clip_events_epoch),
            "rng_unchanged_by_eval": bool(rng_eval_ok),
            "rng_cpu_sha256": C.rng_state_hashes(rng_before_eval)["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(C.rng_state_hashes(rng_before_eval)["torch_cuda_sha256"]),
            "notes": "" if report is not None else "validation not scheduled at this epoch",
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_curve(curve_path, curve_rows)
        _write_per_task(per_task_path, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{model_id}-{recipe}] {stage} epoch {epoch}/{final_epoch} loss {row['train_loss']:.6f} "
            f"validAP {valid_txt} LR {row['lr_first']:.3e}->{row['lr_last']:.3e} "
            f"train {train_seconds:.1f}s step_p50 {row['step_ms_p50']:.2f}ms"
        )

        if epoch == final_epoch:
            final_payload = R.checkpoint_payload(
                model_id=model_id,
                recipe=recipe,
                stage=stage,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=rng_before_eval,
                plan=D.plan_for_checkpoint(plan, int(epoch)),
                model_audit=restore_audit["model_audit"],
                source_audit={
                    "source_checkpoint": resume_payload.get("_path"),
                    "source_sha256": C.sha256_file(_abs_path(resume_payload.get("_path"))),
                    "source_epoch": int(resume_payload.get("epoch", -1)),
                    "source_global_step": int(resume_payload.get("global_step", -1)),
                    "source_valid_ap": resume_payload.get("valid_ap"),
                    "restore_audit": restore_audit,
                },
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"{stage}_epoch{final_epoch}",
                extra={
                    "status_at_save": "COMPLETED",
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": train_eval_summary,
                    "first_step": first_step,
                    "eval_rng_checks": eval_rng_checks,
                },
            )
            final_checkpoint_sha = R.save_checkpoint(final_payload, final_path)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    eval_completed = [int(k) for k in fixed_reports]
    if (
        global_steps_run == int(final_epoch) * C.STEPS_PER_EPOCH
        and set(eval_completed) == set(eval_epochs)
        and non_finite_events == 0
    ):
        status = "COMPLETED"
        stop_reason = f"COMPLETED_{stage.upper()}"
    else:
        status = "INCOMPLETE"
        if non_finite_events:
            stop_reason = "NON_FINITE_EVENT"
        elif stop_reason in ("", "RUNNING"):
            stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end = C.capture_rng_state()
    torch.save(
        {
            "schema": "lcrc-pcba-continuation-rng-v1",
            "round_id": C.ROUND_ID,
            "stage": stage,
            "model": model_id,
            "recipe": recipe,
            "start": rng_start_hashes,
            "end": C.rng_state_hashes(rng_end),
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)
    curve_sha = _write_curve(curve_path, curve_rows)
    per_task_sha = _write_per_task(per_task_path, per_task_rows)
    total_seconds = float(time.perf_counter() - t_start)

    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "stage": stage,
        "model": model_id,
        "model_name": C.SOURCE[model_id]["arm_name"],
        "recipe": recipe,
        "recipe_name": C.RECIPE_NAMES[recipe],
        "seed": C.SEED,
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": C.N_TASKS,
        "start_epoch": int(start_epoch),
        "final_epoch": int(final_epoch),
        "epochs_run": int(epochs_run),
        "last_completed_epoch": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(set(eval_completed) == set(eval_epochs)),
        "fixed_epoch_reports": fixed_reports,
        "train_eval_final": train_eval_summary,
        "source_checkpoint": resume_payload.get("_path"),
        "source_checkpoint_sha256": C.sha256_file(_abs_path(resume_payload.get("_path"))),
        "source_epoch": int(resume_payload.get("epoch", -1)),
        "restore_audit": restore_audit,
        "training_order_path": C.track_rel(plan_path),
        "training_order_sha256": plan_file_sha,
        "batch_plan": D.plan_for_checkpoint(plan, int(epochs_run)) if epochs_run >= C.EPOCH40 else None,
        "training_order_summary": plan_summary,
        "rng_state_start": rng_start_hashes,
        "rng_state_end": C.rng_state_hashes(rng_end),
        "rng_state_path": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "first_step": first_step,
        "eval_rng_checks": eval_rng_checks,
        "checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "checkpoint_sha256": final_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "task_counts": counts,
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "non_finite_events": int(non_finite_events),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "data_load_seconds": float(task.load_seconds),
        "train_seconds": float(sum(float(r.get("train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("valid_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(
            np.mean([r["epoch_seconds"] for r in curve_rows if r["epoch"] >= start_epoch])
        )
        if curve_rows
        else 0.0,
        "steps_per_second_mean": float(
            np.mean([r["steps_per_second"] for r in curve_rows if r.get("steps_per_second")])
        )
        if any(r.get("steps_per_second") for r in curve_rows)
        else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "recipe_config": C.recipe_config(recipe),
        "lr_schedule": C.schedule_audit(),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done {model_id}-{recipe} {stage} status={status} epochs={epochs_run}/{final_epoch} "
        f"gpu_hours={run['gpu_hours']:.4f} checkpoint={final_checkpoint_sha[:12] if final_checkpoint_sha else None}"
    )
    return run


def _abs_path(path: Any) -> Path:
    p = Path(str(path))
    if p.is_absolute():
        return p
    return C.REPO / p


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(C.MODEL_ORDER))
    parser.add_argument("--recipe", required=True, choices=list(C.RECIPE_ORDER))
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--final-epoch", type=int, required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
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
                "stage": "stage1" if args.start_epoch == C.EPOCH40 + 1 else "stage2",
                "model": args.model,
                "recipe": args.recipe,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        task = D.load_and_gate()
        resume_payload = R.load_checkpoint(args.resume)
        run = train(
            task,
            model_id=args.model,
            recipe=args.recipe,
            start_epoch=int(args.start_epoch),
            final_epoch=int(args.final_epoch),
            out_dir=out_dir,
            device=device,
            resume_payload=resume_payload,
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
                "stage": "stage1" if args.start_epoch == C.EPOCH40 + 1 else "stage2",
                "model": args.model,
                "recipe": args.recipe,
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"FAILED {args.model}-{args.recipe}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
