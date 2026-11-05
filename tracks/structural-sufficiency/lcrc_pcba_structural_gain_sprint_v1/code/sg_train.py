"""Formal Phase 1/2/3 training for the structural-gain sprint.

Phase 1: 12 epochs from the epoch-9 warm start (weights + restored AdamW
moments), four arms, pre-registered evaluation epochs 0/4/8/12, and a complete
epoch-12 continuation checkpoint.

Phase 2: strict continuation of the phase-1 epoch-12 checkpoint (model, AdamW,
scheduler, RNG, batch order) for the promoted arm and the C control, epochs
13..40, evaluation 20/30/40.

Phase 3: from-scratch {C,W} x {seed0,seed1}, 40 epochs, evaluation 10/20/30/40.

Official Test targets are unreachable: ``lr_data`` refuses a finite Test label
and this file never indexes ``task.split['test']``.
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

import sg_common as C  # noqa: E402
import sg_checkpoint as CK  # noqa: E402
import sg_data as D  # noqa: E402
import sg_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "steps_per_epoch",
    "base_lr",
    "new_lr",
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
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "strict_improve",
    "grad_norm_max_epoch",
    "grad_norm_mean_epoch",
    "clip_events_epoch",
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
    "train_observed",
    "train_positive",
    "stratum",
]


class Tracker:
    """Best Validation AP, ties -> earlier evaluated epoch."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_metric = -math.inf
        self.scores: list[dict[str, Any]] = []

    def consider(self, epoch: int, metric: float) -> bool:
        strict = self.best_epoch is None or float(metric) > float(self.best_metric)
        self.scores.append({"epoch": int(epoch), "metric": float(metric), "strict_improve": bool(strict)})
        if strict:
            self.best_epoch = int(epoch)
            self.best_metric = float(metric)
        return strict


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
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_TASK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in PER_TASK_FIELDS})
    return C.sha256_file(path)


def _scheduler_state(phase: int, global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "phase12_cosine_40epoch" if int(phase) in (1, 2) else "scratch_warmup_cosine_40epoch",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(),
    }


def _plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, order_seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(order_seed),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode()),
    }


def train(
    task: Any,
    *,
    phase: int,
    arm: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    source: Mapping[str, Any] | None,
    resume: Mapping[str, Any] | None,
    soft_seconds: float,
    hard_seconds: float,
    order_seed: int | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(phase) not in (1, 2, 3):
        raise ValueError(f"unknown phase {phase}")
    if int(phase) in (1, 2) and source is None:
        raise RuntimeError("Phase 1/2 needs the warm-start checkpoint")
    if int(phase) == 2 and resume is None:
        raise RuntimeError("Phase 2 needs the Phase-1 epoch-12 checkpoint")
    eval_epochs = tuple(int(x) for x in C.EVAL_EPOCHS[int(phase)])
    final_epoch = 12 if int(phase) == 1 else 40
    if int(phase) in (1, 2):
        order_seed = int(C.SEED if order_seed is None else order_seed)
    else:
        order_seed = int(seed if order_seed is None else order_seed)

    ref, ref_info, base_init = M.build_reference(task.spec, task.norm, C.N_TASKS, int(seed))
    seed_info = None
    if int(phase) == 3:
        warm_state = None
        model, model_audit = M.build_arm(
            arm,
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=int(seed),
            device=device,
            base_init_state=base_init,
            warm_start_state=None,
            zero_residual=True,
        )
        optimizer, opt_audit = M.make_optimizer_scratch(model)
        start_epoch = 1
        global_step = 0
        source_audit = None
        resume_audit = None
    else:
        warm_state = source["state_dict"]
        model, model_audit = M.build_arm(
            arm,
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=int(C.SEED),
            device=device,
            base_init_state=base_init,
            warm_start_state=warm_state,
            zero_residual=True,
        )
        optimizer, opt_audit = M.make_optimizer_warm(model, source["optimizer_state_dict"])
        source_audit = CK.verify_source_metadata(source, ref)
        source_audit["initialization_fingerprint"] = model_audit["base_state_fingerprint_after_load"]
        if int(phase) == 1:
            start_epoch = 1
            global_step = 0
            resume_audit = None
        else:
            assert resume is not None
            resume_audit = M.load_continuation(model, optimizer, resume, arm=arm)
            if int(resume.get("epoch", -1)) != int(C.PHASE1_EPOCHS):
                raise RuntimeError(f"Phase-1 checkpoint epoch {resume.get('epoch')} != 12")
            if resume.get("phase") != 1:
                raise RuntimeError("Phase-2 resume checkpoint is not a Phase-1 checkpoint")
            start_epoch = int(resume["epoch"]) + 1
            global_step = int(resume["global_step"])
            if global_step != C.PHASE1_EPOCHS * C.STEPS_PER_EPOCH:
                raise RuntimeError(f"Phase-1 consumed steps {global_step} != 12*2738")

    plan = D.make_plan40(task, seed=order_seed)
    plan_audit = D.plan_summary(plan, task)
    if int(phase) == 2:
        assert resume is not None
        now = _plan_meta(plan, C.PHASE1_EPOCHS, order_seed)
        saved = resume.get("batch_plan") or {}
        if now["prefix_sha256"] != saved.get("prefix_sha256") or now["global_plan_sha256"] != saved.get(
            "global_plan_sha256"
        ):
            raise RuntimeError(f"Phase-2 batch plan does not continue the Phase-1 plan: {now} vs {saved}")
    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": _plan_meta(plan, C.TOTAL_EPOCHS, order_seed), **plan_audit})
    order_sha = C.sha256_file(order_path)

    if int(phase) == 2:
        assert resume is not None
        rng_hashes = M.restore_rng_state(resume["rng_state"])
        if rng_hashes["combined_sha256"] != (resume.get("rng_state_hashes") or {}).get("combined_sha256"):
            raise RuntimeError("restored RNG state does not match the Phase-1 checkpoint hashes")
    else:
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    rng_start_hashes = dict(rng_hashes)

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    tracker = Tracker()
    clip = float(C.RECIPE["gradient_clip"] if int(phase) in (1, 2) else C.SCRATCH_RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{final_epoch}.pt"
    partial_path = out_dir / "checkpoint_partial.pt"
    rng_path = out_dir / "rng_state.pt"

    counts = D.task_counts(task)
    train_counts = counts["train"]
    strata = [
        "support_lt5"
        if train_counts["positive"][t] < 5
        else "support_5_19"
        if train_counts["positive"][t] < 20
        else "support_ge20"
        for t in range(C.N_TASKS)
    ]

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    best_epoch: int | None = None
    best_ap = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_opt: dict[str, Any] | None = None
    best_report: Mapping[str, Any] | None = None
    best_rng: Mapping[str, Any] | None = None
    best_checkpoint_sha = ""
    fixed_reports: dict[int, dict[str, Any]] = {}
    eval_completed: list[int] = []
    epochs_run = int(start_epoch) - 1
    global_steps_run = int(global_step)
    non_finite_events = 0
    clip_events_total = 0
    partial_epoch_discarded = False
    stop_reason = "RUNNING"
    final_checkpoint_sha = ""
    train_eval_summary: dict[str, Any] | None = None
    started_from_epoch = int(start_epoch - 1)

    def update_best(epoch: int, report: Mapping[str, Any], rng_state: Mapping[str, Any]) -> None:
        nonlocal best_epoch, best_ap, best_state, best_opt, best_report, best_rng, best_checkpoint_sha
        best_epoch = int(epoch)
        best_ap = float(report["ap"])
        best_state = M.state_cpu(model)
        best_opt = M.optimizer_state_cpu(optimizer)
        best_report = dict(report)
        best_rng = rng_state
        payload = CK.checkpoint_payload(
            phase=int(phase),
            arm=arm,
            seed=int(seed),
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            scheduler_state=_scheduler_state(int(phase), int(global_steps_run), optimizer),
            rng_state=rng_state,
            plan=_plan_meta(plan, int(epoch), order_seed),
            model_audit=model_audit,
            source_audit=source_audit,
            valid_ap=best_ap,
            valid_report=report,
            kind="best_valid_ap",
            extra={"optimizer_state_audit": opt_audit, "resume_audit": resume_audit},
        )
        payload["state_dict"] = best_state
        payload["optimizer_state_dict"] = best_opt
        best_checkpoint_sha = CK.save_checkpoint(payload, best_path)

    def record_eval(epoch: int, report: Mapping[str, Any]) -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
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
                    "train_observed": int(train_counts["observed"][t]),
                    "train_positive": int(train_counts["positive"][t]),
                    "stratum": strata[t],
                }
            )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    # ---------------- epoch 0, Phase 1 only ----------------
    if int(phase) == 1:
        model.eval()
        t_eval = time.perf_counter()
        report0 = D.evaluate_validation(model, task, device, reason="epoch_0_pre_update")
        valid_seconds0 = float(time.perf_counter() - t_eval)
        strict0 = tracker.consider(0, float(report0["ap"]))
        rng0 = C.capture_rng_state()
        rng0_hashes = C.rng_state_hashes(rng0)
        rng_epoch_hashes.append({"epoch": 0, **rng0_hashes})
        if strict0:
            update_best(0, report0, rng0)
        record_eval(0, report0)
        curve_rows.append(
            {
                "epoch": 0,
                "evaluated": True,
                "steps_per_epoch": int(C.STEPS_PER_EPOCH),
                "base_lr": float(C.scheduled_lr(0, phase=1, kind="base")),
                "new_lr": float(C.scheduled_lr(0, phase=1, kind="new")) if arm != "C" else None,
                "train_loss": None,
                "valid_ap": float(report0["ap"]),
                "valid_bce": float(report0["valid_bce"]),
                "n_tasks_with_observed_labels": int(report0["n_tasks_with_observed_labels"]),
                "n_tasks_scored": int(report0["n_tasks_scored"]),
                "train_seconds": 0.0,
                "valid_seconds": valid_seconds0,
                "epoch_seconds": valid_seconds0,
                "cumulative_seconds": float(time.perf_counter() - t_start),
                "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
                "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
                "step_ms_mean": None,
                "step_ms_p50": None,
                "step_ms_p90": None,
                "steps_per_second": None,
                "train_molecules_per_second": None,
                "best_epoch_so_far": int(tracker.best_epoch),
                "best_valid_ap_so_far": float(tracker.best_metric),
                "is_best": bool(strict0),
                "strict_improve": bool(strict0),
                "grad_norm_max_epoch": None,
                "grad_norm_mean_epoch": None,
                "clip_events_epoch": 0,
                "rng_cpu_sha256": rng0_hashes["torch_cpu_sha256"],
                "rng_cuda_sha256": "|".join(rng0_hashes["torch_cuda_sha256"]),
                "notes": "epoch 0 = warm-start epoch-9 weights, before any optimizer update",
            }
        )
        _write_curve(curve_path, curve_rows)
        _write_per_task(per_task_path, per_task_rows)
        C.log(
            f"[{arm}] phase1 epoch0 validAP {report0['ap']:.8f} "
            f"delta_hist {abs(report0['ap'] - C.AP0):.3e} ({valid_seconds0:.1f}s)"
        )
        model.train()

    # ---------------- epochs ----------------
    for epoch in range(int(start_epoch), int(final_epoch) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        elapsed = time.perf_counter() - t_start
        if elapsed > float(soft_seconds):
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
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            lrs = M.set_lrs(optimizer, phase=int(phase), step=int(global_steps_run))
            t0 = time.perf_counter()
            batch = stager.batch(BM.collate(task.cache, rows.tolist()))
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
            optimizer.step()
            global_steps_run += 1
            losses.append(float(loss.detach().cpu().item()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            partial_epoch_discarded = True
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{arm}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        steps_arr = np.asarray(step_times, dtype=np.float64)
        step_mean = float(steps_arr.mean()) if steps_arr.size else float("nan")
        step_p50 = float(np.percentile(steps_arr, 50)) if steps_arr.size else float("nan")
        step_p90 = float(np.percentile(steps_arr, 90)) if steps_arr.size else float("nan")
        steps_per_sec = float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan")
        report: dict[str, Any] | None = None
        strict = False
        valid_seconds = 0.0
        train_eval: dict[str, Any] | None = None
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            report = D.evaluate_validation(model, task, device, reason=f"epoch_{epoch}_official_validation")
            valid_seconds = float(time.perf_counter() - t_eval)
            strict = tracker.consider(epoch, float(report["ap"]))
            rng_eval = C.capture_rng_state()
            if strict:
                update_best(epoch, report, rng_eval)
            record_eval(epoch, report)
            if epoch == final_epoch:
                # Train macro AP / BCE only at the pre-registered final epoch.
                t_tr = time.perf_counter()
                train_report = D.evaluate_train(model, task, device)
                train_eval_path = out_dir / f"train_eval_epoch{epoch}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "arm": arm,
                        "phase": int(phase),
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
                    "per_task_ap": train_report["per_task_ap"],
                    "seconds": float(train_report["seconds"]),
                    "eval_seconds_with_eval": float(time.perf_counter() - t_tr),
                    "path": C.track_rel(train_eval_path),
                }
            model.train()
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        cumulative = float(time.perf_counter() - t_start)
        epochs_run = int(epoch)
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "steps_per_epoch": int(len(batches)),
            "base_lr": float(optimizer.param_groups[0]["lr"]),
            "new_lr": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None,
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
            "best_epoch_so_far": int(tracker.best_epoch),
            "best_valid_ap_so_far": float(tracker.best_metric),
            "is_best": bool(strict),
            "strict_improve": bool(strict),
            "grad_norm_max_epoch": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean_epoch": float(np.mean(grad_norms)) if grad_norms else None,
            "clip_events_epoch": int(clip_events_epoch),
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
            "notes": "" if report is not None else "validation not scheduled at this epoch",
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_curve(curve_path, curve_rows)
        _write_per_task(per_task_path, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{arm}] phase{phase} epoch {epoch}/{final_epoch} loss {row['train_loss']:.6f} "
            f"validAP {valid_txt} baseLR {row['base_lr']:.3e} "
            f"train {train_seconds:.1f}s step_p50 {row['step_ms_p50']:.2f}ms"
            f"{' *best*' if strict else ''}"
        )

        # Save a complete continuation checkpoint at the pre-registered final epoch.
        if epoch == final_epoch and epochs_run == final_epoch:
            rng_final = C.capture_rng_state()
            payload = CK.checkpoint_payload(
                phase=int(phase),
                arm=arm,
                seed=int(seed),
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                scheduler_state=_scheduler_state(int(phase), int(global_steps_run), optimizer),
                rng_state=rng_final,
                plan=_plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                source_audit=source_audit,
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"phase{phase}_epoch{final_epoch}",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": None
                    if train_eval_summary is None
                    else {k: v for k, v in train_eval_summary.items() if k != "per_task_ap"},
                    "status_at_save": "COMPLETED",
                },
            )
            final_checkpoint_sha = CK.save_checkpoint(payload, final_path)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    completed = epochs_run >= final_epoch and set(eval_completed) == set(eval_epochs) and non_finite_events == 0
    if completed:
        status = "COMPLETED"
        stop_reason = f"COMPLETED_PHASE{phase}"
    else:
        status = "INCOMPLETE"
        if non_finite_events:
            stop_reason = "NON_FINITE_EVENT"
        elif not stop_reason or stop_reason == "RUNNING":
            stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end = C.capture_rng_state()
    rng_end_hashes = C.rng_state_hashes(rng_end)
    torch.save(
        {
            "schema": "lcrc-pcba-structural-gain-rng-v1",
            "round_id": C.ROUND_ID,
            "phase": int(phase),
            "arm": arm,
            "seed": int(seed),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    # Partial-stop checkpoint (never overwrites the epoch-12/40 continuation file).
    if not completed:
        payload = CK.checkpoint_payload(
            phase=int(phase),
            arm=arm,
            seed=int(seed),
            epoch=int(epochs_run),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            scheduler_state=_scheduler_state(int(phase), int(global_steps_run), optimizer),
            rng_state=rng_end,
            plan=_plan_meta(plan, int(max(epochs_run, 0)), order_seed),
            model_audit=model_audit,
            source_audit=source_audit,
            valid_ap=None if not curve_rows else curve_rows[-1].get("valid_ap"),
            valid_report=None,
            kind="partial_stop",
            extra={
                "optimizer_state_audit": opt_audit,
                "resume_audit": resume_audit,
                "consumed_epochs": int(epochs_run),
                "consumed_steps": int(global_steps_run),
                "partial_epoch_discarded": bool(partial_epoch_discarded),
                "status_at_save": "INCOMPLETE",
                "stop_reason_at_save": stop_reason,
            },
        )
        CK.save_checkpoint(payload, partial_path)

    curve_sha = _write_curve(curve_path, curve_rows)
    per_task_sha = _write_per_task(per_task_path, per_task_rows)
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": "lcrc-pcba-structural-gain-run-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "phase": int(phase),
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "role": C.ARMS[arm]["role"],
        "block_mode": C.ARMS[arm]["blocks"],
        "seed": int(seed),
        "order_seed": int(order_seed),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "variant": "C1 + " + C.ARMS[arm]["blocks"],
        "parameters": int(model_audit["total_parameter_count"]),
        "new_parameters": int(model_audit["new_parameter_count"]),
        "base_parameters": int(model_audit["base_parameter_count"]),
        "start_epoch": int(started_from_epoch),
        "final_epoch": int(final_epoch),
        "epochs_run": int(epochs_run),
        "last_completed_epoch": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(set(eval_completed) == set(eval_epochs)),
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "train_eval_final": None
        if train_eval_summary is None
        else {k: v for k, v in train_eval_summary.items() if k != "per_task_ap"},
        "train_eval_final_path": None if train_eval_summary is None else train_eval_summary["path"],
        "task_counts": counts,
        "task_strata": {str(t): strata[t] for t in range(C.N_TASKS)},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "source_audit": source_audit,
        "resume_audit": resume_audit,
        "reference_model_info": ref_info,
        "batch_plan": _plan_meta(plan, max(int(epochs_run), 0), order_seed),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": order_sha,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_ap),
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "continuation_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "continuation_checkpoint_sha256": final_checkpoint_sha or None,
        "partial_checkpoint": C.track_rel(partial_path) if partial_path.exists() else None,
        "scores": tracker.scores,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "recipe": C.recipe_for_arm(arm, int(phase)),
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
            np.mean([r["epoch_seconds"] for r in curve_rows if r["epoch"] > started_from_epoch])
        )
        if epochs_run > started_from_epoch
        else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows if r.get("steps_per_second")]))
        if any(r.get("steps_per_second") for r in curve_rows)
        else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done phase={phase} arm={arm} status={status} epochs={epochs_run}/{final_epoch} "
        f"best_epoch={best_epoch} best_ap={best_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    parser.add_argument("--order-seed", type=int, default=-1)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": "lcrc-pcba-structural-gain-run-v1",
                "round_id": C.ROUND_ID,
                "phase": int(args.phase),
                "arm": args.arm,
                "seed": int(args.seed),
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        C.LC.assert_inherited_constants()
        task = D.load_task()
        source = CK.load_source_checkpoint() if int(args.phase) in (1, 2) else None
        resume = CK.load_checkpoint(args.resume) if args.resume else None
        if int(args.phase) != 2 and resume is not None:
            raise RuntimeError("--resume is only valid for Phase 2")
        run = train(
            task,
            phase=int(args.phase),
            arm=args.arm,
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            source=source,
            resume=resume,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            order_seed=None if int(args.order_seed) < 0 else int(args.order_seed),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": "lcrc-pcba-structural-gain-run-v1",
                "round_id": C.ROUND_ID,
                "phase": int(args.phase),
                "arm": args.arm,
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
        C.log(f"FAILED phase={args.phase} arm={args.arm}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
