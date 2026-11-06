"""Unified from-scratch trainer for A0/A1/A2/A3/COMBO (seed 0, 40 epochs).

One recipe for every arm: official Train only, batch 128, AdamW, 1-epoch
warmup 3e-5 -> 3e-4 then per-step cosine to 1e-5 over the frozen 109,520
updates, dropout 0.10, grad clip 5.0, FP32, observed-entry masked BCE, no early
stopping, fixed epoch-40 comparison.  Validation AP is the official OGB
macro AP over the 127 scorable tasks at epochs 10/20/30/40.

The batch order is the frozen seed-0 40-epoch plan (`06f43729...`); the A0 run
must reproduce the historical A initialisation and plan bitwise.
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

import ep_common as C  # noqa: E402
import ep_checkpoint as CK  # noqa: E402
import ep_data as D  # noqa: E402
import ep_model as M  # noqa: E402
import nmp_model as NM  # noqa: E402
import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "steps_per_epoch",
    "learning_rate",
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
    "valid_positive_stratum",
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


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


def _plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(seed),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode()),
    }


def _strata(task: Any) -> tuple[list[str], dict[str, Any]]:
    counts = D.task_counts(task)
    valid_pos = counts["valid"]["positive"]
    strata = [
        "lt5" if p < 5 else "s5_19" if p < 20 else "ge20" for p in valid_pos
    ]
    return strata, counts


def train(
    task: Any,
    *,
    arm: str,
    modules: tuple[str, ...],
    seed: int,
    out_dir: Path,
    device: torch.device,
    final_epoch: int,
    eval_epochs: Sequence[int],
    soft_seconds: float,
    hard_seconds: float,
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_epochs = tuple(sorted(int(e) for e in eval_epochs))
    if min(eval_epochs) < 1 or max(eval_epochs) > int(final_epoch) or int(final_epoch) not in eval_epochs:
        raise ValueError(f"eval epochs {eval_epochs} must lie in 1..{final_epoch} and include the final epoch")

    features = D.load_features(task, modules)
    ref, ref_info, base_init = NM.build_reference(task.spec, task.norm, C.N_TASKS, int(seed))
    model, model_audit = M.build_model(
        arm,
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=int(seed),
        device=device,
        base_init_state=base_init,
        combine=",".join(modules) if arm == "COMBO" else "",
    )
    if model_audit["base_init_fingerprint"] != C.HISTORICAL_PARITY[0]["base_init_fingerprint"]:
        raise RuntimeError("base C1/A initialisation fingerprint drifted from the historical seed-0 value")
    optimizer, opt_audit = NM.make_optimizer_scratch(model)

    plan = D.make_plan40(task, seed=int(seed))
    plan_audit = D.plan_summary(plan, task)
    if plan[0]["global_plan_sha256"] != C.HISTORICAL_PARITY[0]["global_plan_sha256"]:
        raise RuntimeError(f"batch-order plan {plan[0]['global_plan_sha256']} != frozen seed-0 plan")
    if _plan_meta(plan, C.TOTAL_EPOCHS, int(seed))["prefix_sha256"] != C.HISTORICAL_PARITY[0][
        "consumed40_prefix_sha256"
    ]:
        raise RuntimeError("40-epoch batch-order prefix hash != frozen seed-0 plan")

    if resume is not None:
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        if global_steps_run != int(resume["epoch"]) * C.STEPS_PER_EPOCH:
            raise RuntimeError("resume checkpoint global_step is not epoch-aligned")
        resume_audit = CK.restore_for_resume(
            model, optimizer, resume, arm=arm, modules=modules, seed=int(seed)
        )
        if tuple(int(x) for x in resume.get("modules", ())) != modules:
            raise RuntimeError("resume module set mismatch")
        seed_info = None
        rng_hashes = NM.restore_rng_state(resume["rng_state"])
        if rng_hashes["combined_sha256"] != (resume.get("rng_state_hashes") or {}).get("combined_sha256"):
            raise RuntimeError("restored RNG state differs from the checkpoint hashes")
        saved_plan = resume.get("batch_plan") or {}
        if saved_plan.get("global_plan_sha256") != plan[0]["global_plan_sha256"]:
            raise RuntimeError("resume batch plan mismatch")
    else:
        start_epoch = 1
        global_steps_run = 0
        resume_audit = None
        # Historical from-scratch order: build model/optimizer/plan first, then
        # seed every stochastic stream immediately before the first forward.
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    rng_start_hashes = dict(rng_hashes)

    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": _plan_meta(plan, C.TOTAL_EPOCHS, int(seed)), **plan_audit})

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.SCRATCH_RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    final_path = out_dir / f"checkpoint_epoch{final_epoch}.pt"
    partial_path = out_dir / "checkpoint_partial.pt"
    rng_path = out_dir / "rng_state.pt"

    valid_strata, counts = _strata(task)
    train_counts = counts["train"]
    fixed_reports: dict[int, dict[str, Any]] = {}
    eval_completed: list[int] = []
    epochs_run = int(start_epoch) - 1
    gpu_peak_alloc = gpu_peak_reserved = 0
    clip_events_total = 0
    non_finite_events = 0
    partial_epoch_discarded = False
    stop_reason = "RUNNING"
    final_checkpoint_sha = ""
    train_eval_summary: dict[str, Any] | None = None

    for epoch in range(int(start_epoch), int(final_epoch) + 1):
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
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            NM.set_lrs(optimizer, step=int(global_steps_run))
            t0 = time.perf_counter()
            batch = stager.batch(D.augmented_collate(task, rows, features))
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
            del batch, y, logits, loss
        if aborted:
            partial_epoch_discarded = True
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{arm}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        steps_arr = np.asarray(step_times, dtype=np.float64)
        report: dict[str, Any] | None = None
        valid_seconds = 0.0
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            report = D.evaluate_validation(model, task, features, device, reason=f"epoch_{epoch}_official_validation")
            valid_seconds = float(time.perf_counter() - t_eval)
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
                        "valid_positive_stratum": valid_strata[t],
                    }
                )
            if epoch == final_epoch:
                t_tr = time.perf_counter()
                train_report = D.evaluate_train(model, task, features, device)
                train_eval_path = out_dir / f"train_eval_epoch{epoch}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "arm": arm,
                        "modules": list(modules),
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_ap": None if report is None else float(report["ap"]),
            "valid_bce": None if report is None else float(report["valid_bce"]),
            "n_tasks_with_observed_labels": None if report is None else int(report["n_tasks_with_observed_labels"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "train_seconds": train_seconds,
            "valid_seconds": valid_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "step_ms_mean": float(steps_arr.mean() * 1e3),
            "step_ms_p50": float(np.percentile(steps_arr, 50) * 1e3),
            "step_ms_p90": float(np.percentile(steps_arr, 90) * 1e3),
            "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan"),
            "train_molecules_per_second": float(len(task.train_rows) / train_seconds) if train_seconds > 0 else float("nan"),
            "grad_norm_max_epoch": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean_epoch": float(np.mean(grad_norms)) if grad_norms else None,
            "clip_events_epoch": int(clip_events_epoch),
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
            "notes": "" if report is not None else "validation not scheduled at this epoch",
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{arm}] epoch {epoch}/{final_epoch} loss {row['train_loss']:.6f} validAP {valid_txt} "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.1f}s step_p50 {row['step_ms_p50']:.2f}ms"
        )

        if epoch == final_epoch and epochs_run == final_epoch:
            payload = CK.checkpoint_payload(
                arm=arm,
                modules=modules,
                seed=int(seed),
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=_plan_meta(plan, int(epoch), int(seed)),
                model_audit=model_audit,
                features=features.hashes,
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"final_epoch{final_epoch}",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": None if train_eval_summary is None else dict(train_eval_summary),
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
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED_40_EPOCHS"
    elif non_finite_events:
        stop_reason = "NON_FINITE_EVENT"
    elif not stop_reason or stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end = C.capture_rng_state()
    torch.save(
        {
            "schema": "lcrc-pcba-explicit-performance-rng-v1",
            "round_id": C.ROUND_ID,
            "arm": arm,
            "modules": list(modules),
            "seed": int(seed),
            "start": rng_start_hashes,
            "end": C.rng_state_hashes(rng_end),
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    if not completed:
        payload = CK.checkpoint_payload(
            arm=arm,
            modules=modules,
            seed=int(seed),
            epoch=int(epochs_run),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=rng_end,
            plan=_plan_meta(plan, int(max(epochs_run, 0)), int(seed)),
            model_audit=model_audit,
            features=features.hashes,
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

    curve_sha = _write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "arm_label": C.arm_label(arm, modules),
        "modules": list(modules),
        "seed": int(seed),
        "order_seed": int(seed),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["total_parameters"]),
        "base_parameters": int(model_audit["base_parameters"]),
        "extra_parameters": int(model_audit["extra_parameters"]),
        "start_epoch": int(start_epoch),
        "final_epoch": int(final_epoch),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(set(eval_completed) == set(eval_epochs)),
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "train_eval_final": train_eval_summary,
        "task_counts": counts,
        "valid_positive_strata": {str(t): valid_strata[t] for t in range(C.N_TASKS)},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "reference_model_info": ref_info,
        "batch_plan": _plan_meta(plan, max(int(epochs_run), 0), int(seed)),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": C.sha256_file(order_path),
        "features": features.as_audit(),
        "checkpoint_epoch40": C.track_rel(final_path) if final_path.exists() else None,
        "checkpoint_epoch40_sha256": final_checkpoint_sha or None,
        "partial_checkpoint": C.track_rel(partial_path) if partial_path.exists() else None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": C.rng_state_hashes(rng_end),
        "seed_state": seed_info,
        "recipe": C.recipe_for(arm, modules),
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
        f"done arm={arm} modules={modules} status={status} epochs={epochs_run}/{final_epoch} "
        f"ap40={fixed_reports.get(final_epoch, {}).get('ap') if fixed_reports else None} "
        f"gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(C.ARM_ORDER) + ["COMBO"])
    parser.add_argument("--combine", default="")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--final-epoch", type=int, default=C.FINAL_EPOCH)
    parser.add_argument("--eval-epochs", default="10,20,30,40")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        modules = C.arm_modules(args.arm, args.combine)
    except Exception as exc:  # noqa: BLE001
        C.log(f"FAILED argument check: {exc}")
        return 1
    eval_epochs = tuple(int(x) for x in args.eval_epochs.split(",") if x.strip())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "arm": args.arm,
                "modules": list(modules),
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
        task = D.load_task()
        resume = CK.load_checkpoint(args.resume) if args.resume else None
        run = train(
            task,
            arm=args.arm,
            modules=modules,
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            final_epoch=int(args.final_epoch),
            eval_epochs=eval_epochs,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            resume=resume,
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "arm": args.arm,
                "modules": list(modules),
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
        C.log(f"FAILED arm={args.arm}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
