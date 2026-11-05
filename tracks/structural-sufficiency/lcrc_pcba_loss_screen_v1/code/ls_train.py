"""Formal six-epoch loss-screen arm.

Both arms start from the same L-low-lr epoch-9 checkpoint, restore the same
AdamW moments and the same RNG state, and use the same pre-generated six-epoch
minibatch order.  The only difference is the loss:

* ``M``: historical ``pc_metrics.masked_bce_loss`` unchanged.
* ``T``: task-macro BCE with frozen full-Train ``n_t``.

Official Test targets are unreachable: the label artifact has every Test row
set to NaN, ``lr_data.load_task`` refuses a finite Test label, and this file
never indexes a Test row.
"""

from __future__ import annotations

import argparse
import csv
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

import ls_checkpoint as CK  # noqa: E402
import ls_common as C  # noqa: E402
import ls_losses as L  # noqa: E402
import lr_data  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "steps_per_epoch",
    "lr",
    "train_loss",
    "raw_masked_bce",
    "valid_ap",
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
    "epoch_molecules_per_second",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "strict_improve",
    "grad_norm_mean",
    "grad_norm_max",
    "grad_clip_events_epoch",
    "grad_clip_events_cumulative",
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


def _stratum(train_positive: int) -> str:
    if train_positive <= 100:
        return "sparse_le_100"
    if train_positive <= 1000:
        return "medium_101_1000"
    return "ample_gt_1000"


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _checkpoint_payload(
    *,
    task: Any,
    group: str,
    epoch: int,
    state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    valid_ap: float | None,
    valid_report: Mapping[str, Any] | None,
    rng_state: Mapping[str, Any],
    order_summary: Mapping[str, Any],
    initialization: Mapping[str, Any],
    optimizer_audit: Mapping[str, Any],
    model_info: Mapping[str, Any],
    n_task: np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-loss-screen-checkpoint-v1",
        "round_id": C.ROUND_ID,
        "group": group,
        "group_name": C.GROUPS[group]["name"],
        "role": C.GROUPS[group]["role"],
        "loss": C.GROUPS[group]["loss"],
        "dataset": C.DATASET,
        "variant": "C1",
        "seed": C.SEED,
        "n_outputs": C.N_TASKS,
        "batch_size": C.BATCH_SIZE,
        "epoch": int(epoch),
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "metric": "Validation AP (OGB official Evaluator)",
        "valid_report_summary": None
        if valid_report is None
        else {
            "ap": float(valid_report["ap"]),
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "state_dict": {k: v.detach().cpu().clone() for k, v in state.items()},
        "optimizer_state_dict": optimizer_state,
        "optimizer": "AdamW",
        "optimizer_state_audit": dict(optimizer_audit),
        "learning_rate": float(C.RECIPE["learning_rate"]),
        "recipe": C.recipe_for_group(group),
        "loss_definition": {
            "M": "pc_metrics.masked_bce_loss (historical, unchanged)",
            "T": "L_T = N/(B*T) * sum_{i,t}(m_it * ell_it / n_t)",
        }[group],
        "n_task": n_task.tolist(),
        "n_task_sha256": C.sha256_array(n_task),
        "feature_spec_hash": task.spec.hash(),
        "norm_fingerprint": task.norm["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256,
        "label_artifact_sha256": task.label_artifact_sha256,
        "cache_sha256_manifest": C.SOURCE["cache_sha256"],
        "history": {
            "source_round": C.SOURCE["round"],
            "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
            "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
            "source_epoch": int(C.SOURCE["best_epoch"]),
            "source_valid_ap": float(C.SOURCE["best_valid_ap"]),
            "source_optimizer_state_loaded": True,
            "adamw_reinitialised": False,
            "new_six_epoch_order": True,
            "not_bitwise_continuation_of_epoch_10_plus": True,
            "limitation": (
                "weights, AdamW moments and RNG start from the epoch-9 checkpoint, but the "
                "six-epoch batch order is a new fixed fork; this is not a bitwise replay of the "
                "historical epoch-10+ trajectory"
            ),
        },
        "initialization": dict(initialization),
        "model_info": dict(model_info),
        "training_order": {
            "global_plan_sha256": order_summary["global_plan_sha256"],
            "generator_state_sha256": order_summary["generator_state_sha256"],
            "seed": order_summary["seed"],
            "batch_size": order_summary["batch_size"],
            "epochs": order_summary["epochs"],
            "not_historical_epoch_10_plus_order": bool(
                order_summary.get("not_historical_epoch_10_plus_order", False)
            ),
        },
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _save_selection(
    path: Path,
    *,
    best_epoch: int,
    best_ap: float,
    checkpoint_path: Path,
    checkpoint_sha: str,
    tracker: C.Tracker,
) -> str:
    payload = {
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(best_ap),
        "checkpoint": C.track_rel(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "scores": tracker.scores,
        "tie_check": tracker.tie_check(),
        "role": "auxiliary best-Validation checkpoint; main comparison is fixed epoch 6",
        "updated_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(path, payload)
    return C.sha256_file(path)


def train(
    task: Any,
    checkpoint: Mapping[str, Any],
    *,
    group: str,
    out_dir: Path,
    device: torch.device,
    assigned_pool: str,
    soft_seconds: float,
    hard_seconds: float,
    max_epochs: int = C.TOTAL_EPOCHS,
    eval_epochs: Sequence[int] = C.EVAL_EPOCHS,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    gpu_occupancy_start = C.gpu_occupancy()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_epochs = tuple(int(x) for x in eval_epochs)
    if eval_epochs != tuple(C.EVAL_EPOCHS):
        raise RuntimeError(f"eval epochs {eval_epochs} != frozen {C.EVAL_EPOCHS}")
    if int(max_epochs) != C.TOTAL_EPOCHS:
        raise RuntimeError(f"max_epochs {max_epochs} != frozen {C.TOTAL_EPOCHS}")

    C.seed_everything(C.SEED)
    model, model_info, load_audit = CK.build_model_from_checkpoint(
        task, checkpoint, device=device, group=group
    )
    optimizer = CK.make_optimizer(model)
    optimizer_audit = CK.load_optimizer_state(
        optimizer, checkpoint["optimizer_state_dict"], model
    )
    rng_restore = C.restore_rng_state(checkpoint["rng_state"])
    rng_start = C.capture_rng_state()
    rng_start_hashes = C.rng_state_hashes(rng_start)
    source_rng_hashes = checkpoint["rng_state_hashes"]
    rng_matches_source = {
        "python_random": rng_start_hashes["python_random_sha256"]
        == source_rng_hashes["python_random_sha256"],
        "numpy": rng_start_hashes["numpy_sha256"] == source_rng_hashes["numpy_sha256"],
        "torch_cpu": rng_start_hashes["torch_cpu_sha256"]
        == source_rng_hashes["torch_cpu_sha256"],
        "torch_cuda": rng_start_hashes["torch_cuda_sha256"]
        == source_rng_hashes["torch_cuda_sha256"],
    }
    if not (rng_matches_source["python_random"] and rng_matches_source["numpy"] and rng_matches_source["torch_cpu"]):
        raise RuntimeError(f"RNG restore mismatch: {rng_matches_source}")

    plan = C.make_order_plan(task.train_rows, epochs=C.TOTAL_EPOCHS)
    order_summary = C.order_summary(plan, n_train_rows=int(task.train_rows.size))
    isolation = C.assert_order_isolation(
        plan, task.train_rows, task.valid_rows, task.test_rows
    )
    if not isolation["passed"]:
        raise RuntimeError(f"training order isolation gate failed: {isolation}")
    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"summary": order_summary, "isolation": isolation})
    order_sha = C.sha256_file(order_path)

    train_labels = task.labels[task.train_rows]
    n_task = L.task_counts(train_labels)
    n_pos_task = (train_labels == 1.0).sum(axis=0).astype(np.int64)
    n_task_t = torch.as_tensor(n_task, dtype=torch.float32, device=device)

    stager = PinnedStager(device, depth=2)
    watchdog = C.Watchdog(hard_seconds)
    tracker = C.Tracker()
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    selection_path = out_dir / "selection.json"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / "checkpoint_final.pt"
    partial_path = out_dir / "checkpoint_partial.pt"
    rng_path = out_dir / "rng_state.pt"
    first_step_path = out_dir / "first_step.json"
    first_step_logits_path = out_dir / "first_step_logits.pt"

    best_state: dict[str, torch.Tensor] | None = None
    best_opt_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_rng: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_ap = -math.inf
    best_checkpoint_sha = ""
    final_checkpoint_sha = ""
    final_ap: float | None = None
    epochs_run = 0
    partial_epoch_discarded = False
    stop_reason = "COMPLETED_6_EPOCHS"
    grad_norm_max = 0.0
    clip_events = 0
    non_finite_events = 0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    eval_completed: list[int] = []
    first_step_recorded = False

    def update_best(epoch: int, report: Mapping[str, Any], rng_state: Mapping[str, Any]) -> None:
        nonlocal best_state, best_opt_state, best_metrics, best_rng, best_epoch, best_ap
        nonlocal best_checkpoint_sha
        best_state = _cpu_state(model)
        best_opt_state = CK.optimizer_state_cpu(optimizer)
        best_metrics = dict(report)
        best_rng = rng_state
        best_epoch = int(epoch)
        best_ap = float(report["ap"])
        payload = _checkpoint_payload(
            task=task,
            group=group,
            epoch=int(epoch),
            state=best_state,
            optimizer_state=best_opt_state,
            valid_ap=best_ap,
            valid_report=best_metrics,
            rng_state=best_rng,
            order_summary=order_summary,
            initialization=load_audit,
            optimizer_audit=optimizer_audit,
            model_info=model_info,
            n_task=n_task,
            extra={"checkpoint_kind": "best_valid_ap_auxiliary"},
        )
        best_checkpoint_sha = C.atomic_torch_save(payload, best_path)
        _save_selection(
            selection_path,
            best_epoch=best_epoch,
            best_ap=best_ap,
            checkpoint_path=best_path,
            checkpoint_sha=best_checkpoint_sha,
            tracker=tracker,
        )

    def record_eval(epoch: int, report: Mapping[str, Any], valid_seconds: float) -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        eval_completed.append(int(epoch))
        for t, ap in enumerate(report["per_task_ap"]):
            per_task_rows.append(
                {
                    "epoch": int(epoch),
                    "task": int(t),
                    "ap": "" if ap is None else float(ap),
                    "observed": int(report["observed"][t]),
                    "positive": int(report["positive"][t]),
                    "negative": int(report["negative"][t]),
                    "train_observed": int(n_task[t]),
                    "train_positive": int(n_pos_task[t]),
                    "stratum": _stratum(int(n_pos_task[t])),
                }
            )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    # ---- epoch 0: loaded epoch-9 weights, before any update ----
    model.eval()
    t_eval = time.perf_counter()
    report0 = CK.evaluate_validation(model, task, device, reason="epoch_0_pre_update")
    valid_seconds0 = time.perf_counter() - t_eval
    delta0 = abs(float(report0["ap"]) - float(C.SOURCE["best_valid_ap"]))
    if delta0 > 1e-4:
        raise RuntimeError(
            f"epoch-0 AP {report0['ap']} does not reproduce {C.SOURCE['best_valid_ap']} "
            f"(delta {delta0})"
        )
    strict0 = tracker.consider(0, float(report0["ap"]))
    rng0 = C.capture_rng_state()
    rng0_hashes = C.rng_state_hashes(rng0)
    rng_epoch_hashes.append({"epoch": 0, **rng0_hashes})
    if strict0:
        update_best(0, report0, rng0)
    record_eval(0, report0, valid_seconds0)
    curve.append(
        {
            "epoch": 0,
            "evaluated": True,
            "steps_per_epoch": int(order_summary["steps_per_epoch"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": None,
            "raw_masked_bce": None,
            "valid_ap": float(report0["ap"]),
            "n_tasks_with_observed_labels": int(report0["n_tasks_with_observed_labels"]),
            "n_tasks_scored": int(report0["n_tasks_scored"]),
            "train_seconds": 0.0,
            "valid_seconds": float(valid_seconds0),
            "epoch_seconds": float(valid_seconds0),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "step_ms_mean": None,
            "step_ms_p50": None,
            "step_ms_p90": None,
            "steps_per_second": None,
            "train_molecules_per_second": None,
            "epoch_molecules_per_second": None,
            "best_epoch_so_far": int(tracker.best_epoch),
            "best_valid_ap_so_far": float(tracker.best_metric),
            "is_best": bool(strict0),
            "strict_improve": bool(strict0),
            "grad_norm_mean": None,
            "grad_norm_max": None,
            "grad_clip_events_epoch": None,
            "grad_clip_events_cumulative": None,
            "rng_cpu_sha256": rng0_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng0_hashes["torch_cuda_sha256"]),
            "notes": "epoch 0 = epoch-9 checkpoint weights and AdamW state, before any update",
        }
    )
    _write_curve(curve_path, curve)
    _write_per_task(per_task_path, per_task_rows)
    C.log(
        f"[{group}] epoch 0 (pre-update) validAP {report0['ap']:.8f} "
        f"delta_hist {delta0:.3e} tasks {report0['n_tasks_scored']} {valid_seconds0:.1f}s"
    )

    # ---- epochs 1..6 ----
    for epoch in range(1, int(max_epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        losses: list[float] = []
        raw_losses: list[float] = []
        gnorms: list[float] = []
        step_times: list[float] = []
        epoch_clip_events = 0
        aborted = False
        batches = plan[epoch - 1]["batches"]
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch_cpu = BM.collate(task.cache, rows.tolist())
            y_cpu = np.asarray(task.labels[rows], dtype=np.float32)
            batch = stager.batch(batch_cpu)
            y = stager.labels(task.labels[rows])
            logits = model(batch)
            if tuple(logits.shape) != (len(rows), C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), C.N_TASKS)}")
            if C.GROUPS[group]["loss"] == "masked_bce":
                loss = L.masked_bce_loss(logits, y)
                raw_loss = loss.detach()
            else:
                loss, raw_loss = L.task_macro_bce_loss(logits, y, n_task_t)
            if not bool(torch.isfinite(loss)):
                non_finite_events += 1
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}: {float(loss)}")

            if not first_step_recorded:
                first_step_recorded = True
                logits_cpu = logits.detach().cpu()
                torch.save(
                    {"rows_sha256": C.sha256_int64(rows), "labels_sha256": C.sha256_array(y_cpu), "logits": logits_cpu},
                    first_step_logits_path,
                )
                C.write_json(
                    first_step_path,
                    {
                        "epoch": int(epoch),
                        "step": int(step),
                        "rows_sha256": C.sha256_int64(rows),
                        "labels_sha256": C.sha256_array(y_cpu),
                        "loss": float(loss.detach().cpu().item()),
                        "raw_masked_bce": float(raw_loss.detach().cpu().item()),
                        "logits_sha256": C.sha256_array(logits_cpu.numpy()),
                        "logits_shape": list(logits_cpu.shape),
                        "rng_before_step": C.rng_state_hashes(rng_start),
                        "rng_after_forward": C.rng_state_hashes(C.capture_rng_state()),
                        "batch_size": int(len(rows)),
                        "n_observed": int(np.isfinite(y_cpu).sum()),
                    },
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            if not math.isfinite(gnorm):
                non_finite_events += 1
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} step {step}")
            if gnorm > clip:
                clip_events += 1
                epoch_clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            raw_losses.append(float(raw_loss.detach().cpu().item()))
            gnorms.append(float(gnorm))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        train_seconds = float(time.perf_counter() - t_epoch)
        if aborted:
            partial_epoch_discarded = True
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{group}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        steps_arr = np.asarray(step_times, dtype=np.float64)
        step_p50 = float(np.percentile(steps_arr, 50)) if steps_arr.size else float("nan")
        step_mean = float(steps_arr.mean()) if steps_arr.size else float("nan")
        step_p90 = float(np.percentile(steps_arr, 90)) if steps_arr.size else float("nan")
        steps_per_sec = float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan")
        train_mol_per_sec = (
            float(len(task.train_rows) / train_seconds) if train_seconds > 0 else float("nan")
        )

        report: dict[str, Any] | None = None
        valid_seconds = 0.0
        strict = False
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            report = CK.evaluate_validation(
                model, task, device, reason=f"epoch_{epoch}_official_validation"
            )
            valid_seconds = float(time.perf_counter() - t_eval)
            strict = tracker.consider(epoch, float(report["ap"]))
            rng_eval = C.capture_rng_state()
            rng_eval_hashes = C.rng_state_hashes(rng_eval)
            rng_epoch_hashes.append({"epoch": int(epoch), **rng_eval_hashes})
            if strict:
                update_best(epoch, report, rng_eval)
            record_eval(epoch, report, valid_seconds)
            model.train()
        else:
            rng_now = C.capture_rng_state()
            rng_now_hashes = C.rng_state_hashes(rng_now)
            rng_epoch_hashes.append({"epoch": int(epoch), **rng_now_hashes})

        epoch_seconds = float(time.perf_counter() - t_epoch)
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "steps_per_epoch": int(len(batches)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "raw_masked_bce": float(np.mean(raw_losses)) if raw_losses else float("nan"),
            "valid_ap": None if report is None else float(report["ap"]),
            "n_tasks_with_observed_labels": None
            if report is None
            else int(report["n_tasks_with_observed_labels"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "train_seconds": float(train_seconds),
            "valid_seconds": float(valid_seconds),
            "epoch_seconds": float(epoch_seconds),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "step_ms_mean": float(step_mean * 1e3),
            "step_ms_p50": float(step_p50 * 1e3),
            "step_ms_p90": float(step_p90 * 1e3),
            "steps_per_second": float(steps_per_sec),
            "train_molecules_per_second": float(train_mol_per_sec),
            "epoch_molecules_per_second": (
                float(len(task.train_rows) / epoch_seconds) if epoch_seconds > 0 else float("nan")
            ),
            "best_epoch_so_far": int(tracker.best_epoch),
            "best_valid_ap_so_far": float(tracker.best_metric),
            "is_best": bool(strict),
            "strict_improve": bool(strict),
            "grad_norm_mean": float(np.mean(gnorms)) if gnorms else float("nan"),
            "grad_norm_max": float(np.max(gnorms)) if gnorms else float("nan"),
            "grad_clip_events_epoch": int(epoch_clip_events),
            "grad_clip_events_cumulative": int(clip_events),
            "rng_cpu_sha256": rng_epoch_hashes[-1]["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_epoch_hashes[-1]["torch_cuda_sha256"]),
            "notes": "" if report is not None else "validation not scheduled at this epoch",
        }
        curve.append(row)
        epochs_run = int(epoch)
        _write_curve(curve_path, curve)
        _write_per_task(per_task_path, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{group}] epoch {epoch}/{max_epochs} loss {row['train_loss']:.6f} "
            f"raw {row['raw_masked_bce']:.6f} validAP {valid_txt} "
            f"lr {row['lr']:.3e} train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
            f"step_p50 {row['step_ms_p50']:.2f}ms clips {epoch_clip_events}"
            f"{' *best*' if strict else ''}"
        )
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    if tracker.best_epoch is None or best_state is None or best_opt_state is None:
        raise RuntimeError("training ended without selecting a Validation checkpoint")
    tie = tracker.tie_check()
    if not tie["consistent"]:
        raise RuntimeError(f"selection re-derivation inconsistent: {tie}")

    rng_end = C.capture_rng_state()
    rng_end_hashes = C.rng_state_hashes(rng_end)
    C.write_json(
        rng_path.with_suffix(".json"),
        {"start": rng_start_hashes, "end": rng_end_hashes, "epochs": rng_epoch_hashes},
    )
    torch.save(
        {
            "schema": "lcrc-pcba-loss-screen-rng-v1",
            "round_id": C.ROUND_ID,
            "group": group,
            "seed": C.SEED,
            "start": rng_start,
            "end": rng_end,
            "start_hashes": rng_start_hashes,
            "end_hashes": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    final_state = _cpu_state(model)
    final_opt_state = CK.optimizer_state_cpu(optimizer)
    final_rng = C.capture_rng_state()
    final_ap = None if not curve or curve[-1].get("valid_ap") is None else float(curve[-1]["valid_ap"])
    final_payload = _checkpoint_payload(
        task=task,
        group=group,
        epoch=int(epochs_run),
        state=final_state,
        optimizer_state=final_opt_state,
        valid_ap=final_ap,
        valid_report=None,
        rng_state=final_rng,
        order_summary=order_summary,
        initialization=load_audit,
        optimizer_audit=optimizer_audit,
        model_info=model_info,
        n_task=n_task,
        extra={
            "checkpoint_kind": "final_epoch_6" if epochs_run == max_epochs else "partial_stop",
            "status_at_save": "COMPLETED" if epochs_run == max_epochs else "INCOMPLETE",
            "stop_reason_at_save": stop_reason,
        },
    )
    final_checkpoint_sha = C.atomic_torch_save(final_payload, final_path)

    if epochs_run < max_epochs:
        partial_payload = _checkpoint_payload(
            task=task,
            group=group,
            epoch=int(epochs_run),
            state=_cpu_state(model),
            optimizer_state=CK.optimizer_state_cpu(optimizer),
            valid_ap=final_ap,
            valid_report=None,
            rng_state=C.capture_rng_state(),
            order_summary=order_summary,
            initialization=load_audit,
            optimizer_audit=optimizer_audit,
            model_info=model_info,
            n_task=n_task,
            extra={
                "checkpoint_kind": "partial_stop",
                "status_at_save": "INCOMPLETE",
                "stop_reason_at_save": stop_reason,
                "partial_epoch_discarded": bool(partial_epoch_discarded),
            },
        )
        C.atomic_torch_save(partial_payload, partial_path)

    # Independent reload of the selected best checkpoint on the GPU inference path.
    reload_model, _ = BM.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, device)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(best_payload["state_dict"], strict=True)
    reload_model.eval()
    reload_report = CK.evaluate_validation(
        reload_model, task, device, reason="reload_best_checkpoint"
    )
    reload_delta = abs(float(reload_report["ap"]) - float(best_ap))
    if reload_delta > 1e-4:
        raise RuntimeError(
            f"reloaded best checkpoint does not reproduce AP {best_ap}: "
            f"reloaded={reload_report['ap']} delta={reload_delta}"
        )

    total_seconds = float(time.perf_counter() - t_start)
    gpu_occupancy_end = C.gpu_occupancy()
    all_eval = set(eval_completed) == set(eval_epochs)
    if epochs_run == int(max_epochs) and all_eval and non_finite_events == 0:
        status = "COMPLETED"
        stop_reason = "COMPLETED_6_EPOCHS"
    else:
        status = "INCOMPLETE"
        if non_finite_events:
            stop_reason = "NON_FINITE_EVENT"
        elif not all_eval:
            stop_reason = stop_reason if stop_reason != "COMPLETED_6_EPOCHS" else "MISSING_EVAL_EPOCH"
        else:
            stop_reason = stop_reason if stop_reason != "COMPLETED_6_EPOCHS" else "MISSING_EPOCH"

    curve_sha = _write_curve(curve_path, curve)
    per_task_sha = _write_per_task(per_task_path, per_task_rows)
    selection_sha = _save_selection(
        selection_path,
        best_epoch=int(best_epoch),
        best_ap=float(best_ap),
        checkpoint_path=best_path,
        checkpoint_sha=best_checkpoint_sha,
        tracker=tracker,
    )

    run = {
        "schema": "lcrc-pcba-loss-screen-run-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "group": group,
        "group_name": C.GROUPS[group]["name"],
        "role": C.GROUPS[group]["role"],
        "loss": C.GROUPS[group]["loss"],
        "status": status,
        "stop_reason": stop_reason,
        "resource_limited": bool(status != "COMPLETED"),
        "partial_epoch_discarded": bool(partial_epoch_discarded),
        "dataset": C.DATASET,
        "variant": "C1",
        "variant_definition": "LCRC-C1 / NO_BOND_MEAN",
        "seed": int(C.SEED),
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_info["parameters"]),
        "batch_size": int(C.BATCH_SIZE),
        "max_epochs": int(max_epochs),
        "epochs_run": int(epochs_run),
        "eval_epochs_requested": list(eval_epochs),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(all_eval),
        "assigned_pool": assigned_pool,
        "actual_device": str(device),
        "environment": C.environment(),
        "gpu_occupancy_start": gpu_occupancy_start,
        "gpu_occupancy_end": gpu_occupancy_end,
        "recipe": C.recipe_for_group(group),
        "loss_definition": {
            "M": "pc_metrics.masked_bce_loss (historical, unchanged)",
            "T": "L_T = N/(B*T) * sum_{i,t}(m_it * ell_it / n_t)",
        }[group],
        "n_task": n_task.tolist(),
        "n_task_sha256": C.sha256_array(n_task),
        "n_task_sum": int(n_task.sum()),
        "n_pos_task": n_pos_task.tolist(),
        "seed_state": {
            "seed": int(C.SEED),
            "note": "global RNG is restored from the source checkpoint after model/optimizer load",
        },
        "rng_restore": rng_restore,
        "rng_start_hashes": rng_start_hashes,
        "rng_matches_source": {k: bool(v) for k, v in rng_matches_source.items()},
        "initialization_method": (
            "epoch-9 checkpoint weights plus the same checkpoint's AdamW moments restored; "
            "same RNG start; new fixed six-epoch batch order"
        ),
        "initialization": load_audit,
        "optimizer_state": optimizer_audit,
        "model_info": model_info,
        "history": {
            "source_round": C.SOURCE["round"],
            "source_checkpoint": C.track_rel(C.SOURCE["checkpoint"]),
            "source_checkpoint_sha256": C.SOURCE["checkpoint_sha256"],
            "source_epoch": int(C.SOURCE["best_epoch"]),
            "source_valid_ap": float(C.SOURCE["best_valid_ap"]),
            "source_optimizer_state_loaded": True,
            "adamw_reinitialised": False,
            "new_six_epoch_order": True,
            "not_bitwise_continuation_of_epoch_10_plus": True,
            "limitation": (
                "weights, AdamW moments and RNG start from the epoch-9 checkpoint, but the "
                "six-epoch batch order is a new fixed fork; this is not a bitwise replay of the "
                "historical epoch-10+ trajectory"
            ),
        },
        "data": task.data_audit,
        "cache_identity": task.cache_identity,
        "training_order": order_summary,
        "training_order_isolation": isolation,
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": order_sha,
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(best_ap),
        "best_valid_tasks_scored": int(best_metrics["n_tasks_scored"]),
        "best_valid_tasks_with_observed_labels": int(best_metrics["n_tasks_with_observed_labels"]),
        "best_checkpoint": C.track_rel(best_path),
        "best_checkpoint_sha256": best_checkpoint_sha,
        "best_checkpoint_role": "auxiliary only; main comparison is fixed epoch 6",
        "final_epoch": int(epochs_run),
        "final_valid_ap": final_ap,
        "final_checkpoint": C.track_rel(final_path),
        "final_checkpoint_sha256": final_checkpoint_sha,
        "partial_checkpoint": C.track_rel(partial_path) if partial_path.exists() else None,
        "partial_checkpoint_sha256": C.sha256_file(partial_path) if partial_path.exists() else None,
        "selection": C.track_rel(selection_path),
        "selection_sha256": selection_sha,
        "scores": tracker.scores,
        "tie_check": tie,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "rng_epoch_hashes": rng_epoch_hashes,
        "first_step": C.read_json(first_step_path) if first_step_path.exists() else None,
        "first_step_logits": C.track_rel(first_step_logits_path)
        if first_step_logits_path.exists()
        else None,
        "reload_best_valid_ap": float(reload_report["ap"]),
        "reload_best_valid_delta": float(reload_delta),
        "reload_best_valid_tasks_scored": int(reload_report["n_tasks_scored"]),
        "gpu_seconds": float(total_seconds),
        "gpu_hours": float(total_seconds / 3600.0),
        "data_load_seconds": float(task.load_seconds),
        "train_seconds": float(sum(float(r["train_seconds"]) for r in curve)),
        "valid_seconds": float(sum(float(r["valid_seconds"]) for r in curve)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve])) if curve else 0.0,
        "steps_per_second_mean": float(
            np.mean([r["steps_per_second"] for r in curve if r.get("steps_per_second") is not None])
        )
        if any(r.get("steps_per_second") is not None for r in curve)
        else 0.0,
        "step_ms_p50_mean": float(
            np.mean([r["step_ms_p50"] for r in curve if r.get("step_ms_p50") is not None])
        )
        if any(r.get("step_ms_p50") is not None for r in curve)
        else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "grad_clip_events": int(clip_events),
        "grad_norm_max": float(grad_norm_max),
        "non_finite_events": int(non_finite_events),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "budget": C.BUDGET,
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed",
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done group={group} status={status} best_epoch={best_epoch} best_ap={best_ap:.8f} "
        f"final_epoch={epochs_run} final_ap={final_ap} gpu_hours={run['gpu_hours']:.6f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(C.GROUPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assigned-pool", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--soft-seconds", type=float, default=float(C.BUDGET["per_group_soft_seconds"]))
    parser.add_argument("--hard-seconds", type=float, default=float(C.BUDGET["per_group_hard_seconds"]))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": "lcrc-pcba-loss-screen-run-v1",
                "round_id": C.ROUND_ID,
                "group": args.group,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        C.assert_inherited_constants()
        task = lr_data.load_task()
        checkpoint = CK.load_source_checkpoint()
        run = train(
            task,
            checkpoint,
            group=args.group,
            out_dir=out_dir,
            device=device,
            assigned_pool=args.assigned_pool or str(C.GROUPS[args.group]["assigned_pool"]),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001 - recorded for the failure table
        C.write_json(
            out_dir / "run.json",
            {
                "schema": "lcrc-pcba-loss-screen-run-v1",
                "round_id": C.ROUND_ID,
                "group": args.group,
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"FAILED group={args.group}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
