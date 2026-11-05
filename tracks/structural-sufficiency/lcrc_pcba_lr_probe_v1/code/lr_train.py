"""Formal 12-epoch learning-rate probe arm (fixed LR, no scheduler).

Only the fixed learning rate differs between the two arms.  The historical
epoch-29 weights are loaded into a freshly built frozen LCRC-C1; AdamW moments
are re-initialised because the historical checkpoint contains no optimizer
state.  Both arms use the exact same 12-epoch row order and the same dropout
RNG seed, and every step goes through the already-gated ``PinnedStager``.

Official Test targets are unreachable: the label artifact has every Test row set
to NaN, ``lr_data.load_task`` refuses a finite Test label, and this file never
indexes ``task.split['test']``.
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

import lr_common as C  # noqa: E402
import lr_data  # noqa: E402
import lr_model  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "steps_per_epoch",
    "lr",
    "train_loss",
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
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "notes",
]


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


class Watchdog:
    """Arm a hard-stop flag at ``hard_seconds``; the loop saves and exits."""

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


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _write_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in CURVE_FIELDS})
    return C.sha256_file(path)


def _write_per_task(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ["epoch", "task", "ap", "observed", "positive", "negative"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


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
    model_info: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-lr-probe-checkpoint-v1",
        "round_id": C.ROUND_ID,
        "group": group,
        "group_name": C.GROUPS[group]["name"],
        "role": C.GROUPS[group]["role"],
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
        "learning_rate": float(C.GROUPS[group]["learning_rate"]),
        "recipe": C.recipe_for_group(group),
        "feature_spec_hash": task.spec.hash(),
        "norm_fingerprint": task.norm["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256,
        "label_artifact_sha256": task.label_artifact_sha256,
        "cache_sha256_manifest": C.HIST["cache_sha256"],
        "history": {
            "source_round": C.HIST["round"],
            "source_checkpoint": C.track_rel(C.HIST["checkpoint"]),
            "source_checkpoint_sha256": C.HIST["checkpoint_sha256"],
            "source_best_epoch": C.HIST["best_epoch"],
            "source_best_valid_ap": C.HIST["best_valid_ap"],
            "source_optimizer_state_available": False,
            "adamw_reinitialised": True,
            "not_seamless_continuation": True,
            "limitation": (
                "the historical checkpoint contains no AdamW moments or step count, so both arms "
                "start AdamW from zero state; this is a fresh-optimizer probe from the same weights, "
                "not a seamless continuation of the epoch-29 optimizer trajectory"
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
    tracker: Tracker,
) -> str:
    payload = {
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(best_ap),
        "checkpoint": C.track_rel(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "scores": tracker.scores,
        "tie_check": tracker.tie_check(),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_epochs = tuple(int(x) for x in eval_epochs)
    if eval_epochs != tuple(C.EVAL_EPOCHS):
        raise RuntimeError(f"eval epochs {eval_epochs} != frozen {C.EVAL_EPOCHS}")
    if int(max_epochs) != C.TOTAL_EPOCHS:
        raise RuntimeError(f"max_epochs {max_epochs} != frozen {C.TOTAL_EPOCHS}")

    C.seed_everything(C.SEED)
    model, model_info, load_audit = lr_model.build_from_checkpoint(
        task, checkpoint, device=device, group=group
    )
    # Explicit, common training-start RNG.  Both arms reset here, so dropout
    # masks are identical when the weights and batch order are identical.
    seed_info = C.seed_everything(C.SEED)
    rng_start = C.capture_rng_state()
    rng_start_hashes = C.rng_state_hashes(rng_start)

    plan = lr_data.make_order_plan(task.train_rows)
    order_summary = lr_data.order_summary(plan, n_train_rows=int(task.train_rows.size))
    isolation = lr_data.assert_order_isolation(
        plan, task.train_rows, task.valid_rows, task.test_rows
    )
    if not isolation["passed"]:
        raise RuntimeError(f"training order isolation gate failed: {isolation}")
    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"summary": order_summary, "isolation": isolation})
    order_sha = C.sha256_file(order_path)

    optimizer = lr_model.make_optimizer(model, learning_rate=float(C.GROUPS[group]["learning_rate"]))
    if not isinstance(optimizer, torch.optim.AdamW):
        raise RuntimeError("optimizer is not AdamW")
    if any(g.get("initial_lr", None) is not None for g in optimizer.param_groups):
        raise RuntimeError("unexpected scheduler state on a fresh optimizer")
    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    tracker = Tracker()
    clip = float(C.PROBE_RECIPE["gradient_clip"])
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

    best_state: dict[str, torch.Tensor] | None = None
    best_opt_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_rng: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_ap = -math.inf
    best_checkpoint_sha = ""
    best_selection_sha = ""
    final_checkpoint_sha = ""
    final_ap: float | None = None
    epochs_run = 0
    partial_epoch_discarded = False
    stop_reason = "COMPLETED_12_EPOCHS"
    grad_norm_max = 0.0
    clip_events = 0
    non_finite_events = 0
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    eval_completed: list[int] = []

    def update_best(epoch: int, report: Mapping[str, Any], rng_state: Mapping[str, Any]) -> None:
        nonlocal best_state, best_opt_state, best_metrics, best_rng, best_epoch, best_ap
        nonlocal best_checkpoint_sha, best_selection_sha
        best_state = _cpu_state(model)
        best_opt_state = lr_model.optimizer_state_cpu(optimizer)
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
            model_info=model_info,
            extra={"checkpoint_kind": "best_valid_ap"},
        )
        best_checkpoint_sha = _atomic_torch_save(payload, best_path)
        best_selection_sha = _save_selection(
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
                }
            )
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    # ---- epoch 0: load, evaluate, and snapshot the historical weights ----
    model.eval()
    t_eval = time.perf_counter()
    report0 = lr_model.evaluate_validation(model, task, device, reason="epoch_0_pre_update")
    valid_seconds0 = time.perf_counter() - t_eval
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
            "rng_cpu_sha256": rng0_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng0_hashes["torch_cuda_sha256"]),
            "notes": "epoch 0 = historical epoch-29 weights, before any optimizer update",
        }
    )
    _write_curve(curve_path, curve)
    _write_per_task(per_task_path, per_task_rows)
    C.log(
        f"[{group}] epoch 0 (pre-update) validAP {report0['ap']:.8f} "
        f"delta_hist {abs(report0['ap'] - C.HIST['best_valid_ap']):.3e} "
        f"tasks {report0['n_tasks_scored']} {valid_seconds0:.1f}s"
    )

    # ---- epochs 1..12 ----
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
        step_times: list[float] = []
        aborted = False
        batches = plan[epoch - 1]["batches"]
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(BM.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            logits = model(batch)
            if tuple(logits.shape) != (len(rows), C.N_TASKS):
                raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), C.N_TASKS)}")
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
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
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
            report = lr_model.evaluate_validation(
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
            f"validAP {valid_txt} "
            f"lr {row['lr']:.3e} train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
            f"step_p50 {row['step_ms_p50']:.2f}ms"
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

    # Capture the final stochastic state before any reload rebuilds the model.
    rng_end = C.capture_rng_state()
    rng_end_hashes = C.rng_state_hashes(rng_end)
    C.write_json(
        rng_path.with_suffix(".json"), {"start": rng_start_hashes, "end": rng_end_hashes, "epochs": rng_epoch_hashes}
    )
    torch.save(
        {
            "schema": "lcrc-pcba-lr-probe-rng-v1",
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

    # Final epoch-12 checkpoint (or the last completed state if incomplete).
    final_state = _cpu_state(model)
    final_opt_state = lr_model.optimizer_state_cpu(optimizer)
    final_rng = C.capture_rng_state()
    final_ap = None if not curve or curve[-1].get("valid_ap") is None else float(curve[-1]["valid_ap"])
    if curve and curve[-1]["epoch"] == max_epochs and curve[-1].get("valid_ap") is not None:
        # Reuse the last evaluation report's summary from selection/curve.
        final_ap = float(curve[-1]["valid_ap"])
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
        model_info=model_info,
        extra={
            "checkpoint_kind": "final_epoch_12" if epochs_run == max_epochs else "partial_stop",
            "status_at_save": "COMPLETED" if epochs_run == max_epochs else "INCOMPLETE",
            "stop_reason_at_save": stop_reason,
        },
    )
    final_checkpoint_sha = _atomic_torch_save(final_payload, final_path)

    if epochs_run < max_epochs:
        partial_payload = _checkpoint_payload(
            task=task,
            group=group,
            epoch=int(epochs_run),
            state=_cpu_state(model),
            optimizer_state=lr_model.optimizer_state_cpu(optimizer),
            valid_ap=final_ap,
            valid_report=None,
            rng_state=C.capture_rng_state(),
            order_summary=order_summary,
            initialization=load_audit,
            model_info=model_info,
            extra={
                "checkpoint_kind": "partial_stop",
                "status_at_save": "INCOMPLETE",
                "stop_reason_at_save": stop_reason,
                "partial_epoch_discarded": bool(partial_epoch_discarded),
            },
        )
        _atomic_torch_save(partial_payload, partial_path)

    # Independent reload of the selected best checkpoint on the GPU inference path.
    reload_model, _ = BM.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, device)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(best_payload["state_dict"], strict=True)
    reload_model.eval()
    reload_report = lr_model.evaluate_validation(
        reload_model, task, device, reason="reload_best_checkpoint"
    )
    reload_delta = abs(float(reload_report["ap"]) - float(best_ap))
    if reload_delta > 1e-4:
        raise RuntimeError(
            f"reloaded best checkpoint does not reproduce AP {best_ap}: "
            f"reloaded={reload_report['ap']} delta={reload_delta}"
        )

    total_seconds = float(time.perf_counter() - t_start)
    all_eval = set(eval_completed) == set(eval_epochs)
    if epochs_run == int(max_epochs) and all_eval and non_finite_events == 0:
        status = "COMPLETED"
        stop_reason = "COMPLETED_12_EPOCHS"
    else:
        status = "INCOMPLETE"
        if non_finite_events:
            stop_reason = "NON_FINITE_EVENT"
        elif all_eval and epochs_run == int(max_epochs):
            stop_reason = "COMPLETED_BUT_NON_FINITE"
        elif all_eval:
            stop_reason = stop_reason if stop_reason != "COMPLETED_12_EPOCHS" else "MISSING_EPOCH"
        else:
            stop_reason = stop_reason if stop_reason != "COMPLETED_12_EPOCHS" else "MISSING_EVAL_EPOCH"

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
        "schema": "lcrc-pcba-lr-probe-run-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "group": group,
        "group_name": C.GROUPS[group]["name"],
        "role": C.GROUPS[group]["role"],
        "status": status,
        "stop_reason": stop_reason,
        "resource_limited": bool(status != "COMPLETED"),
        "partial_epoch_discarded": bool(partial_epoch_discarded),
        "dataset": C.DATASET,
        "variant": "C1",
        "variant_definition": C.PC.MODEL_VARIANT_DEFINITION,
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
        "recipe": C.recipe_for_group(group),
        "seed_state": seed_info,
        "initialization_method": (
            "historical epoch-29 checkpoint loaded into a freshly built frozen LCRC-C1; "
            "AdamW moments re-initialised in both arms (historical checkpoint has no optimizer state)"
        ),
        "initialization": load_audit,
        "model_info": model_info,
        "history": {
            "source_round": C.HIST["round"],
            "source_checkpoint": C.track_rel(C.HIST["checkpoint"]),
            "source_checkpoint_sha256": C.HIST["checkpoint_sha256"],
            "source_best_epoch": C.HIST["best_epoch"],
            "source_best_valid_ap": C.HIST["best_valid_ap"],
            "source_optimizer_state_available": False,
            "adamw_reinitialised": True,
            "not_seamless_continuation": True,
            "limitation": (
                "the historical checkpoint contains no AdamW moments or step count; both arms start "
                "AdamW from zero state.  This is a fresh-optimizer probe from the same weights, not a "
                "seamless continuation of the epoch-29 optimizer trajectory."
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
                "schema": "lcrc-pcba-lr-probe-run-v1",
                "round_id": C.ROUND_ID,
                "group": args.group,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        return 2
    try:
        C.assert_inherited_constants()
        task = lr_data.load_task()
        checkpoint = lr_model.load_checkpoint()
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
                "schema": "lcrc-pcba-lr-probe-run-v1",
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
