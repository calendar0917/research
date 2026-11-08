"""Formal from-scratch training for the LCRC-W2 TopRank round.

One model, one seed, 40 epochs, one trajectory.  The model, data, batch order,
optimiser, LR schedule, dropout, clipping and precision are inherited unchanged
from the backbone-scaling W2 run; the only difference is

    loss = masked_bce_loss(logits, y) + lambda_rank * rank_loss(logits, y)

with ``lambda_rank`` frozen beforehand by ``tr_audit.py`` from 16 Train batches
using the historical W2 best-Validation checkpoint.  The formal model is always
built from the seed-0 initialisation; the calibration checkpoint is never used
as an initialisation.

No official Test target is read, indexed or scored here.
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

import tr_common as C  # noqa: E402
import tr_loss as L  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_loss_total",
    "train_bce",
    "train_rank",
    "train_lambda_rank_times_rank",
    "valid_ap",
    "valid_bce",
    "n_tasks_scored",
    "n_tasks_with_observed_labels",
    "epoch_train_seconds",
    "epoch_valid_seconds",
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
    "rank_active_tasks_mean",
    "rank_active_tasks_median",
    "rank_active_task_fraction_mean",
    "rank_no_positive_task_fraction_mean",
    "rank_no_negative_task_fraction_mean",
    "rank_implicit_pairs_mean",
    "rank_positive_coverage_mean",
    "rank_negative_coverage_mean",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
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
    "rank_active_batches",
    "rank_active_fraction",
]
RANK_STAT_KEYS = (
    "n_active_tasks",
    "n_no_positive_tasks",
    "n_no_negative_tasks",
    "n_positive_entries",
    "n_negative_entries",
    "n_positive_entries_active",
    "n_negative_entries_active",
    "implicit_pairs",
)


# ---------------------------------------------------------------------------
# the single training step
# ---------------------------------------------------------------------------


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y: torch.Tensor,
    *,
    global_step: int,
    clip: float,
    lambda_rank: float,
    check_labels: bool = False,
) -> dict[str, Any]:
    lr = C.set_lrs(optimizer, step=int(global_step))
    logits = model(batch)
    if tuple(logits.shape) != (int(y.shape[0]), C.N_TASKS):
        raise RuntimeError(
            f"model output {tuple(logits.shape)} != {(int(y.shape[0]), C.N_TASKS)}"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits at step {global_step}")
    if check_labels:
        observed = torch.isfinite(y)
        if bool((observed & ~((y == 0) | (y == 1))).any()):
            raise RuntimeError("observed labels are not binary")
    loss, parts = L.total_loss_parts(logits, y, float(lambda_rank))
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip)))
    if not math.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")
    optimizer.step()
    return {
        "loss_total": float(parts["loss_total"]),
        "loss_bce": float(parts["loss_bce"]),
        "loss_rank": float(parts["loss_rank"]),
        "lambda_rank_times_rank": float(parts["lambda_rank_times_rank"]),
        "grad_norm": gnorm,
        "lr": lr,
        "clipped": bool(gnorm > float(clip)),
        "stats": {k: parts[k] for k in RANK_STAT_KEYS},
        "active_mask": parts["active_mask"],
    }


# ---------------------------------------------------------------------------
# helpers
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
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


def recipe_payload(lambda_rank: float) -> dict[str, Any]:
    recipe = dict(C.RECIPE)
    recipe["loss"] = (
        "masked_bce_loss(logits, y) + lambda_rank * per_assay_hard_negative_rank_loss"
    )
    recipe["lambda_rank"] = float(lambda_rank)
    recipe["test_policy"] = (
        "official Test targets are NaN in the local label artifact and are never indexed"
    )
    return recipe


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(),
    }


def load_calibration(path: Path | str) -> dict[str, Any]:
    payload = C.read_json(Path(path))
    if payload.get("schema") != C.SCHEMA_CALIBRATION:
        raise RuntimeError(f"calibration schema {payload.get('schema')} is wrong")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError("calibration belongs to another round")
    if payload.get("status") != "FROZEN" or payload.get("preflight_all_pass") is not True:
        raise RuntimeError("calibration is not a frozen, preflight-passed record")
    if payload.get("test_labels_read") is not False:
        raise RuntimeError("calibration must declare test_labels_read=false")
    if payload.get("train_only") is not True:
        raise RuntimeError("calibration must be Train-only")
    if not payload.get("rng_unchanged"):
        raise RuntimeError("calibration changed the global RNG")
    if int(payload.get("valid_batches", 0)) < int(C.CALIBRATION["min_valid_batches"]):
        raise RuntimeError("calibration has too few valid rank batches")
    lam = float(payload["lambda_rank"])
    if not math.isfinite(lam) or not (C.CALIBRATION["lambda_min"] <= lam <= C.CALIBRATION["lambda_max"]):
        raise RuntimeError(f"frozen lambda_rank {lam} is outside the pre-registered bounds")
    if int(payload.get("n_batches") != int(C.CALIBRATION["n_batches"])):
        raise RuntimeError("calibration batch count is not the frozen 16")
    return payload


def checkpoint_payload(
    *,
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
    calibration: Mapping[str, Any],
    calibration_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": C.MODEL_KEY,
        "model_name": "LCRC-A-Wide-2x + TopRank objective",
        "seed": int(C.SEED),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["total_parameters"]),
        "state_dict": C.state_cpu(model),
        "optimizer_state_dict": C.optimizer_state_cpu(optimizer),
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
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "valid_report_summary": None
        if valid_report is None
        else {
            "ap": float(valid_report["ap"]),
            "valid_bce": float(valid_report["valid_bce"]),
            "n_tasks_scored": int(valid_report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": int(
                valid_report["n_tasks_with_observed_labels"]
            ),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "calibration": {
            "path": C.track_rel(calibration_path),
            "sha256": C.sha256_file(calibration_path),
            "lambda_rank": float(calibration["lambda_rank"]),
            "g_bce_median": float(calibration["g_bce_median"]),
            "g_rank_median": float(calibration["g_rank_median"]),
            "ratio_raw": float(calibration["ratio_raw"]),
            "clipped": bool(calibration["clipped"]),
            "valid_batches": int(calibration["valid_batches"]),
        },
        "recipe": recipe_payload(float(calibration["lambda_rank"])),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the local label artifact and are never indexed",
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


def restore(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("model") != C.MODEL_KEY:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != {C.MODEL_KEY}")
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
    hashes = C.restore_rng_state(payload["rng_state"])
    if hashes["combined_sha256"] != (payload.get("rng_state_hashes") or {}).get("combined_sha256"):
        raise RuntimeError("restored RNG state does not match the checkpoint hashes")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "rng_state_hashes": hashes,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    task: Any,
    *,
    out_dir: Path,
    device: torch.device,
    eval_every: int,
    epochs: int,
    calibration_path: Path,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(calibration_path)
    calibration_sha = C.sha256_file(calibration_path)
    lambda_rank = float(calibration["lambda_rank"])
    eval_epochs = tuple(e for e in range(1, int(epochs) + 1) if e % int(eval_every) == 0)
    formal = int(epochs) == int(C.TOTAL_EPOCHS) and int(eval_every) == 1
    order_seed = int(C.SEED)
    gpu_start = C.gpu_occupancy()

    model, model_audit = C.build_w2(task, device=device)
    optimizer, opt_audit = C.make_optimizer(model)
    plan = C.make_plan40(task, seed=order_seed)
    plan_audit = C.BC.SD.plan_summary(plan, task)
    if plan[0]["global_plan_sha256"] != C.GLOBAL_PLAN_SHA256:
        raise RuntimeError("order plan drifted")

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer)
        saved = resume.get("batch_plan") or {}
        now = C.plan_meta(plan, int(resume["epoch"]), order_seed)
        if now["global_plan_sha256"] != saved.get("global_plan_sha256") or now[
            "prefix_sha256"
        ] != saved.get("prefix_sha256"):
            raise RuntimeError("resumed plan does not match the checkpoint plan")
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        seed_info = C.seed_everything(int(C.SEED))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": C.plan_meta(plan, int(epochs), order_seed), **plan_audit})

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
        "support_lt5"
        if valid_pos[t] < 5
        else "support_5_19"
        if valid_pos[t] < 20
        else "support_ge20"
        for t in range(C.N_TASKS)
    ]

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    rank_epochs: list[dict[str, Any]] = []
    task_active_total = torch.zeros(C.N_TASKS, dtype=torch.float64)
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_valid.csv"
    rank_path = out_dir / "rank_diagnostics.json"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{int(epochs)}.pt"
    rng_path = out_dir / "rng_state.pt"

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    best_epoch: int | None = None
    best_ap = -math.inf
    best_checkpoint_sha = ""
    final_checkpoint_sha = ""
    last_checkpoint_sha = ""
    eval_completed: list[int] = []
    fixed_reports: dict[int, dict[str, Any]] = {}
    non_finite_events = 0
    clip_events_total = 0
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    train_eval_summary: dict[str, Any] | None = None
    epoch0_report: dict[str, Any] | None = None
    epoch40_valid_ap: float | None = None

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(
                gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device))
            )

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
            "valid_probs_sha256": report.get("valid_probs_sha256"),
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
                    "rank_active_batches": int(task_active_epoch[t].item()),
                    "rank_active_fraction": float(
                        task_active_epoch[t].item() / max(1, int(len(batches)))
                    ),
                }
            )
        update_peak()

    # epoch-0 diagnostic: the seed-0 initialisation, no training step
    if resume is None:
        model.eval()
        t_eval = time.perf_counter()
        with torch.inference_mode():
            report0 = C.BC.SD.evaluate_validation(
                model, task, device, reason="epoch_0_initialisation_diagnostic"
            )
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
        C.log(f"[toprank] epoch0 diagnostic validAP {epoch0_report['ap']:.6f}")

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
        losses_total: list[float] = []
        losses_bce: list[float] = []
        losses_rank: list[float] = []
        losses_lam: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        acc: dict[str, torch.Tensor] = {}
        task_active_epoch = torch.zeros(C.N_TASKS, dtype=torch.float64)
        active_per_batch: list[torch.Tensor] = []
        pairs_per_batch: list[torch.Tensor] = []
        pos_cover_per_batch: list[float] = []
        neg_cover_per_batch: list[float] = []
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.BC.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=global_steps_run,
                clip=clip,
                lambda_rank=lambda_rank,
                check_labels=(step == 1 and epoch == 1),
            )
            global_steps_run += 1
            losses_total.append(out["loss_total"])
            losses_bce.append(out["loss_bce"])
            losses_rank.append(out["loss_rank"])
            losses_lam.append(out["lambda_rank_times_rank"])
            grad_norms.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            for key, value in out["stats"].items():
                if key not in acc:
                    acc[key] = value.detach().to(torch.float64)
                else:
                    acc[key] += value.detach().to(torch.float64)
            active_mask = out["active_mask"].detach().to(torch.float64)
            task_active_total += active_mask.cpu()
            task_active_epoch += active_mask.cpu()
            active_per_batch.append(out["stats"]["n_active_tasks"].detach())
            pairs_per_batch.append(out["stats"]["implicit_pairs"].detach())
            n_pos = float(out["stats"]["n_positive_entries"])
            n_neg = float(out["stats"]["n_negative_entries"])
            n_pos_a = float(out["stats"]["n_positive_entries_active"])
            n_neg_a = float(out["stats"]["n_negative_entries_active"])
            pos_cover_per_batch.append(n_pos_a / n_pos if n_pos > 0 else float("nan"))
            neg_cover_per_batch.append(n_neg_a / n_neg if n_neg > 0 else float("nan"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[toprank] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        active_t = torch.stack(active_per_batch).double() if active_per_batch else torch.zeros(0, dtype=torch.float64)
        pairs_t = torch.stack(pairs_per_batch).double() if pairs_per_batch else torch.zeros(0, dtype=torch.float64)
        steps_run = int(len(step_times))
        task_batch_slots = max(1, steps_run * C.N_TASKS)
        rank_epoch = {
            "epoch": int(epoch),
            "steps": steps_run,
            "lambda_rank": float(lambda_rank),
            "active_tasks_mean": float(active_t.mean()) if active_t.numel() else 0.0,
            "active_tasks_median": float(active_t.median()) if active_t.numel() else 0.0,
            "active_tasks_min": float(active_t.min()) if active_t.numel() else 0.0,
            "active_tasks_max": float(active_t.max()) if active_t.numel() else 0.0,
            "active_task_fraction_mean": float(active_t.mean() / C.N_TASKS)
            if active_t.numel()
            else 0.0,
            "task_batch_activation_fraction": float(acc["n_active_tasks"].item() / task_batch_slots),
            "tasks_active_at_least_once": int((task_active_total > 0).sum().item()),
            "no_positive_task_fraction_mean": float(
                acc["n_no_positive_tasks"].item() / task_batch_slots
            ),
            "no_negative_task_fraction_mean": float(
                acc["n_no_negative_tasks"].item() / task_batch_slots
            ),
            "positive_coverage_mean": float(np.nanmean(pos_cover_per_batch))
            if pos_cover_per_batch
            else 0.0,
            "negative_coverage_mean": float(np.nanmean(neg_cover_per_batch))
            if neg_cover_per_batch
            else 0.0,
            "implicit_pairs_mean": float(pairs_t.mean()) if pairs_t.numel() else 0.0,
            "implicit_pairs_median": float(pairs_t.median()) if pairs_t.numel() else 0.0,
            "implicit_pairs_total": float(acc["implicit_pairs"].item()),
            "implicit_pairs_per_positive": float(
                acc["implicit_pairs"].item() / max(1.0, acc["n_positive_entries"].item())
            ),
            "positive_entries_total": float(acc["n_positive_entries"].item()),
            "negative_entries_total": float(acc["n_negative_entries"].item()),
            "loss_bce_mean": float(np.mean(losses_bce)) if losses_bce else float("nan"),
            "loss_rank_mean": float(np.mean(losses_rank)) if losses_rank else float("nan"),
            "lambda_rank_times_rank_mean": float(np.mean(losses_lam)) if losses_lam else float("nan"),
            "loss_total_mean": float(np.mean(losses_total)) if losses_total else float("nan"),
            "gradient_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        }
        rank_epochs.append(rank_epoch)
        C.write_json(rank_path, {"round_id": C.ROUND_ID, "epochs": rank_epochs})

        report: dict[str, Any] | None = None
        valid_seconds = 0.0
        strict = False
        if epoch in eval_epochs:
            model.eval()
            t_eval = time.perf_counter()
            with torch.inference_mode():
                report = C.BC.SD.evaluate_validation(
                    model, task, device, reason=f"epoch_{epoch}_official_validation"
                )
            valid_seconds = float(time.perf_counter() - t_eval)
            strict = best_epoch is None or float(report["ap"]) > float(best_ap)
            if strict:
                best_epoch = int(epoch)
                best_ap = float(report["ap"])
                payload = checkpoint_payload(
                    epoch=int(epoch),
                    global_step=int(global_steps_run),
                    model=model,
                    optimizer=optimizer,
                    rng_state=C.capture_rng_state(),
                    plan=C.plan_meta(plan, int(epoch), order_seed),
                    model_audit=model_audit,
                    valid_ap=best_ap,
                    valid_report=report,
                    kind="best_valid_ap",
                    calibration=calibration,
                    calibration_path=calibration_path,
                    extra={
                        "optimizer_state_audit": opt_audit,
                        "rank_diagnostics_epoch": rank_epoch,
                    },
                )
                best_checkpoint_sha = atomic_save(payload, best_path)
            record_eval(epoch, report)
            if epoch == int(epochs):
                epoch40_valid_ap = float(report["ap"])
                t_tr = time.perf_counter()
                with torch.inference_mode():
                    train_report = C.BC.SD.evaluate_train(model, task, device)
                train_eval_path = out_dir / f"train_eval_epoch{int(epoch)}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "model": C.MODEL_KEY,
                        "seed": int(C.SEED),
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
                    "eval_seconds": float(time.perf_counter() - t_tr),
                    "path": C.track_rel(train_eval_path),
                }
            model.train()
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        update_peak()
        epochs_run = int(epoch)
        cumulative = float(time.perf_counter() - t_start)
        row = {
            "epoch": int(epoch),
            "evaluated": bool(report is not None),
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss_total": rank_epoch["loss_total_mean"],
            "train_bce": rank_epoch["loss_bce_mean"],
            "train_rank": rank_epoch["loss_rank_mean"],
            "train_lambda_rank_times_rank": rank_epoch["lambda_rank_times_rank_mean"],
            "valid_ap": None if report is None else float(report["ap"]),
            "valid_bce": None if report is None else float(report["valid_bce"]),
            "n_tasks_scored": None if report is None else int(report["n_tasks_scored"]),
            "n_tasks_with_observed_labels": None
            if report is None
            else int(report["n_tasks_with_observed_labels"]),
            "epoch_train_seconds": train_seconds,
            "epoch_valid_seconds": valid_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
            "steps_per_second": float(len(step_times) / train_seconds)
            if train_seconds > 0
            else float("nan"),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events_epoch),
            "rank_active_tasks_mean": rank_epoch["active_tasks_mean"],
            "rank_active_tasks_median": rank_epoch["active_tasks_median"],
            "rank_active_task_fraction_mean": rank_epoch["active_task_fraction_mean"],
            "rank_no_positive_task_fraction_mean": rank_epoch["no_positive_task_fraction_mean"],
            "rank_no_negative_task_fraction_mean": rank_epoch["no_negative_task_fraction_mean"],
            "rank_implicit_pairs_mean": rank_epoch["implicit_pairs_mean"],
            "rank_positive_coverage_mean": rank_epoch["positive_coverage_mean"],
            "rank_negative_coverage_mean": rank_epoch["negative_coverage_mean"],
            "best_epoch_so_far": None if best_epoch is None else int(best_epoch),
            "best_valid_ap_so_far": None if best_epoch is None else float(best_ap),
            "is_best": bool(strict),
            "checkpoint_seconds": None,
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[toprank] epoch {epoch}/{int(epochs)} bce {row['train_bce']:.6f} "
            f"rank {row['train_rank']:.6f} active {row['rank_active_tasks_mean']:.1f} "
            f"validAP {valid_txt} LR {row['learning_rate']:.3e} "
            f"train {train_seconds:.1f}s valid {valid_seconds:.1f}s"
            f"{' *best*' if strict else ''}"
        )

        if epoch == int(epochs):
            payload = checkpoint_payload(
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=C.plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"epoch{int(epochs)}_primary",
                calibration=calibration,
                calibration_path=calibration_path,
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": None if train_eval_summary is None else dict(train_eval_summary),
                    "status_at_save": "COMPLETED",
                    "primary_comparison_epoch": int(C.TOTAL_EPOCHS),
                    "rank_diagnostics_epoch": rank_epoch,
                },
            )
            final_checkpoint_sha = atomic_save(payload, final_path)

        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            plan=C.plan_meta(plan, int(epoch), order_seed),
            model_audit=model_audit,
            valid_ap=None if report is None else float(report["ap"]),
            valid_report=None,
            kind="last_epoch_resumable",
            calibration=calibration,
            calibration_path=calibration_path,
            extra={"status_at_save": "RUNNING_OR_COMPLETED", "optimizer_state_audit": opt_audit},
        )
        last_checkpoint_sha = atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    completed = (
        epochs_run >= int(epochs)
        and set(eval_completed) == set(eval_epochs)
        and non_finite_events == 0
        and bool(final_checkpoint_sha)
    )
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end = C.capture_rng_state()
    rng_end_hashes = C.rng_state_hashes(rng_end)
    torch.save(
        {
            "schema": "lcrc-pcba-toprank-rng-v1",
            "round_id": C.ROUND_ID,
            "seed": int(C.SEED),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    curve_sha = write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    task_coverage_path = out_dir / "rank_task_coverage.csv"
    coverage_rows = []
    for t in range(C.N_TASKS):
        active = int(task_active_total[t].item())
        coverage_rows.append(
            {
                "task": t,
                "active_batches": active,
                "steps": int(epochs_run) * int(C.STEPS_PER_EPOCH),
                "active_fraction": float(active / max(1, int(epochs_run) * int(C.STEPS_PER_EPOCH))),
                "train_observed": int(train_counts["observed"][t]),
                "train_positive": int(train_counts["positive"][t]),
                "valid_positive": int(valid_pos[t]),
                "stratum": strata[t],
            }
        )
    coverage_sha = write_csv(
        task_coverage_path,
        [
            "task",
            "active_batches",
            "steps",
            "active_fraction",
            "train_observed",
            "train_positive",
            "valid_positive",
            "stratum",
        ],
        coverage_rows,
    )
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": C.MODEL_KEY,
        "model_name": "LCRC-A-Wide-2x + TopRank objective",
        "seed": int(C.SEED),
        "order_seed": int(order_seed),
        "formal_run": bool(formal),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["total_parameters"]),
        "base_parameters": int(model_audit["base_parameters"]),
        "readout_parameters": int(model_audit["readout_parameters"]),
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
        "batch_plan": C.plan_meta(plan, max(int(epochs_run), 0), order_seed),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": C.sha256_file(order_path),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_ap),
        "epoch40_valid_ap": epoch40_valid_ap,
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "primary_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "primary_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_valid": C.track_rel(per_task_path),
        "per_task_valid_sha256": per_task_sha,
        "rank_task_coverage": C.track_rel(task_coverage_path),
        "rank_task_coverage_sha256": coverage_sha,
        "rank_diagnostics": C.track_rel(rank_path),
        "rank_diagnostics_sha256": C.sha256_file(rank_path),
        "rank_epochs": rank_epochs,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "calibration": {
            "path": C.track_rel(calibration_path),
            "sha256": calibration_sha,
            "lambda_rank": float(lambda_rank),
            "g_bce_median": float(calibration["g_bce_median"]),
            "g_rank_median": float(calibration["g_rank_median"]),
            "ratio_raw": float(calibration["ratio_raw"]),
            "clipped": bool(calibration["clipped"]),
            "valid_batches": int(calibration["valid_batches"]),
            "batch_indices": calibration["batch_indices"],
        },
        "recipe": recipe_payload(float(calibration["lambda_rank"])),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "non_finite_events": int(non_finite_events),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows]))
        if curve_rows
        else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows]))
        if curve_rows
        else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the local label artifact and are never indexed",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done status={status} epochs={epochs_run}/{int(epochs)} best_epoch={best_epoch} "
        f"best_ap={best_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=C.TOTAL_EPOCHS)
    parser.add_argument("--calibration", required=True)
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
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        task = C.load_task_gated()
        resume = (
            torch.load(args.resume, map_location="cpu", weights_only=False)
            if args.resume
            else None
        )
        run = train(
            task,
            out_dir=out_dir,
            device=device,
            eval_every=int(args.eval_every),
            epochs=int(args.epochs),
            calibration_path=C.relative_or_absolute(args.calibration),
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
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
