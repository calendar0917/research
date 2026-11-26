"""FSRP v0 trainer: two arms continue FIXED1 seed 0 from epoch 25 to 33.

The recipe, the data path, the order plan and the LR schedule are inherited
unchanged (AdamW moments restored, per-step 40-epoch cosine prefix, masked BCE,
gradient clip 5, batch 128).  Test is never scored and no Test label is loaded.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sr_common as R
import sr_model as SM

C = R.C
DT = R.DT
DM = R.DM

from pf_pipeline import PinnedStager  # noqa: E402  (inherited, audited transfer path)

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce_step_mean",
    "train_ap",
    "train_bce",
    "valid_ap",
    "valid_bce",
    "n_tasks_scored_train",
    "n_tasks_scored_valid",
    "epoch_train_seconds",
    "valid_seconds",
    "train_eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "epoch_train_evaluated",
    "mask_draws",
    "mask_dropped_fraction_mean",
    "mask_same_tensor_both_blocks",
    "mask_directions_shared",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "mask_rng_fingerprint",
    "gpu_peak_allocated_bytes",
]
PER_TASK_FIELDS = ["epoch", "split", "task", "ap", "observed", "positive", "negative"]


def _spec_norm() -> tuple[Any, Any]:
    task = _TASK[0]
    return task.spec, task.norm


_TASK: list[Any] = [None]


def _build(arm: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec, norm = _spec_norm()
    if arm == "BASE_CONT":
        return DM.build_ds_model(spec, norm, C.N_TASKS, arm="FIXED1", seed=R.SEED, device=device)
    if arm == "STRUCT_DROP":
        model = SM.StructuralDropModel(
            spec, norm, C.N_TASKS, dict(C.BC.config_for_scale(2)),
            struct_drop_q=R.Q, mask_seed=R.MASK_SEED,
        )
        model.to(device)
        audit = C.parameter_audit(model)
        if not audit["all_match"]:
            raise RuntimeError(f"parameter audit failed for STRUCT_DROP: {audit['checks']}")
        if int(audit["total"]) != R.PARAMS_TOTAL:
            raise RuntimeError(f"STRUCT_DROP parameters {audit['total']} != {R.PARAMS_TOTAL}")
        return model, {"parameters": audit, "arm": "FIXED1+STRUCT_DROP"}
    raise KeyError(arm)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return R.sha256_file(path)


def _eval(model: torch.nn.Module, task: Any, device: torch.device, split: str, epoch: int, *, per_task: bool) -> dict[str, Any]:
    return C.evaluate_split(
        model, task, device, split, per_task=per_task,
        reason=f"epoch_{epoch}_full_{split}",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = R.utcnow()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    arm = str(args.arm)
    if arm not in R.ARMS:
        raise KeyError(arm)
    if int(args.epochs) != R.END_EPOCH:
        raise RuntimeError("this round fixes the continuation horizon at epoch 33")

    task = R.load_task_without_test()
    _TASK[0] = task
    payload, base_audit, base_path = R.load_base_checkpoint(args.base_checkpoint)
    plan, plan_audit = R.plan_and_identity(task)
    resume_gate = R.verify_plan_prefix(payload, plan, R.START_EPOCH)

    model, model_audit = _build(arm, device)
    optimizer, opt_audit = DT.build_optimizer(model)
    resume_audit = DT.restore(payload, model, optimizer, arm="FIXED1", seed=R.SEED)
    if arm == "STRUCT_DROP":
        model.restore_mask_rng_state(payload.get("mask_rng_state"))
    lr_at_start = R.current_lr_at(optimizer, R.EXPECTED_BASE_STEP)
    if abs(lr_at_start - R.EXPECTED_LR_AT_68450) > 1e-15:
        raise RuntimeError(f"restored LR at step {R.EXPECTED_BASE_STEP} is {lr_at_start}")
    global_steps = R.EXPECTED_BASE_STEP
    init_state_fingerprint = R.model_state_fingerprint(model)
    init_optimizer_fingerprint = R.optimizer_fingerprint(optimizer)

    C.write_json(
        out_dir / "training_order.json",
        {"plan": C.plan_meta(plan, R.HORIZON, R.ORDER_SEED), **plan_audit},
    )

    stager = PinnedStager(device, depth=2)
    watchdog = DT.Watchdog(float(args.hard_seconds))
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{R.END_EPOCH}.pt"

    def _mask_extra() -> dict[str, Any]:
        if arm == "STRUCT_DROP":
            return model.finalize_checkpoint_extra()
        return {"structural_dropout": None}

    def _save(epoch: int, kind: str, path: Path, valid_ap: float | None) -> str:
        mask_state = model.mask_rng_state() if arm == "STRUCT_DROP" else None
        payload_out = R.checkpoint_payload(
            arm=arm,
            epoch=int(epoch),
            global_step=int(global_steps),
            model=model,
            optimizer=optimizer,
            mask_rng_state=mask_state,
            plan=C.plan_meta(plan, int(epoch), R.ORDER_SEED),
            model_audit={**model_audit, "arm": "FIXED1" if arm == "BASE_CONT" else "FIXED1+STRUCT_DROP"},
            valid_ap=valid_ap,
            kind=kind,
            extra={
                "optimizer_state_audit": opt_audit,
                "base_checkpoint": base_audit,
                "resume_audit": resume_audit,
                "initial_state_fingerprint": init_state_fingerprint,
                "initial_optimizer_fingerprint": init_optimizer_fingerprint,
                "lr_at_resume": lr_at_start,
                "consumed_epochs": int(epoch),
                "consumed_steps": int(global_steps),
                "test_used_for_selection": False,
                "test_scored": False,
                **_mask_extra(),
            },
        )
        return R.atomic_save(payload_out, path)

    # ---- epoch-25 common baseline (complete Train + complete Valid) ------
    model.eval()
    t0 = time.perf_counter()
    baseline_valid = _eval(model, task, device, "valid", R.START_EPOCH, per_task=True)
    baseline_train = _eval(model, task, device, "train", R.START_EPOCH, per_task=True)
    baseline_seconds = float(time.perf_counter() - t0)
    baseline = {
        "schema": "fsrp-baseline-v1",
        "round_id": R.ROUND_ID,
        "epoch": R.START_EPOCH,
        "global_step": R.EXPECTED_BASE_STEP,
        "checkpoint": base_audit,
        "arm": arm,
        "valid": {
            k: baseline_valid[k]
            for k in ("ap", "bce", "n_tasks_scored", "logits_sha256", "probs_sha256",
                      "per_task_ap", "observed", "positive", "negative")
        },
        "train": {
            k: baseline_train[k]
            for k in ("ap", "bce", "n_tasks_scored", "logits_sha256", "per_task_ap",
                      "observed", "positive", "negative")
        },
        "lr": lr_at_start,
        "seconds": baseline_seconds,
        "model_state_fingerprint": init_state_fingerprint,
        "optimizer_fingerprint": init_optimizer_fingerprint,
        "rng_state_hashes": C.rng_state_hashes(C.capture_rng_state()),
        "mask_rng_fingerprint": None if arm == "BASE_CONT" else R.nested_fingerprint(model.mask_rng_state()),
    }
    R.write_json(out_dir / "baseline_epoch25.json", baseline)
    C.log(
        f"[{arm}] epoch25 baseline validAP {baseline_valid['ap']:.9f} validBCE {baseline_valid['bce']:.9f} "
        f"trainAP {baseline_train['ap']:.9f} trainBCE {baseline_train['bce']:.9f}"
    )

    best_epoch: int | None = None
    best_valid_ap = -math.inf
    best_valid_bce: float | None = None
    best_checkpoint_sha = ""
    last_checkpoint_sha = ""
    final_checkpoint_sha = ""
    clip_events_total = 0
    eval_completed = False
    cumulative = 0.0
    gpu_peak_alloc = 0
    stop_reason = "RUNNING"
    mask_dropped_fractions: list[float] = []

    def update_peak() -> None:
        nonlocal gpu_peak_alloc
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))

    for epoch in R.TRAIN_EPOCHS:
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(args.soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        grad_norms: list[float] = []
        step_times: list[float] = []
        clip_events_epoch = 0
        mask_draws_before = int(getattr(model, "mask_draws", 0))
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = DT.train_step(
                model, optimizer, batch, y,
                global_step=global_steps, clip=clip, audit_new=False,
            )
            global_steps += 1
            losses.append(out["loss"])
            grad_norms.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{arm}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        rep_valid = _eval(model, task, device, "valid", epoch, per_task=True)
        valid_ap = float(rep_valid["ap"])
        valid_bce = float(rep_valid["bce"])
        train_evaluated = epoch in R.TRAIN_EVAL_EPOCHS or epoch == R.END_EPOCH
        if train_evaluated:
            t1 = time.perf_counter()
            rep_train = _eval(model, task, device, "train", epoch, per_task=True)
            train_eval_seconds = float(time.perf_counter() - t1)
        else:
            rep_train = None
            train_eval_seconds = 0.0
        eval_completed = True
        strict = best_epoch is None or valid_ap > best_valid_ap
        if strict:
            best_epoch = int(epoch)
            best_valid_ap = valid_ap
            best_valid_bce = valid_bce
            best_checkpoint_sha = _save(epoch, "best_valid_ap", best_path, best_valid_ap)
        if rep_valid.get("per_task_ap") is not None:
            for t, ap in enumerate(rep_valid["per_task_ap"]):
                per_task_rows.append(
                    {
                        "epoch": int(epoch), "split": "valid", "task": int(t),
                        "ap": "" if ap is None else float(ap),
                        "observed": int(rep_valid["observed"][t]),
                        "positive": int(rep_valid["positive"][t]),
                        "negative": int(rep_valid["negative"][t]),
                    }
                )
        if rep_train is not None and rep_train.get("per_task_ap") is not None:
            for t, ap in enumerate(rep_train["per_task_ap"]):
                per_task_rows.append(
                    {
                        "epoch": int(epoch), "split": "train", "task": int(t),
                        "ap": "" if ap is None else float(ap),
                        "observed": int(rep_train["observed"][t]),
                        "positive": int(rep_train["positive"][t]),
                        "negative": int(rep_train["negative"][t]),
                    }
                )
        mask_draws = int(getattr(model, "mask_draws", 0))
        dropped_fraction = None
        same_tensor = None
        directions_shared = None
        if arm == "STRUCT_DROP":
            events = getattr(model, "mask_events", [])
            if mask_draws > mask_draws_before and events:
                new_events = events[mask_draws_before:]
                dropped_fraction = float(np.mean([e["dropped_fraction_physical"] for e in new_events]))
                same_tensor = bool(all(e["same_mask_tensor_both_blocks"] for e in new_events))
                directions_shared = bool(all(e["directions_share_physical_mask"] for e in new_events))
                mask_dropped_fractions.append(dropped_fraction)
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        update_peak()
        cumulative = float(time.perf_counter() - t_start)
        clip_events_total += int(clip_events_epoch)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row = {
            "epoch": int(epoch),
            "global_step": int(global_steps),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce_step_mean": _mean(losses),
            "train_ap": None if rep_train is None else float(rep_train["ap"]),
            "train_bce": None if rep_train is None else float(rep_train["bce"]),
            "valid_ap": valid_ap,
            "valid_bce": valid_bce,
            "n_tasks_scored_train": None if rep_train is None else int(rep_train["n_tasks_scored"]),
            "n_tasks_scored_valid": int(rep_valid["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "valid_seconds": float(rep_valid["seconds"]),
            "train_eval_seconds": train_eval_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else None,
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else None,
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else None,
            "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else None,
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean": _mean(grad_norms),
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": None if best_epoch is None else int(best_epoch),
            "best_valid_ap_so_far": None if best_epoch is None else float(best_valid_ap),
            "is_best": bool(strict),
            "epoch_train_evaluated": bool(train_evaluated),
            "mask_draws": int(getattr(model, "mask_draws", 0)) if arm == "STRUCT_DROP" else None,
            "mask_dropped_fraction_mean": dropped_fraction,
            "mask_same_tensor_both_blocks": same_tensor,
            "mask_directions_shared": directions_shared,
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
            "mask_rng_fingerprint": None if arm == "BASE_CONT" else R.nested_fingerprint(model.mask_rng_state()),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        }
        curve_rows.append(row)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        last_checkpoint_sha = _save(epoch, "last_epoch_resumable", last_path, valid_ap)
        if epoch == R.END_EPOCH:
            final_checkpoint_sha = _save(epoch, f"epoch{epoch}_horizon_endpoint", final_path, valid_ap)
        C.log(
            f"[{arm}] ep {epoch}/{R.END_EPOCH} bce {row['train_bce_step_mean']:.6f} "
            f"validAP {valid_ap:.6f} validBCE {valid_bce:.6f} "
            f"trainAP {row['train_ap']} LR {row['learning_rate']:.3e} "
            f"train {train_seconds:.1f}s eval {rep_valid['seconds']:.1f}s "
            f"{'*best*' if strict else ''}"
        )
    watchdog.cancel()
    stager.release()

    expected = list(R.TRAIN_EPOCHS)
    done = [int(r["epoch"]) for r in curve_rows]
    completed = done == expected and bool(final_checkpoint_sha)
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH"
    status = "COMPLETED" if completed else "INCOMPLETE"

    # ---- post-hoc evaluation of the best checkpoint ---------------------
    best_eval: dict[str, Any] | None = None
    if best_path.is_file():
        payload_best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload_best["state_dict"], strict=True)
        model.eval()
        t0 = time.perf_counter()
        rep = {}
        for split in ("train", "valid"):
            rep[split] = _eval(model, task, device, split, int(payload_best["epoch"]), per_task=True)
        best_eval = {
            "epoch": int(payload_best["epoch"]),
            "global_step": int(payload_best["global_step"]),
            "selection_metric": "valid.ap",
            "valid": {k: rep["valid"][k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "train": {k: rep["train"][k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "seconds": float(time.perf_counter() - t0),
        }
        best_eval["valid_ap_minus_recorded"] = float(best_eval["valid"]["ap"]) - float(best_valid_ap)
        if abs(best_eval["valid_ap_minus_recorded"]) > 1e-4:
            raise RuntimeError(
                f"best checkpoint Valid AP {best_eval['valid']['ap']} != recorded {best_valid_ap}"
            )
        R.write_json(out_dir / "best_checkpoint_eval.json", best_eval)

    total_seconds = float(time.perf_counter() - t_start)
    curve_sha = _write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    final_row = curve_rows[-1] if curve_rows else {}
    endpoint = {
        "epoch": int(final_row.get("epoch", -1)),
        "global_step": int(final_row.get("global_step", -1)),
        "valid_ap": final_row.get("valid_ap"),
        "valid_bce": final_row.get("valid_bce"),
        "train_ap": final_row.get("train_ap"),
        "train_bce": final_row.get("train_bce"),
    }
    last_three = [r["valid_ap"] for r in curve_rows[-3:] if r.get("valid_ap") is not None]
    run = {
        "schema": R.SCHEMA_RUN,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": arm,
        "arm_role": R.ARM_ROLE[arm],
        "seed": R.SEED,
        "order_seed": R.ORDER_SEED,
        "horizon": R.END_EPOCH,
        "start_epoch": R.START_EPOCH,
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "epochs_run": len(done),
        "global_steps_run": int(global_steps),
        "completed_epochs": done,
        "base_checkpoint": base_audit,
        "resume_audit": resume_audit,
        "resume_gate": resume_gate,
        "plan_identity": plan_audit,
        "batch_plan": C.plan_meta(plan, (done[-1] if done else R.START_EPOCH), R.ORDER_SEED),
        "lr_at_resume": lr_at_start,
        "initial_state_fingerprint": init_state_fingerprint,
        "initial_optimizer_fingerprint": init_optimizer_fingerprint,
        "baseline_epoch25": {
            "valid_ap": baseline_valid["ap"],
            "valid_bce": baseline_valid["bce"],
            "train_ap": baseline_train["ap"],
            "train_bce": baseline_train["bce"],
            "valid_logits_sha256": baseline_valid["logits_sha256"],
            "train_logits_sha256": baseline_train["logits_sha256"],
        },
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_valid_ap),
        "best_valid_bce": None if best_epoch is None else float(best_valid_bce),
        "horizon_endpoint": endpoint,
        "last_three_mean_valid_ap": float(np.mean(last_three)) if last_three else None,
        "best_checkpoint_eval": best_eval,
        "best_checkpoint": C.track_rel(best_path) if best_path.is_file() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "horizon_checkpoint": C.track_rel(final_path) if final_path.is_file() else None,
        "horizon_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.is_file() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "structural_dropout": None if arm == "BASE_CONT" else {
            "q": R.Q,
            "mask_seed": R.MASK_SEED,
            "mask_draws": int(model.mask_draws),
            "eval_forward_calls": int(model.eval_forward_calls),
            "mask_draws_eval": 0,
            "mean_epoch_dropped_fraction": float(np.mean(mask_dropped_fractions)) if mask_dropped_fractions else None,
            "n_epochs_with_drop_stats": len(mask_dropped_fractions),
            "scope": "directed message term of both FIXED1 blocks, one Bernoulli draw per physical bond per training batch",
        },
        "recipe": {
            "batch_size": C.BATCH_SIZE,
            "optimizer": "AdamW",
            "weight_decay": float(C.RECIPE["weight_decay"]),
            "learning_rate_schedule": "original 40-epoch warmup+cosine prefix, resumed at step 68450",
            "gradient_clip": clip,
            "loss": "masked BCE",
            "horizon_override": "epoch 25 -> 33 continuation in isolated round code (schedule not rescaled)",
            "model_dropout": 0.10,
            "train_eval_epochs": list(R.TRAIN_EVAL_EPOCHS),
            "selection": "highest full official Valid macro AP over epochs 26..33 (strict improvement)",
        },
        "test_scored": False,
        "test_labels_loaded": False,
        "test_used_for_selection": False,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "eval_seconds": float(sum(float(r.get("epoch_seconds") or 0.0) - float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "soft_seconds": float(args.soft_seconds),
        "hard_seconds": float(args.hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "rng_state_hashes_start": baseline["rng_state_hashes"],
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "started_utc": started_utc,
        "finished_utc": R.utcnow(),
        "eval_completed": eval_completed,
    }
    R.write_json(out_dir / "run.json", run)
    C.log(f"done arm={arm} status={status} best_epoch={best_epoch} best_valid={best_valid_ap} gpu_hours={run['gpu_hours']:.4f}")
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(R.ARMS))
    parser.add_argument("--epochs", type=int, default=R.END_EPOCH)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        R.write_json(out_dir / "run.json", {
            "schema": R.SCHEMA_RUN, "round_id": R.ROUND_ID, "arm": args.arm,
            "status": "INCOMPLETE", "stop_reason": "CUDA_UNAVAILABLE", "finished_utc": R.utcnow(),
        })
        return 2
    try:
        run = train(args)
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        R.write_json(out_dir / "run.json", {
            "schema": R.SCHEMA_RUN, "round_id": R.ROUND_ID, "arm": str(args.arm),
            "status": "FAILED", "stop_reason": type(exc).__name__, "detail": str(exc),
            "traceback": traceback.format_exc(), "environment": C.environment(),
            "finished_utc": R.utcnow(),
        })
        C.log(f"FAILED arm={args.arm}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
