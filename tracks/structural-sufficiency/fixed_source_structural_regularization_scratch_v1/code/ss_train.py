"""From-scratch trainer for ``STRUCT_DROP_SCRATCH`` (25 epochs, seed 0).

Identical to the historical DSCR FIXED1 seed-0 from-scratch run except that the
inherited structural-unit message dropout is active from the first training
step.  Inherited unchanged: batch 128, AdamW (weight_decay 1e-3, betas
0.9/0.999, eps 1e-8, freshly initialised), masked BCE, gradient clip 5, fp32,
the frozen seed-0 40-epoch order plan and the frozen 40-epoch warmup+cosine
schedule *prefix* (never rescaled; steps 0..68449).

Evaluation: complete official Valid every epoch; complete official Train at
epochs 12, 20, 25 and at the best checkpoint.  Test is never loaded or scored.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ss_common as R
import ss_model as M

C = R.C
DM = R.DM
DT = R.DT

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
    "mask_kept_sum_mean",
    "mask_same_tensor_both_blocks",
    "mask_directions_shared",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
    "mask_rng_fingerprint",
    "gpu_peak_allocated_bytes",
]
PER_TASK_FIELDS = ["epoch", "split", "task", "ap", "observed", "positive", "negative"]


def _eval(model: torch.nn.Module, task: Any, device: torch.device, split: str, epoch: int) -> dict[str, Any]:
    return C.evaluate_split(model, task, device, split, per_task=True, reason=f"epoch_{epoch}_full_{split}")


def _mask_epoch_stats(model: torch.nn.Module) -> dict[str, Any]:
    events = list(model.mask_events)
    model.mask_events.clear()
    if not events:
        return {"n": 0}
    return {
        "n": len(events),
        "dropped_fraction_mean": float(np.mean([e["dropped_fraction_physical"] for e in events])),
        "kept_fraction_mean": float(np.mean([e["kept_fraction_directed"] for e in events])),
        "same_tensor_both_blocks": bool(all(e["same_mask_tensor_both_blocks"] for e in events)),
        "directions_shared": bool(all(e["directions_share_physical_mask"] for e in events)),
        "n_physical_min": int(min(e["n_physical_bonds"] for e in events)),
        "n_physical_max": int(max(e["n_physical_bonds"] for e in events)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = R.utcnow()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if int(args.epochs) != R.EPOCHS:
        raise RuntimeError(f"this round fixes the horizon at {R.EPOCHS} epochs")

    # ---- data / plan (frozen, read-only) --------------------------------
    task = R.load_task_without_test()
    plan, plan_audit = R.plan_and_identity(task)
    plan_gate = R.verify_plan_identity(plan)

    # ---- model: exactly the FIXED1 seed-0 initialisation ---------------
    model, model_audit = M.build_scratch_model(task, device=device)
    optimizer, opt_audit = M.build_optimizer(model)
    fresh = M.optimizer_is_fresh(optimizer)
    if not fresh["fresh"]:
        raise RuntimeError(f"optimizer is not freshly initialised: {fresh}")
    if abs(fresh["lr_at_construction"][0] - float(C.RECIPE["warmup_start_lr"])) > 1e-15:
        raise RuntimeError(f"optimizer lr at construction {fresh['lr_at_construction']}")

    # the historical run seeded *after* building model/optimizer/plan; mirrored here
    seed_info = C.seed_everything(int(R.SEED))
    rng_start = C.rng_state_hashes(C.capture_rng_state())
    init_state_fingerprint = R.model_state_fingerprint(model)
    init_optimizer_fingerprint = R.optimizer_fingerprint(optimizer)
    if init_state_fingerprint != R.EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(f"initial state {init_state_fingerprint} != {R.EXPECTED_INITIAL_STATE_SHA256}")

    C.write_json(
        out_dir / "training_order.json",
        {"plan": C.plan_meta(plan, R.EPOCHS, R.ORDER_SEED), **plan_audit, "gate": plan_gate},
    )

    stager = PinnedStager(device, depth=2)
    watchdog = DT.Watchdog(float(args.hard_seconds))
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # ---- epoch-0 initialisation diagnostic (eval mode, RNG untouched) --
    before = C.capture_rng_state()
    model.eval()
    t0 = time.perf_counter()
    rep0 = _eval(model, task, device, "valid", 0)
    initial_seconds = float(time.perf_counter() - t0)
    if C.rng_state_hashes(before)["combined_sha256"] != C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]:
        raise RuntimeError("epoch-0 evaluation moved the training RNG state")
    initial_eval = {
        "schema": "fsrp-scratch-initial-eval-v1",
        "round_id": R.ROUND_ID,
        "epoch": 0,
        "global_step": 0,
        "arm": R.ARM,
        "valid": {
            k: rep0[k]
            for k in ("ap", "bce", "n_tasks_scored", "logits_sha256", "probs_sha256",
                      "per_task_ap", "observed", "positive", "negative")
        },
        "initial_state_sha256": init_state_fingerprint,
        "initial_optimizer_fingerprint": init_optimizer_fingerprint,
        "optimizer_fresh": fresh,
        "lr_first_step_scheduled": float(C.scheduled_lr(0, phase=3, kind="all")),
        "rng_state_hashes": rng_start,
        "mask_draws_before_first_step": int(model.mask_draws),
        "seconds": initial_seconds,
        "model_audit": model_audit,
        "plan_gate": plan_gate,
        "environment": C.environment(),
    }
    R.write_json(out_dir / "initial_eval.json", initial_eval)
    C.log(
        f"[{R.ARM}] epoch0 validAP {rep0['ap']:.9f} validBCE {rep0['bce']:.9f} "
        f"params {model_audit['parameters']['total']} mask_draws {model.mask_draws}"
    )

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    mask_epoch_stats: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{R.EPOCHS}.pt"

    best_epoch: int | None = None
    best_valid_ap = -math.inf
    best_valid_bce: float | None = None
    best_checkpoint_sha = ""
    last_checkpoint_sha = ""
    final_checkpoint_sha = ""
    clip_events_total = 0
    cumulative = 0.0
    gpu_peak_alloc = 0
    stop_reason = "RUNNING"
    global_steps = 0

    def update_peak() -> None:
        nonlocal gpu_peak_alloc
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))

    def _save(epoch: int, kind: str, path: Path, valid_ap: float) -> str:
        payload = R.checkpoint_payload(
            epoch=epoch,
            global_step=global_steps,
            model=model,
            optimizer=optimizer,
            mask_rng_state=model.mask_rng_state(),
            plan=C.plan_meta(plan, int(epoch), R.ORDER_SEED),
            model_audit=model_audit,
            valid_ap=valid_ap,
            kind=kind,
            extra={
                "optimizer_state_audit": opt_audit,
                "optimizer_fresh_at_start": fresh,
                "consumed_epochs": int(epoch),
                "consumed_steps": int(global_steps),
                "structural_dropout_runtime": model.finalize_checkpoint_extra(),
                "test_used_for_selection": False,
                "test_scored": False,
            },
        )
        return R.atomic_save(payload, path)

    for epoch in range(1, R.EPOCHS + 1):
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
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = DT.train_step(
                model, optimizer, batch, y, global_step=global_steps, clip=clip, audit_new=False
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
            C.log(f"[{R.ARM}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        rep_valid = _eval(model, task, device, "valid", epoch)
        valid_ap = float(rep_valid["ap"])
        valid_bce = float(rep_valid["bce"])
        train_evaluated = epoch in R.TRAIN_EVAL_EPOCHS
        if train_evaluated:
            t1 = time.perf_counter()
            rep_train = _eval(model, task, device, "train", epoch)
            train_eval_seconds = float(time.perf_counter() - t1)
        else:
            rep_train = None
            train_eval_seconds = 0.0

        strict = best_epoch is None or valid_ap > best_valid_ap
        if strict:
            best_epoch = int(epoch)
            best_valid_ap = valid_ap
            best_valid_bce = valid_bce
            best_checkpoint_sha = _save(epoch, "best_valid_ap", best_path, best_valid_ap)
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
        if rep_train is not None:
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

        mstats = _mask_epoch_stats(model)
        mask_epoch_stats.append({"epoch": int(epoch), **mstats})
        if mstats.get("n", 0) != len(batches):
            raise RuntimeError(f"epoch {epoch}: {mstats.get('n')} mask draws != {len(batches)} batches")
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        update_peak()
        cumulative = float(time.perf_counter() - t_start)
        clip_events_total += int(clip_events_epoch)

        row = {
            "epoch": int(epoch),
            "global_step": int(global_steps),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce_step_mean": float(np.mean(losses)) if losses else None,
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
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": None if best_epoch is None else int(best_epoch),
            "best_valid_ap_so_far": None if best_epoch is None else float(best_valid_ap),
            "is_best": bool(strict),
            "epoch_train_evaluated": bool(train_evaluated),
            "mask_draws": int(model.mask_draws),
            "mask_dropped_fraction_mean": mstats.get("dropped_fraction_mean"),
            "mask_kept_sum_mean": mstats.get("kept_fraction_mean"),
            "mask_same_tensor_both_blocks": mstats.get("same_tensor_both_blocks"),
            "mask_directions_shared": mstats.get("directions_shared"),
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
            "mask_rng_fingerprint": R.nested_fingerprint(model.mask_rng_state()),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        }
        curve_rows.append(row)
        R.write_csv(curve_path, CURVE_FIELDS, curve_rows)
        R.write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        last_checkpoint_sha = _save(epoch, "last_epoch_resumable", last_path, valid_ap)
        if epoch == R.EPOCHS:
            final_checkpoint_sha = _save(epoch, f"epoch{epoch}_horizon_endpoint", final_path, valid_ap)
        C.log(
            f"[{R.ARM}] ep {epoch}/{R.EPOCHS} bce {row['train_bce_step_mean']:.6f} "
            f"validAP {valid_ap:.6f} validBCE {valid_bce:.6f} trainAP {row['train_ap']} "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.1f}s valid {rep_valid['seconds']:.1f}s "
            f"drop {row['mask_dropped_fraction_mean']:.4f} {'*best*' if strict else ''}"
        )
    watchdog.cancel()
    stager.release()

    done = [int(r["epoch"]) for r in curve_rows]
    completed = done == list(range(1, R.EPOCHS + 1)) and bool(final_checkpoint_sha)
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH"
    status = "COMPLETED" if completed else "INCOMPLETE"

    # ---- best-checkpoint full Train + Valid re-evaluation ---------------
    best_eval: dict[str, Any] | None = None
    if best_path.is_file():
        payload_best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload_best["state_dict"], strict=True)
        model.eval()
        t0 = time.perf_counter()
        rep: dict[str, Any] = {}
        for split in ("train", "valid"):
            rep[split] = _eval(model, task, device, split, int(payload_best["epoch"]))
        best_eval = {
            "epoch": int(payload_best["epoch"]),
            "global_step": int(payload_best["global_step"]),
            "selection_metric": "valid.ap",
            "valid": {
                k: rep["valid"][k]
                for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")
            },
            "train": {
                k: rep["train"][k]
                for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")
            },
            "seconds": float(time.perf_counter() - t0),
        }
        best_eval["valid_ap_minus_recorded"] = float(best_eval["valid"]["ap"]) - float(best_valid_ap)
        if abs(best_eval["valid_ap_minus_recorded"]) > 2e-4:
            raise RuntimeError(
                f"best checkpoint Valid AP {best_eval['valid']['ap']} != recorded {best_valid_ap}"
            )
        R.write_json(out_dir / "best_checkpoint_eval.json", best_eval)

    total_seconds = float(time.perf_counter() - t_start)
    curve_sha = R.write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = R.write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    final_row = curve_rows[-1] if curve_rows else {}
    last5 = [r["valid_ap"] for r in curve_rows[-5:] if r.get("valid_ap") is not None]
    run = {
        "schema": R.SCHEMA_RUN,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "arm_role": "FIXED1 + structural-unit message dropout (q=0.10) trained from scratch",
        "seed": R.SEED,
        "order_seed": R.ORDER_SEED,
        "horizon": R.EPOCHS,
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "epochs_run": len(done),
        "global_steps_run": int(global_steps),
        "completed_epochs": done,
        "initialisation": "official FIXED1 seed-0 constructor state (no trained checkpoint loaded)",
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "optimizer_fresh_at_start": fresh,
        "plan_identity": plan_audit,
        "plan_gate": plan_gate,
        "batch_plan": C.plan_meta(plan, (done[-1] if done else 0), R.ORDER_SEED),
        "seed_state": seed_info,
        "initial_state_sha256": init_state_fingerprint,
        "initial_optimizer_fingerprint": init_optimizer_fingerprint,
        "rng_state_hashes_start": rng_start,
        "initial_eval": {
            "valid_ap": initial_eval["valid"]["ap"],
            "valid_bce": initial_eval["valid"]["bce"],
            "valid_logits_sha256": initial_eval["valid"]["logits_sha256"],
            "seconds": initial_seconds,
        },
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_valid_ap),
        "best_valid_bce": None if best_epoch is None else float(best_valid_bce),
        "horizon_endpoint": {
            "epoch": int(final_row.get("epoch", -1)),
            "global_step": int(final_row.get("global_step", -1)),
            "valid_ap": final_row.get("valid_ap"),
            "valid_bce": final_row.get("valid_bce"),
            "train_ap": final_row.get("train_ap"),
            "train_bce": final_row.get("train_bce"),
        },
        "last5_mean_valid_ap": float(np.mean(last5)) if last5 else None,
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
        "structural_dropout": {
            "q": R.Q,
            "mask_seed": R.MASK_SEED,
            "active_from_step": 0,
            "mask_draws_total": int(model.mask_draws),
            "eval_forward_calls": int(model.eval_forward_calls),
            "mask_draws_in_eval": 0,
            "mask_rng_fingerprint_end": R.nested_fingerprint(model.mask_rng_state()),
            "per_epoch": mask_epoch_stats,
            "mean_epoch_dropped_fraction": float(
                np.mean([s["dropped_fraction_mean"] for s in mask_epoch_stats if s.get("n")])
            )
            if mask_epoch_stats
            else None,
            "scope": "directed message term of both FIXED1 blocks, one Bernoulli draw per physical bond per training batch, "
                     "shared by both directions and both blocks, inverted-dropout scaling 1/(1-q)",
        },
        "recipe": {
            "batch_size": C.BATCH_SIZE,
            "optimizer": "AdamW from scratch (no restored moments)",
            "weight_decay": float(C.RECIPE["weight_decay"]),
            "adam_betas": list(C.RECIPE["adam_betas"]),
            "adam_eps": float(C.RECIPE["adam_eps"]),
            "learning_rate_schedule": "frozen 40-epoch warmup+cosine prefix, steps 0..68449 (not rescaled)",
            "peak_lr": float(C.RECIPE["learning_rate"]),
            "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
            "gradient_clip": clip,
            "loss": "masked BCE",
            "model_dropout": 0.10,
            "precision": "fp32",
            "epochs": R.EPOCHS,
            "train_eval_epochs": list(R.TRAIN_EVAL_EPOCHS),
            "selection": "highest full official Valid macro AP over epochs 1..25 (strict improvement)",
        },
        "test_scored": False,
        "test_labels_loaded": False,
        "test_used_for_selection": False,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "eval_seconds": float(
            sum(float(r.get("epoch_seconds") or 0.0) - float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)
        ),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "soft_seconds": float(args.soft_seconds),
        "hard_seconds": float(args.hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "started_utc": started_utc,
        "finished_utc": R.utcnow(),
    }
    R.write_json(out_dir / "run.json", run)
    C.log(
        f"done arm={R.ARM} status={status} best_epoch={best_epoch} best_valid={best_valid_ap} "
        f"gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=R.EPOCHS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        R.write_json(
            out_dir / "run.json",
            {
                "schema": R.SCHEMA_RUN,
                "round_id": R.ROUND_ID,
                "arm": R.ARM,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "finished_utc": R.utcnow(),
            },
        )
        return 2
    try:
        run = train(args)
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        R.write_json(
            out_dir / "run.json",
            {
                "schema": R.SCHEMA_RUN,
                "round_id": R.ROUND_ID,
                "arm": R.ARM,
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": R.utcnow(),
            },
        )
        C.log(f"FAILED arm={R.ARM}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
