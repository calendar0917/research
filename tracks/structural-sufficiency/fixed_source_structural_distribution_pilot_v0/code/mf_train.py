"""Formal 12-epoch MEANFIELD training run (single process, single GPU, seed 0).

Protocol (frozen in ``PROTOCOL.md``):

* official complete PCBA Train / Valid, batch 128, the frozen 40-epoch order plan
  of seed 0 and the frozen 40-epoch per-step cosine **prefix** (never recompressed);
* inherited recipe verbatim from ``ds_common.RECIPE`` (AdamW, wd 1e-3, warmup
  3e-5 -> 3e-4 in epoch 1, clip 5, dropout 0.10, masked BCE);
* full Valid AP/BCE in eval mode after every epoch;
* full Train AP/BCE at epochs 4, 8 and 12, plus a re-evaluation of the best
  checkpoint;
* best (highest full Valid macro AP, ties -> earlier epoch) and last checkpoint;
* **no Test access at all**: the evaluation wrapper rejects any split other than
  train/valid and no Test label is loaded.

Before the first training step the job runs the structural checks and the
real-Valid contrast measurement; an ``INSUFFICIENT_CONTRAST`` verdict stops the
job before any epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch  # noqa: E402

import mf_common as R  # noqa: E402
import mf_model as MF  # noqa: E402
import mf_checks as CK  # noqa: E402
import mf_contrast as CT  # noqa: E402
import ds_common as C  # noqa: E402
import ds_model as DM  # noqa: E402
import ds_train as T  # noqa: E402
import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce_step_mean",
    "train_bce",
    "train_ap",
    "train_evaluated",
    "valid_bce",
    "valid_ap",
    "n_tasks_scored_valid",
    "epoch_train_seconds",
    "valid_eval_seconds",
    "train_eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "grad_norm_mean",
    "grad_norm_max",
    "gradient_clip_events",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "R1",
    "R2",
    "Rtotal",
    "rms_h1_minus_h0",
    "rms_h2_minus_h1",
    "rms_m1_meanfield_minus_fixed1",
    "rms_m1_fixed1",
    "relative_m1_change",
    "n_directed_edges",
    "n_groups",
    "mean_group_size",
    "singleton_group_fraction_of_edges",
    "changed_message_fraction",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
]
PER_TASK_FIELDS = ["epoch", "split", "task", "ap", "observed", "positive", "negative"]
GRAD_TENSORS = (
    "bond_proj.weight",
    "blocks.0.fc1.weight",
    "blocks.0.fc2.weight",
    "blocks.1.fc1.weight",
    "blocks.1.fc2.weight",
)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, task: Any, device: torch.device, split: str, *, reason: str,
             keep_logits: bool = False) -> dict[str, Any]:
    """Full-split eval-mode scoring; Test is not an accepted split."""
    if split not in ("train", "valid"):
        raise RuntimeError(f"split {split!r} is not evaluated in this round (Test isolation)")
    return C.evaluate_split(
        model, task, device, split, per_task=True, keep_logits=keep_logits, reason=reason
    )


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def group_diagnostics(model: MF.MFModel, h0: torch.Tensor, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Group statistics and the actual message/aggregation contrast on one batch."""
    src, dst, edge_msg = DM.directed_bonds(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, audit = MF.group_index(bf_dir, gb_dir)
    blk1, blk2 = model.blocks[0], model.blocks[1]
    z1 = blk1.norm(h0)
    m1_mf, p1, p1_mf = MF.meanfield_message(z1, src, dst, edge_msg, gid, n_groups)
    if int(src.numel()) == 0:
        return {
            "n_directed_edges": 0,
            "n_groups": 0,
            "mean_group_size": 0.0,
            "singleton_group_fraction_of_edges": 0.0,
            "changed_message_fraction": 0.0,
            "rms_m1_fixed1": 0.0,
            "rms_m1_meanfield_minus_fixed1": 0.0,
            "relative_m1_change": 0.0,
        }
    m1_ref = torch.zeros_like(z1).index_add_(0, dst, p1)
    sizes = MF.group_sizes(gid, n_groups)
    n_edges = int(p1.shape[0])
    changed = int((p1_mf != p1).any(dim=1).sum().item())
    rms_base = float(m1_ref.double().pow(2).mean().sqrt().item())
    rms_change = float((m1_mf - m1_ref).double().pow(2).mean().sqrt().item())
    return {
        "n_directed_edges": n_edges,
        "n_groups": int(n_groups),
        "mean_group_size": float(n_edges / n_groups),
        "singleton_group_fraction_of_edges": float((sizes == 1).sum().item() * 1.0 / n_edges),
        "changed_message_fraction": float(changed / n_edges),
        "rms_m1_fixed1": rms_base,
        "rms_m1_meanfield_minus_fixed1": rms_change,
        "relative_m1_change": (rms_change / rms_base) if rms_base else 0.0,
    }


def state_usage(model: MF.MFModel, h0: torch.Tensor, batch: Mapping[str, Any]) -> dict[str, Any]:
    """``R1/R2/Rtotal`` exactly as in DSCR (uses the round's own forward)."""
    with torch.inference_mode():
        h1, h2, _info = model.structural_layers(h0, batch)
        d1, d2 = h1 - h0, h2 - h1
        rms = lambda x: float(x.detach().double().pow(2).mean().sqrt().item()) if x.numel() else 0.0
        r0, r1 = rms(h0), rms(h1)
        return {
            "R1": (rms(d1) / r0) if r0 else None,
            "R2": (rms(d2) / r1) if r1 else None,
            "Rtotal": (rms(h2 - h0) / r0) if r0 else None,
            "rms_h1_minus_h0": rms(d1),
            "rms_h2_minus_h1": rms(d2),
        }


# ---------------------------------------------------------------------------
# checkpoint payload
# ---------------------------------------------------------------------------


def checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_audit: Mapping[str, Any],
    plan_meta: Mapping[str, Any],
    valid_ap: float | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": R.SCHEMA_CHECKPOINT,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "horizon": R.HORIZON,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "state_dict": C.state_cpu(model),
        "optimizer_state_dict": C.optimizer_state_cpu(optimizer),
        "scheduler_state": {
            "kind": "scratch_warmup_cosine_40epoch_prefix",
            "global_step": int(global_step),
            "steps_per_epoch": int(R.STEPS_PER_EPOCH),
            "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
            "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        },
        "batch_plan": dict(plan_meta),
        "model_audit": dict(model_audit),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "selection_metric": "full official Valid macro AP (official OGB Evaluator)",
        "information_access_note": R.ACCESS_NOTE,
        "test_used_for_selection": False,
        "test_evaluated": False,
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    task: Any,
    *,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs)
    ensure = R.ensure_dirs
    ensure()

    # ---- 1. structural checks (no data needed except the Test-isolation gate)
    checks = CK.run_structural_checks(task)
    C.write_json(out_dir / "structural_checks.json", checks)
    C.log(f"structural checks {checks['n_passed']}/{checks['n_checks']} passed")

    # ---- 2. model + contrast gate
    model, model_audit = MF.build_mf_model(task.spec, task.norm, C.N_TASKS, seed=R.SEED, device=device)
    model_audit = dict(model_audit)
    model_audit["initial_state_sha256"] = C.state_fingerprint(C.state_cpu(model))
    contrast = CT.measure(task, model, device)
    C.write_json(out_dir / "contrast.json", contrast)
    C.log(
        f"contrast {contrast['status']}: rel RMS block1 {contrast['intervention']['block1_relative_rms_change']:.4f}, "
        f"changed messages {contrast['intervention']['changed_message_fraction']:.4f}, "
        f"singleton edges {contrast['groups']['singleton_fraction_of_edges']:.4f}"
    )
    allocation = {
        "schema": R.SCHEMA_CHECKS,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "created_utc": C.utcnow(),
        "structural_checks": checks,
        "contrast": contrast,
        "contrast_gate": contrast["status"],
        "training_authorised": bool(contrast["passed"]),
        "information_access_note": R.ACCESS_NOTE,
    }
    C.write_json(out_dir / "allocation_checks.json", allocation)
    if not contrast["passed"]:
        C.log("INSUFFICIENT_CONTRAST: formal training is not started")
        run = {
            "schema": R.SCHEMA_RUN,
            "round_id": R.ROUND_ID,
            "status": "STOPPED",
            "stop_reason": "INSUFFICIENT_CONTRAST",
            "model_audit": model_audit,
            "contrast": contrast,
            "finished_utc": C.utcnow(),
        }
        C.write_json(out_dir / "run.json", run)
        return run

    # ---- 3. optimiser / plan / RNG
    optimizer, opt_audit = T.build_optimizer(model)
    plan, plan_audit = C.plan_for_seed(task, R.SEED)
    seed_info = C.seed_everything(int(R.SEED))
    rng_start = C.rng_state_hashes(C.capture_rng_state())
    C.write_json(out_dir / "training_order.json", {"plan": C.plan_meta(plan, epochs, R.SEED), **plan_audit})

    stager = PinnedStager(device, depth=2)
    watchdog = T.Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    diag_rows = [int(x) for x in task.valid_rows[: int(C.DIAG_VALID_ROWS)]]
    diag_batch = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, diag_rows).items()
    }
    with torch.inference_mode():
        _a, _b, _scal, diag_h0 = model.encode_h0(diag_batch)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_hashes_epoch: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    best_path = out_dir / "checkpoint_best.pt"
    last_path = out_dir / "checkpoint_last.pt"
    best_epoch: int | None = None
    best_valid_ap = -math.inf
    best_sha = ""
    last_sha = ""
    eval_completed: list[int] = []
    train_eval_epochs: list[int] = []
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    clip_events_total = 0
    global_steps_run = 0
    epochs_run = 0
    stop_reason = "RUNNING"

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    def eval_rng_guarded(epoch: int, split: str, *, keep_logits: bool = False) -> dict[str, Any]:
        before = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        rep = evaluate(model, task, device, split, reason=f"epoch_{epoch}_{split}", keep_logits=keep_logits)
        after = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("evaluation moved the training RNG state")
        return rep

    def record_per_task(epoch: int, split: str, rep: Mapping[str, Any]) -> None:
        for t, ap in enumerate(rep["per_task_ap"]):
            per_task_rows.append(
                {
                    "epoch": int(epoch),
                    "split": split,
                    "task": int(t),
                    "ap": "" if ap is None else float(ap),
                    "observed": int(rep["observed"][t]),
                    "positive": int(rep["positive"][t]),
                    "negative": int(rep["negative"][t]),
                }
            )

    # ---- epoch 0 diagnostic (no training step)
    model.eval()
    rep0 = eval_rng_guarded(0, "valid")
    epoch0 = {"valid_ap": float(rep0["ap"]), "valid_bce": float(rep0["bce"]), "usage": state_usage(model, diag_h0, diag_batch)}
    C.log(
        f"[{R.ARM} seed{R.SEED}] epoch0 validAP {epoch0['valid_ap']:.9f} "
        f"(untrained W2 seed0 reference {C.W2_SEED0['epoch0_valid_ap']:.9f})"
    )

    for epoch in range(1, epochs + 1):
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
        grad_acc: dict[str, list[float]] = {name: [] for name in GRAD_TENSORS}
        clip_events = 0
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = T.train_step(
                model, optimizer, batch, y, global_step=global_steps_run, clip=clip, audit_new=True
            )
            global_steps_run += 1
            losses.append(out["loss"])
            grad_norms.append(out["grad_norm"])
            clip_events += int(out["clipped"])
            for name in GRAD_TENSORS:
                value = out["new_grads"].get(name)
                if value is not None:
                    grad_acc[name].append(float(value))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        model.eval()
        vrep = eval_rng_guarded(epoch, "valid")
        valid_ap = float(vrep["ap"])
        do_train_eval = epoch in R.TRAIN_EVAL_EPOCHS
        trep = eval_rng_guarded(epoch, "train") if do_train_eval else None
        if do_train_eval:
            train_eval_epochs.append(int(epoch))
        eval_completed.append(int(epoch))
        strict = best_epoch is None or valid_ap > float(best_valid_ap)
        if strict:
            best_epoch, best_valid_ap = int(epoch), valid_ap
            payload = checkpoint_payload(
                epoch=epoch,
                global_step=global_steps_run,
                model=model,
                optimizer=optimizer,
                model_audit=model_audit,
                plan_meta=C.plan_meta(plan, epoch, R.SEED),
                valid_ap=best_valid_ap,
                kind="best_valid_ap",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "train_ap_at_save": None if trep is None else float(trep["ap"]),
                    "train_bce_at_save": None if trep is None else float(trep["bce"]),
                },
            )
            best_sha = T._atomic_save(payload, best_path)
        record_per_task(epoch, "valid", vrep)
        if trep is not None:
            record_per_task(epoch, "train", trep)
        usage = state_usage(model, diag_h0, diag_batch)
        with torch.inference_mode():
            group_diag = group_diagnostics(model, diag_h0, diag_batch)
        rng = C.rng_state_hashes(C.capture_rng_state())
        rng_hashes_epoch.append({"epoch": int(epoch), **rng})
        update_peak()
        epochs_run = int(epoch)
        clip_events_total += clip_events
        cumulative = float(time.perf_counter() - t_start)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row = {
            "epoch": int(epoch),
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce_step_mean": float(np.mean(losses)) if losses else float("nan"),
            "train_bce": None if trep is None else float(trep["bce"]),
            "train_ap": None if trep is None else float(trep["ap"]),
            "train_evaluated": bool(do_train_eval),
            "valid_bce": float(vrep["bce"]),
            "valid_ap": valid_ap,
            "n_tasks_scored_valid": int(vrep["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "valid_eval_seconds": float(vrep["seconds"]),
            "train_eval_seconds": None if trep is None else float(trep["seconds"]),
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "grad_norm_mean": _mean(grad_norms),
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events),
            "best_epoch_so_far": int(best_epoch),
            "best_valid_ap_so_far": float(best_valid_ap),
            "is_best": bool(strict),
            "R1": usage["R1"],
            "R2": usage["R2"],
            "Rtotal": usage["Rtotal"],
            "rms_h1_minus_h0": usage["rms_h1_minus_h0"],
            "rms_h2_minus_h1": usage["rms_h2_minus_h1"],
            "rms_m1_meanfield_minus_fixed1": group_diag["rms_m1_meanfield_minus_fixed1"],
            "rms_m1_fixed1": group_diag["rms_m1_fixed1"],
            "relative_m1_change": group_diag["relative_m1_change"],
            "n_directed_edges": group_diag["n_directed_edges"],
            "n_groups": group_diag["n_groups"],
            "mean_group_size": group_diag["mean_group_size"],
            "singleton_group_fraction_of_edges": group_diag["singleton_group_fraction_of_edges"],
            "changed_message_fraction": group_diag["changed_message_fraction"],
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "rng_cpu_sha256": rng["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng["torch_cuda_sha256"]),
        }
        curve_rows.append(row)
        T._write_csv(curve_path, CURVE_FIELDS, curve_rows)
        T._write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        train_ap_txt = "-" if row["train_ap"] is None else f"{row['train_ap']:.6f}"
        C.log(
            f"[{R.ARM} seed{R.SEED}] ep {epoch}/{epochs} bce {row['train_bce_step_mean']:.6f} "
            f"trainAP {train_ap_txt} validAP {valid_ap:.6f} LR {row['learning_rate']:.3e} "
            f"train {train_seconds:.1f}s {'*best*' if strict else ''}"
        )
        payload_last = checkpoint_payload(
            epoch=epoch,
            global_step=global_steps_run,
            model=model,
            optimizer=optimizer,
            model_audit=model_audit,
            plan_meta=C.plan_meta(plan, epoch, R.SEED),
            valid_ap=valid_ap,
            kind="last_epoch_resumable",
            extra={"status_at_save": "RUNNING_OR_COMPLETED", "optimizer_state_audit": opt_audit},
        )
        last_sha = T._atomic_save(payload_last, last_path)

    watchdog.cancel()
    stager.release()
    completed = epochs_run >= epochs and sorted(eval_completed) == list(range(1, epochs + 1))
    stop_reason = "COMPLETED" if completed else (stop_reason if stop_reason != "RUNNING" else "MISSING_EPOCH")

    # ---- best-checkpoint re-evaluation (Valid + Train, no weight change)
    best_eval: dict[str, Any] | None = None
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            brep_valid = eval_rng_guarded(int(payload["epoch"]), "valid", keep_logits=True)
            brep_train = eval_rng_guarded(int(payload["epoch"]), "train")
        np.save(out_dir / "valid_logits_best.npy", brep_valid["logits"].astype(np.float32))
        best_eval = {
            "epoch": int(payload["epoch"]),
            "train": {k: brep_train[k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "valid": {k: brep_valid[k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "valid_ap_minus_recorded": float(brep_valid["ap"]) - float(best_valid_ap),
            "selection_metric": "valid.ap",
            "test_evaluated": False,
        }
        record_per_task(int(payload["epoch"]), "valid", brep_valid)
        record_per_task(int(payload["epoch"]), "train", brep_train)
        C.write_json(out_dir / "best_checkpoint_eval.json", best_eval)
        T._write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)

    manifest = {
        "schema": R.SCHEMA_MANIFEST,
        "round_id": R.ROUND_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "horizon": int(epochs),
        "parameters": int(model_audit["parameters"]["total"]),
        "initial_state_sha256": model_audit["initial_state_sha256"],
        "checkpoints": [],
    }
    for kind, path, sha in (("best_valid_ap", best_path, best_sha), ("last_epoch", last_path, last_sha)):
        if path.is_file():
            manifest["checkpoints"].append(
                {
                    "kind": kind,
                    "path": C.track_rel(path),
                    "sha256": sha or C.sha256_file(path),
                    "bytes": int(path.stat().st_size),
                    "epoch": int(best_epoch) if kind == "best_valid_ap" else int(epochs_run),
                }
            )
    manifest["valid_logits_best_sha256"] = (
        C.sha256_file(out_dir / "valid_logits_best.npy") if (out_dir / "valid_logits_best.npy").is_file() else None
    )
    C.write_json(out_dir / "checkpoint_manifest.json", manifest)

    final_row = curve_rows[-1] if curve_rows else {}
    run = {
        "schema": R.SCHEMA_RUN,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "seed": R.SEED,
        "order_seed": R.SEED,
        "horizon": int(epochs),
        "status": "COMPLETED" if completed else "INCOMPLETE",
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_completed": sorted(eval_completed),
        "train_eval_epochs": sorted(train_eval_epochs),
        "epoch0_diagnostic": epoch0,
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "batch_plan": C.plan_meta(plan, max(int(epochs_run), 0), R.SEED),
        "plan_identity": plan_audit,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_valid_ap),
        "horizon_endpoint": {
            "epoch": int(epochs_run),
            "valid_ap": final_row.get("valid_ap"),
            "valid_bce": final_row.get("valid_bce"),
            "train_ap": final_row.get("train_ap"),
            "train_bce": final_row.get("train_bce"),
        },
        "best_checkpoint_eval": best_eval,
        "checkpoint_manifest": C.track_rel(out_dir / "checkpoint_manifest.json"),
        "contrast_status": contrast["status"],
        "structural_checks_passed": bool(checks["all_passed"]),
        "recipe": {
            **dict(C.RECIPE),
            "round": R.ROUND_ID,
            "arm": R.ARM,
            "seed": R.SEED,
            "horizon": int(epochs),
            "prefix_of_frozen_schedule": True,
            "eval_cadence": "Valid every epoch; Train at epochs 4/8/12 + best checkpoint; no Test",
            "selection": "highest full official Valid macro AP within the horizon, ties -> earlier epoch",
            "test_evaluated": False,
            "test_used_for_selection": False,
        },
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": float(time.perf_counter() - t_start),
        "gpu_hours": float(time.perf_counter() - t_start) / 3600.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": C.gpu_occupancy(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "seed_state": seed_info,
        "rng_state_hashes_start": rng_start,
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "rng_epoch_hashes": rng_hashes_epoch,
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "information_access_note": R.ACCESS_NOTE,
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done status={run['status']} epochs={epochs_run}/{epochs} best_epoch={best_epoch} "
        f"best_valid={best_valid_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(R.RUN_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=R.HORIZON)
    parser.add_argument("--soft-seconds", type=float, default=2100.0)
    parser.add_argument("--hard-seconds", type=float, default=2700.0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out_dir / "run.json", {"schema": R.SCHEMA_RUN, "status": "INCOMPLETE",
                                            "stop_reason": "CUDA_UNAVAILABLE", "finished_utc": C.utcnow()})
        return 2
    try:
        task = C.SD.load_task()
        run = train(
            task,
            out_dir=out_dir,
            device=device,
            epochs=int(args.epochs),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
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
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
