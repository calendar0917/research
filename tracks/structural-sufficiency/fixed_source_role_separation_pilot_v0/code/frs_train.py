"""FSR v0 training driver (preflight cost measurement + formal run).

The update rule, the frozen data path and the frozen seed-0 order plan are the
inherited ones (``ds_common``).  Only the horizon changes: the round always uses
a *prefix* of the frozen 40-epoch cosine schedule, never a recompressed one.

Per-epoch evaluation: full official Validation macro AP/BCE every epoch; full
official Train AP/BCE at the epochs the protocol names (4, 8, 12 that are
actually reached) and for the final best checkpoint.  **No Test evaluation
exists in this round.**
"""

from __future__ import annotations

import argparse
import csv
import json
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
import torch.nn as nn  # noqa: E402

import frs_common as FC  # noqa: E402  (sets sys.path to the inherited code)
import ds_common as C  # noqa: E402
import ds_model as DSM  # noqa: E402
import frs_model as FM  # noqa: E402
import frs_mechanism as MECH  # noqa: E402
import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "arm",
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
    "valid_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "gap_norm_max",
    "gap_clip_events",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "R1",
    "R2",
    "Rtotal",
    "rms_h1_minus_h0",
    "rms_h2_minus_h1",
    "rms_z1",
    "rms_m1",
    "rms_z_target",
    "rms_m2",
    "rms_fc1_term_0",
    "rms_correction_0",
    "rms_fc1_term_1",
    "rms_correction_1",
    "lowrank_weight_norm_0",
    "lowrank_weight_norm_1",
    "grad_norm_lowrank_0_A",
    "grad_norm_lowrank_0_B",
    "grad_norm_lowrank_1_A",
    "grad_norm_lowrank_1_B",
    "grad_norm_lowrank_total",
    "grad_norm_inherited_new_total",
    "grad_norm_w2_total",
]

PER_TASK_FIELDS = ["arm", "epoch", "split", "task", "ap", "observed", "positive", "negative"]


class Watchdog:
    def __init__(self, hard_seconds: float):
        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self._timer = threading.Timer(self.hard_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
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


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def build_optimizer(model: nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("one parameter group (inherited recipe)")
    return optimizer, {
        "optimizer": "AdamW",
        "n_param_groups": 1,
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "per_step_cosine_40epoch_with_1epoch_linear_warmup (horizon is a prefix)",
    }


def _group_norm(model: nn.Module, predicate) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if not predicate(name) or p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y: torch.Tensor,
    *,
    global_step: int,
    clip: float,
) -> dict[str, Any]:
    lr = C.set_lrs(optimizer, step=int(global_step))
    logits = model(batch)
    if tuple(logits.shape) != (int(y.shape[0]), int(C.N_TASKS)):
        raise RuntimeError(f"model output {tuple(logits.shape)} != {(int(y.shape[0]), int(C.N_TASKS))}")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits at step {global_step}")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = {
        name: (None if p.grad is None else float(p.grad.detach().norm().item()))
        for name, p in model.named_parameters()
        if FC.is_lowrank(name)
    }
    lowrank_norm = _group_norm(model, FC.is_lowrank)
    inherited_norm = _group_norm(model, FC.is_inherited_new)
    w2_norm = _group_norm(model, lambda n: not FC.is_frs_new(n))
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if not math.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grad_norm": gnorm,
        "lr": lr,
        "clipped": bool(gnorm > clip),
        "lowrank_grads": grads,
        "grad_norm_lowrank_total": lowrank_norm,
        "grad_norm_inherited_new_total": inherited_norm,
        "grad_norm_w2_total": w2_norm,
    }


def evaluation_protocol(epochs: int) -> dict[str, Any]:
    train_epochs = sorted({e for e in FC.TRAIN_EVAL_EPOCHS if e <= int(epochs)})
    return {
        "valid_every_epoch": True,
        "train_eval_epochs": train_epochs,
        "train_eval_best_checkpoint_at_end": True,
        "test_evaluated": False,
        "test_policy": "NOT_RUN / FORBIDDEN_THIS_ROUND",
        "checkpoint_selection": "highest full official Valid macro AP within the horizon; tie -> earlier epoch",
    }


def diagnostic_usage(model: FM.FRSModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    """R1/R2 and the two pre-nonlinearity terms on a fixed real Validation batch."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            a, b, scal, h0 = model.encode_h0(batch)
            src, dst, edge_msg = DSM.directed_bonds(model, batch)
            h1, m1 = model.stage1(h0, src, dst, edge_msg)
            h2, m2, info = model.stage2(h1, h0, src, dst, edge_msg)
            z1 = model.blocks[0].norm(h0)
            z_target = model.blocks[1].norm(h1)
            term = []
            corr = []
            for i, (z, m) in enumerate(((z1, m1), (z_target, m2))):
                term.append(DSM.rms(model.blocks[i].fc1(z + m)))
                corr.append(DSM.rms(model.lowrank[i](model.branch_input(z, m))))
            r0, r1 = DSM.rms(h0), DSM.rms(h1)
            return {
                "rms_h0": r0,
                "rms_h1": r1,
                "rms_h2": DSM.rms(h2),
                "rms_h1_minus_h0": DSM.rms(h1 - h0),
                "rms_h2_minus_h1": DSM.rms(h2 - h1),
                "rms_h2_minus_h0": DSM.rms(h2 - h0),
                "R1": (DSM.rms(h1 - h0) / r0) if r0 else None,
                "R2": (DSM.rms(h2 - h1) / r1) if r1 else None,
                "Rtotal": (DSM.rms(h2 - h0) / r0) if r0 else None,
                "rms_z1": DSM.rms(z1),
                "rms_m1": DSM.rms(m1),
                "rms_z_target": DSM.rms(z_target),
                "rms_m2": DSM.rms(m2),
                "rms_fc1_term_0": term[0],
                "rms_correction_0": corr[0],
                "rms_fc1_term_1": term[1],
                "rms_correction_1": corr[1],
                "correction_to_fc1_ratio_0": (corr[0] / term[0]) if term[0] else None,
                "correction_to_fc1_ratio_1": (corr[1] / term[1]) if term[1] else None,
                "lowrank_weight_norm_0": float(model.lowrank[0].effective_weight().norm().item()),
                "lowrank_weight_norm_1": float(model.lowrank[1].effective_weight().norm().item()),
                "n_atoms": int(h0.shape[0]),
                "n_directed_edges": int(src.numel()),
                "layer2_reads": info.get("layer2_reads"),
                "finite_h2": bool(torch.isfinite(h2).all().item()),
            }
    finally:
        if was_training:
            model.train()


def checkpoint_payload(
    *,
    arm: str,
    seed: int,
    horizon: int,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    plan_meta: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": FC.SCHEMA_CHECKPOINT,
        "round_id": FC.ROUND_ID,
        "protocol_id": FC.PROTOCOL_ID,
        "arm": str(arm),
        "base_computation": "FIXED1",
        "role_branch_input": "z + m" if arm == "TIED" else "z - m",
        "seed": int(seed),
        "horizon": int(horizon),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "state_dict": C.state_cpu(model),
        "optimizer_state_dict": C.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": {
            "kind": "scratch_warmup_cosine_40epoch_prefix",
            "global_step": int(global_step),
            "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
            "steps_per_epoch": int(C.STEPS_PER_EPOCH),
            "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        },
        "rng_state": dict(rng_state),
        "batch_plan": dict(plan_meta),
        "model_audit": dict(model_audit),
        "valid_ap": None if valid_ap is None else float(valid_ap),
        "selection_metric": "full official Valid macro AP (official OGB Evaluator)",
        "recipe": C.recipe("FIXED1", int(seed), int(horizon)),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "created_utc": C.utcnow(),
        "test_role": FC.TEST_ROLE,
        "test_used_for_selection": False,
        "test_scored": False,
    }
    if extra:
        payload.update(dict(extra))
    return payload


# ---------------------------------------------------------------------------
# cost measurement
# ---------------------------------------------------------------------------


def preflight(task: Any, *, device: torch.device, steps: int, out: Path) -> dict[str, Any]:
    t_start = time.perf_counter()
    plan, plan_audit = C.plan_for_seed(task, FC.SEED)
    model, model_audit = FM.build_frs_model(task.spec, task.norm, C.N_TASKS, arm="TIED",
                                            seed=FC.SEED, device=device)
    optimizer, _ = build_optimizer(model)
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    batches = plan[0]["batches"]
    step_times: list[float] = []
    assemble_times: list[float] = []
    model.train()
    for i in range(int(steps)):
        t0 = time.perf_counter()
        batch = stager.batch(C.PC.collate(task.cache, batches[i].tolist()))
        y = stager.labels(task.labels[batches[i]])
        assemble_times.append(float(time.perf_counter() - t0))
        t1 = time.perf_counter()
        train_step(model, optimizer, batch, y, global_step=i, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(float(time.perf_counter() - t1))
    stager.release()
    arr = np.asarray(step_times, dtype=np.float64)
    step_mean = float(arr.mean())
    t_valid = time.perf_counter()
    valid = C.evaluate_split(model, task, device, "valid", per_task=True, reason="preflight")
    valid_seconds = float(time.perf_counter() - t_valid)
    t_train = time.perf_counter()
    train_rep = C.evaluate_split(model, task, device, "train", per_task=False, reason="preflight")
    train_seconds = float(time.perf_counter() - t_train)
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    per_epoch_train = step_mean * int(C.STEPS_PER_EPOCH)
    per_epoch = per_epoch_train + valid_seconds
    budget_seconds = 40.0 * 60.0  # per-arm wall clock target inside the 60 min round
    overhead = 25.0  # checkpoint + usage metric + csv writes

    def project(horizon: int) -> dict[str, Any]:
        n_train_eval = len([e for e in FC.TRAIN_EVAL_EPOCHS if e <= horizon])
        train_eval_seconds = (n_train_eval * train_seconds) + train_seconds  # + best checkpoint
        total = horizon * (per_epoch + overhead) + train_eval_seconds
        return {
            "horizon": int(horizon),
            "projected_seconds": float(total),
            "projected_minutes": float(total / 60.0),
            "projected_gpu_hours_per_arm": float(total / 3600.0),
            "projected_gpu_hours_two_arms_parallel": float(total / 3600.0),
            "n_train_evals": int(n_train_eval) + 1,
        }

    horizon = FC.PREFERRED_HORIZON
    proj = project(horizon)
    if proj["projected_seconds"] > budget_seconds:
        horizon = FC.FALLBACK_HORIZON
        proj = project(horizon)
        decision = "FALLBACK_8"
    else:
        decision = "PREFERRED_12"
    payload = {
        "schema": FC.SCHEMA_PREFLIGHT,
        "round_id": FC.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "device": str(device),
        "batch_size": int(C.BATCH_SIZE),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "measured_steps": int(steps),
        "step_ms_mean": float(step_mean * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "assemble_ms_mean": float(np.mean(assemble_times) * 1e3),
        "train_epoch_seconds_projected": float(per_epoch_train),
        "valid_eval_seconds": valid_seconds,
        "train_eval_seconds": train_seconds,
        "per_epoch_seconds_projected": float(per_epoch),
        "per_arm_overhead_seconds": overhead,
        "gpu_peak_allocated_bytes": peak_alloc,
        "parameters": int(model_audit["parameters"]["total"]),
        "valid_ap_at_init": float(valid["ap"]),
        "train_ap_at_init": float(train_rep["ap"]),
        "plan": plan_audit,
        "budget_seconds_per_arm": budget_seconds,
        "projections": [project(h) for h in FC.HORIZON_GRID],
        "chosen_horizon": int(horizon),
        "decision": decision,
        "measured_seconds": float(time.perf_counter() - t_start),
        "note": "projection covers two arms running in parallel (wall clock = one arm); GPU-hours summed separately in the report",
    }
    FC.write_json(Path(out), payload)
    FC.log(f"preflight: step {step_mean*1e3:.1f} ms, valid {valid_seconds:.1f}s, train {train_seconds:.1f}s, "
           f"projected {proj['projected_minutes']:.1f} min @ {horizon} epochs -> {decision}")
    return payload


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def train(
    task: Any,
    *,
    arm: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    soft_seconds: float,
    hard_seconds: float,
    mechanism_rows: int,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs)
    if epochs not in FC.HORIZON_GRID:
        raise RuntimeError(f"horizon {epochs} outside the frozen grid {FC.HORIZON_GRID}")
    eval_protocol = evaluation_protocol(epochs)
    torch.manual_seed(int(seed))
    C.seed_everything(int(seed))
    rng_start = C.rng_state_hashes(C.capture_rng_state())
    plan, plan_audit = C.plan_for_seed(task, int(seed))
    model, model_audit = FM.build_frs_model(task.spec, task.norm, C.N_TASKS, arm=str(arm),
                                            seed=int(seed), device=device)
    model_audit = dict(model_audit)
    model_audit["initial_state_sha256"] = C.state_fingerprint(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )
    optimizer, opt_audit = build_optimizer(model)
    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    counts = C.SD.task_counts(task)
    diag_rows = [int(x) for x in task.valid_rows[: int(C.DIAG_VALID_ROWS)]]
    diag_batch = C.PC.collate(task.cache, diag_rows)
    diag_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in diag_batch.items()}

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    usage_history: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_valid.csv"
    usage_path = out_dir / "structural_usage.json"
    best_path = out_dir / "checkpoint_best.pt"
    last_path = out_dir / "checkpoint_last.pt"

    def eval_all(epoch: int, *, train_eval: bool) -> tuple[dict[str, Any], float]:
        before = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            valid_rep = C.evaluate_split(model, task, device, "valid", per_task=True,
                                         reason=f"epoch_{epoch}_full_valid")
            train_rep = (
                C.evaluate_split(model, task, device, "train", per_task=False,
                                 reason=f"epoch_{epoch}_full_train")
                if train_eval else None
            )
        seconds = float(time.perf_counter() - t0)
        after = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("evaluation moved the training RNG state")
        model.train()
        return {"valid": valid_rep, "train": train_rep}, seconds

    best_epoch: int | None = None
    best_valid_ap = -math.inf
    best_sha = ""
    last_sha = ""
    best_valid_rep: dict[str, Any] | None = None
    epochs_run = 0
    stop_reason = "RUNNING"
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    for epoch in range(1, epochs + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events = 0
        lowrank_grad_acc = {
            f"lowrank.{b}.{p}.weight": [] for b in (0, 1) for p in ("A", "B")
        }
        lowrank_total: list[float] = []
        inherited_total: list[float] = []
        w2_total: list[float] = []
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = train_step(model, optimizer, batch, y, global_step=epochs_run * int(C.STEPS_PER_EPOCH) + len(losses),
                             clip=clip)
            losses.append(out["loss"])
            grad_norms.append(out["grad_norm"])
            clip_events += int(out["clipped"])
            lowrank_total.append(float(out["grad_norm_lowrank_total"]))
            inherited_total.append(float(out["grad_norm_inherited_new_total"]))
            w2_total.append(float(out["grad_norm_w2_total"]))
            for name, value in out["lowrank_grads"].items():
                if value is not None:
                    lowrank_grad_acc[name].append(float(value))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            FC.log(f"[{arm}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        train_eval = epoch in eval_protocol["train_eval_epochs"]
        reports, eval_seconds = eval_all(epoch, train_eval=train_eval)
        valid_rep = reports["valid"]
        valid_ap = float(valid_rep["ap"])
        for t, ap in enumerate(valid_rep["per_task_ap"]):
            per_task_rows.append(
                {
                    "arm": arm, "epoch": int(epoch), "split": "valid", "task": int(t),
                    "ap": "" if ap is None else float(ap),
                    "observed": int(valid_rep["observed"][t]),
                    "positive": int(valid_rep["positive"][t]),
                    "negative": int(valid_rep["negative"][t]),
                }
            )
        strict = best_epoch is None or valid_ap > float(best_valid_ap)
        if strict:
            best_epoch, best_valid_ap = int(epoch), valid_ap
            best_valid_rep = {
                "epoch": int(epoch),
                "valid": {k: valid_rep[k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored")},
            }
            payload = checkpoint_payload(
                arm=arm, seed=int(seed), horizon=epochs, epoch=int(epoch),
                global_step=int(epoch * C.STEPS_PER_EPOCH), model=model, optimizer=optimizer,
                rng_state=C.capture_rng_state(), plan_meta=C.plan_meta(plan, epoch, int(seed)),
                model_audit=model_audit, valid_ap=best_valid_ap, kind="best_valid_ap",
                extra={"optimizer_state_audit": opt_audit},
            )
            best_sha = _atomic_save(payload, best_path)
            FC.write_json(out_dir / "best_epoch_reports.json", best_valid_rep)
        usage = diagnostic_usage(model, diag_batch)
        usage_history.append({"epoch": int(epoch), "global_step": int(epoch * C.STEPS_PER_EPOCH), **usage})
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        update_peak()
        epochs_run = int(epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        cumulative = float(time.perf_counter() - t_start)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row: dict[str, Any] = {
            "arm": arm,
            "epoch": int(epoch),
            "global_step": int(epoch * C.STEPS_PER_EPOCH),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce_step_mean": float(np.mean(losses)) if losses else float("nan"),
            "train_evaluated": bool(train_eval),
            "train_bce": (float(reports["train"]["bce"]) if reports["train"] else None),
            "train_ap": (float(reports["train"]["ap"]) if reports["train"] else None),
            "valid_bce": float(valid_rep["bce"]),
            "valid_ap": valid_ap,
            "n_tasks_scored_valid": int(valid_rep["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "valid_seconds": float(valid_rep["seconds"]),
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "gap_norm_max": float(max(grad_norms)) if grad_norms else None,
            "gap_clip_events": int(clip_events),
            "best_epoch_so_far": int(best_epoch),
            "best_valid_ap_so_far": float(best_valid_ap),
            "is_best": bool(strict),
        }
        row.update({k: usage.get(k) for k in (
            "R1", "R2", "Rtotal", "rms_h1_minus_h0", "rms_h2_minus_h1",
            "rms_z1", "rms_m1", "rms_z_target", "rms_m2",
            "rms_fc1_term_0", "rms_correction_0", "rms_fc1_term_1", "rms_correction_1",
            "lowrank_weight_norm_0", "lowrank_weight_norm_1",
        )})
        row["grad_norm_lowrank_0_A"] = _mean(lowrank_grad_acc["lowrank.0.A.weight"])
        row["grad_norm_lowrank_0_B"] = _mean(lowrank_grad_acc["lowrank.0.B.weight"])
        row["grad_norm_lowrank_1_A"] = _mean(lowrank_grad_acc["lowrank.1.A.weight"])
        row["grad_norm_lowrank_1_B"] = _mean(lowrank_grad_acc["lowrank.1.B.weight"])
        row["grad_norm_lowrank_total"] = _mean(lowrank_total)
        row["grad_norm_inherited_new_total"] = _mean(inherited_total)
        row["grad_norm_w2_total"] = _mean(w2_total)
        curve_rows.append(row)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        FC.write_json(
            usage_path,
            {
                "schema": "fsr-structural-usage-v1",
                "round_id": FC.ROUND_ID,
                "arm": arm,
                "diagnostic_batch": "first official Validation rows, eval mode",
                "history": usage_history,
            },
        )
        FC.log(
            f"[{arm}] ep {epoch}/{epochs} bce {row['train_bce_step_mean']:.6f} "
            f"validAP {valid_ap:.6f} LR {row['learning_rate']:.3e} "
            f"train {train_seconds:.1f}s valid {row['valid_seconds']:.1f}s "
            f"{'*best*' if strict else ''}"
        )
        payload_last = checkpoint_payload(
            arm=arm, seed=int(seed), horizon=epochs, epoch=int(epoch),
            global_step=int(epoch * C.STEPS_PER_EPOCH), model=model, optimizer=optimizer,
            rng_state=C.capture_rng_state(), plan_meta=C.plan_meta(plan, epoch, int(seed)),
            model_audit=model_audit, valid_ap=valid_ap, kind="last_epoch",
        )
        last_sha = _atomic_save(payload_last, last_path)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    completed = epochs_run >= epochs and best_epoch is not None
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"

    # ---- best-checkpoint evaluation (Valid per task + Train AP/BCE) --------
    best_eval: dict[str, Any] | None = None
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            breps = {
                "valid": C.evaluate_split(model, task, device, "valid", per_task=True,
                                          reason="best_checkpoint_valid"),
                "train": C.evaluate_split(model, task, device, "train", per_task=False,
                                          reason="best_checkpoint_train"),
            }
        best_eval = {
            "epoch": int(payload["epoch"]),
            "seconds": float(time.perf_counter() - t0),
            "valid": {k: breps["valid"][k] for k in (
                "ap", "bce", "per_task_ap", "observed", "positive", "negative",
                "n_tasks_scored", "logits_sha256", "probs_sha256")},
            "train": {k: breps["train"][k] for k in ("ap", "bce", "n_tasks_scored", "logits_sha256")},
            "selection_metric": "valid.ap",
            "test_scored": False,
        }
        best_eval["valid_ap_minus_recorded"] = float(best_eval["valid"]["ap"]) - float(best_valid_ap)
        if abs(best_eval["valid_ap_minus_recorded"]) > 1e-4:
            raise RuntimeError(
                f"best-checkpoint Valid AP {best_eval['valid']['ap']} != recorded {best_valid_ap}"
            )
        FC.write_json(out_dir / "best_checkpoint_eval.json", best_eval)

    # ---- mechanism diagnostics (no interpretation pipeline) ----------------
    mechanism = MECH.run(model=model, task=task, device=device, arm=str(arm),
                         best_checkpoint=best_path if best_path.is_file() else None,
                         model_audit=model_audit, n_rows=int(mechanism_rows))
    mechanism["best_epoch"] = None if best_epoch is None else int(best_epoch)
    mechanism["completed"] = bool(completed)
    FC.write_json(out_dir / "mechanism.json", mechanism)

    total_seconds = float(time.perf_counter() - t_start)
    final_row = curve_rows[-1] if curve_rows else {}
    valid_rows = [r for r in curve_rows if r.get("valid_ap") is not None]
    last3 = valid_rows[-3:]
    run = {
        "schema": FC.SCHEMA_RUN,
        "round_id": FC.ROUND_ID,
        "protocol_id": FC.PROTOCOL_ID,
        "arm": str(arm),
        "base_computation": "FIXED1",
        "role_branch": "low-rank correction uses z + m (same input as fc1)" if arm == "TIED"
        else "low-rank correction uses z - m (role-separated input)",
        "seed": int(seed),
        "horizon": int(epochs),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "parameter_audit": model_audit["parameters"],
        "rank": int(FC.RANK),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(epochs_run * C.STEPS_PER_EPOCH),
        "eval_protocol": eval_protocol,
        "task_counts": counts,
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "plan_identity": plan_audit,
        "batch_plan": C.plan_meta(plan, int(epochs_run), int(seed)),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_valid_ap),
        "best_valid_bce": (None if best_valid_rep is None else float(best_valid_rep["valid"]["bce"])),
        "horizon_endpoint": {
            "epoch": int(epochs_run),
            "valid_ap": final_row.get("valid_ap"),
            "valid_bce": final_row.get("valid_bce"),
            "train_ap": final_row.get("train_ap"),
            "train_bce": final_row.get("train_bce"),
        },
        "last3_mean_valid_ap": (float(np.mean([float(r["valid_ap"]) for r in last3])) if last3 else None),
        "last3_epochs": [int(r["epoch"]) for r in last3],
        "best_checkpoint_eval": best_eval,
        "best_checkpoint": C.track_rel(best_path) if best_path.is_file() else None,
        "best_checkpoint_sha256": best_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.is_file() else None,
        "last_checkpoint_sha256": last_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": C.sha256_file(curve_path),
        "per_task_valid": C.track_rel(per_task_path),
        "per_task_valid_sha256": C.sha256_file(per_task_path),
        "structural_usage": C.track_rel(usage_path),
        "mechanism": C.track_rel(out_dir / "mechanism.json"),
        "recipe": C.recipe("FIXED1", int(seed), int(epochs)),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "tf32": {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "grad_clip": clip,
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "eval_seconds": float(sum(float(r.get("epoch_seconds") or 0.0) - float(r.get("epoch_train_seconds") or 0.0)
                                  for r in curve_rows)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "rng_state_hashes_start": rng_start,
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "rng_epoch_hashes": rng_epoch_hashes,
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_role": FC.TEST_ROLE,
        "test_used_for_selection": False,
        "test_scored": False,
        "inheritance": {
            "source_round": FC.HIST["source_round"],
            "FIXED1_best_valid_ap_25": FC.HIST["best_valid_ap_25"],
            "W2_best_valid_ap_40": FC.HIST["w2_best_valid_ap_40"],
            "historical_same_horizon_reference": "results/runs/FIXED1_seed0 (25 epochs, real curve)",
        },
    }
    FC.write_json(out_dir / "run.json", run)
    FC.log(f"done arm={arm} status={status} epochs={epochs_run}/{epochs} "
           f"best_epoch={best_epoch} best_valid={best_valid_ap} "
           f"gpu_hours={run['gpu_hours']:.4f}")
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "train"], default="train")
    parser.add_argument("--arm", choices=list(FC.ARMS), default="TIED")
    parser.add_argument("--seed", type=int, default=FC.SEED)
    parser.add_argument("--epochs", type=int, default=FC.PREFERRED_HORIZON)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--soft-seconds", type=float, default=2400.0)
    parser.add_argument("--hard-seconds", type=float, default=3120.0)
    parser.add_argument("--mechanism-rows", type=int, default=64)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.mode == "preflight":
        out = Path(args.out_dir or (FC.ANALYSIS / "preflight_timing.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        task = C.SD.load_task()
        payload = preflight(task, device=device, steps=int(args.steps), out=out)
        return 0 if payload else 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and not torch.cuda.is_available():
        FC.write_json(out_dir / "run.json", {"schema": FC.SCHEMA_RUN, "round_id": FC.ROUND_ID,
                                             "arm": args.arm, "status": "INCOMPLETE",
                                             "stop_reason": "CUDA_UNAVAILABLE",
                                             "finished_utc": C.utcnow()})
        return 2
    try:
        task = C.SD.load_task()
        run = train(task, arm=str(args.arm), seed=int(args.seed), out_dir=out_dir, device=device,
                    epochs=int(args.epochs), soft_seconds=float(args.soft_seconds),
                    hard_seconds=float(args.hard_seconds), mechanism_rows=int(args.mechanism_rows))
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        FC.write_json(out_dir / "run.failed.json", {
            "schema": FC.SCHEMA_RUN, "round_id": FC.ROUND_ID, "arm": args.arm,
            "seed": int(args.seed), "status": "FAILED", "stop_reason": type(exc).__name__,
            "detail": str(exc), "traceback": traceback.format_exc(),
            "environment": C.environment(), "finished_utc": C.utcnow(),
        })
        FC.log(f"FAILED {args.arm}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
