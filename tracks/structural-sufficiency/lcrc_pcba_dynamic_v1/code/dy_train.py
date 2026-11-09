"""Formal from-scratch training for W2-DYNAMIC (one model, one seed, 40 epochs).

The training step is exposed as :func:`train_step` so the pre-training profiling
gate measures the *real* step (same collate/stager/model/optimizer/clip/loss code
path), never a re-implementation.

Protocol (frozen in ``PROTOCOL.md``): AdamW, weight decay 1e-3, 1-epoch linear
warmup 3e-5 -> 3e-4 then per-step cosine 3e-4 -> 1e-5 over the 40-epoch horizon,
dropout 0.10 (inherited W2 ``drop``), gradient clip 5.0, FP32, masked BCE,
batch 128, the historical W2 seed-0 40-epoch batch-order plan, official
Validation AP every epoch.

Every epoch also records, on one fixed small Validation batch, the dynamic-usage
RMS ratios ``R1``, ``R2``, ``Rtotal`` and the gradient norms of the new modules.

No official Test target is read, indexed or scored.
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

import dy_common as C  # noqa: E402
import dy_model as M  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce",
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
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
    "checkpoint_seconds",
    "R1",
    "R2",
    "Rtotal",
    "rms_h1_minus_h0",
    "rms_h2_minus_h1",
    "rms_h2_minus_h0",
    "fc2_norm_blocks0",
    "fc2_norm_blocks1",
    "bond_proj_weight_norm",
    "finite_fraction_h2",
    "cos_h0_h1",
    "grad_norm_bond_proj",
    "grad_norm_blocks0_fc1",
    "grad_norm_blocks0_fc2",
    "grad_norm_blocks1_fc1",
    "grad_norm_blocks1_fc2",
    "grad_norm_new_total",
    "grad_norm_w2_total",
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
]
GRAD_FIELDS = [
    "epoch",
    "step",
    "bond_proj.weight",
    "bond_proj.bias",
    "blocks.0.fc1.weight",
    "blocks.0.fc1.bias",
    "blocks.0.fc2.weight",
    "blocks.0.fc2.bias",
    "blocks.1.fc1.weight",
    "blocks.1.fc1.bias",
    "blocks.1.fc2.weight",
    "blocks.1.fc2.bias",
]
GRAD_TENSORS = tuple(GRAD_FIELDS[2:])


# ---------------------------------------------------------------------------
# the single training step (shared with the profiling gate)
# ---------------------------------------------------------------------------


def grad_norms(model: torch.nn.Module) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, p in model.named_parameters():
        if not C.is_new_parameter(name):
            continue
        out[name] = None if p.grad is None else float(p.grad.detach().norm().item())
    return out


def group_norm(model: torch.nn.Module, *, new: bool) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if C.is_new_parameter(name) != bool(new) or p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y: torch.Tensor,
    *,
    global_step: int,
    clip: float,
    collect_grads: bool = True,
) -> dict[str, Any]:
    lr = C.set_lrs(optimizer, step=int(global_step))
    logits = model(batch)
    if tuple(logits.shape) != (int(y.shape[0]), C.N_TASKS):
        raise RuntimeError(f"model output {tuple(logits.shape)} != {(int(y.shape[0]), C.N_TASKS)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits at step {global_step}")
    loss = C.MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = grad_norms(model) if collect_grads else {}
    new_norm = group_norm(model, new=True) if collect_grads else float("nan")
    w2_norm = group_norm(model, new=False) if collect_grads else float("nan")
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if not math.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "grad_norm": gnorm,
        "lr": lr,
        "clipped": bool(gnorm > clip),
        "new_grads": grads,
        "grad_norm_new_total": new_norm,
        "grad_norm_w2_total": w2_norm,
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


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


def plan_meta(plan: Sequence[Mapping[str, Any]], consumed: int, order_seed: int) -> dict[str, Any]:
    prefix = [p["row_order_sha256"] for p in plan[: int(consumed)]]
    return {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "order_seed": int(order_seed),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(len(plan)),
        "consumed_epochs": int(consumed),
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode()),
    }


def diagnostic_batch(task: Any, device: torch.device) -> dict[str, Any]:
    rows = [int(x) for x in task.valid_rows[: int(C.DIAG_VALID_ROWS)]]
    batch = C.PC.collate(task.cache, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "n_param_groups": len(optimizer.param_groups),
    }


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def build_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "single_group_same_lr_for_w2_and_dynamic": True,
        "scheduler": "per_step_cosine_40epoch_with_1epoch_linear_warmup",
    }
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("this round uses exactly one parameter group (one LR for every parameter)")
    return optimizer, audit


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
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": C.MODEL_NAME,
        "seed": int(C.SEED),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "state_dict": C.NM.state_cpu(model),
        "optimizer_state_dict": C.NM.optimizer_state_cpu(optimizer),
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
            "n_tasks_with_observed_labels": int(valid_report["n_tasks_with_observed_labels"]),
            "valid_logits_sha256": valid_report.get("valid_logits_sha256"),
            "valid_probs_sha256": valid_report.get("valid_probs_sha256"),
        },
        "recipe": C.recipe(),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_evaluated": False,
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed before the freeze",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore(
    payload: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("model") != C.MODEL_NAME:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != requested {C.MODEL_NAME}")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    if int(payload["global_step"]) > 0 and steps != [float(payload["global_step"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != payload {payload['global_step']}")
    hashes = C.NM.restore_rng_state(payload["rng_state"])
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
    seed: int,
    out_dir: Path,
    device: torch.device,
    epochs: int = C.TOTAL_EPOCHS,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs)
    if epochs != int(C.TOTAL_EPOCHS):
        raise RuntimeError(f"this round is fixed at {C.TOTAL_EPOCHS} epochs, got {epochs}")
    formal = epochs == int(C.TOTAL_EPOCHS)
    order_seed = int(seed)
    gpu_start = C.gpu_occupancy()

    model, model_audit = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device
    )
    optimizer, opt_audit = build_optimizer(model)

    plan = C.SD.make_plan40(task, seed=order_seed)
    plan_audit = C.SD.plan_summary(plan, task)
    got_plan = plan[0]["global_plan_sha256"]
    if got_plan != C.FROZEN_PLAN["global_plan_sha256"]:
        raise RuntimeError(f"order plan {got_plan} != frozen W2 plan {C.FROZEN_PLAN['global_plan_sha256']}")
    consumed_prefix = C.sha256_bytes("|".join(p["row_order_sha256"] for p in plan).encode())
    if consumed_prefix != C.FROZEN_PLAN["consumed40_prefix_sha256"]:
        raise RuntimeError(f"40-epoch plan prefix {consumed_prefix} != frozen W2 prefix")

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer)
        saved = resume.get("batch_plan") or {}
        now = plan_meta(plan, int(resume["epoch"]), order_seed)
        if now["global_plan_sha256"] != saved.get("global_plan_sha256") or now["prefix_sha256"] != saved.get("prefix_sha256"):
            raise RuntimeError("resumed plan does not match the checkpoint plan")
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    order_path = out_dir / "training_order.json"
    C.write_json(order_path, {"plan": plan_meta(plan, int(epochs), order_seed), **plan_audit})

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    counts = C.SD.task_counts(task)
    train_counts = counts["train"]
    valid_pos = counts["valid"]["positive"]
    strata = [
        "support_lt5" if valid_pos[t] < 5 else "support_5_19" if valid_pos[t] < 20 else "support_ge20"
        for t in range(C.N_TASKS)
    ]

    diag = diagnostic_batch(task, device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    usage_history: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    grad_path = out_dir / "grad_history.csv"
    usage_path = out_dir / "dynamic_usage.json"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{epochs}.pt"
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
    clip_events_total = 0
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    train_eval_summary: dict[str, Any] | None = None
    epoch0_report: dict[str, Any] | None = None

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

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
                }
            )
        update_peak()

    # epoch-0 diagnostic: initialisation state, no training step, RNG untouched
    if resume is None:
        model.eval()
        t_eval = time.perf_counter()
        with torch.inference_mode():
            report0 = C.SD.evaluate_validation(model, task, device, reason="epoch_0_initialisation_diagnostic")
        epoch0_usage = M.usage_metrics(model, diag)
        epoch0_report = {
            "epoch": 0,
            "ap": float(report0["ap"]),
            "valid_bce": float(report0["valid_bce"]),
            "n_tasks_scored": int(report0["n_tasks_scored"]),
            "seconds": float(time.perf_counter() - t_eval),
            "global_step": 0,
            "dynamic_usage": epoch0_usage,
            "rng_after": C.rng_state_hashes(C.capture_rng_state()),
        }
        if epoch0_report["rng_after"]["combined_sha256"] != rng_start_hashes["combined_sha256"]:
            raise RuntimeError("epoch-0 diagnostic evaluation changed the RNG state")
        # The untrained model must reproduce the historical untrained W2 AP.
        if abs(epoch0_report["ap"] - C.W2_SEED0["epoch0_valid_ap"]) > 1e-4:
            raise RuntimeError(
                f"epoch-0 Valid AP {epoch0_report['ap']} != historical untrained W2 "
                f"{C.W2_SEED0['epoch0_valid_ap']}"
            )
        update_peak()
        C.log(
            f"[{C.MODEL_NAME}] epoch0 diagnostic validAP {epoch0_report['ap']:.9f} "
            f"(W2 {C.W2_SEED0['epoch0_valid_ap']:.9f}) Rtotal {epoch0_usage['Rtotal']}"
        )

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
        losses: list[float] = []
        step_times: list[float] = []
        grad_norms_total: list[float] = []
        grad_acc: dict[str, list[float]] = {name: [] for name in GRAD_FIELDS[2:]}
        new_total_epoch: list[float] = []
        w2_total_epoch: list[float] = []
        clip_events_epoch = 0
        aborted = False
        for step, rows in enumerate(batches, start=1):
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = train_step(
                model, optimizer, batch, y, global_step=global_steps_run, clip=clip, collect_grads=True
            )
            global_steps_run += 1
            losses.append(out["loss"])
            grad_norms_total.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            new_total_epoch.append(float(out["grad_norm_new_total"]))
            w2_total_epoch.append(float(out["grad_norm_w2_total"]))
            row = {"epoch": int(epoch), "step": int(global_steps_run)}
            for name in GRAD_FIELDS[2:]:
                value = out["new_grads"].get(name)
                row[name] = None if value is None else float(value)
                if value is not None:
                    grad_acc[name].append(float(value))
            grad_rows.append(row)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{C.MODEL_NAME}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        report: dict[str, Any] | None = None
        valid_seconds = 0.0
        strict = False
        if True:  # official Validation AP is evaluated every epoch (frozen protocol)
            model.eval()
            t_eval = time.perf_counter()
            with torch.inference_mode():
                report = C.SD.evaluate_validation(
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
                    plan=plan_meta(plan, int(epoch), order_seed),
                    model_audit=model_audit,
                    valid_ap=best_ap,
                    valid_report=report,
                    kind="best_valid_ap",
                    extra={"optimizer_state_audit": opt_audit},
                )
                best_checkpoint_sha = _atomic_save(payload, best_path)
            record_eval(epoch, report)
            if epoch == int(epochs):
                t_tr = time.perf_counter()
                with torch.inference_mode():
                    train_report = C.SD.evaluate_train(model, task, device)
                train_eval_path = out_dir / f"train_eval_epoch{int(epoch)}.json"
                C.write_json(
                    train_eval_path,
                    {
                        "epoch": int(epoch),
                        "model": C.MODEL_NAME,
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
                    "eval_seconds": float(time.perf_counter() - t_tr),
                    "path": C.track_rel(train_eval_path),
                }
            model.train()
        usage = M.usage_metrics(model, diag)
        usage_history.append({"epoch": int(epoch), "global_step": int(global_steps_run), **usage})
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        update_peak()
        epochs_run = int(epoch)
        cumulative = float(time.perf_counter() - t_start)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row = {
            "epoch": int(epoch),
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce": float(np.mean(losses)) if losses else float("nan"),
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
            "steps_per_second": float(len(step_times) / train_seconds) if train_seconds > 0 else float("nan"),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "grad_norm_max": float(max(grad_norms_total)) if grad_norms_total else None,
            "grad_norm_mean": float(np.mean(grad_norms_total)) if grad_norms_total else None,
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": None if best_epoch is None else int(best_epoch),
            "best_valid_ap_so_far": None if best_epoch is None else float(best_ap),
            "is_best": bool(strict),
            "checkpoint_seconds": None,
            "R1": usage["R1"],
            "R2": usage["R2"],
            "Rtotal": usage["Rtotal"],
            "rms_h1_minus_h0": usage["rms_h1_minus_h0"],
            "rms_h2_minus_h1": usage["rms_h2_minus_h1"],
            "rms_h2_minus_h0": usage["rms_h2_minus_h0"],
            "fc2_norm_blocks0": usage["fc2_weight_norm"]["blocks.0"],
            "fc2_norm_blocks1": usage["fc2_weight_norm"]["blocks.1"],
            "bond_proj_weight_norm": usage["bond_proj_weight_norm"],
            "finite_fraction_h2": usage["finite_fraction_h2"],
            "cos_h0_h1": usage["cos_h0_h1"],
            "grad_norm_bond_proj": _mean([v for v in grad_acc.get("bond_proj.weight", [])]),
            "grad_norm_blocks0_fc1": _mean(grad_acc.get("blocks.0.fc1.weight", [])),
            "grad_norm_blocks0_fc2": _mean(grad_acc.get("blocks.0.fc2.weight", [])),
            "grad_norm_blocks1_fc1": _mean(grad_acc.get("blocks.1.fc1.weight", [])),
            "grad_norm_blocks1_fc2": _mean(grad_acc.get("blocks.1.fc2.weight", [])),
            "grad_norm_new_total": _mean(new_total_epoch),
            "grad_norm_w2_total": _mean(w2_total_epoch),
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
        if epoch % 5 == 0 or epoch == int(epochs):
            _write_csv(grad_path, GRAD_FIELDS, grad_rows)
        C.write_json(
            usage_path,
            {
                "schema": "lcrc-pcba-dynamic-usage-v1",
                "round_id": C.ROUND_ID,
                "diagnostic_rows": int(C.DIAG_VALID_ROWS),
                "diagnostic_batch": "first official Validation rows, eval mode (dropout identity)",
                "history": usage_history,
            },
        )
        valid_txt = "" if report is None else f"{float(report['ap']):.6f}"
        C.log(
            f"[{C.MODEL_NAME}] epoch {epoch}/{int(epochs)} bce {row['train_bce']:.6f} "
            f"validAP {valid_txt} LR {row['learning_rate']:.3e} "
            f"R1 {row['R1']:.4f} R2 {row['R2']:.4f} Rtot {row['Rtotal']:.4f} "
            f"train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
            f"step_p50 {row['step_ms_p50']:.2f}ms{' *best*' if strict else ''}"
        )

        if epoch == int(epochs):
            payload = checkpoint_payload(
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                plan=plan_meta(plan, int(epoch), order_seed),
                model_audit=model_audit,
                valid_ap=None if report is None else float(report["ap"]),
                valid_report=report,
                kind=f"epoch{int(epochs)}_primary",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "train_eval": None if train_eval_summary is None else dict(train_eval_summary),
                    "dynamic_usage": usage,
                    "status_at_save": "COMPLETED",
                },
            )
            final_checkpoint_sha = _atomic_save(payload, final_path)

        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            plan=plan_meta(plan, int(epoch), order_seed),
            model_audit=model_audit,
            valid_ap=None if report is None else float(report["ap"]),
            valid_report=None,
            kind="last_epoch_resumable",
            extra={"status_at_save": "RUNNING_OR_COMPLETED", "optimizer_state_audit": opt_audit},
        )
        last_checkpoint_sha = _atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        _write_csv(curve_path, CURVE_FIELDS, curve_rows)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    completed = (
        epochs_run >= int(epochs)
        and set(eval_completed) == set(range(1, int(epochs) + 1))
        and bool(final_checkpoint_sha)
    )
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH_OR_EVAL"

    rng_end_hashes = C.rng_state_hashes(C.capture_rng_state())
    torch.save(
        {
            "schema": "lcrc-pcba-dynamic-rng-v1",
            "round_id": C.ROUND_ID,
            "model": C.MODEL_NAME,
            "seed": int(seed),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    curve_sha = _write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    grad_sha = _write_csv(grad_path, GRAD_FIELDS, grad_rows)
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "model": C.MODEL_NAME,
        "role": "diagnostic_boundary_experiment_two_layer_real_bond_mp",
        "seed": int(seed),
        "order_seed": int(order_seed),
        "formal_run": bool(formal),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "parameter_groups": model_audit["parameters"]["groups"],
        "final_epoch": int(epochs),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_requested": list(range(1, int(epochs) + 1)),
        "eval_epochs_completed": sorted(eval_completed),
        "all_eval_epochs_completed": bool(set(eval_completed) == set(range(1, int(epochs) + 1))),
        "epoch0_diagnostic": epoch0_report,
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "train_eval_final": train_eval_summary,
        "task_counts": counts,
        "task_strata": {str(t): strata[t] for t in range(C.N_TASKS)},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "batch_plan": plan_meta(plan, max(int(epochs_run), 0), order_seed),
        "frozen_plan": dict(C.FROZEN_PLAN),
        "training_order_summary": plan_audit["summary"],
        "training_order_isolation": plan_audit["isolation"],
        "training_order_path": C.track_rel(order_path),
        "training_order_sha256": C.sha256_file(order_path),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_ap),
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "primary_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "primary_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "per_task_ap": C.track_rel(per_task_path),
        "per_task_ap_sha256": per_task_sha,
        "grad_history": C.track_rel(grad_path),
        "grad_history_sha256": grad_sha,
        "dynamic_usage": C.track_rel(usage_path),
        "dynamic_usage_sha256": C.sha256_file(usage_path),
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "recipe": C.recipe(),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows])) if curve_rows else 0.0,
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
        "test_policy": "official Test targets are NaN in the label artifact and are never indexed before the freeze",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done model={C.MODEL_NAME} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_ap={best_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=C.TOTAL_EPOCHS)
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
                "model": C.MODEL_NAME,
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
        task = C.SD.load_task()
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        run = train(
            task,
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            epochs=int(args.epochs),
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
                "model": C.MODEL_NAME,
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
        C.log(f"FAILED model={C.MODEL_NAME}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
