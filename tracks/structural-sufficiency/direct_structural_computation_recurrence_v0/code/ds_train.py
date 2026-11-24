"""Formal from-scratch training for one DSCR v0 arm.

The training step, the frozen recipe, the data path and the order plans are the
historical W2 / W2-DYNAMIC ones.  The only addition is the per-epoch evaluation
protocol of this round: **every** epoch scores the complete official Train,
Valid and Test splits in eval mode (official OGB macro AP + masked BCE), while
checkpoint selection reads only the full official Valid AP.

Test numbers are written to the curve and to separate files and are never read
by any decision function in this module.
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
import torch.nn as nn  # noqa: E402

import ds_common as C  # noqa: E402
import ds_model as M  # noqa: E402
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
    "valid_bce",
    "valid_ap",
    "test_bce",
    "test_ap",
    "n_tasks_scored_train",
    "n_tasks_scored_valid",
    "n_tasks_scored_test",
    "epoch_train_seconds",
    "train_eval_seconds",
    "valid_seconds",
    "test_seconds",
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
    "R1",
    "R2",
    "Rtotal",
    "rms_h1_minus_h0",
    "rms_h2_minus_h1",
    "fc2_norm_blocks0",
    "fc2_norm_blocks1",
    "bond_proj_weight_norm",
    "cos_h0_h1",
    "n_directed_edges",
    "n_pairs",
    "mean_degree",
    "max_degree",
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
PER_TASK_FIELDS = ["epoch", "split", "task", "ap", "observed", "positive", "negative"]
GRAD_TENSORS = (
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
)


# ---------------------------------------------------------------------------
# training step
# ---------------------------------------------------------------------------


def _new_grad_norms(model: nn.Module) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, p in model.named_parameters():
        if not C.is_new_parameter(name):
            continue
        out[name] = None if p.grad is None else float(p.grad.detach().norm().item())
    return out


def _group_norm(model: nn.Module, *, new: bool) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if C.is_new_parameter(name) != bool(new) or p.grad is None:
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
    audit_new: bool = True,
) -> dict[str, Any]:
    lr = C.set_lrs(optimizer, step=int(global_step))
    logits = model(batch)
    if tuple(logits.shape) != (int(y.shape[0]), C.N_TASKS):
        raise RuntimeError(f"model output {tuple(logits.shape)} != {(int(y.shape[0]), C.N_TASKS)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"non-finite logits at step {global_step}")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = _new_grad_norms(model) if audit_new else {}
    new_norm = _group_norm(model, new=True) if audit_new else float("nan")
    w2_norm = _group_norm(model, new=False) if audit_new else float("nan")
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
# small helpers
# ---------------------------------------------------------------------------


class Watchdog:
    def __init__(self, hard_seconds: float):
        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self._t0 = time.perf_counter()
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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


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
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "single_group_same_lr_for_every_parameter": True,
        "scheduler": "per_step_cosine_40epoch_with_1epoch_linear_warmup (horizon is a prefix)",
    }
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("this round uses exactly one parameter group")
    return optimizer, audit


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch_prefix",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(C.TOTAL_STEPS_40),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "n_param_groups": len(optimizer.param_groups),
    }


def diagnostic_batch(task: Any, device: torch.device) -> dict[str, Any]:
    rows = [int(x) for x in task.valid_rows[: int(C.DIAG_VALID_ROWS)]]
    batch = C.PC.collate(task.cache, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


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
    plan: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_ap: float | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": str(arm),
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
        "selection_metric": "full official Valid macro AP (official OGB Evaluator)",
        "recipe": C.recipe(arm, seed, horizon),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "created_utc": C.utcnow(),
        "test_role": C.TEST_ROLE,
        "test_used_for_selection": False,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def restore(payload: Mapping[str, Any], model: nn.Module, optimizer: torch.optim.Optimizer, *, arm: str, seed: int) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if str(payload.get("arm")) != str(arm) or int(payload.get("seed")) != int(seed):
        raise RuntimeError(f"checkpoint {payload.get('arm')}/seed{payload.get('seed')} != {arm}/seed{seed}")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
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
    arm: str,
    seed: int,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    resume: Mapping[str, Any] | None,
    soft_seconds: float,
    hard_seconds: float,
    test_labels: np.ndarray,
    gamma: float | None = None,
    prior_gpu_seconds: float = 0.0,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs)
    if epochs not in C.HORIZONS:
        raise RuntimeError(f"horizon {epochs} not in {C.HORIZONS}")
    order_seed = int(seed)
    gpu_start = C.gpu_occupancy()

    if arm == "W2":
        model, model_audit = M.build_w2(task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device)
    else:
        model, model_audit = M.build_ds_model(
            task.spec, task.norm, C.N_TASKS, arm=arm, seed=int(seed), device=device
        )
    model_audit = dict(model_audit)
    model_audit["initial_state_sha256"] = C.state_fingerprint(C.state_cpu(model))
    optimizer, opt_audit = build_optimizer(model)

    plan, plan_audit = C.plan_for_seed(task, order_seed)

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer, arm=arm, seed=int(seed))
        saved = resume.get("batch_plan") or {}
        now = C.plan_meta(plan, int(resume["epoch"]), order_seed)
        if now["global_plan_sha256"] != saved.get("global_plan_sha256") or now["prefix_sha256"] != saved.get("prefix_sha256"):
            raise RuntimeError("resumed plan does not match the checkpoint plan")
        if int(resume.get("horizon", epochs)) != epochs:
            raise RuntimeError("resumed horizon changed")
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        seed_info = C.seed_everything(int(seed))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    C.write_json(
        out_dir / "training_order.json",
        {"plan": C.plan_meta(plan, int(epochs), order_seed), **plan_audit},
    )

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    counts = C.SD.task_counts(task)
    strata = [
        "support_lt5" if counts["valid"]["positive"][t] < 5 else
        "support_5_19" if counts["valid"]["positive"][t] < 20 else "support_ge20"
        for t in range(C.N_TASKS)
    ]

    diag = diagnostic_batch(task, device) if arm != "W2" else None
    deep = arm != "W2"

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    usage_history: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    per_task_path = out_dir / "per_task_ap.csv"
    usage_path = out_dir / "structural_usage.json"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best.pt"
    final_path = out_dir / f"checkpoint_epoch{epochs}.pt"

    best_epoch: int | None = None
    best_valid_ap = -math.inf
    best_checkpoint_sha = ""
    last_checkpoint_sha = ""
    final_checkpoint_sha = ""
    eval_completed: list[int] = []
    resume_repair: dict[str, Any] | None = None
    if resume is not None:
        # a resume must never erase an already written curve: re-adopt it and
        # re-derive the frozen selection record (same rule: strict improvement, earlier epoch wins)
        curve_rows = _read_csv(curve_path)
        per_task_rows = _read_csv(per_task_path)
        if curve_rows:
            rows = [(int(r["epoch"]), float(r["valid_ap"])) for r in curve_rows if r.get("valid_ap")]
            if rows:
                best_epoch, best_valid_ap = max(rows, key=lambda item: (item[1], -item[0]))
            eval_completed = [int(r["epoch"]) for r in curve_rows]
            epochs_run = len(curve_rows)
            recorded = resume.get("valid_ap")
            last_row = curve_rows[-1] if curve_rows else {}
            if recorded is not None and last_row.get("valid_ap") and abs(float(recorded) - float(last_row["valid_ap"])) > 1e-5:
                raise RuntimeError(
                    f"resume: curve last {last_row['valid_ap']} disagrees with checkpoint {recorded}"
                )
            prev = out_dir / "run.failed.json"
            resume_repair = {
                "resumed_from_epoch": int(resume["epoch"]),
                "curve_rows_restored": len(curve_rows),
                "best_from_curve": {"epoch": int(best_epoch), "valid_ap": float(best_valid_ap)},
                "previous_status": None,
                "previous_detail": None,
            }
            if prev.is_file():
                old = C.read_json(prev)
                resume_repair["previous_status"] = old.get("status")
                resume_repair["previous_detail"] = old.get("detail")
                resume_repair["previous_stop_reason"] = old.get("stop_reason")
        usage_existing = C.read_json(usage_path) if usage_path.is_file() else None
        if usage_existing and usage_existing.get("history"):
            usage_history = list(usage_existing["history"])
    best_epoch_reports: dict[str, Any] = {}
    clip_events_total = 0
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    epoch0_report: dict[str, Any] | None = None
    action_path = out_dir / "best_epoch_reports.json"

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    def _eval_all(epoch: int, keep: bool) -> tuple[dict[str, Any], dict[str, Any], float]:
        """Full eval-mode scoring of Train / Valid / Test; RNG must not move."""
        before = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            train_rep = C.evaluate_split(
                model, task, device, "train", per_task=keep, keep_logits=keep,
                reason=f"epoch_{epoch}_full_train",
            )
            valid_rep = C.evaluate_split(
                model, task, device, "valid", per_task=True, keep_logits=keep,
                reason=f"epoch_{epoch}_full_valid",
            )
            test_rep = C.evaluate_split(
                model, task, device, "test", y=test_labels, per_task=True, keep_logits=keep,
                reason=f"epoch_{epoch}_full_test",
            )
        seconds = float(time.perf_counter() - t0)
        after = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("evaluation moved the training RNG state")
        model.train()
        return {"train": train_rep, "valid": valid_rep, "test": test_rep}, {"rng_unchanged": True}, seconds

    def _record_per_task(epoch: int, reports: Mapping[str, Any]) -> None:
        for split in ("train", "valid", "test"):
            rep = reports[split]
            if rep.get("per_task_ap") is None:
                continue
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

    # ---- epoch-0 initialisation diagnostic (no training step, RNG untouched)
    if resume is None:
        before = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            rep0_valid = C.evaluate_split(model, task, device, "valid", reason="epoch_0_initialisation")
            rep0_test = C.evaluate_split(model, task, device, "test", y=test_labels, reason="epoch_0_initialisation")
        after = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("epoch-0 diagnostic moved the RNG state")
        epoch0_usage = M.usage_metrics(model, diag) if deep else None
        epoch0_report = {
            "epoch": 0,
            "valid_ap": float(rep0_valid["ap"]),
            "valid_bce": float(rep0_valid["bce"]),
            "test_ap": float(rep0_test["ap"]),
            "test_bce": float(rep0_test["bce"]),
            "seconds": float(time.perf_counter() - t0),
            "usage": epoch0_usage,
        }
        if deep and C.state_fingerprint(C.state_subset(C.state_cpu(model), new=True)) != model_audit["new_state_sha256"]:
            raise RuntimeError("new-module state changed before the first training step")
        update_peak()
        C.log(
            f"[{arm} seed{seed}] epoch0 validAP {epoch0_report['valid_ap']:.9f} "
            f"testAP {epoch0_report['test_ap']:.9f} (W2 seed0 reference "
            f"{C.W2_SEED0['epoch0_valid_ap']:.9f})"
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
        grad_acc: dict[str, list[float]] = {name: [] for name in GRAD_TENSORS}
        new_total_epoch: list[float] = []
        w2_total_epoch: list[float] = []
        clip_events_epoch = 0
        aborted = False
        for rows in batches:
            if watchdog.stop.is_set():
                aborted = True
                break
            t0 = time.perf_counter()
            batch = stager.batch(C.PC.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = train_step(model, optimizer, batch, y, global_step=global_steps_run, clip=clip, audit_new=deep)
            global_steps_run += 1
            losses.append(out["loss"])
            grad_norms_total.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            if deep:
                new_total_epoch.append(float(out["grad_norm_new_total"]))
                w2_total_epoch.append(float(out["grad_norm_w2_total"]))
                for name in GRAD_TENSORS:
                    value = out["new_grads"].get(name)
                    if value is not None:
                        grad_acc[name].append(float(value))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{arm} seed{seed}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        is_final = epoch == int(epochs)
        reports, eval_audit, eval_seconds = _eval_all(epoch, keep=is_final)
        eval_completed.append(int(epoch))
        valid_ap = float(reports["valid"]["ap"])
        strict = best_epoch is None or valid_ap > float(best_valid_ap)
        if strict:
            best_epoch = int(epoch)
            best_valid_ap = valid_ap
            best_epoch_reports = {
                "epoch": int(epoch),
                "valid": {k: reports["valid"][k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored")},
                "test": {k: reports["test"][k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored")},
                "train": {k: reports["train"][k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored")},
            }
            payload = checkpoint_payload(
                arm=arm, seed=seed, horizon=epochs, epoch=epoch, global_step=global_steps_run,
                model=model, optimizer=optimizer, rng_state=C.capture_rng_state(),
                plan=C.plan_meta(plan, epoch, order_seed), model_audit=model_audit,
                valid_ap=best_valid_ap, kind="best_valid_ap",
                extra={"optimizer_state_audit": opt_audit, "valid_report_summary": best_epoch_reports["valid"]},
            )
            best_checkpoint_sha = _atomic_save(payload, best_path)
            C.write_json(action_path, best_epoch_reports)
        _record_per_task(epoch, reports)
        usage = M.usage_metrics(model, diag) if deep else {}
        if deep:
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
            "train_bce_step_mean": float(np.mean(losses)) if losses else float("nan"),
            "train_bce": float(reports["train"]["bce"]),
            "train_ap": float(reports["train"]["ap"]),
            "valid_bce": float(reports["valid"]["bce"]),
            "valid_ap": valid_ap,
            "test_bce": float(reports["test"]["bce"]),
            "test_ap": float(reports["test"]["ap"]),
            "n_tasks_scored_train": int(reports["train"]["n_tasks_scored"]),
            "n_tasks_scored_valid": int(reports["valid"]["n_tasks_scored"]),
            "n_tasks_scored_test": int(reports["test"]["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "train_eval_seconds": float(reports["train"]["seconds"]),
            "valid_seconds": float(reports["valid"]["seconds"]),
            "test_seconds": float(reports["test"]["seconds"]),
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
            "best_valid_ap_so_far": None if best_epoch is None else float(best_valid_ap),
            "is_best": bool(strict),
            "R1": usage.get("R1"),
            "R2": usage.get("R2"),
            "Rtotal": usage.get("Rtotal"),
            "rms_h1_minus_h0": usage.get("rms_h1_minus_h0"),
            "rms_h2_minus_h1": usage.get("rms_h2_minus_h1"),
            "fc2_norm_blocks0": (usage.get("fc2_weight_norm") or {}).get("blocks.0"),
            "fc2_norm_blocks1": (usage.get("fc2_weight_norm") or {}).get("blocks.1"),
            "bond_proj_weight_norm": usage.get("bond_proj_weight_norm"),
            "cos_h0_h1": usage.get("cos_h0_h1"),
            "n_directed_edges": usage.get("n_directed_edges"),
            "n_pairs": usage.get("n_pairs"),
            "mean_degree": usage.get("mean_degree"),
            "max_degree": usage.get("max_degree"),
            "grad_norm_bond_proj": _mean(grad_acc.get("bond_proj.weight", [])),
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
        if deep:
            C.write_json(
                usage_path,
                {
                    "schema": "dscr-structural-usage-v1",
                    "round_id": C.ROUND_ID,
                    "arm": arm,
                    "diagnostic_rows": int(C.DIAG_VALID_ROWS),
                    "diagnostic_batch": "first official Validation rows, eval mode",
                    "history": usage_history,
                },
            )
        C.log(
            f"[{arm} seed{seed}] ep {epoch}/{epochs} bce {row['train_bce_step_mean']:.6f} "
            f"trainAP {row['train_ap']:.6f} validAP {valid_ap:.6f} testAP {row['test_ap']:.6f} "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.1f}s eval {eval_seconds:.1f}s "
            f"{'*best*' if strict else ''}"
        )

        if is_final:
            final_payload = checkpoint_payload(
                arm=arm, seed=seed, horizon=epochs, epoch=epoch, global_step=global_steps_run,
                model=model, optimizer=optimizer, rng_state=C.capture_rng_state(),
                plan=C.plan_meta(plan, epoch, order_seed), model_audit=model_audit,
                valid_ap=valid_ap, kind=f"epoch{epochs}_horizon_endpoint",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "status_at_save": "COMPLETED",
                    "train_per_task_ap": reports["train"].get("per_task_ap"),
                    "valid_per_task_ap": reports["valid"].get("per_task_ap"),
                    "test_per_task_ap": reports["test"].get("per_task_ap"),
                },
            )
            final_checkpoint_sha = _atomic_save(final_payload, final_path)
            np.save(out_dir / "valid_logits_final.npy", reports["valid"]["logits"].astype(np.float32))
            np.save(out_dir / "test_logits_final.npy", reports["test"]["logits"].astype(np.float32))
        payload_last = checkpoint_payload(
            arm=arm, seed=seed, horizon=epochs, epoch=epoch, global_step=global_steps_run,
            model=model, optimizer=optimizer, rng_state=C.capture_rng_state(),
            plan=C.plan_meta(plan, epoch, order_seed), model_audit=model_audit,
            valid_ap=valid_ap, kind="last_epoch_resumable",
            extra={"status_at_save": "RUNNING_OR_COMPLETED", "optimizer_state_audit": opt_audit},
        )
        last_checkpoint_sha = _atomic_save(payload_last, last_path)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    expected_epochs = list(range(int(start_epoch), int(epochs) + 1))
    completed = (
        epochs_run >= int(epochs)
        and sorted(eval_completed) == expected_epochs
        and bool(final_checkpoint_sha)
    )
    if not completed and resume is not None and int(start_epoch) > int(epochs):
        # repair resume: the horizon was already finished in an earlier attempt
        completed = (
            epochs_run >= int(epochs)
            and sorted(eval_completed) == list(range(1, int(epochs) + 1))
            and final_path.is_file()
        )
        if completed and not final_checkpoint_sha:
            final_checkpoint_sha = C.sha256_file(final_path)
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"
    elif stop_reason == "RUNNING":
        stop_reason = "MISSING_EPOCH"

    # ---- frozen-data evaluation of the best checkpoint (per-task + logits)
    best_eval: dict[str, Any] | None = None
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            breps = {}
            for split in ("train", "valid", "test"):
                breps[split] = C.evaluate_split(
                    model, task, device, split, y=test_labels if split == "test" else None,
                    per_task=True, keep_logits=(split != "train"),
                    reason=f"best_checkpoint_{split}",
                )
        np.save(out_dir / "valid_logits_best.npy", breps["valid"]["logits"].astype(np.float32))
        np.save(out_dir / "test_logits_best.npy", breps["test"]["logits"].astype(np.float32))
        best_eval = {
            "epoch": int(payload["epoch"]),
            "seconds": float(time.perf_counter() - t0),
            "train": {k: breps["train"][k] for k in ("ap", "bce", "logits_sha256", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "valid": {k: breps["valid"][k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "test": {k: breps["test"][k] for k in ("ap", "bce", "logits_sha256", "probs_sha256", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "selection_metric": "valid.ap",
            "test_role": C.TEST_ROLE,
        }
        C.write_json(out_dir / "best_checkpoint_eval.json", best_eval)
        # Re-scoring the same weights differs by up to ~1.4e-5 AP on this GPU: the
        # layer-2 scatter uses atomic adds (index_add_), whose summation order is not
        # fixed, and a single near-tie swap moves one task's AP.  The gate below is a
        # corruption guard, not a bit-exactness claim.
        best_eval["valid_ap_minus_recorded"] = float(best_eval["valid"]["ap"]) - float(best_valid_ap)
        if abs(best_eval["valid_ap_minus_recorded"]) > 1e-4:
            raise RuntimeError(
                f"best-checkpoint Valid AP {best_eval['valid']['ap']} != recorded {best_valid_ap}"
            )

    rng_end_hashes = C.rng_state_hashes(C.capture_rng_state())
    curve_sha = _write_csv(curve_path, CURVE_FIELDS, curve_rows)
    per_task_sha = _write_csv(per_task_path, PER_TASK_FIELDS, per_task_rows)
    total_seconds = float(time.perf_counter() - t_start)
    final_row = curve_rows[-1] if curve_rows else {}
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "arm_role": C.ARMS[arm]["role"],
        "access_class": C.ARMS[arm]["class"],
        "seed": int(seed),
        "order_seed": int(order_seed),
        "horizon": int(epochs),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_TASKS),
        "parameters": int(model_audit["parameters"]["total"]),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "eval_epochs_completed": sorted(eval_completed),
        "epoch0_diagnostic": epoch0_report,
        "task_counts": counts,
        "task_strata_valid": {str(t): strata[t] for t in range(C.N_TASKS)},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "batch_plan": C.plan_meta(plan, max(int(epochs_run), 0), order_seed),
        "plan_identity": plan_audit,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_ap": None if best_epoch is None else float(best_valid_ap),
        "horizon_endpoint": {
            "epoch": int(epochs_run),
            "valid_ap": final_row.get("valid_ap"),
            "test_ap": final_row.get("test_ap"),
            "train_ap": final_row.get("train_ap"),
            "valid_bce": final_row.get("valid_bce"),
            "test_bce": final_row.get("test_bce"),
        },
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
        "structural_usage": C.track_rel(usage_path) if usage_path.is_file() else None,
        "structural_usage_sha256": C.sha256_file(usage_path) if usage_path.is_file() else None,
        "valid_logits_best": C.track_rel(out_dir / "valid_logits_best.npy") if (out_dir / "valid_logits_best.npy").is_file() else None,
        "test_logits_best": C.track_rel(out_dir / "test_logits_best.npy") if (out_dir / "test_logits_best.npy").is_file() else None,
        "valid_logits_final": C.track_rel(out_dir / "valid_logits_final.npy") if (out_dir / "valid_logits_final.npy").is_file() else None,
        "test_logits_final": C.track_rel(out_dir / "test_logits_final.npy") if (out_dir / "test_logits_final.npy").is_file() else None,
        "recipe": C.recipe(arm, seed, epochs),
        "model_dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "gamma": float(gamma) if gamma is not None else float(C.GAMMA),
        "tf32": {
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_seconds_prior_attempts": float(prior_gpu_seconds),
        "gpu_hours": (total_seconds + float(prior_gpu_seconds)) / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "eval_seconds": float(sum(float(r.get("epoch_seconds") or 0.0) - float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "seed_state": seed_info,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "rng_epoch_hashes": rng_epoch_hashes,
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "resume_repair": resume_repair,
        "epoch0_diagnostic_note": None if epoch0_report or resume is None else "epoch 0 ran in a previous attempt; see analysis/checks.json A2.4 and the run log",
        "test_role": C.TEST_ROLE,
        "test_used_for_selection": False,
        "test_scored_every_epoch": True,
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done arm={arm} seed={seed} status={status} epochs={epochs_run}/{epochs} "
        f"best_epoch={best_epoch} best_valid={best_valid_ap} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(C.ARM_ORDER))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=C.DEFAULT_HORIZON)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--prior-gpu-seconds", type=float, default=0.0,
                        help="GPU seconds already spent on this unit by an earlier attempt (ledger continuity)")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out_dir / "run.json", {"schema": C.SCHEMA_RUN, "round_id": C.ROUND_ID, "arm": args.arm,
                                            "seed": int(args.seed), "status": "INCOMPLETE",
                                            "stop_reason": "CUDA_UNAVAILABLE", "finished_utc": C.utcnow()})
        return 2
    try:
        task = C.SD.load_task()
        test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        run = train(
            task,
            arm=str(args.arm),
            seed=int(args.seed),
            out_dir=out_dir,
            device=device,
            epochs=int(args.epochs),
            resume=resume,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            test_labels=np.asarray(test_info["labels"]),
            gamma=args.gamma,
            prior_gpu_seconds=float(args.prior_gpu_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "arm": str(args.arm),
                "seed": int(args.seed),
                "horizon": int(args.epochs),
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED arm={args.arm}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
