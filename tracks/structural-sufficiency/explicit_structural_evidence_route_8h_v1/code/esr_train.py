"""Training drivers for the small datasets and PCBA.

Small datasets (Lipo / ESOL / Tox21): frozen R3 recipe -- AdamW, per-step cosine
from 3e-4 to 1e-5 with no warmup, 360 epochs, batch 128, dropout 0.10, clip 5.0,
FP32, seed 0, one frozen batch order shared by every arm of a dataset.

PCBA: the frozen 40-epoch recipe (1-epoch linear warmup 3e-5 -> 3e-4, then
per-step cosine to 1e-5 over the *40-epoch* horizon, never compressed), the
official Train order plan of seed 0, batch 128.  The formal screening horizon is
the common 25-epoch prefix; a run can be resumed to 40 epochs from the exact
``checkpoint_last`` state (model, optimizer, scheduler, RNG, plan position).

Both drivers persist:
``run.json``, ``curve.csv``, ``checkpoint_best.pt``, ``checkpoint_last.pt`` and
``checkpoint_final.pt`` (small) / ``checkpoint_epoch{N}.pt`` (PCBA).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import esr_common as C
import esr_data as D
import esr_model as M

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class BestTracker:
    def __init__(self, view: D.View):
        self.view = view
        self.best_epoch: int | None = None
        self.best_primary: float | None = None
        self.best_secondary: float | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        primary = float(self.view.primary(metrics))
        secondary = float(metrics.get("masked_bce", float("nan")))
        if self.best_epoch is None:
            better = True
        elif self.view.task == "multitask" and primary == self.best_primary:
            better = secondary < float(self.best_secondary)
        else:
            better = self.view.better(primary, self.best_primary)
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
        self.best_secondary = secondary
        self.best_metrics = dict(metrics)
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        assert self._state is not None
        return self._state

    def to_device(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in self.state.items()}


def _curve_row(epoch: int, lr: float, train_stats: Mapping[str, Any], metrics: Mapping[str, Any], primary_key: str, seconds: Mapping[str, float], improved: bool) -> dict[str, Any]:
    row = {"epoch": int(epoch), "lr": float(lr), "is_best": bool(improved)}
    row.update({f"train_{k}": float(v) for k, v in train_stats.items()})
    row["valid_primary"] = float(metrics[primary_key])
    row.update(seconds)
    for key in ("rmse", "mae", "r2", "macro_roc_auc", "macro_pr_auc", "masked_bce", "ap", "valid_bce"):
        if key in metrics and metrics[key] is not None:
            row[f"valid_{key}"] = float(metrics[key])
    return row


def _write_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _train_stats(view: D.View, sums: Mapping[str, float], n: int) -> dict[str, float]:
    out = {"n_observations": float(n)}
    if view.task == "regression":
        mse = sums["sq"] / max(int(sums["n"]), 1)
        out["mse_standardized"] = mse
        out["rmse_standardized"] = math.sqrt(max(mse, 0.0))
        out["mae_standardized"] = sums["abs"] / max(int(sums["n"]), 1)
    else:
        out["masked_bce"] = sums["bce"] / max(int(sums["n"]), 1)
    return out


def _make_optimizer(model: torch.nn.Module, recipe: Mapping[str, Any], *, start_lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(start_lr),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )


def _batch_loss(view: D.View, model, batch, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pred = model(batch)
    if view.task == "regression":
        pred = pred.reshape(-1)
        target = y.reshape(-1)
        loss = F.mse_loss(pred, target)
        err = pred - target
        stats = {
            "abs": float(err.detach().abs().sum()),
            "sq": float((err.detach() * err.detach()).sum()),
            "n": float(target.numel()),
        }
        return loss, stats
    if view.task == "multitask":
        mask = torch.isfinite(y)
        if int(mask.sum()) == 0:
            return None, {"n": 0.0}
        per = F.binary_cross_entropy_with_logits(pred, torch.nan_to_num(y), reduction="none")
        loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, {"bce": float(loss.detach()) * int(mask.sum()), "n": float(mask.sum())}
    raise RuntimeError(view.task)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{PROCESS_T0:.0f}")
    torch.save(payload, tmp)
    tmp.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# small datasets
# ---------------------------------------------------------------------------


def train_small(
    *,
    dataset: str,
    arm: str,
    seed: int = 0,
    out_dir: Path,
    device: torch.device,
    epochs: int | None = None,
    smoke: bool = False,
    eval_every: int = 1,
    hidden_width: int | None = None,
    max_dist: int = 2,
    relation_mode: str = "real",
    tag: str = "",
    resume: Path | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    recipe = dict(C.RECIPE_SMALL)
    epochs = int(epochs if epochs is not None else recipe["epochs"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = tag or f"{dataset.lower()}-{arm.lower()}-s{seed}"
    started = time.time()
    C.register_run(
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": arm,
            "seed": int(seed),
            "stage": "smoke" if smoke else "formal",
            "requested_epochs": int(epochs),
            "status": "running",
            "gpu_id": str(device),
            "started_utc": C.utcnow(),
            "checkpoint_path": C.track_rel(out_dir),
        }
    )
    view = D.load_view(dataset, include_test=False, max_dist=max_dist)
    view = D.view_with_pairs(view, relation_mode=relation_mode, max_dist=max_dist)
    model, model_info = M.build_model(
        arm=arm, view=view, seed=int(seed), device=device, hidden_width=hidden_width
    )
    n_train = int(view.rows["train"].size)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    opt = _make_optimizer(model, recipe, start_lr=float(recipe["peak_lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=float(recipe["min_lr"]))
    gen = torch.Generator().manual_seed(int(seed) + int(recipe["shuffle_offset"]))
    y_train_loss = view.loss_target("train")
    y_valid = view.y["valid"]
    tracker = BestTracker(view)
    start_epoch = 1
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("arm") != arm or payload.get("dataset") != dataset:
            raise RuntimeError("resume checkpoint belongs to another run")
        model.load_state_dict(payload["state_dict"])
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng"])
        gen.set_state(payload["shuffle_generator"])
        start_epoch = int(payload["epoch"]) + 1
        tracker.best_epoch = payload.get("best_epoch")
        tracker.best_primary = payload.get("best_primary")
        tracker.best_secondary = payload.get("best_secondary")
        tracker.best_metrics = payload.get("best_metrics")
        tracker._state = payload.get("best_state")
        C.log(f"{run_id}: resumed at epoch {start_epoch}")
    curve: list[dict[str, Any]] = []
    clip_events = 0
    gpu_peak = 0
    skipped = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    primary_key = "ap" if dataset == C.PCBA else (
        "rmse" if view.task == "regression" else "macro_roc_auc"
    )
    ran_epochs = 0
    for epoch in range(start_epoch, epochs + 1):
        ran_epochs += 1
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=gen).tolist()
        sums = {"abs": 0.0, "sq": 0.0, "bce": 0.0, "n": 0}
        for step in range(0, len(perm), batch_size):
            rows = perm[step : step + batch_size]
            cache_rows = [int(view.rows["train"][i]) for i in rows]
            batch = view.to_device(view.collate("train", cache_rows), device)
            y = y_train_loss[rows].to(device)
            loss, stats = _batch_loss(view, model, batch, y)
            if loss is None:
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            opt.step()
            sched.step()
            for k in ("abs", "sq", "bce"):
                sums[k] += float(stats.get(k, 0.0))
            sums["n"] += int(stats.get("n", 0))
        train_seconds = time.perf_counter() - t0
        t_eval = time.perf_counter()
        metrics, _logits = D.evaluate(model, view, "valid", device, batch_size=256)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, metrics)
        if improved:
            tracker.capture(model)
        if epoch % int(eval_every) == 0 or epoch == epochs:
            curve.append(
                _curve_row(
                    epoch,
                    float(opt.param_groups[0]["lr"]),
                    _train_stats(view, sums, int(sums["n"])),
                    metrics,
                    primary_key,
                    {"train_seconds": train_seconds, "valid_seconds": eval_seconds},
                    improved,
                )
            )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))
        if epoch == start_epoch or epoch % 50 == 0 or epoch == epochs:
            C.log(
                f"{run_id} epoch {epoch}/{epochs} {primary_key}={tracker.best_primary:.6f} "
                f"(best {tracker.best_epoch}) train {train_seconds:.1f}s"
            )
        if max_epochs is not None and ran_epochs >= int(max_epochs):
            C.log(f"{run_id}: stopping after {ran_epochs} epoch(s) (max_epochs={max_epochs})")
            break
    wall = time.perf_counter() - started
    # last-epoch metrics are recomputed from the final weights before restoring best
    metrics_final, _ = D.evaluate(model, view, "valid", device, batch_size=256)
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(tracker.to_device(device))
    metrics_best, logits_best = D.evaluate(model, view, "valid", device, batch_size=256)
    recheck = abs(float(view.primary(metrics_best)) - float(tracker.best_primary))
    if recheck > 1e-5:
        raise RuntimeError(f"best-checkpoint re-evaluation differs by {recheck}")
    train_metrics, _ = D.evaluate(model, view, "train", device, batch_size=512)
    payload_common = {
        "schema": C.SCHEMA_CHECKPOINT,
        "run_id": run_id,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "model_info": model_info,
        "best_epoch": int(tracker.best_epoch or 0),
        "epochs": int(epochs),
    }
    _save_checkpoint(out_dir / "checkpoint_best.pt", {**payload_common, "state_dict": tracker.state, "kind": "best"})
    _save_checkpoint(
        out_dir / "checkpoint_final.pt",
        {**payload_common, "state_dict": final_state, "kind": "final_epoch"},
    )
    _save_checkpoint(
        out_dir / "checkpoint_last.pt",
        {
            **payload_common,
            "state_dict": final_state,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": _rng_state(),
            "shuffle_generator": gen.get_state(),
            "epoch": int(curve[-1]["epoch"]) if curve else start_epoch,
            "best_epoch": tracker.best_epoch,
            "best_primary": tracker.best_primary,
            "best_secondary": tracker.best_secondary,
            "best_metrics": tracker.best_metrics,
            "best_state": tracker._state,
            "kind": "resumable",
        },
    )
    np.save(out_dir / "valid_logits_best.npy", logits_best)
    _write_curve(out_dir / "curve.csv", curve)
    run = {
        **payload_common,
        "status": "completed",
        "recipe": recipe,
        "view_audit": view.audit(),
        "n_train": n_train,
        "steps_per_epoch": steps_per_epoch,
        "total_updates": int(sched.last_epoch),
        "parameters": int(model_info["parameters"]),
        "grad_clip_events": int(clip_events),
        "skipped_batches_no_label": int(skipped),
        "gpu_peak_allocated_bytes": int(gpu_peak),
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0) if device.type == "cuda" else 0.0,
        "best_valid_metrics": metrics_best,
        "final_epoch_valid_metrics": metrics_final,
        "train_metrics_at_best": train_metrics,
        "late_horizon": [r for r in curve if r["epoch"] > max(1, epochs - 20)],
        "environment": C.environment(),
        "git": C.git_commit(),
        "budget": C.budget_state(),
        "finished_utc": C.utcnow(),
        "smoke": bool(smoke),
        "relation_mode": relation_mode,
        "max_dist": int(max_dist),
    }
    C.write_json(out_dir / "run.json", run)
    C.register_run(
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": arm,
            "seed": int(seed),
            "stage": "smoke" if smoke else "formal",
            "requested_epochs": int(epochs),
            "completed_epochs": int(epochs),
            "expected_gpu_hours": float(wall / 3600.0),
            "actual_gpu_hours": float(wall / 3600.0) if device.type == "cuda" else 0.0,
            "gpu_id": str(device),
            "status": "completed",
            "checkpoint_path": C.track_rel(out_dir),
            "best_valid_primary": float(tracker.best_primary or float("nan")),
            "finished_utc": C.utcnow(),
        }
    )
    return run


# ---------------------------------------------------------------------------
# PCBA
# ---------------------------------------------------------------------------


def pcba_plan(epochs: int = 40) -> list[dict[str, Any]]:
    task = D.SGD.load_task()
    plan = D.SGD.make_plan40(task)
    if len(plan) < epochs:
        raise RuntimeError("order plan shorter than requested epochs")
    return plan


def train_pcba(
    *,
    arm: str,
    seed: int = 0,
    out_dir: Path,
    device: torch.device,
    epochs: int = 25,
    resume: Path | None = None,
    smoke: bool = False,
    hidden_width: int | None = None,
    relation_mode: str = "real",
    max_dist: int = 2,
    tag: str = "",
    train_eval_epochs: Sequence[int] = (1, 10, 25, 40),
    max_steps: int | None = None,
) -> dict[str, Any]:
    dataset = C.PCBA
    meta = C.DATASETS[dataset]
    recipe = dict(C.RECIPE_PCBA)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = tag or f"pcba-{arm.lower()}-s{seed}-e{epochs}"
    started = time.time()
    C.register_run(
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": arm,
            "seed": int(seed),
            "stage": "smoke" if smoke else "formal",
            "requested_epochs": int(epochs),
            "status": "running",
            "gpu_id": str(device),
            "started_utc": C.utcnow(),
            "checkpoint_path": C.track_rel(out_dir),
        }
    )
    view = D.load_view(dataset, include_test=False, max_dist=max_dist)
    view = D.view_with_pairs(view, relation_mode=relation_mode, max_dist=max_dist)
    model, model_info = M.build_model(arm=arm, view=view, seed=int(seed), device=device, hidden_width=hidden_width)
    n_train = int(view.rows["train"].size)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    plan = pcba_plan(40)
    start_epoch = 1
    global_step = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("arm") != arm:
            raise RuntimeError(f"resume checkpoint is for {payload.get('arm')}, not {arm}")
        model.load_state_dict(payload["state_dict"])
        opt = _make_optimizer(model, recipe, start_lr=float(recipe["warmup_start_lr"]))
        opt.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        _restore_rng(payload["rng"])
        C.log(f"{run_id}: resumed at epoch {start_epoch} (global_step {global_step})")
    else:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        opt = _make_optimizer(model, recipe, start_lr=float(C.scheduled_lr_pcba(0, phase=3, kind="all")))
    y_train = view.loss_target("train")
    y_valid = view.y["valid"]
    # PCBA addresses its one global cache by official row id; map to split order
    pos_of_row = np.full(int(view.caches["train"]["n_molecules"]), -1, dtype=np.int64)
    pos_of_row[view.rows["train"]] = np.arange(int(view.rows["train"].size), dtype=np.int64)
    tracker = BestTracker(view)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_ms: list[float] = []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sums = {"bce": 0.0, "n": 0}
        batches = plan[epoch - 1]["batches"]
        for step_index, rows in enumerate(batches):
            if max_steps is not None and (epoch - 1) * steps_per_epoch + step_index >= int(max_steps):
                break
            lr = float(C.scheduled_lr_pcba(global_step, phase=3, kind="all"))
            for group in opt.param_groups:
                group["lr"] = lr
            tstep = time.perf_counter()
            batch = view.to_device(view.collate("train", [int(r) for r in rows]), device)
            pos = pos_of_row[np.asarray(rows, dtype=np.int64)]
            if np.any(pos < 0):
                raise RuntimeError("order-plan row is not in the official Train split")
            y = y_train[torch.as_tensor(pos, dtype=torch.long)].to(device)
            loss, stats = _batch_loss(view, model, batch, y)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step_index}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
            opt.step()
            global_step += 1
            sums["bce"] += float(stats["bce"])
            sums["n"] += int(stats["n"])
            if epoch >= 2 and step_index >= 10:
                step_ms.append((time.perf_counter() - tstep) * 1000.0)
        train_seconds = time.perf_counter() - t0
        t_eval = time.perf_counter()
        metrics, _ = D.evaluate(model, view, "valid", device, batch_size=256)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, metrics)
        if improved:
            tracker.capture(model)
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "lr": float(opt.param_groups[0]["lr"]),
            "train_seconds": float(train_seconds),
            "valid_seconds": float(eval_seconds),
            "train_masked_bce": float(sums["bce"] / max(sums["n"], 1)),
            "valid_ap": float(metrics["ap"]),
            "valid_masked_bce": float(metrics["masked_bce"]),
            "valid_macro_roc_auc": float(metrics.get("macro_roc_auc", float("nan")) or float("nan")),
            "grad_clip_events": int(clip_events),
            "is_best": bool(improved),
        }
        if epoch in set(int(x) for x in train_eval_epochs) and not smoke:
            t_tr = time.perf_counter()
            train_metrics, _ = D.evaluate(model, view, "train", device, batch_size=512)
            row["train_full_ap"] = float(train_metrics["ap"])
            row["train_full_bce"] = float(train_metrics["masked_bce"])
            row["train_eval_seconds"] = float(time.perf_counter() - t_tr)
        curve.append(row)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))
        C.log(
            f"{run_id} epoch {epoch}/{epochs} step {global_step} valid_ap={metrics['ap']:.6f} "
            f"best={tracker.best_primary:.6f}@{tracker.best_epoch} train {train_seconds:.1f}s"
        )
        checkpoint = {
            "schema": C.SCHEMA_CHECKPOINT,
            "run_id": run_id,
            "dataset": dataset,
            "arm": arm,
            "seed": int(seed),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_info": model_info,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "rng": _rng_state(),
            "plan": {"global_plan_sha256": plan[0]["global_plan_sha256"], "consumed_epochs": int(epoch)},
        }
        _save_checkpoint(out_dir / "checkpoint_last.pt", checkpoint)
        if epoch in (25, 40) or epoch == epochs:
            _save_checkpoint(out_dir / f"checkpoint_epoch{epoch}.pt", checkpoint)
        if improved:
            _save_checkpoint(
                out_dir / "checkpoint_best.pt",
                {**checkpoint, "kind": "best", "best_epoch": int(tracker.best_epoch or 0)},
            )
        _write_curve(out_dir / "curve.csv", curve)
        C.write_json(
            out_dir / "progress.json",
            {
                "run_id": run_id,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_valid_ap": float(tracker.best_primary or float("nan")),
                "best_epoch": int(tracker.best_epoch or 0),
                "gpu_hours_so_far": float((time.time() - started) / 3600.0),
                "budget": C.budget_state(),
            },
        )
        if max_steps is not None and global_step >= int(max_steps):
            C.log(f"{run_id}: stopping after max_steps={max_steps}")
            break
    wall = time.perf_counter() - started
    metrics_final, _ = D.evaluate(model, view, "valid", device, batch_size=256)
    final_epoch = int(curve[-1]["epoch"]) if curve else start_epoch
    steps_ms = np.asarray(step_ms, dtype=np.float64) if step_ms else np.zeros(1)
    run = {
        "schema": C.SCHEMA_RUN,
        "run_id": run_id,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "recipe": recipe,
        "view_audit": view.audit(),
        "n_train": n_train,
        "steps_per_epoch": steps_per_epoch,
        "epochs_requested": int(epochs),
        "completed_epochs": int(final_epoch),
        "global_step": int(global_step),
        "parameters": int(model_info["parameters"]),
        "model_info": model_info,
        "best_epoch": int(tracker.best_epoch or 0),
        "best_valid_ap": float(tracker.best_primary or float("nan")),
        "best_valid_metrics": tracker.best_metrics,
        "prefix_final_valid_ap": float(metrics_final["ap"]),
        "final_epoch_valid_metrics": metrics_final,
        "curve": curve,
        "step_ms": {"p50": float(np.percentile(steps_ms, 50)), "p90": float(np.percentile(steps_ms, 90))},
        "gpu_peak_allocated_bytes": int(gpu_peak),
        "grad_clip_events": int(clip_events),
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0) if device.type == "cuda" else 0.0,
        "order_plan": {
            "global_plan_sha256": plan[0]["global_plan_sha256"],
            "consumed_epochs": int(final_epoch),
            "first_epoch_rows_sha256": plan[0]["row_order_sha256"],
        },
        "relation_mode": relation_mode,
        "max_dist": int(max_dist),
        "smoke": bool(smoke),
        "environment": C.environment(),
        "git": C.git_commit(),
        "budget": C.budget_state(),
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", run)
    C.register_run(
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": arm,
            "seed": int(seed),
            "stage": "smoke" if smoke else "formal",
            "requested_epochs": int(epochs),
            "completed_epochs": int(final_epoch),
            "expected_gpu_hours": float(wall / 3600.0),
            "actual_gpu_hours": float(wall / 3600.0) if device.type == "cuda" else 0.0,
            "gpu_id": str(device),
            "status": "completed",
            "checkpoint_path": C.track_rel(out_dir),
            "best_valid_primary": float(tracker.best_primary or float("nan")),
            "finished_utc": C.utcnow(),
        }
    )
    return run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESR training driver")
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--arm", required=True, choices=list(C.ARMS) + list(C.CONDITIONAL_ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--hidden-width", type=int, default=None)
    ap.add_argument("--max-dist", type=int, default=2)
    ap.add_argument("--relation-mode", default="real", choices=["real", "blind"])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if args.dataset == C.PCBA:
        run = train_pcba(
            arm=args.arm,
            seed=args.seed,
            out_dir=Path(args.out_dir),
            device=device,
            epochs=int(args.epochs if args.epochs is not None else C.DATASETS[C.PCBA]["epochs"]),
            resume=Path(args.resume) if args.resume else None,
            smoke=args.smoke,
            hidden_width=args.hidden_width,
            relation_mode=args.relation_mode,
            max_dist=args.max_dist,
            tag=args.tag,
            max_steps=args.max_steps,
        )
    else:
        run = train_small(
            dataset=args.dataset,
            arm=args.arm,
            seed=args.seed,
            out_dir=Path(args.out_dir),
            device=device,
            epochs=args.epochs,
            smoke=args.smoke,
            eval_every=args.eval_every,
            hidden_width=args.hidden_width,
            max_dist=args.max_dist,
            relation_mode=args.relation_mode,
            tag=args.tag,
            resume=Path(args.resume) if args.resume else None,
            max_epochs=args.max_epochs,
        )
    C.log(f"done {run['run_id']}: {json.dumps({k: run.get(k) for k in ('best_epoch', 'best_valid_ap', 'parameters')}, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
