"""Frozen supervised recipe for R and N.

The recipe is copied from the historical ``lcrc_lipo_screen_v1`` A run:
Fit2352 only, AdamW 1e-3 -> 1e-5 per-step cosine, batch 128, 360 epochs,
6840 updates, clip 5.0, soup = mean of the last 5 epoch-end state dicts.
No early stopping and no Monitor access exist in this module.
"""

from __future__ import annotations

import csv
import hashlib
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import access_guard
import common as C
import data as D
import fit_labels
import model as M


@dataclass
class TrainConfig:
    arm: str
    seed: int
    device: str = "cpu"
    smoke: bool = False
    fit_limit: int | None = None
    epochs: int = C.EPOCHS
    batch_size: int = C.BATCH_SIZE
    learning_rate: float = C.LR
    min_lr: float = C.LR_MIN
    weight_decay: float = C.WEIGHT_DECAY
    grad_clip: float = C.GRAD_CLIP
    dropout: float = C.DROPOUT
    soup_last: int = C.SOUP_LAST
    out_root: Path = field(default_factory=lambda: C.RUNS)
    threads: int = 4

    def validate(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}, got {self.arm!r}")
        if self.seed not in C.SEEDS:
            raise ValueError(f"seed must be one of {C.SEEDS}, got {self.seed!r}")
        if self.batch_size != C.BATCH_SIZE:
            raise ValueError("the frozen recipe uses batch_size=128")
        if self.learning_rate != C.LR or self.min_lr != C.LR_MIN:
            raise ValueError("the frozen recipe uses lr 1e-3 -> 1e-5")
        if self.weight_decay != C.WEIGHT_DECAY or self.grad_clip != C.GRAD_CLIP:
            raise ValueError("the frozen recipe uses weight_decay 1e-5 and clip 5.0")
        if self.dropout != 0.0:
            raise ValueError("the frozen recipe uses dropout 0")
        if self.soup_last != C.SOUP_LAST:
            raise ValueError("the frozen recipe averages exactly the last 5 epochs")
        if self.smoke:
            if self.device != "cpu":
                raise PermissionError("smoke is CPU-only")
            if self.fit_limit is None or not (32 <= int(self.fit_limit) <= 512):
                raise ValueError("smoke fit_limit must be in [32, 512]")
            if not (1 <= int(self.epochs) <= 40):
                raise ValueError("smoke epochs must be in [1, 40]")
        else:
            if self.fit_limit is not None:
                raise ValueError("formal training must use all 2352 Fit rows")
            if self.epochs != C.EPOCHS:
                raise ValueError("formal training must use exactly 360 epochs")
            if self.device not in ("cuda", "cpu"):
                raise ValueError("device must be cuda or cpu")

    def run_id(self) -> str:
        if self.smoke:
            return f"smoke_{self.arm}_s{self.seed}_n{self.fit_limit}_e{self.epochs}"
        return f"{self.arm}_s{self.seed}"


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


def _write_curve_csv(path: Path, curve: list[dict[str, Any]]) -> None:
    fields = [
        "epoch", "updates", "train_mse_normalised", "train_mae_normalised",
        "lr_last", "grad_norm_max", "grad_norm_mean", "clip_events",
        "delta_abs_max", "delta_mean", "delta_std", "seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in curve:
            writer.writerow({k: row.get(k) for k in fields})


def train_one(cfg: TrainConfig) -> dict[str, Any]:
    cfg.validate()
    torch.set_num_threads(int(cfg.threads))
    access_guard.arm()

    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    fit = fit_labels.load_fit_labels(arm_guard=False)
    y_fit = fit["y"]
    target_norm = D.load_target_norm(verify=False)
    stats = fit_labels.fit_label_stats(y_fit)
    for key in ("mean", "scale", "fit_label_vector_sha256"):
        if target_norm[key] != stats[key]:
            raise PermissionError(f"saved target normalisation {key} does not match the Fit labels")

    fit_rows = [int(r) for r in caches["fit"]["rows"]]
    n = int(cfg.fit_limit) if cfg.fit_limit is not None else C.SPLIT_SIZES["fit"]
    train_positions = D.fit_training_positions(n)
    fit_cache = caches["fit"]["cache"]
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total_updates = steps_per_epoch * cfg.epochs

    device = torch.device(cfg.device)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, model_info = M.build_model(cfg.arm, spec, norm, cfg.seed, device)
    if not model_info["trunk_init_matches_historical"]:
        raise RuntimeError(
            f"trunk init hash {model_info['trunk_state_hash']} != historical {model_info['trunk_expected_hash']}"
        )
    soup_report = M.soup_safety_report(model)
    if not soup_report["all_state_entries_float"]:
        raise RuntimeError(f"parameter averaging is not safe: {soup_report['non_float_entries']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=cfg.min_lr)
    generator = torch.Generator().manual_seed(int(cfg.seed) + C.TRAIN_SHUFFLE_OFFSET)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    batch_alignment: list[dict[str, Any]] = []
    order_hash_actual = hashlib.sha256()
    grad_norm_max = 0.0
    clip_events_total = 0
    nonfinite_events = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        epoch_abs = 0.0
        epoch_sq = 0.0
        seen = 0
        lr_last = None
        epoch_plans: list[D.FitBatchPlan] = []
        epoch_grad_max = 0.0
        epoch_grad_sum = 0.0
        epoch_grad_count = 0
        epoch_clip = 0
        epoch_delta_abs = 0.0
        epoch_delta_sum = 0.0
        epoch_delta_sq = 0.0
        epoch_delta_count = 0
        for rows in D.shuffle_batches(n, cfg.batch_size, generator):
            order_hash_actual.update(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes())
            plan = D.plan_fit_batch(rows, train_positions=train_positions, cache=fit_cache, fit_global_rows=fit_rows)
            info = D.assemble_fit_batch(fit_cache, y_fit, plan)
            epoch_plans.append(plan)
            batch = D.to_device(info["batch"], device)
            y = torch.as_tensor(
                (info["y"] - target_norm["mean"]) / target_norm["scale"],
                dtype=torch.float32,
                device=device,
            )
            pred, aux = model(batch, return_aux=True)
            delta = aux["delta_bond"].detach()
            if not torch.isfinite(delta).all():
                nonfinite_events += 1
                raise RuntimeError(f"non-finite bond delta at epoch {epoch}")
            epoch_delta_abs = max(epoch_delta_abs, float(delta.abs().max()) if delta.numel() else 0.0)
            if delta.numel():
                epoch_delta_sum += float(delta.sum())
                epoch_delta_sq += float((delta * delta).sum())
                epoch_delta_count += int(delta.numel())
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            epoch_grad_max = max(epoch_grad_max, gnorm)
            epoch_grad_sum += gnorm
            epoch_grad_count += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            if gnorm > cfg.grad_clip:
                clip_events_total += 1
                epoch_clip += 1
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            error = (pred.view(-1) - y.view(-1)).detach()
            epoch_abs += float(error.abs().sum())
            epoch_sq += float((error * error).sum())
            seen += int(error.numel())
        dmean = epoch_delta_sum / max(epoch_delta_count, 1)
        dstd = math.sqrt(max(epoch_delta_sq / max(epoch_delta_count, 1) - dmean * dmean, 0.0))
        curve.append({
            "epoch": epoch,
            "updates": epoch * steps_per_epoch,
            "train_mae_normalised": epoch_abs / max(seen, 1),
            "train_mse_normalised": epoch_sq / max(seen, 1),
            "lr_last": lr_last,
            "grad_norm_max": epoch_grad_max,
            "grad_norm_mean": epoch_grad_sum / max(epoch_grad_count, 1),
            "clip_events": epoch_clip,
            "delta_abs_max": epoch_delta_abs,
            "delta_mean": dmean,
            "delta_std": dstd,
            "seconds": time.perf_counter() - t_epoch,
        })
        batch_alignment.append({"epoch": epoch, **D.fit_batch_fingerprints(epoch_plans, y_fit)})
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > cfg.soup_last:
            snapshots.pop(0)
        if epoch % 40 == 0 or epoch == 1 or epoch == cfg.epochs:
            C.log(
                f"[{cfg.run_id()}] ep={epoch:03d} mse_norm={curve[-1]['train_mse_normalised']:.6f} "
                f"grad_max={curve[-1]['grad_norm_max']:.3f} lr={lr_last:.3e} {curve[-1]['seconds']:.2f}s"
            )
        if not cfg.smoke and (time.perf_counter() - started) > C.RUN_WALL_CAP_SECONDS:
            access_guard.disarm()
            raise RuntimeError(
                f"run exceeded the {C.RUN_WALL_CAP_SECONDS}s wall cap at epoch {epoch}; "
                "logs are preserved and no formal result will be reported"
            )

    wall = time.perf_counter() - started
    independent_order_hash = D.batch_order_hash(cfg.seed, n=n, epochs=cfg.epochs)
    members = list(range(cfg.epochs - len(snapshots) + 1, cfg.epochs + 1))
    if members != list(range(max(1, cfg.epochs - cfg.soup_last + 1), cfg.epochs + 1)):
        raise RuntimeError(f"soup members {members} are not the fixed last-5 epochs")

    soup = M.average_states(snapshots)
    last_state = snapshots[-1]
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    soup_state_hash = M.state_dict_fingerprint(soup)
    last_state_hash = M.state_dict_fingerprint(last_state)
    init_state_hash = model_info["state_hash"]
    trunk_init_hash = model_info["trunk_state_hash"]

    fit_pred = D.predict_cache(model, fit_cache, device, return_aux=True)
    fit_pred_raw = D.denormalise(fit_pred["pred"], target_norm)[:n]
    fit_y_raw = y_fit[:n]
    fit_mae = float(np.abs(fit_pred_raw - fit_y_raw).mean())
    fit_rmse = float(np.sqrt(np.mean((fit_pred_raw - fit_y_raw) ** 2)))
    delta_raw = fit_pred["delta"] * float(target_norm["scale"])
    delta_fit = delta_raw[:n]

    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    last_pred = D.predict_cache(model, fit_cache, device, return_aux=False)["pred"]
    last_pred_raw = D.denormalise(last_pred, target_norm)[:n]
    last_mae = float(np.abs(last_pred_raw - fit_y_raw).mean())
    last_rmse = float(np.sqrt(np.mean((last_pred_raw - fit_y_raw) ** 2)))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})

    out_dir = Path(cfg.out_root) / cfg.run_id()
    C.ensure_dir(out_dir)
    soup_path = out_dir / "soup.pt"
    torch.save(soup, soup_path)
    curve_path = out_dir / "curve.csv"
    _write_curve_csv(curve_path, curve)

    device_info = C.gpu_runtime_info()
    peak_gpu = None
    if cfg.device == "cuda":
        peak_gpu = {
            "max_memory_allocated_gb": float(torch.cuda.max_memory_allocated()) / 1024**3,
            "max_memory_reserved_gb": float(torch.cuda.max_memory_reserved()) / 1024**3,
        }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "run_id": cfg.run_id(),
        "arm": cfg.arm,
        "seed": int(cfg.seed),
        "smoke": bool(cfg.smoke),
        "status": "COMPLETED",
        "device": str(device),
        "device_info": device_info,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "recipe": {
            "optimizer": "AdamW",
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "peak_lr": cfg.learning_rate,
            "min_lr": cfg.min_lr,
            "weight_decay": cfg.weight_decay,
            "scheduler": "per_step_cosine",
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_updates": total_updates,
            "grad_clip": cfg.grad_clip,
            "dropout": cfg.dropout,
            "loss": "mse_on_fit_standardised_target",
            "soup_last": cfg.soup_last,
            "soup_members": members,
        },
        "optimizer_config": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": total_updates,
            "eta_min": cfg.min_lr,
        },
        "data": {
            "n_train_rows": n,
            "fit_limit": cfg.fit_limit,
            "fit_rows_sha256": C.row_sha256([fit_rows[p] for p in train_positions]),
            "fit_cache_positions": [int(train_positions[0]), int(train_positions[-1])],
            "index_semantics": "shuffle_pos -> cache_pos (0..n-1) -> global_row_id (identity/provenance only)",
            "fit_label_audit": fit["audit"],
            "target_normalisation": {
                "mean": target_norm["mean"],
                "scale": target_norm["scale"],
                "source": target_norm["source"],
                "fit_label_vector_sha256": target_norm["fit_label_vector_sha256"],
            },
        },
        "model": model_info,
        "soup_safety": soup_report,
        "init_state_hash": init_state_hash,
        "trunk_init_state_hash": trunk_init_hash,
        "soup_state_hash": soup_state_hash,
        "last_epoch_state_hash": last_state_hash,
        "soup_tensor_file": C.track_rel(soup_path),
        "soup_sha256": C.sha256_file(soup_path),
        "curve_file": C.track_rel(curve_path),
        "batch_order": {
            "actual_hash": order_hash_actual.hexdigest(),
            "independent_recomputation": independent_order_hash,
            "matches": order_hash_actual.hexdigest() == independent_order_hash,
            "offset": C.TRAIN_SHUFFLE_OFFSET,
        },
        "batch_alignment": {
            "checked_in_training_loop": True,
            "scope": "every optimizer batch of every epoch of this run",
            "per_epoch": batch_alignment,
        },
        "anomalies": {
            "grad_clip_events": clip_events_total,
            "grad_norm_max": grad_norm_max,
            "nonfinite_events": nonfinite_events,
        },
        "new_branch_output": {
            "fit_delta_raw_mean": float(delta_fit.mean()),
            "fit_delta_raw_std": float(delta_fit.std()),
            "fit_delta_raw_abs_max": float(np.abs(delta_fit).max()),
            "fit_delta_raw_p05": float(np.percentile(delta_fit, 5)),
            "fit_delta_raw_p50": float(np.percentile(delta_fit, 50)),
            "fit_delta_raw_p95": float(np.percentile(delta_fit, 95)),
            "curve_delta_abs_max": max(row["delta_abs_max"] for row in curve),
            "curve_delta_std_max": max(row["delta_std"] for row in curve),
        },
        "fit_diagnostics": {
            "soup_fit_rmse_raw": fit_rmse,
            "soup_fit_mae_raw": fit_mae,
            "last_epoch_fit_rmse_raw": last_rmse,
            "last_epoch_fit_mae_raw": last_mae,
            "scope": "Fit rows only; not a generalisation metric",
        },
        "curve": curve,
        "wall_seconds": wall,
        "gpu_hours": (wall / 3600.0) if cfg.device == "cuda" else 0.0,
        "peak_gpu_memory": peak_gpu,
        "peak_rss_mb": C.peak_rss_mb(),
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "access_guard": access_guard.guard_status(),
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{cfg.run_id()}] soup fit_rmse={fit_rmse:.5f} fit_mae={fit_mae:.5f} "
        f"delta_std={delta_fit.std():.4f} wall={wall:.1f}s updates={total_updates}"
    )
    access_guard.disarm()
    return payload


def save_training_state(path, model, optimizer, scheduler, *, epoch: int, generator: torch.Generator, extra=None):
    state = {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "generator_state": generator.get_state().clone(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "extra": dict(extra or {}),
    }
    p = Path(path)
    C.ensure_dir(p.parent)
    torch.save(state, p)
    return {"path": C.track_rel(p), "sha256": C.sha256_file(p), "model_state_hash": M.state_dict_fingerprint(state["model"])}


def load_training_state(path, model, optimizer=None, scheduler=None):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


__all__ = ["TrainConfig", "train_one", "save_training_state", "load_training_state"]
