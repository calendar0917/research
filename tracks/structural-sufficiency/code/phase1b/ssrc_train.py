"""Phase 1b training / evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ssrc_data import LabelVault, gather_batch, sha256_file
from ssrc_models import (
    GRAPH_ONLY_KEYS,
    PATCH_ONLY_KEYS,
    RELATIONAL_KEYS,
    ModelConfig,
    assert_graph_only_batch,
    assert_patch_only_batch,
    assert_relational_batch,
    build_model,
    filter_batch,
    parameter_count,
)

MODEL_KEYS = {
    "S0": PATCH_ONLY_KEYS,
    "R1": RELATIONAL_KEYS,
    "M1": RELATIONAL_KEYS,
    "M2": RELATIONAL_KEYS,
    "G_ref": GRAPH_ONLY_KEYS,
}

MODEL_ASSERT = {
    "S0": assert_patch_only_batch,
    "R1": assert_relational_batch,
    "M1": assert_relational_batch,
    "M2": assert_relational_batch,
    "G_ref": assert_graph_only_batch,
}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4
    grad_clip: float = 5.0
    eval_train_every: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip,
            "eval_train_every": self.eval_train_every,
            "model": self.model.as_dict(),
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([int(seed), int(epoch)]).permutation(n)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / sst) if sst > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": int(y.size),
        "mean_error": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
    }


def _model_batch(model_name: str, batch: Mapping[str, Any]) -> dict[str, Any]:
    keys = MODEL_KEYS[model_name]
    out = filter_batch(batch, keys)
    MODEL_ASSERT[model_name](out)
    return out


@torch.no_grad()
def predict_rows(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y_all: torch.Tensor,
    cache: Mapping[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for start in range(0, len(rows), cfg.batch_size):
        chunk = [int(r) for r in rows[start : start + cfg.batch_size]]
        y = y_all[torch.as_tensor(chunk, dtype=torch.long)]
        batch = _to_device(gather_batch(cache, chunk, y), device)
        batch = _model_batch(model_name, batch)
        pred = model(batch).detach().cpu().numpy()
        preds.append(pred)
        truths.append(y.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(truths)


def _measure_flops(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y_all: torch.Tensor,
    cache: Mapping[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        chunk = [int(r) for r in rows[: cfg.batch_size]]
        y = y_all[torch.as_tensor(chunk, dtype=torch.long)]
        batch = _to_device(gather_batch(cache, chunk, y), device)
        batch = _model_batch(model_name, batch)
        with FlopCounterMode(display=False) as fcm:
            model(batch)
        return {
            "flops_forward": int(fcm.get_total_flops()),
            "flops_method": "FlopCounterMode forward; PyG scatter/GINE message ops may be uncounted (lower bound)",
            "flops_batch_size": len(chunk),
            "flops_ok": True,
        }
    except Exception as exc:  # pragma: no cover
        return {"flops_forward": None, "flops_method": "FlopCounterMode failed", "flops_error": repr(exc), "flops_ok": False}


def _measure_step_time(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y_all: torch.Tensor,
    cache: Mapping[str, Any],
    cfg: TrainConfig,
    device: torch.device,
    n_steps: int = 10,
) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    chunk = [int(r) for r in rows[: cfg.batch_size]]
    y = y_all[torch.as_tensor(chunk, dtype=torch.long)]
    batch = _to_device(gather_batch(cache, chunk, y), device)
    batch = _model_batch(model_name, batch)
    times: list[float] = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(batch)
        loss = F.l1_loss(pred, batch["y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda_version"] = torch.version.cuda
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        info["git_commit"] = None
        info["git_dirty"] = None
    return info


def train_one(
    model_name: str,
    dataset: str,
    seed: int,
    train_rows: Sequence[int],
    monitor_rows: Sequence[int],
    reserved_rows: Sequence[int],
    y_all: torch.Tensor,
    cache: Mapping[str, Any],
    cfg: TrainConfig,
    device: torch.device,
    out_path: Path,
    provenance: Mapping[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    atom_dim = int(cache["meta"]["atom_dim"])
    edge_dim = int(cache["meta"]["bond_dim"])
    model = build_model(model_name, atom_dim, edge_dim, cfg.model).to(device)
    params = parameter_count(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.patience
    curves: list[dict[str, Any]] = []
    step_times: list[float] = []
    last_train_metrics: dict[str, float] | None = None
    last_monitor_metrics: dict[str, float] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_monitor_metrics: dict[str, float] | None = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        perm = _batch_permutation(len(train_rows), seed, epoch)
        losses: list[float] = []
        for start in range(0, len(train_rows), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            chunk = [int(train_rows[int(i)]) for i in idx]
            y = y_all[torch.as_tensor(chunk, dtype=torch.long)]
            batch = _to_device(gather_batch(cache, chunk, y), device)
            batch = _model_batch(model_name, batch)
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred = model(batch)
            loss = F.l1_loss(pred, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - t0) * 1000.0)
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        mon_pred, mon_true = predict_rows(model, model_name, monitor_rows, y_all, cache, cfg, device)
        mon_metrics = regression_metrics(mon_true, mon_pred)
        last_monitor_metrics = mon_metrics

        do_train_eval = (epoch % cfg.eval_train_every == 0) or (epoch == cfg.epochs - 1)
        if do_train_eval:
            tr_pred, tr_true = predict_rows(model, model_name, train_rows, y_all, cache, cfg, device)
            tr_metrics = regression_metrics(tr_true, tr_pred)
            last_train_metrics = tr_metrics
        else:
            tr_metrics = last_train_metrics

        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "lr": float(opt.param_groups[0]["lr"]),
                **({f"train_{k}": v for k, v in tr_metrics.items()} if tr_metrics else {}),
                **{f"monitor_{k}": v for k, v in mon_metrics.items()},
            }
        )

        improved = mon_metrics["mae"] < best_mae - cfg.min_delta
        if improved:
            best_mae = mon_metrics["mae"]
            best_epoch = epoch
            best_state = _state_to_cpu(model.state_dict())
            best_monitor_metrics = dict(mon_metrics)
            patience_left = cfg.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - t_start
    peak_alloc = float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / 1e6) if device.type == "cuda" else None

    if best_state is None:
        raise RuntimeError(f"{model_name}/{dataset}/seed{seed}: no finite monitor MAE")
    model.load_state_dict(best_state)
    best_mon_pred, best_mon_true = predict_rows(model, model_name, monitor_rows, y_all, cache, cfg, device)
    best_monitor_metrics = regression_metrics(best_mon_true, best_mon_pred)
    best_tr_pred, best_tr_true = predict_rows(model, model_name, train_rows, y_all, cache, cfg, device)
    best_train_metrics = regression_metrics(best_tr_true, best_tr_pred)

    flops = _measure_flops(model, model_name, train_rows, y_all, cache, cfg, device)
    step_ms = _measure_step_time(model, model_name, train_rows, y_all, cache, cfg, device)

    payload: dict[str, Any] = {
        "protocol_id": f"ssrc-phase1b-{dataset}-v1",
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "smoke": bool(smoke),
        "n_train": len(train_rows),
        "n_monitor": len(monitor_rows),
        "n_reserved_not_scored": len(reserved_rows),
        "train_row_sha256": hashlib.sha256(",".join(str(int(r)) for r in train_rows).encode()).hexdigest(),
        "monitor_row_sha256": hashlib.sha256(",".join(str(int(r)) for r in monitor_rows).encode()).hexdigest(),
        "config": cfg.as_dict(),
        "config_sha256": cfg.sha256(),
        "parameters": int(params),
        "flops": flops,
        "step_time_ms_median": step_ms,
        "step_time_ms_mean": float(np.mean(step_times)) if step_times else None,
        "step_time_ms_p90": float(np.quantile(step_times, 0.90)) if step_times else None,
        "n_steps": len(step_times),
        "train_seconds": train_seconds,
        "gpu_seconds": train_seconds if device.type == "cuda" else 0.0,
        "peak_gpu_memory_mb_allocated": peak_alloc,
        "peak_gpu_memory_mb_reserved": peak_reserved,
        "best_epoch": int(best_epoch),
        "best_monitor_mae": float(best_mae),
        "best_metrics": {"train": best_train_metrics, "monitor": best_monitor_metrics},
        "last_metrics": {"train": last_train_metrics, "monitor": last_monitor_metrics},
        "curves": curves,
        "monitor_mol_index": [int(r) for r in monitor_rows],
        "monitor_y": [float(x) for x in best_mon_true.tolist()],
        "monitor_pred_best": [float(x) for x in best_mon_pred.tolist()],
        "train_mol_index": [int(r) for r in train_rows],
        "train_pred_best": [float(x) for x in best_tr_pred.tolist()],
        "environment": environment_info(device),
        "provenance": dict(provenance),
        "information_access": sorted(MODEL_KEYS[model_name]),
        "scope": {
            "official_valid_test_labels_read": False,
            "internal_reserved_labels_read": False,
            "reserved_labels_retained": 0,
            "labels_used": "Fit/Monitor only",
            "model_selection": "best Monitor MAE; monitor used for early stopping",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
