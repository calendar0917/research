"""Phase 1a training / evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from phase1a_data import (
    PatchCache,
    collate_graph_batch,
    collate_patch_lists,
)
from phase1a_models import (
    ModelG,
    ModelS,
    ModelX,
    assert_graph_only_batch,
    assert_patch_only_batch,
    parameter_count,
)


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64
    local_layers: int = 2
    s_ffn_layers: int = 2
    s_ffn_hidden: int = 128
    x_attn_layers: int = 2
    x_heads: int = 4
    x_ffn_hidden: int = 128
    dropout: float = 0.1
    g_hidden: int = 96
    g_layers: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 30
    min_delta: float = 1e-4
    grad_clip: float = 5.0
    radius: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "local_layers": self.local_layers,
            "s_ffn_layers": self.s_ffn_layers,
            "s_ffn_hidden": self.s_ffn_hidden,
            "x_attn_layers": self.x_attn_layers,
            "x_heads": self.x_heads,
            "x_ffn_hidden": self.x_ffn_hidden,
            "dropout": self.dropout,
            "g_hidden": self.g_hidden,
            "g_layers": self.g_layers,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip,
            "radius": self.radius,
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


def build_models(seed: int, cfg: TrainConfig, device: torch.device) -> dict[str, torch.nn.Module]:
    """Build S, X, G.

    S and X share exactly the same local encoder and regression head
    initialisation; X then adds its own attention stack, S its own token FFNs.
    """
    torch.manual_seed(seed)
    s = ModelS(
        hidden=cfg.hidden,
        local_layers=cfg.local_layers,
        ffn_layers=cfg.s_ffn_layers,
        ffn_hidden=cfg.s_ffn_hidden,
        dropout=cfg.dropout,
    ).to(device)
    torch.manual_seed(seed)
    x = ModelX(
        hidden=cfg.hidden,
        local_layers=cfg.local_layers,
        attn_layers=cfg.x_attn_layers,
        nhead=cfg.x_heads,
        ffn_hidden=cfg.x_ffn_hidden,
        dropout=cfg.dropout,
    ).to(device)
    x.local.load_state_dict(s.local.state_dict())
    x.head.load_state_dict(s.head.state_dict())
    torch.manual_seed(seed)
    g = ModelG(hidden=cfg.g_hidden, layers=cfg.g_layers, dropout=cfg.dropout).to(device)
    return {"S": s, "X": x, "G": g}


def _batch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([int(seed), int(epoch)]).permutation(n)


def _slice(seq: Sequence[Any], idx: np.ndarray) -> list[Any]:
    return [seq[int(i)] for i in idx]


def _make_batch(
    model_name: str,
    rows: Sequence[int],
    y: torch.Tensor,
    patch_cache: PatchCache,
) -> dict[str, torch.Tensor]:
    mol_index = torch.as_tensor([int(r) for r in rows], dtype=torch.long)
    if model_name == "G":
        raws = [patch_cache.raws[int(r)] for r in rows]
        batch = collate_graph_batch(raws, y, mol_index)
        assert_graph_only_batch(batch)
        return batch
    patch_lists = patch_cache.patch_lists(rows)
    batch = collate_patch_lists(patch_lists, y, mol_index)
    assert_patch_only_batch(batch)
    return batch


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict_with_labels(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y_raw: torch.Tensor,
    y_mean: float,
    y_std: float,
    patch_cache: PatchCache,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for start in range(0, len(rows), cfg.batch_size):
        chunk_rows = rows[start : start + cfg.batch_size]
        y_chunk = y_raw[start : start + cfg.batch_size]
        y_std_chunk = (y_chunk - y_mean) / y_std
        batch = _make_batch(model_name, chunk_rows, y_std_chunk, patch_cache)
        batch = _to_device(batch, device)
        pred = model(batch).detach().cpu().numpy()
        preds.append(pred * y_std + y_mean)
        truths.append(y_chunk.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(truths)


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


def _state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _measure_flops(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y: torch.Tensor,
    patch_cache: PatchCache,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Forward FLOPs on one fixed batch; PyG scatter ops may be uncounted."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        chunk_rows = list(rows[: cfg.batch_size])
        y_chunk = y[: len(chunk_rows)]
        batch = _to_device(_make_batch(model_name, chunk_rows, y_chunk, patch_cache), device)
        with FlopCounterMode(display=False) as fcm:
            model(batch)
        flops = int(fcm.get_total_flops())
        return {
            "flops_forward": flops,
            "flops_method": "torch.utils.flop_counter.FlopCounterMode forward; PyG scatter ops may be uncounted (lower bound)",
            "flops_batch_size": len(chunk_rows),
            "flops_ok": True,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "flops_forward": None,
            "flops_method": "FlopCounterMode failed",
            "flops_error": repr(exc),
            "flops_ok": False,
        }


def _measure_step_time(
    model: torch.nn.Module,
    model_name: str,
    rows: Sequence[int],
    y: torch.Tensor,
    patch_cache: PatchCache,
    cfg: TrainConfig,
    device: torch.device,
    n_steps: int = 10,
) -> float:
    """Median training-step wall time (ms) for one fixed batch."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    chunk_rows = list(rows[: cfg.batch_size])
    y_chunk = y[: len(chunk_rows)]
    batch = _to_device(_make_batch(model_name, chunk_rows, y_chunk, patch_cache), device)
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


def train_one(
    model_name: str,
    seed: int,
    n_label: str | int,
    train_rows: Sequence[int],
    monitor_rows: Sequence[int],
    train_y_raw: torch.Tensor,
    monitor_y_raw: torch.Tensor,
    patch_cache: PatchCache,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
    reserved_rows: Sequence[int],
) -> dict[str, Any]:
    seed_everything(seed)
    models = build_models(seed, cfg, device)
    model = models[model_name]
    params = parameter_count(model)

    y_mean = float(train_y_raw.mean().item())
    y_std = float(train_y_raw.std(unbiased=False).item())
    if y_std <= 0:
        raise ValueError("degenerate training labels")
    train_y_std = (train_y_raw - y_mean) / y_std

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = cfg.patience
    curves: list[dict[str, Any]] = []
    last_monitor_pred: np.ndarray | None = None
    last_train_pred: np.ndarray | None = None
    last_monitor_metrics: dict[str, float] | None = None
    last_train_metrics: dict[str, float] | None = None
    step_times: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        perm = _batch_permutation(len(train_rows), seed, epoch)
        epoch_losses: list[float] = []
        for start in range(0, len(train_rows), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            chunk_rows = _slice(train_rows, idx)
            y_chunk = train_y_std[torch.as_tensor(idx, dtype=torch.long)]
            batch = _to_device(_make_batch(model_name, chunk_rows, y_chunk, patch_cache), device)
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_step0 = time.perf_counter()
            pred = model(batch)
            loss = F.l1_loss(pred, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - t_step0) * 1000.0)
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        train_pred, train_true = predict_with_labels(
            model, model_name, train_rows, train_y_raw, y_mean, y_std, patch_cache, cfg, device
        )
        mon_pred, mon_true = predict_with_labels(
            model, model_name, monitor_rows, monitor_y_raw, y_mean, y_std, patch_cache, cfg, device
        )
        train_metrics = regression_metrics(train_true, train_pred)
        mon_metrics = regression_metrics(mon_true, mon_pred)
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                "lr": float(opt.param_groups[0]["lr"]),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"monitor_{k}": v for k, v in mon_metrics.items()},
            }
        )
        last_monitor_pred = mon_pred
        last_train_pred = train_pred
        last_monitor_metrics = mon_metrics
        last_train_metrics = train_metrics

        improved = mon_metrics["mae"] < best_mae - cfg.min_delta
        if improved:
            best_mae = mon_metrics["mae"]
            best_epoch = epoch
            best_state = _state_to_cpu(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - t_start
    peak_alloc = (
        float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else None
    )
    peak_reserved = (
        float(torch.cuda.max_memory_reserved(device) / 1e6) if device.type == "cuda" else None
    )

    if best_state is None:
        raise RuntimeError(f"{model_name} n={n_label} seed={seed}: no finite monitor MAE")
    model.load_state_dict(best_state)
    best_train_pred, best_train_true = predict_with_labels(
        model, model_name, train_rows, train_y_raw, y_mean, y_std, patch_cache, cfg, device
    )
    best_mon_pred, best_mon_true = predict_with_labels(
        model, model_name, monitor_rows, monitor_y_raw, y_mean, y_std, patch_cache, cfg, device
    )
    best_train_metrics = regression_metrics(best_train_true, best_train_pred)
    best_mon_metrics = regression_metrics(best_mon_true, best_mon_pred)

    flops = _measure_flops(model, model_name, train_rows, train_y_std, patch_cache, cfg, device)
    step_ms = _measure_step_time(model, model_name, train_rows, train_y_std, patch_cache, cfg, device)

    payload: dict[str, Any] = {
        "protocol_id": "ssrc-phase1a-lipo-v1",
        "model": model_name,
        "seed": int(seed),
        "n_label": n_label,
        "n_train": len(train_rows),
        "n_monitor": len(monitor_rows),
        "train_row_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(train_rows, dtype=np.int64)).tobytes()
        ).hexdigest(),
        "monitor_row_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(monitor_rows, dtype=np.int64)).tobytes()
        ).hexdigest(),
        "config": cfg.as_dict(),
        "config_sha256": cfg.sha256(),
        "parameters": int(params),
        "parameters_by_module": {
            "total": int(params),
            "local": int(parameter_count(model.local)) if hasattr(model, "local") else None,
            "head": int(parameter_count(model.head)),
        },
        "flops": flops,
        "step_time_ms_median": step_ms,
        "step_time_ms_mean": float(np.mean(step_times)) if step_times else None,
        "step_time_ms_p90": float(np.quantile(step_times, 0.90)) if step_times else None,
        "step_time_total_seconds": float(np.sum(step_times) / 1000.0) if step_times else None,
        "n_steps": len(step_times),
        "train_seconds": train_seconds,
        "gpu_seconds": train_seconds if device.type == "cuda" else 0.0,
        "peak_gpu_memory_mb_allocated": peak_alloc,
        "peak_gpu_memory_mb_reserved": peak_reserved,
        "preprocess_seconds": patch_cache.build_seconds,
        "patch_cache_n_patches": patch_cache.n_patches,
        "label_standardization": {"mean": y_mean, "std": y_std, "n_fit": len(train_rows)},
        "best_epoch": int(best_epoch),
        "best_monitor_mae": float(best_mae),
        "best_metrics": {"train": best_train_metrics, "monitor": best_mon_metrics},
        "last_metrics": {"train": last_train_metrics, "monitor": last_monitor_metrics},
        "curves": curves,
        "monitor_mol_index": [int(r) for r in monitor_rows],
        "monitor_y": [float(x) for x in monitor_y_raw.tolist()],
        "monitor_pred_best": [float(x) for x in best_mon_pred.tolist()],
        "monitor_pred_last": [float(x) for x in last_monitor_pred.tolist()] if last_monitor_pred is not None else None,
        "train_mol_index": [int(r) for r in train_rows],
        "train_y": [float(x) for x in train_y_raw.tolist()],
        "train_pred_best": [float(x) for x in best_train_pred.tolist()],
        "train_pred_last": [float(x) for x in last_train_pred.tolist()] if last_train_pred is not None else None,
        "environment": environment_info(device),
        "scope": {
            "official_valid_test_labels_read": False,
            "internal_reserved_labels_read": False,
            "reserved_rows_in_train_or_monitor": bool(
                set(int(r) for r in list(train_rows) + list(monitor_rows))
                & set(int(r) for r in reserved_rows)
            ),
            "labels_used": "Fit/Monitor only",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_n{n_label}_seed{seed}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


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
