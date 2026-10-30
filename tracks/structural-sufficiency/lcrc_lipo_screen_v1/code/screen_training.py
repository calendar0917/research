"""Frozen supervised training recipe for arms A, B and C.

Recipe (frozen before any GPU run):

* Fit2352 only; Monitor/Reserved/official valid+test are never touched.
* target ``z = (y - mean_fit) / std_fit``; MSE loss.
* AdamW(peak_lr=1e-3, weight_decay=1e-5), per-optimizer-step cosine decay to 1e-5.
* batch 128, 360 epochs -> 19 steps/epoch -> 6840 updates.
* gradient clipping at 5.0, dropout 0.
* checkpoint = elementwise mean of the last 5 epoch-end state dicts (no Monitor
  selection, no arm-specific epochs).

The module deliberately has no Monitor import, no Monitor evaluation hook and no
early-stopping callback.
"""

from __future__ import annotations

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
import fit_labels
import screen_common as C
import screen_data as D
import screen_models as M


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

    def validate(self) -> None:
        if self.arm not in C.NN_ARMS:
            raise ValueError(f"arm must be one of {C.NN_ARMS}, got {self.arm!r}")
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
                raise PermissionError("CPU smoke is CPU-only")
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


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return math.sqrt(total)


def train_one(cfg: TrainConfig) -> dict[str, Any]:
    cfg.validate()
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
    if cfg.fit_limit is not None:
        train_rows = fit_rows[: int(cfg.fit_limit)]
    else:
        train_rows = fit_rows
    n = len(train_rows)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total_updates = steps_per_epoch * cfg.epochs

    device = torch.device(cfg.device)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model, model_info = M.build_model(cfg.arm, spec, norm, cfg.seed, device)
    soup_report = M.soup_safety_report(model)
    if not soup_report["all_state_entries_float"]:
        raise RuntimeError(f"parameter averaging is not safe: {soup_report['non_float_entries']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=cfg.min_lr
    )
    generator = torch.Generator().manual_seed(int(cfg.seed) + C.TRAIN_SHUFFLE_OFFSET)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    order_hash_actual = hashlib.sha256()
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite_events = 0
    started = time.perf_counter()

    train_row_index = {int(r): i for i, r in enumerate(fit_rows)}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        epoch_abs = 0.0
        epoch_sq = 0.0
        seen = 0
        lr_last = None
        for rows in D.shuffle_batches(n, cfg.batch_size, generator):
            order_hash_actual.update(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes())
            # ``rows`` indexes the training subset, map back to cache rows.
            cache_rows = [train_rows[int(i)] for i in rows]
            batch = D.to_device(D.collate(caches["fit"]["cache"], cache_rows), device)
            y = torch.as_tensor(
                (y_fit[[train_row_index[r] for r in cache_rows]] - target_norm["mean"]) / target_norm["scale"],
                dtype=torch.float32,
                device=device,
            )
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            grad_norm_max = max(grad_norm_max, gnorm)
            if gnorm > cfg.grad_clip:
                clip_events += 1
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            error = (pred.view(-1) - y.view(-1)).detach()
            epoch_abs += float(error.abs().sum())
            epoch_sq += float((error * error).sum())
            seen += int(error.numel())
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps_per_epoch,
                "train_mae_normalised": epoch_abs / max(seen, 1),
                "train_mse_normalised": epoch_sq / max(seen, 1),
                "lr_last": lr_last,
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > cfg.soup_last:
            snapshots.pop(0)
        if epoch % 40 == 0 or epoch == 1 or epoch == cfg.epochs:
            C.log(
                f"[{cfg.run_id()}] ep={epoch:03d} train_mae_norm={curve[-1]['train_mae_normalised']:.6f} "
                f"lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s"
            )

    wall = time.perf_counter() - started
    independent_order_hash = D.batch_order_hash(cfg.seed, n=n, epochs=cfg.epochs)
    members = list(range(cfg.epochs - len(snapshots) + 1, cfg.epochs + 1))
    first_member = max(1, cfg.epochs - cfg.soup_last + 1)
    if members != list(range(first_member, cfg.epochs + 1)):
        raise RuntimeError(f"soup members {members} are not the fixed last-5 epochs")

    soup = M.average_states(snapshots)
    last_state = snapshots[-1]
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    soup_state_hash = M.state_dict_fingerprint(soup)
    last_state_hash = M.state_dict_fingerprint(last_state)

    # Fit-only diagnostics: labels are permitted here, Monitor is not touched.
    fit_pred_norm = D.predict_cache(model, caches["fit"]["cache"], device)["pred"]
    fit_pred_raw = D.denormalise(fit_pred_norm, target_norm)[:n]
    fit_y_raw = y_fit[:n]
    fit_mae = float(np.abs(fit_pred_raw - fit_y_raw).mean())
    fit_rmse = float(np.sqrt(np.mean((fit_pred_raw - fit_y_raw) ** 2)))

    model.load_state_dict({k: v.to(device) for k, v in last_state.items()})
    last_pred_norm = D.predict_cache(model, caches["fit"]["cache"], device)["pred"]
    last_pred_raw = D.denormalise(last_pred_norm, target_norm)[:n]
    last_mae = float(np.abs(last_pred_raw - fit_y_raw).mean())
    last_rmse = float(np.sqrt(np.mean((last_pred_raw - fit_y_raw) ** 2)))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})

    out_dir = Path(cfg.out_root) / cfg.run_id()
    C.ensure_dir(out_dir)
    torch.save(soup, out_dir / "soup.pt")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "run_id": cfg.run_id(),
        "arm": cfg.arm,
        "seed": int(cfg.seed),
        "smoke": bool(cfg.smoke),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": C.git_commit(),
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
        "data": {
            "n_train_rows": n,
            "fit_limit": cfg.fit_limit,
            "fit_rows_sha256": C.prep_common.row_sha256(train_rows),
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
        "init_state_hash": model_info["state_hash"],
        "soup_state_hash": soup_state_hash,
        "last_epoch_state_hash": last_state_hash,
        "soup_tensor_file": C.track_rel(out_dir / "soup.pt"),
        "batch_order": {
            "actual_hash": order_hash_actual.hexdigest(),
            "independent_recomputation": independent_order_hash,
            "matches": order_hash_actual.hexdigest() == independent_order_hash,
            "offset": C.TRAIN_SHUFFLE_OFFSET,
        },
        "anomalies": {
            "grad_clip_events": clip_events,
            "grad_norm_max": grad_norm_max,
            "nonfinite_events": nonfinite_events,
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
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "access_guard": access_guard.guard_status(),
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{cfg.run_id()}] soup fit_rmse={fit_rmse:.5f} fit_mae={fit_mae:.5f} "
        f"wall={wall:.1f}s updates={total_updates}"
    )
    access_guard.disarm()
    return payload


# ---------------------------------------------------------------------------
# resumable state (used by the CPU gate; mirrors an optimizer-step checkpoint)
# ---------------------------------------------------------------------------


def save_training_state(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    generator: torch.Generator,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    return {
        "path": C.track_rel(p),
        "sha256": C.sha256_file(p),
        "epoch": int(epoch),
        "model_state_hash": M.state_dict_fingerprint(state["model"]),
    }


def load_training_state(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


__all__ = ["TrainConfig", "train_one", "save_training_state", "load_training_state"]
