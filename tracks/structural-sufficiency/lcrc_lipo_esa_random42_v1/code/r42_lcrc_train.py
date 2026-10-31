"""LCRC-A-R3-scratch training on the random42 Train3360 view.

The network is the frozen LCRC-A implementation from
``lcrc_cross_task_preparation_v1/code/lcrc_model.py`` (one local encode, explicit
S1/S2 statistics, per-atom contribution + global rho readout, no iterative
hidden-state message passing).  The optimiser / schedule / soup logic mirrors
the v2 R3 recipe exactly:

    AdamW(peak_lr=3e-4, wd=1e-3), per-step cosine to 1e-5,
    batch 128, 360 epochs, dropout 0.10 at h, grad clip 5.0,
    average of the last five epoch-end state_dicts, no early stopping.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_metrics as MET
import r42_native_train as NT

import lcrc_data  # noqa: E402  (PREP_CODE is on sys.path via r42_data)
import lcrc_model  # noqa: E402


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    return {key: torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0) for key in keys}


def _predict_lcrc(model: torch.nn.Module, cache: Mapping[str, Any], device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = lcrc_data.collate(cache, rows)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def train_lcrc(
    *,
    cfg: Mapping[str, Any],
    split: Mapping[str, Any],
    labels: L.TrainValidLabels,
    device: torch.device,
    run_dir: Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    soup_last: int | None = None,
    label: str = "LCRC-A-R3-scratch",
) -> dict[str, Any]:
    labels = L.assert_train_valid_only(labels)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    views = D.build_lcrc_views(split, labels)
    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    if limit_train is not None:
        train_cache = D.select_cache(train_cache, list(range(min(int(limit_train), int(train_cache["n_molecules"])))), split_label="smoke_train")
    if limit_valid is not None:
        valid_cache = D.select_cache(valid_cache, list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))), split_label="smoke_valid")
    # Normalisation is always fitted on the actual training view (Train3360 or
    # the documented smoke prefix); Validation never contributes.
    train_norm = lcrc_data.fit_norm(train_cache, spec, source="train3360_structures" if limit_train is None else "smoke_train_structures")
    target_norm = D.fit_target_scaler(labels.y_train[: int(train_cache["n_molecules"])])
    n_train = int(train_cache["n_molecules"])
    epochs = int(max_epochs if max_epochs is not None else cfg["epochs"])
    soup_k = int(soup_last if soup_last is not None else C.SOUP_LAST)
    if epochs <= 0 or soup_k <= 0:
        raise ValueError("epochs and soup_last must be positive")
    batch_size = int(cfg["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    torch.manual_seed(int(cfg["seed"]))
    model = lcrc_model.LCRC(spec, train_norm, lcrc_model.DEFAULT_CONFIG).to(device)
    params_total = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["peak_lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(cfg["min_lr"]))
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + C.SHUFFLE_OFFSET)
    y_train_all = labels.y_train[:n_train]
    y_train_std = NT.target_standardise(y_train_all, target_norm)
    y_valid = labels.y_valid[: int(valid_cache["n_molecules"])]

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_peak = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        batch_order.extend(batches)
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        for rows in batches:
            batch = lcrc_data.collate(train_cache, rows)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            y = y_train_std[rows].to(device)
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(cfg["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            scheduler.step()
            err = (pred.view(-1) - y.view(-1)).detach()
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(err.numel())
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > soup_k:
            snapshots.pop(0)

        valid_pred = NT.denormalise(_predict_lcrc(model, valid_cache, device), target_norm)
        valid_metrics = MET.all_metrics(valid_pred, y_valid)
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss_standardized": float(sq_sum / max(seen, 1)),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "valid_rmse_raw": valid_metrics["rmse"],
                "valid_mae_raw": valid_metrics["mae"],
                "valid_r2_raw": valid_metrics["r2"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - t0,
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    soup = _average_states(snapshots)
    soup_hash = state_dict_fingerprint(soup)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    model.eval()

    train_pred = NT.denormalise(_predict_lcrc(model, train_cache, device), target_norm)
    train_metrics = MET.all_metrics(train_pred, y_train_all)
    valid_pred = NT.denormalise(_predict_lcrc(model, valid_cache, device), target_norm)
    valid_metrics = MET.all_metrics(valid_pred, y_valid)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        _predict_lcrc(model, valid_cache, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    if device.type == "cuda":
        gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": soup,
            "model": "LCRC-A-R3-scratch",
            "soup_members": [int(e) for e in range(epochs - len(snapshots) + 1, epochs + 1)],
            "soup_state_hash": soup_hash,
        },
        checkpoint_path,
    )
    np.savez(run_dir / "train_predictions.npz", rows=np.asarray(split["rows"]["train"][:n_train], dtype=np.int64), pred=train_pred)
    np.savez(
        run_dir / "valid_predictions.npz",
        rows=np.asarray(split["rows"]["valid"][: int(valid_cache["n_molecules"])], dtype=np.int64),
        pred=valid_pred,
    )

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "model": "LCRC-A-R3-scratch",
        "label": label,
        "seed": int(cfg["seed"]),
        "config": dict(cfg),
        "config_hash": C.config_hash(dict(cfg)),
        "upstream_commit": C.UPSTREAM_COMMIT,
        "env": {
            **C.env_info(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "precision": "fp32",
            "optimizer": "torch.optim.AdamW",
        },
        "data": {
            "split_file_sha256": C.sha256_file(C.SPLIT_FILE),
            "train_rows_sha256": C.rows_sha256_int64(split["rows"]["train"][:n_train]),
            "valid_rows_sha256": C.rows_sha256_int64(split["rows"]["valid"][: int(valid_cache["n_molecules"])]),
            "n_train": n_train,
            "n_valid": int(valid_cache["n_molecules"]),
            "lcrc_cache_path": C.track_rel(views["built"]["path"]),
            "lcrc_cache_sha256": views["built"]["file_sha256"],
            "lcrc_cache_content_fingerprint": views["built"]["content_fingerprint"],
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
            "target_norm": target_norm,
            "label_audit": dict(labels.audit),
            "test_labels_read": False,
        },
        "model_audit": {
            "parameters_total": params_total,
            "class": "lcrc_model.LCRC (LCRC-A, lipo_v1 schema)",
            "has_ssl": False,
            "has_iterative_message_passing": False,
            "checkpoint_loaded": False,
            "input_widths": {
                "atom_feat": int(spec.atom_feat_dim),
                "bond_feat": int(spec.bond_feat_dim),
                "global": int(spec.global_dim),
                "scalars": int(spec.n_scalars),
                "n_nbr_codes": int(spec.n_nbr_codes),
                "n_path_codes": int(spec.n_path_codes),
            },
        },
        "training": {
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "batch_size": batch_size,
            "batch_order_sha256": NT.batch_order_hash(batch_order),
            "early_stopping": False,
            "checkpoint_rule": str(cfg["checkpoint"]),
            "soup_members": [int(e) for e in range(epochs - len(snapshots) + 1, epochs + 1)],
            "soup_member_count": len(snapshots),
            "scheduler": "per_step_cosine",
            "gradient_clip": float(cfg["gradient_clip"]),
            "dropout": float(cfg["dropout"]),
            "loss": "mse_on_train3360_standardised_target",
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "valid": {**valid_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "train_loss_standardized_last_epoch": float(curve[-1]["train_loss_standardized"]),
            "train_mae_standardized_last_epoch": float(curve[-1]["train_mae_standardized"]),
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "inference_seconds_per_valid420": float(inference_seconds),
            "inference_batch_size": 256,
            "inference_device": str(device),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
        },
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(grad_norm_max)},
        "curve": curve,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "soup_state_hash": soup_hash,
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
        },
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    C.log(
        f"[{label}] params={params_total} epochs={len(curve)} train={train_metrics['rmse']:.5f} "
        f"valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["train_lcrc", "_predict_lcrc"]
