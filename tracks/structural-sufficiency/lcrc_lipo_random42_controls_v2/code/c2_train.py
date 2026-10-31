"""Training loops for the two new controls.

The S2-off loop mirrors the *executed* random42 v1 LCRC-A recipe exactly
(AdamW 3e-4 -> 1e-5 per-step cosine, wd 1e-3, batch 128, 360 epochs, last-5
soup, no early stopping, no stochastic dropout).  The only change is the model
builder: ``v2_models.LCRCCapacityMatched`` instead of ``lcrc_model.LCRC``.

The PNA-small loop mirrors random42 v1 PNA-native exactly except that the width
comes from the frozen CPU-only parameter-count search.  It loads only Train3360
and Validation420 structure objects.
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

import c2_common as C
import c2_data as CD
import c2_models as CM

import r42_data as D  # noqa: E402
import r42_labels as L  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402
import lcrc_data  # noqa: E402


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = list(states[0].keys())
    return {key: torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0) for key in keys}


def _move_lcrc_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _predict_lcrc(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(cache["n_molecules"]), int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), int(cache["n_molecules"]))))
            batch = _move_lcrc_batch(lcrc_data.collate(cache, rows), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# S2-off / capacity-matched
# ---------------------------------------------------------------------------


def train_s2off(
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
    label: str = "LCRC-S2off-capacity-matched",
) -> dict[str, Any]:
    labels = L.assert_train_valid_only(labels)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    views = CD.build_lcrc_views(split, labels)
    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    if limit_train is not None:
        train_cache = D.select_cache(train_cache, list(range(min(int(limit_train), int(train_cache["n_molecules"])))), split_label="smoke_train")
    if limit_valid is not None:
        valid_cache = D.select_cache(valid_cache, list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))), split_label="smoke_valid")
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    if limit_train is None:
        train_norm = views["train_norm"]
        target_norm = views["target_norm"]
    else:
        train_norm = lcrc_data.fit_norm(train_cache, spec, source="smoke_train_structures")
        target_norm = D.fit_target_scaler(labels.y_train[:n_train])
    epochs = int(max_epochs if max_epochs is not None else cfg["epochs"])
    soup_k = int(soup_last if soup_last is not None else C.SOUP_LAST)
    if epochs <= 0 or soup_k <= 0:
        raise ValueError("epochs and soup_last must be positive")
    batch_size = int(cfg["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    model, base, model_audit = CM.build_s2off(spec, train_norm, int(cfg["seed"]), device)
    params_total = int(model_audit["parameters_total"])
    params_effective = int(model_audit["parameters_effective"])
    y_train_all = labels.y_train[:n_train]
    y_train_std = NT.target_standardise(y_train_all, target_norm)
    y_valid = labels.y_valid[:n_valid]

    # Pre-training mechanical gates on one real Train batch.
    gate_rows = list(range(min(8, n_train)))
    gate_batch = _move_lcrc_batch(lcrc_data.collate(train_cache, gate_rows), device)
    s2_gate_pre = CM.s2off_invariance(model, gate_batch)
    if not s2_gate_pre["identical_restricted_batch"] or not s2_gate_pre["identical_perturbed_s2"]:
        raise RuntimeError(f"S2-off output depends on typed-S2 inputs: {s2_gate_pre}")
    grad_gate = CM.s2off_gradient_gate(model, gate_batch)
    if not grad_gate["all_replacement_receive_nonzero_grad"]:
        raise RuntimeError(f"S2-off replacement branch has a parameter without gradient: {grad_gate}")
    if not grad_gate["dead_s2_input_columns_zero_grad"]:
        raise RuntimeError(f"S2-off dead S2 input columns receive gradient: {grad_gate}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["peak_lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(cfg["min_lr"])
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + C.SHUFFLE_OFFSET)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
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
            batch = _move_lcrc_batch(lcrc_data.collate(train_cache, rows), device)
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

    s2_gate_post = CM.s2off_invariance(model, _move_lcrc_batch(lcrc_data.collate(valid_cache, list(range(min(8, n_valid)))), device))
    if not s2_gate_post["identical_restricted_batch"] or not s2_gate_post["identical_perturbed_s2"]:
        raise RuntimeError(f"post-soup S2-off output depends on typed-S2 inputs: {s2_gate_post}")

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
    soup_members = [int(e) for e in range(epochs - len(snapshots) + 1, epochs + 1)]
    torch.save(
        {
            "state_dict": soup,
            "model": "LCRC-S2off-capacity-matched",
            "soup_members": soup_members,
            "soup_state_hash": soup_hash,
            "capacity_audit": model_audit["capacity"],
        },
        checkpoint_path,
    )
    np.savez(run_dir / "train_predictions.npz", rows=np.asarray(split["rows"]["train"][:n_train], dtype=np.int64), pred=train_pred)
    np.savez(run_dir / "valid_predictions.npz", rows=np.asarray(split["rows"]["valid"][:n_valid], dtype=np.int64), pred=valid_pred)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "model": "LCRC-S2off-capacity-matched",
        "label": label,
        "seed": int(cfg["seed"]),
        "config": dict(cfg),
        "config_hash": C.config_hash(dict(cfg)),
        "env": {
            **C.env_info(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "precision": "fp32",
            "optimizer": "torch.optim.AdamW",
        },
        "data": {
            "split_file": C.track_rel(C.V1_SPLIT_FILE),
            "split_file_sha256": C.V1_SPLIT_SHA256,
            "train_rows_sha256": C.rows_sha256_int64(split["rows"]["train"][:n_train]),
            "valid_rows_sha256": C.rows_sha256_int64(split["rows"]["valid"][:n_valid]),
            "n_train": n_train,
            "n_valid": n_valid,
            "lcrc_cache_path": C.track_rel(views["built"]["path"]),
            "lcrc_cache_sha256": views["built"]["file_sha256"],
            "lcrc_cache_content_fingerprint": views["built"]["content_fingerprint"],
            "lcrc_cache_source": views["built"]["source"],
            "v1_cache_content_fingerprint": views["built"]["v1_content_fingerprint"],
            "lcrc_cache_labels_present": False,
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
            "train_norm_is_v1_frozen": limit_train is None,
            "target_norm": target_norm,
            "label_audit": dict(labels.audit),
            "test_labels_read": False,
            "test_predictions_read": False,
        },
        "model_audit": {
            "parameters_total_nominal": params_total,
            "parameters_effective": params_effective,
            "parameters_requires_grad": int(model_audit["parameters_requires_grad"]),
            "parameters_dead": int(params_total - params_effective),
            "class": "v2_models.LCRCCapacityMatched (imported unchanged)",
            "capacity": model_audit["capacity"],
            "shared_init": model_audit["shared_init"],
            "shared_state_keys": model_audit["shared_state_keys"],
            "alternative_branch": "two-layer MLP on [a, S1_sum, S1_mean, S1joint] only",
            "typed_s2_inputs_used": False,
            "new_message_passing_or_attention": False,
        },
        "s2_off_gates": {
            "pre_train_invariance": s2_gate_pre,
            "pre_train_gradient": grad_gate,
            "post_soup_invariance": s2_gate_post,
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
            "soup_members": soup_members,
            "soup_member_count": len(snapshots),
            "scheduler": "per_step_cosine",
            "gradient_clip": float(cfg["gradient_clip"]),
            "dropout_configured": float(cfg["dropout_configured"]),
            "dropout_applied": float(cfg["dropout_applied"]),
            "dropout_note": "random42 v1's executed LCRC loop did not apply dropout; both arms use that executed recipe",
            "loss": "mse_on_train3360_standardised_target" if limit_train is None else "mse_on_smoke_standardised_target",
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
        "test_predictions_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    C.log(
        f"[{label}] nominal={params_total} effective={params_effective} epochs={len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# PNA-small
# ---------------------------------------------------------------------------


def _pna_forward(model, batch, precision: str, device: torch.device):
    with NT._autocast(device, precision):
        _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    return pred


def train_pna_small(
    *,
    cfg: Mapping[str, Any],
    split: Mapping[str, Any],
    labels: L.TrainValidLabels,
    device: torch.device,
    run_dir: Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    labels = L.assert_train_valid_only(labels)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    precision = str(cfg["precision"])
    max_epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    train_rows = [int(r) for r in split["rows"]["train"]]
    valid_rows = [int(r) for r in split["rows"]["valid"]]
    if limit_train is not None:
        train_rows = train_rows[: int(limit_train)]
    if limit_valid is not None:
        valid_rows = valid_rows[: int(limit_valid)]
    train_data = CD.load_structures_for_rows(train_rows)
    valid_data = CD.load_structures_for_rows(valid_rows)
    dims = CD.structure_dimensions()
    if dims["num_features"] != 79 or dims["edge_dim"] != 13:
        raise PermissionError("upstream Chemprop feature dimensions drifted")
    if len(train_data) != len(train_rows) or len(valid_data) != len(valid_rows):
        raise ValueError("structure/label length mismatch")

    target_norm = D.fit_target_scaler(labels.y_train[: len(train_rows)] if limit_train is not None else labels.y_train)
    y_train_std = NT.target_standardise(labels.y_train[: len(train_rows)], target_norm)
    y_valid_raw = labels.y_valid[: len(valid_rows)]
    y_valid_std = NT.target_standardise(y_valid_raw, target_norm)
    if y_train_std.numel() != len(train_data) or y_valid_raw.size != len(valid_data):
        raise ValueError("label/structure length mismatch")

    model, extra_audit = NT.build_pna(
        cfg, device, run_dir, train_data, dims["num_features"], dims["edge_dim"]
    )
    params_total = int(sum(p.numel() for p in model.parameters()))
    params_requires_grad = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if params_total != int(C.read_json(C.WIDTH_SEARCH)["selected_parameters_total"]):
        raise PermissionError(
            f"PNA-small parameter count {params_total} != frozen width-search count "
            f"{C.read_json(C.WIDTH_SEARCH)['selected_parameters_total']}"
        )

    from torch_geometric.data import Batch

    gate_batch = Batch.from_data_list(train_data[: min(16, len(train_data))]).to(device)
    grad_coverage = CM.pna_gradient_coverage(model, gate_batch)
    if not grad_coverage["all_parameter_tensors_receive_grad"]:
        raise RuntimeError(f"PNA-small has a parameter tensor without gradient: {grad_coverage}")
    del gate_batch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["lr_scheduler_factor"]),
        patience=int(cfg["lr_scheduler_patience"]),
    )

    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    batch_size = int(cfg["batch_size"])
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    patience = int(cfg["early_stopping_patience"])
    best_val = float("inf")
    best_epoch = None
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[dict[str, Any]] = []
    all_batches: list[list[int]] = []
    started = time.perf_counter()
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    stopped_early = False
    last_train_loss = None
    last_train_mae = None
    clip_events = 0
    grad_norm_max = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(train_data), generator=generator).tolist()
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        all_batches.extend(batches)
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        for rows in batches:
            batch = Batch.from_data_list([train_data[i] for i in rows]).to(device)
            y = y_train_std[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            # Keep the loss inside autocast, matching the historical v1 loop:
            # bf16 predictions and float32 targets are not mixed outside it.
            with NT._autocast(device, precision):
                _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = F.mse_loss(pred.reshape(-1), y.reshape(-1))
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch}")
            if gnorm > float(cfg["gradient_clip"]):
                clip_events += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            err = (pred.reshape(-1) - y.reshape(-1)).detach()
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(err.numel())
        last_train_loss = sq_sum / max(seen, 1)
        last_train_mae = abs_sum / max(seen, 1)

        valid_std = NT._predict(
            model,
            valid_data,
            kind="pna",
            device=device,
            batch_size=batch_size,
            precision=precision,
        )
        valid_raw = NT.denormalise(valid_std, target_norm)
        val_loss = float(np.mean((valid_std - y_valid_std.numpy()) ** 2))
        val_metrics = MET.all_metrics(valid_raw, y_valid_raw)
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss_standardized": float(last_train_loss),
                "train_mae_standardized": float(last_train_mae),
                "val_loss_standardized": val_loss,
                "val_rmse_raw": val_metrics["rmse"],
                "val_mae_raw": val_metrics["mae"],
                "val_r2_raw": val_metrics["r2"],
                "lr": lr_now,
                "seconds": time.perf_counter() - t0,
            }
        )
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - (best_epoch or 0) >= patience:
            stopped_early = True
            break
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    train_pred_std = NT._predict(
        model, train_data, kind="pna", device=device, batch_size=batch_size, precision=precision
    )
    train_pred = NT.denormalise(train_pred_std, target_norm)
    train_metrics = MET.all_metrics(train_pred, labels.y_train[: len(train_rows)])
    valid_pred_std = NT._predict(
        model, valid_data, kind="pna", device=device, batch_size=batch_size, precision=precision
    )
    valid_pred = NT.denormalise(valid_pred_std, target_norm)
    valid_metrics = MET.all_metrics(valid_pred, y_valid_raw)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        NT._predict(model, valid_data, kind="pna", device=device, batch_size=batch_size, precision=precision)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    if device.type == "cuda":
        gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": "PNA-small",
            "best_epoch": int(best_epoch),
            "val_loss_standardized": float(best_val),
            "selected_width": int(C.read_json(C.WIDTH_SEARCH)["selected_width"]),
        },
        checkpoint_path,
    )
    np.savez(run_dir / "valid_predictions.npz", rows=np.asarray(valid_rows, dtype=np.int64), pred=valid_pred)
    np.savez(run_dir / "train_predictions.npz", rows=np.asarray(train_rows, dtype=np.int64), pred=train_pred)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "model": "PNA-small",
        "label": label or "PNA-small",
        "seed": int(cfg["seed"]),
        "config": dict(cfg),
        "config_hash": C.config_hash(dict(cfg)),
        "env": {
            **C.env_info(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "precision": precision,
            "optimizer": "torch.optim.AdamW",
            "upstream_optimizer_requested": "bitsandbytes.optim.AdamW8bit",
        },
        "data": {
            "split_file": C.track_rel(C.V1_SPLIT_FILE),
            "split_file_sha256": C.V1_SPLIT_SHA256,
            "train_rows_sha256": C.rows_sha256_int64(train_rows),
            "valid_rows_sha256": C.rows_sha256_int64(valid_rows),
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "structures": C.verify_structures(),
            "num_features": dims["num_features"],
            "edge_dim": dims["edge_dim"],
            "target_norm": target_norm,
            "label_audit": dict(labels.audit),
            "test_labels_read": False,
            "test_structures_loaded": False,
        },
        "model_audit": {
            "parameters_total": params_total,
            "parameters_requires_grad": params_requires_grad,
            "parameters_effective": params_total,
            "class": "gnn.graph_models.Estimator(conv_type=PNA)",
            "width": int(C.read_json(C.WIDTH_SEARCH)["selected_width"]),
            "width_search": {
                "path": C.track_rel(C.WIDTH_SEARCH),
                "sha256": C.sha256_file(C.WIDTH_SEARCH),
                "selected_parameters_total": int(C.read_json(C.WIDTH_SEARCH)["selected_parameters_total"]),
                "relative_difference": float(C.read_json(C.WIDTH_SEARCH)["relative_difference"]),
                "tightly_size_matched": bool(C.read_json(C.WIDTH_SEARCH)["tightly_size_matched"]),
            },
            "extra": extra_audit,
            "gradient_coverage_one_batch": grad_coverage,
            "only_width_changed": True,
        },
        "training": {
            "epochs_planned": max_epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "batch_size": batch_size,
            "batch_order_sha256": NT.batch_order_hash(all_batches),
            "early_stopping": True,
            "early_stopping_patience": patience,
            "stopped_early": stopped_early,
            "checkpoint_rule": str(cfg["checkpoint"]),
            "best_epoch": int(best_epoch),
            "best_val_loss_standardized": float(best_val),
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=15)",
            "gradient_clip": float(cfg["gradient_clip"]),
            "loss": "mse_on_train3360_standardised_target" if limit_train is None else "mse_on_smoke_standardised_target",
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD", "checkpoint": "best_validation_loss"},
            "valid": {**valid_metrics, "units": "raw_logD", "checkpoint": "best_validation_loss"},
            "train_loss_standardized_last_epoch": float(last_train_loss) if last_train_loss is not None else None,
            "train_mae_standardized_last_epoch": float(last_train_mae) if last_train_mae is not None else None,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "inference_seconds_per_valid420": float(inference_seconds),
            "inference_batch_size": batch_size,
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
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
        },
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
        "test_predictions_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    C.log(
        f"[{payload['label']}] params={params_total} epochs={len(curve)} best_epoch={best_epoch} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["train_s2off", "train_pna_small", "_predict_lcrc", "state_dict_fingerprint"]
