"""Formal PNA-native / ESA-native training loops.

Both networks are the vendored upstream classes (``gnn.graph_models`` and
``esa.models``) instantiated with the frozen configuration.  Only the training
driver is local: it replaces the upstream Lightning Trainer so that

* Validation420 is the only monitor (Test is never loaded or evaluated);
* the learning-rate schedule and early-stopping rule are the frozen ones;
* the reported metrics are denormalised to raw logD before comparison.

Upstream's mixed-precision setting (``precision="bf16"``) is kept for formal GPU
runs; CPU gates run in fp32.  The optimiser is ``torch.optim.AdamW`` because
bitsandbytes is unavailable (recorded in the implementation manifest).
"""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_metrics as MET
import r42_upstream as U


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def target_standardise(y: np.ndarray, norm: Mapping[str, Any]) -> torch.Tensor:
    return torch.as_tensor((np.asarray(y, dtype=np.float64) - float(norm["mean"])) / float(norm["scale"]), dtype=torch.float32)


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


def batch_order_hash(batches: Sequence[Sequence[int]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for batch in batches:
        h.update(np.ascontiguousarray(np.asarray(batch, dtype=np.int64)).tobytes())
    return h.hexdigest()


def _autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def pna_degrees(train_data: Sequence[Any], out_dir: Path) -> dict[str, Any]:
    """Train3360-only PNA degree histogram, written in the upstream layout."""
    from torch_geometric.utils import degree

    deg = torch.zeros(5000, dtype=torch.long)
    for data in train_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    path = Path(out_dir) / "saved_PNA_degrees" / "degrees.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deg, path)
    nonzero = [(int(i), int(v)) for i, v in enumerate(deg.tolist()) if v]
    n_atoms = int(sum(int(d.num_nodes) for d in train_data))
    total_degree = int(sum(int(d.edge_index.shape[1]) for d in train_data))
    if int(deg.sum()) != n_atoms:
        raise RuntimeError("PNA degree histogram does not cover every training atom")
    if sum(i * v for i, v in nonzero) != total_degree:
        raise RuntimeError("PNA degree histogram total degree does not match the training edge count")
    return {
        "source": "train3360_graphs_only",
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "length": int(deg.numel()),
        "nonzero": nonzero,
        "n_atoms": n_atoms,
        "total_degree": total_degree,
        "valid_or_test_used": False,
        "histogram_sha256": C.sha256_bytes(np.ascontiguousarray(deg.numpy()).tobytes()),
    }


def build_pna(cfg: Mapping[str, Any], device: torch.device, run_dir: Path, train_data: Sequence[Any], num_features: int, edge_dim: int):
    mods = U.install()
    degree_info = pna_degrees(train_data, run_dir)
    torch.manual_seed(int(cfg["seed"]))
    model = mods["graph_models"].Estimator(
        task_type="regression",
        num_features=int(num_features),
        gnn_intermediate_dim=int(cfg["gnn_intermediate_dim"]),
        output_node_dim=int(cfg["output_node_dim"]),
        num_layers=int(cfg["num_layers"]),
        conv_type=str(cfg["conv_type"]),
        edge_dim=int(edge_dim),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        linear_output_size=1,
        scaler=None,
        monitor_loss_name="val_loss",
        out_path=str(run_dir),
        train_dataset_for_PNA=list(train_data),
        regression_loss_fn="mse",
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        optimiser_weight_decay=float(cfg["weight_decay"]),
        use_cpu=(device.type == "cpu"),
    )
    # The constructor loads the histogram file written above; certify that the
    # loaded tensor is the Train3360 histogram and nothing else.
    loaded = torch.load(Path(run_dir) / "saved_PNA_degrees" / "degrees.pt", map_location="cpu", weights_only=False)
    expected_total = int(sum(v for _, v in degree_info["nonzero"]))
    if int(loaded.numel()) != int(degree_info["length"]) or int(loaded.sum()) != expected_total:
        raise RuntimeError("PNA constructor did not load the Train3360 degree histogram")
    degree_info["loaded_by_model_verified"] = True
    return model.to(device), degree_info


def build_esa(cfg: Mapping[str, Any], device: torch.device, run_dir: Path, max_edge_global: int, num_features: int, edge_dim: int):
    mods = U.install()
    torch.manual_seed(int(cfg["seed"]))
    model = mods["esa_models"].Estimator(
        task_type="regression",
        num_features=int(num_features),
        graph_dim=int(cfg["graph_dim"]),
        edge_dim=int(edge_dim),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        linear_output_size=1,
        scaler=None,
        monitor_loss_name="val_loss",
        xformers_or_torch_attn=str(cfg["attention_backend"]),
        hidden_dims=list(cfg["hidden_dims"]),
        num_heads=list(cfg["num_heads"]),
        num_sabs=None,
        sab_dropout=float(cfg["sab_dropout"]),
        mab_dropout=float(cfg["mab_dropout"]),
        pma_dropout=float(cfg["pma_dropout"]),
        apply_attention_on=str(cfg["apply_attention_on"]),
        layer_types=list(cfg["layer_types"]),
        use_mlps=bool(cfg["use_mlps"]),
        set_max_items=int(max_edge_global),
        regression_loss_fn="mse",
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        optimiser_weight_decay=float(cfg["weight_decay"]),
        mlp_hidden_size=int(cfg["mlp_hidden_size"]),
        mlp_type=str(cfg["mlp_type"]),
        attn_residual_dropout=float(cfg["attn_residual_dropout"]),
        norm_type=str(cfg["norm_type"]),
        triu_attn_mask=False,
        output_save_dir=str(run_dir),
        use_bfloat16=(str(cfg["precision"]) == "bf16"),
        is_node_task=False,
        posenc=[],
        num_mlp_layers=int(cfg["mlp_layers"]),
        pre_or_post=str(cfg["pre_or_post"]),
        pma_residual_dropout=float(cfg["pma_residual_dropout"]),
        use_mlp_ln=bool(cfg["use_mlp_ln"]),
        mlp_dropout=float(cfg["mlp_dropout"]),
    )
    num_max_items = int(mods["esa_models"].nearest_multiple_of_8(int(max_edge_global) + 1))
    return model.to(device), {
        "set_max_items": int(max_edge_global),
        "num_max_items": num_max_items,
        "attention_backend": str(cfg["attention_backend"]),
        "precision": str(cfg["precision"]),
    }


def _predict(
    model: torch.nn.Module,
    data_list: Sequence[Any],
    *,
    kind: str,
    device: torch.device,
    batch_size: int,
    precision: str,
    num_max_items: int | None = None,
) -> np.ndarray:
    from torch_geometric.data import Batch

    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data_list), int(batch_size)):
            chunk = data_list[start : start + int(batch_size)]
            batch = Batch.from_data_list(chunk).to(device)
            with _autocast(device, precision):
                if kind == "pna":
                    _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                elif kind == "esa":
                    pred = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr.float(), int(num_max_items), batch)
                else:
                    raise ValueError(kind)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out).astype(np.float64)


def train_native(
    *,
    kind: str,
    cfg: Mapping[str, Any],
    split: Mapping[str, Any],
    labels: L.TrainValidLabels,
    device: torch.device,
    run_dir: Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    precision_override: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    labels = L.assert_train_valid_only(labels)
    if kind not in ("pna", "esa"):
        raise ValueError(kind)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    C.ensure_dir(C.ANALYSIS)
    precision = precision_override or str(cfg["precision"])
    max_epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    train_rows = [int(r) for r in split["rows"]["train"]]
    valid_rows = [int(r) for r in split["rows"]["valid"]]
    if limit_train is not None:
        train_rows = train_rows[: int(limit_train)]
    if limit_valid is not None:
        valid_rows = valid_rows[: int(limit_valid)]
    data_all = D.load_structure_data()
    train_data = [data_all[r] for r in train_rows]
    valid_data = [data_all[r] for r in valid_rows]
    arrays = D.load_structure_arrays()
    num_features = int(arrays["atom_feat"].shape[1])
    edge_dim = int(arrays["edge_attr"].shape[1])
    max_edge_global = int(arrays["max_edge"].max())

    target_norm = D.fit_target_scaler(labels.y_train)
    y_train_std = target_standardise(labels.y_train[: len(train_rows)] if limit_train is not None else labels.y_train, target_norm)
    y_valid_std = target_standardise(labels.y_valid[: len(valid_rows)] if limit_valid is not None else labels.y_valid, target_norm)
    if y_train_std.numel() != len(train_data) or y_valid_std.numel() != len(valid_data):
        raise ValueError("label/structure length mismatch")

    if kind == "pna":
        model, extra_audit = build_pna(cfg, device, run_dir, train_data, num_features, edge_dim)
        num_max_items = None
    else:
        model, extra_audit = build_esa(cfg, device, run_dir, max_edge_global, num_features, edge_dim)
        mods = U.install()
        num_max_items = int(mods["esa_models"].nearest_multiple_of_8(max_edge_global + 1))
        extra_audit["num_max_items"] = num_max_items
    params_total = int(sum(p.numel() for p in model.parameters()))

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
    epochs_planned = max_epochs
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
    stopped_early = False
    last_train_loss = None
    last_train_mae = None
    clip_events = 0
    grad_norm_max = 0.0

    from torch_geometric.data import Batch

    for epoch in range(1, epochs_planned + 1):
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
            with _autocast(device, precision):
                if kind == "pna":
                    _, _, pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                else:
                    pred = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr.float(), int(num_max_items), batch)
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

        valid_std = _predict(
            model, valid_data, kind=kind, device=device, batch_size=batch_size, precision=precision, num_max_items=num_max_items
        )
        valid_raw = denormalise(valid_std, target_norm)
        y_valid_raw = labels.y_valid[: len(valid_rows)] if limit_valid is not None else labels.y_valid
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

    train_pred_std = _predict(model, train_data, kind=kind, device=device, batch_size=batch_size, precision=precision, num_max_items=num_max_items)
    train_pred = denormalise(train_pred_std, target_norm)
    train_metrics = MET.all_metrics(train_pred, labels.y_train[: len(train_rows)] if limit_train is not None else labels.y_train)
    valid_pred_std = _predict(model, valid_data, kind=kind, device=device, batch_size=batch_size, precision=precision, num_max_items=num_max_items)
    valid_pred = denormalise(valid_pred_std, target_norm)
    y_valid_raw = labels.y_valid[: len(valid_rows)] if limit_valid is not None else labels.y_valid
    valid_metrics = MET.all_metrics(valid_pred, y_valid_raw)

    # Inference timing: three repeats over Validation420 at the frozen batch size.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        _predict(model, valid_data, kind=kind, device=device, batch_size=batch_size, precision=precision, num_max_items=num_max_items)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    if device.type == "cuda":
        gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "model": kind, "best_epoch": int(best_epoch), "val_loss_standardized": float(best_val)}, checkpoint_path)
    np.savez(run_dir / "valid_predictions.npz", rows=np.asarray(valid_rows, dtype=np.int64), pred=valid_pred)
    np.savez(run_dir / "train_predictions.npz", rows=np.asarray(train_rows, dtype=np.int64), pred=train_pred)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "model": kind,
        "label": label or kind,
        "seed": int(cfg["seed"]),
        "config": dict(cfg),
        "config_hash": C.config_hash(dict(cfg)),
        "upstream_commit": C.UPSTREAM_COMMIT,
        "env": {
            **C.env_info(),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "precision": precision,
            "attention_backend": str(cfg.get("attention_backend", "")) if kind == "esa" else "",
            "optimizer": "torch.optim.AdamW",
            "upstream_optimizer_requested": "bitsandbytes.optim.AdamW8bit",
        },
        "data": {
            "split_file_sha256": C.sha256_file(C.SPLIT_FILE),
            "train_rows_sha256": C.rows_sha256_int64(train_rows),
            "valid_rows_sha256": C.rows_sha256_int64(valid_rows),
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "structures_sha256": D.structure_summary()["sha256"],
            "num_features": num_features,
            "edge_dim": edge_dim,
            "target_norm": target_norm,
            "label_audit": dict(labels.audit),
            "test_labels_read": False,
        },
        "model_audit": {
            "parameters_total": params_total,
            "extra": extra_audit,
            "class": "gnn.graph_models.Estimator(conv_type=PNA)" if kind == "pna" else "esa.models.Estimator",
        },
        "training": {
            "epochs_planned": epochs_planned,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "batch_size": batch_size,
            "batch_order_sha256": batch_order_hash(all_batches),
            "early_stopping": True,
            "early_stopping_patience": patience,
            "stopped_early": stopped_early,
            "checkpoint_rule": str(cfg["checkpoint"]),
            "best_epoch": int(best_epoch),
            "best_val_loss_standardized": float(best_val),
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=15)",
            "gradient_clip": float(cfg["gradient_clip"]),
            "loss": "mse_on_train3360_standardised_target",
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
    }
    C.write_json(run_dir / "train.json", payload)
    C.log(
        f"[{payload['label']}] params={params_total} epochs={len(curve)} best_epoch={best_epoch} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["train_native", "build_pna", "build_esa", "pna_degrees", "set_seed", "denormalise", "target_standardise", "batch_order_hash"]
