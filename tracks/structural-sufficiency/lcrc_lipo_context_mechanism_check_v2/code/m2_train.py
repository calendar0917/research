"""One shared training loop for BASE_s1, CONTEXT_s1 and ROLEBLIND_s0.

The loop is the frozen v1 recipe, reproduced step for step: AdamW, peak LR
3e-4 -> minimum LR 1e-5 with a per-step cosine schedule, weight decay 1e-3,
batch 128, 360 epochs, 27 steps per epoch (9,720 updates), gradient clipping
5.0, MSE on the Train3360-standardised target, and a checkpoint equal to the
average of the last five epoch-end state dicts.  No early stopping and no
Validation-based checkpoint selection.

BASE_s1 and CONTEXT_s1 share the seed-1 public initialisation, the exact same
batch order (a ``torch.Generator`` seeded with ``seed + SHUFFLE_OFFSET``) and
the same post-construction RNG reseed, so they also see the same dropout masks.
ROLEBLIND_s0 uses seed 0 and the same public initialisation as CONTEXT_s0.

Test data is never passed to this module.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import m2_common as C
import m2_data as MD
import m2_models as M

import lcrc_data  # noqa: E402
import r42_data as D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402
import x1_models as X1M  # noqa: E402

PROCESS_T0 = time.perf_counter()


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("empty soup")
    keys = list(states[0].keys())
    return {key: torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0) for key in keys}


def _grad_summary(model: torch.nn.Module, names: Sequence[str]) -> dict[str, float]:
    params = dict(model.named_parameters())
    return {name: float(params[name].grad.detach().norm()) for name in names if params[name].grad is not None}


def train_arm(
    *,
    arm_key: str,
    cfg: Mapping[str, Any],
    views: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    soup_last: int | None = None,
    pool: str = "local",
    smoke: bool = False,
) -> dict[str, Any]:
    if arm_key not in C.ARM_SPECS:
        raise ValueError(f"unknown arm {arm_key!r}")
    arm_spec = C.ARM_SPECS[arm_key]
    family = str(arm_spec["family"])
    seed = int(arm_spec["seed"])
    recipe = dict(cfg["training"])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    labels = views["labels"]
    split = views["split"]
    if limit_train is not None:
        train_cache = D.select_cache(
            train_cache,
            list(range(min(int(limit_train), int(train_cache["n_molecules"])))),
            split_label="smoke_train",
        )
    if limit_valid is not None:
        valid_cache = D.select_cache(
            valid_cache,
            list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))),
            split_label="smoke_valid",
        )
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    if limit_train is None:
        train_norm = views["train_norm"]
        target_norm = views["target_norm"]
    else:
        train_norm = lcrc_data.fit_norm(train_cache, spec, source="smoke_train_structures")
        target_norm = D.fit_target_scaler(labels["y_train"][:n_train])
    epochs = int(max_epochs if max_epochs is not None else recipe["epochs"])
    soup_k = int(soup_last if soup_last is not None else C.SOUP_LAST)
    if epochs <= 0 or soup_k <= 0:
        raise ValueError("epochs and soup_last must be positive")
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    model, meta = M.build_family(family, spec, train_norm, seed, device)
    init_hash = C.state_dict_fingerprint(model.state_dict())
    init_new_params = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if name.startswith("context.")
    }
    params_total = M.parameter_count(model)
    params_base = int(meta.get("parameters_base", params_total))
    new_names = M.new_parameter_names(model) if family in ("context", "roleblind") else []

    # Identical dropout masks across arms: reseed the global RNGs AFTER model
    # construction (CONTEXT/ROLEBLIND consume extra init draws, BASE does not).
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["min_lr"])
    )
    generator = torch.Generator().manual_seed(seed + C.SHUFFLE_OFFSET)
    y_train_all = np.asarray(labels["y_train"][:n_train], dtype=np.float64)
    y_train_std = NT.target_standardise(y_train_all, target_norm)
    y_valid = np.asarray(labels["y_valid"][:n_valid], dtype=np.float64)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    probe_steps: list[dict[str, Any]] = []
    probe_agg: dict[str, dict[str, float]] = {name: {"min": math.inf, "max": -math.inf, "sum": 0.0, "n": 0} for name in new_names}
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
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for step_index, rows in enumerate(batches):
            batch = M.to_device(lcrc_data.collate(train_cache, rows), device)
            y = y_train_std[rows].to(device)
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            if new_names:
                norms = _grad_summary(model, new_names)
                global_step = (epoch - 1) * steps_per_epoch + step_index + 1
                if global_step <= 5:
                    probe_steps.append({"global_step": global_step, "grad_norm": norms})
                for name, value in norms.items():
                    probe_agg[name]["min"] = min(probe_agg[name]["min"], value)
                    probe_agg[name]["max"] = max(probe_agg[name]["max"], value)
                    probe_agg[name]["sum"] += value
                    probe_agg[name]["n"] += 1
            optimizer.step()
            scheduler.step()
            err = (pred.view(-1) - y.view(-1)).detach()
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(err.numel())
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > soup_k:
            snapshots.pop(0)

        valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
        valid_metrics = MET.all_metrics(valid_pred, y_valid)
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss_standardized": float(sq_sum / max(seen, 1)),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "train_rmse_standardized": math.sqrt(float(sq_sum / max(seen, 1))),
                "valid_rmse_raw": valid_metrics["rmse"],
                "valid_mae_raw": valid_metrics["mae"],
                "valid_r2_raw": valid_metrics["r2"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "seconds": time.perf_counter() - t0,
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    soup_members = [int(e) for e in range(epochs - len(snapshots) + 1, epochs + 1)]
    soup = _average_states(snapshots)
    soup_hash = C.state_dict_fingerprint(soup)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    model.eval()

    train_pred = NT.denormalise(M.predict(model, train_cache, device), target_norm)
    train_metrics = MET.all_metrics(train_pred, y_train_all)
    valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
    valid_metrics = MET.all_metrics(valid_pred, y_valid)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        M.predict(model, valid_cache, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    if device.type == "cuda":
        gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": soup,
            "model": C.ARM_MODEL_NAME[family],
            "arm": arm_key,
            "family": family,
            "seed": seed,
            "run_key": arm_key,
            "protocol_id": C.PROTOCOL_ID,
            "soup_members": soup_members,
            "soup_state_hash": soup_hash,
            "init_state_hash": init_hash,
        },
        checkpoint_path,
    )
    np.savez(
        run_dir / "train_predictions.npz",
        rows=np.asarray(split["rows"]["train"][:n_train], dtype=np.int64),
        pred=train_pred,
        y_true=np.asarray(y_train_all, dtype=np.float64),
    )
    np.savez(
        run_dir / "valid_predictions.npz",
        rows=np.asarray(split["rows"]["valid"][:n_valid], dtype=np.int64),
        pred=valid_pred,
        y_true=np.asarray(y_valid, dtype=np.float64),
    )
    with open(run_dir / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    diagnostics: dict[str, Any] = {}
    context_usage: dict[str, Any] = {}
    if family in ("context", "roleblind"):
        context_usage["valid420"] = X1M.context_diagnostics(model, valid_cache, device)
        context_usage["train"] = X1M.context_diagnostics(model, train_cache, device)
        disabled_valid = NT.denormalise(X1M.predict_with_context_disabled(model, valid_cache, device), target_norm)
        disabled_metrics = MET.all_metrics(disabled_valid, y_valid)
        context_usage["w_out_zero_ablation_valid"] = {
            "rmse": disabled_metrics["rmse"],
            "mae": disabled_metrics["mae"],
            "r2": disabled_metrics["r2"],
            "mean_abs_prediction_change_vs_full_context": float(np.abs(valid_pred - disabled_valid).mean()),
            "max_abs_prediction_change_vs_full_context": float(np.abs(valid_pred - disabled_valid).max()),
            "rmse_change_vs_full_context": float(disabled_metrics["rmse"] - valid_metrics["rmse"]),
            "note": "full trained model vs the same trained model with W_out zeroed in eval (its own base path)",
        }
        context_usage["trained_new_parameter_delta_from_init"] = {
            name: float((param.detach().cpu() - init_new_params[name]).abs().max())
            for name, param in model.named_parameters()
            if name.startswith("context.")
        }
        if new_names:
            context_usage["gradient_probe"] = {
                "first_five_steps": probe_steps,
                "aggregate": {
                    name: {
                        "min": vals["min"],
                        "max": vals["max"],
                        "mean": vals["sum"] / max(vals["n"], 1),
                        "n_steps": vals["n"],
                        "all_nonzero": bool(vals["min"] > 0.0),
                    }
                    for name, vals in probe_agg.items()
                },
                "all_new_tensors_received_nonzero_gradient": bool(
                    all(vals["min"] > 0.0 for vals in probe_agg.values())
                ),
            }
        diagnostics["rho_after_soup"] = X1M.rho_input_audit(model, valid_cache, list(range(min(8, n_valid))))
        diagnostics["dropout"] = X1M.dropout_activity_audit(model, valid_cache, list(range(min(8, n_valid))))
        diagnostics["context_usage"] = context_usage
        if family == "roleblind":
            diagnostics["roleblind_usage"] = M.roleblind_usage(model, valid_cache, device)
            diagnostics["roleblind_aggregation_audit"] = M.roleblind_aggregation_audit(
                model, valid_cache, list(range(min(8, n_valid))), device
            )

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm_key,
        "run_key": arm_key,
        "family": family,
        "model": C.ARM_MODEL_NAME[family],
        "seed": seed,
        "split_seed": C.SPLIT_SEED,
        "smoke": bool(smoke),
        "pool": str(pool),
        "device": str(device),
        "device_note": "rr sets CUDA_VISIBLE_DEVICES per pool; the pool GPU is cuda:0 in-process",
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "precision": "fp32",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": __import__("platform").python_version(),
        "hostname": __import__("platform").node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(dict(cfg)),
        "data": {
            "split_file": C.track_rel(C.V1_SPLIT_FILE),
            "split_file_sha256": C.V1_SPLIT_SHA256,
            "labels_file": C.track_rel(C.V1_LABELS_FILE),
            "labels_file_sha256": C.V1_LABELS_SHA256,
            "train_rows_sha256": C.rows_sha256_int64(split["rows"]["train"][:n_train]),
            "valid_rows_sha256": C.rows_sha256_int64(split["rows"]["valid"][:n_valid]),
            "n_train": n_train,
            "n_valid": n_valid,
            "lcrc_cache_path": C.track_rel(views["cache_built"]["path"]),
            "lcrc_cache_sha256": views["cache_built"]["file_sha256"],
            "lcrc_cache_content_fingerprint": views["cache_built"]["content_fingerprint"],
            "lcrc_cache_labels_present": False,
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
            "train_norm_is_frozen_v1": limit_train is None,
            "target_norm": dict(target_norm),
            "label_audit": dict(labels["audit"]),
            "test_row_values_used": False,
            "test_labels_read": False,
            "test_predictions_read": False,
            "test_metrics_read": False,
            "label_scope": "train3360_and_valid420_only",
        },
        "model_audit": {
            "parameters": int(params_total),
            "parameters_base": int(params_base),
            "parameters_new": int(meta["parameters_new"]) if family in ("context", "roleblind") else 0,
            "parameter_breakdown": M.parameter_breakdown(model),
            "init_state_hash": init_hash,
            "soup_state_hash": soup_hash,
            "shared_init": meta.get("shared_init"),
            "roleblind_init_parity_with_context": meta.get("init_parity_with_context"),
            "dropout_hook_calls": int(meta["dropout_hook"].calls),
            "context_definition": meta.get("context_definition"),
        },
        "recipe": {
            "optimizer": "AdamW",
            "adam_betas": [float(x) for x in recipe["adam_betas"]],
            "adam_eps": float(recipe["adam_eps"]),
            "peak_lr": float(recipe["peak_lr"]),
            "min_lr": float(recipe["min_lr"]),
            "scheduler": "per_step_cosine",
            "weight_decay": float(recipe["weight_decay"]),
            "batch_size": batch_size,
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "gradient_clip": float(recipe["gradient_clip"]),
            "dropout": float(recipe["dropout"]),
            "dropout_position": "final atom-level local h (v2 DropoutAtLocal on model.local)",
            "dropout_rng_reseeded_after_init": True,
            "loss": "mse_on_train3360_standardised_target",
            "checkpoint_rule": str(recipe["checkpoint"]),
            "soup_last": soup_k,
            "soup_members": soup_members,
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": False,
            "batch_order_sha256": NT.batch_order_hash(batch_order),
            "seed": seed,
            "split_seed": C.SPLIT_SEED,
            "shuffle_generator_seed": seed + C.SHUFFLE_OFFSET,
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "valid": {**valid_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "train_loss_standardized_last_epoch": float(curve[-1]["train_loss_standardized"]),
            "train_mae_standardized_last_epoch": float(curve[-1]["train_mae_standardized"]),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "inference_seconds_per_valid420": float(inference_seconds),
            "inference_batch_size": 256,
            "inference_device": str(device),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
        },
        "diagnostics": diagnostics,
        "curve": curve,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
            "curve_path": C.track_rel(run_dir / "curve.csv"),
            "curve_sha256": C.sha256_file(run_dir / "curve.csv"),
        },
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    if diagnostics:
        C.write_json(run_dir / "diagnostics.json", diagnostics)
    C.log(
        f"[{arm_key}] family={family} seed={seed} params={params_total} epochs={len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["train_arm", "PROCESS_T0"]
