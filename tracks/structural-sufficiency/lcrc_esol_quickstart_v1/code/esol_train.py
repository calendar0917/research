"""Best-single-Validation-epoch training loop for the ESOL arms.

The recipe is the validated R3 recipe of
``lcrc_lipo_best_epoch_distance_fix_v1`` (and of the two rounds after it),
unchanged:

* AdamW, peak LR 3e-4, min LR 1e-5, betas (0.9, 0.999), eps 1e-8, wd 1e-3;
* per-step cosine schedule over ``steps_per_epoch * epochs``;
* batch 128, gradient clip 5.0, MSE on the Train-standardised target;
* dropout on the final atom-level local ``h`` (0.10 BASE, 0.20 DROP20);
* full 360 epochs, no early stopping, no soup / EMA / SWA / SSL;
* checkpoint = the single epoch with the lowest Validation RMSE (strict
  improvement, so a tie keeps the earlier epoch);
* the best state is restored, Validation is recomputed from scratch and must
  match, and the curve argmin must equal the recorded best epoch.

Training sees ``views`` built with ``include_test=False``: no Test cache, no
Test label slice, and no Test metric anywhere in this module.
"""

from __future__ import annotations

import csv
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import esol_common as C
import esol_data as D
import esol_models as M

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# best-epoch tracker
# ---------------------------------------------------------------------------


class BestEpochTracker:
    """Keep the state dict of the lowest-Validation-RMSE epoch (strict)."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_rmse: float = math.inf
        self.best_mae: float | None = None
        self._state: dict[str, torch.Tensor] | None = None
        self.updates: list[int] = []

    def update(self, epoch: int, rmse: float, mae: float, model: torch.nn.Module) -> bool:
        rmse = float(rmse)
        if self.best_epoch is not None and not rmse < self.best_rmse:
            return False
        self.best_epoch = int(epoch)
        self.best_rmse = rmse
        self.best_mae = float(mae)
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.updates.append(int(epoch))
        return True

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("BestEpochTracker has no state")
        return self._state

    def to_device(self, device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in self.state.items()}


def target_standardise(y: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(y, dtype=np.float64) - float(norm["mean"])) / float(norm["scale"])


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


def batch_order_hash(batches) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(np.ascontiguousarray(np.asarray(batch, dtype=np.int64)).tobytes())
    return digest.hexdigest()


def rng_state_sha256() -> str:
    return C.sha256_bytes(torch.get_rng_state().numpy().tobytes())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _physchem_matrix(views: Mapping[str, Any], model_kind: str) -> np.ndarray | None:
    """Train-only standardised descriptor matrix, only for the PHYSCHEM6 arm."""
    if model_kind != "physchem6":
        return None
    raw = np.asarray(views["physchem"], dtype=np.float64)
    norm = views["physchem_norm"]
    mean = np.asarray(norm["mean"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    return ((raw - mean) / scale).astype(np.float32)


def _readout_stats(model, cache, device, physchem, y=None) -> dict[str, Any]:
    """Descriptive split of the prediction into local sum and global correction."""
    model.eval()
    rows = list(range(int(cache["n_molecules"])))
    contrib: list[np.ndarray] = []
    rho: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start : start + 256]
            batch = M.to_device(M.collate(cache, chunk, physchem=physchem), device)
            _y, aux = model(batch, return_aux=True)
            contrib.append(aux["contrib"].detach().float().cpu().numpy())
            rho.append(aux["rho"].detach().float().cpu().numpy())
    c = np.concatenate(contrib).astype(np.float64)
    r = np.concatenate(rho).astype(np.float64)
    y = np.zeros_like(c) if y is None else np.asarray(y, dtype=np.float64)
    out = {
        "contrib_mean": float(c.mean()),
        "contrib_std": float(c.std()),
        "rho_mean": float(r.mean()),
        "rho_std": float(r.std()),
        "corr_contrib_rho": float(np.corrcoef(c, r)[0, 1]) if c.size > 2 and c.std() > 0 and r.std() > 0 else 0.0,
        "mean_abs_rho": float(np.mean(np.abs(r))),
    }
    if y.size == c.size and bool(np.any(y)):
        out["corr_rho_y"] = float(np.corrcoef(r, y)[0, 1]) if r.std() > 0 else 0.0
        out["corr_contrib_y"] = float(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0.0
    return out


def curve_diagnostics(curve: list[dict[str, Any]], best_epoch: int, scale: float) -> dict[str, Any]:
    """Train/Validation trajectory descriptors used by the P1 verdict."""
    train_std = np.asarray([row["train_rmse_standardized"] for row in curve], dtype=np.float64)
    valid = np.asarray([row["valid_rmse_raw"] for row in curve], dtype=np.float64)
    best_index = int(best_epoch) - 1
    window = {}
    for offset in (0, 25, 50, 100):
        epoch = min(int(best_epoch) + offset, len(curve))
        window[f"valid_rmse_epoch_{epoch}"] = float(valid[epoch - 1])
    return {
        "best_epoch": int(best_epoch),
        "epochs_run": len(curve),
        "train_rmse_raw_at_best": float(train_std[best_index] * scale),
        "train_rmse_raw_first": float(train_std[0] * scale),
        "train_rmse_raw_last": float(train_std[-1] * scale),
        "valid_rmse_first": float(valid[0]),
        "valid_rmse_best": float(valid[best_index]),
        "valid_rmse_last": float(valid[-1]),
        "valid_rmse_last_minus_best": float(valid[-1] - valid[best_index]),
        "valid_rmse_best_plus_25_minus_best": float(window[f"valid_rmse_epoch_{min(int(best_epoch) + 25, len(curve))}"] - valid[best_index]),
        "gap_valid_minus_train_at_best": float(valid[best_index] - train_std[best_index] * scale),
        "train_still_improving_at_best": bool(best_index == 0 or train_std[best_index] < train_std[max(best_index - 10, 0)]),
        "final_epoch_train_rmse_raw": float(train_std[-1] * scale),
        "final_epoch_valid_rmse": float(valid[-1]),
        "frac_epochs_after_best": float((len(curve) - int(best_epoch)) / len(curve)),
        "valid_curve_windows": window,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_arm(
    *,
    arm: str,
    cfg: Mapping[str, Any],
    views: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    pool: str = "local",
    smoke: bool = False,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    model_kind, seed = C.arm_parts(arm)
    recipe = dict(cfg["training"])
    dropout = float(C.MODEL_DROPOUT[model_kind])
    if float(recipe["dropout"]) != 0.10:
        raise ValueError("the frozen recipe dropout must stay 0.10; arms differ by MODEL_DROPOUT")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    labels = views["labels"]
    split = views["split"]
    if limit_train is not None:
        train_cache = D.select_cache(train_cache, list(range(min(int(limit_train), int(train_cache["n_molecules"])))), split_label="smoke_train")
    if limit_valid is not None:
        valid_cache = D.select_cache(valid_cache, list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))), split_label="smoke_valid")
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])

    if limit_train is not None:
        import lcrc_data as PREP_DATA

        train_norm = PREP_DATA.fit_norm(train_cache, spec, source="smoke_train_structures")
        target_norm = {"mean": float(np.mean(labels["y_train"][:n_train])), "scale": 1.0, "source": "smoke"}
    else:
        train_norm = views["train_norm"]
        target_norm = views["target_norm"]

    physchem = _physchem_matrix(views, model_kind)
    model, meta = M.build(
        model_kind,
        spec,
        train_norm,
        seed,
        device,
        dropout=dropout,
        physchem_norm=views.get("physchem_norm"),
    )
    init_hash = C.state_dict_fingerprint(model.state_dict())
    params_total = M.parameter_count(model)
    params_base = int(meta.get("parameters_base", params_total))

    epochs = int(max_epochs if max_epochs is not None else recipe["epochs"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)

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
        optimizer, T_max=steps_per_epoch * epochs, eta_min=float(recipe["min_lr"])
    )
    generator = torch.Generator().manual_seed(seed + C.SHUFFLE_OFFSET)
    y_train_all = np.asarray(labels["y_train"][:n_train], dtype=np.float64)
    y_train_std = torch.as_tensor(target_standardise(y_train_all, target_norm), dtype=torch.float32)
    y_valid = np.asarray(labels["y_valid"][:n_valid], dtype=np.float64)

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    started = time.perf_counter()
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

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
            batch = M.to_device(M.collate(train_cache, rows, physchem=physchem), device)
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
            optimizer.step()
            scheduler.step()
            err = (pred.view(-1) - y.view(-1)).detach()
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(err.numel())
        train_seconds = time.perf_counter() - t0

        t_eval = time.perf_counter()
        valid_pred = denormalise(M.predict(model, valid_cache, device, physchem=physchem), target_norm)
        valid_metrics = C.all_metrics(valid_pred, y_valid)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.update(epoch, valid_metrics["rmse"], valid_metrics["mae"], model)
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss_standardized": float(sq_sum / max(seen, 1)),
                "train_rmse_standardized": math.sqrt(float(sq_sum / max(seen, 1))),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "valid_rmse_raw": float(valid_metrics["rmse"]),
                "valid_mae_raw": float(valid_metrics["mae"]),
                "valid_r2_raw": float(valid_metrics["r2"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "train_seconds": train_seconds,
                "valid_seconds": eval_seconds,
                "is_best": bool(improved),
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any validation epoch")
    rng_after_training = rng_state_sha256()
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = C.state_dict_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = C.state_dict_fingerprint(best_state)
    checkpoint_is_final_epoch = bool(best_state_hash == final_state_hash)

    model.load_state_dict(tracker.to_device(device))
    model.eval()
    valid_pred = denormalise(M.predict(model, valid_cache, device, physchem=physchem), target_norm)
    valid_metrics = C.all_metrics(valid_pred, y_valid)
    valid_recheck_abs_diff = abs(float(valid_metrics["rmse"]) - float(tracker.best_rmse))
    if not valid_recheck_abs_diff <= 1.0e-6:
        raise RuntimeError(
            f"reloaded best checkpoint Validation RMSE {valid_metrics['rmse']} != recorded "
            f"{tracker.best_rmse} (diff {valid_recheck_abs_diff})"
        )
    curve_valid = np.asarray([row["valid_rmse_raw"] for row in curve], dtype=np.float64)
    curve_argmin_epoch = int(np.argmin(curve_valid)) + 1
    if curve_argmin_epoch != int(tracker.best_epoch):
        raise RuntimeError(f"curve argmin {curve_argmin_epoch} != best epoch {tracker.best_epoch}")
    if abs(float(curve_valid.min()) - float(tracker.best_rmse)) > 1.0e-9:
        raise RuntimeError("tracker best RMSE differs from the curve minimum")

    train_pred = denormalise(M.predict(model, train_cache, device, physchem=physchem), target_norm)
    train_metrics = C.all_metrics(train_pred, y_train_all)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        M.predict(model, valid_cache, device, batch_size=256, physchem=physchem)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0

    diagnostics = {
        "curve": curve_diagnostics(curve, int(tracker.best_epoch), float(target_norm["scale"])),
        "readout_valid": _readout_stats(model, valid_cache, device, physchem, y=y_valid),
        "readout_train": _readout_stats(model, train_cache, device, physchem, y=y_train_all[:n_train]),
        "dropout": {
            "p": float(dropout),
            "position": recipe["dropout_position"],
            "hook_calls_last_forward": int(meta["dropout_hook"].calls),
        },
        "gradient_probe": _gradient_probe(model, valid_cache, device, physchem),
        "parameter_breakdown": M.parameter_breakdown(model),
    }

    curve_path = run_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": C.MODEL_DISPLAY[model_kind],
            "family": model_kind,
            "arm": arm,
            "seed": seed,
            "run_key": arm,
            "protocol_id": C.PROTOCOL_ID,
            "checkpoint_rule": recipe["checkpoint"],
            "dropout": float(dropout),
            "best_epoch": int(tracker.best_epoch),
            "best_valid_rmse": float(tracker.best_rmse),
            "best_valid_mae": float(tracker.best_mae),
            "epochs_run": len(curve),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch": checkpoint_is_final_epoch,
            "init_state_hash": init_hash,
            "curve_sha256": C.sha256_file(curve_path),
            "soup_state_hash": None,
            "soup_members": None,
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

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "run_key": arm,
        "family": model_kind,
        "model": C.MODEL_DISPLAY[model_kind],
        "seed": seed,
        "split_seed": int(C.SPLIT_SEED),
        "split_scope": str(views.get("split_scope", "esol_scaffold_seed_20261103")),
        "smoke": bool(smoke),
        "pool": str(pool),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "precision": "fp32",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": __import__("platform").python_version(),
        "hostname": __import__("platform").node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.sha256_json(dict(cfg)),
        "config_file_sha256": C.config_sha256(),
        "code_hashes": C.code_hashes(),
        "data": {
            "n_train": n_train,
            "n_valid": n_valid,
            "lcrc_cache_content_fingerprint": views["cache_built"]["content_fingerprint"],
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": views["train_norm_fingerprint"],
            "target_norm": dict(target_norm),
            "label_audit": dict(labels["audit"]),
            "test_row_values_used": False,
            "test_labels_read": False,
            "test_cache_built": False,
            "label_scope": "esol_train_and_validation_only",
        },
        "model_audit": {
            "parameters": int(params_total),
            "parameters_base": int(params_base),
            "parameters_new": int(params_total - params_base),
            "parameter_breakdown": M.parameter_breakdown(model),
            "init_state_hash": init_hash,
            "base_init_state_hash": meta.get("base_init_state_hash"),
            "shared_init": {
                k: v for k, v in dict(meta.get("shared_init", {"all_shared_bitwise_identical": True})).items() if k != "shared_keys"
            },
            "dropout": float(dropout),
            "definition": meta.get("definition"),
        },
        "checkpoint": {
            "rule": recipe["checkpoint"],
            "best_epoch": int(tracker.best_epoch),
            "best_valid_rmse": float(tracker.best_rmse),
            "best_valid_mae": float(tracker.best_mae),
            "epochs_run": len(curve),
            "best_is_last_epoch": bool(int(tracker.best_epoch) == len(curve)),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch_state": checkpoint_is_final_epoch,
            "curve_argmin_epoch": curve_argmin_epoch,
            "curve_min_valid_rmse": float(curve_valid.min()),
            "valid_rmse_recheck_after_reload": float(valid_metrics["rmse"]),
            "valid_rmse_recheck_abs_diff": float(valid_recheck_abs_diff),
            "selection_uses_test": False,
            "soup_used": False,
            "ema_used": False,
            "swa_used": False,
            "n_improvements": len(tracker.updates),
            "improvement_epochs": tracker.updates,
            "path": C.track_rel(checkpoint_path),
            "sha256": C.sha256_file(checkpoint_path),
            "curve_path": C.track_rel(curve_path),
            "curve_sha256": C.sha256_file(curve_path),
        },
        "recipe": {
            "optimizer": recipe["optimizer"],
            "adam_betas": [float(x) for x in recipe["adam_betas"]],
            "adam_eps": float(recipe["adam_eps"]),
            "peak_lr": float(recipe["peak_lr"]),
            "min_lr": float(recipe["min_lr"]),
            "scheduler": recipe["scheduler"],
            "weight_decay": float(recipe["weight_decay"]),
            "batch_size": batch_size,
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "gradient_clip": float(recipe["gradient_clip"]),
            "dropout": float(dropout),
            "dropout_position": recipe["dropout_position"],
            "loss": recipe["loss"],
            "checkpoint_rule": recipe["checkpoint"],
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": True,
            "validation_used_for_early_stopping": False,
            "test_used_for_any_selection": False,
            "batch_order_sha256": batch_order_hash(batch_order),
            "shuffle_generator_seed": seed + C.SHUFFLE_OFFSET,
            "torch_rng_state_sha256_after_training": rng_after_training,
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logS", "checkpoint": "best_epoch"},
            "valid": {**valid_metrics, "units": "raw_logS", "checkpoint": "best_epoch"},
            "train_loss_standardized_last_epoch": float(curve[-1]["train_loss_standardized"]),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "inference_seconds_per_valid_split": float(inference_seconds),
            "mean_epoch_seconds": float(np.mean([row["train_seconds"] + row["valid_seconds"] for row in curve])),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
        },
        "diagnostics": diagnostics,
        "curve": curve,
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
        },
        "label_scope": "esol_train_and_validation_only",
        "test_labels_read": False,
    }
    payload["data"].update(dict(views["data_override"]))
    payload.update(dict(views["protocol_override"]))
    C.write_json(run_dir / "train.json", payload)
    C.write_json(run_dir / "diagnostics.json", diagnostics)
    C.log(
        f"[{arm}] family={model_kind} seed={seed} best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


def _gradient_probe(model, valid_cache, device, physchem, steps: int = 2) -> dict[str, Any]:
    """Finite backward pass with every parameter receiving a gradient."""
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-4, weight_decay=0.0)
    rows = list(range(min(16, int(valid_cache["n_molecules"]))))
    batch = M.to_device(M.collate(valid_cache, rows, physchem=physchem), device)
    history = []
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.mse_loss(pred, torch.zeros_like(pred))
        loss.backward()
        grads = {
            name: float(param.grad.detach().norm()) if param.grad is not None else None
            for name, param in model.named_parameters()
        }
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "loss_finite": bool(np.isfinite(float(loss.detach()))),
                "n_params_with_grad": int(sum(1 for v in grads.values() if v is not None)),
                "n_params_with_nonzero_grad": int(sum(1 for v in grads.values() if v)),
                "all_grads_finite": bool(
                    all(param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in model.parameters())
                ),
            }
        )
        optimizer.step()
    model.eval()
    return {
        "steps": int(steps),
        "history": history,
        "passed": bool(
            all(row["loss_finite"] and row["all_grads_finite"] and row["n_params_with_nonzero_grad"] == row["n_params_with_grad"] for row in history)
        ),
    }


__all__ = [
    "BestEpochTracker",
    "target_standardise",
    "denormalise",
    "batch_order_hash",
    "curve_diagnostics",
    "train_arm",
    "PROCESS_T0",
]
