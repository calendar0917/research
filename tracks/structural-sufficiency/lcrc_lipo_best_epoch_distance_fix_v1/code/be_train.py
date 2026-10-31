"""Best-single-epoch training loop (no soup, no EMA, no early stopping).

Identical to the frozen R3 recipe of the composition round except for the
checkpoint rule:

* every model is trained for the full 360 epochs;
* after every epoch the model is switched to ``eval()`` and the complete
  Validation420 RMSE is computed;
* the epoch state dict is kept (on CPU) only when its Validation RMSE is
  *strictly* lower than the best one seen so far, so ties keep the earlier
  epoch;
* at the end the best epoch state dict is restored, Validation420 is recomputed
  from scratch and must match the recorded value, and only then are the
  checkpoint, the curve and ``train.json`` written.

Nothing is averaged, nothing is selected on Test420 and the last epoch is never
used by default.
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

import be_common as C
import be_data as D
import be_models as M

import dc_train as DCT  # read-only diagnostics helpers  # noqa: E402
import lcrc_data  # noqa: E402
import r42_data as R42D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# best-epoch tracker
# ---------------------------------------------------------------------------


class BestEpochTracker:
    """Keeps the state dict of the epoch with the lowest Validation RMSE.

    Strict improvement only (``rmse < best - 0``), so an exact tie keeps the
    earlier epoch.  The state dict is cloned onto CPU at the moment of the
    improvement, so later training cannot mutate it.
    """

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


# ---------------------------------------------------------------------------
# lightweight diagnostics
# ---------------------------------------------------------------------------


def _grad_norms(model: torch.nn.Module, names: Sequence[str]) -> dict[str, float]:
    params = dict(model.named_parameters())
    return {
        name: float(params[name].grad.detach().norm())
        for name in names
        if params[name].grad is not None
    }


def gradient_probe(model, cache, names, device="cpu", steps: int = 2, lr: float | None = None) -> dict[str, Any]:
    if not names:
        return {"skipped": True, "reason": "family has no new parameters", "passed": True}
    lr = float(C.TRAIN_RECIPE["peak_lr"] if lr is None else lr)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    params = dict(model.named_parameters())
    before = {name: params[name].detach().clone() for name in names}
    history = []
    rows = list(range(min(16, int(cache["n_molecules"]))))
    for step in range(int(steps)):
        batch = M.to_device(D.collate(cache, rows), device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.mse_loss(pred, torch.zeros_like(pred))
        loss.backward()
        grads = {
            name: {
                "finite": bool(params[name].grad is not None and torch.isfinite(params[name].grad).all()),
                "norm": float(params[name].grad.detach().norm()) if params[name].grad is not None else 0.0,
            }
            for name in names
        }
        history.append({"step": step + 1, "loss": float(loss.detach()), "grads": grads})
        optimizer.step()
    delta = {name: float((params[name].detach() - before[name]).abs().max()) for name in names}
    passed = bool(
        all(g["finite"] for row in history for g in row["grads"].values())
        and all(history[-1]["grads"][name]["norm"] > 0.0 for name in names)
        and all(v > 0.0 for v in delta.values())
    )
    model.eval()
    return {
        "steps": int(steps),
        "history": history,
        "parameter_max_abs_update": delta,
        "all_updated": bool(all(v > 0.0 for v in delta.values())),
        "passed": passed,
    }


def correction_usage(model, family: str, cache, device="cpu", batch_size: int = 128) -> dict[str, Any]:
    """Magnitude/usage statistics of the learned correction over a whole cache."""
    model.eval()
    n = int(cache["n_molecules"])
    change_l2: list[float] = []
    graph_max: list[float] = []
    d34_nonempty = d5_nonempty = both_empty = 0
    n_distance_atoms = 0
    n_atoms_total = 0
    n_het_atoms = 0
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), n)))
            batch = M.to_device(D.collate(cache, rows), device)
            _y, aux = model(batch, return_aux=True)
            atom_batch = batch["atom_batch"].detach().cpu()
            n_atoms_total += int(atom_batch.numel())
            if "h_new" not in aux:
                continue  # BASE has no correction
            change = (aux["h_new"] - aux["h_original"]).detach().float().cpu().norm(dim=1).numpy()
            change_l2.extend(change.tolist())
            for g in range(len(rows)):
                mask = (atom_batch == g).numpy()
                graph_max.append(float(change[mask].max()) if mask.any() else 0.0)
            if "het_mask" in aux:
                n_het_atoms += int(aux["het_mask"].detach().cpu().sum())
            if "d34_counts" in aux:
                c34 = aux["d34_counts"].detach().cpu().numpy()
                c5 = aux["d5_counts"].detach().cpu().numpy()
                d34_nonempty += int((c34 > 0).sum())
                d5_nonempty += int((c5 > 0).sum())
                both_empty += int(((c34 == 0) & (c5 == 0)).sum())
                n_distance_atoms += int(c34.size)
    arr = np.asarray(change_l2, dtype=np.float64)
    out: dict[str, Any] = {
        "n_molecules": n,
        "n_atoms": n_atoms_total,
        "n_hetero_atoms": n_het_atoms,
        "change_l2_mean": float(arr.mean()) if arr.size else 0.0,
        "change_l2_p95": float(np.quantile(arr, 0.95)) if arr.size else 0.0,
        "change_l2_max": float(arr.max()) if arr.size else 0.0,
        "graph_max_change_mean": float(np.mean(graph_max)) if graph_max else 0.0,
        "fraction_atoms_with_nonzero_change": float(np.mean(arr > 0)) if arr.size else 0.0,
    }
    if n_distance_atoms:
        out.update(
            {
                "n_distance_atoms": n_distance_atoms,
                "fraction_d34_nonempty": float(d34_nonempty / n_distance_atoms),
                "fraction_d5_nonempty": float(d5_nonempty / n_distance_atoms),
                "fraction_both_empty": float(both_empty / n_distance_atoms),
            }
        )
    return out


def rng_state_sha256() -> str:
    state = torch.get_rng_state().numpy().tobytes()
    return C.sha256_bytes(state)


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
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    pool: str = "local",
    smoke: bool = False,
) -> dict[str, Any]:
    if arm not in C.ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}")
    spec_meta = C.ARM_SPECS[arm]
    family = str(spec_meta["family"])
    seed = int(spec_meta["seed"])
    recipe = dict(cfg["training"])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    labels = views["labels"]
    split = views["split"]
    if limit_train is not None:
        train_cache = D.build_distance_buckets(
            R42D.select_cache(
                train_cache,
                list(range(min(int(limit_train), int(train_cache["n_molecules"])))),
                split_label="smoke_train",
            )
        )
        views = {**views, "train_cache": train_cache}
    if limit_valid is not None:
        valid_cache = D.build_distance_buckets(
            R42D.select_cache(
                valid_cache,
                list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))),
                split_label="smoke_valid",
            )
        )
        views = {**views, "valid_cache": valid_cache}
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    if limit_train is not None:
        train_norm = lcrc_data.fit_norm(train_cache, spec, source="smoke_train_structures")
        target_norm = {"mean": float(np.mean(labels["y_train"][:n_train])), "scale": 1.0, "source": "smoke"}
    else:
        train_norm = views["train_norm"]
        target_norm = views["target_norm"]
    epochs = int(max_epochs if max_epochs is not None else recipe["epochs"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)

    model, meta = M.build_family(family, spec, train_norm, seed, device)
    init_hash = C.state_dict_fingerprint(model.state_dict())
    new_names = M.new_parameter_names(model, family)
    params_total = M.parameter_count(model)
    params_base = int(meta.get("parameters_base", params_total))

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
    y_train_std = NT.target_standardise(y_train_all, target_norm)
    y_valid = np.asarray(labels["y_valid"][:n_valid], dtype=np.float64)

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    probe_steps: list[dict[str, Any]] = []
    probe_agg = {name: {"min": math.inf, "max": -math.inf, "sum": 0.0, "n": 0} for name in new_names}
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
            batch = M.to_device(D.collate(train_cache, rows), device)
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
                norms = _grad_norms(model, new_names)
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
        train_seconds = time.perf_counter() - t0

        t_eval = time.perf_counter()
        valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
        valid_metrics = MET.all_metrics(valid_pred, y_valid)
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

    # Restore the recorded best epoch and re-verify Validation from scratch.
    model.load_state_dict(tracker.to_device(device))
    model.eval()
    valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
    valid_metrics = MET.all_metrics(valid_pred, y_valid)
    # fp32 forward passes are not bitwise reproducible on GPU, so the reload
    # check uses a 1e-6 tolerance on the raw-logD RMSE.
    valid_recheck_abs_diff = abs(float(valid_metrics["rmse"]) - float(tracker.best_rmse))
    if not valid_recheck_abs_diff <= 1.0e-6:
        raise RuntimeError(
            f"reloaded best checkpoint Validation RMSE {valid_metrics['rmse']} != recorded "
            f"{tracker.best_rmse} (diff {valid_recheck_abs_diff})"
        )

    curve_valid = np.asarray([row["valid_rmse_raw"] for row in curve], dtype=np.float64)
    curve_argmin_epoch = int(np.argmin(curve_valid)) + 1
    if curve_argmin_epoch != int(tracker.best_epoch):
        raise RuntimeError(
            f"curve argmin epoch {curve_argmin_epoch} != tracker best epoch {tracker.best_epoch}"
        )
    curve_best_rmse = float(curve_valid.min())
    if abs(curve_best_rmse - float(tracker.best_rmse)) > 1.0e-9:
        raise RuntimeError("tracker best RMSE differs from the curve minimum")

    train_pred = NT.denormalise(M.predict(model, train_cache, device), target_norm)
    train_metrics = MET.all_metrics(train_pred, y_train_all)

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

    # ------------------------------------------------------------------
    # diagnostics on the restored best epoch checkpoint
    # ------------------------------------------------------------------
    diagnostics: dict[str, Any] = {}
    if family == "base":
        diagnostics["rho_after_best_epoch"] = {
            "skipped": True,
            "reason": "BASE has no correction, so rho trivially reads the original h",
            "passed": True,
        }
    else:
        diagnostics["rho_after_best_epoch"] = DCT.rho_input_audit(
            model, valid_cache, list(range(min(8, n_valid)))
        )
    diagnostics["dropout"] = DCT.dropout_activity_audit(model, valid_cache, list(range(min(8, n_valid))))
    diagnostics["correction_usage_valid"] = correction_usage(model, family, valid_cache, device)
    diagnostics["correction_usage_train"] = correction_usage(model, family, train_cache, device)
    disabled_valid = NT.denormalise(
        M.predict_with_correction_disabled(model, valid_cache, family, device), target_norm
    )
    disabled_metrics = MET.all_metrics(disabled_valid, y_valid)
    diagnostics["w_out_zero_ablation_valid"] = {
        "rmse": float(disabled_metrics["rmse"]),
        "mae": float(disabled_metrics["mae"]),
        "r2": float(disabled_metrics["r2"]),
        "mean_abs_prediction_change_vs_full": float(np.abs(valid_pred - disabled_valid).mean()),
        "max_abs_prediction_change_vs_full": float(np.abs(valid_pred - disabled_valid).max()),
        "rmse_change_vs_full": float(disabled_metrics["rmse"] - valid_metrics["rmse"]),
        "note": "same best-epoch model with the correction output projection zeroed in eval",
    }
    diagnostics["gradient_probe"] = gradient_probe(model, valid_cache, new_names, device)
    if new_names:
        diagnostics["gradient_probe"]["training_probe"] = {
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
        }
    if family in M.DISTANCE_FAMILIES:
        diagnostics["distance_coverage_valid"] = D.distance_bucket_coverage(valid_cache)
        diagnostics["distance_coverage_train"] = D.distance_bucket_coverage(train_cache)

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    curve_path = run_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": C.MODEL_DISPLAY[family],
            "family": family,
            "arm": arm,
            "seed": seed,
            "run_key": arm,
            "protocol_id": C.PROTOCOL_ID,
            "checkpoint_rule": C.TRAIN_RECIPE["checkpoint"],
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
        "family": family,
        "model": C.MODEL_DISPLAY[family],
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
        "config_hash": C.sha256_json(dict(cfg)),
        "config_file_sha256": C.config_file_sha256(),
        "code_hashes": C.code_file_hashes(),
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
            "target_norm": dict(target_norm),
            "label_audit": dict(labels["audit"]),
            "distance_bucket_meta": dict(train_cache.get("distance_bucket_meta", {})),
            "test_row_values_used": False,
            "test_labels_read": False,
            "test_predictions_read": False,
            "test_metrics_read": False,
            "label_scope": "train3360_and_valid420_only",
        },
        "model_audit": {
            "parameters": int(params_total),
            "parameters_base": int(params_base),
            "parameters_new": int(params_total - params_base),
            "parameter_breakdown": M.parameter_breakdown(model),
            "init_state_hash": init_hash,
            "base_init_state_hash": meta.get("base_init_state_hash"),
            "shared_init_all_shared_bitwise_identical": bool(
                meta.get("shared_init", {}).get("all_shared_bitwise_identical", True)
            ),
            "dropout_hook_calls": int(meta["dropout_hook"].calls),
            "definition": meta.get("definition"),
        },
        "checkpoint": {
            "rule": C.TRAIN_RECIPE["checkpoint"],
            "best_epoch": int(tracker.best_epoch),
            "best_valid_rmse": float(tracker.best_rmse),
            "best_valid_mae": float(tracker.best_mae),
            "epochs_run": len(curve),
            "epoch_of_last_step": len(curve),
            "best_is_last_epoch": bool(int(tracker.best_epoch) == len(curve)),
            "checkpoint_is_final_epoch_state": checkpoint_is_final_epoch,
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "best_state_hash_equals_final_epoch_state": checkpoint_is_final_epoch,
            "curve_argmin_epoch": curve_argmin_epoch,
            "curve_min_valid_rmse": curve_best_rmse,
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
            "dropout_position": recipe["dropout_position"],
            "dropout_rng_reseeded_after_init": True,
            "loss": "mse_on_train3360_standardised_target",
            "checkpoint_rule": str(recipe["checkpoint"]),
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": True,
            "validation_used_for_early_stopping": False,
            "test_used_for_any_selection": False,
            "batch_order_sha256": NT.batch_order_hash(batch_order),
            "shuffle_generator_seed": seed + C.SHUFFLE_OFFSET,
            "torch_rng_state_sha256_after_training": rng_after_training,
            "seed": seed,
            "split_seed": C.SPLIT_SEED,
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD", "checkpoint": "best_epoch"},
            "valid": {**valid_metrics, "units": "raw_logD", "checkpoint": "best_epoch"},
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
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    C.write_json(run_dir / "diagnostics.json", diagnostics)
    C.log(
        f"[{arm}] family={family} seed={seed} best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = [
    "BestEpochTracker",
    "train_arm",
    "correction_usage",
    "gradient_probe",
    "PROCESS_T0",
]
