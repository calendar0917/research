"""Formal training driver for the LCRC performance long-run.

One run = one dataset/split/seed/arm.  The recipe is the frozen R3 recipe, with
the single pre-registered exception for ``ogbg-molhiv`` (160-epoch cosine
horizon, minimum 60 epochs, early-stop patience 30).

Sequence, enforced in code:

1. build/load the label-free train+valid view (chem8 only for local-chemistry
   arms; readout arms see byte-identical base caches);
2. construct the arm with BASE-initialised shared tensors and zero-initialised
   extensions;
3. train on Train, select exactly one best Validation epoch;
4. restore that state and re-verify the Validation metric;
5. only then build/read the Test cache and score Test once;
6. write checkpoint, curve, predictions and a full provenance record.

Test never influences the checkpoint, the epoch, a hyper-parameter or a width.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import cl_common as C  # noqa: I001

import mb2_data as D
import mb2_metrics as MET
import mb2_train as T2
import lcrc_data as PREP_DATA

import cl_chem8 as CH
import cl_data as CD
import cl_model as CM


PROCESS_T0 = time.perf_counter()


def _recheck_tolerance(task: str, y_valid, task_names) -> tuple[float, str]:
    """Reproducibility bound for the best-epoch recheck (same logic as Sprint 03)."""
    yy = np.asarray(y_valid, dtype=np.float64)
    if task == "regression":
        return float(C.REPRO_TOL), "continuous metric; fixed float tolerance"
    if task == "binary":
        y = yy.reshape(-1)
        pos = int(np.count_nonzero(y == 1))
        neg = int(np.count_nonzero(y == 0))
        pairs = pos * neg
        step = 1.0 / pairs if pairs else 1.0
        return max(float(C.REPRO_TOL), 1.5 * step), f"one ROC-AUC discordant pair = 1/({pos}*{neg}) = {step:.3e}"
    y2 = yy.reshape(yy.shape[0], -1) if yy.ndim == 1 else yy
    steps = []
    for j in range(y2.shape[1]):
        col = y2[np.isfinite(y2[:, j]), j]
        pos = int(np.count_nonzero(col == 1))
        neg = int(np.count_nonzero(col == 0))
        if pos and neg:
            steps.append(1.0 / (pos * neg))
    step = max(steps) if steps else 1.0
    return max(float(C.REPRO_TOL), 1.5 * step), f"largest per-task ROC-AUC step = {step:.3e} over {len(steps)} tasks"


def _masked_cell_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(y)
    if int(mask.sum()) == 0:
        raise ValueError("no observed label in the batch")
    per_cell = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    return (per_cell * mask).sum() / mask.sum().clamp(min=1.0)


def resolve_loss(task: str) -> dict[str, Any]:
    if task == "regression":
        return {"name": "mse", "definition": "mean squared error on Train-standardised targets"}
    if task == "binary":
        return {"name": "bce", "definition": "binary_cross_entropy_with_logits, no pos_weight"}
    if task == "multitask":
        return {"name": "masked_cell", "definition": "sum_observed BCE / n_observed_cells (frozen Tox21 loss)"}
    raise ValueError(f"unknown task {task!r}")


def train_run(
    *,
    dataset: str,
    split: str,
    arm: str,
    seed: int,
    device: torch.device,
    out_dir,
    protocol: str = "default",
    scale: str = "base",
    test: bool = True,
    smoke: bool = False,
    epochs_override: int | None = None,
    run_key: str | None = None,
) -> dict[str, Any]:
    if not smoke and int(seed) not in (0, 1, 2):
        raise ValueError("formal runs use seed 0/1/2 only")
    protocol_record = None
    if not smoke:
        protocol_record = C.verify_protocol_freeze()
    recipe = C.recipe_for(protocol)
    epochs_planned = int(recipe["epochs"])
    if smoke:
        epochs = int(epochs_override or 2)
        if epochs > 2:
            raise ValueError("smoke runs may not exceed 2 epochs")
    else:
        if epochs_override is not None and int(epochs_override) != epochs_planned:
            raise ValueError(f"formal {protocol} runs must use {epochs_planned} epochs")
        epochs = epochs_planned
    min_epochs = int(recipe.get("minimum_epochs", 0))
    patience = int(recipe.get("early_stop_patience", 0))
    cfg = dict(C.SCALE_M_CONFIG) if scale == "M" else None
    loss_info = resolve_loss(C.dataset_entry(dataset)["task"])

    view, base_spec, uses_chem8 = CD.view_for_arm(dataset, split, arm)
    task = view.task
    spec = view.spec
    n_train = int(view.train_cache["n_molecules"])
    n_valid = int(view.valid_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if arm == "base":
        model, meta = CM.build_base_model(
            spec,
            view.train_norm,
            seed,
            device,
            cfg=cfg,
            dropout=float(recipe["dropout"]),
            n_tasks=view.targets.n_tasks,
        )
    else:
        model, meta = CM.build_variant_model(
            spec,
            base_spec,
            view.train_norm,
            seed,
            device,
            arm=arm,
            cfg=cfg,
            dropout=float(recipe["dropout"]),
            n_tasks=view.targets.n_tasks,
        )
    init_hash = str(meta["init_state_hash"])
    params_total = int(meta["parameters"])

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
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
    generator = torch.Generator().manual_seed(int(seed) + C.SHUFFLE_OFFSET)

    y_train_ref = view.targets.loss_reference(view.target_norm)
    y_valid = view.targets.y_valid
    if task == "regression":
        y_train_loss = torch.as_tensor(np.asarray(y_train_ref, dtype=np.float64).reshape(-1), dtype=torch.float32)
    elif task == "binary":
        flat = np.asarray(y_train_ref, dtype=np.float64).reshape(-1)
        if set(np.unique(flat).tolist()) != {0.0, 1.0}:
            raise ValueError(f"{dataset}: Train split must contain both classes")
        y_train_loss = torch.as_tensor(flat, dtype=torch.float32)
    else:
        y_train_loss = torch.as_tensor(np.asarray(y_train_ref, dtype=np.float64), dtype=torch.float32)

    tracker = T2.BestEpochTracker(task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
    skipped_batches = 0
    early_stopped = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        abs_sum = 0.0
        sq_sum = 0.0
        bce_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        epoch_skipped = 0
        loss_sum = 0.0
        loss_steps = 0
        for step_index, rows in enumerate(batches):
            batch = D.to_device(PREP_DATA.collate(view.train_cache, rows), device)
            y = y_train_loss[rows].to(device)
            pred = model(batch)
            if task == "regression":
                pred = pred.reshape(-1)
                step_loss = F.mse_loss(pred, y.reshape(-1))
                err = pred - y.reshape(-1)
                abs_sum += float(err.abs().sum())
                sq_sum += float((err * err).sum())
            elif task == "binary":
                pred = pred.reshape(-1)
                step_loss = F.binary_cross_entropy_with_logits(pred, y.reshape(-1))
                bce_sum += float(step_loss.detach()) * int(y.numel())
            else:
                y2 = y.reshape(y.shape[0], -1) if y.ndim == 1 else y
                mask = torch.isfinite(y2)
                if int(mask.sum()) == 0:
                    skipped_batches += 1
                    epoch_skipped += 1
                    continue
                step_loss = _masked_cell_loss(pred.reshape(y2.shape[0], -1), y2)
                bce_sum += float(step_loss.detach()) * int(mask.sum())
                seen += int(mask.sum())
            optimizer.zero_grad(set_to_none=True)
            step_loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(step_loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            optimizer.step()
            scheduler.step()
            loss_sum += float(step_loss.detach())
            loss_steps += 1
            if task != "multitask":
                seen += int(y.numel())
        train_seconds = time.perf_counter() - t0

        t_eval = time.perf_counter()
        valid_raw = D.predict_matrix(model, view.valid_cache, device)
        valid_metrics = MET.task_metrics(task, valid_raw, y_valid, view.target_norm, view.task_names)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, valid_metrics)
        if improved:
            tracker.capture(model)
        row: dict[str, Any] = {
            "epoch": epoch,
            "steps": steps_per_epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm_max": float(epoch_gnorm_max),
            "grad_clip_events": int(epoch_clips),
            "skipped_batches_no_observed_label": int(epoch_skipped),
            "train_seconds": train_seconds,
            "valid_seconds": eval_seconds,
            "is_best": bool(improved),
            "train_loss": float(loss_sum / max(loss_steps, 1)),
        }
        if task == "regression":
            train_mse = float(sq_sum / max(seen, 1))
            row["train_mse_standardized"] = train_mse
            row["train_rmse_standardized"] = math.sqrt(train_mse)
            row["train_mae_standardized"] = float(abs_sum / max(seen, 1))
            row["valid_rmse_raw"] = float(valid_metrics["rmse"])
            row["valid_mae_raw"] = float(valid_metrics["mae"])
            row["valid_r2_raw"] = float(valid_metrics["r2"])
        elif task == "binary":
            row["train_bce_logits"] = float(bce_sum / max(seen, 1))
            row["valid_roc_auc"] = float(valid_metrics["roc_auc"])
            row["valid_pr_auc"] = float(valid_metrics["pr_auc"])
            row["valid_bce"] = float(valid_metrics["bce"])
        else:
            row["train_masked_bce"] = float(bce_sum / max(seen, 1))
            row["valid_macro_roc_auc"] = float(valid_metrics["macro_roc_auc"])
            row["valid_macro_pr_auc"] = float(valid_metrics["macro_pr_auc"])
            row["valid_masked_bce"] = float(valid_metrics["masked_bce"])
            row["valid_definable_tasks"] = int(valid_metrics["n_definable_tasks"])
        curve.append(row)
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))
        if patience and epoch >= min_epochs and (epoch - int(tracker.best_epoch)) >= patience:
            early_stopped = True
            break

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any validation improvement")

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = CM.M.state_dict_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = CM.M.state_dict_fingerprint(best_state)
    checkpoint_is_final_epoch = bool(best_state_hash == final_state_hash)
    model.load_state_dict(tracker.to_device(device))
    model.eval()
    valid_raw = D.predict_matrix(model, view.valid_cache, device)
    valid_raw_repeat = D.predict_matrix(model, view.valid_cache, device)
    nondet_max = float(np.max(np.abs(valid_raw - valid_raw_repeat))) if valid_raw.size else 0.0
    valid_metrics = MET.task_metrics(task, valid_raw, y_valid, view.target_norm, view.task_names)
    recheck_key = MET.primary_name(task)
    recheck_diff = abs(float(valid_metrics[recheck_key]) - float(tracker.best_metrics[recheck_key]))
    recheck_tol, recheck_basis = _recheck_tolerance(task, y_valid, view.task_names)
    if recheck_diff > recheck_tol:
        raise RuntimeError(
            f"reloaded best checkpoint Validation {recheck_key} {valid_metrics[recheck_key]} != recorded "
            f"{tracker.best_metrics[recheck_key]} (diff {recheck_diff} > {recheck_tol}: {recheck_basis})"
        )
    primary_key = {"regression": "valid_rmse_raw", "binary": "valid_roc_auc", "multitask": "valid_macro_roc_auc"}[task]
    curve_values = np.asarray([row[primary_key] for row in curve], dtype=np.float64)
    argbest_epoch = int(np.argmin(curve_values) + 1) if task == "regression" else int(np.argmax(curve_values) + 1)
    if argbest_epoch != int(tracker.best_epoch):
        raise RuntimeError(f"curve argbest {argbest_epoch} != tracked best epoch {tracker.best_epoch}")

    train_raw = D.predict_matrix(model, view.train_cache, device)
    train_metrics = MET.task_metrics(task, train_raw, view.targets.y_train, view.target_norm, view.task_names)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        D.predict_matrix(model, view.valid_cache, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": meta["name"],
            "arm": arm,
            "model_config": dict(model.cfg),
            "scale": scale,
            "schema": spec.name,
            "feature_spec_hash": spec.hash(),
            "base_feature_spec_hash": base_spec.hash(),
            "dataset": dataset,
            "task": task,
            "split": split,
            "protocol": protocol,
            "loss": loss_info["name"],
            "n_outputs": int(view.targets.n_tasks),
            "seed": int(seed),
            "protocol_id": C.PROTOCOL_ID,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": recheck_key,
            "best_metric_value": float(tracker.best_primary),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch": checkpoint_is_final_epoch,
            "init_state_hash": init_hash,
            "parameters": params_total,
            "target_norm": None if view.target_norm is None else dict(view.target_norm),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "dropout": float(recipe["dropout"]),
            "weight_decay": float(recipe["weight_decay"]),
            "best_reload_check": {
                "metric": recheck_key,
                "reloaded": float(valid_metrics[recheck_key]),
                "recorded": float(tracker.best_metrics[recheck_key]),
                "abs_diff": float(recheck_diff),
                "tolerance": float(recheck_tol),
                "tolerance_basis": recheck_basis,
                "raw_nondeterminism_max_abs": nondet_max,
            },
            "epochs_run": len(curve),
            "curve_sha256": C.sha256_file(curve_path),
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    np.savez(
        out_dir / "train_predictions.npz",
        rows=np.asarray(view.rows["train"], dtype=np.int64),
        raw=np.asarray(train_raw, dtype=np.float64),
        y_true=np.asarray(view.targets.y_train, dtype=np.float64),
    )
    np.savez(
        out_dir / "valid_predictions.npz",
        rows=np.asarray(view.rows["valid"], dtype=np.int64),
        raw=np.asarray(valid_raw, dtype=np.float64),
        y_true=np.asarray(y_valid, dtype=np.float64),
    )

    test_payload: dict[str, Any] | None = None
    n_test = 0
    if test and not smoke:
        # The best-Validation checkpoint is already frozen above.  Only now is
        # the Test cache built and its label column decoded.
        test_cache, y_test, test_rows, _base_at_eval = CD.test_view(dataset, split)
        if uses_chem8:
            test_cache = CH.extend_cache(test_cache, spec, dataset=dataset)
        if C.sha256_file(checkpoint_path) == "":
            raise RuntimeError("checkpoint was not written")
        raw_test = D.predict_matrix(model, test_cache, device)
        test_metrics = MET.task_metrics(task, raw_test, y_test, view.target_norm, view.task_names)
        if task == "regression":
            pred_test = raw_test.reshape(-1) * float(view.target_norm["scale"]) + float(view.target_norm["mean"])
            np.savez(
                out_dir / "test_predictions.npz",
                rows=np.asarray(test_rows, dtype=np.int64),
                raw=np.asarray(raw_test, dtype=np.float64).reshape(-1),
                pred=np.asarray(pred_test, dtype=np.float64),
                y_true=np.asarray(y_test, dtype=np.float64).reshape(-1),
            )
        else:
            np.savez(
                out_dir / "test_predictions.npz",
                rows=np.asarray(test_rows, dtype=np.int64),
                raw=np.asarray(raw_test, dtype=np.float64),
                y_true=np.asarray(y_test, dtype=np.float64),
            )
        n_test = int(len(test_rows))
        test_payload = {
            "protocol_id": C.PROTOCOL_ID,
            "run_key": run_key,
            "dataset": dataset,
            "split": split,
            "arm": arm,
            "task": task,
            "metrics": test_metrics,
            "n_test": n_test,
            "test_rows": [int(x) for x in test_rows],
            "test_rows_sha256": C.row_sha(test_rows),
            "best_epoch": int(tracker.best_epoch),
            "valid_metrics": valid_metrics,
            "checkpoint_path": C.track_rel(checkpoint_path),
            "checkpoint_sha256": C.sha256_file(checkpoint_path),
            "test_predictions_sha256": C.sha256_file(out_dir / "test_predictions.npz"),
            "test_labels_read": True,
            "test_used_for_any_selection": False,
            "test_used_for_development": True,
            "exposure": "DEVELOPMENT_TEST (exploratory round; Test never selected the checkpoint)",
            "test_read_utc": C.utcnow(),
        }
        C.write_json(out_dir / "test_eval.json", test_payload)

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze": protocol_record,
        "run_key": run_key,
        "dataset": dataset,
        "task": task,
        "split": split,
        "arm": arm,
        "arm_label": C.ARM_LABEL[arm],
        "scale": scale,
        "protocol": protocol,
        "loss": loss_info,
        "seed": int(seed),
        "smoke": bool(smoke),
        "split_record": {
            "kind": "published" if split == "published" else split,
            "rows": {k: {"n": len(v), "sha256": C.row_sha(v)} for k, v in view.rows.items()},
        },
        "source": view.source_meta,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": "fp32",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": C.code_hashes(),
        "data": {
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": n_test,
            "feature_spec_name": spec.name,
            "feature_spec_hash": spec.hash(),
            "base_feature_spec_hash": base_spec.hash(),
            "chem8": uses_chem8,
            "train_cache_fingerprint": view.train_cache["meta"].get("content_fingerprint"),
            "train_cache_base_fingerprint": view.train_cache["meta"].get("base_content_fingerprint"),
            "valid_cache_fingerprint": view.valid_cache["meta"].get("content_fingerprint"),
            "valid_cache_base_fingerprint": view.valid_cache["meta"].get("base_content_fingerprint"),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": None if view.target_norm is None else dict(view.target_norm),
            "test_labels_read": bool(test_payload and test_payload["test_labels_read"]),
        },
        "model": {
            "name": meta["name"],
            "arm": arm,
            "schema": spec.name,
            "n_outputs": int(view.targets.n_tasks),
            "parameters": params_total,
            "parameters_breakdown": dict(meta["parameters_breakdown"]),
            "config": dict(meta["config"]),
            "input_widths": dict(meta["input_widths"]),
            "init_state_hash": init_hash,
            "shared_encoder_state_hash": meta["shared_encoder_state_hash"],
            "base_parameters": meta.get("base_parameters"),
            "extension_parameters": meta.get("extension_parameters"),
            "base_init_state_hash": meta.get("base_init_state_hash"),
            "base_copy": meta.get("base_copy"),
            "extension_zero_init": meta.get("extension_zero_init"),
            "iterative_hidden_message_passing": False,
            "has_attention": False,
            "dropout": float(recipe["dropout"]),
            "dropout_position": recipe["dropout_position"],
        },
        "recipe": {
            "optimizer": recipe["optimizer"],
            "adam_betas": [float(x) for x in recipe["adam_betas"]],
            "adam_eps": float(recipe["adam_eps"]),
            "peak_lr": float(recipe["peak_lr"]),
            "min_lr": float(recipe["min_lr"]),
            "weight_decay": float(recipe["weight_decay"]),
            "batch_size": batch_size,
            "dropout": float(recipe["dropout"]),
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "scheduler": recipe["scheduler"],
            "scheduler_horizon_epochs": int(recipe.get("scheduler_horizon_epochs", epochs)),
            "gradient_clip": float(recipe["gradient_clip"]),
            "loss": loss_info["name"],
            "checkpoint_rule": tracker.rule(),
            "minimum_epochs": min_epochs,
            "early_stop_patience": patience,
            "early_stopped": bool(early_stopped),
            "early_stopping": bool(patience),
            "validation_used_for_checkpoint_selection": True,
            "test_used_for_any_selection": False,
            "soup": False,
            "ema": False,
            "swa": False,
            "ssl": False,
            "hyperparameter_search": False,
            "shuffle_generator_seed": int(seed) + C.SHUFFLE_OFFSET,
        },
        "metrics": {
            "train": {**train_metrics, "checkpoint": "best_epoch"},
            "valid": {**valid_metrics, "checkpoint": "best_epoch"},
            "test": None if test_payload is None else {**test_payload["metrics"], "checkpoint": "best_epoch"},
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": recheck_key,
            "best_metric_value": float(tracker.best_primary),
            "epochs_run": len(curve),
            "n_improvements": len(tracker.updates),
            "improvement_epochs": list(tracker.updates),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch_state": checkpoint_is_final_epoch,
            "curve_argbest_epoch": argbest_epoch,
            "curve_best_value": float(curve_values.min() if task == "regression" else curve_values.max()),
            "valid_recheck_metric": recheck_key,
            "valid_recheck_abs_diff": float(recheck_diff),
            "selection_uses_test": False,
            "path": C.track_rel(checkpoint_path),
            "sha256": C.sha256_file(checkpoint_path),
            "curve_path": C.track_rel(curve_path),
            "curve_sha256": C.sha256_file(curve_path),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
            "skipped_batches_no_observed_label": int(skipped_batches),
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "mean_epoch_seconds": float(np.mean([row["train_seconds"] + row["valid_seconds"] for row in curve])),
            "inference_seconds_per_valid_split": float(inference_seconds),
            "inference_batch_size": 256,
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
            "device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        },
        "curve": curve,
        "predictions": {
            "train": C.track_rel(out_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(out_dir / "train_predictions.npz"),
            "valid": C.track_rel(out_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(out_dir / "valid_predictions.npz"),
        },
        "test_eval": None if test_payload is None else C.track_rel(out_dir / "test_eval.json"),
        "test_labels_read": bool(test_payload and test_payload["test_labels_read"]),
        "test_used_for_development": bool(test_payload),
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{dataset} {split} {arm} s{seed}] task={task} best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={MET.primary_value(task, train_metrics):.5f} valid={MET.primary_value(task, valid_metrics):.5f} "
        f"test={None if test_payload is None else round(MET.primary_value(task, test_payload['metrics']), 5)} "
        f"wall={wall:.1f}s"
    )
    return {"best_epoch": int(tracker.best_epoch), "wall": wall, "test": test_payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint-04 formal training run")
    parser.add_argument("--dataset", required=True, choices=list(C.V2C.DATASET_ORDER))
    parser.add_argument("--split", default="published")
    parser.add_argument("--arm", required=True, choices=list(C.ARMS))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--protocol", default="default", choices=["default", "hiv_short"])
    parser.add_argument("--scale", default="base", choices=["base", "M"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--key", default=None, help="analysis/runs/<key>")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
    elif args.epochs is not None:
        planned = int(C.recipe_for(args.protocol)["epochs"])
        if args.epochs != planned:
            raise SystemExit(f"formal runs must use {planned} epochs")

    if args.protocol == "hiv_short" and args.dataset != "ogbg-molhiv":
        raise SystemExit("hiv_short is registered for ogbg-molhiv only")
    if args.protocol != "hiv_short" and args.dataset == "ogbg-molhiv":
        raise SystemExit("ogbg-molhiv must use the pre-registered hiv_short protocol")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.key:
        out_dir = C.run_dir(args.key)
        run_key = args.key
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        run_key = None
    elif args.smoke:
        out_dir = C.RUNS / f"smoke_{args.dataset}_{args.arm}"
        run_key = None
    else:
        raise SystemExit("formal runs need --key or --out-dir")
    info = train_run(
        dataset=args.dataset,
        split=args.split,
        arm=args.arm,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        protocol=args.protocol,
        scale=args.scale,
        test=not args.no_test,
        smoke=bool(args.smoke),
        epochs_override=args.epochs,
        run_key=run_key,
    )
    C.log(f"wrote {C.track_rel(out_dir)} best_epoch={info['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
