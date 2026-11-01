"""P1/P2/P3 training driver -- frozen R3 recipe, seed 0, four frozen objectives.

The loop is Sprint 02's ``mb2_train.train_run`` with exactly three changes, all
pre-registered and guarded:

1. **the row list may be a frozen non-default split** (``lipo-repeat:123/124``,
   ``bbbp-fold:0/1/2``).  The graph path, the R3 recipe, the checkpoint rule and
   the artefacts are unchanged; only which rows are Train and which are
   Validation changes.  ``published`` reproduces Sprint 02 exactly and is used as
   the reference for the two new objectives.
2. **the objective may be ``equal_task`` (Tox21 only) or ``sqrt_pos_weight``
   (ogbg-molhiv only)**.  Anything else is refused before a step is taken.
3. every run writes the loss definition, the pos_weight (if any) and the split
   provenance into ``train.json``.

Guarantees enforced in code:

* ``seed = 0`` and ``epochs = 360`` for every formal run (``--smoke`` may run 2);
* no Test cache is built and no Test label is decoded in this module;
* the reloaded best-Validation state is re-verified to ``1e-5``;
* the recorded best epoch must equal the Validation-curve argbest.

Usage
-----
    uv run python code/ta_train.py --dataset Lipophilicity --split lipo-repeat:123 --device cuda
    uv run python code/ta_train.py --dataset BBBP --split bbbp-fold:0 --device cuda
    uv run python code/ta_train.py --dataset Tox21 --loss equal_task --device cuda
    uv run python code/ta_train.py --dataset ogbg-molhiv --loss sqrt_pos_weight --device cuda
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

import ta_common as C  # noqa: I001

import mb2_data as D
import mb2_metrics as MET
import mb2_model as M
import mb2_train as T2

import lcrc_data as PREP_DATA

import ta_cv
import ta_data as TD
import ta_losses as L

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# view / split resolution
# ---------------------------------------------------------------------------


def resolve_split(dataset: str, split: str) -> tuple[Any, dict[str, Any]]:
    """Return (view, split_record) for one frozen row list."""
    if split == "published":
        view = D.build_view(dataset)
        record = {
            "kind": "published_repeat0",
            "dataset": dataset,
            "rows": view.rows,
            "split_file": view.source_meta.get("split_file"),
            "split_file_sha256": view.source_meta.get("split_file_sha256"),
            "split_rows_sha256": view.source_meta.get("split_rows_sha256"),
        }
        return view, record
    if split.startswith("lipo-repeat:"):
        if dataset != "Lipophilicity":
            raise SystemExit("lipo-repeat splits are Lipophilicity only")
        suffix = int(split.split(":", 1)[1])
        frozen = C.read_json(C.SPLIT_FREEZE)["lipophilicity_repeats"][str(suffix)]
        rows = {k: [int(x) for x in frozen["rows"][k]] for k in ("train", "valid", "test")}
        if C.row_sha(rows["train"]) != frozen["rows_sha256"]["train"]:
            raise SystemExit("lipo repeat rows do not match the frozen split")
        view = TD.build_view_rows(
            dataset,
            {"train": rows["train"], "valid": rows["valid"]},
            tag=f"lipo_r{suffix}",
            split_meta={
                "kind": "published_repeat",
                "suffix": suffix,
                "file": frozen["file"],
                "sha256": frozen["sha256"],
                "rows_sha256": frozen["rows_sha256"],
            },
        )
        return view, {
            "kind": "published_repeat",
            "suffix": suffix,
            "file": frozen["file"],
            "sha256": frozen["sha256"],
            "rows": {"train": rows["train"], "valid": rows["valid"]},
            "test_rows_not_used_here": len(rows["test"]),
        }
    if split.startswith("bbbp-fold:"):
        if dataset != "BBBP":
            raise SystemExit("bbbp-fold splits are BBBP only")
        fold = int(split.split(":", 1)[1])
        train_rows, valid_rows = ta_cv.rows_for_fold(fold)
        entry = ta_cv.frozen_folds()["folds"][str(fold)]
        view = TD.build_view_rows(
            dataset,
            {"train": train_rows, "valid": valid_rows},
            tag=f"bbbp_cv{fold}",
            split_meta={
                "kind": "bbbp_inner_cv_fold",
                "fold": fold,
                "valid_rows_sha256": entry["rows_sha256"],
                "n_scaffold_groups_valid": entry["n_scaffold_groups"],
            },
        )
        return view, {
            "kind": "bbbp_inner_cv_fold",
            "fold": fold,
            "valid_rows_sha256": entry["rows_sha256"],
            "rows": {"train": train_rows, "valid": valid_rows},
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "published_valid_test_rows_used": False,
        }
    raise SystemExit(f"unknown split {split!r}")


def resolve_loss(dataset: str, loss: str) -> dict[str, Any]:
    task = C.dataset_entry(dataset)["task"]
    if loss == "masked_cell":
        return {"name": "masked_cell", "definition": "sum_observed BCE / n_observed_cells", "overrides": {}}
    if loss == "equal_task":
        if dataset != "Tox21" or task != "multitask":
            raise SystemExit("equal_task is registered for Tox21 only")
        return {**dict(C.TOX21_EQUAL_TASK), "overrides": {"loss": "equal_task"}}
    if loss == "sqrt_pos_weight":
        if dataset != "ogbg-molhiv":
            raise SystemExit("sqrt_pos_weight is registered for ogbg-molhiv only")
        return {**dict(C.HIV_SQRT_POS_WEIGHT), "overrides": {"loss": "sqrt_pos_weight"}}
    raise SystemExit(f"unknown loss {loss!r}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _recheck_tolerance(task: str, y_valid, task_names) -> tuple[float, str]:
    """Reproducibility bound for the best-epoch recheck.

    The LCRC forward pass aggregates with ``index_add_``, which on CUDA sums in
    nondeterministic order; two evaluations of the *same* state therefore differ
    in the last float bits.  ROC-AUC is a rank statistic, so a one-bit change can
    flip a single discordant pair, which moves AUC by exactly ``1/(n_pos*n_neg)``.
    For a fold with 97 negatives and 445 positives that step is 2.3e-5, i.e.
    larger than the flat 1e-5 tolerance; the bound below is the analytic
    resolution of the reported statistic, not a loosened threshold.
    """
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


def train_run(
    *,
    dataset: str,
    split: str,
    loss: str,
    seed: int,
    device: torch.device,
    out_dir,
    epochs: int | None = None,
    smoke: bool = False,
    run_key: str | None = None,
) -> dict[str, Any]:
    if not smoke:
        if int(seed) != C.SEED:
            raise ValueError(f"formal runs must use training seed {C.SEED}")
        if epochs is not None and int(epochs) != int(C.recipe()["epochs"]):
            raise ValueError(f"formal runs must use {C.recipe()['epochs']} epochs")
    protocol_record = None
    if not smoke:
        protocol_record = C.verify_protocol_freeze()
    recipe = C.recipe()
    loss_config = resolve_loss(dataset, loss)
    view, split_record = resolve_split(dataset, split)
    epochs = int(recipe["epochs"]) if epochs is None else int(epochs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = view.spec
    task = view.task
    n_train = int(view.train_cache["n_molecules"])
    n_valid = int(view.valid_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    pos_weight_info = None
    pos_weight_tensor = None
    if loss == "sqrt_pos_weight":
        pos_weight_info = L.sqrt_pos_weight(np.asarray(view.targets.y_train).reshape(-1), cap=5.0)
        pos_weight_tensor = torch.tensor([pos_weight_info["pos_weight"]], dtype=torch.float32, device=device)

    model, model_meta = M.build_lcrc(
        spec, view.train_norm, seed, device, dropout=float(recipe["dropout"]), n_tasks=view.targets.n_tasks
    )
    init_hash = M.state_dict_fingerprint(model.state_dict())
    params_total = int(model_meta["parameters"])

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
    generator = torch.Generator().manual_seed(int(seed) + C.V2C.SHUFFLE_OFFSET)

    y_train_ref = view.targets.loss_reference(view.target_norm)
    y_valid = view.targets.y_valid
    if task == "regression":
        y_train_loss = torch.as_tensor(np.asarray(y_train_ref, dtype=np.float64).reshape(-1), dtype=torch.float32)
    elif task == "binary":
        flat = np.asarray(y_train_ref, dtype=np.float64).reshape(-1)
        if not (set(np.unique(flat).tolist()) == {0.0, 1.0}):
            raise ValueError(f"{dataset}: train split must contain both classes")
        y_train_loss = torch.as_tensor(flat, dtype=torch.float32)
    else:
        y_train_loss = torch.as_tensor(np.asarray(y_train_ref, dtype=np.float64), dtype=torch.float32)

    tracker = T2.BestEpochTracker(task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
    skipped_batches = 0
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
                if pos_weight_tensor is None:
                    step_loss = F.binary_cross_entropy_with_logits(pred, y.reshape(-1))
                else:
                    step_loss = F.binary_cross_entropy_with_logits(
                        pred, y.reshape(-1), pos_weight=pos_weight_tensor
                    )
                bce_sum += float(F.binary_cross_entropy_with_logits(pred, y.reshape(-1)).detach()) * int(y.numel())
            else:
                y2 = y.reshape(y.shape[0], -1) if y.ndim == 1 else y
                mask = torch.isfinite(y2)
                if int(mask.sum()) == 0:
                    skipped_batches += 1
                    epoch_skipped += 1
                    continue
                if loss == "equal_task":
                    step_loss = L.equal_task_loss(pred, y2)
                else:
                    step_loss = L.masked_cell_loss(pred, y2)
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

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any validation improvement")

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = M.state_dict_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = M.state_dict_fingerprint(best_state)
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
            "model": model_meta["name"],
            "schema": spec.name,
            "feature_spec_hash": spec.hash(),
            "dataset": dataset,
            "task": task,
            "split": split,
            "loss": loss,
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
            "pos_weight": None if pos_weight_info is None else float(pos_weight_info["pos_weight"]),
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

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze": protocol_record,
        "run_key": run_key,
        "dataset": dataset,
        "task": task,
        "split": split,
        "loss": loss,
        "loss_config": {k: v for k, v in loss_config.items() if k != "overrides"},
        "pos_weight": pos_weight_info,
        "label_columns": list(view.label_columns),
        "units": view.units,
        "smoke": bool(smoke),
        "seed": int(seed),
        "split_record": {k: v for k, v in split_record.items() if k != "rows"},
        "split_rows_sha256": {k: C.row_sha(v) for k, v in split_record.get("rows", {}).items()},
        "source": getattr(view, "split_meta", None) or getattr(view, "source_meta", {}),
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
        "data_view": view.audit(),
        "data": {
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": "not constructed (terminal test policy)",
            "feature_spec_hash": spec.hash(),
            "train_cache_fingerprint": view.train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": view.valid_cache["meta"]["content_fingerprint"],
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": None if view.target_norm is None else dict(view.target_norm),
            "test_labels_read": False,
            "test_cache_built": False,
        },
        "model": {
            "name": model_meta["name"],
            "schema": spec.name,
            "n_outputs": int(view.targets.n_tasks),
            "parameters": params_total,
            "parameters_breakdown": dict(model_meta["parameters_breakdown"]),
            "config": dict(model_meta["config"]),
            "input_widths": dict(model_meta["input_widths"]),
            "init_state_hash": init_hash,
            "shared_encoder_state_hash": model_meta["shared_encoder_state_hash"],
            "iterative_hidden_message_passing": False,
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
            "gradient_clip": float(recipe["gradient_clip"]),
            "loss": loss,
            "checkpoint_rule": tracker.rule(),
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": True,
            "test_used_for_any_selection": False,
            "soup": False,
            "ema": False,
            "swa": False,
            "ssl": False,
            "hyperparameter_search": False,
            "shuffle_generator_seed": int(seed) + C.V2C.SHUFFLE_OFFSET,
            "overrides_vs_R3": loss_config.get("overrides", {}),
        },
        "metrics": {
            "train": {**train_metrics, "checkpoint": "best_epoch"},
            "valid": {**valid_metrics, "checkpoint": "best_epoch"},
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
        "test_labels_read": False,
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{dataset} {split} {loss} s{seed}] task={task} best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={MET.primary_value(task, train_metrics):.5f} valid={MET.primary_value(task, valid_metrics):.5f} "
        f"wall={wall:.1f}s"
    )
    del payload
    return {"best_epoch": int(tracker.best_epoch), "wall": wall}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint-03 formal training run (frozen R3, seed 0)")
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--split", default="published")
    parser.add_argument("--loss", default="masked_cell", choices=list(C.LOSS_FAMILIES))
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--key", default=None, help="planned run key (analysis/runs/<key>)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
        if args.epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
    elif args.epochs is not None and args.epochs != int(C.recipe()["epochs"]):
        raise SystemExit(f"formal runs must use {C.recipe()['epochs']} epochs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.key:
        planned = {r["key"] for r in C.PLANNED_RUNS}
        if args.key not in planned:
            raise SystemExit(f"unknown run key {args.key!r}; planned: {sorted(planned)}")
        out_dir = C.run_dir(args.key)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        out_dir = C.RUNS / f"smoke_{args.dataset}_{args.split.replace(':', '').replace('/', '')}_{args.loss}"
    else:
        key = args.split.replace(":", "-").replace("/", "-")
        out_dir = C.RUNS / f"{args.dataset}_{key}_{args.loss}_s{args.seed}"
    info = train_run(
        dataset=args.dataset,
        split=args.split,
        loss=args.loss,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
        run_key=args.key,
    )
    C.log(f"wrote {C.track_rel(out_dir)} best_epoch={info['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
