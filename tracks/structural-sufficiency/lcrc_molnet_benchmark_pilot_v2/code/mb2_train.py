"""P1 training driver: frozen R3 recipe, training seed 0, three task families.

The loop is Sprint 01's (``mb_train.py``) with two additions and no other change:

* **multitask** (Tox21): 12 logits, ``BCEWithLogitsLoss(reduction="none")`` with an
  observed-label mask, ``loss = (per_label * mask).sum() / mask.sum()``; a batch
  with no observed label is skipped and counted.  Checkpoint rule: highest
  Validation macro ROC-AUC, tie -> lower masked BCE, then earlier epoch.
* **probe overrides** (P3): ``dropout`` / ``weight_decay`` may be overridden, but
  only for a name in ``C.PROBE_CANDIDATES`` and only with exactly the pre-registered
  value.  Anything else raises before a step is taken.

Guarantees enforced in code, not just documented:

* formal runs use training seed 0 and exactly 360 epochs;
* the module never constructs a Test cache and never decodes a Test label;
* the single best-Validation state is reloaded and re-verified before it is written;
* the recorded best epoch must equal the Validation-curve argbest.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import mb2_common as C
import mb2_data as D
import mb2_metrics as MET
import mb2_model as M

import lcrc_data as PREP_DATA

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# best-epoch tracker
# ---------------------------------------------------------------------------


class BestEpochTracker:
    """Keep the single best-Validation epoch state (strict improvement)."""

    def __init__(self, task: str):
        self.task = task
        self.best_epoch: int | None = None
        self.best_primary: float | None = None
        self.best_secondary: float | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.updates: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, Any]) -> bool:
        primary = MET.primary_value(self.task, metrics)
        secondary = float(metrics.get("masked_bce", float("nan"))) if self.task == "multitask" else float("nan")
        if self.best_epoch is None:
            better = True
        elif self.task == "multitask" and primary == self.best_primary:
            better = secondary < float(self.best_secondary)
        else:
            better = MET.better(self.task, primary, self.best_primary)
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = float(primary)
        self.best_secondary = secondary
        self.best_metrics = dict(metrics)
        self.updates.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state

    def to_device(self, device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in self.state.items()}

    def rule(self) -> str:
        return {
            "regression": "lowest_validation_rmse__strict_improvement__tie_keeps_earlier",
            "binary": "highest_validation_roc_auc__tie_lower_validation_bce__then_earlier_epoch",
            "multitask": "highest_validation_macro_roc_auc__tie_lower_masked_bce__then_earlier_epoch",
        }[self.task]


# ---------------------------------------------------------------------------
# recipe resolution
# ---------------------------------------------------------------------------


def resolve_recipe(dataset: str, probe: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (recipe, probe_record).  Refuses any non-allowlisted deviation."""
    if probe is None:
        return dict(C.RECIPE), {"probe": False, "name": None, "overrides": {}, "effective_differences_vs_R3": {}}
    if probe not in C.PROBE_CANDIDATES:
        raise SystemExit(f"unknown probe {probe!r}; allowed: {list(C.PROBE_ORDER)}")
    spec = C.PROBE_CANDIDATES[probe]
    if spec["dataset"] != dataset:
        raise SystemExit(f"probe {probe!r} is registered for {spec['dataset']}, not {dataset}")
    recipe = C.recipe_for(dict(spec["overrides"]))
    diff = {k: (C.RECIPE.get(k), v) for k, v in recipe.items() if C.RECIPE.get(k) != v}
    if set(diff) != set(spec["overrides"]):
        raise SystemExit(f"probe {probe!r} would change more than the frozen overrides: {diff}")
    return recipe, {
        "probe": True,
        "name": probe,
        "overrides": dict(spec["overrides"]),
        "hypothesis": spec["hypothesis"],
        "effective_differences_vs_R3": {k: {"R3": a, "probe": b} for k, (a, b) in diff.items()},
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_run(
    *,
    dataset: str,
    seed: int,
    device: torch.device,
    out_dir,
    epochs: int | None = None,
    smoke: bool = False,
    probe: str | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    if not smoke:
        if int(seed) != C.TRAIN_SEED:
            raise ValueError(f"formal runs must use training seed {C.TRAIN_SEED}")
        if epochs is not None and int(epochs) != int(C.RECIPE["epochs"]):
            raise ValueError(f"formal runs must use {C.RECIPE['epochs']} epochs")
    recipe, probe_record = resolve_recipe(dataset, probe)
    epochs = int(C.RECIPE["epochs"]) if epochs is None else int(epochs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view = D.build_view(dataset)
    spec = view.spec
    task = view.task
    n_train = int(view.train_cache["n_molecules"])
    n_valid = int(view.valid_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

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
    generator = torch.Generator().manual_seed(int(seed) + C.SHUFFLE_OFFSET)

    y_train_ref = view.targets.loss_reference(view.target_norm)
    y_valid = view.targets.y_valid
    if task == "regression":
        y_train_loss = torch.as_tensor(y_train_ref.reshape(-1), dtype=torch.float32)
    elif task == "binary":
        flat = np.asarray(y_train_ref, dtype=np.float64).reshape(-1)
        if not (set(np.unique(flat).tolist()) == {0.0, 1.0}):
            raise ValueError(f"{dataset}: train split must contain both classes 0 and 1")
        y_train_loss = torch.as_tensor(flat, dtype=torch.float32)
    else:
        y_train_loss = torch.as_tensor(np.asarray(y_train_ref, dtype=np.float64), dtype=torch.float32)

    tracker = BestEpochTracker(task)
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
        for step_index, rows in enumerate(batches):
            batch = D.to_device(PREP_DATA.collate(view.train_cache, rows), device)
            y = y_train_loss[rows].to(device)
            pred = model(batch)
            if task == "regression":
                pred = pred.reshape(-1)
                loss = F.mse_loss(pred, y.reshape(-1))
                err = pred - y.reshape(-1)
                abs_sum += float(err.abs().sum())
                sq_sum += float((err * err).sum())
            elif task == "binary":
                pred = pred.reshape(-1)
                loss = F.binary_cross_entropy_with_logits(pred, y.reshape(-1))
                bce_sum += float(loss.detach()) * int(y.numel())
            else:
                mask = torch.isfinite(y)
                if int(mask.sum()) == 0:
                    skipped_batches += 1
                    epoch_skipped += 1
                    continue
                per_label = F.binary_cross_entropy_with_logits(pred, torch.nan_to_num(y), reduction="none")
                loss = (per_label * mask).sum() / mask.sum().clamp(min=1.0)
                bce_sum += float(loss.detach()) * int(mask.sum())
                seen += int(mask.sum())
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

    # ---- reload the best checkpoint and re-verify -----------------------
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = M.state_dict_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = M.state_dict_fingerprint(best_state)
    checkpoint_is_final_epoch = bool(best_state_hash == final_state_hash)
    model.load_state_dict(tracker.to_device(device))
    model.eval()
    valid_raw = D.predict_matrix(model, view.valid_cache, device)
    valid_metrics = MET.task_metrics(task, valid_raw, y_valid, view.target_norm, view.task_names)
    recheck_key = MET.primary_name(task)
    recheck_diff = abs(float(valid_metrics[recheck_key]) - float(tracker.best_metrics[recheck_key]))
    if recheck_diff > C.REPRO_TOL:
        raise RuntimeError(
            f"reloaded best checkpoint Validation {recheck_key} {valid_metrics[recheck_key]} != recorded "
            f"{tracker.best_metrics[recheck_key]} (diff {recheck_diff})"
        )
    primary_key = {"regression": "valid_rmse_raw", "binary": "valid_roc_auc", "multitask": "valid_macro_roc_auc"}[task]
    curve_values = np.asarray([row[primary_key] for row in curve], dtype=np.float64)
    argbest_epoch = (
        int(np.argmin(curve_values) + 1) if task == "regression" else int(np.argmax(curve_values) + 1)
    )
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

    # ---- artifacts ------------------------------------------------------
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
            "n_outputs": int(view.targets.n_tasks),
            "seed": int(seed),
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
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
            "dropout": float(recipe["dropout"]),
            "weight_decay": float(recipe["weight_decay"]),
            "probe": probe_record["name"],
            "best_reload_check": {
                "metric": recheck_key,
                "reloaded": float(valid_metrics[recheck_key]),
                "recorded": float(tracker.best_metrics[recheck_key]),
                "abs_diff": float(recheck_diff),
                "tolerance": float(C.REPRO_TOL),
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
        "dataset": dataset,
        "task": task,
        "label_columns": view.label_columns,
        "units": view.units,
        "smoke": bool(smoke),
        "probe": probe_record,
        "seed": int(seed),
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        "split_file": view.source_meta.get("split_file"),
        "split_file_sha256": view.source_meta.get("split_file_sha256"),
        "split_rows_sha256": view.source_meta.get("split_rows_sha256"),
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
            "train_cache_build_seconds": float(view.train_cache.get("build_seconds", 0.0)),
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
            "loss": {
                "regression": "mse_on_train_standardised_target",
                "binary": "bce_with_logits",
                "multitask": "masked_bce_with_logits__sum_over_observed_cells",
            }[task],
            "checkpoint_rule": tracker.rule(),
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": True,
            "test_used_for_any_selection": False,
            "soup": False,
            "ema": False,
            "swa": False,
            "ssl": False,
            "hyperparameter_search": False,
            "shuffle_generator_seed": int(seed) + C.SHUFFLE_OFFSET,
            "overrides_vs_R3": probe_record["effective_differences_vs_R3"],
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
        f"[{dataset}{'' if probe is None else ' ' + probe} s{seed}] task={task} "
        f"best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={MET.primary_value(task, train_metrics):.5f} "
        f"valid={MET.primary_value(task, valid_metrics):.5f} wall={wall:.1f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from pathlib import Path

    parser = argparse.ArgumentParser(description="P1 formal training run (frozen LCRC-A, seed 0)")
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--seed", type=int, default=C.TRAIN_SEED)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="smoke only; formal runs are 360")
    parser.add_argument("--smoke", action="store_true", help="engineering smoke, not a result")
    parser.add_argument("--probe", default=None, help="P3 validation-only candidate name")
    args = parser.parse_args(argv)

    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
        if args.epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
    elif args.epochs is not None and args.epochs != int(C.RECIPE["epochs"]):
        raise SystemExit(f"formal runs must use {C.RECIPE['epochs']} epochs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.probe:
        out_dir = C.RUNS / f"probe_{args.probe}_s{args.seed}"
    elif args.smoke:
        out_dir = C.RUNS / f"smoke_{args.dataset}_s{args.seed}"
    else:
        out_dir = C.RUNS / f"{args.dataset}_s{args.seed}"
    payload = train_run(
        dataset=args.dataset,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
        probe=args.probe,
    )
    C.log(f"wrote {C.track_rel(out_dir)}")
    del payload
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
