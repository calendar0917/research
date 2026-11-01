"""P1: one formal training run per dataset, training seed 0, frozen R3 recipe.

The loop is the validated ``lcrc_esol_quickstart_v1`` / ``lcrc_lipo_*`` loop with
two task-specific branches and no other changes:

* AdamW, peak LR 3e-4, min LR 1e-5, betas (0.9, 0.999), eps 1e-8, wd 1e-3;
* per-step cosine schedule over ``steps_per_epoch * 360``;
* batch 128, gradient clip 5.0;
* dropout 0.10 on the final atom-level local representation ``h``;
* full 360 epochs, no early stopping, no soup / EMA / SWA / SSL / search;
* checkpoint = the single Validation-best epoch (strict improvement, so ties
  keep the earlier epoch).

Task branches
-------------
regression
    MSE on the Train-standardised target; Validation RMSE is the checkpoint
    criterion; raw-unit metrics come from denormalised predictions.
classification
    ``BCEWithLogitsLoss`` on the raw scalar; the scalar is a logit and
    ``sigmoid`` gives the probability.  The checkpoint criterion is the highest
    Validation ROC-AUC (tie -> lower Validation BCE, then earlier epoch).
    Class labels are never z-scored and never used for class weights.

The module never constructs a Test cache and never reads a Test label: the
``DatasetView`` contains only Train and Validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import mb_common as C
import mb_data as D
import mb_model as M

import lcrc_data as PREP_DATA

PROCESS_T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    err = pred - y
    ss_res = float((err * err).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(y.size),
        "rmse": float(math.sqrt(ss_res / max(y.size, 1))),
        "mae": float(np.abs(err).mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
    }


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prob = 1.0 / (1.0 + np.exp(-logits))
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"ROC-AUC is undefined for {classes.size} class(es) in a split")
    bce = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
    return {
        "n": int(y.size),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "bce": bce,
        "log_loss": float(np.mean(-(y * np.log(np.clip(prob, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - prob, 1e-12, 1))))),
        "accuracy_at_0.5": float(((prob >= 0.5).astype(np.float64) == y).mean()),
        "positive_rate": float((y == 1).mean()),
        "pred_positive_rate_at_0.5": float((prob >= 0.5).mean()),
        "sigmoid_probability": True,
        "threshold_used_for_selection": None,
    }


def task_metrics(task: str, raw: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any] | None) -> dict[str, float]:
    if task == "regression":
        if target_norm is None:
            raise ValueError("regression requires a target normaliser")
        pred = np.asarray(raw, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
        return regression_metrics(pred, y)
    return classification_metrics(raw, y)


# ---------------------------------------------------------------------------
# best-epoch tracker
# ---------------------------------------------------------------------------


@dataclass
class BestEpochTracker:
    """Keep the single best-Validation epoch state (strict improvement)."""

    task: str
    best_epoch: int | None = None
    best_primary: float = math.inf
    best_secondary: float = math.inf
    best_metrics: dict[str, float] | None = None
    updates: list[int] | None = None
    _state: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.task == "classification":
            self.best_primary = -math.inf
        self.updates = []

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        if self.task == "regression":
            primary = float(metrics["rmse"])
            better = self.best_epoch is None or primary < self.best_primary
            secondary = float("nan")
        else:
            primary = float(metrics["roc_auc"])
            secondary = float(metrics["bce"])
            if self.best_epoch is None:
                better = True
            elif primary > self.best_primary:
                better = True
            elif primary == self.best_primary and secondary < self.best_secondary:
                better = True
            else:
                better = False
        if not better:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
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

    def selection_reason(self, epoch: int, metrics: Mapping[str, float]) -> str | None:
        if self.best_epoch != epoch:
            return None
        if self.task == "regression":
            return "lowest_validation_rmse"
        return "highest_validation_roc_auc"


# ---------------------------------------------------------------------------
# code / config hashing
# ---------------------------------------------------------------------------


def code_hashes() -> dict[str, str]:
    out = {}
    for name in ("mb_common.py", "mb_source.py", "mb_rdkit.py", "mb_data.py", "mb_model.py", "mb_train.py", "mb_eval.py"):
        p = C.CODE_DIR / name
        if p.exists():
            out[name] = C.sha256_file(p)
    out.update({f"prep/{k}": v for k, v in C.reused_module_hashes().items()})
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_run(
    *,
    dataset: str,
    seed: int,
    device: torch.device,
    out_dir: Path,
    epochs: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if not smoke:
        if int(seed) != C.TRAIN_SEED:
            raise ValueError(f"formal runs must use training seed {C.TRAIN_SEED}")
        if epochs is not None and int(epochs) != int(C.RECIPE["epochs"]):
            raise ValueError(f"formal runs must use {C.RECIPE['epochs']} epochs")
    epochs = int(C.RECIPE["epochs"]) if epochs is None else int(epochs)
    recipe = dict(C.RECIPE)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view = D.build_view(dataset)
    spec = view.spec
    n_train = int(view.train_cache["n_molecules"])
    n_valid = int(view.valid_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    model, model_meta = M.build_lcrc(spec, view.train_norm, seed, device)
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

    y_train = np.asarray(view.y_train, dtype=np.float64)
    y_valid = np.asarray(view.y_valid, dtype=np.float64)
    if view.task == "regression":
        y_train_scaled = torch.as_tensor(
            (y_train - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
            dtype=torch.float32,
        )
        y_train_loss = y_train_scaled
    else:
        if not (set(np.unique(y_train).tolist()) == {0.0, 1.0}):
            raise ValueError(f"{dataset}: train split must contain both classes 0 and 1")
        y_train_loss = torch.as_tensor(y_train, dtype=torch.float32)

    tracker = BestEpochTracker(task=view.task)
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
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
        for step_index, rows in enumerate(batches):
            batch = D.to_device(PREP_DATA.collate(view.train_cache, rows), device)
            y = y_train_loss[rows].to(device)
            pred = model(batch).reshape(-1)
            if view.task == "regression":
                loss = F.mse_loss(pred, y)
                err = pred - y
                abs_sum += float(err.abs().sum())
                sq_sum += float((err * err).sum())
            else:
                loss = F.binary_cross_entropy_with_logits(pred, y)
                bce_sum += float(loss.detach()) * int(y.numel())
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
            seen += int(y.numel())
        train_seconds = time.perf_counter() - t0

        t_eval = time.perf_counter()
        valid_raw = D.predict_scores(model, view.valid_cache, device)
        valid_metrics = task_metrics(view.task, valid_raw, y_valid, view.target_norm)
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
            "train_seconds": train_seconds,
            "valid_seconds": eval_seconds,
            "is_best": bool(improved),
        }
        if view.task == "regression":
            train_mse = float(sq_sum / max(seen, 1))
            row["train_mse_standardized"] = train_mse
            row["train_rmse_standardized"] = math.sqrt(train_mse)
            row["train_mae_standardized"] = float(abs_sum / max(seen, 1))
            row["valid_rmse_raw"] = float(valid_metrics["rmse"])
            row["valid_mae_raw"] = float(valid_metrics["mae"])
            row["valid_r2_raw"] = float(valid_metrics["r2"])
        else:
            row["train_bce_logits"] = float(bce_sum / max(seen, 1))
            row["valid_roc_auc"] = float(valid_metrics["roc_auc"])
            row["valid_pr_auc"] = float(valid_metrics["pr_auc"])
            row["valid_bce"] = float(valid_metrics["bce"])
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
    valid_raw = D.predict_scores(model, view.valid_cache, device)
    valid_metrics = task_metrics(view.task, valid_raw, y_valid, view.target_norm)
    recheck_key = "rmse" if view.task == "regression" else "roc_auc"
    recheck_diff = abs(float(valid_metrics[recheck_key]) - float(tracker.best_metrics[recheck_key]))
    if recheck_diff > 1e-9:
        raise RuntimeError(
            f"reloaded best checkpoint Validation {recheck_key} {valid_metrics[recheck_key]} != recorded "
            f"{tracker.best_metrics[recheck_key]} (diff {recheck_diff})"
        )
    primary_key = "valid_rmse_raw" if view.task == "regression" else "valid_roc_auc"
    curve_values = np.asarray([row[primary_key] for row in curve], dtype=np.float64)
    argbest_epoch = int(np.argmin(curve_values) + 1) if view.task == "regression" else int(np.argmax(curve_values) + 1)
    if argbest_epoch != int(tracker.best_epoch):
        raise RuntimeError(f"curve argbest {argbest_epoch} != tracked best epoch {tracker.best_epoch}")

    train_raw = D.predict_scores(model, view.train_cache, device)
    train_metrics = task_metrics(view.task, train_raw, y_train, view.target_norm)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        D.predict_scores(model, view.valid_cache, device, batch_size=256)
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
            "model": "LCRC-A",
            "schema": spec.name,
            "feature_spec_hash": spec.hash(),
            "dataset": dataset,
            "task": view.task,
            "seed": int(seed),
            "split_repeat_index": C.SPLIT_REPEAT_INDEX,
            "protocol_id": C.PROTOCOL_ID,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse" if view.task == "regression" else "valid_roc_auc",
            "best_metric_value": float(tracker.best_primary),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch": checkpoint_is_final_epoch,
            "init_state_hash": init_hash,
            "parameters": params_total,
            "target_norm": None if view.target_norm is None else dict(view.target_norm),
            "epochs_run": len(curve),
            "curve_sha256": C.sha256_file(curve_path),
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    np.savez(
        out_dir / "train_predictions.npz",
        rows=np.asarray(view.rows["train"], dtype=np.int64),
        raw=train_raw,
        y_true=y_train,
    )
    np.savez(
        out_dir / "valid_predictions.npz",
        rows=np.asarray(view.rows["valid"], dtype=np.int64),
        raw=valid_raw,
        y_true=y_valid,
    )

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": dataset,
        "task": view.task,
        "label_column": view.label_column,
        "units": view.units,
        "smoke": bool(smoke),
        "seed": int(seed),
        "split_repeat_index": C.SPLIT_REPEAT_INDEX,
        "split_file": view.source_meta["split_file"],
        "split_file_sha256": view.source_meta["split_file_sha256"],
        "split_rows_sha256": view.source_meta["split_rows_sha256"],
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": "fp32",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": code_hashes(),
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
            "name": "LCRC-A",
            "schema": spec.name,
            "parameters": params_total,
            "parameters_breakdown": dict(model_meta["parameters_breakdown"]),
            "config": dict(model_meta["config"]),
            "input_widths": dict(model_meta["input_widths"]),
            "init_state_hash": init_hash,
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
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "scheduler": recipe["scheduler"],
            "gradient_clip": float(recipe["gradient_clip"]),
            "loss": "mse_on_train_standardised_target" if view.task == "regression" else "bce_with_logits",
            "checkpoint_rule": (
                "lowest_validation_rmse__strict_improvement__tie_keeps_earlier"
                if view.task == "regression"
                else "highest_validation_roc_auc__tie_lower_validation_bce__then_earlier_epoch"
            ),
            "early_stopping": False,
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
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse" if view.task == "regression" else "valid_roc_auc",
            "best_metric_value": float(tracker.best_primary),
            "epochs_run": len(curve),
            "n_improvements": len(tracker.updates),
            "improvement_epochs": list(tracker.updates),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch_state": checkpoint_is_final_epoch,
            "curve_argbest_epoch": argbest_epoch,
            "curve_best_value": float(curve_values.min() if view.task == "regression" else curve_values.max()),
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
        f"[{dataset} s{seed}] task={view.task} best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={train_metrics.get('rmse', train_metrics.get('roc_auc')):.5f} "
        f"valid={valid_metrics.get('rmse', valid_metrics.get('roc_auc')):.5f} wall={wall:.1f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 formal training run (frozen LCRC-A, seed 0)")
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--seed", type=int, default=C.TRAIN_SEED)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="smoke only; formal runs are 360")
    parser.add_argument("--smoke", action="store_true", help="2-epoch engineering smoke, not a result")
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
    out_dir = Path(args.out_dir) if args.out_dir else C.RUNS / f"{args.dataset}_s{args.seed}"
    if args.smoke:
        out_dir = Path(args.out_dir) if args.out_dir else C.RUNS / f"smoke_{args.dataset}_s{args.seed}"
    payload = train_run(
        dataset=args.dataset,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
    )
    C.log(f"wrote {C.track_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
