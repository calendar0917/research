"""Six fixed training units: {C1, FULL} x seeds {0,1,2} on the Fit partition.

The recipe is the frozen R3 one inherited from the dissection round (AdamW,
per-step cosine 3e-4 -> 1e-5, wd 1e-3, batch 128, dropout 0.10 on the shared
``h``, gradient clip 5.0, 360 full epochs, no early stopping).  The only things
that change are the *data*: training on Fit, checkpoint selection on Select
(original logD units, strictly smaller, ties keep the earlier epoch).

C1 (``NO_BOND_MEAN``, 228,674 parameters) is the main model; FULL (``A0``,
247,106) exists only to check whether the diagnostic phenomenon depends on the
simplification.  Each seed builds one FULL initial state on CPU and remaps it by
fixed column deletion, exactly as the dissection round did, so C1 and FULL of
the same seed share every shared parameter bitwise and see the same batches.
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

import edit_common as C
import edit_data as ED

import dissect_model as M
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


def raw_metrics(standardised: np.ndarray, y: np.ndarray, target_norm: Mapping[str, Any]) -> dict[str, float]:
    pred = np.asarray(standardised, dtype=np.float64) * float(target_norm["scale"]) + float(target_norm["mean"])
    return regression_metrics(pred, y)


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self.improvement_epochs: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        if self.best_epoch is not None and not primary < self.best_primary:
            return False
        self.best_epoch = int(epoch)
        self.best_primary = primary
        self.best_metrics = dict(metrics)
        self.improvement_epochs.append(int(epoch))
        return True

    def capture(self, model: torch.nn.Module) -> None:
        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @property
    def state(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("tracker has no captured state")
        return self._state


def run_matrix_allows(model_variant: str, seed: int) -> bool:
    path = C.CONFIGS / "run_matrix.json"
    if not path.exists():
        return False
    matrix = C.read_json(path)
    return f"{model_variant}:s{int(seed)}" in matrix["allowed_units"]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_unit(
    *,
    model_variant: str,
    seed: int,
    device: torch.device,
    out_dir: Path | None = None,
    epochs: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if model_variant not in C.MODELS:
        raise ValueError(f"unknown model {model_variant!r}")
    allowed = run_matrix_allows(model_variant, seed)
    if not allowed and not smoke:
        raise PermissionError(f"{model_variant}:s{seed} is not a pre-registered unit")
    recipe = dict(C.RECIPE)
    epochs = int(C.EPOCHS) if epochs is None else int(epochs)
    if not smoke and epochs != int(C.EPOCHS):
        raise SystemExit(f"formal runs must use {C.EPOCHS} epochs")
    out_dir = C.run_dir(model_variant, seed) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = ED.frozen_spec()
    fit_cache = ED.load_cache("fit")
    select_cache = ED.load_cache("select")
    train_norm = ED.train_norm()
    tgt = ED.target_norm()
    y_fit = ED.y_aligned("fit", np.asarray(fit_cache["rows"]).tolist(), "train")
    y_select = ED.y_aligned("select", np.asarray(select_cache["rows"]).tolist(), "select")

    n_train = int(fit_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs

    full_state = M.build_full_state(spec, train_norm, int(seed))
    init_state = M.remap_state(model_variant, full_state)
    init_hash = M.state_fingerprint(init_state)
    full_init_hash = M.state_fingerprint(full_state)
    shared = M.shared_state_equal(model_variant, full_state, init_state)

    model = M.DissectLCRC(spec, train_norm, model_variant)
    model.load_state_dict(init_state, strict=True)
    model.to(device)
    params_total = M.parameter_count(model)
    expected = C.EXPECTED_PARAMS[model_variant]
    if params_total != expected:
        raise RuntimeError(f"{model_variant}: {params_total} trainable parameters != frozen {expected}")

    dropout_calls = {"n": 0}
    if hasattr(model, "drop"):
        model.drop.register_forward_hook(lambda *_: dropout_calls.__setitem__("n", dropout_calls["n"] + 1))

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
    generator = torch.Generator().manual_seed(int(seed) + int(C.RECIPE["shuffle_offset"]))
    y_fit_scaled = torch.as_tensor(
        (y_fit - float(tgt["mean"])) / float(tgt["scale"]), dtype=torch.float32
    )

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    grad_norm_max = 0.0
    clip_events = 0
    batch_order_probe: list[int] = []
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        if epoch == 1:
            batch_order_probe = perm[:10]
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for step_index, rows in enumerate(batches):
            batch = ED.to_device(ED.collate(fit_cache, rows), device)
            y = y_fit_scaled[rows].to(device)
            pred = model(batch).reshape(-1)
            loss = F.mse_loss(pred, y)
            err = pred - y
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
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
        select_std = ED.predict_scores(model, select_cache, device)
        select_metrics = raw_metrics(select_std, y_select, tgt)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, select_metrics)
        if improved:
            tracker.capture(model)
        train_mse = float(sq_sum / max(seen, 1))
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "train_seconds": float(train_seconds),
                "select_seconds": float(eval_seconds),
                "is_best": bool(improved),
                "train_mse_standardized": train_mse,
                "train_rmse_standardized": math.sqrt(train_mse),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "select_rmse_raw": float(select_metrics["rmse"]),
                "select_mae_raw": float(select_metrics["mae"]),
                "select_r2_raw": float(select_metrics["r2"]),
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any selection improvement")

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = M.state_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = M.state_fingerprint(best_state)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    select_std = ED.predict_scores(model, select_cache, device)
    select_metrics = raw_metrics(select_std, y_select, tgt)
    recheck = abs(float(select_metrics["rmse"]) - float(tracker.best_metrics["rmse"]))
    if recheck > 1e-5:
        raise RuntimeError(f"reloaded best checkpoint Select RMSE differs by {recheck}")
    curve_values = np.asarray([row["select_rmse_raw"] for row in curve], dtype=np.float64)
    if int(np.argmin(curve_values) + 1) != int(tracker.best_epoch):
        raise RuntimeError("curve argbest != tracked best epoch")
    fit_std = ED.predict_scores(model, fit_cache, device)
    fit_metrics = raw_metrics(fit_std, y_fit, tgt)

    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": "LCRC-A-local-edit-diagnosis",
            "variant": model_variant,
            "variant_name": M.C.VARIANT_NAMES[model_variant],
            "feature_spec_hash": spec.hash(),
            "dataset": "Lipophilicity",
            "partition": "fit",
            "seed": int(seed),
            "protocol_id": C.ROUND_ID,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "select_rmse",
            "best_metric_value": float(tracker.best_primary),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "init_state_hash": init_hash,
            "full_init_state_hash": full_init_hash,
            "parameters": params_total,
            "parameters_expected": expected,
            "target_norm": dict(tgt),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(train_norm),
            "epochs_run": len(curve),
            "smoke": bool(smoke),
        },
        checkpoint_path,
    )
    np.savez(
        out_dir / "fit_predictions.npz",
        rows=np.asarray(fit_cache["rows"], dtype=np.int64),
        raw_standardised=fit_std,
        raw=fit_std * float(tgt["scale"]) + float(tgt["mean"]),
        y_true=y_fit,
    )
    np.savez(
        out_dir / "select_predictions.npz",
        rows=np.asarray(select_cache["rows"], dtype=np.int64),
        raw_standardised=select_std,
        raw=select_std * float(tgt["scale"]) + float(tgt["mean"]),
        y_true=y_select,
    )

    payload: dict[str, Any] = {
        "round": C.ROUND_ID,
        "status": "completed",
        "smoke": bool(smoke),
        "dataset": "Lipophilicity",
        "partition_train": "fit",
        "partition_select": "select",
        "diagnose_or_audit_labels_read": False,
        "model": {
            "variant": model_variant,
            "variant_name": M.C.VARIANT_NAMES[model_variant],
            "parameters": params_total,
            "parameters_expected": expected,
            "init_state_hash": init_hash,
            "full_init_state_hash": full_init_hash,
            "shared_init_bitwise_equal_to_full": bool(shared["all_equal"]),
            "dropout_modules": 1 if hasattr(model, "drop") else 0,
            "dropout_calls_in_run": int(dropout_calls["n"]),
        },
        "seed": int(seed),
        "data": {
            "n_fit": n_train,
            "n_select": int(select_cache["n_molecules"]),
            "fit_fingerprint": fit_cache["meta"]["content_fingerprint"],
            "select_fingerprint": select_cache["meta"]["content_fingerprint"],
            "fit_rows_sha256": C.row_sha256(np.asarray(fit_cache["rows"]).tolist()),
            "select_rows_sha256": C.row_sha256(np.asarray(select_cache["rows"]).tolist()),
            "target_norm": dict(tgt),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(train_norm),
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
            "gradient_clip": float(recipe["gradient_clip"]),
            "dropout": float(recipe["dropout"]),
            "loss": "mse_on_fit_standardised_target",
            "checkpoint_rule": "lowest_select_rmse__strict_improvement__tie_keeps_earlier",
            "early_stopping": False,
            "test_used_for_any_selection": False,
            "shuffle_generator_seed": int(seed) + int(C.RECIPE["shuffle_offset"]),
            "batch_order_probe_epoch1_first10": batch_order_probe,
        },
        "metrics": {"fit": fit_metrics, "select": select_metrics},
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "select_rmse",
            "best_metric_value": float(tracker.best_primary),
            "n_improvements": len(tracker.improvement_epochs),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "curve_argbest_epoch": int(np.argmin(curve_values) + 1),
            "select_recheck_abs_diff": float(recheck),
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
            "mean_epoch_seconds": float(np.mean([row["train_seconds"] + row["select_seconds"] for row in curve])),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
            "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "hostname": platform.node(),
            "device": str(device),
        },
        "provenance": {
            "git_commit": C.git_commit(),
            "git_dirty_tracked": C.git_dirty(),
            "code_hashes": C.code_hashes(),
        },
        "predictions": {
            "fit": C.track_rel(out_dir / "fit_predictions.npz"),
            "select": C.track_rel(out_dir / "select_predictions.npz"),
        },
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(
        f"[{model_variant} s{seed}] best_epoch={tracker.best_epoch}/{len(curve)} "
        f"fit_rmse={fit_metrics['rmse']:.5f} select_rmse={select_metrics['rmse']:.5f} "
        f"params={params_total} wall={wall:.1f}s"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local-edit diagnosis: one training unit")
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
        if args.epochs > 2:
            raise SystemExit("smoke runs may not exceed 2 epochs")
    elif args.epochs is not None and args.epochs != int(C.EPOCHS):
        raise SystemExit(f"formal runs must use {C.EPOCHS} epochs")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    out_dir = Path(args.out_dir) if args.out_dir else (
        C.RUNS / f"smoke_s{args.seed}_{args.model}" if args.smoke else C.run_dir(args.model, args.seed)
    )
    train_unit(
        model_variant=args.model,
        seed=int(args.seed),
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
