"""One formal training run for one (repeat, variant, seed).

The optimization loop is the frozen R3 recipe (``mb_train.py`` / ``mb2_train.py``
semantics): AdamW, per-step cosine over ``27 * 360 = 9720`` updates, batch 128,
gradient clip 5.0, MSE on the Train-standardised target, one dropout ``p=0.10``
on the shared local representation ``h``, 360 full epochs, best-Validation-RMSE
checkpoint with strict improvement (ties keep the earlier epoch).

The module never builds the Test structure cache and never decodes a Test
target: the view contains Train and Validation only.
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

import dissect_common as C
import dissect_data as D
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


# ---------------------------------------------------------------------------
# best epoch
# ---------------------------------------------------------------------------


class BestEpochTracker:
    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_primary = math.inf
        self.best_metrics: dict[str, float] | None = None
        self.improvement_epochs: list[int] = []
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        primary = float(metrics["rmse"])
        better = self.best_epoch is None or primary < self.best_primary
        if not better:
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


# ---------------------------------------------------------------------------
# pre-registered run matrix guard
# ---------------------------------------------------------------------------


def matrix_allows(repeat: int, variant: str) -> tuple[bool, str]:
    """Only pre-registered (repeat, variant) cells may run."""
    path = C.CONFIGS / "run_matrix.json"
    if not path.exists():
        return False, f"frozen run matrix {C.track_rel(path)} is missing"
    matrix = C.read_json(path)
    cell = f"r{repeat}:{variant}"
    if cell in matrix["allowed_cells"]:
        return True, "frozen_matrix"
    rule = matrix["cross_split_rule"]
    if repeat in rule["repeats"]:
        if variant in rule["variants"]:
            return True, "frozen_matrix_cross_split"
        if variant == rule["compact_variant"]:
            return True, "frozen_matrix_cross_split_compact"
    return False, f"{cell} is not a pre-registered cell"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_run(
    *,
    variant: str,
    seed: int,
    repeat: int,
    device: torch.device,
    out_dir: Path,
    epochs: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if variant not in C.VARIANT_DEFS:
        raise ValueError(f"unknown variant {variant!r}")
    allowed, why = matrix_allows(int(repeat), variant)
    if not allowed and not smoke:
        raise PermissionError(why)
    if not smoke:
        if int(seed) not in C.SEEDS:
            raise ValueError(f"formal runs must use training seeds {C.SEEDS}")
        if epochs is not None and int(epochs) != int(C.RECIPE["epochs"]):
            raise ValueError(f"formal runs must use {C.RECIPE['epochs']} epochs")
    epochs = int(C.RECIPE["epochs"]) if epochs is None else int(epochs)
    recipe = dict(C.RECIPE)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    view = D.build_view(int(repeat))
    spec = view.spec
    n_train = int(view.train_cache["n_molecules"])
    n_valid = int(view.valid_cache["n_molecules"])
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_updates = steps_per_epoch * epochs
    target_norm = view.target_norm

    # ---- shared FULL initial state, then remap into the variant ----------
    full_state = M.build_full_state(spec, view.train_norm, int(seed))
    init_state = M.remap_state(variant, full_state)
    init_hash = M.state_fingerprint(init_state)
    full_init_hash = M.state_fingerprint(full_state)
    shared = M.shared_state_equal(variant, full_state, init_state)

    model = M.DissectLCRC(spec, view.train_norm, variant)
    model.load_state_dict(init_state, strict=True)
    model.to(device)
    params_total = M.parameter_count(model)
    expected = C.EXPECTED_PARAMS[variant]
    if params_total != expected:
        raise RuntimeError(f"{variant}: trainable parameters {params_total} != frozen {expected}")

    # Dropout call audit: exactly one dropout module, one call per forward.
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
    generator = torch.Generator().manual_seed(int(seed) + int(C.SHUFFLE_OFFSET))

    y_train = np.asarray(view.y_train, dtype=np.float64)
    y_valid = np.asarray(view.y_valid, dtype=np.float64)
    y_train_scaled = torch.as_tensor(
        (y_train - float(target_norm["mean"])) / float(target_norm["scale"]), dtype=torch.float32
    )

    tracker = BestEpochTracker()
    curve: list[dict[str, Any]] = []
    clip_events = 0
    grad_norm_max = 0.0
    gpu_peak = 0
    batch_order_probe: list[int] = []
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
            batch = D.to_device(PREP_DATA.collate(view.train_cache, rows), device)
            y = y_train_scaled[rows].to(device)
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
        valid_raw_standardised = D.predict_scores(model, view.valid_cache, device)
        valid_metrics = raw_metrics(valid_raw_standardised, y_valid, target_norm)
        eval_seconds = time.perf_counter() - t_eval
        improved = tracker.consider(epoch, valid_metrics)
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
                "valid_seconds": float(eval_seconds),
                "is_best": bool(improved),
                "train_mse_standardized": train_mse,
                "train_rmse_standardized": math.sqrt(train_mse),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "valid_rmse_raw": float(valid_metrics["rmse"]),
                "valid_mae_raw": float(valid_metrics["mae"]),
                "valid_r2_raw": float(valid_metrics["r2"]),
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise RuntimeError("training ended without any validation improvement")

    # ---- reload the best checkpoint and re-verify ------------------------
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_state_hash = M.state_fingerprint(final_state)
    best_state = tracker.state
    best_state_hash = M.state_fingerprint(best_state)
    checkpoint_is_final_epoch = bool(best_state_hash == final_state_hash)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    valid_raw_standardised = D.predict_scores(model, view.valid_cache, device)
    valid_metrics = raw_metrics(valid_raw_standardised, y_valid, target_norm)
    recheck_diff = abs(float(valid_metrics["rmse"]) - float(tracker.best_metrics["rmse"]))
    if recheck_diff > C.REPRO_TOL:
        raise RuntimeError(
            f"reloaded best checkpoint Valid RMSE {valid_metrics['rmse']} != recorded "
            f"{tracker.best_metrics['rmse']} (diff {recheck_diff})"
        )
    curve_values = np.asarray([row["valid_rmse_raw"] for row in curve], dtype=np.float64)
    argbest_epoch = int(np.argmin(curve_values) + 1)
    if argbest_epoch != int(tracker.best_epoch):
        raise RuntimeError(f"curve argbest {argbest_epoch} != tracked best epoch {tracker.best_epoch}")

    train_raw_standardised = D.predict_scores(model, view.train_cache, device)
    train_metrics = raw_metrics(train_raw_standardised, y_train, target_norm)

    cuda_nondeterminism: dict[str, Any] | None = None
    if smoke and device.type == "cuda":
        a = D.predict_scores(model, view.valid_cache, device)
        b = D.predict_scores(model, view.valid_cache, device)
        cpu_model = M.load_checkpoint_model(
            spec, view.train_norm, variant, {k: v.cpu() for k, v in model.state_dict().items()}, torch.device("cpu")
        )
        c = D.predict_scores(cpu_model, view.valid_cache, torch.device("cpu"))
        cuda_nondeterminism = {
            "cuda_vs_cuda_max_abs_delta_standardised": float(np.abs(a - b).max()),
            "cuda_vs_cpu_max_abs_delta_standardised": float(np.abs(a - c).max()),
            "cuda_vs_cpu_rmse_delta_raw": float(
                abs(
                    raw_metrics(a, y_valid, target_norm)["rmse"]
                    - raw_metrics(c, y_valid, target_norm)["rmse"]
                )
            ),
        }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        D.predict_scores(model, view.valid_cache, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    dropout_calls_observed = int(dropout_calls["n"])

    # ---- artifacts -------------------------------------------------------
    curve_path = out_dir / "curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model": "LCRC-A-dissection",
            "variant": variant,
            "variant_name": C.VARIANT_NAMES[variant],
            "schema": spec.name,
            "feature_spec_hash": spec.hash(),
            "dataset": "Lipophilicity",
            "repeat": int(repeat),
            "split_file": view.source_meta["split_file"],
            "split_file_sha256": view.source_meta["split_file_sha256"],
            "seed": int(seed),
            "protocol_id": C.PROTOCOL_ID,
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse",
            "best_metric_value": float(tracker.best_primary),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "init_state_hash": init_hash,
            "full_init_state_hash": full_init_hash,
            "parameters": params_total,
            "parameters_expected": expected,
            "target_norm": dict(target_norm),
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "best_reload_check": {
                "metric": "valid_rmse",
                "reloaded": float(valid_metrics["rmse"]),
                "recorded": float(tracker.best_metrics["rmse"]),
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
        raw_standardised=train_raw_standardised,
        raw=train_raw_standardised * float(target_norm["scale"]) + float(target_norm["mean"]),
        y_true=y_train,
    )
    np.savez(
        out_dir / "valid_predictions.npz",
        rows=np.asarray(view.rows["valid"], dtype=np.int64),
        raw_standardised=valid_raw_standardised,
        raw=valid_raw_standardised * float(target_norm["scale"]) + float(target_norm["mean"]),
        y_true=y_valid,
    )

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "dataset": "Lipophilicity",
        "task": "regression",
        "label_column": view.source_meta["label_column"],
        "units": view.source_meta["units"],
        "status": "completed",
        "smoke": bool(smoke),
        "repeat": int(repeat),
        "seed": int(seed),
        "variant": variant,
        "variant_name": C.VARIANT_NAMES[variant],
        "variant_definition": dict(C.VARIANT_DEFS[variant]),
        "variant_local_cols": list(C.variant_cols(variant)[0]),
        "variant_rho_cols": list(C.variant_cols(variant)[1]),
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
        "code_hashes": C.code_hashes(),
        "data_view": view.audit(),
        "data": {
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": "not constructed (Test is scored only after the stage freeze)",
            "feature_spec_hash": spec.hash(),
            "train_cache_fingerprint": view.train_cache["meta"]["content_fingerprint"],
            "valid_cache_fingerprint": view.valid_cache["meta"]["content_fingerprint"],
            "train_norm_fingerprint": PREP_DATA.norm_fingerprint(view.train_norm),
            "target_norm": dict(target_norm),
            "test_labels_read": False,
            "test_cache_built": False,
        },
        "model": {
            "name": "LCRC-A-dissection",
            "variant": variant,
            "schema": spec.name,
            "parameters": params_total,
            "parameters_nominal": params_total,
            "parameters_expected": expected,
            "parameters_match": bool(params_total == expected),
            "parameters_breakdown": M.module_breakdown(model),
            "config": dict(M.DEFAULT_CONFIG),
            "has_local": bool(M.C.VARIANT_DEFS[variant]["use_local"]),
            "has_f": bool(M.C.VARIANT_DEFS[variant]["use_f"]),
            "has_rho": bool(M.C.VARIANT_DEFS[variant]["use_rho"]),
            "has_global_bias": bool(M.C.VARIANT_DEFS[variant]["add_bias"]),
            "s2_aggregation": M.C.VARIANT_DEFS[variant]["s2_agg"],
            "add_reduction": M.C.VARIANT_DEFS[variant]["add_reduction"],
            "init_state_hash": init_hash,
            "full_init_state_hash": full_init_hash,
            "shared_init_bitwise_equal_to_full": bool(shared["all_equal"]),
            "shared_init_checks": shared["checks"],
            "iterative_hidden_message_passing": False,
            "dropout": float(recipe["dropout"]),
            "dropout_position": recipe["dropout_position"],
            "dropout_modules": 1 if hasattr(model, "drop") else 0,
            "dropout_calls_in_run": dropout_calls_observed,
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
            "loss": "mse_on_train_standardised_target",
            "checkpoint_rule": "lowest_validation_rmse__strict_improvement__tie_keeps_earlier",
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": True,
            "test_used_for_any_selection": False,
            "soup": False,
            "ema": False,
            "swa": False,
            "ssl": False,
            "hyperparameter_search": False,
            "shuffle_generator_seed": int(seed) + int(C.SHUFFLE_OFFSET),
            "batch_order_probe_epoch1_first10": batch_order_probe,
        },
        "metrics": {
            "train": {**train_metrics, "checkpoint": "best_epoch"},
            "valid": {**valid_metrics, "checkpoint": "best_epoch"},
        },
        "checkpoint": {
            "best_epoch": int(tracker.best_epoch),
            "best_metric_name": "valid_rmse",
            "best_metric_value": float(tracker.best_primary),
            "epochs_run": len(curve),
            "n_improvements": len(tracker.improvement_epochs),
            "improvement_epochs": list(tracker.improvement_epochs),
            "best_state_hash": best_state_hash,
            "final_epoch_state_hash": final_state_hash,
            "checkpoint_is_final_epoch_state": checkpoint_is_final_epoch,
            "curve_argbest_epoch": argbest_epoch,
            "curve_best_value": float(curve_values.min()),
            "valid_recheck_metric": "valid_rmse",
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
            "cuda_eval_nondeterminism": cuda_nondeterminism,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "mean_epoch_seconds": float(np.mean([row["train_seconds"] + row["valid_seconds"] for row in curve])),
            "inference_seconds_per_valid_split": float(inference_seconds),
            "inference_batch_size": 256,
            "gpu_hours": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
            "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0) if device.type == "cuda" else 0.0,
        },
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
        f"[r{repeat} {variant} {C.VARIANT_NAMES[variant]} s{seed}] best_epoch={tracker.best_epoch}/{len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} "
        f"params={params_total} wall={wall:.1f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC-A dissection: one formal training run")
    parser.add_argument("--variant", required=True, choices=list(C.VARIANT_ORDER))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=C.MAIN_REPEAT, choices=list(C.REPEATS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="smoke only; formal runs are 360")
    parser.add_argument("--smoke", action="store_true", help="engineering smoke, not a result")
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
    elif args.smoke:
        out_dir = C.RUNS / f"smoke_r{args.repeat}_{args.variant}_s{args.seed}"
    else:
        out_dir = C.run_dir(args.repeat, args.variant, args.seed)
    payload = train_run(
        variant=args.variant,
        seed=int(args.seed),
        repeat=int(args.repeat),
        device=device,
        out_dir=out_dir,
        epochs=args.epochs,
        smoke=bool(args.smoke),
    )
    C.log(f"wrote {C.track_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
