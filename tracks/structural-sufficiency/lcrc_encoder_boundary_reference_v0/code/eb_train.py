"""Formal training of the single new unit B (GINE encoder + A's post_local).

Inherited recipe (CSC v0, frozen): batch 128, 360 epochs, AdamW 3e-4 -> 1e-5
per-step cosine without warmup, wd 1e-3, betas (0.9, 0.999), eps 1e-8, clip 5,
FP32, standardised-target MSE, one parameter group, no auxiliary loss /
pretraining / distillation / EMA / SWA / early stopping.

Every epoch scores the full official Train and Valid splits in eval mode; Test is
never built, scored or read.  Checkpoint selection reads only the Valid RMSE.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import eb_common as E
import cs_common as C

C.setup_paths()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cs_data as D  # noqa: E402
import eb_data as BD  # noqa: E402
import eb_model as BM  # noqa: E402

CURVE_FIELDS = [
    "epoch", "global_step", "learning_rate", "train_loss_step_mean",
    "train_rmse_raw", "train_mae_raw", "valid_rmse_raw", "valid_mae_raw",
    "train_seconds", "eval_seconds", "epoch_seconds", "cumulative_seconds",
    "grad_norm_max", "grad_norm_mean", "clip_events",
    "best_epoch_so_far", "best_valid_rmse_so_far", "is_best", "rng_sha256",
]


def build_optimizer(model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("this round uses exactly one parameter group")
    audit = {
        "optimizer": "AdamW", "mode": "from_scratch", "n_param_groups": 1,
        "n_parameters": int(sum(p.numel() for p in params)),
        "peak_lr": float(C.RECIPE["peak_lr"]), "min_lr": float(C.RECIPE["min_lr"]),
        "betas": list(C.RECIPE["adam_betas"]), "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "per_step_cosine_no_warmup",
    }
    return optimizer, audit


def evaluate(split: BD.GineSplit, rows: np.ndarray, targets_raw: np.ndarray, model, scaler, device, batch_size: int = 256):
    order = np.asarray(rows, dtype=np.int64)
    preds, ys = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, order.size, batch_size):
            chunk = order[start:start + batch_size]
            batch = BD.make_batch(split, chunk, device)
            pred = model(batch).detach().cpu().numpy().astype(np.float64).reshape(-1)
            preds.append(pred)
            ys.append(np.asarray(targets_raw[chunk], dtype=np.float64))
    if was_training:
        model.train()
    pred_std = np.concatenate(preds) if preds else np.zeros(0, np.float64)
    y_std = np.concatenate(ys) if ys else np.zeros(0, np.float64)
    mean, std = float(scaler["mean"]), float(scaler["std"])
    pred_raw = pred_std * std + mean
    y_raw = y_std * std + mean
    return {
        "row_ids": np.asarray(split.row_ids[order], dtype=np.int64),
        "pred_raw": pred_raw,
        "y_raw": y_raw,
        "rmse_raw": C.rmse(pred_raw, y_raw),
        "mae_raw": C.mae(pred_raw, y_raw),
    }


def save_predictions(path: Path, payload: Mapping[str, Any]) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v) for k, v in payload.items()})
    os.replace(tmp, path)
    return E.sha256_file(path)


def train_unit(bundle, splits: Mapping[str, BD.GineSplit], n_categories, *, out_dir: Path, device, epochs: int, hard_seconds: float):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = E.DATASET
    t_start = time.perf_counter()
    started = C.utcnow()
    C.set_seeds(C.SEED)
    steps_epoch = C.steps_per_epoch(dataset)
    total_steps = epochs * steps_epoch

    model, model_audit = BM.build_model(bundle.spec, bundle.norm, n_categories, seed=C.SEED, device=device)
    model_audit["initial_state_sha256"] = C.state_fingerprint(C.state_cpu(model))
    optimizer, opt_audit = build_optimizer(model)
    plan = D.batch_order_plan(dataset, epochs)
    a_plan = E.read_json(E.A_RUN / "training_order.json")["plan_sha256"]
    if epochs == int(C.RECIPE["epochs"]) and plan["plan_sha256"] != a_plan:
        raise RuntimeError(f"batch plan {plan['plan_sha256']} != A plan {a_plan}")

    tr, va = splits["train"], splits["valid"]
    train_order_rows = np.arange(int(tr.row_ids.size), dtype=np.int64)
    valid_rows = np.arange(int(va.row_ids.size), dtype=np.int64)
    train_targets_raw = np.asarray(bundle.targets_raw["train"], dtype=np.float64)
    # targets_raw is in cache-local order; split arrays use the same order
    valid_targets_raw = np.asarray(bundle.targets_raw["valid"], dtype=np.float64)
    scaler = bundle.scaler

    curve_rows: list[dict[str, Any]] = []
    best_epoch = None
    best_valid = math.inf
    best_state = None
    best_metrics = None
    best_predictions = None
    final_predictions = None
    clip_events_total = 0
    global_step = 0
    stop_reason = "RUNNING"
    timing_projection = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if time.perf_counter() - t_start > float(hard_seconds):
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        order = plan["plan"][epoch - 1]
        losses, gnorm = [], []
        clip_events = 0
        aborted = False
        for start in range(0, order.size, int(C.RECIPE["batch_size"])):
            if time.perf_counter() - t_start > float(hard_seconds):
                aborted = True
                break
            rows = order[start:start + int(C.RECIPE["batch_size"])]
            batch = BD.make_batch(tr, rows, device)
            y = torch.as_tensor(
                np.asarray((train_targets_raw[rows] - scaler["mean"]) / scaler["std"], dtype=np.float32),
                device=device,
            )
            lr = C.lr_at(global_step, total_steps)
            optimizer.param_groups[0]["lr"] = float(lr)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch).view(-1)
            loss = F.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}")
            loss.backward()
            g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
            if not math.isfinite(g):
                raise RuntimeError(f"non-finite gradient norm at step {global_step}")
            clip_events += int(g > float(C.RECIPE["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
            gnorm.append(g)
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            break
        train_seconds = float(time.perf_counter() - t_epoch)

        t_eval = time.perf_counter()
        rep_tr = evaluate(tr, train_order_rows, train_targets_raw, model, scaler, device)
        rep_va = evaluate(va, valid_rows, valid_targets_raw, model, scaler, device)
        eval_seconds = float(time.perf_counter() - t_eval)
        valid_rmse = float(rep_va["rmse_raw"])
        # keep the exact same ascending-row-id order as A's saved predictions
        pred_valid = {k: rep_va[k] for k in ("row_ids", "pred_raw", "y_raw")}
        is_best = valid_rmse < best_valid
        if is_best:
            best_valid = valid_rmse
            best_epoch = int(epoch)
            best_state = C.state_cpu(model)
            best_metrics = {
                "train": {"rmse_raw": rep_tr["rmse_raw"], "mae_raw": rep_tr["mae_raw"]},
                "valid": {"rmse_raw": rep_va["rmse_raw"], "mae_raw": rep_va["mae_raw"]},
            }
            best_predictions = {
                "train_row_ids": rep_tr["row_ids"], "train_pred_raw": rep_tr["pred_raw"], "train_y_raw": rep_tr["y_raw"],
                "valid_row_ids": rep_va["row_ids"], "valid_pred_raw": rep_va["pred_raw"], "valid_y_raw": rep_va["y_raw"],
                "epoch": np.asarray([epoch]),
            }
            C.atomic_torch_save(
                out_dir / "checkpoint_best.pt",
                {
                    "schema": E.SCHEMA_CHECKPOINT, "kind": "best_valid_rmse",
                    "epoch": int(epoch), "global_step": int(global_step),
                    "state_dict": best_state, "dataset": dataset, "arm": "B", "seed": C.SEED,
                    "model_audit": model_audit, "optimizer_audit": opt_audit,
                    "identity": bundle.extra["identity_audit"],
                    "train_plan_sha256": plan["plan_sha256"],
                    "valid_rmse_raw": valid_rmse, "created_utc": C.utcnow(),
                    "git_commit": C.git_commit(),
                },
            )
            save_predictions(out_dir / "predictions_best.npz", best_predictions)
            E.write_json(out_dir / "best_checkpoint_eval.json", {
                "epoch": int(epoch), "valid_rmse_raw": valid_rmse,
                "selection_metric": "valid_rmse_raw",
                **{f"{s}_{m}_raw": best_metrics[s][f"{m}_raw"] for s in ("train", "valid") for m in ("rmse", "mae")},
            })
        final_predictions = {
            "train_row_ids": rep_tr["row_ids"], "train_pred_raw": rep_tr["pred_raw"], "train_y_raw": rep_tr["y_raw"],
            "valid_row_ids": rep_va["row_ids"], "valid_pred_raw": rep_va["pred_raw"], "valid_y_raw": rep_va["y_raw"],
            "epoch": np.asarray([epoch]),
        }
        clip_events_total += int(clip_events)

        def _mean(values):
            return float(np.mean(values)) if len(values) else None

        row = {
            "epoch": int(epoch), "global_step": int(global_step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss_step_mean": _mean(losses),
            "train_rmse_raw": rep_tr["rmse_raw"], "train_mae_raw": rep_tr["mae_raw"],
            "valid_rmse_raw": valid_rmse, "valid_mae_raw": rep_va["mae_raw"],
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "grad_norm_max": float(max(gnorm)) if gnorm else None,
            "grad_norm_mean": _mean(gnorm), "clip_events": int(clip_events),
            "best_epoch_so_far": best_epoch, "best_valid_rmse_so_far": float(best_valid) if best_epoch else None,
            "is_best": bool(is_best), "rng_sha256": C.rng_hashes()["torch_cpu"],
        }
        curve_rows.append(row)
        C.write_csv(out_dir / "train_curve.csv", curve_rows, CURVE_FIELDS)
        if epoch == 2:
            mean_epoch = float(np.mean([r["epoch_seconds"] for r in curve_rows]))
            timing_projection = {
                "epochs_probed": 2, "mean_epoch_seconds": mean_epoch,
                "projected_total_seconds": mean_epoch * epochs,
                "hard_seconds": float(hard_seconds),
                "projected_total_gpu_hours": mean_epoch * epochs / 3600.0,
            }
            E.write_json(out_dir / "timing_precheck.json", timing_projection)
            if timing_projection["projected_total_seconds"] > float(hard_seconds):
                stop_reason = "TIMING_PRECHECK_FAIL"
                C.log(f"[B] timing precheck failed: {timing_projection}")
                break
        if epoch % 20 == 1 or is_best or epoch == epochs:
            C.log(
                f"[B] ep {epoch}/{epochs} train {row['train_rmse_raw']:.4f} valid {valid_rmse:.4f} "
                f"{'*best*' if is_best else ''} ({row['epoch_seconds']:.1f}s)"
            )
    stop_reason = "COMPLETED" if len(curve_rows) == epochs else stop_reason

    if curve_rows:
        model.eval()
        rep_tr = evaluate(tr, train_order_rows, train_targets_raw, model, scaler, device)
        rep_va = evaluate(va, valid_rows, valid_targets_raw, model, scaler, device)
        final_predictions = {
            "train_row_ids": rep_tr["row_ids"], "train_pred_raw": rep_tr["pred_raw"], "train_y_raw": rep_tr["y_raw"],
            "valid_row_ids": rep_va["row_ids"], "valid_pred_raw": rep_va["pred_raw"], "valid_y_raw": rep_va["y_raw"],
            "epoch": np.asarray([len(curve_rows)]),
        }
        C.atomic_torch_save(
            out_dir / "checkpoint_final.pt",
            {
                "schema": E.SCHEMA_CHECKPOINT, "kind": "final_epoch",
                "epoch": int(len(curve_rows)), "global_step": int(global_step),
                "state_dict": C.state_cpu(model), "dataset": dataset, "arm": "B", "seed": C.SEED,
                "model_audit": model_audit, "optimizer_audit": opt_audit,
                "identity": bundle.extra["identity_audit"],
                "train_plan_sha256": plan["plan_sha256"],
                "created_utc": C.utcnow(), "git_commit": C.git_commit(),
            },
        )
        save_predictions(out_dir / "predictions_final.npz", final_predictions)
        final_metrics = {
            "epoch": int(len(curve_rows)),
            "train_rmse_raw": rep_tr["rmse_raw"], "train_mae_raw": rep_tr["mae_raw"],
            "valid_rmse_raw": rep_va["rmse_raw"], "valid_mae_raw": rep_va["mae_raw"],
        }
    else:
        final_metrics = None

    # inference throughput at the best weights
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    t0 = time.perf_counter()
    n_rows = 0
    with torch.no_grad():
        for split, rows, targets in ((va, valid_rows, valid_targets_raw), (tr, train_order_rows, train_targets_raw)):
            n_rows += int(rows.size)
            for start in range(0, rows.size, int(C.RECIPE["eval_batch_size"])):
                chunk = rows[start:start + int(C.RECIPE["eval_batch_size"])]
                model(BD.make_batch(split, chunk, device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = float(time.perf_counter() - t0)

    gpu_peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total_seconds = float(time.perf_counter() - t_start)

    def _last20(key):
        vals = [float(r[key]) for r in curve_rows[-20:]] if curve_rows else []
        return float(np.mean(vals)) if vals else None

    run = {
        "schema": E.SCHEMA_RUN, "round_id": E.ROUND_ID, "protocol_id": E.PROTOCOL_ID,
        "dataset": dataset, "arm": "B", "arm_name": "GINE_ENCODER", "role": "capability boundary reference",
        "seed": C.SEED, "epochs_planned": int(epochs), "epochs_run": int(len(curve_rows)),
        "global_steps_run": int(global_step),
        "status": "COMPLETED" if stop_reason == "COMPLETED" else "INCOMPLETE",
        "stop_reason": stop_reason, "timing_precheck": timing_projection,
        "parameters": int(model_audit["parameter_groups"]["total"]),
        "parameter_groups": model_audit["parameter_groups"],
        "model_audit": model_audit, "optimizer_audit": opt_audit,
        "identity": bundle.extra["identity_audit"],
        "target_scaler": {k: scaler[k] for k in ("mean", "std", "n", "ddof")},
        "train_plan_sha256": plan["plan_sha256"],
        "best_epoch": best_epoch, "best_valid_rmse_raw": float(best_valid) if best_epoch else None,
        "best_metrics": best_metrics, "final_metrics": final_metrics,
        "last20_mean_metrics": {
            f"{s}_{m}_raw": _last20(f"{s}_{m}_raw") for s in ("train", "valid") for m in ("rmse", "mae")
        },
        "train_seconds": float(sum(float(r["train_seconds"]) for r in curve_rows)),
        "eval_seconds": float(sum(float(r["eval_seconds"]) for r in curve_rows)),
        "total_seconds": total_seconds, "gpu_hours": total_seconds / 3600.0,
        "gpu_peak_allocated_bytes": gpu_peak_alloc,
        "inference_rows": int(n_rows), "inference_seconds": inference_seconds,
        "inference_rows_per_second": float(n_rows / inference_seconds) if inference_seconds > 0 else None,
        "checkpoint_best": E.track_rel(out_dir / "checkpoint_best.pt") if (out_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_best_sha256": E.sha256_file(out_dir / "checkpoint_best.pt") if (out_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_final": E.track_rel(out_dir / "checkpoint_final.pt") if (out_dir / "checkpoint_final.pt").is_file() else None,
        "checkpoint_final_sha256": E.sha256_file(out_dir / "checkpoint_final.pt") if (out_dir / "checkpoint_final.pt").is_file() else None,
        "curve": E.track_rel(out_dir / "train_curve.csv"),
        "curve_sha256": E.sha256_file(out_dir / "train_curve.csv") if (out_dir / "train_curve.csv").is_file() else None,
        "predictions_best_sha256": E.sha256_file(out_dir / "predictions_best.npz") if (out_dir / "predictions_best.npz").is_file() else None,
        "recipe": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in C.RECIPE.items()},
            "round_id": E.ROUND_ID, "dataset": dataset, "arm": "B", "arm_name": "GINE_ENCODER",
            "seed": C.SEED, "steps_per_epoch": steps_epoch, "total_steps": total_steps,
            "width": E.OUT_WIDTH, "gine_hidden": E.GINE_HIDDEN, "gine_layers": E.GINE_LAYERS,
        },
        "dropout": float(model.encoder.drop.p),
        "precision": "fp32",
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "grad_clip": float(C.RECIPE["gradient_clip"]), "grad_clip_events": int(clip_events_total),
        "environment": C.environment(), "git_commit": C.git_commit(), "git_dirty": C.git_dirty(),
        "started_utc": started, "finished_utc": C.utcnow(),
        "splits_used": list(E.SPLITS_USED), "test_scored": False, "test_used_for_selection": False,
    }
    E.write_json(out_dir / "run.json", run)
    C.log(f"done B status={run['status']} best_epoch={best_epoch} best_valid={run['best_valid_rmse_raw']} gpu_hours={run['gpu_hours']:.4f}")
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=int(C.RECIPE["epochs"]))
    parser.add_argument("--hard-seconds", type=float, default=E.TRAIN_HARD_SECONDS)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        E.write_json(out_dir / "run.json", {"schema": E.SCHEMA_RUN, "status": "INCOMPLETE",
                                            "stop_reason": "CUDA_UNAVAILABLE", "finished_utc": C.utcnow()})
        return 2
    try:
        bundle = BD.load_bundle()
        sq = BD.load_sq_cache()
        splits = BD.build_splits(bundle, sq)
        categories = BD.category_counts([splits["train"], splits["valid"]])
        if categories != E.EXPECTED_CATEGORIES:
            raise RuntimeError(f"category vocabulary {categories} != historical {E.EXPECTED_CATEGORIES}")
        run = train_unit(bundle, splits, categories, out_dir=out_dir, device=device,
                         epochs=int(args.epochs), hard_seconds=float(args.hard_seconds))
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        E.write_json(out_dir / "run.json", {
            "schema": E.SCHEMA_RUN, "status": "FAILED", "stop_reason": type(exc).__name__,
            "detail": str(exc), "traceback": traceback.format_exc(),
            "environment": C.environment(), "finished_utc": C.utcnow(),
        })
        C.log(f"FAILED B: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
