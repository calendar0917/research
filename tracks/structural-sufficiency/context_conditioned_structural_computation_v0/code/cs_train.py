"""Formal from-scratch training for one CSC v0 unit (dataset x arm, seed 0).

Frozen recipe (PROTOCOL.md section 3.2): batch 128, 360 epochs, AdamW peak
3e-4 -> min 1e-5 by per-step cosine with no warmup, wd 1e-3, betas (0.9,0.999),
eps 1e-8, clip 5, FP32, standardised-target MSE, one parameter group, no
auxiliary loss / pretraining / distillation / EMA / SWA / early stopping.

Every epoch scores the full official Train, Valid and Test splits in eval mode.
Checkpoint selection reads only the Valid RMSE (strictly smaller replaces, ties
keep the earlier epoch).  Test is a DEVELOPMENT_TEST set: fully exposed and
reported, never used for any decision.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cs_data as D  # noqa: E402
import cs_model as M  # noqa: E402

GRAD_WATCH = (
    "bond_proj.weight",
    "blocks.0.fc1.weight",
    "blocks.0.fc2.weight",
    "blocks.1.fc1.weight",
    "blocks.1.fc2.weight",
    "gate.0.weight",
    "gate.2.weight",
)

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "learning_rate",
    "train_loss_step_mean",
    "train_rmse_raw",
    "train_mae_raw",
    "valid_rmse_raw",
    "valid_mae_raw",
    "test_rmse_raw",
    "test_mae_raw",
    "train_seconds",
    "eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "grad_norm_max",
    "grad_norm_mean",
    "clip_events",
    "grad_gate_out",
    "grad_gate_in",
    "grad_block0_fc2",
    "grad_block1_fc2",
    "grad_block1_fc1",
    "grad_bond_proj",
    "delta_rms",
    "Q_rms",
    "rms_dh",
    "rms_h2_minus_h1",
    "R1",
    "R2",
    "best_epoch_so_far",
    "best_valid_rmse_so_far",
    "is_best",
    "rng_sha256",
]


def build_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["peak_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("this round uses exactly one parameter group and one LR for every parameter")
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": 1,
        "n_parameters": int(sum(p.numel() for p in params)),
        "peak_lr": float(C.RECIPE["peak_lr"]),
        "min_lr": float(C.RECIPE["min_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "per_step_cosine_no_warmup",
    }
    return optimizer, audit


def grad_norms(model: torch.nn.Module) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, p in model.named_parameters():
        if name in GRAD_WATCH:
            out[name] = None if p.grad is None else float(p.grad.detach().norm().item())
    return out


def diagnostic_batch(bundle: Any, device, *, n_rows: int = 64) -> dict[str, Any]:
    order, _ids, _y = D.split_order(bundle, "valid")
    rows = order[: int(n_rows)]
    return D.make_batch(bundle, "valid", rows, device)


def path_of(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / name


def save_predictions(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v) for k, v in payload.items()})
    os.replace(tmp, path)
    return C.sha256_file(path)


def train_unit(
    bundle: Any,
    arm: str,
    *,
    out_dir: Path,
    device,
    epochs: int | None = None,
    soft_seconds: float = 3000.0,
    hard_seconds: float = 3600.0,
    resume: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(bundle.dataset)
    epochs = int(C.RECIPE["epochs"] if epochs is None else epochs)
    steps_epoch = C.steps_per_epoch(dataset)
    total_steps = epochs * steps_epoch
    t_start = time.perf_counter()
    started = C.utcnow()
    C.set_seeds(C.SEED)

    model, model_audit = M.build_csc(bundle.spec, bundle.norm, arm=arm, seed=C.SEED, device=device)
    model_audit["initial_state_sha256"] = C.state_fingerprint(C.state_cpu(model))
    optimizer, opt_audit = build_optimizer(model)
    plan = D.batch_order_plan(dataset, epochs)
    C.write_json(out_dir / "training_order.json", {k: v for k, v in plan.items() if k != "plan"} | {"first_batch_rows": plan["plan"][0][: int(C.RECIPE["batch_size"])].tolist()})

    diag = diagnostic_batch(bundle, device)
    curve_rows: list[dict[str, Any]] = []
    gate_history: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_valid = math.inf
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    clip_events_total = 0
    rolling: list[dict[str, Any]] = []
    global_step = 0
    stop_reason = "RUNNING"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _eval_all() -> dict[str, dict[str, Any]]:
        out = {}
        for split in C.SPLITS:
            t0 = time.perf_counter()
            rep = D.evaluate(bundle, model, split, device)
            rep["seconds"] = float(time.perf_counter() - t0)
            out[split] = rep
        return out

    for epoch in range(1, epochs + 1):
        if time.perf_counter() - t_start > float(hard_seconds):
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        order = plan["plan"][epoch - 1]
        losses: list[float] = []
        gnorm: list[float] = []
        gwatch: dict[str, list[float]] = {name: [] for name in GRAD_WATCH}
        clip_events = 0
        aborted = False
        for start in range(0, order.size, int(C.RECIPE["batch_size"])):
            if time.perf_counter() - t_start > float(hard_seconds):
                aborted = True
                break
            rows = order[start:start + int(C.RECIPE["batch_size"])]
            batch = D.make_batch(bundle, "train", rows, device)
            y = torch.as_tensor(
                np.asarray((bundle.targets_raw["train"][rows] - bundle.scaler["mean"]) / bundle.scaler["std"], dtype=np.float32),
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
            for name, value in grad_norms(model).items():
                if value is not None:
                    gwatch[name].append(value)
            optimizer.step()
            losses.append(float(loss.detach()))
            gnorm.append(g)
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            break
        train_seconds = float(time.perf_counter() - t_epoch)

        t_eval = time.perf_counter()
        reports = _eval_all()
        eval_seconds = float(time.perf_counter() - t_eval)
        valid_rmse = float(reports["valid"]["rmse_raw"])
        is_best = valid_rmse < best_valid
        if is_best:
            best_valid = valid_rmse
            best_epoch = int(epoch)
            best_state = C.state_cpu(model)
            best_metrics = {split: {k: reports[split][k] for k in ("rmse_raw", "mae_raw")} for split in C.SPLITS}
            C.atomic_torch_save(
                out_dir / "checkpoint_best.pt",
                {
                    "schema": C.SCHEMA_CHECKPOINT,
                    "kind": "best_valid_rmse",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "state_dict": best_state,
                    "dataset": dataset,
                    "arm": arm,
                    "seed": C.SEED,
                    "model_audit": model_audit,
                    "recipe": C.recipe(dataset, arm),
                    "identity": bundle.extra["identity_audit"],
                    "train_plan_sha256": plan["plan_sha256"],
                    "valid_rmse_raw": valid_rmse,
                    "created_utc": C.utcnow(),
                    "git_commit": C.git_commit(),
                },
            )
            save_predictions(
                out_dir / "predictions_best.npz",
                {
                    "valid_row_ids": reports["valid"]["row_ids"],
                    "valid_pred_raw": reports["valid"]["pred_raw"],
                    "valid_y_raw": reports["valid"]["y_raw"],
                    "test_row_ids": reports["test"]["row_ids"],
                    "test_pred_raw": reports["test"]["pred_raw"],
                    "test_y_raw": reports["test"]["y_raw"],
                    "train_row_ids": reports["train"]["row_ids"],
                    "train_pred_raw": reports["train"]["pred_raw"],
                    "train_y_raw": reports["train"]["y_raw"],
                    "epoch": np.asarray([epoch]),
                },
            )
            C.write_json(
                out_dir / "best_checkpoint_eval.json",
                {
                    "epoch": int(epoch),
                    "valid_rmse_raw": valid_rmse,
                    "selection_metric": "valid_rmse_raw",
                    **{f"{split}_rmse_raw": best_metrics[split]["rmse_raw"] for split in C.SPLITS},
                    **{f"{split}_mae_raw": best_metrics[split]["mae_raw"] for split in C.SPLITS},
                },
            )
        rolling.append(
            {
                "epoch": int(epoch),
                "valid_pred_raw": reports["valid"]["pred_raw"],
                "test_pred_raw": reports["test"]["pred_raw"],
                "valid_rmse_raw": reports["valid"]["rmse_raw"],
                "test_rmse_raw": reports["test"]["rmse_raw"],
                "valid_mae_raw": reports["valid"]["mae_raw"],
                "test_mae_raw": reports["test"]["mae_raw"],
                "train_rmse_raw": reports["train"]["rmse_raw"],
                "train_mae_raw": reports["train"]["mae_raw"],
            }
        )
        rolling = rolling[-20:]

        usage = M.usage_metrics(model, diag)
        gate_hist = M.gate_statistics(model, diag)
        gate_hist["epoch"] = int(epoch)
        gate_history.append(gate_hist)
        clip_events_total += int(clip_events)

        def _mean(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if len(values) else None

        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss_step_mean": float(np.mean(losses)) if losses else None,
            "train_rmse_raw": reports["train"]["rmse_raw"],
            "train_mae_raw": reports["train"]["mae_raw"],
            "valid_rmse_raw": valid_rmse,
            "valid_mae_raw": reports["valid"]["mae_raw"],
            "test_rmse_raw": reports["test"]["rmse_raw"],
            "test_mae_raw": reports["test"]["mae_raw"],
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "grad_norm_max": float(max(gnorm)) if gnorm else None,
            "grad_norm_mean": _mean(gnorm),
            "clip_events": int(clip_events),
            "grad_gate_out": _mean(gwatch.get("gate.2.weight", [])),
            "grad_gate_in": _mean(gwatch.get("gate.0.weight", [])),
            "grad_block0_fc2": _mean(gwatch.get("blocks.0.fc2.weight", [])),
            "grad_block1_fc2": _mean(gwatch.get("blocks.1.fc2.weight", [])),
            "grad_block1_fc1": _mean(gwatch.get("blocks.1.fc1.weight", [])),
            "grad_bond_proj": _mean(gwatch.get("bond_proj.weight", [])),
            "delta_rms": gate_hist.get("delta_rms"),
            "Q_rms": gate_hist.get("Q_rms"),
            "rms_dh": gate_hist.get("rms_dh"),
            "rms_h2_minus_h1": usage.get("rms_h2_minus_h1"),
            "R1": usage.get("R1"),
            "R2": usage.get("R2"),
            "best_epoch_so_far": best_epoch,
            "best_valid_rmse_so_far": float(best_valid) if best_epoch else None,
            "is_best": bool(is_best),
            "rng_sha256": C.rng_hashes()["torch_cpu"],
        }
        curve_rows.append(row)
        C.write_csv(out_dir / "train_curve.csv", curve_rows, CURVE_FIELDS)
        C.write_json(out_dir / "gate_history.json", gate_history)
        if epoch % 20 == 1 or is_best or epoch == epochs:
            C.log(
                f"[{dataset} {arm}] ep {epoch}/{epochs} train {row['train_rmse_raw']:.4f} "
                f"valid {valid_rmse:.4f} test {row['test_rmse_raw']:.4f} "
                f"delta {row['delta_rms']} Q {row['Q_rms']} ggate {row['grad_gate_out']} "
                f"{'*best*' if is_best else ''} ({row['epoch_seconds']:.1f}s)"
            )

    stop_reason = "COMPLETED" if len(curve_rows) == epochs else stop_reason

    # ---- predictions of the final epoch and the last-20 average
    final_pred = None
    final_checkpoint_sha = None
    if curve_rows:
        model.eval()
        reports = _eval_all()
        final_pred = reports
        C.atomic_torch_save(
            out_dir / "checkpoint_final.pt",
            {
                "schema": C.SCHEMA_CHECKPOINT,
                "kind": "final_epoch",
                "epoch": int(len(curve_rows)),
                "global_step": int(global_step),
                "state_dict": C.state_cpu(model),
                "dataset": dataset,
                "arm": arm,
                "seed": C.SEED,
                "model_audit": model_audit,
                "recipe": C.recipe(dataset, arm),
                "identity": bundle.extra["identity_audit"],
                "train_plan_sha256": plan["plan_sha256"],
                "created_utc": C.utcnow(),
                "git_commit": C.git_commit(),
            },
        )
        final_checkpoint_sha = C.sha256_file(out_dir / "checkpoint_final.pt")
        save_predictions(
            out_dir / "predictions_final.npz",
            {
                "valid_row_ids": reports["valid"]["row_ids"],
                "valid_pred_raw": reports["valid"]["pred_raw"],
                "valid_y_raw": reports["valid"]["y_raw"],
                "test_row_ids": reports["test"]["row_ids"],
                "test_pred_raw": reports["test"]["pred_raw"],
                "test_y_raw": reports["test"]["y_raw"],
                "train_row_ids": reports["train"]["row_ids"],
                "train_pred_raw": reports["train"]["pred_raw"],
                "train_y_raw": reports["train"]["y_raw"],
                "epoch": np.asarray([len(curve_rows)]),
            },
        )
        last20 = rolling[-20:]
        save_predictions(
            out_dir / "predictions_last20.npz",
            {
                "valid_row_ids": reports["valid"]["row_ids"],
                "valid_pred_raw": np.mean([r["valid_pred_raw"] for r in last20], axis=0),
                "valid_y_raw": reports["valid"]["y_raw"],
                "test_row_ids": reports["test"]["row_ids"],
                "test_pred_raw": np.mean([r["test_pred_raw"] for r in last20], axis=0),
                "test_y_raw": reports["test"]["y_raw"],
                "epochs": np.asarray([r["epoch"] for r in last20]),
            },
        )

    # ---- inference throughput (best checkpoint weights when available)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    t0 = time.perf_counter()
    n_rows = 0
    with torch.no_grad():
        for split in ("valid", "test"):
            order, _ids, _y = D.split_order(bundle, split)
            n_rows += int(order.size)
            for start in range(0, order.size, int(C.RECIPE["eval_batch_size"])):
                chunk = order[start:start + int(C.RECIPE["eval_batch_size"])]
                model(D.make_batch(bundle, split, chunk, device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = float(time.perf_counter() - t0)

    gpu_peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total_seconds = float(time.perf_counter() - t_start)

    def _mean_of(key: str) -> float | None:
        vals = [float(r[key]) for r in curve_rows[-20:]] if curve_rows else []
        return float(np.mean(vals)) if vals else None

    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "role": C.ARMS[arm]["role"],
        "seed": C.SEED,
        "epochs_planned": int(epochs),
        "epochs_run": int(len(curve_rows)),
        "global_steps_run": int(global_step),
        "status": "COMPLETED" if stop_reason == "COMPLETED" else "INCOMPLETE",
        "stop_reason": stop_reason,
        "parameters": int(model_audit["n_parameters"]),
        "n_parameters_new": int(model_audit["n_parameters_new"]),
        "model_audit": model_audit,
        "optimizer_audit": opt_audit,
        "identity": bundle.extra["identity_audit"],
        "target_scaler": {k: bundle.scaler[k] for k in ("mean", "std", "n", "ddof")},
        "train_plan_sha256": plan["plan_sha256"],
        "best_epoch": best_epoch,
        "best_valid_rmse_raw": float(best_valid) if best_epoch else None,
        "best_metrics": best_metrics,
        "final_metrics": {
            "epoch": int(len(curve_rows)),
            **{f"{split}_{m}_raw": (final_pred[split][f"{m}_raw"] if final_pred else None) for split in C.SPLITS for m in ("rmse", "mae")},
        },
        "last20_mean_metrics": {
            f"{split}_{m}_raw": _mean_of(f"{split}_{m}_raw")
            for split in C.SPLITS
            for m in ("rmse", "mae")
        },
        "train_seconds": float(sum(float(r["train_seconds"]) for r in curve_rows)),
        "eval_seconds": float(sum(float(r["eval_seconds"]) for r in curve_rows)),
        "total_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "gpu_peak_allocated_bytes": gpu_peak_alloc,
        "inference_rows": int(n_rows),
        "inference_seconds": inference_seconds,
        "inference_rows_per_second": float(n_rows / inference_seconds) if inference_seconds > 0 else None,
        "checkpoint_best": C.track_rel(out_dir / "checkpoint_best.pt") if (out_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_best_sha256": C.sha256_file(out_dir / "checkpoint_best.pt") if (out_dir / "checkpoint_best.pt").is_file() else None,
        "checkpoint_final": C.track_rel(out_dir / "checkpoint_final.pt") if (out_dir / "checkpoint_final.pt").is_file() else None,
        "checkpoint_final_sha256": final_checkpoint_sha,
        "curve": C.track_rel(out_dir / "train_curve.csv"),
        "curve_sha256": C.sha256_file(out_dir / "train_curve.csv") if (out_dir / "train_curve.csv").is_file() else None,
        "recipe": C.recipe(dataset, arm),
        "dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "grad_clip": float(C.RECIPE["gradient_clip"]),
        "grad_clip_events": int(clip_events_total),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "started_utc": started,
        "finished_utc": C.utcnow(),
        "test_role": C.TEST_ROLE,
        "test_used_for_selection": False,
        "test_scored_every_epoch": True,
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done {dataset} {arm} status={run['status']} best_epoch={best_epoch} "
        f"best_valid={run['best_valid_rmse_raw']} gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    parser.add_argument("--arm", required=True, choices=list(C.ARMS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=int(C.RECIPE["epochs"]))
    parser.add_argument("--soft-seconds", type=float, default=3000.0)
    parser.add_argument("--hard-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(out_dir / "run.json", {"schema": C.SCHEMA_RUN, "dataset": args.dataset, "arm": args.arm,
                                            "status": "INCOMPLETE", "stop_reason": "CUDA_UNAVAILABLE",
                                            "finished_utc": C.utcnow()})
        return 2
    try:
        bundle = D.load_bundle(args.dataset)
        run = train_unit(
            bundle,
            str(args.arm),
            out_dir=out_dir,
            device=device,
            epochs=int(args.epochs),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "dataset": args.dataset,
                "arm": str(args.arm),
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED {args.dataset} {args.arm}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
