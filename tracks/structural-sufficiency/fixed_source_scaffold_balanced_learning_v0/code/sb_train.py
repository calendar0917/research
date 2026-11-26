"""Formal training of the single `SCAFFOLD_BALANCED` arm (ogbg-mollipo, seed 0, 360 epochs).

The model, the optimiser, the batch order and the whole recipe are the frozen
CSC v0 / FIXED1 objects, imported read-only.  The **only** difference from the
historical FIXED1 run is the loss: the plain standardised-target MSE is replaced
by the fixed weighted MSE of ``PROTOCOL.md`` section 5,

    weight_i = N_train / (K_train * n_scaffold_i)
    loss     = mean_batch(weight_i * per_molecule_mse)

with no re-normalisation inside a batch, no clipping, no sqrt, no resampling and
no dynamic update.  Every epoch scores the full official Train and Valid splits in
eval mode; the Test split is never touched (the data layer refuses it).
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sb_common as C

import torch  # noqa: E402

import cs_model as M  # noqa: E402  (frozen CSC v0 architecture, read-only)
import sb_data as D  # noqa: E402
import sb_scaffold as S  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "learning_rate",
    "train_loss_step_mean",
    "train_weighted_mse_trainmode",
    "train_mse_unweighted_trainmode",
    "train_rmse_raw",
    "train_mae_raw",
    "valid_rmse_raw",
    "valid_mae_raw",
    "grad_norm_mean",
    "grad_norm_max",
    "clip_events",
    "n_steps",
    "train_seconds",
    "eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "best_epoch_so_far",
    "best_valid_rmse_so_far",
    "is_best",
]


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``mean(w_i * (pred_i - y_i)^2)`` — no in-batch re-normalisation."""
    se = (pred - target) ** 2
    return (weight * se).mean()


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
        raise RuntimeError("exactly one parameter group / one LR (frozen recipe)")
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
        "loss": "weighted MSE on standardised targets (fixed per-molecule weights)",
    }
    return optimizer, audit


def save_predictions(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{k: np.asarray(v) for k, v in payload.items()})
    tmp.replace(path)
    return C.sha256_file(path)


def train_unit(
    bundle: Any,
    *,
    out_dir: Path,
    device,
    official: Mapping[str, Any],
    scaffold_by_row: Mapping[int, str],
    k_train: int,
    epochs: int = C.EPOCHS,
    hard_seconds: float = 2100.0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(epochs)
    steps_epoch = C.steps_per_epoch(C.DATASET)
    total_steps = epochs * steps_epoch
    t_start = time.perf_counter()
    started = C.utcnow()
    C.set_seeds(C.SEED)

    model, model_audit = M.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device=device)
    if model.gate is not None:
        raise RuntimeError("arm A must not have a Gate")
    initial_state_sha = C.state_fingerprint(C.state_cpu(model))
    model_audit["initial_state_sha256"] = initial_state_sha
    if initial_state_sha != C.HIST["initial_state_sha256"]:
        raise RuntimeError(
            f"initial state {initial_state_sha} != historical FIXED1 {C.HIST['initial_state_sha256']}"
        )
    optimizer, opt_audit = build_optimizer(model)

    wpack = D.train_weights(bundle, official, scaffold_by_row, k_train=int(k_train))
    weights = wpack["weights"]
    if abs(float(weights.mean()) - 1.0) > 1e-12:
        raise RuntimeError(f"weight mean {weights.mean()} != 1")

    plan = D.batch_order_plan(epochs)
    C.write_json(
        out_dir / "training_order.json",
        {k: v for k, v in plan.items() if k != "plan"}
        | {"first_batch_rows": plan["plan"][0][: int(C.RECIPE["batch_size"])].tolist()},
    )

    curve_rows: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_valid = math.inf
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    clip_events_total = 0
    steps_total = 0
    rolling: list[dict[str, Any]] = []
    global_step = 0
    stop_reason = "RUNNING"

    def _eval(split: str) -> dict[str, Any]:
        return D.evaluate(bundle, model, split, device)

    for epoch in range(1, epochs + 1):
        if time.perf_counter() - t_start > float(hard_seconds):
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        order = plan["plan"][epoch - 1]
        if order.size != C.N_TRAIN or np.unique(order).size != C.N_TRAIN:
            raise RuntimeError(f"epoch {epoch}: order is not a permutation of the Train rows")
        losses: list[float] = []
        gnorm: list[float] = []
        clip_events = 0
        step_wse = 0.0     # sum_i w_i * se_i over the epoch (train mode)
        step_se = 0.0      # sum_i se_i over the epoch (train mode)
        step_w = 0.0       # sum_i w_i over the epoch
        n_rows_seen = 0
        aborted = False
        for start in range(0, order.size, int(C.RECIPE["batch_size"])):
            if time.perf_counter() - t_start > float(hard_seconds):
                aborted = True
                break
            rows = order[start:start + int(C.RECIPE["batch_size"])]
            batch = D.make_batch(bundle, "train", rows, device)
            y = torch.as_tensor(
                np.asarray(
                    (bundle.targets_raw["train"][rows] - bundle.scaler["mean"]) / bundle.scaler["std"],
                    dtype=np.float32,
                ),
                device=device,
            )
            w = torch.as_tensor(weights[rows].astype(np.float32), device=device)
            lr = C.lr_at(global_step, total_steps)
            optimizer.param_groups[0]["lr"] = float(lr)
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch).view(-1)
            se = (pred - y) ** 2
            loss = (w * se).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}")
            loss.backward()
            g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"])))
            if not math.isfinite(g):
                raise RuntimeError(f"non-finite gradient norm at step {global_step}")
            clip_events += int(g > float(C.RECIPE["gradient_clip"]))
            optimizer.step()
            det = se.detach().double()
            wd = w.detach().double()
            step_wse += float((wd * det).sum())
            step_se += float(det.sum())
            step_w += float(wd.sum())
            n_rows_seen += int(rows.size)
            losses.append(float(loss.detach()))
            gnorm.append(g)
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            break
        # weights enter the loss as float32 tensors, so the epoch sum is 1 +/- 3e-8 per molecule
        if n_rows_seen != C.N_TRAIN or len(losses) != steps_epoch or abs(step_w - C.N_TRAIN) > 1e-3:
            raise RuntimeError(
                f"epoch {epoch}: rows {n_rows_seen}/{C.N_TRAIN} steps {len(losses)}/{steps_epoch} "
                f"weight_sum {step_w!r}"
            )
        train_seconds = float(time.perf_counter() - t_epoch)

        t_eval = time.perf_counter()
        reports = {split: _eval(split) for split in C.ALLOWED_SPLITS}
        eval_seconds = float(time.perf_counter() - t_eval)
        valid_rmse = float(reports["valid"]["rmse_raw"])
        is_best = valid_rmse < best_valid
        if is_best:
            best_valid = valid_rmse
            best_epoch = int(epoch)
            best_state = C.state_cpu(model)
            best_metrics = {s: {k: reports[s][k] for k in ("rmse_raw", "mae_raw")} for s in C.ALLOWED_SPLITS}
            C.atomic_torch_save(
                out_dir / "checkpoint_best.pt",
                {
                    "schema": C.SCHEMA_CHECKPOINT,
                    "kind": "best_valid_rmse",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "state_dict": best_state,
                    "dataset": C.DATASET,
                    "arm": "SCAFFOLD_BALANCED",
                    "seed": C.SEED,
                    "model_audit": model_audit,
                    "recipe": C.recipe_of_record(k_train=int(k_train), weight_sha256=wpack["sha256"]),
                    "identity": bundle.extra["identity_audit"],
                    "train_plan_sha256": plan["plan_sha256"],
                    "weight_sha256": wpack["sha256"],
                    "valid_rmse_raw": valid_rmse,
                    "created_utc": C.utcnow(),
                    "git_commit": C.git_commit(),
                    "test_scored": False,
                },
            )
            save_predictions(
                out_dir / "predictions_best.npz",
                {
                    "valid_row_ids": reports["valid"]["row_ids"],
                    "valid_pred_raw": reports["valid"]["pred_raw"],
                    "valid_y_raw": reports["valid"]["y_raw"],
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
                    **{f"{s}_{m}_raw": best_metrics[s][f"{m}_raw"] for s in C.ALLOWED_SPLITS for m in ("rmse", "mae")},
                },
            )
        rolling.append(
            {
                "epoch": int(epoch),
                "valid_pred_raw": reports["valid"]["pred_raw"],
                "valid_rmse_raw": reports["valid"]["rmse_raw"],
                "valid_mae_raw": reports["valid"]["mae_raw"],
            }
        )
        rolling = rolling[-20:]
        clip_events_total += int(clip_events)
        steps_total += len(losses)

        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss_step_mean": float(np.mean(losses)),
            "train_weighted_mse_trainmode": float(step_wse / C.N_TRAIN),
            "train_mse_unweighted_trainmode": float(step_se / C.N_TRAIN),
            "train_rmse_raw": reports["train"]["rmse_raw"],
            "train_mae_raw": reports["train"]["mae_raw"],
            "valid_rmse_raw": valid_rmse,
            "valid_mae_raw": reports["valid"]["mae_raw"],
            "grad_norm_mean": float(np.mean(gnorm)) if gnorm else None,
            "grad_norm_max": float(max(gnorm)) if gnorm else None,
            "clip_events": int(clip_events),
            "n_steps": int(len(losses)),
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "best_epoch_so_far": best_epoch,
            "best_valid_rmse_so_far": float(best_valid) if best_epoch else None,
            "is_best": bool(is_best),
        }
        curve_rows.append(row)
        C.write_csv(out_dir / "train_curve.csv", curve_rows, CURVE_FIELDS)
        if epoch % 20 == 1 or is_best or epoch == epochs:
            C.log(
                f"[SCAFFOLD_BALANCED] ep {epoch}/{epochs} train {row['train_rmse_raw']:.4f} "
                f"valid {valid_rmse:.4f} best {row['best_valid_rmse_so_far']:.4f} "
                f"wloss {row['train_loss_step_mean']:.4f} clip {row['clip_events']} "
                f"{'*best*' if is_best else ''} ({row['epoch_seconds']:.1f}s)"
            )

    stop_reason = "COMPLETED" if len(curve_rows) == epochs else stop_reason

    # ---- final epoch and last-20 predictions
    final_pred = None
    final_checkpoint_sha = None
    if curve_rows:
        model.eval()
        reports = {split: _eval(split) for split in C.ALLOWED_SPLITS}
        final_pred = reports
        C.atomic_torch_save(
            out_dir / "checkpoint_final.pt",
            {
                "schema": C.SCHEMA_CHECKPOINT,
                "kind": "final_epoch",
                "epoch": int(len(curve_rows)),
                "global_step": int(global_step),
                "state_dict": C.state_cpu(model),
                "dataset": C.DATASET,
                "arm": "SCAFFOLD_BALANCED",
                "seed": C.SEED,
                "model_audit": model_audit,
                "recipe": C.recipe_of_record(k_train=int(k_train), weight_sha256=wpack["sha256"]),
                "identity": bundle.extra["identity_audit"],
                "train_plan_sha256": plan["plan_sha256"],
                "weight_sha256": wpack["sha256"],
                "created_utc": C.utcnow(),
                "git_commit": C.git_commit(),
                "test_scored": False,
            },
        )
        final_checkpoint_sha = C.sha256_file(out_dir / "checkpoint_final.pt")
        save_predictions(
            out_dir / "predictions_final.npz",
            {
                "valid_row_ids": reports["valid"]["row_ids"],
                "valid_pred_raw": reports["valid"]["pred_raw"],
                "valid_y_raw": reports["valid"]["y_raw"],
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
                "epochs": np.asarray([r["epoch"] for r in last20]),
            },
        )

    # ---- inference throughput (best checkpoint when available)
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    t0 = time.perf_counter()
    n_rows = 0
    with torch.no_grad():
        for split in C.ALLOWED_SPLITS:
            order, _ids, _y = D.split_order(bundle, split)
            n_rows += int(order.size)
            for start in range(0, order.size, int(C.RECIPE["eval_batch_size"])):
                chunk = order[start:start + int(C.RECIPE["eval_batch_size"])]
                model(D.make_batch(bundle, split, chunk, device))
    inference_seconds = float(time.perf_counter() - t0)
    total_seconds = float(time.perf_counter() - t_start)

    def _mean_of(key: str) -> float | None:
        vals = [float(r[key]) for r in curve_rows[-20:] if r.get(key) is not None] if curve_rows else []
        return float(np.mean(vals)) if vals else None

    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "dataset": C.DATASET,
        "arm": "SCAFFOLD_BALANCED",
        "seed": C.SEED,
        "epochs_planned": int(epochs),
        "epochs_run": int(len(curve_rows)),
        "global_steps_run": int(global_step),
        "status": "COMPLETED" if stop_reason == "COMPLETED" else "INCOMPLETE",
        "stop_reason": stop_reason,
        "parameters": int(model_audit["n_parameters"]),
        "model_audit": model_audit,
        "optimizer_audit": opt_audit,
        "identity": bundle.extra["identity_audit"],
        "target_scaler": {k: bundle.scaler[k] for k in ("mean", "std", "n", "ddof")},
        "train_plan_sha256": plan["plan_sha256"],
        "train_plan_matches_historical": bool(plan["matches_historical_fixed1_plan"]),
        "weight_sha256": wpack["sha256"],
        "weight_rule": C.WEIGHT_RULE,
        "k_train": int(k_train),
        "best_epoch": best_epoch,
        "best_valid_rmse_raw": float(best_valid) if best_epoch else None,
        "best_metrics": best_metrics,
        "final_metrics": {
            "epoch": int(len(curve_rows)),
            **{
                f"{s}_{m}_raw": (final_pred[s][f"{m}_raw"] if final_pred else None)
                for s in C.ALLOWED_SPLITS
                for m in ("rmse", "mae")
            },
        },
        "last20_mean_metrics": {
            f"{s}_{m}_raw": _mean_of(f"{s}_{m}_raw") for s in C.ALLOWED_SPLITS for m in ("rmse", "mae")
        },
        "train_seconds": float(sum(float(r["train_seconds"]) for r in curve_rows)),
        "eval_seconds": float(sum(float(r["eval_seconds"]) for r in curve_rows)),
        "total_seconds": total_seconds,
        "wall_seconds": total_seconds,
        "gpu_hours": 0.0,
        "device": str(device),
        "inference_rows": int(n_rows),
        "inference_seconds": inference_seconds,
        "inference_rows_per_second": float(n_rows / inference_seconds) if inference_seconds > 0 else None,
        "checkpoint_best": C.rel(out_dir / "checkpoint_best.pt"),
        "checkpoint_best_sha256": C.sha256_file(out_dir / "checkpoint_best.pt"),
        "checkpoint_final_sha256": final_checkpoint_sha,
        "curve": C.rel(out_dir / "train_curve.csv"),
        "curve_sha256": C.sha256_file(out_dir / "train_curve.csv"),
        "grad_clip": float(C.RECIPE["gradient_clip"]),
        "grad_clip_events": int(clip_events_total),
        "grad_clip_fraction": float(clip_events_total / steps_total) if steps_total else None,
        "steps_run": int(steps_total),
        "recipe": C.recipe_of_record(k_train=int(k_train), weight_sha256=wpack["sha256"]),
        "dropout": float(getattr(model, "drop").p),
        "precision": "fp32",
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "started_utc": started,
        "finished_utc": C.utcnow(),
        "test_used": False,
        "test_scored": False,
        "valid_metrics_unweighted": True,
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"done status={run['status']} best_epoch={best_epoch} best_valid={run['best_valid_rmse_raw']} "
        f"wall={total_seconds:.1f}s"
    )
    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(C.RUNS_DIR / "scaffold_balanced_seed0"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--hard-seconds", type=float, default=2100.0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    try:
        frozen = C.read_json(C.CONFIG_DIR / "experiment.json")
        official = S.load_official_rows()
        bundle, _audit = D.load_bundle()
        scaffold_by_row = S.scaffold_of_rows(official["smiles"], np.concatenate([official["train_rows"], official["valid_rows"]]))
        k_train = len(set(scaffold_by_row[int(r)] for r in official["train_rows"]))
        if k_train != int(frozen["weights"]["k_train"]):
            raise RuntimeError(f"K_train {k_train} != frozen {frozen['weights']['k_train']}")
        run = train_unit(
            bundle,
            out_dir=out_dir,
            device=device,
            official=official,
            scaffold_by_row=scaffold_by_row,
            k_train=k_train,
            epochs=int(args.epochs),
            hard_seconds=float(args.hard_seconds),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "dataset": C.DATASET,
                "arm": "SCAFFOLD_BALANCED",
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"FAILED: {exc}")
        raise
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
