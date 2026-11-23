"""FGCR-v0 single-seed training / evaluation entry point.

Formal run:
    uv run python .../code/fgcr_train.py --run-id seed0

GPU cost preflight (<=2 epochs, same seed and recipe):
    uv run python .../code/fgcr_train.py --preflight --run-id preflight_seed0

Only Valid RMSE selects the checkpoint and stops training.  Test metrics are
computed and stored every epoch (explicitly authorized for this round) but are
never used for gradients, standardization, selection or stopping.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import fgcr_common as C
import fgcr_data as D
import fgcr_inference as I
import fgcr_model as M

METRIC_COLUMNS = [
    "epoch", "lr", "train_loss", "grad_norm",
    "train_rmse", "train_mae",
    "valid_rmse", "valid_mae",
    "test_rmse", "test_mae",
    "train_split_eval_seconds", "valid_split_eval_seconds",
    "test_split_eval_seconds", "train_step_seconds", "epoch_seconds",
    "peak_vram_gb",
]


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float,
          minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(peak) * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return float(minimum) + 0.5 * (float(peak) - float(minimum)) * (
        1.0 + math.cos(math.pi * progress))


def _resolve_out(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = C.REPO_ROOT / path
    return path


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def _batches(rows: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, int(rows.size), int(batch_size)):
        yield rows[start:start + int(batch_size)]


def evaluate(model: M.FGCRModel, bundle: D.DataBundle, plan: I.GlobalPlan,
             rows: np.ndarray, *, batch_size: int, device: str
             ) -> dict[str, Any]:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    pred_std = np.zeros(rows.size, dtype=np.float64)
    t0 = time.perf_counter()
    offset = 0
    with torch.no_grad():
        for part in _batches(rows, batch_size):
            batch = D.collate_rows(bundle, part.tolist(), plan, device)
            out = model(batch)
            pred_std[offset:offset + part.size] = out["yhat_std"].double().cpu().numpy()
            offset += int(part.size)
    _sync(device)
    seconds = time.perf_counter() - t0
    target = bundle.labels[rows].astype(np.float64)
    metrics = C.metrics_raw(pred_std, target, bundle.scaler)
    return {
        "pred_std": pred_std,
        "pred_raw": pred_std * bundle.scaler["std"] + bundle.scaler["mean"],
        "target_raw": target,
        "row_ids": rows.copy(),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "seconds": seconds,
    }


def _save_best(path: Path, model: M.FGCRModel, cfg: Mapping[str, Any],
               bundle: D.DataBundle, plan_stats: Mapping[str, Any],
               epoch: int, metrics: Mapping[str, Any],
               predictions: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": C.jsonable(cfg),
        "scaler": dict(bundle.scaler),
        "atom_dims": list(bundle.atom_dims),
        "bond_dims": list(bundle.bond_dims),
        "epoch": int(epoch),
        "metrics": C.jsonable(metrics),
        "plan_stats": C.jsonable(plan_stats),
        "prediction_keys": sorted(predictions.keys()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _save_predictions(path: Path, predictions: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split, values in predictions.items():
        arrays[f"{split}_row_ids"] = np.asarray(values["row_ids"], dtype=np.int64)
        arrays[f"{split}_y_true"] = np.asarray(values["target_raw"], dtype=np.float64)
        arrays[f"{split}_y_pred"] = np.asarray(values["pred_raw"], dtype=np.float64)
        arrays[f"{split}_y_pred_std"] = np.asarray(values["pred_std"], dtype=np.float64)
        arrays[f"{split}_rmse"] = np.asarray(values["rmse"], dtype=np.float64)
        arrays[f"{split}_mae"] = np.asarray(values["mae"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _verdict(valid_rmse: float, cfg: Mapping[str, Any]) -> tuple[str, str]:
    v = cfg["verdict"]
    if valid_rmse <= float(v["predictive_candidate_max_valid_rmse"]):
        return "PREDICTIVE_CANDIDATE", "CONTINUE_DISCUSSION"
    if valid_rmse <= float(v["weak_signal_max_valid_rmse"]):
        return "WEAK_PREDICTIVE_SIGNAL", "STOP_THIS_IMPLEMENTATION"
    return "NO_COMPETITIVE_SIGNAL", "STOP_THIS_IMPLEMENTATION"


def _write_failure(out_dir: Path, payload: Mapping[str, Any]) -> None:
    C.write_json(out_dir / "run.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(C.CONFIG_PATH))
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="preflight only; formal epochs are frozen")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    if int(args.seed) != int(cfg["recipe"]["seed"]):
        raise SystemExit(f"seed is frozen to {cfg['recipe']['seed']}")
    if args.max_epochs is not None and not args.preflight:
        raise SystemExit("--max-epochs is only allowed with --preflight")

    run_id = args.run_id or ("preflight_seed0" if args.preflight else "seed0")
    out_dir = _resolve_out(
        args.out,
        C.RESULTS_DIR / ("preflight_seed0" if args.preflight else "seed0"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "metrics.csv").exists() or (out_dir / "run.json").exists():
        raise SystemExit(f"refusing to overwrite an existing run in {out_dir}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()

    C.seed_everything(int(args.seed))
    t_process = time.perf_counter()

    log_path = out_dir / "train.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"run_id={run_id} preflight={args.preflight} seed={args.seed} "
        f"device={device} commit={C.git_commit()} dirty={C.git_dirty()}")

    bundle = D.load_bundle()
    log(f"data cache_sha256={bundle.cache_sha256} sizes="
        f"{[int(bundle.train_idx.size), int(bundle.valid_idx.size), int(bundle.test_idx.size)]} "
        f"scaler={bundle.scaler}")
    log(f"data audit={C.dumps_json(bundle.audit)}")

    # One exact junction plan for the whole corpus.
    graphs_all = D.graphs_for_rows(bundle, range(bundle.n_graphs))
    graph_pairs = [(int(g["n_atoms"]),
                    list(zip(g["bond_u"].tolist(), g["bond_v"].tolist())))
                   for g in graphs_all]
    t_plan = time.perf_counter()
    plan = I.build_corpus_plan(graph_pairs, smax=int(cfg["inference"]["smax"]))
    plan_seconds = time.perf_counter() - t_plan
    plan_stats = I.plan_stats(plan)
    log(f"exact plan {C.dumps_json(plan_stats)} build_seconds={plan_seconds:.3f}")
    if plan_stats["width_max"] > 8:
        payload = {
            "status": "BLOCKED",
            "reason": f"induced width {plan_stats['width_max']} > 8",
            "plan_stats": plan_stats,
            "commit": C.git_commit(),
        }
        _write_failure(out_dir, payload)
        log("BLOCKED: induced width > 8")
        return 2

    model, model_info = M.build_model(bundle.atom_dims, bundle.bond_dims,
                                      int(args.seed), config=cfg)
    model = model.to(device)
    log(f"model parameters={C.dumps_json(model_info['parameters'])} "
        f"init_hash={model_info['init_hash']}")
    log(f"model readout_in_dim={model_info['readout_in_dim']} "
        f"sentence_in_dim={model_info['sentence_in_dim']}")

    recipe = dict(cfg["recipe"])
    epochs = int(recipe["max_epochs"])
    if args.preflight:
        epochs = int(min(epochs, int(cfg["preflight"]["max_epochs"])))
        if args.max_epochs is not None:
            epochs = int(min(epochs, int(args.max_epochs)))
    batch_mol = int(recipe["batch_molecules"])
    eval_batch = int(recipe["eval_batch_molecules"])
    steps_per_epoch = int(math.ceil(bundle.train_idx.size / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["peak_lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    shuffle_rng = np.random.default_rng(int(args.seed) + 101)

    curves: list[dict[str, Any]] = []
    best = {"valid_rmse": float("inf"), "epoch": -1, "metrics": None,
            "predictions": None}
    step = 0
    stopped_early = False
    t_train_eval = 0.0
    first_batch: dict[str, Any] = {}
    metrics_path = out_dir / "metrics.csv"

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        _sync(device)
        model.train()
        order = shuffle_rng.permutation(int(bundle.train_idx.size))
        train_rows = bundle.train_idx[order]
        losses: list[float] = []
        grad_norms: list[float] = []
        epoch_first_grad = None
        for part in _batches(train_rows, batch_mol):
            lr = lr_at(step, warmup_steps, total_steps,
                       float(recipe["peak_lr"]), float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            batch = D.collate_rows(bundle, part.tolist(), plan, device)
            out = model(batch)
            loss = F.mse_loss(out["yhat_std"], batch["y_std"])
            if not torch.isfinite(loss):
                payload = {
                    "status": "BLOCKED",
                    "reason": f"non-finite training loss at epoch {epoch} step {step}",
                    "epoch": epoch, "step": step,
                    "plan_stats": plan_stats, "commit": C.git_commit(),
                }
                _write_failure(out_dir, payload)
                log("BLOCKED: non-finite training loss")
                return 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(),
                                        float(recipe["gradient_clip"]))
            grad_norm_value = float(grad_norm.detach().cpu().item())
            if not math.isfinite(grad_norm_value):
                payload = {
                    "status": "BLOCKED",
                    "reason": f"non-finite gradient norm at epoch {epoch} step {step}",
                    "epoch": epoch, "step": step,
                    "plan_stats": plan_stats, "commit": C.git_commit(),
                }
                _write_failure(out_dir, payload)
                log("BLOCKED: non-finite gradient norm")
                return 2
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            grad_norms.append(grad_norm_value)
            if epoch == 1 and step == 0:
                epoch_first_grad = grad_norm_value
                first_batch = {
                    "loss": losses[-1],
                    "grad_norm": grad_norm_value,
                    "lr": lr,
                    "n_molecules": int(part.size),
                }
            step += 1
        _sync(device)
        train_seconds = time.perf_counter() - t_epoch

        eval_results: dict[str, dict[str, Any]] = {}
        for split in ("train", "valid", "test"):
            rows = bundle.split_rows(split)
            result = evaluate(model, bundle, plan, rows,
                              batch_size=eval_batch, device=device)
            eval_results[split] = result

        _sync(device)
        epoch_seconds = time.perf_counter() - t_epoch
        t_train_eval += epoch_seconds
        train_loss = float(np.mean(losses))
        grad_norm_mean = float(np.mean(grad_norms))
        metrics = {
            "epoch": epoch,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "grad_norm": grad_norm_mean,
            "train_rmse": eval_results["train"]["rmse"],
            "train_mae": eval_results["train"]["mae"],
            "valid_rmse": eval_results["valid"]["rmse"],
            "valid_mae": eval_results["valid"]["mae"],
            "test_rmse": eval_results["test"]["rmse"],
            "test_mae": eval_results["test"]["mae"],
            "train_split_eval_seconds": eval_results["train"]["seconds"],
            "valid_split_eval_seconds": eval_results["valid"]["seconds"],
            "test_split_eval_seconds": eval_results["test"]["seconds"],
            "train_step_seconds": train_seconds,
            "epoch_seconds": epoch_seconds,
            "peak_vram_gb": _peak_vram_gb(),
        }
        curves.append(metrics)
        C.append_csv(metrics_path, metrics, METRIC_COLUMNS)
        log(f"epoch {epoch:3d} lr={metrics['lr']:.3e} loss={train_loss:.6f} "
            f"grad={grad_norm_mean:.4f} train={metrics['train_rmse']:.6f}/"
            f"{metrics['train_mae']:.6f} valid={metrics['valid_rmse']:.6f}/"
            f"{metrics['valid_mae']:.6f} test={metrics['test_rmse']:.6f}/"
            f"{metrics['test_mae']:.6f} sec={epoch_seconds:.2f} "
            f"vram={metrics['peak_vram_gb']:.3f}GB")

        if not args.preflight and metrics["valid_rmse"] < best["valid_rmse"]:
            best = {
                "valid_rmse": metrics["valid_rmse"],
                "epoch": epoch,
                "metrics": dict(metrics),
                "predictions": {
                    split: {
                        "row_ids": eval_results[split]["row_ids"],
                        "target_raw": eval_results[split]["target_raw"],
                        "pred_raw": eval_results[split]["pred_raw"],
                        "pred_std": eval_results[split]["pred_std"],
                        "rmse": eval_results[split]["rmse"],
                        "mae": eval_results[split]["mae"],
                    }
                    for split in ("train", "valid", "test")
                },
            }
            _save_best(out_dir / "best.pt", model, cfg, bundle, plan_stats,
                       epoch, metrics, best["predictions"])
            _save_predictions(out_dir / "predictions_best.npz", best["predictions"])
            log(f"      best checkpoint updated: epoch={epoch} "
                f"valid_rmse={best['valid_rmse']:.6f}")

        if not args.preflight:
            patience = int(recipe["patience"])
            min_epochs = int(recipe["min_epochs"])
            if epoch >= min_epochs and (epoch - int(best["epoch"])) >= patience:
                stopped_early = True
                log(f"early stop at epoch {epoch}: best epoch {best['epoch']}, "
                    f"patience {patience}")
                break

    process_seconds = time.perf_counter() - t_process
    peak_vram = _peak_vram_gb()

    if args.preflight:
        per_epoch = [float(row["epoch_seconds"]) for row in curves]
        mean_epoch = float(np.mean(per_epoch)) if per_epoch else float("nan")
        projected_seconds = mean_epoch * int(cfg["preflight"]["projection_epochs"])
        budget_seconds = float(cfg["preflight"]["budget_gpu_hours"]) * 3600.0
        budget_ok = bool(projected_seconds <= budget_seconds)
        payload = {
            "status": "PASS" if budget_ok else "BLOCKED_BUDGET",
            "preflight": True,
            "run_id": run_id,
            "n_epochs_run": len(curves),
            "per_epoch_seconds": per_epoch,
            "mean_epoch_seconds": mean_epoch,
            "train_eval_seconds_total": t_train_eval,
            "plan_build_seconds": plan_seconds,
            "process_seconds": process_seconds,
            "projected_epochs": int(cfg["preflight"]["projection_epochs"]),
            "projected_seconds": projected_seconds,
            "projected_gpu_hours": projected_seconds / 3600.0,
            "budget_gpu_hours": float(cfg["preflight"]["budget_gpu_hours"]),
            "budget_ok": budget_ok,
            "peak_vram_gb": peak_vram,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "parameters": model_info["parameters"],
            "init_hash": model_info["init_hash"],
            "first_batch": first_batch,
            "metrics_last_epoch": curves[-1] if curves else None,
            "plan_stats": plan_stats,
            "data_audit": bundle.audit,
            "commit": C.git_commit(),
            "dirty": C.git_dirty(),
            "config": cfg,
            "config_sha256": C.sha256_file(Path(args.config)),
            "cache_sha256": bundle.cache_sha256,
            "environment": C.environment_snapshot(),
            "finished_at": C.utc_stamp(),
        }
        C.write_json(out_dir / "preflight.json", payload)
        C.write_json(C.ANALYSIS_DIR / "preflight_gpu.json", payload)
        log(f"PREFLIGHT status={payload['status']} "
            f"projected_gpu_hours={payload['projected_gpu_hours']:.4f} "
            f"budget={payload['budget_gpu_hours']} peak_vram={peak_vram:.3f}GB")
        log_handle.close()
        return 0

    # Formal run: the best checkpoint must exist.
    if best["metrics"] is None:
        payload = {
            "status": "BLOCKED",
            "reason": "no epoch produced a valid checkpoint",
            "commit": C.git_commit(),
            "plan_stats": plan_stats,
        }
        _write_failure(out_dir, payload)
        log("BLOCKED: no valid checkpoint")
        return 2

    best_epoch = int(best["epoch"])
    best_metrics = best["metrics"]
    verdict, decision = _verdict(best_metrics["valid_rmse"], cfg)
    log(f"best epoch={best_epoch} valid_rmse={best_metrics['valid_rmse']:.6f} "
        f"verdict={verdict}")

    # Reload the best checkpoint and reproduce the prediction metrics through
    # the same evaluation operator.  This is an engineering integrity check.
    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu",
                            weights_only=False)
    reloaded = M.load_model_from_state(checkpoint["state_dict"],
                                       bundle.atom_dims, bundle.bond_dims,
                                       config=cfg, device=device)
    reloaded.eval()
    reload_results: dict[str, Any] = {}
    max_pred_diff = 0.0
    for split in ("train", "valid", "test"):
        rows = bundle.split_rows(split)
        result = evaluate(reloaded, bundle, plan, rows,
                          batch_size=eval_batch, device=device)
        saved = best["predictions"][split]
        diff = float(np.max(np.abs(result["pred_std"] - saved["pred_std"]))) \
            if result["pred_std"].size else 0.0
        max_pred_diff = max(max_pred_diff, diff)
        reload_results[split] = {
            "rmse": result["rmse"], "mae": result["mae"],
            "max_abs_pred_diff": diff,
        }
    reload_ok = bool(max_pred_diff <= 1e-5)
    log(f"checkpoint reload max_pred_diff={max_pred_diff:.3e} ok={reload_ok}")

    payload = {
        "status": "PASS",
        "preflight": False,
        "run_id": run_id,
        "seed": int(args.seed),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "commit": C.git_commit(),
        "dirty": C.git_dirty(),
        "config": cfg,
        "config_sha256": C.sha256_file(Path(args.config)),
        "cache_sha256": bundle.cache_sha256,
        "data_audit": bundle.audit,
        "parameters": model_info["parameters"],
        "init_hash": model_info["init_hash"],
        "plan_stats": plan_stats,
        "plan_build_seconds": plan_seconds,
        "epochs_run": len(curves),
        "epochs_configured": epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "verdict": verdict,
        "decision": decision,
        "train_eval_seconds_total": t_train_eval,
        "process_seconds": process_seconds,
        "peak_vram_gb": peak_vram,
        "first_batch": first_batch,
        "checkpoint_reload": {"ok": reload_ok,
                              "max_abs_pred_diff": max_pred_diff,
                              "metrics": reload_results},
        "environment": C.environment_snapshot(),
        "finished_at": C.utc_stamp(),
    }
    C.write_json(out_dir / "run.json", payload)
    C.write_json(C.ANALYSIS_DIR / "run_formal.json", payload)
    C.write_json(C.ANALYSIS_DIR / "curves.json", {"curves": curves})
    log(f"FORMAL status={payload['status']} best_epoch={best_epoch} "
        f"valid_rmse={best_metrics['valid_rmse']:.6f} verdict={verdict} "
        f"decision={decision} train_eval_seconds_total={t_train_eval:.2f} "
        f"peak_vram={peak_vram:.3f}GB")
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())