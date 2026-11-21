"""Seed-0 training for the two direct-local-object arms.

Two launches per arm, no restart:

* ``stage-a`` trains epochs 1..5 and saves model+optimizer+RNG so the shared
  horizon can be decided from the observed epoch-3..5 timing;
* ``continue`` resumes at epoch 6 with the common horizon (240 or 160) and the
  cosine schedule computed for that horizon.

Every epoch evaluates Train, Valid and Test in eval mode and reports frozen-
scaler, original-label-unit RMSE/MAE.  Model selection is Valid only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import dlo_common as C
import dlo_model as M
from dlo_cache import ArmCache, load_payload


def epoch_orders(n: int, epochs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return [rng.permutation(n) for _ in range(int(epochs))]


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, start: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return start + (peak - start) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def parameter_hash(model) -> str:
    return state_hash({name: p for name, p in model.named_parameters()})


def to_tensors(pack: Mapping[str, Any], device: str):
    out: dict[str, Any] = {}
    for key, value in pack.items():
        if key in {"n_molecules", "n_objects", "n_frames"}:
            out[key] = int(value)
        elif key in {"frame_weight", "frame_node_mask", "frame_edge_mask", "frame_adj", "obj_bucket"}:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)
        else:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.long, device=device)
    return out


def raw_metrics(pred_std: np.ndarray, labels: np.ndarray, scaler: Mapping[str, float]) -> dict[str, float]:
    pred = pred_std * float(scaler["std"]) + float(scaler["mean"])
    return {"rmse": C.rmse(pred, labels), "mae": C.mae(pred, labels)}


def predict(cache: ArmCache, model, rows: np.ndarray, device: str, batch: int,
            zero_objects: bool = False) -> np.ndarray:
    model.eval()
    out = np.zeros(len(rows), np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            b = to_tensors(cache.pack(chunk), device)
            y = model.forward_packed(b, zero_objects=zero_objects)
            out[start:start + len(chunk)] = y.detach().double().cpu().numpy()
    return out


def evaluate_all(cache: ArmCache, model, device: str, batch: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for split in C.SPLITS:
        rows = cache.split_rows[split]
        p = predict(cache, model, rows, device, batch)
        preds[split] = p
        metrics[split] = raw_metrics(p, cache.labels[rows], cache.scaler)
    return preds, metrics


def save_predictions(path: Path, cache: ArmCache, preds: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        rows = cache.split_rows[split]
        p = np.asarray(preds[split], np.float64)
        payload[f"{split}_row_id"] = np.asarray(cache.row_ids[rows], np.int64)
        payload[f"{split}_label"] = np.asarray(cache.labels[rows], np.float64)
        payload[f"{split}_pred_std"] = p
        payload[f"{split}_pred"] = p * float(cache.scaler["std"]) + float(cache.scaler["mean"])
    np.savez(path, **payload)
    return C.sha256_file(path)


def build_fresh(cfg: Mapping[str, Any], payload: Mapping[str, Any], stats: Mapping[str, Any], arm: str, device: str):
    C.set_deterministic(int(cfg["recipe"]["seed"]))
    model = M.build_model(cfg, payload, seed=int(cfg["recipe"]["seed"])).to(device)
    support = M.apply_train_support(model, stats)
    M.bind_constants(model, stats, arm)
    return model, support


def _curve_row(run_id: str, epoch: int, lr: float, metrics: Mapping[str, Mapping[str, float]],
               online_loss: float, epoch_seconds: float, cumulative_seconds: float) -> dict[str, Any]:
    row: dict[str, Any] = {"run_id": run_id, "epoch": int(epoch), "lr": float(lr)}
    for split in C.SPLITS:
        row[f"{split}_rmse"] = float(metrics[split]["rmse"])
        row[f"{split}_mae"] = float(metrics[split]["mae"])
    row["online_loss"] = float(online_loss) if online_loss is not None else ""
    row["epoch_seconds"] = float(epoch_seconds)
    row["cumulative_seconds"] = float(cumulative_seconds)
    return row


def _deadline_reached(stop_unix: float | None) -> bool:
    return stop_unix is not None and time.time() >= float(stop_unix)


def _train_epochs(*, cfg, cache: ArmCache, model, optimizer, orders, device: str,
                  first_epoch: int, horizon: int, train_rows: np.ndarray,
                  curves: list[dict[str, Any]], run_id: str, global_step: int,
                  best: dict[str, Any], epoch_times: list[float],
                  t_start: float, stop_unix: float | None) -> dict[str, Any]:
    rec = cfg["recipe"]
    batch = int(rec["batch"])
    eval_batch = int(rec["eval_batch"])
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch))
    warmup_steps = int(rec["warmup_epochs"]) * steps_per_epoch
    total_steps = int(horizon) * steps_per_epoch
    stopped_at = first_epoch - 1
    status = "running"
    for epoch in range(first_epoch, int(horizon) + 1):
        if _deadline_reached(stop_unix):
            status = "partial_deadline"
            print(f"[{run_id}] wall-clock deadline reached before epoch {epoch}", flush=True)
            break
        model.train()
        loss_sum = 0.0
        n_seen = 0
        order = orders[epoch - 1]
        t_epoch = time.perf_counter()
        for start in range(0, len(train_rows), batch):
            idx = order[start:start + batch]
            chunk = train_rows[idx]
            b = to_tensors(cache.pack(chunk), device)
            optimizer.param_groups[0]["lr"] = lr_at(
                global_step, warmup_steps, total_steps,
                float(rec["peak_lr"]), float(rec["warmup_start_lr"]), float(rec["min_lr"]),
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            y = model.forward_packed(b)
            target = torch.as_tensor(cache.target_std(chunk), dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(rec["clip"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(chunk)
            n_seen += len(chunk)
        preds, metrics = evaluate_all(cache, model, device, eval_batch)
        epoch_seconds = time.perf_counter() - t_epoch
        epoch_times.append(epoch_seconds)
        cumulative = time.perf_counter() - t_start
        lr_used = float(optimizer.param_groups[0]["lr"])
        curves.append(_curve_row(run_id, epoch, lr_used, metrics, loss_sum / max(1, n_seen),
                                 epoch_seconds, cumulative))
        print(
            f"[{run_id}] epoch {epoch}/{horizon} "
            f"train={metrics['train']['rmse']:.4f} valid={metrics['valid']['rmse']:.4f} "
            f"test={metrics['test']['rmse']:.4f} lr={lr_used:.2e} {epoch_seconds:.2f}s",
            flush=True,
        )
        if metrics["valid"]["rmse"] < float(best["valid_rmse"]):
            best = {
                "valid_rmse": float(metrics["valid"]["rmse"]),
                "epoch": int(epoch),
                "state": copy.deepcopy(model.state_dict()),
                "metrics": {k: dict(v) for k, v in metrics.items()},
            }
        stopped_at = epoch
        status = "completed" if epoch == int(horizon) else "running"
    return {
        "model": model, "optimizer": optimizer, "curves": curves,
        "global_step": global_step, "stopped_at": stopped_at, "status": status,
        "best": best, "epoch_times": epoch_times,
    }


def _save_run(out_dir: Path, cfg: Mapping[str, Any], arm: str, cache: ArmCache,
              model, result: Mapping[str, Any], init_hash: str, support: Mapping[str, Any],
              stage_a: Mapping[str, Any] | None, device: str, tf32: Mapping[str, Any],
              command: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_id = out_dir.name
    curves = result["curves"]
    eval_batch = int(cfg["recipe"]["eval_batch"])
    final_preds = {
        split: predict(cache, model, cache.split_rows[split], device, eval_batch)
        for split in C.SPLITS
    }
    final_metrics = {
        split: raw_metrics(final_preds[split], cache.labels[cache.split_rows[split]], cache.scaler)
        for split in C.SPLITS
    }
    torch.save(model.state_dict(), out_dir / "state_final.pt")
    model.load_state_dict(result["best"]["state"])
    best_preds = {
        split: predict(cache, model, cache.split_rows[split], device, eval_batch)
        for split in C.SPLITS
    }
    torch.save(result["best"]["state"], out_dir / "state_best.pt")
    best_pred_sha = save_predictions(out_dir / "predictions_best.npz", cache, best_preds)
    final_pred_sha = save_predictions(out_dir / "predictions_final.npz", cache, final_preds)
    best_metrics = {
        split: raw_metrics(best_preds[split], cache.labels[cache.split_rows[split]], cache.scaler)
        for split in C.SPLITS
    }
    C.write_csv(out_dir / "curves.csv", curves)
    best_epoch = int(result["best"]["epoch"])
    best_row = curves[best_epoch - 1]
    final_row = curves[-1]
    epoch_times = list(result["epoch_times"])
    steady = epoch_times[2:5] if len(epoch_times) >= 5 else epoch_times[-3:]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "protocol_id": cfg["protocol_id"],
        "horizon": int(result["horizon"]),
        "planned_horizon": int(cfg["recipe"]["default_horizon"]),
        "stopped_at": int(result["stopped_at"]),
        "status": str(result["status"]),
        "completed": bool(result["stopped_at"] == int(result["horizon"])),
        "best_epoch": best_epoch,
        "best_valid_rmse": float(result["best"]["valid_rmse"]),
        "best": {
            "epoch": best_epoch,
            "train_rmse": float(best_row["train_rmse"]),
            "valid_rmse": float(best_row["valid_rmse"]),
            "test_rmse": float(best_row["test_rmse"]),
            "train_mae": float(best_row["train_mae"]),
            "valid_mae": float(best_row["valid_mae"]),
            "test_mae": float(best_row["test_mae"]),
            "recomputed": best_metrics,
        },
        "final": {
            "epoch": int(final_row["epoch"]),
            "train_rmse": float(final_row["train_rmse"]),
            "valid_rmse": float(final_row["valid_rmse"]),
            "test_rmse": float(final_row["test_rmse"]),
            "train_mae": float(final_row["train_mae"]),
            "valid_mae": float(final_row["valid_mae"]),
            "test_mae": float(final_row["test_mae"]),
            "recomputed": final_metrics,
        },
        "epoch_seconds": [float(x) for x in epoch_times],
        "steady_epoch_seconds_median": float(np.median(steady)) if steady else None,
        "wall_seconds": float(result["wall_seconds"]),
        "wall_seconds_total": float(result["wall_seconds_total"]),
        "gpu_hours": float(result["wall_seconds_total"] / 3600.0),
        "steps_per_epoch": int(result["steps_per_epoch"]),
        "global_step": int(result["global_step"]),
        "parameters": model.parameter_ledger(),
        "init_state_hash": init_hash,
        "train_support": support,
        "tf32": tf32,
        "cache_sha256": getattr(cache, "cache_sha256", None),
        "predictions_best_sha256": best_pred_sha,
        "predictions_final_sha256": final_pred_sha,
        "state_hashes": {
            "state_best": C.sha256_file(out_dir / "state_best.pt"),
            "state_final": C.sha256_file(out_dir / "state_final.pt"),
        },
        "max_vram_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
        "regime": C.environment(),
        "run_tag": os.environ.get("DLO_RUN_TAG"),
        "command": command,
        "stage_a": dict(stage_a) if stage_a is not None else None,
    }
    if extra:
        payload.update(dict(extra))
    C.write_json(out_dir / "run.json", payload)
    return payload


def _intervention(out_dir: Path, cache: ArmCache, cfg: Mapping[str, Any],
                  model, device: str) -> dict[str, Any]:
    best_state = torch.load(out_dir / "state_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state)
    rows = cache.split_rows["valid"]
    normal = predict(cache, model, rows, device, int(cfg["recipe"]["eval_batch"]))
    zero = predict(cache, model, rows, device, int(cfg["recipe"]["eval_batch"]), zero_objects=True)
    labels = cache.labels[rows]
    scaler = cache.scaler

    def raw(p):
        return p * scaler["std"] + scaler["mean"]

    payload = {
        "valid_rmse_normal": C.rmse(raw(normal), labels),
        "valid_rmse_zero_objects": C.rmse(raw(zero), labels),
        "valid_rmse_delta": C.rmse(raw(zero), labels) - C.rmse(raw(normal), labels),
        "pred_rms_normal": float(np.sqrt(np.mean(raw(normal) ** 2))),
        "pred_rms_zero_objects": float(np.sqrt(np.mean(raw(zero) ** 2))),
        "pred_rms_ratio_zero_to_normal": float(
            np.sqrt(np.mean(raw(zero) ** 2)) / max(1e-12, np.sqrt(np.mean(raw(normal) ** 2)))
        ),
        "pred_change_rms": float(np.sqrt(np.mean((raw(zero) - raw(normal)) ** 2))),
        "note": "h_o set to zero, four structural counts kept; fixed best-Valid weights, no retraining",
    }
    C.write_json(out_dir / "intervention.json", payload)
    return payload


def stage_a(args) -> None:
    cfg = C.load_config()
    stats = C.read_json(args.stats)
    payload = load_payload(args.cache)
    cache = ArmCache(payload, stats, args.arm)
    out_dir = C.run_dir(args.out)
    run_id = out_dir.name
    device = args.device
    C.set_deterministic(int(cfg["recipe"]["seed"]))
    tf32 = C.enable_tf32(bool(cfg["recipe"]["tf32"]))
    model, support = build_fresh(cfg, payload, stats, args.arm, device)
    init_param_hash = parameter_hash(model)
    init_hash = state_hash(model.state_dict())
    train_rows = cache.split_rows["train"]
    rec = cfg["recipe"]
    steps_per_epoch = max(1, math.ceil(len(train_rows) / int(rec["batch"])))
    orders = epoch_orders(len(train_rows), int(rec["default_horizon"]), int(rec["order_rng_seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rec["peak_lr"]),
                                  weight_decay=float(rec["weight_decay"]))
    t_start = time.perf_counter()
    result = _train_epochs(
        cfg=cfg, cache=cache, model=model, optimizer=optimizer, orders=orders, device=device,
        first_epoch=1, horizon=int(args.epochs), train_rows=train_rows, curves=[],
        run_id=run_id, global_step=0, best={"valid_rmse": float("inf"), "epoch": None},
        epoch_times=[], t_start=t_start, stop_unix=args.stop_unix,
    )
    result["horizon"] = int(args.epochs)
    result["wall_seconds"] = time.perf_counter() - t_start
    result["wall_seconds_total"] = result["wall_seconds"]
    result["steps_per_epoch"] = steps_per_epoch
    epoch_times = result["epoch_times"]
    steady = epoch_times[2:5] if len(epoch_times) >= 5 else epoch_times[-3:]
    median = float(np.median(steady)) if steady else float("nan")
    cost = {
        "median_epoch_seconds_epochs_3_to_5": median,
        "n_epochs_observed": len(epoch_times),
        "estimate_240_minutes": median * 240 * float(rec["cost_safety"]) / 60.0,
        "estimate_160_minutes": median * 160 * float(rec["cost_safety"]) / 60.0,
        "threshold_minutes": float(rec["cost_threshold_minutes"]),
        "epoch_seconds": [float(x) for x in epoch_times],
    }
    C.write_json(out_dir / "stage_a_cost.json", cost)
    ckpt = {
        "arm": args.arm,
        "epoch": int(result["stopped_at"]),
        "global_step": int(result["global_step"]),
        "model": result["model"].state_dict(),
        "optimizer": result["optimizer"].state_dict(),
        "curves": result["curves"],
        "epoch_times": result["epoch_times"],
        "wall_seconds": result["wall_seconds"],
        "init_state_hash": init_hash,
        "init_parameter_hash": init_param_hash,
        "support": support,
        "best": {k: v for k, v in result["best"].items() if k != "state"},
        "best_state": result["best"]["state"],
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, out_dir / "state_epoch5.pt")
    C.write_csv(out_dir / "curves_stage_a.csv", result["curves"])
    _save_run(out_dir, cfg, args.arm, cache, result["model"], result, init_hash, support,
              cost, device, tf32, " ".join(os.sys.argv),
              extra={"stage_a_cost": cost, "init_parameter_hash": init_param_hash})
    print(f"[{args.arm}] stage A done: {result['stopped_at']} epochs, median {median:.2f}s, "
          f"est240 {cost['estimate_240_minutes']:.1f} min, est160 {cost['estimate_160_minutes']:.1f} min", flush=True)


def continue_run(args) -> None:
    cfg = C.load_config()
    stats = C.read_json(args.stats)
    payload = load_payload(args.cache)
    cache = ArmCache(payload, stats, args.arm)
    out_dir = C.run_dir(args.out)
    ckpt = torch.load(out_dir / "state_epoch5.pt", map_location="cpu", weights_only=False)
    if int(ckpt["epoch"]) != int(cfg["recipe"]["probe_epochs"]):
        raise ValueError("resume checkpoint is not the epoch-5 probe")
    stage_a_cost = C.read_json(out_dir / "stage_a_cost.json")
    device = args.device
    model, support = build_fresh(cfg, payload, stats, args.arm, device)
    init_param_hash = parameter_hash(model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["peak_lr"]),
                                  weight_decay=float(cfg["recipe"]["weight_decay"]))
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tf32 = C.enable_tf32(bool(cfg["recipe"]["tf32"]))
    train_rows = cache.split_rows["train"]
    rec = cfg["recipe"]
    steps_per_epoch = max(1, math.ceil(len(train_rows) / int(rec["batch"])))
    orders = epoch_orders(len(train_rows), int(args.horizon), int(rec["order_rng_seed"]))
    curves = list(ckpt["curves"])
    t_start = time.perf_counter()
    best = {
        "valid_rmse": float(ckpt["best"]["valid_rmse"]),
        "epoch": int(ckpt["best"]["epoch"]),
        "state": ckpt["best_state"],
        "metrics": {},
    }
    result = _train_epochs(
        cfg=cfg, cache=cache, model=model, optimizer=optimizer, orders=orders, device=device,
        first_epoch=int(ckpt["epoch"]) + 1, horizon=int(args.horizon), train_rows=train_rows,
        curves=curves, run_id=out_dir.name, global_step=int(ckpt["global_step"]),
        best=best, epoch_times=list(ckpt["epoch_times"]),
        t_start=t_start, stop_unix=args.stop_unix,
    )
    result["horizon"] = int(args.horizon)
    result["wall_seconds"] = time.perf_counter() - t_start
    result["wall_seconds_total"] = result["wall_seconds"] + float(ckpt["wall_seconds"])
    result["steps_per_epoch"] = steps_per_epoch
    ground = _save_run(out_dir, cfg, args.arm, cache, model, result, str(ckpt["init_state_hash"]),
                       support, stage_a_cost, device, tf32, " ".join(os.sys.argv),
                       extra={"stage_a_cost": stage_a_cost, "init_parameter_hash": init_param_hash,
                              "stage_a_init_parameter_hash": ckpt.get("init_parameter_hash")})
    _intervention(out_dir, cache, cfg, model, device)
    print(f"[{args.arm}] done at epoch {result['stopped_at']}/{args.horizon} "
          f"best={ground['best_valid_rmse']:.6f}@{ground['best_epoch']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage-a", "continue"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True, choices=C.ARMS)
        p.add_argument("--cache", default=str(C.data_dir() / "dlo_objects_s0.npz"))
        p.add_argument("--stats", default=str(C.data_dir() / "dlo_cache_stats.json"))
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--stop-unix", type=float, default=None)
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--horizon", type=int, default=240)
    args = ap.parse_args()
    if args.cmd == "stage-a":
        stage_a(args)
    else:
        continue_run(args)


if __name__ == "__main__":
    main()
