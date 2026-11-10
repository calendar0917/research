"""Formal from-scratch training for the PCQM4Mv2 transfer pilot.

Two runs (300K, 1M), one seed, 40 epochs each, one formal run per row of the
pre-registered table.  Each epoch: train on every batch of the run's fixed
subset, then evaluate

1. the **official full Validation split** with the official
   ``PCQM4Mv2Evaluator`` (MAE in eV) and an independent NumPy
   ``mean(|pred - true|)`` cross-check on the same predictions,
2. the fixed 20,000-row Train monitor subset (same rows in both runs).

Checkpoint selection is by Validation MAE only (ties -> earlier epoch).  The
hidden Test splits are never indexed anywhere in this module.

The training step is exposed as :func:`train_step` so the smoke test and the
cost probe measure the *real* step (same collate / stager / model / optimizer /
clip / loss code path), never a re-implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402
import pq_model as M  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_l1_z",
    "train_l1_z_molecule_weighted",
    "valid_mae_eV",
    "valid_mae_eV_numpy",
    "valid_mae_crosscheck_delta_eV",
    "valid_pred_mean_eV",
    "valid_pred_std_eV",
    "monitor_mae_eV",
    "monitor_n",
    "epoch_train_seconds",
    "epoch_valid_seconds",
    "epoch_monitor_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "peak_rss_mb",
    "grad_norm_max",
    "grad_norm_mean",
    "gradient_clip_events",
    "best_epoch_so_far",
    "best_valid_mae_eV_so_far",
    "is_best",
    "checkpoint_seconds",
    "rng_cpu_sha256",
    "rng_cuda_sha256",
]


# ---------------------------------------------------------------------------
# the single training step (shared with the smoke test / cost probe)
# ---------------------------------------------------------------------------


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y_z: torch.Tensor,
    *,
    global_step: int,
    steps_epoch: int,
    steps_total: int,
    clip: float,
) -> dict[str, Any]:
    """One optimizer step of the fixed recipe (L1 on z)."""
    lr = M.set_lrs(optimizer, step=int(global_step), steps_epoch=int(steps_epoch), steps_total=int(steps_total))
    pred = M.predict_z(model, batch)
    n = int(y_z.shape[0])
    if tuple(pred.shape) != (n,):
        raise RuntimeError(f"model output {tuple(pred.shape)} != {(n,)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError(f"non-finite prediction at step {global_step}")
    loss = torch.mean(torch.abs(pred - y_z))
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {global_step}: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if not math.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": gnorm, "lr": lr, "clipped": bool(gnorm > clip), "n": n}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict_rows(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    device: torch.device,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardised predictions for ``rows`` (fixed ascending order)."""
    rows = np.asarray([int(r) for r in rows], dtype=np.int64)
    out = np.empty(rows.size, dtype=np.float32)
    t0 = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for i in range(0, rows.size, int(batch_size)):
            chunk = rows[i : i + int(batch_size)]
            batch = D.collate(cache, chunk.tolist())
            batch = {k: (v.to(device, non_blocking=False) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = M.predict_z(model, batch)
            out[i : i + chunk.size] = pred.detach().float().cpu().numpy()
    payload = {
        "seconds": float(time.perf_counter() - t0),
        "n": int(rows.size),
        "batch_size": int(batch_size),
        "pred_z_sha256": C.sha256_bytes(out.tobytes()),
        "pred_z_mean": float(out.mean()),
        "pred_z_std": float(out.std()),
        "pred_z_min": float(out.min()),
        "pred_z_max": float(out.max()),
    }
    return out, payload


def mae_report(
    evaluator: Any,
    y_true_eV: np.ndarray,
    y_pred_eV: np.ndarray,
) -> dict[str, Any]:
    """Official evaluator MAE plus the independent NumPy cross-check."""
    yt = np.asarray(y_true_eV, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_eV, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"truth {yt.shape} != prediction {yp.shape}")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("non-finite value in MAE computation")
    official = float(evaluator.eval({"y_pred": yp, "y_true": yt})["mae"])
    independent = float(np.mean(np.abs(yp - yt)))
    return {
        "mae_eV": official,
        "mae_eV_numpy": independent,
        "crosscheck_delta_eV": abs(official - independent),
        "n": int(yt.size),
        "pred_mean_eV": float(yp.mean()),
        "pred_std_eV": float(yp.std()),
        "pred_min_eV": float(yp.min()),
        "pred_max_eV": float(yp.max()),
        "target_mean_eV": float(yt.mean()),
        "target_std_eV": float(yt.std()),
        "pred_sha256": C.sha256_bytes(np.asarray(yp, dtype=np.float64).tobytes()),
        "abs_error_sum_eV": float(np.abs(yp - yt).sum()),
        "evaluator": "ogb.lsc.PCQM4Mv2Evaluator.eval({'y_pred','y_true'}) -> {'mae'}",
        "units": "eV",
    }


def evaluate_split(
    model: torch.nn.Module,
    task: Mapping[str, Any],
    device: torch.device,
    *,
    which: str,
    batch_size: int = 512,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one of ``valid`` / ``monitor`` and return (report, preds_z)."""
    if which == "valid":
        cache, rows, y_eV = task["valid_cache"], task["valid_rows"], task["valid_y_eV"]
    elif which == "monitor":
        cache, rows, y_eV = task["train_cache"], task["monitor_rows"], task["monitor_y_eV"]
    else:  # pragma: no cover - defensive
        raise KeyError(which)
    pred_z, info = predict_rows(model, cache, rows, device, batch_size=batch_size)
    pred_eV = D.from_z(pred_z, task["label_norm"])
    report = mae_report(task["evaluator"], y_eV, pred_eV)
    report.update(
        {
            "split": which,
            "rows": int(len(rows)),
            "inference_seconds": info["seconds"],
            "pred_z_sha256": info["pred_z_sha256"],
            "pred_z_mean": info["pred_z_mean"],
            "pred_z_std": info["pred_z_std"],
        }
    )
    return report, pred_z


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class Watchdog:
    def __init__(self, hard_seconds: float):
        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self.fired_at: float | None = None
        self._t0 = time.perf_counter()
        self._timer = threading.Timer(self.hard_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.fired_at = time.perf_counter() - self._t0
        self.stop.set()

    def cancel(self) -> None:
        self._timer.cancel()


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})
    return C.sha256_file(path)


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def plan_meta(task: Mapping[str, Any], consumed: int) -> dict[str, Any]:
    orders = task["orders"]
    prefix = orders["epoch_permutation_sha256"][: int(consumed)]
    return {
        "order_seed": int(orders["order_seed"]),
        "n_rows": int(orders["n_rows"]),
        "batch_size": int(C.BATCH_SIZE),
        "epochs": int(orders["epochs"]),
        "consumed_epochs": int(consumed),
        "combined_sha256": orders["combined_sha256"],
        "prefix_sha256": C.sha256_bytes("|".join(prefix).encode("utf-8")),
    }


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer, task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "scratch_warmup_cosine_40epoch_pcqm",
        "global_step": int(global_step),
        "total_steps_40_epochs": int(task["steps_total"]),
        "steps_per_epoch": int(task["steps_epoch"]),
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(int(task["n_rows"])),
    }


def checkpoint_payload(
    *,
    task: Mapping[str, Any],
    run: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng_state: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    valid_report: Mapping[str, Any] | None,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": C.MODEL_KEY,
        "model_name": C.MODEL_NAME,
        "seed": int(task["seed"]),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_kind": kind,
        "dataset": C.DATASET,
        "n_rows": int(task["n_rows"]),
        "n_outputs": int(C.N_OUT),
        "parameters": int(model_audit["total_parameters"]),
        "state_dict": M.state_cpu(model),
        "optimizer_state_dict": M.optimizer_state_cpu(optimizer),
        "optimizer": "AdamW",
        "scheduler_state": scheduler_state(global_step, optimizer, task),
        "rng_state": rng_state,
        "rng_state_hashes": C.rng_state_hashes(rng_state),
        "batch_plan": plan_meta(task, int(epoch)),
        "model_audit": dict(model_audit),
        "feature_spec_hash": task["spec"].hash(),
        "feature_norm": {
            "fingerprint": task["feature_norm"]["norm_fingerprint"],
            "source": task["feature_norm"]["source"],
        },
        "label_norm": dict(task["label_norm"]),
        "cache": dict(task["cache_manifest"]),
        "valid_mae_eV": None if valid_report is None else float(valid_report["mae_eV"]),
        "valid_report_summary": None
        if valid_report is None
        else {
            "mae_eV": float(valid_report["mae_eV"]),
            "mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "n": int(valid_report["n"]),
            "pred_sha256": valid_report["pred_sha256"],
            "pred_z_sha256": valid_report["pred_z_sha256"],
        },
        "recipe": C.recipe_for_run(run),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "test_evaluated": False,
        "test_policy": "official test-dev / test-challenge labels are hidden and are never indexed",
        "created_utc": C.utcnow(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _rng_hash_diffs(saved: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Per-generator RNG hash comparison (CUDA lists compared element-wise)."""
    diffs = [
        key
        for key in ("python_sha256", "numpy_sha256", "torch_cpu_sha256")
        if saved.get(key) != now.get(key)
    ]
    a = list(saved.get("torch_cuda_sha256") or [])
    b = list(now.get("torch_cuda_sha256") or [])
    n = min(len(a), len(b))
    if a[:n] != b[:n]:
        diffs.append("torch_cuda_sha256")
    return diffs


def restore(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    run: str,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("run") != run:
        raise RuntimeError(f"checkpoint run {payload.get('run')} != requested {run}")
    if int(payload.get("n_rows", -1)) != int(task["n_rows"]):
        raise RuntimeError("checkpoint was trained on a different number of rows")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    steps = sorted(
        {
            float(v["step"].detach().cpu().item()) if torch.is_tensor(v["step"]) else float(v["step"])
            for v in optimizer.state_dict()["state"].values()
        }
    )
    if int(payload["global_step"]) > 0 and steps != [float(payload["global_step"])]:
        raise RuntimeError(f"restored AdamW steps {steps} != {payload['global_step']}")
    info = M.restore_rng_state(payload["rng_state"])
    now_hashes = C.rng_state_hashes(C.capture_rng_state())
    saved_hashes = payload.get("rng_state_hashes") or {}
    diffs = _rng_hash_diffs(saved_hashes, now_hashes)
    if diffs:
        raise RuntimeError(f"restored RNG state does not match the checkpoint hashes: {diffs}")
    saved = payload.get("batch_plan") or {}
    now = plan_meta(task, int(payload["epoch"]))
    if saved.get("prefix_sha256") != now["prefix_sha256"] or int(saved.get("order_seed", -1)) != int(
        now["order_seed"]
    ):
        raise RuntimeError("resumed plan does not match the checkpoint plan")
    if (payload.get("label_norm") or {}).get("mean_eV") != task["label_norm"]["mean_eV"]:
        raise RuntimeError("checkpoint label normalisation does not match the round constants")
    return {
        "resumed_from_epoch": int(payload["epoch"]),
        "resumed_global_step": int(payload["global_step"]),
        "optimizer_state_restored": True,
        "rng_state_hashes": now_hashes,
        "rng_restore": info,
        "replayed_epoch_order_hashes_verified": True,
    }


def replay_epoch_orders(task: Mapping[str, Any], upto: int) -> list[str]:
    """Replay the generator stream, verifying the precomputed per-epoch hashes."""
    rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    seen: list[str] = []
    for epoch in range(1, int(upto) + 1):
        order = rng.permutation(int(task["n_rows"])).astype(np.int64)
        digest = C.sha256_bytes(order.tobytes())
        expected = task["orders"]["epoch_permutation_sha256"][epoch - 1]
        if digest != expected:
            raise RuntimeError(f"epoch {epoch} order hash {digest} != planned {expected}")
        seen.append(digest)
    return seen


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(
    task: Mapping[str, Any],
    *,
    run: str,
    epochs: int,
    out_dir: Path,
    device: torch.device,
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
    eval_batch_size: int = 512,
    allow_partial_epochs: bool = False,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    n_rows = int(task["n_rows"])
    steps_epoch = int(task["steps_epoch"])
    steps_total = int(task["steps_total"])
    if int(epochs) != int(C.RUNS_TABLE[run]["epochs"]) and not allow_partial_epochs:
        raise RuntimeError(f"formal run must use {C.RUNS_TABLE[run]['epochs']} epochs, got {epochs}")
    gpu_start = C.gpu_occupancy()
    deterministic = C.apply_deterministic_settings()

    model, model_audit = M.build_model(task["spec"], task["norm"], seed=int(task["seed"]), device=device)
    optimizer, opt_audit = M.make_optimizer(model)

    resume_audit: Mapping[str, Any] | None = None
    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer, run=run, task=task)
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        seed_info = C.seed_everything(int(task["seed"]))
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    # replay (and verify) the epoch permutations up to the first epoch to run
    order_rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    for epoch in range(1, start_epoch):
        _ = order_rng.permutation(n_rows)
    verified_orders = replay_epoch_orders(task, start_epoch - 1)

    stager = PinnedStager(device, depth=2)
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_rows: list[dict[str, Any]] = []
    rng_epoch_hashes: list[dict[str, Any]] = []
    curve_path = out_dir / "train_curve.csv"
    last_path = out_dir / "checkpoint_last.pt"
    best_path = out_dir / "checkpoint_best_valid.pt"
    final_path = out_dir / f"checkpoint_epoch{int(epochs)}.pt"
    rng_path = out_dir / "rng_state.pt"

    gpu_peak_alloc = 0
    gpu_peak_reserved = 0
    best_epoch: int | None = None
    best_mae = math.inf
    best_checkpoint_sha = ""
    final_checkpoint_sha = ""
    last_checkpoint_sha = ""
    eval_completed: list[int] = []
    fixed_reports: dict[int, dict[str, Any]] = {}
    monitor_reports: dict[int, dict[str, Any]] = {}
    epochs_run = int(start_epoch) - 1
    stop_reason = "RUNNING"
    clip_events_total = 0

    def update_peak() -> None:
        nonlocal gpu_peak_alloc, gpu_peak_reserved
        if device.type == "cuda":
            gpu_peak_alloc = max(gpu_peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            gpu_peak_reserved = max(gpu_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

    for epoch in range(int(start_epoch), int(epochs) + 1):
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        if time.perf_counter() - t_start > float(soft_seconds):
            stop_reason = "SOFT_WALL_LIMIT"
            break
        t_epoch = time.perf_counter()
        order = order_rng.permutation(n_rows).astype(np.int64)
        digest = C.sha256_bytes(order.tobytes())
        if digest != task["orders"]["epoch_permutation_sha256"][epoch - 1]:
            raise RuntimeError(f"epoch {epoch} order hash {digest} != planned plan")
        verified_orders.append(digest)
        model.train()
        losses: list[float] = []
        weighted: list[tuple[float, int]] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        n_batches = 0
        for i in range(0, order.size, int(C.BATCH_SIZE)):
            if watchdog.stop.is_set():
                aborted = True
                break
            rows = order[i : i + int(C.BATCH_SIZE)]
            t0 = time.perf_counter()
            batch = stager.batch(D.collate(task["train_cache"], rows.tolist()))
            y = stager.labels(task["train_y_z"][rows])
            out = train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=global_steps_run,
                steps_epoch=steps_epoch,
                steps_total=steps_total,
                clip=clip,
            )
            global_steps_run += 1
            n_batches += 1
            losses.append(out["loss"])
            weighted.append((out["loss"], int(out["n"])))
            grad_norms.append(out["grad_norm"])
            clip_events_epoch += int(out["clipped"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append(float(time.perf_counter() - t0))
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{run}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        if n_batches != steps_epoch:
            raise RuntimeError(f"epoch {epoch} ran {n_batches} steps, expected {steps_epoch}")

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        t_eval = time.perf_counter()
        valid_report, valid_pred = evaluate_split(model, task, device, which="valid", batch_size=eval_batch_size)
        valid_seconds = float(time.perf_counter() - t_eval)
        t_mon = time.perf_counter()
        monitor_report, _ = evaluate_split(model, task, device, which="monitor", batch_size=eval_batch_size)
        monitor_seconds = float(time.perf_counter() - t_mon)
        strict = best_epoch is None or float(valid_report["mae_eV"]) < float(best_mae)
        if strict:
            best_epoch = int(epoch)
            best_mae = float(valid_report["mae_eV"])
            payload = checkpoint_payload(
                task=task,
                run=run,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                model_audit=model_audit,
                valid_report=valid_report,
                kind="best_valid_mae",
                extra={"optimizer_state_audit": opt_audit},
            )
            best_checkpoint_sha = _atomic_save(payload, best_path)
        eval_completed.append(int(epoch))
        fixed_reports[int(epoch)] = {
            "mae_eV": float(valid_report["mae_eV"]),
            "mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "crosscheck_delta_eV": float(valid_report["crosscheck_delta_eV"]),
            "n": int(valid_report["n"]),
            "pred_mean_eV": float(valid_report["pred_mean_eV"]),
            "pred_std_eV": float(valid_report["pred_std_eV"]),
            "pred_sha256": valid_report["pred_sha256"],
            "pred_z_sha256": valid_report["pred_z_sha256"],
            "inference_seconds": float(valid_report["inference_seconds"]),
            "step": int(global_steps_run),
        }
        monitor_reports[int(epoch)] = {
            "mae_eV": float(monitor_report["mae_eV"]),
            "mae_eV_numpy": float(monitor_report["mae_eV_numpy"]),
            "n": int(monitor_report["n"]),
            "pred_mean_eV": float(monitor_report["pred_mean_eV"]),
            "pred_std_eV": float(monitor_report["pred_std_eV"]),
            "pred_sha256": monitor_report["pred_sha256"],
            "inference_seconds": float(monitor_report["inference_seconds"]),
        }
        model.train()
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        rng_epoch_hashes.append({"epoch": int(epoch), **rng_hashes})
        update_peak()
        epochs_run = int(epoch)
        cumulative = float(time.perf_counter() - t_start)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        total_n = sum(n for _loss, n in weighted)
        weighted_loss = float(sum(loss * n for loss, n in weighted) / total_n) if total_n else float("nan")
        row = {
            "epoch": int(epoch),
            "evaluated": True,
            "global_step": int(global_steps_run),
            "steps_per_epoch": int(n_batches),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_l1_z": mean_loss,
            "train_l1_z_molecule_weighted": weighted_loss,
            "valid_mae_eV": float(valid_report["mae_eV"]),
            "valid_mae_eV_numpy": float(valid_report["mae_eV_numpy"]),
            "valid_mae_crosscheck_delta_eV": float(valid_report["crosscheck_delta_eV"]),
            "valid_pred_mean_eV": float(valid_report["pred_mean_eV"]),
            "valid_pred_std_eV": float(valid_report["pred_std_eV"]),
            "monitor_mae_eV": float(monitor_report["mae_eV"]),
            "monitor_n": int(monitor_report["n"]),
            "epoch_train_seconds": train_seconds,
            "epoch_valid_seconds": valid_seconds,
            "epoch_monitor_seconds": monitor_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": cumulative,
            "step_ms_mean": float(arr.mean() * 1e3),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
            "steps_per_second": float(n_batches / train_seconds) if train_seconds > 0 else float("nan"),
            "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
            "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
            "peak_rss_mb": C.peak_rss_mb(),
            "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "gradient_clip_events": int(clip_events_epoch),
            "best_epoch_so_far": int(best_epoch),
            "best_valid_mae_eV_so_far": float(best_mae),
            "is_best": bool(strict),
            "checkpoint_seconds": None,
            "rng_cpu_sha256": rng_hashes["torch_cpu_sha256"],
            "rng_cuda_sha256": "|".join(rng_hashes["torch_cuda_sha256"]),
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        C.log(
            f"[{run}] epoch {epoch}/{int(epochs)} trainL1(z) {mean_loss:.5f} "
            f"validMAE {row['valid_mae_eV']:.5f} eV monitorMAE {row['monitor_mae_eV']:.5f} eV "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.1f}s valid {valid_seconds:.1f}s "
            f"step_p50 {row['step_ms_p50']:.2f}ms{' *best*' if strict else ''}"
        )

        if epoch == int(epochs):
            payload = checkpoint_payload(
                task=task,
                run=run,
                epoch=int(epoch),
                global_step=int(global_steps_run),
                model=model,
                optimizer=optimizer,
                rng_state=C.capture_rng_state(),
                model_audit=model_audit,
                valid_report=valid_report,
                kind=f"epoch{int(epochs)}_primary",
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "status_at_save": "COMPLETED",
                },
            )
            final_checkpoint_sha = _atomic_save(payload, final_path)

        t_ck = time.perf_counter()
        payload_last = checkpoint_payload(
            task=task,
            run=run,
            epoch=int(epoch),
            global_step=int(global_steps_run),
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            model_audit=model_audit,
            valid_report=valid_report,
            kind="last_epoch_resumable",
            extra={"optimizer_state_audit": opt_audit, "verified_epoch_orders": len(verified_orders)},
        )
        last_checkpoint_sha = _atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()

    # On resume the earlier epochs were evaluated in the previous process, so the
    # completion test looks at the highest evaluated epoch, not at how many were
    # evaluated here.
    evaluated_through = max(eval_completed) if eval_completed else 0
    completed = (
        epochs_run >= int(epochs) and evaluated_through >= int(epochs) and bool(final_checkpoint_sha)
    )
    status = "COMPLETED" if completed else "INCOMPLETE"
    if completed:
        stop_reason = "COMPLETED"

    rng_end_hashes = C.rng_state_hashes(C.capture_rng_state())
    torch.save(
        {
            "schema": "lcrc-pcqm4mv2-rng-v1",
            "round_id": C.ROUND_ID,
            "run": run,
            "seed": int(task["seed"]),
            "start": rng_start_hashes,
            "end": rng_end_hashes,
            "epochs": rng_epoch_hashes,
            "captured_utc": C.utcnow(),
        },
        rng_path,
    )
    rng_sha = C.sha256_file(rng_path)

    curve_sha = write_csv(curve_path, CURVE_FIELDS, curve_rows)
    total_seconds = float(time.perf_counter() - t_start)
    run_payload = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": C.MODEL_KEY,
        "model_name": C.MODEL_NAME,
        "seed": int(task["seed"]),
        "formal_run": bool(not allow_partial_epochs and int(epochs) == int(C.RUNS_TABLE[run]["epochs"])),
        "status": status,
        "stop_reason": stop_reason,
        "dataset": C.DATASET,
        "n_outputs": int(C.N_OUT),
        "train_size": int(n_rows),
        "train_cache_rows": int(task["train_cache"]["n_molecules"]),
        "parameters": int(model_audit["total_parameters"]),
        "base_parameters": int(model_audit["base_parameters"]),
        "readout_parameters": int(model_audit["readout_parameters"]),
        "epochs": int(epochs),
        "epochs_run": int(epochs_run),
        "global_steps_run": int(global_steps_run),
        "steps_per_epoch": int(steps_epoch),
        "steps_total": int(steps_total),
        "eval_epochs_completed": sorted(eval_completed),
        "eval_epochs_completed_this_process": len(eval_completed),
        "evaluated_through_epoch": int(evaluated_through),
        "fixed_epoch_reports": {str(k): v for k, v in fixed_reports.items()},
        "monitor_epoch_reports": {str(k): v for k, v in monitor_reports.items()},
        "model_audit": model_audit,
        "optimizer_state_audit": opt_audit,
        "resume_audit": resume_audit,
        "batch_plan": plan_meta(task, max(int(epochs_run), 0)),
        "epoch_order_hashes_verified": len(verified_orders),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_valid_mae_eV": None if best_epoch is None else float(best_mae),
        "final_epoch_valid_mae_eV": None
        if str(int(epochs)) not in fixed_reports
        else float(fixed_reports[str(int(epochs))]["mae_eV"]),
        "final_epoch_monitor_mae_eV": None
        if str(int(epochs)) not in monitor_reports
        else float(monitor_reports[str(int(epochs))]["mae_eV"]),
        "best_checkpoint": C.track_rel(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": best_checkpoint_sha or None,
        "primary_checkpoint": C.track_rel(final_path) if final_path.exists() else None,
        "primary_checkpoint_sha256": final_checkpoint_sha or None,
        "last_checkpoint": C.track_rel(last_path) if last_path.exists() else None,
        "last_checkpoint_sha256": last_checkpoint_sha or None,
        "curve": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "rng_state": C.track_rel(rng_path),
        "rng_state_sha256": rng_sha,
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": rng_end_hashes,
        "seed_state": seed_info,
        "spec_hash": task["spec"].hash(),
        "feature_norm": task["feature_norm"],
        "label_norm": task["label_norm"],
        "cache_manifest": task["cache_manifest"],
        "monitor_plan": task["monitor_plan"],
        "data_isolation": task["data_isolation"],
        "recipe": C.recipe_for_run(run),
        "model_dropout": float(model.drop.p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows)),
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "monitor_seconds": float(sum(float(r.get("epoch_monitor_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows])) if curve_rows else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows]))
        if curve_rows
        else 0.0,
        "gpu_peak_allocated_bytes": int(gpu_peak_alloc),
        "gpu_peak_reserved_bytes": int(gpu_peak_reserved),
        "peak_rss_mb": C.peak_rss_mb(),
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "deterministic_settings": deterministic,
        "host_facts": C.host_facts(),
        "gpu_occupancy_start": gpu_start,
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official test-dev / test-challenge labels are hidden and are never indexed",
    }
    C.write_json(out_dir / "run.json", run_payload)
    C.log(
        f"done run={run} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_valid_mae={best_mae:.6f} eV gpu_hours={run_payload['gpu_hours']:.4f}"
    )
    return run_payload


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(C.RUNS_TABLE))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=C.TOTAL_EPOCHS)
    parser.add_argument("--resume", default="")
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "run": args.run,
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        task = D.load_task(args.run)
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        payload = train(
            task,
            run=args.run,
            epochs=int(args.epochs),
            out_dir=out_dir,
            device=device,
            resume=resume,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            eval_batch_size=int(args.eval_batch_size),
        )
        return 0 if payload["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "run": args.run,
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"FAILED run={args.run}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
