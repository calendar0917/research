"""Formal and gated training for the efficiency + width round.

Two things are trained here:

* **capacity runs** ``PCQM_W1_1M`` / ``PCQM_W3_1M`` -- identical data, recipe,
  seed, batch order, batch size and epochs; only the width scale differs.  The
  historical ``PCQM_W2_1M`` is *not* retrained; its saved curve is the control.
* **throughput implementations** used by the A/B benchmark -- the frozen path
  (``pq_train.train_step``, per-step synchronisation, frozen collate) and the
  optimised path (E1 vectorised collate, E2 deferred host synchronisation,
  E3 bounded producer prefetch).  Every implementation computes the same
  numbers; the gates in ``ew_verify`` prove it.

Formal runs are fixed at 40 epochs, batch 128, seed 0, FP32, AdamW (peak LR
3e-4, WD 1e-3), dropout 0.10, 1-epoch warmup + cosine to 1e-5 over the 40-epoch
horizon, L1 on the standardised gap, gradient clip 5.0, no early stopping.
Nothing in this module can change those: ``--epochs`` is checked against the
pre-registered table and ``--limit-batches`` marks a run as non-formal.
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

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_collate as EC  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402
import ew_pipeline as PIPE  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

import torch  # noqa: E402

#: Throughput implementations.  ``prefetch`` is a queue depth (0 = synchronous
#: main-thread collation); ``defer`` moves per-step host synchronisation to
#: fixed-size blocks; ``collate`` picks E1 or the frozen collation.
IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "frozen": {"collate": "frozen", "defer": False, "prefetch": 0, "check_interval": 0},
    "E1": {"collate": "fast", "defer": False, "prefetch": 0, "check_interval": 0},
    "E2": {"collate": "frozen", "defer": True, "prefetch": 0, "check_interval": 64},
    "E3": {"collate": "frozen", "defer": False, "prefetch": 3, "check_interval": 0},
    "E1E3": {"collate": "fast", "defer": False, "prefetch": 3, "check_interval": 0},
    "E1E2": {"collate": "fast", "defer": True, "prefetch": 0, "check_interval": 64},
    "E1E2E3": {"collate": "fast", "defer": True, "prefetch": 3, "check_interval": 64},
    "optimized": {"collate": "fast", "defer": True, "prefetch": 3, "check_interval": 64},
}

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
    "train_molecules_per_second",
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
    "implementation",
]


# ---------------------------------------------------------------------------
# training steps: frozen (history) and deferred-sync (optimised)
# ---------------------------------------------------------------------------


def train_step_deferred(
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
    """Same math as ``pq_train.train_step``; host synchronisation deferred.

    The finiteness checks, the gradient-norm read and the clip event are kept,
    but they stay on the device as 0-dim tensors and are read back once per
    block (``DeferredLog.flush``).  If any flag is set, the block's exact step
    index is recovered from the stored flags and raised there.
    """
    lr = M.set_lrs(optimizer, step=int(global_step), steps_epoch=int(steps_epoch), steps_total=int(steps_total))
    pred = M.predict_z(model, batch)
    n = int(y_z.shape[0])
    if tuple(pred.shape) != (n,):
        raise RuntimeError(f"model output {tuple(pred.shape)} != {(n,)}")
    bad = ~torch.isfinite(pred).all()
    loss = torch.mean(torch.abs(pred - y_z))
    bad = bad | ~torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    bad = bad | ~torch.isfinite(gnorm)
    clipped = gnorm > clip
    optimizer.step()
    return {
        "loss": loss.detach(),
        "grad_norm": gnorm.detach(),
        "bad": bad.detach(),
        "clipped": clipped.detach(),
        "lr": lr,
        "n": n,
    }


class DeferredLog:
    """Collect per-step device scalars; one host read-back per block."""

    def __init__(self, *, check_interval: int, clip: float):
        self.check_interval = int(check_interval)
        self.clip = float(clip)
        self.loss: list[torch.Tensor] = []
        self.grad_norm: list[torch.Tensor] = []
        self.bad: list[torch.Tensor] = []
        self.clipped: list[torch.Tensor] = []
        self.counts: list[int] = []
        self.losses: list[float] = []
        self.n_per_step: list[int] = []
        self.grad_norms: list[float] = []
        self.block_step_ms: list[float] = []
        self.clip_events = 0
        self.steps = 0
        self.blocks = 0
        self.flushes = 0

    def add(self, out: Mapping[str, Any]) -> None:
        self.loss.append(out["loss"])
        self.grad_norm.append(out["grad_norm"])
        self.bad.append(out["bad"])
        self.clipped.append(out["clipped"])
        self.counts.append(int(out["n"]))
        self.steps += 1

    def ready(self) -> bool:
        return self.check_interval > 0 and self.steps > 0 and self.steps % self.check_interval == 0

    def pending(self) -> int:
        return len(self.loss)

    def flush(self, *, block_seconds: float | None = None, first_step: int = 0) -> None:
        """Read the pending block back once (one device synchronisation)."""
        if not self.loss:
            return
        pack = torch.cat(
            [
                torch.stack(self.loss).to(torch.float32),
                torch.stack(self.grad_norm).to(torch.float32),
                torch.stack(self.bad).to(torch.float32),
                torch.stack(self.clipped).to(torch.float32),
            ]
        )
        host = pack.cpu().numpy()
        n = len(self.loss)
        loss = host[:n]
        grad_norm = host[n : 2 * n]
        bad = host[2 * n : 3 * n]
        clipped = host[3 * n : 4 * n]
        if bool(np.any(bad > 0.5)):
            idx = int(np.argmax(bad > 0.5))
            step = int(first_step) + self.steps - n + idx
            raise RuntimeError(f"non-finite loss/gradient/prediction at step {step}")
        if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(grad_norm)):
            step = int(first_step) + self.steps - n
            raise RuntimeError(f"non-finite value in deferred log at block starting at step {step}")
        self.losses.extend(float(x) for x in loss.tolist())
        self.grad_norms.extend(float(x) for x in grad_norm.tolist())
        self.n_per_step.extend(self.counts)
        self.counts.clear()
        self.clip_events += int(np.sum(clipped > 0.5))
        if block_seconds is not None and n:
            self.block_step_ms.extend([1e3 * float(block_seconds) / n] * n)
        self.loss.clear()
        self.grad_norm.clear()
        self.bad.clear()
        self.clipped.clear()
        self.blocks += 1
        self.flushes += 1


# ---------------------------------------------------------------------------
# epoch machinery
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
    return C.PQ_TRAIN.plan_meta(task, int(consumed))


def scheduler_state(global_step: int, optimizer: torch.optim.Optimizer, task: Mapping[str, Any]) -> dict[str, Any]:
    return C.PQ_TRAIN.scheduler_state(global_step, optimizer, task)


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
    implementation: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": task["model"],
        "model_name": task["model_name"],
        "scale": int(task["scale"]),
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
        "implementation": dict(implementation),
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
    return C.PQ_TRAIN._rng_hash_diffs(saved, now)


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
    if int(payload.get("scale", -1)) != int(task["scale"]):
        raise RuntimeError(f"checkpoint scale {payload.get('scale')} != {task['scale']}")
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
    diffs = _rng_hash_diffs(payload.get("rng_state_hashes") or {}, now_hashes)
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
        "checkpoint_implementation": payload.get("implementation"),
    }


def replay_epoch_orders(task: Mapping[str, Any], upto: int) -> list[str]:
    return C.PQ_TRAIN.replay_epoch_orders(task, int(upto))


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
    implementation: str = "optimized",
    resume: Mapping[str, Any] | None = None,
    soft_seconds: float,
    hard_seconds: float,
    eval_batch_size: int = 512,
    allow_partial_epochs: bool = False,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    if run not in C.RUNS_TABLE:
        raise KeyError(run)
    if implementation not in IMPLEMENTATIONS:
        raise KeyError(implementation)
    impl = dict(IMPLEMENTATIONS[implementation])
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = int(task["n_rows"])
    steps_epoch = int(task["steps_epoch"]) if limit_batches is None else int(limit_batches)
    steps_total = int(task["steps_total"])
    formal = bool(
        limit_batches is None
        and not allow_partial_epochs
        and int(epochs) == int(C.RUNS_TABLE[run]["epochs"])
        and implementation in ("optimized", "E1E2E3")
    )
    if limit_batches is not None:
        C.log(f"[{run}] GATED run: only {limit_batches} batches/epoch -> NOT a formal run")
    if int(epochs) != int(C.RUNS_TABLE[run]["epochs"]) and not (allow_partial_epochs or limit_batches is not None):
        raise RuntimeError(f"formal run must use {C.RUNS_TABLE[run]['epochs']} epochs, got {epochs}")

    gpu_start = C.gpu_occupancy()
    deterministic = C.apply_deterministic_settings()

    seed_info: Mapping[str, Any] | None = None
    if resume is not None:
        seed_info = C.seed_everything(int(task["seed"]))
    else:
        seed_info = C.seed_everything(int(task["seed"]))
    model, model_audit = M.build_model(task["spec"], task["norm"], scale=int(task["scale"]), seed=int(task["seed"]), device=device)
    if int(model_audit["total_parameters"]) != int(task["expected_parameters"]):
        raise RuntimeError(
            f"built {model_audit['total_parameters']} parameters, pre-registered {task['expected_parameters']}"
        )
    optimizer, opt_audit = M.make_optimizer(model)

    resume_audit: Mapping[str, Any] | None = None
    if resume is not None:
        resume_audit = restore(resume, model, optimizer, run=run, task=task)
        start_epoch = int(resume["epoch"]) + 1
        global_steps_run = int(resume["global_step"])
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
    else:
        rng_hashes = C.rng_state_hashes(C.capture_rng_state())
        start_epoch = 1
        global_steps_run = 0
    rng_start_hashes = dict(rng_hashes)

    order_rng = np.random.default_rng(int(task["orders"]["order_seed"]))
    for epoch in range(1, start_epoch):
        _ = order_rng.permutation(n_rows)
    verified_orders = replay_epoch_orders(task, start_epoch - 1)

    stager = PinnedStager(device, depth=2)
    collate_fn = lambda cache, rows: EC.collate_for(cache, rows, implementation=str(impl["collate"]))  # noqa: E731
    prefetcher = (
        PIPE.Prefetcher(task["train_cache"], task["train_y_z"], collator=collate_fn, batch_size=int(C.BATCH_SIZE), depth=int(impl["prefetch"]))
        if int(impl["prefetch"]) > 0
        else None
    )
    watchdog = Watchdog(hard_seconds)
    clip = float(C.RECIPE["gradient_clip"])
    check_interval = int(impl["check_interval"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    impl_payload = {
        "name": implementation,
        "collate": impl["collate"],
        "deferred_sync": bool(impl["defer"]),
        "check_interval": int(check_interval),
        "prefetch_depth": int(impl["prefetch"]),
        "frozen_reference": implementation == "frozen",
        "note": (
            "frozen = historical code path (frozen collate, per-step host sync, no prefetch)"
            if implementation == "frozen"
            else "optimized = E1 vectorised collate + E2 block-deferred host sync + E3 bounded producer prefetch; "
            "all input tensors, losses, gradients and updates are checked equal to the frozen path"
        ),
    }

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
    producer_counters: dict[str, Any] = {}
    total_sync_flushes = 0
    total_steps_measured = 0

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
        mean_losses: list[float] = []
        weighted: list[tuple[float, int]] = []
        step_times: list[float] = []
        grad_norms: list[float] = []
        clip_events_epoch = 0
        aborted = False
        n_batches = 0
        molecules_epoch = 0
        deferred = DeferredLog(check_interval=check_interval, clip=clip) if bool(impl["defer"]) else None
        if prefetcher is not None:
            source = prefetcher.epoch(order)
        else:
            source = None

        block_start = time.perf_counter()
        block_first_step = int(global_steps_run)
        iterator = (
            source
            if source is not None
            else (
                (order[i : i + int(C.BATCH_SIZE)], None, None)
                for i in range(0, order.size, int(C.BATCH_SIZE))
            )
        )
        for rows, raw, y_pref in iterator:
            if watchdog.stop.is_set():
                aborted = True
                break
            if limit_batches is not None and n_batches >= int(limit_batches):
                break
            t0 = time.perf_counter()
            if raw is None:
                raw = collate_fn(task["train_cache"], rows.tolist())
                y_np = np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32)
            else:
                y_np = y_pref
            batch = stager.batch(raw)
            y = stager.labels(y_np)
            if deferred is None:
                out = C.PQ_TRAIN.train_step(
                    model,
                    optimizer,
                    batch,
                    y,
                    global_step=global_steps_run,
                    steps_epoch=int(task["steps_epoch"]),
                    steps_total=steps_total,
                    clip=clip,
                )
                global_steps_run += 1
                n_batches += 1
                molecules_epoch += int(rows.size)
                mean_losses.append(out["loss"])
                weighted.append((out["loss"], int(out["n"])))
                grad_norms.append(out["grad_norm"])
                clip_events_epoch += int(out["clipped"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_times.append(float(time.perf_counter() - t0))
            else:
                out = train_step_deferred(
                    model,
                    optimizer,
                    batch,
                    y,
                    global_step=global_steps_run,
                    steps_epoch=int(task["steps_epoch"]),
                    steps_total=steps_total,
                    clip=clip,
                )
                global_steps_run += 1
                n_batches += 1
                molecules_epoch += int(rows.size)
                deferred.add(out)
                if deferred.ready():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    block_seconds = float(time.perf_counter() - block_start)
                    deferred.flush(block_seconds=block_seconds, first_step=block_first_step)
                    total_sync_flushes += 1
                    block_start = time.perf_counter()
                    block_first_step = int(global_steps_run)
        if source is not None:
            source.close()
        if aborted:
            stop_reason = "HARD_WALL_LIMIT"
            C.log(f"[{run}] hard wall limit during epoch {epoch}; partial epoch discarded")
            break
        if deferred is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if deferred.pending():
                block_seconds = float(time.perf_counter() - block_start)
                deferred.flush(block_seconds=block_seconds, first_step=block_first_step)
                total_sync_flushes += 1
            mean_losses = list(deferred.losses)
            grad_norms = list(deferred.grad_norms)
            clip_events_epoch = int(deferred.clip_events)
            step_times = list(deferred.block_step_ms)
            if len(deferred.n_per_step) != len(mean_losses):
                raise RuntimeError("deferred per-step molecule counts are inconsistent")
            weighted = [(v, int(n)) for v, n in zip(mean_losses, deferred.n_per_step)]
        total_steps_measured += int(n_batches)
        if n_batches != steps_epoch:
            raise RuntimeError(f"epoch {epoch} ran {n_batches} steps, expected {steps_epoch}")
        if not mean_losses:
            raise RuntimeError(f"epoch {epoch} produced no loss values")

        train_seconds = float(time.perf_counter() - t_epoch)
        arr = np.asarray(step_times, dtype=np.float64)
        t_eval = time.perf_counter()
        valid_report, valid_pred = C.PQ_TRAIN.evaluate_split(
            model, task, device, which="valid", batch_size=eval_batch_size
        )
        valid_seconds = float(time.perf_counter() - t_eval)
        t_mon = time.perf_counter()
        monitor_report, _ = C.PQ_TRAIN.evaluate_split(
            model, task, device, which="monitor", batch_size=eval_batch_size
        )
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
                implementation=impl_payload,
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
        mean_loss = float(np.mean(mean_losses))
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
            "train_molecules_per_second": float(molecules_epoch / train_seconds)
            if train_seconds > 0
            else float("nan"),
            "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
            "step_ms_p50": float(np.percentile(arr, 50) * 1e3) if arr.size else float("nan"),
            "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
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
            "implementation": implementation,
        }
        curve_rows.append(row)
        clip_events_total += int(clip_events_epoch)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        C.log(
            f"[{run}:{implementation}] epoch {epoch}/{int(epochs)} trainL1(z) {mean_loss:.5f} "
            f"validMAE {row['valid_mae_eV']:.5f} eV monitorMAE {row['monitor_mae_eV']:.5f} eV "
            f"LR {row['learning_rate']:.3e} train {train_seconds:.1f}s "
            f"({row['train_molecules_per_second']:.0f} mol/s) valid {valid_seconds:.1f}s "
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
                implementation=impl_payload,
                extra={
                    "optimizer_state_audit": opt_audit,
                    "resume_audit": resume_audit,
                    "consumed_epochs": int(epoch),
                    "consumed_steps": int(global_steps_run),
                    "status_at_save": "COMPLETED",
                    "formal_run": formal,
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
            implementation=impl_payload,
            extra={"optimizer_state_audit": opt_audit, "verified_epoch_orders": len(verified_orders)},
        )
        last_checkpoint_sha = _atomic_save(payload_last, last_path)
        row["checkpoint_seconds"] = float(time.perf_counter() - t_ck)
        write_csv(curve_path, CURVE_FIELDS, curve_rows)
        if prefetcher is not None:
            producer_counters = dict(prefetcher.counters)
        update_peak()
        if watchdog.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break

    watchdog.cancel()
    stager.release()
    if prefetcher is not None:
        prefetcher.close()
        producer_counters = dict(prefetcher.counters)

    evaluated_through = max(eval_completed) if eval_completed else 0
    completed = epochs_run >= int(epochs) and evaluated_through >= int(epochs) and bool(final_checkpoint_sha)
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
    train_seconds_total = float(sum(float(r.get("epoch_train_seconds") or 0.0) for r in curve_rows))
    molecules_seen = int(sum(int(r["steps_per_epoch"]) for r in curve_rows) * int(C.BATCH_SIZE))
    run_payload = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "run": run,
        "model": task["model"],
        "model_name": task["model_name"],
        "scale": int(task["scale"]),
        "seed": int(task["seed"]),
        "formal_run": bool(formal),
        "implementation": impl_payload,
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
        "limit_batches": None if limit_batches is None else int(limit_batches),
        "eval_epochs_completed": sorted(eval_completed),
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
        if int(epochs) not in fixed_reports
        else float(fixed_reports[int(epochs)]["mae_eV"]),
        "final_epoch_monitor_mae_eV": None
        if int(epochs) not in monitor_reports
        else float(monitor_reports[int(epochs)]["mae_eV"]),
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
        "inherited_task_identity": task.get("inherited_task_identity"),
        "recipe": C.recipe_for_run(run),
        "model_dropout": float(model.drop.p),
        "precision": "fp32",
        "grad_clip": clip,
        "grad_clip_events": int(clip_events_total),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "train_seconds": train_seconds_total,
        "valid_seconds": float(sum(float(r.get("epoch_valid_seconds") or 0.0) for r in curve_rows)),
        "monitor_seconds": float(sum(float(r.get("epoch_monitor_seconds") or 0.0) for r in curve_rows)),
        "epoch_seconds_mean": float(np.mean([r["epoch_seconds"] for r in curve_rows])) if curve_rows else 0.0,
        "steps_per_second_mean": float(np.mean([r["steps_per_second"] for r in curve_rows]))
        if curve_rows
        else 0.0,
        "throughput": {
            "measured_steps": int(total_steps_measured),
            "train_molecules": int(molecules_seen),
            "train_seconds": train_seconds_total,
            "molecules_per_second_train": float(molecules_seen / train_seconds_total)
            if train_seconds_total > 0
            else None,
            "molecules_per_second_total_wall": float(molecules_seen / total_seconds) if total_seconds > 0 else None,
            "steps_per_second_train": float(sum(int(r["steps_per_epoch"]) for r in curve_rows) / train_seconds_total)
            if train_seconds_total > 0
            else None,
            "step_ms_mean_all_epochs": float(
                np.mean([float(r["step_ms_mean"]) for r in curve_rows])
            )
            if curve_rows
            else None,
            "step_ms_p50_all_epochs": float(np.mean([float(r["step_ms_p50"]) for r in curve_rows]))
            if curve_rows
            else None,
            "step_ms_p90_all_epochs": float(np.mean([float(r["step_ms_p90"]) for r in curve_rows]))
            if curve_rows
            else None,
            "deferred_sync_flushes": int(total_sync_flushes),
            "producer_counters": producer_counters,
            "step_timing_mode": "block-deferred" if bool(impl["defer"]) else "per-step (frozen)",
        },
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
    C.write_json(out_dir / "run.json", C.json_safe(run_payload))
    C.log(
        f"done run={run} impl={implementation} status={status} epochs={epochs_run}/{int(epochs)} "
        f"best_epoch={best_epoch} best_valid_mae={best_mae:.6f} eV gpu_hours={run_payload['gpu_hours']:.4f} "
        f"mol/s={run_payload['throughput']['molecules_per_second_train']:.1f}"
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
    parser.add_argument("--implementation", default="optimized", choices=sorted(IMPLEMENTATIONS))
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--verify-hashes", action="store_true", default=True)
    parser.add_argument("--no-verify-hashes", dest="verify_hashes", action="store_false")
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
        task = DATA.load_task(args.run, verify_hashes=bool(args.verify_hashes))
        resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
        payload = train(
            task,
            run=args.run,
            epochs=int(args.epochs),
            out_dir=out_dir,
            device=device,
            implementation=str(args.implementation),
            resume=resume,
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            eval_batch_size=int(args.eval_batch_size),
            limit_batches=None if int(args.limit_batches) <= 0 else int(args.limit_batches),
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
