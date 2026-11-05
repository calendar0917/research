"""Step-level performance decomposition of the frozen PCBA training loop.

Measured object: the *unmodified* ``pc_train.py`` inner loop (frozen LCRC-C1,
AdamW, per-step cosine, masked BCE) over the fixed 300-step batch plan.

Three passes over the **same** batches, all after 20 warmup steps:

``coarse``
    The baseline loop, unchanged except for CUDA events recorded around the
    phases and a single ``_sync()`` at the end of each step.
    Events measure *stream* spans, so CUDA-async enqueue is not mistaken for
    kernel time; enqueue time is recorded separately with ``perf_counter``.
    This pass is the throughput reference.

``isolated``
    The same arithmetic with a ``_sync()`` at every phase
    boundary, which makes the per-phase attribution unambiguous (the loop is
    already serialised by ``torch.isfinite`` / ``float(...)`` / ``.item()``
    syncs, so this costs little but removes any doubt).

``profiler``
    A ``torch.profiler`` window (CPU + CUDA activities) over 20 steps of the
    same loop: top CUDA kernels, top CPU ops, kernel count per step.

Explicitly un-measurable / recorded as limitations:
  * page-fault *time* is not attributable per phase; minor/major fault counts
    are reported from ``getrusage`` around each pass,
  * NVML utilisation is sampled at 250 ms, i.e. it is a coarse *observation*,
    not a per-step measurement,
  * CPU frequency scaling / co-tenant activity on the shared host cannot be
    excluded; timings are reported as distributions, not single numbers.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pf_common as PF

PILOT = PF.PILOT


def _imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_imports()

import torch  # noqa: E402

import bp_model as BM  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PCM  # noqa: E402
import pf_collate as PC_C  # noqa: E402
import pc_common as PC  # noqa: E402

RECIPE = dict(PC.RECIPE)


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    if not a.size:
        return {"n": 0, "sum_seconds": 0.0, "mean_seconds": 0.0, "p50_seconds": 0.0,
                "p90_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0,
                "total_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
    return {
        "n": int(a.size),
        "sum_seconds": float(a.sum()),
        "mean_seconds": float(a.mean()),
        "p50_seconds": float(np.percentile(a, 50)),
        "p90_seconds": float(np.percentile(a, 90)),
        "min_seconds": float(a.min()),
        "max_seconds": float(a.max()),
        "total_ms": float(a.sum() * 1e3),
        "p50_ms": float(np.percentile(a, 50) * 1e3),
        "p90_ms": float(np.percentile(a, 90) * 1e3),
    }


def _share(part: float, whole: float) -> float:
    return float(part / whole) if whole else 0.0


# ---------------------------------------------------------------------------
# state construction (identical to pc_train.train)
# ---------------------------------------------------------------------------


def build_state(inputs: PF.Inputs, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(PF.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PF.SEED)
    model, model_info = PCM.build_c1(inputs.spec, inputs.norm, PC.N_TASKS, PF.SEED, device)
    steps_per_epoch = int(math.ceil(len(inputs.train_rows) / PF.BATCH_SIZE))
    total_updates = steps_per_epoch * int(RECIPE["scheduler_horizon_epochs"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(RECIPE["peak_lr"]),
        betas=tuple(float(x) for x in RECIPE["adam_betas"]),
        eps=float(RECIPE["adam_eps"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1), eta_min=float(RECIPE["min_lr"])
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model_info": model_info,
        "steps_per_epoch": steps_per_epoch,
        "scheduler_total_updates": total_updates,
    }


def reset_rng() -> None:
    torch.manual_seed(PF.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PF.SEED)


def fresh_state(inputs: PF.Inputs, device: torch.device) -> dict[str, Any]:
    reset_rng()
    return build_state(inputs, device)


# ---------------------------------------------------------------------------
# baseline step (verbatim arithmetic); returns the un-enqueued loss tensor
# ---------------------------------------------------------------------------


def baseline_step(state, inputs, rows, device, collate=PC_C.collate_reference):
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    batch = PCM.to_device(collate(inputs.cache, rows.tolist()), device)
    y = torch.as_tensor(inputs.labels[rows], dtype=torch.float32, device=device)
    logits = model(batch)
    if logits.shape != (len(rows), PC.N_TASKS):
        raise RuntimeError(f"model output {tuple(logits.shape)} != {(len(rows), PC.N_TASKS)}")
    loss = MET.masked_bce_loss(logits, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {float(loss)}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["gradient_clip"])))
    if not math.isfinite(gnorm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    scheduler.step()
    loss_value = float(loss.detach().cpu().item())
    return loss_value, gnorm


# ---------------------------------------------------------------------------
# GPU observations
# ---------------------------------------------------------------------------


class NvmlSampler(threading.Thread):
    """Coarse ``nvidia-smi`` utilisation samples (not per-step measurements)."""

    def __init__(self, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.max.sm,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5)
                for line in out.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    self.samples.append(
                        {
                            "index": int(parts[0]),
                            "util_gpu_pct": float(parts[1]),
                            "util_mem_pct": float(parts[2]),
                            "mem_used_mib": float(parts[3]),
                            "clock_sm_mhz": float(parts[4]),
                            "clock_sm_max_mhz": float(parts[5]),
                            "power_w": float(parts[6]),
                        }
                    )
            except Exception:  # pragma: no cover - NVML may be broken on the host
                self.samples.append({"error": True})
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        ok = [s for s in self.samples if not s.get("error")]
        if not ok:
            return {
                "available": False,
                "note": "nvidia-smi sampling failed or returned no samples on this host",
            }
        util = np.asarray([s["util_gpu_pct"] for s in ok], dtype=np.float64)
        clk = np.asarray([s["clock_sm_mhz"] for s in ok], dtype=np.float64)
        clkmax = np.asarray([s["clock_sm_max_mhz"] for s in ok], dtype=np.float64)
        mem = np.asarray([s["util_mem_pct"] for s in ok], dtype=np.float64)
        pwr = np.asarray([s["power_w"] for s in ok], dtype=np.float64)
        return {
            "available": True,
            "interval_s": self.interval_s,
            "n_samples": int(util.size),
            "device_indices": sorted({int(s["index"]) for s in ok}),
            "util_gpu_pct_mean": float(util.mean()),
            "util_gpu_pct_p50": float(np.percentile(util, 50)),
            "util_gpu_pct_max": float(util.max()),
            "util_mem_pct_mean": float(mem.mean()),
            "clock_sm_mhz_mean": float(clk.mean()),
            "clock_sm_mhz_max_ratio_mean": float((clk / np.maximum(clkmax, 1)).mean()),
            "power_w_mean": float(pwr.mean()),
            "granularity_note": (
                "sampled at 250 ms; use the per-step CUDA-event spans for phase attribution, "
                "this is only a coarse utilisation observation"
            ),
        }


def _sync() -> None:
    """Drain the stream when a GPU is present (no-op on a CPU-only box)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _NoEvent:
    def record(self) -> None:  # pragma: no cover - CPU-only dry run
        pass

    def elapsed_time(self, _other: Any) -> float:  # pragma: no cover
        return 0.0


def _new_events(n: int) -> list[Any]:
    if torch.cuda.is_available():
        return [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    return [_NoEvent() for _ in range(n)]


def _rusage() -> dict[str, int]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "ru_utime_ms": int(r.ru_utime * 1e3),
        "ru_stime_ms": int(r.ru_stime * 1e3),
        "minor_page_faults": int(r.ru_minflt),
        "major_page_faults": int(r.ru_majflt),
        "maxrss_kib": int(r.ru_maxrss),
    }


def _rusage_delta(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    return {k: int(b[k] - a[k]) for k in a}


# ---------------------------------------------------------------------------
# pass 1: coarse (baseline loop + CUDA events, one sync per step)
# ---------------------------------------------------------------------------


def pass_coarse(state, inputs, batches, device) -> dict[str, Any]:
    """Baseline loop; CUDA events give *stream* spans, ``perf_counter`` enqueue."""
    model, optimizer, scheduler = state["model"], state["optimizer"], state["scheduler"]
    ev = _new_events(7)
    has_events = bool(torch.cuda.is_available())
    cpu: dict[str, list[float]] = defaultdict(list)
    gpu: dict[str, list[float]] = defaultdict(list)
    wall: list[float] = []
    model.train()
    _sync()
    for rows in batches:
        t_step = time.perf_counter()
        ev[0].record()
        t = time.perf_counter()
        batch = PC_C.collate_reference(inputs.cache, rows.tolist())
        cpu["collate"].append(time.perf_counter() - t)
        ev[1].record()
        t = time.perf_counter()
        batch_g = PCM.to_device(batch, device)
        cpu["to_device_enqueue"].append(time.perf_counter() - t)
        t = time.perf_counter()
        y = torch.as_tensor(inputs.labels[rows], dtype=torch.float32, device=device)
        cpu["labels_enqueue"].append(time.perf_counter() - t)
        ev[2].record()

        t = time.perf_counter()
        logits = model(batch_g)
        cpu["forward_enqueue"].append(time.perf_counter() - t)
        ev[3].record()

        t = time.perf_counter()
        loss = MET.masked_bce_loss(logits, y)
        finite = bool(torch.isfinite(loss))
        cpu["loss_and_sync"].append(time.perf_counter() - t)
        ev[4].record()

        t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        cpu["backward_enqueue"].append(time.perf_counter() - t)
        ev[5].record()

        t = time.perf_counter()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["gradient_clip"])))
        optimizer.step()
        scheduler.step()
        ev[6].record()
        loss_value = float(loss.detach().cpu().item())
        cpu["optimizer_and_sync"].append(time.perf_counter() - t)

        _sync()
        wall.append(time.perf_counter() - t_step)
        # spans are read *after* the stream is drained, so they are true GPU time
        if has_events:
            for name, i in (("h2d", 1), ("forward", 2), ("loss", 3), ("backward", 4), ("optimizer", 5)):
                gpu[name].append(float(ev[i].elapsed_time(ev[i + 1])) * 1e-3)
        if not finite:
            raise RuntimeError("non-finite loss")
        if not math.isfinite(gnorm):
            raise RuntimeError("non-finite gradient norm")
    return {
        "wall": wall,
        "cpu": dict(cpu),
        "gpu": dict(gpu),
        "stage_span": gpu["forward"],
        "rows": len(batches),
    }


# ---------------------------------------------------------------------------
# pass 2: isolated (sync at every phase boundary)
# ---------------------------------------------------------------------------


def pass_isolated(state, inputs, batches, device) -> dict[str, Any]:
    model, optimizer, scheduler = state["model"], state["optimizer"], state["scheduler"]
    phases = (
        "collate_cpu",
        "h2d_and_labels",
        "forward_gpu",
        "loss_gpu",
        "backward_gpu",
        "optimizer_gpu",
        "scalar_read_sync",
    )
    acc = defaultdict(list)
    wall: list[float] = []
    model.train()
    _sync()
    for rows in batches:
        t_step = time.perf_counter()

        t = time.perf_counter()
        batch = PC_C.collate_reference(inputs.cache, rows.tolist())
        acc["collate_cpu"].append(time.perf_counter() - t)

        _sync()
        t = time.perf_counter()
        batch_g = PCM.to_device(batch, device)
        y = torch.as_tensor(inputs.labels[rows], dtype=torch.float32, device=device)
        _sync()
        acc["h2d_and_labels"].append(time.perf_counter() - t)

        t = time.perf_counter()
        logits = model(batch_g)
        _sync()
        acc["forward_gpu"].append(time.perf_counter() - t)

        t = time.perf_counter()
        loss = MET.masked_bce_loss(logits, y)
        finite = bool(torch.isfinite(loss))
        _sync()
        acc["loss_gpu"].append(time.perf_counter() - t)

        t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _sync()
        acc["backward_gpu"].append(time.perf_counter() - t)

        t = time.perf_counter()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(RECIPE["gradient_clip"])))
        optimizer.step()
        scheduler.step()
        _sync()
        acc["optimizer_gpu"].append(time.perf_counter() - t)

        t = time.perf_counter()
        loss_value = float(loss.detach().cpu().item())
        acc["scalar_read_sync"].append(time.perf_counter() - t)

        wall.append(time.perf_counter() - t_step)
        if not finite or not math.isfinite(gnorm):
            raise RuntimeError("non-finite loss or gradient norm")
    return {"wall": wall, "phases": {k: list(v) for k, v in acc.items()}, "phase_order": list(phases)}


# ---------------------------------------------------------------------------
# pass 3: torch.profiler kernel window
# ---------------------------------------------------------------------------


def pass_profiler(state, inputs, batches, device) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    model = state["model"]
    model.train()
    for rows in batches[:5]:  # warmup outside the window
        baseline_step(state, inputs, rows, device)
    _sync()
    cuda_available = torch.cuda.is_available()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda_available else [])
    with profile(activities=acts, record_shapes=False, profile_memory=False) as prof:
        for rows in batches[:PF.PROFILER_STEPS]:
            baseline_step(state, inputs, rows, device)
        _sync()
    ka = prof.key_averages()
    table = ka.table(sort_by="self_cuda_time_total", row_limit=25)

    def _top(key: str, n: int = 15) -> list[dict[str, Any]]:
        rows = []
        for e in ka:
            rows.append(
                {
                    "key": str(e.key),
                    "calls": int(e.count),
                    "cpu_us_total": float(e.self_cpu_time_total),
                    "cuda_us_total": float(getattr(e, "self_cuda_time_total", 0.0)),
                    "cuda_us_mean": float(getattr(e, "self_cuda_time_total", 0.0)) / max(int(e.count), 1),
                }
            )
        rows.sort(key=lambda r: r[key], reverse=True)
        return rows[:n]

    gpu_total = float(sum(getattr(e, "self_cuda_time_total", 0.0) for e in ka))
    cpu_total = float(sum(e.self_cpu_time_total for e in ka))
    return {
        "steps": int(PF.PROFILER_STEPS),
        "activities": [str(a) for a in acts],
        "cuda_time_total_us": gpu_total,
        "cpu_time_total_us": cpu_total,
        "cuda_us_per_step": gpu_total / PF.PROFILER_STEPS,
        "cpu_us_per_step": cpu_total / PF.PROFILER_STEPS,
        "top_cuda": _top("cuda_us_total"),
        "top_cpu": _top("cpu_us_total"),
        "table": table.splitlines(),
        "kernel_count_total": int(sum(int(e.count) for e in ka)),
        "kernel_count_per_step": float(sum(int(e.count) for e in ka) / PF.PROFILER_STEPS),
    }


# ---------------------------------------------------------------------------
# data pipeline micro-benchmarks
# ---------------------------------------------------------------------------


def bench_dataloader(inputs, batches, device, *, with_device: bool) -> dict[str, Any]:
    """collate alone vs collate + to_device over the same batches (CPU only)."""
    collate_s: list[float] = []
    transfer_s: list[float] = []
    _sync()
    for i, rows in enumerate(batches):
        t = time.perf_counter()
        batch = PC_C.collate_reference(inputs.cache, rows.tolist())
        collate_s.append(time.perf_counter() - t)
        if with_device:
            if device.type == "cuda":
                _sync()
            t = time.perf_counter()
            PCM.to_device(batch, device)
            if device.type == "cuda":
                _sync()
            transfer_s.append(time.perf_counter() - t)
    out = {"collate": _stats(collate_s)}
    if with_device:
        out["transfer"] = _stats(transfer_s)
        out["collate_plus_transfer_total_s"] = float(sum(collate_s) + sum(transfer_s))
    return out


def bench_collate_components(inputs, batches) -> dict[str, Any]:
    """Instrumented collate: attribute CPU cost inside one batch."""
    acc: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    for rows in batches:
        out, marks = PC_C.collate_instrumented(inputs.cache, rows.tolist())
        prev = 0.0
        for name, t in marks.rows:
            acc[name].append(t - prev)
            prev = t
        totals.append(prev)
    groups = {
        "ptr_read": ("ptr_read",),
        "index_build_arange_loops": (
            "atom_idx_concat",
            "atom_local_concat",
            "bond_idx_concat",
            "bond_local_concat",
            "nbr_index_build",
            "path_index_build",
        ),
        "gather_from_mmap": (
            "bond_index_gather",
            "nbr_gather",
            "path_index_gather",
        ),
        "field_gather_and_tensor_wrap": ("field_gather_and_wrap",),
    }
    out: dict[str, Any] = {"segments": {k: _stats(v) for k, v in acc.items()}, "total": _stats(totals)}
    out["groups"] = {
        g: _stats([sum(row) for row in zip(*[acc[n] for n in names])]) for g, names in groups.items()
    }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(PF.ANALYSIS / "profile.json"))
    parser.add_argument("--timed-steps", type=int, default=PF.TIMED_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=PF.WARMUP_STEPS)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--skip-isolated", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    PF.log(f"environment: {PF.environment()}")

    inputs = PF.load_inputs()
    need = args.warmup_steps + args.timed_steps
    batches = PF.fixed_batches(inputs.train_rows, max(need, PF.PROFILER_STEPS + 5))
    plan = PF.batch_plan_fingerprint(batches[:need])
    PF.log(f"batch plan: {plan['rows_sha256'][:16]} over {plan['n_batches']} batches")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _sync()

    result: dict[str, Any] = {
        "schema": "lcrc-pcba-pipeline-profile-v1",
        "round_id": PF.ROUND_ID,
        "created_utc": PF.utcnow(),
        "git_commit": PF.git_commit(),
        "purpose": (
            "decompose one PCBA training step of the *frozen* pipeline into CPU data "
            "pipeline / host-to-device / GPU forward / loss / backward / optimizer / "
            "python+sync, before any optimisation is attempted"
        ),
        "environment": PF.environment(),
        "device": str(device),
        "protocol": {
            "batch_size": PF.BATCH_SIZE,
            "warmup_steps": int(args.warmup_steps),
            "timed_steps": int(args.timed_steps),
            "seed": PF.SEED,
            "batch_plan": plan,
            "optimizer": RECIPE["optimizer"],
            "peak_lr": RECIPE["peak_lr"],
            "scheduler": RECIPE["scheduler"],
            "gradient_clip": RECIPE["gradient_clip"],
            "dtype": "float32",
        },
        "inputs": {
            "cache": inputs.identity,
            "labels": {
                "path": PF.track_rel(PF.LABELS_PATH),
                "sha256": PF.sha256_file(PF.LABELS_PATH),
                "shape": list(inputs.labels.shape),
                "n_test_rows_all_nan": int(np.isnan(inputs.labels[inputs.test_rows]).all(axis=1).sum()),
            },
            "split_sizes": {k: int(v.size) for k, v in inputs.split.items()},
            "n_tasks": PC.N_TASKS,
            "feature_spec_hash": inputs.spec.hash(),
            "norm_fingerprint": inputs.norm["norm_fingerprint"],
        },
    }

    state = fresh_state(inputs, device)
    result["model"] = {
        "variant": PC.MODEL_VARIANT,
        "parameters": int(sum(p.numel() for p in state["model"].parameters())),
        "steps_per_epoch": int(state["steps_per_epoch"]),
        "scheduler_total_updates": int(state["scheduler_total_updates"]),
        "model_info": state["model_info"],
    }

    # ---- warmup (not measured) -------------------------------------------
    reset_rng()
    _sync()
    t_warm = time.perf_counter()
    for rows in batches[: args.warmup_steps]:
        baseline_step(state, inputs, rows, device)
    _sync()
    result["warmup_seconds"] = float(time.perf_counter() - t_warm)
    PF.log(f"warmup {args.warmup_steps} steps in {result['warmup_seconds']:.2f}s")

    timed = batches[args.warmup_steps : args.warmup_steps + args.timed_steps]

    # ---- pass 1: coarse --------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = NvmlSampler() if device.type == "cuda" else None
    if sampler:
        sampler.start()
    ru0 = _rusage()
    t0 = time.perf_counter()
    coarse = pass_coarse(state, inputs, timed, device)
    wall_coarse = time.perf_counter() - t0
    ru1 = _rusage()
    if sampler:
        sampler.stop()
    gpu_mem = {}
    if device.type == "cuda":
        gpu_mem = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
    result["coarse"] = {
        "n_steps": int(len(timed)),
        "wall_seconds": float(wall_coarse),
        "step_wall": _stats(coarse["wall"]),
        "steps_per_second": float(len(timed) / wall_coarse),
        "cpu_enqueue": {k: _stats(v) for k, v in coarse["cpu"].items()},
        "cpu_enqueue_total_seconds": float(sum(sum(v) for v in coarse["cpu"].values())),
        "gpu_event_spans": {k: _stats(v) for k, v in coarse["gpu"].items()},
        "gpu_event_span_total_seconds": float(sum(sum(v) for v in coarse["gpu"].values())),
        "rusage": _rusage_delta(ru0, ru1),
        "gpu_memory": gpu_mem,
        "nvml": sampler.summary() if sampler else {"available": False, "note": "cpu run"},
        "note": (
            "CUDA events are recorded in-stream and read only after the step is "
            "synchronised, so gpu_event_spans are kernel execution spans, not "
            "enqueue time; cpu_enqueue is the wall time spent adding work to the stream"
        ),
    }
    gpu_busy = result["coarse"]["gpu_event_span_total_seconds"]
    result["coarse"]["gpu_busy_seconds"] = float(gpu_busy)
    result["coarse"]["gpu_busy_fraction"] = _share(gpu_busy, wall_coarse)
    result["coarse"]["cpu_bound_fraction"] = 1.0 - result["coarse"]["gpu_busy_fraction"]
    del coarse
    PF.log(
        f"coarse: {len(timed)} steps in {wall_coarse:.2f}s "
        f"= {len(timed) / wall_coarse:.2f} step/s, step p50 "
        f"{result['coarse']['step_wall']['p50_ms']:.1f} ms"
    )

    # ---- pass 2: isolated ------------------------------------------------
    if not args.skip_isolated:
        state = fresh_state(inputs, device)
        reset_rng()
        for rows in batches[: args.warmup_steps]:
            baseline_step(state, inputs, rows, device)
        _sync()
        ru0 = _rusage()
        t0 = time.perf_counter()
        iso = pass_isolated(state, inputs, timed, device)
        wall_iso = time.perf_counter() - t0
        ru1 = _rusage()
        phase_stats = {k: _stats(v) for k, v in iso["phases"].items()}
        accounted = sum(phase_stats[k]["sum_seconds"] for k in iso["phase_order"])
        step_wall = _stats(iso["wall"])
        result["isolated"] = {
            "n_steps": int(len(timed)),
            "wall_seconds": float(wall_iso),
            "step_wall": step_wall,
            "steps_per_second": float(len(timed) / wall_iso),
            "phases": phase_stats,
            "phase_share_of_accounted": {
                k: _share(phase_stats[k]["sum_seconds"], accounted) for k in iso["phase_order"]
            },
            "phase_share_of_step_wall": {
                k: _share(phase_stats[k]["sum_seconds"], step_wall["sum_seconds"]) for k in iso["phase_order"]
            },
            "accounted_seconds": float(accounted),
            "unaccounted_seconds": float(step_wall["sum_seconds"] - accounted),
            "unaccounted_note": (
                "wall not covered by an explicit phase: per-phase "
                "_sync() latency and the python loop itself"
            ),
            "rusage": _rusage_delta(ru0, ru1),
        }
        PF.log(
            "isolated: "
            + ", ".join(f"{k}={v['sum_seconds']:.2f}s" for k, v in phase_stats.items())
        )
        del iso

    # ---- pass 3: data pipeline micro-benchmarks --------------------------
    state = fresh_state(inputs, device)
    result["data_pipeline"] = {
        "collate_only": bench_dataloader(inputs, timed[:100], device, with_device=False),
        "collate_plus_transfer": bench_dataloader(inputs, timed[:100], device, with_device=True),
        "components": bench_collate_components(inputs, timed[:100]),
    }
    PF.log(
        f"collate {result['data_pipeline']['collate_only']['collate']['p50_ms']:.2f} ms/batch "
        f"(p90 {result['data_pipeline']['collate_only']['collate']['p90_ms']:.2f})"
    )

    # ---- pass 4: profiler window ----------------------------------------
    if not args.skip_profiler:
        state = fresh_state(inputs, device)
        result["profiler"] = pass_profiler(state, inputs, batches, device)
        PF.log(
            f"profiler: {result['profiler']['cuda_us_per_step']:.0f} us CUDA / step, "
            f"{result['profiler']['kernel_count_per_step']:.0f} kernels / step"
        )

    # ---- summary ---------------------------------------------------------
    result["summary"] = _summarise(result)
    result["finished_utc"] = PF.utcnow()
    PF.write_json(args.out, result)
    PF.log(f"wrote {PF.track_rel(args.out)}")
    print(json_dumps(result["summary"]))
    return 0


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True, default=PF._json_default)


def _summarise(result: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    coarse = result.get("coarse", {})
    if coarse:
        sw = coarse["step_wall"]
        out["total_step_wall_p50_ms"] = sw["p50_ms"]
        out["total_step_wall_p90_ms"] = sw["p90_ms"]
        out["steps_per_second"] = coarse["steps_per_second"]
        out["gpu_busy_fraction"] = coarse["gpu_busy_fraction"]
        out["gpu_event_spans"] = {
            k: {"total_ms": v["total_ms"], "share": _share(v["sum_seconds"], sw["sum_seconds"]),
                "p50_ms": v["p50_ms"]}
            for k, v in coarse["gpu_event_spans"].items()
        }
        out["cpu_enqueue"] = {
            k: {"total_ms": v["total_ms"], "share": _share(v["sum_seconds"], sw["sum_seconds"]),
                "p50_ms": v["p50_ms"]}
            for k, v in coarse["cpu_enqueue"].items()
        }
    iso = result.get("isolated")
    if iso:
        total = iso["step_wall"]["sum_seconds"]
        out["decomposition"] = {
            k: {
                "total_ms": v["total_ms"],
                "share": _share(v["sum_seconds"], total),
                "p50_ms": v["p50_ms"],
                "p90_ms": v["p90_ms"],
            }
            for k, v in iso["phases"].items()
        }
        out["decomposition"]["_unaccounted"] = {
            "total_ms": iso["unaccounted_seconds"] * 1e3,
            "share": _share(iso["unaccounted_seconds"], total),
        }
        out["largest_measured_phase"] = max(
            iso["phases"].items(), key=lambda kv: kv[1]["sum_seconds"]
        )[0]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
