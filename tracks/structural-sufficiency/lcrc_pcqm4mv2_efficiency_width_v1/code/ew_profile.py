"""Stage 1 -- measured bottleneck analysis of the *frozen* W2 training step.

Runs the historical training step (same collate, ``PinnedStager``, model,
optimizer, clip and loss) on the real 1M ``pcqm_train1m`` mmap cache for
``--steps`` measured steps after ``--warmup`` unmeasured ones, and records

* CPU wall time of every host phase (collate, staging issue, forward, loss,
  backward, clip, optimizer, explicit sync, total),
* CUDA-event device time of the same phases (so asynchronous execution is
  never summed as if it were host time),
* a ``torch.profiler`` trace of a short window (CPU ops + CUDA kernels),
* cold-cache vs hot-cache throughput (``posix_fadvise(DONTNEED)`` on the cache
  files before the cold pass; the hot pass replays the same rows),
* process CPU utilisation, RSS, GPU utilisation, peak GPU memory,
* step-time p50/p90 and real molecules/sec,
* the amortised per-epoch cost of full Validation + Train-monitor evaluation
  and checkpoint writes.

The instrumented step is a line-by-line copy of ``pq_train.train_step`` with
timers inserted; before measuring, it is checked on a cloned model/optimizer
against the frozen step (loss, grad norm and post-step parameters must agree).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# host / process probes
# ---------------------------------------------------------------------------


def _read_proc_stat(pid: str = "self") -> tuple[float, float]:
    path = Path("/proc/self/stat") if pid == "self" else Path(f"/proc/{pid}/stat")
    fields = path.read_text().split()
    ticks = os.sysconf("SC_CLK_TCK")
    return float(fields[13]) / ticks, float(fields[14]) / ticks


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemFree:", "MemAvailable:", "Cached:", "Dirty:")):
            key, rest = line.split(":", 1)
            out[key] = int(rest.strip().split()[0]) * 1024
    return out


class Sampler(threading.Thread):
    """1 Hz utilisation sampler (own CPU time + nvidia-smi GPU utilisation)."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.stop_flag = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self._u0: tuple[float, float] | None = None
        self._t0 = 0.0

    def run(self) -> None:
        self._u0 = _read_proc_stat()
        self._t0 = time.perf_counter()
        while not self.stop_flag.wait(self.interval):
            u, s = _read_proc_stat()
            now = time.perf_counter()
            cpu_seconds = (u + s) - (self._u0[0] + self._u0[1])
            wall = max(now - self._t0, 1e-9)
            sample: dict[str, Any] = {
                "t": round(now - self._t0, 3),
                "process_cpu_cores": round(cpu_seconds / wall, 3),
                "process_cpu_fraction_of_host": round(cpu_seconds / wall / max(os.cpu_count() or 1, 1), 5),
                "rss_mb": round(C.peak_rss_mb(), 1),
            }
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                ).strip()
                util: dict[str, Any] = {}
                for line in raw.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        util[parts[0]] = {
                            "gpu_util_pct": float(parts[1]),
                            "mem_util_pct": float(parts[2]),
                            "mem_used_mib": float(parts[3]),
                        }
                sample["nvidia_smi"] = util
            except Exception:  # noqa: BLE001 - probe only
                sample["nvidia_smi"] = None
            self.samples.append(sample)
            self._u0, self._t0 = (u, s), now
            self._refresh_gpu_peak()

    def _refresh_gpu_peak(self) -> None:
        self.samples[-1]["gpu_peak_allocated_mb"] = round(
            float(torch.cuda.max_memory_allocated()) / 2**20, 1
        ) if torch.cuda.is_available() else None

    def finish(self) -> dict[str, Any]:
        self.stop_flag.set()
        self.join(timeout=10)
        gpus: dict[str, list[float]] = {}
        for s in self.samples:
            for idx, payload in (s.get("nvidia_smi") or {}).items():
                gpus.setdefault(idx, []).append(float(payload["gpu_util_pct"]))
        cores = [float(s["process_cpu_cores"]) for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "process_cpu_cores_mean": float(np.mean(cores)) if cores else None,
            "process_cpu_cores_max": float(np.max(cores)) if cores else None,
            "gpu_util_pct_visible": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in gpus.items()},
            "samples": self.samples[-20:],
        }


def cache_residency(path: Path, sample_pages: int = 4096) -> dict[str, Any]:
    """Fraction of sampled 4 KiB pages of ``path`` resident in the page cache.

    Uses ``mincore(2)`` on a private read-only mapping; it does not change the
    file or the page cache.  Only a regular stride sample is checked, so the
    number is an estimate of residency, not an exact count.
    """
    import ctypes
    import mmap

    size = int(Path(path).stat().st_size)
    if size == 0:
        return {"page_fraction_resident": None, "pages_sampled": 0}
    page = mmap.PAGESIZE
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_COPY)
        try:
            n_pages = (size + page - 1) // page
            n_sample = min(int(sample_pages), n_pages)
            offsets = np.linspace(0, max(n_pages - 1, 0), num=n_sample, dtype=np.int64) * page
            vec = (ctypes.c_ubyte * n_sample)()
            base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            # mincore needs a page-aligned start; instead of an arbitrary stride
            # we map each sampled page's address individually via ctypes.
            resident = 0
            for off in offsets.tolist():
                addr = ctypes.c_void_p(base + off)
                one = (ctypes.c_ubyte * 1)()
                rc = libc.mincore(addr, ctypes.c_size_t(page), ctypes.byref(one))
                if rc == 0 and (one[0] & 1):
                    resident += 1
            return {
                "page_fraction_resident": float(resident / n_sample),
                "pages_sampled": int(n_sample),
                "bytes": size,
            }
        finally:
            mm.close()


def drop_page_cache(paths: Sequence[Path]) -> dict[str, Any]:
    """Evict the given files from the page cache (reclaimable, data untouched)."""
    before = _meminfo()
    evicted = 0
    for path in paths:
        if not Path(path).is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            evicted += 1
        finally:
            os.close(fd)
    time.sleep(0.2)
    after = _meminfo()
    sample_path = Path(paths[0]) if paths else None
    residence = None
    if sample_path is not None and sample_path.is_file():
        try:
            residence = cache_residency(sample_path)
        except Exception as exc:  # noqa: BLE001 - probe only
            residence = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "files": evicted,
        "cached_before": before.get("Cached"),
        "cached_after": after.get("Cached"),
        "cached_delta": int(before.get("Cached", 0)) - int(after.get("Cached", 0)),
        "residency_after_eviction_first_file": residence,
    }


# ---------------------------------------------------------------------------
# instrumented copy of the frozen step
# ---------------------------------------------------------------------------


class Events:
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"
        names = ("pre_stage", "post_stage", "forward", "loss", "backward", "clip", "step")
        self.events = {n: torch.cuda.Event(enable_timing=True) for n in names} if self.enabled else {}

    def record(self, name: str) -> None:
        if self.enabled:
            self.events[name].record()

    def gpu_ms(self, first: str, second: str) -> float:
        if not self.enabled:
            return float("nan")
        return float(self.events[first].elapsed_time(self.events[second]))


def instrumented_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y_z: torch.Tensor,
    *,
    global_step: int,
    steps_epoch: int,
    steps_total: int,
    clip: float,
    ev: Events,
    host: dict[str, float],
) -> dict[str, Any]:
    """``pq_train.train_step`` with timers.  Same ops, same order, same math."""
    t = time.perf_counter()
    lr = M.set_lrs(optimizer, step=int(global_step), steps_epoch=int(steps_epoch), steps_total=int(steps_total))
    host["lr"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    ev.record("forward")
    pred = M.predict_z(model, batch)
    host["forward_submit"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    n = int(y_z.shape[0])
    if tuple(pred.shape) != (n,):
        raise RuntimeError(f"model output {tuple(pred.shape)} != {(n,)}")
    finite_pred = bool(torch.isfinite(pred).all())
    if not finite_pred:
        raise RuntimeError(f"non-finite prediction at step {global_step}")
    ev.record("loss")
    loss = torch.mean(torch.abs(pred - y_z))
    finite_loss = bool(torch.isfinite(loss))
    if not finite_loss:
        raise RuntimeError(f"non-finite loss at step {global_step}: {float(loss)}")
    host["loss_submit"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    host["backward_submit"] = (time.perf_counter() - t) * 1e3
    ev.record("backward")

    t = time.perf_counter()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    host["clip_host_with_sync"] = (time.perf_counter() - t) * 1e3
    ev.record("clip")
    if not np.isfinite(gnorm):
        raise RuntimeError(f"non-finite gradient norm at step {global_step}")

    t = time.perf_counter()
    optimizer.step()
    host["optimizer_submit"] = (time.perf_counter() - t) * 1e3
    ev.record("step")
    return {"loss": float(loss.detach()), "grad_norm": gnorm, "lr": lr, "clipped": bool(gnorm > clip), "n": n}


def _clone_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        {
            "state": {
                k: {kk: (vv.detach().clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in optimizer.state_dict()["state"].items()
            },
            "param_groups": [dict(g) for g in optimizer.state_dict()["param_groups"]],
        },
    )


def step_equivalence_gate(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    y_z: torch.Tensor,
    *,
    clip: float,
    steps_epoch: int,
    steps_total: int,
    ev: Events,
) -> dict[str, Any]:
    """Instrumented step vs the frozen step from the same state, bitwise."""
    state0, opt0 = _clone_state(model, optimizer)
    frozen = C.PQ_TRAIN.train_step(
        model,
        optimizer,
        batch,
        y_z,
        global_step=0,
        steps_epoch=steps_epoch,
        steps_total=steps_total,
        clip=clip,
    )
    state_frozen, opt_frozen = _clone_state(model, optimizer)

    model.load_state_dict(state0, strict=True)
    optimizer.load_state_dict(opt0)
    host: dict[str, float] = {}
    ours = instrumented_step(
        model,
        optimizer,
        batch,
        y_z,
        global_step=0,
        steps_epoch=steps_epoch,
        steps_total=steps_total,
        clip=clip,
        ev=ev,
        host=host,
    )
    state_ours, opt_ours = _clone_state(model, optimizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    param_max = max(
        float((state_frozen[k].to(torch.float64) - state_ours[k].to(torch.float64)).abs().max())
        for k in state_frozen
        if state_frozen[k].is_floating_point()
    )
    same_params = all(bool(torch.equal(state_frozen[k], state_ours[k])) for k in state_frozen)
    same_opt = all(
        bool(torch.equal(opt_frozen["state"][k][kk], opt_ours["state"][k][kk]))
        for k in opt_frozen["state"]
        for kk in opt_frozen["state"][k]
        if torch.is_tensor(opt_frozen["state"][k][kk])
    )
    return {
        "loss_equal": bool(frozen["loss"] == ours["loss"]),
        "grad_norm_equal": bool(frozen["grad_norm"] == ours["grad_norm"]),
        "loss_frozen": float(frozen["loss"]),
        "loss_instrumented": float(ours["loss"]),
        "grad_norm_frozen": float(frozen["grad_norm"]),
        "grad_norm_instrumented": float(ours["grad_norm"]),
        "parameters_bitwise_equal": bool(same_params),
        "optimizer_state_bitwise_equal": bool(same_opt),
        "parameter_max_abs_delta": param_max,
        "all_pass": bool(
            frozen["loss"] == ours["loss"]
            and frozen["grad_norm"] == ours["grad_norm"]
            and same_params
            and same_opt
            and param_max == 0.0
        ),
    }


# ---------------------------------------------------------------------------
# measured passes
# ---------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
        "sum": float(a.sum()),
    }


def measured_pass(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rows: np.ndarray,
    steps: int,
    batch_size: int,
    clip: float,
    sampler: Sampler | None,
) -> dict[str, Any]:
    """One pass of ``steps`` frozen batches in the given row order."""
    cache = task["train_cache"]
    y_z_all = task["train_y_z"]
    stager = PinnedStager(device, depth=2)
    ev = Events(device)
    host_names = [
        "collate",
        "stage_issue",
        "forward_submit",
        "loss_submit",
        "backward_submit",
        "clip_host_with_sync",
        "optimizer_submit",
        "sync",
        "total",
        "lr",
    ]
    rec: dict[str, list[float]] = {k: [] for k in host_names}
    inner_names = ["lr", "forward_submit", "loss_submit", "backward_submit", "clip_host_with_sync", "optimizer_submit"]
    inner: dict[str, list[float]] = {k: [] for k in inner_names}
    gpu_names = [
        ("h2d", "pre_stage", "post_stage"),
        ("forward", "post_stage", "forward"),
        ("loss", "forward", "loss"),
        ("backward", "loss", "backward"),
        ("clip", "backward", "clip"),
        ("optimizer", "clip", "step"),
        ("device_step_span", "post_stage", "step"),
        ("device_step_span_incl_h2d", "pre_stage", "step"),
    ]
    gpu: dict[str, list[float]] = {name: [] for name, _a, _b in gpu_names}
    molecules = 0
    t_pass = time.perf_counter()
    for i in range(int(steps)):
        rows_b = rows[i * batch_size : (i + 1) * batch_size]
        t = time.perf_counter()
        raw = C.PQ_DATA.collate(cache, rows_b.tolist())
        rec["collate"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        ev.record("pre_stage")
        batch = stager.batch(raw)
        y = stager.labels(y_z_all[rows_b])
        rec["stage_issue"].append((time.perf_counter() - t) * 1e3)
        ev.record("post_stage")

        t0 = time.perf_counter()
        host_step: dict[str, float] = {}
        out = instrumented_step(
            model,
            optimizer,
            batch,
            y,
            global_step=i,
            steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]),
            clip=clip,
            ev=ev,
            host=host_step,
        )
        for k in inner_names:
            inner[k].append(float(host_step.get(k, 0.0)))
        del out
        t_sync = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec["sync"].append((time.perf_counter() - t_sync) * 1e3)
        rec["total"].append((time.perf_counter() - t0) * 1e3)
        molecules += int(rows_b.size)
        if ev.enabled:
            for name, a, b in gpu_names:
                gpu[name].append(ev.gpu_ms(a, b))
    wall = float(time.perf_counter() - t_pass)
    stager.release()
    result = {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "molecules": int(molecules),
        "wall_seconds": wall,
        "molecules_per_second": float(molecules / wall) if wall > 0 else None,
        "steps_per_second": float(int(steps) / wall) if wall > 0 else None,
        "step_ms": _stats(rec["total"]),
        "host_ms": {k: _stats(v) for k, v in rec.items() if k != "total"},
        "host_submitted_ms": {k: _stats(v) for k, v in inner.items()},
        "device_span_ms": {k: _stats(v) for k, v in gpu.items()},
        "stage_counters": dict(stager.counters),
        "peak_rss_mb": C.peak_rss_mb(),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    if sampler is not None:
        pass  # sampler aggregates across the whole profile run
    return result


def profiler_window(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rows: np.ndarray,
    steps: int,
    batch_size: int,
    clip: float,
) -> dict[str, Any]:
    """torch.profiler over a short window: top CPU ops and CUDA kernels."""
    cache = task["train_cache"]
    y_z_all = task["train_y_z"]
    stager = PinnedStager(device, depth=2)
    ev = Events(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    out: dict[str, Any] = {"steps": int(steps), "activities": [str(a) for a in activities]}
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for i in range(int(steps)):
            rows_b = rows[i * batch_size : (i + 1) * batch_size]
            with torch.profiler.record_function("collate"):
                raw = C.PQ_DATA.collate(cache, rows_b.tolist())
            with torch.profiler.record_function("stage"):
                batch = stager.batch(raw)
                y = stager.labels(y_z_all[rows_b])
            with torch.profiler.record_function("train_step"):
                instrumented_step(
                    model,
                    optimizer,
                    batch,
                    y,
                    global_step=i,
                    steps_epoch=int(task["steps_epoch"]),
                    steps_total=int(task["steps_total"]),
                    clip=clip,
                    ev=ev,
                    host={},
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    stager.release()
    key = prof.key_averages()
    out["events_total_cpu_ms"] = float(sum(e.self_cpu_time_total for e in key) / 1e3)
    out["cuda_time_total_ms"] = float(sum(e.self_device_time_total for e in key) / 1e3) if device.type == "cuda" else 0.0

    def _row(e: Any) -> dict[str, Any]:
        return {
            "key": e.key,
            "calls": int(e.count),
            "self_cpu_ms": round(float(e.self_cpu_time_total) / 1e3, 4),
            "cpu_total_ms": round(float(e.cpu_time_total) / 1e3, 4),
            "cuda_ms": round(float(getattr(e, "self_device_time_total", 0.0)) / 1e3, 4),
        }

    rows_sorted = sorted(key, key=lambda e: e.self_cpu_time_total, reverse=True)
    out["top_cpu"] = [_row(e) for e in rows_sorted[:25]]
    if device.type == "cuda":
        rows_cuda = sorted(key, key=lambda e: getattr(e, "self_device_time_total", 0.0), reverse=True)
        out["top_cuda"] = [_row(e) for e in rows_cuda[:25]]
    # aggregate by operator family for the report
    families: dict[str, float] = {}
    for e in key:
        name = e.key.split("::")[0] if "::" in e.key else e.key.split("(")[0]
        families[name] = families.get(name, 0.0) + float(e.self_cpu_time_total) / 1e3
    out["cpu_families_top"] = [
        {"family": k, "self_cpu_ms": round(v, 4)}
        for k, v in sorted(families.items(), key=lambda kv: kv[1], reverse=True)[:20]
    ]
    return out


def eval_and_checkpoint_overhead(
    task: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    out_dir: Path,
) -> dict[str, Any]:
    """Full Validation + monitor evaluation + checkpoint write, as measured."""
    C.log("[profile] full Validation pass ...")
    t = time.perf_counter()
    valid_report, _ = C.PQ_TRAIN.evaluate_split(model, task, device, which="valid", batch_size=512)
    valid_s = float(time.perf_counter() - t)
    C.log(f"[profile] Validation {valid_s:.2f}s MAE {valid_report['mae_eV']:.6f} eV")
    t = time.perf_counter()
    monitor_report, _ = C.PQ_TRAIN.evaluate_split(model, task, device, which="monitor", batch_size=512)
    monitor_s = float(time.perf_counter() - t)
    t = time.perf_counter()
    payload = C.PQ_TRAIN.checkpoint_payload(
        task=task,
        run="PROFILE_W2",
        epoch=0,
        global_step=0,
        model=model,
        optimizer=optimizer,
        rng_state=C.capture_rng_state(),
        model_audit={"total_parameters": int(sum(p.numel() for p in model.parameters()))},
        valid_report=valid_report,
        kind="profile",
    )
    path = out_dir / "_profile_checkpoint.pt"
    C.PQ_TRAIN._atomic_save(payload, path)
    ckpt_s = float(time.perf_counter() - t)
    size = int(path.stat().st_size)
    path.unlink(missing_ok=True)
    per_epoch = valid_s + monitor_s + ckpt_s
    return {
        "valid_seconds": valid_s,
        "valid_mae_eV": float(valid_report["mae_eV"]),
        "valid_n": int(valid_report["n"]),
        "monitor_seconds": monitor_s,
        "monitor_mae_eV": float(monitor_report["mae_eV"]),
        "checkpoint_seconds": ckpt_s,
        "checkpoint_bytes": size,
        "per_epoch_seconds": per_epoch,
        "steps_per_epoch": int(task["steps_epoch"]),
        "amortised_ms_per_step": 1e3 * per_epoch / int(task["steps_epoch"]),
        "amortised_fraction_of_step_ms": None,  # filled by the caller
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--profiler-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "profile.json"))
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if int(args.batch_size) != C.BATCH_SIZE:
        C.log(f"[profile] NOTE batch_size {args.batch_size} != formal 128 (throughput probe only)")

    t_start = time.perf_counter()
    deterministic = C.apply_deterministic_settings()
    gpu0 = C.gpu_occupancy()
    task = DATA.load_task("PCQM_W3_1M", verify_hashes=True)
    task["run"] = "PROFILE_W2"
    task["scale"] = 2
    task["model"] = "W2"
    task["model_name"] = C.MODEL_NAMES[2]
    C.seed_everything(0)
    model, audit = M.build_model(task["spec"], task["norm"], scale=2, seed=0, device=device)
    optimizer, opt_audit = M.make_optimizer(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    eq = M.historical_w2_equivalence(task["spec"], task["norm"], seed=0)
    C.log(f"[profile] W2 builder equivalence: bitwise={eq['bitwise_identical']}")

    # one batch for the instrumented-vs-frozen step gate
    rng = np.random.default_rng(int(args.seed))
    gate_rows = np.sort(rng.choice(C.TRAIN_ROWS, size=C.BATCH_SIZE, replace=False))
    gate_batch = C.PQ_DATA.collate(task["train_cache"], gate_rows.tolist())
    stager = PinnedStager(device, depth=1)
    gate_batch_gpu = stager.batch(gate_batch)
    gate_y = stager.labels(task["train_y_z"][gate_rows])
    gate = step_equivalence_gate(
        model,
        optimizer,
        gate_batch_gpu,
        gate_y,
        clip=float(C.RECIPE["gradient_clip"]),
        steps_epoch=int(task["steps_epoch"]),
        steps_total=int(task["steps_total"]),
        ev=Events(device),
    )
    stager.release()
    C.log(f"[profile] instrumented-step gate: {gate['all_pass']} (loss {gate['loss_instrumented']:.8f})")
    if not gate["all_pass"]:
        raise SystemExit("instrumented step is not equivalent to the frozen step")

    # warm kernels on the *Validation* cache so the cold Train pass is not
    # polluted by first-touch CUDA compilation
    valid_rows = np.arange(int(task["valid_cache"]["n_molecules"]), dtype=np.int64)
    warm_order = np.random.default_rng(int(args.seed) + 1).permutation(valid_rows.size)
    warm_rows = valid_rows[warm_order][: int(args.warmup) * C.BATCH_SIZE]
    C.log(f"[profile] {args.warmup} kernel-warmup steps on the Validation cache ...")
    sampler = Sampler(interval=1.0)
    sampler.start()
    measured_pass(
        task,
        model,
        optimizer,
        device,
        rows=warm_rows,
        steps=int(args.warmup),
        batch_size=C.BATCH_SIZE,
        clip=float(C.RECIPE["gradient_clip"]),
        sampler=None,
    )

    train_files = sorted(p for p in (C.OLD_CACHE / "pcqm_train1m").glob("*") if p.is_file())
    order_cold = np.random.default_rng(int(args.seed) + 2).permutation(C.TRAIN_ROWS)
    cold_rows = order_cold[: int(args.steps) * C.BATCH_SIZE]
    evict = {"skipped": True}
    cold = None
    if not args.skip_cold:
        try:
            pre_residency = cache_residency(train_files[0]) if train_files else None
        except Exception as exc:  # noqa: BLE001
            pre_residency = {"error": f"{type(exc).__name__}: {exc}"}
        evict = drop_page_cache(train_files)
        evict["residency_before_eviction_first_file"] = pre_residency
        C.log(
            f"[profile] cold pass ({args.steps} steps): evicted {evict['files']} files, "
            f"page-cache delta {evict['cached_delta'] / 2**30:.2f} GiB"
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cold = measured_pass(
            task,
            model,
            optimizer,
            device,
            rows=cold_rows,
            steps=int(args.steps),
            batch_size=C.BATCH_SIZE,
            clip=float(C.RECIPE["gradient_clip"]),
            sampler=sampler,
        )
        C.log(
            f"[profile] cold: {cold['step_ms']['mean']:.2f} ms/step, "
            f"{cold['molecules_per_second']:.1f} mol/s"
        )

    # hot pass: exact same rows, now in page cache
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    C.log(f"[profile] hot pass ({args.steps} steps, same rows) ...")
    hot = measured_pass(
        task,
        model,
        optimizer,
        device,
        rows=cold_rows,
        steps=int(args.steps),
        batch_size=C.BATCH_SIZE,
        clip=float(C.RECIPE["gradient_clip"]),
        sampler=sampler,
    )
    C.log(
        f"[profile] hot: {hot['step_ms']['mean']:.2f} ms/step, {hot['molecules_per_second']:.1f} mol/s"
    )

    C.log(f"[profile] torch.profiler window ({args.profiler_steps} steps) ...")
    prof = profiler_window(
        task,
        model,
        optimizer,
        device,
        rows=cold_rows[: int(args.profiler_steps) * C.BATCH_SIZE],
        steps=int(args.profiler_steps),
        batch_size=C.BATCH_SIZE,
        clip=float(C.RECIPE["gradient_clip"]),
    )

    overhead = eval_and_checkpoint_overhead(task, model, optimizer, device, out_dir=Path(args.out).parent)
    if hot and hot["step_ms"].get("mean"):
        overhead["amortised_fraction_of_step_ms"] = float(
            overhead["amortised_ms_per_step"] / hot["step_ms"]["mean"]
        )
    sampler_payload = sampler.finish()

    host = C.host_facts()
    mem = _meminfo()
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_PROFILE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "device": str(device),
        "host": host,
        "cpu_count": os.cpu_count(),
        "meminfo_bytes": mem,
        "deterministic_settings": deterministic,
        "gpu_occupancy_start": gpu0,
        "torch_version": torch.__version__,
        "model": "LCRC-W2 (frozen historical configuration)",
        "parameters": int(audit["total_parameters"]),
        "widths": audit["widths"],
        "builder_equivalence_to_frozen_w2": eq,
        "instrumented_step_gate": gate,
        "task_load_seconds": task["load_seconds"],
        "cache": {
            "train_rows": int(task["train_cache"]["n_molecules"]),
            "valid_rows": int(task["valid_cache"]["n_molecules"]),
            "train_files": len(train_files),
            "train_bytes": int(sum(p.stat().st_size for p in train_files)),
        },
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup),
        "batch_size": int(C.BATCH_SIZE),
        "cache_eviction": evict,
        "cold": cold,
        "hot": hot,
        "profiler": prof,
        "epoch_overhead": overhead,
        "utilisation": sampler_payload,
        "wall_seconds": float(time.perf_counter() - t_start),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_rss_mb": C.peak_rss_mb(),
        "cpu_seconds": C.cpu_seconds(),
        "loss_and_clip": {"gradient_clip": float(C.RECIPE["gradient_clip"])},
        "notes": [
            "host_ms values are host wall time of that phase; where the frozen step forces a "
            "synchronisation the wait is included in the phase it happens in",
            "gpu_ms values are CUDA-event deltas on the default stream (never summed with host time)",
            "molecules/sec uses the measured pass wall time, including synchronisation",
        ],
    }
    payload = C.json_safe(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, payload)
    C.log(f"[profile] wrote {C.track_rel(out)} ({payload['wall_seconds']:.1f}s wall)")
    summary = {
        "cold_step_ms_mean": None if cold is None else cold["step_ms"]["mean"],
        "hot_step_ms_mean": hot["step_ms"]["mean"],
        "hot_step_ms_p50": hot["step_ms"]["p50"],
        "hot_step_ms_p90": hot["step_ms"]["p90"],
        "hot_molecules_per_second": hot["molecules_per_second"],
        "device_step_span_ms_mean": hot["device_span_ms"]["device_step_span"]["mean"],
        "device_step_span_incl_h2d_ms_mean": hot["device_span_ms"]["device_step_span_incl_h2d"]["mean"],
        "amortised_ms_per_step": overhead["amortised_ms_per_step"],
    }
    C.write_json(out.with_name("profile_summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
