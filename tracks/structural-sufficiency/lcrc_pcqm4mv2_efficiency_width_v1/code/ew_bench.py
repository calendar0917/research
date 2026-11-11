"""Stage 2 -- measured A/B of the throughput implementations.

Each configuration starts from the same seed-0 W2 state, consumes the same
row order and takes the same number of steps, so the final model state must be
identical across configurations -- that is asserted as a hash comparison in the
same run.  Throughput is end-to-end wall time (data assembly + host staging +
GPU transfer + forward + backward + clip + update + all synchronisation).

Configurations
--------------
``frozen``  historical code path (frozen collate, per-step sync, no prefetch)
``E1``      vectorised collate only
``E2``      block-deferred host synchronisation only
``E3``      bounded producer prefetch only
``E1E2``    E1 + E2
``optimized`` / ``E1E2E3``  all three
``frozen_repeat`` same as frozen, run last, to expose drift/variance

Batch sizes 256 and 512 are probed **for throughput investigation only**; the
formal capacity runs stay at batch 128 and no probe result enters a training
recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
import ew_train as T  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

import torch  # noqa: E402


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
    }


def run_config(
    task: Mapping[str, Any],
    *,
    implementation: str,
    steps: int,
    warmup: int,
    order: np.ndarray,
    batch_size: int,
    device: torch.device,
    scale: int = 2,
) -> dict[str, Any]:
    impl = dict(T.IMPLEMENTATIONS[implementation])
    C.seed_everything(int(task["seed"]))
    model, audit = M.build_model(task["spec"], task["norm"], scale=int(scale), seed=int(task["seed"]), device=device)
    optimizer, _opt_audit = M.make_optimizer(model)
    collate_fn = lambda cache, rows: EC.collate_for(cache, rows, implementation=str(impl["collate"]))  # noqa: E731
    stager = PinnedStager(device, depth=2)
    prefetcher = (
        PIPE.Prefetcher(
            task["train_cache"],
            task["train_y_z"],
            collator=collate_fn,
            batch_size=int(batch_size),
            depth=int(impl["prefetch"]),
        )
        if int(impl["prefetch"]) > 0
        else None
    )
    clip = float(C.RECIPE["gradient_clip"])
    check_interval = int(impl["check_interval"])
    y_z_all = task["train_y_z"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_batches = int(warmup) + int(steps)
    if int(order.size) < total_batches * int(batch_size):
        raise ValueError("row order is shorter than warmup+measured steps")
    rows_warm = order[: int(warmup) * int(batch_size)]
    rows_meas = order[int(warmup) * int(batch_size) : total_batches * int(batch_size)]

    def _window(window_rows: np.ndarray, *, start_step: int) -> list[float]:
        nonlocal global_step
        deferred = T.DeferredLog(check_interval=check_interval, clip=clip) if bool(impl["defer"]) else None
        block_start = time.perf_counter()
        first_step = int(start_step)
        times: list[float] = []
        iterator = (
            prefetcher.epoch(window_rows)
            if prefetcher is not None
            else (
                (window_rows[i : i + batch_size], None, None)
                for i in range(0, window_rows.size, batch_size)
            )
        )
        for rows, raw, y_pref in iterator:
            if raw is None:
                raw = collate_fn(task["train_cache"], rows.tolist())
                y_np = np.ascontiguousarray(y_z_all[rows], dtype=np.float32)
            else:
                y_np = y_pref
            batch = stager.batch(raw)
            y = stager.labels(y_np)
            t0 = time.perf_counter()
            if deferred is None:
                out = C.PQ_TRAIN.train_step(
                    model,
                    optimizer,
                    batch,
                    y,
                    global_step=global_step,
                    steps_epoch=int(task["steps_epoch"]),
                    steps_total=int(task["steps_total"]),
                    clip=clip,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(float(time.perf_counter() - t0))
                losses.append(float(out["loss"]))
                grad_norms.append(float(out["grad_norm"]))
            else:
                out = T.train_step_deferred(
                    model,
                    optimizer,
                    batch,
                    y,
                    global_step=global_step,
                    steps_epoch=int(task["steps_epoch"]),
                    steps_total=int(task["steps_total"]),
                    clip=clip,
                )
                deferred.add(out)
                if deferred.ready():
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    deferred.flush(block_seconds=float(time.perf_counter() - block_start), first_step=first_step)
                    block_start = time.perf_counter()
            global_step += 1
        if deferred is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if deferred.pending():
                deferred.flush(block_seconds=float(time.perf_counter() - block_start), first_step=first_step)
            times.extend(deferred.block_step_ms)
            losses.extend(deferred.losses)
            grad_norms.extend(deferred.grad_norms)
        return times

    global_step = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    cpu0 = C.cpu_seconds()
    t_warm = time.perf_counter()
    _window(rows_warm, start_step=0)
    warmup_wall = float(time.perf_counter() - t_warm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cpu_warm = float(C.cpu_seconds() - cpu0)
    # measured window
    cpu1 = C.cpu_seconds()
    t_meas = time.perf_counter()
    step_times = _window(rows_meas, start_step=int(warmup))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_wall = float(time.perf_counter() - t_meas)
    cpu_measured = float(C.cpu_seconds() - cpu1)
    if prefetcher is not None:
        prefetcher.close()
    stager.release()

    measured_steps = max(len(step_times), 1)
    molecules = int(rows_meas.size)
    result = {
        "implementation": implementation,
        "impl": impl,
        "batch_size": int(batch_size),
        "scale": int(scale),
        "parameters": int(audit["total_parameters"]),
        "warmup_steps": int(warmup),
        "measured_steps": int(len(step_times)),
        "warmup_wall_seconds": warmup_wall,
        "wall_seconds_all": warmup_wall + measured_wall,
        "wall_seconds_measured": measured_wall,
        "steps_per_second": float(len(step_times) / measured_wall) if measured_wall > 0 else None,
        "molecules_per_second": float(molecules / measured_wall) if measured_wall > 0 else None,
        "step_ms": _stats(step_times),
        "process_cpu_seconds_all": cpu_warm + cpu_measured,
        "process_cpu_seconds_measured": cpu_measured,
        "process_cpu_seconds_per_measured_step": float(cpu_measured / max(len(step_times), 1)),
        "final_loss": float(losses[-1]) if losses else None,
        "final_grad_norm": float(grad_norms[-1]) if grad_norms else None,
        "loss_first_last_sha256": C.sha256_bytes(
            (f"{losses[0]:.9g}|{losses[-1]:.9g}" if losses else "empty").encode()
        ),
        "state_fingerprint": C.PQ.BS.state_fingerprint(model.state_dict()),
        "weights_sha256": C.sha256_bytes(
            b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())
        ),
        "optimizer_fingerprint": C.sha256_bytes(
            b"".join(
                (v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
                for st in optimizer.state_dict()["state"].values()
                for v in st.values()
            )
        ),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_rss_mb": C.peak_rss_mb(),
        "producer_counters": dict(prefetcher.counters) if prefetcher is not None else None,
        "step_timing_mode": "block-deferred" if bool(impl["defer"]) else "per-step",
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "bench.json"))
    parser.add_argument("--probe-batch-sizes", default="256,512")
    parser.add_argument("--probe-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    C.apply_deterministic_settings()
    task = DATA.load_task("PCQM_W3_1M", verify_hashes=True)
    n_batches_needed = int(args.warmup) + int(args.steps)
    order = np.random.default_rng(int(args.seed)).permutation(C.TRAIN_ROWS)[: n_batches_needed * C.BATCH_SIZE]
    configs = ["frozen", "E1", "E2", "E3", "E1E2", "E1E2E3", "frozen_repeat"]
    results: dict[str, Any] = {}
    order_impl = {c: ("frozen" if c == "frozen_repeat" else c) for c in configs}
    t0 = time.perf_counter()
    for name in configs:
        C.log(f"[bench] {name} ({args.steps} steps, batch {C.BATCH_SIZE}) ...")
        res = run_config(
            task,
            implementation=order_impl[name],
            steps=int(args.steps),
            warmup=int(args.warmup),
            order=order,
            batch_size=C.BATCH_SIZE,
            device=device,
            scale=int(args.scale),
        )
        res["label"] = name
        results[name] = res
        C.log(
            f"[bench] {name}: {res['step_ms']['mean']:.2f} ms/step, "
            f"{res['molecules_per_second']:.1f} mol/s, final loss {res['final_loss']:.8f}"
        )

    baseline = float(results["frozen"]["step_ms"]["mean"])
    baseline2 = float(results["frozen_repeat"]["step_ms"]["mean"])
    for name, res in results.items():
        res["speedup_vs_frozen_step_ms"] = baseline / float(res["step_ms"]["mean"])
        res["speedup_vs_frozen_end_to_end"] = (
            float(results["frozen"]["molecules_per_second"]) / float(res["molecules_per_second"])
            if res["molecules_per_second"]
            else None
        )
    fingerprints = {name: res["weights_sha256"] for name, res in results.items() if name != "frozen_repeat"}
    fingerprints["frozen_repeat"] = results["frozen_repeat"]["weights_sha256"]
    equivalence = {
        "all_weights_identical": len(set(fingerprints.values())) == 1,
        "fingerprints": fingerprints,
        "frozen_vs_frozen_repeat_gain": baseline / baseline2 if baseline2 else None,
    }
    if not equivalence["all_weights_identical"]:
        C.log("[bench] WARNING: implementations did not reach the same weights")

    probes: dict[str, Any] = {}
    for bs in [int(x) for x in str(args.probe_batch_sizes).split(",") if x.strip()]:
        if bs == C.BATCH_SIZE:
            continue
        name = f"optimized_bs{bs}"
        C.log(f"[bench] probe {name} ({args.probe_steps} steps) -- throughput only, never a formal recipe")
        probe_order = np.random.default_rng(int(args.seed) + bs).permutation(C.TRAIN_ROWS)[
            : (int(args.probe_steps) + 20) * bs
        ]
        res = run_config(
            task,
            implementation="E1E2E3",
            steps=int(args.probe_steps),
            warmup=20,
            order=probe_order,
            batch_size=bs,
            device=device,
            scale=int(args.scale),
        )
        res["tail_equivalent_across_batches"] = None
        probes[name] = res
        C.log(f"[bench] {name}: {res['step_ms']['mean']:.2f} ms/step, {res['molecules_per_second']:.1f} mol/s")

    best = max(
        (name for name in ("E1", "E2", "E3", "E1E2", "E1E2E3") if name in results),
        key=lambda n: float(results[n]["molecules_per_second"] or 0.0),
    )
    payload = {
        "schema": C.SCHEMA_BENCH,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "device": str(device),
        "host_facts": C.host_facts(),
        "gpu_occupancy": C.gpu_occupancy(),
        "scale": int(args.scale),
        "batch_size": int(C.BATCH_SIZE),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup),
        "configs": results,
        "equivalence": equivalence,
        "batch_size_probes": probes,
        "frozen_step_ms_mean": baseline,
        "frozen_repeat_step_ms_mean": baseline2,
        "speedup_best_config": baseline / float(results[best]["step_ms"]["mean"]),
        "best_config": best,
        "all_configs_same_weights": bool(equivalence["all_weights_identical"]),
        "wall_seconds": float(time.perf_counter() - t0),
        "notes": [
            "frozen_repeat is the same code as frozen, run last, to expose drift",
            "every non-probe configuration must reach the same weights fingerprint",
            "batch-256/512 probes are throughput investigations only; formal runs stay at 128",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, C.json_safe(payload))
    print(json.dumps({k: v for k, v in payload.items() if k in ("best_config", "speedup_best_config", "all_configs_same_weights", "frozen_step_ms_mean", "frozen_repeat_step_ms_mean")}, indent=1))
    C.log(f"[bench] wrote {C.track_rel(out)} wall {payload['wall_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
