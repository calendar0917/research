"""Stage 2 -- interleaved, repeated A/B of the throughput implementations.

Single-pass timing on this shared host drifts (a pilot run saw the *same*
frozen configuration measure 36.4 ms and then 44.1 ms in one session because of
external load / clocks).  This benchmark therefore

1. pre-touches every row it will use, so no configuration pays first-touch
   page-fault cost,
2. runs each configuration in several short rounds with the round order
   alternating (forward, reverse, forward), and
3. reports the **median** of the per-round end-to-end throughput, plus the
   min/max spread.

Every configuration starts from the same seed-0 W2 state and consumes the same
rows in the same order, so the final weights must be **identical** across all
configurations and rounds; the hash comparison is part of the result.

Configurations (E1/E2/E3 are measured alone and combined):

``frozen`` historical path · ``E1`` vectorised collate · ``E2`` block-deferred
host sync · ``E3`` producer prefetch · ``E1E2``, ``E1E2c16``, ``E1E2c8``,
``E1E3``, ``E1E2E3c16`` · ``frozen_t1`` / ``E1_t1`` use one ATen CPU thread
(diagnostic: the step is kernel-launch-bound and the process burns many cores).

Batch sizes 256/512 are probed **for throughput investigation only**; the
formal capacity runs stay at batch 128.
"""

from __future__ import annotations

import argparse
import json
import statistics
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

#: label -> (implementation name, overrides)
BENCH_CONFIGS: dict[str, dict[str, Any]] = {
    "frozen": {"impl": "frozen"},
    "E1": {"impl": "E1"},
    "E2": {"impl": "E2"},
    "E3": {"impl": "E3"},
    "E1E2": {"impl": "E1E2"},
    "E1E2c16": {"impl": "E1E2", "check_interval": 16},
    "E1E2c8": {"impl": "E1E2", "check_interval": 8},
    "E1E3": {"impl": "E1E3"},
    "E1E2E3c16": {"impl": "E1E2E3", "check_interval": 16},
    "frozen_t1": {"impl": "frozen", "threads": 1},
    "E1_t1": {"impl": "E1", "threads": 1},
}


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
    check_interval: int | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    impl = dict(T.IMPLEMENTATIONS[implementation])
    interval = int(impl["check_interval"]) if check_interval is None else int(check_interval)
    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(int(threads))
    try:
        C.seed_everything(int(task["seed"]))
        model, audit = M.build_model(
            task["spec"], task["norm"], scale=int(scale), seed=int(task["seed"]), device=device
        )
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
        y_z_all = task["train_y_z"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        total_batches = int(warmup) + int(steps)
        if int(order.size) < total_batches * int(batch_size):
            raise ValueError("row order is shorter than warmup+measured steps")
        rows_warm = order[: int(warmup) * int(batch_size)]
        rows_meas = order[int(warmup) * int(batch_size) : total_batches * int(batch_size)]
        losses: list[float] = []
        grad_norms: list[float] = []
        global_step = 0

        def _window(window_rows: np.ndarray, *, start_step: int) -> list[float]:
            nonlocal global_step
            deferred = T.DeferredLog(check_interval=interval, clip=clip) if bool(impl["defer"]) else None
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
                    times.append(1e3 * float(time.perf_counter() - t0))
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
                        deferred.flush(
                            block_seconds=float(time.perf_counter() - block_start), first_step=first_step
                        )
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

        cpu0 = C.cpu_seconds()
        t_warm = time.perf_counter()
        _window(rows_warm, start_step=0)
        warmup_wall = float(time.perf_counter() - t_warm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cpu_warm = float(C.cpu_seconds() - cpu0)
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
        molecules = int(rows_meas.size)
        return {
            "implementation": implementation,
            "impl": impl,
            "check_interval": int(interval),
            "threads": None if threads is None else int(threads),
            "batch_size": int(batch_size),
            "scale": int(scale),
            "parameters": int(audit["total_parameters"]),
            "warmup_steps": int(warmup),
            "measured_steps": int(len(step_times)),
            "warmup_wall_seconds": warmup_wall,
            "wall_seconds_measured": measured_wall,
            "steps_per_second": float(len(step_times) / measured_wall) if measured_wall > 0 else None,
            "molecules_per_second": float(molecules / measured_wall) if measured_wall > 0 else None,
            "step_ms": _stats(step_times),
            "process_cpu_seconds_measured": cpu_measured,
            "process_cpu_seconds_warmup": cpu_warm,
            "process_cpu_seconds_per_measured_step": float(cpu_measured / max(len(step_times), 1)),
            "final_loss": float(losses[-1]) if losses else None,
            "final_grad_norm": float(grad_norms[-1]) if grad_norms else None,
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
            "producer_counters": dict(prefetcher.counters) if prefetcher is not None else None,
            "step_timing_mode": "block-deferred" if bool(impl["defer"]) else "per-step",
        }
    finally:
        if threads is not None:
            torch.set_num_threads(int(previous_threads))


def _pre_touch(task: Mapping[str, Any], order: np.ndarray, batch_size: int) -> None:
    cache = task["train_cache"]
    for i in range(0, int(order.size), batch_size):
        EC.collate_v2(cache, order[i : i + batch_size].tolist())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "bench.json"))
    parser.add_argument("--probe-batch-sizes", default="256,512")
    parser.add_argument("--probe-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--configs", default=",".join(BENCH_CONFIGS))
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    C.apply_deterministic_settings()
    task = DATA.load_task("PCQM_W3_1M", verify_hashes=True)
    labels = [x for x in str(args.configs).split(",") if x]
    unknown = [x for x in labels if x not in BENCH_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}")
    n_batches_needed = int(args.warmup) + int(args.steps)
    order = np.random.default_rng(int(args.seed)).permutation(C.TRAIN_ROWS)[: n_batches_needed * C.BATCH_SIZE]
    C.log(f"[bench] pre-touching {n_batches_needed * C.BATCH_SIZE} rows ...")
    _pre_touch(task, order, C.BATCH_SIZE)

    rounds: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    t0 = time.perf_counter()
    for round_idx in range(int(args.rounds)):
        seq = labels if round_idx % 2 == 0 else list(reversed(labels))
        C.log(f"[bench] round {round_idx + 1}/{args.rounds}: {seq}")
        for label in seq:
            cfg = BENCH_CONFIGS[label]
            res = run_config(
                task,
                implementation=str(cfg["impl"]),
                steps=int(args.steps),
                warmup=int(args.warmup),
                order=order,
                batch_size=C.BATCH_SIZE,
                device=device,
                scale=int(args.scale),
                check_interval=cfg.get("check_interval"),
                threads=cfg.get("threads"),
            )
            res["label"] = label
            rounds[label].append(res)
            C.log(
                f"[bench] r{round_idx + 1} {label}: {res['step_ms']['mean']:.2f} ms/step, "
                f"{res['molecules_per_second']:.1f} mol/s, final loss {res['final_loss']:.8f}"
            )

    aggregates: dict[str, Any] = {}
    for label, results in rounds.items():
        mols = [float(r["molecules_per_second"]) for r in results]
        med = float(statistics.median(mols))
        steps_ms = [float(r["step_ms"]["mean"]) for r in results]
        aggregates[label] = {
            "label": label,
            "rounds": len(results),
            "molecules_per_second_median": med,
            "molecules_per_second_values": mols,
            "molecules_per_second_min": min(mols),
            "molecules_per_second_max": max(mols),
            "molecules_per_second_spread_pct": float(100.0 * (max(mols) - min(mols)) / med) if med else None,
            "step_ms_mean_values": steps_ms,
            "step_ms_mean_median": float(statistics.median(steps_ms)),
            "step_ms_p50_median": float(statistics.median([float(r["step_ms"]["p50"]) for r in results])),
            "step_ms_p90_median": float(statistics.median([float(r["step_ms"]["p90"]) for r in results])),
            "process_cpu_seconds_per_step_median": float(
                statistics.median([float(r["process_cpu_seconds_per_measured_step"]) for r in results])
            ),
            "gpu_peak_mb": float(statistics.median([r["gpu_peak_allocated_bytes"] for r in results])) / 2**20,
        }
    frozen_med = aggregates["frozen"]["molecules_per_second_median"]
    for label, agg in aggregates.items():
        agg["speedup_vs_frozen"] = float(agg["molecules_per_second_median"] / frozen_med) if frozen_med else None
    fingerprints = {label: sorted({r["weights_sha256"] for r in results}) for label, results in rounds.items()}
    all_hashes = {h for hashes in fingerprints.values() for h in hashes}
    equivalence = {
        "all_configs_and_rounds_same_weights": len(all_hashes) == 1,
        "distinct_weight_hashes": sorted(all_hashes),
        "per_config_hashes": fingerprints,
        "note": "each config/round starts from the same seed-0 state and consumes the same rows",
    }
    non_frozen = [x for x in labels if x != "frozen"]
    best = max(non_frozen, key=lambda n: aggregates[n]["molecules_per_second_median"])

    probes: dict[str, Any] = {}
    for bs in [int(x) for x in str(args.probe_batch_sizes).split(",") if x.strip()]:
        if bs == C.BATCH_SIZE:
            continue
        name = f"E1_bs{bs}"
        C.log(f"[bench] probe {name} ({args.probe_steps} steps) -- throughput only, never a formal recipe")
        probe_order = np.random.default_rng(int(args.seed) + bs).permutation(C.TRAIN_ROWS)[
            : (int(args.probe_steps) + 20) * bs
        ]
        _pre_touch(task, probe_order, bs)
        res = run_config(
            task,
            implementation="E1",
            steps=int(args.probe_steps),
            warmup=20,
            order=probe_order,
            batch_size=bs,
            device=device,
            scale=int(args.scale),
        )
        probes[name] = {
            "batch_size": int(bs),
            "molecules_per_second": res["molecules_per_second"],
            "step_ms_mean": res["step_ms"]["mean"],
            "gpu_peak_mb": res["gpu_peak_allocated_bytes"] / 2**20,
            "measured_steps": res["measured_steps"],
        }
        C.log(f"[bench] {name}: {res['step_ms']['mean']:.2f} ms/step, {res['molecules_per_second']:.1f} mol/s")

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
        "rounds": int(args.rounds),
        "labels": labels,
        "configs": BENCH_CONFIGS,
        "aggregates": aggregates,
        "per_round": rounds,
        "equivalence": equivalence,
        "batch_size_probes": probes,
        "frozen_molecules_per_second_median": frozen_med,
        "best_config": best,
        "speedup_best_vs_frozen": aggregates[best]["speedup_vs_frozen"],
        "best_step_ms_mean": aggregates[best]["step_ms_mean_median"],
        "all_configs_same_weights": bool(equivalence["all_configs_and_rounds_same_weights"]),
        "wall_seconds": float(time.perf_counter() - t0),
        "notes": [
            "medians over interleaved rounds; the pilot single-pass numbers are in bench_pilot.json",
            "every non-probe configuration must reach the same weights fingerprint",
            "batch-256/512 probes are throughput investigations only; formal runs stay at 128",
            "the process burns many CPU cores per step; t1 configurations use one ATen thread",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, C.json_safe(payload))
    print(
        json.dumps(
            {
                "frozen_mol_s": frozen_med,
                "best_config": best,
                "best_mol_s": aggregates[best]["molecules_per_second_median"],
                "speedup_best_vs_frozen": aggregates[best]["speedup_vs_frozen"],
                "all_configs_same_weights": payload["all_configs_same_weights"],
                "aggregates": {
                    k: {
                        "mol_s_median": v["molecules_per_second_median"],
                        "speedup": v["speedup_vs_frozen"],
                        "spread_pct": v["molecules_per_second_spread_pct"],
                    }
                    for k, v in aggregates.items()
                },
                "probes": probes,
            },
            indent=1,
        )
    )
    C.log(f"[bench] wrote {C.track_rel(out)} wall {payload['wall_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
