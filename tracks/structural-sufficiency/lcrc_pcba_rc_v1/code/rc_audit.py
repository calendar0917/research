"""Pre-flight audit, distance-cache build and the GPU cost profile.

Subcommands
-----------

``build-cache``  build the label-free shortest-bond-path distance cache
``checks``       run the twelve correctness checks on the selected device and
                 write ``analysis/preflight.json`` (a failing check exits 4)
``profile``      measure the real training step of RC-REAL / RC-BLIND against a
                 plain W2 step on the same batches and project the 40-epoch
                 GPU-hour cost (``analysis/profile.json``)
``inference``    measure validation-set inference time of RC vs W2

Nothing here trains a formal model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

import rc_common as C  # noqa: E402
import bs_common as BC  # noqa: E402
import bs_model as BSM  # noqa: E402
import bs_train as BT  # noqa: E402
import rc_checks as CHK  # noqa: E402
import rc_dist as D  # noqa: E402
import rc_model as M  # noqa: E402
import rc_train as T  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

PREFLIGHT_PATH = C.ANALYSIS / "preflight.json"
PROFILE_PATH = C.ANALYSIS / "profile.json"
INFERENCE_PATH = C.ANALYSIS / "inference_timing.json"
DIST_CACHE_BUILD_PATH = C.ANALYSIS / "distance_cache_build.json"


def _synthetic_batch() -> dict[str, Any]:
    """The internally consistent tiny batch used by the correctness checks."""
    sys.path.insert(0, str(C.TESTS))
    from conftest import synthetic_batch

    graphs = [CHK.TINY_GRAPHS[name] for name in ("chain4_plus_chord", "chain5", "ring6", "two_components")]
    return synthetic_batch(graphs=graphs, seed=19)


# ---------------------------------------------------------------------------
# build-cache
# ---------------------------------------------------------------------------


def cmd_build_cache(args: argparse.Namespace) -> int:
    out = Path(args.out)
    meta = D.build_cache(out, verify_fraction=float(args.verify_fraction))
    record = {
        "schema": "lcrc-pcba-rc-distance-cache-build-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "cpu_seconds": float(meta["build_seconds"]),
        "labels_read": False,
        "note": (
            "Structure-only build from atom_ptr/bond_ptr/bond_index. The label artifact is "
            "never opened, so the official Test rows enter the cache without their targets."
        ),
        "meta": meta,
    }
    C.write_json(DIST_CACHE_BUILD_PATH, record)
    C.log(f"[build-cache] wrote {out} ({meta['n_unordered_pairs']} pairs) in {meta['build_seconds']:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def cmd_checks(args: argparse.Namespace) -> int:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    task = None
    dist = None
    try:
        task = D.load_task_gated()
    except Exception as exc:  # noqa: BLE001
        C.log(f"[checks] real task unavailable ({type(exc).__name__}: {exc}); synthetic only")
    try:
        dist = D.load_cache(args.dist_cache)
    except Exception as exc:  # noqa: BLE001
        C.log(f"[checks] distance cache unavailable ({type(exc).__name__}: {exc})")
    spec = task.spec if task is not None else __import__("feature_spec").lipo_v1_spec()
    norm = task.norm if task is not None else {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }
    batch = _synthetic_batch()
    if device != "cpu":
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    grad_batches = [batch]
    if task is not None and dist is not None:
        plan = BC.SD.make_plan40(task, seed=C.SEED)
        grad_batches = []
        for i in range(3):
            rows = np.asarray(plan[0]["batches"][i], dtype=np.int64)[:64]
            b = D.collate_rel(task, dist, rows.tolist())
            grad_batches.append(
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            )
    record = CHK.run_all(spec, norm, batch, task=task, dist=dist, device=device, grad_batches=grad_batches)
    record["distance_cache_available"] = dist is not None
    record["real_task_available"] = task is not None
    record["host_facts"] = C.host_facts()
    record["gpu_occupancy"] = C.gpu_occupancy()
    C.write_json(PREFLIGHT_PATH, record)
    for check in record["checks"]:
        C.log(f"[checks] {'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    C.log(f"[checks] {record['n_passed']}/{record['n_checks']} passed on {device}")
    return 0 if record["all_passed"] else 4


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean() * 1e3),
        "p50_ms": float(np.percentile(arr, 50) * 1e3),
        "p90_ms": float(np.percentile(arr, 90) * 1e3),
        "min_ms": float(arr.min() * 1e3),
        "max_ms": float(arr.max() * 1e3),
        "steps_per_second": float(1.0 / arr.mean()),
    }


def _time_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Any,
    dist: Mapping[str, Any] | None,
    rows_list: Sequence[np.ndarray],
    device: torch.device,
    *,
    steps: int,
    warmup: int,
    global_step: int,
) -> dict[str, Any]:
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    model.train()
    times: list[float] = []
    collate_times: list[float] = []
    for i in range(int(warmup) + int(steps)):
        rows = rows_list[i % len(rows_list)]
        t0 = time.perf_counter()
        raw = BC.PC.collate(task.cache, rows.tolist()) if dist is None else D.collate_rel(task, dist, rows.tolist())
        t1 = time.perf_counter()
        batch = stager.batch(raw)
        y = stager.labels(task.labels[rows])
        BT.train_step(model, optimizer, batch, y, global_step=int(global_step) + i, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        if i >= int(warmup):
            times.append(float(t2 - t1))
            collate_times.append(float(t1 - t0))
    stager.release()
    return {
        "step": _percentiles(times),
        "collate_cpu": _percentiles(collate_times),
        "measured_steps": int(steps),
        "warmup_steps": int(warmup),
    }


def cmd_profile(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.dist_cache)
    plan = BC.SD.make_plan40(task, seed=C.SEED)
    rows_list = [np.asarray(plan[0]["batches"][i], dtype=np.int64) for i in range(int(args.batches))]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    record: dict[str, Any] = {
        "schema": C.SCHEMA_PROFILE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": {"torch": torch.get_num_threads(), "omp_env": os.environ.get("OMP_NUM_THREADS")},
        "batches": int(args.batches),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "batch_rows": [int(r.size) for r in rows_list],
        "runs": {},
    }

    # W2 baseline on the same rows / same recipe / no pair payload
    w2, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    w2_opt, _ = BSM.make_optimizer(w2)
    record["runs"]["W2"] = _time_steps(
        w2, w2_opt, task, None, rows_list, device, steps=int(args.steps), warmup=int(args.warmup), global_step=0
    )
    del w2, w2_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        opt, _ = BSM.make_optimizer(model)
        record["runs"][key] = _time_steps(
            model, opt, task, dist, rows_list, device, steps=int(args.steps), warmup=int(args.warmup), global_step=0
        )
        record["runs"][key]["parameters"] = int(audit["total_parameters"])
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))

    # projection: 40 epochs x 2738 steps + one Validation pass per epoch + a final Train AP pass
    valid_seconds = float(args.valid_seconds)
    train_eval_seconds = float(args.train_eval_seconds)
    steps_per_epoch = C.STEPS_PER_EPOCH
    for key, run in record["runs"].items():
        step_s = float(run["step"]["p50_ms"]) / 1e3
        train_s = step_s * steps_per_epoch * C.TOTAL_EPOCHS
        total = train_s + valid_seconds * C.TOTAL_EPOCHS + train_eval_seconds
        run["projection"] = {
            "step_seconds_p50": step_s,
            "train_seconds_40_epochs": train_s,
            "valid_seconds_40": valid_seconds * C.TOTAL_EPOCHS,
            "train_eval_seconds": train_eval_seconds,
            "total_seconds": total,
            "total_gpu_hours": total / 3600.0,
            "budget_target_hours": float(C.BUDGET["gpu_target_hours_per_arm"]),
            "budget_hard_hours": float(C.BUDGET["gpu_hard_hours_per_arm"]),
            "within_target": bool(total / 3600.0 <= float(C.BUDGET["gpu_target_hours_per_arm"])),
            "within_hard": bool(total / 3600.0 <= float(C.BUDGET["gpu_hard_hours_per_arm"])),
        }
    base = float(record["runs"]["W2"]["step"]["p50_ms"])
    for key, run in record["runs"].items():
        run["step_overhead_vs_w2"] = float(run["step"]["p50_ms"]) / base - 1.0
    record["w2_reference_step_ms_p50"] = base
    C.write_json(PROFILE_PATH, record)
    for key, run in record["runs"].items():
        C.log(
            f"[profile] {key} step p50 {run['step']['p50_ms']:.2f} ms "
            f"({run['step_overhead_vs_w2'] * 100:+.1f}% vs W2) "
            f"projected {run['projection']['total_gpu_hours']:.3f} GPU-h "
            f"(target {C.BUDGET['gpu_target_hours_per_arm']})"
        )
    return 0


# ---------------------------------------------------------------------------
# inference timing
# ---------------------------------------------------------------------------


def cmd_inference(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.dist_cache)
    rows = np.asarray(task.valid_rows, dtype=np.int64)[: int(args.rows)]
    record: dict[str, Any] = {
        "schema": "lcrc-pcba-rc-inference-timing-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "rows": int(rows.size),
        "batch_size": 256,
        "runs": {},
    }
    for key, builder in (("W2", BSM.build_model), ("REAL", M.build_model), ("BLIND", M.build_model)):
        if key == "W2":
            model, audit = builder(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        else:
            model, audit = builder(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        model.eval()
        use_dist = None if key == "W2" else dist
        _ = T.predict_logits(model, task, use_dist or dist, rows[:256], device)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        T.predict_logits(model, task, use_dist or dist, rows, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        record["runs"][key] = {
            "seconds": float(seconds),
            "rows_per_second": float(rows.size / seconds),
            "milliseconds_per_1k_rows": float(seconds / rows.size * 1e3 * 1e3),
            "parameters": int(audit["total_parameters"]),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    base = record["runs"]["W2"]["seconds"]
    for key, run in record["runs"].items():
        run["relative_to_w2"] = float(run["seconds"] / base)
    C.write_json(INFERENCE_PATH, record)
    for key, run in record["runs"].items():
        C.log(
            f"[inference] {key} {run['rows_per_second']:.1f} rows/s "
            f"({run['relative_to_w2']:.3f}x W2)"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    bc = sub.add_parser("build-cache")
    bc.add_argument("--out", default=str(C.DIST_CACHE_DIR))
    bc.add_argument("--verify-fraction", type=float, default=0.002)

    ck = sub.add_parser("checks")
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--dist-cache", default=str(C.DIST_CACHE_DIR))

    pr = sub.add_parser("profile")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--dist-cache", default=str(C.DIST_CACHE_DIR))
    pr.add_argument("--batches", type=int, default=4)
    pr.add_argument("--steps", type=int, default=25)
    pr.add_argument("--warmup", type=int, default=8)
    pr.add_argument("--valid-seconds", type=float, default=8.0)
    pr.add_argument("--train-eval-seconds", type=float, default=60.0)

    inf = sub.add_parser("inference")
    inf.add_argument("--device", default="cuda")
    inf.add_argument("--dist-cache", default=str(C.DIST_CACHE_DIR))
    inf.add_argument("--rows", type=int, default=8192)

    args = parser.parse_args(argv)
    if args.mode == "build-cache":
        return cmd_build_cache(args)
    if args.mode == "checks":
        return cmd_checks(args)
    if args.mode == "profile":
        return cmd_profile(args)
    if args.mode == "inference":
        return cmd_inference(args)
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
