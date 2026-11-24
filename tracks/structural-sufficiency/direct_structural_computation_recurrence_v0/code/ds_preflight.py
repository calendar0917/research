"""DSCR v0 throughput preflight and frozen-horizon decision.

Measures, on the real PCBA batches and the real training step:

* per-arm training step time (collate + forward + backward + clip + AdamW + the
  new-module gradient audit), on the real plan prefix;
* full Train / Valid / Test inference + official AP cost;
* checkpoint save / reload cost;
* the same numbers under a two-GPU co-tenant (run this probe twice in parallel).

Then it simulates the frozen 8-unit queue on two GPUs with a 15% time reserve and
writes ``analysis/preflight_timing.json``, ``analysis/schedule.json`` and
``configs/horizon.json``.  No model accuracy is used: the horizon is chosen from
timing only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import ds_common as C  # noqa: E402
import ds_model as M  # noqa: E402
import ds_train as T  # noqa: E402
import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def timed_steps(
    model: torch.nn.Module,
    task: Any,
    device: torch.device,
    rows_batches: Sequence[np.ndarray],
    *,
    n_warmup: int = 5,
    n_steps: int = 30,
) -> dict[str, Any]:
    stager = PinnedStager(device, depth=2)
    opt, _ = T.build_optimizer(model)
    model.train()
    times = []
    step = 0
    total = n_warmup + n_steps
    for i in range(total):
        b = rows_batches[i % len(rows_batches)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batch = stager.batch(C.PC.collate(task.cache, b.tolist()))
        y = stager.labels(task.labels[b])
        out = T.train_step(model, opt, batch, y, global_step=step, clip=5.0, audit_new=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step += 1
        if i >= n_warmup:
            times.append(dt)
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_steps": int(n_steps),
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "steps_per_second": float(1.0 / arr.mean()),
        "last_loss": float(out["loss"]),
    }


def time_eval(model: torch.nn.Module, task: Any, device: torch.device, test_labels: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    model.eval()
    for split in ("train", "valid", "test"):
        t0 = time.perf_counter()
        with torch.inference_mode():
            rep = C.evaluate_split(
                model, task, device, split,
                y=test_labels if split == "test" else None,
                per_task=(split != "train"), reason="preflight_timing",
            )
        out[split] = {
            "seconds": float(time.perf_counter() - t0),
            "infer_seconds": float(rep["infer_seconds"]),
            "metric_seconds": float(rep["metric_seconds"]),
            "ap": float(rep["ap"]),
        }
    out["total_seconds"] = float(sum(v["seconds"] for k, v in out.items() if isinstance(v, dict)))
    return out


def time_checkpoint(model: torch.nn.Module, out_dir: Path) -> dict[str, Any]:
    opt, _ = T.build_optimizer(model)
    path = out_dir / "_preflight_checkpoint.pt"
    t0 = time.perf_counter()
    payload = T.checkpoint_payload(
        arm="PREFLIGHT", seed=0, horizon=0, epoch=0, global_step=0, model=model, optimizer=opt,
        rng_state=C.capture_rng_state(), plan={"global_plan_sha256": "preflight"}, model_audit={"parameters": {"total": 0}},
        valid_ap=None, kind="preflight",
    )
    sha = T._atomic_save(payload, path)
    save_s = float(time.perf_counter() - t0)
    t1 = time.perf_counter()
    payload2 = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload2["state_dict"], strict=True)
    load_s = float(time.perf_counter() - t1)
    size = int(path.stat().st_size)
    path.unlink()
    return {"save_seconds": save_s, "reload_seconds": load_s, "bytes": size, "sha256": sha}


def project(
    per_arm: Mapping[str, Any],
    arms: Sequence[str],
    steps_per_epoch: int,
    horizon: int,
) -> dict[str, Any]:
    out = {}
    for arm in arms:
        rec = per_arm[arm]
        epoch_s = steps_per_epoch * rec["step_ms_mean"] / 1e3 + rec["eval_total_seconds"] + rec.get("checkpoint_seconds", 0.0)
        out[arm] = {
            "epoch_seconds": float(epoch_s),
            "horizon_seconds": float(epoch_s * horizon),
            "horizon_gpu_hours": float(epoch_s * horizon / 3600.0),
        }
    return out


def simulate(
    epoch_seconds: Mapping[str, float],
    units: Sequence[Sequence[str]],
    *,
    available_seconds: float,
    reserve_fraction: float = 0.15,
) -> dict[str, Any]:
    """Wave simulation of the frozen unit list on two GPUs (greedy: first free GPU)."""
    gpu_free = [0.0, 0.0]
    assignment = []
    for wave in units:
        gpu = int(np.argmin(gpu_free))
        start = gpu_free[gpu]
        dur = max(epoch_seconds[a] for a in wave)
        gpu_free[gpu] = start + dur
        assignment.append({"gpu": gpu, "units": list(wave), "start_seconds": start,
                           "duration_seconds": dur, "finish_seconds": gpu_free[gpu]})
    makespan = max(gpu_free)
    return {
        "assignment": assignment,
        "makespan_seconds": float(makespan),
        "makespan_hours": float(makespan / 3600.0),
        "available_seconds": float(available_seconds),
        "reserve_fraction": float(reserve_fraction),
        "fits": bool(makespan <= available_seconds / (1.0 + reserve_fraction)),
    }


#: The frozen unit list of the protocol, in priority order (waves of <=2 units).
UNIT_WAVES = [
    ["FIXED1_seed0", "REAL2_seed0"],
    ["SELF_seed0", "BAG2_seed0"],
    ["W2_seed1", "DYNAMIC_seed1"],
    ["Cstar_seed1", "Kstar_seed1"],
]
#: Seed-1 unit 7/8 depend on the seed-0 outcome, but their *cost* is identical to
#: a fixed pair of deep arms plus one W2, so the cost model uses these labels.
UNIT_COST = {
    "FIXED1_seed0": "FIXED1",
    "REAL2_seed0": "REAL2",
    "SELF_seed0": "SELF",
    "BAG2_seed0": "BAG2",
    "W2_seed1": "W2",
    "DYNAMIC_seed1": "DYNAMIC",
    "Cstar_seed1": "REAL2",
    "Kstar_seed1": "BAG2",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS / "preflight_timing.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--deadline-utc", default="", help="wall-clock deadline (epoch seconds)")
    parser.add_argument("--remaining-seconds", type=float, default=0.0)
    parser.add_argument("--report-reserve-seconds", type=float, default=2700.0)
    parser.add_argument("--written-report-reserve-seconds", type=float, default=0.0)
    parser.add_argument("--write-schedule", action="store_true")
    parser.add_argument("--arms", default="")
    parser.add_argument("--eval-arms", default="W2,BAG2")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    device = torch.device(args.device)
    task = C.SD.load_task()
    test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    test_labels = np.asarray(test_info["labels"])
    plan, plan_meta = C.plan_for_seed(task, 0)
    batches = plan[0]["batches"][:8]
    arms = [a for a in (args.arms.split(",") if args.arms else list(C.ARM_ORDER)) if a]
    eval_arms = [a for a in args.eval_arms.split(",") if a]

    out: dict[str, Any] = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "plan": plan_meta,
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "arms": {},
        "startup": {},
        "environment": C.environment(),
        "n_steps": int(args.n_steps),
        "created_utc": C.utcnow(),
    }
    t_load = time.perf_counter()
    C.SD.load_task()
    out["startup"]["task_load_seconds"] = float(time.perf_counter() - t_load)
    out["startup"]["test_label_load_seconds"] = float(time.perf_counter() - t_load)

    for arm in arms:
        t_build = time.perf_counter()
        if arm == "W2":
            model, audit = M.build_w2(task.spec, task.norm, C.N_TASKS, seed=0, device=device)
        else:
            model, audit = M.build_ds_model(task.spec, task.norm, C.N_TASKS, arm=arm, seed=0, device=device)
        build_s = float(time.perf_counter() - t_build)
        rec: dict[str, Any] = {
            "parameters": int(audit["parameters"]["total"]),
            "build_seconds": build_s,
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        step = timed_steps(model, task, device, [np.asarray(b) for b in batches],
                           n_warmup=args.n_warmup, n_steps=args.n_steps)
        rec.update(step)
        rec["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        if arm in eval_arms:
            ev = time_eval(model, task, device, test_labels)
            rec["eval"] = ev
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.cpu()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out["arms"][arm] = rec
        C.log(f"[preflight] {arm} step_ms {step['step_ms_mean']:.2f} (p50 {step['step_ms_p50']:.2f}) build {build_s:.1f}s")

    # checkpoint cost, measured on the last built arm class
    ck_model, _ = M.build_ds_model(task.spec, task.norm, C.N_TASKS, arm="REAL2", seed=0, device="cpu")
    ck = time_checkpoint(ck_model, C.RUNS / "_preflight")
    out["checkpoint"] = ck
    del ck_model

    # ---- projection (per-epoch and per-horizon)
    eval_total = {
        arm: (out["arms"][arm]["eval"]["total_seconds"] if "eval" in out["arms"][arm] else None)
        for arm in out["arms"]
    }
    known = [v for v in eval_total.values() if v is not None]
    eval_fallback = float(np.mean(known)) if known else 0.0
    per_arm = {}
    for arm in C.ARM_ORDER:
        if arm in out["arms"]:
            base = out["arms"][arm]
            per_arm[arm] = {
                "step_ms_mean": base["step_ms_mean"],
                "eval_total_seconds": eval_total.get(arm) if eval_total.get(arm) is not None else eval_fallback,
                "checkpoint_seconds": ck["save_seconds"],
                "measured_step": arm in out["arms"],
                "measured_eval": eval_total.get(arm) is not None,
            }
        else:
            # unmeasured arm: use the mean of the measured deep arms
            deep = [a for a in out["arms"] if a != "W2"]
            step_ms = float(np.mean([out["arms"][a]["step_ms_mean"] for a in deep])) if deep else 0.0
            per_arm[arm] = {"step_ms_mean": step_ms, "eval_total_seconds": eval_fallback,
                            "checkpoint_seconds": ck["save_seconds"], "measured_step": False, "measured_eval": False}
    out["per_arm_projection_inputs"] = per_arm
    out["projections"] = {
        str(h): project(per_arm, list(C.ARM_ORDER), C.STEPS_PER_EPOCH, h) for h in C.HORIZONS
    }

    if args.write_schedule:
        available = float(args.remaining_seconds) - float(args.report_reserve_seconds)
        plans = {}
        for label, waves in (
            ("8_units", UNIT_WAVES),
            ("7_units_no_unit8", UNIT_WAVES[:3] + [["Cstar_seed1"]]),
            ("6_units_no_W2_seed1", UNIT_WAVES[:2] + [["DYNAMIC_seed1"], ["Cstar_seed1"]]),
        ):
            for h in C.HORIZONS:
                epoch_seconds = {
                    unit: per_arm["W2" if UNIT_COST[unit] == "W2" else UNIT_COST[unit]]["step_ms_mean"] * C.STEPS_PER_EPOCH / 1e3
                    + per_arm[UNIT_COST[unit]]["eval_total_seconds"]
                    + per_arm[UNIT_COST[unit]]["checkpoint_seconds"]
                    for wave in waves for unit in wave
                }
                sim = simulate(epoch_seconds, waves, available_seconds=available)
                plans[f"{label}_H{h}"] = {**sim, "horizon": h, "label": label,
                                          "epoch_seconds": {k: float(v) for k, v in epoch_seconds.items()}}
        chosen = None
        for label in ("8_units", "7_units_no_unit8", "6_units_no_W2_seed1"):
            feasible = [h for h in C.HORIZONS if plans[f"{label}_H{h}"]["fits"]]
            if feasible:
                chosen = {"label": label, "horizon": max(feasible)}
                break
        if chosen is None:
            chosen = {"label": "DEGRADED_seed0_only", "horizon": min(C.HORIZONS)}
        units = {
            "8_units": UNIT_WAVES,
            "7_units_no_unit8": UNIT_WAVES[:3] + [["Cstar_seed1"]],
            "6_units_no_W2_seed1": UNIT_WAVES[:2] + [["DYNAMIC_seed1"], ["Cstar_seed1"]],
        }.get(chosen["label"], [["FIXED1_seed0", "REAL2_seed0"], ["SELF_seed0", "BAG2_seed0"]])
        C.write_json(
            C.ANALYSIS / "schedule.json",
            {
                "schema": C.SCHEMA_SCHEDULE,
                "round_id": C.ROUND_ID,
                "chosen": chosen,
                "units": units,
                "horizon": int(chosen["horizon"]),
                "available_seconds_after_report_reserve": available,
                "report_reserve_seconds": float(args.report_reserve_seconds),
                "plans": plans,
                "simulator": "two GPUs, greedy first-free assignment, wave duration = max(unit duration), 15% reserve",
                "reserve_fraction": 0.15,
                "note": "horizon is chosen from timing only; no accuracy enters this decision",
                "created_utc": C.utcnow(),
            },
        )
        C.write_json(
            C.CONFIGS / "horizon.json",
            {
                "schema": "dscr-horizon-freeze-v1",
                "horizon": int(chosen["horizon"]),
                "schedule_label": chosen["label"],
                "selected_by": "analysis/preflight_timing.json + analysis/schedule.json (timing only)",
                "hours": [int(h) for h in C.HORIZONS],
                "lr_schedule": "prefix of the frozen 40-epoch per-step cosine; never recompressed",
                "frozen_utc": C.utcnow(),
            },
        )
        out["schedule"] = {"chosen": chosen}

    out["seconds"] = float(time.perf_counter() - t0)
    C.write_json(Path(args.out), out)
    C.log(f"[preflight] done in {out['seconds']:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
