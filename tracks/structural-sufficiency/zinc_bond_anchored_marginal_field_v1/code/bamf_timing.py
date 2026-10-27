"""GPU timing / projection for BAMF-v1.

Runs three complete Fit8k epochs (63 updates each, full Monitor1k evaluation)
through the exact same code path as the formal run, then projects the full
47,400-update run.  Also micro-benchmarks G vs I forward+backward on one batch
so the cost of the exact elimination is separated from the rest of the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (
    str(_HERE),
    str(_SUFF / "zinc_arch_discovery_v1" / "code"),
    str(_SUFF / "zinc_dual_channel_composer_v1" / "code"),
    str(_SUFF / "zinc_dccl_bottleneck_v1" / "code"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bamf_models as BM  # noqa: E402
import bamf_runner as R  # noqa: E402
import dcb_train as DT  # noqa: E402

# 3 complete epochs at 63 updates/epoch; the formal run is 753 epochs.
TIMING_EPOCHS = 3
FULL_EPOCHS = 753
FULL_UPDATES = 47_400
PER_RUN_GATE_HOURS = 2.5
TOTAL_GATE_HOURS = 12.0


def _microbench(device, arm: str) -> dict:
    split = DT.split_rows()
    cache = D.load_cache("train")
    norm = DT.fit_only_norm(cache, split["fit8k"])
    plan = R.build_plan(cache) if arm == "G" else None
    state: dict = {}
    builder = R.make_builder(arm, norm, plan, state)
    collate = R.make_collate(arm, plan, device, state)
    model, info = builder("BAMF", norm, 0, device)
    model.train()
    rows = split["fit8k"][:128]
    b = T1._to_device(collate(cache, rows, "BAMF", valid=False), device)
    y = torch.as_tensor(np.asarray(cache["labels"], dtype=np.float64)[rows],
                        dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []
    for i in range(8):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        pred = model(b)
        loss = F.l1_loss(pred.view(-1), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        opt.step()
    return {
        "arm": arm,
        "batch_rows": 128,
        "first_warmup_seconds": times[0],
        "steady_seconds": float(np.median(times[2:])),
        "all_seconds": times,
        "parameters": int(info["parameters"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["G", "I"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rr-job-id", default=None)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t_total = time.perf_counter()
    info = R.train(
        args.arm, 0, device,
        total_updates=TIMING_EPOCHS * 63,
        tag="timing", force=True, rr_job_id=args.rr_job_id,
    )
    total_seconds = time.perf_counter() - t_total
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    sum_epoch = float(np.sum(epoch_seconds))
    mean_epoch = float(np.mean(epoch_seconds))
    steady_epoch = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    final_eval = float(info["wall_seconds"]) - sum_epoch
    plan_info = info.get("plan") or {}
    plan_seconds = float(plan_info.get("build_seconds") or 0.0)
    startup = max(0.0, total_seconds - float(info["wall_seconds"]) - plan_seconds)
    projected_train = FULL_EPOCHS * mean_epoch
    projected_total = 1.15 * projected_train + startup + final_eval + plan_seconds + 30.0
    peak_mb = None
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    out = {
        "protocol_id": R.ROUND_ID,
        "arm": args.arm,
        "seed": 0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timing_epochs": TIMING_EPOCHS,
        "timing_updates": int(info["total_updates"]),
        "steps_per_epoch": int(info["steps_per_epoch"]),
        "epoch_seconds_including_full_monitor": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady_epoch,
        "sum_epoch_seconds": sum_epoch,
        "plan_build_seconds": plan_seconds,
        "plan": info.get("plan"),
        "startup_seconds": startup,
        "final_monitor_eval_seconds": final_eval,
        "projected_full_epochs": FULL_EPOCHS,
        "projected_updates": FULL_UPDATES,
        "projected_train_seconds": projected_train,
        "projected_total_seconds_with_15pct_and_overheads": projected_total,
        "projected_gpu_hours": projected_total / 3600.0,
        "per_run_gate_gpu_hours": PER_RUN_GATE_HOURS,
        "total_round_gate_gpu_hours": TOTAL_GATE_HOURS,
        "within_per_run_gate": bool(projected_total / 3600.0 <= PER_RUN_GATE_HOURS),
        "peak_cuda_memory_mb": peak_mb,
        "parameters": int(info["parameters"]),
        "parameter_audit": info["parameter_audit"],
        "initial_state_sha256": info["model_info"]["initial_state_sha256"],
        "total_seconds_measured": total_seconds,
        "rr_job_id": args.rr_job_id,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
        "microbench_G": None,
        "microbench_I": None,
    }
    # The microbench is intentionally separate from the full-epoch projection:
    # it measures the backward through the marginal computation on one batch.
    out["microbench_G"] = _microbench(device, "G")
    out["microbench_I"] = _microbench(device, "I")
    out["microbench_G_over_I_steady"] = (
        out["microbench_G"]["steady_seconds"] / out["microbench_I"]["steady_seconds"]
    )
    R.ANALYSIS.mkdir(parents=True, exist_ok=True)
    path = R.ANALYSIS / f"timing_{args.arm}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in (
        "arm", "mean_epoch_seconds", "projected_gpu_hours", "within_per_run_gate",
        "peak_cuda_memory_mb", "plan_build_seconds", "microbench_G_over_I_steady",
    )}, indent=2))
    return 0 if out["within_per_run_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
