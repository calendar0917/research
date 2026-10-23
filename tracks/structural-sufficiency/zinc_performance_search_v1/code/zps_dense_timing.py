"""Batch-size / throughput timing for the dense GRIT arm (arm C).

Runs on the full official train split (10,000 graphs) and a full 1,000-molecule
valid epoch, for the declared candidate batch sizes.  No epoch here is a result;
this only chooses a horizon that fits the round's per-run cost ceiling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import zps_grit_dense as G  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def time_batch(train_g, valid_g, bs: int, device: str, steps: int, cfg: dict) -> dict:
    y_valid = np.asarray([g["y"] for g in valid_g], dtype=np.float64)
    torch.manual_seed(0)
    model = G.DenseGrit(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(7)
    order = torch.randperm(len(train_g), generator=gen).tolist()
    # warm-up step (cudnn/allocator) not counted
    model.train()
    b = G.collate(train_g, order[:bs], device)
    pred = G.batch_pred(model, b).view(-1)
    loss = F.l1_loss(pred, b["y"])
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize() if str(device).startswith("cuda") else None
    times = []
    for s in range(1, steps + 1):
        rows = order[s * bs:(s + 1) * bs]
        if len(rows) < bs:
            rows = order[:bs]
        t0 = time.perf_counter()
        model.train()
        b = G.collate(train_g, rows, device)
        pred = G.batch_pred(model, b).view(-1)
        loss = F.l1_loss(pred, b["y"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    _, valid_mae = G.evaluate(model, valid_g, np.arange(len(valid_g)), y_valid, device,
                              batch_size=bs)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    valid_seconds = time.perf_counter() - t0
    step_mean = float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))
    steps_per_epoch = math.ceil(len(train_g) / bs)
    epoch_seconds = step_mean * steps_per_epoch + valid_seconds
    return {
        "batch_size": bs,
        "steps_measured": len(times),
        "step_seconds": times,
        "mean_step_seconds": step_mean,
        "steps_per_epoch": steps_per_epoch,
        "valid_seconds": valid_seconds,
        "projected_epoch_seconds": epoch_seconds,
        "projected_2000ep_hours_with_startup_15pct": 1.15 * 2000 * epoch_seconds / 3600.0,
        "projected_1500ep_hours_with_startup_15pct": 1.15 * 1500 * epoch_seconds / 3600.0,
        "valid_mae_after_measured_steps": float(valid_mae),
        "train_mae_after_measured_steps": float(loss.detach()),
        "parameters": G.param_count(model),
        "official_test_loaded": False,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", default="32,64,128")
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args(argv)
    train_g = G.prepare("train")
    valid_g = G.prepare("valid")
    out = {"device": args.device, "timing_rows": "full official train 10000",
           "results": []}
    for bs in [int(x) for x in args.batches.split(",")]:
        r = time_batch(train_g, valid_g, bs, args.device, args.steps, dict(G.NATIVE))
        print(f"[timing] batch {bs}: {r['mean_step_seconds']*1000:.1f} ms/step, "
              f"{r['projected_epoch_seconds']:.2f} s/epoch, 2000ep="
              f"{r['projected_2000ep_hours_with_startup_15pct']:.2f} h", flush=True)
        out["results"].append(r)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "dense_timing.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
