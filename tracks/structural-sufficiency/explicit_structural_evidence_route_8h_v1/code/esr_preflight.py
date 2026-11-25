"""GPU/CPU throughput preflight.

Measures, on real batches and after warmup:

* p50 / p90 optimiser-step time per arm,
* one full split evaluation time,
* peak allocated device memory,
* the per-molecule relation-pair distribution,
* the projected epoch wall time and the 25/40-epoch GPU-hour cost.

Nothing here trains a model to convergence or selects an arm: it is a resource
measurement only (a short random-label-free step loop).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import esr_common as C
import esr_data as D
import esr_model as M


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"p50": 0.0, "p90": 0.0, "mean": 0.0, "n": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "n": float(arr.size),
    }


def bench(
    view: D.View,
    arm: str,
    device: torch.device,
    *,
    steps: int = 6,
    warmup: int = 3,
    batch_size: int = 128,
    eval_batches: int = 6,
) -> dict[str, Any]:
    model, info = M.build_model(arm=arm, view=view, seed=0, device=device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    y_all = view.loss_target("train")
    rng = np.random.default_rng(1234)
    n_train = int(view.rows["train"].size)
    times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    for step in range(int(warmup) + int(steps)):
        rows = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        cache_rows = [int(view.rows["train"][i]) for i in rows]
        batch = view.to_device(view.collate("train", cache_rows), device)
        y = y_all[torch.as_tensor(rows, dtype=torch.long)].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pred = model(batch)
        if view.task == "regression":
            loss = F.mse_loss(pred.reshape(-1), y.reshape(-1))
        else:
            mask = torch.isfinite(y)
            loss = (F.binary_cross_entropy_with_logits(pred, torch.nan_to_num(y), reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        if step >= warmup:
            times.append(dt)
    step_stats = _percentiles(times)
    # full-split evaluation cost
    model.eval()
    n_eval = int(view.rows["valid"].size)
    t0 = time.perf_counter()
    with torch.no_grad():
        done = 0
        for start in range(0, n_eval, 256):
            chunk = [int(view.rows["valid"][i]) for i in range(start, min(start + 256, n_eval))]
            batch = view.to_device(view.collate("valid", chunk), device)
            model(batch)
            done += len(chunk)
            if start // 256 + 1 >= int(eval_batches):
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = (time.perf_counter() - t0) * (n_eval / max(done, 1))
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    pair_meta = view.pairs["train"]["meta"]["n_pairs_per_molecule"]
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    epoch_seconds = step_stats["p50"] * steps_per_epoch + eval_seconds
    out = {
        "arm": arm,
        "dataset": view.dataset,
        "device": str(device),
        "parameters": int(info["parameters"]),
        "hidden_width": info.get("hidden_width"),
        "batch_size": int(batch_size),
        "steps_per_epoch": steps_per_epoch,
        "step_ms": {k: v * 1000.0 for k, v in step_stats.items()},
        "valid_eval_seconds": float(eval_seconds),
        "peak_allocated_bytes": peak,
        "n_relation_pairs_per_molecule": pair_meta,
        "projected_epoch_seconds": float(epoch_seconds),
        "projected_gpu_hours": {
            f"{e}_epochs": float((epoch_seconds + eval_seconds * 0) * e / 3600.0) for e in (1, 25, 40, 360)
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESR throughput preflight")
    ap.add_argument("--dataset", default="PCBA", choices=list(C.DATASETS))
    ap.add_argument("--arms", nargs="+", default=["BASE", "SREN-ADD", "SREN-EVID", "SREN-BANK", "HIDDEN", "SREN-COMPOUND", "UNARY"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=str(C.ANALYSIS / "preflight_timing.json"))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    view = D.load_view(args.dataset, include_test=False, max_dist=2, workers=8)
    report = {
        "round_id": C.ROUND_ID,
        "dataset": args.dataset,
        "captured_utc": C.utcnow(),
        "environment": C.environment(),
        "gpu": {
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "total_memory": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None,
        },
        "n_train": int(view.rows["train"].size),
        "n_valid": int(view.rows["valid"].size),
        "pair_counts": {split: view.pairs[split]["meta"]["n_pairs_per_molecule"] for split in view.pairs},
        "arms": {},
        "timing_source": "real batches, warmup discarded, p50 over the measured steps; eval projected from a 6-batch sample",
    }
    for arm in args.arms:
        t0 = time.perf_counter()
        C.log(f"bench {args.dataset}/{arm} on {device}")
        report["arms"][arm] = bench(view, arm, device, steps=args.steps, warmup=args.warmup)
        C.log(
            f"{arm}: p50 {report['arms'][arm]['step_ms']['p50']:.1f} ms/step, epoch "
            f"{report['arms'][arm]['projected_epoch_seconds'] / 60:.1f} min, peak "
            f"{report['arms'][arm]['peak_allocated_bytes'] / 2**30:.2f} GiB ({time.perf_counter() - t0:.1f}s)"
        )
    C.write_json(Path(args.out), report)
    C.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
