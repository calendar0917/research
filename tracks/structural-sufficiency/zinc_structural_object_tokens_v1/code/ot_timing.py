"""Measure where a B_object_tokens training step spends its time.

Only static, label-free inputs are used.  Prints a per-section wall-clock
breakdown (with CUDA synchronisation) and, with ``--ops``, the top CUDA ops
from ``torch.profiler`` for one batch.  This is a diagnostic tool; it does not
change the model or the recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ot_data as O  # noqa: E402
import ot_models as OM  # noqa: E402
import ot_runner as R  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--ops", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cache = O.load_cache("train")
    norm = D.load_norm()
    gm = R.group_means()
    model, info = R.build_arm_model("B", norm, device, group_mean=gm)
    opt = R.make_optimizer(model, R.WD_B)
    y = np.asarray(cache["labels"], dtype=np.float64)
    gen = torch.Generator().manual_seed(0)
    order = torch.randperm(len(y), generator=gen).numpy()
    t_collate = t_transfer = t_fwd = t_bwd = 0.0
    stats = defaultdict(float)
    for bi in range(args.batches):
        chunk = order[bi * args.batch:(bi + 1) * args.batch]
        sync(device)
        t0 = time.perf_counter()
        batch_cpu = O.collate_ot(cache, chunk)
        t1 = time.perf_counter()
        batch = T1._to_device(batch_cpu, device)
        sync(device)
        t2 = time.perf_counter()
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        model.train()
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        sync(device)
        t3 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync(device)
        t4 = time.perf_counter()
        opt.step()
        t_collate += t1 - t0
        t_transfer += t2 - t1
        t_fwd += t3 - t2
        t_bwd += t4 - t3
        ntok = float(batch["tok_valid"].sum())
        stats["tokens"] += ntok
        stats["Tmax"] += float(batch["Tmax"])
        stats["atoms"] += float(batch["atom_feat"].shape[0])
        stats["pairs"] += float(batch["branch_pat_atom"].shape[0])
        stats["rings"] += float(batch["ring_pat_atom"].shape[0])
        print(f"[batch {bi}] Tmax={int(batch['Tmax'])} tokens={int(ntok)} "
              f"collate={1e3*(t1-t0):.1f}ms transfer={1e3*(t2-t1):.1f}ms "
              f"fwd={1e3*(t3-t2):.1f}ms bwd={1e3*(t4-t3):.1f}ms", flush=True)
    n = float(args.batches)
    per_batch = {"collate_ms": 1e3 * t_collate / n, "transfer_ms": 1e3 * t_transfer / n,
                 "forward_ms": 1e3 * t_fwd / n, "backward_ms": 1e3 * t_bwd / n,
                 "step_ms": 1e3 * (t_collate + t_transfer + t_fwd + t_bwd) / n}
    out = {"device": str(device), "batches": args.batches, "batch_size": args.batch,
           "per_batch": per_batch,
           "projected_epoch_seconds": 79 * per_batch["step_ms"] / 1e3,
           "means": {k: v / n for k, v in stats.items()},
           "torch_threads": torch.get_num_threads(),
           "parameters": info["parameters"]}
    print(json.dumps(out, indent=2), flush=True)

    if args.ops:
        chunk = order[:args.batch]
        batch = T1._to_device(O.collate_ot(cache, chunk), device)
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        for _ in range(2):
            pred = model(batch)
            F.l1_loss(pred.view(-1), target).backward()
            opt.zero_grad(set_to_none=True)
        sync(device)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            pred = model(batch)
            F.l1_loss(pred.view(-1), target).backward()
            opt.zero_grad(set_to_none=True)
            sync(device)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)
        cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=12)
        print(cpu, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
