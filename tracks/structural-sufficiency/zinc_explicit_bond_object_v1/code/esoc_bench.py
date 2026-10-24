"""Section-level timing for ESOC-v1 (diagnostic only, no result)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402
import esoc_train as T  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def bench_steps(steps: int = 20, device: str = "cuda:0", arm: str = "real") -> dict:
    scales = T.read_json(ANALYSIS / "scales.json")
    cfg = T.load_config()["recipe"]
    model = M.build_model(0, arm, scales, n_layers=int(cfg["n_layers"])).to(device)
    opt = T.make_optimizer(model, 2e-3, 1e-5)
    data = D.SplitTensors(D.load_cache("train"), device)
    rows = T.batch_order(0, 1, data.n, 32)

    def sync():
        torch.cuda.synchronize()

    marks = {}

    def tick(name, prev):
        sync()
        now = time.perf_counter()
        marks[name] = marks.get(name, 0.0) + (now - prev)
        return now

    # warm-up
    for s in range(3):
        b = data.batch(rows[s * 32:(s + 1) * 32])
        p = model(b)
        loss = (p - b["y"]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats()
    total0 = time.perf_counter()
    for s in range(steps):
        t = time.perf_counter()
        b = data.batch(rows[s * 32:(s + 1) * 32])
        t = tick("batch_gather", t)
        h = model.local(b["loc_type"], b["loc_role"], b["loc_et"], b["loc_mask"],
                        b["loc_nnodes"], b["loc_nedges"], b["bond_mask"], arm)
        t = tick("local_encoder", t)
        r = M.rrwp(b["bonds"], b["bond_mask"], b["n_atoms"])
        bias = model.pe(r).permute(0, 3, 1, 2)
        t = tick("rrwp_and_pe", t)
        counts = model._counts(b)
        g = model.gdesc((counts - model.count_mean) / model.count_scale)
        keep = torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
        for blk in model.blocks:
            h = blk(h, bias, b["bond_mask"], g, keep)
        t = tick("trunk_forward", t)
        out = torch.cat([(h * b["bond_mask"][..., None]).sum(1) / model.ref_bonds,
                         (h * b["bond_mask"][..., None]).pow(2).sum(1) / model.ref_bonds,
                         b["atom_count"] @ model.local.atom.weight,
                         counts, b["global33"]], dim=-1)
        pred = model.readout(out).squeeze(-1)
        loss = (pred - b["y"]).abs().mean()
        t = tick("readout_forward", t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t = tick("backward", t)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        t = tick("optimizer", t)
    sync()
    wall = time.perf_counter() - total0
    steps_per_epoch = 10000 // 32
    return {
        "steps": steps,
        "arm": arm,
        "wall_seconds": wall,
        "ms_per_step": wall / steps * 1000.0,
        "sections_ms_per_step": {k: v / steps * 1000.0 for k, v in marks.items()},
        "s_per_epoch_train_only": wall / steps * steps_per_epoch,
        "projected_hours_2500_epochs_train_only": wall / steps * steps_per_epoch * 2500 / 3600.0,
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20),
        "param_total": M.param_audit(model)["param_total"],
        "environment": C.environment(),
    }


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arm", default="real")
    args = p.parse_args(argv)
    res = bench_steps(args.steps, args.device, args.arm)
    (ANALYSIS / "raw").mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "raw" / "bench.json", res)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
