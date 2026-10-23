"""Sparse upstream-GRIT arm C (audited vendored runtime), used when the dense
execution cannot meet the declared per-run cost ceiling.

This module does not reimplement GRIT: it calls ``zpp_zsrt.train`` with a
declared variant override (10 layers, width 64, 8 heads -> 473,473 parameters)
and records the round's declaration.  It exists so that a 10-layer native GRIT
model can be run inside the round's <=2 h per-run budget without the 20 h
upstream runner.

Deviations from the upstream config, declared before launch:
  * batch 64 instead of 32 (time budget), epochs 1500 instead of 2000,
  * per-step cosine + linear warm-up (repo convention; upstream graphgym uses
    a per-epoch cosine),
  * the sparse upstream execution itself (the dense port was measured and is
    not cheaper on this workload).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_ZPP = _SS / "zinc_perf_protocol_v1" / "code"
for _p in (str(_HERE), str(_ZPP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zpp_common as ZC  # noqa: E402
import zpp_zsrt as ZS  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
ROUND_ID = "zinc-performance-search-v1"

# Declared C0: upstream GRIT backbone (10 layers / 64 wide / 8 heads), native
# model hyper-parameters, horizon adapted to the 2 h per-run ceiling.
C0 = dict(
    layers=10, n_heads=8, dim_hidden=64, ksteps=21,
    dropout=0.0, attn_dropout=0.2, layer_norm=False, batch_norm=True,
    update_e=True, edge_enhance=True, O_e=True, norm_e=True, deg_scaler=True,
    full_attn=True, rrwp=True, graph_pooling="add", loss="l1", patience=0,
    batch_size=64, max_epochs=1500, warmup_epochs=50,
    base_lr=1e-3, min_lr=1e-6, weight_decay=1e-5, clip_grad_norm=1.0,
)


def make_variant(base_cfg: dict, name: str = "C0") -> str:
    doc = ZC.load_yaml(ZC.ZSRT_YAML)
    base = dict(doc["base"])
    base.update(base_cfg)
    doc["variants"] = {name: {}}
    doc["base"] = base
    path = ANALYSIS / "zsrt_c_override.yaml"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    import yaml

    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh)
    return path


def train(seed: int, device, *, epochs: int | None = None, batch_size: int | None = None,
          tag: str | None = None, force: bool = False, reason: str | None = None) -> dict:
    variant = dict(C0)
    if epochs is not None:
        variant["max_epochs"] = int(epochs)
    if batch_size is not None:
        variant["batch_size"] = int(batch_size)
    path = make_variant(variant, "C0")
    # point the loader at the override file and the run root at this round
    ZC.ZSRT_YAML = path
    ZC.RUNS = RUNS
    out = RUNS / (f"{tag}_s{int(seed)}" if tag else f"zsrt_C0_s{int(seed)}")
    out.mkdir(parents=True, exist_ok=True)
    declaration = {
        "round": ROUND_ID, "arm": "C", "variant": "C0-sparse", "seed": int(seed),
        "config": variant, "epochs": variant["max_epochs"],
        "batch_size": variant["batch_size"],
        "steps_per_epoch": (10000 + variant["batch_size"] - 1) // variant["batch_size"],
        "updates": ((10000 + variant["batch_size"] - 1) // variant["batch_size"]) * variant["max_epochs"],
        "warmup_steps": variant["warmup_epochs"] * ((10000 + variant["batch_size"] - 1) // variant["batch_size"]),
        "reason": reason,
        "implementation": "vendored upstream GRIT GritTransformer (commit 6c988ea), sparse runtime",
        "deviations": ["batch 64 (upstream 32)", "1500 epochs (upstream 2000)",
                       "per-step cosine", "official test never read"],
        "official_test_loaded": False,
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ZC.write_json(out / "search.json", declaration)
    info = ZS.train("C0", seed, device, tag=tag, force=force)
    info["round"] = ROUND_ID
    info["search_declaration"] = declaration
    ZC.write_json(out / "train.json", info)
    print(f"[grits C0 s{seed}] best={info['best_valid']['valid_mae']:.6f}"
          f"@{info['best_valid']['epoch']} wall={info['wall_seconds']:.0f}s", flush=True)
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default=None)
    args = p.parse_args(argv)
    dev = torch.device(args.device)
    if args.command == "timing":
        t0 = time.perf_counter()
        info = train(args.seed, dev, epochs=args.epochs or 3,
                     batch_size=args.batch_size, tag="grits_timing", force=True,
                     reason="timing only")
        total = time.perf_counter() - t0
        sec = [float(e["seconds"]) for e in info["curve"]]
        print(json.dumps({"epoch_seconds": sec, "mean": sum(sec) / len(sec),
                          "wall": total}, indent=1), flush=True)
    else:
        train(args.seed, dev, epochs=args.epochs, batch_size=args.batch_size,
              tag=args.tag, force=args.force, reason=args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
