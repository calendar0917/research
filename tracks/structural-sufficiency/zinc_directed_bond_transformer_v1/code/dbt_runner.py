"""CLI for the zinc-directed-bond-transformer-v1 round.

    python tracks/structural-sufficiency/zinc_directed_bond_transformer_v1/code/dbt_runner.py prep
    ... checks
    ... timing --device cuda:0
    ... train  --device cuda:0 [--resume]
    ... analysis
    ... test   --device cuda:0      # terminal, after the freeze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_analysis as BA  # noqa: E402
import dbt_checks as BC  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_models as BM  # noqa: E402
import dbt_test as BX  # noqa: E402
import dbt_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"


def prep(splits=("train", "valid")) -> dict:
    out: dict = {}
    for split in splits:
        path = BD.cache_path(split)
        t0 = time.perf_counter()
        if path.exists():
            data = BD.load_relation_split(split)
            src = "reuse"
        else:
            data = BD.build_relation_split(split)
            BD.save_relation_split(data)
            src = "build"
        out[split] = {"path": str(path.relative_to(C.REPO)), "source": src,
                      "sha256": C.sha256_file(path),
                      "seconds": float(time.perf_counter() - t0), "meta": data["meta"]}
        C.log(f"[dbt prep] {split} {src} {out[split]['seconds']:.1f}s")
    scales = BD.fit_graph_scales(BD.load_relation_split("train"))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "graph_scales.json", scales)
    C.write_json(ANALYSIS / "prep.json", {**out, "graph_scales": scales})
    C.log(f"[dbt prep] graph scales {scales}")
    return out


def timing(device, epochs: int = 2) -> dict:
    info = BT.train(0, device, epochs=epochs, tag="timing", force=True)
    per_epoch = float(np.mean([c["seconds"] for c in info["curve"]]))
    out = {
        "device": str(device),
        "parameters": info["parameter_audit"]["parameters"],
        "parameters_in_optimizer": info["parameter_audit"]["parameters_in_optimizer"],
        "epoch_seconds": [c["seconds"] for c in info["curve"]],
        "mean_epoch_seconds": per_epoch,
        "steps_per_epoch": info["recipe"]["steps_per_epoch"],
        "projected_gpu_hours_1500_epochs": per_epoch * 1500 / 3600.0,
        "best_valid_two_epoch": info["best_valid"],
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated(device) / 2 ** 20)
        if str(device).startswith("cuda") else None,
        "environment": info["environment"],
    }
    C.write_json(ANALYSIS / "timing.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("--splits", default="train,valid")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("checks")
    s.add_argument("--best", default=None)
    s = sub.add_parser("timing")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--epochs", type=int, default=2)
    s = sub.add_parser("train")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag", default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--save-every", type=int, default=25)
    s = sub.add_parser("analysis")
    s.add_argument("--run", default=str(_EXP / "runs" / "dbt_s0"))
    s = sub.add_parser("test")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--best", default=str(_EXP / "runs" / "dbt_s0" / "best.pt"))
    s.add_argument("--run", default=str(_EXP / "runs" / "dbt_s0"))
    s.add_argument("--freeze-out", default=None)
    s.add_argument("--no-test", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "prep":
        prep(tuple(x for x in args.splits.split(",") if x))
        return 0
    if args.cmd == "checks":
        sys.argv = [sys.argv[0]] + (["--best", args.best] if args.best else [])
        return BC.main([] if not args.best else ["--best", args.best])
    device = torch.device(getattr(args, "device", "cpu"))
    if args.cmd == "timing":
        timing(device, args.epochs)
        return 0
    if args.cmd == "train":
        BT.train(args.seed, device, tag=args.tag, force=args.force,
                 resume=args.resume, save_every=args.save_every)
        return 0
    if args.cmd == "analysis":
        return BA.main(["--run", args.run])
    if args.cmd == "test":
        extra = ["--device", str(device), "--best", args.best, "--run", args.run]
        if args.freeze_out:
            extra += ["--freeze-out", args.freeze_out]
        if args.no_test:
            extra += ["--no-test"]
        return BX.main(extra)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
