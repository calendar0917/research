"""Sequential run queue on one GPU, with per-unit provenance and budget checks.

Each unit still writes its own ``run.json`` / checkpoints / registry entry (the
trainers do that); this driver only orders units and refuses to start one that
cannot finish before the wall-clock stop or that would exceed the GPU hard cap.

Usage::

    python esr_queue.py --gpu-tag gpu0 --units LIPO:BASE LIPO:UNARY ...
    python esr_queue.py --gpu-tag gpu1 --units PCBA:SREN-ADD:25 PCBA:SREN-EVID:25
    python esr_queue.py --gpu-tag gpu1 --units PCBA:SREN-EVID:40 --resume runs/pcba-sren-evid-s0-e25/checkpoint_last.pt

A unit token is ``DATASET:ARM[:epochs[:max_dist[:relation_mode]]]``; ``--resume``
applies to the first unit only (the others train from scratch).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import esr_common as C
import esr_train as T


def parse_unit(token: str) -> dict[str, Any]:
    parts = token.split(":")
    if len(parts) < 2:
        raise ValueError(f"unit token {token!r} must be DATASET:ARM[:epochs[:max_dist[:relation_mode]]]")
    arm = "-".join(parts[1].split("+"))
    if not arm.startswith("SREN-") and arm not in ("BASE", "HIDDEN", "UNARY"):
        arm = f"SREN-{arm}"
    if arm not in C.ARMS and arm != C.FLEX:
        raise ValueError(f"unknown arm {arm!r}; known: {list(C.ARMS) + [C.FLEX]}")
    unit = {
        "dataset": parts[0],
        "arm": arm,
        "epochs": int(parts[2]) if len(parts) > 2 and parts[2] else None,
        "max_dist": int(parts[3]) if len(parts) > 3 and parts[3] else 2,
        "relation_mode": parts[4] if len(parts) > 4 and parts[4] else "real",
    }
    return unit


def unit_run_id(unit: dict[str, Any], tag: str = "") -> str:
    suffix = f"-d{unit['max_dist']}" if unit["max_dist"] != 2 else ""
    suffix += f"-{unit['relation_mode']}" if unit["relation_mode"] != "real" else ""
    base = f"{unit['dataset'].lower()}-{unit['arm'].lower()}-s0{suffix}"
    if unit["epochs"] is not None and unit["dataset"] == C.PCBA:
        base += f"-e{unit['epochs']}"
    return f"{base}-{tag}" if tag else base


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESR sequential GPU queue")
    ap.add_argument("--gpu-tag", required=True)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-root", default=str(C.RUNS))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-analysis-hours", type=float, default=1.0)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    root = Path(args.run_root)
    started = time.time()
    C.log(f"queue {args.gpu_tag}: {len(args.units)} unit(s) on {device}")
    for index, token in enumerate(args.units):
        unit = parse_unit(token)
        resumed = Path(args.resume) if (args.resume and index == 0) else None
        run_id = unit_run_id(unit, args.tag)
        out_dir = root / run_id
        C.log(f"queue {args.gpu_tag}: starting {run_id} (resume={resumed})")
        if unit["dataset"] == C.PCBA:
            T.train_pcba(
                arm=unit["arm"],
                seed=0,
                out_dir=out_dir,
                device=device,
                epochs=int(unit["epochs"] or C.DATASETS[C.PCBA]["epochs"]),
                resume=resumed,
                hidden_width=None,
                relation_mode=unit["relation_mode"],
                max_dist=int(unit["max_dist"]),
                tag=run_id,
            )
        else:
            T.train_small(
                dataset=unit["dataset"],
                arm=unit["arm"],
                seed=0,
                out_dir=out_dir,
                device=device,
                epochs=unit["epochs"],
                hidden_width=None,
                relation_mode=unit["relation_mode"],
                max_dist=int(unit["max_dist"]),
                tag=run_id,
            )
        C.log(f"queue {args.gpu_tag}: finished {run_id}; budget {C.budget_state()}")
    C.log(f"queue {args.gpu_tag}: all units done in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
