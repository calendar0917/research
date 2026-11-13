"""Stage driver for the oracle task-relevance round.

Examples::

    # Stage A (local CPU): Part A clarifications + the two oracle label tables
    uv run python tracks/.../code/run.py --stages partA,oracle

    # Stage B (res, one GPU per task): train + forward CF evaluation
    uv run python tracks/.../code/run.py --stages train,evaluate --task wc_logp --device cuda:0
    uv run python tracks/.../code/run.py --stages train,evaluate --task molwt  --device cuda:1

    # Stage C (local CPU): statistics, figures
    uv run python tracks/.../code/run.py --stages analysis,figures

    # Stage D (local CPU, zero training / zero inference): the method + weighted
    # response material (this round's integration deliverable)
    uv run python tracks/.../code/run.py --stages weighted,method_figures
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import round_common as RC

STAGES = ("partA", "oracle", "train", "evaluate", "analysis", "figures", "weighted", "method_figures")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="partA,oracle")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--task", default="wc_logp", choices=list(RC.TASKS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    t0 = time.perf_counter()
    rc = 0
    for stage in stages:
        RC.log(f"=== stage {stage} ===")
        if stage == "partA":
            import part_a
            rc = part_a.main([])
        elif stage == "oracle":
            import prepare_oracle
            rc = prepare_oracle.main()
        elif stage == "train":
            import train_task
            rc = train_task.main(["--task", args.task, "--device", args.device] + (["--out-dir", args.out_dir] if args.out_dir else []))
        elif stage == "evaluate":
            import evaluate
            rc = evaluate.main(["--task", args.task, "--device", args.device, "--splits", args.splits] + (["--out-dir", args.out_dir] if args.out_dir else []))
        elif stage == "analysis":
            import analysis
            rc = analysis.main([])
        elif stage == "weighted":
            import weighted_response
            rc = weighted_response.main(["--splits", args.splits])
        elif stage == "method_figures":
            import method_figures
            rc = method_figures.main([])
        else:
            import figures
            rc = figures.main([])
        if rc not in (0, None):
            RC.log(f"stage {stage} returned {rc}; stopping")
            return int(rc)
    RC.log(f"run done in {time.perf_counter() - t0:.1f}s (stages={stages})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
