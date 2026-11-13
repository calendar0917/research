"""Stage driver for the multi-motif mechanism round (zero training).

Examples::

    # Stage A — structure prep (label- and prediction-blind; needs RDKit)
    uv run python tracks/.../code/run.py --stages prepare --device cpu

    # Stage B — frozen inference (TraINED on gpu0, INIT on gpu1)
    uv run python tracks/.../code/run.py --stages infer --device cuda:0 \
        --tasks scopes_trained,internal_trained
    uv run python tracks/.../code/run.py --stages infer --device cuda:0 --tasks scopes_init

    # Stage C — local statistics + figures
    uv run python tracks/.../code/run.py --stages analysis,figures --device cpu
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import common as C

STAGES = ("prepare", "infer", "analysis", "figures")
TASKS = ("scopes_trained", "scopes_init", "internal_trained")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="prepare,infer,analysis")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tasks", default="scopes_trained,internal_trained")
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    t0 = time.perf_counter()
    rc = 0
    for stage in stages:
        C.log(f"=== stage {stage} ===")
        if stage == "prepare":
            import prepare

            rc = prepare.main()
        elif stage == "infer":
            import infer

            rc = infer.main(
                [
                    "--device", str(args.device),
                    "--batch-size", str(args.batch_size),
                    "--tasks", str(args.tasks),
                    "--splits", str(args.splits),
                ]
            )
        elif stage == "analysis":
            import analysis

            rc = analysis.main()
        else:
            import figures

            rc = figures.main()
        if rc not in (0, None):
            C.log(f"stage {stage} returned {rc}; stopping")
            return int(rc)
    C.log(f"run done in {time.perf_counter() - t0:.1f}s (stages={stages})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
