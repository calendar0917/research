"""Stage driver for the amide-dependency round (zero training).

Examples::

    # compute host (no RDKit needed): structure prep + frozen inference + analysis
    uv run python tracks/.../code/run.py --stages prepare,infer,analysis,pairs --device cuda:0

    # local host: everything, including the three structure figures
    uv run python tracks/.../code/run.py --stages prepare,infer,analysis,pairs,figures --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import common as C

STAGES = ("prepare", "infer", "analysis", "pairs", "figures")
DEFAULT_STAGES = "prepare,infer,analysis,pairs"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=DEFAULT_STAGES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")

    t0 = time.perf_counter()
    for stage in stages:
        C.log(f"=== stage {stage} ===")
        if stage == "prepare":
            import prepare

            rc = prepare.main()
        elif stage == "infer":
            import infer

            rc = infer.main(["--device", str(args.device), "--batch-size", str(args.batch_size)])
        elif stage == "analysis":
            import analysis

            rc = analysis.main()
        elif stage == "pairs":
            import pairs

            rc = pairs.main()
        else:
            try:
                import figures

                rc = figures.main()
            except ImportError as exc:  # RDKit / matplotlib are local-only
                C.log(f"figures skipped: {exc}")
                rc = 0
        if rc:
            C.log(f"stage {stage} returned {rc}; stopping")
            return int(rc)
    C.log(f"all stages done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
