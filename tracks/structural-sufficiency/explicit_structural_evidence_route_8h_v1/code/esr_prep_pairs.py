"""Build the label-free pair caches (CPU) and the frozen configs.

Run once per host before any GPU job: ``--datasets LIPO TOX21 ESOL PCBA``.
PCBA writes one cache for the full 437,929-graph array (~400 MB, git-ignored).
"""

from __future__ import annotations

import argparse
import time
from typing import Sequence

import esr_common as C
import esr_data as D
import esr_pairs as P


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESR pair-cache preparation")
    ap.add_argument("--datasets", nargs="+", default=["LIPO", "TOX21", "ESOL", "PCBA"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(C.ANALYSIS / "cache_manifest.json"))
    args = ap.parse_args(argv)
    report = {"round_id": C.ROUND_ID, "started_utc": C.utcnow(), "datasets": {}, "environment": C.environment()}
    for dataset in args.datasets:
        t0 = time.perf_counter()
        C.log(f"preparing pairs for {dataset}")
        view = D.load_view(dataset, include_test=False, max_dist=2, workers=int(args.workers))
        report["datasets"][dataset] = {
            "audit": view.audit(),
            "seconds": float(time.perf_counter() - t0),
            "pair_cache": {
                split: {
                    "path": C.track_rel(P.pair_cache_dir(dataset, split, 2)),
                    "meta": {
                        k: v
                        for k, v in view.pairs[split]["meta"].items()
                        if k in ("n_pairs", "n_pairs_per_molecule", "max_dist", "relation_dim", "sha256", "build_seconds")
                    },
                }
                for split in view.pairs
            },
        }
        C.log(f"{dataset}: pairs ready in {time.perf_counter() - t0:.1f}s")
    report["finished_utc"] = C.utcnow()
    C.write_json(C.Path(args.out), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
