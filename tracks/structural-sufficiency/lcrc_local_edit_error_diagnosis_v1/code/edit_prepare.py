"""Build the four label-free structure caches and the Fit-only normalisation.

Run once per execution host before the six training units.  Nothing here reads a
target value except the Fit targets that define the target normalisation.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import edit_common as C
import edit_data as ED


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the local-edit structure caches")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    payload = ED.prepare_all(rebuild=bool(args.rebuild))
    C.log(
        "[prepare] feature_spec="
        f"{payload['feature_spec_hash'][:16]} partitions="
        + ", ".join(f"{k}:{v['n_molecules']}" for k, v in payload["partitions"].items())
    )
    tgt = payload["fit_norm"]["target"]
    C.log(f"[prepare] target mean={tgt['mean']:.6f} scale={tgt['scale']:.6f} n={tgt['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
