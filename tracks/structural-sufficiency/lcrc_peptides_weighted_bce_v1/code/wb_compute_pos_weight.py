"""Compute the frozen Train-only ``pos_weight`` vector and write the audit file.

Usage (CPU, repository root):

    uv run python tracks/.../code/wb_compute_pos_weight.py

The script reads only the official Train label array used by Fast v1.  It does
not open Validation labels, any Test artifact, any cache or any checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import wb_common as C
import wb_pos_weight as PW


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.POS_WEIGHT_JSON))
    args = parser.parse_args(argv)
    out = Path(args.out)
    y_train = np.load(C.TRAIN_LABELS)
    try:
        payload = PW.build_payload(y_train)
    except PW.PosWeightError as exc:
        C.log(f"POS_WEIGHT_FAILED: {exc}")
        return 2
    C.write_json(out, payload)
    C.log(f"wrote {C.track_rel(out)} with {len(payload['pos_weight'])} Train-only weights")
    for row in payload["labels"]:
        C.log(
            f"  {row['label_index']:>2} {row['label']:<24} pos={row['train_pos']:>5} "
            f"neg={row['train_neg']:>5} w={row['pos_weight']:.10f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
