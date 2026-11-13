"""Build the shared, label-free CPU artefacts before any GPU job.

    1. frozen ``lipo_v1`` structure caches (this round's directory only),
    2. the two fixed structural vectors ``s3`` / ``s2`` with Train-only
       normalisation,
    3. the frozen ordinary batch order (shuffle generator 917000).

The three artefacts are shared by all arms.  Run this once per host before the
parallel training jobs; it never reads a target byte for the vectors and never
touches an upstream cache.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

import common as C
import data as D
import features as F
import schedule as S


def build(*, rebuild: bool = False) -> dict[str, Any]:
    cache_manifest = D.build_all_caches(rebuild=rebuild)
    caches = {split: D.load_cache(split) for split in C.SPLITS}
    C.log("[prepare] building U3/U2 vectors")
    features = F.build_features(caches)
    C.log("[prepare] building the frozen batch order")
    order = S.build_schedule()

    degenerate = {split: F.null_group_stats(caches[split], features[f"{split}_s3"]) for split in C.SPLITS}
    payload = {
        "schema": "lcrc-lipo-s3-locality-prepare-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "caches": cache_manifest,
        "features": {
            "path": C.track_rel(C.feature_path()),
            "sha256": C.sha256_file(C.feature_path()),
            "norm_s3": features["norm_s3"],
            "norm_s2": features["norm_s2"],
            "n_atoms_per_split": {
                split: int(np.asarray(features[f"{split}_atom_ptr"])[-1]) for split in C.SPLITS
            },
        },
        "batch_order": order["manifest"],
        "null3_degeneracy": degenerate,
        "target_norm": cache_manifest["target_norm"],
        "train_norm_fingerprint": C.PREP_DATA.norm_fingerprint(cache_manifest["train_norm"]),
        "note": "label-free; Train-only mean/std for both structural vectors",
    }
    C.write_json(C.ANALYSIS / "prepare_manifest.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build structure caches, U3/U2 vectors and batch order")
    ap.add_argument("--cache-only", action="store_true", help="build every shared CPU artefact and exit")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    payload = build(rebuild=args.rebuild)
    C.log(
        "[prepare] done: "
        f"n_atoms={payload['features']['n_atoms_per_split']} "
        f"s3_zero_var={payload['features']['norm_s3']['zero_variance_indices']} "
        f"s2_zero_var={payload['features']['norm_s2']['zero_variance_indices']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
