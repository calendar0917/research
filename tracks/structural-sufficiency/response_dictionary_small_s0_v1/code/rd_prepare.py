"""Build the two content-addressed fragment caches (CPU only).

    uv run python .../rd_prepare.py            # both datasets
    uv run python .../rd_prepare.py --dataset ogbg-molesol

The caches store fragment tensors, per-molecule fixed views, official row ids,
raw labels and the Train scaler.  They are small, label-carrying derivatives of
the already committed official OGB caches; every consumer re-verifies the
SHA-256 recorded in the round manifest / report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_common as RC  # noqa: E402
import rd_fragment as RF  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=["ogbg-molesol", "ogbg-mollipo"])
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--max-fragment-atoms", type=int, default=8)
    args = parser.parse_args(argv)
    datasets = args.dataset or ["ogbg-molesol", "ogbg-mollipo"]
    manifest = {}
    for dataset in datasets:
        arrays = RF.prepare_dataset_cache(
            dataset, max_views=args.max_views, max_fragment_atoms=args.max_fragment_atoms
        )
        path = RF.save_dataset_cache(dataset, arrays)
        digest = RC.sha256_file(path)
        meta = {
            "path": str(path.relative_to(RC.REPO_ROOT)),
            "sha256": digest,
            "bytes": int(path.stat().st_size),
            "n_fragments": int(arrays["frag_n_atoms"].shape[0]),
            "canonical_hash_matches_verified": int(arrays["canonical_hash_matches_verified"][0]),
            "splits": {
                split: json.loads(str(arrays[f"{split}_meta_json"])) for split in ("train", "valid", "test")
            },
            "source_cache_sha256": str(arrays["cache_sha256"]),
            "split_sha256": json.loads(str(arrays["split_sha256"])),
            "scaler": json.loads(str(arrays["scaler_json"])),
        }
        manifest[dataset] = meta
        print(json.dumps({dataset: {k: v for k, v in meta.items() if k != "splits"}}, indent=2, sort_keys=True))
    RC.write_json(RC.DATA_DIR / "fragment_cache_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
