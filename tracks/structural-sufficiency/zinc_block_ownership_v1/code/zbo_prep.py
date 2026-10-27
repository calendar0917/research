"""Build the label-free relation and block caches for this round.

Usage:
    python .../zbo_prep.py            # train + valid
    python .../zbo_prep.py --split train

Writes ``cache/zbo_rel_{split}.pt`` and ``cache/zbo_blocks_{split}.pt`` and
``analysis/prep.json``.  Only structural fields of the frozen official caches
are read; labels are never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zbo_common as BC  # noqa: E402  (also puts bosi_data on the path)
import bosi_data as BD  # noqa: E402
import zbo_blocks as ZB  # noqa: E402

KNOWN_REL_HASH = {
    "train": "c698a2f30e0fe9f2fc51e79f37156e81da56f44b6a47843293dddb308ceda2b7",
    "valid": "6505e23bdcbbaf42b64aa38f11af06474f76081d0d363cddd25ea6427c7cebbb",
}


def build_split(split: str) -> dict:
    t0 = time.perf_counter()
    cache = BC.load_cache(split)
    rel = BD.build_relation_cache(cache)
    rel_hash = BD.relation_content_hash(rel)
    rel_out = BC.rel_path(split)
    rel_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rel, rel_out)

    blocks = ZB.build_split_blocks(cache)
    block_out = BC.block_path(split)
    torch.save(blocks, block_out)
    stats = ZB.block_stats(blocks)

    record = {
        "split": split,
        "source_cache": BC.rel_or_abs(BC.DATA_TRAIN if split == "train" else BC.DATA_VALID),
        "source_sha256": BC.sha256_file(BC.DATA_TRAIN if split == "train" else BC.DATA_VALID),
        "relation_cache": BC.rel_or_abs(rel_out),
        "relation_file_sha256": BC.sha256_file(rel_out),
        "relation_content_sha256": rel_hash,
        "relation_known_hash": KNOWN_REL_HASH[split],
        "relation_hash_matches_known": rel_hash == KNOWN_REL_HASH[split],
        "block_cache": BC.rel_or_abs(block_out),
        "block_file_sha256": BC.sha256_file(block_out),
        "block_content_sha256": blocks["content_sha256"],
        "block_stats": stats,
        "official_valid_loaded": split == "valid",
        "official_test_loaded": False,
        "labels_read": False,
        "build_seconds": time.perf_counter() - t0,
    }
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "valid", "both"], default="both")
    args = p.parse_args(argv)
    splits = ["train", "valid"] if args.split == "both" else [args.split]
    out = {}
    for split in splits:
        rec = build_split(split)
        out[split] = rec
        BC.log(f"[prep] {split}: blocks={rec['block_stats']['n_blocks']} "
               f"multi={rec['block_stats']['n_multi_blocks']} "
               f"ge2_multi_frac={rec['block_stats']['fraction_molecules_ge2_multi']:.4f} "
               f"rel_hash_ok={rec['relation_hash_matches_known']}")
        if not rec["relation_hash_matches_known"]:
            raise RuntimeError(f"{split}: relation content hash drifted from the "
                               f"frozen BOSI/JM cache")
    BC.write_json(BC.ANALYSIS / "prep.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
