"""Build the label-free BOSI relation caches for the official train/valid caches.

Pure function of atom/bond connectivity and bond types (``bosi_data.relation_arrays``):
T (triangle closure) > Q (extra common neighbour) > R (same biconnected block,
>=3 edges) > N.  No label, no dataset statistic, no RNG.

Usage:
    python tracks/.../zinc_bosi_joint_moment_v1/code/bjm_prep.py [--splits train,valid]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bjm_common as BC  # noqa: E402
import bosi_data as BD  # noqa: E402

#: content hash recorded by zinc_bosi_architecture_v0 for the same 10k cache.
TRAIN_CONTENT_HASH_V0 = "c698a2f30e0fe9f2fc51e79f37156e81da56f44b6a47843293dddb308ceda2b7"


def build(split: str) -> dict:
    t0 = time.perf_counter()
    cache = BC.load_cache(split)
    rel = BD.build_relation_cache(cache)
    BC.CACHE.mkdir(parents=True, exist_ok=True)
    out = BC.rel_path(split)
    torch.save(rel, out)
    dt = time.perf_counter() - t0

    deg = np.asarray(rel["half_deg"])
    npart = np.diff(np.asarray(rel["partner_ptr"]))
    n_rel = np.bincount(np.asarray(rel["partner_rel"]), minlength=BD.N_REL)
    n_half = int(len(rel["half_atom_raw"]))
    content = BD.relation_content_hash(rel)
    info = {
        "split": split,
        "source_cache": BC.rel_or_abs({"train": BC.DATA_TRAIN,
                                       "valid": BC.DATA_VALID}[split]),
        "source_sha256": BC.sha256_file({"train": BC.DATA_TRAIN,
                                        "valid": BC.DATA_VALID}[split]),
        "out": BC.rel_or_abs(out),
        "out_sha256": BC.sha256_file(out),
        "content_sha256": content,
        "n_molecules": int(rel["n_molecules"]),
        "n_half_edges": n_half,
        "n_partners": int(len(rel["partner_rel"])),
        "build_seconds": float(dt),
        "max_atom_degree": int(deg.max()) if n_half else 0,
        "max_partners_per_half_edge": int(npart.max()) if n_half else 0,
        "partner_count_hist": [int(x) for x in np.bincount(npart,
                                                          minlength=int(npart.max()) + 1)],
        "relation_counts": {name: int(c) for name, c in zip(BD.REL_NAMES, n_rel)},
        "relation_fractions": {name: float(c) / max(int(n_rel.sum()), 1)
                               for name, c in zip(BD.REL_NAMES, n_rel)},
        "official_valid_loaded": split == "valid",
        "official_test_loaded": False,
        "built": BC.now(),
    }
    if split == "train":
        info["matches_bosi_v0_content_hash"] = content == TRAIN_CONTENT_HASH_V0
        if not info["matches_bosi_v0_content_hash"]:
            raise RuntimeError(
                f"train relation content hash {content} != BOSI v0 {TRAIN_CONTENT_HASH_V0}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", default="train,valid")
    args = p.parse_args(argv)
    out = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        info = build(split)
        out[split] = info
        BC.log(f"[bjm-prep] {split}: {info['n_half_edges']} half-edges, "
               f"{info['n_partners']} partner rows, max_deg={info['max_atom_degree']}, "
               f"max_partners={info['max_partners_per_half_edge']}, "
               f"content={info['content_sha256'][:8]}, {info['build_seconds']:.1f}s")
    BC.write_json(BC.ANALYSIS / "prep.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
