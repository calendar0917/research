"""Build the label-free BOSI relation cache for the frozen official-train rows.

The cache is a pure function of atom/bond connectivity and bond types.  It is
built once per machine and reused by the CPU gates and by training, so the
training step does no graph algorithm at all.

Usage:
    python tracks/.../zinc_bosi_architecture_v0/code/bosi_prep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402
import bosi_data as BD  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    cache = BC.load_train_cache()
    rel = BD.build_relation_cache(cache)
    BC.CACHE.mkdir(parents=True, exist_ok=True)
    out = BC.CACHE / "bosi_rel_train.pt"
    import torch

    torch.save(rel, out)
    dt = time.perf_counter() - t0

    n_rel = np.bincount(np.asarray(rel["partner_rel"]), minlength=BD.N_REL)
    n_half = int(len(rel["half_atom_raw"]))
    deg = np.asarray(rel["half_deg"])
    info = {
        "source_cache": str(BC.DATA_CACHE),
        "source_sha256": BC.sha256_file(BC.DATA_CACHE),
        "out": str(out),
        "out_sha256": BC.sha256_file(out),
        "content_sha256": BD.relation_content_hash(rel),
        "n_molecules": int(rel["n_molecules"]),
        "n_half_edges": n_half,
        "n_partners": int(len(rel["partner_rel"])),
        "build_seconds": float(dt),
        "max_atom_degree": int(deg.max()) if n_half else 0,
        "max_partners_per_half_edge": int(np.diff(np.asarray(rel["partner_ptr"])).max())
        if n_half else 0,
        "relation_counts": {name: int(c) for name, c in zip(BD.REL_NAMES, n_rel)},
        "relation_fractions": {name: float(c) / max(int(n_rel.sum()), 1)
                               for name, c in zip(BD.REL_NAMES, n_rel)},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    BC.write_json(BC.ANALYSIS / "prep.json", info)
    BC.log(f"[bosi-prep] {n_half} half-edges, {len(rel['partner_rel'])} partner rows, "
           f"max_deg={info['max_atom_degree']}, {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
