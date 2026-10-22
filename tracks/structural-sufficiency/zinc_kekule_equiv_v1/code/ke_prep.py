"""Build the verified valid-split Kekule representation artifact.

Run locally (RDKit is required).  No labels, no predictions, no test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ke_chem as KC  # noqa: E402
import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"


def build(split: str = "valid", max_reps: int = 8) -> tuple[list[dict], dict]:
    ds = KG.load_pyg(split)
    records: list[dict] = []
    status = Counter()
    reasons = Counter()
    for i in range(len(ds)):
        cats, edges, orig_orders, ei, ea = KG.graph_arrays(ds[i])
        res = KC.generate_representations(cats, edges, orig_orders, max_reps=max_reps)
        reps = list(res.reps) if res.reps else [orig_orders.copy()]
        status[res.status] += 1
        if not res.supported:
            reasons[res.reason] += 1
        records.append({
            "row": i,
            "n_atoms": int(len(cats)),
            "edges": edges,
            "orig_orders": orig_orders,
            "reps": reps,
            "status": res.status,
            "truncated": bool(res.truncated),
            "n_total": int(res.n_total),
            "aromatic_edge_set": set(res.aromatic_edges),
            "n_inferred_h": int(res.n_inferred_h),
            "graph_sha256": KG.graph_sha256(cats, ei, ea),
        })
    meta = {
        "split": split,
        "n_molecules": len(records),
        "max_reps": int(max_reps),
        "rdkit_version": KC.rdkit_version(),
        "status_counts": dict(status),
        "unsupported_reasons": dict(reasons),
        "n_aromatic_molecules": int(sum(1 for r in records if r["aromatic_edge_set"])),
        "n_truncated": int(sum(1 for r in records if r["truncated"])),
        "n_no_alternative": int(sum(1 for r in records if r["status"] == "no_alternative")),
        "n_unsupported": int(sum(1 for r in records if r["status"] == "unsupported")),
        "test_loaded": False,
    }
    return records, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="valid", choices=["valid"])
    p.add_argument("--max-reps", type=int, default=8)
    p.add_argument("--out", default=str(ANALYSIS / "valid_reps.npz"))
    args = p.parse_args(argv)
    records, meta = build(args.split, args.max_reps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    KG.save_reps_artifact(out, records, meta)
    (ANALYSIS / "prep_summary.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
