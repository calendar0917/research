"""Independent brute-force check of the Kekule matching enumerator.

For every molecule with at most 24 aromatic edges, count legal matchings up to
10 by a plain edge-wise recursion and compare with :func:`ke_chem.enumerate_kekule`.
This is a completeness check for the ``exhaustive`` / ``truncated`` labels.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ke_chem as KC  # noqa: E402
import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
LIMIT = 10
MAX_EDGES = 24


def brute_count(mol, limit: int = LIMIT) -> int:
    caps = KC.aromatic_capacities(mol)
    edges = []
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges.append((min(u, v), max(u, v)))
    edges.sort()
    atoms = sorted(caps)
    pos = {a: i for i, a in enumerate(atoms)}
    init = tuple(caps[a] for a in atoms)

    @lru_cache(maxsize=None)
    def rec(k: int, rem: tuple[int, ...]) -> int:
        if k == len(edges):
            return 1 if all(x == 0 for x in rem) else 0
        u, v = edges[k]
        total = rec(k + 1, rem)  # single
        if total >= limit:
            return limit
        if rem[pos[u]] > 0 and rem[pos[v]] > 0:
            lst = list(rem)
            lst[pos[u]] -= 1
            lst[pos[v]] -= 1
            total += rec(k + 1, tuple(lst))
        return min(total, limit)

    return rec(0, init)


def check(split: str) -> dict:
    ds = KG.load_pyg(split)
    n = 0
    skipped = 0
    mismatches = []
    max_edges = 0
    for row in range(len(ds)):
        cats, edges, orig, ei, ea = KG.graph_arrays(ds[row])
        mol = KC.build_mol_from_graph(cats, edges, orig)
        ar, caps, sols = KC.enumerate_kekule(mol, limit=LIMIT)
        if len(ar) > MAX_EDGES:
            skipped += 1
            continue
        n += 1
        max_edges = max(max_edges, len(ar))
        b = brute_count(mol, limit=LIMIT)
        if b != len(sols):
            mismatches.append({"row": row, "n_aromatic_edges": len(ar),
                               "brute": b, "enumerator": len(sols)})
    return {"split": split, "checked": n, "skipped_over_max_edges": skipped,
            "max_checked_edges": max_edges, "mismatches": len(mismatches),
            "mismatch_examples": mismatches[:8]}


def main() -> int:
    result = {
        "limit": LIMIT,
        "max_edges_checked": MAX_EDGES,
        "valid": check("valid"),
        "train": check("train"),
        "official_test_loaded": False,
    }
    (ANALYSIS / "enum_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
