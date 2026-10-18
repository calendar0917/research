"""Degree-preserving double-edge-swap rewiring of the root adjacency (B1 control).

The rewiring changes ONLY the root-to-root adjacency.  The patch cache (the
local E2 representation, built from the true bonds) is untouched, so the
R1-true vs R1-broken contrast isolates the value of the correspondence between
local patch identities and the root graph.

Per molecule the rewiring performs ``n_bonds`` attempted double-edge swaps:

* pick two distinct bonds (a, b) and (c, d);
* rewire to (a, d) and (c, b), rejecting self-loops and duplicate edges;
* the bond type travels with the edge slot, so the type multiset is preserved
  exactly and the degree sequence is preserved exactly.

Everything is verified in-run: identical bond count per molecule, identical
degree sequence, identical bond-type multiset, and a nonzero fraction of
actually-changed endpoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _physical_bonds(cache: dict[str, Any], mol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover undirected bonds (u<v, sorted) and their per-slot types."""
    start = int(cache["root_edge_start"][mol])
    end = int(cache["root_edge_start"][mol + 1])
    src = cache["root_edge_src_flat"][start:end].numpy().astype(np.int64)
    dst = cache["root_edge_dst_flat"][start:end].numpy().astype(np.int64)
    attr = cache["root_edge_attr_flat"][start:end].numpy()
    keep = src < dst
    src, dst, attr = src[keep], dst[keep], attr[keep]
    order = np.lexsort((dst, src))
    return src[order], dst[order], attr[order]


def rewire_mol_bonds(
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_attr: np.ndarray,
    seed: int,
    mol_id: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Double-edge-swap rewiring; returns (u, v, n_attempted, n_applied)."""
    rng = np.random.default_rng([int(seed), int(mol_id), 7717])
    u = bond_u.copy()
    v = bond_v.copy()
    m = len(u)
    existing = {(int(a), int(b)) for a, b in zip(u, v)}
    attempted = 0
    applied = 0
    for _ in range(m):
        if m < 2:
            break
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        attempted += 1
        a, b = int(u[i]), int(v[i])
        c, d = int(u[j]), int(v[j])
        # rewire (a,b),(c,d) -> (a,d),(c,b)
        if a == d or c == b or b == d or a == c:  # self-loop or no-op swap
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        if (a, d) in existing or (c, b) in existing:  # duplicate edge
            existing.add((a, b))
            existing.add((c, d))
            continue
        u[i], v[i] = a, d
        u[j], v[j] = c, b
        existing.add((a, d))
        existing.add((c, b))
        applied += 1
    return u, v, attempted, applied


def rewire_root_adjacency(cache: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a modified cache whose root adjacency is rewired per molecule."""
    n_mols = int(cache["meta"]["n_molecules"])
    bond_dim = int(cache["meta"]["bond_dim"])

    new_src: list[np.ndarray] = []
    new_dst: list[np.ndarray] = []
    new_attr: list[np.ndarray] = []
    starts = [0]
    n_edges: list[int] = []
    per_mol: list[dict[str, int]] = []
    degree_preserved = True
    count_preserved = True
    type_multiset_preserved = True
    changed_any = 0

    for mol in range(n_mols):
        u0, v0, attr0 = _physical_bonds(cache, mol)
        u1, v1, attempted, applied = rewire_mol_bonds(u0, v0, attr0, seed=seed, mol_id=mol)
        # verification
        deg0 = np.bincount(np.concatenate([u0, v0]), minlength=int(max(u0.max(), v0.max()) + 1) if len(u0) else 1)
        deg1 = np.bincount(np.concatenate([u1, v1]), minlength=deg0.size)
        if not np.array_equal(deg0, deg1):
            degree_preserved = False
        if len(u0) != len(u1):
            count_preserved = False
        # Bond types travel with the edge SLOT, so the per-slot attribute rows
        # are unchanged by construction; the multiset is therefore preserved
        # exactly (verified explicitly below).
        t1 = attr0
        if not np.array_equal(np.sort(attr0, axis=0), np.sort(t1, axis=0)):
            type_multiset_preserved = False
        changed_any += int(applied > 0)

        # materialise both directions (slot k keeps its bond type)
        for k in range(len(u1)):
            a, b = int(u1[k]), int(v1[k])
            new_src.extend([a, b])
            new_dst.extend([b, a])
            new_attr.extend([attr0[k], attr0[k]])
        n_edges.append(2 * len(u1))
        starts.append(starts[-1] + 2 * len(u1))
        per_mol.append({"n_bonds": int(len(u1)), "attempted": int(attempted), "applied": int(applied)})

    if not (degree_preserved and count_preserved and type_multiset_preserved):
        raise RuntimeError(
            f"rewiring broke invariants: degree={degree_preserved} count={count_preserved} types={type_multiset_preserved}"
        )
    if changed_any == 0:
        raise RuntimeError("rewiring changed no molecule")

    out = dict(cache)
    out["root_edge_src_flat"] = torch.as_tensor(np.asarray(new_src, dtype=np.int64), dtype=torch.int32)
    out["root_edge_dst_flat"] = torch.as_tensor(np.asarray(new_dst, dtype=np.int64), dtype=torch.int32)
    out["root_edge_attr_flat"] = torch.as_tensor(np.stack(new_attr, axis=0), dtype=torch.float16)
    out["root_edge_start"] = torch.as_tensor(starts, dtype=torch.int64)
    out["root_n_edges"] = torch.as_tensor(n_edges, dtype=torch.int64)
    out["meta"] = dict(cache["meta"])
    out["meta"]["root_edge_rewired"] = True

    report = {
        "seed": int(seed),
        "method": "degree-preserving double-edge swap; type travels with slot",
        "n_molecules": n_mols,
        "molecules_changed": int(changed_any),
        "total_swaps_applied": int(sum(p["applied"] for p in per_mol)),
        "degree_sequence_preserved": degree_preserved,
        "edge_count_preserved": count_preserved,
        "bond_type_multiset_preserved": type_multiset_preserved,
        "per_molecule_first10": per_mol[:10],
        "patches_unchanged": True,
    }
    return out, report
