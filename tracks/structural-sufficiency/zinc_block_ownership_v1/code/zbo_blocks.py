"""Deterministic biconnected-block ownership for ``zinc_block_ownership_v1``.

Every undirected physical bond is assigned to exactly one block:

* a bridge is a singleton block (``n_edges=1, n_atoms=2``);
* every other block is a maximal 2-connected component (a ring system).

The primary implementation is an edge-stack Tarjan algorithm.  A second,
independent NetworkX implementation is used as a reference in the CPU gates.
The historical ``zinc_arch_discovery_v1`` block cache is a third cross-check
only; its ``block_joint`` / ``block_dist`` / ``block_overlap`` features and its
``GlobalAttention`` model are *not* used anywhere in this round.

Content ownership is the only arm switch (see ``block_content``):

* ``R`` — the true per-block sum of ``c_e`` and ``c_e**2``;
* ``N`` — the same totals allocated to blocks by block edge count (global
  mean field);
* ``S`` — singleton blocks keep their true content; multi-edge blocks share a
  mean field that is conserved across the multi-edge blocks of the molecule.

All three arms keep identical block scalars, parameter shapes, initial state,
batch order and recipe.  Before the per-block mean normalisation, the summed
first and second moments are identical across arms by construction.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

N_BLOCK_SCALARS = 4
BLOCK_IN = 196


# ---------------------------------------------------------------------------
# independent biconnected-component algorithms
# ---------------------------------------------------------------------------


def tarjan_biconnected_edges(n_atoms: int, bond_index: np.ndarray) -> list[list[int]]:
    """Edge sets of the biconnected components (Tarjan edge stack).

    Bridges are returned as single-edge components.  Every edge appears exactly
    once.  Isolated atoms contribute no component.
    """
    edges = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(int(n_atoms))]
    for eid, (u, v) in enumerate(edges):
        u, v = int(u), int(v)
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    disc = [-1] * int(n_atoms)
    low = [0] * int(n_atoms)
    parent_edge = [-1] * int(n_atoms)
    stack: list[int] = []
    comps: list[list[int]] = []
    clock = 0

    def dfs(u: int) -> None:
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        for v, eid in adj[u]:
            if eid == parent_edge[u]:
                continue
            if disc[v] == -1:
                parent_edge[v] = eid
                stack.append(eid)
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    comp: list[int] = []
                    while True:
                        e = stack.pop()
                        comp.append(e)
                        if e == eid:
                            break
                    comps.append(comp)
            elif disc[v] < disc[u]:
                stack.append(eid)
                low[u] = min(low[u], disc[v])

    for u in range(int(n_atoms)):
        if disc[u] == -1:
            dfs(u)
    # In a well-formed DFS the stack is empty here.  Keep the guard so a
    # malformed graph can never silently drop an edge.
    if stack:
        comps.append(list(stack))
    return comps


def networkx_biconnected_edges(n_atoms: int, bond_index: np.ndarray) -> list[list[int]]:
    """Independent reference: NetworkX biconnected component edge sets."""
    import networkx as nx

    edges = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    g = nx.Graph()
    g.add_nodes_from(range(int(n_atoms)))
    g.add_edges_from((int(u), int(v)) for u, v in edges)
    key_to_bond = {(min(int(u), int(v)), max(int(u), int(v))): eid
                   for eid, (u, v) in enumerate(edges)}
    comps: list[list[int]] = []
    for comp in nx.biconnected_component_edges(g):
        ids = []
        for u, v in comp:
            key = (min(int(u), int(v)), max(int(u), int(v)))
            ids.append(key_to_bond[key])
        comps.append(sorted(ids))
    return comps


def tarjan_articulation_points(n_atoms: int, bond_index: np.ndarray) -> set[int]:
    """Cut vertices of the whole graph (independent of NetworkX)."""
    edges = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    disc = [-1] * int(n_atoms)
    low = [0] * int(n_atoms)
    parent = [-1] * int(n_atoms)
    points: set[int] = set()
    clock = 0

    def dfs(u: int) -> None:
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    points.add(u)
                if parent[u] != -1 and low[v] >= disc[u]:
                    points.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for u in range(int(n_atoms)):
        if disc[u] == -1:
            dfs(u)
    return points


def _partition_key(comps: Sequence[Sequence[int]]) -> list[frozenset]:
    return sorted((frozenset(int(e) for e in comp) for comp in comps),
                  key=lambda s: (min(s) if s else -1, len(s), tuple(sorted(s))))


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def build_molecule_blocks(n_atoms: int, bond_index: np.ndarray) -> dict:
    """Block arrays for one molecule, with independent cross-checks."""
    import networkx as nx

    edges = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    n_atoms = int(n_atoms)
    n_bonds = len(edges)

    comps = tarjan_biconnected_edges(n_atoms, edges)
    ref = networkx_biconnected_edges(n_atoms, edges)
    if _partition_key(comps) != _partition_key(ref):
        raise RuntimeError("Tarjan and NetworkX biconnected partitions disagree")

    # Deterministic block order; model output is order-invariant anyway.
    comps = sorted(comps, key=lambda c: (min(c), len(c), tuple(sorted(c))))
    bond_block = np.full(n_bonds, -1, dtype=np.int64)
    for bid, comp in enumerate(comps):
        for e in comp:
            if bond_block[int(e)] != -1:
                raise RuntimeError(f"bond {e} assigned to two blocks")
            bond_block[int(e)] = bid
    if n_bonds and (bond_block < 0).any():
        raise RuntimeError("some bonds were not assigned to a block")

    art_tarjan = tarjan_articulation_points(n_atoms, edges)
    g = nx.Graph()
    g.add_nodes_from(range(n_atoms))
    g.add_edges_from((int(u), int(v)) for u, v in edges)
    art_nx = {int(a) for a in nx.articulation_points(g)}
    if art_tarjan != art_nx:
        raise RuntimeError(f"articulation points disagree: {art_tarjan} vs {art_nx}")

    n_edges = np.zeros(len(comps), dtype=np.int64)
    n_atoms_b = np.zeros(len(comps), dtype=np.int64)
    n_artic = np.zeros(len(comps), dtype=np.int64)
    block_atoms: list[list[int]] = []
    for bid, comp in enumerate(comps):
        atoms = sorted({int(a) for e in comp for a in edges[int(e)]})
        block_atoms.append(atoms)
        n_edges[bid] = len(comp)
        n_atoms_b[bid] = len(atoms)
        n_artic[bid] = sum(1 for a in atoms if a in art_nx)
    rank = n_edges - n_atoms_b + 1
    if (rank < 0).any():
        raise RuntimeError("negative cyclomatic rank")

    return {
        "n_blocks": len(comps),
        "bond_block": bond_block,
        "block_atoms": block_atoms,
        "block_bonds": [sorted(int(e) for e in comp) for comp in comps],
        "n_edges": n_edges,
        "n_atoms": n_atoms_b,
        "rank": rank,
        "n_artic": n_artic,
        "articulation_points": sorted(art_nx),
    }


# ---------------------------------------------------------------------------
# whole split
# ---------------------------------------------------------------------------


def block_feature_matrix(n_edges: np.ndarray, n_atoms: np.ndarray,
                         rank: np.ndarray, n_artic: np.ndarray) -> np.ndarray:
    feats = np.stack([
        np.log1p(n_edges.astype(np.float64)),
        np.log1p(n_atoms.astype(np.float64)),
        rank.astype(np.float64),
        np.log1p(n_artic.astype(np.float64)),
    ], axis=1)
    return feats.astype(np.float32)


def content_hash(payload: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(payload):
        value = payload[key]
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(str(value).encode())
    return h.hexdigest()


def build_split_blocks(cache: Mapping[str, Any]) -> dict:
    """Build the flat block cache for a frozen split.

    Reads only ``atom_ptr``, ``bond_ptr`` and ``bond_index``.  Never touches
    labels, ``global_feat``, ``root_*`` or any target-derived field.
    """
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    n_mol = len(ap) - 1
    n_bonds_total = len(bi)

    block_ptr = np.zeros(n_mol + 1, dtype=np.int64)
    bond_block = np.full(n_bonds_total, -1, dtype=np.int64)
    n_edges_parts, n_atoms_parts, rank_parts, n_artic_parts = [], [], [], []
    sig_parts = []

    for i in range(n_mol):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        r = build_molecule_blocks(a1 - a0, bi[b0:b1])
        nb = int(r["n_blocks"])
        block_ptr[i + 1] = block_ptr[i] + nb
        bond_block[b0:b1] = r["bond_block"] + block_ptr[i]
        n_edges_parts.append(r["n_edges"])
        n_atoms_parts.append(r["n_atoms"])
        rank_parts.append(r["rank"])
        n_artic_parts.append(r["n_artic"])
        sig_parts.append(np.stack([r["n_edges"], r["n_atoms"],
                                   r["rank"], r["n_artic"]], axis=1))

    n_edges = (np.concatenate(n_edges_parts) if n_edges_parts
               else np.zeros(0, dtype=np.int64))
    n_atoms = (np.concatenate(n_atoms_parts) if n_atoms_parts
               else np.zeros(0, dtype=np.int64))
    rank = (np.concatenate(rank_parts) if rank_parts
            else np.zeros(0, dtype=np.int64))
    n_artic = (np.concatenate(n_artic_parts) if n_artic_parts
               else np.zeros(0, dtype=np.int64))
    sig = (np.concatenate(sig_parts) if sig_parts
           else np.zeros((0, 4), dtype=np.int64))
    feats = block_feature_matrix(n_edges, n_atoms, rank, n_artic)

    if n_bonds_total and (bond_block < 0).any():
        raise RuntimeError("global bond-to-block map has unassigned bonds")
    if len(np.unique(bond_block)) != len(n_edges):
        raise RuntimeError("bond-to-block map does not cover every block exactly")

    payload = {
        "block_ptr": block_ptr,
        "bond_block": bond_block,
        "block_n_edges": n_edges,
        "block_n_atoms": n_atoms,
        "block_rank": rank,
        "block_n_artic": n_artic,
        "block_sig": sig,
        "block_feats": feats,
        "n_molecules": n_mol,
        "n_blocks": len(n_edges),
        "n_bonds": n_bonds_total,
    }
    payload["content_sha256"] = content_hash(payload)
    return payload


def block_stats(blocks: Mapping[str, Any]) -> dict:
    """Label-free block statistics used by the Phase-0 stop gate."""
    bptr = np.asarray(blocks["block_ptr"], dtype=np.int64)
    ne = np.asarray(blocks["block_n_edges"], dtype=np.int64)
    n_mol = len(bptr) - 1
    sizes = ne
    n_multi = np.zeros(n_mol, dtype=np.int64)
    n_blocks = np.zeros(n_mol, dtype=np.int64)
    for i in range(n_mol):
        g0, g1 = int(bptr[i]), int(bptr[i + 1])
        s = sizes[g0:g1]
        n_blocks[i] = len(s)
        n_multi[i] = int((s >= 2).sum())
    hist = {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
    return {
        "n_molecules": int(n_mol),
        "n_blocks": int(len(sizes)),
        "n_singleton_blocks": int((sizes == 1).sum()),
        "n_multi_blocks": int((sizes >= 2).sum()),
        "singleton_block_fraction": float((sizes == 1).mean()) if len(sizes) else 0.0,
        "block_size_hist": hist,
        "mean_blocks_per_molecule": float(n_blocks.mean()) if n_mol else 0.0,
        "mean_multi_blocks_per_molecule": float(n_multi.mean()) if n_mol else 0.0,
        "molecules_ge1_multi": int((n_multi >= 1).sum()),
        "molecules_ge2_multi": int((n_multi >= 2).sum()),
        "fraction_molecules_ge1_multi": float((n_multi >= 1).mean()) if n_mol else 0.0,
        "fraction_molecules_ge2_multi": float((n_multi >= 2).mean()) if n_mol else 0.0,
        "max_block_edges": int(sizes.max()) if len(sizes) else 0,
    }


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def _ranges(ptr: np.ndarray, rows: np.ndarray):
    starts = np.asarray(ptr, dtype=np.int64)[rows]
    stops = np.asarray(ptr, dtype=np.int64)[rows + 1]
    counts = stops - starts
    idx = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(len(rows)), counts).astype(np.int64)
    offs = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1:
        offs[1:] = np.cumsum(counts)[:-1]
    return idx, batch, offs, counts


def collate(cache: Mapping[str, Any], rel: Mapping[str, Any],
            blocks: Mapping[str, Any], rows: Sequence[int]) -> dict:
    """BOSI relation batch plus block ownership tensors."""
    import torch

    import bosi_data as BD

    rows = np.asarray(rows, dtype=np.int64)
    base = BD.collate(cache, rel, rows)
    B = len(rows)

    bptr = np.asarray(blocks["block_ptr"], dtype=np.int64)
    block_idx, block_batch, block_off, block_counts = _ranges(bptr, rows)
    bond_idx, bond_batch, bond_off, bond_counts = _ranges(
        np.asarray(cache["bond_ptr"], dtype=np.int64), rows)

    bond_to_block = (np.asarray(blocks["bond_block"], dtype=np.int64)[bond_idx]
                     - bptr[rows][bond_batch] + block_off[bond_batch])
    if len(block_idx) and (bond_to_block < 0).any():
        raise RuntimeError("negative local block id in batch")
    if len(block_idx) and (bond_to_block >= len(block_idx)).any():
        raise RuntimeError("local block id out of range in batch")

    def T(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(x))
        return t.to(dtype) if dtype is not None else t

    base.update({
        "n_blocks": int(len(block_idx)),
        "block_rows": T(block_idx, torch.int64),
        "block_batch": T(block_batch, torch.int64),
        "block_feats": torch.from_numpy(
            np.asarray(blocks["block_feats"], dtype=np.float32)[block_idx]),
        "block_n_edges": torch.from_numpy(
            np.asarray(blocks["block_n_edges"], dtype=np.float32)[block_idx]),
        "bond_to_block": T(bond_to_block, torch.int64),
        "graph_n_bonds": torch.from_numpy(
            np.bincount(bond_batch, minlength=B).astype(np.float32)),
    })
    return base


# ---------------------------------------------------------------------------
# R / N / S content ownership
# ---------------------------------------------------------------------------


def block_content(c, batch: Mapping[str, Any], arm: str):
    """Per-block raw first/second moments before normalisation.

    ``c`` is ``[n_bonds, 96]``.  Returns ``(first, second)`` each
    ``[n_blocks, 96]``.  The per-molecule sums are identical for R, N and S.
    """
    import torch

    if arm not in ("R", "N", "S"):
        raise ValueError(f"unknown block arm {arm!r}")
    b2b = batch["bond_to_block"]
    bb = batch["block_batch"]
    bn = batch["block_n_edges"]
    B = int(batch["n_graphs"])
    n_blk = int(batch["n_blocks"])
    W = c.shape[1]

    first = c.new_zeros((n_blk, W)).index_add_(0, b2b, c)
    second = c.new_zeros((n_blk, W)).index_add_(0, b2b, c * c)

    if arm == "R":
        return first, second

    if arm == "N":
        s1 = c.new_zeros((B, W)).index_add_(0, batch["bond_batch"], c)
        s2 = c.new_zeros((B, W)).index_add_(0, batch["bond_batch"], c * c)
        e = c.new_zeros((B,)).index_add_(
            0, batch["bond_batch"], torch.ones_like(batch["bond_batch"], dtype=c.dtype))
        frac = bn / e[bb].clamp(min=1.0)
        return frac.unsqueeze(1) * s1[bb], frac.unsqueeze(1) * s2[bb]

    # S: singleton blocks keep true content; multi-edge blocks share a
    # conserved mean field among themselves.
    is_single = bn == 1.0
    multi_bond = bn[b2b] > 1.0
    bmask = batch["bond_batch"][multi_bond]
    s1m = c.new_zeros((B, W)).index_add_(0, bmask, c[multi_bond])
    s2m = c.new_zeros((B, W)).index_add_(0, bmask, c[multi_bond] * c[multi_bond])
    em = c.new_zeros((B,)).index_add_(
        0, bb, torch.where(bn > 1.0, bn, torch.zeros_like(bn)))
    frac_m = bn / em[bb].clamp(min=1.0)
    cf = torch.where(is_single.unsqueeze(1), first,
                     frac_m.unsqueeze(1) * s1m[bb])
    cs = torch.where(is_single.unsqueeze(1), second,
                     frac_m.unsqueeze(1) * s2m[bb])
    return cf, cs


def block_input(first, second, batch: Mapping[str, Any]):
    """``u_B`` in R^196 (per-block mean content + 4 structural scalars)."""
    import torch

    denom = batch["block_n_edges"].clamp(min=1.0).unsqueeze(1)
    return torch.cat(
        [first / denom, second / denom, batch["block_feats"]], dim=1)


def pooled_block_channel(phi, batch: Mapping[str, Any], out_dim: int = 32):
    """``sum_B phi(u_B)`` per molecule, shape ``[n_graphs, out_dim]``."""
    import torch

    B = int(batch["n_graphs"])
    h = phi.new_zeros((B, out_dim)).index_add_(0, batch["block_batch"], phi)
    return h
