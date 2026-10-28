"""Bridge (separator) structures for SCE and label-free bridge statistics.

A bridge is a physical bond whose deletion increases the number of connected
components.  For every bridge e=(a,b) we store the two real connected sides
A_e (containing a) and B_e (containing b).  Atom ids are molecule-local; the
collate helper lifts them to batch-global ids.  No labels are read here.
"""

from __future__ import annotations

import numpy as np
import torch

import nl_common as N


def _bridges(n_atoms: int, bond_index: np.ndarray) -> list[int]:
    """Tarjan bridge finding on an undirected multigraph (edge ids kept)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for e, (u, v) in enumerate(np.asarray(bond_index, dtype=np.int64)):
        adj[int(u)].append((int(v), e))
        adj[int(v)].append((int(u), e))
    disc = [-1] * n_atoms
    low = [0] * n_atoms
    timer = [0]
    out: list[int] = []

    # Iterative DFS (no recursion limit dependence).
    for root in range(n_atoms):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer[0]
        timer[0] += 1
        stack = [(root, -1, 0)]
        while stack:
            v, parent_edge, pos = stack[-1]
            if pos < len(adj[v]):
                stack[-1] = (v, parent_edge, pos + 1)
                u, e = adj[v][pos]
                if e == parent_edge:
                    continue
                if disc[u] == -1:
                    disc[u] = low[u] = timer[0]
                    timer[0] += 1
                    stack.append((u, e, 0))
                else:
                    low[v] = min(low[v], disc[u])
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[v])
                    if low[v] > disc[p]:
                        out.append(parent_edge)
    return sorted(set(out))


def _side(n_atoms: int, bond_index: np.ndarray, edge: int, start: int) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for e, (u, v) in enumerate(np.asarray(bond_index, dtype=np.int64)):
        if e == edge:
            continue
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return sorted(seen)


def build_split(cache: dict) -> dict:
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    all_bonds = np.asarray(cache["bond_index"], dtype=np.int64)

    n_mol = len(atom_ptr) - 1
    bridge_ptr = [0]
    n_bridge = np.zeros(n_mol, dtype=np.int64)
    bond_local: list[int] = []
    side_a_atoms: list[int] = []
    side_b_atoms: list[int] = []
    side_a_size: list[int] = []
    side_b_size: list[int] = []
    ptr_a = [0]
    ptr_b = [0]
    max_bridge_degree = 0

    for g in range(n_mol):
        n = int(atom_ptr[g + 1] - atom_ptr[g])
        bi = all_bonds[bond_ptr[g]:bond_ptr[g + 1]]
        bridges = _bridges(n, bi)
        n_bridge[g] = len(bridges)
        for e in bridges:
            a, b = int(bi[e, 0]), int(bi[e, 1])
            A = _side(n, bi, e, a)
            B = _side(n, bi, e, b)
            assert len(A) + len(B) == n and len(A) > 0 and len(B) > 0
            bond_local.append(e)
            side_a_atoms.extend(A)
            side_b_atoms.extend(B)
            side_a_size.append(len(A))
            side_b_size.append(len(B))
            ptr_a.append(len(side_a_atoms))
            ptr_b.append(len(side_b_atoms))
        bridge_ptr.append(len(bond_local))
        if bridges:
            deg = np.zeros(n, dtype=np.int64)
            for e in bridges:
                deg[bi[e, 0]] += 1
                deg[bi[e, 1]] += 1
            max_bridge_degree = max(max_bridge_degree, int(deg.max()))

    assert len(ptr_a) == len(ptr_b) == len(bond_local) + 1
    return {
        "n_molecules": int(n_mol),
        "bridge_ptr": np.asarray(bridge_ptr, dtype=np.int64),
        "n_bridges": n_bridge,
        "bond_local": np.asarray(bond_local, dtype=np.int64),
        "ptr_a": np.asarray(ptr_a, dtype=np.int64),
        "ptr_b": np.asarray(ptr_b, dtype=np.int64),
        "side_a_atom": np.asarray(side_a_atoms, dtype=np.int64),
        "side_b_atom": np.asarray(side_b_atoms, dtype=np.int64),
        "side_a_size": np.asarray(side_a_size, dtype=np.int64),
        "side_b_size": np.asarray(side_b_size, dtype=np.int64),
        "max_bridge_degree": int(max_bridge_degree),
    }


def prep() -> dict:
    out = {"official_test_loaded": False, "splits": {}}
    for split in ("train", "valid"):
        cache = N.D.load_cache(split)
        data = build_split(cache)
        path = N.cache_path(f"bridge_{split}")
        N.ensure_dir(path.parent)
        torch.save(data, path)
        out["splits"][split] = {
            "path": N.L.track_rel(path),
            "sha256": N.L.C.sha256_file(path),
            "n_molecules": data["n_molecules"],
            "n_bridges_total": int(data["n_bridges"].sum()),
            "molecules_with_bridge": int((data["n_bridges"] > 0).sum()),
            "molecules_without_bridge": int((data["n_bridges"] == 0).sum()),
            "max_bridges_per_molecule": int(data["n_bridges"].max()),
            "max_bridge_degree": data["max_bridge_degree"],
            "side_a_size_median": int(np.median(data["side_a_size"])) if len(data["side_a_size"]) else 0,
        }
    return out


def load(split: str) -> dict:
    return torch.load(N.cache_path(f"bridge_{split}"), map_location="cpu", weights_only=False)


def collate(cache: dict, bridge: dict, rows) -> dict[str, torch.Tensor]:
    """Bridge tensors for a row batch; side atoms are lifted to global ids."""
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_counts = atom_ptr[rows + 1] - atom_ptr[rows]
    bond_counts = bond_ptr[rows + 1] - bond_ptr[rows]
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)

    bptr = np.asarray(bridge["bridge_ptr"], dtype=np.int64)
    counts = bptr[rows + 1] - bptr[rows]
    idx = np.concatenate([np.arange(bptr[r], bptr[r] + c) for r, c in zip(rows, counts)]) \
        if B else np.zeros(0, np.int64)
    bridge_graph = np.repeat(np.arange(B, dtype=np.int64), counts) if B else np.zeros(0, np.int64)

    ptr_a = np.asarray(bridge["ptr_a"], dtype=np.int64)
    ptr_b = np.asarray(bridge["ptr_b"], dtype=np.int64)
    seg_a = np.repeat(np.arange(len(idx), dtype=np.int64),
                      [ptr_a[i + 1] - ptr_a[i] for i in idx]) if len(idx) else np.zeros(0, np.int64)
    seg_b = np.repeat(np.arange(len(idx), dtype=np.int64),
                      [ptr_b[i + 1] - ptr_b[i] for i in idx]) if len(idx) else np.zeros(0, np.int64)
    loc_a = np.concatenate([np.arange(ptr_a[i], ptr_a[i + 1]) for i in idx]) \
        if len(idx) else np.zeros(0, np.int64)
    loc_b = np.concatenate([np.arange(ptr_b[i], ptr_b[i + 1]) for i in idx]) \
        if len(idx) else np.zeros(0, np.int64)
    side_a_atom = np.asarray(bridge["side_a_atom"], dtype=np.int64)[loc_a] + atom_off[bridge_graph[seg_a]]
    side_b_atom = np.asarray(bridge["side_b_atom"], dtype=np.int64)[loc_b] + atom_off[bridge_graph[seg_b]]
    sa_size = np.asarray(bridge["side_a_size"], dtype=np.int64)[idx]
    sb_size = np.asarray(bridge["side_b_size"], dtype=np.int64)[idx]
    bond_global = np.asarray(bridge["bond_local"], dtype=np.int64)[idx] + bond_off[bridge_graph]

    return {
        "n_bridges": int(len(idx)),
        "bridge_graph": torch.from_numpy(bridge_graph),
        "bridge_bond": torch.from_numpy(bond_global),
        "n_bridges_per_graph": torch.from_numpy(counts.astype(np.int64)),
        "side_a_atom": torch.from_numpy(side_a_atom),
        "side_b_atom": torch.from_numpy(side_b_atom),
        "side_a_seg": torch.from_numpy(seg_a),
        "side_b_seg": torch.from_numpy(seg_b),
        "side_a_w": torch.from_numpy((1.0 / sa_size.astype(np.float32))),
        "side_b_w": torch.from_numpy((1.0 / sb_size.astype(np.float32))),
        "side_a_size": torch.from_numpy(sa_size.astype(np.float32)),
        "side_b_size": torch.from_numpy(sb_size.astype(np.float32)),
    }


__all__ = ["build_split", "prep", "load", "collate", "_bridges", "_side"]
