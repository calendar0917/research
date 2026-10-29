"""SPC static decomposition (task §3.1) -- deterministic, label-blind.

For every articulation vertex ``v`` the graph ``G - v`` is split into its
connected components ``C_vj``.  Each component (plus ``v`` and the bonds from
``v`` into it) is one rooted branch.  Branch summaries use only atom/bond
*content* sums (learned ``a_v``/``b_e``), branch sizes and the cyclomatic rank
of the induced subgraph on ``C_vj ∪ {v}``.

No learned state is ever fed back into another learned state: the whole
partition is a pure function of the raw graph, computed with ordinary graph
algorithms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import xs_common as C

CACHE_DIR = C.SPC_DIR / "cache"


# ---------------------------------------------------------------------------
# graph algorithms (deterministic; independent NetworkX reference in gates)
# ---------------------------------------------------------------------------


def _adjacency(n_atoms: int, bonds: Sequence[tuple[int, int, int]]):
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v, _t in bonds:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
    for a in adj:
        a.sort()
    return adj


def articulation_points(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> list[int]:
    """Tarjan cut vertices, DFS over sorted adjacency (fully deterministic)."""
    adj = _adjacency(n_atoms, bonds)
    disc = [-1] * n_atoms
    low = [0] * n_atoms
    parent = [-1] * n_atoms
    cut = set()
    timer = 0

    def dfs(u: int) -> None:
        nonlocal timer
        disc[u] = low[u] = timer
        timer += 1
        children = 0
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    cut.add(u)
                elif parent[u] != -1 and low[v] >= disc[u]:
                    cut.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for root in range(n_atoms):
        if disc[root] == -1:
            dfs(root)
    return sorted(cut)


def components_without(n_atoms: int, bonds: Sequence[tuple[int, int, int]], v: int) -> list[list[int]]:
    """Connected components of ``G - v``, ordered by their smallest node."""
    adj = _adjacency(n_atoms, bonds)
    seen = [False] * n_atoms
    seen[v] = True
    comps = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])
    return comps


def enumerate_molecule(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> dict:
    bonds = [(int(u), int(v), int(t)) for u, v, t in bonds]
    adj = _adjacency(n_atoms, bonds)
    bond_of = {}
    for e, (u, v, _t) in enumerate(bonds):
        bond_of[(min(u, v), max(u, v))] = e
    incident = [[] for _ in range(n_atoms)]
    for e, (u, v, _t) in enumerate(bonds):
        incident[u].append(e)
        incident[v].append(e)
    for a in incident:
        a.sort()

    cuts = articulation_points(n_atoms, bonds)
    cv_atom: list[int] = []
    br_cv: list[int] = []
    br_natoms: list[int] = []
    br_nbonds: list[int] = []
    br_cyc: list[int] = []
    br_atom: list[list[int]] = []
    br_bond: list[list[int]] = []
    cv_npairs: list[int] = []
    pair_br: list[tuple[int, int]] = []
    pair_cv: list[int] = []

    for v in cuts:
        cvi = len(cv_atom)
        cv_atom.append(v)
        comps = components_without(n_atoms, bonds, v)
        branch_ids = []
        for comp in comps:
            comp_set = set(comp)
            internal = [bond_of[(min(u, w), max(u, w))]
                        for (u, w, _t) in bonds
                        if u in comp_set and w in comp_set]
            connectors = [e for e in incident[v]
                          if (bonds[e][0] if bonds[e][1] == v else bonds[e][1]) in comp_set]
            all_bonds = sorted(set(internal) | set(connectors))
            n_v = len(comp) + 1
            n_e = len(all_bonds)
            cyc = n_e - n_v + 1
            assert cyc >= 0, (v, comp, cyc)
            branch_ids.append(len(br_cv))
            br_cv.append(cvi)
            br_natoms.append(len(comp))
            br_nbonds.append(len(all_bonds))
            br_cyc.append(int(cyc))
            br_atom.append(sorted(comp))
            br_bond.append(all_bonds)
        k = len(comps)
        cv_npairs.append(k * (k - 1) // 2)
        for ii in range(k):
            for jj in range(ii + 1, k):
                pair_br.append((branch_ids[ii], branch_ids[jj]))
                pair_cv.append(cvi)

    br_atom_ptr = [0]
    br_bond_ptr = [0]
    for atoms in br_atom:
        br_atom_ptr.append(br_atom_ptr[-1] + len(atoms))
    for bd in br_bond:
        br_bond_ptr.append(br_bond_ptr[-1] + len(bd))

    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(len(bonds)),
        "cv_atom": np.asarray(cv_atom, dtype=np.int64),
        "cv_npairs": np.asarray(cv_npairs, dtype=np.int64),
        "br_cv": np.asarray(br_cv, dtype=np.int64),
        "br_natoms": np.asarray(br_natoms, dtype=np.int64),
        "br_nbonds": np.asarray(br_nbonds, dtype=np.int64),
        "br_cyc": np.asarray(br_cyc, dtype=np.int64),
        "br_atom_ptr": np.asarray(br_atom_ptr, dtype=np.int64),
        "br_atom_idx": np.asarray([x for a in br_atom for x in a], dtype=np.int64),
        "br_bond_ptr": np.asarray(br_bond_ptr, dtype=np.int64),
        "br_bond_idx": np.asarray([x for b in br_bond for x in b], dtype=np.int64),
        "pair_br": np.asarray(pair_br, dtype=np.int64).reshape(-1, 2),
        "pair_cv": np.asarray(pair_cv, dtype=np.int64),
    }


def networkx_reference(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> dict:
    """Independent reference used only by the CPU gates (NetworkX)."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(int(n_atoms)))
    for u, v, _t in bonds:
        g.add_edge(int(u), int(v))
    cuts = sorted(nx.articulation_points(g))
    out = {}
    for v in cuts:
        h = g.copy()
        h.remove_node(v)
        comps = [sorted(int(x) for x in c) for c in nx.connected_components(h)]
        comps.sort(key=lambda c: c[0])
        out[v] = comps
    return {"cut_vertices": cuts, "components": out}


# ---------------------------------------------------------------------------
# whole split
# ---------------------------------------------------------------------------


def _molecules(cache: Mapping[str, Any]):
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = []
    for m in range(len(ap) - 1):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        out.append((a1 - a0, [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e]))
                              for e in range(b0, b1)]))
    return out


def build_split(cache: Mapping[str, Any], name: str) -> dict:
    mols = _molecules(cache)
    cv_ptr, br_ptr, pair_ptr = [0], [0], [0]
    cv_atom, cv_npairs = [], []
    br_cv, br_natoms, br_nbonds, br_cyc = [], [], [], []
    br_atom_ptr, br_atom_idx = [0], []
    br_bond_ptr, br_bond_idx = [0], []
    pair_br, pair_cv = [], []
    for n_atoms, bonds in mols:
        obj = enumerate_molecule(n_atoms, bonds)
        nb = len(obj["br_cv"])
        ncv = len(obj["cv_atom"])
        cv_atom.append(obj["cv_atom"])
        cv_npairs.append(obj["cv_npairs"])
        br_cv.append(obj["br_cv"] + cv_ptr[-1])
        br_natoms.append(obj["br_natoms"])
        br_nbonds.append(obj["br_nbonds"])
        br_cyc.append(obj["br_cyc"])
        br_atom_ptr.extend((obj["br_atom_ptr"][1:] + br_atom_ptr[-1]).tolist())
        br_atom_idx.append(obj["br_atom_idx"])
        br_bond_ptr.extend((obj["br_bond_ptr"][1:] + br_bond_ptr[-1]).tolist())
        br_bond_idx.append(obj["br_bond_idx"])
        pair_br.append(obj["pair_br"] + br_ptr[-1])
        pair_cv.append(obj["pair_cv"] + cv_ptr[-1])
        cv_ptr.append(ncv + cv_ptr[-1])
        br_ptr.append(nb + br_ptr[-1])
        pair_ptr.append(len(obj["pair_cv"]) + pair_ptr[-1])

    def cat(seq, nd, dtype=np.int64):
        seq = [np.asarray(x) for x in seq]
        seq = [x.reshape(-1, nd) if nd > 1 else x.reshape(-1) for x in seq]
        if not seq:
            return np.zeros((0, nd) if nd > 1 else (0,), dtype=dtype)
        return np.concatenate(seq, 0).astype(dtype)

    return {
        "version": C.STRUCTURE_VERSION,
        "family": "spc",
        "split": name,
        "n_molecules": len(mols),
        "cv_ptr": np.asarray(cv_ptr, dtype=np.int64),
        "br_ptr": np.asarray(br_ptr, dtype=np.int64),
        "pair_ptr": np.asarray(pair_ptr, dtype=np.int64),
        "cv_atom": cat(cv_atom, 1),
        "cv_npairs": cat(cv_npairs, 1),
        "br_cv": cat(br_cv, 1),
        "br_natoms": cat(br_natoms, 1),
        "br_nbonds": cat(br_nbonds, 1),
        "br_cyc": cat(br_cyc, 1),
        "br_atom_ptr": np.asarray(br_atom_ptr, dtype=np.int64),
        "br_atom_idx": cat(br_atom_idx, 1),
        "br_bond_ptr": np.asarray(br_bond_ptr, dtype=np.int64),
        "br_bond_idx": cat(br_bond_idx, 1),
        "pair_br": cat(pair_br, 2),
        "pair_cv": cat(pair_cv, 1),
    }


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"spc_{split}.pt"


def load_split(split: str, *, cache: Mapping[str, Any] | None = None, build: bool = True) -> dict:
    import torch

    path = cache_path(split)
    if path.exists():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("version") == C.STRUCTURE_VERSION:
            return obj
    if not build:
        raise FileNotFoundError(f"missing SPC structure cache {path}")
    if cache is None:
        cache = C.load_cache(split)
    obj = build_split(cache, split)
    C.ensure_dir(CACHE_DIR)
    C.atomic_torch_save(obj, path)
    return obj


def coverage_stats(obj: Mapping[str, Any]) -> dict:
    cv_ptr = np.asarray(obj["cv_ptr"])
    br_ptr = np.asarray(obj["br_ptr"])
    pair_ptr = np.asarray(obj["pair_ptr"])
    cv_per_mol = np.diff(cv_ptr)
    br_per_mol = np.diff(br_ptr)
    # branches per cut vertex, from br_cv counts
    br_cv = np.asarray(obj["br_cv"])
    k = np.bincount(br_cv, minlength=max(1, len(np.asarray(obj["cv_atom"]))))
    return {
        "n_molecules": int(len(cv_per_mol)),
        "frac_molecules_with_cut_vertex": float((cv_per_mol > 0).mean()),
        "cut_vertices_per_molecule": _describe(cv_per_mol),
        "branches_per_molecule": _describe(br_per_mol),
        "branches_per_cut_vertex": _describe(k),
        "pairs_per_molecule": _describe(np.diff(pair_ptr)),
        "total_cut_vertices": int(cv_per_mol.sum()),
        "total_branches": int(br_per_mol.sum()),
        "total_pairs": int(np.diff(pair_ptr).sum()),
        "max_branches_per_cut_vertex": int(k.max()) if k.size else 0,
        "max_pairs_per_cut_vertex": int(np.max(np.asarray(obj["cv_npairs"]))) if len(obj["cv_npairs"]) else 0,
    }


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)), "max": float(x.max()),
    }


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------


def collate(obj: Mapping[str, Any], base: Mapping[str, Any], rows: np.ndarray) -> dict:
    import torch
    from xs_data import segments

    rows = np.asarray(rows, dtype=np.int64)
    dev = base["atom_feat"].device

    cv_idx, cv_mol, _ = segments(obj["cv_ptr"], rows)
    br_idx, br_mol, _ = segments(obj["br_ptr"], rows)
    pair_idx, pair_mol, _ = segments(obj["pair_ptr"], rows)

    def t(x, dtype=torch.int64):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=dev)

    atom_off = np.asarray(base["atom_off"], dtype=np.int64)
    bond_off = np.asarray(base["bond_off"], dtype=np.int64)

    cv_atom = t(np.asarray(obj["cv_atom"])[cv_idx]) + t(atom_off[cv_mol])
    cv_npairs = t(np.asarray(obj["cv_npairs"])[cv_idx], torch.float32)
    cv_batch = t(cv_mol)

    br_cv_local = np.asarray(obj["br_cv"])[br_idx]
    # batch-local cut-vertex row for every branch
    cv_slot_of_mol = np.zeros(len(rows), dtype=np.int64)
    cv_slot_of_mol[1:] = np.cumsum(np.diff(np.asarray(obj["cv_ptr"]))[rows])[:-1]
    br_cv = t(br_cv_local - np.asarray(obj["cv_ptr"])[rows][br_mol] + cv_slot_of_mol[br_mol])

    br_natoms = np.asarray(obj["br_natoms"])[br_idx]
    br_nbonds = np.asarray(obj["br_nbonds"])[br_idx]
    br_cyc = np.asarray(obj["br_cyc"])[br_idx]
    br_atom_w = 1.0 / np.sqrt(1.0 + np.asarray(obj["br_natoms"])[br_idx])
    br_bond_w = 1.0 / np.sqrt(1.0 + np.asarray(obj["br_nbonds"])[br_idx])
    if np.any(br_natoms <= 0) or np.any(br_nbonds <= 0):
        raise RuntimeError("SPC branch without atoms or bonds")

    # branch -> atom incidences
    a_flat, a_owner, _ = segments(obj["br_atom_ptr"], br_idx)
    atom_local = np.asarray(obj["br_atom_idx"])[a_flat]
    a_mol = br_mol[a_owner]
    atom_rows = atom_off[a_mol] + atom_local
    a_cnt = np.diff(np.asarray(obj["br_atom_ptr"]))[br_idx]
    a_ptr = np.concatenate([[0], np.cumsum(a_cnt)]).astype(np.int64)
    # branch -> bond incidences
    b_flat, b_owner, _ = segments(obj["br_bond_ptr"], br_idx)
    bond_local = np.asarray(obj["br_bond_idx"])[b_flat]
    b_mol = br_mol[b_owner]
    bond_rows = bond_off[b_mol] + bond_local
    b_cnt = np.diff(np.asarray(obj["br_bond_ptr"]))[br_idx]
    b_ptr = np.concatenate([[0], np.cumsum(b_cnt)]).astype(np.int64)

    pair_br_local = np.asarray(obj["pair_br"])[pair_idx]
    br_slot_of_mol = np.zeros(len(rows), dtype=np.int64)
    br_slot_of_mol[1:] = np.cumsum(np.diff(np.asarray(obj["br_ptr"]))[rows])[:-1]
    pair_br = t(pair_br_local - np.asarray(obj["br_ptr"])[rows][pair_mol][:, None]
                + br_slot_of_mol[pair_mol][:, None])
    pair_cv_local = np.asarray(obj["pair_cv"])[pair_idx]
    pair_cv = t(pair_cv_local - np.asarray(obj["cv_ptr"])[rows][pair_mol] + cv_slot_of_mol[pair_mol])

    return {
        "spc_cv_atom": cv_atom,
        "spc_cv_batch": cv_batch,
        "spc_cv_npairs": cv_npairs,
        "spc_br_cv": br_cv,
        "spc_br_atom_ptr": t(a_ptr),
        "spc_br_atom_idx": t(atom_rows),
        "spc_br_atom_w": t(br_atom_w[a_owner], torch.float32),
        "spc_br_bond_ptr": t(b_ptr),
        "spc_br_bond_idx": t(bond_rows),
        "spc_br_bond_w": t(br_bond_w[b_owner], torch.float32),
        "spc_br_natoms": t(br_natoms, torch.float32),
        "spc_br_nbonds": t(br_nbonds, torch.float32),
        "spc_br_cyc": t(br_cyc, torch.float32),
        "spc_pair_br": pair_br,
        "spc_pair_cv": pair_cv,
    }
