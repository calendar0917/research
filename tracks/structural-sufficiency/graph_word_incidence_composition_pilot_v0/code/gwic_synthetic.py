"""Small synthetic typed graphs used by GWIC-v0 tests and checks."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import gwic_data as D


def make_view(name: str, n_atoms: int,
              bonds: Sequence[tuple[int, int, int]],
              atom_types: Sequence[int] | None = None,
              row: int = 0) -> dict[str, Any]:
    atom = np.zeros((n_atoms, 9), np.int64)
    if atom_types is None:
        atom_types = [1] * n_atoms
    atom[:, 0] = np.asarray(atom_types, np.int64)
    bond_feat = np.zeros((len(bonds), 3), np.int64)
    bu = np.zeros(len(bonds), np.int64)
    bv = np.zeros(len(bonds), np.int64)
    degree = np.zeros(n_atoms, np.int64)
    for i, (u, v, t) in enumerate(bonds):
        bu[i], bv[i] = int(u), int(v)
        bond_feat[i, 0] = int(t)
        degree[int(u)] += 1
        degree[int(v)] += 1
    # deterministic two-colouring-free ring flag: exact cycle rank > 0 means some
    # atoms are in a cycle; the flag is only descriptive and is not tested here.
    in_cycle = np.zeros(n_atoms, np.int64)
    n_edges = len(bonds)
    # find bridges by brute force for this tiny graph
    for e in range(n_edges):
        adj = [[] for _ in range(n_atoms)]
        for f, (u, v, _) in enumerate(bonds):
            if f == e:
                continue
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        seen = {0}
        stack = [0]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) == n_atoms:
            in_cycle[int(bu[e])] = 1
            in_cycle[int(bv[e])] = 1
    min_ring = in_cycle * 3
    view = {
        "row": int(row), "n_atoms": int(n_atoms), "n_bonds": int(n_edges),
        "atom_feat": atom, "bond_feat": bond_feat, "bond_u": bu, "bond_v": bv,
        "degree": degree, "min_ring": min_ring, "in_cycle": in_cycle,
        "smiles": name,
        "graph_n_atoms": int(n_atoms), "graph_n_bonds": int(n_edges),
        "graph_cycle_rank": int(max(0, n_edges - n_atoms + 1)),
        "graph_n_components": 1,
    }
    view["atom_scalars"] = D.atom_scalars(view)
    return view


def permute_view(view: Mapping[str, Any], perm: Sequence[int],
                 bond_order: Sequence[int] | None = None) -> dict[str, Any]:
    """Apply an atom permutation and an independent edge ordering."""
    n = int(view["n_atoms"])
    perm = np.asarray(perm, np.int64)
    inv = np.empty(n, np.int64)
    inv[perm] = np.arange(n)
    atom = np.asarray(view["atom_feat"], np.int64)[perm]
    bu = inv[np.asarray(view["bond_u"], np.int64)]
    bv = inv[np.asarray(view["bond_v"], np.int64)]
    bf = np.asarray(view["bond_feat"], np.int64)
    degree = np.asarray(view["degree"], np.int64)[perm]
    min_ring = np.asarray(view["min_ring"], np.int64)[perm]
    in_cycle = np.asarray(view["in_cycle"], np.int64)[perm]
    order = np.arange(len(bu)) if bond_order is None else np.asarray(bond_order, np.int64)
    out = {
        "row": int(view["row"]), "n_atoms": n, "n_bonds": int(len(bu)),
        "atom_feat": atom, "bond_feat": bf[order], "bond_u": bu[order],
        "bond_v": bv[order], "degree": degree, "min_ring": min_ring,
        "in_cycle": in_cycle, "smiles": str(view["smiles"]),
        "graph_n_atoms": int(view["graph_n_atoms"]),
        "graph_n_bonds": int(view["graph_n_bonds"]),
        "graph_cycle_rank": int(view["graph_cycle_rank"]),
        "graph_n_components": int(view["graph_n_components"]),
    }
    out["atom_scalars"] = D.atom_scalars(out)
    return out


def raw_from_views(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    node_ptr = [0]
    bond_ptr = [0]
    atom_parts, bond_parts = [], []
    for i, view in enumerate(views):
        view["row"] = i
        atom_parts.append(np.asarray(view["atom_feat"], np.int64))
        bond_parts.append(np.asarray(view["bond_feat"], np.int64))
        node_ptr.append(node_ptr[-1] + int(view["n_atoms"]))
        bond_ptr.append(bond_ptr[-1] + int(view["n_bonds"]))
    return {
        "node_ptr": np.asarray(node_ptr, np.int64),
        "bond_ptr": np.asarray(bond_ptr, np.int64),
        "atom_feat": np.concatenate(atom_parts, axis=0),
        "bond_feat": np.concatenate(bond_parts, axis=0),
        "smiles": np.asarray([str(v["smiles"]) for v in views]),
    }
