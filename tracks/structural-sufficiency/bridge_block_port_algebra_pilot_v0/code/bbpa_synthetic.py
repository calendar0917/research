"""Small typed graphs used by the engineering checks and unit tests."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def graph(n_atoms: int, edges: Sequence[tuple[int, int]], *,
          atom_types: Sequence[int] | None = None,
          bond_types: Sequence[int] | None = None) -> dict[str, Any]:
    n = int(n_atoms)
    bu = np.asarray([e[0] for e in edges], np.int64)
    bv = np.asarray([e[1] for e in edges], np.int64)
    if atom_types is None:
        atom_types = [0] * n
    if bond_types is None:
        bond_types = [0] * len(edges)
    if len(atom_types) != n or len(bond_types) != len(edges):
        raise ValueError("feature length mismatch")
    atom_feat = np.zeros((n, 9), np.int64)
    atom_feat[:, 0] = np.asarray(atom_types, np.int64)
    # a small deterministic variation in a second categorical field
    atom_feat[:, 1] = (np.arange(n) % 4).astype(np.int64)
    bond_feat = np.zeros((len(edges), 3), np.int64)
    bond_feat[:, 0] = np.asarray(bond_types, np.int64)
    bond_feat[:, 1] = (np.arange(len(edges)) % 3).astype(np.int64)
    degree = np.zeros(n, np.int64)
    for u, v in edges:
        degree[int(u)] += 1
        degree[int(v)] += 1
    return {
        "n_atoms": n, "n_bonds": len(edges),
        "atom_feat": atom_feat, "bond_feat": bond_feat,
        "bu": bu, "bv": bv, "degree": degree,
        "min_ring": np.zeros(n, np.int64), "in_cycle": (degree > 1).astype(np.int64),
        "smiles": "synthetic",
    }


def combine(graphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    atom_ptr = [0]
    bond_ptr = [0]
    atom_feat = []
    bond_feat = []
    bu = []
    bv = []
    degree = []
    min_ring = []
    in_cycle = []
    smiles = []
    for g in graphs:
        a0 = atom_ptr[-1]
        b0 = bond_ptr[-1]
        atom_feat.append(g["atom_feat"])
        bond_feat.append(g["bond_feat"])
        # the frozen cache stores molecule-local endpoints; keep that convention
        bu.append(g["bu"])
        bv.append(g["bv"])
        degree.append(g["degree"])
        min_ring.append(g["min_ring"])
        in_cycle.append(g["in_cycle"])
        smiles.append(g["smiles"])
        atom_ptr.append(a0 + g["n_atoms"])
        bond_ptr.append(b0 + g["n_bonds"])
    return {
        "node_ptr": np.asarray(atom_ptr, np.int64),
        "bond_ptr": np.asarray(bond_ptr, np.int64),
        "atom_feat": np.concatenate(atom_feat) if atom_feat else np.zeros((0, 9), np.int64),
        "bond_feat": np.concatenate(bond_feat) if bond_feat else np.zeros((0, 3), np.int64),
        "bond_u": np.concatenate(bu) if bu else np.zeros(0, np.int64),
        "bond_v": np.concatenate(bv) if bv else np.zeros(0, np.int64),
        "degree": np.concatenate(degree) if degree else np.zeros(0, np.int64),
        "min_ring": np.concatenate(min_ring) if min_ring else np.zeros(0, np.int64),
        "in_cycle": np.concatenate(in_cycle) if in_cycle else np.zeros(0, np.int64),
        "smiles": np.asarray(smiles),
        "row_ids": np.arange(len(graphs), dtype=np.int64),
        "train_idx": np.arange(len(graphs), dtype=np.int64),
        "valid_idx": np.zeros(0, np.int64),
        "test_idx": np.zeros(0, np.int64),
        "dataset": np.asarray("synthetic"), "cache_version": np.asarray("synthetic"),
        "graph_n_components": np.ones(len(graphs), np.int64),
        "graph_cycle_rank": np.zeros(len(graphs), np.int64),
        "graph_n_bonds": np.asarray([g["n_bonds"] for g in graphs], np.int64),
        "graph_n_atoms": np.asarray([g["n_atoms"] for g in graphs], np.int64),
    }


def benzene() -> dict[str, Any]:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return graph(6, edges, atom_types=[0] * 6, bond_types=[3] * 6)


def ethane() -> dict[str, Any]:
    return graph(2, [(0, 1)], atom_types=[0, 0], bond_types=[0])


def chain4() -> dict[str, Any]:
    return graph(4, [(0, 1), (1, 2), (2, 3)], atom_types=[0, 1, 0, 1],
                 bond_types=[0, 1, 0])


def star4() -> dict[str, Any]:
    return graph(4, [(0, 1), (0, 2), (0, 3)], atom_types=[0, 2, 0, 2],
                 bond_types=[0, 0, 1])


def disubstituted_benzene(second_position: int) -> dict[str, Any]:
    """Benzene (0..5) with substituent atoms 6 and 7 at positions 0 and p."""
    if second_position not in (1, 2, 3):
        raise ValueError("second_position must be 1, 2 or 3")
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (0, 6), (second_position, 7)]
    atom_types = [0] * 6 + [1, 2]
    bond_types = [3] * 6 + [0, 0]
    return graph(8, edges, atom_types=atom_types, bond_types=bond_types)


def concatenated_chain(blocks: int, atoms_per_block: int = 2) -> dict[str, Any]:
    """A block chain where each block is a ring-like cycle connected by bridges."""
    atom_types = []
    bond_types = []
    edges: list[tuple[int, int]] = []
    atom = 0
    block_atoms = []
    # each block is a triangle; blocks are connected by a single bridge bond
    for b in range(int(blocks)):
        a = atom
        edges.extend([(a, a + 1), (a + 1, a + 2), (a + 2, a)])
        atom_types.extend([b % 5, (b + 1) % 5, (b + 2) % 5])
        bond_types.extend([0, 1, 0])
        block_atoms.append(a)
        atom += 3
    for b in range(int(blocks) - 1):
        edges.append((block_atoms[b], block_atoms[b + 1]))
        bond_types.append(2)
    return graph(atom, edges, atom_types=atom_types, bond_types=bond_types)