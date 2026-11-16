"""Small graph builders shared by the tests (no data files, no labels read)."""

from __future__ import annotations

import numpy as np

from sq_ast import PackedGraph


def make_packed(graphs, *, dataset="synthetic", split="train", labels=None) -> PackedGraph:
    """Build a packed split from ``[(atom_types, [(u, v), ...]), ...]``.

    Degrees, connected components and cycle rank are computed exactly; the
    min-ring bucket uses :func:`sq_data.min_ring_buckets`.
    """
    import sq_data as D

    atom_rows = []
    node_graph = []
    bu, bv, bf = [], [], []
    n_atoms_list, n_bonds_list, comps, ranks = [], [], [], []
    for graph_index, (types, edges) in enumerate(graphs):
        offset = len(atom_rows)
        for symbol in types:
            row = np.zeros(9, dtype=np.int16)
            row[0] = int(symbol)
            atom_rows.append(row)
            node_graph.append(graph_index)
        physical = sorted({(min(u, v), max(u, v)) for u, v in edges})
        for u, v in physical:
            bu.append(offset + u)
            bv.append(offset + v)
            bf.append(np.asarray([0, 0, 0], dtype=np.int16))
        n = len(types)
        n_atoms_list.append(n)
        n_bonds_list.append(len(physical))
        comps.append(D.connected_components(n, np.asarray([u for u, _ in physical]), np.asarray([v for _, v in physical])))
        ranks.append(len(physical) - n + comps[-1])
    n_nodes = len(atom_rows)
    atom_feat = np.stack(atom_rows) if atom_rows else np.zeros((0, 9), dtype=np.int16)
    u = np.asarray(bu, dtype=np.int32)
    v = np.asarray(bv, dtype=np.int32)
    degree = np.zeros(n_nodes, dtype=np.int16)
    for a, b in zip(bu, bv):
        degree[a] += 1
        degree[b] += 1
    rings, in_cycle = D.min_ring_buckets(n_nodes, u, v)
    return PackedGraph(
        dataset=dataset,
        split=split,
        n_graphs=len(graphs),
        n_nodes=n_nodes,
        node_graph=np.asarray(node_graph, dtype=np.int32),
        atom_feat=atom_feat,
        degree=degree,
        min_ring=rings.astype(np.int16),
        in_cycle=in_cycle,
        n_bonds=int(len(bu)),
        bond_u=u,
        bond_v=v,
        bond_feat=np.stack(bf) if bf else np.zeros((0, 3), dtype=np.int16),
        graph_n_atoms=np.asarray(n_atoms_list, dtype=np.int32),
        graph_n_bonds=np.asarray(n_bonds_list, dtype=np.int32),
        graph_n_components=np.asarray(comps, dtype=np.int32),
        graph_cycle_rank=np.asarray(ranks, dtype=np.int32),
        row_ids=np.arange(len(graphs), dtype=np.int64),
        labels=np.asarray(labels if labels is not None else np.zeros(len(graphs)), dtype=np.float32),
        smiles=[f"mol{i}" for i in range(len(graphs))],
    )


def random_tree_molecules(rng, n_molecules: int, *, min_atoms=6, max_atoms=14):
    """Random trees with atom types drawn from {6, 7, 8}."""
    graphs = []
    for _ in range(n_molecules):
        n = int(rng.integers(min_atoms, max_atoms + 1))
        types = rng.choice([6, 7, 8], size=n).tolist()
        edges = [(i, int(rng.integers(0, i))) for i in range(1, n)]
        graphs.append((types, edges))
    return graphs
