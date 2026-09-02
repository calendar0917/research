"""Ring-cell lifting used by the MolHIV CIN--Beam8 incremental audit.

The lift follows the molecular setup of CIN/CWN: atoms are 0-cells,
undirected bonds are 1-cells, and chordless cycles of length at most six are
2-cells.  It augments the existing Beam8 ``HeteroData`` object so that the
same molecule, split and categorical chemistry features are used by both
branches.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .graph import Graph
from .molhiv_beam8_incidence import MolhivBeam8Incidence, to_heterodata
from .molhiv_ring_cells import chordless_cycles


def _nonidentity_permutation(size: int, seed: int) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(size)
    if np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def add_ring_complex(
    data: Any,
    graph: Graph,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    *,
    graph_index: int,
    max_ring_size: int = 6,
    shuffle_seed: int = 20260815,
) -> Any:
    """Add atom--bond--ring cell relations to a PyG ``HeteroData`` object."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runner dependency
        raise RuntimeError(f"PyTorch is required for the CIN lift: {exc}") from exc

    bonds = tuple(sorted(graph.edges()))
    bond_lookup = {edge: index for index, edge in enumerate(bonds)}
    if set(bond_lookup) != set(edge_features):
        raise ValueError("graph bonds and categorical edge features disagree")
    if bonds:
        bond_x = np.stack([np.asarray(edge_features[edge], dtype=np.int64) for edge in bonds])
    else:
        width = len(next(iter(edge_features.values()))) if edge_features else 3
        bond_x = np.empty((0, width), dtype=np.int64)

    atom_boundary_source: list[int] = []
    atom_boundary_target: list[int] = []
    atom_upper_source: list[int] = []
    atom_upper_target: list[int] = []
    atom_upper_shared: list[int] = []
    for bond_index, (left, right) in enumerate(bonds):
        atom_boundary_source.extend([left, right])
        atom_boundary_target.extend([bond_index, bond_index])
        atom_upper_source.extend([left, right])
        atom_upper_target.extend([right, left])
        atom_upper_shared.extend([bond_index, bond_index])

    rings = tuple(chordless_cycles(graph, min_size=3, max_size=max_ring_size))
    bond_ring_source: list[int] = []
    bond_ring_target: list[int] = []
    bond_upper_source: list[int] = []
    bond_upper_target: list[int] = []
    bond_upper_shared: list[int] = []
    for ring_index, ring in enumerate(rings):
        boundary: list[int] = []
        for offset, left in enumerate(ring):
            right = ring[(offset + 1) % len(ring)]
            edge = (left, right) if left < right else (right, left)
            if edge not in bond_lookup:
                raise ValueError("ring boundary is missing from the graph bonds")
            boundary.append(bond_lookup[edge])
        if len(set(boundary)) != len(ring):
            raise ValueError("ring boundary must contain one distinct bond per atom")
        for bond_index in boundary:
            bond_ring_source.append(bond_index)
            bond_ring_target.append(ring_index)
        for source in boundary:
            for target in boundary:
                if source == target:
                    continue
                bond_upper_source.append(source)
                bond_upper_target.append(target)
                bond_upper_shared.append(ring_index)

    ring_permutation = _nonidentity_permutation(
        len(rings), shuffle_seed ^ ((int(graph_index) + 1) * 0x9E3779B1)
    )
    ring_shuffle_source = ring_permutation
    ring_shuffle_target = np.arange(len(rings), dtype=np.int64)

    data["bond_cell"].x = torch.tensor(bond_x, dtype=torch.long)
    data["bond_cell"].num_nodes = len(bonds)
    data["ring"].size = torch.tensor([len(ring) for ring in rings], dtype=torch.long)
    data["ring"].num_nodes = len(rings)

    data["atom", "cell_boundary", "bond_cell"].edge_index = torch.tensor(
        [atom_boundary_source, atom_boundary_target], dtype=torch.long
    ).reshape(2, -1)
    data["bond_cell", "cell_boundary", "ring"].edge_index = torch.tensor(
        [bond_ring_source, bond_ring_target], dtype=torch.long
    ).reshape(2, -1)
    data["atom", "cell_upper", "atom"].edge_index = torch.tensor(
        [atom_upper_source, atom_upper_target], dtype=torch.long
    ).reshape(2, -1)
    data["atom", "cell_upper", "atom"].shared = torch.tensor(
        atom_upper_shared, dtype=torch.long
    )
    data["bond_cell", "cell_upper", "bond_cell"].edge_index = torch.tensor(
        [bond_upper_source, bond_upper_target], dtype=torch.long
    ).reshape(2, -1)
    data["bond_cell", "cell_upper", "bond_cell"].shared = torch.tensor(
        bond_upper_shared, dtype=torch.long
    )
    data["ring", "token_shuffle", "ring"].edge_index = torch.tensor(
        np.stack([ring_shuffle_source, ring_shuffle_target]), dtype=torch.long
    )

    endpoint_store = data["patch", "bond_endpoint", "atom"]
    endpoint_atoms = (
        endpoint_store.edge_index[1].detach().cpu().numpy()
        if "edge_index" in endpoint_store
        else np.empty(0, dtype=np.int64)
    )
    if len(endpoint_atoms) % 2:
        raise ValueError("patch bond endpoints must occur in pairs")
    endpoint_bonds: list[int] = []
    for offset in range(0, len(endpoint_atoms), 2):
        left = int(endpoint_atoms[offset])
        right = int(endpoint_atoms[offset + 1])
        edge = (left, right) if left < right else (right, left)
        if edge not in bond_lookup:
            raise ValueError("patch endpoint pair is missing from the bond-cell lift")
        endpoint_bonds.extend([bond_lookup[edge], bond_lookup[edge]])
    endpoint_store.bond_cell = torch.tensor(endpoint_bonds, dtype=torch.long)
    return data


def to_cin_beam8_heterodata(
    item: MolhivBeam8Incidence,
    graph: Graph,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    *,
    max_ring_size: int = 6,
    shuffle_seed: int = 20260815,
) -> Any:
    data = to_heterodata(item)
    return add_ring_complex(
        data,
        graph,
        edge_features,
        graph_index=item.graph_index,
        max_ring_size=max_ring_size,
        shuffle_seed=shuffle_seed,
    )


def ring_complex_summary(items: Sequence[Any]) -> dict[str, float | int]:
    if not items:
        raise ValueError("cannot summarize an empty ring-complex collection")
    bonds = np.asarray([int(item["bond_cell"].num_nodes) for item in items], dtype=float)
    rings = np.asarray([int(item["ring"].num_nodes) for item in items], dtype=float)
    boundary = np.asarray(
        [int(item["bond_cell", "cell_boundary", "ring"].edge_index.shape[1]) for item in items],
        dtype=float,
    )
    return {
        "graphs": len(items),
        "mean_bond_cells": float(bonds.mean()),
        "mean_ring_cells": float(rings.mean()),
        "max_ring_cells": int(rings.max()),
        "ring_graph_fraction": float(np.mean(rings > 0)),
        "mean_ring_boundary_incidence": float(boundary.mean()),
    }
