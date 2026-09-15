"""Tests for the zero-training rank-1 structural-encoder audit (Workstream B).

Fast unit tests only: no checkpoints, no full data, no training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import (
    zinc_structural_encoder_rank1_audit as r1,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)


def _path_data() -> Data:
    """Typed path 0-1-2-3 with distinct atom and bond types."""
    edges = [(0, 1), (1, 2), (2, 3)]
    bond_types = [0, 1, 2]
    bi = edges + [(v, u) for u, v in edges]
    edge_attr = bond_types + bond_types
    return Data(
        edge_index=torch.tensor(bi, dtype=torch.long).t().contiguous(),
        x=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=4,
    )


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


def test_molecule_descriptors_match_patch_graph():
    graph = sspe._patch_graphs_from_dataset([_path_data()])[0]
    row = r1.molecule_descriptors(graph)
    scalar = row["scalar"]
    # patch 0 is the radius-2 ego graph of node 0: nodes {0,1,2}, edges {0-1,1-2}
    assert scalar[0, 0] == 3  # n_atoms
    assert scalar[0, 1] == 2  # n_bonds
    assert scalar[0, 2] == 1  # root_degree
    assert scalar[0, 5] == 0  # branching (max degree 2)
    assert scalar[0, 6] == 0  # cycle rank
    assert scalar[0, 7] == 1  # boundary size (one node at distance 2)
    # per-patch atom counts sum to the patch node count
    assert np.allclose(row["atom_counts"].sum(axis=1), scalar[:, 0])
    # bond counts sum to the bond count
    assert np.allclose(row["bond_counts"].sum(axis=1), scalar[:, 1])
    # distance histogram sums to the patch node count
    assert np.allclose(row["dist_hist"].sum(axis=1), scalar[:, 0])
    # every patch has a root with degree >= 1 in this path
    assert np.all(scalar[:, 2] >= 1)


def test_descriptor_matrix_names_and_shape():
    graph = sspe._patch_graphs_from_dataset([_path_data()])
    patch_cont = np.zeros((graph[0].n_patches, 5))
    matrix, names = r1.descriptor_matrix(graph, patch_cont)
    assert matrix.shape == (graph[0].n_patches, len(names))
    assert len(names) == len(set(names))
    assert names[: len(r1.SCALAR_NAMES)] == list(r1.SCALAR_NAMES)
    assert "patch_cont_0" in names


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------


def test_spectrum_rank1_reconstruction():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=16)
    direction /= np.linalg.norm(direction)
    scores = rng.normal(size=500)
    matrix = np.outer(scores, direction) + rng.normal(scale=1.0e-6, size=(500, 16))
    result = r1._spectrum(matrix, "test")
    assert result["pc1_explained_variance"] > 0.99
    assert abs(sum(result["explained_variance_ratio"]) - 1.0) < 1.0e-9
    # deterministic sign convention
    fixed = r1._sign_fix(np.asarray(result["pc1_direction"]))
    assert np.allclose(fixed, result["pc1_direction"])


def test_unique_structure_weighting():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    keys = ["a", "a", "b", "b"]
    unique, n = r1._unique_structure_matrix(matrix, keys)
    assert n == 2
    assert np.allclose(sorted(unique[:, 0].tolist()), [0.0, 1.0])


# ---------------------------------------------------------------------------
# linear probe
# ---------------------------------------------------------------------------


def test_ridge_r2_recovers_linear_map():
    rng = np.random.default_rng(1)
    train_x = rng.normal(size=(200, 5))
    weights = rng.normal(size=5)
    train_y = train_x @ weights + 0.5
    valid_x = rng.normal(size=(100, 5))
    valid_y = valid_x @ weights + 0.5
    result = r1._ridge_r2(train_x, train_y, valid_x, valid_y, alpha=1.0e-8)
    assert result["valid_r2"] > 0.999


# ---------------------------------------------------------------------------
# frozen functional override
# ---------------------------------------------------------------------------


class _LinearEncoder(torch.nn.Module):
    """Small stand-in for the structural encoder (output width 16)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)

    def forward(self, data):
        return self.linear(data.x)


def _dummy(rows: int):
    data = Data(x=torch.randn(rows, 3))
    return data


def test_structural_override_modes():
    torch.manual_seed(0)
    base = _LinearEncoder()
    centre = np.zeros(16)
    direction = np.zeros(16)
    direction[0] = 1.0

    data = _dummy(7)
    with torch.no_grad():
        expected_full = base(data)

    override_full = r1._StructuralOverride(base, centre, direction, "full")
    override_mean = r1._StructuralOverride(base, centre, direction, "mean")
    override_rank1 = r1._StructuralOverride(base, centre, direction, "rank1")
    with torch.no_grad():
        out_full = override_full(data)
        out_mean = override_mean(data)
        out_rank1 = override_rank1(data)
    assert torch.allclose(out_full, expected_full)
    assert torch.allclose(out_mean, torch.zeros_like(out_mean))
    # rank-1 reconstruction only keeps the projection along the direction
    assert torch.allclose(out_rank1[:, 1:], torch.zeros_like(out_rank1[:, 1:]))
    assert out_rank1.shape == (7, 16)
    # idempotence: feeding the rank-1 output back through gives the same value
    rank1_again = r1._StructuralOverride(
        _ConstantEncoder(out_rank1), centre, direction, "rank1"
    )
    with torch.no_grad():
        assert torch.allclose(rank1_again(_dummy(7)), out_rank1)


class _ConstantEncoder(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, data):
        return self.value
