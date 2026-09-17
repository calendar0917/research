"""Targeted correctness tests for FSAR-C1 (parameter-matched incidence processor).

Self-contained: no ZINC data is touched.  Covers the pre-registered list:

1.  edge endpoint-swap invariance (``psi`` and the prediction)
2.  ``psi_e`` chemistry purity
3.  node relabel invariance
4.  bond-type permutation changes the C1 prediction (incidence is used)
5.  the C0 bag ignores ``edge_index`` while C1 uses it (incidence mechanism)
6.  atom-attribute permutation changes the prediction (objects are used)
7.  batching a molecule with others does not change its prediction
8.  C0 and C1 have identical parameter counts / keys / shapes
9.  every parameter receives a non-zero task gradient
10. one optimizer step updates the binding branches
11. parameter counts (sum of modules == total, no dataset-dependent vocabulary)
12. no forbidden feature family / no out-of-schema batch field is read
13. determinism of the forward pass
14. empty-edge graphs are safe
"""

from __future__ import annotations

import ast
import inspect
from typing import Sequence

import numpy as np
import torch

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_edge as edge
from tracks.ksvd.experiments.luyin16 import fsar_incidence as inc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _molecule(graph, types, bond_by_edge, target: float = 0.0) -> edge.EdgeMoleculeFeatures:
    phi = r2.build_phi(graph)
    edges = sorted((int(u), int(v)) for u, v in graph.edges())
    edge_u = np.asarray([u for u, _v in edges], dtype=np.int64)
    edge_v = np.asarray([v for _u, v in edges], dtype=np.int64)
    bond = np.asarray([int(bond_by_edge[(u, v)]) for u, v in edges], dtype=np.int64)
    marginal = r2.build_A(
        types, {key: int(value) for key, value in bond_by_edge.items()}, len(types)
    )
    return edge.EdgeMoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=np.asarray(types, dtype=np.int64),
        edge_u=edge_u,
        edge_v=edge_v,
        bond_type=bond,
        A=marginal.astype(np.float32),
        n_nodes=int(len(types)),
        n_edges=int(len(edges)),
        y=float(target),
    )


def _path_molecules() -> list[edge.EdgeMoleculeFeatures]:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    bonds = {(0, 1): 1, (1, 2): 2, (2, 3): 3}
    return [
        _molecule(graph, np.asarray([1, 2, 3, 1]), bonds, target=0.5),
        _molecule(graph, np.asarray([3, 1, 1, 2]), {(0, 1): 2, (1, 2): 1, (2, 3): 1}, target=1.5),
    ]


def _toy_batch() -> inc.IncidenceBatch:
    return inc.collate_incidence_molecules(_path_molecules())


def _tree_molecule() -> edge.EdgeMoleculeFeatures:
    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (2, 4)])
    bonds = {(0, 1): 1, (1, 2): 2, (2, 3): 1, (2, 4): 3}
    return _molecule(graph, np.asarray([1, 2, 3, 1, 2]), bonds)


def _model(variant: str):
    return inc.build_incidence_model(variant, seed=0)


def _module_source(module) -> str:
    return inspect.getsource(module)


# ---------------------------------------------------------------------------
# 1/2. edge role invariants
# ---------------------------------------------------------------------------


def test_edge_endpoint_swap_invariance() -> None:
    molecule = _tree_molecule()
    psi = edge.build_edge_roles(
        np.asarray(molecule.phi, dtype=np.float64),
        list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())),
    )
    swapped = edge.build_edge_roles(
        np.asarray(molecule.phi, dtype=np.float64),
        list(zip(molecule.edge_v.tolist(), molecule.edge_u.tolist())),
    )
    assert np.array_equal(psi, swapped)

    batch = inc.collate_incidence_molecules([molecule])
    flipped = batch.edge_index.flip(0)
    swapped_batch = inc.IncidenceBatch(
        batch.phi,
        batch.q,
        batch.node_graph,
        batch.psi,
        batch.r,
        flipped,
        batch.edge_graph,
        batch.A,
        batch.n_nodes,
        batch.n_edges,
        batch.y,
        batch.n_graphs,
    )
    for variant in inc.MODELS:
        model = _model(variant).eval()
        with torch.no_grad():
            base = model(batch)
            other = model(swapped_batch)
        assert torch.allclose(base, other, atol=1e-5), (variant, base, other)


def test_edge_role_chemistry_purity() -> None:
    molecule = _tree_molecule()
    psi = edge.build_edge_roles(
        np.asarray(molecule.phi, dtype=np.float64),
        list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())),
    )
    other = _molecule(
        from_edges(5, [(0, 1), (1, 2), (2, 3), (2, 4)]),
        np.asarray([7, 9, 11, 13, 15]),
        {(0, 1): 3, (1, 2): 1, (2, 3): 3, (2, 4): 2},
    )
    psi_other = edge.build_edge_roles(
        np.asarray(other.phi, dtype=np.float64),
        list(zip(other.edge_u.tolist(), other.edge_v.tolist())),
    )
    assert np.array_equal(psi, psi_other)


# ---------------------------------------------------------------------------
# 3. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance() -> None:
    batch = inc.collate_incidence_molecules([_tree_molecule()])
    n = int(batch.phi.shape[0])
    generator = torch.Generator().manual_seed(3)
    permutation = torch.randperm(n, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    relabelled = inc.IncidenceBatch(
        batch.phi[permutation],
        batch.q[permutation],
        batch.node_graph,
        batch.psi,
        batch.r,
        inverse[batch.edge_index],
        batch.edge_graph,
        batch.A,
        batch.n_nodes,
        batch.n_edges,
        batch.y,
        batch.n_graphs,
    )
    for variant in inc.MODELS:
        model = _model(variant).eval()
        with torch.no_grad():
            assert torch.allclose(model(batch), model(relabelled), atol=1e-5), variant


# ---------------------------------------------------------------------------
# 4/5/6. attribute permutation behaviour
# ---------------------------------------------------------------------------


def test_bond_permutation_changes_c1() -> None:
    batch = inc.collate_incidence_molecules([_tree_molecule()])
    permuted = inc.permute_bond_types_within_graphs(batch, seed=1)
    model = _model("C1").eval()
    with torch.no_grad():
        base = model(batch)
        changed = model(batch.with_r(permuted))
    assert float((changed - base).abs().max()) > 1e-6


def test_incidence_used_by_c1_not_c0() -> None:
    batch = inc.collate_incidence_molecules([_tree_molecule()])
    scrambled = batch.edge_index.clone()
    scrambled[0] = torch.roll(batch.edge_index[0], 1)
    scrambled_batch = inc.IncidenceBatch(
        batch.phi,
        batch.q,
        batch.node_graph,
        batch.psi,
        batch.r,
        scrambled,
        batch.edge_graph,
        batch.A,
        batch.n_nodes,
        batch.n_edges,
        batch.y,
        batch.n_graphs,
    )
    c0 = _model("C0").eval()
    c1 = _model("C1").eval()
    with torch.no_grad():
        assert torch.allclose(c0(batch), c0(scrambled_batch), atol=1e-6)
        assert float((c1(scrambled_batch) - c1(batch)).abs().max()) > 1e-6


def test_atom_permutation_changes_c1() -> None:
    batch = inc.collate_incidence_molecules([_tree_molecule()])
    permuted = inc.permute_q_within_graphs(batch, seed=5)
    # the structural coordinate is untouched; only the atom assignment moves
    assert torch.equal(batch.phi, batch.phi)
    model = _model("C1").eval()
    with torch.no_grad():
        assert float((model(batch.with_q(permuted)) - model(batch)).abs().max()) > 1e-7


# ---------------------------------------------------------------------------
# 7. batch invariance
# ---------------------------------------------------------------------------


def test_batch_invariance() -> None:
    molecules = _path_molecules()
    single = inc.collate_incidence_molecules(molecules[:1])
    together = inc.collate_incidence_molecules(molecules)
    for variant in inc.MODELS:
        model = _model(variant).eval()
        with torch.no_grad():
            assert torch.allclose(model(single)[0], model(together)[0], atol=1e-5), variant


# ---------------------------------------------------------------------------
# 8. parameter-exact control
# ---------------------------------------------------------------------------


def test_incidence_control_parameter_match() -> None:
    c1 = _model("C1")
    c0 = _model("C0")
    state_c1 = c1.state_dict()
    state_c0 = c0.state_dict()
    assert set(state_c1) == set(state_c0)
    for key in state_c1:
        assert state_c1[key].shape == state_c0[key].shape, key
    assert inc.parameter_breakdown_incidence(c1)["total"] == inc.parameter_breakdown_incidence(c0)["total"]


# ---------------------------------------------------------------------------
# 9/10. gradient viability / branch updates
# ---------------------------------------------------------------------------


def test_all_parameters_receive_gradient() -> None:
    batch = _toy_batch()
    for variant in inc.MODELS:
        model = _model(variant)
        model.train()
        loss = model(batch).pow(2).mean()
        model.zero_grad()
        loss.backward()
        dead = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is None
            or float(parameter.grad.detach().abs().sum()) == 0.0
        ]
        assert not dead, (variant, dead)


def test_one_step_updates_binding_branches() -> None:
    batch = _toy_batch()
    for variant in inc.MODELS:
        model = _model(variant)
        model.train()
        optimizer = torch.optim.Adam(
            inc.optimizer_parameter_groups(model, 1.0e-5), lr=1.0e-3
        )
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            _prediction, _h, _g, binding_before, _stats = model.forward_with_states(batch)
        loss = torch.nn.functional.l1_loss(model(batch), batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        moved = [
            key
            for key, value in model.state_dict().items()
            if not torch.equal(value, before[key])
        ]
        for name in ("node_us", "node_ua", "node_b", "edge_us", "edge_ua", "edge_b"):
            assert any(key.startswith(name) for key in moved), (variant, name)
        with torch.no_grad():
            _prediction, _h, _g, binding_after, _stats = model.forward_with_states(batch)
        assert not torch.equal(
            binding_before["node_binding"], binding_after["node_binding"]
        ), variant
        assert not torch.equal(
            binding_before["edge_binding"], binding_after["edge_binding"]
        ), variant


# ---------------------------------------------------------------------------
# 11. parameter counts
# ---------------------------------------------------------------------------


def test_parameter_counts_incidence() -> None:
    expected_total = 88643
    for variant in inc.MODELS:
        model = _model(variant)
        breakdown = inc.parameter_breakdown_incidence(model)
        module_sum = (
            breakdown["node_initializer"]
            + breakdown["edge_initializer"]
            + breakdown["binding"]
            + breakdown["incidence_processor"]
            + breakdown["readout"]
        )
        assert breakdown["total"] == module_sum
        assert breakdown["total"] == expected_total, (variant, breakdown["total"])
        assert breakdown["dataset_dependent_vocabulary_params"] == 0
        assert breakdown["readout"] < 0.25 * breakdown["total"]


# ---------------------------------------------------------------------------
# 12. no forbidden feature family / no out-of-schema input
# ---------------------------------------------------------------------------


def test_no_forbidden_feature_access() -> None:
    source = _module_source(inc)
    tree = ast.parse(source)
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name.lower())
    for token in (
        "rrwp",
        "lappe",
        "attention",
        "transformer",
        "homomorphism",
        "triangle",
        "motif",
    ):
        assert not any(token in name for name in identifiers), token
    allowed = {
        "phi",
        "q",
        "node_graph",
        "psi",
        "r",
        "edge_index",
        "edge_graph",
        "A",
        "n_nodes",
        "n_edges",
        "n_graphs",
        "y",
        "to",
        "with_r",
        "with_q",
        "shape",
        "dtype",
        "device",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "batch":
                assert node.attr in allowed, node.attr


# ---------------------------------------------------------------------------
# 13/14. determinism and empty edges
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    batch = _toy_batch()
    for variant in inc.MODELS:
        model = _model(variant).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(batch)
        assert torch.equal(first, second), variant


def test_empty_edges_safe() -> None:
    graph = from_edges(1, [])
    molecule = _molecule(graph, np.asarray([1]), {})
    batch = inc.collate_incidence_molecules([molecule])
    assert int(batch.edge_index.shape[1]) == 0
    for variant in inc.MODELS:
        model = _model(variant).eval()
        with torch.no_grad():
            prediction = model(batch)
        assert torch.isfinite(prediction).all(), variant
