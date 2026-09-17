"""Targeted correctness tests for FSAR-R2-AR0-EDGE.

Self-contained (no ZINC data) except the positive/negative controls, which use
the runner's tiny synthetic molecules.  Covers the pre-registered edge list:

1.  edge endpoint-swap invariance ``psi_uv = psi_vu``
2.  ``psi_e`` chemistry purity
3.  node relabel invariance of BV / BVE / BVEM
4.  bond permutation leaves S, A, node C_V, P_E invariant
5.  bond permutation changes C_E on an asymmetric toy
6.  exact small-graph enumeration ``mean_pi C_E = 0``
7.  scalar ``mean_pi <W_E, C_E> = 0``
8.  undirected bond canonicalization
9.  train-only edge RMS scaler
10. batched edge statistics equal the numpy reference
11. ``W_E`` / ``W_ME`` nonzero task gradient
12. no edge bypass
13. BV / BVE / BVEM shared initialization identity
14. zero-init prediction identity
15. parameter counts
16. assignment-only positive control (+ marginal-only negative control)
"""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_edge as edge
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dummy_edge_scalers() -> dict[str, np.ndarray]:
    node = np.ones((edge.PHI_DIM, edge.ATOM_CATEGORIES), dtype=np.float32)
    edge_shape = np.ones((edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES), dtype=np.float32)
    return {
        "C_rms": node,
        "C_mask": node.copy(),
        "P_rms": node.copy(),
        "P_mask": node.copy(),
        "CE_rms": edge_shape,
        "CE_mask": edge_shape.copy(),
        "PE_rms": edge_shape.copy(),
        "PE_mask": edge_shape.copy(),
    }


def _molecule(graph, types, bond_by_edge, target=0.0) -> edge.EdgeMoleculeFeatures:
    phi = r2.build_phi(graph)
    edges = sorted((int(u), int(v)) for u, v in graph.edges())
    edge_u = np.asarray([u for u, _v in edges], dtype=np.int64)
    edge_v = np.asarray([v for _u, v in edges], dtype=np.int64)
    bond = np.asarray([int(bond_by_edge[(u, v)]) for u, v in edges], dtype=np.int64)
    marginal = r2.build_A(types, {key: int(value) for key, value in bond_by_edge.items()}, len(types))
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


def _path_graph(n: int):
    return from_edges(n, [(index, index + 1) for index in range(n - 1)])


def _path_bonds(n: int, double_edge: int = -1) -> dict[tuple[int, int], int]:
    bonds = {(index, index + 1): 1 for index in range(n - 1)}
    if double_edge >= 0:
        bonds[(double_edge, double_edge + 1)] = 2
    return bonds


def _relabel(graph, perm):
    n = len(perm)
    inverse = {int(old): int(new) for new, old in enumerate(perm)}
    edges = [(inverse[int(u)], inverse[int(v)]) for u, v in graph.edges()]
    return from_edges(n, edges)


def _predict(variant: str, molecules, seed: int = 0) -> np.ndarray:
    model = edge.build_edge_model(variant, _dummy_edge_scalers(), seed=int(seed))
    model.eval()
    batch = edge.collate_edge_molecules(molecules)
    with torch.no_grad():
        return model(batch).view(-1).numpy()


def _module_source(module) -> str:
    return Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")


def _toy_batch() -> edge.FSAREdgeBatch:
    molecules = [
        _molecule(_path_graph(5), np.asarray([0, 1, 2, 1, 0]), _path_bonds(5, 1), target=1.0),
        _molecule(_path_graph(4), np.asarray([1, 0, 1, 1]), _path_bonds(4), target=0.0),
    ]
    return edge.collate_edge_molecules(molecules)


# ---------------------------------------------------------------------------
# 1. edge endpoint-swap invariance
# ---------------------------------------------------------------------------


def test_edge_endpoint_swap_invariance() -> None:
    phi = np.random.default_rng(0).normal(size=(6, edge.PHI_DIM))
    roles_forward = edge.build_edge_roles(phi, [(0, 3), (2, 5)])
    roles_backward = edge.build_edge_roles(phi, [(3, 0), (5, 2)])
    assert np.allclose(roles_forward, roles_backward, atol=0.0)
    # swapping the endpoints of a single edge is exactly a no-op
    single = edge.build_edge_roles(phi, [(1, 4)])
    assert np.allclose(single, edge.build_edge_roles(phi, [(4, 1)]), atol=0.0)


# ---------------------------------------------------------------------------
# 2. psi chemistry purity
# ---------------------------------------------------------------------------


def test_edge_role_chemistry_purity() -> None:
    signature = inspect.signature(edge.build_edge_roles)
    assert list(signature.parameters) == ["phi", "edges"]
    source = _module_source(edge)
    start = source.index("def build_edge_roles")
    end = source.index("def one_hot_bond")
    body = source[start:end]
    for forbidden in ("node_types", "edge_types", "atom_idx", "bond_type", "struct_bond"):
        assert forbidden not in body
    # identical phi -> identical roles regardless of any attribute values
    phi = r2.build_phi(_path_graph(5))
    assert np.allclose(
        edge.build_edge_roles(phi, [(0, 1)]), edge.build_edge_roles(phi, [(0, 1)]), atol=0.0
    )


# ---------------------------------------------------------------------------
# 3. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance() -> None:
    graph = _path_graph(5)
    types = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    bonds = _path_bonds(5, 2)
    base = _molecule(graph, types, bonds, target=1.0)
    perm = [3, 0, 4, 1, 2]
    relabelled_graph = _relabel(graph, perm)
    relabelled_types = np.asarray([types[perm[i]] for i in range(5)], dtype=np.int64)
    inverse = {int(old): int(new) for new, old in enumerate(perm)}
    relabelled_bonds = {
        tuple(sorted((inverse[u], inverse[v]))): value for (u, v), value in bonds.items()
    }
    relabelled = _molecule(relabelled_graph, relabelled_types, relabelled_bonds, target=1.0)
    for variant in edge.MODELS:
        assert np.allclose(
            _predict(variant, [base]), _predict(variant, [relabelled]), atol=1e-5
        ), variant


# ---------------------------------------------------------------------------
# 4/5. bond permutation channel invariance / sensitivity
# ---------------------------------------------------------------------------


def test_bond_permutation_channel_invariance() -> None:
    batch = _toy_batch()
    permuted = edge.permute_bond_types_within_graphs(batch, seed=5)
    shuffled_batch = batch.with_r(permuted)
    # S is computed from phi only; A is a graph-level count; node statistics use q.
    model = edge.build_edge_model("BVE", _dummy_edge_scalers(), seed=0)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model.encode_s(batch), model.encode_s(shuffled_batch), atol=0.0)
        assert torch.allclose(batch.A, shuffled_batch.A, atol=0.0)
        c_node_a, _p_a = model.compute_node_statistics(batch)
        c_node_b, _p_b = model.compute_node_statistics(shuffled_batch)
        assert torch.allclose(c_node_a, c_node_b, atol=0.0)
        _c_a, p_edge_a = model.compute_edge_statistics(batch)
        _c_b, p_edge_b = model.compute_edge_statistics(shuffled_batch)
        assert torch.allclose(p_edge_a, p_edge_b, atol=1e-5)
    # the bond-type multiset is preserved inside every graph
    for graph_id in range(batch.n_graphs):
        mask = batch.edge_graph == graph_id
        assert torch.allclose(
            torch.sort(batch.r[mask], dim=0).values, torch.sort(permuted[mask], dim=0).values
        )


def test_bond_permutation_changes_ce() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (2, 4)])
    molecule = _molecule(graph, np.zeros(5, dtype=np.int64), {(0, 1): 1, (1, 2): 2, (2, 3): 1, (2, 4): 1})
    batch = edge.collate_edge_molecules([molecule])
    permuted = edge.permute_bond_types_within_graphs(batch, seed=3)
    shuffled_batch = batch.with_r(permuted)
    model = edge.build_edge_model("BVE", _dummy_edge_scalers(), seed=0)
    model.eval()
    with torch.no_grad():
        c_a, _p = model.compute_edge_statistics(batch)
        c_b, _p2 = model.compute_edge_statistics(shuffled_batch)
    assert not torch.allclose(c_a, c_b, atol=1e-6)


# ---------------------------------------------------------------------------
# 6/7. exact permutation expectations
# ---------------------------------------------------------------------------


def test_exact_permutation_expectation_edge() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    phi = r2.build_phi(graph)
    roles = edge.build_edge_roles(phi, [(0, 1), (1, 2), (2, 3)])
    r = edge.one_hot_bond(np.asarray([1, 2, 1], dtype=np.int64))
    accumulator = np.zeros((edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES))
    count = 0
    for order in itertools.permutations(range(3)):
        c_matrix, _p = edge.edge_center_stats(roles, r[list(order)])
        accumulator += c_matrix
        count += 1
    assert count == 6
    assert np.allclose(accumulator / count, 0.0, atol=1e-10)


def test_scalar_edge_exact_expectation() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    phi = r2.build_phi(graph)
    roles = edge.build_edge_roles(phi, [(0, 1), (1, 2), (2, 3)])
    r = edge.one_hot_bond(np.asarray([1, 2, 1], dtype=np.int64))
    weight = torch.as_tensor(np.random.default_rng(3).normal(size=(edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES)))
    scalar = 0.0
    count = 0
    for order in itertools.permutations(range(3)):
        c_matrix, _p = edge.edge_center_stats(roles, r[list(order)])
        scalar += float((torch.as_tensor(c_matrix) * weight).sum())
        count += 1
    assert abs(scalar / count) < 1e-9


# ---------------------------------------------------------------------------
# 8. undirected bond canonicalization
# ---------------------------------------------------------------------------


def test_undirected_bond_canonicalization() -> None:
    edge_index = torch.as_tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.as_tensor([[1], [1], [2], [2], [3], [3]], dtype=torch.long)
    data = Data(
        x=torch.as_tensor([[1], [2], [3], [1]], dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.zeros(1, 1),
    )
    graph, node_types, edge_types = _data_to_graph(data)
    assert len(edge_types) == 3  # three undirected bonds, not six directed entries
    molecule = _molecule(graph, node_types, edge_types)
    assert molecule.edge_u.shape[0] == 3
    roles = edge.build_edge_roles(molecule.phi, list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist())))
    assert roles.shape[0] == 3
    assert molecule.bond_type.tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# 9. train-only edge scaler
# ---------------------------------------------------------------------------


def test_train_only_edge_scaler() -> None:
    rng = np.random.default_rng(0)
    train_stats = [rng.normal(size=(edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES)) + 3.0 for _ in range(6)]
    held_out = [rng.normal(size=(edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES)) * 10.0 + 50.0 for _ in range(4)]
    train_only = edge.fit_edge_scalers(train_stats, train_stats)
    with_held_out = edge.fit_edge_scalers(train_stats + held_out, train_stats + held_out)
    manual_rms = np.sqrt(np.stack([matrix**2 for matrix in train_stats]).mean(axis=0) + edge.SCALER_EPS)
    assert np.allclose(train_only["CE_rms"], np.where(train_only["CE_mask"] > 0, manual_rms, 1.0))
    assert not np.allclose(train_only["CE_rms"], with_held_out["CE_rms"])
    centered_std = np.stack([m - m.mean(axis=0, keepdims=True) for m in train_stats]).std(axis=0)
    assert not np.allclose(manual_rms, centered_std)


# ---------------------------------------------------------------------------
# 10. batched statistics match the numpy reference
# ---------------------------------------------------------------------------


def test_batched_edge_stats_match_numpy() -> None:
    molecules = [
        _molecule(_path_graph(4), np.asarray([0, 1, 1, 1]), _path_bonds(4, 1)),
        _molecule(_path_graph(5), np.asarray([1, 0, 2, 1, 2]), _path_bonds(5, 3)),
    ]
    batch = edge.collate_edge_molecules(molecules)
    c_batch, p_batch = edge.segment_edge_stats(batch.psi, batch.r, batch.edge_graph, batch.n_graphs)
    for index, molecule in enumerate(molecules):
        roles = edge.build_edge_roles(
            molecule.phi, list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist()))
        )
        c_ref, p_ref = edge.edge_center_stats(roles, edge.one_hot_bond(molecule.bond_type))
        assert np.allclose(c_batch[index].numpy(), c_ref, atol=1e-4)
        assert np.allclose(p_batch[index].numpy(), p_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 11. gradient viability
# ---------------------------------------------------------------------------


def test_edge_weight_gradient_viability() -> None:
    batch = _toy_batch()
    for variant, name in (("BVE", "W_E"), ("BVEM", "W_ME")):
        model = edge.build_edge_model(variant, _dummy_edge_scalers(), seed=0)
        model.train()
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
        model.zero_grad()
        loss.backward()
        weight = getattr(model, name)
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0.0
        assert float(model.W_B.grad.abs().sum()) > 0.0
        assert float(model.s_encoder[0].weight.grad.abs().sum()) > 0.0
        assert float(model.base[0].weight.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 12. no edge bypass
# ---------------------------------------------------------------------------


def test_no_edge_bypass() -> None:
    batch = _toy_batch()
    for variant in edge.EDGE_MODELS:
        model = edge.build_edge_model(variant, _dummy_edge_scalers(), seed=0)
        model.eval()
        with torch.no_grad():
            if variant == "BVE":
                model.W_E.copy_(torch.randn_like(model.W_E))
            else:
                model.W_ME.copy_(torch.randn_like(model.W_ME))
            c_node, _p = model.compute_node_statistics(batch)
            expected = model.base_prediction(batch) + model.node_assignment_term(c_node)
            zeros = torch.zeros(
                batch.n_graphs, edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES, dtype=batch.psi.dtype
            )
            original = model.compute_edge_statistics
            model.compute_edge_statistics = lambda *a, **k: (zeros, zeros)
            prediction = model(batch)
            model.compute_edge_statistics = original
        assert torch.allclose(expected, prediction, atol=1e-6)
    model = edge.build_edge_model("BVE", _dummy_edge_scalers(), seed=0)
    assert isinstance(model.W_E, torch.nn.Parameter)
    assert tuple(model.W_E.shape) == (edge.EDGE_ROLE_DIM, edge.BOND_CATEGORIES)
    assert not any("W_E_bias" in key for key in model.state_dict())
    source = _module_source(edge)
    tree = ast.parse(source)
    allowed = {
        "phi",
        "q",
        "node_graph",
        "psi",
        "r",
        "edge_graph",
        "A",
        "n_nodes",
        "n_edges",
        "n_graphs",
        "y",
        "to",
        "with_r",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "batch":
                assert node.attr in allowed, node.attr


# ---------------------------------------------------------------------------
# 13/14. shared init / zero-init identity
# ---------------------------------------------------------------------------


def test_bv_bve_bvem_share_init() -> None:
    models = {variant: edge.build_edge_model(variant, _dummy_edge_scalers(), seed=7) for variant in edge.MODELS}
    reference = models["BV"].state_dict()
    for variant in ("BVE", "BVEM"):
        current = models[variant].state_dict()
        for key, value in reference.items():
            assert key in current, key
            assert torch.equal(value, current[key]), (variant, key)
        assert any(key.startswith("W_E") or key.startswith("W_ME") for key in current)
        # only the edge parameter differs
        extra = set(current) - set(reference)
        assert len(extra) == 1
        assert float(current[list(extra)[0]].abs().sum()) == 0.0


def test_zero_init_prediction_identity() -> None:
    batch = _toy_batch()
    models = {variant: edge.build_edge_model(variant, _dummy_edge_scalers(), seed=9) for variant in edge.MODELS}
    predictions = {}
    for variant, model in models.items():
        model.eval()
        with torch.no_grad():
            predictions[variant] = model(batch)
    assert torch.equal(predictions["BV"], predictions["BVE"])
    assert torch.equal(predictions["BV"], predictions["BVEM"])


# ---------------------------------------------------------------------------
# 15. parameter counts
# ---------------------------------------------------------------------------


def test_parameter_counts_edge() -> None:
    models = {variant: edge.build_edge_model(variant, _dummy_edge_scalers(), seed=0) for variant in edge.MODELS}
    breakdown = {variant: edge.parameter_breakdown_edge(model) for variant, model in models.items()}
    assert breakdown["BV"]["total"] == 24797
    assert breakdown["BVE"]["total"] == 24797 + edge.EDGE_ROLE_DIM * edge.BOND_CATEGORIES
    assert breakdown["BVEM"]["total"] == 24797 + edge.EDGE_ROLE_DIM * edge.BOND_CATEGORIES
    assert breakdown["BV"]["edge_binding"] == 0
    assert breakdown["BVE"]["node_binding"] == breakdown["BVEM"]["node_binding"] == edge.PHI_DIM * edge.ATOM_CATEGORIES
    for variant in edge.MODELS:
        assert breakdown[variant]["dataset_dependent_vocabulary_params"] == 0


# ---------------------------------------------------------------------------
# 16. synthetic controls
# ---------------------------------------------------------------------------


def test_positive_control_edge_assignment() -> None:
    from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0_edge as runner

    molecules = runner.synthetic_edge_positive(n_examples=200, seed=0)
    results = {variant: runner._train_edge_toy(molecules, variant, seed=0, epochs=250) for variant in edge.MODELS}
    assert results["BVE"]["valid_mae"] < 0.2, results
    assert results["BV"]["valid_mae"] > 0.2, results
    assert results["BVEM"]["valid_mae"] > 0.2, results
    assert results["BVE"]["valid_mae"] < min(
        results["BV"]["valid_mae"], results["BVEM"]["valid_mae"]
    ) - 0.1, results
    assert results["BVE"]["edge_weight_norm"] > 0.0


def test_negative_control_edge_marginal_only() -> None:
    from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0_edge as runner

    molecules = runner.synthetic_edge_negative(n_examples=300, seed=0)
    results = {variant: runner._train_edge_toy(molecules, variant, seed=0, epochs=250) for variant in edge.MODELS}
    # no repeatable BVE advantage over both controls
    assert not (
        results["BVE"]["valid_mae"] < results["BV"]["valid_mae"] - 0.05
        and results["BVE"]["valid_mae"] < results["BVEM"]["valid_mae"] - 0.05
    ), results


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
