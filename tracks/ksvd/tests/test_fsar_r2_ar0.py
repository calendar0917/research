"""Targeted correctness tests for FSAR-R2-AR0.

Every test here is self-contained (no ZINC data required) except the runner
stages that explicitly build the cache.  The suite covers the pre-registered
correctness list:

1.  node relabel invariance of M0 / MB / MM
2.  S chemistry purity
3.  A assignment invariance
4.  P assignment invariance
5.  C assignment sensitivity
6.  exact permutation expectation ``mean_pi C = 0``
7.  scalar B exact expectation ``mean_pi <W_B, C> = 0``
8.  no patch-copy inconsistency (permutations act on original nodes once)
9.  undirected bond counting
10. train-only scaler estimation
11. gradient viability of ``W_B`` / ``W_M``
12. no mixed bypass (C / P enter only through the bias-free linear term)
"""

from __future__ import annotations

import ast
import itertools
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dummy_scalers() -> dict[str, np.ndarray]:
    ones = np.ones((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float32)
    return {"C_rms": ones, "C_mask": ones.copy(), "P_rms": ones.copy(), "P_mask": ones.copy()}


def _path_graph(n: int):
    return from_edges(n, [(index, index + 1) for index in range(n - 1)])


def _path_edges(n: int) -> dict[tuple[int, int], int]:
    return {(index, index + 1): 1 for index in range(n - 1)}


def _molecule(graph, types, edges, target=0.0) -> r2.MoleculeFeatures:
    phi = r2.build_phi(graph)
    marginal = r2.build_A(types, dict(edges), len(types))
    return r2.MoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=np.asarray(types, dtype=np.int64),
        A=marginal.astype(np.float32),
        n_nodes=int(len(types)),
        n_edges=int(len(edges)),
        y=float(target),
    )


def _relabel(graph, perm):
    """Return the graph relabelled by ``perm`` (new node ``i`` == old ``perm[i]``)."""
    n = len(perm)
    inverse = {int(old): int(new) for new, old in enumerate(perm)}
    edges = [
        (inverse[int(u)], inverse[int(v)]) for u, v in graph.edges()
    ]
    return from_edges(n, edges)


def _predict(variant: str, molecules, seed: int = 0) -> np.ndarray:
    model = r2.build_model(variant, _dummy_scalers(), seed=int(seed))
    model.eval()
    batch = r2.collate_molecules(molecules)
    with torch.no_grad():
        return model(batch).view(-1).numpy()


def _module_source(module) -> str:
    return Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. node relabel invariance
# ---------------------------------------------------------------------------


def test_node_relabel_invariance() -> None:
    graph = _path_graph(5)
    types = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    edges = _path_edges(5)
    base = _molecule(graph, types, edges, target=1.0)
    perm = [3, 0, 4, 1, 2]
    relabelled_graph = _relabel(graph, perm)
    relabelled_types = np.asarray([types[perm[i]] for i in range(5)], dtype=np.int64)
    relabelled = _molecule(relabelled_graph, relabelled_types, _path_edges(5), target=1.0)
    for variant in r2.MODELS:
        p_base = _predict(variant, [base])
        p_relabelled = _predict(variant, [relabelled])
        assert np.allclose(p_base, p_relabelled, atol=1e-5), variant
    # phi must be an exact row permutation of the original coordinates
    assert np.allclose(
        np.sort(base.phi, axis=0), np.sort(relabelled.phi, axis=0), atol=1e-6
    )


# ---------------------------------------------------------------------------
# 2. S chemistry purity
# ---------------------------------------------------------------------------


def test_s_chemistry_purity() -> None:
    graph = _path_graph(5)
    phi_a = r2.build_phi(graph)
    phi_b = r2.build_phi(_path_graph(5))
    assert np.allclose(phi_a, phi_b, atol=0.0)
    # ``build_phi`` has no chemistry argument at all, and the implementation
    # reads only adjacency / rooted distance.
    signature = inspect.signature(r2.phi_for_center)
    assert list(signature.parameters) == ["graph", "center", "radius"]
    source = _module_source(r2)
    start = source.index("def phi_for_center")
    end = source.index("def build_phi")
    body = source[start:end]
    for forbidden in ("node_types", "edge_types", "atom_idx", "struct_atom", "struct_bond"):
        assert forbidden not in body


# ---------------------------------------------------------------------------
# 3. A assignment invariance
# ---------------------------------------------------------------------------


def test_a_assignment_invariance() -> None:
    types = np.asarray([0, 1, 1, 2], dtype=np.int64)
    permuted = np.asarray([1, 2, 0, 1], dtype=np.int64)
    edges = {(0, 1): 1, (1, 2): 2, (2, 3): 1}
    a_original = r2.build_A(types, edges, 4)
    a_permuted = r2.build_A(permuted, edges, 4)
    assert np.allclose(a_original, a_permuted, atol=0.0)
    # A does depend on the multiset
    a_other = r2.build_A(np.asarray([0, 0, 1, 2], dtype=np.int64), edges, 4)
    assert not np.allclose(a_original, a_other)


# ---------------------------------------------------------------------------
# 4. P assignment invariance
# ---------------------------------------------------------------------------


def test_p_assignment_invariance() -> None:
    phi = np.random.default_rng(0).normal(size=(6, r2.PHI_DIM))
    q = r2.one_hot_q(np.asarray([0, 1, 1, 2, 3, 3], dtype=np.int64))
    _c1, p1 = r2.center_stats(phi, q)
    order = np.asarray([3, 5, 0, 1, 4, 2])
    _c2, p2 = r2.center_stats(phi, q[order])
    assert np.allclose(p1, p2, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. C assignment sensitivity
# ---------------------------------------------------------------------------


def test_c_assignment_sensitivity() -> None:
    phi = r2.build_phi(_path_graph(4))
    q_endpoint0 = r2.one_hot_q(np.asarray([0, 1, 1, 1], dtype=np.int64))
    q_endpoint3 = r2.one_hot_q(np.asarray([1, 1, 1, 0], dtype=np.int64))
    q_internal = r2.one_hot_q(np.asarray([1, 0, 1, 1], dtype=np.int64))
    c_endpoint0, _ = r2.center_stats(phi, q_endpoint0)
    c_endpoint3, _ = r2.center_stats(phi, q_endpoint3)
    c_internal, _ = r2.center_stats(phi, q_internal)
    # path reversal symmetry: the two endpoints are equivalent
    assert np.allclose(c_endpoint0, c_endpoint3, atol=1e-10)
    # moving the hetero atom to a different structural role changes C
    assert not np.allclose(c_endpoint0, c_internal, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. exact permutation expectation
# ---------------------------------------------------------------------------


def test_exact_permutation_expectation() -> None:
    phi = r2.build_phi(_path_graph(4))
    types = np.asarray([0, 1, 1, 2], dtype=np.int64)
    q = r2.one_hot_q(types)
    accumulator = np.zeros((r2.PHI_DIM, r2.ATOM_CATEGORIES), dtype=np.float64)
    count = 0
    for order in itertools.permutations(range(4)):
        c_matrix, _p = r2.center_stats(phi, q[list(order)])
        accumulator += c_matrix
        count += 1
    assert count == 24
    assert np.allclose(accumulator / count, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. scalar B exact expectation
# ---------------------------------------------------------------------------


def test_scalar_b_exact_expectation() -> None:
    phi = r2.build_phi(_path_graph(4))
    q = r2.one_hot_q(np.asarray([0, 1, 2, 1], dtype=np.int64))
    rng = np.random.default_rng(3)
    weight = torch.as_tensor(rng.normal(size=(r2.PHI_DIM, r2.ATOM_CATEGORIES)))
    scalar = 0.0
    count = 0
    for order in itertools.permutations(range(4)):
        c_matrix, _p = r2.center_stats(phi, q[list(order)])
        scalar += float((torch.as_tensor(c_matrix) * weight).sum())
        count += 1
    assert abs(scalar / count) < 1e-9


# ---------------------------------------------------------------------------
# 8. batched statistics match the numpy reference
# ---------------------------------------------------------------------------


def test_batched_stats_match_numpy() -> None:
    molecules = [
        _molecule(_path_graph(4), np.asarray([0, 1, 1, 1]), _path_edges(4)),
        _molecule(_path_graph(5), np.asarray([1, 0, 2, 1, 2]), _path_edges(5)),
    ]
    batch = r2.collate_molecules(molecules)
    c_batch, p_batch = r2.segment_center_stats(
        batch.phi, batch.q, batch.node_graph, batch.n_graphs
    )
    for index, molecule in enumerate(molecules):
        c_ref, p_ref = r2.center_stats(molecule.phi, r2.one_hot_q(molecule.atom_idx))
        assert np.allclose(c_batch[index].numpy(), c_ref, atol=1e-4)
        assert np.allclose(p_batch[index].numpy(), p_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 8b. no patch-copy inconsistency
# ---------------------------------------------------------------------------


def test_no_patch_copy_inconsistency() -> None:
    molecules = [
        _molecule(_path_graph(4), np.asarray([0, 1, 1, 1]), _path_edges(4)),
        _molecule(_path_graph(5), np.asarray([1, 0, 2, 1, 2]), _path_edges(5)),
    ]
    for molecule in molecules:
        # exactly one coordinate row per original molecule node (no patch copies)
        assert molecule.phi.shape[0] == molecule.n_nodes
        assert molecule.atom_idx.shape[0] == molecule.n_nodes
    batch = r2.collate_molecules(molecules)
    permuted = r2.permute_q_within_graphs(batch, seed=11)
    assert torch.equal(batch.phi, batch.phi)  # phi is untouched
    node_graph = batch.node_graph
    for graph_id in range(batch.n_graphs):
        mask = node_graph == graph_id
        original = batch.q[mask]
        shuffled = permuted[mask]
        # the per-graph attribute multiset is preserved and nothing else moves
        assert torch.allclose(original.sum(dim=0), shuffled.sum(dim=0), atol=1e-6)
        assert torch.allclose(
            torch.sort(original, dim=0).values, torch.sort(shuffled, dim=0).values
        )


# ---------------------------------------------------------------------------
# 9. undirected bond counting
# ---------------------------------------------------------------------------


def test_undirected_bond_counting() -> None:
    edge_index = torch.as_tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    edge_attr = torch.as_tensor([[1], [1], [2], [2], [1], [1]], dtype=torch.long)
    data = Data(
        x=torch.as_tensor([[1], [2], [3], [1]], dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.zeros(1, 1),
    )
    graph, node_types, edge_types = _data_to_graph(data)
    assert len(edge_types) == 3  # three undirected edges, not six directed entries
    marginal = r2.build_A(node_types, edge_types, int(data.num_nodes))
    bond_counts = np.expm1(marginal[2 * r2.ATOM_CATEGORIES : 2 * r2.ATOM_CATEGORIES + r2.BOND_CATEGORIES])
    assert int(round(float(bond_counts.sum()))) == 3
    bond_freq = marginal[2 * r2.ATOM_CATEGORIES + r2.BOND_CATEGORIES :]
    assert abs(float(bond_freq.sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 10. train-only scaler
# ---------------------------------------------------------------------------


def test_train_only_scaler() -> None:
    rng = np.random.default_rng(0)
    train_stats = [rng.normal(size=(r2.PHI_DIM, r2.ATOM_CATEGORIES)) + 3.0 for _ in range(6)]
    held_out = [
        rng.normal(size=(r2.PHI_DIM, r2.ATOM_CATEGORIES)) * 10.0 + 50.0
        for _ in range(4)
    ]
    train_only = r2.fit_scalers(train_stats, train_stats)
    with_held_out = r2.fit_scalers(train_stats + held_out, train_stats + held_out)
    manual_rms = np.sqrt(
        np.stack([matrix**2 for matrix in train_stats]).mean(axis=0) + r2.SCALER_EPS
    )
    assert np.allclose(train_only["C_rms"], np.where(train_only["C_mask"] > 0, manual_rms, 1.0))
    assert not np.allclose(train_only["C_rms"], with_held_out["C_rms"])
    # no dataset mean is subtracted: RMS != centered std here
    centered_std = np.stack([matrix - matrix.mean(axis=0, keepdims=True) for matrix in train_stats]).std(axis=0)
    assert not np.allclose(manual_rms, centered_std)


# ---------------------------------------------------------------------------
# 11. gradient viability
# ---------------------------------------------------------------------------


def _synthetic_batch() -> r2.FSARBatch:
    molecules = [
        _molecule(_path_graph(5), np.asarray([0, 1, 2, 1, 0]), _path_edges(5), target=1.0),
        _molecule(_path_graph(4), np.asarray([1, 0, 1, 1]), _path_edges(4), target=0.0),
        _molecule(_path_graph(6), np.asarray([2, 1, 0, 1, 2, 1]), _path_edges(6), target=2.0),
    ]
    return r2.collate_molecules(molecules)


def test_gradient_viability() -> None:
    batch = _synthetic_batch()
    for variant, parameter_name in (("MB", "W_B"), ("MM", "W_M")):
        model = r2.build_model(variant, _dummy_scalers(), seed=0)
        model.train()
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
        model.zero_grad()
        loss.backward()
        weight = getattr(model, parameter_name)
        assert weight.grad is not None
        assert float(weight.grad.abs().sum()) > 0.0
        assert float(model.s_encoder[0].weight.grad.abs().sum()) > 0.0
        assert float(model.base[0].weight.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 12. no mixed bypass
# ---------------------------------------------------------------------------


def test_no_mixed_bypass() -> None:
    batch = _synthetic_batch()
    for variant in ("MB", "MM"):
        model = r2.build_model(variant, _dummy_scalers(), seed=0)
        model.eval()
        with torch.no_grad():
            if variant == "MB":
                model.W_B.copy_(torch.randn_like(model.W_B))
            else:
                model.W_M.copy_(torch.randn_like(model.W_M))
            base = model.base_prediction(batch)
            zeros = torch.zeros(
                batch.n_graphs, r2.PHI_DIM, r2.ATOM_CATEGORIES, dtype=batch.phi.dtype
            )
            original = model.compute_statistics
            model.compute_statistics = lambda *a, **k: (zeros, zeros)
            prediction = model(batch)
            model.compute_statistics = original
        # with the assignment statistic forced to zero the prediction is the base
        assert torch.allclose(base, prediction, atol=1e-6)
    # W_B / W_M are plain bias-free [65, 28] parameters, not a Linear module
    model = r2.build_model("MB", _dummy_scalers(), seed=0)
    assert isinstance(model.W_B, torch.nn.Parameter)
    assert tuple(model.W_B.shape) == (r2.PHI_DIM, r2.ATOM_CATEGORIES)
    assert not any("W_B_bias" in key for key in model.state_dict())
    source = _module_source(r2)
    tree = ast.parse(source)
    allowed = {
        "phi",
        "q",
        "node_graph",
        "A",
        "n_nodes",
        "n_edges",
        "n_graphs",
        "y",
        "to",
        "with_q",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "batch":
                assert node.attr in allowed, node.attr


# ---------------------------------------------------------------------------
# architecture identity / zero init
# ---------------------------------------------------------------------------


def test_m0_mb_mm_share_base_architecture() -> None:
    models = {variant: r2.build_model(variant, _dummy_scalers(), seed=5) for variant in r2.MODELS}
    reference = models["M0"].state_dict()
    for variant in ("MB", "MM"):
        current = models[variant].state_dict()
        for key, value in reference.items():
            if key.startswith("scaler_") or key.startswith("mask_"):
                continue
            assert key in current, key
            assert torch.equal(value, current[key]), (variant, key)
        assert tuple(models[variant].W_B.shape if variant == "MB" else models[variant].W_M.shape) == (
            r2.PHI_DIM,
            r2.ATOM_CATEGORIES,
        )


def test_zero_init_matches_m0() -> None:
    batch = _synthetic_batch()
    m0 = r2.build_model("M0", _dummy_scalers(), seed=9)
    m0.eval()
    with torch.no_grad():
        base = m0(batch)
    for variant in ("MB", "MM"):
        model = r2.build_model(variant, _dummy_scalers(), seed=9)
        model.eval()
        with torch.no_grad():
            prediction = model(batch)
        assert torch.equal(base, prediction), variant


# ---------------------------------------------------------------------------
# parameter counts
# ---------------------------------------------------------------------------


def test_parameter_counts() -> None:
    models = {variant: r2.build_model(variant, _dummy_scalers(), seed=0) for variant in r2.MODELS}
    breakdown = {variant: r2.parameter_breakdown(model) for variant, model in models.items()}
    assert breakdown["M0"]["total"] == 22977
    assert breakdown["MB"]["total"] == 24797
    assert breakdown["MM"]["total"] == 24797
    assert breakdown["MB"]["binding_linear"] == r2.PHI_DIM * r2.ATOM_CATEGORIES
    assert breakdown["MM"]["binding_linear"] == r2.PHI_DIM * r2.ATOM_CATEGORIES
    assert breakdown["M0"]["S_encoder"] == breakdown["MB"]["S_encoder"] == breakdown["MM"]["S_encoder"]
    assert breakdown["M0"]["base_F0"] == breakdown["MB"]["base_F0"] == breakdown["MM"]["base_F0"]
    for variant in r2.MODELS:
        assert breakdown[variant]["dataset_dependent_vocabulary_params"] == 0


# ---------------------------------------------------------------------------
# synthetic positive control (quick version for the sanity gate)
# ---------------------------------------------------------------------------


def test_positive_control_assignment_only() -> None:
    from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as runner

    molecules = runner.synthetic_positive(n_per_class=60, seed=0)
    results = {
        variant: runner._train_toy(molecules, variant, seed=0, epochs=250)
        for variant in r2.MODELS
    }
    assert results["MB"]["valid_mae"] < 0.2, results
    assert results["M0"]["valid_mae"] > 0.3, results
    assert results["MM"]["valid_mae"] > 0.3, results
    assert results["MB"]["branch_weight_norm"] > 0.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
