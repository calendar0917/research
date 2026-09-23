"""PEC-v0 Gate 0 — data-free correctness checks (CPU only).

Imported by both the runner (``zinc_pec_v0 gate0``) and the pytest module
``tracks/ksvd/tests/test_pec_v0.py``.  No ZINC data, no checkpoints, no
official test.
"""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import pec_v0 as pec


def _ring(n: int = 6):
    return from_edges(n, [(i, (i + 1) % n) for i in range(n)])


def _chain(n: int = 5):
    return from_edges(n, [(i, i + 1) for i in range(n - 1)])


def _chem(graph, atom_categories, bond_category=1):
    atom_types = {node: atom_categories[node % len(atom_categories)] for node in graph.nodes}
    edge_types = {graph.edge_key(u, v): bond_category for u, v in graph.edges()}
    return atom_types, edge_types


def _sample(graph, atom_categories=(0, 1, 2, 3), bond_category=1, y=0.0, seed=0):
    del seed
    atom_types, edge_types = _chem(graph, atom_categories, bond_category)
    return pec.build_sample(
        graph,
        [atom_types[node] for node in graph.nodes],
        edge_types,
        y=y,
    )


def _dictionary(seed: int = 0):
    rng = np.random.RandomState(seed)
    d_node = pec.T.normalize_columns(rng.randn(pec.NODE_ROLE_DIM, pec.K_V))
    d_edge = pec.T.normalize_columns(rng.randn(pec.EDGE_ROLE_DIM, pec.K_E))
    return d_node, d_edge


def _model(role_mode="sparse", ablation="true", seed=0):
    d_node, d_edge = _dictionary()
    return pec.build_model(
        role_mode, d_node=d_node, d_edge=d_edge, ablation=ablation, seed=seed
    )


def _batch(samples):
    return pec.collate(list(samples))


# ---------------------------------------------------------------------------
# Gate 0 checks (also used by the runner)
# ---------------------------------------------------------------------------


def check_purity() -> dict[str, object]:
    """Chemistry change must not move the rooted roles or any sparse code."""
    graph = _ring(6)
    base = _sample(graph, atom_categories=(0, 1, 2, 3), bond_category=1)
    changed = _sample(graph, atom_categories=(5, 6, 7, 8), bond_category=3)
    assert np.array_equal(base.node_basis, changed.node_basis)
    assert np.array_equal(base.edge_basis, changed.edge_basis)
    assert np.array_equal(base.occ_shell, changed.occ_shell)
    assert np.array_equal(base.eocc_shellpair, changed.eocc_shellpair)

    model = _model("sparse")
    codes_node = model._node_coord(torch.as_tensor(base.node_basis, dtype=torch.float32))
    codes_edge = model._edge_coord(torch.as_tensor(base.edge_basis, dtype=torch.float32))
    assert codes_node.shape == (base.node_basis.shape[0], pec.K_V)
    assert codes_edge.shape == (base.edge_basis.shape[0], pec.K_E)
    assert torch.isfinite(codes_node).all() and torch.isfinite(codes_edge).all()
    return {
        "node_basis_unchanged_under_chemistry": True,
        "edge_basis_unchanged_under_chemistry": True,
        "codes_finite": True,
    }


def check_chemistry_isolation() -> dict[str, object]:
    """The raw chemical primitive must be a function of chemistry only."""
    graph = _chain(5)
    atoms = [1, 2, 3, 4, 5]
    edge_types = {graph.edge_key(u, v): 1 for u, v in graph.edges()}
    a = pec.build_sample(graph, atoms, edge_types)
    b = pec.build_sample(graph, [7] * 5, edge_types)
    assert np.array_equal(a.atom_idx, np.asarray(atoms, dtype=np.int64))
    # topology roles are read from the untyped adjacency only
    assert np.array_equal(a.node_basis, b.node_basis)
    assert np.array_equal(a.edge_basis, b.edge_basis)
    assert not np.array_equal(a.atom_idx, b.atom_idx)
    return {
        "atom_idx_is_chemistry_only": True,
        "node_basis_topology_only": True,
        "edge_basis_topology_only": True,
    }


def check_assignment_sensitivity() -> dict[str, object]:
    """Fixed topology / marginals, shuffled placement must change E_i."""
    graph = _ring(6)
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    base = pec.build_sample(
        graph, [atom_types[node] for node in graph.nodes], edge_types
    )
    perm = [3, 0, 5, 1, 4, 2]
    shuffled_atoms = [atom_types[perm[node]] for node in graph.nodes]
    shuffled = pec.build_sample(graph, shuffled_atoms, edge_types)
    assert sorted(shuffled.atom_idx.tolist()) == sorted(base.atom_idx.tolist())
    assert np.array_equal(base.node_basis, shuffled.node_basis)
    assert np.array_equal(base.edge_basis, shuffled.edge_basis)

    model = _model("sparse")
    batch = _batch([base, shuffled])
    env = model.encode_environments(batch)
    n_first = int(base.n_nodes)
    delta = (env[:n_first] - env[n_first:]).abs().max().item()
    assert delta > 0.0
    return {"chemistry_shuffle_changes_environment": float(delta)}


def check_environment_freeze() -> dict[str, object]:
    """E_i must be bit-identical when pair data or other molecules change."""
    first = _sample(_ring(6), seed=1)
    second = _sample(_chain(5), seed=2)
    model = _model("sparse")
    batch = _batch([first, second])
    env_before = model.encode_environments(batch)

    chain = _chain(5)
    mutated = pec.collate(
        [
            first,
            pec.build_sample(
                chain,
                [9] * 5,
                {chain.edge_key(u, v): 3 for u, v in chain.edges()},
            ),
        ]
    )
    mutated["pair_rho"] = torch.randn_like(mutated["pair_rho"])
    env_after = model.encode_environments(mutated)

    n_first = int(first.n_nodes)
    assert torch.equal(env_before[:n_first], env_after[:n_first])
    assert not torch.equal(env_before[n_first:], env_after[n_first:])
    return {"environment_freeze_bit_identical": True}


def check_static_contract() -> dict[str, object]:
    """No pair -> centre write-back: pairs are read once and never written back."""
    sample = _sample(_ring(6), seed=3)
    batch = _batch([sample])
    model = _model("sparse")

    calls = {"n": 0}

    def _count(_module, _inputs, _output):
        calls["n"] += 1

    handle = model.environment.register_forward_hook(_count)
    out = model(batch)
    handle.remove()
    assert calls["n"] == 1
    assert torch.isfinite(out["prediction"]).all()

    # removing the pair pass entirely must still yield a valid forward
    bag = _model("sparse", ablation="bag")
    out_bag = bag(batch)
    assert torch.isfinite(out_bag["prediction"]).all()

    # environment formation must not read pair tensors: mutating them changes
    # nothing about E_i
    env_a = model.encode_environments(batch)
    batch_b = dict(batch)
    batch_b["pair_rho"] = torch.randn_like(batch["pair_rho"])
    env_b = model.encode_environments(batch_b)
    assert torch.equal(env_a, env_b)
    return {"environment_module_called_once": True, "bag_forward_finite": True}


def check_relabel_invariance() -> dict[str, object]:
    graph = _ring(6)
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    base = pec.build_sample(
        graph, [atom_types[node] for node in graph.nodes], edge_types
    )
    perm = [2, 0, 5, 3, 1, 4]
    inverse = {old: new for new, old in enumerate(perm)}
    relabelled_edges = [(inverse[u], inverse[v]) for u, v in graph.edges()]
    relabelled = from_edges(len(perm), relabelled_edges)
    relabelled_atoms = [0] * len(perm)
    for old, new in inverse.items():
        relabelled_atoms[new] = atom_types[old]
    relabelled_types = {
        relabelled.edge_key(u, v): edge_types[graph.edge_key(perm[u], perm[v])]
        for u, v in relabelled.edges()
    }
    moved = pec.build_sample(relabelled, relabelled_atoms, relabelled_types)

    model = _model("sparse")
    out_base = model(_batch([base]))["prediction"]
    out_moved = model(_batch([moved]))["prediction"]
    delta = float((out_base - out_moved).abs().max())
    assert delta < 1.0e-5, delta
    return {"relabel_prediction_max_abs_diff": delta}


def check_sparse_correctness() -> dict[str, object]:
    sample = _sample(_ring(7), seed=4)
    model = _model("sparse")
    codes_node = model._node_coord(torch.as_tensor(sample.node_basis, dtype=torch.float32))
    codes_edge = model._edge_coord(torch.as_tensor(sample.edge_basis, dtype=torch.float32))
    l0_node = int((codes_node != 0).sum(dim=1).max())
    l0_edge = int((codes_edge != 0).sum(dim=1).max())
    assert l0_node <= pec.S_V, l0_node
    assert l0_edge <= pec.S_E, l0_edge
    return {"max_l0_node": l0_node, "max_l0_edge": l0_edge}


def check_gradients() -> dict[str, float]:
    sample = _sample(_ring(6), seed=5)
    batch = _batch([sample, _sample(_chain(4), seed=6)])
    model = _model("sparse")
    out = model(batch)
    loss = out["prediction"].pow(2).mean()
    loss.backward()
    report: dict[str, float] = {}
    for name, parameter in [
        ("d_node", model.d_node),
        ("d_edge", model.d_edge),
        ("environment", next(model.environment.parameters())),
        ("pair", next(model.pair.parameters())),
        ("reader", next(model.reader.parameters())),
    ]:
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        value = float(grad.abs().sum())
        assert value > 0.0, f"{name} gradient is exactly zero"
        report[name] = value
    del loss, out
    return report


def check_dense_matched_params() -> dict[str, float]:
    sparse = _model("sparse")
    dense = _model("dense")
    coarse = _model("coarse")
    return {
        "sparse": float(pec.n_params(sparse)),
        "dense": float(pec.n_params(dense)),
        "coarse": float(pec.n_params(coarse)),
    }




GATE0_CHECKS = {
    "purity": check_purity,
    "chemistry_isolation": check_chemistry_isolation,
    "assignment_sensitivity": check_assignment_sensitivity,
    "environment_freeze": check_environment_freeze,
    "static_contract": check_static_contract,
    "relabel_invariance": check_relabel_invariance,
    "sparse_correctness": check_sparse_correctness,
    "gradients": check_gradients,
}


def run_gate0() -> dict[str, object]:
    report: dict[str, object] = {}
    for name, function in GATE0_CHECKS.items():
        report[name] = function()
    report["_parameter_accounting"] = check_dense_matched_params()
    return report
