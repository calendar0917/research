"""Focused CPU tests for E2E-DictEnv-v0 (no ZINC, no GPU)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e


# ---------------------------------------------------------------------------
# synthetic molecule builder
# ---------------------------------------------------------------------------


def _synthetic_data(n: int, edges: list[tuple[int, int]], *, seed: int = 0):
    graph = from_edges(n, edges)
    edge_types = {graph.edge_key(a, b): (index % e2e.BOND_CATEGORIES) for index, (a, b) in enumerate(edges)}
    incidence = e2e.env_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((n, e2e.PHI_DIM)).astype(np.float32)
    data = Data()
    data.num_nodes = n
    data.dict_phi = torch.as_tensor(phi)
    data.dict_atom = torch.as_tensor(rng.integers(0, e2e.ATOM_CATEGORIES, size=n), dtype=torch.long)
    data.env_scalars = torch.as_tensor(rng.standard_normal((n, 6)).astype(np.float32))
    data.env_occ_node = incidence["occ_node"]
    data.env_occ_root = incidence["occ_root"]
    data.env_occ_shell = incidence["occ_shell"]
    data.env_bond_root = incidence["bond_root"]
    data.env_bond_shellpair = incidence["bond_shellpair"]
    data.env_bond_type = incidence["bond_type"]
    n_pairs = n * (n - 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if pairs:
        data.pair_index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).T)
        data.pair_relation = torch.randn(n_pairs, e2e.RELATION_WIDTH)
        data.pair_bucket = torch.as_tensor(rng.integers(0, e2e.DISTANCE_BUCKETS, size=n_pairs), dtype=torch.long)
    else:
        data.pair_index = torch.empty((2, 0), dtype=torch.long)
        data.pair_relation = torch.empty((0, e2e.RELATION_WIDTH))
        data.pair_bucket = torch.empty((0,), dtype=torch.long)
    data.global_context = torch.randn(1, e2e.GLOBAL_WIDTH)
    data.topology_features = torch.randn(1, 25)
    data.y = torch.tensor([0.5])
    return data


def _synthetic_batch(n_molecules: int = 3):
    molecules = [
        _synthetic_data(n=4, edges=[(0, 1), (1, 2), (2, 3)], seed=index) for index in range(n_molecules)
    ]
    return e2e.env_collate(molecules)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_accounting_exact():
    local = e2e.local_parameter_count()
    backend = e2e.backend_parameter_count()
    total = e2e.total_parameter_count()
    assert local["dictionary"] == 2080
    assert local["W_R"] == 2112
    assert local["W_C"] == 1792
    assert local["S"] == 192
    assert local["W_B"] == 64
    assert local["P_shell"] == 96
    assert local["env_mlp"] == 44463
    assert local["subtotal"] == 50799
    assert backend["subtotal"] == 15359
    assert total["whole_model"] == 66158
    assert total["difference_vs_fec_s1"] == -12


def test_model_parameter_counts_and_init_matching():
    sparse = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    dense = e2e.build_model(
        e2e.DENSE_ARM, seed=0, reference_state=sparse.state_dict()
    )
    sp = sum(parameter.numel() for parameter in sparse.parameters())
    dp = sum(parameter.numel() for parameter in dense.parameters())
    assert sp == dp == 66158
    for key, value in sparse.state_dict().items():
        assert torch.equal(value, dense.state_dict()[key]), key


def test_dictionary_init_is_the_sdb_artifact():
    from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb

    D_sdb, _rand, _pca = zsdb.load_dictionary()
    sparse = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    assert np.array_equal(sparse.D.detach().numpy(), np.asarray(D_sdb, dtype=np.float32))


# ---------------------------------------------------------------------------
# incidence / shuffle
# ---------------------------------------------------------------------------


def test_env_incidence_path_graph():
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    edge_types = {(0, 1): 0, (1, 2): 1, (2, 3): 2}
    incidence = e2e.env_incidence(graph, edge_types)
    # occurrences: root0 {0,1,2}, root1 {0,1,2,3}, root2 {1,2,3,0}, root3 {2,3,1}
    assert incidence["occ_root"].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
    assert incidence["occ_node"].tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]
    assert incidence["occ_shell"].tolist() == [0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 1, 0]
    # exactly one occurrence per root is shell 0
    for root in range(4):
        mask = incidence["occ_root"] == root
        assert int((incidence["occ_shell"][mask] == 0).sum()) == 1
    # at least one bond occurrence per non-trivial root
    assert int(incidence["bond_root"].shape[0]) > 0


def test_shuffled_occ_node_preserves_multiset_and_groups():
    n, edges = 5, [(0, 1), (1, 2), (2, 3), (3, 4)]
    graph = from_edges(n, edges)
    edge_types = {graph.edge_key(a, b): 0 for a, b in edges}
    incidence = e2e.env_incidence(graph, edge_types)
    shuffled = e2e.shuffled_occ_node_for_molecule(
        incidence["occ_node"], incidence["occ_root"], incidence["occ_shell"], seed=7
    )
    assert sorted(shuffled.tolist()) == sorted(incidence["occ_node"].tolist())
    # within every (root, shell) group the node multiset is preserved
    groups = {}
    for index in range(len(shuffled)):
        groups.setdefault(
            (int(incidence["occ_root"][index]), int(incidence["occ_shell"][index])), []
        ).append(index)
    for positions in groups.values():
        before = sorted(incidence["occ_node"][positions].tolist())
        after = sorted(shuffled[positions].tolist())
        assert before == after


# ---------------------------------------------------------------------------
# coding operators
# ---------------------------------------------------------------------------


def test_tied_iht_exact_sparsity_and_determinism():
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    phi = torch.randn(32, e2e.PHI_DIM)
    alpha_a = model.code(phi)
    alpha_b = model.code(phi)
    assert torch.equal(alpha_a, alpha_b)
    l0 = (alpha_a.abs() > 0).sum(dim=1)
    assert int(l0.max()) <= e2e.SPARSITY


def test_dense_code_is_dense_and_matches_definition():
    model = e2e.build_model(e2e.DENSE_ARM, seed=0)
    phi = torch.randn(8, e2e.PHI_DIM)
    z = model.code(phi)
    dbar = e2e.normalized_dictionary(model.D)
    assert torch.allclose(z, phi @ dbar, atol=1e-6)


def test_reconstruction_is_tied():
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    phi = torch.randn(8, e2e.PHI_DIM)
    alpha = model.code(phi)
    rec = model.reconstruct(phi, alpha)
    dbar = e2e.normalized_dictionary(model.D)
    assert torch.allclose(rec, alpha @ dbar.t(), atol=1e-6)
    loss = model.reconstruction_loss(phi, alpha)
    assert torch.isfinite(loss) and float(loss) > 0.0


# ---------------------------------------------------------------------------
# purity / freeze / bypass on synthetic batches
# ---------------------------------------------------------------------------


def test_forward_shapes_and_environment_dim():
    batch = _synthetic_batch()
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    with torch.no_grad():
        prediction, aux = model(batch, return_aux=True)
    assert prediction.shape[0] == 3
    assert aux["E"].shape == (int(batch.num_nodes), e2e.ENV_DIM)
    assert aux["coord"].shape == (int(batch.num_nodes), e2e.K_ATOMS)


def test_environment_freeze_under_pair_relation_mutation():
    batch = _synthetic_batch(n_molecules=2)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    captured = {}
    handle = model.env_mlp.register_forward_hook(lambda _m, _i, out: captured.setdefault("E", out.detach().clone()))
    try:
        with torch.no_grad():
            model(batch)
        before = captured["E"].clone()
        mutated = batch.clone()
        mutated.pair_relation = torch.randn_like(mutated.pair_relation)
        with torch.no_grad():
            model(mutated)
        after = captured["E"].clone()
    finally:
        handle.remove()
    assert torch.equal(before, after)


def test_zero_code_is_independent_of_phi():
    batch = _synthetic_batch(n_molecules=2)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    with torch.no_grad():
        _p, aux_a = model(batch, coord_zero=True, return_aux=True)
        mutated = batch.clone()
        mutated.dict_phi = torch.randn_like(mutated.dict_phi)
        _p2, aux_b = model(mutated, coord_zero=True, return_aux=True)
        _p3, aux_c = model(mutated, coord_zero=False, return_aux=True)
    assert torch.equal(aux_a["E"], aux_b["E"])
    assert not torch.equal(aux_b["E"], aux_c["E"])


def test_pair_encoder_called_once():
    batch = _synthetic_batch(n_molecules=2)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).eval()
    counts = {"pair": 0, "relation": 0}

    def _hook(name):
        def _inner(_m, _i, _o):
            counts[name] += 1

        return _inner

    handles = [
        model.pair_encoder.register_forward_hook(_hook("pair")),
        model.relation_encoder.register_forward_hook(_hook("relation")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"pair": 1, "relation": 1}


def test_no_forbidden_modules_in_state_dict():
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0)
    keys = list(model.state_dict().keys())
    for key in keys:
        lowered = key.lower()
        assert "patch_encoder" not in lowered
        assert "typed" not in lowered
        assert "parent_embedding" not in lowered
        assert "local_env_adapter" not in lowered
        assert "attention" not in lowered
        assert "lstm" not in lowered and "gru" not in lowered


def test_gradient_reaches_dictionary():
    batch = _synthetic_batch(n_molecules=2)
    model = e2e.build_model(e2e.SPARSE_ARM, seed=0).train()
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    loss = torch.nn.functional.l1_loss(prediction, batch.y.view(-1))
    loss.backward()
    assert model.D.grad is not None
    assert float(model.D.grad.norm()) > 0.0


def test_env_collate_offsets_occurrences():
    molecules = [
        _synthetic_data(n=4, edges=[(0, 1), (1, 2), (2, 3)], seed=0),
        _synthetic_data(n=3, edges=[(0, 1), (1, 2)], seed=1),
    ]
    batch = e2e.env_collate(molecules)
    first = int(molecules[0].num_nodes)
    second_offset = first
    occ_first = int(molecules[0].env_occ_node.shape[0])
    # second molecule occurrence nodes must be shifted by the first node count
    assert int(batch.env_occ_node[occ_first:].min()) >= second_offset
    assert int(batch.env_occ_root.max()) == int(batch.num_nodes) - 1
