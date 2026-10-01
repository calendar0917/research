"""Targeted checks for E2E-DictEnv-Hier-Relation-v1.

Synthetic (data-free) checks cover the frozen algebra, the grouping /
permutation semantics, the intervention branch purity and the dictionary
gradient liveness.  The data-backed checks are skipped when the frozen prepare
artifacts are absent (fresh clone), and never touch the official test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_hier_relation_v1 as core
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_hier_relation_v1 as run

RESULTS_DIR = run.RESULTS_DIR
CACHE_DIR = run.CACHE_DIR


# ---------------------------------------------------------------------------
# data-free algebra
# ---------------------------------------------------------------------------


def test_layout_constants_and_parameter_budget():
    assert core.D_X == 712 == core.MAX_INPUT_DIM
    assert core.JOINT_DIM == 709 and core.EXTRA_SIZE_DIM == 3
    assert core.REL_INPUT_DIM == 228
    assert core.REL_BLOCKS == {
        "mu": (0, 96),
        "delta": (96, 192),
        "mu_delta": (192, 224),
        "bond": (224, 228),
    }
    assert core.READOUT_DIM == 586
    assert core.NODE_POOL_DIM + core.REL_POOL_DIM + core.COUNT_DIM + core.TOPO_OUT == 586
    assert core.REL_SPARSITY == 8
    assert core.MESSAGE_PASSING is False and core.DEPTH == 1 and core.NODE_WRITE_BACK is False
    model = core.HierRelationModel()
    report = core.parameter_report(model)
    assert report["trainable"] == report["expected_trainable"] == 128 * core.D_X + 54825
    assert report["trainable"] == 145961
    assert report["matches_expected"]


def test_projection_is_fixed_plus_minus_one_over_sqrt():
    projection = core.build_projection()
    assert tuple(projection.shape) == (core.NODE_ATOMS, core.PROJ_DIM)
    unique = torch.unique(projection)
    assert float(unique.abs().max()) == pytest.approx(1.0 / math.sqrt(core.PROJ_DIM), rel=1e-6)
    assert int(unique.numel()) == 2
    again = core.build_projection()
    assert torch.equal(projection, again)
    assert core.projection_sha256(projection) == core.projection_sha256(again)
    # an independent generator would give a different tensor
    assert core.projection_sha256(core.build_projection(seed=1)) != core.projection_sha256(projection)


def test_dedup_edges_counts_each_physical_key_once():
    # the same physical key appears once per rooted patch that contains it
    u = np.array([0, 1, 0, 2, 1, 0], dtype=np.int64)
    v = np.array([1, 0, 1, 1, 0, 1], dtype=np.int64)
    t = np.array([1, 1, 1, 2, 1, 1], dtype=np.int64)
    lo, hi, bond_class, stats = run._dedup_edges(u, v, t, 3)
    assert stats["occurrences"] == 6 and stats["unique"] == 2
    assert lo.tolist() == [0, 1] and hi.tolist() == [1, 2]
    assert bond_class.tolist() == [1, 2]
    assert bool(np.all(lo < hi))
    with pytest.raises(RuntimeError):
        run._dedup_edges(u, v, np.array([1, 2, 1, 2, 1, 1]), 3)
    with pytest.raises(RuntimeError):
        run._dedup_edges(np.array([0]), np.array([3]), np.array([1]), 3)


def test_relation_object_is_endpoint_swap_invariant():
    torch.manual_seed(0)
    m = torch.randn(5, core.PROJ_DIM)
    d = torch.randn(5, core.PROJ_DIM)
    u = torch.tensor([0, 1, 2, 3])
    v = torch.tensor([1, 2, 3, 4])
    bond = torch.tensor([1, 3, 1, 2])
    forward = core.relation_objects(m[u], d[u], m[v], d[v], bond)
    reverse = core.relation_objects(m[v], d[v], m[u], d[u], bond)
    assert forward.shape == (4, core.REL_INPUT_DIM)
    assert torch.equal(forward, reverse)
    for block, (lo, hi) in core.REL_BLOCKS.items():
        assert hi - lo in (96, 32, 4), block


def test_scaler_from_sums_matches_reference_fit():
    rng = np.random.default_rng(0)
    block = rng.normal(size=(127, 6))
    block[:, 4] = 0.0  # a zero-RMS coordinate must be masked
    slices = {"a": (0, 3), "b": (3, 5), "c": (5, 6)}
    reference = core.fit_frozen_scaler("toy", slices, block)
    sums = np.sum(block.astype(np.float64) ** 2, axis=0)
    from_sums = run._fit_scaler_from_sums("toy", slices, sums, block.shape[0])
    assert np.allclose(reference.scale, from_sums.scale, atol=1e-12)
    assert np.array_equal(reference.mask, from_sums.mask)
    assert np.allclose(reference.weight, from_sums.weight, atol=1e-12)
    assert np.allclose(reference.gain, from_sums.gain, atol=1e-12)
    assert int(np.sum(from_sums.mask == 0.0)) == 1
    energy = core.scaled_block_energy(block, from_sums)
    assert all(abs(value - 1.0) < 1e-9 for value in energy.values())


def test_second_moment_basis_rank_detection_and_signs():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(64, 5))
    matrix = latent @ rng.normal(size=(5, 40))
    basis, info = core.second_moment_basis(matrix, 12, seed=0)
    assert tuple(basis.shape) == (40, 12)
    assert info["effective_rank"] == 5
    assert info["supplemented_columns"] == 7
    assert info["orthonormality_max_error"] < 1e-10
    for index in range(basis.shape[1]):
        column = basis[:, index]
        assert column[int(np.argmax(np.abs(column)))] > 0.0
    full, info_full = core.second_moment_basis(matrix, 5, seed=0)
    assert info_full["supplemented_columns"] == 0
    assert float(np.abs(full.T @ full - np.eye(5)).max()) < 1e-6  # returned as float32


# ---------------------------------------------------------------------------
# synthetic batch (grouping / permutation / pooling / forward)
# ---------------------------------------------------------------------------


def synthetic_batch(seed: int = 0) -> core.HierBatch:
    torch.manual_seed(seed)
    counts = [7, 9, 6, 8]
    graphs = len(counts)
    nodes = int(sum(counts))
    graph_id = torch.repeat_interleave(torch.arange(graphs), torch.tensor(counts))
    atom_type = torch.tensor(
        [0, 0, 1, 2, 2, 2, 3] + [1, 1, 4, 4, 4, 5, 5, 5, 6] + [0, 7, 7, 8, 8, 8]
        + [1, 1, 2, 2, 2, 3, 3, 3],
        dtype=torch.long,
    )
    eu: list[int] = []
    ev: list[int] = []
    et: list[int] = []
    eg: list[int] = []
    offset = 0
    for graph, count in enumerate(counts):
        for step in range(count - 1):
            eu.append(offset + step)
            ev.append(offset + step + 1)
            et.append(1 + (step % 3))
            eg.append(graph)
        if count > 3:
            eu.append(offset)
            ev.append(offset + 3)
            et.append(2)
            eg.append(graph)
        offset += count
    return core.HierBatch(
        x=torch.randn(nodes, core.D_X) * 0.4,
        graph_id=graph_id,
        atom_type=atom_type,
        edge_index=torch.tensor([eu, ev], dtype=torch.long),
        edge_type=torch.tensor(et, dtype=torch.long),
        edge_graph=torch.tensor(eg, dtype=torch.long),
        topology=torch.randn(graphs, core.TOPO_IN),
        y=torch.randn(graphs),
        mol_index=torch.arange(graphs),
        n_graphs=graphs,
    )


def test_grouping_never_crosses_graphs_and_singleton_delta_is_zero():
    batch = synthetic_batch()
    torch.manual_seed(1)
    z = torch.randn(batch.x.shape[0], core.NODE_ATOMS)
    mu, delta = core.shared_and_deviation(z, batch.graph_id, batch.atom_type, batch.n_graphs)
    key = core.group_key(batch.graph_id, batch.atom_type, batch.n_graphs)
    # every group mean equals the mean of exactly the rows sharing (graph, type)
    for value in torch.unique(key).tolist():
        rows = torch.nonzero(key == value, as_tuple=False).flatten()
        assert torch.allclose(mu[rows], z[rows].mean(dim=0), atol=1e-6)
    counts = torch.zeros(int(batch.n_graphs) * core.ATOM_CATEGORIES, dtype=torch.long)
    counts = counts.index_add(0, key, torch.ones_like(key))
    singleton = counts[key] == 1
    assert bool(singleton.any())
    assert float(delta[singleton].abs().max()) == 0.0
    # an unrelated graph cannot change another graph's group mean
    other = core.shared_and_deviation(
        torch.cat([z, torch.randn(5, core.NODE_ATOMS)], dim=0),
        torch.cat([batch.graph_id, torch.full((5,), 3)]),
        torch.cat([batch.atom_type, torch.zeros(5, dtype=torch.long)]),
        batch.n_graphs,
    )[0][: z.shape[0]]
    assert torch.allclose(mu, other, atol=1e-6)


def test_in_group_permutation_is_bijection_and_preserves_group_means():
    batch = synthetic_batch()
    key = core.group_key(batch.graph_id, batch.atom_type, batch.n_graphs)
    for seed in core.REL_PERM_SEEDS:
        perm = core.in_group_permutation(batch.graph_id, batch.atom_type, batch.n_graphs, seed)
        assert sorted(perm.tolist()) == list(range(int(batch.x.shape[0])))
        assert torch.equal(key, key.index_select(0, perm))  # stays inside the group
        torch.manual_seed(2)
        z = torch.randn(batch.x.shape[0], core.NODE_ATOMS)
        before, _ = core.shared_and_deviation(z, batch.graph_id, batch.atom_type, batch.n_graphs)
        after, _ = core.shared_and_deviation(
            z.index_select(0, perm), batch.graph_id, batch.atom_type, batch.n_graphs
        )
        assert float((before - after).abs().max()) <= 1e-6  # BLAS reduction order
        if int(batch.x.shape[0]) > 0:
            assert not torch.equal(z, z.index_select(0, perm)) or perm.tolist() == list(
                range(int(batch.x.shape[0]))
            )


def test_pooling_empty_graph_is_zero_and_std_is_epsilon_floor():
    values = torch.randn(4, 3)
    index = torch.tensor([0, 0, 2, 2])
    pooled = core.pool_sum_mean_std(values, index, 3)
    assert tuple(pooled.shape) == (3, 9)
    assert float(pooled[1].abs().max()) == 0.0  # empty graph
    constant = torch.ones(3, 2)
    index_all = torch.zeros(3, dtype=torch.long)
    out = core.pool_sum_mean_std(constant, index_all, 1)
    assert float(out[0, 0:2].max()) == 3.0
    assert float(out[0, 2:4].max()) == 1.0
    assert float(out[0, 4:6].max()) == pytest.approx(math.sqrt(core.STD_EPS), rel=1e-6)


def test_forward_shapes_reconstruction_and_sparse_budget():
    batch = synthetic_batch()
    model = core.HierRelationModel().eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out["pred"].shape) == (batch.n_graphs,)
    assert tuple(out["z"].shape) == (batch.x.shape[0], core.NODE_ATOMS)
    assert tuple(out["r_scaled"].shape) == (batch.edge_type.numel(), core.REL_INPUT_DIM)
    assert tuple(out["features"].shape) == (batch.n_graphs, core.READOUT_DIM)
    dbar = model.node_dictionary
    hand_x = model.node_input(batch.x)
    hand_z = hand_x @ dbar
    hand_hat = hand_z @ dbar.t()
    assert torch.allclose(hand_z, out["z"], atol=1e-6)
    assert torch.allclose(hand_hat, out["x_hat"], atol=1e-6)
    hand_r_node = ((hand_x - hand_hat) ** 2).mean() / (hand_x**2).mean()
    assert abs(float(hand_r_node) - float(out["r_node"])) < 1e-6
    hand_r_rel = ((out["r_scaled"] - out["r_hat"]) ** 2).mean() / (out["r_scaled"] ** 2).mean()
    assert abs(float(hand_r_rel) - float(out["r_rel"])) < 1e-6
    l0 = (out["beta"] != 0).sum(dim=1)
    assert int(l0.max()) <= core.REL_SPARSITY
    # every column of both dictionaries is unit norm
    for dictionary in (model.node_dictionary, model.relation_dictionary):
        assert torch.allclose(
            dictionary.norm(dim=0), torch.ones(dictionary.shape[1]), atol=1e-5
        )
    report = core.parameter_report(model)
    assert report["trainable"] == 145961


def test_loss_weights_and_empty_edge_batch_has_zero_relation_term():
    batch = synthetic_batch()
    model = core.HierRelationModel().eval()
    loss, parts = model.loss(batch)
    expected = parts["task"] + core.LAMBDA_NODE * parts["r_node"] + core.LAMBDA_REL * parts["r_rel"]
    assert abs(float(loss.detach()) - float(expected)) < 1e-6
    empty = core.HierBatch(
        x=torch.randn(3, core.D_X),
        graph_id=torch.zeros(3, dtype=torch.long),
        atom_type=torch.zeros(3, dtype=torch.long),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        edge_graph=torch.zeros(0, dtype=torch.long),
        topology=torch.randn(1, core.TOPO_IN),
        y=torch.randn(1),
        mol_index=torch.zeros(1, dtype=torch.long),
        n_graphs=1,
    )
    with torch.no_grad():
        out = model(empty)
    assert float(out["r_rel"]) == 0.0
    assert bool(torch.isfinite(out["pred"]).all())
    assert float(out["rel_pool"].abs().max()) == 0.0
    assert float(out["counts"][0, 1]) == 0.0


def test_endpoint_permutation_changes_only_the_relation_branch():
    batch = synthetic_batch()
    model = core.HierRelationModel().eval()
    with torch.no_grad():
        base = model(batch)
        permuted = model(batch, mode="permute_endpoints", perm_seed=core.REL_PERM_SEEDS[0])
    assert torch.equal(base["z"], permuted["z"])
    assert torch.equal(base["x_hat"], permuted["x_hat"])
    assert torch.equal(base["node_pool"], permuted["node_pool"])
    assert torch.equal(base["mu"], permuted["mu"])
    assert torch.equal(base["delta"], permuted["delta"])
    assert torch.equal(base["r_node"], permuted["r_node"])
    assert not torch.equal(base["r_scaled"], permuted["r_scaled"])
    assert not torch.equal(base["pred"], permuted["pred"])
    with pytest.raises(core.ContractViolation):
        model(batch, mode="permute_endpoints")


def test_interventions_touch_only_their_branch():
    batch = synthetic_batch()
    model = core.HierRelationModel().eval()
    with torch.no_grad():
        base = model(batch)
        zero_beta = model(batch, mode="zero_beta")
        zero_node = model(batch, mode="zero_node_pool")
        zero_delta = model(batch, mode="zero_delta_rel")
    assert torch.equal(base["node_pool"], zero_beta["node_pool"])
    assert float(zero_beta["beta"].abs().max()) == 0.0
    assert float(zero_beta["rel_pool"][:, : 2 * core.REL_ATOMS].abs().max()) == 0.0
    assert float(zero_beta["rel_pool"][:, 2 * core.REL_ATOMS :].max()) <= math.sqrt(
        core.STD_EPS
    ) * 1.001
    assert torch.equal(base["features"][:, core.NODE_POOL_DIM :], zero_node["features"][:, core.NODE_POOL_DIM :])
    assert torch.equal(base["r_node"], zero_node["r_node"])
    assert torch.equal(base["rel_pool"], zero_node["rel_pool"])
    assert torch.equal(base["mu"], zero_delta["mu"])
    assert torch.equal(base["node_pool"], zero_delta["node_pool"])
    assert float(zero_delta["r_scaled"][:, 96:224].abs().max()) == 0.0
    assert torch.equal(base["r_scaled"][:, :96], zero_delta["r_scaled"][:, :96])
    assert torch.equal(base["r_scaled"][:, 224:], zero_delta["r_scaled"][:, 224:])
    with pytest.raises(core.ContractViolation):
        model(batch, mode="not_a_mode")


def test_mae_only_gradient_reaches_both_dictionaries_and_moves_directions():
    batch = synthetic_batch()
    model = core.HierRelationModel()
    dbar_node = model.node_dictionary
    dbar_rel = model.relation_dictionary
    out = model.forward_with_dictionaries(batch, dbar_node, dbar_rel)
    mae = torch.nn.functional.l1_loss(out["pred"].view(-1), batch.y.view(-1))
    grads = torch.autograd.grad(mae, [model.D_node, model.D_rel, dbar_node, dbar_rel])
    for gradient in grads:
        assert gradient is not None and bool(torch.isfinite(gradient).all())
    assert float(grads[0].norm()) > 0.0
    assert float(grads[1].norm()) > 0.0
    assert float(grads[2].norm()) > 0.0  # gradient into the normalised node dictionary
    assert float(grads[3].norm()) > 0.0  # gradient into the normalised relation dictionary
    probe = run._update_probe(model, batch, steps=1)
    assert probe["directions_updated"]
    assert probe["moved_not_rotated_node"] and probe["moved_not_rotated_rel"]


def test_batch_composition_and_slot_offsets_are_consistent():
    struct = _synthetic_struct()
    first = struct.batch(torch.arange(0, 2, dtype=torch.long))
    second = struct.batch(torch.arange(2, 4, dtype=torch.long))
    whole = struct.batch(torch.arange(0, 4, dtype=torch.long))
    assert torch.equal(torch.cat([first.x, second.x]), whole.x)
    assert torch.equal(torch.cat([first.y, second.y]), whole.y)
    assert int(whole.edge_type.numel()) == int(first.edge_type.numel()) + int(
        second.edge_type.numel()
    )
    # edge endpoints must be local to the batch and inside it
    for batch in (first, second, whole):
        assert int(batch.edge_index.min()) >= 0
        assert int(batch.edge_index.max()) < int(batch.x.shape[0])
        assert int(batch.edge_graph.max()) < batch.n_graphs
    # a non-contiguous slot list must give the same molecules
    reordered = struct.batch(torch.tensor([2, 0], dtype=torch.long))
    assert torch.equal(reordered.y, struct.y[torch.tensor([2, 0])])


def _synthetic_struct() -> core.SplitTensors:
    torch.manual_seed(3)
    counts = [4, 5, 6, 3]
    nodes = int(sum(counts))
    node_ptr = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.long)
    eu: list[int] = []
    ev: list[int] = []
    et: list[int] = []
    edge_ptr = [0]
    offset = 0
    for count in counts:
        for step in range(count - 1):
            eu.append(offset + step)
            ev.append(offset + step + 1)
            et.append(1 + step % 2)
        edge_ptr.append(len(eu))
        offset += count
    return core.SplitTensors(
        x=torch.randn(nodes, core.D_X),
        node_ptr=node_ptr,
        atom_type=torch.randint(0, core.ATOM_CATEGORIES, (nodes,)),
        edge_index=torch.tensor([eu, ev], dtype=torch.long),
        edge_type=torch.tensor(et, dtype=torch.long),
        edge_ptr=torch.tensor(edge_ptr, dtype=torch.long),
        topology=torch.randn(len(counts), core.TOPO_IN),
        y=torch.randn(len(counts)),
        mol_id=torch.arange(len(counts)),
        meta={},
    )


# ---------------------------------------------------------------------------
# data-backed identity checks (skipped without the frozen prepare artifacts)
# ---------------------------------------------------------------------------


def _prepared() -> bool:
    return (
        (RESULTS_DIR / "hier_init_state.pt").exists()
        and (RESULTS_DIR / "hier_node_input.json").exists()
        and (CACHE_DIR / "hier_node_input_train.pt").exists()
        and (CACHE_DIR / "hier_struct_train.pt").exists()
    )


@pytest.mark.skipif(not _prepared(), reason="prepare artifacts absent")
def test_prepared_identity_and_real_batch_shapes():
    node_input = run._read_json(RESULTS_DIR / "hier_node_input.json")
    structure = run._read_json(RESULTS_DIR / "hier_structure.json")
    init = run._read_json(RESULTS_DIR / "hier_init.json")
    assert node_input["splits"]["train"]["rows"] == 231664
    assert node_input["splits"]["train"]["dim"] == core.D_X
    assert structure["splits"]["train"]["n_nodes"] == 231664
    assert structure["splits"]["train"]["n_edges"] > 0
    assert structure["splits"]["train"]["edge_occurrence_ratio"] > 1.0  # occurrences != edges
    assert init["node_dictionary"]["kept_coordinates"] == core.NODE_ATOMS
    assert init["projection"]["trainable"] is False
    assert init["official_test_loaded"] is False
    train = run.load_struct("train")
    assert int(train.x.shape[1]) == core.D_X
    batch = train.batch(torch.arange(8, dtype=torch.long))
    model = run.build_model(initialised=True).eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out["features"].shape) == (8, core.READOUT_DIM)
    assert int(out["beta"].shape[1]) == core.REL_ATOMS
    assert int((out["beta"] != 0).sum(dim=1).max()) <= core.REL_SPARSITY
    assert bool(torch.isfinite(out["pred"]).all())
    # the batch must contain exactly the molecules' physical edges, once each
    packed = batch.edge_index[0].numpy().astype(np.int64) * 10 ** 7 + batch.edge_index[1].numpy()
    assert len(set(packed.tolist())) == len(packed)
    assert not bool((batch.edge_index[0] >= batch.edge_index[1]).any())
    assert run._env_blob is not None
    assert not (run.TRACK_ROOT / "results/e2e_dictenv_p1/cache/env_test.pt").exists()


@pytest.mark.skipif(not _prepared(), reason="prepare artifacts absent")
def test_init_state_matches_the_frozen_record():
    init = run._read_json(RESULTS_DIR / "hier_init.json")
    state = run._load_init_state()
    assert core.state_sha256(state) == str(init["init_state"]["state_sha256"])
    model = run.build_model(initialised=True)
    assert core.state_sha256(model.state_dict()) == str(init["init_state"]["state_sha256"])
    assert core.projection_sha256(model.projection) == str(init["projection"]["sha256_f32"])
    for key in ("projection", "node_input_mask", "extra_gain", "rel_gain"):
        assert key in state
    assert int(state["projection"].numel()) == core.NODE_ATOMS * core.PROJ_DIM
