"""Focused CPU tests for E2E-DictEnv-A1 (no ZINC download, no GPU).

Covers the frozen pre-registration objects: the 433-D layout, hand-checkable toy
patch references, permutation/relabel semantics, the train-only scaler, the
arm-agnostic coder modes, model contracts and parameter accounting.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

RUNNER_PATH = Path(a1.__file__).with_name("zinc_e2e_dictenv_a1.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _molecule(n: int, edges: list[tuple[int, int]], atom_types: np.ndarray | None = None,
              bond_types: np.ndarray | None = None) -> dict[str, object]:
    graph = from_edges(n, edges)
    atom_types = np.arange(n) % a1.ATOM_CATEGORIES if atom_types is None else np.asarray(atom_types)
    if bond_types is None:
        bond_types = np.arange(len(edges)) % a1.BOND_CATEGORIES
    edge_types = {
        graph.edge_key(int(a), int(b)): int(bond_types[index])
        for index, (a, b) in enumerate(edges)
    }
    return {
        "graph": graph,
        "atom_types": np.asarray(atom_types, dtype=np.int64),
        "bond_types": np.asarray(bond_types, dtype=np.int64),
        "edge_types": edge_types,
        "edges": [(int(a), int(b)) for a, b in edges],
    }


def _path_molecule(n: int) -> dict[str, object]:
    return _molecule(n, [(index, index + 1) for index in range(n - 1)])


def _synthetic_data(n: int = 4, *, seed: int = 0, dictionary: np.ndarray | None = None):
    """Synthetic single-molecule batch with a real A1-shaped ``dict_phi``."""
    molecule = _path_molecule(n)
    graph, edge_types = molecule["graph"], molecule["edge_types"]
    incidence = v0.env_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    data = Data()
    data.num_nodes = int(n)
    width = a1.A1_DIM if dictionary is None else int(dictionary.shape[0])
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, width)).astype(np.float32))
    data.dict_atom = torch.as_tensor(rng.integers(0, a1.ATOM_CATEGORIES, size=n), dtype=torch.long)
    data.env_occ_node = incidence["occ_node"]
    data.env_occ_root = incidence["occ_root"]
    data.env_occ_shell = incidence["occ_shell"]
    data.env_bond_root = incidence["bond_root"]
    data.env_bond_shellpair = incidence["bond_shellpair"]
    data.env_bond_type = incidence["bond_type"]
    data.env_bond_u = incidence["bond_u"]
    data.env_bond_v = incidence["bond_v"]
    data.anchor = p1.build_anchor_raw(
        data.dict_atom,
        data.env_occ_node,
        data.env_occ_root,
        data.env_bond_root,
        data.env_bond_type,
        int(n),
    )
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    data.pair_index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).T)
    data.pair_relation = torch.randn(len(pairs), p1.RELATION_WIDTH_RAW)
    data.pair_bucket = torch.as_tensor(rng.integers(0, p1.DISTANCE_BUCKETS, size=len(pairs)), dtype=torch.long)
    data.global_context = torch.randn(1, p1.GLOBAL_WIDTH)
    data.topology_features = torch.randn(1, p1.TOPOLOGY_HIDDEN + 9)
    data.y = torch.tensor([0.5])
    return data


def _batch(n: int = 4, *, seed: int = 0, dictionary: np.ndarray | None = None):
    return p1.env_collate([_synthetic_data(n, seed=seed, dictionary=dictionary)])


def _probe_dictionary(dim: int = a1.A1_DIM, seed: int = 0) -> np.ndarray:
    return sdb.random_normalized_dictionary(dim, a1.DICT_K, seed).astype(np.float32)


# ---------------------------------------------------------------------------
# frozen protocol constants
# ---------------------------------------------------------------------------


def test_frozen_protocol_constants():
    assert a1.PROTOCOL_VERSION == "e2e_dictenv_a1"
    assert a1.A1_DIM == 433
    assert a1.JOINT_V_DIM == 308 and a1.JOINT_E_DIM == 60
    assert a1.DICT_K == 32 and a1.DICT_S == 8
    assert a1.IHT_CANDIDATE_STEPS == (10, 30, 100, 200)
    assert a1.IHT_QUALIFY_MAX_REC == 0.002
    assert a1.MATERIAL == 0.003
    assert a1.MECHANISM_THRESHOLD == 0.010
    assert a1.CONTINUITY_AUC_PASS == 0.70
    assert a1.HORIZON == 320
    assert a1.LAMBDA_REC == 33.95873017865987
    assert a1.ARMS == ("TOPO", "INDEP", "REAL")
    assert a1.ARM_COORDINATE == {"TOPO": "phi", "INDEP": "indep", "REAL": "real"}


def test_block_layout_is_contiguous_and_named():
    assert a1.BLOCK_SLICES["S"] == slice(0, 65)
    assert a1.BLOCK_SLICES["V"] == slice(65, 373)
    assert a1.BLOCK_SLICES["E"] == slice(373, 433)
    assert a1.ARM_INPUT_DIM == {"TOPO": 65, "INDEP": 433, "REAL": 433}


# ---------------------------------------------------------------------------
# hand-checkable toy patch reference
# ---------------------------------------------------------------------------


def test_toy_patch_hand_reference():
    """Hand-computed reference for the 3-atom path ``0 - 1 - 2`` rooted at 1."""
    molecule = _molecule(3, [(0, 1), (1, 2)], atom_types=np.asarray([2, 5, 2]), bond_types=np.asarray([1, 3]))
    blocks = a1.patch_blocks(molecule["graph"], 1, molecule["atom_types"], molecule["edge_types"])
    assert blocks["n_patch"] == 3 and blocks["m_patch"] == 2
    assert list(blocks["nodes"]) == [0, 1, 2]
    # node basis: [root] [shell-0, shell-1, shell-2] [degree] [nbr x 3] [walk x 3]
    assert np.array_equal(blocks["node_basis"][:, 0], np.asarray([0.0, 1.0, 0.0]))
    assert np.array_equal(blocks["node_basis"][:, 1:4], np.asarray([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
    assert np.allclose(blocks["node_basis"][:, 4], np.log1p(np.asarray([1.0, 2.0, 1.0])))
    # q/r are the chemistry one-hots in patch node order / sorted edge order
    assert np.array_equal(blocks["q"], a1.one_hot_rows(np.asarray([2, 5, 2]), a1.ATOM_CATEGORIES))
    assert np.array_equal(blocks["r"], a1.one_hot_rows(np.asarray([1, 3]), a1.BOND_CATEGORIES))
    node_basis, q, edge_basis, r = blocks["node_basis"], blocks["q"], blocks["edge_basis"], blocks["r"]
    assert np.array_equal(blocks["joint_v"], node_basis.T @ q)
    assert np.array_equal(blocks["marginal_v"], np.outer(node_basis.sum(0), q.sum(0)) / 3.0)
    assert np.array_equal(blocks["joint_e"], edge_basis.T @ r)
    assert np.array_equal(blocks["marginal_e"], np.outer(edge_basis.sum(0), r.sum(0)) / 2.0)
    # explicit hand value: the root row carries atom type 5 -> J^V[0, 5] == 1
    assert blocks["joint_v"][0, 5] == 1.0
    assert blocks["marginal_v"][0, 5] == 1.0 / 3.0
    # and the independence null is exactly rank 1 by construction
    assert np.linalg.matrix_rank(blocks["marginal_v"]) == 1


def test_toy_zero_edge_patch():
    molecule = _molecule(1, [], atom_types=np.asarray([7]))
    blocks = a1.patch_blocks(molecule["graph"], 0, molecule["atom_types"], molecule["edge_types"])
    assert blocks["m_patch"] == 0
    assert blocks["joint_e"].shape == (a1.EDGE_BASIS_DIM, a1.BOND_CATEGORIES)
    assert np.all(blocks["joint_e"] == 0.0)
    assert np.all(blocks["marginal_e"] == 0.0)
    assert np.isfinite(blocks["marginal_v"]).all()
    assert blocks["marginal_v"][0, 7] == 1.0


def test_one_hot_rows_out_of_range_is_zero():
    out = a1.one_hot_rows(np.asarray([-1, 0, 2, 99]), 4)
    assert np.array_equal(
        out,
        np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float64),
    )


def test_patch_blocks_is_deterministic():
    molecule = _path_molecule(5)
    left = a1.patch_blocks(molecule["graph"], 2, molecule["atom_types"], molecule["edge_types"])
    right = a1.patch_blocks(molecule["graph"], 2, molecule["atom_types"], molecule["edge_types"])
    for key in ("joint_v", "joint_e", "marginal_v", "marginal_e", "node_basis", "edge_basis"):
        assert np.array_equal(left[key], right[key]), key


# ---------------------------------------------------------------------------
# invariance / assignment semantics
# ---------------------------------------------------------------------------


def test_edge_order_and_endpoint_swap_invariance():
    molecule = _path_molecule(4)
    swapped_graph = from_edges(4, [(1, 0), (2, 1), (3, 2)])
    swapped_edges = {
        swapped_graph.edge_key(int(a), int(b)): int(bond_type)
        for (a, b), bond_type in zip([(1, 0), (2, 1), (3, 2)], molecule["bond_types"])
    }
    for root in range(4):
        base = a1.patch_blocks(molecule["graph"], root, molecule["atom_types"], molecule["edge_types"])
        swapped = a1.patch_blocks(swapped_graph, root, molecule["atom_types"], swapped_edges)
        assert np.array_equal(base["joint_v"], swapped["joint_v"]), root
        assert np.array_equal(base["joint_e"], swapped["joint_e"]), root
        assert np.array_equal(base["marginal_e"], swapped["marginal_e"]), root


def test_node_relabel_invariance_of_blocks():
    molecule = _path_molecule(5)
    permutation = np.asarray([3, 0, 4, 1, 2])
    inverse = np.argsort(permutation)
    relabelled = from_edges(5, [(int(permutation[a]), int(permutation[b])) for a, b in molecule["edges"]])
    relabelled_edges = {
        relabelled.edge_key(int(permutation[a]), int(permutation[b])): int(bond_type)
        for (a, b), bond_type in zip(molecule["edges"], molecule["bond_types"])
    }
    relabelled_atoms = molecule["atom_types"][inverse]
    for root in range(5):
        base = a1.patch_blocks(molecule["graph"], root, molecule["atom_types"], molecule["edge_types"])
        moved = a1.patch_blocks(relabelled, int(permutation[root]), relabelled_atoms, relabelled_edges)
        assert np.array_equal(base["joint_v"], moved["joint_v"]), root
        assert np.array_equal(base["joint_e"], moved["joint_e"]), root
        assert np.array_equal(base["marginal_v"], moved["marginal_v"]), root
        assert np.array_equal(base["marginal_e"], moved["marginal_e"]), root


def test_within_patch_attribute_permutation_real_changes_indep_does_not():
    molecule = _molecule(4, [(0, 1), (1, 2), (2, 3)], atom_types=np.asarray([1, 2, 3, 4]))
    blocks = a1.patch_blocks(molecule["graph"], 1, molecule["atom_types"], molecule["edge_types"])
    order = np.asarray([2, 0, 3, 1])
    q = blocks["q"][order]
    moved_v, _ = a1.joint_from_basis(blocks["node_basis"], blocks["edge_basis"], q, blocks["r"])
    indep_v, _ = a1.marginal_from_basis(blocks["node_basis"], blocks["edge_basis"], q, blocks["r"])
    assert not np.allclose(blocks["joint_v"], moved_v)
    assert np.array_equal(blocks["marginal_v"], indep_v)
    assert np.array_equal(q.sum(0), blocks["q"].sum(0))
    r = blocks["r"][np.asarray([2, 0, 1])]
    _, moved_e = a1.joint_from_basis(blocks["node_basis"], blocks["edge_basis"], blocks["q"], r)
    _, indep_e = a1.marginal_from_basis(blocks["node_basis"], blocks["edge_basis"], blocks["q"], r)
    assert not np.allclose(blocks["joint_e"], moved_e)
    assert np.array_equal(blocks["marginal_e"], indep_e)


def test_marginal_is_the_assignment_independent_analytic_null():
    rng = np.random.default_rng(0)
    node_basis = rng.standard_normal((7, a1.NODE_BASIS_DIM))
    edge_basis = rng.standard_normal((9, a1.EDGE_BASIS_DIM))
    q = a1.one_hot_rows(rng.integers(0, a1.ATOM_CATEGORIES, size=7), a1.ATOM_CATEGORIES)
    r = a1.one_hot_rows(rng.integers(0, a1.BOND_CATEGORIES, size=9), a1.BOND_CATEGORIES)
    reference_v, reference_e = a1.marginal_from_basis(node_basis, edge_basis, q, r)
    for _ in range(5):
        moved_v, moved_e = a1.marginal_from_basis(
            node_basis, edge_basis, q[rng.permutation(7)], r[rng.permutation(9)]
        )
        np.testing.assert_array_equal(moved_v, reference_v)
        np.testing.assert_array_equal(moved_e, reference_e)


def test_joint_and_marginal_share_the_same_marginals():
    molecule = _path_molecule(4)
    blocks = a1.patch_blocks(molecule["graph"], 2, molecule["atom_types"], molecule["edge_types"])
    for key, dim in (("v", a1.NODE_BASIS_DIM), ("e", a1.EDGE_BASIS_DIM)):
        joint = blocks[f"joint_{key}"].reshape(dim, -1).sum(1)
        marginal = blocks[f"marginal_{key}"].reshape(dim, -1).sum(1)
        np.testing.assert_allclose(joint, marginal, rtol=1e-12, atol=1e-12)


def test_basis_builder_matches_the_audited_fsar_v2_builder():
    from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2

    molecule = _path_molecule(5)
    graph = molecule["graph"]
    for root in range(5):
        blocks = a1.patch_blocks(graph, root, molecule["atom_types"], molecule["edge_types"])
        nodes, _index, node_basis, edge_basis, edges = v2._explicit_basis_for_patch(graph, root)
        assert list(blocks["nodes"]) == list(nodes)
        assert np.array_equal(blocks["node_basis"], node_basis)
        assert np.array_equal(blocks["edge_basis"], edge_basis)
        assert [(int(a), int(b)) for a, b in blocks["edges"]] == [(int(a), int(b)) for a, b in edges]


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def test_block_scaler_train_only_rms_and_mask():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((500, a1.JOINT_V_DIM))
    train[:, 3] = 0.0
    scaler = a1.fit_block_scaler("V", train)
    rms = np.sqrt((train ** 2).mean(0))
    hand_mask = (rms > a1.SCALER_FLOOR).astype(np.float64)
    hand_scale = np.where(hand_mask > 0.0, np.sqrt((train ** 2).mean(0) + a1.SCALER_EPS), 1.0)
    np.testing.assert_array_equal(scaler.scale, hand_scale)
    np.testing.assert_array_equal(scaler.mask, hand_mask)
    assert scaler.masked_coordinates == 1
    raw = {
        "phi": rng.standard_normal((500, a1.PHI_DIM)),
        "joint_v": train,
        "joint_e": np.zeros((500, a1.JOINT_E_DIM)),
    }
    full = a1.fit_object_scaler("real", raw)
    x = a1.apply_object_scaler(full, raw)
    block = full.blocks["V"]
    expected = (train / block.scale[None, :]) * block.mask[None, :] * block.weight
    np.testing.assert_allclose(x[:, a1.BLOCK_SLICES["V"]], expected, rtol=1e-12)
    # the zero-RMS coordinate must be masked out, not divided by the 1e-9 floor
    assert np.all(x[:, a1.BLOCK_SLICES["V"]][:, 3] == 0.0)


def test_zero_rms_coordinate_is_masked_not_divided():
    train = np.ones((10, 4))
    train[:, 1] = 0.0
    scaler = a1.fit_block_scaler("S", train)
    assert scaler.scale[1] > 0.0
    assert scaler.mask[1] == 0.0
    assert np.isfinite(scaler.scale).all()


def test_real_and_indep_scaling_procedure_is_identical():
    rng = np.random.default_rng(3)
    raw = {
        "phi": rng.standard_normal((64, a1.PHI_DIM)),
        "joint_v": rng.standard_normal((64, a1.JOINT_V_DIM)),
        "joint_e": rng.standard_normal((64, a1.JOINT_E_DIM)),
        "marginal_v": rng.standard_normal((64, a1.JOINT_V_DIM)),
        "marginal_e": rng.standard_normal((64, a1.JOINT_E_DIM)),
    }
    real = a1.fit_object_scaler("real", raw)
    indep = a1.fit_object_scaler("indep", raw)
    np.testing.assert_array_equal(real.blocks["S"].scale, indep.blocks["S"].scale)
    np.testing.assert_array_equal(real.blocks["S"].weight, indep.blocks["S"].weight)
    x_real = a1.apply_object_scaler(real, raw)
    x_indep = a1.apply_object_scaler(indep, raw)
    assert x_real.shape == x_indep.shape == (64, a1.A1_DIM)
    # identical S block, per-arm V/E scaler/weight (no cross-arm sharing)
    np.testing.assert_array_equal(x_real[:, a1.BLOCK_SLICES["S"]], x_indep[:, a1.BLOCK_SLICES["S"]])
    assert not np.array_equal(x_real[:, a1.BLOCK_SLICES["V"]], x_indep[:, a1.BLOCK_SLICES["V"]])
    for scaler in (real, indep):
        for name in a1.BLOCKS:
            block = scaler.blocks[name]
            assert block.block_energy * block.weight ** 2 == pytest.approx(1.0, rel=1e-6)


def test_concatenate_blocks_layout():
    rng = np.random.default_rng(1)
    raw = {
        "phi": rng.standard_normal((7, a1.PHI_DIM)),
        "joint_v": np.full((7, a1.JOINT_V_DIM), 1.0),
        "joint_e": np.full((7, a1.JOINT_E_DIM), 2.0),
        "marginal_v": np.full((7, a1.JOINT_V_DIM), 3.0),
        "marginal_e": np.full((7, a1.JOINT_E_DIM), 4.0),
    }
    real = a1.concatenate_blocks(raw, "real")
    indep = a1.concatenate_blocks(raw, "indep")
    assert real.shape == indep.shape == (7, a1.A1_DIM)
    np.testing.assert_array_equal(real[:, a1.BLOCK_SLICES["V"]], raw["joint_v"])
    np.testing.assert_array_equal(real[:, a1.BLOCK_SLICES["E"]], raw["joint_e"])
    np.testing.assert_array_equal(indep[:, a1.BLOCK_SLICES["V"]], raw["marginal_v"])
    np.testing.assert_array_equal(indep[:, a1.BLOCK_SLICES["E"]], raw["marginal_e"])
    np.testing.assert_array_equal(real[:, a1.BLOCK_SLICES["S"]], indep[:, a1.BLOCK_SLICES["S"]])
    with pytest.raises(ValueError):
        a1.concatenate_blocks(raw, "topo")


def test_scaler_rejects_wrong_block_width():
    rng = np.random.default_rng(0)
    raw = {
        "phi": rng.standard_normal((4, a1.PHI_DIM)),
        "joint_v": rng.standard_normal((4, a1.JOINT_V_DIM)),
        "joint_e": rng.standard_normal((4, a1.JOINT_E_DIM)),
    }
    scaler = a1.fit_object_scaler("real", raw)
    with pytest.raises(RuntimeError):
        a1.apply_block_scaler(scaler, np.zeros((4, a1.PHI_DIM)), np.zeros((4, 7)), np.zeros((4, a1.JOINT_E_DIM)))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_parameter_accounting_matches_preregistration():
    topo = a1.total_parameter_count("TOPO")
    real = a1.total_parameter_count("REAL")
    indep = a1.total_parameter_count("INDEP")
    assert topo["dictionary"] == 65 * 32
    assert topo["whole_model"] == 97487
    assert real["dictionary"] == 433 * 32
    assert real["whole_model"] == 109263
    assert real["local_subtotal"] == 94160
    assert real["backend_subtotal"] == 15103
    assert indep["whole_model"] == real["whole_model"]
    for arm in a1.ARMS:
        model = a1.build_model(arm, _probe_dictionary(a1.ARM_INPUT_DIM[arm]), seed=0)
        assert sum(parameter.numel() for parameter in model.parameters()) == topo_check(arm)
        assert a1.total_parameter_count(arm)["within_budget"] is True


def topo_check(arm: str) -> int:
    return int(a1.total_parameter_count(arm)["whole_model"])


def test_parameter_breakdown_equals_p2_backbone_plus_dictionary_delta():
    baseline = p2.parameter_breakdown(p2.P2Config("A1_TOPO", "h1", a1.DICE_D_E, 32, 8, 0.25, a1.HORIZON, "a1"))
    topo = a1.parameter_breakdown("TOPO")
    assert baseline["dictionary"] == topo["dictionary"] == 65 * 32
    for key in ("node_binding", "edge_binding", "decoder", "backend_subtotal"):
        assert baseline[key] == topo[key]
    assert a1.parameter_breakdown("REAL")["dictionary"] - topo["dictionary"] == (433 - 65) * 32
    assert a1.parameter_breakdown("INDEP") == a1.parameter_breakdown("REAL")


def test_model_bit_identical_to_p2_h1_backbone():
    dictionary = _probe_dictionary()
    batch = _batch(4, dictionary=dictionary)
    torch.manual_seed(0)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    config = p2.P2Config("Z_H1", "h1", a1.DICE_D_E, 32, 8, 0.25, a1.HORIZON, "a1")
    reference = p2.P2Model(config, dictionary)
    reference.load_state_dict(model.state_dict())
    reference.eval()
    with torch.no_grad():
        left, aux_left = model(batch, return_aux=True)
        right, aux_right = reference(batch, return_aux=True)
    assert torch.equal(left, right)
    assert torch.equal(aux_left["E"], aux_right["E"])


def test_omp_frozen_mode_requires_precomputed_codes():
    dictionary = _probe_dictionary()
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_OMP_FROZEN, seed=0).eval()
    phi = torch.as_tensor(np.random.default_rng(0).standard_normal((3, a1.A1_DIM)), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        model.code(phi)
    codes = torch.zeros(3, a1.DICT_K)
    codes[:, : a1.DICT_S] = 0.25
    np.testing.assert_allclose(model.code(phi, codes).numpy(), codes.numpy())
    batch = _batch(3, dictionary=dictionary)
    batch.precomputed_coord = codes
    with torch.no_grad():
        _prediction, aux = model(batch, return_aux=True)
    assert torch.equal(aux["coord"], codes)


def test_coding_modes_and_exact_sparsity():
    dictionary = _probe_dictionary()
    batch = _batch(6, dictionary=dictionary)
    Dbar = v0.normalized_dictionary(torch.as_tensor(dictionary))
    iht = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    dense = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_DENSE_TIED, seed=0).eval()
    with torch.no_grad():
        alpha_iht = iht.code(batch.dict_phi)
        alpha_dense = dense.code(batch.dict_phi)
    assert int((alpha_iht.abs() > 0).sum(1).max()) <= a1.DICT_S
    np.testing.assert_allclose(alpha_dense.numpy(), (batch.dict_phi @ Dbar).numpy(), rtol=1e-5, atol=1e-6)


def test_iht_reconstruction_never_worse_than_zero_code():
    dictionary = _probe_dictionary()
    batch = _batch(8, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=60, seed=0).eval()
    with torch.no_grad():
        alpha = model.code(batch.dict_phi)
        reconstruction = model.reconstruct(batch.dict_phi, alpha)
    residual = ((batch.dict_phi - reconstruction) ** 2).sum(1)
    trivial = (batch.dict_phi ** 2).sum(1)
    assert bool((residual <= trivial).all())


def test_reconstruction_loss_formula():
    dictionary = _probe_dictionary()
    batch = _batch(4, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_DENSE_TIED, seed=0).eval()
    with torch.no_grad():
        alpha = model.code(batch.dict_phi)
        loss = model.reconstruction_loss(batch.dict_phi, alpha)
    hand = (
        ((batch.dict_phi - alpha @ v0.normalized_dictionary(model.D).t()) ** 2).sum(1)
        / ((batch.dict_phi ** 2).sum(1) + v0.EPS)
    ).mean()
    assert float(loss) == pytest.approx(float(hand), rel=1e-6)


def test_forward_shapes_and_sparsity():
    dictionary = _probe_dictionary()
    batch = _batch(5, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    with torch.no_grad():
        prediction, aux = model(batch, return_aux=True)
    assert prediction.shape == (1,)
    assert aux["E"].shape == (5, p1.ENV_DIM)
    assert aux["coord"].shape == (5, a1.DICT_K)
    assert aux["phi"].shape == (5, a1.A1_DIM)
    assert int((aux["coord"].abs() > 0).sum(1).max()) <= a1.DICT_S


def test_forward_uses_dict_phi_as_dictionary_input_only():
    dictionary = _probe_dictionary()
    batch = _batch(4, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    with torch.no_grad():
        codes = model.code(batch.dict_phi)
        _p0, aux = model(batch, return_aux=True)
    assert torch.equal(aux["coord"], codes)
    assert aux["coord"].shape[1] == a1.DICT_K


def test_coord_zero_and_role_shuffle_interventions_are_effective():
    dictionary = _probe_dictionary()
    batch = _batch(4, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    with torch.no_grad():
        _p0, aux_base = model(batch, return_aux=True)
        _p1, aux_zero = model(batch, coord_zero=True, return_aux=True)
        _p2, aux_node = model(batch, occ_coord_node=torch.roll(batch.env_occ_node, 1, dims=0), return_aux=True)
        _p3, aux_bond = model(batch, bond_u=torch.roll(batch.env_bond_u, 1, dims=0), return_aux=True)
    assert not torch.equal(aux_base["E"], aux_zero["E"])
    assert not torch.equal(aux_base["E"], aux_node["E"])
    assert not torch.equal(aux_base["E"], aux_bond["E"])
    # role shuffles must not move the chemistry rows
    assert torch.equal(aux_base["coord"], aux_node["coord"])
    assert torch.equal(aux_base["coord"], aux_bond["coord"])


def test_unknown_forward_kwargs_are_rejected():
    dictionary = _probe_dictionary()
    batch = _batch(2, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, seed=0).eval()
    with pytest.raises(TypeError):
        model(batch, silent_no_op=True)


def test_dictionary_width_is_guarded_by_the_arm():
    with pytest.raises(RuntimeError):
        a1.build_model("REAL", _probe_dictionary(65), seed=0)
    with pytest.raises(RuntimeError):
        a1.build_model("TOPO", _probe_dictionary(a1.A1_DIM), seed=0)


def test_environment_freeze_against_pair_chemistry():
    dictionary = _probe_dictionary()
    batch = _batch(4, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, coding_mode=a1.CODING_IHT, iht_steps=10, seed=0).eval()
    mutated = batch.clone()
    mutated.pair_relation = torch.randn_like(mutated.pair_relation) * 3.0
    with torch.no_grad():
        _p0, aux_base = model(batch, return_aux=True)
        _p1, aux_mutated = model(mutated, return_aux=True)
    assert torch.equal(aux_base["E"], aux_mutated["E"])


def test_no_pair_to_centre_in_forward():
    dictionary = _probe_dictionary()
    batch = _batch(3, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, seed=0).eval()
    original = zpp.PatchPathModel._pool_pairs_to_centres
    calls = {"count": 0}

    def _guard(*_args, **_kwargs):
        calls["count"] += 1
        raise RuntimeError("pair->centre is forbidden")

    zpp.PatchPathModel._pool_pairs_to_centres = _guard
    try:
        with torch.no_grad():
            model(batch)
    finally:
        zpp.PatchPathModel._pool_pairs_to_centres = original
    assert calls["count"] == 0


def test_poisoned_legacy_descriptors_are_ignored():
    dictionary = _probe_dictionary()
    batch = _batch(3, dictionary=dictionary)
    model = a1.build_model("REAL", dictionary, seed=0).eval()
    poisoned = batch.clone()
    poisoned.patch_cont = torch.randn(3, 146)
    poisoned.atom_shell = torch.randn(3, 3)
    poisoned.bond_shell = torch.randn(3, 3)
    with torch.no_grad():
        clean = model(batch)
        dirty = model(poisoned)
    assert torch.equal(clean, dirty)


def test_no_forbidden_names_in_a1_source():
    forbidden = {
        "patch_cont",
        "atom_shell",
        "bond_shell",
        "path_bond_mean",
        "adjacent_bond_type",
        "canonical_slot",
        "patch_slot_order",
    }
    tree = ast.parse(Path(a1.__file__).read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not (names & forbidden)


# ---------------------------------------------------------------------------
# runner-level contracts
# ---------------------------------------------------------------------------


def test_runner_never_requests_the_official_test_split():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    string_literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "test" not in string_literals
    assert "test" not in {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    # every split literal used by the runner is train/valid only
    assert {"train", "valid"} <= string_literals


def test_runner_records_official_test_loaded_false_everywhere():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    flags = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value is False
    ]
    assert flags  # the provenance payload always pins the flag to False
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert text.count('"official_test_loaded": False') >= 10


def test_runner_arm_order_is_topo_indep_real():
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as runner

    assert tuple(runner.a1.ARMS) == ("TOPO", "INDEP", "REAL")


def test_gpu_policy_requires_physical_gpu1(monkeypatch):
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as runner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError):
        runner._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError):
        runner._set_device_policy("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    device = runner._set_device_policy("cpu")
    assert device.type == "cpu"


def test_stage_list_is_complete_and_ordered():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    expected = (
        "cache",
        "scaler",
        "dictionaries",
        "omp-codes",
        "correctness",
        "assignment",
        "health",
        "continuity",
        "posthoc-continuity",
        "omp-screen",
        "select-omp",
        "iht-diag",
        "formal",
        "mechanism",
        "liveness",
        "specificity",
        "accounting",
        "decision",
        "report",
        "all",
    )
    choices = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not (isinstance(node.args[0], ast.Constant) and node.args[0].value == "stage"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(keyword.value, ast.Tuple):
                choices = tuple(element.value for element in keyword.value.elts)
    assert choices == expected
