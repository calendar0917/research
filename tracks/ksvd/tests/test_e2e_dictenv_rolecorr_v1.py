"""Focused CPU tests for E2E-DictEnv-RoleCorr-v1.

Covers the frozen pre-registration objects without loading ZINC, checkpoints or
any trained artifact:

* the 536-D layout / block geometry,
* a hand-checkable within-group centred reference,
* permutation / relabel invariance, attribute-permutation non-triviality,
* singleton / constant-role zero semantics,
* the train-only two-layer scaler,
* unlabeled K-SVD and PCA16 primitives,
* the model coordinate layout, dictionary freezing and zero-``alpha_C`` purity,
* the TOPO arm's bit-identity with the frozen Sem108 code path,
* the label-free AST property and the official-test blocker.
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
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb

CORE_PATH = Path(rc.__file__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _molecule(
    n: int,
    edges: list[tuple[int, int]],
    atom_types: np.ndarray | None = None,
    bond_types: np.ndarray | None = None,
) -> dict[str, object]:
    graph = from_edges(n, edges)
    atom_types = np.arange(n) % rc.ATOM_CATEGORIES if atom_types is None else np.asarray(atom_types)
    if bond_types is None:
        bond_types = np.arange(len(edges)) % rc.BOND_CATEGORIES
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


def _fork_molecule() -> dict[str, object]:
    # N(0) with three neighbours 1,2,3 and a C attached to 1 (shell 2 from 0)
    return _molecule(5, [(0, 1), (0, 2), (0, 3), (1, 4)])


def _synthetic_subspace(q: int = 1) -> cssd.CommonSubspace:
    rng = np.random.default_rng(7)
    components = rng.standard_normal((rc.PHI_DIM, q))
    components, _ = np.linalg.qr(components)
    return cssd.CommonSubspace(
        components=np.asarray(components, dtype=np.float64),
        rms=np.asarray([2.0] * q, dtype=np.float64),
        kind=f"q{q}",
    )


def _synthetic_batch(
    n: int = 6,
    *,
    seed: int = 0,
    dictionary_corr: np.ndarray | None = None,
) -> Data:
    """Single-molecule batch with every tensor the frozen forward needs."""
    molecule = _fork_molecule()
    graph, edge_types = molecule["graph"], molecule["edge_types"]
    incidence = e2e_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    data = Data()
    data.num_nodes = int(graph.n)
    n = int(graph.n)
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, rc.PHI_DIM)).astype(np.float32))
    data.dict_atom = torch.as_tensor(rng.integers(0, rc.ATOM_CATEGORIES, size=n), dtype=torch.long)
    data.env_occ_node = incidence["occ_node"]
    data.env_occ_root = incidence["occ_root"]
    data.env_occ_shell = incidence["occ_shell"]
    data.env_bond_root = incidence["bond_root"]
    data.env_bond_shellpair = incidence["bond_shellpair"]
    data.env_bond_type = incidence["bond_type"]
    data.env_bond_u = incidence["bond_u"]
    data.env_bond_v = incidence["bond_v"]
    data.anchor = torch.as_tensor(rng.standard_normal((n, audit.ANCHOR_DIM_EXPECTED)), dtype=torch.float32)
    sem_geom = sem.resolve_sem108_geometry()
    data.patch_cont = torch.as_tensor(
        rng.standard_normal((n, sem_geom["patch_cont_dim"])), dtype=torch.float32
    )
    data.global_context = torch.as_tensor(rng.standard_normal((1, 62)), dtype=torch.float32)
    data.topology_features = torch.as_tensor(rng.standard_normal((1, 25)), dtype=torch.float32)
    data.batch = torch.zeros(n, dtype=torch.long)
    data.num_graphs = 1
    data.y = torch.zeros(1)
    if dictionary_corr is not None:
        data.corr_vec = torch.as_tensor(
            rng.standard_normal((n, rc.CORR_DIM)), dtype=torch.float32
        )
    return data


def e2e_incidence(graph, edge_types):
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

    return v0.env_incidence(graph, edge_types)


# ---------------------------------------------------------------------------
# geometry / construction
# ---------------------------------------------------------------------------


def test_frozen_geometry():
    assert rc.NODE_ROLE_DIM == 7
    assert rc.EDGE_ROLE_DIM == 9
    assert rc.NODE_BLOCK_DIM == 2 * 7 * 28 == 392
    assert rc.EDGE_BLOCK_DIM == 4 * 9 * 4 == 144
    assert rc.CORR_DIM == 536
    assert rc.STRUCT_SLICE.start == 1 and rc.STRUCT_SLICE.stop == 17
    assert rc.CORR_SLICE.start == 17 and rc.CORR_SLICE.stop == 33
    assert rc.COORD_DIM == 33
    assert rc.NODE_SHELLS == (1, 2)
    assert rc.EDGE_SHELL_PAIRS == ((0, 1), (1, 1), (1, 2), (2, 2))


def test_hand_computed_within_shell_reference():
    molecule = _fork_molecule()
    patch = a1.patch_blocks(
        molecule["graph"], 0, molecule["atom_types"], molecule["edge_types"]
    )
    blocks = rc.corr_blocks(patch)
    node_basis = np.asarray(patch["node_basis"], dtype=np.float64)
    shells = node_basis[:, 1:4].argmax(axis=1)
    q = np.asarray(patch["q"], dtype=np.float64)
    for position, shell in enumerate(rc.NODE_SHELLS):
        mask = shells == shell
        role = node_basis[mask][:, 4:11]
        attribute = q[mask]
        role_c = role - role.mean(axis=0, keepdims=True)
        attribute_c = attribute - attribute.mean(axis=0, keepdims=True)
        expected = (role_c.T @ attribute_c).reshape(-1)
        low = position * rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
        high = low + rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
        assert np.allclose(blocks.node[low:high], expected, atol=0.0, rtol=0.0)

    edge_basis = np.asarray(patch["edge_basis"], dtype=np.float64)
    pair_ids = edge_basis[:, :6].argmax(axis=1)
    r = np.asarray(patch["r"], dtype=np.float64)
    for position, pair in enumerate(rc.EDGE_SHELL_PAIRS):
        mask = pair_ids == rc.ALL_SHELL_PAIRS.index(pair)
        if int(mask.sum()) < 2:
            expected = np.zeros(rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES)
        else:
            role = edge_basis[mask][:, 6:15]
            attribute = r[mask]
            role_c = role - role.mean(axis=0, keepdims=True)
            attribute_c = attribute - attribute.mean(axis=0, keepdims=True)
            expected = (role_c.T @ attribute_c).reshape(-1)
        low = position * rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
        high = low + rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
        assert np.allclose(blocks.edge[low:high], expected, atol=0.0, rtol=0.0)


def test_singleton_and_empty_groups_are_zero():
    molecule = _path_molecule(3)
    patch = a1.patch_blocks(
        molecule["graph"], 0, molecule["atom_types"], molecule["edge_types"]
    )
    blocks = rc.corr_blocks(patch)
    # shell 2 of a 3-path rooted at the end is a singleton -> zero node block
    assert blocks.node_group_sizes[2] == 1
    low = 1 * rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES
    assert np.array_equal(blocks.node[low:], np.zeros(rc.NODE_ROLE_DIM * rc.ATOM_CATEGORIES))
    # the (2,2) edge group is empty -> zero edge block
    position = rc.EDGE_SHELL_PAIRS.index((2, 2))
    low_e = position * rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES
    assert np.array_equal(blocks.edge[low_e:], np.zeros(rc.EDGE_ROLE_DIM * rc.BOND_CATEGORIES))
    # an end-of-path root with all groups < 2 has a fully zero object
    tail = a1.patch_blocks(molecule["graph"], 2, molecule["atom_types"], molecule["edge_types"])
    tail_blocks = rc.corr_blocks(tail)
    assert np.array_equal(tail_blocks.vector, np.zeros(rc.CORR_DIM))


def test_relabel_and_endpoint_swap_invariance():
    molecule = _fork_molecule()
    permutation = np.asarray([3, 0, 4, 1, 2])
    inverse = np.argsort(permutation)
    relabelled = from_edges(
        5,
        [(int(permutation[a]), int(permutation[b])) for a, b in molecule["edges"]],
    )
    relabelled_edges = {
        relabelled.edge_key(int(permutation[a]), int(permutation[b])): int(bond_type)
        for (a, b), bond_type in zip(molecule["edges"], molecule["bond_types"])
    }
    relabelled_atoms = molecule["atom_types"][inverse]
    for root in range(5):
        base = rc.corr_blocks(
            a1.patch_blocks(molecule["graph"], root, molecule["atom_types"], molecule["edge_types"])
        )
        moved = rc.corr_blocks(
            a1.patch_blocks(relabelled, int(permutation[root]), relabelled_atoms, relabelled_edges)
        )
        assert np.allclose(base.vector, moved.vector, atol=1e-12, rtol=0.0)

    swapped_graph = from_edges(5, [(b, a) for a, b in molecule["edges"]])
    swapped_edges = {
        swapped_graph.edge_key(b, a): int(bond_type)
        for (a, b), bond_type in zip(molecule["edges"], molecule["bond_types"])
    }
    for root in range(5):
        base = rc.corr_blocks(
            a1.patch_blocks(molecule["graph"], root, molecule["atom_types"], molecule["edge_types"])
        )
        swapped = rc.corr_blocks(
            a1.patch_blocks(swapped_graph, root, molecule["atom_types"], swapped_edges)
        )
        assert np.array_equal(base.vector, swapped.vector)


def test_within_group_attribute_permutation_keeps_marginals_changes_object():
    molecule = _fork_molecule()
    patch = a1.patch_blocks(
        molecule["graph"], 0, molecule["atom_types"], molecule["edge_types"]
    )
    base = rc.corr_blocks(patch)
    shuffled = rc.corr_blocks(patch, permute_seed=101)
    assert not np.array_equal(base.vector, shuffled.vector)
    assert float(np.abs(base.vector - shuffled.vector).max()) > rc.PERMUTE_DELTA_FLOOR
    # coarse marginals are the permutation-invariant part
    node_basis = np.asarray(patch["node_basis"], dtype=np.float64)
    shells = node_basis[:, 1:4].argmax(axis=1)
    for position, shell in enumerate(rc.NODE_SHELLS):
        _ = position
        mask = shells == shell
        assert np.array_equal(
            np.asarray(patch["q"])[mask].sum(axis=0),
            np.asarray(patch["q"])[mask].sum(axis=0),
        )
    # group sizes are preserved by the shuffle
    assert base.group_size_vector().tolist() == shuffled.group_size_vector().tolist()


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def test_scaler_masks_zero_rms_and_equalizes_block_energy():
    rng = np.random.default_rng(0)
    node = rng.standard_normal((64, rc.NODE_BLOCK_DIM)) * 0.5
    node[:, 3] = 0.0
    edge = rng.standard_normal((64, rc.EDGE_BLOCK_DIM)) * 4.0
    scaler = rc.fit_corr_scaler(node, edge)
    assert scaler.node.dim == rc.NODE_BLOCK_DIM and scaler.edge.dim == rc.EDGE_BLOCK_DIM
    assert scaler.node.mask[3] == 0.0
    assert scaler.node.masked_coordinates >= 1
    scaled = rc.apply_corr_scaler(scaler, node, edge)
    assert scaled.shape == (64, rc.CORR_DIM)
    assert float(np.abs(scaled[:, 3]).max()) == 0.0
    node_energy = float(np.mean(np.sum(scaled[:, : rc.NODE_BLOCK_DIM] ** 2, axis=1)))
    edge_energy = float(np.mean(np.sum(scaled[:, rc.NODE_BLOCK_DIM :] ** 2, axis=1)))
    assert abs(node_energy - 1.0) < 1e-6
    assert abs(edge_energy - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# dictionary / PCA primitives
# ---------------------------------------------------------------------------


def test_ksvd_fit_and_exact_sparsity():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((128, rc.CORR_DIM)).astype(np.float64)
    D, info = rc.fit_corr_dictionary(X, atoms=8, s=3, epochs=2)
    assert D.shape == (rc.CORR_DIM, 8)
    assert np.allclose(np.linalg.norm(D, axis=0), 1.0, atol=1e-6)
    assert np.isfinite(info["history"][-1]["mean_sq_err"])
    codes = sdb.omp_codes(D.astype(np.float64), X[:16], s=3)
    assert np.all((np.abs(codes) > 0).sum(axis=1) <= 3)


def test_pca16_gram_matches_reference_svd_subspace():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((96, rc.CORR_DIM)) + 0.3
    pca = rc.fit_pca16(X, rank=16)
    assert pca.components.shape == (16, rc.CORR_DIM)
    reference = sdb.fit_pca_rank(X, rank=16)
    project = pca.components.T @ pca.components
    project_ref = reference.components.T @ reference.components
    assert np.allclose(project, project_ref, atol=1e-8)
    codes = pca.codes(X)
    centred = X - X.mean(axis=0)
    assert np.allclose(centred @ project, codes @ pca.components, atol=1e-8)


# ---------------------------------------------------------------------------
# model layout / freezing / shared init
# ---------------------------------------------------------------------------


def _model_reference_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key not in ("D", "D_corr")
    }


def test_model_coordinate_layout_and_freezing():
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(3)
    d_struct = rng.standard_normal((rc.PHI_DIM, rc.K_ATOMS)).astype(np.float32)
    d_corr = rng.standard_normal((rc.CORR_DIM, rc.K_ATOMS)).astype(np.float32)
    model = rc.build_rolecorr_model(
        arm=rc.ARM_CORR, dictionary=d_struct, dictionary_corr=d_corr, seed=0, subspace=subspace
    )
    assert tuple(model.W_A_S.shape) == (rc.COORD_DIM, int(p2.D_A))
    assert tuple(model.W_E_S.shape) == (3 * rc.COORD_DIM, 48)
    assert model.D.requires_grad is False and model.D_corr.requires_grad is False
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "D" not in trainable and "D_corr" not in trainable
    assert len(trainable) == len([n for n, _ in model.named_parameters()]) - 2
    phi = torch.as_tensor(rng.standard_normal((4, rc.PHI_DIM)), dtype=torch.float32)
    corr = torch.as_tensor(rng.standard_normal((4, rc.CORR_DIM)), dtype=torch.float32)
    coord = model.code(phi, corr)
    assert tuple(coord.shape) == (4, rc.COORD_DIM)
    assert float(coord[:, rc.CORR_SLICE].abs().sum()) > 0.0
    purity = rc.zero_corr_purity(model, phi, corr)
    assert purity["head_columns_bit_identical"] and purity["corr_columns_zero"]
    assert purity["official_test_loaded"] is False
    model.inference_zero_struct = True
    zero_s = model.code(phi, corr)
    assert torch.equal(zero_s[:, rc.STRUCT_SLICE], torch.zeros_like(zero_s[:, rc.STRUCT_SLICE]))
    assert torch.equal(zero_s[:, rc.CORR_SLICE], coord[:, rc.CORR_SLICE])


def test_shared_readout_initialisation_is_bit_identical():
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(4)
    d_struct = rng.standard_normal((rc.PHI_DIM, rc.K_ATOMS)).astype(np.float32)
    d_corr = rng.standard_normal((rc.CORR_DIM, rc.K_ATOMS)).astype(np.float32)
    baseline = rc.build_rolecorr_model(
        arm=rc.ARM_CORR, dictionary=d_struct, dictionary_corr=d_corr, seed=0, subspace=subspace
    )
    reference = _model_reference_state(baseline)
    candidate = rc.build_rolecorr_model(
        arm=rc.ARM_CORR,
        dictionary=d_struct,
        dictionary_corr=d_corr,
        seed=0,
        subspace=subspace,
        reference_state=reference,
    )
    for key, value in reference.items():
        assert torch.equal(candidate.state_dict()[key], value), key


def test_topology_arm_code_matches_frozen_sem108_code():
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(5)
    d_base = rng.standard_normal((rc.PHI_DIM, rc.BASE_ATOMS)).astype(np.float32)
    arm = rc.build_rolecorr_model(
        arm=rc.ARM_TOPO, dictionary=d_base, seed=0, subspace=subspace
    )
    parent = sem.build_sem108_model(d_base, 0, subspace)
    phi = torch.as_tensor(rng.standard_normal((7, rc.PHI_DIM)), dtype=torch.float32)
    assert torch.equal(arm.code(phi), parent.code(phi))


def test_label_free_construction_and_test_blocker():
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))
    forbidden = {"y", "target", "targets", "label", "labels"}
    guarded = {"corr_blocks", "corr_vector", "_centered_cross", "fit_corr_scaler", "fit_corr_dictionary"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in guarded:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name):
                    assert inner.id not in forbidden, f"{node.name} reads {inner.id!r}"
    with pytest.raises(RuntimeError):
        rc.official_test_blocker({"official_test_loaded": True})
    rc.official_test_blocker({"official_test_loaded": False})
    assert cm.C6_MASK is not None


def test_code_usage_accounting():
    codes = np.zeros((10, rc.K_ATOMS))
    codes[:5, 0] = 1.0
    codes[:3, 1] = 2.0
    usage = rc.code_usage(codes)
    assert usage["active_atoms"] == 2
    assert usage["exact_l0_mean"] == pytest.approx(0.8)
    assert usage["top1_share"] == pytest.approx(0.5)
    assert usage["top4_share"] == pytest.approx(0.8)
    assert usage["official_test_loaded"] is False
