"""Focused CPU tests for E2E-DictEnv-RoleCorr-Increment-v2.

Covers the frozen pre-registration objects without loading ZINC, checkpoints or
any trained artifact:

* the 49-wide coordinate geometry and the binding-extension layout,
* the base K32/s8 coordinate's bit-identity with the frozen CSSD path,
* the appended structural block's bit-identity with the RoleCorr-v1 path,
* the sparse/dense block layout, zero-purity and shared readout initialisation,
* the distribution-preserving block shuffle,
* the route-2 joint scaler / dictionary / PCA geometry,
* the explicit device resolver and the official-test blocker.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_increment_v2 as inc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rolecorr_v1 as rc
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _incidence(graph, edge_types):
    return v0.env_incidence(graph, edge_types)


def _synthetic_subspace(seed: int = 7) -> cssd.CommonSubspace:
    rng = np.random.default_rng(seed)
    components = rng.standard_normal((inc.PHI_DIM, 1))
    components, _ = np.linalg.qr(components)
    return cssd.CommonSubspace(
        components=np.asarray(components, dtype=np.float64),
        rms=np.asarray([2.0], dtype=np.float64),
        kind="q1",
    )


def _synthetic_dictionaries(seed: int = 11):
    rng = np.random.default_rng(seed)
    return {
        "base": rng.standard_normal((inc.PHI_DIM, inc.BASE_ATOMS)).astype(np.float32),
        "extra": rng.standard_normal((inc.PHI_DIM, inc.EXTRA_ATOMS)).astype(np.float32),
        "corr": rng.standard_normal((inc.CORR_DIM, inc.EXTRA_ATOMS)).astype(np.float32),
        "joint": rng.standard_normal((inc.JOINT_DIM, inc.JOINT_ATOMS)).astype(np.float32),
        "pca_mean": rng.standard_normal(inc.CORR_DIM),
        "pca_components": rng.standard_normal((inc.EXTRA_ATOMS, inc.CORR_DIM)),
        "joint_mean": rng.standard_normal(inc.JOINT_DIM),
        "joint_components": rng.standard_normal((inc.JOINT_ATOMS, inc.JOINT_DIM)),
    }


def _synthetic_batch(seed: int = 0, corr: bool = True, joint: bool = True) -> Data:
    graph = from_edges(5, [(0, 1), (0, 2), (0, 3), (1, 4)])
    atom_types = np.arange(5) % rc.ATOM_CATEGORIES
    bond_types = np.arange(4) % rc.BOND_CATEGORIES
    edges = [(0, 1), (0, 2), (0, 3), (1, 4)]
    edge_types = {
        graph.edge_key(a, b): int(bond_types[i]) for i, (a, b) in enumerate(edges)
    }
    incidence = _incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    n = int(graph.n)
    data = Data()
    data.num_nodes = n
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, inc.PHI_DIM)).astype(np.float32))
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
    geometry = sem.resolve_sem108_geometry()
    data.patch_cont = torch.as_tensor(
        rng.standard_normal((n, geometry["patch_cont_dim"])), dtype=torch.float32
    )
    data.global_context = torch.as_tensor(rng.standard_normal((1, 62)), dtype=torch.float32)
    data.topology_features = torch.as_tensor(rng.standard_normal((1, 25)), dtype=torch.float32)
    data.batch = torch.zeros(n, dtype=torch.long)
    data.num_graphs = 1
    data.y = torch.zeros(1)
    data.pair_index = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 4]], dtype=torch.long)
    data.pair_relation = torch.as_tensor(rng.standard_normal((4, 23)), dtype=torch.float32)
    data.pair_bucket = torch.as_tensor(rng.integers(0, 5, size=4), dtype=torch.long)
    if corr:
        data.corr_vec = torch.as_tensor(
            rng.standard_normal((n, inc.CORR_DIM)), dtype=torch.float32
        )
    if joint:
        data.joint_vec = torch.as_tensor(
            rng.standard_normal((n, inc.JOINT_DIM)), dtype=torch.float32
        )
    return data


def _build_all_arms(seed: int = 0):
    dictionaries = _synthetic_dictionaries()
    subspace = _synthetic_subspace()
    reference = None
    models = {}
    for arm in inc.ROUTE1_ARMS:
        kwargs: dict[str, object] = {}
        if arm == inc.ARM_EXTRA_STRUCT:
            kwargs["block_dictionary"] = dictionaries["extra"]
        elif arm == inc.ARM_CORR_ADD:
            kwargs["block_dictionary"] = dictionaries["corr"]
        elif arm == inc.ARM_CORR_PCA_ADD:
            kwargs["pca_mean"] = dictionaries["pca_mean"]
            kwargs["pca_components"] = dictionaries["pca_components"]
        model = inc.build_increment_model(
            arm=arm,
            dictionary=dictionaries["base"],
            seed=seed,
            subspace=subspace,
            reference_state=reference,
            **kwargs,
        )
        if reference is None:
            reference = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
                if key not in ("D", "D_block")
            }
        models[arm] = model
    for arm in inc.ROUTE2_ARMS:
        kwargs = {}
        if arm == inc.ARM_JOINT_SPARSE:
            kwargs["block_dictionary"] = dictionaries["joint"]
        else:
            kwargs["pca_mean"] = dictionaries["joint_mean"]
            kwargs["pca_components"] = dictionaries["joint_components"]
        models[arm] = inc.build_increment_model(
            arm=arm,
            dictionary=dictionaries["base"],
            seed=seed,
            subspace=subspace,
            reference_state=reference,
            **kwargs,
        )
    return models, dictionaries


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_frozen_geometry():
    assert inc.COMMON_DIM == 1
    assert inc.BASE_ATOMS == 32 and inc.BASE_SPARSITY == 8
    assert inc.EXTRA_ATOMS == 16 and inc.EXTRA_SPARSITY == 4
    assert inc.COORD_DIM == 49
    assert inc.BASE_SLICE.start == 1 and inc.BASE_SLICE.stop == 33
    assert inc.EXTRA_SLICE.start == 33 and inc.EXTRA_SLICE.stop == 49
    assert inc.JOINT_DIM == 65 + 108 + 536 == 709
    assert inc.JOINT_ATOMS == 48 and inc.JOINT_SPARSITY == 12
    assert inc.JOINT_SLICE.start == 1 and inc.JOINT_SLICE.stop == 49
    assert inc.SCREEN_ABS_GATE == 0.003
    assert inc.ROUTE_OF_ARM[inc.ARM_CORR_ADD] == 1
    assert inc.ROUTE_OF_ARM[inc.ARM_JOINT_SPARSE] == 2


def test_coordinate_slices_payload():
    slices = inc.coordinate_slices()
    assert slices["common"] == [0, 1]
    assert slices["base"] == [1, 33]
    assert slices["block"] == [33, 49]
    assert slices["coord_dim"] == 49


# ---------------------------------------------------------------------------
# binding extension / shared init
# ---------------------------------------------------------------------------


def test_binding_extension_layout_and_shared_init():
    models, dictionaries = _build_all_arms()
    cssd_model = cssd.build_cssd_model(dictionaries["base"], 0, _synthetic_subspace())
    parent_a = cssd_model.W_A_S.detach()
    parent_e = cssd_model.W_E_S.detach()
    for arm, model in models.items():
        assert not model.D.requires_grad
        assert tuple(model.W_A_S.shape) == (49, parent_a.shape[1])
        assert torch.equal(model.W_A_S[:33], parent_a)
        assert bool((model.W_A_S[inc.EXTRA_SLICE] == 0).all())
        assert tuple(model.W_E_S.shape) == (3 * 49, parent_e.shape[1])
        edge = model.W_E_S.detach().reshape(3, 49, -1)
        parent_edge = parent_e.reshape(3, 33, -1)
        assert torch.equal(edge[:, :33], parent_edge)
        assert bool((edge[:, inc.EXTRA_SLICE] == 0).all())


def test_shared_readout_initialisation_bit_identical():
    models, _ = _build_all_arms()
    reference = models[inc.ARM_EXTRA_STRUCT].state_dict()
    for arm, model in models.items():
        state = model.state_dict()
        for key, value in reference.items():
            if key in ("D", "D_block"):
                continue
            assert key in state, f"{arm} missing {key}"
            assert tuple(state[key].shape) == tuple(value.shape)
            if key.startswith("block_pca"):
                continue
            assert torch.equal(state[key], value), f"{arm} differs at {key}"


# ---------------------------------------------------------------------------
# coordinate algebra
# ---------------------------------------------------------------------------


def test_base_coordinate_matches_frozen_cssd_path():
    models, dictionaries = _build_all_arms()
    batch = _synthetic_batch()
    cssd_model = cssd.build_cssd_model(dictionaries["base"], 0, _synthetic_subspace())
    # the parent repairs dead projected columns at init; the child applies the
    # same repair to the same parameter
    assert torch.equal(models[inc.ARM_EXTRA_STRUCT].D, cssd_model.D)
    with torch.no_grad():
        parent_coord = cssd_model.code(batch.dict_phi)
        mine = models[inc.ARM_EXTRA_STRUCT].code(batch.dict_phi, None)
    assert torch.equal(mine[:, inc.BASE_SLICE], parent_coord[:, 1:])
    assert torch.equal(mine[:, :1], parent_coord[:, :1])


def test_extra_structural_block_matches_rolecorr_v1_path():
    models, dictionaries = _build_all_arms()
    subspace = _synthetic_subspace()
    v1 = rc.build_rolecorr_model(
        arm=rc.ARM_CORR,
        dictionary=dictionaries["extra"],
        seed=0,
        subspace=subspace,
        dictionary_corr=dictionaries["corr"],
        freeze_dictionary=True,
    )
    batch = _synthetic_batch()
    with torch.no_grad():
        v1_struct = v1.structural_codes(batch.dict_phi)
        mine = models[inc.ARM_EXTRA_STRUCT].code(batch.dict_phi, None)
        v1_corr = v1.corr_codes(batch.corr_vec)
        mine_b = models[inc.ARM_CORR_ADD].code(batch.dict_phi, batch.corr_vec)
    assert torch.equal(mine[:, inc.EXTRA_SLICE], v1_struct)
    assert torch.equal(mine_b[:, inc.EXTRA_SLICE], v1_corr)


def test_coordinate_sparsity_and_block_modes():
    models, _ = _build_all_arms()
    batch = _synthetic_batch()
    with torch.no_grad():
        for arm in inc.ROUTE1_ARMS:
            model = models[arm]
            block = model.extract_block(batch)
            coord = model.code(batch.dict_phi, block)
            assert tuple(coord.shape) == (5, 49)
            assert int((coord[:, inc.BASE_SLICE] != 0).sum(dim=1).max()) <= inc.BASE_SPARSITY
            l0 = int((coord[:, inc.EXTRA_SLICE] != 0).sum(dim=1).max())
            if inc.ARM_BLOCK_MODE[arm] == inc.BLOCK_MODE_SPARSE:
                assert l0 <= inc.ARM_BLOCK_SPARSITY[arm]
            else:
                assert l0 == inc.EXTRA_ATOMS
        for arm in inc.ROUTE2_ARMS:
            model = models[arm]
            coord = model.code(batch.dict_phi, model.extract_block(batch))
            assert tuple(coord.shape) == (5, 49)
            l0 = int((coord[:, inc.JOINT_SLICE] != 0).sum(dim=1).max())
            if inc.ARM_BLOCK_MODE[arm] == inc.BLOCK_MODE_SPARSE:
                assert l0 <= inc.JOINT_SPARSITY
            else:
                assert l0 == inc.JOINT_ATOMS


def test_zero_purity_route1_and_route2():
    models, _ = _build_all_arms()
    batch = _synthetic_batch()
    for arm in inc.ROUTE1_ARMS:
        model = models[arm]
        block = model.extract_block(batch)
        purity = inc.zero_block_purity(model, batch.dict_phi, block)
        assert purity["other_columns_bit_identical"]
        assert purity["block_columns_zero"]
        base = inc.zero_base_purity(model, batch.dict_phi, block)
        assert base["applicable"] and base["base_columns_zero"]
        assert base["other_slices_bit_identical"]
    for arm in inc.ROUTE2_ARMS:
        model = models[arm]
        purity = inc.zero_block_purity(
            model, batch.dict_phi, model.extract_block(batch)
        )
        assert purity["span"] == [1, 49]
        assert purity["other_columns_bit_identical"]
        assert purity["block_columns_zero"]
        assert inc.zero_base_purity(model, batch.dict_phi, None)["applicable"] is False


def test_block_shuffle_preserves_multiset_and_is_deterministic():
    codes = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    shuffled = inc.shuffle_block_rows(codes, 42, batch)
    again = inc.shuffle_block_rows(codes, 42, batch)
    assert torch.equal(shuffled, again)
    for graph in (0, 1):
        rows = batch == graph
        original = codes[rows].numpy()
        permuted = shuffled[rows].numpy()
        assert np.array_equal(
            original[np.lexsort(original.T)], permuted[np.lexsort(permuted.T)]
        )
    other = inc.shuffle_block_rows(codes, 43, batch)
    assert not torch.equal(shuffled, other) or torch.equal(shuffled, codes)


# ---------------------------------------------------------------------------
# forward / mask semantics
# ---------------------------------------------------------------------------


def test_forward_runs_with_c6_mask_and_is_block_sensitive():
    models, _ = _build_all_arms()
    batch = _synthetic_batch()
    for arm in inc.ROUTE1_ARMS + inc.ROUTE2_ARMS:
        model = models[arm]
        model.eval()
        with torch.no_grad():
            prediction = model(batch, mask=cm_mask())
            assert torch.isfinite(prediction).all()
            assert tuple(prediction.shape) == (1,)
    # the block coordinate responds to its input
    model = models[inc.ARM_CORR_ADD]
    model.eval()
    with torch.no_grad():
        real = model.code(batch.dict_phi, batch.corr_vec)
        perturbed = batch.clone()
        perturbed.corr_vec = batch.corr_vec + 0.5 * torch.randn_like(batch.corr_vec)
        moved = model.code(batch.dict_phi, perturbed.corr_vec)
        assert float((real[:, inc.EXTRA_SLICE] - moved[:, inc.EXTRA_SLICE]).abs().max()) > 1e-6
        assert torch.equal(real[:, : inc.EXTRA_SLICE.start], moved[:, : inc.EXTRA_SLICE.start])
    # the shared initialisation zeroes the extension rows; a deterministic
    # non-zero pattern must make the prediction respond to the block coordinate
    with torch.no_grad():
        saved_a = model.W_A_S.detach().clone()
        saved_e = model.W_E_S.detach().clone()
        generator = torch.Generator().manual_seed(1234)
        model.W_A_S[inc.EXTRA_SLICE] = 0.1 * torch.randn(
            (inc.EXTRA_ATOMS, saved_a.shape[1]), generator=generator
        )
        for block in range(3):
            rows = slice(
                block * inc.COORD_DIM + inc.EXTRA_SLICE.start,
                block * inc.COORD_DIM + inc.EXTRA_SLICE.stop,
            )
            model.W_E_S[rows] = 0.1 * torch.randn(
                (inc.EXTRA_ATOMS, saved_e.shape[1]), generator=generator
            )
        base = model(batch, mask=cm_mask())
        model.inference_zero_block = True
        try:
            zeroed = model(batch, mask=cm_mask())
        finally:
            model.inference_zero_block = False
        model.W_A_S.copy_(saved_a)
        model.W_E_S.copy_(saved_e)
    assert float((base - zeroed).abs().max()) > 1e-6
    assert bool((model.W_A_S[inc.EXTRA_SLICE] == 0).all())


def cm_mask():
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm

    return cm.C6_MASK


# ---------------------------------------------------------------------------
# route 2 objects
# ---------------------------------------------------------------------------


def test_joint_scaler_roundtrip_and_energy_balance():
    rng = np.random.default_rng(3)
    struct = rng.standard_normal((64, inc.JOINT_STRUCT_DIM))
    sem = 5.0 * rng.standard_normal((64, inc.JOINT_SEM_DIM))
    corr = 0.1 * rng.standard_normal((64, inc.JOINT_CORR_DIM))
    scaler = inc.fit_joint_scaler(struct, sem, corr)
    scaled = inc.apply_joint_scaler(scaler, struct, sem, corr)
    assert scaled.shape == (64, inc.JOINT_DIM)
    for name, block in (("struct", struct), ("sem", sem), ("corr", corr)):
        sl = inc.JOINT_BLOCK_SLICES[name]
        energy = float(np.mean((scaled[:, sl] ** 2).sum(axis=1)))
        assert abs(energy - 1.0) < 1e-6, (name, energy)
    restored = inc.JointScaler.from_json(scaler.to_json())
    again = inc.apply_joint_scaler(restored, struct, sem, corr)
    assert np.array_equal(scaled, again)


def test_joint_pca_geometry():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((128, inc.JOINT_DIM))
    pca = inc.fit_joint_pca(X, rank=inc.JOINT_ATOMS)
    assert pca.components.shape == (inc.JOINT_ATOMS, inc.JOINT_DIM)
    gram = pca.components @ pca.components.T
    assert np.allclose(gram, np.eye(inc.JOINT_ATOMS), atol=1e-8)
    codes = pca.codes(X)
    assert codes.shape == (128, inc.JOINT_ATOMS)


# ---------------------------------------------------------------------------
# device resolver / test blocker
# ---------------------------------------------------------------------------


def test_resolve_device_and_test_blocker():
    assert inc.resolve_device(None).type == "cpu"
    assert inc.resolve_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert inc.resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            inc.resolve_device("cuda")
    with pytest.raises(RuntimeError):
        inc.official_test_blocker({"official_test_loaded": True})
    inc.official_test_blocker({"official_test_loaded": False})
