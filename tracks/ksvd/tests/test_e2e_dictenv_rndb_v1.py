"""Focused tests for ``e2e_dictenv_rndb_v1`` (Rolewise Nonlinear Dictionary Binding).

CPU only; the official ZINC test split is never touched.  The tests cover the
pre-registered correctness list: the node/edge per-coordinate decomposition
identity, psi-off parent equivalence, inactive-role strict zero, dictionary-role
permutation equivariance, common-coordinate exclusion from psi, absence of bias,
shared psi across roles, unchanged shell/shellpair/C6 semantics, the
official-test blocker, the parameter budget and gradient viability.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_rndb_v1 as rndb


def _synthetic_phi(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=2.0, scale=1.0, size=(n, cssd.PHI_DIM))
    base[:, 5] = 0.0
    base[:, 8] = 0.0
    return base


def _subspace(seed: int = 0) -> cssd.CommonSubspace:
    return cssd.build_common_subspace(_synthetic_phi(seed=seed), 1)


def _dictionary() -> np.ndarray:
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _synthetic_batch(n_nodes: int = 48, n_occ: int = 160, n_bond: int = 120, seed: int = 3) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)
    data = SimpleNamespace()
    data.dict_phi = torch.randn(n_nodes, cssd.PHI_DIM, generator=generator)
    data.dict_atom = torch.randint(0, p2.ATOM_CATEGORIES, (n_nodes,), generator=generator)
    data.anchor = torch.randn(n_nodes, audit.ANCHOR_DIM_EXPECTED, generator=generator)
    data.env_occ_node = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_root = torch.randint(0, n_nodes, (n_occ,), generator=generator)
    data.env_occ_shell = torch.randint(0, p2.N_SHELLS, (n_occ,), generator=generator)
    data.env_bond_u = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_v = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_type = torch.randint(0, p2.BOND_CATEGORIES, (n_bond,), generator=generator)
    data.env_bond_root = torch.randint(0, n_nodes, (n_bond,), generator=generator)
    data.env_bond_shellpair = torch.randint(0, p2.SHELLPAIR_CLASSES, (n_bond,), generator=generator)
    return data


def _models(seed: int = 0):
    subspace = _subspace()
    dictionary = _dictionary()
    model = rndb.build_rndb_model(dictionary, seed, subspace)
    parent = cssd.build_cssd_model(dictionary, seed, subspace)
    return model, parent, subspace


# ---------------------------------------------------------------------------
# parameter budget / structure
# ---------------------------------------------------------------------------


def test_parameter_budget_hidden_widths_and_no_bias() -> None:
    model, _parent, _subspace_ = _models()
    assert model.psi_node_hidden == 32
    assert model.psi_edge_hidden == 16
    for psi in (model.psi_A, model.psi_E):
        assert isinstance(psi, torch.nn.Sequential)
        for layer in psi:
            if hasattr(layer, "bias"):
                assert layer.bias is None, "psi must be bias-free (strict zero invariant)"
    audit_payload = rndb.parameter_audit(model, 97727)
    assert audit_payload["new_params"] == 7680
    assert audit_payload["new_params_ok"]
    assert audit_payload["ratio_ok"]
    assert audit_payload["psi_A_params"] == 96 * 32 + 32 * 96
    assert audit_payload["psi_E_params"] == 48 * 16 + 16 * 48


def test_psi_shared_across_roles_and_common_excluded_from_basis() -> None:
    model, _parent, _subspace_ = _models()
    # one shared module, not a per-role list.
    assert isinstance(model.psi_A, torch.nn.Sequential)
    assert isinstance(model.psi_E, torch.nn.Sequential)
    n = model.n_common
    node_basis = model._node_psi_basis().clone()
    edge_basis = tuple(block.clone() for block in model._edge_psi_basis())
    with torch.no_grad():
        model.W_A_S[:n] += 3.0
        d_struct = 3 * (n + model.n_dict)
        for block in range(3):
            start = block * (n + model.n_dict)
            model.W_E_S[start : start + n] += 3.0
        perturbed_node = model._node_psi_basis()
        perturbed_edge = tuple(block.clone() for block in model._edge_psi_basis())
        model.W_A_S[:n] -= 3.0
        for block in range(3):
            start = block * (n + model.n_dict)
            model.W_E_S[start : start + n] -= 3.0
    assert torch.equal(node_basis, perturbed_node), "node psi basis must ignore common rows"
    for left, right in zip(edge_basis, perturbed_edge):
        assert torch.equal(left, right), "edge psi basis must ignore common rows"


# ---------------------------------------------------------------------------
# G0 decomposition identity
# ---------------------------------------------------------------------------


def test_node_and_edge_decomposition_identity_synthetic() -> None:
    model, _parent, _subspace_ = _models()
    data = _synthetic_batch()
    coord = model.code(data.dict_phi)
    p_all, parent = rndb.node_role_contributions(model, coord, data.env_occ_node, data.dict_atom)
    assert rndb.max_abs_diff(p_all.sum(dim=1), parent) <= rndb.DECOMPOSITION_TOL
    pe_all, pe_parent = rndb.edge_role_contributions(
        model, coord, data.env_bond_u, data.env_bond_v, data.env_bond_type
    )
    assert rndb.max_abs_diff(pe_all.sum(dim=1), pe_parent) <= rndb.DECOMPOSITION_TOL


def test_decomposition_identity_real_batch() -> None:
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

    data_list = p1run.load_split("train", subset=32)
    if not data_list:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    model, _parent, _subspace_ = _models()
    batch = next(iter(p1.make_env_loader(data_list, 64, False, 0)))
    coord = model.code(batch.dict_phi)
    p_all, parent = rndb.node_role_contributions(model, coord, batch.env_occ_node, batch.dict_atom)
    assert rndb.max_abs_diff(p_all.sum(dim=1), parent) <= rndb.DECOMPOSITION_TOL
    pe_all, pe_parent = rndb.edge_role_contributions(
        model, coord, batch.env_bond_u, batch.env_bond_v, batch.env_bond_type
    )
    assert rndb.max_abs_diff(pe_all.sum(dim=1), pe_parent) <= rndb.DECOMPOSITION_TOL


# ---------------------------------------------------------------------------
# G1 psi-off parent equivalence
# ---------------------------------------------------------------------------


def test_psi_off_equals_parent_environments_synthetic() -> None:
    model, parent, _subspace_ = _models()
    data = _synthetic_batch()
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        model.psi_node_on = False
        model.psi_edge_on = False
        rndb_env = model.environments_masked(coord, data, cm.C6_MASK)
        model.psi_node_on = True
        model.psi_edge_on = True
        parent_env = parent.environments_masked(coord, data, cm.C6_MASK)
    assert torch.equal(rndb_env, parent_env)


def test_psi_off_equals_parent_forward_real() -> None:
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run

    data_list = p1run.load_split("valid", subset=16)
    if not data_list:  # pragma: no cover
        pytest.skip("ZINC encoded data not available")
    model, parent, _subspace_ = _models()
    batch = next(iter(p1.make_env_loader(data_list, 32, False, 0)))
    model.eval()
    parent.eval()
    with torch.no_grad():
        model.psi_node_on = False
        model.psi_edge_on = False
        pred_rndb, aux_rndb = model(batch, mask=cm.C6_MASK, return_aux=True)
        pred_parent, aux_parent = parent(batch, mask=cm.C6_MASK, return_aux=True)
        model.psi_node_on = True
        model.psi_edge_on = True
    assert float((pred_rndb - pred_parent).abs().max()) <= rndb.PSI_OFF_TOL
    assert float((aux_rndb["E"] - aux_parent["E"]).abs().max()) <= rndb.PSI_OFF_TOL


# ---------------------------------------------------------------------------
# G2 inactive-role strict zero
# ---------------------------------------------------------------------------


def test_inactive_role_strict_zero() -> None:
    model, _parent, _subspace_ = _models()
    data = _synthetic_batch()
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        assert float(model.psi_A(torch.zeros(4, 7, int(p2.D_A))).abs().max()) == 0.0
        assert float(model.psi_E(torch.zeros(4, 7, 48)).abs().max()) == 0.0
        zero_dict = coord.clone()
        zero_dict[:, model.n_common :] = 0.0
        u_inactive = model.rndb_node_occurrence(zero_dict, data)
        ue_inactive = model.rndb_edge_occurrence(zero_dict, data)
        model.dict_zero = True
        u_common = model.rndb_node_occurrence(coord, data)
        ue_common = model.rndb_edge_occurrence(coord, data)
        model.dict_zero = False
    assert torch.equal(u_inactive, u_common)
    assert torch.equal(ue_inactive, ue_common)


# ---------------------------------------------------------------------------
# G3 role permutation equivariance
# ---------------------------------------------------------------------------


def test_role_permutation_equivariance() -> None:
    model, _parent, _subspace_ = _models()
    data = _synthetic_batch()
    coord = model.code(data.dict_phi)
    n = model.n_common
    d_struct = int(coord.shape[1])
    permutation = torch.randperm(model.n_dict, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        coord_perm = coord.clone()
        coord_perm[:, n:] = coord[:, n:][:, permutation]
        w_a = model.W_A_S.detach().clone()
        w_e = model.W_E_S.detach().clone()
        model.W_A_S[n:] = w_a[n:][permutation]
        for block in range(3):
            start = block * d_struct + n
            model.W_E_S[start : start + model.n_dict] = w_e[start : start + model.n_dict][permutation]
        u_perm = model.rndb_node_occurrence(coord_perm, data)
        ue_perm = model.rndb_edge_occurrence(coord_perm, data)
        model.W_A_S.copy_(w_a)
        model.W_E_S.copy_(w_e)
        u_orig = model.rndb_node_occurrence(coord, data)
        ue_orig = model.rndb_edge_occurrence(coord, data)
    assert float((u_perm - u_orig).abs().max()) <= 1e-5
    assert float((ue_perm - ue_orig).abs().max()) <= 1e-5


# ---------------------------------------------------------------------------
# G5 frozen shell / C6 semantics
# ---------------------------------------------------------------------------


def test_shell_shellpair_and_c6_unchanged() -> None:
    assert rndb.RNDB_MASK is cm.C6_MASK
    assert cm.c6_equivalence_check()
    assert p2.N_SHELLS == 3
    assert p2.SHELLPAIR_CLASSES == 6
    model, parent, _subspace_ = _models()
    assert type(model.node_encoder) is type(parent.node_encoder)
    assert type(model.edge_encoder) is type(parent.edge_encoder)
    assert type(model.fusion) is type(parent.fusion)
    assert type(model.reader) is type(parent.reader)
    assert type(model.topology_encoder) is type(parent.topology_encoder)
    assert type(model.global_encoder) is type(parent.global_encoder)


# ---------------------------------------------------------------------------
# G6 official-test blocker + gradient viability
# ---------------------------------------------------------------------------


def test_official_test_blocker() -> None:
    rndb.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        rndb.official_test_blocker({"official_test_loaded": True})


def test_gradient_viability_synthetic() -> None:
    model, _parent, _subspace_ = _models()
    data = _synthetic_batch()
    model.train()
    coord = model.code(data.dict_phi)
    environment = model.environments_masked(coord, data, cm.C6_MASK)
    loss = environment.pow(2).mean() + model.D.pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert float(model.D.grad.norm()) > 0.0
    assert float(model.psi_A[0].weight.grad.norm()) > 0.0
    assert float(model.psi_A[2].weight.grad.norm()) > 0.0
    assert float(model.psi_E[0].weight.grad.norm()) > 0.0
    assert float(model.psi_E[2].weight.grad.norm()) > 0.0


def test_parent_parameters_bit_identical_to_cssd() -> None:
    """The parent RNG stream is preserved; only the psi tensors are new."""
    subspace = _subspace()
    dictionary = _dictionary()
    model = rndb.build_rndb_model(dictionary, 0, subspace)
    parent = cssd.build_cssd_model(dictionary, 0, subspace)
    state_model = model.state_dict()
    state_parent = parent.state_dict()
    shared = {
        key: value
        for key, value in state_parent.items()
        if key in state_model and state_model[key].shape == value.shape
    }
    assert len(shared) >= 50
    for key, value in shared.items():
        assert torch.equal(value, state_model[key]), f"parent parameter {key} differs"
    assert sorted(set(state_model) - set(state_parent)) == [
        "psi_A.0.weight",
        "psi_A.2.weight",
        "psi_E.0.weight",
        "psi_E.2.weight",
    ]
