"""Focused tests for ``e2e_dictenv_jointbond_v1`` (JointBond).

Data-free: a deterministic synthetic dictionary / batch and a synthetic common
subspace are used so the module runs in a fresh checkout.  The official ZINC
test split is never touched.  Covers the pre-registered correctness list: exact
parameter contract, one-hook parent extension with bit-identical branch-off
forward, endpoint correspondence (joint swap invariance; single swaps move),
branch-local residual-code zero, gradients into the branch and the dictionary,
shellpair routing / batch offsets against an independent float64 reference, the
parent structure-shuffle endpoint semantics, frozen parent path reuse and the
official-test blocker.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_jointbond_v1 as jb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem

SEED = 0


def _dictionary() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.asarray(rng.normal(size=(cssd.PHI_DIM, cssd.K_ATOMS)), dtype=np.float32)


def _subspace() -> cssd.CommonSubspace:
    rng = np.random.default_rng(1)
    return cssd.build_common_subspace(rng.normal(size=(256, cssd.PHI_DIM)), q=1)


def _models(seed: int = SEED):
    subspace = _subspace()
    dictionary = _dictionary()
    model = jb.build_jointbond_model(dictionary, seed, subspace)
    parent = sem.build_sem108_model(dictionary, seed, subspace)
    return model, parent, dictionary, subspace


# ---------------------------------------------------------------------------
# geometry / parameter contract
# ---------------------------------------------------------------------------


def test_frozen_branch_geometry_and_parameter_contract() -> None:
    assert jb.JOINT_ALPHA_DIM == 32
    assert jb.JOINT_ATOM_DIM == 28
    assert jb.JOINT_BOND_DIM == 4
    assert jb.JOINT_H == 16
    assert jb.JOINT_T_DIM == 48
    assert jb.JOINT_NEW_PARAMETERS == 4224
    assert jb.SEM108_CANDIDATE_PARAMETERS == 97709
    assert jb.JOINT_EXPECTED_TOTAL == 101933
    model, parent, _dictionary, _subspace = _models()
    parent_params = int(sum(p.numel() for p in parent.parameters()))
    audit = jb.parameter_audit(model, parent_params)
    assert parent_params == jb.SEM108_CANDIDATE_PARAMETERS
    assert audit["new_params"] == jb.JOINT_NEW_PARAMETERS
    assert audit["candidate_params"] == jb.JOINT_EXPECTED_TOTAL
    assert audit["new_params_exact"] and audit["total_exact"]
    assert audit["parameter_matched"] is False
    assert model.joint_A.weight.shape == (jb.JOINT_H, jb.JOINT_ALPHA_DIM)
    assert model.joint_C.weight.shape == (jb.JOINT_H, jb.JOINT_ATOM_DIM)
    assert model.joint_B.weight.shape == (jb.JOINT_T_DIM, jb.JOINT_BOND_DIM)
    assert model.joint_F[0].weight.shape == (jb.JOINT_F_HIDDEN, jb.JOINT_T_DIM)
    assert model.joint_F[2].weight.shape == (jb.JOINT_T_DIM, jb.JOINT_F_HIDDEN)


def test_parent_shared_parameters_bit_identical_and_only_joint_keys_added() -> None:
    model, parent, _dictionary, _subspace = _models()
    state_model, state_parent = model.state_dict(), parent.state_dict()
    shared = {
        key: value
        for key, value in state_parent.items()
        if key in state_model
        and state_model[key].shape == value.shape
        and not key.startswith(("fusion.", "anchor_encoder."))
    }
    assert len(shared) >= 40
    for key, value in shared.items():
        assert torch.equal(value, state_model[key]), f"parent parameter {key} differs"
    assert sorted(set(state_model) - set(state_parent)) == [
        "joint_A.weight",
        "joint_B.weight",
        "joint_C.weight",
        "joint_F.0.weight",
        "joint_F.2.weight",
    ]
    assert model.anchor_encoder is None


def test_parent_backend_classes_reused_not_copied() -> None:
    model, parent, _dictionary, _subspace = _models()
    for name in (
        "node_encoder",
        "edge_encoder",
        "reader",
        "pair_encoder",
        "distance_gate",
        "global_encoder",
        "topology_encoder",
        "relation_encoder",
        "pair_projection",
    ):
        assert type(getattr(model, name)) is type(getattr(parent, name)), name
    assert cm.C6_MASK is cssd.CSSD_MASK
    assert cm.c6_equivalence_check()
    assert sem.SEM108Model._edge_response_delta(None, None, None, None, None) is None


# ---------------------------------------------------------------------------
# branch off / on
# ---------------------------------------------------------------------------


def test_branch_off_is_bit_identical_to_parent_with_parent_weights() -> None:
    _model, parent, dictionary, subspace = _models()
    equivalence = jb.build_jointbond_model(dictionary, SEED, subspace)
    equivalence.load_state_dict(parent.state_dict(), strict=False)
    equivalence.eval()
    parent.eval()
    data = jb.synthetic_batch(seed=3, n_graphs=2)
    off = jb.merge_joint_mask(cm.C6_MASK, jb.JointBondMask(joint_branch_off=True))
    with torch.no_grad():
        pred_parent, aux_parent = parent(data, mask=cm.C6_MASK, return_aux=True)
        pred_off, aux_off = equivalence(data, mask=off, return_aux=True)
        pred_on, _aux_on = equivalence(data, mask=cm.C6_MASK, return_aux=True)
    assert torch.equal(pred_off, pred_parent)
    assert torch.equal(aux_off["E"], aux_parent["E"])
    assert not torch.equal(pred_on, pred_parent), "the branch must not be inert"
    assert jb.merge_joint_mask(cm.C6_MASK, jb.JointBondMask()).as_dict()["is_identity"] is False
    assert cm.C6_MASK.is_identity() is False


def test_branch_off_returns_none_and_zero_alpha_zeroes_the_branch() -> None:
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = jb.synthetic_batch(seed=5)
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        off = jb.merge_joint_mask(cm.C6_MASK, jb.JointBondMask(joint_branch_off=True))
        assert model.joint_edge_delta(coord, data, mask=off) is None
        zeroed = model.joint_edge_delta(
            coord, data, mask=jb.merge_joint_mask(cm.C6_MASK, jb.JointBondMask(joint_alpha_zero=True))
        )
        assert float(zeroed.abs().max()) == 0.0
        assert float(zeroed.norm()) == 0.0
        shifted = coord.clone()
        shifted[:, : model.common_dim] = 0.0
        delta_shifted = model.joint_edge_delta(shifted, data, mask=cm.C6_MASK)
        delta_plain = model.joint_edge_delta(coord, data, mask=cm.C6_MASK)
        assert torch.equal(delta_shifted, delta_plain), "the branch must ignore the common coordinate"


# ---------------------------------------------------------------------------
# endpoint correspondence
# ---------------------------------------------------------------------------


def test_joint_swap_both_endpoints_invariant_single_swaps_move() -> None:
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = jb.synthetic_batch(seed=7)
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        alpha = coord[:, model.common_dim :]
        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=jb.JOINT_ATOM_DIM).to(coord.dtype)
        bu, bv, bt = data.env_bond_u, data.env_bond_v, data.env_bond_type
        au, qu, av, qv = alpha[bu], q[bu], alpha[bv], q[bv]
        base = jb.joint_branch_from_endpoints(model, au, qu, av, qv, bt)
        swap_both = jb.joint_branch_from_endpoints(model, av, qv, au, qu, bt)
        swap_atom = jb.joint_branch_from_endpoints(model, au, qv, av, qu, bt)
        swap_structure = jb.joint_branch_from_endpoints(model, av, qu, au, qv, bt)
        routed = model.joint_edge_response(coord, data, bu, bv, cm.C6_MASK)
    assert torch.equal(base, swap_both)
    assert torch.equal(base, routed)
    assert float((base - swap_atom).abs().max()) > 0.0
    assert float((base - swap_structure).abs().max()) > 0.0
    changed = float(((base - swap_atom).abs().sum(dim=1) > 0).double().mean())
    assert changed > 0.0


def test_synthetic_distinguishable_endpoint_pair() -> None:
    model, _parent, _dictionary, _subspace = _models()
    generator = torch.Generator().manual_seed(101)
    alpha_u = torch.randn(1, jb.JOINT_ALPHA_DIM, generator=generator)
    alpha_v = torch.randn(1, jb.JOINT_ALPHA_DIM, generator=generator)
    q_u = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=jb.JOINT_ATOM_DIM).to(torch.float32)
    q_v = torch.nn.functional.one_hot(torch.tensor([9]), num_classes=jb.JOINT_ATOM_DIM).to(torch.float32)
    bond_type = torch.tensor([2])
    with torch.no_grad():
        base = jb.joint_branch_from_endpoints(model, alpha_u, q_u, alpha_v, q_v, bond_type)
        swap_both = jb.joint_branch_from_endpoints(model, alpha_v, q_v, alpha_u, q_u, bond_type)
        swap_atom = jb.joint_branch_from_endpoints(model, alpha_u, q_v, alpha_v, q_u, bond_type)
        swap_structure = jb.joint_branch_from_endpoints(model, alpha_v, q_u, alpha_u, q_v, bond_type)
        zeroed = jb.joint_branch_from_endpoints(
            model, torch.zeros_like(alpha_u), q_u, torch.zeros_like(alpha_v), q_v, bond_type
        )
    assert int(q_u.argmax()) != int(q_v.argmax())
    assert float((alpha_u - alpha_v).abs().max()) > 0.0
    assert float((base - swap_both).abs().max()) == 0.0
    assert float((base - swap_atom).abs().max()) > 0.0
    assert float((base - swap_structure).abs().max()) > 0.0
    assert float(zeroed.abs().max()) == 0.0


def test_structure_shuffle_reads_both_quantities_at_the_shuffled_real_endpoint() -> None:
    """Parent edge-assignment shuffle: structure and semantics stay paired."""
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = jb.synthetic_batch(seed=9, n_graphs=1)
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(int(data.env_bond_u.numel())):
        key = (int(data.env_bond_root[index]), int(data.env_bond_shellpair[index]))
        groups.setdefault(key, []).append(index)
    su, sv = data.env_bond_u.clone(), data.env_bond_v.clone()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        rolled = indices[1:] + indices[:1]
        su[indices] = data.env_bond_u[rolled]
        sv[indices] = data.env_bond_v[rolled]
    data.env_bond_u_shuffled = su
    data.env_bond_v_shuffled = sv
    shuffled_mask = sem.SEMMask(use_edge_shuffle=True)
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        shuffled = model.joint_edge_response(coord, data, su, sv, shuffled_mask)
        alpha = coord[:, model.common_dim :]
        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=jb.JOINT_ATOM_DIM).to(coord.dtype)
        manual = jb.joint_branch_from_endpoints(
            model, alpha[su], q[su], alpha[sv], q[sv], data.env_bond_type
        )
        unshuffled = model.joint_edge_response(
            coord, data, data.env_bond_u, data.env_bond_v, shuffled_mask
        )
        delta_from_shuffled_path = model.joint_edge_delta(
            coord, data, bond_u=su, bond_v=sv, mask=shuffled_mask
        )
        slots_shuffled = model.joint_edge_slots(coord, data, shuffled_mask)
        slots_plain = model.joint_edge_slots(coord, data, cm.C6_MASK)
    assert not torch.equal(su, data.env_bond_u)
    assert torch.equal(shuffled, manual)
    assert float((shuffled - unshuffled).abs().max()) > 0.0
    assert torch.equal(delta_from_shuffled_path, shuffled)
    assert not torch.equal(slots_shuffled, slots_plain)


# ---------------------------------------------------------------------------
# gradients / routing / diagnostics
# ---------------------------------------------------------------------------


def test_gradients_reach_branch_and_dictionary() -> None:
    model, _parent, _dictionary, _subspace = _models()
    data = jb.synthetic_batch(seed=11)
    prediction, aux = model(data, mask=cm.C6_MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), data.y.view(-1)) + float(
        cm.H1_LAMBDA
    ) * model.reconstruction_loss(aux["phi"], aux["coord"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in model.joint_parameters().items():
        assert parameter.grad is not None, name
        assert float(parameter.grad.norm()) > 0.0, name
    assert float(model.D.grad.norm()) > 0.0
    model.zero_grad(set_to_none=True)


def test_routing_matches_float64_reference_and_endpoint_formulation() -> None:
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = jb.synthetic_batch(seed=7, n_graphs=2)
    generator = torch.Generator().manual_seed(11)
    rows = [
        (data.batch == graph).nonzero(as_tuple=False).view(-1) for graph in (0, 1)
    ]
    bond_u: list[int] = []
    bond_v: list[int] = []
    bond_root: list[int] = []
    for group in rows:
        endpoints = group[torch.randint(0, int(group.numel()), (45, 2), generator=generator)]
        bond_u.extend(int(value) for value in endpoints[:, 0])
        bond_v.extend(int(value) for value in endpoints[:, 1])
        bond_root.extend(
            int(value)
            for value in group[torch.randint(0, int(group.numel()), (45,), generator=generator)]
        )
    data.env_bond_u = torch.tensor(bond_u, dtype=torch.long)
    data.env_bond_v = torch.tensor(bond_v, dtype=torch.long)
    data.env_bond_root = torch.tensor(bond_root, dtype=torch.long)
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        slots = model.joint_edge_slots(coord, data, cm.C6_MASK)
        reference = jb.reference_edge_slots(model, coord, data, mask=cm.C6_MASK)
        delta = model.joint_edge_response(coord, data, data.env_bond_u, data.env_bond_v, cm.C6_MASK)
        alpha = coord[:, model.common_dim :]
        q = torch.nn.functional.one_hot(data.dict_atom, num_classes=jb.JOINT_ATOM_DIM).to(coord.dtype)
        manual = jb.joint_branch_from_endpoints(
            model,
            alpha[data.env_bond_u],
            q[data.env_bond_u],
            alpha[data.env_bond_v],
            q[data.env_bond_v],
            data.env_bond_type,
        )
        shifted = model.joint_edge_response(
            coord, data, (data.env_bond_u + 1) % int(coord.shape[0]), data.env_bond_v, cm.C6_MASK
        )
    assert float((delta - manual).abs().max()) == 0.0
    assert float((delta - shifted).abs().max()) > 0.0
    assert float(np.abs(reference - slots.double().numpy()).max()) <= 1.0e-4
    assert float((data.batch[data.env_bond_u] != data.batch[data.env_bond_v]).double().mean()) == 0.0


def test_diagnostics_report_branch_scale_and_health() -> None:
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = jb.synthetic_batch(seed=13)
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        scale = jb.response_scale_stats(model, coord, data, cm.C6_MASK)
        norms = jb.parameter_norms(model)
    assert scale["official_test_loaded"] is False
    assert norms["official_test_loaded"] is False
    assert norms["joint_branch_dead"] is False
    assert scale["joint_branch_response"]["mean"] > 0.0
    assert scale["parent_edge_response"]["mean"] > 0.0
    assert set(norms) >= set(model.joint_parameters()) | {"joint_branch_dead"}
    assert int(p2.SHELLPAIR_CLASSES) == 6


def test_official_test_blocker() -> None:
    jb.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        jb.official_test_blocker({"official_test_loaded": True})
