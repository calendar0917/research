"""Focused tests for ``e2e_dictenv_sem108_v1`` (CSSD-Sem108).

CPU only; the official ZINC test split is never touched.  Covers the
pre-registered correctness list: exact Sem108 identity/geometry, size2
retention, removal of the anchor compression path, parent shared-parameter and
full-forward bit-identity, forbidden T1-block absence, semantic load-bearing
perturbation, frozen C6/backend reuse, the parameter-matching contract, the
official-test blocker and gradient viability.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

SUBSPACE_PATH = (
    REPO_ROOT
    / "tracks/ksvd/results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json"
)


def _dictionary() -> np.ndarray:
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    dictionary, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    return dictionary


def _subspace() -> cssd.CommonSubspace:
    payload = json.loads(SUBSPACE_PATH.read_text(encoding="utf-8"))["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(payload["components"], dtype=np.float64),
        rms=np.asarray(payload["rms"], dtype=np.float64),
        kind=str(payload["kind"]),
    )


def _models(seed: int = 0):
    subspace = _subspace()
    dictionary = _dictionary()
    model = sem.build_sem108_model(dictionary, seed, subspace)
    parent = cssd.build_cssd_model(dictionary, seed, subspace)
    return model, parent, dictionary, subspace


# ---------------------------------------------------------------------------
# geometry / interface
# ---------------------------------------------------------------------------


def test_sem108_geometry_resolved_from_implementation() -> None:
    geometry = sem.resolve_sem108_geometry()
    assert geometry["atom_shell_cols"] == 84
    assert geometry["bond_shell_cols"] == 24
    assert geometry["sem_dim"] == 108
    assert geometry["n_shells"] == 3
    assert geometry["n_shellpairs"] == 6
    assert geometry["blocks"]["root_atom"] == [108, 136]
    assert sem.SIZE2_BLOCK == tuple(int(value) for value in audit.ANCHOR_GROUPS["size"])


def test_interface_is_sem108_plus_size2_and_forbidden_blocks_absent() -> None:
    model, _parent, _dictionary, _subspace = _models()
    data = run._synthetic_batch()
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        interface = model.semantic_interface(coord, data, sem.SEMMask())
    assert interface.shape == (data.patch_cont.shape[0], 110)
    assert torch.equal(interface[:, :108], data.patch_cont[:, :108])
    assert torch.equal(interface[:, 108:110], data.anchor[:, 60:62])
    # the anchor size2 block and patch_cont[:, 140:142] are bit-identical on the
    # real standardized split; asserted by correctness gate G3 in the runner.
    forbidden = sem.forbidden_block_columns(data.patch_cont)
    assert sorted(forbidden) == ["incident_bonds", "root_atom", "topology_scalars"]


def test_forbidden_t1_blocks_do_not_reach_predictions() -> None:
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    data = run._synthetic_batch()
    forbidden = run._clone_batch(data)
    forbidden.patch_cont = data.patch_cont.clone()
    forbidden.patch_cont[:, 108:146] += 5.0
    semantic = run._clone_batch(data)
    semantic.patch_cont = data.patch_cont.clone()
    semantic.patch_cont[:, 0:108] += 0.25
    with torch.no_grad():
        clean = model(data, mask=cm.C6_MASK)
        blocked = model(forbidden, mask=cm.C6_MASK)
        perturbed = model(semantic, mask=cm.C6_MASK)
    assert torch.equal(clean, blocked), "forbidden T1 blocks must be invisible"
    assert not torch.equal(clean, perturbed), "Sem108 must be load-bearing"
    assert sem.SEM_INTERFACE_DIM == 110


# ---------------------------------------------------------------------------
# parent identity / parameter contract
# ---------------------------------------------------------------------------


def test_closed_form_parameter_matching() -> None:
    assert sem.SEM_FUSION_IN == 446
    assert sem.SEM_FUSION_HIDDEN == 114
    assert sem.SEM_FUSION_OUT == 48
    assert sem.CLOSED_FORM["interface_gap"] == 18
    assert sem.parent_local_interface_parameters()["local_interface_params"] == 56496


def test_parameter_budget_and_anchor_encoder_removed() -> None:
    model, parent, _dictionary, _subspace = _models()
    parent_params = int(sum(p.numel() for p in parent.parameters()))
    payload = sem.parameter_audit(model, parent_params)
    assert parent_params == 97727
    assert payload["candidate_params"] == 97709
    assert payload["delta"] == -18
    assert payload["relative_delta"] <= sem.PARAM_RATIO_PREFERRED
    assert payload["parameter_ratio_within_bound"]
    assert model.anchor_encoder is None
    assert "anchor_encoder.0.weight" not in model.state_dict()
    assert model.sem_fusion_hidden == 114
    assert model.fusion[0].weight.shape == (114, 446)
    assert model.fusion[2].weight.shape == (48, 114)


def test_parent_shared_parameters_bit_identical() -> None:
    model, parent, _dictionary, _subspace = _models()
    state_model = model.state_dict()
    state_parent = parent.state_dict()
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
    assert sorted(set(state_parent) - set(state_model)) == [
        "anchor_encoder.0.bias",
        "anchor_encoder.0.weight",
        "anchor_encoder.2.bias",
        "anchor_encoder.2.weight",
    ]


def test_parent_equivalence_model_forward_bit_identical() -> None:
    _model, parent, dictionary, subspace = _models()
    equivalent = sem.build_parent_equivalence_model(dictionary, 0, subspace)
    equivalent.load_state_dict(parent.state_dict())
    equivalent.eval()
    parent.eval()
    data = run._synthetic_batch()
    with torch.no_grad():
        pred_parent, aux_parent = parent(data, mask=cm.C6_MASK, return_aux=True)
        pred_equiv, aux_equiv = equivalent(data, mask=cm.C6_MASK, return_aux=True)
    assert torch.equal(pred_parent, pred_equiv)
    assert torch.equal(aux_parent["E"], aux_equiv["E"])


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def test_mask_blocks_are_orthogonal_and_shuffles_are_distribution_preserving() -> None:
    model, _parent, _dictionary, _subspace = _models()
    data = run._synthetic_batch()
    with torch.no_grad():
        coord = model.code(data.dict_phi)
        base = model.semantic_interface(coord, data, sem.SEMMask())
        atom_zero = model.semantic_interface(coord, data, sem.SEMMask(sem_atom_zero=True))
        bond_zero = model.semantic_interface(coord, data, sem.SEMMask(sem_bond_zero=True))
        block_zero = model.semantic_interface(coord, data, sem.SEMMask(sem_block_zero=True))
    assert torch.equal(atom_zero[:, 84:108], base[:, 84:108])
    assert float(atom_zero[:, 0:84].abs().max()) == 0.0
    assert torch.equal(bond_zero[:, 0:84], base[:, 0:84])
    assert float(bond_zero[:, 84:108].abs().max()) == 0.0
    assert float(block_zero[:, 0:108].abs().max()) == 0.0
    assert torch.equal(block_zero[:, 108:110], base[:, 108:110])
    shuffled = model.semantic_interface(
        coord, data, sem.SEMMask(use_sem_row_shuffle=True, sem_shuffle_seed=101)
    )
    assert torch.equal(shuffled[:, 108:110], base[:, 108:110])
    assert not torch.equal(shuffled[:, :108], base[:, :108])
    assert float((shuffled[:, :108] - base[:, :108]).abs().max()) > 0.0


def test_merge_sem_mask_accepts_plain_c6_mask() -> None:
    merged = sem.merge_sem_mask(cm.C6_MASK, sem.SEMMask(sem_atom_zero=True, size2_zero=True))
    assert isinstance(merged, sem.SEMMask)
    assert merged.sem_atom_zero
    assert merged.size2_zero
    assert tuple(merged.anchor_zero_groups) == tuple(cm.C6_MASK.anchor_zero_groups)
    assert not merged.sem_block_zero
    assert cm.C6_MASK is cssd.CSSD_MASK
    assert cm.c6_equivalence_check()


def test_frozen_backend_reused_not_copied() -> None:
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


# ---------------------------------------------------------------------------
# blocker / gradients / runner helpers
# ---------------------------------------------------------------------------


def test_official_test_blocker() -> None:
    sem.official_test_blocker({"official_test_loaded": False})
    with pytest.raises(RuntimeError):
        sem.official_test_blocker({"official_test_loaded": True})


def test_gradient_viability_and_interface_activity() -> None:
    model, _parent, _dictionary, _subspace = _models()
    data = run._synthetic_batch()
    diagnostics = run._gradient_diagnostics(model, data, cm.C6_MASK, cm.H1_LAMBDA)
    for key in ("grad_D", "grad_fusion_W1", "grad_fusion_W2", "grad_W_A_S", "grad_W_E_S"):
        assert diagnostics[key] > 0.0, key
    activity = run._interface_activity(model, data, cm.C6_MASK)
    assert activity["passed"]


def test_residual_dictionary_view_matches_parent_semantics() -> None:
    model, _parent, _dictionary, _subspace = _models()
    data = run._synthetic_batch()
    view = run._ResidualDictionaryView(model)
    code = view.code(data.dict_phi)
    assert code.shape == (data.dict_phi.shape[0], 32)
    with torch.no_grad():
        reconstruction = view.reconstruct(data.dict_phi, code)
        assert reconstruction.shape == data.dict_phi.shape
        assert view.reconstruction_loss(data.dict_phi, code) >= 0.0
        prediction, _aux = view(data, mask=cm.C6_MASK, return_aux=True)
        assert prediction.shape == (1,)


def test_relabel_invariance_real_batch() -> None:
    """G12 on real valid molecules: a node relabelling cannot change predictions."""
    model, _parent, _dictionary, _subspace = _models()
    model.eval()
    payload = run._relabel_invariance(model, n_molecules=2)
    assert payload["passed"], payload
