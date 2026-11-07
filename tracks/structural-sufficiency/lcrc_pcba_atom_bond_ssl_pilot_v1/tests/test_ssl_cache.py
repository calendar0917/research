"""Cache-backed CPU tests: masked views, partitions and the encoder transfer.

Marked ``slow`` because they memory-map the 9 GB frozen ``pcba_v1`` cache.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ssl_common as C
import ssl_data as D
import ssl_model as M
import ssl_train as T

pytestmark = pytest.mark.slow


def test_structure_partition_and_audit(structure):
    assert structure.probe_rows.size == C.N_PROBE
    assert structure.fit_rows.size + structure.probe_rows.size == structure.train_rows.size
    assert np.intersect1d(structure.probe_rows, structure.fit_rows).size == 0
    assert np.isin(structure.probe_rows, structure.train_rows).all()
    assert np.isin(structure.fit_rows, structure.train_rows).all()
    audit = structure.audit
    assert audit["loader_file_access"]["label_artifact_opened"] is False
    assert audit["label_artifact_opened"] is False
    assert audit["labels_read"] is False
    assert audit["probe"]["n"] == C.N_PROBE
    assert audit["feature_spec"]["atom_feat_dim"] == 51
    assert audit["feature_spec"]["bond_feat_dim"] == 13


def test_masked_views_are_bijective_and_bounded(structure, batch):
    counts = batch.counts()
    assert counts["masked_atoms"] >= counts["molecules"]
    assert 0.10 <= counts["masked_atoms"] / counts["atoms"] <= 0.20
    assert 0.10 <= counts["masked_bonds"] / counts["bonds"] <= 0.20
    assert torch.equal(
        batch.atom_view["atom_feat"][batch.atom_ids], torch.zeros(counts["masked_atoms"], 51)
    )
    assert torch.equal(
        batch.bond_view["bond_feat"][batch.bond_ids], torch.zeros(counts["masked_bonds"], 13)
    )
    endpoints = batch.bond_view["bond_index"][batch.bond_ids].reshape(-1)
    assert torch.equal(batch.bond_view["atom_feat"][endpoints], torch.zeros(endpoints.numel(), 51))
    # the atom view only zeroes the masked atoms
    untouched = torch.ones(counts["atoms"], dtype=torch.bool)
    untouched[batch.atom_ids] = False
    assert bool((batch.atom_view["atom_feat"][untouched] != 0).any())
    # the bond view additionally zeroes the endpoints of masked bonds
    assert int((batch.bond_view["atom_feat"] == 0).any(dim=1).sum()) >= int(endpoints.unique().numel())


def test_model_inputs_are_structure_only(structure, batch):
    assert set(D.model_input(batch.atom_view)) == {"atom_feat", "bond_feat", "bond_index"}
    for forbidden in ("atom_scalar", "global", "nbr_code", "path_cat"):
        assert forbidden not in batch.atom_view


def test_targets_are_the_hidden_classes(structure, batch):
    atom_type = structure.arrays["atom_type"]
    bond_type = structure.arrays["bond_type"]
    atom_off = batch.atom_view["atom_off"].numpy()
    bond_off = batch.bond_view["bond_off"].numpy()
    atom_pos = 0
    bond_pos = 0
    for i, row in enumerate(batch.rows):
        local = batch.atom_mask_by_molecule[i]
        k = int(local.size)
        ids = batch.atom_ids[atom_pos : atom_pos + k].numpy()
        assert np.array_equal(ids, local + atom_off[i])
        # the hidden truth in the *cache* (note: probe rows are absent from the
        # batch, so a batch position is not a cache index)
        hidden_atoms = np.asarray(atom_type[structure.atom_ptr[row] + local])
        assert np.array_equal(hidden_atoms, batch.atom_targets[atom_pos : atom_pos + k].numpy())
        assert np.array_equal(np.asarray(batch.atom_view["atom_type"][ids]), hidden_atoms)
        atom_pos += k

        blocal = batch.bond_mask_by_molecule[i]
        kb = int(blocal.size)
        bids = batch.bond_ids[bond_pos : bond_pos + kb].numpy()
        assert np.array_equal(bids, blocal + bond_off[i])
        hidden = np.asarray(bond_type[structure.bond_ptr[row] + blocal]) - 1
        assert np.array_equal(hidden, batch.bond_targets[bond_pos : bond_pos + kb].numpy())
        assert np.array_equal(np.asarray(batch.bond_view["bond_type"][bids]) - 1, hidden)
        assert hidden.min() >= 0 and hidden.max() < C.N_BOND_CLASSES
        bond_pos += kb
    assert atom_pos == int(batch.atom_ids.numel())
    assert bond_pos == int(batch.bond_ids.numel())


def test_pretext_encoders_match_seed0_scratch(structure, pretext, scratch):
    model, audit = pretext
    assert audit["encoder_state_matches_seed0_scratch"] is True
    expected = M.encoder_fingerprint(scratch["state"])
    assert M.encoder_fingerprint(model) == expected
    assert audit["encoder_parameters"] == C.ENCODER_PARAMETERS
    assert audit["atom_head_input"] == 288 and audit["bond_head_input"] == 480
    assert audit["atom_head_classes"] == 16 and audit["bond_head_classes"] == 5
    assert audit["dropout_modules"] == []


def test_scratch_a_matches_historical_initialisation(structure, scratch, norm):
    model, build_audit, state = T.build_a0_scratch(structure.spec, norm, torch.device("cpu"))
    assert build_audit["base_init_fingerprint"] == C.HISTORICAL_PARITY["base_init_fingerprint"]
    assert C.parameter_count(model) == C.EXPECTED_PARAMETERS
    assert build_audit["model_audit"]["new_parameter_count"] == C.EXPECTED_NEW_PARAMETERS
    assert int(build_audit["model_audit"]["base_parameter_elements"]) == C.EXPECTED_BASE_PARAMETERS


def test_transfer_touches_only_the_encoders(structure, pretext, scratch, norm):
    model, _ = pretext
    downstream, _, scratch_state = T.build_a0_scratch(structure.spec, norm, torch.device("cpu"))
    # simulate a trained encoder pair: every tensor must differ from scratch
    trained = {k: v.clone() for k, v in M.encoder_state(model).items()}
    for k, v in trained.items():
        v.add_(0.01 if v.dim() == 2 else -0.002)
    payload = {
        "encoder_state": trained,
        "encoder_fingerprint": C.state_fingerprint(trained),
        "update": int(C.PRETRAIN["final_update_for_export"]),
    }
    audit = T.transfer_ssl_init(downstream, scratch_state, payload)
    assert audit["non_transferred_tensors_all_equal_to_seed0_scratch"] is True
    assert audit["n_encoder_tensors_changed"] == audit["n_encoder_tensors"]
    assert audit["parameter_count"] == C.EXPECTED_PARAMETERS
    assert audit["pretext_heads_present"] == []
    assert audit["ssl_optimizer_state_transferred"] is False
    assert M.encoder_fingerprint(downstream) == C.state_fingerprint(trained)
    assert M.encoder_fingerprint(downstream) != M.encoder_fingerprint(scratch_state)


def test_prior_counts_are_consistent(structure, priors):
    n_atoms = int((structure.atom_ptr[structure.fit_rows + 1] - structure.atom_ptr[structure.fit_rows]).sum())
    n_bonds = int((structure.bond_ptr[structure.fit_rows + 1] - structure.bond_ptr[structure.fit_rows]).sum())
    assert priors["atom"]["n"] == n_atoms
    assert priors["bond"]["n"] == n_bonds
    assert sum(priors["atom"]["counts"]) == n_atoms
    assert sum(priors["bond"]["counts"]) == n_bonds
    atom_p = np.asarray(priors["atom"]["frequencies"])
    assert atom_p.min() >= 0.0 and atom_p.max() <= 1.0
    assert abs(float(atom_p.sum()) - 1.0) < 1e-9
    assert priors["atom"]["majority_class"] in (2, 3)  # C or C_aromatic


def test_gradients_reach_all_groups_on_a_small_batch(structure, pretext):
    model, _ = pretext
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rows = structure.fit_rows[:32]
    pb = D.build_pretext_batch(structure, rows, visit_index=1)
    loss, _ = model.loss(
        D.model_input(pb.atom_view), D.model_input(pb.bond_view),
        pb.atom_ids, pb.bond_ids, pb.atom_targets, pb.bond_targets,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    groups = {
        "atom_encoder.": 0,
        "bond_encoder.": 0,
        "atom_head.": 0,
        "bond_head.": 0,
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for prefix in groups:
            if name.startswith(prefix):
                groups[prefix] += float(param.grad.norm())
    assert all(v > 0 for v in groups.values()), groups
    assert np.isfinite(float(loss))
