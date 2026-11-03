"""Phase A: the constructive structural witness must hold exactly as registered.

Every assertion here is a pre-registered gate of ``PROTOCOL.md`` §3, so a failure
means the round must stop rather than reinterpret the design.
"""

from __future__ import annotations

import numpy as np
import pytest

import nb_common as C
import nb_features as F
import nb_structural as S

from rdkit import Chem


@pytest.fixture(scope="module")
def pair():
    mol_a = S.build_ring_molecule(C.NITROGEN_INDEX_A)
    mol_b = S.build_ring_molecule(C.NITROGEN_INDEX_B)
    return mol_a, mol_b


def test_molecules_are_valid_and_have_the_expected_formula(pair):
    mol_a, mol_b = pair
    rec_a = S.molecule_record(mol_a, "A")
    rec_b = S.molecule_record(mol_b, "B")
    assert rec_a["formula"] == rec_b["formula"] == C.EXPECTED_FORMULA
    assert rec_a["n_heavy_atoms"] == rec_b["n_heavy_atoms"] == 18
    assert rec_a["atom_counts"] == {"C": 16, "N": 1, "O": 1}
    assert rec_b["atom_counts"] == {"C": 16, "N": 1, "O": 1}
    assert rec_a["bond_types"] == rec_b["bond_types"] == ["SINGLE"]
    assert set(rec_a["formal_charges"]) == {0}
    assert set(rec_b["formal_charges"]) == {0}
    # 16 CH2 + 1 NH + 1 O in an 18-ring
    assert rec_a["total_hydrogens"] == rec_b["total_hydrogens"] == 33


def test_canonical_smiles_match_the_registered_strings(pair):
    mol_a, mol_b = pair
    assert Chem.MolToSmiles(mol_a) == C.EXPECTED_SMILES_A
    assert Chem.MolToSmiles(mol_b) == C.EXPECTED_SMILES_B


def test_nitrogen_and_oxygen_positions(pair):
    mol_a, mol_b = pair
    rec_a = S.molecule_record(mol_a, "A")
    rec_b = S.molecule_record(mol_b, "B")
    assert rec_a["construction_oxygen_index"] == C.OXYGEN_INDEX
    assert rec_a["construction_nitrogen_index"] == C.NITROGEN_INDEX_A
    assert rec_b["construction_oxygen_index"] == C.OXYGEN_INDEX
    assert rec_b["construction_nitrogen_index"] == C.NITROGEN_INDEX_B
    assert rec_a["symbols"][0] == "O" and rec_a["symbols"][6] == "N"
    assert rec_b["symbols"][0] == "O" and rec_b["symbols"][8] == "N"


def test_n_o_shortest_graph_distance(pair):
    mol_a, mol_b = pair
    rec_a = S.molecule_record(mol_a, "A")
    rec_b = S.molecule_record(mol_b, "B")
    assert rec_a["n_o_shortest_distance"] == C.EXPECTED_NO_DISTANCE_A == 6
    assert rec_b["n_o_shortest_distance"] == C.EXPECTED_NO_DISTANCE_B == 8


def test_molecules_are_not_isomorphic_under_labels(pair):
    mol_a, mol_b = pair
    result = S.labelled_isomorphic(mol_a, mol_b)
    assert result["isomorphic"] is False
    assert result["search_nodes"] > 0  # the search really ran
    # sanity: the routine must return True for an isomorphic (relabelled) copy
    perm = [int(x) for x in np.random.default_rng(7).permutation(mol_a.GetNumAtoms())]
    relabelled = Chem.RenumberAtoms(mol_a, [int(i) for i in np.argsort(perm)])
    assert S.labelled_isomorphic(mol_a, relabelled)["isomorphic"] is True
    # and must reject a molecule that differs by one element
    broken = Chem.RWMol(mol_a)
    broken.GetAtomWithIdx(3).SetAtomicNum(7)
    assert S.labelled_isomorphic(mol_a, broken.GetMol())["isomorphic"] is False


def test_radius_two_local_environment_multiset_is_identical(pair):
    mol_a, mol_b = pair
    env_a = S.rooted_environment_multiset(mol_a)
    env_b = S.rooted_environment_multiset(mol_b)
    assert env_a["signature_multiset"] == env_b["signature_multiset"]
    assert env_a["environment_counts"] == env_b["environment_counts"]
    assert env_a["n_roots"] == env_b["n_roots"] == 18


def test_morgan_2048_fingerprint_is_identical(pair):
    mol_a, mol_b = pair
    fp_a = S.morgan_2048_bits(mol_a)
    fp_b = S.morgan_2048_bits(mol_b)
    assert fp_a.shape == (2048,)
    assert np.array_equal(fp_a, fp_b)
    assert fp_a.sum() > 0


def test_production_global42_is_identical():
    spec = C.frozen_spec()
    rec_a = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_A), "A")
    rec_b = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_B), "B")
    mol_a = S.production_tensors(rec_a["canonical_smiles"], spec)["mol"]
    mol_b = S.production_tensors(rec_b["canonical_smiles"], spec)["mol"]
    g_a = np.asarray(mol_a["global"])
    g_b = np.asarray(mol_b["global"])
    assert g_a.shape == g_b.shape == (C.GLOBAL_DIM,)
    assert np.array_equal(g_a, g_b), np.flatnonzero(g_a != g_b)


def test_type_distance_features_separate_the_pair():
    rec_a = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_A), "A")
    rec_b = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_B), "B")
    fa = F.pair_feature_molecule(rec_a["canonical_smiles"])
    fb = F.pair_feature_molecule(rec_b["canonical_smiles"])
    assert not np.array_equal(fa["real"], fb["real"])
    assert np.array_equal(fa["null"], fb["null"])
    assert np.array_equal(fa["row_margin"], fb["row_margin"])
    assert np.array_equal(fa["col_margin"], fb["col_margin"])
    assert fa["total"] == fb["total"]
    no = C.TYPE_PAIRS.index((C.ELEMENT_CLASSES.index("N"), C.ELEMENT_CLASSES.index("O")))
    assert fa["counts"][no, F.distance_bin(6)] == 1
    assert fa["counts"][no, F.distance_bin(8)] == 0
    assert fb["counts"][no, F.distance_bin(8)] == 1
    assert fb["counts"][no, F.distance_bin(6)] == 0


def test_minimum_cost_row_matching_semantics():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(18, 8))
    perm = rng.permutation(18)
    b = a[perm]
    matched = S.minimum_cost_row_matching(a, b)
    assert matched["identical"] is True
    assert matched["total_cost"] == 0.0
    assert matched["exact_multiset_equal"] is True
    noisy = a.copy()
    noisy[0, 0] += 1.0
    assert S.minimum_cost_row_matching(a, noisy)["identical"] is False


@pytest.mark.parametrize("variant", C.FORWARD_VARIANTS)
def test_production_forward_is_blind_to_the_pair(variant):
    """A0/C1 must return the same prediction for A and B (float32 tolerance)."""
    import torch

    spec = C.frozen_spec()
    norm_info = S.lipo_train_norm()
    assert norm_info["matches_executed_run"], norm_info["fingerprint"]

    rec_a = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_A), "A")
    rec_b = S.molecule_record(S.build_ring_molecule(C.NITROGEN_INDEX_B), "B")
    pay_a = S.production_tensors(rec_a["canonical_smiles"], spec)["mol"]
    pay_b = S.production_tensors(rec_b["canonical_smiles"], spec)["mol"]
    cache = S.two_molecule_cache([pay_a, pay_b], spec)

    import ptb_model as PM

    model, info = PM.build_model(spec, norm_info["norm"], variant, 0, torch.device("cpu"))
    model.eval()
    res = S.forward_record(model, cache, [0, 1], spec=spec)
    match = S.minimum_cost_row_matching(res["h_graphs"][0], res["h_graphs"][1])
    assert match["identical"] is True
    assert abs(float(res["y"][0] - res["y"][1])) <= C.PREDICTION_TOL
    assert abs(float(res["contrib"][0] - res["contrib"][1])) <= C.PREDICTION_TOL
    assert abs(float(res["rho"][0] - res["rho"][1])) <= C.PREDICTION_TOL
    assert np.abs(res["mean_h"][0] - res["mean_h"][1]).max() <= C.PREDICTION_TOL
    assert np.abs(res["mean_h2"][0] - res["mean_h2"][1]).max() <= C.PREDICTION_TOL
    assert np.array_equal(res["global_raw"][0], res["global_raw"][1])
    assert info["local_input_cols"] > 0
