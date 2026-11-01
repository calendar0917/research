"""P0 regression tests: row correspondence, alignment, forward wiring, collisions.

These are the reusable guards that a *future* representation change must keep
passing.  They are written against the frozen pipeline (they import it, they do
not modify it).
"""

from __future__ import annotations

import numpy as np
import pytest

import ta_common as C


# ---------------------------------------------------------------------------
# 1. RDKit SMILES <-> raw artifact <-> cache row correspondence
# ---------------------------------------------------------------------------


def test_raw_artifact_matches_published_smiles_and_cache_rows():
    import mb2_raw_artifact as RAW
    import mb2_source as S
    import ta_data as TD

    dataset = "BBBP"
    rows = [3, 17, 101]
    published = S.all_smiles(dataset)
    raw = RAW.load_rows(dataset, rows, verify=True)
    for r, row in zip(rows, raw):
        assert str(row["smiles"]) == str(published[r])
        assert int(row["mol_index"]) == r

    cache, info = TD.build_or_load_cache(dataset, rows, tag="test_rows", split_label="test_rows")
    assert list(np.asarray(cache["rows"]).tolist()) == rows
    assert cache["molecule_ids"] == [str(r) for r in rows]
    assert cache["smiles"] == [str(published[r]) for r in rows]
    # the cache must be reproducible from the committed artifact, row by row
    import lcrc_data as PREP_DATA
    import mb2_data as D
    import mb2_rdkit as RDK

    spec = D.frozen_spec()
    atom_ptr = np.asarray(cache["atom_ptr"])
    for i, r in enumerate(rows):
        graph = RDK.raw_row_to_graph(raw[i], r, spec)
        mol = PREP_DATA.build_molecule(graph, spec)
        sl = slice(int(atom_ptr[i]), int(atom_ptr[i + 1]))
        assert np.array_equal(np.asarray(cache["atom_feat"])[sl], np.asarray(mol["atom_feat"]))
        assert np.array_equal(np.asarray(cache["atom_type"])[sl], np.asarray(mol["atom_type"]))
        assert np.array_equal(np.asarray(cache["global"])[i], np.asarray(mol["global"]))
    assert cache["meta"]["labels_present"] is False
    assert info["rows_sha256"] == C.row_sha(rows)


# ---------------------------------------------------------------------------
# 2. no cross-molecule misalignment in a batch
# ---------------------------------------------------------------------------


def test_collate_never_shifts_across_molecules(freesolv_rows):
    import lcrc_data as PREP_DATA
    import mb2_data as D
    import ta_data as TD

    train_rows, valid_rows = freesolv_rows
    rows = list(train_rows[:3])
    cache, _ = TD.build_or_load_cache("FreeSolv", rows, tag="test_align", split_label="test_align")
    batch = PREP_DATA.collate(cache, [0, 1, 2])
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_off = np.asarray(batch["atom_off"])
    assert atom_off.tolist() == [0, int(atom_ptr[1]), int(atom_ptr[2])]

    # per-molecule rebuild, then offset the local bond index the same way
    spec = D.frozen_spec()
    import mb2_raw_artifact as RAW
    import mb2_rdkit as RDK

    raw = RAW.load_rows("FreeSolv", rows, verify=True)
    for i, r in enumerate(rows):
        mol = PREP_DATA.build_molecule(RDK.raw_row_to_graph(raw[i], r, spec), spec)
        a0 = int(atom_off[i])
        a1 = a0 + int(mol["n_atoms"])
        assert np.array_equal(batch["atom_feat"][a0:a1].numpy(), np.asarray(mol["atom_feat"]))
        assert np.array_equal(batch["atom_scalar"][a0:a1].numpy(), np.asarray(mol["atom_scalar"]))
        # path_end is global inside the batch and must land inside this molecule
        local_path = (batch["path_atom"] == a0) | (
            (batch["path_atom"] >= a0) & (batch["path_atom"] < a1)
        )
        ends = batch["path_end"][local_path].numpy()
        if ends.size:
            assert ends.min() >= a0 and ends.max() < a1
        # bond indices must be shifted into the batch's atom range
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        bi = batch["bond_index"].numpy()
        mask = (batch["bond_batch"] == i)
        assert np.array_equal(bi[mask], np.asarray(mol["bond_index"]) + a0)
        assert int(mask.sum()) == b1 - b0
    # global statistics must not be mixed between molecules
    assert np.array_equal(batch["global"].numpy(), np.asarray(cache["global"])[[0, 1, 2]])


# ---------------------------------------------------------------------------
# 3. every declared field really enters the forward pass
# ---------------------------------------------------------------------------


def test_declared_fields_are_wired_and_documentation_tensors_are_not_read():
    import ta_audit as A

    audit = A.forward_key_audit("FreeSolv", 3)
    sens = A.sensitivity_audit("FreeSolv", 3)
    assert sens["fields_with_no_response"] == []
    assert sens["passed"] is True
    # atom_type / bond_type are declared in the batch but never read by the model
    assert "atom_type" in audit["keys_present_but_not_read"]
    assert "bond_type" in audit["keys_present_but_not_read"]
    assert audit["atom_type_read"] is False
    for key in (
        "atom_feat",
        "atom_scalar",
        "bond_feat",
        "bond_index",
        "nbr_code",
        "path_cat",
        "path_end63",
        "path_end",
        "path_closure",
        "global",
    ):
        assert key in audit["keys_read_by_forward"]


def test_s1_s2_are_joint_codes_not_marginals():
    import ta_audit as A

    joint = A.joint_code_tests()
    assert joint["s1"]["passed"] is True
    assert joint["s1"]["marginal_element_multiset_identical"] is True
    assert joint["s1"]["marginal_bond_multiset_identical"] is True
    assert joint["s2"]["passed"] is True
    assert joint["s2_pairing"]["passed"] is True
    assert joint["s2_pairing"]["additive_terms_identical"] is True


# ---------------------------------------------------------------------------
# 6. chirality / bond stereo: verified encoding, verified collision
# ---------------------------------------------------------------------------


def test_tetrahedral_chirality_is_a_real_representation_collision():
    import ta_collision as COL

    result = COL.tetrahedral_test("BACE")
    assert result["raw_artifact_identical"] is True, "the frozen raw adapter must not read chirality"
    assert result["model_input"]["all_input_tensors_identical"] is True
    assert result["verdict"].startswith("REPRESENTATION_COLLISION")
    assert result["cip_a"] != result["cip_b"]


def test_ez_stereo_reaches_bond_features_only():
    import ta_collision as COL

    result = COL.ez_test("BBBP")
    assert result["tensors_that_change"] == ["bond_feat"]
    assert result["s1_s2_codes_identical"] is True
    assert "stereo" in result["verdict"] or result["verdict"].startswith("PARTIAL")


def test_frozen_spec_hash_is_unchanged():
    import mb2_data as D

    spec = D.frozen_spec()
    payload = C.read_json(C.PREP / "analysis/feature_spec_lipo.json")
    assert spec.hash() == payload["spec_hash"]
    assert spec.atom_feat_dim == 51
    assert spec.bond_feat_dim == 13
    assert spec.global_dim == 42
    assert spec.n_scalars == 9
    assert spec.n_nbr_codes == 80
    assert spec.n_path_codes == 400
