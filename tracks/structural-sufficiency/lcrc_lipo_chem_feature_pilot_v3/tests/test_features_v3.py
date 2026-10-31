"""Feature identity, order and label-independence tests."""

from __future__ import annotations

import hashlib

import numpy as np

import v3_common as C
import v3_features as F


def test_exactly_two_feature_versions_in_the_frozen_order(store):
    assert tuple(C.CHEM12_NAMES) == (
        "TPSA",
        "MolWt",
        "NumHAcceptors",
        "NumHDonors",
        "NumRotatableBonds",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "RingCount",
        "FractionCSP3",
        "HeavyAtomCount",
        "MolMR",
    )
    assert tuple(C.CHEM13_NAMES) == tuple(C.CHEM12_NAMES) + ("MolLogP",)
    assert store.audit["chem12"]["names"] == list(C.CHEM12_NAMES)
    assert store.audit["chem13"]["names"] == list(C.CHEM13_NAMES)


def test_chem13_is_chem12_plus_mollogp(store):
    assert store.chem12.shape == (2352, 12)
    assert store.chem13.shape == (2352, 13)
    assert np.array_equal(store.chem13[:, :12], store.chem12)
    assert np.array_equal(store.chem13[:, 12], store.physchem13[:, 0])
    assert not np.array_equal(store.chem13[:, 12], store.chem12[:, 0])


def test_feature_hashes_are_pinned_and_finite(store):
    assert store.audit["chem12"]["sha256"] == hashlib.sha256(
        np.ascontiguousarray(store.chem12).tobytes()
    ).hexdigest()
    assert store.audit["chem13"]["sha256"] == hashlib.sha256(
        np.ascontiguousarray(store.chem13).tobytes()
    ).hexdigest()
    assert np.isfinite(store.chem13).all()
    assert store.audit["labels_used"] is False
    assert store.audit["monitor_structures_used"] is False
    assert store.audit["reserved_structures_used"] is False
    assert store.audit["official_valid_test_used"] is False


def test_canonical_smiles_rows_align_with_the_frozen_manifest(store):
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    assert store.rows == fit_rows
    import adapters as prep_adapters

    manifest = prep_adapters.load_lipo_row_manifest(C.C.v1c.LIPO_ROW_MANIFEST)
    for row, digest in zip(store.rows, store.canonical_hashes):
        assert digest == manifest["canonical_smiles_sha256"][int(row)]


def test_independent_rdkit_recompute_sample(store):
    check = store.audit["independent_descriptor_check"]
    assert check["n_sampled"] >= 20
    assert check["all_equal"] is True
    assert all(x["all_13_equal"] for x in check["samples"])


def test_augmented_spec_only_appends_global_fields():
    base = C.load_frozen_spec()
    spec12 = F.augmented_spec("chem12")
    spec13 = F.augmented_spec("chem13")
    assert spec12.global_dim == base.global_dim + 12
    assert spec13.global_dim == base.global_dim + 13
    assert spec12.atom_feat_dim == base.atom_feat_dim
    assert spec12.bond_feat_dim == base.bond_feat_dim
    assert spec12.n_scalars == base.n_scalars
    assert spec12.atom_feature_fields == base.atom_feature_fields
    assert spec12.bond_feature_fields == base.bond_feature_fields
    assert spec12.atom_scalar_fields == base.atom_scalar_fields
    assert [f.name for f in spec12.global_feature_fields][-12:] == list(C.CHEM12_NAMES)
    assert [f.name for f in spec13.global_feature_fields][-13:] == list(C.CHEM13_NAMES)
    assert spec12.hash() != spec13.hash()
