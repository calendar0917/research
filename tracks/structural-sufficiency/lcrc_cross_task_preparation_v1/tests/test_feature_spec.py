from __future__ import annotations

import json

import numpy as np

import cpu_gates as G
from feature_spec import FeatureSpec


def test_lipo_spec_roundtrip_and_hash(spec):
    d = spec.to_dict()
    loaded = FeatureSpec.from_dict(json.loads(json.dumps(d)))
    assert loaded.hash() == spec.hash()
    assert spec.atom_feat_dim == 51
    assert spec.bond_feat_dim == 13
    assert spec.n_scalars == 9
    assert spec.global_dim == 42
    assert spec.n_atom_types == 16
    assert spec.n_bond_types == 5
    assert spec.n_nbr_codes == 80
    assert spec.n_path_codes == 400
    assert spec.supports_disconnected is True
    assert spec.atom_unk_index == 15
    assert spec.bond_unk_id == 5
    assert spec.atom_type_names[15] == "UNK"
    assert spec.bond_type_names[4] == "UNK"


def test_lipo_path_dtype_is_not_uint8(spec):
    assert np.dtype(spec.structural_code_dtype).itemsize >= 4
    assert spec.n_path_codes > 255
    assert spec.n_path_codes <= spec.code_max + 1


def test_lipo_unknown_policies_are_explicit(spec):
    assert "UNK" in spec.atom_unknown_policy
    assert "UNK" in spec.bond_unknown_policy
    assert "finite" in spec.disconnected_policy.lower()


def test_lipo_fields_have_unique_names_and_no_label_fields(spec):
    for group in (spec.atom_feature_fields, spec.bond_feature_fields, spec.atom_scalar_fields, spec.global_feature_fields):
        names = [f.name for f in group]
        assert len(names) == len(set(names))
        assert not any(n.lower() in {"exp", "label", "target", "y", "logp"} for n in names)
    assert spec.atom_feat_dim == sum(f.dim for f in spec.atom_feature_fields)
    assert spec.bond_feat_dim == sum(f.dim for f in spec.bond_feature_fields)
    assert spec.n_scalars == sum(f.dim for f in spec.atom_scalar_fields)
    assert spec.global_dim == sum(f.dim for f in spec.global_feature_fields)


def test_legacy_spec_reproduces_frozen_dimensions(legacy_spec):
    assert legacy_spec.atom_feat_dim == 108
    assert legacy_spec.bond_feat_dim == 7
    assert legacy_spec.global_dim == 33
    assert legacy_spec.n_scalars == 9
    assert legacy_spec.n_atom_types == 21
    assert legacy_spec.n_bond_types == 3
    assert legacy_spec.n_nbr_codes == 63
    assert legacy_spec.n_path_codes == 189
    assert legacy_spec.supports_disconnected is False
    assert "reject" in legacy_spec.disconnected_policy.lower()
    assert legacy_spec.atom_unk_index is None
    assert legacy_spec.bond_unk_id is None


def test_path_code_overflow_gate(spec):
    report = G.gate_overflow(spec)
    assert report["status"] == "PASS"
    assert report["max_path_code"] == 399
    assert report["uint8_wrap_value"] == 143
    assert report["dtype"] == "int32"


def test_spec_source_provenance_is_serialisable(spec):
    d = spec.to_dict()
    assert "source_provenance" in d
    assert isinstance(d["source_provenance"], dict)
    json.dumps(d, sort_keys=True)
