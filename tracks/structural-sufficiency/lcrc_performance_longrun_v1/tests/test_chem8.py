"""CHEMLOCAL8 channel definitions, artefact alignment and cache extension."""

from __future__ import annotations

import json

import numpy as np
import torch

import cl_common as C
import cl_chem8 as CH
import cl_data as CD
import lcrc_data as PREP_DATA


def _mol(smiles: str):
    mol, method = CH._parse_like_frozen_adapter(smiles)
    return mol, method


def test_hand_chiral_channels():
    import pytest

    pytest.importorskip("rdkit", reason="RDKit is a local data-prep dependency; the committed artifact is used on GPU hosts")
    mol, method = _mol("C[C@@H](N)C(=O)O")
    assert method == "strict"
    out, _ = CH.chem8_for_mol(mol)
    assert out.shape == (6, 8)
    assert out[1, 0] == 1.0 and out[1, 1] == 0.0 and out[1, 2] == 0.0

    mol, _ = _mol("C[C@H](N)C(=O)O")
    out, _ = CH.chem8_for_mol(mol)
    assert out[1, 1] == 1.0 and out[1, 0] == 0.0

    mol, _ = _mol("CC(N)C(=O)O")
    out, _ = CH.chem8_for_mol(mol)
    assert out[1, 2] == 1.0 and out[1, 0] == 0.0 and out[1, 1] == 0.0


def test_hand_donor_acceptor_and_hybridisation():
    import pytest

    pytest.importorskip("rdkit", reason="RDKit is a local data-prep dependency; the committed artifact is used on GPU hosts")
    mol, _ = _mol("CCO")
    out, _ = CH.chem8_for_mol(mol)
    # C, C, O: O is both donor and acceptor; carbons are SP3.
    assert out[2, 3] == 1.0 and out[2, 4] == 1.0
    assert out[0, 7] == 1.0 and out[1, 7] == 1.0
    assert out[:, 5].sum() == 0 and out[:, 6].sum() == 0

    mol, _ = _mol("c1ccccc1")
    out, _ = CH.chem8_for_mol(mol)
    assert out[:, 6].sum() == 6.0
    assert out[:, 5].sum() == 0 and out[:, 7].sum() == 0

    mol, _ = _mol("C#C")
    out, _ = CH.chem8_for_mol(mol)
    assert out[:, 5].sum() == 2.0


def test_spec_extension_is_exactly_eight_columns():
    base = CD.base_view("ESOL", "published").spec
    ext = CH.chem8_spec(base)
    assert ext.atom_feat_dim == base.atom_feat_dim + 8
    assert tuple(f.name for f in ext.atom_feature_fields[-8:]) == CH.CHANNEL_NAMES
    assert ext.global_dim == base.global_dim and ext.n_scalars == base.n_scalars
    assert ext.n_nbr_codes == base.n_nbr_codes and ext.n_path_codes == base.n_path_codes
    assert ext.hash() != base.hash()
    # Idempotent and serialisable.
    assert CH.chem8_spec(ext).hash() == ext.hash()
    from feature_spec import FeatureSpec

    assert FeatureSpec.from_dict(ext.to_dict()).hash() == ext.hash()


def test_artifact_manifest_verifies():
    payload = C.read_json(C.CHEM8_MANIFEST)
    assert payload["label_free"] is True
    assert tuple(payload["channel_names"]) == CH.CHANNEL_NAMES
    for dataset in C.V2C.DATASET_ORDER:
        entry = payload["datasets"][dataset]
        assert C.sha256_file(CH.artifact_path(dataset)) == entry["sha256"]
        arrays = CH.load_artifact(dataset)
        assert list(arrays["channel_names"]) == list(CH.CHANNEL_NAMES)
        if int(arrays["chem8"].shape[0]):
            assert set(np.unique(arrays["chem8"]).tolist()) <= {0, 1}
        assert np.all(np.asarray(arrays["method"]) >= 0)
    # The Al-valence Tox21 fallbacks are recorded, not silently dropped.
    assert payload["datasets"]["Tox21"]["n_fallback"] > 0


def test_cache_alignment_and_norm_unchanged():
    base = CD.base_view("ESOL", "published")
    ext = CD.extend_view(base, "ESOL")
    b = base.train_cache
    e = ext.train_cache
    assert np.array_equal(np.asarray(b["atom_ptr"]), np.asarray(e["atom_ptr"]))
    assert list(b["smiles"]) == list(e["smiles"])
    assert list(b["molecule_ids"]) == list(e["molecule_ids"])
    assert e["atom_feat"].shape == (b["atom_feat"].shape[0], b["atom_feat"].shape[1] + 8)
    assert np.array_equal(
        np.asarray(b["atom_feat"], dtype=np.float32), np.asarray(e["atom_feat"])[:, : b["atom_feat"].shape[1]]
    )
    rows = [int(x) for x in b["rows"]]
    expected = CH.channels_for_rows("ESOL", rows, smiles=list(b["smiles"]))
    assert np.array_equal(np.asarray(e["atom_feat"])[:, b["atom_feat"].shape[1] :], expected)
    assert PREP_DATA.norm_fingerprint(base.train_norm) == PREP_DATA.norm_fingerprint(ext.train_norm)
    PREP_DATA.validate_cache(e, ext.spec)


def test_artifact_atom_order_matches_frozen_raw_artifact():
    """The chem8 artefact was parsed from the same SMILES in the same order."""
    import mb2_raw_artifact as RAW

    arrays = CH.load_artifact("ESOL")
    raw = RAW.load_arrays("ESOL", verify=True)
    assert np.array_equal(np.asarray(raw["atom_ptr"]), np.asarray(arrays["atom_ptr"]))
    assert [str(x) for x in raw["smiles"]] == [str(x) for x in arrays["smiles"]]
