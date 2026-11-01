"""RDKit -> frozen raw attribute -> ``lipo_v1`` graph adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

import mb_data as D
import mb_rdkit as R

import adapters as PREP_ADAPTERS
import molecular_graph as MG


@pytest.fixture(scope="module")
def spec():
    return D.frozen_spec()


def test_benzene_ring_is_aromatic_carbon(spec):
    row = R.smiles_to_raw_row("c1ccccc1", 0)
    assert row["atom_class"].tolist() == [3] * 6
    assert row["edge_class"].tolist() == [3] * 12  # aromatic bonds, both directions
    graph = R.raw_row_to_graph(row, 0, spec)
    assert graph.n_atoms == 6 and graph.n_bonds == 6
    analysis = MG.analyse_graph(graph)
    assert bool(analysis.ring_atom.all())
    assert set(graph.bond_type.tolist()) == {4}


def test_unknown_element_maps_to_unk_not_dropped(spec):
    row = R.smiles_to_raw_row("[Se]c1ccccc1", 0)
    assert row["atom_class"][0] == R.ATOM_UNK
    graph = R.raw_row_to_graph(row, 0, spec)
    assert graph.atom_type[0] == spec.atom_unk_index
    assert graph.n_atoms == 7


def test_charge_and_hydrogen_buckets(spec):
    row = R.smiles_to_raw_row("[NH4+]", 0)
    assert row["atom_charge"].tolist() == [R.CHARGE_BUCKETS.index(1)]
    assert row["atom_h"].tolist() == [4]
    row2 = R.smiles_to_raw_row("C[CH2-]", 0)
    assert row2["atom_charge"].tolist() == [R.CHARGE_BUCKETS.index(0), R.CHARGE_BUCKETS.index(-1)]


def test_high_hydrogen_count_uses_four_plus_bucket():
    # RDKit itself refuses chemically impossible valence, so the bucket rule is
    # tested at the mapping level: any count above 4 collapses into bucket 4 and
    # is never silently dropped or mapped to UNK.
    assert R._h_code(5) == R.H_BUCKETS.index(4)
    assert R._h_code(9) == R.H_BUCKETS.index(4)
    assert R._h_code(-1) == R.H_UNK
    assert R._h_code(4) == R.H_BUCKETS.index(4)


def test_stereo_bucket_is_recorded():
    row = R.smiles_to_raw_row("F/C=C/F", 0)
    assert set(row["edge_stereo"].tolist()) >= {0, 1}


def test_disconnected_salt_is_kept_whole(spec):
    row = R.smiles_to_raw_row("CC(=O)[O-].[Na+]", 0)
    assert row["n_components"] == 2
    graph = R.raw_row_to_graph(row, 0, spec)
    assert graph.n_atoms == 5
    analysis = MG.analyse_graph(graph)
    assert analysis.n_components == 2


def test_unparseable_smiles_raises_explicitly():
    with pytest.raises(ValueError, match="could not parse"):
        R.smiles_to_raw_row("not-a-smiles", 0)


def test_graph_matches_upstream_adapter_for_a_real_row(spec):
    src = __import__("mb_source").load_dataset("ESOL", labels="none")
    row = R.smiles_to_raw_row(src["smiles"][0], 0)
    graph = R.raw_row_to_graph(row, 0, spec)
    rebuilt = PREP_ADAPTERS.lipo_row_to_graph(row, 0, spec)
    assert graph.atom_feat.shape == rebuilt.atom_feat.shape
    assert np.array_equal(graph.atom_feat, rebuilt.atom_feat)
    assert np.array_equal(graph.bond_feat, rebuilt.bond_feat)
    assert np.array_equal(graph.atom_type, rebuilt.atom_type)
    assert np.array_equal(graph.bond_type, rebuilt.bond_type)


def test_feature_matrix_widths_match_the_frozen_spec(spec):
    row = R.smiles_to_raw_row("CCO", 0)
    graph = R.raw_row_to_graph(row, 0, spec)
    assert graph.atom_feat.shape == (3, spec.atom_feat_dim)
    assert graph.bond_feat.shape == (2, spec.bond_feat_dim)
    assert graph.atom_feat.dtype == np.float32


def test_atom_attribute_audit_counts(spec):
    rows = [R.smiles_to_raw_row(s, i) for i, s in enumerate(["CCO", "[Se]", "c1ccccc1"])]
    audit = R.atom_attribute_audit(rows)
    assert audit["n_atoms_total"] == 3 + 1 + 6
    assert audit["atom_unk_count"] == 1
    assert audit["n_molecules"] == 3
