"""Expression boundary of the frozen representation (synthetic, cache-free)."""

from __future__ import annotations

import numpy as np

import tsb_gates as G
import tsb_g0
import tsb_mol
import tsb_morgan


def test_two_labelled_cycles_share_wl_0_3_but_differ_in_a_b_distance():
    payload = G.synthetic_boundary()
    assert payload["passed"], payload
    evidence = payload["evidence"]
    assert evidence["graph_distance"] == [8, 12]
    assert evidence["wl_round_multiset_equal"] == [True, True, True, True]
    assert evidence["g0_vector_equal"] is True
    assert evidence["morgan_vector_equal"] is True


def test_a_smaller_cycle_does_not_share_the_multiset():
    """Control: when the marks are within WL reach the statistic does change."""
    mol_a = G.labelled_cycle(12, 4)
    mol_b = G.labelled_cycle(12, 6)
    _m_a, _g_a, rounds_a = tsb_mol.molecule_tokens_from_mol(mol_a, rounds=3)
    _m_b, _g_b, rounds_b = tsb_mol.molecule_tokens_from_mol(mol_b, rounds=3)
    counts_a = [dict(tsb_mol.count_tokens(r)) for r in rounds_a]
    counts_b = [dict(tsb_mol.count_tokens(r)) for r in rounds_b]
    differing = [counts_a[r] != counts_b[r] for r in range(4)]
    assert any(differing), "a 12-ring with distance 4 vs 6 should be distinguishable"
    assert G.graph_distance(mol_a, 0, 4) != G.graph_distance(mol_b, 0, 6)


def test_parse_failure_is_a_hard_error():
    try:
        tsb_mol.parse_smiles("this-is-not-a-smiles")
    except ValueError as exc:
        assert "could not parse" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid SMILES must raise, never silently become a zero vector")


def test_g0_columns_are_frozen_and_finite():
    import tsb_data as D

    smiles = D.load_smiles()[:50]
    graphs = [tsb_mol.as_graph(tsb_mol.parse_smiles(s)) for s in smiles]
    matrix = tsb_g0.matrix(graphs)
    assert matrix.shape == (50, tsb_g0.DIM)
    assert np.all(np.isfinite(matrix))
    assert len(set(tsb_g0.COLUMN_NAMES)) == tsb_g0.DIM
    assert tsb_g0.DIM == len(tsb_g0.spec()["column_names"])
