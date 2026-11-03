"""Distance graph, pair counting, distance bins, normalisation and NULL margins."""

from __future__ import annotations

import numpy as np
import pytest

import nb_common as C
import nb_features as F


def _chain(n: int) -> str:
    return "C" * n


def test_heavy_graph_on_a_single_chain():
    g = F.heavy_graph(_chain(7))  # C-C-...-C, 7 heavy atoms
    assert g["n_heavy_atoms"] == 7
    assert g["n_heavy_bonds"] == 6
    assert g["n_components"] == 1
    dist = np.asarray(g["distance"])
    assert dist[0, 6] == 6
    assert dist[3, 4] == 1
    assert np.array_equal(dist, dist.T)
    assert np.all(np.diag(dist) == 0)


def test_explicit_hydrogen_is_not_a_node():
    plain = F.heavy_graph("CCCCCCOCCCCCC")
    with_h = F.heavy_graph("[H]CCCCCCOCCCCCC")  # one explicit H on a terminal carbon
    assert with_h["n_heavy_atoms"] == plain["n_heavy_atoms"]
    assert with_h["n_hydrogen_nodes_removed"] == plain["n_hydrogen_nodes_removed"] + 1
    assert np.array_equal(np.asarray(plain["distance"]), np.asarray(with_h["distance"]))


def test_isolated_hydrogen_and_salt_components_are_excluded():
    salt = F.pair_feature_molecule("[Na+].CCCCCCCCCC")  # Na is one heavy atom component
    assert salt["n_components_heavy"] == 2
    # the Na atom can never form a pair, so all counted pairs are inside the chain
    expected = F.pair_feature_molecule("CCCCCCCCCC")["total"]
    assert salt["total"] == expected


def test_pairs_across_components_are_never_counted():
    two_chains = F.pair_feature_molecule("CCCCCCCCCC.CCCCCCCCCC")
    one_chain = F.pair_feature_molecule("CCCCCCCCCCCCCCCCCCCC")
    assert two_chains["total"] < one_chain["total"]
    # cross-component distances are -1 and stay uncounted
    dist = np.asarray(two_chains["distance"])
    assert (dist < 0).any()


def test_pair_counting_respects_the_five_bond_minimum():
    rec = F.pair_feature_molecule("CCCCCC")  # 6 atoms, max distance 5
    assert rec["total"] == 1
    assert rec["counts"][C.TYPE_PAIRS.index((0, 0)), F.distance_bin(5)] == 1
    short = F.pair_feature_molecule("CCCCC")  # max distance 4
    assert short["total"] == 0
    assert np.all(short["real"] == 0) and np.all(short["null"] == 0)


def test_normalisation_divides_by_the_pair_total():
    rec = F.pair_feature_molecule("CCCCCCCCCCCC")
    assert rec["scale"] == float(rec["total"])
    assert np.isclose(rec["real"].sum(), 1.0)
    assert np.isclose(rec["null"].sum(), 1.0)


def test_distance_bin_assignment_in_the_matrix():
    rec = F.pair_feature_molecule("CCCCCC")  # one C-C pair at distance 5
    matrix = rec["counts"]
    assert matrix[0, 0] == 1
    assert matrix.sum() == 1


def test_null_is_the_outer_product_of_the_margins():
    rec = F.pair_feature_molecule(C.EXPECTED_SMILES_A)
    raw = rec["counts"].astype(np.float64)
    total = raw.sum()
    expected = np.outer(raw.sum(axis=1), raw.sum(axis=0)) / total
    assert np.allclose((rec["null"] * rec["scale"]).reshape(C.N_TYPE_PAIRS, C.N_DIST_BINS), expected, atol=1e-10)


def test_null_margins_conserved_per_molecule():
    smiles = [
        C.EXPECTED_SMILES_A,
        C.EXPECTED_SMILES_B,
        "CCCCCCCCCCCC",
        "CCCCCC",
        "CCO",  # no pair with distance >= 5
        "c1ccccc1CCCCCCCCO",  # aromatic + long tail
        "[Na+].CCCCCCCCCC",
    ]
    for s in smiles:
        rec = F.pair_feature_molecule(s)
        data = {
            "real": rec["real"][None, :],
            "null": rec["null"][None, :],
            "totals": np.asarray([rec["total"]], dtype=np.int32),
            "row_margin": rec["row_margin"][None, :],
            "col_margin": rec["col_margin"][None, :],
        }
        res = F.margin_residuals(data)
        assert res["row_margin_conserved"], s
        assert res["col_margin_conserved"], s
        assert res["total_conserved"], s
        assert res["raw_row_margin_matches_counts"], s
        assert res["raw_col_margin_matches_counts"], s
        assert res["raw_total_matches_counts"], s
        if rec["total"] == 0:
            assert np.all(rec["real"] == 0) and np.all(rec["null"] == 0)


def test_null_margins_conserved_on_a_real_batch():
    """A mixed batch over real MolHIV Train SMILES (no labels involved)."""
    data = F.build_feature_matrices(
        [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "OCC1OC(O)C(O)C(O)C1O",
            "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
            "CC1=C(C(=O)N(N1)C2=CC=CC=C2)C3=CC=CC=C3",
            "CCCCCCCCCCCCCCCCCCCCCCCC",
            "c1ccc2c(c1)ccc3c2cccc3",
        ],
        label="test-batch",
    )
    res = F.margin_residuals(data)
    assert res["row_margin_conserved"] and res["col_margin_conserved"] and res["total_conserved"]
    assert res["row_margin_max_abs_residual"] <= 1e-8
    assert res["col_margin_max_abs_residual"] <= 1e-8
    assert res["raw_row_margin_matches_counts"] and res["raw_col_margin_matches_counts"]


def test_real_and_null_are_not_the_same_matrix():
    a = F.pair_feature_molecule(C.EXPECTED_SMILES_A)
    assert not np.array_equal(a["real"], a["null"])
    # ...but they must have the same total mass after the shared normalisation
    assert np.isclose(a["real"].sum(), a["null"].sum())


def test_real_equals_null_when_the_count_matrix_is_rank_one():
    """A degenerate control: if counts are exactly rank one, REAL == NULL."""
    counts = np.zeros((C.N_TYPE_PAIRS, C.N_DIST_BINS))
    counts[:, 0] = np.arange(1, C.N_TYPE_PAIRS + 1)  # only one populated bin
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    total = counts.sum()
    null = np.outer(row, col) / total
    assert np.allclose(counts, null)


@pytest.mark.parametrize("smiles", [C.EXPECTED_SMILES_A, C.EXPECTED_SMILES_B, "c1ccccc1CCCCCCCC", "CCO"])
def test_feature_vector_shapes(smiles):
    rec = F.pair_feature_molecule(smiles)
    assert rec["real"].shape == (C.N_PAIR_FEATURES,)
    assert rec["null"].shape == (C.N_PAIR_FEATURES,)
    assert rec["counts"].shape == (C.N_TYPE_PAIRS, C.N_DIST_BINS)
    assert np.isfinite(rec["real"]).all() and np.isfinite(rec["null"]).all()
    assert (rec["real"] >= 0).all() and (rec["null"] >= 0).all()
