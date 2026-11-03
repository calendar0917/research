"""Element mapping: six fixed classes, 21 unordered type pairs, distance bins."""

from __future__ import annotations

import itertools

import numpy as np

import nb_common as C
import nb_features as F


def test_element_classes_are_the_registered_six():
    assert C.ELEMENT_CLASSES == ("C", "N", "O", "S", "X", "OTHER")


def test_element_mapping_of_symbols():
    expected = {
        "C": "C",
        "N": "N",
        "O": "O",
        "S": "S",
        "F": "X",
        "Cl": "X",
        "Br": "X",
        "I": "X",
        "P": "OTHER",
        "B": "OTHER",
        "Si": "OTHER",
        "Se": "OTHER",
        "Na": "OTHER",
        "Al": "OTHER",
        "Ge": "OTHER",
        "H": "OTHER",  # never a node, but the mapping must still be total
    }
    for symbol, name in expected.items():
        assert C.ELEMENT_CLASSES[F.element_class(symbol)] == name, symbol


def test_aromatic_and_aliphatic_share_a_class():
    """Element class depends on the element only (aromaticity is irrelevant)."""
    aliphatic = F.pair_feature_molecule("CCCCCCOCCCCCC")  # pure single bonds
    aromatic = F.pair_feature_molecule("c1ccccc1O")  # aromatic ring
    assert set(np.asarray(aliphatic["classes"]).tolist()) == {
        C.ELEMENT_CLASSES.index("C"),
        C.ELEMENT_CLASSES.index("O"),
    }
    assert set(np.asarray(aromatic["classes"]).tolist()) == {
        C.ELEMENT_CLASSES.index("C"),
        C.ELEMENT_CLASSES.index("O"),
    }
    # pyridine vs piperidine: both nitrogens classify as N
    assert set(np.asarray(F.pair_feature_molecule("c1ccncc1")["classes"]).tolist()) == {
        C.ELEMENT_CLASSES.index("C"),
        C.ELEMENT_CLASSES.index("N"),
    }
    assert set(np.asarray(F.pair_feature_molecule("C1CCNCC1")["classes"]).tolist()) == {
        C.ELEMENT_CLASSES.index("C"),
        C.ELEMENT_CLASSES.index("N"),
    }
    assert F.element_class("N") == C.ELEMENT_CLASSES.index("N")


def test_type_pair_enumeration_is_complete_and_unique():
    assert len(C.TYPE_PAIRS) == 21
    assert len(set(C.TYPE_PAIRS)) == 21
    assert all(i <= j for i, j in C.TYPE_PAIRS)
    assert set(C.TYPE_PAIRS) == {(min(a, b), max(a, b)) for a, b in itertools.combinations_with_replacement(range(6), 2)}


def test_type_pair_index_is_symmetric_and_bijective():
    seen = set()
    for i, j in C.TYPE_PAIRS:
        p = F.type_pair_index(i, j)
        seen.add(p)
        assert F.type_pair_index(j, i) == p
        assert C.TYPE_PAIRS[p] == (i, j)
    assert seen == set(range(21))


def test_type_pair_index_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        F.type_pair_index(0, 6)
    with pytest.raises(ValueError):
        F.type_pair_index(-1, 2)


def test_distance_bins_are_the_registered_six():
    assert C.DIST_BIN_LABELS == ("5", "6", "7", "8", "9", ">=10")
    assert C.DIST_BIN_LOWER == (5, 6, 7, 8, 9, 10)
    assert [F.distance_bin(d) for d in range(4, 13)] == [-1, 0, 1, 2, 3, 4, 5, 5, 5]
    assert F.distance_bin(999) == 5
    assert F.distance_bin(-1) == -1
    assert C.N_PAIR_FEATURES == 126
    assert len(C.PAIR_FEATURE_LAYOUT) == 126
