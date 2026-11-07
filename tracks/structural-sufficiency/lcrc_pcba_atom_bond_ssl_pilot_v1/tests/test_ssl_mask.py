"""Fast CPU tests for the frozen mask/probe definitions (no cache access)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import ssl_common as C
import ssl_data as D


def test_mask_count_rounds_half_up_and_keeps_one():
    assert C.n_masked(0) == 0
    assert C.n_masked(1) == 1
    assert C.n_masked(3) == 1
    assert C.n_masked(10) == 2      # 1.5 -> 2
    assert C.n_masked(20) == 3
    assert C.n_masked(26) == 4      # 3.9 -> 4
    assert C.n_masked(100) == 15
    # never zero for a non-empty set, never more than 15%+1
    for n in range(1, 500):
        k = C.n_masked(n)
        assert 1 <= k <= max(1, n)


def test_mask_seed_bytes_are_isolated():
    a = C.mask_seed_bytes(C.MASK_ATOM_TAG, 0, 42, 0)
    b = C.mask_seed_bytes(C.MASK_BOND_TAG, 0, 42, 0)
    c = C.mask_seed_bytes(C.MASK_ATOM_TAG, 0, 42, 1)
    d = C.mask_seed_bytes(C.MASK_ATOM_TAG, 0, 43, 0)
    assert len({a, b, c, d}) == 4


def test_sample_molecule_mask_is_deterministic_and_minimal():
    first = D.sample_molecule_mask(row_id=12345, visit_index=0, n_atoms=26, n_bonds=28)
    again = D.sample_molecule_mask(row_id=12345, visit_index=0, n_atoms=26, n_bonds=28)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    atoms, bonds = first
    assert atoms.size == C.n_masked(26)
    assert bonds.size == C.n_masked(28)
    assert np.all(np.diff(atoms) > 0) and np.all(np.diff(bonds) > 0)
    assert atoms.min() >= 0 and atoms.max() < 26
    assert bonds.min() >= 0 and bonds.max() < 28


def test_mask_changes_with_visit_and_row():
    a = D.sample_molecule_mask(row_id=7, visit_index=0, n_atoms=30, n_bonds=30)
    b = D.sample_molecule_mask(row_id=7, visit_index=1, n_atoms=30, n_bonds=30)
    c = D.sample_molecule_mask(row_id=8, visit_index=0, n_atoms=30, n_bonds=30)
    assert not np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_atom_and_bond_streams_are_independent():
    atoms, bonds = D.sample_molecule_mask(row_id=99, visit_index=2, n_atoms=40, n_bonds=40)
    # different tag -> different generator; identical index sets would mean a
    # shared stream, which must not happen for a 40-object draw
    assert not np.array_equal(atoms, bonds)


def test_bondless_molecule_has_no_bond_mask():
    atoms, bonds = D.sample_molecule_mask(row_id=5, visit_index=0, n_atoms=1, n_bonds=0)
    assert atoms.size == 1 and bonds.size == 0


def test_probe_selection_rule_and_determinism():
    train = np.arange(9000, dtype=np.int64)
    probe = D.probe_rows_from_train(train)
    assert probe.size == C.N_PROBE
    assert np.all(np.diff(probe) > 0)
    expected = sorted(
        train.tolist(),
        key=lambda r: (
            hashlib.sha256(f"{C.PROBE_ENTROPY}{int(r)}".encode("utf-8")).hexdigest(),
            int(r),
        ),
    )[: C.N_PROBE]
    assert np.array_equal(probe, np.sort(np.asarray(expected, dtype=np.int64)))
    again = D.probe_rows_from_train(train)
    assert np.array_equal(probe, again)


def test_prior_ce_matches_manual_entropy():
    freqs = [0.5, 0.25, 0.25]
    targets = np.array([0, 1, 2])
    import math

    expected = -np.mean([math.log(0.5), math.log(0.25), math.log(0.25)])
    assert D.prior_ce(targets, freqs) == pytest.approx(expected)


def test_decision_bands_are_the_frozen_ones():
    assert C.decision_label(0.02) == "PROMISING_SINGLE_SEED"
    assert C.decision_label(0.005) == "PROMISING_SINGLE_SEED"
    assert C.decision_label(0.002) == "WEAK_SIGNAL"
    assert C.decision_label(0.0) == "NO_CLEAR_GAIN"
    assert C.decision_label(-0.0019) == "NO_CLEAR_GAIN"
    assert C.decision_label(-0.002) == "NEGATIVE_TRANSFER"
    assert C.AP40_A0 == pytest.approx(0.24635609609821457)


def test_encoder_geometry_constants():
    assert C.ATOM_ENCODER_PARAMETERS == 51 * 128 + 128 + 128 * 96 + 96
    assert C.BOND_ENCODER_PARAMETERS == 301 * 160 + 160 + 160 * 96 + 96
    assert C.ENCODER_PARAMETERS == 82816
    assert C.ATOM_HEAD_INPUT == 288
    assert C.BOND_HEAD_INPUT == 480
    assert C.EXPECTED_PARAMETERS == 287644
    assert C.EXPECTED_BASE_PARAMETERS + C.EXPECTED_NEW_PARAMETERS == C.EXPECTED_PARAMETERS
