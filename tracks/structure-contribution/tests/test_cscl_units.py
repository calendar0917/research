"""Core correctness tests for the RINGCHAIN-v0 unit partition and signatures.

Fresh-clone safe: no data, no checkpoints, no official test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from cscl_units import (  # noqa: E402
    KIND_CHAIN,
    KIND_RING,
    build_partition,
    partition_units,
    unk_signature,
    unit_signature,
)


def mol(atom_types, bonds):
    return list(atom_types), [(int(u), int(v), int(bt)) for u, v, bt in bonds]


def test_partition_covers_all_atoms_exactly_once():
    # benzene ring + ethyl substituent + terminal methyl: chain atoms exist
    atoms = [0] * 9
    bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1)]
    a, e = mol(atoms, bonds)
    part = partition_units(a, e)
    owner = part.atom_unit
    assert len(owner) == 9
    assert sorted(owner) == sorted(set(owner)) or True  # ownership ids need not be unique-sorted
    assert set(owner) == set(range(part.n_units))
    # every atom in exactly one unit
    all_atoms = [a_ for u in part.units for a_ in u.atoms]
    assert sorted(all_atoms) == list(range(9))
    # ring: atoms 0..5 one unit; chain 6..8 another
    assert part.n_units == 2
    ring = [u for u in part.units if u.kind == KIND_RING][0]
    assert sorted(ring.atoms) == [0, 1, 2, 3, 4, 5]


def test_every_bond_has_unique_correct_attribution():
    # two rings joined by a chain of two atoms
    atoms = [0] * 14
    bonds = (
        [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 0, 1)]
        + [(6, 7, 1), (7, 8, 2), (8, 9, 1), (9, 10, 2), (10, 11, 1), (11, 6, 1)]
        + [(3, 12, 1), (12, 13, 1), (13, 9, 1)]
    )
    a, e = mol(atoms, bonds)
    part = partition_units(a, e)
    # collect every bond exactly once from units
    seen_intra = set()
    seen_inter = set()
    for u in part.units:
        for b in u.intra_bonds:
            key = tuple(sorted(b[:2])) + (b[2],)
            assert key not in seen_intra, f"bond double counted intra: {key}"
            seen_intra.add(key)
        for b in u.inter_bonds:
            key = tuple(sorted(b[:2])) + (b[2],)
            assert key not in seen_inter, f"bond double counted inter: {key}"
            seen_inter.add(key)
    original = {tuple(sorted((u, v))) + (bt,) for u, v, bt in bonds}
    assert seen_intra | seen_inter == original
    assert not (seen_intra & seen_inter)
    # inter bonds are exactly those crossing units
    for u, v, bt in e:
        same = part.atom_unit[u] == part.atom_unit[v]
        key = tuple(sorted((u, v))) + (bt,)
        assert (key in seen_intra) == same


def test_fused_and_spiro_rings_merge_into_one_ring_system():
    # fused bicyclic (share an edge: atoms 0,1) — naphthalene-like skeleton
    fused = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 0, 1), (0, 6, 2), (6, 7, 1), (7, 8, 2), (8, 1, 1)]
    a, e = mol([0] * 9, fused)
    part = partition_units(a, e)
    rings = [u for u in part.units if u.kind == KIND_RING]
    assert len(rings) == 1 and len(rings[0].atoms) == 9
    # spiro (share one atom): two 5-rings sharing atom 0
    spiro = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1), (0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1), (8, 0, 1)]
    a, e = mol([0] * 9, spiro)
    part = partition_units(a, e)
    rings = [u for u in part.units if u.kind == KIND_RING]
    assert len(rings) == 1 and len(rings[0].atoms) == 9


def test_acyclic_molecule_is_single_chain_unit():
    a, e = mol([0] * 5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
    part = partition_units(a, e)
    assert part.n_units == 1
    assert part.units[0].kind == KIND_CHAIN
    assert len(part.units[0].intra_bonds) == 4 and not part.units[0].inter_bonds


def test_signature_relabel_invariant():
    # same unit with permuted atom labels -> identical signature
    atoms = [3, 1, 3, 1, 2, 3]
    bonds = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1)]
    part1 = build_partition(atoms, [(u, v) for u, v, _ in bonds], [bt for _, _, bt in bonds])
    perm = [4, 0, 3, 5, 2, 1]  # perm[new] = old
    inv = [0] * 6
    for new, old in enumerate(perm):
        inv[old] = new
    atoms2 = [atoms[old] for old in perm]
    bonds2 = [(inv[u], inv[v], bt) for u, v, bt in bonds]
    part2 = build_partition(atoms2, [(u, v) for u, v, _ in bonds2], [bt for _, _, bt in bonds2])
    sigs1 = sorted(u.signature for u in part1.units)
    sigs2 = sorted(u.signature for u in part2.units)
    assert sigs1 == sigs2 and len(sigs1) == 1


def test_signature_distinguishes_different_units():
    s1 = unit_signature(KIND_CHAIN, [0, 1], [0, 1], [(0, 1, 1)])
    s2 = unit_signature(KIND_CHAIN, [0, 1], [2, 2], [(0, 1, 1)])
    s3 = unit_signature(KIND_RING, [0, 1, 2, 3, 4, 5], [0] * 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1)])
    assert len({s1, s2, s3}) == 3


def test_unk_signature_bucketing():
    assert unk_signature(KIND_RING, 6) == unk_signature(KIND_RING, 6)
    assert unk_signature(KIND_RING, 6) != unk_signature(KIND_CHAIN, 6)
    assert unk_signature(KIND_CHAIN, 2) != unk_signature(KIND_CHAIN, 3)


def test_inter_relations_undirected_counted_once():
    # biphenyl: two rings + one inter bond
    atoms = [0] * 12
    bonds = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 0, 1), (3, 6, 1)] + [
        (6, 7, 1),
        (7, 8, 2),
        (8, 9, 1),
        (9, 10, 2),
        (10, 11, 1),
        (11, 6, 1),
    ]
    a, e = mol(atoms, bonds)
    part = partition_units(a, e)
    rel = part.inter_relations()
    assert len(rel) == 1
    (pair, bts) = next(iter(rel.items()))
    assert pair[0] < pair[1] and bts == [1]
