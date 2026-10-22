"""Fast CPU tests for the branching-fragment representation (no datasets)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
_V1 = _CODE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_CODE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bf_data as B  # noqa: E402


def fake_obj(categories, edges):
    cats = np.asarray(categories, dtype=np.int64)
    n = len(cats)
    feat = np.zeros((n, 108), dtype=np.float32)
    feat[np.arange(n), cats] = 1.0
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return SimpleNamespace(
        n_atoms=n,
        atom_feat=feat,
        bond_index=edges,
        bond_type=np.ones((len(edges),), dtype=np.int64),
    )


def counts(obj):
    per = B.enumerate_rooted_branching(obj)
    return {f: int(sum(sum(d.values()) for d in per[i].values()))
            for i, f in enumerate(B.FAMILIES)}


def test_key_layout_and_root_slot():
    obj = fake_obj([3, 1, 2, 1], [(0, 1), (0, 2), (0, 3)])
    per = B.enumerate_rooted_branching(obj)
    key = next(iter(per[0][0]))
    assert len(key) == B.KEY_LEN == 15
    raw = np.frombuffer(key, dtype=np.uint8)
    assert int(raw[0]) == 3  # root atom category at slot 0
    assert int(raw[4]) == 0  # unused atom slot
    pair = raw[B.MAX_NODES:]
    assert int((pair == 1).sum()) == 3  # root-to-leaf bonds


def test_canonicalisation_is_order_invariant():
    cats = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    bmap = {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 1}
    a = B.canonical_key(cats, bmap, 0, (1, 2, 3))
    b = B.canonical_key(cats, bmap, 0, (3, 1, 2))
    c = B.canonical_key(cats, bmap, 0, (2, 3, 1))
    assert a == b == c


def test_extra_edge_changes_content_not_count():
    plain = fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3)])
    closed = fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3), (1, 2)])
    kp = next(iter(B.enumerate_rooted_branching(plain)[0][0]))
    kc = next(iter(B.enumerate_rooted_branching(closed)[0][0]))
    assert kp != kc
    pp = np.frombuffer(kp, dtype=np.uint8)[B.MAX_NODES:]
    pc = np.frombuffer(kc, dtype=np.uint8)[B.MAX_NODES:]
    assert int((pp == 1).sum()) == 3 and int((pc == 1).sum()) == 4


def test_hand_counts():
    star3 = fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3)])
    star4 = fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)])
    path = fake_obj([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    fork = fake_obj([0, 1, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (1, 4)])
    assert counts(star3) == {"S3": 1, "S4": 0, "F4": 3}
    assert counts(star4) == {"S3": 4, "S4": 1, "F4": 12}
    assert counts(path) == {"S3": 0, "S4": 0, "F4": 0}
    assert counts(fork) == {"S3": 1, "S4": 0, "F4": 3}


def test_f4_dedup_on_k4():
    k4 = fake_obj([0, 1, 2, 3],
                  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    per = B.enumerate_rooted_branching(k4)[2]
    for root in range(4):
        assert sum(per[root].values()) == 1, per[root]


def test_multiplicity_kept_for_equal_content():
    star4 = fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)])
    per = B.enumerate_rooted_branching(star4)
    assert len(per[0][0]) == 1 and sum(per[0][0].values()) == 4  # C(4,3)
    assert len(per[1][0]) == 1 and sum(per[1][0].values()) == 1


def test_mask_layout():
    length = np.array([4, 5], dtype=np.int64)
    node_mask, pair_mask = B.pattern_masks(length)
    assert node_mask.sum(axis=1).tolist() == [4, 5]
    assert pair_mask.sum(axis=1).tolist() == [6, 10]


def test_cache_roundtrip_and_metadata():
    star4 = fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)])
    pc = B.build_from_objects([star4], "__test__", {"source": "unit"}, verbose=False)
    assert pc["meta"]["labels_used"] is False
    assert pc["meta"]["test_used"] is False
    assert int(pc["meta"]["occurrences_by_family"]["S3"]) == 4
    assert int(pc["meta"]["occurrences_by_family"]["S4"]) == 1
    assert int(pc["meta"]["occurrences_by_family"]["F4"]) == 12
    pa = np.asarray(pc["pat_atom"])
    pp = np.asarray(pc["pat_pair"])
    pl = np.asarray(pc["pat_len"])
    nm, pm = B.pattern_masks(pl)
    for p in range(len(pl)):
        assert int(pa[p, int(pl[p]):].max(initial=0)) == 0
        assert np.isin(pp[p][pm[p]], (0, 1, 2, 3)).all()
