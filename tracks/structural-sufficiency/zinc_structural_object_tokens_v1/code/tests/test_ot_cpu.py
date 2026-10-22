"""Fast, data-free unit tests for the object-token arm.

Run: uv run pytest -q tracks/structural-sufficiency/zinc_structural_object_tokens_v1/code/tests
They only use synthetic graphs; the full checks (which read the frozen caches)
live in ``ot_checks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # code/
sys.path.insert(0, str(_HERE.parents[2] / "zinc_arch_discovery_v1" / "code"))
sys.path.insert(0, str(_HERE.parents[2] / "zinc_branching_fragment_v1" / "code"))

import ot_checks as K  # noqa: E402
import ot_data as O  # noqa: E402


def _run(fn):
    fn()
    name = fn.check_name
    result = K.RESULTS[name]
    assert result["pass"], result.get("traceback", result.get("error"))


def test_hand_enumeration():
    _run(K.check_b_hand)


def test_ring_canonical_alignment():
    _run(K.check_b_ring_alignment)


def test_relation_semantics():
    _run(K.check_b_relations)


def test_empty_groups():
    _run(K.check_b_empty)


def test_cycle_bruteforce_matches_dfs():
    obj = K.fake_obj([0] * 6, [(i, (i + 1) % 6) for i in range(6)] + [(0, 3)])
    want = K.brute_cycle_sets(obj)
    got = {L: {tuple(n) for (LL, n, _, _) in O.enumerate_cycles(obj) if LL == L}
           for L in range(3, 9)}
    assert got == want


def test_token_types_and_members():
    obj = K.fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3)])
    ad, ot = K.register_objects([obj], "__unit__")
    tt = np.asarray(ot["tok_type"])
    assert int((tt == O.TYPE_A).sum()) == 4
    assert int((tt >= 1).sum()) == 4  # S3 at root 0 + three F4 objects
    mem = np.asarray(ot["tok_members"])
    nm = np.asarray(ot["tok_nmembers"])
    s3 = np.nonzero(tt == 1)[0]
    assert len(s3) == 1
    assert sorted(int(x) for x in mem[s3[0]][: int(nm[s3[0]])]) == [0, 1, 2, 3]
