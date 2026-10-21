"""Small CPU regression tests for the induced-pattern pilot.

These are deliberately narrow: enumeration/multiplicity, the six-ring closure
and the Q masking.  The broader CPU audit lives in ``ip_cpu_checks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_HERE = Path(__file__).resolve().parents[1]
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ip_data  # noqa: E402
import ip_models  # noqa: E402


def _obj(categories, edges):
    cats = np.asarray(categories, dtype=np.int64)
    n = len(cats)
    feat = np.zeros((n, 108), dtype=np.float32)
    feat[np.arange(n), cats] = 1.0
    return SimpleNamespace(
        n_atoms=n,
        atom_feat=feat,
        bond_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        bond_type=np.ones((len(edges),), dtype=np.int64),
    )


def test_square_enumeration_counts():
    p = ip_data.enumerate_rooted_paths(_obj([0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 0)]))
    totals = {l: sum(sum(d.values()) for d in p[l].values()) for l in ip_data.LENGTHS}
    assert totals == {2: 8, 3: 8, 4: 0, 5: 0}
    only = next(iter(p[2].values()))
    assert only == {0: 2, 1: 2, 2: 2, 3: 2}


def test_six_ring_closure_and_q_masking():
    ring = _obj([0] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    chain = _obj([0] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    pr = ip_data.enumerate_rooted_paths(ring)[5]
    pc = ip_data.enumerate_rooted_paths(chain)[5]
    rk, ck = next(iter(pr)), next(iter(pc))
    idx = ip_data.PAIR_INDEX.index((0, 5))
    rp = np.frombuffer(rk[6:], dtype=np.uint8)
    cp = np.frombuffer(ck[6:], dtype=np.uint8)
    assert rp[idx] == 1 and cp[idx] == 0
    assert rk != ck
    assert ip_data.q_key_from_key(rk) == ip_data.q_key_from_key(ck)


def test_model_parameter_cap_and_zero_increment():
    branch = ip_models.RootedPathBranch()
    n_branch = sum(p.numel() for p in branch.parameters())
    assert n_branch == 70_776
    batch = {
        "pat_atom": torch.zeros((0, 6), dtype=torch.long),
        "pat_pair": torch.zeros((0, 15), dtype=torch.long),
        "pat_len": torch.zeros((0,), dtype=torch.long),
        "pat_node_mask": torch.zeros((0, 6), dtype=torch.bool),
        "pat_pair_mask": torch.zeros((0, 15), dtype=torch.bool),
        "occ_pat": torch.zeros((0,), dtype=torch.long),
        "occ_len": torch.zeros((0,), dtype=torch.long),
        "occ_root": torch.zeros((0,), dtype=torch.long),
        "occ_count": torch.zeros((0,), dtype=torch.float32),
        "n_atoms_total": 2,
    }
    with torch.no_grad():
        d = branch.eval()(batch)
    assert d.shape == (2, 96)
    assert torch.equal(d, torch.zeros_like(d))
