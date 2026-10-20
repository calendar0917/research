"""Lightweight CPU tests for zinc-exact-structure-residual-v1.

No ZINC data, no checkpoint and no GPU is required.  The tests cover the
properties the protocol depends on: orthonormal non-zero P, exact zero forward
at beta=0, the two prior formulas, ancestor-closed deterministic truncation and
the deepest-retained-type fallback rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
_REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, _REPO, _REPO / "tracks/structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import esr_data as D  # noqa: E402
import esr_models as M  # noqa: E402


def _keys(prefix: str, n: int) -> list[bytes]:
    return [f"{prefix}-{i}".encode() for i in range(n)]


def test_orthonormal_columns_deterministic_and_nonzero():
    p = M.orthonormal_columns(96, 4, seed=3)
    q = M.orthonormal_columns(96, 4, seed=3)
    assert torch.equal(p, q)
    gram = p.t() @ p
    assert torch.allclose(gram, torch.eye(4), atol=1e-6)
    assert torch.all(p.norm(dim=0) > 0.999)
    # sign convention: the largest-magnitude entry of every column is positive
    for j in range(4):
        assert float(p[:, j][p[:, j].abs().argmax()]) > 0.0


def test_zero_beta_forward_is_exactly_zero():
    depth = np.array([1, 2], dtype=np.int32)
    parent = np.array([-1, 0], dtype=np.int32)
    r = M.AtomTypeResidual(2, depth, parent, atom_out=96, rank=4, seed=0, arm="E", sh2=10.0)
    types = torch.tensor([0, 1, -1], dtype=torch.long)
    out = r(types)
    assert out.shape == (3, 96)
    assert torch.equal(out, torch.zeros_like(out))
    assert float(r.prior()[0]) == 0.0


def test_forward_lookup_and_negative_type():
    depth = np.array([1, 2, 2], dtype=np.int32)
    parent = np.array([-1, 0, 0], dtype=np.int32)
    r = M.AtomTypeResidual(3, depth, parent, atom_out=6, rank=2, seed=1, arm="E", sh2=1.0)
    with torch.no_grad():
        r.beta.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5]]))
    types = torch.tensor([0, 2, -1], dtype=torch.long)
    out = r(types)
    expect = r.P @ r.beta.t()
    assert torch.allclose(out[0], expect[:, 0], atol=1e-6)
    assert torch.allclose(out[1], expect[:, 2], atol=1e-6)
    assert torch.equal(out[2], torch.zeros(6))


def test_prior_E_and_H_formula():
    depth = np.array([1, 2, 2], dtype=np.int32)
    parent = np.array([-1, 0, 0], dtype=np.int32)
    sh2 = 4.0
    r = M.AtomTypeResidual(3, depth, parent, atom_out=8, rank=3, seed=2, arm="E", sh2=sh2)
    with torch.no_grad():
        r.beta.copy_(torch.tensor([[0.2, 0.0, 0.1], [0.4, 0.3, 0.0], [0.0, 0.0, 0.0]]))
    a = r.correction_matrix()  # [8, 3]
    per_e_manual = (a * a).sum(0) / (torch.tensor([1, 2, 2], dtype=a.dtype) * sh2)
    mean_e, per_e = r.prior()
    assert torch.allclose(per_e, per_e_manual, atol=1e-6)
    assert torch.allclose(mean_e, per_e_manual.mean(), atol=1e-6)

    r.arm = "H"
    mean_h, per_h_types = r.prior()
    diff = torch.empty_like(a)
    diff[:, 0] = a[:, 0]
    diff[:, 1] = a[:, 1] - a[:, 0]
    diff[:, 2] = a[:, 2] - a[:, 0]
    per_h_manual = (diff * diff).sum(0) / sh2
    assert torch.allclose(mean_h, per_h_manual.mean(), atol=1e-6)
    assert torch.allclose(per_h_types, per_h_manual, atol=1e-6)


def _records():
    """Two synthetic molecules; r2 types 0,1 in mol0 and 1,2 in mol1."""
    r1 = _keys("r1", 3)
    r2 = _keys("r2", 3)
    return [
        {"r2": [r2[0], r2[1]], "r1": [r1[0], r1[1]], "root_atom_type": [0, 1]},
        {"r2": [r2[1], r2[2]], "r1": [r1[1], r1[2]], "root_atom_type": [1, 2]},
    ]


def test_vocab_ancestor_closure_and_support_unit():
    vocab = D.build_vocab(_records(), n_train_molecules=2, atom_out=8, rank=2, base_parameters=100, param_cap=10_000)
    depth = vocab["depth"]
    parent = vocab["parent"]
    assert vocab["n_r1"] == 3 and vocab["n_r2"] == 3
    assert np.all(depth[parent[depth == 2]] == 1)
    # support = number of distinct molecules, not occurrences
    support = dict(zip(vocab["keys"], vocab["support"]))
    assert support[_keys("r2", 3)[1]] == 2
    assert support[_keys("r2", 3)[0]] == 1


def test_vocab_truncation_is_deterministic_and_charges_ancestors():
    records = _records()
    small = D.build_vocab(records, n_train_molecules=2, atom_out=8, rank=2, base_parameters=100, param_cap=100 + 16 + 2 * 4)
    again = D.build_vocab(records, n_train_molecules=2, atom_out=8, rank=2, base_parameters=100, param_cap=100 + 16 + 2 * 4)
    assert small["keys"] == again["keys"]
    assert small["n_types"] <= 4
    assert small["total_inference_parameters"] <= 100 + 16 + 2 * 4
    # every retained r2 has its r1 ancestor retained
    r1_set = set(small["keys"][: small["n_r1"]])
    for i, d in enumerate(small["depth"]):
        if int(d) == 2:
            assert small["keys"][int(small["parent"][i])] in r1_set


def test_assign_types_deepest_retained_fallback():
    vocab = {
        "keys": [b"r1-a", b"r2-a"],
        "depth": np.array([1, 2], dtype=np.int32),
        "support": np.array([5, 5], dtype=np.int32),
        "parent": np.array([-1, 0], dtype=np.int32),
        "root_atom_type": np.array([1, 1], dtype=np.int32),
        "n_types": 2,
        "index": {b"r1-a": 0, b"r2-a": 1},
    }
    assigned = D.assign_types(
        [{"r2": [b"r2-a", b"r2-unknown"], "r1": [b"r1-a", b"r1-a"]}],
        vocab,
    )[0]
    assert assigned["t_star"].tolist() == [1, 0]  # deep when seen, parent when OOV
    assert assigned["r2_oov"].tolist() == [0, 1]
    assert assigned["zero_correction"].tolist() == [0, 0]
    # an OOV r1 also falls back to the zero correction
    assigned2 = D.assign_types([{"r2": [b"x"], "r1": [b"y"]}], vocab)[0]
    assert assigned2["t_star"].tolist() == [-1]
    assert assigned2["zero_correction"].tolist() == [1]


def test_sy_definition():
    y = np.array([0.0, 1.0, 2.0, 10.0])
    out = D.sy_from_labels(y)
    med = 1.5
    assert out["median"] == pytest.approx(med)
    assert out["sy"] == pytest.approx(np.abs(y - med).mean())


def test_parameter_budget_accounting():
    vocab = D.build_vocab(_records(), n_train_molecules=2, atom_out=96, rank=4, base_parameters=413_505, param_cap=500_000)
    assert vocab["total_inference_parameters"] == 413_505 + 96 * 4 + 4 * vocab["n_types"]
    assert vocab["total_inference_parameters"] <= 500_000
    assert vocab["fixed_residual_parameters"] == 384
