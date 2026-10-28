"""Fast, data-free regression tests for LCRC-v1.

These do not touch the ZINC cache, labels, checkpoints or the official test
split.  They exercise the model math on constructed graphs; the full gate suite
is ``code/lc_cpu_checks.py`` (which does require the local label-free cache).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import lc_common as L  # noqa: E402
import lc_data as D  # noqa: E402
import lc_models as M  # noqa: E402


def _norm():
    return {
        "global_mean": np.zeros(L.GLOBAL_DIM, np.float32),
        "global_scale": np.ones(L.GLOBAL_DIM, np.float32),
        "scalar_mean": np.zeros(L.N_SCALARS, np.float32),
        "scalar_scale": np.ones(L.N_SCALARS, np.float32),
        "source": "test",
    }


def _molecule():
    n = 6
    atom_feat = np.zeros((n, L.ATOM_FEAT_DIM), np.float32)
    for i in range(n):
        atom_feat[i, 0] = 1.0
        atom_feat[i, 28 + 2] = 1.0
        atom_feat[i, 33] = 1.0
        atom_feat[i, 34 + 3] = 1.0
        atom_feat[i, 40] = 2.0
    bi = np.asarray([[i, (i + 1) % n] for i in range(n)], np.int64)
    bt = np.ones(n, np.int64)
    bf = np.zeros((n, L.BOND_FEAT_DIM), np.float32)
    bf[:, 0] = 1.0
    bf[:, 4] = 1.0
    return D.build_from_arrays(atom_feat, np.zeros(n, np.int64), bi, bt, bf,
                               np.zeros(L.GLOBAL_DIM, np.float32))


def _cache(mols):
    c = D._concat_molecules(mols)
    c["n_molecules"] = len(mols)
    c["labels"] = np.zeros(len(mols), np.float32)
    c["meta"] = {}
    return c


def test_arms_same_initial_prediction_and_param_cap():
    norm = _norm()
    cache = _cache([_molecule()])
    batch = D.collate(cache, [0])
    preds = {}
    for arm in ("A", "R", "N"):
        model, info = M.build_model(arm, norm, 0, "cpu")
        assert info["parameters"] <= 500_000
        assert not info["has_junction_module"]
        model.eval()
        with torch.no_grad():
            preds[arm] = model(batch).numpy()
    assert np.array_equal(preds["A"], preds["R"])
    assert np.array_equal(preds["A"], preds["N"])


def test_renumbering_and_bond_direction_invariance():
    norm = _norm()
    mol = _molecule()
    base = _cache([mol])
    perm = np.asarray([3, 0, 5, 2, 1, 4], np.int64)
    perm_mol = D.build_from_arrays(
        mol["atom_feat"], mol["atom_type"], mol["bond_index"], mol["bond_type"],
        mol["bond_feat"], mol["global"], perm=perm,
    )
    rev_mol = D.build_from_arrays(
        mol["atom_feat"], mol["atom_type"], mol["bond_index"], mol["bond_type"],
        mol["bond_feat"], mol["global"], reverse_bonds=True,
    )
    for arm in ("A", "R", "N"):
        model, _ = M.build_model(arm, norm, 0, "cpu")
        model.eval()
        with torch.no_grad():
            p0 = model(D.collate(base, [0])).numpy()
            p1 = model(D.collate(_cache([perm_mol]), [0])).numpy()
            p2 = model(D.collate(_cache([rev_mol]), [0])).numpy()
        assert np.max(np.abs(p0 - p1)) < 1e-5
        assert np.max(np.abs(p0 - p2)) < 1e-5


def test_r_and_n_differ_when_correspondence_differs():
    torch.manual_seed(3)
    rel = M.RelationHead(M.DEFAULT_CONFIG)
    torch.manual_seed(4)
    with torch.no_grad():
        rel.head.weight.normal_(0, 0.1)
    h = torch.randn(8, 64)
    pairs = torch.as_tensor([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [4, 6]])
    cat_a = torch.as_tensor([0, 0, 1, 1, 2, 2])
    cat_b = torch.as_tensor([2, 2, 1, 1, 0, 0])
    graph = torch.zeros(6, dtype=torch.long)
    z = rel.pair_content(h, pairs[:, 0], pairs[:, 1])
    Ma, na, ea = rel.aggregate(z, graph, cat_a, 1, "R")
    Mb, nb, eb = rel.aggregate(z, graph, cat_b, 1, "R")
    Na, _, _ = rel.aggregate(z, graph, cat_a, 1, "N")
    Nb, _, _ = rel.aggregate(z, graph, cat_b, 1, "N")
    assert torch.allclose(na, nb)
    assert (Ma - Mb).abs().max() > 1e-5
    assert (Na - Nb).abs().max() == 0.0
    assert (rel.readout(Ma, na, ea) - rel.readout(Mb, nb, eb)).abs() > 1e-6
