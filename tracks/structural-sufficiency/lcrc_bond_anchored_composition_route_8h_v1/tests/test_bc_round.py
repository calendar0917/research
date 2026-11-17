"""CPU-only structural tests for the BAJC round (no training, no GPU).

These tests reproduce the parts of the Stage-0 gates that must never regress:
the event object, the reversal symmetry, the zero residual, the residual-off
identity and the paired-control identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import bc_common as C  # noqa: E402
import bc_data as D  # noqa: E402
import bc_models as M  # noqa: E402
import bc_path3 as P3  # noqa: E402
import bc_train as T  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    return D.bundle("lipo")


@pytest.fixture(scope="module")
def paths3(bundle):
    b = bundle
    out = {}
    for split in ("valid",):
        out[split] = P3.build_split(b.caches[split], b.spec, b.dataset, split)
    return out


def test_event_object_is_well_formed(bundle, paths3):
    p3 = paths3["valid"]
    assert p3.n_molecules == 420
    assert p3.n_events > 0
    atoms = p3.atoms
    assert atoms.shape[1] == 4
    assert np.all(p3.counts > 0)
    # canonical direction: (a,b,c,d) <= (d,c,b,a)
    assert np.all(
        [tuple(row) <= (row[3], row[2], row[1], row[0]) for row in atoms.tolist()]
    )
    # four distinct atoms in every event
    assert all(len(set(row)) == 4 for row in atoms.tolist())
    assert set(np.unique(p3.closure4).tolist()) <= {0, 1}


def test_occurrences_resolve_to_the_frozen_enumeration(bundle, paths3):
    p3 = paths3["valid"]
    occ = p3.occ
    a = p3.atoms
    root, end = p3.occ_root, p3.occ_end
    mid = p3.occ_map.path_middle
    assert np.array_equal(root[occ[:, 0]], a[:, 0])
    assert np.array_equal(mid[occ[:, 0]], a[:, 1])
    assert np.array_equal(end[occ[:, 0]], a[:, 2])
    assert np.array_equal(root[occ[:, 1]], a[:, 1])
    assert np.array_equal(mid[occ[:, 1]], a[:, 2])
    assert np.array_equal(end[occ[:, 1]], a[:, 3])
    assert np.array_equal(root[occ[:, 2]], a[:, 3])
    assert np.array_equal(end[occ[:, 2]], a[:, 1])
    assert np.array_equal(root[occ[:, 3]], a[:, 2])
    assert np.array_equal(end[occ[:, 3]], a[:, 0])


def test_anchor_bond_is_the_middle_bond(bundle, paths3):
    p3 = paths3["valid"]
    cache = bundle.caches["valid"]
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    counts = np.diff(np.asarray(cache["bond_ptr"], dtype=np.int64))
    offset = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    molecule = np.repeat(np.arange(p3.n_molecules), p3.counts)
    rows = offset[molecule] + p3.anchor
    u, v = bond_index[rows, 0], bond_index[rows, 1]
    assert np.array_equal(np.sort(np.stack([u, v], axis=1), axis=1),
                          np.sort(np.stack([p3.atoms[:, 1], p3.atoms[:, 2]], axis=1), axis=1))


def test_zero_residual_and_residual_off(bundle, paths3):
    model, audit = T.build_model("lipo", bundle, "JOINT", "E2E", seed=0)
    model.eval()
    bt = D.make_batch(bundle, "valid", np.arange(0, 16), paths3["valid"])
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
        off = model(bt.batch, bt.p3, branch=False, return_parts=True)
    assert float(parts["delta_comp"].abs().max()) == 0.0
    assert torch.equal(parts["pred_std"], parts["pred_base_std"])
    assert torch.equal(off["pred_std"], parts["pred_base_std"])
    assert audit["parameter_report"]["branch"] == 17392
    assert audit["parameter_report"]["branch_within_budget"]


def test_reversal_symmetry(bundle, paths3):
    model, _ = T.build_model("lipo", bundle, "JOINT", "E2E", seed=0)
    gen = torch.Generator().manual_seed(4242)
    with torch.no_grad():
        for p in model.composer.parameters():
            p.add_(torch.empty_like(p).normal_(0.0, 0.05, generator=gen))
    model.eval()
    bt = D.make_batch(bundle, "valid", np.arange(0, 16), paths3["valid"])
    with torch.no_grad():
        evidence = model.s2_evidence(bt.batch, model.trunk.atom_encoder(bt.batch["atom_feat"]))
        bond = _bond(model, bt.batch)
        forward = model.compose(evidence, bond, bt.p3, n_graphs=bt.n_graphs)
        swapped = dict(bt.p3)
        swapped["p3_occ"] = bt.p3["p3_occ"][:, [2, 3, 0, 1]]
        reverse = model.compose(evidence, bond, swapped, n_graphs=bt.n_graphs)
    assert float(forward["c_P"].abs().max()) > 0.0
    assert float((forward["c_P"] - reverse["c_P"]).abs().max()) < 1e-6


def _bond(model, batch):
    trunk = model.trunk
    a = trunk.atom_encoder(batch["atom_feat"])
    idx = batch["bond_index"]
    au, av = a[idx[:, 0]], a[idx[:, 1]]
    return trunk.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1))


def test_paired_control_identity(bundle):
    joint, _ = T.build_model("lipo", bundle, "JOINT", "E2E", seed=0)
    marginal, _ = T.build_model("lipo", bundle, "MARGINAL", "E2E", seed=0)
    report = M.assert_equal_branch_counts(joint, marginal)
    assert report["equal"]
    assert report["joint_shapes"] == report["marginal_shapes"]
    assert M.branch_state_hash(joint) == M.branch_state_hash(marginal)


def test_event_additivity(bundle, paths3):
    model, _ = T.build_model("lipo", bundle, "MARGINAL", "E2E", seed=0)
    gen = torch.Generator().manual_seed(777)
    with torch.no_grad():
        model.composer.readout.weight.add_(torch.empty_like(model.composer.readout.weight).normal_(
            0.0, 0.05, generator=gen))
    model.eval()
    bt = D.make_batch(bundle, "valid", np.arange(0, 24), paths3["valid"])
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
    delta = parts["delta_comp"].numpy().astype(np.float64)
    c = parts["c_P"].numpy().astype(np.float64)
    g = bt.p3["p3_graph"].numpy().astype(np.int64)
    summed = np.bincount(g, weights=c, minlength=bt.n_graphs) if c.size else np.zeros(bt.n_graphs)
    assert float(np.max(np.abs(summed - delta))) < 1e-5
