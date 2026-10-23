"""Fast CPU tests for zinc-support-transfer-probe-v1 (no data, no checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import zst_head as H  # noqa: E402


def _fake_features(n=6, u=3):
    f = object.__new__(H.Features)
    rng = np.random.default_rng(0)
    f.ctx = rng.normal(size=(n, 64))
    f.aux = np.abs(rng.normal(size=(n, 2))) + 1.0
    f.atoms = rng.normal(size=(n, u, 110))
    f.mask = np.ones((n, u))
    f.gate = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    f.rows = np.arange(n)
    f.n_unseen = f.mask.sum(1)
    return H.attach_tensors(f)


def test_zero_init_head_reproduces_base_exactly():
    f = _fake_features()
    for model in (H.HeadG(), H.HeadS()):
        h = H.predict_head(model, f)
        assert np.all(np.abs(h) == 0.0)


def test_hard_gate_makes_ungated_molecules_identical():
    f = _fake_features()
    model = H.HeadS()
    model.readout[-1].bias.data.fill_(0.25)
    h = H.predict_head(model, f)
    assert np.all(h[f.gate == 0] == 0.0)
    assert np.all(np.abs(h[f.gate == 1] - 0.25) < 1e-6)


def test_padding_does_not_enter_pooling():
    f = _fake_features()
    model = H.HeadS()
    model.readout[-1].bias.data.fill_(0.0)
    model.readout[-1].weight.data.normal_(0, 0.1)
    h1 = H.predict_head(model, f)
    f2 = object.__new__(H.Features)
    f2.__dict__.update(f.__dict__)
    # append an extra all-padding slot filled with a huge value
    f2.atoms = np.concatenate([f.atoms, np.full((f.atoms.shape[0], 1, f.atoms.shape[2]), 1e6)], axis=1)
    f2.mask = np.concatenate([f.mask, np.zeros((f.mask.shape[0], 1))], axis=1)
    H.attach_tensors(f2)
    assert np.all(np.abs(H.predict_head(model, f2) - h1) < 1e-5)


def test_standardiser_does_not_amplify_constant_columns():
    f = _fake_features()
    f.atoms[:, :, 5] = 7.0
    std = H.Standardiser(f)
    assert std.at_sd[5] == 1.0


def test_lr_schedule_endpoints():
    assert H._lr(0, 1500) == pytest.approx(H.LR)
    assert H._lr(1500, 1500) == pytest.approx(H.MIN_LR, abs=1e-12)


def test_metrics_trimmed_and_sign():
    y = np.array([0.0, 0.0, 0.0, 10.0])
    base = np.array([0.1, -0.1, 0.2, 0.0])
    corr = np.array([0.05, -0.05, 0.1, 0.0])
    m = H.metrics(y, base, corr)
    assert m["mae_corrected"] < m["mae_base"]
    assert m["trimmed_delta"] < 0
    assert 0.0 <= m["frac_improved"] <= 1.0


def test_key_clusters_partition():
    tkeys = np.array([11, 22])
    mol2 = [np.array([11, 99]), np.array([22]), np.array([99]), np.array([11, 22])]
    rows = np.array([0, 1, 2, 3])
    cl = H.key_cluster_ids(rows, tkeys, mol2)
    assert cl[0] == 0 and cl[1] == 1 and cl[2] == -1 and cl[3] == 0


# ---------------------------------------------------------------------------
# A3 — support features must live in the right key space
# ---------------------------------------------------------------------------


def _toy_keys(seed=0):
    """Tiny synthetic key tables: two independent id spaces, k1 has several k2."""
    import zst_common as Z

    rng = np.random.default_rng(seed)
    n_mol, n_key1, n_key2 = 40, 7, 13
    parent = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 6])[:n_key2]
    mol1, mol2 = [], []
    for _ in range(n_mol):
        k1 = np.unique(rng.integers(0, n_key1, size=rng.integers(1, 4)))
        kids = np.unique(np.concatenate([np.flatnonzero(parent == k) for k in k1]))
        mol1.append(k1)
        mol2.append(kids)
    atom_k1, atom_k2 = [], []
    for k1, k2s in zip(mol1, mol2):
        atom_k1.extend([int(k1[0])] * max(1, len(k2s)))
        atom_k2.extend([int(k2s[0])] * max(1, len(k2s)))
    keys = {"n_key1": np.int64(n_key1), "n_key2": np.int64(n_key2),
            "atom_key1": np.asarray(atom_k1, dtype=np.int32),
            "atom_key2": np.asarray(atom_k2, dtype=np.int32),
            "mol_k1": np.concatenate(mol1).astype(np.int32),
            "mol_k2": np.concatenate(mol2).astype(np.int32),
            "mol_k1_off": np.concatenate([[0], np.cumsum([len(a) for a in mol1])]),
            "mol_k2_off": np.concatenate([[0], np.cumsum([len(a) for a in mol2])])}
    return Z, keys, mol1, mol2


def test_k1_support_counts_molecules_not_child_sum():
    Z, keys, mol1, mol2 = _toy_keys()
    rows = np.arange(len(mol1))
    sv = Z.per_atom_support(rows, mol1, mol2, keys, int(keys["n_key1"]), int(keys["n_key2"]))
    direct = np.zeros(int(keys["n_key1"]), dtype=int)
    for m in rows.tolist():
        direct[mol1[m]] += 1
    assert np.array_equal(sv["sup1"], direct)
    # the k1 count must not equal the sum of its children's k2 counts
    child_sum = np.zeros(int(keys["n_key1"]), dtype=int)
    parent = np.zeros(int(keys["n_key2"]), dtype=int)
    parent[keys["atom_key2"]] = keys["atom_key1"]
    for k2 in range(int(keys["n_key2"])):
        child_sum[parent[k2]] += sv["sup2"][k2]
    assert not np.array_equal(child_sum, sv["sup1"])


def test_support_features_are_invariant_to_key_renumbering():
    Z, keys, mol1, mol2 = _toy_keys(1)
    rows = np.arange(len(mol1))
    n1, n2 = int(keys["n_key1"]), int(keys["n_key2"])
    sv = Z.per_atom_support(rows, mol1, mol2, keys, n1, n2)
    rng = np.random.default_rng(3)
    p1, p2 = rng.permutation(n1), rng.permutation(n2)
    perm = {"mol_k1": p1, "atom_key1": p1, "mol_k2": p2, "atom_key2": p2}
    kp = dict(keys)
    for name, p in perm.items():
        arr = np.asarray(keys[name], dtype=np.int64)
        kp[name] = p[arr]
    m1p, m2p = Z.molecule_key_sets(kp, n=len(mol1))
    svp = Z.per_atom_support(rows, m1p, m2p, kp, n1, n2)
    assert np.array_equal(sv["atom_n1"], svp["atom_n1"])
    assert np.array_equal(sv["atom_n2"], svp["atom_n2"])
    # and the pre-fix expression (k1 id used to index the k2 table) is not invariant
    atom_k1 = np.asarray(keys["atom_key1"], dtype=np.int64)
    atom_k1p = np.asarray(kp["atom_key1"], dtype=np.int64)
    assert not np.array_equal(sv["sup2"][atom_k1], svp["sup2"][atom_k1p])
