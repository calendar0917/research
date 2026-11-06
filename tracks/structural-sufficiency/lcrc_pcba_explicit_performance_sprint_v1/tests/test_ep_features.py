"""Pure-function tests for the A3 branch signature and the A2 descriptor cache."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

import ep_common as C
import ep_features as F


# ---------------------------------------------------------------------------
# branch signature algebra
# ---------------------------------------------------------------------------


def test_branch_key_matches_lexicographic_tuple_order():
    tuples = [(bt, at, bool(e)) for bt in range(1, 5) for at in range(16) for e in (0, 1)]
    keys = [F.branch_key(*t) for t in tuples]
    assert keys == sorted(keys), "integer key order must equal tuple order"
    assert len(set(keys)) == len(keys)
    for t, k in zip(tuples, keys):
        assert F.decode_branch_key(k) == t


def test_signature_is_deterministic_and_uses_blake2b():
    keys = [F.branch_key(1, 2, False), F.branch_key(4, 15, True)]
    payload = F.signature_bytes(3, 2, 7, keys)
    expected = hashlib.blake2b(payload, digest_size=8).digest()
    digest, bucket = F.signature_digest(3, 2, 7, keys)
    assert digest == int.from_bytes(expected, "little")
    assert 0 <= bucket < C.N_BRANCH_BUCKETS
    assert F.signature_digest(3, 2, 7, keys) == (digest, bucket)
    # Header and branch list are both injective.
    assert F.signature_bytes(3, 2, 7, keys) != F.signature_bytes(3, 2, 7, list(reversed(keys)))
    assert F.signature_bytes(3, 2, 7, keys) != F.signature_bytes(3, 2, 8, keys)


def test_chain_reverse_edge_excludes_the_root():
    # 0 - 1 - 2 chain, atom types 5,6,7, all single bonds.
    atom_types = [5, 6, 7]
    src = [0, 1, 1, 2]
    dst = [1, 0, 2, 1]
    bts = [1, 1, 1, 1]
    out = F.molecule_branch_tokens(atom_types, src, dst, bts)
    # token(0->1): intermediate 1 has neighbours {0,2}, excluding 0 leaves 2.
    keys_0_1 = [F.branch_key(1, 7, False)]
    # token(1->0): intermediate 0 has neighbours {1}, excluding 1 leaves nothing.
    keys_1_0: list[int] = []
    d01 = F.signature_digest(5, 1, 6, keys_0_1)[0]
    d10 = F.signature_digest(6, 1, 5, keys_1_0)[0]
    # token(1->2): intermediate 2 has only neighbour 1, so no second-hop branch.
    d12 = F.signature_digest(6, 1, 7, [])[0]
    # token(2->1): intermediate 1 has neighbours {0,2}, excluding 2 leaves 0.
    d21 = F.signature_digest(7, 1, 6, [F.branch_key(1, 5, False)])[0]
    assert out["digests"] == [d01, d10, d12, d21]
    assert out["per_atom"][0] == [d01 % C.N_BRANCH_BUCKETS]
    assert out["per_atom"][1] == [d10 % C.N_BRANCH_BUCKETS, d12 % C.N_BRANCH_BUCKETS]
    assert out["per_atom"][2] == [d21 % C.N_BRANCH_BUCKETS]
    assert out["empty_branch_lists"] == 2


def test_branch_token_groups_second_hop_neighbours_by_intermediate_atom():
    """A repeated S2 sum over both intermediates would collide; the token must not."""
    # v=0 connects to u1=1 and u2=2.  u1 has one DOUBLE C branch (3);
    # u2 has one DOUBLE C branch (4).  Same branch multiset overall, different
    # intermediate environments.
    atom_types = [0, 1, 6, 6, 6]
    src = [0, 1, 1, 3, 1, 2, 2, 4, 2]
    dst = [1, 0, 3, 1, 2, 0, 4, 2, 1]
    # bond(v,u1)=SINGLE, bond(v,u2)=SINGLE, bond(u1,3)=DOUBLE, bond(u2,4)=TRIPLE
    bts = [1, 1, 2, 2, 1, 1, 3, 3, 1]
    # remove the spurious (1,2) bridge: keep v->u1, u1->v, u1->3, 3->u1, v->u2, u2->v, u2->4, 4->u2
    src = [0, 1, 1, 3, 0, 2, 2, 4]
    dst = [1, 0, 3, 1, 2, 0, 4, 2]
    bts = [1, 1, 2, 2, 1, 1, 3, 3]
    out = F.molecule_branch_tokens(atom_types, src, dst, bts)
    d01 = F.signature_digest(0, 1, 1, [F.branch_key(2, 6, False)])[0]
    d02 = F.signature_digest(0, 1, 6, [F.branch_key(3, 6, False)])[0]
    assert out["digests"][0] == d01
    assert out["digests"][4] == d02
    assert d01 != d02
    # The naive union of all second-hop branches from v would be the same list
    # in both tokens; the real token keeps each branch grouped under its own
    # intermediate atom, so the two digests differ.
    naive_union = sorted([F.branch_key(2, 6, False), F.branch_key(3, 6, False)])
    naive_digest = F.signature_digest(0, 1, 1, naive_union)[0]
    assert naive_digest not in (d01, d02)


def test_is_edge_triangle_flag_and_branch_sort_independence():
    # triangle 0-1-2-0 plus a leaf 3 on 1; check is_edge via a v-w triangle.
    atom_types = [1, 1, 1, 7]
    src = [0, 1, 1, 2, 2, 0, 1, 3]
    dst = [1, 0, 2, 1, 0, 2, 3, 1]
    bts = [1, 1, 1, 1, 1, 1, 1, 1]
    out = F.molecule_branch_tokens(atom_types, src, dst, bts)
    # token(0->1): intermediate 1 neighbours {0,2,3}, excluding 0 -> w=2 (v-w edge
    # exists: 0-2) and w=3 (no 0-3 edge).
    keys = sorted([F.branch_key(1, 1, True), F.branch_key(1, 7, False)])
    assert out["digests"][0] == F.signature_digest(1, 1, 1, keys)[0]
    # Reversing the branch enumeration order is irrelevant: signatures use a
    # canonical sorted list.
    src2 = [0, 1, 1, 2, 2, 0, 1, 3]
    dst2 = [1, 0, 2, 1, 0, 2, 3, 1]
    perm = list(reversed(range(len(src2))))  # reverse the bond-record order
    out2 = F.molecule_branch_tokens(atom_types, [src2[i] for i in perm], [dst2[i] for i in perm], [bts[i] for i in perm])
    assert sorted(out2["digests"]) == sorted(out["digests"])


def test_atom_renumbering_preserves_tokens_up_to_the_mapping():
    atom_types = [2, 2, 7, 8, 2, 6]
    # a small branched graph
    edges = [(0, 1, 1), (1, 2, 1), (1, 3, 2), (0, 4, 1), (4, 5, 3)]
    src, dst, bts = [], [], []
    for a, b, t in edges:
        src += [a, b]
        dst += [b, a]
        bts += [t, t]
    base = F.molecule_branch_tokens(atom_types, src, dst, bts)
    perm = [4, 5, 0, 1, 2, 3]  # new index -> old index
    inv = [0] * len(perm)
    for new, old in enumerate(perm):
        inv[old] = new
    atom_types2 = [atom_types[old] for old in perm]
    src2 = [inv[a] for a in src]
    dst2 = [inv[b] for b in dst]
    other = F.molecule_branch_tokens(atom_types2, src2, dst2, bts)
    for old in range(len(atom_types)):
        assert sorted(base["per_atom"][old]) == sorted(other["per_atom"][inv[old]])
    assert sorted(base["digests"]) == sorted(other["digests"])


def test_zero_degree_atom_has_no_tokens():
    out = F.molecule_branch_tokens([3, 4, 5], [], [], [])
    assert out["tokens"] == 0
    assert out["zero_degree_atoms"] == [0, 1, 2]
    assert all(per == [] for per in out["per_atom"])


# ---------------------------------------------------------------------------
# RDKit descriptors
# ---------------------------------------------------------------------------


def test_rdkit_descriptors_frozen_order_and_values():
    pytest.importorskip("rdkit")
    values, err = F._rdkit_descriptors("CCO")
    assert err == ""
    assert len(values) == C.N_PHYSCHEM
    tpsa, hbd, hba, logmw, logp, rot, charge, arom, maxring, fcsp3 = values
    assert tpsa == pytest.approx(20.23, abs=0.02)
    assert hbd == 1 and hba == 1
    assert logmw == pytest.approx(math.log1p(46.069), abs=0.02)
    assert logp == pytest.approx(-0.0014, abs=0.01)
    assert rot == 0
    assert charge == 0
    assert arom == 0
    assert maxring == 0
    assert fcsp3 == pytest.approx(1.0)


def test_rdkit_ring_and_charge_descriptors():
    pytest.importorskip("rdkit")
    values, err = F._rdkit_descriptors("c1ccccc1")
    assert err == ""
    assert values[C.PHYSCHEM_NAMES.index("NumAromaticRings")] == 1
    assert values[C.PHYSCHEM_NAMES.index("MaxRingAtomCount")] == 6
    values, err = F._rdkit_descriptors("CC(=O)[O-]")
    assert err == ""
    assert values[C.PHYSCHEM_NAMES.index("TotalFormalCharge")] == -1
    values, err = F._rdkit_descriptors("not-a-smiles(")
    assert values == [] and err


def test_physchem_loader_train_only_norm_and_missing_flag(tmp_path, monkeypatch):
    """Synthetic full-size cache: finite Train/Valid, NaN Test, one failed row."""
    n = C.N_MOLECULES
    rng = np.random.default_rng(0)
    raw = np.full((n, C.N_PHYSCHEM), np.nan, dtype=np.float32)
    built = np.zeros(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)
    train = np.arange(0, 100, dtype=np.int64)
    valid = np.arange(100, 140, dtype=np.int64)
    test = np.arange(140, 180, dtype=np.int64)
    rows = np.sort(np.concatenate([train, valid]))
    raw[rows] = rng.normal(size=(len(rows), C.N_PHYSCHEM)).astype(np.float32)
    built[rows] = True
    failed[rows[3]] = True
    raw[rows[3]] = np.nan
    ok = train[built[train] & ~failed[train]]
    mean = raw[ok].astype(np.float64).mean(0)
    scale = raw[ok].astype(np.float64).std(0)
    scale[scale == 0] = 1.0
    cache = tmp_path / "physchem.npz"
    np.savez(cache, raw=raw, built=built, failed=failed, attempted_rows=rows,
             train_mean=mean, train_scale=scale)
    meta = {
        "schema": "ep-physchem-cache-v1",
        "algorithm": C.PHYSCHEM_ALGO,
        "train_only_norm": {"fingerprint": "x", "mean": mean.tolist(), "scale": scale.tolist()},
        "files": {"cache_sha256": C.sha256_file(cache)},
        "source": {"smiles_sha256": C.sha256_file(F.smiles_path())},
    }
    meta_path = tmp_path / "physchem.json"
    C.write_json(meta_path, meta)

    class T:
        pass

    task = T()
    task.train_rows, task.valid_rows, task.test_rows = train, valid, test
    loaded = F.load_physchem(task, cache=cache, meta_path=meta_path)
    m = loaded["matrix"]
    assert np.isfinite(m[train]).all() and np.isfinite(m[valid]).all()
    assert np.isnan(m[test]).all()
    fail_row = int(rows[3])
    assert loaded["failed"][fail_row]
    assert m[fail_row, C.N_PHYSCHEM] == 1.0
    assert np.allclose(m[fail_row, :C.N_PHYSCHEM], 0.0, atol=1e-6), "missing row uses the Train mean"
    # Non-missing rows standardise to mean 0 over Train.
    assert np.allclose(m[train][~loaded["failed"][train], :C.N_PHYSCHEM].mean(0), 0.0, atol=1e-5)
    # Train-only: perturbing Valid/Test raw values must not change the loader.
    raw2 = raw.copy()
    raw2[valid] += 5.0
    np.savez(cache, raw=raw2, built=built, failed=failed, attempted_rows=rows,
             train_mean=mean, train_scale=scale)
    meta["files"]["cache_sha256"] = C.sha256_file(cache)
    C.write_json(meta_path, meta)
    loaded2 = F.load_physchem(task, cache=cache, meta_path=meta_path)
    assert np.array_equal(loaded["matrix"][train], loaded2["matrix"][train])
    # A finite Test row is a hard refusal.
    raw3 = raw.copy()
    raw3[test[0]] = 1.0
    built3 = built.copy()
    built3[test[0]] = True
    np.savez(cache, raw=raw3, built=built3, failed=failed, attempted_rows=rows,
             train_mean=mean, train_scale=scale)
    meta["files"]["cache_sha256"] = C.sha256_file(cache)
    C.write_json(meta_path, meta)
    with pytest.raises(RuntimeError):
        F.load_physchem(task, cache=cache, meta_path=meta_path)
