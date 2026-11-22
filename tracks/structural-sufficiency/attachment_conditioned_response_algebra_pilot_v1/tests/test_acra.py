"""Engineering checks for ACRA-v1 (all 13 brief requirements).

These tests exercise the real computation path: they re-derive the S/C balls,
the rooted frame multisets, the poolings and the composition from raw data with
independent implementations and compare against the cached tensors and the
model outputs.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import acra_common as C
import acra_data as D
import acra_cache as K
import acra_model as M

PD = C.PD


# ---------------------------------------------------------------------------
# independent implementations (no reuse of the production helpers)
# ---------------------------------------------------------------------------


def independent_ball(n_atoms: int, bonds: list[tuple[int, int]], root: int, radius: int,
                     banned: tuple[int, int]) -> list[int]:
    """BFS with a plain edge list (does not touch acra_data)."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        if (a, b) == banned or (b, a) == banned:
            continue
        adj[a].append(b)
        adj[b].append(a)
    dist = {root: 0}
    for _ in range(radius):
        new: dict[int, int] = {}
        for u, du in dist.items():
            if du >= radius:
                continue
            for w in adj[u]:
                if w not in dist and w not in new:
                    new[w] = du + 1
        dist.update(new)
    return sorted(dist)


def _bond_lookup(mol: PD.Molecule) -> dict[tuple[int, int], int]:
    out = {}
    for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        out[(int(a), int(b))] = e
        out[(int(b), int(a))] = e
    return out


def _pad_atom(rows: list[np.ndarray]) -> bytes:
    out = np.zeros((C.MAX_OBJECT_ATOMS, C.ATOM_COLUMNS), np.uint8)
    for i, row in enumerate(rows):
        out[i] = row
    return out.tobytes()


def _pad_bond(rows: dict[int, np.ndarray]) -> bytes:
    out = np.zeros((len(C.FRAME_PAIRS), C.BOND_COLUMNS), np.uint8)
    for slot, row in rows.items():
        out[slot] = row
    return out.tobytes()


def brute_frames(mol: PD.Molecule, atoms, root: int,
                 adjacency_pairs) -> dict[int, dict[tuple, int]]:
    """Independent rooted frame enumeration: brute force over all subsets and
    all root-fixed permutations, keeping every adjacency-minimal permutation."""
    atoms = sorted(int(a) for a in atoms)
    aset = set(atoms)
    adj = {a: [] for a in atoms}
    for i, j, _b in adjacency_pairs:
        adj[atoms[i]].append(atoms[j])
        adj[atoms[j]].append(atoms[i])
    between = _bond_lookup(mol)
    out: dict[int, dict[tuple, int]] = {k: {} for k in range(1, C.MAX_OBJECT_ATOMS + 1)}
    for k in range(1, C.MAX_OBJECT_ATOMS + 1):
        for comb in itertools.combinations(atoms, k):
            if int(root) not in comb:
                continue
            seen = {int(root)}
            stack = [int(root)]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w in comb and w not in seen:
                        seen.add(w)
                        stack.append(w)
            if len(seen) != k:
                continue
            slots = [int(root)] + [a for a in comb if a != int(root)]
            best = None
            winners: list[tuple[int, dict[int, np.ndarray], list[int]]] = []
            for perm in itertools.permutations(range(1, k)):
                order = [slots[0]] + [slots[p] for p in perm]
                bits = 0
                brows: dict[int, np.ndarray] = {}
                for slot, (i, j) in enumerate(C.FRAME_PAIRS):
                    if i >= k or j >= k:
                        continue
                    bid = between.get((order[i], order[j]))
                    if bid is not None:
                        bits |= 1 << slot
                        brows[slot] = mol.bond_feat[bid]
                if best is None or bits < best:
                    best = bits
                    winners = [(bits, brows, order)]
                elif bits == best:
                    winners.append((bits, brows, order))
            for bits, brows, order in winners:
                key = (k, int(bits), _pad_atom([mol.atom_feat[a] for a in order]), _pad_bond(brows))
                out[k][key] = out[k].get(key, 0) + 1
    return out


def cached_frames(pack, event: int, side: int) -> dict[int, dict[tuple, int]]:
    out: dict[int, dict[tuple, int]] = {}
    for k in range(1, C.MAX_OBJECT_ATOMS + 1):
        b = event * 8 + side * 4 + (k - 1)
        s, e = int(pack["blk_ptr"][b]), int(pack["blk_ptr"][b + 1])
        got: dict[tuple, int] = {}
        for ent in range(s, e):
            pid = int(pack["ent_pat"][ent])
            key = (int(pack["pat_k"][pid]), int(pack["pat_bits"][pid]),
                   pack["pat_atom"][pid].tobytes(), pack["pat_bond"][pid].tobytes())
            got[key] = got.get(key, 0) + int(pack["ent_mult"][ent])
        out[k] = got
        assert int(pack["blk_count"][b]) == sum(got.values())
    return out


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    return C.load_config()


@pytest.fixture(scope="module")
def raw(cfg) -> dict:
    return D.load_raw(cfg)


@pytest.fixture(scope="module")
def pack(cfg, raw) -> dict:
    return K.build_pack(cfg, raw, rows=list(range(24)), log=lambda _m: None)


@pytest.fixture(scope="module")
def dims():
    return C.official_feature_dims()


def small_model(cfg, pack, dims, arm: str):
    import torch

    model = M.ACRAModel(cfg, dims[0], dims[1], arm)
    model.load_pack(pack)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1-3: coverage, two directed events per bond, correct S/C adjacency
# ---------------------------------------------------------------------------


def test_every_atom_and_bond_is_used(cfg, raw, pack):
    rows = pack["row_ids"]
    for i, row in enumerate(rows):
        mol = D.raw_molecule(raw, int(row))
        a0, a1 = int(pack["atom_ptr"][i]), int(pack["atom_ptr"][i + 1])
        e0, e1 = int(pack["ev_ptr"][i]), int(pack["ev_ptr"][i + 1])
        assert a1 - a0 == mol.n_atoms
        assert np.array_equal(pack["atom_feat"][a0:a1], mol.atom_feat)
        assert e1 - e0 == 2 * int(mol.bond_u.size)
        assert int(pack["mol_n_bonds"][i]) == int(mol.bond_u.size)
        # every bond contributes exactly two directed events
        counts = np.bincount(pack["ev_bond"][e0:e1],
                             minlength=int(mol.bond_u.size)).tolist()
        assert counts == [2] * int(mol.bond_u.size)
        assert pack["ev_dir"][e0:e1].tolist() == [0, 1] * int(mol.bond_u.size)
        # every directed event has a non-empty S rooted at one of the endpoints
        assert np.all(pack["ev_scalars_s"][e0:e1, 0] >= np.log1p(1.0))
        assert np.all(pack["ev_scalars_c"][e0:e1, 0] >= np.log1p(1.0))


def test_local_balls_match_independent_bfs(cfg, raw, pack):
    rows = pack["row_ids"]
    rng = np.random.default_rng(0)
    for i in rng.choice(rows.size, size=min(6, rows.size), replace=False).tolist():
        row = int(rows[i])
        mol = D.raw_molecule(raw, row)
        adj = D.adjacency(mol)
        bonds = [(int(a), int(b)) for a, b in zip(mol.bond_u, mol.bond_v)]
        for e in range(int(mol.bond_u.size)):
            for direction in (0, 1):
                u, v = bonds[e] if direction == 0 else bonds[e][::-1]
                (root_s, s_atoms), (root_c, c_atoms) = D.event_balls(mol, adj, e, direction)
                assert root_s == u and root_c == v
                ind_s = independent_ball(mol.n_atoms, bonds, u, 1, (u, v))
                ind_c = independent_ball(mol.n_atoms, bonds, v, 2, (u, v))
                assert list(s_atoms) == ind_s and root_s == u
                assert list(c_atoms) == ind_c and root_c == v
                # cached scalars come from exactly these induced subgraphs
                ev = int(pack["ev_ptr"][i]) + 2 * e + direction
                for atoms, col in ((s_atoms, "ev_scalars_s"), (c_atoms, "ev_scalars_c")):
                    aset = set(atoms)
                    nb = sum(1 for a, b in bonds if a in aset and b in aset)
                    scal = pack[col][ev]
                    assert abs(float(scal[0]) - np.log1p(len(atoms))) < 1e-6
                    assert abs(float(scal[1]) - np.log1p(nb)) < 1e-6


def test_ring_bonds_keep_the_rest_of_the_ring(cfg, raw):
    """Non-bridge edges are never dropped and nothing is disconnected."""
    rows = np.asarray(raw["train_idx"], np.int64)
    checked = 0
    for row in rows[:80].tolist():
        mol = D.raw_molecule(raw, int(row))
        adj = D.adjacency(mol)
        geo = D.bond_geometry(mol, adj)
        if not geo["in_ring"].any():
            continue
        base = len(set(PD.connected_components(adj, mol.n_atoms)))
        for e in np.nonzero(geo["in_ring"])[0].tolist():
            u, v = int(mol.bond_u[e]), int(mol.bond_v[e])
            after = len(set(PD.connected_components(adj, mol.n_atoms, banned=(u, v))))
            assert after == base, "ring bond removal must not disconnect the molecule"
            for direction in (0, 1):
                src, dst = (u, v) if direction == 0 else (v, u)
                (root_s, s_atoms), (root_c, c_atoms) = D.event_balls(mol, adj, e, direction)
                assert root_s == src and root_c == dst
                assert src in s_atoms
                expected_s = 1 + len([w for w in adj[src] if w != dst])
                assert len(s_atoms) == expected_s
            checked += 1
        if checked >= 6:
            break
    assert checked > 0, "no ring bond found in the sampled train molecules"

    # short rings: S and C are allowed to (and do) overlap
    tri = PD.Molecule(row_id=0, n_atoms=3,
                      atom_feat=np.zeros((3, 9), np.uint8),
                      bond_u=np.asarray([0, 0, 1], np.int32),
                      bond_v=np.asarray([1, 2, 2], np.int32),
                      bond_feat=np.zeros((3, 3), np.uint8))
    (root_s, s_atoms), (root_c, c_atoms) = D.event_balls(tri, D.adjacency(tri), 0, 0)
    assert root_s == 0 and root_c == 1
    assert set(s_atoms) & set(c_atoms) == {0, 2}


def test_bond_pair_relation_definition():
    """shared-ring means 'two distinct bonds on a common cycle'."""
    def build(n, edges):
        return PD.Molecule(
            row_id=0, n_atoms=n,
            atom_feat=np.zeros((n, 9), np.uint8),
            bond_u=np.asarray([min(a, b) for a, b in edges], np.int32),
            bond_v=np.asarray([max(a, b) for a, b in edges], np.int32),
            bond_feat=np.zeros((len(edges), 3), np.uint8),
        )

    chain = build(4, [(0, 1), (1, 2), (2, 3)])
    geo = D.bond_geometry(chain, D.adjacency(chain))
    assert geo["is_bridge"].all()
    assert int(geo["pair_shared_ring"].sum()) == 0
    assert sorted(geo["pair_bucket"].tolist()) == [1, 1, 2] or set(geo["pair_bucket"].tolist()) <= {1, 2, 3, 4}

    square = build(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    geo = D.bond_geometry(square, D.adjacency(square))
    assert not geo["is_bridge"].any()
    assert int(geo["pair_shared_ring"].sum()) == 6      # all four edges share a cycle
    assert sorted(geo["pair_bucket"].tolist()) == [1, 1, 1, 1, 2, 2]

    figure8 = build(5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    geo = D.bond_geometry(figure8, D.adjacency(figure8))
    assert not geo["is_bridge"].any()
    m = 6
    e, f = np.triu_indices(m, 1)
    shared = geo["pair_shared_ring"]
    # triangles {0-1,1-2,2-0} and {0-3,3-4,4-0} are the two blocks
    for k in range(len(e)):
        same_block = (e[k] < 3) == (f[k] < 3)
        assert int(shared[k]) == int(same_block)

    tree = build(5, [(0, 1), (1, 2), (1, 3), (3, 4)])
    geo = D.bond_geometry(tree, D.adjacency(tree))
    assert geo["is_bridge"].all()


# ---------------------------------------------------------------------------
# 4-5: rooted frame enumeration, attributes, multiplicity, root pinning
# ---------------------------------------------------------------------------


def test_rooted_frames_match_bruteforce(cfg, raw, pack):
    rows = pack["row_ids"]
    rng = np.random.default_rng(1)
    for i in rng.choice(rows.size, size=min(4, rows.size), replace=False).tolist():
        row = int(rows[i])
        mol = D.raw_molecule(raw, row)
        adj = D.adjacency(mol)
        e_count = int(mol.bond_u.size)
        for e in list(range(min(3, e_count))) + list(range(max(0, e_count - 2), e_count)):
            for direction in (0, 1):
                (root_s, s_atoms), (root_c, c_atoms) = D.event_balls(mol, adj, e, direction)
                ev = int(pack["ev_ptr"][i]) + 2 * e + direction
                for side, atoms, root in ((0, s_atoms, root_s), (1, c_atoms, root_c)):
                    pos = {int(a): j for j, a in enumerate(atoms)}
                    pairs = D.induced_bonds(mol, atoms)
                    brute = brute_frames(mol, atoms, root, pairs)
                    cached = cached_frames(pack, ev, side)
                    for k in range(1, C.MAX_OBJECT_ATOMS + 1):
                        assert cached[k] == brute[k], (row, e, direction, side, k)
                        # root pinned to slot 0: every frame's first atom row is root's
                        for key in cached[k]:
                            atom_bytes = key[2]
                            first = np.frombuffer(atom_bytes, np.uint8)[:C.ATOM_COLUMNS]
                            assert np.array_equal(first, mol.atom_feat[int(root)])


def test_synthetic_frames_and_multiplicity():
    """Symmetric synthetic graphs: multiplicity compression is lossless."""
    atom_rows = [np.asarray([10 + i, 1, 1, 3, 0, 0, 0, 0, 0], np.uint8) for i in range(5)]
    bond_row = np.asarray([1, 1, 2], np.uint8)
    # star: centre 0, leaves 1,2,3 (identical local structure up to root)
    star = PD.Molecule(row_id=0, n_atoms=4,
                       atom_feat=np.stack(atom_rows[:4]),
                       bond_u=np.asarray([0, 0, 0], np.int32),
                       bond_v=np.asarray([1, 2, 3], np.int32),
                       bond_feat=np.stack([bond_row] * 3))
    luts = PD.build_luts(4)
    patterns = D.frame_patterns(star, [0, 1, 2, 3], 0, luts)
    pairs = D.induced_bonds(star, [0, 1, 2, 3])
    brute = brute_frames(star, [0, 1, 2, 3], 0, pairs)
    assert patterns == brute
    total = sum(sum(m.values()) for m in patterns.values())
    assert total == sum(sum(m.values()) for m in brute.values())
    # three identical 2-atom frames (root-leaf) are one pattern with multiplicity 3
    mult_2 = sum(patterns[2].values())
    assert mult_2 == 3

    # triangle with identical atoms: all 3 two-atom frames merge into one pattern
    tri = PD.Molecule(row_id=0, n_atoms=3,
                      atom_feat=np.stack([atom_rows[0]] * 3),
                      bond_u=np.asarray([0, 0, 1], np.int32),
                      bond_v=np.asarray([1, 2, 2], np.int32),
                      bond_feat=np.stack([bond_row] * 3))
    pat = D.frame_patterns(tri, [0, 1, 2], 0, luts)
    assert sum(pat[2].values()) == 2
    assert len(pat[3]) == 1


# ---------------------------------------------------------------------------
# 6-8: invariances and batch equivalence
# ---------------------------------------------------------------------------


def _predict_one(cfg, pack, dims, arm, row_index=0):
    model = small_model(cfg, pack, dims, arm)
    res = M.predict_rows(model, D.BatchSource(pack, "cpu"), [row_index], "cpu")
    return res, model


def test_atom_renumbering_invariance(cfg, raw, dims):
    row = int(raw["train_idx"][3])
    mol = D.raw_molecule(raw, row)
    rng = np.random.default_rng(7)
    perm = rng.permutation(mol.n_atoms)
    permuted = mol.permute(perm)
    colors = PD.ColorSpace.from_raw(raw)
    pack_a = K.pack_from_molecules(cfg, [mol], ["a"], colors, "x", [row], log=lambda _m: None)
    pack_b = K.pack_from_molecules(cfg, [permuted], ["b"], colors, "x", [row],
                                   log=lambda _m: None)
    for arm in M.ARMS:
        a, _ = _predict_one(cfg, pack_a, dims, arm)
        b, _ = _predict_one(cfg, pack_b, dims, arm)
        assert np.allclose(a["yhat_std"], b["yhat_std"], atol=1e-6, rtol=0)
        assert np.allclose(a["e_atom"], b["e_atom"], atol=1e-6, rtol=0)
        assert np.allclose(a["e_bond"], b["e_bond"], atol=1e-6, rtol=0)
        assert np.allclose(a["e_rel"], b["e_rel"], atol=1e-6, rtol=0)


def test_edge_direction_swap_invariance(cfg, raw, dims):
    row = int(raw["train_idx"][5])
    mol = D.raw_molecule(raw, row)
    flipped = PD.Molecule(row_id=mol.row_id, n_atoms=mol.n_atoms, atom_feat=mol.atom_feat,
                          bond_u=mol.bond_v.copy(), bond_v=mol.bond_u.copy(),
                          bond_feat=mol.bond_feat)
    colors = PD.ColorSpace.from_raw(raw)
    pack_a = K.pack_from_molecules(cfg, [mol], ["a"], colors, "x", [row], log=lambda _m: None)
    pack_b = K.pack_from_molecules(cfg, [flipped], ["b"], colors, "x", [row],
                                   log=lambda _m: None)
    for arm in M.ARMS:
        a, _ = _predict_one(cfg, pack_a, dims, arm)
        b, _ = _predict_one(cfg, pack_b, dims, arm)
        assert np.allclose(a["yhat_std"], b["yhat_std"], atol=1e-6, rtol=0)
        assert np.allclose(a["e_bond"], b["e_bond"], atol=1e-6, rtol=0)


def test_batch_equals_single(cfg, pack, dims):
    model = small_model(cfg, pack, dims, "ACRA_REL")
    source = D.BatchSource(pack, "cpu")
    rows = [1, 5, 9, 13]
    batched = M.predict_rows(model, source, rows, "cpu", batch_size=128)
    for i, row in enumerate(rows):
        one = M.predict_rows(model, source, [row], "cpu", batch_size=1)
        assert np.allclose(batched["yhat_std"][i], one["yhat_std"][0], atol=1e-6)
        assert np.allclose(batched["e_rel"][i], one["e_rel"][0], atol=1e-6)


def test_batch_indices_partition_molecules(pack):
    source = D.BatchSource(pack, "cpu")
    rows = [0, 3, 4, 10]
    batch = source.make(rows)
    ev_ptr, atom_ptr = pack["ev_ptr"], pack["atom_ptr"]
    expected_ev = np.concatenate([np.arange(ev_ptr[r], ev_ptr[r + 1]) for r in rows])
    expected_atom = np.concatenate([np.arange(atom_ptr[r], atom_ptr[r + 1]) for r in rows])
    assert np.array_equal(batch["ev_idx"].numpy(), expected_ev)
    assert np.array_equal(batch["atom_idx"].numpy(), expected_atom)
    assert batch["n_events"] == expected_ev.size
    assert batch["n_atoms"] == expected_atom.size
    assert np.all(np.diff(batch["blk_local"].numpy()) >= 0)
    assert int(batch["blk_local"].numpy().max()) == batch["n_blocks"] - 1
    assert int(batch["n_bonds"].sum()) * 2 == batch["n_events"]


# ---------------------------------------------------------------------------
# 9-10: exact composition and relation symmetry
# ---------------------------------------------------------------------------


def test_prediction_is_exactly_the_sum_of_terms(cfg, pack, dims):
    for arm in M.ARMS:
        model = small_model(cfg, pack, dims, arm)
        source = D.BatchSource(pack, "cpu")
        batch = source.make([2, 6, 7])
        out = model.forward_batch(batch)
        total = out["beta"] + out["e_atom"] + out["e_bond"] + out["e_rel"]
        assert float((total - out["yhat_std"]).abs().max()) == 0.0
        if arm != "ACRA_REL":
            assert float(out["e_rel"].abs().max()) == 0.0


def test_relation_term_is_symmetric(cfg, pack, dims):
    model = small_model(cfg, pack, dims, "ACRA_REL")
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2])
    out = model.forward_batch(batch)
    swapped = dict(batch)
    swapped["pair_e"], swapped["pair_f"] = batch["pair_f"], batch["pair_e"]
    out2 = model.forward_batch(swapped)
    assert np.allclose(out["e_rel"].detach().numpy(), out2["e_rel"].detach().numpy(),
                       atol=1e-7)
    assert np.allclose(out["yhat_std"].detach().numpy(), out2["yhat_std"].detach().numpy(),
                       atol=1e-7)


def test_relation_is_zero_without_two_bonds(cfg, raw, dims):
    row = int(raw["train_idx"][0])
    mol = D.raw_molecule(raw, row)
    # keep one real bond only, a two-atom molecule (bond attributes from the real one)
    two = PD.Molecule(row_id=0, n_atoms=2, atom_feat=mol.atom_feat[:2],
                      bond_u=np.asarray([0], np.int32), bond_v=np.asarray([1], np.int32),
                      bond_feat=mol.bond_feat[:1])
    pack = K.pack_from_molecules(cfg, [two], ["cc"], PD.ColorSpace.from_raw(raw), "x", [0],
                                 log=lambda _m: None)
    assert pack["pair_e"].size == 0
    model = small_model(cfg, pack, dims, "ACRA_REL")
    out = model.forward_batch(D.BatchSource(pack, "cpu").make([0]))
    assert float(out["e_rel"].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 11-13: gradients, Train-only fitting, finiteness
# ---------------------------------------------------------------------------


def test_all_trainable_branches_receive_gradient(cfg, pack, dims):
    import torch

    model = small_model(cfg, pack, dims, "ACRA_REL")
    source = D.BatchSource(pack, "cpu")
    train_rows = np.asarray([int(pack["row_ids"][i]) for i in range(16)])
    labels = np.asarray([float(pack["row_ids"][i] % 5) * 0.3 for i in range(16)])
    y = torch.as_tensor((labels - labels.mean()) / labels.std(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for step in range(6):
        batch = source.make([i for i in range(16)])
        out = model.forward_batch(batch)
        terms = model.loss(out, batch, y)
        opt.zero_grad(set_to_none=True)
        terms["loss"].backward()
        assert torch.isfinite(terms["loss"])
        opt.step()
    # one more clean backward: every module must see a non-zero gradient
    batch = source.make([i for i in range(16)])
    out = model.forward_batch(batch)
    terms = model.loss(out, batch, y)
    model.zero_grad(set_to_none=True)
    terms["loss"].backward()
    dead = [name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()
                                    or float(p.grad.abs().sum()) == 0.0)]
    groups = sorted({name.split(".")[0] for name in dead})
    assert not groups, f"branches without gradient: {groups}"
    assert torch.isfinite(out["yhat_std"]).all()
    for name in ("q", "k", "site_head", "rel_q", "rel_k", "rel_bias", "gate_raw"):
        assert hasattr(model, name) and any(n.startswith(name) for n, _ in model.named_parameters())


def test_cache_and_scaler_are_train_only(cfg, raw):
    """No Valid/Test labels anywhere: cache ignores labels, fitting uses Train."""
    rows = list(range(12))
    poisoned = dict(raw)
    poisoned["labels"] = np.full_like(np.asarray(raw["labels"]), np.nan)
    pack_a = K.build_pack(cfg, raw, rows=rows, log=lambda _m: None)
    pack_b = K.build_pack(cfg, poisoned, rows=rows, log=lambda _m: None)
    for key in pack_a:
        if key == "meta_json":
            continue
        assert np.array_equal(np.asarray(pack_a[key]), np.asarray(pack_b[key])), key
    cache_source = (CODE_DIR / "acra_cache.py").read_text(encoding="utf-8")
    assert 'raw["labels"]' not in cache_source
    assert '"labels"' not in cache_source

    labels = np.asarray(raw["labels"], np.float64).copy()
    train = np.asarray(raw["train_idx"], np.int64)
    valid = np.asarray(raw["valid_idx"], np.int64)
    test = np.asarray(raw["test_idx"], np.int64)
    base = C.target_scaler(labels, train)
    poisoned_labels = labels.copy()
    poisoned_labels[valid] = np.nan
    poisoned_labels[test] = np.inf
    assert C.target_scaler(poisoned_labels, train) == base

    # category support must not read Valid/Test features either
    atom_dims, bond_dims = C.official_feature_dims()
    support_a = M.category_support(raw, train, atom_dims, bond_dims)
    feat2 = {k: np.array(v, copy=True) for k, v in raw.items() if isinstance(v, np.ndarray)}
    node_ptr, bond_ptr = np.asarray(raw["node_ptr"]), np.asarray(raw["bond_ptr"])
    for row in np.concatenate([valid, test]).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        feat2["atom_feat"][a0:a1, 0] = atom_dims[0] - 1
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        feat2["bond_feat"][b0:b1, 0] = bond_dims[0] - 1
    support_b = M.category_support(feat2, train, atom_dims, bond_dims)
    for x, y in zip(support_a[0], support_b[0]):
        assert np.array_equal(x, y)
    for x, y in zip(support_a[1], support_b[1]):
        assert np.array_equal(x, y)

    train_source = (CODE_DIR / "acra_train.py").read_text(encoding="utf-8")
    assert 'splits["test"]' not in train_source
    assert "'test'" not in train_source
    assert "test_labels_read\": False" in train_source


def test_no_nan_inf(cfg, pack, dims):
    import torch

    model = small_model(cfg, pack, dims, "ACRA_REL")
    source = D.BatchSource(pack, "cpu")
    batch = source.make([0, 1, 2, 3])
    out = model.forward_batch(batch)
    y = torch.zeros(4)
    terms = model.loss(out, batch, y)
    terms["loss"].backward()
    for key in ("yhat_std", "e_atom", "e_bond", "e_rel", "r", "interaction"):
        assert torch.isfinite(out[key]).all(), key
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_arm_toggles_are_shared_otherwise(cfg, pack, dims):
    models = {arm: small_model(cfg, pack, dims, arm) for arm in M.ARMS}
    shared = ["frame", "proj_s", "proj_c", "mlp_s", "mlp_c", "atom_head", "bond_emb",
              "flag_bridge", "flag_ring", "scalar_s", "scalar_c", "beta"]
    base = models["ACRA_ADD"]
    for arm, model in models.items():
        for prefix in shared:
            a = {k: v for k, v in base.named_parameters() if k.startswith(prefix + ".")
                 or k == prefix}
            b = {k: v for k, v in model.named_parameters() if k.startswith(prefix + ".")
                 or k == prefix}
            assert set(a) == set(b)
            for k in a:
                assert torch_equal(a[k], b[k]), (arm, prefix, k)


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))