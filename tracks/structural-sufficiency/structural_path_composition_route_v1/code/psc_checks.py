"""Engineering checks for the PSC route (PROTOCOL.md section 10).

These are mechanism/engineering checks, not scientific results.  They are
designed around the real risk points: path enumeration, canonicalisation,
padding/reversal, batch-vs-single equivalence, BAG invariance, the SUM ledger,
PAIR/DIAG algebra, fit-only statistics, init sharing and gradient flow.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import psc_common as C
import psc_data as D
import psc_model as M
import psc_paths as PP

CHECKS: list[dict[str, Any]] = []


def check(name: str, fn: Callable[[], dict[str, Any]]) -> None:
    try:
        info = fn() or {}
        CHECKS.append({"name": name, "status": "PASS", "info": info})
        print(f"PASS {name} {json.dumps(C.jsonable(info), sort_keys=True)[:220]}")
    except Exception as exc:  # noqa: BLE001
        CHECKS.append({"name": name, "status": "FAIL", "error": repr(exc)})
        print(f"FAIL {name}: {exc!r}")
        raise


# ---------------------------------------------------------------------------
# 1. enumeration
# ---------------------------------------------------------------------------


def brute_force_paths(adj: Sequence[Sequence[int]], max_len: int) -> np.ndarray:
    """Independent count: full directed DFS, then ``undirected = directed / 2``.

    Every simple path with two distinct endpoints is produced exactly twice by
    the directed enumeration, so this identity does not reuse the canonical
    orientation rule of ``enumerate_paths``.
    """
    n = len(adj)
    directed = np.zeros(max_len + 1, np.int64)
    for root in range(n):
        directed[0] += 1
        stack = [(root,)]
        while stack:
            seq = stack.pop()
            if len(seq) - 1 >= max_len:
                continue
            for nxt in adj[seq[-1]]:
                if nxt in seq:
                    continue
                directed[len(seq)] += 1
                stack.append(seq + (nxt,))
    counts = directed.copy()
    counts[1:] = directed[1:] // 2
    if np.any(directed[1:] % 2):
        raise AssertionError("odd directed path count")
    return counts


def check_enumeration() -> dict[str, Any]:
    raw = C.load_raw_structure()
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    # ring-heavy + branched + acyclic molecules
    rows = [0, 1, 2, 7, 11]
    max_len = 4
    out = {}
    for row in rows:
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        adj = PP.adjacency(n, bu[b0:b1], bv[b0:b1])
        paths, counts = PP.enumerate_paths(adj, max_len)
        ref = brute_force_paths(adj, max_len)
        if not np.array_equal(counts, ref):
            raise AssertionError(f"row {row}: counts {counts} != brute force {ref}")
        if counts[0] != n:
            raise AssertionError("length-0 count != atoms")
        if counts[1] != (b1 - b0):
            raise AssertionError("length-1 count != bonds")
        if len(paths) != counts.sum():
            raise AssertionError("path list length != counted paths")
        out[f"row{row}"] = counts.tolist()
    # a cycle: every edge is on a cycle and both directions collapse
    adj = [[1, 3], [0, 2], [1, 3], [2, 0]]
    _, counts = PP.enumerate_paths(adj, 4)
    ref = brute_force_paths(adj, 4)
    if not np.array_equal(counts, ref):
        raise AssertionError(f"cycle counts {counts} != {ref}")
    # duplicates at identical token sequences must be distinct occurrences
    atom = np.zeros((2, C.ATOM_COLUMNS), np.uint8)       # two identical atoms
    bond = np.zeros((1, C.BOND_COLUMNS), np.uint8)
    bond_row = np.asarray([[-1, 0], [0, -1]])
    a1_, b1_, k1 = PP.tokens_for((0, 1), atom, bond, bond_row)
    a2_, b2_, k2 = PP.tokens_for((1, 0), atom, bond, bond_row)
    out["identical_atom_pair_same_key"] = bool(k1 == k2)
    return out


# ---------------------------------------------------------------------------
# 2. canonicalisation / renumbering
# ---------------------------------------------------------------------------


def check_renumbering() -> dict[str, Any]:
    import torch

    raw = C.load_raw_structure()
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"], np.uint8)
    bond_feat = np.asarray(raw["bond_feat"], np.uint8)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    rng = np.random.default_rng(7)
    max_len = 4
    keys_ref = None
    for trial in range(3):
        row = 2  # same molecule, independently renumbered each trial
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        perm = rng.permutation(n)
        # atom i is renumbered to perm[i]: new_atoms[perm[i]] = old_atoms[i]
        mol_atoms = atom_feat[a0:a1][np.argsort(perm)]
        mol_bonds = bond_feat[b0:b1]
        adj = PP.adjacency(n, perm[bu[b0:b1]], perm[bv[b0:b1]])
        bond_row = np.full((n, n), -1, np.int64)
        for e, (u, v) in enumerate(zip(perm[bu[b0:b1]], perm[bv[b0:b1]])):
            bond_row[int(u), int(v)] = e
            bond_row[int(v), int(u)] = e
        paths, counts = PP.enumerate_paths(adj, max_len)
        keys = sorted(PP.tokens_for(seq, mol_atoms, mol_bonds, bond_row)[2] for seq in paths)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise AssertionError("canonical keys changed under atom renumbering")
    return {"rows": "row2 + 2 random rows", "keys": len(keys_ref or [])}


# ---------------------------------------------------------------------------
# 3/4. padding, reversal, batching, BAG invariance
# ---------------------------------------------------------------------------


def _synthetic_paths(L: int, dims, n_rows: int = 3
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    atom_dims, bond_dims = dims
    rng = np.random.default_rng(11)
    atom = np.zeros((n_rows, L + 1, C.ATOM_COLUMNS), np.int64)
    for i, d in enumerate(atom_dims):
        atom[:, :, i] = rng.integers(0, int(d), size=(n_rows, L + 1))
    bond = np.zeros((n_rows, L, C.BOND_COLUMNS), np.int64)
    for i, d in enumerate(bond_dims):
        bond[:, :, i] = rng.integers(0, int(d), size=(n_rows, L))
    plen = rng.integers(0, L + 1, size=n_rows)
    names = [f"p{i}" for i in range(n_rows)]
    return atom, bond, plen, names


def check_padding_reversal(dims) -> dict[str, Any]:
    import torch

    L = 5
    atom_dims, bond_dims = dims
    model, _ = M.build_model(atom_dims, bond_dims, L, "SUM", 0)
    model.eval()
    atom, bond, plen, _ = _synthetic_paths(L, dims, 4)
    atom_t = torch.as_tensor(atom).long()
    bond_t = torch.as_tensor(bond).long()
    plen_t = torch.as_tensor(plen).long()
    x = model.encoder.build_input(atom_t, bond_t, plen_t, reverse=False)
    xr = model.encoder.build_input(atom_t, bond_t, plen_t, reverse=True)
    if x.shape[1] != 289:
        raise AssertionError(f"input dim {x.shape[1]} != 289 for L=5")
    for i in range(4):
        l = int(plen[i])
        a = x[i, : 32 * (l + 1)].reshape(l + 1, 32)
        ar = xr[i, : 32 * (l + 1)].reshape(l + 1, 32)
        if not torch.allclose(a.flip(0), ar, atol=1e-6):
            raise AssertionError("reversal does not reverse valid atom slots")
        # padded atom slots stay exactly zero
        if l + 1 < L + 1 and torch.abs(x[i, 32 * (l + 1):32 * (L + 1)]).max() > 0:
            raise AssertionError("padding leaked into atom slots")
    # masks must be 0 on padded slots
    masks = x[i, 32 * (L + 1) + 16 * L: 32 * (L + 1) + 16 * L + L + 1]
    if int(masks.sum()) != l + 1:
        raise AssertionError("atom mask length mismatch")
    return {"input_dim": int(x.shape[1]), "plen": plen.tolist()}


def check_batch_vs_single(dims) -> dict[str, Any]:
    import torch

    cfg = C.load_config()
    bundle = D.load_bundle()
    split = C.split_spec("D0", bundle.raw)
    L = 3
    rows = np.union1d(bundle.official_train, bundle.official_valid)
    meta = PP.ensure_cache(bundle.raw, rows, L)
    cache = PP.PathCache(meta, device="cpu")
    c_l = PP.c_l_from_cache(cache, split["fit_rows"])
    atom_dims, bond_dims = dims
    model, _ = M.build_model(atom_dims, bond_dims, L, "SUM", 0)
    model.eval()
    sample = np.asarray(split["fit_rows"])[:16]
    with torch.no_grad():
        b = cache.gather(sample, c_l, bag=False)
        batch_out = model.forward_batch(b)["yhat_std"].numpy()
        single = np.asarray([model.forward_batch(cache.gather([int(r)], c_l, False))
                             ["yhat_std"].item() for r in sample])
    err = float(np.max(np.abs(batch_out - single)))
    if err > 1e-5:
        raise AssertionError(f"batch/single mismatch {err}")
    # repeated path counts: batch-weighted SUM of local scores equals the ledger
    ids = cache.ent_path[cache.mol_ptr[cache.positions(sample)[0]]:cache.mol_ptr[cache.positions(sample)[0] + 1]]
    mult = cache.ent_mult[cache.mol_ptr[cache.positions(sample)[0]]:cache.mol_ptr[cache.positions(sample)[0] + 1]]
    s = M.path_scores(model, cache, ids)
    ledger = float(np.sum(mult * c_l[cache.path_len[ids]] * s)) + float(model.beta)
    if abs(ledger - single[0]) > 1e-4:
        raise AssertionError(f"explicit occurrence ledger {ledger} != {single[0]}")
    return {"max_abs_batch_single": err, "max_len": L}


def check_bag_invariance(dims) -> dict[str, Any]:
    import torch

    L = 3
    atom_dims, bond_dims = dims
    model, _ = M.build_model(atom_dims, bond_dims, L, "POOL", 0)
    model.eval()
    raw = C.load_raw_structure()
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"], np.uint8)
    bond_feat = np.asarray(raw["bond_feat"], np.uint8)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    row = 3
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    n = a1 - a0
    adj = PP.adjacency(n, bu[b0:b1], bv[b0:b1])
    bond_row = np.full((n, n), -1, np.int64)
    for e, (u, v) in enumerate(zip(bu[b0:b1], bv[b0:b1])):
        bond_row[int(u), int(v)] = e
        bond_row[int(v), int(u)] = e
    paths, _ = PP.enumerate_paths(adj, L)
    rng = np.random.default_rng(3)
    atom_vals, bond_vals, plens = [], [], []
    for seq in paths:
        a, b, _ = PP.tokens_for(seq, atom_feat[a0:a1], bond_feat[b0:b1], bond_row)
        perm = rng.permutation(a.shape[0])
        a_perm = a[perm]
        b_perm = b[rng.permutation(b.shape[0])] if b.shape[0] else b
        pa = np.zeros((L + 1, C.ATOM_COLUMNS), np.uint8)
        pb = np.zeros((L, C.BOND_COLUMNS), np.uint8)
        pa[:a.shape[0]] = a
        pb[:b.shape[0]] = b
        pa2 = np.zeros_like(pa)
        pb2 = np.zeros_like(pb)
        pa2[:a_perm.shape[0]] = a_perm
        pb2[:b_perm.shape[0]] = b_perm
        atom_vals.append((pa, pa2))
        bond_vals.append((pb, pb2))
        plens.append(a.shape[0] - 1)
    def bag_of(atoms, bonds, l):
        out_a = np.zeros((L + 1, C.ATOM_COLUMNS), np.uint8)
        out_b = np.zeros((L, C.BOND_COLUMNS), np.uint8)
        out_a[:l + 1] = PP._sort_rows(atoms[:l + 1])
        if l:
            out_b[:l] = PP._sort_rows(bonds[:l])
        return out_a, out_b

    with torch.no_grad():
        def score(idx):
            pairs = [bag_of(atom_vals[i][idx], bond_vals[i][idx], plens[i])
                     for i in range(len(paths))]
            a = torch.as_tensor(np.stack([x[0] for x in pairs])).long()
            b = torch.as_tensor(np.stack([x[1] for x in pairs])).long()
            p = torch.as_tensor(np.asarray(plens)).long()
            return model.encoder(a, b, p).numpy()
        e0 = score(0)
        e1 = score(1)
        ep = np.stack([model.encoder(
            torch.as_tensor(bag_of(atom_vals[i][0], bond_vals[i][0], plens[i])[0][None]).long(),
            torch.as_tensor(bag_of(atom_vals[i][0], bond_vals[i][0], plens[i])[1][None]).long(),
            torch.as_tensor([plens[i]]).long()).numpy()[0] for i in range(len(paths))])
    err = float(np.max(np.abs(e0 - e1)))
    err_perm = float(np.max(np.abs(e0 - ep)))
    if err > 1e-5 or err_perm > 1e-5:
        raise AssertionError(f"BAG not permutation invariant: permuted {err} batched {err_perm}")
    # ORDER must distinguish a non-palindromic sequence from its reverse
    k = int(np.argmax(plens))
    a = torch.as_tensor(atom_vals[k][0][None]).long()
    b = torch.as_tensor(bond_vals[k][0][None]).long()
    p = torch.as_tensor([plens[k]]).long()
    with torch.no_grad():
        x = model.encoder.build_input(a, b, p, reverse=False)
        y = model.encoder.build_input(a, b, p, reverse=True)
    if plens[k] >= 1 and torch.allclose(x, y):
        raise AssertionError("ORDER input identical for a sequence and its reversal")
    return {"bag_perm_max_abs": err, "bag_single_max_abs": err_perm,
            "order_rev_distinguishable": True}


# ---------------------------------------------------------------------------
# 6. PAIR / DIAG algebra and sampling
# ---------------------------------------------------------------------------


def check_pair_algebra() -> dict[str, Any]:
    bundle = D.load_bundle()
    import psc_train as T
    dir_ = T.ensure_pair_directory(bundle)
    split = C.split_spec("D0", bundle.raw)
    targets = D.build_pair_targets(dir_, split["fit_rows"], bundle.labels.size)
    rng = np.random.default_rng(0)
    e = rng.normal(size=bundle.labels.size)
    w = targets.w
    pair = float(np.sum(w * (e[targets.b] - e[targets.a]) ** 2))
    diag = float(np.sum(w * (e[targets.b] ** 2 + e[targets.a] ** 2)))
    cross = float(np.sum(w * e[targets.a] * e[targets.b]))
    if abs(pair - (diag - 2 * cross)) > 1e-9:
        raise AssertionError("PAIR != DIAG - 2 cross")
    if abs(diag - float(np.sum(targets.d * e ** 2))) > 1e-9:
        raise AssertionError("DIAG != sum d_i e_i^2")
    if abs(float(w.sum()) - 1.0) > 1e-12:
        raise AssertionError("pair weights do not sum to 1")
    # sampler unbiasedness for a quadratic statistic
    sampler = D.PairSampler(targets, seed=5)
    draws = []
    for _ in range(4000):
        a, b = sampler.sample(32)
        draws.append(np.mean((e[b] - e[a]) ** 2))
    if abs(float(np.mean(draws)) - pair) > 0.05 * max(abs(pair), 1e-6):
        raise AssertionError(f"sampler mean {np.mean(draws)} far from {pair}")
    # no heldout endpoint can enter the fit pair graph
    fit = np.zeros(bundle.labels.size, bool)
    fit[np.asarray(split["fit_rows"])] = True
    if not (fit[targets.a].all() and fit[targets.b].all()):
        raise AssertionError("pair targets contain non-fit endpoints")
    return {"n_unique_pairs": int(targets.a.size), "n_components": int(targets.n_components),
            "sampler_mean": float(np.mean(draws)), "exact_pair": pair,
            "sum_w": float(w.sum())}


# ---------------------------------------------------------------------------
# 7. fit-only statistics
# ---------------------------------------------------------------------------


def check_fit_only() -> dict[str, Any]:
    bundle = D.load_bundle()
    split = C.split_spec("D0", bundle.raw)
    fit = np.asarray(split["fit_rows"], np.int64)
    dev = np.asarray(split["eval_rows"], np.int64)
    if np.intersect1d(fit, dev).size:
        raise AssertionError("fit/eval overlap")
    scaler_a = C.target_scaler(bundle.labels, fit)
    labels2 = bundle.labels.copy()
    labels2[dev] = 1e6
    scaler_b = C.target_scaler(labels2, fit)
    if scaler_a != scaler_b:
        raise AssertionError("scaler depends on heldout labels")
    from psc_train import _feature_dims
    atom_dims, bond_dims = _feature_dims()
    ok_a = D.category_support(bundle.raw, fit, atom_dims, bond_dims)
    ok_b = D.category_support(bundle.raw, fit, atom_dims, bond_dims)
    if not all(np.array_equal(x, y) for x, y in zip(ok_a[0], ok_b[0])):
        raise AssertionError("category support not deterministic")
    # a category that exists only outside the fit rows must be masked out
    node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    dev_atoms = set()
    for row in dev.tolist():
        dev_atoms.update(np.asarray(bundle.raw["atom_feat"])[int(node_ptr[row]):
                                                              int(node_ptr[row + 1]), 0].tolist())
    fit_atoms = set(np.asarray(bundle.raw["atom_feat"])[
        np.concatenate([np.arange(int(node_ptr[r]), int(node_ptr[r + 1])) for r in fit]).astype(np.int64), 0].tolist())
    only_dev = sorted(dev_atoms - fit_atoms)
    masked = [int(x) for x in only_dev if not ok_a[0][int(x)]]
    if only_dev and len(masked) != len(only_dev):
        raise AssertionError("heldout-only raw category not masked")
    return {"n_fit": int(fit.size), "n_dev": int(dev.size),
            "heldout_only_atom_categories": len(only_dev), "masked": len(masked),
            "scaler": scaler_a}


# ---------------------------------------------------------------------------
# 8. initialisation + gradient + overfit smoke
# ---------------------------------------------------------------------------


def check_init_and_grad(dims) -> dict[str, Any]:
    import torch

    L = 3
    atom_dims, bond_dims = dims
    models = {}
    for readout in ("SUM", "POOL"):
        m, h = M.build_model(atom_dims, bond_dims, L, readout, 0)
        models[readout] = (m, h)
    if models["SUM"][1] != models["POOL"][1]:
        raise AssertionError("shared init differs between SUM and POOL")
    m, hashes = models["SUM"]
    atom_np, bond_np, plen_np, _ = _synthetic_paths(L, dims, 12)
    atom = torch.as_tensor(atom_np).long()
    bond = torch.as_tensor(bond_np).long()
    plen = torch.as_tensor(plen_np).long()
    batch = {"atom": atom, "bond": bond, "plen": plen,
             "inv": torch.arange(12), "owner": torch.arange(12),
             "weight": torch.ones(12), "n_molecules": 12, "is_bag": False,
             "n_entries": 12}
    out = m.forward_batch(batch)
    if not torch.isfinite(out["yhat_std"]).all():
        raise AssertionError("non-finite forward")
    loss = (out["yhat_std"] ** 2).mean()
    loss.backward()
    grads = {name: float(p.grad.abs().sum()) for name, p in m.named_parameters()
             if p.grad is not None}
    zero = [k for k, v in grads.items() if v == 0.0]
    if zero:
        raise AssertionError(f"zero gradients on {zero}")
    m2, _ = M.build_model(atom_dims, bond_dims, L, "SUM", 1)
    if m2.shared_state()["encoder_sha256"] == hashes["encoder_sha256"]:
        raise AssertionError("different seeds produced identical init")
    return {"sum_pool_init_equal": True, "n_grad_tensors": len(grads),
            "params": M.parameter_ledger(m)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heavy", action="store_true")
    args = ap.parse_args(argv)
    from psc_train import _feature_dims

    dims = _feature_dims()
    check("1_enumeration", check_enumeration)
    check("2_renumbering", check_renumbering)
    check("3_padding_reversal", lambda: check_padding_reversal(dims))
    check("4_order_bag", lambda: check_bag_invariance(dims))
    if not args.skip_heavy:
        check("3b_batch_vs_single", lambda: check_batch_vs_single(dims))
        check("6_pair_algebra", check_pair_algebra)
        check("7_fit_only", check_fit_only)
    check("8_init_grad", lambda: check_init_and_grad(dims))
    out = C.ROUND_DIR / "analysis" / "checks.json"
    C.write_json(out, {"generated_utc": C.now_utc(), "git_commit": C.git_commit(),
                       "environment": C.environment(), "checks": CHECKS})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())