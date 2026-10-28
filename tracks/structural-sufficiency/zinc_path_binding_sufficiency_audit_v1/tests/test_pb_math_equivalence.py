"""Mathematical layer of the audit: equivalence, gradient identity, kernel.

Self-contained (small real molecules rebuilt from the frozen P3 objects, frozen
A_s0 checkpoint, CPU).  These tests pin the three claims the report is allowed to
make:

* flat per-path sum ≡ branch-first (factorised) sum, **including gradients**;
* ``T_null`` is the matched-scale marginal model (unbiased for independent paths);
* ``X_v``/S2 are a non-injective compression of the path-content configuration:
  a within-cell re-assignment leaves every S2/X component identical and changes T
  (and T in turn does not determine S2).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pb_common as P
import pb_paths as PP
import pb_features as F
import lc_data as LD
from pb_run import gate_sum


def _frozen():
    model, info = P.load_frozen_a("lcrcA_s0")
    return model, info


def _real_fragment(mol_ids=(0, 1, 2)):
    """Rebuild a few valid molecules from the frozen P3 objects (real 108-d feats)."""
    objs, _ = P.L.D.load_p3_objects("valid")
    mols = [LD.build_molecule_from_object(objs[i]) for i in mol_ids]
    cache = LD._concat_molecules(mols)
    cache["split"] = "fragment"
    cache["n_molecules"] = len(mols)
    cache["meta"] = {"feature_version": P.L.FEATURE_VERSION, "split": "fragment"}
    return cache


def test_fragment_matches_frozen_cache_rows():
    va = P.load_cache("valid")
    frag = _real_fragment((0, 1, 2))
    ap = np.asarray(va["atom_ptr"])
    a_end = int(ap[3])
    assert np.array_equal(np.asarray(frag["atom_feat"]), np.asarray(va["atom_feat"])[:a_end])
    assert np.array_equal(np.asarray(frag["path_cat"]),
                          np.asarray(va["path_cat"])[: int(np.asarray(frag["path_ptr"])[-1])])
    ref = PP.crosscheck_split(frag)
    assert ref["passed"]


def test_flat_equals_factorized_and_matches_streaming_stats():
    cache = _real_fragment()
    model, _ = _frozen()
    enc = F.encode(model, cache)
    paths = PP.split_paths(cache)
    proj = P.projections(P.PRIMARY_SEED)
    q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
    t_stats = F.t_stats(paths, q, k, z, enc["n_atoms"])["T_real"]
    t_flat = F.flat_t_reference(enc["a"], enc["b"], paths, proj, enc["n_atoms"])
    t_fact = F.factorized_t_reference(enc["a"], enc["b"], paths, proj, enc["n_atoms"])
    assert np.abs(t_flat - t_fact).max() < 1e-10
    assert np.abs(t_stats - t_flat).max() < 1e-10
    assert np.abs(t_stats).max() > 1e-3


def test_gradients_of_flat_and_factorized_forms_are_identical():
    torch.manual_seed(0)
    n_atoms, d = 4, 4
    root = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    mid = torch.tensor([3, 3, 2, 3, 2, 3, 3, 1, 1])
    base = torch.randn(len(root), d, dtype=torch.float64)
    # Q must be a function of (v, w) only -- that is exactly the hypothesis of the identity
    first: dict[tuple[int, int], int] = {}
    for i, (r, m) in enumerate(zip(root.tolist(), mid.tolist())):
        first.setdefault((r, m), i)
    pick = torch.tensor([first[(r, m)] for r, m in zip(root.tolist(), mid.tolist())])
    uniq, inv = torch.unique(pick, return_inverse=True)

    def run(fn):
        # Q is *shared* inside a (v, w) branch -- that is exactly the hypothesis of
        # the identity, so the gradient that must agree is d/dQ_branch, not d/dq_path.
        table = base[: len(uniq)].clone().requires_grad_(True)
        q = table[inv]
        k = (base * 1.3).clone().requires_grad_(True)
        z = (base * 0.7).clone().requires_grad_(True)
        out = fn(q, k, z)
        out.sum().backward()
        return out.detach(), table.grad, k.grad, z.grad

    def denom():
        n = torch.bincount(root, minlength=n_atoms).to(torch.float64)
        return torch.sqrt(torch.clamp(n, min=1.0)).unsqueeze(1)

    def flat(q, k, z):
        return torch.zeros(n_atoms, d, dtype=torch.float64).index_add(0, root, q * k * z) / denom()

    def factorized(q, k, z):
        pair = root * n_atoms + mid
        total = torch.zeros(n_atoms, d, dtype=torch.float64)
        for p in torch.unique(pair):
            sel = pair == p
            v = int(root[sel][0])
            qv = q[sel][0]
            total = total.index_add(0, torch.tensor([v]), (qv * (k[sel] * z[sel]).sum(0)).unsqueeze(0))
        return total / denom()

    o1, gq1, gk1, gz1 = run(flat)
    o2, gq2, gk2, gz2 = run(factorized)
    assert torch.allclose(o1, o2, atol=1e-12)
    assert torch.allclose(gq1, gq2, atol=1e-12)
    assert torch.allclose(gk1, gk2, atol=1e-12)
    assert torch.allclose(gz1, gz2, atol=1e-12)


def test_null_is_unbiased_for_independent_paths():
    rng = np.random.default_rng(7)
    draws = 2000
    for n in (1, 2, 4):
        dsum = np.zeros(8)
        for _ in range(draws):
            q, k, z = (rng.normal(size=(n, 8)) for _ in range(3))
            dsum += (q * k * z).sum(0) / np.sqrt(n) \
                - np.sqrt(n) * q.mean(0) * k.mean(0) * z.mean(0)
        sd = np.sqrt((dsum ** 2).mean() / draws)
        assert np.abs(dsum / draws).max() / max(sd, 1e-12) < 5.0
    # n = 1 -> no within-root variation -> D is identically zero
    q, k, z = (rng.normal(size=(1, 8)) for _ in range(3))
    assert np.allclose((q * k * z).sum(0), np.sqrt(1) * q.mean(0) * k.mean(0) * z.mean(0))


def test_within_cell_reassignment_keeps_S2_and_changes_T():
    cache = _real_fragment()
    model, _ = _frozen()
    enc = F.encode(model, cache)
    paths = PP.split_paths(cache)
    proj = P.projections(P.PRIMARY_SEED)
    q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
    cat = np.asarray(cache["path_cat"], dtype=np.int64)
    end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    clos = np.asarray(cache["path_closure"], dtype=np.int64)
    cell = ((paths["root"].astype(np.int64) * 189 + cat) * 63 + end63) * 2 + clos
    uniq, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
    for ci in np.argsort(counts)[::-1]:
        members = np.flatnonzero(inverse == ci)
        if len(members) < 2:
            continue
        root = int(paths["root"][members[0]])
        others = np.flatnonzero(paths["root"] == root)
        pos = {int(p): i for i, p in enumerate(others)}
        local = [pos[int(p)] for p in members]
        ends = paths["end"]
        perm_ends = ends[others].copy()
        perm_ends[local] = ends[members][::-1]
        c_real = gate_sum(model, cat[others], enc["a"][ends[others]])
        c_perm = gate_sum(model, cat[others], enc["a"][perm_ends])
        assert np.abs(c_real - c_perm).max() < 1e-12  # S2 content summary identical
        kk, zz = k[others].copy(), z[others].copy()
        for src, dst in zip(members, members[::-1]):
            kk[pos[int(dst)]] = k[int(src)]
            zz[pos[int(dst)]] = z[int(src)]
        t_real = (q[others] * k[others] * z[others]).sum(0) / np.sqrt(len(others))
        t_swap = (q[others] * kk * zz).sum(0) / np.sqrt(len(others))
        assert np.abs(t_real - t_swap).max() > 1e-4
        return
    pytest.skip("no root with a multi-path (cat,end63,closure) cell in the fragment")


def test_T_does_not_determine_S2_either():
    """Same endpoint swap: T is exactly invariant, the typed content summary moves.

    Two two-paths that share the middle atom ``w`` have the same ``Q``; exchanging
    their endpoints permutes the summands of ``T`` (so ``T`` is unchanged) but
    re-assigns ``a_u`` across two different path types inside ``content_agg``.
    """
    cache = _real_fragment()
    model, _ = _frozen()
    enc = F.encode(model, cache)
    paths = PP.split_paths(cache)
    proj = P.projections(P.PRIMARY_SEED)
    q, k, z = F.path_factors(enc["a"], enc["b"], paths, proj)
    cat = np.asarray(cache["path_cat"], dtype=np.int64)
    a_all = enc["a"]
    gates = np.asarray(model.local.path_gate.weight.detach()).ravel()
    for root in np.unique(paths["root"]):
        sel = np.flatnonzero(paths["root"] == root)
        for i in sel:
            for j in sel:
                if i == j or paths["mid"][i] != paths["mid"][j]:
                    continue
                if cat[i] == cat[j] or abs(gates[cat[i]] - gates[cat[j]]) < 1e-4:
                    continue
                if np.abs(a_all[paths["end"][i]] - a_all[paths["end"][j]]).max() < 1e-3:
                    continue  # identical content would make the counterfactual vacuous
                t0 = (q[sel] * k[sel] * z[sel]).sum(0)
                # exchange the two endpoints (same branch -> same Q -> T unchanged)
                kk, zz = k[sel].copy(), z[sel].copy()
                pi, pj = int(np.flatnonzero(sel == i)[0]), int(np.flatnonzero(sel == j)[0])
                kk[pi], kk[pj] = k[j], k[i]
                zz[pi], zz[pj] = z[j], z[i]
                t1 = (q[sel] * kk * zz).sum(0)
                assert np.allclose(t0, t1, atol=1e-12)
                contents = a_all[paths["end"][sel]].copy()
                contents[pi], contents[pj] = a_all[paths["end"][j]], a_all[paths["end"][i]]
                c0 = gate_sum(model, cat[sel], a_all[paths["end"][sel]])
                c1 = gate_sum(model, cat[sel], contents)
                assert np.abs(c0 - c1).max() > 1e-7
                return
    pytest.skip("no suitable root in the fragment")
