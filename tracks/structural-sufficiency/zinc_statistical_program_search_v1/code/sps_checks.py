"""Production correctness checks for the statistical-program bodies.

These run on real cached ZINC *structures* (no labels are used except where a
loss value is needed for a finite-difference check; targets there are synthetic
constants).  No official test structure is loaded.

Run:  uv run python .../sps_checks.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_AD = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_AD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import sps_data as SD  # noqa: E402
import sps_models as M  # noqa: E402

RESULTS: dict[str, object] = {}


def _to_dtype(batch: dict, dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


def _one_ad(ad, row):
    a0, a1 = int(ad["atom_ptr"][row]), int(ad["atom_ptr"][row + 1])
    b0, b1 = int(ad["bond_ptr"][row]), int(ad["bond_ptr"][row + 1])
    d0, d1 = int(ad["dist_ptr"][row]), int(ad["dist_ptr"][row + 1])
    n = a1 - a0
    return {
        "n": n,
        "m": b1 - b0,
        "atom_feat": np.asarray(ad["atom_feat"])[a0:a1],
        "bond_feat": np.asarray(ad["bond_feat"])[b0:b1],
        "bond_index": np.asarray(ad["bond_index"])[b0:b1].astype(np.int64),
        "dist": np.asarray(ad["dist_flat"])[d0:d1].reshape(n, n).astype(np.int64),
        "global": np.asarray(ad["global_feat"])[row],
    }


def q_batch(mols, dtype=torch.float32):
    """Build a Q batch from raw single-molecule dicts (independent of the cache view)."""
    ar = []
    br = []
    atom_feat, bond_feat, bond_index = [], [], []
    atom_off = 0
    bond_off = 0
    a_counts, b_counts, root_cnt, glob = [], [], [], []
    for mol in mols:
        n, m = mol["n"], mol["m"]
        Dm = mol["dist"]
        bu, bv = mol["bond_index"][:, 0], mol["bond_index"][:, 1]
        dflat = Dm.reshape(-1)
        grp = np.where(dflat <= 2, 0, np.where(dflat <= 4, 1, 2))
        root = np.repeat(np.arange(n), n)
        atom = np.tile(np.arange(n), n)
        ar.append((root + atom_off, atom + atom_off, np.minimum(dflat, 9), grp))
        cnt = np.zeros((n, 6))
        for g in range(3):
            cnt[:, g] = (grp.reshape(n, n) == g).sum(1)
        dmin = np.minimum(Dm[:, bu], Dm[:, bv])
        bgrp = np.where(dmin.reshape(-1) <= 2, 0, np.where(dmin.reshape(-1) <= 4, 1, 2))
        broot = np.repeat(np.arange(n), m)
        blocal = np.tile(np.arange(m), n)
        iu = broot * n + bu[blocal]
        iv = broot * n + bv[blocal]
        br.append((broot + atom_off, blocal + bond_off, bgrp, iu, iv))
        for g in range(3):
            cnt[:, 3 + g] = (bgrp.reshape(n, m) == g).sum(1) if m else 0.0
        root_cnt.append(cnt)
        atom_feat.append(mol["atom_feat"])
        bond_feat.append(mol["bond_feat"])
        bond_index.append(mol["bond_index"] + atom_off)
        a_counts.append(n)
        b_counts.append(m)
        glob.append(mol["global"])
        atom_off += n
        bond_off += m
    A = sum(a_counts)
    rec_base = np.concatenate([[0], np.cumsum([len(x[0]) for x in ar])[:-1]])
    batch = {
        "n_graphs": len(mols),
        "atom_feat": torch.as_tensor(np.concatenate(atom_feat), dtype=dtype),
        "bond_feat": torch.as_tensor(np.concatenate(bond_feat), dtype=dtype),
        "bond_index": torch.as_tensor(np.concatenate(bond_index), dtype=torch.int64),
        "atom_batch": torch.as_tensor(np.repeat(np.arange(len(mols)), a_counts), dtype=torch.int64),
        "atom_local": torch.as_tensor(
            np.concatenate([np.arange(n) for n in a_counts]), dtype=torch.int64),
        "bond_batch": torch.as_tensor(np.repeat(np.arange(len(mols)), b_counts), dtype=torch.int64),
        "bond_local": torch.as_tensor(
            np.concatenate([np.arange(max(m, 0)) for m in b_counts]), dtype=torch.int64),
        "global": torch.as_tensor(np.stack(glob), dtype=dtype),
        "atom_counts": torch.as_tensor(np.asarray(a_counts), dtype=dtype),
        "bond_counts": torch.as_tensor(np.asarray(b_counts), dtype=dtype),
        "atom_rec_atom": torch.as_tensor(np.concatenate([x[1] for x in ar]), dtype=torch.int64),
        "atom_rec_dist": torch.as_tensor(np.concatenate([x[2] for x in ar]), dtype=torch.int64),
        "atom_rec_group": torch.as_tensor(np.concatenate([x[3] for x in ar]), dtype=torch.int64),
        "atom_rec_root": torch.as_tensor(np.concatenate([x[0] for x in ar]), dtype=torch.int64),
        "bond_rec_bond": torch.as_tensor(np.concatenate([x[1] for x in br]), dtype=torch.int64),
        "bond_rec_group": torch.as_tensor(np.concatenate([x[2] for x in br]), dtype=torch.int64),
        "bond_rec_root": torch.as_tensor(np.concatenate([x[0] for x in br]), dtype=torch.int64),
        "bond_rec_iu": torch.as_tensor(
            np.concatenate([x[3] + rec_base[i] for i, x in enumerate(br)]), dtype=torch.int64),
        "bond_rec_iv": torch.as_tensor(
            np.concatenate([x[4] + rec_base[i] for i, x in enumerate(br)]), dtype=torch.int64),
        "root_cnt": torch.as_tensor(np.concatenate(root_cnt), dtype=dtype),
    }
    del A
    return batch


def f_batch(mols, dtype=torch.float32):
    B = len(mols)
    N = max(m["n"] for m in mols)
    E = max(1, max(m["m"] for m in mols))
    atom_feat, bond_feat, bond_index, glob = [], [], [], []
    atom_batch, atom_local, bond_batch, bond_local = [], [], [], []
    edge_u = np.zeros((B, E), dtype=np.int64)
    edge_v = np.zeros((B, E), dtype=np.int64)
    edge_mask = np.zeros((B, E), dtype=bool)
    atom_off = 0
    bond_off = 0
    a_counts, b_counts = [], []
    for i, mol in enumerate(mols):
        n, m = mol["n"], mol["m"]
        atom_feat.append(mol["atom_feat"])
        bond_feat.append(mol["bond_feat"])
        bond_index.append(mol["bond_index"] + atom_off)
        atom_batch.append(np.arange(n) + 0)
        atom_local.append(np.arange(n))
        bond_batch.append(np.full(m, i))
        bond_local.append(np.arange(m))
        if m:
            edge_u[i, :m] = mol["bond_index"][:, 0]
            edge_v[i, :m] = mol["bond_index"][:, 1]
            edge_mask[i, :m] = True
        glob.append(mol["global"])
        a_counts.append(n)
        b_counts.append(m)
        atom_off += n
        bond_off += m
    return {
        "n_graphs": B, "Nmax": N, "Emax": E,
        "atom_feat": torch.as_tensor(np.concatenate(atom_feat), dtype=dtype),
        "bond_feat": torch.as_tensor(np.concatenate(bond_feat), dtype=dtype),
        "bond_index": torch.as_tensor(np.concatenate(bond_index), dtype=torch.int64),
        "atom_batch": torch.as_tensor(np.concatenate([np.full(len(x), i) for i, x in enumerate(atom_local)]), dtype=torch.int64),
        "atom_local": torch.as_tensor(np.concatenate(atom_local), dtype=torch.int64),
        "bond_batch": torch.as_tensor(np.concatenate([np.full(len(x), i) for i, x in enumerate(bond_local)]), dtype=torch.int64),
        "bond_local": torch.as_tensor(np.concatenate(bond_local), dtype=torch.int64),
        "global": torch.as_tensor(np.stack(glob), dtype=dtype),
        "atom_counts": torch.as_tensor(np.asarray(a_counts), dtype=dtype),
        "bond_counts": torch.as_tensor(np.asarray(b_counts), dtype=dtype),
        "edge_u": torch.as_tensor(edge_u), "edge_v": torch.as_tensor(edge_v),
        "edge_mask": torch.as_tensor(edge_mask),
        "edge_batch": torch.as_tensor(np.concatenate([np.full(m, i) for i, m in enumerate(b_counts)]), dtype=torch.int64),
        "edge_local": torch.as_tensor(np.concatenate([np.arange(m) for m in b_counts]), dtype=torch.int64),
    }


def permute_mol(mol, perm_atom, perm_bond=None, flip=None):
    """Relabel atoms / reorder-flip physical bonds; the graph is unchanged."""
    n, m = mol["n"], mol["m"]
    out = dict(mol)
    out["atom_feat"] = mol["atom_feat"][perm_atom]
    inv = np.argsort(perm_atom)
    out["dist"] = mol["dist"][np.ix_(perm_atom, perm_atom)]
    bi = mol["bond_index"][perm_bond] if perm_bond is not None else mol["bond_index"].copy()
    bi = inv[bi]
    if flip is not None:
        f = np.asarray(flip, dtype=bool)
        bi[f] = bi[f][:, ::-1]
    out["bond_index"] = bi
    out["bond_feat"] = mol["bond_feat"][perm_bond] if perm_bond is not None else mol["bond_feat"].copy()
    return out


def _fit():
    return SD.load_fit()


def _model(cand, seed=0, dtype=torch.float32):
    model, info = M.build_model(cand, _fit(), seed, torch.device("cpu"),
                                ssl_path=None, dropout=0.0)
    if dtype == torch.float64:
        model = model.double()
    model.eval()
    return model, info


def measure_matches_reference():
    """Q measure: vectorised scatter read == brute-force per-root reference."""
    torch.manual_seed(7)
    R, A, G = 37, 5, 6
    k = torch.randn(R, 16, dtype=torch.float64)
    v = torch.randn(R, 8, dtype=torch.float64)
    root = torch.randint(0, A, (R,), dtype=torch.int64)
    grp = torch.randint(0, G, (R,), dtype=torch.int64)
    q = torch.randn(4, 16, dtype=torch.float64)
    cnt = torch.zeros(A, G, dtype=torch.float64)
    for r, g in zip(root.tolist(), grp.tolist()):
        cnt[r, g] += 1
    model, _ = _model("QF", dtype=torch.float64)
    got = model._read(k, v, root, grp, q, A, cnt).reshape(A, G, 4, 17)
    err = 0.0
    for a in range(A):
        for g in range(G):
            m = (root == a) & (grp == g)
            if not bool(m.any()):
                continue
            sc = q @ k[m].T / math.sqrt(16)
            lse = torch.logsumexp(sc, dim=-1) - math.log(float(m.sum()))
            al = sc.softmax(-1)
            ref = torch.cat([lse[:, None], al @ v[m], al @ (v[m] ** 2)], -1)
            err = max(err, float((got[a, g] - ref).abs().max()))
    RESULTS["q_measure_vs_bruteforce_max_abs_error"] = err
    # every empty (root, group) segment must be exactly zero, never NaN
    empty = (cnt == 0)
    RESULTS["q_measure_empty_group_is_zero"] = bool(
        torch.count_nonzero(got[empty]) == 0 and bool(torch.isfinite(got).all())
    )
    # count is information: duplicating records is not a no-op
    k2 = torch.cat([k, k]); v2 = torch.cat([v, v])
    root2 = torch.cat([root, root]); grp2 = torch.cat([grp, grp]); cnt2 = cnt * 2
    got2 = model._read(k2, v2, root2, grp2, q, A, cnt2).reshape(A, G, 4, 17)
    RESULTS["q_measure_duplicate_records_max_abs_diff"] = float((got2 - got).abs().max())
    return err


def forest_identity_and_reference():
    ad = D.load_cache("valid")
    mol = _one_ad(ad, 0)
    model, _ = _model("FC", dtype=torch.float64)
    batch = _to_dtype(f_batch([mol], dtype=torch.float64), torch.float64)
    with torch.no_grad():
        a, b = model.encoders(batch["atom_feat"], batch["bond_feat"], batch["bond_index"])
        w = 0.1 + 1.9 * torch.sigmoid(model.conductance(b).squeeze(-1))
        n, m = mol["n"], mol["m"]
        bi = mol["bond_index"]
        # dense reference in float64 with an explicit incidence matrix
        S = np.zeros((n, m))
        for j, (u, v) in enumerate(bi):
            S[u, j] = 1.0
            S[v, j] = -1.0
        worst_id = 0.0
        r_ref, p_ref = {}, {}
        for tau in M.SCALES:
            L = S @ np.diag(tau * w.detach().numpy()) @ S.T
            Q = np.linalg.inv(np.eye(n) + L)
            r = np.diag(Q)
            p = tau * w.detach().numpy() * (
                Q[bi[:, 0], bi[:, 0]] + Q[bi[:, 1], bi[:, 1]]
                - Q[bi[:, 0], bi[:, 1]] - Q[bi[:, 1], bi[:, 0]]
            )
            r_ref[tau] = r
            p_ref[tau] = p
            worst_id = max(worst_id, float(abs(r.sum() + p.sum() - n)))
    RESULTS["f_forest_identity_max_abs_error"] = worst_id
    # independent enumeration of all forests for a 4-cycle + chord
    g_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    wt = np.array([0.4, 1.3, 0.7, 1.9, 0.6])
    tot = 0.0
    re = np.zeros(4)
    ee = np.zeros(len(g_edges))
    for bits in itertools.product([0, 1], repeat=len(g_edges)):
        parent = list(range(4))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ok, prod = True, 1.0
        for j, use in enumerate(bits):
            if not use:
                continue
            u, v = g_edges[j]
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            parent[ru] = rv
            prod *= wt[j]
        if not ok:
            continue
        comps: dict[int, list[int]] = {}
        for i in range(4):
            comps.setdefault(find(i), []).append(i)
        wsum = prod * math.prod(len(c) for c in comps.values())
        tot += wsum
        for comp in comps.values():
            for i in comp:
                re[i] += wsum / len(comp)
        for j, use in enumerate(bits):
            if use:
                ee[j] += wsum
    # matrix formula on the same graph (tau=1, weights wt)
    S = np.zeros((4, len(g_edges)))
    for j, (u, v) in enumerate(g_edges):
        S[u, j] = 1.0
        S[v, j] = -1.0
    Q = np.linalg.inv(np.eye(4) + S @ np.diag(wt) @ S.T)
    r_m = np.diag(Q)
    p_m = wt * np.array([
        Q[u, u] + Q[v, v] - Q[u, v] - Q[v, u] for u, v in g_edges
    ])
    RESULTS["f_enumeration"] = {
        "logZ_matrix_vs_enumeration_abs_error": float(abs(np.log(np.linalg.det(np.eye(4) + S @ np.diag(wt) @ S.T)) - math.log(tot))),
        "root_max_abs_error": float(np.abs(r_m - re / tot).max()),
        "edge_max_abs_error": float(np.abs(p_m - ee / tot).max()),
        "roots_plus_edges": float(r_m.sum() + p_m.sum()),
    }
    return worst_id


def invariance_and_batching():
    ad = D.load_cache("valid")
    mols = [_one_ad(ad, i) for i in (0, 1, 2)]
    rng = np.random.default_rng(11)
    out = {}
    for cand in ("QF", "FC"):
        model, _ = _model(cand, dtype=torch.float64)
        mk = q_batch if cand == "QF" else f_batch
        base = model(_to_dtype(mk(mols, dtype=torch.float64), torch.float64))
        perm_mols = []
        for mol in mols:
            n, m = mol["n"], mol["m"]
            perm_atom = rng.permutation(n)
            perm_bond = rng.permutation(m)
            flip = rng.integers(0, 2, size=m).astype(bool)
            perm_mols.append(permute_mol(mol, perm_atom, perm_bond, flip))
        perm = model(_to_dtype(mk(perm_mols, dtype=torch.float64), torch.float64))
        singles = [model(_to_dtype(mk([mol], dtype=torch.float64), torch.float64)) for mol in mols]
        singles_cat = torch.cat(singles)
        rev = model(_to_dtype(mk(mols[::-1], dtype=torch.float64), torch.float64))
        out[cand] = {
            "atom_bond_permutation_max_abs_diff": float((perm - base).abs().max()),
            "batch_recomposition_max_abs_diff": float((singles_cat - base).abs().max()),
            "batch_order_max_abs_diff": float((rev - base.flip(0)).abs().max()),
        }
        # train/eval parity (no batch-dependent normalisation is used)
        model.train()
        tr = model(_to_dtype(mk(mols, dtype=torch.float64), torch.float64))
        out[cand]["train_eval_max_abs_diff"] = float((tr - base).abs().max())
        model.eval()
    RESULTS["invariances"] = out
    return out


def finite_difference():
    ad = D.load_cache("valid")
    mols = [_one_ad(ad, 3)]
    out = {}
    for cand, names in (("QF", ["query", "role_atom"]), ("FC", ["conductance"])):
        model, _ = _model(cand, dtype=torch.float64)
        mk = q_batch if cand == "QF" else f_batch
        batch = _to_dtype(mk(mols, dtype=torch.float64), torch.float64)
        y = torch.tensor([0.37], dtype=torch.float64)
        params = []
        for nm, p in model.named_parameters():
            if any(nm.startswith(pref) for pref in names):
                params.append((nm, p))
        loss = ((model(batch) - y) ** 2).sum()
        grads = torch.autograd.grad(loss, [p for _, p in params])
        errs = {}
        for (nm, p), g in zip(params, grads):
            flat = p.reshape(-1)
            idx = torch.arange(0, flat.numel(), max(1, flat.numel() // 5))[:5]
            worst = 0.0
            eps = 1e-6
            for j in idx.tolist():
                orig = float(flat[j])
                with torch.no_grad():
                    flat[j] = orig + eps
                lp = float(((model(batch) - y) ** 2).sum())
                with torch.no_grad():
                    flat[j] = orig - eps
                lm = float(((model(batch) - y) ** 2).sum())
                with torch.no_grad():
                    flat[j] = orig
                fd = (lp - lm) / (2 * eps)
                an = float(g.reshape(-1)[j])
                rel = abs(fd - an) / max(1e-8, abs(fd) + abs(an))
                worst = max(worst, rel)
            errs[nm] = worst
        out[cand] = errs
    RESULTS["finite_difference_relative_error"] = out
    return out


def parameter_and_optimizer_audit():
    out = {}
    for cand in M.CANDIDATES:
        model, info = _model(cand)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        out[cand] = {
            "parameters": info["parameters"],
            "under_500k": bool(info["parameters"] <= 500_000),
            "optimizer_parameters": sum(p.numel() for g in opt.param_groups for p in g["params"]),
            "all_grad_enabled": all(p.requires_grad for p in model.parameters()),
            "breakdown": info["parameters_breakdown"],
        }
    RESULTS["parameter_audit"] = out
    return out


def no_bn_and_checkpoint_reload():
    ad = D.load_cache("valid")
    mol = _one_ad(ad, 5)
    out = {}
    for cand in ("QF", "FC"):
        model, _ = _model(cand)
        has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
        mk = q_batch if cand == "QF" else f_batch
        batch = mk([mol])
        with torch.no_grad():
            p0 = model(batch)
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        model2, _ = _model(cand)
        model2.load_state_dict(sd)
        with torch.no_grad():
            p1 = model2(batch)
        out[cand] = {"has_batchnorm": has_bn,
                     "reload_max_abs_diff": float((p0 - p1).abs().max())}
    RESULTS["reload"] = out


def edge_cases():
    ad = D.load_cache("valid")
    mol = _one_ad(ad, 0)
    # 2-atom single-bond molecule: groups 1,2 are empty for every root
    tiny = {
        "n": 2, "m": 1,
        "atom_feat": mol["atom_feat"][:2],
        "bond_feat": mol["bond_feat"][:1],
        "bond_index": np.array([[0, 1]]),
        "dist": np.array([[0, 1], [1, 0]]),
        "global": mol["global"],
    }
    out = {}
    mq, _ = _model("QF")
    batch = q_batch([tiny])
    with torch.no_grad():
        p = mq(batch)
    out["q_tiny_finite"] = bool(torch.isfinite(p).all())
    # far / mid bond groups (indices 4,5) are empty: they must be exactly zero
    out["q_tiny_empty_bond_groups"] = bool(
        (batch["root_cnt"][:, 4:] == 0).all() and (batch["root_cnt"][:, 3] == 1).all()
        and torch.count_nonzero(batch["root_cnt"][:, 4:] * 1.0) == 0
    )
    # isolated-atom molecule (single atom, no bonds): F must not divide by zero
    iso = dict(tiny)
    iso.update({"n": 1, "m": 0, "atom_feat": tiny["atom_feat"][:1],
                "bond_feat": tiny["bond_feat"][:0],
                "bond_index": np.zeros((0, 2), dtype=np.int64),
                "dist": np.zeros((1, 1), dtype=np.int64)})
    mf, _ = _model("FC")
    with torch.no_grad():
        pf = mf(f_batch([iso]))
    out["f_isolated_atom_finite"] = bool(torch.isfinite(pf).all())
    RESULTS["edge_cases"] = out
    return out


def cache_and_collate_consistency():
    """The precomputed cache view and the reference builder must agree exactly."""
    from ad_train import _to_device  # noqa: F401  (only to prove reuse is importable)
    ad = D.load_cache("valid")
    rows = np.array([0, 1, 2, 7], dtype=np.int64)
    fit = _fit()
    out = {}
    for cand, mk in (("QF", q_batch), ("FC", f_batch)):
        model, _ = M.build_model(cand, fit, 0, torch.device("cpu"), ssl_path=None)
        model.eval()
        coll = SD.collate_factory(cand, "valid")
        with torch.no_grad():
            p_cache = model(coll(None, rows, cand, valid=True))
            p_ref = model(mk([_one_ad(ad, int(r)) for r in rows]))
        out[cand] = {"cache_vs_reference_max_abs_diff": float((p_cache - p_ref).abs().max())}
    # counts are preserved exactly: atom groups sum to n, bond groups to m
    view = SD.load_q_view("valid")
    atom_ptr = np.asarray(ad["atom_ptr"])
    bond_ptr = np.asarray(ad["bond_ptr"])
    n = np.diff(atom_ptr).astype(np.float64)
    m = np.diff(bond_ptr).astype(np.float64)
    cnt = np.asarray(view["root_grp_count"])
    out["count_identity"] = {
        "atom_group_sum_equals_n": bool(np.allclose(cnt[:, :3].sum(1), np.repeat(n, n.astype(int)))),
        "bond_group_sum_equals_m": bool(np.allclose(cnt[:, 3:].sum(1), np.repeat(m, n.astype(int)))),
        "min_group_count": float(cnt.min()),
    }
    # the collated batch carries no label-derived tensor
    batch = SD.collate_factory("QF", "valid")(None, rows, "QF", valid=True)
    out["batch_keys"] = sorted(batch.keys())
    out["no_label_key"] = not any("label" in k or k == "y" for k in batch)
    RESULTS["cache_collate"] = out
    return out


def main() -> int:
    torch.set_num_threads(4)
    measure_matches_reference()
    forest_identity_and_reference()
    invariance_and_batching()
    finite_difference()
    parameter_and_optimizer_audit()
    no_bn_and_checkpoint_reload()
    edge_cases()
    cache_and_collate_consistency()
    payload = json.dumps(RESULTS, indent=2, default=float)
    print(payload)
    C.write_json(_HERE.parent / "analysis" / "cpu_checks.json", RESULTS)
    checks = [
        RESULTS["q_measure_vs_bruteforce_max_abs_error"] < 1e-12,
        RESULTS["f_forest_identity_max_abs_error"] < 1e-9,
        RESULTS["f_enumeration"]["root_max_abs_error"] < 1e-10,
        RESULTS["f_enumeration"]["edge_max_abs_error"] < 1e-10,
        all(v["parameters"] <= 500_000 for v in RESULTS["parameter_audit"].values()),
        all(v["atom_bond_permutation_max_abs_diff"] < 1e-9 for v in RESULTS["invariances"].values()),
        all(v["batch_recomposition_max_abs_diff"] < 1e-9 for v in RESULTS["invariances"].values()),
        all(v["train_eval_max_abs_diff"] < 1e-9 for v in RESULTS["invariances"].values()),
        all(e < 1e-4 for d in RESULTS["finite_difference_relative_error"].values() for e in d.values()),
        all(not v["has_batchnorm"] and v["reload_max_abs_diff"] == 0.0 for v in RESULTS["reload"].values()),
        RESULTS["edge_cases"]["q_tiny_finite"] and RESULTS["edge_cases"]["f_isolated_atom_finite"],
        all(v["cache_vs_reference_max_abs_diff"] < 1e-9
            for k in ("QF", "FC") for v in [RESULTS["cache_collate"][k]]),
        RESULTS["cache_collate"]["count_identity"]["atom_group_sum_equals_n"],
        RESULTS["cache_collate"]["count_identity"]["bond_group_sum_equals_m"],
        RESULTS["cache_collate"]["no_label_key"],
    ]
    print("\nALL CHECKS PASS" if all(checks) else f"\nCHECKS FAILED: {checks}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
