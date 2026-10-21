"""CPU execution/degeneracy checks for zinc-arch-exploration-v3.

All checks are label-free except the optional tiny overfit probe, which only
uses train rows (an execution check, never a generalization claim).  The
official test split is not touched here.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SUF = _EXP.parent
_V1 = _SUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_SUF.parents[1] / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import bjoc_masked_pretraining_v1 as MP  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import zae_data as ZD  # noqa: E402
import zae_models as ZM  # noqa: E402
import zae_ssl as ZS  # noqa: E402

OUT = _EXP / "analysis"


def _mol_from_raw(x, ei, ea, idx=0) -> tuple[Any, dict]:
    obj = sro.build_mol_objects(idx, np.asarray(x).reshape(-1), np.asarray(ei), np.asarray(ea).reshape(-1), "")
    atom_type = np.argmax(np.asarray(obj.atom_feat)[:, :28], axis=1).astype(np.int64)
    bond_type = np.asarray(obj.bond_type, dtype=np.int64)
    cache = ZD.pack_objects([obj], atom_type, bond_type)
    pair = ZD.build_pair_mol(int(obj.n_atoms), np.asarray(obj.bond_index, dtype=np.int64), bond_type)
    pair.update({"split": "single", "n_molecules": 1, "n_atoms_graph": np.asarray([obj.n_atoms], np.int16),
                 "pair_ptr": np.asarray([0, obj.n_atoms ** 2], np.int64), "meta": {"feature_version": "single"}})
    return obj, {"cache": cache, "pair": pair}


def check_params(norm) -> dict:
    out = {}
    for arm in ("A", "B"):
        model, info = ZM.build_model(arm, norm, 0, torch.device("cpu"))
        out[arm] = {
            "parameters": info["parameter_count"],
            "under_cap": bool(info["parameter_count"] <= 500_000),
            "headroom": int(500_000 - info["parameter_count"]),
            "initial_state_sha256": info["initial_state_sha256"],
        }
        if arm == "B":
            out[arm]["invariant_dim"] = int(model.inv_dim)
            out[arm]["triple_terms"] = len(model.triples)
            out[arm]["kernel_pairs"] = len(model.kernel_pairs)
        assert out[arm]["under_cap"], out[arm]
    # A-SSL temporary heads
    heads = ZS.SSLHeads()
    out["A_SSL_heads"] = C.parameter_count(heads)
    return out


def check_diag_conventions(valid_cache, pair_valid) -> dict:
    rows = np.arange(20)
    b = ZD.collate_pair(valid_cache, pair_valid, rows)
    self_mask = b["pair_self"].numpy()
    i, j = b["pair_i"].numpy(), b["pair_j"].numpy()
    n = b["n_atoms"].numpy()
    off_counts = np.bincount(b["pair_graph"].numpy(), weights=1.0 - self_mask, minlength=20)
    self_counts = np.bincount(b["pair_graph"].numpy(), weights=self_mask, minlength=20)
    out = {
        "self_mask_matches_index_equality": bool(np.array_equal(self_mask.astype(bool), i == j)),
        "diagonal_count_equals_n_atoms": bool(np.array_equal(self_counts.astype(int), n.astype(int))),
        "offdiag_count_equals_n(n-1)": bool(np.array_equal(off_counts.astype(int), (n * (n - 1)).astype(int))),
        "dense_index_unique_per_pair": bool(len(np.unique(b["pair_dense"].numpy())) == b["pair_dense"].numel()),
        "pair_adj_matches_bond_table": bool(
            (b["pair_adj"].numpy() > 0).sum() == 2 * int(b["n_bonds"].sum())
        ),
        "pair_bond_in_range": bool(b["pair_bond"].max().item() < b["bond_feat"].shape[0]),
    }
    return out, b


def check_permutation(norm, n_mols: int = 3) -> dict:
    import ad_data as D

    ds = D.pyg_zinc("valid")
    out = {}
    for arm in ("A", "B"):
        model, _ = ZM.build_model(arm, norm, 0, torch.device("cpu"))
        model.eval()
        diffs = []
        sorted_u = []
        for k in range(n_mols):
            d = ds[k]
            x = d.x.reshape(-1).long().numpy()
            ei = d.edge_index.numpy()
            ea = d.edge_attr.reshape(-1).long().numpy()
            n = x.shape[0]
            rng = np.random.default_rng(1000 + k)
            perm = rng.permutation(n)                     # old -> new
            x_p = np.zeros_like(x)
            x_p[perm] = x
            ei_p = perm[ei]
            obj_a, tab_a = _mol_from_raw(x, ei, ea, 0)
            obj_b, tab_b = _mol_from_raw(x_p, ei_p, ea, 0)
            batch_a = ZD.collate_pair(tab_a["cache"], tab_a["pair"], [0])
            batch_b = ZD.collate_pair(tab_b["cache"], tab_b["pair"], [0])
            with torch.no_grad():
                ya = model(batch_a).item()
                yb = model(batch_b).item()
            diffs.append(abs(ya - yb))
            if arm == "A":
                with torch.no_grad():
                    enc_a = model.encode_pairs(batch_a)
                    enc_b = model.encode_pairs(batch_b)
                ua = np.sort(enc_a["U"].numpy(), axis=0)
                ub = np.sort(enc_b["U"].numpy(), axis=0)
                sorted_u.append(float(np.max(np.abs(ua - ub))))
        out[arm] = {
            "n_molecules": n_mols,
            "max_abs_prediction_diff": float(max(diffs)),
            "prediction_diff_each": [float(x) for x in diffs],
        }
        if arm == "A":
            out[arm]["max_abs_sorted_U_diff"] = float(max(sorted_u))
    return out


def check_batch_isolation(norm, valid_cache, pair_valid, n_mols: int = 4) -> dict:
    rows = np.arange(n_mols)
    out = {}
    for arm in ("A", "B"):
        model, _ = ZM.build_model(arm, norm, 0, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            alone = []
            for r in rows:
                b1 = ZD.collate_pair(valid_cache, pair_valid, [r])
                alone.append(float(model(b1).item()))
            b_all = ZD.collate_pair(valid_cache, pair_valid, rows)
            joint = model(b_all).numpy()
            # padded with a longer molecule to change Nmax
            order = np.concatenate([[int(np.argmax(np.asarray(valid_cache["n_atoms_graph"])))], rows])
            b_pad = ZD.collate_pair(valid_cache, pair_valid, order)
            padded = model(b_pad).numpy()[1:]
        out[arm] = {
            "max_abs_alone_vs_batch": float(np.max(np.abs(np.asarray(alone) - joint))),
            "max_abs_alone_vs_padded_batch": float(np.max(np.abs(np.asarray(alone) - padded))),
        }
    return out


def check_kernels(valid_cache, pair_valid) -> dict:
    rows = np.arange(8)
    b = ZD.collate_pair(valid_cache, pair_valid, rows)
    ops = ZM.build_kernels(b["adj_dense"], b["adj_typed"], b["node_mask"])
    mask = b["node_mask"].float()
    cached = ZM.build_kernels(b["adj_dense"], b["adj_typed"], b["node_mask"], b.get("kern_dense"))
    cache_diff = max(float((x - y).abs().max()) for x, y in zip(ops, cached))
    out = {"n_ops": len(ops), "names": list(ZM.KERNEL_NAMES), "per_op": {},
           "max_abs_diff_on_the_fly_vs_cached": cache_diff}
    for name, k in zip(ZM.KERNEL_NAMES, ops):
        pad_leak = float((k * (1.0 - mask.unsqueeze(2) * mask.unsqueeze(1))).abs().max())
        sym = float((k - k.transpose(1, 2)).abs().max())
        out["per_op"][name] = {
            "finite": bool(torch.isfinite(k).all()),
            "max_abs": float(k.abs().max()),
            "max_pad_leak": pad_leak,
            "max_symmetry_error": sym,
            "fro_norm_mean": float(k.flatten(1).norm(dim=1).mean()),
        }
    # heat kernels must be PSD with eigenvalues in [0,1]
    heat = ops[: len(ZM.HEAT_SCALES)]
    ev = torch.stack([torch.linalg.eigvalsh(k) for k in heat])
    out["heat_eigenvalue_range"] = [float(ev.min()), float(ev.max())]
    return out


def check_U_adds_information(norm, valid_cache, pair_valid) -> dict:
    """A: the contraction must be triadic, and must not reduce to X or to a moment."""
    model, _ = ZM.build_model("A", norm, 0, torch.device("cpu"))
    model.eval()
    rows = np.arange(24)
    b = ZD.collate_pair(valid_cache, pair_valid, rows)
    with torch.no_grad():
        a, bb, w = model.encoders(b)
        X, _g = model.pair_features(b, a, bb, w)
        U = model.compose(X, a, b)
        # (1) variance across pairs: not a constant / not a single global value
        u_std = float(U.std(dim=0).mean())
        # (2) linear predictability of U from X (ridge least squares on packed pairs)
        Xn, Un = X.numpy(), U.numpy()
        A = np.concatenate([Xn, np.ones((Xn.shape[0], 1), dtype=np.float32)], axis=1)
        coef, *_ = np.linalg.lstsq(A, Un, rcond=None)
        resid = Un - A @ coef
        r2 = 1.0 - (resid ** 2).sum() / max(((Un - Un.mean(0)) ** 2).sum(), 1e-12)
        # (3) intermediate-atom dependence: perturbing atom k's content changes
        #     U_ij for pairs that do not touch k at all
        k = 3
        a2 = a.clone()
        a2[k] = a2[k] + 1.0
        U2 = model.compose(X, a2, b)
        pi, pj = b["pair_i"], b["pair_j"]
        untouched = (pi != k) & (pj != k)
        u_delta = (U2 - U).abs()
        # (4) locality of the contraction: perturbing the pair (i,p) row of X
        #     changes U_ij for all j, while perturbing an unrelated pair (p,q)
        #     with p,q not in {i,j} leaves U_ij exactly unchanged
        i0 = int(pi[0].item())
        cand = ((pi == i0) & (pj != i0)).nonzero().reshape(-1)
        p_atom = int(pj[cand[0]].item())
        Xp = X.clone()
        row = (pi == i0) & (pj == p_atom)
        Xp[row] = Xp[row] + 0.25
        Up = model.compose(Xp, a, b)
        to_j = pj != i0
        jj = int(pj[to_j][0].item())
        q_atom = None
        for cand_atom in range(int(b["n_atoms"].max().item())):
            if cand_atom not in (i0, jj):
                q_atom = cand_atom
                break
        Xq = X.clone()
        row2 = (pi == p_atom) & (pj == q_atom)
        Xq[row2] = Xq[row2] + 0.25
        Uq = model.compose(Xq, a, b)
        target = (pi == i0) & (pj == jj)
        out = {
            "U_mean_std_across_pairs": u_std,
            "linear_R2_of_U_from_X": float(r2),
            "residual_energy_fraction": float(1.0 - r2),
            "U_change_when_unrelated_atom_content_changes": float(u_delta[untouched].max()),
            "U_relative_change_unrelated_atom": float(
                u_delta[untouched].mean() / U[untouched].abs().mean().clamp(min=1e-9)
            ),
            "U_change_for_pair_with_perturbed_left_index": float((Up - U)[target].abs().max()),
            "U_change_for_unrelated_pair_row": float((Uq - U)[target].abs().max()),
        }
    assert out["residual_energy_fraction"] > 0.1, out
    assert out["U_change_when_unrelated_atom_content_changes"] > 0, out
    assert out["U_change_for_pair_with_perturbed_left_index"] > 0, out
    assert out["U_change_for_unrelated_pair_row"] == 0.0, out
    return out


def _triple_invariant_vector(model: ZM.SpectralJoint, batch) -> np.ndarray:
    """Triple block only, recomputed by running the model's invariant branch."""
    with torch.no_grad():
        a, b, w = model.encoders(batch)
        s = model.endpoint(a, w, batch)
        B, D = int(batch["n_graphs"]), int(batch["Nmax"])
        R = model.n_chan
        ops = ZM.build_kernels(batch["adj_dense"], batch["adj_typed"], batch["node_mask"])
        K = len(ops)
        z0 = torch.zeros((B * D, R))
        U = z0.index_copy(0, batch["atom_slot"], torch.tanh(model.ch_u(s))).view(B, D, R)
        V = z0.index_copy(0, batch["atom_slot"], torch.tanh(model.ch_v(s))).view(B, D, R)
        Wc = z0.index_copy(0, batch["atom_slot"], torch.tanh(model.ch_w(s))).view(B, D, R)
        n_atoms = batch["n_atoms"].unsqueeze(1)
        fro = torch.stack([ops[k].flatten(1).norm(dim=1) for k in range(K)], dim=1)
        pair_vals = []
        for (t, s_i) in model.kernel_pairs:
            o = ops[t] * ops[s_i].transpose(1, 2)
            pv = torch.einsum("bij,bir,bjr->br", o, U, V)
            pair_vals.append(pv / (fro[:, t] * fro[:, s_i]).unsqueeze(1).clamp(min=1e-6) / n_atoms)
        pair_vec = torch.cat(pair_vals, dim=1)
        m_all = []
        for t in range(K):
            scaled = ops[t].unsqueeze(1) * V.transpose(1, 2).unsqueeze(2)
            for s_i in range(K):
                m_all.append(torch.matmul(scaled, ops[s_i].unsqueeze(1)))
        m_stack = torch.stack(m_all, dim=2)
        outer = (U.unsqueeze(2) * Wc.unsqueeze(1)).permute(0, 3, 1, 2)
        g_stack = torch.stack([ops[q].transpose(1, 2) for q in range(K)], dim=1).unsqueeze(1)
        g_stack = outer.unsqueeze(2) * g_stack
        tri_all = torch.einsum("brpik,brqik->brpq", m_stack, g_stack)
        tri_pairs = torch.as_tensor([t * K + s_i for (t, s_i, _q) in model.triples], dtype=torch.int64)
        tri_q = torch.as_tensor([q for (_t, _s, q) in model.triples], dtype=torch.int64)
        tri_t = torch.as_tensor([t for (t, _s, _q) in model.triples], dtype=torch.int64)
        tri_s = torch.as_tensor([s_i for (_t, s_i, _q) in model.triples], dtype=torch.int64)
        tri = tri_all.index_select(2, tri_pairs).gather(
            3, tri_q.view(1, 1, -1, 1).expand(B, R, tri_q.numel(), 1)
        ).squeeze(-1)
        scale = (fro[:, tri_t] * fro[:, tri_s] * fro[:, tri_q]).clamp(min=1e-6) * n_atoms
        tri = tri / scale.unsqueeze(1)
        sums = torch.cat([U.sum(1), V.sum(1), Wc.sum(1), (U * V * Wc).sum(1)], dim=1)
    return tri.reshape(B, -1).numpy(), pair_vec.numpy(), sums.numpy()


def check_triple_non_degenerate(norm, valid_cache, pair_valid, n_mols: int = 512) -> dict:
    model, _ = ZM.build_model("B", norm, 0, torch.device("cpu"))
    model.eval()
    rows = np.arange(n_mols)
    b = ZD.collate_pair(valid_cache, pair_valid, rows)
    tri, pair, sums = _triple_invariant_vector(model, b)
    R = model.n_chan
    tri3 = tri.reshape(n_mols, R, len(model.triples))

    def standardized(A_):
        return (A_ - A_.mean(0)) / np.maximum(A_.std(0), 1e-8)

    def r2_of(Y):
        Ys = standardized(Y)
        A_ = standardized(np.concatenate([pair, sums], axis=1))
        coef, *_ = np.linalg.lstsq(A_, Ys, rcond=None)
        res = Ys - A_ @ coef
        return float(1.0 - (res ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum())

    groups: dict[str, list[int]] = {}
    Hh = len(ZM.HEAT_SCALES)
    for idx, (t, s_i, q) in enumerate(model.triples):
        if max(t, s_i, q) < Hh:
            g = "HHH"
        elif t >= Hh and s_i < Hh and q < Hh:
            g = "THH"
        elif t >= Hh and s_i >= Hh and q < Hh:
            g = "TTH"
        else:
            g = "TTT"
        groups.setdefault(g, []).append(idx)
    per_group = {}
    for g, idx in groups.items():
        Y = tri3[:, :, idx].reshape(n_mols, -1)
        per_group[g] = {
            "dim": int(len(idx) * R),
            "R2_from_pair_plus_first_order": r2_of(Y),
            "residual_energy_fraction": float(1.0 - r2_of(Y)),
            "mean_abs": float(np.abs(Y).mean()),
        }
    # typed-operator sensitivity: flip every double bond of one molecule to
    # single in the kernel adjacency only (channel content kept fixed)
    single = ZD.collate_pair(valid_cache, pair_valid, [0])
    bt = single["bond_type"]
    dbl = (bt == 2).nonzero().reshape(-1)
    tri_0, _p0, _s0 = _triple_invariant_vector(model, single)
    typed_sensitivity = None
    if dbl.numel():
        flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in single.items()}
        bt2 = bt.clone()
        bt2[dbl] = 1
        D_ = int(single["Nmax"])
        ij = single["bond_dense_ij"]
        gidx = ij // (D_ * D_)
        rem = ij % (D_ * D_)
        ii, jj = rem // D_, rem % D_
        adj_typed = torch.zeros_like(single["adj_typed"])
        ok = bt2 >= 1
        adj_typed[gidx[ok], ii[ok], jj[ok], bt2[ok] - 1] = 1.0
        adj_typed[gidx[ok], jj[ok], ii[ok], bt2[ok] - 1] = 1.0
        flip["adj_typed"] = adj_typed
        flip["adj_dense"] = adj_typed.sum(-1)
        tri_f, _pf, _sf = _triple_invariant_vector(model, flip)
        typed_sensitivity = float(np.abs(tri_f - tri_0).max())
    out = {
        "n_molecules": n_mols,
        "triple_dim": int(tri.shape[1]),
        "triple_std_mean": float(np.abs(tri).std(axis=0).mean()),
        "R2_all_triples_from_pair_plus_first_order": r2_of(tri),
        "residual_energy_fraction_all_triples": float(1.0 - r2_of(tri)),
        "per_group": per_group,
        "pair_dim": int(pair.shape[1]),
        "first_order_dim": int(sums.shape[1]),
        "triple_change_when_bond_type_flipped": typed_sensitivity,
        "triple_repeat_deterministic": float(np.abs(
            tri - _triple_invariant_vector(model, b)[0]).max()),
    }
    assert out["residual_energy_fraction_all_triples"] > 1e-3, out
    assert typed_sensitivity is None or typed_sensitivity > 0, "typed operators are inert"
    assert out["triple_std_mean"] > 0, "triple block is constant"
    return out


def check_ssl_mask() -> dict:
    """Mask consistency / no answer channel for the A-SSL input path."""
    ds = D.pyg_zinc("train")
    out = {"per_molecule": []}
    keys = ("atom_feat", "bond_feat", "junc_feat", "global", "pair_dist", "pair_common",
            "pair_ring", "pair_block", "pair_adj", "pair_bond", "adj_dense", "adj_typed", "atom_type")
    for k in range(3):
        d = ds[k]
        x = d.x.reshape(-1).long().numpy()
        ei = d.edge_index.numpy()
        ea = d.edge_attr.reshape(-1).long().numpy()
        obj, _tab = _mol_from_raw(x, ei, ea, 0)
        bond_index = np.asarray(obj.bond_index, np.int64)
        atom_ids, bond_ids = MP.sample_mask(0, k, 0, obj.n_atoms, bond_index.shape[0])
        a0 = int(atom_ids[0])
        alt_type = (int(x[a0]) + 1) % 21

        def masked_inputs(x_base, ea_base):
            x_m, ea_m = MP.mask_raw_arrays(x_base, ei, ea_base, bond_index, atom_ids, bond_ids)
            _o, tab = _mol_from_raw(x_m, ei, ea_m, 0)
            return ZD.collate_pair(tab["cache"], tab["pair"], [0])

        b_same = masked_inputs(x, ea)
        # same mask, different *hidden* atom class and different *hidden* masked
        # bond classes: every input tensor must be byte-identical
        x_b = x.copy()
        x_b[a0] = alt_type
        ea_b = ea.copy()
        mapping = MP.directed_bond_index(ei)
        for bid in bond_ids:
            u, v = int(bond_index[bid][0]), int(bond_index[bid][1])
            cols = mapping[(min(u, v), max(u, v))]
            for col in cols:
                ea_b[col] = int(ea_b[col]) % 3 + 1
        b_alt = masked_inputs(x_b, ea_b)
        max_diff = max(float((b_same[k_].float() - b_alt[k_].float()).abs().max()) for k_ in keys)
        _o_un, tab_un = _mol_from_raw(x, ei, ea, 0)
        b_un = ZD.collate_pair(tab_un["cache"], tab_un["pair"], [0])
        changed = float((b_same["atom_feat"] - b_un["atom_feat"]).abs().max())
        bond_changed = float((b_same["bond_feat"] - b_un["bond_feat"]).abs().max())
        # neighbour histograms must no longer count the hidden atom type
        blocks = sro.atom_blocks()
        nbidx = blocks["neighbour_types"]
        t_true = int(x[a0])
        nbrs = [int(v) if int(u) == a0 else int(u) for u, v in bond_index
                if int(u) == a0 or int(v) == a0]
        feat_m = np.asarray(b_same["atom_feat"])
        feat_u = np.asarray(b_un["atom_feat"])
        hist_delta = []
        hist_ok = True
        for j in nbrs:
            before = float(feat_u[j, nbidx][t_true])
            after = float(feat_m[j, nbidx][t_true])
            hist_delta.append(before - after)
            if abs(before - after - 1.0) > 1e-6:
                hist_ok = False
        # masked bond must not appear as a *typed* edge in the kernel adjacency
        typed_masked_edge = float(b_same["adj_typed"].sum())
        typed_unmasked_edge = float(b_un["adj_typed"].sum())
        out["per_molecule"].append({
            "mol": k, "masked_atoms": int(len(atom_ids)), "masked_bonds": int(len(bond_ids)),
            "masked_atom_ids": atom_ids.tolist(), "masked_bond_ids": bond_ids.tolist(),
            "inputs_identical_under_alternative_hidden_types": max_diff,
            "masked_inputs_differ_from_unmasked_atom": changed,
            "masked_inputs_differ_from_unmasked_bond": bond_changed,
            "neighbour_histogram_drops_hidden_type_by_exactly_one": hist_ok,
            "neighbour_histogram_deltas": hist_delta,
            "typed_edges_masked_vs_unmasked": [typed_masked_edge, typed_unmasked_edge],
        })
        assert max_diff < 1e-6, ("hidden type leaks into the masked input", out)
        assert changed > 0, "masking did not change any atom feature"
        assert bond_changed > 0, "masking did not change any bond feature"
        assert hist_ok, "neighbour histogram still counts the hidden atom type"
        assert typed_masked_edge < typed_unmasked_edge, "masked bond still acts as a typed edge"
    return out


def check_collate_determinism(valid_cache, pair_valid) -> dict:
    rows = np.arange(7)
    b1 = ZD.collate_pair(valid_cache, pair_valid, rows)
    b2 = ZD.collate_pair(valid_cache, pair_valid, rows)
    same = all(torch.equal(b1[k], b2[k]) for k in b1 if torch.is_tensor(b1[k]))
    out = {"collate_deterministic": bool(same)}
    # pair cache integrity vs the frozen object bond table
    ptr = np.asarray(pair_valid["pair_ptr"])
    ok = True
    for r in range(5):
        n = int(np.asarray(pair_valid["n_atoms_graph"])[r])
        if int(ptr[r + 1] - ptr[r]) != n * n:
            ok = False
    out["pair_cache_sizes_match_n_squared"] = bool(ok)
    return out


def check_grad_flow(norm, valid_cache, pair_valid) -> dict:
    rows = np.arange(32)
    b = ZD.collate_pair(valid_cache, pair_valid, rows)
    batch = {k: v for k, v in b.items()}
    out = {}
    for arm in ("A", "B"):
        model, _ = ZM.build_model(arm, norm, 0, torch.device("cpu"))
        y = model(batch)
        loss = y.abs().mean()
        loss.backward()
        grad_by_name = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                grad_by_name[name] = 0.0
            else:
                grad_by_name[name] = float(p.grad.abs().sum())
        zeros = [k for k, v in grad_by_name.items() if v == 0.0]
        out[arm] = {
            "loss_finite": bool(torch.isfinite(loss)),
            "n_params": len(grad_by_name),
            "params_with_zero_grad": zeros,
            "all_finite": bool(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)),
        }
    return out


def check_source_gradient_paths(norm, valid_cache, pair_valid) -> dict:
    """The contraction parameters must receive gradient not only through X."""
    b = ZD.collate_pair(valid_cache, pair_valid, np.arange(32))
    model, _ = ZM.build_model("A", norm, 0, torch.device("cpu"))
    y = model(b).abs().mean()
    y.backward()
    names = {k: float(v.grad.abs().sum()) for k, v in model.named_parameters() if v.grad is not None}
    out = {k: names.get(k, None) for k in
           ("phi.weight", "psi.weight", "chi.weight", "h_mlp.lins.0.weight", "x_mlp.lins.0.weight",
            "endpoint.mlp.lins.0.weight", "reader.lins.0.weight", "encoders.atom_encoder.lins.0.weight")}
    out["all_nonzero"] = bool(all(v is not None and v > 0 for v in out.values() if isinstance(v, float)))
    return out


def check_recipe_arithmetic() -> dict:
    import zae_train as ZT

    n_train, batch = 10000, int(ZT.RECIPE["batch_size"])
    steps = math.ceil(n_train / batch)
    return {
        "batch_size": batch,
        "steps_per_epoch": steps,
        "planned_epochs": int(ZT.RECIPE["epochs"]),
        "planned_updates": steps * int(ZT.RECIPE["epochs"]),
        "last_partial_batch_rows": n_train - (steps - 1) * batch,
        "warmup_steps_declared": int(ZT.RECIPE["warmup_steps"]),
        "warmup_effective": min(int(ZT.RECIPE["warmup_steps"]), max(1, steps * int(ZT.RECIPE["epochs"]) // 20)),
        "ssl": {k: ZT.SSL_RECIPE[k] for k in ("epochs", "batch_size", "lr", "warmup_steps")},
    }


def overfit_probe(norm, train_cache, pair_train, *, steps: int = 80, batch: int = 8) -> dict:
    """Tiny supervised overfit on train rows: an execution probe, not evidence."""
    rows = np.arange(32)
    y = torch.as_tensor(np.asarray(train_cache["labels"])[rows], dtype=torch.float32)
    out = {}
    for arm in ("A", "B"):
        model, _ = ZM.build_model(arm, norm, 0, torch.device("cpu"))
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        losses = []
        t0 = time.perf_counter()
        for step in range(steps):
            idx = rows[:batch]
            b = ZD.collate_pair(train_cache, pair_train, idx)
            pred = model(b)
            loss = F.l1_loss(pred, y[:batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        out[arm] = {
            "rows": int(len(rows)),
            "steps": steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "min_loss": float(min(losses)),
            "seconds": float(time.perf_counter() - t0),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)
    norm = D.load_norm()
    valid_cache = D.load_cache("valid")
    pair_valid = ZD.load_pair_split("valid")
    payload: dict[str, Any] = {"round": "zinc-arch-exploration-v3", "official_test_loaded": False}
    payload["params"] = check_params(norm)
    diag, _b = check_diag_conventions(valid_cache, pair_valid)
    payload["diag_conventions"] = diag
    assert all(diag.values())
    payload["permutation_invariance"] = check_permutation(norm)
    payload["batch_isolation"] = check_batch_isolation(norm, valid_cache, pair_valid)
    payload["kernels"] = check_kernels(valid_cache, pair_valid)
    payload["a_contraction"] = check_U_adds_information(norm, valid_cache, pair_valid)
    payload["b_triple_block"] = check_triple_non_degenerate(norm, valid_cache, pair_valid)
    payload["ssl_mask"] = check_ssl_mask()
    payload["collate"] = check_collate_determinism(valid_cache, pair_valid)
    payload["grad_flow"] = check_grad_flow(norm, valid_cache, pair_valid)
    payload["a_source_paths"] = check_source_gradient_paths(norm, valid_cache, pair_valid)
    payload["recipe"] = check_recipe_arithmetic()
    train_cache = D.load_cache("train")
    pair_train = ZD.load_pair_split("train")
    payload["overfit_probe"] = overfit_probe(norm, train_cache, pair_train)
    C.write_json(OUT / "cpu_checks.json", payload)
    C.log(json.dumps({k: v for k, v in payload.items() if k not in ("kernels",)}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
