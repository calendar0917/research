"""Pre-training engineering checks for WOBR-v0.

Writes ``analysis/checks.json`` and ``analysis/candidate_coverage.json``.
Every check is a direct computation on the frozen cache / model: candidates are
re-derived, sparsemax is compared to independent references, contributions are
re-summed, and synthetic same-count/different-connection graphs are contrasted.
"""

from __future__ import annotations

import copy
import json
import math
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import wobr_canonical as CAN
import wobr_common as C
import wobr_data as D
import wobr_model as M
import wobr_sparsemax as SP
import wobr_structure as S
import wobr_synthetic as SYN
import wobr_tensor as T


def _norm(bundle: D.DataBundle) -> dict[str, np.ndarray]:
    return D.normalisers(bundle)


def _cfg() -> dict[str, Any]:
    return C.load_config()


def _single_batch(view: Mapping[str, Any], device: Any = "cpu"):
    view = dict(view)
    view["atom_scalars"] = D.atom_scalars(view)
    raw = SYN.raw_from_views([view])
    st = S.build_molecule_struct(view)
    return T.collate([0], {0: st}, raw, device), st, raw


def _double_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.double() if torch.is_tensor(v) and v.dtype.is_floating_point else v)
            for k, v in batch.items()}


def _build_zeroed(bundle: D.DataBundle, seed: int = 0, double: bool = False):
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, seed,
                                _norm(bundle), config=_cfg())
    atom_ok, bond_ok = D.category_support(bundle.raw, bundle.train_idx,
                                          bundle.atom_dims, bundle.bond_dims)
    zeroed = model.zero_unseen(atom_ok, bond_ok, bundle.atom_dims, bundle.bond_dims)
    if double:
        model.double()
    model.eval()
    return model, info, zeroed, (atom_ok, bond_ok)


def _rec(checks: list[dict[str, Any]], ident: str, name: str, status: bool,
         details: Any) -> None:
    checks.append({"id": ident, "name": name,
                   "status": "PASS" if status else "FAIL",
                   "details": C.jsonable(details)})


def _bfs_components(n: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    adj = [[] for _ in range(n)]
    for u, v in zip(bu.tolist(), bv.tolist()):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    comp = np.full(n, -1, np.int64)
    cid = 0
    for start in range(n):
        if comp[start] >= 0:
            continue
        stack = [start]
        comp[start] = cid
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if comp[y] < 0:
                    comp[y] = cid
                    stack.append(y)
        cid += 1
    return comp


def _candidate_molecule_audit(st: S.MoleculeStruct) -> dict[str, int]:
    W = int(st.n_words)
    support = np.asarray(st.word_support, np.int64)
    internal = np.asarray(st.word_internal, np.int64)
    sizes = np.asarray(st.word_size, np.int64)
    roots = np.asarray(st.word_root, np.int64)
    bu = np.asarray(st.bond_u, np.int64)
    bv = np.asarray(st.bond_v, np.int64)
    m = int(st.n_bonds)
    bad = {"bad_root": 0, "bad_size": 0, "bad_internal": 0, "bad_connected": 0,
           "bad_boundary_count": 0, "bad_boundary_types": 0, "bad_singleton": 0,
           "bad_unique": 0, "bad_scalars": 0}
    if W == 0:
        return bad
    valid = support >= 0
    safe = np.where(valid, support, 0)
    root_present = ((safe == roots[:, None]) & valid).any(axis=1)
    bad["bad_root"] += int(np.sum(~root_present))
    bad["bad_size"] += int(np.sum((sizes < 1) | (sizes > 4)))
    bad["bad_size"] += int(np.sum(valid.sum(axis=1) != sizes))

    # every internal bond is exactly an induced bond, and every induced bond
    # appears in the stored internal list.
    in_u = ((support[:, :, None] == bu[None, None, :]) & valid[:, :, None]).any(axis=1)
    in_v = ((support[:, :, None] == bv[None, None, :]) & valid[:, :, None]).any(axis=1)
    induced = in_u & in_v
    induced_count = induced.sum(axis=1)
    bad["bad_internal"] += int(np.sum(induced_count != np.asarray(st.word_internal_count)))

    # connectivity from the true root slot using the internal induced edges.
    safe_internal = np.where(internal >= 0, internal, 0)
    bu_e = bu[safe_internal]
    bv_e = bv[safe_internal]
    eq_u = support[:, None, :] == bu_e[:, :, None]
    eq_v = support[:, None, :] == bv_e[:, :, None]
    slot_u = eq_u.argmax(axis=2)
    slot_v = eq_v.argmax(axis=2)
    valid_edge = internal >= 0
    bad["bad_internal"] += int(np.sum(valid_edge & ~(eq_u.any(axis=2) & eq_v.any(axis=2))))
    root_slot = (support == roots[:, None]).argmax(axis=1)
    reach = np.left_shift(1, root_slot.astype(np.int64))
    # up to three relaxation rounds are enough for |S| <= 4
    for _ in range(3):
        for term in range(internal.shape[1]):
            eu = slot_u[:, term]
            ev = slot_v[:, term]
            hu = (reach >> eu) & 1
            hv = (reach >> ev) & 1
            add = (hu << ev) | (hv << eu)
            reach = reach | np.where(valid_edge[:, term], add, 0)
    target = np.left_shift(1, sizes.astype(np.int64)) - 1
    bad["bad_connected"] += int(np.sum(reach != target))

    # root degree, size, internal count and cycle rank in the 25 scalars.
    scal = np.asarray(st.word_scalars, np.float64)
    root_deg = np.sum(valid_edge & ((bu_e == roots[:, None]) | (bv_e == roots[:, None])),
                      axis=1)
    expected_scal = np.zeros((W, 25), np.float64)
    expected_scal[:, 0] = sizes / 4.0
    expected_scal[:, 1] = st.word_internal_count / 6.0
    expected_scal[:, 2] = (st.word_internal_count - sizes + 1) / 3.0
    expected_scal[:, 3] = root_deg / 3.0
    scalar_err = np.maximum(np.abs(scal[:, :4] - expected_scal[:, :4]).max(axis=1)
                            if W else np.zeros(0), 0.0)
    bad["bad_scalars"] += int(np.sum(scalar_err > 1e-6))

    # singleton exactly once per root, all supports unique within a root, and
    # the exact boundary count / categorical type fractions for every word.
    ptr = np.asarray(st.word_root_ptr, np.int64)
    bf = np.asarray(st.bond_feat, np.int64)
    degree = np.asarray(st.degree, np.int64)
    for v in range(int(st.n_atoms)):
        lo, hi = int(ptr[v]), int(ptr[v + 1])
        if hi <= lo:
            bad["bad_singleton"] += 1
            continue
        sup = support[lo:hi]
        sz = sizes[lo:hi]
        singles = int(np.sum((sz == 1) & (sup[:, 0] == v)))
        if singles != 1:
            bad["bad_singleton"] += 1
        tuples = {tuple(int(x) for x in row if int(x) >= 0) for row in sup.tolist()}
        if len(tuples) != hi - lo:
            bad["bad_unique"] += 1
        r2 = np.zeros(int(st.n_atoms), bool)
        r2[v] = True
        a1 = st.dist1_atoms[st.dist1_ptr[v]:st.dist1_ptr[v + 1]]
        a2 = st.dist2_atoms[st.dist2_ptr[v]:st.dist2_ptr[v + 1]]
        r2[a1] = True
        r2[a2] = True
        valid_sup = sup >= 0
        in_s_u = ((sup[:, :, None] == bu[None, None, :]) & valid_sup[:, :, None]).any(axis=1)
        in_s_v = ((sup[:, :, None] == bv[None, None, :]) & valid_sup[:, :, None]).any(axis=1)
        boundary = (in_s_u != in_s_v) & r2[bu][None, :] & r2[bv][None, :]
        bcount = boundary.sum(axis=1)
        if np.any(bcount != np.asarray(st.word_boundary_count[lo:hi])):
            bad["bad_boundary_count"] += 1
        type_counts = np.zeros((hi - lo, 13), np.float64)
        off = 0
        for col, width in enumerate(S.BOUNDARY_FIELDS):
            for cat in range(width):
                type_counts[:, off + cat] = np.sum(
                    boundary & (bf[:, col] == cat)[None, :], axis=1)
            off += width
        frac = type_counts / np.maximum(1, bcount)[:, None]
        stored = scal[lo:hi, 12:25]
        if np.max(np.abs(frac - stored)) > 1e-6:
            bad["bad_boundary_types"] += 1
        safe_sup = np.where(sup >= 0, sup, 0)
        total_deg = (degree[safe_sup] * valid_sup).sum(axis=1)
        ratio = bcount / np.maximum(1.0, total_deg)
        if np.max(np.abs(ratio - scal[lo:hi, 10])) > 1e-6:
            bad["bad_scalars"] += 1
        if np.max(np.abs(np.log1p(bcount) - scal[lo:hi, 11])) > 1e-6:
            bad["bad_scalars"] += 1
    return bad


def main(max_rows: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    checks: list[dict[str, Any]] = []
    bundle = D.load_bundle()
    cfg = _cfg()
    atom_dims, bond_dims = bundle.atom_dims, bundle.bond_dims
    all_rows = np.union1d(np.union1d(bundle.train_idx, bundle.valid_idx), bundle.test_idx)
    if max_rows:
        all_rows = all_rows[:int(max_rows)]
    model, info, zeroed, (atom_ok, bond_ok) = _build_zeroed(bundle)
    model.eval()

    # Build the full corpus once; later checks also use it for candidate stats.
    t_build = time.time()
    full = S.build_all(bundle.raw, all_rows)
    build_seconds = time.time() - t_build
    rss_after_build = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    array_bytes = 0
    for st in full.values():
        for value in st.__dict__.values():
            if isinstance(value, np.ndarray):
                array_bytes += int(value.nbytes)

    # 1. candidates, singleton, induced connectivity, internal/boundary exactness
    attr_ok = True
    node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(bundle.raw["bond_ptr"], np.int64)
    raw_atom = np.asarray(bundle.raw["atom_feat"])
    raw_bond = np.asarray(bundle.raw["bond_feat"])
    for row in [int(x) for x in all_rows[:12]]:
        st = full[row]
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        attr_ok &= bool(np.array_equal(st.atom_feat, raw_atom[a0:a1])
                        and np.array_equal(st.bond_feat, raw_bond[b0:b1])
                        and np.array_equal(st.bond_u, np.asarray(bundle.raw["bond_u"])[b0:b1])
                        and np.array_equal(st.bond_v, np.asarray(bundle.raw["bond_v"])[b0:b1]))
    totals = {k: 0 for k in ("bad_root", "bad_size", "bad_internal", "bad_connected",
                             "bad_boundary_count", "bad_boundary_types", "bad_singleton",
                             "bad_unique", "bad_scalars")}
    n_words_checked = 0
    for row in [int(x) for x in all_rows]:
        st = full[row]
        bad = _candidate_molecule_audit(st)
        for k in totals:
            totals[k] += bad[k]
        n_words_checked += int(st.n_words)
    conn_ok = all(v == 0 for v in totals.values()) and attr_ok
    _rec(checks, "1_candidates_real_connected_singleton",
         "every candidate is a real connected induced support; every root has exactly one singleton",
         conn_ok, {"n_molecules": int(all_rows.size), "n_words_checked": int(n_words_checked),
                   **totals, "attributes_match_raw": bool(attr_ok),
                   "max_support_size": 4})

    # 2. atom renumbering, edge reversal, bond order and candidate order
    cand_rows = [int(r) for r in all_rows if full[int(r)].n_atoms >= 6
                 and full[int(r)].n_bonds >= 6]
    row = cand_rows[0]
    view = D.molecule_view(bundle.raw, row)
    view["atom_scalars"] = D.atom_scalars(view)
    b0, st0, raw0 = _single_batch(view)
    dmodel, _, _, _ = _build_zeroed(bundle, double=True)
    with torch.no_grad():
        y0 = float(dmodel(_double_batch(b0))["yhat_std"].item())
    rng = np.random.default_rng(7)
    perm = rng.permutation(int(view["n_atoms"]))
    border = rng.permutation(int(view["n_bonds"]))
    viewp = SYN.permute_view(view, perm, border, reverse_edges=True)
    bp, stp, rawp = _single_batch(viewp)
    with torch.no_grad():
        yp = float(dmodel(_double_batch(bp))["yhat_std"].item())
    st2 = copy.deepcopy(st0)
    for r in range(st2.n_atoms):
        lo, hi = int(st2.word_root_ptr[r]), int(st2.word_root_ptr[r + 1])
        if hi - lo > 1:
            idx = np.arange(lo, hi)[::-1]
            st2.word_support[lo:hi] = st2.word_support[idx]
            st2.word_internal[lo:hi] = st2.word_internal[idx]
            st2.word_scalars[lo:hi] = st2.word_scalars[idx]
            st2.word_size[lo:hi] = st2.word_size[idx]
            st2.word_internal_count[lo:hi] = st2.word_internal_count[idx]
            st2.word_boundary_count[lo:hi] = st2.word_boundary_count[idx]
    bc = T.collate([0], {0: st2}, raw0, "cpu")
    with torch.no_grad():
        yc = float(dmodel(_double_batch(bc))["yhat_std"].item())
    inv_err = max(abs(y0 - yp), abs(y0 - yc))
    _rec(checks, "2_renumber_edge_candidate_invariance",
         "atom renumbering, edge reversal, bond order and candidate order do not change prediction",
         inv_err < 1e-8, {"row": row, "y": y0, "y_permuted": yp,
                          "y_candidate_reordered": yc, "max_abs_error": inv_err,
                          "tolerance": 1e-8, "dtype": "float64"})

    # 3. sparsemax: nonnegative, normalised, exact zeros, padding zero, references
    rng = np.random.default_rng(11)
    score_np = rng.normal(size=(128, 24))
    count = rng.integers(1, 25, size=128)
    mask_np = np.arange(24)[None, :] < count[:, None]
    score_np[~mask_np] = -1e6
    score64 = torch.tensor(score_np, dtype=torch.float64, requires_grad=True)
    mask_t = torch.as_tensor(mask_np)
    alpha64 = SP.sparsemax_padded(score64, mask_t)
    ref_bisect = SP.sparsemax_reference(score_np, mask_np)
    ref_sorted = SP.sparsemax_reference_sorted(score_np, mask_np)
    sums = alpha64.sum(dim=1).detach().numpy()
    pad_max = float(alpha64[~mask_t].abs().max()) if bool((~mask_t).any()) else 0.0
    ref_err = max(float(np.max(np.abs(alpha64.detach().numpy() - ref_bisect))),
                  float(np.max(np.abs(alpha64.detach().numpy() - ref_sorted))))
    zeros = float((alpha64.detach().numpy()[mask_np] == 0).mean())
    grad_ok = True
    loss = (alpha64 * torch.linspace(1.0, 2.0, 24, dtype=torch.float64)).sum()
    loss.backward()
    grad_ok = bool(torch.isfinite(score64.grad).all() and score64.grad.abs().sum() > 0)
    # packed path against padded reference
    ptr = torch.as_tensor([0, 1, 3, 6, 10], dtype=torch.long)
    packed_scores = torch.as_tensor(rng.normal(size=10), dtype=torch.float64)
    packed_alpha = SP.sparsemax_packed(packed_scores, ptr, 4)
    packed_ref = np.concatenate([
        SP.sparsemax_reference(packed_scores[ptr[i]:ptr[i + 1]].numpy().astype(np.float64))
        for i in range(4)])
    packed_err = float(np.max(np.abs(packed_alpha.numpy() - packed_ref)))
    sparsemax_ok = bool(np.all(alpha64.detach().numpy() >= -1e-12)
                        and np.max(np.abs(sums - 1.0)) < 1e-9
                        and pad_max == 0.0 and ref_err < 1e-8
                        and np.all(alpha64.detach().numpy().sum(axis=1) > 0)
                        and zeros > 0.0 and grad_ok and packed_err < 1e-8)
    _rec(checks, "3_sparsemax_normalised_padding_reference",
         "sparsemax is nonnegative, sums to 1 per root, can be exactly zero, and matches two independent references",
         sparsemax_ok, {"rows": 128, "max_candidates": 24,
                        "min_alpha": float(alpha64.detach().min()),
                        "max_alpha": float(alpha64.detach().max()),
                        "max_sum_error": float(np.max(np.abs(sums - 1.0))),
                        "padding_max_abs": pad_max,
                        "exact_zero_fraction_real_entries": zeros,
                        "reference_max_abs_error": ref_err,
                        "packed_reference_max_abs_error": packed_err,
                        "score_gradient_finite_nonzero": grad_ok})

    # 4. same atom/bond counts, different connection -> different bilinear terms
    path = SYN.make_view("path4", 4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
                         atom_types=[5] * 4)
    star = SYN.make_view("star4", 4, [(0, 1, 1), (0, 2, 1), (0, 3, 1)],
                         atom_types=[5] * 4)
    bpath, _, _ = _single_batch(path)
    bstar, _, _ = _single_batch(star)
    with torch.no_grad():
        opath = dmodel(_double_batch(bpath), return_aux=True)
        ostar = dmodel(_double_batch(bstar), return_aux=True)
        qdiff = float((opath["Q"] - ostar["Q"]).abs().max())
        ydiff = float((opath["yhat_std"] - ostar["yhat_std"]).abs().max())
    _rec(checks, "4_synthetic_same_counts_different_connection",
         "same atom/bond counts with different real connection produce different adjacent bilinear terms",
         qdiff > 1e-10 and ydiff > 0.0,
         {"path_atoms": 4, "path_bonds": 3, "star_atoms": 4, "star_bonds": 3,
          "max_abs_Q_diff": qdiff, "max_abs_prediction_diff": ydiff})

    # 5. exact contribution identity and per-bond word-pair expansion
    syn = SYN.make_view("identity6", 6, [(0, 1, 1), (1, 2, 1), (2, 3, 1),
                                         (3, 4, 1), (4, 5, 1), (1, 4, 1)],
                        atom_types=[5] * 6)
    bsyn, _, _ = _single_batch(syn)
    bsyn = _double_batch(bsyn)
    with torch.no_grad():
        osyn = dmodel(bsyn, return_aux=True)
        csyn = dmodel.word_contributions(bsyn, osyn)
    target_syn = (osyn["yhat_std"] - dmodel.beta
                  - dmodel.c_n * torch.log1p(osyn["atom_count"])
                  - dmodel.c_m * torch.log1p(osyn["bond_count"]))
    sum_syn = torch.zeros(int(bsyn["n_molecules"]), dtype=torch.float64)
    sum_syn.index_add_(0, bsyn["word_mol"], csyn["c"])
    err_syn = float((sum_syn - target_syn).abs().max())

    # per-bond expansion on the synthetic graph and on a real Train batch
    def _expansion_error(batch: Mapping[str, Any], out: Mapping[str, Any],
                         model_: M.WobrModel) -> float:
        w = out["w"]
        alpha = out["alpha"]
        lam = out["lam"]
        Uw = model_.U(w)
        Uz = out["Uz"]
        idx = batch["bond_index"]
        ptr = batch["root_word_ptr"]
        err = 0.0
        for e in range(int(idx.shape[0])):
            u, v = int(idx[e, 0]), int(idx[e, 1])
            pu0, pu1 = int(ptr[u]), int(ptr[u + 1])
            pv0, pv1 = int(ptr[v]), int(ptr[v + 1])
            if pu1 == pu0 or pv1 == pv0:
                continue
            au = alpha[pu0:pu1]
            av = alpha[pv0:pv1]
            qpair = (lam[e][None, None, :] * Uw[pu0:pu1, None, :]
                     * Uw[None, pv0:pv1, :]).sum(dim=2) / math.sqrt(16)
            qexp = (au[:, None] * av[None, :] * qpair).sum()
            err = max(err, float((qexp - out["Q"][e]).abs()))
            # independent check: direct z_u, z_v
            direct = (lam[e] * Uz[u] * Uz[v]).sum() / math.sqrt(16)
            err = max(err, float((direct - out["Q"][e]).abs()))
        return err

    expand_syn = _expansion_error(bsyn, osyn, dmodel)
    real_rows = np.asarray(all_rows[:32], np.int64)
    breal = T.collate(real_rows.tolist(), full, bundle.raw, "cpu")
    model.float()
    model.eval()
    with torch.no_grad():
        oreal = model(breal, return_aux=True)
        creal = model.word_contributions(breal, oreal)
    target_real = (oreal["yhat_std"] - model.beta
                   - model.c_n * torch.log1p(oreal["atom_count"])
                   - model.c_m * torch.log1p(oreal["bond_count"]))
    sum_real = torch.zeros(int(breal["n_molecules"]), dtype=torch.float32)
    sum_real.index_add_(0, breal["word_mol"], creal["c"])
    err_real = float((sum_real - target_real).abs().max())
    expand_real = _expansion_error(breal, oreal, model)
    id_ok = err_syn < 1e-10 and err_real < 1e-4 and expand_syn < 1e-10 and expand_real < 1e-4
    _rec(checks, "5_exact_word_contribution_identity",
         "sum of word contributions equals prediction minus beta and count terms; word-pair bond expansion equals direct z_u,z_v bilinear term",
         id_ok, {"synthetic_float64_max_abs_error": err_syn,
                 "real_float32_max_abs_error": err_real,
                 "synthetic_expansion_max_abs_error": expand_syn,
                 "real_expansion_max_abs_error": expand_real,
                 "tolerances": {"float64": 1e-10, "float32": 1e-4},
                 "synthetic_atoms": 6, "synthetic_bonds": 6,
                 "real_batch_molecules": int(real_rows.size),
                 "real_batch_atoms": int(breal["n_atoms"]),
                 "real_batch_bonds": int(breal["n_bonds"])})

    # 6. single vs batch inference
    sample_rows = np.asarray(all_rows[:8], np.int64)
    bsample = T.collate(sample_rows.tolist(), full, bundle.raw, "cpu")
    with torch.no_grad():
        ybatch = model(bsample)["yhat_std"].numpy()
        ysingle = []
        for r in sample_rows.tolist():
            b = T.collate([r], full, bundle.raw, "cpu")
            ysingle.append(float(model(b)["yhat_std"].item()))
    bs_err = float(np.max(np.abs(ybatch - np.asarray(ysingle))))
    _rec(checks, "6_batch_single_consistency",
         "batched and single-molecule inference agree", bs_err < 1e-5,
         {"n_molecules": int(sample_rows.size), "max_abs_error": bs_err})

    # 7. finite non-zero gradients on a real batch
    grad_model, _, _, _ = _build_zeroed(bundle, seed=3)
    grad_model.train()
    grad_rows = np.asarray(all_rows[:32], np.int64)
    bgrad = T.collate(grad_rows.tolist(), full, bundle.raw, "cpu")
    gout = grad_model(bgrad)
    target = torch.as_tensor(bundle.labels_std()[grad_rows], dtype=torch.float32)
    loss = torch.mean((gout["yhat_std"] - target) ** 2)
    grad_model.zero_grad(set_to_none=True)
    loss.backward()
    groups = {
        "atom_emb": grad_model.atom_emb, "bond_raw_emb": grad_model.bond_raw_emb,
        "bond_encoder": grad_model.bond_encoder, "path_joint": grad_model.path_joint,
        "path_emb": grad_model.path_emb, "nbr_emb": grad_model.nbr_emb,
        "backbone_mlp": grad_model.backbone_mlp,
        "word_joint": grad_model.word_joint, "word_mlp": grad_model.word_mlp,
        "score_mlp": grad_model.score_mlp, "a": grad_model.a,
        "U": grad_model.U, "bond_interaction": grad_model.bond_interaction,
        "beta": grad_model.beta, "c_n": grad_model.c_n, "c_m": grad_model.c_m,
    }
    grad_details = {}
    grad_ok = True
    for name, module in groups.items():
        params = [module] if isinstance(module, torch.nn.Parameter) else list(module.parameters())
        vals = []
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                grad_ok = False
            vals.append(float(p.grad.pow(2).sum()))
        total = float(np.sum(vals)) if vals else 0.0
        grad_details[name] = total
        grad_ok &= total > 0.0
    all_params_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in grad_model.parameters())
    _rec(checks, "7_finite_nonzero_gradients",
         "all key modules, including word encoder, score, a, U and bond interaction, have finite non-zero gradients",
         bool(grad_ok and all_params_finite),
         {"gradient_squared_norms": grad_details,
          "all_parameter_gradients_finite": bool(all_params_finite)})

    # 8. reproducible initialisation and explicit initial values
    C.seed_everything(0)
    m1, i1 = M.build_model(bundle.atom_dims, bundle.bond_dims, 0, _norm(bundle), config=cfg)
    m1.zero_unseen(atom_ok, bond_ok, bundle.atom_dims, bundle.bond_dims)
    C.seed_everything(0)
    m2, i2 = M.build_model(bundle.atom_dims, bundle.bond_dims, 0, _norm(bundle), config=cfg)
    m2.zero_unseen(atom_ok, bond_ok, bundle.atom_dims, bundle.bond_dims)
    same_state = all(torch.equal(v, m2.state_dict()[k]) for k, v in m1.state_dict().items())
    fp1 = C.state_fingerprint(m1.state_dict())
    fp2 = C.state_fingerprint(m2.state_dict())
    init_ok = bool(same_state and fp1 == fp2 and i1["init"]["state_fingerprint"]
                   == i2["init"]["state_fingerprint"]
                   and abs(i1["init"]["a_std"] - 0.02) < 0.01
                   and i1["init"]["U_weight_absmax"] <= i1["init"]["U_xavier_bound"] + 1e-12
                   and i1["init"]["beta_c_n_c_m"] == [0.0, 0.0, 0.0])
    _rec(checks, "8_initialisation_reproducible",
         "the same seed reconstructs the same initial state and the frozen explicit initialisation values",
         init_ok, {"same_state_dict": bool(same_state),
                   "state_fingerprint": fp1, "second_fingerprint": fp2,
                   "before_support_fingerprint": i1["init"]["state_fingerprint"],
                   **{k: i1["init"][k] for k in ("a_mean", "a_std", "U_xavier_bound",
                                                 "U_weight_absmax",
                                                 "bond_interaction_last_weight_std",
                                                 "beta_c_n_c_m")}})

    # 9. data identity, Train-only scaler and metric restoration
    sha = C.sha256_file(C.RAW_CACHE)
    train_idx = np.asarray(bundle.train_idx, np.int64)
    valid_idx = np.asarray(bundle.valid_idx, np.int64)
    test_idx = np.asarray(bundle.test_idx, np.int64)
    sizes = [train_idx.size, valid_idx.size, test_idx.size]
    disjoint = (np.intersect1d(train_idx, valid_idx).size == 0
                and np.intersect1d(train_idx, test_idx).size == 0
                and np.intersect1d(valid_idx, test_idx).size == 0)
    cover = np.array_equal(np.union1d(np.union1d(train_idx, valid_idx), test_idx),
                           np.arange(4200))
    scaler = {"mean": float(bundle.labels[train_idx].mean()),
              "std": float(bundle.labels[train_idx].std(ddof=0))}
    scaler_ok = abs(scaler["mean"] - bundle.scaler["mean"]) < 1e-12 \
        and abs(scaler["std"] - bundle.scaler["std"]) < 1e-12 \
        and bundle.scaler["n"] == int(train_idx.size)
    all_scaler = {"mean": float(bundle.labels.mean()), "std": float(bundle.labels.std(ddof=0))}
    z_train = bundle.labels_std()[train_idx]
    z_ok = abs(float(z_train.mean())) < 1e-12 and abs(float(z_train.std(ddof=0)) - 1.0) < 1e-12
    pred_std = np.asarray([-1.3, 0.0, 2.4], np.float64)
    restored = pred_std * scaler["std"] + scaler["mean"]
    metric_ok = np.allclose(restored, pred_std * bundle.scaler["std"] + bundle.scaler["mean"])
    emb_ok = (len(bundle.atom_dims) == 9 and len(bundle.bond_dims) == 3
              and all(int(e.num_embeddings) == int(d) for e, d in
                      zip(model.atom_emb, bundle.atom_dims))
              and all(int(e.num_embeddings) == int(d) for e, d in
                      zip(model.bond_raw_emb, bundle.bond_dims)))
    zero_support_ok = True
    for field, emb in enumerate(model.atom_emb):
        mask = torch.as_tensor(atom_ok[field], dtype=torch.bool, device=emb.weight.device)
        zero_support_ok &= bool(torch.all(emb.weight[~mask] == 0))
    for field, emb in enumerate(model.bond_raw_emb):
        mask = torch.as_tensor(bond_ok[field], dtype=torch.bool, device=emb.weight.device)
        zero_support_ok &= bool(torch.all(emb.weight[~mask] == 0))
    data_ok = (sha == C.RAW_CACHE_SHA256 and sizes == [3360, 420, 420]
               and disjoint and cover and scaler_ok and z_ok and metric_ok
               and emb_ok and zero_support_ok
               and bool(np.isfinite(bundle.labels).all()))
    _rec(checks, "9_data_split_scaler_metrics_categories",
         "data SHA, official split, Train-only standardisation, raw-unit restoration and full OGB category dimensions are correct",
         data_ok, {"raw_cache_sha256": sha,
                   "split_sizes": sizes, "disjoint": bool(disjoint),
                   "cover_0_4199": bool(cover), "train_scaler": scaler,
                   "bundle_scaler": bundle.scaler, "all_rows_scaler": all_scaler,
                   "train_standardised_mean": float(z_train.mean()),
                   "train_standardised_std": float(z_train.std(ddof=0)),
                   "metric_restoration_ok": bool(metric_ok),
                   "atom_dims": bundle.atom_dims, "bond_dims": bundle.bond_dims,
                   "embedding_tables_match_ogb": bool(emb_ok),
                   "unsupported_rows_are_zero": bool(zero_support_ok),
                   "unseen_category_counts": bundle.unseen,
                   "test_metric_label": cfg["split"]["test_label"]})

    # 10. isolated atoms and disconnected molecules are preserved
    isolated_rows = []
    n_isolated = 0
    for row in all_rows.tolist():
        st = full[int(row)]
        iso = int(np.sum(np.asarray(st.degree) == 0))
        if iso:
            isolated_rows.append(int(row))
            n_isolated += iso
    multi_rows = [int(r) for r in all_rows if int(np.asarray(bundle.raw["graph_n_components"])[int(r)]) > 1]
    isolated_ok = True
    for row in isolated_rows[:20]:
        st = full[row]
        for v in np.flatnonzero(np.asarray(st.degree) == 0).tolist():
            lo, hi = int(st.word_root_ptr[v]), int(st.word_root_ptr[v + 1])
            singles = [w for w in range(lo, hi) if int(st.word_size[w]) == 1]
            isolated_ok &= len(singles) == 1
            if singles:
                w = singles[0]
                isolated_ok &= int(st.word_boundary_count[w]) == 0
                isolated_ok &= int(st.word_internal_count[w]) == 0
    multi_ok = True
    for row in multi_rows:
        st = full[row]
        comp = _bfs_components(st.n_atoms, st.bond_u, st.bond_v)
        for v in range(st.n_atoms):
            r2 = np.zeros(st.n_atoms, bool)
            r2[v] = True
            r2[st.dist1_atoms[st.dist1_ptr[v]:st.dist1_ptr[v + 1]]] = True
            r2[st.dist2_atoms[st.dist2_ptr[v]:st.dist2_ptr[v + 1]]] = True
            if len(set(comp[r2].tolist())) != 1:
                multi_ok = False
        for e in range(st.n_bonds):
            if comp[int(st.bond_u[e])] != comp[int(st.bond_v[e])]:
                multi_ok = False
    batch_iso_rows = isolated_rows[:8] if isolated_rows else [int(all_rows[0])]
    biso = T.collate(batch_iso_rows, full, bundle.raw, "cpu")
    with torch.no_grad():
        yiso = model(biso)["yhat_std"]
    iso_ok = bool(isolated_ok and multi_ok and torch.isfinite(yiso).all()
                  and int(biso["n_atoms"]) == int(sum(full[r].n_atoms for r in batch_iso_rows)))
    _rec(checks, "10_isolated_atoms_components_preserved",
         "isolated atoms and disconnected components are retained, encoded with singletons and never cross-component",
         iso_ok, {"isolated_atoms": int(n_isolated),
                  "rows_with_isolated_atoms": len(isolated_rows),
                  "multi_component_molecules": len(multi_rows),
                  "isolated_singleton_boundary_zero": bool(isolated_ok),
                  "multi_component_r2_stays_in_component": bool(multi_ok),
                  "batch_forward_finite": bool(torch.isfinite(yiso).all())})

    # 11. complete candidate statistics and real-batch peak resources
    stats_all = S.structure_stats(full, all_rows)
    per_split = {}
    root_hist_all = []
    for name, rows in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
        rows = np.asarray([int(r) for r in rows if int(r) in full], np.int64)
        per_split[name] = S.structure_stats(full, rows)
        if rows.size:
            root_hist_all.append(np.concatenate([np.diff(full[int(r)].word_root_ptr) for r in rows]))
    root_hist = np.concatenate(root_hist_all) if root_hist_all else np.zeros(0, np.int64)
    hist, hist_edges = np.histogram(root_hist, bins=np.arange(0, int(root_hist.max()) + 2))
    hist_payload = {str(int(hist_edges[i])): int(hist[i]) for i in range(hist.size)}
    cand_bytes = array_bytes
    # one real training step on a batch of 64 Train molecules
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_model, _, _, _ = _build_zeroed(bundle, seed=0)
    step_model = step_model.to(device)
    step_model.train()
    batch_rows = np.asarray(all_rows[:64], np.int64)
    t_col = time.time()
    b64 = T.collate(batch_rows.tolist(), full, bundle.raw, device)
    collate_seconds = time.time() - t_col
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(step_model.parameters(), lr=3e-4, weight_decay=1e-4)
    t_fwd = time.time()
    out = step_model(b64)
    fwd_seconds = time.time() - t_fwd
    target = torch.as_tensor(bundle.labels_std()[batch_rows], device=device,
                             dtype=torch.float32)
    loss = torch.mean((out["yhat_std"] - target) ** 2)
    t_bwd = time.time()
    loss.backward()
    bwd_seconds = time.time() - t_bwd
    torch.nn.utils.clip_grad_norm_(step_model.parameters(), 5.0)
    opt.step()
    peak_vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    resources = {
        "device": device, "batch_molecules": 64,
        "batch_atoms": int(b64["n_atoms"]), "batch_bonds": int(b64["n_bonds"]),
        "batch_words": int(b64["n_words"]),
        "collate_seconds": collate_seconds, "forward_seconds": fwd_seconds,
        "backward_seconds": bwd_seconds, "peak_vram_bytes": peak_vram,
        "peak_rss_bytes": rss,
    }
    coverage = {
        "all": stats_all,
        "train": per_split["train"], "valid": per_split["valid"],
        "test": per_split["test"],
        "root_candidate_histogram": hist_payload,
        "structure_build_seconds": build_seconds,
        "structure_array_bytes": int(cand_bytes),
        "peak_rss_after_build_bytes": int(rss_after_build),
        "one_batch_resources": resources,
    }
    C.write_json(C.ANALYSIS_DIR / "candidate_coverage.json", coverage)
    scale_ok = bool(stats_all["roots_without_words"] == 0
                    and stats_all["n_words"] == int(sum(full[r].n_words for r in all_rows))
                    and stats_all["n_words"] > 0
                    and all(per_split[s]["roots_without_words"] == 0
                            for s in ("train", "valid", "test")))
    _rec(checks, "11_candidate_scale_and_batch_resources",
         "complete candidate enumeration, singleton coverage and a measured real-batch resource footprint",
         scale_ok, {"candidate_total": stats_all["n_words"],
                    "words_by_size": stats_all["words_by_size_complete"],
                    "words_per_root_max": stats_all["words_per_root_max"],
                    "words_per_molecule_max": stats_all["words_per_molecule_max"],
                    "roots_without_words": stats_all["roots_without_words"],
                    "per_split": per_split, "resources": resources,
                    "structure_array_bytes": int(cand_bytes),
                    "structure_build_seconds": build_seconds,
                    "peak_rss_after_build_bytes": int(rss_after_build)})

    payload = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_json(cfg),
        "code_sha256": C.code_sha([
            "wobr_common.py", "wobr_data.py", "wobr_structure.py", "wobr_canonical.py",
            "wobr_sparsemax.py", "wobr_tensor.py", "wobr_model.py", "wobr_checks.py",
        ]),
        "gwic_sources": C.assert_gwic_sources(),
        "checks": checks,
        "n_pass": int(sum(c["status"] == "PASS" for c in checks)),
        "n_total": len(checks),
        "all_pass": bool(all(c["status"] == "PASS" for c in checks)),
        "full_candidate_count": int(stats_all["n_words"]),
        "full_candidate_count_by_size": stats_all["words_by_size_complete"],
        "full_words_per_root_max": int(stats_all["words_per_root_max"]),
        "zeroed_categories": zeroed,
        "smoke_max_rows": int(max_rows) if max_rows else None,
        "elapsed_seconds": time.time() - t0,
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", payload)
    print(json.dumps(C.jsonable(payload), indent=1, sort_keys=True)[:6000])
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None,
                        help="local smoke only; formal checks must use the full 4200 rows")
    args = parser.parse_args()
    main(args.max_rows)