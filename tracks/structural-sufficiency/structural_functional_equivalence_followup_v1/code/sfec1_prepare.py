"""Correction A and preparation of the unified model prediction table.

Outputs
-------
``analysis/correction_table.csv``
    Parent v0 versus canonical-orientation recomputation for the P1/P2/P3 and
    Valid orientation-affected subsets.
``data/prepared_pcqm.npz``
    Frozen discovery fits, triple statistics and one row per evaluation pair
    observation with all model predictions and enable flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D


def _v0_style_controlled_masking(
    b: D.Bundle, base: dict[str, Any], cfg: dict[str, Any], rk: dict[int, str],
    rank_to_sid: np.ndarray, pairs_disc_can: dict[str, np.ndarray],
    pairs_eval_can: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Parent P2 controlled masking, recomputed with canonical pairs."""
    from sfec1_data import fit_shared, fit_local_ctx, predict_local
    ctx = b.ctx_disc
    q_id = b.C["q_id"]
    q_ctx = q_id[ctx.C].astype(np.int64)
    q_eval = q_id[b.ctx_eval.C].astype(np.int64)
    n_q = int(max(q_ctx.max(), q_eval.max()) + 1)
    design = D.make_design(ctx, b.rank, q_ctx)
    disc_counts = D.pair_count_matrix(pairs_disc_can)
    eval_counts = D.pair_count_matrix(pairs_eval_can)
    max_pairs = int(cfg["support"]["strict_mask_max_pairs"])
    min_eval = int(cfg["support"]["strict_mask_min_eval_contexts"])
    cand = []
    for a in range(64):
        for bb in range(a + 1, 64):
            if disc_counts[a, bb] <= 0 or eval_counts[a, bb] < min_eval:
                continue
            cand.append((a, bb, int(disc_counts[a, bb]), int(eval_counts[a, bb])))
    cand.sort(key=lambda x: (-x[2], -x[3], rk.get(x[0], ""), rk.get(x[1], "")))
    out = []
    pooled = []
    for a, bb, dn, en in cand[:max_pairs]:
        rows = (pairs_disc_can["A"] == a) & (pairs_disc_can["B"] == bb)
        contexts = np.unique(pairs_disc_can["ctx"][rows])
        a_sid = int(rank_to_sid[a])
        b_sid = int(rank_to_sid[bb])
        keep = np.ones(len(ctx.S), dtype=bool)
        for ci in contexts.tolist():
            lo, hi = int(ctx.off[ci]), int(ctx.off[ci + 1])
            pos = np.flatnonzero(ctx.S[lo:hi] == b_sid)
            if len(pos):
                keep[lo + int(pos[0])] = False
        cell_ctx = np.repeat(np.arange(ctx.n_ctx, dtype=np.int64), ctx.k)
        rank_cell = b.rank[ctx.S].astype(np.int64)
        has_a = np.bincount(cell_ctx, weights=((rank_cell == a) & keep).astype(np.float64),
                            minlength=ctx.n_ctx) > 0
        has_b = np.bincount(cell_ctx, weights=((rank_cell == bb) & keep).astype(np.float64),
                            minlength=ctx.n_ctx) > 0
        direct_after = int(np.sum(has_a & has_b))
        rec = {
            "A_rank": a, "B_rank": bb, "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
            "n_discovery_direct": dn, "n_evaluation": en,
            "applicable": False,
        }
        if direct_after:
            rec["reason"] = "direct co-occurrence remains after masking"
            out.append(rec)
            continue
        masked_fit = fit_shared(design, keep=keep)
        if masked_fit["comp"][a] < 0 or masked_fit["comp"][a] != masked_fit["comp"][bb]:
            rec["reason"] = "disconnected after masking"
            out.append(rec)
            continue
        local_fit = fit_local_ctx(
            design, masked_fit["b"], n_q=n_q,
            lam=float(cfg["estimators"]["local_lambda"]),
            min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
            keep=keep, q_ctx_all=q_ctx,
        )
        erows = (pairs_eval_can["A"] == a) & (pairs_eval_can["B"] == bb)
        pred_shared = np.full(int(erows.sum()), masked_fit["b"][bb] - masked_fit["b"][a])
        pl = predict_local(
            pairs_eval_can["ctx"][erows], pairs_eval_can["A"][erows],
            pairs_eval_can["B"][erows], q_eval, masked_fit["b"],
            {"comp": local_fit["comp"], "supported": local_fit["supported"],
             "d": local_fit["d"]},
        )
        delta = pairs_eval_can["delta"][erows]
        mu = float(np.median(pairs_disc_can["delta"][rows])) if np.any(rows) else float("nan")
        rec.update({
            "applicable": True,
            "n_evaluation_contexts": int(len(np.unique(pairs_eval_can["ctx"][erows]))),
            "shared_pred": float(masked_fit["b"][bb] - masked_fit["b"][a]),
            "mae_zero": float(np.mean(np.abs(delta))),
            "mae_shared": float(np.mean(np.abs(delta - pred_shared))),
            "mae_pair_full": float(np.mean(np.abs(delta - mu))) if np.isfinite(mu) else None,
            "mae_local": float(np.mean(np.abs(delta - pl["pred"]))),
            "local_coverage": float(np.mean(pl["enabled"])),
        })
        rec["relative_gain_shared_vs_zero"] = 1.0 - rec["mae_shared"] / max(rec["mae_zero"], 1e-12)
        rec["relative_gain_local_vs_shared"] = 1.0 - rec["mae_local"] / max(rec["mae_shared"], 1e-12)
        out.append(rec)
        pooled.append((erows, pred_shared, pl["pred"], mu))
    if pooled:
        mask = np.zeros(len(pairs_eval_can["delta"]), dtype=bool)
        ps = np.full(len(mask), np.nan)
        pl = np.full(len(mask), np.nan)
        pf = np.full(len(mask), np.nan)
        for erows, s, l, mu in pooled:
            mask |= erows
            ps[erows] = s
            pl[erows] = l
            pf[erows] = mu
        delta = pairs_eval_can["delta"][mask]
        pool = {
            "n_pairs": len(pooled),
            "n_obs": int(mask.sum()),
            "n_ctx": int(len(np.unique(pairs_eval_can["ctx"][mask]))),
            "mae_zero": float(np.mean(np.abs(delta))),
            "mae_shared": float(np.mean(np.abs(delta - ps[mask]))),
            "mae_local": float(np.mean(np.abs(delta - pl[mask]))),
            "mae_pair_full": float(np.mean(np.abs(delta - pf[mask]))),
        }
        pool["relative_gain_shared_vs_zero"] = 1.0 - pool["mae_shared"] / max(pool["mae_zero"], 1e-12)
        pool["relative_gain_local_vs_shared"] = 1.0 - pool["mae_local"] / max(pool["mae_shared"], 1e-12)
    else:
        pool = {"n_pairs": 0, "n_obs": 0}
    return {"pairs": out, "pool": pool, "n_candidates": len(cand)}


def _pair_set(pairs: dict[str, np.ndarray]) -> set[tuple[int, int, int]]:
    A = np.minimum(pairs["A"], pairs["B"])
    B = np.maximum(pairs["A"], pairs["B"])
    return set(zip(pairs["ctx"].tolist(), A.tolist(), B.tolist()))


def run_correction(b: D.Bundle, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rk = D.rank_key(b.rank, b.S)
    rank_to_sid = D.rank_to_sid(b.rank)
    # parent orientation
    pd0 = D.build_pairs(b.ctx_disc, b.rank, canonical=False)
    pe0 = D.build_pairs(b.ctx_eval, b.rank, canonical=False)
    ds0 = D.pair_stats(pd0)
    es0 = D.pair_stats(pe0)
    # canonical
    pd = b.pairs_disc
    pe = b.pairs_eval
    ds = D.pair_stats(pd)
    es = D.pair_stats(pe)
    # canonical pair sets must be identical observation sets after orientation
    set0 = _pair_set(pd0)
    set_new = _pair_set(pd)
    set0_eval = _pair_set(pe0)
    set_new_eval = _pair_set(pe)
    assert set0 == set_new, "discovery observation set changed under canonicalisation"
    assert set0_eval == set_new_eval, "evaluation observation set changed under canonicalisation"
    n_rev_disc = int(np.sum(pd0["A"] > pd0["B"]))
    n_rev_eval = int(np.sum(pe0["A"] > pe0["B"]))
    # v0 supported P1 pairs
    eval_n0 = D.lookup_pair_count(es0, ds0["pair_key"])
    supported0 = (ds0["n"] >= 10) & (eval_n0 >= 5)
    eval_n = D.lookup_pair_count(es, ds["pair_key"])
    supported = (ds["n"] >= 10) & (eval_n >= 5)
    key0_eval = D.pair_key(pe0["A"], pe0["B"])
    mu0 = D.lookup_pair(ds0, key0_eval, "median")
    n0 = D.lookup_pair_count(ds0, key0_eval)
    base0 = np.isfinite(mu0) & (n0 >= 1) & np.isin(key0_eval, ds0["pair_key"][supported0])
    key_eval = D.pair_key(pe["A"], pe["B"])
    mu = D.lookup_pair(ds, key_eval, "median")
    n = D.lookup_pair_count(ds, key_eval)
    base = np.isfinite(mu) & (n >= 1) & np.isin(key_eval, ds["pair_key"][supported])
    assert int(base0.sum()) == int(base.sum()), (base0.sum(), base.sum())
    assert int(supported0.sum()) == int(supported.sum())
    # metrics on the parent supported set (invariant because both prediction and
    # delta flip together; verify numerically)
    d0 = pe0["delta"][base0]
    p0 = mu0[base0]
    d1 = pe["delta"][base]
    p1 = mu[base]
    rc0, nc0 = D.compact_ctx(pe0["ctx"][base0])
    rc1, nc1 = D.compact_ctx(pe["ctx"][base])
    metrics0 = D.weighted_metric(d0, p0, rc0, np.zeros(len(d0), int), nc0)
    metrics1 = D.weighted_metric(d1, p1, rc1, np.zeros(len(d1), int), nc1)
    # pair-level orientation effects: the stored orientation of an unordered
    # pair is deterministic (cells inside a context are sorted by S_id), so a
    # pair type is either fully reversed or not reversed at all.
    key0_row = D.pair_key(pd0["A"], pd0["B"])
    canon0 = (np.minimum(pd0["A"], pd0["B"]).astype(np.int64) * 64
              + np.maximum(pd0["A"], pd0["B"]).astype(np.int64))
    rev_rows = pd0["A"] > pd0["B"]
    rev_count = np.bincount(canon0, weights=rev_rows.astype(np.float64))
    all_count = np.bincount(canon0)
    rev_frac = rev_count / np.maximum(all_count, 1)
    reversed_pair = rev_frac > 0.5
    assert np.all((rev_frac < 1e-12) | (rev_frac > 1 - 1e-12))
    common_keys = np.intersect1d(ds0["pair_key"], ds["pair_key"])
    rev_key = np.zeros(len(ds["pair_key"]), dtype=bool)
    pos_common = np.searchsorted(ds["pair_key"], 0)
    canon_keys_all = ds["pair_key"]
    # map the canonical key of every parent row to a boolean
    rev_by_key = dict(zip(canon0.tolist(), rev_rows.tolist()))
    # a canonical unordered pair is reversed iff any of its rows was reversed
    rev_map: dict[int, bool] = {}
    for kk, rr in zip(canon0.tolist(), rev_rows.tolist()):
        rev_map[int(kk)] = bool(rev_map.get(int(kk), False) or bool(rr))
    for i, kk in enumerate(canon_keys_all.tolist()):
        rev_key[i] = rev_map.get(int(kk), False)
    sign_flip = rev_key
    rows: list[dict[str, Any]] = []

    def add(subset, metric, v0, corrected, note=""):
        rows.append({
            "subset": subset, "metric": metric, "v0": v0,
            "corrected": corrected,
            "difference": (None if v0 is None or corrected is None else float(corrected) - float(v0)),
            "note": note,
        })

    add("P1/supported", "n_pairs", int(supported0.sum()), int(supported.sum()),
        "same unordered structural pairs; v0 stored orientation is mixed")
    add("P1/supported", "n_observation_rows", int(base0.sum()), int(base.sum()),
        "the evaluation observation set is unchanged")
    add("P1/supported", "pair_MAE_observation_equal",
        metrics0["observation"]["mae"], metrics1["observation"]["mae"],
        "MAE is invariant when delta and prediction flip together")
    add("P1/supported", "pair_RMSE",
        metrics0["observation"]["rmse"], metrics1["observation"]["rmse"])
    add("P1/supported", "pair_sign_accuracy",
        metrics0["observation"]["sign_accuracy"], metrics1["observation"]["sign_accuracy"])
    _, pid_supported = np.unique(key_eval[base], return_inverse=True)
    add("P1/supported", "pair_type_equal_MAE_PAIR", None,
        float(D.pair_macro_mae(pid_supported, np.abs(p1 - d1))),
        "true macro over 1602 pair types; v0 did not report this weighting")
    add("P1/supported", "pair_type_equal_MAE_ZERO", None,
        float(D.pair_macro_mae(pid_supported, np.abs(d1))),
        "same 2x2 weighting for the no-prediction baseline")
    rect0 = int(np.sum(ds0["n"].astype(np.int64) * (ds0["n"].astype(np.int64) - 1) // 2))
    add("P1/discovery", "rectangle_count_sum_AB_k_AB(k_AB-1)/2", rect0, rect0,
        "2x2 rectangles are context-pairs sharing an A,B pair; this is not the 297996 pair-observation count")
    add("P1/discovery", "fraction_reversed_pair_rows",
        n_rev_disc / max(len(pd0["delta"]), 1), 0.0,
        "canonical A<B removes all reversed rows")
    add("P1/evaluation", "fraction_reversed_pair_rows",
        n_rev_eval / max(len(pe0["delta"]), 1), 0.0)
    add("P1/discovery", "n_pairs_with_flipped_median_sign",
        int(np.sum(sign_flip)), 0,
        "v0 pairs whose reported median had the opposite sign to the canonical delta")
    add("P1/discovery", "n_pairs_affected_by_direction",
        int(np.sum(rev_key)), 0,
        "pairs whose stored orientation was B->A")
    p3_v0 = C.read_json(C.PARENT_ANALYSIS / "p3_pcqm.json")
    add("P3/all", "n_q_supported", p3_v0["n_q_supported"], p3_v0["n_q_supported"],
        "SHARED/LOCAL use stored A,B consistently; canonicalisation does not change them")
    add("P3/all", "SHARED_MAE", p3_v0["metrics"]["all"]["SHARED"]["mae"],
        p3_v0["metrics"]["all"]["SHARED"]["mae"])
    add("P3/all", "LOCAL_MAE", p3_v0["metrics"]["all"]["LOCAL"]["mae"],
        p3_v0["metrics"]["all"]["LOCAL"]["mae"])
    add("P3/all", "LOCAL_NULL_MAE", p3_v0["metrics"]["all"]["LOCAL_NULL"]["mae"],
        p3_v0["metrics"]["all"]["LOCAL_NULL"]["mae"])
    # v0-style P2 masking recomputed with canonical pairs
    design = D.make_design(b.ctx_disc, b.rank, b.C["q_id"][b.ctx_disc.C].astype(np.int64))
    base_fit = D.fit_shared(design)
    p2_v0 = C.read_json(C.PARENT_ANALYSIS / "p2_pcqm.json")
    p2_new = _v0_style_controlled_masking(
        b, base_fit, cfg, rk, rank_to_sid, pd, pe
    )
    add("P2/controlled_masking", "n_applicable_pairs",
        sum(1 for x in p2_v0["controlled_masking"]["pairs"] if x.get("applicable")),
        p2_new["pool"].get("n_pairs", 0),
        "canonical pair rows make direct discovery co-occurrence visible")
    for field, label in (("n_obs", "n_observations"), ("mae_zero", "ZERO_MAE"),
                         ("mae_shared", "SHARED_MAE"), ("mae_local", "LOCAL_MAE"),
                         ("mae_pair_full", "PAIR_FULL_MAE")):
        add("P2/controlled_masking", label,
            p2_v0["controlled_masking"]["pool"].get(field),
            p2_new["pool"].get(field),
            "v0-style masking: only the direct B cell is deleted (not strict identity masking)")
    add("P2/controlled_masking", "relative_gain_shared_vs_zero",
        p2_v0["controlled_masking"]["pool"].get("relative_gain_shared_vs_zero"),
        p2_new["pool"].get("relative_gain_shared_vs_zero"))
    add("P2/controlled_masking", "relative_gain_local_vs_shared",
        p2_v0["controlled_masking"]["pool"].get("relative_gain_local_vs_shared"),
        p2_new["pool"].get("relative_gain_local_vs_shared"))
    # Valid orientation coverage (parent keyed mu by stored train orientation,
    # while valid pairs were stored in valid-S_id order)
    valid_prefix = (C.PARENT_DIR / Path(cfg["datasets"]["pcqm"]["valid_prefix"])).resolve()
    valid_cells = D.load_cells(valid_prefix)
    valid_S = D.load_S(valid_prefix)
    valid_M = D.load_M(valid_prefix, columns=["mol_id", "mol_key"])
    s_digest = {D.digest(str(k)): int(i) for i, k in enumerate(b.S["S_key"].tolist())}
    vmap = np.asarray([s_digest.get(D.digest(str(k)), -1) for k in valid_S["S_key"].tolist()], dtype=np.int64)
    train_mol = set(D.digest(str(k)) for k in b.M["mol_key"].tolist())
    valid_mol_dup = np.asarray([D.digest(str(k)) in train_mol for k in valid_M["mol_key"].tolist()])
    cells_v = {k: v.copy() for k, v in valid_cells.items()}
    mapped = vmap[cells_v["S_id"].astype(np.int64)]
    cells_v["S_id"] = mapped
    seen_s = mapped >= 0
    dup = valid_mol_dup[cells_v["mol_id"].astype(np.int64)]
    base_mask = seen_s & b.primary[np.maximum(mapped, 0)] & ~dup
    labels = b.labels
    ctxv = D.build_contexts(cells_v, base_mask, labels, len(b.S["S_id"]), b.primary, b.rank)
    pv0 = D.build_pairs(ctxv, b.rank, canonical=False)
    pv = D.build_pairs(ctxv, b.rank, canonical=True)
    key_v0 = D.pair_key(pv0["A"], pv0["B"])
    key_v = D.pair_key(pv["A"], pv["B"])
    use_v0 = np.isfinite(D.lookup_pair(ds0, key_v0, "median")) & (D.lookup_pair_count(ds0, key_v0) >= 10)
    use_v = np.isfinite(D.lookup_pair(ds, key_v, "median")) & (D.lookup_pair_count(ds, key_v) >= 10)
    add("Valid/orientation", "n_pair_observations_with_frozen_mu",
        int(use_v0.sum()), int(use_v.sum()),
        "parent lookup used the mixed stored orientation; canonical lookup recovers the rest")
    add("Valid/orientation", "n_pair_types_with_frozen_mu",
        int(len(np.unique(key_v0[use_v0]))), int(len(np.unique(key_v[use_v]))))
    if np.any(use_v0):
        add("Valid/orientation", "PAIR_MAE_on_v0_visible_subset",
            float(np.mean(np.abs(pv0["delta"][use_v0] - D.lookup_pair(ds0, key_v0[use_v0], "median")))),
            float(np.mean(np.abs(pv["delta"][use_v0] - D.lookup_pair(ds, key_v[use_v0], "median")))),
            "same observation subset; no missing-mu rows are added")
    details = {
        "n_discovery_pair_observations": int(len(pd0["delta"])),
        "n_evaluation_pair_observations": int(len(pe0["delta"])),
        "n_reversed_discovery_rows": n_rev_disc,
        "n_reversed_evaluation_rows": n_rev_eval,
        "fraction_reversed_discovery_rows": n_rev_disc / max(len(pd0["delta"]), 1),
        "fraction_reversed_evaluation_rows": n_rev_eval / max(len(pe0["delta"]), 1),
        "n_pairs_canonical": int(len(ds["pair_key"])),
        "n_pairs_with_flipped_sign": int(np.sum(sign_flip)),
        "max_abs_p1_mae_difference": abs(metrics0["observation"]["mae"] - metrics1["observation"]["mae"]),
        "max_abs_p1_rmse_difference": abs(metrics0["observation"]["rmse"] - metrics1["observation"]["rmse"]),
        "p1_pair_type_equal_mae_pair": float(D.pair_macro_mae(pid_supported, np.abs(p1 - d1))),
        "p1_pair_type_equal_mae_zero": float(D.pair_macro_mae(pid_supported, np.abs(d1))),
        "rectangle_count_definition": "sum over canonical A,B pairs of k_AB*(k_AB-1)/2, where k_AB is the number of discovery contexts containing both A and B",
        "rectangle_count_discovery": rect0,
        "rectangle_count_note": "This counts unordered context pairs sharing an A,B pair; it is not the pair-observation count 297996.",
        "p1_v0_supported_n_obs": int(base0.sum()),
        "p1_canonical_supported_n_obs": int(base.sum()),
        "p2_v0style_corrected": p2_new,
        "valid_v0_visible_obs": int(use_v0.sum()),
        "valid_canonical_visible_obs": int(use_v.sum()),
    }
    return rows, details


def prepare(b: D.Bundle, cfg: dict[str, Any]) -> dict[str, Any]:
    q_id = b.C["q_id"]
    q_disc = q_id[b.ctx_disc.C].astype(np.int64)
    q_eval = q_id[b.ctx_eval.C].astype(np.int64)
    n_q = int(max(q_disc.max(), q_eval.max()) + 1)
    design = D.make_design(b.ctx_disc, b.rank, q_disc)
    shared = D.fit_shared(design)
    local = D.fit_local_ctx(
        design, shared["b"], n_q=n_q,
        lam=float(cfg["estimators"]["local_lambda"]),
        min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
        q_ctx_all=q_disc,
    )
    q_null = D.make_null_q(b.ctx_disc, q_disc, int(cfg["estimators"]["null_seed"]))
    local_null = D.fit_local_ctx(
        design, shared["b"], n_q=n_q,
        lam=float(cfg["estimators"]["local_lambda"]),
        min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
        q_ctx_all=q_null,
    )
    ds = D.pair_stats(b.pairs_disc)
    es = D.pair_stats(b.pairs_eval)
    tri = D.triple_stats(b.pairs_disc, q_disc, n_q)
    tri_n = D.triple_stats(b.pairs_disc, q_null, n_q)
    pe = b.pairs_eval
    A = pe["A"].astype(np.int64)
    B = pe["B"].astype(np.int64)
    key = D.pair_key(A, B)
    mu = D.lookup_pair(ds, key, "median")
    mean = D.lookup_pair(ds, key, "mean")
    n_disc = D.lookup_pair_count(ds, key)
    n_eval_ctx = D.lookup_pair_count(es, key)
    comp = shared["comp"]
    main = (n_disc >= int(cfg["support"]["p1_min_discovery_contexts"])) & \
           (n_eval_ctx >= int(cfg["support"]["p1_min_evaluation_contexts"])) & \
           (comp[A] >= 0) & (comp[A] == comp[B])
    pred_shared = shared["b"][B] - shared["b"][A]
    pred_local = D.predict_local(pe["ctx"], A, B, q_eval, shared["b"], local)
    pred_null = D.predict_local(pe["ctx"], A, B, q_eval, shared["b"], local_null)
    q_row = q_eval[pe["ctx"]]
    tmed, tcnt = D.lookup_triple(tri, A, B, q_row, n_q, "median")
    tmean, _ = D.lookup_triple(tri, A, B, q_row, n_q, "mean")
    tnmed, tncnt = D.lookup_triple(tri_n, A, B, q_row, n_q, "median")
    pq_enabled = tcnt >= int(cfg["support"]["conditional_min_discovery_contexts"])
    pqn_enabled = tncnt >= int(cfg["support"]["conditional_min_discovery_contexts"])
    pair_q = np.where(pq_enabled, tmed, mu)
    pair_q_mean = np.where(pq_enabled, tmean, mean)
    pair_q_null = np.where(pqn_enabled, tnmed, mu)
    allcond = main & pq_enabled & pqn_enabled & pred_local["enabled"] & pred_null["enabled"]
    clean = D.pair_clean_mask(b.ctx_eval, pe, D.molecule_clean_flags(b.cells, b.M)["clean"])
    preds = {
        "ZERO": np.zeros(len(key)),
        "PAIR": mu,
        "PAIR_MEAN": mean,
        "SHARED": pred_shared,
        "PAIR_Q": pair_q,
        "PAIR_Q_MEAN": pair_q_mean,
        "PAIR_Q_NULL": pair_q_null,
        "LOCAL": pred_local["pred"],
        "LOCAL_NULL": pred_null["pred"],
    }
    enabled = {
        "PAIR_Q": pq_enabled,
        "PAIR_Q_MEAN": pq_enabled,
        "PAIR_Q_NULL": pqn_enabled,
        "LOCAL": pred_local["enabled"],
        "LOCAL_NULL": pred_null["enabled"],
    }
    # compact pair index for bootstrap / per-pair outputs (all canonical pairs)
    ukey, pair_index = np.unique(key, return_inverse=True)
    payload = {
        "eval": {
            "ctx": pe["ctx"], "A": A, "B": B, "delta": pe["delta"], "key": key,
            "pair_index": pair_index.astype(np.int64), "q": q_eval,
            "n_disc": n_disc, "n_eval_ctx": n_eval_ctx,
            "main": main, "allcond": allcond, "clean": clean,
            "preds": preds, "enabled": enabled,
            "pos_i": pe["pos_i"], "pos_j": pe["pos_j"],
        },
        "disc": {
            "pair_key": ds["pair_key"], "A": ds["A"], "B": ds["B"], "n": ds["n"],
            "median": ds["median"], "mean": ds["mean"], "mad": ds["mad"],
            "p80_abs_dev": ds["p80_abs_dev"], "sd": ds["sd"], "sign_pos": ds["sign_pos"],
        },
        "triple_true": tri,
        "triple_null": tri_n,
        "shared_b": shared["b"], "shared_comp": shared["comp"],
        "local_d": local["d"], "local_comp": local["comp"],
        "local_supported": local["supported"], "local_nctx": local["n_contexts"],
        "local_null_d": local_null["d"], "local_null_comp": local_null["comp"],
        "local_null_supported": local_null["supported"],
        "n_q": np.asarray([n_q]),
        "q_disc": q_disc, "q_null": q_null,
        "ukey": ukey,
    }
    return payload


def main() -> int:
    cfg = C.load_config()
    b = D.load_bundle(cfg)
    rows, details = run_correction(b, cfg)
    C.write_csv(C.ANALYSIS_DIR / "correction_table.csv", rows)
    C.write_json(C.ANALYSIS_DIR / "correction_details.json", details)
    payload = prepare(b, cfg)
    # flatten to npz
    arrays = {}
    for k, v in payload["eval"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                arrays[f"eval_{k}_{kk}"] = vv
        else:
            arrays[f"eval_{k}"] = v
    for k, v in payload["disc"].items():
        arrays[f"disc_{k}"] = v
    arrays["triple_true_key"] = payload["triple_true"]["key"]
    arrays["triple_true_n"] = payload["triple_true"]["n"]
    arrays["triple_true_median"] = payload["triple_true"]["median"]
    arrays["triple_true_mean"] = payload["triple_true"]["mean"]
    arrays["triple_true_p80"] = payload["triple_true"]["p80_abs_dev"]
    arrays["triple_null_key"] = payload["triple_null"]["key"]
    arrays["triple_null_n"] = payload["triple_null"]["n"]
    arrays["triple_null_median"] = payload["triple_null"]["median"]
    arrays["triple_null_mean"] = payload["triple_null"]["mean"]
    for k in ("shared_b", "shared_comp", "local_d", "local_comp",
              "local_supported", "local_nctx", "local_null_d", "local_null_comp",
              "local_null_supported", "n_q", "q_disc", "q_null", "ukey"):
        arrays[k] = payload[k]
    np.savez(C.DATA_DIR / "prepared_pcqm.npz", **arrays)
    C.write_json(C.ANALYSIS_DIR / "prepare_summary.json", {
        "n_eval_rows": int(len(payload["eval"]["delta"])),
        "n_main": int(payload["eval"]["main"].sum()),
        "n_allcond": int(payload["eval"]["allcond"].sum()),
        "n_clean": int(payload["eval"]["clean"].sum()),
        "n_q": int(payload["n_q"][0]),
        "n_q_supported_local": int(payload["local_supported"].sum()),
        "n_q_supported_null": int(payload["local_null_supported"].sum()),
        "n_pairs": int(len(payload["ukey"])),
        "n_triples": int(len(payload["triple_true"]["key"])),
    })
    print(json.dumps(C.read_json(C.ANALYSIS_DIR / "prepare_summary.json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
