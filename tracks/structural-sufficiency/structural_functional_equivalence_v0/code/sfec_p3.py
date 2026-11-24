"""Stage E: P3 context-conditioned local rule with matched NULL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec_common as C
import sfec_data as D
from sfec_p1 import compact_ctx, load_dataset, metric, rank_key, sign_accuracy
from sfec_p2 import fit_shared, pair_rows


def _fit_local(
    ctx: D.Contexts,
    rank: np.ndarray,
    q_ctx: np.ndarray,
    bvec: np.ndarray,
    *,
    n_q: int,
    lam: float,
    min_contexts: int,
    cell_keep: np.ndarray | None = None,
) -> dict[str, Any]:
    d = np.zeros((n_q, 64), dtype=np.float64)
    comp = -np.ones((n_q, 64), dtype=np.int16)
    supported = np.zeros(n_q, dtype=bool)
    n_ctx_used = np.zeros(n_q, dtype=np.int64)
    n_cells_used = np.zeros(n_q, dtype=np.int64)
    order = np.argsort(q_ctx, kind="stable")
    qs = q_ctx[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]]) if len(qs) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(qs)] if len(qs) else np.zeros(0, np.int64)
    for st, en in zip(starts.tolist(), ends.tolist()):
        q = int(qs[st])
        idxs = order[st:en]
        L = np.zeros((64, 64), dtype=np.float64)
        rhs = np.zeros(64, dtype=np.float64)
        used = 0
        cells = 0
        for ci in idxs.tolist():
            lo, hi = int(ctx.off[ci]), int(ctx.off[ci + 1])
            if hi - lo < 2:
                continue
            S = ctx.S[lo:hi]
            y = ctx.y[lo:hi]
            if cell_keep is not None:
                keep = cell_keep[lo:hi]
                S = S[keep]
                y = y[keep]
            if len(S) < 2:
                continue
            O = rank[S].astype(np.int64)
            yt = y - y.mean()
            bt = bvec[O] - bvec[O].mean()
            r = yt - bt
            k = len(O)
            L[np.ix_(O, O)] += (np.eye(k) - 1.0 / k) / k
            rhs[O] += r / k
            used += 1
            cells += k
        if used < min_contexts:
            continue
        supported[q] = True
        n_ctx_used[q] = used
        n_cells_used[q] = cells
        d[q] = np.linalg.solve(L + lam * np.eye(64), rhs)
        adj = (np.abs(L) > 1e-12) & ~np.eye(64, dtype=bool)
        cid = 0
        for start in range(64):
            if comp[q, start] != -1 or not np.any(adj[start]):
                continue
            stack = [start]
            comp[q, start] = cid
            while stack:
                v = stack.pop()
                for w in np.flatnonzero(adj[v]).tolist():
                    if comp[q, w] == -1:
                        comp[q, w] = cid
                        stack.append(w)
            cid += 1
    return {
        "d": d,
        "comp": comp,
        "supported": supported,
        "n_contexts": n_ctx_used,
        "n_cells": n_cells_used,
    }


def _predict(
    eval_ctx: D.Contexts,
    eval_pairs: dict[str, np.ndarray],
    rank: np.ndarray,
    q_ctx: np.ndarray,
    bvec: np.ndarray,
    fit: dict[str, Any],
    *,
    use_true_q: bool,
) -> dict[str, np.ndarray]:
    n = len(eval_pairs["delta"])
    shared = bvec[eval_pairs["B"].astype(np.int64)] - bvec[eval_pairs["A"].astype(np.int64)]
    pred = shared.copy()
    enabled = np.zeros(n, dtype=bool)
    corr = np.zeros(n, dtype=np.float64)
    d = fit["d"]
    comp = fit["comp"]
    supported = fit["supported"]
    for i in range(n):
        q = int(q_ctx[int(eval_pairs["ctx"][i])])
        a = int(eval_pairs["A"][i]); bb = int(eval_pairs["B"][i])
        if supported[q] and comp[q, a] >= 0 and comp[q, a] == comp[q, bb]:
            c = d[q, bb] - d[q, a]
            pred[i] = shared[i] + c
            corr[i] = c
            enabled[i] = True
    return {"shared": shared, "pred": pred, "enabled": enabled, "correction": corr}


def run_p3(name: str, prefix: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    b = load_dataset(name, prefix)
    ctx = b["disc_ctx"]
    eval_ctx = b["eval_ctx"]
    disc_pairs = b["disc_pairs"]
    eval_pairs = b["eval_pairs"]
    base = fit_shared(ctx, b["rank"])
    bvec = base["b"]
    q_id = b["c_table"]["q_id"]
    q_ctx = q_id[ctx.C].astype(np.int64)
    q_eval = q_id[eval_ctx.C].astype(np.int64)
    n_q = int(max(q_ctx.max(), q_eval.max()) + 1) if len(q_ctx) and len(q_eval) else int(q_id.max() + 1)
    lam = float(cfg["estimators"]["local_lambda"])
    min_ctx = int(cfg["estimators"]["local_min_contexts_per_q"])
    fit = _fit_local(ctx, b["rank"], q_ctx, bvec, n_q=n_q, lam=lam, min_contexts=min_ctx)
    # matched NULL: permute q within (bond type, k) strata
    rng = np.random.default_rng(int(cfg["estimators"]["null_seed"]))
    q_null = q_ctx.copy()
    k_ctx = ctx.k.astype(np.int64)
    for kval in np.unique(k_ctx).tolist():
        idx = np.flatnonzero(k_ctx == kval)
        if len(idx) > 1:
            q_null[idx] = rng.permutation(q_ctx[idx])
    fit_null = _fit_local(ctx, b["rank"], q_null, bvec, n_q=n_q, lam=lam, min_contexts=min_ctx)
    pred_local = _predict(eval_ctx, eval_pairs, b["rank"], q_eval, bvec, fit, use_true_q=True)
    pred_null = _predict(eval_ctx, eval_pairs, b["rank"], q_eval, bvec, fit_null, use_true_q=True)
    delta = eval_pairs["delta"]
    shared = pred_local["shared"]
    local = pred_local["pred"]
    null = pred_null["pred"]
    row_ctx, n_ctx_used = compact_ctx(eval_pairs["ctx"])
    reps = cfg["statistics"]["bootstrap_replicates"]
    seed = cfg["statistics"]["bootstrap_seed"]
    # masks
    m_base = np.ones(len(delta), dtype=bool)
    m_both = pred_local["enabled"] & pred_null["enabled"]
    # high non-additivity pairs, frozen on discovery
    disc_stats = D.pair_stats(disc_pairs)
    disc_pred = bvec[disc_stats["B"].astype(np.int64)] - bvec[disc_stats["A"].astype(np.int64)]
    nonadd = np.abs(disc_stats["median"] - disc_pred)
    ok_pairs = disc_stats["n"] >= cfg["support_gates"]["p1_min_discovery_contexts"]
    thr = float(np.quantile(nonadd[ok_pairs], 0.75)) if np.any(ok_pairs) else float("inf")
    high_keys = disc_stats["pair_key"][ok_pairs & (nonadd >= thr)]
    eval_key = eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)
    m_high = np.isin(eval_key, high_keys)
    metrics = {}
    for label, mask in (("all", m_base), ("both_enabled", m_both), ("high_nonadditivity", m_high)):
        if not np.any(mask):
            metrics[label] = {"n": 0}
            continue
        d = delta[mask]
        rc, nc = compact_ctx(eval_pairs["ctx"][mask])
        rec = {"n_obs": int(len(d)), "n_ctx": int(nc),
               "n_pairs": int(len(np.unique(eval_key[mask])))}
        for model, pred in (("SHARED", shared), ("LOCAL", local), ("LOCAL_NULL", null)):
            m = metric(d, pred[mask])
            bm = D.bootstrap_mae(rc, np.abs(d - pred[mask]), nc, replicates=reps,
                                 seed=seed + 31, context_equal=False)
            m["mae_boot"] = bm["point"]
            m["mae_ci"] = [bm["lo"], bm["hi"]]
            m["sign_accuracy"] = sign_accuracy(d, pred[mask])
            rec[model] = m
        bd1 = D.bootstrap_diff(rc, np.abs(d - shared[mask]), np.abs(d - local[mask]), nc,
                               replicates=reps, seed=seed + 32, context_equal=False)
        bd2 = D.bootstrap_diff(rc, np.abs(d - local[mask]), np.abs(d - null[mask]), nc,
                               replicates=reps, seed=seed + 33, context_equal=False)
        rec["diff_SHARED_minus_LOCAL"] = {"point": bd1["diff"], "ci": [bd1["lo"], bd1["hi"]]}
        rec["diff_LOCAL_minus_NULL"] = {"point": bd2["diff"], "ci": [bd2["lo"], bd2["hi"]]}
        rec["relative_gain_local_vs_shared"] = (
            1.0 - rec["LOCAL"]["mae_boot"] / max(rec["SHARED"]["mae_boot"], 1e-12)
        )
        rec["relative_gain_local_vs_null"] = (
            1.0 - rec["LOCAL"]["mae_boot"] / max(rec["LOCAL_NULL"]["mae_boot"], 1e-12)
        )
        metrics[label] = rec
    # support / coverage
    q_supported = np.flatnonzero(fit["supported"])
    q_all = np.unique(q_ctx)
    unseen_q = ~np.isin(q_eval, q_supported)
    coverage = float(np.mean(pred_local["enabled"])) if len(delta) else 0.0
    coverage_both = float(np.mean(m_both)) if len(delta) else 0.0
    # per-q support
    q_rows = []
    q_keys = C.read_json(Path(str(prefix) + ".keys.json"))["q_keys"]
    order = np.argsort(q_ctx, kind="stable")
    qs = q_ctx[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]]) if len(qs) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(qs)] if len(qs) else np.zeros(0, np.int64)
    for st, en in zip(starts.tolist(), ends.tolist()):
        q = int(qs[st])
        idx = order[st:en]
        n_ge2 = int(np.sum(ctx.k[idx] >= 2))
        q_rows.append({
            "q_id": q, "q": q_keys[q] if q < len(q_keys) else "",
            "n_contexts": int(len(idx)), "n_contexts_k_ge_2": n_ge2,
            "supported": bool(fit["supported"][q]),
            "n_s_types": int(np.sum(fit["comp"][q] >= 0)) if q < len(fit["comp"]) else 0,
            "n_components": int(len(np.unique(fit["comp"][q][fit["comp"][q] >= 0]))) if q < len(fit["comp"]) else 0,
        })
    # judgement
    allm = metrics.get("all", {})
    gain_shared = allm.get("relative_gain_local_vs_shared", float("nan"))
    gain_null = allm.get("relative_gain_local_vs_null", float("nan"))
    ci1 = allm.get("diff_SHARED_minus_LOCAL", {}).get("ci", [float("nan")] * 2)
    ci2 = allm.get("diff_LOCAL_minus_NULL", {}).get("ci", [float("nan")] * 2)
    support_ok = len(q_supported) >= cfg["estimators"]["local_min_q_groups"] and coverage >= cfg["estimators"]["local_coverage_min"]
    beats_both = (
        np.isfinite(gain_shared) and gain_shared > 0 and gain_null > 0
        and ci1[0] > 0 and ci2[1] < 0
    )
    if not support_ok:
        judgement = "CONTEXT_SUPPORT_INSUFFICIENT"
        reason = "fewer than 10 supported q groups or correction coverage below 50%"
    elif beats_both and gain_shared >= cfg["judgement"]["p3_min_relative_gain"]:
        judgement = "TRANSFERABLE_CONTEXT_RULE"
        reason = "LOCAL beats SHARED and matched NULL by >=10% with positive paired intervals"
    elif np.isfinite(gain_shared) and gain_shared > 0:
        judgement = "CONTEXT_RULE_SCREEN_ONLY"
        reason = "positive point estimate but below the 10% screen or not both intervals positive"
    else:
        judgement = "NO_SUPPORTED_CONTEXT_RULE"
        reason = "LOCAL does not improve on SHARED"
    payload = {
        "dataset": name,
        "prefix": str(prefix),
        "n_discovery_contexts": int(ctx.n_ctx),
        "n_evaluation_pair_observations": int(len(delta)),
        "lambda": lam,
        "min_contexts_per_q": min_ctx,
        "n_q_total": int(len(q_all)),
        "n_q_supported": int(len(q_supported)),
        "correction_coverage": coverage,
        "correction_coverage_both": coverage_both,
        "unseen_q_fraction": float(np.mean(unseen_q)) if len(unseen_q) else float("nan"),
        "correction_abs_median": float(np.median(np.abs(pred_local["correction"][pred_local["enabled"]]))) if np.any(pred_local["enabled"]) else None,
        "correction_abs_p90": float(np.quantile(np.abs(pred_local["correction"][pred_local["enabled"]]), 0.90)) if np.any(pred_local["enabled"]) else None,
        "metrics": metrics,
        "q_support": q_rows,
        "high_nonadditivity_threshold": thr,
        "judgement": judgement,
        "reason": reason,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / f"p3_{name}.json", payload)
    C.write_csv(out_dir / f"context_rule_support_{name}.csv", q_rows)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pcqm", "zinc"])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = C.load_config()
    out = Path(args.out) if args.out else C.ANALYSIS_DIR
    payload = run_p3(args.dataset, Path(args.prefix), out, cfg)
    print(json.dumps({k: payload[k] for k in (
        "dataset", "n_q_total", "n_q_supported", "correction_coverage",
        "unseen_q_fraction", "judgement", "reason")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())