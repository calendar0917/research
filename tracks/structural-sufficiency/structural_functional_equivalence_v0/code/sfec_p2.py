"""Stage D: P2 shared composition model and unobserved replacement transfer."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import sfec_common as C
import sfec_data as D
from sfec_p1 import compact_ctx, load_dataset, metric, rank_key, sign_accuracy


def fit_shared(
    ctx: D.Contexts,
    rank: np.ndarray,
    n_s: int = 64,
    cell_keep: np.ndarray | None = None,
) -> dict[str, Any]:
    """Context-centred least squares for F(C,S)=a_C+b_S, equal weight per C."""
    L = np.zeros((n_s, n_s), dtype=np.float64)
    rhs = np.zeros(n_s, dtype=np.float64)
    n_used = 0
    for i in range(ctx.n_ctx):
        a, b = int(ctx.off[i]), int(ctx.off[i + 1])
        if b - a < 2:
            continue
        O = rank[ctx.S[a:b]].astype(np.int64)
        y = ctx.y[a:b]
        if cell_keep is not None:
            m = cell_keep[a:b]
            O = O[m]
            y = y[m]
        k = len(O)
        if k < 2:
            continue
        yt = y - y.mean()
        L[np.ix_(O, O)] += (np.eye(k) - 1.0 / k) / k
        rhs[O] += yt / k
        n_used += 1
    bvec = np.zeros(n_s, dtype=np.float64)
    comp = -np.ones(n_s, dtype=np.int16)
    adj = (np.abs(L) > 1e-12) & ~np.eye(n_s, dtype=bool)
    cid = 0
    for start in range(n_s):
        if comp[start] != -1 or not np.any(adj[start]):
            continue
        q = deque([start])
        comp[start] = cid
        members = []
        while q:
            v = q.popleft()
            members.append(v)
            for w in np.flatnonzero(adj[v]).tolist():
                if comp[w] == -1:
                    comp[w] = cid
                    q.append(w)
        ix = np.asarray(sorted(members), dtype=np.int64)
        if len(ix) >= 2:
            sub = L[np.ix_(ix, ix)]
            sol = np.linalg.lstsq(sub, rhs[ix], rcond=1e-12)[0]
            sol = sol - sol.mean()
            bvec[ix] = sol
        cid += 1
    # gradient check on identified rows
    grad = L @ bvec - rhs
    return {
        "b": bvec,
        "comp": comp,
        "L": L,
        "rhs": rhs,
        "n_contexts_used": int(n_used),
        "grad_max_abs": float(np.max(np.abs(grad))) if n_s else 0.0,
        "residual_sum": float(np.sum((L @ bvec - rhs) ** 2)),
    }


def component_path_length(L: np.ndarray, a: int, b: int) -> int | None:
    adj = (np.abs(L) > 1e-12) & ~np.eye(len(L), dtype=bool)
    if not adj[a, b]:
        pass
    dist = -np.ones(len(L), dtype=np.int64)
    dist[a] = 0
    q = deque([a])
    while q:
        v = q.popleft()
        if v == b:
            return int(dist[v])
        for w in np.flatnonzero(adj[v]).tolist():
            if dist[w] == -1:
                dist[w] = dist[v] + 1
                q.append(w)
    return None


def pair_rows(pairs: dict[str, np.ndarray], a: int, b: int) -> np.ndarray:
    return (pairs["A"] == a) & (pairs["B"] == b)


def run_p2(name: str, prefix: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    b = load_dataset(name, prefix)
    ctx = b["disc_ctx"]
    eval_ctx = b["eval_ctx"]
    disc_pairs = b["disc_pairs"]
    eval_pairs = b["eval_pairs"]
    base = fit_shared(ctx, b["rank"])
    bvec = base["b"]
    comp = base["comp"]
    disc_counts = D.pair_matrix(disc_pairs)
    eval_counts = D.pair_matrix(eval_pairs)
    disc_S_count = np.bincount(b["rank"][ctx.S].astype(np.int64), minlength=64)
    rk = rank_key(b["rank"], b["s_table"])
    rank_to_sid = -np.ones(64, dtype=np.int64)
    for sid, r in enumerate(b["rank"].tolist()):
        if r >= 0:
            rank_to_sid[int(r)] = int(sid)
    # natural unobserved replacements
    natural = []
    for a in range(64):
        for bb in range(a + 1, 64):
            if eval_counts[a, bb] <= 0 or disc_counts[a, bb] > 0:
                continue
            if disc_S_count[a] == 0 or disc_S_count[bb] == 0:
                continue
            if comp[a] < 0 or comp[a] != comp[bb]:
                continue
            rows = pair_rows(eval_pairs, a, bb)
            if not np.any(rows):
                continue
            delta = eval_pairs["delta"][rows]
            pred = float(bvec[bb] - bvec[a])
            natural.append({
                "A_rank": a, "B_rank": bb,
                "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
                "n_evaluation": int(rows.sum()),
                "n_discovery_direct": int(disc_counts[a, bb]),
                "n_discovery_A": int(disc_S_count[a]),
                "n_discovery_B": int(disc_S_count[bb]),
                "component": int(comp[a]),
                "path_length": component_path_length(base["L"], a, bb),
                "shared_pred": pred,
                "delta_median": float(np.median(delta)),
                "delta_abs_median": float(np.median(np.abs(delta))),
                "mae_zero": float(np.mean(np.abs(delta))),
                "mae_shared": float(np.mean(np.abs(delta - pred))),
                "rmse_shared": float(np.sqrt(np.mean((delta - pred) ** 2))),
                "sign_accuracy_shared": sign_accuracy(delta, np.full_like(delta, pred)),
            })
    natural_pool = _pool_rows(eval_pairs, comp, bvec, [x for x in natural])
    # controlled masking
    masked = _controlled_masking(b, base, cfg, rk, rank_to_sid)
    # summary / judgement
    gate = cfg["support_gates"]
    pool = masked.get("pool", {})
    improvement = pool.get("relative_gain_shared_vs_zero", float("nan"))
    ci = pool.get("diff_zero_minus_shared", {}).get("ci", [float("nan"), float("nan")])
    n_pairs = len(masked.get("pairs", []))
    n_ok_pairs = sum(1 for p in masked.get("pairs", []) if p.get("applicable") and p["n_evaluation"] >= gate["p2_min_evaluation_contexts_per_pair"])
    support_ok = n_ok_pairs >= gate["p2_min_masked_pairs"] and n_pairs > 0
    if not support_ok:
        judgement = "INSUFFICIENT_CROSSED_SUPPORT"
        reason = "fewer than 10 masked pairs with >=5 evaluation contexts"
    elif np.isfinite(improvement) and improvement >= cfg["judgement"]["p2_min_relative_gain"] and ci[0] > 0:
        judgement = "COMPOSITION_TRANSFER_SIGNAL"
        reason = "SHARED beats ZERO by >=15% with a positive paired interval"
    elif np.isfinite(improvement) and improvement > 0:
        judgement = "COMPOSITION_SCREEN_ONLY"
        reason = "positive point estimate but below the 15% screen or uncertain interval"
    else:
        judgement = "NO_SUPPORTED_COMPOSITION_TRANSFER"
        reason = "no clear improvement over ZERO"
    payload = {
        "dataset": name,
        "prefix": str(prefix),
        "n_discovery_contexts": int(ctx.n_ctx),
        "n_discovery_contexts_with_k_ge_2": int(base["n_contexts_used"]),
        "shared_fit": {
            "b": bvec.tolist(),
            "comp": comp.tolist(),
            "grad_max_abs": base["grad_max_abs"],
            "residual_sum": base["residual_sum"],
            "n_components": int(len(np.unique(comp[comp >= 0]))),
            "sizes": {int(c): int(np.sum(comp == c)) for c in np.unique(comp[comp >= 0])},
        },
        "natural_unobserved": {
            "n_pairs": len(natural),
            "n_evaluation_observations": int(sum(x["n_evaluation"] for x in natural)),
            "pairs": natural,
            "pool": natural_pool,
        },
        "controlled_masking": masked,
        "support_ok": bool(support_ok),
        "judgement": judgement,
        "reason": reason,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / f"p2_{name}.json", payload)
    _write_csv(out_dir / f"unseen_composition_results_{name}.csv", natural, masked.get("pairs", []))
    return payload


def _pool_rows(eval_pairs, comp, bvec, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled metrics for a set of rank pairs (natural unobserved)."""
    if not pairs:
        return {"n_obs": 0}
    keys = np.asarray([int(x["A_rank"]) * 64 + int(x["B_rank"]) for x in pairs], dtype=np.int64)
    key_rows = eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)
    mask = np.isin(key_rows, keys)
    if not np.any(mask):
        return {"n_obs": 0}
    delta = eval_pairs["delta"][mask]
    pred = np.asarray(
        [bvec[int(a) * 0 + int(b)] - bvec[int(a)] for a, b in zip(eval_pairs["A"][mask], eval_pairs["B"][mask])],
        dtype=np.float64,
    )
    row_ctx, n_ctx = compact_ctx(eval_pairs["ctx"][mask])
    m0 = metric(delta, np.zeros_like(delta))
    m1 = metric(delta, pred)
    return {
        "n_obs": int(len(delta)),
        "n_ctx": int(n_ctx),
        "mae_zero": m0["mae"],
        "mae_shared": m1["mae"],
        "relative_gain_shared_vs_zero": 1.0 - m1["mae"] / max(m0["mae"], 1e-12),
        "sign_accuracy_shared": sign_accuracy(delta, pred),
    }


def _controlled_masking(b: dict[str, Any], base: dict[str, Any], cfg: dict[str, Any], rk: dict[int, str], rank_to_sid: np.ndarray) -> dict[str, Any]:
    from sfec_p3 import _fit_local
    ctx = b["disc_ctx"]
    q_id = b["c_table"]["q_id"]
    q_ctx = q_id[ctx.C].astype(np.int64)
    q_eval = q_id[b["eval_ctx"].C].astype(np.int64)
    n_q = int(max(q_ctx.max(), q_eval.max()) + 1) if len(q_ctx) and len(q_eval) else int(q_id.max() + 1)
    eval_pairs = b["eval_pairs"]
    disc_pairs = b["disc_pairs"]
    disc_counts = D.pair_matrix(disc_pairs)
    eval_counts = D.pair_matrix(eval_pairs)
    max_pairs = int(cfg["support_gates"]["p2_max_masked_pairs"])
    min_eval = int(cfg["support_gates"]["p2_min_evaluation_contexts_per_pair"])
    cand = []
    for a in range(64):
        for bb in range(a + 1, 64):
            if disc_counts[a, bb] <= 0 or eval_counts[a, bb] < min_eval:
                continue
            cand.append((a, bb, int(disc_counts[a, bb]), int(eval_counts[a, bb])))
    cand.sort(key=lambda x: (-x[2], -x[3], rk.get(x[0], ""), rk.get(x[1], "")))
    out = []
    pooled_rows = []
    for a, bb, dn, en in cand:
        if len(out) >= max_pairs:
            break
        rows = pair_rows(disc_pairs, a, bb)
        contexts = np.unique(disc_pairs["ctx"][rows])
        a_sid = int(rank_to_sid[a]); b_sid = int(rank_to_sid[bb])
        keep = np.ones(len(ctx.S), dtype=bool)
        affected = 0
        for ci in contexts.tolist():
            lo, hi = int(ctx.off[ci]), int(ctx.off[ci + 1])
            S = ctx.S[lo:hi]
            pos = np.flatnonzero(S == b_sid)
            if len(pos):
                keep[lo + int(pos[0])] = False
                affected += 1
        # verify no direct co-occurrence remains
        direct_after = 0
        for ci in range(ctx.n_ctx):
            lo, hi = int(ctx.off[ci]), int(ctx.off[ci + 1])
            if hi - lo < 2:
                continue
            S = ctx.S[lo:hi][keep[lo:hi]]
            if np.any(S == a_sid) and np.any(S == b_sid):
                direct_after += 1
        if direct_after:
            out.append({"A_rank": a, "B_rank": bb, "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
                        "n_discovery_direct": dn, "n_evaluation": en, "applicable": False,
                        "reason": "direct co-occurrence remains after masking"})
            continue
        masked_fit = fit_shared(ctx, b["rank"], n_s=64, cell_keep=keep)
        if masked_fit["comp"][a] < 0 or masked_fit["comp"][a] != masked_fit["comp"][bb]:
            out.append({"A_rank": a, "B_rank": bb, "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
                        "n_discovery_direct": dn, "n_evaluation": en, "applicable": False,
                        "reason": "disconnected after masking"})
            continue
        path = component_path_length(masked_fit["L"], a, bb)
        pred = float(masked_fit["b"][bb] - masked_fit["b"][a])
        erows = pair_rows(eval_pairs, a, bb)
        delta = eval_pairs["delta"][erows]
        n_eval_ctx = len(np.unique(eval_pairs["ctx"][erows]))
        # condition corrections fitted after masking (q never sees the removed cell)
        local_fit = _fit_local(
            ctx, b["rank"], q_ctx, masked_fit["b"], n_q=n_q,
            lam=float(cfg["estimators"]["local_lambda"]),
            min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
            cell_keep=keep,
        )
        pred_local = np.full(len(delta), pred, dtype=np.float64)
        local_enabled = np.zeros(len(delta), dtype=bool)
        for j, ci in enumerate(eval_pairs["ctx"][erows].tolist()):
            q = int(q_eval[int(ci)])
            if (
                local_fit["supported"][q]
                and local_fit["comp"][q, a] >= 0
                and local_fit["comp"][q, a] == local_fit["comp"][q, bb]
            ):
                pred_local[j] = pred + local_fit["d"][q, bb] - local_fit["d"][q, a]
                local_enabled[j] = True
        # PAIR-FULL: direct discovery response from the unmasked data
        mu = _pair_median(disc_pairs, a, bb)
        rec = {
            "A_rank": a, "B_rank": bb, "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
            "n_discovery_direct": dn, "n_discovery_contexts_affected": affected,
            "n_evaluation": int(erows.sum()), "n_evaluation_contexts": int(n_eval_ctx),
            "applicable": True,
            "path_length_masked": path,
            "shared_pred": pred,
            "pair_full_mu": float(mu) if np.isfinite(mu) else None,
            "delta_median": float(np.median(delta)),
            "delta_abs_median": float(np.median(np.abs(delta))),
            "mae_zero": float(np.mean(np.abs(delta))),
            "mae_shared": float(np.mean(np.abs(delta - pred))),
            "mae_pair_full": float(np.mean(np.abs(delta - mu))) if np.isfinite(mu) else None,
            "rmse_shared": float(np.sqrt(np.mean((delta - pred) ** 2))),
            "sign_accuracy_shared": sign_accuracy(delta, np.full_like(delta, pred)),
            "mae_local": float(np.mean(np.abs(delta - pred_local))),
            "rmse_local": float(np.sqrt(np.mean((delta - pred_local) ** 2))),
            "sign_accuracy_local": sign_accuracy(delta, pred_local),
            "local_coverage": float(np.mean(local_enabled)),
        }
        rec["relative_gain_shared_vs_zero"] = 1.0 - rec["mae_shared"] / max(rec["mae_zero"], 1e-12)
        rec["gap_shared_vs_pair_full"] = (
            rec["mae_shared"] - rec["mae_pair_full"] if rec["mae_pair_full"] is not None else None
        )
        rec["relative_gain_local_vs_shared"] = 1.0 - rec["mae_local"] / max(rec["mae_shared"], 1e-12)
        rec["relative_gain_local_vs_zero"] = 1.0 - rec["mae_local"] / max(rec["mae_zero"], 1e-12)
        out.append(rec)
        pooled_rows.append((erows, pred, mu, pred_local))
    # pooled metrics
    if pooled_rows:
        mask = np.zeros(len(eval_pairs["delta"]), dtype=bool)
        pred = np.full(len(eval_pairs["delta"]), np.nan)
        pair_full = np.full(len(eval_pairs["delta"]), np.nan)
        pred_local = np.full(len(eval_pairs["delta"]), np.nan)
        for erows, p, mu, pl in pooled_rows:
            mask |= erows
            pred[erows] = p
            pair_full[erows] = mu
            pred_local[erows] = pl
        delta = eval_pairs["delta"][mask]
        pr = pred[mask]
        pl = pred_local[mask]
        pf = pair_full[mask]
        row_ctx, n_ctx = compact_ctx(eval_pairs["ctx"][mask])
        m0 = metric(delta, np.zeros_like(delta))
        m1 = metric(delta, pr)
        m2 = metric(delta[pf == pf], pf[pf == pf])
        reps = cfg["statistics"]["bootstrap_replicates"]
        seed = cfg["statistics"]["bootstrap_seed"]
        b0 = D.bootstrap_mae(row_ctx, np.abs(delta), n_ctx, replicates=reps, seed=seed + 11, context_equal=False)
        b1 = D.bootstrap_mae(row_ctx, np.abs(delta - pr), n_ctx, replicates=reps, seed=seed + 12, context_equal=False)
        bd = D.bootstrap_diff(row_ctx, np.abs(delta), np.abs(delta - pr), n_ctx, replicates=reps, seed=seed + 13, context_equal=False)
        b2 = D.bootstrap_mae(row_ctx, np.abs(delta - pl), n_ctx, replicates=reps, seed=seed + 14, context_equal=False)
        bd2 = D.bootstrap_diff(row_ctx, np.abs(delta - pr), np.abs(delta - pl), n_ctx, replicates=reps, seed=seed + 15, context_equal=False)
        # macro over applicable pairs
        app = [x for x in out if x.get("applicable")]
        macro_zero = float(np.mean([x["mae_zero"] for x in app])) if app else float("nan")
        macro_shared = float(np.mean([x["mae_shared"] for x in app])) if app else float("nan")
        macro_pair = float(np.mean([x["mae_pair_full"] for x in app if x["mae_pair_full"] is not None])) if app else float("nan")
        macro_local = float(np.mean([x["mae_local"] for x in app])) if app else float("nan")
        pool = {
            "n_pairs": len(app),
            "n_obs": int(len(delta)),
            "n_ctx": int(n_ctx),
            "mae_zero": b0["point"], "mae_zero_ci": [b0["lo"], b0["hi"]],
            "mae_shared": b1["point"], "mae_shared_ci": [b1["lo"], b1["hi"]],
            "relative_gain_shared_vs_zero": 1.0 - b1["point"] / max(b0["point"], 1e-12),
            "diff_zero_minus_shared": {"point": bd["diff"], "ci": [bd["lo"], bd["hi"]]},
            "sign_accuracy_shared": sign_accuracy(delta, pr),
            "mae_pair_full": m2["mae"],
            "gap_shared_vs_pair_full": m1["mae"] - m2["mae"],
            "mae_local": b2["point"], "mae_local_ci": [b2["lo"], b2["hi"]],
            "relative_gain_local_vs_zero": 1.0 - b2["point"] / max(b0["point"], 1e-12),
            "relative_gain_local_vs_shared": 1.0 - b2["point"] / max(b1["point"], 1e-12),
            "diff_shared_minus_local": {"point": bd2["diff"], "ci": [bd2["lo"], bd2["hi"]]},
            "local_coverage": float(np.mean([x["local_coverage"] for x in app])) if app else float("nan"),
            "macro_pair_equal": {
                "mae_zero": macro_zero, "mae_shared": macro_shared,
                "mae_local": macro_local, "mae_pair_full": macro_pair,
                "relative_gain": 1.0 - macro_shared / max(macro_zero, 1e-12),
                "relative_gain_local_vs_shared": 1.0 - macro_local / max(macro_shared, 1e-12),
            },
        }
    else:
        pool = {"n_pairs": 0, "n_obs": 0}
    return {"pairs": out, "pool": pool, "n_candidates": len(cand)}


def _pair_median(disc_pairs: dict[str, np.ndarray], a: int, b: int) -> float:
    rows = pair_rows(disc_pairs, a, b)
    if not np.any(rows):
        return float("nan")
    return float(np.median(disc_pairs["delta"][rows]))


def _write_csv(path: Path, natural: list[dict], masked: list[dict]) -> None:
    rows = []
    for x in natural:
        rows.append({"kind": "natural_unobserved", **x})
    for x in masked:
        rows.append({"kind": "controlled_masked", **x})
    if rows:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        C.write_csv(path, rows, columns=cols)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pcqm", "zinc"])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = C.load_config()
    out = Path(args.out) if args.out else C.ANALYSIS_DIR
    payload = run_p2(args.dataset, Path(args.prefix), out, cfg)
    print(json.dumps({k: payload[k] for k in (
        "dataset", "support_ok", "judgement", "reason",
        "natural_unobserved", "shared_fit")}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())