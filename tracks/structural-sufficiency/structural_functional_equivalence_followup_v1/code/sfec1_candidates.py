"""Phase E: conditional functional-rule discovery on discovery labels only.

All candidate counts and coverages are computed over unique canonical
``(A,B,q)`` triples, never over repeated context rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D


def _eval_lookup(
    cand_keys: np.ndarray, pred: np.ndarray, eks: np.ndarray, ed: np.ndarray,
    ectx: np.ndarray, tau: float,
) -> dict[str, np.ndarray]:
    cand_keys = np.asarray(cand_keys, dtype=np.int64)
    lo = np.searchsorted(eks, cand_keys, side="left")
    hi = np.searchsorted(eks, cand_keys, side="right")
    n = hi - lo
    out = {
        "n_eval": n.astype(np.int64),
        "n_eval_contexts": np.zeros(len(cand_keys), dtype=np.int64),
        "eval_median": np.full(len(cand_keys), np.nan),
        "eval_frac_within_tau": np.full(len(cand_keys), np.nan),
        "eval_pred_error_median": np.full(len(cand_keys), np.nan),
        "eval_mae": np.full(len(cand_keys), np.nan),
    }
    for i, (a, b) in enumerate(zip(lo.tolist(), hi.tolist())):
        if b <= a:
            continue
        vals = ed[a:b]
        ctx = ectx[a:b]
        out["n_eval_contexts"][i] = len(np.unique(ctx))
        out["eval_median"][i] = float(np.median(vals))
        err = np.abs(vals - float(pred[i]))
        out["eval_frac_within_tau"][i] = float(np.mean(err <= tau))
        out["eval_pred_error_median"][i] = float(np.median(err))
        out["eval_mae"][i] = float(np.mean(err))
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    b = D.load_bundle(cfg)
    prep = {k: v for k, v in np.load(C.DATA_DIR / "prepared_pcqm.npz").items()}
    n_q = int(prep["n_q"][0])
    q_disc = b.C["q_id"][b.ctx_disc.C].astype(np.int64)
    q_eval = b.C["q_id"][b.ctx_eval.C].astype(np.int64)
    q_keys = C.read_json(Path(str(b.prefix) + ".keys.json"))["q_keys"]
    pd = b.pairs_disc
    A = pd["A"].astype(np.int64)
    B = pd["B"].astype(np.int64)
    key = D.pair_key(A, B)
    q = q_disc[pd["ctx"]]
    tri = {
        "key": prep["triple_true_key"], "n": prep["triple_true_n"],
        "median": prep["triple_true_median"], "mean": prep["triple_true_mean"],
        "p80_abs_dev": prep["triple_true_p80"],
    }
    tri_key = tri["key"].astype(np.int64)
    tri_n = tri["n"].astype(np.int64)
    tri_med = tri["median"].astype(np.float64)
    tri_mean = tri["mean"].astype(np.float64)
    tri_p80 = tri["p80_abs_dev"].astype(np.float64)
    tri_pair = tri_key // n_q
    tri_q = tri_key % n_q
    tri_A = tri_pair // 64
    tri_B = tri_pair % 64
    tri_supported = tri_n >= int(cfg["support"]["conditional_min_discovery_contexts"])
    # row-level lookup only for dispersion
    row_med, row_cnt = D.lookup_triple(tri, A, B, q, n_q, "median")
    row_supported = row_cnt >= int(cfg["support"]["conditional_min_discovery_contexts"])
    ds = {k: prep[f"disc_{k}"] for k in
          ("pair_key", "A", "B", "n", "median", "mean", "mad", "p80_abs_dev", "sd", "sign_pos")}
    pair_med = D.lookup_pair(ds, key, "median")
    # evaluation rows
    evA = prep["eval_A"].astype(np.int64)
    evB = prep["eval_B"].astype(np.int64)
    ev_ctx = prep["eval_ctx"]
    ev_delta = prep["eval_delta"]
    ev_key = D.pair_key(evA, evB)
    ev_q = q_eval[ev_ctx]
    evk = ev_key * n_q + ev_q
    order = np.argsort(evk, kind="stable")
    eks = evk[order]
    ed = ev_delta[order]
    ectx = ev_ctx[order]
    n_eval_main = int(prep["eval_main"].sum())
    n_eval_all = int(len(ev_key))
    n_eval_ctx_by_pair = np.bincount(ev_key, minlength=int(ev_key.max()) + 1)
    # dispersion
    res_global = pd["delta"] - pair_med
    res_cond = pd["delta"] - row_med
    disp = {
        "n_supported_rows": int(row_supported.sum()),
        "n_supported_triples": int(tri_supported.sum()),
        "n_pairs_with_supported_rows": int(len(np.unique(key[row_supported]))),
        "pooled_p80_abs_global": float(np.quantile(np.abs(res_global[row_supported]), 0.80)),
        "pooled_p80_abs_conditional": float(np.quantile(np.abs(res_cond[row_supported]), 0.80)),
        "pooled_median_abs_global": float(np.median(np.abs(res_global[row_supported]))),
        "pooled_median_abs_conditional": float(np.median(np.abs(res_cond[row_supported]))),
    }
    disp["pooled_p80_reduction"] = 1.0 - disp["pooled_p80_abs_conditional"] / max(disp["pooled_p80_abs_global"], 1e-12)
    disp["pooled_median_abs_reduction"] = 1.0 - disp["pooled_median_abs_conditional"] / max(disp["pooled_median_abs_global"], 1e-12)
    per_pair = []
    for k in np.unique(key[row_supported]).tolist():
        m = (key == k) & row_supported
        if int(m.sum()) < 10:
            continue
        pg = float(np.quantile(np.abs(res_global[m]), 0.80))
        pc = float(np.quantile(np.abs(res_cond[m]), 0.80))
        per_pair.append((int(k), int(m.sum()), pg, pc, 1.0 - pc / max(pg, 1e-12),
                         float(np.median(np.abs(res_cond[m])))))
    pp = np.asarray([x[4] for x in per_pair]) if per_pair else np.zeros(0)
    disp.update({
        "n_pairs_ge_10_supported_rows": int(len(per_pair)),
        "per_pair_p80_reduction_median": float(np.median(pp)) if len(pp) else None,
        "per_pair_p80_reduction_q25": float(np.quantile(pp, 0.25)) if len(pp) else None,
        "per_pair_p80_reduction_q75": float(np.quantile(pp, 0.75)) if len(pp) else None,
        "fraction_pairs_improved": float(np.mean(pp > 0)) if len(pp) else None,
    })
    C.write_csv(C.ANALYSIS_DIR / "conditional_dispersion_by_pair.csv", [
        {"pair_key": x[0], "A_rank": x[0] // 64, "B_rank": x[0] % 64,
         "n_supported_rows": x[1], "p80_global": x[2], "p80_conditional": x[3],
         "p80_reduction": x[4], "median_abs_conditional": x[5]}
        for x in per_pair
    ])
    # sign reversals over unique supported triples
    reversal = {}
    for thr in (0.05, 0.10):
        pairs = []
        n_groups = 0
        n_ver = 0
        n_rep = 0
        n_pairs_any = 0
        n_pairs_both = 0
        for k in np.unique(tri_pair[tri_supported]).tolist():
            m = (tri_pair == k) & tri_supported
            meds = tri_med[m]
            qs = tri_q[m]
            pos = np.unique(qs[meds > thr])
            neg = np.unique(qs[meds < -thr])
            if len(pos) == 0 or len(neg) == 0:
                continue
            rec = {
                "pair_key": int(k), "A_rank": int(k) // 64, "B_rank": int(k) % 64,
                "n_pos_q": int(len(pos)), "n_neg_q": int(len(neg)),
                "min_median": float(meds.min()), "max_median": float(meds.max()),
            }
            ok_pos = 0
            ok_neg = 0
            any_ver = False
            group_rows = []
            for qq in np.r_[pos, neg].tolist():
                t = (tri_pair == k) & (tri_q == qq) & tri_supported
                mm = float(tri_med[t][0])
                eg_lo = int(np.searchsorted(eks, int(k) * n_q + int(qq), side="left"))
                eg_hi = int(np.searchsorted(eks, int(k) * n_q + int(qq), side="right"))
                n_groups += 1
                g = {"q": int(qq), "discovery_median": mm, "n_eval": eg_hi - eg_lo,
                     "n_eval_contexts": 0, "eval_median": None, "reproduced": None}
                if eg_hi > eg_lo:
                    vals = ed[eg_lo:eg_hi]
                    ctxs = ectx[eg_lo:eg_hi]
                    g["n_eval_contexts"] = int(len(np.unique(ctxs)))
                    g["eval_median"] = float(np.median(vals))
                    if g["n_eval_contexts"] >= 5:
                        n_ver += 1
                        any_ver = True
                        g["reproduced"] = bool(np.sign(g["eval_median"]) == np.sign(mm))
                        if g["reproduced"]:
                            n_rep += 1
                            if qq in set(pos.tolist()):
                                ok_pos += 1
                            else:
                                ok_neg += 1
                group_rows.append(g)
            rec["groups"] = group_rows
            if any_ver:
                n_pairs_any += 1
                if ok_pos > 0 and ok_neg > 0:
                    n_pairs_both += 1
                    rec["both_signs_reproduced"] = True
                else:
                    rec["both_signs_reproduced"] = False
            pairs.append(rec)
        reversal[str(thr)] = {
            "threshold": thr,
            "n_reversal_pairs": len(pairs),
            "n_groups": n_groups,
            "n_verifiable_groups": n_ver,
            "n_reproduced_groups": n_rep,
            "group_reproduction_rate": n_rep / max(n_ver, 1),
            "n_pairs_with_any_verifiable_group": n_pairs_any,
            "n_pairs_both_signs_reproduced": n_pairs_both,
            "pairs": pairs,
        }
    # candidates at three tolerances, unique triples
    tau0 = float(cfg["judgement"]["conditional_candidate_tau"])
    taus = sorted(set([tau0] + [float(x) for x in cfg["judgement"]["conditional_candidate_tau_sensitivity"]]))
    candidate_rows = []
    summaries = {}
    example_rows = []
    for tau in taus:
        sel = tri_supported & (tri_p80 <= tau)
        idx = np.flatnonzero(sel)
        keys = tri_key[idx]
        pred = tri_med[idx]
        stats = _eval_lookup(keys, pred, eks, ed, ectx, tau)
        verifiable = stats["n_eval_contexts"] >= int(cfg["support"]["conditional_min_evaluation_contexts"])
        near = np.abs(tri_med[idx]) <= tau
        recs = []
        for j, ii in enumerate(idx.tolist()):
            recs.append({
                "tau": tau, "A_rank": int(tri_A[ii]), "B_rank": int(tri_B[ii]),
                "A_key": "", "B_key": "", "q_id": int(tri_q[ii]),
                "q": str(q_keys[int(tri_q[ii])]) if int(tri_q[ii]) < len(q_keys) else "",
                "n_discovery": int(tri_n[ii]),
                "discovery_median": float(tri_med[ii]),
                "discovery_mean": float(tri_mean[ii]),
                "discovery_p80_abs_dev": float(tri_p80[ii]),
                "abs_median": float(abs(tri_med[ii])),
                "near_zero": bool(near[j]),
                "n_eval": int(stats["n_eval"][j]),
                "n_eval_contexts": int(stats["n_eval_contexts"][j]),
                "verifiable": bool(verifiable[j]),
                "eval_median": float(stats["eval_median"][j]) if stats["n_eval"][j] else None,
                "eval_frac_within_tau": float(stats["eval_frac_within_tau"][j]) if stats["n_eval"][j] else None,
                "eval_pred_error_median": float(stats["eval_pred_error_median"][j]) if stats["n_eval"][j] else None,
                "eval_mae": float(stats["eval_mae"][j]) if stats["n_eval"][j] else None,
            })
        def coverage(ver_only: bool):
            sel_cov = verifiable if ver_only else np.ones(len(recs), bool)
            cov_keys = np.unique(keys[sel_cov]) if np.any(sel_cov) else np.zeros(0, np.int64)
            cov = np.isin(evk, cov_keys)
            return int(cov.sum()), float(cov.mean()), int(len(np.unique(ev_key[cov])))
        cov_all = coverage(False)
        cov_ver = coverage(True)
        g = (ds["n"] >= int(cfg["support"]["conditional_min_discovery_contexts"])) & (ds["p80_abs_dev"] <= tau)
        gkeys = ds["pair_key"][g]
        gcov = np.isin(ev_key, gkeys)
        g_ver = np.asarray([int(n_eval_ctx_by_pair[int(x)]) >= 5 for x in gkeys.tolist()]) if len(gkeys) else np.zeros(0, bool)
        gsum = {
            "n_candidates": int(g.sum()),
            "n_verifiable": int(g_ver.sum()),
            "n_pairs": int(len(np.unique(gkeys))),
            "eval_obs_coverage": int(gcov.sum()),
            "eval_obs_coverage_fraction": float(gcov.mean()),
            "eval_pair_coverage": int(len(np.unique(ev_key[gcov]))),
        }
        summaries[str(tau)] = {
            "tau": tau,
            "n_candidates": int(sel.sum()),
            "n_verifiable_candidates": int(verifiable.sum()),
            "n_candidate_pairs": int(len(np.unique(tri_pair[idx]))),
            "n_verifiable_candidate_pairs": int(len(np.unique(tri_pair[idx][verifiable]))),
            "candidate_pair_type_coverage_all_pairs": float(len(np.unique(tri_pair[idx])) / max(len(ds["pair_key"]), 1)),
            "candidate_pair_type_coverage_supported_pairs": float(len(np.unique(tri_pair[idx])) / max(int(np.sum(ds["n"] >= 10)), 1)),
            "eval_obs_coverage_all": cov_all[0],
            "eval_obs_coverage_all_fraction": cov_all[1],
            "eval_pair_coverage_all": cov_all[2],
            "eval_obs_coverage_verifiable": cov_ver[0],
            "eval_obs_coverage_verifiable_fraction": cov_ver[1],
            "eval_pair_coverage_verifiable": cov_ver[2],
            "eval_obs_fraction_of_main": cov_all[0] / max(n_eval_main, 1),
            "n_near_zero": int(near.sum()),
            "n_nonzero": int((~near).sum()),
            "n_verifiable_near_zero": int((near & verifiable).sum()),
            "n_verifiable_nonzero": int((~near & verifiable).sum()),
            "global_pair_candidates": gsum,
        }
        if abs(tau - tau0) < 1e-12:
            near_pos = [r for r in recs if r["near_zero"] and r["verifiable"] and r["n_eval"] >= 10
                        and (r["eval_frac_within_tau"] or 0) >= 0.75]
            near_neg = [r for r in recs if r["near_zero"] and r["verifiable"] and
                        (r["eval_frac_within_tau"] or 0) < 0.5]
            nonzero = [r for r in recs if not r["near_zero"] and r["verifiable"]
                       and (r["eval_frac_within_tau"] or 0) >= 0.6
                       and r["eval_median"] is not None
                       and np.sign(r["eval_median"]) == np.sign(r["discovery_median"])]
            near_pos.sort(key=lambda x: (-x["n_discovery"], x["A_rank"], x["B_rank"], x["q_id"]))
            near_neg.sort(key=lambda x: (-x["n_discovery"], x["A_rank"], x["B_rank"], x["q_id"]))
            nonzero.sort(key=lambda x: (-abs(x["discovery_median"]), -x["n_discovery"]))
            picked = []
            picked.extend([("conditional_near_zero_positive", r) for r in near_pos[:4]])
            picked.extend([("conditional_near_zero_failure", r) for r in near_neg[:1]])
            picked.extend([("conditional_nonzero_stable", r) for r in nonzero[:4]])
            rev_ok = [r for r in reversal["0.05"]["pairs"] if r.get("both_signs_reproduced")]
            rev_ok.sort(key=lambda x: (-abs(x["max_median"] - x["min_median"]), x["pair_key"]))
            for rec in rev_ok[:2]:
                for g in rec["groups"]:
                    if g.get("reproduced") and g["n_eval_contexts"] >= 5:
                        picked.append(("sign_reversal_group", {
                            "A_rank": rec["A_rank"], "B_rank": rec["B_rank"],
                            "q_id": g["q"],
                            "q": str(q_keys[g["q"]]) if g["q"] < len(q_keys) else "",
                            "n_discovery": None, "discovery_median": g["discovery_median"],
                            "discovery_p80_abs_dev": None, "n_eval": g["n_eval"],
                            "n_eval_contexts": g["n_eval_contexts"],
                            "eval_median": g["eval_median"],
                            "eval_frac_within_tau": None,
                            "near_zero": abs(g["discovery_median"]) <= tau0,
                            "verifiable": True,
                        }))
            for etype, r in picked[:10]:
                example_rows.append({"example_type": etype, **r})
        candidate_rows.extend(recs)
    rk = D.rank_key(b.rank, b.S)
    for r in candidate_rows:
        r["A_key"] = rk.get(r["A_rank"], "")
        r["B_key"] = rk.get(r["B_rank"], "")
    for r in example_rows:
        r["A_key"] = rk.get(r["A_rank"], "")
        r["B_key"] = rk.get(r["B_rank"], "")
    C.write_csv(C.ANALYSIS_DIR / "conditional_function_candidates.csv", candidate_rows)
    C.write_csv(C.ANALYSIS_DIR / "conditional_examples.csv", example_rows)
    payload = {
        "tau_primary": tau0,
        "dispersion": disp,
        "sign_reversal": {k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in reversal.items()},
        "sign_reversal_examples": {k: [x for x in v["pairs"] if x.get("both_signs_reproduced")][:10]
                                   for k, v in reversal.items()},
        "candidates": summaries,
        "n_eval_rows_all": n_eval_all,
        "n_eval_rows_main": n_eval_main,
        "notes": "Candidate counts are unique (A,B,q) triples. Discovery uses discovery labels only; evaluation labels are used only for verification.",
    }
    C.write_json(C.ANALYSIS_DIR / "conditional_function_candidates.json", payload)
    return payload


def main() -> int:
    cfg = C.load_config()
    payload = run(cfg)
    print(json.dumps({
        "dispersion": payload["dispersion"],
        "sign_reversal": payload["sign_reversal"],
        "candidate_summary": {k: {kk: v[kk] for kk in (
            "n_candidates", "n_verifiable_candidates", "n_candidate_pairs",
            "n_verifiable_candidate_pairs", "eval_obs_coverage_all",
            "eval_obs_coverage_all_fraction", "eval_obs_coverage_verifiable",
            "eval_obs_coverage_verifiable_fraction", "n_near_zero", "n_nonzero",
            "n_verifiable_near_zero", "n_verifiable_nonzero")}
            for k, v in payload["candidates"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
