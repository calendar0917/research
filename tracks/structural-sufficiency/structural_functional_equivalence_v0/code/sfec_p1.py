"""Stage C: P1 response stability, transfer and functional-equivalence screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec_common as C
import sfec_data as D


def compact_ctx(row_ctx: np.ndarray) -> tuple[np.ndarray, int]:
    if len(row_ctx) == 0:
        return row_ctx.astype(np.int64), 0
    u, inv = np.unique(row_ctx.astype(np.int64), return_inverse=True)
    return inv.astype(np.int64), int(len(u))


def metric(delta: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - delta
    ae = np.abs(err)
    return {
        "n": int(len(delta)),
        "mae": float(ae.mean()) if len(ae) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "bias": float(err.mean()) if len(err) else float("nan"),
    }


def sign_accuracy(delta: np.ndarray, pred: np.ndarray) -> float:
    m = np.abs(delta) > 1e-12
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.sign(delta[m]) == np.sign(pred[m])))


def map_pair_values(
    eval_pairs: dict[str, np.ndarray], disc_stats: dict[str, np.ndarray], field: str
) -> np.ndarray:
    key = eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)
    pos = np.searchsorted(disc_stats["pair_key"], key)
    out = np.full(len(key), np.nan, dtype=np.float64)
    ok = (pos < len(disc_stats["pair_key"]))
    ok[ok] = disc_stats["pair_key"][pos[ok]] == key[ok]
    out[ok] = disc_stats[field][pos[ok]]
    return out


def map_pair_counts(
    eval_pairs: dict[str, np.ndarray], stats: dict[str, np.ndarray]
) -> np.ndarray:
    key = eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)
    return pair_counts_for_keys(key, stats)


def pair_counts_for_keys(key: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    key = np.asarray(key, dtype=np.int64)
    pos = np.searchsorted(stats["pair_key"], key)
    out = np.zeros(len(key), dtype=np.int64)
    ok = (pos < len(stats["pair_key"]))
    ok[ok] = stats["pair_key"][pos[ok]] == key[ok]
    out[ok] = stats["n"][pos[ok]]
    return out


def rank_key(rank: np.ndarray, s_table: dict[str, np.ndarray]) -> dict[int, str]:
    out = {}
    for sid, r in enumerate(rank.tolist()):
        if r >= 0:
            out[int(r)] = str(s_table["S_key"][sid])
    return out


def load_dataset(name: str, prefix: Path) -> dict[str, Any]:
    cells = D.load_cells(prefix)
    uni = D.load_universe(prefix)
    s_table = D.load_S(prefix)
    c_table = D.load_C(prefix, columns=["C_id", "q_id", "nC", "scaffold_id", "hash_mod", "hash64"])
    if name == "pcqm":
        labels = C.load_pcqm_labels()
        train = cells["split_id"] == 0
    elif name == "zinc":
        labels = C.zinc_labels("train")
        train = cells["split_id"] == 0
    else:
        raise ValueError(name)
    primary = uni["primary"]
    rank = uni["rank"]
    disc_mask = uni["discovery_mask"] & train
    eval_mask = uni["primary_evaluation_cells"] & train
    disc_ctx = D.build_contexts(cells, disc_mask, labels, len(s_table["S_id"]), primary, rank)
    eval_ctx = D.build_contexts(cells, eval_mask, labels, len(s_table["S_id"]), primary, rank)
    disc_pairs = D.build_pairs(disc_ctx, rank)
    eval_pairs = D.build_pairs(eval_ctx, rank)
    return {
        "name": name,
        "prefix": prefix,
        "cells": cells,
        "uni": uni,
        "s_table": s_table,
        "c_table": c_table,
        "labels": labels,
        "train": train,
        "primary": primary,
        "rank": rank,
        "disc_ctx": disc_ctx,
        "eval_ctx": eval_ctx,
        "disc_pairs": disc_pairs,
        "eval_pairs": eval_pairs,
    }


def run_p1(name: str, prefix: Path, out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    b = load_dataset(name, prefix)
    disc_pairs = b["disc_pairs"]
    eval_pairs = b["eval_pairs"]
    disc_stats = D.pair_stats(disc_pairs)
    eval_stats = D.pair_stats(eval_pairs)
    rng = np.random.default_rng(cfg["statistics"]["bootstrap_seed"])
    dd = D.pair_double_differences(disc_pairs, rng, cfg["statistics"]["max_context_pairs_for_D"])
    tau = float(cfg["datasets"][name]["tau"])
    taus = sorted(set([tau] + [float(x) for x in cfg["datasets"][name]["tau_sensitivity"]]))
    # supported pairs: >=10 discovery, >=5 primary evaluation
    disc_n = disc_stats["n"]
    eval_n_by_pair = pair_counts_for_keys(disc_stats["pair_key"], eval_stats)
    supported = (disc_n >= cfg["support_gates"]["p1_min_discovery_contexts"]) & (
        eval_n_by_pair >= cfg["support_gates"]["p1_min_evaluation_contexts"]
    )
    support_pass = bool(b["uni"]["support_gates"]["p1_pass"]) if "support_gates" in b["uni"] else False
    # support json is not in universe; read from prefix
    support_json = Path(str(prefix) + ".support.json")
    support_meta = C.read_json(support_json) if support_json.exists() else {}
    support_pass = bool(support_meta.get("support_gates", {}).get("p1_pass", False))
    # evaluation rows for pairs with at least one discovery observation
    mu = map_pair_values(eval_pairs, disc_stats, "median")
    disc_n_by_row = map_pair_counts(eval_pairs, disc_stats)
    base = np.isfinite(mu) & (disc_n_by_row >= 1)
    # main transfer subset: supported pairs
    key = eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)
    supported_keys = disc_stats["pair_key"][supported]
    is_supported_row = np.isin(key, supported_keys)
    # metrics
    transfer = {}
    for subset_name, subset in (
        ("all_with_discovery", base),
        ("supported", base & is_supported_row),
    ):
        if not np.any(subset):
            transfer[subset_name] = {"n": 0}
            continue
        delta = eval_pairs["delta"][subset]
        mu_s = mu[subset]
        row_ctx, n_ctx = compact_ctx(eval_pairs["ctx"][subset])
        res = {"n_obs": int(len(delta)), "n_ctx": int(n_ctx),
               "n_pairs": int(len(np.unique(key[subset])))}
        for weighting, ce in (("pair_equal", False), ("context_equal", True)):
            m0 = metric(delta, np.zeros_like(delta))
            m1 = metric(delta, mu_s)
            b0 = D.bootstrap_mae(row_ctx, np.abs(delta), n_ctx,
                                 replicates=cfg["statistics"]["bootstrap_replicates"],
                                 seed=cfg["statistics"]["bootstrap_seed"] + 1,
                                 context_equal=ce)
            b1 = D.bootstrap_mae(row_ctx, np.abs(delta - mu_s), n_ctx,
                                 replicates=cfg["statistics"]["bootstrap_replicates"],
                                 seed=cfg["statistics"]["bootstrap_seed"] + 2,
                                 context_equal=ce)
            bd = D.bootstrap_diff(row_ctx, np.abs(delta), np.abs(delta - mu_s), n_ctx,
                                  replicates=cfg["statistics"]["bootstrap_replicates"],
                                  seed=cfg["statistics"]["bootstrap_seed"] + 3,
                                  context_equal=ce)
            m0 = dict(m0); m1 = dict(m1)
            m0["mae"] = float(b0["point"]); m1["mae"] = float(b1["point"])
            res[weighting] = {
                "ZERO": {**m0, "ci": [b0["lo"], b0["hi"]]},
                "PAIR": {**m1, "ci": [b1["lo"], b1["hi"]],
                         "sign_accuracy": sign_accuracy(delta, mu_s),
                         "skill_vs_zero": 1.0 - m1["mae"] / max(m0["mae"], 1e-12)},
                "diff_ZERO_minus_PAIR": {"point": bd["diff"], "ci": [bd["lo"], bd["hi"]]},
            }
        # scaffold-cluster bootstrap sensitivity (PCQM contexts have scaffold_id)
        ctx_ids = np.unique(eval_pairs["ctx"][subset].astype(np.int64))
        scaf_ctx = b["c_table"]["scaffold_id"][b["eval_ctx"].C[ctx_ids]].astype(np.int64)
        uniq_scaf, scaf_inv = np.unique(scaf_ctx, return_inverse=True)
        if len(uniq_scaf) >= 2:
            err0 = np.abs(delta)
            err1 = np.abs(delta - mu_s)
            ctx_n = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
            ctx_e0 = np.bincount(row_ctx, weights=err0, minlength=n_ctx)
            ctx_e1 = np.bincount(row_ctx, weights=err1, minlength=n_ctx)
            n_per = np.bincount(scaf_inv, weights=ctx_n, minlength=len(uniq_scaf)).astype(np.float64)
            s0 = np.bincount(scaf_inv, weights=ctx_e0, minlength=len(uniq_scaf))
            s1 = np.bincount(scaf_inv, weights=ctx_e1, minlength=len(uniq_scaf))
            rg = np.random.default_rng(cfg["statistics"]["bootstrap_seed"] + 4)
            diffs = np.empty(cfg["statistics"]["bootstrap_replicates"], dtype=np.float64)
            m0s = np.empty_like(diffs); m1s = np.empty_like(diffs)
            for rep in range(len(diffs)):
                pick = rg.integers(0, len(uniq_scaf), size=len(uniq_scaf))
                den = max(float(n_per[pick].sum()), 1.0)
                m0s[rep] = float(s0[pick].sum()) / den
                m1s[rep] = float(s1[pick].sum()) / den
                diffs[rep] = m0s[rep] - m1s[rep]
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            res["scaffold_cluster"] = {
                "n_scaffolds": int(len(uniq_scaf)),
                "mae_zero": float(err0.mean()),
                "mae_pair": float(err1.mean()),
                "diff_zero_minus_pair": float(err0.mean() - err1.mean()),
                "diff_ci": [float(lo), float(hi)],
                "note": "scaffolds resampled with replacement; all pair observations of a selected scaffold are kept together",
            }
        transfer[subset_name] = res
    # candidate functional equivalence (frozen on discovery)
    candidates = {}
    for t in taus:
        cand = (disc_stats["p80_abs_dev"] <= t) & (disc_stats["n"] >= cfg["support_gates"]["p1_min_discovery_contexts"])
        rec = {
            "tau": t,
            "n_candidates": int(cand.sum()),
            "coverage_supported_pairs": float(cand.sum() / max(supported.sum(), 1)),
            "n_discovery_observations": int(disc_stats["n"][cand].sum()),
        }
        if cand.any():
            cand_keys = disc_stats["pair_key"][cand]
            rows = np.isin(key, cand_keys) & base
            if np.any(rows):
                d_eval = eval_pairs["delta"][rows]
                mu_eval = mu[rows]
                resid = np.abs(d_eval - mu_eval)
                rec.update({
                    "n_evaluation_observations": int(rows.sum()),
                    "fraction_within_tau": float(np.mean(resid <= t)),
                    "median_abs_residual": float(np.median(resid)),
                    "p80_abs_residual": float(np.quantile(resid, 0.80)),
                    "discovery_p80_median": float(np.median(disc_stats["p80_abs_dev"][cand])),
                    "mu_abs_median": float(np.median(np.abs(disc_stats["median"][cand]))),
                    "near_zero_candidates": int(np.sum(np.abs(disc_stats["median"][cand]) <= t)),
                    "non_zero_candidates": int(np.sum(np.abs(disc_stats["median"][cand]) > t)),
                })
            else:
                rec.update({"n_evaluation_observations": 0})
        candidates[str(t)] = rec
    # examples
    rk = rank_key(b["rank"], b["s_table"])
    examples = []
    if len(disc_stats["pair_key"]):
        for idx in np.argsort(-disc_stats["n"])[:20]:
            a = int(disc_stats["A"][idx]); bb = int(disc_stats["B"][idx])
            examples.append({
                "A_rank": a, "B_rank": bb,
                "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
                "n_discovery": int(disc_stats["n"][idx]),
                "median": float(disc_stats["median"][idx]),
                "mad": float(disc_stats["mad"][idx]),
                "p80_abs_dev": float(disc_stats["p80_abs_dev"][idx]),
                "sign_pos": float(disc_stats["sign_pos"][idx]),
                "n_evaluation": int(eval_n_by_pair[idx]),
            })
    # judgment
    supported_transfer = transfer.get("supported", {})
    pair_mae = supported_transfer.get("pair_equal", {}).get("PAIR", {}).get("mae", float("nan"))
    zero_mae = supported_transfer.get("pair_equal", {}).get("ZERO", {}).get("mae", float("nan"))
    pair_better = bool(np.isfinite(pair_mae) and np.isfinite(zero_mae) and pair_mae < zero_mae)
    cand10 = candidates[str(tau)]
    cand_cov = cand10.get("fraction_within_tau", float("nan"))
    if not support_pass:
        judgement = "INSUFFICIENT_CROSSED_SUPPORT"
        reason = "structural support gate failed"
    elif pair_better and cand10["n_candidates"] >= 10 and np.isfinite(cand_cov) and cand_cov >= 0.5:
        judgement = "REUSABLE_RESPONSE_SIGNAL"
        reason = "PAIR beats ZERO and discovery candidates retain evaluation residuals within tau"
    elif pair_better:
        judgement = "PARTIAL_RESPONSE_REUSE"
        reason = "general replacement direction transfers but stable candidate coverage is limited"
    else:
        judgement = "NO_SUPPORTED_RESPONSE_REUSE"
        reason = "PAIR did not beat ZERO on evaluation"
    payload = {
        "dataset": name,
        "prefix": str(prefix),
        "tau": tau,
        "n_discovery_contexts": int(b["disc_ctx"].n_ctx),
        "n_evaluation_contexts": int(b["eval_ctx"].n_ctx),
        "n_discovery_pair_observations": int(len(disc_pairs["delta"])),
        "n_evaluation_pair_observations": int(len(eval_pairs["delta"])),
        "n_pairs_discovery": int(len(disc_stats["pair_key"])),
        "n_pairs_evaluation": int(len(eval_stats["pair_key"])),
        "n_supported_pairs": int(supported.sum()),
        "support_gate": support_meta.get("support_gates", {}),
        "discovery": {
            "delta_median": float(np.median(disc_pairs["delta"])) if len(disc_pairs["delta"]) else None,
            "delta_mad": float(np.median(np.abs(disc_pairs["delta"] - np.median(disc_pairs["delta"])))) if len(disc_pairs["delta"]) else None,
            "abs_D": {k: dd[k] for k in ("median", "p90", "mean", "n", "total_possible", "sampled_pairs")},
            "mu_abs_median": float(np.median(np.abs(disc_stats["median"]))) if len(disc_stats["median"]) else None,
            "sign_pos_median": float(np.median(disc_stats["sign_pos"])) if len(disc_stats["sign_pos"]) else None,
        },
        "transfer": transfer,
        "candidates": candidates,
        "judgement": judgement,
        "reason": reason,
        "examples": examples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / f"p1_{name}.json", payload)
    _write_csvs(name, out_dir, disc_stats, eval_pairs, eval_stats, mu, b, tau, taus)
    return payload


def _write_csvs(name, out_dir, disc_stats, eval_pairs, eval_stats, mu, b, tau, taus):
    rk = rank_key(b["rank"], b["s_table"])
    rows = []
    for i, key in enumerate(disc_stats["pair_key"]):
        a = int(disc_stats["A"][i]); bb = int(disc_stats["B"][i])
        rows.append({
            "dataset": name, "A_rank": a, "B_rank": bb,
            "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
            "n_discovery": int(disc_stats["n"][i]),
            "mu_median": float(disc_stats["median"][i]),
            "mad": float(disc_stats["mad"][i]),
            "p80_abs_dev": float(disc_stats["p80_abs_dev"][i]),
            "sign_pos": float(disc_stats["sign_pos"][i]),
            "mean": float(disc_stats["mean"][i]),
            "sd": float(disc_stats["sd"][i]),
        })
    C.write_csv(out_dir / f"response_transfer_{name}.csv", rows)
    cand_rows = []
    for t in taus:
        cand = (disc_stats["p80_abs_dev"] <= t) & (disc_stats["n"] >= 10)
        for i in np.flatnonzero(cand):
            key = int(disc_stats["pair_key"][i])
            rows_eval = (eval_pairs["A"].astype(np.int64) * 64 + eval_pairs["B"].astype(np.int64)) == key
            a = int(disc_stats["A"][i]); bb = int(disc_stats["B"][i])
            if np.any(rows_eval):
                resid = np.abs(eval_pairs["delta"][rows_eval] - mu[rows_eval])
                frac = float(np.mean(resid <= t))
                med = float(np.median(resid))
                n_eval = int(rows_eval.sum())
            else:
                frac = float("nan"); med = float("nan"); n_eval = 0
            cand_rows.append({
                "dataset": name, "tau": t, "A_rank": a, "B_rank": bb,
                "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
                "n_discovery": int(disc_stats["n"][i]),
                "mu": float(disc_stats["median"][i]),
                "p80_abs_dev": float(disc_stats["p80_abs_dev"][i]),
                "abs_mu": float(abs(disc_stats["median"][i])),
                "n_evaluation": n_eval,
                "fraction_within_tau": frac,
                "median_eval_abs_resid": med,
            })
    C.write_csv(out_dir / f"functional_equivalence_candidates_{name}.csv", cand_rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pcqm", "zinc"])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = C.load_config()
    out = Path(args.out) if args.out else C.ANALYSIS_DIR
    payload = run_p1(args.dataset, Path(args.prefix), out, cfg)
    print(json.dumps({k: payload[k] for k in (
        "dataset", "n_discovery_contexts", "n_evaluation_contexts",
        "n_supported_pairs", "judgement", "reason")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())