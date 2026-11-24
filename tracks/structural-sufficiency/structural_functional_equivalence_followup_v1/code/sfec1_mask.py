"""Phase C: strict removal of direct molecular supervision and refit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D

MODELS = ["ZERO", "SHARED_MASKED", "LOCAL_MASKED", "LOCAL_NULL_MASKED",
          "PAIR_FULL", "PAIR_Q_FULL"]
COMPARISONS = [
    ("LOCAL_MASKED", "SHARED_MASKED"),
    ("LOCAL_MASKED", "LOCAL_NULL_MASKED"),
    ("LOCAL_MASKED", "ZERO"),
    ("LOCAL_MASKED", "PAIR_FULL"),
    ("LOCAL_MASKED", "PAIR_Q_FULL"),
    ("SHARED_MASKED", "ZERO"),
]


def _cell_molecule_lookup(b: D.Bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = b.cells
    n_s = len(b.S["S_id"])
    idx = np.flatnonzero(
        b.uni["discovery_mask"] & (cells["split_id"] == 0) & b.primary[cells["S_id"]]
    )
    key = (cells["C_id"][idx].astype(np.int64) * n_s
           + cells["S_id"][idx].astype(np.int64)) * 8 + cells["bond_type"][idx].astype(np.int64)
    mol = cells["mol_id"][idx].astype(np.int64)
    order = np.argsort(key, kind="stable")
    return key[order], mol[order], idx


def _mols_for_cell(
    ks: np.ndarray, mo: np.ndarray, c: int, sid: int, bt: int, n_s: int
) -> np.ndarray:
    k = (int(c) * int(n_s) + int(sid)) * 8 + int(bt)
    lo = int(np.searchsorted(ks, k, side="left"))
    hi = int(np.searchsorted(ks, k, side="right"))
    return mo[lo:hi]


def _metrics_block(
    delta: np.ndarray, preds: dict[str, np.ndarray], row_ctx: np.ndarray,
    pair_idx: np.ndarray, n_ctx: int, cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reps = int(cfg["statistics"]["bootstrap_replicates"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    points = {}
    for name, pred in preds.items():
        err = np.asarray(pred, np.float64) - delta
        ae = np.abs(err)
        sq = err ** 2
        base = D.basic_metric(delta, pred)
        cnt = np.bincount(row_ctx, minlength=n_ctx).astype(np.float64)
        ctx_ae = np.bincount(row_ctx, weights=ae, minlength=n_ctx)
        ctx_sq = np.bincount(row_ctx, weights=sq, minlength=n_ctx)
        ok = cnt > 0
        pair_sums = np.bincount(pair_idx, weights=ae)
        pair_cnt = np.bincount(pair_idx)
        pok = pair_cnt > 0
        points[name] = {
            "observation": {"mae": base["mae"], "rmse": base["rmse"],
                            "sign_accuracy": base["sign_accuracy"], "bias": base["bias"]},
            "context": {
                "mae": float(np.mean(ctx_ae[ok] / cnt[ok])),
                "rmse": float(np.sqrt(np.mean(ctx_sq[ok] / cnt[ok]))),
                "sign_accuracy": base["sign_accuracy"],
            },
            "pair_type": {
                "mae": float(np.mean(pair_sums[pok] / pair_cnt[pok])),
                "rmse": float(np.sqrt(np.mean(
                    np.bincount(pair_idx, weights=sq)[pok] / pair_cnt[pok]
                ))),
                "sign_accuracy": base["sign_accuracy"],
            },
        }
    boot_obs = D.bootstrap_obs_context(
        delta, preds, row_ctx, n_ctx, context_equal=False, reps=reps,
        seed=seed + 201, comparisons=COMPARISONS,
    )
    boot_ctx = D.bootstrap_obs_context(
        delta, preds, row_ctx, n_ctx, context_equal=True, reps=reps,
        seed=seed + 202, comparisons=COMPARISONS,
    )
    boot_pair = D.bootstrap_pair_macro(
        delta, preds, row_ctx, pair_idx, n_ctx, reps=reps,
        seed=seed + 203, comparisons=COMPARISONS,
    )
    result = {"points": points, "bootstrap": {
        "observation": boot_obs, "context": boot_ctx, "pair_type": boot_pair,
    }}
    rows = []
    for name in MODELS:
        for weighting, key in (("observation", "observation"), ("context_equal", "context"),
                               ("pair_type_equal", "pair_type")):
            b = boot_obs["models"].get(name) if weighting == "observation" else (
                boot_ctx["models"].get(name) if weighting == "context_equal" else boot_pair["models"].get(name)
            )
            rows.append({
                "row_kind": "model", "weighting": weighting, "model": name,
                "comparison": "", "mae": points[name][key]["mae"],
                "rmse": points[name][key]["rmse"],
                "sign_accuracy": points[name][key]["sign_accuracy"],
                "ci_lo": b["ci"][0], "ci_hi": b["ci"][1],
                "relative_gain_vs_first": "",
            })
    for a, bb in COMPARISONS:
        for weighting, boot, key in (
            ("observation", boot_obs, "observation"),
            ("context_equal", boot_ctx, "context"),
            ("pair_type_equal", boot_pair, "pair_type"),
        ):
            k = f"{a}_minus_{bb}"
            d = boot["diffs"][k]
            ma = points[a][key]["mae"]
            mb = points[bb][key]["mae"]
            rows.append({
                "row_kind": "paired_diff", "weighting": weighting, "model": "",
                "comparison": k, "mae": d["point"], "rmse": "",
                "sign_accuracy": "", "ci_lo": d["ci"][0], "ci_hi": d["ci"][1],
                "relative_gain_vs_first": 1.0 - mb / max(ma, 1e-12),
            })
    by_pair = []
    return result, rows, by_pair


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    b = D.load_bundle(cfg)
    q_id = b.C["q_id"]
    q_eval = q_id[b.ctx_eval.C].astype(np.int64)
    n_q = int(max(q_id[b.ctx_disc.C].max(), q_eval.max()) + 1)
    n_s = len(b.S["S_id"])
    pd = b.pairs_disc
    pe = b.pairs_eval
    dc = D.pair_count_matrix(pd)
    ec = D.pair_count_matrix(pe)
    rk = D.rank_key(b.rank, b.S)
    r2s = D.rank_to_sid(b.rank)
    cand = []
    for a in range(64):
        for bb in range(a + 1, 64):
            if dc[a, bb] > 0 and ec[a, bb] >= int(cfg["support"]["strict_mask_min_eval_contexts"]):
                cand.append((a, bb, int(dc[a, bb]), int(ec[a, bb])))
    cand.sort(key=lambda x: (-x[2], -x[3], rk.get(x[0], ""), rk.get(x[1], "")))
    selected = cand[: int(cfg["support"]["strict_mask_max_pairs"])]
    ks, mo, disc_rows = _cell_molecule_lookup(b)
    cells = b.cells
    labels = b.labels
    train = cells["split_id"] == 0
    # frozen unmasked triple statistics for reference-only PAIR_Q_FULL
    prep = np.load(C.DATA_DIR / "prepared_pcqm.npz")
    tri = {
        "key": prep["triple_true_key"], "n": prep["triple_true_n"],
        "median": prep["triple_true_median"], "mean": prep["triple_true_mean"],
        "p80_abs_dev": prep["triple_true_p80"],
    }
    n_mol = int(cells["mol_id"].max()) + 1
    pair_records: list[dict[str, Any]] = []
    eval_parts = {name: [] for name in MODELS}
    delta_parts = []
    ctx_parts = []
    pair_parts = []
    q_parts = []
    evaluable = 0
    for pidx, (a, bb, dn, en) in enumerate(selected):
        rows = (pd["A"] == a) & (pd["B"] == bb)
        direct_ctx = np.unique(pd["ctx"][rows])
        b_sid = int(r2s[bb])
        removed = np.zeros(n_mol, dtype=bool)
        n_b_cells = 0
        for ci in direct_ctx.tolist():
            lo, hi = int(b.ctx_disc.off[ci]), int(b.ctx_disc.off[ci + 1])
            pos = np.flatnonzero(b.ctx_disc.S[lo:hi] == b_sid)
            if not len(pos):
                continue
            n_b_cells += 1
            c = int(b.ctx_disc.C[ci])
            bt = int(b.ctx_disc.cell_b[lo + int(pos[0])])
            mm = _mols_for_cell(ks, mo, c, b_sid, bt, n_s)
            removed[mm] = True
        n_removed_mols = int(removed.sum())
        disc_mask = b.uni["discovery_mask"] & train & ~removed[cells["mol_id"].astype(np.int64)]
        n_removed_rows = int(np.sum(b.uni["discovery_mask"] & train)) - int(disc_mask.sum())
        ctx_m = D.build_contexts(cells, disc_mask, labels, n_s, b.primary, b.rank)
        q_m = q_id[ctx_m.C].astype(np.int64)
        design_m = D.make_design(ctx_m, b.rank, q_m)
        masked_fit = D.fit_shared(design_m)
        local_fit = D.fit_local_ctx(
            design_m, masked_fit["b"], n_q=n_q,
            lam=float(cfg["estimators"]["local_lambda"]),
            min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
            q_ctx_all=q_m,
        )
        q_null_m = D.make_null_q(ctx_m, q_m, int(cfg["estimators"]["null_seed"]))
        local_null_fit = D.fit_local_ctx(
            design_m, masked_fit["b"], n_q=n_q,
            lam=float(cfg["estimators"]["local_lambda"]),
            min_contexts=int(cfg["estimators"]["local_min_contexts_per_q"]),
            q_ctx_all=q_null_m,
        )
        # verification
        pairs_m = D.build_pairs(ctx_m, b.rank, canonical=True)
        direct_after = int(np.sum((pairs_m["A"] == a) & (pairs_m["B"] == bb)))
        left_rows = cells["mol_id"][np.flatnonzero(disc_mask)].astype(np.int64)
        mols_left = int(np.sum(removed[left_rows]))
        connected = bool(masked_fit["comp"][a] >= 0 and masked_fit["comp"][a] == masked_fit["comp"][bb])
        erows = (pe["A"] == a) & (pe["B"] == bb)
        delta = pe["delta"][erows]
        ectx = pe["ctx"][erows]
        n_eval_ctx = int(len(np.unique(ectx)))
        pred_shared = np.full(len(delta), masked_fit["b"][bb] - masked_fit["b"][a])
        pred_local = D.predict_local(
            ectx, pe["A"][erows], pe["B"][erows], q_eval, masked_fit["b"],
            {"comp": local_fit["comp"], "supported": local_fit["supported"],
             "d": local_fit["d"]},
        )
        pred_null = D.predict_local(
            ectx, pe["A"][erows], pe["B"][erows], q_eval, masked_fit["b"],
            {"comp": local_null_fit["comp"], "supported": local_null_fit["supported"],
             "d": local_null_fit["d"]},
        )
        mu_full = float(np.median(pd["delta"][rows])) if np.any(rows) else float("nan")
        q_row = q_eval[ectx]
        tmed, tcnt = D.lookup_triple(tri, pe["A"][erows], pe["B"][erows], q_row, n_q, "median")
        pair_q_full = np.where(tcnt >= int(cfg["support"]["conditional_min_discovery_contexts"]),
                               tmed, mu_full)
        preds_pair = {
            "ZERO": np.zeros(len(delta)),
            "SHARED_MASKED": pred_shared,
            "LOCAL_MASKED": pred_local["pred"],
            "LOCAL_NULL_MASKED": pred_null["pred"],
            "PAIR_FULL": np.full(len(delta), mu_full),
            "PAIR_Q_FULL": pair_q_full,
        }
        rec = {
            "A_rank": a, "B_rank": bb, "A_key": rk.get(a, ""), "B_key": rk.get(bb, ""),
            "n_discovery_direct": int(dn),
            "n_evaluation": int(len(delta)),
            "n_evaluation_contexts": n_eval_ctx,
            "direct_contexts_masked": int(len(direct_ctx)),
            "n_B_cells_deleted": int(n_b_cells),
            "n_molecules_removed": n_removed_mols,
            "n_discovery_cut_rows_removed": n_removed_rows,
            "n_discovery_contexts_remaining": int(ctx_m.n_ctx),
            "n_discovery_contexts_k_ge_2_remaining": int((ctx_m.k >= 2).sum()),
            "direct_cooccurrence_after_mask": direct_after,
            "removed_molecule_cells_left_in_fit": mols_left,
            "connected_after_mask": connected,
            "local_coverage": float(np.mean(pred_local["enabled"])),
            "local_null_coverage": float(np.mean(pred_null["enabled"])),
        }
        for name, pred in preds_pair.items():
            rec[f"mae_{name}"] = float(np.mean(np.abs(pred - delta)))
            rec[f"rmse_{name}"] = float(np.sqrt(np.mean((pred - delta) ** 2)))
            rec[f"sign_{name}"] = D.basic_metric(delta, pred)["sign_accuracy"]
        rec["relative_gain_local_vs_shared"] = (
            1.0 - rec["mae_LOCAL_MASKED"] / max(rec["mae_SHARED_MASKED"], 1e-12)
        )
        rec["relative_gain_local_vs_null"] = (
            1.0 - rec["mae_LOCAL_MASKED"] / max(rec["mae_LOCAL_NULL_MASKED"], 1e-12)
        )
        rec["relative_gain_local_vs_zero"] = (
            1.0 - rec["mae_LOCAL_MASKED"] / max(rec["mae_ZERO"], 1e-12)
        )
        ok = (direct_after == 0 and mols_left == 0 and connected and n_eval_ctx >= 5)
        rec["evaluable"] = bool(ok)
        if not ok:
            rec["reason"] = (
                "disconnected after masking" if not connected else
                ("direct co-occurrence remains" if direct_after else
                 ("removed molecule cells remain" if mols_left else "fewer than 5 evaluation contexts"))
            )
        pair_records.append(rec)
        if ok:
            evaluable += 1
            for name in MODELS:
                eval_parts[name].append(preds_pair[name])
            delta_parts.append(delta)
            ctx_parts.append(ectx)
            pair_parts.append(np.full(len(delta), evaluable - 1, dtype=np.int64))
            q_parts.append(q_row)
    if delta_parts:
        delta_all = np.concatenate(delta_parts)
        ctx_all = np.concatenate(ctx_parts)
        pair_idx = np.concatenate(pair_parts)
        preds_all = {name: np.concatenate(eval_parts[name]) for name in MODELS}
        row_ctx, n_ctx = D.compact_ctx(ctx_all)
        result, rows, _ = _metrics_block(delta_all, preds_all, row_ctx, pair_idx, n_ctx, cfg)
        # leave-one-pair-out
        loo = []
        for p in range(evaluable):
            m = pair_idx != p
            ms = float(np.mean(np.abs(preds_all["SHARED_MASKED"][m] - delta_all[m])))
            ml = float(np.mean(np.abs(preds_all["LOCAL_MASKED"][m] - delta_all[m])))
            mn = float(np.mean(np.abs(preds_all["LOCAL_NULL_MASKED"][m] - delta_all[m])))
            loo.append({
                "removed_pair": p,
                "removed_A_key": [x["A_key"] for x in pair_records if x.get("evaluable")][p],
                "removed_B_key": [x["B_key"] for x in pair_records if x.get("evaluable")][p],
                "mae_shared": ms, "mae_local": ml, "mae_local_null": mn,
                "relative_gain_local_vs_shared": 1.0 - ml / max(ms, 1e-12),
                "relative_gain_local_vs_null": 1.0 - ml / max(mn, 1e-12),
            })
        # pair contribution to pooled observation MAE gain
        contrib = []
        for p in range(evaluable):
            m = pair_idx == p
            ms = float(np.mean(np.abs(preds_all["SHARED_MASKED"][m] - delta_all[m])))
            ml = float(np.mean(np.abs(preds_all["LOCAL_MASKED"][m] - delta_all[m])))
            contrib.append((ms - ml) * int(m.sum()) / max(int(len(delta_all)), 1))
        total_gain = float(np.mean(np.abs(preds_all["SHARED_MASKED"] - delta_all))
                           - np.mean(np.abs(preds_all["LOCAL_MASKED"] - delta_all)))
        max_share = (max(contrib) / total_gain) if total_gain > 0 else float("nan")
        result.update({
            "n_evaluable_pairs": int(evaluable),
            "n_selected_pairs": int(len(selected)),
            "n_obs": int(len(delta_all)),
            "n_contexts": int(n_ctx),
            "leave_one_pair_out": loo,
            "loo_min_relative_gain_local_vs_shared": float(min(x["relative_gain_local_vs_shared"] for x in loo)),
            "loo_max_relative_gain_local_vs_shared": float(max(x["relative_gain_local_vs_shared"] for x in loo)),
            "max_pair_contribution_share": float(max_share),
            "total_observation_gain_local_vs_shared": total_gain,
        })
    else:
        result = {"n_evaluable_pairs": 0, "n_obs": 0, "n_contexts": 0,
                  "points": {}, "bootstrap": {}, "leave_one_pair_out": []}
        rows = []
    # judgement
    support_ok = bool(
        evaluable >= int(cfg["support"]["strict_mask_min_pairs"])
        and all(x["n_evaluation_contexts"] >= 5 for x in pair_records if x.get("evaluable"))
    )
    if not support_ok:
        judgment = "STRICT_MASK_SUPPORT_INSUFFICIENT"
        reason = "fewer than 10 evaluable strictly masked pairs or a pair has <5 evaluation contexts"
    else:
        pts = result["points"]
        gs = 1.0 - pts["LOCAL_MASKED"]["observation"]["mae"] / max(pts["SHARED_MASKED"]["observation"]["mae"], 1e-12)
        gn = 1.0 - pts["LOCAL_MASKED"]["observation"]["mae"] / max(pts["LOCAL_NULL_MASKED"]["observation"]["mae"], 1e-12)
        gz = 1.0 - pts["LOCAL_MASKED"]["observation"]["mae"] / max(pts["ZERO"]["observation"]["mae"], 1e-12)
        gc = 1.0 - pts["LOCAL_MASKED"]["context"]["mae"] / max(pts["SHARED_MASKED"]["context"]["mae"], 1e-12)
        gpt = 1.0 - pts["LOCAL_MASKED"]["pair_type"]["mae"] / max(pts["SHARED_MASKED"]["pair_type"]["mae"], 1e-12)
        gnc = 1.0 - pts["LOCAL_MASKED"]["context"]["mae"] / max(pts["LOCAL_NULL_MASKED"]["context"]["mae"], 1e-12)
        gptn = 1.0 - pts["LOCAL_MASKED"]["pair_type"]["mae"] / max(pts["LOCAL_NULL_MASKED"]["pair_type"]["mae"], 1e-12)
        bs = result["bootstrap"]["observation"]["diffs"]["LOCAL_MASKED_minus_SHARED_MASKED"]["ci"]
        bn = result["bootstrap"]["observation"]["diffs"]["LOCAL_MASKED_minus_LOCAL_NULL_MASKED"]["ci"]
        bz = result["bootstrap"]["observation"]["diffs"]["LOCAL_MASKED_minus_ZERO"]["ci"]
        direction_ok = all(x > 0 for x in (gs, gn, gz, gc, gpt, gnc, gptn))
        intervals_ok = (bs[1] < 0 and bn[1] < 0 and bz[1] < 0)
        not_one_pair = result["loo_min_relative_gain_local_vs_shared"] > 0
        practical = gs >= float(cfg["judgement"]["strict_mask_practical_gain"])
        result.update({
            "relative_gain_local_vs_shared": gs,
            "relative_gain_local_vs_null": gn,
            "relative_gain_local_vs_zero": gz,
            "relative_gain_local_vs_shared_context": gc,
            "relative_gain_local_vs_shared_pair_type": gpt,
            "practical_screen_10pct": bool(practical),
            "intervals_support_positive_gain": bool(intervals_ok),
            "direction_consistent": bool(direction_ok),
            "gain_not_entirely_one_pair": bool(not_one_pair),
        })
        if direction_ok and intervals_ok and not_one_pair and practical:
            judgment = "CONDITIONAL_COMPOSITION_SUPPORTED"
            reason = "LOCAL_MASKED beats SHARED_MASKED, LOCAL_NULL_MASKED and ZERO on the same strict-mask set; direction is consistent; leave-one-pair-out gain remains positive; the pre-declared 10% practical screen is met"
        elif direction_ok and intervals_ok and not_one_pair:
            judgment = "CONDITIONAL_COMPOSITION_SCREEN_ONLY"
            reason = "all four strict statistical conditions hold, but the pre-declared 10% practical screen is not met; the gain is real but small"
        elif gs > 0 and gn > 0 and gz > 0:
            judgment = "CONDITIONAL_COMPOSITION_SCREEN_ONLY"
            reason = "positive point gains over all controls, but at least one strict screen condition failed"
        else:
            judgment = "NO_SUPPORTED_CONDITIONAL_RULE"
            reason = "LOCAL_MASKED does not beat the controls on the strict-mask set"
    result["judgement"] = judgment
    result["reason"] = reason
    result["selected_pairs"] = selected
    result["pair_records"] = pair_records
    # csv
    rows_out = []
    for rec in pair_records:
        rows_out.append({"row_kind": "pair", "weighting": "observation", **rec})
    for row in rows:
        rows_out.append({"row_kind": row.get("row_kind", "pooled"),
                         "weighting": row.get("weighting", ""), **row})
    cols: list[str] = []
    for row in rows_out:
        for key in row:
            if key not in cols:
                cols.append(key)
    C.write_csv(C.ANALYSIS_DIR / "strict_mask_results.csv", rows_out, columns=cols)
    C.write_json(C.ANALYSIS_DIR / "strict_mask_details.json", result)
    return result


def main() -> int:
    cfg = C.load_config()
    result = run(cfg)
    print(json.dumps({k: result[k] for k in (
        "n_selected_pairs", "n_evaluable_pairs", "n_obs", "n_contexts",
        "relative_gain_local_vs_shared", "relative_gain_local_vs_null",
        "relative_gain_local_vs_zero", "loo_min_relative_gain_local_vs_shared",
        "max_pair_contribution_share", "direction_consistent",
        "intervals_support_positive_gain", "gain_not_entirely_one_pair",
        "practical_screen_10pct", "judgement", "reason") if k in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
