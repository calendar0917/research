"""Phase A analysis, part 2: dose-response, bias share, counterfactual headroom,
and the pre-fixed target-free similarity/support readout.

Amendment A1 (documented in REPORT.md, made after inspecting only the *similarity
distribution* and the kNN reference level, never a group error table): the raw
count cosine of PLAN.md is replaced by IDF-weighted WL cosine as the primary
whole-molecule similarity, because the raw version is near-degenerate (all top-k
similarities 0.89-0.95) and its kNN reference is worse than the global median
predictor.  Both variants are reported.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

import dca_common as C


def support_tiers(graphs, fit_graphs):
    """Per-molecule unseen / low4 / supported atom fractions w.r.t. a fit set."""
    fit_sets, _ = C.radius2_signature_sets(fit_graphs)
    freq = {}
    for s in fit_sets:
        for k in s:
            freq[k] = freq.get(k, 0) + 1
    mol_sets, mol_atoms = C.radius2_signature_sets(graphs)
    unseen, low4, supported, all_support = [], [], [], []
    for atoms in mol_atoms:
        n = len(atoms)
        if n == 0:
            unseen.append(0.0); low4.append(0.0); supported.append(0.0)
            all_support.append([])
            continue
        c = [freq.get(a, 0) for a in atoms]
        unseen.append(float(np.mean([x == 0 for x in c])))
        low4.append(float(np.mean([1 <= x <= 4 for x in c])))
        supported.append(float(np.mean([x >= 5 for x in c])))
        all_support.append(c)
    return {
        "unseen_frac": np.asarray(unseen),
        "low4_frac": np.asarray(low4),
        "supported_frac": np.asarray(supported),
        "min_support": np.asarray([min(c) if c else 0 for c in all_support]),
        "median_support": np.asarray([float(np.median(c)) if c else 0 for c in all_support]),
    }


def idf_wl_similarity(train_graphs, valid_graphs, h=2):
    ft = C.wl_features(train_graphs, h)
    fv = C.wl_features(valid_graphs, h)
    idx = C.build_vocab(ft)
    X = C.vectorize(ft, idx, normalize=False)
    Q = C.vectorize(fv, idx, normalize=False)
    df = np.asarray(X.sum(0)).reshape(-1)
    idf = np.log((X.shape[0] + 1) / (1 + df)) + 1.0
    W = sp.diags(idf)
    Xw, Qw = X @ W, Q @ W

    def unit(M):
        n = np.sqrt(np.asarray(M.multiply(M).sum(1)).reshape(-1))
        n[n == 0] = 1.0
        return sp.diags(1.0 / n) @ M

    return np.asarray((unit(Qw) @ unit(Xw).T).todense(), dtype=np.float64), len(idx)


def cells_of(values, cuts):
    return np.searchsorted(cuts, values, side="right")


def main() -> int:
    train = C.load_cache("train")
    valid = C.load_cache("valid")
    y = np.asarray(valid["labels"], dtype=np.float64)
    y_train = np.asarray(train["labels"], dtype=np.float64)
    pm = pd.read_csv(C.ANALYSIS / "per_molecule.csv").sort_values("row_id").reset_index(drop=True)
    e4 = np.asarray([pm[f"se_s{k}"].to_numpy() for k in range(4)])
    ae = np.abs(e4).mean(0)
    tg = C.molecule_graphs(train)
    vg = C.molecule_graphs(valid)
    sup = support_tiers(vg, tg)

    out: dict = {}

    # ---------------------------------------------------------------- dose response
    u = sup["unseen_frac"]
    ubin = np.where(u == 0, "0", np.where(u <= 0.05, "(0,5%]",
                   np.where(u <= 0.10, "(5,10%]", ">10%")))
    rows = []
    for b in ("0", "(0,5%]", "(5,10%]", ">10%"):
        m = ubin == b
        rows.append({
            "bin": b, "n": int(m.sum()),
            "mae": float(ae[m].mean()),
            "ae_share": float(ae[m].sum() / ae.sum()),
            "signed": float(e4[:, m].mean()),
            "bias_share_of_mse": float(e4[:, m].mean() ** 2 / (e4[:, m] ** 2).mean()),
        })
    out["dose_response_unseen_frac"] = rows

    # support-count dose response (env support >= 5 atoms)
    sfr = sup["supported_frac"]
    sb = np.where(sfr >= 0.999, "all_supported",
                  np.where(sfr >= 0.95, ">=95%", np.where(sfr >= 0.80, "80-95%", "<80%")))
    rows = []
    for b in ("all_supported", ">=95%", "80-95%", "<80%"):
        m = sb == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mae": float(ae[m].mean()),
                     "ae_share": float(ae[m].sum() / ae.sum())})
    out["dose_response_supported_frac"] = rows

    # --------------------------------------------------- bias vs dispersion per group
    groups = {
        "r2_unseen": u > 0,
        "r2_seen": u == 0,
        "seen_full": (u == 0) & (sup["low4_frac"] == 0),
        "seen_low4": (u == 0) & (sup["low4_frac"] > 0),
        "unseen_ring_lt4": (u > 0) & (pm.n_rings.to_numpy() < 4),
        "unseen_ring_ge4": (u > 0) & (pm.n_rings.to_numpy() >= 4),
        "seen_ring_ge4": (u == 0) & (pm.n_rings.to_numpy() >= 4),
        "tail_y_le_-3": y <= -3,
        "ordinary": y > -3,
    }
    rows = []
    for name, m in groups.items():
        if not m.any():
            continue
        e = e4[:, m]
        rows.append({
            "group": name, "n": int(m.sum()),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "signed": float(e.mean()),
            "bias_share_of_mse": float(e.mean() ** 2 / (e ** 2).mean()),
            "seed_spread_of_signed": float(e.mean(1).std(ddof=1)),
            "y_mean": float(y[m].mean()), "y_sd": float(y[m].std()),
        })
    out["bias_vs_dispersion"] = rows

    # ------------------------------------------------------- counterfactual headroom
    m_un, m_se = (u > 0), (u == 0)
    seen_mae = float(ae[m_se].mean())
    un_mae = float(ae[m_un].mean())
    out["headroom"] = {
        "mae_total": float(ae.mean()),
        "mae_seen": seen_mae,
        "mae_unseen": un_mae,
        "unseen_ae_share": float(ae[m_un].sum() / ae.sum()),
        "n_unseen": int(m_un.sum()),
        "mae_if_unseen_equal_seen": float((ae[m_se].sum() + seen_mae * m_un.sum()) / len(ae)),
        "unseen_mae_needed_for_total_0.060_holding_seen": float((0.060 * len(ae) - ae[m_se].sum()) / m_un.sum()),
        "delta_needed_total": float(ae.mean() - 0.060),
        "target": 0.060,
    }

    # ------------------------------------------------ target-free similarity support
    sim_idf, vocab_idf = idf_wl_similarity(tg, vg)
    tr_sets, _ = C.radius2_signature_sets(tg)
    va_sets, _ = C.radius2_signature_sets(vg)
    sim_tan = C.binary_tanimoto(va_sets, tr_sets)

    sim_blocks = {}
    st_store = {}
    for name, sim in (("wl2_idf_cosine", sim_idf), ("r2_tanimoto", sim_tan)):
        st = C.neighbour_stats(sim, y_train, k=C.K_NN)
        st_store[name] = st
        knn_err = np.abs(st["nn_wpred"] - y)
        sim_blocks[name] = {
            "definition": ("IDF-weighted WL subtree kernel h=2, cosine, k=10"
                           if name == "wl2_idf_cosine" else
                           "Tanimoto on exact typed radius-2 signature sets, k=10"),
            "sim_top1_quantiles": [float(x) for x in np.quantile(st["sim_top1"], [0, .1, .5, .9, 1])],
            "sim_topk_quantiles": [float(x) for x in np.quantile(st["sim_topk"], [0, .1, .5, .9, 1])],
            "nn_ystd_quantiles": [float(x) for x in np.quantile(st["nn_ystd"], [0, .1, .5, .9, 1])],
            "knn_reference_mae": float(knn_err.mean()),
            "median_abs_global": float(np.abs(y - np.median(y_train)).mean()),
            "spearman_top1_vs_nn_label_gap": C.spearman(st["sim_top1"], np.abs(st["nn_wpred"] - y)),
            "spearman_simtopk_ae": C.spearman(st["sim_topk"], ae),
            "spearman_simtopk_ae_partial": C.partial_spearman(
                st["sim_topk"], ae, [pm.n_atoms.to_numpy(), pm.cycle_rank.to_numpy(), np.abs(y)]),
            "spearman_nnystd_ae": C.spearman(st["nn_ystd"], ae),
            "spearman_nnystd_ae_partial": C.partial_spearman(
                st["nn_ystd"], ae, [pm.n_atoms.to_numpy(), pm.cycle_rank.to_numpy(), np.abs(y)]),
        }
    out["similarity"] = sim_blocks

    # tercile cross (primary = idf cosine): support x label dispersion
    st = st_store["wl2_idf_cosine"]
    sq = np.quantile(st["sim_topk"], [1 / 3, 2 / 3])
    dq = np.quantile(st["nn_ystd"], [1 / 3, 2 / 3])
    sbin = cells_of(st["sim_topk"], sq)
    dbin = cells_of(st["nn_ystd"], dq)
    rows = []
    for s in range(3):
        for d in range(3):
            m = (sbin == s) & (dbin == d)
            rows.append({
                "support_bin": int(s + 1), "dispersion_bin": int(d + 1), "n": int(m.sum()),
                "mae": float(ae[m].mean()), "ae_share": float(ae[m].sum() / ae.sum()),
                "knn_mae": float(np.abs(st["nn_wpred"][m] - y[m]).mean()),
                "sim_topk": float(st["sim_topk"][m].mean()),
                "nn_ystd": float(st["nn_ystd"][m].mean()),
                "unseen_frac_within": float((u[m] > 0).mean()),
            })
    out["support_x_dispersion"] = {
        "cuts_sim_topk": [float(x) for x in sq],
        "cuts_nn_ystd": [float(x) for x in dq],
        "cells": rows,
    }
    m_good = (sbin == 2) & (dbin == 0)
    m_low = sbin == 0
    order = np.argsort(-ae)
    hard = np.zeros(len(ae), dtype=bool)
    hard[order[:100]] = True
    out["hard_core"] = {
        "definition": "top 100 valid molecules by seed-mean |error|",
        "n": int(hard.sum()),
        "ae_share": float(ae[hard].sum() / ae.sum()),
        "mae": float(ae[hard].mean()),
        "sim_topk_hard": float(st["sim_topk"][hard].mean()),
        "sim_topk_rest": float(st["sim_topk"][~hard].mean()),
        "nn_ystd_hard": float(st["nn_ystd"][hard].mean()),
        "nn_ystd_rest": float(st["nn_ystd"][~hard].mean()),
        "knn_mae_hard": float(np.abs(st["nn_wpred"][hard] - y[hard]).mean()),
        "frac_unseen_r2": float((u[hard] > 0).mean()),
        "frac_tail": float((y[hard] <= -3).mean()),
        "frac_high_support_low_disp": float(m_good[hard].mean()),
        "frac_low_support": float(m_low[hard].mean()),
        "mass_high_support_low_disp": {
            "n": int(m_good.sum()), "ae_share": float(ae[m_good].sum() / ae.sum()),
            "mae": float(ae[m_good].mean())},
        "mass_low_support": {
            "n": int(m_low.sum()), "ae_share": float(ae[m_low].sum() / ae.sum()),
            "mae": float(ae[m_low].mean())},
    }

    # per-group kNN vs model (primary similarity)
    rows = []
    for nm, m in (("seen+ordinary", (u == 0) & (y > -3)), ("unseen+ordinary", (u > 0) & (y > -3)),
                  ("seen+tail", (u == 0) & (y <= -3)), ("unseen+tail", (u > 0) & (y <= -3))):
        rows.append({"group": nm, "n": int(m.sum()), "dcc_mae": float(ae[m].mean()),
                     "knn_mae": float(np.abs(st["nn_wpred"][m] - y[m]).mean()),
                     "knn_mae_tan": float(np.abs(st_store["r2_tanimoto"]["nn_wpred"][m] - y[m]).mean())})
    out["knn_vs_dcc"] = rows

    # --------------------------------------------------- store merged per-molecule
    pm2 = pm.copy()
    for k in ("unseen_frac", "low4_frac", "supported_frac", "min_support", "median_support"):
        pm2[f"fitset_{k}"] = sup[k]
    for name, stx in st_store.items():
        for key in ("sim_top1", "sim_topk", "nn_ystd", "nn_wpred"):
            pm2[f"{name}_{key}"] = stx[key]
    pm2.to_csv(C.ANALYSIS / "per_molecule_features.csv", index=False)

    C.write_json(C.ANALYSIS / "analysis_part2.json", out)
    print(json.dumps({k: out[k] for k in ("headroom", "dose_response_unseen_frac")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
