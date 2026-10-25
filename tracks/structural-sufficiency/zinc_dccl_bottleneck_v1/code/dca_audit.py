"""Phase A: zero-GPU bottleneck audit of the frozen DCC-L champion.

Reads only official train + official valid.  Never constructs or reads test.
Outputs analysis/per_molecule.csv, analysis/groups.json, analysis/report_tables.md.
"""

from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pandas as pd

import dca_common as C


def mae(e):
    return float(np.mean(np.abs(e)))


def group_row(mask, y, e4):
    """e4: [4, n] signed errors."""
    n = int(mask.sum())
    if n == 0:
        return None
    a = np.abs(e4[:, mask])
    se = e4[:, mask].mean(1)
    tot = float(np.abs(e4).mean(0).sum())  # total AE = sum of per-row seed-mean AE
    same = len(set(np.sign(np.round(se, 12)))) == 1
    row = {
        "n": n,
        "n_frac": n / len(y),
        "mae": float(a.mean()),
        "mae_per_seed": [float(x) for x in a.mean(1)],
        "mae_seed_sd": float(a.mean(1).std(ddof=1)),
        "signed": float(e4[:, mask].mean()),
        "signed_per_seed": [float(x) for x in se],
        "abs_signed_share_of_ae": float(abs(e4[:, mask].mean()) / max(a.mean(), 1e-12)),
        "ae_share": float(a.mean(0).sum() / tot),
        "seed_sign_consistent": bool(same),
        "systematic_bias": bool(same and np.abs(se).min() > 1.5 * se.std(ddof=1)),
        "sign_agree_frac": float(
            np.mean(np.all(np.sign(e4[:, mask]) == np.sign(e4[:, mask])[:1], axis=0))
        ),
    }
    return row


def mece_tables(df, y, e4):
    tables = OrderedDict()

    # ---- table 1: U x T x R
    u = df.r2_unseen_atom_frac.to_numpy() > 0
    t = y <= -3.0
    r = np.where(df.n_rings.to_numpy() <= 1, "r0-1",
                 np.where(df.n_rings.to_numpy() <= 3, "r2-3", "r>=4"))
    cells = []
    for uu in (False, True):
        for tt in (False, True):
            for rr in ("r0-1", "r2-3", "r>=4"):
                m = (u == uu) & (t == tt) & (r == rr)
                row = group_row(m, y, e4)
                if row is None:
                    continue
                row.update({"r2_unseen": bool(uu), "tail": bool(tt), "rings": rr})
                cells.append(row)
    tables["U_x_T_x_R"] = cells
    assert sum(c["n"] for c in cells) == len(y)

    # ---- table 2: size tertile (train cuts) x support tier
    s_cuts = df.attrs["size_cuts"]
    sb = np.searchsorted(s_cuts, df.n_atoms.to_numpy(), side="right")
    sb_label = np.array([f"size_q{k + 1}" for k in sb])
    low4 = df.r2_low4_atom_frac.to_numpy() > 0
    tier = np.where(u, "unseen", np.where(low4, "seen_low4", "seen_full"))
    cells = []
    for s in ("size_q1", "size_q2", "size_q3"):
        for tt in ("seen_full", "seen_low4", "unseen"):
            m = (sb_label == s) & (tier == tt)
            row = group_row(m, y, e4)
            if row is None:
                continue
            row.update({"size": s, "support": tt})
            cells.append(row)
    tables["S_x_support"] = cells
    assert sum(c["n"] for c in cells) == len(y)

    # ---- table 3: U x T x size (median split)
    s_med = float(np.median(df.n_atoms.to_numpy()))
    big = df.n_atoms.to_numpy() > s_med
    cells = []
    for uu in (False, True):
        for tt in (False, True):
            for bb in (False, True):
                m = (u == uu) & (t == tt) & (big == bb)
                row = group_row(m, y, e4)
                if row is None:
                    continue
                row.update({"r2_unseen": bool(uu), "tail": bool(tt), "large": bool(bb)})
                cells.append(row)
    tables["U_x_T_x_size"] = cells
    assert sum(c["n"] for c in cells) == len(y)
    return tables


def confounder_block(df, y, e4):
    ae = np.abs(e4).mean(0)
    nat = df.n_atoms.to_numpy().astype(float)
    cr = df.cycle_rank.to_numpy().astype(float)
    ay = np.abs(y)
    x = df.r2_unseen_atom_frac.to_numpy().astype(float)
    out = {
        "spearman_raw": C.spearman(x, ae),
        "spearman_partial_n_atoms": C.partial_spearman(x, ae, [nat]),
        "spearman_partial_natoms_cyclerank": C.partial_spearman(x, ae, [nat, cr]),
        "spearman_partial_natoms_cyclerank_absy": C.partial_spearman(x, ae, [nat, cr, ay]),
        "spearman_ae_absy": C.spearman(ay, ae),
        "spearman_ae_natoms": C.spearman(nat, ae),
        "spearman_ae_cyclerank": C.spearman(cr, ae),
        "spearman_ae_nrings": C.spearman(df.n_rings.to_numpy().astype(float), ae),
    }
    # size x ring standardised unseen - seen delta
    s_cuts = df.attrs["size_cuts"]
    sb = np.searchsorted(s_cuts, df.n_atoms.to_numpy(), side="right")
    r = np.where(df.n_rings.to_numpy() <= 1, 0, np.where(df.n_rings.to_numpy() <= 3, 1, 2))
    u = df.r2_unseen_atom_frac.to_numpy() > 0
    strata = []
    num = den = 0.0
    for s in range(3):
        for rr in range(3):
            m_u = (sb == s) & (r == rr) & u
            m_s = (sb == s) & (r == rr) & ~u
            cell = {
                "size_bin": s + 1,
                "ring_bin": rr,
                "n_unseen": int(m_u.sum()),
                "n_seen": int(m_s.sum()),
                "mae_unseen": mae(e4[:, m_u]) if m_u.sum() else None,
                "mae_seen": mae(e4[:, m_s]) if m_s.sum() else None,
            }
            if m_u.sum() >= 10 and m_s.sum() >= 10:
                d = cell["mae_unseen"] - cell["mae_seen"]
                cell["delta"] = d
                w = int(m_u.sum())
                num += w * d
                den += w
            strata.append(cell)
    out["strata_delta_unseen_minus_seen"] = strata
    out["delta_standardised_by_size_ring"] = float(num / den) if den else None
    out["weight_molecules"] = int(den)

    # single-molecule dominance check (drop rows with AE > 1)
    keep = ae <= 1.0
    out["drop_ae_gt1"] = {
        "n": int(keep.sum()),
        "mae_all": mae(e4[:, keep]),
        "delta_unseen_minus_seen": float(
            mae(e4[:, keep & u]) - mae(e4[:, keep & ~u])
        ),
        "mae_unseen": mae(e4[:, keep & u]),
        "mae_seen": mae(e4[:, keep & ~u]),
    }
    # per-seed deltas (is the contrast seed-stable?)
    out["per_seed_delta_unseen_minus_seen"] = [
        float(np.abs(e4[k, u]).mean() - np.abs(e4[k, ~u]).mean()) for k in range(e4.shape[0])
    ]
    return out


def seed_block(df, y, e4, tables):
    corr = np.corrcoef(e4)
    out = {
        "signed_error_pearson_between_seeds": corr.tolist(),
        "min_pairwise_corr": float(corr.min()),
        "mean_pairwise_corr": float((corr.sum() - np.trace(corr)) / (corr.size - corr.shape[0])),
        "global_sign_agree_frac": float(
            np.mean(np.all(np.sign(e4) == np.sign(e4)[:1], axis=0))
        ),
    }
    per_table = {}
    for name, cells in tables.items():
        per_table[name] = [
            {
                "label": {k: v for k, v in c.items()
                          if k in ("r2_unseen", "tail", "rings", "size", "support", "large")},
                "n": c["n"],
                "signed_per_seed": c["signed_per_seed"],
                "seed_sign_consistent": c["seed_sign_consistent"],
                "systematic_bias": c["systematic_bias"],
                "sign_agree_frac": c["sign_agree_frac"],
            }
            for c in cells
        ]
    out["per_table"] = per_table
    return out


def similarity_block(df, y, e4, sim_wl, sim_tan, y_train):
    ae = np.abs(e4).mean(0)
    out = {}
    stats = {}
    for name, sim in (("wl2_cosine", sim_wl), ("r2_tanimoto", sim_tan)):
        st = C.neighbour_stats(sim, y_train, k=C.K_NN)
        err = np.abs(st["nn_wpred"] - y)
        out[name] = {
            "def": ("WL subtree kernel h=2 (bond-typed), cosine, k=10"
                    if name == "wl2_cosine" else
                    "Tanimoto on exact typed radius-2 signature sets, k=10"),
            "knn_reference_mae": float(err.mean()),
            "knn_reference_mae_unseen": float(err[df.r2_unseen_atom_frac.to_numpy() > 0].mean()),
            "knn_reference_mae_seen": float(err[df.r2_unseen_atom_frac.to_numpy() <= 0].mean()),
            "spearman_simtopk_ae": C.spearman(st["sim_topk"], ae),
            "spearman_simtopk_ae_partial": C.partial_spearman(
                st["sim_topk"], ae,
                [df.n_atoms.to_numpy(), df.cycle_rank.to_numpy(), np.abs(y)]),
            "spearman_nnystd_ae": C.spearman(st["nn_ystd"], ae),
            "spearman_nnystd_ae_partial": C.partial_spearman(
                st["nn_ystd"], ae,
                [df.n_atoms.to_numpy(), df.cycle_rank.to_numpy(), np.abs(y)]),
            "spearman_simtopk_nnystd": C.spearman(st["sim_topk"], st["nn_ystd"]),
        }
        stats[name] = st
    primary = stats["wl2_cosine"]
    # tercile cross of support x label dispersion
    sq = np.quantile(primary["sim_topk"], [1 / 3, 2 / 3])
    dq = np.quantile(primary["nn_ystd"], [1 / 3, 2 / 3])
    sb = np.searchsorted(sq, primary["sim_topk"], side="right")
    db = np.searchsorted(dq, primary["nn_ystd"], side="right")
    cells = []
    tot = float(ae.sum())
    for s in range(3):
        for d in range(3):
            m = (sb == s) & (db == d)
            if not m.any():
                continue
            cells.append({
                "support_bin": s + 1, "dispersion_bin": d + 1, "n": int(m.sum()),
                "mae": float(ae[m].mean()),
                "mae_p10": float(ae[m].mean()),
                "ae_share": float(ae[m].sum() / tot),
                "knn_err": float(np.abs(primary["nn_wpred"][m] - y[m]).mean()),
                "sim_topk": float(primary["sim_topk"][m].mean()),
                "nn_ystd": float(primary["nn_ystd"][m].mean()),
            })
    out["support_x_dispersion"] = {
        "cuts_sim_topk": [float(x) for x in sq],
        "cuts_nn_ystd": [float(x) for x in dq],
        "cells": cells,
    }
    # high support / low dispersion mass = evidence against H1 for the hard core
    m_good = (sb == 2) & (db == 0)
    m_low = (sb == 0)
    out["mass_high_support_low_dispersion"] = {
        "n": int(m_good.sum()),
        "ae_share": float(ae[m_good].sum() / tot),
        "mae": float(ae[m_good].mean()),
    }
    out["mass_low_support"] = {
        "n": int(m_low.sum()),
        "ae_share": float(ae[m_low].sum() / tot),
        "mae": float(ae[m_low].mean()),
    }
    # hard core profile (top 100 by AE)
    order = np.argsort(-ae)
    hard = np.zeros(len(ae), dtype=bool)
    hard[order[:100]] = True
    rest = ~hard
    out["hard_core_top100"] = {
        "n": int(hard.sum()),
        "ae_share": float(ae[hard].sum() / tot),
        "mae": float(ae[hard].mean()),
        "sim_topk_hard": float(primary["sim_topk"][hard].mean()),
        "sim_topk_rest": float(primary["sim_topk"][rest].mean()),
        "nn_ystd_hard": float(primary["nn_ystd"][hard].mean()),
        "nn_ystd_rest": float(primary["nn_ystd"][rest].mean()),
        "knn_err_hard": float(np.abs(primary["nn_wpred"][hard] - y[hard]).mean()),
        "knn_err_rest": float(np.abs(primary["nn_wpred"][rest] - y[rest]).mean()),
        "frac_in_high_support_low_disp": float((hard & m_good).sum() / hard.sum()),
        "frac_in_low_support": float((hard & m_low).sum() / hard.sum()),
        "frac_unseen_r2": float((hard & (df.r2_unseen_atom_frac.to_numpy() > 0)).sum() / hard.sum()),
        "frac_tail": float((hard & (y <= -3)).sum() / hard.sum()),
    }
    # per-group kNN vs model, table-1 groups
    u = df.r2_unseen_atom_frac.to_numpy() > 0
    out["knn_vs_dcc_by_unseen_tail"] = [
        {
            "r2_unseen": bool(uu), "tail": bool(tt),
            "n": int(((u == uu) & ((y <= -3) == tt)).sum()),
            "dcc_mae": float(ae[(u == uu) & ((y <= -3) == tt)].mean()),
            "knn_mae": float(np.abs(primary["nn_wpred"] - y)[(u == uu) & ((y <= -3) == tt)].mean()),
        }
        for uu in (False, True) for tt in (False, True)
    ]
    return out, primary


def top_rows(df, y, e4, primary, k=20):
    ae = np.abs(e4).mean(0)
    order = np.argsort(-ae)[:k]
    rows = []
    for i in order:
        rows.append({
            "row": int(df.row_id.iloc[i]),
            "y": float(y[i]),
            "ae": float(ae[i]),
            "signed_per_seed": [float(x) for x in e4[:, i]],
            "n_atoms": int(df.n_atoms.iloc[i]),
            "n_rings": int(df.n_rings.iloc[i]),
            "max_ring": int(df.max_ring_size.iloc[i]),
            "r2_unseen_frac": float(df.r2_unseen_atom_frac.iloc[i]),
            "sim_topk": float(primary["sim_topk"][i]),
            "nn_ystd": float(primary["nn_ystd"][i]),
            "nn_pred": float(primary["nn_wpred"][i]),
            "nn_pred_err": float(abs(primary["nn_wpred"][i] - y[i])),
        })
    return rows


def main() -> int:
    train = C.load_cache("train")
    valid = C.load_cache("valid")
    y, pred = C.load_dcc(valid)
    e4 = pred - y[None, :]
    y_train = np.asarray(train["labels"], dtype=np.float64)

    hist = pd.read_csv(C.AUD / "analysis" / "per_molecule.csv").sort_values("row_id")
    df = hist[["row_id", "y_raw", "n_atoms", "n_rings", "cycle_rank", "max_ring_size",
               "r2_unseen_atom_frac", "r2_low4_atom_frac"]].reset_index(drop=True)
    if not np.allclose(df.y_raw.to_numpy(), y, atol=1e-6):
        raise RuntimeError("historical per-molecule table misaligned")

    tg = C.molecule_graphs(train)
    vg = C.molecule_graphs(valid)
    # validate the r2 unseen statistic against the historical audit column
    tr_sets, _ = C.radius2_signature_sets(tg)
    va_sets, va_atoms = C.radius2_signature_sets(vg)
    freq = {}
    for s in tr_sets:
        for k in s:
            freq[k] = freq.get(k, 0) + 1
    unseen_frac = np.asarray(
        [np.mean([freq.get(a, 0) == 0 for a in atoms]) if atoms else 0.0
         for atoms in va_atoms], dtype=np.float64)
    low4_frac = np.asarray(
        [np.mean([1 <= freq.get(a, 0) <= 4 for a in atoms]) if atoms else 0.0
         for atoms in va_atoms], dtype=np.float64)
    check = {
        "max_abs_diff_r2_unseen_vs_history": float(np.abs(unseen_frac - df.r2_unseen_atom_frac.to_numpy()).max()),
        "max_abs_diff_r2_low4_vs_history": float(np.abs(low4_frac - df.r2_low4_atom_frac.to_numpy()).max()),
    }

    ft = C.wl_features(tg, h=2)
    fv = C.wl_features(vg, h=2)
    wl_idx = C.build_vocab(ft)
    wl_vocab = len(wl_idx)
    xw = C.vectorize(ft, wl_idx)
    xv = C.vectorize(fv, wl_idx)
    sim_wl = np.asarray((xv @ xw.T).todense(), dtype=np.float64)
    sim_tan = C.binary_tanimoto(va_sets, tr_sets)

    df.attrs["size_cuts"] = np.asarray(
        np.quantile(np.asarray([len(a) for a, _ in tg], dtype=float), [1 / 3, 2 / 3]),
        dtype=float)

    tables = mece_tables(df, y, e4)
    conf = confounder_block(df, y, e4)
    seedb = seed_block(df, y, e4, tables)
    simb, primary = similarity_block(df, y, e4, sim_wl, sim_tan, y_train)

    overall = {
        "n_valid": int(len(y)),
        "mae_per_seed": [mae(e4[k]) for k in range(4)],
        "mae_seed_mean": float(np.mean([mae(e4[k]) for k in range(4)])),
        "mae_seed_sd": float(np.std([mae(e4[k]) for k in range(4)], ddof=1)),
        "median_ae": float(np.median(np.abs(e4).mean(0))),
        "p90_ae": float(np.quantile(np.abs(e4).mean(0), 0.9)),
        "top10_ae_share": float(np.sort(np.abs(e4).mean(0))[::-1][:10].sum() / np.abs(e4).mean(0).sum()),
        "top100_ae_share": float(np.sort(np.abs(e4).mean(0))[::-1][:100].sum() / np.abs(e4).mean(0).sum()),
    }

    payload = {
        "round": "zinc-dccl-bottleneck-v1",
        "phase": "A",
        "official_test_read": False,
        "model": {"name": "DCC-L (B-R0)", "params": 494193, "seeds": list(C.DCC_SEEDS),
                  "pred_file": "zinc_performance_search_v1/analysis/per_molecule_valid.csv"},
        "input_sha256": {
            "ad_train": C.sha256_file(C.V1 / "cache" / "ad_train.pt"),
            "ad_valid": C.sha256_file(C.V1 / "cache" / "ad_valid.pt"),
            "historical_per_molecule": C.sha256_file(C.AUD / "analysis" / "per_molecule.csv"),
        },
        "checks": {**check,
                   "wl_vocab": int(wl_vocab),
                   "similarity_shape": list(sim_wl.shape),
                   "size_cuts_train_terciles": [float(x) for x in df.attrs["size_cuts"]]},
        "overall": overall,
        "tables_mece": tables,
        "confounders": conf,
        "seed_structure": seedb,
        "similarity_support": simb,
        "top20_rows": top_rows(df, y, e4, primary),
    }
    C.write_json(C.ANALYSIS / "groups.json", payload)

    # per-molecule table
    out = df.copy()
    out["ae_mean"] = np.abs(e4).mean(0)
    for k in range(4):
        out[f"pred_s{k}"] = pred[k]
        out[f"se_s{k}"] = e4[k]
    for name, st in (("wl2", primary), ("tan", C.neighbour_stats(sim_tan, y_train, k=C.K_NN))):
        for key, v in st.items():
            out[f"{name}_{key}"] = v
    out["knn_pred"] = primary["nn_wpred"]
    out["knn_err"] = np.abs(primary["nn_wpred"] - y)
    out.to_csv(C.ANALYSIS / "per_molecule.csv", index=False)
    print(json.dumps({"overall": overall, "checks": check}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
