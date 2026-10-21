"""CPU audit B: is error explained by rare local structure or by common ones?

The J pattern frequency table is built from *independent official-train
molecules* only.  The valid split is queried, never used to build or filter the
vocabulary/frequencies.  All keys are exact content tuples/bytes; no learned
embedding, no y, no raw atom-id value is part of a pattern.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import audit_common as C  # noqa: E402

ANALYSIS = C.ANALYSIS
LENGTHS = [2, 3, 4, 5]
BUCKETS = [("f0", 0, 0), ("f1", 1, 1), ("f24", 2, 4), ("f519", 5, 19), ("f20", 20, 10**9)]


def mol_of_entries(cache: dict) -> np.ndarray:
    ptr = np.asarray(cache["ent_ptr"])
    return np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))


def build_pattern_frequency(train: dict) -> dict:
    """{length: per-pattern number of independent train molecules containing it}."""
    n_train = int(train["meta"]["n_molecules"])
    n_pat = int(np.asarray(train["pat_len_j"]).size)
    ent_len = np.asarray(train["ent_len"], dtype=np.int64)
    ent_pat = np.asarray(train["ent_pat_j"], dtype=np.int64)
    ent_mol = mol_of_entries(train)
    out = {}
    for l in LENGTHS:
        m = ent_len == (LENGTHS.index(l))
        comb = ent_pat[m].astype(np.int64) * n_train + ent_mol[m]
        u = np.unique(comb)
        up = (u // n_train).astype(np.int64)
        pu, pc = np.unique(up, return_counts=True)
        f = np.zeros(n_pat, dtype=np.int64)
        f[pu] = pc
        out[l] = f
    return out


def build_valid_to_train(train: dict, valid: dict) -> np.ndarray:
    """Map every valid pattern id to the train id with the *same content key*.

    Pattern ids are split-local sorted addresses.  Comparing valid id ``k`` to
    train id ``k`` directly would be a content error; this map uses the exact
    atom/pair bytes (with length kept separate) and returns -1 for unseen.
    """
    tr_atom = np.asarray(train["pat_atom_j"])
    tr_pair = np.asarray(train["pat_pair_j"])
    tr_len = np.asarray(train["pat_len_j"], dtype=np.int64)
    va_atom = np.asarray(valid["pat_atom_j"])
    va_pair = np.asarray(valid["pat_pair_j"])
    va_len = np.asarray(valid["pat_len_j"], dtype=np.int64)
    out = np.full(len(va_len), -1, dtype=np.int64)
    for l in LENGTHS:
        t_ids = np.where(tr_len == l)[0]
        v_ids = np.where(va_len == l)[0]
        lookup = {}
        for tid in t_ids:
            key = tr_atom[tid].astype(np.uint8).tobytes() + tr_pair[tid].astype(np.uint8).tobytes()
            lookup[key] = int(tid)
        for vid in v_ids:
            key = va_atom[vid].astype(np.uint8).tobytes() + va_pair[vid].astype(np.uint8).tobytes()
            out[vid] = lookup.get(key, -1)
    return out


def coverage_for_split(cache: dict, freq: dict, *, loo: bool = False, pat_map: np.ndarray | None = None) -> dict:
    """Occurrence-weighted coverage per molecule, per length and overall.

    If loo=True, the molecule itself is removed from its pattern frequency
    (used only to obtain train-only thresholds for descriptive bins).
    """
    n_mol = int(cache["meta"]["n_molecules"])
    n_pat = int(np.asarray(cache["pat_len_j"]).size)
    ent_len = np.asarray(cache["ent_len"], dtype=np.int64)
    ent_pat = np.asarray(cache["ent_pat_j"], dtype=np.int64)
    ent_cnt = np.asarray(cache["ent_count"], dtype=np.int64)
    ent_mol = mol_of_entries(cache)
    out = {}
    for l in LENGTHS:
        li = LENGTHS.index(l)
        m = ent_len == li
        mol = ent_mol[m]
        pat = ent_pat[m]
        cnt = ent_cnt[m]
        mapped = pat if pat_map is None else pat_map[pat]
        f = np.where(mapped >= 0, freq[l][np.maximum(mapped, 0)], 0).astype(np.int64)
        if loo:
            f = np.maximum(f - 1, 0)
        tot = np.zeros(n_mol, dtype=np.float64)
        np.add.at(tot, mol, cnt.astype(np.float64))
        for b, lo, hi in BUCKETS:
            w = np.where((f >= lo) & (f <= hi), cnt, 0).astype(np.float64)
            arr = np.zeros(n_mol, dtype=np.float64)
            np.add.at(arr, mol, w)
            out[f"l{l}_{b}"] = arr
        out[f"l{l}_tot"] = tot
        # unique pattern ratios: number of distinct patterns vs distinct unseen patterns
        base = int(n_pat) + 1
        comb = mol.astype(np.int64) * base + pat.astype(np.int64)
        uu = np.unique(comb)
        umol = (uu // base).astype(np.int64)
        upat = (uu % base).astype(np.int64)
        upat_mapped = upat if pat_map is None else pat_map[upat]
        unseen = (np.where(upat_mapped >= 0, freq[l][np.maximum(upat_mapped, 0)], 0) == 0).astype(np.float64)
        us = np.zeros(n_mol, dtype=np.float64)
        ut = np.zeros(n_mol, dtype=np.float64)
        np.add.at(us, umol, unseen)
        np.add.at(ut, umol, 1.0)
        out[f"l{l}_unseen_unique_ratio"] = np.divide(us, ut, out=np.zeros_like(us), where=ut > 0)
    # overall occurrence masses
    tot_all = sum(out[f"l{l}_tot"] for l in LENGTHS)
    out["tot_occ"] = tot_all
    for b, _, _ in BUCKETS:
        out[f"all_{b}"] = sum(out[f"l{l}_{b}"] for l in LENGTHS)
        out[f"all_{b}_mass"] = np.divide(
            out[f"all_{b}"], np.maximum(tot_all, 1), out=np.zeros_like(tot_all), where=tot_all > 0
        )
    # low-frequency mass = singleton + 2-4 bucket (pre-fixed)
    out["all_lowfreq_mass"] = out["all_f1_mass"] + out["all_f24_mass"]
    # unique-pattern unseen ratio overall (computed from unique counts)
    out["all_unseen_unique_ratio"] = sum(out[f"l{l}_unseen_unique_ratio"] for l in LENGTHS) / len(LENGTHS)
    return out


def radius_signatures(cache: dict):
    """Typed radius-1 and radius-2 rooted signatures from the raw graph.

    radius-1: (central atom category, sorted (bond type, neighbour category)).
    radius-2: (central atom category, sorted (bond type, radius-1 signature of
    neighbour)).  These are exact nested tuples, deterministic and label-free.
    """
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    at = np.asarray(cache["atom_type"])
    bi = np.asarray(cache["bond_index"])
    bt = np.asarray(cache["bond_type"])
    n = int(cache["n_molecules"])
    out1, out2 = [], []
    for i in range(n):
        cats = at[ap[i] : ap[i + 1]].astype(np.int64)
        adj = [[] for _ in range(len(cats))]
        for (u, v), t in zip(bi[bp[i] : bp[i + 1]], bt[bp[i] : bp[i + 1]]):
            u, v, t = int(u), int(v), int(t)
            adj[u].append((v, t))
            adj[v].append((u, t))
        for a in adj:
            a.sort()
        s1 = [(int(cats[v]), tuple(sorted((t, int(cats[u])) for u, t in adj[v]))) for v in range(len(cats))]
        s2 = [(int(cats[v]), tuple(sorted((t, s1[u]) for u, t in adj[v]))) for v in range(len(cats))]
        out1.append(s1)
        out2.append(s2)
    return out1, out2


def signature_coverage(train_sigs, valid_sigs, cache: dict) -> dict:
    freq = defaultdict(int)
    for sigs in train_sigs:
        for s in set(sigs):
            freq[s] += 1
    n = len(valid_sigs)
    unseen = np.zeros(n, dtype=np.float64)
    low4 = np.zeros(n, dtype=np.float64)
    mid = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    uniq_unseen = np.zeros(n, dtype=np.float64)
    uniq_total = np.zeros(n, dtype=np.float64)
    for i, sigs in enumerate(valid_sigs):
        vals = np.asarray([freq.get(s, 0) for s in sigs], dtype=np.int64)
        unseen[i] = float(np.mean(vals == 0))
        low4[i] = float(np.mean((vals >= 1) & (vals <= 4)))
        mid[i] = float(np.mean((vals >= 5) & (vals <= 19)))
        high[i] = float(np.mean(vals >= 20))
        uniq = set(sigs)
        uniq_unseen[i] = float(np.mean([freq.get(s, 0) == 0 for s in uniq]))
        uniq_total[i] = float(len(uniq))
    return {
        "unseen_atom_fraction": unseen,
        "low4_atom_fraction": low4,
        "mid_atom_fraction": mid,
        "high_atom_fraction": high,
        "unseen_unique_signature_fraction": uniq_unseen,
        "n_unique_signatures": uniq_total,
        "n_train_signatures": len(freq),
    }


def partial_spearman(x, y, control) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rc = np.argsort(np.argsort(control)).astype(float)
    # residualise ranks linearly on control rank
    bx = np.polyfit(rc, rx, 1)
    by = np.polyfit(rc, ry, 1)
    ex = rx - np.polyval(bx, rc)
    ey = ry - np.polyval(by, rc)
    return float(np.corrcoef(ex, ey)[0, 1])


def group_summary(values, groups, ae_by_model) -> list:
    out = []
    for g in np.unique(groups):
        m = groups == g
        row = {"group": int(g), "n": int(m.sum()), "value_mean": float(np.mean(values[m]))}
        for k, ae in ae_by_model.items():
            row[f"mae_{k}"] = float(ae[m].mean())
        out.append(row)
    return out


def main() -> int:
    runs, y, valid = C.load_all_predictions()
    names = list(runs)
    pred = {k: runs[k]["pred"] for k in names}
    ae = {k: np.abs(pred[k] - y) for k in names}
    train = C.load_path_cache("train")
    valid_path = C.load_path_cache("valid")
    freq = build_pattern_frequency(train)
    valid_to_train = build_valid_to_train(train, valid_path)
    cov = coverage_for_split(valid_path, freq, loo=False, pat_map=valid_to_train)
    cov_train_loo = coverage_for_split(train, freq, loo=True)

    # Fixed descriptive bins for valid path-unseen mass.  Train LOO mass is
    # degenerate at zero for most rows (96% of path occurrences are covered by
    # >=20 train molecules), so quantiles of train LOO are not informative.
    cov_edges = np.asarray([1e-12, 0.01, 0.05])
    cov_bin = np.digitize(cov["all_f0_mass"], cov_edges, right=True)

    # Typed radius-1/2 signatures.
    tr1, tr2 = radius_signatures(C.load_cache("train"))
    va1, va2 = radius_signatures(valid)
    sig1 = signature_coverage(tr1, va1, valid)
    sig2 = signature_coverage(tr2, va2, valid)

    # Raw structural descriptors.
    stats = [C.graph_stats(valid, i) for i in range(len(y))]
    nat = np.asarray([s["n_atoms"] for s in stats], dtype=float)
    cr = np.asarray([s["cycle_rank"] for s in stats], dtype=float)
    maxring = np.asarray([s["max_ring_size"] for s in stats], dtype=float)
    nrings = np.asarray([s["n_rings"] for s in stats], dtype=float)

    # Size bins fixed from train distribution (quartiles of n_atoms).
    tr_stats = C.load_cache("train")
    tr_nat = np.asarray(tr_stats["n_atoms_graph"], dtype=float)
    size_q = np.quantile(tr_nat, [0.25, 0.5, 0.75])
    size_bin = np.searchsorted(size_q, nat, side="right")
    # Cycle-rank bins fixed by the train distribution: 0-1, 2, 3, 4, 5+.
    tr_cr = []
    import networkx as nx

    ap = np.asarray(tr_stats["atom_ptr"])
    bp = np.asarray(tr_stats["bond_ptr"])
    bi = np.asarray(tr_stats["bond_index"])
    for i in range(tr_stats["n_molecules"]):
        g = nx.Graph()
        g.add_nodes_from(range(int(tr_stats["n_atoms_graph"][i])))
        for u, v in bi[bp[i] : bp[i + 1]]:
            g.add_edge(int(u), int(v))
        tr_cr.append(int(tr_stats["n_bonds_graph"][i]) - int(tr_stats["n_atoms_graph"][i]) + nx.number_connected_components(g))
    tr_cr = np.asarray(tr_cr)
    cycle_bin = np.where(cr <= 1, 0, np.where(cr >= 5, 4, cr - 1))  # 0:0-1 1:2 2:3 3:4 4:5+

    # Coverage vs error relations.
    coverage_keys = ["all_f0_mass", "all_f1_mass", "all_f24_mass", "all_f519_mass", "all_f20_mass", "all_lowfreq_mass"]
    relations = {}
    for key in coverage_keys:
        row = {"mean": float(cov[key].mean()), "median": float(np.median(cov[key])), "p90": float(np.percentile(cov[key], 90)), "max": float(cov[key].max())}
        for k in names:
            row[f"spearman_ae_{k}"] = float(spearmanr(cov[key], ae[k]).statistic)
            row[f"partial_spearman_ae_{k}_n_atoms"] = partial_spearman(cov[key], ae[k], nat)
        row["spearman_delta_j_minus_p3b"] = float(spearmanr(cov[key], ae["J_s0"] - ae["P3B_s0"]).statistic)
        row["partial_spearman_delta_j_minus_p3b_n_atoms"] = partial_spearman(cov[key], ae["J_s0"] - ae["P3B_s0"], nat)
        row["spearman_delta_q_minus_p3b"] = float(spearmanr(cov[key], ae["Q_s0"] - ae["P3B_s0"]).statistic)
        relations[key] = row

    relations["radius1"] = {
        "mean_unseen_atom_fraction": float(sig1["unseen_atom_fraction"].mean()),
        "mean_low4_atom_fraction": float(sig1["low4_atom_fraction"].mean()),
        "spearman_ae_P3B": float(spearmanr(sig1["unseen_atom_fraction"], ae["P3B_s0"]).statistic),
        "partial_spearman_ae_P3B_n_atoms": partial_spearman(sig1["unseen_atom_fraction"], ae["P3B_s0"], nat),
        "n_train_signatures": sig1["n_train_signatures"],
    }
    relations["radius2"] = {
        "mean_unseen_atom_fraction": float(sig2["unseen_atom_fraction"].mean()),
        "mean_low4_atom_fraction": float(sig2["low4_atom_fraction"].mean()),
        "mean_unseen_unique_signature_fraction": float(sig2["unseen_unique_signature_fraction"].mean()),
        "spearman_ae_P3B": float(spearmanr(sig2["unseen_atom_fraction"], ae["P3B_s0"]).statistic),
        "partial_spearman_ae_P3B_n_atoms": partial_spearman(sig2["unseen_atom_fraction"], ae["P3B_s0"], nat),
        "spearman_delta_j_minus_p3b": float(spearmanr(sig2["unseen_atom_fraction"], ae["J_s0"] - ae["P3B_s0"]).statistic),
        "n_train_signatures": sig2["n_train_signatures"],
    }

    # Group means by fixed size/cycle/coverage bins.
    groups = {
        "size_bin_train_quartiles": group_summary(nat, size_bin, ae),
        "cycle_rank_bin_train": group_summary(cr, cycle_bin, ae),
        "unseen_mass_bin_fixed": group_summary(cov["all_f0_mass"], cov_bin, ae),
    }
    # Within-size coverage associations, descriptive.
    within_size = []
    for b in range(4):
        m = size_bin == b
        within_size.append(
            {
                "size_bin": b,
                "n": int(m.sum()),
                "spearman_unseen_vs_AE_P3B": float(spearmanr(cov["all_f0_mass"][m], ae["P3B_s0"][m]).statistic),
                "spearman_unseen_vs_deltaJ": float(spearmanr(cov["all_f0_mass"][m], (ae["J_s0"] - ae["P3B_s0"])[m]).statistic),
            }
        )

    # Per-length coverage distribution table.
    per_length = {}
    for l in LENGTHS:
        tot = cov[f"l{l}_tot"]
        per_length[str(l)] = {
            "mean_occurrences_per_molecule": float(tot.mean()),
            "mean_unseen_occurrence_mass": float(np.mean(cov[f"l{l}_f0"] / np.maximum(tot, 1))),
            "mean_lowfreq_1_4_occurrence_mass": float(np.mean((cov[f"l{l}_f1"] + cov[f"l{l}_f24"]) / np.maximum(tot, 1))),
            "mean_unseen_unique_pattern_ratio": float(cov[f"l{l}_unseen_unique_ratio"].mean()),
            "mean_seen_occurrence_fraction_ge20": float(np.mean(cov[f"l{l}_f20"] / np.maximum(tot, 1))),
        }

    # Counterexamples: high rarity but low error, low rarity but high error.
    hi_rar = cov["all_f0_mass"] >= np.quantile(cov["all_f0_mass"], 0.8)
    lo_err = ae["P3B_s0"] <= np.quantile(ae["P3B_s0"], 0.2)
    lo_rar = cov["all_f0_mass"] <= np.quantile(cov["all_f0_mass"], 0.2)
    hi_err = ae["P3B_s0"] >= np.quantile(ae["P3B_s0"], 0.8)
    counter = {
        "high_rarity_low_error": [int(i) for i in np.where(hi_rar & lo_err)[0]],
        "low_rarity_high_error": [int(i) for i in np.where(lo_rar & hi_err)[0]],
        "high_rarity_high_error": [int(i) for i in np.where(hi_rar & hi_err)[0]],
        "low_rarity_low_error": [int(i) for i in np.where(lo_rar & lo_err)[0]],
        "note": "Descriptive quintile cross-tab; no row is dropped from any model metric.",
    }

    # Machine-readable per-molecule table (all 1000 valid rows).
    table_path = ANALYSIS / "per_molecule.csv"
    fields = [
        "row_id", "graph_sha256", "y_raw",
        "n_atoms", "n_bonds", "n_components", "cycle_rank", "n_rings", "max_ring_size", "degree_max",
        "atom_counts_21", "bond_counts_1_3", "size_bin", "cycle_bin", "unseen_mass_bin",
    ]
    for k in names:
        fields += [f"pred_{k}", f"signed_error_{k}", f"abs_error_{k}"]
    fields += ["median_ae_all7", "mean_ae_all7", "P3B_worst10"]
    fields += ["j_tot_occ", "j_unseen_mass", "j_lowfreq_mass", "j_unseen_unique_ratio"]
    for l in LENGTHS:
        fields += [
            f"j_l{l}_occurrences", f"j_l{l}_unseen_occ_mass", f"j_l{l}_f1_occ_mass",
            f"j_l{l}_f24_occ_mass", f"j_l{l}_f519_occ_mass", f"j_l{l}_f20_occ_mass",
            f"j_l{l}_unseen_unique_ratio",
        ]
    fields += ["r1_unseen_atom_frac", "r1_low4_atom_frac", "r2_unseen_atom_frac", "r2_low4_atom_frac", "r2_unseen_unique_frac"]
    med_all = np.median(np.stack([ae[k] for k in names], axis=0), axis=0)
    mean_all = np.mean(np.stack([ae[k] for k in names], axis=0), axis=0)
    worst10 = set(np.argsort(ae["P3B_s0"])[-10:].tolist())
    with open(table_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(len(y)):
            row = {
                "row_id": i,
                "graph_sha256": C.graph_sha256(valid, i),
                "y_raw": f"{y[i]:.9g}",
                "n_atoms": int(nat[i]),
                "n_bonds": int(stats[i]["n_bonds"]),
                "n_components": int(stats[i]["n_components"]),
                "cycle_rank": int(cr[i]),
                "n_rings": int(nrings[i]),
                "max_ring_size": int(maxring[i]),
                "degree_max": int(stats[i]["degree_max"]),
                "atom_counts_21": "|".join(str(x) for x in stats[i]["atom_counts_21"]),
                "bond_counts_1_3": "|".join(str(x) for x in stats[i]["bond_counts_1_3"]),
                "size_bin": int(size_bin[i]),
                "cycle_bin": int(cycle_bin[i]),
                "unseen_mass_bin": int(cov_bin[i]),
                "median_ae_all7": f"{med_all[i]:.9g}",
                "mean_ae_all7": f"{mean_all[i]:.9g}",
                "P3B_worst10": int(i in worst10),
                "j_tot_occ": f"{cov['tot_occ'][i]:.9g}",
                "j_unseen_mass": f"{cov['all_f0_mass'][i]:.9g}",
                "j_lowfreq_mass": f"{cov['all_lowfreq_mass'][i]:.9g}",
                "j_unseen_unique_ratio": f"{cov['all_unseen_unique_ratio'][i]:.9g}",
                "r1_unseen_atom_frac": f"{sig1['unseen_atom_fraction'][i]:.9g}",
                "r1_low4_atom_frac": f"{sig1['low4_atom_fraction'][i]:.9g}",
                "r2_unseen_atom_frac": f"{sig2['unseen_atom_fraction'][i]:.9g}",
                "r2_low4_atom_frac": f"{sig2['low4_atom_fraction'][i]:.9g}",
                "r2_unseen_unique_frac": f"{sig2['unseen_unique_signature_fraction'][i]:.9g}",
            }
            for k in names:
                row[f"pred_{k}"] = f"{pred[k][i]:.9g}"
                row[f"signed_error_{k}"] = f"{(pred[k][i] - y[i]):.9g}"
                row[f"abs_error_{k}"] = f"{ae[k][i]:.9g}"
            for l in LENGTHS:
                tot = cov[f"l{l}_tot"][i]
                for tag in ["f1", "f24", "f519", "f20"]:
                    row[f"j_l{l}_{tag}_occ_mass"] = f"{(cov[f'l{l}_{tag}'][i] / max(tot, 1)):.9g}"
                row[f"j_l{l}_occurrences"] = f"{tot:.9g}"
                row[f"j_l{l}_unseen_occ_mass"] = f"{(cov[f'l{l}_f0'][i] / max(tot, 1)):.9g}"
                row[f"j_l{l}_unseen_unique_ratio"] = f"{cov[f'l{l}_unseen_unique_ratio'][i]:.9g}"
            w.writerow(row)

    # Pattern-frequency table summary (no valid rows involved in frequency).
    pat_summary = {}
    pat_len = np.asarray(train["pat_len_j"], dtype=np.int64)
    for l in LENGTHS:
        f = freq[l]
        nz = f > 0
        n_len = int((pat_len == l).sum())
        f_len = f[pat_len == l]
        vmask = np.asarray(valid_path["ent_len"], dtype=np.int64) == LENGTHS.index(l)
        vp = np.unique(np.asarray(valid_path["ent_pat_j"], dtype=np.int64)[vmask])
        vp_map = valid_to_train[vp]
        pat_summary[str(l)] = {
            "n_train_patterns_total": n_len,
            "n_train_patterns_seen_at_least_once": int((f_len > 0).sum()),
            "n_train_patterns_freq_ge20": int((f_len >= 20).sum()),
            "max_train_molecule_frequency": int(f.max()),
            "n_valid_unique_patterns_observed": int(len(vp)),
            "n_valid_unique_patterns_absent_from_train": int((vp_map < 0).sum()),
        }

    summary = {
        "protocol_id": "zinc-generalization-audit-v1",
        "split": "official valid 1000; frequencies built from official train 10000 only",
        "official_test_loaded": False,
        "pattern_frequency_summary": pat_summary,
        "per_length_coverage": per_length,
        "overall_coverage": {
            "mean_unseen_occurrence_mass": float(cov["all_f0_mass"].mean()),
            "median_unseen_occurrence_mass": float(np.median(cov["all_f0_mass"])),
            "mean_lowfreq_1_4_occurrence_mass": float(cov["all_lowfreq_mass"].mean()),
            "mean_seen_ge20_occurrence_mass": float(cov["all_f20_mass"].mean()),
            "mean_unseen_unique_pattern_ratio": float(cov["all_unseen_unique_ratio"].mean()),
            "mean_occurrences_per_molecule": float(cov["tot_occ"].mean()),
        },
        "coverage_relations": relations,
        "group_means": groups,
        "within_size_associations": within_size,
        "counterexamples": counter,
        "coverage_bin_edges": [0.0, 1e-12, 0.01, 0.05],
        "coverage_bin_labels": ["zero", "(0,1%]", "(1%,5%]", ">5%"],
        "size_bin_thresholds_from_train": [float(x) for x in size_q],
        "cycle_bin_definition": "0: rank 0-1; 1: 2; 2: 3; 3: 4; 4: rank>=5 (train distribution fixed)",
        "notes": [
            "Occurrence mass weights every actual path occurrence; unique-pattern ratio is reported separately.",
            "Unseen means zero independent train molecules contained the exact J pattern; it does not mean the content cannot be encoded by the shared phi.",
            "Radius-1/2 signatures are exact nested tuples from raw atom/bond categories; no learned embedding or y.",
            "Coverage bins use train leave-one-out masses only; no valid labels are used to define thresholds.",
        ],
    }
    C.write_json(ANALYSIS / "coverage_summary.json", summary)
    C.write_json(
        ANALYSIS / "signature_summary.json",
        {
            "radius1": {
                "n_train_signatures": int(sig1["n_train_signatures"]),
                "mean_unseen_atom_fraction": float(sig1["unseen_atom_fraction"].mean()),
                "mean_low4_atom_fraction": float(sig1["low4_atom_fraction"].mean()),
                "mean_mid_atom_fraction": float(sig1["mid_atom_fraction"].mean()),
                "mean_high_atom_fraction": float(sig1["high_atom_fraction"].mean()),
            },
            "radius2": {
                "n_train_signatures": int(sig2["n_train_signatures"]),
                "mean_unseen_atom_fraction": float(sig2["unseen_atom_fraction"].mean()),
                "mean_low4_atom_fraction": float(sig2["low4_atom_fraction"].mean()),
                "mean_mid_atom_fraction": float(sig2["mid_atom_fraction"].mean()),
                "mean_high_atom_fraction": float(sig2["high_atom_fraction"].mean()),
                "mean_unseen_unique_signature_fraction": float(sig2["unseen_unique_signature_fraction"].mean()),
            },
        },
    )
    print(json.dumps({"ok": True, "per_length": per_length, "overall": summary["overall_coverage"], "relations": {k: relations[k] for k in ["all_f0_mass", "all_lowfreq_mass"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
