"""Zero-GPU closure analysis for GOS-v1: paired A vs DCC-L, and gate mass.

Reads only frozen per-molecule valid predictions and frozen checkpoints.
Official test is never constructed.  Nothing here selects a new prediction
route; every stratum is descriptive.

    python code/gos_mechanism_closure.py paired
    python code/gos_mechanism_closure.py mass
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402
import gos_runner as GR  # noqa: E402

ANALYSIS = _EXP / "analysis"
ZPS = _SSUF / "zinc_performance_search_v1" / "analysis"
A_SEEDS = (0, 1)
DCC_SEEDS = (0, 1, 2, 3)
TAIL_Y = -3.0
BOOT = 4000
BOOT_SEED = 20261026


def _load_predictions():
    d = np.load(ANALYSIS / "per_molecule_valid.npz")
    y = np.asarray(d["y_raw"], dtype=np.float64)
    rows = [int(x) for x in d["row_id"]]
    meta = {}
    with open(ZPS / "per_molecule_valid.csv") as fh:
        for r in csv.DictReader(fh):
            meta[int(r["row_id"])] = r
    pred_a = {s: np.asarray(d[f"pred_A_s{s}"], dtype=np.float64) for s in A_SEEDS}
    pred_d = {}
    for s in DCC_SEEDS:
        k = f"pred_B-R0-online-s{s}"
        pred_d[s] = np.asarray([float(meta[i][k]) for i in rows], dtype=np.float64)
    n_rings = np.asarray([int(meta[i]["n_rings"]) for i in rows], dtype=np.int64)
    unseen_frac = np.asarray([float(meta[i]["r2_unseen_atom_frac"]) for i in rows], dtype=np.float64)
    return d, y, rows, pred_a, pred_d, n_rings, unseen_frac


def _ring_descriptors(rows, n_atoms, n_bonds):
    """Label-free graph descriptors: 2-core atom fraction and longest basis cycle."""
    cache = D.load_cache("valid")
    bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])
    ring_atom_frac = np.zeros(len(rows), dtype=np.float64)
    max_cycle = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        n = int(n_atoms[i])
        adj = [[] for _ in range(n)]
        for u, v in bi[bp[row]:bp[row + 1]]:
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        # exact 2-core by leaf peeling (independent of RDKit / SMILES)
        deg = np.asarray([len(x) for x in adj], dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        queue = [j for j in range(n) if deg[j] <= 1]
        while queue:
            u = queue.pop()
            if not alive[u]:
                continue
            alive[u] = False
            for v in adj[u]:
                if alive[v]:
                    deg[v] -= 1
                    if deg[v] <= 1:
                        queue.append(v)
        ring_atom_frac[i] = float(alive.mean())
        # cycle basis on the 2-core (networkx cycle_basis is fast here)
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(np.where(alive)[0].tolist())
        g.add_edges_from((int(u), int(v)) for u, v in bi[bp[row]:bp[row + 1]]
                         if alive[int(u)] and alive[int(v)])
        cyc = nx.cycle_basis(g) if g.number_of_edges() else []
        max_cycle[i] = max((len(c) for c in cyc), default=0)
    return ring_atom_frac, max_cycle


def _boot_ci(diff, rng, n_boot=BOOT):
    k = len(diff)
    if k == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, k, size=(n_boot, k))
    m = diff[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _corr(x, y):
    from scipy.stats import spearmanr
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(spearmanr(x, y).statistic),
    }


def paired() -> int:
    d, y, rows, pred_a, pred_d, n_rings, unseen_frac = _load_predictions()
    cache = D.load_cache("valid")
    n_atoms = np.asarray(cache["n_atoms_graph"], dtype=np.int64)
    n_bonds = np.asarray(cache["n_bonds_graph"], dtype=np.int64)
    ring_atom_frac, max_cycle = _ring_descriptors(rows, n_atoms, n_bonds)
    unseen = unseen_frac > 0

    ae_a = {s: np.abs(pred_a[s] - y) for s in A_SEEDS}
    ae_d = {s: np.abs(pred_d[s] - y) for s in DCC_SEEDS}
    err_a = {s: pred_a[s] - y for s in A_SEEDS}
    err_d = {s: pred_d[s] - y for s in DCC_SEEDS}

    # Seed-average absolute error (not an ensemble prediction): this equals the
    # average of the 8 A-vs-DCCL per-molecule AE differences and therefore does
    # not give the 4-seed family an artificial smoothing advantage.
    ae_af = np.mean([ae_a[s] for s in A_SEEDS], axis=0)
    ae_df = np.mean([ae_d[s] for s in DCC_SEEDS], axis=0)

    pairwise = {}
    for sa in A_SEEDS:
        for sd in DCC_SEEDS:
            diff = ae_a[sa] - ae_d[sd]
            pairwise[f"A_s{sa}_vs_DCCL_s{sd}"] = {
                "signed_error_corr": _corr(err_a[sa], err_d[sd]),
                "absolute_error_corr": _corr(ae_a[sa], ae_d[sd]),
                "mae_A": float(ae_a[sa].mean()),
                "mae_D": float(ae_d[sd].mean()),
                "diff_A_minus_D": float(diff.mean()),
                "win_rate_A": float((diff < 0).mean()),
            }

    # ---- fixed worst sets -------------------------------------------------
    ae_sets = {"A_s0": ae_a[0], "A_s1": ae_a[1],
               "DCCL_s0": ae_d[0], "A_seed_avg_AE": ae_af,
               "DCCL_seed_avg_AE": ae_df}
    worst = {name: {k: np.argsort(-ae, kind="stable")[:k] for k in (50, 100, 200)}
             for name, ae in ae_sets.items()}

    def set_report(ref_name, k):
        ref = worst[ref_name][k]
        ref_set = set(ref.tolist())
        out = {"reference": ref_name, "k": k, "members": int(len(ref_set))}
        for other, ae in ae_sets.items():
            oth = set(worst[other][k].tolist())
            inter = len(ref_set & oth)
            out[other] = {
                "overlap": inter,
                "jaccard": float(inter / len(ref_set | oth)),
                "ae_share_of_other_total": float(ae[list(ref_set)].sum() / ae.sum()),
            }
        for name, ae in ae_sets.items():
            out.setdefault("own_top_share", {})[name] = float(
                ae[worst[name][k]].sum() / ae.sum())
        return out

    worst_report = {
        f"ref_{ref}_k{k}": set_report(ref, k)
        for ref in ("DCCL_s0", "DCCL_seed_avg_AE")
        for k in (50, 100, 200)
    }

    # ---- descriptive strata ----------------------------------------------
    q = np.quantile(n_atoms, [1 / 3, 2 / 3])
    size_band = np.digitize(n_atoms, q)
    strata = {
        "size_low_natoms": size_band == 0,
        "size_mid_natoms": size_band == 1,
        "size_high_natoms": size_band == 2,
        "rings_0_1": n_rings <= 1,
        "rings_2": n_rings == 2,
        "rings_3": n_rings == 3,
        "rings_4plus": n_rings >= 4,
        "r2_seen": ~unseen,
        "r2_unseen": unseen,
        "ordinary_y": y > TAIL_Y,
        "tail_y": y <= TAIL_Y,
    }
    rng = np.random.default_rng(BOOT_SEED)
    strata_out = {}
    for name, mask in strata.items():
        diff = (ae_af - ae_df)[mask]
        lo, hi = _boot_ci(diff, rng)
        d0 = (ae_a[0] - ae_d[0])[mask]
        d1 = (ae_a[1] - ae_d[1])[mask]
        strata_out[name] = {
            "n": int(mask.sum()),
            "mae_A_seed_avg": float(ae_af[mask].mean()),
            "mae_DCCL_seed_avg": float(ae_df[mask].mean()),
            "diff_A_minus_D": float(diff.mean()),
            "ci95": [lo, hi],
            "win_rate_A": float((diff < 0).mean()),
            "same_seed_diff_A0_minus_D0": float(d0.mean()),
            "same_seed_diff_A1_minus_D1": float(d1.mean()),
            "sign_consistent": bool(np.sign(d0.mean()) == np.sign(d1.mean()) == np.sign(diff.mean())),
        }

    out = {
        "protocol": "zinc-global-operator-search-v1",
        "kind": "zero-GPU paired valid analysis (descriptive; no test, no routing)",
        "n_molecules": int(len(y)),
        "A_models": [f"A_s{s}" for s in A_SEEDS],
        "DCCL_models": [f"B-R0 s{s}" for s in DCC_SEEDS],
        "note_seed_avg": "A/DCCL seed-average AE is descriptive only. It is the average "
                         "of per-model absolute errors (equivalently, the average of the "
                         "2x4 paired per-molecule differences), not an ensemble prediction.",
        "size_cut_natoms": [float(q[0]), float(q[1])],
        "n_rings_distribution": {str(k): int((n_rings == k).sum()) for k in np.unique(n_rings)},
        "pairwise": pairwise,
        "seed_average_AE": {
            "mae_A": float(ae_af.mean()),
            "mae_D": float(ae_df.mean()),
            "diff_A_minus_D": float((ae_af - ae_df).mean()),
            "win_rate_A": float(((ae_af - ae_df) < 0).mean()),
            "pairwise_signed_corr_range": [
                min(v["signed_error_corr"]["pearson"] for v in pairwise.values()),
                max(v["signed_error_corr"]["pearson"] for v in pairwise.values())],
            "pairwise_abs_corr_range": [
                min(v["absolute_error_corr"]["pearson"] for v in pairwise.values()),
                max(v["absolute_error_corr"]["pearson"] for v in pairwise.values())],
        },
        "worst_sets": worst_report,
        "strata": strata_out,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "paired_A_vs_dccl.json", out)
    np.savez(
        ANALYSIS / "paired_A_vs_dccl_per_molecule.npz",
        row_id=np.asarray(rows), y_raw=y, n_atoms=n_atoms, n_bonds=n_bonds,
        n_rings=n_rings, ring_atom_frac=ring_atom_frac, max_cycle=max_cycle,
        r2_unseen_atom_frac=unseen_frac,
        pred_A_s0=pred_a[0], pred_A_s1=pred_a[1],
        **{f"pred_DCCL_s{s}": pred_d[s] for s in DCC_SEEDS},
    )

    lines = ["# Paired valid analysis: A vs DCC-L (descriptive, no test)", ""]
    lines.append(f"- n = {len(y)} official valid molecules; A = {out['A_models']}; "
                 f"DCC-L = {out['DCCL_models']}.")
    lines.append(f"- seed-average AE MAE: A {ae_af.mean():.6f}, DCC-L {ae_df.mean():.6f}, "
                 f"diff {(ae_af-ae_df).mean():+.6f}; this equals the average of the 8 "
                 f"per-seed paired differences.")
    lines.append(f"- pairwise signed-error corr range {out['seed_average_AE']['pairwise_signed_corr_range'][0]:.3f}-"
                 f"{out['seed_average_AE']['pairwise_signed_corr_range'][1]:.3f}; "
                 f"absolute-error corr range {out['seed_average_AE']['pairwise_abs_corr_range'][0]:.3f}-"
                 f"{out['seed_average_AE']['pairwise_abs_corr_range'][1]:.3f}.")
    lines += ["", "## Same-seed pairs", "",
              "| pair | MAE A | MAE D | diff | win A | signed corr | abs corr |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in pairwise.items():
        lines.append(f"| {k} | {v['mae_A']:.6f} | {v['mae_D']:.6f} | "
                     f"{v['diff_A_minus_D']:+.6f} | {v['win_rate_A']:.3f} | "
                     f"{v['signed_error_corr']['pearson']:.3f} | "
                     f"{v['absolute_error_corr']['pearson']:.3f} |")
    lines += ["", "## Descriptions by fixed structural stratum", "",
              "| stratum | n | MAE A | MAE D | diff A-D | 95% CI | win A | same-seed A0-D0 | A1-D1 |",
              "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for k, v in strata_out.items():
        lines.append(f"| {k} | {v['n']} | {v['mae_A_seed_avg']:.6f} | "
                     f"{v['mae_DCCL_seed_avg']:.6f} | {v['diff_A_minus_D']:+.6f} | "
                     f"[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | {v['win_rate_A']:.3f} | "
                     f"{v['same_seed_diff_A0_minus_D0']:+.6f} | "
                     f"{v['same_seed_diff_A1_minus_D1']:+.6f} |")
    lines += ["", "## Fixed worst-set overlap", ""]
    for k, v in worst_report.items():
        lines.append(f"- {k}: overlap with A_s0 {v['A_s0']['overlap']}/"
                     f"{v['A_s0']['jaccard']:.3f} Jaccard; A_s1 {v['A_s1']['overlap']}/"
                     f"{v['A_s1']['jaccard']:.3f}; DCCL_s0 {v['DCCL_s0']['overlap']}/"
                     f"{v['DCCL_s0']['jaccard']:.3f}; share of A seed-average AE in this set "
                     f"{v['A_seed_avg_AE']['ae_share_of_other_total']:.3f}, of DCC-L seed-average AE "
                     f"{v['DCCL_seed_avg_AE']['ae_share_of_other_total']:.3f}.")
    (ANALYSIS / "paired_A_vs_dccl.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# gate mass
# ---------------------------------------------------------------------------


def _collect_mass(checkpoint: Path, rows: np.ndarray, cache, collate, mean_nb: float):
    norm = D.load_norm()
    model, info = GM.build_model("A", norm, 0, "cpu", mean_n_bonds=mean_nb,
                                 mix_mode="legacy")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    layers = list(model.layers)
    recs = [[] for _ in layers]
    for li, layer in enumerate(layers):
        orig = layer._attend

        def wrap(*args, _orig=orig, _li=li):
            out = _orig(*args)
            recs[_li].append(out["mass"].detach().float().cpu().numpy())
            return out

        layer._attend = wrap

    cell, mol, bond, mval = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            chunk = rows[start:start + 128]
            batch = collate(cache, chunk, "P3N", valid=True)
            _ = model(batch)
            nb = batch["rel_mask"].sum(1).numpy().astype(int)
            for li in range(len(layers)):
                m = recs[li][-1]                       # [B,H,N,1]
                for h in range(m.shape[1]):
                    for i, row in enumerate(chunk):
                        n = int(nb[i])
                        cell.extend([li * m.shape[1] + h] * n)
                        mol.extend([int(row)] * n)
                        bond.extend(range(n))
                        mval.extend(m[i, h, :n, 0].tolist())
    return (np.asarray(cell, np.int64), np.asarray(mol, np.int64),
            np.asarray(bond, np.int64), np.asarray(mval, np.float64),
            info, int(ck.get("epoch", -1)))


def _group_stats(cell, mol, values, n_cells, n_mol):
    """Mean/std/CV per (cell, molecule), vectorised over the 12 x 1000 groups."""
    order = np.lexsort((mol, cell))
    c = cell[order]
    m = mol[order]
    v = values[order]
    key = c.astype(np.int64) * n_mol + m
    _, start, counts = np.unique(key, return_index=True, return_counts=True)
    sums = np.add.reduceat(v, start)
    sumsq = np.add.reduceat(v * v, start)
    means = sums / counts
    var = np.maximum(sumsq / counts - means * means, 0.0)
    stds = np.sqrt(var)
    cv = stds / np.maximum(means, 1e-12)
    cell_of = key[start] // n_mol
    mol_of = key[start] % n_mol
    mean = np.full((n_cells, n_mol), np.nan)
    std = np.full_like(mean, np.nan)
    cvv = np.full_like(mean, np.nan)
    mean[cell_of, mol_of] = means
    std[cell_of, mol_of] = stds
    cvv[cell_of, mol_of] = cv
    return mean, std, cvv, (c, m, v, start, counts, means, sumsq)


def _ss_decomposition(values, cell, mol, n_cells, groups=None):
    if groups is None:
        _, _, _, groups = _group_stats(cell, mol, values, n_cells, int(mol.max()) + 1)
    c, m, v, start, counts, means, sumsq = groups
    n = len(values)
    cell_sum = np.zeros(n_cells)
    cell_cnt = np.zeros(n_cells)
    for cc in range(n_cells):
        sel = c == cc
        cell_sum[cc] = v[sel].sum()
        cell_cnt[cc] = sel.sum()
    cell_mean = cell_sum / cell_cnt
    grand = v.mean()
    ss_cell = n / n_cells * ((cell_mean - grand) ** 2).sum()
    group_cell = c[start]
    ss_mol = float((counts * (means - cell_mean[group_cell]) ** 2).sum())
    ss_within = float((sumsq - counts * means * means).sum())
    total = ss_cell + ss_mol + ss_within
    return {
        "between_layer_head": float(ss_cell / total),
        "between_molecules": float(ss_mol / total),
        "within_molecule_queries": float(ss_within / total),
        "grand_mean": float(grand),
        "cell_means": cell_mean.tolist(),
    }


def _within_molecule_r2(values, cell, mol, features, n_cells):
    """Pooled within-cell, within-molecule R^2 of query mass on local bond features."""
    n_mol = int(mol.max()) + 1
    key = cell.astype(np.int64) * n_mol + mol
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    y = np.asarray(values, dtype=np.float64)
    ysum = np.bincount(inv, weights=y, minlength=len(counts))
    ym = (ysum / counts)[inv]
    X = np.column_stack([features[k] for k in features])
    Xsum = np.column_stack([
        np.bincount(inv, weights=X[:, j], minlength=len(counts))
        for j in range(X.shape[1])])
    Xm = (Xsum / counts[:, None])[inv]
    yc = y - ym
    Xc = X - Xm
    out = {}
    for c in range(n_cells):
        sel = cell == c
        yy = yc[sel]
        XX = Xc[sel]
        if np.sum(yy ** 2) <= 0:
            continue
        beta = np.linalg.lstsq(XX, yy, rcond=None)[0]
        fit = XX @ beta
        out[str(c)] = float(1.0 - ((yy - fit) ** 2).sum() / (yy ** 2).sum())
    return out


def mass() -> int:
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    cache = D.load_cache("valid")
    rows = np.arange(int(cache["labels"].shape[0]), dtype=np.int64)
    n_atoms = np.asarray(cache["n_atoms_graph"], dtype=np.int64)
    n_bonds = np.asarray(cache["n_bonds_graph"], dtype=np.int64)
    ring_atom_frac, max_cycle = _ring_descriptors(rows, n_atoms, n_bonds)

    # structural labels come from the frozen previous-round per-molecule table
    _, y, _, _, _, n_rings, unseen_frac = _load_predictions()
    d = np.load(ANALYSIS / "per_molecule_valid.npz")
    ae_a0 = np.abs(np.asarray(d["pred_A_s0"], dtype=np.float64) - y)

    checkpoints = {
        "ho_A_s0": _EXP / "runs" / "ho_A_s0" / "best.pt",
        "A_s0": _EXP / "runs" / "A_s0" / "best.pt",
    }
    out = {
        "protocol": "zinc-global-operator-search-v1",
        "kind": "frozen-checkpoint gate-mass description on official valid (no test)",
        "definition": "s_(layer,head,query) = sum_key attention(query,key) * gate(query,key)",
        "n_molecules": int(len(rows)),
        "n_queries_by_cell": 0,
        "runs": {},
    }
    per_mol_out = {"row_id": rows, "y_raw": y, "n_atoms": n_atoms, "n_bonds": n_bonds,
                   "n_rings": n_rings, "ring_atom_frac": ring_atom_frac,
                   "max_cycle": max_cycle, "r2_unseen_atom_frac": unseen_frac}

    for tag, ckpt in checkpoints.items():
        cell, mol, bond, mval, info, epoch = _collect_mass(ckpt, rows, cache, collate, mean_nb)
        cell_names = [f"L{li}H{h}" for li in range(3) for h in range(4)]
        n_cells = len(cell_names)
        if "n_queries_by_cell" in out and isinstance(out["n_queries_by_cell"], int):
            out["n_queries_by_cell"] = int(len(mval))
        mean, std, cv, group_data = _group_stats(cell, mol, mval, n_cells, len(rows))
        mass_stats = _ss_decomposition(mval, cell, mol, n_cells, group_data)
        log_vals = np.log10(np.clip(mval, 1e-12, None))
        log_groups = _group_stats(cell, mol, log_vals, n_cells, len(rows))
        logmass_stats = _ss_decomposition(log_vals, cell, mol, n_cells, log_groups[3])

        # global distribution
        dist = {
            "mean": float(mval.mean()),
            "median": float(np.median(mval)),
            "q05": float(np.quantile(mval, 0.05)),
            "q95": float(np.quantile(mval, 0.95)),
            "min": float(mval.min()),
            "max": float(mval.max()),
            "frac_below_0.01": float((mval < 0.01).mean()),
            "frac_below_0.10": float((mval < 0.10).mean()),
            "frac_above_1.50": float((mval > 1.50).mean()),
        }
        per_cell = {}
        for c, name in enumerate(cell_names):
            x = mval[cell == c]
            per_cell[name] = {
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "std": float(x.std()),
                "frac_below_0.10": float((x < 0.10).mean()),
                "median_within_molecule_cv": float(np.nanmedian(cv[c])),
                "molecule_mean_sd": float(np.nanstd(mean[c])),
            }

        # molecule-level structure regressions (per cell, then a range)
        Xmol = np.column_stack([
            np.log1p(n_bonds), n_rings, ring_atom_frac,
            np.log1p(max_cycle), (unseen_frac > 0).astype(np.float64),
        ])
        Xmol = np.column_stack([np.ones(len(rows)), Xmol])
        struct_r2 = {}
        corr_log_nb = {}
        for c, name in enumerate(cell_names):
            yy = mean[c]
            beta = np.linalg.lstsq(Xmol, yy, rcond=None)[0]
            fit = Xmol @ beta
            struct_r2[name] = float(1 - ((yy - fit) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
            corr_log_nb[name] = float(np.corrcoef(yy, np.log1p(n_bonds))[0, 1])

        # within-molecule local features: ring bond, endpoint degree sum, bond type
        bp = np.asarray(cache["bond_ptr"])
        bi = np.asarray(cache["bond_index"])
        bf = np.asarray(cache["bond_feat"])
        gidx = bp[mol] + bond
        deg = np.bincount(bi.reshape(-1), minlength=int(n_atoms.sum()))
        local = {
            "ring_bond": (bf[gidx, 4] > 0.5).astype(np.float64),
            "endpoint_degree_sum": (deg[bi[gidx, 0]] + deg[bi[gidx, 1]]).astype(np.float64),
            "double_bond": (np.asarray(cache["bond_type"])[gidx] == 2).astype(np.float64),
        }
        local_r2 = _within_molecule_r2(np.log10(np.clip(mval, 1e-12, None)),
                                       cell, mol, local, n_cells)

        # descriptive association with the frozen A_s0 error (not a supervision claim)
        err_corr = {}
        for c, name in enumerate(cell_names):
            err_corr[name] = {
                "mean_mass_vs_ae": float(np.corrcoef(mean[c], ae_a0)[0, 1]),
                "within_molecule_cv_vs_ae": float(np.corrcoef(cv[c], ae_a0)[0, 1]),
            }

        per_mol_out[f"mean_mass_{tag}"] = mean
        per_mol_out[f"within_cv_{tag}"] = cv
        out["runs"][tag] = {
            "checkpoint": str(ckpt.relative_to(C.REPO)),
            "epoch": epoch,
            "checkpoint_sha256": C.sha256_file(ckpt),
            "parameter_count": info["parameters"],
            "global": dist,
            "variance_share_mass": mass_stats,
            "variance_share_log10_mass": logmass_stats,
            "per_cell": per_cell,
            "molecule_mean_mass_corr_with_log_n_bonds": corr_log_nb,
            "molecule_mean_mass_struct_r2": struct_r2,
            "within_molecule_local_r2_log10": local_r2,
            "descriptive_error_corr": err_corr,
        }

    # cross-checkpoint agreement of per-molecule mass statistics
    common = np.isfinite(per_mol_out["mean_mass_ho_A_s0"]).all(1)
    out["cross_checkpoint"] = {
        "note": "Pearson/Spearman over the 12 layer-head cells of the per-molecule mean mass.",
        "mean_mass_pearson_by_cell": [
            float(np.corrcoef(per_mol_out["mean_mass_ho_A_s0"][c],
                              per_mol_out["mean_mass_A_s0"][c])[0, 1])
            for c in range(12)],
        "mean_mass_spearman_by_cell": [
            float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
                per_mol_out["mean_mass_ho_A_s0"][c],
                per_mol_out["mean_mass_A_s0"][c]).statistic)
            for c in range(12)],
    }
    C.write_json(ANALYSIS / "gate_mass.json", out)
    np.savez(ANALYSIS / "gate_mass_per_molecule.npz", **per_mol_out)

    # compact markdown
    lines = ["# Gate mass from frozen A (official valid, descriptive)", ""]
    lines.append("s(layer,head,query) = sum_key attention * gate, from the frozen legacy "
                 "A checkpoint; no test, no new supervision.")
    for tag, r in out["runs"].items():
        g = r["global"]
        vm = r["variance_share_mass"]
        vl = r["variance_share_log10_mass"]
        lines += ["", f"## {tag} (epoch {r['epoch']})", "",
                  f"- mass mean {g['mean']:.4f}, median {g['median']:.4f}, "
                  f"5-95% [{g['q05']:.4g}, {g['q95']:.4f}], "
                  f"frac<0.01 {g['frac_below_0.01']:.3f}, <0.10 {g['frac_below_0.10']:.3f}.",
                  f"- variance share (mass): layer/head {vm['between_layer_head']:.3f}, "
                  f"between molecules {vm['between_molecules']:.3f}, "
                  f"within-molecule queries {vm['within_molecule_queries']:.3f}; "
                  f"log10: {vl['between_layer_head']:.3f} / {vl['between_molecules']:.3f} / "
                  f"{vl['within_molecule_queries']:.3f}.",
                  f"- molecule-mean mass R2 on [log bonds, rings, ring-atom frac, max cycle, "
                  f"unseen flag]: median {np.median(list(r['molecule_mean_mass_struct_r2'].values())):.3f} "
                  f"(range {min(r['molecule_mean_mass_struct_r2'].values()):.3f}-"
                  f"{max(r['molecule_mean_mass_struct_r2'].values()):.3f}).",
                  f"- within-molecule log-mass R2 on local bond features: median "
                  f"{np.median(list(r['within_molecule_local_r2_log10'].values())):.3f} "
                  f"(range {min(r['within_molecule_local_r2_log10'].values()):.3f}-"
                  f"{max(r['within_molecule_local_r2_log10'].values()):.3f}).",
                  "- layer/head cell means: " + ", ".join(
                      f"{k}={v['mean']:.3f}" for k, v in r["per_cell"].items()) + "."]
    (ANALYSIS / "gate_mass.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["paired", "mass"])
    args = p.parse_args(argv)
    return paired() if args.command == "paired" else mass()


if __name__ == "__main__":
    raise SystemExit(main())
