"""CPU prep: bridge caches, label-free lambda choice, coverage diagnostics."""

from __future__ import annotations

import argparse

import numpy as np

import nl_bridge
import nl_common as N


def lambda_analysis(n_mol: int = 1000) -> dict:
    """Propagation range of K=(lambda I + L)^-1 on real train graphs.

    Uniform conduction, no labels, no model.  For each candidate lambda we
    report the row-normalised off-diagonal mass vs graph distance.
    """
    cache = N.train_cache()
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bonds = np.asarray(cache["bond_index"], dtype=np.int64)
    lams = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    rec = {lam: {"mean_dist": [], "p90_dist": [], "mass_ge3": [], "cond": []} for lam in lams}
    step = max(1, (len(atom_ptr) - 1) // n_mol)
    for g in range(0, len(atom_ptr) - 1, step):
        n = int(atom_ptr[g + 1] - atom_ptr[g])
        bi = bonds[bond_ptr[g]:bond_ptr[g + 1]]
        if n > 60:
            continue
        # graph distance
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from((int(u), int(v)) for u, v in bi)
        D = nx.floyd_warshall_numpy(G)
        W = np.zeros((n, n))
        for u, v in bi:
            W[int(u), int(v)] = W[int(v), int(u)] = 1.0
        L = np.diag(W.sum(1)) - W
        for lam in lams:
            K = np.linalg.inv(lam * np.eye(n) + L)
            M = K * (1.0 - np.eye(n))
            s = M.sum(1, keepdims=True)
            P = M / np.maximum(s, 1e-12)
            d = np.where(np.eye(n) > 0, 0.0, D)
            mean_d = float((P * d).sum(1).mean())
            # p90 over atoms of the distance containing 90% of mass
            p90 = []
            for i in range(n):
                order = np.argsort(d[i])
                c = np.cumsum(P[i, order])
                k = int(np.searchsorted(c, 0.9))
                p90.append(float(d[i, order[min(k, n - 1)]]))
            rec[lam]["mean_dist"].append(mean_d)
            rec[lam]["p90_dist"].append(float(np.mean(p90)))
            rec[lam]["mass_ge3"].append(float((P * (d >= 3)).sum(1).mean()))
            rec[lam]["cond"].append(float(np.linalg.cond(lam * np.eye(n) + L)))
    out = {"official_test_loaded": False, "labels_used": False,
           "n_molecules": len(next(iter(rec.values()))["mean_dist"]),
           "candidates": {}}
    for lam, d in rec.items():
        out["candidates"][str(lam)] = {
            "mean_graph_distance": float(np.mean(d["mean_dist"])),
            "mean_p90_distance": float(np.mean(d["p90_dist"])),
            "offdiag_mass_at_distance_ge3": float(np.mean(d["mass_ge3"])),
            "condition_number_mean": float(np.mean(d["cond"])),
            "condition_number_max": float(np.max(d["cond"])),
        }
    # Pick the smallest lambda whose mean reach is >= 3 graph steps and whose
    # p90 reach is >= 4, unless it is numerically extreme.  Fixed before training.
    chosen = None
    for lam in lams:
        c = out["candidates"][str(lam)]
        if c["mean_graph_distance"] >= 3.0 and c["mean_p90_distance"] >= 4.0 \
                and c["condition_number_max"] <= 1.0e4:
            chosen = lam  # keep the largest lambda that still reaches the range
    if chosen is None:
        chosen = 0.5
    out["chosen_lambda"] = float(chosen)
    out["rule"] = ("largest lambda whose label-free mean reach is >=3 graph steps, "
                   "p90 reach >=4, cond<=1e4; no valid MAE used")
    return out


def bridge_coverage() -> dict:
    """Bridge coverage over real valid molecules and r2-unseen subsets (label-free)."""
    valid = N.D.load_cache("valid")
    bridge = nl_bridge.load("valid")
    n_bridges = np.asarray(bridge["n_bridges"])
    # radius-2 support tier, label-free: recompute the historical definition
    # from the frozen cache is expensive; reuse the frozen per-molecule table
    # in LCRC analysis (label-free columns only) if present.
    out = {
        "valid_molecules": int(len(n_bridges)),
        "with_bridge": int((n_bridges > 0).sum()),
        "without_bridge": int((n_bridges == 0).sum()),
        "fraction_with_bridge": float((n_bridges > 0).mean()),
        "median_bridges": float(np.median(n_bridges)),
        "mean_bridges": float(n_bridges.mean()),
        "max_bridges": int(n_bridges.max()),
    }
    table = N.LCRC_DIR / "analysis" / "per_molecule_valid.csv"
    if table.exists():
        import csv
        rows = list(csv.DictReader(open(table)))
        if rows and "r2_unseen_atom_frac" in rows[0]:
            unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0 for r in rows])
            out["r2_unseen_molecules"] = int(unseen.sum())
            out["r2_unseen_with_bridge"] = int((n_bridges[unseen] > 0).sum())
            out["r2_unseen_fraction_with_bridge"] = float((n_bridges[unseen] > 0).mean())
            out["seen_fraction_with_bridge"] = float((n_bridges[~unseen] > 0).mean())
            out["r2_unseen_mean_bridges"] = float(n_bridges[unseen].mean())
            out["seen_mean_bridges"] = float(n_bridges[~unseen].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lambda", action="store_true")
    args = ap.parse_args()
    report = {"official_test_loaded": False}
    report["bridge_prep"] = nl_bridge.prep()
    report["coverage"] = bridge_coverage()
    if not args.skip_lambda:
        report["lambda"] = lambda_analysis()
    N.ensure_dir(N.ANALYSIS)
    N.L.write_json(N.ANALYSIS / "nl_prep.json", report)
    print(json_dumps(report))


def json_dumps(obj):
    import json
    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    main()
