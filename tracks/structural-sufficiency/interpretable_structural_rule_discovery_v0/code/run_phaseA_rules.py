"""Phase A runner: single rules, real co-path joint rules, matched random joint rules.

Writes ``analysis/rule_registry.csv`` (all candidates + filter status) and
``analysis/phaseA_rules.json`` (machine-readable pools used by Phases B/C).

Train only.  No Valid/Test label is read here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import isrd_common as K
import isrd_rules as R


def _cell_supports(act_a: np.ndarray, act_b: np.ndarray, scaf: np.ndarray) -> tuple[list[int], list[int]]:
    masks = (act_a & act_b, act_a & ~act_b, ~act_a & act_b, ~act_a & ~act_b)
    n = [int(m.sum()) for m in masks]
    s = [int(np.unique(scaf[m]).size) for m in masks]
    return n, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-single", type=int, default=R.MAX_SINGLE)
    ap.add_argument("--max-pair", type=int, default=R.MAX_PAIR)
    args = ap.parse_args()
    t0 = time.perf_counter()

    data = K.Data()
    singles, sstats = R.mine_single(data)
    print(f"[A] singles {len(singles)} {sstats}", flush=True)
    for c in singles:
        c["sources"] = {k: {"n_splits": v["n_splits"], "n_trees": len(v["trees"]),
                            "trees": sorted(v["trees"])} for k, v in c["sources"].items()}
    K.write_json(K.ANALYSIS / "phaseA_singles.json",
                 {"stats": sstats, "rules": [K_clean(c) for c in singles]})

    pairs, pstats = R.mine_pairs(data, singles, max_pairs=args.max_pair)
    print(f"[A] real pairs {len(pairs)} {pstats}", flush=True)

    # ---- matched random control pairs -------------------------------------
    train = data.rows["train"]
    ytr = data.y[train]
    scaf_tr = data.scaffold_code[train]
    act = {c["rule_id"]: R.rule_activation(data, c, train).astype(bool) for c in singles}
    freq = {k: float(v.mean()) for k, v in act.items()}
    fam = {c["rule_id"]: R.data_family(data, c["key"]) for c in singles}
    gran = {c["rule_id"]: c["granularity"] for c in singles}
    co_path = {(p["a"], p["b"]): p["co_path"] for p in pairs}

    rng = np.random.default_rng(K.SEED)
    ids = [c["rule_id"] for c in singles]
    used: set[str] = set(p["a"] for p in pairs) | set(p["b"] for p in pairs)
    randoms: list[dict] = []
    match_log: list[dict] = []
    for p in pairs:
        target = {"gran_pair": p["gran_pair"], "family_pair": p["family_pair"],
                  "freq_a": p["freq_a"], "freq_b": p["freq_b"], "freq_joint": p["freq_joint"],
                  "min_cell": min(p["cell_n"])}
        best = None
        for attempt in range(400):
            a, b = rng.choice(len(ids), size=2, replace=False)
            ra, rb = ids[a], ids[b]
            if ra == rb:
                continue
            if gran[ra] != gran[rb]:
                continue
            if (ra, rb) in co_path or (rb, ra) in co_path:
                continue
            gp = "+".join(sorted([gran[ra], gran[rb]]))
            fp = "+".join(sorted([fam[ra], fam[rb]]))
            # same granularity pair is mandatory (the point of the C+F booster is
            # unavailable); the family pair is relaxed only after 200 exact tries
            if gp != target["gran_pair"]:
                continue
            if fp != target["family_pair"] and attempt < 200:
                continue
            aa, bb = act[ra], act[rb]
            cn, cs = _cell_supports(aa, bb, scaf_tr)
            if min(cn) < R.PAIR_MIN_CELL_N or min(cs) < R.PAIR_MIN_CELL_SCAF:
                continue
            if bool(((aa & bb) == aa).all() or ((aa & bb) == bb).all()):
                continue
            cost = (abs(freq[ra] - target["freq_a"]) + abs(freq[rb] - target["freq_b"])
                    + 2.0 * abs(float((aa & bb).mean()) - target["freq_joint"])
                    + 2.0 * abs(np.log1p(min(cn)) - np.log1p(target["min_cell"])) / 3.0
                + (0.0 if fp == target["family_pair"] else 0.05))
            if best is None or cost < best[0]:
                best = (cost, ra, rb, cn, cs, cost)
            if cost < 0.02:
                break
        if best is None:
            match_log.append({"real_pair": p["pair_id"], "matched": False})
            continue
        cost, ra, rb, cn, cs, _ = best
        randoms.append({"pair_id": f"N{len(randoms):03d}", "a": ra, "b": rb,
                        "family_pair_exact": bool(fp == target["family_pair"]),
                        "matched_real_pair": p["pair_id"],
                        "freq_a": freq[ra], "freq_b": freq[rb],
                        "freq_joint": float((act[ra] & act[rb]).mean()),
                        "cell_n": cn, "cell_scaf": cs,
                        "gran_pair": target["gran_pair"], "family_pair": target["family_pair"],
                        "match_cost": float(cost)})
        match_log.append({"real_pair": p["pair_id"], "matched": True,
                          "random_pair": f"N{len(randoms)-1:03d}", "cost": float(cost)})

    n_matched = len(randoms)
    resid = {
        "freq_a": [abs(r["freq_a"] - pairs[i]["freq_a"]) for i, r in
                   enumerate(randoms)] if n_matched else [],
        "freq_b": [abs(r["freq_b"] - pairs[i]["freq_b"]) for i, r in enumerate(randoms)] if n_matched else [],
        "freq_joint": [abs(r["freq_joint"] - pairs[i]["freq_joint"]) for i, r in enumerate(randoms)] if n_matched else [],
        "min_cell": [abs(min(r["cell_n"]) - min(pairs[i]["cell_n"])) for i, r in enumerate(randoms)] if n_matched else [],
    }
    match_stats = {
        "n_real_pairs": len(pairs), "n_random_pairs": n_matched,
        "coverage": float(n_matched / max(1, len(pairs))),
        "median_abs_diff": {k: float(np.median(v)) if v else None for k, v in resid.items()},
        "max_abs_diff": {k: float(np.max(v)) if v else None for k, v in resid.items()},
    }
    print(f"[A] random pairs {n_matched} coverage={match_stats['coverage']:.2f}", flush=True)

    payload = {
        "round_id": K.ROUND_ID,
        "phase": "A: candidate rule discovery",
        "seconds": float(time.perf_counter() - t0),
        "single_stats": sstats,
        "pair_stats": pstats,
        "match_stats": match_stats,
        "match_log": match_log,
        "singles": [K_clean(c) for c in singles],
        "pairs": [K_clean(p) for p in pairs],
        "randoms": randoms,
        "boosters": {k: {"path": K.rel(v), "sha256": K.sha256_file(v)} for k, v in K.BOOSTERS.items()},
        "bundle": {"path": K.rel(K.BUNDLE), "sha256": K.sha256_file(K.BUNDLE)},
        "git": K.git_state(),
        "environment": K.environment(),
    }
    K.write_json(K.ANALYSIS / "phaseA_rules.json", payload)
    write_registry(data, singles, pairs, randoms, match_stats)
    print(f"[A] done in {time.perf_counter() - t0:.1f}s", flush=True)


def K_clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in ("train_xy", "activation_train", "joint_train"):
            continue
        out[k] = v
    return out


def write_registry(data: K.Data, singles: list[dict], pairs: list[dict],
                   randoms: list[dict], match_stats: dict) -> None:
    import pandas as pd

    reg = K.registry()
    rows = []
    for c in singles:
        gran, col = c["granularity"], c["column"]
        meta = reg[gran].iloc[col]
        arm_sources = "+".join(sorted(c["sources"]))
        trees = {a: c["sources"][a]["trees"] for a in c["sources"]}
        rows.append({
            "rule_id": c["rule_id"], "kind": "SINGLE", "rank": c["rank"],
            "granularity": gran, "column": int(col), "family": str(meta["family"]),
            "canonical_key": str(meta["canonical_key"]),
            "node_signature": str(meta["node_signature"]), "bond_signature": str(meta["bond_signature"]),
            "element_sequence": str(meta["element_sequence"]),
            "hybridization_sequence": str(meta["hybridization_sequence"]),
            "condition": f"x_{gran}[{col}] >= {c['threshold']:.8g}",
            "threshold": c["threshold"], "raw_count_equivalent": bool(c["raw_count_equivalent"]),
            "source_arms": arm_sources, "n_split_nodes": c["n_splits"],
            "n_source_trees": c["n_source_trees"],
            "source_trees": ";".join(f"{a}:{len(t)}" for a, t in trees.items()),
            "train_ge_n": c["n_train_ge"], "train_lt_n": c["n_train_lt"],
            "train_ge_scaf": c["train_scaffold_ge"], "train_lt_scaf": c["train_scaffold_lt"],
            "train_ge_pos": c["train_pos_ge"], "train_lt_pos": c["train_pos_lt"],
            "status": "SELECTED_SINGLE",
            "note": f"merged duplicates: {len(c['merged'])}",
        })
    for p in pairs:
        rows.append({"rule_id": p["pair_id"], "kind": "JOINT_REAL", "rank": p["rank"],
                     "granularity": p["gran_pair"], "column": "", "family": p["family_pair"],
                     "canonical_key": "", "node_signature": "", "bond_signature": "",
                     "element_sequence": "", "hybridization_sequence": "",
                     "condition": f"{p['a']} AND {p['b']}",
                     "threshold": "", "raw_count_equivalent": "",
                     "source_arms": "+".join(sorted({a for i in (p['ia'], p['ib'])
                                                     for a in singles[i]['sources']})),
                     "n_split_nodes": p["co_path"], "n_source_trees": p["co_path"],
                     "source_trees": f"co_path={p['co_path']}",
                     "train_ge_n": p["cell_n"][0], "train_lt_n": sum(p["cell_n"][1:]),
                     "train_ge_scaf": p["cell_scaf"][0], "train_lt_scaf": max(p["cell_scaf"][1:]),
                     "train_ge_pos": "", "train_lt_pos": "",
                     "status": "SELECTED_JOINT_REAL",
                     "note": f"cells n={p['cell_n']} scaf={p['cell_scaf']} freq_joint={p['freq_joint']:.4f}"})
    for r in randoms:
        rows.append({"rule_id": r["pair_id"], "kind": "JOINT_RANDOM", "rank": "",
                     "granularity": r["gran_pair"], "column": "", "family": r["family_pair"],
                     "canonical_key": "", "node_signature": "", "bond_signature": "",
                     "element_sequence": "", "hybridization_sequence": "",
                     "condition": f"{r['a']} AND {r['b']}", "threshold": "",
                     "raw_count_equivalent": "", "source_arms": "matched_control",
                     "n_split_nodes": 0, "n_source_trees": 0,
                     "source_trees": f"matched_to={r['matched_real_pair']}",
                     "train_ge_n": r["cell_n"][0], "train_lt_n": sum(r["cell_n"][1:]),
                     "train_ge_scaf": r["cell_scaf"][0], "train_lt_scaf": max(r["cell_scaf"][1:]),
                     "train_ge_pos": "", "train_lt_pos": "",
                     "status": "SELECTED_JOINT_MATCHED_RANDOM",
                     "note": f"cells n={r['cell_n']} scaf={r['cell_scaf']} freq_joint={r['freq_joint']:.4f}"})
    df = pd.DataFrame(rows)
    K.ANALYSIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(K.ANALYSIS / "rule_registry.csv", index=False)
    print(f"[A] registry rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
