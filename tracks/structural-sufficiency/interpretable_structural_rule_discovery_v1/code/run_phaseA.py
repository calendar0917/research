"""Phase A runner (ISRD-v1): repaired single rules, real co-path pairs, mapping checks.

Writes
    analysis/phaseA_rules.json     single pool, real pair pool, all real path pairs
    analysis/phaseA_mapping.json   the A1/A2 repair evidence
    analysis/rule_registry.csv     every selected single / real joint rule

Train only for all rule statistics.  No Valid/Test label is read here.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

import isrd1_common as K
import isrd1_rules as R


def key_to_json(key: tuple) -> list:
    return [key[0], int(key[1]), float(key[2]), key[3]]


def key_from_json(val: list) -> tuple:
    return (str(val[0]), int(val[1]), R.f32(float(val[2])), str(val[3]))


def clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in ("activation_train", "joint_train", "x_train", "key"):
            continue
        if k == "sources":
            out[k] = {a: {"n_nodes": s["n_nodes"], "n_trees": len(s["trees"]),
                          "trees": sorted(int(t) for t in s["trees"])}
                      for a, s in v.items()}
        elif k in ("key_a", "key_b"):
            out[k] = key_to_json(v)
        else:
            out[k] = v
    return out


def mapping_checks(data: K.Data, rng: np.random.Generator) -> dict[str, Any]:
    """A1: direct column-value checks between the booster design and the mapped block."""
    import xgboost as xgb

    sys.path.insert(0, str(K.SGA / "code"))
    from run_phaseA_xgb import design_rows, unpack_h

    with np.load(K.BUNDLE) as z:
        bundle = {k: np.asarray(z[k]) for k in z.files}
    h = unpack_h(bundle)
    rows = np.sort(rng.choice(data.rows["train"], size=2000, replace=False))
    out: dict[str, Any] = {"h_width": int(h.shape[1]), "h_has_nan": bool(np.isnan(h).any()),
                           "structural_blocks_have_nan": {
                               g: bool(np.isnan(data.block[g].data).any()) for g in ("C", "F")},
                           "arms": {}}
    for arm in ("X-C", "X-F"):
        b = R.load_booster(arm)
        width = R.booster_width(b)
        gran, bw = K.ARM_BLOCK[arm]
        nodes = R.split_nodes(b, arm)
        assert all(nd["granularity"] == gran for nd in nodes), "non-arm block decoded"
        X = design_rows(h, [(gran, data.block[gran])], rows)
        # value identity for every distinct structural feature of the arm
        feats = sorted({nd["feature"] for nd in nodes})
        worst = 0.0
        checked = 0
        for f in feats:
            g, col = K.decode_feature(arm, f)
            assert g == gran
            ours = data.sub32(g, col, rows)
            worst = max(worst, float(np.max(np.abs(X[:, f] - ours))))
            checked += 1
        # boundary / block identity
        f0 = K.structure_feature(arm, gran, 0)
        boundary = {"feature": int(f0), "decoded": list(K.decode_feature(arm, f0)),
                    "matches_block_col0": float(np.max(np.abs(
                        X[:, f0] - data.sub32(gran, 0, rows)))),
                    "matches_other_block_col0": float(np.max(np.abs(
                        X[:, f0] - data.sub32("F" if gran == "C" else "C", 0, rows))))}
        out["arms"][arm] = {
            "booster_num_features": width,
            "expected_width": K.ARM_WIDTH[arm],
            "block": gran, "block_width": bw,
            "n_structural_split_nodes": len(nodes),
            "n_distinct_structural_features": len(feats),
            "granularities_decoded": sorted({nd["granularity"] for nd in nodes}),
            "max_abs_value_diff_mapped_vs_booster_column": worst,
            "n_features_value_checked": checked,
            "boundary_f2258": boundary if arm == "X-F" else None,
            "n_rows_checked": int(rows.size),
        }
    # branch / threshold / direction replay on the real design
    out["path_replay"] = {}
    for arm in ("X-C", "X-F"):
        gran, _ = K.ARM_BLOCK[arm]
        b = R.load_booster(arm)
        X = design_rows(h, [(gran, data.block[gran])], rows)
        dm = xgb.DMatrix(X)
        leaf = b.predict(dm, pred_leaf=True).astype(np.int64)
        dumps = [json.loads(t) for t in b.get_dump(dump_format="json")]
        assign = np.full((rows.size, leaf.shape[1]), -1, dtype=np.int64)
        checked_trees = min(30, leaf.shape[1])
        for tree in range(checked_trees):
            # locate every leaf by walking the dump with the real feature values
            root = dumps[tree]
            stack = [(root, np.arange(rows.size))]
            while stack:
                node, idx = stack.pop()
                if idx.size == 0:
                    continue
                if "leaf" in node:
                    assign[idx, tree] = int(node["nodeid"])
                    continue
                feat = int(node["split"][1:])
                thr = np.float32(node["split_condition"])
                v = X[idx, feat]
                yes = v < thr                       # non-missing: yes branch
                children = {int(c["nodeid"]): c for c in node["children"]}
                stack.append((children[int(node["yes"])], idx[yes]))
                stack.append((children[int(node["no"])], idx[~yes]))
        mism = {t: int((assign[:, t] != leaf[:, t]).sum()) for t in range(checked_trees)}
        out["path_replay"][arm] = {"trees_checked": checked_trees, "n_rows": int(rows.size),
                                   "n_leaf_mismatch": sum(mism.values()),
                                   "per_tree_mismatch": mism,
                                   "missing_branch_used": False,
                                   "note": "leaf ids from directed real-path traversal vs "
                                           "xgboost pred_leaf on the reconstructed arm design"}
        del X, dm
    return out


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(K.SEED)
    data = K.Data()
    print(f"[A] data loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

    mapping = mapping_checks(data, rng)
    assert all(v["max_abs_value_diff_mapped_vs_booster_column"] == 0.0
               for v in mapping["arms"].values()), "mapped structure columns disagree"
    assert not any(mapping["structural_blocks_have_nan"].values())
    print("[A] mapping checks passed", flush=True)

    singles, sstats = R.mine_single(data)
    print(f"[A] singles {len(singles)} {json.dumps(sstats['filtered_reasons'])}", flush=True)
    pairs, pstats, all_pairs = R.mine_pairs(data, singles)
    print(f"[A] pairs {len(pairs)} gran={pstats['gran_pair_counts']} "
          f"mixed={pstats['n_mixed_granularity_pairs']}", flush=True)
    assert pstats["n_mixed_granularity_pairs"] == 0, "C-F pair from a single-granularity booster"
    assert not any(p["gran_pair"] == "C+F" for p in pairs)

    payload = {
        "round_id": K.ROUND_ID,
        "phase": "A: repaired candidate rule discovery",
        "seconds": float(time.perf_counter() - t0),
        "single_stats": sstats,
        "pair_stats": pstats,
        "singles": [clean(c) for c in singles],
        "pairs": [clean(p) for p in pairs],
        "all_real_path_pairs": [[key_to_json(a), key_to_json(b)] for (a, b) in all_pairs],
        "boosters": {a: {"path": K.rel(p), "sha256": K.sha256_file(p),
                         "num_features": R.booster_width(R.load_booster(a))}
                     for a, p in K.BOOSTERS.items()},
        "bundle": {"path": K.rel(K.BUNDLE), "sha256": K.sha256_file(K.BUNDLE)},
        "git": K.git_state(),
        "environment": K.environment(),
    }
    K.write_json(K.ANALYSIS / "phaseA_rules.json", payload)
    K.write_json(K.ANALYSIS / "phaseA_mapping.json", mapping)
    write_registry(data, singles, pairs)
    print(f"[A] done in {time.perf_counter() - t0:.1f}s", flush=True)


def write_registry(data: K.Data, singles: list[dict], pairs: list[dict]) -> None:
    import pandas as pd

    reg = K.registry()
    rows = []
    for c in singles:
        meta = reg[c["granularity"]].iloc[c["column"]]
        rows.append({
            "rule_id": c["rule_id"], "kind": "SINGLE", "rank": c["rank"],
            "granularity": c["granularity"], "column": int(c["column"]),
            "family": str(meta["family"]), "canonical_key": str(meta["canonical_key"]),
            "node_signature": str(meta["node_signature"]),
            "bond_signature": str(meta["bond_signature"]),
            "element_sequence": str(meta["element_sequence"]),
            "condition": R.condition_key_str(c["key"]),
            "threshold": c["threshold"], "direction": c["direction"],
            "raw_count_k_empirical": c["reading"].get("empirically_equivalent_to_count_ge"),
            "source_arms": "+".join(sorted(c["sources"])),
            "n_source_trees": c["n_source_trees"], "n_split_nodes": c["n_nodes"],
            "train_active_n": c["n_train_active"], "train_inactive_n": c["n_train_inactive"],
            "train_active_scaf": c["train_active_scaf"],
            "train_inactive_scaf": c["train_inactive_scaf"],
            "train_active_pos": c["train_active_pos"],
            "train_inactive_pos": c["train_inactive_pos"],
            "n_merged_duplicates": len(c["merged"]),
            "status": "SELECTED_SINGLE",
        })
    for p in pairs:
        rows.append({
            "rule_id": p["pair_id"], "kind": "JOINT_REAL", "rank": p["rank"],
            "granularity": p["gran_pair"], "column": "", "family": p["family_pair"],
            "canonical_key": "", "node_signature": "", "bond_signature": "",
            "element_sequence": "", "condition": f"{p['condition_a']} AND {p['condition_b']}",
            "threshold": "", "direction": "", "raw_count_k_empirical": "",
            "source_arms": "+".join(sorted({p["key_a"][0], p["key_b"][0]})),
            "n_source_trees": p["co_path"], "n_split_nodes": p["co_path"],
            "train_active_n": p["cell_n"][0], "train_inactive_n": sum(p["cell_n"][1:]),
            "train_active_scaf": p["cell_scaf"][0], "train_inactive_scaf": max(p["cell_scaf"][1:]),
            "train_active_pos": int((data.y[data.rows["train"]][
                R.pair_activation(data, p, {c["rule_id"]: c for c in singles},
                                  data.rows["train"]).astype(bool)]).sum()),
            "train_inactive_pos": "", "n_merged_duplicates": 0,
            "status": "SELECTED_JOINT_REAL",
        })
    df = pd.DataFrame(rows)
    K.ANALYSIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(K.ANALYSIS / "rule_registry.csv", index=False)
    print(f"[A] registry rows={len(df)}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(K.SGA / "code"))
    main()
