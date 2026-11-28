"""Phase A rule discovery with repaired arm mapping and real path semantics (ISRD-v1).

Repairs relative to v0 (see ``ERRATA.md``):

* A1 — structural feature indices are decoded **per arm** (X-C: H+C, X-F: H+F)
  through :func:`isrd1_common.decode_feature`; range checks, no interval guessing.
* A2 — every root-to-leaf path is read node by node, keeping the actual threshold,
  the branch direction actually taken and the missing direction.  A condition is
  ``(granularity, column, float32 threshold, direction)`` with direction ``ge``
  (``x >= t``) or ``lt`` (``x < t``).
* A3 — single candidates are *directed* conditions; dedup by Train activation keeps
  every source (arm/tree/node) and every merged condition.  Joint candidates are
  real co-path pairs of two different structures, both conditions directed.
* A4 — joint main-effect eligibility uses the exact condition identity (complement
  allowed, threshold must match), and the matched random control is drawn from the
  exact-condition closure of the selected U64 pool with explicit tolerances.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np

import isrd1_common as K

FAMILY_NAME = {0: "size-2", 1: "size-3", 2: "size-4", 3: "cycle-5", 4: "cycle-6"}
MIN_SIDE_N = 30
MIN_SIDE_SCAF = 5
PAIR_MIN_CELL_N = 20
PAIR_MIN_CELL_SCAF = 3
MAX_SINGLE = 512
MAX_PAIR = 256
MAX_PREFILTER = 12000
FREQ_TOL = 0.05
JOINT_TOL = 0.03


# ---------------------------------------------------------------------------
# condition identity
# ---------------------------------------------------------------------------


def f32(value: float) -> float:
    return float(np.float32(value))


def condition_key(cond: dict) -> tuple[str, int, float, str]:
    return (cond["granularity"], int(cond["column"]), f32(cond["threshold"]),
            str(cond["direction"]))


def complement(key: tuple[str, int, float, str]) -> tuple[str, int, float, str]:
    return (key[0], key[1], key[2], "lt" if key[3] == "ge" else "ge")


def key_to_json(key: tuple) -> list:
    return [key[0], int(key[1]), float(key[2]), key[3]]


def key_from_json(val: list) -> tuple:
    return (str(val[0]), int(val[1]), f32(float(val[2])), str(val[3]))


def condition_key_str(key: tuple[str, int, float, str]) -> str:
    return f"{key[0]}[{key[1]}] {'>=' if key[3] == 'ge' else '<'} {key[2]:.9g}"


# ---------------------------------------------------------------------------
# boosters and real tree paths
# ---------------------------------------------------------------------------


def load_booster(arm: str):
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(K.BOOSTERS[arm]))
    return b


def booster_width(booster) -> int:
    return int(json.loads(booster.save_config())["learner"]["learner_model_param"]["num_feature"])


def split_nodes(booster, arm: str) -> list[dict[str, Any]]:
    """Every split node of one booster with its arm-decoded structural identity."""
    df = booster.trees_to_dataframe()
    f = df["Feature"].astype(str)
    ok = f.str.match(r"^f\d+$").to_numpy()
    out: list[dict[str, Any]] = []
    for row in df[ok].itertuples(index=False):
        gidx = int(str(row.Feature)[1:])
        gran, col = K.decode_feature(arm, gidx)          # raises on out-of-range
        rec = {"arm": arm, "tree": int(row.Tree), "node": int(str(row.ID).split("-")[-1]),
               "feature": gidx, "granularity": gran, "column": int(col),
               "threshold": f32(float(row.Split)),
               "gain": float(row.Gain), "cover": float(row.Cover)}
        if gran != "H":
            rec["direction"] = "ge"                        # split node itself = no branch
            out.append(rec)
    return out


def _child_map(node: dict) -> dict[int, dict]:
    return {int(c["nodeid"]): c for c in node["children"]}


def path_conditions(booster, arm: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Real root-to-leaf structural conditions, per leaf.

    Returns ``(tree, leaf_node) -> {"conditions": tuple[cond, ...], "n_h_splits": int,
    "leaf_value": float}`` where each condition carries its own threshold, the
    direction actually taken and the node/missing metadata.
    """
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for tree, tree_json in enumerate(booster.get_dump(dump_format="json")):
        root = json.loads(tree_json)

        def walk(node: dict, path: list[dict], n_h: int) -> None:
            if "leaf" in node:
                out[(tree, int(node["nodeid"]))] = {
                    "conditions": tuple(path), "n_h_splits": n_h,
                    "leaf_value": float(node["leaf"])}
                return
            feat = int(node["split"][1:])
            gran, col = K.decode_feature(arm, feat)
            thr = f32(float(node["split_condition"]))
            children = _child_map(node)
            yes_id, no_id = int(node["yes"]), int(node["no"])
            missing = int(node["missing"])
            base = {"arm": arm, "tree": int(tree), "node": int(node["nodeid"]),
                    "feature": feat, "granularity": gran, "column": int(col),
                    "threshold": thr,
                    "missing_goes": "yes" if missing == yes_id else "no"}
            if gran == "H":
                walk(children[yes_id], path, n_h + 1)
                walk(children[no_id], path, n_h + 1)
            else:
                # yes branch: x < t ; no branch: x >= t   (non-missing values)
                walk(children[yes_id],
                     path + [{**base, "direction": "lt"}], n_h)
                walk(children[no_id],
                     path + [{**base, "direction": "ge"}], n_h)

        walk(root, [], 0)
    return out


def path_pairs(paths: dict[tuple[int, int], dict[str, Any]]) -> dict[tuple, dict]:
    """All real co-path condition pairs (different structures), with path sources."""
    pairs: dict[tuple, dict] = {}
    for (tree, leaf), rec in paths.items():
        keys = [condition_key(c) for c in rec["conditions"]]
        # deduplicate identical conditions on one path (same condition twice is not
        # a two-structure pair anyway)
        uniq = list(dict.fromkeys(keys))
        for a, b in itertools.combinations(sorted(uniq, key=condition_key_str), 2):
            if a[0] == b[0] and a[1] == b[1]:
                continue
            rec2 = pairs.setdefault((a, b), {"co_path_nodes": 0, "paths": []})
            rec2["co_path_nodes"] += 1
            if len(rec2["paths"]) < 12:
                rec2["paths"].append({"tree": int(tree), "leaf": int(leaf),
                                      "n_conditions": len(keys),
                                      "n_h_splits": int(rec["n_h_splits"])})
    return pairs


# ---------------------------------------------------------------------------
# single-condition candidates
# ---------------------------------------------------------------------------


def mine_single(data: K.Data, max_prefilter: int = MAX_PREFILTER) -> tuple[list[dict], dict]:
    """Directed single-structure conditions, Train-only support and dedup."""
    nodes: list[dict] = []
    arm_nodes: dict[str, int] = {}
    for arm in ("X-C", "X-F"):
        b = load_booster(arm)
        assert booster_width(b) == K.ARM_WIDTH[arm], (arm, booster_width(b))
        nd = split_nodes(b, arm)
        arm_nodes[arm] = len(nd)
        nodes.extend(nd)

    agg: dict[tuple, dict] = {}
    for nd in nodes:
        for direction in ("ge", "lt"):
            key = (nd["granularity"], nd["column"], nd["threshold"], direction)
            rec = agg.setdefault(key, {"granularity": key[0], "column": key[1],
                                       "threshold": key[2], "direction": key[3],
                                       "sources": {}, "nodes": [], "n_nodes": 0,
                                       "total_gain": 0.0})
            src = rec["sources"].setdefault(nd["arm"], {"n_nodes": 0, "trees": set()})
            src["n_nodes"] += 1
            src["trees"].add(nd["tree"])
            rec["n_nodes"] += 1
            rec["total_gain"] += nd["gain"]
            if len(rec["nodes"]) < 64:
                rec["nodes"].append({"arm": nd["arm"], "tree": nd["tree"],
                                     "node": nd["node"], "branch": direction})

    ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["n_nodes"], -kv[1]["total_gain"],
                                                 condition_key_str(kv[0])))
    truncated = max(0, len(ranked) - max_prefilter)
    ranked = ranked[:max_prefilter]

    train = data.rows["train"]
    ytr = data.y[train]
    scaf_tr = data.scaffold_code[train]
    cands: list[dict] = []
    reasons: dict[str, int] = {}
    for key, rec in ranked:
        x = data.sub32(rec["granularity"], rec["column"], train)
        thr = np.float32(rec["threshold"])
        act = (x >= thr) if rec["direction"] == "ge" else (x < thr)
        n_act = int(act.sum())
        n_inact = int((~act).sum())
        if n_act == 0 or n_inact == 0:
            reasons["constant"] = reasons.get("constant", 0) + 1
            continue
        if min(n_act, n_inact) < MIN_SIDE_N:
            reasons["min_side_molecules"] = reasons.get("min_side_molecules", 0) + 1
            continue
        scaf_act = int(np.unique(scaf_tr[act]).size)
        scaf_inact = int(np.unique(scaf_tr[~act]).size)
        if min(scaf_act, scaf_inact) < MIN_SIDE_SCAF:
            reasons["min_side_scaffolds"] = reasons.get("min_side_scaffolds", 0) + 1
            continue
        cands.append({**rec, "key": key, "activation_train": act,
                      "n_train_active": n_act, "n_train_inactive": n_inact,
                      "train_active_scaf": scaf_act, "train_inactive_scaf": scaf_inact,
                      "train_active_pos": int(ytr[act].sum()),
                      "train_inactive_pos": int(ytr[~act].sum()),
                      "x_train": x})

    cands.sort(key=lambda c: (-c["n_nodes"], -len(c["sources"].get("X-C", {}).get("trees", ()))
                              - len(c["sources"].get("X-F", {}).get("trees", ())),
                              condition_key_str(c["key"])))
    for c in cands:
        c.pop("x_train", None)
    kept: list[dict] = []
    seen: dict[bytes, int] = {}
    n_dup = 0
    for c in cands:
        sig = np.packbits(c["activation_train"]).tobytes()
        if sig in seen:
            k = seen[sig]
            kept[k]["merged"].append({"condition": condition_key_str(c["key"]),
                                      "granularity": c["granularity"], "column": c["column"],
                                      "threshold": c["threshold"], "direction": c["direction"],
                                      "source_arms": sorted(c["sources"]),
                                      "n_nodes": c["n_nodes"]})
            kept[k]["n_nodes"] += c["n_nodes"]
            n_dup += 1
            continue
        seen[sig] = len(kept)
        c["merged"] = []
        kept.append(c)

    for c in kept:
        c["reading"] = threshold_reading(c, data)
        c["n_source_trees"] = sum(len(v["trees"]) for v in c["sources"].values())
        c["stable_id"] = condition_key_str(c["key"])
    kept.sort(key=lambda c: (-c["n_source_trees"], -c["n_nodes"], c["stable_id"]))
    selected = kept[:MAX_SINGLE]
    for i, c in enumerate(selected):
        c["rule_id"] = f"S{i:03d}"
        c["rank"] = i
    for c in kept[len(selected):]:
        c.pop("activation_train", None)
    stats = {"n_split_nodes": len(nodes), "n_split_nodes_by_arm": arm_nodes,
             "n_distinct_conditions": len(agg), "n_prefilter_truncated": truncated,
             "n_after_support_filter": len(cands), "n_after_dedup": len(kept),
             "n_duplicate_merged": n_dup, "n_selected": len(selected),
             "n_dropped_by_cap": max(0, len(kept) - len(selected)),
             "min_side_molecules": MIN_SIDE_N, "min_side_scaffolds": MIN_SIDE_SCAF,
             "filtered_reasons": reasons}
    return selected, stats


def threshold_reading(c: dict, data: K.Data) -> dict[str, Any]:
    """Verify on Train what the scaled threshold means in raw occurrence counts."""
    raw = data.raw_count(c["granularity"], c["column"])[data.rows["train"]]
    act = c["activation_train"]
    out: dict[str, Any] = {"train_raw_min_active": int(raw[act].min()) if act.any() else None,
                           "train_raw_max_inactive": int(raw[~act].max()) if (~act).any() else None}
    for k in (1, 2, 3):
        out[f"train_equals_raw_ge_{k}"] = bool(np.array_equal(act, raw >= k))
    k = int(raw[act].min()) if act.any() else None
    if k is not None:
        out["k_min_active"] = k
        out["train_equals_raw_ge_kmin"] = bool(np.array_equal(act, raw >= k))
        out["empirically_equivalent_to_count_ge"] = (k if out["train_equals_raw_ge_kmin"] else None)
    return out


def rule_activation(data: K.Data, rule: dict, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        x = data.col(rule["granularity"], rule["column"])        # all 41 127 rows
    else:
        x = data.sub32(rule["granularity"], rule["column"], rows)
    thr = np.float32(rule["threshold"])
    act = (x >= thr) if rule["direction"] == "ge" else (x < thr)
    return act.astype(np.float64)


def rule_matrix(data: K.Data, rules: list[dict], rows: np.ndarray | None = None) -> np.ndarray:
    if not rules:
        n = data.y.size if rows is None else rows.size
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack([rule_activation(data, r, rows) for r in rules])


# ---------------------------------------------------------------------------
# real joint candidates
# ---------------------------------------------------------------------------


def mine_pairs(data: K.Data, singles: list[dict], max_pairs: int = MAX_PAIR) -> tuple[list[dict], dict]:
    """Real co-path pairs (PROJECTED_PAIR) restricted to the single-rule pool."""
    index = {c["key"]: i for i, c in enumerate(singles)}
    paths_all: dict[tuple[int, int], dict[str, Any]] = {}
    for arm in ("X-C", "X-F"):
        paths_all.update(path_conditions(load_booster(arm), arm))
    all_pairs = path_pairs(paths_all)

    train = data.rows["train"]
    ytr = data.y[train]
    scaf_tr = data.scaffold_code[train]
    sindex = {c["rule_id"]: c for c in singles}
    cache = {i: data.sub32(c["granularity"], c["column"], train)
             for i, c in enumerate(singles)}

    def acts(i: int) -> np.ndarray:
        c = singles[i]
        x = cache[i]
        return (x >= np.float32(c["threshold"])) if c["direction"] == "ge" else \
            (x < np.float32(c["threshold"]))

    cands: list[dict] = []
    reasons: dict[str, int] = {}
    n_real_path_pairs = 0
    for (ka, kb), rec in all_pairs.items():
        ia, ib = index.get(ka), index.get(kb)
        if ia is None or ib is None:
            reasons["condition_outside_pool"] = reasons.get("condition_outside_pool", 0) + 1
            continue
        n_real_path_pairs += 1
        if singles[ia]["granularity"] != singles[ib]["granularity"]:
            reasons["mixed_granularity"] = reasons.get("mixed_granularity", 0) + 1
            continue
        if singles[ia]["granularity"] == "C" and singles[ib]["granularity"] == "C":
            gran_pair = "C+C"
        elif singles[ia]["granularity"] == "F" and singles[ib]["granularity"] == "F":
            gran_pair = "F+F"
        else:
            raise AssertionError("mixed C-F pair from a single-granularity booster")
        a, b = acts(ia), acts(ib)
        ab = a & b
        masks = (ab, a & ~b, ~a & b, ~a & ~b)
        cell_n = [int(m.sum()) for m in masks]
        if min(cell_n) < PAIR_MIN_CELL_N:
            reasons["cell_support_molecules"] = reasons.get("cell_support_molecules", 0) + 1
            continue
        cell_s = [int(np.unique(scaf_tr[m]).size) for m in masks]
        if min(cell_s) < PAIR_MIN_CELL_SCAF:
            reasons["cell_support_scaffolds"] = reasons.get("cell_support_scaffolds", 0) + 1
            continue
        if ab.sum() == 0 or (~(a | b)).sum() == 0:
            reasons["constant"] = reasons.get("constant", 0) + 1
            continue
        if np.array_equal(a, b) or np.array_equal(ab, a) or np.array_equal(ab, b):
            reasons["containment"] = reasons.get("containment", 0) + 1
            continue
        fam_pair = "+".join(sorted([data_family(data, ka), data_family(data, kb)]))
        cands.append({"a": singles[ia]["rule_id"], "b": singles[ib]["rule_id"],
                      "ia": ia, "ib": ib, "co_path": rec["co_path_nodes"],
                      "path_examples": rec["paths"],
                      "condition_a": condition_key_str(ka), "condition_b": condition_key_str(kb),
                      "key_a": ka, "key_b": kb,
                      "freq_a": float(a.mean()), "freq_b": float(b.mean()),
                      "freq_joint": float(ab.mean()), "joint_train": ab,
                      "cell_n": cell_n, "cell_scaf": cell_s,
                      "gran_pair": gran_pair, "family_pair": fam_pair})

    cands.sort(key=lambda c: (-c["co_path"], c["condition_a"], c["condition_b"]))
    kept: list[dict] = []
    seen: set[bytes] = set()
    for c in cands:
        sig = np.packbits(c["joint_train"]).tobytes()
        if sig in seen:
            reasons["duplicate_activation"] = reasons.get("duplicate_activation", 0) + 1
            continue
        seen.add(sig)
        c["pair_id"] = f"P{len(kept):03d}"
        c["rank"] = len(kept)
        kept.append(c)
    selected = kept[:max_pairs]
    for c in selected:
        del c["joint_train"]
    stats = {"n_paths": len(paths_all), "n_real_path_pairs": len(all_pairs),
             "n_real_path_pairs_inside_pool": n_real_path_pairs,
             "n_after_filters_and_dedup": len(kept), "n_selected": len(selected),
             "filtered_reasons": reasons,
             "min_cell_molecules": PAIR_MIN_CELL_N, "min_cell_scaffolds": PAIR_MIN_CELL_SCAF,
             "n_mixed_granularity_pairs": int(reasons.get("mixed_granularity", 0)),
             "gran_pair_counts": _counts(c["gran_pair"] for c in selected)}
    return selected, stats, all_pairs


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def data_family(data: K.Data, key: tuple[str, int, float, str]) -> str:
    gran, col = key[0], key[1]
    reg = K.registry()
    try:
        return str(reg[gran]["family"].iloc[col])
    except Exception:
        return "unknown"


def pair_activation(data: K.Data, pair: dict, by_id: dict[str, dict],
                    rows: np.ndarray | None = None) -> np.ndarray:
    a = rule_activation(data, by_id[pair["a"]], rows)
    b = rule_activation(data, by_id[pair["b"]], rows)
    return a * b


# ---------------------------------------------------------------------------
# A4: exact main-effect eligibility and matched random pairs
# ---------------------------------------------------------------------------


def u64_condition_closure(u64_rules: list[dict]) -> dict[tuple, dict]:
    """Selected U64 conditions plus their exact complements (same structure/threshold)."""
    pool: dict[tuple, dict] = {}
    for r in u64_rules:
        pool[r["key"]] = {"key": r["key"], "origin": "U64", "rule_id": r["rule_id"],
                          "rule": r}
        ck = complement(r["key"])
        if ck not in pool:
            pool[ck] = {"key": ck, "origin": "U64_COMPLEMENT", "rule_id": None,
                        "rule": {**r, "direction": ck[3], "key": ck}}
    return pool


def with_resolved_pairs(pairs: list[dict], pool: dict[tuple, dict]) -> list[dict]:
    """Attach the exact resolved condition records; drop pairs failing eligibility."""
    out = []
    for p in pairs:
        ra, rb = pool.get(p["key_a"]), pool.get(p["key_b"])
        if ra is None or rb is None:
            continue
        q = dict(p)
        q["resolved_a"] = {"key": p["key_a"], "origin": ra["origin"],
                           "u64_rule_id": ra["rule_id"]}
        q["resolved_b"] = {"key": p["key_b"], "origin": rb["origin"],
                           "u64_rule_id": rb["rule_id"]}
        out.append(q)
    return out


def match_random_pairs(data: K.Data, real: list[dict], pool: dict[tuple, dict],
                       all_real_pairs: dict[tuple, dict], seed: int = K.SEED) -> tuple[list[dict], dict]:
    """One-to-one matched random pairs from the U64 exact-condition closure."""
    train = data.rows["train"]
    scaf_tr = data.scaffold_code[train]
    keys = list(pool.keys())
    acts = np.column_stack([rule_activation(data, pool[k]["rule"], train).astype(bool)
                            for k in keys]) if keys else np.zeros((train.size, 0), dtype=bool)
    fam = [data_family(data, k) for k in keys]
    gran = [k[0] for k in keys]

    # candidate pair table (i<j), filtered by the same 4-cell support rule
    table: list[dict] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if gran[i] != gran[j]:
                continue
            if (keys[i][0], keys[i][1]) == (keys[j][0], keys[j][1]):
                continue
            if (keys[i], keys[j]) in all_real_pairs or (keys[j], keys[i]) in all_real_pairs:
                continue
            a, b = acts[:, i], acts[:, j]
            ab = a & b
            cell_n = [int(ab.sum()), int((a & ~b).sum()), int((~a & b).sum()),
                      int((~a & ~b).sum())]
            if min(cell_n) < PAIR_MIN_CELL_N:
                continue
            cell_s = [int(np.unique(scaf_tr[m]).size)
                      for m in (ab, a & ~b, ~a & b, ~a & ~b)]
            if min(cell_s) < PAIR_MIN_CELL_SCAF:
                continue
            if np.array_equal(a, b) or np.array_equal(ab, a) or np.array_equal(ab, b):
                continue
            table.append({"i": i, "j": j, "gran_pair": "+".join(sorted([gran[i], gran[j]])),
                          "family_pair": "+".join(sorted([fam[i], fam[j]])),
                          "freq_a": float(a.mean()), "freq_b": float(b.mean()),
                          "freq_joint": float(ab.mean()), "cell_n": cell_n,
                          "cell_scaf": cell_s})
    # index by (gran_pair, family_pair) for fast lookup
    by_gf: dict[tuple[str, str], list[int]] = {}
    for t, rec in enumerate(table):
        by_gf.setdefault((rec["gran_pair"], rec["family_pair"]), []).append(t)

    rng = np.random.default_rng(seed)
    used_pair: set[tuple[int, int]] = set()
    matched: list[dict] = []
    log: list[dict] = []
    for p in real:
        cands = by_gf.get((p["gran_pair"], p["family_pair"]), [])
        # allow the swapped assignment of the two marginals
        options = []
        for t in cands:
            rec = table[t]
            if (rec["i"], rec["j"]) in used_pair:
                continue
            for swap in (0, 1):
                fa, fb = (rec["freq_a"], rec["freq_b"]) if swap == 0 else (rec["freq_b"], rec["freq_a"])
                d1, d2 = abs(fa - p["freq_a"]), abs(fb - p["freq_b"])
                dj = abs(rec["freq_joint"] - p["freq_joint"])
                if d1 > FREQ_TOL or d2 > FREQ_TOL or dj > JOINT_TOL:
                    continue
                options.append((d1 + d2 + 2.0 * dj, d1, d2, dj, t, swap))
        if not options:
            log.append({"real_pair": p["pair_id"], "matched": False})
            continue
        options.sort(key=lambda o: (o[0], o[4], o[5]))
        cost, d1, d2, dj, t, swap = options[0]
        rec = table[t]
        used_pair.add((rec["i"], rec["j"]))
        ka, kb = (keys[rec["i"]], keys[rec["j"]]) if swap == 0 else (keys[rec["j"]], keys[rec["i"]])
        matched.append({"pair_id": f"N{len(matched):03d}", "key_a": ka, "key_b": kb,
                        "condition_a": condition_key_str(ka), "condition_b": condition_key_str(kb),
                        "matched_real_pair": p["pair_id"],
                        "gran_pair": p["gran_pair"], "family_pair": p["family_pair"],
                        "freq_a": float(rec["freq_a"] if swap == 0 else rec["freq_b"]),
                        "freq_b": float(rec["freq_b"] if swap == 0 else rec["freq_a"]),
                        "freq_joint": rec["freq_joint"], "cell_n": rec["cell_n"],
                        "cell_scaf": rec["cell_scaf"],
                        "abs_diff_freq_a": d1, "abs_diff_freq_b": d2, "abs_diff_freq_joint": dj,
                        "cost": float(cost),
                        "origin_a": pool[ka]["origin"], "origin_b": pool[kb]["origin"]})
        log.append({"real_pair": p["pair_id"], "matched": True,
                    "random_pair": matched[-1]["pair_id"], "cost": float(cost)})
    del acts

    n_real = len(real)
    resid = {
        "freq_a": [m["abs_diff_freq_a"] for m in matched],
        "freq_b": [m["abs_diff_freq_b"] for m in matched],
        "freq_joint": [m["abs_diff_freq_joint"] for m in matched],
    }
    stats = {"n_real_pairs_considered": n_real, "n_matched_pairs": len(matched),
             "coverage": float(len(matched) / max(1, n_real)),
             "n_pool_conditions": len(keys),
             "n_pool_candidate_pairs": len(table),
             "tolerances": {"marginal_abs_freq": FREQ_TOL, "joint_abs_freq": JOINT_TOL},
             "median_abs_diff": {k: (float(np.median(v)) if v else None) for k, v in resid.items()},
             "max_abs_diff": {k: (float(np.max(v)) if v else None) for k, v in resid.items()},
             "one_to_one_without_replacement": True}
    return matched, {"stats": stats, "log": log}


def score_rule_columns(cols_train: np.ndarray, resid: np.ndarray, curv: np.ndarray,
                       C: float = 0.03) -> np.ndarray:
    mu = cols_train.mean(axis=0)
    sd = cols_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    zc = (cols_train - mu) / sd
    numer = zc.T @ resid
    denom = ((zc ** 2).T @ curv) + 1.0 / C
    return numer ** 2 / denom
