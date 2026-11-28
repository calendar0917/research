"""Phase A: candidate single-structure rules and joint rules from real tree paths.

Mining rule (PROTOCOL.md §4):

* a *condition* is exactly one XGBoost split node on a structural column of X-C or X-F,

      r(G) = 1[ x_s(G) >= t ]      (right-hand / "no" child of the raw tree)

  with the complement ``1[x_s < t]`` (left-hand / "yes" child).
* single-rule candidates are conditions, deduplicated by Train activation vector;
* joint-rule candidates are pairs of conditions on **different** structural features
  that actually co-occur on one root-to-leaf path (``PROJECTED_PAIR``); they are *not*
  the original tree rule and inherit no leaf weight;
* matched random pairs never co-occur on a real path.

Everything is Train-only except the frozen design itself.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

import numpy as np

import isrd_common as K

FAMILY_NAME = {0: "size-2", 1: "size-3", 2: "size-4", 3: "cycle-5", 4: "cycle-6"}
MIN_SIDE_N = 30
MIN_SIDE_SCAF = 5
PAIR_MIN_CELL_N = 20
PAIR_MIN_CELL_SCAF = 3
MAX_SINGLE = 512
MAX_PAIR = 256


# ---------------------------------------------------------------------------
# tree traversal
# ---------------------------------------------------------------------------


def load_booster(arm: str):
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(K.BOOSTERS[arm]))
    return b


def split_nodes(booster, arm: str) -> list[dict[str, Any]]:
    """All structural split nodes with their tree id and both child directions."""
    df = booster.trees_to_dataframe()
    f = df["Feature"].astype(str)
    ok = f.str.match(r"^f\d+$").to_numpy()
    fnum = np.full(f.size, -1, dtype=np.int64)
    fnum[ok] = f[ok].str.slice(1).astype(np.int64).to_numpy()
    mask = fnum >= K.H_WIDTH
    out: list[dict[str, Any]] = []
    for row in df[mask].itertuples(index=False):
        gidx = int(str(row.Feature)[1:])
        gran = "C" if gidx < K.H_WIDTH + K.N_C else "F"
        out.append({"arm": arm, "tree": int(row.Tree), "node": int(str(row.ID).split("-")[-1]),
                    "global_feature": gidx, "granularity": gran,
                    "column": gidx - K.OFFSET[gran], "threshold": float(row.Split),
                    "gain": float(row.Gain), "cover": float(row.Cover)})
    return out


def path_conditions(booster, arm: str, struct_cols: set[tuple[str, int]],
                    cond_index: dict[int, int] | None = None) -> dict[tuple[int, int], tuple[int, ...]]:
    """Leaf paths as ordered tuples of global-feature indices.

    Returns ``(tree, leaf_node) -> tuple of global structure feature indices`` in the
    order they appear on the root-to-leaf path (only structural split nodes).
    """
    import xgboost as xgb

    d = booster.get_dump(dump_format="json")
    import json

    out: dict[tuple[int, int], tuple[int, ...]] = {}

    def walk(node: dict, tree: int, path: list[int]) -> None:
        if "leaf" in node:
            out[(tree, int(node["nodeid"]))] = tuple(path)
            return
        fidx = int(node["split"][1:])
        if fidx >= K.H_WIDTH:
            gran = "C" if fidx < K.H_WIDTH + K.N_C else "F"
            if (gran, fidx - K.OFFSET[gran]) in struct_cols and \
                    (cond_index is None or fidx in cond_index):
                path = path + [fidx]
        walk(node["children"][0], tree, path)
        walk(node["children"][1], tree, path)

    for tree, tree_json in enumerate(d):
        walk(json.loads(tree_json), tree, [])
    return out


# ---------------------------------------------------------------------------
# single rules
# ---------------------------------------------------------------------------


def mine_single(data: K.Data, max_prefilter: int = 4000) -> tuple[list[dict], dict]:
    """Extract, evaluate and filter single-structure rule candidates (Train only)."""
    nodes: list[dict] = []
    for arm in ("X-C", "X-F"):
        nodes.extend(split_nodes(load_booster(arm), arm))
    train = data.rows["train"]
    ytr = data.y[train]
    scaf_tr = data.scaffold_code[train]
    n_atoms = data.n_atoms[train]

    # aggregate identical (gran, column, threshold) conditions across arms/trees
    agg: dict[tuple[str, int, float], dict] = {}
    for nd in nodes:
        key = (nd["granularity"], nd["column"], nd["threshold"])
        rec = agg.setdefault(key, {"granularity": nd["granularity"], "column": nd["column"],
                                   "threshold": nd["threshold"],
                                   "sources": {}, "n_splits": 0, "total_gain": 0.0})
        src = rec["sources"].setdefault(nd["arm"], {"n_splits": 0, "trees": set()})
        src["n_splits"] += 1
        src["trees"].add(nd["tree"])
        rec["n_splits"] += 1
        rec["total_gain"] += nd["gain"]

    # cost guard: evaluate at most `max_prefilter` conditions, ranked by n_splits then gain
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["n_splits"], -kv[1]["total_gain"], kv[0]))
    truncated = max(0, len(ranked) - max_prefilter)
    ranked = ranked[:max_prefilter]

    # cheap support pass on Train (column x is materialised once per distinct condition)
    pre: list[dict] = []
    for key, rec in ranked:
        x = data.sub(rec["granularity"], rec["column"], train).astype(np.float32)
        act = x >= np.float32(rec["threshold"])
        n_yes = int((~act).sum())
        n_no = int(act.sum())
        if min(n_yes, n_no) == 0 or min(n_yes, n_no) < MIN_SIDE_N:
            continue
        pre.append({"rec": rec, "key": key, "x": x, "act": act})
    # one pass per unique column for scaffold / label supports
    by_col: dict[tuple[str, int], list[int]] = {}
    for i, item in enumerate(pre):
        by_col.setdefault((item["rec"]["granularity"], item["rec"]["column"]), []).append(i)
    for idxs in by_col.values():
        for i in idxs:
            act = pre[i]["act"]
            pre[i]["sup"] = {
                "train_scaffold_ge": int(np.unique(scaf_tr[act]).size),
                "train_scaffold_lt": int(np.unique(scaf_tr[~act]).size),
                "train_pos_ge": int(ytr[act].sum()),
                "train_pos_lt": int(ytr[~act].sum()),
            }
    cands: list[dict] = []
    for item in pre:
        sup = item["sup"]
        if sup["train_scaffold_ge"] < MIN_SIDE_SCAF or sup["train_scaffold_lt"] < MIN_SIDE_SCAF:
            continue
        act = item["act"]
        cands.append({**item["rec"], "key": item["key"], "n_train_ge": int(act.sum()),
                      "n_train_lt": int((~act).sum()), **sup,
                      "raw_count_equivalent": _raw_count_equiv(item["x"], act, n_atoms),
                      "activation_train": act})

    # duplicate activation vectors (Train identical) -> merge, keep highest n_splits
    cands.sort(key=lambda c: (-c["n_splits"], c["granularity"], c["column"], c["threshold"]))
    kept: list[dict] = []
    seen: dict[bytes, int] = {}
    n_dup = 0
    for c in cands:
        sig = np.packbits(c["activation_train"]).tobytes()
        if sig in seen:
            k = seen[sig]
            kept[k]["merged"].append({"granularity": c["granularity"], "column": c["column"],
                                      "threshold": c["threshold"], "n_splits": c["n_splits"]})
            kept[k]["n_splits"] += c["n_splits"]
            n_dup += 1
            continue
        seen[sig] = len(kept)
        c["merged"] = []
        cands2 = dict(c)
        kept.append(cands2)

    # ranking: distinct source trees desc, then stable id
    for c in kept:
        c["n_source_trees"] = len(c["sources"].get("X-C", {}).get("trees", ())) + \
            len(c["sources"].get("X-F", {}).get("trees", ()))
        c["stable_id"] = f"{c['granularity']}:{c['column']}:{c['threshold']:.6g}"
    kept.sort(key=lambda c: (-c["n_source_trees"], -c["n_splits"], c["stable_id"]))
    selected = kept[:MAX_SINGLE]
    for i, c in enumerate(selected):
        c["rule_id"] = f"S{i:03d}"
        c["rank"] = i
    stats = {"n_split_nodes": len(nodes), "n_distinct_conditions": len(agg),
             "n_prefilter_truncated": truncated,
             "n_after_support_filter": len(cands) + 0, "n_after_dedup": len(kept),
             "n_duplicate_merged": n_dup, "n_selected": len(selected),
             "n_dropped_by_cap": max(0, len(kept) - len(selected)),
             "min_side_molecules": MIN_SIDE_N, "min_side_scaffolds": MIN_SIDE_SCAF}
    return selected, stats


def _raw_count_equiv(x: np.ndarray, act: np.ndarray, n_atoms: np.ndarray) -> bool:
    """Is the scaled threshold ``x >= t`` logically a raw-count threshold?

    ``x = c/sqrt(n_atoms) >= t  <=>  c >= t*sqrt(n_atoms)``.  The rule is a plain
    raw-count rule iff every molecule on the ``x < t`` side has the *same* raw count
    (then the bound is ``c >= that count + 1``); it is the "presence" rule iff the
    ``x >= t`` side is exactly ``c >= 1``.
    """
    if act.all() or (~act).all():
        return False
    c = x.astype(np.float64) * np.sqrt(n_atoms)
    c_int = np.rint(c)
    if np.max(np.abs(c - c_int)) > 1e-6:
        return False
    lt_values = np.unique(c_int[~act])
    if lt_values.size == 1:
        return True
    return bool((c_int[act] == 1).all() and (c_int[~act] == 0).all())


def rule_activation(data: K.Data, rule: dict, rows: np.ndarray | None = None) -> np.ndarray:
    if rows is None:
        x = data.col(rule["granularity"], rule["column"])
    else:
        x = data.sub(rule["granularity"], rule["column"], rows)
    return (x >= np.float32(rule["threshold"])).astype(np.float64)


def rule_matrix(data: K.Data, rules: list[dict], rows: np.ndarray | None = None) -> np.ndarray:
    if not rules:
        n = data.y.size if rows is None else rows.size
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack([rule_activation(data, r, rows) for r in rules])


# ---------------------------------------------------------------------------
# joint rules
# ---------------------------------------------------------------------------


def mine_pairs(data: K.Data, singles: list[dict], max_pairs: int = MAX_PAIR) -> tuple[list[dict], dict]:
    """Real co-path pairs (PROJECTED_PAIR) with Train support filtering."""
    index_of = {(c["granularity"], c["column"], float(np.float32(c["threshold"]))): i
                for i, c in enumerate(singles)}
    # conditions keyed by (gran, col, threshold) — complement allowed later
    cond_global = {(c["granularity"], c["column"], float(np.float32(c["threshold"]))):
                   K.OFFSET[c["granularity"]] + c["column"] for c in singles}
    struct_cols = {(c["granularity"], c["column"]) for c in singles}
    g2key = {v: k for k, v in cond_global.items()}

    paths: list[list[tuple[str, int, float]]] = []
    n_paths = 0
    for arm in ("X-C", "X-F"):
        b = load_booster(arm)
        per_tree = path_conditions(b, arm, struct_cols)
        for (tree, leaf), feats in per_tree.items():
            n_paths += 1
            if len(feats) < 2:
                continue
            keys = [g2key[f] for f in feats]
            paths.append(keys)

    pair_count: dict[tuple[Any, Any], dict] = {}
    for keys in paths:
        for a, b_ in itertools.combinations(sorted(set(keys)), 2):
            if a[0] == b_[0] and a[1] == b_[1]:
                continue
            rec = pair_count.setdefault((a, b_), {"co_path": 0, "paths": []})
            rec["co_path"] += 1
            if len(rec["paths"]) < 8:
                rec["paths"].append(keys)

    train = data.rows["train"]
    ytr = data.y[train]
    scaf_tr = data.scaffold_code[train]
    n_at = data.n_atoms[train]
    cache = {i: data.sub(c["granularity"], c["column"], train).astype(np.float32)
             for i, c in enumerate(singles)}

    def cells(ai: int, bi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = cache[ai] >= np.float32(singles[ai]["threshold"])
        b = cache[bi] >= np.float32(singles[bi]["threshold"])
        return a, b, a & b

    cands: list[dict] = []
    reasons: dict[str, int] = {}
    for (ka, kb), rec in pair_count.items():
        ia, ib = index_of[ka], index_of[kb]
        if singles[ia]["granularity"] == singles[ib]["granularity"] and \
                singles[ia]["column"] == singles[ib]["column"]:
            reasons["same_feature"] = reasons.get("same_feature", 0) + 1
            continue
        a, b, ab = cells(ia, ib)
        cell_n = [int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()),
                  int((~a & ~b).sum())]
        cell_s = [int(np.unique(scaf_tr[m]).size) for m in (a & b, a & ~b, ~a & b, ~a & ~b)]
        if min(cell_n) < PAIR_MIN_CELL_N or min(cell_s) < PAIR_MIN_CELL_SCAF:
            reasons["cell_support"] = reasons.get("cell_support", 0) + 1
            continue
        if ab.sum() == 0 or (~(a | b)).sum() == 0:
            reasons["constant"] = reasons.get("constant", 0) + 1
            continue
        eq = np.array_equal(a, b) or np.array_equal(ab, a) or np.array_equal(ab, b)
        if eq:
            reasons["containment"] = reasons.get("containment", 0) + 1
            continue
        cands.append({
            "a": singles[ia]["rule_id"], "b": singles[ib]["rule_id"],
            "ia": ia, "ib": ib, "co_path": rec["co_path"],
            "freq_a": float(a[ab.size and slice(None)].mean()), "freq_b": float(b.mean()),
            "freq_joint": float(ab.mean()), "joint_train": ab,
            "cell_n": cell_n, "cell_scaf": cell_s,
            "gran_pair": "+".join(sorted([singles[ia]["granularity"], singles[ib]["granularity"]])),
            "family_pair": "+".join(sorted([data_family(data, ka), data_family(data, kb)])),
            "single_rule_sources": [ka, kb],
        })
    cands.sort(key=lambda c: (-c["co_path"], c["a"], c["b"]))
    # duplicate activation removal among pairs
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
    stats = {"n_paths": n_paths, "n_distinct_pairs_on_paths": len(pair_count),
             "n_after_filters_and_dedup": len(kept), "n_selected": len(selected),
             "filtered_reasons": reasons,
             "min_cell_molecules": PAIR_MIN_CELL_N, "min_cell_scaffolds": PAIR_MIN_CELL_SCAF}
    return selected, stats


def data_family(data: K.Data, key: tuple[str, int, float]) -> str:
    """Structural family (size-2/3/4, cycle-5/6) from the frozen registry."""
    gran, col, _t = key
    if not hasattr(data, "_fam"):
        import pandas as pd

        reg = K.registry()
        data._fam = {g: df["family"].to_numpy() for g, df in reg.items()}
    try:
        return str(data._fam[gran][col])
    except Exception:
        return "unknown"


def pair_activation(data: K.Data, pair: dict, by_id: dict[str, dict],
                    rows: np.ndarray | None = None) -> np.ndarray:
    """Boolean activation of the projected pair, resolved by single-rule id."""
    a = rule_activation(data, by_id[pair["a"]], rows)
    b = rule_activation(data, by_id[pair["b"]], rows)
    return (a * b)
