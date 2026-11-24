"""Stage A/B structural support, vocabulary and context split.

This module never reads labels.  It writes the frozen universe used by all
later stages.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import sfec_common as C


def load_cells(prefix: Path) -> dict[str, np.ndarray]:
    z = np.load(prefix.with_suffix(".npz"))
    return {k: z[k] for k in z.files}


def load_C_table(prefix: Path, columns=None) -> dict[str, np.ndarray]:
    tab = pq.read_table(prefix.with_suffix(".C.parquet"), columns=columns)
    return {name: tab[name].to_numpy() for name in tab.column_names}


def load_S_table(prefix: Path) -> dict[str, np.ndarray]:
    tab = pq.read_table(prefix.with_suffix(".S.parquet"))
    return {name: tab[name].to_numpy(zero_copy_only=False) for name in tab.column_names}


def group_starts(cids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(cids, kind="stable")
    cs = cids[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]) if len(cs) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(cs)] if len(cs) else np.zeros(0, np.int64)
    return order, starts, ends


def unique_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.stack([a.astype(np.int64), b.astype(np.int64)], axis=1), axis=0)


def top_vocabulary(
    cells: dict[str, np.ndarray],
    s_table: dict[str, np.ndarray],
    train_mask: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    n_s = len(s_table["S_id"])
    bt_values = np.unique(cells["bond_type"][train_mask])
    primary = np.zeros(n_s, dtype=bool)
    rank = np.full(n_s, -1, dtype=np.int16)
    details = []
    for bt in bt_values.tolist():
        m = train_mask & (cells["bond_type"] == bt)
        if not np.any(m):
            continue
        pairs = unique_pairs(cells["S_id"][m], cells["mol_id"][m])
        df = np.bincount(pairs[:, 0].astype(np.int64), minlength=n_s)
        observed = np.flatnonzero(df > 0)
        order = sorted(observed.tolist(), key=lambda i: (-int(df[i]), s_table["S_key"][i]))
        chosen = order[:top_k]
        chosen_sorted = sorted(chosen, key=lambda i: s_table["S_key"][i])
        for r, sid in enumerate(chosen_sorted):
            primary[sid] = True
            rank[sid] = r
        details.append(
            {
                "bond_type": int(bt),
                "n_observed_s": int(len(observed)),
                "n_chosen": int(len(chosen)),
                "chosen_s_ids": [int(x) for x in chosen_sorted],
                "chosen_keys": [str(s_table["S_key"][x]) for x in chosen_sorted],
                "df": [int(df[x]) for x in chosen_sorted],
                "frequency_order": [int(x) for x in chosen],
            }
        )
    return {"primary": primary, "rank": rank, "details": details}


def bipartite_stats(
    c_ids: np.ndarray, s_ids: np.ndarray, n_c: int, n_s: int
) -> dict[str, Any]:
    pairs = unique_pairs(c_ids, s_ids)
    c_obs = np.unique(pairs[:, 0])
    s_obs = np.unique(pairs[:, 1])
    c_map = {int(c): i for i, c in enumerate(c_obs.tolist())}
    s_map = {int(s): i for i, s in enumerate(s_obs.tolist())}
    rows = np.asarray([c_map[int(c)] for c in pairs[:, 0].tolist()], dtype=np.int64)
    cols = np.asarray([len(c_obs) + s_map[int(s)] for s in pairs[:, 1].tolist()], dtype=np.int64)
    n = len(c_obs) + len(s_obs)
    m = len(pairs)
    if n == 0:
        return {"n_vertices": 0, "n_edges": 0, "n_components": 0,
                "largest_component": 0, "two_core": 0, "cycle_rank": 0,
                "n_c_observed": 0, "n_s_observed": 0}
    A = sparse.coo_matrix((np.ones(2 * m, dtype=np.int8), (np.r_[rows, cols], np.r_[cols, rows])),
                          shape=(n, n)).tocsr()
    k, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    # 2-core by iterative degree-1 peeling
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int32)
    alive = np.ones(n, dtype=bool)
    q = deque(np.flatnonzero(deg < 2).tolist())
    while q:
        v = q.popleft()
        if not alive[v]:
            continue
        alive[v] = False
        for nb in A.indices[A.indptr[v]:A.indptr[v + 1]]:
            if alive[nb]:
                deg[nb] -= 1
                if deg[nb] < 2:
                    q.append(int(nb))
    return {
        "n_vertices": int(n),
        "n_edges": int(m),
        "n_components": int(k),
        "largest_component": int(sizes.max()) if len(sizes) else 0,
        "two_core": int(alive.sum()),
        "cycle_rank": int(m - n + k),
        "n_c_observed": int(len(c_obs)),
        "n_s_observed": int(len(s_obs)),
    }


def rectangle_stats(
    cells: dict[str, np.ndarray],
    train_mask: np.ndarray,
    primary: np.ndarray,
    rank: np.ndarray,
    *,
    full_pair_guard: int = 20_000_000,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # aggregate counts over all C, full and primary
    for name, mask in (("full", train_mask), ("primary", train_mask & primary[cells["S_id"]])):
        pairs = unique_pairs(cells["C_id"][mask], cells["S_id"][mask])
        if len(pairs) == 0:
            out[name] = {"n_cells": 0, "rectangles": 0, "pairs": {}}
            continue
        orders, starts, ends = group_starts(pairs[:, 0])
        k = ends - starts
        rect = int(np.sum(k * (k - 1) // 2))
        out[name] = {
            "n_cells": int(len(pairs)),
            "n_contexts": int(len(k)),
            "rectangles": rect,
            "max_s_per_context": int(k.max()) if len(k) else 0,
            "mean_s_per_context": float(k.mean()) if len(k) else 0.0,
            "context_degree_hist": _hist(k),
        }
    # primary pair-level counts per bond type
    pair_tables = []
    for bt in np.unique(cells["bond_type"][train_mask]).tolist():
        m = train_mask & (cells["bond_type"] == bt) & primary[cells["S_id"]]
        pairs = unique_pairs(cells["C_id"][m], cells["S_id"][m])
        if len(pairs) == 0:
            continue
        kdim = int(rank[pairs[:, 1]].max()) + 1
        mat = np.zeros((kdim, kdim), dtype=np.int32)
        order, starts, ends = group_starts(pairs[:, 0])
        for a, b in zip(starts.tolist(), ends.tolist()):
            r = rank[pairs[order[a:b], 1]].astype(np.int64)
            if len(r) >= 2:
                ii, jj = np.triu_indices(len(r), 1)
                np.add.at(mat, (r[ii], r[jj]), 1)
        mat = mat + mat.T
        thresholds = {}
        for t in (1, 2, 5, 10, 20):
            thresholds[str(t)] = int(np.count_nonzero(np.triu(mat, 1) >= t))
        pair_tables.append(
            {
                "bond_type": int(bt),
                "kdim": int(kdim),
                "total_rectangles": int(np.triu(mat, 1).sum()),
                "pairs": thresholds,
                "matrix": mat,
            }
        )
    out["primary_pair_tables"] = pair_tables
    # full pair-level counts if the number of pair occurrences is modest
    full_pairs: dict[tuple[int, int], int] = {}
    pairs = unique_pairs(cells["C_id"][train_mask], cells["S_id"][train_mask])
    if len(pairs):
        order, starts, ends = group_starts(pairs[:, 0])
        total = int(np.sum((ends - starts) * (ends - starts - 1) // 2))
        out["full"]["pair_occurrences"] = total
        if total <= full_pair_guard:
            from collections import Counter
            counter: Counter = Counter()
            for a, b in zip(starts.tolist(), ends.tolist()):
                ss = pairs[order[a:b], 1].tolist()
                for i in range(len(ss)):
                    for j in range(i + 1, len(ss)):
                        key = (ss[i], ss[j]) if ss[i] < ss[j] else (ss[j], ss[i])
                        counter[key] += 1
            out["full"]["pairs_by_threshold"] = {
                str(t): int(sum(1 for v in counter.values() if v >= t))
                for t in (1, 2, 5, 10, 20)
            }
            out["full"]["n_pairs"] = int(len(counter))
            full_pairs = dict(counter)
        else:
            out["full"]["pairs_by_threshold"] = "SKIPPED_TOO_LARGE"
            out["full"]["n_pairs"] = None
    out["full_pair_counts"] = full_pairs
    return out


def _hist(values: np.ndarray) -> dict[str, int]:
    if len(values) == 0:
        return {}
    u, c = np.unique(values, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(u, c) if int(a) <= 20}


def context_split(
    cells: dict[str, np.ndarray],
    c_table: dict[str, np.ndarray],
    train_mask: np.ndarray,
    *,
    discovery_percent: int = 80,
    salt: int = 0,
) -> dict[str, Any]:
    n_c = len(c_table["C_id"])
    n_mol = int(cells["mol_id"].max()) + 1 if len(cells) else 0
    hash_mod = np.asarray(c_table["hash_mod"], dtype=np.int16)
    hash64 = np.asarray(c_table["hash64"], dtype=np.uint64)
    train_c = np.unique(cells["C_id"][train_mask])
    eval_c = train_c[hash_mod[train_c] >= discovery_percent]
    disc_c = train_c[hash_mod[train_c] < discovery_percent]
    eval_c_set = np.zeros(n_c, dtype=bool)
    eval_c_set[eval_c] = True
    disc_c_set = np.zeros(n_c, dtype=bool)
    disc_c_set[disc_c] = True
    eval_mol = np.zeros(n_mol, dtype=bool)
    eval_mol[cells["mol_id"][train_mask & eval_c_set[cells["C_id"]]]] = True
    disc_mask = train_mask & disc_c_set[cells["C_id"]] & ~eval_mol[cells["mol_id"]]
    eval_mask = train_mask & eval_c_set[cells["C_id"]]
    # greedy molecule-disjoint primary evaluation contexts, hash order
    eval_idx = np.flatnonzero(eval_mask)
    order, starts, ends = group_starts(cells["C_id"][eval_idx])
    groups = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        idx = eval_idx[order[a:b]]
        cid = int(cells["C_id"][idx[0]])
        mols = np.unique(cells["mol_id"][idx])
        groups.append((int(hash64[cid]), cid, mols, idx))
    groups.sort(key=lambda x: x[0])
    used = np.zeros(n_mol, dtype=bool)
    selected_c = []
    selected_cells = []
    for _, cid, mols, idx in groups:
        if np.any(used[mols]):
            continue
        selected_c.append(int(cid))
        selected_cells.append(idx)
        used[mols] = True
    primary_eval_c = np.zeros(n_c, dtype=bool)
    primary_eval_c[np.asarray(selected_c, dtype=np.int64)] = True
    primary_eval_cells = np.zeros(len(cells["C_id"]), dtype=bool)
    if selected_cells:
        primary_eval_cells[np.concatenate(selected_cells)] = True
    return {
        "n_c": n_c,
        "n_molecules": n_mol,
        "n_train_c": int(len(train_c)),
        "n_discovery_c": int(len(disc_c)),
        "n_evaluation_c": int(len(eval_c)),
        "n_evaluation_molecules": int(eval_mol.sum()),
        "n_discovery_cells": int(disc_mask.sum()),
        "n_evaluation_cells": int(eval_mask.sum()),
        "n_primary_evaluation_c": int(len(selected_c)),
        "n_primary_evaluation_cells": int(primary_eval_cells.sum()),
        "discovery_mask": disc_mask,
        "evaluation_mask": eval_mask,
        "evaluation_c": eval_c_set,
        "discovery_c": disc_c_set,
        "evaluation_molecules": eval_mol,
        "primary_evaluation_c": primary_eval_c,
        "primary_evaluation_cells": primary_eval_cells,
        "selected_evaluation_c": selected_c,
    }


def pair_support_matrices(
    cells: dict[str, np.ndarray],
    rank: np.ndarray,
    mask: np.ndarray,
    primary: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    for bt in np.unique(cells["bond_type"]).tolist():
        m = mask & (cells["bond_type"] == bt) & primary[cells["S_id"]]
        pairs = unique_pairs(cells["C_id"][m], cells["S_id"][m])
        if len(pairs) == 0:
            out.append({"bond_type": int(bt), "matrix": np.zeros((0, 0), np.int32), "kdim": 0})
            continue
        kdim = int(rank[pairs[:, 1]].max()) + 1
        mat = np.zeros((kdim, kdim), dtype=np.int32)
        order, starts, ends = group_starts(pairs[:, 0])
        for a, b in zip(starts.tolist(), ends.tolist()):
            r = rank[pairs[order[a:b], 1]].astype(np.int64)
            if len(r) >= 2:
                ii, jj = np.triu_indices(len(r), 1)
                np.add.at(mat, (r[ii], r[jj]), 1)
        mat = mat + mat.T
        out.append({"bond_type": int(bt), "matrix": mat, "kdim": int(kdim)})
    return out


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args(argv)
    prefix = Path(args.prefix)
    out_prefix = Path(args.out) if args.out else prefix
    cfg = C.load_config()
    cells = load_cells(prefix)
    s_table = load_S_table(prefix)
    c_table = load_C_table(prefix, columns=["C_id", "q_id", "nC", "scaffold_id", "hash_mod", "hash64"])
    train_mask = cells["split_id"] == 0
    vocab = top_vocabulary(cells, s_table, train_mask, args.top_k)
    primary = vocab["primary"]
    rank = vocab["rank"]
    # coverage before/after
    train_cells = int(train_mask.sum())
    train_mols = int(np.unique(cells["mol_id"][train_mask]).size)
    prim_cells = int((train_mask & primary[cells["S_id"]]).sum())
    prim_mols = int(np.unique(cells["mol_id"][train_mask & primary[cells["S_id"]]]).size)
    train_c = np.unique(cells["C_id"][train_mask])
    prim_c = np.unique(cells["C_id"][train_mask & primary[cells["S_id"]]])
    coverage = {
        "train_cells": train_cells,
        "train_molecules": train_mols,
        "train_contexts": int(len(train_c)),
        "primary_cells": prim_cells,
        "primary_molecules": prim_mols,
        "primary_contexts": int(len(prim_c)),
        "cell_fraction": prim_cells / max(train_cells, 1),
        "molecule_fraction": prim_mols / max(train_mols, 1),
        "context_fraction": len(prim_c) / max(len(train_c), 1),
    }
    # degree distributions
    pairs_all = unique_pairs(cells["C_id"][train_mask], cells["S_id"][train_mask])
    pairs_prim = unique_pairs(
        cells["C_id"][train_mask & primary[cells["S_id"]]],
        cells["S_id"][train_mask & primary[cells["S_id"]]],
    )
    degree_all = np.bincount(pairs_all[:, 0].astype(np.int64), minlength=len(c_table["C_id"])) if len(pairs_all) else np.zeros(len(c_table["C_id"]), np.int64)
    degree_prim = np.bincount(pairs_prim[:, 0].astype(np.int64), minlength=len(c_table["C_id"])) if len(pairs_prim) else np.zeros(len(c_table["C_id"]), np.int64)
    degree_stats = {
        "full": _degree_summary(degree_all[train_c]),
        "primary": _degree_summary(degree_prim[train_c]),
        "full_contexts_with_one": int(np.sum(degree_all[train_c] == 1)),
        "primary_contexts_with_one": int(np.sum(degree_prim[train_c] == 1)),
    }
    bipart_full = bipartite_stats(cells["C_id"][train_mask], cells["S_id"][train_mask], len(c_table["C_id"]), len(s_table["S_id"]))
    bipart_prim = bipartite_stats(
        cells["C_id"][train_mask & primary[cells["S_id"]]],
        cells["S_id"][train_mask & primary[cells["S_id"]]],
        len(c_table["C_id"]), len(s_table["S_id"]),
    )
    rect = rectangle_stats(cells, train_mask, primary, rank)
    split = context_split(cells, c_table, train_mask, discovery_percent=cfg["context_split"]["discovery_percent"], salt=cfg["context_split"]["salt"])
    disc_pairs = pair_support_matrices(cells, rank, split["discovery_mask"], primary)
    eval_pairs = pair_support_matrices(cells, rank, split["primary_evaluation_cells"], primary)
    pair_gate = []
    n_support_pairs = 0
    n_eval_obs = 0
    for d, e in zip(disc_pairs, eval_pairs):
        if d["kdim"] == 0 or e["kdim"] == 0:
            continue
        k = min(d["kdim"], e["kdim"])
        dd = d["matrix"][:k, :k]
        ee = e["matrix"][:k, :k]
        ok = (dd >= 10) & (ee >= 5) & np.triu(np.ones_like(dd, dtype=bool), 1)
        n_support_pairs += int(ok.sum())
        n_eval_obs += int(np.where(ok, ee, 0).sum())
        pair_gate.append({"bond_type": int(d["bond_type"]), "n_supported_pairs": int(ok.sum()), "evaluation_pair_observations": int(np.where(ok, ee, 0).sum())})
    gates = {
        "p1_supported_pairs": int(n_support_pairs),
        "p1_evaluation_contexts_disjoint": int(split["n_primary_evaluation_c"]),
        "p1_evaluation_pair_observations": int(n_eval_obs),
        "p1_pass": bool(
            n_support_pairs >= cfg["support_gates"]["p1_min_pairs"]
            and split["n_primary_evaluation_c"] >= cfg["support_gates"]["p1_min_evaluation_contexts_disjoint"]
            and n_eval_obs >= cfg["support_gates"]["p1_min_evaluation_observations"]
        ),
    }
    support = {
        "dataset_prefix": str(prefix),
        "n_cells_total": int(len(cells["C_id"])),
        "n_cells_train": int(train_mask.sum()),
        "n_cells_other_splits": int((~train_mask).sum()),
        "n_molecules_total": int(cells["mol_id"].max() + 1),
        "n_C": int(len(c_table["C_id"])),
        "n_S": int(len(s_table["S_id"])),
        "n_q": int(len(np.unique(c_table["q_id"]))),
        "coverage": coverage,
        "vocabulary": {
            "top_k": args.top_k,
            "per_bond_type": [
                {k: v for k, v in d.items() if k != "chosen_keys"} | {"n_chosen_keys": len(d["chosen_keys"])}
                for d in vocab["details"]
            ],
        },
        "degree": degree_stats,
        "bipartite_full": bipart_full,
        "bipartite_primary": bipart_prim,
        "rectangles": {k: v for k, v in rect.items() if k != "full_pair_counts"},
        "context_split": {
            **{k: v for k, v in split.items() if not isinstance(v, (np.ndarray, list))},
            "selected_evaluation_c_count": len(split["selected_evaluation_c"]),
        },
        "support_gates": gates,
        "pair_gate_by_bond_type": pair_gate,
    }
    C.write_json(Path(str(out_prefix) + ".support.json"), support)
    # arrays for downstream
    np.savez(
        Path(str(out_prefix) + ".universe.npz"),
        primary=primary,
        rank=rank,
        discovery_mask=split["discovery_mask"],
        evaluation_mask=split["evaluation_mask"],
        evaluation_c=split["evaluation_c"],
        discovery_c=split["discovery_c"],
        evaluation_molecules=split["evaluation_molecules"],
        primary_evaluation_c=split["primary_evaluation_c"],
        primary_evaluation_cells=split["primary_evaluation_cells"],
        train_mask=train_mask,
        hash_mod=c_table["hash_mod"],
        hash64=c_table["hash64"],
        q_id=c_table["q_id"],
        nC=c_table["nC"],
        scaffold_id=c_table["scaffold_id"],
    )
    # rectangle csv
    rows = []
    for name in ("full", "primary"):
        d = rect[name]
        rows.append({
            "universe": name,
            "bond_type": "all",
            "n_cells": d.get("n_cells", 0),
            "n_contexts": d.get("n_contexts", 0),
            "rectangles": d.get("rectangles", 0),
            "max_s_per_context": d.get("max_s_per_context", 0),
            "mean_s_per_context": d.get("mean_s_per_context", 0),
            "pairs_ge_1": (d.get("pairs_by_threshold") or {}).get("1", ""),
            "pairs_ge_2": (d.get("pairs_by_threshold") or {}).get("2", ""),
            "pairs_ge_5": (d.get("pairs_by_threshold") or {}).get("5", ""),
            "pairs_ge_10": (d.get("pairs_by_threshold") or {}).get("10", ""),
            "pairs_ge_20": (d.get("pairs_by_threshold") or {}).get("20", ""),
        })
    for table in rect["primary_pair_tables"]:
        rows.append({
            "universe": "primary",
            "bond_type": table["bond_type"],
            "n_cells": "",
            "n_contexts": "",
            "rectangles": table["total_rectangles"],
            "max_s_per_context": "",
            "mean_s_per_context": "",
            "pairs_ge_1": table["pairs"]["1"],
            "pairs_ge_2": table["pairs"]["2"],
            "pairs_ge_5": table["pairs"]["5"],
            "pairs_ge_10": table["pairs"]["10"],
            "pairs_ge_20": table["pairs"]["20"],
        })
    C.write_csv(Path(str(out_prefix) + ".rectangles.csv"), rows)
    print(json.dumps({k: support[k] for k in ("coverage", "bipartite_full", "bipartite_primary", "support_gates", "context_split")}, indent=2))
    return 0


def _degree_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
        "hist": _hist(values),
    }


if __name__ == "__main__":
    raise SystemExit(main())