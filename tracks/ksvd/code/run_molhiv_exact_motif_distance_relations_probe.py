"""Distance-1/2 relations between exact local motif words on MolHIV.

The base representation is the full radius-0..3 exact rooted motif occurrence
vocabulary.  For each outer official-train scaffold fold, the K most frequent motif words
at a predeclared source radius are selected using outer-fit document frequency only.
Additional sparse features record unordered word pairs whose root atoms are at
shortest-path distance exactly 1 or 2.  This is an independent GNN-free,
KSVD-free model; no learned message passing is used.

As a structural negative control, motif assignments are deterministically
permuted among atoms within each molecule before relation extraction.  This
preserves each molecule's motif multiset while destroying which motifs occupy
which local positions.  Official valid/test graphs are never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import rooted_signatures, transformed


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def triangular_pair(a: int, b: int, k: int) -> int:
    if a > b:
        a, b = b, a
    return a * k - a * (a - 1) // 2 + (b - a)


def graph_pair_sets(edge_index: np.ndarray, n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    edges: set[tuple[int, int]] = set()
    for j in range(edge_index.shape[1]):
        u, v = int(edge_index[0, j]), int(edge_index[1, j])
        if u == v:
            continue
        p = (u, v) if u < v else (v, u)
        edges.add(p); nbrs[u].add(v); nbrs[v].add(u)
    dist2: set[tuple[int, int]] = set()
    for middle in range(n):
        ns = sorted(nbrs[middle])
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                u, v = ns[i], ns[j]
                p = (u, v) if u < v else (v, u)
                if p not in edges:
                    dist2.add(p)
    return sorted(edges), sorted(dist2)


def relation_matrix(
    atom_tokens: list[np.ndarray],
    pair_sets: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
    selected_global: np.ndarray,
    graph_indices: np.ndarray,
    shuffled: bool,
    seed: int,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    k = int(len(selected_global))
    n_tri = k * (k + 1) // 2
    global_to_local = {int(g): i for i, g in enumerate(selected_global.tolist())}
    indptr = [0]; indices: list[int] = []; data: list[int] = []
    eligible_atoms = 0; total_atoms = 0; represented_pairs = [0, 0]; total_pairs = [0, 0]
    for graph_index in graph_indices.tolist():
        tokens = np.asarray(atom_tokens[int(graph_index)], dtype=np.int64)
        if shuffled and len(tokens) > 1:
            # Per-graph seed makes the control independent of row order.
            rng = np.random.default_rng(np.uint64(seed) ^ np.uint64((int(graph_index) + 1) * 0x9E3779B1))
            tokens = tokens[rng.permutation(len(tokens))]
        local = np.asarray([global_to_local.get(int(t), -1) for t in tokens], dtype=np.int32)
        total_atoms += len(local); eligible_atoms += int(np.sum(local >= 0))
        counts: Counter[int] = Counter()
        d1, d2 = pair_sets[int(graph_index)]
        for distance_index, pairs in enumerate((d1, d2)):
            total_pairs[distance_index] += len(pairs)
            for u, v in pairs:
                a, b = int(local[u]), int(local[v])
                if a < 0 or b < 0:
                    continue
                represented_pairs[distance_index] += 1
                counts[distance_index * n_tri + triangular_pair(a, b, k)] += 1
        for col, count in sorted(counts.items()):
            indices.append(col); data.append(count)
        indptr.append(len(indices))
    mat = sparse.csr_matrix((np.asarray(data, dtype=np.int16), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)), shape=(len(graph_indices), 2 * n_tri))
    return mat, {
        "selected_atom_coverage": float(eligible_atoms / max(total_atoms, 1)),
        "distance1_pair_coverage": float(represented_pairs[0] / max(total_pairs[0], 1)),
        "distance2_pair_coverage": float(represented_pairs[1] / max(total_pairs[1], 1)),
        "raw_relation_slots": int(2 * n_tri),
    }


def fit_eval(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_held: sparse.csr_matrix, y_held: np.ndarray, c: float) -> tuple[dict[str, Any], np.ndarray]:
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
    model.fit(x_fit, y_fit)
    pf = model.predict_proba(x_fit)[:, 1]; ph = model.predict_proba(x_held)[:, 1]
    return {
        "trainable_parameters": int(x_fit.shape[1] + 1),
        "fit_auc": float(roc_auc_score(y_fit, pf)),
        "heldout_auc": float(roc_auc_score(y_held, ph)),
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
    }, ph


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = np.asarray([r["heldout_auc"] for r in rows]); f = np.asarray([r["fit_auc"] for r in rows])
    return {"fold_auc": a.tolist(), "mean_auc": float(a.mean()), "sample_std_auc": float(a.std(ddof=1)), "min_fold_auc": float(a.min()), "mean_fit_auc": float(f.mean()), "mean_parameters": float(np.mean([r["trainable_parameters"] for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--relation-min-df", type=int, default=5)
    ap.add_argument("--relation-radius", type=int, choices=(0, 1, 2, 3), default=2)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/exact_motif_distance_relations_fulltrain_probe_20260729.json")
    args = ap.parse_args(); t0 = time.time()
    repo = Path(__file__).resolve().parents[3]

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split()
    train_graph_indices = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    y = np.asarray(dataset.labels).reshape(-1).astype(np.int64)[train_graph_indices]
    position = np.full(len(dataset), -1, dtype=np.int64); position[train_graph_indices] = np.arange(len(train_graph_indices))
    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train_graph_indices):
            raise ValueError("official train mismatch")
        outer = []
        for fold in range(3):
            fit_graph = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
            held_graph = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
            outer.append((position[fit_graph], position[held_graph], fit_graph, held_graph))
    if np.any(position[forbidden] >= 0):
        raise AssertionError("forbidden split mapped into train rows")

    token_to_col: dict[tuple[int, bytes], int] = {}
    token_radius: list[int] = []
    row_maps: list[Counter[int]] = []
    # Indexed by original dataset graph id; forbidden entries remain None and
    # are never touched.
    atom_relation: list[np.ndarray | None] = [None] * len(dataset)
    pair_sets: list[tuple[list[tuple[int, int]], list[tuple[int, int]]] | None] = [None] * len(dataset)
    for row, graph_index in enumerate(train_graph_indices.tolist()):
        g, _ = dataset[int(graph_index)]
        node_feat = np.asarray(g["node_feat"], dtype=np.int64)
        edge_index = np.asarray(g["edge_index"], dtype=np.int64)
        edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
        signatures = rooted_signatures(node_feat, edge_index, edge_feat, 3)
        counts: Counter[int] = Counter(); relation_cols_for_atoms: list[int] = []
        for radius, values in enumerate(signatures):
            for value in values:
                key = (radius, value)
                col = token_to_col.get(key)
                if col is None:
                    col = len(token_to_col); token_to_col[key] = col; token_radius.append(radius)
                counts[col] += 1
                if radius == args.relation_radius:
                    relation_cols_for_atoms.append(col)
        atom_relation[int(graph_index)] = np.asarray(relation_cols_for_atoms, dtype=np.int32)
        pair_sets[int(graph_index)] = graph_pair_sets(edge_index, len(node_feat))
        row_maps.append(counts)
        if (row + 1) % 5000 == 0:
            print(f"extracted {row + 1}/{len(train_graph_indices)} graphs vocab={len(token_to_col)}", flush=True)
    # Type narrowing after the isolation-safe extraction loop.
    atom_relation_ready: list[np.ndarray] = [np.empty(0, dtype=np.int32) if x is None else x for x in atom_relation]
    pair_sets_ready: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = [([], []) if x is None else x for x in pair_sets]
    indptr = [0]; indices: list[int] = []; data: list[int] = []
    for counts in row_maps:
        for col, count in sorted(counts.items()):
            indices.append(col); data.append(count)
        indptr.append(len(indices))
    x_all = sparse.csr_matrix((np.asarray(data, dtype=np.int16), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)), shape=(len(train_graph_indices), len(token_to_col)))
    radius_arr = np.asarray(token_radius, dtype=np.int8); del row_maps, atom_relation, pair_sets

    out: dict[str, Any] = {
        "protocol_id": "molhiv-exact-motif-distance-relations-v1", "date": "2026-07-29",
        "scope": f"exact official-train only; radius-{args.relation_radius} word pairs at atom distance 1/2",
        "official_valid_evaluations": 0, "official_test_evaluations": 0,
        "isolation": "dataset items were accessed only for official-train graph indices",
        "config": vars(args), "n_train": int(len(y)), "n_positive": int(y.sum()),
        "n_unique_motifs": int(x_all.shape[1]), "folds": [], "aggregate": {},
    }
    all_results: dict[str, list[dict[str, Any]]] = {}
    for fold, (fit, held, fit_graph, held_graph) in enumerate(outer):
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)}", flush=True)
        df = np.asarray((x_all[fit] > 0).sum(axis=0)).ravel()
        base_cols = np.flatnonzero(df >= args.min_df)
        xb_fit = transformed(x_all[fit][:, base_cols], "binary")
        xb_held = transformed(x_all[held][:, base_cols], "binary")
        fold_out: dict[str, Any] = {"fold": fold, "n_base_features": int(len(base_cols)), "relations": {}, "results": {}}
        for c in (0.01, 0.03):
            key = f"baseline_binary_c{c:g}"
            row, ph = fit_eval(xb_fit, y[fit], xb_held, y[held], c)
            row.update({"C": c, "representation": "motif occurrence only", "heldout_probabilities": ph.tolist()})
            fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)

        source_cols = np.flatnonzero(radius_arr == args.relation_radius)
        # Frequency order is outer-fit only, with global column id as a fixed tie-break.
        order = np.lexsort((source_cols, -df[source_cols]))
        for k in (32, 64, 128):
            selected = source_cols[order[: min(k, len(source_cols))]]
            real_all, coverage = relation_matrix(atom_relation_ready, pair_sets_ready, selected, train_graph_indices, False, 20260729 + fold)
            shuffled_all, shuffled_coverage = relation_matrix(atom_relation_ready, pair_sets_ready, selected, train_graph_indices, True, 20261729 + fold)
            rel_df = np.asarray((real_all[fit] > 0).sum(axis=0)).ravel()
            rel_cols = np.flatnonzero(rel_df >= args.relation_min_df)
            shuffled_df = np.asarray((shuffled_all[fit] > 0).sum(axis=0)).ravel()
            shuffled_cols = np.flatnonzero(shuffled_df >= args.relation_min_df)
            xr_fit = transformed(real_all[fit][:, rel_cols], "binary"); xr_held = transformed(real_all[held][:, rel_cols], "binary")
            xs_fit = transformed(shuffled_all[fit][:, shuffled_cols], "binary"); xs_held = transformed(shuffled_all[held][:, shuffled_cols], "binary")
            concat_fit = sparse.hstack([xb_fit, xr_fit], format="csr"); concat_held = sparse.hstack([xb_held, xr_held], format="csr")
            shuffled_concat_fit = sparse.hstack([xb_fit, xs_fit], format="csr"); shuffled_concat_held = sparse.hstack([xb_held, xs_held], format="csr")
            fold_out["relations"][f"k{k}"] = {
                **coverage, "relation_source_radius": int(args.relation_radius), "selected_source_global_columns": selected.astype(int).tolist(),
                "n_observed_relation_features": int(len(rel_cols)),
                "n_shuffled_observed_relation_features": int(len(shuffled_cols)),
                "shuffled_coverage": shuffled_coverage,
            }
            for c in (0.01, 0.03):
                key = f"concat_distance12_k{k}_binary_c{c:g}"
                row, ph = fit_eval(concat_fit, y[fit], concat_held, y[held], c)
                row.update({"C": c, "K": k, "n_base_features": int(len(base_cols)), "n_relation_features": int(len(rel_cols)), "heldout_probabilities": ph.tolist()})
                fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
            key = f"relation_only_distance12_k{k}_binary_c0.03"
            row, ph = fit_eval(xr_fit, y[fit], xr_held, y[held], 0.03)
            row.update({"C": 0.03, "K": k, "n_relation_features": int(len(rel_cols)), "heldout_probabilities": ph.tolist()})
            fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
            key = f"control_shuffled_concat_distance12_k{k}_binary_c0.03"
            row, ph = fit_eval(shuffled_concat_fit, y[fit], shuffled_concat_held, y[held], 0.03)
            row.update({"C": 0.03, "K": k, "n_relation_features": int(len(shuffled_cols)), "heldout_probabilities": ph.tolist()})
            fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
        out["folds"].append(fold_out)

    out["aggregate"] = {key: summarize(rows) for key, rows in all_results.items() if len(rows) == 3}
    out["ranking"] = sorted(({"candidate": key, **value} for key, value in out["aggregate"].items()), key=lambda r: (r["mean_auc"], r["min_fold_auc"]), reverse=True)
    out["baseline_reference"] = {"candidate": "baseline_binary_c0.03", **out["aggregate"]["baseline_binary_c0.03"]}
    out["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": args.output, "top20": out["ranking"][:20], "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
