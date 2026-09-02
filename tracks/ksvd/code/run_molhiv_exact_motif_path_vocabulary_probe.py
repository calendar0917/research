"""Exact local motifs plus transferable typed simple-path vocabulary.

Rooted radius-0..3 motif occurrences are augmented with canonical simple paths
of one to four bonds.  Every path word contains all OGB atom features along the
path and all OGB bond features between them.  The path representation is
coarser than a full rooted radius-2/3 neighbourhood, so it can transfer a useful
chemical sequence even when the surrounding scaffold changes.

All vocabularies and document-frequency filters are fitted inside each outer
official-train scaffold fold.  A node-assignment-shuffled path vocabulary is a
negative control: atom feature rows are permuted within each molecule while the
same topology and bond features are retained.  Official valid/test graphs are
never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes, rooted_signatures, transformed


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def sparse_from_counters(rows: list[Counter[int]], n_cols: int) -> sparse.csr_matrix:
    indptr = [0]; indices: list[int] = []; data: list[int] = []
    for counts in rows:
        for col, count in sorted(counts.items()):
            indices.append(col); data.append(count)
        indptr.append(len(indices))
    return sparse.csr_matrix((np.asarray(data, dtype=np.int16), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)), shape=(len(rows), n_cols))


def enumerate_simple_paths(edge_index: np.ndarray, n: int, max_length: int) -> tuple[list[list[tuple[int, ...]]], dict[tuple[int, int], int]]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    edge_pos: dict[tuple[int, int], int] = {}
    for j in range(edge_index.shape[1]):
        u, v = int(edge_index[0, j]), int(edge_index[1, j])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in edge_pos:
            edge_pos[key] = j
        nbrs[u].add(v); nbrs[v].add(u)
    paths: list[set[tuple[int, ...]]] = [set() for _ in range(max_length + 1)]
    for start in range(n):
        stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [(start, (start,), frozenset((start,)))]
        while stack:
            u, nodes, used = stack.pop()
            length = len(nodes) - 1
            if length:
                rev = nodes[::-1]
                paths[length].add(nodes if nodes <= rev else rev)
            if length == max_length:
                continue
            for v in nbrs[u]:
                if v not in used:
                    stack.append((v, nodes + (v,), used | frozenset((v,))))
    return [sorted(x) for x in paths], edge_pos


def path_signature(nodes: tuple[int, ...], atom_labels: list[bytes], edge_labels: dict[tuple[int, int], bytes]) -> bytes:
    forward = bytearray(b"P" + bytes([len(nodes) - 1]))
    reverse = bytearray(b"P" + bytes([len(nodes) - 1]))
    for i, u in enumerate(nodes):
        forward += atom_labels[u]
        if i + 1 < len(nodes):
            v = nodes[i + 1]; forward += edge_labels[(u, v) if u < v else (v, u)]
    rev_nodes = nodes[::-1]
    for i, u in enumerate(rev_nodes):
        reverse += atom_labels[u]
        if i + 1 < len(rev_nodes):
            v = rev_nodes[i + 1]; reverse += edge_labels[(u, v) if u < v else (v, u)]
    return digest(bytes(forward if forward <= reverse else reverse))


def fit_eval(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_held: sparse.csr_matrix, y_held: np.ndarray, c: float) -> tuple[dict[str, Any], np.ndarray]:
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
    model.fit(x_fit, y_fit)
    pf = model.predict_proba(x_fit)[:, 1]; ph = model.predict_proba(x_held)[:, 1]
    return {
        "trainable_parameters": int(x_fit.shape[1] + 1), "fit_auc": float(roc_auc_score(y_fit, pf)),
        "heldout_auc": float(roc_auc_score(y_held, ph)), "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
    }, ph


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = np.asarray([r["heldout_auc"] for r in rows]); f = np.asarray([r["fit_auc"] for r in rows])
    return {"fold_auc": a.tolist(), "mean_auc": float(a.mean()), "sample_std_auc": float(a.std(ddof=1)), "min_fold_auc": float(a.min()), "mean_fit_auc": float(f.mean()), "mean_parameters": float(np.mean([r["trainable_parameters"] for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-path-length", type=int, default=4)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/exact_motif_path_vocabulary_fulltrain_probe_20260729.json")
    args = ap.parse_args(); t0 = time.time()
    if args.max_path_length < 1 or args.max_path_length > 6:
        raise ValueError("max path length must be in 1..6")
    repo = Path(__file__).resolve().parents[3]

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split(); train_graph_indices = np.asarray(split["train"], dtype=np.int64)
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

    motif_to_col: dict[tuple[int, bytes], int] = {}; motif_rows: list[Counter[int]] = []
    path_to_col: dict[tuple[int, bytes], int] = {}; path_length: list[int] = []; path_rows: list[Counter[int]] = []
    shuffled_to_col: dict[tuple[int, bytes], int] = {}; shuffled_length: list[int] = []; shuffled_rows: list[Counter[int]] = []
    path_occurrences = np.zeros(args.max_path_length + 1, dtype=np.int64)
    for row, graph_index in enumerate(train_graph_indices.tolist()):
        g, _ = dataset[int(graph_index)]
        node_feat = np.asarray(g["node_feat"], dtype=np.int64); edge_index = np.asarray(g["edge_index"], dtype=np.int64); edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
        motifs = rooted_signatures(node_feat, edge_index, edge_feat, 3)
        motif_counts: Counter[int] = Counter()
        for radius, values in enumerate(motifs):
            for value in values:
                key = (radius, value); col = motif_to_col.get(key)
                if col is None: col = len(motif_to_col); motif_to_col[key] = col
                motif_counts[col] += 1
        motif_rows.append(motif_counts)

        paths, edge_pos = enumerate_simple_paths(edge_index, len(node_feat), args.max_path_length)
        atom_labels = [digest(b"A" + feature_bytes(node_feat[u])) for u in range(len(node_feat))]
        edge_labels = {key: digest(b"B" + feature_bytes(edge_feat[j])) for key, j in edge_pos.items()}
        rng = np.random.default_rng(np.uint64(20260729) ^ np.uint64((int(graph_index) + 1) * 0x9E3779B1))
        perm = rng.permutation(len(atom_labels)); shuffled_atom_labels = [atom_labels[int(perm[u])] for u in range(len(atom_labels))]
        path_counts: Counter[int] = Counter(); shuffled_counts: Counter[int] = Counter()
        for length in range(1, args.max_path_length + 1):
            path_occurrences[length] += len(paths[length])
            for nodes in paths[length]:
                value = path_signature(nodes, atom_labels, edge_labels); key = (length, value); col = path_to_col.get(key)
                if col is None: col = len(path_to_col); path_to_col[key] = col; path_length.append(length)
                path_counts[col] += 1
                svalue = path_signature(nodes, shuffled_atom_labels, edge_labels); skey = (length, svalue); scol = shuffled_to_col.get(skey)
                if scol is None: scol = len(shuffled_to_col); shuffled_to_col[skey] = scol; shuffled_length.append(length)
                shuffled_counts[scol] += 1
        path_rows.append(path_counts); shuffled_rows.append(shuffled_counts)
        if (row + 1) % 5000 == 0:
            print(f"extracted {row + 1}/{len(train_graph_indices)} motif_vocab={len(motif_to_col)} path_vocab={len(path_to_col)}", flush=True)

    x_motif = sparse_from_counters(motif_rows, len(motif_to_col)); del motif_rows
    x_path = sparse_from_counters(path_rows, len(path_to_col)); del path_rows
    x_shuffled = sparse_from_counters(shuffled_rows, len(shuffled_to_col)); del shuffled_rows
    path_length_arr = np.asarray(path_length, dtype=np.int8); shuffled_length_arr = np.asarray(shuffled_length, dtype=np.int8)

    out: dict[str, Any] = {
        "protocol_id": "molhiv-exact-motif-transferable-simple-paths-v1", "date": "2026-07-29",
        "scope": "exact official-train only; rooted motifs plus canonical typed paths",
        "official_valid_evaluations": 0, "official_test_evaluations": 0,
        "isolation": "dataset items were accessed only for official-train graph indices",
        "config": vars(args), "n_train": int(len(y)), "n_positive": int(y.sum()),
        "n_unique_motifs": int(x_motif.shape[1]), "n_unique_paths": int(x_path.shape[1]),
        "n_unique_shuffled_paths": int(x_shuffled.shape[1]),
        "unique_paths_by_length": {str(l): int(np.sum(path_length_arr == l)) for l in range(1, args.max_path_length + 1)},
        "path_occurrences_by_length": {str(l): int(path_occurrences[l]) for l in range(1, args.max_path_length + 1)},
        "folds": [], "aggregate": {},
    }
    all_results: dict[str, list[dict[str, Any]]] = {}
    max_lengths = tuple(range(2, args.max_path_length + 1))
    for fold, (fit, held, fit_graph, held_graph) in enumerate(outer):
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)}", flush=True)
        motif_df = np.asarray((x_motif[fit] > 0).sum(axis=0)).ravel(); motif_cols = np.flatnonzero(motif_df >= args.min_df)
        xb_fit = transformed(x_motif[fit][:, motif_cols], "binary"); xb_held = transformed(x_motif[held][:, motif_cols], "binary")
        path_df = np.asarray((x_path[fit] > 0).sum(axis=0)).ravel(); shuffled_df = np.asarray((x_shuffled[fit] > 0).sum(axis=0)).ravel()
        fold_out: dict[str, Any] = {"fold": fold, "n_motif_features": int(len(motif_cols)), "results": {}, "path_feature_counts": {}}
        for c in (0.01, 0.03):
            key = f"baseline_motif_binary_c{c:g}"; row, ph = fit_eval(xb_fit, y[fit], xb_held, y[held], c)
            row.update({"C": c, "heldout_probabilities": ph.tolist()}); fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
        for max_len in max_lengths:
            pcols = np.flatnonzero((path_df >= args.min_df) & (path_length_arr <= max_len))
            spcols = np.flatnonzero((shuffled_df >= args.min_df) & (shuffled_length_arr <= max_len))
            xp_fit = transformed(x_path[fit][:, pcols], "binary"); xp_held = transformed(x_path[held][:, pcols], "binary")
            concat_fit = sparse.hstack([xb_fit, xp_fit], format="csr"); concat_held = sparse.hstack([xb_held, xp_held], format="csr")
            fold_out["path_feature_counts"][f"length1_{max_len}"] = {"real": int(len(pcols)), "shuffled": int(len(spcols))}
            for c in (0.01, 0.03):
                key = f"concat_paths1_{max_len}_binary_c{c:g}"; row, ph = fit_eval(concat_fit, y[fit], concat_held, y[held], c)
                row.update({"C": c, "max_path_length": max_len, "n_path_features": int(len(pcols)), "heldout_probabilities": ph.tolist()}); fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
            key = f"path_only1_{max_len}_binary_c0.03"; row, ph = fit_eval(xp_fit, y[fit], xp_held, y[held], 0.03)
            row.update({"C": 0.03, "max_path_length": max_len, "n_path_features": int(len(pcols)), "heldout_probabilities": ph.tolist()}); fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
            if max_len == args.max_path_length:
                xsp_fit = transformed(x_shuffled[fit][:, spcols], "binary"); xsp_held = transformed(x_shuffled[held][:, spcols], "binary")
                key = f"control_shuffled_concat_paths1_{max_len}_binary_c0.03"; row, ph = fit_eval(sparse.hstack([xb_fit, xsp_fit], format="csr"), y[fit], sparse.hstack([xb_held, xsp_held], format="csr"), y[held], 0.03)
                row.update({"C": 0.03, "max_path_length": max_len, "n_path_features": int(len(spcols)), "heldout_probabilities": ph.tolist()}); fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
        out["folds"].append(fold_out)

    out["aggregate"] = {key: summarize(rows) for key, rows in all_results.items() if len(rows) == 3}
    out["ranking"] = sorted(({"candidate": key, **value} for key, value in out["aggregate"].items()), key=lambda r: (r["mean_auc"], r["min_fold_auc"]), reverse=True)
    out["baseline_reference"] = {"candidate": "baseline_motif_binary_c0.03", **out["aggregate"]["baseline_motif_binary_c0.03"]}
    out["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": args.output, "vocab": out["unique_paths_by_length"], "top20": out["ranking"][:20], "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
