"""Predeclared exact-path candidates on MolHIV official validation.

Candidate set was frozen from three scaffold folds inside official train:
  1. path-only lengths 1..4, binary, C=0.03;
  2. rooted motifs + paths 1..4, binary, C in {0.01, 0.03};
  3. compact rooted motifs + paths 1..3, binary, C=0.03.

Only official train builds vocabularies and document-frequency filters. Official
validation is lookup-only. Official test graphs are never accessed.
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
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes, rooted_signatures, transformed
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths, path_signature, sparse_from_counters


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def sparse_hashes(x: sparse.csr_matrix) -> dict[str, Any]:
    return {"shape": list(x.shape), "nnz": int(x.nnz), "data_sha256": sha(x.data), "indices_sha256": sha(x.indices), "indptr_sha256": sha(x.indptr)}


def encode_graphs(dataset: Any, indices: np.ndarray, motif_vocab: dict[tuple[int, bytes], int], path_vocab: dict[tuple[int, bytes], int], add: bool, max_path_length: int) -> tuple[list[Counter[int]], list[Counter[int]]]:
    motif_rows: list[Counter[int]] = []; path_rows: list[Counter[int]] = []
    for row, graph_index in enumerate(indices.tolist()):
        g, _ = dataset[int(graph_index)]
        node_feat = np.asarray(g["node_feat"], dtype=np.int64); edge_index = np.asarray(g["edge_index"], dtype=np.int64); edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
        motif_counts: Counter[int] = Counter()
        for radius, values in enumerate(rooted_signatures(node_feat, edge_index, edge_feat, 3)):
            for value in values:
                key = (radius, value); col = motif_vocab.get(key)
                if col is None and add:
                    col = len(motif_vocab); motif_vocab[key] = col
                if col is not None: motif_counts[col] += 1
        motif_rows.append(motif_counts)
        paths, edge_pos = enumerate_simple_paths(edge_index, len(node_feat), max_path_length)
        atom_labels = [digest(b"A" + feature_bytes(node_feat[u])) for u in range(len(node_feat))]
        edge_labels = {key: digest(b"B" + feature_bytes(edge_feat[j])) for key, j in edge_pos.items()}
        path_counts: Counter[int] = Counter()
        for length in range(1, max_path_length + 1):
            for nodes in paths[length]:
                key = (length, path_signature(nodes, atom_labels, edge_labels)); col = path_vocab.get(key)
                if col is None and add:
                    col = len(path_vocab); path_vocab[key] = col
                if col is not None: path_counts[col] += 1
        path_rows.append(path_counts)
        if (row + 1) % 5000 == 0:
            print(f"encoded {row + 1}/{len(indices)} add={add} motif_vocab={len(motif_vocab)} path_vocab={len(path_vocab)}", flush=True)
    return motif_rows, path_rows


def fit_candidate(name: str, x_train: sparse.csr_matrix, y_train: np.ndarray, x_valid: sparse.csr_matrix, y_valid: np.ndarray, valid_indices: np.ndarray, c: float, meta: dict[str, Any]) -> tuple[dict[str, Any], LogisticRegression]:
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
    model.fit(x_train, y_train)
    pt = model.predict_proba(x_train)[:, 1]; pv = model.predict_proba(x_valid)[:, 1]
    row = {
        "candidate": name, **meta, "C": c, "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000, "random_state": 0,
        "trainable_parameters": int(x_train.shape[1] + 1), "fit_auc": float(roc_auc_score(y_train, pt)),
        "official_valid_auc": float(roc_auc_score(y_valid, pv)), "coefficient_sha256": sha(np.asarray(model.coef_, dtype=np.float64)),
        "intercept_sha256": sha(np.asarray(model.intercept_, dtype=np.float64)), "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
        "official_valid_indices": valid_indices.astype(int).tolist(), "official_valid_y": y_valid.astype(int).tolist(),
        "official_valid_probabilities": pv.tolist(), "official_valid_probability_sha256": sha(pv),
        "train_feature_hashes": sparse_hashes(x_train),
    }
    return row, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/exact_path_official_valid_20260729.json")
    ap.add_argument("--freeze", default="tracks/ksvd/results/molhiv/exact_path_official_test_freeze_v1.json")
    args = ap.parse_args(); t0 = time.time(); repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb")); split = dataset.get_idx_split()
    train = np.asarray(split["train"], dtype=np.int64); valid = np.asarray(split["valid"], dtype=np.int64); test = np.asarray(split["test"], dtype=np.int64)
    labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64); y_train = labels[train]; y_valid = labels[valid]

    motif_vocab: dict[tuple[int, bytes], int] = {}; path_vocab: dict[tuple[int, bytes], int] = {}
    train_motif_rows, train_path_rows = encode_graphs(dataset, train, motif_vocab, path_vocab, True, 4)
    valid_motif_rows, valid_path_rows = encode_graphs(dataset, valid, motif_vocab, path_vocab, False, 4)
    # Isolation assertion: only train and valid indices have been passed to dataset.__getitem__.
    if set(train.tolist()) & set(test.tolist()) or set(valid.tolist()) & set(test.tolist()):
        raise AssertionError("official split overlap")
    xm_train_all = sparse_from_counters(train_motif_rows, len(motif_vocab)); xm_valid_all = sparse_from_counters(valid_motif_rows, len(motif_vocab))
    xp_train_all = sparse_from_counters(train_path_rows, len(path_vocab)); xp_valid_all = sparse_from_counters(valid_path_rows, len(path_vocab))
    motif_df = np.asarray((xm_train_all > 0).sum(axis=0)).ravel(); motif_cols = np.flatnonzero(motif_df >= args.min_df)
    path_lengths = np.empty(len(path_vocab), dtype=np.int8)
    for (length, _), col in path_vocab.items(): path_lengths[col] = length
    path_df = np.asarray((xp_train_all > 0).sum(axis=0)).ravel()
    path_cols4 = np.flatnonzero(path_df >= args.min_df); path_cols3 = np.flatnonzero((path_df >= args.min_df) & (path_lengths <= 3))
    xm_train = transformed(xm_train_all[:, motif_cols], "binary"); xm_valid = transformed(xm_valid_all[:, motif_cols], "binary")
    xp4_train = transformed(xp_train_all[:, path_cols4], "binary"); xp4_valid = transformed(xp_valid_all[:, path_cols4], "binary")
    xp3_train = transformed(xp_train_all[:, path_cols3], "binary"); xp3_valid = transformed(xp_valid_all[:, path_cols3], "binary")
    matrices = {
        "path_only1_4_binary_c0.03": (xp4_train, xp4_valid, 0.03, {"representation": "path_only", "max_path_length": 4, "n_motif_features": 0, "n_path_features": int(len(path_cols4))}),
        "concat_paths1_4_binary_c0.01": (sparse.hstack([xm_train, xp4_train], format="csr"), sparse.hstack([xm_valid, xp4_valid], format="csr"), 0.01, {"representation": "motif_plus_path", "max_path_length": 4, "n_motif_features": int(len(motif_cols)), "n_path_features": int(len(path_cols4))}),
        "concat_paths1_4_binary_c0.03": (sparse.hstack([xm_train, xp4_train], format="csr"), sparse.hstack([xm_valid, xp4_valid], format="csr"), 0.03, {"representation": "motif_plus_path", "max_path_length": 4, "n_motif_features": int(len(motif_cols)), "n_path_features": int(len(path_cols4))}),
        "concat_paths1_3_binary_c0.03": (sparse.hstack([xm_train, xp3_train], format="csr"), sparse.hstack([xm_valid, xp3_valid], format="csr"), 0.03, {"representation": "motif_plus_path", "max_path_length": 3, "n_motif_features": int(len(motif_cols)), "n_path_features": int(len(path_cols3))}),
    }
    results = []; models: dict[str, LogisticRegression] = {}
    for name, (xt, xv, c, meta) in matrices.items():
        row, model = fit_candidate(name, xt, y_train, xv, y_valid, valid, c, meta); results.append(row); models[name] = model
        print(f"{name}: valid={row['official_valid_auc']:.6f} fit={row['fit_auc']:.6f} params={row['trainable_parameters']}", flush=True)
    selected = sorted(results, key=lambda r: (-r["official_valid_auc"], r["trainable_parameters"], r["candidate"]))[0]
    out = {
        "protocol_id": "molhiv-exact-path-official-valid-v1", "date": "2026-07-29", "status": "official_valid_candidate_selection_complete",
        "scope": "predeclared candidates selected on official valid; official test not accessed", "official_valid_evaluations": len(results), "official_test_evaluations": 0,
        "test_isolation": "dataset items were accessed only for exact official train and valid indices",
        "candidate_policy": "four candidates frozen from exact official-train three-fold scaffold results; select max valid AUC, then fewer parameters",
        "n_train": int(len(train)), "n_train_positive": int(y_train.sum()), "n_valid": int(len(valid)), "n_valid_positive": int(y_valid.sum()),
        "min_df": args.min_df, "raw_motif_vocab": int(len(motif_vocab)), "raw_path_vocab": int(len(path_vocab)),
        "eligible_motif_features": int(len(motif_cols)), "eligible_path_features_l4": int(len(path_cols4)), "eligible_path_features_l3": int(len(path_cols3)),
        "train_indices_sha256": sha(train), "valid_indices_sha256": sha(valid), "results": results, "selected_candidate": selected, "elapsed_sec": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    cfg = {
        "protocol_id": "molhiv-exact-path-official-test-freeze-v1", "date": "2026-07-29", "status": "frozen_before_test_evaluation",
        "selection_source": args.output, "selection_source_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "selected_candidate": selected, "config": {"min_df": args.min_df, "representation": selected["representation"], "max_path_length": selected["max_path_length"], "C": selected["C"], "class_weight": "balanced", "solver": "liblinear", "max_iter": 3000, "random_state": 0},
        "expected_train_indices_sha256": sha(train), "expected_train_feature_hashes": selected["train_feature_hashes"],
        "expected_coefficient_sha256": selected["coefficient_sha256"], "expected_intercept_sha256": selected["intercept_sha256"], "expected_n_iter": selected["n_iter"],
        "disclosure": "MolHIV official test has been evaluated elsewhere in this repository; this is a retrospective controlled frozen evaluation, not an untouched-test claim.",
    }
    Path(args.freeze).write_text(json.dumps(cfg, indent=2))
    seal = hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest(); Path(args.freeze + ".sha256").write_text(f"{seal}  {args.freeze}\n")
    print(json.dumps({"output": args.output, "selected": {"candidate": selected["candidate"], "valid_auc": selected["official_valid_auc"], "parameters": selected["trainable_parameters"]}, "freeze": args.freeze, "freeze_sha256": seal, "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__": main()
