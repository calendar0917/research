"""Frozen retrospective official-test evaluation for the exact path model."""
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


def encode(dataset: Any, indices: np.ndarray, motif_vocab: dict[tuple[int, bytes], int], path_vocab: dict[tuple[int, bytes], int], add: bool, max_path_length: int, need_motif: bool) -> tuple[list[Counter[int]], list[Counter[int]]]:
    motif_rows: list[Counter[int]] = []; path_rows: list[Counter[int]] = []
    for row, graph_index in enumerate(indices.tolist()):
        g, _ = dataset[int(graph_index)]
        node_feat = np.asarray(g["node_feat"], dtype=np.int64); edge_index = np.asarray(g["edge_index"], dtype=np.int64); edge_feat = np.asarray(g["edge_feat"], dtype=np.int64)
        motif_counts: Counter[int] = Counter()
        if need_motif:
            for radius, values in enumerate(rooted_signatures(node_feat, edge_index, edge_feat, 3)):
                for value in values:
                    key = (radius, value); col = motif_vocab.get(key)
                    if col is None and add: col = len(motif_vocab); motif_vocab[key] = col
                    if col is not None: motif_counts[col] += 1
        motif_rows.append(motif_counts)
        paths, edge_pos = enumerate_simple_paths(edge_index, len(node_feat), max_path_length)
        atom_labels = [digest(b"A" + feature_bytes(node_feat[u])) for u in range(len(node_feat))]
        edge_labels = {key: digest(b"B" + feature_bytes(edge_feat[j])) for key, j in edge_pos.items()}
        path_counts: Counter[int] = Counter()
        for length in range(1, max_path_length + 1):
            for nodes in paths[length]:
                key = (length, path_signature(nodes, atom_labels, edge_labels)); col = path_vocab.get(key)
                if col is None and add: col = len(path_vocab); path_vocab[key] = col
                if col is not None: path_counts[col] += 1
        path_rows.append(path_counts)
        if (row + 1) % 5000 == 0:
            print(f"encoded {row + 1}/{len(indices)} add={add} path_vocab={len(path_vocab)}", flush=True)
    return motif_rows, path_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="tracks/ksvd/results/molhiv/exact_path_official_test_freeze_v1.json")
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/exact_path_official_test_20260729.json")
    args = ap.parse_args(); t0 = time.time(); freeze_path = Path(args.freeze)
    seal_path = Path(args.freeze + ".sha256"); actual_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    expected_hash = seal_path.read_text().split()[0]
    if actual_hash != expected_hash: raise ValueError("freeze manifest SHA256 mismatch")
    freeze = json.loads(freeze_path.read_text()); cfg = freeze["config"]
    repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb")); split = dataset.get_idx_split()
    train = np.asarray(split["train"], dtype=np.int64); valid = np.asarray(split["valid"], dtype=np.int64); test = np.asarray(split["test"], dtype=np.int64)
    if sha(train) != freeze["expected_train_indices_sha256"]: raise ValueError("official train indices mismatch")
    labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64); y_train = labels[train]; y_test = labels[test]
    need_motif = cfg["representation"] == "motif_plus_path"
    motif_vocab: dict[tuple[int, bytes], int] = {}; path_vocab: dict[tuple[int, bytes], int] = {}
    train_motif_rows, train_path_rows = encode(dataset, train, motif_vocab, path_vocab, True, int(cfg["max_path_length"]), need_motif)
    test_motif_rows, test_path_rows = encode(dataset, test, motif_vocab, path_vocab, False, int(cfg["max_path_length"]), need_motif)
    if set(valid.tolist()) & (set(train.tolist()) | set(test.tolist())): raise AssertionError("official split overlap")
    xp_train_all = sparse_from_counters(train_path_rows, len(path_vocab)); xp_test_all = sparse_from_counters(test_path_rows, len(path_vocab))
    path_df = np.asarray((xp_train_all > 0).sum(axis=0)).ravel(); path_cols = np.flatnonzero(path_df >= int(cfg["min_df"]))
    xp_train = transformed(xp_train_all[:, path_cols], "binary"); xp_test = transformed(xp_test_all[:, path_cols], "binary")
    if need_motif:
        xm_train_all = sparse_from_counters(train_motif_rows, len(motif_vocab)); xm_test_all = sparse_from_counters(test_motif_rows, len(motif_vocab))
        motif_df = np.asarray((xm_train_all > 0).sum(axis=0)).ravel(); motif_cols = np.flatnonzero(motif_df >= int(cfg["min_df"]))
        x_train = sparse.hstack([transformed(xm_train_all[:, motif_cols], "binary"), xp_train], format="csr")
        x_test = sparse.hstack([transformed(xm_test_all[:, motif_cols], "binary"), xp_test], format="csr")
    else:
        motif_cols = np.empty(0, dtype=np.int64); x_train = xp_train; x_test = xp_test
    actual_train_hashes = sparse_hashes(x_train)
    if actual_train_hashes != freeze["expected_train_feature_hashes"]:
        raise ValueError(f"frozen train feature mismatch: {actual_train_hashes} != {freeze['expected_train_feature_hashes']}")
    model = LogisticRegression(C=float(cfg["C"]), class_weight=cfg["class_weight"], solver=cfg["solver"], max_iter=int(cfg["max_iter"]), random_state=int(cfg["random_state"]))
    model.fit(x_train, y_train)
    coef_hash = sha(np.asarray(model.coef_, dtype=np.float64)); intercept_hash = sha(np.asarray(model.intercept_, dtype=np.float64)); n_iter = np.asarray(model.n_iter_, dtype=int).tolist()
    if coef_hash != freeze["expected_coefficient_sha256"] or intercept_hash != freeze["expected_intercept_sha256"] or n_iter != freeze["expected_n_iter"]:
        raise ValueError("frozen classifier reproduction mismatch")
    p_train = model.predict_proba(x_train)[:, 1]; p_test = model.predict_proba(x_test)[:, 1]
    out = {
        "protocol_id": "molhiv-exact-path-official-test-v1", "date": "2026-07-29", "status": "retrospective_controlled_frozen_evaluation_complete",
        "freeze_manifest": args.freeze, "freeze_manifest_sha256": actual_hash, "selected_candidate": freeze["selected_candidate"],
        "fit_split": "exact official train", "evaluation_split": "exact official test", "official_valid_evaluations_in_this_runner": 0, "official_test_evaluations_in_this_runner": 1,
        "historical_test_disclosure": freeze["disclosure"], "valid_isolation": "official valid graph items were not accessed by this runner",
        "n_train": int(len(train)), "n_train_positive": int(y_train.sum()), "n_test": int(len(test)), "n_test_positive": int(y_test.sum()),
        "representation": cfg["representation"], "max_path_length": int(cfg["max_path_length"]), "n_motif_features": int(len(motif_cols)), "n_path_features": int(len(path_cols)),
        "trainable_parameters": int(x_train.shape[1] + 1), "fit_auc": float(roc_auc_score(y_train, p_train)), "official_test_auc": float(roc_auc_score(y_test, p_test)),
        "test_indices": test.astype(int).tolist(), "test_y": y_test.astype(int).tolist(), "test_probabilities": p_test.tolist(), "test_probability_sha256": sha(p_test),
        "train_feature_hashes": actual_train_hashes, "test_feature_hashes": sparse_hashes(x_test), "coefficient_sha256": coef_hash, "intercept_sha256": intercept_hash, "n_iter": n_iter,
        "elapsed_sec": time.time() - t0,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": args.output, "freeze_sha256": actual_hash, "candidate": freeze["selected_candidate"]["candidate"], "official_valid_auc_frozen_selection": freeze["selected_candidate"]["official_valid_auc"], "official_test_auc": out["official_test_auc"], "parameters": out["trainable_parameters"], "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__": main()
