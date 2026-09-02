"""Robust learners for the full typed path vocabulary on official-train only.

This experiment keeps the representation fixed: canonical simple paths of one
through four bonds containing all OGB atom and bond features.  It tests whether
the main limitation is the *way path evidence is aggregated*, rather than a
missing vocabulary item.

Candidates are deliberately small and predeclared:
  - ordinary binary bag-of-paths logistic regression (reference);
  - scaffold-frequency reweighting, so repeated training scaffolds dominate less;
  - NB log-count-ratio scaling, a supervised but fold-local path evidence weight;
  - IDF/BM25 and row-normalized pooling, which reduce molecule-size effects;
  - NB evidence extrema appended to the sparse path vector, a small MIL-like
    readout that can react to the strongest positive/negative path evidence.

Every vocabulary filter and supervised transform is fitted inside an outer
scaffold fold.  Dataset graph items are accessed only for official-train indices.
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
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes
from code.run_molhiv_exact_motif_path_vocabulary_probe import (
    enumerate_simple_paths,
    path_signature,
    sparse_from_counters,
)


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def binary(x: sparse.csr_matrix) -> sparse.csr_matrix:
    z = x.astype(np.float32, copy=True)
    z.data.fill(1.0)
    return z


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    held = np.asarray([r["heldout_auc"] for r in rows], dtype=np.float64)
    fit = np.asarray([r["fit_auc"] for r in rows], dtype=np.float64)
    return {
        "fold_auc": held.tolist(),
        "mean_auc": float(held.mean()),
        "sample_std_auc": float(held.std(ddof=1)),
        "min_fold_auc": float(held.min()),
        "mean_fit_auc": float(fit.mean()),
        "mean_parameters": float(np.mean([r["trainable_parameters"] for r in rows])),
    }


def class_scaffold_weights(y: np.ndarray, groups: np.ndarray, gamma: float) -> np.ndarray:
    """Class balance times inverse scaffold frequency, normalized per class."""
    group_count = Counter(groups.tolist())
    gw = np.asarray([group_count[g] ** (-gamma) for g in groups.tolist()], dtype=np.float64)
    # Keep the total weight of each class equal and make gamma comparable.
    out = np.zeros(len(y), dtype=np.float64)
    for cls in (0, 1):
        mask = y == cls
        scaled = gw[mask] / max(float(gw[mask].mean()), 1e-12)
        out[mask] = scaled * (len(y) / (2.0 * int(mask.sum())))
    return out


def fit_eval(
    x_fit: sparse.csr_matrix,
    y_fit: np.ndarray,
    x_held: sparse.csr_matrix,
    y_held: np.ndarray,
    c: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    model = LogisticRegression(
        C=c,
        class_weight=None if sample_weight is not None else "balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=0,
    )
    model.fit(x_fit, y_fit, sample_weight=sample_weight)
    pfit = model.predict_proba(x_fit)[:, 1]
    pheld = model.predict_proba(x_held)[:, 1]
    row = {
        "trainable_parameters": int(x_fit.shape[1] + 1),
        "fit_auc": float(roc_auc_score(y_fit, pfit)),
        "heldout_auc": float(roc_auc_score(y_held, pheld)),
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
        "heldout_probabilities": pheld.tolist(),
        "heldout_probability_sha256": sha(pheld),
    }
    return row, pheld


def nb_ratio(x: sparse.csr_matrix, y: np.ndarray, alpha: float = 1.0, clip: float = 4.0) -> np.ndarray:
    pos = alpha + np.asarray(x[y == 1].sum(axis=0), dtype=np.float64).ravel()
    neg = alpha + np.asarray(x[y == 0].sum(axis=0), dtype=np.float64).ravel()
    pos /= pos.sum()
    neg /= neg.sum()
    return np.clip(np.log(pos / neg), -clip, clip).astype(np.float32)


def idf_vector(x: sparse.csr_matrix) -> np.ndarray:
    df = np.asarray((x > 0).sum(axis=0), dtype=np.float64).ravel()
    n = x.shape[0]
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def bm25_pair(
    train_counts: sparse.csr_matrix,
    held_counts: sparse.csr_matrix,
    k1: float = 1.2,
    b: float = 0.5,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    df = np.asarray((train_counts > 0).sum(axis=0), dtype=np.float64).ravel()
    n = train_counts.shape[0]
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    train_dl = np.asarray(train_counts.sum(axis=1), dtype=np.float64).ravel()
    avgdl = max(float(train_dl.mean()), 1.0)

    def apply(x: sparse.csr_matrix) -> sparse.csr_matrix:
        z = x.astype(np.float32, copy=True)
        dl = np.asarray(x.sum(axis=1), dtype=np.float64).ravel()
        row_ids = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        denom = z.data + k1 * (1.0 - b + b * dl[row_ids] / avgdl)
        z.data = z.data * (k1 + 1.0) / denom
        return z.multiply(idf).tocsr()

    return apply(train_counts), apply(held_counts)


def row_sqrt_normalize(x: sparse.csr_matrix) -> sparse.csr_matrix:
    z = binary(x)
    nnz = np.diff(z.indptr).astype(np.float32)
    scale = 1.0 / np.sqrt(np.maximum(nnz, 1.0))
    return sparse.diags(scale, format="csr") @ z


def evidence_extrema(x: sparse.csr_matrix, ratio: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Strongest positive/negative NB evidence and active count by path length."""
    out = np.zeros((x.shape[0], 12), dtype=np.float32)
    for i in range(x.shape[0]):
        cols = x.indices[x.indptr[i] : x.indptr[i + 1]]
        if not len(cols):
            continue
        for length in range(1, 5):
            use = cols[lengths[cols] == length]
            base = 3 * (length - 1)
            if len(use):
                vals = ratio[use]
                out[i, base] = max(float(vals.max()), 0.0)
                out[i, base + 1] = max(float((-vals).max()), 0.0)
                out[i, base + 2] = np.log1p(len(use))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/path_robust_learner_fulltrain_probe_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset

    ds = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = ds.get_idx_split()
    train = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    y = np.asarray(ds.labels).reshape(-1).astype(np.int64)[train]
    pos = np.full(len(ds), -1, dtype=np.int64)
    pos[train] = np.arange(len(train))

    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train):
            raise ValueError("official train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer = []
        for fold in range(3):
            fit_global = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
            held_global = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
            outer.append((pos[fit_global], pos[held_global]))
    if np.any(pos[forbidden] >= 0):
        raise AssertionError("forbidden split contamination")

    vocab: dict[tuple[int, bytes], int] = {}
    path_lengths: list[int] = []
    rows: list[Counter[int]] = []
    raw_occurrences_by_length = Counter()
    for row_i, graph_i in enumerate(train.tolist()):
        graph, _ = ds[int(graph_i)]
        nf = np.asarray(graph["node_feat"], dtype=np.int64)
        ei = np.asarray(graph["edge_index"], dtype=np.int64)
        ef = np.asarray(graph["edge_feat"], dtype=np.int64)
        paths, edge_pos = enumerate_simple_paths(ei, len(nf), 4)
        atom_labels = [digest(b"A" + feature_bytes(nf[u])) for u in range(len(nf))]
        edge_labels = {key: digest(b"B" + feature_bytes(ef[j])) for key, j in edge_pos.items()}
        counts: Counter[int] = Counter()
        for length in range(1, 5):
            for nodes in paths[length]:
                key = (length, path_signature(nodes, atom_labels, edge_labels))
                col = vocab.get(key)
                if col is None:
                    col = len(vocab)
                    vocab[key] = col
                    path_lengths.append(length)
                counts[col] += 1
                raw_occurrences_by_length[length] += 1
        rows.append(counts)
        if (row_i + 1) % 5000 == 0:
            print(f"extracted {row_i + 1}/{len(train)} vocab={len(vocab)}", flush=True)
    x_all = sparse_from_counters(rows, len(vocab))
    lengths_all = np.asarray(path_lengths, dtype=np.int8)
    del rows, vocab

    out: dict[str, Any] = {
        "protocol_id": "molhiv-full-typed-path-robust-learner-v1",
        "date": "2026-07-29",
        "scope": "official-train only; fixed full typed paths length 1-4",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "isolation": "dataset graph items accessed only for official train",
        "candidate_policy": "representation fixed; compare predeclared aggregation/weighting rules",
        "config": vars(args),
        "n_train": int(len(train)),
        "n_positive": int(y.sum()),
        "n_scaffolds": int(len(set(groups.tolist()))),
        "raw_vocab": int(x_all.shape[1]),
        "raw_occurrences_by_length": {str(k): int(v) for k, v in sorted(raw_occurrences_by_length.items())},
        "folds": [],
        "aggregate": {},
    }
    all_results: dict[str, list[dict[str, Any]]] = {}

    for fold, (fit, held) in enumerate(outer):
        df = np.asarray((x_all[fit] > 0).sum(axis=0), dtype=np.int64).ravel()
        cols = np.flatnonzero(df >= args.min_df)
        lengths = lengths_all[cols]
        counts_fit = x_all[fit][:, cols].tocsr()
        counts_held = x_all[held][:, cols].tocsr()
        xb_fit = binary(counts_fit)
        xb_held = binary(counts_held)
        total_held_occ = float(x_all[held].sum())
        retained_held_occ = float(counts_held.sum())
        diagnostics = {
            "fold": fold,
            "n_fit": int(len(fit)),
            "n_held": int(len(held)),
            "fit_positive": int(y[fit].sum()),
            "held_positive": int(y[held].sum()),
            "n_fit_scaffolds": int(len(set(groups[fit].tolist()))),
            "n_held_scaffolds": int(len(set(groups[held].tolist()))),
            "n_features": int(len(cols)),
            "held_occurrence_coverage": retained_held_occ / max(total_held_occ, 1.0),
            "held_zero_feature_graphs": int(np.sum(np.diff(counts_held.indptr) == 0)),
            "results": {},
        }
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)} features={len(cols)} coverage={diagnostics['held_occurrence_coverage']:.4f}", flush=True)

        def record(key: str, xf: sparse.csr_matrix, xh: sparse.csr_matrix, c: float, sw: np.ndarray | None = None, extra: dict[str, Any] | None = None) -> None:
            rr, _ = fit_eval(xf, y[fit], xh, y[held], c, sw)
            rr.update({"C": c})
            if extra:
                rr.update(extra)
            diagnostics["results"][key] = rr
            all_results.setdefault(key, []).append(rr)
            print(f"  {key}: {rr['heldout_auc']:.6f}", flush=True)

        # Reference capacity/regularization.
        record("binary_c0.01", xb_fit, xb_held, 0.01)
        record("binary_c0.03", xb_fit, xb_held, 0.03)

        # Reduce domination by large repeated scaffold groups.
        for gamma in (0.25, 0.5, 1.0):
            sw = class_scaffold_weights(y[fit], groups[fit], gamma)
            record(f"binary_scaffold_weight_g{gamma:g}_c0.03", xb_fit, xb_held, 0.03, sw, {"scaffold_weight_gamma": gamma})

        # Fold-local supervised path evidence scaling (NB-SVM style).
        ratio = nb_ratio(xb_fit, y[fit], alpha=1.0, clip=4.0)
        xnb_fit = xb_fit.multiply(ratio).tocsr()
        xnb_held = xb_held.multiply(ratio).tocsr()
        record("nb_binary_c0.01", xnb_fit, xnb_held, 0.01, extra={"nb_alpha": 1.0, "nb_clip": 4.0})
        record("nb_binary_c0.03", xnb_fit, xnb_held, 0.03, extra={"nb_alpha": 1.0, "nb_clip": 4.0})
        sw = class_scaffold_weights(y[fit], groups[fit], 0.25)
        record("nb_binary_scaffold_weight_g0.25_c0.03", xnb_fit, xnb_held, 0.03, sw, {"nb_alpha": 1.0, "nb_clip": 4.0, "scaffold_weight_gamma": 0.25})

        # Unsupervised document-frequency and molecule-length corrections.
        idf = idf_vector(xb_fit)
        record("tfidf_binary_c0.01", xb_fit.multiply(idf).tocsr(), xb_held.multiply(idf).tocsr(), 0.01)
        bm_fit, bm_held = bm25_pair(counts_fit, counts_held, k1=1.2, b=0.5)
        record("bm25_c0.03", bm_fit, bm_held, 0.03, extra={"bm25_k1": 1.2, "bm25_b": 0.5})
        record("row_sqrt_normalized_binary_c0.03", row_sqrt_normalize(counts_fit), row_sqrt_normalize(counts_held), 0.03)

        # A tiny set-style readout: append strongest positive/negative path
        # evidence for each length.  Only 12 additional trainable coefficients.
        ef = evidence_extrema(xb_fit, ratio, lengths)
        eh = evidence_extrema(xb_held, ratio, lengths)
        scale = np.maximum(ef.std(axis=0, ddof=0), 1e-3)
        ef = ef / scale
        eh = eh / scale
        xmil_fit = sparse.hstack([xnb_fit, sparse.csr_matrix(ef)], format="csr")
        xmil_held = sparse.hstack([xnb_held, sparse.csr_matrix(eh)], format="csr")
        record("nb_plus_evidence_extrema_c0.01", xmil_fit, xmil_held, 0.01, extra={"dense_readout_features": 12})
        record("nb_plus_evidence_extrema_c0.03", xmil_fit, xmil_held, 0.03, extra={"dense_readout_features": 12})

        out["folds"].append(diagnostics)

    out["aggregate"] = {k: summarize(v) for k, v in all_results.items() if len(v) == 3}
    out["ranking"] = sorted(
        ({"candidate": k, **v} for k, v in out["aggregate"].items()),
        key=lambda r: (r["mean_auc"], r["min_fold_auc"]), reverse=True,
    )
    out["baseline_reference"] = {"candidate": "binary_c0.03", **out["aggregate"]["binary_c0.03"]}
    out["elapsed_sec"] = time.time() - t0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": str(output), "top20": out["ranking"][:20], "elapsed_sec": out["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
