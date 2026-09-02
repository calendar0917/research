"""Soft scaffold-stability regularisation for exact local motif vocabularies.

This development-only probe uses exact official-train graphs and the existing
three outer scaffold folds.  It keeps every motif passing an outer-fit document
frequency threshold.  Instead of hard top-k selection, nested scaffold
partitions estimate a reliability in [0, 1], and each feature column is scaled
between ``floor`` and 1 before L2-logistic regression.  Smaller scales therefore
receive stronger effective regularisation but are not deleted.

A shuffled-label reliability is evaluated as a negative control.  Official
valid/test graphs are never accessed.
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
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_stable_exact_motif_probe import rooted_signatures, transformed, z_association


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def partition_z(x_binary: sparse.csr_matrix, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    rows: list[np.ndarray] = []
    meta: list[dict[str, int]] = []
    for inner, (_, held) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        held = np.asarray(held, dtype=np.int64)
        rows.append(z_association(x_binary[held], y[held]))
        meta.append({
            "inner_partition": inner,
            "n": int(len(held)),
            "n_positive": int(y[held].sum()),
            "n_groups": int(len(set(groups[held].tolist()))),
        })
    return np.stack(rows, axis=0), meta


def positive_rank(values: np.ndarray) -> np.ndarray:
    """Map strictly positive scores to deterministic ranks in (0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    idx = np.flatnonzero(values > 0)
    if not len(idx):
        return out
    # Stable deterministic tie handling: equal values receive their average rank.
    vals = values[idx]
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    ranks = np.empty(len(idx), dtype=np.float64)
    start = 0
    while start < len(idx):
        end = start + 1
        while end < len(idx) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end) / len(idx)
        start = end
    out[idx] = ranks
    return out


def reliability_from_z(z: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float]]:
    abs_z = np.abs(z)
    sign_consistency = np.abs(np.sum(z, axis=0)) / np.maximum(np.sum(abs_z, axis=0), 1e-12)
    if method == "hardmin_rank":
        same_sign = np.all(z > 0, axis=0) | np.all(z < 0, axis=0)
        raw = np.where(same_sign, np.min(abs_z, axis=0), 0.0)
        reliability = positive_rank(raw)
    elif method == "consensus_strength":
        # Continuous sign agreement times the percentile of median association
        # strength. Unlike hardmin_rank, one mildly disagreeing partition does
        # not force a motif to the floor.
        strength_rank = positive_rank(np.median(abs_z, axis=0))
        raw = sign_consistency * strength_rank
        reliability = np.clip(raw, 0.0, 1.0)
    else:
        raise ValueError(method)
    q = np.quantile(reliability, [0, .25, .5, .75, .9, .99, 1])
    return reliability.astype(np.float32), {
        "fraction_nonzero": float(np.mean(reliability > 0)),
        "mean": float(np.mean(reliability)),
        "q0": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
        "q75": float(q[3]), "q90": float(q[4]), "q99": float(q[5]), "q100": float(q[6]),
        "mean_sign_consistency": float(np.mean(sign_consistency)),
    }


def scale_columns(x: sparse.csr_matrix, scales: np.ndarray) -> sparse.csr_matrix:
    return x.multiply(np.asarray(scales, dtype=np.float32)).tocsr()


def fit_eval(x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_held: sparse.csr_matrix, y_held: np.ndarray, c: float) -> tuple[dict[str, Any], np.ndarray]:
    model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0)
    model.fit(x_fit, y_fit)
    pf = model.predict_proba(x_fit)[:, 1]
    ph = model.predict_proba(x_held)[:, 1]
    return {
        "trainable_parameters": int(x_fit.shape[1] + 1),
        "fit_auc": float(roc_auc_score(y_fit, pf)),
        "heldout_auc": float(roc_auc_score(y_held, ph)),
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
    }, ph


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = np.asarray([r["heldout_auc"] for r in rows], dtype=float)
    f = np.asarray([r["fit_auc"] for r in rows], dtype=float)
    return {
        "fold_auc": a.tolist(), "mean_auc": float(a.mean()),
        "sample_std_auc": float(a.std(ddof=1)), "min_fold_auc": float(a.min()),
        "mean_fit_auc": float(f.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/exact_motif_soft_stability_fulltrain_probe_20260729.json")
    args = ap.parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]

    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split()
    train_graph_indices = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    labels_all = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    y = labels_all[train_graph_indices]
    position = np.full(len(dataset), -1, dtype=np.int64)
    position[train_graph_indices] = np.arange(len(train_graph_indices), dtype=np.int64)
    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train_graph_indices):
            raise ValueError("official train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer = []
        for fold in range(3):
            fit_graph = np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)
            held_graph = np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)
            outer.append((position[fit_graph], position[held_graph], fit_graph, held_graph))
    if np.any(position[forbidden] >= 0):
        raise AssertionError("forbidden split mapped into train rows")

    token_to_col: dict[tuple[int, bytes], int] = {}
    row_maps: list[Counter[int]] = []
    token_radius: list[int] = []
    for row, graph_index in enumerate(train_graph_indices.tolist()):
        g, _ = dataset[int(graph_index)]
        signatures = rooted_signatures(np.asarray(g["node_feat"], dtype=np.int64), np.asarray(g["edge_index"], dtype=np.int64), np.asarray(g["edge_feat"], dtype=np.int64), 3)
        counts: Counter[int] = Counter()
        for radius, values in enumerate(signatures):
            for value in values:
                key = (radius, value)
                col = token_to_col.get(key)
                if col is None:
                    col = len(token_to_col); token_to_col[key] = col; token_radius.append(radius)
                counts[col] += 1
        row_maps.append(counts)
        if (row + 1) % 5000 == 0:
            print(f"extracted {row + 1}/{len(train_graph_indices)} graphs vocab={len(token_to_col)}", flush=True)
    indptr = [0]; indices: list[int] = []; data: list[int] = []
    for counts in row_maps:
        for col, count in sorted(counts.items()):
            indices.append(col); data.append(count)
        indptr.append(len(indices))
    x_all = sparse.csr_matrix((np.asarray(data, dtype=np.int16), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)), shape=(len(train_graph_indices), len(token_to_col)))
    del row_maps

    out: dict[str, Any] = {
        "protocol_id": "molhiv-exact-motif-soft-scaffold-stability-v1",
        "date": "2026-07-29",
        "scope": "exact official-train only; three outer scaffold folds; nested soft reliability",
        "official_valid_evaluations": 0, "official_test_evaluations": 0,
        "isolation": "dataset items were accessed only for official-train graph indices",
        "config": vars(args), "n_train": int(len(y)), "n_positive": int(y.sum()),
        "n_unique_motifs": int(x_all.shape[1]), "feature_matrix_nnz": int(x_all.nnz),
        "train_indices_sha256": sha(train_graph_indices), "folds": [], "aggregate": {},
    }
    all_results: dict[str, list[dict[str, Any]]] = {}
    floors = (0.25, 0.5, 0.75)
    methods = ("hardmin_rank", "consensus_strength")
    for fold, (fit, held, fit_graph, held_graph) in enumerate(outer):
        print(f"outer fold {fold}: fit={len(fit)} held={len(held)}", flush=True)
        df = np.asarray((x_all[fit] > 0).sum(axis=0)).ravel()
        cols = np.flatnonzero(df >= args.min_df)
        xfit = transformed(x_all[fit][:, cols], "binary")
        xheld = transformed(x_all[held][:, cols], "binary")
        z, inner_meta = partition_z(xfit, y[fit], groups[fit], 20260729 + fold)
        rng = np.random.default_rng(20260729 + fold)
        shuffled_y = y[fit].copy(); rng.shuffle(shuffled_y)
        z_shuffled, shuffled_meta = partition_z(xfit, shuffled_y, groups[fit], 20260829 + fold)
        fold_out: dict[str, Any] = {
            "fold": fold, "n_fit": int(len(fit)), "n_heldout": int(len(held)),
            "n_features": int(len(cols)), "fit_graph_indices_sha256": sha(fit_graph),
            "heldout_graph_indices_sha256": sha(held_graph), "nested_partitions": inner_meta,
            "shuffled_nested_partitions": shuffled_meta, "reliability": {}, "results": {},
        }
        # Unscaled references at both predeclared capacities.
        for c in (0.01, 0.03):
            key = f"baseline_binary_c{c:g}"
            row, ph = fit_eval(xfit, y[fit], xheld, y[held], c)
            row.update({"C": c, "scaling": "none", "heldout_probabilities": ph.tolist()})
            fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
        for method in methods:
            rel, rel_meta = reliability_from_z(z, method)
            shuffled_rel, shuffled_rel_meta = reliability_from_z(z_shuffled, method)
            fold_out["reliability"][method] = rel_meta
            fold_out["reliability"][f"shuffled_{method}"] = shuffled_rel_meta
            for floor in floors:
                scales = floor + (1.0 - floor) * rel
                xs_fit = scale_columns(xfit, scales); xs_held = scale_columns(xheld, scales)
                for c in (0.01, 0.03):
                    key = f"soft_{method}_floor{floor:g}_binary_c{c:g}"
                    row, ph = fit_eval(xs_fit, y[fit], xs_held, y[held], c)
                    row.update({"C": c, "scaling": method, "floor": floor, "scale_mean": float(scales.mean()), "scale_min": float(scales.min()), "scale_max": float(scales.max()), "heldout_probabilities": ph.tolist()})
                    fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
                # One fixed capacity is enough for the negative control.
                shuffled_scales = floor + (1.0 - floor) * shuffled_rel
                key = f"control_shuffled_{method}_floor{floor:g}_binary_c0.01"
                row, ph = fit_eval(scale_columns(xfit, shuffled_scales), y[fit], scale_columns(xheld, shuffled_scales), y[held], 0.01)
                row.update({"C": 0.01, "scaling": f"shuffled_{method}", "floor": floor, "scale_mean": float(shuffled_scales.mean()), "heldout_probabilities": ph.tolist()})
                fold_out["results"][key] = row; all_results.setdefault(key, []).append(row)
        out["folds"].append(fold_out)

    out["aggregate"] = {key: summarize(rows) for key, rows in all_results.items() if len(rows) == 3}
    out["ranking"] = sorted(({"candidate": key, **value} for key, value in out["aggregate"].items()), key=lambda r: (r["mean_auc"], r["min_fold_auc"]), reverse=True)
    out["baseline_reference"] = {"candidate": "baseline_binary_c0.01", **out["aggregate"]["baseline_binary_c0.01"]}
    out["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": args.output, "top20": out["ranking"][:20], "elapsed_sec": out["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
