"""Representation-level diagnostics for fold-only SSL latent dictionaries.

The cached inputs were unit-normalized before OMP. Because OMP refits active
coefficients by least squares, ||y - D x||^2 = 1 - ||D x||^2 (up to numerical
roundoff). This lets us audit reconstruction without storing the private SSL
latents again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows_for_graphs(offsets: np.ndarray, graph_indices: np.ndarray) -> np.ndarray:
    parts = [
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in graph_indices
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _distribution(top1: np.ndarray, n_atoms: int) -> np.ndarray:
    counts = np.bincount(top1, minlength=n_atoms).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-15))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _quantiles(x: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    vals = np.quantile(x, qs)
    return {f"{q:g}": float(v) for q, v in zip(qs, vals)}


def _split_stats(
    D: np.ndarray,
    X: np.ndarray,
    rows: np.ndarray,
    offsets: np.ndarray,
    graph_indices: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    codes = X[rows].astype(np.float64)
    abs_codes = np.abs(codes)
    reconstruction = codes @ D.T
    explained = np.einsum("ij,ij->i", reconstruction, reconstruction)
    residual = np.maximum(0.0, 1.0 - explained)
    nnz = (abs_codes > 1e-10).sum(axis=1)
    l1 = abs_codes.sum(axis=1)
    top1 = abs_codes.argmax(axis=1)
    dominance = abs_codes.max(axis=1) / np.maximum(l1, 1e-12)
    p = _distribution(top1, D.shape[1])

    graph_unique_fraction = []
    graph_top1_entropy = []
    cursor = 0
    for i in graph_indices:
        n = int(offsets[int(i) + 1] - offsets[int(i)])
        graph_atoms = top1[cursor:cursor + n]
        cursor += n
        gp = _distribution(graph_atoms, D.shape[1])
        graph_unique_fraction.append(float(np.count_nonzero(gp) / D.shape[1]))
        graph_top1_entropy.append(_entropy(gp) / max(np.log(D.shape[1]), 1e-12))
    if cursor != len(top1):
        raise AssertionError("graph/node row mismatch")

    entropy = _entropy(p)
    order = np.sort(p)[::-1]
    stats: dict[str, Any] = {
        "n_graphs": int(len(graph_indices)),
        "n_nodes": int(len(rows)),
        "reconstruction_mse_mean": float(residual.mean()),
        "reconstruction_mse_quantiles": _quantiles(residual),
        "explained_energy_mean": float(explained.mean()),
        "nnz_mean": float(nnz.mean()),
        "coefficient_l1_mean": float(l1.mean()),
        "coefficient_dominance_mean": float(dominance.mean()),
        "coefficient_dominance_quantiles": _quantiles(dominance),
        "top1_entropy": entropy,
        "top1_entropy_normalized": float(entropy / max(np.log(D.shape[1]), 1e-12)),
        "top1_effective_atoms": float(np.exp(entropy)),
        "top1_max_atom_share": float(order[0]),
        "top1_top4_share": float(order[:4].sum()),
        "active_top1_atoms": int(np.count_nonzero(p)),
        "graph_unique_atom_fraction_mean": float(np.mean(graph_unique_fraction)),
        "graph_top1_entropy_normalized_mean": float(np.mean(graph_top1_entropy)),
    }
    return stats, p, residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with np.load(args.token_cache, allow_pickle=False) as source:
        arrays = {k: np.asarray(source[k]) for k in source.files}
    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)

    offsets = arrays["offsets"]
    fit_rows = _rows_for_graphs(offsets, fit_indices)
    heldout_rows = _rows_for_graphs(offsets, heldout_indices)
    if set(fit_indices.tolist()) & set(heldout_indices.tolist()):
        raise AssertionError("fit/heldout graph overlap")

    families = sorted(k.removeprefix("dictionary_") for k in arrays if k.startswith("dictionary_"))
    result: dict[str, Any] = {
        "protocol_id": "molhiv-fold-only-ssl-latent-dictionary-diagnostic-v1",
        "labels_used": False,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "config": vars(args),
        "fit_heldout_overlap": 0,
        "families": {},
    }
    residuals: dict[str, dict[str, np.ndarray]] = {}
    for family in families:
        D = arrays[f"dictionary_{family}"].astype(np.float64)
        X = arrays[f"tokens_{family}"].astype(np.float64)
        fit_stats, fit_p, fit_residual = _split_stats(
            D, X, fit_rows, offsets, fit_indices
        )
        held_stats, held_p, held_residual = _split_stats(
            D, X, heldout_rows, offsets, heldout_indices
        )
        result["families"][family] = {
            "fit": fit_stats,
            "heldout": held_stats,
            "heldout_minus_fit_reconstruction_mse": float(
                held_stats["reconstruction_mse_mean"] - fit_stats["reconstruction_mse_mean"]
            ),
            "fit_heldout_top1_js_divergence": _js_divergence(fit_p, held_p),
        }
        residuals[family] = {"fit": fit_residual, "heldout": held_residual}

    comparisons: dict[str, Any] = {}
    if "ksvd" in residuals:
        for other in families:
            if other == "ksvd":
                continue
            comparisons[f"ksvd_vs_{other}"] = {}
            for split in ["fit", "heldout"]:
                delta = residuals[other][split] - residuals["ksvd"][split]
                comparisons[f"ksvd_vs_{other}"][split] = {
                    "mean_mse_advantage": float(delta.mean()),
                    "fraction_nodes_ksvd_lower_mse": float((delta > 1e-10).mean()),
                    "fraction_nodes_tied": float((np.abs(delta) <= 1e-10).mean()),
                    "advantage_quantiles": _quantiles(delta),
                }
    result["paired_reconstruction_comparisons"] = comparisons

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
