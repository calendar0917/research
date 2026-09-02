"""Build strict fold-only graph-level discriminative KSVD codes for MolHIV.

The first-level representation is a frozen, label-free node sparse-code cache.
For each molecule we pool the full distribution of normalized atom assignments
(mean, maximum, and standard deviation).  A second dictionary is then fitted on
whole-molecule descriptors from the inner-train fold, augmented with a small
one-hot graph-label block.  At encoding time only the descriptor block of the
learned dictionary is retained; held-out scaffold labels are never accessed.

This differs from earlier patch-level label augmentation: each supervision target
is attached to one complete molecule descriptor rather than copied to every node
or patch.  Official-valid/test molecules are neither described nor encoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.ksvd import _omp, ksvd


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def normalize_columns(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64).copy()
    norms = np.linalg.norm(D, axis=0)
    if np.any(norms <= 1e-10):
        raise RuntimeError("second-level dictionary contains a zero descriptor atom")
    return D / norms[None, :]


def graph_descriptors(
    node_codes: np.ndarray,
    offsets: np.ndarray,
    graph_indices: np.ndarray,
) -> np.ndarray:
    """Pool a permutation-invariant distribution of first-level assignments."""
    d = int(node_codes.shape[1])
    out = np.zeros((len(graph_indices), 3 * d), dtype=np.float64)
    for row, graph_i in enumerate(graph_indices.tolist()):
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        code = np.abs(np.asarray(node_codes[lo:hi], dtype=np.float64))
        denom = code.sum(axis=1, keepdims=True)
        q = code / np.maximum(denom, 1e-10)
        out[row, :d] = q.mean(axis=0)
        out[row, d : 2 * d] = q.max(axis=0)
        out[row, 2 * d :] = q.std(axis=0)
    return out


def fit_dictionary(
    Y_aug: np.ndarray,
    method: str,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if method == "ksvd":
        D, _, info = ksvd(
            Y_aug,
            n_atoms=n_atoms,
            T=sparsity,
            T_min=1,
            n_iter=iterations,
            seed=seed,
        )
        return D, {"method": method, "ksvd_info": info}
    if method == "pca":
        rank = min(n_atoms, min(Y_aug.shape))
        U, singular_values, _ = randomized_svd(
            Y_aug, n_components=rank, random_state=seed
        )
        D = U[:, :rank]
        if rank < n_atoms:
            raise ValueError("PCA rank is smaller than requested dictionary size")
        return D, {
            "method": method,
            "singular_values": singular_values[:rank].tolist(),
        }
    if method == "random_patch":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            Y_aug.shape[1], size=n_atoms, replace=Y_aug.shape[1] < n_atoms
        )
        D = normalize_columns(Y_aug[:, chosen])
        return D, {"method": method, "chosen_balanced_columns": chosen.tolist()}
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-token-cache", required=True)
    ap.add_argument("--local-family", choices=("ksvd", "pca", "random_patch"), default="ksvd")
    ap.add_argument("--upper-family", choices=("ksvd", "pca", "random_patch"), default="ksvd")
    ap.add_argument("--inner-split-cache", required=True)
    ap.add_argument("--inner-fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--n-atoms", type=int, default=16)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=5)
    ap.add_argument("--label-alpha", type=float, default=0.3)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--balance-seed", type=int, default=20260727)
    ap.add_argument("--descriptor-clip", type=float, default=5.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit-output", default=None)
    args = ap.parse_args()

    if args.n_atoms <= 0 or not 1 <= args.sparsity <= args.n_atoms:
        raise ValueError("invalid second-level dictionary size/sparsity")
    if args.ksvd_iter <= 0 or args.label_alpha < 0.0:
        raise ValueError("invalid K-SVD iterations or label alpha")
    if args.inner_fold < 0:
        raise ValueError("--inner-fold must be nonnegative")

    started = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=False,
    )
    labels = np.asarray(bundle.y, dtype=np.int64).reshape(-1)
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("MolHIV labels are not binary")

    with np.load(args.node_token_cache, allow_pickle=False) as base:
        offsets = np.asarray(base["offsets"], dtype=np.int64)
        original_indices = np.asarray(base["original_indices"], dtype=np.int64)
        official_train = np.asarray(base["train_indices"], dtype=np.int64)
        official_valid = np.asarray(base["valid_indices"], dtype=np.int64)
        official_test = np.asarray(base["test_indices"], dtype=np.int64)
        local_codes = np.asarray(base[f"tokens_{args.local_family}"], dtype=np.float64)

    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    if not np.array_equal(original_indices, expected_original):
        raise ValueError("node cache original indices do not match loaded dataset")
    if offsets.shape != (len(labels) + 1,) or local_codes.shape[0] != int(offsets[-1]):
        raise ValueError("invalid node-token cache offsets")

    with np.load(args.inner_split_cache, allow_pickle=False) as folds:
        inner_train = np.asarray(
            folds[f"fold_{args.inner_fold}_train_indices"], dtype=np.int64
        )
        inner_valid = np.asarray(
            folds[f"fold_{args.inner_fold}_valid_indices"], dtype=np.int64
        )
        cached_official_train = np.asarray(folds["official_train_indices"], dtype=np.int64)
    if not np.array_equal(official_train, cached_official_train):
        raise ValueError("inner-fold cache and node cache official train differ")
    if set(inner_train.tolist()) & set(inner_valid.tolist()):
        raise ValueError("inner train/valid overlap")
    if set(inner_train.tolist()) | set(inner_valid.tolist()) != set(official_train.tolist()):
        raise ValueError("inner fold does not partition official train")

    # Official-valid/test are deliberately omitted.  Their descriptor rows stay zero.
    descriptor_dim = 3 * local_codes.shape[1]
    descriptors = np.zeros((len(labels), descriptor_dim), dtype=np.float64)
    descriptors[official_train] = graph_descriptors(local_codes, offsets, official_train)

    fit_raw = descriptors[inner_train]
    fit_mean = fit_raw.mean(axis=0, keepdims=True)
    fit_scale = fit_raw.std(axis=0, keepdims=True)
    fit_scale = np.maximum(fit_scale, 1e-6)
    standardized = np.zeros_like(descriptors)
    standardized[official_train] = (descriptors[official_train] - fit_mean) / fit_scale
    if args.descriptor_clip > 0.0:
        standardized[official_train] = np.clip(
            standardized[official_train], -args.descriptor_clip, args.descriptor_clip
        )
    norms = np.linalg.norm(standardized[official_train], axis=1, keepdims=True)
    standardized[official_train] /= np.maximum(norms, 1e-8)

    fit_labels = labels[inner_train]
    positives = inner_train[fit_labels == 1]
    negatives = inner_train[fit_labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise RuntimeError("inner train lacks a class")
    rng = np.random.default_rng(args.balance_seed + args.inner_fold)
    selected_negatives = rng.choice(
        negatives, size=len(positives), replace=len(negatives) < len(positives)
    )
    balanced_indices = np.concatenate([positives, selected_negatives])
    rng.shuffle(balanced_indices)
    balanced_labels = labels[balanced_indices]

    Y_desc = standardized[balanced_indices].T
    one_hot = np.stack([1 - balanced_labels, balanced_labels], axis=0).astype(np.float64)
    Y_aug = np.vstack([Y_desc, args.label_alpha * one_hot])
    augmented_norms = np.linalg.norm(Y_aug, axis=0, keepdims=True)
    Y_aug /= np.maximum(augmented_norms, 1e-8)

    D_aug, fit_info = fit_dictionary(
        Y_aug,
        method=args.upper_family,
        n_atoms=args.n_atoms,
        sparsity=args.sparsity,
        iterations=args.ksvd_iter,
        seed=args.dict_seed,
    )
    D_descriptor = normalize_columns(D_aug[:descriptor_dim])

    graph_codes = np.zeros((len(labels), D_descriptor.shape[1]), dtype=np.float64)
    for graph_i in official_train.tolist():
        graph_codes[graph_i] = _omp(
            D_descriptor, standardized[graph_i], args.sparsity
        )

    # Graph-balanced fit-only RMS; repeated node rows then preserve this scaling.
    code_rms = np.sqrt(np.mean(np.square(graph_codes[inner_train]), axis=0))
    code_rms = np.maximum(code_rms, 1e-6)
    graph_codes[official_train] /= code_rms[None, :]

    node_graph_codes = np.zeros((int(offsets[-1]), D_descriptor.shape[1]), dtype=np.float32)
    for graph_i in official_train.tolist():
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        node_graph_codes[lo:hi] = graph_codes[graph_i].astype(np.float32)

    # Hard leakage checks.
    valid_nodes_nonzero = 0
    test_nodes_nonzero = 0
    for graph_i in official_valid.tolist():
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        valid_nodes_nonzero += int(np.count_nonzero(node_graph_codes[lo:hi]))
    for graph_i in official_test.tolist():
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        test_nodes_nonzero += int(np.count_nonzero(node_graph_codes[lo:hi]))
    if valid_nodes_nonzero or test_nodes_nonzero:
        raise RuntimeError("official-valid/test graph codes must stay zero")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=original_indices,
        train_indices=official_train,
        valid_indices=official_valid,
        test_indices=official_test,
        tokens_ksvd=node_graph_codes,
        hierarchical_dictionary=D_descriptor.astype(np.float32),
        descriptor_mean=fit_mean.astype(np.float32),
        descriptor_scale=fit_scale.astype(np.float32),
        graph_code_rms=code_rms.astype(np.float32),
        balanced_fit_indices=balanced_indices,
    )

    audit = {
        "protocol": "fold-only graph-level label-consistent hierarchical sparse dictionary",
        "node_token_cache": args.node_token_cache,
        "output": str(output),
        "local_family": args.local_family,
        "upper_family": args.upper_family,
        "inner_fold": args.inner_fold,
        "n_inner_train": int(len(inner_train)),
        "n_inner_valid": int(len(inner_valid)),
        "n_inner_train_positive": int(len(positives)),
        "n_balanced_dictionary_graphs": int(len(balanced_indices)),
        "descriptor_dim": int(descriptor_dim),
        "graph_code_dim": int(D_descriptor.shape[1]),
        "graph_code_sparsity_mean_inner_train": float(
            np.mean(np.count_nonzero(graph_codes[inner_train], axis=1))
        ),
        "label_alpha": args.label_alpha,
        "labels_used": True,
        "labels_used_indices_scope": "inner_train_only",
        "heldout_scaffold_labels_used": False,
        "official_valid_described_or_encoded": False,
        "official_test_described_or_encoded": False,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "explicit_ring_features": False,
        "fit_info": fit_info,
        "fingerprints": {
            "original_indices_sha256": array_hash(original_indices),
            "official_train_sha256": array_hash(official_train),
            "inner_train_sha256": array_hash(inner_train),
            "inner_valid_sha256": array_hash(inner_valid),
            "balanced_fit_indices_sha256": array_hash(balanced_indices),
        },
        "elapsed_sec": time.time() - started,
    }
    audit_output = Path(args.audit_output) if args.audit_output else output.with_suffix(".json")
    audit_output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
