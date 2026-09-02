"""Build graph-label weakly supervised positive-novelty KSVD controls for MolHIV.

For each strict inner fold, a background dictionary is fitted only on patches from
negative bags.  High background-reconstruction-error patches are then selected
within each positive bag (a fixed top-k per molecule), and a second dictionary is
fitted on their normalized residuals.  Labels are therefore used only at graph/bag
level; they are never copied onto every patch.  KSVD, PCA, and random-patch
pipelines are matched in dimensions, sparsity, patch pools, and witness count.
Official-valid/test graphs are neither vectorized nor encoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ogb.utils.features import get_atom_feature_dims
from sklearn.utils.extmath import randomized_svd

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.build_molhiv_node_tokens import _random_patch_dictionary
from code.data_molhiv import load_molhiv
from code.ksvd import _omp, ksvd
from code.molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()




def maxmin_column_dictionary(
    Y: np.ndarray, n_atoms: int, seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Select real patch columns by seeded max-min absolute cosine diversity."""
    if Y.ndim != 2 or Y.shape[1] < n_atoms:
        raise ValueError("dictionary pool has fewer columns than atoms")
    norms = np.linalg.norm(Y, axis=0)
    if np.any(norms <= 1e-12):
        raise ValueError("max-min initialization received a zero-norm patch")
    normalized = Y / norms[None, :]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(Y.shape[1]))
    selected = [first]
    max_absolute_correlation = np.abs(normalized[:, first].T @ normalized)
    max_absolute_correlation[first] = np.inf
    while len(selected) < n_atoms:
        atom = int(np.argmin(max_absolute_correlation))
        selected.append(atom)
        max_absolute_correlation = np.maximum(
            max_absolute_correlation,
            np.abs(normalized[:, atom].T @ normalized),
        )
        max_absolute_correlation[selected] = np.inf
    return Y[:, selected].astype(np.float64).copy(), selected

def fit_dictionary(
    Y: np.ndarray, family: str, n_atoms: int, sparsity: int,
    iterations: int, seed: int, *,
    ksvd_initialization: str = "random_columns",
    ksvd_coherence_step: float = 0.0,
    ksvd_anchor_strength: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if Y.shape[1] < n_atoms:
        raise ValueError("dictionary pool has fewer columns than atoms")
    if family == "ksvd":
        if ksvd_initialization not in {
            "random_columns", "deterministic_svd", "maxmin_columns"
        }:
            raise ValueError(f"unknown KSVD initialization: {ksvd_initialization}")
        initial_dictionary = None
        initialization_info: dict[str, Any] = {
            "kind": ksvd_initialization,
        }
        if ksvd_initialization == "maxmin_columns":
            initial_dictionary, selected_indices = maxmin_column_dictionary(
                Y, n_atoms, seed
            )
            initialization_info["selected_column_indices"] = selected_indices
            initialization_info["selected_dictionary_sha256"] = array_hash(
                initial_dictionary.astype(np.float32)
            )
            initialization_info["distance"] = "one_minus_absolute_cosine"
            initialization_info["first_column_rule"] = "seeded_uniform_draw"
        elif ksvd_initialization == "deterministic_svd":
            U, singular_values, _ = np.linalg.svd(Y, full_matrices=False)
            initial_dictionary = np.asarray(U[:, :n_atoms], dtype=np.float64).copy()
            # Canonicalize SVD signs so the initializer is byte-stable rather
            # than relying on a backend's arbitrary singular-vector signs.
            for atom in range(initial_dictionary.shape[1]):
                pivot = int(np.argmax(np.abs(initial_dictionary[:, atom])))
                if initial_dictionary[pivot, atom] < 0.0:
                    initial_dictionary[:, atom] *= -1.0
            initialization_info["singular_values"] = (
                singular_values[:n_atoms].astype(float).tolist()
            )
            initialization_info["sign_rule"] = "largest_absolute_loading_positive"
        D, _, info = ksvd(
            Y, n_atoms=n_atoms, T=sparsity, T_min=1,
            n_iter=iterations, seed=seed,
            initial_dictionary=initial_dictionary,
            coherence_step=ksvd_coherence_step,
            anchor_strength=ksvd_anchor_strength,
        )
        return D, {
            "family": family,
            "initialization": initialization_info,
            "ksvd": info,
        }
    if family == "pca":
        U, singular_values, _ = randomized_svd(
            Y, n_components=n_atoms, n_iter=5, random_state=seed
        )
        return np.asarray(U, dtype=np.float64), {
            "family": family,
            "singular_values": singular_values.tolist(),
            "randomized_svd_seed": int(seed),
        }
    if family == "random_patch":
        return _random_patch_dictionary(Y, n_atoms, seed), {
            "family": family, "matched_column_draw_seed": int(seed)
        }
    raise ValueError(family)


def summarize_graph(
    background_error2: np.ndarray,
    joint_error2: np.ndarray,
    novelty_codes: np.ndarray,
) -> np.ndarray:
    gain = np.maximum(background_error2 - joint_error2, 0.0)
    relative_gain = gain / np.maximum(background_error2, 1e-8)
    novelty_energy = np.sqrt(np.square(novelty_codes).sum(axis=1))

    def six(values: np.ndarray) -> np.ndarray:
        return np.asarray([
            values.mean(), values.std(),
            *np.quantile(values, [0.50, 0.75, 0.90]), values.max(),
        ], dtype=np.float64)

    scalar = np.concatenate([
        six(np.sqrt(np.maximum(background_error2, 0.0))),
        six(np.sqrt(np.maximum(joint_error2, 0.0))),
        six(gain), six(relative_gain), six(novelty_energy),
    ])
    absolute = np.abs(novelty_codes)
    n_nodes, n_atoms = absolute.shape
    top_count = min(3, n_nodes)
    top3 = np.partition(
        absolute, n_nodes - top_count, axis=0
    )[n_nodes - top_count:].mean(axis=0)
    atom = np.concatenate([
        absolute.mean(axis=0), absolute.max(axis=0), top3,
        (absolute > 1e-10).mean(axis=0),
    ])
    return np.concatenate([
        scalar, atom,
        np.asarray([float(n_nodes), np.log1p(n_nodes)], dtype=np.float64),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--inner-split-cache", required=True)
    ap.add_argument("--inner-fold", type=int, required=True)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--background-patches", type=int, default=6000)
    ap.add_argument("--background-patches-per-graph", type=int, default=2)
    ap.add_argument("--positive-witnesses-per-graph", type=int, default=2)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--families", default="ksvd,pca,random_patch")
    ap.add_argument(
        "--shared-background-family",
        choices=("none", "ksvd", "pca", "random_patch"),
        default="none",
        help=(
            "optionally use one matched background dictionary and identical "
            "positive witness residual pool for all novelty-dictionary controls"
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.n_atoms <= 0 or not 1 <= args.sparsity <= args.n_atoms:
        raise ValueError("invalid atom count or sparsity")
    if args.background_patches <= 0 or args.background_patches_per_graph <= 0:
        raise ValueError("background patch budgets must be positive")
    if args.positive_witnesses_per_graph <= 0:
        raise ValueError("positive witness count must be positive")
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    if set(families) != {"ksvd", "pca", "random_patch"}:
        raise ValueError("this matched experiment requires ksvd,pca,random_patch")

    started = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed, with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    graphs, labels = bundle.graphs, bundle.y
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)
    with np.load(args.inner_split_cache, allow_pickle=False) as folds:
        inner_train = np.asarray(
            folds[f"fold_{args.inner_fold}_train_indices"], dtype=np.int64
        )
        inner_valid = np.asarray(
            folds[f"fold_{args.inner_fold}_valid_indices"], dtype=np.int64
        )
        cached_train = np.asarray(folds["official_train_indices"], dtype=np.int64)
    if not np.array_equal(cached_train, official_train):
        raise ValueError("fold cache official train mismatch")
    if set(inner_train.tolist()) & set(inner_valid.tolist()):
        raise ValueError("inner fold overlap")
    if set(inner_train.tolist()) | set(inner_valid.tolist()) != set(official_train.tolist()):
        raise ValueError("inner fold does not partition official train")

    negatives = inner_train[labels[inner_train] == 0]
    positives = inner_train[labels[inner_train] == 1]
    if len(negatives) == 0 or len(positives) == 0:
        raise RuntimeError("inner train lacks a class")

    wl_topology_dim = (
        args.max_nodes
        + args.max_nodes * (args.max_nodes + 1) // 2
        + 64 * 3 + 3
    )
    labeled_base_dim = wl_topology_dim + 64 + 16 + 64 * 3
    explicit_ring_dim = 13
    center_start = labeled_base_dim + explicit_ring_dim
    center_end = center_start + sum(get_atom_feature_dims())
    patch_view_indices: np.ndarray | None = None

    def patch(graph_i: int, node_i: int) -> np.ndarray:
        nonlocal patch_view_indices
        vec = centered_ego_vector(
            graphs[graph_i], node_i,
            bundle.node_feats[graph_i], bundle.edge_feats[graph_i],
            radius=args.radius, max_nodes=args.max_nodes,
        )
        y = np.asarray(vec, dtype=np.float64)
        y /= max(float(np.linalg.norm(y)), 1e-12)
        if patch_view_indices is None:
            patch_view_indices = np.concatenate([
                np.arange(labeled_base_dim, dtype=np.int64),
                np.arange(center_start, y.size, dtype=np.int64),
            ])
            if center_end > y.size:
                raise RuntimeError("unexpected centered patch layout")
        y = y[patch_view_indices]
        return y / max(float(np.linalg.norm(y)), 1e-12)

    # Graph-balanced negative background pool: a fixed number of candidate
    # centers per negative molecule, followed by a matched global subsample.
    center_rng = np.random.default_rng(args.dict_seed + 104729)
    candidates: list[np.ndarray] = []
    candidate_graphs: list[int] = []
    for graph_i in negatives.tolist():
        n_nodes = graphs[graph_i].n
        take = min(args.background_patches_per_graph, n_nodes)
        chosen = center_rng.choice(n_nodes, size=take, replace=False)
        for node_i in chosen.tolist():
            candidates.append(patch(graph_i, int(node_i)))
            candidate_graphs.append(graph_i)
    if len(candidates) < args.n_atoms:
        raise RuntimeError("too few negative background candidates")
    candidate_matrix = np.stack(candidates, axis=1)
    reservoir_rng = np.random.default_rng(args.dict_seed + 130363)
    n_background = min(args.background_patches, candidate_matrix.shape[1])
    chosen_background = reservoir_rng.choice(
        candidate_matrix.shape[1], size=n_background, replace=False
    )
    background_pool = candidate_matrix[:, chosen_background]
    print(
        f"negative background pool={background_pool.shape}; "
        f"graphs={len(negatives)} positives={len(positives)}",
        flush=True,
    )

    background: dict[str, np.ndarray] = {}
    background_info: dict[str, Any] = {}
    # Matched controls use the same numeric seed.  Family-dependent seed
    # offsets would confound the dictionary family with a different random
    # realization (notably randomized PCA and random-patch sampling).
    background_seed = args.dict_seed
    for family in families:
        D, info = fit_dictionary(
            background_pool, family, args.n_atoms, args.sparsity,
            args.ksvd_iter, background_seed,
        )
        background[family] = D
        background_info[family] = info
    if args.shared_background_family != "none":
        shared_family = args.shared_background_family
        shared_dictionary = background[shared_family].copy()
        shared_info = background_info[shared_family]
        background = {family: shared_dictionary.copy() for family in families}
        background_info = {
            family: {
                "family": family,
                "mode": "shared_background_control",
                "source_family": shared_family,
                "source_fit_info": shared_info,
            }
            for family in families
        }
        print(
            f"using shared {shared_family} background for all novelty controls",
            flush=True,
        )

    # Family-specific positive witnesses are the top background residuals in
    # each positive bag.  The per-bag cap prevents large positive molecules from
    # receiving more weak supervision merely because they contain more atoms.
    novelty_pools: dict[str, list[np.ndarray]] = {f: [] for f in families}
    witness_records: dict[str, list[dict[str, Any]]] = {f: [] for f in families}
    positive_patch_cache: dict[int, np.ndarray] = {}
    for count, graph_i in enumerate(positives.tolist(), 1):
        Yg = np.stack(
            [patch(graph_i, node_i) for node_i in range(graphs[graph_i].n)],
            axis=1,
        )
        positive_patch_cache[graph_i] = Yg
        for family in families:
            D = background[family]
            X = np.stack([
                _omp(D, Yg[:, j], args.sparsity)
                for j in range(Yg.shape[1])
            ], axis=1)
            residual = Yg - D @ X
            error2 = np.square(residual).sum(axis=0)
            take = min(args.positive_witnesses_per_graph, Yg.shape[1])
            chosen = np.argsort(-error2)[:take]
            for node_i in chosen.tolist():
                r = residual[:, node_i]
                norm = float(np.linalg.norm(r))
                if norm > 1e-10:
                    novelty_pools[family].append(r / norm)
                    witness_records[family].append({
                        "graph": int(graph_i), "node": int(node_i),
                        "background_error": float(np.sqrt(error2[node_i])),
                    })
        if count % 50 == 0 or count == len(positives):
            print(f"selected positive witnesses {count}/{len(positives)}", flush=True)

    novelty: dict[str, np.ndarray] = {}
    novelty_info: dict[str, Any] = {}
    novelty_seed = args.dict_seed + 1000003
    for family in families:
        Ynovel = np.stack(novelty_pools[family], axis=1)
        D, info = fit_dictionary(
            Ynovel, family, args.n_atoms, args.sparsity,
            args.ksvd_iter, novelty_seed,
        )
        novelty[family] = D
        novelty_info[family] = {
            **info,
            "witness_count": int(Ynovel.shape[1]),
            "witness_error_mean": float(np.mean([
                row["background_error"] for row in witness_records[family]
            ])),
            "witness_error_min": float(np.min([
                row["background_error"] for row in witness_records[family]
            ])),
            "witness_error_max": float(np.max([
                row["background_error"] for row in witness_records[family]
            ])),
        }
        print(
            f"learned {family} novelty dictionary from {Ynovel.shape[1]} witnesses",
            flush=True,
        )

    offsets = graph_node_offsets(graphs)
    total_nodes = int(offsets[-1])
    tokens = {
        f: np.zeros((total_nodes, args.n_atoms), dtype=np.float32)
        for f in families
    }
    feature_dim = 30 + 4 * args.n_atoms + 2
    graph_features = {
        f: np.zeros((len(graphs), feature_dim), dtype=np.float32)
        for f in families
    }
    for count, graph_i in enumerate(official_train.tolist(), 1):
        if graph_i in positive_patch_cache:
            Yg = positive_patch_cache[graph_i]
        else:
            Yg = np.stack(
                [patch(graph_i, node_i) for node_i in range(graphs[graph_i].n)],
                axis=1,
            )
        lo = int(offsets[graph_i])
        hi = int(offsets[graph_i + 1])
        for family in families:
            Dbg = background[family]
            Dnov = novelty[family]
            Xbg = np.stack([
                _omp(Dbg, Yg[:, j], args.sparsity)
                for j in range(Yg.shape[1])
            ], axis=1)
            residual = Yg - Dbg @ Xbg
            Z = np.stack([
                _omp(Dnov, residual[:, j], args.sparsity)
                for j in range(Yg.shape[1])
            ], axis=1)
            remaining = residual - Dnov @ Z
            background_error2 = np.square(residual).sum(axis=0)
            joint_error2 = np.square(remaining).sum(axis=0)
            tokens[family][lo:hi] = Z.T.astype(np.float32)
            graph_features[family][graph_i] = summarize_graph(
                background_error2, joint_error2, Z.T
            ).astype(np.float32)
        if count % 500 == 0 or count == len(official_train):
            print(f"encoded official-train graphs {count}/{len(official_train)}", flush=True)

    # Executable split isolation: no official-valid/test vectorization or coding.
    for graph_i in np.concatenate([official_valid, official_test]).tolist():
        lo = int(offsets[graph_i])
        hi = int(offsets[graph_i + 1])
        for family in families:
            if np.any(tokens[family][lo:hi] != 0):
                raise RuntimeError("official-valid/test node tokens are nonzero")
            if np.any(graph_features[family][graph_i] != 0):
                raise RuntimeError("official-valid/test graph features are nonzero")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        "train_indices": official_train,
        "valid_indices": official_valid,
        "test_indices": official_test,
    }
    for family in families:
        archive[f"tokens_{family}"] = tokens[family]
        archive[f"dictionary_{family}"] = novelty[family].astype(np.float32)
        archive[f"background_dictionary_{family}"] = background[family].astype(np.float32)
        archive[f"graph_features_{family}"] = graph_features[family]
    np.savez_compressed(output, **archive)

    meta = {
        "protocol_id": "molhiv-graph-label-mil-positive-residual-dictionary-v1",
        "test_policy": "official-valid/test patches were not vectorized or encoded",
        "config": vars(args),
        "families": families,
        "patch_view": {
            "name": "no_ring",
            "explicit_ring_features": False,
            "atom_or_bond_label_features": True,
            "radius": int(args.radius),
            "max_nodes": int(args.max_nodes),
            "feature_dim": int(background_pool.shape[0]),
        },
        "inner_train_count": int(len(inner_train)),
        "inner_valid_count": int(len(inner_valid)),
        "inner_train_positive_count": int(len(positives)),
        "inner_train_negative_count": int(len(negatives)),
        "negative_background_candidate_count": int(candidate_matrix.shape[1]),
        "negative_background_patch_count": int(background_pool.shape[1]),
        "background_dictionary_info": background_info,
        "novelty_dictionary_info": novelty_info,
        "graph_feature_dim": int(feature_dim),
        "encoded_graph_count": int(len(official_train)),
        "official_valid_encoded": False,
        "official_test_encoded": False,
        "official_train_sha256": array_hash(official_train),
        "inner_train_sha256": array_hash(inner_train),
        "inner_valid_sha256": array_hash(inner_valid),
        "elapsed_sec": float(time.time() - started),
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
