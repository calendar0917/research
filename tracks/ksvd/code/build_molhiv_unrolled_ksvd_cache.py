"""Build a strict selection-phase cache for KSVD-initialized unrolling.

The cache keeps the frozen OMP codes and dictionary from an existing node-token
archive, then computes only the sufficient statistic ``y @ D`` for normalized
radius-r patches.  By default patches are vectorized for official-train graphs
only; official-valid and official-test rows remain exactly zero.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ogb.utils.features import get_atom_feature_dims

from .data_molhiv import load_molhiv
from .molhiv_node_tokens import centered_ego_vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument(
        "--patch-view", choices=("full", "no_ring"), default="full",
        help="patch coordinates used by the base dictionary",
    )
    ap.add_argument(
        "--include-official-valid",
        action="store_true",
        help="post-freeze/full-train use only; default selection cache encodes official train only",
    )
    args = ap.parse_args()

    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    bundle = load_molhiv(
        root=root,
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    graphs = bundle.graphs
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)

    base_path = Path(args.base_cache)
    with np.load(base_path, allow_pickle=False) as base:
        required = {
            "offsets", "original_indices", "train_indices", "valid_indices",
            "test_indices", "dictionary_ksvd", "tokens_ksvd",
        }
        missing = required.difference(base.files)
        if missing:
            raise KeyError(f"base cache is missing {sorted(missing)}")
        offsets = np.asarray(base["offsets"], dtype=np.int64)
        original_indices = np.asarray(base["original_indices"], dtype=np.int64)
        train_indices = np.asarray(base["train_indices"], dtype=np.int64)
        valid_indices = np.asarray(base["valid_indices"], dtype=np.int64)
        test_indices = np.asarray(base["test_indices"], dtype=np.int64)
        dictionary = np.asarray(base["dictionary_ksvd"], dtype=np.float32)
        tokens = np.asarray(base["tokens_ksvd"], dtype=np.float32)

    if not np.array_equal(original_indices, expected_original):
        raise ValueError("base cache original_indices do not match loaded MolHIV subset")
    if offsets.shape != (len(graphs) + 1,):
        raise ValueError(f"invalid offsets shape {offsets.shape}")
    if tokens.shape != (int(offsets[-1]), dictionary.shape[1]):
        raise ValueError("token/dictionary/offset dimensions are inconsistent")

    # Normalize defensively. Existing KSVD dictionaries already have unit columns,
    # but recording the exact normalized D keeps the sufficient statistic and Gram
    # matrix mathematically consistent in the runner.
    D = dictionary.astype(np.float64)
    D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
    correlations = np.zeros((tokens.shape[0], D.shape[1]), dtype=np.float32)
    encode_indices = (
        np.concatenate([train_indices, valid_indices])
        if args.include_official_valid
        else train_indices
    )

    encoded_nodes = 0
    nnz_sum = 0
    patch_dim = None
    patch_view_indices = None
    if args.patch_view == "no_ring":
        atom_dims = get_atom_feature_dims()
        wl_topology_dim = (
            args.max_nodes
            + args.max_nodes * (args.max_nodes + 1) // 2
            + 64 * 3 + 3
        )
        labeled_base_dim = wl_topology_dim + 64 + 16 + 64 * 3
        explicit_ring_dim = 13
        center_start = labeled_base_dim + explicit_ring_dim
        full_probe = centered_ego_vector(
            graphs[int(encode_indices[0])], 0,
            bundle.node_feats[int(encode_indices[0])],
            bundle.edge_feats[int(encode_indices[0])],
            radius=args.radius, max_nodes=args.max_nodes,
        )
        patch_view_indices = np.concatenate([
            np.arange(labeled_base_dim, dtype=np.int64),
            np.arange(center_start, len(full_probe), dtype=np.int64),
        ])
        if center_start + sum(atom_dims) > len(full_probe):
            raise RuntimeError("unexpected centered patch layout")
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        g = graphs[i]
        for u in g.nodes:
            y = np.asarray(
                centered_ego_vector(
                    g,
                    int(u),
                    bundle.node_feats[i],
                    bundle.edge_feats[i],
                    radius=args.radius,
                    max_nodes=args.max_nodes,
                ),
                dtype=np.float64,
            )
            if patch_view_indices is not None:
                y = y[patch_view_indices]
            if y.size != D.shape[0]:
                raise ValueError(f"patch dim {y.size} does not match dictionary dim {D.shape[0]}")
            patch_dim = int(y.size)
            nnz_sum += int(np.count_nonzero(y))
            y /= max(float(np.linalg.norm(y)), 1e-12)
            correlations[int(offsets[i] + int(u))] = (y @ D).astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(encode_indices):
            print(
                f"correlated graphs {count}/{len(encode_indices)}; nodes={encoded_nodes}",
                flush=True,
            )

    # Executable isolation invariants. A selection cache must not vectorize either
    # held-out official split; even a post-freeze valid cache never touches test.
    untouched = test_indices
    if not args.include_official_valid:
        untouched = np.concatenate([valid_indices, test_indices])
    for raw_i in untouched:
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if np.any(correlations[lo:hi] != 0):
            raise AssertionError(f"held-out graph {i} has nonzero patch correlations")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=original_indices,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        dictionary_ksvd=D.astype(np.float32),
        tokens_ksvd=tokens,
        correlations_ksvd=correlations,
    )
    gram = D.T @ D
    meta = {
        "protocol_id": "molhiv-ksvd-unrolled-sufficient-statistic-cache-v1",
        "base_cache": str(base_path),
        "output": str(output),
        "config": vars(args),
        "patch_policy": (
            "official train and valid patches vectorized after configuration freeze; official test untouched"
            if args.include_official_valid
            else "selection-only: official train patches vectorized; official valid and test untouched"
        ),
        "n_graphs": len(graphs),
        "encoded_graphs": int(len(encode_indices)),
        "encoded_nodes": int(encoded_nodes),
        "total_nodes": int(tokens.shape[0]),
        "patch_dim": patch_dim,
        "mean_patch_nnz": float(nnz_sum / max(encoded_nodes, 1)),
        "mean_patch_density": float(nnz_sum / max(encoded_nodes * int(patch_dim or 1), 1)),
        "dictionary_shape": list(D.shape),
        "dictionary_column_norm_min": float(np.linalg.norm(D, axis=0).min()),
        "dictionary_column_norm_max": float(np.linalg.norm(D, axis=0).max()),
        "gram_lipschitz": float(np.linalg.eigvalsh(gram).max()),
        "elapsed_sec": time.time() - t0,
        "official_valid_patch_evaluations": int(args.include_official_valid),
        "official_test_patch_evaluations": 0,
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output} and {meta_path}; elapsed={meta['elapsed_sec']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
