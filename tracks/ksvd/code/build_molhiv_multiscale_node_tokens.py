"""Build matched radius-1 + radius-2 localized node-token caches for MolHIV.

The default configuration matches the established single-scale model in neural
input width and total OMP sparsity: r1 uses D=16/T=1 and r2 uses D=16/T=2,
then the two codes are concatenated into a 32-dimensional, at-most-3-sparse
node token.  Dictionaries and normalization inputs are official-train only;
official-test node patches are never vectorized or sparse-coded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .build_molhiv_node_tokens import (
    _csv,
    _random_patch_dictionary,
    _reservoir_train_patches,
)
from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd
from .molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def _learn_dictionary(
    family: str,
    Y: np.ndarray,
    *,
    n_atoms: int,
    sparsity: int,
    ksvd_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if family == "random_patch":
        return _random_patch_dictionary(Y, n_atoms, seed), {
            "matched_column_draw_seed": seed,
        }
    if family == "pca":
        U, _, _ = randomized_svd(
            Y,
            n_components=n_atoms,
            n_iter=5,
            random_state=seed,
        )
        return np.asarray(U, dtype=np.float64), {"randomized_svd_seed": seed}
    if family == "ksvd":
        D, _, info = ksvd(
            Y,
            n_atoms=n_atoms,
            T=sparsity,
            T_min=1,
            n_iter=ksvd_iter,
            seed=seed,
        )
        return D, info
    raise ValueError(f"unknown family {family!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000, help="0 means full dataset")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--radius-1", type=int, default=1)
    ap.add_argument("--radius-2", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms-1", type=int, default=16)
    ap.add_argument("--n-atoms-2", type=int, default=16)
    ap.add_argument("--sparsity-1", type=int, default=1)
    ap.add_argument("--sparsity-2", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--max-train-patches", type=int, default=6000)
    ap.add_argument("--families", default="ksvd")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.radius_1 >= args.radius_2:
        raise ValueError("--radius-1 must be smaller than --radius-2")
    if args.n_atoms_1 <= 0 or args.n_atoms_2 <= 0:
        raise ValueError("dictionary sizes must be positive")
    if not 1 <= args.sparsity_1 <= args.n_atoms_1:
        raise ValueError("invalid --sparsity-1")
    if not 1 <= args.sparsity_2 <= args.n_atoms_2:
        raise ValueError("invalid --sparsity-2")

    t0 = time.time()
    families = _csv(args.families)
    unknown = set(families) - {"random_patch", "pca", "ksvd"}
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")

    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)

    # The reservoir algorithm depends only on stream position and RNG draws.
    # Using identical seed/capacity at both radii therefore selects matched
    # train-node identities while retaining radius-specific feature vectors.
    Y1, stats1 = _reservoir_train_patches(
        graphs,
        tr,
        bundle.node_feats,
        bundle.edge_feats,
        radius=args.radius_1,
        max_nodes=args.max_nodes,
        capacity=args.max_train_patches,
        seed=args.dict_seed,
    )
    Y2, stats2 = _reservoir_train_patches(
        graphs,
        tr,
        bundle.node_feats,
        bundle.edge_feats,
        radius=args.radius_2,
        max_nodes=args.max_nodes,
        capacity=args.max_train_patches,
        seed=args.dict_seed,
    )
    if stats1["n_train_node_patches_raw"] != stats2["n_train_node_patches_raw"]:
        raise AssertionError("scale reservoirs observed different train-node counts")
    print(
        f"matched train patch pools r{args.radius_1}={Y1.shape} "
        f"r{args.radius_2}={Y2.shape}; raw={stats1['n_train_node_patches_raw']}",
        flush=True,
    )

    dictionaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dictionary_info: dict[str, Any] = {}
    for family in families:
        D1, info1 = _learn_dictionary(
            family,
            Y1,
            n_atoms=args.n_atoms_1,
            sparsity=args.sparsity_1,
            ksvd_iter=args.ksvd_iter,
            seed=args.dict_seed,
        )
        D2, info2 = _learn_dictionary(
            family,
            Y2,
            n_atoms=args.n_atoms_2,
            sparsity=args.sparsity_2,
            ksvd_iter=args.ksvd_iter,
            seed=args.dict_seed,
        )
        dictionaries[family] = (D1, D2)
        dictionary_info[family] = {
            f"radius_{args.radius_1}": info1,
            f"radius_{args.radius_2}": info2,
        }
    print(
        "learned dictionaries "
        + ", ".join(
            f"{family}:r{args.radius_1}{D1.shape}+r{args.radius_2}{D2.shape}"
            for family, (D1, D2) in dictionaries.items()
        ),
        flush=True,
    )

    offsets = graph_node_offsets(graphs)
    total_nodes = int(offsets[-1])
    tokens = {
        family: np.zeros(
            (total_nodes, D1.shape[1] + D2.shape[1]), dtype=np.float32
        )
        for family, (D1, D2) in dictionaries.items()
    }
    encode_indices = np.concatenate([tr, va])
    encoded_nodes = 0
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        g = graphs[i]
        for u in g.nodes:
            vectors = []
            for radius in (args.radius_1, args.radius_2):
                y = np.asarray(
                    centered_ego_vector(
                        g,
                        int(u),
                        bundle.node_feats[i],
                        bundle.edge_feats[i],
                        radius=radius,
                        max_nodes=args.max_nodes,
                    ),
                    dtype=np.float64,
                )
                y /= max(float(np.linalg.norm(y)), 1e-12)
                vectors.append(y)
            row = int(offsets[i] + u)
            for family, (D1, D2) in dictionaries.items():
                z1 = _omp(D1, vectors[0], args.sparsity_1)
                z2 = _omp(D2, vectors[1], args.sparsity_2)
                tokens[family][row] = np.concatenate([z1, z2]).astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(encode_indices):
            print(f"encoded graphs {count}/{len(encode_indices)}; nodes={encoded_nodes}", flush=True)

    # Executable isolation invariant: official-test rows must remain untouched.
    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        for family in families:
            if np.any(tokens[family][lo:hi] != 0):
                raise AssertionError(f"{family} encoded official-test nodes")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        "train_indices": tr,
        "valid_indices": va,
        "test_indices": te,
        "scale_slices": np.asarray(
            [0, args.n_atoms_1, args.n_atoms_1 + args.n_atoms_2], dtype=np.int64
        ),
        "scale_radii": np.asarray([args.radius_1, args.radius_2], dtype=np.int64),
        "scale_sparsities": np.asarray(
            [args.sparsity_1, args.sparsity_2], dtype=np.int64
        ),
    }
    for family, (D1, D2) in dictionaries.items():
        key = f"{family}_multiscale"
        archive[f"dictionary_{family}_r{args.radius_1}"] = D1.astype(np.float32)
        archive[f"dictionary_{family}_r{args.radius_2}"] = D2.astype(np.float32)
        archive[f"tokens_{key}"] = tokens[family]
    np.savez_compressed(output, **archive)

    meta = {
        "protocol_id": "molhiv-localized-multiscale-node-tokens-v1",
        "test_policy": "official test node patches were not vectorized or sparse-coded",
        "matching": {
            "total_token_dim": args.n_atoms_1 + args.n_atoms_2,
            "total_max_sparsity": args.sparsity_1 + args.sparsity_2,
            "reference": "single radius-2 D=32/T=3",
            "matched_reservoir_node_identities": True,
        },
        "config": vars(args),
        "data_meta": bundle.meta,
        "patch_stats": {
            f"radius_{args.radius_1}": stats1,
            f"radius_{args.radius_2}": stats2,
        },
        "dictionary_info": dictionary_info,
        "families": [f"{family}_multiscale" for family in families],
        "total_nodes": total_nodes,
        "encoded_train_valid_nodes": encoded_nodes,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output} and {meta_path}", flush=True)


if __name__ == "__main__":
    main()
