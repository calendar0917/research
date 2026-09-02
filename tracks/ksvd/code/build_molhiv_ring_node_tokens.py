"""Append explicit ring-cell K-SVD codes to the best radius-2 node tokens.

Only official-train rings are used to learn the ring dictionary.  Rings are
encoded for official train+valid and broadcast to their member atoms; official
test graphs are neither ring-enumerated here nor sparse-coded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd
from .molhiv_ring_cells import chordless_cycles, ring_cell_vector


def _unit_ring_vector(g, cycle, node_feat, edge_feat, min_size, max_size):
    y = ring_cell_vector(
        g, cycle, node_feat, edge_feat, min_size=min_size, max_size=max_size
    ).astype(np.float64, copy=False)
    return y / max(float(np.linalg.norm(y)), 1e-12)


def _reservoir_train_rings(
    graphs,
    train_idx,
    node_feats,
    edge_feats,
    min_size: int,
    max_size: int,
    capacity: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    seen = 0
    graphs_with_rings = 0
    size_counts: dict[int, int] = {k: 0 for k in range(min_size, max_size + 1)}
    for raw_i in train_idx:
        i = int(raw_i)
        cycles = chordless_cycles(graphs[i], min_size=min_size, max_size=max_size)
        graphs_with_rings += int(bool(cycles))
        for cycle in cycles:
            size_counts[len(cycle)] += 1
            y = _unit_ring_vector(
                graphs[i], cycle, node_feats[i], edge_feats[i], min_size, max_size
            )
            if reservoir is None:
                reservoir = np.zeros((y.size, capacity), dtype=np.float64)
            if y.size != reservoir.shape[0]:
                raise ValueError("ring-cell feature dimension changed within dataset")
            if seen < capacity:
                reservoir[:, seen] = y
            else:
                j = int(rng.integers(0, seen + 1))
                if j < capacity:
                    reservoir[:, j] = y
            seen += 1
    if reservoir is None or seen == 0:
        raise RuntimeError("no chordless official-train rings were found")
    used = min(seen, capacity)
    return reservoir[:, :used], {
        "n_train_rings_raw": int(seen),
        "n_train_rings_used": int(used),
        "n_train_graphs_with_rings": int(graphs_with_rings),
        "train_ring_size_counts": {str(k): int(v) for k, v in size_counts.items()},
        "feature_dim": int(reservoir.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-token-cache", required=True)
    ap.add_argument("--max-graphs", type=int, default=8000, help="0 means full dataset")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--min-ring-size", type=int, default=3)
    ap.add_argument("--max-ring-size", type=int, default=6)
    ap.add_argument("--ring-atoms", type=int, default=16)
    ap.add_argument("--ring-sparsity", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-train-rings", type=int, default=6000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not 1 <= args.ring_sparsity <= args.ring_atoms:
        raise ValueError("invalid ring sparsity")

    t0 = time.time()
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)

    base_path = Path(args.base_token_cache)
    with np.load(base_path) as base:
        offsets = np.asarray(base["offsets"], dtype=np.int64)
        original_indices = np.asarray(base["original_indices"], dtype=np.int64)
        expected = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
        if not np.array_equal(original_indices, expected):
            raise ValueError("base cache original_indices mismatch")
        if offsets.shape != (len(graphs) + 1,):
            raise ValueError("base cache offset shape mismatch")
        base_tokens = np.asarray(base["tokens_ksvd"], dtype=np.float32)
    if base_tokens.shape[0] != int(offsets[-1]):
        raise ValueError("base token row count mismatch")
    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        if np.any(base_tokens[lo:hi] != 0):
            raise ValueError("base cache contains encoded official-test nodes")

    Ytr, ring_stats = _reservoir_train_rings(
        graphs,
        tr,
        bundle.node_feats,
        bundle.edge_feats,
        min_size=args.min_ring_size,
        max_size=args.max_ring_size,
        capacity=args.max_train_rings,
        seed=args.dict_seed,
    )
    print(f"train ring pool {Ytr.shape}; raw={ring_stats['n_train_rings_raw']}", flush=True)
    D, _, dictionary_info = ksvd(
        Ytr,
        n_atoms=args.ring_atoms,
        T=args.ring_sparsity,
        T_min=1,
        n_iter=args.ksvd_iter,
        seed=args.dict_seed,
    )
    print(f"learned ring KSVD dictionary {D.shape}", flush=True)

    ring_tokens = np.zeros((int(offsets[-1]), args.ring_atoms), dtype=np.float32)
    memberships = np.zeros(int(offsets[-1]), dtype=np.int16)
    # A complementary graph-level K-SVD motif histogram.  It is later repeated
    # over a graph's atoms only to reuse the strict node-token runner; mean
    # pooling recovers exactly one graph descriptor.
    graph_ring_tokens = np.zeros(
        (int(offsets[-1]), 4 * args.ring_atoms), dtype=np.float32
    )
    graph_ring_memberships = np.zeros(int(offsets[-1]), dtype=np.int16)
    encoded_rings = 0
    encoded_graphs_with_rings = 0
    encode_size_counts = {k: 0 for k in range(args.min_ring_size, args.max_ring_size + 1)}
    encode_indices = np.concatenate([tr, va])
    for count, raw_i in enumerate(encode_indices, 1):
        i = int(raw_i)
        cycles = chordless_cycles(
            graphs[i], min_size=args.min_ring_size, max_size=args.max_ring_size
        )
        graph_codes: list[np.ndarray] = []
        encoded_graphs_with_rings += int(bool(cycles))
        for cycle in cycles:
            encode_size_counts[len(cycle)] += 1
            y = _unit_ring_vector(
                graphs[i], cycle, bundle.node_feats[i], bundle.edge_feats[i],
                args.min_ring_size, args.max_ring_size,
            )
            z = _omp(D, y, args.ring_sparsity).astype(np.float32)
            graph_codes.append(z)
            for u in cycle:
                row = int(offsets[i] + u)
                ring_tokens[row] += z
                memberships[row] += 1
            encoded_rings += 1
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if graph_codes:
            Z = np.stack(graph_codes, axis=0)
            graph_readout = np.concatenate(
                [
                    Z.mean(axis=0),
                    np.abs(Z).mean(axis=0),
                    np.abs(Z).max(axis=0),
                    (np.abs(Z) > 1e-10).mean(axis=0),
                ]
            ).astype(np.float32)
        else:
            graph_readout = np.zeros(4 * args.ring_atoms, dtype=np.float32)
        graph_ring_tokens[lo:hi] = graph_readout
        graph_ring_memberships[lo:hi] = 1
        if count % 500 == 0 or count == len(encode_indices):
            print(
                f"encoded graphs {count}/{len(encode_indices)}; rings={encoded_rings}",
                flush=True,
            )
    mask = memberships > 0
    ring_tokens[mask] /= memberships[mask, None].astype(np.float32)

    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        if np.any(ring_tokens[lo:hi] != 0) or np.any(memberships[lo:hi] != 0):
            raise AssertionError("official-test ring nodes were encoded")
        if np.any(graph_ring_tokens[lo:hi] != 0) or np.any(graph_ring_memberships[lo:hi] != 0):
            raise AssertionError("official-test ring graph readouts were encoded")
    augmented = np.concatenate([base_tokens, ring_tokens], axis=1)
    graph_augmented = np.concatenate([base_tokens, graph_ring_tokens], axis=1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=original_indices,
        train_indices=tr,
        valid_indices=va,
        test_indices=te,
        dictionary_ring_ksvd=D.astype(np.float32),
        tokens_ksvd_ring=augmented,
        token_slices=np.asarray([0, base_tokens.shape[1], augmented.shape[1]], dtype=np.int64),
        ring_memberships=memberships,
        tokens_ksvd_ring_graph=graph_augmented,
        graph_token_slices=np.asarray(
            [0, base_tokens.shape[1], graph_augmented.shape[1]], dtype=np.int64
        ),
        graph_ring_memberships=graph_ring_memberships,
    )
    meta = {
        "protocol_id": "molhiv-explicit-ring-cell-node-tokens-v1",
        "test_policy": "official test rings were not enumerated, vectorized, or sparse-coded",
        "config": vars(args),
        "data_meta": bundle.meta,
        "base_token_cache": str(base_path),
        "base_token_dim": int(base_tokens.shape[1]),
        "ring_token_dim": int(args.ring_atoms),
        "total_token_dim": int(augmented.shape[1]),
        "graph_ring_readout_dim": int(graph_ring_tokens.shape[1]),
        "graph_total_token_dim": int(graph_augmented.shape[1]),
        "ring_stats": ring_stats,
        "dictionary_info": dictionary_info,
        "encoded_train_valid_rings": int(encoded_rings),
        "encoded_train_valid_graphs_with_rings": int(encoded_graphs_with_rings),
        "encoded_ring_size_counts": {str(k): int(v) for k, v in encode_size_counts.items()},
        "n_atoms_with_ring_membership": int(mask.sum()),
        "max_ring_memberships_per_atom": int(memberships.max(initial=0)),
        "aggregation": "mean ring sparse code broadcast to member atoms",
        "graph_aggregation": "signed mean + mean abs + max abs + support rate of ring sparse codes",
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
