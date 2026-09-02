"""Build a strictly inner-train label-reweighted localized KSVD cache for MolHIV.

Graph labels affect only the class balance of atom-centered patches offered to
K-SVD.  The dictionary still learns all atom directions from chemical patch
vectors, and labels are never appended to node tokens.  Official-valid and
Official-test nodes are deliberately left unencoded during screening.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd
from .molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def _offer(
    reservoirs: dict[int, np.ndarray | None],
    graph_slots: dict[int, np.ndarray],
    seen: dict[int, int],
    capacities: dict[int, int],
    label: int,
    graph_idx: int,
    vec: np.ndarray,
    rng: np.random.Generator,
) -> None:
    capacity = capacities[label]
    if capacity <= 0:
        return
    if reservoirs[label] is None:
        reservoirs[label] = np.zeros((vec.size, capacity), dtype=np.float64)
    reservoir = reservoirs[label]
    assert reservoir is not None
    n_seen = seen[label]
    if n_seen < capacity:
        slot = n_seen
    else:
        slot = int(rng.integers(0, n_seen + 1))
    if slot < capacity:
        reservoir[:, slot] = vec
        graph_slots[label][slot] = graph_idx
    seen[label] = n_seen + 1


def _balanced_inner_patch_pool(
    bundle,
    inner_train: np.ndarray,
    *,
    radius: int,
    max_nodes: int,
    capacity: int,
    positive_fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features are required")
    n_pos = int(round(capacity * positive_fraction))
    capacities = {1: n_pos, 0: capacity - n_pos}
    reservoirs: dict[int, np.ndarray | None] = {0: None, 1: None}
    graph_slots = {
        label: np.full(capacities[label], -1, dtype=np.int64) for label in (0, 1)
    }
    seen = {0: 0, 1: 0}
    rng = np.random.default_rng(seed)

    for count, raw_i in enumerate(inner_train, 1):
        i = int(raw_i)
        label = int(bundle.y[i] > 0.5)
        graph = bundle.graphs[i]
        for raw_u in graph.nodes:
            vec = np.asarray(
                centered_ego_vector(
                    graph,
                    int(raw_u),
                    bundle.node_feats[i],
                    bundle.edge_feats[i],
                    radius=radius,
                    max_nodes=max_nodes,
                ),
                dtype=np.float64,
            )
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            _offer(
                reservoirs, graph_slots, seen, capacities,
                label, i, vec, rng,
            )
        if count % 1000 == 0 or count == len(inner_train):
            print(
                f"pooled inner-train graphs {count}/{len(inner_train)}; "
                f"seen neg/pos={seen[0]}/{seen[1]}",
                flush=True,
            )

    blocks = []
    represented: dict[str, int] = {}
    for label in (0, 1):
        if seen[label] < capacities[label]:
            raise RuntimeError(
                f"class {label} has {seen[label]} patches, fewer than requested "
                f"{capacities[label]}"
            )
        reservoir = reservoirs[label]
        assert reservoir is not None
        blocks.append(reservoir)
        represented[str(label)] = int(np.unique(graph_slots[label]).size)
    Y = np.concatenate(blocks, axis=1)
    order = rng.permutation(Y.shape[1])
    Y = Y[:, order]
    return Y, {
        "method": "inner_train_graph_label_balanced_node_patch_reservoir",
        "capacity": int(capacity),
        "positive_fraction": float(positive_fraction),
        "selected_negative_patches": int(capacities[0]),
        "selected_positive_patches": int(capacities[1]),
        "seen_negative_patches": int(seen[0]),
        "seen_positive_patches": int(seen[1]),
        "represented_graphs_by_label": represented,
        "feature_dim": int(Y.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--inner-split-seed", type=int, default=1729)
    ap.add_argument("--inner-valid-fraction", type=float, default=0.15)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=3)
    ap.add_argument("--max-train-patches", type=int, default=6000)
    ap.add_argument("--positive-fraction", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not 0.0 < args.positive_fraction < 1.0:
        raise ValueError("--positive-fraction must be in (0, 1)")

    started = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.inner_valid_fraction,
        random_state=args.inner_split_seed,
    )
    inner_train_pos, inner_valid_pos = next(splitter.split(tr, bundle.y[tr]))
    inner_tr = tr[inner_train_pos]
    inner_va = tr[inner_valid_pos]

    Y, pool_stats = _balanced_inner_patch_pool(
        bundle,
        inner_tr,
        radius=args.radius,
        max_nodes=args.max_nodes,
        capacity=args.max_train_patches,
        positive_fraction=args.positive_fraction,
        seed=args.dict_seed,
    )
    print(f"task-aware inner patch pool {Y.shape}", flush=True)
    dictionary, _, dictionary_info = ksvd(
        Y,
        n_atoms=args.n_atoms,
        T=args.sparsity,
        T_min=1,
        n_iter=args.ksvd_iter,
        seed=args.dict_seed,
    )
    print(
        f"learned task-aware dictionary {dictionary.shape}; "
        f"reconstruction={dictionary_info.get('recon_rel')}",
        flush=True,
    )

    offsets = graph_node_offsets(graphs)
    tokens = np.zeros((int(offsets[-1]), args.n_atoms), dtype=np.float32)
    encoded_nodes = 0
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    # Encode official-train only. This covers inner-train and inner-valid while
    # leaving official-valid/test entirely untouched in the screening cache.
    for count, raw_i in enumerate(tr, 1):
        i = int(raw_i)
        graph = graphs[i]
        for raw_u in graph.nodes:
            u = int(raw_u)
            vec = np.asarray(
                centered_ego_vector(
                    graph,
                    u,
                    bundle.node_feats[i],
                    bundle.edge_feats[i],
                    radius=args.radius,
                    max_nodes=args.max_nodes,
                ),
                dtype=np.float64,
            )
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            tokens[int(offsets[i] + u)] = _omp(
                dictionary, vec, args.sparsity
            ).astype(np.float32)
            encoded_nodes += 1
        if count % 500 == 0 or count == len(tr):
            print(
                f"encoded official-train graphs {count}/{len(tr)}; "
                f"nodes={encoded_nodes}",
                flush=True,
            )

    for split_name, indices in (("official-valid", va), ("official-test", te)):
        for raw_i in indices:
            i = int(raw_i)
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            if np.any(tokens[lo:hi] != 0):
                raise AssertionError(f"{split_name} node tokens were unexpectedly encoded")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        offsets=offsets,
        original_indices=np.asarray(bundle.meta["original_indices"], dtype=np.int64),
        train_indices=tr,
        valid_indices=va,
        test_indices=te,
        inner_train_indices=inner_tr,
        inner_valid_indices=inner_va,
        dictionary_ksvd_task_aware=dictionary.astype(np.float32),
        tokens_ksvd_task_aware=tokens,
    )
    meta = {
        "protocol_id": "molhiv-localized-task-aware-ksvd-inner-screen-cache-v1",
        "test_policy": (
            "dictionary learned from inner-train only; official-valid and official-test "
            "node patches were not vectorized or sparse-coded"
        ),
        "config": vars(args),
        "n_official_train": int(len(tr)),
        "n_inner_train": int(len(inner_tr)),
        "n_inner_valid": int(len(inner_va)),
        "n_inner_train_positive": int(bundle.y[inner_tr].sum()),
        "pool_stats": pool_stats,
        "dictionary_info": dictionary_info,
        "encoded_official_train_nodes": int(encoded_nodes),
        "encoded_official_valid_nodes": 0,
        "encoded_official_test_nodes": 0,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "elapsed_sec": time.time() - started,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
