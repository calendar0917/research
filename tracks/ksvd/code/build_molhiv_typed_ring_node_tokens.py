"""Build size-typed ring-cell K-SVD node tokens for MolHIV.

Separate dictionaries prevent the abundant six-member rings from consuming the
whole ring vocabulary.  Ring sizes 3/4 share a small dictionary; sizes 5 and 6
receive dedicated capacities.  Every dictionary is learned from official train
only and official test rings are never enumerated or encoded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .build_molhiv_ring_node_tokens import _unit_ring_vector
from .data_molhiv import load_molhiv
from .ksvd import _omp, ksvd
from .molhiv_ring_cells import chordless_cycles


TYPE_ORDER = ("small", "five", "six")


def ring_type(cycle: tuple[int, ...]) -> str:
    if len(cycle) <= 4:
        return "small"
    if len(cycle) == 5:
        return "five"
    if len(cycle) == 6:
        return "six"
    raise ValueError("unsupported ring size")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-token-cache", required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--dict-seed", type=int, default=0)
    ap.add_argument("--atoms-small", type=int, default=4)
    ap.add_argument("--atoms-five", type=int, default=8)
    ap.add_argument("--atoms-six", type=int, default=16)
    ap.add_argument("--sparsity-small", type=int, default=1)
    ap.add_argument("--sparsity-five", type=int, default=2)
    ap.add_argument("--sparsity-six", type=int, default=2)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument("--max-rings-small", type=int, default=1000)
    ap.add_argument("--max-rings-five", type=int, default=3000)
    ap.add_argument("--max-rings-six", type=int, default=6000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    atoms = {typ: int(getattr(args, f"atoms_{typ}")) for typ in TYPE_ORDER}
    sparsity = {typ: int(getattr(args, f"sparsity_{typ}")) for typ in TYPE_ORDER}
    capacity = {typ: int(getattr(args, f"max_rings_{typ}")) for typ in TYPE_ORDER}
    for typ in TYPE_ORDER:
        if not 1 <= sparsity[typ] <= atoms[typ]:
            raise ValueError(f"invalid sparsity for {typ}")

    t0 = time.time()
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    bundle = load_molhiv(max_graphs=max_graphs, seed=args.data_seed, with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features were not loaded")
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.base_token_cache) as base:
        offsets = np.asarray(base["offsets"], dtype=np.int64)
        original_indices = np.asarray(base["original_indices"], dtype=np.int64)
        base_tokens = np.asarray(base["tokens_ksvd"], dtype=np.float32)
    if not np.array_equal(original_indices, np.asarray(bundle.meta["original_indices"])):
        raise ValueError("base cache original_indices mismatch")
    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        if np.any(base_tokens[lo:hi] != 0):
            raise ValueError("base cache contains encoded official-test nodes")

    rng = {typ: np.random.default_rng(args.dict_seed + 1009 * j) for j, typ in enumerate(TYPE_ORDER)}
    reservoirs: dict[str, np.ndarray | None] = {typ: None for typ in TYPE_ORDER}
    seen = {typ: 0 for typ in TYPE_ORDER}
    for raw_i in tr:
        i = int(raw_i)
        for cycle in chordless_cycles(graphs[i], 3, 6):
            typ = ring_type(cycle)
            y = _unit_ring_vector(
                graphs[i], cycle, bundle.node_feats[i], bundle.edge_feats[i], 3, 6
            )
            if reservoirs[typ] is None:
                reservoirs[typ] = np.zeros((y.size, capacity[typ]), dtype=np.float64)
            R = reservoirs[typ]
            assert R is not None
            n = seen[typ]
            if n < capacity[typ]:
                R[:, n] = y
            else:
                pick = int(rng[typ].integers(0, n + 1))
                if pick < capacity[typ]:
                    R[:, pick] = y
            seen[typ] += 1

    dictionaries: dict[str, np.ndarray] = {}
    dictionary_info = {}
    for j, typ in enumerate(TYPE_ORDER):
        R = reservoirs[typ]
        if R is None or seen[typ] == 0:
            raise RuntimeError(f"no train rings for type {typ}")
        Y = R[:, : min(seen[typ], capacity[typ])]
        D, _, info = ksvd(
            Y,
            n_atoms=atoms[typ],
            T=sparsity[typ],
            T_min=1,
            n_iter=args.ksvd_iter,
            seed=args.dict_seed + 1009 * j,
        )
        dictionaries[typ] = D
        dictionary_info[typ] = info
        print(f"{typ}: pool={Y.shape} raw={seen[typ]} dictionary={D.shape}", flush=True)

    slices = [0]
    for typ in TYPE_ORDER:
        slices.append(slices[-1] + atoms[typ])
    ring_tokens = np.zeros((int(offsets[-1]), slices[-1]), dtype=np.float32)
    type_memberships = np.zeros((int(offsets[-1]), len(TYPE_ORDER)), dtype=np.int16)
    encoded_counts = {typ: 0 for typ in TYPE_ORDER}
    for count, raw_i in enumerate(np.concatenate([tr, va]), 1):
        i = int(raw_i)
        for cycle in chordless_cycles(graphs[i], 3, 6):
            typ = ring_type(cycle)
            j = TYPE_ORDER.index(typ)
            y = _unit_ring_vector(
                graphs[i], cycle, bundle.node_feats[i], bundle.edge_feats[i], 3, 6
            )
            z = _omp(dictionaries[typ], y, sparsity[typ]).astype(np.float32)
            lo_s, hi_s = slices[j], slices[j + 1]
            for u in cycle:
                row = int(offsets[i] + u)
                ring_tokens[row, lo_s:hi_s] += z
                type_memberships[row, j] += 1
            encoded_counts[typ] += 1
        if count % 500 == 0 or count == len(tr) + len(va):
            print(f"encoded graphs {count}/{len(tr)+len(va)} rings={sum(encoded_counts.values())}", flush=True)
    for j, typ in enumerate(TYPE_ORDER):
        mask = type_memberships[:, j] > 0
        ring_tokens[mask, slices[j]:slices[j + 1]] /= type_memberships[mask, j, None]
    memberships = type_memberships.sum(axis=1)
    for i in te:
        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
        if np.any(ring_tokens[lo:hi] != 0) or np.any(memberships[lo:hi] != 0):
            raise AssertionError("official-test rings were encoded")

    augmented = np.concatenate([base_tokens, ring_tokens], axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        "offsets": offsets,
        "original_indices": original_indices,
        "train_indices": tr,
        "valid_indices": va,
        "test_indices": te,
        "tokens_ksvd_ring_typed": augmented,
        "token_slices": np.asarray([0, base_tokens.shape[1], augmented.shape[1]], dtype=np.int64),
        "typed_ring_slices": np.asarray(slices, dtype=np.int64),
        "ring_memberships": memberships,
        "typed_ring_memberships": type_memberships,
    }
    for typ, D in dictionaries.items():
        archive[f"dictionary_ring_{typ}"] = D.astype(np.float32)
    np.savez_compressed(output, **archive)
    meta = {
        "protocol_id": "molhiv-size-typed-ring-cell-node-tokens-v1",
        "test_policy": "official test rings were not enumerated, vectorized, or sparse-coded",
        "config": vars(args),
        "type_order": list(TYPE_ORDER),
        "atoms": atoms,
        "sparsity": sparsity,
        "ring_token_dim": int(slices[-1]),
        "total_token_dim": int(augmented.shape[1]),
        "train_ring_counts": seen,
        "encoded_train_valid_ring_counts": encoded_counts,
        "dictionary_info": dictionary_info,
        "elapsed_sec": time.time() - t0,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
