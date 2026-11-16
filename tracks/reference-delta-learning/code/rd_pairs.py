"""Frozen joint training pairs for the C arm (delta learner).

Construction is a single frozen RNG pass (``np.random.default_rng(1701)``) over
the Train queries in ascending global row_id order.  For query ``i``

1. draw 16 references uniformly without replacement from the legal Top64,
2. draw 16 references uniformly without replacement from the legal Train
   references *outside* the Top64,
3. if either pool is short, take what exists and top up uniformly from the
   remaining legal references (never a repeat, never self, never a same-SMILES
   row).

Each selected reference ``j`` produces two records

    z(i,j) = [x_i, x_j, x_i - x_j]      target (y_i - y_j) / sigma
    z(j,i) = [x_j, x_i, x_j - x_i]      target -(y_i - y_j) / sigma

both with sample weight ``1 / (2 q_i)`` so every source query contributes
exactly one unit of weight.  The sampling never looks at a label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rd_common as C
import rd_data as D


def eligible_order_train(task: D.TaskData) -> list[np.ndarray]:
    """All legal Train references per Train query, sorted by (sim desc, row_id asc).

    One int32 array per Train query (``n_train - 1`` entries unless exact-SMILES
    duplicates remove more).  The large task is 3360 arrays of 3359 int32
    (45 MB), fine in RAM.  Deterministic and label free, so the frozen pair
    pass can be re-derived exactly.
    """
    import rd_refs as R

    train_bits = task.bits["train"]
    train_row_id = task.row_id["train"].astype(np.int64)
    smiles = task.smiles["train"]
    n_train = int(train_bits.shape[0])
    sim = R.tanimoto_matrix(train_bits, train_bits)
    same: dict[str, list[int]] = {}
    for position, text in enumerate(smiles):
        same.setdefault(text, []).append(position)
    order: list[np.ndarray] = []
    for query in range(n_train):
        exclude = np.zeros(n_train, dtype=bool)
        exclude[query] = True
        for position in same.get(smiles[query], []):
            exclude[position] = True
        valid = np.nonzero(~exclude)[0]
        if valid.size == 0:
            raise ValueError(f"{task.dataset}: Train query {query} has no legal reference")
        scores = sim[query, valid]
        ranking = np.lexsort((train_row_id[valid], -scores))
        order.append(valid[ranking].astype(np.int32))
    return order


def _sample_without_replacement(rng: np.random.Generator, size: int, take: int) -> np.ndarray:
    if size <= 0 or take <= 0:
        return np.zeros(0, dtype=np.int64)
    return rng.choice(size, size=min(take, size), replace=False).astype(np.int64)


def build_pairs(
    task: D.TaskData,
    eligible_order: np.ndarray,
    *,
    seed: int = C.PAIR_RNG_SEED,
    n_top: int = C.N_TOP,
    n_tail: int = C.N_TAIL,
    max_pairs: int = C.MAX_PAIRS,
    top_pool: int = C.TOP_POOL,
) -> dict[str, Any]:
    """Frozen pair list (records in the order they were drawn)."""
    rng = np.random.default_rng(seed)
    n_train = len(eligible_order)
    y = task.y["train"].astype(np.float64)
    sigma = float(task.scaler["std"])
    if n_train != int(y.size):
        raise ValueError("eligible order and Train labels disagree")
    src: list[int] = []
    ref: list[int] = []
    origin: list[int] = []
    sign: list[int] = []
    target: list[float] = []
    weight: list[float] = []
    q_of_query = np.zeros(n_train, dtype=np.int32)
    chosen_lists: list[list[int]] = []
    for query in range(n_train):
        order = np.asarray(eligible_order[query], dtype=np.int32)
        top = order[:top_pool]
        tail = order[top_pool:]
        chosen = [int(top[i]) for i in _sample_without_replacement(rng, int(top.size), n_top)]
        chosen += [int(tail[i]) for i in _sample_without_replacement(rng, int(tail.size), n_tail)]
        if len(chosen) < min(max_pairs, int(order.size)):
            taken = set(chosen)
            remaining = [int(j) for j in order.tolist() if int(j) not in taken]
            need = min(max_pairs, int(order.size)) - len(chosen)
            picks = _sample_without_replacement(rng, len(remaining), need)
            chosen += [remaining[int(p)] for p in picks]
        if len(set(chosen)) != len(chosen):
            raise AssertionError("a reference was drawn twice for one query")
        if query in chosen:
            raise AssertionError("a query drew itself as a reference")
        q = len(chosen)
        q_of_query[query] = q
        chosen_lists.append(chosen)
        if q == 0:
            continue
        w = 1.0 / (2.0 * q)
        for j in chosen:
            delta = (float(y[query]) - float(y[j])) / sigma
            src.append(query)
            ref.append(int(j))
            origin.append(query)
            sign.append(1)
            target.append(delta)
            weight.append(w)
            src.append(int(j))
            ref.append(query)
            origin.append(query)
            sign.append(-1)
            target.append(-delta)
            weight.append(w)
    spec = {
        "dataset": task.dataset,
        "seed": int(seed),
        "n_train": n_train,
        "n_records": len(src),
        "src_local": np.asarray(src, dtype=np.int32),
        "ref_local": np.asarray(ref, dtype=np.int32),
        "sign": np.asarray(sign, dtype=np.int8),
        "target_std": np.asarray(target, dtype=np.float64),
        "weight": np.asarray(weight, dtype=np.float64),
        "q_per_query": q_of_query,
        "query_local": np.asarray(origin, dtype=np.int32),
        "chosen": chosen_lists,
        "top_pool": int(top_pool),
        "n_top": int(n_top),
        "n_tail": int(n_tail),
        "max_pairs": int(max_pairs),
    }
    spec["stats"] = pair_stats(spec)
    spec["hash"] = pair_hash(spec)
    return spec


def pair_stats(spec: dict[str, Any]) -> dict[str, Any]:
    src = np.asarray(spec["src_local"], dtype=np.int64)
    ref = np.asarray(spec["ref_local"], dtype=np.int64)
    n_records = int(src.size)
    n_train = int(spec["n_train"])
    q = np.asarray(spec["q_per_query"], dtype=np.int64)
    if int(q.sum()) * 2 != n_records:
        raise AssertionError(f"{q.sum()} pairs but {n_records} directed records")
    if np.any(src == ref):
        raise AssertionError("a training record has identical endpoints")
    if src.size == 0 or int(src.min()) < 0 or int(src.max()) >= n_train or int(ref.max()) >= n_train:
        raise AssertionError("a training record endpoint is outside Train")
    codes = src * n_train + ref
    n_distinct_directed = int(np.unique(codes).size)
    unordered = np.minimum(src, ref) * n_train + np.maximum(src, ref)
    n_distinct_unordered = int(np.unique(unordered).size)
    origin = np.asarray(spec["query_local"], dtype=np.int64)
    weights_per_query = np.zeros(n_train, dtype=np.float64)
    np.add.at(weights_per_query, origin, np.asarray(spec["weight"], dtype=np.float64))
    # 2*q_i records of weight 1/(2 q_i) sum to 1 up to float64 rounding; for
    # the frozen q_i = 32 the weight 1/64 is exact.
    if not np.allclose(weights_per_query[q > 0], 1.0, atol=0.0, rtol=1e-12):
        raise AssertionError("per-source-query weight does not sum to 1")
    if int((q > 0).sum()) != n_train:
        raise AssertionError("a Train query produced no pair")
    if q.size == 0 or int(q.min()) <= 0:
        raise AssertionError("at least one Train query has q_i = 0")
    endpoint_counts = np.zeros(n_train, dtype=np.int64)
    np.add.at(endpoint_counts, src, 1)
    np.add.at(endpoint_counts, ref, 1)
    return {
        "n_records": n_records,
        "n_records_expected": int(2 * n_train * C.MAX_PAIRS),
        "matches_expected_count": bool(n_records == 2 * n_train * C.MAX_PAIRS),
        "n_query_pair_slots": int(q.sum()),
        "n_distinct_directed_records": n_distinct_directed,
        "n_distinct_unordered_pairs": n_distinct_unordered,
        "duplicate_directed_rate": float(1.0 - n_distinct_directed / float(n_records)),
        "pairs_seen_from_both_endpoints": int((n_records - n_distinct_directed) // 2),
        "q_min": int(q.min()),
        "q_max": int(q.max()),
        "q_mean": float(q.mean()),
        "weight_sum_by_origin": float(weights_per_query.sum()),
        "weight_sum": float(np.asarray(spec["weight"], dtype=np.float64).sum()),
        "weight_sum_equals_n_train": bool(
            abs(float(np.asarray(spec["weight"]).sum()) - float(n_train)) <= 1e-9 * float(n_train)
        ),
        "endpoint_frequency_min": int(endpoint_counts.min()),
        "endpoint_frequency_max": int(endpoint_counts.max()),
        "endpoint_frequency_mean": float(endpoint_counts.mean()),
        "all_endpoints_in_train": True,
    }


def pair_hash(spec: dict[str, Any]) -> str:
    payload = {
        "dataset": spec["dataset"],
        "seed": int(spec["seed"]),
        "query_local": np.asarray(spec["query_local"]).tolist(),
        "src_local": np.asarray(spec["src_local"]).tolist(),
        "ref_local": np.asarray(spec["ref_local"]).tolist(),
        "sign": np.asarray(spec["sign"]).tolist(),
        "target_std": np.round(np.asarray(spec["target_std"], dtype=np.float64), 12).tolist(),
        "weight": np.round(np.asarray(spec["weight"], dtype=np.float64), 17).tolist(),
    }
    return C.sha256_json(payload)


def save_pairs(spec: dict[str, Any], path) -> dict[str, Any]:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        dataset=np.asarray(spec["dataset"]),
        seed=np.asarray(spec["seed"]),
        n_train=np.asarray(spec["n_train"]),
        query_local=spec["query_local"],
        src_local=spec["src_local"],
        ref_local=spec["ref_local"],
        sign=spec["sign"],
        target_std=spec["target_std"],
        weight=spec["weight"],
        q_per_query=spec["q_per_query"],
    )
    return {"path": str(target), "sha256": C.sha256_file(target), "bytes": int(target.stat().st_size)}


def load_pairs(path, dataset: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    spec: dict[str, Any] = {
        "dataset": str(arrays["dataset"]),
        "seed": int(arrays["seed"]),
        "n_train": int(arrays["n_train"]),
        "query_local": arrays["query_local"].astype(np.int32),
        "src_local": arrays["src_local"].astype(np.int32),
        "ref_local": arrays["ref_local"].astype(np.int32),
        "sign": arrays["sign"].astype(np.int8),
        "target_std": arrays["target_std"].astype(np.float64),
        "weight": arrays["weight"].astype(np.float64),
        "q_per_query": arrays["q_per_query"].astype(np.int32),
    }
    if spec["dataset"] != dataset:
        raise ValueError(f"pair file is for {spec['dataset']}, expected {dataset}")
    spec["stats"] = pair_stats(spec)
    spec["hash"] = pair_hash(spec)
    return spec


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------


def _validate_pair_domains(
    query_x: np.ndarray,
    reference_x: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate two *explicit* input domains, never one implicit Train domain.

    The caller decides which split each side of ``z(query, reference)`` comes
    from.  Training pairs pass Train for both sides; evaluation pairs pass the
    evaluated split for the query and Train for the reference.  Keeping the two
    domains as separate arguments is what makes a Val/Test query local index
    impossible to silently reinterpret as a Train row.
    """
    qx = np.asarray(query_x)
    rx = np.asarray(reference_x)
    q = np.asarray(query_local, dtype=np.int64).reshape(-1)
    r = np.asarray(reference_local, dtype=np.int64).reshape(-1)
    if qx.ndim != 2 or rx.ndim != 2:
        raise ValueError("query_x and reference_x must both be 2-D design matrices")
    if qx.shape[1] != rx.shape[1]:
        raise ValueError(f"feature width mismatch: {qx.shape[1]} != {rx.shape[1]}")
    if q.size != r.size:
        raise ValueError(f"query/reference slot count mismatch: {q.size} != {r.size}")
    if q.size and (int(q.min()) < 0 or int(q.max()) >= int(qx.shape[0])):
        raise IndexError("a query_local index is outside its own input domain")
    if r.size and (int(r.min()) < 0 or int(r.max()) >= int(rx.shape[0])):
        raise IndexError("a reference_local index is outside the Train input domain")
    return qx, rx, q, r


def pair_matrix(
    *,
    query_x: np.ndarray,
    reference_x: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
    chunk: int = 8192,
) -> np.ndarray:
    """Dense float32 ``[x_q, x_r, x_q - x_r]`` (6177 columns).

    ``query_x``/``query_local`` and ``reference_x``/``reference_local`` are two
    independent domains.  A Val/Test evaluation passes ``query_x=task.x[split]``
    and ``reference_x=task.x["train"]``; a Train--Train training pair passes
    ``task.x["train"]`` for both sides.
    """
    qx, rx, q, r = _validate_pair_domains(
        query_x, reference_x, query_local, reference_local
    )
    n = int(q.size)
    n_features = int(qx.shape[1])
    out = np.empty((n, 3 * n_features), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        head = qx[q[start:stop]]
        tail = rx[r[start:stop]]
        out[start:stop, :n_features] = head
        out[start:stop, n_features:2 * n_features] = tail
        out[start:stop, 2 * n_features:] = head - tail
    return out


def pair_feature_blocks(
    *,
    query_x: np.ndarray,
    reference_x: np.ndarray,
    query_local: np.ndarray,
    reference_local: np.ndarray,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-major copies of the three 2059-column blocks.

    ``X1``/``X2``/``X3`` are ``(2059, n_records)`` contiguous float32, so a tree
    split reads one contiguous row instead of striding through a wide matrix.
    ``X3`` is computed in float32 exactly as the DMatrix column would be.

    As in :func:`pair_matrix`, the two sides are explicit domains.  The reverse
    pair is produced from the *correct* forward blocks by ``BlockLayout`` (swap
    X1/X2, negate X3); swapping only local indices while keeping the Train
    matrix would be a different, wrong operation.
    """
    qx, rx, q, r = _validate_pair_domains(
        query_x, reference_x, query_local, reference_local
    )
    n = int(q.size)
    n_features = int(qx.shape[1])
    x1 = np.empty((n_features, n), dtype=np.float32)
    x2 = np.empty((n_features, n), dtype=np.float32)
    x3 = np.empty((n_features, n), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        head = qx[q[start:stop]]
        tail = rx[r[start:stop]]
        x1[:, start:stop] = head.T
        x2[:, start:stop] = tail.T
        x3[:, start:stop] = (head - tail).T
    return x1, x2, x3


def evaluation_pairs(table_split: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (query, reference) slots over the frozen Top64 table.

    Missing slots (fewer than 64 legal references) are sentinel *Train* slots
    with weight zero, never the query's local index interpreted in Train.  The
    frozen weight matrix is zero there, so they do not contribute to N/R/C; a
    cross-domain padding index would nevertheless be ill-defined for a Val/Test
    query and is therefore avoided explicitly.

    All queries are kept; only individual padded slots are inactive.
    """
    ref_local = np.asarray(table_split["ref_local"], dtype=np.int32)
    n_kept = np.asarray(table_split["n_kept"], dtype=np.int32)
    n_query, width = ref_local.shape
    query = np.repeat(np.arange(n_query, dtype=np.int32), width)
    ref = ref_local.reshape(-1).astype(np.int32).copy()
    flat_kept = np.repeat(n_kept, width)
    slot = np.tile(np.arange(width, dtype=np.int32), n_query)
    missing = slot >= flat_kept
    # 0 is always a legal Train-local sentinel; the corresponding weight is 0.
    ref[missing] = 0
    return query, ref
