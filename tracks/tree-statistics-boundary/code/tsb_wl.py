"""Rooted typed-WL token vocabularies and count / presence features.

Protocol rules implemented here (frozen; see ``PROTOCOL.md`` §3.2)
----------------------------------------------------------------
* round 0 is the atom attribute signature; round ``k+1`` is
  ``current signature + sorted multiset of (bond attributes, neighbour
  signature)`` -- canonical strings, never ``hash()``;
* the vocabulary of each round is built from the **full official Train rows of
  that one published split** only (all 3,360 rows, also when the model is
  trained on 1,000/2,000 rows);
* the 512 most frequent tokens are kept, ties broken by the canonical token
  string ascending; one extra OOV column absorbs every other token;
* rounds are independent, so ``WL2`` and ``WL3`` share the exact same rounds
  0-2 columns (same names, same order, same OOV policy);
* ``WL2_BIN`` is the same vocabulary with counts clipped to 0/1.

The missing-value policy is "no missing values": an unseen token contributes
to the OOV column and nothing is NaN.  A molecule that cannot be parsed is a
hard error (there are none in this dataset; the count is reported).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

import tsb_common as C


def aggregate_token_counts(per_molecule: Iterable[Mapping[str, int]]) -> Counter:
    total: Counter = Counter()
    for counts in per_molecule:
        total.update(counts)
    return total


def select_vocabulary(token_counts: Mapping[str, int], size: int = C.WL_VOCAB_SIZE) -> list[str]:
    """Top-``size`` tokens by frequency, ties by canonical string ascending."""
    items = sorted(token_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [str(token) for token, _count in items[: int(size)]]


def vocabulary_payload(
    tokens: Sequence[str],
    token_counts: Mapping[str, int],
    *,
    round_index: int,
    train_rows_sha256: str,
    n_train_rows: int,
) -> dict[str, Any]:
    return {
        "schema": "tsb-typed-wl-vocab-v1",
        "protocol_id": C.PROTOCOL_ID,
        "round": int(round_index),
        "n_vocab": int(len(tokens)),
        "oov_column": True,
        "vocab_source": "official Train rows of this published split (label-free)",
        "train_rows_sha256": train_rows_sha256,
        "n_train_rows": int(n_train_rows),
        "atom_attributes": list(C.WL_ATOM_ATTRIBUTES),
        "bond_attributes": list(C.WL_BOND_ATTRIBUTES),
        "selection_rule": (
            "top-512 token frequency over Train atoms; ties broken by canonical "
            "token string ascending; ranks are the column order"
        ),
        "tokens": [str(t) for t in tokens],
        "token_sha256": [C.sha256_bytes(str(t).encode()) for t in tokens],
        "train_occurrences": [int(token_counts.get(t, 0)) for t in tokens],
        "columns": column_names(tokens, round_index),
    }


def column_names(tokens: Sequence[str], round_index: int) -> list[str]:
    names = [
        f"wl{int(round_index)}_t{rank:03d}_{C.sha256_bytes(str(token).encode())[:12]}"
        for rank, token in enumerate(tokens)
    ]
    names.append(f"wl{int(round_index)}_oov")
    return names


def build_matrix(
    per_molecule: Sequence[Mapping[str, int]],
    tokens: Sequence[str],
    *,
    binary: bool,
) -> sp.csr_matrix:
    """CSR count matrix ``(n_molecules, len(tokens)+1)`` for one round.

    Column ``i`` is ``tokens[i]``; the last column is the OOV bucket.  With
    ``binary=True`` every nonzero count becomes 1 (the OOV bucket included:
    it records whether the molecule had *any* unseen token in this round).
    """
    rank = {token: i for i, token in enumerate(tokens)}
    n_cols = len(tokens) + 1
    oov = n_cols - 1
    indptr = np.zeros(len(per_molecule) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[int] = []
    for row, counts in enumerate(per_molecule):
        row_counts: dict[int, int] = {}
        oov_total = 0
        for token, value in counts.items():
            value = int(value)
            col = rank.get(str(token))
            if col is None:
                oov_total += value
            else:
                row_counts[col] = row_counts.get(col, 0) + value
        if oov_total:
            row_counts[oov] = oov_total
        for col in sorted(row_counts):
            indices.append(col)
            data.append(1 if binary else row_counts[col])
        indptr[row + 1] = len(indices)
    matrix = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(per_molecule), n_cols),
        dtype=np.float32,
    )
    matrix.sort_indices()
    return matrix


def coverage_stats(
    per_molecule: Sequence[Mapping[str, int]],
    tokens: Sequence[str],
    *,
    train_rows: np.ndarray,
) -> dict[str, Any]:
    """OOV and column-usage statistics for one round on one split."""
    rank = {token: i for i, token in enumerate(tokens)}
    total_tokens = 0
    oov_tokens = 0
    rows_with_oov = 0
    seen_cols = np.zeros(len(tokens), dtype=bool)
    for counts in per_molecule:
        row_oov = 0
        for token, value in counts.items():
            value = int(value)
            total_tokens += value
            col = rank.get(str(token))
            if col is None:
                row_oov += value
                oov_tokens += value
            else:
                seen_cols[col] = True
        if row_oov:
            rows_with_oov += 1
    n_rows = len(per_molecule)
    in_train = np.zeros(n_rows, dtype=bool)
    in_train[np.asarray(train_rows, dtype=np.int64)] = True
    return {
        "n_rows": int(n_rows),
        "token_occurrences": int(total_tokens),
        "oov_occurrences": int(oov_tokens),
        "oov_fraction": float(oov_tokens / total_tokens) if total_tokens else 0.0,
        "rows_with_oov": int(rows_with_oov),
        "rows_with_oov_fraction": float(rows_with_oov / n_rows) if n_rows else 0.0,
        "in_vocab_token_fraction": float(1.0 - oov_tokens / total_tokens) if total_tokens else 1.0,
        "vocab_tokens_used": int(seen_cols.sum()),
        "vocab_coverage_fraction": float(seen_cols.sum() / len(tokens)) if tokens else 0.0,
        "n_columns": int(len(tokens) + 1),
        "n_train_rows": int(in_train.sum()),
        "n_eval_rows": int(n_rows - in_train.sum()),
    }


def nonzero_column_fraction(matrix: sp.csr_matrix) -> float:
    used = np.zeros(matrix.shape[1], dtype=bool)
    used[np.unique(matrix.indices)] = True
    return float(used.sum() / matrix.shape[1]) if matrix.shape[1] else 0.0


def sparse_density(matrix: sp.csr_matrix) -> float:
    return float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])) if matrix.size else 0.0


def save_csr(path, matrix: sp.csr_matrix) -> None:
    np.savez_compressed(
        path,
        data=matrix.data.astype(np.float32),
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )


def load_csr(path) -> sp.csr_matrix:
    with np.load(path, allow_pickle=False) as handle:
        data = np.asarray(handle["data"], dtype=np.float32)
        indices = np.asarray(handle["indices"], dtype=np.int32)
        indptr = np.asarray(handle["indptr"], dtype=np.int64)
        shape = tuple(int(x) for x in handle["shape"])
    return sp.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)
