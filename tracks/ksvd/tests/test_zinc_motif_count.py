from __future__ import annotations

import numpy as np
from scipy import sparse
import torch

from tracks.ksvd.experiments.luyin16.zinc_motif_count import (
    _build_morgan_counts,
    _encode_wl_counts,
    _fit_topk_vocab,
    _hstack_dense_sparse,
    _morgan_tokens,
    _object_row_array,
    _typed_wl_tokens,
)


class _Data:
    def __init__(self, x: list[int], edges: list[tuple[int, int, int]]) -> None:
        self.x = torch.tensor(x, dtype=torch.long).view(-1, 1)
        edge_index = []
        edge_attr = []
        for index, (u, v, bond) in enumerate(edges):
            edge_index.extend([(u, v), (v, u)])
            edge_attr.extend([bond, bond])
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        self.num_nodes = len(x)


def _relabel(data: _Data, permutation: np.ndarray) -> _Data:
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    edges = []
    for position in range(0, data.edge_index.shape[1], 2):
        u = int(data.edge_index[0, position])
        v = int(data.edge_index[1, position])
        edges.append((int(inverse[u]), int(inverse[v]), int(data.edge_attr[position])))
    return _Data([int(data.x[int(permutation[i])].item()) for i in range(len(permutation))], edges)


def test_typed_wl_and_morgan_are_relabeling_invariant() -> None:
    data = _Data([0, 1, 0, 2, 3], [(0, 1, 1), (1, 2, 2), (1, 3, 1), (3, 4, 1)])
    permutation = np.asarray([3, 0, 4, 2, 1], dtype=np.int64)
    relabeled = _relabel(data, permutation)
    first_wl = _typed_wl_tokens(data)
    second_wl = _typed_wl_tokens(relabeled)
    first_morgan = _morgan_tokens(data)
    second_morgan = _morgan_tokens(relabeled)
    for first, second in zip(first_wl, second_wl, strict=True):
        assert sorted(first) == sorted(second)
    for first, second in zip(first_morgan, second_morgan, strict=True):
        assert sorted(first) == sorted(second)


def test_topk_vocabulary_is_fit_only_on_requested_rows() -> None:
    rows = [
        (("common",), ("a",)),
        (("common",), ("b",)),
        (("heldout-only",), ("c",)),
    ]
    vocab = _fit_topk_vocab(rows, [0, 1], rounds=2, top_k=1)
    assert "heldout-only" not in vocab[0]
    matrix = _encode_wl_counts(rows, vocab, rounds=2, top_k=1)
    assert matrix.shape == (3, 2 * 2 * (1 + 1))
    assert matrix[2].nnz > 0


def test_count_readouts_preserve_raw_and_normalized_scales() -> None:
    rows = [
        (("a", "a"), ("b",)),
        (("a",), ("b",)),
    ]
    vocab = _fit_topk_vocab(rows, [0, 1], rounds=2, top_k=4)
    matrix = _encode_wl_counts(rows, vocab, rounds=2, top_k=4)
    assert sparse.isspmatrix_csr(matrix)
    # For round 0, the raw count of ``a`` is two in graph 0 and one in graph 1.
    assert matrix[0].sum() > matrix[1].sum()
    assert matrix[0].sum() > 0.0
    morgan = _build_morgan_counts([_morgan_tokens(_Data([0, 1], [(0, 1, 1)]))], bits=32)
    assert morgan.shape == (1, 64)
    assert sparse.isspmatrix_csr(_hstack_dense_sparse(np.zeros((1, 3), dtype=np.float32), morgan))


def test_object_cache_array_keeps_one_item_per_graph() -> None:
    rows = [((b"a",), (b"b",)), ((b"c",), (b"d",))]
    packed = _object_row_array(rows)
    assert packed.shape == (2,)
    assert packed.tolist() == rows
