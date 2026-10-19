"""Minimal torch_sparse shim for the frozen GRIT reference.

Dense implementation, adequate for ZINC-12k graphs (max ~38 nodes).  It keeps
the RRWP definition identical: row-normalised random-walk powers, including the
identity, returned as full N x N pair features.  `coalesce` delegates to PyG's
coalesce, which performs the same duplicate-index reduction.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.utils import coalesce as _pyg_coalesce


def coalesce(index: torch.Tensor, value: torch.Tensor, m: int, n: int,
             op: str = "add"):
    reduce = "sum" if op == "add" else op
    out = _pyg_coalesce(index, value, num_nodes=max(int(m), int(n)),
                        reduce=reduce)
    return out[0], out[1]


class SparseTensor:
    """Dense-backed stand-in for the small subset of torch_sparse.SparseTensor."""

    def __init__(self, dense: torch.Tensor):
        self._dense = dense

    @classmethod
    def from_edge_index(cls, edge_index: torch.Tensor,
                        edge_weight: Optional[torch.Tensor] = None,
                        sparse_sizes=None) -> "SparseTensor":
        if sparse_sizes is None:
            n = int(edge_index.max()) + 1
            m = n
        else:
            m, n = int(sparse_sizes[0]), int(sparse_sizes[1])
        dense = torch.zeros((m, n), dtype=torch.float,
                            device=edge_index.device)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                     device=edge_index.device)
        dense.index_put_((edge_index[0], edge_index[1]), edge_weight,
                         accumulate=True)
        return cls(dense)

    @classmethod
    def from_dense(cls, dense: torch.Tensor, has_value: bool = True) -> "SparseTensor":
        return cls(dense)

    def to_dense(self) -> torch.Tensor:
        return self._dense

    def sum(self, dim: int) -> torch.Tensor:
        return self._dense.sum(dim=dim)

    def __mul__(self, other) -> "SparseTensor":
        return SparseTensor(self._dense * other)

    def coo(self):
        if self._dense.dim() < 2:
            raise ValueError("SparseTensor shim expects at least 2 dimensions")
        m, n = self._dense.shape[:2]
        rows = torch.arange(m, device=self._dense.device).repeat_interleave(n)
        cols = torch.arange(n, device=self._dense.device).repeat(m)
        vals = self._dense.reshape(m * n, -1)
        if vals.size(1) == 1:
            vals = vals[:, 0]
        return rows, cols, vals
