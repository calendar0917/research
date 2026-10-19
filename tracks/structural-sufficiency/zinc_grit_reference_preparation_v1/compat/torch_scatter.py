"""Minimal torch_scatter shim for the frozen GRIT reference on PyG>=2.6.

Only the subset used by GRIT is implemented.  Implementations delegate to
torch_geometric.utils.scatter so the reduction math is unchanged; the shim only
adds the historical `out=` argument and scatter_max's (values, argmax) tuple.
"""
from __future__ import annotations

import torch
from torch_geometric.utils import scatter as _pyg_scatter


def _norm_reduce(reduce: str) -> str:
    return "sum" if reduce == "add" else reduce


def scatter(src, index, dim=0, dim_size=None, reduce="sum", out=None):
    result = _pyg_scatter(src, index, dim=dim, dim_size=dim_size,
                          reduce=_norm_reduce(reduce))
    if out is not None:
        out.copy_(result)
        return out
    return result


def scatter_add(src, index, dim=0, dim_size=None, out=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="add", out=out)


def scatter_max(src, index, dim=0, dim_size=None):
    values = _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="max")
    # argmax is not used by GRIT (only values are indexed); return None to make
    # that explicit if a future caller tries to use it.
    return values, None
