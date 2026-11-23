"""Sparsemax over each root's candidate scores (temperature fixed at 1).

Production path is a vectorised sort/threshold implementation on padded
``[n_roots, max_candidates]`` tensors.  ``sparsemax_reference`` is an
independent NumPy bisection solve of

    sum_i max(z_i - tau, 0) = 1

and is used only by the pre-training checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def sparsemax_padded(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sparsemax on padded rows; ``mask`` is bool, True for real candidates."""
    if scores.ndim != 2 or mask.shape != scores.shape:
        raise ValueError("sparsemax_padded expects [rows, candidates] score and mask")
    k = scores.shape[1]
    if k == 0:
        raise ValueError("sparsemax_padded requires at least one column")
    neg = torch.finfo(scores.dtype).min / 8
    z = scores.masked_fill(~mask, neg)
    z_sorted, _ = torch.sort(z, dim=1, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=1)
    idx = torch.arange(1, k + 1, dtype=scores.dtype, device=scores.device)
    # Standard closed form: largest j with 1 + j*z_(j) > sum_{i<=j} z_(i).
    support = 1.0 + idx.unsqueeze(0) * z_sorted > z_cumsum
    counts = mask.sum(dim=1).clamp(min=1, max=k)
    k_support = torch.minimum(support.sum(dim=1).clamp(min=1, max=k), counts)
    last = z_cumsum.gather(1, (k_support - 1).unsqueeze(1)).squeeze(1)
    tau = (last - 1.0) / k_support.to(scores.dtype)
    alpha = torch.clamp(scores - tau.unsqueeze(1), min=0.0)
    return alpha * mask.to(scores.dtype)


def sparsemax_reference(scores: np.ndarray, mask: np.ndarray | None = None,
                        iterations: int = 200) -> np.ndarray:
    """Independent NumPy bisection reference for one or more score rows."""
    z = np.asarray(scores, np.float64)
    if z.ndim == 1:
        z = z[None, :]
        squeeze = True
    else:
        squeeze = False
    if mask is None:
        m = np.ones_like(z, bool)
    else:
        m = np.asarray(mask, bool)
        if m.shape != z.shape:
            raise ValueError("mask shape mismatch")
    out = np.zeros_like(z, np.float64)
    for row in range(z.shape[0]):
        vals = z[row][m[row]]
        if vals.size == 0:
            continue
        hi = float(vals.max())
        lo = hi - 1.0
        # solve sum(max(vals - tau, 0)) = 1 by bisection
        for _ in range(int(iterations)):
            mid = 0.5 * (lo + hi)
            s = np.maximum(vals - mid, 0.0).sum()
            if s > 1.0:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        out[row][m[row]] = np.maximum(vals - tau, 0.0)
    return out[0] if squeeze else out


def sparsemax_reference_sorted(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Second independent NumPy reference: sorted cumulative-sum scan."""
    z = np.asarray(scores, np.float64)
    if z.ndim == 1:
        z = z[None, :]
        squeeze = True
    else:
        squeeze = False
    if mask is None:
        m = np.ones_like(z, bool)
    else:
        m = np.asarray(mask, bool)
        if m.shape != z.shape:
            raise ValueError("mask shape mismatch")
    out = np.zeros_like(z, np.float64)
    for row in range(z.shape[0]):
        orig = np.flatnonzero(m[row])
        if orig.size == 0:
            continue
        order = orig[np.argsort(z[row][orig])[::-1]]
        vals = z[row][order]
        cum = np.cumsum(vals)
        k_best = 0
        for j in range(1, vals.size + 1):
            if 1.0 + j * vals[j - 1] > cum[j - 1]:
                k_best = j
        k_best = max(1, k_best)
        tau = (cum[k_best - 1] - 1.0) / k_best
        out[row][order] = np.maximum(vals - tau, 0.0)
    return out[0] if squeeze else out


def sparsemax_packed(scores: torch.Tensor, root_word_ptr: torch.Tensor,
                     n_atoms: int) -> torch.Tensor:
    """Sparsemax over packed candidates contiguous per root.

    ``root_word_ptr`` has length ``n_atoms + 1`` and gives each root's candidate
    range.  Returns weights in the same packed order as ``scores``.
    """
    if scores.ndim != 1:
        raise ValueError("sparsemax_packed expects a packed score vector")
    counts = root_word_ptr[1:] - root_word_ptr[:-1]
    if int(counts.min().item()) <= 0:
        raise ValueError("every root must own at least one candidate")
    n_words = int(scores.numel())
    if int(counts.sum().item()) != n_words:
        raise ValueError("root_word_ptr does not cover the packed scores")
    max_k = int(counts.max().item())
    offsets = root_word_ptr[:-1]
    cols = torch.arange(max_k, device=scores.device)
    idx = offsets.unsqueeze(1) + cols.unsqueeze(0)
    mask = cols.unsqueeze(0) < counts.unsqueeze(1)
    idx = idx.clamp(min=0, max=max(0, n_words - 1))
    padded = scores.index_select(0, idx.reshape(-1)).view(n_atoms, max_k)
    # Padding must be neutral during the sort.  A duplicate of a real score
    # can otherwise enter the top-k and change tau (wrong sum).
    padded = padded.masked_fill(~mask, torch.finfo(scores.dtype).min / 8)
    alpha_pad = sparsemax_padded(padded, mask)
    # Boolean row-major selection reconstructs exactly the packed order.
    return alpha_pad[mask]