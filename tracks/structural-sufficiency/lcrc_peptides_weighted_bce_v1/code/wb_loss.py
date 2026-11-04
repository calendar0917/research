"""Weighted BCE with per-label ``pos_weight`` and optional missing-label mask.

For Peptides-func every Train/Validation label is observed, so the training
call takes the exact ``reduction="mean"`` path requested by the protocol.  The
masked path is retained for completeness and computes the elementwise loss
*before* masking, then averages only over valid entries.  It therefore never
flattens ``[batch, labels]`` logits and never mis-broadcasts ``pos_weight``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class WeightedBCEError(RuntimeError):
    pass


def validate_pos_weight(
    pos_weight: Any, n_labels: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    tensor = torch.as_tensor(pos_weight, dtype=dtype, device=device)
    if tensor.ndim != 1 or int(tensor.shape[0]) != int(n_labels):
        raise WeightedBCEError(f"pos_weight shape {tuple(tensor.shape)} != ({n_labels},)")
    if not bool(torch.isfinite(tensor).all()):
        raise WeightedBCEError("pos_weight contains non-finite values")
    if bool((tensor <= 0).any()):
        raise WeightedBCEError("pos_weight contains non-positive values")
    return tensor


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | Any,
    *,
    n_labels: int | None = None,
) -> torch.Tensor:
    """BCEWithLogits with a per-label positive weight.

    ``pos_weight`` is ``[C]`` and is applied to the last axis of ``[N, C]``
    logits.  With all labels finite this is exactly

        F.binary_cross_entropy_with_logits(logits, targets,
                                           pos_weight=pos_weight,
                                           reduction="mean")

    When labels are missing, the elementwise reduction is used and only valid
    entries are averaged.  Invalid targets are replaced by zero *after* the
    mask is fixed; they stay masked out, so they contribute no value or
    gradient.
    """
    if logits.ndim != 2 or targets.ndim != 2:
        raise WeightedBCEError(f"logits/targets must be 2-D, got {logits.shape} / {targets.shape}")
    if logits.shape != targets.shape:
        raise WeightedBCEError(f"logits {tuple(logits.shape)} != targets {tuple(targets.shape)}")
    if logits.device != targets.device:
        raise WeightedBCEError(f"logits device {logits.device} != targets device {targets.device}")
    c = int(logits.shape[1]) if n_labels is None else int(n_labels)
    if int(logits.shape[1]) != c:
        raise WeightedBCEError(f"logits last dim {logits.shape[1]} != n_labels {c}")
    weight = validate_pos_weight(pos_weight, c, logits.device, logits.dtype)

    mask = torch.isfinite(targets)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    if bool(mask.all()):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight, reduction="mean"
        )
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    per_element = F.binary_cross_entropy_with_logits(
        logits, safe_targets, pos_weight=weight, reduction="none"
    )
    return per_element[mask].mean()


__all__ = ["WeightedBCEError", "validate_pos_weight", "weighted_bce_with_logits"]
