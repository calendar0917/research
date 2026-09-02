"""Stable structural-slot API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("canonical_slots")
CanonicalOrderResult = _legacy.CanonicalOrderResult
exact_canonical_order = _legacy.exact_canonical_order
reorder_cover_structurally = _legacy.reorder_cover_structurally
signature_order = _legacy.signature_order

__all__ = [
    "CanonicalOrderResult",
    "exact_canonical_order",
    "signature_order",
    "reorder_cover_structurally",
]
