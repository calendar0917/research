"""Patch vectorization and structural slot ordering."""

from .canonical_slots import (
    CanonicalOrderResult,
    exact_canonical_order,
    reorder_cover_structurally,
    signature_order,
)
from .vectorize import (
    adjacency_padded,
    canonical_adjacency_features,
    flatten_upper,
    labeled_wl_patch_features,
    labeled_wl_ring_patch_features,
    patch_feature_dim,
    ring_patch_features,
    subgraph_extra_features,
    wl_patch_features,
)
from .ring_context import (
    RING_CONTEXT_NAMES,
    RingContextIndex,
    build_ring_context_index,
    chordless_cycles,
)

__all__ = [
    "adjacency_padded",
    "flatten_upper",
    "canonical_adjacency_features",
    "wl_patch_features",
    "labeled_wl_patch_features",
    "labeled_wl_ring_patch_features",
    "ring_patch_features",
    "patch_feature_dim",
    "subgraph_extra_features",
    "CanonicalOrderResult",
    "exact_canonical_order",
    "signature_order",
    "reorder_cover_structurally",
    "RING_CONTEXT_NAMES",
    "RingContextIndex",
    "build_ring_context_index",
    "chordless_cycles",
]
