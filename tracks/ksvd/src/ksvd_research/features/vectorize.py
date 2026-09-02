"""Stable patch-vectorization API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("vectorize")
adjacency_padded = _legacy.adjacency_padded
canonical_adjacency_features = _legacy.canonical_adjacency_features
flatten_upper = _legacy.flatten_upper
labeled_wl_patch_features = _legacy.labeled_wl_patch_features
labeled_wl_ring_patch_features = _legacy.labeled_wl_ring_patch_features
patch_feature_dim = _legacy.patch_feature_dim
ring_patch_features = _legacy.ring_patch_features
subgraph_extra_features = _legacy.subgraph_extra_features
wl_patch_features = _legacy.wl_patch_features

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
]
