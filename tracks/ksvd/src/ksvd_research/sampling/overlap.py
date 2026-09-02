"""Patch-cover API backed by the historical implementation.

The full legacy cover implementation is intentionally not moved in this
reorganization.  This boundary exposes the stable names needed by canonical
slot ordering and future experiments without changing any historical script.
"""

from .._legacy import import_legacy


_legacy = import_legacy("overlap_cover")
OrderedPatch = _legacy.OrderedPatch
PatchCover = _legacy.PatchCover
PatchTransition = _legacy.PatchTransition
_make_cover = _legacy._make_cover
_make_patch = _legacy._make_patch
audit_cover = _legacy.audit_cover
cover_vectors = _legacy.cover_vectors
make_slot_persistent_cover = _legacy.make_slot_persistent_cover
patch_budget = _legacy.patch_budget
remap_cover = _legacy.remap_cover
sample_edge_target_bridge_cover = _legacy.sample_edge_target_bridge_cover
sample_frontier_cover = _legacy.sample_frontier_cover
sample_independent_walk_cover = _legacy.sample_independent_walk_cover
sample_multi_chain_target_cover = _legacy.sample_multi_chain_target_cover
sample_sliding_walk_cover = _legacy.sample_sliding_walk_cover

# Public spellings for new code; private spellings remain available for old
# tests and are kept out of the package's documented API.
make_patch = _make_patch
make_cover = _make_cover

__all__ = [
    "OrderedPatch",
    "PatchTransition",
    "PatchCover",
    "make_patch",
    "make_cover",
    "patch_budget",
    "sample_independent_walk_cover",
    "sample_sliding_walk_cover",
    "sample_frontier_cover",
    "sample_edge_target_bridge_cover",
    "sample_multi_chain_target_cover",
    "audit_cover",
    "remap_cover",
    "make_slot_persistent_cover",
    "cover_vectors",
]
