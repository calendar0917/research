"""Stable patch-sampling API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("sample")
SampleBundle = _legacy.SampleBundle
SampleConfig = _legacy.SampleConfig
_cap_nodes = _legacy._cap_nodes
_weighted_choice = _legacy._weighted_choice
node2vec_walk = _legacy.node2vec_walk
run_method = _legacy.run_method
sample_B0 = _legacy.sample_B0
sample_rw = _legacy.sample_rw

__all__ = [
    "SampleConfig",
    "SampleBundle",
    "node2vec_walk",
    "sample_B0",
    "sample_rw",
    "run_method",
]
