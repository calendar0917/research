"""Stable dictionary-learning API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("ksvd")
_omp = _legacy._omp
ksvd = _legacy.ksvd
mil_attention_weights = _legacy.mil_attention_weights
pool_X = _legacy.pool_X
readout_X = _legacy.readout_X

__all__ = ["ksvd", "mil_attention_weights", "pool_X", "readout_X"]
