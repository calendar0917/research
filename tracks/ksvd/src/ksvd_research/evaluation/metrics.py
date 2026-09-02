"""Sampling metrics API backed by the historical implementation."""

from .._legacy import import_legacy


_legacy = import_legacy("metrics")
compare_methods = _legacy.compare_methods
evaluate_bundle = _legacy.evaluate_bundle

__all__ = ["evaluate_bundle", "compare_methods"]
