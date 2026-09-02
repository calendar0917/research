"""Coverage-driven random-walk sampling API."""

from .._legacy import import_legacy


_legacy = import_legacy("coverage_sample")
CoverageConfig = _legacy.CoverageConfig
_pick_seeds = _legacy._pick_seeds
_uncovered_start = _legacy._uncovered_start
sample_coverage = _legacy.sample_coverage
trajectory_cover_metrics = _legacy.trajectory_cover_metrics

__all__ = ["CoverageConfig", "sample_coverage", "trajectory_cover_metrics"]
