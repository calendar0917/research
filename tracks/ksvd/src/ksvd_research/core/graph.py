"""Stable graph API backed by the historical implementation.

This is intentionally a thin compatibility layer during the migration.  It
gives new code a package-qualified import path while preserving exact legacy
behavior and keeping the old files available for reproduction.
"""

from .._legacy import import_legacy


_legacy = import_legacy("graph")
Graph = _legacy.Graph
from_edges = _legacy.from_edges
ring_chords = _legacy.ring_chords
star_graph = _legacy.star_graph

__all__ = ["Graph", "from_edges", "ring_chords", "star_graph"]
