"""Reusable KSVD research primitives.

The historical scripts under :mod:`tracks.ksvd.code` remain available for
reproduction.  New experiments should import this package instead of relying
on the legacy ``code`` module name.
"""

from .core.graph import Graph, from_edges, ring_chords, star_graph
from .core.ksvd import ksvd

__version__ = "0.1.0"

__all__ = [
    "Graph",
    "from_edges",
    "ring_chords",
    "star_graph",
    "ksvd",
]
