"""Structural (ring/cycle) context for Compact-v3.

Purpose
-------
Compact-v2 forms a context-free patch representation: ``Encode(patch)``.
Compact-v3 studies ``Encode(patch | structural context)`` by *modifying* the
exact patch embedding with a small multiplicative conditioning module driven
by higher-order ring/cycle context of the centre node.

This module is deliberately *not* a CIN:

* no ring/cell nodes, no atom-ring incidence graph, no ring message passing;
* ring/cycle information is only a *conditioning signal* that will be fused
  into the patch representation later; it never becomes a new computation
  node of the model.

Data policy
-----------
The extractor uses only what the molecular graph already carries:

* node types (``data.x`` categories),
* edge types (``data.edge_attr`` categories),
* topology.

No RDKit-only annotation is introduced.  ``max_cycle_len`` bounds cycle
enumeration so the definition is explicit and configurable, not silently
unbounded.

Correctness properties (tested in
``tests/test_zinc_patch_path_structural_context_v3.py``):

* **permutation invariance** — per-node context keys depend on node
  types / edge types / cycle topology only, never on node IDs;
* **direction invariance** — a cycle read clockwise or counter-clockwise
  yields the same canonical typed signature;
* **train-only vocabulary** — context keys are collected from train records
  only; unseen keys map to ``UNK_CONTEXT``; keys with no ring map to
  ``NO_RING``.

Token layout
------------
::

    0 = NO_RING     (conditioning delta forced to zero)
    1 = UNK_CONTEXT (learned but never fitted on train; appears only on
                     validation/test)
    2.. = known     (ranked by train occurrence, then lexicographic key)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

COARSE_WIDTH = 4  # in_cycle, cycle_count, min_cycle_len, max_cycle_len
NO_RING_TOKEN = 0
UNK_CONTEXT_TOKEN = 1
CONTEXT_VERSION = b"v1:"


@dataclass(frozen=True)
class NodeStructuralContext:
    """Per-node higher-order ring context (0 entry when the node has no ring)."""

    in_cycle: bool
    cycle_count: int
    min_cycle_len: int | None
    max_cycle_len: int | None
    key: bytes | None
    coarse: np.ndarray

    @property
    def is_ring(self) -> bool:
        return self.in_cycle


def _canonical_cycle(cycle: Sequence[int]) -> tuple[int, ...]:
    """Canonical rotation/direction of a simple cycle (deduplication only).

    Node IDs appear here only to *deduplicate* the enumerated cycles; the
    resulting set of cycles is a graph invariant.  Typed signatures never use
    node IDs (see ``_typed_cycle_signature``), so node relabelling cannot
    change any context key.
    """
    as_tuple = tuple(int(node) for node in cycle)
    size = len(as_tuple)
    rotations: list[tuple[int, ...]] = [
        tuple(as_tuple[index:] + as_tuple[:index]) for index in range(size)
    ]
    reversed_cycle = tuple([as_tuple[0]] + list(reversed(as_tuple[1:])))
    rotations.extend(
        tuple(reversed_cycle[index:] + reversed_cycle[:index])
        for index in range(size)
    )
    return min(rotations)


def _find_cycles(graph: Any, max_cycle_len: int) -> set[tuple[int, ...]]:
    """Enumerate every simple cycle up to ``max_cycle_len`` exactly once.

    Classic min-node DFS: start from node ``v`` and only extend through nodes
    with ID greater than ``v``; a cycle is recorded when the path returns to
    ``v``.  Every simple cycle therefore has exactly one starting node (its
    minimum-ID node) and is recorded at most twice (both traversal
    directions); ``_canonical_cycle`` deduplicates them.
    """
    max_cycle_len = int(max_cycle_len)
    if max_cycle_len < 3:
        raise ValueError(f"max_cycle_len must be at least 3, got {max_cycle_len}")
    cycles: set[tuple[int, ...]] = set()
    for start in graph.nodes:
        start = int(start)
        path: list[int] = [start]
        visited: set[int] = {start}

        def walk(node: int) -> None:
            for neighbor in graph.neighbors(node):
                neighbor = int(neighbor)
                if neighbor == start:
                    if len(path) >= 3:
                        cycles.add(_canonical_cycle(path))
                    continue
                if neighbor <= start or neighbor in visited:
                    continue
                if len(path) >= max_cycle_len:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                walk(neighbor)
                path.pop()
                visited.remove(neighbor)

        walk(start)
    return cycles


def _typed_cycle_signature(
    graph: Any,
    cycle: Sequence[int],
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    center: int,
) -> bytes:
    """Center-anchored canonical typed cycle signature.

    The signature is the alternating ``node_type, edge_type, ...`` sequence
    read around the cycle starting at ``center``.  Both traversal directions
    are computed and the lexicographically smaller bytes value is returned,
    so traversal direction and node numbering can never change the token.
    """
    center = int(center)
    as_tuple = tuple(int(node) for node in cycle)
    start = as_tuple.index(center)
    order = list(as_tuple[start:]) + list(as_tuple[:start])
    size = len(order)

    def traversal(ordered: Sequence[int]) -> bytes:
        pieces: list[str] = []
        for index in range(size):
            left = int(ordered[index])
            right = int(ordered[(index + 1) % size])
            pieces.append(str(int(node_types[left])))
            bond = int(edge_types[graph.edge_key(left, right)])
            pieces.append(str(int(bond)))
        return CONTEXT_VERSION + ";".join(pieces).encode("ascii")

    forward = traversal(order)
    reverse = traversal([order[0]] + list(reversed(order[1:])))
    return min(forward, reverse)


def _node_key(ordered_signatures: Sequence[bytes]) -> bytes:
    return CONTEXT_VERSION + b"CYCLE|" + b"|".join(sorted(ordered_signatures))


def extract_structural_context(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    max_cycle_len: int = 8,
) -> tuple[dict[int, NodeStructuralContext], tuple[int, ...]]:
    """Extract per-node ring context (and the graph's cycle lengths).

    Returns ``(contexts, cycle_lengths)`` where ``contexts`` maps each node ID
    to its :class:`NodeStructuralContext` and ``cycle_lengths`` is the sorted
    tuple of the graph's simple-cycle lengths (bounded by ``max_cycle_len``).

    The graph is enumerated once: all the per-node fields below are derived
    from the same cycle set, so cycle count / min / max are mutually
    consistent.
    """
    max_cycle_len = int(max_cycle_len)
    cycles = _find_cycles(graph, max_cycle_len)
    cycle_lengths = tuple(sorted(len(cycle) for cycle in cycles))

    by_node: dict[int, list[tuple[int, ...]]] = {
        int(node): [] for node in graph.nodes
    }
    for cycle in cycles:
        for node in cycle:
            by_node[int(node)].append(cycle)

    contexts: dict[int, NodeStructuralContext] = {}
    for node in graph.nodes:
        node = int(node)
        containing = by_node[node]
        if not containing:
            contexts[node] = NodeStructuralContext(
                in_cycle=False,
                cycle_count=0,
                min_cycle_len=None,
                max_cycle_len=None,
                key=None,
                coarse=np.zeros(COARSE_WIDTH, dtype=np.float32),
            )
            continue
        signatures = sorted(
            {
                _typed_cycle_signature(
                    graph, cycle, node_types, edge_types, int(node)
                )
                for cycle in containing
            }
        )
        lengths = [len(cycle) for cycle in containing]
        min_length = int(min(lengths))
        max_length = int(max(lengths))
        contexts[node] = NodeStructuralContext(
            in_cycle=True,
            cycle_count=len(containing),
            min_cycle_len=min_length,
            max_cycle_len=max_length,
            key=_node_key(signatures),
            coarse=np.asarray(
                [1.0, float(len(containing)), float(min_length), float(max_length)],
                dtype=np.float32,
            ),
        )
    return contexts, cycle_lengths


def fit_structural_vocabulary(
    records: Sequence[Any],
) -> dict[bytes, int]:
    """Build the ``key -> token`` vocabulary from train records only.

    Token IDs: 0 = NO_RING, 1 = UNK_CONTEXT, known keys start at 2 and are
    ordered by (train occurrence count descending, key ascending) so the ID
    assignment is deterministic and label-free.
    """
    counts: Counter[bytes] = Counter()
    for record in records:
        for patch in record.patches:
            context = getattr(patch, "structural_context", None)
            if context is not None and context.key is not None:
                counts[context.key] += 1
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    return {key: index + 2 for index, key in enumerate(ordered)}


def encode_structural_token(
    context: NodeStructuralContext | None, vocabulary: Mapping[bytes, int]
) -> int:
    """Map a node context to its token (0 = NO_RING, 1 = UNK, 2+ = known)."""
    if context is None or not context.in_cycle or context.key is None:
        return NO_RING_TOKEN
    return int(vocabulary.get(context.key, UNK_CONTEXT_TOKEN))


def coarsen(context: NodeStructuralContext | None) -> np.ndarray:
    """Return the coarse 4D feature vector (zeros for NO_RING)."""
    if context is None or not context.in_cycle:
        return np.zeros(COARSE_WIDTH, dtype=np.float32)
    return np.asarray(context.coarse, dtype=np.float32)


def context_statistics(
    records: Sequence[Any],
    vocabulary: Mapping[bytes, int] | None,
) -> dict[str, Any]:
    """Summarize context coverage for one split given a (train-fitted) vocabulary."""
    patches = [
        getattr(patch, "structural_context", None)
        for record in records
        for patch in record.patches
    ]
    ring_patches = [context for context in patches if context is not None and context.in_cycle]
    n = max(len(patches), 1)
    statistics = {
        "n_patches": int(len(patches)),
        "no_ring_fraction": float(1.0 - len(ring_patches) / n),
        "in_ring_fraction": float(len(ring_patches) / n),
        "mean_cycle_count": (
            float(np.mean([c.cycle_count for c in ring_patches])) if ring_patches else 0.0
        ),
        "max_cycle_count": (
            int(max(c.cycle_count for c in ring_patches)) if ring_patches else 0
        ),
        "min_cycle_length_mean": (
            float(np.mean([c.min_cycle_len for c in ring_patches])) if ring_patches else 0.0
        ),
        "max_cycle_length_mean": (
            float(np.mean([c.max_cycle_len for c in ring_patches])) if ring_patches else 0.0
        ),
    }
    if vocabulary is not None:
        known = 0
        for context in ring_patches:
            if context.key is not None and context.key in vocabulary:
                known += 1
        statistics["oov_fraction"] = float(1.0 - known / max(len(ring_patches), 1))
        statistics["known_ring_fraction"] = float(known / max(len(ring_patches), 1))
    return statistics


def cycle_length_distribution(records: Sequence[Any]) -> dict[str, int]:
    """Global cycle-length histogram (all simple cycles, bounded by max_cycle_len)."""
    counter: Counter[int] = Counter()
    for record in records:
        for length in getattr(record, "cycle_lengths", ()):
            counter[int(length)] += 1
    return {str(length): counter[length] for length in sorted(counter)}
