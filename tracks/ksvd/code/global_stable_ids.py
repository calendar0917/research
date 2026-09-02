"""Relabel-invariant structural classes and practical graph-global node IDs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from .from_scratch_unplanted_representation import validate_simple_adjacency


Digest = bytes


@dataclass(frozen=True)
class StableNodeIDs:
    """Stable structural classes plus conservative unique-ID diagnostics.

    ``canonical_ids`` are exact stable IDs only where ``unique_mask`` is true.
    Within a non-singleton class they use input order merely as an opaque handle.
    """

    method: str
    canonical_ids: tuple[int, ...]
    class_ids: tuple[int, ...]
    unique_mask: tuple[bool, ...]
    order: tuple[int, ...]
    class_sizes: tuple[int, ...]
    class_keys: tuple[tuple, ...]

    @property
    def singleton_fraction(self) -> float:
        return float(np.mean(self.unique_mask)) if self.unique_mask else 0.0

    @property
    def fully_singleton(self) -> bool:
        return all(self.unique_mask)

    @property
    def largest_class(self) -> int:
        return max(self.class_sizes, default=0)

    @property
    def ambiguous_class_count(self) -> int:
        return sum(size > 1 for size in self.class_sizes)


def _integer_core_numbers(adjacency: np.ndarray) -> np.ndarray:
    """Compute k-core numbers by deterministic simultaneous peeling."""
    n_nodes = adjacency.shape[0]
    maximum_degree = int(np.max(np.sum(adjacency, axis=1), initial=0))
    core = np.zeros(n_nodes, dtype=np.int64)
    for k in range(1, maximum_degree + 1):
        alive = np.ones(n_nodes, dtype=bool)
        while True:
            alive_indices = np.flatnonzero(alive)
            if alive_indices.size == 0:
                break
            degrees = np.sum(
                adjacency[np.ix_(alive_indices, alive_indices)], axis=1
            )
            remove = alive_indices[degrees < k]
            if remove.size == 0:
                break
            alive[remove] = False
        core[alive] = k
    return core


def _distance_histograms(adjacency: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n_nodes = adjacency.shape[0]
    output = []
    for root in range(n_nodes):
        distances = np.full(n_nodes, n_nodes + 1, dtype=np.int64)
        distances[root] = 0
        queue = [root]
        for node in queue:
            for raw_neighbor in np.flatnonzero(adjacency[node]):
                neighbor = int(raw_neighbor)
                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        if np.any(distances > n_nodes):
            raise ValueError("stable IDs require a connected graph")
        histogram = np.bincount(distances, minlength=n_nodes).astype(np.int64)
        last = int(np.max(distances))
        output.append(tuple(int(value) for value in histogram[1 : last + 1]))
    return tuple(output)


def structural_signatures(adjacency: np.ndarray) -> tuple[tuple, ...]:
    """Return integer-only initial signatures, independent of numeric node IDs."""
    values = np.asarray(adjacency, dtype=np.int8)
    validate_simple_adjacency(values)
    degrees = np.sum(values, axis=1).astype(np.int64)
    triangles = np.diag(
        values.astype(np.int64) @ values.astype(np.int64) @ values.astype(np.int64)
    ) // 2
    cores = _integer_core_numbers(values)
    distances = _distance_histograms(values)
    power = np.eye(values.shape[0], dtype=np.int64)
    matrix = values.astype(np.int64)
    closed_walks: list[np.ndarray] = []
    for exponent in range(1, 6):
        power = power @ matrix
        if exponent >= 2:
            closed_walks.append(np.diag(power).copy())
    return tuple(
        (
            int(degrees[node]),
            int(triangles[node]),
            int(cores[node]),
            distances[node],
            tuple(int(walk[node]) for walk in closed_walks),
        )
        for node in range(values.shape[0])
    )


def _digest_parts(parts: Sequence[bytes]) -> Digest:
    digest = sha256()
    for part in parts:
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def _initial_digest(signature: tuple, *, rooted: bool) -> Digest:
    return _digest_parts(
        (b"stable-node-id-v1", repr(signature).encode("ascii"), bytes((int(rooted),)))
    )


def _wl_digest_refinement(
    adjacency: np.ndarray,
    signatures: Sequence[tuple],
    *,
    root: int | None,
    rounds: int | None = None,
) -> tuple[tuple[Digest, ...], tuple[tuple[int, ...], ...]]:
    """Run fixed-round digest WL and return per-node digests plus class-size history."""
    n_nodes = adjacency.shape[0]
    count = n_nodes if rounds is None else int(rounds)
    if count < 1:
        raise ValueError("round count must be positive")
    digests = tuple(
        _initial_digest(signature, rooted=(root == node))
        for node, signature in enumerate(signatures)
    )
    history = []
    for _iteration in range(count):
        history.append(tuple(sorted(Counter(digests).values())))
        updated = []
        for node in range(n_nodes):
            neighbors = sorted(digests[int(value)] for value in np.flatnonzero(adjacency[node]))
            updated.append(_digest_parts((b"wl", digests[node], *neighbors)))
        digests = tuple(updated)
    history.append(tuple(sorted(Counter(digests).values())))
    return digests, tuple(history)


def _rooted_fingerprint(
    adjacency: np.ndarray,
    signatures: Sequence[tuple],
    root: int,
) -> tuple:
    digests, history = _wl_digest_refinement(
        adjacency, signatures, root=root
    )
    cells: dict[Digest, list[int]] = {}
    for node, digest in enumerate(digests):
        cells.setdefault(digest, []).append(node)
    ordered_digests = tuple(sorted(cells))
    cell_sizes = tuple(len(cells[digest]) for digest in ordered_digests)
    quotient = []
    for left_digest in ordered_digests:
        left = np.asarray(cells[left_digest], dtype=np.int64)
        for right_digest in ordered_digests:
            right = np.asarray(cells[right_digest], dtype=np.int64)
            quotient.append(int(np.sum(adjacency[np.ix_(left, right)])))
    return (
        digests[root],
        history,
        ordered_digests,
        cell_sizes,
        tuple(quotient),
    )


def _build_result(method: str, keys: Sequence[tuple]) -> StableNodeIDs:
    node_keys = tuple(keys)
    unique_keys = tuple(sorted(set(node_keys)))
    key_to_class = {key: index for index, key in enumerate(unique_keys)}
    class_ids = tuple(key_to_class[key] for key in node_keys)
    members = {
        class_id: tuple(
            node for node, value in enumerate(class_ids) if value == class_id
        )
        for class_id in range(len(unique_keys))
    }
    class_sizes = tuple(len(members[index]) for index in range(len(unique_keys)))
    unique_mask = tuple(class_sizes[class_ids[node]] == 1 for node in range(len(node_keys)))
    order = tuple(
        node
        for class_id in range(len(unique_keys))
        for node in members[class_id]
    )
    canonical_ids_list = [0] * len(node_keys)
    for canonical_id, node in enumerate(order):
        canonical_ids_list[node] = canonical_id
    return StableNodeIDs(
        method=method,
        canonical_ids=tuple(canonical_ids_list),
        class_ids=class_ids,
        unique_mask=unique_mask,
        order=order,
        class_sizes=class_sizes,
        class_keys=unique_keys,
    )


def compute_global_wl_ids(adjacency: np.ndarray) -> StableNodeIDs:
    values = np.asarray(adjacency, dtype=np.int8)
    signatures = structural_signatures(values)
    digests, _history = _wl_digest_refinement(values, signatures, root=None)
    keys = tuple((digest,) for digest in digests)
    return _build_result("global_wl", keys)


def compute_rooted_wl_ids(adjacency: np.ndarray) -> StableNodeIDs:
    values = np.asarray(adjacency, dtype=np.int8)
    signatures = structural_signatures(values)
    global_digests, _history = _wl_digest_refinement(values, signatures, root=None)
    counts = Counter(global_digests)
    keys = tuple(
        (
            global_digests[node],
            _rooted_fingerprint(values, signatures, node)
            if counts[global_digests[node]] > 1
            else (),
        )
        for node in range(values.shape[0])
    )
    return _build_result("rooted_wl", keys)


def ambiguity_audit(
    adjacency: np.ndarray, ids: StableNodeIDs
) -> dict[str, float | int]:
    """Check whether unresolved same-class pairs are simple swap automorphisms."""
    values = np.asarray(adjacency, dtype=np.int8)
    pairs = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            if ids.class_ids[left] == ids.class_ids[right]:
                pairs.append((left, right))
    automorphic = 0
    for left, right in pairs:
        others = [
            node for node in range(values.shape[0]) if node not in (left, right)
        ]
        automorphic += int(
            np.array_equal(values[left, others], values[right, others])
        )
    return {
        "ambiguous_pair_count": len(pairs),
        "ambiguous_swap_automorphism_fraction": (
            automorphic / len(pairs) if pairs else 1.0
        ),
    }


def reorder_by_stable_ids(
    adjacency: np.ndarray, ids: StableNodeIDs
) -> np.ndarray:
    values = np.asarray(adjacency, dtype=np.int8)
    if values.shape != (len(ids.order), len(ids.order)):
        raise ValueError("stable ID order is incompatible with adjacency")
    order = np.asarray(ids.order, dtype=np.int64)
    return values[np.ix_(order, order)].copy()


def mapped_id_audit(
    base: StableNodeIDs,
    relabeled: StableNodeIDs,
    permutation: np.ndarray,
) -> dict[str, float]:
    """Compare independently computed IDs under new-index -> old-index mapping."""
    mapping = np.asarray(permutation, dtype=np.int64)
    if mapping.shape != (len(base.class_ids),):
        raise ValueError("permutation and stable IDs are incompatible")
    class_matches = [
        relabeled.class_ids[new] == base.class_ids[int(old)]
        for new, old in enumerate(mapping)
    ]
    singleton_trials = 0
    singleton_matches = 0
    concrete_matches = []
    for new, raw_old in enumerate(mapping):
        old = int(raw_old)
        concrete_matches.append(
            relabeled.canonical_ids[new] == base.canonical_ids[old]
        )
        if base.unique_mask[old]:
            singleton_trials += 1
            singleton_matches += int(
                relabeled.canonical_ids[new] == base.canonical_ids[old]
                and relabeled.unique_mask[new]
            )
    return {
        "stable_class_match_rate": float(np.mean(class_matches)),
        "singleton_unique_id_match_rate": (
            singleton_matches / max(singleton_trials, 1)
        ),
        "all_node_concrete_id_match_rate": float(np.mean(concrete_matches)),
        "singleton_trial_count": int(singleton_trials),
    }
