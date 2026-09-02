"""Continuous overlapping patch covers for graph reconstruction audits."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

from .from_scratch_unplanted_representation import (
    adjacency_to_upper_vector,
    is_connected,
    relabel_adjacency_and_order,
    sample_walk_patch,
    validate_simple_adjacency,
)


@dataclass(frozen=True)
class OrderedPatch:
    node_ids: tuple[int, ...]
    center: int
    adjacency: np.ndarray


@dataclass(frozen=True)
class PatchTransition:
    left_to_right_slots: tuple[tuple[int, int], ...]

    @property
    def overlap_size(self) -> int:
        return len(self.left_to_right_slots)


@dataclass(frozen=True)
class PatchCover:
    method: str
    patches: tuple[OrderedPatch, ...]
    transitions: tuple[PatchTransition, ...]
    segment_ids: tuple[int, ...]
    target_edges: tuple[tuple[int, int] | None, ...]
    bridge_lengths: tuple[int, ...]


def patch_budget(
    adjacency: np.ndarray,
    *,
    patch_size: int,
    target_overlap: int,
    edge_capacity_multiplier: float = 1.5,
) -> int:
    """Choose a matched budget from node- and edge-capacity lower bounds."""
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if not 1 <= target_overlap < patch_size <= n_nodes:
        raise ValueError("expected 1 <= target_overlap < patch_size <= n_nodes")
    stride = patch_size - target_overlap
    node_bound = 1 + int(np.ceil(max(n_nodes - patch_size, 0) / stride))
    edge_count = int(adjacency.sum() // 2)
    pair_capacity = patch_size * (patch_size - 1) // 2
    edge_bound = int(np.ceil(edge_count / pair_capacity))
    return max(node_bound, int(np.ceil(edge_capacity_multiplier * edge_bound)))


def _make_patch(adjacency: np.ndarray, node_ids: Sequence[int], center: int) -> OrderedPatch:
    nodes = tuple(int(node) for node in node_ids)
    if len(nodes) != len(set(nodes)):
        raise ValueError("patch nodes must be distinct")
    if center not in nodes:
        raise ValueError("patch center must belong to the patch")
    induced = adjacency[np.ix_(nodes, nodes)].astype(np.int8, copy=True)
    return OrderedPatch(node_ids=nodes, center=int(center), adjacency=induced)


def _transition(left: OrderedPatch, right: OrderedPatch) -> PatchTransition:
    right_slots = {node: slot for slot, node in enumerate(right.node_ids)}
    pairs = tuple(
        (left_slot, right_slots[node])
        for left_slot, node in enumerate(left.node_ids)
        if node in right_slots
    )
    return PatchTransition(left_to_right_slots=pairs)


def _make_cover(
    method: str,
    patches: Sequence[OrderedPatch],
    segment_ids: Sequence[int] | None = None,
    target_edges: Sequence[tuple[int, int] | None] | None = None,
    bridge_lengths: Sequence[int] | None = None,
) -> PatchCover:
    patches = tuple(patches)
    if not patches:
        raise ValueError("a patch cover cannot be empty")
    if segment_ids is None:
        segments = tuple(0 for _ in patches)
    else:
        segments = tuple(int(value) for value in segment_ids)
        if len(segments) != len(patches):
            raise ValueError("segment_ids must match the patch count")
        if segments[0] != 0 or any(
            right not in (left, left + 1) for left, right in zip(segments, segments[1:])
        ):
            raise ValueError("segment_ids must start at zero and increase contiguously")
    if target_edges is None:
        targets = tuple(None for _ in patches)
    else:
        targets = tuple(
            None if edge is None else tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in target_edges
        )
        if len(targets) != len(patches):
            raise ValueError("target_edges must match the patch count")
    if bridge_lengths is None:
        bridges = tuple(0 for _ in patches)
    else:
        bridges = tuple(int(value) for value in bridge_lengths)
        if len(bridges) != len(patches) or any(value < 0 for value in bridges):
            raise ValueError("bridge_lengths must be non-negative and match patches")
    transitions = tuple(_transition(left, right) for left, right in zip(patches, patches[1:]))
    return PatchCover(
        method=method,
        patches=patches,
        transitions=transitions,
        segment_ids=segments,
        target_edges=targets,
        bridge_lengths=bridges,
    )


def sample_independent_walk_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
) -> PatchCover:
    """Independent rooted walks, retaining the historical first-discovery slots."""
    validate_simple_adjacency(adjacency)
    if n_patches < 1:
        raise ValueError("n_patches must be positive")
    n_nodes = adjacency.shape[0]
    roots: list[int] = []
    while len(roots) < n_patches:
        roots.extend(int(node) for node in rng.permutation(n_nodes))
    patches = []
    for root in roots[:n_patches]:
        sampled = sample_walk_patch(adjacency, root, rng, patch_size=patch_size)
        patches.append(_make_patch(adjacency, sampled.node_ids, root))
    return _make_cover("independent_walk", patches)


def _distances_to_set(adjacency: np.ndarray, targets: set[int]) -> np.ndarray:
    n_nodes = adjacency.shape[0]
    distance = np.full(n_nodes, n_nodes + 1, dtype=np.int64)
    frontier = list(targets)
    for node in frontier:
        distance[node] = 0
    cursor = 0
    while cursor < len(frontier):
        node = frontier[cursor]
        cursor += 1
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if distance[value] > distance[node] + 1:
                distance[value] = distance[node] + 1
                frontier.append(value)
    return distance


def _extend_walk_patch(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    current: int,
    retained: Sequence[int],
    forbidden: set[int],
    patch_size: int,
    globally_seen: set[int],
    max_steps: int = 5000,
) -> tuple[list[int], int]:
    """Continue one walk until retained nodes are extended to a full patch."""
    nodes = list(dict.fromkeys(int(node) for node in retained))
    node_set = set(nodes)
    n_nodes = adjacency.shape[0]
    for _ in range(max_steps):
        if len(nodes) == patch_size:
            return nodes, current
        neighbors = [int(node) for node in np.flatnonzero(adjacency[current])]
        globally_new = [
            node for node in neighbors if node not in globally_seen and node not in forbidden
        ]
        locally_new = [
            node for node in neighbors if node not in node_set and node not in forbidden
        ]
        if globally_new:
            next_node = int(rng.choice(globally_new))
        elif locally_new:
            next_node = int(rng.choice(locally_new))
        else:
            eligible = set(range(n_nodes)) - node_set - forbidden
            if not eligible:
                raise RuntimeError("no eligible node remains for a sliding patch")
            distances = _distances_to_set(adjacency, eligible)
            best = min(int(distances[node]) for node in neighbors)
            choices = [node for node in neighbors if distances[node] == best]
            next_node = int(rng.choice(choices))
        current = next_node
        if current not in node_set and current not in forbidden:
            nodes.append(current)
            node_set.add(current)
            globally_seen.add(current)
    raise RuntimeError("sliding walk could not complete a patch within max_steps")


def sample_sliding_walk_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    """Slice one graph-level coverage walk into overlapping node windows."""
    validate_simple_adjacency(adjacency)
    if not 0 < target_overlap < patch_size:
        raise ValueError("target_overlap must be between zero and patch_size")
    if n_patches < 1:
        raise ValueError("n_patches must be positive")
    start = int(rng.integers(adjacency.shape[0]))
    globally_seen = {start}
    first, current = _extend_walk_patch(
        adjacency,
        rng,
        current=start,
        retained=[start],
        forbidden=set(),
        patch_size=patch_size,
        globally_seen=globally_seen,
    )
    windows = [tuple(first)]
    while len(windows) < n_patches:
        previous = windows[-1]
        retained = previous[-target_overlap:]
        forbidden = set(previous) - set(retained)
        current = retained[-1]
        window, current = _extend_walk_patch(
            adjacency,
            rng,
            current=current,
            retained=retained,
            forbidden=forbidden,
            patch_size=patch_size,
            globally_seen=globally_seen,
        )
        windows.append(tuple(window))
    patches = [_make_patch(adjacency, window, window[0]) for window in windows]
    return _make_cover("sliding_walk", patches)


def _candidate_score(
    adjacency: np.ndarray,
    candidate: int,
    current_nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    new_pairs = sum(
        tuple(sorted((candidate, node))) not in observed_pairs
        for node in current_nodes
        if node != candidate
    )
    new_edges = sum(
        adjacency[candidate, node] != 0
        and tuple(sorted((candidate, node))) not in covered_edges
        for node in current_nodes
        if node != candidate
    )
    boundary_degree = int(np.count_nonzero(adjacency[candidate]))
    return (int(new_pairs), int(new_edges), int(candidate not in covered_nodes), boundary_degree)


def _select_best(
    candidates: Iterable[int],
    score,
    rng: np.random.Generator,
) -> int:
    candidates = tuple(sorted(set(int(node) for node in candidates)))
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    scored = [(score(node), node) for node in candidates]
    best = max(item[0] for item in scored)
    choices = [node for value, node in scored if value == best]
    return int(rng.choice(choices))


def _observed_pairs_from_nodes(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _frontier_expand(
    adjacency: np.ndarray,
    seed_nodes: Sequence[int],
    rng: np.random.Generator,
    *,
    patch_size: int,
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
    forbidden: set[int] | None = None,
) -> list[int]:
    nodes = list(dict.fromkeys(int(node) for node in seed_nodes))
    forbidden = set() if forbidden is None else set(forbidden)
    n_nodes = adjacency.shape[0]
    while len(nodes) < patch_size:
        node_set = set(nodes)
        candidates = {
            int(neighbor)
            for node in nodes
            for neighbor in np.flatnonzero(adjacency[node])
            if int(neighbor) not in node_set and int(neighbor) not in forbidden
        }
        if not candidates:
            distances = _distances_to_set(adjacency, node_set)
            eligible = [
                node for node in range(n_nodes) if node not in node_set and node not in forbidden
            ]
            if not eligible:
                raise RuntimeError("no eligible frontier node remains")
            minimum = min(int(distances[node]) for node in eligible)
            candidates = {
                node for node in eligible if distances[node] == minimum
            }
        selected = _select_best(
            candidates,
            lambda candidate: _candidate_score(
                adjacency,
                candidate,
                nodes,
                observed_pairs,
                covered_nodes,
                covered_edges,
            ),
            rng,
        )
        nodes.append(selected)
    return nodes


def sample_frontier_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    """Build a continuous cover by retaining boundary nodes and expanding outward."""
    validate_simple_adjacency(adjacency)
    if not is_connected(adjacency):
        raise ValueError("frontier cover currently requires a connected graph")
    if not 0 < target_overlap < patch_size:
        raise ValueError("target_overlap must be between zero and patch_size")
    if n_patches < 1:
        raise ValueError("n_patches must be positive")

    observed_pairs: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    covered_edges: set[tuple[int, int]] = set()
    start = int(rng.integers(adjacency.shape[0]))
    first_nodes = _frontier_expand(
        adjacency,
        [start],
        rng,
        patch_size=patch_size,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    patches: list[OrderedPatch] = []

    def register(nodes: Sequence[int]) -> None:
        pairs = _observed_pairs_from_nodes(nodes)
        observed_pairs.update(pairs)
        covered_nodes.update(nodes)
        covered_edges.update(pair for pair in pairs if adjacency[pair[0], pair[1]] != 0)

    patches.append(_make_patch(adjacency, first_nodes, first_nodes[0]))
    register(first_nodes)

    for _ in range(1, n_patches):
        previous = patches[-1]
        previous_nodes = list(previous.node_ids)
        outside_degree = {
            node: int(
                np.count_nonzero(
                    adjacency[node, [other for other in range(adjacency.shape[0]) if other not in previous.node_ids]]
                )
            )
            for node in previous_nodes
        }
        retained: list[int] = []
        remaining = set(previous_nodes)
        while len(retained) < target_overlap:
            selected = _select_best(
                remaining,
                lambda node: (
                    outside_degree[node],
                    int(node not in covered_nodes),
                    int(np.count_nonzero(adjacency[node])),
                ),
                rng,
            )
            retained.append(selected)
            remaining.remove(selected)

        nodes = _frontier_expand(
            adjacency,
            retained,
            rng,
            patch_size=patch_size,
            observed_pairs=observed_pairs,
            covered_nodes=covered_nodes,
            covered_edges=covered_edges,
            forbidden=set(previous_nodes) - set(retained),
        )
        new_nodes = [node for node in nodes if node not in retained]
        center = new_nodes[0] if new_nodes else retained[0]
        patches.append(_make_patch(adjacency, nodes, center))
        register(nodes)
    return _make_cover("frontier_cover", patches)


def _true_edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left in range(adjacency.shape[0])
        for right in range(left + 1, adjacency.shape[0])
        if adjacency[left, right] != 0
    }


def _register_nodes(
    adjacency: np.ndarray,
    nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> None:
    pairs = _observed_pairs_from_nodes(nodes)
    observed_pairs.update(pairs)
    covered_nodes.update(nodes)
    covered_edges.update(pair for pair in pairs if adjacency[pair[0], pair[1]] != 0)


def _uncovered_edge_deficits(
    n_nodes: int,
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
) -> np.ndarray:
    deficits = np.zeros(n_nodes, dtype=np.int64)
    for left, right in true_edges - covered_edges:
        deficits[left] += 1
        deficits[right] += 1
    return deficits


def _ordered_uncovered_edges(
    adjacency: np.ndarray,
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    deficits = _uncovered_edge_deficits(adjacency.shape[0], true_edges, covered_edges)
    buckets: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for edge in true_edges - covered_edges:
        left, right = edge
        key = (
            int(deficits[left] + deficits[right]),
            int(left not in covered_nodes) + int(right not in covered_nodes),
            int(min(deficits[left], deficits[right])),
        )
        buckets.setdefault(key, []).append(edge)
    ordered: list[tuple[int, int]] = []
    for key in sorted(buckets, reverse=True):
        values = sorted(buckets[key])
        permutation = rng.permutation(len(values))
        ordered.extend(values[int(index)] for index in permutation)
    return ordered


def _shortest_path_avoiding(
    adjacency: np.ndarray,
    start: int,
    target: int,
    blocked: set[int],
) -> list[int] | None:
    if start == target:
        return [int(start)]
    parent = {int(start): -1}
    queue = [int(start)]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for raw_neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(raw_neighbor)
            if neighbor in blocked or neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == target:
                path = [neighbor]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    return None


def _choose_target_bundle(
    adjacency: np.ndarray,
    previous_nodes: Sequence[int],
    true_edges: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
    *,
    new_slot_count: int,
) -> tuple[tuple[int, int], int, list[int], int]:
    """Choose target edge, retained anchor, new bridge bundle, and bridge length."""
    previous_set = set(previous_nodes)
    for edge in _ordered_uncovered_edges(
        adjacency, true_edges, covered_edges, covered_nodes, rng
    ):
        if edge[0] in previous_set or edge[1] in previous_set:
            continue
        feasible: list[tuple[int, int, list[int], int]] = []
        for orientation in (edge, (edge[1], edge[0])):
            target, partner = orientation
            for anchor in previous_nodes:
                blocked = previous_set - {int(anchor)}
                path = _shortest_path_avoiding(
                    adjacency, int(anchor), int(target), blocked
                )
                if path is None:
                    continue
                bundle = list(dict.fromkeys(path[1:] + [int(partner)]))
                if len(bundle) <= new_slot_count:
                    feasible.append((len(bundle), int(anchor), bundle, len(path) - 1))
        if feasible:
            shortest = min(item[0] for item in feasible)
            choices = [item for item in feasible if item[0] == shortest]
            selected = choices[int(rng.integers(len(choices)))]
            _, anchor, bundle, bridge_length = selected
            return edge, anchor, bundle, bridge_length
    raise RuntimeError("no feasible uncovered target edge fits the next patch")


def _select_retained_for_bundle(
    adjacency: np.ndarray,
    previous_nodes: Sequence[int],
    anchor: int,
    bundle: Sequence[int],
    rng: np.random.Generator,
    *,
    target_overlap: int,
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> list[int]:
    retained = [int(anchor)]
    remaining = set(int(node) for node in previous_nodes) - {int(anchor)}
    while len(retained) < target_overlap:
        connected = {
            node
            for node in remaining
            if any(adjacency[node, kept] != 0 for kept in retained)
        }
        candidates = connected or remaining
        selected = _select_best(
            candidates,
            lambda node: _candidate_score(
                adjacency,
                node,
                retained + list(bundle),
                observed_pairs,
                covered_nodes,
                covered_edges,
            ),
            rng,
        )
        retained.append(selected)
        remaining.remove(selected)
    return retained


def _target_seed_patch(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    patch_size: int,
    true_edges: set[tuple[int, int]],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[OrderedPatch, tuple[int, int]]:
    ordered_targets = _ordered_uncovered_edges(
        adjacency, true_edges, covered_edges, covered_nodes, rng
    )
    if not ordered_targets:
        ordered_targets = sorted(true_edges)
    target = ordered_targets[0]
    nodes = _frontier_expand(
        adjacency,
        list(target),
        rng,
        patch_size=patch_size,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    return _make_patch(adjacency, nodes, target[0]), target


def _target_bridge_next_patch(
    adjacency: np.ndarray,
    previous: OrderedPatch,
    rng: np.random.Generator,
    *,
    patch_size: int,
    target_overlap: int,
    true_edges: set[tuple[int, int]],
    observed_pairs: set[tuple[int, int]],
    covered_nodes: set[int],
    covered_edges: set[tuple[int, int]],
) -> tuple[OrderedPatch, tuple[int, int], int]:
    target, anchor, bundle, bridge_length = _choose_target_bundle(
        adjacency,
        previous.node_ids,
        true_edges,
        covered_edges,
        covered_nodes,
        rng,
        new_slot_count=patch_size - target_overlap,
    )
    retained = _select_retained_for_bundle(
        adjacency,
        previous.node_ids,
        anchor,
        bundle,
        rng,
        target_overlap=target_overlap,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    forbidden = set(previous.node_ids) - set(retained)
    nodes = _frontier_expand(
        adjacency,
        retained + bundle,
        rng,
        patch_size=patch_size,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
        forbidden=forbidden,
    )
    return _make_patch(adjacency, nodes, target[0]), target, bridge_length


def sample_edge_target_bridge_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    """One continuous chain guided toward globally under-covered target edges."""
    validate_simple_adjacency(adjacency)
    if not is_connected(adjacency):
        raise ValueError("target bridge cover requires a connected graph")
    if not 0 < target_overlap < patch_size:
        raise ValueError("target_overlap must be between zero and patch_size")
    true_edges = _true_edge_set(adjacency)
    observed_pairs: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    covered_edges: set[tuple[int, int]] = set()
    patches: list[OrderedPatch] = []
    targets: list[tuple[int, int]] = []
    bridges: list[int] = []

    first, target = _target_seed_patch(
        adjacency,
        rng,
        patch_size=patch_size,
        true_edges=true_edges,
        observed_pairs=observed_pairs,
        covered_nodes=covered_nodes,
        covered_edges=covered_edges,
    )
    patches.append(first)
    targets.append(target)
    bridges.append(0)
    _register_nodes(
        adjacency, first.node_ids, observed_pairs, covered_nodes, covered_edges
    )
    while len(patches) < n_patches:
        patch, target, bridge_length = _target_bridge_next_patch(
            adjacency,
            patches[-1],
            rng,
            patch_size=patch_size,
            target_overlap=target_overlap,
            true_edges=true_edges,
            observed_pairs=observed_pairs,
            covered_nodes=covered_nodes,
            covered_edges=covered_edges,
        )
        patches.append(patch)
        targets.append(target)
        bridges.append(bridge_length)
        _register_nodes(
            adjacency, patch.node_ids, observed_pairs, covered_nodes, covered_edges
        )
    return _make_cover(
        "edge_target_bridge",
        patches,
        target_edges=targets,
        bridge_lengths=bridges,
    )


def sample_multi_chain_target_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
    segment_length: int = 4,
) -> PatchCover:
    """Target-edge cover with fixed-length continuous segments and global resets."""
    validate_simple_adjacency(adjacency)
    if segment_length < 2:
        raise ValueError("segment_length must be at least two")
    true_edges = _true_edge_set(adjacency)
    observed_pairs: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    covered_edges: set[tuple[int, int]] = set()
    patches: list[OrderedPatch] = []
    targets: list[tuple[int, int]] = []
    bridges: list[int] = []
    segments: list[int] = []

    for patch_index in range(n_patches):
        segment_id = patch_index // segment_length
        segment_start = patch_index % segment_length == 0
        if segment_start:
            patch, target = _target_seed_patch(
                adjacency,
                rng,
                patch_size=patch_size,
                true_edges=true_edges,
                observed_pairs=observed_pairs,
                covered_nodes=covered_nodes,
                covered_edges=covered_edges,
            )
            bridge_length = 0
        else:
            patch, target, bridge_length = _target_bridge_next_patch(
                adjacency,
                patches[-1],
                rng,
                patch_size=patch_size,
                target_overlap=target_overlap,
                true_edges=true_edges,
                observed_pairs=observed_pairs,
                covered_nodes=covered_nodes,
                covered_edges=covered_edges,
            )
        patches.append(patch)
        targets.append(target)
        bridges.append(bridge_length)
        segments.append(segment_id)
        _register_nodes(
            adjacency, patch.node_ids, observed_pairs, covered_nodes, covered_edges
        )
    return _make_cover(
        "multi_chain_target",
        patches,
        segments,
        targets,
        bridges,
    )


def _shortest_path_distances(adjacency: np.ndarray, source: int) -> np.ndarray:
    return _distances_to_set(adjacency, {int(source)})


def audit_cover(adjacency: np.ndarray, cover: PatchCover) -> dict[str, Any]:
    """Measure continuity, coverage, and exact stitching of an ordered cover."""
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    true_edges = {
        (left, right)
        for left in range(n_nodes)
        for right in range(left + 1, n_nodes)
        if adjacency[left, right] != 0
    }
    all_pairs = set(combinations(range(n_nodes), 2))
    observed: dict[tuple[int, int], list[int]] = {}
    covered_nodes: set[int] = set()
    for patch in cover.patches:
        covered_nodes.update(patch.node_ids)
        for left_slot, right_slot in combinations(range(len(patch.node_ids)), 2):
            pair = tuple(sorted((patch.node_ids[left_slot], patch.node_ids[right_slot])))
            observed.setdefault(pair, []).append(int(patch.adjacency[left_slot, right_slot]))

    observed_pairs = set(observed)
    observed_edges = true_edges & observed_pairs
    conflicts = sum(len(set(values)) > 1 for values in observed.values())
    observed_correct = sum(
        int(round(float(np.mean(values)))) == int(adjacency[pair[0], pair[1]])
        for pair, values in observed.items()
    )
    predicted_edges = {
        pair for pair, values in observed.items() if float(np.mean(values)) >= 0.5
    }
    full_correct = len((predicted_edges & true_edges) | ((all_pairs - predicted_edges) & (all_pairs - true_edges)))

    consecutive_overlap = []
    consecutive_jaccard = []
    center_distances = []
    new_nodes = []
    new_pairs = []
    running_nodes: set[int] = set()
    running_pairs: set[tuple[int, int]] = set()
    patch_connected = []
    persistent_slot_matches = 0
    persistent_slot_trials = 0
    for index, patch in enumerate(cover.patches):
        node_set = set(patch.node_ids)
        patch_pairs = _observed_pairs_from_nodes(patch.node_ids)
        new_nodes.append(len(node_set - running_nodes))
        new_pairs.append(len(patch_pairs - running_pairs))
        running_nodes.update(node_set)
        running_pairs.update(patch_pairs)
        patch_connected.append(is_connected(patch.adjacency))
        if index > 0 and cover.segment_ids[index] == cover.segment_ids[index - 1]:
            previous = set(cover.patches[index - 1].node_ids)
            overlap = len(previous & node_set)
            consecutive_overlap.append(overlap)
            consecutive_jaccard.append(overlap / len(previous | node_set))
            distances = _shortest_path_distances(adjacency, cover.patches[index - 1].center)
            center_distances.append(int(distances[patch.center]))
            for left_slot, right_slot in cover.transitions[index - 1].left_to_right_slots:
                persistent_slot_trials += 1
                persistent_slot_matches += int(left_slot == right_slot)

    nonconsecutive_overlap = []
    nonconsecutive_jaccard = []
    for left in range(len(cover.patches)):
        for right in range(left + 1, len(cover.patches)):
            if (
                right == left + 1
                and cover.segment_ids[left] == cover.segment_ids[right]
            ):
                continue
            left_nodes = set(cover.patches[left].node_ids)
            right_nodes = set(cover.patches[right].node_ids)
            overlap = len(left_nodes & right_nodes)
            nonconsecutive_overlap.append(overlap)
            nonconsecutive_jaccard.append(overlap / len(left_nodes | right_nodes))

    multiplicities = np.asarray([len(observed[edge]) for edge in observed_edges], dtype=np.float64)
    mean_multiplicity = float(np.mean(multiplicities)) if multiplicities.size else 0.0
    cv_multiplicity = (
        float(np.std(multiplicities) / mean_multiplicity) if mean_multiplicity > 0 else 0.0
    )

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "method": cover.method,
        "patch_count": len(cover.patches),
        "patch_size": len(cover.patches[0].node_ids),
        "node_coverage": len(covered_nodes) / n_nodes,
        "true_edge_coverage": len(observed_edges) / max(len(true_edges), 1),
        "node_pair_coverage": len(observed_pairs) / len(all_pairs),
        "observed_pair_consistency": 1.0 - conflicts / max(len(observed_pairs), 1),
        "observed_pair_accuracy": observed_correct / max(len(observed_pairs), 1),
        "full_adjacency_accuracy": full_correct / len(all_pairs),
        "patch_connected_rate": float(np.mean(patch_connected)),
        "segment_count": int(max(cover.segment_ids) + 1),
        "continuous_transition_fraction": float(
            np.mean(
                [
                    cover.segment_ids[index] == cover.segment_ids[index - 1]
                    for index in range(1, len(cover.patches))
                ]
            )
        )
        if len(cover.patches) > 1
        else 1.0,
        "continuous_shared_slot_persistence_rate": float(
            persistent_slot_matches / persistent_slot_trials
        )
        if persistent_slot_trials
        else 1.0,
        "target_edge_hit_rate": float(
            np.mean(
                [
                    edge is None or set(edge).issubset(patch.node_ids)
                    for patch, edge in zip(cover.patches, cover.target_edges)
                ]
            )
        ),
        "bridge_length_mean": float(np.mean(cover.bridge_lengths)),
        "bridge_length_maximum": int(max(cover.bridge_lengths)),
        "edge_observation_multiplicity_mean": mean_multiplicity,
        "edge_observation_multiplicity_cv": cv_multiplicity,
        "consecutive_overlap_mean": mean(consecutive_overlap),
        "consecutive_jaccard_mean": mean(consecutive_jaccard),
        "nonconsecutive_overlap_mean": mean(nonconsecutive_overlap),
        "nonconsecutive_jaccard_mean": mean(nonconsecutive_jaccard),
        "consecutive_nonconsecutive_jaccard_gap": mean(consecutive_jaccard)
        - mean(nonconsecutive_jaccard),
        "within_segment_overlap_mean": mean(consecutive_overlap),
        "within_segment_jaccard_mean": mean(consecutive_jaccard),
        "within_segment_nonlocal_jaccard_gap": mean(consecutive_jaccard)
        - mean(nonconsecutive_jaccard),
        "consecutive_center_distance_mean": mean(center_distances),
        "new_nodes_per_patch_mean": mean(new_nodes),
        "new_pairs_per_patch_mean": mean(new_pairs),
        "transition_overlap_exact": [transition.overlap_size for transition in cover.transitions],
    }


def mapped_replay_relabel_invariance(
    adjacency: np.ndarray,
    cover: PatchCover,
    permutation: np.ndarray,
) -> dict[str, Any]:
    """Replay an abstract cover after relabeling and compare patches/transitions."""
    relabeled_patches = []
    for patch in cover.patches:
        relabeled, mapped_order = relabel_adjacency_and_order(
            adjacency, patch.node_ids, permutation
        )
        mapped_center = mapped_order[patch.node_ids.index(patch.center)]
        relabeled_patches.append(_make_patch(relabeled, mapped_order, mapped_center))
    permutation = np.asarray(permutation, dtype=np.int64)
    inverse = np.empty(permutation.size, dtype=np.int64)
    inverse[permutation] = np.arange(permutation.size)
    mapped_targets = [
        None
        if edge is None
        else (int(inverse[edge[0]]), int(inverse[edge[1]]))
        for edge in cover.target_edges
    ]
    replay = _make_cover(
        cover.method,
        relabeled_patches,
        cover.segment_ids,
        mapped_targets,
        cover.bridge_lengths,
    )
    adjacency_matches = [
        np.array_equal(left.adjacency, right.adjacency)
        for left, right in zip(cover.patches, replay.patches)
    ]
    transition_matches = [
        left.left_to_right_slots == right.left_to_right_slots
        for left, right in zip(cover.transitions, replay.transitions)
    ]
    return {
        "patch_adjacency_match_rate": float(np.mean(adjacency_matches)),
        "transition_slot_map_match_rate": float(np.mean(transition_matches))
        if transition_matches
        else 1.0,
    }


def remap_cover(
    cover: PatchCover,
    node_mapping: np.ndarray,
    target_adjacency: np.ndarray,
) -> PatchCover:
    """Map cover node ids into a target graph while retaining local slot order."""
    validate_simple_adjacency(target_adjacency)
    node_mapping = np.asarray(node_mapping, dtype=np.int64)
    if node_mapping.shape != (target_adjacency.shape[0],):
        raise ValueError("node_mapping must have one entry per target graph node")
    patches = []
    for patch in cover.patches:
        mapped_nodes = tuple(int(node_mapping[node]) for node in patch.node_ids)
        mapped_center = int(node_mapping[patch.center])
        patches.append(_make_patch(target_adjacency, mapped_nodes, mapped_center))
    mapped_targets = [
        None
        if edge is None
        else (int(node_mapping[edge[0]]), int(node_mapping[edge[1]]))
        for edge in cover.target_edges
    ]
    return _make_cover(
        cover.method,
        patches,
        cover.segment_ids,
        mapped_targets,
        cover.bridge_lengths,
    )


def cover_set_similarity(left: PatchCover, right: PatchCover) -> float:
    """Position-wise patch-set Jaccard for two equal-budget covers."""
    if len(left.patches) != len(right.patches):
        raise ValueError("covers must have the same patch count")
    similarities = []
    for left_patch, right_patch in zip(left.patches, right.patches):
        left_nodes = set(left_patch.node_ids)
        right_nodes = set(right_patch.node_ids)
        similarities.append(len(left_nodes & right_nodes) / len(left_nodes | right_nodes))
    return float(np.mean(similarities))


def make_slot_persistent_cover(
    adjacency: np.ndarray,
    cover: PatchCover,
    *,
    method: str = "slot_persistent",
) -> PatchCover:
    """Reorder frozen node sets so shared nodes retain slots within each segment."""
    validate_simple_adjacency(adjacency)
    reordered: list[OrderedPatch] = []
    for index, patch in enumerate(cover.patches):
        segment_start = index == 0 or cover.segment_ids[index] != cover.segment_ids[index - 1]
        if segment_start:
            nodes = list(patch.node_ids)
        else:
            previous = reordered[-1]
            current_set = set(patch.node_ids)
            slots: list[int | None] = [None] * len(patch.node_ids)
            shared = set(previous.node_ids) & current_set
            for slot, node in enumerate(previous.node_ids):
                if node in shared:
                    slots[slot] = int(node)
            new_nodes = [node for node in patch.node_ids if node not in shared]
            new_cursor = 0
            for slot, value in enumerate(slots):
                if value is None:
                    slots[slot] = int(new_nodes[new_cursor])
                    new_cursor += 1
            if new_cursor != len(new_nodes):
                raise RuntimeError("persistent slot fill did not consume all new nodes")
            nodes = [int(node) for node in slots if node is not None]
        reordered.append(_make_patch(adjacency, nodes, patch.center))
    return _make_cover(
        method,
        reordered,
        cover.segment_ids,
        cover.target_edges,
        cover.bridge_lengths,
    )


def cover_vectors(cover: PatchCover) -> np.ndarray:
    """Return ordered induced-adjacency vectors for later KSVD stages."""
    return np.stack([adjacency_to_upper_vector(patch.adjacency) for patch in cover.patches])
