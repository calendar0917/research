"""ZINC patch--path pooling with one explicit, non-message-passing MLP.

This is the next representation/readout candidate after the dense canonical
patch experiment.  It keeps an exact rooted typed patch as a train-only
categorical token, adds a small continuous shell descriptor for unseen/OOV
patches, and forms a symmetric relation object for every unordered pair of
centres.  Pair interactions are conditioned on the shortest-path relation
*before* graph pooling.  Pair outputs are then pooled separately by distance
bucket using first/second moments and pair mass.

There is no centre-to-centre message passing: a patch state is computed once,
and each pair is evaluated independently before the invariant readout.  The
only predictive head is the final MLP, which is trained end-to-end with the
patch encoder and pair function.

The official validation phase fits token vocabularies and continuous
standardisation on official train only.  The test phase refits those
label-free transforms on train+validation after the validation-selected epoch
has been frozen.  The official test labels never affect representation or
epoch selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16.structural_context import (
    COARSE_WIDTH,
    NO_RING_TOKEN,
    UNK_CONTEXT_TOKEN,
    coarsen,
    context_statistics,
    cycle_length_distribution,
    encode_structural_token,
    extract_structural_context,
    fit_structural_vocabulary,
)
from tracks.ksvd.experiments.luyin16.zinc_exact_patch_relation import (
    _patch_cache_key,
)
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    DEFAULT_TYPED_TOKENIZER_VERSION,
    TYPED_TOKENIZER_V1_HISTORICAL,
    build_colored_incidence,
    corrected_canonical_key,
    historical_certificate,
    resolve_typed_tokenizer_version,
    typed_tokenizer_fingerprint,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_patch_path_pooling.yaml"

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
PATCH_RADIUS = 2
DISTANCE_BUCKETS = 5  # 1, 2, 3, 4, 5+
SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
CONTEXT_RADIUS = 3

# Continuous outer-shell context used by the optional multiscale candidate.
# It records the atom composition of shell 3, bond composition for shell-2/3
# and shell-3/3 edges, and six invariant mass/degree quantities.  Radius-2
# exact token identity remains the categorical structural object.
CONTEXT_WIDTH = ATOM_CATEGORIES + 2 * BOND_CATEGORIES + 6

# Shell-wise atom proportions (3 x 28), shell-pair bond proportions (6 x 4),
# root/incident chemistry (28 + 4), and a few size/cycle/degree scalars.
SHELL_WIDTH = 3 * ATOM_CATEGORIES + len(SHELL_PAIRS) * BOND_CATEGORIES + 28 + 4 + 6
RELATION_WIDTH = (
    DISTANCE_BUCKETS  # distance bucket one-hot
    + 1  # log shortest-path distance
    + 5  # patch overlap and size relation
    + 3  # boundary overlap and centre containment
    + BOND_CATEGORIES  # bond composition averaged over shortest paths
    + 1  # log number of shortest paths
    + BOND_CATEGORIES  # adjacent bond type, zero for non-adjacent pairs
)
PATCH_HIDDEN = 64
PAIR_HIDDEN = 32
GLOBAL_WIDTH = 62  # global_feature_views(...)["global_all"]

# --- Compact-v5 multi-quantile regression (objective-only change) ---------
# The compact-v4 representation/graph head hidden structure is frozen; v5
# only changes the regression output parameterization and the training loss.
#
#   quantile_mode="none"        exact compact-v4 scalar output + L1 (guard)
#   quantile_mode="median_only" scalar q50 (= m) + loss 2*pinball_0.5 == L1
#   quantile_mode="q10_q50_q90" non-crossing q10/q50/q90 readout + combined
#                               quantile loss (q50 stays the point prediction)
#
# Loss scale rule: 2 * pinball(y-q, 0.5) == abs(y-q) exactly (both halves of
# the pinball are ±0.5*e, so doubling recovers |e| bit-for-bit).  This keeps
# the q50 main-task gradient magnitude identical to the original compact-v4 L1.
QUANTILE_MODES = ("none", "median_only", "q10_q50_q90")
QUANTILE_TAUS = (0.10, 0.50, 0.90)
# Pre-registered lambda grid for stage 1 (objective-lottery protection):
# only these three values may be run for quantile_mode="q10_q50_q90".
PREREGISTERED_QUANTILE_LAMBDAS = (0.10, 0.25, 0.50)
DEFAULT_QUANTILE_LAMBDA = 0.25


def pinball_loss(error: torch.Tensor, tau: float) -> torch.Tensor:
    """Elementwise pinball (quantile) loss for one quantile level tau.

    With ``error = y - q_tau``:

        L_tau(e) = max(tau * e, (tau - 1) * e)

    For tau = 0.5 this equals 0.5 * |e|, so ``2 * pinball_loss(e, 0.5)`` is
    exactly ``abs(e)`` (both branches are exact powers-of-two scalings of e).
    """
    tau = float(tau)
    return torch.maximum(tau * error, (tau - 1.0) * error)


def _validate_quantile_config(
    quantile_mode: str,
    quantile_lambda: float | None,
) -> float | None:
    """Validate a (mode, lambda) pair; returns the effective lambda."""
    if quantile_mode not in QUANTILE_MODES:
        raise ValueError(
            f"unknown quantile_mode={quantile_mode!r}; expected one of "
            f"{sorted(QUANTILE_MODES)}"
        )
    if quantile_mode == "q10_q50_q90":
        lambda_value = float(
            DEFAULT_QUANTILE_LAMBDA
            if quantile_lambda is None
            else quantile_lambda
        )
        # Stage-1 pre-registration: do not open the lambda grid.  Only the
        # three declared values may run for the full three-quantile model.
        if lambda_value not in PREREGISTERED_QUANTILE_LAMBDAS:
            raise ValueError(
                f"quantile_lambda={lambda_value} not in pre-registered grid "
                f"{PREREGISTERED_QUANTILE_LAMBDAS}; re-register a new stage "
                "before widening the grid"
            )
        return lambda_value
    return None


def _quantile_point_prediction_torch(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """The benchmark point estimate: q50 (column 1) or the scalar output."""
    if prediction.ndim == 2:
        return prediction[:, 1]
    return prediction


def _quantile_point_prediction_numpy(
    predictions: np.ndarray,
) -> np.ndarray:
    """Numpy variant of the q50 point-prediction extraction."""
    if predictions.ndim == 2:
        return predictions[:, 1]
    return predictions


def quantile_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile_mode: str = "none",
    quantile_lambda: float | None = None,
) -> torch.Tensor:
    """Training loss for the requested output/objective mode.

    ``target`` is always 1-D per-molecule; ``prediction`` is 1-D for
    ``none``/``median_only`` and (n, 3) for ``q10_q50_q90`` with columns
    [q10, q50, q90].  ``quantile_mode="none"`` returns exactly
    ``F.l1_loss(prediction, target)`` so the compact-v4 path is untouched.
    """
    lambda_value = _validate_quantile_config(quantile_mode, quantile_lambda)
    if quantile_mode == "none":
        return F.l1_loss(prediction, target)
    if quantile_mode == "median_only":
        # 2 * pinball_0.5 == |y - q50|; forward-equal (and, with power-of-two
        # scalings only, gradient-equal) to the original L1.
        error = target - prediction
        return (2.0 * pinball_loss(error, 0.5)).mean()
    # q10_q50_q90: q50 (median) is the main task at L1 scale; q10/q90 are
    # auxiliary distributional tasks weighted by lambda.
    q10 = prediction[:, 0]
    q50 = prediction[:, 1]
    q90 = prediction[:, 2]
    error_mid = target - q50
    main = (2.0 * pinball_loss(error_mid, 0.5)).mean()
    assert lambda_value is not None
    auxiliary = pinball_loss(target - q10, 0.10).mean() + pinball_loss(
        target - q90, 0.90
    ).mean()
    return main + lambda_value * auxiliary



def _shell_pairs_for_radius(radius: int) -> tuple[tuple[int, int], ...]:
    radius = int(radius)
    if radius < 1:
        raise ValueError(f"radius must be positive, got {radius}")
    return tuple(
        (left, right)
        for left in range(radius + 1)
        for right in range(left, radius + 1)
    )


def _shell_width_for_radius(radius: int) -> int:
    shell_pairs = _shell_pairs_for_radius(int(radius))
    return (
        (int(radius) + 1) * ATOM_CATEGORIES
        + len(shell_pairs) * BOND_CATEGORIES
        + 28
        + 4
        + 6
    )


@dataclass(frozen=True)
class PatchRecord:
    typed_certificate: bytes
    parent_certificate: bytes
    nodes: frozenset[int]
    boundary: frozenset[int]
    shell_descriptor: np.ndarray
    context_descriptor: np.ndarray | None = None
    # Compact-v3: higher-order structural context of the centre node
    # (ring/cycle conditioning signal; None in compact-v2 mode).
    structural_context: Any | None = None


@dataclass(frozen=True)
class GraphRecord:
    patches: tuple[PatchRecord, ...]
    pair_index: np.ndarray
    pair_relation: np.ndarray
    pair_bucket: np.ndarray
    global_context: np.ndarray
    y: float
    # Bounded simple-cycle lengths of this graph (structural context mode only).
    cycle_lengths: tuple[int, ...] = ()
    # Compact-v4: raw global topology features (unstandardized), one vector
    # per graph; see notes/compact_v4_global_topology_channel.md.
    topology_features: np.ndarray | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _ego_distances(graph: Any, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _shortest_path_summary(
    graph: Any,
    source: int,
    edge_types: Mapping[tuple[int, int], int],
) -> dict[int, tuple[int, float, np.ndarray]]:
    """Return distance, path count, and mean bond composition to every node.

    The bond composition is averaged over all shortest paths.  This avoids
    choosing an arbitrary node-ID-dependent path when multiple shortest paths
    exist, and keeps the relation permutation invariant.
    """
    source = int(source)
    distances = {source: 0}
    path_counts: dict[int, float] = {source: 1.0}
    bond_sums: dict[int, np.ndarray] = {
        source: np.zeros(BOND_CATEGORIES, dtype=np.float64)
    }
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.neighbors(node)):
            neighbor = int(neighbor)
            bond = int(edge_types[graph.edge_key(node, neighbor)])
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                path_counts[neighbor] = 0.0
                bond_sums[neighbor] = np.zeros(BOND_CATEGORIES, dtype=np.float64)
                queue.append(neighbor)
            if distances[neighbor] != distances[node] + 1:
                continue
            path_counts[neighbor] += path_counts[node]
            bond_sums[neighbor] += bond_sums[node]
            bond_sums[neighbor][bond] += path_counts[node]
    output: dict[int, tuple[int, float, np.ndarray]] = {}
    for node, distance in distances.items():
        count = max(float(path_counts[node]), 1.0)
        output[int(node)] = (
            int(distance),
            float(path_counts[node]),
            (bond_sums[node] / count).astype(np.float32),
        )
    return output


def _shell_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    distances: Mapping[int, int],
    radius: int,
) -> tuple[np.ndarray, frozenset[int], frozenset[int]]:
    radius = int(radius)
    shell_pairs = _shell_pairs_for_radius(radius)
    nodes = frozenset(int(node) for node in distances)
    induced = graph.induced(set(nodes))
    n_nodes = len(nodes)
    n_edges = induced.num_edges()
    atom_shell = np.zeros((radius + 1, ATOM_CATEGORIES), dtype=np.float32)
    for node in nodes:
        atom = int(node_types[node])
        shell = int(distances[node])
        if atom < 0 or atom >= ATOM_CATEGORIES:
            raise ValueError(f"atom category {atom} outside schema")
        atom_shell[shell, atom] += 1.0
    atom_shell /= max(float(n_nodes), 1.0)

    bond_shell = np.zeros((len(shell_pairs), BOND_CATEGORIES), dtype=np.float32)
    shell_pair_index = {pair: index for index, pair in enumerate(shell_pairs)}
    for left, right in induced.edges():
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        index = shell_pair_index.get(pair)
        if index is None:
            raise RuntimeError(f"unexpected shell pair {pair}")
        bond = int(edge_types[graph.edge_key(int(left), int(right))])
        if bond < 0 or bond >= BOND_CATEGORIES:
            raise ValueError(f"bond category {bond} outside schema")
        bond_shell[index, bond] += 1.0
    bond_shell /= max(float(n_edges), 1.0)

    root_atom = np.zeros(ATOM_CATEGORIES, dtype=np.float32)
    root_atom[int(node_types[int(center)])] = 1.0
    incident_bonds = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    for neighbor in graph.neighbors(int(center)):
        bond = int(edge_types[graph.edge_key(int(center), int(neighbor))])
        incident_bonds[bond] += 1.0
    incident_bonds /= max(float(len(graph.neighbors(int(center)))), 1.0)

    cycle_rank = max(int(n_edges) - int(n_nodes) + 1, 0)
    degrees = np.asarray([len(graph.neighbors(node)) for node in nodes], dtype=np.float32)
    scalars = np.asarray(
        [
            np.log1p(float(n_nodes)),
            np.log1p(float(n_edges)),
            float(len(nodes) and sum(int(distance) == radius for distance in distances.values()))
            / max(float(n_nodes), 1.0),
            float(cycle_rank) / max(float(n_nodes), 1.0),
            float(len(graph.neighbors(int(center)))) / 4.0,
            float(degrees.mean()) / 4.0 if degrees.size else 0.0,
        ],
        dtype=np.float32,
    )
    descriptor = np.concatenate(
        [atom_shell.reshape(-1), bond_shell.reshape(-1), root_atom, incident_bonds, scalars]
    ).astype(np.float32, copy=False)
    expected_width = _shell_width_for_radius(radius)
    if descriptor.shape != (expected_width,):
        raise RuntimeError(
            f"shell descriptor width changed: {descriptor.shape}; expected {(expected_width,)}"
        )
    boundary = frozenset(
        int(node) for node, distance in distances.items() if int(distance) == radius
    )
    return descriptor, nodes, boundary


def _outer_context_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    distances: Mapping[int, int],
) -> np.ndarray:
    """Return a compact invariant descriptor for the radius-3 outer shell.

    The radius-2 patch already supplies the exact rooted identity and full
    shell-0..2 histogram.  This side channel deliberately adds only new
    radius-3 information, instead of duplicating the entire radius-3 shell
    descriptor or introducing a 45k-token vocabulary.
    """
    outer = [int(node) for node, distance in distances.items() if int(distance) == 3]
    outer_set = set(outer)
    atom = np.zeros(ATOM_CATEGORIES, dtype=np.float32)
    for node in outer:
        atom[int(node_types[node])] += 1.0
    atom /= max(float(len(outer)), 1.0)

    bond_blocks = np.zeros((2, BOND_CATEGORIES), dtype=np.float32)
    edge_count = 0
    outer_edge_count = 0
    for left, right in graph.induced(set(distances)).edges():
        left_distance = int(distances[int(left)])
        right_distance = int(distances[int(right)])
        if 3 not in (left_distance, right_distance):
            continue
        bond = int(edge_types[graph.edge_key(int(left), int(right))])
        # Sort the shell labels before assigning the block.  The graph edge
        # iterator is not a semantic ordering and must not affect the
        # descriptor.
        shell_pair = tuple(sorted((left_distance, right_distance)))
        if shell_pair == (3, 3):
            block = 0
        elif shell_pair == (2, 3):
            block = 1
        else:
            raise RuntimeError(f"unexpected radius-3 edge shell pair {shell_pair}")
        bond_blocks[block, bond] += 1.0
        edge_count += 1
        outer_edge_count += int(left_distance == 3 and right_distance == 3)
    bond_blocks /= max(float(edge_count), 1.0)

    outer_degrees = np.asarray([len(graph.neighbors(node)) for node in outer], dtype=np.float32)
    scalars = np.asarray(
        [
            np.log1p(float(len(outer))),
            np.log1p(float(edge_count)),
            float(outer_edge_count) / max(float(edge_count), 1.0),
            float(len(outer)) / max(float(len(distances)), 1.0),
            float(outer_degrees.mean()) / 4.0 if outer_degrees.size else 0.0,
            float(outer_degrees.std()) / 4.0 if outer_degrees.size else 0.0,
        ],
        dtype=np.float32,
    )
    descriptor = np.concatenate([atom, bond_blocks.reshape(-1), scalars]).astype(
        np.float32, copy=False
    )
    if descriptor.shape != (CONTEXT_WIDTH,):
        raise RuntimeError(f"outer context width changed: {descriptor.shape}")
    return descriptor


def _typed_certificate(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
    cache: dict[bytes, bytes],
    tokenizer_version: str = DEFAULT_TYPED_TOKENIZER_VERSION,
) -> bytes:
    """Return a rooted typed incidence token at any radius.

    ``zinc_exact_patch_relation._canonical_typed_patch`` also emits a
    certificate, but its fixed-width descriptor intentionally rejects patches
    larger than 14 nodes.  Radius-3 ZINC patches occasionally exceed that
    width, so the scalable token path must construct only the canonical
    incidence graph and omit the fixed descriptor.

    Two tokenizer versions exist (``typed_patch_tokenizer``):

    * ``typed_tokenizer_v1_historical`` -- the historical bytes
      ``pynauty.certificate(incidence)``; retained verbatim so every historical
      run is bit-identical and reproducible.
    * ``typed_tokenizer_v2_corrected`` -- certificate **plus** the canonical
      semantic color sequence, a complete invariant of the colored incidence
      graph (atom type + bond type + root designation + root-distance class).

    The only thing that changes between versions is the byte string used as
    the token; radius, descriptor, model and loss are untouched.
    """
    version = resolve_typed_tokenizer_version(tokenizer_version)
    key = _patch_cache_key(graph, int(center), node_types, edge_types, int(radius))
    certificate = cache.get(key)
    if certificate is None:
        incidence = build_colored_incidence(
            graph, int(center), node_types, edge_types, int(radius)
        )
        if version == TYPED_TOKENIZER_V1_HISTORICAL:
            certificate = historical_certificate(incidence)
        else:
            certificate = corrected_canonical_key(incidence)
        cache[key] = certificate
    return certificate


def _pair_relation(
    left: PatchRecord,
    right: PatchRecord,
    path_summary: tuple[int, float, np.ndarray],
    adjacent_bond: int | None,
    patch_radius: int,
) -> tuple[np.ndarray, int]:
    distance, path_count, path_bond_mean = path_summary
    bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1
    distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float32)
    distance_one_hot[bucket] = 1.0

    intersection = len(left.nodes & right.nodes)
    union = len(left.nodes | right.nodes)
    left_size = len(left.nodes)
    right_size = len(right.nodes)
    overlap = np.asarray(
        [
            float(intersection) / max(float(int(patch_radius) * int(patch_radius) + 10), 1.0),
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size))
            / max(float(int(patch_radius) * int(patch_radius) + 10), 1.0),
        ],
        dtype=np.float32,
    )
    boundary_intersection = len(left.boundary & right.boundary)
    boundary_union = len(left.boundary | right.boundary)
    boundary_min = min(len(left.boundary), len(right.boundary))
    boundary_features = np.asarray(
        [
            float(boundary_intersection) / max(float(boundary_union), 1.0),
            float(boundary_intersection) / max(float(boundary_min), 1.0),
            float(boundary_intersection > 0),
        ],
        dtype=np.float32,
    )
    adjacent = np.zeros(BOND_CATEGORIES, dtype=np.float32)
    if adjacent_bond is not None:
        adjacent[int(adjacent_bond)] = 1.0
    relation = np.concatenate(
        [
            distance_one_hot,
            np.asarray([np.log1p(float(distance))], dtype=np.float32),
            overlap,
            boundary_features,
            np.asarray(path_bond_mean, dtype=np.float32),
            np.asarray([np.log1p(float(path_count))], dtype=np.float32),
            adjacent,
        ]
    ).astype(np.float32, copy=False)
    if relation.shape != (RELATION_WIDTH,):
        raise RuntimeError(f"relation width changed: {relation.shape}; expected {RELATION_WIDTH}")
    return relation, bucket


def _graph_record(
    data: Any,
    global_context: np.ndarray,
    certificate_cache: dict[bytes, bytes],
    patch_radius: int = PATCH_RADIUS,
    context_radius: int = 0,
    structural_mode: str = "none",
    max_cycle_len: int = 10,
    topology_features: np.ndarray | None = None,
    tokenizer_version: str = DEFAULT_TYPED_TOKENIZER_VERSION,
) -> GraphRecord:
    patch_radius = int(patch_radius)
    graph, node_types, edge_types = _data_to_graph(data)
    node_contexts: dict[int, Any] | None = None
    cycle_lengths: tuple[int, ...] = ()
    if str(structural_mode) != "none":
        # One bounded cycle enumeration per graph; every patch reuses it.
        node_contexts, cycle_lengths = extract_structural_context(
            graph, node_types, edge_types, max_cycle_len=int(max_cycle_len)
        )
    centers = list(graph.nodes)
    patches: list[PatchRecord] = []
    for center in centers:
        distances = _ego_distances(graph, int(center), patch_radius)
        descriptor, nodes, boundary = _shell_descriptor(
            graph, int(center), node_types, edge_types, distances, patch_radius
        )
        context_descriptor = None
        if int(context_radius) > patch_radius:
            context_distances = _ego_distances(graph, int(center), int(context_radius))
            context_descriptor = _outer_context_descriptor(
                graph, int(center), node_types, edge_types, context_distances
            )
        typed = _typed_certificate(
            graph,
            int(center),
            node_types,
            edge_types,
            patch_radius,
            certificate_cache,
            tokenizer_version,
        )
        parent = _typed_certificate(
            graph,
            int(center),
            node_types,
            edge_types,
            max(1, patch_radius - 1),
            certificate_cache,
            tokenizer_version,
        )
        patches.append(
            PatchRecord(
                typed_certificate=typed,
                parent_certificate=parent,
                nodes=nodes,
                boundary=boundary,
                shell_descriptor=descriptor,
                context_descriptor=context_descriptor,
                structural_context=(
                    node_contexts[int(center)] if node_contexts is not None else None
                ),
            )
        )

    shortest_paths = {
        int(source): _shortest_path_summary(graph, int(source), edge_types)
        for source in centers
    }
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_relations: list[np.ndarray] = []
    pair_buckets: list[int] = []
    for left_index, left_center in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right_center = int(centers[right_index])
            path_summary = shortest_paths[int(left_center)][right_center]
            distance = int(path_summary[0])
            adjacent_bond = None
            if distance == 1:
                adjacent_bond = int(edge_types[graph.edge_key(int(left_center), right_center)])
            relation, bucket = _pair_relation(
                patches[left_index],
                patches[right_index],
                path_summary,
                adjacent_bond,
                patch_radius,
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_relations.append(relation)
            pair_buckets.append(int(bucket))

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    pair_relation = np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
    pair_bucket = np.asarray(pair_buckets, dtype=np.int64)
    if pair_relation.shape[1] != RELATION_WIDTH:
        raise RuntimeError(f"pair relation matrix width changed: {pair_relation.shape}")
    return GraphRecord(
        patches=tuple(patches),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=np.asarray(global_context, dtype=np.float32),
        y=float(data.y.view(-1)[0]),
        cycle_lengths=cycle_lengths,
        topology_features=topology_features,
    )


def _extract_split(
    dataset: Any,
    split: str,
    certificate_cache: dict[bytes, bytes],
    patch_radius: int = PATCH_RADIUS,
    context_radius: int = 0,
    structural_mode: str = "none",
    max_cycle_len: int = 10,
    topology_mode: str = "none",
    topology_matrix: np.ndarray | None = None,
    tokenizer_version: str = DEFAULT_TYPED_TOKENIZER_VERSION,
) -> tuple[list[GraphRecord], dict[str, Any]]:
    started = time.perf_counter()
    contexts = global_feature_views(dataset)["global_all"]
    records: list[GraphRecord] = []
    for index, data in enumerate(dataset):
        topology_row = None
        if str(topology_mode) != "none":
            if topology_matrix is None:
                raise RuntimeError("topology_mode != none requires a topology matrix")
            topology_row = topology_matrix[index]
        records.append(
            _graph_record(
                data,
                contexts[index],
                certificate_cache,
                patch_radius=int(patch_radius),
                context_radius=int(context_radius),
                structural_mode=str(structural_mode),
                max_cycle_len=int(max_cycle_len),
                topology_features=topology_row,
                tokenizer_version=str(tokenizer_version),
            )
        )
        if index and index % 500 == 0:
            print(
                f"patch-path features {split}: {index}/{len(dataset)} "
                f"certificate_cache={len(certificate_cache)}",
                flush=True,
            )
    metadata = {
        "n_graphs": int(len(records)),
        "mean_centres": float(np.mean([len(row.patches) for row in records])),
        "mean_pairs": float(np.mean([row.pair_relation.shape[0] for row in records])),
        "mean_patch_nodes": float(
            np.mean([len(patch.nodes) for row in records for patch in row.patches])
        ),
        "max_patch_nodes": int(
            max((len(patch.nodes) for row in records for patch in row.patches), default=0)
        ),
        "context_radius": int(context_radius),
        "patch_radius": int(patch_radius),
        "context_width": int(CONTEXT_WIDTH if int(context_radius) > PATCH_RADIUS else 0),
        "structural_context_mode": str(structural_mode),
        "structural_context_max_cycle_len": int(max_cycle_len),
        "structural_context_cycles": int(
            sum(len(row.cycle_lengths) for row in records)
        ),
        "cycle_length_distribution": cycle_length_distribution(records),
        "topology_mode": str(topology_mode),
        "typed_tokenizer_version": resolve_typed_tokenizer_version(tokenizer_version),
        "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
            tokenizer_version, int(patch_radius)
        ),
        "seconds": float(time.perf_counter() - started),
    }
    return records, metadata


class Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"standardizer expects non-empty matrix, got {matrix.shape}")
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32, copy=False
        )
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized values")
        return output


def _fit_vocabulary(
    records: Sequence[GraphRecord],
    field: str,
    maximum: int,
    minimum_frequency: int,
) -> dict[bytes, int]:
    counts: Counter[bytes] = Counter()
    for record in records:
        for patch in record.patches:
            counts[getattr(patch, field)] += 1
    ordered = sorted(
        (key for key, count in counts.items() if count >= int(minimum_frequency)),
        key=lambda key: (-counts[key], key),
    )[: int(maximum)]
    # ID zero is a learned OOV token; known tokens start at one.
    return {key: index + 1 for index, key in enumerate(ordered)}


def _vocabulary_stats(
    records: Sequence[GraphRecord],
    vocabulary: Mapping[bytes, int],
    field: str,
) -> dict[str, Any]:
    values = [getattr(patch, field) for record in records for patch in record.patches]
    known = sum(value in vocabulary for value in values)
    types = set(values)
    known_types = sum(value in vocabulary for value in types)
    graph_all = sum(
        all(getattr(patch, field) in vocabulary for patch in record.patches)
        for record in records
    )
    return {
        "occurrences": int(len(values)),
        "unique_types": int(len(types)),
        "vocabulary_size": int(len(vocabulary)),
        "known_occurrence_fraction": float(known / max(len(values), 1)),
        "known_type_fraction": float(known_types / max(len(types), 1)),
        "all_patches_known_graph_fraction": float(graph_all / max(len(records), 1)),
    }


def _patch_matrix(records: Sequence[GraphRecord]) -> np.ndarray:
    return np.stack(
        [patch.shell_descriptor for record in records for patch in record.patches], axis=0
    ).astype(np.float32, copy=False)


def _context_patch_matrix(records: Sequence[GraphRecord]) -> np.ndarray:
    values = [
        patch.context_descriptor
        for record in records
        for patch in record.patches
    ]
    if not values or any(value is None for value in values):
        raise ValueError("radius-3 context descriptors are not present")
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _context_matrix(records: Sequence[GraphRecord]) -> np.ndarray:
    return np.stack([record.global_context for record in records], axis=0).astype(
        np.float32, copy=False
    )


def _encode_records(
    records: Sequence[GraphRecord],
    typed_vocabulary: Mapping[bytes, int],
    parent_vocabulary: Mapping[bytes, int],
    patch_standardizer: Standardizer,
    context_standardizer: Standardizer,
    context_patch_standardizer: Standardizer | None = None,
    structural_mode: str = "none",
    structural_vocabulary: Mapping[bytes, int] | None = None,
    topology_mode: str = "none",
    topology_standardizer: Standardizer | None = None,
) -> list[Data]:
    output: list[Data] = []
    for record in records:
        patch_cont = patch_standardizer.transform(
            np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
        )
        if context_patch_standardizer is None:
            patch_context = np.zeros(
                (len(record.patches), 0), dtype=np.float32
            )
        else:
            patch_context = context_patch_standardizer.transform(
                _context_patch_matrix([record])
            )
        typed = np.asarray(
            [typed_vocabulary.get(patch.typed_certificate, 0) for patch in record.patches],
            dtype=np.int64,
        )
        parent = np.asarray(
            [parent_vocabulary.get(patch.parent_certificate, 0) for patch in record.patches],
            dtype=np.int64,
        )
        if str(structural_mode) == "typed_ring":
            structural_token = np.asarray(
                [
                    encode_structural_token(patch.structural_context, structural_vocabulary or {})
                    for patch in record.patches
                ],
                dtype=np.int64,
            )
            structural_coarse = np.stack(
                [coarsen(patch.structural_context) for patch in record.patches], axis=0
            ).astype(np.float32, copy=False)
        elif str(structural_mode) == "coarse_ring":
            structural_token = np.zeros(len(record.patches), dtype=np.int64)
            structural_coarse = np.stack(
                [coarsen(patch.structural_context) for patch in record.patches], axis=0
            ).astype(np.float32, copy=False)
        else:
            structural_token = np.zeros(len(record.patches), dtype=np.int64)
            structural_coarse = np.zeros(
                (len(record.patches), COARSE_WIDTH), dtype=np.float32
            )
        has_topology = str(topology_mode) != "none"
        if has_topology:
            if topology_standardizer is None or record.topology_features is None:
                raise RuntimeError(
                    "topology_mode != none requires a train-fit topology standardizer"
                )
            topology = topology_standardizer.transform(
                record.topology_features[None, :]
            )
        else:
            topology = np.zeros((1, 0), dtype=np.float32)
        output.append(
            Data(
                patch_cont=torch.from_numpy(patch_cont),
                patch_context=torch.from_numpy(patch_context),
                typed_token=torch.from_numpy(typed),
                parent_token=torch.from_numpy(parent),
                structural_token=torch.from_numpy(structural_token),
                structural_coarse=torch.from_numpy(structural_coarse),
                pair_index=torch.from_numpy(record.pair_index),
                pair_relation=torch.from_numpy(record.pair_relation),
                pair_bucket=torch.from_numpy(record.pair_bucket),
                global_context=torch.from_numpy(
                    context_standardizer.transform(record.global_context[None, :])
                ),
                y=torch.tensor([record.y], dtype=torch.float32),
                num_nodes=len(record.patches),
                topology_features=torch.from_numpy(topology),
            )
        )
    return output


class _MLPBlock(nn.Module):
    def __init__(self, width: int, hidden: int, output: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(width), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(output)),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _FactorizedEmbedding(nn.Module):
    """Exact-token lookup with a low-rank parameterization.

    The lookup still has one row per exact rooted patch, so no patch identity
    is hashed or merged.  Only the embedding table is factorized as
    ``V x rank`` followed by a shared linear map to the requested output
    width.  This lets us keep a 32D token representation under the CIN-sized
    parameter budget.
    """

    def __init__(self, vocabulary_size: int, output_width: int, rank: int) -> None:
        super().__init__()
        if int(rank) < 1 or int(rank) > int(output_width):
            raise ValueError("embedding rank must be in [1, output_width]")
        self.rank = int(rank)
        self.output_width = int(output_width)
        self.embedding = nn.Embedding(int(vocabulary_size), int(rank))
        self.projection = nn.Linear(int(rank), int(output_width), bias=False)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token))


class _HybridEmbedding(nn.Module):
    """Use a full embedding for frequent exact tokens and a low-rank one for
    rare exact tokens.

    ``_fit_vocabulary`` assigns IDs in decreasing occurrence frequency and
    reserves ID zero for OOV.  Thus ``full_count`` is a parameterisation
    boundary, not a vocabulary truncation: every token keeps its own row and
    rare tokens are not merged or hashed.
    """

    def __init__(
        self,
        vocabulary_size: int,
        output_width: int,
        rank: int,
        full_count: int,
    ) -> None:
        super().__init__()
        vocabulary_size = int(vocabulary_size)
        output_width = int(output_width)
        rank = int(rank)
        full_count = int(full_count)
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if not 1 <= full_count <= vocabulary_size:
            raise ValueError("full_count must be in [1, vocabulary_size]")
        if rank < 1 or rank > output_width:
            raise ValueError("embedding rank must be in [1, output_width]")
        self.vocabulary_size = vocabulary_size
        self.output_width = output_width
        self.rank = rank
        self.full_count = full_count
        self.full = nn.Embedding(full_count, output_width)
        self.rare = (
            None
            if full_count == vocabulary_size
            else _FactorizedEmbedding(vocabulary_size - full_count, output_width, rank)
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        token = token.long()
        full_mask = token < int(self.full_count)
        full_index = token.clamp(min=0, max=int(self.full_count) - 1)
        full_value = self.full(full_index)
        if self.rare is None:
            # This branch is used for the small parent vocabulary, where all
            # typed radius-1 tokens fit in the full table.
            return full_value
        rare_index = (token - int(self.full_count)).clamp(
            min=0, max=int(self.vocabulary_size - self.full_count) - 1
        )
        rare_value = self.rare(rare_index)
        return torch.where(full_mask.unsqueeze(-1), full_value, rare_value)


PARAMETER_AUDIT_BLOCKS = (
    ("typed_token_embedding", "typed_embedding"),
    ("parent_token_embedding", "parent_embedding"),
    ("patch_encoder", "patch_encoder"),
    ("pair_projection", "pair_projection"),
    ("relation_encoder", "relation_encoder"),
    ("distance_gate", "distance_gate"),
    ("pair_encoder", "pair_encoder"),
    ("center_context", "center_update"),
    ("global_encoder", "global_encoder"),
    ("graph_head", "head"),
    ("structural_context_embedding", "structural_context_embedding"),
    ("context_patch_projection", "context_patch_projection"),
    ("context_condition_projection", "context_condition_projection"),
    ("context_delta_output", "context_delta_output"),
    ("topology_encoder", "topology_encoder"),
)


def _block_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def _embedding_spec(module: nn.Module) -> dict[str, Any]:
    """Describe an embedding module regardless of mode (full/factorized/hybrid)."""
    vocabulary_size = int(
        getattr(module, "vocabulary_size", None)
        or getattr(module, "num_embeddings", 0)
    )
    output_width = int(
        getattr(module, "output_width", None)
        or getattr(module, "embedding_dim", 0)
    )
    return {
        "vocabulary_size": vocabulary_size,
        "output_width": output_width,
        "rank": int(getattr(module, "rank", 0)),
        "full_count": int(getattr(module, "full_count", 0)),
    }


def audit_parameters(model: PatchPathModel) -> dict[str, Any]:
    """Count trainable (``requires_grad=True``) parameters per major module.

    The per-block total is cross-checked against the global parameter
    iterator, so an audit result can never silently disagree with
    ``sum(p.numel() for p in model.parameters() if p.requires_grad)``.
    """
    blocks: dict[str, int] = {}
    for label, attribute in PARAMETER_AUDIT_BLOCKS:
        blocks[label] = _block_parameters(getattr(model, attribute))
    accounted = sum(blocks.values())
    total_global = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    if accounted != total_global:
        raise RuntimeError(
            f"parameter audit inconsistency: blocks={accounted} != global={total_global}"
        )
    dimensions = {
        "patch_descriptor_dim": int(model.shell_width),
        "context_descriptor_dim": int(model.context_width),
        "typed_token_width": int(model.token_width),
        "parent_token_width": int(model.parent_width),
        "patch_hidden_dim": int(model.patch_hidden),
        "relation_descriptor_dim": int(RELATION_WIDTH),
        "pair_hidden_dim": int(model.pair_hidden),
        "distance_buckets": int(DISTANCE_BUCKETS),
        "unary_pooled_dim": int(model.pooled_unary_width),
        "pair_pooled_dim": int(model.pooled_pair_width),
        "graph_readout_dim": int(model.readout_width),
        "graph_head_hidden_dims": [
            int(model.head_hidden_0),
            int(model.head_hidden_1),
        ],
        "center_context_hidden_dim": (
            None if model.center_update is None else int(model.center_context_hidden)
        ),
        "global_encoder_width": int(model.global_output_width),
        "structural_context_mode": str(model.structural_context_mode),
        "structural_context_fusion": str(model.structural_context_fusion),
        "structural_context_dim": int(model.structural_context_dim),
        "structural_context_condition_rank": int(model.structural_context_condition_rank),
        "structural_context_vocabulary_size": int(model.structural_context_vocabulary_size),
        "topology_mode": str(model.topology_mode),
        "topology_input_width": int(model.topology_input_width),
        "topology_hidden_dim": int(model.topology_hidden_dim),
        "topology_out_dim": int(model.topology_out_dim),
        "unified_graph_width": int(model.unified_graph_width),
    }
    return {
        "blocks": blocks,
        "total_trainable": total_global,
        "fraction": {
            label: (float(count / total_global) if total_global else 0.0)
            for label, count in blocks.items()
        },
        "dimensions": dimensions,
        "embedding": {
            "mode": str(model.embedding_mode),
            "rank": int(model.embedding_rank),
            "typed": _embedding_spec(model.typed_embedding),
            "parent": _embedding_spec(model.parent_embedding),
        },
    }


def _print_parameter_audit(model: PatchPathModel, phase_label: str) -> dict[str, Any]:
    audit = audit_parameters(model)
    blocks = audit["blocks"]
    fractions = audit["fraction"]
    print(f"Model parameter audit (phase={phase_label})", flush=True)
    for label, _attribute in PARAMETER_AUDIT_BLOCKS:
        if blocks[label] == 0 and (
            label == "center_context"
            or label == "structural_context_embedding"
            or label.startswith("context_")
        ):
            continue
        print(
            f"{label}: {blocks[label]} ({fractions[label] * 100.0:.2f}%)",
            flush=True,
        )
    print("total_trainable_params: " + str(audit["total_trainable"]), flush=True)
    dims = audit["dimensions"]
    print("Representation dimensions", flush=True)
    for key in (
        "patch_descriptor_dim",
        "typed_token_width",
        "parent_token_width",
        "patch_hidden_dim",
        "relation_descriptor_dim",
        "pair_hidden_dim",
        "distance_buckets",
        "unary_pooled_dim",
        "pair_pooled_dim",
        "graph_readout_dim",
        "center_context_hidden_dim",
        "structural_context_dim",
        "structural_context_condition_rank",
    ):
        print(f"{key}: {dims[key]}", flush=True)
    print(f"structural_context: mode={dims['structural_context_mode']} "
          f"fusion={dims['structural_context_fusion']} "
          f"vocab(incl NO_RING/UNK)={dims['structural_context_vocabulary_size']}", flush=True)
    print(f"embedding: {audit['embedding']}", flush=True)
    return audit


def _make_embedding(
    vocabulary_size: int,
    output_width: int,
    *,
    mode: str,
    rank: int,
    full_count: int | None = None,
) -> nn.Module:
    if mode == "full":
        return nn.Embedding(int(vocabulary_size), int(output_width))
    if mode == "factorized":
        return _FactorizedEmbedding(int(vocabulary_size), int(output_width), int(rank))
    if mode == "hybrid":
        if full_count is None:
            raise ValueError("hybrid embedding requires full_count")
        return _HybridEmbedding(
            int(vocabulary_size), int(output_width), int(rank), int(full_count)
        )
    raise ValueError(
        f"unknown embedding_mode={mode!r}; expected full, factorized, or hybrid"
    )


class PatchPathModel(nn.Module):
    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        patch_hidden: int,
        pair_hidden: int,
        token_width: int,
        dropout: float,
        embedding_mode: str = "full",
        embedding_rank: int = 16,
        parent_embedding_rank: int | None = None,
        hybrid_full_typed_tokens: int | None = None,
        hybrid_full_parent_tokens: int | None = None,
        readout: str = "moments",
        node_readout: str | None = None,
        pair_readout: str | None = None,
        shell_width: int = SHELL_WIDTH,
        context_width: int = 0,
        direct_token_readout: bool = False,
        center_context: bool = False,
        center_context_hidden: int | None = None,
        graph_head_hidden_0: int | None = None,
        graph_head_hidden_1: int | None = None,
        structural_context_mode: str = "none",
        structural_context_fusion: str = "condition",
        structural_context_dim: int = 8,
        structural_context_embedding_rank: int = 1,
        structural_context_condition_rank: int = 8,
        structural_context_vocabulary_size: int = 2,
        topology_mode: str = "none",
        topology_input_width: int = 0,
        topology_hidden_dim: int = 16,
        topology_out_dim: int = 8,
        quantile_mode: str = "none",
    ) -> None:
        super().__init__()
        if quantile_mode not in QUANTILE_MODES:
            raise ValueError(
                f"unknown quantile_mode={quantile_mode!r}; expected one of "
                f"{sorted(QUANTILE_MODES)}"
            )
        self.quantile_mode = str(quantile_mode)
        self.patch_hidden = int(patch_hidden)
        self.pair_hidden = int(pair_hidden)
        self.embedding_mode = str(embedding_mode)
        self.embedding_rank = int(embedding_rank)
        self.parent_embedding_rank = int(
            embedding_rank if parent_embedding_rank is None else parent_embedding_rank
        )
        self.readout = str(readout)
        self.node_readout = str(node_readout or self.readout)
        self.pair_readout = str(pair_readout or self.readout)
        self.shell_width = int(shell_width)
        self.context_width = int(context_width)
        self.direct_token_readout = bool(direct_token_readout)
        self.center_context = bool(center_context)
        self.token_width = int(token_width)
        self.global_output_width = 32
        self.structural_context_mode = str(structural_context_mode)
        self.structural_context_fusion = str(structural_context_fusion)
        self.structural_context_dim = int(structural_context_dim)
        self.structural_context_embedding_rank = int(structural_context_embedding_rank)
        self.structural_context_condition_rank = int(structural_context_condition_rank)
        self.structural_context_vocabulary_size = int(structural_context_vocabulary_size)
        # Compact-v4: global topology channel.  This is NOT a ring network and
        # NOT a second predictor: it is a tiny MLP that maps the graph-level
        # invariant topology vector to ``z_topology``, which is concatenated
        # into the single unified graph representation consumed by the sole
        # regression head (see notes/compact_v4_global_topology_channel.md).
        self.topology_mode = str(topology_mode)
        self.topology_input_width = int(topology_input_width)
        self.topology_hidden_dim = int(topology_hidden_dim)
        self.topology_out_dim = int(topology_out_dim)
        if self.topology_mode != "none":
            if self.topology_input_width < 1 or self.topology_hidden_dim < 1 or self.topology_out_dim < 1:
                raise ValueError(
                    "topology_mode != none requires positive topology input/hidden/out dims"
                )
            self.topology_encoder = nn.Sequential(
                nn.Linear(self.topology_input_width, self.topology_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.topology_hidden_dim, self.topology_out_dim),
            )
        else:
            self.topology_encoder = None
        valid_context_modes = {"none", "coarse_ring", "typed_ring"}
        if self.structural_context_mode not in valid_context_modes:
            raise ValueError(
                f"unknown structural_context_mode={self.structural_context_mode!r}; "
                f"expected one of {sorted(valid_context_modes)}"
            )
        if self.structural_context_fusion not in {"condition", "concat"}:
            raise ValueError(
                f"unknown structural_context_fusion={self.structural_context_fusion!r}; "
                f"expected 'condition' or 'concat'"
            )
        if self.structural_context_dim < 1 or self.structural_context_condition_rank < 1:
            raise ValueError("structural context dims must be positive")
        valid_readouts = {"moments", "mean_std", "sum_mean_std", "distribution"}
        if self.node_readout not in valid_readouts or self.pair_readout not in valid_readouts:
            raise ValueError(
                "unknown readout; expected moments, mean_std, sum_mean_std, or distribution"
            )
        # Compact-v3 structural-context conditioning signal.  This is not a
        # ring network: the module only builds ``e_ctx`` (an embedding of the
        # ring context) and optionally a low-rank multiplicative gate on the
        # exact patch embedding.  Nothing here creates ring nodes or passes
        # messages on rings.
        self.context_patch_projection: nn.Module | None = None
        self.context_condition_projection: nn.Module | None = None
        self.context_delta_output: nn.Module | None = None
        self.structural_context_embedding: nn.Module | None = None
        structural_input_width = 0
        if self.structural_context_mode != "none":
            if self.structural_context_mode == "typed_ring":
                if self.structural_context_vocabulary_size < 2:
                    raise ValueError(
                        "typed_ring requires structural_context_vocabulary_size >= 2 "
                        "(NO_RING + UNK_CONTEXT at least)"
                    )
                # Rows: [UNK_CONTEXT, known_0, known_1, ...]; NO_RING is
                # masked to zeros in the forward pass.  One row per known
                # typed context (no merging/hashing), low-rank factorized.
                self.structural_context_embedding = _FactorizedEmbedding(
                    self.structural_context_vocabulary_size - 1,
                    self.structural_context_dim,
                    min(self.structural_context_embedding_rank, self.structural_context_dim),
                )
            else:  # coarse_ring: continuous 4D coarse vector
                self.structural_context_embedding = nn.Linear(
                    COARSE_WIDTH, self.structural_context_dim, bias=False
                )
            if self.structural_context_fusion == "condition":
                # ``e_patch + W_out(tanh(W_p e_patch) * tanh(W_c e_ctx))``
                # NO_RING is masked so that the delta is exactly zero there:
                # the model is bit-identical to compact-v2 on ringless
                # patches (and exactly compact-v2 at initialisation, since
                # W_out is zero-initialised).
                self.context_patch_projection = nn.Linear(
                    int(token_width), self.structural_context_condition_rank, bias=False
                )
                self.context_condition_projection = nn.Linear(
                    self.structural_context_dim, self.structural_context_condition_rank, bias=False
                )
                self.context_delta_output = nn.Linear(
                    self.structural_context_condition_rank, int(token_width)
                )
                nn.init.zeros_(self.context_delta_output.weight)
                nn.init.zeros_(self.context_delta_output.bias)
            else:  # concat ablation: context goes into the patch encoder input
                structural_input_width = self.structural_context_dim
        typed_full_count = (
            None
            if hybrid_full_typed_tokens is None
            else int(hybrid_full_typed_tokens)
        )
        self.typed_embedding = _make_embedding(
            int(typed_vocabulary_size),
            int(token_width),
            mode=self.embedding_mode,
            rank=self.embedding_rank,
            full_count=typed_full_count,
        )
        parent_width = max(int(token_width // 2), 1)
        self.parent_width = int(parent_width)
        parent_rank = min(self.parent_embedding_rank, parent_width)
        parent_full_count = (
            None
            if hybrid_full_parent_tokens is None
            else int(hybrid_full_parent_tokens)
        )
        self.parent_embedding = _make_embedding(
            int(parent_vocabulary_size),
            parent_width,
            mode=self.embedding_mode,
            rank=parent_rank,
            full_count=parent_full_count,
        )
        if self.direct_token_readout and self.embedding_mode != "factorized":
            raise ValueError(
                "direct_token_readout currently requires embedding_mode=factorized "
                "so the raw low-rank code remains a compact identity channel"
            )
        self.direct_token_code_width = (
            int(self.embedding_rank) if self.direct_token_readout else 0
        )
        self.patch_encoder = _MLPBlock(
            self.shell_width
            + self.context_width
            + int(token_width)
            + parent_width
            + int(structural_input_width),
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = _MLPBlock(GLOBAL_WIDTH, max(int(patch_hidden // 2), 32), 32, float(dropout))
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = _MLPBlock(
            RELATION_WIDTH, max(int(pair_hidden), 32), int(pair_hidden), float(dropout)
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = _MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        if self.center_context:
            # Keep the original global pair readout, but additionally retain
            # which pair relations share a centre.  For every centre and
            # distance bucket we aggregate the mean, standard deviation and
            # mass of incident pair embeddings.  The final projection is
            # zero-initialised, so the complete model is exactly the original
            # patch-path model at initialisation.
            self.center_context_width = DISTANCE_BUCKETS * (
                2 * int(pair_hidden) + 1
            )
            update_hidden = int(
                center_context_hidden
                if center_context_hidden is not None
                else max(2 * int(patch_hidden), 96)
            )
            self.center_context_hidden = int(update_hidden)
            self.center_update = nn.Sequential(
                nn.Linear(
                    int(patch_hidden) + self.center_context_width,
                    update_hidden,
                ),
                nn.LayerNorm(update_hidden),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(update_hidden, int(patch_hidden)),
            )
            nn.init.zeros_(self.center_update[-1].weight)
            nn.init.zeros_(self.center_update[-1].bias)
        else:
            self.center_context_width = 0
            self.center_update = None
            self.center_context_hidden = None
        # Unary and pair readouts are invariant moment summaries.  The final
        # module is the sole graph-level prediction head.
        pooled_unary_width = self._pooled_width(int(patch_hidden), self.node_readout)
        pooled_pair_width = self._pooled_width(int(pair_hidden), self.pair_readout)
        direct_width = (
            2 * int(self.direct_token_code_width) + 1
            if self.direct_token_readout
            else 0
        )
        readout_width = (
            pooled_unary_width
            + DISTANCE_BUCKETS * pooled_pair_width
            + direct_width
        )
        self.pooled_unary_width = int(pooled_unary_width)
        self.pooled_pair_width = int(pooled_pair_width)
        self.readout_width = int(readout_width)
        # Graph head hidden widths are configurable; when the candidate does
        # not set them, the historical default (``max(2*patch_hidden, 96)``
        # and ``patch_hidden``) is preserved exactly so old configs keep the
        # same architecture and parameter counts.
        self.head_hidden_0 = (
            int(graph_head_hidden_0)
            if graph_head_hidden_0 is not None
            else max(int(patch_hidden) * 2, 96)
        )
        self.head_hidden_1 = (
            int(graph_head_hidden_1)
            if graph_head_hidden_1 is not None
            else int(patch_hidden)
        )
        if self.head_hidden_0 < 1 or self.head_hidden_1 < 1:
            raise ValueError("graph head hidden dimensions must be positive")
        topology_width = self.topology_out_dim if self.topology_encoder is not None else 0
        self.unified_graph_width = int(readout_width + 32 + topology_width)
        # v5: only the final output width may change.  ``none``/``median_only``
        # keep the exact compact-v4 scalar head (bit-identical construction);
        # ``q10_q50_q90`` widens only the last Linear to 3 raw outputs
        # [m, d_low_raw, d_high_raw] (parameter delta = +66 for the frozen
        # 32-wide hidden_1).  The decoding into non-crossing quantiles happens
        # in forward(); no extra MLP is added.
        head_output_dim = 3 if self.quantile_mode == "q10_q50_q90" else 1
        self.head = nn.Sequential(
            nn.Linear(self.unified_graph_width, self.head_hidden_0),
            nn.LayerNorm(self.head_hidden_0),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.head_hidden_0, self.head_hidden_1),
            nn.ReLU(),
            nn.Linear(self.head_hidden_1, head_output_dim),
        )

    @staticmethod
    def _pooled_width(width: int, mode: str) -> int:
        if mode in {"moments", "mean_std"}:
            return 2 * int(width) + 1
        if mode == "sum_mean_std":
            return 3 * int(width) + 1
        if mode == "distribution":
            return 4 * int(width) + 1
        raise ValueError(f"unknown pooling mode {mode!r}")

    @staticmethod
    def _grouped_max(
        value: torch.Tensor,
        group: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        maximum = torch.full(
            (int(n_groups), value.shape[1]),
            -float("inf"),
            device=value.device,
            dtype=value.dtype,
        )
        if value.numel():
            maximum.scatter_reduce_(
                0,
                group.view(-1, 1).expand(-1, value.shape[1]),
                value,
                reduce="amax",
                include_self=True,
            )
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    def _pool_values(
        self,
        value: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
        mode: str,
    ) -> torch.Tensor:
        total = torch.zeros((n_graphs, value.shape[1]), device=value.device, dtype=value.dtype)
        total.index_add_(0, batch, value)
        counts = torch.bincount(batch, minlength=n_graphs).to(value.dtype).unsqueeze(1)
        if mode == "moments":
            squared = torch.zeros_like(total)
            squared.index_add_(0, batch, value * value)
            output = torch.cat([total, squared, torch.log1p(counts)], dim=1)
        elif mode == "mean_std":
            mean = total / counts.clamp_min(1.0)
            squared = torch.zeros_like(total)
            squared.index_add_(0, batch, value * value)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            output = torch.cat([mean, std, torch.log1p(counts)], dim=1)
        elif mode == "sum_mean_std":
            mean = total / counts.clamp_min(1.0)
            squared = torch.zeros_like(total)
            squared.index_add_(0, batch, value * value)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            output = torch.cat([total, mean, std, torch.log1p(counts)], dim=1)
        elif mode == "distribution":
            mean = total / counts.clamp_min(1.0)
            squared = torch.zeros_like(total)
            squared.index_add_(0, batch, value * value)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            maximum = self._grouped_max(value, batch, n_graphs)
            output = torch.cat([total, mean, std, maximum, torch.log1p(counts)], dim=1)
        else:
            raise ValueError(f"unknown pooling mode {mode!r}")
        expected = self._pooled_width(value.shape[1], mode)
        if output.shape[1] != expected:
            raise RuntimeError(f"pooled width changed for mode={mode}: {output.shape[1]} != {expected}")
        return output

    def _pool_nodes(self, value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        return self._pool_values(value, batch, n_graphs, self.node_readout)

    def _pool_pairs(
        self,
        value: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for bucket in range(DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_batch = pair_batch[mask]
            total = torch.zeros((n_graphs, value.shape[1]), device=value.device, dtype=value.dtype)
            counts = torch.zeros((n_graphs, 1), device=value.device, dtype=value.dtype)
            if current.numel():
                total.index_add_(0, current_batch, current)
                counts.index_add_(
                    0,
                    current_batch,
                    torch.ones((current_batch.shape[0], 1), device=value.device, dtype=value.dtype),
                )
            if self.pair_readout == "moments":
                squared = torch.zeros_like(total)
                if current.numel():
                    squared.index_add_(0, current_batch, current * current)
                pooled = torch.cat([total, squared, torch.log1p(counts)], dim=1)
            elif self.pair_readout == "mean_std":
                mean = total / counts.clamp_min(1.0)
                squared = torch.zeros_like(total)
                if current.numel():
                    squared.index_add_(0, current_batch, current * current)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                std = torch.sqrt(variance + 1.0e-8)
                pooled = torch.cat([mean, std, torch.log1p(counts)], dim=1)
            elif self.pair_readout == "sum_mean_std":
                mean = total / counts.clamp_min(1.0)
                squared = torch.zeros_like(total)
                if current.numel():
                    squared.index_add_(0, current_batch, current * current)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                std = torch.sqrt(variance + 1.0e-8)
                pooled = torch.cat([total, mean, std, torch.log1p(counts)], dim=1)
            elif self.pair_readout == "distribution":
                mean = total / counts.clamp_min(1.0)
                squared = torch.zeros_like(total)
                if current.numel():
                    squared.index_add_(0, current_batch, current * current)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                std = torch.sqrt(variance + 1.0e-8)
                maximum = self._grouped_max(current, current_batch, n_graphs)
                pooled = torch.cat([total, mean, std, maximum, torch.log1p(counts)], dim=1)
            else:
                raise ValueError(f"unknown pooling mode {self.pair_readout!r}")
            expected = self._pooled_width(value.shape[1], self.pair_readout)
            if pooled.shape[1] != expected:
                raise RuntimeError(
                    f"pair pooled width changed for mode={self.pair_readout}: "
                    f"{pooled.shape[1]} != {expected}"
                )
            blocks.append(pooled)
        return torch.cat(blocks, dim=1)

    def _pool_pairs_to_centres(
        self,
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        """Pool incident pair states per centre and distance bucket.

        Pair rows are unordered, so every row contributes identically to both
        endpoints.  Mean/std rather than raw sums keep the centre update
        numerically stable across molecule sizes; log-count retains relation
        mass explicitly.
        """
        blocks: list[torch.Tensor] = []
        width = int(value.shape[1])
        for bucket in range(DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_source = source[mask]
            current_target = target[mask]
            total = torch.zeros(
                (int(n_centres), width), device=value.device, dtype=value.dtype
            )
            squared = torch.zeros_like(total)
            counts = torch.zeros(
                (int(n_centres), 1), device=value.device, dtype=value.dtype
            )
            if current.numel():
                endpoints = torch.cat([current_source, current_target], dim=0)
                duplicated = torch.cat([current, current], dim=0)
                total.index_add_(0, endpoints, duplicated)
                squared.index_add_(0, endpoints, duplicated * duplicated)
                counts.index_add_(
                    0,
                    endpoints,
                    torch.ones(
                        (endpoints.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                )
            denominator = counts.clamp_min(1.0)
            mean = total / denominator
            variance = (squared / denominator - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            occupied = (counts > 0).to(value.dtype)
            blocks.append(
                torch.cat(
                    [mean, std * occupied, torch.log1p(counts)], dim=1
                )
            )
        return torch.cat(blocks, dim=1)

    def _structural_context_embedding_value(self, data: Data) -> torch.Tensor:
        """Build ``e_ctx`` from the node's structural context.

        * ``typed_ring``: row lookup of the canonical typed cycle signature
          (token 0 = NO_RING is masked to zeros; token 1 = UNK_CONTEXT keeps
          its own learned row).
        * ``coarse_ring``: linear map of the 4D coarse vector.
        """
        if self.structural_context_mode == "typed_ring":
            token = data.structural_token
            if token.numel() == 0:
                return torch.zeros(
                    (0, self.structural_context_dim), device=token.device, dtype=token.dtype
                )
            # Rows: [UNK_CONTEXT, known_0, known_1, ...]; NO_RING (token 0)
            # lands on the UNK row but is masked to zeros below.
            index = (token - UNK_CONTEXT_TOKEN).clamp(min=0)
            value = self.structural_context_embedding(index)
            return value * (token > NO_RING_TOKEN).float().unsqueeze(-1)
        return self.structural_context_embedding(data.structural_coarse)

    def _structural_no_ring_mask(self, data: Data) -> torch.Tensor:
        """1.0 for patches that carry a ring context, 0.0 for NO_RING."""
        if self.structural_context_mode == "typed_ring":
            return (data.structural_token > NO_RING_TOKEN).float().unsqueeze(-1)
        return (data.structural_coarse[:, 0] > 0.5).float().unsqueeze(-1)

    def _condition_patch(
        self, e_patch: torch.Tensor, e_ctx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Low-rank multiplicative conditioning.

        ``e_patch_conditioned = e_patch + delta`` with
        ``delta = W_out(tanh(W_p e_patch) * tanh(W_c e_ctx))``; when
        ``mask`` is zero (NO_RING) the delta is exactly zero and the output
        is bit-identical to the input.
        """
        p = self.context_patch_projection(e_patch)
        c = self.context_condition_projection(e_ctx)
        delta = self.context_delta_output(torch.tanh(p) * torch.tanh(c))
        return e_patch + delta * mask

    def forward(self, data: Data) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        e_patch = self.typed_embedding(data.typed_token)
        structural_blocks: list[torch.Tensor] = []
        if self.structural_context_mode != "none":
            e_ctx = self._structural_context_embedding_value(data)
            if self.structural_context_fusion == "condition":
                e_patch = self._condition_patch(
                    e_patch, e_ctx, self._structural_no_ring_mask(data)
                )
            else:
                structural_blocks.append(e_ctx)
        patch = self.patch_encoder(
            torch.cat(
                [
                    data.patch_cont,
                    data.patch_context,
                    e_patch,
                    self.parent_embedding(data.parent_token),
                    *structural_blocks,
                ],
                dim=1,
            )
        )
        unary = self._pool_nodes(patch, data.batch, n_graphs)
        direct_blocks: list[torch.Tensor] = []
        if self.direct_token_readout:
            # The factorized table has one independent rank-dimensional code
            # per exact token.  Pooling this code directly preserves a
            # compact identity/count channel in addition to the nonlinear
            # patch-state moments below.
            token_code = self.typed_embedding.embedding(data.typed_token)
            direct_blocks.append(self._pool_values(
                token_code, data.batch, n_graphs, "moments"
            ))

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected_left = self.pair_projection(patch[source])
        projected_right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        product = projected_left * projected_right
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        pair_value = self.pair_encoder(pair_input)
        pair_batch = data.batch[source]
        if self.center_update is not None:
            center_context = self._pool_pairs_to_centres(
                pair_value,
                source,
                target,
                data.pair_bucket,
                int(patch.shape[0]),
            )
            patch = patch + self.center_update(
                torch.cat([patch, center_context], dim=1)
            )
            # The unary readout is deliberately recomputed after the update.
            # The direct pair readout below remains the original scheme-seven
            # path, while unary moments now retain pair-incidence structure.
            unary = self._pool_nodes(patch, data.batch, n_graphs)
        relation_readout = self._pool_pairs(
            pair_value, pair_batch, data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        readout_blocks = [unary, relation_readout]
        readout_blocks.extend(direct_blocks)
        readout_blocks.append(graph_hidden)
        if self.topology_encoder is not None:
            topology = data.topology_features
            if topology.ndim == 1:
                topology = topology.unsqueeze(0)
            if int(topology.shape[1]) != self.topology_input_width:
                raise RuntimeError(
                    f"topology width mismatch: data={int(topology.shape[1])} "
                    f"model={self.topology_input_width}"
                )
            readout_blocks.append(self.topology_encoder(topology))
        unified = torch.cat(readout_blocks, dim=1)
        if self.quantile_mode == "q10_q50_q90":
            # Non-crossing parameterization: raw head outputs are
            # [m, d_low_raw, d_high_raw]; d_low/d_high are softplus-positive
            # widths, so q10 = m - d_low <= m = q50 <= q90 = m + d_high by
            # construction (no explicit crossing penalty needed).
            raw = self.head(unified)
            m = raw[:, 0]
            d_low = F.softplus(raw[:, 1])
            d_high = F.softplus(raw[:, 2])
            q10 = m - d_low
            q90 = m + d_high
            return torch.stack([q10, m, q90], dim=1)
        return self.head(unified).view(-1)


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _predict_values(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets, predictions) per molecule, in loader order."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(predictions).astype(np.float64),
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Validation MAE of the point prediction (q50 for multi-quantile)."""
    targets, predictions = _predict_values(model, loader, device)
    return float(mean_absolute_error(targets, _quantile_point_prediction_numpy(predictions)))


def _shuffled_topology_mae(
    model: nn.Module,
    graphs: Sequence[Data],
    device: torch.device,
    seed: int,
    batch_size: int = 128,
) -> float:
    """Validation-only diagnostic: MAE with molecule<->topology pairing broken.

    The molecule order is fixed; each molecule receives the topology feature
    vector of a *different* molecule (a fixed random derangement).  Targets
    stay attached to the molecules, so a drop of the gain here means the
    model was genuinely using the topology channel.
    """
    rng = np.random.default_rng(int(seed))
    n = len(graphs)
    permutation = rng.permutation(n)
    # force a derangement (no molecule keeps its own topology vector)
    while np.any(permutation == np.arange(n)):
        permutation = rng.permutation(n)
    shuffled: list[Data] = []
    for index, graph in enumerate(graphs):
        copy = graph.clone()
        copy.topology_features = graphs[int(permutation[index])].topology_features.clone()
        shuffled.append(copy)
    loader = DataLoader(shuffled, batch_size=int(batch_size), shuffle=False)
    return _evaluate(model, loader, device)


def _train_phase(
    train_graphs: Sequence[Data],
    eval_graphs: Sequence[Data],
    *,
    typed_vocabulary_size: int,
    parent_vocabulary_size: int,
    config: Mapping[str, Any],
    seed: int,
    select_best: bool,
    epochs: int,
        shell_width: int = SHELL_WIDTH,
        context_width: int = 0,
        direct_token_readout: bool = False,
        phase_label: str | None = None,
        structural_context_vocabulary_size: int = 0,
        topology_input_width: int = 0,
        topology_hidden_dim: int = 16,
        topology_out_dim: int = 8,
        topology_shuffle_test: bool = True,
        quantile_mode: str | None = None,
        quantile_lambda: float | None = None,
) -> dict[str, Any]:
    model_config = config["model"]
    quantile_mode = str(
        quantile_mode
        if quantile_mode is not None
        else model_config.get("quantile_mode", "none")
    )
    quantile_lambda = (
        model_config.get("quantile_lambda")
        if quantile_lambda is None
        else quantile_lambda
    )
    effective_lambda = _validate_quantile_config(quantile_mode, quantile_lambda)
    structural_context_mode = str(model_config.get("structural_context_mode", "none"))
    topology_mode = str(model_config.get("topology_mode", "none"))
    if topology_mode != "none" and topology_input_width < 1:
        raise RuntimeError(
            "topology_mode != none requires a fitted topology standardizer "
            "(topology_input_width >= 1)"
        )
    if structural_context_mode == "typed_ring":
        model_config = dict(model_config)
        if int(structural_context_vocabulary_size) < 2:
            raise RuntimeError(
                "typed_ring requires a fitted structural context vocabulary "
                "(structural_context_vocabulary_size >= 2)"
            )
        model_config["structural_context_vocabulary_size"] = int(
            structural_context_vocabulary_size
        )
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")
    _seed_everything(seed)
    model = PatchPathModel(
        typed_vocabulary_size,
        parent_vocabulary_size,
        patch_hidden=int(model_config.get("patch_hidden", PATCH_HIDDEN)),
        pair_hidden=int(model_config.get("pair_hidden", PAIR_HIDDEN)),
        token_width=int(model_config.get("token_width", 32)),
        dropout=float(model_config.get("dropout", 0.05)),
        embedding_mode=str(model_config.get("embedding_mode", "full")),
        embedding_rank=int(model_config.get("embedding_rank", 16)),
        parent_embedding_rank=(
            None
            if model_config.get("parent_embedding_rank") is None
            else int(model_config["parent_embedding_rank"])
        ),
        hybrid_full_typed_tokens=(
            None
            if model_config.get("hybrid_full_typed_tokens") is None
            else int(model_config["hybrid_full_typed_tokens"])
        ),
        hybrid_full_parent_tokens=(
            None
            if model_config.get("hybrid_full_parent_tokens") is None
            else int(model_config["hybrid_full_parent_tokens"])
        ),
        readout=str(model_config.get("readout", "moments")),
        node_readout=model_config.get("node_readout"),
        pair_readout=model_config.get("pair_readout"),
        shell_width=int(shell_width),
        context_width=int(context_width),
        direct_token_readout=bool(direct_token_readout),
        center_context=bool(model_config.get("center_context", False)),
        center_context_hidden=(
            None
            if model_config.get("center_context_hidden") is None
            else int(model_config["center_context_hidden"])
        ),
        graph_head_hidden_0=(
            None
            if model_config.get("graph_head_hidden_0") is None
            else int(model_config["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model_config.get("graph_head_hidden_1") is None
            else int(model_config["graph_head_hidden_1"])
        ),
        structural_context_mode=str(model_config.get("structural_context_mode", "none")),
        structural_context_fusion=str(model_config.get("structural_context_fusion", "condition")),
        structural_context_dim=int(model_config.get("structural_context_dim", 8)),
        structural_context_embedding_rank=int(
            model_config.get("structural_context_embedding_rank", 1)
        ),
        structural_context_condition_rank=int(
            model_config.get("structural_context_condition_rank", 8)
        ),
        structural_context_vocabulary_size=int(
            model_config.get("structural_context_vocabulary_size", 2)
        ),
        topology_mode=topology_mode,
        topology_input_width=int(topology_input_width),
        topology_hidden_dim=int(
            model_config.get("topology_hidden_dim", 16)
        ),
        topology_out_dim=int(
            model_config.get("topology_out_dim", 8)
        ),
        quantile_mode=quantile_mode,
    ).to(device)
    parameter_audit: dict[str, Any] | None = None
    if bool(model_config.get("parameter_audit", False)):
        phase_label = phase_label or ("validation-selection" if select_best else "train-valid-refit")
        parameter_audit = _print_parameter_audit(model, phase_label)
        budget = model_config.get("expected_max_trainable_params")
        if budget is not None:
            total = int(parameter_audit["total_trainable"])
            budget = int(budget)
            print(f"Trainable params: {total}", flush=True)
            print(f"Parameter budget: {budget}", flush=True)
            if total <= budget:
                print("PASS", flush=True)
            else:
                print("FAIL", flush=True)
                raise RuntimeError(
                    f"parameter budget exceeded: {total} > {budget} "
                    f"(phase={phase_label}); model is fatter than the compact "
                    f"budget, refuse to train silently"
                )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    loader = _make_loader(
        train_graphs,
        int(model_config.get("batch_size", 128)),
        True,
        seed + 91011,
    )
    eval_loader = _make_loader(
        eval_graphs,
        int(model_config.get("batch_size", 128)),
        False,
        seed + 91012,
    )
    patience = int(model_config.get("patience", 10))
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, float | int]] = []
    stale = 0
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            target = batch.y.view(-1)
            loss = quantile_regression_loss(
                prediction,
                target,
                quantile_mode=quantile_mode,
                quantile_lambda=effective_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(float(epoch_loss))
        current_mae = None
        if select_best:
            current_mae = _evaluate(model, eval_loader, device)
            trace.append({"epoch": int(epoch), "mae": float(current_mae)})
            if current_mae < best_mae:
                best_mae = float(current_mae)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            suffix = "" if current_mae is None else f" valid_mae={current_mae:.6f}"
            loss_name = "l1" if quantile_mode == "none" else "q_loss"
            print(
                f"patch-path phase={'select' if select_best else 'refit'} "
                f"epoch={epoch:03d}/{epochs} {loss_name}={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"patch-path early_stop epoch={epoch} best_epoch={best_epoch}", flush=True
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    eval_targets, eval_predictions = _predict_values(model, eval_loader, device)
    final_mae = float(
        mean_absolute_error(
            eval_targets, _quantile_point_prediction_numpy(eval_predictions)
        )
    )
    diagnostics: dict[str, Any] = {}
    if (
        select_best
        and topology_mode != "none"
        and bool(topology_shuffle_test)
        and best_state is not None
    ):
        diagnostics["valid_shuffled_topology_mae"] = _shuffled_topology_mae(
            model,
            eval_graphs,
            device,
            seed=int(seed),
            batch_size=int(model_config.get("batch_size", 128)),
        )
        diagnostics["shuffle_seed"] = int(seed)
    return {
        "model": model,
        "mae": float(final_mae),
        "best_mae": None if not select_best else float(best_mae),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "device": str(device),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_audit": parameter_audit,
        "diagnostics": diagnostics,
        "eval_targets": eval_targets if select_best else None,
        "eval_predictions": eval_predictions if select_best else None,
    }


def _phase_data(
    fit_records: Sequence[GraphRecord],
    other_records: Sequence[GraphRecord],
    *,
    config: Mapping[str, Any],
) -> tuple[list[Data], list[Data], dict[str, Any]]:
    representation = config["representation"]
    structural_mode = str(config.get("model", {}).get("structural_context_mode", "none"))
    topology_mode = str(config.get("model", {}).get("topology_mode", "none"))
    topology_input_width = ztopo.raw_width(
        topology_mode, input_width_hint=config.get("model", {}).get("topology_input_width")
    )
    patch_radius = int(representation.get("patch_radius", PATCH_RADIUS))
    typed_vocabulary = _fit_vocabulary(
        fit_records,
        "typed_certificate",
        int(representation.get("max_typed_tokens", 8192)),
        int(representation.get("minimum_typed_frequency", 1)),
    )
    parent_vocabulary = _fit_vocabulary(
        fit_records,
        "parent_certificate",
        int(representation.get("max_parent_tokens", 2048)),
        int(representation.get("minimum_parent_frequency", 1)),
    )
    patch_standardizer = Standardizer.fit(_patch_matrix(fit_records))
    context_standardizer = Standardizer.fit(_context_matrix(fit_records))
    has_context = any(
        patch.context_descriptor is not None
        for record in fit_records
        for patch in record.patches
    )
    context_patch_standardizer = (
        Standardizer.fit(_context_patch_matrix(fit_records)) if has_context else None
    )
    structural_vocabulary = None
    if str(structural_mode) == "typed_ring":
        structural_vocabulary = fit_structural_vocabulary(fit_records)
    topology_standardizer = None
    if topology_mode != "none":
        fit_topology = np.stack(
            [record.topology_features for record in fit_records], axis=0
        ).astype(np.float32, copy=False)
        if fit_topology.shape[1] != topology_input_width:
            raise RuntimeError(
                f"topology width mismatch: records={fit_topology.shape[1]} "
                f"config={topology_input_width} mode={topology_mode}"
            )
        topology_standardizer = Standardizer.fit(fit_topology)
    encoded_fit = _encode_records(
        fit_records,
        typed_vocabulary,
        parent_vocabulary,
        patch_standardizer,
        context_standardizer,
        context_patch_standardizer,
        structural_mode=structural_mode,
        structural_vocabulary=structural_vocabulary,
        topology_mode=topology_mode,
        topology_standardizer=topology_standardizer,
    )
    encoded_other = _encode_records(
        other_records,
        typed_vocabulary,
        parent_vocabulary,
        patch_standardizer,
        context_standardizer,
        context_patch_standardizer,
        structural_mode=structural_mode,
        structural_vocabulary=structural_vocabulary,
        topology_mode=topology_mode,
        topology_standardizer=topology_standardizer,
    )
    audit = {
        "typed_vocabulary": {
            "fit": _vocabulary_stats(fit_records, typed_vocabulary, "typed_certificate"),
            "other": _vocabulary_stats(other_records, typed_vocabulary, "typed_certificate"),
        },
        "parent_vocabulary": {
            "fit": _vocabulary_stats(fit_records, parent_vocabulary, "parent_certificate"),
            "other": _vocabulary_stats(other_records, parent_vocabulary, "parent_certificate"),
        },
        "typed_vocabulary_size_with_oov": int(len(typed_vocabulary) + 1),
        "parent_vocabulary_size_with_oov": int(len(parent_vocabulary) + 1),
        "patch_standardizer_fit_graphs": int(len(fit_records)),
        "context_standardizer_fit_graphs": int(len(fit_records)),
        "context_patch_width": int(CONTEXT_WIDTH if has_context else 0),
        "context_patch_standardizer_fit_graphs": int(len(fit_records)) if has_context else 0,
        "patch_radius": patch_radius,
        "shell_width": int(_shell_width_for_radius(patch_radius)),
        "direct_token_readout": bool(config.get("model", {}).get("direct_token_readout", False)),
        "structural_context": {
            "mode": structural_mode,
            "fit": context_statistics(fit_records, structural_vocabulary),
            "other": context_statistics(other_records, structural_vocabulary),
            "vocabulary_size_with_reserved": (
                int(len(structural_vocabulary)) + 2 if structural_vocabulary is not None else None
            ),
            "cycle_lengths_fit": cycle_length_distribution(fit_records),
            "cycle_lengths_other": cycle_length_distribution(other_records),
        },
        "structural_context_vocabulary_size": (
            int(len(structural_vocabulary)) + 2 if structural_vocabulary is not None else 0
        ),
        "topology": {
            "mode": topology_mode,
            "input_width": int(topology_input_width),
            "standardizer_fit_graphs": int(len(fit_records)),
            "mean": (
                topology_standardizer.mean.tolist()
                if topology_standardizer is not None
                else None
            ),
            "scale": (
                topology_standardizer.scale.tolist()
                if topology_standardizer is not None
                else None
            ),
        },
    }
    return encoded_fit, encoded_other, audit


def _format_mae(value: float | None) -> str:
    if value is None:
        return "blocked"
    return f"{value:.6f}"


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    center_context = bool(result["representation"].get("center_context", False))
    test_markdown = _format_mae(evaluation["test_after_train_valid_refit"]["mae"])
    lines = [
        f"# {result['protocol_id']}",
        "",
        (
            "Exact rooted typed patch tokens with shortest-path-conditioned pair pooling, "
            "one zero-initialised centre-context relation update, and one MLP."
            if center_context
            else "Exact rooted typed patch tokens with shortest-path-conditioned pair pooling; one MLP, no message passing."
        ),
        "",
        f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
        f"- patch shell descriptor: `{result['representation']['shell_width']}D`; relation descriptor: `{result['representation']['relation_width']}D`",
        f"- readout: `{result['representation']['readout']}`",
        f"- message passing: `{result['representation']['message_passing']}`; attention: `{result['representation']['attention']}`",
        (
            ""
            if result["representation"].get("structural_context", {}).get("mode") == "none"
            else f"- structural context: `{result['representation']['structural_context']['mode']}` "
            f"fusion `{result['representation']['structural_context']['fusion']}` "
            f"max_cycle_len `{result['representation']['structural_context']['max_cycle_len']}`; "
            "ring/cycle context only modifies the patch representation "
            "(no ring message passing / ring nodes)"
        ),
        "",
        "| head | valid MAE | test MAE after train+valid refit | selected epoch |",
        "|---|---:|---:|---:|",
        f"| `single_mlp` | {evaluation['valid']['mae']:.6f} | {test_markdown} | {evaluation['valid']['selected_epoch']} |",
        "",
        f"- train-only valid typed token coverage: `{result['vocabulary_audit']['valid']['typed_vocabulary']['other']['known_occurrence_fraction']:.4f}`",
        (
            "- train+valid test typed token coverage: `blocked`"
            if not result["vocabulary_audit"]["test_after_train_valid_refit"]
            else "- train+valid test typed token coverage: "
            f"`{result['vocabulary_audit']['test_after_train_valid_refit']['typed_vocabulary']['other']['known_occurrence_fraction']:.4f}`"
        ),
        f"- trainable parameters: `{evaluation['parameters']}`",
        f"- runtime: `{result['runtime']['seconds']:.1f}s`",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    result_json = _resolve(config["output"]["json"])
    result_markdown = _resolve(config["output"]["markdown"])
    test_policy = str(config.get("test_policy", "terminal"))
    if test_policy not in ("terminal", "no_test"):
        raise ValueError(
            f"unknown test_policy={test_policy!r}; expected 'terminal' or 'no_test'"
        )
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    thread_count = int(config.get("runtime", {}).get("torch_threads", 4))
    torch.set_num_threads(max(thread_count, 1))
    representation_config = config.get("representation", {})
    patch_radius = int(representation_config.get("patch_radius", PATCH_RADIUS))
    context_radius = int(representation_config.get("context_radius", 0))
    typed_tokenizer_version = resolve_typed_tokenizer_version(
        representation_config.get("typed_tokenizer_version")
    )
    structural_context_mode = str(
        config.get("model", {}).get("structural_context_mode", "none")
    )
    topology_mode = str(config.get("model", {}).get("topology_mode", "none"))
    topology_hidden_dim = int(config.get("model", {}).get("topology_hidden_dim", 16))
    topology_out_dim = int(config.get("model", {}).get("topology_out_dim", 8))
    topology_shuffle_test = bool(config.get("model", {}).get("topology_shuffle_test", False))
    quantile_mode = str(config.get("model", {}).get("quantile_mode", "none"))
    quantile_lambda = _validate_quantile_config(
        quantile_mode, config.get("model", {}).get("quantile_lambda")
    )
    topology_input_width = ztopo.raw_width(
        topology_mode,
        input_width_hint=config.get("model", {}).get("topology_input_width"),
    )
    structural_context_max_cycle_len = int(
        representation_config.get("structural_context_max_cycle_len", 10)
    )
    if topology_mode not in ("none", "longest", "spectrum", "hinge", "capacity_control"):
        raise ValueError(
            f"unknown topology_mode={topology_mode!r}; expected none, longest, "
            f"spectrum, hinge, or capacity_control"
        )
    if structural_context_mode not in ("none", "coarse_ring", "typed_ring"):
        raise ValueError(
            f"unknown structural_context_mode={structural_context_mode!r}; "
            f"expected none, coarse_ring, or typed_ring"
        )
    if structural_context_mode != "none" and structural_context_max_cycle_len < 3:
        raise ValueError(
            f"structural_context_max_cycle_len must be >= 3, got {structural_context_max_cycle_len}"
        )
    if patch_radius < 1:
        raise ValueError(f"patch_radius must be positive, got {patch_radius}")
    if context_radius not in (0, patch_radius + 1):
        raise ValueError(
            f"context_radius must be 0 or patch_radius+1={patch_radius + 1}, got {context_radius}"
        )
    shell_width = _shell_width_for_radius(patch_radius)
    model_config = config["model"]

    load_test = test_policy == "terminal"
    datasets = tuple(
        _load_zinc(data_root, split) for split in ("train", "val", *(["test"] if load_test else []))
    )
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    certificate_cache: dict[bytes, bytes] = {}
    records: list[list[GraphRecord]] = []
    feature_metadata: dict[str, Any] = {}
    topology_matrices: dict[str, np.ndarray] = {}
    loaded_splits = ("train", "valid", "test") if load_test else ("train", "valid")
    for split, dataset in zip(loaded_splits, datasets, strict=True):
        cache_key = "valid" if split == "val" else split
        if topology_mode != "none":
            matrix, frame, topo_meta = ztopo.matrices_for_split(
                cache_key,
                dataset,
                topology_mode,
                force=False,
                input_width=(
                    int(model_config.get("topology_input_width"))
                    if topology_mode == "capacity_control"
                    else None
                ),
            )
            topology_matrices[cache_key] = matrix
            feature_metadata.setdefault("topology_cache", {})[cache_key] = topo_meta
        split_records, metadata = _extract_split(
            dataset,
            split,
            certificate_cache,
            patch_radius=patch_radius,
            context_radius=context_radius,
            structural_mode=structural_context_mode,
            max_cycle_len=structural_context_max_cycle_len,
            topology_mode=topology_mode,
            topology_matrix=topology_matrices.get(cache_key),
            tokenizer_version=typed_tokenizer_version,
        )
        records.append(split_records)
        feature_metadata[split] = metadata
    if not load_test:
        feature_metadata["test"] = {"n_graphs": None, "blocked": True}
    feature_metadata["certificate_cache_entries"] = int(len(certificate_cache))

    train_records, valid_records = records[:2]
    test_records = records[2] if load_test else []
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")

    valid_train_data, valid_eval_data, valid_audit = _phase_data(
        train_records, valid_records, config=config
    )
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        typed_vocabulary_size=int(valid_audit["typed_vocabulary_size_with_oov"]),
        parent_vocabulary_size=int(valid_audit["parent_vocabulary_size_with_oov"]),
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 50)),
        shell_width=shell_width,
        context_width=int(CONTEXT_WIDTH if context_radius > patch_radius else 0),
        direct_token_readout=bool(model_config.get("direct_token_readout", False)),
        structural_context_vocabulary_size=int(
            valid_audit["structural_context_vocabulary_size"]
        ),
        topology_input_width=int(valid_audit["topology"]["input_width"]),
        topology_hidden_dim=topology_hidden_dim,
        topology_out_dim=topology_out_dim,
        topology_shuffle_test=topology_shuffle_test,
    )
    selected_epoch = int(valid_phase["selected_epoch"])

    # --- Primary benchmark protocol measurement (read-only, additive) ---    # The frozen validation-selected checkpoint (fit on official train only)
    # is evaluated exactly once on official test.  This is the
    # literature-comparable protocol (train -> validation selection -> frozen
    # checkpoint -> single test evaluation); it retrains nothing and never
    # refits vocabularies/standardizers on train+validation.  Gated by the
    # config flag evaluation.selection_checkpoint_test (default False), so
    # existing runs remain bit-identical when the flag is off.
    selection_test_phase = None
    selection_test_audit = {}
    if bool(config.get("evaluation", {}).get("selection_checkpoint_test", False)):
        if not load_test:
            raise RuntimeError(
                "evaluation.selection_checkpoint_test requires terminal test "
                "access (test_policy=terminal); official test is not loaded "
                "in non-terminal runs"
            )
        _sel_train_data, selection_test_graphs, selection_test_audit = _phase_data(
            train_records, test_records, config=config
        )
        # Guard: the refit of train-only vocabularies must equal the one the
        # selection phase actually used (catches any nondeterminism before
        # it silently corrupts a frozen-checkpoint evaluation).
        if int(selection_test_audit["typed_vocabulary_size_with_oov"]) != int(
            valid_audit["typed_vocabulary_size_with_oov"]
        ) or int(selection_test_audit["parent_vocabulary_size_with_oov"]) != int(
            valid_audit["parent_vocabulary_size_with_oov"]
        ):
            raise RuntimeError(
                "selection-checkpoint test vocab size mismatch vs selection "
                f"phase: {selection_test_audit['typed_vocabulary_size_with_oov']} "
                f"vs {valid_audit['typed_vocabulary_size_with_oov']}"
            )
        selection_test_loader = _make_loader(
            selection_test_graphs,
            int(model_config.get("batch_size", 128)),
            False,
            seed + 91013,
        )
        sel_targets, sel_predictions = _predict_values(
            valid_phase["model"], selection_test_loader, device
        )
        sel_point = _quantile_point_prediction_numpy(sel_predictions)
        selection_test_phase = {
            "mae": float(mean_absolute_error(sel_targets, sel_point)),
            "parameters": int(valid_phase["parameters"]),
            "checkpoint": "validation-selected best state; train-only fits",
            "targets": sel_targets.tolist(),
            "predictions": sel_point.tolist(),
        }
        if quantile_mode == "q10_q50_q90":
            selection_test_phase["quantile_predictions"] = {
                "q10": sel_predictions[:, 0].tolist(),
                "q50": sel_predictions[:, 1].tolist(),
                "q90": sel_predictions[:, 2].tolist(),
            }
        print(
            "selection-checkpoint test MAE = "
            f"{selection_test_phase['mae']!r} (frozen best-valid state, "
            "train-only fits)",
            flush=True,
        )

    refit_phase = None
    test_audit = {}
    skip_refit = bool(
        config.get("evaluation", {}).get("skip_train_valid_refit", False)
    )
    if load_test and skip_refit:
        print(
            "train+valid refit SKIPPED (evaluation.skip_train_valid_refit); "
            "only the frozen selection checkpoint is evaluated on test",
            flush=True,
        )

    if load_test and not skip_refit:
        refit_train_records = list(train_records) + list(valid_records)
        refit_train_data, refit_test_data, test_audit = _phase_data(
            refit_train_records, test_records, config=config
        )
        refit_phase = _train_phase(
            refit_train_data,
            refit_test_data,
            typed_vocabulary_size=int(test_audit["typed_vocabulary_size_with_oov"]),
            parent_vocabulary_size=int(test_audit["parent_vocabulary_size_with_oov"]),
            config=config,
            seed=seed,
            select_best=False,
            epochs=selected_epoch,
            shell_width=shell_width,
            context_width=int(CONTEXT_WIDTH if context_radius > patch_radius else 0),
            direct_token_readout=bool(model_config.get("direct_token_readout", False)),
            structural_context_vocabulary_size=int(
                test_audit["structural_context_vocabulary_size"]
            ),
            topology_input_width=int(test_audit["topology"]["input_width"]),
            topology_hidden_dim=topology_hidden_dim,
            topology_out_dim=topology_out_dim,
            topology_shuffle_test=False,
        )

    artifacts: list[str] = []
    if bool(model_config.get("save_state_dict", False)):
        state_path = Path(result_json).with_name(
            Path(result_json).stem + "_selection_state.pt"
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(valid_phase["model"].state_dict(), state_path)
        artifacts.append(str(state_path))
        if refit_phase is not None:
            refit_state_path = Path(result_json).with_name(
                Path(result_json).stem + "_refit_state.pt"
            )
            torch.save(refit_phase["model"].state_dict(), refit_state_path)
            artifacts.append(str(refit_state_path))

    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": seed,
        "test_policy": test_policy,
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                "train": len(datasets[0]) if datasets else None,
                "valid": len(datasets[1]) if len(datasets) > 1 else None,
                "test": len(datasets[2]) if load_test else None,
            },
            "source": source_audit(data_root),
            "valid_vocab_scope": "official train only",
            "test_vocab_scope": "official train+valid only",
            "test_labels_used_for_selection": False,
            "test_accessed": bool(load_test),
        },
        "representation": {
            "radius": patch_radius,
            "context_radius": context_radius,
            "typed_tokenizer_version": typed_tokenizer_version,
            "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
                typed_tokenizer_version, patch_radius
            ),
            "centres": "every atom",
            "exact_patch": (
                "rooted colored-incidence typed canonical token "
                f"({typed_tokenizer_version}); configurable-radius token with "
                "one-radius-lower parent token"
            ),
            "shell_width": shell_width,
            "context_width": int(CONTEXT_WIDTH if context_radius > patch_radius else 0),
            "relation_width": RELATION_WIDTH,
            "distance_buckets": ["1", "2", "3", "4", "5+"],
            "relation_definition": "distance bucket, shortest-path bond composition averaged over all shortest paths, path count, patch overlap/boundary overlap, and adjacent bond type",
            "readout": "configurable invariant unary/pair pooling with log mass",
            "node_readout": str(
                model_config.get("node_readout", model_config.get("readout", "moments"))
            ),
            "pair_readout": str(
                model_config.get("pair_readout", model_config.get("readout", "moments"))
            ),
            "direct_token_readout": bool(model_config.get("direct_token_readout", False)),
            "center_context": bool(model_config.get("center_context", False)),
            "center_context_definition": (
                "per-centre, per-distance mean/std/log-count of incident pair embeddings; zero-initialised residual update"
                if bool(model_config.get("center_context", False))
                else None
            ),
            "message_passing": bool(model_config.get("center_context", False)),
            "message_passing_layers": (
                1 if bool(model_config.get("center_context", False)) else 0
            ),
            "attention": False,
            "label_free": True,
            "structural_context": {
                "mode": structural_context_mode,
                "max_cycle_len": structural_context_max_cycle_len,
                "fusion": str(model_config.get("structural_context_fusion", "condition")),
                "dim": int(model_config.get("structural_context_dim", 8)),
                "embedding_rank": int(
                    model_config.get("structural_context_embedding_rank", 1)
                ),
                "condition_rank": int(
                    model_config.get("structural_context_condition_rank", 8)
                ),
                "definition": (
                    "center-anchored canonical typed cycle signature (node types + "
                    "edge types + cycle topology); NO_RING=0, UNK_CONTEXT=1; "
                    "conditioning delta forced to zero for NO_RING"
                    if structural_context_mode == "typed_ring"
                    else "4D coarse ring vector: in_cycle / cycle_count / min_cycle_len / "
                    "max_cycle_len; delta forced to zero for NO_RING"
                    if structural_context_mode == "coarse_ring"
                    else None
                ),
                "conditioning": (
                    "e_patch + W_out(tanh(W_p e_patch) * tanh(W_c e_ctx)); "
                    "no ring message passing, no ring nodes"
                    if structural_context_mode != "none"
                    and str(model_config.get("structural_context_fusion", "condition"))
                    == "condition"
                    else None
                ),
                "message_passing_on_rings": False,
            },
            "topology": {
                "mode": topology_mode,
                "input_width": int(topology_input_width),
                "hidden_dim": int(topology_hidden_dim),
                "out_dim": int(topology_out_dim),
                "feature_names": ztopo.feature_names(
                    topology_mode,
                    input_width=(
                        int(model_config.get("topology_input_width"))
                        if topology_mode == "capacity_control"
                        else None
                    ),
                ),
                "definition": (
                    "graph-level permutation-invariant global topology vector: "
                    "exact longest simple cycle + cycle-length spectrum + "
                    "minimum-cycle-basis summary + cycle rank; joint-trained "
                    "single graph head (NOT a second predictor, NOT a ring "
                    "network)"
                    if topology_mode != "none"
                    else None
                ),
                "target_formula_input": False,
                "cache": feature_metadata.get("topology_cache"),
                "standardizer": valid_audit.get("topology"),
            },
            "dimensions": (
                valid_phase["parameter_audit"]["dimensions"]
                if valid_phase.get("parameter_audit") is not None
                else None
            ),
            "parameter_audit": (
                valid_phase["parameter_audit"]
                if valid_phase.get("parameter_audit") is not None
                else None
            ),
        },
        "feature_build": feature_metadata,
        "vocabulary_audit": {
            "valid": valid_audit,
            "test_with_selection_checkpoint": (
                selection_test_audit if selection_test_phase is not None else None
            ),
            "test_after_train_valid_refit": test_audit,
        },
        "training": {
            **dict(model_config),
            "loss": (
                "L1 / mean absolute error"
                if quantile_mode == "none"
                else "quantile / 2*pinball_0.5 + lambda_q*(pinball_0.1+pinball_0.9)"
                if quantile_mode == "q10_q50_q90"
                else "2*pinball_0.5 (== L1 scale)"
            ),
            "quantile_mode": quantile_mode,
            "quantile_lambda": quantile_lambda,
            "quantile_parameterization": (
                "none"
                if quantile_mode != "q10_q50_q90"
                else "m, softplus(d_low), softplus(d_high) -> q10=m-d_low, q90=m+d_high (non-crossing)"
            ),
            "device": str(device),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "diagnostics": valid_phase.get("diagnostics", {}),
        "evaluation": {
            "valid": {
                "mae": float(valid_phase["mae"]),
                "best_mae": float(valid_phase["best_mae"]),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
                "targets": valid_phase["eval_targets"].tolist()
                if valid_phase.get("eval_targets") is not None
                else None,
                "predictions": _quantile_point_prediction_numpy(
                    valid_phase["eval_predictions"]
                ).tolist()
                if valid_phase.get("eval_predictions") is not None
                else None,
            },
            "test_after_train_valid_refit": (
                {
                    "mae": float(refit_phase["mae"]),
                    "epochs_run": int(refit_phase["epochs_run"]),
                    "parameter_audit": refit_phase["parameter_audit"],
                    "parameters": int(refit_phase["parameters"]),
                }
                if refit_phase is not None
                else {
                    "mae": None,
                    "epochs_run": 0,
                    "blocked": not load_test,
                    "skipped": bool(load_test and skip_refit),
                }
            ),
            "test_with_selection_checkpoint": (
                {
                    "mae": float(selection_test_phase["mae"]),
                    "parameters": int(selection_test_phase["parameters"]),
                    "checkpoint": selection_test_phase["checkpoint"],
                    "targets": selection_test_phase["targets"],
                    "predictions": selection_test_phase["predictions"],
                }
                if selection_test_phase is not None
                else {"mae": None, "blocked": True}
            ),
            "parameters": int(valid_phase["parameters"]),
        },
        "artifacts": artifacts,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "pynauty": importlib.metadata.version("pynauty"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    valid_eval_raw = (
        np.asarray(valid_phase["eval_predictions"], dtype=np.float64)
        if valid_phase.get("eval_predictions") is not None
        else None
    )
    if quantile_mode == "q10_q50_q90" and valid_eval_raw is not None:
        if valid_eval_raw.ndim != 2 or valid_eval_raw.shape[1] != 3:
            raise RuntimeError(
                "q10_q50_q90 mode must produce (n, 3) predictions, got "
                f"shape={valid_eval_raw.shape}"
            )
        q10 = valid_eval_raw[:, 0]
        q50 = valid_eval_raw[:, 1]
        q90 = valid_eval_raw[:, 2]
        result["evaluation"]["valid"]["quantile_predictions"] = {
            "q10": q10.tolist(),
            "q50": q50.tolist(),
            "q90": q90.tolist(),
        }
        # Per-molecule validation outputs (molecule_id by val-split position,
        # which is the official val dataset order; the audit feature tables
        # use the identical ordering, verified bit-exact in the post-v4 audit).
        per_molecule: list[dict[str, Any]] = []
        valid_targets = np.asarray(result["evaluation"]["valid"]["targets"], dtype=np.float64)
        for index in range(int(valid_targets.shape[0])):
            per_molecule.append(
                {
                    "molecule_id": f"valid:{index:04d}",
                    "target": float(valid_targets[index]),
                    "q10": float(q10[index]),
                    "q50": float(q50[index]),
                    "q90": float(q90[index]),
                    "width80": float(q90[index] - q10[index]),
                    "lower_width": float(q50[index] - q10[index]),
                    "upper_width": float(q90[index] - q50[index]),
                    "abs_error": float(abs(valid_targets[index] - q50[index])),
                    "signed_residual": float(valid_targets[index] - q50[index]),
                }
            )
        per_molecule_path = Path(result_json).with_name(
            "validation_predictions.csv"
        )
        with per_molecule_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_molecule[0].keys()))
            writer.writeheader()
            writer.writerows(per_molecule)
        result["artifacts"] = list(result["artifacts"]) + [
            str(per_molecule_path)
        ]
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "valid_mae": result["evaluation"]["valid"]["mae"],
                "test_mae": result["evaluation"]["test_after_train_valid_refit"]["mae"],
                "selected_epoch": result["evaluation"]["valid"]["selected_epoch"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
