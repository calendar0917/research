"""Clean MolHIV structure--attribute fusion screen.

The experiment keeps the full all-centre radius-2 induced substrate fixed and
separates five choices that were previously mixed together:

* topology-only structural roles (coarse or rooted WL);
* strict chemical attributes with OGB degree/ring coordinates removed;
* marginal concatenation;
* aligned role x attribute statistics;
* graph readout and a fixed XGBoost probe.

Within-patch shuffles preserve every patch's structural and attribute
marginals while breaking their alignment.  The official test split is never
encoded or evaluated.  Exact rooted topology is used only in a small,
label-free coverage/collision audit before the supervised screen.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    BOND_DIM,
    REPO_ROOT,
    _aggregate_folds,
    _cycle_membership,
    _delta,
    _delta_against_controls,
    _ego_distances,
    _fit_auc_views,
    _fold_indices,
    _frozen_s_rows,
    _graph_context,
    _resolve,
    _select_rows,
    _sha256,
    _summary,
    _write_json,
    atomic_number_group,
    compact_bond_semantics,
    one_hot_index,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/structural_role_fusion_screen.yaml"
)

STRICT_ATOM_DIM = 40
ROLE_SCHEMAS = ("coarse", "rooted_wl")


@dataclass(frozen=True)
class PatchRoles:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    node_ids: np.ndarray
    edge_ids: np.ndarray
    node_keys: tuple[int, ...]
    edge_keys: tuple[int, ...]
    patch_key: int


def _stable_int(namespace: str, token: object) -> int:
    payload = (namespace + "|" + repr(token)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _stable_bin(namespace: str, token: object, n_bins: int) -> int:
    if int(n_bins) <= 0:
        raise ValueError("hash dimensions must be positive")
    return _stable_int(namespace, token) % int(n_bins)


def strict_atom_semantics(
    raw_atom_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chemistry-only atom features and compact element-group ids.

    OGB coordinates 2 (total degree) and 8 (is-in-ring) are deliberately
    omitted because they directly duplicate topology-derived information.
    """
    raw = np.asarray(raw_atom_features, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected OGB atom features [N,9], got {raw.shape}")
    rows: list[np.ndarray] = []
    groups: list[int] = []
    for feature in raw:
        group = atomic_number_group(int(feature[0]))
        groups.append(group)
        charge = int(feature[3])
        charge_bucket = (
            3 if charge == 11 else 0 if charge < 5 else 1 if charge == 5 else 2
        )
        num_h = int(feature[4])
        radical = int(feature[5])
        radical_bucket = (
            3 if radical == 5 else 0 if radical == 0 else 1 if radical == 1 else 2
        )
        pieces = (
            one_hot_index(group, 14),
            one_hot_index(int(feature[1]), 5),
            one_hot_index(charge_bucket, 4),
            one_hot_index(num_h if 0 <= num_h <= 3 else 4, 5),
            one_hot_index(radical_bucket, 4),
            one_hot_index(int(feature[6]), 6),
            one_hot_index(int(feature[7]), 2),
        )
        row = np.concatenate(pieces).astype(np.float32, copy=False)
        if row.shape != (STRICT_ATOM_DIM,):
            raise RuntimeError(f"strict atom dimension mismatch: {row.shape}")
        rows.append(row)
    matrix = (
        np.stack(rows, axis=0)
        if rows
        else np.zeros((0, STRICT_ATOM_DIM), dtype=np.float32)
    )
    return matrix, np.asarray(groups, dtype=np.int64)


def representation_dimensions(config: Mapping[str, Any]) -> dict[str, int]:
    node_roles = int(config["node_role_bins"])
    edge_roles = int(config["edge_role_bins"])
    return {
        "node_role": node_roles,
        "edge_role": edge_roles,
        "node_attribute": STRICT_ATOM_DIM,
        "edge_attribute": BOND_DIM,
        "node_raw": node_roles * STRICT_ATOM_DIM,
        "edge_raw": edge_roles * BOND_DIM,
        "typed_edge": int(config["typed_edge_bins"]),
        "node_binding": node_roles * STRICT_ATOM_DIM,
        "edge_binding": edge_roles * BOND_DIM,
        "context": 5,
    }


def _coarse_patch_roles(
    graph,
    center: int,
    config: Mapping[str, Any],
) -> PatchRoles:
    distances = _ego_distances(graph, int(center), int(config["radius"]))
    nodes = tuple(sorted(distances))
    induced = graph.induced(set(nodes))
    cycle_nodes, cycle_edges = _cycle_membership(induced)
    degree_bins = int(config["degree_bins"])
    node_role_bins = int(config["node_role_bins"])
    edge_role_bins = int(config["edge_role_bins"])
    node_ids = []
    for node in nodes:
        degree = min(len(induced.neighbors(node)), degree_bins - 1)
        role = ((int(distances[node]) * degree_bins + degree) * 2) + int(
            node in cycle_nodes
        )
        if role >= node_role_bins:
            raise ValueError("node_role_bins is too small for the coarse schema")
        node_ids.append(role)
    edges = tuple(sorted(induced.edges()))
    shell_pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    edge_ids = []
    for left, right in edges:
        pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        role = shell_pairs.index(pair) * 2 + int(
            graph.edge_key(left, right) in cycle_edges
        )
        if role >= edge_role_bins:
            raise ValueError("edge_role_bins is too small for the coarse schema")
        edge_ids.append(role)
    patch_key = _stable_int(
        "coarse-patch",
        (
            tuple(sorted(Counter(node_ids).items())),
            tuple(sorted(Counter(edge_ids).items())),
        ),
    )
    return PatchRoles(
        nodes=nodes,
        edges=edges,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        edge_ids=np.asarray(edge_ids, dtype=np.int64),
        node_keys=tuple(int(value) for value in node_ids),
        edge_keys=tuple(int(value) for value in edge_ids),
        patch_key=patch_key,
    )


def _rooted_wl_patch_roles(
    graph,
    center: int,
    config: Mapping[str, Any],
) -> PatchRoles:
    distances = _ego_distances(graph, int(center), int(config["radius"]))
    nodes = tuple(sorted(distances))
    induced = graph.induced(set(nodes))
    colors = {
        node: _stable_int(
            "rooted-wl-initial",
            (
                int(node == center),
                int(distances[node]),
                len(induced.neighbors(node)),
            ),
        )
        for node in nodes
    }
    for iteration in range(int(config["wl_rounds"])):
        colors = {
            node: _stable_int(
                f"rooted-wl-{iteration}",
                (
                    colors[node],
                    tuple(sorted(colors[neighbor] for neighbor in induced.neighbors(node))),
                ),
            )
            for node in nodes
        }
    node_keys = tuple(int(colors[node]) for node in nodes)
    node_ids = np.asarray(
        [
            _stable_bin("rooted-wl-node-role", key, int(config["node_role_bins"]))
            for key in node_keys
        ],
        dtype=np.int64,
    )
    edges = tuple(sorted(induced.edges()))
    edge_keys_list: list[int] = []
    edge_ids: list[int] = []
    for left, right in edges:
        token = (
            tuple(sorted((colors[left], colors[right]))),
            tuple(sorted((int(distances[left]), int(distances[right])))),
        )
        key = _stable_int("rooted-wl-edge-key", token)
        edge_keys_list.append(key)
        edge_ids.append(
            _stable_bin(
                "rooted-wl-edge-role", key, int(config["edge_role_bins"])
            )
        )
    patch_key = _stable_int(
        "rooted-wl-patch",
        (
            int(colors[int(center)]),
            tuple(sorted(Counter(node_keys).items())),
            tuple(sorted(Counter(edge_keys_list).items())),
        ),
    )
    return PatchRoles(
        nodes=nodes,
        edges=edges,
        node_ids=node_ids,
        edge_ids=np.asarray(edge_ids, dtype=np.int64),
        node_keys=node_keys,
        edge_keys=tuple(edge_keys_list),
        patch_key=patch_key,
    )


def patch_roles(
    graph,
    center: int,
    schema: str,
    config: Mapping[str, Any],
) -> PatchRoles:
    if schema == "coarse":
        return _coarse_patch_roles(graph, center, config)
    if schema == "rooted_wl":
        return _rooted_wl_patch_roles(graph, center, config)
    raise ValueError(f"unknown role schema: {schema}")


def _entity_statistics(
    roles: np.ndarray,
    attributes: np.ndarray,
    n_roles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if roles.ndim != 1 or attributes.ndim != 2 or roles.shape[0] != attributes.shape[0]:
        raise ValueError("roles and attributes are not aligned")
    n_entities = int(roles.shape[0])
    if n_entities == 0:
        return (
            np.zeros(n_roles, dtype=np.float32),
            np.zeros(attributes.shape[1], dtype=np.float32),
            np.zeros((n_roles, attributes.shape[1]), dtype=np.float32),
        )
    role_marginal = np.bincount(roles, minlength=n_roles).astype(np.float32)
    role_marginal /= float(n_entities)
    attribute_marginal = attributes.mean(axis=0, dtype=np.float32)
    joint = np.zeros((n_roles, attributes.shape[1]), dtype=np.float32)
    np.add.at(joint, roles, attributes)
    joint /= float(n_entities)
    return role_marginal, attribute_marginal, joint


def _typed_edge_histogram(
    edge_roles: np.ndarray,
    edge_endpoint_rows: Sequence[tuple[int, int]],
    node_groups: np.ndarray,
    raw_bonds: Sequence[tuple[int, ...]],
    n_bins: int,
) -> np.ndarray:
    output = np.zeros(int(n_bins), dtype=np.float32)
    for index, (left_row, right_row) in enumerate(edge_endpoint_rows):
        groups = tuple(sorted((int(node_groups[left_row]), int(node_groups[right_row]))))
        token = (int(edge_roles[index]), groups, tuple(raw_bonds[index]))
        output[_stable_bin("typed-edge", token, int(n_bins))] += 1.0
    if edge_roles.size:
        output /= float(edge_roles.size)
    return output


def _patch_statistics(
    roles: PatchRoles,
    node_attributes: np.ndarray,
    node_groups: np.ndarray,
    edge_attributes: np.ndarray,
    raw_bonds: Sequence[tuple[int, ...]],
    config: Mapping[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    dims = representation_dimensions(config)
    node_attrs = node_attributes
    groups = node_groups
    edge_attrs = edge_attributes
    bonds = list(raw_bonds)
    if rng is not None:
        if node_attrs.shape[0] > 1:
            order = rng.permutation(node_attrs.shape[0])
            node_attrs = node_attrs[order]
            groups = groups[order]
        if edge_attrs.shape[0] > 1:
            order = rng.permutation(edge_attrs.shape[0])
            edge_attrs = edge_attrs[order]
            bonds = [bonds[int(index)] for index in order]
    node_role, node_attribute, node_joint = _entity_statistics(
        roles.node_ids, node_attrs, dims["node_role"]
    )
    edge_role, edge_attribute, edge_joint = _entity_statistics(
        roles.edge_ids, edge_attrs, dims["edge_role"]
    )
    node_position = {node: index for index, node in enumerate(roles.nodes)}
    endpoints = [
        (node_position[left], node_position[right]) for left, right in roles.edges
    ]
    typed_edge = _typed_edge_histogram(
        roles.edge_ids,
        endpoints,
        groups,
        bonds,
        dims["typed_edge"],
    )
    node_binding = node_joint - node_role[:, None] * node_attribute[None, :]
    edge_binding = edge_joint - edge_role[:, None] * edge_attribute[None, :]
    return {
        "node_role": node_role,
        "edge_role": edge_role,
        "node_attribute": node_attribute,
        "edge_attribute": edge_attribute,
        "node_raw": node_joint.reshape(-1),
        "edge_raw": edge_joint.reshape(-1),
        "typed_edge": typed_edge,
        "node_binding": node_binding.reshape(-1),
        "edge_binding": edge_binding.reshape(-1),
    }


def patch_feature_rows(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    schema: str,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return one primitive feature row per atom-centred patch.

    ``graph_features`` intentionally performs an all-centre mean readout.  A
    separate row-level helper is kept here for controlled readout ablations:
    callers can preserve the population of local patches and apply a fixed
    permutation-invariant distribution readout without changing the patch
    object or its structural/attribute semantics.

    The returned arrays have shape ``[n_centres, width]`` (except for the
    empty-graph fallback).  The mean of each primitive array is exactly the
    corresponding primitive block produced by ``graph_features``.
    """
    atom_semantics, atom_groups = strict_atom_semantics(node_features)
    bond_semantics = {
        graph.edge_key(left, right): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    raw_bond = {
        graph.edge_key(left, right): tuple(
            int(value) for value in np.asarray(values).reshape(-1)
        )
        for (left, right), values in edge_features.items()
    }
    centers = list(graph.nodes)
    max_centers = config.get("max_centers_per_graph")
    if max_centers is not None and len(centers) > int(max_centers):
        centers = centers[: int(max_centers)]
    keys = (
        "node_role",
        "edge_role",
        "node_attribute",
        "edge_attribute",
        "node_raw",
        "edge_raw",
        "typed_edge",
        "node_binding",
        "edge_binding",
    )
    rows: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for center in centers:
        roles = patch_roles(graph, int(center), schema, config)
        node_attrs = np.stack(
            [atom_semantics[int(node)] for node in roles.nodes], axis=0
        ).astype(np.float32, copy=False)
        groups = np.asarray(
            [atom_groups[int(node)] for node in roles.nodes], dtype=np.int64
        )
        if roles.edges:
            edge_attrs = np.stack(
                [
                    bond_semantics[graph.edge_key(left, right)]
                    for left, right in roles.edges
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            bonds = [
                raw_bond[graph.edge_key(left, right)]
                for left, right in roles.edges
            ]
        else:
            edge_attrs = np.zeros((0, BOND_DIM), dtype=np.float32)
            bonds = []
        node_counts.append(len(roles.nodes))
        edge_counts.append(len(roles.edges))
        values = _patch_statistics(
            roles, node_attrs, groups, edge_attrs, bonds, config
        )
        for key in keys:
            rows[key].append(np.asarray(values[key], dtype=np.float32).reshape(-1))

    dims = representation_dimensions(config)
    output: dict[str, np.ndarray] = {}
    for key in keys:
        if rows[key]:
            output[key] = np.stack(rows[key], axis=0).astype(np.float32, copy=False)
        else:
            output[key] = np.zeros((0, dims[key]), dtype=np.float32)
    output["node_count"] = np.asarray(node_counts, dtype=np.float32)
    output["edge_count"] = np.asarray(edge_counts, dtype=np.float32)
    output["context"] = _graph_context(
        graph, len(centers), node_counts, edge_counts
    )
    return output


def graph_features(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    schema: str,
    config: Mapping[str, Any],
    *,
    shuffle_repeats: int = 0,
    shuffle_seed: int = 0,
) -> dict[str, np.ndarray]:
    atom_semantics, atom_groups = strict_atom_semantics(node_features)
    bond_semantics = {
        graph.edge_key(left, right): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    raw_bond = {
        graph.edge_key(left, right): tuple(
            int(value) for value in np.asarray(values).reshape(-1)
        )
        for (left, right), values in edge_features.items()
    }
    centers = list(graph.nodes)
    max_centers = config.get("max_centers_per_graph")
    if max_centers is not None and len(centers) > int(max_centers):
        # This branch is smoke-only.  Main protocols use every atom centre.
        centers = centers[: int(max_centers)]
    keys = (
        "node_role",
        "edge_role",
        "node_attribute",
        "edge_attribute",
        "node_raw",
        "edge_raw",
        "typed_edge",
        "node_binding",
        "edge_binding",
    )
    true_accumulator = {key: [] for key in keys}
    shuffled_accumulators = [
        {key: [] for key in keys} for _ in range(int(shuffle_repeats))
    ]
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for center in centers:
        roles = patch_roles(graph, int(center), schema, config)
        node_attrs = np.stack(
            [atom_semantics[int(node)] for node in roles.nodes], axis=0
        ).astype(np.float32, copy=False)
        groups = np.asarray(
            [atom_groups[int(node)] for node in roles.nodes], dtype=np.int64
        )
        if roles.edges:
            edge_attrs = np.stack(
                [bond_semantics[graph.edge_key(left, right)] for left, right in roles.edges],
                axis=0,
            ).astype(np.float32, copy=False)
            bonds = [raw_bond[graph.edge_key(left, right)] for left, right in roles.edges]
        else:
            edge_attrs = np.zeros((0, BOND_DIM), dtype=np.float32)
            bonds = []
        node_counts.append(len(roles.nodes))
        edge_counts.append(len(roles.edges))
        true = _patch_statistics(
            roles, node_attrs, groups, edge_attrs, bonds, config
        )
        for key, value in true.items():
            true_accumulator[key].append(value)
        for repeat, accumulator in enumerate(shuffled_accumulators):
            rng = np.random.default_rng(
                int(shuffle_seed) + 1000003 * (repeat + 1) + 7919 * int(center)
            )
            shuffled = _patch_statistics(
                roles,
                node_attrs,
                groups,
                edge_attrs,
                bonds,
                config,
                rng=rng,
            )
            for key, value in shuffled.items():
                accumulator[key].append(value)
    dims = representation_dimensions(config)

    def average(accumulator: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key, values in accumulator.items():
            if values:
                output[key] = np.mean(
                    np.stack(values, axis=0), axis=0, dtype=np.float32
                ).reshape(-1)
            else:
                output[key] = np.zeros(dims[key], dtype=np.float32)
            if not np.all(np.isfinite(output[key])):
                raise FloatingPointError(f"non-finite block {key}")
        return output

    output = average(true_accumulator)
    output["context"] = _graph_context(
        graph, len(centers), node_counts, edge_counts
    )
    for repeat, accumulator in enumerate(shuffled_accumulators):
        shuffled = average(accumulator)
        for key, value in shuffled.items():
            output[f"{key}_shuffled_{repeat}"] = value
    return output


def assemble_views(
    blocks: Mapping[str, np.ndarray],
    frozen_s: np.ndarray,
    shuffle_repeats: int,
) -> dict[str, np.ndarray]:
    context = np.asarray(blocks["context"], dtype=np.float32)
    axis = 0 if context.ndim == 1 else 1

    def join(*parts: np.ndarray) -> np.ndarray:
        return np.concatenate(parts, axis=axis).astype(np.float32, copy=False)

    topology = join(blocks["node_role"], blocks["edge_role"], context)
    attribute = join(
        blocks["node_attribute"], blocks["edge_attribute"], context
    )
    marginals = join(
        blocks["node_role"],
        blocks["edge_role"],
        blocks["node_attribute"],
        blocks["edge_attribute"],
        context,
    )
    raw_factorized = join(
        marginals, blocks["node_raw"], blocks["edge_raw"]
    )
    raw = join(raw_factorized, blocks["typed_edge"])
    centered = join(
        marginals, blocks["node_binding"], blocks["edge_binding"]
    )
    result = {
        "a_strict": attribute,
        "t": topology,
        "t_a": marginals,
        "f_raw_factorized": raw_factorized,
        "f_raw": raw,
        "f_centered": centered,
        "mixed_s": np.asarray(frozen_s, dtype=np.float32),
        "mixed_s_f_raw_factorized": join(
            np.asarray(frozen_s, dtype=np.float32), raw_factorized
        ),
        "mixed_s_f_raw": join(np.asarray(frozen_s, dtype=np.float32), raw),
        "mixed_s_f_centered": join(
            np.asarray(frozen_s, dtype=np.float32), centered
        ),
    }
    for repeat in range(int(shuffle_repeats)):
        suffix = f"_shuffled_{repeat}"
        shuffled_marginals = join(
            blocks[f"node_role{suffix}"],
            blocks[f"edge_role{suffix}"],
            blocks[f"node_attribute{suffix}"],
            blocks[f"edge_attribute{suffix}"],
            context,
        )
        shuffled_raw = join(
            shuffled_marginals,
            blocks[f"node_raw{suffix}"],
            blocks[f"edge_raw{suffix}"],
            blocks[f"typed_edge{suffix}"],
        )
        shuffled_raw_factorized = join(
            shuffled_marginals,
            blocks[f"node_raw{suffix}"],
            blocks[f"edge_raw{suffix}"],
        )
        shuffled_centered = join(
            shuffled_marginals,
            blocks[f"node_binding{suffix}"],
            blocks[f"edge_binding{suffix}"],
        )
        result[f"f_raw_shuffled_{repeat}"] = shuffled_raw
        result[f"f_raw_factorized_shuffled_{repeat}"] = (
            shuffled_raw_factorized
        )
        result[f"f_centered_shuffled_{repeat}"] = shuffled_centered
    return result


def audit_invariance(
    bundle,
    indices: np.ndarray,
    schema: str,
    representation: Mapping[str, Any],
    *,
    n_graphs: int,
    permutations_per_graph: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    rng = np.random.default_rng(int(seed))
    candidates = np.asarray(indices, dtype=np.int64)
    if candidates.size > int(n_graphs):
        candidates = rng.choice(candidates, size=int(n_graphs), replace=False)
    keys = (
        "node_role",
        "edge_role",
        "node_attribute",
        "edge_attribute",
        "node_raw",
        "edge_raw",
        "typed_edge",
        "node_binding",
        "edge_binding",
        "context",
    )
    drift = {key: [] for key in keys}
    for raw_index in candidates:
        index = int(raw_index)
        base = graph_features(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            schema,
            representation,
        )
        for _ in range(int(permutations_per_graph)):
            permutation = rng.permutation(bundle.graphs[index].n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                bundle.graphs[index],
                bundle.node_feats[index],
                bundle.edge_feats[index],
                permutation,
            )
            changed = graph_features(
                changed_graph,
                changed_nodes,
                changed_edges,
                schema,
                representation,
            )
            for key in keys:
                drift[key].append(
                    float(np.max(np.abs(base[key] - changed[key]), initial=0.0))
                )
    summaries = {key: _summary(values) for key, values in drift.items()}
    return {
        "schema": schema,
        "n_graphs": int(candidates.size),
        "permutations_per_graph": int(permutations_per_graph),
        "tolerance": float(tolerance),
        "maximum_drift": {key: row["maximum"] for key, row in summaries.items()},
        "pass": bool(
            all(row["maximum"] <= float(tolerance) for row in summaries.values())
        ),
    }


def _rooted_exact_certificate(graph, center: int, radius: int) -> bytes:
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover
        raise ImportError("exact topology audit requires pynauty") from exc
    distances = _ego_distances(graph, int(center), int(radius))
    nodes = tuple(sorted(distances))
    local = {node: index for index, node in enumerate(nodes)}
    adjacency = {
        local[node]: [local[neighbor] for neighbor in graph.neighbors(node) if neighbor in local]
        for node in nodes
    }
    root = local[int(center)]
    rest = set(range(len(nodes))) - {root}
    coloring = [{root}]
    if rest:
        coloring.append(rest)
    nauty_graph = pynauty.Graph(
        number_of_vertices=len(nodes),
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    return bytes(pynauty.certificate(nauty_graph))


def _mapping_collision_summary(mapping: Mapping[int, set[bytes]]) -> dict[str, Any]:
    widths = np.asarray([len(values) for values in mapping.values()], dtype=np.int64)
    return {
        "n_representation_keys": int(len(mapping)),
        "ambiguous_key_fraction": float(np.mean(widths > 1)) if widths.size else 0.0,
        "maximum_exact_types_per_key": int(widths.max()) if widths.size else 0,
        "mean_exact_types_per_key": float(widths.mean()) if widths.size else 0.0,
    }


def exact_topology_audit(
    bundle,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    representation: Mapping[str, Any],
    *,
    max_train_graphs: int,
    max_valid_graphs: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    def choose(indices: np.ndarray, maximum: int) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.size <= int(maximum):
            return np.sort(values)
        return np.sort(rng.choice(values, size=int(maximum), replace=False))

    selected_train = choose(train_indices, max_train_graphs)
    selected_valid = choose(valid_indices, max_valid_graphs)
    train_exact: list[bytes] = []
    valid_exact: list[bytes] = []
    coarse_map: dict[int, set[bytes]] = defaultdict(set)
    wl_map: dict[int, set[bytes]] = defaultdict(set)

    def consume(indices: np.ndarray, destination: list[bytes]) -> None:
        for raw_index in indices:
            graph = bundle.graphs[int(raw_index)]
            for center in graph.nodes:
                certificate = _rooted_exact_certificate(
                    graph, int(center), int(representation["radius"])
                )
                destination.append(certificate)
                coarse_map[
                    patch_roles(graph, int(center), "coarse", representation).patch_key
                ].add(certificate)
                wl_map[
                    patch_roles(graph, int(center), "rooted_wl", representation).patch_key
                ].add(certificate)

    consume(selected_train, train_exact)
    consume(selected_valid, valid_exact)
    train_types = set(train_exact)
    valid_types = set(valid_exact)
    return {
        "train_graphs": int(selected_train.size),
        "valid_graphs": int(selected_valid.size),
        "train_patches": int(len(train_exact)),
        "valid_patches": int(len(valid_exact)),
        "train_exact_types": int(len(train_types)),
        "valid_exact_types": int(len(valid_types)),
        "valid_patch_seen_rate": float(
            np.mean([value in train_types for value in valid_exact])
        )
        if valid_exact
        else 0.0,
        "valid_type_seen_rate": float(
            np.mean([value in train_types for value in valid_types])
        )
        if valid_types
        else 0.0,
        "coarse": _mapping_collision_summary(coarse_map),
        "rooted_wl": _mapping_collision_summary(wl_map),
        "labels_used": False,
    }


def _build_dataset_blocks(
    bundle,
    indices: np.ndarray,
    schema: str,
    representation: Mapping[str, Any],
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    assert bundle.node_feats is not None and bundle.edge_feats is not None
    unique_indices = np.asarray(indices, dtype=np.int64)
    first_index = int(unique_indices[0])
    first = graph_features(
        bundle.graphs[first_index],
        bundle.node_feats[first_index],
        bundle.edge_feats[first_index],
        schema,
        representation,
        shuffle_repeats=repeats,
        shuffle_seed=seed + first_index,
    )
    arrays = {
        name: np.zeros((unique_indices.size, value.shape[0]), dtype=np.float32)
        for name, value in first.items()
    }
    for position, raw_index in enumerate(unique_indices):
        index = int(raw_index)
        blocks = first if position == 0 else graph_features(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            schema,
            representation,
            shuffle_repeats=repeats,
            shuffle_seed=seed + index,
        )
        for name in arrays:
            arrays[name][position] = blocks[name]
        if position and position % 500 == 0:
            print(
                f"[{schema}] feature graphs: {position}/{unique_indices.size}",
                flush=True,
            )
    return unique_indices, arrays


def _cross_schema_delta(
    schema_results: Mapping[str, Mapping[str, Any]],
    candidate_schema: str,
    baseline_schema: str,
    view: str,
) -> dict[str, Any]:
    candidate = schema_results[candidate_schema]["folds"]
    baseline = schema_results[baseline_schema]["folds"]
    values = [
        float(left["scores"][view]["valid_auc"] - right["scores"][view]["valid_auc"])
        for left, right in zip(candidate, baseline, strict=True)
    ]
    return {
        "candidate": f"{candidate_schema}:{view}",
        "baseline": f"{baseline_schema}:{view}",
        "fold_deltas": values,
        "mean_delta": float(np.mean(values)),
        "fold_wins": int(np.sum(np.asarray(values) > 0.0)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MolHIV clean structural-role fusion screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Full all-centre radius-2 induced patches; strict atom attributes exclude OGB degree and is-in-ring. No K-SVD, attention, Optuna, or official test.",
        "",
        "## Exact topology audit",
        "",
    ]
    audit = result["exact_topology_audit"]
    lines.extend(
        [
            f"- train/valid sampled patches: {audit['train_patches']} / {audit['valid_patches']}",
            f"- exact rooted topology train-to-valid patch coverage: {audit['valid_patch_seen_rate']:.4f}",
            f"- coarse ambiguous-key fraction: {audit['coarse']['ambiguous_key_fraction']:.4f}",
            f"- rooted-WL ambiguous-key fraction: {audit['rooted_wl']['ambiguous_key_fraction']:.4f}",
            "",
        ]
    )
    for schema, payload in result["schemas"].items():
        lines.extend(
            [
                f"## {schema}",
                "",
                f"Relabel invariance: **{'PASS' if payload['object_audit']['pass'] else 'FAIL'}**",
                "",
                "| view | mean validation ROC-AUC | fold std |",
                "|---|---:|---:|",
            ]
        )
        for name, row in payload["aggregate"].items():
            lines.append(
                f"| `{name}` | {row['mean_auc']:.6f} | {row['std_auc']:.6f} |"
            )
        lines.extend(["", "Gates:", ""])
        for name, gate in payload["gates"].items():
            lines.append(
                f"- `{name}`: {gate['mean_delta']:+.6f}, wins {gate['fold_wins']}/{result['n_folds']} — **{'PASS' if gate['passed'] else 'FAIL'}**"
            )
        lines.append("")
    cross = result.get("cross_schema_gate")
    lines.extend(["## Decision", ""])
    if cross is not None:
        lines.append(
            f"- rooted-WL factorized raw fusion minus coarse: {cross['mean_delta']:+.6f}, wins {cross['fold_wins']}/{result['n_folds']} — **{'PASS' if cross['passed'] else 'FAIL'}**"
        )
    lines.extend([f"- Decision: **{result['decision']}**", ""])
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    classifier = config["classifier"]
    result_json = _resolve(config["output_json"])
    result_markdown = _resolve(config["output_markdown"])
    frozen_path = _resolve(data_config["frozen_features"])
    folds_path = _resolve(data_config["scaffold_folds"])
    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    labels = bundle.y.astype(np.int64)
    start = time.perf_counter()

    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    split_mode = str(screen.get("split", "scaffold_folds"))
    if split_mode == "scaffold_folds":
        fold_ids = sorted(
            int(name.removeprefix("fold_").removesuffix("_train_indices"))
            for name in fold_archive
            if name.startswith("fold_") and name.endswith("_train_indices")
        )
        selections = [
            _fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids
        ]
    elif split_mode == "official_valid":
        train = original[
            np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
        ]
        valid = original[
            np.asarray(fold_archive["official_valid_indices"], dtype=np.int64)
        ]
        train = _select_rows(
            labels,
            train,
            screen.get("max_train_graphs_per_fold"),
            int(screen["seed"]),
        )
        valid = _select_rows(
            labels,
            valid,
            screen.get("max_valid_graphs_per_fold"),
            int(screen["seed"]) + 1,
        )
        fold_ids = [0]
        selections = [(train, valid)]
    else:
        raise ValueError(f"unknown split mode: {split_mode}")
    selected_union = np.unique(
        np.concatenate([np.concatenate(pair) for pair in selections])
    ).astype(np.int64)
    print(
        f"selected graphs={selected_union.size}; folds={fold_ids}; "
        f"schemas={list(representation['schemas'])}",
        flush=True,
    )

    exact_cfg = config["exact_audit"]
    exact_audit = exact_topology_audit(
        bundle,
        selections[0][0],
        selections[0][1],
        representation,
        max_train_graphs=int(exact_cfg["max_train_graphs"]),
        max_valid_graphs=int(exact_cfg["max_valid_graphs"]),
        seed=int(exact_cfg["seed"]),
    )
    print(
        "exact audit: "
        f"valid patch coverage={exact_audit['valid_patch_seen_rate']:.4f}; "
        f"coarse ambiguity={exact_audit['coarse']['ambiguous_key_fraction']:.4f}; "
        f"wl ambiguity={exact_audit['rooted_wl']['ambiguous_key_fraction']:.4f}",
        flush=True,
    )

    audit_candidates = original[
        np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
    ]
    shuffle_repeats = int(screen["shuffle_repeats"])
    model_seeds = [int(value) for value in classifier.get("model_seeds", [0])]
    schema_results: dict[str, Any] = {}
    for schema in representation["schemas"]:
        schema = str(schema)
        if schema not in ROLE_SCHEMAS:
            raise ValueError(f"unsupported schema in config: {schema}")
        object_audit = audit_invariance(
            bundle,
            audit_candidates,
            schema,
            representation,
            n_graphs=int(config["audit"]["n_graphs"]),
            permutations_per_graph=int(config["audit"]["permutations_per_graph"]),
            seed=int(config["audit"]["seed"]),
            tolerance=float(config["audit"]["tolerance"]),
        )
        if not object_audit["pass"]:
            raise RuntimeError(f"{schema} failed relabel invariance audit")
        unique_indices, arrays = _build_dataset_blocks(
            bundle,
            selected_union,
            schema,
            representation,
            shuffle_repeats,
            int(screen["seed"]),
        )
        index_to_row = {
            int(index): position for position, index in enumerate(unique_indices)
        }
        folds: list[dict[str, Any]] = []
        for fold, (train_indices, valid_indices) in zip(
            fold_ids, selections, strict=True
        ):
            fold_indices = np.concatenate([train_indices, valid_indices]).astype(
                np.int64
            )
            rows = np.asarray(
                [index_to_row[int(index)] for index in fold_indices], dtype=np.int64
            )
            fold_blocks = {name: value[rows] for name, value in arrays.items()}
            views = assemble_views(
                fold_blocks,
                _frozen_s_rows(frozen, fold_indices),
                shuffle_repeats,
            )
            score_views = screen.get("score_views")
            if score_views is not None:
                requested = [str(value) for value in score_views]
                missing = [name for name in requested if name not in views]
                if missing:
                    raise ValueError(f"unknown score views: {missing}")
                views = {name: views[name] for name in requested}
            scores, by_seed = _fit_auc_views(
                views,
                labels[fold_indices],
                len(train_indices),
                classifier,
                model_seeds,
            )
            folds.append(
                {
                    "fold": int(fold),
                    "n_train": int(train_indices.size),
                    "n_valid": int(valid_indices.size),
                    "n_train_positive": int(labels[train_indices].sum()),
                    "n_valid_positive": int(labels[valid_indices].sum()),
                    "scores": scores,
                    "scores_by_model_seed": by_seed,
                }
            )
            print(
                f"[{schema}] fold {fold}: T+A={scores['t_a']['valid_auc']:.6f}; "
                f"F_raw={scores['f_raw']['valid_auc']:.6f}; "
                f"S+F={scores['mixed_s_f_raw']['valid_auc']:.6f}",
                flush=True,
            )
        aggregate = {
            name: _aggregate_folds(folds, name) for name in folds[0]["scores"]
        }
        raw_controls = [
            f"f_raw_shuffled_{repeat}" for repeat in range(shuffle_repeats)
        ]
        factorized_controls = [
            f"f_raw_factorized_shuffled_{repeat}"
            for repeat in range(shuffle_repeats)
        ]
        centered_controls = [
            f"f_centered_shuffled_{repeat}" for repeat in range(shuffle_repeats)
        ]
        gates = {
            "factorized_raw_vs_marginals": _delta(
                folds, "f_raw_factorized", "t_a"
            ),
            "factorized_raw_vs_shuffle": _delta_against_controls(
                folds, "f_raw_factorized", factorized_controls
            ),
            "raw_vs_marginals": _delta(folds, "f_raw", "t_a"),
            "raw_vs_shuffle": _delta_against_controls(
                folds, "f_raw", raw_controls
            ),
            "centered_vs_marginals": _delta(folds, "f_centered", "t_a"),
            "centered_vs_shuffle": _delta_against_controls(
                folds, "f_centered", centered_controls
            ),
            "typed_edge_increment": _delta(
                folds, "f_raw", "f_raw_factorized"
            ),
            "mixed_factorized_increment": _delta(
                folds, "mixed_s_f_raw_factorized", "mixed_s"
            ),
            "mixed_increment": _delta(
                folds, "mixed_s_f_raw", "mixed_s"
            ),
            "mixed_centered_increment": _delta(
                folds, "mixed_s_f_centered", "mixed_s"
            ),
        }
        minimum_delta = float(screen["minimum_delta"])
        minimum_wins = int(screen["minimum_fold_wins"])
        for gate in gates.values():
            gate["minimum_mean_delta"] = minimum_delta
            gate["minimum_fold_wins"] = minimum_wins
            gate["passed"] = bool(
                gate["mean_delta"] >= minimum_delta
                and gate["fold_wins"] >= minimum_wins
            )
        schema_results[schema] = {
            "object_audit": object_audit,
            "aggregate": aggregate,
            "folds": folds,
            "gates": gates,
        }
        del arrays

    cross = None
    if "rooted_wl" in schema_results and "coarse" in schema_results:
        cross = _cross_schema_delta(
            schema_results, "rooted_wl", "coarse", "f_raw_factorized"
        )
        cross["minimum_mean_delta"] = float(screen["minimum_delta"])
        cross["minimum_fold_wins"] = int(screen["minimum_fold_wins"])
        cross["passed"] = bool(
            cross["mean_delta"] >= float(screen["minimum_delta"])
            and cross["fold_wins"] >= int(screen["minimum_fold_wins"])
        )
    primary_schema = "rooted_wl" if "rooted_wl" in schema_results else next(iter(schema_results))
    wl_gates = schema_results[primary_schema]["gates"]
    factorized_pass = (
        wl_gates["factorized_raw_vs_marginals"]["passed"]
        and wl_gates["factorized_raw_vs_shuffle"]["passed"]
        and wl_gates["mixed_factorized_increment"]["passed"]
    )
    centered_pass = (
        wl_gates["centered_vs_marginals"]["passed"]
        and wl_gates["centered_vs_shuffle"]["passed"]
        and wl_gates["mixed_centered_increment"]["passed"]
    )
    structure_gate = True if cross is None else bool(cross["passed"])
    if factorized_pass and centered_pass and structure_gate:
        decision = "ROOTED_WL_CLEAN_FUSION_PASS"
    elif (factorized_pass or centered_pass) and structure_gate:
        decision = "ROOTED_WL_ONE_CLEAN_FUSION_VIEW_PASS"
    elif (
        wl_gates["factorized_raw_vs_shuffle"]["passed"]
        or wl_gates["centered_vs_shuffle"]["passed"]
    ) and structure_gate:
        decision = "RICHER_STRUCTURE_DEPENDENCE_PASS_NO_STABLE_TASK_INCREMENT"
    elif (
        wl_gates["factorized_raw_vs_shuffle"]["passed"]
        or wl_gates["centered_vs_shuffle"]["passed"]
    ):
        decision = "CLEAN_DEPENDENCE_DETECTED_RICHER_STRUCTURE_NOT_BETTER"
    elif "coarse" in schema_results and (
        schema_results["coarse"]["gates"]["factorized_raw_vs_shuffle"]["passed"]
        or schema_results["coarse"]["gates"]["centered_vs_shuffle"]["passed"]
    ):
        decision = "COARSE_DEPENDENCE_ONLY"
    else:
        decision = "CLEAN_LOCAL_FUSION_NO_GO"

    result = {
        "protocol_id": config["protocol_id"],
        "config": config,
        "data": {
            "dataset": data_config["dataset"],
            "selected_graphs": int(selected_union.size),
            "split": (
                "official-train-only scaffold folds"
                if split_mode == "scaffold_folds"
                else "official train to official validation frozen evaluation"
            ),
            "official_test_encoded_or_evaluated": False,
        },
        "audit_boundary": {
            "frozen_s_sha256": _sha256(frozen_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "config_sha256": _sha256(config_path),
            "roles_use_attributes": False,
            "strict_atom_excludes": ["degree", "is_in_ring"],
            "shuffle": "independent within-patch row permutation preserving patch marginals",
        },
        "representation": {
            "dimensions": representation_dimensions(representation),
            "schemas": list(representation["schemas"]),
            "radius": int(representation["radius"]),
            "centers": "all graph nodes",
            "normalization": "entity-normalized patch statistics, then all-centre mean",
            "raw_fusion": "marginals + node-role x atom + edge-role x bond + hashed typed edge",
        },
        "exact_topology_audit": exact_audit,
        "n_folds": len(fold_ids),
        "schemas": schema_results,
        "cross_schema_gate": cross,
        "decision": decision,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    _write_json(result_json, result)
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
                "decision": result["decision"],
                "cross_schema_gate": result["cross_schema_gate"],
                "rooted_wl_gates": result["schemas"]["rooted_wl"]["gates"],
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
