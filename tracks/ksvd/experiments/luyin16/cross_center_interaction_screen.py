"""Quick screen for centre-level structure--attribute interactions.

The existing MolHIV route computes one invariant row per atom-centred
radius-2 patch and then pools those rows before XGBoost.  This runner keeps
the local object fixed and tests two interaction axes without GINE, attention,
or K-SVD:

* ``cross_cov``: covariance between topology and chemistry rows *across
  centres of the same graph*;
* ``binding``: within-patch role--attribute residuals, retaining their
  centre-population mean and standard deviation.

All high-dimensional interaction blocks are projected with train-fold-only
PCA.  The controls independently shuffle attribute rows across centres or
within each patch.  Official validation/test are never loaded by this
protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    _aggregate_folds,
    _delta,
    _delta_against_controls,
    _fit_auc_views,
    _fold_indices,
    _frozen_s_rows,
    _resolve,
    _select_rows,
    _sha256,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    BOND_DIM,
    REPO_ROOT,
    _entity_statistics,
    compact_bond_semantics,
    patch_feature_rows,
    patch_roles,
    strict_atom_semantics,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/cross_center_interaction_screen.yaml"

LOCAL_BLOCKS = ("node_role", "edge_role", "node_attribute", "edge_attribute")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _distribution_mean_std(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [n_centres,width], got {values.shape}")
    if values.shape[0] == 0:
        return np.zeros(2 * values.shape[1], dtype=np.float32)
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0)]
    ).astype(np.float32, copy=False)


def _cross_covar(topology: np.ndarray, attributes: np.ndarray) -> np.ndarray:
    """Return a flattened centre-population covariance matrix."""
    topo = np.asarray(topology, dtype=np.float32)
    attr = np.asarray(attributes, dtype=np.float32)
    if topo.ndim != 2 or attr.ndim != 2 or topo.shape[0] != attr.shape[0]:
        raise ValueError(f"unaligned centre rows: {topo.shape} vs {attr.shape}")
    n = topo.shape[0]
    if n <= 1:
        return np.zeros(topo.shape[1] * attr.shape[1], dtype=np.float32)
    topo_centered = topo - topo.mean(axis=0, keepdims=True)
    attr_centered = attr - attr.mean(axis=0, keepdims=True)
    return ((topo_centered.T @ attr_centered) / float(n)).reshape(-1).astype(
        np.float32, copy=False
    )


def _canonical_row_order(topology: np.ndarray, attributes: np.ndarray) -> np.ndarray:
    """Return a node-label-independent order for centre/entity rows.

    Equal rows are interchangeable, so their residual ordering cannot affect
    any statistic.  Sorting by the complete row values avoids using the
    internal graph node id as a tie breaker.
    """
    topo = np.asarray(topology, dtype=np.float32)
    attr = np.asarray(attributes, dtype=np.float32)
    if topo.shape[0] != attr.shape[0]:
        raise ValueError("topology and attribute rows are not aligned")
    return np.asarray(
        sorted(
            range(topo.shape[0]),
            key=lambda i: (tuple(float(x) for x in topo[i]), tuple(float(x) for x in attr[i])),
        ),
        dtype=np.int64,
    )


def _stable_patch_seed(
    base_seed: int,
    roles: Any,
    node_attrs: np.ndarray,
    edge_attrs: np.ndarray,
) -> int:
    """Derive a relabel-invariant RNG seed from a patch's semantic content."""
    payload = repr(
        (
            int(base_seed),
            int(roles.patch_key),
            tuple(
                sorted(
                    (
                        int(role),
                        tuple(float(value) for value in row),
                    )
                    for role, row in zip(roles.node_ids, node_attrs, strict=True)
                )
            ),
            tuple(
                sorted(
                    (
                        int(role),
                        tuple(float(value) for value in row),
                    )
                    for role, row in zip(roles.edge_ids, edge_attrs, strict=True)
                )
            ),
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**63 - 1)


def _local_rows(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build true centre rows: topology, attributes, binding, context."""
    blocks = patch_feature_rows(
        graph, node_features, edge_features, "rooted_wl", representation
    )
    topology = np.concatenate(
        [blocks["node_role"], blocks["edge_role"]], axis=1
    ).astype(np.float32, copy=False)
    attributes = np.concatenate(
        [blocks["node_attribute"], blocks["edge_attribute"]], axis=1
    ).astype(np.float32, copy=False)
    binding = np.concatenate(
        [blocks["node_binding"], blocks["edge_binding"]], axis=1
    ).astype(np.float32, copy=False)
    context = np.asarray(blocks["context"], dtype=np.float32).reshape(-1)
    return topology, attributes, binding, context


def _shuffled_binding_rows(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Return per-centre binding rows after within-patch attribute shuffles.

    Structural roles are left untouched.  Each patch receives a seed derived
    from its invariant semantic content, so the null remains unchanged after
    graph node relabelling.
    """
    atom_semantics, _atom_groups = strict_atom_semantics(node_features)
    bond_semantics = {
        graph.edge_key(left, right): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    rows: list[np.ndarray] = []
    centers = list(graph.nodes)
    maximum = representation.get("max_centers_per_graph")
    if maximum is not None and len(centers) > int(maximum):
        centers = centers[: int(maximum)]
    for center in centers:
        roles = patch_roles(graph, int(center), "rooted_wl", representation)
        node_attrs = np.stack(
            [atom_semantics[int(node)] for node in roles.nodes], axis=0
        ).astype(np.float32, copy=False)
        if roles.edges:
            edge_attrs = np.stack(
                [
                    bond_semantics[graph.edge_key(left, right)]
                    for left, right in roles.edges
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            edge_attrs = np.zeros((0, BOND_DIM), dtype=np.float32)
        # Canonicalize each entity list before drawing a permutation.  This
        # makes the null itself permutation invariant rather than merely the
        # unshuffled patch object.
        rng = np.random.default_rng(
            _stable_patch_seed(seed, roles, node_attrs, edge_attrs)
        )
        node_order = np.asarray(
            sorted(
                range(len(roles.node_ids)),
                key=lambda i: (
                    int(roles.node_ids[i]),
                    tuple(float(value) for value in node_attrs[i]),
                ),
            ),
            dtype=np.int64,
        )
        node_roles = np.asarray(roles.node_ids, dtype=np.int64)[node_order]
        node_values = node_attrs[node_order]
        if node_values.shape[0] > 1:
            node_values = node_values[rng.permutation(node_values.shape[0])]
        node_role, node_attribute, node_joint = _entity_statistics(
            node_roles, node_values, int(representation["node_role_bins"])
        )
        if roles.edge_ids.size:
            edge_order = np.asarray(
                sorted(
                    range(len(roles.edge_ids)),
                    key=lambda i: (
                        int(roles.edge_ids[i]),
                        tuple(float(value) for value in edge_attrs[i]),
                    ),
                ),
                dtype=np.int64,
            )
            edge_roles = np.asarray(roles.edge_ids, dtype=np.int64)[edge_order]
            edge_values = edge_attrs[edge_order]
            if edge_values.shape[0] > 1:
                edge_values = edge_values[rng.permutation(edge_values.shape[0])]
            edge_role, edge_attribute, edge_joint = _entity_statistics(
                edge_roles, edge_values, int(representation["edge_role_bins"])
            )
        else:
            edge_joint = np.zeros(
                (int(representation["edge_role_bins"]), BOND_DIM), dtype=np.float32
            )
            edge_role = np.zeros(int(representation["edge_role_bins"]), dtype=np.float32)
            edge_attribute = np.zeros(BOND_DIM, dtype=np.float32)
        node_binding = node_joint - node_role[:, None] * node_attribute[None, :]
        edge_binding = edge_joint - edge_role[:, None] * edge_attribute[None, :]
        rows.append(
            np.concatenate([node_binding.reshape(-1), edge_binding.reshape(-1)])
            .astype(np.float32, copy=False)
        )
    width = (
        int(representation["node_role_bins"]) * 40
        + int(representation["edge_role_bins"]) * BOND_DIM
    )
    return (
        np.stack(rows, axis=0).astype(np.float32, copy=False)
        if rows
        else np.zeros((0, width), dtype=np.float32)
    )


def _graph_interaction_features(
    bundle,
    index: int,
    representation: Mapping[str, Any],
    shuffle_repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("interaction features require node and edge attributes")
    graph = bundle.graphs[int(index)]
    node_features = bundle.node_feats[int(index)]
    edge_features = bundle.edge_feats[int(index)]
    topology, attributes, binding, context = _local_rows(
        graph, node_features, edge_features, representation
    )
    marginal = _distribution_mean_std(np.concatenate([topology, attributes], axis=1))
    cross = _cross_covar(topology, attributes)
    cross_shuffled: list[np.ndarray] = []
    for repeat in range(int(shuffle_repeats)):
        rng = np.random.default_rng(int(seed) + 1000003 * (repeat + 1))
        canonical = _canonical_row_order(topology, attributes)
        topo_ordered = topology[canonical]
        attr_ordered = attributes[canonical]
        order = rng.permutation(attr_ordered.shape[0])
        cross_shuffled.append(_cross_covar(topo_ordered, attr_ordered[order]))
    binding_distribution = _distribution_mean_std(binding)
    binding_shuffled = []
    for repeat in range(int(shuffle_repeats)):
        shuffled = _shuffled_binding_rows(
            graph,
            node_features,
            edge_features,
            representation,
            int(seed) + 2000003 * (repeat + 1),
        )
        binding_shuffled.append(_distribution_mean_std(shuffled))
    return {
        "marginal": marginal,
        "cross_cov": cross,
        "binding": binding_distribution,
        "context": context,
        **{
            f"cross_cov_shuffled_{repeat}": value
            for repeat, value in enumerate(cross_shuffled)
        },
        **{
            f"binding_shuffled_{repeat}": value
            for repeat, value in enumerate(binding_shuffled)
        },
    }


def _build_cache(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    shuffle_repeats: int,
    seed: int,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("cannot build an empty interaction cache")
    signature_payload = {
        "schema": "rooted_wl_cross_center_interaction_v1",
        "indices": selected.tolist(),
        "representation": dict(representation),
        "shuffle_repeats": int(shuffle_repeats),
        "seed": int(seed),
        "encoder": _sha256(Path(__file__).resolve()),
    }
    signature = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if str(np.asarray(archive["signature"]).reshape(-1)[0]) != signature:
                raise ValueError(f"interaction cache signature mismatch: {cache_path}")
            return {name: np.asarray(archive[name]) for name in archive.files if name != "signature"}
    first = _graph_interaction_features(
        bundle, int(selected[0]), representation, shuffle_repeats, seed + int(selected[0])
    )
    arrays: dict[str, np.ndarray] = {
        "dataset_indices": selected,
        "labels": np.asarray(bundle.y[selected], dtype=np.int64),
    }
    for name, value in first.items():
        arrays[name] = np.zeros((selected.size, value.shape[0]), dtype=np.float32)
    for position, raw_index in enumerate(selected):
        values = first if position == 0 else _graph_interaction_features(
            bundle,
            int(raw_index),
            representation,
            shuffle_repeats,
            seed + int(raw_index),
        )
        for name, value in values.items():
            arrays[name][position] = value
        if position and position % 100 == 0:
            print(f"interaction feature graphs: {position}/{selected.size}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, signature=np.asarray([signature]), **arrays)
    temporary.replace(cache_path)
    return arrays


def _audit_invariance(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    shuffle_repeats: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("invariance audit needs at least one graph")
    maxima: dict[str, float] = {}
    rng = np.random.default_rng(int(seed))
    for raw_index in selected:
        index = int(raw_index)
        graph = bundle.graphs[index]
        nodes = bundle.node_feats[index]
        edges = bundle.edge_feats[index]
        base = _graph_interaction_features(
            bundle, index, representation, shuffle_repeats, seed + index
        )
        permutation = rng.permutation(graph.n)
        changed_graph, changed_nodes, changed_edges = relabel_graph_features(
            graph, nodes, edges, permutation
        )

        class _OneGraphBundle:
            graphs = [changed_graph]
            node_feats = [changed_nodes]
            edge_feats = [changed_edges]
            y = np.asarray([0], dtype=np.int64)

        changed = _graph_interaction_features(
            _OneGraphBundle(), 0, representation, shuffle_repeats, seed + index
        )
        for name in base:
            drift = float(np.max(np.abs(base[name] - changed[name])))
            maxima[name] = max(maxima.get(name, 0.0), drift)
    passed = all(value <= float(tolerance) for value in maxima.values())
    return {"pass": bool(passed), "max_abs_drift": maxima, "tolerance": float(tolerance)}


def _fit_projection_model(
    train: np.ndarray,
    rank: int,
) -> tuple[PCA | None, np.ndarray, dict[str, Any]]:
    values = np.asarray(train, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"projection input must be 2-D, got {values.shape}")
    n_components = min(int(rank), values.shape[0], values.shape[1])
    if n_components <= 0:
        return (
            None,
            np.zeros((values.shape[0], 0), dtype=np.float32),
            {"n_components": 0, "explained_variance_ratio_sum": 0.0},
        )
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    train_out = model.fit_transform(values).astype(np.float32, copy=False)
    return (
        model,
        train_out,
        {
            "n_components": int(n_components),
            "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
        },
    )


def _transform_projection(model: PCA | None, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if model is None:
        return np.zeros((array.shape[0], 0), dtype=np.float32)
    return model.transform(array).astype(np.float32, copy=False)


def _load_excluded_indices(
    path: str | Path | Sequence[str | Path] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a prior selection cache used to make a disjoint confirmation set.

    The cache is intentionally treated as a selection manifest only: no
    features or labels from the prior run are reused.  Keeping this boundary
    explicit prevents an accidental train/confirmation cache mix-up.
    """
    if path is None:
        return np.zeros(0, dtype=np.int64), {
            "enabled": False,
            "path": None,
            "sources": [],
            "n_indices": 0,
            "sha256": None,
        }
    raw_sources = [path] if isinstance(path, (str, Path)) else list(path)
    if not raw_sources:
        raise ValueError("exclusion manifest list cannot be empty")
    source_rows: list[dict[str, Any]] = []
    index_parts: list[np.ndarray] = []
    for raw_source in raw_sources:
        source = Path(raw_source)
        if not source.exists():
            raise FileNotFoundError(f"exclusion manifest does not exist: {source}")
        with np.load(source, allow_pickle=False) as archive:
            if "dataset_indices" not in archive.files:
                raise ValueError(f"exclusion manifest lacks dataset_indices: {source}")
            indices = np.asarray(archive["dataset_indices"], dtype=np.int64).reshape(-1)
        source_unique = np.unique(indices)
        index_parts.append(source_unique)
        source_rows.append(
            {
                "path": str(source),
                "n_indices": int(source_unique.size),
                "sha256": _sha256(source),
            }
        )
    unique = np.unique(np.concatenate(index_parts))
    return unique, {
        "enabled": True,
        "path": source_rows[0]["path"] if len(source_rows) == 1 else None,
        "sources": source_rows,
        "n_indices": int(unique.size),
        "sha256": source_rows[0]["sha256"] if len(source_rows) == 1 else None,
    }


def _fold_indices_excluding(
    archive: Mapping[str, np.ndarray],
    fold: int,
    labels: np.ndarray,
    screen: Mapping[str, Any],
    excluded_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fold after removing a prior discovery set.

    Selection uses the same class-balanced deterministic sampler as the
    discovery protocol, but the exclusion is applied before sampling.  This
    makes the confirmation split disjoint at the dataset-index level while
    preserving the requested train/valid budgets.
    """
    original = np.asarray(archive["original_indices"], dtype=np.int64)
    train = original[np.asarray(archive[f"fold_{fold}_train_indices"], dtype=np.int64)]
    valid = original[np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)]
    excluded = np.asarray(excluded_indices, dtype=np.int64).reshape(-1)
    if excluded.size:
        train = train[~np.isin(train, excluded)]
        valid = valid[~np.isin(valid, excluded)]
    seed = int(screen["seed"]) + 1009 * int(fold)
    return (
        _select_rows(labels, train, screen.get("max_train_graphs_per_fold"), seed),
        _select_rows(labels, valid, screen.get("max_valid_graphs_per_fold"), seed + 1),
    )


def _fit_projection(
    train: np.ndarray,
    valid: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convenience wrapper retained for unit-level callers."""
    model, train_out, metadata = _fit_projection_model(train, rank)
    heldout = np.asarray(valid, dtype=np.float32)
    if heldout.ndim != 2 or heldout.shape[1] != np.asarray(train).shape[1]:
        raise ValueError(f"projection shapes mismatch: {np.asarray(train).shape} vs {heldout.shape}")
    return (
        train_out,
        _transform_projection(model, heldout),
        metadata,
    )


def _fit_standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"standardizer input must be 2-D, got {values.shape}")
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), scale


def _apply_standardizer(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != mean.shape[1]:
        raise ValueError(
            f"standardizer shapes mismatch: {array.shape}, {mean.shape}, {scale.shape}"
        )
    return ((array - mean) / scale).astype(np.float32, copy=False)


def _row_outer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"unaligned outer-product rows: {lhs.shape} vs {rhs.shape}")
    return np.einsum("ni,nj->nij", lhs, rhs, optimize=True).reshape(lhs.shape[0], -1).astype(
        np.float32, copy=False
    )


def _conditional_interaction_blocks(
    cross: np.ndarray,
    binding: np.ndarray,
    n_train: int,
    cross_shuffled: Sequence[np.ndarray],
    binding_shuffled: Sequence[np.ndarray],
    rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build two low-capacity, explicitly conditional fusion blocks.

    ``norm_gate`` multiplies standardized binding coordinates by the
    standardized magnitude of cross-centre heterogeneity. ``bilinear_gate``
    projects the standardized 8x8 outer product back to the fixed PCA rank.
    All scalers and the bilinear PCA are fit on true fold-train rows only;
    null blocks reuse those exact coordinates.
    """
    cross_values = np.asarray(cross, dtype=np.float32)
    binding_values = np.asarray(binding, dtype=np.float32)
    if cross_values.ndim != 2 or binding_values.ndim != 2:
        raise ValueError("conditional interaction inputs must be 2-D")
    if cross_values.shape[0] != binding_values.shape[0]:
        raise ValueError("conditional interaction rows are not aligned")
    if not 0 < int(n_train) < cross_values.shape[0]:
        raise ValueError(f"invalid conditional interaction train size: {n_train}")
    if len(cross_shuffled) != len(binding_shuffled):
        raise ValueError("conditional interaction null repeat counts differ")

    cross_mean, cross_scale = _fit_standardizer(cross_values[:n_train])
    binding_mean, binding_scale = _fit_standardizer(binding_values[:n_train])
    cross_z = _apply_standardizer(cross_values, cross_mean, cross_scale)
    binding_z = _apply_standardizer(binding_values, binding_mean, binding_scale)
    cross_null_z = [
        _apply_standardizer(values, cross_mean, cross_scale)
        for values in cross_shuffled
    ]
    binding_null_z = [
        _apply_standardizer(values, binding_mean, binding_scale)
        for values in binding_shuffled
    ]

    normalizer = float(np.sqrt(max(1, cross_z.shape[1])))
    true_norm = np.linalg.norm(cross_z, axis=1, keepdims=True) / normalizer
    norm_mean, norm_scale = _fit_standardizer(true_norm[:n_train])

    def norm_gate(cross_rows: np.ndarray, binding_rows: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(cross_rows, axis=1, keepdims=True) / normalizer
        standardized = _apply_standardizer(magnitude, norm_mean, norm_scale)
        return (standardized * binding_rows).astype(np.float32, copy=False)

    true_outer = _row_outer(cross_z, binding_z)
    bilinear_model, bilinear_train, bilinear_meta = _fit_projection_model(
        true_outer[:n_train], rank
    )

    def bilinear_gate(cross_rows: np.ndarray, binding_rows: np.ndarray) -> np.ndarray:
        outer = _row_outer(cross_rows, binding_rows)
        return np.concatenate(
            [
                _transform_projection(bilinear_model, outer[:n_train]),
                _transform_projection(bilinear_model, outer[n_train:]),
            ],
            axis=0,
        )

    blocks: dict[str, np.ndarray] = {
        "norm_gate": norm_gate(cross_z, binding_z),
        "bilinear_gate": np.concatenate(
            [
                bilinear_train,
                _transform_projection(bilinear_model, true_outer[n_train:]),
            ],
            axis=0,
        ),
    }
    for repeat, (cross_null, binding_null) in enumerate(
        zip(cross_null_z, binding_null_z, strict=True)
    ):
        blocks[f"norm_gate_cross_true_binding_shuffled_{repeat}"] = norm_gate(
            cross_z, binding_null
        )
        blocks[f"norm_gate_cross_shuffled_binding_true_{repeat}"] = norm_gate(
            cross_null, binding_z
        )
        blocks[f"norm_gate_both_shuffled_{repeat}"] = norm_gate(
            cross_null, binding_null
        )
        blocks[f"bilinear_gate_cross_true_binding_shuffled_{repeat}"] = bilinear_gate(
            cross_z, binding_null
        )
        blocks[f"bilinear_gate_cross_shuffled_binding_true_{repeat}"] = bilinear_gate(
            cross_null, binding_z
        )
        blocks[f"bilinear_gate_both_shuffled_{repeat}"] = bilinear_gate(
            cross_null, binding_null
        )
    return blocks, {
        "cross_width": int(cross_values.shape[1]),
        "binding_width": int(binding_values.shape[1]),
        "norm_gate_width": int(binding_values.shape[1]),
        "bilinear_raw_width": int(true_outer.shape[1]),
        "bilinear_projection": bilinear_meta,
    }


def _fit_late_fusion(
    marginal: np.ndarray,
    binding: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    classifier: Mapping[str, Any],
    seeds: Sequence[int],
    alpha: float,
) -> dict[str, Any]:
    y_train = np.asarray(labels[:n_train], dtype=np.int64)
    y_valid = np.asarray(labels[n_train:], dtype=np.int64)
    fixed = {
        key: classifier[key]
        for key in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "gamma",
            "n_jobs",
        )
        if key in classifier
    }
    weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    records = []
    for seed in seeds:
        predictions = []
        for matrix in (marginal, binding):
            model = XGBClassifier(
                **fixed,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=weight,
                random_state=int(seed),
                tree_method="hist",
            )
            model.fit(matrix[:n_train], y_train)
            predictions.append(model.predict_proba(matrix[n_train:])[:, 1])
        blended = float(alpha) * predictions[0] + (1.0 - float(alpha)) * predictions[1]
        records.append({"seed": int(seed), "valid_auc": float(roc_auc_score(y_valid, blended))})
    values = [row["valid_auc"] for row in records]
    return {
        "rows": records,
        "mean_auc": float(np.mean(values)),
        "std_auc": float(np.std(values)),
        "alpha_marginal": float(alpha),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Centre-level structure--attribute interaction screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Official validation/test were not encoded or evaluated.",
        "",
        f"Discovery exclusion enabled: **{result.get('exclusion', {}).get('enabled', False)}** "
        f"({result.get('exclusion', {}).get('n_indices', 0)} dataset indices).",
        f"Selected/discovery overlap: **{result['data'].get('excluded_discovery_overlap', 0)}**.",
        "",
        "## Invariance audit",
        "",
        f"- pass: **{result['invariance_audit']['pass']}**",
        f"- maximum drift: {max(result['invariance_audit']['max_abs_drift'].values()):.3e}",
        "",
    ]
    lines.extend(["## Fold scores", "", "| fold | view | valid ROC-AUC |", "|---:|---|---:|"])
    for fold in result["folds"]:
        for name, score in fold["scores"].items():
            lines.append(f"| {fold['fold']} | `{name}` | {score['valid_auc']:.6f} |")
    lines.extend(["", "## Aggregate", "", "| view | mean AUC | fold std |", "|---|---:|---:|"])
    for name, row in result["aggregate"].items():
        lines.append(f"| `{name}` | {row['mean_auc']:.6f} | {row['std_auc']:.6f} |")
    lines.extend(["", "## Gates", ""])
    for name, gate in result["gates"].items():
        lines.append(
            f"- `{name}`: {gate['mean_delta']:+.6f}, wins {gate['fold_wins']}/{result['n_folds']}"
        )
    lines.extend(
        [
            "",
            "The interaction blocks are projected with fold-train-only PCA; shuffled controls preserve graph-local row marginals while breaking the tested correspondence. `both_vs_double_shuffle` pairs the centre and patch shuffles by repeat.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    classifier = config["classifier"]
    result_json = _resolve(config["output_json"])
    result_markdown = _resolve(config["output_markdown"])
    cache_path = _resolve(config["cache_npz"])
    frozen_path = _resolve(data_config["frozen_features"])
    folds_path = _resolve(data_config["scaffold_folds"])
    exclusion_value = screen.get("exclude_indices_npz")
    if isinstance(exclusion_value, (list, tuple)):
        exclusion_paths: Path | list[Path] | None = [
            _resolve(value) for value in exclusion_value
        ]
    else:
        exclusion_paths = _resolve(exclusion_value) if exclusion_value else None
    excluded_indices, exclusion = _load_excluded_indices(exclusion_paths)
    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    start = time.perf_counter()

    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in fold_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    if excluded_indices.size:
        selections = [
            _fold_indices_excluding(
                fold_archive, fold, labels, screen, excluded_indices
            )
            for fold in fold_ids
        ]
    else:
        selections = [
            _fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids
        ]
    selected_union = np.unique(
        np.concatenate([np.concatenate(pair) for pair in selections])
    ).astype(np.int64)
    overlap = np.intersect1d(selected_union, excluded_indices)
    if overlap.size:
        raise RuntimeError(
            "confirmation selection overlaps excluded discovery indices: "
            f"{overlap.size} graphs"
        )
    print(
        f"selected graphs={selected_union.size}; folds={fold_ids}; interaction=centre-level",
        flush=True,
    )
    audit_candidates = selected_union[: int(config["audit"]["n_graphs"])]
    invariance = _audit_invariance(
        bundle,
        audit_candidates,
        representation,
        int(screen["shuffle_repeats"]),
        int(config["audit"]["seed"]),
        float(config["audit"]["tolerance"]),
    )
    if not invariance["pass"]:
        raise RuntimeError(f"interaction invariance audit failed: {invariance}")
    arrays = _build_cache(
        bundle,
        selected_union,
        representation,
        int(screen["shuffle_repeats"]),
        int(screen["feature_seed"]),
        cache_path,
    )
    index_to_row = {
        int(index): position
        for position, index in enumerate(np.asarray(arrays["dataset_indices"], dtype=np.int64))
    }
    model_seeds = [int(value) for value in classifier.get("model_seeds", [0])]
    pca_rank = int(screen["pca_rank"])
    folds: list[dict[str, Any]] = []
    projection_meta: dict[str, Any] = {}
    for fold, (train_indices, valid_indices) in zip(fold_ids, selections, strict=True):
        fold_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
        row_ids = np.asarray([index_to_row[int(index)] for index in fold_indices], dtype=np.int64)
        n_train = int(train_indices.size)
        y_fold = labels[fold_indices]
        s_fold = _frozen_s_rows(frozen, fold_indices)
        marginal = np.asarray(arrays["marginal"][row_ids], dtype=np.float32)
        context = np.asarray(arrays["context"][row_ids], dtype=np.float32)
        base = np.concatenate([s_fold, marginal, context], axis=1)
        cross_model, cross_train, cross_meta = _fit_projection_model(
            arrays["cross_cov"][row_ids[:n_train]], pca_rank
        )
        cross_valid = _transform_projection(
            cross_model, arrays["cross_cov"][row_ids[n_train:]]
        )
        binding_model, binding_train, binding_meta = _fit_projection_model(
            arrays["binding"][row_ids[:n_train]], pca_rank
        )
        binding_valid = _transform_projection(
            binding_model, arrays["binding"][row_ids[n_train:]]
        )
        cross = np.concatenate([cross_train, cross_valid], axis=0)
        binding = np.concatenate([binding_train, binding_valid], axis=0)
        views = {
            "s": s_fold,
            "s_marginal": base,
            "s_cross_cov": np.concatenate([base, cross], axis=1),
            "s_binding": np.concatenate([base, binding], axis=1),
            "s_both": np.concatenate([base, cross, binding], axis=1),
        }
        cross_shuffled_projected: list[np.ndarray] = []
        binding_shuffled_projected: list[np.ndarray] = []
        for repeat in range(int(screen["shuffle_repeats"])):
            cross_shuf_train = _transform_projection(
                cross_model,
                arrays[f"cross_cov_shuffled_{repeat}"][row_ids[:n_train]],
            )
            cross_shuf_valid = _transform_projection(
                cross_model,
                arrays[f"cross_cov_shuffled_{repeat}"][row_ids[n_train:]],
            )
            binding_shuf_train = _transform_projection(
                binding_model,
                arrays[f"binding_shuffled_{repeat}"][row_ids[:n_train]],
            )
            binding_shuf_valid = _transform_projection(
                binding_model,
                arrays[f"binding_shuffled_{repeat}"][row_ids[n_train:]],
            )
            cross_shuf = np.concatenate(
                [cross_shuf_train, cross_shuf_valid], axis=0
            )
            binding_shuf = np.concatenate(
                [binding_shuf_train, binding_shuf_valid], axis=0
            )
            cross_shuffled_projected.append(cross_shuf)
            binding_shuffled_projected.append(binding_shuf)
            views[f"s_cross_cov_shuffled_{repeat}"] = np.concatenate(
                [base, cross_shuf], axis=1
            )
            views[f"s_binding_shuffled_{repeat}"] = np.concatenate(
                [base, binding_shuf], axis=1
            )
            # Matched double null: use the same repeat's centre-shuffle and
            # patch-shuffle coordinates together.  This preserves the
            # dimensionality and PCA fitting path of ``s_both`` while
            # breaking both tested correspondences.
            views[f"s_both_shuffled_{repeat}"] = np.concatenate(
                [
                    base,
                    cross_shuf,
                    binding_shuf,
                ],
                axis=1,
            )
            # Mixed controls isolate the conditional contribution of each
            # interaction axis.  They keep one true block and replace the
            # other with the matched repeat's null coordinates.
            views[f"s_cross_true_binding_shuffled_{repeat}"] = np.concatenate(
                [
                    base,
                    cross,
                    binding_shuf,
                ],
                axis=1,
            )
            views[f"s_cross_shuffled_binding_true_{repeat}"] = np.concatenate(
                [
                    base,
                    cross_shuf,
                    binding,
                ],
                axis=1,
            )
        conditional_meta = None
        if bool(screen.get("conditional_gate", False)):
            conditional_blocks, conditional_meta = _conditional_interaction_blocks(
                cross,
                binding,
                n_train,
                cross_shuffled_projected,
                binding_shuffled_projected,
                pca_rank,
            )
            for name, block in conditional_blocks.items():
                views[f"s_{name}"] = np.concatenate(
                    [views["s_both"], block], axis=1
                )
        scores, by_seed = _fit_auc_views(
            views, y_fold, n_train, classifier, model_seeds
        )
        late = _fit_late_fusion(
            views["s_marginal"], views["s_binding"], y_fold, n_train, classifier, model_seeds, 0.5
        )
        scores["late_marginal_binding"] = {
            "valid_auc": late["mean_auc"],
            "valid_auc_std": late["std_auc"],
            "dimension": int(views["s_marginal"].shape[1]),
        }
        folds.append(
            {
                "fold": int(fold),
                "n_train": n_train,
                "n_valid": int(valid_indices.size),
                "n_train_positive": int(labels[train_indices].sum()),
                "n_valid_positive": int(labels[valid_indices].sum()),
                # Keep the exact dataset-index manifest in the result so a
                # confirmation run can be audited without reconstructing the
                # sampler state.
                "train_dataset_indices": train_indices.tolist(),
                "valid_dataset_indices": valid_indices.tolist(),
                "scores": scores,
                "scores_by_model_seed": by_seed,
                "late_fusion": late,
            }
        )
        projection_meta[str(fold)] = {
            "cross_cov": cross_meta,
            "binding": binding_meta,
            "conditional_gate": conditional_meta,
        }
        print(
            f"fold {fold}: marginal={scores['s_marginal']['valid_auc']:.6f}; "
            f"cross={scores['s_cross_cov']['valid_auc']:.6f}; "
            f"binding={scores['s_binding']['valid_auc']:.6f}; "
            f"late={scores['late_marginal_binding']['valid_auc']:.6f}",
            flush=True,
        )
    view_names = list(folds[0]["scores"].keys())
    aggregate = {name: _aggregate_folds(folds, name) for name in view_names}
    cross_controls = [
        f"s_cross_cov_shuffled_{repeat}" for repeat in range(int(screen["shuffle_repeats"]))
    ]
    binding_controls = [
        f"s_binding_shuffled_{repeat}" for repeat in range(int(screen["shuffle_repeats"]))
    ]
    both_controls = [
        f"s_both_shuffled_{repeat}" for repeat in range(int(screen["shuffle_repeats"]))
    ]
    cross_true_binding_null_controls = [
        f"s_cross_true_binding_shuffled_{repeat}"
        for repeat in range(int(screen["shuffle_repeats"]))
    ]
    cross_null_binding_true_controls = [
        f"s_cross_shuffled_binding_true_{repeat}"
        for repeat in range(int(screen["shuffle_repeats"]))
    ]
    gates = {
        "cross_cov_vs_marginal": _delta(folds, "s_cross_cov", "s_marginal"),
        "cross_cov_vs_center_shuffle": _delta_against_controls(
            folds, "s_cross_cov", cross_controls
        ),
        "binding_vs_marginal": _delta(folds, "s_binding", "s_marginal"),
        "binding_vs_patch_shuffle": _delta_against_controls(
            folds, "s_binding", binding_controls
        ),
        "both_vs_marginal": _delta(folds, "s_both", "s_marginal"),
        "both_vs_cross_cov": _delta(folds, "s_both", "s_cross_cov"),
        "both_vs_binding": _delta(folds, "s_both", "s_binding"),
        "both_vs_double_shuffle": _delta_against_controls(
            folds, "s_both", both_controls
        ),
        "both_vs_cross_true_binding_shuffle": _delta_against_controls(
            folds, "s_both", cross_true_binding_null_controls
        ),
        "both_vs_cross_shuffle_binding_true": _delta_against_controls(
            folds, "s_both", cross_null_binding_true_controls
        ),
        "late_vs_marginal": _delta(folds, "late_marginal_binding", "s_marginal"),
    }
    if bool(screen.get("conditional_gate", False)):
        for gate_name in ("norm_gate", "bilinear_gate"):
            candidate = f"s_{gate_name}"
            gates[f"{gate_name}_vs_both"] = _delta(
                folds, candidate, "s_both"
            )
            gates[f"{gate_name}_vs_double_shuffle"] = _delta_against_controls(
                folds,
                candidate,
                [
                    f"s_{gate_name}_both_shuffled_{repeat}"
                    for repeat in range(int(screen["shuffle_repeats"]))
                ],
            )
            gates[f"{gate_name}_vs_cross_true_binding_shuffle"] = (
                _delta_against_controls(
                    folds,
                    candidate,
                    [
                        f"s_{gate_name}_cross_true_binding_shuffled_{repeat}"
                        for repeat in range(int(screen["shuffle_repeats"]))
                    ],
                )
            )
            gates[f"{gate_name}_vs_cross_shuffle_binding_true"] = (
                _delta_against_controls(
                    folds,
                    candidate,
                    [
                        f"s_{gate_name}_cross_shuffled_binding_true_{repeat}"
                        for repeat in range(int(screen["shuffle_repeats"]))
                    ],
                )
            )
    minimum_delta = float(screen["minimum_delta"])
    minimum_wins = int(screen["minimum_fold_wins"])
    for gate in gates.values():
        gate["minimum_mean_delta"] = minimum_delta
        gate["minimum_fold_wins"] = minimum_wins
        gate["passed"] = bool(
            gate["mean_delta"] >= minimum_delta and gate["fold_wins"] >= minimum_wins
        )
    result = {
        "protocol_id": config["protocol_id"],
        "official_validation_or_test_evaluated": False,
        "data": {
            "dataset": data_config["dataset"],
            "selected_graphs": int(selected_union.size),
            "folds": fold_ids,
            "split": "official-train-only scaffold folds",
            "excluded_discovery_overlap": int(overlap.size),
        },
        "exclusion": exclusion,
        "representation": {
            "radius": int(representation["radius"]),
            "node_role_bins": int(representation["node_role_bins"]),
            "edge_role_bins": int(representation["edge_role_bins"]),
            "local_blocks": list(LOCAL_BLOCKS),
            "centres": "all graph nodes",
            "interaction_axes": [
                "cross-centre topology-attribute covariance",
                "within-patch role-attribute binding distribution",
            ],
            "control_axes": [
                "centre-shuffle",
                "patch-shuffle",
                "matched double-shuffle",
                "mixed conditional shuffles",
            ],
            "conditional_gate_enabled": bool(screen.get("conditional_gate", False)),
        },
        "invariance_audit": invariance,
        "cache": {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
        },
        "pca_rank": pca_rank,
        "projection_meta": projection_meta,
        "n_folds": len(folds),
        "folds": folds,
        "aggregate": aggregate,
        "gates": gates,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
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
                "protocol_id": result["protocol_id"],
                "invariance_pass": result["invariance_audit"]["pass"],
                "aggregate": result["aggregate"],
                "gates": result["gates"],
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
