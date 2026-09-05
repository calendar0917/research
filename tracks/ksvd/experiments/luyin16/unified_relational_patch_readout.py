"""Unified relational patch readout for ZINC and MolHIV.

The two datasets use different raw atom/bond schemas, but this runner exposes
the same representation boundary to the predictor:

    local patch descriptor
        -> hierarchical stable patch keys + train-only prototype code
        -> invariant unary, within-patch and relation-conditioned moments
        -> fixed-width exact/count sketches
        -> one small MLP head

There is no message passing, attention, score-level ensemble, or second
predictive model.  Dataset-specific code is restricted to loading the already
audited label-free patch records and splitting each descriptor into topology
and chemistry coordinates.  The resulting graph feature layout and head are
identical for both tasks.

The current experiment deliberately uses a train-only K-SVD *initial*
dictionary as a compact prototype coordinate.  It does not claim that K-SVD
updates improve the task: earlier experiments showed the opposite can happen.
Exact patch identity is retained separately by a stable signed count sketch,
so the prototype coordinate cannot replace the categorical object.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ksvd_research.core.ksvd import ksvd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16 import molhiv_ksvd_patch_path_pooling as molhiv_records
from tracks.ksvd.experiments.luyin16 import zinc_ksvd_patch_path_pooling as zinc_records
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zinc_proxy
from tracks.ksvd.experiments.luyin16.molhiv_ksvd_patch_path_pooling import _batch_omp


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/unified_relational_patch_zinc.yaml"


@dataclass
class RawGraph:
    """Compact label-free graph record shared by both dataset adapters."""

    descriptors: np.ndarray
    token_hash: np.ndarray
    parent_hash: np.ndarray
    pair_index: np.ndarray
    pair_relation: np.ndarray
    pair_bucket: np.ndarray
    context: np.ndarray
    y: float


@dataclass
class Projector:
    mean: np.ndarray
    scale: np.ndarray
    pca: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        flat = matrix.ndim == 1
        if flat:
            matrix = matrix[None, :]
        standardized = (matrix - self.mean) / self.scale
        output = self.pca.transform(standardized).astype(np.float32, copy=False)
        return output[0] if flat else output


@dataclass
class TransformBundle:
    full: Projector
    structure: Projector
    attribute: Projector
    relation: Projector
    context: Projector
    dictionary: np.ndarray
    dictionary_gram: np.ndarray
    dictionary_error: dict[str, Any]


@dataclass(frozen=True)
class FeatureLayout:
    structure_width: int
    attribute_width: int
    prototype_width: int
    relation_width: int
    distance_buckets: int
    unary_sketch_width: int
    parent_sketch_width: int
    pair_sketch_width: int
    context_width: int

    @property
    def width(self) -> int:
        w = int(self.structure_width)
        a = int(self.attribute_width)
        p = int(self.prototype_width)
        r = int(self.relation_width)
        b = int(self.distance_buckets)
        return (
            3 * w
            + 3 * a
            + 3 * p
            + w * a
            + b * w * a
            + b * (2 * r + 1)
            + int(self.unary_sketch_width)
            + int(self.parent_sketch_width)
            + int(self.pair_sketch_width)
            + int(self.context_width)
            + 4
        )


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _digest(value: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little"))


def _coarse_patch_key(descriptor: np.ndarray) -> np.uint64:
    """A stable lower-resolution parent key, independent of node numbering."""
    row = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    chunks = np.array_split(row, 16)
    summary = np.asarray(
        [
            float(row.mean()),
            float(row.std()),
            float(row.min(initial=0.0)),
            float(row.max(initial=0.0)),
            float(np.count_nonzero(np.abs(row) > 1.0e-6)),
            *[float(chunk.sum()) for chunk in chunks],
        ],
        dtype=np.float32,
    )
    # Quantisation makes this a genuine backoff key rather than a second copy
    # of a high-dimensional exact descriptor.
    quantized = np.round(summary, 2).astype(np.float16, copy=False)
    return _digest(quantized.tobytes())


def _record_to_raw(record: Any, *, exact_identity: bool) -> RawGraph:
    descriptors = np.stack(
        [np.asarray(patch.shell_descriptor, dtype=np.float32) for patch in record.patches],
        axis=0,
    ).astype(np.float32, copy=False)
    token_hash: list[np.uint64] = []
    parent_hash: list[np.uint64] = []
    for patch, descriptor in zip(record.patches, descriptors, strict=True):
        certificate = getattr(patch, "typed_certificate", b"")
        if exact_identity and certificate:
            token_hash.append(_digest(bytes(certificate)))
        else:
            token_hash.append(_digest(descriptor.tobytes()))
        parent_hash.append(_coarse_patch_key(descriptor))
    pair_relation = np.asarray(record.pair_relation, dtype=np.float32)
    pair_index = np.asarray(record.pair_index, dtype=np.int64)
    pair_bucket = np.asarray(record.pair_bucket, dtype=np.int64)
    return RawGraph(
        descriptors=descriptors,
        token_hash=np.asarray(token_hash, dtype=np.uint64),
        parent_hash=np.asarray(parent_hash, dtype=np.uint64),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        context=np.asarray(record.global_context, dtype=np.float32).reshape(-1),
        y=float(record.y),
    )


def _mix64(left: np.ndarray, right: np.ndarray, relation: np.ndarray) -> np.ndarray:
    """Vectorised deterministic 64-bit mixer for count-sketch coordinates."""
    a = np.asarray(left, dtype=np.uint64)
    b = np.asarray(right, dtype=np.uint64)
    c = np.asarray(relation, dtype=np.uint64)
    value = a ^ (b + np.uint64(0x9E3779B97F4A7C15)) ^ (
        c * np.uint64(0xD6E8FEB86659FD93)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


def _signed_sketch(keys: np.ndarray, width: int, salt: int) -> np.ndarray:
    values = np.asarray(keys, dtype=np.uint64).reshape(-1)
    if values.size == 0:
        return np.zeros(int(width), dtype=np.float32)
    mixed = _mix64(values, np.full(values.shape, np.uint64(salt)), np.zeros_like(values))
    indices = mixed % np.uint64(int(width))
    signs = np.where((mixed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
    output = np.zeros(int(width), dtype=np.float32)
    np.add.at(output, indices.astype(np.int64), signs)
    return output


def _relation_codes(
    relation: np.ndarray,
    buckets: np.ndarray,
    *,
    bond_width: int,
    distance_buckets: int,
) -> np.ndarray:
    """Compress relation details into a stable categorical pair condition."""
    matrix = np.asarray(relation, dtype=np.float32)
    bucket = np.asarray(buckets, dtype=np.int64).reshape(-1)
    if matrix.shape[0] != bucket.shape[0]:
        raise ValueError("relation and bucket row counts differ")
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    fixed_distance_buckets = int(distance_buckets)
    if fixed_distance_buckets < 1:
        raise ValueError("distance_buckets must be positive")
    if np.any(bucket < 0) or np.any(bucket >= fixed_distance_buckets):
        raise ValueError("pair distance bucket is outside the fixed relation schema")
    # The first distance_width coordinates are one-hot, followed by log
    # distance and overlap.  This layout is shared by the audited ZINC/HIV
    # relation builders.
    overlap_column = fixed_distance_buckets + 1
    if overlap_column >= matrix.shape[1]:
        raise ValueError("relation matrix is shorter than its fixed distance schema")
    overlap_bin = np.clip(np.rint(matrix[:, overlap_column] * 4.0), 0, 7).astype(np.int64)
    tail_width = int(bond_width)
    if tail_width < 1 or tail_width > matrix.shape[1]:
        raise ValueError("bond_width is incompatible with relation matrix")
    adjacent = np.argmax(matrix[:, -tail_width:], axis=1).astype(np.int64)
    return bucket * 64 + overlap_bin * 16 + adjacent


def _pair_sketch(
    graph: RawGraph,
    relation_codes: np.ndarray,
    width: int,
) -> np.ndarray:
    if graph.pair_index.shape[1] == 0:
        return np.zeros(int(width), dtype=np.float32)
    source = graph.pair_index[0]
    target = graph.pair_index[1]
    left = graph.token_hash[source]
    right = graph.token_hash[target]
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    mixed = _mix64(low, high, relation_codes.astype(np.uint64, copy=False))
    indices = mixed % np.uint64(int(width))
    signs = np.where((mixed >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
    output = np.zeros(int(width), dtype=np.float32)
    np.add.at(output, indices.astype(np.int64), signs)
    return output / max(float(np.sqrt(graph.pair_index.shape[1])), 1.0)


def _sample_rows(
    graphs: Sequence[RawGraph],
    getter,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Reservoir sample rows without materialising the complete train matrix."""
    limit = int(maximum)
    if limit < 2:
        raise ValueError("sample maximum must be at least two")
    rng = np.random.default_rng(int(seed))
    reservoir: list[np.ndarray] = []
    seen = 0
    for graph in graphs:
        values = np.asarray(getter(graph), dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        for row in values:
            if seen < limit:
                reservoir.append(row.copy())
            else:
                replacement = int(rng.integers(0, seen + 1))
                if replacement < limit:
                    reservoir[replacement] = row.copy()
            seen += 1
    if len(reservoir) < 2:
        raise ValueError("not enough rows for a train-only projector")
    return np.stack(reservoir, axis=0).astype(np.float32, copy=False)


def _sample_pair_rows(graphs: Sequence[RawGraph], maximum: int, seed: int) -> np.ndarray:
    """Sample a small, approximately graph-balanced relation matrix."""
    rng = np.random.default_rng(int(seed))
    graph_count = max(len(graphs), 1)
    per_graph = max(1, int(maximum) // graph_count)
    rows: list[np.ndarray] = []
    for graph in graphs:
        matrix = graph.pair_relation
        if matrix.shape[0] <= per_graph:
            rows.append(matrix)
        elif matrix.shape[0]:
            selected = rng.choice(matrix.shape[0], size=per_graph, replace=False)
            rows.append(matrix[np.sort(selected)])
    if not rows:
        raise ValueError("no pair relation rows available")
    result = np.concatenate(rows, axis=0)
    if result.shape[0] > int(maximum):
        selected = rng.choice(result.shape[0], size=int(maximum), replace=False)
        result = result[np.sort(selected)]
    return result.astype(np.float32, copy=False)


def _fit_projector(rows: np.ndarray, width: int, seed: int) -> Projector:
    matrix = np.asarray(rows, dtype=np.float32)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    standardized = ((matrix - mean) / scale).astype(np.float32, copy=False)
    components = min(int(width), standardized.shape[0], standardized.shape[1])
    if components < 1:
        raise ValueError("projector has no usable component")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=int(seed))
    pca.fit(standardized)
    return Projector(mean=mean, scale=scale, pca=pca)


def _split_descriptor(descriptors: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(descriptors, dtype=np.float32)
    if dataset == "zinc":
        from tracks.ksvd.experiments.luyin16.zinc_exact_patch_relation import (
            ATOM_CATEGORIES,
            MAX_PATCH_NODES,
            NODE_SLOT_WIDTH,
            PATCH_WIDTH,
        )

        if matrix.shape[1] != int(PATCH_WIDTH):
            raise ValueError(f"unexpected ZINC descriptor width {matrix.shape}")
        node_width = int(MAX_PATCH_NODES) * int(NODE_SLOT_WIDTH)
        node = matrix[:, :node_width].reshape(
            matrix.shape[0], int(MAX_PATCH_NODES), int(NODE_SLOT_WIDTH)
        )
        attribute = node[:, :, : int(ATOM_CATEGORIES)].reshape(matrix.shape[0], -1)
        structure = np.concatenate(
            [node[:, :, int(ATOM_CATEGORIES) :].reshape(matrix.shape[0], -1),
             matrix[:, node_width:]],
            axis=1,
        )
        return structure.astype(np.float32, copy=False), attribute.astype(np.float32, copy=False)
    if dataset == "molhiv":
        from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as base

        atom_shell = (base.PATCH_RADIUS + 1) * base.ATOM_WIDTH
        bond_shell = len(base.SHELL_PAIRS) * base.BOND_WIDTH
        root_start = atom_shell + bond_shell
        incident_start = root_start + base.ATOM_WIDTH
        expected = incident_start + base.BOND_WIDTH + 6
        if matrix.shape[1] != int(expected):
            raise ValueError(f"unexpected MolHIV descriptor width {matrix.shape}; expected {expected}")
        attribute = np.concatenate(
            [matrix[:, :atom_shell], matrix[:, root_start:incident_start], matrix[:, incident_start:incident_start + base.BOND_WIDTH]],
            axis=1,
        )
        structure = np.concatenate(
            [matrix[:, atom_shell:root_start], matrix[:, incident_start + base.BOND_WIDTH :]],
            axis=1,
        )
        return structure.astype(np.float32, copy=False), attribute.astype(np.float32, copy=False)
    raise ValueError(f"unknown dataset {dataset!r}")


def _fit_transforms(
    train_graphs: Sequence[RawGraph],
    *,
    dataset: str,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[TransformBundle, dict[str, Any]]:
    representation = config["representation"]
    model = config["model"]
    sample_limit = int(representation.get("max_patch_samples", 20000))
    descriptor_sample = _sample_rows(
        train_graphs, lambda graph: graph.descriptors, sample_limit, seed + 11
    )
    structure_sample, attribute_sample = _split_descriptor(descriptor_sample, dataset)
    relation_sample = _sample_pair_rows(
        train_graphs, int(representation.get("max_relation_samples", 20000)), seed + 13
    )
    context_sample = np.stack([graph.context for graph in train_graphs], axis=0).astype(
        np.float32, copy=False
    )
    projection_width = int(model.get("prototype_width", 16))
    relation_width = int(model.get("relation_code_width", 8))
    full = _fit_projector(descriptor_sample, projection_width, seed + 101)
    structure = _fit_projector(structure_sample, projection_width, seed + 102)
    attribute = _fit_projector(attribute_sample, projection_width, seed + 103)
    relation = _fit_projector(relation_sample, relation_width, seed + 104)
    context = _fit_projector(
        context_sample,
        int(model.get("context_width", 16)),
        seed + 105,
    )

    full_z = full.transform(descriptor_sample)
    dictionary, _, dictionary_info = ksvd(
        full_z.T.astype(np.float64, copy=False),
        n_atoms=min(int(model.get("prototype_atoms", 16)), full_z.shape[1], full_z.shape[0]),
        T=min(int(model.get("prototype_sparsity", 2)), full_z.shape[1]),
        T_min=1,
        n_iter=0,
        seed=seed + 107,
    )
    dictionary = np.asarray(dictionary, dtype=np.float64)
    bundle = TransformBundle(
        full=full,
        structure=structure,
        attribute=attribute,
        relation=relation,
        context=context,
        dictionary=dictionary,
        dictionary_gram=dictionary.T @ dictionary,
        dictionary_error=dict(dictionary_info),
    )
    metadata = {
        "patch_sample_rows": int(descriptor_sample.shape[0]),
        "relation_sample_rows": int(relation_sample.shape[0]),
        "descriptor_width": int(descriptor_sample.shape[1]),
        "structure_input_width": int(structure_sample.shape[1]),
        "attribute_input_width": int(attribute_sample.shape[1]),
        "prototype_width": int(dictionary.shape[1]),
        "prototype_sparsity": int(model.get("prototype_sparsity", 2)),
        "dictionary_mode": "train-only K-SVD initialization, zero update iterations",
        "dictionary": dictionary_info,
    }
    return bundle, metadata


def _graph_feature(
    graph: RawGraph,
    transforms: TransformBundle,
    *,
    dataset: str,
    layout: FeatureLayout,
    bond_width: int,
) -> np.ndarray:
    descriptors = graph.descriptors
    structure_raw, attribute_raw = _split_descriptor(descriptors, dataset)
    structure = transforms.structure.transform(structure_raw)
    attribute = transforms.attribute.transform(attribute_raw)
    prototype_input = transforms.full.transform(descriptors)
    prototype, prototype_error = _batch_omp(
        prototype_input,
        transforms.dictionary,
        sparsity=min(2, transforms.dictionary.shape[1]),
        gram=transforms.dictionary_gram,
        chunk_size=4096,
    )
    prototype = np.asarray(prototype, dtype=np.float32)
    n_centres = structure.shape[0]
    if n_centres == 0:
        raise ValueError("graph has no centres")
    structure_mean = structure.mean(axis=0)
    attribute_mean = attribute.mean(axis=0)
    structure_centered = structure - structure_mean
    attribute_centered = attribute - attribute_mean
    unary_structure = np.concatenate([structure_mean, structure.std(axis=0), structure.max(axis=0)])
    unary_attribute = np.concatenate([attribute_mean, attribute.std(axis=0), attribute.max(axis=0)])
    unary_prototype = np.concatenate([prototype.mean(axis=0), prototype.std(axis=0), prototype.max(axis=0)])
    self_joint = (structure_centered.T @ attribute_centered / float(n_centres)).reshape(-1)

    pair_count = int(graph.pair_index.shape[1])
    relation_codes = _relation_codes(
        graph.pair_relation,
        graph.pair_bucket,
        bond_width=int(bond_width),
        distance_buckets=int(layout.distance_buckets),
    )
    pair_relation_code = transforms.relation.transform(graph.pair_relation) if pair_count else np.zeros((0, layout.relation_width), dtype=np.float32)
    cross_blocks: list[np.ndarray] = []
    relation_blocks: list[np.ndarray] = []
    source = graph.pair_index[0] if pair_count else np.zeros(0, dtype=np.int64)
    target = graph.pair_index[1] if pair_count else np.zeros(0, dtype=np.int64)
    for bucket in range(int(layout.distance_buckets)):
        mask = graph.pair_bucket == int(bucket)
        indices = np.flatnonzero(mask)
        count = int(indices.size)
        cross_sum = np.zeros((layout.structure_width, layout.attribute_width), dtype=np.float64)
        relation_values = pair_relation_code[indices] if count else np.zeros((0, layout.relation_width), dtype=np.float32)
        if count:
            # Chunking keeps the temporary outer-product tensor bounded for
            # MolHIV, where a graph can contain hundreds of centre pairs.
            for start in range(0, count, 4096):
                current = indices[start : start + 4096]
                left = structure_centered[source[current]]
                right = attribute_centered[target[current]]
                reverse_left = structure_centered[target[current]]
                reverse_right = attribute_centered[source[current]]
                cross_sum += np.einsum("ni,nj->ij", left, right, optimize=True)
                cross_sum += np.einsum("ni,nj->ij", reverse_left, reverse_right, optimize=True)
            cross = (cross_sum / float(count)).astype(np.float32, copy=False)
            relation_mean = relation_values.mean(axis=0)
            relation_std = relation_values.std(axis=0)
        else:
            cross = np.zeros((layout.structure_width, layout.attribute_width), dtype=np.float32)
            relation_mean = np.zeros(layout.relation_width, dtype=np.float32)
            relation_std = np.zeros(layout.relation_width, dtype=np.float32)
        cross_blocks.append(cross.reshape(-1))
        relation_blocks.append(
            np.concatenate([relation_mean, relation_std, np.asarray([np.log1p(float(count))], dtype=np.float32)])
        )

    unary_sketch = _signed_sketch(graph.token_hash, layout.unary_sketch_width, 0xA17C9E3D)
    parent_sketch = _signed_sketch(graph.parent_hash, layout.parent_sketch_width, 0xB51D4A27)
    unary_sketch /= max(float(np.sqrt(n_centres)), 1.0)
    pair_sketch = _pair_sketch(graph, relation_codes, layout.pair_sketch_width)
    context = transforms.context.transform(graph.context)
    scalar = np.asarray(
        [
            np.log1p(float(n_centres)),
            np.log1p(float(pair_count)),
            float(np.mean(prototype_error)) if prototype_error.size else 0.0,
            float(np.std(prototype_error)) if prototype_error.size else 0.0,
        ],
        dtype=np.float32,
    )
    feature = np.concatenate(
        [
            unary_structure,
            unary_attribute,
            unary_prototype,
            self_joint.astype(np.float32, copy=False),
            *cross_blocks,
            *relation_blocks,
            unary_sketch,
            parent_sketch,
            pair_sketch,
            context,
            scalar,
        ]
    ).astype(np.float32, copy=False)
    if feature.shape != (layout.width,):
        raise RuntimeError(f"unified feature width changed: {feature.shape}; expected {(layout.width,)}")
    if not np.isfinite(feature).all():
        raise FloatingPointError("unified graph feature contains non-finite values")
    return feature


def _build_features(
    graphs: Sequence[RawGraph],
    transforms: TransformBundle,
    *,
    dataset: str,
    layout: FeatureLayout,
    bond_width: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    values: list[np.ndarray] = []
    labels: list[float] = []
    token_hashes: list[np.ndarray] = []
    for index, graph in enumerate(graphs):
        values.append(
            _graph_feature(
                graph,
                transforms,
                dataset=dataset,
                layout=layout,
                bond_width=bond_width,
            )
        )
        labels.append(float(graph.y))
        token_hashes.append(graph.token_hash)
        if (index + 1) % 1000 == 0 or index + 1 == len(graphs):
            print(f"unified features {dataset}/{split}: {index + 1}/{len(graphs)}", flush=True)
    matrix = np.stack(values, axis=0).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.float32)
    unique_tokens = len({int(value) for row in token_hashes for value in row})
    metadata = {
        "n_graphs": int(len(graphs)),
        "positive_graphs": int(np.sum(label_array > 0.5)),
        "mean_centres": float(np.mean([graph.descriptors.shape[0] for graph in graphs])),
        "mean_pairs": float(np.mean([graph.pair_index.shape[1] for graph in graphs])),
        "unique_patch_hashes": int(unique_tokens),
        "seconds": float(time.perf_counter() - started),
    }
    return matrix, label_array, metadata


class GraphFeatureStandardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "GraphFeatureStandardizer":
        matrix = np.asarray(values, dtype=np.float32)
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite graph-standardized feature")
        return output


class UnifiedMLP(nn.Module):
    def __init__(self, input_width: int, hidden: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(input_width), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(bottleneck)),
            nn.ReLU(),
            nn.Linear(int(bottleneck), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).view(-1)


def _metric(prediction: np.ndarray, target: np.ndarray, task: str) -> float:
    if task == "zinc":
        return float(mean_absolute_error(target.astype(np.float64), prediction.astype(np.float64)))
    return float(roc_auc_score(target.astype(np.float64), prediction.astype(np.float64)))


def _train_head(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    eval_values: np.ndarray,
    eval_labels: np.ndarray,
    *,
    task: str,
    config: Mapping[str, Any],
    seed: int,
    select_best: bool,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["model"]
    _seed_everything(seed)
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")
    model = UnifiedMLP(
        train_values.shape[1],
        int(model_config.get("head_hidden", 32)),
        int(model_config.get("head_bottleneck", 16)),
        float(model_config.get("dropout", 0.05)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_values), torch.from_numpy(train_labels.astype(np.float32))
    )
    eval_dataset = TensorDataset(
        torch.from_numpy(eval_values), torch.from_numpy(eval_labels.astype(np.float32))
    )
    generator = torch.Generator().manual_seed(int(seed) + 91011)
    loader = DataLoader(
        train_dataset,
        batch_size=int(model_config.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(model_config.get("batch_size", 256)),
        shuffle=False,
        num_workers=0,
    )
    positive = int(np.sum(train_labels > 0.5))
    negative = int(train_labels.size - positive)
    pos_weight = float(negative / max(positive, 1))
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    best_value = float("inf") if task == "zinc" else -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    trace: list[dict[str, float | int]] = []
    stale = 0
    patience = int(model_config.get("patience", 10))
    losses: list[float] = []

    def evaluate() -> float:
        model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for values, labels in eval_loader:
                values = values.to(device)
                predictions.append(model(values).cpu().numpy() if task == "zinc" else torch.sigmoid(model(values)).cpu().numpy())
                targets.append(labels.numpy())
        return _metric(np.concatenate(predictions), np.concatenate(targets), task)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for values, labels in loader:
            values = values.to(device)
            labels = labels.to(device)
            logits = model(values)
            if task == "zinc":
                loss = F.l1_loss(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * int(labels.shape[0])
            count += int(labels.shape[0])
        losses.append(total / max(count, 1))
        current = evaluate() if select_best else None
        if current is not None:
            trace.append({"epoch": int(epoch), "loss": float(losses[-1]), "metric": float(current)})
            improved = current < best_value if task == "zinc" else current > best_value
            if improved:
                best_value = float(current)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 10) == 0:
            suffix = "" if current is None else f" valid_{'mae' if task == 'zinc' else 'auc'}={current:.6f}"
            print(f"unified head task={task} phase={'select' if select_best else 'refit'} epoch={epoch:03d}/{epochs} loss={losses[-1]:.6f}{suffix}", flush=True)
        if select_best and stale >= patience:
            print(f"unified head early_stop task={task} epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_value = evaluate()
    return {
        "model": model,
        "metric": float(final_value),
        "best_metric": None if not select_best else float(best_value),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "pos_weight": pos_weight,
    }


def _load_zinc_graphs(config: Mapping[str, Any], split: str) -> tuple[list[RawGraph], dict[str, Any]]:
    data_root = _resolve(config["data"]["root"])
    cache_dir = _resolve(config["runtime"]["record_cache"])
    dataset_split = {"train": "train", "valid": "val", "test": "test"}[split]
    dataset = zinc_proxy._load_zinc(data_root, dataset_split)
    records, metadata, cache_hit = zinc_records._extract_or_load_split(
        data_root,
        dataset,
        split,
        cache_dir,
        {},
    )
    graphs = [_record_to_raw(record, exact_identity=True) for record in records]
    del records, dataset
    gc.collect()
    metadata = {**metadata, "cache_hit": bool(cache_hit)}
    return graphs, metadata


def _load_molhiv_graphs(config: Mapping[str, Any], split: str) -> tuple[list[RawGraph], dict[str, Any]]:
    data_root = _resolve(config["data"]["root"])
    cache_dir = _resolve(config["runtime"]["record_cache"])
    bundle = load_molhiv(root=data_root, with_features=True)
    indices = np.asarray(bundle.split[split], dtype=np.int64)
    records, metadata, cache_hit = molhiv_records._extract_or_load_split(
        bundle,
        indices,
        split,
        "shell",
        cache_dir,
    )
    graphs = [_record_to_raw(record, exact_identity=False) for record in records]
    del records, bundle
    gc.collect()
    metadata = {**metadata, "cache_hit": bool(cache_hit)}
    return graphs, metadata


def _load_graphs(config: Mapping[str, Any], dataset: str, split: str) -> tuple[list[RawGraph], dict[str, Any]]:
    return _load_zinc_graphs(config, split) if dataset == "zinc" else _load_molhiv_graphs(config, split)


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    metric = "MAE" if result["task"] == "zinc" else "ROC-AUC"
    return "\n".join(
        [
            f"# {result['protocol_id']}",
            "",
            "Unified relational patch readout: hierarchical patch keys, train-only K-SVD-init prototype code, invariant unary/joint moments, explicit distance-conditioned cross-centre moments, and fixed signed count sketches; one MLP head.",
            "",
            f"- dataset: `{result['dataset']}`; split: `{result['data']['split']}`",
            f"- descriptor: `{result['representation']['descriptor_width']}D`; shared graph feature width: `{result['representation']['feature_width']}D`",
            f"- prototype: `K={result['representation']['prototype_width']}`, `T={result['representation']['prototype_sparsity']}`, train-only K-SVD initialization",
            f"- readout: `{result['representation']['readout']}`",
            f"- message passing: `{result['representation']['message_passing']}`; attention: `{result['representation']['attention']}`",
            "",
            f"| head | valid {metric} | test {metric} after train+valid refit | selected epoch |",
            "|---|---:|---:|---:|",
            f"| `single_mlp` | {evaluation['valid']['metric']:.6f} | {evaluation['test_after_train_valid_refit']['metric']:.6f} | {evaluation['valid']['selected_epoch']} |",
            "",
            f"- trainable parameters: `{evaluation['parameters']}`",
            f"- valid-phase train-only descriptor sample: `{result['transforms']['valid_train_only']['patch_sample_rows']}` patch rows",
            f"- valid-phase train-only exact/descriptor patch hashes: `{result['feature_build']['valid_phase_train']['unique_patch_hashes']}`",
            f"- runtime: `{result['runtime']['seconds']:.1f}s`",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = str(config["dataset"])
    if dataset not in {"zinc", "molhiv"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))
    train_graphs, train_meta = _load_graphs(config, dataset, "train")
    valid_graphs, valid_meta = _load_graphs(config, dataset, "valid")
    transforms, transform_meta = _fit_transforms(
        train_graphs, dataset=dataset, config=config, seed=seed
    )
    model_config = config["model"]
    representation = config["representation"]
    layout = FeatureLayout(
        structure_width=int(model_config.get("prototype_width", 16)),
        attribute_width=int(model_config.get("prototype_width", 16)),
        prototype_width=int(model_config.get("prototype_atoms", 16)),
        relation_width=int(model_config.get("relation_code_width", 8)),
        distance_buckets=int(representation.get("distance_buckets", 5 if dataset == "zinc" else 6)),
        unary_sketch_width=int(representation.get("unary_sketch_width", 512)),
        parent_sketch_width=int(representation.get("parent_sketch_width", 128)),
        pair_sketch_width=int(representation.get("pair_sketch_width", 2048)),
        context_width=int(model_config.get("context_width", 16)),
    )
    bond_width = 4 if dataset == "zinc" else 13
    train_values, train_labels, train_feature_meta = _build_features(
        train_graphs, transforms, dataset=dataset, layout=layout, bond_width=bond_width, split="train"
    )
    valid_values, valid_labels, valid_feature_meta = _build_features(
        valid_graphs, transforms, dataset=dataset, layout=layout, bond_width=bond_width, split="valid"
    )
    feature_standardizer = GraphFeatureStandardizer.fit(train_values)
    train_values = feature_standardizer.transform(train_values)
    valid_values = feature_standardizer.transform(valid_values)
    valid_phase = _train_head(
        train_values,
        train_labels,
        valid_values,
        valid_labels,
        task=dataset,
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 50)),
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    valid_metric = float(valid_phase["metric"])
    parameters = int(valid_phase["parameters"])
    del valid_phase["model"]
    # The validation representation is deliberately discarded before the
    # terminal refit.  The refit must rebuild every learned coordinate and
    # the graph-level standardizer on official train+valid only.
    del train_values, valid_values, transforms
    gc.collect()

    refit_graphs = [*train_graphs, *valid_graphs]
    del train_graphs, valid_graphs
    refit_transforms, refit_transform_meta = _fit_transforms(
        refit_graphs, dataset=dataset, config=config, seed=seed
    )
    refit_values, refit_labels, refit_train_feature_meta = _build_features(
        refit_graphs,
        refit_transforms,
        dataset=dataset,
        layout=layout,
        bond_width=bond_width,
        split="train+valid",
    )
    del refit_graphs
    gc.collect()

    test_graphs, test_meta = _load_graphs(config, dataset, "test")
    test_values, test_labels, test_feature_meta = _build_features(
        test_graphs,
        refit_transforms,
        dataset=dataset,
        layout=layout,
        bond_width=bond_width,
        split="test",
    )
    del refit_transforms
    refit_feature_standardizer = GraphFeatureStandardizer.fit(refit_values)
    refit_train_values = refit_feature_standardizer.transform(refit_values)
    test_values = refit_feature_standardizer.transform(test_values)
    refit_phase = _train_head(
        refit_train_values,
        refit_labels,
        test_values,
        test_labels,
        task=dataset,
        config=config,
        seed=seed,
        select_best=False,
        epochs=selected_epoch,
    )
    test_metric = float(refit_phase["metric"])
    del refit_phase["model"], test_graphs, refit_values, refit_train_values
    gc.collect()
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "dataset": dataset,
        "task": dataset,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(_resolve(config["data"]["root"])),
            "split": "PyG ZINC official train/valid/test" if dataset == "zinc" else "OGB ogbg-molhiv official scaffold train/valid/test",
            "sizes": {"train": int(train_labels.size), "valid": int(valid_labels.size), "test": int(test_labels.size)},
            "positive": None if dataset == "zinc" else {"train": int(np.sum(train_labels > 0.5)), "valid": int(np.sum(valid_labels > 0.5)), "test": int(np.sum(test_labels > 0.5))},
            "test_labels_used_for_selection": False,
            "representation_fit_scope": (
                "official train only for valid; official train+valid for terminal test refit"
            ),
        },
        "representation": {
            "descriptor_width": int(transform_meta["descriptor_width"]),
            "structure_code_width": int(layout.structure_width),
            "attribute_code_width": int(layout.attribute_width),
            "prototype_width": int(layout.prototype_width),
            "prototype_sparsity": int(model_config.get("prototype_sparsity", 2)),
            "relation_code_width": int(layout.relation_width),
            "distance_buckets": int(layout.distance_buckets),
            "unary_sketch_width": int(layout.unary_sketch_width),
            "parent_sketch_width": int(layout.parent_sketch_width),
            "pair_sketch_width": int(layout.pair_sketch_width),
            "context_width": int(layout.context_width),
            "feature_width": int(layout.width),
            "readout": "unary mean/std/max + within-patch centered joint + distance-conditioned cross-centre centered joint + exact/parent unary sketches + unordered patch-pair/relation sketch + relation moments",
            "message_passing": False,
            "attention": False,
            "one_predictive_head": True,
            "label_free": True,
        },
        "transforms": {
            "valid_train_only": transform_meta,
            "test_refit_train_valid": refit_transform_meta,
        },
        "feature_build": {
            "valid_phase_train": train_feature_meta,
            "valid_phase_valid": valid_feature_meta,
            "test_refit_train_valid": refit_train_feature_meta,
            "test_refit_test": test_feature_meta,
            "source_train": train_meta,
            "source_valid": valid_meta,
            "source_test": test_meta,
        },
        "training": {**dict(model_config), "loss": "L1 / mean absolute error" if dataset == "zinc" else "balanced binary cross entropy with logits", "metric": "MAE" if dataset == "zinc" else "ROC-AUC", "one_predictive_head": True},
        "evaluation": {
            "valid": {"metric": valid_metric, "selected_epoch": selected_epoch, "best_metric": valid_metric},
            "test_after_train_valid_refit": {"metric": test_metric, "epochs_run": selected_epoch},
            "parameters": parameters,
        },
        "runtime": {"seconds": float(time.perf_counter() - started), "python": sys.version, "platform": platform.platform(), "numpy": importlib.metadata.version("numpy"), "torch": importlib.metadata.version("torch"), "script_sha256": _sha256(Path(__file__).resolve())},
    }
    output_json = _resolve(config["output"]["json"])
    output_markdown = _resolve(config["output"]["markdown"])
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    metric = "valid_mae" if result["task"] == "zinc" else "valid_auc"
    test_metric = "test_mae" if result["task"] == "zinc" else "test_auc"
    print(json.dumps({metric: result["evaluation"]["valid"]["metric"], test_metric: result["evaluation"]["test_after_train_valid_refit"]["metric"], "selected_epoch": result["evaluation"]["valid"]["selected_epoch"], "parameters": result["evaluation"]["parameters"], "feature_width": result["representation"]["feature_width"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
