"""Unified, trainable typed relational patch-set encoder.

This is the v2 follow-up to ``unified_relational_patch_readout.py``.  It keeps
the same *calculation* for ZINC and MolHIV, but moves the learnable functions to
the patch and pair level before invariant pooling:

    structure/attribute patch adapters
        -> low-rank conditional binding + exact/backoff token embedding
        -> shared patch encoder
        -> explicit symmetric pair encoder
        -> sum/sum-of-squares/max/count invariant readout
        -> one graph-level head

There is no message passing, attention, K-SVD, or score-level ensemble.  Both
datasets can use an exact rooted typed certificate as a train-only vocabulary
embedding when their record mode provides one.  The continuous
structure/attribute adapters and explicit pair relation are kept in place so
the certificate is an additional identity channel, rather than a replacement
for the fixed-width descriptors.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import platform
import pickle
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16 import molhiv_ksvd_patch_path_pooling as molhiv_shell_records
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as molhiv_exact_records
from tracks.ksvd.experiments.luyin16 import zinc_ksvd_patch_path_pooling as zinc_records
from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zinc_proxy
from tracks.ksvd.experiments.luyin16.unified_relational_patch_readout import (
    _coarse_patch_key,
    _split_descriptor,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/unified_typed_relational_patch_encoder_zinc.yaml"
)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = (
            (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        ).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized values")
        return output


@dataclass
class TransformBundle:
    structure: Standardizer
    attribute: Standardizer
    auxiliary: Standardizer
    relation: Standardizer
    context: Standardizer
    typed_vocabulary: dict[bytes, int]
    backoff_vocabulary: dict[bytes, int]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    structure_width: int
    attribute_width: int
    relation_width: int
    context_width: int
    distance_buckets: int
    typed_certificates: bool


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


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _fit_standardizer(rows: np.ndarray) -> Standardizer:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"standardizer expects a non-empty matrix, got {matrix.shape}")
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    return Standardizer(mean=mean, scale=scale)


def _fit_streaming(
    records: Sequence[Any],
    getter,
    width: int,
) -> Standardizer:
    total = np.zeros(int(width), dtype=np.float64)
    squared = np.zeros(int(width), dtype=np.float64)
    count = 0
    for record in records:
        values = np.asarray(getter(record), dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != int(width):
            raise ValueError(
                f"streaming feature width changed: {values.shape[1]} != {width}"
            )
        values64 = values.astype(np.float64, copy=False)
        total += values64.sum(axis=0, dtype=np.float64)
        squared += np.square(values64, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += int(values.shape[0])
    if count < 1:
        raise ValueError("cannot fit a standardizer on an empty record collection")
    mean = total / float(count)
    variance = np.maximum(squared / float(count) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
    return Standardizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )


def _descriptor_matrix(record: Any) -> np.ndarray:
    return np.stack(
        [np.asarray(patch.shell_descriptor, dtype=np.float32) for patch in record.patches],
        axis=0,
    ).astype(np.float32, copy=False)


def _patch_auxiliary(record: Any, dataset: str) -> np.ndarray:
    """Return mass/geometry features that are not normalized composition.

    The raw descriptors intentionally contain different schemas for ZINC and
    MolHIV.  These four invariant quantities make the shared patch encoder
    aware of patch mass and boundary size without introducing node IDs.
    """
    descriptors = _descriptor_matrix(record)
    rows: list[list[float]] = []
    for patch, descriptor in zip(record.patches, descriptors, strict=True):
        n_nodes = float(len(patch.nodes))
        n_boundary = float(len(patch.boundary))
        if dataset == "zinc":
            # The exact ZINC descriptor ends in one-hot induced-edge slots.
            # Count occupied edge slots; each occupied slot has exactly one
            # bond category coordinate.
            from tracks.ksvd.experiments.luyin16.zinc_exact_patch_relation import (
                BOND_CATEGORIES,
                MAX_PATCH_NODES,
                NODE_SLOT_WIDTH,
            )

            edge_start = int(MAX_PATCH_NODES) * int(NODE_SLOT_WIDTH)
            edge_values = descriptor[edge_start:]
            edge_count = float(
                np.count_nonzero(
                    edge_values.reshape(-1, int(BOND_CATEGORIES)).sum(axis=1)
                )
            )
        else:
            # MolHIV shell descriptors end with
            # [log(nodes), log(edges), boundary fraction, cycle, ...].
            edge_log = float(descriptor[-5]) if descriptor.size >= 6 else 0.0
            edge_count = float(np.expm1(np.clip(edge_log, 0.0, 20.0)))
        norm = float(np.linalg.norm(descriptor.astype(np.float64)))
        rows.append(
            [
                float(np.log1p(n_nodes)),
                float(np.log1p(n_boundary)),
                float(np.log1p(edge_count)),
                float(np.log1p(norm)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _parts_for_record(record: Any, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    return _split_descriptor(_descriptor_matrix(record), dataset)


def _typed_key(patch: Any, dataset: str) -> bytes | None:
    del dataset
    if getattr(patch, "typed_certificate", b""):
        return bytes(patch.typed_certificate)
    return None


def _backoff_key(patch: Any, dataset: str) -> bytes | None:
    # Preserve the established ZINC coarse fallback.  For rooted MolHIV the
    # natural hierarchy is exact radius-2 certificate -> exact radius-1 parent
    # certificate.  Shell records have neither field and intentionally remain
    # OOV-only, which keeps the old control protocol reproducible.
    if dataset == "molhiv" and getattr(patch, "parent_certificate", b""):
        return b"parent-v1:" + bytes(patch.parent_certificate)
    if dataset != "zinc":
        return None
    descriptor = np.asarray(patch.shell_descriptor, dtype=np.float32)
    value = _coarse_patch_key(descriptor)
    return b"coarse-v1:" + np.asarray(value, dtype=np.uint64).tobytes()


def _fit_vocabulary(
    records: Sequence[Any],
    *,
    kind: str,
    dataset: str,
    maximum: int,
    minimum_frequency: int,
) -> dict[bytes, int]:
    counts: Counter[bytes] = Counter()
    for record in records:
        for patch in record.patches:
            key = (
                _typed_key(patch, dataset)
                if kind == "typed"
                else _backoff_key(patch, dataset)
            )
            if key is not None:
                counts[key] += 1
    ordered = sorted(
        (key for key, count in counts.items() if count >= int(minimum_frequency)),
        key=lambda key: (-counts[key], key),
    )[: int(maximum)]
    # Zero is reserved for an unseen/OOV object.
    return {key: index + 1 for index, key in enumerate(ordered)}


def _vocabulary_stats(
    records: Sequence[Any],
    vocabulary: Mapping[bytes, int],
    *,
    kind: str,
    dataset: str,
) -> dict[str, Any]:
    values: list[bytes] = []
    for record in records:
        for patch in record.patches:
            key = (
                _typed_key(patch, dataset)
                if kind == "typed"
                else _backoff_key(patch, dataset)
            )
            if key is not None:
                values.append(key)
    known = sum(value in vocabulary for value in values)
    unique = set(values)
    return {
        "occurrences": int(len(values)),
        "unique_types": int(len(unique)),
        "vocabulary_size": int(len(vocabulary)),
        "known_occurrence_fraction": float(known / max(len(values), 1)),
        "known_type_fraction": float(
            sum(value in vocabulary for value in unique) / max(len(unique), 1)
        ),
    }


def _fit_transforms(
    records: Sequence[Any],
    *,
    dataset: str,
    spec: DatasetSpec,
    representation: Mapping[str, Any],
) -> tuple[TransformBundle, dict[str, Any]]:
    structure = _fit_streaming(
        records,
        lambda record: _parts_for_record(record, dataset)[0],
        spec.structure_width,
    )
    attribute = _fit_streaming(
        records,
        lambda record: _parts_for_record(record, dataset)[1],
        spec.attribute_width,
    )
    auxiliary = _fit_streaming(
        records,
        lambda record: _patch_auxiliary(record, dataset),
        4,
    )
    relation = _fit_streaming(
        records,
        lambda record: record.pair_relation,
        spec.relation_width,
    )
    context = _fit_standardizer(
        np.stack(
            [np.asarray(record.global_context, dtype=np.float32) for record in records],
            axis=0,
        )
    )
    typed = _fit_vocabulary(
        records,
        kind="typed",
        dataset=dataset,
        maximum=int(representation.get("max_typed_tokens", 8192)),
        minimum_frequency=int(representation.get("minimum_typed_frequency", 1)),
    )
    backoff = _fit_vocabulary(
        records,
        kind="backoff",
        dataset=dataset,
        maximum=int(representation.get("max_backoff_tokens", 2048)),
        minimum_frequency=int(representation.get("minimum_backoff_frequency", 1)),
    )
    bundle = TransformBundle(
        structure=structure,
        attribute=attribute,
        auxiliary=auxiliary,
        relation=relation,
        context=context,
        typed_vocabulary=typed,
        backoff_vocabulary=backoff,
    )
    metadata = {
        "structure_input_width": int(spec.structure_width),
        "attribute_input_width": int(spec.attribute_width),
        "relation_input_width": int(spec.relation_width),
        "context_input_width": int(spec.context_width),
        "typed_vocabulary_size_with_oov": int(len(typed) + 1),
        "backoff_vocabulary_size_with_oov": int(len(backoff) + 1),
        "typed_vocabulary": _vocabulary_stats(
            records, typed, kind="typed", dataset=dataset
        ),
        "backoff_vocabulary": _vocabulary_stats(
            records, backoff, kind="backoff", dataset=dataset
        ),
        "fit_graphs": int(len(records)),
        "typed_mode": (
            "exact rooted typed certificate"
            if any(
                getattr(patch, "typed_certificate", b"")
                for record in records
                for patch in record.patches
            )
            else "OOV-only"
        ),
        "backoff_mode": (
            "exact radius-1 parent certificate for MolHIV"
            if dataset == "molhiv"
            and any(
                getattr(patch, "parent_certificate", b"")
                for record in records
                for patch in record.patches
            )
            else (
                "coarse invariant descriptor key"
                if dataset == "zinc"
                else "OOV-only"
            )
        ),
    }
    return bundle, metadata


def _encode_records(
    records: Sequence[Any],
    *,
    dataset: str,
    transforms: TransformBundle,
) -> list[Data]:
    output: list[Data] = []
    for record in records:
        structure_raw, attribute_raw = _parts_for_record(record, dataset)
        auxiliary_raw = _patch_auxiliary(record, dataset)
        structure = transforms.structure.transform(structure_raw)
        attribute = transforms.attribute.transform(attribute_raw)
        auxiliary = transforms.auxiliary.transform(auxiliary_raw)
        typed = np.asarray(
            [
                transforms.typed_vocabulary.get(_typed_key(patch, dataset), 0)
                if _typed_key(patch, dataset) is not None
                else 0
                for patch in record.patches
            ],
            dtype=np.int64,
        )
        backoff = np.asarray(
            [
                transforms.backoff_vocabulary.get(_backoff_key(patch, dataset), 0)
                if _backoff_key(patch, dataset) is not None
                else 0
                for patch in record.patches
            ],
            dtype=np.int64,
        )
        context = transforms.context.transform(
            np.asarray(record.global_context, dtype=np.float32)[None, :]
        )
        output.append(
            Data(
                patch_structure=torch.from_numpy(structure),
                patch_attribute=torch.from_numpy(attribute),
                patch_auxiliary=torch.from_numpy(auxiliary),
                typed_token=torch.from_numpy(typed),
                backoff_token=torch.from_numpy(backoff),
                pair_index=torch.from_numpy(np.asarray(record.pair_index, dtype=np.int64)),
                pair_relation=torch.from_numpy(
                    transforms.relation.transform(
                        np.asarray(record.pair_relation, dtype=np.float32)
                    )
                ),
                pair_bucket=torch.from_numpy(
                    np.asarray(record.pair_bucket, dtype=np.int64)
                ),
                global_context=torch.from_numpy(context),
                y=torch.tensor([float(record.y)], dtype=torch.float32),
                num_nodes=len(record.patches),
            )
        )
    return output


class _MLPBlock(nn.Module):
    def __init__(self, width: int, hidden: int, output: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(width), int(hidden)),
            nn.LayerNorm(int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(output)),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _FactorizedEmbedding(nn.Module):
    """One row per vocabulary object with a compact low-rank parameterization."""

    def __init__(self, vocabulary_size: int, output_width: int, rank: int) -> None:
        super().__init__()
        if int(vocabulary_size) < 1:
            raise ValueError("vocabulary_size must be positive")
        if int(rank) < 1 or int(rank) > int(output_width):
            raise ValueError("embedding rank must be in [1, output_width]")
        self.embedding = nn.Embedding(int(vocabulary_size), int(rank))
        self.projection = nn.Linear(int(rank), int(output_width), bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(values.long()))


class RelationalPatchSetModel(nn.Module):
    """The single predictive model shared by both dataset adapters."""

    def __init__(
        self,
        *,
        spec: DatasetSpec,
        typed_vocabulary_size: int,
        backoff_vocabulary_size: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        model_config = config["model"]
        structure_code = int(model_config.get("structure_code_width", 32))
        attribute_code = int(model_config.get("attribute_code_width", 32))
        binding_width = int(model_config.get("binding_width", 16))
        token_width = int(model_config.get("token_width", 24))
        backoff_width = int(model_config.get("backoff_width", 12))
        self.patch_hidden = int(model_config.get("patch_hidden", 48))
        self.pair_hidden = int(model_config.get("pair_hidden", 24))
        self.distance_buckets = int(spec.distance_buckets)
        dropout = float(model_config.get("dropout", 0.05))

        self.structure_encoder = _MLPBlock(
            spec.structure_width,
            max(structure_code, 64),
            structure_code,
            dropout,
        )
        self.attribute_encoder = _MLPBlock(
            spec.attribute_width,
            max(attribute_code, 64),
            attribute_code,
            dropout,
        )
        self.typed_embedding = _FactorizedEmbedding(
            int(typed_vocabulary_size), token_width, int(model_config.get("token_rank", 16))
        )
        self.backoff_embedding = _FactorizedEmbedding(
            int(backoff_vocabulary_size),
            backoff_width,
            min(int(model_config.get("backoff_rank", 8)), backoff_width),
        )
        self.binding_structure = nn.Linear(structure_code, binding_width, bias=False)
        self.binding_attribute = nn.Linear(attribute_code, binding_width, bias=False)
        fusion_width = (
            structure_code
            + attribute_code
            + binding_width
            + token_width
            + backoff_width
            + 4
        )
        self.patch_fusion = _MLPBlock(
            fusion_width,
            max(self.patch_hidden, 64),
            self.patch_hidden,
            dropout,
        )

        self.relation_encoder = _MLPBlock(
            spec.relation_width,
            max(self.pair_hidden, 32),
            self.pair_hidden,
            dropout,
        )
        self.pair_projection = nn.Linear(self.patch_hidden, self.pair_hidden, bias=False)
        self.distance_gate = nn.Embedding(self.distance_buckets, self.pair_hidden)
        self.pair_encoder = _MLPBlock(
            4 * self.pair_hidden,
            max(2 * self.pair_hidden, 48),
            self.pair_hidden,
            dropout,
        )

        context_hidden = int(model_config.get("context_code_width", 16))
        self.context_encoder = _MLPBlock(
            spec.context_width,
            max(context_hidden, 32),
            context_hidden,
            dropout,
        )
        self.context_hidden = context_hidden
        # Each invariant block contains sum, sum-of-squares, max, and one
        # scalar log-count.  The vector-valued part therefore has 3d, not 4d,
        # coordinates.
        node_readout_width = 3 * self.patch_hidden + 1
        pair_readout_width = 3 * self.pair_hidden + 1
        head_input = (
            node_readout_width
            + self.distance_buckets * pair_readout_width
            + context_hidden
        )
        head_hidden = int(model_config.get("head_hidden", 64))
        self.head = nn.Sequential(
            nn.Linear(head_input, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, int(model_config.get("head_bottleneck", 32))),
            nn.ReLU(),
            nn.Linear(int(model_config.get("head_bottleneck", 32)), 1),
        )

    @staticmethod
    def _grouped_max(
        values: torch.Tensor,
        groups: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        maximum = torch.full(
            (int(n_groups), values.shape[1]),
            -float("inf"),
            dtype=values.dtype,
            device=values.device,
        )
        if values.numel():
            maximum.scatter_reduce_(
                0,
                groups.view(-1, 1).expand(-1, values.shape[1]),
                values,
                reduce="amax",
                include_self=True,
            )
        return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))

    @classmethod
    def _pool(
        cls,
        values: torch.Tensor,
        groups: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        total = torch.zeros(
            (int(n_groups), values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        total.index_add_(0, groups, values)
        squared = torch.zeros_like(total)
        squared.index_add_(0, groups, values * values)
        counts = torch.bincount(groups, minlength=int(n_groups)).to(values.dtype)
        counts = counts.unsqueeze(1)
        maximum = cls._grouped_max(values, groups, n_groups)
        return torch.cat([total, squared, maximum, torch.log1p(counts)], dim=1)

    def _pool_pairs(
        self,
        values: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for bucket in range(self.distance_buckets):
            mask = pair_bucket == int(bucket)
            blocks.append(
                self._pool(values[mask], pair_batch[mask], n_graphs)
            )
        return torch.cat(blocks, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        n_graphs = int(data.y.view(-1).shape[0])
        structure = self.structure_encoder(data.patch_structure)
        attribute = self.attribute_encoder(data.patch_attribute)
        binding = self.binding_structure(structure) * self.binding_attribute(attribute)
        patch_input = torch.cat(
            [
                structure,
                attribute,
                binding,
                self.typed_embedding(data.typed_token),
                self.backoff_embedding(data.backoff_token),
                data.patch_auxiliary,
            ],
            dim=1,
        )
        patch = self.patch_fusion(patch_input)
        unary = self._pool(patch, data.batch, n_graphs)

        source = data.pair_index[0]
        target = data.pair_index[1]
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket.long()))
        pair_input = torch.cat(
            [left + right, torch.abs(left - right), left * right * gate, relation],
            dim=1,
        )
        pair = self.pair_encoder(pair_input)
        pair_batch = data.batch[source]
        pair_readout = self._pool_pairs(
            pair, pair_batch, data.pair_bucket.long(), n_graphs
        )
        context = self.context_encoder(data.global_context)
        if context.shape[0] != n_graphs:
            raise RuntimeError(
                f"context graph count changed: {context.shape[0]} != {n_graphs}"
            )
        return self.head(torch.cat([unary, pair_readout, context], dim=1)).view(-1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            predictions.append(
                logits.cpu().numpy()
                if task == "zinc"
                else torch.sigmoid(logits).cpu().numpy()
            )
            targets.append(batch.y.view(-1).cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    if task == "zinc":
        return float(mean_absolute_error(target, pred))
    return float(roc_auc_score(target, pred))


def _train_phase(
    train_data: Sequence[Data],
    eval_data: Sequence[Data],
    *,
    spec: DatasetSpec,
    transforms: TransformBundle,
    config: Mapping[str, Any],
    seed: int,
    task: str,
    select_best: bool,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["model"]
    _seed_everything(seed)
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")
    model = RelationalPatchSetModel(
        spec=spec,
        typed_vocabulary_size=len(transforms.typed_vocabulary) + 1,
        backoff_vocabulary_size=len(transforms.backoff_vocabulary) + 1,
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    batch_size = int(model_config.get("batch_size", 128))
    loader = DataLoader(
        list(train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed) + 91011),
        num_workers=0,
    )
    eval_loader = DataLoader(
        list(eval_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    positive = sum(float(graph.y.item()) > 0.5 for graph in train_data)
    negative = len(train_data) - int(positive)
    pos_weight = float(negative / max(int(positive), 1))
    weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    best_value = float("inf") if task == "zinc" else -float("inf")
    best_epoch = 1
    best_state: dict[str, Any] | None = None
    stale = 0
    patience = int(model_config.get("patience", 10))
    losses: list[float] = []
    trace: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            target = batch.y.view(-1)
            loss = (
                F.l1_loss(logits, target)
                if task == "zinc"
                else F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * int(target.shape[0])
            count += int(target.shape[0])
        epoch_loss = total / max(count, 1)
        losses.append(float(epoch_loss))
        current = _evaluate(model, eval_loader, device, task) if select_best else None
        if current is not None:
            key = "mae" if task == "zinc" else "auc"
            trace.append({"epoch": int(epoch), "loss": float(epoch_loss), key: float(current)})
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
            print(
                f"unified-v2 task={task} phase={'select' if select_best else 'refit'} "
                f"epoch={epoch:03d}/{epochs} loss={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"unified-v2 early_stop task={task} epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final = _evaluate(model, eval_loader, device, task)
    return {
        "model": model,
        "metric": float(final),
        "best_metric": None if not select_best else float(best_value),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "losses": losses,
        "trace": trace,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "pos_weight": float(pos_weight),
    }


def _load_zinc_graphs(config: Mapping[str, Any], split: str) -> tuple[list[Any], dict[str, Any], bool]:
    root = _resolve(config["data"]["root"])
    cache = _resolve(config["runtime"]["zinc_record_cache"])
    dataset = zinc_proxy._load_zinc(root, {"train": "train", "valid": "val", "test": "test"}[split])
    records, metadata, hit = zinc_records._extract_or_load_split(
        root, dataset, split, cache, {}
    )
    del dataset
    gc.collect()
    return records, {**metadata, "cache_hit": bool(hit)}, bool(hit)


def _load_molhiv_graphs(config: Mapping[str, Any], split: str) -> tuple[list[Any], dict[str, Any], bool]:
    root = _resolve(config["data"]["root"])
    cache = _resolve(config["runtime"]["molhiv_record_cache"])
    bundle = load_molhiv(root=root, with_features=True)
    indices = np.asarray(bundle.split[split], dtype=np.int64)
    mode = str(config.get("runtime", {}).get("molhiv_record_mode", "shell"))
    if mode == "exact_rooted":
        records, metadata, hit = _extract_or_load_molhiv_exact_rooted(
            bundle, indices, split, cache
        )
    elif mode == "shell":
        records, metadata, hit = molhiv_shell_records._extract_or_load_split(
            bundle, indices, split, "shell", cache
        )
    else:
        raise ValueError(f"unknown MolHIV record mode: {mode!r}")
    del bundle
    gc.collect()
    return records, {**metadata, "cache_hit": bool(hit)}, bool(hit)


def _molhiv_record_cache_signature(
    bundle: Any,
    indices: Sequence[int],
    split: str,
) -> str:
    """Signature for the independent exact-rooted MolHIV record cache."""
    digest = hashlib.sha256()
    digest.update(b"molhiv-unified-exact-rooted-record-cache-v1")
    digest.update(str(split).encode("utf-8"))
    digest.update(str(bundle.meta.get("name", "ogbg-molhiv")).encode("utf-8"))
    digest.update(str(bundle.meta.get("n_full", "")).encode("ascii"))
    digest.update(str(bundle.meta.get("n_used", "")).encode("ascii"))
    digest.update(
        np.asarray(bundle.meta.get("original_indices", []), dtype=np.int64).tobytes()
    )
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _extract_or_load_molhiv_exact_rooted(
    bundle: Any,
    indices: Sequence[int],
    split: str,
    cache_dir: Path,
) -> tuple[list[Any], dict[str, Any], bool]:
    """Build/load exact rooted records without touching the shell cache."""
    path = cache_dir / f"exact_rooted_{split}.pkl"
    signature = _molhiv_record_cache_signature(bundle, indices, split)
    if path.exists():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"exact-rooted record cache is not a mapping: {path}")
        if (
            payload.get("schema") != "molhiv-unified-exact-rooted-record-cache-v1"
            or payload.get("signature") != signature
        ):
            raise ValueError(f"exact-rooted record cache signature mismatch: {path}")
        records = payload.get("records")
        metadata = payload.get("metadata")
        if not isinstance(records, list) or not isinstance(metadata, Mapping):
            raise ValueError(f"exact-rooted record cache payload is malformed: {path}")
        print(f"MolHIV unified exact-rooted record cache hit: {path}", flush=True)
        return records, dict(metadata), True

    # The exact implementation owns the canonical colored-incidence
    # certificate and the explicit pair construction.  A fresh cache dict is
    # sufficient here because this loader is called once per split and the
    # persisted record cache avoids recomputation on later runs.
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV exact-rooted records require node and edge features")
    records, metadata = molhiv_exact_records._extract_split(
        bundle,
        indices,
        split,
        {},
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "molhiv-unified-exact-rooted-record-cache-v1",
        "signature": signature,
        "records": list(records),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f"MolHIV unified exact-rooted record cache saved: {path}", flush=True)
    return records, metadata, False


def _load_graphs(config: Mapping[str, Any], dataset: str, split: str):
    return (
        _load_zinc_graphs(config, split)
        if dataset == "zinc"
        else _load_molhiv_graphs(config, split)
    )


def _spec_from_records(dataset: str, records: Sequence[Any]) -> DatasetSpec:
    if not records:
        raise ValueError("cannot infer a dataset spec from empty records")
    structure, attribute = _parts_for_record(records[0], dataset)
    relation_width = int(records[0].pair_relation.shape[1])
    context_width = int(np.asarray(records[0].global_context).reshape(-1).shape[0])
    distance_buckets = (
        5 if dataset == "zinc" else 6
    )
    return DatasetSpec(
        name=dataset,
        structure_width=int(structure.shape[1]),
        attribute_width=int(attribute.shape[1]),
        relation_width=relation_width,
        context_width=context_width,
        distance_buckets=distance_buckets,
        typed_certificates=bool(
            getattr(records[0].patches[0], "typed_certificate", b"")
        ),
    )


def _render_markdown(result: Mapping[str, Any]) -> str:
    task = result["task"]
    metric = "MAE" if task == "zinc" else "ROC-AUC"
    evaluation = result["evaluation"]
    return "\n".join(
        [
            f"# {result['protocol_id']}",
            "",
            "Trainable patch/pair encoder with explicit symmetric relations and one invariant readout head; no message passing, attention, K-SVD, or ensemble.",
            "",
            f"- dataset: `{result['dataset']}`; split: `{result['data']['split']}`",
            f"- patch input: structure `{result['representation']['structure_input_width']}D` + attribute `{result['representation']['attribute_input_width']}D` + auxiliary mass `4D`",
            f"- exact/backoff: `{result['representation']['typed_mode']}`; `{result['representation']['backoff_mode']}`",
            f"- pair input: relation `{result['representation']['relation_input_width']}D`, `{result['representation']['distance_buckets']}` distance buckets",
            f"- readout: `{result['representation']['readout']}`",
            f"- message passing: `{result['representation']['message_passing']}`; attention: `{result['representation']['attention']}`; K-SVD: `{result['representation']['ksvd']}`",
            "",
            f"| head | valid {metric} | test {metric} after train+valid refit | selected epoch |",
            "|---|---:|---:|---:|",
            f"| `one_head` | {evaluation['valid']['metric']:.6f} | {evaluation['test_after_train_valid_refit']['metric']:.6f} | {evaluation['valid']['selected_epoch']} |",
            "",
            f"- trainable parameters: `{evaluation['parameters']}`",
            f"- valid train-only exact token coverage: `{result['transforms']['valid_train_only']['typed_vocabulary']['known_occurrence_fraction']:.4f}`",
            f"- runtime: `{result['runtime']['seconds']:.1f}s`",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = str(config["dataset"])
    if dataset not in {"zinc", "molhiv"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    task = dataset
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))

    # Keep the raw test records out of memory until validation and the
    # train+valid refit representation have been built.  MolHIV's label-free
    # record cache is several gigabytes when unpickled.
    train_records, train_meta, train_hit = _load_graphs(config, dataset, "train")
    valid_records, valid_meta, valid_hit = _load_graphs(config, dataset, "valid")
    spec = _spec_from_records(dataset, train_records)
    split_sizes = {
        "train": int(len(train_records)),
        "valid": int(len(valid_records)),
    }
    split_positive = {
        "train": int(sum(float(record.y) > 0.5 for record in train_records)),
        "valid": int(sum(float(record.y) > 0.5 for record in valid_records)),
    }
    representation = config["representation"]
    valid_transforms, valid_transform_meta = _fit_transforms(
        train_records,
        dataset=dataset,
        spec=spec,
        representation=representation,
    )
    valid_train_data = _encode_records(
        train_records, dataset=dataset, transforms=valid_transforms
    )
    valid_eval_data = _encode_records(
        valid_records, dataset=dataset, transforms=valid_transforms
    )
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        spec=spec,
        transforms=valid_transforms,
        config=config,
        seed=seed,
        task=task,
        select_best=True,
        epochs=int(config["model"].get("epochs", 50 if dataset == "zinc" else 20)),
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    valid_metric = float(valid_phase["metric"])
    parameters = int(valid_phase["parameters"])
    del valid_phase["model"], valid_train_data, valid_eval_data, valid_transforms
    gc.collect()

    refit_records = [*train_records, *valid_records]
    refit_transforms, refit_transform_meta = _fit_transforms(
        refit_records,
        dataset=dataset,
        spec=spec,
        representation=representation,
    )
    refit_train_data = _encode_records(
        refit_records, dataset=dataset, transforms=refit_transforms
    )
    del refit_records, train_records, valid_records
    gc.collect()

    test_records, test_meta, test_hit = _load_graphs(config, dataset, "test")
    split_sizes["test"] = int(len(test_records))
    split_positive["test"] = int(sum(float(record.y) > 0.5 for record in test_records))
    refit_test_data = _encode_records(
        test_records, dataset=dataset, transforms=refit_transforms
    )
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        spec=spec,
        transforms=refit_transforms,
        config=config,
        seed=seed,
        task=task,
        select_best=False,
        epochs=selected_epoch,
    )
    test_metric = float(refit_phase["metric"])
    del refit_phase["model"], refit_train_data, refit_test_data
    gc.collect()
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(_resolve(config["data"]["root"])),
            "split": (
                "PyG ZINC subset=True official train/valid/test"
                if dataset == "zinc"
                else "OGB ogbg-molhiv official scaffold train/valid/test"
            ),
            "sizes": split_sizes,
            "positive": (
                None
                if dataset == "zinc"
                else split_positive
            ),
            "record_cache_hits": {
                "train": bool(train_hit),
                "valid": bool(valid_hit),
                "test": bool(test_hit),
            },
            "test_labels_used_for_selection": False,
            "representation_fit_scope": "official train only for valid; official train+valid for terminal test refit",
        },
        "representation": {
            "structure_input_width": int(spec.structure_width),
            "attribute_input_width": int(spec.attribute_width),
            "relation_input_width": int(spec.relation_width),
            "context_input_width": int(spec.context_width),
            "distance_buckets": int(spec.distance_buckets),
            "typed_mode": (
                "exact rooted typed certificate embedding"
                if valid_transform_meta["typed_mode"] == "exact rooted typed certificate"
                else "OOV-only token table"
            ),
            "backoff_mode": (
                "exact radius-1 parent certificate embedding"
                if valid_transform_meta["backoff_mode"]
                == "exact radius-1 parent certificate for MolHIV"
                else (
                    "coarse invariant descriptor embedding"
                    if valid_transform_meta["backoff_mode"]
                    == "coarse invariant descriptor key"
                    else "OOV-only token table"
                )
            ),
            "readout": "patch/pair MLP before sum + sum-of-squares + max + log-count invariant pooling",
            "message_passing": False,
            "attention": False,
            "ksvd": False,
            "one_predictive_head": True,
            "label_free_representation": True,
        },
        "transforms": {
            "valid_train_only": valid_transform_meta,
            "test_refit_train_valid": refit_transform_meta,
        },
        "training": {
            **dict(config["model"]),
            "loss": (
                "L1 / mean absolute error"
                if dataset == "zinc"
                else "balanced binary cross entropy with logits"
            ),
            "metric": "MAE" if dataset == "zinc" else "ROC-AUC",
            "device": str(config["model"].get("device", "cpu")),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "metric": valid_metric,
                "best_metric": valid_metric,
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "metric": test_metric,
                "epochs_run": selected_epoch,
            },
            "parameters": parameters,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "torch_geometric": importlib.metadata.version("torch-geometric"),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
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
    key = "mae" if result["task"] == "zinc" else "auc"
    print(
        json.dumps(
            {
                "valid_" + key: result["evaluation"]["valid"]["metric"],
                "test_" + key: result["evaluation"]["test_after_train_valid_refit"]["metric"],
                "selected_epoch": result["evaluation"]["valid"]["selected_epoch"],
                "parameters": result["evaluation"]["parameters"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
