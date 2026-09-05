"""ZINC exact rooted-patch K-SVD with explicit pair pooling.

This is the compressed version of the exact ZINC patch-path model.  Each atom
centre is represented by the invariant 840-dimensional canonical descriptor
from ``zinc_exact_patch_relation``.  A train-only K-SVD dictionary converts
that descriptor to a fixed-width sparse code; the code is then consumed by a
single end-to-end MLP head together with explicit shortest-path pair
relations.

The important distinction from the earlier shell-K-SVD experiment is that
the dictionary is learned on the complete canonical patch descriptor, not on
an already marginalised shell histogram.  No message passing, attention, or
second predictive model is used.  The dictionary and all standardisers are
fit on official train for validation and on train+validation for the terminal
test refit; test labels never affect selection.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import pickle
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

from ksvd_research.core.ksvd import ksvd

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as base
from tracks.ksvd.experiments.luyin16.molhiv_ksvd_patch_path_pooling import (
    StreamingStandardizer,
    _batch_omp,
    _sample_dictionary_matrix,
)
from tracks.ksvd.experiments.luyin16.zinc_exact_patch_relation import (
    PATCH_WIDTH,
    _canonical_typed_patch,
    _patch_cache_key,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/zinc_ksvd_patch_path_pooling.yaml"
)

PATCH_RADIUS = base.PATCH_RADIUS
DISTANCE_BUCKETS = base.DISTANCE_BUCKETS
RELATION_WIDTH = base.RELATION_WIDTH
GLOBAL_WIDTH = base.GLOBAL_WIDTH


@dataclass(frozen=True)
class ExactGraphRecord:
    """A graph record whose ``shell_descriptor`` stores the exact descriptor."""

    patches: tuple[base.PatchRecord, ...]
    pair_index: np.ndarray
    pair_relation: np.ndarray
    pair_bucket: np.ndarray
    global_context: np.ndarray
    y: float


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _exact_graph_record(
    data: Any,
    global_context: np.ndarray,
    certificate_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]],
) -> ExactGraphRecord:
    graph, node_types, edge_types = base._data_to_graph(data)
    centres = list(graph.nodes)
    patches: list[base.PatchRecord] = []
    for centre in centres:
        distances = base._ego_distances(graph, int(centre), PATCH_RADIUS)
        cache_key = _patch_cache_key(
            graph, int(centre), node_types, edge_types, PATCH_RADIUS
        )
        patch = certificate_cache.get(cache_key)
        if patch is None:
            descriptor, nodes, certificate, metadata = _canonical_typed_patch(
                graph,
                int(centre),
                node_types,
                edge_types,
                radius=PATCH_RADIUS,
            )
            patch = (
                np.asarray(descriptor, dtype=np.float32),
                frozenset(int(node) for node in nodes),
                bytes(certificate),
                dict(metadata),
            )
            certificate_cache[cache_key] = patch
        descriptor, nodes, certificate, _metadata = patch
        boundary = frozenset(
            int(node)
            for node, distance in distances.items()
            if int(distance) == PATCH_RADIUS
        )
        patches.append(
            base.PatchRecord(
                typed_certificate=certificate,
                parent_certificate=b"",
                nodes=nodes,
                boundary=boundary,
                shell_descriptor=descriptor,
            )
        )

    shortest_paths = {
        int(source): base._shortest_path_summary(graph, int(source), edge_types)
        for source in centres
    }
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_relations: list[np.ndarray] = []
    pair_buckets: list[int] = []
    for left_index, left_centre in enumerate(centres):
        for right_index in range(left_index + 1, len(centres)):
            right_centre = int(centres[right_index])
            path_summary = shortest_paths[int(left_centre)].get(right_centre)
            distance = int(path_summary[0]) if path_summary is not None else 0
            adjacent_bond = (
                edge_types[graph.edge_key(int(left_centre), right_centre)]
                if distance == 1
                else None
            )
            relation, bucket = base._pair_relation(
                patches[left_index],
                patches[right_index],
                path_summary,
                adjacent_bond,
                PATCH_RADIUS,
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_relations.append(relation)
            pair_buckets.append(int(bucket))

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    pair_relation = (
        np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
        if pair_relations
        else np.zeros((0, RELATION_WIDTH), dtype=np.float32)
    )
    pair_bucket = np.asarray(pair_buckets, dtype=np.int64)
    if pair_relation.shape != (len(pair_sources), RELATION_WIDTH):
        raise RuntimeError(f"pair relation shape changed: {pair_relation.shape}")
    return ExactGraphRecord(
        patches=tuple(patches),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=np.asarray(global_context, dtype=np.float32),
        y=float(data.y.view(-1)[0]),
    )


def _extract_split(
    dataset: Any,
    split: str,
    certificate_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]],
) -> tuple[list[ExactGraphRecord], dict[str, Any]]:
    started = time.perf_counter()
    contexts = base.global_feature_views(dataset)["global_all"]
    records: list[ExactGraphRecord] = []
    for index, data in enumerate(dataset):
        records.append(_exact_graph_record(data, contexts[index], certificate_cache))
        if index and index % 500 == 0:
            print(
                f"ZINC exact K-SVD patch features {split}: {index}/{len(dataset)} "
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
        "seconds": float(time.perf_counter() - started),
    }
    return records, metadata


def _record_cache_signature(
    root: Path,
    split: str,
    dataset_size: int,
) -> str:
    payload = {
        "schema": "zinc-exact-ksvd-record-cache-v1",
        "split": str(split),
        "dataset_size": int(dataset_size),
        "radius": int(PATCH_RADIUS),
        "patch_width": int(PATCH_WIDTH),
        "exact_module": _sha256(
            Path(__file__).resolve().parent / "zinc_exact_patch_relation.py"
        ),
        "base_module": _sha256(Path(__file__).resolve().parent / "zinc_patch_path_pooling.py"),
        "raw_files": base.source_audit(root).get("raw_files", []),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_record_cache(
    path: Path,
    *,
    signature: str,
) -> tuple[list[ExactGraphRecord], dict[str, Any]]:
    # When this runner is invoked with ``python -m``, pickle may record the
    # dataclass under ``__main__``.  Register the canonical class as a
    # compatibility alias before loading caches produced by that invocation.
    # The cache contains no executable objects beyond these data records.
    import __main__

    if not hasattr(__main__, "ExactGraphRecord"):
        setattr(__main__, "ExactGraphRecord", ExactGraphRecord)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"record cache is not a mapping: {path}")
    if payload.get("signature") != signature:
        raise ValueError(f"record cache signature mismatch: {path}")
    if payload.get("schema") != "zinc-exact-ksvd-record-cache-v1":
        raise ValueError(f"record cache schema mismatch: {path}")
    records = payload.get("records")
    metadata = payload.get("metadata")
    if not isinstance(records, list) or not isinstance(metadata, Mapping):
        raise ValueError(f"record cache payload is malformed: {path}")
    return records, dict(metadata)


def _save_record_cache(
    path: Path,
    *,
    signature: str,
    records: Sequence[ExactGraphRecord],
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "zinc-exact-ksvd-record-cache-v1",
        "signature": signature,
        "records": list(records),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _extract_or_load_split(
    root: Path,
    dataset: Any,
    split: str,
    cache_dir: Path | None,
    certificate_cache: dict[bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]],
) -> tuple[list[ExactGraphRecord], dict[str, Any], bool]:
    signature = _record_cache_signature(root, split, len(dataset))
    if cache_dir is None:
        records, metadata = _extract_split(dataset, split, certificate_cache)
        return records, metadata, False
    path = cache_dir / f"exact_{split}.pkl"
    if path.exists():
        records, metadata = _load_record_cache(path, signature=signature)
        print(f"ZINC exact K-SVD record cache hit: {path}", flush=True)
        return records, metadata, True
    records, metadata = _extract_split(dataset, split, certificate_cache)
    _save_record_cache(path, signature=signature, records=records, metadata=metadata)
    print(f"ZINC exact K-SVD record cache saved: {path}", flush=True)
    return records, metadata, False


def _context_matrix(records: Sequence[ExactGraphRecord]) -> np.ndarray:
    return np.stack([record.global_context for record in records], axis=0).astype(
        np.float32, copy=False
    )


def _encode_records(
    records: Sequence[ExactGraphRecord],
    patch_standardizer: StreamingStandardizer,
    context_standardizer: base.Standardizer,
    dictionary: np.ndarray,
    sparsity: int,
    *,
    block_records: int = 256,
) -> list[Data]:
    output: list[Data] = []
    dictionary64 = np.asarray(dictionary, dtype=np.float64)
    gram = dictionary64.T @ dictionary64
    for block_start in range(0, len(records), int(block_records)):
        block = records[block_start : block_start + int(block_records)]
        patch_values_by_record: list[np.ndarray] = []
        offsets = [0]
        for record in block:
            values = patch_standardizer.transform(
                np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
            )
            patch_values_by_record.append(values)
            offsets.append(offsets[-1] + int(values.shape[0]))
        packed = np.concatenate(patch_values_by_record, axis=0)
        packed_code, packed_error = _batch_omp(
            packed,
            dictionary64,
            sparsity,
            gram=gram,
        )
        for local, record in enumerate(block):
            start = int(offsets[local])
            stop = int(offsets[local + 1])
            patch_values = patch_values_by_record[local]
            patch_aux = np.stack(
                [
                    packed_error[start:stop],
                    np.log1p(np.linalg.norm(patch_values, axis=1)).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            output.append(
                Data(
                    patch_code=torch.from_numpy(packed_code[start:stop]),
                    patch_aux=torch.from_numpy(patch_aux),
                    pair_index=torch.from_numpy(record.pair_index),
                    pair_relation=torch.from_numpy(record.pair_relation),
                    pair_bucket=torch.from_numpy(record.pair_bucket),
                    global_context=torch.from_numpy(
                        context_standardizer.transform(record.global_context[None, :])
                    ),
                    y=torch.tensor([record.y], dtype=torch.float32),
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class CompressedExactPatchPathModel(nn.Module):
    def __init__(
        self,
        *,
        code_width: int,
        patch_hidden: int,
        pair_hidden: int,
        dropout: float,
        readout: str = "moments",
        node_readout: str | None = None,
        pair_readout: str | None = None,
    ) -> None:
        super().__init__()
        self.patch_hidden = int(patch_hidden)
        self.pair_hidden = int(pair_hidden)
        self.readout = str(readout)
        self.node_readout = str(node_readout or self.readout)
        self.pair_readout = str(pair_readout or self.readout)
        valid = {"moments", "mean_std", "sum_mean_std", "distribution"}
        if self.node_readout not in valid or self.pair_readout not in valid:
            raise ValueError(
                "unknown readout; expected moments, mean_std, sum_mean_std, or distribution"
            )

        self.patch_encoder = _MLPBlock(
            2 * int(code_width) + 2,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = _MLPBlock(
            GLOBAL_WIDTH,
            max(int(patch_hidden // 2), 32),
            32,
            float(dropout),
        )
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = _MLPBlock(
            RELATION_WIDTH,
            max(int(pair_hidden), 32),
            int(pair_hidden),
            float(dropout),
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = _MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        pooled_unary = self._pooled_width(int(patch_hidden), self.node_readout)
        pooled_pair = self._pooled_width(int(pair_hidden), self.pair_readout)
        readout_width = pooled_unary + DISTANCE_BUCKETS * pooled_pair
        head_width = max(int(patch_hidden) * 2, 96)
        self.head = nn.Sequential(
            nn.Linear(readout_width + 32, head_width),
            nn.LayerNorm(head_width),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_width, int(patch_hidden)),
            nn.ReLU(),
            nn.Linear(int(patch_hidden), 1),
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
        total = torch.zeros(
            (n_graphs, value.shape[1]), device=value.device, dtype=value.dtype
        )
        total.index_add_(0, batch, value)
        counts = torch.bincount(batch, minlength=n_graphs).to(value.dtype).unsqueeze(1)
        squared = torch.zeros_like(total)
        squared.index_add_(0, batch, value * value)
        if mode == "moments":
            result = torch.cat([total, squared, torch.log1p(counts)], dim=1)
        elif mode == "mean_std":
            mean = total / counts.clamp_min(1.0)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            result = torch.cat([mean, torch.sqrt(variance + 1.0e-8), torch.log1p(counts)], dim=1)
        elif mode == "sum_mean_std":
            mean = total / counts.clamp_min(1.0)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            result = torch.cat(
                [total, mean, torch.sqrt(variance + 1.0e-8), torch.log1p(counts)], dim=1
            )
        elif mode == "distribution":
            mean = total / counts.clamp_min(1.0)
            variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
            maximum = self._grouped_max(value, batch, n_graphs)
            result = torch.cat(
                [
                    total,
                    mean,
                    torch.sqrt(variance + 1.0e-8),
                    maximum,
                    torch.log1p(counts),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"unknown pooling mode {mode!r}")
        expected = self._pooled_width(value.shape[1], mode)
        if result.shape[1] != expected:
            raise RuntimeError(f"pooled width changed: {result.shape[1]} != {expected}")
        return result

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
            total = torch.zeros(
                (n_graphs, value.shape[1]), device=value.device, dtype=value.dtype
            )
            counts = torch.zeros((n_graphs, 1), device=value.device, dtype=value.dtype)
            if current.numel():
                total.index_add_(0, current_batch, current)
                counts.index_add_(
                    0,
                    current_batch,
                    torch.ones(
                        (current_batch.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                )
            squared = torch.zeros_like(total)
            if current.numel():
                squared.index_add_(0, current_batch, current * current)
            if self.pair_readout == "moments":
                pooled = torch.cat([total, squared, torch.log1p(counts)], dim=1)
            elif self.pair_readout == "mean_std":
                mean = total / counts.clamp_min(1.0)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                pooled = torch.cat(
                    [mean, torch.sqrt(variance + 1.0e-8), torch.log1p(counts)], dim=1
                )
            elif self.pair_readout == "sum_mean_std":
                mean = total / counts.clamp_min(1.0)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                pooled = torch.cat(
                    [
                        total,
                        mean,
                        torch.sqrt(variance + 1.0e-8),
                        torch.log1p(counts),
                    ],
                    dim=1,
                )
            elif self.pair_readout == "distribution":
                mean = total / counts.clamp_min(1.0)
                variance = (squared / counts.clamp_min(1.0) - mean * mean).clamp_min(0.0)
                maximum = self._grouped_max(current, current_batch, n_graphs)
                pooled = torch.cat(
                    [
                        total,
                        mean,
                        torch.sqrt(variance + 1.0e-8),
                        maximum,
                        torch.log1p(counts),
                    ],
                    dim=1,
                )
            else:
                raise ValueError(f"unknown pair readout {self.pair_readout!r}")
            expected = self._pooled_width(value.shape[1], self.pair_readout)
            if pooled.shape[1] != expected:
                raise RuntimeError(f"pair pooled width changed: {pooled.shape[1]} != {expected}")
            blocks.append(pooled)
        return torch.cat(blocks, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        global_context = data.global_context
        if global_context.ndim == 1:
            global_context = global_context.unsqueeze(0)
        n_graphs = int(global_context.shape[0])
        patch_input = torch.cat(
            [data.patch_code, torch.abs(data.patch_code), data.patch_aux], dim=1
        )
        patch = self.patch_encoder(patch_input)
        unary = self._pool_values(
            value=patch,
            batch=data.batch,
            n_graphs=n_graphs,
            mode=self.node_readout,
        )

        source = data.pair_index[0]
        target = data.pair_index[1]
        projected_left = self.pair_projection(patch[source])
        projected_right = self.pair_projection(patch[target])
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket))
        pair_input = torch.cat(
            [
                projected_left + projected_right,
                torch.abs(projected_left - projected_right),
                projected_left * projected_right * gate,
                relation,
            ],
            dim=1,
        )
        pair_value = self.pair_encoder(pair_input)
        pair_batch = data.batch[source]
        relation_readout = self._pool_pairs(
            pair_value, pair_batch, data.pair_bucket, n_graphs
        )
        graph_hidden = self.global_encoder(global_context)
        return self.head(torch.cat([unary, relation_readout, graph_hidden], dim=1)).view(-1)


def _make_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return float(
        mean_absolute_error(
            np.concatenate(targets).astype(np.float64),
            np.concatenate(predictions).astype(np.float64),
        )
    )


def _train_phase(
    train_graphs: Sequence[Data],
    eval_graphs: Sequence[Data],
    *,
    code_width: int,
    config: Mapping[str, Any],
    seed: int,
    select_best: bool,
    epochs: int,
) -> dict[str, Any]:
    model_config = config["model"]
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")
    _seed_everything(seed)
    model = CompressedExactPatchPathModel(
        code_width=int(code_width),
        patch_hidden=int(model_config.get("patch_hidden", 64)),
        pair_hidden=int(model_config.get("pair_hidden", 32)),
        dropout=float(model_config.get("dropout", 0.05)),
        readout=str(model_config.get("readout", "moments")),
        node_readout=model_config.get("node_readout"),
        pair_readout=model_config.get("pair_readout"),
    ).to(device)
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
            loss = F.l1_loss(prediction, target)
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
            print(
                f"zinc exact K-SVD patch-path phase={'select' if select_best else 'refit'} "
                f"epoch={epoch:03d}/{epochs} l1={epoch_loss:.6f}{suffix}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"zinc exact K-SVD patch-path early_stop epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_mae = _evaluate(model, eval_loader, device)
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
    }


def _fit_phase_data(
    fit_records: Sequence[ExactGraphRecord],
    other_records: Sequence[ExactGraphRecord],
    *,
    representation: Mapping[str, Any],
    seed: int,
) -> tuple[list[Data], list[Data], dict[str, Any]]:
    patch_standardizer = StreamingStandardizer.fit_patch_records(fit_records)
    context_standardizer = base.Standardizer.fit(_context_matrix(fit_records))
    dictionary_matrix = _sample_dictionary_matrix(
        fit_records,
        patch_standardizer,
        int(representation.get("max_dictionary_patches", 12000)),
        seed + 1729,
    )
    dictionary_mode = str(representation.get("dictionary_mode", "final"))
    if dictionary_mode not in {"init", "final"}:
        raise ValueError(
            f"unknown dictionary_mode={dictionary_mode!r}; expected init or final"
        )
    started = time.perf_counter()
    dictionary, _, dictionary_info = ksvd(
        dictionary_matrix,
        n_atoms=int(representation.get("n_atoms", 64)),
        T=int(representation.get("sparsity", 4)),
        T_min=1,
        n_iter=(
            0
            if dictionary_mode == "init"
            else int(representation.get("ksvd_iter", 3))
        ),
        seed=seed + 2718,
    )
    dictionary_seconds = time.perf_counter() - started
    encoded_fit = _encode_records(
        fit_records,
        patch_standardizer,
        context_standardizer,
        dictionary,
        int(representation.get("sparsity", 4)),
    )
    encoded_other = _encode_records(
        other_records,
        patch_standardizer,
        context_standardizer,
        dictionary,
        int(representation.get("sparsity", 4)),
    )
    audit = {
        "patch_width": int(dictionary_matrix.shape[0]),
        "descriptor_width": int(dictionary_matrix.shape[0]),
        "descriptor": "exact_canonical",
        "dictionary_samples": int(dictionary_matrix.shape[1]),
        "dictionary": dictionary_info,
        "dictionary_seconds": float(dictionary_seconds),
        "code_width": int(dictionary.shape[1]),
        "sparsity": int(representation.get("sparsity", 4)),
        "patch_aux_width": 2,
        "standardizer_fit_graphs": int(len(fit_records)),
        "dictionary_mode": dictionary_mode,
    }
    return encoded_fit, encoded_other, audit


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    representation = result["representation"]
    return "\n".join(
        [
            f"# {result['protocol_id']}",
            "",
            "Exact invariant rooted patch descriptor compressed by train-only K-SVD; explicit shortest-path pair pooling and one MLP, no message passing.",
            "",
            f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
            f"- patch descriptor: `{representation['patch_width']}D exact canonical`; dictionary: `K={representation['dictionary_atoms']}`, `T={representation['sparsity']}`",
            f"- pair relation: `{representation['relation_width']}D`; readout: `{representation['readout']}`",
            f"- message passing: `{representation['message_passing']}`; attention: `{representation['attention']}`",
            "",
            "| head | valid MAE | test MAE after train+valid refit | selected epoch |",
            "|---|---:|---:|---:|",
            f"| `single_mlp` | {evaluation['valid']['mae']:.6f} | {evaluation['test_after_train_valid_refit']['mae']:.6f} | {evaluation['valid']['selected_epoch']} |",
            "",
            f"- trainable parameters: `{evaluation['parameters']}`",
            f"- total parameters including frozen dictionary: `{evaluation['parameters_including_dictionary']}`",
            f"- dictionary `{representation['dictionary_mode']}` relative reconstruction: `{representation['dictionary_recon_rel']:.6f}`",
            f"- runtime: `{result['runtime']['seconds']:.1f}s`",
            "",
        ]
    )


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    model_config = config["model"]
    representation = config["representation"]
    seed = int(config.get("seed", 0))
    started = time.perf_counter()
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))
    data_root = _resolve(data_config["root"])
    cache_value = config.get("runtime", {}).get("record_cache")
    cache_dir = None if cache_value in (None, "", False) else _resolve(str(cache_value))

    datasets = tuple(base._load_zinc(data_root, split) for split in ("train", "val", "test"))
    split_names = ("train", "valid", "test")
    certificate_cache: dict[
        bytes, tuple[np.ndarray, frozenset[int], bytes, dict[str, int]]
    ] = {}
    records: dict[str, list[ExactGraphRecord]] = {}
    feature_metadata: dict[str, Any] = {}
    cache_hits: dict[str, bool] = {}
    for name, dataset in zip(split_names, datasets, strict=True):
        records[name], feature_metadata[name], cache_hits[name] = _extract_or_load_split(
            data_root,
            dataset,
            name,
            cache_dir,
            certificate_cache,
        )
    feature_metadata["certificate_cache_entries"] = int(len(certificate_cache))

    train_records = records["train"]
    valid_records = records["valid"]
    test_records = records["test"]
    valid_train_data, valid_eval_data, valid_audit = _fit_phase_data(
        train_records,
        valid_records,
        representation=representation,
        seed=seed,
    )
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        code_width=int(valid_audit["code_width"]),
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 50)),
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    valid_mae = float(valid_phase["mae"])
    valid_parameters = int(valid_phase["parameters"])
    valid_dictionary_info = valid_audit["dictionary"]
    del valid_train_data, valid_eval_data, valid_phase["model"]
    gc.collect()

    refit_train_records = list(train_records) + list(valid_records)
    refit_train_data, refit_test_data, test_audit = _fit_phase_data(
        refit_train_records,
        test_records,
        representation=representation,
        seed=seed,
    )
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        code_width=int(test_audit["code_width"]),
        config=config,
        seed=seed,
        select_best=False,
        epochs=selected_epoch,
    )

    dictionary_atoms = int(valid_audit["code_width"])
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                name: len(dataset)
                for name, dataset in zip(split_names, datasets, strict=True)
            },
            "source": base.source_audit(data_root),
            "test_labels_used_for_selection": False,
            "record_cache": None if cache_dir is None else str(cache_dir),
            "record_cache_hits": cache_hits,
        },
        "representation": {
            "radius": PATCH_RADIUS,
            "centres": "every atom",
            "descriptor": "exact canonical rooted typed patch descriptor",
            "patch_width": int(valid_audit["descriptor_width"]),
            "dictionary_atoms": dictionary_atoms,
            "sparsity": int(representation.get("sparsity", 4)),
            "relation_width": RELATION_WIDTH,
            "global_width": GLOBAL_WIDTH,
            "distance_buckets": ["1", "2", "3", "4", "5+"],
            "dictionary_fit": "train-only for valid; train+valid for test refit",
            "dictionary_mode": str(valid_audit["dictionary_mode"]),
            "dictionary_recon_rel": float(valid_dictionary_info["recon_rel"]),
            "readout": str(model_config.get("readout", "moments")),
            "node_readout": str(
                model_config.get("node_readout", model_config.get("readout", "moments"))
            ),
            "pair_readout": str(
                model_config.get("pair_readout", model_config.get("readout", "moments"))
            ),
            "message_passing": False,
            "attention": False,
            "label_free": True,
        },
        "feature_build": feature_metadata,
        "dictionary_audit": {
            "valid_train": valid_audit,
            "test_after_train_valid_refit": test_audit,
        },
        "training": {
            **dict(model_config),
            "loss": "L1 / mean absolute error",
            "device": str(torch.device(str(model_config.get("device", "cpu")))),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "mae": valid_mae,
                "best_mae": float(valid_phase["best_mae"]),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "mae": float(refit_phase["mae"]),
                "epochs_run": int(refit_phase["epochs_run"]),
            },
            "parameters": valid_parameters,
            "parameters_including_dictionary": int(
                valid_parameters + int(valid_audit["descriptor_width"]) * dictionary_atoms
            ),
        },
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
    print(
        json.dumps(
            {
                "valid_mae": result["evaluation"]["valid"]["mae"],
                "test_mae": result["evaluation"]["test_after_train_valid_refit"]["mae"],
                "selected_epoch": result["evaluation"]["valid"]["selected_epoch"],
                "parameters": result["evaluation"]["parameters"],
                "parameters_including_dictionary": result["evaluation"]["parameters_including_dictionary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
