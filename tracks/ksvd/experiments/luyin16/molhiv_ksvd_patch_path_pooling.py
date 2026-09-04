"""MolHIV explicit patch--path pooling with a frozen K-SVD patch code.

This is the compact follow-up to ``molhiv_patch_path_pooling``.  It keeps the
same all-centre radius-2 shell descriptor, explicit shortest-path pair
relation, invariant pooling and one MLP head, but removes the large table of
independent exact-patch embeddings.  A train-only K-SVD dictionary converts a
standardized patch descriptor into a fixed-width sparse code.  The dictionary
is refit on train+valid only for the terminal test evaluation.  The runner
also supports a mean/std or mean/std/max distribution readout and a label-free
record cache so those alternatives can be compared without rebuilding graphs.

The first implementation deliberately does not use the exact certificate:
K-SVD is applied to a continuous, permutation-invariant descriptor rather
than to an arbitrary one-hot patch ID.  This is a parameter-budget and
generalization experiment, not a claim that the shell descriptor preserves
all internal patch topology.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
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
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ksvd_research.core.ksvd import ksvd
from ksvd_research.data import load_molhiv

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as base


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/molhiv_ksvd_patch_path_pooling.yaml"

# ``structured_v1`` adds two explicit, local relations to the shell
# marginal descriptor:
#
#   * endpoint-distance-conditioned ordered two-walk bond relations from the
#     centre (2 * 13 * 13 = 338 coordinates);
#   * shell-pair-conditioned endpoint atomic-number mass (6 * 119 = 714).
#
# The descriptor is still permutation invariant, but unlike a shell histogram
# it records which bond sequence reaches a second-shell node and which atom
# types are joined by an induced local edge.  With K=64 this fits below the
# CIN-sized total-parameter budget together with the current head.
STRUCTURED_PATH_WIDTH = 2 * base.BOND_WIDTH * base.BOND_WIDTH
STRUCTURED_EDGE_ATOM_WIDTH = len(base.SHELL_PAIRS) * base.ATOM_FEATURE_DIMS[0]
STRUCTURED_SHELL_WIDTH = (
    base.SHELL_WIDTH + STRUCTURED_PATH_WIDTH + STRUCTURED_EDGE_ATOM_WIDTH
)


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
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _structured_shell_descriptor(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    distances: Mapping[int, int],
) -> tuple[np.ndarray, frozenset[int], frozenset[int]]:
    """Return the base shell descriptor plus explicit local relations.

    The two-walk block enumerates ``center -> middle -> endpoint`` paths in
    the induced radius-2 ego.  Endpoints at distance 1 mark local closures
    (e.g. triangles), while endpoints at distance 2 mark genuine second-shell
    paths.  The ordered bond-feature outer product retains the direction from
    the root and therefore does not collapse a two-bond sequence into a
    single bond marginal.

    The edge block records endpoint atomic-number mass for each shell pair.
    Both additions are sums over unordered/locally rooted objects and are
    independent of the input node numbering.
    """
    base_descriptor, nodes, boundary = base._shell_descriptor(
        graph, int(center), node_types, edge_types, distances
    )

    path_block = np.zeros((2, base.BOND_WIDTH * base.BOND_WIDTH), dtype=np.float32)
    path_counts = np.zeros(2, dtype=np.float32)
    for middle in sorted(graph.neighbors(int(center))):
        middle = int(middle)
        first_bond = base._feature_one_hot(
            edge_types[graph.edge_key(int(center), middle)], base.BOND_FEATURE_DIMS
        )
        for endpoint in sorted(graph.neighbors(middle)):
            endpoint = int(endpoint)
            endpoint_distance = int(distances.get(endpoint, -1))
            if endpoint_distance not in (1, 2):
                continue
            second_bond = base._feature_one_hot(
                edge_types[graph.edge_key(middle, endpoint)], base.BOND_FEATURE_DIMS
            )
            path_block[endpoint_distance - 1] += np.outer(
                first_bond, second_bond
            ).reshape(-1)
            path_counts[endpoint_distance - 1] += 1.0
    for index in range(2):
        if path_counts[index] > 0.0:
            path_block[index] /= path_counts[index]

    edge_atom_block = np.zeros(
        (len(base.SHELL_PAIRS), base.ATOM_FEATURE_DIMS[0]), dtype=np.float32
    )
    shell_pair_index = {pair: index for index, pair in enumerate(base.SHELL_PAIRS)}
    edge_count_by_shell = np.zeros(len(base.SHELL_PAIRS), dtype=np.float32)
    induced = graph.induced(set(nodes))
    for left, right in induced.edges():
        shell_pair = tuple(sorted((int(distances[left]), int(distances[right]))))
        shell_index = shell_pair_index[shell_pair]
        left_atomic_number = int(node_types[int(left), 0])
        right_atomic_number = int(node_types[int(right), 0])
        edge_atom_block[shell_index, left_atomic_number] += 1.0
        edge_atom_block[shell_index, right_atomic_number] += 1.0
        edge_count_by_shell[shell_index] += 2.0
    for index in range(len(base.SHELL_PAIRS)):
        if edge_count_by_shell[index] > 0.0:
            edge_atom_block[index] /= edge_count_by_shell[index]

    descriptor = np.concatenate(
        [base_descriptor, path_block.reshape(-1), edge_atom_block.reshape(-1)]
    ).astype(np.float32, copy=False)
    if descriptor.shape != (STRUCTURED_SHELL_WIDTH,):
        raise RuntimeError(
            f"structured descriptor width changed: {descriptor.shape}; "
            f"expected {(STRUCTURED_SHELL_WIDTH,)}"
        )
    return descriptor, nodes, boundary


def _relabel_graph_features(
    graph: Any,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    permutation: np.ndarray,
) -> tuple[Any, np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Relabel a graph for the local descriptor invariance audit."""
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(graph.n)):
        raise ValueError("permutation must contain every node id exactly once")
    from ksvd_research.core import from_edges

    changed_graph = from_edges(
        graph.n,
        [
            (int(permutation[left]), int(permutation[right]))
            for left, right in graph.edges()
        ],
    )
    changed_nodes = np.empty_like(node_features)
    changed_nodes[permutation] = node_features
    changed_edges: dict[tuple[int, int], np.ndarray] = {}
    for (left, right), values in edge_features.items():
        mapped = (int(permutation[left]), int(permutation[right]))
        key = mapped if mapped[0] < mapped[1] else (mapped[1], mapped[0])
        changed_edges[key] = np.asarray(values).copy()
    return changed_graph, changed_nodes, changed_edges


def audit_descriptor_invariance(
    bundle: Any,
    indices: Sequence[int],
    descriptor_name: str,
    *,
    n_graphs: int = 32,
    permutations_per_graph: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Check that the structured descriptor is independent of node IDs."""
    rng = np.random.default_rng(int(seed))
    candidates = np.asarray(indices, dtype=np.int64)
    if candidates.size > int(n_graphs):
        candidates = rng.choice(candidates, size=int(n_graphs), replace=False)

    # Compare the row multisets lexicographically.  Sorting each coordinate
    # independently would be a weaker check because it could hide a
    # coordinate permutation between rows.
    def _sorted_rows(value: np.ndarray) -> np.ndarray:
        order = np.lexsort(
            tuple(value[:, column] for column in range(value.shape[1] - 1, -1, -1))
        )
        return value[order]

    drifts: list[float] = []
    for raw_index in candidates:
        index = int(raw_index)
        graph = bundle.graphs[index]
        node_features = np.asarray(bundle.node_feats[index], dtype=np.int64)
        edge_features = bundle.edge_feats[index]
        distances_by_centre = {
            int(centre): base._ego_distances(graph, int(centre), base.PATCH_RADIUS)
            for centre in graph.nodes
        }
        reference = np.stack(
            [
                (
                    base._shell_descriptor(
                        graph, int(centre), node_features, edge_features,
                        distances_by_centre[int(centre)]
                    )[0]
                    if descriptor_name == "shell"
                    else _structured_shell_descriptor(
                        graph, int(centre), node_features, edge_features,
                        distances_by_centre[int(centre)]
                    )[0]
                )
                for centre in graph.nodes
            ],
            axis=0,
        )
        for _ in range(int(permutations_per_graph)):
            permutation = rng.permutation(graph.n)
            changed_graph, changed_nodes, changed_edges = _relabel_graph_features(
                graph, node_features, edge_features, permutation
            )
            changed = np.stack(
                [
                    (
                        base._shell_descriptor(
                            changed_graph, int(centre), changed_nodes, changed_edges,
                            base._ego_distances(changed_graph, int(centre), base.PATCH_RADIUS)
                        )[0]
                        if descriptor_name == "shell"
                        else _structured_shell_descriptor(
                            changed_graph, int(centre), changed_nodes, changed_edges,
                            base._ego_distances(changed_graph, int(centre), base.PATCH_RADIUS)
                        )[0]
                    )
                    for centre in changed_graph.nodes
                ],
                axis=0,
            )
            reference_sorted = _sorted_rows(reference)
            changed_sorted = _sorted_rows(changed)
            drifts.append(float(np.max(np.abs(reference_sorted - changed_sorted))))
    return {
        "graphs": int(candidates.size),
        "permutations_per_graph": int(permutations_per_graph),
        "maximum_coordinate_drift_after_sorting": float(max(drifts, default=0.0)),
        "mean_coordinate_drift_after_sorting": float(np.mean(drifts) if drifts else 0.0),
    }


def _compositional_graph_record(
    graph: Any,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    y: float,
    descriptor_name: str,
) -> base.GraphRecord:
    """Build the same relation object without computing exact certificates."""
    centres = list(graph.nodes)
    patches: list[base.PatchRecord] = []
    for centre in centres:
        distances = base._ego_distances(graph, int(centre), base.PATCH_RADIUS)
        if descriptor_name == "shell":
            descriptor, nodes, boundary = base._shell_descriptor(
                graph, int(centre), node_types, edge_types, distances
            )
        elif descriptor_name == "structured_v1":
            descriptor, nodes, boundary = _structured_shell_descriptor(
                graph, int(centre), node_types, edge_types, distances
            )
        else:
            raise ValueError(f"unknown descriptor_name={descriptor_name!r}")
        # The base record is reused so pair construction and downstream code
        # remain byte-for-byte compatible with the existing relation protocol.
        patches.append(
            base.PatchRecord(
                typed_certificate=b"",
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
                patches[left_index], patches[right_index], path_summary, adjacent_bond
            )
            pair_sources.append(int(left_index))
            pair_targets.append(int(right_index))
            pair_relations.append(relation)
            pair_buckets.append(int(bucket))

    pair_index = np.asarray([pair_sources, pair_targets], dtype=np.int64)
    pair_relation = (
        np.stack(pair_relations, axis=0).astype(np.float32, copy=False)
        if pair_relations
        else np.zeros((0, base.RELATION_WIDTH), dtype=np.float32)
    )
    pair_bucket = np.asarray(pair_buckets, dtype=np.int64)
    return base.GraphRecord(
        patches=tuple(patches),
        pair_index=pair_index,
        pair_relation=pair_relation,
        pair_bucket=pair_bucket,
        global_context=base._global_context(graph, node_types, edge_types),
        y=float(y),
    )


def _extract_split(
    bundle: Any,
    indices: Sequence[int],
    split: str,
    descriptor_name: str,
) -> tuple[list[base.GraphRecord], dict[str, Any]]:
    started = time.perf_counter()
    records: list[base.GraphRecord] = []
    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        index_int = int(index)
        graph = bundle.graphs[index_int]
        node_types = np.asarray(bundle.node_feats[index_int], dtype=np.int64)
        edge_types = {
            (int(left), int(right)): tuple(int(value) for value in values)
            for (left, right), values in bundle.edge_feats[index_int].items()
        }
        records.append(
            _compositional_graph_record(
                graph,
                node_types,
                edge_types,
                float(bundle.y[index_int]),
                descriptor_name,
            )
        )
        if (position + 1) % 1000 == 0 or position + 1 == len(indices):
            print(
                f"MolHIV K-SVD patch-path features {split}: "
                f"{position + 1}/{len(indices)}",
                flush=True,
            )
    metadata = {
        "n_graphs": int(len(records)),
        "positive_graphs": int(sum(record.y > 0.5 for record in records)),
        "mean_centres": float(np.mean([len(record.patches) for record in records]))
        if records
        else 0.0,
        "mean_pairs": float(np.mean([record.pair_relation.shape[0] for record in records]))
        if records
        else 0.0,
        "mean_patch_nodes": float(
            np.mean([len(patch.nodes) for record in records for patch in record.patches])
        )
        if records
        else 0.0,
        "max_patch_nodes": int(
            max((len(patch.nodes) for record in records for patch in record.patches), default=0)
        ),
        "seconds": float(time.perf_counter() - started),
    }
    return records, metadata


def _record_cache_signature(
    bundle: Any,
    indices: Sequence[int],
    split: str,
    descriptor_name: str,
) -> str:
    """Return a dataset/split signature for the label-free record cache."""
    digest = hashlib.sha256()
    digest.update(b"molhiv-ksvd-record-cache-v1")
    digest.update(str(split).encode("utf-8"))
    digest.update(str(descriptor_name).encode("utf-8"))
    digest.update(str(bundle.meta.get("name", "ogbg-molhiv")).encode("utf-8"))
    digest.update(str(bundle.meta.get("n_full", "")).encode("ascii"))
    digest.update(str(bundle.meta.get("n_used", "")).encode("ascii"))
    # ``load_molhiv(max_graphs=...)`` remaps sampled graphs to contiguous
    # indices.  Include the original global indices so caches from two smoke
    # seeds cannot be mistaken for the same graph population.
    digest.update(
        np.asarray(bundle.meta.get("original_indices", []), dtype=np.int64).tobytes()
    )
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _load_record_cache(
    path: Path,
    *,
    signature: str,
    descriptor_name: str,
) -> tuple[list[base.GraphRecord], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"record cache is not a mapping: {path}")
    if payload.get("schema") != "molhiv-ksvd-record-cache-v1":
        raise ValueError(f"record cache schema mismatch: {path}")
    if payload.get("signature") != signature or payload.get("descriptor") != descriptor_name:
        raise ValueError(f"record cache signature mismatch: {path}")
    records = payload.get("records")
    metadata = payload.get("metadata")
    if not isinstance(records, list) or not isinstance(metadata, Mapping):
        raise ValueError(f"record cache payload is malformed: {path}")
    return records, dict(metadata)


def _save_record_cache(
    path: Path,
    *,
    signature: str,
    descriptor_name: str,
    records: Sequence[base.GraphRecord],
    metadata: Mapping[str, Any],
) -> None:
    """Persist feature records atomically; cache contains no learned weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "molhiv-ksvd-record-cache-v1",
        "signature": signature,
        "descriptor": descriptor_name,
        "records": list(records),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _extract_or_load_split(
    bundle: Any,
    indices: Sequence[int],
    split: str,
    descriptor_name: str,
    cache_dir: Path | None,
) -> tuple[list[base.GraphRecord], dict[str, Any], bool]:
    """Build records once and reuse them for candidate readout experiments."""
    if cache_dir is None:
        records, metadata = _extract_split(bundle, indices, split, descriptor_name)
        return records, metadata, False
    path = cache_dir / f"{descriptor_name}_{split}.pkl"
    signature = _record_cache_signature(bundle, indices, split, descriptor_name)
    if path.exists():
        records, metadata = _load_record_cache(
            path, signature=signature, descriptor_name=descriptor_name
        )
        print(f"MolHIV K-SVD record cache hit: {path}", flush=True)
        return records, metadata, True
    records, metadata = _extract_split(bundle, indices, split, descriptor_name)
    _save_record_cache(
        path,
        signature=signature,
        descriptor_name=descriptor_name,
        records=records,
        metadata=metadata,
    )
    print(f"MolHIV K-SVD record cache saved: {path}", flush=True)
    return records, metadata, False


class StreamingStandardizer:
    """Feature standardizer fitted without materializing all patch rows."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit_patch_records(cls, records: Sequence[base.GraphRecord]) -> "StreamingStandardizer":
        total = None
        squared = None
        count = 0
        for record in records:
            values = np.stack([patch.shell_descriptor for patch in record.patches], axis=0).astype(
                np.float64, copy=False
            )
            if total is None:
                total = np.zeros(values.shape[1], dtype=np.float64)
                squared = np.zeros(values.shape[1], dtype=np.float64)
            total += values.sum(axis=0, dtype=np.float64)
            squared += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
            count += int(values.shape[0])
        if total is None or squared is None or count == 0:
            raise ValueError("cannot standardize an empty patch collection")
        mean64 = total / float(count)
        variance = np.maximum(squared / float(count) - np.square(mean64), 0.0)
        scale = np.sqrt(variance)
        scale[~np.isfinite(scale) | (scale < 1.0e-6)] = 1.0
        return cls(mean64.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32, copy=False
        )
        if not np.isfinite(output).all():
            raise FloatingPointError("non-finite standardized patch values")
        return output


def _sample_dictionary_matrix(
    records: Sequence[base.GraphRecord],
    standardizer: StreamingStandardizer,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Deterministic reservoir sample, returned as (features, samples)."""
    rng = np.random.default_rng(int(seed))
    reservoir: list[np.ndarray] = []
    seen = 0
    limit = int(maximum)
    if limit < 2:
        raise ValueError("dictionary sample must contain at least two patches")
    for record in records:
        values = standardizer.transform(
            np.stack([patch.shell_descriptor for patch in record.patches], axis=0)
        )
        for row in values:
            if seen < limit:
                reservoir.append(row.copy())
            else:
                replacement = int(rng.integers(0, seen + 1))
                if replacement < limit:
                    reservoir[replacement] = row.copy()
            seen += 1
    if len(reservoir) < 2:
        raise RuntimeError("not enough patches for dictionary learning")
    return np.stack(reservoir, axis=1).astype(np.float64, copy=False)


def _batch_omp(
    values: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    *,
    chunk_size: int = 4096,
    gram: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch OMP using the small per-sample normal equations.

    ``values`` is (n_samples, n_features), dictionary is (n_features, K).
    The implementation avoids one Python-level OMP call per atom centre while
    retaining the least-squares support update used by the historical solver.
    """
    matrix = np.asarray(values, dtype=np.float64)
    dictionary64 = np.asarray(dictionary, dtype=np.float64)
    if matrix.ndim != 2 or dictionary64.ndim != 2 or matrix.shape[1] != dictionary64.shape[0]:
        raise ValueError("invalid batch OMP shapes")
    n_samples = int(matrix.shape[0])
    n_atoms = int(dictionary64.shape[1])
    steps = min(max(int(sparsity), 1), n_atoms)
    gram_matrix = dictionary64.T @ dictionary64 if gram is None else np.asarray(gram, dtype=np.float64)
    if gram_matrix.shape != (n_atoms, n_atoms):
        raise ValueError(f"invalid dictionary Gram shape {gram_matrix.shape}")
    codes = np.zeros((n_samples, n_atoms), dtype=np.float32)
    relative_errors = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, int(chunk_size)):
        stop = min(start + int(chunk_size), n_samples)
        current = matrix[start:stop]
        cross = current @ dictionary64
        support = np.full((stop - start, steps), -1, dtype=np.int64)
        selected = np.zeros((stop - start, n_atoms), dtype=bool)
        coefficients = np.zeros((stop - start, steps), dtype=np.float64)
        correlation = cross.copy()
        for step in range(steps):
            correlation[selected] = 0.0
            atom = np.argmax(np.abs(correlation), axis=1)
            support[:, step] = atom
            selected[np.arange(stop - start), atom] = True
            active = support[:, : step + 1]
            active_gram = gram_matrix[active[:, :, None], active[:, None, :]]
            active_cross = cross[np.arange(stop - start)[:, None], active]
            # A tiny diagonal keeps duplicate/near-duplicate dictionary
            # atoms from making one whole batch fail.  This is numerically
            # equivalent to the historical least-squares OMP for the well
            # conditioned supports encountered here.
            regularized = active_gram + 1.0e-8 * np.eye(step + 1)[None, :, :]
            active_coeff = np.linalg.solve(
                regularized,
                active_cross[..., None],
            )[..., 0]
            coefficients[:, : step + 1] = active_coeff
            correlation = cross - np.einsum(
                "ni,nij->nj",
                active_coeff,
                gram_matrix[active, :],
            )
        reconstructed = np.zeros_like(current)
        for step in range(steps):
            reconstructed += dictionary64[:, support[:, step]].T * coefficients[:, step, None]
            local_codes = codes[start:stop]
            local_codes[np.arange(stop - start), support[:, step]] = coefficients[:, step].astype(
                np.float32
            )
        residual = current - reconstructed
        relative_errors[start:stop] = (
            np.linalg.norm(residual, axis=1)
            / np.maximum(np.linalg.norm(current, axis=1), 1.0e-12)
        ).astype(np.float32)
    return codes, relative_errors


def _encode_records(
    records: Sequence[base.GraphRecord],
    patch_standardizer: StreamingStandardizer,
    context_standardizer: base.Standardizer,
    dictionary: np.ndarray,
    sparsity: int,
    block_records: int = 256,
) -> list[Data]:
    """Encode records in blocks so OMP shares one dictionary Gram matrix.

    The old implementation invoked ``_batch_omp`` once per graph.  That is
    mathematically fine, but it recomputed ``D.T @ D`` for every molecule.
    Packing a bounded number of graphs preserves graph boundaries while
    making the expensive matrix multiplication and OMP setup amortized.
    """
    output: list[Data] = []
    if int(block_records) < 1:
        raise ValueError("block_records must be positive")
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
            packed, dictionary64, sparsity, gram=gram
        )
        for local, record in enumerate(block):
            start = int(offsets[local])
            stop = int(offsets[local + 1])
            patch_values = patch_values_by_record[local]
            code = packed_code[start:stop]
            relative_error = packed_error[start:stop]
            patch_norm = np.log1p(np.linalg.norm(patch_values, axis=1)).astype(np.float32)
            patch_aux = np.stack([relative_error, patch_norm], axis=1).astype(
                np.float32, copy=False
            )
            output.append(
                Data(
                    patch_code=torch.from_numpy(code),
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


class CompressedPatchPathModel(nn.Module):
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
        valid_readouts = {"moments", "mean_std", "distribution", "sum_mean_std"}
        if self.node_readout not in valid_readouts or self.pair_readout not in valid_readouts:
            raise ValueError(
                "unknown readout; expected moments, mean_std, sum_mean_std, or distribution"
            )
        patch_input = 2 * int(code_width) + 2
        self.patch_encoder = base._MLPBlock(
            patch_input,
            max(int(patch_hidden), 64),
            int(patch_hidden),
            float(dropout),
        )
        self.global_encoder = base._MLPBlock(
            base.GLOBAL_WIDTH,
            max(int(patch_hidden // 2), 32),
            32,
            float(dropout),
        )
        self.pair_projection = nn.Linear(int(patch_hidden), int(pair_hidden), bias=False)
        self.relation_encoder = base._MLPBlock(
            base.RELATION_WIDTH,
            max(int(pair_hidden), 32),
            int(pair_hidden),
            float(dropout),
        )
        self.distance_gate = nn.Embedding(base.DISTANCE_BUCKETS, int(pair_hidden))
        self.pair_encoder = base._MLPBlock(
            4 * int(pair_hidden),
            max(2 * int(pair_hidden), 64),
            int(pair_hidden),
            float(dropout),
        )
        pooled_unary_width = self._pooled_width(int(patch_hidden), self.node_readout)
        pooled_pair_width = self._pooled_width(int(pair_hidden), self.pair_readout)
        readout_width = pooled_unary_width + base.DISTANCE_BUCKETS * pooled_pair_width
        self.head = nn.Sequential(
            nn.Linear(readout_width + 32, max(int(patch_hidden) * 2, 96)),
            nn.LayerNorm(max(int(patch_hidden) * 2, 96)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(int(patch_hidden) * 2, 96), int(patch_hidden)),
            nn.ReLU(),
            nn.Linear(int(patch_hidden), 1),
        )

    @staticmethod
    def _pooled_width(width: int, mode: str) -> int:
        if mode == "moments":
            return 2 * int(width) + 1
        if mode == "mean_std":
            return 2 * int(width) + 1
        if mode == "sum_mean_std":
            return 3 * int(width) + 1
        if mode == "distribution":
            # Keep graph mass while adding the centre-population mean/std and
            # a rare-instance-sensitive max.  This is the small differentiable
            # analogue of the successful clean distribution readout probe.
            return 4 * int(width) + 1
        raise ValueError(f"unknown pooling mode {mode!r}")

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
            raise RuntimeError(
                f"pooled width changed for mode={mode}: {output.shape[1]} != {expected}"
            )
        return output

    def _pool_nodes(self, value: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        return self._pool_values(value, batch, n_graphs, self.node_readout)

    @staticmethod
    def _grouped_max(
        value: torch.Tensor,
        group: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        """Differentiable grouped max without a Python loop over graphs."""
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

    def _pool_pairs(
        self,
        value: torch.Tensor,
        pair_batch: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_graphs: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for bucket in range(base.DISTANCE_BUCKETS):
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
                        (current_batch.shape[0], 1), device=value.device, dtype=value.dtype
                    ),
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
                raise ValueError(f"unknown pooling mode {self.readout!r}")
            expected = self._pooled_width(value.shape[1], self.pair_readout)
            if pooled.shape[1] != expected:
                raise RuntimeError(
                    f"pair pooled width changed for mode={self.pair_readout}: "
                    f"{pooled.shape[1]} != {expected}"
                )
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
        unary = self._pool_nodes(patch, data.batch, n_graphs)

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


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
    )


def _evaluate_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(torch.sigmoid(model(batch)).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    return float(
        roc_auc_score(
            np.concatenate(targets).astype(np.float64),
            np.concatenate(predictions).astype(np.float64),
        )
    )


def _positive_weight(graphs: Sequence[Data]) -> float:
    labels = np.asarray([float(graph.y.item()) for graph in graphs], dtype=np.float32)
    positives = int(np.sum(labels > 0.5))
    negatives = int(labels.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"balanced BCE needs both classes: {positives=} {negatives=}")
    return float(negatives / positives)


def _train_phase(
    train_graphs: Sequence[Data],
    eval_graphs: Sequence[Data],
    *,
    code_width: int,
    config: Mapping[str, Any],
    seed: int,
    select_best: bool,
    epochs: int,
    pos_weight: float,
) -> dict[str, Any]:
    model_config = config["model"]
    device = torch.device(str(model_config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model requested CUDA but CUDA is unavailable")
    _seed_everything(seed)
    model = CompressedPatchPathModel(
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
    batch_size = int(model_config.get("batch_size", 64))
    loader = _make_loader(train_graphs, batch_size, True, seed + 91011)
    eval_loader = _make_loader(eval_graphs, batch_size, False, seed + 91012)
    patience = int(model_config.get("patience", 5))
    evaluate_every = int(model_config.get("evaluate_every", 1))
    weight = torch.as_tensor(float(pos_weight), dtype=torch.float32, device=device)
    best_auc = -float("inf")
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
            target = batch.y.view(-1)
            loss = F.binary_cross_entropy_with_logits(model(batch), target, pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        epoch_loss = total_loss / max(seen, 1)
        losses.append(float(epoch_loss))
        if select_best and (
            epoch == 1 or epoch % max(evaluate_every, 1) == 0 or epoch == int(epochs)
        ):
            current_auc = _evaluate_auc(model, eval_loader, device)
            trace.append({"epoch": int(epoch), "train_bce": float(epoch_loss), "auc": float(current_auc)})
            if current_auc > best_auc:
                best_auc = float(current_auc)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            print(
                f"molhiv K-SVD patch-path phase=select epoch={epoch:03d}/{epochs} "
                f"bce={epoch_loss:.6f} valid_auc={current_auc:.6f}",
                flush=True,
            )
        elif not select_best and (
            epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 5) == 0
        ):
            print(
                f"molhiv K-SVD patch-path phase=refit epoch={epoch:03d}/{epochs} "
                f"bce={epoch_loss:.6f}",
                flush=True,
            )
        if select_best and stale >= patience:
            print(
                f"molhiv K-SVD patch-path early_stop epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    final_auc = _evaluate_auc(model, eval_loader, device)
    return {
        "model": model,
        "auc": float(final_auc),
        "best_auc": None if not select_best else float(best_auc),
        "selected_epoch": int(best_epoch if select_best else epochs),
        "epochs_run": int(len(losses)),
        "trace": trace,
        "losses": losses,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _fit_phase_data(
    fit_records: Sequence[base.GraphRecord],
    other_records: Sequence[base.GraphRecord],
    *,
    representation: Mapping[str, Any],
    seed: int,
) -> tuple[list[Data], list[Data], dict[str, Any]]:
    patch_standardizer = StreamingStandardizer.fit_patch_records(fit_records)
    context_standardizer = base.Standardizer.fit(base._context_matrix(fit_records))
    dictionary_matrix = _sample_dictionary_matrix(
        fit_records,
        patch_standardizer,
        int(representation.get("max_dictionary_patches", 12000)),
        seed + 1729,
    )
    started = time.perf_counter()
    dictionary_mode = str(representation.get("dictionary_mode", "final"))
    if dictionary_mode not in {"init", "final"}:
        raise ValueError(
            f"unknown dictionary_mode={dictionary_mode!r}; expected init or final"
        )
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
        fit_records, patch_standardizer, context_standardizer, dictionary, int(representation.get("sparsity", 4))
    )
    encoded_other = _encode_records(
        other_records, patch_standardizer, context_standardizer, dictionary, int(representation.get("sparsity", 4))
    )
    audit = {
        "patch_width": int(dictionary_matrix.shape[0]),
        "dictionary_samples": int(dictionary_matrix.shape[1]),
        "dictionary": dictionary_info,
        "dictionary_seconds": float(dictionary_seconds),
        "code_width": int(dictionary.shape[1]),
        "sparsity": int(representation.get("sparsity", 4)),
        "patch_aux_width": 2,
        "standardizer_fit_graphs": int(len(fit_records)),
        "descriptor": str(representation.get("descriptor", "shell")),
        "descriptor_width": int(dictionary_matrix.shape[0]),
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
            "Train-only K-SVD sparse patch code with explicit shortest-path pair pooling; one MLP, no message passing.",
            "",
            f"- split: `{result['data']['split']}`; sizes `{result['data']['sizes']}`",
            f"- patch descriptor: `{representation['patch_width']}D`; dictionary: `K={representation['dictionary_atoms']}`, `T={representation['sparsity']}`",
            f"- pair relation: `{representation['relation_width']}D`; readout: `{representation['readout']}`",
            f"- message passing: `{representation['message_passing']}`; attention: `{representation['attention']}`",
            "",
            "| head | valid ROC-AUC | test ROC-AUC after train+valid refit | selected epoch |",
            "|---|---:|---:|---:|",
            f"| `single_mlp` | {evaluation['valid']['auc']:.6f} | {evaluation['test_after_train_valid_refit']['auc']:.6f} | {evaluation['valid']['selected_epoch']} |",
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
    started = time.perf_counter()
    seed = int(config.get("seed", 0))
    descriptor_name = str(representation.get("descriptor", "shell"))
    torch.set_num_threads(max(int(config.get("runtime", {}).get("torch_threads", 4)), 1))
    cache_value = config.get("runtime", {}).get("record_cache")
    record_cache_dir = None if cache_value in (None, "", False) else _resolve(str(cache_value))

    max_graphs = data_config.get("max_graphs")
    bundle = load_molhiv(
        root=_resolve(data_config["root"]),
        max_graphs=None if max_graphs is None else int(max_graphs),
        seed=seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV OGB node/edge features were not loaded")
    split_indices = {
        name: np.asarray(bundle.split[name], dtype=np.int64)
        for name in ("train", "valid", "test")
    }
    split_meta = {
        name: {
            "n_graphs": int(values.size),
            "positive_graphs": int(bundle.y[values].sum()),
        }
        for name, values in split_indices.items()
    }

    feature_metadata: dict[str, Any] = {}
    train_records, feature_metadata["train"], train_cache_hit = _extract_or_load_split(
        bundle, split_indices["train"], "train", descriptor_name, record_cache_dir
    )
    valid_records, feature_metadata["valid"], valid_cache_hit = _extract_or_load_split(
        bundle, split_indices["valid"], "valid", descriptor_name, record_cache_dir
    )
    valid_train_data, valid_eval_data, valid_audit = _fit_phase_data(
        train_records, valid_records, representation=representation, seed=seed
    )
    valid_pos_weight = _positive_weight(valid_train_data)
    valid_phase = _train_phase(
        valid_train_data,
        valid_eval_data,
        code_width=int(valid_audit["code_width"]),
        config=config,
        seed=seed,
        select_best=True,
        epochs=int(model_config.get("epochs", 20)),
        pos_weight=valid_pos_weight,
    )
    selected_epoch = int(valid_phase["selected_epoch"])
    valid_auc = float(valid_phase["auc"])
    valid_parameters = int(valid_phase["parameters"])
    valid_dictionary_info = valid_audit["dictionary"]
    del valid_train_data, valid_eval_data, valid_phase["model"]
    del train_records, valid_records
    gc.collect()

    refit_indices = np.concatenate([split_indices["train"], split_indices["valid"]]).astype(
        np.int64, copy=False
    )
    refit_records, feature_metadata["train_valid_refit"], refit_cache_hit = _extract_or_load_split(
        bundle, refit_indices, "train+valid", descriptor_name, record_cache_dir
    )
    test_records, feature_metadata["test"], test_cache_hit = _extract_or_load_split(
        bundle, split_indices["test"], "test", descriptor_name, record_cache_dir
    )
    refit_train_data, refit_test_data, test_audit = _fit_phase_data(
        refit_records, test_records, representation=representation, seed=seed
    )
    refit_pos_weight = _positive_weight(refit_train_data)
    refit_phase = _train_phase(
        refit_train_data,
        refit_test_data,
        code_width=int(test_audit["code_width"]),
        config=config,
        seed=seed,
        select_best=False,
        epochs=selected_epoch,
        pos_weight=refit_pos_weight,
    )

    dictionary_atoms = int(valid_audit["code_width"])
    result = {
        "protocol_id": str(config["protocol_id"]),
        "status": "completed",
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "data": {
            "root": str(_resolve(data_config["root"])),
            "split": "OGB ogbg-molhiv official scaffold train/valid/test",
            "sizes": {name: int(values.size) for name, values in split_indices.items()},
            "positive": {
                name: int(split_meta[name]["positive_graphs"]) for name in split_indices
            },
            "test_labels_used_for_selection": False,
            "record_cache": None if record_cache_dir is None else str(record_cache_dir),
            "record_cache_hits": {
                "train": train_cache_hit,
                "valid": valid_cache_hit,
                "train_valid_refit": refit_cache_hit,
                "test": test_cache_hit,
            },
        },
        "representation": {
            "radius": base.PATCH_RADIUS,
            "centres": "every atom",
        "patch_width": int(valid_audit["descriptor_width"]),
            "descriptor": descriptor_name,
            "patch_descriptor": (
                "invariant radius-2 shell atom/bond composition, root/incident "
                "chemistry, size/cycle/degree scalars"
                if descriptor_name == "shell"
                else "shell descriptor + endpoint-distance typed two-walk bond relations + shell-pair endpoint atomic-number mass"
            ),
            "dictionary_atoms": dictionary_atoms,
            "sparsity": int(representation.get("sparsity", 4)),
            "relation_width": int(base.RELATION_WIDTH),
            "global_width": int(base.GLOBAL_WIDTH),
            "dictionary_fit": "train-only for valid; train+valid for test refit",
            "dictionary_mode": str(valid_audit.get("dictionary_mode", "final")),
            "dictionary_recon_rel": float(valid_dictionary_info["recon_rel"]),
            # Retain the old key for consumers of the first structured report.
            "dictionary_final_recon_rel": float(valid_dictionary_info["recon_rel"]),
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
            "loss": "balanced binary cross entropy with logits",
            "metric": "ROC-AUC",
            "device": str(torch.device(str(model_config.get("device", "cpu")))),
            "selected_epoch": selected_epoch,
            "one_predictive_head": True,
        },
        "evaluation": {
            "valid": {
                "auc": valid_auc,
                "best_auc": float(valid_phase["best_auc"]),
                "selected_epoch": selected_epoch,
                "epochs_run": int(valid_phase["epochs_run"]),
                "trace": valid_phase["trace"],
            },
            "test_after_train_valid_refit": {
                "auc": float(refit_phase["auc"]),
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
    config_path = _resolve(args.config)
    result = run(config_path)
    print(
        json.dumps(
            {
                "valid_auc": result["evaluation"]["valid"]["auc"],
                "test_auc": result["evaluation"]["test_after_train_valid_refit"]["auc"],
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
