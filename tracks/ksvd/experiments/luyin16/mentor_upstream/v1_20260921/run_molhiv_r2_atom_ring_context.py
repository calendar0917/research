#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2 shared-atom × explicit-ring-context features for OGBG-MolHIV.

This script freezes the existing radius-2 K-SVD dictionary and asks a focused
question: which learned local structural atoms occur inside which mesoscopic
ring contexts?

For each node/root v, its signed sparse K-SVD code is converted into a
non-negative atom-use distribution

    a_v[k] = |x_v[k]| / (sum_j |x_v[j]| + eps).

Six overlapping node contexts are extracted from chordless induced cycles of
size 3..6:

    ring_any       : node belongs to any induced 3..6 cycle
    ring5          : node belongs to a 5-cycle
    ring6          : node belongs to a 6-cycle
    aromatic_ring  : node belongs to a fully aromatic cycle
    multi_ring     : node belongs to >=2 cycles or to a fused/spiro/overlap ring
    ring_boundary  : node is outside all cycles but adjacent to a ring node

For every graph and context c, three 64-D statistics are generated:

    mass[k,c] = (1/|V|) sum_v a_v[k] 1[v in c]

    conditional[k,c] = mean_{v in c} a_v[k]

    enrichment[k,c] = mass[k,c] - coverage[c] * global_atom_mean[k]

Each representation appends the six context coverages, producing 6*K+6 =
390 dimensions for K=64.

The script has three subcommands:

    self-test  : synthetic checks
    build      : build atom-context graph features
    classify   : evaluate R2 backbone + context ablations with the exact old
                 MolHIV XGBoost/Optuna protocol

No HIV labels are used to construct the context features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np


CONTEXT_NAMES: Tuple[str, ...] = (
    "ring_any",
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)
N_CONTEXTS = len(CONTEXT_NAMES)
SPLIT_NAMES = {0: "train", 1: "valid", 2: "test"}


# =============================================================================
# Generic helpers
# =============================================================================


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, ensure_ascii=False, sort_keys=True)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def assert_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        raise FloatingPointError(f"{name} contains non-finite values; first={bad[:5].tolist()}")


def stable_config_hash(payload: Mapping[str, object]) -> str:
    text = json.dumps(json_ready(dict(payload)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_module_from_path(path: Path, module_name: str):
    """Import a Python file safely on Python 3.9, including @dataclass modules."""
    path = path.expanduser().resolve()
    require_file(path, "Python module")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.9 dataclasses consult sys.modules while the class is created.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def flatten_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [N] or [N,1], got {labels.shape}")
    return labels.astype(np.int64, copy=False)


def create_or_open_memmap(path: Path, shape: Tuple[int, ...], dtype, resume: bool):
    if resume:
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"Resume array mismatch for {path}: {array.shape}/{array.dtype}, "
                f"expected {shape}/{np.dtype(dtype)}"
            )
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


# =============================================================================
# Sparse R2 node codes
# =============================================================================


def load_r2_export(
    ksvd_module,
    cache_dir: Path,
    result_dir: Path,
    expected_atoms: int,
    expected_sparsity: int,
) -> Dict[str, object]:
    cache = ksvd_module.PilotCache.load(str(cache_dir))
    dictionary = np.load(
        require_file(result_dir / "dictionary_scaled.npy", "scaled dictionary"),
        mmap_mode="r",
    )
    code_indices = np.load(
        require_file(result_dir / "code_indices.npy", "code indices"), mmap_mode="r"
    )
    code_values = np.load(
        require_file(result_dir / "code_values.npy", "code values"), mmap_mode="r"
    )
    code_nnz = np.load(
        require_file(result_dir / "code_nnz.npy", "code nnz"), mmap_mode="r"
    )
    root_ids = np.load(
        require_file(cache_dir / "root_ids.npy", "root ids"), mmap_mode="r"
    )

    feature_dim, n_atoms = dictionary.shape
    if int(n_atoms) != int(expected_atoms):
        raise RuntimeError(f"Expected K={expected_atoms}, dictionary has K={n_atoms}")
    if feature_dim != int(cache.layout.feature_dim):
        raise RuntimeError(
            f"Dictionary F={feature_dim} does not match cache F={cache.layout.feature_dim}"
        )
    total_patches = int(cache.graph_offsets[-1])
    expected_shape = (total_patches, int(expected_sparsity))
    if code_indices.shape != expected_shape or code_values.shape != expected_shape:
        raise RuntimeError(
            f"Expected sparse codes {expected_shape}, got "
            f"indices={code_indices.shape}, values={code_values.shape}"
        )
    if code_nnz.shape != (total_patches,):
        raise RuntimeError(f"code_nnz shape mismatch: {code_nnz.shape}")
    if root_ids.shape != (total_patches,):
        raise RuntimeError(f"root_ids shape mismatch: {root_ids.shape}")

    return {
        "cache": cache,
        "n_atoms": int(n_atoms),
        "sparsity": int(expected_sparsity),
        "code_indices": code_indices,
        "code_values": code_values,
        "code_nnz": code_nnz,
        "root_ids": root_ids,
    }


def dense_normalized_atom_use(
    local_indices: np.ndarray,
    local_values: np.ndarray,
    local_nnz: np.ndarray,
    local_root_ids: np.ndarray,
    num_nodes: int,
    n_atoms: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert one graph's sparse signed root codes to [num_nodes,K] |code| distributions."""
    local_indices = np.asarray(local_indices)
    local_values = np.asarray(local_values, dtype=np.float32)
    local_nnz = np.asarray(local_nnz, dtype=np.int64)
    local_root_ids = np.asarray(local_root_ids, dtype=np.int64)

    n_rows, width = local_indices.shape
    if local_values.shape != (n_rows, width):
        raise ValueError("local code index/value shape mismatch")
    if local_nnz.shape != (n_rows,) or local_root_ids.shape != (n_rows,):
        raise ValueError("local nnz/root shape mismatch")
    if np.any(local_root_ids < 0) or np.any(local_root_ids >= num_nodes):
        raise IndexError("root id outside graph")
    if len(np.unique(local_root_ids)) != num_nodes:
        raise RuntimeError(
            f"Expected one rooted patch per node: unique roots={len(np.unique(local_root_ids))}, "
            f"num_nodes={num_nodes}"
        )

    dense = np.zeros((num_nodes, n_atoms), dtype=np.float32)
    for row in range(n_rows):
        nnz = min(max(int(local_nnz[row]), 0), width)
        if nnz == 0:
            continue
        atom_ids = np.asarray(local_indices[row, :nnz], dtype=np.int64)
        values = np.abs(np.asarray(local_values[row, :nnz], dtype=np.float32))
        valid = (atom_ids >= 0) & (atom_ids < n_atoms) & np.isfinite(values)
        if not np.any(valid):
            continue
        np.add.at(dense[int(local_root_ids[row])], atom_ids[valid], values[valid])

    row_sum = dense.sum(axis=1, keepdims=True)
    nonzero = row_sum[:, 0] > eps
    dense[nonzero] /= row_sum[nonzero]
    return dense


# =============================================================================
# Ring context extraction
# =============================================================================


def is_fully_aromatic_cycle(
    cycle: Sequence[int],
    raw_atom: np.ndarray,
    edge_features: Mapping[Tuple[int, int], np.ndarray],
    ring_module,
) -> bool:
    nodes = np.asarray(cycle, dtype=np.int64)
    if nodes.size == 0 or raw_atom.shape[1] <= 7:
        return False
    if not bool(np.all(raw_atom[nodes, 7] == 1)):
        return False
    for key in ring_module.cycle_edges(cycle):
        feature = np.asarray(edge_features[key]).reshape(-1)
        if feature.size == 0 or int(feature[0]) != 3:
            return False
    return True


def build_context_masks(
    raw_atom: np.ndarray,
    adjacency: Sequence[Set[int]],
    edge_features: Mapping[Tuple[int, int], np.ndarray],
    cycles: Sequence[Tuple[int, ...]],
    ring_module,
) -> Tuple[np.ndarray, Dict[str, float]]:
    num_nodes = int(raw_atom.shape[0])
    masks = np.zeros((num_nodes, N_CONTEXTS), dtype=np.bool_)
    memberships = np.zeros(num_nodes, dtype=np.int32)

    ring_relations = ring_module.analyze_ring_relations(cycles, adjacency)
    fused = np.asarray(ring_relations["fused_neighbors"], dtype=np.float32)
    spiro = np.asarray(ring_relations["spiro_neighbors"], dtype=np.float32)
    overlap = np.asarray(ring_relations["overlap_neighbors"], dtype=np.float32)

    for ring_id, cycle in enumerate(cycles):
        nodes = np.asarray(cycle, dtype=np.int64)
        memberships[nodes] += 1
        masks[nodes, 0] = True
        if len(cycle) == 5:
            masks[nodes, 1] = True
        if len(cycle) == 6:
            masks[nodes, 2] = True
        if is_fully_aromatic_cycle(cycle, raw_atom, edge_features, ring_module):
            masks[nodes, 3] = True
        if fused[ring_id] > 0 or spiro[ring_id] > 0 or overlap[ring_id] > 0:
            masks[nodes, 4] = True

    masks[memberships >= 2, 4] = True

    ring_nodes = np.flatnonzero(masks[:, 0])
    if ring_nodes.size:
        boundary: Set[int] = set()
        ring_set = set(int(v) for v in ring_nodes)
        for node in ring_nodes:
            for neighbor in adjacency[int(node)]:
                if int(neighbor) not in ring_set:
                    boundary.add(int(neighbor))
        if boundary:
            masks[np.asarray(sorted(boundary), dtype=np.int64), 5] = True

    diagnostics = {
        "num_rings": float(len(cycles)),
        "num_ring_nodes": float(np.sum(masks[:, 0])),
        "num_ring5_nodes": float(np.sum(masks[:, 1])),
        "num_ring6_nodes": float(np.sum(masks[:, 2])),
        "num_aromatic_ring_nodes": float(np.sum(masks[:, 3])),
        "num_multi_ring_nodes": float(np.sum(masks[:, 4])),
        "num_ring_boundary_nodes": float(np.sum(masks[:, 5])),
    }
    return masks, diagnostics


def pool_atom_context(
    atom_use: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atom_use = np.asarray(atom_use, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.bool_)
    if atom_use.ndim != 2 or masks.ndim != 2:
        raise ValueError("atom_use/masks must be matrices")
    if atom_use.shape[0] != masks.shape[0] or masks.shape[1] != N_CONTEXTS:
        raise ValueError(
            f"shape mismatch atom_use={atom_use.shape}, masks={masks.shape}"
        )
    num_nodes, n_atoms = atom_use.shape
    global_mean = atom_use.mean(axis=0) if num_nodes else np.zeros(n_atoms, dtype=np.float32)
    coverage = masks.mean(axis=0).astype(np.float32) if num_nodes else np.zeros(N_CONTEXTS, dtype=np.float32)

    mass = np.zeros((N_CONTEXTS, n_atoms), dtype=np.float32)
    conditional = np.zeros_like(mass)
    for context_id in range(N_CONTEXTS):
        selected = masks[:, context_id]
        if np.any(selected):
            selected_sum = atom_use[selected].sum(axis=0)
            mass[context_id] = selected_sum / max(num_nodes, 1)
            conditional[context_id] = selected_sum / int(np.sum(selected))
    enrichment = mass - coverage[:, None] * global_mean[None, :]

    return (
        np.concatenate([mass.reshape(-1), coverage], axis=0).astype(np.float32),
        np.concatenate([conditional.reshape(-1), coverage], axis=0).astype(np.float32),
        np.concatenate([enrichment.reshape(-1), coverage], axis=0).astype(np.float32),
        coverage,
    )


# =============================================================================
# Feature paths / loading
# =============================================================================


def build_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "mass": out_dir / "atom_ring_context_mass.npy",
        "conditional": out_dir / "atom_ring_context_conditional.npy",
        "enrichment": out_dir / "atom_ring_context_enrichment.npy",
        "diagnostics": out_dir / "atom_ring_context_diagnostics.npy",
        "metadata": out_dir / "atom_ring_context_metadata.npz",
        "names": out_dir / "atom_ring_context_feature_names.json",
        "manifest": out_dir / "atom_ring_context_manifest.json",
        "progress": out_dir / "atom_ring_context_progress.json",
    }


def context_feature_names(n_atoms: int, prefix: str) -> List[str]:
    names = [
        f"{prefix}.{context}.atom_{atom_id:03d}"
        for context in CONTEXT_NAMES
        for atom_id in range(n_atoms)
    ]
    names += [f"{prefix}.coverage.{context}" for context in CONTEXT_NAMES]
    return names


def load_context_payload(feature_dir: Path) -> Dict[str, object]:
    paths = build_paths(feature_dir)
    manifest = load_json(require_file(paths["manifest"], "context manifest"))
    if not bool(manifest.get("complete", False)):
        raise RuntimeError("Atom-ring context build is not complete")
    metadata = np.load(require_file(paths["metadata"], "context metadata"), allow_pickle=False)
    names = load_json(require_file(paths["names"], "context names"))
    mass = np.load(require_file(paths["mass"], "context mass"), mmap_mode="r")
    conditional = np.load(
        require_file(paths["conditional"], "context conditional"), mmap_mode="r"
    )
    enrichment = np.load(
        require_file(paths["enrichment"], "context enrichment"), mmap_mode="r"
    )
    diagnostics = np.load(
        require_file(paths["diagnostics"], "context diagnostics"), mmap_mode="r"
    )

    dataset_indices = np.asarray(metadata["dataset_indices"], dtype=np.int64)
    graph_split = np.asarray(metadata["graph_split"], dtype=np.uint8)
    labels = flatten_labels(np.asarray(metadata["labels"]))
    n_atoms = int(np.asarray(metadata["n_atoms"]).item())
    feature_dim = n_atoms * N_CONTEXTS + N_CONTEXTS
    n_graphs = len(dataset_indices)
    for label, matrix in [
        ("mass", mass),
        ("conditional", conditional),
        ("enrichment", enrichment),
    ]:
        if matrix.shape != (n_graphs, feature_dim):
            raise RuntimeError(f"{label} shape mismatch: {matrix.shape}, expected {(n_graphs, feature_dim)}")
    if diagnostics.shape != (n_graphs, 7):
        raise RuntimeError(f"diagnostics shape mismatch: {diagnostics.shape}")

    return {
        "dataset_indices": dataset_indices,
        "graph_split": graph_split,
        "labels": labels,
        "n_atoms": n_atoms,
        "mass": mass,
        "conditional": conditional,
        "enrichment": enrichment,
        "diagnostics": diagnostics,
        "mass_names": list(names["mass_names"]),
        "conditional_names": list(names["conditional_names"]),
        "enrichment_names": list(names["enrichment_names"]),
        "manifest": manifest,
    }


# =============================================================================
# Build
# =============================================================================


def run_build(args: argparse.Namespace) -> None:
    if not 3 <= int(args.max_ring_size) <= 8:
        raise ValueError("--max-ring-size must be between 3 and 8")

    ksvd_script = Path(args.ksvd_script).expanduser().resolve()
    ring_script = Path(args.ring_script).expanduser().resolve()
    cache_dir = Path(args.ksvd_cache_dir).expanduser().resolve()
    result_dir = Path(args.ksvd_result_dir).expanduser().resolve()
    diagnostic_dir = Path(args.diagnostic_feature_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = build_paths(out_dir)

    ksvd = import_module_from_path(ksvd_script, "molhiv_r2_context_ksvd")
    ring = import_module_from_path(ring_script, "molhiv_r2_context_ring")
    local = ring.load_diagnostic_local_features(diagnostic_dir)
    export = load_r2_export(
        ksvd_module=ksvd,
        cache_dir=cache_dir,
        result_dir=result_dir,
        expected_atoms=args.expected_atoms,
        expected_sparsity=args.expected_sparsity,
    )
    cache = export["cache"]

    if not np.array_equal(
        np.asarray(cache.dataset_indices, dtype=np.int64),
        np.asarray(local["dataset_indices"], dtype=np.int64),
    ):
        raise RuntimeError("R2 structural cache and diagnostic feature dataset order differ")
    if not np.array_equal(
        np.asarray(cache.graph_split, dtype=np.uint8),
        np.asarray(local["graph_split"], dtype=np.uint8),
    ):
        raise RuntimeError("R2 structural cache and diagnostic feature split flags differ")

    dataset = ksvd.load_molhiv(args.dataset_root)
    n_graphs = len(cache.dataset_indices)
    n_atoms = int(export["n_atoms"])
    feature_dim = n_atoms * N_CONTEXTS + N_CONTEXTS

    config = {
        "ksvd_script": str(ksvd_script),
        "ring_script": str(ring_script),
        "ksvd_cache_dir": str(cache_dir),
        "ksvd_result_dir": str(result_dir),
        "diagnostic_feature_dir": str(diagnostic_dir),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "n_graphs": n_graphs,
        "n_atoms": n_atoms,
        "sparsity": int(export["sparsity"]),
        "max_ring_size": int(args.max_ring_size),
        "contexts": list(CONTEXT_NAMES),
        "feature_dim": feature_dim,
    }
    config_hash = stable_config_hash(config)

    if args.rebuild:
        for path in paths.values():
            if path.exists():
                path.unlink()

    resume = paths["progress"].is_file()
    start_graph = 0
    aggregate: Dict[str, float] = {
        "total_rings": 0.0,
        "graphs_with_rings": 0.0,
        **{f"coverage_sum.{name}": 0.0 for name in CONTEXT_NAMES},
    }
    if resume:
        progress = load_json(paths["progress"])
        if progress.get("config_hash") != config_hash:
            raise RuntimeError("Existing progress has a different configuration; use --rebuild")
        start_graph = int(progress.get("next_graph", 0))
        aggregate.update({str(k): float(v) for k, v in progress.get("aggregate", {}).items()})
        print(f"Resuming atom-ring context build at graph {start_graph}/{n_graphs}")

    mass_array = create_or_open_memmap(
        paths["mass"], (n_graphs, feature_dim), np.float32, resume
    )
    conditional_array = create_or_open_memmap(
        paths["conditional"], (n_graphs, feature_dim), np.float32, resume
    )
    enrichment_array = create_or_open_memmap(
        paths["enrichment"], (n_graphs, feature_dim), np.float32, resume
    )
    diagnostics_array = create_or_open_memmap(
        paths["diagnostics"], (n_graphs, 7), np.float32, resume
    )

    print("MolHIV R2 atom × ring-context build")
    print(f"  graphs             : {n_graphs}")
    print(f"  R2 atoms/sparsity  : {n_atoms}/{int(export['sparsity'])}")
    print(f"  ring sizes         : 3..{int(args.max_ring_size)}")
    print(f"  contexts           : {', '.join(CONTEXT_NAMES)}")
    print(f"  feature dim/mode   : {feature_dim}")
    print("  label use          : none")

    graph_offsets = np.asarray(cache.graph_offsets, dtype=np.int64)
    code_indices = export["code_indices"]
    code_values = export["code_values"]
    code_nnz = export["code_nnz"]
    root_ids = export["root_ids"]

    for graph_pos in range(start_graph, n_graphs):
        dataset_index = int(cache.dataset_indices[graph_pos])
        data = dataset[dataset_index]
        raw_atom = ring.torch_to_numpy(data.x, dtype=np.int64)
        edge_index = ring.torch_to_numpy(data.edge_index, dtype=np.int64)
        edge_attr = ring.torch_to_numpy(data.edge_attr, dtype=np.int64)
        num_nodes = int(raw_atom.shape[0])

        start = int(graph_offsets[graph_pos])
        end = int(graph_offsets[graph_pos + 1])
        if end - start != num_nodes:
            raise RuntimeError(
                f"Graph {graph_pos}: cache roots={end-start}, dataset nodes={num_nodes}"
            )
        atom_use = dense_normalized_atom_use(
            local_indices=np.asarray(code_indices[start:end]),
            local_values=np.asarray(code_values[start:end], dtype=np.float32),
            local_nnz=np.asarray(code_nnz[start:end]),
            local_root_ids=np.asarray(root_ids[start:end]),
            num_nodes=num_nodes,
            n_atoms=n_atoms,
        )

        adjacency, edge_features = ring.build_undirected_graph(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )
        cycles = ring.enumerate_chordless_cycles(
            adjacency,
            min_size=3,
            max_size=int(args.max_ring_size),
        )
        masks, diagnostics = build_context_masks(
            raw_atom=raw_atom,
            adjacency=adjacency,
            edge_features=edge_features,
            cycles=cycles,
            ring_module=ring,
        )
        mass, conditional, enrichment, coverage = pool_atom_context(atom_use, masks)
        assert_finite("mass", mass)
        assert_finite("conditional", conditional)
        assert_finite("enrichment", enrichment)

        mass_array[graph_pos] = mass
        conditional_array[graph_pos] = conditional
        enrichment_array[graph_pos] = enrichment
        diagnostics_array[graph_pos] = np.asarray(
            [
                float(num_nodes),
                diagnostics["num_rings"],
                diagnostics["num_ring_nodes"],
                diagnostics["num_ring5_nodes"],
                diagnostics["num_ring6_nodes"],
                diagnostics["num_aromatic_ring_nodes"],
                diagnostics["num_multi_ring_nodes"],
            ],
            dtype=np.float32,
        )

        aggregate["total_rings"] += diagnostics["num_rings"]
        aggregate["graphs_with_rings"] += float(diagnostics["num_rings"] > 0)
        for context_id, context_name in enumerate(CONTEXT_NAMES):
            aggregate[f"coverage_sum.{context_name}"] += float(coverage[context_id])

        completed = graph_pos + 1
        if completed % max(1, int(args.checkpoint_every)) == 0 or completed == n_graphs:
            mass_array.flush()
            conditional_array.flush()
            enrichment_array.flush()
            diagnostics_array.flush()
            save_json(
                paths["progress"],
                {
                    "config_hash": config_hash,
                    "next_graph": completed,
                    "aggregate": aggregate,
                },
            )
            print(
                f"  processed {completed:6d}/{n_graphs} | "
                f"rings={int(aggregate['total_rings'])} | "
                f"graphs_with_rings={int(aggregate['graphs_with_rings'])}"
            )

    np.savez_compressed(
        paths["metadata"],
        dataset_indices=np.asarray(local["dataset_indices"], dtype=np.int64),
        graph_split=np.asarray(local["graph_split"], dtype=np.uint8),
        labels=np.asarray(local["labels"], dtype=np.int64),
        n_atoms=np.asarray(n_atoms, dtype=np.int64),
        sparsity=np.asarray(int(export["sparsity"]), dtype=np.int64),
        max_ring_size=np.asarray(int(args.max_ring_size), dtype=np.int64),
        feature_dim=np.asarray(feature_dim, dtype=np.int64),
    )
    save_json(
        paths["names"],
        {
            "context_names": list(CONTEXT_NAMES),
            "mass_names": context_feature_names(n_atoms, "atom_ring_mass"),
            "conditional_names": context_feature_names(n_atoms, "atom_ring_conditional"),
            "enrichment_names": context_feature_names(n_atoms, "atom_ring_enrichment"),
            "diagnostic_names": [
                "num_nodes",
                "num_rings",
                "num_ring_nodes",
                "num_ring5_nodes",
                "num_ring6_nodes",
                "num_aromatic_ring_nodes",
                "num_multi_ring_nodes",
            ],
        },
    )
    completed_aggregate = dict(aggregate)
    completed_aggregate["mean_rings_per_graph"] = aggregate["total_rings"] / max(n_graphs, 1)
    completed_aggregate["fraction_graphs_with_rings"] = aggregate["graphs_with_rings"] / max(n_graphs, 1)
    for context_name in CONTEXT_NAMES:
        completed_aggregate[f"mean_coverage.{context_name}"] = (
            aggregate[f"coverage_sum.{context_name}"] / max(n_graphs, 1)
        )
    save_json(
        paths["manifest"],
        {
            "complete": True,
            "config": config,
            "config_hash": config_hash,
            "aggregate": completed_aggregate,
            "files": {key: str(path) for key, path in paths.items()},
        },
    )
    if paths["progress"].exists():
        paths["progress"].unlink()

    print("\nR2 atom × ring-context build complete")
    print(f"  mean rings/graph          : {completed_aggregate['mean_rings_per_graph']:.4f}")
    print(f"  graphs with rings         : {completed_aggregate['fraction_graphs_with_rings']:.4f}")
    for context_name in CONTEXT_NAMES:
        print(
            f"  mean coverage {context_name:<13}: "
            f"{completed_aggregate[f'mean_coverage.{context_name}']:.6f}"
        )
    print(f"  output                    : {out_dir}")


# =============================================================================
# Classification
# =============================================================================


def supported_modes() -> List[str]:
    return [
        "r2_recon",
        "r2_recon_ring_summary",
        "atom_context_mass_only",
        "atom_context_conditional_only",
        "atom_context_enrichment_only",
        "r2_recon_atom_context_mass",
        "r2_recon_atom_context_conditional",
        "r2_recon_atom_context_enrichment",
        "r2_recon_ring_summary_atom_context_mass",
        "r2_recon_ring_summary_atom_context_conditional",
        "r2_recon_ring_summary_atom_context_enrichment",
    ]


def build_mode(
    mode: str,
    local: Mapping[str, np.ndarray],
    ring_payload: Mapping[str, object],
    context: Mapping[str, object],
    ring_module,
) -> Tuple[np.ndarray, List[str]]:
    composition = np.asarray(local["composition"], dtype=np.float32)
    recon = np.asarray(local["recon_typed"], dtype=np.float32)
    ring_summary = np.asarray(ring_payload["summary"], dtype=np.float32)
    mass = np.asarray(context["mass"], dtype=np.float32)
    conditional = np.asarray(context["conditional"], dtype=np.float32)
    enrichment = np.asarray(context["enrichment"], dtype=np.float32)

    composition_names = ring_module.composition_feature_names()
    recon_names = ring_module.recon_typed_feature_names()
    ring_names = list(ring_payload["summary_names"])
    mass_names = list(context["mass_names"])
    conditional_names = list(context["conditional_names"])
    enrichment_names = list(context["enrichment_names"])

    backbone = np.concatenate([composition, recon], axis=1).astype(np.float32)
    backbone_names = composition_names + recon_names

    if mode == "r2_recon":
        return backbone, backbone_names
    if mode == "r2_recon_ring_summary":
        return np.concatenate([backbone, ring_summary], axis=1), backbone_names + ring_names
    if mode == "atom_context_mass_only":
        return mass, mass_names
    if mode == "atom_context_conditional_only":
        return conditional, conditional_names
    if mode == "atom_context_enrichment_only":
        return enrichment, enrichment_names
    if mode == "r2_recon_atom_context_mass":
        return np.concatenate([backbone, mass], axis=1), backbone_names + mass_names
    if mode == "r2_recon_atom_context_conditional":
        return np.concatenate([backbone, conditional], axis=1), backbone_names + conditional_names
    if mode == "r2_recon_atom_context_enrichment":
        return np.concatenate([backbone, enrichment], axis=1), backbone_names + enrichment_names
    if mode == "r2_recon_ring_summary_atom_context_mass":
        return (
            np.concatenate([backbone, ring_summary, mass], axis=1),
            backbone_names + ring_names + mass_names,
        )
    if mode == "r2_recon_ring_summary_atom_context_conditional":
        return (
            np.concatenate([backbone, ring_summary, conditional], axis=1),
            backbone_names + ring_names + conditional_names,
        )
    if mode == "r2_recon_ring_summary_atom_context_enrichment":
        return (
            np.concatenate([backbone, ring_summary, enrichment], axis=1),
            backbone_names + ring_names + enrichment_names,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def run_classify(args: argparse.Namespace) -> None:
    classifier = import_module_from_path(
        Path(args.classification_script), "molhiv_r2_context_old_protocol_classifier"
    )
    ring_module = import_module_from_path(
        Path(args.ring_script), "molhiv_r2_context_ring_classifier"
    )
    local = ring_module.load_diagnostic_local_features(
        Path(args.diagnostic_feature_dir).expanduser().resolve()
    )
    ring_payload = ring_module.load_ring_payload(
        Path(args.ring_feature_dir).expanduser().resolve()
    )
    context = load_context_payload(Path(args.context_feature_dir).expanduser().resolve())

    for key in ("dataset_indices", "graph_split", "labels"):
        if not np.array_equal(np.asarray(local[key]), np.asarray(ring_payload[key])):
            raise RuntimeError(f"Local and ring payload mismatch: {key}")
        if not np.array_equal(np.asarray(local[key]), np.asarray(context[key])):
            raise RuntimeError(f"Local and atom-context payload mismatch: {key}")

    available = supported_modes()
    unknown = [mode for mode in args.modes if mode not in available]
    if unknown:
        raise ValueError(f"Unknown modes {unknown}; available={available}")

    out_dir = Path(args.result_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier.set_seed(int(args.global_seed))
    positions = classifier.split_positions(np.asarray(local["graph_split"], dtype=np.uint8))
    labels = flatten_labels(np.asarray(local["labels"]))

    summaries: List[Dict[str, object]] = []
    for mode in args.modes:
        matrix, feature_names = build_mode(
            mode=mode,
            local=local,
            ring_payload=ring_payload,
            context=context,
            ring_module=ring_module,
        )
        matrix = np.asarray(matrix, dtype=np.float32)
        assert_finite(mode, matrix)
        if matrix.shape[1] != len(feature_names):
            raise RuntimeError(
                f"Mode {mode}: matrix dim={matrix.shape[1]}, names={len(feature_names)}"
            )
        summary = classifier.evaluate_mode(
            mode=mode,
            matrix=matrix,
            feature_names=feature_names,
            labels=labels,
            positions=positions,
            trials=int(args.trials),
            tune_seeds=args.tune_seeds,
            final_seeds=args.final_seeds,
            optuna_seed=int(args.optuna_seed),
            xgb_n_jobs=int(args.xgb_n_jobs),
            out_dir=out_dir,
        )
        summaries.append(summary)
        save_json(out_dir / "all_modes.partial.json", {"modes": summaries})

    ranked = sorted(
        summaries,
        key=lambda item: float(item["scores"]["valid"]["mean"]),
        reverse=True,
    )
    save_json(out_dir / "all_modes_summary.json", {"modes": summaries, "ranking": ranked})
    with (out_dir / "all_modes_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "mode", "dim", "train_mean", "train_std", "valid_mean", "valid_std", "test_mean", "test_std"])
        for rank, summary in enumerate(ranked, start=1):
            scores = summary["scores"]
            writer.writerow(
                [
                    rank,
                    summary["mode"],
                    summary["feature_dim"],
                    scores["train"]["mean"],
                    scores["train"]["std"],
                    scores["valid"]["mean"],
                    scores["valid"]["std"],
                    scores["test"]["mean"],
                    scores["test"]["std"],
                ]
            )

    print("\n" + "=" * 128)
    print("R2 atom × ring-context ranking by mean validation ROC-AUC")
    print("=" * 128)
    for rank, summary in enumerate(ranked, start=1):
        scores = summary["scores"]
        print(
            f"{rank:02d}. {summary['mode']:<58} dim={int(summary['feature_dim']):4d} | "
            f"valid={float(scores['valid']['mean']):.6f}±{float(scores['valid']['std']):.6f} | "
            f"test={float(scores['test']['mean']):.6f}±{float(scores['test']['std']):.6f}"
        )


# =============================================================================
# Self-test
# =============================================================================


def run_self_test(args: argparse.Namespace) -> None:
    class TinyRingModule:
        @staticmethod
        def cycle_edges(cycle):
            output = []
            for i, u in enumerate(cycle):
                v = cycle[(i + 1) % len(cycle)]
                output.append((u, v) if u < v else (v, u))
            return output

        @staticmethod
        def analyze_ring_relations(cycles, adjacency):
            n = len(cycles)
            return {
                "fused_neighbors": np.zeros(n, dtype=np.float32),
                "spiro_neighbors": np.zeros(n, dtype=np.float32),
                "overlap_neighbors": np.zeros(n, dtype=np.float32),
            }

    raw_atom = np.zeros((8, 9), dtype=np.int64)
    raw_atom[:6, 7] = 1
    adjacency: List[Set[int]] = [set() for _ in range(8)]
    edge_features: Dict[Tuple[int, int], np.ndarray] = {}
    cycle = tuple(range(6))
    for u, v in TinyRingModule.cycle_edges(cycle):
        adjacency[u].add(v)
        adjacency[v].add(u)
        edge_features[(u, v)] = np.asarray([3, 0, 1], dtype=np.int64)
    adjacency[0].add(6)
    adjacency[6].add(0)
    edge_features[(0, 6)] = np.asarray([0, 0, 0], dtype=np.int64)
    adjacency[6].add(7)
    adjacency[7].add(6)
    edge_features[(6, 7)] = np.asarray([0, 0, 0], dtype=np.int64)

    masks, _ = build_context_masks(
        raw_atom=raw_atom,
        adjacency=adjacency,
        edge_features=edge_features,
        cycles=[cycle],
        ring_module=TinyRingModule,
    )
    assert np.all(masks[:6, 0])
    assert np.all(masks[:6, 2])
    assert np.all(masks[:6, 3])
    assert masks[6, 5] and not masks[7, 5]
    assert not np.any(masks[:, 1])
    assert not np.any(masks[:, 4])

    atom_use = np.zeros((8, 4), dtype=np.float32)
    atom_use[np.arange(8), np.arange(8) % 4] = 1.0
    mass, conditional, enrichment, coverage = pool_atom_context(atom_use, masks)
    expected_dim = 4 * N_CONTEXTS + N_CONTEXTS
    assert mass.shape == (expected_dim,)
    assert conditional.shape == (expected_dim,)
    assert enrichment.shape == (expected_dim,)
    assert math.isclose(float(coverage[0]), 6.0 / 8.0)
    assert math.isclose(float(coverage[5]), 1.0 / 8.0)
    assert np.all(np.isfinite(enrichment))

    indices = np.asarray([[0, 2], [1, 3]], dtype=np.int64)
    values = np.asarray([[2.0, -1.0], [-3.0, 1.0]], dtype=np.float32)
    nnz = np.asarray([2, 2], dtype=np.int64)
    roots = np.asarray([1, 0], dtype=np.int64)
    dense = dense_normalized_atom_use(indices, values, nnz, roots, num_nodes=2, n_atoms=4)
    assert np.allclose(dense.sum(axis=1), 1.0)
    assert np.allclose(dense[1, [0, 2]], [2 / 3, 1 / 3])
    assert np.allclose(dense[0, [1, 3]], [3 / 4, 1 / 4])

    print("self-test passed")
    print(f"  contexts              : {', '.join(CONTEXT_NAMES)}")
    print(f"  K=64 feature dim/mode : {64 * N_CONTEXTS + N_CONTEXTS}")
    print("  C6/aromatic/boundary  : correct")
    print("  sparse-root alignment : correct")
    print("  mass/conditional/enrichment pooling: correct")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R2 shared structural atoms conditioned on explicit ring membership."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test", help="Run synthetic checks")
    self_test.set_defaults(func=run_self_test)

    build = subparsers.add_parser("build", help="Build graph-level atom × ring-context features")
    build.add_argument("--ksvd-script", default="./molhiv_online_structural_ksvd_full.py")
    build.add_argument("--ring-script", default="./run_molhiv_explicit_ring_oracle.py")
    build.add_argument("--ksvd-cache-dir", required=True)
    build.add_argument("--ksvd-result-dir", required=True)
    build.add_argument("--diagnostic-feature-dir", required=True)
    build.add_argument("--dataset-root", default="./data/ogb")
    build.add_argument("--out-dir", required=True)
    build.add_argument("--expected-atoms", type=int, default=64)
    build.add_argument("--expected-sparsity", type=int, default=8)
    build.add_argument("--max-ring-size", type=int, default=6)
    build.add_argument("--checkpoint-every", type=int, default=500)
    build.add_argument("--rebuild", action="store_true")
    build.set_defaults(func=run_build)

    classify = subparsers.add_parser("classify", help="Evaluate atom-context ablations")
    classify.add_argument(
        "--classification-script",
        default="./run_molhiv_shared_atom_classification_old_protocol.py",
    )
    classify.add_argument("--ring-script", default="./run_molhiv_explicit_ring_oracle.py")
    classify.add_argument("--diagnostic-feature-dir", required=True)
    classify.add_argument("--ring-feature-dir", required=True)
    classify.add_argument("--context-feature-dir", required=True)
    classify.add_argument("--result-dir", required=True)
    classify.add_argument(
        "--modes",
        nargs="+",
        default=[
            "r2_recon",
            "r2_recon_ring_summary",
            "atom_context_enrichment_only",
            "r2_recon_atom_context_mass",
            "r2_recon_atom_context_conditional",
            "r2_recon_atom_context_enrichment",
            "r2_recon_ring_summary_atom_context_enrichment",
        ],
    )
    classify.add_argument("--trials", type=int, default=20)
    classify.add_argument("--tune-seeds", nargs="+", type=int, default=[0, 42, 123])
    classify.add_argument(
        "--final-seeds", nargs="+", type=int, default=[0, 42, 123, 1024, 2026]
    )
    classify.add_argument("--global-seed", type=int, default=0)
    classify.add_argument("--optuna-seed", type=int, default=2026)
    classify.add_argument("--xgb-n-jobs", type=int, default=-1)
    classify.set_defaults(func=run_classify)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
