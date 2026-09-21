#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three diagnostics for MolHIV fixed-coordinate structural K-SVD.

Experiment A: reconstruct -> typed-patch pooling
------------------------------------------------
For every rooted fixed-coordinate patch x and sparse code c:

    x_hat = D_raw c

Decode both the ORIGINAL cached structural patch x and x_hat into the same
208-D typed-patch descriptor used by run_molhiv_direct_typed_patch_pool.py,
then pool each graph with mean/std/max (624-D).

The decoder intentionally uses the ORIGINAL shell/mask channels as layout
metadata for both x and x_hat. This isolates loss in topology/bond/atom
reconstruction from instability in reconstructed shell/mask metadata.

Generated graph features:
    raw_decoded_typed_pool          [N, 624]
    recon_decoded_typed_pool        [N, 624]

If a historical direct typed-patch cache is supplied, the script compares its
624-D feature against raw_decoded_typed_pool. The latter may differ slightly
because the K-SVD cache uses a fixed capacity M and can truncate the rare
radius-2 patch above the train-derived q0.999 capacity.

Experiment B: complete code second moment
-----------------------------------------
For C_G in R^[n_patches, K], save the upper triangle of:

    S_G = C_G^T C_G / n_patches

For K=64 this is 2080 dimensions and restores signed atom co-activation.

Experiment C: permutation stability
-----------------------------------
Audit sparse-code stability when positions tied BEFORE the final node-id
canonical-order tie break are permuted. The script rebuilds exact tie groups
from OGB raw graph data using the same rooted typed-WL ordering as the K-SVD
cache, consistently permutes every positional channel, re-encodes with OMP,
and reports support Jaccard, code cosine and relative L2 change.

Dependencies
------------
This file expects these existing scripts beside it (or pass explicit paths):
    molhiv_online_structural_ksvd_full.py
    run_molhiv_shared_atom_classification_old_protocol.py

Subcommands
-----------
1) build
   Build A/B graph features from the exported K64/s8 codes.

2) classify
   Evaluate selected diagnostic modes with the EXACT historical typed-patch
   XGBoost/Optuna protocol by importing the old-protocol classification script.

3) permutation
   Run the exact-tie or all-within-shell permutation audit.

4) self-test
   Small synthetic checks for packing/permutation/descriptor invariance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PATCH_TYPED_DIM = 208
GRAPH_TYPED_DIM = 624
ATOM_DIM = 48
BOND_DIM = 13
SHELL_DIM = 3
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
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, ensure_ascii=False, indent=2)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def import_module_from_path(path: Path, module_name: str):
    path = path.expanduser().resolve()
    require_file(path, module_name)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def flatten_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [N] or [N,1], got {labels.shape}")
    labels = labels.astype(np.int64, copy=False)
    if not np.array_equal(np.unique(labels), np.asarray([0, 1], dtype=np.int64)):
        raise ValueError(f"Expected binary labels, got {np.unique(labels).tolist()}")
    return labels


def open_output_memmap(path: Path, shape: Tuple[int, ...], rebuild: bool) -> np.memmap:
    if path.exists() and not rebuild:
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise RuntimeError(
                f"Existing {path.name} mismatch: shape={array.shape}, dtype={array.dtype}; "
                f"expected shape={shape}, float32"
            )
        return array
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )


def split_positions(graph_split: np.ndarray) -> Dict[str, np.ndarray]:
    graph_split = np.asarray(graph_split, dtype=np.int64).reshape(-1)
    return {
        name: np.flatnonzero(graph_split == flag).astype(np.int64)
        for flag, name in SPLIT_NAMES.items()
    }


# =============================================================================
# Load structural export and aligned composition/typed caches
# =============================================================================


def load_structural_export(
    ksvd_module,
    cache_dir: Path,
    result_dir: Path,
    expected_atoms: int,
    expected_sparsity: int,
) -> Dict[str, object]:
    cache = ksvd_module.PilotCache.load(str(cache_dir))

    dictionary_raw = np.load(
        require_file(result_dir / "dictionary_raw.npy", "raw dictionary"),
        mmap_mode="r",
    )
    dictionary_scaled = np.load(
        require_file(result_dir / "dictionary_scaled.npy", "scaled dictionary"),
        mmap_mode="r",
    )
    scaler_npz = np.load(
        require_file(result_dir / "block_scaler.npz", "block scaler"),
        allow_pickle=False,
    )
    scale_vector = np.asarray(scaler_npz["scale_vector"], dtype=np.float32)

    code_indices = np.load(
        require_file(result_dir / "code_indices.npy", "code indices"),
        mmap_mode="r",
    )
    code_values = np.load(
        require_file(result_dir / "code_values.npy", "code values"),
        mmap_mode="r",
    )
    code_nnz = np.load(
        require_file(result_dir / "code_nnz.npy", "code nnz"),
        mmap_mode="r",
    )

    feature_dim, n_atoms = dictionary_raw.shape
    if dictionary_scaled.shape != dictionary_raw.shape:
        raise RuntimeError("dictionary_raw/dictionary_scaled shape mismatch")
    if feature_dim != cache.layout.feature_dim:
        raise RuntimeError(
            f"Dictionary F={feature_dim}, cache F={cache.layout.feature_dim}"
        )
    if scale_vector.shape != (feature_dim,):
        raise RuntimeError(f"scale_vector shape mismatch: {scale_vector.shape}")
    if n_atoms != int(expected_atoms):
        raise RuntimeError(f"Expected K={expected_atoms}, found K={n_atoms}")

    total_patches = int(cache.graph_offsets[-1])
    expected_code_shape = (total_patches, int(expected_sparsity))
    if code_indices.shape != expected_code_shape or code_values.shape != expected_code_shape:
        raise RuntimeError(
            f"Expected code arrays {expected_code_shape}, got "
            f"indices={code_indices.shape}, values={code_values.shape}"
        )
    if code_nnz.shape != (total_patches,):
        raise RuntimeError(f"code_nnz shape mismatch: {code_nnz.shape}")

    return {
        "cache": cache,
        "dictionary_raw": dictionary_raw,
        "dictionary_scaled": dictionary_scaled,
        "scale_vector": scale_vector,
        "code_indices": code_indices,
        "code_values": code_values,
        "code_nnz": code_nnz,
        "n_atoms": int(n_atoms),
        "sparsity": int(expected_sparsity),
    }


def load_aligned_composition(
    composition_cache: Path,
    structural_dataset_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    archive = np.load(require_file(composition_cache, "composition cache"), allow_pickle=False)
    required = {
        "graph_atom_sem",
        "graph_bond_sem",
        "macro",
        "labels",
        "dataset_indices",
    }
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"Composition cache missing keys: {sorted(missing)}")

    source_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
    if len(np.unique(source_indices)) != len(source_indices):
        raise RuntimeError("Duplicate dataset_indices in composition cache")
    lookup = {int(dataset_index): pos for pos, dataset_index in enumerate(source_indices)}
    try:
        order = np.asarray(
            [lookup[int(dataset_index)] for dataset_index in structural_dataset_indices],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"Dataset index missing from composition cache: {exc}") from exc

    composition = np.concatenate(
        [
            np.asarray(archive["graph_atom_sem"], dtype=np.float32)[order],
            np.asarray(archive["graph_bond_sem"], dtype=np.float32)[order],
            np.asarray(archive["macro"], dtype=np.float32)[order],
        ],
        axis=1,
    ).astype(np.float32)
    labels = flatten_labels(np.asarray(archive["labels"])[order])
    if composition.shape != (len(structural_dataset_indices), 69):
        raise RuntimeError(f"Composition shape mismatch: {composition.shape}")
    return {
        "composition": composition,
        "labels": labels,
        "dataset_indices": np.asarray(structural_dataset_indices, dtype=np.int64),
    }


def load_aligned_typed_pool(
    typed_cache: Optional[Path],
    structural_dataset_indices: np.ndarray,
    labels: np.ndarray,
) -> Optional[np.ndarray]:
    if typed_cache is None:
        return None
    archive = np.load(require_file(typed_cache, "historical typed-patch cache"), allow_pickle=False)
    required = {"typed_patch_pool", "dataset_indices", "labels"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"Typed cache missing keys: {sorted(missing)}")
    source_indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
    lookup = {int(dataset_index): pos for pos, dataset_index in enumerate(source_indices)}
    order = np.asarray(
        [lookup[int(dataset_index)] for dataset_index in structural_dataset_indices],
        dtype=np.int64,
    )
    matrix = np.asarray(archive["typed_patch_pool"], dtype=np.float32)[order]
    typed_labels = flatten_labels(np.asarray(archive["labels"])[order])
    if matrix.shape != (len(structural_dataset_indices), GRAPH_TYPED_DIM):
        raise RuntimeError(f"Typed pool shape mismatch: {matrix.shape}")
    if not np.array_equal(typed_labels, labels):
        raise RuntimeError("Typed-patch and composition labels disagree")
    return matrix


# =============================================================================
# Sparse-code helpers
# =============================================================================


def dense_codes_for_rows(
    code_indices: np.ndarray,
    code_values: np.ndarray,
    code_nnz: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    num_rows, width = code_indices.shape
    dense = np.zeros((num_rows, n_atoms), dtype=np.float32)
    slots = np.arange(width, dtype=np.int64)[None, :]
    active = slots < np.asarray(code_nnz, dtype=np.int64)[:, None]
    row_ids, slot_ids = np.nonzero(active)
    atom_ids = np.asarray(code_indices[row_ids, slot_ids], dtype=np.int64)
    values = np.asarray(code_values[row_ids, slot_ids], dtype=np.float32)
    if atom_ids.size:
        if atom_ids.min() < 0 or atom_ids.max() >= n_atoms:
            raise RuntimeError("Sparse code atom index outside dictionary range")
        dense[row_ids, atom_ids] = values
    return dense


def reconstruct_sparse_rows(
    dictionary_raw: np.ndarray,
    local_indices: np.ndarray,
    local_values: np.ndarray,
    local_nnz: np.ndarray,
) -> np.ndarray:
    num_rows, width = local_indices.shape
    feature_dim = dictionary_raw.shape[0]
    reconstructed = np.zeros((num_rows, feature_dim), dtype=np.float32)
    for slot in range(width):
        valid = slot < np.asarray(local_nnz, dtype=np.int64)
        if not np.any(valid):
            continue
        ids = np.asarray(local_indices[valid, slot], dtype=np.int64)
        coeff = np.asarray(local_values[valid, slot], dtype=np.float32)
        reconstructed[valid] += coeff[:, None] * np.asarray(
            dictionary_raw[:, ids].T,
            dtype=np.float32,
        )
    return reconstructed


def atom_full_from_dense_codes(codes: np.ndarray) -> np.ndarray:
    num_patches = codes.shape[0]
    nonzero = np.abs(codes) > 1e-10
    frequency = nonzero.mean(axis=0)
    positive_mass = np.maximum(codes, 0.0).mean(axis=0)
    negative_mass = np.maximum(-codes, 0.0).mean(axis=0)
    rms = np.sqrt(np.square(codes).mean(axis=0))
    max_abs = np.abs(codes).max(axis=0)
    return np.concatenate(
        [
            frequency,
            positive_mass,
            negative_mass,
            rms,
            max_abs,
            np.asarray([np.log1p(num_patches)], dtype=np.float32),
        ]
    ).astype(np.float32)


# =============================================================================
# Vectorized structural-row -> 208-D typed descriptor decoder
# =============================================================================


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    output = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def typed_descriptors_from_structural_rows(
    value_rows: np.ndarray,
    layout_rows: np.ndarray,
    layout,
    edge_threshold: float,
    clip_reconstructed_semantics: bool,
) -> np.ndarray:
    """Decode [B,F] structural rows into [B,208] typed descriptors.

    `layout_rows` supplies exact raw shell/mask metadata. `value_rows` supplies
    topology, bond and atom values and may be either raw x or reconstructed x_hat.
    """
    values = np.asarray(value_rows, dtype=np.float32)
    layout_source = np.asarray(layout_rows, dtype=np.float32)
    if values.ndim != 2 or values.shape != layout_source.shape:
        raise ValueError(
            f"value/layout rows must share [B,F], got {values.shape}/{layout_source.shape}"
        )

    batch = values.shape[0]
    capacity = int(layout.capacity)
    pair_i = np.asarray(layout.pair_i, dtype=np.int64)
    pair_j = np.asarray(layout.pair_j, dtype=np.int64)
    num_pairs = int(layout.num_pairs)

    topology = values[:, layout.block_slice("topology")]
    bond = values[:, layout.block_slice("bond")].reshape(batch, BOND_DIM, num_pairs)
    atom = values[:, layout.block_slice("atom")].reshape(batch, capacity, ATOM_DIM)

    raw_shell = layout_source[:, layout.block_slice("shell")].reshape(
        batch, capacity, SHELL_DIM
    )
    raw_mask = layout_source[:, layout.block_slice("mask")]
    valid = raw_mask > 0.5
    shell_id = np.argmax(raw_shell, axis=2)
    shell1_mask = valid & (shell_id == 1)
    shell2_mask = valid & (shell_id == 2)

    if clip_reconstructed_semantics:
        atom = np.clip(atom, 0.0, 1.0)
        bond = np.clip(bond, 0.0, 1.0)

    root_atom = atom[:, 0, :]
    shell1_count = shell1_mask.sum(axis=1).astype(np.float32)
    shell2_count = shell2_mask.sum(axis=1).astype(np.float32)
    valid_count = valid.sum(axis=1).astype(np.float32)

    shell1_atom = safe_divide(
        np.einsum("bm,bmf->bf", shell1_mask.astype(np.float32), atom),
        shell1_count[:, None],
    )
    shell2_atom = safe_divide(
        np.einsum("bm,bmf->bf", shell2_mask.astype(np.float32), atom),
        shell2_count[:, None],
    )

    pair_valid = valid[:, pair_i] & valid[:, pair_j]
    edge = (topology > float(edge_threshold)) & pair_valid

    si = shell_id[:, pair_i]
    sj = shell_id[:, pair_j]
    root_shell1 = ((si == 0) & (sj == 1)) | ((si == 1) & (sj == 0))
    shell1_internal = (si == 1) & (sj == 1)
    shell1_shell2 = ((si == 1) & (sj == 2)) | ((si == 2) & (sj == 1))
    shell2_internal = (si == 2) & (sj == 2)

    def bond_mean(category: np.ndarray) -> np.ndarray:
        selected = edge & category & pair_valid
        counts = selected.sum(axis=1).astype(np.float32)
        sums = np.einsum("bp,bcp->bc", selected.astype(np.float32), bond)
        return safe_divide(sums, counts[:, None])

    root_shell1_bond = bond_mean(root_shell1)
    shell1_internal_bond = bond_mean(shell1_internal)
    shell1_shell2_bond = bond_mean(shell1_shell2)
    shell2_internal_bond = bond_mean(shell2_internal)

    num_edges = edge.sum(axis=1).astype(np.float32)
    possible_all = valid_count * np.maximum(valid_count - 1.0, 0.0) / 2.0
    patch_density = safe_divide(num_edges, possible_all)

    root_edges = edge & ((pair_i[None, :] == 0) | (pair_j[None, :] == 0))
    root_degree = root_edges.sum(axis=1).astype(np.float32)
    root_degree_ratio = safe_divide(root_degree, np.maximum(valid_count - 1.0, 1.0))

    nonroot = np.maximum(valid_count - 1.0, 1.0)
    shell1_ratio = shell1_count / nonroot
    shell2_ratio = shell2_count / nonroot

    shell1_edges = (edge & shell1_internal).sum(axis=1).astype(np.float32)
    shell2_edges = (edge & shell2_internal).sum(axis=1).astype(np.float32)
    shell12_edges = (edge & shell1_shell2).sum(axis=1).astype(np.float32)
    shell1_possible = shell1_count * np.maximum(shell1_count - 1.0, 0.0) / 2.0
    shell2_possible = shell2_count * np.maximum(shell2_count - 1.0, 0.0) / 2.0
    shell12_possible = shell1_count * shell2_count
    shell1_density = safe_divide(shell1_edges, shell1_possible)
    shell12_density = safe_divide(shell12_edges, shell12_possible)
    shell2_density = safe_divide(shell2_edges, shell2_possible)

    # The direct typed-patch descriptor uses E - V + 1 because every exact B2
    # ego patch is connected. Use the same formula after thresholding x_hat.
    cycle_rank = np.maximum(num_edges - valid_count + 1.0, 0.0)
    cycle_rank_norm = safe_divide(cycle_rank, np.maximum(valid_count, 1.0))

    adjacency = np.zeros((batch, capacity, capacity), dtype=np.float32)
    edge_float = edge.astype(np.float32)
    adjacency[:, pair_i, pair_j] = edge_float
    adjacency[:, pair_j, pair_i] = edge_float
    # trace(A^3)/6; capacity is small and graph batches are one molecule at a time.
    triangles = np.einsum("bij,bjk,bki->b", adjacency, adjacency, adjacency) / 6.0
    triangle_per_node = safe_divide(triangles, np.maximum(valid_count, 1.0))

    topology_descriptor = np.stack(
        [
            np.log1p(valid_count),
            patch_density,
            root_degree_ratio,
            shell1_ratio,
            shell2_ratio,
            shell1_density,
            shell12_density,
            shell2_density,
            cycle_rank_norm,
            triangle_per_node,
        ],
        axis=1,
    ).astype(np.float32)

    shell_sizes = np.stack(
        [np.log1p(shell1_count), np.log1p(shell2_count)],
        axis=1,
    ).astype(np.float32)

    descriptor = np.concatenate(
        [
            root_atom,
            shell1_atom,
            shell2_atom,
            root_shell1_bond,
            shell1_internal_bond,
            shell1_shell2_bond,
            shell2_internal_bond,
            topology_descriptor,
            shell_sizes,
        ],
        axis=1,
    ).astype(np.float32)
    if descriptor.shape != (batch, PATCH_TYPED_DIM):
        raise RuntimeError(f"Typed descriptor shape mismatch: {descriptor.shape}")
    return descriptor


def pool_typed_descriptors(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != PATCH_TYPED_DIM:
        raise ValueError(f"Expected [n,208], got {descriptors.shape}")
    return np.concatenate(
        [descriptors.mean(axis=0), descriptors.std(axis=0), descriptors.max(axis=0)],
        axis=0,
    ).astype(np.float32)


# =============================================================================
# Experiment A/B feature build
# =============================================================================


def feature_paths(out_dir: Path) -> Dict[str, Path]:
    return {
        "raw_decoded_typed_pool": out_dir / "raw_decoded_typed_pool.npy",
        "recon_decoded_typed_pool": out_dir / "recon_decoded_typed_pool.npy",
        "atom_full": out_dir / "atom_full.npy",
        "code_second_moment": out_dir / "code_second_moment.npy",
        "graph_diagnostics": out_dir / "graph_reconstruction_diagnostics.npy",
        "metadata": out_dir / "diagnostic_metadata.npz",
        "manifest": out_dir / "build_manifest.json",
        "progress": out_dir / "build_progress.json",
    }


def run_build(args: argparse.Namespace) -> None:
    ksvd = import_module_from_path(Path(args.ksvd_script), "molhiv_ksvd_core_for_diag")
    export = load_structural_export(
        ksvd_module=ksvd,
        cache_dir=Path(args.ksvd_cache_dir),
        result_dir=Path(args.ksvd_result_dir),
        expected_atoms=args.expected_atoms,
        expected_sparsity=args.expected_sparsity,
    )
    cache = export["cache"]
    n_graphs = len(cache.dataset_indices)
    n_atoms = int(export["n_atoms"])
    upper_i, upper_j = np.triu_indices(n_atoms)
    second_dim = len(upper_i)
    if second_dim != n_atoms * (n_atoms + 1) // 2:
        raise RuntimeError("Second-moment upper-triangle dimension error")

    composition_payload = load_aligned_composition(
        Path(args.composition_cache),
        np.asarray(cache.dataset_indices, dtype=np.int64),
    )
    historical_typed = load_aligned_typed_pool(
        Path(args.typed_patch_cache) if args.typed_patch_cache else None,
        np.asarray(cache.dataset_indices, dtype=np.int64),
        composition_payload["labels"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = feature_paths(out_dir)

    if args.rebuild:
        for path in paths.values():
            if path.exists():
                path.unlink()

    raw_pool = open_output_memmap(
        paths["raw_decoded_typed_pool"], (n_graphs, GRAPH_TYPED_DIM), args.rebuild
    )
    recon_pool = open_output_memmap(
        paths["recon_decoded_typed_pool"], (n_graphs, GRAPH_TYPED_DIM), args.rebuild
    )
    atom_full = open_output_memmap(
        paths["atom_full"], (n_graphs, 5 * n_atoms + 1), args.rebuild
    )
    code_second = open_output_memmap(
        paths["code_second_moment"], (n_graphs, second_dim), args.rebuild
    )
    # Per graph: raw_energy, error_energy, typed_sq_sum, typed_abs_sum,
    # typed_value_count, mask_correct, mask_total, shell_correct, shell_total,
    # graph_energy_retention. Storing additive quantities makes resume-safe
    # global diagnostics possible.
    graph_diag = open_output_memmap(
        paths["graph_diagnostics"], (n_graphs, 10), args.rebuild
    )

    start_graph = 0
    if paths["progress"].exists() and not args.rebuild:
        with paths["progress"].open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        start_graph = int(progress.get("next_graph", 0))
        if start_graph < 0 or start_graph > n_graphs:
            raise RuntimeError(f"Invalid resume graph: {start_graph}")
        print(f"Resuming feature build at graph {start_graph}/{n_graphs}")

    dictionary_raw = np.asarray(export["dictionary_raw"], dtype=np.float32)
    code_indices = export["code_indices"]
    code_values = export["code_values"]
    code_nnz = export["code_nnz"]

    print("\nK-SVD information diagnostics: build A/B features")
    print(f"  graphs              : {n_graphs}")
    print(f"  rooted patches      : {int(cache.graph_offsets[-1])}")
    print(f"  patch capacity/F    : {cache.layout.capacity}/{cache.layout.feature_dim}")
    print(f"  dictionary K/s      : {n_atoms}/{export['sparsity']}")
    print(f"  second moment dim   : {second_dim}")
    print(f"  edge threshold      : {args.edge_threshold}")
    print("  reconstructed layout: raw shell/mask oracle metadata")

    aggregate_retention_num = 0.0
    aggregate_retention_den = 0.0
    aggregate_typed_sq = 0.0
    aggregate_typed_abs = 0.0
    aggregate_typed_count = 0
    mask_correct = 0
    mask_total = 0
    shell_correct = 0
    shell_total = 0

    for graph_pos in range(start_graph, n_graphs):
        start = int(cache.graph_offsets[graph_pos])
        end = int(cache.graph_offsets[graph_pos + 1])
        raw_rows = np.asarray(cache.patches[start:end], dtype=np.float32)
        local_indices = np.asarray(code_indices[start:end])
        local_values = np.asarray(code_values[start:end], dtype=np.float32)
        local_nnz = np.asarray(code_nnz[start:end], dtype=np.int64)

        dense_codes = dense_codes_for_rows(
            local_indices,
            local_values,
            local_nnz,
            n_atoms=n_atoms,
        )
        reconstructed = reconstruct_sparse_rows(
            dictionary_raw,
            local_indices,
            local_values,
            local_nnz,
        )

        raw_desc = typed_descriptors_from_structural_rows(
            raw_rows,
            raw_rows,
            cache.layout,
            edge_threshold=args.edge_threshold,
            clip_reconstructed_semantics=False,
        )
        recon_desc = typed_descriptors_from_structural_rows(
            reconstructed,
            raw_rows,
            cache.layout,
            edge_threshold=args.edge_threshold,
            clip_reconstructed_semantics=True,
        )

        raw_pool[graph_pos] = pool_typed_descriptors(raw_desc)
        recon_pool[graph_pos] = pool_typed_descriptors(recon_desc)
        atom_full[graph_pos] = atom_full_from_dense_codes(dense_codes)
        second = dense_codes.T @ dense_codes / float(max(len(dense_codes), 1))
        code_second[graph_pos] = second[upper_i, upper_j].astype(np.float32)

        error = raw_rows - reconstructed
        raw_energy = float(np.square(raw_rows).sum(dtype=np.float64))
        error_energy = float(np.square(error).sum(dtype=np.float64))
        retention = 1.0 - error_energy / max(raw_energy, 1e-12)
        typed_diff = raw_desc - recon_desc
        typed_rmse = float(np.sqrt(np.mean(np.square(typed_diff), dtype=np.float64)))
        typed_mae = float(np.mean(np.abs(typed_diff), dtype=np.float64))

        raw_mask = raw_rows[:, cache.layout.block_slice("mask")] > 0.5
        recon_mask = reconstructed[:, cache.layout.block_slice("mask")] > 0.5
        local_mask_agreement = float(np.mean(raw_mask == recon_mask))

        raw_shell = raw_rows[:, cache.layout.block_slice("shell")].reshape(
            len(raw_rows), cache.layout.capacity, SHELL_DIM
        )
        recon_shell = reconstructed[:, cache.layout.block_slice("shell")].reshape(
            len(raw_rows), cache.layout.capacity, SHELL_DIM
        )
        valid = raw_mask
        if np.any(valid):
            local_shell_agreement = float(
                np.mean(np.argmax(raw_shell, axis=2)[valid] == np.argmax(recon_shell, axis=2)[valid])
            )
        else:
            local_shell_agreement = 1.0

        local_typed_sq_sum = float(np.square(typed_diff).sum(dtype=np.float64))
        local_typed_abs_sum = float(np.abs(typed_diff).sum(dtype=np.float64))
        local_typed_count = int(typed_diff.size)
        local_mask_correct = int(np.sum(raw_mask == recon_mask))
        local_mask_total = int(raw_mask.size)
        local_shell_correct = int(
            np.sum(
                np.argmax(raw_shell, axis=2)[valid]
                == np.argmax(recon_shell, axis=2)[valid]
            )
        )
        local_shell_total = int(np.sum(valid))
        graph_diag[graph_pos] = np.asarray(
            [
                raw_energy,
                error_energy,
                local_typed_sq_sum,
                local_typed_abs_sum,
                local_typed_count,
                local_mask_correct,
                local_mask_total,
                local_shell_correct,
                local_shell_total,
                retention,
            ],
            dtype=np.float32,
        )

        aggregate_retention_num += error_energy
        aggregate_retention_den += raw_energy
        aggregate_typed_sq += local_typed_sq_sum
        aggregate_typed_abs += local_typed_abs_sum
        aggregate_typed_count += local_typed_count
        mask_correct += local_mask_correct
        mask_total += local_mask_total
        shell_correct += local_shell_correct
        shell_total += local_shell_total

        completed = graph_pos + 1
        if args.progress_every > 0 and (
            completed % args.progress_every == 0 or completed == n_graphs
        ):
            raw_pool.flush()
            recon_pool.flush()
            atom_full.flush()
            code_second.flush()
            graph_diag.flush()
            save_json(paths["progress"], {"next_graph": completed})
            print(f"  built {completed:6d}/{n_graphs}", flush=True)

    raw_pool.flush()
    recon_pool.flush()
    atom_full.flush()
    code_second.flush()
    graph_diag.flush()

    np.savez(
        paths["metadata"],
        dataset_indices=np.asarray(cache.dataset_indices, dtype=np.int64),
        graph_split=np.asarray(cache.graph_split, dtype=np.uint8),
        labels=np.asarray(composition_payload["labels"], dtype=np.int64),
        composition=np.asarray(composition_payload["composition"], dtype=np.float32),
        upper_i=upper_i.astype(np.int16),
        upper_j=upper_j.astype(np.int16),
        n_atoms=np.asarray(n_atoms, dtype=np.int64),
        sparsity=np.asarray(export["sparsity"], dtype=np.int64),
        capacity=np.asarray(cache.layout.capacity, dtype=np.int64),
        feature_dim=np.asarray(cache.layout.feature_dim, dtype=np.int64),
    )

    raw_array = np.asarray(raw_pool)
    recon_array = np.asarray(recon_pool)
    feature_comparison: Dict[str, object] = {
        "raw_vs_reconstructed": compare_feature_matrices(raw_array, recon_array),
    }
    if historical_typed is not None:
        feature_comparison["historical_typed_vs_raw_decoded"] = compare_feature_matrices(
            historical_typed, raw_array
        )
        feature_comparison["historical_typed_vs_reconstructed_decoded"] = compare_feature_matrices(
            historical_typed, recon_array
        )

    # Recompute global diagnostics from the completed memmap so resumed runs
    # include graphs processed before the interruption.
    diag_array = np.asarray(graph_diag, dtype=np.float64)
    total_raw_energy = float(diag_array[:, 0].sum())
    total_error_energy = float(diag_array[:, 1].sum())
    total_typed_sq = float(diag_array[:, 2].sum())
    total_typed_abs = float(diag_array[:, 3].sum())
    total_typed_count = float(diag_array[:, 4].sum())
    total_mask_correct = float(diag_array[:, 5].sum())
    total_mask_count = float(diag_array[:, 6].sum())
    total_shell_correct = float(diag_array[:, 7].sum())
    total_shell_count = float(diag_array[:, 8].sum())
    global_diag = {
        "raw_structural_energy_retention": float(
            1.0 - total_error_energy / max(total_raw_energy, 1e-12)
        ),
        "mean_graph_energy_retention": float(diag_array[:, 9].mean()),
        "typed_patch_rmse": float(
            math.sqrt(total_typed_sq / max(total_typed_count, 1.0))
        ),
        "typed_patch_mae": float(
            total_typed_abs / max(total_typed_count, 1.0)
        ),
        "reconstructed_mask_accuracy": float(
            total_mask_correct / max(total_mask_count, 1.0)
        ),
        "reconstructed_shell_accuracy_on_raw_valid_positions": float(
            total_shell_correct / max(total_shell_count, 1.0)
        ),
    }

    manifest = {
        "experiment": "MolHIV K-SVD information diagnostics A/B",
        "ksvd_cache_dir": str(Path(args.ksvd_cache_dir).resolve()),
        "ksvd_result_dir": str(Path(args.ksvd_result_dir).resolve()),
        "composition_cache": str(Path(args.composition_cache).resolve()),
        "typed_patch_cache": (
            str(Path(args.typed_patch_cache).resolve()) if args.typed_patch_cache else None
        ),
        "n_graphs": n_graphs,
        "n_atoms": n_atoms,
        "sparsity": int(export["sparsity"]),
        "second_moment_dim": second_dim,
        "edge_threshold": float(args.edge_threshold),
        "decoder_layout": "raw shell/mask metadata for raw and reconstructed rows",
        "output_files": {key: str(path.resolve()) for key, path in paths.items()},
        "global_diagnostics": global_diag,
        "feature_comparison": feature_comparison,
    }
    save_json(paths["manifest"], manifest)
    if paths["progress"].exists():
        paths["progress"].unlink()

    print("\nBuild complete")
    for key, value in global_diag.items():
        print(f"  {key:<55}: {value:.8f}")
    if historical_typed is not None:
        print("  historical typed vs raw decoded:")
        for key, value in feature_comparison["historical_typed_vs_raw_decoded"].items():
            print(f"    {key:<20}: {value}")
    print(f"  outputs: {out_dir.resolve()}")


def compare_feature_matrices(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Feature comparison shape mismatch: {a.shape}/{b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    flat_a = a.reshape(-1).astype(np.float64)
    flat_b = b.reshape(-1).astype(np.float64)
    if np.std(flat_a) > 0 and np.std(flat_b) > 0:
        correlation = float(np.corrcoef(flat_a, flat_b)[0, 1])
    else:
        correlation = float("nan")
    return {
        "shape": list(a.shape),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
        "pearson_flat": correlation,
        "exact_equal": bool(np.array_equal(a, b)),
        "sha256_a": sha256_array(a),
        "sha256_b": sha256_array(b),
    }


# =============================================================================
# Experiment A/B classification using the exact old protocol
# =============================================================================


def generic_names(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}.{index:04d}" for index in range(dim)]


def load_built_features(out_dir: Path) -> Dict[str, np.ndarray]:
    paths = feature_paths(out_dir)
    metadata_archive = np.load(require_file(paths["metadata"], "diagnostic metadata"), allow_pickle=False)
    payload: Dict[str, np.ndarray] = {
        "dataset_indices": np.asarray(metadata_archive["dataset_indices"], dtype=np.int64),
        "graph_split": np.asarray(metadata_archive["graph_split"], dtype=np.uint8),
        "labels": np.asarray(metadata_archive["labels"], dtype=np.int64),
        "composition": np.asarray(metadata_archive["composition"], dtype=np.float32),
        "upper_i": np.asarray(metadata_archive["upper_i"], dtype=np.int64),
        "upper_j": np.asarray(metadata_archive["upper_j"], dtype=np.int64),
        "n_atoms": np.asarray(metadata_archive["n_atoms"]),
        "sparsity": np.asarray(metadata_archive["sparsity"]),
        "raw_decoded_typed_pool": np.load(
            require_file(paths["raw_decoded_typed_pool"], "raw decoded typed pool"),
            mmap_mode="r",
        ),
        "recon_decoded_typed_pool": np.load(
            require_file(paths["recon_decoded_typed_pool"], "reconstructed decoded typed pool"),
            mmap_mode="r",
        ),
        "atom_full": np.load(require_file(paths["atom_full"], "atom full"), mmap_mode="r"),
        "code_second_moment": np.load(
            require_file(paths["code_second_moment"], "code second moment"),
            mmap_mode="r",
        ),
    }
    return payload


def build_classification_modes(
    built: Mapping[str, np.ndarray],
    historical_typed: Optional[np.ndarray],
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    composition = np.asarray(built["composition"], dtype=np.float32)
    raw_typed = np.asarray(built["raw_decoded_typed_pool"], dtype=np.float32)
    recon_typed = np.asarray(built["recon_decoded_typed_pool"], dtype=np.float32)
    atom_full = np.asarray(built["atom_full"], dtype=np.float32)
    second = np.asarray(built["code_second_moment"], dtype=np.float32)

    modes: Dict[str, Tuple[np.ndarray, List[str]]] = {
        "composition": (composition, generic_names("composition", composition.shape[1])),
        "raw_decoded_typed": (raw_typed, generic_names("raw_typed", raw_typed.shape[1])),
        "recon_decoded_typed": (
            recon_typed,
            generic_names("recon_typed", recon_typed.shape[1]),
        ),
        "composition_raw_decoded_typed": (
            np.concatenate([composition, raw_typed], axis=1).astype(np.float32),
            generic_names("composition_raw_typed", composition.shape[1] + raw_typed.shape[1]),
        ),
        "composition_recon_decoded_typed": (
            np.concatenate([composition, recon_typed], axis=1).astype(np.float32),
            generic_names("composition_recon_typed", composition.shape[1] + recon_typed.shape[1]),
        ),
        "atom_full": (atom_full, generic_names("atom_full", atom_full.shape[1])),
        "composition_atom_full": (
            np.concatenate([composition, atom_full], axis=1).astype(np.float32),
            generic_names("composition_atom_full", composition.shape[1] + atom_full.shape[1]),
        ),
        "code_second_moment": (
            second,
            generic_names("code_second_moment", second.shape[1]),
        ),
        "composition_code_second_moment": (
            np.concatenate([composition, second], axis=1).astype(np.float32),
            generic_names("composition_code_second", composition.shape[1] + second.shape[1]),
        ),
        "composition_atom_full_code_second_moment": (
            np.concatenate([composition, atom_full, second], axis=1).astype(np.float32),
            generic_names(
                "composition_atom_full_code_second",
                composition.shape[1] + atom_full.shape[1] + second.shape[1],
            ),
        ),
    }
    if historical_typed is not None:
        historical_typed = np.asarray(historical_typed, dtype=np.float32)
        modes["historical_typed"] = (
            historical_typed,
            generic_names("historical_typed", historical_typed.shape[1]),
        )
        modes["composition_historical_typed"] = (
            np.concatenate([composition, historical_typed], axis=1).astype(np.float32),
            generic_names(
                "composition_historical_typed",
                composition.shape[1] + historical_typed.shape[1],
            ),
        )
    return modes


def run_classify(args: argparse.Namespace) -> None:
    classifier = import_module_from_path(
        Path(args.classification_script),
        "molhiv_old_protocol_classifier_for_diag",
    )
    built = load_built_features(Path(args.feature_dir))
    historical_typed = load_aligned_typed_pool(
        Path(args.typed_patch_cache) if args.typed_patch_cache else None,
        np.asarray(built["dataset_indices"], dtype=np.int64),
        flatten_labels(built["labels"]),
    )
    modes = build_classification_modes(built, historical_typed)
    unknown = [mode for mode in args.modes if mode not in modes]
    if unknown:
        raise ValueError(f"Unavailable modes {unknown}; available={sorted(modes)}")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    classifier.set_seed(args.global_seed)
    positions = classifier.split_positions(np.asarray(built["graph_split"], dtype=np.uint8))
    labels = flatten_labels(built["labels"])

    summaries = []
    for mode in args.modes:
        matrix, names = modes[mode]
        summary = classifier.evaluate_mode(
            mode=mode,
            matrix=matrix,
            feature_names=names,
            labels=labels,
            positions=positions,
            trials=args.trials,
            tune_seeds=args.tune_seeds,
            final_seeds=args.final_seeds,
            optuna_seed=args.optuna_seed,
            xgb_n_jobs=args.xgb_n_jobs,
            out_dir=result_dir,
        )
        summaries.append(summary)
        save_json(result_dir / "all_modes_summary.partial.json", {"summaries": summaries})

    summaries_sorted = sorted(
        summaries,
        key=lambda item: float(item["valid_mean"]),
        reverse=True,
    )
    save_json(result_dir / "all_modes_summary.json", {"summaries": summaries_sorted})
    with (result_dir / "all_modes_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "mode",
            "feature_dim",
            "train_mean",
            "train_std",
            "valid_mean",
            "valid_std",
            "test_mean",
            "test_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries_sorted:
            writer.writerow({field: summary.get(field) for field in fields})

    print("\n" + "=" * 104)
    print("Diagnostic mode ranking by mean validation ROC-AUC")
    print("=" * 104)
    for rank, summary in enumerate(summaries_sorted, start=1):
        print(
            f"{rank:02d}. {summary['mode']:<46} dim={summary['feature_dim']:4d} | "
            f"valid={summary['valid_mean']:.6f}±{summary['valid_std']:.6f} | "
            f"test={summary['test_mean']:.6f}±{summary['test_std']:.6f}"
        )


# =============================================================================
# Experiment C permutation audit
# =============================================================================


def unpack_one_row(row: np.ndarray, layout) -> Dict[str, np.ndarray]:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    capacity = layout.capacity
    topology = row[layout.block_slice("topology")]
    topology_matrix = np.zeros((capacity, capacity), dtype=np.float32)
    topology_matrix[layout.pair_i, layout.pair_j] = topology
    topology_matrix[layout.pair_j, layout.pair_i] = topology

    bond_flat = row[layout.block_slice("bond")].reshape(BOND_DIM, layout.num_pairs)
    bond_matrix = np.zeros((BOND_DIM, capacity, capacity), dtype=np.float32)
    bond_matrix[:, layout.pair_i, layout.pair_j] = bond_flat
    bond_matrix[:, layout.pair_j, layout.pair_i] = bond_flat

    return {
        "topology": topology_matrix,
        "bond": bond_matrix,
        "atom": row[layout.block_slice("atom")].reshape(capacity, ATOM_DIM),
        "shell": row[layout.block_slice("shell")].reshape(capacity, SHELL_DIM),
        "mask": row[layout.block_slice("mask")].reshape(capacity),
    }


def pack_one_row(parts: Mapping[str, np.ndarray], layout) -> np.ndarray:
    row = np.zeros(layout.feature_dim, dtype=np.float32)
    topology = np.asarray(parts["topology"], dtype=np.float32)
    bond = np.asarray(parts["bond"], dtype=np.float32)
    row[layout.block_slice("topology")] = topology[layout.pair_i, layout.pair_j]
    row[layout.block_slice("bond")] = bond[:, layout.pair_i, layout.pair_j].reshape(-1)
    row[layout.block_slice("atom")] = np.asarray(parts["atom"], dtype=np.float32).reshape(-1)
    row[layout.block_slice("shell")] = np.asarray(parts["shell"], dtype=np.float32).reshape(-1)
    row[layout.block_slice("mask")] = np.asarray(parts["mask"], dtype=np.float32).reshape(-1)
    return row


def permute_patch_positions(row: np.ndarray, permutation: np.ndarray, layout) -> np.ndarray:
    parts = unpack_one_row(row, layout)
    permutation = np.asarray(permutation, dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(layout.capacity)):
        raise ValueError("Invalid position permutation")
    permuted = {
        "topology": parts["topology"][np.ix_(permutation, permutation)],
        "bond": parts["bond"][:, permutation][:, :, permutation],
        "atom": parts["atom"][permutation],
        "shell": parts["shell"][permutation],
        "mask": parts["mask"][permutation],
    }
    return pack_one_row(permuted, layout)


def exact_order_and_tie_groups(
    ksvd,
    root: int,
    adjacency: Sequence[set],
    edge_semantic_map: Mapping[Tuple[int, int], np.ndarray],
    raw_atom_features: np.ndarray,
    wl_iterations: int,
    capacity: int,
) -> Tuple[List[int], List[List[int]]]:
    shell1, shell2 = ksvd.rooted_shells(root, adjacency)
    nodes = [int(root)] + [int(v) for v in shell1] + [int(v) for v in shell2]
    node_set = set(nodes)
    shell_id = {int(root): 0}
    shell_id.update({int(v): 1 for v in shell1})
    shell_id.update({int(v): 2 for v in shell2})

    labels: Dict[int, str] = {}
    for node in nodes:
        local_degree = len(adjacency[node].intersection(node_set))
        base = (shell_id[node], tuple(int(x) for x in raw_atom_features[node]), local_degree)
        labels[node] = ksvd.stable_hash(base)

    for _ in range(int(wl_iterations)):
        new_labels: Dict[int, str] = {}
        for node in nodes:
            neighbors = []
            for neighbor in adjacency[node].intersection(node_set):
                key = (node, neighbor) if node < neighbor else (neighbor, node)
                bond_ids = tuple(int(x) for x in np.flatnonzero(edge_semantic_map[key]))
                neighbors.append((bond_ids, labels[int(neighbor)]))
            new_labels[node] = ksvd.stable_hash((labels[node], tuple(sorted(neighbors))))
        labels = new_labels

    def tie_key(node: int):
        local_degree = len(adjacency[node].intersection(node_set))
        neighbor_atom_signatures = tuple(
            sorted(
                tuple(int(x) for x in raw_atom_features[n])
                for n in adjacency[node].intersection(node_set)
            )
        )
        return (
            labels[node],
            tuple(int(x) for x in raw_atom_features[node]),
            -local_degree,
            neighbor_atom_signatures,
        )

    ordered_shell1 = sorted((int(v) for v in shell1), key=lambda node: tie_key(node) + (node,))
    ordered_shell2 = sorted((int(v) for v in shell2), key=lambda node: tie_key(node) + (node,))
    ordered_nodes = ([root] + ordered_shell1 + ordered_shell2)[:capacity]
    position = {node: pos for pos, node in enumerate(ordered_nodes)}

    groups: List[List[int]] = []
    for shell_nodes in [ordered_shell1, ordered_shell2]:
        buckets: Dict[Tuple, List[int]] = {}
        for node in shell_nodes:
            if node not in position:
                continue
            buckets.setdefault(tie_key(node), []).append(position[node])
        groups.extend(group for group in buckets.values() if len(group) > 1)
    return ordered_nodes, groups


def support_jaccard(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    sa = set(np.flatnonzero(np.abs(a) > eps).tolist())
    sb = set(np.flatnonzero(np.abs(b) > eps).tolist())
    union = sa | sb
    return float(len(sa & sb) / max(len(union), 1))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.linalg.norm(a - b) <= 1e-12 else 0.0
    return float(np.dot(a, b) / denom)


def run_permutation(args: argparse.Namespace) -> None:
    ksvd = import_module_from_path(Path(args.ksvd_script), "molhiv_ksvd_core_for_perm")
    export = load_structural_export(
        ksvd_module=ksvd,
        cache_dir=Path(args.ksvd_cache_dir),
        result_dir=Path(args.ksvd_result_dir),
        expected_atoms=args.expected_atoms,
        expected_sparsity=args.expected_sparsity,
    )
    cache = export["cache"]
    dictionary = np.asarray(export["dictionary_scaled"], dtype=np.float32)
    scale_vector = np.asarray(export["scale_vector"], dtype=np.float32)
    n_atoms = int(export["n_atoms"])
    rng = np.random.default_rng(args.seed)

    graph_positions = np.arange(len(cache.dataset_indices), dtype=np.int64)
    if args.split != "all":
        inverse_split = {name: flag for flag, name in SPLIT_NAMES.items()}
        graph_positions = cache.split_graph_positions(inverse_split[args.split])
    rng.shuffle(graph_positions)

    dataset = None
    if args.permutation_mode == "exact-ties":
        dataset = ksvd.load_molhiv(args.dataset_root)

    records: List[Dict[str, object]] = []
    collected_patches = 0
    graphs_scanned = 0
    total_tie_groups = 0

    print("\nK-SVD permutation stability audit")
    print(f"  mode              : {args.permutation_mode}")
    print(f"  split             : {args.split}")
    print(f"  target patches    : {args.max_patches}")
    print(f"  permutations/patch: {args.permutations_per_patch}")

    for graph_pos in graph_positions:
        if collected_patches >= args.max_patches:
            break
        graph_pos = int(graph_pos)
        graphs_scanned += 1
        start = int(cache.graph_offsets[graph_pos])
        end = int(cache.graph_offsets[graph_pos + 1])
        root_ids = np.asarray(cache.root_ids[start:end], dtype=np.int64)

        if args.permutation_mode == "exact-ties":
            data = dataset[int(cache.dataset_indices[graph_pos])]
            adjacency, edge_semantic_map = ksvd.build_graph_data(data)
            raw_atom_features = data.x.detach().cpu().numpy().astype(np.int64)

        root_order = np.arange(len(root_ids), dtype=np.int64)
        rng.shuffle(root_order)
        for local_root_pos in root_order:
            if collected_patches >= args.max_patches:
                break
            patch_row = start + int(local_root_pos)
            raw_row = np.asarray(cache.patches[patch_row], dtype=np.float32)
            root_id = int(root_ids[local_root_pos])

            if args.permutation_mode == "exact-ties":
                _, groups = exact_order_and_tie_groups(
                    ksvd=ksvd,
                    root=root_id,
                    adjacency=adjacency,
                    edge_semantic_map=edge_semantic_map,
                    raw_atom_features=raw_atom_features,
                    wl_iterations=int(cache.metadata.get("wl_iterations", 3)),
                    capacity=cache.layout.capacity,
                )
            else:
                shell = raw_row[cache.layout.block_slice("shell")].reshape(
                    cache.layout.capacity, SHELL_DIM
                )
                mask = raw_row[cache.layout.block_slice("mask")] > 0.5
                shell_id = np.argmax(shell, axis=1)
                groups = []
                for shell_value in [1, 2]:
                    positions = np.flatnonzero(mask & (shell_id == shell_value)).tolist()
                    if len(positions) > 1:
                        groups.append(positions)

            if not groups:
                continue
            total_tie_groups += len(groups)
            collected_patches += 1

            local_indices = np.asarray(export["code_indices"][patch_row : patch_row + 1])
            local_values = np.asarray(export["code_values"][patch_row : patch_row + 1], dtype=np.float32)
            local_nnz = np.asarray(export["code_nnz"][patch_row : patch_row + 1], dtype=np.int64)
            original_code = dense_codes_for_rows(
                local_indices, local_values, local_nnz, n_atoms=n_atoms
            )[0]

            original_scaled = raw_row * scale_vector
            original_reconstruction = dictionary @ original_code
            original_retention = 1.0 - float(
                np.square(original_scaled - original_reconstruction).sum(dtype=np.float64)
                / max(np.square(original_scaled).sum(dtype=np.float64), 1e-12)
            )
            original_typed = typed_descriptors_from_structural_rows(
                raw_row[None, :],
                raw_row[None, :],
                cache.layout,
                edge_threshold=0.5,
                clip_reconstructed_semantics=False,
            )[0]

            for perm_id in range(args.permutations_per_patch):
                permutation = np.arange(cache.layout.capacity, dtype=np.int64)
                changed = False
                for group in groups:
                    shuffled = np.asarray(group, dtype=np.int64).copy()
                    rng.shuffle(shuffled)
                    if not np.array_equal(shuffled, np.asarray(group, dtype=np.int64)):
                        changed = True
                    permutation[np.asarray(group, dtype=np.int64)] = shuffled
                if not changed:
                    # Force a non-identity permutation if every random shuffle happened to be identity.
                    group = groups[0]
                    permutation[group[0]], permutation[group[1]] = (
                        permutation[group[1]],
                        permutation[group[0]],
                    )

                permuted_row = permute_patch_positions(raw_row, permutation, cache.layout)
                permuted_scaled = permuted_row * scale_vector
                permuted_code = ksvd.omp_encode(
                    dictionary,
                    permuted_scaled[:, None],
                    sparsity=args.expected_sparsity,
                )[:, 0]
                permuted_reconstruction = dictionary @ permuted_code
                permuted_retention = 1.0 - float(
                    np.square(permuted_scaled - permuted_reconstruction).sum(dtype=np.float64)
                    / max(np.square(permuted_scaled).sum(dtype=np.float64), 1e-12)
                )
                permuted_typed = typed_descriptors_from_structural_rows(
                    permuted_row[None, :],
                    permuted_row[None, :],
                    cache.layout,
                    edge_threshold=0.5,
                    clip_reconstructed_semantics=False,
                )[0]

                records.append(
                    {
                        "graph_pos": graph_pos,
                        "dataset_index": int(cache.dataset_indices[graph_pos]),
                        "root_id": root_id,
                        "patch_row": patch_row,
                        "permutation_id": perm_id,
                        "num_permuted_groups": len(groups),
                        "num_permuted_positions": int(sum(len(group) for group in groups)),
                        "support_jaccard": support_jaccard(original_code, permuted_code),
                        "code_cosine": cosine_similarity(original_code, permuted_code),
                        "code_relative_l2": float(
                            np.linalg.norm(original_code - permuted_code)
                            / max(float(np.linalg.norm(original_code)), 1e-12)
                        ),
                        "original_retention": original_retention,
                        "permuted_retention": permuted_retention,
                        "retention_delta": permuted_retention - original_retention,
                        "typed_descriptor_max_abs_delta": float(
                            np.max(np.abs(original_typed - permuted_typed))
                        ),
                    }
                )

        if args.progress_every > 0 and graphs_scanned % args.progress_every == 0:
            print(
                f"  scanned graphs={graphs_scanned} | tie patches={collected_patches} | "
                f"records={len(records)}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"permutation_{args.permutation_mode}_{args.split}.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    metrics = [
        "support_jaccard",
        "code_cosine",
        "code_relative_l2",
        "original_retention",
        "permuted_retention",
        "retention_delta",
        "typed_descriptor_max_abs_delta",
    ]
    summary: Dict[str, object] = {
        "permutation_mode": args.permutation_mode,
        "split": args.split,
        "graphs_scanned": graphs_scanned,
        "patches_with_permutable_groups": collected_patches,
        "total_groups": total_tie_groups,
        "num_records": len(records),
        "permutations_per_patch": args.permutations_per_patch,
        "seed": args.seed,
    }
    if records:
        for metric in metrics:
            values = np.asarray([float(record[metric]) for record in records], dtype=np.float64)
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    save_json(out_dir / f"permutation_{args.permutation_mode}_{args.split}_summary.json", summary)

    print("\nPermutation audit complete")
    print(f"  graphs scanned       : {graphs_scanned}")
    print(f"  patches with groups  : {collected_patches}")
    print(f"  records              : {len(records)}")
    if records:
        print(f"  support Jaccard mean : {summary['support_jaccard']['mean']:.6f}")
        print(f"  code cosine mean     : {summary['code_cosine']['mean']:.6f}")
        print(f"  relative L2 mean     : {summary['code_relative_l2']['mean']:.6f}")
        print(
            "  typed descriptor max delta (sanity): "
            f"{summary['typed_descriptor_max_abs_delta']['max']:.6g}"
        )
    else:
        print("  No permutable groups found. This itself means the final node-id tie break is rare under the exact key.")
    print(f"  output               : {out_dir.resolve()}")


# =============================================================================
# Self-test
# =============================================================================


def run_self_test(args: argparse.Namespace) -> None:
    ksvd = import_module_from_path(Path(args.ksvd_script), "molhiv_ksvd_core_self_test")
    layout = ksvd.PatchLayout.create(5)
    row = np.zeros(layout.feature_dim, dtype=np.float32)
    # valid root + two shell1 + one shell2
    atom = row[layout.block_slice("atom")].reshape(5, ATOM_DIM)
    shell = row[layout.block_slice("shell")].reshape(5, SHELL_DIM)
    mask = row[layout.block_slice("mask")]
    atom[0, 2] = 1
    atom[1, 3] = 1
    atom[2, 4] = 1
    atom[3, 5] = 1
    shell[0, 0] = 1
    shell[1:3, 1] = 1
    shell[3, 2] = 1
    mask[:4] = 1

    pair_lookup = {(int(i), int(j)): idx for idx, (i, j) in enumerate(zip(layout.pair_i, layout.pair_j))}
    topology = row[layout.block_slice("topology")]
    bond = row[layout.block_slice("bond")].reshape(BOND_DIM, layout.num_pairs)
    for u, v in [(0, 1), (0, 2), (1, 2), (1, 3)]:
        pid = pair_lookup[(min(u, v), max(u, v))]
        topology[pid] = 1
        bond[0, pid] = 1

    descriptor = typed_descriptors_from_structural_rows(
        row[None, :], row[None, :], layout, 0.5, False
    )[0]
    permutation = np.asarray([0, 2, 1, 3, 4], dtype=np.int64)
    permuted = permute_patch_positions(row, permutation, layout)
    permuted_descriptor = typed_descriptors_from_structural_rows(
        permuted[None, :], permuted[None, :], layout, 0.5, False
    )[0]
    if not np.allclose(descriptor, permuted_descriptor, atol=1e-7):
        raise AssertionError(
            f"Typed descriptor is not invariant; max delta={np.max(np.abs(descriptor-permuted_descriptor))}"
        )
    roundtrip = pack_one_row(unpack_one_row(row, layout), layout)
    if not np.array_equal(row, roundtrip):
        raise AssertionError("pack/unpack roundtrip failed")
    print("self-test passed")


# =============================================================================
# CLI
# =============================================================================


def add_common_structural_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ksvd-script",
        default="./molhiv_online_structural_ksvd_full.py",
        help="Path to the structural K-SVD script used to build the cache.",
    )
    parser.add_argument("--ksvd-cache-dir", required=True)
    parser.add_argument("--ksvd-result-dir", required=True)
    parser.add_argument("--expected-atoms", type=int, default=64)
    parser.add_argument("--expected-sparsity", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MolHIV K-SVD information-loss, second-moment and permutation diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build Experiment A/B graph features")
    add_common_structural_args(build)
    build.add_argument("--composition-cache", required=True)
    build.add_argument(
        "--typed-patch-cache",
        default=None,
        help="Optional historical components_molhiv_direct_typed_patch_pool...npz",
    )
    build.add_argument("--out-dir", required=True)
    build.add_argument("--edge-threshold", type=float, default=0.5)
    build.add_argument("--progress-every", type=int, default=500)
    build.add_argument("--rebuild", action="store_true")
    build.set_defaults(func=run_build)

    classify = subparsers.add_parser(
        "classify", help="Classify Experiment A/B features with exact old protocol"
    )
    classify.add_argument(
        "--classification-script",
        default="./run_molhiv_shared_atom_classification_old_protocol.py",
    )
    classify.add_argument("--feature-dir", required=True)
    classify.add_argument("--typed-patch-cache", default=None)
    classify.add_argument("--result-dir", required=True)
    classify.add_argument(
        "--modes",
        nargs="+",
        default=[
            "composition_historical_typed",
            "composition_raw_decoded_typed",
            "composition_recon_decoded_typed",
            "composition_atom_full",
            "composition_code_second_moment",
            "composition_atom_full_code_second_moment",
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

    permutation = subparsers.add_parser("permutation", help="Run Experiment C")
    add_common_structural_args(permutation)
    permutation.add_argument("--dataset-root", default="./data/ogb")
    permutation.add_argument("--out-dir", required=True)
    permutation.add_argument(
        "--permutation-mode",
        choices=["exact-ties", "all-within-shell"],
        default="exact-ties",
    )
    permutation.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    permutation.add_argument("--max-patches", type=int, default=1000)
    permutation.add_argument("--permutations-per-patch", type=int, default=3)
    permutation.add_argument("--seed", type=int, default=2026)
    permutation.add_argument("--progress-every", type=int, default=500)
    permutation.set_defaults(func=run_permutation)

    self_test = subparsers.add_parser("self-test", help="Run synthetic helper tests")
    self_test.add_argument(
        "--ksvd-script",
        default="./molhiv_online_structural_ksvd_full.py",
    )
    self_test.set_defaults(func=run_self_test)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
