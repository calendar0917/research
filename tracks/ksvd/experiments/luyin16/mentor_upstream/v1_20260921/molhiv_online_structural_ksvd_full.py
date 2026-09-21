#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolHIV shared ONLINE structural atoms with OMP + mini-batch K-SVD.

Supports both the original 2000-graph pilot and the leakage-safe official-full mode:
learn the dictionary only from the official train split, while official valid/test are
cache-only evaluation splits.

This script deliberately does NOT factorize the 208-D pooled/statistical typed-patch
vector.  The factorized object is a fixed-coordinate typed rooted structural patch.

For each root v, construct its exact radius-2 ego patch and place it in one common
capacity M with this deterministic coordinate convention:

    position 0        : root
    following slots   : shell-1 nodes in rooted typed-WL order
    following slots   : shell-2 nodes in rooted typed-WL order
    remaining slots   : zero padding

Each patch is flattened into one column x_{G,v} containing:

    1 topology upper-triangle channel
    13 bond-semantic upper-triangle channels
    M x 48 compact atom-semantic coordinates
    M x 3 root/shell-1/shell-2 indicators
    M valid-position mask

For a batch of patches:

    X_B in R^[F, N_B]
    D   in R^[F, K]       shared across every graph
    C_B in R^[K, N_B]     patch-specific sparse codes

and the objective is

    min ||X_B - D C_B||_F^2
    s.t. ||c_i||_0 <= sparsity, ||d_k||_2 = 1.

Sparse coding is Orthogonal Matching Pursuit (OMP).  Dictionary atoms are updated
with the K-SVD residual rank-1 SVD rule.  A small replay reservoir of patch row
indices limits catastrophic forgetting without collecting the whole dataset into
one optimization matrix.

Subcommands
-----------
1) build-cache
   --dataset-scope pilot: original 2000-graph train-internal pilot.
   --dataset-scope official-full: cache official train/valid/test; determine the
   common capacity from official train only.

2) train
   Learn the shared dictionary online, report raw/scaled matrix-energy retention,
   save unpacked atom tensors, and save every pilot patch's sparse encoding as
   fixed-width code_indices/code_values arrays.

3) self-test
   Run a synthetic sparse-factorization sanity test without OGB.

Typical commands
----------------
python -u molhiv_online_structural_ksvd_pilot.py build-cache \
  --dataset-root ./data/ogb \
  --cache-dir ./cache/molhiv_structural_ksvd_pilot2000 \
  --n-graphs 2000 --patch-size-policy max

python -u molhiv_online_structural_ksvd_pilot.py train \
  --cache-dir ./cache/molhiv_structural_ksvd_pilot2000 \
  --out-dir ./results/molhiv_structural_ksvd_pilot2000_seed0 \
  --seed 0 --n-atoms 32 --sparsity 4

Notes
-----
* K-SVD atoms and codes are signed, as in the original K-SVD formulation.
* Atom matrices are continuous basis patterns, not guaranteed to be legal molecules.
* The final nearest-real-patch metadata makes every learned atom inspectable.
* The node-order tie-break uses node id only after rooted typed-WL refinement.  Exact
  canonical graph labelling is a separate methodological issue and should be audited
  later; molecular typing resolves most practical ties in this pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.linear_model import orthogonal_mp
except Exception as exc:  # pragma: no cover - clear runtime message
    raise ImportError(
        "scikit-learn is required for OMP. Install it in the experiment environment."
    ) from exc


# =============================================================================
# Constants: compact semantics exactly match the previous MolHIV typed work
# =============================================================================

DATASET_NAME = "ogbg-molhiv"

ELEMENT_GROUPS = [
    ("H", {1}),
    ("B", {5}),
    ("C", {6}),
    ("N", {7}),
    ("O", {8}),
    ("F", {9}),
    ("Si", {14}),
    ("P", {15}),
    ("S", {16}),
    ("Cl", {17}),
    ("Se", {34}),
    ("Br", {35}),
    ("I", {53}),
    ("other", set()),
]

ATOM_SEMANTIC_NAMES = (
    [f"element={name}" for name, _ in ELEMENT_GROUPS]
    + [f"chirality={index}" for index in range(5)]
    + [
        "degree=0",
        "degree=1",
        "degree=2",
        "degree=3",
        "degree=4",
        "degree>=5_or_misc",
    ]
    + [
        "formal_charge<0",
        "formal_charge=0",
        "formal_charge>0",
        "formal_charge=misc",
    ]
    + [
        "numH=0",
        "numH=1",
        "numH=2",
        "numH=3",
        "numH>=4_or_misc",
    ]
    + [
        "radical_e=0",
        "radical_e=1",
        "radical_e>=2",
        "radical_e=misc",
    ]
    + [f"hybridization={index}" for index in range(6)]
    + ["aromatic=False", "aromatic=True"]
    + ["in_ring=False", "in_ring=True"]
)

BOND_SEMANTIC_NAMES = (
    [f"bond_type={index}" for index in range(5)]
    + [f"bond_stereo={index}" for index in range(6)]
    + ["conjugated=False", "conjugated=True"]
)

ATOM_DIM = len(ATOM_SEMANTIC_NAMES)  # 48
BOND_DIM = len(BOND_SEMANTIC_NAMES)  # 13
SHELL_DIM = 3

if ATOM_DIM != 48 or BOND_DIM != 13:
    raise RuntimeError("Unexpected compact semantic dimensions.")

SPLIT_TRAIN = 0
SPLIT_VALID = 1
SPLIT_TEST = 2

SPLIT_NAMES = {
    SPLIT_TRAIN: "train",
    SPLIT_VALID: "valid",
    SPLIT_TEST: "test",
}


# =============================================================================
# Reproducibility and JSON helpers
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(dict(payload)), handle, ensure_ascii=False, indent=2)


# =============================================================================
# Compact atom/bond semantic encoders
# =============================================================================


def one_hot_index(index: int, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.uint8)
    index = int(index)
    if 0 <= index < size:
        output[index] = 1
    else:
        output[-1] = 1
    return output


def atomic_number_group(raw_atomic_index: int) -> int:
    raw_atomic_index = int(raw_atomic_index)
    if 0 <= raw_atomic_index < 118:
        atomic_number = raw_atomic_index + 1
        for group_id, (_, atomic_numbers) in enumerate(ELEMENT_GROUPS[:-1]):
            if atomic_number in atomic_numbers:
                return group_id
    return len(ELEMENT_GROUPS) - 1


def compact_atom_semantics(raw_atom_features: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_atom_features, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f"Expected OGB atom features [N,9], got {raw.shape}")

    output = np.zeros((raw.shape[0], ATOM_DIM), dtype=np.uint8)

    for node_id, feature in enumerate(raw):
        pieces: List[np.ndarray] = []
        pieces.append(one_hot_index(atomic_number_group(feature[0]), len(ELEMENT_GROUPS)))
        pieces.append(one_hot_index(feature[1], 5))

        degree_index = int(feature[2])
        pieces.append(one_hot_index(degree_index if 0 <= degree_index <= 4 else 5, 6))

        charge_index = int(feature[3])
        if charge_index == 11:
            charge_bucket = 3
        elif charge_index < 5:
            charge_bucket = 0
        elif charge_index == 5:
            charge_bucket = 1
        else:
            charge_bucket = 2
        pieces.append(one_hot_index(charge_bucket, 4))

        num_h_index = int(feature[4])
        pieces.append(one_hot_index(num_h_index if 0 <= num_h_index <= 3 else 4, 5))

        radical_index = int(feature[5])
        if radical_index == 5:
            radical_bucket = 3
        elif radical_index == 0:
            radical_bucket = 0
        elif radical_index == 1:
            radical_bucket = 1
        else:
            radical_bucket = 2
        pieces.append(one_hot_index(radical_bucket, 4))

        pieces.append(one_hot_index(feature[6], 6))
        pieces.append(one_hot_index(feature[7], 2))
        pieces.append(one_hot_index(feature[8], 2))

        row = np.concatenate(pieces).astype(np.uint8)
        if row.shape != (ATOM_DIM,):
            raise RuntimeError(f"Atom semantic dimension mismatch: {row.shape}")
        output[node_id] = row

    return output


def compact_bond_semantics(raw_bond_feature: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_bond_feature, dtype=np.int64).reshape(-1)
    if raw.shape != (3,):
        raise ValueError(f"Expected OGB bond feature [3], got {raw.shape}")
    return np.concatenate(
        [
            one_hot_index(raw[0], 5),
            one_hot_index(raw[1], 6),
            one_hot_index(raw[2], 2),
        ]
    ).astype(np.uint8)


# =============================================================================
# Typed rooted structural patch layout
# =============================================================================


@dataclass(frozen=True)
class PatchLayout:
    capacity: int
    pair_i: np.ndarray
    pair_j: np.ndarray
    blocks: Dict[str, Tuple[int, int]]
    feature_dim: int

    @property
    def num_pairs(self) -> int:
        return int(self.pair_i.shape[0])

    @classmethod
    def create(cls, capacity: int) -> "PatchLayout":
        capacity = int(capacity)
        if capacity <= 1:
            raise ValueError("Patch capacity must be at least 2.")

        pair_i, pair_j = np.triu_indices(capacity, k=1)
        num_pairs = len(pair_i)

        cursor = 0
        blocks: Dict[str, Tuple[int, int]] = {}

        blocks["topology"] = (cursor, cursor + num_pairs)
        cursor += num_pairs

        blocks["bond"] = (cursor, cursor + BOND_DIM * num_pairs)
        cursor += BOND_DIM * num_pairs

        blocks["atom"] = (cursor, cursor + capacity * ATOM_DIM)
        cursor += capacity * ATOM_DIM

        blocks["shell"] = (cursor, cursor + capacity * SHELL_DIM)
        cursor += capacity * SHELL_DIM

        blocks["mask"] = (cursor, cursor + capacity)
        cursor += capacity

        return cls(
            capacity=capacity,
            pair_i=pair_i.astype(np.int64),
            pair_j=pair_j.astype(np.int64),
            blocks=blocks,
            feature_dim=int(cursor),
        )

    def block_slice(self, name: str) -> slice:
        start, end = self.blocks[name]
        return slice(int(start), int(end))

    def to_metadata(self) -> Dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "num_pairs": int(self.num_pairs),
            "feature_dim": int(self.feature_dim),
            "blocks": {name: [int(a), int(b)] for name, (a, b) in self.blocks.items()},
            "atom_semantic_dim": ATOM_DIM,
            "bond_semantic_dim": BOND_DIM,
            "shell_dim": SHELL_DIM,
            "atom_semantic_names": ATOM_SEMANTIC_NAMES,
            "bond_semantic_names": BOND_SEMANTIC_NAMES,
            "coordinate_convention": (
                "root first; shell1 then shell2; each shell ordered by rooted typed-WL; "
                "upper-triangle edge channels; zero padding"
            ),
        }


# =============================================================================
# Graph extraction and deterministic rooted ordering
# =============================================================================


def build_graph_data(data) -> Tuple[List[set], Dict[Tuple[int, int], np.ndarray]]:
    num_nodes = int(data.num_nodes)
    adjacency: List[set] = [set() for _ in range(num_nodes)]
    edge_semantic_map: Dict[Tuple[int, int], np.ndarray] = {}

    edge_index = data.edge_index.detach().cpu().numpy()
    edge_attr = data.edge_attr.detach().cpu().numpy()

    for edge_id in range(edge_index.shape[1]):
        u = int(edge_index[0, edge_id])
        v = int(edge_index[1, edge_id])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        semantic = compact_bond_semantics(edge_attr[edge_id])
        previous = edge_semantic_map.get(key)
        if previous is not None:
            if not np.array_equal(previous, semantic):
                raise ValueError(f"Bidirectional bond attributes disagree for edge {key}")
            continue
        edge_semantic_map[key] = semantic
        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency, edge_semantic_map


def rooted_shells(root: int, adjacency: Sequence[set]) -> Tuple[List[int], List[int]]:
    shell1 = sorted(int(node) for node in adjacency[root])
    shell1_set = set(shell1)
    shell2_set = set()
    for node in shell1:
        shell2_set.update(adjacency[node])
    shell2_set.discard(root)
    shell2_set.difference_update(shell1_set)
    return shell1, sorted(int(node) for node in shell2_set)


def stable_hash(payload: object) -> str:
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def rooted_typed_wl_order(
    root: int,
    shell1: Sequence[int],
    shell2: Sequence[int],
    adjacency: Sequence[set],
    edge_semantic_map: Mapping[Tuple[int, int], np.ndarray],
    raw_atom_features: np.ndarray,
    wl_iterations: int = 3,
) -> Tuple[List[int], List[int]]:
    """
    Produce one deterministic coordinate order shared across graphs.

    The final node-id tie-break is used only after rooted shell, atom categories,
    local degree, bond categories, and several WL refinement rounds agree.
    """
    nodes = [int(root)] + [int(v) for v in shell1] + [int(v) for v in shell2]
    node_set = set(nodes)
    shell_id = {int(root): 0}
    shell_id.update({int(v): 1 for v in shell1})
    shell_id.update({int(v): 2 for v in shell2})

    labels: Dict[int, str] = {}
    for node in nodes:
        local_degree = len(adjacency[node].intersection(node_set))
        base = (shell_id[node], tuple(int(x) for x in raw_atom_features[node]), local_degree)
        labels[node] = stable_hash(base)

    for _ in range(int(wl_iterations)):
        new_labels: Dict[int, str] = {}
        for node in nodes:
            neighbors = []
            for neighbor in adjacency[node].intersection(node_set):
                key = (node, neighbor) if node < neighbor else (neighbor, node)
                bond_ids = tuple(int(x) for x in np.flatnonzero(edge_semantic_map[key]))
                neighbors.append((bond_ids, labels[int(neighbor)]))
            new_labels[node] = stable_hash((labels[node], tuple(sorted(neighbors))))
        labels = new_labels

    def key(node: int):
        local_degree = len(adjacency[node].intersection(node_set))
        neighbor_atom_signatures = tuple(
            sorted(tuple(int(x) for x in raw_atom_features[n]) for n in adjacency[node].intersection(node_set))
        )
        return (
            labels[node],
            tuple(int(x) for x in raw_atom_features[node]),
            -local_degree,
            neighbor_atom_signatures,
            int(node),
        )

    return sorted((int(v) for v in shell1), key=key), sorted((int(v) for v in shell2), key=key)


def exact_radius2_size(root: int, adjacency: Sequence[set]) -> int:
    shell1, shell2 = rooted_shells(root, adjacency)
    return 1 + len(shell1) + len(shell2)


def extract_structural_patch_matrix(
    data,
    layout: PatchLayout,
    wl_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    patches : uint8 [num_nodes, F]
    root_ids : int64 [num_nodes]
    full_sizes : int32 [num_nodes]
    """
    adjacency, edge_semantic_map = build_graph_data(data)
    raw_atom = data.x.detach().cpu().numpy().astype(np.int64)
    atom_sem = compact_atom_semantics(raw_atom)

    num_nodes = int(data.num_nodes)
    patches = np.zeros((num_nodes, layout.feature_dim), dtype=np.uint8)
    full_sizes = np.zeros(num_nodes, dtype=np.int32)

    pair_lookup = -np.ones((layout.capacity, layout.capacity), dtype=np.int64)
    pair_lookup[layout.pair_i, layout.pair_j] = np.arange(layout.num_pairs, dtype=np.int64)

    topology_start, _ = layout.blocks["topology"]
    bond_start, _ = layout.blocks["bond"]
    atom_start, _ = layout.blocks["atom"]
    shell_start, _ = layout.blocks["shell"]
    mask_start, _ = layout.blocks["mask"]

    for root in range(num_nodes):
        shell1, shell2 = rooted_shells(root, adjacency)
        shell1, shell2 = rooted_typed_wl_order(
            root=root,
            shell1=shell1,
            shell2=shell2,
            adjacency=adjacency,
            edge_semantic_map=edge_semantic_map,
            raw_atom_features=raw_atom,
            wl_iterations=wl_iterations,
        )

        full_size = 1 + len(shell1) + len(shell2)
        full_sizes[root] = full_size

        nodes = [root] + shell1 + shell2
        shells = [0] + [1] * len(shell1) + [2] * len(shell2)

        nodes = nodes[: layout.capacity]
        shells = shells[: layout.capacity]
        valid_len = len(nodes)

        row = patches[root]

        # Position-wise atom, shell and mask channels.
        for pos, (node, shell) in enumerate(zip(nodes, shells)):
            a0 = atom_start + pos * ATOM_DIM
            row[a0 : a0 + ATOM_DIM] = atom_sem[node]

            s0 = shell_start + pos * SHELL_DIM
            row[s0 + int(shell)] = 1
            row[mask_start + pos] = 1

        # Upper-triangle topology and 13 bond-semantic channels.
        for i in range(valid_len):
            u = int(nodes[i])
            for j in range(i + 1, valid_len):
                v = int(nodes[j])
                key = (u, v) if u < v else (v, u)
                bond_sem = edge_semantic_map.get(key)
                if bond_sem is None:
                    continue

                pair_id = int(pair_lookup[i, j])
                row[topology_start + pair_id] = 1
                active_bond_ids = np.flatnonzero(bond_sem)
                for bond_id in active_bond_ids:
                    row[bond_start + int(bond_id) * layout.num_pairs + pair_id] = 1

    return patches, np.arange(num_nodes, dtype=np.int64), full_sizes


# =============================================================================
# OGB loading and pilot selection
# =============================================================================


def load_molhiv(dataset_root: str):
    import torch

    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_compat
    from ogb.graphproppred import PygGraphPropPredDataset

    return PygGraphPropPredDataset(name=DATASET_NAME, root=dataset_root)


def as_numpy_indices(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64).reshape(-1)


def size_stratified_pilot_selection(
    dataset,
    official_train_indices: np.ndarray,
    n_graphs: int,
    n_bins: int,
    valid_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_graphs > len(official_train_indices):
        raise ValueError("Requested more pilot graphs than the official train split contains.")

    rng = np.random.default_rng(seed)
    sizes = np.zeros(len(official_train_indices), dtype=np.int64)

    print(f"Scanning {len(official_train_indices)} official-train graph sizes...")
    for pos, dataset_index in enumerate(official_train_indices):
        sizes[pos] = int(dataset[int(dataset_index)].num_nodes)
        if (pos + 1) % 5000 == 0:
            print(f"  size scan {pos + 1}/{len(official_train_indices)}")

    rank_order = np.argsort(sizes, kind="stable")
    rank_bins = np.array_split(rank_order, int(n_bins))

    base = n_graphs // n_bins
    remainder = n_graphs % n_bins

    selected: List[int] = []
    split_flags: List[int] = []
    selected_bin_ids: List[int] = []

    for bin_id, rank_positions in enumerate(rank_bins):
        target = base + (1 if bin_id < remainder else 0)
        if target > len(rank_positions):
            raise RuntimeError(f"Size bin {bin_id} is too small for target={target}.")

        chosen_positions = rng.choice(rank_positions, size=target, replace=False)
        chosen_indices = official_train_indices[chosen_positions].astype(np.int64)
        rng.shuffle(chosen_indices)

        valid_count = int(round(target * valid_fraction))
        valid_count = min(max(valid_count, 1), target - 1)

        for local_pos, dataset_index in enumerate(chosen_indices):
            selected.append(int(dataset_index))
            split_flags.append(SPLIT_VALID if local_pos < valid_count else SPLIT_TRAIN)
            selected_bin_ids.append(int(bin_id))

    selected_array = np.asarray(selected, dtype=np.int64)
    split_array = np.asarray(split_flags, dtype=np.uint8)
    bin_array = np.asarray(selected_bin_ids, dtype=np.int16)

    # Shuffle graph order while keeping all aligned metadata.
    permutation = rng.permutation(len(selected_array))
    return selected_array[permutation], split_array[permutation], bin_array[permutation]


def official_full_selection(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all official MolHIV graphs in train -> valid -> test order."""
    official = dataset.get_idx_split()
    train = as_numpy_indices(official["train"])
    valid = as_numpy_indices(official["valid"])
    test = as_numpy_indices(official["test"])

    selected = np.concatenate([train, valid, test]).astype(np.int64)
    split_flags = np.concatenate(
        [
            np.full(len(train), SPLIT_TRAIN, dtype=np.uint8),
            np.full(len(valid), SPLIT_VALID, dtype=np.uint8),
            np.full(len(test), SPLIT_TEST, dtype=np.uint8),
        ]
    )
    # Size bins are not used in official-full mode, but keep an aligned array.
    bin_ids = np.full(len(selected), -1, dtype=np.int16)
    return selected, split_flags, bin_ids


def split_truncation_report(
    graph_offsets: np.ndarray,
    split_flags: np.ndarray,
    full_patch_sizes: np.ndarray,
    capacity: int,
) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for split_flag, split_name in SPLIT_NAMES.items():
        graph_positions = np.flatnonzero(split_flags == split_flag)
        roots = 0
        truncated = 0
        sizes: List[np.ndarray] = []
        for graph_pos in graph_positions:
            start = int(graph_offsets[graph_pos])
            end = int(graph_offsets[graph_pos + 1])
            values = np.asarray(full_patch_sizes[start:end], dtype=np.int32)
            roots += len(values)
            truncated += int(np.sum(values > capacity))
            if len(values):
                sizes.append(values)
        if sizes:
            joined = np.concatenate(sizes)
            report[split_name] = {
                "num_graphs": int(len(graph_positions)),
                "num_roots": int(roots),
                "truncated_roots": int(truncated),
                "truncation_rate": float(truncated / max(roots, 1)),
                "b2_min": int(joined.min()),
                "b2_mean": float(joined.mean()),
                "b2_p95": float(np.quantile(joined, 0.95)),
                "b2_p99": float(np.quantile(joined, 0.99)),
                "b2_max": int(joined.max()),
            }
        else:
            report[split_name] = {
                "num_graphs": 0,
                "num_roots": 0,
                "truncated_roots": 0,
                "truncation_rate": 0.0,
            }
    return report


# =============================================================================
# Cache construction
# =============================================================================


def choose_capacity(
    sizes: np.ndarray,
    policy: str,
    quantile: float,
    explicit_capacity: Optional[int],
) -> int:
    if explicit_capacity is not None:
        capacity = int(explicit_capacity)
    elif policy == "max":
        capacity = int(np.max(sizes))
    elif policy == "quantile":
        capacity = int(math.ceil(float(np.quantile(sizes, quantile))))
    else:
        raise ValueError(f"Unknown patch-size policy: {policy}")

    if capacity <= 1:
        raise RuntimeError(f"Invalid chosen capacity={capacity}")
    return capacity


def build_cache(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_molhiv(args.dataset_root)
    official = dataset.get_idx_split()
    official_train = as_numpy_indices(official["train"])

    if args.dataset_scope == "pilot":
        selected_indices, split_flags, bin_ids = size_stratified_pilot_selection(
            dataset=dataset,
            official_train_indices=official_train,
            n_graphs=args.n_graphs,
            n_bins=args.n_bins,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )
    elif args.dataset_scope == "official-full":
        selected_indices, split_flags, bin_ids = official_full_selection(dataset)
    else:
        raise ValueError(f"Unknown dataset scope: {args.dataset_scope}")

    # Save split identity immediately. This also makes interrupted builds inspectable.
    np.save(cache_dir / "selected_dataset_indices.npy", selected_indices)
    np.save(cache_dir / "graph_split.npy", split_flags)
    np.save(cache_dir / "graph_size_bin.npy", bin_ids)

    print(f"Scanning node counts for {len(selected_indices)} cached graphs...")
    graph_num_nodes = np.zeros(len(selected_indices), dtype=np.int64)
    for graph_pos, dataset_index in enumerate(selected_indices):
        graph_num_nodes[graph_pos] = int(dataset[int(dataset_index)].num_nodes)
        if (graph_pos + 1) % 5000 == 0:
            print(f"  node-count scan {graph_pos + 1}/{len(selected_indices)}")

    graph_offsets = np.zeros(len(selected_indices) + 1, dtype=np.int64)
    graph_offsets[1:] = np.cumsum(graph_num_nodes)
    total_patches = int(graph_offsets[-1])
    np.save(cache_dir / "graph_num_nodes.npy", graph_num_nodes)
    np.save(cache_dir / "graph_offsets.npy", graph_offsets)

    plan_path = cache_dir / "build_plan.json"
    if args.resume and plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        capacity = int(plan["capacity"])
        layout = PatchLayout.create(capacity)
        if int(plan["feature_dim"]) != layout.feature_dim:
            raise RuntimeError("Existing build plan has an incompatible feature dimension.")
        print(f"Resuming with existing train-derived capacity M={capacity}, F={layout.feature_dim}")
    else:
        # IMPORTANT: capacity is estimated from dictionary-train graphs only.
        capacity_graphs = np.flatnonzero(split_flags == SPLIT_TRAIN).astype(np.int64)
        train_patch_total = int(np.sum(graph_num_nodes[capacity_graphs]))
        train_patch_sizes = np.zeros(train_patch_total, dtype=np.int32)
        cursor = 0
        print(
            "\nAuditing exact radius-2 sizes on dictionary-train graphs only "
            f"({len(capacity_graphs)} graphs, {train_patch_total} roots)..."
        )
        for local_pos, graph_pos in enumerate(capacity_graphs):
            data = dataset[int(selected_indices[int(graph_pos)])]
            adjacency, _ = build_graph_data(data)
            for root in range(int(data.num_nodes)):
                train_patch_sizes[cursor] = exact_radius2_size(root, adjacency)
                cursor += 1
            if (local_pos + 1) % 1000 == 0:
                print(f"  train capacity audit {local_pos + 1}/{len(capacity_graphs)}")

        if cursor != train_patch_total:
            raise RuntimeError(f"Capacity audit root mismatch: expected {train_patch_total}, got {cursor}")

        capacity = choose_capacity(
            sizes=train_patch_sizes,
            policy=args.patch_size_policy,
            quantile=args.patch_size_quantile,
            explicit_capacity=args.patch_capacity,
        )
        layout = PatchLayout.create(capacity)
        train_truncated = int(np.sum(train_patch_sizes > capacity))
        train_truncation_rate = train_truncated / max(len(train_patch_sizes), 1)

        plan = {
            "dataset_scope": args.dataset_scope,
            "capacity": int(capacity),
            "feature_dim": int(layout.feature_dim),
            "train_capacity_audit": {
                "num_graphs": int(len(capacity_graphs)),
                "num_roots": int(len(train_patch_sizes)),
                "b2_min": int(train_patch_sizes.min()),
                "b2_mean": float(train_patch_sizes.mean()),
                "b2_p95": float(np.quantile(train_patch_sizes, 0.95)),
                "b2_p99": float(np.quantile(train_patch_sizes, 0.99)),
                "b2_p999": float(np.quantile(train_patch_sizes, 0.999)),
                "b2_max": int(train_patch_sizes.max()),
                "truncated_roots": int(train_truncated),
                "truncation_rate": float(train_truncation_rate),
            },
        }
        save_json(plan_path, plan)

    split_counts = {
        SPLIT_NAMES[flag]: int(np.sum(split_flags == flag))
        for flag in SPLIT_NAMES
    }
    print("\nStructural-patch cache plan")
    print(f"  dataset scope       : {args.dataset_scope}")
    print(f"  graphs train/valid/test: {split_counts['train']}/{split_counts['valid']}/{split_counts['test']}")
    print(f"  total rooted patches: {total_patches}")
    print(f"  common capacity M   : {capacity} (chosen from train only)")
    print(f"  structural F        : {layout.feature_dim}")
    print(f"  uint8 cache size    : {total_patches * layout.feature_dim / 1024**3:.2f} GiB")

    patches_path = cache_dir / "patches_uint8.npy"
    root_ids_path = cache_dir / "root_ids.npy"
    full_sizes_path = cache_dir / "full_patch_sizes.npy"
    progress_path = cache_dir / "build_progress.json"

    can_resume = (
        args.resume
        and patches_path.exists()
        and root_ids_path.exists()
        and full_sizes_path.exists()
        and progress_path.exists()
    )

    if can_resume:
        patches_memmap = np.load(patches_path, mmap_mode="r+")
        root_ids_memmap = np.load(root_ids_path, mmap_mode="r+")
        full_sizes_memmap = np.load(full_sizes_path, mmap_mode="r+")
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        start_graph_pos = int(progress.get("next_graph_position", 0))
        expected_shape = (total_patches, layout.feature_dim)
        if patches_memmap.shape != expected_shape:
            raise RuntimeError(
                f"Resume patch shape mismatch: expected {expected_shape}, got {patches_memmap.shape}"
            )
        print(f"Resuming structural extraction at graph {start_graph_pos}/{len(selected_indices)}")
    else:
        patches_memmap = np.lib.format.open_memmap(
            patches_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total_patches, layout.feature_dim),
        )
        root_ids_memmap = np.lib.format.open_memmap(
            root_ids_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_patches,),
        )
        full_sizes_memmap = np.lib.format.open_memmap(
            full_sizes_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_patches,),
        )
        start_graph_pos = 0

    checkpoint_every = max(int(args.checkpoint_every_graphs), 1)
    for graph_pos in range(start_graph_pos, len(selected_indices)):
        dataset_index = int(selected_indices[graph_pos])
        data = dataset[dataset_index]
        patch_matrix, graph_root_ids, full_sizes = extract_structural_patch_matrix(
            data=data,
            layout=layout,
            wl_iterations=args.wl_iterations,
        )
        start = int(graph_offsets[graph_pos])
        end = int(graph_offsets[graph_pos + 1])
        patches_memmap[start:end] = patch_matrix
        root_ids_memmap[start:end] = graph_root_ids.astype(np.int32)
        full_sizes_memmap[start:end] = full_sizes.astype(np.int32)

        completed = graph_pos + 1
        if completed % checkpoint_every == 0 or completed == len(selected_indices):
            patches_memmap.flush()
            root_ids_memmap.flush()
            full_sizes_memmap.flush()
            save_json(
                progress_path,
                {
                    "next_graph_position": int(completed),
                    "total_graphs": int(len(selected_indices)),
                    "capacity": int(capacity),
                    "feature_dim": int(layout.feature_dim),
                },
            )
            print(f"  extracted {completed}/{len(selected_indices)} graphs")

    patches_memmap.flush()
    root_ids_memmap.flush()
    full_sizes_memmap.flush()

    truncation_by_split = split_truncation_report(
        graph_offsets=graph_offsets,
        split_flags=split_flags,
        full_patch_sizes=full_sizes_memmap,
        capacity=capacity,
    )

    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_scope": args.dataset_scope,
        "selection_seed": int(args.seed),
        "n_graphs": int(len(selected_indices)),
        "n_dictionary_train_graphs": int(np.sum(split_flags == SPLIT_TRAIN)),
        "n_dictionary_valid_graphs": int(np.sum(split_flags == SPLIT_VALID)),
        "n_test_graphs": int(np.sum(split_flags == SPLIT_TEST)),
        "total_patches": total_patches,
        "patch_size_policy": args.patch_size_policy,
        "patch_size_quantile": float(args.patch_size_quantile),
        "capacity_source": "dictionary-train split only",
        "truncation_by_split": truncation_by_split,
        "wl_iterations": int(args.wl_iterations),
        "cache_complete": True,
        "layout": layout.to_metadata(),
    }
    save_json(cache_dir / "metadata.json", metadata)

    if progress_path.exists():
        progress_path.unlink()

    print("\nTruncation report")
    for split_name in ["train", "valid", "test"]:
        item = truncation_by_split[split_name]
        print(
            f"  {split_name:<5}: roots={item['num_roots']} | "
            f"truncated={item['truncated_roots']} ({item['truncation_rate']:.6%}) | "
            f"B2 mean/p99/max={item.get('b2_mean', float('nan')):.2f}/"
            f"{item.get('b2_p99', float('nan')):.2f}/{item.get('b2_max', -1)}"
        )
    print(f"\nSaved structural-patch cache: {cache_dir.resolve()}")


# =============================================================================
# Cache loading and block scaling
# =============================================================================


@dataclass
class PilotCache:
    cache_dir: Path
    patches: np.ndarray
    graph_offsets: np.ndarray
    graph_split: np.ndarray
    dataset_indices: np.ndarray
    root_ids: np.ndarray
    full_patch_sizes: np.ndarray
    layout: PatchLayout
    metadata: Dict[str, object]

    @classmethod
    def load(cls, cache_dir: str) -> "PilotCache":
        path = Path(cache_dir)
        with (path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        capacity = int(metadata["layout"]["capacity"])
        layout = PatchLayout.create(capacity)
        if layout.feature_dim != int(metadata["layout"]["feature_dim"]):
            raise RuntimeError("Patch layout does not match cache metadata.")

        patches = np.load(path / "patches_uint8.npy", mmap_mode="r")
        graph_offsets = np.load(path / "graph_offsets.npy")
        graph_split = np.load(path / "graph_split.npy")
        dataset_indices = np.load(path / "selected_dataset_indices.npy")
        root_ids = np.load(path / "root_ids.npy", mmap_mode="r")
        full_patch_sizes = np.load(path / "full_patch_sizes.npy", mmap_mode="r")

        expected_shape = (int(graph_offsets[-1]), layout.feature_dim)
        if patches.shape != expected_shape:
            raise RuntimeError(f"Patch cache shape mismatch: expected {expected_shape}, got {patches.shape}")

        return cls(
            cache_dir=path,
            patches=patches,
            graph_offsets=graph_offsets.astype(np.int64),
            graph_split=graph_split.astype(np.uint8),
            dataset_indices=dataset_indices.astype(np.int64),
            root_ids=root_ids,
            full_patch_sizes=full_patch_sizes,
            layout=layout,
            metadata=metadata,
        )

    def graph_patch_indices(self, graph_pos: int) -> np.ndarray:
        start = int(self.graph_offsets[graph_pos])
        end = int(self.graph_offsets[graph_pos + 1])
        return np.arange(start, end, dtype=np.int64)

    def split_graph_positions(self, split_flag: int) -> np.ndarray:
        return np.flatnonzero(self.graph_split == int(split_flag)).astype(np.int64)


@dataclass
class BlockScaler:
    vector: np.ndarray
    block_scales: Dict[str, float]
    mode: str

    @classmethod
    def fit(cls, cache: PilotCache, mode: str) -> "BlockScaler":
        vector = np.ones(cache.layout.feature_dim, dtype=np.float32)
        block_scales: Dict[str, float] = {name: 1.0 for name in cache.layout.blocks}

        if mode == "none":
            return cls(vector=vector, block_scales=block_scales, mode=mode)
        if mode != "equal_energy":
            raise ValueError(f"Unknown block scaling mode: {mode}")

        train_graphs = cache.split_graph_positions(SPLIT_TRAIN)
        total_patches = 0
        block_energy = {name: 0.0 for name in cache.layout.blocks}

        for graph_pos in train_graphs:
            indices = cache.graph_patch_indices(int(graph_pos))
            rows = np.asarray(cache.patches[indices], dtype=np.float32)
            total_patches += rows.shape[0]
            for name in cache.layout.blocks:
                sl = cache.layout.block_slice(name)
                block_energy[name] += float(np.square(rows[:, sl]).sum(dtype=np.float64))

        for name in cache.layout.blocks:
            mean_energy = block_energy[name] / max(total_patches, 1)
            scale = 1.0 / math.sqrt(max(mean_energy, 1e-12))
            block_scales[name] = float(scale)
            vector[cache.layout.block_slice(name)] = np.float32(scale)

        return cls(vector=vector, block_scales=block_scales, mode=mode)

    def transform_rows(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows, dtype=np.float32) * self.vector[None, :]

    def transform_columns(self, columns: np.ndarray) -> np.ndarray:
        return np.asarray(columns, dtype=np.float32) * self.vector[:, None]

    def inverse_columns(self, columns: np.ndarray) -> np.ndarray:
        return np.asarray(columns, dtype=np.float32) / self.vector[:, None]


# =============================================================================
# OMP sparse coding
# =============================================================================


def normalize_columns(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=0)
    norms = np.maximum(norms, eps)
    return matrix / norms[None, :]


def omp_encode(dictionary: np.ndarray, samples: np.ndarray, sparsity: int) -> np.ndarray:
    """
    Parameters
    ----------
    dictionary : [F, K]
    samples    : [F, N]

    Returns
    -------
    codes      : [K, N], at most `sparsity` nonzeros per column.
    """
    dictionary = np.asarray(dictionary, dtype=np.float64, order="F")
    samples = np.asarray(samples, dtype=np.float64, order="F")

    if dictionary.ndim != 2 or samples.ndim != 2:
        raise ValueError("OMP expects 2-D dictionary and sample matrices.")
    if dictionary.shape[0] != samples.shape[0]:
        raise ValueError(f"OMP feature mismatch: D={dictionary.shape}, X={samples.shape}")

    effective_sparsity = min(int(sparsity), int(dictionary.shape[1]))
    if effective_sparsity <= 0:
        raise ValueError("Sparsity must be positive.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = orthogonal_mp(
            dictionary,
            samples,
            n_nonzero_coefs=effective_sparsity,
            tol=None,
            precompute=True,
            copy_X=True,
            return_path=False,
        )

    code = np.asarray(code, dtype=np.float32)
    if code.ndim == 1:
        code = code[:, None]
    if code.shape != (dictionary.shape[1], samples.shape[1]):
        raise RuntimeError(f"Unexpected OMP code shape: {code.shape}")
    return code


# =============================================================================
# K-SVD atom update
# =============================================================================


def leading_rank1(
    residual: np.ndarray,
    mode: str,
    power_iterations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float32)
    if residual.ndim != 2 or residual.shape[1] == 0:
        raise ValueError("Rank-1 SVD requires a nonempty residual matrix.")

    if residual.shape[1] == 1:
        column = residual[:, 0]
        sigma = float(np.linalg.norm(column))
        if sigma <= 1e-12:
            return np.zeros_like(column), 0.0, np.ones(1, dtype=np.float32)
        return (column / sigma).astype(np.float32), sigma, np.ones(1, dtype=np.float32)

    if mode == "exact":
        u, singular_values, vh = np.linalg.svd(residual, full_matrices=False)
        return u[:, 0].astype(np.float32), float(singular_values[0]), vh[0].astype(np.float32)

    if mode != "power":
        raise ValueError(f"Unknown SVD mode: {mode}")

    v = rng.standard_normal(residual.shape[1]).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-12)

    u = np.zeros(residual.shape[0], dtype=np.float32)
    for _ in range(max(int(power_iterations), 1)):
        u = residual @ v
        u_norm = float(np.linalg.norm(u))
        if u_norm <= 1e-12:
            return u, 0.0, v
        u /= u_norm

        v = residual.T @ u
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-12:
            return u, 0.0, v
        v /= v_norm

    sigma = float(u @ (residual @ v))
    if sigma < 0:
        u = -u
        sigma = -sigma
    return u.astype(np.float32), sigma, v.astype(np.float32)


def ksvd_dictionary_update(
    samples: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    sweeps: int,
    svd_mode: str,
    power_iterations: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    One or more classic K-SVD dictionary sweeps with fixed OMP supports.

    For atom k:
        Omega_k = {i : C[k,i] != 0}
        E_k     = X[:,Omega_k] - D C[:,Omega_k] + d_k C[k,Omega_k]
        E_k ~= u sigma v^T
        d_k <- u
        C[k,Omega_k] <- sigma v^T
    """
    X = np.asarray(samples, dtype=np.float32)
    D = np.asarray(dictionary, dtype=np.float32).copy()
    C = np.asarray(codes, dtype=np.float32).copy()

    if X.shape[0] != D.shape[0] or D.shape[1] != C.shape[0] or X.shape[1] != C.shape[1]:
        raise ValueError(f"K-SVD shape mismatch: X={X.shape}, D={D.shape}, C={C.shape}")

    revived = 0
    tiny_updates = 0

    reconstruction = D @ C

    for _ in range(int(sweeps)):
        atom_order = rng.permutation(D.shape[1])

        for atom_id in atom_order:
            active = np.flatnonzero(np.abs(C[atom_id]) > 1e-10)

            if active.size == 0:
                residual = X - reconstruction
                residual_norms = np.sum(np.square(residual), axis=0)
                sample_id = int(np.argmax(residual_norms))
                candidate = residual[:, sample_id]
                candidate_norm = float(np.linalg.norm(candidate))

                if candidate_norm <= 1e-12:
                    candidate = X[:, sample_id]
                    candidate_norm = float(np.linalg.norm(candidate))

                if candidate_norm > 1e-12:
                    D[:, atom_id] = candidate / candidate_norm
                    C[atom_id, :] = 0.0
                    C[atom_id, sample_id] = candidate_norm
                    reconstruction[:, sample_id] += D[:, atom_id] * candidate_norm
                    revived += 1
                continue

            old_atom = D[:, atom_id].copy()
            old_coefficients = C[atom_id, active].copy()

            restricted_residual = (
                X[:, active]
                - reconstruction[:, active]
                + old_atom[:, None] * old_coefficients[None, :]
            )

            u, sigma, v = leading_rank1(
                restricted_residual,
                mode=svd_mode,
                power_iterations=power_iterations,
                rng=rng,
            )

            if sigma <= 1e-10 or float(np.linalg.norm(u)) <= 1e-10:
                tiny_updates += 1
                continue

            D[:, atom_id] = u
            C[atom_id, active] = sigma * v

            reconstruction[:, active] = (
                X[:, active]
                - restricted_residual
                + D[:, atom_id, None] * C[atom_id, active][None, :]
            )

    D = normalize_columns(D)
    return D, C, {"revived_atoms": int(revived), "tiny_updates": int(tiny_updates)}


# =============================================================================
# Graph-balanced patch sampling and replay reservoir
# =============================================================================


def sample_graph_patch_indices(
    cache: PilotCache,
    graph_positions: Sequence[int],
    patches_per_graph: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: List[np.ndarray] = []
    for graph_pos in graph_positions:
        indices = cache.graph_patch_indices(int(graph_pos))
        if patches_per_graph > 0 and len(indices) > patches_per_graph:
            indices = rng.choice(indices, size=patches_per_graph, replace=False)
        selected.append(np.asarray(indices, dtype=np.int64))
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(selected)


class UniqueReplayReservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.items: List[int] = []
        self.item_set = set()
        self.unique_seen = 0

    def add(self, indices: Sequence[int]) -> None:
        if self.capacity <= 0:
            return
        for raw_index in indices:
            index = int(raw_index)
            if index in self.item_set:
                continue
            self.unique_seen += 1
            if len(self.items) < self.capacity:
                self.items.append(index)
                self.item_set.add(index)
                continue

            replacement = int(self.rng.integers(0, self.unique_seen))
            if replacement < self.capacity:
                old = self.items[replacement]
                self.item_set.remove(old)
                self.items[replacement] = index
                self.item_set.add(index)

    def sample(self, count: int) -> np.ndarray:
        if count <= 0 or not self.items:
            return np.empty(0, dtype=np.int64)
        count = min(int(count), len(self.items))
        return self.rng.choice(np.asarray(self.items, dtype=np.int64), size=count, replace=False)


# =============================================================================
# Dictionary initialization
# =============================================================================


def build_balanced_initial_reservoir(
    cache: PilotCache,
    patches_per_graph: int,
    max_patches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    train_graphs = cache.split_graph_positions(SPLIT_TRAIN)
    indices = sample_graph_patch_indices(cache, train_graphs, patches_per_graph, rng)
    if max_patches > 0 and len(indices) > max_patches:
        indices = rng.choice(indices, size=max_patches, replace=False)
    return np.asarray(indices, dtype=np.int64)


def kmeanspp_real_patch_initialization(
    samples: np.ndarray,
    n_atoms: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means++ selection, but every initial atom remains an actual normalized patch."""
    X = np.asarray(samples, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Initialization samples must be [F,N].")
    if X.shape[1] < n_atoms:
        raise ValueError(f"Need at least {n_atoms} initialization patches, got {X.shape[1]}")

    normalized = normalize_columns(X)
    chosen = np.zeros(n_atoms, dtype=np.int64)
    chosen[0] = int(rng.integers(0, X.shape[1]))

    best_distance = np.sum(np.square(normalized - normalized[:, chosen[0], None]), axis=0)

    for atom_id in range(1, n_atoms):
        probabilities = np.maximum(best_distance, 0.0)
        probability_sum = float(probabilities.sum())
        if probability_sum <= 1e-12:
            remaining = np.setdiff1d(np.arange(X.shape[1]), chosen[:atom_id], assume_unique=False)
            chosen[atom_id] = int(rng.choice(remaining))
        else:
            probabilities /= probability_sum
            chosen[atom_id] = int(rng.choice(X.shape[1], p=probabilities))

        distance = np.sum(
            np.square(normalized - normalized[:, chosen[atom_id], None]), axis=0
        )
        best_distance = np.minimum(best_distance, distance)

    dictionary = normalized[:, chosen]
    return dictionary.astype(np.float32), chosen


# =============================================================================
# Reconstruction and energy metrics
# =============================================================================


def retention_from_energies(residual_energy: float, input_energy: float) -> float:
    if input_energy <= 1e-12:
        return 1.0 if residual_energy <= 1e-12 else 0.0
    return float(1.0 - residual_energy / input_energy)


def evaluate_dictionary(
    cache: PilotCache,
    scaler: BlockScaler,
    dictionary: np.ndarray,
    graph_positions: Sequence[int],
    sparsity: int,
    max_patches_per_graph: int = 0,
    seed: int = 0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    raw_input_total = 0.0
    raw_residual_total = 0.0
    scaled_input_total = 0.0
    scaled_residual_total = 0.0

    raw_block_input = {name: 0.0 for name in cache.layout.blocks}
    raw_block_residual = {name: 0.0 for name in cache.layout.blocks}
    scaled_block_input = {name: 0.0 for name in cache.layout.blocks}
    scaled_block_residual = {name: 0.0 for name in cache.layout.blocks}

    graph_raw_retentions: List[float] = []
    graph_scaled_retentions: List[float] = []

    usage_count = np.zeros(dictionary.shape[1], dtype=np.float64)
    usage_abs = np.zeros(dictionary.shape[1], dtype=np.float64)
    active_counts: List[int] = []
    patch_count = 0

    for graph_pos in graph_positions:
        indices = cache.graph_patch_indices(int(graph_pos))
        if max_patches_per_graph > 0 and len(indices) > max_patches_per_graph:
            indices = rng.choice(indices, size=max_patches_per_graph, replace=False)

        raw_rows = np.asarray(cache.patches[indices], dtype=np.float32)
        raw_X = raw_rows.T
        scaled_X = scaler.transform_columns(raw_X)
        codes = omp_encode(dictionary, scaled_X, sparsity=sparsity)
        scaled_hat = dictionary @ codes
        raw_hat = scaler.inverse_columns(scaled_hat)

        raw_residual = raw_X - raw_hat
        scaled_residual = scaled_X - scaled_hat

        raw_e = float(np.square(raw_X).sum(dtype=np.float64))
        raw_r = float(np.square(raw_residual).sum(dtype=np.float64))
        scaled_e = float(np.square(scaled_X).sum(dtype=np.float64))
        scaled_r = float(np.square(scaled_residual).sum(dtype=np.float64))

        raw_input_total += raw_e
        raw_residual_total += raw_r
        scaled_input_total += scaled_e
        scaled_residual_total += scaled_r
        graph_raw_retentions.append(retention_from_energies(raw_r, raw_e))
        graph_scaled_retentions.append(retention_from_energies(scaled_r, scaled_e))

        for name in cache.layout.blocks:
            sl = cache.layout.block_slice(name)
            raw_block_input[name] += float(np.square(raw_X[sl]).sum(dtype=np.float64))
            raw_block_residual[name] += float(np.square(raw_residual[sl]).sum(dtype=np.float64))
            scaled_block_input[name] += float(np.square(scaled_X[sl]).sum(dtype=np.float64))
            scaled_block_residual[name] += float(np.square(scaled_residual[sl]).sum(dtype=np.float64))

        active = np.abs(codes) > 1e-10
        usage_count += active.sum(axis=1)
        usage_abs += np.abs(codes).sum(axis=1)
        active_counts.extend(active.sum(axis=0).astype(int).tolist())
        patch_count += codes.shape[1]

    usage_probability = usage_count / max(float(usage_count.sum()), 1e-12)
    positive_usage = usage_probability[usage_probability > 0]
    entropy = float(-np.sum(positive_usage * np.log(positive_usage))) if positive_usage.size else 0.0
    effective_atoms = float(np.exp(entropy)) if positive_usage.size else 0.0

    normalized_dictionary = normalize_columns(dictionary)
    atom_cosine = np.abs(normalized_dictionary.T @ normalized_dictionary)
    np.fill_diagonal(atom_cosine, 0.0)

    raw_block_retention = {
        name: retention_from_energies(raw_block_residual[name], raw_block_input[name])
        for name in cache.layout.blocks
    }
    scaled_block_retention = {
        name: retention_from_energies(scaled_block_residual[name], scaled_block_input[name])
        for name in cache.layout.blocks
    }
    core_names = [name for name in ["topology", "bond", "atom"] if name in raw_block_retention]
    raw_core_mean = float(np.mean([raw_block_retention[name] for name in core_names]))
    scaled_core_mean = float(np.mean([scaled_block_retention[name] for name in core_names]))

    return {
        "num_graphs": int(len(graph_positions)),
        "num_patches": int(patch_count),
        "raw_global_energy_retention": retention_from_energies(raw_residual_total, raw_input_total),
        "raw_graph_mean_energy_retention": float(np.mean(graph_raw_retentions)),
        "scaled_global_energy_retention": retention_from_energies(scaled_residual_total, scaled_input_total),
        "scaled_graph_mean_energy_retention": float(np.mean(graph_scaled_retentions)),
        "raw_relative_frobenius_error": float(math.sqrt(raw_residual_total / max(raw_input_total, 1e-12))),
        "scaled_relative_frobenius_error": float(math.sqrt(scaled_residual_total / max(scaled_input_total, 1e-12))),
        "raw_block_energy_retention": raw_block_retention,
        "scaled_block_energy_retention": scaled_block_retention,
        "raw_core_equal_block_energy_retention": raw_core_mean,
        "scaled_core_equal_block_energy_retention": scaled_core_mean,
        "mean_active_atoms": float(np.mean(active_counts)) if active_counts else 0.0,
        "usage_count": usage_count,
        "usage_abs": usage_abs,
        "dead_atoms": int(np.sum(usage_count == 0)),
        "effective_atoms": effective_atoms,
        "max_abs_atom_cosine": float(atom_cosine.max()) if atom_cosine.size else 0.0,
    }


def concise_metric_line(name: str, metrics: Mapping[str, object]) -> str:
    return (
        f"[{name}] raw_global={metrics['raw_global_energy_retention']:.4f} | "
        f"raw_graph={metrics['raw_graph_mean_energy_retention']:.4f} | "
        f"scaled_global={metrics['scaled_global_energy_retention']:.4f} | "
        f"scaled_graph={metrics['scaled_graph_mean_energy_retention']:.4f} | "
        f"core={metrics['raw_core_equal_block_energy_retention']:.4f} | "
        f"active={metrics['mean_active_atoms']:.2f} | dead={metrics['dead_atoms']} | "
        f"K_eff={metrics['effective_atoms']:.2f} | max|cos|={metrics['max_abs_atom_cosine']:.4f}"
    )


# =============================================================================
# Dictionary canonicalization, unpacking and final sparse-code export
# =============================================================================


def canonicalize_atom_signs(dictionary: np.ndarray) -> np.ndarray:
    D = np.asarray(dictionary, dtype=np.float32).copy()
    for atom_id in range(D.shape[1]):
        pivot = int(np.argmax(np.abs(D[:, atom_id])))
        if D[pivot, atom_id] < 0:
            D[:, atom_id] *= -1.0
    return D


def unpack_dictionary(dictionary_raw: np.ndarray, layout: PatchLayout) -> Dict[str, np.ndarray]:
    K = dictionary_raw.shape[1]
    M = layout.capacity
    P = layout.num_pairs

    topology = np.zeros((K, M, M), dtype=np.float32)
    bond = np.zeros((K, BOND_DIM, M, M), dtype=np.float32)

    topology_flat = dictionary_raw[layout.block_slice("topology")].T
    bond_flat = dictionary_raw[layout.block_slice("bond")].T.reshape(K, BOND_DIM, P)

    for atom_id in range(K):
        topology[atom_id, layout.pair_i, layout.pair_j] = topology_flat[atom_id]
        topology[atom_id, layout.pair_j, layout.pair_i] = topology_flat[atom_id]
        for channel in range(BOND_DIM):
            bond[atom_id, channel, layout.pair_i, layout.pair_j] = bond_flat[atom_id, channel]
            bond[atom_id, channel, layout.pair_j, layout.pair_i] = bond_flat[atom_id, channel]

    atom = dictionary_raw[layout.block_slice("atom")].T.reshape(K, M, ATOM_DIM)
    shell = dictionary_raw[layout.block_slice("shell")].T.reshape(K, M, SHELL_DIM)
    mask = dictionary_raw[layout.block_slice("mask")].T.reshape(K, M)

    return {
        "atom_topology": topology,
        "atom_bond": bond,
        "atom_node_semantics": atom,
        "atom_shell": shell,
        "atom_mask": mask,
    }


def dense_codes_to_fixed_sparse(codes: np.ndarray, sparsity: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K, N = codes.shape
    indices = np.full((N, sparsity), -1, dtype=np.int16 if K < 32767 else np.int32)
    values = np.zeros((N, sparsity), dtype=np.float32)
    nnz = np.zeros(N, dtype=np.uint8)

    for sample_id in range(N):
        active = np.flatnonzero(np.abs(codes[:, sample_id]) > 1e-10)
        if active.size:
            active = active[np.argsort(np.abs(codes[active, sample_id]))[::-1]]
        active = active[:sparsity]
        count = len(active)
        if count:
            indices[sample_id, :count] = active
            values[sample_id, :count] = codes[active, sample_id]
        nnz[sample_id] = count

    return indices, values, nnz


def export_all_codes_and_nearest_patches(
    cache: PilotCache,
    scaler: BlockScaler,
    dictionary: np.ndarray,
    sparsity: int,
    out_dir: Path,
    nearest_graph_positions: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    total_patches = int(cache.graph_offsets[-1])
    code_indices = np.lib.format.open_memmap(
        out_dir / "code_indices.npy",
        mode="w+",
        dtype=np.int16 if dictionary.shape[1] < 32767 else np.int32,
        shape=(total_patches, sparsity),
    )
    code_values = np.lib.format.open_memmap(
        out_dir / "code_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_patches, sparsity),
    )
    code_nnz = np.lib.format.open_memmap(
        out_dir / "code_nnz.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(total_patches,),
    )

    nearest_abs_corr = np.full(dictionary.shape[1], -np.inf, dtype=np.float64)
    nearest_signed_corr = np.zeros(dictionary.shape[1], dtype=np.float64)
    nearest_patch_row = np.full(dictionary.shape[1], -1, dtype=np.int64)
    nearest_graph_pos = np.full(dictionary.shape[1], -1, dtype=np.int64)
    nearest_root_id = np.full(dictionary.shape[1], -1, dtype=np.int64)

    usage_count = np.zeros(dictionary.shape[1], dtype=np.float64)
    usage_abs = np.zeros(dictionary.shape[1], dtype=np.float64)

    nearest_graph_set = None
    if nearest_graph_positions is not None:
        nearest_graph_set = set(int(x) for x in nearest_graph_positions)

    for graph_pos in range(len(cache.dataset_indices)):
        patch_rows = cache.graph_patch_indices(graph_pos)
        raw_rows = np.asarray(cache.patches[patch_rows], dtype=np.float32)
        scaled_X = scaler.transform_rows(raw_rows).T
        codes = omp_encode(dictionary, scaled_X, sparsity=sparsity)
        local_indices, local_values, local_nnz = dense_codes_to_fixed_sparse(codes, sparsity)

        code_indices[patch_rows] = local_indices
        code_values[patch_rows] = local_values
        code_nnz[patch_rows] = local_nnz

        active = np.abs(codes) > 1e-10
        usage_count += active.sum(axis=1)
        usage_abs += np.abs(codes).sum(axis=1)

        if nearest_graph_set is None or graph_pos in nearest_graph_set:
            sample_norms = np.linalg.norm(scaled_X, axis=0)
            valid_norms = np.maximum(sample_norms, 1e-12)
            correlations = dictionary.T @ scaled_X / valid_norms[None, :]

            for atom_id in range(dictionary.shape[1]):
                local_sample = int(np.argmax(np.abs(correlations[atom_id])))
                local_abs = float(abs(correlations[atom_id, local_sample]))
                if local_abs > nearest_abs_corr[atom_id]:
                    nearest_abs_corr[atom_id] = local_abs
                    nearest_signed_corr[atom_id] = float(correlations[atom_id, local_sample])
                    global_row = int(patch_rows[local_sample])
                    nearest_patch_row[atom_id] = global_row
                    nearest_graph_pos[atom_id] = int(graph_pos)
                    nearest_root_id[atom_id] = int(cache.root_ids[global_row])

    code_indices.flush()
    code_values.flush()
    code_nnz.flush()
    del code_indices, code_values, code_nnz

    nearest = {
        "nearest_abs_correlation": nearest_abs_corr,
        "nearest_signed_correlation": nearest_signed_corr,
        "nearest_patch_row": nearest_patch_row,
        "nearest_graph_position": nearest_graph_pos,
        "nearest_dataset_index": np.asarray(
            [cache.dataset_indices[pos] if pos >= 0 else -1 for pos in nearest_graph_pos], dtype=np.int64
        ),
        "nearest_root_id": nearest_root_id,
        "usage_count": usage_count,
        "usage_abs": usage_abs,
    }
    np.savez(out_dir / "atom_nearest_real_patches.npz", **nearest)
    return nearest


def fixed_graph_monitor_subset(
    graph_positions: np.ndarray,
    max_graphs: int,
    seed: int,
) -> np.ndarray:
    graph_positions = np.asarray(graph_positions, dtype=np.int64)
    if max_graphs <= 0 or len(graph_positions) <= max_graphs:
        return graph_positions.copy()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(graph_positions, size=int(max_graphs), replace=False)).astype(np.int64)


# =============================================================================
# Online training
# =============================================================================


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cache = PilotCache.load(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_graphs = cache.split_graph_positions(SPLIT_TRAIN)
    valid_graphs = cache.split_graph_positions(SPLIT_VALID)
    test_graphs = cache.split_graph_positions(SPLIT_TEST)

    if len(train_graphs) == 0 or len(valid_graphs) == 0:
        raise RuntimeError("Cache must contain nonempty dictionary-train and validation splits.")

    monitor_train_graphs = fixed_graph_monitor_subset(
        train_graphs, args.monitor_train_graphs, seed=args.seed + 701
    )
    monitor_valid_graphs = fixed_graph_monitor_subset(
        valid_graphs, args.monitor_valid_graphs, seed=args.seed + 702
    )
    np.save(out_dir / "monitor_train_graph_positions.npy", monitor_train_graphs)
    np.save(out_dir / "monitor_valid_graph_positions.npy", monitor_valid_graphs)

    print("Training/evaluation split sizes")
    print(f"  dictionary train : {len(train_graphs)}")
    print(f"  official/internal valid: {len(valid_graphs)}")
    print(f"  test encode-only : {len(test_graphs)}")
    print(f"  epoch monitor train/valid: {len(monitor_train_graphs)}/{len(monitor_valid_graphs)}")

    scaler = BlockScaler.fit(cache, mode=args.block_scaling)
    np.savez(
        out_dir / "block_scaler.npz",
        scale_vector=scaler.vector,
        block_names=np.asarray(list(cache.layout.blocks.keys())),
        block_scales=np.asarray(
            [scaler.block_scales[name] for name in cache.layout.blocks], dtype=np.float32
        ),
    )

    init_indices = build_balanced_initial_reservoir(
        cache=cache,
        patches_per_graph=args.init_patches_per_graph,
        max_patches=args.init_max_patches,
        rng=rng,
    )
    init_rows = np.asarray(cache.patches[init_indices], dtype=np.float32)
    init_X = scaler.transform_rows(init_rows).T
    dictionary, chosen_local = kmeanspp_real_patch_initialization(
        samples=init_X,
        n_atoms=args.n_atoms,
        rng=rng,
    )
    chosen_global = init_indices[chosen_local]

    np.save(out_dir / "initial_dictionary_scaled.npy", dictionary)
    np.save(out_dir / "initial_atom_patch_rows.npy", chosen_global)

    replay = UniqueReplayReservoir(capacity=args.replay_capacity, seed=args.seed + 1009)
    history: List[Dict[str, object]] = []
    best_selection = -np.inf
    best_epoch = 0
    best_dictionary = dictionary.copy()

    # Fixed monitor subsets avoid a complete 32,901-graph OMP pass every epoch.
    initial_train = evaluate_dictionary(
        cache,
        scaler,
        dictionary,
        monitor_train_graphs,
        args.sparsity,
        max_patches_per_graph=args.eval_patches_per_graph,
        seed=args.seed + 1,
    )
    initial_valid = evaluate_dictionary(
        cache,
        scaler,
        dictionary,
        monitor_valid_graphs,
        args.sparsity,
        max_patches_per_graph=args.eval_patches_per_graph,
        seed=args.seed + 2,
    )
    print(concise_metric_line("epoch 0 monitor-train", initial_train))
    print(concise_metric_line("epoch 0 monitor-valid", initial_valid))
    history.append({"epoch": 0, "train": initial_train, "valid": initial_valid})
    initial_selection = (
        initial_train if args.selection_monitor == "train" else initial_valid
    )
    best_selection = float(initial_selection["scaled_graph_mean_energy_retention"])
    np.save(out_dir / "best_dictionary_scaled.npy", best_dictionary)
    save_json(out_dir / "best_selection_metrics.json", initial_selection)

    for epoch in range(1, args.epochs + 1):
        shuffled = rng.permutation(train_graphs)
        epoch_batch_retentions: List[float] = []
        epoch_revived = 0

        for start in range(0, len(shuffled), args.graphs_per_batch):
            graph_batch = shuffled[start : start + args.graphs_per_batch]
            new_indices = sample_graph_patch_indices(
                cache=cache,
                graph_positions=graph_batch,
                patches_per_graph=args.patches_per_graph,
                rng=rng,
            )
            replay_indices = replay.sample(args.replay_per_batch)
            update_indices = np.concatenate([new_indices, replay_indices])

            raw_rows = np.asarray(cache.patches[update_indices], dtype=np.float32)
            X = scaler.transform_rows(raw_rows).T

            codes = omp_encode(dictionary, X, sparsity=args.sparsity)
            dictionary, updated_codes, update_info = ksvd_dictionary_update(
                samples=X,
                dictionary=dictionary,
                codes=codes,
                sweeps=args.dictionary_sweeps,
                svd_mode=args.svd_mode,
                power_iterations=args.power_iterations,
                rng=rng,
            )

            residual = X - dictionary @ updated_codes
            batch_retention = retention_from_energies(
                float(np.square(residual).sum(dtype=np.float64)),
                float(np.square(X).sum(dtype=np.float64)),
            )
            epoch_batch_retentions.append(batch_retention)
            epoch_revived += int(update_info["revived_atoms"])
            replay.add(new_indices)

        print(
            f"epoch {epoch:03d} online batches | mean scaled retention="
            f"{np.mean(epoch_batch_retentions):.4f} | replay={len(replay.items)} | "
            f"revived={epoch_revived}"
        )

        should_evaluate = epoch == args.epochs or epoch % args.eval_every == 0
        if should_evaluate:
            train_metrics = evaluate_dictionary(
                cache,
                scaler,
                dictionary,
                monitor_train_graphs,
                args.sparsity,
                max_patches_per_graph=args.eval_patches_per_graph,
                seed=args.seed + epoch * 11,
            )
            valid_metrics = evaluate_dictionary(
                cache,
                scaler,
                dictionary,
                monitor_valid_graphs,
                args.sparsity,
                max_patches_per_graph=args.eval_patches_per_graph,
                seed=args.seed + epoch * 13,
            )
            print(concise_metric_line(f"epoch {epoch} monitor-train", train_metrics))
            print(concise_metric_line(f"epoch {epoch} monitor-valid", valid_metrics))
            print("  monitor-valid raw block retention:", valid_metrics["raw_block_energy_retention"])

            record = {
                "epoch": int(epoch),
                "mean_online_batch_scaled_retention": float(np.mean(epoch_batch_retentions)),
                "replay_size": int(len(replay.items)),
                "revived_atoms": int(epoch_revived),
                "train": train_metrics,
                "valid": valid_metrics,
            }
            history.append(record)
            save_json(out_dir / "history.partial.json", {"history": history})

            selection_metrics = (
                train_metrics if args.selection_monitor == "train" else valid_metrics
            )
            score = float(selection_metrics["scaled_graph_mean_energy_retention"])
            if score > best_selection:
                best_selection = score
                best_epoch = epoch
                best_dictionary = dictionary.copy()
                np.save(out_dir / "best_dictionary_scaled.npy", best_dictionary)
                save_json(out_dir / "best_selection_metrics.json", selection_metrics)

        np.savez(
            out_dir / "last_checkpoint.npz",
            dictionary=dictionary,
            epoch=np.asarray(epoch, dtype=np.int64),
            replay_indices=np.asarray(replay.items, dtype=np.int64),
        )

    # Canonical signs and atom order must depend on train only.
    best_dictionary = canonicalize_atom_signs(best_dictionary)
    train_usage_metrics = evaluate_dictionary(
        cache,
        scaler,
        best_dictionary,
        monitor_train_graphs,
        args.sparsity,
        max_patches_per_graph=args.eval_patches_per_graph,
        seed=args.seed + 999,
    )
    usage_order = np.argsort(np.asarray(train_usage_metrics["usage_abs"]))[::-1]
    best_dictionary = best_dictionary[:, usage_order]

    print("\nFinal complete-split evaluation (fixed dictionary; no updates)")
    final_train = evaluate_dictionary(
        cache,
        scaler,
        best_dictionary,
        train_graphs,
        args.sparsity,
        max_patches_per_graph=0,
        seed=args.seed + 1001,
    )
    final_valid = evaluate_dictionary(
        cache,
        scaler,
        best_dictionary,
        valid_graphs,
        args.sparsity,
        max_patches_per_graph=0,
        seed=args.seed + 1002,
    )
    final_test = None
    if len(test_graphs):
        final_test = evaluate_dictionary(
            cache,
            scaler,
            best_dictionary,
            test_graphs,
            args.sparsity,
            max_patches_per_graph=0,
            seed=args.seed + 1003,
        )

    dictionary_raw = scaler.inverse_columns(best_dictionary)
    np.save(out_dir / "dictionary_scaled.npy", best_dictionary)
    np.save(out_dir / "dictionary_raw.npy", dictionary_raw)
    np.save(out_dir / "atom_usage_order.npy", usage_order)

    unpacked = unpack_dictionary(dictionary_raw, cache.layout)
    for name, array in unpacked.items():
        np.save(out_dir / f"{name}.npy", array)

    nearest: Dict[str, object] = {}
    if not args.skip_code_export:
        nearest = export_all_codes_and_nearest_patches(
            cache=cache,
            scaler=scaler,
            dictionary=best_dictionary,
            sparsity=args.sparsity,
            out_dir=out_dir,
            nearest_graph_positions=train_graphs,
        )

    summary = {
        "best_epoch": int(best_epoch),
        "dictionary_selection_monitor": args.selection_monitor,
        "best_monitor_scaled_graph_mean_energy_retention": float(best_selection),
        "config": {key: value for key, value in vars(args).items() if key != "func"},
        "cache_metadata": cache.metadata,
        "monitor_graph_counts": {
            "train": int(len(monitor_train_graphs)),
            "valid": int(len(monitor_valid_graphs)),
        },
        "block_scaler": {
            "mode": scaler.mode,
            "block_scales": scaler.block_scales,
        },
        "final_train": final_train,
        "final_valid": final_valid,
        "final_test": final_test,
        "nearest_real_patch_train_only": nearest,
        "matrix_shapes": {
            "dictionary": list(best_dictionary.shape),
            "all_sparse_code_indices": None
            if args.skip_code_export
            else [int(cache.graph_offsets[-1]), int(args.sparsity)],
            "all_sparse_code_values": None
            if args.skip_code_export
            else [int(cache.graph_offsets[-1]), int(args.sparsity)],
        },
    }
    save_json(out_dir / "history.json", {"history": history})
    save_json(out_dir / "summary.json", summary)

    print(concise_metric_line("train", final_train))
    print(concise_metric_line("valid", final_valid))
    if final_test is not None:
        print(concise_metric_line("test", final_test))
    print("valid raw block retention:", final_valid["raw_block_energy_retention"])
    if final_test is not None:
        print("test raw block retention:", final_test["raw_block_energy_retention"])
    print(f"Saved result directory: {out_dir.resolve()}")


# =============================================================================
# Synthetic self-test of X ~= D C
# =============================================================================


def self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    feature_dim = 96
    true_atoms = 8
    learned_atoms = 8
    samples = 500
    sparsity = 2

    true_dictionary = normalize_columns(rng.normal(size=(feature_dim, true_atoms)).astype(np.float32))
    true_codes = np.zeros((true_atoms, samples), dtype=np.float32)
    for sample_id in range(samples):
        active = rng.choice(true_atoms, size=sparsity, replace=False)
        true_codes[active, sample_id] = rng.normal(size=sparsity)
    X = true_dictionary @ true_codes + 0.01 * rng.normal(size=(feature_dim, samples)).astype(np.float32)

    init_columns = rng.choice(samples, size=learned_atoms, replace=False)
    dictionary = normalize_columns(X[:, init_columns])

    initial_codes = omp_encode(dictionary, X, sparsity=sparsity)
    initial_retention = retention_from_energies(
        float(np.square(X - dictionary @ initial_codes).sum()), float(np.square(X).sum())
    )

    for _ in range(12):
        codes = omp_encode(dictionary, X, sparsity=sparsity)
        dictionary, _, _ = ksvd_dictionary_update(
            samples=X,
            dictionary=dictionary,
            codes=codes,
            sweeps=1,
            svd_mode="exact",
            power_iterations=8,
            rng=rng,
        )

    final_codes = omp_encode(dictionary, X, sparsity=sparsity)
    final_retention = retention_from_energies(
        float(np.square(X - dictionary @ final_codes).sum()), float(np.square(X).sum())
    )

    print(f"synthetic X shape: {X.shape}")
    print(f"learned D shape : {dictionary.shape}")
    print(f"learned C shape : {final_codes.shape}")
    print(f"initial energy retention: {initial_retention:.6f}")
    print(f"final energy retention  : {final_retention:.6f}")

    if final_retention <= initial_retention:
        raise RuntimeError("Self-test failed: K-SVD did not improve reconstruction.")
    print("self-test passed")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MolHIV online shared structural atoms: OMP + mini-batch K-SVD."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser(
        "build-cache",
        help="Build pilot or leakage-safe official-full structural patch cache.",
    )
    cache_parser.add_argument("--dataset-root", type=str, default="./data/ogb")
    cache_parser.add_argument("--cache-dir", type=str, required=True)
    cache_parser.add_argument(
        "--dataset-scope",
        choices=["pilot", "official-full"],
        default="official-full",
        help=(
            "pilot: sample official train and make an internal split; "
            "official-full: cache official train/valid/test and learn capacity from train only."
        ),
    )
    cache_parser.add_argument("--n-graphs", type=int, default=2000)
    cache_parser.add_argument("--n-bins", type=int, default=10)
    cache_parser.add_argument("--valid-fraction", type=float, default=0.20)
    cache_parser.add_argument("--seed", type=int, default=2026)
    cache_parser.add_argument(
        "--patch-size-policy", choices=["max", "quantile"], default="quantile"
    )
    cache_parser.add_argument("--patch-size-quantile", type=float, default=0.999)
    cache_parser.add_argument("--patch-capacity", type=int, default=None)
    cache_parser.add_argument("--wl-iterations", type=int, default=3)
    cache_parser.add_argument(
        "--checkpoint-every-graphs",
        type=int,
        default=250,
        help="Flush cache and save resumable progress every N graphs.",
    )
    cache_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted extraction using build_plan/progress files.",
    )
    cache_parser.set_defaults(func=build_cache)

    train_parser = subparsers.add_parser(
        "train", help="Learn shared atoms from split=train and encode all cached splits."
    )
    train_parser.add_argument("--cache-dir", type=str, required=True)
    train_parser.add_argument("--out-dir", type=str, required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--n-atoms", type=int, default=32)
    train_parser.add_argument("--sparsity", type=int, default=6)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--graphs-per-batch", type=int, default=32)
    train_parser.add_argument("--patches-per-graph", type=int, default=8)
    train_parser.add_argument("--dictionary-sweeps", type=int, default=1)
    train_parser.add_argument("--svd-mode", choices=["power", "exact"], default="power")
    train_parser.add_argument("--power-iterations", type=int, default=6)
    train_parser.add_argument("--replay-capacity", type=int, default=32768)
    train_parser.add_argument("--replay-per-batch", type=int, default=256)
    train_parser.add_argument("--init-patches-per-graph", type=int, default=4)
    train_parser.add_argument("--init-max-patches", type=int, default=32768)
    train_parser.add_argument(
        "--block-scaling", choices=["none", "equal_energy"], default="equal_energy"
    )
    train_parser.add_argument("--eval-every", type=int, default=1)
    train_parser.add_argument(
        "--selection-monitor",
        choices=["train", "valid"],
        default="valid",
        help=(
            "Monitor used to select the dictionary epoch. Use train for a strict "
            "train-only feature-learning protocol; valid preserves historical behavior."
        ),
    )
    train_parser.add_argument(
        "--monitor-train-graphs",
        type=int,
        default=1000,
        help="Fixed train graph subset used for epoch monitoring; 0 means all.",
    )
    train_parser.add_argument(
        "--monitor-valid-graphs",
        type=int,
        default=1000,
        help="Fixed valid graph subset used for epoch selection; 0 means all.",
    )
    train_parser.add_argument(
        "--eval-patches-per-graph",
        type=int,
        default=8,
        help="Patch subsample per monitor graph; 0 means all. Final evaluation always uses all patches.",
    )
    train_parser.add_argument(
        "--skip-code-export",
        action="store_true",
        help="Skip the final all-patch code_indices/code_values export.",
    )
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser("self-test", help="Synthetic OMP + K-SVD sanity test.")
    test_parser.add_argument("--seed", type=int, default=0)
    test_parser.set_defaults(func=self_test)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "build-cache":
        if args.dataset_scope == "pilot":
            if args.n_graphs <= 1:
                raise ValueError("--n-graphs must be greater than 1 in pilot mode")
            if args.n_bins <= 0:
                raise ValueError("--n-bins must be positive")
            if not 0.0 < args.valid_fraction < 1.0:
                raise ValueError("--valid-fraction must be between 0 and 1")
        if not 0.0 < args.patch_size_quantile <= 1.0:
            raise ValueError("--patch-size-quantile must be in (0,1]")
        if args.checkpoint_every_graphs <= 0:
            raise ValueError("--checkpoint-every-graphs must be positive")
    elif args.command == "train":
        positive_names = [
            "n_atoms",
            "sparsity",
            "epochs",
            "graphs_per_batch",
            "patches_per_graph",
            "dictionary_sweeps",
            "power_iterations",
            "eval_every",
        ]
        for name in positive_names:
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.sparsity > args.n_atoms:
            raise ValueError("--sparsity cannot exceed --n-atoms")
        if args.replay_capacity < 0 or args.replay_per_batch < 0:
            raise ValueError("Replay sizes cannot be negative")
        if args.monitor_train_graphs < 0 or args.monitor_valid_graphs < 0:
            raise ValueError("Monitor graph counts cannot be negative")
        if args.eval_patches_per_graph < 0:
            raise ValueError("--eval-patches-per-graph cannot be negative")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
