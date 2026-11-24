"""Data for GSRO v0: reuse the committed label-free mollipo cache + split.

The only new data work in this round is the fixed *node* input (raw OGB atom
categoricals + the joint ``atomic number x bond type`` one-hop count vector,
log1p then Train-only scaling) and the fixed *bond* categorical input.  The raw
graph, physical-bond convention, official scaffold split and row ids come from
``tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz``
via that track's loader (SHA-256 pinned in the config).

No label is used for any feature, vocabulary or scaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import gsro_common as C

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Graph:
    row_id: int
    n: int
    atom_cat: np.ndarray  # [n, 9] int64 (vocabulary indices, 0 = unseen)
    nb_counts: np.ndarray  # [n, P+1] float32 (log1p + Train z-score)
    bond_cat: np.ndarray  # [E, 3] int64 (vocabulary indices)
    bu: np.ndarray  # [E] int64 local endpoint u
    bv: np.ndarray  # [E] int64 local endpoint v
    y: float  # raw target
    y_std: float  # standardised target

    @property
    def n_bonds(self) -> int:
        return int(self.bu.size)


@dataclass
class Split:
    name: str
    graphs: list[Graph]

    @property
    def row_ids(self) -> np.ndarray:
        return np.asarray([g.row_id for g in self.graphs], dtype=np.int64)

    @property
    def y(self) -> np.ndarray:
        return np.asarray([g.y for g in self.graphs], dtype=np.float64)

    @property
    def y_std(self) -> np.ndarray:
        return np.asarray([g.y_std for g in self.graphs], dtype=np.float64)

    @property
    def n_atoms(self) -> np.ndarray:
        return np.asarray([g.n for g in self.graphs], dtype=np.int64)


@dataclass
class PreparedData:
    payload_sha256: str
    payload_path: str
    splits: dict[str, Split]
    atom_vocab: dict[int, dict[int, int]]  # column -> raw value -> index (1..)
    atom_vocab_sizes: list[int]
    bond_vocab: dict[int, dict[int, int]]
    bond_vocab_sizes: list[int]
    joint_pairs: list[tuple[int, int]]  # Train pair vocabulary, unknown = P
    count_mean: np.ndarray
    count_std: np.ndarray
    n_bar: float
    mu_y: float
    sigma_y: float
    n_unseen_atom_values: int
    n_unknown_pair_occurrences: int

    @property
    def n_joint_features(self) -> int:
        return len(self.joint_pairs) + 1

    @property
    def node_input_dim(self) -> int:
        return ATOM_COLUMNS * 16 + self.n_joint_features

    def identity(self) -> dict[str, Any]:
        return {
            "cache_sha256": self.payload_sha256,
            "split_sizes": {name: len(self.splits[name].graphs) for name in SPLITS},
            "split_rowid_sha256": {
                name: C.sha256_array(self.splits[name].row_ids) for name in SPLITS
            },
            "atom_vocab_sizes": list(self.atom_vocab_sizes),
            "bond_vocab_sizes": list(self.bond_vocab_sizes),
            "joint_pair_vocabulary": [list(p) for p in self.joint_pairs],
            "n_joint_features": self.n_joint_features,
            "node_input_dim": self.node_input_dim,
            "n_bar": self.n_bar,
            "mu_y": self.mu_y,
            "sigma_y": self.sigma_y,
            "n_unseen_atom_values": self.n_unseen_atom_values,
            "n_unknown_pair_occurrences": self.n_unknown_pair_occurrences,
            "count_feature_mean": self.count_mean,
            "count_feature_std": self.count_std,
        }


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------


def _train_vocab(values: np.ndarray, columns: int) -> tuple[dict[int, dict[int, int]], list[int]]:
    vocab: dict[int, dict[int, int]] = {}
    sizes: list[int] = []
    for j in range(columns):
        raw = sorted(int(v) for v in np.unique(values[:, j]).tolist())
        vocab[j] = {value: index + 1 for index, value in enumerate(raw)}
        sizes.append(len(raw) + 1)  # index 0 = unseen
    return vocab, sizes


def _map_column(raw: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros(raw.shape[0], dtype=np.int64)
    for value, index in mapping.items():
        out[raw == value] = index
    return out


def _joint_pair_counts(
    n: int,
    z: np.ndarray,
    bu: np.ndarray,
    bv: np.ndarray,
    bond_type: np.ndarray,
    pair_index: dict[tuple[int, int], int],
    n_pairs: int,
) -> np.ndarray:
    """Joint ``(neighbour atomic number, bond type)`` counts per atom.

    The last column is the explicit unknown bucket for a pair unseen in Train.
    """
    counts = np.zeros((n, n_pairs + 1), dtype=np.float64)
    for a, b, bt in zip(bu.tolist(), bv.tolist(), bond_type.tolist()):
        for src, dst in ((a, b), (b, a)):
            key = (int(z[dst]), int(bt))
            slot = pair_index.get(key, n_pairs)
            counts[src, slot] += 1.0
    return counts


def load_prepared(config: Mapping[str, Any] | None = None) -> PreparedData:
    config = C.load_config() if config is None else config
    sq_data = C.import_sq_data()
    cache_path = C.SQ_DATA_DIR / str(config["cache_file"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"reused mollipo cache missing: {cache_path}")
    payload = sq_data.load_cache(
        str(config["dataset"]),
        base=C.SQ_DATA_DIR,
        verify_sha=str(config["cache_sha256"]),
    )

    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"], dtype=np.int64)
    bond_feat = np.asarray(payload["bond_feat"], dtype=np.int64)
    bond_u = np.asarray(payload["bond_u"], dtype=np.int64)
    bond_v = np.asarray(payload["bond_v"], dtype=np.int64)
    labels = np.asarray(payload["labels"], dtype=np.float64)
    split_indices = {name: np.asarray(payload[f"{name}_idx"], dtype=np.int64) for name in SPLITS}
    for name in SPLITS:
        if name == "train" and split_indices[name].size != 3360:
            raise ValueError(f"unexpected train split size {split_indices[name].size}")
        if name != "train" and split_indices[name].size != 420:
            raise ValueError(f"unexpected {name} split size {split_indices[name].size}")

    train_idx = split_indices["train"]
    train_atoms = np.concatenate(
        [atom_feat[node_ptr[g] : node_ptr[g + 1]] for g in train_idx.tolist()], axis=0
    )
    train_bonds = np.concatenate(
        [bond_feat[bond_ptr[g] : bond_ptr[g + 1]] for g in train_idx.tolist()], axis=0
    )
    atom_vocab, atom_vocab_sizes = _train_vocab(train_atoms, ATOM_COLUMNS)
    bond_vocab, bond_vocab_sizes = _train_vocab(train_bonds, BOND_COLUMNS)

    # Joint pair vocabulary: observed directed (neighbour Z, bond type) in Train.
    pair_set: set[tuple[int, int]] = set()
    for g in train_idx.tolist():
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        z = atom_feat[a0:a1, 0]
        for a, b, bt in zip(
            bond_u[b0:b1].tolist(), bond_v[b0:b1].tolist(), bond_feat[b0:b1, 0].tolist()
        ):
            pair_set.add((int(z[a]), int(bt)))
            pair_set.add((int(z[b]), int(bt)))
    joint_pairs = sorted(pair_set)
    pair_index = {pair: index for index, pair in enumerate(joint_pairs)}
    n_pairs = len(joint_pairs)

    # Raw counts for every graph, then Train-only scaling.
    raw_counts: dict[int, np.ndarray] = {}
    n_unseen_atom = 0
    n_unknown_pairs = 0
    for g in range(int(node_ptr.size) - 1):
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        z = atom_feat[a0:a1, 0]
        raw = atom_feat[a0:a1]
        for j in range(ATOM_COLUMNS):
            n_unseen_atom += int(
                np.sum(~np.isin(raw[:, j], list(atom_vocab[j].keys())))
            )
        counts = _joint_pair_counts(
            a1 - a0,
            z,
            bond_u[b0:b1],
            bond_v[b0:b1],
            bond_feat[b0:b1, 0],
            pair_index,
            n_pairs,
        )
        n_unknown_pairs += int(counts[:, n_pairs].sum())
        raw_counts[g] = np.log1p(counts)

    train_count_rows = np.concatenate([raw_counts[g] for g in train_idx.tolist()], axis=0)
    count_mean = train_count_rows.mean(axis=0)
    count_std = train_count_rows.std(axis=0, ddof=0)
    count_std = np.where(count_std < 1e-12, 1.0, count_std)

    mu_y = float(labels[train_idx].mean())
    sigma_y = float(labels[train_idx].std(ddof=0))
    if not np.isfinite(sigma_y) or sigma_y <= 0:
        raise ValueError("degenerate Train target scale")
    n_bar = float(np.mean([node_ptr[g + 1] - node_ptr[g] for g in train_idx.tolist()]))

    splits: dict[str, Split] = {}
    for name in SPLITS:
        graphs: list[Graph] = []
        for g in split_indices[name].tolist():
            a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
            b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
            raw = atom_feat[a0:a1]
            atom_cat = np.stack(
                [_map_column(raw[:, j], atom_vocab[j]) for j in range(ATOM_COLUMNS)], axis=1
            )
            bond_cat = np.stack(
                [
                    _map_column(bond_feat[b0:b1, j], bond_vocab[j])
                    for j in range(BOND_COLUMNS)
                ],
                axis=1,
            )
            scaled = ((raw_counts[g] - count_mean[None, :]) / count_std[None, :]).astype(
                np.float32
            )
            y_val = float(labels[g])
            graphs.append(
                Graph(
                    row_id=int(g),
                    n=int(a1 - a0),
                    atom_cat=atom_cat.astype(np.int64),
                    nb_counts=scaled,
                    bond_cat=bond_cat.astype(np.int64),
                    bu=bond_u[b0:b1].astype(np.int64),
                    bv=bond_v[b0:b1].astype(np.int64),
                    y=y_val,
                    y_std=(y_val - mu_y) / sigma_y,
                )
            )
        splits[name] = Split(name=name, graphs=graphs)

    return PreparedData(
        payload_sha256=str(payload["cache_sha256"]),
        payload_path=str(payload["cache_path"]),
        splits=splits,
        atom_vocab=atom_vocab,
        atom_vocab_sizes=atom_vocab_sizes,
        bond_vocab=bond_vocab,
        bond_vocab_sizes=bond_vocab_sizes,
        joint_pairs=joint_pairs,
        count_mean=count_mean.astype(np.float64),
        count_std=count_std.astype(np.float64),
        n_bar=n_bar,
        mu_y=mu_y,
        sigma_y=sigma_y,
        n_unseen_atom_values=int(n_unseen_atom),
        n_unknown_pair_occurrences=int(n_unknown_pairs),
    )


# ---------------------------------------------------------------------------
# batching (fixed size-sorted buckets; batch order is shuffled per epoch)
# ---------------------------------------------------------------------------


def _as_tensor(array: np.ndarray, device, dtype):
    import torch

    return torch.as_tensor(np.ascontiguousarray(array), device=device, dtype=dtype)


def collate(graphs: Sequence[Graph], device) -> dict[str, Any]:
    """Pad a group of molecules into dense tensors (padding is inert)."""
    import torch

    batch = len(graphs)
    n_max = max(g.n for g in graphs)
    e_max = max(g.n_bonds for g in graphs)
    n_joint = int(graphs[0].nb_counts.shape[1])

    atom_cat = np.zeros((batch, n_max, ATOM_COLUMNS), dtype=np.int64)
    nb_counts = np.zeros((batch, n_max, n_joint), dtype=np.float64)
    node_mask = np.zeros((batch, n_max), dtype=bool)
    bond_cat = np.zeros((batch, e_max, BOND_COLUMNS), dtype=np.int64)
    bu = np.zeros((batch, e_max), dtype=np.int64)
    bv = np.zeros((batch, e_max), dtype=np.int64)
    edge_mask = np.zeros((batch, e_max), dtype=bool)
    n = np.zeros(batch, dtype=np.float64)
    row_ids = np.zeros(batch, dtype=np.int64)
    y = np.zeros(batch, dtype=np.float64)
    y_std = np.zeros(batch, dtype=np.float64)

    for i, g in enumerate(graphs):
        atom_cat[i, : g.n] = g.atom_cat
        nb_counts[i, : g.n] = g.nb_counts
        node_mask[i, : g.n] = True
        bond_cat[i, : g.n_bonds] = g.bond_cat
        bu[i, : g.n_bonds] = g.bu
        bv[i, : g.n_bonds] = g.bv
        edge_mask[i, : g.n_bonds] = True
        n[i] = g.n
        row_ids[i] = g.row_id
        y[i] = g.y
        y_std[i] = g.y_std

    return {
        "atom_cat": _as_tensor(atom_cat, device, torch.long),
        "nb_counts": _as_tensor(nb_counts, device, torch.float32),
        "node_mask": _as_tensor(node_mask, device, torch.bool),
        "bond_cat": _as_tensor(bond_cat, device, torch.long),
        "bu": _as_tensor(bu, device, torch.long),
        "bv": _as_tensor(bv, device, torch.long),
        "edge_mask": _as_tensor(edge_mask, device, torch.bool),
        "n": _as_tensor(n, device, torch.float32),
        "row_ids": _as_tensor(row_ids, device, torch.long),
        "y": _as_tensor(y, device, torch.float64),
        "y_std": _as_tensor(y_std, device, torch.float32),
    }


def single_graph(graph: Graph, device, dtype=None):
    """One molecule as unpadded tensors (used by the checks / explanation)."""
    import torch

    dtype = torch.float32 if dtype is None else dtype
    return {
        "atom_cat": _as_tensor(graph.atom_cat[None], device, torch.long),
        "nb_counts": _as_tensor(graph.nb_counts[None], device, dtype),
        "node_mask": _as_tensor(np.ones((1, graph.n), dtype=bool), device, torch.bool),
        "bond_cat": _as_tensor(graph.bond_cat[None], device, torch.long),
        "bu": _as_tensor(graph.bu[None], device, torch.long),
        "bv": _as_tensor(graph.bv[None], device, torch.long),
        "edge_mask": _as_tensor(np.ones((1, graph.n_bonds), dtype=bool), device, torch.bool),
        "n": _as_tensor(np.asarray([graph.n], dtype=np.float64), device, dtype),
        "row_ids": _as_tensor(np.asarray([graph.row_id], dtype=np.int64), device, torch.long),
        "y": _as_tensor(np.asarray([graph.y], dtype=np.float64), device, dtype),
        "y_std": _as_tensor(np.asarray([graph.y_std], dtype=np.float64), device, dtype),
    }


def size_sorted_batches(split: Split, batch_size: int, device) -> list[dict[str, Any]]:
    """Sort by atom count and chunk: minimal padding, fixed membership."""
    order = sorted(range(len(split.graphs)), key=lambda i: (split.graphs[i].n, split.graphs[i].row_id))
    batches: list[dict[str, Any]] = []
    for start in range(0, len(order), batch_size):
        group = [split.graphs[i] for i in order[start : start + batch_size]]
        batches.append(collate(group, device))
    return batches
