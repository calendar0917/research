"""Compact-v4 T=2 recurrent pair--centre + ordered shortest-path bond encoder.

Hypothesis under test
---------------------
The T=2 weight-tied recurrent pair--centre skeleton is treated as frozen.  The
suspicion is that the remaining bottleneck is the *relation primitive* itself:
the current shortest-path descriptor keeps the path length and the bond
*composition averaged over shortest paths* (plus the adjacent-bond one-hot for
``d = 1``), but discards the **order** in which the bond types appear along the
path.

Phase A (audit, no training) measures how often two pairs share the exact
path-related part of the descriptor but have a different ordered shortest-path
bond sequence (aliasing / collision).

Phase B adds a minimal ordered bond-sequence encoder.  The only change to the
relation primitive is

    q_ij = Q(h_i, h_j, r_ij, s_ij)

where ``s_ij`` is a small, position-aware, reversal-invariant encoding of the
ordered bond sequence.  The existing recurrent skeleton (``h0 -> q0 -> A0 ->
h1 -> q1 -> A1 -> h2 -> readout``), the distance buckets, the centre pooling,
the head, the optimizer and the training protocol are all untouched.

The path encoder consumes a fixed-size, permutation-invariant summary of the
shortest-path bond sequences:

    counts[pos, bond] = mean over shortest paths of 1[bond at position]

with each path first canonicalised to ``min(seq, reverse(seq))`` so that
``(i, j)`` and ``(j, i)`` receive identical encodings.  Positions beyond
``PATH_MAX_LEN`` are folded into the last slot.  The encoder is

    x[pos] = sum_b counts[pos, b] * bond_embedding[b] + position_embedding[pos]
    s      = MLP(flatten(x))            # -> PATH_DIM

which is a small shared bond embedding plus a learned position embedding,
followed by a two-layer MLP.  No Transformer, no RNN, no atom sequence.

Phase D adds a *matched* orderless control: identical architecture, identical
parameter count, identical computation graph, but the bond types of every path
are sorted before the same counts are computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rc
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc, _data_to_graph

# ---------------------------------------------------------------------------
# frozen locations / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/compact_v4_ordered_path_bond"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
RUNS_DIR = RESULTS_DIR / "runs"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURE_DIR = RESULTS_DIR / "figures"

PROTOCOL_VERSION = "compact_v4_ordered_path_bond_v1"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

BOND_CATEGORIES = int(zpp.BOND_CATEGORIES)
ATOM_CATEGORIES = int(zpp.ATOM_CATEGORIES)

# Path-encoder geometry.  Deliberately tiny.
PATH_MAX_LEN = 12
PATH_TOKEN_DIM = 8
PATH_HIDDEN = 16
PATH_DIM = 8
PATH_MODE_FIELD = {
    "ordered": "pair_path_counts",
    "orderless": "pair_path_counts_sorted",
}
ENUM_CAP = 512

# Frozen reference: T=2 recurrent pair--centre official-valid MAE.
REF_RECURRENT_SEED0 = 0.13837560486892472
REF_RECURRENT_SEED1 = 0.13343997858563672
REF_RECURRENT_PARAMS = 82115
EXPECTED_SMALL_TOTAL = 82115

# Pre-registered Phase-C / Phase-D gates (positive => candidate better).
GATE_STRONG = 0.002
GATE_WEAK = 0.001
NO_SIGNAL_BAND = 0.001


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _n_params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# shortest-path bond sequences
# ---------------------------------------------------------------------------


def _adjacency_bond(graph: Any, edge_types: Mapping[tuple[int, int], int]):
    return {
        int(u): sorted(
            (int(v), int(edge_types[graph.edge_key(int(u), int(v))]))
            for v in graph.neighbors(int(u))
        )
        for u in graph.nodes
    }


def _bfs_parents(adj_bond: Mapping[int, Sequence[tuple[int, int]]], src: int):
    """Shortest-path DAG parents (parent, bond) for every node reachable from src."""
    distances = {int(src): 0}
    parents: dict[int, list[tuple[int, int]]] = defaultdict(list)
    queue = deque([int(src)])
    while queue:
        node = queue.popleft()
        for neighbor, bond in adj_bond[node]:
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                parents[int(neighbor)].append((int(node), int(bond)))
                queue.append(int(neighbor))
            elif distances[int(neighbor)] == distances[node] + 1:
                parents[int(neighbor)].append((int(node), int(bond)))
    return distances, parents


def _enumerate_shortest_paths(
    src: int,
    tgt: int,
    parents: Mapping[int, Sequence[tuple[int, int]]],
    cap: int,
) -> list[tuple[int, ...]]:
    """All shortest paths ``src -> tgt`` as **node sequences** (src first)."""
    if src == tgt:
        return [(int(src),)]
    out: list[tuple[int, ...]] = []
    stack: list[tuple[int, tuple[int, ...]]] = [(int(tgt), (int(tgt),))]
    while stack:
        node, path = stack.pop()
        for parent, _bond in parents.get(node, ()):  # type: ignore[arg-type]
            new_path = (int(parent),) + path
            if parent == src:
                out.append(new_path)
                if len(out) > cap:
                    return out
            else:
                stack.append((int(parent), new_path))
    return out


def _bond_sequence(node_path: Sequence[int], graph: Any, edge_types) -> tuple[int, ...]:
    return tuple(
        int(edge_types[graph.edge_key(int(node_path[k]), int(node_path[k + 1]))])
        for k in range(len(node_path) - 1)
    )


def _canonical_bond(sequence: Sequence[int]) -> tuple[int, ...]:
    forward = tuple(int(x) for x in sequence)
    backward = forward[::-1]
    return forward if forward <= backward else backward


def _canonical_pair(
    node_path: Sequence[int],
    graph: Any,
    edge_types,
    node_types: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (canonical bond sequence, aligned atom-type sequence)."""
    bonds = _bond_sequence(node_path, graph, edge_types)
    atoms = tuple(int(node_types[int(v)]) for v in node_path)
    if bonds <= bonds[::-1]:
        return bonds, atoms
    return bonds[::-1], atoms[::-1]


def _accumulate(counts: np.ndarray, sequence: Sequence[int], max_len: int) -> None:
    for position, token in enumerate(sequence):
        slot = position if position < max_len else max_len - 1
        counts[slot, int(token)] += 1.0


def _process_graph(data: Any, max_len: int, cap: int):
    """Ordered and orderless path-count tensors for one raw molecule.

    Rows follow the canonical ``(i, j), i < j`` order used by
    ``zpp._graph_record`` and therefore align with ``data.pair_index``.
    """
    graph, _node_types, edge_types = _data_to_graph(data)
    adj_bond = _adjacency_bond(graph, edge_types)
    centers = list(graph.nodes)
    n = len(centers)
    n_pairs = n * (n - 1) // 2
    counts = np.zeros((n_pairs, max_len, BOND_CATEGORIES), dtype=np.float32)
    counts_sorted = np.zeros_like(counts)
    row = 0
    for i_index in range(n):
        src = int(centers[i_index])
        _distances, parents = _bfs_parents(adj_bond, src)
        for j_index in range(i_index + 1, n):
            tgt = int(centers[j_index])
            paths = _enumerate_shortest_paths(src, tgt, parents, cap)
            if not paths:
                raise RuntimeError(f"unreachable pair {src}->{tgt}")
            n_paths = float(len(paths))
            for node_path in paths:
                bonds = _bond_sequence(node_path, graph, edge_types)
                canonical = _canonical_bond(bonds)
                _accumulate(counts[row], canonical, max_len)
                _accumulate(counts_sorted[row], tuple(sorted(bonds)), max_len)
            counts[row] /= n_paths
            counts_sorted[row] /= n_paths
            row += 1
    return counts, counts_sorted


# ---------------------------------------------------------------------------
# data: attach ordered / orderless path counts to the frozen encoded datasets
# ---------------------------------------------------------------------------

_CACHE_VERSION = f"path_counts_v2_L{PATH_MAX_LEN}_cap{ENUM_CAP}"


def _compute_split_counts(raw_dataset):
    counts_parts: list[np.ndarray] = []
    sorted_parts: list[np.ndarray] = []
    offsets = [0]
    started = time.perf_counter()
    for index, data in enumerate(raw_dataset):
        counts, counts_sorted = _process_graph(data, PATH_MAX_LEN, ENUM_CAP)
        counts_parts.append(counts)
        sorted_parts.append(counts_sorted)
        offsets.append(offsets[-1] + counts.shape[0])
        if index and index % 2000 == 0:
            print(
                f"path-counts {index}/{len(raw_dataset)} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    return (
        np.concatenate(counts_parts, axis=0),
        np.concatenate(sorted_parts, axis=0),
        np.asarray(offsets, dtype=np.int64),
    )


def _zinc_split(split: str) -> str:
    return "val" if str(split) in {"valid", "val"} else str(split)


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"{_CACHE_VERSION}_{split}.npz"


def _load_or_compute(split: str, *, force: bool = False):
    cache = _cache_path(split)
    if cache.exists() and not force:
        payload = np.load(cache)
        return payload["counts"], payload["counts_sorted"], payload["offsets"]
    raw = _load_zinc(ZINC_ROOT, _zinc_split(split))
    counts, counts_sorted, offsets = _compute_split_counts(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, counts=counts, counts_sorted=counts_sorted, offsets=offsets)
    return counts, counts_sorted, offsets


def _attach_split(data_list: Sequence[Data], counts, counts_sorted, offsets) -> None:
    if len(data_list) != len(offsets) - 1:
        raise RuntimeError("split length mismatch")
    for index, data in enumerate(data_list):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        expected = int(data.pair_index.shape[1])
        if hi - lo != expected:
            raise RuntimeError(
                f"pair count mismatch at graph {index}: counts={hi - lo} data={expected}"
            )
        data.pair_path_counts = torch.from_numpy(counts[lo:hi].copy())
        data.pair_path_counts_sorted = torch.from_numpy(counts_sorted[lo:hi].copy())


def load_augmented_encoded(*, force: bool = False):
    """Frozen encoded train/valid datasets + attached path-bond statistics."""
    train_data, valid_data, audit = shead.build_encoded_records()
    for split, data_list in (("train", train_data), ("valid", valid_data)):
        counts, counts_sorted, offsets = _load_or_compute(split, force=force)
        _attach_split(data_list, counts, counts_sorted, offsets)
    return train_data, valid_data, audit


# ---------------------------------------------------------------------------
# Phase A: path information audit
# ---------------------------------------------------------------------------


def _f32_key(value: Any) -> bytes:
    return struct.pack("<f", float(value))


def _f32_array_key(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype=np.float32).tobytes()


def audit_path_information(*, split: str = "valid", cap: int = ENUM_CAP) -> dict[str, Any]:
    """Audit ordered bond-sequence aliasing of the current path descriptor.

    Only the *path-related* part of ``pair_relation`` is used as the grouping
    key: ``(distance, path_count, path_bond_mean)`` -- taken verbatim from the
    production ``zpp._shortest_path_summary``.  Endpoint / patch-overlap
    features are deliberately excluded so they cannot mask a pure path-order
    loss.
    """
    raw = _load_zinc(ZINC_ROOT, _zinc_split(split))
    distance_hist: Counter[int] = Counter()
    path_count_hist: Counter[int] = Counter()
    groups: dict[int, dict[bytes, set]] = defaultdict(lambda: defaultdict(set))
    group_counts: dict[int, dict[bytes, int]] = defaultdict(lambda: defaultdict(int))
    atom_groups: dict[int, dict[bytes, set]] = defaultdict(lambda: defaultdict(set))
    atom_group_counts: dict[int, dict[bytes, int]] = defaultdict(lambda: defaultdict(int))
    n_pairs = 0
    n_multi = 0
    n_skipped = 0
    max_len = 0
    started = time.perf_counter()
    for index, data in enumerate(raw):
        graph, node_types, edge_types = _data_to_graph(data)
        adj_bond = _adjacency_bond(graph, edge_types)
        centers = list(graph.nodes)
        n = len(centers)
        for i_index in range(n):
            src = int(centers[i_index])
            distances, parents = _bfs_parents(adj_bond, src)
            summary = zpp._shortest_path_summary(graph, src, edge_types)
            for j_index in range(i_index + 1, n):
                tgt = int(centers[j_index])
                distance, path_count, bond_mean = summary[tgt]
                paths = _enumerate_shortest_paths(src, tgt, parents, cap)
                n_pairs += 1
                distance_hist[int(distance)] += 1
                path_count_hist[min(len(paths), 50)] += 1
                max_len = max(max_len, int(distance))
                if len(paths) > 1:
                    n_multi += 1
                if len(paths) > cap:
                    n_skipped += 1
                    continue
                signatures = set()
                atom_signatures = set()
                for node_path in paths:
                    canonical, atoms = _canonical_pair(
                        node_path, graph, edge_types, node_types
                    )
                    signatures.add(canonical)
                    atom_signatures.add(atoms)
                key = (
                    int(distance),
                    _f32_key(path_count),
                    _f32_array_key(bond_mean),
                )
                groups[int(distance)][key].add(tuple(sorted(signatures)))
                group_counts[int(distance)][key] += 1
                atom_groups[int(distance)][key].add(tuple(sorted(atom_signatures)))
                atom_group_counts[int(distance)][key] += 1
        if index and index % 200 == 0:
            print(
                f"audit {index}/{len(raw)} elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    by_length: list[dict[str, Any]] = []
    total_mixed_pairs = 0
    total_pairs_used = 0
    total_atom_mixed_pairs = 0
    for distance in sorted(groups):
        mixed_groups = 0
        mixed_pairs = 0
        pairs_here = 0
        for key, signatures in groups[distance].items():
            count = group_counts[distance][key]
            pairs_here += count
            if len(signatures) > 1:
                mixed_groups += 1
                mixed_pairs += count
        atom_mixed_pairs = 0
        for key, signatures in atom_groups[distance].items():
            if len(signatures) > 1:
                atom_mixed_pairs += atom_group_counts[distance][key]
        total_mixed_pairs += mixed_pairs
        total_atom_mixed_pairs += atom_mixed_pairs
        total_pairs_used += pairs_here
        by_length.append(
            {
                "distance": int(distance),
                "pairs": int(pairs_here),
                "groups": int(len(groups[distance])),
                "mixed_groups": int(mixed_groups),
                "mixed_pairs": int(mixed_pairs),
                "collision_fraction": float(mixed_pairs / max(pairs_here, 1)),
                "atom_mixed_pairs": int(atom_mixed_pairs),
                "atom_collision_fraction": float(atom_mixed_pairs / max(pairs_here, 1)),
            }
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "split": str(split),
        "n_pairs": int(n_pairs),
        "n_multiple_shortest_paths": int(n_multi),
        "fraction_multiple_shortest_paths": float(n_multi / max(n_pairs, 1)),
        "n_skipped_over_cap": int(n_skipped),
        "enumeration_cap": int(cap),
        "max_shortest_path_length": int(max_len),
        "distance_histogram": {int(k): int(v) for k, v in sorted(distance_hist.items())},
        "path_count_histogram": {int(k): int(v) for k, v in sorted(path_count_hist.items())},
        "grouping_key": "distance + path_count + path_bond_mean (path-related descriptor only)",
        "multiple_shortest_path_handling": (
            "each path canonicalised to min(seq, reverse(seq)); the true signature is "
            "the sorted multiset of canonical bond sequences over all shortest paths"
        ),
        "bond_collision_fraction_overall": float(total_mixed_pairs / max(total_pairs_used, 1)),
        "atom_collision_fraction_overall": float(total_atom_mixed_pairs / max(total_pairs_used, 1)),
        "by_path_length": by_length,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"path_audit_{split}.json", summary)
    _write_csv(
        RESULTS_DIR / f"path_audit_{split}_by_length.csv",
        by_length,
        (
            "distance",
            "pairs",
            "groups",
            "mixed_groups",
            "mixed_pairs",
            "collision_fraction",
            "atom_mixed_pairs",
            "atom_collision_fraction",
        ),
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_pairs",
                    "fraction_multiple_shortest_paths",
                    "bond_collision_fraction_overall",
                    "atom_collision_fraction_overall",
                )
            },
            indent=2,
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Phase B: ordered bond-sequence encoder
# ---------------------------------------------------------------------------


class OrderedPathRecurrentPairCentreModel(rc.PatchPathRecurrentPairCentreModel):
    """T=2 recurrent pair--centre with an ordered shortest-path bond encoder.

    The recurrent skeleton, shared modules, readout, head and every other
    tensor are inherited unchanged.  The only modification is the relation
    primitive: a small ``s_ij`` is concatenated to the pair-encoder input.
    """

    def __init__(
        self,
        typed_vocabulary_size: int,
        parent_vocabulary_size: int,
        *,
        path_dim: int = PATH_DIM,
        path_token_dim: int = PATH_TOKEN_DIM,
        path_hidden: int = PATH_HIDDEN,
        path_max_len: int = PATH_MAX_LEN,
        path_mode: str = "ordered",
        **kwargs: Any,
    ) -> None:
        super().__init__(typed_vocabulary_size, parent_vocabulary_size, **kwargs)
        if path_mode not in PATH_MODE_FIELD:
            raise ValueError(f"unknown path_mode={path_mode!r}")
        self.path_dim = int(path_dim)
        self.path_token_dim = int(path_token_dim)
        self.path_hidden = int(path_hidden)
        self.path_max_len = int(path_max_len)
        self.path_mode = str(path_mode)
        self._path_field = PATH_MODE_FIELD[self.path_mode]

        # -- path encoder (shared bond embedding + learned position) --------
        self.path_bond_embedding = nn.Embedding(BOND_CATEGORIES, self.path_token_dim)
        self.path_position_embedding = nn.Embedding(self.path_max_len, self.path_token_dim)
        self.path_encoder = nn.Sequential(
            nn.Linear(self.path_max_len * self.path_token_dim, self.path_hidden),
            nn.ReLU(),
            nn.Linear(self.path_hidden, self.path_dim),
        )

        # -- widen the pair encoder, preserving the baseline columns --------
        dropout = float(self.pair_encoder.layers[3].p)
        self._extend_pair_encoder(dropout)

        # Diagnostics / mechanism switches (never parameters).
        self.force_zero_path = False
        self._cached_path_encoding: torch.Tensor | None = None

    def _extend_pair_encoder(self, dropout: float) -> None:
        old = self.pair_encoder
        old_weight = old.layers[0].weight
        old_bias = old.layers[0].bias
        old_width = int(old_weight.shape[1])
        hidden = int(old_weight.shape[0])
        output = int(old.layers[4].weight.shape[0])
        new = zpp._MLPBlock(old_width + self.path_dim, hidden, output, dropout)
        with torch.no_grad():
            new.layers[0].weight.zero_()
            new.layers[0].weight[:, :old_width].copy_(old_weight)
            # Small non-zero init on the new path columns.  A zero-init final
            # fusion is known to give exactly zero upstream gradient at step 0
            # and can collapse the new branch (see the compact-v6 repair note);
            # a small normal keeps the path branch demonstrably alive from the
            # first step while leaving the shared baseline columns exact.
            nn.init.normal_(
                new.layers[0].weight[:, old_width:], mean=0.0, std=0.01
            )
            new.layers[0].bias.copy_(old_bias)
            new.layers[1].load_state_dict(old.layers[1].state_dict())
            new.layers[4].load_state_dict(old.layers[4].state_dict())
        self.pair_encoder = new

    # -- path encoding -------------------------------------------------------
    def _compute_path_encoding(self, data: Data) -> torch.Tensor | None:
        if self.force_zero_path:
            return None
        counts = getattr(data, self._path_field, None)
        if counts is None:
            return None
        counts = counts.to(dtype=self.path_bond_embedding.weight.dtype)
        x = torch.einsum("plb,bd->pld", counts, self.path_bond_embedding.weight)
        x = x + self.path_position_embedding.weight.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        return self.path_encoder(x)

    def encode(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        self._cached_path_encoding = self._compute_path_encoding(data)
        return super().encode(data)

    def encode_original(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        self._cached_path_encoding = self._compute_path_encoding(data)
        return super().encode_original(data)

    def _pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        product = left * right
        path_encoding = self._cached_path_encoding
        if path_encoding is None:
            path_encoding = relation.new_zeros((int(relation.shape[0]), self.path_dim))
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                product * gate,
                relation,
                path_encoding,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)


# ---------------------------------------------------------------------------
# builders (exact shared-init matching with the recurrent baseline)
# ---------------------------------------------------------------------------


def build_ordered(
    seed: int = 0,
    *,
    path_mode: str = "ordered",
    typed_vocabulary_size: int = 6785,
    parent_vocabulary_size: int = 32,
    head_seed: int = shead.SMALL_HEAD_SEED,
    **path_kwargs: Any,
) -> OrderedPathRecurrentPairCentreModel:
    baseline = shead.build_baseline(seed)
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    shead._seed_everything(seed)
    model = OrderedPathRecurrentPairCentreModel(
        int(typed_vocabulary_size),
        int(parent_vocabulary_size),
        recurrence_rounds=int(rc.RECURRENCE_ROUNDS),
        recurrence_enabled=True,
        recurrence_mode="refresh",
        path_mode=str(path_mode),
        **path_kwargs,
        **shead._base_kwargs(),
    )
    with torch.no_grad():
        state = model.state_dict()
        for key, value in baseline_state.items():
            if key in state and state[key].shape == value.shape:
                state[key].copy_(value)
    torch.manual_seed(int(head_seed))
    model.head = GenericReader(int(model.unified_graph_width), shead.SMALL_HEAD_HIDDEN)
    return model


def build_orderless(seed: int = 0, **path_kwargs: Any) -> OrderedPathRecurrentPairCentreModel:
    return build_ordered(seed, path_mode="orderless", **path_kwargs)


def ordered_param_count(seed: int = 0) -> int:
    return _n_params(build_ordered(seed))


# ---------------------------------------------------------------------------
# training plumbing
# ---------------------------------------------------------------------------


def train(
    build_fn,
    train_data,
    valid_data,
    *,
    seed: int,
    tag: str,
    expected_total: int,
):
    original = (shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR)
    shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = CURVE_DIR, STATE_DIR, RUNS_DIR
    try:
        return shead.train_model(
            build_fn=build_fn,
            train_data=train_data,
            valid_data=valid_data,
            seed=int(seed),
            tag=str(tag),
            save_state=True,
            real_batch_identity=True,
            expected_total=int(expected_total),
        )
    finally:
        shead.CURVE_DIR, shead.STATE_DIR, shead.RUNS_DIR = original


def run_ordered(seed: int = 0, *, force_data: bool = False) -> dict[str, Any]:
    train_data, valid_data, _ = load_augmented_encoded(force=force_data)
    expected = ordered_param_count(seed)
    summary = train(
        lambda s: build_ordered(s),
        train_data,
        valid_data,
        seed=int(seed),
        tag="ordered",
        expected_total=expected,
    )
    _write_json(RESULTS_DIR / f"ordered_seed{seed}.json", summary)
    return summary


def run_orderless(seed: int = 0, *, force_data: bool = False) -> dict[str, Any]:
    train_data, valid_data, _ = load_augmented_encoded(force=force_data)
    expected = _n_params(build_orderless(seed))
    summary = train(
        lambda s: build_orderless(s),
        train_data,
        valid_data,
        seed=int(seed),
        tag="orderless",
        expected_total=expected,
    )
    _write_json(RESULTS_DIR / f"orderless_seed{seed}.json", summary)
    return summary


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------


def _recurrent_state_matches(ordered: nn.Module, recurrent: nn.Module) -> dict[str, Any]:
    ordered_state = ordered.state_dict()
    recurrent_state = recurrent.state_dict()
    matched = []
    identical = True
    max_diff = 0.0
    for key, value in recurrent_state.items():
        if key not in ordered_state or ordered_state[key].shape != value.shape:
            continue
        diff = float((ordered_state[key] - value).abs().max())
        matched.append(key)
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            identical = False
    return {
        "n_matched_tensors": len(matched),
        "max_abs_diff": max_diff,
        "all_shared_tensors_identical": bool(identical),
    }


def _reversal_invariance_checks() -> dict[str, Any]:
    sequences = [(1, 2, 3), (2, 1, 3), (1, 1, 2), (3, 2, 1, 0), (1, 2, 1)]
    direct_ok = True
    examples = []
    for sequence in sequences:
        examples.append(
            {
                "sequence": list(sequence),
                "canonical_forward": list(_canonical_bond(sequence)),
                "canonical_reversed": list(_canonical_bond(sequence[::-1])),
            }
        )
        if _canonical_bond(sequence) != _canonical_bond(sequence[::-1]):
            direct_ok = False

    torch.manual_seed(0)
    model = build_ordered(0)
    model.eval()
    L, B = PATH_MAX_LEN, BOND_CATEGORIES
    counts = torch.zeros(1, L, B)
    counts[0, 0, 1] = 0.5
    counts[0, 1, 2] = 1.0
    counts[0, L - 1, 1] = 0.5
    reversed_counts = torch.flip(counts, dims=[1])
    data_a, data_b = Data(), Data()
    data_a.pair_path_counts = counts
    data_b.pair_path_counts = reversed_counts
    with torch.no_grad():
        enc_a = model._compute_path_encoding(data_a)
        enc_b = model._compute_path_encoding(data_b)
    return {
        "canonical_collapses_reversal": bool(direct_ok),
        "examples": examples,
        "position_flip_changes_encoding": bool(
            float((enc_a - enc_b).abs().max()) > 0.0
        ),
    }


def sanity() -> dict[str, Any]:
    device = torch.device("cpu")
    shead._configure_determinism()
    recurrent = rc.build_recurrent(0).to(device).eval()
    ordered = build_ordered(0).to(device).eval()
    orderless = build_orderless(0).to(device).eval()

    param_ordered = _n_params(ordered)
    param_orderless = _n_params(orderless)
    param_recurrent = _n_params(recurrent)

    train_data, valid_data, _ = load_augmented_encoded()
    batch = next(iter(zpp._make_loader(list(valid_data)[:128], 128, False, 0))).to(device)
    batch.pair_path_counts = batch.pair_path_counts.to(device)
    batch.pair_path_counts_sorted = batch.pair_path_counts_sorted.to(device)

    # (1) path branch off degenerates to the recurrent function exactly.
    ordered.force_zero_path = True
    with torch.no_grad():
        r_ordered_off = ordered.encode(batch)
    rec_shared = rc.build_recurrent(0).to(device).eval()
    with torch.no_grad():
        r_recurrent = rec_shared.encode(batch)
    ordered.force_zero_path = False
    with torch.no_grad():
        r_ordered_on = ordered.encode(batch)
    path_changes_readout = float((r_ordered_on - r_ordered_off).abs().max()) > 0.0

    # (2) baseline pair-encoder columns preserved exactly.
    recurrent_width = int(rec_shared.pair_encoder.layers[0].weight.shape[1])
    with torch.no_grad():
        restored = ordered.pair_encoder.layers[0].weight[:, :recurrent_width].clone()
        baseline_weight = rec_shared.pair_encoder.layers[0].weight
        shared_columns_ok = bool(float((restored - baseline_weight).abs().max()) == 0.0)

    # (3) reversal / multiple-path checks + shared init match.
    reversal = _reversal_invariance_checks()
    state_match = _recurrent_state_matches(ordered, rec_shared)

    # (4) forward/backward finiteness.
    ordered.train()
    out = ordered(batch).view(-1)
    loss = out.mean()
    loss.backward()
    grad_finite = all(
        torch.isfinite(p.grad).all().item()
        for p in ordered.parameters()
        if p.grad is not None
    )
    path_grad = float(ordered.path_encoder[0].weight.grad.abs().max())
    nan_any = bool(torch.isnan(out).any() or torch.isinf(out).any())

    # (5) multiple shortest paths: permutation invariance of the aggregation.
    L, B = PATH_MAX_LEN, BOND_CATEGORIES
    c1 = torch.zeros(1, L, B)
    c1[0, 0, 1] = 0.5
    c1[0, 0, 2] = 0.5
    c2 = torch.zeros(1, L, B)
    c2[0, 0, 2] = 0.5
    c2[0, 0, 1] = 0.5
    d1, d2 = Data(), Data()
    d1.pair_path_counts = c1
    d2.pair_path_counts = c2
    with torch.no_grad():
        e1 = ordered._compute_path_encoding(d1)
        e2 = ordered._compute_path_encoding(d2)
    aggregation_permutation_invariant = float((e1 - e2).abs().max()) == 0.0

    # (6) counts align row-for-row with pair_index.
    counts_align = True
    for index in range(min(10, len(valid_data))):
        if int(valid_data[index].pair_path_counts.shape[0]) != int(
            valid_data[index].pair_index.shape[1]
        ):
            counts_align = False

    checks = {
        "path_off_equals_recurrent": round(
            float((r_ordered_off - r_recurrent).abs().max()), 10
        )
        == 0.0,
        "path_on_changes_readout": bool(path_changes_readout),
        "baseline_pair_encoder_columns_preserved": bool(shared_columns_ok),
        "parameters_ordered": int(param_ordered),
        "parameters_orderless": int(param_orderless),
        "parameters_recurrent": int(param_recurrent),
        "ordered_orderless_param_matched": int(param_ordered) == int(param_orderless),
        "added_parameters": int(param_ordered - param_recurrent),
        "shared_state_match": state_match,
        "reversal_invariance": reversal,
        "aggregation_permutation_invariant": bool(aggregation_permutation_invariant),
        "counts_align_pair_index": bool(counts_align),
        "forward_finite": bool(not nan_any),
        "backward_grads_finite": bool(grad_finite),
        "path_encoder_grad_max": float(path_grad),
        "R_dim": int(r_ordered_on.shape[1]),
        "official_test_loaded": False,
    }
    all_ok = (
        checks["path_off_equals_recurrent"]
        and checks["path_on_changes_readout"]
        and checks["baseline_pair_encoder_columns_preserved"]
        and checks["ordered_orderless_param_matched"]
        and checks["reversal_invariance"]["canonical_collapses_reversal"]
        and checks["aggregation_permutation_invariant"]
        and checks["counts_align_pair_index"]
        and checks["forward_finite"]
        and checks["backward_grads_finite"]
    )
    checks["all_checks_passed"] = bool(all_ok)
    _write_json(RESULTS_DIR / "sanity_ordered_path.json", checks)
    print(json.dumps(checks, indent=2, default=str))
    return checks


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def decision_phase_c(seed: int = 0) -> dict[str, Any]:
    ordered = _read_json(RESULTS_DIR / f"ordered_seed{seed}.json")
    candidate = float(ordered["best_valid_mae"])
    delta = REF_RECURRENT_SEED0 - candidate
    if delta >= GATE_STRONG:
        verdict = "ORDERED_PATH_POSITIVE"
        next_step = "run_orderless_control"
    elif delta >= GATE_WEAK:
        verdict = "ORDERED_PATH_WEAK"
        next_step = "run_orderless_control"
    elif delta > -NO_SIGNAL_BAND:
        verdict = "NO_SIGNAL"
        next_step = "stop_no_path_sweep"
    else:
        verdict = "DEGRADED"
        next_step = "stop_no_path_sweep"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "reference_recurrent_valid_mae": float(REF_RECURRENT_SEED0),
        "ordered_valid_mae": candidate,
        "delta_recurrent_minus_ordered": float(delta),
        "strong_gate": GATE_STRONG,
        "weak_gate": GATE_WEAK,
        "ordered_best_epoch": int(ordered["best_epoch"]),
        "ordered_params": int(ordered["parameters"]),
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_phase_c_seed{seed}.json", payload)
    return payload


def decision_phase_d(seed: int = 0) -> dict[str, Any]:
    ordered = _read_json(RESULTS_DIR / f"ordered_seed{seed}.json")
    orderless = _read_json(RESULTS_DIR / f"orderless_seed{seed}.json")
    o = float(ordered["best_valid_mae"])
    c = float(orderless["best_valid_mae"])
    baseline = float(REF_RECURRENT_SEED0)
    delta_order = baseline - o
    delta_orderless = baseline - c
    delta_ordered_vs_control = c - o  # positive => ordered better than orderless
    if delta_ordered_vs_control >= 0.002:
        verdict = "PATH_ORDER_HAS_INDEPENDENT_VALUE"
        next_step = "ordered_seed1_replication"
    elif delta_ordered_vs_control >= 0.001:
        verdict = "PATH_ORDER_WEAK_SIGNAL"
        next_step = "ordered_seed1_replication_cautious"
    elif delta_ordered_vs_control > -0.001:
        verdict = "RICHER_PATH_BUT_NOT_ORDER"
        next_step = "stop_no_atom_sequence"
    else:
        verdict = "ORDERLESS_BETTER_ORDER_HYPOTHESIS_FALSIFIED"
        next_step = "stop_no_more_sequence_models"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "reference_recurrent_valid_mae": baseline,
        "ordered_valid_mae": o,
        "orderless_valid_mae": c,
        "delta_recurrent_minus_ordered": float(delta_order),
        "delta_recurrent_minus_orderless": float(delta_orderless),
        "delta_orderless_minus_ordered": float(delta_ordered_vs_control),
        "ordered_best_epoch": int(ordered["best_epoch"]),
        "orderless_best_epoch": int(orderless["best_epoch"]),
        "ordered_params": int(ordered["parameters"]),
        "orderless_params": int(orderless["parameters"]),
        "verdict": verdict,
        "next_step": next_step,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"decision_phase_d_seed{seed}.json", payload)
    return payload


def final_decision(seed: int = 0) -> dict[str, Any]:
    """Tie Phase A audit + Phase B implementation + Phase C gate into one record."""
    audit_valid = _read_json(RESULTS_DIR / "path_audit_valid.json")
    audit_train = _read_json(RESULTS_DIR / "path_audit_train.json")
    phase_c = _read_json(RESULTS_DIR / f"decision_phase_c_seed{seed}.json")
    sanity_checks = _read_json(RESULTS_DIR / "sanity_ordered_path.json")
    ordered = _read_json(RESULTS_DIR / f"ordered_seed{seed}.json")
    recurrent = _read_json(rc.RESULTS_DIR / f"recurrent_seed{seed}.json")

    epoch_seconds_recurrent = float(recurrent["wall_clock_s"]) / int(recurrent["epochs_run"])
    epoch_seconds_ordered = float(ordered["wall_clock_s"]) / int(ordered["epochs_run"])

    if phase_c["verdict"] in {"ORDERED_PATH_POSITIVE", "ORDERED_PATH_WEAK"}:
        hypothesis = "path order signal present -> Phase D matched control authorised"
        final = "PATH_ORDER_SIGNAL_PRESENT"
    elif phase_c["verdict"] == "NO_SIGNAL":
        hypothesis = "no independent path-order signal at the seed0 gate"
        final = "NO_PATH_SIGNAL"
    else:
        hypothesis = "ordered bond-sequence encoder degraded the frozen recurrent model"
        final = "NO_PATH_SIGNAL_DEGRADED"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "phase_a": {
            "valid": {
                "n_pairs": int(audit_valid["n_pairs"]),
                "fraction_multiple_shortest_paths": float(
                    audit_valid["fraction_multiple_shortest_paths"]
                ),
                "bond_collision_fraction_overall": float(
                    audit_valid["bond_collision_fraction_overall"]
                ),
                "atom_collision_fraction_overall": float(
                    audit_valid["atom_collision_fraction_overall"]
                ),
                "by_path_length": audit_valid["by_path_length"],
            },
            "train": {
                "n_pairs": int(audit_train["n_pairs"]),
                "fraction_multiple_shortest_paths": float(
                    audit_train["fraction_multiple_shortest_paths"]
                ),
                "bond_collision_fraction_overall": float(
                    audit_train["bond_collision_fraction_overall"]
                ),
                "atom_collision_fraction_overall": float(
                    audit_train["atom_collision_fraction_overall"]
                ),
            },
        },
        "phase_b": {
            "implementation": (
                "position-aware shared bond embedding + learned position embedding -> "
                "2-layer MLP -> PATH_DIM; concatenated to the pair-encoder input"
            ),
            "path_max_len": PATH_MAX_LEN,
            "path_dim": PATH_DIM,
            "reversal_invariance": "canonical min(seq, reverse(seq)) per shortest path",
            "multiple_shortest_paths": "mean over canonicalised shortest paths",
            "parameters_recurrent": int(recurrent["parameters"]),
            "parameters_ordered": int(ordered["parameters"]),
            "added_parameters": int(ordered["parameters"]) - int(recurrent["parameters"]),
            "sanity_all_checks_passed": bool(sanity_checks["all_checks_passed"]),
        },
        "phase_c": phase_c,
        "compute": {
            "recurrent_epoch_seconds": float(epoch_seconds_recurrent),
            "ordered_epoch_seconds": float(epoch_seconds_ordered),
            "epoch_ratio_ordered_over_recurrent": float(
                epoch_seconds_ordered / epoch_seconds_recurrent
            ),
        },
        "phase_d": {
            "triggered": False,
            "reason": "Phase C verdict is not a positive/weak signal",
        },
        "final_judgement": final,
        "summary": hypothesis,
        "stopped": [
            "no orderless matched control",
            "no seed1 replication",
            "no atom-sequence addition",
            "no path-architecture sweep",
        ],
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_decision.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "audit_train",
            "sanity",
            "data",
            "ordered_seed0",
            "orderless_seed0",
            "decision_c",
            "decision_d",
            "final_decision",
        ],
    )
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args(argv)

    if args.stage == "audit":
        audit_path_information(split="valid")
    elif args.stage == "audit_train":
        audit_path_information(split="train")
    elif args.stage == "sanity":
        sanity()
    elif args.stage == "data":
        load_augmented_encoded(force=args.force_data)
        print("augmented data ready")
    elif args.stage == "ordered_seed0":
        run_ordered(0, force_data=args.force_data)
    elif args.stage == "orderless_seed0":
        run_orderless(0, force_data=args.force_data)
    elif args.stage == "decision_c":
        print(json.dumps(decision_phase_c(0), indent=2, default=str))
    elif args.stage == "decision_d":
        print(json.dumps(decision_phase_d(0), indent=2, default=str))
    elif args.stage == "final_decision":
        print(json.dumps(final_decision(0), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
