"""A minimal learnable structure--attribute fusion screen for MolHIV.

This runner is intentionally a new route rather than another statistic/XGBoost
ablation.  Every atom in a molecule is a centre of one complete radius-2
induced patch.  The patch has two encoders:

* a topology-only two-layer GIN-like encoder, which sees adjacency and a root /
  shell marker but no atom or bond values;
* a permutation-invariant DeepSets encoder, which sees the rooted atom feature,
  the multiset of the other atom features, and the multiset of bond features,
  but no adjacency.

The conditional model uses the topology token to route the attribute token to
several small experts.  This is the first test of the hypothesis
``P(attribute | topology)``.  It is compared with structure-only,
attribute-only, simple concatenation, and a matched patch-pair shuffle control.

Only official-train scaffold folds are used.  The official validation/test
splits are never loaded by this experiment.  Epochs and architecture are fixed
before the outer fold is scored; outer validation is not used for model
selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.exact_orbit_fusion_screen import (
    BOND_DIM,
    REPO_ROOT,
    STRICT_ATOM_DIM,
    compact_bond_semantics,
    strict_atom_semantics,
)
from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    _fold_indices,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/rooted_conditional_patch_network.yaml"


@dataclass(frozen=True)
class PatchRecord:
    """One molecule represented as a disconnected collection of rooted patches."""

    index: int
    label: float
    struct_x: np.ndarray  # [sum patch nodes, 4]: root + one-hot shell
    attr_x: np.ndarray  # [sum patch nodes, STRICT_ATOM_DIM]
    root_mask: np.ndarray  # [sum patch nodes]
    node_patch: np.ndarray  # [sum patch nodes]
    edge_index: np.ndarray  # [2, 2 * sum patch edges]
    edge_attr: np.ndarray  # [sum patch edges, BOND_DIM], undirected once
    edge_patch: np.ndarray  # [sum patch edges]
    patch_count: int
    patch_shuffle: np.ndarray  # destination patch -> source patch
    patch_shuffle_matched: np.ndarray  # same, restricted to equal patch sizes


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:  # pragma: no cover - torch is required by the runner
        pass


def _ego_distances(graph, center: int, radius: int = 2) -> dict[int, int]:
    distances = {int(center): 0}
    frontier = [int(center)]
    for depth in range(1, int(radius) + 1):
        nxt: list[int] = []
        for node in frontier:
            for neighbor in sorted(graph.neighbors(node)):
                if int(neighbor) not in distances:
                    distances[int(neighbor)] = depth
                    nxt.append(int(neighbor))
        frontier = nxt
    return distances


def _patch_record(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    *,
    index: int,
    label: float,
    radius: int = 2,
    shuffle_seed: int = 0,
) -> PatchRecord:
    """Build a complete, order-free radius-r patch collection."""
    atom_semantics, _ = strict_atom_semantics(np.asarray(node_features, dtype=np.int64))
    bond_semantics = {
        graph.edge_key(int(left), int(right)): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }

    struct_rows: list[list[float]] = []
    attr_rows: list[np.ndarray] = []
    roots: list[bool] = []
    node_patch: list[int] = []
    edge_rows: list[list[int]] = []
    edge_attrs: list[np.ndarray] = []
    edge_patch: list[int] = []
    patch_count = 0
    patch_sizes: list[tuple[int, int]] = []

    for center in sorted(graph.nodes):
        distances = _ego_distances(graph, int(center), int(radius))
        nodes = tuple(sorted(distances))
        local = {int(node): offset for offset, node in enumerate(nodes)}
        node_offset = len(struct_rows)
        for node in nodes:
            shell = int(distances[node])
            shell_one_hot = [1.0 if shell == k else 0.0 for k in range(3)]
            struct_rows.append([float(int(node) == int(center)), *shell_one_hot])
            attr_rows.append(atom_semantics[int(node)])
            roots.append(int(node) == int(center))
            node_patch.append(patch_count)

        induced = graph.induced(set(nodes))
        patch_sizes.append((len(nodes), len(induced.edges())))
        for left, right in sorted(induced.edges()):
            left_local = node_offset + local[int(left)]
            right_local = node_offset + local[int(right)]
            edge_rows.extend([[left_local, right_local], [right_local, left_local]])
            edge_attrs.append(bond_semantics[graph.edge_key(int(left), int(right))])
            edge_patch.append(patch_count)
        patch_count += 1

    if patch_count <= 0:
        raise ValueError(f"graph {index} has no rooted patches")
    rng = np.random.default_rng(int(shuffle_seed) + 1000003 * int(index))
    patch_shuffle = rng.permutation(patch_count).astype(np.int64)
    matched_rng = np.random.default_rng(int(shuffle_seed) + 2000003 * int(index))
    patch_shuffle_matched = np.arange(patch_count, dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for patch_id, size_key in enumerate(patch_sizes):
        groups.setdefault(size_key, []).append(int(patch_id))
    for group in groups.values():
        if len(group) > 1:
            source = np.asarray(group, dtype=np.int64)
            patch_shuffle_matched[source] = source[matched_rng.permutation(len(group))]
    # An identity permutation is not a useful matched control for a one-patch
    # graph, but retaining it is the only correct operation in that case.
    return PatchRecord(
        index=int(index),
        label=float(label),
        struct_x=np.asarray(struct_rows, dtype=np.float32),
        attr_x=np.stack(attr_rows, axis=0).astype(np.float32, copy=False),
        root_mask=np.asarray(roots, dtype=np.bool_),
        node_patch=np.asarray(node_patch, dtype=np.int64),
        edge_index=(
            np.asarray(edge_rows, dtype=np.int64).T
            if edge_rows
            else np.empty((2, 0), dtype=np.int64)
        ),
        edge_attr=(
            np.stack(edge_attrs, axis=0).astype(np.float32, copy=False)
            if edge_attrs
            else np.empty((0, BOND_DIM), dtype=np.float32)
        ),
        edge_patch=np.asarray(edge_patch, dtype=np.int64),
        patch_count=int(patch_count),
        patch_shuffle=patch_shuffle,
        patch_shuffle_matched=patch_shuffle_matched,
    )


def build_records(bundle, indices: Sequence[int], *, radius: int, shuffle_seed: int) -> list[PatchRecord]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("the conditional patch network requires OGB features")
    records: list[PatchRecord] = []
    for position, raw_index in enumerate(np.asarray(indices, dtype=np.int64).tolist()):
        i = int(raw_index)
        records.append(
            _patch_record(
                bundle.graphs[i],
                bundle.node_feats[i],
                bundle.edge_feats[i],
                index=i,
                label=float(bundle.y[i]),
                radius=int(radius),
                shuffle_seed=int(shuffle_seed),
            )
        )
        if (position + 1) % 250 == 0 or position + 1 == len(indices):
            print(f"patch records {position + 1}/{len(indices)}", flush=True)
    return records


def _collate(records: Sequence[PatchRecord]):
    """Concatenate disconnected patches while retaining patch/graph segments."""
    import torch

    if not records:
        raise ValueError("cannot collate an empty batch")
    node_struct: list[np.ndarray] = []
    node_attr: list[np.ndarray] = []
    root_mask: list[np.ndarray] = []
    node_patch: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_attrs: list[np.ndarray] = []
    edge_patch: list[np.ndarray] = []
    graph_patch: list[np.ndarray] = []
    patch_shuffle: list[np.ndarray] = []
    patch_shuffle_matched: list[np.ndarray] = []
    node_offset = 0
    patch_offset = 0
    for graph_id, record in enumerate(records):
        node_struct.append(record.struct_x)
        node_attr.append(record.attr_x)
        root_mask.append(record.root_mask)
        node_patch.append(record.node_patch + patch_offset)
        if record.edge_index.shape[1]:
            edge_indices.append(record.edge_index + node_offset)
            edge_attrs.append(record.edge_attr)
            edge_patch.append(record.edge_patch + patch_offset)
        graph_patch.append(np.full(record.patch_count, graph_id, dtype=np.int64))
        patch_shuffle.append(record.patch_shuffle + patch_offset)
        patch_shuffle_matched.append(record.patch_shuffle_matched + patch_offset)
        node_offset += record.struct_x.shape[0]
        patch_offset += record.patch_count

    edge_index = (
        np.concatenate(edge_indices, axis=1)
        if edge_indices
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_attr = (
        np.concatenate(edge_attrs, axis=0)
        if edge_attrs
        else np.empty((0, BOND_DIM), dtype=np.float32)
    )
    edge_patch_array = (
        np.concatenate(edge_patch, axis=0) if edge_patch else np.empty(0, dtype=np.int64)
    )
    return {
        "struct_x": torch.from_numpy(np.concatenate(node_struct, axis=0)),
        "attr_x": torch.from_numpy(np.concatenate(node_attr, axis=0)),
        "root_mask": torch.from_numpy(np.concatenate(root_mask, axis=0)),
        "node_patch": torch.from_numpy(np.concatenate(node_patch, axis=0)),
        "edge_index": torch.from_numpy(edge_index),
        "edge_attr": torch.from_numpy(edge_attr),
        "edge_patch": torch.from_numpy(edge_patch_array),
        "graph_patch": torch.from_numpy(np.concatenate(graph_patch, axis=0)),
        "patch_shuffle": torch.from_numpy(np.concatenate(patch_shuffle, axis=0)),
        "patch_shuffle_matched": torch.from_numpy(np.concatenate(patch_shuffle_matched, axis=0)),
        "n_patches": int(patch_offset),
        "n_graphs": int(len(records)),
        "y": torch.tensor([record.label for record in records], dtype=torch.float32),
        "indices": np.asarray([record.index for record in records], dtype=np.int64),
    }


def _segment_mean(values, index, n_segments: int):
    out = values.new_zeros((int(n_segments), values.shape[1]))
    out.index_add_(0, index, values)
    counts = values.new_zeros((int(n_segments), 1))
    counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return out / counts.clamp_min(1.0)


def _segment_sum(values, index, n_segments: int):
    out = values.new_zeros((int(n_segments), values.shape[1]))
    out.index_add_(0, index, values)
    return out


def _segment_max(values, index, n_segments: int):
    import torch

    out = values.new_full((int(n_segments), values.shape[1]), -torch.inf)
    expanded = index[:, None].expand(-1, values.shape[1])
    out.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, values.new_zeros(out.shape))


def _segment_softmax(values, index, n_segments: int):
    import torch

    maxima = values.new_full((int(n_segments),), -torch.inf)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    exp_values = torch.exp(values - maxima[index])
    denom = values.new_zeros((int(n_segments),))
    denom.index_add_(0, index, exp_values)
    return exp_values / denom[index].clamp_min(1e-12)


class _MLP:
    """Namespace helper; the real module is created lazily after torch import."""


def _make_mlp(nn, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
    import torch.nn as nn

    layers: list[Any] = [nn.Linear(int(in_dim), int(hidden)), nn.ReLU()]
    if float(dropout) > 0:
        layers.append(nn.Dropout(float(dropout)))
    layers.extend([nn.Linear(int(hidden), int(out_dim)), nn.ReLU()])
    return nn.Sequential(*layers)


def make_model(variant: str, *, hidden: int, struct_layers: int, experts: int, dropout: float):
    import torch.nn as nn

    if variant not in {
        "structure",
        "attribute",
        "concat",
        "conditional",
        "conditional_film",
        "cross_attention",
    }:
        raise ValueError(f"unknown model variant {variant}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant = variant
            self.hidden = int(hidden)
            self.struct_in = nn.Linear(4, int(hidden))
            self.struct_layers = nn.ModuleList()
            for _ in range(int(struct_layers)):
                self.struct_layers.append(_make_mlp(nn, hidden, hidden, hidden, dropout))
            self.atom_encoder = _make_mlp(nn, STRICT_ATOM_DIM, hidden, hidden, dropout)
            self.bond_encoder = _make_mlp(nn, BOND_DIM, hidden, hidden, dropout)
            self.attr_encoder = _make_mlp(nn, 5 * hidden + 2, hidden, hidden, dropout)
            if variant == "cross_attention":
                self.attr_node_element = _make_mlp(nn, STRICT_ATOM_DIM + 1, hidden, hidden, dropout)
                self.attr_edge_element = _make_mlp(nn, BOND_DIM, hidden, hidden, dropout)
                self.node_type = nn.Parameter(torch.zeros(hidden))
                self.edge_type = nn.Parameter(torch.zeros(hidden))
                self.cross_query = nn.Linear(hidden, hidden, bias=False)
                self.cross_key = nn.Linear(hidden, hidden, bias=False)
                self.cross_value = nn.Linear(hidden, hidden, bias=False)
                self.cross_out = _make_mlp(nn, 2 * hidden, hidden, hidden, dropout)
                self.cross_norm = nn.LayerNorm(hidden)
            self.struct_out = nn.Linear(hidden, hidden)
            self.attr_out = nn.Linear(hidden, hidden)
            if variant == "concat":
                self.concat = _make_mlp(nn, 2 * hidden, hidden, hidden, dropout)
            if variant == "conditional":
                self.gate = nn.Linear(hidden, int(experts))
                self.experts = nn.ModuleList(
                    [_make_mlp(nn, hidden, hidden, hidden, dropout) for _ in range(int(experts))]
                )
                self.cond_struct = nn.Linear(hidden, hidden)
                self.cond_norm = nn.LayerNorm(hidden)
            if variant == "conditional_film":
                # FiLM is initialized to the attribute-only model.  The
                # structure branch can therefore add a conditional effect
                # without first having to learn an arbitrary expert basis.
                self.film_gamma = nn.Linear(hidden, hidden)
                self.film_beta = nn.Linear(hidden, hidden)
                self.film_struct = nn.Linear(hidden, hidden)
                for module in (self.film_gamma, self.film_beta, self.film_struct):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
            self.attention = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(3 * hidden + 1, hidden),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, 1),
            )

        def encode_structure(self, batch):
            h = torch.relu(self.struct_in(batch["struct_x"]))
            source = batch["edge_index"][0]
            target = batch["edge_index"][1]
            for layer in self.struct_layers:
                aggregate = h.new_zeros(h.shape)
                if source.numel():
                    aggregate.index_add_(0, target, h[source])
                h = layer(h + aggregate)
            roots = batch["root_mask"].bool()
            root_h = h[roots]
            root_patch = batch["node_patch"][roots]
            if root_h.shape[0] != batch["n_patches"]:
                raise RuntimeError("every patch must have exactly one rooted node")
            s = root_h.new_zeros((batch["n_patches"], self.hidden))
            s[root_patch] = root_h
            return torch.tanh(self.struct_out(s))

        def encode_attribute(self, batch):
            atom_h = self.atom_encoder(batch["attr_x"])
            patch = batch["node_patch"]
            n_patches = int(batch["n_patches"])
            roots = batch["root_mask"].bool()
            root_h = atom_h[roots]
            root_patch = patch[roots]
            root = atom_h.new_zeros((n_patches, self.hidden))
            root[root_patch] = root_h
            nonroot = ~roots
            other_sum = _segment_sum(atom_h[nonroot], patch[nonroot], n_patches)
            other_count = atom_h.new_zeros((n_patches, 1))
            other_count.index_add_(0, patch[nonroot], atom_h.new_ones((int(nonroot.sum()), 1)))
            other_mean = other_sum / other_count.clamp_min(1.0)
            other_scaled = other_sum / torch.sqrt(other_count.clamp_min(1.0))

            edge_h = self.bond_encoder(batch["edge_attr"])
            if edge_h.shape[0]:
                edge_sum = _segment_sum(edge_h, batch["edge_patch"], n_patches)
                edge_count = edge_h.new_zeros((n_patches, 1))
                edge_count.index_add_(0, batch["edge_patch"], edge_h.new_ones((edge_h.shape[0], 1)))
                edge_mean = edge_sum / edge_count.clamp_min(1.0)
                edge_scaled = edge_sum / torch.sqrt(edge_count.clamp_min(1.0))
            else:
                edge_mean = atom_h.new_zeros((n_patches, self.hidden))
                edge_scaled = edge_mean.clone()
                edge_count = atom_h.new_zeros((n_patches, 1))
            counts = torch.log1p(torch.cat([other_count, edge_count], dim=1))
            a = self.attr_encoder(torch.cat([root, other_mean, other_scaled, edge_mean, edge_scaled, counts], dim=1))
            return torch.tanh(self.attr_out(a))

        def _attribute_permutation(self, batch, shuffle_mode: str):
            if isinstance(shuffle_mode, bool):
                shuffle_mode = "random" if shuffle_mode else "none"
            if shuffle_mode == "random":
                return batch["patch_shuffle"]
            if shuffle_mode == "size_matched":
                return batch["patch_shuffle_matched"]
            if shuffle_mode in {"none", ""}:
                return None
            raise ValueError(f"unknown shuffle mode {shuffle_mode}")

        def fuse(self, s, a, batch, shuffle_mode: str):
            permutation = self._attribute_permutation(batch, shuffle_mode)
            if permutation is not None:
                a = a[permutation]
            if self.variant == "structure":
                return s
            if self.variant == "attribute":
                return a
            if self.variant == "concat":
                return self.concat(torch.cat([s, a], dim=1))
            if self.variant == "conditional_film":
                gamma = 1.0 + 0.5 * torch.tanh(self.film_gamma(s))
                beta = 0.5 * torch.tanh(self.film_beta(s))
                structural = 0.1 * torch.tanh(self.film_struct(s))
                return a * gamma + beta + structural
            if self.variant == "cross_attention":
                raise RuntimeError("cross_attention uses fuse_elements, not fuse")
            gate = torch.softmax(self.gate(s), dim=1)
            expert_values = torch.stack([expert(a) for expert in self.experts], dim=1)
            chemistry = torch.sum(gate[:, :, None] * expert_values, dim=1)
            return self.cond_norm(torch.tanh(self.cond_struct(s) + chemistry))

        def fuse_elements(self, s, batch, shuffle_mode: str):
            """Query the patch's attribute element set with its topology token.

            ``patch_shuffle`` is represented by changing the segment ID of each
            attribute element.  Thus a destination structural patch attends to
            exactly one source attribute set, even when the two patches have
            different numbers of atoms or bonds.
            """
            node_input = torch.cat([batch["attr_x"], batch["root_mask"].float()[:, None]], dim=1)
            node_elements = self.attr_node_element(node_input) + self.node_type
            edge_elements = self.attr_edge_element(batch["edge_attr"]) + self.edge_type
            elements = torch.cat([node_elements, edge_elements], dim=0)
            element_patch = torch.cat([batch["node_patch"], batch["edge_patch"]], dim=0)
            permutation = self._attribute_permutation(batch, shuffle_mode)
            if permutation is not None:
                inverse = torch.argsort(permutation)
                element_patch = inverse[element_patch]
            query = self.cross_query(s)
            key = self.cross_key(elements)
            value = self.cross_value(elements)
            scores = (query[element_patch] * key).sum(dim=1) / math.sqrt(float(self.hidden))
            weights = _segment_softmax(scores, element_patch, int(batch["n_patches"]))
            context = _segment_sum(value * weights[:, None], element_patch, int(batch["n_patches"]))
            return self.cross_norm(s + self.cross_out(torch.cat([s, context], dim=1)))

        def forward(
            self,
            batch,
            *,
            pair_shuffle: bool = False,
            shuffle_mode: str | None = None,
            return_gate: bool = False,
        ):
            if shuffle_mode is None:
                shuffle_mode = "random" if pair_shuffle else "none"
            s = self.encode_structure(batch)
            if self.variant == "cross_attention":
                z = self.fuse_elements(s, batch, str(shuffle_mode))
            else:
                a = self.encode_attribute(batch)
                z = self.fuse(s, a, batch, str(shuffle_mode))
            patch_graph = batch["graph_patch"]
            n_graphs = int(batch["n_graphs"])
            scores = self.attention(z).view(-1)
            weights = _segment_softmax(scores, patch_graph, n_graphs)
            attended = _segment_sum(z * weights[:, None], patch_graph, n_graphs)
            mean = _segment_mean(z, patch_graph, n_graphs)
            maximum = _segment_max(z, patch_graph, n_graphs)
            patch_counts = z.new_zeros((n_graphs, 1))
            patch_counts.index_add_(0, patch_graph, z.new_ones((z.shape[0], 1)))
            graph_repr = torch.cat([attended, mean, maximum, torch.log1p(patch_counts)], dim=1)
            logits = self.head(graph_repr).view(-1)
            if return_gate and self.variant == "conditional":
                return logits, torch.softmax(self.gate(s), dim=1)
            return logits

    import torch

    return Model()


def _loader(records, rows: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    selected = [records[int(row)] for row in np.asarray(rows, dtype=np.int64)]
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        selected,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=_collate,
    )


def _predict(model, loader, device, *, pair_shuffle: bool = False, shuffle_mode: str | None = None):
    import torch

    model.eval()
    ys: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            y = batch["y"].numpy()
            moved = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(moved, pair_shuffle=pair_shuffle, shuffle_mode=shuffle_mode)
            ys.append(y)
            logits.append(out.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(logits)


def _auc(y: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, logits))


def fit_one(
    records: list[PatchRecord],
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    *,
    variant: str,
    seed: int,
    pair_shuffle: bool = False,
    shuffle_mode: str | None = None,
    hidden: int,
    struct_layers: int,
    experts: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    _seed_all(int(seed))
    device = torch.device(device_name)
    model = make_model(
        variant,
        hidden=int(hidden),
        struct_layers=int(struct_layers),
        experts=int(experts),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    labels = np.asarray([records[int(row)].label for row in train_rows], dtype=np.float32)
    positives = max(float(labels.sum()), 1.0)
    pos_weight = torch.tensor((float(len(labels)) - positives) / positives, dtype=torch.float32, device=device)
    if shuffle_mode is None:
        shuffle_mode = "random" if pair_shuffle else "none"
    pair_shuffle = str(shuffle_mode) != "none"
    train_loader = _loader(records, train_rows, int(batch_size), True, int(seed) + 11)
    valid_loader = _loader(records, valid_rows, int(batch_size) * 2, False, int(seed) + 12)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for raw_batch in train_loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in raw_batch.items()}
            logits = model(batch, shuffle_mode=str(shuffle_mode))
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["y"].shape[0])
            total_graphs += int(batch["y"].shape[0])
        y_train, l_train = _predict(model, train_loader, device, shuffle_mode=str(shuffle_mode))
        y_valid, l_valid = _predict(model, valid_loader, device, shuffle_mode=str(shuffle_mode))
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_graphs, 1),
            "train_auc": _auc(y_train, l_train),
            "valid_auc": _auc(y_valid, l_valid),
        }
        history.append(row)
        if epoch == 1 or epoch == int(epochs) or epoch % 5 == 0:
            print(
                f"  {variant}{'_' + str(shuffle_mode) if str(shuffle_mode) != 'none' else ''} seed={seed} "
                f"epoch={epoch:02d} loss={row['loss']:.5f} valid={row['valid_auc']:.6f}",
                flush=True,
            )
    final = history[-1]
    return {
        "variant": variant,
        "pair_shuffle": bool(pair_shuffle),
        "shuffle_mode": str(shuffle_mode),
        "seed": int(seed),
        "train_auc": float(final["train_auc"]),
        "valid_auc": float(final["valid_auc"]),
        "final_loss": float(final["loss"]),
        "history": history,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def audit_record_invariance(bundle, index: int, *, radius: int, shuffle_seed: int) -> dict[str, Any]:
    """Check the raw patch object under two arbitrary node relabelings."""
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("invariance audit requires features")
    graph = bundle.graphs[int(index)]
    nodes = bundle.node_feats[int(index)]
    edges = bundle.edge_feats[int(index)]
    base = _patch_record(
        graph, nodes, edges, index=int(index), label=float(bundle.y[int(index)]), radius=radius, shuffle_seed=shuffle_seed
    )
    rng = np.random.default_rng(918273 + int(index))
    drifts: list[float] = []
    for _ in range(2):
        permutation = rng.permutation(graph.n)
        changed_graph, changed_nodes, changed_edges = relabel_graph_features(graph, nodes, edges, permutation)
        changed = _patch_record(
            changed_graph,
            changed_nodes,
            changed_edges,
            index=int(index),
            label=float(bundle.y[int(index)]),
            radius=radius,
            shuffle_seed=shuffle_seed,
        )
        # Node/patch order may change after relabeling.  Compare invariant
        # multisets of per-patch shape/attribute summaries, not raw rows.
        def patch_signatures(record: PatchRecord) -> list[bytes]:
            out: list[bytes] = []
            for patch_id in range(record.patch_count):
                nodes_mask = record.node_patch == patch_id
                edges_mask = record.edge_patch == patch_id
                payload = (
                    np.sort(record.struct_x[nodes_mask], axis=0).tobytes(),
                    np.sort(record.attr_x[nodes_mask], axis=0).tobytes(),
                    np.sort(record.edge_attr[edges_mask], axis=0).tobytes(),
                )
                out.append(hashlib.sha256(b"".join(payload)).digest())
            return sorted(out)

        drifts.append(float(patch_signatures(base) != patch_signatures(changed)))
    return {
        "index": int(index),
        "permutations": 2,
        "maximum_multiset_mismatch": float(max(drifts, default=0.0)),
        "pass": bool(max(drifts, default=0.0) == 0.0),
    }


def _read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a mapping")
    return payload


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config = _read_config(config_path)
    data_cfg = config["data"]
    rep_cfg = config["representation"]
    train_cfg = config["training"]
    screen_cfg = config["screen"]
    output_json = Path(config["output_json"])
    output_md = Path(config["output_markdown"])
    if not output_json.is_absolute():
        output_json = (REPO_ROOT / output_json).resolve()
    if not output_md.is_absolute():
        output_md = (REPO_ROOT / output_md).resolve()

    max_train = int(screen_cfg.get("max_train_graphs_per_fold", 1200))
    max_valid = int(screen_cfg.get("max_valid_graphs_per_fold", 600))
    epochs = int(train_cfg.get("epochs", 15))
    if smoke:
        max_train = min(max_train, int(screen_cfg.get("smoke_train_graphs_per_fold", 120)))
        max_valid = min(max_valid, int(screen_cfg.get("smoke_valid_graphs_per_fold", 60)))
        epochs = min(epochs, int(train_cfg.get("smoke_epochs", 4)))

    folds_path = Path(data_cfg["scaffold_folds"])
    if not folds_path.is_absolute():
        folds_path = (REPO_ROOT / folds_path).resolve()
    data_root = Path(data_cfg.get("root", "data/ogb"))
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    bundle = load_molhiv(root=data_root, with_features=True)
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}

    fold_ids = [int(v) for v in screen_cfg.get("folds", [0, 1, 2])]
    variants = list(screen_cfg.get("variants", ["structure", "attribute", "concat", "conditional"]))
    conditional_variants = {"conditional", "conditional_film", "cross_attention"}
    shuffle_controls = [str(v) for v in train_cfg.get("shuffle_controls", ["random"])]
    if any(v not in {"random", "size_matched"} for v in shuffle_controls):
        raise ValueError("shuffle_controls must contain only random or size_matched")
    model_seeds = [int(v) for v in train_cfg.get("seeds", [0])]
    t0 = time.time()
    audit_indices = np.asarray(fold_archive["official_train_indices"][:8], dtype=np.int64)
    audit = [
        audit_record_invariance(
            bundle,
            int(index),
            radius=int(rep_cfg.get("radius", 2)),
            shuffle_seed=int(rep_cfg.get("shuffle_seed", 20260930)),
        )
        for index in audit_indices
    ]
    if not all(bool(row["pass"]) for row in audit):
        raise AssertionError("raw rooted patch object failed permutation audit")

    fold_results: list[dict[str, Any]] = []
    for fold in fold_ids:
        train_idx, valid_idx = _fold_indices(
            fold_archive,
            int(fold),
            bundle.y,
            {
                "max_train_graphs_per_fold": max_train,
                "max_valid_graphs_per_fold": max_valid,
                "seed": int(screen_cfg.get("selection_seed", 20260921)),
            },
        )
        all_idx = np.concatenate([train_idx, valid_idx]).astype(np.int64)
        records = build_records(
            bundle,
            all_idx,
            radius=int(rep_cfg.get("radius", 2)),
            shuffle_seed=int(rep_cfg.get("shuffle_seed", 20260930)),
        )
        n_train = int(train_idx.size)
        train_rows = np.arange(n_train, dtype=np.int64)
        valid_rows = np.arange(n_train, len(records), dtype=np.int64)
        result_rows: list[dict[str, Any]] = []
        for seed in model_seeds:
            for variant in variants:
                result_rows.append(
                    fit_one(
                        records,
                        train_rows,
                        valid_rows,
                        variant=str(variant),
                        pair_shuffle=False,
                        seed=int(seed),
                        hidden=int(train_cfg.get("hidden", 48)),
                        struct_layers=int(train_cfg.get("struct_layers", 2)),
                        experts=int(train_cfg.get("experts", 4)),
                        dropout=float(train_cfg.get("dropout", 0.10)),
                        epochs=epochs,
                        batch_size=int(train_cfg.get("batch_size", 24)),
                        learning_rate=float(train_cfg.get("learning_rate", 0.002)),
                        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
                        device_name=str(train_cfg.get("device", "cpu")),
                    )
                )
            for conditional_variant in sorted(conditional_variants & set(variants)):
                for control in shuffle_controls:
                    result_rows.append(
                        fit_one(
                            records,
                            train_rows,
                            valid_rows,
                            variant=conditional_variant,
                            pair_shuffle=True,
                            shuffle_mode=control,
                            seed=int(seed),
                            hidden=int(train_cfg.get("hidden", 48)),
                            struct_layers=int(train_cfg.get("struct_layers", 2)),
                            experts=int(train_cfg.get("experts", 4)),
                            dropout=float(train_cfg.get("dropout", 0.10)),
                            epochs=epochs,
                            batch_size=int(train_cfg.get("batch_size", 24)),
                            learning_rate=float(train_cfg.get("learning_rate", 0.002)),
                            weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
                            device_name=str(train_cfg.get("device", "cpu")),
                        )
                    )
        fold_results.append(
            {
                "fold": int(fold),
                "train_indices": train_idx,
                "valid_indices": valid_idx,
                "n_train": int(n_train),
                "n_valid": int(valid_idx.size),
                "train_positive": int(bundle.y[train_idx].sum()),
                "valid_positive": int(bundle.y[valid_idx].sum()),
                "patch_count_mean": float(np.mean([record.patch_count for record in records])),
                "patch_count_max": int(max(record.patch_count for record in records)),
                "results": result_rows,
            }
        )

    aggregate: dict[str, dict[str, Any]] = {}
    shuffled_names = [
        f"{variant}_shuffled_{control}"
        for variant in variants
        if variant in conditional_variants
        for control in shuffle_controls
    ]
    for variant in variants + shuffled_names:
        if "_shuffled_" in variant:
            base_variant, control = variant.rsplit("_shuffled_", 1)
        else:
            base_variant, control = variant, "none"
        values: list[float] = []
        for fold in fold_results:
            rows = [
                row["valid_auc"]
                for row in fold["results"]
                if row["variant"] == base_variant and row["shuffle_mode"] == control
            ]
            if rows:
                values.append(float(np.mean(rows)))
        aggregate[variant] = {
            "fold_auc": values,
            "mean_auc": float(np.mean(values)) if values else float("nan"),
            "std_auc": float(np.std(values)) if values else float("nan"),
        }

    def deltas(candidate: str, baseline: str) -> dict[str, Any]:
        values = [
            float(aggregate[candidate]["fold_auc"][i] - aggregate[baseline]["fold_auc"][i])
            for i in range(len(fold_results))
        ]
        return {
            "candidate": candidate,
            "baseline": baseline,
            "fold_deltas": values,
            "mean_delta": float(np.mean(values)),
            "fold_wins": int(np.sum(np.asarray(values) > 0.0)),
        }

    delta_rows: dict[str, Any] = {}
    candidate_conditional = next(
        (name for name in ("cross_attention", "conditional_film", "conditional") if name in aggregate),
        None,
    )
    if candidate_conditional is not None and "concat" in aggregate:
        delta_rows[f"{candidate_conditional}_minus_concat"] = deltas(candidate_conditional, "concat")
    if candidate_conditional is not None:
        for control in shuffle_controls:
            control_name = f"{candidate_conditional}_shuffled_{control}"
            if control_name in aggregate:
                delta_rows[f"{candidate_conditional}_minus_patch_pair_{control}"] = deltas(
                    candidate_conditional, control_name
                )
    if "attribute" in aggregate and "structure" in aggregate:
        delta_rows["attribute_minus_structure"] = deltas("attribute", "structure")

    result = {
        "protocol_id": config.get("protocol_id", "molhiv-rooted-conditional-patch-network-v1"),
        "scope": "official-train only; three internal scaffold folds; official valid/test untouched",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "configuration": config,
        "effective_smoke": bool(smoke),
        "effective_max_train_graphs_per_fold": int(max_train),
        "effective_max_valid_graphs_per_fold": int(max_valid),
        "effective_epochs": int(epochs),
        "audit": audit,
        "folds": fold_results,
        "aggregate": aggregate,
        "deltas": delta_rows,
        "elapsed_sec": float(time.time() - t0),
    }
    _write_json(output_json, result)
    lines = [
        "# MolHIV Rooted Conditional Patch Network",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "This is a learnable patch-token screen, not a K-SVD/XGBoost feature sweep.",
        "",
        "| variant | mean scaffold-fold AUC | fold AUC |",
        "|---|---:|---|",
    ]
    for name, row in aggregate.items():
        lines.append(f"| `{name}` | {row['mean_auc']:.6f} | {', '.join(f'{v:.6f}' for v in row['fold_auc'])} |")
    lines.extend(["", "## Mechanism deltas", ""])
    for name, row in delta_rows.items():
        lines.append(
            f"- `{name}`: mean `{row['mean_delta']:+.6f}`, "
            f"wins `{row['fold_wins']}/{len(fold_results)}`; fold deltas `{row['fold_deltas']}`"
        )
    lines.extend(
        [
            "",
            "Conditional variants fuse topology with a permutation-invariant attribute set; cross-attention uses the topology token as a query. Shuffled controls keep the attribute sets but break the same-patch pairing.",
            "",
            "Official validation and test were not encoded or evaluated.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output_json), "aggregate": aggregate, "deltas": delta_rows, "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
