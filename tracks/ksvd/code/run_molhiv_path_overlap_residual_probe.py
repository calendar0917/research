"""Exact path evidence plus a compact atom--path overlap residual for MolHIV.

The strong exact binary path logistic regression remains the additive base.
A small residual network is allowed to use only the deterministic incidence
relation between atoms and the sampled simple paths that contain them:

    atom -> path -> atom -> path

There is no atom--atom molecular-graph message passing, no GINE backbone, and
no KSVD/prototype channel.  Thus the residual specifically tests whether the
missing signal in the exact path bag is how path occurrences overlap through
shared atoms.  The residual gate starts at zero, so epoch 0 is exactly the
path baseline and can be selected by the inner scaffold split.

All data access and all outer evaluation are restricted to official-train.
The outer held fold is not used for epoch selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_exact_motif_path_vocabulary_probe import (
    enumerate_simple_paths,
    path_signature,
    sparse_from_counters,
)
from code.run_molhiv_stable_exact_motif_probe import digest, feature_bytes

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphPathRecord:
    node_fields: np.ndarray  # uint8 [N, 9]
    path_nodes: np.ndarray   # uint16 [P, 5], padded with 0
    path_bonds: np.ndarray   # uint8 [P, 4, 3], padded with 0
    lengths: np.ndarray      # uint8 [P], number of bonds in 1..4
    path_cols: np.ndarray    # int32 [P], exact global vocabulary identity


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def binary(x: sparse.csr_matrix) -> sparse.csr_matrix:
    out = x.astype(np.float32, copy=True)
    out.data.fill(1.0)
    return out


def class_weights(y: np.ndarray) -> np.ndarray:
    out = np.empty(len(y), dtype=np.float32)
    for cls in (0, 1):
        mask = y == cls
        out[mask] = len(y) / (2.0 * max(int(mask.sum()), 1))
    return out


def build_graph_record_and_counts(
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    path_to_col: dict[tuple[int, bytes], int],
    max_length: int,
) -> tuple[GraphPathRecord, Counter[int]]:
    paths, edge_pos = enumerate_simple_paths(edge_index, len(node_feat), max_length)
    n_paths = sum(len(paths[length]) for length in range(1, max_length + 1))
    path_nodes = np.zeros((n_paths, max_length + 1), dtype=np.uint16)
    path_bonds = np.zeros((n_paths, max_length, edge_feat.shape[1]), dtype=np.uint8)
    lengths = np.zeros(n_paths, dtype=np.uint8)
    path_cols = np.zeros(n_paths, dtype=np.int32)

    atom_labels = [digest(b"A" + feature_bytes(row)) for row in node_feat]
    edge_labels: dict[tuple[int, int], bytes] = {}
    for key, pos in edge_pos.items():
        edge_labels[key] = digest(b"B" + feature_bytes(edge_feat[pos]))

    counts: Counter[int] = Counter()
    row = 0
    for length in range(1, max_length + 1):
        for nodes in paths[length]:
            path_nodes[row, : length + 1] = np.asarray(nodes, dtype=np.uint16)
            for j in range(length):
                u, v = int(nodes[j]), int(nodes[j + 1])
                key = (u, v) if u < v else (v, u)
                path_bonds[row, j] = edge_feat[edge_pos[key]].astype(np.uint8, copy=False)
            signature = path_signature(nodes, atom_labels, edge_labels)
            vocab_key = (length, signature)
            col = path_to_col.get(vocab_key)
            if col is None:
                col = len(path_to_col)
                path_to_col[vocab_key] = col
            counts[col] += 1
            lengths[row] = length
            path_cols[row] = col
            row += 1

    return GraphPathRecord(
        node_fields=node_feat.astype(np.uint8, copy=True),
        path_nodes=path_nodes,
        path_bonds=path_bonds,
        lengths=lengths,
        path_cols=path_cols,
    ), counts


def sampled_rows(record: GraphPathRecord, cap_per_length: int, seed: int) -> np.ndarray:
    if cap_per_length <= 0:
        return np.arange(len(record.lengths), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for length in range(1, 5):
        rows = np.flatnonzero(record.lengths == length)
        if len(rows) > cap_per_length:
            rows = np.sort(rng.choice(rows, size=cap_per_length, replace=False))
        chosen.append(rows)
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def make_batch(
    records: list[GraphPathRecord],
    graph_rows: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    path_evidence_rows: list[np.ndarray],
    cap_per_length: int,
    sample_seed: int,
) -> dict[str, torch.Tensor]:
    node_fields: list[np.ndarray] = []
    node_graph: list[np.ndarray] = []
    path_nodes: list[np.ndarray] = []
    path_bonds: list[np.ndarray] = []
    path_lengths: list[np.ndarray] = []
    path_graph: list[np.ndarray] = []
    path_evidence: list[np.ndarray] = []
    member_atom: list[np.ndarray] = []
    member_path: list[np.ndarray] = []
    node_offset = 0
    path_offset = 0

    for local_graph, global_row in enumerate(graph_rows.tolist()):
        record = records[int(global_row)]
        selected = sampled_rows(
            record,
            cap_per_length,
            sample_seed ^ (((int(global_row) + 1) * 0x9E3779B1) & 0xFFFFFFFF),
        )
        n_nodes = len(record.node_fields)
        n_paths = len(selected)
        nodes = record.path_nodes[selected].astype(np.int64, copy=True)
        lengths = record.lengths[selected].astype(np.int64, copy=False)
        nodes += node_offset

        node_fields.append(record.node_fields)
        node_graph.append(np.full(n_nodes, local_graph, dtype=np.int64))
        path_nodes.append(nodes)
        path_bonds.append(record.path_bonds[selected])
        path_lengths.append(lengths)
        path_graph.append(np.full(n_paths, local_graph, dtype=np.int64))
        evidence = path_evidence_rows[int(global_row)]
        if len(evidence) != len(record.lengths):
            raise ValueError(f"missing path evidence for graph row {global_row}")
        path_evidence.append(evidence[selected].astype(np.float32, copy=False))

        for local_path, length in enumerate(lengths.tolist()):
            member_atom.append(nodes[local_path, : length + 1])
            member_path.append(np.full(length + 1, path_offset + local_path, dtype=np.int64))
        node_offset += n_nodes
        path_offset += n_paths

    return {
        "node_fields": torch.from_numpy(np.concatenate(node_fields).astype(np.int64, copy=False)),
        "node_graph": torch.from_numpy(np.concatenate(node_graph)),
        "path_nodes": torch.from_numpy(np.concatenate(path_nodes)),
        "path_bonds": torch.from_numpy(np.concatenate(path_bonds).astype(np.int64, copy=False)),
        "path_lengths": torch.from_numpy(np.concatenate(path_lengths)),
        "path_graph": torch.from_numpy(np.concatenate(path_graph)),
        "path_evidence": torch.from_numpy(np.concatenate(path_evidence)),
        "member_atom": torch.from_numpy(np.concatenate(member_atom)),
        "member_path": torch.from_numpy(np.concatenate(member_path)),
        "base_logits": torch.from_numpy(base_logits[graph_rows].astype(np.float32, copy=False)),
        "labels": torch.from_numpy(labels[graph_rows].astype(np.float32, copy=False)),
        "n_graphs": torch.tensor(len(graph_rows), dtype=torch.int64),
    }


def segment_mean_max(
    values: torch.Tensor, index: torch.Tensor, n_segments: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sums = values.new_zeros((n_segments, values.shape[1]))
    sums.index_add_(0, index, values)
    counts = values.new_zeros(n_segments)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    means = sums / counts.clamp_min(1.0)[:, None]
    maxima = values.new_full((n_segments, values.shape[1]), -torch.inf)
    maxima.scatter_reduce_(
        0, index[:, None].expand_as(values), values, reduce="amax", include_self=True
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return means, maxima, counts


class PathOverlapResidual(nn.Module):
    def __init__(
        self,
        atom_dims: list[int],
        bond_dims: list[int],
        emb_dim: int = 8,
        hidden_dim: int = 16,
        mlp_hidden: int = 32,
        residual_mode: str = "gated",
        incidence_mode: str = "bipartite",
        path_evidence_mode: str = "none",
    ) -> None:
        super().__init__()
        self.atom_embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in atom_dims])
        self.bond_embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in bond_dims])
        self.length_embedding = nn.Embedding(5, emb_dim)
        self.atom_projection = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.SiLU())
        if path_evidence_mode not in ("none", "coefficient"):
            raise ValueError(f"unknown path evidence mode: {path_evidence_mode}")
        self.path_evidence_mode = path_evidence_mode
        evidence_dim = 2 if path_evidence_mode == "coefficient" else 0
        self.path_encoder = nn.Sequential(
            nn.Linear(10 * emb_dim + evidence_dim, mlp_hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(mlp_hidden, hidden_dim), nn.SiLU(),
        )
        self.atom_update = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, mlp_hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(mlp_hidden, hidden_dim), nn.SiLU(),
        )
        self.path_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, mlp_hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(mlp_hidden, hidden_dim), nn.SiLU(),
        )
        graph_dim = 2 * hidden_dim + 4 * (2 * hidden_dim + 1)
        self.graph_head = nn.Sequential(
            nn.Linear(graph_dim, mlp_hidden), nn.SiLU(), nn.Dropout(0.20),
            nn.Linear(mlp_hidden, 1),
        )
        self.residual_mode = residual_mode
        if incidence_mode not in ("bipartite", "disabled"):
            raise ValueError(f"unknown incidence mode: {incidence_mode}")
        self.incidence_mode = incidence_mode
        if residual_mode == "gated":
            self.residual_scale = nn.Parameter(torch.tensor(0.0))
        elif residual_mode == "direct_zero":
            self.register_parameter("residual_scale", None)
            nn.init.zeros_(self.graph_head[-1].weight)
            nn.init.zeros_(self.graph_head[-1].bias)
        else:
            raise ValueError(f"unknown residual mode: {residual_mode}")

        atom_reverse = []
        bond_reverse = []
        for length in range(5):
            ai = list(range(5)); bi = list(range(4))
            ai[: length + 1] = reversed(ai[: length + 1])
            bi[:length] = reversed(bi[:length])
            atom_reverse.append(ai); bond_reverse.append(bi)
        self.register_buffer("atom_reverse_index", torch.tensor(atom_reverse, dtype=torch.long))
        self.register_buffer("bond_reverse_index", torch.tensor(bond_reverse, dtype=torch.long))

    @staticmethod
    def embed_fields(values: torch.Tensor, tables: nn.ModuleList) -> torch.Tensor:
        result = tables[0](values[..., 0])
        for field in range(1, len(tables)):
            result = result + tables[field](values[..., field])
        return result

    def encode_oriented(
        self, path_atom_emb: torch.Tensor, bond_emb: torch.Tensor,
        lengths: torch.Tensor, evidence_features: torch.Tensor | None,
    ) -> torch.Tensor:
        atom_mask = torch.arange(5, device=lengths.device)[None, :] <= lengths[:, None]
        bond_mask = torch.arange(4, device=lengths.device)[None, :] < lengths[:, None]
        parts = [
            (path_atom_emb * atom_mask[..., None]).flatten(1),
            (bond_emb * bond_mask[..., None]).flatten(1),
            self.length_embedding(lengths),
        ]
        if evidence_features is not None:
            parts.append(evidence_features)
        flat = torch.cat(parts, dim=1)
        return self.path_encoder(flat)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_raw = self.embed_fields(batch["node_fields"], self.atom_embeddings)
        atom_h0 = self.atom_projection(node_raw)
        path_atom_raw = node_raw[batch["path_nodes"]]
        bond_raw = self.embed_fields(batch["path_bonds"], self.bond_embeddings)
        lengths = batch["path_lengths"]
        if self.path_evidence_mode == "coefficient":
            evidence = batch["path_evidence"][:, None]
            evidence_features = torch.cat([evidence, evidence.abs()], dim=1)
        else:
            evidence_features = None

        forward = self.encode_oriented(path_atom_raw, bond_raw, lengths, evidence_features)
        ar = self.atom_reverse_index[lengths][..., None].expand(-1, -1, path_atom_raw.shape[-1])
        br = self.bond_reverse_index[lengths][..., None].expand(-1, -1, bond_raw.shape[-1])
        reverse = self.encode_oriented(
            torch.gather(path_atom_raw, 1, ar), torch.gather(bond_raw, 1, br),
            lengths, evidence_features,
        )
        path_h0 = 0.5 * (forward + reverse)

        if self.incidence_mode == "bipartite":
            # path -> atom: atoms learn which encoded path occurrences contain them.
            atom_path_mean, atom_path_max, atom_path_count = segment_mean_max(
                path_h0[batch["member_path"]], batch["member_atom"], len(atom_h0)
            )
            atom_h = self.atom_update(
                torch.cat(
                    [atom_h0, atom_path_mean, atom_path_max, torch.log1p(atom_path_count)[:, None]],
                    dim=1,
                )
            )

            # atom -> path: each path is updated by the context accumulated at its atoms.
            path_atom_mean, path_atom_max, _ = segment_mean_max(
                atom_h[batch["member_atom"]], batch["member_path"], len(path_h0)
            )
            path_h = self.path_update(torch.cat([path_h0, path_atom_mean, path_atom_max], dim=1))
        else:
            # Capacity-matched control: retain atom/path set encoders and the same
            # update MLPs, but remove all atom--path membership information.
            atom_zeros = torch.zeros_like(atom_h0)
            atom_h = self.atom_update(
                torch.cat([atom_h0, atom_zeros, atom_zeros, atom_h0.new_zeros((len(atom_h0), 1))], dim=1)
            )
            path_zeros = torch.zeros_like(path_h0)
            path_h = self.path_update(torch.cat([path_h0, path_zeros, path_zeros], dim=1))

        atom_mean, atom_max, _ = segment_mean_max(
            atom_h, batch["node_graph"], int(batch["n_graphs"])
        )
        slots = batch["path_graph"] * 4 + (lengths - 1)
        path_mean, path_max, path_count = segment_mean_max(
            path_h, slots, int(batch["n_graphs"]) * 4
        )
        per_length = torch.cat([path_mean, path_max, torch.log1p(path_count)[:, None]], dim=1)
        graph_h = torch.cat(
            [atom_mean, atom_max, per_length.reshape(int(batch["n_graphs"]), -1)], dim=1
        )
        residual = self.graph_head(graph_h).squeeze(1)
        if self.residual_mode == "gated":
            logits = batch["base_logits"] + self.residual_scale * residual
        else:
            logits = batch["base_logits"] + residual
        return logits, residual


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    order = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def predict(
    model: nn.Module,
    records: list[GraphPathRecord],
    indices: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    path_evidence_rows: list[np.ndarray],
    batch_size: int,
    cap: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for rows in batches(indices, batch_size, False, sample_seed):
            batch = make_batch(
                records, rows, base_logits, labels, path_evidence_rows, cap, sample_seed
            )
            logits, residual = model(batch)
            predictions.append(torch.sigmoid(logits).numpy())
            residuals.append(residual.numpy())
    return np.concatenate(predictions), np.concatenate(residuals)


def fit_fixed_path_base(
    x_all: sparse.csr_matrix,
    y: np.ndarray,
    fit: np.ndarray,
    evaluations: list[np.ndarray],
    min_df: int,
    c: float,
) -> tuple[np.ndarray, int, dict[str, Any], np.ndarray]:
    df = np.asarray((x_all[fit] > 0).sum(axis=0), dtype=np.int64).ravel()
    cols = np.flatnonzero(df >= min_df)
    x_fit = binary(x_all[fit][:, cols])
    model = LogisticRegression(
        C=c, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=0
    )
    model.fit(x_fit, y[fit])
    logits = np.zeros(len(y), dtype=np.float32)
    logits[fit] = model.decision_function(x_fit).astype(np.float32)
    eval_auc: dict[str, float] = {}
    for i, rows in enumerate(evaluations):
        xe = binary(x_all[rows][:, cols])
        logits[rows] = model.decision_function(xe).astype(np.float32)
        eval_auc[str(i)] = float(roc_auc_score(y[rows], logits[rows]))
    global_coef = np.zeros(x_all.shape[1], dtype=np.float32)
    global_coef[cols] = model.coef_.reshape(-1).astype(np.float32)
    info = {
        "n_features": int(len(cols)),
        "parameters": int(len(cols) + 1),
        "fit_auc": float(roc_auc_score(y[fit], logits[fit])),
        "evaluation_auc": eval_auc,
        "coefficient_l2": float(np.linalg.norm(model.coef_)),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
    }
    return logits, int(len(cols) + 1), info, global_coef


def fit_path_base_for_residual(
    x_all: sparse.csr_matrix,
    y: np.ndarray,
    fit: np.ndarray,
    evaluations: list[np.ndarray],
    groups: np.ndarray,
    min_df: int,
    c: float,
    training_base: str,
    seed: int,
    records: list[GraphPathRecord],
    path_evidence_mode: str,
) -> tuple[np.ndarray, int, dict[str, Any], list[np.ndarray]]:
    """Fit the inference base and optionally replace fit logits by scaffold OOF logits.

    The inference model remains one logistic regression fitted on all ``fit`` rows.
    OOF logits are used only as training offsets for the neural residual, preventing
    the nearly saturated in-sample path fit from hiding transferable errors.
    """
    logits, parameters, info, full_coef = fit_fixed_path_base(
        x_all, y, fit, evaluations, min_df, c
    )
    if path_evidence_mode not in ("none", "coefficient"):
        raise ValueError(path_evidence_mode)
    path_evidence_rows = [np.empty(0, dtype=np.float32) for _ in range(len(y))]
    needed = np.unique(np.concatenate([fit, *evaluations]))
    for row in needed.tolist():
        if path_evidence_mode == "coefficient":
            path_evidence_rows[row] = full_coef[records[row].path_cols]
        else:
            path_evidence_rows[row] = np.zeros(len(records[row].lengths), dtype=np.float32)
    info["training_base"] = training_base
    info["in_sample_training_auc"] = float(roc_auc_score(y[fit], logits[fit]))
    if training_base == "insample":
        info["residual_training_offset_auc"] = info["in_sample_training_auc"]
        return logits, parameters, info, path_evidence_rows
    if training_base != "scaffold_oof":
        raise ValueError(training_base)

    oof = np.full(len(y), np.nan, dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []
    for oof_fold, (train_pos, valid_pos) in enumerate(
        splitter.split(np.zeros(len(fit)), y[fit], groups[fit])
    ):
        subfit = fit[np.asarray(train_pos, dtype=np.int64)]
        subvalid = fit[np.asarray(valid_pos, dtype=np.int64)]
        sublogits, _, subinfo, subcoef = fit_fixed_path_base(
            x_all, y, subfit, [subvalid], min_df, c
        )
        oof[subvalid] = sublogits[subvalid]
        if path_evidence_mode == "coefficient":
            for row in subvalid.tolist():
                path_evidence_rows[row] = subcoef[records[row].path_cols]
        fold_rows.append({
            "fold": int(oof_fold),
            "n_train": int(len(subfit)),
            "n_valid": int(len(subvalid)),
            "n_features": int(subinfo["n_features"]),
            "valid_auc": float(subinfo["evaluation_auc"]["0"]),
        })
    if np.any(~np.isfinite(oof[fit])):
        raise RuntimeError("incomplete scaffold OOF path logits")
    logits[fit] = oof[fit]
    info["residual_training_offset_auc"] = float(roc_auc_score(y[fit], logits[fit]))
    info["oof_folds"] = fold_rows
    return logits, parameters, info, path_evidence_rows


def make_model(
    atom_dims: list[int], bond_dims: list[int], residual_mode: str,
    incidence_mode: str, path_evidence_mode: str,
) -> PathOverlapResidual:
    return PathOverlapResidual(
        atom_dims, bond_dims, emb_dim=8, hidden_dim=16, mlp_hidden=28,
        residual_mode=residual_mode, incidence_mode=incidence_mode,
        path_evidence_mode=path_evidence_mode,
    )


def make_optimizer(model: PathOverlapResidual) -> torch.optim.Optimizer:
    if model.residual_mode == "gated":
        return torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if n != "residual_scale"], "lr": 0.002, "weight_decay": 2e-3},
                {"params": [model.residual_scale], "lr": 0.02, "weight_decay": 0.0},
            ]
        )
    return torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-3)


def scale_value(model: PathOverlapResidual) -> float | None:
    if model.residual_scale is None:
        return None
    return float(model.residual_scale.detach())


def train_with_inner_validation(
    records: list[GraphPathRecord],
    y: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    base_logits: np.ndarray,
    path_evidence_rows: list[np.ndarray],
    atom_dims: list[int],
    bond_dims: list[int],
    batch_size: int,
    cap: int,
    max_epochs: int,
    seed: int,
    residual_mode: str,
    incidence_mode: str,
    path_evidence_mode: str,
) -> tuple[int, list[dict[str, Any]], int]:
    seed_everything(seed)
    model = make_model(
        atom_dims, bond_dims, residual_mode, incidence_mode, path_evidence_mode
    )
    optimizer = make_optimizer(model)
    weights = class_weights(y[train])
    weight_lookup = np.ones(len(y), dtype=np.float32)
    weight_lookup[train] = weights

    base_valid_prob = 1.0 / (1.0 + np.exp(-base_logits[valid]))
    best_auc = float(roc_auc_score(y[valid], base_valid_prob))
    best_epoch = 0
    history: list[dict[str, Any]] = [{
        "epoch": 0,
        "valid_auc": best_auc,
        "fit_auc": float(roc_auc_score(y[train], base_logits[train])),
        "residual_scale": 0.0,
        "mean_loss": None,
    }]
    sample_seed = seed ^ 0xA5A5A5A5

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for rows in batches(train, batch_size, True, seed + epoch):
            batch = make_batch(
                records, rows, base_logits, y, path_evidence_rows, cap, sample_seed
            )
            logits, _ = model(batch)
            row_weights = torch.from_numpy(weight_lookup[rows])
            loss = (F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none") * row_weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        pfit, _ = predict(
            model, records, train, base_logits, y, path_evidence_rows,
            batch_size, cap, sample_seed,
        )
        pvalid, _ = predict(
            model, records, valid, base_logits, y, path_evidence_rows,
            batch_size, cap, sample_seed,
        )
        fit_auc = float(roc_auc_score(y[train], pfit))
        valid_auc = float(roc_auc_score(y[valid], pvalid))
        row = {
            "epoch": epoch,
            "valid_auc": valid_auc,
            "fit_auc": fit_auc,
            "residual_scale": scale_value(model),
            "mean_loss": float(np.mean(losses)),
        }
        history.append(row)
        scale_text = (
            "direct" if row["residual_scale"] is None
            else f"{row['residual_scale']:+.4f}"
        )
        print(
            f"    epoch={epoch:02d} fit={fit_auc:.6f} valid={valid_auc:.6f} "
            f"scale={scale_text} loss={row['mean_loss']:.5f}",
            flush=True,
        )
        if valid_auc > best_auc + 1e-12:
            best_auc = valid_auc
            best_epoch = epoch
    return best_epoch, history, parameter_count(model)


def refit_and_evaluate(
    records: list[GraphPathRecord],
    y: np.ndarray,
    fit: np.ndarray,
    held: np.ndarray,
    base_logits: np.ndarray,
    path_evidence_rows: list[np.ndarray],
    atom_dims: list[int],
    bond_dims: list[int],
    batch_size: int,
    cap: int,
    epochs: int,
    seed: int,
    residual_mode: str,
    incidence_mode: str,
    path_evidence_mode: str,
) -> tuple[np.ndarray, dict[str, Any], int]:
    seed_everything(seed)
    model = make_model(
        atom_dims, bond_dims, residual_mode, incidence_mode, path_evidence_mode
    )
    sample_seed = seed ^ 0xA5A5A5A5
    if epochs > 0:
        optimizer = make_optimizer(model)
        weight_lookup = np.ones(len(y), dtype=np.float32)
        weight_lookup[fit] = class_weights(y[fit])
        for epoch in range(1, epochs + 1):
            model.train()
            for rows in batches(fit, batch_size, True, seed + epoch):
                batch = make_batch(
                    records, rows, base_logits, y, path_evidence_rows, cap, sample_seed
                )
                logits, _ = model(batch)
                row_weights = torch.from_numpy(weight_lookup[rows])
                loss = (F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none") * row_weights).mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
    pfit, rfit = predict(
        model, records, fit, base_logits, y, path_evidence_rows,
        batch_size, cap, sample_seed,
    )
    pheld, rheld = predict(
        model, records, held, base_logits, y, path_evidence_rows,
        batch_size, cap, sample_seed,
    )
    info = {
        "epochs": int(epochs),
        "fit_auc": float(roc_auc_score(y[fit], pfit)),
        "heldout_auc": float(roc_auc_score(y[held], pheld)),
        "residual_scale": scale_value(model),
        "fit_residual_std": float(np.std(rfit)),
        "heldout_residual_std": float(np.std(rheld)),
    }
    return pheld, info, parameter_count(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--folds", default="1")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--base-c", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cap-per-length", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--training-base", choices=("insample", "scaffold_oof"), default="insample")
    ap.add_argument("--residual-mode", choices=("gated", "direct_zero"), default="gated")
    ap.add_argument("--incidence-mode", choices=("bipartite", "disabled"), default="bipartite")
    ap.add_argument("--path-evidence", choices=("none", "coefficient"), default="none")
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/path_overlap_residual_fold1_probe_20260729.json")
    args = ap.parse_args()
    requested_folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if not requested_folds or any(x not in (0, 1, 2) for x in requested_folds):
        raise ValueError(f"invalid folds: {requested_folds}")

    t0 = time.time()
    repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    torch.set_num_threads(min(8, max(torch.get_num_threads(), 1)))
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split()
    train_graphs = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([
        np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)
    ])
    y = np.asarray(dataset.labels).reshape(-1).astype(np.int64)[train_graphs]
    position = np.full(len(dataset), -1, dtype=np.int64)
    position[train_graphs] = np.arange(len(train_graphs))

    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train_graphs):
            raise ValueError("official train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(3):
            fit = position[np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)]
            held = position[np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)]
            outer.append((fit, held))
    if np.any(position[forbidden] >= 0):
        raise AssertionError("official valid/test contamination")

    path_to_col: dict[tuple[int, bytes], int] = {}
    records: list[GraphPathRecord] = []
    path_rows: list[Counter[int]] = []
    total_paths = 0
    for row, graph_index in enumerate(train_graphs.tolist()):
        graph, _ = dataset[int(graph_index)]
        record, counts = build_graph_record_and_counts(
            np.asarray(graph["node_feat"], dtype=np.int64),
            np.asarray(graph["edge_index"], dtype=np.int64),
            np.asarray(graph["edge_feat"], dtype=np.int64),
            path_to_col,
            max_length=4,
        )
        records.append(record)
        path_rows.append(counts)
        total_paths += len(record.lengths)
        if (row + 1) % 5000 == 0:
            print(
                f"built {row+1}/{len(train_graphs)} paths={total_paths} vocab={len(path_to_col)}",
                flush=True,
            )
    x_all = sparse_from_counters(path_rows, len(path_to_col))
    del path_rows

    atom_dims = list(get_atom_feature_dims())
    bond_dims = list(get_bond_feature_dims())
    out: dict[str, Any] = {
        "protocol_id": "molhiv-exact-path-plus-atom-path-overlap-residual-v1",
        "date": "2026-07-29",
        "scope": "official-train only; inner scaffold epoch selection; outer scaffold evaluation",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "isolation": "dataset graph items accessed only for official train",
        "architecture": "fixed exact path logistic base + zero-gated atom-path incidence residual; no atom-atom GNN, GINE, KSVD, or prototype bank",
        "config": vars(args),
        "n_train": int(len(train_graphs)),
        "n_positive": int(y.sum()),
        "n_unique_paths": int(x_all.shape[1]),
        "total_simple_paths": int(total_paths),
        "mean_paths_per_graph": float(total_paths / len(train_graphs)),
        "folds": [],
    }

    for fold in requested_folds:
        fit, held = outer[fold]
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=20260729 + fold
        )
        inner_train_pos, inner_valid_pos = next(
            splitter.split(np.zeros(len(fit)), y[fit], groups[fit])
        )
        inner_train = fit[np.asarray(inner_train_pos, dtype=np.int64)]
        inner_valid = fit[np.asarray(inner_valid_pos, dtype=np.int64)]
        print(
            f"outer fold {fold}: inner_train={len(inner_train)} inner_valid={len(inner_valid)} "
            f"outer_fit={len(fit)} outer_held={len(held)}",
            flush=True,
        )

        inner_logits, inner_base_params, inner_base, inner_path_evidence = fit_path_base_for_residual(
            x_all, y, inner_train, [inner_valid], groups,
            args.min_df, args.base_c, args.training_base, 20260729 + 1000 * fold,
            records, args.path_evidence,
        )
        print(
            f"  inner path baseline={inner_base['evaluation_auc']['0']:.6f} "
            f"features={inner_base['n_features']}",
            flush=True,
        )
        selected_epoch, history, residual_params = train_with_inner_validation(
            records, y, inner_train, inner_valid, inner_logits, inner_path_evidence,
            atom_dims, bond_dims, args.batch_size, args.cap_per_length,
            args.max_epochs, 20260729 + 100 * fold, args.residual_mode,
            args.incidence_mode, args.path_evidence,
        )

        outer_logits, outer_base_params, outer_base, outer_path_evidence = fit_path_base_for_residual(
            x_all, y, fit, [held], groups,
            args.min_df, args.base_c, args.training_base, 20260729 + 2000 * fold,
            records, args.path_evidence,
        )
        outer_baseline_auc = float(outer_base["evaluation_auc"]["0"])
        pheld, refit, residual_params2 = refit_and_evaluate(
            records, y, fit, held, outer_logits, outer_path_evidence,
            atom_dims, bond_dims, args.batch_size, args.cap_per_length,
            selected_epoch, 20260729 + 100 * fold, args.residual_mode,
            args.incidence_mode, args.path_evidence,
        )
        row = {
            "fold": fold,
            "n_inner_train": int(len(inner_train)),
            "n_inner_valid": int(len(inner_valid)),
            "n_outer_fit": int(len(fit)),
            "n_outer_held": int(len(held)),
            "selected_epoch": int(selected_epoch),
            "inner_baseline_auc": float(inner_base["evaluation_auc"]["0"]),
            "inner_selected_auc": float(max(x["valid_auc"] for x in history)),
            "inner_history": history,
            "outer_baseline_auc": outer_baseline_auc,
            "heldout_auc": float(refit["heldout_auc"]),
            "delta_auc": float(refit["heldout_auc"] - outer_baseline_auc),
            "outer_base": outer_base,
            "refit": refit,
            "additive_path_parameters": int(outer_base_params),
            "residual_parameters": int(residual_params2),
            "total_parameters": int(outer_base_params + residual_params2),
            "heldout_probabilities": pheld.tolist(),
            "heldout_probability_sha256": sha(pheld),
        }
        out["folds"].append(row)
        print(json.dumps({
            k: row[k] for k in [
                "fold", "selected_epoch", "inner_baseline_auc", "inner_selected_auc",
                "outer_baseline_auc", "heldout_auc", "delta_auc", "additive_path_parameters",
                "residual_parameters", "total_parameters",
            ]
        }, indent=2), flush=True)

    if len(out["folds"]) == 3:
        aucs = np.asarray([x["heldout_auc"] for x in out["folds"]])
        bases = np.asarray([x["outer_baseline_auc"] for x in out["folds"]])
        out["aggregate"] = {
            "fold_auc": aucs.tolist(),
            "mean_auc": float(aucs.mean()),
            "sample_std_auc": float(aucs.std(ddof=1)),
            "min_fold_auc": float(aucs.min()),
            "baseline_fold_auc": bases.tolist(),
            "baseline_mean_auc": float(bases.mean()),
            "mean_delta_auc": float((aucs - bases).mean()),
        }
    out["elapsed_sec"] = time.time() - t0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "output": str(output),
        "folds": [{k: r[k] for k in [
            "fold", "selected_epoch", "outer_baseline_auc", "heldout_auc", "delta_auc",
            "total_parameters",
        ]} for r in out["folds"]],
        "elapsed_sec": out["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
