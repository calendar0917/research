"""Standalone compositional path-set model for MolHIV.

Unlike the exact path vocabulary, this model does not allocate one coefficient
per path word.  It composes a path from the original OGB atom and bond fields,
uses the same small encoder for every path, enforces reversal invariance, and
pools path representations by path length.  It has no atom-to-atom message
passing, no GINE backbone, and no KSVD/prototype side channel.

Epoch count is selected on a scaffold-separated split inside each outer-fit
fold.  The outer held fold is evaluated only after refitting for the selected
number of epochs.  Dataset graph items are accessed only for official-train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_exact_motif_path_vocabulary_probe import enumerate_simple_paths

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PathRecord:
    atoms: np.ndarray   # uint8 [P, 5, 9], padded
    bonds: np.ndarray   # uint8 [P, 4, 3], padded
    lengths: np.ndarray # uint8 [P], 1..4


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_record(node_feat: np.ndarray, edge_index: np.ndarray, edge_feat: np.ndarray) -> PathRecord:
    paths, edge_pos = enumerate_simple_paths(edge_index, len(node_feat), 4)
    n_paths = sum(len(paths[length]) for length in range(1, 5))
    atoms = np.zeros((n_paths, 5, node_feat.shape[1]), dtype=np.uint8)
    bonds = np.zeros((n_paths, 4, edge_feat.shape[1]), dtype=np.uint8)
    lengths = np.zeros(n_paths, dtype=np.uint8)
    row = 0
    for length in range(1, 5):
        for nodes in paths[length]:
            atoms[row, : length + 1] = node_feat[np.asarray(nodes, dtype=np.int64)].astype(np.uint8)
            for j in range(length):
                u, v = int(nodes[j]), int(nodes[j + 1])
                key = (u, v) if u < v else (v, u)
                bonds[row, j] = edge_feat[edge_pos[key]].astype(np.uint8)
            lengths[row] = length
            row += 1
    return PathRecord(atoms=atoms, bonds=bonds, lengths=lengths)


def sampled_rows(record: PathRecord, cap_per_length: int, seed: int) -> np.ndarray:
    if cap_per_length <= 0:
        return np.arange(len(record.lengths), dtype=np.int64)
    chosen: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for length in range(1, 5):
        idx = np.flatnonzero(record.lengths == length)
        if len(idx) > cap_per_length:
            idx = np.sort(rng.choice(idx, size=cap_per_length, replace=False))
        chosen.append(idx)
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def make_batch(records: list[PathRecord], graph_rows: np.ndarray, cap: int, sample_seed: int) -> dict[str, torch.Tensor]:
    atoms: list[np.ndarray] = []
    bonds: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    graph_ids: list[np.ndarray] = []
    for local_graph, row in enumerate(graph_rows.tolist()):
        rec = records[int(row)]
        idx = sampled_rows(rec, cap, sample_seed ^ ((int(row) + 1) * 0x9E3779B1 & 0xFFFFFFFF))
        atoms.append(rec.atoms[idx])
        bonds.append(rec.bonds[idx])
        lengths.append(rec.lengths[idx])
        graph_ids.append(np.full(len(idx), local_graph, dtype=np.int64))
    return {
        "atoms": torch.from_numpy(np.concatenate(atoms, axis=0).astype(np.int64, copy=False)),
        "bonds": torch.from_numpy(np.concatenate(bonds, axis=0).astype(np.int64, copy=False)),
        "lengths": torch.from_numpy(np.concatenate(lengths, axis=0).astype(np.int64, copy=False)),
        "graph_ids": torch.from_numpy(np.concatenate(graph_ids, axis=0)),
        "n_graphs": torch.tensor(len(graph_rows), dtype=torch.int64),
    }


class CompositionalPathSet(nn.Module):
    def __init__(self, atom_dims: list[int], bond_dims: list[int], emb_dim: int = 8, path_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.atom_embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in atom_dims])
        self.bond_embeddings = nn.ModuleList([nn.Embedding(d, emb_dim) for d in bond_dims])
        self.length_embedding = nn.Embedding(5, emb_dim)
        self.path_encoder = nn.Sequential(
            nn.Linear(10 * emb_dim, hidden), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden, path_dim), nn.SiLU(),
        )
        graph_in = 4 * (2 * path_dim + 1)
        self.graph_head = nn.Sequential(
            nn.Linear(graph_in, hidden), nn.SiLU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
        )
        atom_rev = []
        bond_rev = []
        for length in range(5):
            ai = list(range(5)); bi = list(range(4))
            ai[: length + 1] = reversed(ai[: length + 1])
            bi[:length] = reversed(bi[:length])
            atom_rev.append(ai); bond_rev.append(bi)
        self.register_buffer("atom_reverse_index", torch.tensor(atom_rev, dtype=torch.long))
        self.register_buffer("bond_reverse_index", torch.tensor(bond_rev, dtype=torch.long))

    def embed_fields(self, values: torch.Tensor, tables: nn.ModuleList) -> torch.Tensor:
        result = tables[0](values[..., 0])
        for field in range(1, len(tables)):
            result = result + tables[field](values[..., field])
        return result

    def encode_oriented(self, atom_h: torch.Tensor, bond_h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        atom_mask = torch.arange(5, device=lengths.device)[None, :] <= lengths[:, None]
        bond_mask = torch.arange(4, device=lengths.device)[None, :] < lengths[:, None]
        atom_h = atom_h * atom_mask[..., None]
        bond_h = bond_h * bond_mask[..., None]
        length_h = self.length_embedding(lengths)
        flat = torch.cat([atom_h.flatten(1), bond_h.flatten(1), length_h], dim=1)
        return self.path_encoder(flat)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        atoms, bonds = batch["atoms"], batch["bonds"]
        lengths, graph_ids = batch["lengths"], batch["graph_ids"]
        n_graphs = int(batch["n_graphs"])
        atom_h = self.embed_fields(atoms, self.atom_embeddings)
        bond_h = self.embed_fields(bonds, self.bond_embeddings)
        forward = self.encode_oriented(atom_h, bond_h, lengths)
        ar = self.atom_reverse_index[lengths]
        br = self.bond_reverse_index[lengths]
        ar = ar[..., None].expand(-1, -1, atom_h.shape[-1])
        br = br[..., None].expand(-1, -1, bond_h.shape[-1])
        reverse = self.encode_oriented(torch.gather(atom_h, 1, ar), torch.gather(bond_h, 1, br), lengths)
        path_h = 0.5 * (forward + reverse)

        slot = graph_ids * 4 + (lengths - 1)
        n_slots = n_graphs * 4
        sums = path_h.new_zeros((n_slots, path_h.shape[1]))
        sums.index_add_(0, slot, path_h)
        counts = path_h.new_zeros(n_slots)
        counts.index_add_(0, slot, torch.ones_like(slot, dtype=path_h.dtype))
        means = sums / counts.clamp_min(1.0)[:, None]
        maxima = path_h.new_full((n_slots, path_h.shape[1]), -torch.inf)
        maxima.scatter_reduce_(0, slot[:, None].expand_as(path_h), path_h, reduce="amax", include_self=True)
        maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
        pooled = torch.cat([means, maxima, torch.log1p(counts)[:, None]], dim=1)
        graph_h = pooled.reshape(n_graphs, -1)
        return self.graph_head(graph_h).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    order = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def predict(model: nn.Module, records: list[PathRecord], indices: np.ndarray, batch_size: int, cap: int, seed: int) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for graph_rows in batches(indices, batch_size, False, seed):
            batch = make_batch(records, graph_rows, cap, seed)
            out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out)


def fit_with_validation(
    records: list[PathRecord], y: np.ndarray,
    train_idx: np.ndarray, valid_idx: np.ndarray,
    atom_dims: list[int], bond_dims: list[int],
    batch_size: int, cap: int, max_epochs: int, seed: int,
) -> tuple[int, list[dict[str, float | int]], int]:
    seed_everything(seed)
    model = CompositionalPathSet(atom_dims, bond_dims)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    pos_weight = torch.tensor(float(np.sum(y[train_idx] == 0) / max(np.sum(y[train_idx] == 1), 1)))
    history: list[dict[str, float | int]] = []
    best_auc, best_epoch = -np.inf, 1
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []
        for graph_rows in batches(train_idx, batch_size, True, seed + epoch):
            batch = make_batch(records, graph_rows, cap, seed + 10000 * epoch)
            target = torch.from_numpy(y[graph_rows].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        pv = predict(model, records, valid_idx, batch_size, cap, seed=0)
        auc = float(roc_auc_score(y[valid_idx], pv))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_auc": auc})
        print(f"    epoch {epoch}: loss={np.mean(losses):.5f} inner_auc={auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
    return best_epoch, history, params


def refit_and_evaluate(
    records: list[PathRecord], y: np.ndarray,
    fit_idx: np.ndarray, held_idx: np.ndarray,
    atom_dims: list[int], bond_dims: list[int],
    batch_size: int, cap: int, epochs: int, seed: int,
) -> tuple[np.ndarray, float, float, int]:
    seed_everything(seed)
    model = CompositionalPathSet(atom_dims, bond_dims)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    pos_weight = torch.tensor(float(np.sum(y[fit_idx] == 0) / max(np.sum(y[fit_idx] == 1), 1)))
    for epoch in range(1, epochs + 1):
        model.train()
        for graph_rows in batches(fit_idx, batch_size, True, seed + epoch):
            batch = make_batch(records, graph_rows, cap, seed + 10000 * epoch)
            target = torch.from_numpy(y[graph_rows].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), target, pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    pfit = predict(model, records, fit_idx, batch_size, cap, seed=0)
    pheld = predict(model, records, held_idx, batch_size, cap, seed=0)
    return pheld, float(roc_auc_score(y[fit_idx], pfit)), float(roc_auc_score(y[held_idx], pheld)), params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--folds", default="0", help="comma-separated outer folds; use 0,1,2 for full probe")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cap-per-length", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/compositional_path_set_fold0_probe_20260729.json")
    args = ap.parse_args()
    requested_folds = [int(x) for x in args.folds.split(",") if x.strip()]
    t0 = time.time(); repo = Path(__file__).resolve().parents[3]
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    torch.set_num_threads(min(8, max(torch.get_num_threads(), 1)))
    ds = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = ds.get_idx_split(); train = np.asarray(split["train"], dtype=np.int64)
    forbidden = np.concatenate([np.asarray(split["valid"], dtype=np.int64), np.asarray(split["test"], dtype=np.int64)])
    y = np.asarray(ds.labels).reshape(-1).astype(np.int64)[train]
    pos = np.full(len(ds), -1, dtype=np.int64); pos[train] = np.arange(len(train))
    with np.load(args.fold_cache, allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["official_train_indices"], dtype=np.int64), train): raise ValueError("train mismatch")
        groups = np.asarray(z["train_scaffold_groups"]).astype(str)
        outer = []
        for fold in range(3):
            outer.append((pos[np.asarray(z[f"fold_{fold}_train_indices"], dtype=np.int64)], pos[np.asarray(z[f"fold_{fold}_valid_indices"], dtype=np.int64)]))
    if np.any(pos[forbidden] >= 0): raise AssertionError("forbidden split contamination")

    records: list[PathRecord] = []
    total_paths = 0
    for row, graph_i in enumerate(train.tolist()):
        graph, _ = ds[int(graph_i)]
        rec = build_record(np.asarray(graph["node_feat"], dtype=np.int64), np.asarray(graph["edge_index"], dtype=np.int64), np.asarray(graph["edge_feat"], dtype=np.int64))
        records.append(rec); total_paths += len(rec.lengths)
        if (row + 1) % 5000 == 0: print(f"built {row+1}/{len(train)} paths={total_paths}", flush=True)

    atom_dims = list(get_atom_feature_dims()); bond_dims = list(get_bond_feature_dims())
    out: dict[str, Any] = {
        "protocol_id": "molhiv-compositional-reversal-invariant-path-set-v1",
        "date": "2026-07-29", "scope": "official-train only",
        "official_valid_evaluations": 0, "official_test_evaluations": 0,
        "isolation": "dataset graph items accessed only for official train",
        "architecture": "shared raw-feature path encoder; reverse average; per-length mean/max pooling; no message passing",
        "config": vars(args), "n_train": int(len(train)), "n_positive": int(y.sum()),
        "total_simple_paths": int(total_paths), "mean_paths_per_graph": float(total_paths / len(train)),
        "folds": [],
    }
    for fold in requested_folds:
        fit, held = outer[fold]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260729 + fold)
        it_pos, iv_pos = next(splitter.split(np.zeros(len(fit)), y[fit], groups[fit]))
        inner_train, inner_valid = fit[it_pos], fit[iv_pos]
        print(f"outer fold {fold}: inner_train={len(inner_train)} inner_valid={len(inner_valid)} outer_held={len(held)}", flush=True)
        selected_epoch, history, params = fit_with_validation(
            records, y, inner_train, inner_valid, atom_dims, bond_dims,
            args.batch_size, args.cap_per_length, args.max_epochs, 20260729 + fold,
        )
        pheld, fit_auc, held_auc, params2 = refit_and_evaluate(
            records, y, fit, held, atom_dims, bond_dims,
            args.batch_size, args.cap_per_length, selected_epoch, 20260729 + fold,
        )
        row = {
            "fold": fold, "selected_epoch": selected_epoch, "inner_history": history,
            "inner_best_auc": float(max(x["valid_auc"] for x in history)),
            "fit_auc": fit_auc, "heldout_auc": held_auc,
            "trainable_parameters": params2,
            "heldout_probabilities": pheld.tolist(), "heldout_probability_sha256": sha(pheld),
            "n_inner_train": int(len(inner_train)), "n_inner_valid": int(len(inner_valid)),
            "n_outer_fit": int(len(fit)), "n_outer_held": int(len(held)),
        }
        out["folds"].append(row)
        print(json.dumps({k: row[k] for k in ["fold", "selected_epoch", "inner_best_auc", "fit_auc", "heldout_auc", "trainable_parameters"]}, indent=2), flush=True)
    if len(out["folds"]) == 3:
        aucs = np.asarray([r["heldout_auc"] for r in out["folds"]])
        out["aggregate"] = {"fold_auc": aucs.tolist(), "mean_auc": float(aucs.mean()), "sample_std_auc": float(aucs.std(ddof=1)), "min_fold_auc": float(aucs.min())}
    out["elapsed_sec"] = time.time() - t0
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": str(output), "folds": [{k:r[k] for k in ["fold","selected_epoch","inner_best_auc","fit_auc","heldout_auc","trainable_parameters"]} for r in out["folds"]], "elapsed_sec": out["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
