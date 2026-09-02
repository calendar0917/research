"""Small scaffold-fold classification pilot for the slot/shared-atom bridge."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TRACK = Path(__file__).resolve().parents[1]
if str(TRACK) not in sys.path:
    sys.path.insert(0, str(TRACK))
from code.data_molhiv import load_molhiv
from code.molhiv_beam8_incidence import build_molhiv_beam8_incidence
from code.run_molhiv_beam8_masked_chemistry_gate import random_bfs_cover_slots
from code.run_molhiv_slot_bridge_reconstruction import graph_adjacency, item_slot_features

BRANCHES = ("NO_BRIDGE", "TRUE_BRIDGE", "SHUFFLED_BRIDGE")


def seed_all(torch: Any, seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)


def stratified_limit(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed); values = np.asarray(indices, dtype=np.int64)
    pos = values[labels[values] > 0.5]; neg = values[labels[values] <= 0.5]
    n_pos = min(len(pos), max(1, int(round(limit * len(pos) / len(values)))))
    n_pos = min(n_pos, limit - 1); n_neg = limit - n_pos
    picked = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(picked); return np.sort(picked).astype(np.int64)


def graph_item(item: Any, graph: Any, atom_dims: Sequence[int], branch: str) -> dict[str, Any]:
    slots = item_slot_features(item, graph, atom_dims)
    nodes = item.slot_nodes; p = len(nodes)
    bridge = np.zeros((p, 8, p * 8), dtype=np.float32)
    for target_patch, target_nodes in enumerate(nodes):
        for target_slot, node in enumerate(target_nodes):
            hits = [source_patch * 8 + source_slot for source_patch, source_nodes in enumerate(nodes) if source_patch != target_patch for source_slot, source_node in enumerate(source_nodes) if int(source_node) == int(node)]
            if hits: bridge[target_patch, target_slot, hits] = 1.0 / len(hits)
    if branch == "SHUFFLED_BRIDGE" and p:
        for patch in range(p): bridge[patch] = np.roll(bridge[patch], 1 + (patch + int(item.graph_index)) % 7, axis=0)
    if branch == "NO_BRIDGE": bridge.fill(0.0)
    return {"graph_index": int(item.graph_index), "slots": slots.astype(np.float32), "bridge": bridge, "label": float(item.label if hasattr(item, "label") else 0.0)}


def collate_factory(torch: Any):
    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_p = max(x["slots"].shape[0] for x in batch); f = batch[0]["slots"].shape[-1]
        slots = np.zeros((len(batch), max_p, 8, f), np.float32); valid = np.zeros((len(batch), max_p), bool); bridge = np.zeros((len(batch), max_p, 8, max_p * 8), np.float32)
        labels = np.asarray([x["label"] for x in batch], np.float32); ids = np.asarray([x["graph_index"] for x in batch], np.int64)
        for row, x in enumerate(batch):
            p = x["slots"].shape[0]; slots[row, :p] = x["slots"]; valid[row, :p] = True; bridge[row, :p, :, :p * 8] = x["bridge"]
        return {"slots": torch.from_numpy(slots), "valid": torch.from_numpy(valid), "bridge": torch.from_numpy(bridge), "labels": torch.from_numpy(labels), "ids": ids}
    return collate


def build_model(nn: Any, torch: Any, feature_dim: int, hidden: int = 64, heads: int = 4) -> Any:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.inp = nn.Linear(feature_dim, hidden); self.slot = nn.Embedding(8, hidden)
            local = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, .1, activation="gelu", batch_first=True, norm_first=True)
            glob = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, .1, activation="gelu", batch_first=True, norm_first=True)
            self.local = nn.TransformerEncoder(local, 1); self.glob = nn.TransformerEncoder(glob, 1); self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)); self.head = nn.Linear(hidden, 1)
        def forward(self, slots: Any, valid: Any, bridge: Any) -> Any:
            b, p, s, _ = slots.shape; ids = torch.arange(8, device=slots.device); x = self.inp(slots) + self.slot(ids)[None, None]
            x = self.local(x.reshape(b * p, s, -1)).reshape(b, p, s, -1); x = x * valid[:, :, None, None]
            pooled = torch.nan_to_num(self.glob(x.mean(2), src_key_padding_mask=~valid)); pooled = pooled * valid[:, :, None]
            global_state = pooled.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            flat = x.reshape(b, p * s, -1)[:, None].expand(-1, p, -1, -1).reshape(b * p, p * s, -1)
            bridge_state = torch.bmm(bridge.reshape(b * p, s, p * s), flat).reshape(b, p, s, -1)
            target = self.fuse(torch.cat([x, bridge_state, global_state[:, None, None].expand(-1, p, s, -1)], -1)).mean(2)
            target = target * valid[:, :, None]; graph_state = target.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            return self.head(graph_state).squeeze(1)
    return Model()


def run_branch(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]], seed: int, epochs: int, batch_size: int, hidden: int) -> dict[str, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    seed_all(torch, seed); collate = collate_factory(torch); model = build_model(nn, torch, train[0]["slots"].shape[-1], hidden)
    loader = DataLoader(list(train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1), collate_fn=collate); opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    pos = sum(x["label"] for x in train); weight = torch.tensor((len(train) - pos) / max(pos, 1.0))
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True); loss = nn.functional.binary_cross_entropy_with_logits(model(batch["slots"], batch["valid"], batch["bridge"]), batch["labels"], pos_weight=weight); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval(); probs = []; labels = []
    with torch.no_grad():
        for batch in DataLoader(list(test), batch_size=batch_size, shuffle=False, collate_fn=collate): probs.extend(torch.sigmoid(model(batch["slots"], batch["valid"], batch["bridge"])).numpy()); labels.extend(batch["labels"].numpy())
    return {"roc_auc": float(roc_auc_score(labels, probs)), "average_precision": float(average_precision_score(labels, probs))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--limit", type=int, default=300); ap.add_argument("--cover", choices=("beam8", "random_bfs"), default="beam8"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--seed", type=int, default=20260817); ap.add_argument("--output", type=Path, default=Path("tracks/ksvd/results/molhiv/molhiv_slot_bridge_classification_pilot_20260817.json")); args = ap.parse_args()
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    bundle = load_molhiv(with_features=True); atom_dims = tuple(int(x) for x in get_atom_feature_dims()); bond_dims = tuple(int(x) for x in get_bond_feature_dims())
    with np.load(args.fold_cache, allow_pickle=False) as folds: train = np.asarray(folds[f"fold_{args.fold}_train_indices"], np.int64); test = np.asarray(folds[f"fold_{args.fold}_valid_indices"], np.int64)
    train = stratified_limit(train, bundle.y, args.limit, args.seed + 1); test = stratified_limit(test, bundle.y, args.limit, args.seed + 2)
    all_items = {branch: {"train": [], "test": []} for branch in BRANCHES}
    for split, indices in (("train", train), ("test", test)):
        for count, index in enumerate(indices, 1):
            graph = bundle.graphs[int(index)]
            if args.cover == "beam8": item = build_molhiv_beam8_incidence(int(index), graph, float(bundle.y[int(index)]), bundle.node_feats[int(index)], bundle.edge_feats[int(index)], atom_feature_dims=atom_dims, bond_feature_dims=bond_dims, seed=args.seed)
            else:
                adjacency = graph_adjacency(graph); slots = random_bfs_cover_slots(adjacency, (adjacency != 0).astype(np.int16), np.zeros(graph.n, np.int64), seed=args.seed + int(index), patch_size=8, overlap=2, edge_capacity_multiplier=1.5); item = SimpleNamespace(graph_index=int(index), atom_features=bundle.node_feats[int(index)], slot_nodes=slots, label=float(bundle.y[int(index)]))
            for branch in BRANCHES: all_items[branch][split].append(graph_item(item, graph, atom_dims, branch))
            if count % 100 == 0: print(f"built {split} {count}/{len(indices)}", flush=True)
    results = {}
    for offset, branch in enumerate(BRANCHES): results[branch] = run_branch(all_items[branch]["train"], all_items[branch]["test"], args.seed + 101 * offset, args.epochs, args.batch_size, args.hidden); print(branch, results[branch], flush=True)
    payload = {"protocol_id": "molhiv-slot-shared-atom-bridge-classification-v1", "scope": "official-train internal Bemis-Murcko fold only", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"labels_used": True, "ksvd_used": False, "gnn_used": False}, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8"); return 0


if __name__ == "__main__": raise SystemExit(main())
