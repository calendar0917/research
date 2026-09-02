"""MolHIV structure-only probe for the Beam8 slot/shared-atom bridge.

Each sample masks one Beam8 patch and predicts its 8-slot internal adjacency
from the remaining patches.  The three branches differ only in the exact
slot-to-slot bridge: TRUE_BRIDGE, SHUFFLED_BRIDGE, and NO_BRIDGE.  This is a
mechanism test; HIV labels, KSVD, GIN and GINE are deliberately unused.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TRACK = Path(__file__).resolve().parents[1]
if str(TRACK) not in sys.path:
    sys.path.insert(0, str(TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_beam8_incidence import build_molhiv_beam8_incidence
from code.run_molhiv_beam8_masked_chemistry_gate import random_bfs_cover_slots

BRANCHES = ("NO_BRIDGE", "TRUE_BRIDGE", "SHUFFLED_BRIDGE")
PAIRS = tuple(zip(*np.triu_indices(8, k=1)))


@dataclass(frozen=True)
class Sample:
    graph_index: int
    context: np.ndarray  # (patches-1, 8, feature_dim)
    valid_context: np.ndarray  # (patches-1,)
    bridge: np.ndarray  # (8, (patches-1)*8)
    target: np.ndarray  # (28,)
    target_mask: np.ndarray  # (28,)


def seed_all(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def graph_adjacency(graph: Any) -> np.ndarray:
    out = np.zeros((graph.n, graph.n), dtype=np.float32)
    for left, right in graph.edges():
        out[left, right] = out[right, left] = 1.0
    return out


def item_slot_features(item: Any, graph: Any, atom_dims: Sequence[int]) -> np.ndarray:
    """Retain atom categories plus local topology for each canonical slot."""
    adjacency = graph_adjacency(graph)
    dims = np.maximum(np.asarray(atom_dims, dtype=np.float32), 1.0)
    rows = []
    for patch, nodes in enumerate(item.slot_nodes):
        values = np.zeros((8, len(atom_dims) + 8 + 2), dtype=np.float32)
        for slot, node in enumerate(nodes):
            atom = np.asarray(item.atom_features[node], dtype=np.float32) / dims
            local_row = np.zeros(8, dtype=np.float32)
            local_row[: len(nodes)] = [adjacency[node, other] for other in nodes]
            degree = np.asarray([local_row.sum() / 7.0], dtype=np.float32)
            center = np.asarray([float(node == item.slot_nodes[patch][0])], dtype=np.float32)
            values[slot] = np.concatenate([atom, local_row, degree, center])
        rows.append(values)
    return np.stack(rows, axis=0)


def make_samples(item: Any, graph: Any, atom_dims: Sequence[int], branch: str, target_indices: Sequence[int] | None = None) -> list[Sample]:
    if branch not in BRANCHES:
        raise ValueError(branch)
    features = item_slot_features(item, graph, atom_dims)
    adjacency = graph_adjacency(graph)
    patches = item.slot_nodes
    result: list[Sample] = []
    selected_targets = range(len(patches)) if target_indices is None else target_indices
    for target_index in selected_targets:
        target_nodes = patches[int(target_index)]
        context_indices = [i for i in range(len(patches)) if i != target_index]
        context = features[context_indices]
        bridge = np.zeros((8, len(context_indices) * 8), dtype=np.float32)
        for target_slot, node in enumerate(target_nodes):
            hits = []
            for context_pos, patch_index in enumerate(context_indices):
                for context_slot, context_node in enumerate(patches[patch_index]):
                    if int(context_node) == int(node):
                        hits.append(context_pos * 8 + context_slot)
            if hits:
                bridge[target_slot, hits] = 1.0 / len(hits)
        if branch == "NO_BRIDGE":
            bridge.fill(0.0)
        elif branch == "SHUFFLED_BRIDGE" and bridge.shape[0] > 1:
            bridge = np.roll(bridge, 1 + (target_index + item.graph_index) % 7, axis=0)
        target = np.zeros(28, dtype=np.float32)
        mask = np.zeros(28, dtype=np.float32)
        for pair_index, (left, right) in enumerate(PAIRS):
            if left < len(target_nodes) and right < len(target_nodes):
                target[pair_index] = adjacency[target_nodes[left], target_nodes[right]]
                mask[pair_index] = 1.0
        result.append(Sample(int(item.graph_index), context, np.ones(len(context), bool), bridge, target, mask))
    return result


class RandomCoverItem:
    def __init__(self, graph_index: int, atom_features: np.ndarray, slot_nodes: Sequence[Sequence[int]]) -> None:
        self.graph_index = int(graph_index)
        self.atom_features = np.asarray(atom_features, dtype=np.int64)
        self.slot_nodes = tuple(tuple(int(x) for x in nodes) for nodes in slot_nodes)
        self.n_patches = len(self.slot_nodes)


def import_torch() -> tuple[Any, Any, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    return torch, nn, DataLoader


def collate_factory(torch: Any):
    def collate(batch: Sequence[Sample]) -> dict[str, Any]:
        max_patches = max(x.context.shape[0] for x in batch)
        feat_dim = batch[0].context.shape[-1]
        context = np.zeros((len(batch), max_patches, 8, feat_dim), np.float32)
        valid = np.zeros((len(batch), max_patches), bool)
        bridge = np.zeros((len(batch), 8, max_patches * 8), np.float32)
        target = np.stack([x.target for x in batch])
        target_mask = np.stack([x.target_mask for x in batch])
        graph_indices = np.asarray([x.graph_index for x in batch], np.int64)
        for row, x in enumerate(batch):
            count = x.context.shape[0]
            context[row, :count] = x.context
            valid[row, :count] = True
            bridge[row, :, : count * 8] = x.bridge
        return {"context": torch.from_numpy(context), "valid": torch.from_numpy(valid),
                "bridge": torch.from_numpy(bridge), "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask),
                "graph_indices": graph_indices}
    return collate


def build_model(nn: Any, torch: Any, feature_dim: int, hidden: int = 64, heads: int = 4) -> Any:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(feature_dim, hidden)
            self.slot = nn.Embedding(8, hidden)
            local = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, 0.1, activation="gelu", batch_first=True, norm_first=True)
            glob = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, 0.1, activation="gelu", batch_first=True, norm_first=True)
            self.local = nn.TransformerEncoder(local, 1)
            self.glob = nn.TransformerEncoder(glob, 1)
            self.mask = nn.Parameter(torch.randn(hidden) * 0.02)
            self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
            self.dec = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, context: Any, valid: Any, bridge: Any) -> Any:
            b, p, s, _ = context.shape
            ids = torch.arange(8, device=context.device)
            x = self.inp(context) + self.slot(ids)[None, None]
            x = self.local(x.reshape(b * p, s, -1)).reshape(b, p, s, -1)
            x = x * valid[:, :, None, None]
            pooled = self.glob(x.mean(2), src_key_padding_mask=~valid)
            pooled = torch.nan_to_num(pooled)
            pooled = pooled * valid[:, :, None]
            global_state = pooled.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            bridge_state = torch.bmm(bridge, x.reshape(b, p * s, -1))
            slot_ids = self.slot(ids)[None].expand(b, -1, -1)
            target = self.fuse(torch.cat([slot_ids + self.mask[None, None], bridge_state, global_state[:, None].expand(-1, 8, -1)], -1))
            rows = []
            for left, right in PAIRS:
                a, c = target[:, left], target[:, right]
                rows.append(self.dec(torch.cat([a, c, a * c], -1)))
            return torch.cat(rows, 1)
    return Model()


def train_eval(train: Sequence[Sample], test: Sequence[Sample], *, seed: int, epochs: int, batch_size: int, hidden: int) -> dict[str, Any]:
    torch, nn, DataLoader = import_torch()
    seed_all(torch, seed)
    collate = collate_factory(torch)
    model = build_model(nn, torch, train[0].context.shape[-1], hidden=hidden)
    loader = DataLoader(list(train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1), collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positive = sum(float((x.target * x.target_mask).sum()) for x in train)
    total = sum(float(x.target_mask.sum()) for x in train)
    pos_weight = torch.tensor((total - positive) / max(positive, 1.0))
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["context"], batch["valid"], batch["bridge"])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["target"], pos_weight=pos_weight, reduction="none")
            loss = (loss * batch["target_mask"]).sum() / batch["target_mask"].sum().clamp_min(1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    model.eval(); predictions = []; targets = []; masks = []; graphs = []
    test_loader = DataLoader(list(test), batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in test_loader:
            predictions.append(torch.sigmoid(model(batch["context"], batch["valid"], batch["bridge"])).numpy())
            targets.append(batch["target"].numpy()); masks.append(batch["target_mask"].numpy()); graphs.extend(batch["graph_indices"].tolist())
    pred, truth, mask = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    per_graph = []
    for graph_index in sorted(set(graphs)):
        sel = np.asarray(graphs) == graph_index
        p, y, m = pred[sel], truth[sel], mask[sel]
        p, y = p[m > 0], y[m > 0]
        if len(y) == 0:
            continue
        binary = p >= 0.5
        tp = int(np.count_nonzero(binary & (y == 1))); fp = int(np.count_nonzero(binary & (y == 0))); fn = int(np.count_nonzero((~binary) & (y == 1)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        per_graph.append((float(np.sqrt(np.mean((p - y) ** 2))), f1))
    return {"rmse": float(np.mean([x[0] for x in per_graph])), "f1": float(np.mean([x[1] for x in per_graph])), "graphs": len(per_graph), "parameters": int(sum(x.numel() for x in model.parameters()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300, help="graphs per internal fold; 0 means all")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--cover", choices=("beam8", "random_bfs"), default="beam8")
    ap.add_argument("--branch", choices=BRANCHES, default=None, help="run one branch only; useful for full-scale streaming runs")
    ap.add_argument("--targets-per-graph", type=int, default=0, help="0 uses every target patch; positive values sample this many targets per graph")
    ap.add_argument("--output", type=Path, default=Path("tracks/ksvd/results/molhiv/molhiv_slot_bridge_reconstruction_pilot_20260817.json"))
    args = ap.parse_args()
    try:
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    except ImportError as exc:
        raise RuntimeError("OGB is required") from exc
    bundle = load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV categorical features were not loaded")
    with np.load(args.fold_cache, allow_pickle=False) as folds:
        train = np.asarray(folds[f"fold_{args.fold}_train_indices"], np.int64)
        test = np.asarray(folds[f"fold_{args.fold}_valid_indices"], np.int64)
    rng = np.random.default_rng(args.seed + args.fold)
    if args.limit > 0:
        train = np.sort(rng.choice(train, size=min(args.limit, len(train)), replace=False))
        test = np.sort(rng.choice(test, size=min(args.limit, len(test)), replace=False))
    atom_dims = tuple(int(x) for x in get_atom_feature_dims()); bond_dims = tuple(int(x) for x in get_bond_feature_dims())
    branches = (args.branch,) if args.branch is not None else BRANCHES
    if args.targets_per_graph < 0:
        raise ValueError("--targets-per-graph must be nonnegative")
    built: dict[str, dict[str, list[Sample]]] = {b: {"train": [], "test": []} for b in branches}
    counts = {"train_graphs": len(train), "test_graphs": len(test), "train_patches": 0, "test_patches": 0}
    for split_name, indices in (("train", train), ("test", test)):
        for count, index in enumerate(indices, 1):
            graph = bundle.graphs[int(index)]
            if args.cover == "beam8":
                item = build_molhiv_beam8_incidence(int(index), graph, float(bundle.y[int(index)]), bundle.node_feats[int(index)], bundle.edge_feats[int(index)], atom_feature_dims=atom_dims, bond_feature_dims=bond_dims, seed=args.seed)
            else:
                adjacency = graph_adjacency(graph)
                slots = random_bfs_cover_slots(adjacency, (adjacency != 0).astype(np.int16), np.zeros(graph.n, dtype=np.int64), seed=args.seed + int(index), patch_size=8, overlap=2, edge_capacity_multiplier=1.5)
                item = RandomCoverItem(int(index), bundle.node_feats[int(index)], slots)
            counts[f"{split_name}_patches"] += item.n_patches
            target_indices = None
            if args.targets_per_graph > 0 and item.n_patches > args.targets_per_graph:
                target_indices = np.sort(np.random.default_rng(args.seed + int(index) + 17).choice(item.n_patches, size=args.targets_per_graph, replace=False)).tolist()
            for branch in branches:
                built[branch][split_name].extend(make_samples(item, graph, atom_dims, branch, target_indices))
            if count % 100 == 0: print(f"built {split_name} {count}/{len(indices)}", flush=True)
    results = {}
    for offset, branch in enumerate(branches):
        results[branch] = train_eval(built[branch]["train"], built[branch]["test"], seed=args.seed + offset * 101, epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden)
        print(branch, json.dumps(results[branch]), flush=True)
    if set(BRANCHES).issubset(results):
        true, no, shuf = results["TRUE_BRIDGE"]["rmse"], results["NO_BRIDGE"]["rmse"], results["SHUFFLED_BRIDGE"]["rmse"]
        decision = {"true_vs_no_reduction": float((no - true) / max(no, 1e-12)), "true_vs_shuffled_reduction": float((shuf - true) / max(shuf, 1e-12)), "pass_2pct": bool(no > true * 1.02 and shuf > true * 1.02)}
    else:
        decision = {"single_branch_run": True, "branches": list(branches)}
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    payload = {"protocol_id": "molhiv-slot-shared-atom-bridge-reconstruction-v1", "scope": "official-train internal Bemis-Murcko fold only", "fold": args.fold, "counts": counts, "config": config | {"atom_dims": atom_dims, "bond_dims": bond_dims, "labels_used": False, "ksvd_used": False, "gnn_used": False}, "results": results, "decision": decision}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
