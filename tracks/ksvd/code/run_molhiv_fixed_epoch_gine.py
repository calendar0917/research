"""Protocol-matched fixed-epoch original-node GINE baseline on MolHIV.

Only the 6,400 official-train graphs in the 8,000-graph development subset are
converted to PyG data.  One Bemis--Murcko scaffold fold is used for fitting and
one for held-out evaluation.  The held-out fold is evaluated exactly once,
after epoch 30; official-valid and official-test are neither encoded nor
evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv


def _sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.fold not in (0, 1, 2) or min(args.epochs, args.batch_size, args.hidden, args.layers) <= 0:
        raise ValueError("invalid configuration")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as exc:
        raise RuntimeError(f"missing GINE dependencies: {exc}") from exc

    t0 = time.time()
    _seed_everything(args.seed, torch)
    device = torch.device(args.device)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("atom/bond features were not loaded")
    labels = np.asarray(bundle.y, dtype=np.float32)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices mismatch")
        if "official_train_indices" in folds.files and not np.array_equal(
            np.asarray(folds["official_train_indices"], dtype=np.int64), official_train
        ):
            raise ValueError("fold cache official_train_indices mismatch")

    official_train_set = set(official_train.tolist())
    if set(fit_indices.tolist()) | set(heldout_indices.tolist()) != official_train_set:
        raise AssertionError("fold does not partition official train")
    if set(fit_indices.tolist()) & set(heldout_indices.tolist()):
        raise AssertionError("fit/heldout overlap")
    if set(fit_indices.tolist()) & set(official_valid.tolist()) or set(fit_indices.tolist()) & set(official_test.tolist()):
        raise AssertionError("fit leaks official valid/test")
    if set(heldout_indices.tolist()) & set(official_valid.tolist()) or set(heldout_indices.tolist()) & set(official_test.tolist()):
        raise AssertionError("heldout leaks official valid/test")

    # Encode only official-train graphs.  The dictionary is keyed by remapped
    # subset index so no official-valid/test PyG object is ever constructed.
    data_by_idx: dict[int, Any] = {}
    for graph_i_raw in official_train:
        graph_i = int(graph_i_raw)
        graph = bundle.graphs[graph_i]
        src: list[int] = []
        dst: list[int] = []
        attrs: list[np.ndarray] = []
        for u, v in sorted(graph.edges()):
            attr = np.asarray(bundle.edge_feats[graph_i][(u, v)], dtype=np.int64)
            src.extend([u, v])
            dst.extend([v, u])
            attrs.extend([attr, attr])
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(np.stack(attrs), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)
        data_by_idx[graph_i] = Data(
            x=torch.tensor(bundle.node_feats[graph_i], dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([labels[graph_i]], dtype=torch.float32),
            graph_index=torch.tensor([graph_i], dtype=torch.long),
        )
    if set(data_by_idx) != official_train_set:
        raise AssertionError("encoded graph set differs from official train")

    def make_loader(indices: np.ndarray, shuffle: bool, phase_offset: int) -> Any:
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(args.seed + 9173 + phase_offset)
        return DataLoader(
            [data_by_idx[int(i)] for i in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=args.num_workers,
        )

    class FixedGINE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.atom_encoder = AtomEncoder(args.hidden)
            self.bond_encoder = BondEncoder(args.hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(args.layers):
                mlp = nn.Sequential(
                    nn.Linear(args.hidden, args.hidden),
                    nn.ReLU(),
                    nn.Linear(args.hidden, args.hidden),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(args.hidden))
            self.jk_gates = nn.Parameter(torch.zeros(max(args.layers - 1, 0)))
            self.head = nn.Linear(args.hidden, 1)

        def forward(self, data: Any) -> Any:
            x = self.atom_encoder(data.x)
            edge_attr = self.bond_encoder(data.edge_attr)
            states = []
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
                x = F.dropout(x, p=args.dropout, training=self.training)
                states.append(global_mean_pool(x, data.batch))
            graph_h = states[-1]
            for gate, earlier_h in zip(self.jk_gates, states[:-1]):
                graph_h = graph_h + torch.tanh(gate) * earlier_h
            return self.head(graph_h).view(-1)

    model = FixedGINE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(fit_indices, True, 0)
    fit_loader = make_loader(fit_indices, False, 1)
    heldout_loader = make_loader(heldout_indices, False, 2)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n_seen += len(y)
        mean_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"fixed_gine epoch={epoch:02d} loss={mean_loss:.5f}", flush=True)

    @torch.no_grad()
    def evaluate(loader: Any) -> dict[str, Any]:
        model.eval()
        ys: list[np.ndarray] = []
        logits_all: list[np.ndarray] = []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            ys.append(batch.y.view(-1).cpu().numpy())
            logits_all.append(logits.cpu().numpy())
        y = np.concatenate(ys).astype(np.float64)
        logits = np.concatenate(logits_all).astype(np.float64)
        score = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        return {
            "auc": float(roc_auc_score(y, score)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "score_sha256": _sha256(score),
        }

    # Exactly one post-training evaluation per fit/held-out split.
    fit_metrics = evaluate(fit_loader)
    heldout_metrics = evaluate(heldout_loader)
    report = {
        "protocol_id": "molhiv-fixed-epoch-original-node-gine-scaffold-v1",
        "date": "2026-07-28",
        "fold": args.fold,
        "seed": args.seed,
        "architecture": {
            "atom_encoder": "OGB AtomEncoder",
            "bond_encoder": "OGB BondEncoder",
            "supervised_original_node_gine_layers": args.layers,
            "mlp_per_layer": "Linear-ReLU-Linear",
            "normalization": "BatchNorm1d after every GINE layer",
            "jk_readout": "zero-initialized gated sum of mean-pooled layer states",
            "graph_readout": "mean",
            "ksvd_features": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "epoch_policy": f"single evaluation after fixed epoch {args.epochs}",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "heldout_evaluations": 1,
        },
        "config": vars(args),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "encoded_graphs": int(len(data_by_idx)),
        "encoded_official_valid_graphs": 0,
        "encoded_official_test_graphs": 0,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "final_jk_gates": torch.tanh(model.jk_gates).detach().cpu().tolist(),
        "fit": fit_metrics,
        "heldout": heldout_metrics,
        "history": history,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "elapsed_sec": time.time() - t0,
        "output": args.output,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "fold": args.fold, "seed": args.seed,
        "fit_auc": fit_metrics["auc"], "heldout_auc": heldout_metrics["auc"],
        "final_jk_gates": report["final_jk_gates"], "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
