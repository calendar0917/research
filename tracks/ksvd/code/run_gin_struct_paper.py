"""
GIN ± structure under GIN-paper *optimistic* protocol (SELF-CHECK).

Protocol (paper-optimistic / Xu-style habit):
  - stratified 10-fold, fold_seed=42 (same spirit as paper/experiments)
  - NO independent val set
  - for each fold: train fixed HPs; **select epoch by max held-out Acc on that fold's test**
  - report mean±std of those selected Acc over folds
  - Do NOT mix with strict-v0 main tables

Structure:
  - shared KSVD dict learned on **outer-train only** (honest w.r.t. folds)
  - graph embedding broadcast-concat to every node as extra input channels
  - variants: gin_only | gin+struct

Usage (tracks/ksvd):
  /path/to/paper/.venv/bin/python -m code.run_gin_struct_paper --dataset MUTAG --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

_TRACK = Path(__file__).resolve().parents[1]
_PAPER = _TRACK.resolve().parents[1] / "paper"
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.graph import Graph  # noqa: E402
from code.pipeline import PipelineConfig, embed_dataset_shared_dict, graph_from_edge_index  # noqa: E402


PROTOCOL = {
    "protocol_id": "paper-optimistic-gin-struct-v0",
    "role": "self_check_only",
    "n_folds": 10,
    "fold_seed": 42,
    "epoch_select": "max held-out Acc on fold test (optimistic; Xu/GIN paper habit)",
    "note": "NOT strict-v0; NOT comparable to Errica nested; structure D from train fold only",
}


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class GINPaper(nn.Module):
    """Layer-wise sum of scores (powerful-gnns style)."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 4, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            self.convs.append(GINConv(_mlp(d_in, hidden), train_eps=False))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.predictors = nn.ModuleList(
            [nn.Linear(in_dim, num_classes)] + [nn.Linear(hidden, num_classes) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, batch):
        reps = [x]
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            reps.append(h)
        score = 0
        for pred, rep in zip(self.predictors, reps):
            logits = pred(global_add_pool(rep, batch))
            logits = F.dropout(logits, p=self.dropout, training=self.training)
            score = score + logits
        return score


def load_tud_pyg(name: str, root: Path):
    from torch_geometric.datasets import TUDataset

    root.mkdir(parents=True, exist_ok=True)
    return TUDataset(root=str(root), name=name)


def graphs_from_dataset(ds) -> list[Graph]:
    gs = []
    for data in ds:
        n = int(data.num_nodes)
        ei = data.edge_index.cpu().numpy()
        gs.append(graph_from_edge_index(n, ei))
    return gs


def labels_from_dataset(ds) -> np.ndarray:
    ys = []
    for data in ds:
        y = data.y
        ys.append(int(y.view(-1)[0].item()) if y.numel() > 1 else int(y.item()))
    y = np.array(ys, dtype=np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(v)] for v in y], dtype=np.int64)


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        correct += int((pred == data.y.view(-1)).sum())
        total += data.num_graphs
    return correct / max(total, 1)


def train_fold_optimistic(
    train_data,
    test_data,
    in_dim: int,
    num_classes: int,
    epochs: int,
    hidden: int,
    lr: float,
    dropout: float,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = GINPaper(in_dim, hidden, num_classes, num_layers=4, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    te_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    best_acc = -1.0
    best_ep = -1
    curve = []
    for ep in range(1, epochs + 1):
        model.train()
        for data in tr_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y.view(-1))
            loss.backward()
            opt.step()
        acc = accuracy(model, te_loader, device)
        curve.append(acc)
        if acc >= best_acc:
            best_acc = acc
            best_ep = ep
    return {"best_acc": best_acc, "best_epoch": best_ep, "curve": curve}


def attach_struct(ds, struct: np.ndarray):
    """Broadcast graph-level struct vector to each node; concat with x."""
    from torch_geometric.data import Data

    out = []
    for i, data in enumerate(ds):
        n = data.num_nodes
        s = torch.tensor(struct[i], dtype=torch.float32)
        s_nodes = s.unsqueeze(0).expand(n, -1)
        if data.x is None:
            x = s_nodes
        else:
            x = torch.cat([data.x.float(), s_nodes], dim=-1)
        y = data.y.view(-1).long()
        if y.numel() > 1:
            y = y[:1]
        # remap happens outside; keep raw then fix
        out.append(
            Data(
                x=x,
                edge_index=data.edge_index,
                y=y,
                num_nodes=n,
            )
        )
    return out


def remap_dataset_y(data_list, remap: dict[int, int]):
    for d in data_list:
        d.y = torch.tensor([remap[int(d.y.view(-1)[0].item())]], dtype=torch.long)
    return data_list


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    root = _TRACK.parents[1] / "data" / "TUD"
    ds = load_tud_pyg(args.dataset, root)
    graphs = graphs_from_dataset(ds)
    y = labels_from_dataset(ds)
    uniq = sorted(set(int(v) for v in y))
    # build list of Data with float x
    from torch_geometric.data import Data

    raw = []
    for i, data in enumerate(ds):
        x = data.x.float() if data.x is not None else torch.ones((data.num_nodes, 1))
        yy = data.y.view(-1)
        yi = int(yy[0].item())
        # remap
        # done via y array
        raw.append(Data(x=x, edge_index=data.edge_index, y=torch.tensor([int(y[i])]), num_nodes=data.num_nodes))

    num_classes = int(y.max()) + 1
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=PROTOCOL["fold_seed"])

    cfg_struct = PipelineConfig(
        method="M0",
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=10,
        num_walks=16,
        edge_decay=0.7,
        seed=0,
        n_atoms=12,
        T=3,
        T_min=2,
        ksvd_iter=5,
        order_mode="bfs",
        readout_mode="rich",
    )

    results = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "hps": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "lr": args.lr,
            "dropout": args.dropout,
            "layers": 4,
        },
        "variants": {},
    }

    for variant in ["gin_only", "gin_plus_struct"]:
        print(f"=== {variant} ===")
        fold_scores = []
        fold_epochs = []
        t0 = time.time()
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if variant == "gin_plus_struct":
                Xstruct, meta = embed_dataset_shared_dict(
                    graphs, cfg_struct, train_idx=tr, max_train_cols=3000
                )
                # z-score struct on train
                mu = Xstruct[tr].mean(axis=0)
                sig = Xstruct[tr].std(axis=0) + 1e-6
                Xs = (Xstruct - mu) / sig
                data_all = attach_struct(raw, Xs)
            else:
                data_all = raw
                meta = {}

            train_data = [data_all[i] for i in tr]
            test_data = [data_all[i] for i in te]
            in_dim = int(train_data[0].x.size(-1))
            out = train_fold_optimistic(
                train_data,
                test_data,
                in_dim=in_dim,
                num_classes=num_classes,
                epochs=args.epochs,
                hidden=args.hidden,
                lr=args.lr,
                dropout=args.dropout,
                device=device,
                seed=1000 + fold,
            )
            fold_scores.append(out["best_acc"])
            fold_epochs.append(out["best_epoch"])
            print(
                f"  fold {fold}: acc={out['best_acc']:.3f} @ep{out['best_epoch']} in_dim={in_dim}"
            )

        results["variants"][variant] = {
            "acc_mean": float(np.mean(fold_scores)),
            "acc_std": float(np.std(fold_scores)),
            "acc_percent_mean": float(np.mean(fold_scores) * 100),
            "acc_percent_std": float(np.std(fold_scores) * 100),
            "fold_scores": fold_scores,
            "fold_best_epochs": fold_epochs,
            "seconds": time.time() - t0,
        }
        print(
            f"  >> {variant}: {results['variants'][variant]['acc_percent_mean']:.1f} "
            f"± {results['variants'][variant]['acc_percent_std']:.1f} %"
        )

    out_path = _TRACK / "results" / "gin_struct_paper.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "GIN_STRUCT_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {out_path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# GIN ± structure (paper-optimistic)",
        "",
        f"Dataset: **{r['dataset']}**",
        "",
        f"Protocol: `{r['protocol']['protocol_id']}` — {r['protocol']['epoch_select']}",
        "",
        "**Role: self-check only. Do not mix with strict-v0.**",
        "",
        f"HPs: `{r['hps']}`",
        "",
        "| variant | Acc % (mean±std) |",
        "|---------|------------------|",
    ]
    for k, v in r["variants"].items():
        lines.append(
            f"| {k} | {v['acc_percent_mean']:.1f} ± {v['acc_percent_std']:.1f} |"
        )
    lines += [
        "",
        "## Structure injection",
        "",
        "- Shared KSVD (M0, rich readout) on **train fold only**",
        "- Graph vector z-scored on train, **broadcast-concat to every node** as extra channels",
        "- GIN-0, 4 layers, layer-wise sum scores",
        "",
        "## vs literature",
        "",
        "- CIN/GIN paper MUTAG numbers use similar optimistic 10-fold habits but **not identical code/HPs**",
        "- Use this table only to see if structure **helps this GIN** under the same protocol",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
