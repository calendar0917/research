"""
Node-level KSVD coefficients fused into GIN (paper-optimistic protocol).

Variants:
  gin_only          — original node features
  gin_node_B0       — concat per-node |coef| from 1-hop patch + shared D
  gin_node_RW       — concat per-node |coef| from RW patch + shared D
  gin_graph_bcast   — old (bad) graph-level s broadcast (control)

Protocol: paper-optimistic-gin-struct-v0 (10-fold, epoch @ max held-out Acc).
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.node_struct import (  # noqa: E402
    NodeStructConfig,
    encode_dataset_nodes,
    learn_shared_D,
    zscore_node_features,
)
from code.pipeline import (  # noqa: E402
    PipelineConfig,
    embed_dataset_shared_dict,
    graph_from_edge_index,
)


PROTOCOL = {
    "protocol_id": "paper-optimistic-gin-nodestruct-v0",
    "role": "self_check_only",
    "n_folds": 10,
    "fold_seed": 42,
    "epoch_select": "max held-out Acc on fold test (optimistic)",
    "note": "Node-level structure; D from train fold only. NOT strict-v0.",
}


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class GINPaper(nn.Module):
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


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(-1)
        correct += int((pred == data.y.view(-1)).sum())
        total += data.num_graphs
    return correct / max(total, 1)


def train_optimistic(train_data, test_data, in_dim, num_classes, epochs, hidden, lr, dropout, device, seed):
    torch.manual_seed(seed)
    model = GINPaper(in_dim, hidden, num_classes, 4, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = DataLoader(train_data, batch_size=32, shuffle=True)
    te = DataLoader(test_data, batch_size=64, shuffle=False)
    best_acc, best_ep = -1.0, -1
    for ep in range(1, epochs + 1):
        model.train()
        for data in tr:
            data = data.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(data.x, data.edge_index, data.batch), data.y.view(-1))
            loss.backward()
            opt.step()
        acc = accuracy(model, te, device)
        if acc >= best_acc:
            best_acc, best_ep = acc, ep
    return best_acc, best_ep


def load_raw(name: str, root: Path):
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=str(root), name=name)
    graphs = []
    raw = []
    ys = []
    for data in ds:
        n = int(data.num_nodes)
        g = graph_from_edge_index(n, data.edge_index.cpu().numpy())
        graphs.append(g)
        x = data.x.float() if data.x is not None else torch.ones((n, 1))
        yv = int(data.y.view(-1)[0].item())
        ys.append(yv)
        raw.append(Data(x=x, edge_index=data.edge_index, y=torch.tensor([yv]), num_nodes=n))
    y = np.array(ys, dtype=np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    y = np.array([remap[int(v)] for v in y], dtype=np.int64)
    for i, d in enumerate(raw):
        d.y = torch.tensor([int(y[i])])
    return graphs, raw, y, len(uniq)


def fuse_node_struct(raw, structs: list[np.ndarray]) -> list[Data]:
    out = []
    for d, s in zip(raw, structs):
        st = torch.tensor(s, dtype=torch.float32)
        x = torch.cat([d.x, st], dim=-1)
        out.append(Data(x=x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
    return out


def fuse_graph_bcast(raw, graphs, tr_idx, seed=0) -> list[Data]:
    """Old control: graph-level rich readout broadcast."""
    cfg = PipelineConfig(
        method="M0",
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=10,
        num_walks=16,
        edge_decay=0.7,
        seed=seed,
        n_atoms=12,
        T=3,
        T_min=2,
        ksvd_iter=5,
        order_mode="bfs",
        readout_mode="rich",
    )
    X, _ = embed_dataset_shared_dict(graphs, cfg, train_idx=tr_idx, max_train_cols=3000)
    mu, sig = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
    Xs = (X - mu) / sig
    out = []
    for i, d in enumerate(raw):
        s = torch.tensor(Xs[i], dtype=torch.float32).unsqueeze(0).expand(d.num_nodes, -1)
        x = torch.cat([d.x, s], dim=-1)
        out.append(Data(x=x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    root = _TRACK.parents[1] / "data" / "TUD"
    graphs, raw, y, num_classes = load_raw(args.dataset, root)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=PROTOCOL["fold_seed"])

    variants = [
        "gin_only",
        "gin_node_B0",
        "gin_node_RW",
        "gin_graph_bcast",
    ]
    results = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "hps": vars(args),
        "variants": {},
    }

    for variant in variants:
        print(f"=== {variant} ===")
        scores, epochs = [], []
        t0 = time.time()
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            if variant == "gin_only":
                data_all = raw
            elif variant == "gin_graph_bcast":
                data_all = fuse_graph_bcast(raw, graphs, tr, seed=fold)
            else:
                ncfg = NodeStructConfig(
                    mode="B0" if variant == "gin_node_B0" else "RW",
                    max_nodes=8 if variant == "gin_node_B0" else 8,
                    walk_length=6,
                    p=0.5,
                    q=2.0,
                    n_atoms=12,
                    T=3,
                    T_min=2,
                    ksvd_iter=5,
                    seed=fold,
                    feat="abs_coef",  # 12-d per node, not 108
                )
                D, meta = learn_shared_D(graphs, tr, ncfg)
                structs, _ = encode_dataset_nodes(graphs, D, ncfg)
                structs = zscore_node_features(structs, tr)
                data_all = fuse_node_struct(raw, structs)
                if fold == 0:
                    print(f"  struct dim={structs[0].shape[1]} D_recon={meta.get('recon_rel'):.3f}")

            train_data = [data_all[i] for i in tr]
            test_data = [data_all[i] for i in te]
            in_dim = int(train_data[0].x.size(-1))
            acc, ep = train_optimistic(
                train_data,
                test_data,
                in_dim,
                num_classes,
                args.epochs,
                args.hidden,
                args.lr,
                args.dropout,
                device,
                seed=1000 + fold,
            )
            scores.append(acc)
            epochs.append(ep)
            print(f"  fold {fold}: acc={acc:.3f} @ep{ep} in_dim={in_dim}")

        results["variants"][variant] = {
            "acc_mean": float(np.mean(scores)),
            "acc_std": float(np.std(scores)),
            "acc_percent_mean": float(np.mean(scores) * 100),
            "acc_percent_std": float(np.std(scores) * 100),
            "fold_scores": scores,
            "fold_best_epochs": epochs,
            "seconds": time.time() - t0,
        }
        print(
            f"  >> {variant}: {results['variants'][variant]['acc_percent_mean']:.1f} "
            f"± {results['variants'][variant]['acc_percent_std']:.1f} %"
        )

    path = _TRACK / "results" / "node_struct_gin.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "NODE_STRUCT_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# Node-level structure + GIN (paper-optimistic)",
        "",
        f"Dataset: **{r['dataset']}** · `{r['protocol']['protocol_id']}`",
        "",
        "Fusion: for each node v, patch → shared D → **|coef| (n_atoms dim)** → concat with x_v.",
        "",
        "| variant | Acc % | in_dim idea |",
        "|---------|-------|-------------|",
    ]
    ideas = {
        "gin_only": "attr only (~7)",
        "gin_node_B0": "attr + 12 node coef (1-hop)",
        "gin_node_RW": "attr + 12 node coef (RW)",
        "gin_graph_bcast": "attr + ~108 graph s broadcast (old)",
    }
    for k, v in r["variants"].items():
        lines.append(
            f"| {k} | {v['acc_percent_mean']:.1f} ± {v['acc_percent_std']:.1f} | {ideas.get(k, '')} |"
        )
    lines += [
        "",
        "## Expected",
        "",
        "- node_* should beat graph_bcast if per-node structure matters",
        "- node_* vs gin_only: whether structure helps under optimistic GIN",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
