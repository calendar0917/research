"""
GIN ± node-level structure under **Xu et al. GIN paper** reporting protocol.

Paper (arXiv:1810.00826 §7):
  - 10-fold CV
  - Report mean±std of fold (held-out) accuracies
  - **One global epoch** selected: argmax over e of (mean_fold Acc at e)
    (not per-fold max epoch)
  - Adam lr=0.01, decay ×0.5 every 50 epochs
  - Bioinformatics: hidden ∈ {16,32}; we use 32 (fixed for fair relative compare)
  - batch 32, dropout 0.5 (fixed; paper also grids these)
  - 5 layers incl. input ⇒ 4 MP layers; GIN-0

Still optimistic (held-out used for epoch selection) but **matches paper selection rule**.
External literature number GIN MUTAG 89.4±5.6 is **not** same code/HP grid — listed as reference only.

Usage:
  python -m code.run_gin_xu_protocol --dataset MUTAG --epochs 350
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
from code.pipeline import graph_from_edge_index  # noqa: E402

PROTOCOL = {
    "protocol_id": "xu-gin-paper-epoch-v0",
    "source": "Xu et al. How Powerful are GNNs, ICLR 2019 / arXiv:1810.00826 §7",
    "n_folds": 10,
    "fold_seed": 42,
    "epoch_select": "single global epoch = argmax_e mean_fold Acc(e); report mean±std of folds at e*",
    "diff_from_previous": "previous used per-fold max held-out Acc (more optimistic / different)",
    "role": "aligned_self_check_vs_literature_habit",
    "literature_anchor": {
        "GIN_MUTAG": "89.4 ± 5.6 (Xu Table; their full HP grid + code)",
        "note": "anchor only if same protocol family; not same implementation",
    },
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
            from torch_geometric.nn import global_add_pool as gap

            logits = pred(gap(rep, batch))
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


def load_raw(name: str, root: Path):
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=str(root), name=name)
    graphs, raw, ys = [], [], []
    for data in ds:
        n = int(data.num_nodes)
        graphs.append(graph_from_edge_index(n, data.edge_index.cpu().numpy()))
        x = data.x.float() if data.x is not None else torch.ones((n, 1))
        yi = int(data.y.view(-1)[0].item())
        ys.append(yi)
        raw.append(Data(x=x, edge_index=data.edge_index, y=torch.tensor([yi]), num_nodes=n))
    y = np.array(ys, dtype=np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    y = np.array([remap[int(v)] for v in y], dtype=np.int64)
    for i, d in enumerate(raw):
        d.y = torch.tensor([int(y[i])])
    return graphs, raw, y, len(uniq)


def fuse_node(raw, structs):
    out = []
    for d, s in zip(raw, structs):
        x = torch.cat([d.x, torch.tensor(s, dtype=torch.float32)], dim=-1)
        out.append(Data(x=x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
    return out


def run_variant(
    variant: str,
    graphs,
    raw,
    y,
    num_classes: int,
    epochs: int,
    hidden: int,
    batch_size: int,
    dropout: float,
    lr: float,
    device: torch.device,
) -> dict:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=PROTOCOL["fold_seed"])
    folds = list(skf.split(np.zeros(len(y)), y))

    # Precompute node struct per fold (D on train only) if needed
    fold_data = []
    for fold, (tr, te) in enumerate(folds):
        if variant == "gin_only":
            data_all = raw
            in_dim = int(raw[0].x.size(-1))
        else:
            ncfg = NodeStructConfig(
                mode="B0" if variant == "gin_node_B0" else "RW",
                max_nodes=8,
                walk_length=6,
                p=0.5,
                q=2.0,
                n_atoms=12,
                T=3,
                T_min=2,
                ksvd_iter=5,
                seed=fold,
                feat="abs_coef",
            )
            D, _ = learn_shared_D(graphs, tr, ncfg)
            structs, _ = encode_dataset_nodes(graphs, D, ncfg)
            structs = zscore_node_features(structs, tr)
            data_all = fuse_node(raw, structs)
            in_dim = int(data_all[0].x.size(-1))
        fold_data.append((tr, te, data_all, in_dim))

    # Independent model per fold; record Acc[fold, epoch]
    acc_mat = np.zeros((10, epochs), dtype=np.float64)
    t0 = time.time()
    for fold, (tr, te, data_all, in_dim) in enumerate(fold_data):
        torch.manual_seed(1000 + fold)
        model = GINPaper(in_dim, hidden, num_classes, num_layers=4, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader([data_all[i] for i in tr], batch_size=batch_size, shuffle=True)
        te_loader = DataLoader([data_all[i] for i in te], batch_size=64, shuffle=False)
        for ep in range(epochs):
            # lr decay every 50 epochs (paper)
            if ep > 0 and ep % 50 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
            model.train()
            for data in tr_loader:
                data = data.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(data.x, data.edge_index, data.batch), data.y.view(-1))
                loss.backward()
                opt.step()
            acc_mat[fold, ep] = accuracy(model, te_loader, device)
        print(
            f"  {variant} fold {fold}: last={acc_mat[fold, -1]:.3f} "
            f"max={acc_mat[fold].max():.3f} in_dim={in_dim}"
        )

    mean_curve = acc_mat.mean(axis=0)
    e_star = int(np.argmax(mean_curve))
    fold_at_star = acc_mat[:, e_star]
    return {
        "acc_mean": float(fold_at_star.mean()),
        "acc_std": float(fold_at_star.std()),
        "acc_percent_mean": float(fold_at_star.mean() * 100),
        "acc_percent_std": float(fold_at_star.std() * 100),
        "epoch_star": e_star + 1,
        "mean_curve_at_star": float(mean_curve[e_star]),
        "fold_scores_at_star": fold_at_star.tolist(),
        "per_fold_max_mean": float(acc_mat.max(axis=1).mean()),  # old-style for comparison
        "seconds": time.time() - t0,
        "in_dim": int(fold_data[0][3]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--hidden", type=int, default=32, help="paper bioinformatics: 16 or 32")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    root = _TRACK.parents[1] / "data" / "TUD"
    graphs, raw, y, num_classes = load_raw(args.dataset, root)

    results = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "hps": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "lr": args.lr,
            "lr_decay": "×0.5 every 50 epochs",
            "layers_mp": 4,
        },
        "variants": {},
    }

    for variant in ["gin_only", "gin_node_B0", "gin_node_RW"]:
        print(f"=== {variant} (Xu global epoch) ===")
        results["variants"][variant] = run_variant(
            variant,
            graphs,
            raw,
            y,
            num_classes,
            args.epochs,
            args.hidden,
            args.batch_size,
            args.dropout,
            args.lr,
            device,
        )
        v = results["variants"][variant]
        print(
            f"  >> e*={v['epoch_star']}  Acc={v['acc_percent_mean']:.1f}±{v['acc_percent_std']:.1f}%  "
            f"(per-fold-max mean would be {v['per_fold_max_mean']*100:.1f}%)"
        )

    path = _TRACK / "results" / "gin_xu_protocol.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / "GIN_XU_PROTOCOL_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    print(f"wrote {path}\nwrote {sm}")
    return 0


def _render(r: dict) -> str:
    lines = [
        "# GIN ± node structure under **Xu GIN paper epoch rule**",
        "",
        f"Dataset: **{r['dataset']}**",
        "",
        f"Protocol: `{r['protocol']['protocol_id']}`",
        "",
        r["protocol"]["epoch_select"],
        "",
        f"HPs (fixed for relative compare): `{r['hps']}`",
        "",
        "Paper also grids hidden∈{16,32}, batch∈{32,128}, dropout∈{0,0.5}; we fix one setting.",
        "",
        "| variant | Acc % @ global e* | e* | per-fold-max mean % (old) |",
        "|---------|-------------------|----|---------------------------|",
    ]
    for k, v in r["variants"].items():
        lines.append(
            f"| {k} | {v['acc_percent_mean']:.1f} ± {v['acc_percent_std']:.1f} | "
            f"{v['epoch_star']} | {v['per_fold_max_mean']*100:.1f} |"
        )
    lines += [
        "",
        "## Literature anchor (external)",
        "",
        f"- Xu GIN MUTAG: **{r['protocol']['literature_anchor']['GIN_MUTAG']}**",
        f"- {r['protocol']['literature_anchor']['note']}",
        "",
        "## Protocol difference vs our previous 94.6%",
        "",
        "| | Previous (node_struct_gin) | This run (Xu) |",
        "|--|------------------------------|---------------|",
        "| epoch pick | **per-fold** max held-out | **one global** e* on mean curve |",
        "| hidden | 64 (paper: social) | **32** (paper: bio) |",
        "| lr schedule | none | ×0.5 / 50 ep |",
        "",
        "Same-protocol relative compare: gin_only vs gin_node_* in the table above.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
