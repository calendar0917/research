#!/usr/bin/env python3
"""GIN（非严格协议 / paper-optimistic）图分类基线（单文件版）。

协议（与 WL 基线同折叠、可配对比较）
------------------------------------
  - CV: RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)
    —— 与 run_wl_subtree_kernel.py 完全相同的 split（同数据集、同种子 → 逐 fold 配对）
  - 每 fold：在训练折上训练 GIN，**留出折即验证集**（无独立测试集），
    按「验证集 acc 最大」选 epoch，报告该 epoch 的 acc / macro-F1
  - 即 paper-optimistic / Xu-GIN 习惯：选择与报告都在同一个留出折（乐观，自检用）

模型（与 tracks/ksvd code/run_gin_struct_paper.py 同构）
  - GIN-0 (train_eps=False)，4 层，每层 MLP(64)-BN-ReLU，层间 sum-of-scores 读出
  - Adam(lr=0.01)，无 weight decay，dropout=0.5，batch=32，120 epochs
  - 无节点属性数据（本轨 4 个数据集均无 x）：节点特征=常数 1（featureless，house 惯例）
    可用 --node-feat degree 切换为归一化度特征

用法
----
  uv run --group wl-kernel --group ksvd python code/run_gin_nonstrict.py
  uv run --group wl-kernel --group ksvd python code/run_gin_nonstrict.py \
      --datasets IMDB-BINARY IMDB-MULTI --workers 8

输出
  results/{DATASET}__gin_seed{SEED}.json   每 (数据集, 种子)：全部 fold 的 acc/f1/最佳 epoch
  results/{DATASET}__gin_summary.json      跨种子汇总
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

DEFAULT_DATASETS = ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "COLLAB"]
DEFAULT_SEEDS = [0, 42, 123, 1024, 2026, 777, 3407, 999, 111, 888]
REPO_ROOT = Path(__file__).resolve().parents[3]

_DS = None  # fork 后由子进程继承，避免每任务 pickling 数据


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class GINPaper(nn.Module):
    """GIN-0, 4 层, layer-wise sum of scores（与 ksvd 轨 paper-optimistic GIN 同构）。"""

    def __init__(self, in_dim: int, hidden: int, num_classes: int,
                 num_layers: int = 4, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            self.convs.append(GINConv(_mlp(d_in, hidden), train_eps=False))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.predictors = nn.ModuleList(
            [nn.Linear(in_dim, num_classes)]
            + [nn.Linear(hidden, num_classes) for _ in range(num_layers)]
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


def prepare_data(ds, node_feat: str):
    """TUDataset -> list[Data]，特征化（constant 或 normalized degree），标签映射 0..C-1。"""
    from torch_geometric.data import Data

    ys = []
    for data in ds:
        yy = data.y.view(-1)
        ys.append(int(yy[0].item()))
    uniq = sorted(set(ys))
    remap = {u: i for i, u in enumerate(uniq)}
    out = []
    for i, data in enumerate(ds):
        n = int(data.num_nodes)
        if getattr(data, "x", None) is not None and data.x is not None:
            x = data.x.float()
        elif node_feat == "degree":
            deg = data.edge_index[0].bincount(minlength=n).float()
            m = deg.max().clamp(min=1.0)
            x = (deg / m).unsqueeze(1)
        else:
            x = torch.ones((n, 1))
        out.append(
            Data(x=x, edge_index=data.edge_index,
                 y=torch.tensor([remap[ys[i]]], dtype=torch.long), num_nodes=n)
        )
    return out


def train_fold(args: dict) -> dict:
    """单个 fold：训练（训练折）+ 留出折选 epoch（无测试集）。fork 子进程内执行。"""
    ds = _DS
    seed, fold_idx, tr, va, epochs, hidden, lr, dropout, batch_size, device = (
        args["seed"], args["fold_idx"], args["tr"], args["va"], args["epochs"],
        args["hidden"], args["lr"], args["dropout"], args["batch_size"], args["device"],
    )
    torch.set_num_threads(1)
    torch.manual_seed(seed * 10000 + fold_idx)

    train_data = [ds[i] for i in tr]
    val_data = [ds[i] for i in va]
    in_dim = int(train_data[0].x.size(-1))
    num_classes = int(max(ds[i].y.item() for i in tr)) + 1

    model = GINPaper(in_dim, hidden, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_acc, best_f1, best_ep = -1.0, -1.0, -1
    val_accs: list[float] = []
    for ep in range(1, epochs + 1):
        model.train()
        for data in tr_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y.view(-1))
            loss.backward()
            opt.step()
        # ---- 留出折 = 验证集：选 epoch & 报告都在它上面（非严格/乐观） ----
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for data in va_loader:
                data = data.to(device)
                preds.append(model(data.x, data.edge_index, data.batch).argmax(dim=-1).cpu())
                ys.append(data.y.view(-1).cpu())
        preds = torch.cat(preds).numpy()
        ys = torch.cat(ys).numpy()
        acc = float(accuracy_score(ys, preds))
        f1v = float(f1_score(ys, preds, average="macro"))
        val_accs.append(acc)
        if acc >= best_acc:
            best_acc, best_f1, best_ep = acc, f1v, ep

    return {"fold": fold_idx, "best_acc": best_acc, "best_f1": best_f1,
            "best_epoch": best_ep, "val_accs": val_accs}


def run_seed(ds, seed, n_splits, n_repeats, hps, workers, device) -> dict:
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    y = np.array([d.y.item() for d in ds])
    folds = list(rskf.split(np.zeros(len(y)), y))

    tasks = []
    for fold_idx, (tr, va) in enumerate(folds):
        tasks.append({
            "seed": seed, "fold_idx": fold_idx, "tr": tr, "va": va,
            "epochs": hps["epochs"], "hidden": hps["hidden"], "lr": hps["lr"],
            "dropout": hps["dropout"], "batch_size": hps["batch_size"],
            "device": device,
        })

    results: list[dict] = []
    t0 = time.time()
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            for res in pool.map(train_fold, tasks):
                results.append(res)
                if len(results) % 10 == 0:
                    print(f"    [{seed}] fold {len(results)}/{len(tasks)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    else:
        for task in tasks:
            results.append(train_fold(task))
            if len(results) % 10 == 0:
                print(f"    [{seed}] fold {len(results)}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: r["fold"])
    acc = np.array([r["best_acc"] for r in results])
    f1 = np.array([r["best_f1"] for r in results])
    return {
        "seed": seed,
        "fold_acc": acc.tolist(),
        "fold_f1": f1.tolist(),
        "best_epochs": [r["best_epoch"] for r in results],
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "mean_f1": float(f1.mean()),
        "std_f1": float(f1.std()),
        "elapsed_s": float(time.time() - t0),
    }


def summarize(results: list[dict]) -> dict:
    seeds_acc = [r["mean_acc"] for r in results]
    seeds_f1 = [r["mean_f1"] for r in results]
    return {
        "dataset": results[0].get("dataset"),
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "across_seed_mean_acc": float(np.mean(seeds_acc)),
        "across_seed_std_acc": float(np.std(seeds_acc)),
        "across_seed_mean_f1": float(np.mean(seeds_f1)),
        "across_seed_std_f1": float(np.std(seeds_f1)),
        "per_seed": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--node-feat", choices=["constant", "degree"], default="constant")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="并行 fold 进程数（每进程单线程）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "TUD")
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hps = {"epochs": args.epochs, "hidden": args.hidden, "lr": args.lr,
           "dropout": args.dropout, "batch_size": args.batch_size,
           "layers": 4, "node_feat": args.node_feat}
    device = args.device
    print(f"data root: {args.data_root}  device: {device}  workers: {args.workers}")
    print(f"datasets : {args.datasets}\nseeds    : {args.seeds}")
    print(f"HPs      : {hps}")

    global _DS
    for name in args.datasets:
        print(f"\n{'=' * 70}\nDataset: {name}\n{'=' * 70}")
        from torch_geometric.datasets import TUDataset

        root = Path(args.data_root) / name
        ds_raw = TUDataset(root=str(root), name=name)
        _DS = prepare_data(ds_raw, args.node_feat)
        y = np.array([d.y.item() for d in _DS])
        print(f"  {len(_DS)} graphs, {len(np.unique(y))} classes, "
              f"avg nodes {np.mean([int(d.num_nodes) for d in _DS]):.1f}")

        seed_results: list[dict] = []
        for seed in args.seeds:
            out_path = out_dir / f"{name}__gin_seed{seed}.json"
            if out_path.exists() and not args.force:
                r = json.load(open(out_path, encoding="utf-8"))
                r["dataset"] = name
                seed_results.append(r)
                print(f"  seed={seed}: cached {r['mean_acc']:.4f}+-{r['std_acc']:.4f}")
                continue
            print(f"  seed={seed}: running {args.n_splits}x{args.n_repeats} CV ...")
            r = run_seed(_DS, seed, args.n_splits, args.n_repeats, hps,
                         args.workers, device)
            r["dataset"] = name
            print(f"    done acc={r['mean_acc']:.4f}+-{r['std_acc']:.4f} "
                  f"f1={r['mean_f1']:.4f}+-{r['std_f1']:.4f} ({r['elapsed_s']:.0f}s)")
            json.dump(r, open(out_path, "w", encoding="utf-8"), indent=2, default=str)
            seed_results.append(r)

        summary = summarize(seed_results)
        sum_path = out_dir / f"{name}__gin_summary.json"
        json.dump(summary, open(sum_path, "w", encoding="utf-8"), indent=2, default=str)

        print(f"\n  Summary ({name}, {summary['n_seeds']} seeds):")
        for r in seed_results:
            print(f"  {r['seed']:>6}  {r['mean_acc']:.4f}+-{r['std_acc']:.4f}"
                  f"  {r['mean_f1']:.4f}+-{r['std_f1']:.4f}")
        print(f"  across: acc {summary['across_seed_mean_acc']:.4f}"
              f"+-{summary['across_seed_std_acc']:.4f}  "
              f"f1 {summary['across_seed_mean_f1']:.4f}"
              f"+-{summary['across_seed_std_f1']:.4f}")
        print(f"  saved -> {sum_path}")


if __name__ == "__main__":
    main()
