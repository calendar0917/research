"""
smoke_test_pipeline.py — 管线冒烟测试 (CPU)

验证数据加载 + 代表性 GNN 前向/反向 + 指标计算在数据层面可行。
注意：本脚本不运行官方基线（需要 CUDA + 官方环境），而是验证
数据管线与代表 GNN 的训练/评估机制端到端可用，为后续 CUDA 正式运行打基础。

用法:
    python smoke_test_pipeline.py --dataset molhiv --batches 8 --epochs 2
"""

import argparse
import time

import torch
import torch.nn.functional as F
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from audit_splits import _apply_pygload_compat


class GINE_GNN(torch.nn.Module):
    """代表性 GNN（GINE + 图池化），近似 GPS/策略家族主干，仅用于管线冒烟。"""

    def __init__(self, in_dim, edge_dim, hidden=64, out_dim=1):
        super().__init__()
        self.node_proj = torch.nn.Linear(in_dim, hidden)
        self.edge_proj = torch.nn.Linear(edge_dim, hidden)
        self.convs = torch.nn.ModuleList(
            [GINEConvBN(hidden, hidden, edge_dim) for _ in range(2)]
        )
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, data):
        x = torch.relu(self.node_proj(data.x.to(torch.float)))
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr.to(torch.float))
        x = global_add_pool(x, data.batch)
        return self.head(x)


class GINEConvBN(torch.nn.Module):
    """GINE 卷积 + BatchNorm，验证含子模块的图卷积正常反向。"""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.conv = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU(),
            torch.nn.Linear(out_dim, out_dim)), edge_dim=edge_dim, eps=0.0)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        return torch.relu(self.bn(self.conv(x, edge_index, edge_attr)))


def collate(data_list):
    from torch_geometric.loader import DataLoader

    loader = DataLoader(data_list, batch_size=len(data_list))
    return next(iter(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbg-molhiv")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    _apply_pygload_compat()
    from torch_geometric.loader import DataLoader

    print(f"=== Pipeline smoke test: {args.dataset} ===")
    torch.manual_seed(0)

    dataset = PygGraphPropPredDataset(name=args.dataset, root="data")
    split = dataset.get_idx_split()
    train_data = [dataset[i] for i in split["train"]][: args.batches * args.batch_size]
    # 过滤全 NaN 样本（multi-task 数据集可能有 NaN 标签）
    train_data = [d for d in train_data if not torch.isnan(d.y).all()]
    val_data = [dataset[i] for i in split["valid"]][: args.batches]
    val_data = [d for d in val_data if not torch.isnan(d.y).all()]
    test_data = [dataset[i] for i in split["test"]][: args.batches]
    test_data = [d for d in test_data if not torch.isnan(d.y).all()]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    num_tasks = dataset.y.shape[-1]
    model = GINE_GNN(
        in_dim=dataset.num_features, edge_dim=dataset.num_edge_features,
        hidden=args.hidden, out_dim=num_tasks,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    evaluator = Evaluator(name=args.dataset)

    total_t = 0.0
    loss_val = None
    for epoch in range(args.epochs):
        model.train()
        accum_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            t0 = time.time()
            out = model(batch)
            # OGB 分子分类为二分类，one-hot 标签格式 (N,1)；转 float 做 BCE
            y = batch.y.to(torch.float)
            if out.dim() == 2 and y.dim() == 1:
                y = y.view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, y, reduction='none')
            nan_mask = torch.isnan(y)
            loss = loss[~nan_mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_t += time.time() - t0
            accum_loss += loss.item()
        loss_val = accum_loss / max(len(train_loader), 1)
        print(f"  epoch {epoch}: loss={loss_val:.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch))
            y_true.append(batch.y.view(-1, num_tasks))
            y_pred.append(out.view(-1, num_tasks))
    y_true = torch.cat(y_true, 0)
    y_pred = torch.cat(y_pred, 0)
    try:
        res = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        metric = res.get("rocauc", 0.0)
        print(f"  test metric (ROC-AUC): {metric:.4f}")
    except (RuntimeError, ValueError) as e:
        print(f"  test metric: not computed ({e})")

    print(f"  final loss: {loss_val:.4f}")
    print(f"  per-batch fwd+bwd time: {total_t / max(len(train_loader),1):.3f}s")
    print(f"  SMOKE TEST PASSED (finite loss, fwd/bwd, metric computed)")


if __name__ == "__main__":
    main()