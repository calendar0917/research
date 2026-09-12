# DEVIATIONS.md — 基线代码修改记录

> 本文件记录所有对官方代码的兼容性 patch。
> 每个 patch 必须同时保存: source_commit.txt, compatibility.patch, patch_reason.md

## 总则

- 只允许以下情况 patch: PyTorch/PyG API 已删除或改名；数据下载 URL 失效；旧版 Python 语法不兼容；缺少非核心参数。
- 不允许重写核心模型、采样策略、损失函数或数据划分。
- 如果必须重写核心组件，主协议标记为 unavailable，改写版本放 exploratory。

## Patch 记录

| ID | 方法 | 数据集 | 原因 | 状态 | 日期 |
|----|------|--------|------|------|------|
| PL-001 | (审计 wrapper) | molhiv/moltox21/molbace | PyTorch≥2.6 `torch.load` 默认 `weights_only=True`，破坏 ogb 处理后的 .pt 加载 | applied | 2026-08-24 |

## Patch 模板

### Patch: PL-001

**方法**: (审计 wrapper, `audit_splits.py` — 非官方基线代码)
**数据集**: molhiv / moltox21 / molbace
**原因**: PyTorch ≥ 2.6 将 `torch.load` 默认改为 `weights_only=True`，拒绝由 ogb 1.3.6 + torch_geometric dataclass 序列化的 .pt 文件。
**源 commit**: N/A（不涉及上游基线代码）
**修改内容**: 在 ogb 的 `torch.load` 调用前注入 `weights_only=False`（仅影响加载，不影响 split 内容）。
**影响范围**: 仅数据集加载；split hash 基于 `get_idx_split()` 索引计算，与加载方式无关。
**验证**: molhiv 41127 (32901/4113/4113)，moltox21 7831 (6264/783/784)，molbace 1513 (1210/151/152)。

## 运行环境差异（非代码 patch，但影响结果解释）

| 项 | 官方预期 | 本机实际 | 影响 |
|----|----------|----------|------|
| 硬件 | NVIDIA A100/GPU | CPU-only (16 核) | 训练速度 ~50-100x 慢，数值路径一致（无 CUDA 确定性差异） |
| 环境 | torch 1.13 + pyg 2.2 (官方) | 相同版本，CPU 构建 | 无差异 |
| batch_size (molhiv) | HOD-GNN 论文附录: 128；GraphGPS 官方 config: 32 | 32 (官方 config) | 按计划 §7.1 保留官方基线配置；论文数字 78.80 可能基于其自己的 128 设置，DEVIATIONS 记录差异 |
| MOLHIV epochs | 论文 100 | 100 (官方 config max_epoch) | 一致 |
| seed | 论文未披露 | 0 (provisional) | paper-seeds-provisional |

**方法**: <method>
**数据集**: <dataset>
**原因**: 
**源 commit**: 
**修改内容**:
**影响范围**:
**验证**: