# Track: HOD-GNN 基线复现审计

| 项 | 内容 |
|----|------|
| slug | `hod-gnn-replication` |
| 状态 | `active` |
| 创建 | 2026-08-24 |
| 主线位置 | 基线复现审计 |

## 问题（一句话）

HOD-GNN 论文中作为对比基线的 GPS、GraphViT、Full、Random、Policy-Learn 五个方法，在其官方代码和官方配置下，能否在论文报告的六个数据集上复现论文数字？

## 范围

- 做：GPS、GraphViT、Full、Random、Policy-Learn 五个基线在 ZINC、MOLTOX21、MOLBACE、MOLHIV、Peptides-func、Peptides-struct 上的复现实验。
- 做：两套协议（paper-seeds 和 official-seeds），分别记录结果。
- 做：来源登记、split 审计、smoke test、完整实验、汇总审计。
- 不做：HOD-GNN 本身的重新实现。
- 不做：与 ksvd 主线或 gnn-gsn 归档轨的结果混合。
- 不做：论文中标记为 "–" 的单元的补跑。
- 不做：超参数调优（official-tuned 另计）。

## 协议 / 评估

- 是否复用全局图分类 strict：否（本轨独立协议）
- `protocol_id`：hod-gnn-replication-v1
- 主表角色：旁路复现审计

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | ✓ |
| 精读 / 概念 | ✓ |
| 来源登记 | ✓ |
| 环境与 split 审计 | ✓ (OGB: molhiv/moltox21/molbace 已校验；ZINC 阻塞 — 数据源 403) |
| Smoke test | ✓ (管线冒烟通过: MOLHIV/MOLBACE 前向/反向/指标，CPU) |
| paper-seeds 正式实验 | 阻塞（需 CUDA GPU + 官方 conda 环境） |
| official-seeds 正式实验 | 阻塞（需 CUDA GPU + 官方 conda 环境） |
| 汇总和审计 | 待开始 |
| 交付 | 待开始 |

## 入口

| 路径 | 用途 |
|------|------|
| `code/` | 实现与跑法 |
| `configs/` | 配置 |
| `results/` | JSON + registry（本轨唯一数字源） |
| `notes/` | 轨内决策、对照 |
| `docs/` | 轨内给人看的短说明 |

## 禁止

- 与其它轨结果混表横比（除非单独 deliverable 写清协议）
- 在 tracks/ 之外另立仓库扩张本课题（旧仓 `../../paper` 已不可访问）
- 用非官方代码冒充基线复现

## 下一步

1. 完成来源登记 → 见 docs/IMPLEMENTATION_PLAN.md