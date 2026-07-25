# KSVD 轨 · 执行计划（冻结 v0.1 · 2026-07-24）

## 已定决策

| 项 | 决定 |
|----|------|
| 结构管道 | R2：偏置 RW → **诱导子图** → 邻接 pad→\(m\) →（后）KSVD |
| 采样旋钮 | node2vec \(p,q\) + 长度/\(\|S\|\) cap + 边降权 \(\gamma\)（RW-C0）；**不永久删边** |
| 下游对标 | 主：**ogbg-molhiv**（OGB scaffold，test AUC@best val）；辅 ZINC；TUD 仅文献对照 |
| CIN TUD 数字 | 与 GIN 同属 **Xu 乐观 10-fold**；不与 strict 混表 |
| 评估 | 先 L1–L4 过程指标，后 L5 下游；分 `protocol_id` |

## 阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| **0** | 采样器 + 覆盖/重复/规模指标；B0 vs B1 vs M0 | **骨架可跑**；合成图已出 JSON；注意 B0/M0 的 \|S\| 不同，冗余比用 cnt/sg + 同族 B1 vs M0 |
| **1** | 诱导邻接向量化 + KSVD 烟测；重构/原子使用 | 未开始 |
| **2a** | 小分子图级探路（可选 MUTAG 等，自有 protocol） | 部分（MUTAG 融合历史） |
| **2b** | molhiv 主对标（协议 `ogb-molhiv-v0`） | **骨架**；见 `notes/molhiv_phase.md` |
| **2c** | 池化消融 mean/max/attn | 脚本就绪 `run_pool_ablation` |
| **2d** | 双通道 GINE ‖ \(s_G\) | 骨架 `run_molhiv_dual` |
| **3** | 消融 \(p,q,\gamma\)、诱导开/关、字典大小 | 未开始 |

## 协议 ID（只增不改语义）

见 [protocol.md](protocol.md)。

## 入口命令（阶段 0）

```bash
cd tracks/ksvd
python -m code.run_stage0 --config configs/stage0.yaml
```

结果 JSON → `results/stage0/`。
