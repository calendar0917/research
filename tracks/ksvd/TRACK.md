# Track: KSVD 结构字典学习

| 项 | 内容 |
|----|------|
| slug | `ksvd` |
| 状态 | **active** |
| 创建 | 2026-07-23 |
| 主线位置 | KSVD（相对 Kernel / GSN） |
| 定义笔记 | [notes/definition.md](notes/definition.md) |
| 主录音 | `docs/luyin/luyin10.txt`（+ luyin3/4 机制） |

## 问题（一句话）

如何用 **数据驱动的稀疏字典（KSVD）** 把图局部结构变成 **可还原** 的显式表征，并在感受野受限时（探索：随机游走构子图）仍保持可辩护的结构意义与下游可用性。

## 范围

- **做**
  1. 组内 KSVD 定义与 I/O（\(Y,D,X\)、readout）— 见 `notes/definition.md`
  2. 一阶邻域基线的缺陷与理论边界（>1-WL、感受野上限）
  3. 随机游走相关文献调研 → 约束下扩大感受野 / 降边冗余的方案草稿
  4. 最小实验设计（协议 lock 后）
- **不做**
  - 重开 GNN 全库复现（`gnn-gsn` archived）
  - 未调研前直接实现「无约束 RW 删边」当最终方法
  - 跨轨混协议主表

## 协议 / 评估

- 登记：[notes/protocol.md](notes/protocol.md)
- 主下游：`ogb-molhiv-v0`（未跑）；过程：`stage0-sample-v0`
- TUD/CIN 乐观 10-fold：**只引用**，不进 strict 主表
- 目的分维：A / B **分表**

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | **done** |
| 精读 / RW 调研 / CIN 协议 | **done** |
| 执行计划 | [notes/plan.md](notes/plan.md) |
| **阶段 0–3 烟测** | **done** → `results/SUMMARY.md` · `results/full_pipeline.json` |
| molhiv 主对标 | **P0/P1 骨架已就绪**（`run_molhiv_probe` / `run_molhiv_dual`）；缺依赖时需装 ogb |
| 池化 / 融合阶段 | **进行中** → `notes/molhiv_phase.md` |
| 交付 | 摘要已写；正式论文表未做 |

## 入口

| 路径 | 用途 |
|------|------|
| `code/` | 实现 |
| `configs/` | 配置 |
| `results/` | 本轨数字唯一源 |
| `notes/` | 决策与定义 |
| `docs/` | 短说明 |

## 禁止

- 与 gnn-gsn 结果混表横比（除非单独 deliverable 写清协议）
- 在 `../../paper` 扩张本课题
- 把表达力证明与 TUD/OGB Acc 混为一谈

## 下一步

1. ~~写清组内 KSVD 定义~~ → `notes/definition.md`
2. ~~RW 文献调研 v0.1~~ → `notes/rw_survey.md`（主候选 **R2+RW-C0**）
3. ~~精读 P0~~；~~评估+分阶段实验草案~~ → `notes/rw_survey.md` §7–§8
4. **已定**：向量化 = 诱导邻接 pad→\(m\)；下游对标 **CIN / molhiv 方向**（见 [cin.md](../../docs/literature/deep/cin.md)）
5. ~~烟测 / followup / fixes~~ → `results/FIXES_SUMMARY.md`：**共享字典**在合成 0.55→0.86
6. ~~rich 读出 / distant / MUTAG gate~~ → `results/NEXT_SUMMARY.md`（distant 被局部度破解，RW 反降分）
7. ~~C4 vs C8~~ → `results/C4_SUMMARY.md`：**B0~0.85，RW+共享KSVD~1.0**
8. ~~可视化 + GIN paper-optimistic 拼接~~ → `results/viz/` · `GIN_STRUCT_SUMMARY.md`（拼接 **降分**）
9. 共享字典说明：`notes/shared_dict.md`（非一开始就定，fixes 后采纳）
10. ~~节点级系数融合~~ → `NODE_STRUCT_SUMMARY.md`（节点级 93.6% ≫ 广播 83.5%；≈ gin_only）
11. ~~Xu 全局 epoch 协议对齐~~ → `GIN_XU_PROTOCOL_SUMMARY.md`（gin_only **89.4%** 贴论文）
12. ~~融合对照（concat/residual/gate）~~ → `FUSION_XU_SUMMARY.md`（concat 90.4 微升；残差/门控≈only）
13. ~~RW 融合~~ → `FUSION_XU_RW_SUMMARY.md`（gate 90.9；concat/residual 变差）；`notes/rw_params.md`
14. ~~多 walk（r=5）+ pool~~ → `FUSION_XU_RW_R5_SUMMARY.md`（concat 回升但仍 ≤ only）
15. ~~文献 v0.2 + 精读补全 + 调研报告~~ → `notes/rw_survey.md` · `docs/survey_report.md`
16. ~~RW 可行性+CIN路线~~ → `notes/rw_feasibility_and_cin_path.md`
17. ~~过程指标 R1/R4/R5/R6~~ → `results/COVERAGE_RF_SUMMARY.md` · `code/coverage_sample.py`
18. ~~图级取样做实~~ → `GRAPH_LEVEL_SOLID_SUMMARY.md` · `notes/graph_level_algorithm.md`
19. ~~交互可视化 HTML~~ → `results/viz_pipeline/index.html`
20. 与导师对报告；分子下游后置
21. ~~molhiv 阶段计划 + 协议细则~~ → `notes/molhiv_phase.md` · `protocol.md`
22. ~~MIL attention pool + 合成消融脚本~~ → `ksvd.pool_X` · `run_pool_ablation.py`
23. ~~结构-only / 双通道骨架~~ → `run_molhiv_probe.py` · `run_molhiv_dual.py`
24. **下一步**：装 ogb → `run_pool_ablation` → `run_molhiv_probe --max-graphs 2000`
