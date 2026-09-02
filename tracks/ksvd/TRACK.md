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

### luyin16 阶段入口

- 阶段说明：`notes/luyin16_plan.md`
- 配置目录：`configs/luyin16/`
- 结果目录：`results/luyin16/`
- 新实验入口：`experiments/luyin16/`
- 可复用包：`src/ksvd_research/`；新代码不要依赖顶层 `code` 名称
- 原则：先验证结构特征的独立价值，再决定是否进入结构—语义融合；不继续无目的堆叠模型。

## 入口

| 路径 | 用途 |
|------|------|
| `src/ksvd_research/` | 可复用核心包（当前为兼容层，逐步迁移） |
| `experiments/luyin16/` | luyin16 新实验入口 |
| `code/` | 历史实现与复现入口，不再平铺新增实验 |
| `tests/` | pytest 维护测试 |
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
24. ~~装 ogb + 探针~~ → `results/molhiv/MOLHIV_PHASE_SUMMARY.md`
25. **发现**：n8k 结构-only max test≈0.61 > degree；n3k dual concat **未超** gine_only
26. ~~设计消融 n6k~~ → `results/molhiv/DESIGN_VERDICT.md`
27. **有效**：coverage + max + 小字典 A8T2；残差相对 size +3～4pt valid
28. **无效**：抬 cover、mean、图级 concat；下一步节点级融合 / patch 语义
29. ~~B/C/D/A next-round~~ → `results/molhiv/NEXT_ROUND_VERDICT.md`
30. **下调**：residual 多种子不稳；chem/ring/node_gate 均无稳定超 GINE
31. **收窄主线**：采样+可还原字典机制；或环/cell 对象升级；molhiv 增益暂非主证据
32. ~~luyin14 路线闭环~~ → `results/luyin14/ROUTE_CLOSURE_20260812.md`
33. **补边 no-go**：EDGE100−FAIR95 在四数据集均值全负；结合既有 rate 审计，默认 FAIR95 + residual sidecar
34. **关系 no-go**：TRUE−SHUFFLED 仅 IMDB-BINARY +1.1pt，未跨数据集复现
35. **融合 no-go**：MUTAG/PTC_MR 的 KSVD 融合均未超 feature-only；简单 feature+stats 反而稳定更强
36. **路线结论**：普通 KSVD 固定为 compressor / diagnostic baseline；不再扫描 K/T、restart、fusion depth
37. ~~rich sparse-code readout 迁移审计~~ → `results/luyin14/RICH_READOUT_20260812.md`
38. **rich 归因结论**：相对 coarse 在两个 IMDB 上 `+3.9/+2.4pt`、均 9/9 wins；但 FINAL−INIT 仅 `+0.4/+0.3pt`，且没有 graph-stats 外增量
39. ~~relation-conditioned structured pursuit~~ → `results/luyin14/STRUCTURED_PURSUIT_20260812.md`
40. **structured no-go**：support agreement 明显上升，但 reconstruction 恶化 `+33%` 到 `+271%`，三个主 gate 全失败
41. ~~uncompressed RAW relation terminal screen~~ → `results/luyin14/RAW_RELATION_20260812.md`
42. **RAW relation 结论**：TRUE 相对弱 bag 在 IMDB-BINARY/MUTAG 有增益，但不稳定胜 SHUFFLED、四数据集均无 stats 外增量；停止同一 TUD 上的 patch-graph/Transformer
43. **下一研究问题**：若继续 downstream，改做低阶统计匹配、标签由 patch 空间组合决定的 benchmark；先跑 RAW/INIT/FINAL × BAG/TRUE/SHUFFLED，再决定是否需要 relation-aware encoder
44. 完整决策：`results/luyin14/NEXT_DIRECTION_DECISION_20260812.md`
45. ~~节点级 strict concat/FiLM~~ → `results/luyin14/NODE_LEVEL_FUSION_STAGE_A_20260812.md`：MUTAG 有位置绑定信号，但 FINAL=INIT；PTC_MR 不复现
46. ~~OOF late fusion / patch-local bilinear~~ → `results/luyin14/OOF_MULTIVIEW_FUSION_20260812.md`、`PATCH_LOCAL_MULTIMODAL_20260812.md`：融合容量不是主瓶颈
47. ~~structure–attribute shared-code dictionary~~ → `results/luyin14/JOINT_MULTIVIEW_DICTIONARY_STAGE_A_20260812.md`：解决无类型 patch 的零学习余量，但跨数据集 gate 失败
48. ~~edge-aware joint dictionary + GINE，3 split seeds~~ → `results/luyin14/EDGE_AWARE_JOINT_MULTI_SPLIT_20260812.md`
49. **edge-aware 结论**：seed-0 MUTAG 强正，但 9-fold 汇总仅 `+4.3pt`、5/9 wins，binding 3/9 wins；PTC_MR `-2.1pt`，不稳定
50. **下一优先**：换更大、带 node/edge attributes 的真实 benchmark 做固定协议外部验证；先升级 typed radius-2 patch 语义，不增加 cross-attention
51. ~~TU 中等规模外部验证~~ → `results/luyin14/TUD_EXTERNAL_VALIDATION_20260813.md`
52. **Mutagenicity Stage A**：seed 0 FINAL−GINE `+7.0pt`、TRUE−SHUFFLED `+3.3pt`、FINAL−INIT `+3.9pt`，按协议扩展
53. **Mutagenicity 9-fold 结论**：FINAL−GINE `-0.4pt`（3/9 wins）；binding `+0.8pt`（5/9），FINAL−INIT `+0.6pt`（5/9），强信号不跨 split
54. **NCI1 结论**：FINAL−GIN `+4.4pt`，但 FINAL−INIT `-1.3pt`；shared prototype 有用，不能归因于 K-SVD updates
55. **radius-2 substrate PASS**：两数据集 r2 ego mean≈6.3--6.5，cap8 平均保留 95.7%/98.1%，仍是局部对象；下一轮只可换 patch 语义，不调融合容量
56. ~~TU typed radius-2 Stage A~~ → `results/luyin14/TUD_RADIUS2_VALIDATION_20260813.md`
57. **Mutagenicity radius-2 no-go**：typed patch types 424，但 TRUE=SHUFFLED=INIT≈0.751，重构改善未转分类增益
58. **NCI1 radius-2 9-fold**：FINAL−GIN `+5.0pt`（8/9）、TRUE−SHUFFLED `+1.8pt`（7/9），但 FINAL−INIT `-0.06pt`
59. **路线判断**：radius-2 joint patch representation 有稳定信号；普通无监督 K-SVD updates 仍无任务归因，不进入 cross-attention
60. ~~luyin16 导师固定特征概念复现~~ → `results/luyin16/MENTOR_CONCEPT_ROUTE_VERDICT_20260829.md`
61. **MolHIV fixed-feature 结论**：显式 topology+chemistry + XGBoost official-valid `0.7817`；但 chem/topology K-SVD update、S 外融合和 624D reconstruction proxy 全部无稳定增量
62. **停止项**：不继续扫 K/T、Beam、objective 或 fusion；导师约 0.80 路线必须先取得 69/624D 上游 schema、payload 与 best params，否则只能称 proxy
63. **统计对象补充**：slot-aligned `8×64+28×4=624` proxy 的 raw `0.7014` 高于全局统计 proxy `0.6649`，但仍无 S 外增量；occurrence-sum `0.5821`，不再猜聚合方式
64. ~~clean structural-role fusion~~ → `results/luyin16/CLEAN_STRUCTURAL_ROLE_FUSION_20260901.md`
65. **结构编码结论**：去掉 OGB degree/ring 后，rooted-WL factorized raw 在 official-train 三折相对 coarse `+5.47pt`、3/3 wins；coarse role 确实过弱
66. **迁移结论**：official-valid 上 centered 仍胜 shuffle `+1.56pt`，但仅比 `T+A` `+0.07pt`；factorized raw 比 `T+A` `-1.34pt`，依赖存在但无稳定标签增量
67. **停止项**：typed-edge 内部 `-1.64pt`、0/3 wins，exact topology valid type coverage 仅 `49.5%`；不扫 WL bins/rounds、exact orbit、path、attention 或 K-SVD
68. ~~task-aligned interaction / Optuna audit~~ → `results/luyin16/TASK_ALIGNED_INTERACTION_TUNING_20260901.md`
69. **超参数结论**：nested Optuna 使 `T+A` `+1.52pt`、centered `+0.23pt`、raw `-1.52pt`；raw/centered 相对 `T+A` 的 gap 分别缩小 `3.03pt/1.28pt`，均 0/3 folds 改善，未调参不是 interaction 失败的解释
70. **稀疏 residual no-go**：cross-fitted centered 胜 shuffle `+2.93pt`、2/3 wins，但比 tuned `T+A` `-0.57pt`、仅 1/3 wins；raw 比 baseline `-1.43pt`
71. **路线冻结**：不扩大 Optuna 或继续 selector/top-k/C 搜索，不重新打开 official-valid；rooted-WL `T+A` 保持主线，centered 只保留 dependence diagnostic
72. ~~rooted-WL frozen terminal test~~ → `results/luyin16/STRUCTURAL_ROLE_TERMINAL_TEST_20260901.md`
73. **单分支 test**：train+valid refit `T+A=0.7610`、centered `0.7716`，centered `+1.06pt`、4/5 seeds；调参改善 valid 但未消除 valid--test mismatch
74. **late fusion test**：预注册固定 50/50 `T+A+centered` paired-seed mean `0.7825`，十模型 probability ensemble **`0.7839`**；两个独立 expert 的互补性高于 feature-level residual
75. **终端冻结**：official test 历史上已被查看，本轮只算 controlled terminal evaluation；不再用该 test 调 view/权重/参数，下一步转向新 outer splits 上的 constrained late fusion
76. ~~exact rooted topology × conditional chemistry 快筛~~ → `results/luyin16/EXACT_CONDITIONAL_FUSION_20260901.md`
77. **融合层级结论**：exact/orbit 位置绑定 no-go；rooted-WL 在本轮观测 radius-2 patch 上无 exact collision。真正可复现的是同一 patch 内 topology 与 attribute multiset 的条件配对，true−patch-pair-shuffle `+2.36pt`、8/9 seed×fold wins
78. **预测边界与下一步**：conditional pairing 机制存在，但直接 joint XGBoost 未超过 `S+WL+attribute`；不扫 K/prototype/XGB，不看 valid/test，只允许一次 cross-fitted conditional residual gate
