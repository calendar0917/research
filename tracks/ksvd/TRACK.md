# Track: KSVD 结构字典学习

> **2026-09-07 起**：本 TRACK 不再承担实验流水数据库职责。新实验统一走
> `uv run research run` → `runs/`（本地执行真相，Git-ignored）；有意义 →
> `uv run research promote` → `records/runs/`（durable facts，Git-tracked，
> fresh clone 可读）；判断/决策进 `records/{claims,decisions}`；
> 导航见 `STATE.yaml`。`results/luyin16/` 保留为 legacy evidence archive
> （只读，不再作为新实验数字的默认入口）。下方历史进度保留，不代表当前记录方式。

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
- 主下游：`ogb-molhiv-v0` + PyG `ZINC subset=True`（official split，luyin16 已跑通）；过程：`stage0-sample-v0`
- TUD/CIN 乐观 10-fold：**只引用**，不进 strict 主表
- 目的分维：A / B **分表**

## 进度

| 阶段 | 状态 |
|------|------|
| 问题与范围 | **done** |
| 精读 / RW 调研 / CIN 协议 | **done** |
| 执行计划 | [notes/plan.md](notes/plan.md) |
| **阶段 0–3 烟测** | **done** → `results/SUMMARY.md` · `results/full_pipeline.json` |
| molhiv 主对标 | **done**（luyin16 官方 scaffold：显式统计 S=0.7817，分布读出 0.8341 valid，严格 test 0.8087 / 集成 0.8135） |
| 池化 / 融合阶段 | **已成判断**：机制冻结在统计交互 + XGBoost；可学习融合（attention/FiLM/gate）无稳定增量 |
| 交付 | 总览已写：`results/luyin16/EXPERIMENT_ROUTE_SUMMARY_20260829_ONWARD.md`；正式论文表未做 |

### luyin16 阶段入口

- 阶段说明：`notes/luyin16_plan.md`
- 阶段总览：`results/luyin16/EXPERIMENT_ROUTE_SUMMARY_20260829_ONWARD.md`
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
| `runs/` | 本地执行真相（Git-ignored，控制面自动写） |
| `records/runs/` | durable promoted facts（Git-tracked） |
| `records/claims/` `records/decisions/` | 科研判断 / 路线决策 |
| `results/luyin16/` | legacy evidence archive（2026-09-07 前后历史体系，只读） |
| `notes/` | 决策与定义 |
| `docs/` | 短说明 |

## 禁止

- 与 gnn-gsn 结果混表横比（除非单独 deliverable 写清协议）
- 在 tracks/ 之外另立仓库扩张本课题（旧仓 `../../paper` 已不可访问）
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
79. ~~luyin16 概念复现 → 机制 → 显式模型全路线~~ → `results/luyin16/EXPERIMENT_ROUTE_SUMMARY_20260829_ONWARD.md`；主判：**K-SVD=压缩器而非任务表征学习器**；机制阶段冻结在统计交互 + XGBoost
80. ~~显式模型路线（0903–0905）~~ → ZINC 精确 patch 路径 `0.1872/0.1381` → **层级关系上下文 `0.1816/0.1346`（当前最佳）**；MolHIV patch 路径 `0.8028/0.7852`，统一 OOV-only `0.8383/0.7727`（精确 token 反降到 0.805 valid）
81. **待答两个问题**：①跨中心交互在新骨架划分上是否仍稳定；②ZINC 0.13–0.18 差距来自 patch 身份 / 关系距离 / 读出统计，还是缺少全图上下文

82. ~~compact-v2 budget reallocation (09-07)~~ → `notes/compact_hybrid_v2_budget_reallocation.md`：valid 0.184158 @56（98,549 params）；refit-test 0.135362
83. ~~compact-v2 info-gap audit (09-07)~~ → `notes/compact_v2_information_gap_audit.md`：13 探针全 NO-GO；残差 = target 定义 long-cycle 项（valid 3.5% 分子 = 27.1% MAE mass）
84. ~~v3 ring-context conditioning NO-GO (09-08)~~ → `notes/compact_v3_context_conditioned_patch_representation.md` + `records/decisions/decision-compact-v3-context-conditioning-nogo-20260908.yaml`
85. ~~long-cycle audit GO (09-08)~~ → `notes/zinc_long_cycle_audit.md`：机制验证 + 不变量 oracle 上限 +0.0117
86. ~~compact-v4 global topology channel (09-09)~~ → `notes/compact_v4_global_topology_channel.md`：hinge valid +0.0141；seed-0 refit 口径 −0.0041（**协议口径分裂**，见 addendum）
87. **protocol correction + multi-seed confirmation (09-09)** → `notes/compact_v4_multiseed_protocol_confirmation.md` + `notes/reproducibility_cpu_determinism.md` + `records/claims/claim-929b3e29.yaml` + `records/decisions/decision-d5bf08b3.yaml`：primary benchmark protocol = train→valid selection→frozen checkpoint→test（**无 refit**；refit 降级为 secondary robustness）；seeds 0–3 串行 bit-verified：v4-hinge vs v2 mean paired Δtest **+0.0092（4/4 seeds）→ GO**（非 Strong GO；A 组 sign flip caveat）；下一步唯一推荐 **post-v4 residual audit**
88. ~~post-v4 residual audit (09-09)~~ → `notes/post_v4_residual_audit.md`：**NO CLEAR SECONDARY SIGNAL**（所有结构探针 < 0.003 gate；残差只是 z_SA/稀有度异方差 difficulty，无结构 v5 通道）
89. ~~OOF difficulty / heteroscedasticity confirmation (09-10)~~ → `notes/oof_difficulty_heteroscedasticity_audit.md` + `records/claims/claim-oof-difficulty-confirmed-20260910.yaml` + `records/decisions/decision-oof-difficulty-heteroscedasticity-go-20260910.yaml`：official-train K=5 OOF（frozen v4-hinge，seeds 0+1，valid/test 从未加载）= **difficulty GO 0.374 (5/5) + epistemic GO 0.393 (5/5)**；rarity 阶梯 0.133→0.417 MAE；2-seed ensemble −8.3%；结论：残差是**条件方差**，不是可修正偏差
90. ~~compact-v5 uncertainty-aware multi-quantile regression (09-10)~~ → `notes/compact_v5_multi_quantile_regression.md` + `records/claims/claim-compact-v5-mq-point-nogo-20260910.yaml` + `records/claims/claim-compact-v5-width-internalizes-difficulty-20260910.yaml` + `records/decisions/decision-compact-v5-multi-quantile-nogo-20260910.yaml`：**objective-only，representation 完全冻结**；非交叉 q10/q50/q90 head（+66 params）+ `2*pinball0.5 + λq*(pinball0.1+pinball0.9)`；对照 none/median-only **bit-identical** 于 v4/L1；λ∈{0.10,0.25,0.50} 预注册。Stage-1 seed0 = **Case A**（valid q50 +0.0114；width-|err| 0.254）→ 冻结 4-seed 确认仅 **Mild/Mild**（point +0.0037, 3/4；width 0.211, 4/4）→ 一次性 benchmark test **未 transfer**（0.139403 vs v4 0.136885，paired −0.0025，1/4 seeds）→ **NO-GO**。width 内化 OOF difficulty/epistemic 轴（0.376/0.408）但仅 Mild，**不足以开启 uncertainty-guided weighting**；compact-v4-hinge 仍是当前 candidate
92. ~~corrected token fragmentation & rarity audit (09-10)~~ → `notes/corrected_token_fragmentation_rarity_audit.md` + `records/claims/claim-corrected-token-fragmentation-20260910.yaml` + `records/decisions/decision-corrected-token-fragmentation-mixed-20260910.yaml`：**frozen paired historical-vs-corrected v4（train/valid only；test 从未加载），无训练**。corrected tokenizer 确实大规模切碎 exact vocabulary（r2 6,784→15,218, 2,363 split, mean multiplicity 2.24/max 104；parent 31→512, mean 16.5）并造成 1,065 newly-rare / 396 newly-OOV validation occurrence（631/1000 分子受影响）。**但 paired molecule degradation 并不由 fragmentation 定向解释**：所有 unconditional molecule-level Spearman |ρ| ≤ 0.06（frequency loss −0.025、newly-rare −0.023、newly-OOV −0.056，1–2/4 seeds 同向）；退化由 **baseline-difficulty redistribution** 主导（Spearman(baseline error, degradation) = −0.352，difficulty quintile 单调 +0.031→−0.040，4/4 seeds）——corrected 在 easy bulk 变差、在 hard tail 变好。G1/G2/G3/G4 = +0.0016/+0.0042/+0.0141/−0.0083（G4 反向；difficulty-matched G3 +0.0143、G4 −0.0089）。conditional-on-difficulty 后仅剩很弱、非特异的新增稀有信号（partial ρ≈0.10；already-rare/hist-OOV 同样方向）；rarity-difficulty 关系仅微增（0.305→0.322 / 0.349→0.372）；cross-seed disagreement 微升（+0.0036）但与任何 fragmentation metric 无关（|ρ|≤0.07）而仅与 degradation 相关（+0.31）；HybridEmbedding tier reallocation 16.6% occurrence downgrade 但与 frequency loss 共线（ρ=0.52）无独立信号。**Decision（Case D — MIXED）**：不开启 coarse-shared + exact-residual（也不开 coarse-to-fine / shared-low-rank）；下一步唯一推荐是诊断 baseline-difficulty redistribution（仅诊断，不训练）。simple frequency-fragmentation / statistical-sharing-loss 解释未达 gate。
91. ~~compositional patch sharing oracle (09-10)~~ → `notes/compositional_patch_sharing_oracle.md` + `records/claims/claim-compositional-patch-sharing-nogo-20260910.yaml` + `records/decisions/decision-compositional-patch-sharing-nogo-20260910.yaml`：**frozen 表征 oracle / 干预，不训练**；train-only patch bank + 146D 结构近邻 + 分层 donor 约束 + 5 个预注册变体（NN1/KNN8/Blend50/FrequentMean/RandomDonor）。baseline **4/4 seeds bit-exact**，unaffected molecules **bit-identical**。**NO-GO（Case E）**：所有变体 overall ΔMAE 为负（NN1 −0.0120、KNN8 −0.0109、Blend50 −0.0043、FrequentMean −0.0099、RandomDonor −0.0109；0/4 seeds 正向），危害集中在 rare/OOV affected 分子（rare-seen Δ −0.010..−0.025；OOV −0.012..−0.025），结构 KNN8 **不优于** frequent-mean/random-donor 对照，donor 距离无收益梯度。**附带审计发现**：frozen typed certificate（pynauty.certificate）**非单射**——730/6784 token 混合了不同 root atom（pynauty 的完整不变量还需 color-cell sizes）；记为独立正确性修复项，不作为 sharing 证据
93. ~~fast baseline-difficulty / typed-refinement redistribution audit (09-10)~~ → `notes/baseline_difficulty_typed_refinement_redistribution_audit.md` + `records/claims/claim-baseline-difficulty-typed-refinement-20260910.yaml` + `records/decisions/decision-baseline-difficulty-typed-refinement-20260910.yaml`：**快速决策审计（复用 frozen paired v4 predictions + official-train targets + historical v4 OOF residuals；无训练、test 从未加载）**，只做 H1/H2/H3 三测。**H1**：退化锁死于 historical difficulty（quintile +0.0312/+0.0180/+0.0116/+0.0020/−0.0397；per-seed Spearman(baseline, degradation) −0.218/−0.167/−0.268/−0.172，4/4 negative），且为**中心(bias)主导**——easy Δcentre_MAE +0.0213 > Δspread_MAE +0.0125，4-seed ensemble 在 Q1 自身也变差（0.0245→0.0489），easy ensemble 退化是个体退化的 ~87%（非 ~0）→ **不是方差/正则化损失**；movement 系统性（Q1 harmful 65.5% = wrong-direction 36.9% + overshoot 28.6%，magnitude-only 4.8%）。**H2**：构造 train-only typed-refinement informativeness（historical token 的 corrected children 的 support-weighted train target / historical-OOF-residual dispersion；2,363 split parent 中 707 eligible），与 corrected gain **无独立关系**（ensemble ρ −0.019；partial given difficulty −0.036；7 个 metric 变体 partial ∈[−0.069,−0.025]，0–2/4 seeds）——null 不是 metric 选择。**H3**：2×2 由 difficulty 而非 informativeness 排序（hard+low +0.0252 > hard+high +0.0129；easy+high −0.0268；interaction −0.0080）——high informativeness 反而**弱反预测**。**Decision（Case D — UNEXPLAINED REDISTRIBUTION）**：A 因 ensemble 自身退化而失败（easy ensemble 退化 = 个体 ~87%），B/C 因 informativeness 完全无信号而失败。**NO architecture GO**：关闭 tokenizer-derived architecture 分支（不做 coarse-to-fine / coarse prior / adaptive gate）；仅当出现预注册的、graph-side 可观测的 gate，或更换非 exact-identity 的 tokenizer 变更时才重启。
94. ~~compact-v6 topology–attribute factorization (09-10)~~ → `notes/compact_v6_topology_attribute_factorization.md` + `records/claims/claim-compact-v6-topology-attribute-factorization-nogo-20260910.yaml` + `records/decisions/decision-compact-v6-topology-attribute-factorization-nogo-20260910.yaml`：**唯一变量 = patch 表示**（historical v4-hinge path 逐位冻结 + 新增 shared、permutation-invariant、exact-token-free 的 `(type, coarse-rooted-automorphism-orbit role)` attribute encoder，零初始化 8D `e_attribute` 追加到 patch encoder 输入，+1,470 params → 101,083）。四模型矩阵：v4 / count-control / factorized-role / capacity-control（后三者等参数）。**seed-0 曾 Case A**（valid 0.156421 vs 0.170066，Δ +0.013645；factorized > capacity +0.0062 > count +0.0145），**但 multi-seed（seeds 0–2）不复现**：paired Δ +0.013645/−0.017017/+0.002487（mean **−0.000295**，median +0.002487，**2/3**）< 0.003 gate。**seed-0 收益是 hard-tail-for-bulk trade**：historical quintile Q1/Q2 combined **−0.017265**（bulk gate ≤+0.002 FAIL），Q5 +0.091709 —— 与 corrected tokenizer 同一 redistribution 指纹。**机制**：冻结 seed-0 模型对 attribute-type / role-association **within-patch shuffle 精确不变**（clean==shuffled MAE 0.156421，per-molecule max|Δ|=0.0；`e_attribute` norm std **2.3e-10** → branch collapse），count-control 两 shuffle 亦不变；count-only 无增益（seed-0 −0.000865，mean −0.002243 1/3），factorized−count mean +0.001948（1/3）< 机制 gate。唯一一致信号 factorized > capacity 3/3（mean +0.006180），但分支未被使用 → 属训练轨线而非 attribute 使用。**Decision（Case D at benchmark gate）**：关闭 topology–attribute factorization；不增大 attr dim / 不加深 MLP / 不加 role features / 不开 dictionary-KSVD。**revisit_if**：先诊断并修复零初始化导致的 branch collapse（step-0 上游死梯度 + weight decay 压权重），再以全新 seed sweep + 预注册 graph-side gate 重验。
95. ~~compact-v6 attribute-branch viability repair (09-10)~~ → `notes/compact_v6_attribute_branch_viability_repair.md` + `records/claims/claim-compact-v6-attribute-branch-collapse-and-viability-20260910.yaml` + `records/decisions/decision-compact-v6-attribute-branch-viability-nogo-20260910.yaml`：**先诊断 original collapse，再做唯一最小修复**。梯度 trace 证实原 compact-v6 分支 step-0 处 `e_attribute≡0`（零初始化 final fusion）→ 全部 upstream attribute 参数（embedding/MLP）与 patch_encoder 新增 8 列拿到**精确零梯度**，只有 `fusion.2` 能学；collapse 实际发生在 encoder MLP（frozen checkpoint `h_atom` occurrence std 5.2e-6 vs 输入 0.212，`e_attribute` 跨 patch std 7.6e-12 → 精确 shuffle-invariant）。optimizer 审计干净（14/14）、raw role 有信息（std≈0.30）、adversarial placement pair 有效。**最小修复**：只把 final fusion projection 由 `zeros` 改为 `normal(std=0.01)`（`attribute_fusion_init=small_normal`），其余逐位不动。**100-step viability gate 全过**（V1 patch_std 3.87e-4；V2 三层梯度活；V3 role-shuffle max 3.70e-5；V4 attr-zero max 1.38e-2）→ 分支确可训练/可影响预测。**fresh seed0（test blocked）**：valid 0.170209 vs v4 0.170066，Δ **−0.000143**（< +0.003 gate），且全量训练后分支**再次退化**（patch_std 3.62e-7；role/type shuffle mean≈1.2e-7），Q1/Q2 bulk degradation **+0.023831**（>0.002 FAIL，收益全在 Q5 +0.0511）——同一 redistribution 指纹。**Decision（Outcome 2）**：BRANCH REPAIRED, ARCHITECTURE NO-GO → **永久关闭 topology–attribute factorization**，STOP AT SEED0，不调 scale/dim/role/optimizer。8 tests 全绿。
