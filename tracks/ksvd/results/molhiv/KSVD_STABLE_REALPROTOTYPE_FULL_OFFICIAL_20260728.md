# GNN-free 稳定真实原型：全量 MolHIV official valid/test（2026-07-28）

## 1. 本次回答的问题

本次将 8,000-graph official-train scaffold 开发阶段冻结的方案，扩展到完整
`ogbg-molhiv` 41,127 graphs，并与匹配的固定 epoch 原始节点 GINE 做对照。

冻结的真实原型方案为：

- raw radius-2 patch descriptor（最大 8 个节点）；
- PCA64 metric 只在全部 official train 上拟合；
- 从 official-train 中构造 graph-balanced 256-patch candidate bank；
- 每个 bank 无标签抽取 32 个 observed real patches；
- prototype seeds 固定为 `20260728/20260729/20260730`；
- downstream seed 固定为 `0`；
- fixed 30 epochs，不根据 official valid/test 选择 epoch；
- 三个模型的 sigmoid probability 等权平均。

需要准确命名：最终分类器使用的是 **PCA64 空间中的 observed real-patch
prototypes + GNN-free MIL readout**。KSVD cache 用于同一 patch pipeline 的拟合与审计，
但最终 prototype identity 不是 KSVD atom/direction。因此它证明的是“真实局部 patch
词表”的价值，而不是 KSVD-specific gain。

## 2. 数据与审计

| Item | Value |
|---|---:|
| all graphs | 41,127 |
| official train | 32,901（positive 1,232） |
| official valid | 4,113（positive 81） |
| official test | 4,113（positive 130） |
| official-train index SHA256 | `4e77289653e41be5d2267f54d9f33fda4b70fd622c19b93a282db710a9318b0a` |
| latent SHA256 | `16a4450165e61930e155292e737f36a6d36d5f9d64646242cf3ebe212a34b477` |
| encoded nodes | 1,049,163 |

核验结果：

- 三个真实原型成员之间的 train/valid/test indices 与 labels 完全一致；
- 三个 GINE 成员之间完全一致；
- 真实原型与 GINE 两个 family 之间也完全一致；
- split counts 与 positive counts 均符合 official split；
- sklearn ROC-AUC 与 OGB `Evaluator('ogbg-molhiv')` 逐项一致；
- 所有 probabilities 有限且位于 `[0,1]`。

## 3. 全量结果

### 3.1 GNN-free observed real-prototype MIL

| Prototype bank seed | Params | Official valid AUC | Official test AUC |
|---:|---:|---:|---:|
| 20260728 | 46,658 | 0.764930 | 0.724522 |
| 20260729 | 46,658 | 0.795920 | 0.749734 |
| 20260730 | 46,658 | 0.772992 | 0.747432 |
| **3-bank probability ensemble** | 3 × 46,658 | **0.802488** | **0.767771** |

三个单 bank 的均值与 sample std：

- valid：`0.777947 ± 0.016078`；
- test：`0.740563 ± 0.013940`。

ensemble 相对单成员均值：

- valid：`+0.024540`；
- test：`+0.027208`。

这进一步确认 prototype identity 是重要方差源，而且 label-free vocabulary ensemble
确实能有效压低这种不稳定性。

### 3.2 匹配的 fixed-epoch original-node GINE

GINE 配置：3 layers、hidden 64、mean-pooled gated JK、37,382 parameters、fixed 30
epochs、model seeds `0/1/2`，三个概率等权平均；不使用 KSVD/patch features。

| Model seed | Params | Official valid AUC | Official test AUC |
|---:|---:|---:|---:|
| 0 | 37,382 | 0.770074 | 0.743601 |
| 1 | 37,382 | 0.771571 | 0.752956 |
| 2 | 37,382 | 0.687237 | 0.762381 |
| **3-seed probability ensemble** | 3 × 37,382 | **0.781385** | **0.777993** |

### 3.3 Frozen ensemble-to-ensemble comparison

| Family | Official valid AUC | Official test AUC |
|---|---:|---:|
| GNN-free real-prototype ensemble | **0.802488** | 0.767771 |
| original-node GINE ensemble | 0.781385 | **0.777993** |
| prototype − GINE | **+0.021103** | **−0.010222** |

## 4. 科学结论

1. **真实 patch identity 是有效信号。** 不进行 message passing 的 prototype MIL 在完整
   test 上达到 `0.7678`，与三 seed GINE ensemble 相差约 1.02 AUC points，而 valid 上反而
   高 2.11 points。这已足以说明局部 patch vocabulary 不是无效旁支。

2. **不能声称真实原型超过 GINE。** valid 优势没有迁移到 test；冻结 test 上 GINE
   ensemble 更高。因此当前最稳妥结论是：GNN-free real-prototype route 具有竞争力，但尚未
   稳定超过 GINE。

3. **prototype-bank ensemble 是成功的稳定化，而非小幅偶然修补。** 它在 valid/test
   上都明显超过三个单 bank 的均值，并且全量结果与 8k scaffold development 中“prototype
   identity 是主要方差源”的判断一致。

4. **当前证据不是 KSVD-specific。** nested task-aware prototype selection 未通过开发
   gate；KSVD direction 也未稳定超过 observed real prototypes。最终最强的 GNN-free 分支
   实际依赖 PCA metric + observed patches。因此论文叙事不应写成“KSVD atom 已优于 GNN”，
   而应写成“dictionary/prototype view 可以形成有竞争力的非消息传递分支；KSVD 目前主要承担
   metric、coverage、reconstruction/residual audit”。

5. **GINE 整合仍合理，但不应继续堆叠。** GINE 是必要的结构对照，也可能与 prototype
   branch 互补；然而下一步应验证互补性是否超出普通 ensemble variance reduction，而不是继续
   增加 GINE gate、layer 或复杂 fusion。

## 5. 下一轮建议（回到 8k 开发，避免再用 test 调参）

优先级建议：

1. **共享局部编码、双视角 late fusion**：单独训练 GINE 与 frozen real-prototype branch，
   只在 8k official-train scaffold folds 上冻结一个简单 fusion protocol；必须与
   GINE+GINE 不同 seed ensemble 做 matched control。
2. **prototype occurrence calibration**：保持 prototype identity label-free，只学习
   graph-level occurrence 的低容量校准（max/top-k/coverage/entropy），避免用 graph label
   直接选择 atom identity。
3. **跨 bank consensus features**：不先平均最终概率，而是在 node/graph 层提取多个 bank
   的一致性、覆盖率与 disagreement，测试它们是否提供稳定 residual signal。
4. **KSVD-specific matched controls**：在同一真实 prototype bank 下加入 KSVD reconstruction
   residual/novelty，但必须同时比较 PCA residual、random dictionary residual、shuffled code；
   只有 repeated scaffold folds 上稳定胜出才提升 KSVD 为核心贡献。
5. **停止条件**：8k repeated scaffold mean gain 至少 `+0.005`、赢至少 2/3 folds，并超过
   matched shuffled/random control，才重新进入下一次全量 controlled valid/test。

## 6. 结果文件

- `stable_realproto_full_official_bank20260728_seed0.json`
- `stable_realproto_full_official_bank20260729_seed0.json`
- `stable_realproto_full_official_bank20260730_seed0.json`
- `fixed_epoch30_gine_full_official_seed0.json`
- `fixed_epoch30_gine_full_official_seed1.json`
- `fixed_epoch30_gine_full_official_seed2.json`
- `stable_realproto_full_official_ensemble_summary.json`
- `stable_realproto_full_official_ensemble_summary.log`

由于仓库此前已经查看过 official test，本报告是 **controlled frozen full-data
evaluation**，不是 untouched test；这些 test 数值不能反馈用于本轮之后的结构或权重选择。
