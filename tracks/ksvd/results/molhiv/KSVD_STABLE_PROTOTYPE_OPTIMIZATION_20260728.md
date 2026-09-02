# KSVD 路线：稳定真实原型优化与全量冻结方案（2026-07-28）

## 1. 目标与数据边界

本轮针对 8k official-split terminal 中暴露出的主要问题：随机真实原型 MIL 的单 seed 波动很大，而 KSVD direction 仍弱于真实 observed patch identity。

开发阶段严格限制为：

- 8,000-graph stratified development subset；
- 只使用其中 6,400 个 official-train 图；
- 3 个 official-train-only Bemis--Murcko scaffold outer folds；
- official valid/test 在本轮开发中均不编码、不评估；
- downstream 固定 30 epochs，不按 held-out AUC 选 epoch。

## 2. nested task-matched real prototype

### 2.1 泄漏修复与稳定化

实现文件：`code/run_molhiv_stable_taskmatched_prototypes.py`。

在正式运行前修复了一个重要的 nested-selection 问题：如果候选 patch 来自 inner-valid 图，则该图对候选自身有精确 cosine=1，可能夸大候选的 OOF 重要性。修复后的方案为：

1. 候选库按图均衡采样：从 256 个互不重复的 outer-fit 图各取一个真实 node patch；
2. 候选库只由固定 `prototype_seed` 决定，与 downstream model seed 独立；
3. 每个候选仅在其来源图属于 inner-train 的 inner folds 上计分；3-fold 下每个候选恰好有 2 个 source-clean OOF scores；
4. inner train/valid scaffold 严格不重叠；
5. selector 使用三类 graph occurrence features：maximum positive cosine、top-node mean、all-node mean；
6. 用 inner-valid balanced log-loss ablation rank 聚合候选重要性，再用 MMR 保持多样性；
7. label-shuffled selector 使用完全相同的候选库和 inner partitions。

cache audit 同时确认：

- PCA/KSVD cache 的 dictionary-fit index hash 等于当前 outer-fit fold；
- official valid/test latent rows 全为零；
- latent cache 来自对应 token cache；
- inner scaffold overlap 为零。

### 2.2 Seed-0 gate 结果

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| fixed random real prototype | 0.712212 | 0.670920 | 0.762144 | 0.715092 |
| farthest real prototype | 0.673625 | 0.609613 | 0.771672 | 0.684970 |
| nested task-aware | 0.728902 | 0.683907 | 0.742279 | 0.718363 |
| nested label-shuffled | 0.724287 | 0.650477 | 0.770282 | 0.715016 |
| KSVD direction | 0.722168 | 0.655089 | 0.788815 | 0.722024 |

预注册 promotion gate：

- nested task-aware 相对 fixed random 的 3-fold mean gain 至少 `+0.005`；
- 至少赢 2/3 folds；
- nested task-aware mean 必须高于 label-shuffled mean。

实际结果：

- fold deltas：`+0.016690, +0.012987, -0.019865`；
- mean gain：`+0.003271`；
- fold wins：`2/3`；
- task-aware 相对 shuffled mean gain：`+0.003347`。

**结论：第一条门槛失败，因此 nested task-aware 分支不补跑 seeds 1/2，也不进入全量 official valid/test。** 它有弱任务信号，但不足以证明监督 prototype selection 能稳定解决问题。

## 3. Label-free 稳定化：prototype-bank ensemble

在 task-aware gate 失败后，转向不使用标签选择 prototype identity 的稳定化。比较两种同为 3-model 的 probability ensemble：

1. **model-seed ensemble**：固定 prototype bank `20260728`，downstream seeds `0/1/2`；
2. **prototype-bank ensemble**：固定 downstream seed `0`，prototype-bank seeds `20260728/20260729/20260730`。

每个 bank 都按图均衡地从 256 个不同 official-train development 图各抽一个真实 patch，再固定抽取 32 个 prototype。ensemble 为 sigmoid probability 的算术平均。

| Scheme | Fold 0 | Fold 1 | Fold 2 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| fixed bank 20260728 / seed 0 | 0.712212 | 0.670920 | 0.762144 | 0.715092 | 0.037298 |
| 3 model seeds, one bank | 0.743674 | 0.666471 | 0.764451 | 0.724865 | 0.042153 |
| **3 prototype banks, model seed 0** | **0.748232** | **0.677951** | **0.758473** | **0.728219** | **0.035789** |
| all five available models | 0.749985 | 0.670906 | 0.761360 | 0.727417 | 0.040228 |

相对固定 bank/seed-0：

- model-seed ensemble：mean `+0.009773`，赢 2/3 folds；
- prototype-bank ensemble：mean `+0.013127`，赢 2/3 folds；
- five-model combined：mean `+0.012325`，只赢 1/3 folds。

因此冻结 **3 个 label-free graph-balanced prototype banks + 固定 downstream seed 0**。该结果也说明当前主要方差来源确实包含 prototype identity；直接平均不同真实局部词表，比只平均 downstream initialization 更有效。

结构化汇总：`stable_prototype_ensemble_scaffold_summary.json`。

## 4. 冻结的全量 official protocol

完整数据目标为全部 41,127 graphs：

1. radius-2 permutation-invariant raw patch descriptor；
2. PCA64 与 KSVD32/T3 只在全部 official train 上拟合，保持开发阶段相同的 6,000-patch reservoir 配置；
3. 用冻结 metric 编码 official train/valid/test；
4. 三个 prototype seeds 固定为 `20260728/20260729/20260730`；
5. 每个 bank 从 256 个不同 official-train 图各取一个真实 node patch，再抽 32 prototypes；
6. 三个模型均使用 downstream seed `0`、hidden 64、sparsity 3、batch 128、固定 30 epochs；
7. 每个模型训练完成后，official valid/test 各评估一次；
8. 最终 valid/test 结果为三个 sigmoid probability 的等权算术平均；
9. 不根据 full official valid/test 修改 bank seeds、权重、epoch 或架构。

需要明确：仓库此前实验以及 8k terminal 已经查看过 official test 的一部分，因此该全量结果是 **frozen controlled full-data evaluation**，不能称为 untouched test。

## 5. 全量 official valid/test 结果

冻结的 3-bank GNN-free real-prototype ensemble 已在完整 41,127 graphs 上完成：

| Family | Official valid AUC | Official test AUC |
|---|---:|---:|
| real-prototype ensemble | **0.802488** | 0.767771 |
| fixed-epoch original-node GINE ensemble | 0.781385 | **0.777993** |
| prototype − GINE | **+0.021103** | **−0.010222** |

真实原型三个单 bank test AUC 为 `0.724522/0.749734/0.747432`，ensemble 达到
`0.767771`，说明 prototype-bank averaging 的稳定化在全量上成立。但 valid 优势没有
迁移为 test 优势，test 上 GINE ensemble 高 `0.010222`，因此不能声称真实原型超过 GINE。

更重要的是，最终分支使用 PCA64 latent 中的 observed real patches，而不是 KSVD atom
作为 prototype identity；当前结果支持的是 patch/prototype view，而不是 KSVD-specific
gain。完整审计和逐 seed 数值见：
`KSVD_STABLE_REALPROTOTYPE_FULL_OFFICIAL_20260728.md`。
