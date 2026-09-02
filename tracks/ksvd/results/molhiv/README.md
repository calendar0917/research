# MolHIV results

Protocol: OGB `ogbg-molhiv` official scaffold split.

## Real-patch atom--occurrence bipartite：full official evaluation（2026-07-28）

8k official-train-only 的 3 scaffold folds × 3 seeds 中，farthest real-patch bipartite
相对 matched node MIL 平均提升 `+0.015968`、6/9 wins，因而冻结后进入全图评估。
三 seed probability ensemble 在 official valid/test 上分别为 `0.793553/0.737741`，
matched node MIL 为 `0.785769/0.734443`，对应增益 `+0.007783/+0.003299`。这说明
atom--occurrence incidence 是真实但较弱的信号；不过绝对 test 结果仍明显低于旧
real-prototype ensemble `0.767771` 和 GINE ensemble `0.777993`，因此该实现不晋级为主模型。
详见 `REALPATCH_ATOM_OCCURRENCE_BIPARTITE_FULL_OFFICIAL_20260728.md`。


## Atom--occurrence 二部模型首轮筛选（2026-07-28）

在 8k official-train-only、3 scaffold folds、seed 0 上，无 GINE 的两轮
`atom -> occurrence -> atom` 传播相对 matched node MIL 有正增益：farthest real-patch
`+0.016760`（2/3 wins），KSVD `+0.024242`（2/3 wins）。但 KSVD bipartite 的均值
`0.729134` 仍低于 farthest real-patch bipartite 的 `0.736263`，差 `-0.007129`，只胜
1/3 folds。KSVD 胜 PCA、random direction、shuffled-ID 和 no-ID，说明其方向和稳定身份
有信号，但没有跨过 matched real-prototype gate。按预注册规则，不扩展完整 controls 到更多
seeds，也不运行 official valid/test。详见 `ATOM_OCCURRENCE_BIPARTITE_SCREEN_20260728.md`。



## Real-prototype covariance pair-PCA：full official-train 三折通过（2026-07-28）

固定的两组真实局部结构词汇（`farthest` 与 `scaffold_facility`，各 32 个）先统计距离 1/2 内的具体结构组合，再用只在 outer-fit 真实组合数据上拟合、完全不看标签的 covariance PCA 压到 144 维，作为 frozen compact relation base 的有界残差。

| 指标 | fold 0 | fold 1 | fold 2 | mean |
|---|---:|---:|---:|---:|
| compact base AUC | 0.788469 | 0.743996 | 0.806008 | 0.779491 |
| covariance pair-PCA AUC | 0.792660 | 0.750703 | 0.804660 | 0.782674 |
| gain | +0.004191 | +0.006706 | -0.001348 | **+0.003183** |
| real − assignment-shuffled | +0.004712 | +0.001832 | +0.000198 | **+0.002248** |

预先固定的四项 full-development 晋级条件全部通过：平均增益至少 `+0.002`、至少 2/3 folds 胜 base、real mean 胜 assignment-shuffled、至少 2/3 folds 胜 matched control。correlation PCA 仅 `+0.000863` 且只胜 1/3 folds，继续判定失败。

当前解释是：有效弱信号来自训练分子中反复共同变化、变化幅度较大的具体局部结构组合，而不是任意随机投影，也不是把稀有组合强制放大。配置已冻结到 `label_free_pair_pca_official_valid_freeze_v1.json`；冻结时 official valid/test 编码与评估次数均为 0。详见 `REALPROTOTYPE_LABEL_FREE_PAIR_PCA_20260728.md` 和 `label_free_pair_pca_summary_scaffold3.json`。


## GNN-free 稳定真实原型 full official evaluation（2026-07-28）

8k official-train scaffold development 冻结的三 prototype-bank ensemble 已扩展到完整
41,127 graphs。所有模型固定 30 epochs；prototype branch 固定 downstream seed 0、改变
三个 label-free graph-balanced real-patch banks；GINE baseline 固定架构并平均 seeds 0/1/2。

| family | official-valid ensemble AUC | official-test ensemble AUC |
|---|---:|---:|
| GNN-free observed real-prototype MIL | **0.802488** | 0.767771 |
| fixed-epoch original-node GINE | 0.781385 | **0.777993** |

prototype ensemble 相对 GINE 为 valid `+0.021103`、test `-0.010222`。这表明真实局部 patch
vocabulary 是有竞争力且互补候选，但不能声称超过 GINE；而且最终 prototype 是 PCA64
空间中的 observed patch，不是 KSVD atom，因此也不能称为 KSVD-specific gain。详见
`KSVD_STABLE_REALPROTOTYPE_FULL_OFFICIAL_20260728.md`。本结果仍是 controlled frozen
evaluation，不是 untouched test。


## KSVD-JK 双分支 controlled terminal test（2026-07-26）

在 inner-only 冻结、三 seed official-valid 完成且不再调参后，按固定协议运行：

```text
p = 0.5 * p(GINE-JK) + 0.5 * p(KSVD graph-residual + GINE-JK)
model seeds = 0 / 1 / 2
branch weight = 0.5 / 0.5（未在 official-valid/test 搜索）
```

由于仓库更早的实验已经查看过 official test，本节只能称为 **controlled terminal
evaluation**，不能称为 untouched test；下面的结果不得反馈用于继续选择结构、权重、seed
或 epoch policy。

| family | params | official-valid mean ± std | official-test mean ± std |
|---|---:|---:|---:|
| GINE-JK | 37,382 | 0.792999 ± 0.032936 | 0.747494 ± 0.029477 |
| KSVD graph residual | 43,722 | 0.786122 ± 0.013747 | 0.752637 ± 0.009697 |
| fixed 0.5/0.5 ensemble | 81,104 | **0.801192 ± 0.009863** | **0.761911 ± 0.011764** |

逐 seed fixed ensemble test AUC：

```text
seed0  0.749433
seed1  0.763499
seed2  0.772800
```

结论必须收紧：

- ensemble 的确比两个单分支的三 seed test mean 都高，并保持较低方差；
- 但 official-valid 的 `0.8012` 没有迁移成 `~0.80` test，test mean 只有 `0.7619`；
- 更关键的是，post-hoc 描述性对照中，三个无序的“两 GINE-JK 不同 seed”概率 ensemble
  平均 AUC 为 `0.762023`，与 GINE+KSVD 的 `0.761911` 基本相同；因此当前证据不能证明
  增益来自 KSVD 特有结构，而不是普通的双模型方差压缩；
- 当前双分支还重复执行两个完整 GINE，虽然参数仍低于 CIN-small，但方法叙事和计算效率
  不够干净，不能把它作为“KSVD 已证明有效”的最终答案。

下一步应回到 **inner-only / 新外部 benchmark** 做 KSVD-specificity-first 验证：

1. 同架构、同容量比较 KSVD / PCA / random dictionary / graph-wise shuffled KSVD codes；
2. 改成一个共享 GINE-JK backbone + 一个可单独读出的 KSVD residual head，避免两次 GINE；
3. 优先测试直接 atom-additive 的 `max_v |z_vj|` 线性 head，并输出每个 dictionary atom
   的正负贡献、top-activating ego patches 和跨 split atom stability；
4. 只有 KSVD 在 repeated inner splits 上稳定超过全部 matched controls，才继续把 KSVD 作为核心贡献。

审计文件：

- `node_token_ksvd_jk_ensemble_terminal_test_freeze_v1.json`
- `node_token_ksvd_jk_ensemble_terminal_test_summary_v1.json`
- `node_token_terminal_test_v1_{ginejk,ksvd_graphres}_seed{0,1,2}.json`

## 2026-07-26 单 GINE + sparse-assignment additive readout

为验证 sparse assignment 是否能替代第二个 GINE 分支，新增了只执行一次
GINE-JK 的 atom-conditioned additive readout。它不让 KSVD 单独承担 message passing，
而是用 raw sparse code 的 `|z_vj|` 把同一个 GINE node state 分配到自学习字典 atom：

```text
H_j = sum_v |z_vj| h_v / sum_v |z_vj|
logit = logit_GINE + mean_active_j [tanh(a_j) * u^T H_j]
```

该版本仅增加 96 个参数，初始 atom weights 为零，因此初始预测严格等于 matched
GINE-JK；每个 atom 的贡献可以单独输出。n=8000、固定 inner split 1729、三个 model
seeds 的结果为：

| family | params | inner AUC mean ± sample std |
|---|---:|---:|
| GINE-JK | 37,382 | 0.775824 ± 0.025611 |
| KSVD graph residual | 43,722 | 0.779511 ± 0.017629 |
| KSVD atom-additive | 37,478 | **0.794032 ± 0.007236** |
| PCA atom-additive | 37,478 | 0.783339 ± 0.019610 |
| random-patch atom-additive | 37,478 | 0.776515 ± 0.010549 |

这个表说明方案在工程上可行，并且 KSVD 在固定 split 的 model-seed 平均上最好；但
它不能证明 sparse assignment 已经有第二个 GINE 分支的稳健能力。固定 model seed0、改变
inner split seeds `1729/2718/31415` 后：

| family | per-split AUC | split mean ± sample std |
|---|---|---:|
| GINE-JK | 0.793050 / 0.747595 / 0.700126 | **0.746924 ± 0.046465** |
| KSVD atom-additive | 0.792388 / 0.738636 / 0.689544 | 0.740190 ± 0.051440 |
| PCA atom-additive | 0.805766 / 0.707522 / 0.707552 | 0.740280 ± 0.056713 |

KSVD 在三个 split 上都略低于 matched GINE，且 KSVD/PCA 的 split mean 基本相同
（差约 `-0.00009`）。因此当前正确信息是“**basis-conditioned additive readout
可行**”，而不是“KSVD-specific gain 已证明”。

还测试了更强的 per-atom task direction：每个 atom 使用独立 `u_j`，增加 2,048
参数、总参数 39,430。三个 split AUC 为
`0.786316/0.717051/0.716811`，均值 `0.740059 ± 0.040059`；仍比 GINE split
mean 低 `0.006864`，所以继续增加 readout 容量不是当前突破口，该版本停止。

决策：不进入 official-valid/test，不把 sparse assignment 当作第二个 GINE 的等价替代；
保留 96-parameter 版本作为单-backbone、可分解、低成本 ablation。下一轮若继续，必须直接
优化 **KSVD assignment 的跨 split 稳定性/特异性**，而不是继续堆 readout 参数。

结构化汇总：`node_token_atom_additive_inner_only_summary.json`。本轮新增运行的
`official_valid_evaluations = 0`，`official_test_evaluations = 0`。

## 当前最终状态（2026-07-26）

localized KSVD node-token 路线已冻结，并完成 5-seed controlled terminal
official-test evaluation。

冻结配置：

```text
radius 2
KSVD D32 / OMP T3
GINE hidden64 / 3 layers
train-only zscore signed node tokens
all-layer zero-initialized scalar gated injection
```

配置 ID：

```text
molhiv-r2-d32-t3-h64l3-zscore-signed-alllayer-v1
```

### 五 seed 单模型结果

| family | trainable params | official-valid mean ± std | official-test mean ± std |
|---|---:|---:|---:|
| GINE h64/l3 | 37,380 | 0.769955 ± 0.032409 | 0.752695 ± 0.014685 |
| GINE h70/l3 parameter-matched | 43,404 | 0.789699 ± 0.020088 | **0.763280 ± 0.013142** |
| KSVD h64/l3 D32/T3 | 43,655 | **0.794569 ± 0.037439** | 0.751893 ± 0.015143 |

### 五 seed probability ensemble

| family | official-valid | official-test |
|---|---:|---:|
| GINE h64 | 0.798620 | 0.775208 |
| GINE h70 parameter-matched | 0.810635 | **0.777620** |
| KSVD h64 D32/T3 | **0.828606** | 0.770363 |

最终判断：

- KSVD 在 official-valid 上有明显结构信号，但没有在 official-test 上稳定超过 GINE；
- 当前结果未达到 CIN 文献约 0.81；
- 参数效率是真实优势：KSVD 只有 43,655 downstream trainable parameters；
- 加上固定 `848×32` dictionary 后，总 stored learned values 为 70,791；
- 官方 CIN-small / CIN 参数复核分别为 138,385 / 239,809；
- 更深、更宽、更大 dictionary 和 adaptive router 均已在 inner-only screen 中失败；
- 下一步不建议继续堆普通容量，应转向 cross-scaffold dictionary stability 和更稳健的多 inner-fold epoch selection。

## 协议 disclosure

仓库早期 feasibility 文件曾查看 official test，因此最终结果只能表述为：

> configuration-freeze 后的受控 terminal evaluation，而不是 untouched-test claim。

冻结后的 test 结果不得继续用于选择新模型，再把同一个 test 当作无偏最终评估。

## 最重要文件

| 文件 | 含义 |
|---|---|
| `KSVD_MOLHIV_FINAL_REPORT.md` | 最终详细报告、完整结果、参数审计、失败路线与下一步 |
| `node_token_frozen_config_v1.json` | 冻结配置 manifest |
| `frozen_test_5seeds_summary.json` | GINE-h64 / GINE-h70 / KSVD 的 5-seed valid/test/ensemble 汇总 |
| `cin_parameter_audit.json` | 基于 CIN 官方实现的参数量复核 |
| `frozen_test_{ksvd,gine_h64,gine_h70}_seed*.json` | 每 seed labels、probabilities、epoch 与单次评估审计 |
| `node_tokens_full_a32_t3_frozen_test.json` | 冻结字典 test encoding 审计与 hash |
| `node_token_capacity_inner_only_screen_summary.json` | width/depth 参数 scaling screen |
| `node_token_dictionary_scaling_inner_only_screen_summary.json` | D48/T4 dictionary scaling screen |
| `node_token_adaptive_router_inner_only_screen_summary.json` | node-adaptive router screen |
| `node_token_multiscale_inner_only_screen_summary.json` | radius 1+2 multiscale screen |
| `ring_cell_inner_only_screen_summary.json` | 显式 ring ablation screen |

## 代码入口

- `code/build_molhiv_node_tokens.py`：训练字典并构建 localized token cache；
- `code/run_molhiv_node_tokens_inner.py`：严格 inner epoch selection、full-train retrain、valid/test evaluation；
- `code/encode_molhiv_node_tokens_test.py`：冻结后只编码 official-test，绝不更新 dictionary；
- `code/summarize_molhiv_frozen_test.py`：5-seed 与 probability ensemble 汇总；
- `code/molhiv_node_tokens.py`：patch / dictionary / sparse-code 公共实现。

## 历史路线文件

- `KSVD_FINAL_ROUTES.md`：此前 standalone graph-level KSVD 总结；
- `KSVD_FEASIBILITY_ROUTE.md`：早期 feasibility、matched controls 和失败路线；
- `ksvd_readout_full_5seeds_summary.json`：standalone KSVD readout；
- `ksvd_residual_inner_n8000_5seeds_summary.json`：graph-level GINE residual。

这些历史文件解释了方法如何从 graph-level KSVD 演化到当前 localized node-token，但当前最终结论以 `KSVD_MOLHIV_FINAL_REPORT.md` 和 `frozen_test_5seeds_summary.json` 为准。

### 2026-07-26 runtime-aware optimization screen

- `node_token_runtime_aware_optimization_inner_only_screen_summary.json`: fit-only graph-frequency shrinkage、z-score clipping、parameter EMA，以及固定 D32/T3 下增加 KSVD 迭代次数的 inner-only 快筛。所有新候选均未触碰 official-valid/test；无候选通过 `+0.003 mean、至少 2/3 seed wins、方差不恶化` 的 full-inner 晋级门槛。

### 2026-07-26 graph-balanced KSVD dictionary screen

- `node_token_graph_balanced_inner_only_screen_summary.json`: 每个训练分子最多贡献 8 个字典学习 patch 的 graph-balanced KSVD。n=8000 快筛提升 `+0.00334` 且 2/3 seeds 胜出，但 full-inner 下降 `-0.01007`；把 reservoir 扩至 12000 patches 的 seed-0 refinement 也失败。所有新运行均未评估 official-valid/test。

### 2026-07-26 learned patch-domain KSVD screen

- `node_token_patch_domain_inner_only_screen_summary.json`: train-only unsupervised patch clustering was used in two KSVD-native variants with fixed D32/T3 capacity: cluster-balanced global KSVD and hard-routed `4×8` mixture-of-KSVD. Seed-0 inner AUC fell by `-0.02005` and `-0.02195`, respectively, so both were stopped before additional seeds/full-inner. No official-valid or official-test evaluation was run.

### 2026-07-26 consensus/stability KSVD screen

- `node_token_consensus_stability_inner_only_screen_summary.json`: one full-pool KSVD plus two train-only 80% subsample dictionaries were atom-aligned by Hungarian matching. Atom averaging, aligned-code averaging with top-3 pruning, a conservative `0.75 anchor + 0.25 consensus` shrinkage, and reconstruction-confidence weighting achieved seed-0 inner deltas of `-0.00643`, `-0.00183`, `-0.01136`, and `-0.00839`. None advanced to more seeds or full-inner; official-valid/test evaluations remained zero.

### 2026-07-26 KSVD atom-interaction screen

- `node_token_bilinear_interaction_inner_only_screen_summary.json`: 在保留原普通 KSVD token 注入的同时，加入独立 zero-init gate 的 rank-8 低秩二阶 atom interaction 分支；只增加 1,091 个参数，总参数量 44,746。node-local、恢复 exact OMP support 的 masked node-local、以及沿真实分子边聚合的 edge-neighbor 版本，在 n=8000 seed-0 inner-only 上分别得到 `0.755171`（相对冻结基线 `-0.010672`）、`0.738907`（`-0.026936`）和 `0.754780`（`-0.011063`）。三者均未通过正信号门槛，因此未补 seeds/full-inner；official-valid/test evaluations 均为 0。

### 2026-07-26 dynamic / task-aware / multi-inner-split screen

- `node_token_dynamic_taskaware_multifold_inner_only_summary.json`: dynamic KSVD composition 用每层当前 GINE hidden state 与 KSVD code 做 rank-8 低秩交互，seed-0 inner AUC 为 `0.745130`（相对冻结基线 `-0.020713`）。严格 inner-train 的 label-reweighted KSVD dictionary 在 positive patch fraction `0.10/0.50` 下分别为 `0.739959`（`-0.025884`）和 `0.735390`（`-0.030453`）。冻结 KSVD 基线跨 inner split seeds `1729/2718/31415` 的 per-split best AUC 为 `0.765843/0.755712/0.718344`，最佳 epoch 为 `27/14/26`；minimum-mean-regret consensus epoch 为 17，说明单 inner split 的选择噪声不可忽略。没有候选通过正信号门槛，因此未补 model seeds/full-inner；official-valid/test evaluations 均为 0。

### 2026-07-26 KSVD supervised-unrolling / global-context screen

- `node_token_unrolling_context_inner_only_screen_summary.json`: 首先构建严格 selection cache，只对 official-train 的 162,150 个节点计算归一化 patch 的充分统计量 `yᵀD`；official-valid/test 行保持全零。KSVD 初始化的 1/2-step proximal refinement（scalar/per-atom threshold、独立 residual gate、code-space mix）在主 n=8000 seed-0 上均下降，最佳也只有 `0.754359`，因此停止 dictionary unrolling。
- 随后测试不依赖显式 ring/scaffold 的 **graph-global KSVD context**：把自学习 node-token embedding 在分子内聚合，再以独立 zero-init gates 回注每层。未限幅版本仅增加 3 个参数（43,655 → 43,658），跨 inner split 的稳定性改善，但 model-seed 均值下降。
- 将 context 有效 gate 固定缩放为 `0.25` 后，三个 split seeds `1729/2718/31415` 分别从 `0.765843/0.755712/0.718344` 提升到 `0.768068/0.759620/0.731752`，split mean `+0.006514`，std 从 `0.020426` 降至 `0.015516`；但三个 model seeds 的均值基本持平（`0.766254` vs 冻结参考 `0.766344`），full-inner seed-0 为 `0.816609`，低于冻结参考 `0.820131`。
- 因此该路线保留为 **KSVD-native 稳定性 ablation**，不晋级 official-valid/test；本轮所有新结果的 official-valid/test evaluations 均为 0。

### 2026-07-26 KSVD-JK fixed ensemble / train-only auxiliary screen

本轮首先测试了一个与既有 feature injection 明显不同的方向：把固定 KSVD sparse code 当作**仅训练期的 node-level 自监督教师**，推理路径保持纯 GINE-JK。测试了 balanced support prediction、signed-code cosine prediction、auxiliary weight `0.02/0.05/0.10/0.20`，以及只在前 `1/2/3/5/10/15` epoch 开启的短 curriculum。最佳候选为 code cosine、weight `0.05`、前 5 epoch，n=8000 seed-0 inner AUC `0.785143`，仍低于 matched GINE-JK `0.793050`，因此该路线停止，不补 seeds/full-inner。

随后验证固定、无需调权重的双分支概率集成：

```text
p = 0.5 * p(GINE-JK) + 0.5 * p(KSVD-JK graph-max residual)
```

两个分支在同一个 inner split 上独立训练，并各自仅用 inner-valid 选择 epoch；ensemble weight 预先固定为 `0.5/0.5`。它不是单模型，训练/推理计算约为两倍，但总 trainable parameters 为 `37,382 + 43,722 = 81,104`，仍只有 CIN-small `138,385` 的约 `58.6%`。

n=8000、三个 model seeds：

| family | inner AUC mean ± sample std |
|---|---:|
| GINE-JK | 0.775824 ± 0.025611 |
| KSVD graph-max residual | 0.779511 ± 0.017629 |
| fixed 0.5/0.5 ensemble | **0.785213 ± 0.021113** |

ensemble 相对两个单分支的 mean delta 分别为 `+0.009390` 和 `+0.005702`，均为 2/3 seeds 胜出。

n=8000、inner split seeds `1729/2718/31415`：

| family | split mean ± sample std |
|---|---:|
| GINE-JK | 0.746924 ± 0.046465 |
| KSVD graph-max residual | 0.749860 ± 0.043394 |
| fixed 0.5/0.5 ensemble | **0.759159 ± 0.040706** |

full-inner、三个 model seeds：

```text
GINE-JK:             [0.830100, 0.826079, 0.806157]
KSVD graph residual: [0.817182, 0.813525, 0.819274]
fixed ensemble:      [0.832412, 0.827483, 0.821104]
```

对应 mean ± sample std：

```text
GINE-JK             0.820779 ± 0.012821
KSVD graph residual 0.816661 ± 0.002910
fixed ensemble      0.827000 ± 0.005669
```

fixed ensemble 在三个 full-inner model seeds 上均同时超过两个单分支，mean delta 相对 GINE-JK 为 `+0.006221`，相对 KSVD graph residual 为 `+0.010339`。这是当前最稳定的新正信号，建议作为 configuration-freeze candidate；但它仍然只是 inner-only 结果，**不能与 CIN official-test 直接比较，也不能宣称已稳定达到 0.80 official performance**。

结构化汇总：`node_token_ksvd_jk_ensemble_inner_only_summary.json`。本轮所有新运行的 `official_valid_evaluations` 和 `official_test_evaluations` 均为 0；不得用 official test 继续选择 ensemble 权重或分支。

### 2026-07-26 frozen KSVD-JK ensemble official-valid evaluation

在写入 `node_token_ksvd_jk_ensemble_official_valid_freeze_v1.json` 并冻结两个分支、独立 epoch selection 和固定 `0.5/0.5` probability averaging 后，完成 seeds `0/1/2` 的一次性 official-valid 验证；official test 未运行。

| family | seed0 | seed1 | seed2 | mean ± sample std |
|---|---:|---:|---:|---:|
| GINE-JK | 0.814913 | 0.808960 | 0.755123 | 0.792999 ± 0.032936 |
| KSVD-JK graph-max residual | 0.794196 | 0.770249 | 0.793920 | 0.786122 ± 0.013747 |
| fixed 0.5/0.5 ensemble | 0.811609 | 0.799971 | 0.791997 | **0.801192 ± 0.009863** |

固定 ensemble 相对 GINE-JK 的 official-valid mean delta 为 `+0.008194`，相对 KSVD graph residual 为 `+0.015071`。它并非逐 seed 全胜：相对 GINE-JK 为 1/3 wins，相对 KSVD 分支为 2/3 wins；主要收益是利用两个分支的互补误差显著降低方差，尤其把 GINE-JK seed2 的 `0.755123` 稳定到 `0.791997`。

原 terminal runs 只写入 AUC、没有序列化 official-valid predictions，导致无法计算预冻结的 ensemble。为修复这个记录缺失，增加了只影响输出的 `--capture-official-valid-predictions`，并做了六次 deterministic capture reruns。所有 selected epoch、inner/official-valid AUC、初始化 hashes 和首批 minibatch fingerprints 与原运行逐项完全一致。协议上明确披露：6 次原 official-valid computations + 6 次 prediction-capture repeats，共 12 次计算；没有据此修改任何超参数或 ensemble 权重。详见 `node_token_ksvd_jk_official_valid_capture_amendment_v1.json`。

结构化结果：`node_token_ksvd_jk_ensemble_official_valid_summary_v1.json`。本轮 `official_test_evaluations = 0`。该结果支持“方案可行、主要价值为 KSVD 与 GINE-JK 的稳定性互补”，但仍不能与 CIN official-test 直接比较，也不能据此宣称已达到 CIN test performance。

### 2026-07-28 task-matched relation-bank sidecar and prototype-seed robustness

- `REALPROTOTYPE_TASKMATCHED_RELATION_SIDECAR_20260728.md`: nested task-aware vocabulary does not improve standalone occurrence, but its compact exact-distance 1+2 features add assignment-specific signal when used only as a relation sidecar on the frozen broad farthest/scaffold occurrence pair. Prototype seed `20260728` with residual cap `0.3125` reached mean AUC `0.761675` (`+0.005939`, 3/3 fold wins) over the broad occurrence pair; broad relation pair + sidecar reached `0.762375`.
- `REALPROTOTYPE_TASKMATCHED_RELATION_ROBUSTNESS_20260728.md`: the gain did not survive changing the candidate-bank/prototype seed to `20260729` (`+0.001884` on the broad occurrence pair). The selected task-aware banks had zero exact atom overlap and only `0.582958` mean optimal-matching cosine across prototype seeds. The robustness gate failed, so no official-valid/test run was performed.
- Current interpretation: task-aware local relation composition remains a plausible mechanism, but hard selection of 32 atoms from a random 256-candidate reservoir is unstable. The next route is broad deterministic candidates plus nested scalar relation weights/gates or multi-reservoir selector consensus, not longer random walks or a larger relation head.
- Structured summaries: `taskmatched_relation_sidecar_summary_scaffold3.json` and `taskmatched_relation_protoseed_robustness_summary.json`. All runs in this round have `official_valid_evaluations = 0` and `official_test_evaluations = 0`.

### 2026-07-28 fixed-vocabulary nested relation-gate diagnostic

- `REALPROTOTYPE_FIXED_VOCABULARY_RELATION_GATES_20260728.md`: fixed the broad deterministic `farthest` and `scaffold_facility` real-patch banks, then used source-clean nested inner-scaffold OOF ablation ranks only as smooth scalar gates on prototype assignments. The exact-distance 1+2 joint uniform relation head improved the frozen broad occurrence pair from `0.755736` to `0.760277` (`+0.004541`), confirming the label-free local-relation signal.
- The true-label task gate reached `0.759702` (`+0.003965`) but was below the uniform gate in all three folds (mean `-0.000576`) and below the matched shuffled-label gate in all three folds (mean `-0.001140`). It beat its own assignment-shuffled control in all folds (`+0.002064` mean), so local assignment structure matters, but prototype-wise supervised scalar weighting does not improve it.
- Promotion gate failed. No official-valid/test evaluation was performed. The next route should change the supervised object (for example, a strictly nested low-rank pairwise relation metric) rather than tuning scalar-gate strength or residual cap.
- Structured summary: `fixed_vocabulary_relation_gates_summary_scaffold3.json`. Runner and summarizer: `code/run_molhiv_fixed_vocabulary_relation_gates.py` and `code/summarize_molhiv_fixed_vocabulary_relation_gates.py`.

### 2026-07-29 exact-path atom–path overlap residual diagnostic

- `PATH_OVERLAP_RESIDUAL_20260729.md`: 保留 fold 1 的精确路径线性基础 `0.772548`，新增约 12.8k 参数的原子—路径二部残差，只按真实“路径包含原子”关系聚合，不使用 atom–atom GNN、GINE、KSVD 或 prototype bank。
- 初始 in-sample + zero-gate 版本没有有效学习；改用严格 scaffold OOF 基础分数和直接零初始化残差后，真实重叠模型达到 `0.773073`（`+0.000524`），但关闭成员关系的等容量 path-set 控制达到更高的 `0.773228`（`+0.000680`）。
- 进一步把严格 OOF 精确路径系数附到每个路径 occurrence 后，真实重叠模型为 `0.773279`（`+0.000731`）；严格等参数、等输入但关闭重叠传播的控制反而达到本轮最好 `0.773450`（`+0.000902`，总参数 48,954）。真实重叠传播相对匹配控制低 `0.000171`，且所有增益均远低于 `+0.005` 晋级门槛。
- 结论：简单 atom–path mean/max 消息聚合没有提供可辨认的跨 scaffold 新信号，因此不扩展三折，不评估 official-valid/test。结构化结果为 `path_overlap_residual_fold1_probe_20260729.json`、`path_overlap_residual_oof_direct_fold1_probe_20260729.json`、`path_set_residual_oof_direct_fold1_control_20260729.json`、`path_evidence_overlap_residual_oof_fold1_probe_20260729.json` 和 `path_evidence_set_residual_oof_fold1_control_20260729.json`。

### 2026-07-29 chemical fragment graph diagnostic

- `CHEMICAL_FRAGMENT_GRAPH_20260729.md`: abandoned KSVD/path/GINE as the core and deterministically decomposed each molecule into cyclic ring systems and maximal non-ring chain segments. Full OGB atom/bond fields are encoded inside fragments; only the fragment graph receives neural message passing.
- The initial two-layer run appeared to improve the hard outer fold 1 from fragment-set `0.756517` and shuffled-connection `0.753328` to true fragment-graph `0.777454`, but the three controls accidentally used different training seeds. This result is retained for audit but is not valid structural evidence.
- Under strictly matched initialization/minibatch seeds, the two-layer true graph reached only `0.743984` versus `0.745762` for shuffled connection (`-0.001778`). A one-layer, more strongly regularized version reached `0.751664` versus shuffled `0.754558` (`-0.002894`). Inner selected AUC remained as high as `0.833695`, exposing severe inner-to-hard-outer scaffold mismatch and training-seed instability.
- The current unordered mean/max fragment encoder is rejected: it does not preserve chain order, ring substitution position, or boundary-atom role. No additional outer folds or official-valid/test evaluation was performed. Any continuation must change to position-aware fragment-internal encoding and matched-seed structural controls rather than adding width/layers or a GINE side branch.
- Structured results: `chemical_fragment_graph_fold1_probe_20260729.json`, `chemical_fragment_graph_fold1_matched_seed_probe_20260729.json`, and `chemical_fragment_graph_local1_fold1_matched_probe_20260729.json`. All have `official_valid_evaluations = 0` and `official_test_evaluations = 0`.

## 2026-07-29：位置感知化学片段图终止结论

本轮系统比较了链内位置、环内位置、连接角色、片段内局部原子消息传递和 BRICS 化学块，并在所有关键实验中使用同初始化、同训练顺序的真实连接/打乱连接控制。

全图 fold 1 的关键结果：

| 模型 | set | true graph | shuffled graph |
|---|---:|---:|---:|
| chain+ring | 0.756840 | 0.740292 | 0.736223 |
| chain+ring+ports | 0.751727 | 0.754298 | 0.770610 |
| local atom MP + ports | 0.731787 | 0.719482 | 0.733601 |

BRICS 在 8k 固定 20 轮下为 `0.589786`，低于打乱控制 `0.604520`。

结论：片段内容可达到约 0.75，但片段连接传播没有形成稳定的跨 scaffold 收益；该路线未达到 full fold 1 `0.785` 和 true-minus-shuffled `+0.005` 的双门槛，停止扩展三折及 official valid/test。完整分析见：

```text
tracks/ksvd/docs/CHEMICAL_FRAGMENT_GRAPH_20260729.md
```
