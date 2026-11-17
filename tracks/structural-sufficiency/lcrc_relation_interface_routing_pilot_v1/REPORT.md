# 关系–接口路由 pilot：LCRC 局部编码分块化 + 接口上下文

轮次 `lcrc-relation-interface-routing-pilot-v1`；一切数字均为 **DEVELOPMENT**，Test 列是 **DEVELOPMENT_TEST**。
本轮只训练 R1/R2 四个训练单元；C1、GINE、oracle 基线一律只读复用，不重训、不重选、不重评。

## 1. 本轮问题

1. 把每个 LCRC S1 出现路由到拥有该键的拓扑 block、把每个 S2 出现路由到拥有两条端点键的单个 block 或两个 block 的接口，预测力能否基本保留？
2. 接口贡献是否大到值得作为一个显式对象？
3. 接口是否需要两个端点 block 的上下文（R2 相对 R1）？
4. 门控选出的 block 是否比匹配随机连通子图更好地定位 WC_LOGP 的 oracle 质量？
5. 是否推进到可学习分块 / LCRC-v2？

这是**架构可行性 pilot**，不是最终论文验证；单 seed、无超参搜索、无额外数据集。

## 2. 冻结的分块与路由（CPU 生成，一次，SHA-256 固定）

分块判据：`bridges of the physical bond graph; non-bridge connected components = cycle blocks; maximal bridge paths with degree-2 non-ring interiors = chain blocks; isolated atoms = zero-bond blocks; every bond owned exactly once`

支持门控（Valid）：>=2 blocks AND >=1 cross-boundary S2 occurrence on the molecule；实测 **1.0000** ≥ 阈值 0.50 → `PARTITION_USABLE`。

| 量 | Valid |
|---|---|
| 分子数（≥2 block / 可用） | 420（420 / 420） |
| block 数（均值，最大键数） | 4231（10.074，33 键） |
| block 类型 cycle / chain / isolated | 1206 / 3022 / 3 |
| 单 block 分子数 | 0 |
| 接口数（均值） | 4631（11.026） |
| 共享原子数（占比） | 3088（0.2454） |
| 零键 block | 3 |
| S1 出现数 | 25170 |
| S2 总数 = 内部 + 跨块 | 35554 = 21144 + 14410 |

两份 GPU 读的是**同一份** CPU 缓存（SHA-256）：

| split | routing cache sha256 |
|---|---|
| train | `8fc85c7fba7c8bd11e230f0f278f9be53df70d181eae568eeedb29fbbc2500dc` |
| valid | `5e43da2eb3bd99e36ae761c2bdef22833d6f263821172129fad5ec28967d1ec3` |
| test | `890407976d3e9b9c136b83eb5390c099af39358cfe91433914d6aec073fca6cd` |

## 3. 架构与参数预算

| 模型 | trunk | block/接口读出 | 总计 | 相对 C1 | R2/R1 |
|---|---|---|---|---|---|
| C1（只读基线） | — | — | 228674 | 1.0000 | — |
| R1 | 213376 | 28484 | 241860 | 1.0577 | — |
| R2 | 213376 | 45380 | 258756 | 1.1315 | 1.0699 |

预算规则：total parameters within +/-25% of the frozen C1 total；容量规则：R2/R1 parameter ratio <= 1.1 else CAPACITY_CONFOUNDED。
共享初始权重：`44` 个张量在 R1/R2 之间逐位相同 → **是**。容量混淆：**否**。

块门控是二值的：`y(m) = bias + rho_g + Σ_i m_i c_i + Σ_(i,j) m_i m_j c_ij`，`delta_i = y(all-on) − y(m_i=0)`，`score_i = |delta_i| / sqrt(max(1, n_bonds_i))`。

## 4. 训练与门控状态

| unit | 模型 | 选中 epoch | Train RMSE | Val RMSE | Test RMSE | final Val RMSE | device | 参数量 |
|---|---|---|---|---|---|---|---|---|
| lipo | R1 | 106 | 0.4453 | 0.8398 | 0.8027 | 0.9149 | cuda:0 | 241860 |
| lipo | R2 | 31 | 0.5971 | 0.8116 | 0.7939 | 0.9252 | cuda:0 | 258756 |
| wc_logp | R1 | 237 | 0.0366 | 0.1501 | 0.1170 | 0.1522 | cuda:0 | 241860 |
| wc_logp | R2 | 112 | 0.0538 | 0.1640 | 0.1542 | 0.1731 | cuda:0 | 258756 |

门控总状态：**是**（12 个叶门控分 6 组，失败项：无）。训练后门控是在四个正式 checkpoint 上跑的：lipo/R1, lipo/R2, wc_logp/R1, wc_logp/R2。

## 5. 五个问题的回答

### Q1 固定拓扑分块 + 接口路由是否基本保留了预测力？

| unit | 模型 | split | RMSE | 可比性 |
|---|---|---|---|---|
| lipo | C1 (frozen, Rf) | valid | 0.706885 | DIRECTLY_COMPARABLE |
| lipo | C1 (frozen, Rf) | test | 0.729324 | DIRECTLY_COMPARABLE |
| lipo | GINE-G0 (frozen, Rf) | valid | 0.616077 | DIRECTLY_COMPARABLE |
| lipo | GINE-G0 (frozen, Rf) | test | 0.680001 | DIRECTLY_COMPARABLE |
| lipo | GINE-Gx (frozen, Rf) | valid | 0.630671 | DIRECTLY_COMPARABLE |
| lipo | GINE-Gx (frozen, Rf) | test | 0.674947 | DIRECTLY_COMPARABLE |
| lipo | C1 (frozen, R3) | valid | 0.708624 | RECIPE_DIFFERS |
| lipo | C1 (frozen, R3) | test | 0.725392 | RECIPE_DIFFERS |
| wc_logp | C1 (frozen, R3) | valid | 0.062555 | RECIPE_DIFFERS |
| wc_logp | C1 (frozen, R3) | test | 0.078086 | RECIPE_DIFFERS |
| lipo | R1 | valid | 0.839807 | THIS_ROUND |
| lipo | R1 | test | 0.802693 | THIS_ROUND |
| lipo | R2 | valid | 0.811599 | THIS_ROUND |
| lipo | R2 | test | 0.793868 | THIS_ROUND |
| wc_logp | R1 | valid | 0.150077 | THIS_ROUND |
| wc_logp | R1 | test | 0.116955 | THIS_ROUND |
| wc_logp | R2 | valid | 0.164044 | THIS_ROUND |
| wc_logp | R2 | test | 0.154153 | THIS_ROUND |

| 模型 | Lipo Val RMSE | ÷C1(Rf) | Lipo Test ÷C1 | WC Val RMSE | ÷C1(R3) |
|---|---|---|---|---|---|
| R1 | 0.8398 | 1.1880 | 1.1006 | 0.1501 | 2.3991 |
| R2 | 0.8116 | 1.1481 | 1.0885 | 0.1640 | 2.6224 |

判据：Val RMSE ≤ 1.10 × C1，两个训练单元都要满足。

* R1 保留：**否**；R2 保留：**否**；整体 `PREDICTION_RETENTION_FAIL`。

* **R1**：Lipo Val 是 C1 的 1.188 倍，WC_LOGP Val 是 C1 的 2.399 倍 → 未保留。
* **R2**：Lipo Val 是 C1 的 1.148 倍，WC_LOGP Val 是 C1 的 2.622 倍 → 未保留。

只有在 Lipophilicity 上这个比较是同口径的（C1 也是 recipe Rf、seed 0、同一split 与 scaler）；WC_LOGP 的 C1 基线是 recipe R3，因此那一列的倍数把“架构”和“recipe”混在一起，**不能**归因于路由。

* `lipo/R1`：选中 epoch 的 Train RMSE 0.4453，Val RMSE 0.8398，Val/Train = 1.89 → 拟合不足 / 泛化差距小。
* `lipo/R2`：选中 epoch 的 Train RMSE 0.5971，Val RMSE 0.8116，Val/Train = 1.36 → 拟合不足 / 泛化差距小。
* `wc_logp/R1`：选中 epoch 的 Train RMSE 0.0366，Val RMSE 0.1501，Val/Train = 4.10 → 过拟合。
* `wc_logp/R2`：选中 epoch 的 Train RMSE 0.0538，Val RMSE 0.1640，Val/Train = 3.05 → 过拟合。

解释口径（见 `analysis/decision.json` 的 `caveats`）：本轮按 brief 冻结的架构一次性替换了 C1 读出的两件事——把逐原子的 `Σ_v f(h_v)` 换成 block 内加权平均后约 10 项的`Σ_i f_block(z_i)`，并把 `rho(g, mean h, mean h², log1p n)` 换成只含 `g` 与 `log1p n_atoms` 的 `rho_g`。所以 1.19× / 1.15× 是**整个替换**的代价，不是“路由”单项的代价；要分离两者需要保留 C1 读出的对照模型，那超出本轮范围。

可以确定的是：`bias + rho_g` 单独几乎等于常数预测（见第 6 节），所以预测确实是 block/接口账目贡献的，而不是全局项；block/接口账目在数值上闭合、在 relabelling 下不变，但在这个宽度和这种池化下，它没有达到 C1 的泛化水平。

### Q2 接口贡献是否大到值得作为显式对象？

| unit/split | mean abs(Σc_i) | mean abs(Σc_ij) | median RMS c_i | median RMS c_ij | mean abs(rho_g) |
|---|---|---|---|---|---|
| lipo/R1/test | 0.6063 | 0.4315 | 0.2933 | 0.1277 | 0.2732 |
| lipo/R1/valid | 0.6403 | 0.4225 | 0.3022 | 0.1254 | 0.2740 |
| lipo/R2/test | 0.5092 | 0.4162 | 0.2759 | 0.0944 | 0.2843 |
| lipo/R2/valid | 0.5262 | 0.4267 | 0.2810 | 0.0910 | 0.2814 |
| wc_logp/R1/test | 0.6351 | 0.4787 | 0.2611 | 0.1130 | 0.5251 |
| wc_logp/R1/valid | 0.6039 | 0.5027 | 0.2517 | 0.1120 | 0.5106 |
| wc_logp/R2/test | 0.6164 | 0.3933 | 0.2879 | 0.0975 | 0.5255 |
| wc_logp/R2/valid | 0.5807 | 0.4162 | 0.2746 | 0.0957 | 0.5178 |

`Σc_ij` 是接口项之和，`Σc_i` 是 block 项之和，`rho_g` 是唯一的全图项。接口 RMS 与 block RMS 同量级则说明接口不是装饰项；若 `Σc_ij` 远小于 `Σc_i`，接口可以退化成 block 内部证据。

接口项之和相对 block 项之和的全部 8 个 (unit, split) 组合都在 0.64–0.83 之间（最大出现在 `wc_logp/R1/valid`：`Σc_ij / Σc_i = 0.832`），接口与 block 的单 token RMS 比 0.32–0.44（最大 0.445）。也就是说接口项的量级与 block 项同阶，**不是**可以忽略的修正项：如果只用 block 内部证据，会丢掉三分之一的账目质量。这里给的是量级陈述，本轮没有为它设预登记门控；接口“够大”不等于接口上下文“有用”——后者由 Q3 的配对 bootstrap 回答，而答案是否定的。

### Q3 接口是否需要两个端点 block 的上下文（R2 vs R1）？

| unit/split | RMSE R1 | RMSE R2 | R2−R1（负值=R2 更好） | 95% CI（配对 bootstrap） | P(R2 更好) | n |
|---|---|---|---|---|---|---|
| lipo/test | 0.8027 | 0.7939 | -0.008825 | [-0.073002, 0.057045] | 0.5955 | 420 |
| lipo/valid | 0.8398 | 0.8116 | -0.028208 | [-0.091266, 0.038352] | 0.7765 | 420 |
| wc_logp/test | 0.1170 | 0.1542 | 0.037197 | [0.017857, 0.056033] | 0.0000 | 420 |
| wc_logp/valid | 0.1501 | 0.1640 | 0.013967 | [-0.020106, 0.043476] | 0.2185 | 420 |

配对 bootstrap：2000 次重抽，seed 20261117，分子级配对（同一行重采样同时用于两个模型）。

* `lipo/test`：R2−R1 = -0.008825，95% CI [-0.073002, 0.057045]，P(R2 更好) = 0.596 → 无显著差异。
* `lipo/valid`：R2−R1 = -0.028208，95% CI [-0.091266, 0.038352]，P(R2 更好) = 0.776 → 无显著差异。
* `wc_logp/test`：R2−R1 = 0.037197，95% CI [0.017857, 0.056033]，P(R2 更好) = 0.000 → R2 显著更差（R1 更好）。
* `wc_logp/valid`：R2−R1 = 0.013967，95% CI [-0.020106, 0.043476]，P(R2 更好) = 0.218 → 无显著差异。

结论：在 1.10 倍容量约束下，两端上下文没有带来可分辨的增益；接口的自身路由证据（R1 形式）已经足够，不能仅凭“接口需要一个对象”就推断“接口需要两端上下文”。

反向证据：`wc_logp/test` 上 R2 显著**更差**（CI 不含 0 且 P(R2 更好) ≈ 0）。在这一单元上，把两端 block 的上下文拼进接口不但没帮忙，反而引入了额外的可过拟合自由度。

### Q4 门控选出的 block 是否比匹配随机连通子图更好地定位 WC_LOGP oracle 质量？

| unit/split | 可匹配分子（coverage） | 平均 control 数 | capture（选中 block） | capture（control） | enrichment 均值 | enrichment 95% CI | enrichment>0 比例 | 分摊账目闭合 |
|---|---|---|---|---|---|---|---|---|
| R1/test | 349/420 | 13.5881 | 0.1098 | 0.0846 | 0.0252 | [0.0212, 0.0292] | 0.7278 | 是 |
| R1/valid | 364/420 | 14.0476 | 0.1102 | 0.0831 | 0.0270 | [0.0232, 0.0307] | 0.7280 | 是 |
| R2/test | 359/420 | 13.8643 | 0.1101 | 0.0839 | 0.0262 | [0.0223, 0.0302] | 0.7242 | 是 |
| R2/valid | 364/420 | 13.9976 | 0.1098 | 0.0821 | 0.0277 | [0.0240, 0.0315] | 0.7308 | 是 |

coverage = 可匹配分子 / 全部 420；control 数上限 16、下限 8，RNG 20261117。enrichment 的 95% CI 是分子级 bootstrap（2000 次，seed 20261117），只对可匹配分子计算。

capture 用的是**未分摊**的候选原子集：`Σ_{v∈atoms(block)} |c_v| / Σ_v |c_v|`，因此与同 bond 数、同 atom 数的随机连通子图是可比的。`w(v,i)` 分摊账目另外计算并逐分子验证闭合，只作为记账，不参与上面的比较。

* `R1/test`：可匹配 349/420 个分子（匹配不足不参与结论），enrichment 均值 0.0252，正 enrichment 比例 0.7278，`|delta|` 与分摊到选中 block 的 oracle 质量的 Spearman 相关 0.538。
* `R1/valid`：可匹配 364/420 个分子（匹配不足不参与结论），enrichment 均值 0.0270，正 enrichment 比例 0.7280，`|delta|` 与分摊到选中 block 的 oracle 质量的 Spearman 相关 0.523。
* `R2/test`：可匹配 359/420 个分子（匹配不足不参与结论），enrichment 均值 0.0262，正 enrichment 比例 0.7242，`|delta|` 与分摊到选中 block 的 oracle 质量的 Spearman 相关 0.526。
* `R2/valid`：可匹配 364/420 个分子（匹配不足不参与结论），enrichment 均值 0.0277，正 enrichment 比例 0.7308，`|delta|` 与分摊到选中 block 的 oracle 质量的 Spearman 相关 0.505。

结论：门控 block 的 oracle 质量捕获高于匹配随机连通子图，最大单元 enrichment = 0.0277（95% CI [0.0240, 0.0315]），相对 control 的 0.0821 提升约 33.8%。量级不大：中位选中 block 只有 1 根键、2–3 个原子，因此这是一个“不是随机”的证据，不是“找到了关键子结构”的证据。

注意：WC_LOGP 本身就是逐原子可加量，所以它是对“定位”最有利的任务；在这里没有信号，在实验标签上更不会有。

### Q5 是否推进到可学习分块 / LCRC-v2？

预登记决策：`PAUSE_FIXED_TOPOLOGY_PARTITION`。

| 命题 | 判定依据 | 结论 |
|---|---|---|
| A 结构路由可行 | retention_ok = {'R1': False, 'R2': False} | 否 |
| B 接口上下文有增量 | incremental = False，capacity_confounded = False | 否 |
| C 结构解释有意义 | oracle_positive = True | 是 |

建议：**暂停**固定拓扑分块这条路线。在 1.10 倍预测力约束下，本轮的固定分块 + 接口路由没有通过预测力门控，继续在它上面加可学习分块只会把“表达能力不足”和“分块方式不好”两个原因混在一起。如果还要走结构路由，下一步应先解决 block 内池化（多 token / 注意力），并用同一个门控重新测保留率。

无论哪种结论，本轮都不构成论文级验证：单 seed、单数据集、pilot 规模的门控阈值都是预先登记的局部标尺。

## 6. 读数是否只是 rho_g？

| unit/split | 只用 bias+rho_g 的 RMSE | 完整模型 RMSE | Train 均值常数 RMSE |
|---|---|---|---|
| lipo/R1/valid | 1.200869 | 0.839807 | 1.217439 |
| lipo/R1/test | 1.074325 | 0.802693 | 1.098742 |
| lipo/R2/valid | 1.201036 | 0.811599 | 1.217439 |
| lipo/R2/test | 1.108420 | 0.793868 | 1.098742 |
| wc_logp/R1/valid | 1.184097 | 0.150077 | 1.490494 |
| wc_logp/R1/test | 1.143921 | 0.116955 | 1.402527 |
| wc_logp/R2/valid | 1.176446 | 0.164044 | 1.490494 |
| wc_logp/R2/test | 1.146046 | 0.154153 | 1.402527 |

| unit | mean abs(rho_g) | mean abs(Σc_i) | mean abs(Σc_ij) | 比值 |
|---|---|---|---|---|
| lipo/R1 | 0.2740 | 0.6403 | 0.4225 | 0.258 |
| lipo/R2 | 0.2814 | 0.5262 | 0.4267 | 0.295 |
| wc_logp/R1 | 0.5106 | 0.6039 | 0.5027 | 0.461 |
| wc_logp/R2 | 0.5178 | 0.5807 | 0.4162 | 0.519 |

判据比值 > 4 → `RHO_DOMINATES_STRUCTURE`；本轮标记 **否**。


## 7. 资源与运行

| unit | 选中 epoch | 训练秒（360 epoch） | GPU·h |
|---|---|---|---|
| lipo/R1 | 106 | 489.0 | 0.136 |
| lipo/R2 | 31 | 526.1 | 0.146 |
| wc_logp/R1 | 237 | 493.0 | 0.137 |
| wc_logp/R2 | 112 | 519.0 | 0.144 |

四个训练单元合计 **0.563 GPU·h**（软上限 1 GPU·h/卡，硬上限 2 GPU·h/卡）；R1 在 `res-gpu0`、R2 在 `res-gpu1`，CPU 只做 prepare/checks/aggregate。

## 8. 可复现性

| 项 | 值 |
|---|---|
| commit | `8f07115ee5e3049c4ee04345a151634d5427b060` |
| branch | `exp/lcrc-explanation-task-relevance-v1` |
| freeze hash | `548b8ed6e4a34f611ddfb06c2686693208da98a862465c9a32e1db575081242f` |
| lockfile sha256 | `d063dedc762d1f338661cbd67c59a7245a0efb2966a2b932836530cf5cba7033` |
| started / finished | 2026-11-17T20:44:22 / 2026-11-17T20:46:39 |

聚合环境（CPU，只读产物）：main，python 3.12.15，torch 2.5.1+cu124，numpy 2.1.3，rdkit 2026.09.1，cuda_available False。

四个训练单元的真实 GPU 环境（每单元 trajectory.json 记录）：

| unit | host | torch | torch_cuda | 可见设备 | GPU |
|---|---|---|---|---|---|
| lipo/R1 | a100-2 | 2.5.1+cu124 | 12.4 | CUDA_VISIBLE_DEVICES=0 | NVIDIA A100-SXM4-40GB |
| lipo/R2 | a100-2 | 2.5.1+cu124 | 12.4 | CUDA_VISIBLE_DEVICES=1 | NVIDIA A100-SXM4-40GB |
| wc_logp/R1 | a100-2 | 2.5.1+cu124 | 12.4 | CUDA_VISIBLE_DEVICES=0 | NVIDIA A100-SXM4-40GB |
| wc_logp/R2 | a100-2 | 2.5.1+cu124 | 12.4 | CUDA_VISIBLE_DEVICES=1 | NVIDIA A100-SXM4-40GB |

交付清单：

| 交付物 | 存在 | 字节 |
|---|---|---|
| `analysis/partition_support.json` | 是 | 10809 |
| `analysis/routing_invariants.json` | 是 | 20022 |
| `analysis/parameter_budget.json` | 是 | 2249 |
| `analysis/prediction_comparison.csv` | 是 | 4754 |
| `analysis/oracle_localization.csv` | 是 | 373903 |
| `analysis/interface_ledger.csv` | 是 | 1829755 |
| `analysis/block_ledger.csv` | 是 | 893608 |
| `analysis/native_ledger_summary.json` | 是 | 3330 |
| `analysis/oracle_localization_summary.json` | 是 | 7373 |
| `analysis/bootstrap_r2_vs_r1.json` | 是 | 2283 |
| `analysis/rho_only_baseline.csv` | 是 | 813 |
| `analysis/decision.json` | 是 | 10286 |
| `analysis/run_manifest.json` | 是 | 16155 |

## 9. 本轮明确没有做的事

* 没有重新训练 C1 / GINE / oracle 基线，也没有重选它们的 epoch；
* 没有新 seed、没有超参搜索、没有额外数据集、没有 SMARTS/官能团先验；
* 没有用 property label 或 oracle 贡献定义结构；
* 没有用 Test 做任何选择（Test 只作为 DEVELOPMENT_TEST 报告）；
* 没有做多 seed 稳定性声明，也没有 SOTA 声明。
