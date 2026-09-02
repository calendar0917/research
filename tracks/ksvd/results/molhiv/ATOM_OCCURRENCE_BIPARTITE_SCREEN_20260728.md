# Atom--occurrence 二部模型：8000 图首轮筛选

日期：2026-07-28

## 1. 目的

这轮实现并筛选路线 A：不再叠加完整 GINE，而是把“某个局部原型在分子中的一次连续出现”作为真实中间实体，只允许信息沿

```text
atom -> occurrence -> atom
```

传播。模型没有 atom--atom GINE，也没有 occurrence--occurrence GINE，避免局部图网络绕过字典。

本轮同时回答两个问题：

1. 二部传播本身是否比只做 node-level MIL 更有用；
2. 如果二部传播有用，优势是否确实来自 KSVD，而不是任意一组真实局部原型或随机方向。

## 2. 协议

- 数据：MolHIV 的 8,000 图开发子集；
- 选择数据：仅 official-train 中的 6,400 图；
- 划分：3 个固定 Bemis--Murcko scaffold folds；
- 当前阶段：每个 fold 只跑 seed 0；
- 训练：固定 30 epochs，训练结束后只评估一次 held-out fold；
- local descriptor：完整 radius-2 permutation-invariant raw patch；
- 降维：每个 outer-fit fold 单独拟合 PCA64；
- supervised/unsupervised GNN layers：均为 0；
- 二部轮数：2；hidden：48；
- 二部模型可训练参数：64,611；
- matched node MIL 参数：29,234；
- official valid/test：均未编码、未评估。

代码：

- `code/run_molhiv_atom_occurrence_bipartite.py`
- `code/summarize_molhiv_atom_occurrence_bipartite.py`

## 3. 首轮结果

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| farthest real-patch node MIL | 0.707778 | 0.676961 | 0.773768 | 0.719503 |
| **farthest real-patch bipartite** | **0.755244** | 0.684862 | **0.768683** | **0.736263** |
| KSVD node MIL | 0.706608 | 0.627601 | **0.780467** | 0.704892 |
| **KSVD bipartite** | 0.734898 | **0.689736** | 0.762768 | **0.729134** |
| PCA bipartite | 0.691542 | 0.663054 | 0.741062 | 0.698553 |
| random real-patch bipartite | 0.686340 | 0.642300 | 0.765273 | 0.697971 |
| random-direction bipartite | 0.732980 | 0.625935 | 0.760277 | 0.706397 |
| KSVD-medoid bipartite | 0.717806 | 0.641593 | 0.738595 | 0.699332 |
| KSVD shuffled-ID | 0.618498 | 0.629200 | 0.654452 | 0.634050 |
| KSVD no-ID | 0.718822 | 0.633437 | 0.760232 | 0.704164 |

## 4. 配对比较

### 4.1 二部结构是否有价值

| 比较 | Mean delta | Wins |
|---|---:|---:|
| farthest bipartite - farthest node MIL | **+0.016760** | **2/3** |
| KSVD bipartite - KSVD node MIL | **+0.024242** | **2/3** |

两组 matched comparison 都跨过了预设的 `mean gain >= +0.005`、至少 `2/3 wins` 门槛。因此，**把 occurrence 作为中间实体进行两轮受限传播，确实比单纯把节点响应汇总成词袋更有潜力**。

但 fold 2 中两组二部模型都下降：farthest `-0.005086`，KSVD `-0.017698`。所以当前证据只说明结构值得关注，还不能说明它已经稳定。

### 4.2 优势是否是 KSVD 特有的

| 比较 | Mean delta | Wins |
|---|---:|---:|
| KSVD - farthest real patch | **-0.007129** | 1/3 |
| KSVD - PCA | +0.030582 | 3/3 |
| KSVD - random direction | +0.022737 | 3/3 |
| KSVD - KSVD medoid | +0.029803 | 3/3 |
| KSVD - shuffled-ID | +0.095084 | 3/3 |
| KSVD - no-ID | +0.024971 | 3/3 |

这里出现了一个很清楚的分层结论：

1. KSVD 明显优于 PCA、随机方向、medoid、shuffled-ID 和 no-ID；
2. 因而 KSVD atom 的方向和稳定身份不是完全无意义的；
3. 但最简单的、从训练 fold 真实局部 patch 中无标签选出的 farthest bank，平均仍比 KSVD 高 `0.007129`；
4. 因此当前结果不能支持“KSVD 是这套二部架构中最好的局部词汇”。

## 5. 过拟合审计

shuffled-ID control 在 fold 0/1 的 fit AUC 分别达到约 `0.983/0.978`，held-out 却只有 `0.618/0.629`。这说明 64.6k 参数的二部模型足以记住错误的原型身份；如果原型身份不稳定，增加传播层或自由 gate 很容易把训练拟合提高，却损害 scaffold 外推。

KSVD persistent ID 相对 shuffled-ID 的 3/3 大幅优势，说明稳定原型身份能够抑制这种问题；但 farthest bank 又表明，“来自真实 patch、覆盖分散且确定”的身份目前比自由 KSVD direction 更适合分类。

## 6. 预注册门槛与决定

预设 strict KSVD promotion 要求：

1. 二部模型胜 matched node MIL；
2. KSVD 平均胜 farthest/PCA/random-direction 全部 geometry controls；
3. KSVD identity 胜 shuffled-ID 与 no-ID。

结果为：

- 条件 1：通过；
- 条件 2：**失败**，因为 KSVD 不及 farthest；
- 条件 3：通过。

因此 strict KSVD promotion **失败**。按事先约定，本轮不继续把完整十组 controls 扩展到 seeds 1/2，也不运行 official valid/test，避免在一个未通过 KSVD-specificity gate 的分支上继续消耗计算和测试集预算。

## 7. 当前研究结论

路线 A 没有整体失败，但需要准确改名：

```text
通过首轮验证的是“真实局部结构 occurrence 的受限二部传播”，
而不是“KSVD 二部模型优于 matched alternatives”。
```

它回答了此前“是否必须再加 GINE”的问题：不必须。只用 atom--occurrence membership 的两轮传播就能在 seed-0 三折中提高 matched node MIL，而且参数仍受控。但它也再次暴露了 KSVD 路线的核心问题：KSVD reconstruction directions 可以提供有意义、稳定的坐标系，却仍没有超过更直接的真实局部 prototype vocabulary。

因此：

- 不再沿 CIN-like 或更深 GINE 方向扩张；
- 不把当前二部模型直接包装成 KSVD 主结果；
- 若后续继续二部路线，应以 farthest real-patch bank 为主体，把 KSVD 限制为 coverage/residual/regularization side information，并重新设立独立晋级门槛；
- 在开始该新分支前，应先讨论它是否仍符合论文希望保留的 KSVD 核心贡献。

## 8. Artifacts

- `results/molhiv/atom_occurrence_bipartite_fold0_seed0.json`
- `results/molhiv/atom_occurrence_bipartite_fold1_seed0.json`
- `results/molhiv/atom_occurrence_bipartite_fold2_seed0.json`
- `results/molhiv/atom_occurrence_bipartite_scaffold3_seed0_summary.json`
- `results/molhiv/atom_occurrence_bipartite_fold{0,1,2}_seed0.log`
