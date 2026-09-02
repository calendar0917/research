# luyin14 后续：relation-conditioned structured pursuit screen

> 日期：2026-08-12  
> 状态：结果前冻结  
> 前置结果：`results/luyin14/RICH_READOUT_20260812.md`

## 1. 研究问题

现有 relation readout 是先对每个 patch 独立 OMP，再在图级统计相邻 code 的关系。它无法
修复一个更早的问题：重叠、连续的 patch 可能因为局部噪声发生 support flip。仓库中
group-sparse/structured OMP 被列为候选，但尚无实现或结果。

本轮只检验：

> 在固定字典下，让真实重叠关系参与 sparse support pursuit，是否优于独立 OMP，且是否
> 优于保持关系图统计但打乱 patch↔position 绑定的 structured control。

## 2. 冻结 pursuit

先用原有 OMP 得到 `X0`。随后做两轮同步 support refinement。对 patch `j`：

1. `corr = |D^T y_j| / max(|D^T y_j|)`；
2. `prior` 是 overlap/chain 邻居上一轮 binary supports 的加权平均；
3. `score = 0.75 * corr + 0.25 * prior`；
4. 取固定 `T=3` 个最高分 atom，在该 support 上重新 least-squares 拟合系数。

关系权重固定为现有两个通道的逐元素最大值：真实 node-set Jaccard overlap，以及同 segment
连续 patch 的 chain relation。孤立 patch 保持独立 pursuit。只用一档 `rho=0.25`、两轮；
不得扫描强度或轮数。

`SHUFFLED` 对每图以冻结 seed 同时置换关系矩阵的行列，但不置换 patch/code；它保持关系图
的边数、权重、度分布和谱，只打破关系与 patch 内容的绑定。

## 3. 其余协议

完全复用 luyin14：四数据集、FAIR95 patch、`K=24/T=3/iterations=5`、split seeds
`0/1/2`、3-fold、train-fold-only dictionary/scaler/classifier。分类仍是固定线性 LR。

必报：

- INIT/FINAL independent rich；
- INIT/FINAL TRUE structured rich；
- FINAL SHUFFLED structured rich；
- `STATS` 与 `STATS + FINAL TRUE structured rich`；
- support-agreement、support-change fraction、reconstruction relative change。

带节点特征数据额外报告 `FEATURE_STATS` 与
`FEATURE_STATS + FINAL TRUE structured rich`。

## 4. 晋级门槛

单数据集稳定通过仍要求 mean paired delta `>= +0.01` balanced accuracy 且至少 `6/9`
folds 为正。

1. `FINAL_TRUE - FINAL_INDEPENDENT` 至少 2/4 数据集稳定通过；
2. `FINAL_TRUE - FINAL_SHUFFLED` 至少 2/4 数据集稳定通过；
3. `STATS+FINAL_TRUE - STATS` 至少 2/4 数据集稳定通过；
4. TRUE 的 relation-edge support agreement 应高于 independent，同时 reconstruction error
   不得平均恶化超过 5%。

前三项同时通过才进入 matched PCA/random-patch 与 structured-penalty solver。若 TRUE 只胜
independent、不胜 SHUFFLED，则是 generic support regularization，不是结构绑定。若 TRUE 胜
SHUFFLED 但不胜 independent，说明当前关系存在但 pursuit bias 有害。若 added-value 失败，
则 support 平滑仍只重复 graph statistics。

