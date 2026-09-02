# luyin14 后续：rich sparse-code readout 迁移审计

> 日期：2026-08-12  
> 状态：结果前冻结  
> 前置结果：`results/luyin14/ROUTE_CLOSURE_20260812.md`

## 1. 唯一研究问题

MolHIV 上，保留完整 sparse-code 分布的 rich readout 明显优于只取少量 code
统计；而 luyin14 四数据集闭环只使用了每 atom 的 activation frequency、mean absolute
coefficient 与 RMS coefficient。本轮只检验：

> luyin14 的下游负结论是否部分来自 graph-level readout 过粗，而不是当前 patch space
> 与普通 K-SVD 完全没有任务信号。

这是跨数据集迁移/归因审计，不是重新搜索 IMDB readout。不得改变 sampler、patch size、
K/T、K-SVD iterations、folds、classifier 或节点特征定义。

## 2. 冻结数据与表示

- 数据集：`IMDB-BINARY / IMDB-MULTI / MUTAG / PTC_MR`。
- 完全复用 luyin14 的 FAIR95 patch、GLOBAL-WL stable order、rooted-canonical slots。
- `patch_size=8, overlap=2, Beam4/R1, K=24, T=3, T_min=1, iterations=5`。
- split seeds `0/1/2`，每个 seed 3-fold stratified CV。
- dictionary、centering mean、scaler、classifier 均为 outer-train-fold only。
- 线性 head 固定为 `StandardScaler + LogisticRegression(C=1, max_iter=5000)`。

## 3. 必报 readout

对每图 sparse code `X in R^(K x Npatch)`：

- `COARSE`：每 atom 的 usage、mean `|x|`、RMS，共 `3K` 维；
- `RICH_NO_RECON`：每 atom 的 mean/max/top-3 mean/std/usage/q75/q90 `|x|`、
  mean-square energy、signed mean、winner frequency，共 `10K` 维；
- `RICH`：在 `RICH_NO_RECON` 后追加逐 patch 相对重构误差的
  mean/std/q50/q75/q90/max，以及 patch count、`log1p(count)`，共 `10K+8` 维。

INIT 与 FINAL 必须使用相同 centering、相同 patch、相同 sparsity 分别编码。不得只报告
FINAL，因为本轮必须区分“高维统计本身”与“K-SVD updates 的增量”。

必报 feature sets：

- `STATS`
- `INIT_COARSE`, `FINAL_COARSE`
- `INIT_RICH_NO_RECON`, `FINAL_RICH_NO_RECON`
- `INIT_RICH`, `FINAL_RICH`
- `STATS_INIT_RICH`, `STATS_FINAL_RICH`

MUTAG/PTC_MR 额外报告：

- `FEATURE_ONLY`, `FEATURE_STATS`
- `FEATURE_INIT_RICH`, `FEATURE_FINAL_RICH`
- `FEATURE_STATS_INIT_RICH`, `FEATURE_STATS_FINAL_RICH`

## 4. 预注册判定

每个数据集的“稳定通过”同时要求 mean paired delta `>= +0.01` balanced accuracy，且
9 folds 中至少 `6/9` 为正。

1. **readout gate**：`FINAL_RICH - FINAL_COARSE` 至少 2/4 数据集稳定通过；
2. **KSVD-learning gate**：`FINAL_RICH - INIT_RICH` 至少 2/4 数据集稳定通过；
3. **added-value gate**：`STATS_FINAL_RICH - STATS` 至少 2/4 数据集稳定通过；带节点
   特征数据另看 `FEATURE_STATS_FINAL_RICH - FEATURE_STATS`，但不替代四数据集主 gate。

只有三个主 gate 同时通过，才说明 rich readout 是可迁移的普通 KSVD 后续路线。若只有
readout gate 通过、learning gate 失败，则信号来自初始化原型/高维统计，不归因于 K-SVD。
若 added-value gate 失败，则 rich 只是在重复低阶 graph statistics。若不足 2/4 数据集，
停止该方向，并把 MolHIV rich readout 保留为化学 patch space 下的特例证据。

## 5. 禁止项

- 不扫描 K/T、iterations、C、非线性 classifier、sampler 或 patch vector；
- 不依据单个数据集改 readout；
- 不进入 Transformer、token injection 或 gate；
- 不用 test fold 选择 readout 或阈值。
