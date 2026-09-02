# IMDB-BINARY per-graph dictionary falsification protocol

> 日期：2026-08-01  
> 状态：在运行正式 outer-test 前冻结  
> 目标：检验“每张图单独做 KSVD，再读取字典/稀疏码统计”是否提供超出简单图统计、原始 patch 统计、同一初始化与 per-graph PCA 的稳定增量。

## 1. 科学问题

本实验不检验数据集共享 motif vocabulary，而检验另一条独立假设：

> 一张图内部 WALK patch cloud 的 sparse factorization statistics，能否构成有额外任务价值的 graph descriptor？

对每张图 `G` 单独拟合：

\[
Y_G \approx D_G X_G.
\]

因此结论边界必须是 `per-graph factorization descriptor`，不能解释成跨图共享 atoms。

## 2. 为什么先不直接复刻动态 ego

第一轮固定当前已经审计过的 raw IMDB WALK substrate：

```text
patch size = 7
每图 patches = min(n_nodes, 24)
signal = first-discovery-order induced adjacency upper triangle，21-D
sampling seed = 20260731
```

只改变 dictionary scope：

```text
outer-fold shared dictionary
→ per-graph dictionary
```

不同时改变为动态 ego、padding 和未知 `transformNetwork`，否则无法判断结果来自 dictionary scope 还是 sampler。

## 3. 冻结的 per-graph factorization

每张图只使用自己的 patches，不使用图标签或其他图：

```text
K = 8
T = 2
T_min = 1
KSVD updates = 10
INIT = deterministic maximin real-patch initialization
FINAL = 从完全相同 INIT 做 10 次 KSVD updates
PCA = 同图 patch matrix 的 rank-8 SVD/PCA basis
restarts = 0
centering = none
```

不做 `K/T/updates` outer-test scan。

## 4. 两种 readout

### 4.1 Legacy ordered readout（只作历史敏感性）

近似导师脚本：

```text
D: per-atom mean/std/max
X: per-atom abs-mean/std/abs-max
Gram: X X^T / N 的上三角
```

它依赖 atom index，不是 permutation-invariant，因此不作为主结论 readout。

### 4.2 Primary invariant atom-set readout

每个 atom 先计算：

```text
usage frequency
coefficient abs-mean / RMS / abs-max
coefficient energy share
dictionary abs-mean / std / abs-max
```

然后使用 atom-set 的 `mean/std/min/max`，并加入：

```text
sorted usage/mean-abs/RMS/energy spectra
eigenvalues(X X^T / N)
eigenvalues(D^T D)
dictionary/code absolute off-diagonal summaries
relative reconstruction error
mean active atoms per patch
```

该 readout 必须在同步置换 `D` 的列和 `X` 的行后保持数值不变。

## 5. 强基线与 feature ladder

`STATS`：冻结的 12-D 简单图统计。  
`RAW`：WALK coordinate mean/std + edge-count histogram。  
`INIT/FINAL/PCA`：上述 invariant per-graph descriptors。

主比较：

```text
STATS
STATS + RAW
STATS + RAW + INIT
STATS + RAW + FINAL
STATS + RAW + PCA
```

描述性比较：

```text
INIT / FINAL standalone
STATS + INIT / FINAL
STATS + RAW + legacy INIT / FINAL
legacy FINAL after independent per-graph atom permutations
```

## 6. 评估协议

数据：完整 raw IMDB-BINARY 1000 graphs；cleaned 不参与主实验。

两个 view：

1. raw stratified 5-fold；
2. raw exact-isomorphism-grouped 5-fold。

新 seeds：

```text
outer split = 732101
inner split = 732111 + fold
graph-code shuffle = 732121 + view/fold offset
label shuffle = 732131 + view/fold offset
atom permutation = 732141 + graph index
```

分类器沿用冻结的 L2 logistic protocol：只在 inner validation 选择正则，outer test 不参与选择。

## 7. 主归因与 controls

Primary update gain：

\[
\mathrm{BA}(STATS+RAW+FINAL)-\mathrm{BA}(STATS+RAW+INIT).
\]

Secondary PCA contrast：

\[
\mathrm{BA}(STATS+RAW+FINAL)-\mathrm{BA}(STATS+RAW+PCA).
\]

Controls：

- invariant readout atom-permutation numerical match；
- legacy readout atom-permutation sensitivity；
- graph-descriptor row shuffle；
- label shuffle；
- INIT→FINAL per-graph reconstruction reduction；
- graph-level descriptor finite/nonconstant audit。

## 8. 冻结 gate

主 stratified view 同时满足才支持该路线：

1. invariant readout permutation maximum absolute difference `<= 1e-8`；
2. 5/5 folds 的 mean per-graph reconstruction error 均为 `FINAL < INIT`；
3. `STATS+RAW+FINAL − STATS+RAW+INIT` 至少 4/5 folds 为正；
4. mean update gain `>= +0.01 BA`；
5. aligned FINAL 的 mean BA 不低于 row-shuffled FINAL；
6. label-shuffle mean BA 位于 `[0.45, 0.55]`。

PCA contrast 和 grouped view 为机制/稳健性证据，不单独挽救 primary failure。

## 9. 可能结论

- **PASS**：per-graph KSVD updates 产生超出 INIT、RAW 和 STATS 的稳定 graph-descriptor 增量；仍不代表共享 motif atoms。
- **RECON_ONLY**：FINAL 重建更好，但任务增量失败；说明 per-graph sparse factorization 健康但 readout/task 不受益。
- **READOUT_INVALID**：invariant gate 或 shuffle control 失败，不能解释 task result。
- **FAIL**：连 per-graph reconstruction 或 descriptor stability 都不成立。
