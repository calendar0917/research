# KSVD 从零路线：IMDB-BINARY R0-B atom-pair readout 协议

> 日期：2026-08-01  
> 状态：**结果不可见前冻结**  
> 前置结论：R0-D reconstruction 通过；R0-A 简单 marginal bag-of-codes readout 未通过 raw/stratified incremental task utility gate。  
> 本轮唯一问题：R0-A 的失败是否来自 readout 丢掉了 atoms 的组合使用信息，而不是 dictionary optimization 本身没有 task signal。

## 1. 为什么只改 readout

R0-A 对每个 atom 只保留：

- activation frequency；
- mean absolute coefficient；
- RMS coefficient。

这个 readout 只描述每个 atom 的 marginal usage，不描述两个 learned atoms 是否经常在同一个 patch 中共同出现。由于当前 sparse code 的 `T=2`，最小的组合扩展就是 atom-pair co-activation。

本轮不引入 attention、MIL、GNN 或人工 motif vocabulary。pair 的含义完全由 KSVD 自己产生的 atom index 决定，不预先指定任何合法 graphlet 或语义名称。

这里的 pair relation 是 **同一个 patch 内的 code-support relation**，不是声称已经解决跨 patch 的空间/顺序关系。若本轮仍失败，不能把它解释成所有 patch relation 都被否定。

## 2. 只改变的一项

保持 R0-A 不变：

```text
raw IMDB-BINARY
WALK first-discovery-order induced adjacency
patch_size = 7
dimension = 21
patches/graph = min(n_nodes, 24)
sampling_seed = 20260731
K = 12
T = 2
T_min = 1
KSVD updates = 25
one deterministic maximin INIT
restarts = 0
```

唯一改变：graph-level readout 增加 pair co-activation。

R0-B 使用新 outer split seed：

```text
outer split seed = 731501
inner split seed rule = 731511 + outer_fold_index
```

不复用 R0-A 的 `731401` outer test folds，也不扫描额外 split seeds。

## 3. Pair readout 定义

对图 `g` 的 patch code matrix `A in R^(K x m_g)`，定义 active support：

\[
S_{kj}=\mathbf{1}(|A_{kj}|>10^{-10}).
\]

对每一对 `k<l` 汇聚：

\[
C^{(g)}_{kl}
=\frac{1}{m_g}\sum_{j=1}^{m_g}S_{kj}S_{lj}.
\]

`K=12` 时 pair dimension 为：

\[
\binom{12}{2}=66.
\]

本轮的主 graph feature 是：

```text
STATS
+ R0-A marginal readout of INIT/FINAL (36-D)
+ pair co-activation readout of INIT/FINAL (66-D)
```

因此主比较为：

\[
\Delta_{pair}
=BA(\mathrm{STATS+MARGINAL+PAIR\_FINAL})
-BA(\mathrm{STATS+MARGINAL+PAIR\_INIT}).
\]

同时报告：

\[
\Delta_{pair\ added}
=BA(\mathrm{STATS+MARGINAL+PAIR\_FINAL})
-BA(\mathrm{STATS+MARGINAL\_FINAL}).
\]

第二个量检验 pair readout 是否真的补充了 R0-A 的 marginal readout，而不是只因为换了一次 outer split。

## 4. 数据视图与 inner validation

运行完整 raw 数据的两个 views：

1. raw/stratified；
2. raw/exact-isomorphism-grouped。

每个 view 5 个 outer folds。grouped view 的 exact-isomorphism groups 在 outer 和 inner split 都必须完整保留。

每个 outer train 内部继续使用 deterministic inner 5-fold 的 fold 0 作为 validation：

- classifier standardization 只 fit inner train；
- L2 logistic lambda 只由 validation balanced accuracy 选择；
- outer test 只评估一次；
- 所有 feature sets 共用同一个 inner split。

## 5. Controls

### 5.1 Marginal reference

保留 R0-A 的：

- `STATS`；
- `STATS+MARGINAL_INIT`；
- `STATS+MARGINAL_FINAL`。

### 5.2 Pair primary controls

- `STATS+MARGINAL+PAIR_INIT`；
- `STATS+MARGINAL+PAIR_FINAL`；
- `STATS+MARGINAL+PAIR_SHUFFLED_FINAL`。

shuffle 在每个 train/validation/test split 内独立打乱整行的 FINAL graph representation，保持 marginal 与 pair 的正确对应关系同时被破坏；STATS 与 labels 保持原顺序。

### 5.3 Label shuffle

对 `STATS+MARGINAL+PAIR_FINAL` 的 labels 在每个 split 内独立 deterministic shuffle，仅作泄漏 sanity check。

冻结 base seeds：

```text
graph-code shuffle base seed = 731521
label shuffle base seed = 731531
```

## 6. Registered gate

### 6.1 raw/stratified primary gate

必须同时满足：

1. `Delta_pair > 0` 至少 `4/5` folds；
2. mean `Delta_pair >= 0.02`；
3. mean `Delta_pair_added > 0`；
4. mean `BA(correctly aligned FINAL pair representation) - BA(shuffled FINAL pair representation) > 0`；
5. label-shuffle mean test BA 位于 `[0.45,0.55]`。

### 6.2 raw/exact-isomorphism-grouped mechanism gate

必须同时满足：

1. `Delta_pair >= 0` 至少 `3/5` folds；
2. mean `Delta_pair > 0`；
3. mean `Delta_pair_added >= 0`；
4. correctly aligned FINAL pair representation beats shuffled representation；
5. label-shuffle mean test BA 位于 `[0.45,0.55]`。

### 6.3 Classification

- 两个 view 都通过：`PASS_R0B_PAIR_READOUT_UTILITY`；
- stratified 通过、grouped 失败：`FAIL_R0B_GROUPED_GENERALIZATION`；
- stratified 失败：`FAIL_R0B_PAIR_READOUT_UTILITY`；
- permutation 或 label sanity check 失败：`FAIL_R0B_CONTROL_INTEGRITY`。

R0-B 不因为 R0-A 已失败而降低门槛；它只有在新 outer folds 上证明 pair readout 的增量，才算通过。

## 7. 结果解释边界

如果通过，可以说：

> 在固定的无监督 KSVD basis 上，patch 内 learned-atom 的组合使用信息能够通过简单 pair readout 提供超出 marginal readout 的稳定图级 task utility，并且在 exact-isomorphism-grouped view 仍保持方向一致。

仍不能说：

- 已经表达了跨 patch 的空间关系；
- 每个 atom 或 pair 都有人工可命名语义；
- KSVD 超过公开 benchmark 方法；
- pair readout 一定适用于其他数据集。

如果失败：

1. 不增加 restart；
2. 不扫描 K/T/iterations；
3. 不把 pair threshold、pair normalization、pair subset 在 outer test 上调参；
4. 将结论定位为：在当前 patch support relation 也不足以把 reconstruction gain 转成稳定 task gain；
5. 下一步若仍继续，必须另开协议研究真正的跨 patch relation 或 task/objective mismatch，而不能继续堆 readout 特征。

## 8. Execution record（协议冻结后填写）

完整注册运行已于 2026-08-01 执行，结果见：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0B_PAIR_READOUT_AUDIT_20260801.md`
- `tracks/ksvd/results/from_scratch/imdb_binary_r0b_pair_readout_audit_20260801.json`

本节只登记产物位置，不修改第 1–7 节的 readout、seed、gate 或解释规则。
