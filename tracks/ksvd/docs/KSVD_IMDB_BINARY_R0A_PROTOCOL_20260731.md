# KSVD 从零路线：IMDB-BINARY R0-A 下游增量归因协议

> 日期：2026-07-31  
> 状态：**结果不可见前冻结（registered before execution）**  
> 前置结果：R0-P substrate/signal 与 R0-D dictionary optimization 已通过。  
> 本轮唯一问题：KSVD update 学到的 FINAL graph code，是否比同一 fold、同一表示、同一读出的 INIT graph code 提供稳定的分类增量。

## 1. 为什么现在不能只看 reconstruction

R0-D 已经证明 FINAL 比 INIT 更能重建 held-out WALK patches，但这仍是 patch-level objective。它不能推出：

- 图级分类更好；
- FINAL 比 INIT 含有更多 label-relevant information；
- graph code 能补充简单全局统计；
- learned atoms 必须可被人工命名。

因此 R0-A 不再以 reconstruction error 为主指标，也不把 FINAL standalone accuracy 当成 KSVD update 的归因证据。主比较必须在同一 outer fold 中进行：

\[
\Delta_{update}
=
BA(\mathrm{STATS+FINAL})-BA(\mathrm{STATS+INIT}).
\]

这回答的是：保持 sampler、signal、K、T、初始化方式、readout 和 classifier 不变，只执行 KSVD updates 是否改善下游信息。

## 2. 数据视图与 outer split

第一轮只运行完整 raw IMDB-BINARY，不运行 cleaned：

1. **raw/stratified**：benchmark/reference view；
2. **raw/exact-isomorphism-grouped**：同构组完整、测试未见整体结构的 mechanism/generalization view。

冻结：

```text
outer folds = 5
outer split seed = 731401
patch sampling seed = 20260731
cleaned = not run in first R0-A
```

R0-P 已看过 `731201..731203`，R0-D 使用 `731301`；R0-A 使用新的 `731401`。本轮不扫描额外 split seeds，也不从多个划分中挑最好结果。五个 outer folds 是预先登记的 paired checks，而不是五次随机初始化。

## 3. 每个 outer fold 中允许学习什么

冻结的无监督表示：

```text
patch = 7-node WALK first-discovery-order induced adjacency
patch dimension d = 21
patches/graph = min(n_nodes, 24)
centering = outer-train coordinate mean
K = 12
T = 2
T_min = 1
KSVD updates = 25
INIT = one deterministic maximin initialization
restarts = 0
```

对每个 outer fold：

1. 仅用 outer train patches 拟合 coordinate mean；
2. 仅用 outer train 构造 deterministic INIT；
3. 仅用 outer train 从 INIT 训练 FINAL；
4. Gaussian、medoid 与 PCA controls 也只由 outer train 定义；
5. outer test labels 和 patches 不参与 centering、dictionary、PCA、medoid、readout 或正则选择；
6. patch cache 可全数据预先生成，因为 sampler 固定、label-free，且不跨图拟合参数。

## 4. 图级 readout：本轮只允许最简单的固定汇聚

IMDB 每图 patch 数可变。对一张图的 code matrix `A in R^(K x m_g)`，每个 atom 只汇聚三项：

1. activation frequency：`mean(|a_kj| > 1e-10)`；
2. mean absolute coefficient：`mean(|a_kj|)`；
3. RMS coefficient：`sqrt(mean(a_kj^2))`。

拼接后 graph code 为 `3K=36` 维。

本轮禁止增加：

- atom co-occurrence；
- code histogram bins；
- attention/MIL；
- patch-to-patch relation graph；
- learned nonlinear readout；
- 根据 test 表现修改 threshold 或 pooling。

理由不是这些机制无价值，而是当前先回答最小问题：**最简单的 bag-of-codes readout 是否已经能暴露 KSVD update 的增量。** 如果失败，之后最多一次只修 readout 这一层。

## 5. Inner train/validation 与 classifier

每个 outer train 再固定划为 inner-train/validation：

```text
inner folds = 5
validation = deterministic inner fold 0
inner seed = 731411 + outer_fold_index
```

- stratified view：inner split 按 label stratified；
- grouped view：inner split 继续保持 exact-isomorphism groups 完整；
- 不根据任何 feature/control 的 validation score 选择 validation fold；
- 所有 feature sets 共用同一个 inner split。

classifier 沿用 R0-P 的 package-independent L2 logistic regression：

```text
lambda grid = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
standardization = fit on inner train only
selection metric = validation balanced accuracy
validation tie = choose larger lambda
reported outer-test model = selected inner-train model
```

不搜索 classifier depth、hidden size、dropout、learning rate 或 readout architecture。这样做刻意保守：本轮测试 representation attribution，不追求 IMDB leaderboard 最优分数。

## 6. Feature sets

### 6.1 Primary registered comparisons

- `STATS`：冻结的 12-D simple graph statistics；
- `STATS+INIT`；
- `STATS+FINAL`。

主要报告：

\[
\Delta_{update}=BA(\mathrm{STATS+FINAL})-BA(\mathrm{STATS+INIT})
\]

以及必要但不充分的：

\[
\Delta_{beyond\_stats}=BA(\mathrm{STATS+FINAL})-BA(\mathrm{STATS}).
\]

### 6.2 Secondary controls

- `INIT`；
- `FINAL`；
- `STATS+RAW_WALK`：R0-P 的 42-D WALK mean/std；
- `STATS+PCA12`：对 dense PCA coefficients 使用同一 3K readout；
- `STATS+MEDOID_BAG`：nearest-real-medoid one-hot assignment 的同一 readout；
- `STATS+GAUSSIAN`：固定 Gaussian dictionary 的 T=2 codes。

这些 controls 用于定位失败层，不取代主比较。尤其 PCA reconstruction 更好不意味着其图级 readout 必然更好。

## 7. Negative controls

### 7.1 Conditional graph-code shuffle

在 train、validation、test 各自内部独立打乱 FINAL graph-code rows，保留 STATS 与 labels 原顺序，评估：

```text
STATS + SHUFFLED_FINAL
```

它检验条件增量是否依赖 graph 与其 FINAL code 的正确对应，而不是单纯增加 36 个维度。

### 7.2 Label shuffle

在 train、validation、test 各自内部独立打乱 labels，评估原始：

```text
STATS + FINAL
```

label shuffle 只作实现/泄漏 sanity check，不用于选择任何参数。

冻结 shuffle seed：

```text
graph-code shuffle base seed = 731421
label shuffle base seed = 731431
```

每个 view/fold 从对应 base seed 派生确定性子 seed；不重复 shuffle 后挑选结果。

## 8. 预注册 gate

### 8.1 Primary gate：raw/stratified

必须同时满足：

1. `Delta_update > 0` 至少 `4/5` folds；
2. mean `Delta_update >= 0.02` balanced accuracy；
3. mean `Delta_beyond_stats > 0`；
4. `STATS+SHUFFLED_FINAL` 不得同时满足前两项所定义的 update-level 增量模式；具体用 paired mean
   `BA(STATS+FINAL)-BA(STATS+SHUFFLED_FINAL) > 0` 检查正确 graph-code 对应优于 shuffle；
5. label-shuffle mean test BA 位于 `[0.45, 0.55]`。

### 8.2 Mechanism gate：raw/exact-isomorphism-grouped

必须同时满足：

1. `Delta_update >= 0` 至少 `3/5` folds；
2. mean `Delta_update > 0`；
3. mean `Delta_beyond_stats` 不得为负；
4. paired mean `BA(STATS+FINAL)-BA(STATS+SHUFFLED_FINAL) > 0`；
5. label-shuffle mean test BA 位于 `[0.45, 0.55]`。

### 8.3 Classification

- 两个 gate 都通过：`PASS_R0A_INCREMENTAL_TASK_UTILITY`；
- stratified 通过、grouped 失败：`FAIL_R0A_GROUPED_GENERALIZATION`；
- stratified 失败：`FAIL_R0A_INCREMENTAL_TASK_UTILITY`；
- negative control 失败：`FAIL_R0A_CONTROL_INTEGRITY`。

这里 deliberately 要求 FINAL 同时优于 INIT 且不低于 STATS：只优于一个很弱的 INIT、但仍不能补充 STATS，不足以声称 task-useful discovery。

## 9. 结果出来以后允许和不允许的解释

如果通过，可以说：

> 在完整 raw IMDB-BINARY 上，无人工 atom vocabulary、单 deterministic initialization 的 KSVD updates，通过固定简单 graph-code readout 提供了超出 INIT 且不低于简单 graph statistics 的稳定分类增量；该方向在 exact-isomorphism-grouped view 仍成立。

仍不能说：

- 达到或超过当前 SOTA；
- 与所有公开 IMDB 方法完成公平比较；
- atoms 都有可命名语义；
- WALK 是唯一正确 sampler；
- patch relations 已解决。

如果失败：

1. 不增加 restart；
2. 不在相同 outer-test folds 上扫描 K/T/iterations；
3. 不同时改 sampler、objective 与 readout；
4. 先依据 controls 判断是：无条件 label signal 不足、STATS 已吸收信息，还是简单 readout 丢失 co-occurrence/relations；
5. 若只允许下一次修复，优先将 **readout** 作为单一轴，因为 R0-D 已证明 dictionary optimization 本身健康，而 `luyin11` 明确指出 patch 间关系尚未表达。

## 10. 本轮的科学位置

R0-A 不是 benchmark terminal experiment，而是从：

```text
patch reconstruction works
```

推进到：

```text
KSVD updates expose additional graph-level task information
```

的最小桥梁。只有这一步通过，才值得另开、重新冻结一个与公开方法公平比较的 benchmark protocol；届时仍使用 raw IMDB，并明确报告 standard split、grouped sensitivity、超参数预算与 cleaned sensitivity，而不是用 cleaned 替代 raw。

## 11. Execution record（协议冻结后填写）

完整注册运行已于 2026-07-31 执行，结果见：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0A_DOWNSTREAM_ATTRIBUTION_20260731.md`
- `tracks/ksvd/results/from_scratch/imdb_binary_r0a_downstream_attribution_20260731.json`

本节只登记产物位置，不修改第 1–10 节的 feature、seed、gate 或解释规则。
