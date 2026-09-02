# KSVD 从零路线：IMDB-BINARY R0-X objective-alignment diagnosis

> 日期：2026-08-01  
> 状态：**诊断结果不可见前冻结**  
> 角色：不是新的模型或 benchmark gate；只解释为什么 R0-D reconstruction 通过，而 R0-A/R0-B downstream attribution 失败。

## 1. 本轮不改变任何学习过程

直接复用 R0-D 已保存的两个 raw views、5 folds、train-coordinate mean 和 INIT/FINAL dictionaries：

```text
R0-D outer split seed = 731301
patch sampling seed = 20260731
patch size = 7
K = 12
T = 2
T_min = 1
KSVD updates = 25
restarts = 0
```

本轮不重新训练字典，不选择 K/T/iterations，不尝试新 readout，不产生新的 classification claim。所有量都由冻结字典重新编码得到。

## 2. 要区分的四种失败来源

### H1：objective mismatch

KSVD 降低常见 patch 的 reconstruction error，但 FINAL graph codes 在控制 graph statistics 后，没有比 INIT 更稳定的 label effect。

### H2：statistics redundancy

FINAL graph codes 主要可由节点数、边数、密度、度分布、triangle/transitivity、路径长度等简单 graph statistics 解释，因此无法为 `STATS+FINAL` 提供条件增量。

### H3：frequency domination

总 reconstruction gain 主要来自高频 edge-count/topology bins；稀有但可能有标签意义的 patches 对 Frobenius objective 权重太小。

### H4：readout bottleneck

FINAL 在控制 STATS 后确实增强了跨 train/test 一致的 label effect，但固定 marginal/pair readout 没能转化为分类收益。只有这种结果才支持继续复杂 readout/跨-patch relation。

## 3. Graph-level quantities

对每个 graph 和 INIT/FINAL 分别计算：

1. 36-D marginal graph code：activation frequency、mean absolute coefficient、RMS；
2. graph-balanced relative reconstruction error；
3. reconstruction gain：`INIT error - FINAL error`；
4. 12-D frozen simple graph statistics；
5. 42-D raw WALK mean/std summary；
6. graph label。

所有 regressions 只在 outer train 拟合，再应用到 held-out outer test。

## 4. Statistics redundancy diagnosis

用带 intercept 的 train least squares 分别拟合：

```text
STATS -> INIT graph code
STATS -> FINAL graph code
STATS -> raw WALK summary
STATS -> per-graph reconstruction gain
```

输出 held-out variance-weighted explained fraction：

\[
R^2_{heldout}=1-\frac{\sum\|Y-\hat Y\|^2}{\sum\|Y-\bar Y_{train}\|^2}.
\]

只做固定线性解释，不扫描 ridge penalty。若 FINAL code 的 held-out explained fraction 高、或显著高于 INIT，说明 KSVD update 更强地编码了 STATS 已知的 nuisance/global information。

## 5. Label alignment before and after STATS

对 INIT、FINAL、raw WALK 三种 graph representations：

1. 用 outer-train 均值/标准差标准化；
2. 计算 train class-effect vector：`mean(class1)-mean(class0)`；
3. 在 outer test 计算同一 effect vector；
4. 报告 train/test effect cosine；
5. 报告 test effect 沿 train effect unit direction 的 signed projection。

然后先用 outer-train least squares 回归掉 STATS，再对 residual representation 重复上述计算。

这个 signed projection 不训练 classifier，也不选择 threshold。它只回答：

> train 中发现的 label-associated direction，到了 held-out graphs 是否保持同方向；FINAL 是否比 INIT 更强。

## 6. Reconstruction gain 的标签关联

对每图 reconstruction gain 报告：

- class 0/1 mean；
- pooled Cohen's d；
- train/test effect sign consistency；
- STATS 对 gain 的 held-out explained fraction。

如果 reconstruction gain 在两个类别几乎相同，说明优化成功本身不具有 task selectivity。

## 7. Patch-frequency allocation

按原始 patch edge count `6..21` 分 bin，分别在 train/test 统计：

- patch count/mass；
- INIT 和 FINAL squared residual；
- 每 patch mean error reduction；
- bin 对总正 reconstruction gain 的 contribution share；
- top-3 frequency bins 的 patch mass 与 gain contribution；
- bin mass 与 total positive gain 的 Pearson correlation。

这不是 motif 可解释性 gate，只检查 Frobenius objective 是否主要奖励常见 topology bins。

## 8. 预先冻结的 routing rules

R0-X 不给 KSVD 新的 PASS/FAIL，而是选择下一研究路线。

### 支持继续 readout/relations 的必要条件

必须同时满足：

1. residual FINAL label projection 大于 residual INIT，在 stratified 至少 `4/5` folds；
2. grouped 至少 `3/5` folds；
3. 两个 views 的 mean residual FINAL−INIT projection 都为正。

否则，不能把 R0-A/R0-B 失败主要归因于 readout；优先判为 objective/task alignment 不足。

### 支持 statistics-residual/conditional KSVD

若 FINAL code 的 mean held-out `STATS -> code` explained fraction 高于 INIT，且 residual FINAL label projection没有增强，则优先研究去除 STATS nuisance 后的 conditional/residual objective。

### 支持 frequency-reweighted KSVD

若 top-3 frequency bins 同时承担大部分 patch mass 和 reconstruction gain，而长尾 bins 的 per-patch gain弱，则 frequency weighting 是可解释的 objective repair 候选；仍需新协议，不能在当前结果上调权重。

### 支持 task-aware dictionary

若 reconstruction gain label effect弱、residual FINAL label projection不增强，并且不是单一 stats redundancy/frequency domination可以解释，则普通 reconstruction objective 与分类目标错位。继续分类路线必须明确进入 label-conditioned/task-aware dictionary，不再称为普通无监督 KSVD 自动发现 task-optimal atoms。

## 9. 解释边界

- 本诊断可以定位机制，不能产生新的 IMDB benchmark accuracy claim；
- 使用 outer-test labels 仅计算固定 effect consistency，不选择模型或超参数；
- 不根据结果回改 routing rules；
- 节点属性尚未加入。只有纯结构失败机制明确后，才决定属性应作为独立 view、条件变量还是 task signal，不能直接与 adjacency 拼接后重跑。

## 10. Execution record（协议冻结后填写）

完整诊断已于 2026-08-01 执行，直接复用 R0-D frozen dictionaries，结果见：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0X_ALIGNMENT_DIAGNOSIS_20260801.md`
- `tracks/ksvd/results/from_scratch/imdb_binary_r0x_alignment_diagnosis_20260801.json`

本节只登记产物位置，不修改第 1–9 节的 routing rules。
