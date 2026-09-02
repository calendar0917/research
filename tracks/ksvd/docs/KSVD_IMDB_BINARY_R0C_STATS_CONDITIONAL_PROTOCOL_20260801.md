# KSVD 从零路线：IMDB-BINARY R0-C statistics-conditioned residual KSVD

> 日期：2026-08-01  
> 状态：**结果不可见前冻结**  
> 前置诊断：R0-X 显示 `STATS -> reconstruction gain` 的 held-out explained fraction 为 `0.737/0.672`，而控制 STATS 后 FINAL label direction 没有稳定增强。  
> 本轮唯一修改：dictionary target 从原始 centered patch 改为 outer-train statistics-conditioned residual patch。

## 1. 研究问题

普通 KSVD 优先重建与 graph size、density、degree、triangle/path statistics 一起变化的 patch distribution。R0-C 检验：

> 在不使用 graph labels 的前提下，先移除可由简单 graph statistics 预测的 patch-coordinate mean，KSVD 是否能在剩余 residual structure 上学到 held-out reconstruction 更好、且比 residual INIT 更有分类增量的 basis？

本轮不是 task-aware dictionary；residualizer 和 dictionary 都不使用 labels。

## 2. 保持不变的配置

```text
raw IMDB-BINARY
views = stratified + exact-isomorphism-grouped
outer folds = 5
patch = 7-node WALK first-discovery-order adjacency
patch dimension = 21
patches/graph = min(n_nodes, 24)
patch sampling seed = 20260731
K = 12
T = 2
T_min = 1
KSVD updates = 25
INIT = one deterministic maximin initialization
restarts = 0
readout = activation frequency + mean absolute + RMS (36-D)
classifier = frozen L2 logistic protocol
```

新 split seeds：

```text
outer split seed = 731601
inner split seed rule = 731611 + outer_fold_index
graph-code shuffle base seed = 731621
label shuffle base seed = 731631
```

不复用 R0-A/R0-B outer tests，不扫描其他 seeds。

## 3. Frozen statistics residualizer

对 outer-train graph `g` 的 12-D frozen graph statistics `z_g`：

1. 按 graph 等权计算 outer-train mean/std；
2. 删除 train std `<=1e-12` 的 inactive dimensions；
3. 添加 intercept；
4. 对每个 patch vector `x_gp in R^21` 拟合固定 multivariate least squares：

\[
\hat\mu_g = B[\operatorname{std}(z_g);1].
\]

同一张图的 patches 共用 `mu_g`。拟合采用 graph-balanced weights：图 `g` 的每个 patch 权重为 `1/m_g`，因此每张 outer-train graph 对 residualizer objective 的总权重相同。

Residual target：

\[
r_{gp}=x_{gp}-\hat\mu_g.
\]

约束：

- residualizer 只用 outer train；
- validation/test 只应用 outer-train coefficients；
- 不使用 labels；
- 不选择 ridge penalty；固定使用 SVD least squares；
- 不按结果修改 stats subset、weighting、nonlinearity 或 residual scale。

## 4. 两条 paired dictionary branches

每个 outer fold 同时训练：

### STANDARD reference

```text
x_gp - outer_train_coordinate_mean
-> deterministic maximin INIT
-> 25-update FINAL
```

### CONDITIONAL residual branch

```text
r_gp = x_gp - mu_hat_g
-> deterministic maximin RESIDUAL_INIT
-> 25-update RESIDUAL_FINAL
```

两条 branch 使用相同 `K/T/T_min/updates`，各自只有一个 deterministic INIT。STANDARD 不是用于调参，只是判断 conditional target 是否真的比普通 target 更适合 downstream。

## 5. Reconstruction audit

对 CONDITIONAL branch 报告：

1. residual-space graph-balanced reconstruction error；
2. residual INIT→FINAL relative reduction；
3. full-patch reconstruction：

\[
\hat x_{gp}=\hat\mu_g+D_r a_{gp};
\]

4. predictor-only full-patch error；
5. STANDARD INIT/FINAL full-patch error。

Residual dictionary gate：

- residual held-out graph-balanced INIT→FINAL reduction 至少 `4/5` folds 为正；
- mean reduction `>=10%`；
- 每 fold residual FINAL 至少 `10/12` non-dead atoms；
- 无 fold maximum activation share `>0.60`。

若 residual reconstruction 本身失败，不解释下游结果为 objective repair 成功。

## 6. Feature sets

Primary：

- `STATS`；
- `STATS+RESIDUAL_INIT`；
- `STATS+RESIDUAL_FINAL`。

References：

- `RESIDUAL_INIT`；
- `RESIDUAL_FINAL`；
- `STATS+STANDARD_INIT`；
- `STATS+STANDARD_FINAL`。

Main attribution：

\[
\Delta_{residual\ update}
=BA(STATS+RESIDUAL\_FINAL)-BA(STATS+RESIDUAL\_INIT).
\]

Conditional repair value：

\[
\Delta_{over\ standard}
=BA(STATS+RESIDUAL\_FINAL)-BA(STATS+STANDARD\_FINAL).
\]

## 7. Classifier isolation

每个 outer train 使用 deterministic inner 5-fold 的 fold 0 作为 validation：

- stratified inner split 按 label stratified；
- grouped inner split 保持 exact-isomorphism groups 完整；
- feature standardization 只 fit inner train；
- validation 只从固定 lambda grid 选择 L2 regularization；
- outer test 只评估一次；
- 所有 branches 共用同一 inner split。

Residualizer 和 dictionaries 使用整个 outer train，和 R0-A/R0-B 的无监督 feature-extractor isolation 保持一致。

## 8. Negative controls

- `STATS+SHUFFLED_RESIDUAL_FINAL`：每个 train/validation/test split 内独立打乱 residual FINAL graph-code rows；
- `LABEL_SHUFFLE(STATS+RESIDUAL_FINAL)`：每个 split 内独立打乱 labels；
- 每个 view/fold 只执行一个 deterministic shuffle。

## 9. Registered downstream gate

### raw/stratified primary gate

必须同时满足：

1. residual dictionary reconstruction gate 通过；
2. `Delta_residual_update > 0` 至少 `4/5` folds；
3. mean `Delta_residual_update >=0.02`；
4. mean `BA(STATS+RESIDUAL_FINAL)-BA(STATS) >0`；
5. mean `Delta_over_standard >0`；
6. correctly aligned residual FINAL mean BA 高于 shuffled residual FINAL；
7. label-shuffle mean BA 位于 `[0.45,0.55]`。

### raw/exact-isomorphism-grouped mechanism gate

必须同时满足：

1. residual dictionary reconstruction gate 通过；
2. `Delta_residual_update >=0` 至少 `3/5` folds；
3. mean `Delta_residual_update >0`；
4. mean beyond-STATS gain `>=0`；
5. mean `Delta_over_standard >=0`；
6. alignment control 为正；
7. label-shuffle mean BA 位于 `[0.45,0.55]`。

## 10. Classification

- 两个 view 都通过：`PASS_R0C_STATS_CONDITIONAL_UTILITY`；
- stratified 通过、grouped 失败：`FAIL_R0C_GROUPED_GENERALIZATION`；
- residual reconstruction 失败：`FAIL_R0C_RESIDUAL_DICTIONARY_OPTIMIZATION`；
- controls 失败：`FAIL_R0C_CONTROL_INTEGRITY`；
- 其他 primary failure：`FAIL_R0C_STATS_CONDITIONAL_UTILITY`。

## 11. 失败后的停止规则

如果 R0-C 失败：

1. 不扫描 stats subsets、ridge penalty、residual scaling；
2. 不增加 restart、K/T/updates；
3. 不回到更多 bag readout；
4. 普通无监督 reconstruction KSVD 的 raw IMDB task-utility branch 到此停止；
5. 后续只能二选一：定位为 compressor/basis discovery，或另开 label-conditioned task-aware dictionary 命题。

## 12. 节点属性边界

R0-C 尚不加入节点属性。如果 conditional structure route 失败，不能期待简单拼接属性自动修复 objective mismatch。属性数据应另行设计 structure/attribute multiview decomposition，并分别报告 attribute-only、structure-only 和 fusion 增量。

## 13. Execution record（协议冻结后填写）

完整注册运行已于 2026-08-01 执行，结果见：

- `tracks/ksvd/results/from_scratch/IMDB_BINARY_R0C_STATS_CONDITIONAL_AUDIT_20260801.md`
- `tracks/ksvd/results/from_scratch/imdb_binary_r0c_stats_conditional_audit_20260801.json`

实现阶段的单-fold debug smoke 曾暴露 dictionary 误用 inner-train 而非完整 outer-train 的 isolation bug；该 debug 结果未进入证据。修正后重新运行全部 10 个 folds，正式 JSON/报告只包含修正后的 outer-train dictionary fit。
