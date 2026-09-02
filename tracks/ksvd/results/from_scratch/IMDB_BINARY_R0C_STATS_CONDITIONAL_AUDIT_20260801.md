# IMDB-BINARY R0-C statistics-conditioned residual KSVD audit

> 日期：2026-08-01  
> 数据：完整 raw IMDB-BINARY（1000 图）  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0C_STATS_CONDITIONAL_PROTOCOL_20260801.md`

## 1. Frozen design

- outer split seed：`731601`；5 folds；不扫描额外 split seeds。
- dictionary target：raw patch minus outer-train graph-statistics-predicted graph patch mean。
- residualizer：graph-balanced outer-train multivariate least squares；不使用 labels。
- patch：s=7，d=21，每图 `min(n,24)`；sampling seed `20260731`。
- K=12，T=2，T_min=1，updates=25，每 branch deterministic INIT，restart=0。
- readout：36-D activation frequency + mean absolute + RMS。

## 2. Fold-level attribution

### raw/stratified

| fold | residual recon reduction | STATS | standard FINAL | residual INIT | residual FINAL | residual update | beyond STATS | over standard | shuffle | label shuffle | nondead | max share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3781 | 0.7050 | 0.6800 | 0.7150 | 0.6600 | -0.0550 | -0.0450 | -0.0200 | 0.6450 | 0.4500 | 12/12 | 0.1965 |
| 1 | 0.3407 | 0.7150 | 0.6800 | 0.6700 | 0.7200 | 0.0500 | 0.0050 | 0.0400 | 0.6650 | 0.5000 | 12/12 | 0.2567 |
| 2 | 0.3759 | 0.7100 | 0.6600 | 0.7050 | 0.7100 | 0.0050 | 0.0000 | 0.0500 | 0.7000 | 0.4800 | 12/12 | 0.2332 |
| 3 | 0.3787 | 0.7350 | 0.6850 | 0.7300 | 0.6650 | -0.0650 | -0.0700 | -0.0200 | 0.6700 | 0.5100 | 12/12 | 0.2656 |
| 4 | 0.3374 | 0.7100 | 0.7000 | 0.7200 | 0.7050 | -0.0150 | -0.0050 | 0.0050 | 0.6650 | 0.4450 | 12/12 | 0.2815 |

- residual reconstruction：positive `5/5`；mean reduction `0.3622`；minimum nondead `12/12`；maximum share `0.2815`。
- mean residual update：`-0.0160`；direction `2/5`；
- mean beyond-STATS：`-0.0230`；mean over standard FINAL：`0.0110`；
- mean alignment gain over shuffle：`0.0230`；label shuffle BA：`0.4770`；
- mean full-patch test error：predictor-only `0.4013`；standard INIT/FINAL `0.3254/0.1732`；residual INIT/FINAL `0.2687/0.1761`。
- registered view gate：`FAIL`。

### raw/exact-isomorphism-grouped

| fold | residual recon reduction | STATS | standard FINAL | residual INIT | residual FINAL | residual update | beyond STATS | over standard | shuffle | label shuffle | nondead | max share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4157 | 0.5100 | 0.7400 | 0.5000 | 0.7250 | 0.2250 | 0.2150 | -0.0150 | 0.5150 | 0.4400 | 12/12 | 0.2495 |
| 1 | 0.4179 | 0.6350 | 0.6850 | 0.6450 | 0.6650 | 0.0200 | 0.0300 | -0.0200 | 0.6250 | 0.4500 | 12/12 | 0.2425 |
| 2 | 0.3593 | 0.6700 | 0.6650 | 0.7200 | 0.6050 | -0.1150 | -0.0650 | -0.0600 | 0.6500 | 0.5250 | 12/12 | 0.2278 |
| 3 | 0.3297 | 0.5900 | 0.5700 | 0.5500 | 0.6050 | 0.0550 | 0.0150 | 0.0350 | 0.5650 | 0.4950 | 12/12 | 0.1924 |
| 4 | 0.3371 | 0.7450 | 0.6200 | 0.6100 | 0.6050 | -0.0050 | -0.1400 | -0.0150 | 0.7800 | 0.4150 | 12/12 | 0.2804 |

- residual reconstruction：positive `5/5`；mean reduction `0.3719`；minimum nondead `12/12`；maximum share `0.2804`。
- mean residual update：`0.0360`；direction `3/5`；
- mean beyond-STATS：`0.0110`；mean over standard FINAL：`-0.0150`；
- mean alignment gain over shuffle：`0.0140`；label shuffle BA：`0.4650`；
- mean full-patch test error：predictor-only `0.4044`；standard INIT/FINAL `0.3177/0.1740`；residual INIT/FINAL `0.2723/0.1767`。
- registered view gate：`FAIL`。

## 3. Registered decision

> **FAIL_R0C_STATS_CONDITIONAL_UTILITY**

- controls：`PASS`；
- residual dictionary gate：`PASS`；
- stratified gate：`FAIL`；
- grouped gate：`FAIL`；
- next step：Statistics-conditioned residual target did not establish primary raw/stratified utility; stop unsupervised task-utility expansion.

## 4. Interpretation boundary

R0-C 只测试 statistics-conditioned residual target。通过不等于已经证明 labels 参与字典学习是必要的；失败也不等于节点属性路线必然失败。若失败，按照冻结协议停止普通无监督 KSVD 的 IMDB task-utility 扩展。

## 5. Result interpretation

R0-C 清楚地区分了两个问题：

1. **Residual dictionary optimization 成功。** 两个 views 都是 `5/5` folds 改善 residual held-out reconstruction，平均 reduction 为 `0.362/0.372`；每 fold `12/12` non-dead，maximum usage share 约 `0.28`。因此失败不能归因于 residual target 无法被 KSVD 学习。
2. **Residual reconstruction improvement 仍未稳定转化为 task utility。** Stratified 的 residual FINAL−INIT mean BA 为 `-0.016`、仅 `2/5` 为正；grouped 虽为 `+0.036`、`3/5` 非负，但 residual FINAL 平均仍比同 fold standard FINAL 低 `0.015`，且 grouped fold 0 的 `+0.225` 造成明显均值敏感性。

完整 patch reconstruction 也没有显示 conditional FINAL 优于 standard FINAL：

| view | predictor only | standard INIT | standard FINAL | residual INIT | residual FINAL |
|---|---:|---:|---:|---:|---:|
| stratified | 0.4013 | 0.3254 | 0.1732 | 0.2687 | 0.1761 |
| grouped | 0.4044 | 0.3177 | 0.1740 | 0.2723 | 0.1767 |

Residualization 改善了 INIT 的 full-patch starting point，但 25-update FINAL 与 standard FINAL 收敛到几乎相同的 reconstruction level，conditional FINAL 还略差。这意味着“先移除 STATS-predictable graph patch mean”改变了优化路径，却没有创造稳定的额外 label-residual information。

Controls 正常：correct alignment over shuffle 为 `+0.023/+0.014`，label shuffle 为 `0.477/0.465`。所以总分类：

> **FAIL_R0C_STATS_CONDITIONAL_UTILITY**

按照预冻结停止规则，raw IMDB 上普通无监督 reconstruction-KSVD 的 task-utility 扩展到此停止。后续不能继续扫描 residualizer、frequency weights、K/T/restarts 或 bag readout。可以保留并继续研究的命题是 sparse patch compressor/basis discovery；如果分类价值是必须目标，则需要另立 label-conditioned/task-aware dictionary learning 命题。
