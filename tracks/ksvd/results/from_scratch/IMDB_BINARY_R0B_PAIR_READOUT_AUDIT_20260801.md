# IMDB-BINARY R0-B atom-pair readout audit

> 日期：2026-08-01  
> 数据：完整 raw IMDB-BINARY（1000 图）  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0B_PAIR_READOUT_PROTOCOL_20260801.md`

## 1. Frozen design

- outer split seed：`731501`；5 folds；不扫描额外 split seeds。
- readout-only repair：在 R0-A 的 36-D marginal code 后增加 66-D within-patch atom-pair co-activation。
- patch：s=7，d=21，每图 `min(n,24)`；sampling seed `20260731`。
- K=12，T=2，T_min=1，updates=25，deterministic INIT，restart=0。
- 主归因：`STATS+MARGINAL+PAIR_FINAL` vs `STATS+MARGINAL+PAIR_INIT`。

## 2. Split audit

### raw/stratified

- outer partition gate：`PASS`；group integrity required：`False`；group leakage events：`291`。
- outer test folds：fold 0 n=200 class={'0': 100, '1': 100}, fold 1 n=200 class={'0': 100, '1': 100}, fold 2 n=200 class={'0': 100, '1': 100}, fold 3 n=200 class={'0': 100, '1': 100}, fold 4 n=200 class={'0': 100, '1': 100}。
- selected inner validation sizes：fold 0 160 class={'0': 80, '1': 80}, fold 1 160 class={'0': 80, '1': 80}, fold 2 160 class={'0': 80, '1': 80}, fold 3 160 class={'0': 80, '1': 80}, fold 4 160 class={'0': 80, '1': 80}。

### raw/exact-isomorphism-grouped

- outer partition gate：`PASS`；group integrity required：`True`；group leakage events：`0`。
- outer test folds：fold 0 n=200 class={'0': 100, '1': 100}, fold 1 n=200 class={'0': 100, '1': 100}, fold 2 n=200 class={'0': 100, '1': 100}, fold 3 n=200 class={'0': 100, '1': 100}, fold 4 n=200 class={'0': 100, '1': 100}。
- selected inner validation sizes：fold 0 160 class={'0': 80, '1': 80}, fold 1 160 class={'0': 80, '1': 80}, fold 2 160 class={'0': 80, '1': 80}, fold 3 160 class={'0': 80, '1': 80}, fold 4 160 class={'0': 80, '1': 80}。

## 3. Fold-level primary attribution

### raw/stratified

| fold | STATS+M+INIT | STATS+M+FINAL | STATS+M+P+INIT | STATS+M+P+FINAL | pair update | pair added | shuffled pair | FINALpair-shuffle | label shuffle | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6450 | 0.7200 | 0.6700 | 0.7050 | 0.0350 | -0.0150 | 0.6700 | 0.0350 | 0.4750 | 13.2 |
| 1 | 0.7100 | 0.6850 | 0.6900 | 0.6700 | -0.0200 | -0.0150 | 0.6500 | 0.0200 | 0.5750 | 13.5 |
| 2 | 0.6800 | 0.7350 | 0.6350 | 0.6600 | 0.0250 | -0.0750 | 0.6550 | 0.0050 | 0.5000 | 13.7 |
| 3 | 0.6650 | 0.6300 | 0.6600 | 0.6500 | -0.0100 | 0.0200 | 0.6250 | 0.0250 | 0.5200 | 13.7 |
| 4 | 0.7300 | 0.7200 | 0.6700 | 0.7150 | 0.0450 | -0.0050 | 0.6850 | 0.0300 | 0.5750 | 14.5 |

- pair update direction count：`3/5`（required `4`）；
- mean pair update gain：`0.0150`；
- mean pair added value over marginal FINAL：`-0.0180`；
- mean correct-alignment over shuffle：`0.0230`；
- mean label-shuffle BA：`0.5290`；
- registered view gate：`FAIL`。

### raw/exact-isomorphism-grouped

| fold | STATS+M+INIT | STATS+M+FINAL | STATS+M+P+INIT | STATS+M+P+FINAL | pair update | pair added | shuffled pair | FINALpair-shuffle | label shuffle | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7700 | 0.7750 | 0.7600 | 0.7250 | -0.0350 | -0.0500 | 0.6200 | 0.1050 | 0.4950 | 15.1 |
| 1 | 0.7150 | 0.6850 | 0.7400 | 0.7100 | -0.0300 | 0.0250 | 0.6450 | 0.0650 | 0.4850 | 13.8 |
| 2 | 0.6950 | 0.6600 | 0.6300 | 0.6000 | -0.0300 | -0.0600 | 0.5950 | 0.0050 | 0.5200 | 13.8 |
| 3 | 0.5850 | 0.6050 | 0.5400 | 0.6100 | 0.0700 | 0.0050 | 0.5900 | 0.0200 | 0.5200 | 14.0 |
| 4 | 0.5500 | 0.6200 | 0.5650 | 0.5950 | 0.0300 | -0.0250 | 0.5400 | 0.0550 | 0.4550 | 13.5 |

- pair update direction count：`2/5`（required `3`）；
- mean pair update gain：`0.0010`；
- mean pair added value over marginal FINAL：`-0.0210`；
- mean correct-alignment over shuffle：`0.0500`；
- mean label-shuffle BA：`0.4950`；
- registered view gate：`FAIL`。

## 4. Marginal reference

| view | STATS | STATS+MARGINAL INIT | STATS+MARGINAL FINAL | STATS+M+PAIR INIT | STATS+M+PAIR FINAL |
|---|---:|---:|---:|---:|---:|
| stratified | 0.7070 | 0.6860 | 0.6980 | 0.6650 | 0.6800 |
| exact_isomorphism_grouped | 0.6310 | 0.6630 | 0.6690 | 0.6470 | 0.6480 |

## 5. Registered decision

> **FAIL_R0B_PAIR_READOUT_UTILITY**

- control integrity：`PASS`；
- stratified gate：`FAIL`；
- grouped gate：`FAIL`；
- next step：Conclude that within-patch atom-pair co-activation did not provide the registered stable bridge; do not add more readout statistics on these outer tests.

## 6. Interpretation boundary

R0-B 只测试同一 patch 内 learned atoms 的 support-composition readout。通过不等于已经表达跨 patch 的空间关系；失败也不等于所有 relation readout 都不可能。无论结果如何，不在本轮 outer tests 上继续调 pair threshold、K/T、iterations 或 restart。

## 7. Result interpretation

本轮 controls 正常：两个 view 的 correctly aligned representation 都超过 shuffled representation，label shuffle 也位于注册区间。因此失败不是 permutation 实现问题，而是 pair readout 的主归因与 added-value 条件没有成立。

最关键的两个观察是：

1. pair update 虽然在 stratified 平均为 `+0.015`，但只有 `3/5` folds 为正且低于注册的 `+0.02`；grouped 只有 `2/5` 非负、平均 `+0.001`；
2. 加入 pair 后，FINAL 相对 marginal FINAL 的 mean BA 在 stratified 为 `-0.018`、grouped 为 `-0.021`。也就是说，66-D support co-activation 没有补充旧的 36-D marginal readout，平均反而降低泛化分数。

在新 split seed 上，marginal reference 本身表现为：stratified `STATS=0.707, STATS+FINAL=0.698`，仍未超过 STATS；grouped `STATS=0.631, STATS+FINAL=0.669`，但 FINAL−INIT 只有 `+0.006`。这与 R0-A 的总体边界一致：grouped 中可能存在弱方向，但没有跨 views 建立稳定的 KSVD-update task utility。

因此本轮支持的结论不是“pair 完全没有结构信息”：correct alignment 确实优于 whole-code shuffle。更准确的是：**这种同 patch support pair 信息没有在强 marginal/STATS 条件下形成稳定的额外分类价值。**

一个 post-hoc、只用于定位失败形态的观察是：pair 特征更像增加了拟合容量，而不是改善泛化。stratified 的 FINAL marginal mean train/test BA 为 `0.742/0.698`，加入 pair 后为 `0.773/0.680`；grouped 对应为 `0.708/0.669` 与 `0.722/0.648`。两个 view 都是 train score 上升、test score 下降，因此不应继续在当前 bag readout 上堆更多高维统计量。
