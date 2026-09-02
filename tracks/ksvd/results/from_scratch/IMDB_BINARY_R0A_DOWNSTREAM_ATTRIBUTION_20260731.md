# IMDB-BINARY R0-A downstream incremental attribution

> 日期：2026-07-31  
> 数据：完整 raw IMDB-BINARY（1000 图）  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0A_PROTOCOL_20260731.md`

## 1. Frozen design

- outer split seed：`731401`；5 folds；不扫描额外 split seeds。
- raw/stratified 是 reference view；raw/exact-isomorphism-grouped 是 mechanism view。
- patch：s=7，d=21，每图 `min(n,24)`；sampling seed `20260731`。
- K=12，T=2，T_min=1，updates=25，deterministic INIT，restart=0。
- graph code：每 atom activation frequency + mean absolute coefficient + RMS，共 36 维。
- classifier：inner-train standardization + L2 logistic；validation 只选择冻结 lambda grid。
- 主归因：`BA(STATS+FINAL)-BA(STATS+INIT)`；不是 FINAL standalone score。

## 2. Split audit

### raw/stratified

- outer partition gate：`PASS`；group integrity required：`False`；group leakage events：`315`。
- outer test folds：fold 0 n=200 class={'0': 100, '1': 100}, fold 1 n=200 class={'0': 100, '1': 100}, fold 2 n=200 class={'0': 100, '1': 100}, fold 3 n=200 class={'0': 100, '1': 100}, fold 4 n=200 class={'0': 100, '1': 100}。
- selected inner validation sizes：fold 0 160 class={'0': 80, '1': 80}, fold 1 160 class={'0': 80, '1': 80}, fold 2 160 class={'0': 80, '1': 80}, fold 3 160 class={'0': 80, '1': 80}, fold 4 160 class={'0': 80, '1': 80}。

### raw/exact-isomorphism-grouped

- outer partition gate：`PASS`；group integrity required：`True`；group leakage events：`0`。
- outer test folds：fold 0 n=200 class={'0': 100, '1': 100}, fold 1 n=200 class={'0': 100, '1': 100}, fold 2 n=200 class={'0': 100, '1': 100}, fold 3 n=200 class={'0': 100, '1': 100}, fold 4 n=200 class={'0': 100, '1': 100}。
- selected inner validation sizes：fold 0 160 class={'0': 80, '1': 80}, fold 1 160 class={'0': 80, '1': 80}, fold 2 160 class={'0': 80, '1': 80}, fold 3 160 class={'0': 80, '1': 80}, fold 4 160 class={'0': 80, '1': 80}。

## 3. Fold-level primary attribution

### raw/stratified

| fold | STATS | STATS+INIT | STATS+FINAL | update gain | beyond STATS | shuffled FINAL | FINAL-shuffle | label shuffle | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6800 | 0.6450 | 0.6900 | 0.0450 | 0.0100 | 0.6550 | 0.0350 | 0.4950 | 13.5 |
| 1 | 0.7050 | 0.7100 | 0.6650 | -0.0450 | -0.0400 | 0.7050 | -0.0400 | 0.5350 | 14.2 |
| 2 | 0.7200 | 0.6650 | 0.6800 | 0.0150 | -0.0400 | 0.7000 | -0.0200 | 0.4400 | 13.2 |
| 3 | 0.7200 | 0.7150 | 0.7000 | -0.0150 | -0.0200 | 0.6800 | 0.0200 | 0.4400 | 14.0 |
| 4 | 0.6950 | 0.6650 | 0.6550 | -0.0100 | -0.0400 | 0.6850 | -0.0300 | 0.4750 | 13.2 |

- update direction count：`2/5`（required `4`）；
- mean update gain：`-0.0020`；
- mean beyond-STATS gain：`-0.0260`；
- mean correct-alignment over shuffle：`-0.0070`；
- mean label-shuffle BA：`0.4770`；
- registered view gate：`FAIL`。

### raw/exact-isomorphism-grouped

| fold | STATS | STATS+INIT | STATS+FINAL | update gain | beyond STATS | shuffled FINAL | FINAL-shuffle | label shuffle | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6250 | 0.6150 | 0.5900 | -0.0250 | -0.0350 | 0.5900 | 0.0000 | 0.5100 | 13.9 |
| 1 | 0.5700 | 0.6000 | 0.6200 | 0.0200 | 0.0500 | 0.6150 | 0.0050 | 0.5050 | 13.7 |
| 2 | 0.6850 | 0.6600 | 0.6900 | 0.0300 | 0.0050 | 0.6600 | 0.0300 | 0.4950 | 13.7 |
| 3 | 0.6400 | 0.6450 | 0.6200 | -0.0250 | -0.0200 | 0.6000 | 0.0200 | 0.4450 | 13.6 |
| 4 | 0.5950 | 0.6000 | 0.6300 | 0.0300 | 0.0350 | 0.5800 | 0.0500 | 0.4700 | 12.8 |

- update direction count：`3/5`（required `3`）；
- mean update gain：`0.0060`；
- mean beyond-STATS gain：`0.0070`；
- mean correct-alignment over shuffle：`0.0210`；
- mean label-shuffle BA：`0.4850`；
- registered view gate：`PASS`。

## 4. Secondary controls

这些 control 用于定位表示/读出层，不改变 registered gate。

| view | INIT | FINAL | STATS+RAW | STATS+PCA | STATS+MEDOID | STATS+GAUSSIAN |
|---|---:|---:|---:|---:|---:|---:|
| stratified | 0.6150 | 0.6010 | 0.6840 | 0.6850 | 0.6840 | 0.7030 |
| exact_isomorphism_grouped | 0.6060 | 0.6130 | 0.6350 | 0.6350 | 0.6170 | 0.6200 |

## 5. Registered decision

> **FAIL_R0A_CONTROL_INTEGRITY**

- control integrity：`FAIL`；
- stratified gate：`FAIL`；
- grouped gate：`PASS`；
- next step：Stop attribution claims and inspect only the failed shuffle/control implementation; do not tune K/T/iterations/readout.

## 6. Interpretation boundary

R0-A 只判断固定简单 graph-code readout 下的 incremental task utility。它不等价于公开 benchmark 的公平模型比较，也不要求每个 atom 可命名。若失败，不能用增加随机初始化、扫描 K/T/iterations 或同时修改 sampler/objective/readout 来补救当前 outer-test 结果。

## 7. Post-hoc control implementation check（不改变 gate）

由于注册 classification 使用了 `FAIL_R0A_CONTROL_INTEGRITY` 这一名称，额外检查了 shuffle 是否实现错误或近似 identity：

- graph-code 与 label permutation 都是 deterministic、split-local permutation；
- 每个 split 的 row/label multiset 保持不变；
- 10 个 view-fold 中，train/validation/test permutation 的 fixed points 仅为 `0..4`，相对于 `640/160/200` 的 split size，不是 identity 或近似 identity；
- label-shuffle mean BA 为 stratified `0.477`、grouped `0.485`，均通过 `[0.45,0.55]` sanity range。

因此这里的 “control integrity fail” **不是代码没有真正打乱**，而是注册 gate 中的 substantive condition 失败：在 stratified view，正确对齐的 `STATS+FINAL` mean BA `0.678` 反而低于 `STATS+SHUFFLED_FINAL` 的 `0.685`。即使完全忽略这一 control，stratified 的主 gate 也已经因 update `2/5`、mean update `-0.002`、mean beyond-STATS `-0.026` 而失败。
