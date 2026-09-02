# KSVD U0-P：无人工原子路线的 patch signal-exposure gate

> 日期：2026-07-31
>
> 正式结论：**PASS_U0P_WALK_SIGNAL_EXPOSED**

## 1. 本实验只回答什么

在训练任何字典之前，检验同一批 walk-induced 6-node patches 是否暴露了 LOW/HIGH rewiring regime 信号。
这不是 KSVD 实验，也不评价 atom；它用于避免在输入本身无信号时靠增加 restart 或字典复杂度补救。

三组真实 control 使用完全相同的每图 24 个 patches：

1. `walk_mean_std`：15 个 first-discovery-order adjacency 坐标的均值与标准差（30 维）；
2. `canonical_mean_std`：15 个 rooted-canonical adjacency 坐标的均值与标准差（30 维 baseline）；
3. `edge_count_histogram`：每 patch 的 induced edge count 5..15 频率（11 维）。

负对照只对 primary `walk_mean_std` 完整管线做一次固定 label shuffle。

## 2. 冻结实现口径

- master data seeds：`731101, 731102, 731103, 731104, 731105`；
- train / validation / test：每类 `150 / 50 / 100` 图；
- patches per graph：`24`；
- 图生成：60-node degree-4 ring lattice，LOW 20..40、HIGH 60..80 accepted connected double-edge swaps；
- patch：随机 root，随机游走首次发现 6 个节点，取这 6 个节点的完整 induced adjacency；
- 标准化只拟合 train；常数维删除；
- classifier：确定性 Newton solver 的 L2 logistic regression；
- 目标：mean binary cross entropy + `lambda/2 * ||w||^2`，intercept 不正则；
- lambda grid：`[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]`；validation 并列时选更大的 lambda；
- 每个候选只在 train 拟合，test 只在 validation 选定后评估一次。

## 3. 五个 replicate 的 test balanced accuracy

| seed | WALK mean/std | CAN mean/std | edge histogram | WALK label shuffle |
|---:|---:|---:|---:|---:|
| 731101 | 0.8350 | 0.8000 | 0.8400 | 0.5050 |
| 731102 | 0.8250 | 0.7850 | 0.8250 | 0.5250 |
| 731103 | 0.7700 | 0.7950 | 0.8300 | 0.4850 |
| 731104 | 0.8000 | 0.7800 | 0.8050 | 0.4900 |
| 731105 | 0.8450 | 0.8550 | 0.8300 | 0.5000 |

## 4. 聚合 gate

| control | mean | std | min | count >= 0.65 | real-control gate |
|---|---:|---:|---:|---:|---|
| `walk_mean_std` | 0.8150 | 0.0270 | 0.7700 | 5/5 | PASS |
| `canonical_mean_std` | 0.8030 | 0.0269 | 0.7800 | 5/5 | PASS |
| `edge_count_histogram` | 0.8260 | 0.0116 | 0.8050 | 5/5 | PASS |

Label shuffle mean = `0.5010` (range `0.4850`–`0.5250`): **PASS** for the registered `[0.45, 0.55]` gate.

Passing real controls：`['walk_mean_std', 'canonical_mean_std', 'edge_count_histogram']`。

## 5. 结论与下一步

**PASS_U0P_WALK_SIGNAL_EXPOSED**

Proceed to U0-D with walk first-discovery-order adjacency as the primary 15-D KSVD signal; retain INIT versus FINAL and all registered controls.

本结论只说明 patch pipeline 暴露了可线性读出的 regime 信息。它不说明：

- KSVD 一定能保留或增强该信息；
- canonical adjacency 的欧氏几何问题已经消失；
- learned atoms 必须是合法或可命名的图；
- KSVD 会超过 edge histogram 或简单全局图统计。

## 6. 每个 seed 的 validation 选择

| seed | control | selected lambda | train BA | validation BA | test BA | active dim |
|---:|---|---:|---:|---:|---:|---:|
| 731101 | `walk_mean_std` | 0.1 | 0.8400 | 0.8200 | 0.8350 | 28 |
| 731101 | `canonical_mean_std` | 1 | 0.7533 | 0.8100 | 0.8000 | 26 |
| 731101 | `edge_count_histogram` | 10 | 0.8100 | 0.8400 | 0.8400 | 5 |
| 731101 | `label_shuffle` | 0.01 | 0.6167 | 0.5900 | 0.5050 | 28 |
| 731102 | `walk_mean_std` | 0.01 | 0.8233 | 0.8900 | 0.8250 | 28 |
| 731102 | `canonical_mean_std` | 1 | 0.7233 | 0.8300 | 0.7850 | 26 |
| 731102 | `edge_count_histogram` | 0.1 | 0.7900 | 0.8400 | 0.8250 | 5 |
| 731102 | `label_shuffle` | 0.001 | 0.6233 | 0.5200 | 0.5250 | 28 |
| 731103 | `walk_mean_std` | 0.1 | 0.8600 | 0.8600 | 0.7700 | 28 |
| 731103 | `canonical_mean_std` | 0.1 | 0.8167 | 0.9000 | 0.7950 | 26 |
| 731103 | `edge_count_histogram` | 0.01 | 0.8633 | 0.9100 | 0.8300 | 5 |
| 731103 | `label_shuffle` | 10 | 0.5533 | 0.4700 | 0.4850 | 28 |
| 731104 | `walk_mean_std` | 0.001 | 0.8833 | 0.8200 | 0.8000 | 28 |
| 731104 | `canonical_mean_std` | 0.01 | 0.8633 | 0.8600 | 0.7800 | 26 |
| 731104 | `edge_count_histogram` | 0.01 | 0.8733 | 0.8400 | 0.8050 | 6 |
| 731104 | `label_shuffle` | 0.001 | 0.5900 | 0.5400 | 0.4900 | 28 |
| 731105 | `walk_mean_std` | 0.01 | 0.8267 | 0.8300 | 0.8450 | 28 |
| 731105 | `canonical_mean_std` | 0.01 | 0.8467 | 0.8400 | 0.8550 | 26 |
| 731105 | `edge_count_histogram` | 1 | 0.8267 | 0.8300 | 0.8300 | 6 |
| 731105 | `label_shuffle` | 10 | 0.5300 | 0.5300 | 0.5000 | 28 |
