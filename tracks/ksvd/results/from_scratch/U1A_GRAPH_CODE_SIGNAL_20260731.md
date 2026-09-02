# KSVD U1A：无人工原子路线的 graph-code signal 与 added-value gate

> 日期：2026-07-31
>
> 正式结论：**PASS_GRAPH_CODE_SIGNAL_BUT_NOT_KSVD_ADDED_VALUE**

## 1. 本实验的归因问题

U0-P 已确认输入 patches 有 regime signal，U0-D 已确认单次初始化 KSVD 明显改善 held-out reconstruction。U1A 现在区分：

1. FINAL codes 是否包含图级预测信号；
2. 信号是否只是 raw patches、随机投影或 initializer 已经提供；
3. KSVD 的 25 次更新本身是否带来 paired downstream 增益。

字典训练仍不使用 graph labels。每个 atom 的 graph readout 固定为 activation frequency、mean absolute coefficient 和 coefficient RMS，共 36 维。

## 2. 五个 replicate 的 test balanced accuracy

| seed | simple stats | WALK raw | edge hist | Gaussian | medoid | PCA | INIT | FINAL | code shuffle | label shuffle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 731101 | 0.9300 | 0.8350 | 0.8400 | 0.7300 | 0.7000 | 0.8250 | 0.7600 | 0.7550 | 0.4650 | 0.5100 |
| 731102 | 0.9200 | 0.8250 | 0.8250 | 0.6550 | 0.6050 | 0.8050 | 0.7000 | 0.7650 | 0.4800 | 0.4900 |
| 731103 | 0.9450 | 0.7700 | 0.8300 | 0.7350 | 0.6800 | 0.7850 | 0.6800 | 0.7150 | 0.4350 | 0.5050 |
| 731104 | 0.9350 | 0.8000 | 0.8050 | 0.7350 | 0.6550 | 0.8300 | 0.7200 | 0.7100 | 0.5100 | 0.4500 |
| 731105 | 0.9450 | 0.8450 | 0.8300 | 0.6900 | 0.5500 | 0.8000 | 0.7050 | 0.8050 | 0.4850 | 0.5000 |

Rooted-canonical raw baseline 也被运行并保存在 JSON；表中省略一列仅为可读性。

## 3. 聚合结果

| control | mean | std | min | count >= 0.65 |
|---|---:|---:|---:|---:|
| `simple_graph_statistics` | 0.9350 | 0.0095 | 0.9200 | 5/5 |
| `walk_mean_std` | 0.8150 | 0.0270 | 0.7700 | 5/5 |
| `canonical_mean_std` | 0.8030 | 0.0269 | 0.7800 | 5/5 |
| `edge_count_histogram` | 0.8260 | 0.0116 | 0.8050 | 5/5 |
| `fixed_gaussian` | 0.7090 | 0.0318 | 0.6550 | 5/5 |
| `medoid_bag` | 0.6380 | 0.0543 | 0.5500 | 3/5 |
| `pca12` | 0.8090 | 0.0166 | 0.7850 | 5/5 |
| `init_codes` | 0.7130 | 0.0268 | 0.6800 | 5/5 |
| `final_codes` | 0.7500 | 0.0349 | 0.7100 | 5/5 |
| `graph_code_shuffle` | 0.4750 | 0.0247 | 0.4350 | 0/5 |
| `label_shuffle` | 0.4910 | 0.0215 | 0.4500 | 0/5 |

## 4. 分层判定

- Level 1 `PASS_KSVD_OPTIMIZATION`：**PASS**；
- Level 2 `PASS_GRAPH_CODE_SIGNAL`：**PASS**；
- Level 3 `PASS_KSVD_ADDED_VALUE`：**FAIL**。

FINAL − fixed Gaussian mean = `0.0410`。
FINAL − INIT mean = `0.0370`；FINAL 高于 INIT 的 replicate 数 = `3/5`。
Graph-code shuffle mean = `0.4750`；label shuffle mean = `0.4910`。

## 5. 结论

**PASS_GRAPH_CODE_SIGNAL_BUT_NOT_KSVD_ADDED_VALUE**

Do not attribute graph-level performance to KSVD updates.  The content/code chain is viable, but INIT versus FINAL evidence does not satisfy the registered added-value gate.

解释时必须把 reconstruction 与 downstream 分开：U0-D 的通过只证明 FINAL 是更好的稀疏重建 basis；只有 Level 3 才能说 KSVD update 对当前图级任务有 added value。

此外，simple stats 或 edge histogram 更强并不否定 atom 的统计存在，但说明当前生成任务主要由低阶结构统计决定，不能借此声称发现了不可替代的高阶图原子。

## 6. validation 选择明细

| seed | control | lambda | validation BA | test BA | active dim |
|---:|---|---:|---:|---:|---:|
| 731101 | `simple_graph_statistics` | 0.001 | 0.9200 | 0.9300 | 4 |
| 731101 | `walk_mean_std` | 0.1 | 0.8200 | 0.8350 | 28 |
| 731101 | `canonical_mean_std` | 1 | 0.8100 | 0.8000 | 26 |
| 731101 | `edge_count_histogram` | 10 | 0.8400 | 0.8400 | 5 |
| 731101 | `fixed_gaussian` | 0.1 | 0.8000 | 0.7300 | 36 |
| 731101 | `medoid_bag` | 10 | 0.6800 | 0.7000 | 36 |
| 731101 | `pca12` | 0.001 | 0.8400 | 0.8250 | 24 |
| 731101 | `init_codes` | 0.001 | 0.6800 | 0.7600 | 36 |
| 731101 | `final_codes` | 0.1 | 0.7500 | 0.7550 | 36 |
| 731101 | `graph_code_shuffle` | 0.001 | 0.5300 | 0.4650 | 36 |
| 731101 | `label_shuffle` | 0.01 | 0.4800 | 0.5100 | 36 |
| 731102 | `simple_graph_statistics` | 0.1 | 0.9500 | 0.9200 | 4 |
| 731102 | `walk_mean_std` | 0.01 | 0.8900 | 0.8250 | 28 |
| 731102 | `canonical_mean_std` | 1 | 0.8300 | 0.7850 | 26 |
| 731102 | `edge_count_histogram` | 0.1 | 0.8400 | 0.8250 | 5 |
| 731102 | `fixed_gaussian` | 10 | 0.7800 | 0.6550 | 36 |
| 731102 | `medoid_bag` | 0.1 | 0.6700 | 0.6050 | 36 |
| 731102 | `pca12` | 0.1 | 0.8600 | 0.8050 | 24 |
| 731102 | `init_codes` | 0.0001 | 0.8400 | 0.7000 | 36 |
| 731102 | `final_codes` | 0.001 | 0.8400 | 0.7650 | 36 |
| 731102 | `graph_code_shuffle` | 0.01 | 0.5200 | 0.4800 | 36 |
| 731102 | `label_shuffle` | 10 | 0.4900 | 0.4900 | 36 |
| 731103 | `simple_graph_statistics` | 0.01 | 0.9800 | 0.9450 | 4 |
| 731103 | `walk_mean_std` | 0.1 | 0.8600 | 0.7700 | 28 |
| 731103 | `canonical_mean_std` | 0.1 | 0.9000 | 0.7950 | 26 |
| 731103 | `edge_count_histogram` | 0.01 | 0.9100 | 0.8300 | 5 |
| 731103 | `fixed_gaussian` | 0.01 | 0.7900 | 0.7350 | 36 |
| 731103 | `medoid_bag` | 0.1 | 0.6600 | 0.6800 | 36 |
| 731103 | `pca12` | 0.1 | 0.8700 | 0.7850 | 24 |
| 731103 | `init_codes` | 0.1 | 0.7800 | 0.6800 | 36 |
| 731103 | `final_codes` | 0.01 | 0.7200 | 0.7150 | 36 |
| 731103 | `graph_code_shuffle` | 0.001 | 0.4400 | 0.4350 | 36 |
| 731103 | `label_shuffle` | 10 | 0.5500 | 0.5050 | 36 |
| 731104 | `simple_graph_statistics` | 0.1 | 0.9500 | 0.9350 | 4 |
| 731104 | `walk_mean_std` | 0.001 | 0.8200 | 0.8000 | 28 |
| 731104 | `canonical_mean_std` | 0.01 | 0.8600 | 0.7800 | 26 |
| 731104 | `edge_count_histogram` | 0.01 | 0.8400 | 0.8050 | 6 |
| 731104 | `fixed_gaussian` | 0.001 | 0.7600 | 0.7350 | 36 |
| 731104 | `medoid_bag` | 1 | 0.6600 | 0.6550 | 36 |
| 731104 | `pca12` | 0.0001 | 0.7900 | 0.8300 | 24 |
| 731104 | `init_codes` | 10 | 0.7700 | 0.7200 | 36 |
| 731104 | `final_codes` | 0.001 | 0.8100 | 0.7100 | 36 |
| 731104 | `graph_code_shuffle` | 10 | 0.4700 | 0.5100 | 36 |
| 731104 | `label_shuffle` | 10 | 0.5500 | 0.4500 | 36 |
| 731105 | `simple_graph_statistics` | 0.001 | 0.9200 | 0.9450 | 4 |
| 731105 | `walk_mean_std` | 0.01 | 0.8300 | 0.8450 | 28 |
| 731105 | `canonical_mean_std` | 0.01 | 0.8400 | 0.8550 | 26 |
| 731105 | `edge_count_histogram` | 1 | 0.8300 | 0.8300 | 6 |
| 731105 | `fixed_gaussian` | 0.001 | 0.7100 | 0.6900 | 36 |
| 731105 | `medoid_bag` | 0.01 | 0.5900 | 0.5500 | 36 |
| 731105 | `pca12` | 0.01 | 0.7900 | 0.8000 | 24 |
| 731105 | `init_codes` | 0.0001 | 0.7100 | 0.7050 | 36 |
| 731105 | `final_codes` | 0.1 | 0.7700 | 0.8050 | 36 |
| 731105 | `graph_code_shuffle` | 1 | 0.4700 | 0.4850 | 36 |
| 731105 | `label_shuffle` | 0.0001 | 0.5500 | 0.5000 | 36 |
