# IMDB-BINARY R1-B grouped consensus basis atlas

> 日期：2026-08-01  
> 性质：label-free descriptive atlas；无 PASS/FAIL gate  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_PROTOCOL_20260801.md`

## 1. Construction

- 固定 grouped fold 0 为 reference，不按结果选择。
- 其余 folds 用 exact maximum absolute cosine assignment 对齐，并按 dot-product 对齐 sign。
- 每 fold 每 atom 只取 grouped held-out patches 的 top-5 absolute activations；每个 consensus atom 共 25 exemplars。
- labels 只作 descriptive metadata，不参与匹配、排序或命名。

## 2. Atlas summary

- atom count：`12`；
- mean reference-matched cosine：`0.717`；
- worst atom/fold matched cosine：`0.102`；
- mean unique canonical signatures / 25 exemplars：`9.167`；
- mean canonical effective count：`6.657`；
- mean dominant canonical mass：`0.327`；
- atom exemplar edge-count mean range：`9.960..14.160`。

| atom | match mean/min | graphs | edge mean±std | unique WALK/canonical | effective canonical | dominant mass | labels descriptive |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.846/0.731 | 25 | 10.720±1.638 | 10/8 | 5.765 | 0.280 | {'0': 9, '1': 16} |
| 1 | 0.790/0.597 | 25 | 10.280±1.686 | 11/7 | 5.726 | 0.280 | {'0': 13, '1': 12} |
| 2 | 0.886/0.660 | 25 | 14.000±1.523 | 17/16 | 12.757 | 0.240 | {'0': 21, '1': 4} |
| 3 | 0.689/0.568 | 22 | 10.520±0.900 | 11/10 | 8.253 | 0.200 | {'0': 22, '1': 3} |
| 4 | 0.716/0.408 | 24 | 10.960±1.612 | 8/7 | 5.311 | 0.320 | {'0': 18, '1': 7} |
| 5 | 0.754/0.671 | 24 | 12.440±2.858 | 14/12 | 8.713 | 0.280 | {'0': 17, '1': 8} |
| 6 | 0.699/0.579 | 23 | 14.160±1.617 | 8/6 | 4.183 | 0.400 | {'0': 25} |
| 7 | 0.468/0.102 | 23 | 10.560±1.651 | 8/6 | 3.623 | 0.560 | {'0': 19, '1': 6} |
| 8 | 0.607/0.506 | 24 | 11.560±1.416 | 7/7 | 5.177 | 0.360 | {'0': 19, '1': 6} |
| 9 | 0.770/0.707 | 24 | 13.320±2.111 | 12/11 | 6.975 | 0.360 | {'0': 18, '1': 7} |
| 10 | 0.854/0.687 | 23 | 9.960±1.685 | 11/10 | 7.156 | 0.320 | {'0': 15, '1': 10} |
| 11 | 0.525/0.340 | 25 | 14.040±2.323 | 10/10 | 6.250 | 0.320 | {'0': 23, '1': 2} |

## 3. Canonical representatives

### Atom 0

edge histogram：`{'8': 1, '9': 6, '10': 7, '11': 3, '12': 1, '13': 7}`；continuous atom max coordinate：`0.414`。

- representative 1：mass `0.280`，edges `10`，edge list `[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [2, 6], [3, 4], [3, 5], [4, 5]]`
- representative 2：mass `0.280`，edges `13`，edge list `[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]`
- representative 3：mass `0.200`，edges `9`，edge list `[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 5], [3, 4]]`

### Atom 1

edge histogram：`{'8': 7, '9': 1, '10': 6, '12': 11}`；continuous atom max coordinate：`0.454`。

- representative 1：mass `0.280`，edges `12`，edge list `[[0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.240`，edges `8`，edge list `[[0, 6], [1, 6], [2, 5], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`
- representative 3：mass `0.160`，edges `12`，edge list `[[0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 2

edge histogram：`{'11': 2, '12': 2, '13': 5, '14': 7, '15': 3, '16': 6}`；continuous atom max coordinate：`0.391`。

- representative 1：mass `0.240`，edges `16`，edge list `[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.120`，edges `13`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.080`，edges `14`，edge list `[[0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 3

edge histogram：`{'9': 1, '10': 16, '11': 2, '12': 6}`；continuous atom max coordinate：`0.499`。

- representative 1：mass `0.200`，edges `12`，edge list `[[0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.200`，edges `10`，edge list `[[0, 6], [1, 5], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`
- representative 3：mass `0.160`，edges `10`，edge list `[[0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`

### Atom 4

edge histogram：`{'9': 5, '10': 9, '11': 1, '12': 3, '13': 6, '14': 1}`；continuous atom max coordinate：`0.472`。

- representative 1：mass `0.320`，edges `10`，edge list `[[0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.240`，edges `13`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.200`，edges `9`，edge list `[[0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 5

edge histogram：`{'9': 5, '10': 3, '11': 1, '12': 4, '13': 7, '15': 1, '17': 2, '18': 1, '19': 1}`；continuous atom max coordinate：`0.410`。

- representative 1：mass `0.280`，edges `13`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.200`，edges `9`，edge list `[[0, 5], [0, 6], [1, 4], [1, 6], [2, 3], [2, 6], [3, 6], [4, 6], [5, 6]]`
- representative 3：mass `0.120`，edges `12`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 6

edge histogram：`{'11': 1, '12': 1, '13': 12, '15': 1, '16': 10}`；continuous atom max coordinate：`0.429`。

- representative 1：mass `0.400`，edges `16`，edge list `[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.280`，edges `13`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.200`，edges `13`，edge list `[[0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`

### Atom 7

edge histogram：`{'9': 5, '10': 14, '12': 3, '13': 1, '15': 2}`；continuous atom max coordinate：`0.412`。

- representative 1：mass `0.560`，edges `10`，edge list `[[0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.200`，edges `9`，edge list `[[0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.120`，edges `12`，edge list `[[0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 8

edge histogram：`{'9': 2, '10': 7, '12': 7, '13': 9}`；continuous atom max coordinate：`0.386`。

- representative 1：mass `0.360`，edges `13`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.240`，edges `12`，edge list `[[0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.160`，edges `10`，edge list `[[0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [4, 6], [5, 6]]`

### Atom 9

edge histogram：`{'10': 1, '11': 4, '12': 8, '13': 3, '16': 9}`；continuous atom max coordinate：`0.395`。

- representative 1：mass `0.360`，edges `16`，edge list `[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 2：mass `0.240`，edges `12`，edge list `[[0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.080`，edges `11`，edge list `[[0, 4], [0, 5], [0, 6], [1, 5], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`

### Atom 10

edge histogram：`{'8': 8, '9': 3, '10': 4, '11': 3, '12': 6, '13': 1}`；continuous atom max coordinate：`0.454`。

- representative 1：mass `0.320`，edges `8`，edge list `[[0, 5], [0, 6], [1, 6], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`
- representative 2：mass `0.200`，edges `12`，edge list `[[0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 6], [2, 3], [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.120`，edges `10`，edge list `[[0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`

### Atom 11

edge histogram：`{'9': 1, '10': 1, '11': 2, '13': 10, '16': 8, '17': 2, '18': 1}`；continuous atom max coordinate：`0.610`。

- representative 1：mass `0.320`，edges `13`，edge list `[[0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 6], [2, 3], [2, 4], [2, 6], [3, 4], [3, 6], [4, 6], [5, 6]]`
- representative 2：mass `0.320`，edges `16`，edge list `[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`
- representative 3：mass `0.080`，edges `17`，edge list `[[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]`

## 4. Interpretation boundary

这些 atoms 是 continuous latent basis components。代表 patch 是 post-hoc exemplars，不表示 atom 本身等于某一张合法离散图；label composition 也不能作为 task utility 证据。
