# IMDB-BINARY R0-D：raw WALK dictionary optimization / health audit

> 日期：2026-07-31
>
> 结论：**PASS_R0D_REAL_DICTIONARY_OPTIMIZATION**

## 1. 本轮问题与边界

本轮只检验：同一个 outer-train fold 上的一次 deterministic INIT，经普通 KSVD 更新后，是否稳定改善 held-out sparse reconstruction，并保持字典不坍缩。

- 数据：完整 raw IMDB-BINARY；cleaned 未进入本轮。
- 两个视图：普通 stratified 5-fold 与 exact-isomorphism-grouped 5-fold。
- 表示：7-node WALK first-discovery-order adjacency，21 维。
- 每图 `min(n,24)` patches，sampling seed `20260731`。
- `K=12, T=2, T_min=1, updates=25`，每 fold 仅一个 deterministic maximin INIT，restart=0。
- centering、INIT、KSVD、PCA 和 medoid 均只用 outer train。
- atom 是否可命名不构成 gate；本轮也不使用 graph labels 学字典。

## 2. Split 审计

### raw/stratified

- partition gate：`PASS`；
- exact-group integrity required：`False`；
- fold-group leakage count：`300`；
- test folds：fold 0 n=200 class={'0': 100, '1': 100} groups=143, fold 1 n=200 class={'0': 100, '1': 100} groups=138, fold 2 n=200 class={'0': 100, '1': 100} groups=145, fold 3 n=200 class={'0': 100, '1': 100} groups=159, fold 4 n=200 class={'0': 100, '1': 100} groups=148。

### raw/exact-isomorphism-grouped

- partition gate：`PASS`；
- exact-group integrity required：`True`；
- fold-group leakage count：`0`；
- test folds：fold 0 n=200 class={'0': 100, '1': 100} groups=104, fold 1 n=200 class={'0': 100, '1': 100} groups=107, fold 2 n=200 class={'0': 100, '1': 100} groups=108, fold 3 n=200 class={'0': 100, '1': 100} groups=109, fold 4 n=200 class={'0': 100, '1': 100} groups=109。

fold-group leakage count 表示某个 exact group 同时出现在一个 fold 的 train/test 的事件数，不是泄漏图数；stratified 不以此为 gate，grouped 必须为 0。

## 3. INIT vs FINAL held-out reconstruction

主误差是先对每张 held-out 图计算 patch-matrix relative Frobenius error，再对图等权平均。

### raw/stratified

| fold | train/test graphs | train/test patches | INIT test GB err | FINAL test GB err | relative reduction | nondead | max usage share | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 800/200 | 14018/3508 | 0.7006 | 0.3542 | 0.4944 | 12/12 | 0.2273 | 14.2 |
| 1 | 800/200 | 14106/3420 | 0.6817 | 0.3621 | 0.4688 | 12/12 | 0.2816 | 13.6 |
| 2 | 800/200 | 13989/3537 | 0.7303 | 0.3531 | 0.5166 | 12/12 | 0.2871 | 13.7 |
| 3 | 800/200 | 14029/3497 | 0.6809 | 0.3938 | 0.4217 | 12/12 | 0.2534 | 14.9 |
| 4 | 800/200 | 13962/3564 | 0.7191 | 0.3654 | 0.4919 | 12/12 | 0.2306 | 14.3 |

- positive folds：`5/5`；
- mean graph-balanced reduction：`0.4787`；
- mean patch-weighted reduction：`0.4102`；
- FINAL mean train/test graph-balanced error：`0.3645` / `0.3657`，gap `0.0012`；
- minimum nondead atoms：`12/12`；
- maximum atom activation share：`0.2871`；
- registered view gate：`PASS`。

### raw/exact-isomorphism-grouped

| fold | train/test graphs | train/test patches | INIT test GB err | FINAL test GB err | relative reduction | nondead | max usage share | sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 800/200 | 14227/3299 | 0.6863 | 0.3322 | 0.5160 | 12/12 | 0.2607 | 13.5 |
| 1 | 800/200 | 14063/3463 | 0.6788 | 0.3848 | 0.4331 | 12/12 | 0.2120 | 13.4 |
| 2 | 800/200 | 14107/3419 | 0.7168 | 0.3987 | 0.4439 | 12/12 | 0.2397 | 14.0 |
| 3 | 800/200 | 14124/3402 | 0.6561 | 0.3534 | 0.4614 | 12/12 | 0.2491 | 14.0 |
| 4 | 800/200 | 13583/3943 | 0.6662 | 0.3063 | 0.5402 | 12/12 | 0.2697 | 13.2 |

- positive folds：`5/5`；
- mean graph-balanced reduction：`0.4789`；
- mean patch-weighted reduction：`0.4063`；
- FINAL mean train/test graph-balanced error：`0.3546` / `0.3551`，gap `0.0005`；
- minimum nondead atoms：`12/12`；
- maximum atom activation share：`0.2697`；
- registered view gate：`PASS`。

## 4. Reconstruction controls

PCA-12 是非稀疏 rank-12 reconstruction baseline；Gaussian、medoid、INIT、FINAL 均以同一 T=2 OMP 评估。medoid 控制从 INIT 的真实训练 patch 出发，以 deterministic Lloyd real-patch medoid steps 细化，不做超参数搜索。

### stratified

| fold | Gaussian | Medoid | INIT | FINAL | PCA-12 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8599 | 0.4609 | 0.7006 | 0.3542 | 0.1840 |
| 1 | 0.8601 | 0.3684 | 0.6817 | 0.3621 | 0.1703 |
| 2 | 0.8588 | 0.4299 | 0.7303 | 0.3531 | 0.1813 |
| 3 | 0.8558 | 0.4675 | 0.6809 | 0.3938 | 0.2018 |
| 4 | 0.8573 | 0.3995 | 0.7191 | 0.3654 | 0.1842 |

### exact_isomorphism_grouped

| fold | Gaussian | Medoid | INIT | FINAL | PCA-12 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8597 | 0.3700 | 0.6863 | 0.3322 | 0.1706 |
| 1 | 0.8551 | 0.4286 | 0.6788 | 0.3848 | 0.1918 |
| 2 | 0.8596 | 0.4577 | 0.7168 | 0.3987 | 0.2137 |
| 3 | 0.8560 | 0.4072 | 0.6561 | 0.3534 | 0.1894 |
| 4 | 0.8615 | 0.3448 | 0.6662 | 0.3063 | 0.1561 |

## 5. 跨 fold 字典描述性稳定性

matched atom cosine 与 principal-subspace cosine 只作描述，不用于选择 fold、restart 或模型。

| view | stage | matched cosine mean | minimum pair mean | subspace cosine mean |
|---|---|---:|---:|---:|
| stratified | init | 0.5357 | 0.4812 | 0.7633 |
| stratified | final | 0.7529 | 0.6876 | 0.8964 |
| exact_isomorphism_grouped | init | 0.5614 | 0.5041 | 0.8034 |
| exact_isomorphism_grouped | final | 0.7458 | 0.6959 | 0.8817 |

## 6. 冻结 gate 判定

1. 每个 view 至少 4/5 folds 的 graph-balanced reduction 为正；
2. 每个 view 的 mean reduction 至少 10%；
3. 每 fold 至少 10/12 non-dead atoms；
4. 每 fold maximum single-atom activation share 不超过 0.60；
5. grouped view 不得系统性反向失败。

最终判定：**PASS_R0D_REAL_DICTIONARY_OPTIMIZATION**。

下一步：Freeze these dictionaries/readouts and design R0-A around STATS+INIT versus STATS+FINAL.

无论通过或失败，本结果都不等价于下游分类收益；只有 R0-D 通过后，才允许在 R0-A 检验 `STATS+FINAL - STATS+INIT`。
