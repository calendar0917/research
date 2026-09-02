# IMDB-BINARY R0-X objective-alignment diagnosis

> 日期：2026-08-01  
> 类型：复用 R0-D frozen dictionaries 的机制诊断，不是新的分类 benchmark。  
> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R0X_OBJECTIVE_ALIGNMENT_PROTOCOL_20260801.md`

## 1. Frozen source

- source audit：`tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json`；不重新训练字典。
- patch sampling seed：`20260731`；dictionary split seed：`731301`。
- 检查：STATS redundancy、label residual alignment、reconstruction gain label effect、patch-frequency allocation。

## 2. Fold-level diagnosis

### raw/stratified

| fold | INIT code R2 | FINAL code R2 | gain R2 | residual FINAL−INIT label projection | INIT residual cosine | FINAL residual cosine | gain Cohen d | top3 patch mass | top3 gain share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3608 | 0.4416 | 0.7372 | -0.3017 | 0.5085 | 0.1204 | 0.1156 | 0.6662 | 0.6267 |
| 1 | 0.4251 | 0.4114 | 0.6749 | -0.0091 | 0.4192 | 0.3291 | 0.1161 | 0.6924 | 0.6616 |
| 2 | 0.2988 | 0.3667 | 0.7955 | 0.1769 | -0.0565 | 0.1872 | 0.1270 | 0.6486 | 0.6076 |
| 3 | 0.2640 | 0.2807 | 0.7433 | -0.0363 | 0.2019 | 0.1508 | 0.2164 | 0.6180 | 0.6237 |
| 4 | 0.3990 | 0.4104 | 0.7322 | 0.0062 | 0.1893 | 0.3851 | 0.0870 | 0.6549 | 0.6095 |

- mean held-out explained fraction：INIT code `0.3495`；FINAL code `0.3821`；raw WALK `0.6374`；reconstruction gain `0.7367`。
- mean residual label projection：INIT `0.1551`；FINAL `0.1223`；FINAL−INIT `-0.0328`；direction `2/5`。
- mean residual effect cosine：INIT `0.2525`；FINAL `0.2345`。
- mean reconstruction-gain test Cohen d：`0.1324`。
- top-3 frequency bins：patch mass `0.6560`；positive gain contribution `0.6258`；mass/gain correlation `0.8937`。
- readout route condition：`FAIL`。

### raw/exact-isomorphism-grouped

| fold | INIT code R2 | FINAL code R2 | gain R2 | residual FINAL−INIT label projection | INIT residual cosine | FINAL residual cosine | gain Cohen d | top3 patch mass | top3 gain share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4810 | 0.5242 | 0.8185 | -0.0868 | 0.4114 | 0.2286 | -0.3068 | 0.6299 | 0.5158 |
| 1 | 0.2384 | 0.2888 | 0.5076 | -0.2063 | 0.3388 | 0.1202 | 0.0648 | 0.6416 | 0.6008 |
| 2 | 0.2147 | 0.2481 | 0.6457 | 0.6530 | -0.2905 | 0.3192 | 0.1718 | 0.6046 | 0.5990 |
| 3 | 0.3654 | 0.3959 | 0.6763 | -0.0581 | 0.2835 | 0.2182 | 0.8010 | 0.6708 | 0.5527 |
| 4 | 0.4962 | 0.5670 | 0.7136 | 0.0998 | -0.5108 | -0.4199 | 0.2663 | 0.7215 | 0.6431 |

- mean held-out explained fraction：INIT code `0.3591`；FINAL code `0.4048`；raw WALK `0.6065`；reconstruction gain `0.6723`。
- mean residual label projection：INIT `0.0120`；FINAL `0.0924`；FINAL−INIT `0.0803`；direction `2/5`。
- mean residual effect cosine：INIT `0.0465`；FINAL `0.0933`。
- mean reconstruction-gain test Cohen d：`0.1994`。
- top-3 frequency bins：patch mass `0.6537`；positive gain contribution `0.5823`；mass/gain correlation `0.8497`。
- readout route condition：`FAIL`。

## 3. Route decision

> **PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE**

- readout/cross-patch remains plausible：`False`；
- stats redundancy supported：`True`；
- frequency reweighting candidate：`False`；
- next step：Freeze a residual/conditional objective protocol that removes train-fitted graph-stat nuisance before dictionary learning; do not add a richer readout first.

## 4. Boundary

本诊断没有训练新模型，也没有用结果选择超参数。outer-test labels 只用于固定的 effect consistency 描述，因此不能把本报告中的 projection/R2 当作新的分类性能。

## 5. Route interpretation

R0-X 的最重要结果不是某个新分类分数，而是三个机制量：

1. `STATS -> reconstruction_gain` 的 held-out explained fraction 很高：stratified `0.737`，grouped `0.672`；
2. `STATS -> FINAL code` 的 explained fraction 高于 INIT：stratified `0.350 -> 0.382`，grouped `0.359 -> 0.405`；
3. 控制 STATS 后，FINAL residual label projection 没有稳定增强：stratified mean `-0.033`、方向 `2/5`；grouped mean `+0.080`、方向 `2/5`。

这支持的解释是：KSVD update 的 reconstruction gain 很大一部分与 graph statistics / global nuisance 共同变化，而不是稳定增加 label-residual direction。grouped 中的正均值主要由单个 fold 的 `+0.653` 驱动，不能满足预注册的 `3/5` consistency condition。

Patch-frequency 结果也显示明显但未达到 routing threshold 的 frequency coupling：top-3 edge-count bins 只占约 `0.65` patch mass，却贡献 stratified `0.626`、grouped `0.582` 的 positive gain；bin mass 与 positive gain 的 correlation 为 `0.894/0.850`。因此 frequency weighting 是次要候选，但当前证据更优先指向 conditional/statistics-residual objective，而不是继续堆 readout。

最终路由：

> **PRIORITIZE_STATS_CONDITIONAL_OBJECTIVE**

这不是说“减去 graph statistics 后一定能成功”，而是说下一步最有信息量的单轴实验应该改变 dictionary target，使它不再优先重建已经被 STATS 解释的 variation。若 conditional objective 仍不能产生 residual task utility，再转入 task-aware objective；此时必须明确承认研究命题已经从普通无监督 KSVD 改为 label-conditioned dictionary learning。
