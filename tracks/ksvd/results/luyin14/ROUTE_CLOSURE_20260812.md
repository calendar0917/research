# luyin14 路线闭环：补边、分类、关系与节点特征融合

> 日期：2026-08-12  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_ROUTE_PROTOCOL_20260812.md`  
> 判定：`KSVD_REMAINS_COMPRESSOR_DIAGNOSTIC_BASELINE`

## 1. 冻结判定

- EDGE100 默认 gate：`False`；
- true relation binding gate：`False`；
- feature fusion gate：`False`。

## 2. 补边成本

| dataset | FAIR reach | FAIR patches | FAIR edge/pair | residual edges | EDGE100 patches | added p50/p90 | EDGE100 pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 1.000 | 5.99 | 0.993/0.681 | 1.33 | 6.63 | 0.0/2.0 | 0.691 |
| IMDB-MULTI | 1.000 | 3.87 | 0.997/0.879 | 0.96 | 4.24 | 0.0/1.0 | 0.883 |
| MUTAG | 1.000 | 4.62 | 0.998/0.634 | 0.06 | 4.68 | 0.0/0.0 | 0.636 |
| PTC_MR | 1.000 | 4.47 | 0.999/0.755 | 0.04 | 4.51 | 0.0/0.0 | 0.756 |

EDGE100 只保证真实边覆盖为 100%；节点对覆盖仍可能远低于 100%。

## 3. 分类主表（balanced accuracy）

| dataset | STATS | RAW | INIT | FAIR content | relation graph | TRUE | SHUFFLED | EDGE100 | FAIR+residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-BINARY | 0.708 | 0.629 | 0.631 | 0.632 | 0.674 | 0.680 | 0.670 | 0.632 | 0.673 |
| IMDB-MULTI | 0.463 | 0.450 | 0.445 | 0.448 | 0.459 | 0.466 | 0.464 | 0.442 | 0.456 |
| MUTAG | 0.856 | 0.691 | 0.697 | 0.712 | 0.755 | 0.741 | 0.751 | 0.696 | 0.763 |
| PTC_MR | 0.554 | 0.529 | 0.517 | 0.521 | 0.509 | 0.515 | 0.517 | 0.505 | 0.520 |

## 4. 节点特征融合（balanced accuracy）

| dataset | feature | +stats | +content | +true relation | +residual | gate | best Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MUTAG | 0.825 | 0.874 | 0.765 | 0.757 | 0.756 | 0.769 | -0.056 |
| PTC_MR | 0.564 | 0.585 | 0.548 | 0.562 | 0.553 | 0.555 | -0.002 |

## 5. Paired route deltas

| dataset | EDGE100−FAIR | TRUE−SHUFFLED |
|---|---:|---:|
| IMDB-BINARY | -0.001 | +0.011 |
| IMDB-MULTI | -0.007 | +0.002 |
| MUTAG | -0.016 | -0.009 |
| PTC_MR | -0.016 | -0.002 |

逐折稳定性：

| dataset | EDGE100 W/T/L | TRUE relation W/T/L |
|---|---:|---:|
| IMDB-BINARY | 3/0/6 | 8/0/1 |
| IMDB-MULTI | 2/0/7 | 4/2/3 |
| MUTAG | 3/2/4 | 2/4/3 |
| PTC_MR | 1/0/8 | 5/0/4 |

## 6. 路线结论

1. **补到 100% 不应成为默认路线。** FAIR95 后平均只剩很少真实边，但 EDGE100 在四个数据集上均未提高 balanced accuracy；四个平均差值全为负。已有无标签 rate 审计还显示 BASE+residual 比补到 EDGE100 平均节省约 348 proxy bits，因此默认保留 FAIR95，并把残余边作为显式 sidecar。
2. **当前 patch 关系绑定没有跨数据集成立。** IMDB-BINARY 是唯一接近门槛的正例（约 +1.1 points，8/9 folds 为正），IMDB-MULTI 只有 +0.2 points，两个带特征数据集为负。不能据此进入 Transformer 或更深 relation network。
3. **结构与节点特征的融合未成立。** MUTAG 上所有 KSVD 融合均明显低于 feature-only；PTC_MR 的 true-relation concat 接近持平但没有正增益。相反，简单 `feature+stats` 在两个数据集分别约 +4.9/+2.1 points，说明任务可用信号来自低阶统计，而非当前 KSVD code。
4. **普通 KSVD 的合理定位不变：** 它在重构和压缩上有效，但当前 code/readout 没有形成稳定下游增益。后续不扫描 K/T、restart、fusion depth；若继续研究，只能另立问题，转向合法结构对象与 exact occurrence/incidence。

## 7. 解释边界

- sampling 完全不使用 labels；字典、scaler、classifier 与 gate 均为 train-fold only。
- completion patch 被标为新 segment；没有伪造连续 `o=2` 关系。
- residual sidecar 只使用结构输入可计算的不变统计，不使用 graph label。
- 本表是机制闭环，不是与论文 leaderboard 的统一公开 benchmark。
- 若 gate 未通过，不通过扩大 K/T、增加 restart 或扫描网络深度救结果。
