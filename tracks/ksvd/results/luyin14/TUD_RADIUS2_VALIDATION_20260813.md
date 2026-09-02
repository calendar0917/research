# luyin14：TU radius-2 外部验证汇总

> 日期：2026-08-13  
> 协议：`KSVD_LUYIN14_TUD_RADIUS2_PROTOCOL_20260813.md`  
> 判定：`RADIUS2_REPRESENTATION_SIGNAL_WITHOUT_KSVD_UPDATE_ATTRIBUTION`

## 1. Mutagenicity：radius-2 no-go

| GINE | FINAL TRUE | SHUFFLED | INIT | FINAL−GINE | TRUE−SHUFFLED | FINAL−INIT |
|---:|---:|---:|---:|---:|---:|---:|
| 0.681 | 0.751 | 0.751 | 0.751 | +0.070，3/3 | +0.000，1/3 | +0.000，2/3 |

typed structure patch types 从一跳的 19 增至 424，INIT/FINAL reconstruction 从 `0.6026`
降至 `0.2828`，但正确节点绑定和 K-SVD update 没有形成分类增量。因此不扩展 split seeds。

## 2. NCI1：radius-2 三 split 汇总

| model | mean balanced accuracy | std |
|---|---:|---:|
| GIN_ONLY | 0.707 | 0.027 |
| JOINT_FINAL_TRUE | 0.757 | 0.016 |
| JOINT_FINAL_SHUFFLED | 0.739 | 0.026 |
| JOINT_INIT_TRUE | 0.757 | 0.009 |

| comparison | mean delta | W/T/L |
|---|---:|---:|
| FINAL−GIN | +0.050 | 8/0/1 |
| TRUE−SHUFFLED | +0.018 | 7/0/2 |
| FINAL−INIT | -0.001 | 6/0/3 |

每个 split 的 FINAL−INIT：

| split seed | FINAL−GIN | TRUE−SHUFFLED | FINAL−INIT |
|---:|---:|---:|---:|
| 0 | +0.076 | +0.032 | +0.017 |
| 1 | +0.058 | +0.031 | +0.002 |
| 2 | +0.016 | -0.009 | -0.021 |

NCI1 radius-2 的核心结果是：正确 joint token 比 GIN 稳定强约 5pt，且多数时候优于打乱；
然而 FINAL 与 INIT 几乎相同。因此稳定增益来自 richer joint prototype/representation，而
不是 K-SVD updates。

## 3. 与一跳结果的比较

| route | Mutagenicity FINAL−BASE | NCI1 FINAL−BASE | NCI1 FINAL−INIT |
|---|---:|---:|---:|
| one-hop | -0.4pt（9 folds） | +4.4pt（3 folds） | -1.3pt |
| radius-2 | no 3-seed expansion | +5.0pt（9 folds） | -0.06pt |

radius-2 确实比一跳提供了更丰富的局部语义，尤其在 NCI1 上提高了 TRUE/SHUFFLED 的稳定
差异；但它没有修复普通无监督 K-SVD 的 objective mismatch。

## 4. 最终判断

现在可以支持的命题：

1. 在真实 NCI1 上，节点局部结构与属性的正确绑定有稳定分类信息；
2. radius-2 比一跳更适合承载这种局部联合信息；
3. GIN/GINE 末端的简单拼接不是唯一瓶颈。

不能支持的命题：

1. 不能说 K-SVD updates 本身带来稳定分类提升；
2. 不能说 edge-aware radius-2 在 Mutagenicity 上已成功；
3. 不应继续扫描 K/T、增加 K-SVD iterations、加 cross-attention 或扩大融合器。

## 5. 下一步建议

如果继续这条研究线，问题应从“无监督 K-SVD 能否自动学出分类字典”改成：

> **固定 radius-2 joint patch token，研究 task-aware / supervised residual dictionary 是否能
> 在 NCI1 上超过同一 INIT，并保留 TRUE/SHUFFLED 对照。**

这将明确承认：当前稳定的是局部联合表示，未被证明的是无监督 K-SVD 更新的任务价值。若不想
引入标签进字典学习，则应把 radius-2 INIT joint token 作为一个结构—属性原型基线，KSVD
只作为可选压缩器，不再把它包装成 task-optimal dictionary learner。
