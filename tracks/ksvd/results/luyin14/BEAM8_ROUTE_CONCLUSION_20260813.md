# Beam8 路线终结审计

> 日期：2026-08-13  
> 数据：真实 TU NCI1 / Mutagenicity  
> 结论：`KEEP_AS_SAMPLING_COMPRESSION_DIAGNOSTIC_STOP_CLASSIFICATION_ESCALATION`

## 1. Beam8 已经证明了什么

Beam8 不是“没有用”。从 `s8/o2` 连续 patch chain 出发，它稳定完成了三件事：

1. 相比旧 `s10/o3`，产生更多真正可连接的多-patch 图和更长 chain；
2. TRUE token-to-relation binding 在 NCI1 上可被检测：正式 `s8/o2` 结果中，
   FINAL TRUE−SHUFFLED 为 `+1.51pt`，3/3 folds 为正；
3. 在 Mutagenicity 上，合法 atom-colored/bond-typed canonical 实现通过了 node relabel
   invariance；BASE token rows、multiset、compact readout 均为 `1.000000`。

因此，Beam8 可以保留为连续覆盖、patch 压缩和 relation-binding 诊断 substrate。

## 2. 它为什么没有转化成分类收益

### NCI1

- `s8/o2` FINAL TRUE−SHUFFLED：`+1.51pt`，说明真实绑定可检测；
- FINAL TRUE−BAG：`-0.49pt`，compact relation 后仅约 `+0.05pt`；
- strict OOF late fusion：FEATURE_STATS `70.17%`，fusion `70.27%`，仅 `+0.10pt`；
- KSVD FINAL 对 INIT 没有稳定改善。

这里的含义是：relation signal 存在，但基本被 bag/属性统计覆盖，没有形成稳定的增量预测信息。

### Mutagenicity

合法 attributed BASE 的 `s8/o2` chain geometry 更好，但 binary Beam8 marginal objective
仍忽略稀有 bond semantics：稀有键 aggregate recall 只有 `45.45%`，含稀有键图的覆盖率
只有 `47.87%`。

canonical typed anchor 以平均每图 `0.0583` 个 token 的代价，把：

- 稀有键 aggregate recall 提到 `90.00%`；
- 各 bond type graph coverage 提到 `100%`；
- BASE 表示保持完全不变；
- BASE/ANCHOR token 和 compact readout permutation invariance 全部做到 `1.000000`。

但合法分类审计中：

| comparison | mean | W/T/L |
|---|---:|---:|
| BASE RAW TRUE−SHUFFLED | -0.56pt | 0/0/3 |
| ANCHOR INIT TRUE−BASE INIT TRUE | -0.06pt | 2/0/1 |
| ANCHOR FINAL TRUE−SHUFFLED | -0.98pt | 0/0/3 |
| ANCHOR FINAL TRUE−BAG | -0.59pt | 0/0/3 |
| ANCHOR FINAL−INIT TRUE | -0.09pt | 1/0/2 |
| ANCHOR INIT TRUE−FEATURE_STATS | -2.64pt | 0/0/3 |

六个预注册 classification gates 全部失败。最好基线是 `FEATURE_STATS 70.52%`；结构分支
约为 `68%–69%`。

## 3. 三种贡献必须分开

- **Beam8 sampling/relation**：NCI1 上证明 binding 可检测，但未证明超过 BAG/属性基线；
- **edge semantics**：typed anchor 确实低成本修复 semantic completeness，但没有带来分类增量；
- **KSVD updates**：合法最终实验中 FINAL 不胜 INIT，不能声称普通 KSVD updates 有稳定价值。

这也解释了为什么不能用更复杂的 Transformer、cross-attention 或图像式 fusion 继续扫：
当前缺少的是独立增量信号，不是一个已经通过低容量 gate、只差表达能力的 signal。扩大模型
会把 sampling、relation、attributes 和容量混在一起，无法回答 Beam8/KSVD 到底贡献了什么。

## 4. 作废结果

早期 binary-order typed classification 不满足 node relabel invariance，只能作为实现诊断，
不得作为方法证据。第一次 attributed anchor classification 也因 anchor tie-break 不 invariant
而作废。最终表格只来自通过 BASE/ANCHOR invariance gate 的 canonical two-endpoint anchor。

## 5. 下一步建议

停止在 NCI1/Mutagenicity 上扩大 Beam8 下游分类模型。后续若仍研究 Beam8，问题应改为：

1. 把它作为 sampling/compression 方法，报告 coverage、chain connectivity、semantic recall、
   compression ratio 与运行成本；
2. 若要重新主张分类价值，必须换一个事先说明为何需要 ordered patch relations 的真实任务，
   并先用同样的 BAG/TRUE/SHUFFLED 低容量 gate；
3. 若继续研究 dictionary learning，应重新设计与 relation/edge semantics 对齐的学习目标，
   而不是继续增加普通 reconstruction KSVD 的更新轮数或分类头容量。

当前证据支持的最强表述是：

> Beam8 是有效的连续覆盖与压缩 substrate，真实 relation binding 有时可检测；但在 NCI1
> 与 Mutagenicity 上，edge-semantic completeness、compact relation 和普通 KSVD updates
> 均未建立超过 BAG/属性基线的稳定分类价值。
