# luyin14 edge-aware joint dictionary：3 split-seed 汇总

> 日期：2026-08-12  
> 数据：MUTAG / PTC_MR；真实 TUD 数据，不构造数据集。  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_EDGE_AWARE_JOINT_PROTOCOL_20260812.md`  
> 判定：`EDGE_AWARE_JOINT_NOT_STABLE_ACROSS_SPLITS`

## 1. 为什么做这条路线

此前结构 KSVD 使用无类型 adjacency。节点级诊断显示 rooted `v∪N(v)` patch 在 MUTAG/PTC_MR
只有 4/6 种，K-SVD 训练重构误差约 `1e-14`，INIT 已经接近精确字典，FINAL 没有学习余量。
同时两个分子数据都实际提供 4 类 edge attributes，但旧 patch 和 GIN baseline 都没有使用。

本轮做了两步改变：

1. 把中心节点属性、一跳邻居属性均值和局部结构放在一个 coupled dictionary 中，以同一稀疏
   系数表示 structure–attribute 共同出现；
2. 把 4 类 edge type 编入 canonical patch，并把基线换成读取 edge attributes 的 GINE。

这不是在分类器末端继续 concat，而是把多模态交互前移到 token 学习。

## 2. Stage A 与扩展结果

split seed 0 通过冻结 Stage-A gate：MUTAG 上 FINAL−GINE `+0.134`（3/3），
TRUE−SHUFFLED `+0.079`（2/3），FINAL−INIT `+0.074`（2/3）；PTC_MR 相对 GINE
`+0.010`。因此按协议原样扩展 split seeds 1/2。

但强信号没有跨 split 稳定复现：

| dataset | split seed | FINAL−GINE | TRUE−SHUFFLED | FINAL−INIT |
|---|---:|---:|---:|---:|
| MUTAG | 0 | +0.134 | +0.079 | +0.074 |
| MUTAG | 1 | -0.036 | -0.063 | -0.004 |
| MUTAG | 2 | +0.032 | +0.028 | +0.024 |
| PTC_MR | 0 | +0.010 | -0.003 | +0.003 |
| PTC_MR | 1 | -0.074 | -0.049 | -0.034 |
| PTC_MR | 2 | +0.001 | -0.012 | -0.049 |

## 3. 九个 outer folds 汇总

| dataset | GINE | FINAL TRUE | SHUFFLED | INIT | FINAL−GINE | W/T/L | TRUE−SHUFFLED | W/T/L | FINAL−INIT | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MUTAG | 0.744 | 0.787 | 0.773 | 0.756 | +0.043 | 5/0/4 | +0.014 | 3/2/4 | +0.031 | 5/0/4 |
| PTC_MR | 0.531 | 0.510 | 0.531 | 0.536 | -0.021 | 4/0/5 | -0.021 | 2/0/7 | -0.027 | 2/0/7 |

字典学习不再是数值空操作：

- MUTAG INIT/FINAL reconstruction：`0.2106 → 0.0551`（seed-0 fold mean）；
- PTC_MR：`0.4232 → 0.1775`。

因此这次失败不是“字典根本没有更新”。它是 task transfer 稳定性失败。

## 4. 结论边界

可以保留的认识：

- edge type 是此前遗漏的重要结构语义；
- coupled structure–attribute code 比事后 concat/FiLM 更接近导师所说的融合思路；
- MUTAG 上存在非零机制信号，且 FINAL 的平均方差小于 GINE，但正确绑定只在 3/9 folds
  严格胜 shuffled，不能作为稳定证据。

不能成立的结论：

- 不能声称 edge-aware joint KSVD 稳定提高分子图分类；
- 不能把 MUTAG 的 seed-0 大增益当最终结果；
- 不应在同一两个小数据集上继续调 block weight、K/T、FiLM strength 或 cross-attention。

## 5. 下一步优先级

1. **先换真实 benchmark，而不是换融合头。** 选择至少一个更大的、带 node/edge attributes
   的真实图分类数据集，使用官方 split 或 scaffold/family split；固定本轮 joint dictionary 与
   GINE 协议，做一次外部迁移验证。
2. **patch 语义升级到 typed radius-2 / path context。** 当前 `v∪N(v)` typed patch 仍只有
   13/17 种，主要是局部星形键型；在结果不可见前冻结一个 deterministic radius-2 capped
   sampler，并先审计 unique rate、coverage 与重复率，再决定是否训练字典。
3. **只有真实大数据上 TRUE>SHUFFLED 且 FINAL>INIT 后，才考虑 cross-attention。** 此时
   cross-attention 的对象应是 GINE node states 与 joint sparse tokens，而不是图级 pooled vectors。

如果新增真实 benchmark 仍失败，应把 KSVD 固定为 typed local compressor/diagnostic，分类主线
转为 GINE/子图 GNN，不再围绕融合复杂度继续搜索。
