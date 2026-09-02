# luyin14：TU 中等规模真实数据外部验证

> 日期：2026-08-13  
> 数据：Mutagenicity / NCI1；不构造数据集。  
> 协议：`tracks/ksvd/docs/KSVD_LUYIN14_TUD_EXTERNAL_VALIDATION_PROTOCOL_20260813.md`  
> 判定：`ONE_HOP_JOINT_CODE_NOT_STABLE; RADIUS2_SUBSTRATE_FEASIBLE`

## 1. 为什么选择这两个数据集

| dataset | graphs | classes | mean nodes | node types | edge types | 角色 |
|---|---:|---:|---:|---:|---:|---|
| Mutagenicity | 4,337 | 2 | 30.3 | 14 | 3 | 完整 edge-aware 外部验证 |
| NCI1 | 4,110 | 2 | 29.9 | 37 | 0 | 无 edge type 的 shared-code control |

两者都明显大于 MUTAG/PTC_MR，但远小于 MolHIV，单机 CPU 可以运行严格 3-fold screen。
Mutagenicity 此前未参与当前 joint dictionary 的方法选择；NCI1 已在仓库中，但未用于本轮
edge-aware/shared-code 选择。

数据兼容审计发现 Mutagenicity/NCI1 分别有 297/580 张非连通图。旧 GLOBAL-WL tie-breaker
要求连通图，但两数据集最大度数都只有 4，当前 8-node `v∪N(v)` patch 从不截断邻居。因此
修复为仅在确实需要截断时计算 GLOBAL-WL；该修改不改变任何本轮 patch 内容。

## 2. Stage A

### 2.1 Mutagenicity

| GINE | FINAL TRUE | SHUFFLED | INIT | FINAL−GINE | TRUE−SHUFFLED | FINAL−INIT |
|---:|---:|---:|---:|---:|---:|---:|
| 0.681 | 0.752 | 0.719 | 0.713 | +0.070，3/3 | +0.033，3/3 | +0.039，2/3 |

INIT/FINAL reconstruction：`0.3602 → 0.1399`。四项冻结 gate 全通过，因此按协议扩展
split seeds 1/2。

### 2.2 NCI1

| GIN | FINAL TRUE | SHUFFLED | INIT | FINAL−GIN | TRUE−SHUFFLED | FINAL−INIT |
|---:|---:|---:|---:|---:|---:|---:|
| 0.692 | 0.736 | 0.721 | 0.749 | +0.044，3/3 | +0.015，2/3 | -0.013，1/3 |

INIT/FINAL reconstruction：`0.2147 → 0.0998`。联合局部 token 相对 GIN 有用，但 FINAL
低于 INIT，收益不能归因于 K-SVD updates。按协议停止，不扩展 seeds、不调参。

## 3. Mutagenicity 三个 split seeds

### 3.1 每个 split 的 paired delta

| split seed | FINAL−GINE | W/L | TRUE−SHUFFLED | W/L | FINAL−INIT | W/L |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.070 | 3/0 | +0.033 | 3/0 | +0.039 | 2/1 |
| 1 | -0.030 | 0/3 | +0.002 | 1/2 | +0.012 | 2/1 |
| 2 | -0.053 | 0/3 | -0.012 | 1/2 | -0.034 | 1/2 |

seed 0 的强正结果没有迁移到另外两个数据划分。

### 3.2 九个 outer folds 汇总

| GINE | FINAL TRUE | SHUFFLED | INIT |
|---:|---:|---:|---:|
| 0.728 | 0.724 | 0.716 | 0.719 |

| comparison | mean delta | W/T/L |
|---|---:|---:|
| FINAL−GINE | -0.004 | 3/0/6 |
| TRUE−SHUFFLED | +0.008 | 5/0/4 |
| FINAL−INIT | +0.006 | 5/0/4 |

正确绑定和 K-SVD update 有很弱的平均正值，但都没有形成稳定 fold majority；相对公平 GINE
基线反而略负。因此不能声称一跳 edge-aware joint K-SVD 稳定提高 Mutagenicity 分类。

## 4. 为什么一跳 patch 仍可能过粗

当前 rooted `v∪N(v)` 在 Mutagenicity 只有 19 种 typed structure patterns；它主要描述中心
原子的直接键型星形，不能区分相同一跳键型在第二层连接到什么环境。NCI1 的 binary structure
patterns 也只有 8 种。较高的 joint-pattern 数量主要来自 node attributes，并不等于结构 view
本身丰富。

这解释了两个现象：

1. INIT joint prototypes 经常已经优于纯 GNN；
2. K-SVD reconstruction 明显改善，但分类增量取决于 split，说明它优化的局部重构方向未必是
   稳定任务方向。

## 5. radius-2 substrate 审计（不使用标签、不训练分类器）

| dataset | mean r2 size | median | p90 | max | 完整落入 8 nodes | cap8 平均保留率 | r2=整图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mutagenicity | 6.45 | 5 | 10 | 17 | 79.7% | 95.7% | 0.168% |
| NCI1 | 6.27 | 6 | 9 | 14 | 87.4% | 98.1% | 0.0065% |

因此 radius-2 在两个数据集上仍是局部对象，并且固定 8-node 容量只产生有限截断。它通过了
“值得设计新协议”的 substrate gate，但尚未证明分类有效。

## 6. 结论与下一步

当前可以成立：

- TU 中等规模数据可替代 OGB 完成当前机制 screen，计算规模可控；
- structure–attribute joint token 确实比纯 GNN 含有额外信息；
- edge-aware/正确绑定在部分划分上有效，但一跳 patch 的效果不稳定；
- ordinary K-SVD 的重构改善仍不能自动转化为稳定分类改善。

当前不能成立：

- 不能用 Mutagenicity seed 0 的 `+7pt` 声称方法成功；
- 不能把 NCI1 的 `+4.4pt` 归因于 K-SVD，因为 INIT 更高；
- 不应继续扫描 K/T、抽样量、FiLM 或 cross-attention 来修一跳结果。

下一项若继续，应单独冻结为 **typed radius-2 shared-code protocol**：

1. 先定义与 node relabel 无关的 deterministic cap8 选择/编码；
2. 先要求 relabel stability、截断率和 held-out reconstruction 通过；
3. 分类仍保留 BASE/TRUE/SHUFFLED/INIT 四项；
4. 首轮只用 split seed 0，过 gate 后再扩展；
5. 若 radius-2 仍不能稳定胜 GINE/GIN 与 INIT，停止 TU 分类融合路线。

