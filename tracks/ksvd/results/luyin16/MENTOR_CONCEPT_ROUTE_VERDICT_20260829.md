# luyin16 导师固定特征路线阶段判定

日期：2026-08-29  
状态：**ordinary unsupervised K-SVD downstream branch closed；mentor exact replication blocked by missing upstream schema**  
评估边界：OGB MolHIV official train/validation；**test 从未预测或参与选择**。

## 1. 回答的问题

在拿不到导师 69/624 维上游特征代码时，本阶段独立复现如下研究结构：

```text
train-only coverage patches
→ matched INIT / FINAL K-SVD
→ fixed graph features
→ XGBoost(binary:logistic)
→ official validation attribution
```

目标不是追一个最高 AUC，而是区分：显式统计、raw patch 内容、字典压缩、K-SVD 更新和融合分别贡献什么。

## 2. 完整 official-validation 结果

### 2.1 chem patch 的 code readout

| view | valid ROC-AUC | 相对基线 |
|---|---:|---:|
| `S`：显式拓扑 + 原子/键组成 | **0.7817 ± 0.0066** | — |
| `T_init` | 0.6718 ± 0.0094 | — |
| `T_final` | 0.6521 ± 0.0146 | FINAL−INIT = **−0.0197** |
| `S+T_init` | 0.7717 ± 0.0103 | −0.0100 vs S |
| `S+T_final` | 0.7624 ± 0.0090 | **−0.0193 vs S** |

K-SVD 把 mean graph reconstruction error 从 `0.1732` 降到 `0.1204`，但分类下降。重建优化成功不等于任务方向成功。

### 2.2 `S` 的特征块归因

| block | valid ROC-AUC |
|---|---:|
| explicit topology only（18D） | 0.6831 ± 0.0110 |
| atom composition（174D） | 0.7066 ± 0.0066 |
| bond composition（13D） | 0.5993 ± 0.0067 |
| atom + bond composition（187D） | 0.7196 ± 0.0078 |
| topology + chemistry composition（205D） | **0.7817 ± 0.0066** |

强基线来自拓扑与化学组成在树模型中的互补交互，不能表述为“纯结构单独达到 0.78”。

### 2.3 pure-topology patch 对照

| view | valid ROC-AUC |
|---|---:|
| topology `T_init` | 0.6157 ± 0.0052 |
| topology `T_final` | 0.6201 ± 0.0058 |
| `S + topology T_final` | 0.7539 ± 0.0056 |

重建误差从 `0.1684` 降到 `0.0714`，但 FINAL−INIT 只有 `+0.0043`，且融合相对 S 为 `−0.0278`。topology patch 输入为 28D，旧 K-SVD 将请求的 K32 自动截为实际 K28；因此该实验是 matched requested budget，不冒充严格相同实际 K。

### 2.4 624D typed-reconstruction proxy

`R_v1` 对 52D typed patch 或其重建逐维计算 12 组固定统计，得到 `12×52=624D`。维度与导师脚本相同只是机制线索，**不声称实现相同**。

| view | valid ROC-AUC | 解释 |
|---|---:|---|
| `R_raw` | 0.6649 ± 0.0057 | 未经字典压缩的 typed patch 分布 |
| `R_init` | 0.6521 ± 0.0088 | INIT 稀疏重建 |
| `R_final` | 0.6542 ± 0.0067 | FINAL 稀疏重建 |
| `S+R_raw` | 0.7510 ± 0.0127 | 低于 S |
| `S+R_init` | 0.7628 ± 0.0057 | 低于 S |
| `S+R_final` | 0.7477 ± 0.0110 | 低于 S |

关键差值：

- `R_final−R_init = +0.0021`：更新的任务增量近零；
- `R_final−R_raw = −0.0107`：压缩损失任务信息；
- `S+R_final−S = −0.0340`：没有统计基线外增量；
- `R_final−T_final = +0.0021`：624D reconstruction readout 没有实质修复 code readout。

### 2.5 624D typed-slot proxy v2：统计对象假设

仓库已有的 typed canonical slot 表示给出一个更自然的 624 维分解：

```text
8 个 atom slots × 64 bins + 28 个 slot pairs × 4 bond bins = 624
```

在相同 official validation、相同 K32/3-iteration budget 下，按 patch 对象逐坐标做 mean 聚合：

| view | valid ROC-AUC |
|---|---:|
| `r_raw`：slot-aligned typed object | **0.7014 ± 0.0062** |
| `r_init` | 0.6535 ± 0.0075 |
| `r_final` | 0.6752 ± 0.0153 |
| `S+r_final` | 0.6801 ± 0.0067 |

相对前一个 624D 全局分布 proxy（`r_raw=0.6649`），slot alignment 带来约 `+0.0365`；`r_final−r_init=+0.0216` 也说明统计对象确实会改变 K-SVD update 的任务方向。但它仍低于 `S=0.7817`，融合没有统计基线外增量。

按 occurrence 做 sum 的 development 对照更差（`r_raw=0.5821`、`r_init=0.5305`、`r_final=0.5732`），因此不能把 624D 简化为 patch 数量质量量。该结果支持“对象/对齐重要”，不支持“已恢复导师特征”。

## 3. 阶段判定

| 命题 | 判定 | 证据 |
|---|---|---|
| 固定特征 + XGBoost 是可行的非 GNN 下游范式 | **PASS** | S official-valid 0.7817 |
| 显式拓扑与化学组成存在互补 | **PASS** | 0.6831 / 0.7196 → 0.7817 |
| K-SVD 是健康的 reconstruction learner | **PASS** | chem、topology 重建误差均明显下降 |
| 普通无监督 K-SVD update 有稳定 MolHIV 标签增量 | **FAIL** | chem 为负；topology/R proxy 近零 |
| K-SVD 特征在 S 之外有稳定增量 | **FAIL** | 所有 `S+FINAL` 均低于 S |
| 失败只因 sparse-code readout 太粗 | **FAIL** | 624D reconstruction proxy 仍无增量 |
| 624D 主要差异可能来自统计对象/slot 对齐 | **PARTIAL SUPPORT** | slot proxy raw 0.7014 > global-stat proxy 0.6649；sum 反例 |
| 已精确复现导师 69/624D 特征与约 0.80 结果 | **NOT CLAIMED / BLOCKED** | 上游 feature builders、payload 和 best params 缺失 |

## 4. 为什么现在停止扫 K/T、Beam 和分类器

当前瓶颈不是字典没有收敛，也不是 XGBoost 完全读不出固定特征，而是 reconstruction objective 与 scaffold-held-out HIV 标签方向错位。继续扫描 K、稀疏度、迭代数、Beam 或融合深度会在同一 validation 上制造选择偏差，不能回答新机制问题。

development 子样本也证明了小验证集的不稳定：16 个 validation 正样本时 `T_final−T_init` 为正；完整 validation 81 个正样本时方向反转。因此后续不根据 n8k 数字选路线。

## 5. 若要继续追导师路线，最低限度需要的上游信息

1. 69D `composition` 的逐列 schema 与构造代码；
2. 624D `recon_typed` 的 patch 输入、重建对象、12/其它聚合块定义；
3. ring context mass 的六类环定义、归属规则和 coverage 列；
4. feature payload 的 dataset indices、split 与 checksum；
5. `best_params`、`common_xgb_params` 和 reference summary；
6. 字典是否严格 train-only，以及约 0.8059 是否使用了 official test 做选择。

在这些信息缺失时，任何同维度实现只能称 proxy。

## 6. 下一步路线

1. **优先获取/核对导师上游 schema**，只做 exact feature audit，不先改分类器目标。
2. 如果上游长期不可得，当前 ordinary unsupervised K-SVD MolHIV 性能分支关闭；保留其 compressor / reconstruction diagnostic 身份。
3. 若必须继续方法研究，另立协议进入 task-aware / conditional dictionary objective，并明确标签进入训练目标；这将不再是“普通无监督 K-SVD 自动发现 task-optimal atoms”的命题。
4. luyin16 的长距离/全局结构问题另设 benchmark 和停止条件，不用 MolHIV validation 继续反复试模型。

## 7. 结果入口

- `mentor_concept_v1_k32_official_valid/`：chem code readout；
- `mentor_concept_v1_feature_blocks/`：S 特征块归因；
- `mentor_concept_v1_k32_topology_official_valid/`：pure-topology patch；
- `mentor_typed_reconstruction_proxy_v1/`：624D reconstruction proxy。
