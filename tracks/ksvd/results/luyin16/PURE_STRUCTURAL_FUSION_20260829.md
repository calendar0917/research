# 纯结构 69D + K24/T3/624D 融合代理

日期：2026-08-29  
范围：OGB MolHIV official train/validation；official test 未评估。

## 设计

```text
S_struct = 69D graph-only statistics
R_struct = 624D = 26 coefficient summaries × K24 atoms
patch input = topology-only 28D induced adjacency
K-SVD = K24 / T3 / 5 updates
```

69D 结构统计由以下固定、无化学属性的块组成：

```text
18D topology summary
21D degree histogram
15D degree-pair histogram
15D shortest-path/disconnected histogram
```

624D readout 是一个维度对齐代理，不声称等同导师的 `recon_typed` schema。

## 结果

使用 8-trial Optuna TPE、3-fold train-inner CV、5 个最终模型种子：

| view | dim | valid ROC-AUC |
|---|---:|---:|
| S_struct | 69 | **0.7723 ± 0.0071** |
| R_init_struct | 624 | 0.6460 ± 0.0035 |
| R_final_struct | 624 | 0.6385 ± 0.0053 |
| S_struct + R_init | 693 | 0.7333 ± 0.0089 |
| S_struct + R_final | 693 | 0.7367 ± 0.0045 |

K-SVD 重建误差从 `0.2029` 降到 `0.1179`，但下游分类没有同步改善。

## 判定

该实验没有支持“纯结构 K-SVD 在纯结构统计之上提供增量”。但它只能否定当前代理，不能否定导师路线，原因包括：

1. 当前 69D 是自定义结构统计，并非导师真实 69D；
2. 当前 624D 是 `26×K24` 的人为维度对齐，而非导师真实 `recon_typed`；
3. topology-only patch 输入为 28D adjacency，可能比导师 patch 缺少 typed/canonical/context 信息；
4. 仅使用 8 个 Optuna trials，适合方向判断，不适合声称达到最优。

当前更可靠的结论是：

> 单纯减少统计维度、去掉原子/键特征，并不能自动让 K-SVD 产生额外增益；如果导师 valid 也接近 0.80，关键仍可能在真实 69/624 schema、patch typed context 和 graph-level readout，而不是维度数量本身。

结果入口：

- [`pure_structural_fusion_proxy.py`](../../experiments/luyin16/pure_structural_fusion_proxy.py)
- [`manifest.json`](./pure_structural_fusion_k24_t3/manifest.json)
- [`xgb_optuna_search.json`](./pure_structural_fusion_k24_t3/xgb_optuna_search.json)
