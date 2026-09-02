# Beam8/NCI1 patch-chain classification Stage A

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_NCI1_S8O2_CLASSIFICATION_ADDENDUM_20260813.md`  
> 判定：`BEAM8_SUBSTRATE_PASS_KSVD_UPDATE_NOT_ESTABLISHED`

## 1. Sampling

- graphs：`4110`；非连通图：`580`；
- mean patches：`4.918`；
- mean edge/node coverage：`0.7898` / `0.8112`；
- partial components：`672`。
- single-patch graphs：`436`；graphs with chain edges：`3544`；mean directed chain edges：`3.731`；
- graphs with non-chain overlap：`3536`。

## 2. Classification

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| FEATURE_ONLY | 0.6568 ± 0.0046 | 0.6567 |
| FEATURE_STATS | 0.7017 ± 0.0043 | 0.7017 |
| RAW_BAG | 0.6696 ± 0.0040 | 0.6696 |
| RAW_CHAIN_TRUE | 0.6577 ± 0.0144 | 0.6577 |
| RAW_CHAIN_SHUFFLED | 0.6421 ± 0.0102 | 0.6421 |
| INIT_BAG | 0.6779 ± 0.0080 | 0.6779 |
| INIT_CHAIN_TRUE | 0.6694 ± 0.0110 | 0.6693 |
| FINAL_BAG | 0.6754 ± 0.0035 | 0.6754 |
| FINAL_CHAIN_TRUE | 0.6706 ± 0.0073 | 0.6706 |
| FINAL_CHAIN_SHUFFLED | 0.6555 ± 0.0150 | 0.6555 |

## 3. Paired attribution

| comparison | mean difference | W/T/L |
|---|---:|---:|
| raw_true_vs_shuffled | +0.0156 | 2/0/1 |
| raw_true_vs_bag | -0.0119 | 1/0/2 |
| final_true_vs_shuffled | +0.0151 | 3/0/0 |
| final_true_vs_bag | -0.0049 | 1/0/2 |
| final_vs_init_true | +0.0012 | 2/0/1 |

## 4. Gates

- RAW substrate gate：`True`；
- Beam8 relation gate：`False`；
- KSVD update gate：`False`；

## 5. 解释边界

- TRUE/SHUFFLED 共用 cover、token multiset、关系图、位置元数据和 residual sidecar，只改变 token 到 patch position 的绑定。
- TRUE 优于 SHUFFLED 但低于 BAG，表示真实绑定可检测，却还没有转化为超过无序聚合的分类价值。
- relation 或 RAW 通过但 KSVD gate 失败，只支持 Beam8 chain substrate，不支持普通无监督 KSVD updates 具有分类增益。
- 本轮为固定无参数 relation message passing；只有 relation gate 通过后才进入可学习 patch GNN。
