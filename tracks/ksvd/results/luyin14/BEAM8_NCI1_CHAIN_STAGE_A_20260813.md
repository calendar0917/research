# Beam8/NCI1 patch-chain classification Stage A

> 协议：`tracks/ksvd/docs/KSVD_BEAM8_NCI1_CHAIN_CLASSIFICATION_PROTOCOL_20260813.md`  
> 判定：`BEAM8_SUBSTRATE_PASS_KSVD_UPDATE_NOT_ESTABLISHED`

## 1. Sampling

- graphs：`4110`；非连通图：`580`；
- mean patches：`3.923`；
- mean edge/node coverage：`0.7832` / `0.8007`；
- partial components：`1165`。
- single-patch graphs：`911`；graphs with chain edges：`3018`；mean directed chain edges：`2.736`；
- graphs with non-chain overlap：`2957`。

## 2. Classification

| variant | balanced accuracy | accuracy |
|---|---:|---:|
| FEATURE_ONLY | 0.6568 ± 0.0046 | 0.6567 |
| FEATURE_STATS | 0.7017 ± 0.0043 | 0.7017 |
| RAW_BAG | 0.6793 ± 0.0045 | 0.6793 |
| RAW_CHAIN_TRUE | 0.6611 ± 0.0086 | 0.6611 |
| RAW_CHAIN_SHUFFLED | 0.6404 ± 0.0025 | 0.6404 |
| INIT_BAG | 0.6833 ± 0.0051 | 0.6832 |
| INIT_CHAIN_TRUE | 0.6674 ± 0.0068 | 0.6674 |
| FINAL_BAG | 0.6818 ± 0.0078 | 0.6818 |
| FINAL_CHAIN_TRUE | 0.6706 ± 0.0069 | 0.6706 |
| FINAL_CHAIN_SHUFFLED | 0.6638 ± 0.0163 | 0.6637 |

## 3. Paired attribution

| comparison | mean difference | W/T/L |
|---|---:|---:|
| raw_true_vs_shuffled | +0.0207 | 3/0/0 |
| raw_true_vs_bag | -0.0183 | 0/0/3 |
| final_true_vs_shuffled | +0.0068 | 1/0/2 |
| final_true_vs_bag | -0.0112 | 0/0/3 |
| final_vs_init_true | +0.0032 | 2/0/1 |

## 4. Gates

- RAW substrate gate：`True`；
- Beam8 relation gate：`False`；
- KSVD update gate：`False`；

## 5. 解释边界

- TRUE/SHUFFLED 共用 cover、token multiset、关系图、位置元数据和 residual sidecar，只改变 token 到 patch position 的绑定。
- TRUE 优于 SHUFFLED 但低于 BAG，表示真实绑定可检测，却还没有转化为超过无序聚合的分类价值。
- relation 或 RAW 通过但 KSVD gate 失败，只支持 Beam8 chain substrate，不支持普通无监督 KSVD updates 具有分类增益。
- 本轮为固定无参数 relation message passing；只有 relation gate 通过后才进入可学习 patch GNN。
