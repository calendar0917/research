# Target-edge bridge / multi-chain patch cover 审计

> 日期：2026-08-01  
> 本轮不训练 KSVD、不使用 labels。

## 1. 判定

**PASS_SINGLE_CHAIN_TARGET_BRIDGE**

## 2. 全局结果

| method | edge cover | block edge cover | pair cover | segments | continuous fraction | overlap | gap | connected | target hit | bridge mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent_walk | 0.5875 | 0.6386 | 0.4673 | 1.0000 | 1.0000 | 1.9943 | 0.0000 | 1.0000 | 1.0000 | 0.0000/0.0000 |
| frontier_cover | 0.5979 | 0.5838 | 0.4273 | 1.0000 | 1.0000 | 5.0000 | 0.1348 | 0.8624 | 1.0000 | 0.0000/0.0000 |
| edge_target_bridge | 0.6889 | 0.6846 | 0.4928 | 1.0000 | 1.0000 | 5.0000 | 0.2609 | 1.0000 | 1.0000 | 1.0371/1.7824 |
| multi_chain_target | 0.7152 | 0.7073 | 0.5142 | 4.9861 | 0.7620 | 5.0000 | 0.2603 | 1.0000 | 1.0000 | 0.7980/1.7546 |

## 3. Registered seed gates

| seed | single edge/block/pair gain | continuity/overlap/gap | pass | multi edge/block/pair gain | continuity/overlap/gap | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 820101 | 0.0906/0.1054/0.0672 | 1.0000/5.0000/0.2606 | True | 0.1165/0.1313/0.0883 | 0.7620/5.0000/0.2606 | True |
| 820102 | 0.0876/0.0887/0.0623 | 1.0000/5.0000/0.2612 | True | 0.1137/0.1120/0.0834 | 0.7620/5.0000/0.2600 | True |
| 820103 | 0.0949/0.1082/0.0671 | 1.0000/5.0000/0.2610 | True | 0.1217/0.1272/0.0888 | 0.7620/5.0000/0.2603 | True |

## 4. Family / degree breakdown

| family-degree | method | edge cover | pair cover | continuity |
|---|---|---:|---:|---:|
| regular_d15 | independent_walk | 0.5152 | 0.4041 | 1.0000 |
| regular_d15 | frontier_cover | 0.5451 | 0.3643 | 1.0000 |
| regular_d15 | edge_target_bridge | 0.6353 | 0.4036 | 1.0000 |
| regular_d15 | multi_chain_target | 0.6616 | 0.4220 | 0.7692 |
| regular_d20 | independent_walk | 0.5543 | 0.4827 | 1.0000 |
| regular_d20 | frontier_cover | 0.5978 | 0.4496 | 1.0000 |
| regular_d20 | edge_target_bridge | 0.6877 | 0.5032 | 1.0000 |
| regular_d20 | multi_chain_target | 0.7138 | 0.5255 | 0.7647 |
| regular_d25 | independent_walk | 0.5929 | 0.5383 | 1.0000 |
| regular_d25 | frontier_cover | 0.6307 | 0.5091 | 1.0000 |
| regular_d25 | edge_target_bridge | 0.7174 | 0.5727 | 1.0000 |
| regular_d25 | multi_chain_target | 0.7475 | 0.5993 | 0.7500 |
| small_world_d15 | independent_walk | 0.5289 | 0.4001 | 1.0000 |
| small_world_d15 | frontier_cover | 0.5830 | 0.3738 | 1.0000 |
| small_world_d15 | edge_target_bridge | 0.6657 | 0.4023 | 1.0000 |
| small_world_d15 | multi_chain_target | 0.7068 | 0.4206 | 0.7692 |
| small_world_d20 | independent_walk | 0.5747 | 0.4877 | 1.0000 |
| small_world_d20 | frontier_cover | 0.6326 | 0.4614 | 1.0000 |
| small_world_d20 | edge_target_bridge | 0.7125 | 0.5040 | 1.0000 |
| small_world_d20 | multi_chain_target | 0.7313 | 0.5247 | 0.7647 |
| small_world_d25 | independent_walk | 0.6060 | 0.5442 | 1.0000 |
| small_world_d25 | frontier_cover | 0.6403 | 0.5125 | 1.0000 |
| small_world_d25 | edge_target_bridge | 0.7278 | 0.5721 | 1.0000 |
| small_world_d25 | multi_chain_target | 0.7540 | 0.5975 | 0.7500 |
| block_d15 | independent_walk | 0.5600 | 0.3896 | 1.0000 |
| block_d15 | frontier_cover | 0.5599 | 0.3362 | 1.0000 |
| block_d15 | edge_target_bridge | 0.6423 | 0.3945 | 1.0000 |
| block_d15 | multi_chain_target | 0.6693 | 0.4128 | 0.7753 |
| block_d20 | independent_walk | 0.6473 | 0.4546 | 1.0000 |
| block_d20 | frontier_cover | 0.5885 | 0.3883 | 1.0000 |
| block_d20 | edge_target_bridge | 0.6951 | 0.4942 | 1.0000 |
| block_d20 | multi_chain_target | 0.7116 | 0.5179 | 0.7592 |
| block_d25 | independent_walk | 0.7084 | 0.5040 | 1.0000 |
| block_d25 | frontier_cover | 0.6031 | 0.4504 | 1.0000 |
| block_d25 | edge_target_bridge | 0.7165 | 0.5886 | 1.0000 |
| block_d25 | multi_chain_target | 0.7411 | 0.6072 | 0.7557 |

## 5. 解释边界

- target-edge scheduling 读取输入 adjacency，但不读取 graph labels；它优化的是结构覆盖，不是下游任务。
- multi-chain 若胜出，只能说明一般图需要多个局部连续坐标系，不能把 segment 顺序称为跨图共享位置。
- fresh relabel similarity 仍是随机稳健性指标，不是严格 invariance；严格声明只来自 mapped replay。
- 只有本轮 cover gate 通过后，才允许进入 raw patch 与 KSVD patch 的 stitched reconstruction 对照。
