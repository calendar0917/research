# KSVD E1-T2 跨数据种子五启动确认结果

> 日期：2026-07-31  
> 协议：`KSVD_E1_T2_MULTIDATA_CONFIRMATION_PROTOCOL_20260731.md`  
> 数字源：`e1_t2_multidata_confirmation_20260731.json`

## 1. 冻结配置

- data seeds：`20260731..20260740`（10 个）
- 每个 data seed 的 learner seeds：`[0, 1, 2, 3, 4]`
- train/test patches：1000/300
- K/T/iterations：4/2/25
- selector：最低 train reconstruction，相同则选择较小 learner seed。
- strict success：mean atom cosine >= 0.99 且 test reconstruction <= 0.01。

## 2. 总结果

- oracle controls：**10/10**
- 五候选中包含正确解：**10/10**
- selector 选中正确解：**10/10**
- selection misses：**0**
- 全部 single-start 候选成功率：23/50 = 46.0%

### 被选模型指标

| test reconstruction | atom cosine | minimum atom cosine | support F1 | atom edge-support F1 | edge F1 | exact patch |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0002 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## 3. 每个 data seed

| data seed | oracle | strict candidates / 5 | selected learner seed | train recon | test recon | atom cosine | support F1 | selected strict | selection miss |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 20260731 | PASS | 3/5 | 2 | 0.000078 | 0.000066 | 1.0000 | 1.0000 | Y | N |
| 20260732 | PASS | 2/5 | 0 | 0.000258 | 0.000263 | 1.0000 | 1.0000 | Y | N |
| 20260733 | PASS | 4/5 | 1 | 0.000200 | 0.000180 | 1.0000 | 1.0000 | Y | N |
| 20260734 | PASS | 1/5 | 2 | 0.000227 | 0.000234 | 1.0000 | 1.0000 | Y | N |
| 20260735 | PASS | 2/5 | 0 | 0.000141 | 0.000146 | 1.0000 | 1.0000 | Y | N |
| 20260736 | PASS | 1/5 | 3 | 0.000285 | 0.000242 | 1.0000 | 1.0000 | Y | N |
| 20260737 | PASS | 3/5 | 2 | 0.000199 | 0.000202 | 1.0000 | 1.0000 | Y | N |
| 20260738 | PASS | 2/5 | 0 | 0.000138 | 0.000124 | 1.0000 | 1.0000 | Y | N |
| 20260739 | PASS | 2/5 | 4 | 0.000204 | 0.000182 | 1.0000 | 1.0000 | Y | N |
| 20260740 | PASS | 3/5 | 3 | 0.000000 | 0.000000 | 1.0000 | 1.0000 | Y | N |

## 4. 预注册判断

- 分类：**PASS_FREEZE_FIVE_RESTART_RULE**。
- 结论：E1-T2 后续实验冻结使用 `5 restarts + minimum train reconstruction selector`。
- 下一步只降低 singleton probability，不同时引入边重叠、噪声、置换或采样。

## 5. Gate 明细

- [x] `oracle_controls_10_of_10`
- [x] `groups_containing_success_ge_9_of_10`
- [x] `selectors_successful_ge_9_of_10`
- [x] `selection_miss_count_eq_0`
- [x] `selected_mean_atom_cosine_ge_0.99`
- [x] `selected_mean_test_reconstruction_le_0.01`

## 6. 判定边界

- 本结果只冻结 E1-T2 synthetic setting 的优化规则。
- 它不证明 singleton 稀缺、边支持重叠、噪声、节点置换、随机游走或真实数据条件下仍可恢复。
