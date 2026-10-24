### 训练结果（valid only，官方 test 未加载）

| arm | seed | params | epochs | best-valid | ep | W20 | T100 | final | train@best(eval) | gpu-h |
|---|---|---|---|---|---|---|---|---|---|---|
| D | 0 | 470321 | 600 | 0.084493 | 566 | 0.094786 | 0.100461 | 0.095439 | 0.023578 | 1.107 |
| S | 0 | 470321 | 600 | 0.081838 | 557 | 0.103041 | 0.110148 | 0.099211 | 0.053499 | 1.107 |
| B-R0 (历史) | 0 | 494193 | 600 | 0.078537 | 489 | 0.079082 | n/a | 0.079076 | n/a | n/a |

### D − S（同 seed 0，逐指标）

| 指标 | D − S |
|---|---|
| best_valid | +0.002655 |
| w20 | -0.008255 |
| t100 | -0.009686 |
| final_valid | -0.003772 |

### 与 B-R0 的差（负 = 更好）

| arm | best-valid − B | W20 − B | final − B |
|---|---|---|---|
| D | +0.005956 | +0.015703 | +0.016362 |
| S | +0.003301 | +0.023959 | +0.020135 |

### 配对 bootstrap（valid 行重采样，固定 checkpoint，描述性）

| 对比 | ΔMAE | 95% CI | 前者更好的行占比 |
|---|---|---|---|
| B-R0-s0 minus D-s0 | -0.005956 | [-0.014483, +0.004588] | 0.569 |
| B-R0-s0 minus S-s0 | -0.003301 | [-0.014960, +0.012454] | 0.598 |
| D-s0 minus S-s0 | +0.002655 | [-0.003758, +0.009659] | 0.492 |

### 稳健性（valid）

| arm | MAE | 去 row172 | 去自身 top10 |
|---|---|---|---|
| D | 0.084493 | 0.082009 | 0.073438 |
| S | 0.081838 | 0.081548 | 0.072776 |

### 冻结分组（n / MAE）

| arm | all_1000 | seen_749 | unseen_251 | ordinary_935 | tail_65 | unseen_ordinary_225 | seen_ordinary_710 | unseen_tail_26 |
|---|---|---|---|---|---|---|---|---|
| D | 0.0845 | 0.0612 | 0.1540 | 0.0750 | 0.2206 | 0.1449 | 0.0529 | 0.2325 |
| S | 0.0818 | 0.0582 | 0.1523 | 0.0747 | 0.1844 | 0.1407 | 0.0538 | 0.2534 |
| n | 1000 | 749 | 251 | 935 | 65 | 225 | 710 | 26 |

### 固定权重干预（训练后的 D，规则换成 S，完整重前向）

| 量 | 值 |
|---|---|
| reported_best_valid_checkpoint | 0.0844933 |
| reproduction_absdiff_vs_reported | 4.05726e-09 |
| trained_arm | D |
| seed | 0 |
| checkpoint | structural-sufficiency/zinc_relation_composition_probe_v1/runs/RC_D_s0/best.pt |
| checkpoint_epoch | 566 |
| reported_final_valid | 0.0954388 |
| valid_mae_with_D_rule | 0.0844933 |
| valid_mae_with_S_rule_same_weights | 0.122155 |
| mae_difference_S_minus_D | 0.0376613 |
| prediction_change_abs_mean | 0.084941 |
| prediction_change_abs_p50 | 0.0541702 |
| prediction_change_abs_p90 | 0.216573 |
| prediction_change_abs_max | 1.77703 |
| layer0_relation_state_max_abs_diff | 9.17318 |
| layer0_relation_state_mean_abs_diff | 0.399744 |
