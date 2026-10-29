# Phase A — 指定 11 行 + 两模型 AE top-20 并集的结构诊断（test-as-dev，仅描述）

## A. 指定行

| row | true | LCRC | DCC | shared_failure | r2_support | cycle_penalty | chemical_pattern | possible_explanation |
|---|---:|---:|---:|---|---|---|---|---|
| 702 | -2.297 | -4.305 (AE 2.01) | -3.398 (AE 1.10) | yes | df_min=3/med=490 | excess=1 z=-3.28 [ring>=7] | rings=7x2;macro(7);stereo=1;amidex4 | SHARED_FAILURE (artificial_cycle_penalty) |
| 298 | -2.926 | -1.535 (AE 1.39) | -2.234 (AE 0.69) | yes | df_min=1/med=212 | excess=1 z=-3.28 [ring>=7] | rings=5x1/6x2/7x1;macro(7);stereo=1;etherx3 | SHARED_FAILURE (artificial_cycle_penalty) |
| 159 | +0.027 | -1.123 (AE 1.15) | -1.737 (AE 1.76) | yes | unseen(10/29) | none | rings=5x1/6x3;stereo=8;esterx1;etherx1 | SHARED_FAILURE (unseen_r2) |
| 510 | +1.614 | +0.815 (AE 0.80) | +1.056 (AE 0.56) | yes | unseen(3/30) | none | rings=6x4;stereo=1;amidex2;nitrilex1;etherx1;friedel_aryl_halidex1 | SHARED_FAILURE (unseen_r2) |
| 648 | -2.629 | -1.953 (AE 0.68) | -2.119 (AE 0.51) | yes | unseen(3/16) | none | rings=5x1/6x1;stereo=1;esterx1;etherx1;sulfonylx1 | SHARED_FAILURE (unseen_r2) |
| 796 | -0.605 | +0.052 (AE 0.66) | -0.085 (AE 0.52) | yes | unseen(2/21) | none | rings=3x1/6x2;stereo=2;esterx1;etherx3 | SHARED_FAILURE (unseen_r2) |
| 634 | -1.558 | -2.198 (AE 0.64) | -1.912 (AE 0.35) | yes | df_min=1/med=656 | excess=1 z=-3.28 [ring>=7] | rings=6x3/7x1;macro(7);amidex1;etherx1;sulfonylx1;friedel_aryl_halidex1 | SHARED_FAILURE (artificial_cycle_penalty) |
| 822 | -1.090 | -1.726 (AE 0.64) | -0.935 (AE 0.15) | no | unseen(5/15) | none | rings=4x1;esterx1;nitrilex1;etherx3 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=4.1x)+opposite_signs) |
| 527 | +1.232 | +1.854 (AE 0.62) | +1.678 (AE 0.45) | yes | unseen(5/14) | none | rings=6x1;stereo=1;friedel_aryl_halidex1 | SHARED_FAILURE (unseen_r2) |
| 622 | +2.342 | +2.954 (AE 0.61) | +2.684 (AE 0.34) | no | df_min=39/med=875 | none | rings=5x1/6x2;friedel_aryl_halidex1 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=1.8x)) |
| 402 | -7.661 | -7.233 (AE 0.43) | -4.572 (AE 3.09) | yes | unseen(2/27) | excess=2 z=-6.72 [basis-only(no ring>=7)] | rings=6x5;stereo=4;esterx1;amidex1;etherx1;friedel_aryl_halidex2 | SHARED_FAILURE (cycle_penalty_nonminimal_basis) |

## B. 其余 top-20 并集行（非指定）

| row | true | LCRC | DCC | shared_failure | r2_support | cycle_penalty | chemical_pattern | possible_explanation |
|---|---:|---:|---:|---|---|---|---|---|
| 61 | -1.579 | -1.213 (AE 0.37) | -1.192 (AE 0.39) | no | unseen(5/20) | none | rings=6x1;stereo=3;esterx2;etherx2 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=1.1x)) |
| 84 | -2.147 | -1.597 (AE 0.55) | -2.923 (AE 0.78) | yes | unseen(4/27) | none | rings=5x1/6x3;stereo=3 | SHARED_FAILURE (unseen_r2) |
| 189 | -2.537 | -2.026 (AE 0.51) | -2.372 (AE 0.17) | no | unseen(3/16) | none | rings=5x1/6x1 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=3.1x)) |
| 336 | -0.520 | -0.078 (AE 0.44) | -0.025 (AE 0.49) | yes | df_min=3/med=291 | none | rings=6x1;nitrilex1 | SHARED_FAILURE (seen_supported_r2) |
| 342 | +1.327 | +0.726 (AE 0.60) | -0.162 (AE 1.49) | yes | unseen(9/21) | none | rings=6x2 | SHARED_FAILURE (unseen_r2) |
| 365 | -2.374 | -1.769 (AE 0.61) | -1.818 (AE 0.56) | yes | unseen(5/18) | none | rings=6x2 | SHARED_FAILURE (unseen_r2) |
| 414 | -3.655 | -3.886 (AE 0.23) | -4.025 (AE 0.37) | no | unseen(5/22) | excess=1 z=-3.28 [ring>=7] | rings=5x2/7x1;macro(7);stereo=2;amidex1 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=1.6x)) |
| 479 | -3.856 | -4.275 (AE 0.42) | -4.139 (AE 0.28) | no | unseen(1/11) | none | rings=5x1;stereo=2;amidex2 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=1.5x)) |
| 660 | -0.575 | -0.115 (AE 0.46) | -0.224 (AE 0.35) | yes | unseen(8/21) | none | rings=5x1/6x1;friedel_aryl_halidex2 | SHARED_FAILURE (unseen_r2) |
| 669 | -1.723 | -1.436 (AE 0.29) | -1.033 (AE 0.69) | no | df_min=39/med=1058 | excess=1 z=-3.28 [ring>=7] | rings=6x2/7x1;macro(7);amidex1;etherx3 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=2.4x)) |
| 715 | -2.806 | -2.318 (AE 0.49) | -2.459 (AE 0.35) | no | unseen(6/31) | none | rings=5x4/6x2;stereo=5;amidex5;friedel_aryl_halidex2 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=1.4x)) |
| 790 | -2.985 | -2.642 (AE 0.34) | -2.398 (AE 0.59) | no | df_min=1/med=190 | excess=1 z=-3.28 [ring>=7] | rings=6x2/7x1;macro(7);stereo=1;esterx1;etherx1 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=1.7x)) |
| 792 | -3.465 | -3.266 (AE 0.20) | -3.089 (AE 0.38) | no | unseen(1/13) | none | rings=6x1 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=1.9x)) |
| 954 | +0.022 | +0.245 (AE 0.22) | +0.475 (AE 0.45) | no | unseen(3/36) | none | rings=5x2/6x3;stereo=2;amidex1;etherx2 | MODEL_COMPLEMENTARY (dcc_only(ae_ratio=2.0x)) |
| 961 | +1.375 | +0.852 (AE 0.52) | +1.127 (AE 0.25) | no | unseen(2/32) | none | rings=5x2/6x3;amidex4 | MODEL_COMPLEMENTARY (lcrc_only(ae_ratio=2.1x)) |

列说明：

- `shared_failure` = 同时落在 LCRC-A s0 与 DCC-L s0 的 test AE top-20 内。
- `r2_support` = radius-2 typed rooted 环境在 official train 10k 的 document frequency（该分子原子上的 min / median；`unseen` = 至少一个原子的 DF=0）。
- `cycle_penalty` = 生成器的人工环惩罚 `cycle = -max(0, max(nx.cycle_basis)-6)`，从**已验证身份的 RDKit 分子**独立重算；`z` 是其在标签中的标准化分量。
- `chemical_pattern` 仅在 graph→RDKit 身份校验通过时给出，否则 `UNKNOWN`；`macro(k)` = RDKit 环感知到的 k 元环。
- `possible_explanation` 由 PROTOCOL.md §1.3 的固定规则生成，不按误差大小推断；`cycle_penalty_nonminimal_basis` 表示该惩罚并非来自真实 ≥7 元环，而是某一次 `nx.cycle_basis` 抽到非最小基环（见 `phase_a_basis_artifact.json`）。
- 本轮 test 结果一律标记为 **TEST_AS_DEV_DESCRIPTIVE**，不是新的独立 benchmark。
