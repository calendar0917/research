# zinc_additive_reader_v1 — machine-generated main tables

Frozen checkpoints only. MAE is raw-y L1, lower is better; delta = candidate - reference (negative = candidate better).

## Per-run metrics

| run | params | Monitor | Locked (best) | Locked (final) | train@best | best ep | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_s0 | 453361 | 0.126245 | 0.084394 | 0.084551 | 0.006971 | 560 | 0.3610 |
| A_s1 | 453361 | 0.124696 | 0.084693 | 0.084542 | 0.004564 | 608 | 0.3481 |
| D_s0 | 461521 | 0.125478 | 0.087832 | 0.085457 | 0.019176 | 392 | 0.4008 |
| D_s1 | 461521 | 0.128676 | 0.086537 | 0.086783 | 0.011971 | 495 | 0.4199 |
| C_s0 | 494193 | 0.123151 | 0.083573 | 0.083573 | nan | 437 | 0.3685 |
| C_s1 | 494193 | 0.127772 | 0.083423 | 0.083423 | nan | 552 | 0.3691 |

## Locked groups (best Monitor checkpoint)

| run | all | all-supported | low-support | any-unseen | ordinary | tail y<=-3 |
|---|---:|---:|---:|---:|---:|---:|
| A_s0 | 0.084394 | 0.042606 | 0.061473 | 0.161847 | 0.068213 | 0.306156 |
| A_s1 | 0.084693 | 0.038548 | 0.060158 | 0.169139 | 0.067274 | 0.323445 |
| D_s0 | 0.087832 | 0.042557 | 0.066934 | 0.166248 | 0.070100 | 0.330862 |
| D_s1 | 0.086537 | 0.043275 | 0.063422 | 0.165865 | 0.070055 | 0.312442 |
| C_s0 | 0.083573 | 0.039987 | 0.066116 | 0.155348 | 0.068255 | 0.293533 |
| C_s1 | 0.083423 | 0.040784 | 0.059818 | 0.162758 | 0.067308 | 0.304300 |

## Paired comparisons (best checkpoint)

| comparison | scope | delta MAE | 95% bootstrap CI | frac molecules better | n |
|---|---|---:|---|---:|---:|
| A_s0 vs D_s0 | overall | -0.003438 | [-0.008099, +0.001869] | 0.542 | 1000 |
| A_s0 vs D_s0 | all_supported | +0.000049 | [-0.008075, +0.011702] | 0.571 | 312 |
| A_s0 vs D_s0 | low_support | -0.005462 | [-0.010906, +0.000478] | 0.539 | 401 |
| A_s0 vs D_s0 | any_unseen | -0.004401 | [-0.014913, +0.006109] | 0.516 | 287 |
| A_s0 vs D_s0 | ordinary_y_gt_-3 | -0.001886 | [-0.006438, +0.003325] | 0.534 | 932 |
| A_s0 vs D_s0 | extreme_tail_y_le_-3 | -0.024707 | [-0.052279, +0.005595] | 0.647 | 68 |
| A_s0 vs DCCL_s0 | overall | +0.000820 | [-0.004055, +0.006492] | 0.520 | 1000 |
| A_s0 vs DCCL_s0 | all_supported | +0.002619 | [-0.005215, +0.014188] | 0.513 | 312 |
| A_s0 vs DCCL_s0 | low_support | -0.004644 | [-0.010493, +0.002233] | 0.544 | 401 |
| A_s0 vs DCCL_s0 | any_unseen | +0.006499 | [-0.004965, +0.019183] | 0.495 | 287 |
| A_s0 vs DCCL_s0 | ordinary_y_gt_-3 | -0.000041 | [-0.004671, +0.005185] | 0.515 | 932 |
| A_s0 vs DCCL_s0 | extreme_tail_y_le_-3 | +0.012623 | [-0.023719, +0.056147] | 0.588 | 68 |
| D_s0 vs DCCL_s0 | overall | +0.004258 | [+0.000400, +0.008159] | 0.492 | 1000 |
| D_s0 vs DCCL_s0 | all_supported | +0.002570 | [-0.001424, +0.006620] | 0.506 | 312 |
| D_s0 vs DCCL_s0 | low_support | +0.000818 | [-0.004281, +0.006054] | 0.494 | 401 |
| D_s0 vs DCCL_s0 | any_unseen | +0.010900 | [+0.000925, +0.021205] | 0.474 | 287 |
| D_s0 vs DCCL_s0 | ordinary_y_gt_-3 | +0.001845 | [-0.001791, +0.005385] | 0.496 | 932 |
| D_s0 vs DCCL_s0 | extreme_tail_y_le_-3 | +0.037330 | [+0.011951, +0.065337] | 0.441 | 68 |
| A_s1 vs D_s1 | overall | -0.001844 | [-0.005960, +0.002570] | 0.525 | 1000 |
| A_s1 vs D_s1 | all_supported | -0.004726 | [-0.008630, -0.000742] | 0.564 | 312 |
| A_s1 vs D_s1 | low_support | -0.003264 | [-0.007954, +0.001551] | 0.516 | 401 |
| A_s1 vs D_s1 | any_unseen | +0.003274 | [-0.008526, +0.016564] | 0.495 | 287 |
| A_s1 vs D_s1 | ordinary_y_gt_-3 | -0.002781 | [-0.006783, +0.001563] | 0.532 | 932 |
| A_s1 vs D_s1 | extreme_tail_y_le_-3 | +0.011003 | [-0.013297, +0.034930] | 0.426 | 68 |
| A_s1 vs DCCL_s1 | overall | +0.001270 | [-0.002664, +0.005470] | 0.486 | 1000 |
| A_s1 vs DCCL_s1 | all_supported | -0.002235 | [-0.006210, +0.001720] | 0.490 | 312 |
| A_s1 vs DCCL_s1 | low_support | +0.000340 | [-0.004801, +0.005559] | 0.481 | 401 |
| A_s1 vs DCCL_s1 | any_unseen | +0.006381 | [-0.004160, +0.018114] | 0.488 | 287 |
| A_s1 vs DCCL_s1 | ordinary_y_gt_-3 | -0.000034 | [-0.003894, +0.004088] | 0.495 | 932 |
| A_s1 vs DCCL_s1 | extreme_tail_y_le_-3 | +0.019145 | [-0.002575, +0.042169] | 0.368 | 68 |
| D_s1 vs DCCL_s1 | overall | +0.003114 | [-0.000267, +0.006521] | 0.473 | 1000 |
| D_s1 vs DCCL_s1 | all_supported | +0.002491 | [-0.001266, +0.006203] | 0.462 | 312 |
| D_s1 vs DCCL_s1 | low_support | +0.003604 | [-0.000931, +0.008076] | 0.469 | 401 |
| D_s1 vs DCCL_s1 | any_unseen | +0.003106 | [-0.005815, +0.012241] | 0.491 | 287 |
| D_s1 vs DCCL_s1 | ordinary_y_gt_-3 | +0.002747 | [-0.000529, +0.006122] | 0.472 | 932 |
| D_s1 vs DCCL_s1 | extreme_tail_y_le_-3 | +0.008142 | [-0.009522, +0.027212] | 0.485 | 68 |
| seed-mean A vs D | overall | -0.002641 | [-0.005912, +0.000747] | 0.560 | 1000 |
| seed-mean A vs D | all_supported | -0.002339 | [-0.007114, +0.004060] | 0.603 | 312 |
| seed-mean A vs D | low_support | -0.004363 | [-0.008307, -0.000294] | 0.559 | 401 |
| seed-mean A vs D | any_unseen | -0.000563 | [-0.008653, +0.007850] | 0.516 | 287 |
| seed-mean A vs D | ordinary_y_gt_-3 | -0.002334 | [-0.005471, +0.000945] | 0.564 | 932 |
| seed-mean A vs D | extreme_tail_y_le_-3 | -0.006852 | [-0.026658, +0.013991] | 0.500 | 68 |
| seed-mean A vs DCCL | overall | +0.001045 | [-0.002503, +0.004804] | 0.524 | 1000 |
| seed-mean A vs DCCL | all_supported | +0.000192 | [-0.004622, +0.006521] | 0.535 | 312 |
| seed-mean A vs DCCL | low_support | -0.002152 | [-0.006821, +0.003134] | 0.524 | 401 |
| seed-mean A vs DCCL | any_unseen | +0.006440 | [-0.002304, +0.015674] | 0.512 | 287 |
| seed-mean A vs DCCL | ordinary_y_gt_-3 | -0.000038 | [-0.003351, +0.003438] | 0.528 | 932 |
| seed-mean A vs DCCL | extreme_tail_y_le_-3 | +0.015884 | [-0.007869, +0.044763] | 0.471 | 68 |
| seed-mean D vs DCCL | overall | +0.003686 | [+0.001075, +0.006392] | 0.475 | 1000 |
| seed-mean D vs DCCL | all_supported | +0.002531 | [-0.000234, +0.005247] | 0.458 | 312 |
| seed-mean D vs DCCL | low_support | +0.002211 | [-0.001415, +0.005852] | 0.486 | 401 |
| seed-mean D vs DCCL | any_unseen | +0.007003 | [-0.000057, +0.014020] | 0.477 | 287 |
| seed-mean D vs DCCL | ordinary_y_gt_-3 | +0.002296 | [-0.000113, +0.004753] | 0.477 | 932 |
| seed-mean D vs DCCL | extreme_tail_y_le_-3 | +0.022736 | [+0.006000, +0.041164] | 0.441 | 68 |

## A pathway diagnostics (Locked, best checkpoint)

| run | std(wS) | std(psi) | corr(wS,y) | corr(psi,y) | local share | MAE full | MAE zero-local | delta | collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_s0 | 0.8512 | 1.1307 | +0.9165 | +0.9463 | 0.4334 | 0.084394 | 0.690613 | +0.606220 | False |
| A_s1 | 0.7791 | 1.2247 | +0.8898 | +0.9488 | 0.4489 | 0.084693 | 0.782199 | +0.697506 | False |

## Decision

**INCONCLUSIVE**

- A_mean=0.084543 D_mean=0.087184 C_mean=0.083498
- gain_A_vs_D=0.002641 gain_A_vs_C=-0.001045 gain_D_vs_C=-0.003686
- per-seed A<D=True A<C=False; ordinary gains vs C/D=0.000038/0.002334; top10 gains vs C/D=0.001880/0.004582
- group stability={'low_support': False, 'any_unseen': False}; any_collapse=False
