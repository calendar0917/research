# LCRC-CPC v1 — generated result tables

All AP values are official `ogb.graphproppred.Evaluator("ogbg-molpcba")` AP. Test is an exposed engineering comparison set.

## Main comparison

| Model | Params | Best Valid AP | Valid@40 | Best-Valid Test AP | Test@40 | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 0.265273316 | 0.261866918 | 0.253616863 | 0.250985846 | 1.3949 |
| RC-REAL | 1,051,022 | 0.267717702 | 0.266950286 | 0.255940917 | 0.256167166 | 1.9186 |
| W2-DYNAMIC | 1,128,468 | 0.275067186 | 0.268791116 | 0.262226049 | 0.260136315 | 1.7340 |
| LCRC-CPC | 1,043,333 | 0.265944328 | 0.263993025 | 0.262356011 | 0.261492308 | 1.7620 |

## Recovery relative to W2-DYNAMIC

| Split / checkpoint | CPC AP | W2 AP | DYNAMIC AP | Recovery fraction |
|---|---:|---:|---:|---:|
| Validation best | 0.265944328 | 0.265273316 | 0.275067186 | 0.0685 |
| Validation epoch 40 | 0.263993025 | 0.261866918 | 0.268791116 | 0.3071 |
| Exposed Test best | 0.262356011 | 0.253616863 | 0.262226049 | 1.0151 |
| Exposed Test epoch 40 | 0.261492308 | 0.250985846 | 0.260136315 | 1.1482 |

Pre-registered 75% endpoints: Validation 0.272620, Test 0.260070.

## Support strata, best-valid checkpoints (exposed Test)

| Stratum | N | CPC mean AP | W2 mean AP | DYNAMIC mean AP | CPC-W2 | improved / degraded vs W2 | CPC-DYNAMIC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common high-support (Val+Test >=20 positives) | 106 | 0.280598722 | 0.275816057 | 0.286440162 | +0.004782665 | 72 / 34 | -0.005841440 |
| Low-support | 22 | 0.170273756 | 0.141563792 | 0.140002432 | +0.028709964 | 15 / 6 | +0.030271324 |
| All scored Test tasks | 127 | 0.262356011 | 0.253616863 | 0.262226049 | +0.008739148 | 87 / 40 | +0.000129962 |

## Support strata, best-valid checkpoints (Validation)

| Stratum | N | CPC mean AP | W2 mean AP | DYNAMIC mean AP | CPC-W2 | improved / degraded vs W2 | CPC-DYNAMIC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common high-support | 106 | 0.291600244 | 0.289268942 | 0.300638511 | +0.002331302 | 67 / 39 | -0.009038267 |
| Low-support | 22 | 0.136443042 | 0.144152539 | 0.145992878 | -0.007709497 | 8 / 13 | -0.009549836 |
| All scored Validation tasks | 127 | 0.265944328 | 0.265273316 | 0.275067186 | +0.000671012 | 75 / 52 | -0.009122857 |

## Uncertainty (descriptive 95% intervals)

Task bootstrap, 20,000 resamples of tasks:

| Comparison | Delta | 2.5% | 97.5% | fraction > 0 |
|---|---:|---:|---:|---:|
| high_support: cpc_test_minus_dynamic_test | -0.005841440 | -0.010380995 | -0.001148137 | 0.0069 |
| high_support: cpc_test_minus_w2_test | +0.004782665 | +0.000340475 | +0.009134339 | 0.9820 |
| high_support: cpc_valid_minus_dynamic_valid | -0.009038267 | -0.015681212 | -0.003028273 | 0.0015 |
| high_support: cpc_valid_minus_w2_valid | +0.002331302 | -0.002628089 | +0.007314375 | 0.8231 |
| all_tasks: cpc_test_minus_dynamic_test | +0.000129962 | -0.009967027 | +0.011742431 | 0.4860 |
| all_tasks: cpc_test_minus_w2_test | +0.008739148 | +0.001121739 | +0.017880573 | 0.9885 |
| all_tasks: cpc_valid_minus_dynamic_valid | -0.009122857 | -0.016287103 | -0.002480425 | 0.0036 |
| all_tasks: cpc_valid_minus_w2_valid | +0.000671012 | -0.005564929 | +0.006957789 | 0.5796 |

Molecule bootstrap, 300 resamples of Test molecules:

| Comparison | Delta | 2.5% | 97.5% | fraction > 0 |
|---|---:|---:|---:|---:|
| LCRC_CPC_best_valid_minus_W2_best_valid | +0.007972434 | -0.000069078 | +0.016094488 | 0.9700 |
| LCRC_CPC_epoch40_minus_W2_best_valid | +0.006982744 | -0.000844517 | +0.014684581 | 0.9433 |
| W2_DYNAMIC_best_valid_minus_W2_best_valid | +0.008656009 | +0.000807044 | +0.016563020 | 0.9900 |
| W2_DYNAMIC_epoch40_minus_W2_best_valid | +0.006457368 | -0.000886309 | +0.014911682 | 0.9667 |
| W2_epoch40_minus_W2_best_valid | -0.002682009 | -0.006223017 | +0.000555823 | 0.0700 |

## Mechanism, best-valid checkpoint, Validation only

| Intervention | Valid AP | Valid BCE |
|---|---:|---:|
| real | 0.265944176 | 0.046584398 |
| delta_zero | 0.195371183 | 0.052053991 |
| pairing_shuffle | 0.255172222 | 0.047169266 |

Full-Validation branch: R=0.490205, RMS(delta)=0.533487, RMS(h0)=1.088295, gate mean=0.7212, gate std=0.2980, gate p1=0.0027, gate p99=0.9979, paths/atom=3.0732, zero-path fraction=5.146e-05.

Logit RMS change vs delta-zero: 0.2022; vs pairing-shuffle: 0.0828.

## Cost

Formal run: 1.7620 GPU-h (internal), 1.8447 GPU-h including preflight/freeze/mechanism allocations. Budget target 2.8 / hard 3.5.

## Decision

**CPC_PARTIAL_GAIN** — mechanism valid: True.

