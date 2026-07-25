# molhiv design ablation

Protocol: `molhiv-design-ablation-v0` · n=6000 · seed=0

## Baselines

| name | valid AUC | test AUC |
|------|-----------|----------|
| degree | 0.6051 | 0.5876 |
| size | 0.7654 | 0.7727 |
| process_cov | 0.7644 | 0.7706 |

Degree valid = **0.6051** (designs should beat this to claim structure signal)

## All designs (by valid AUC)

| rank | name | group | valid | test | Δval−deg |
|------|------|-------|-------|------|----------|
| 1 | dict_A8T2 | dict | 0.7018 | 0.6731 | +0.097 |
| 2 | rf_L4m4 | rf | 0.6972 | 0.5833 | +0.092 |
| 3 | rf_L12m12 | rf | 0.6786 | 0.7576 | +0.074 |
| 4 | pool_max | pool | 0.6365 | 0.6366 | +0.031 |
| 5 | samp_cov | sampler | 0.6365 | 0.6366 | +0.031 |
| 6 | rf_L8m8 | rf | 0.6365 | 0.6366 | +0.031 |
| 7 | bias_BFS | bias | 0.6365 | 0.6366 | +0.031 |
| 8 | dict_A16T3 | dict | 0.6365 | 0.6366 | +0.031 |
| 9 | dict_A24T4 | dict | 0.6191 | 0.6988 | +0.014 |
| 10 | bias_neutral | bias | 0.6168 | 0.7420 | +0.012 |
| 11 | pool_attn | pool | 0.6140 | 0.6817 | +0.009 |
| 12 | pool_rich | pool | 0.6104 | 0.6005 | +0.005 |
| 13 | bias_DFS | bias | 0.5515 | 0.6575 | -0.054 |
| 14 | samp_cov_uncovered | sampler | 0.5084 | 0.7000 | -0.097 |
| 15 | pool_mean | pool | 0.4909 | 0.5060 | -0.114 |
| 16 | samp_B0 | sampler | 0.4476 | 0.6255 | -0.157 |
| 17 | samp_cov_no_earlystop | sampler | 0.3543 | 0.5098 | -0.251 |

## Best per group

- **bias**: `bias_BFS` val=0.6365 test=0.6366
- **dict**: `dict_A8T2` val=0.7018 test=0.6731
- **pool**: `pool_max` val=0.6365 test=0.6366
- **rf**: `rf_L4m4` val=0.6972 test=0.5833
- **sampler**: `samp_cov` val=0.6365 test=0.6366

## How to read

- Primary: **valid AUC** ranking (scaffold; avoid test shopping).
- Viable direction: consistently **> degree** on valid, and not only high test.
- If process_cov ≈ best design → sampling stats leak, not KSVD structure.

Elapsed: 182.32s
