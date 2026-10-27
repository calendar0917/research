# ZINC error-mechanism audit — machine digest

Official test loaded: **false**.  CPU only; no neural model trained.

Follow-up decomposition: `code/zema_decomposition.py` -> `analysis/decomposition.json`, `analysis/mece_penalty_unseen_long7.csv`, `analysis/cycle_attribution.csv`, `analysis/oof_conditional.csv`. Narrative report: `REPORT.md`.

## Model MAE

| model | MAE | median AE | P90 AE |
|---|---:|---:|---:|
| DCCL_s0 | 0.078537 | 0.036971 | 0.143344 |
| DCCL_mean | 0.069924 | 0.033738 | 0.137154 |
| BOSI_B_s0 | 0.082405 | 0.037000 | 0.164440 |
| ZBO_R_s0 | 0.076707 | 0.036015 | 0.152221 |
| ZBO_N_s0 | 0.081592 | 0.041610 | 0.161195 |
| ZBO_S_s0 | 0.075762 | 0.035513 | 0.153526 |

`DCCL_per_seed_mean` (historical headline) = **0.075990**; `DCCL_mean` is the MAE of the 4-seed prediction average (diagnostic ensemble), not a single model.

## Oracle ceiling (DCC-L 4-seed mean)

| fixed group | n | AE share | ΔMAE if zero AE | new MAE |
|---|---:|---:|---:|---:|
| label_penalty | 35 | 0.187 | 0.013082 | 0.056842 |
| label_penalty_severe | 5 | 0.119 | 0.008324 | 0.061600 |
| rdkit_large_gt6 | 31 | 0.073 | 0.005107 | 0.064817 |
| mcb_max_gt6 | 31 | 0.073 | 0.005107 | 0.064817 |
| mean_maxbasis_gt6 | 37 | 0.188 | 0.013174 | 0.056750 |
| exact_longest_ge7 | 314 | 0.508 | 0.035490 | 0.034434 |
| exact_longest_ge10 | 144 | 0.279 | 0.019475 | 0.050449 |

## Cross-tab (DCC-L 4-seed mean)

| table | cell | n | MAE | AE share | signed |
|---|---|---:|---:|---:|---:|
| penalty_x_unseen | penalty__unseen | 10 | 0.4923 | 0.070 | +0.4043 |
| penalty_x_unseen | penalty__seen | 25 | 0.3264 | 0.117 | +0.2158 |
| penalty_x_unseen | no_penalty__unseen | 241 | 0.1184 | 0.408 | +0.0062 |
| penalty_x_unseen | no_penalty__seen | 724 | 0.0391 | 0.405 | -0.0057 |
| penalty_x_support | penalty__full | 12 | 0.1015 | 0.017 | +0.0208 |
| penalty_x_support | penalty__low4 | 13 | 0.5340 | 0.099 | +0.3957 |
| penalty_x_support | penalty__unseen | 10 | 0.4923 | 0.070 | +0.4043 |
| penalty_x_support | no_penalty__full | 371 | 0.0296 | 0.157 | -0.0050 |
| penalty_x_support | no_penalty__low4 | 353 | 0.0491 | 0.248 | -0.0065 |
| penalty_x_support | no_penalty__unseen | 241 | 0.1184 | 0.408 | +0.0062 |
| penalty_x_tail | penalty__tail | 18 | 0.6215 | 0.160 | +0.5282 |
| penalty_x_tail | penalty__ordinary | 17 | 0.1115 | 0.027 | -0.0042 |
| penalty_x_tail | no_penalty__tail | 47 | 0.0815 | 0.055 | +0.0223 |
| penalty_x_tail | no_penalty__ordinary | 918 | 0.0577 | 0.758 | -0.0040 |

## global33 ring-scale predictability (train -> valid)

| target | valid R² | valid MAE | valid Spearman |
|---|---:|---:|---:|
| longest_simple_cycle_length_exact_all | 0.943 | 0.165 | 0.859 |
| max_min_basis_cycle_length | 0.892 | 0.010 | 0.484 |
| total_min_basis_cycle_length | 0.999 | 0.026 | 0.993 |
| mean_maxbasis | 0.747 | 0.031 | 0.488 |
| frac_ge7 | 0.913 | 0.011 | 0.314 |

High valid R2 means these scales are largely reconstructable from the 33-D
global descriptor (MCB total is essentially a restatement of cyclomatic rank
and the ring-size histogram).  They are not new per-atom information; see
`REPORT.md` section 4.

## OOF ring-feature diagnostic (compact-v4-hinge, not DCC-L)

| variant | features | R²(log AE) | Spearman(pred, AE) | top-quintile MAE |
|---|---:|---:|---:|---:|
| rarity | 12 | 0.1317 | 0.3499 | 0.2953 |
| structure_global | 31 | 0.1520 | 0.3735 | 0.3253 |
| plus_topo_nonring | 49 | 0.1735 | 0.4019 | 0.3300 |
| plus_new_ring | 53 | 0.1894 | 0.4048 | 0.3679 |
| topo_all | 56 | 0.1931 | 0.4046 | 0.3680 |

**Caveat:** the all-row OOF R2 gain is entirely penalty-row-driven.  On the
9,628 no-penalty rows the same ring scales add `delta R2 = +0.00006` and
`delta Spearman = -0.0008`; see `analysis/oof_conditional.csv` and
`REPORT.md` section 5.  Do not quote the all-row gain as general ring
information.

Full JSON: `analysis/audit.json`; tables: `analysis/*.csv`.
