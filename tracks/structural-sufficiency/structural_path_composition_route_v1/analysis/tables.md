# PSC result tables (auto-generated)

## D0 held-out whole-molecule accuracy

| arm | input | readout | target | seeds | RMSE mean ± sd | MAE | RMSE ACRA_REF − arm [95% CI] |
|---|---|---|---|---|---|---|---|
| PSC_SUM | ORDER | SUM | MOLECULE | 3 | 0.7843 ± 0.0237 | 0.6074 |  |
| PSC_SUM_PAIR | ORDER | SUM | PAIR | 3 | 0.7810 ± 0.0175 | 0.6164 |  |
| PSC_SUM_BAG | BAG | SUM | MOLECULE | 3 | 0.7437 ± 0.0170 | 0.5722 |  |
| PSC_SUM_BAG_PAIR | BAG | SUM | PAIR | 3 | 0.7229 ± 0.0266 | 0.5561 |  |
| PSC_POOL | ORDER | POOL | MOLECULE | 3 | 0.7264 ± 0.0364 | 0.5684 |  |
| PSC_POOL_PAIR | ORDER | POOL | PAIR | 3 | 0.7174 ± 0.0573 | 0.5640 |  |
| PSC_POOL_BAG | BAG | POOL | MOLECULE | 3 | 0.7071 ± 0.0269 | 0.5445 |  |
| PSC_POOL_BAG_PAIR | BAG | POOL | PAIR | 3 | 0.6871 ± 0.0136 | 0.5303 |  |
| ACRA_REF | site/context events | ACRA | parent | 3 | 0.5802 ± 0.0021 | 0.4445 |  |
| ACRA_PAIR | site/context events | ACRA | parent+PAIR | 3 | 0.5847 ± 0.0220 | 0.4502 | -0.0044 [-0.0255, 0.0156] |
| GINE_REF | raw graph | GINE | MOLECULE | 3 | 0.5266 ± 0.0065 | 0.3978 |  |
| PATH_COUNT_RIDGE | – | – | – | 1 | 0.6167 ± 0.0000 | 0.4647 |  |
| PATH_COUNT_KRR | – | – | – | 1 | 0.6550 ± 0.0000 | 0.4952 |  |
| MORGAN_COUNT_KRR | – | – | – | 1 | 0.6593 ± 0.0000 | 0.5204 |  |
| COUNT_ONLY_RIDGE | – | – | – | 1 | 1.1614 ± 0.0000 | 0.9016 |  |
| PSC_POOL_BAG_DIAG | BAG | POOL | DIAG | 3 | 0.7219 ± 0.0440 | 0.5611 |  |

Reference for the last column: ACRA_REF. Positive Δ means the arm is better than the reference.

## D1 held-out whole-molecule accuracy

| arm | input | readout | target | seeds | RMSE mean ± sd | MAE | RMSE ACRA_REF − arm [95% CI] |
|---|---|---|---|---|---|---|---|
| PSC_SUM_BAG_PAIR | BAG | SUM | PAIR | 2 | 0.8081 ± 0.0524 | 0.5808 |  |
| PSC_POOL_PAIR | ORDER | POOL | PAIR | 2 | 0.7547 ± 0.0097 | 0.5761 |  |
| PSC_POOL_BAG_PAIR | BAG | POOL | PAIR | 2 | 0.7055 ± 0.0042 | 0.5333 |  |
| ACRA_REF | site/context events | ACRA | parent | 2 | 0.6376 ± 0.0079 | 0.4766 |  |
| GINE_REF | raw graph | GINE | MOLECULE | 2 | 0.5534 ± 0.0030 | 0.4090 |  |
| PATH_COUNT_RIDGE | – | – | – | 1 | 0.6542 ± 0.0000 | 0.4722 |  |
| PATH_COUNT_KRR | – | – | – | 1 | 0.6802 ± 0.0000 | 0.4882 |  |
| MORGAN_COUNT_KRR | – | – | – | 1 | 0.6921 ± 0.0000 | 0.5008 |  |
| COUNT_ONLY_RIDGE | – | – | – | 1 | 1.1521 ± 0.0000 | 0.9404 |  |

Reference for the last column: ACRA_REF. Positive Δ means the arm is better than the reference.

## Official Valid (scored once after the freeze)

| model | seeds | RMSE mean ± sd | MAE | ACRA_B* − model [95% CI] | per-seed RMSE |
|---|---|---|---|---|---|
| ACRA_REF | 3 | 0.7860 ± 0.0183 | 0.5554 | – | 0.8032, 0.7668, 0.7879 |
| COUNT_ONLY_RIDGE | 1 | 1.1693 ± 0.0000 | 0.9438 | -0.3833 [-0.5048, -0.2438] | 1.1693 |
| GINE_REF | 3 | 0.6220 ± 0.0119 | 0.4542 | 0.1640 [0.0944, 0.2461] | 0.6287, 0.6082, 0.6290 |
| MORGAN_COUNT_KRR | 1 | 0.7448 ± 0.0000 | 0.5632 | 0.0412 [-0.0555, 0.1533] | 0.7448 |
| PATH_COUNT_KRR | 1 | 0.7240 ± 0.0000 | 0.5413 | 0.0620 [-0.0282, 0.1694] | 0.7240 |
| PATH_COUNT_RIDGE | 1 | 0.7839 ± 0.0000 | 0.5596 | 0.0021 [-0.0527, 0.0506] | 0.7839 |
| PSC_POOL_BAG_PAIR | 3 | 0.7795 ± 0.0126 | 0.6036 | 0.0065 [-0.0909, 0.1287] | 0.7662, 0.7812, 0.7911 |
| PSC_SUM_BAG_PAIR | 3 | 0.8447 ± 0.0136 | 0.6389 | -0.0587 [-0.1233, 0.0109] | 0.8354, 0.8604, 0.8384 |


## Pre-specified contrasts, D0 (paired molecule bootstrap)

| question | candidate vs reference | Δ RMSE [95% CI] | p |
|---|---|---|---|
| Q4 ACRA pair regulariser vs parent | ACRA_PAIR vs ACRA_REF | -0.0044 [-0.0255, 0.0156] | 0.636 |
| Q4 PAIR coupling vs MOLECULE (POOL, BAG) | PSC_POOL_BAG_PAIR vs PSC_POOL_BAG | 0.0200 [-0.0056, 0.0439] | 0.113 |
| Q2 ORDER vs BAG (POOL, PAIR) | PSC_POOL_BAG_PAIR vs PSC_POOL_PAIR | 0.0302 [-0.0105, 0.0684] | 0.147 |
| Q3 SUM vs POOL (BAG, PAIR) | PSC_POOL_BAG_PAIR vs PSC_SUM_BAG_PAIR | 0.0358 [0.0080, 0.0648] | 0.012 |
| Q2 ORDER vs BAG (POOL, MOLECULE) | PSC_POOL_BAG vs PSC_POOL | 0.0193 [-0.0265, 0.0617] | 0.379 |
| Q3 SUM vs POOL (BAG, MOLECULE) | PSC_POOL_BAG vs PSC_SUM_BAG | 0.0366 [-0.0048, 0.0766] | 0.080 |
| Q4 PAIR coupling vs MOLECULE (POOL, ORDER) | PSC_POOL_PAIR vs PSC_POOL | 0.0090 [-0.0144, 0.0328] | 0.445 |
| Q3 SUM vs POOL (ORDER, PAIR) | PSC_POOL_PAIR vs PSC_SUM_PAIR | 0.0637 [0.0300, 0.0974] | 0.000 |
| Q3 SUM vs POOL (ORDER, MOLECULE) | PSC_POOL vs PSC_SUM | 0.0579 [0.0238, 0.0929] | 0.001 |
| Q4 PAIR coupling vs MOLECULE (SUM, BAG) | PSC_SUM_BAG_PAIR vs PSC_SUM_BAG | 0.0208 [-0.0016, 0.0423] | 0.070 |
| Q2 ORDER vs BAG (SUM, PAIR) | PSC_SUM_BAG_PAIR vs PSC_SUM_PAIR | 0.0581 [0.0290, 0.0863] | 0.000 |
| Q2 ORDER vs BAG (SUM, MOLECULE) | PSC_SUM_BAG vs PSC_SUM | 0.0406 [0.0124, 0.0716] | 0.006 |
| Q4 PAIR coupling vs MOLECULE (SUM, ORDER) | PSC_SUM_PAIR vs PSC_SUM | 0.0033 [-0.0152, 0.0213] | 0.717 |
| Q1 champion vs ACRA_PAIR | PSC_POOL_BAG_PAIR vs ACRA_PAIR | -0.1025 [-0.1413, -0.0647] | 0.000 |
| Q1 champion vs ACRA_REF | PSC_POOL_BAG_PAIR vs ACRA_REF | -0.1069 [-0.1509, -0.0665] | 0.000 |
| Q1 champion vs COUNT_ONLY_RIDGE | PSC_POOL_BAG_PAIR vs COUNT_ONLY_RIDGE | 0.4743 [0.4050, 0.5439] | 0.000 |
| Q4 PAIR vs DIAG twin (same input/readout, zero pair target) | PSC_POOL_BAG_PAIR vs PSC_POOL_BAG_DIAG | 0.0348 [0.0085, 0.0607] | 0.009 |
| Q1 champion vs GINE_REF | PSC_POOL_BAG_PAIR vs GINE_REF | -0.1605 [-0.2033, -0.1178] | 0.000 |
| Q1 champion vs MORGAN_COUNT_KRR | PSC_POOL_BAG_PAIR vs MORGAN_COUNT_KRR | -0.0278 [-0.0716, 0.0152] | 0.201 |
| Q1 champion vs PATH_COUNT_KRR | PSC_POOL_BAG_PAIR vs PATH_COUNT_KRR | -0.0322 [-0.0778, 0.0104] | 0.155 |
| Q1 champion vs PATH_COUNT_RIDGE | PSC_POOL_BAG_PAIR vs PATH_COUNT_RIDGE | -0.0705 [-0.1145, -0.0279] | 0.001 |

Scaffold-cluster bootstrap (D0, one resample per Bemis–Murcko scaffold cluster, same statistic):

| contrast | Δ RMSE [95% CI] | clusters |
|---|---|---|
| ACRA_PAIR_vs_ACRA_REF | -0.0044 [-0.0292, 0.0198] | 231 |
| PSC_POOL_BAG_PAIR_vs_ACRA_REF | -0.1069 [-0.1527, -0.0599] | 231 |
| PSC_POOL_BAG_PAIR_vs_PSC_POOL_BAG | 0.0200 [-0.0206, 0.0602] | 231 |
| PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR | 0.0302 [-0.0190, 0.0782] | 231 |
| PSC_POOL_BAG_PAIR_vs_PSC_SUM_BAG_PAIR | 0.0358 [-0.0013, 0.0726] | 231 |
| PSC_POOL_BAG_vs_PSC_POOL | 0.0193 [-0.0352, 0.0718] | 231 |
| PSC_POOL_BAG_vs_PSC_SUM_BAG | 0.0366 [-0.0112, 0.0881] | 231 |
| PSC_POOL_PAIR_vs_PSC_POOL | 0.0090 [-0.0234, 0.0402] | 231 |
| PSC_POOL_PAIR_vs_PSC_SUM_PAIR | 0.0637 [0.0133, 0.1127] | 231 |
| PSC_POOL_vs_PSC_SUM | 0.0579 [0.0133, 0.0985] | 231 |
| PSC_SUM_BAG_PAIR_vs_PSC_SUM_BAG | 0.0208 [-0.0058, 0.0475] | 231 |
| PSC_SUM_BAG_PAIR_vs_PSC_SUM_PAIR | 0.0581 [0.0189, 0.0952] | 231 |
| PSC_SUM_BAG_vs_PSC_SUM | 0.0406 [0.0024, 0.0760] | 231 |
| PSC_SUM_PAIR_vs_PSC_SUM | 0.0033 [-0.0253, 0.0306] | 231 |


## Pre-specified contrasts, D1 (paired molecule bootstrap)

| question | candidate vs reference | Δ RMSE [95% CI] | p |
|---|---|---|---|
| Q2 ORDER vs BAG (POOL, PAIR) | PSC_POOL_BAG_PAIR vs PSC_POOL_PAIR | 0.0492 [0.0117, 0.0859] | 0.007 |
| Q3 SUM vs POOL (BAG, PAIR) | PSC_POOL_BAG_PAIR vs PSC_SUM_BAG_PAIR | 0.1026 [0.0548, 0.1529] | 0.000 |
| Q1 champion vs ACRA_REF | PSC_POOL_BAG_PAIR vs ACRA_REF | -0.0679 [-0.1179, -0.0186] | 0.008 |
| Q1 champion vs COUNT_ONLY_RIDGE | PSC_POOL_BAG_PAIR vs COUNT_ONLY_RIDGE | 0.4466 [0.3837, 0.5120] | 0.000 |
| Q1 champion vs GINE_REF | PSC_POOL_BAG_PAIR vs GINE_REF | -0.1521 [-0.1992, -0.1040] | 0.000 |
| Q1 champion vs MORGAN_COUNT_KRR | PSC_POOL_BAG_PAIR vs MORGAN_COUNT_KRR | -0.0134 [-0.0768, 0.0552] | 0.700 |
| Q1 champion vs PATH_COUNT_KRR | PSC_POOL_BAG_PAIR vs PATH_COUNT_KRR | -0.0253 [-0.0803, 0.0305] | 0.363 |
| Q1 champion vs PATH_COUNT_RIDGE | PSC_POOL_BAG_PAIR vs PATH_COUNT_RIDGE | -0.0513 [-0.1031, 0.0002] | 0.051 |

Scaffold-cluster bootstrap (D1, one resample per Bemis–Murcko scaffold cluster, same statistic):

| contrast | Δ RMSE [95% CI] | clusters |
|---|---|---|
| PSC_POOL_BAG_PAIR_vs_ACRA_REF | -0.0679 [-0.1261, -0.0170] | 267 |
| PSC_POOL_BAG_PAIR_vs_PSC_POOL_PAIR | 0.0492 [0.0100, 0.0882] | 267 |
| PSC_POOL_BAG_PAIR_vs_PSC_SUM_BAG_PAIR | 0.1026 [0.0466, 0.1635] | 267 |


## Pair-level metrics, D0

### By matched-pair scope (unique unordered pairs, unweighted)

| scope | model | pairs scored | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|
| fit_fit | ACRA_REF | 2456 | 0.3150 | 0.4304 | 0.851 |
| fit_fit | GINE_REF | 2456 | 0.2741 | 0.3723 | 0.856 |
| fit_fit | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_fit | PATH_COUNT_KRR | 0 | – | – | – |
| fit_fit | PATH_COUNT_RIDGE | 0 | – | – | – |
| fit_fit | PSC_POOL_BAG_DIAG | 2456 | 0.4330 | 0.5835 | 0.779 |
| fit_fit | PSC_POOL_BAG_PAIR | 2456 | 0.4043 | 0.5460 | 0.799 |
| fit_fit | PSC_SUM_BAG_PAIR | 2456 | 0.4194 | 0.5631 | 0.794 |
| fit_heldout | ACRA_REF | 350 | 0.4257 | 0.5547 | 0.836 |
| fit_heldout | GINE_REF | 350 | 0.3416 | 0.4405 | 0.867 |
| fit_heldout | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PATH_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PATH_COUNT_RIDGE | 0 | – | – | – |
| fit_heldout | PSC_POOL_BAG_DIAG | 350 | 0.5429 | 0.7164 | 0.761 |
| fit_heldout | PSC_POOL_BAG_PAIR | 350 | 0.4971 | 0.6603 | 0.790 |
| fit_heldout | PSC_SUM_BAG_PAIR | 350 | 0.4973 | 0.6522 | 0.798 |
| heldout_heldout | ACRA_REF | 413 | 0.4224 | 0.5806 | 0.783 |
| heldout_heldout | GINE_REF | 413 | 0.3886 | 0.5238 | 0.793 |
| heldout_heldout | MORGAN_COUNT_KRR | 413 | 0.4762 | 0.6329 | 0.751 |
| heldout_heldout | PATH_COUNT_KRR | 413 | 0.4620 | 0.6179 | 0.756 |
| heldout_heldout | PATH_COUNT_RIDGE | 413 | 0.4539 | 0.6175 | 0.783 |
| heldout_heldout | PSC_POOL_BAG_DIAG | 413 | 0.4747 | 0.6359 | 0.746 |
| heldout_heldout | PSC_POOL_BAG_PAIR | 413 | 0.4530 | 0.6166 | 0.791 |
| heldout_heldout | PSC_SUM_BAG_PAIR | 413 | 0.4624 | 0.6349 | 0.799 |

### By support class (heldout × heldout, cut-row weighted)

| support class | pairs | weight | model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|---|
| T_seen | 195 | 173.0 | ACRA_REF | 0.3916 | 0.5570 | 0.809 |
| T_seen | 195 | 173.0 | GINE_REF | 0.3783 | 0.5222 | 0.806 |
| T_seen | 195 | 173.0 | MORGAN_COUNT_KRR | 0.4649 | 0.6385 | 0.742 |
| T_seen | 195 | 173.0 | PATH_COUNT_KRR | 0.4306 | 0.5956 | 0.799 |
| T_seen | 195 | 173.0 | PATH_COUNT_RIDGE | 0.4122 | 0.5740 | 0.809 |
| T_seen | 195 | 173.0 | PSC_POOL_BAG_DIAG | 0.4396 | 0.6009 | 0.778 |
| T_seen | 195 | 173.0 | PSC_POOL_BAG_PAIR | 0.4430 | 0.6117 | 0.808 |
| T_seen | 195 | 173.0 | PSC_SUM_BAG_PAIR | 0.4368 | 0.6063 | 0.794 |
| T_unseen_both_frags_unseen | 24 | 19.4 | ACRA_REF | 0.4778 | 0.6036 | 0.597 |
| T_unseen_both_frags_unseen | 24 | 19.4 | GINE_REF | 0.3567 | 0.4598 | 0.611 |
| T_unseen_both_frags_unseen | 24 | 19.4 | MORGAN_COUNT_KRR | 0.4068 | 0.4889 | 0.625 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PATH_COUNT_KRR | 0.4258 | 0.5416 | 0.583 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PATH_COUNT_RIDGE | 0.4380 | 0.5846 | 0.625 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PSC_POOL_BAG_DIAG | 0.3190 | 0.4206 | 0.556 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PSC_POOL_BAG_PAIR | 0.3163 | 0.4256 | 0.611 |
| T_unseen_both_frags_unseen | 24 | 19.4 | PSC_SUM_BAG_PAIR | 0.3069 | 0.4185 | 0.736 |
| T_unseen_frag_both_seen | 156 | 135.8 | ACRA_REF | 0.4286 | 0.5975 | 0.859 |
| T_unseen_frag_both_seen | 156 | 135.8 | GINE_REF | 0.3748 | 0.5118 | 0.864 |
| T_unseen_frag_both_seen | 156 | 135.8 | MORGAN_COUNT_KRR | 0.4924 | 0.6557 | 0.799 |
| T_unseen_frag_both_seen | 156 | 135.8 | PATH_COUNT_KRR | 0.4666 | 0.6260 | 0.818 |
| T_unseen_frag_both_seen | 156 | 135.8 | PATH_COUNT_RIDGE | 0.4586 | 0.6259 | 0.844 |
| T_unseen_frag_both_seen | 156 | 135.8 | PSC_POOL_BAG_DIAG | 0.4894 | 0.6499 | 0.807 |
| T_unseen_frag_both_seen | 156 | 135.8 | PSC_POOL_BAG_PAIR | 0.4679 | 0.6466 | 0.853 |
| T_unseen_frag_both_seen | 156 | 135.8 | PSC_SUM_BAG_PAIR | 0.5018 | 0.6857 | 0.831 |
| T_unseen_one_frag_unseen | 100 | 84.8 | ACRA_REF | 0.4628 | 0.5911 | 0.677 |
| T_unseen_one_frag_unseen | 100 | 84.8 | GINE_REF | 0.4389 | 0.5582 | 0.707 |
| T_unseen_one_frag_unseen | 100 | 84.8 | MORGAN_COUNT_KRR | 0.4891 | 0.6133 | 0.750 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PATH_COUNT_KRR | 0.5271 | 0.6639 | 0.650 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PATH_COUNT_RIDGE | 0.5352 | 0.6923 | 0.680 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PSC_POOL_BAG_DIAG | 0.5583 | 0.7169 | 0.633 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PSC_POOL_BAG_PAIR | 0.4805 | 0.6128 | 0.707 |
| T_unseen_one_frag_unseen | 100 | 84.8 | PSC_SUM_BAG_PAIR | 0.4872 | 0.6478 | 0.770 |

### Novelty (bag_full) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | ACRA_REF | 0.3150 | 0.4304 |
| eta=0 | 2456 | 397 | GINE_REF | 0.2741 | 0.3723 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_DIAG | 0.4330 | 0.5835 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_PAIR | 0.4043 | 0.5460 |
| eta=0 | 2456 | 397 | PSC_SUM_BAG_PAIR | 0.4194 | 0.5631 |

### Novelty (order_full) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | ACRA_REF | 0.3150 | 0.4304 |
| eta=0 | 2456 | 397 | GINE_REF | 0.2741 | 0.3723 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_DIAG | 0.4330 | 0.5835 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_PAIR | 0.4043 | 0.5460 |
| eta=0 | 2456 | 397 | PSC_SUM_BAG_PAIR | 0.4194 | 0.5631 |

### Novelty (order_len3_L) — fit_fit, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2456 | 397 | ACRA_REF | 0.3150 | 0.4304 |
| eta=0 | 2456 | 397 | GINE_REF | 0.2741 | 0.3723 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_DIAG | 0.4330 | 0.5835 |
| eta=0 | 2456 | 397 | PSC_POOL_BAG_PAIR | 0.4043 | 0.5460 |
| eta=0 | 2456 | 397 | PSC_SUM_BAG_PAIR | 0.4194 | 0.5631 |

### Novelty (bag_full) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 21 | 11 | ACRA_REF | 0.4876 | 0.6688 |
| (0.25,0.5] | 21 | 11 | GINE_REF | 0.5310 | 0.6674 |
| (0.25,0.5] | 21 | 11 | MORGAN_COUNT_KRR | 0.5476 | 0.7493 |
| (0.25,0.5] | 21 | 11 | PATH_COUNT_KRR | 0.5984 | 0.8471 |
| (0.25,0.5] | 21 | 11 | PATH_COUNT_RIDGE | 0.6442 | 0.9404 |
| (0.25,0.5] | 21 | 11 | PSC_POOL_BAG_DIAG | 0.4912 | 0.6709 |
| (0.25,0.5] | 21 | 11 | PSC_POOL_BAG_PAIR | 0.4906 | 0.6579 |
| (0.25,0.5] | 21 | 11 | PSC_SUM_BAG_PAIR | 0.5354 | 0.7579 |
| (0.5,1.0] | 1 | 1 | ACRA_REF | 0.8650 | 0.8650 |
| (0.5,1.0] | 1 | 1 | GINE_REF | 1.2806 | 1.2806 |
| (0.5,1.0] | 1 | 1 | MORGAN_COUNT_KRR | 0.5346 | 0.5346 |
| (0.5,1.0] | 1 | 1 | PATH_COUNT_KRR | 0.8241 | 0.8241 |
| (0.5,1.0] | 1 | 1 | PATH_COUNT_RIDGE | 0.7253 | 0.7253 |
| (0.5,1.0] | 1 | 1 | PSC_POOL_BAG_DIAG | 1.1093 | 1.1093 |
| (0.5,1.0] | 1 | 1 | PSC_POOL_BAG_PAIR | 1.0327 | 1.0327 |
| (0.5,1.0] | 1 | 1 | PSC_SUM_BAG_PAIR | 0.3058 | 0.3058 |
| eta=0 | 212 | 44 | ACRA_REF | 0.3766 | 0.5009 |
| eta=0 | 212 | 44 | GINE_REF | 0.3250 | 0.4323 |
| eta=0 | 212 | 44 | MORGAN_COUNT_KRR | 0.4287 | 0.5621 |
| eta=0 | 212 | 44 | PATH_COUNT_KRR | 0.3985 | 0.5272 |
| eta=0 | 212 | 44 | PATH_COUNT_RIDGE | 0.3799 | 0.5058 |
| eta=0 | 212 | 44 | PSC_POOL_BAG_DIAG | 0.4105 | 0.5449 |
| eta=0 | 212 | 44 | PSC_POOL_BAG_PAIR | 0.3968 | 0.5383 |
| eta=0 | 212 | 44 | PSC_SUM_BAG_PAIR | 0.3749 | 0.5128 |

### Novelty (order_full) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 35 | 15 | ACRA_REF | 0.4410 | 0.6048 |
| (0.25,0.5] | 35 | 15 | GINE_REF | 0.5056 | 0.6564 |
| (0.25,0.5] | 35 | 15 | MORGAN_COUNT_KRR | 0.5047 | 0.6977 |
| (0.25,0.5] | 35 | 15 | PATH_COUNT_KRR | 0.5421 | 0.7680 |
| (0.25,0.5] | 35 | 15 | PATH_COUNT_RIDGE | 0.5310 | 0.7883 |
| (0.25,0.5] | 35 | 15 | PSC_POOL_BAG_DIAG | 0.4476 | 0.6040 |
| (0.25,0.5] | 35 | 15 | PSC_POOL_BAG_PAIR | 0.4508 | 0.6089 |
| (0.25,0.5] | 35 | 15 | PSC_SUM_BAG_PAIR | 0.4881 | 0.6900 |
| (0.5,1.0] | 2 | 2 | ACRA_REF | 0.4854 | 0.6231 |
| (0.5,1.0] | 2 | 2 | GINE_REF | 0.6662 | 0.9066 |
| (0.5,1.0] | 2 | 2 | MORGAN_COUNT_KRR | 0.3998 | 0.4219 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_KRR | 0.4605 | 0.5868 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_RIDGE | 0.3940 | 0.5148 |
| (0.5,1.0] | 2 | 2 | PSC_POOL_BAG_DIAG | 0.6319 | 0.7947 |
| (0.5,1.0] | 2 | 2 | PSC_POOL_BAG_PAIR | 0.6098 | 0.7491 |
| (0.5,1.0] | 2 | 2 | PSC_SUM_BAG_PAIR | 0.2655 | 0.3134 |
| eta=0 | 136 | 35 | ACRA_REF | 0.3683 | 0.5038 |
| eta=0 | 136 | 35 | GINE_REF | 0.3182 | 0.4415 |
| eta=0 | 136 | 35 | MORGAN_COUNT_KRR | 0.4294 | 0.5774 |
| eta=0 | 136 | 35 | PATH_COUNT_KRR | 0.3814 | 0.5286 |
| eta=0 | 136 | 35 | PATH_COUNT_RIDGE | 0.3803 | 0.5188 |
| eta=0 | 136 | 35 | PSC_POOL_BAG_DIAG | 0.4146 | 0.5563 |
| eta=0 | 136 | 35 | PSC_POOL_BAG_PAIR | 0.4171 | 0.5679 |
| eta=0 | 136 | 35 | PSC_SUM_BAG_PAIR | 0.4005 | 0.5449 |

### Novelty (order_len3_L) — heldout_heldout, D0

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 17 | 9 | ACRA_REF | 0.4701 | 0.6403 |
| (0.25,0.5] | 17 | 9 | GINE_REF | 0.4992 | 0.6275 |
| (0.25,0.5] | 17 | 9 | MORGAN_COUNT_KRR | 0.4560 | 0.6961 |
| (0.25,0.5] | 17 | 9 | PATH_COUNT_KRR | 0.4960 | 0.7592 |
| (0.25,0.5] | 17 | 9 | PATH_COUNT_RIDGE | 0.5627 | 0.8727 |
| (0.25,0.5] | 17 | 9 | PSC_POOL_BAG_DIAG | 0.4995 | 0.6678 |
| (0.25,0.5] | 17 | 9 | PSC_POOL_BAG_PAIR | 0.4788 | 0.6477 |
| (0.25,0.5] | 17 | 9 | PSC_SUM_BAG_PAIR | 0.5806 | 0.7994 |
| (0.5,1.0] | 2 | 2 | ACRA_REF | 0.6510 | 0.6872 |
| (0.5,1.0] | 2 | 2 | GINE_REF | 1.0087 | 1.0474 |
| (0.5,1.0] | 2 | 2 | MORGAN_COUNT_KRR | 0.5447 | 0.5448 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_KRR | 0.7897 | 0.7905 |
| (0.5,1.0] | 2 | 2 | PATH_COUNT_RIDGE | 0.8031 | 0.8068 |
| (0.5,1.0] | 2 | 2 | PSC_POOL_BAG_DIAG | 0.6982 | 0.8381 |
| (0.5,1.0] | 2 | 2 | PSC_POOL_BAG_PAIR | 0.7269 | 0.7985 |
| (0.5,1.0] | 2 | 2 | PSC_SUM_BAG_PAIR | 0.4573 | 0.5088 |
| eta=0 | 338 | 63 | ACRA_REF | 0.4027 | 0.5629 |
| eta=0 | 338 | 63 | GINE_REF | 0.3656 | 0.4955 |
| eta=0 | 338 | 63 | MORGAN_COUNT_KRR | 0.4789 | 0.6352 |
| eta=0 | 338 | 63 | PATH_COUNT_KRR | 0.4502 | 0.6005 |
| eta=0 | 338 | 63 | PATH_COUNT_RIDGE | 0.4285 | 0.5774 |
| eta=0 | 338 | 63 | PSC_POOL_BAG_DIAG | 0.4744 | 0.6371 |
| eta=0 | 338 | 63 | PSC_POOL_BAG_PAIR | 0.4529 | 0.6208 |
| eta=0 | 338 | 63 | PSC_SUM_BAG_PAIR | 0.4475 | 0.6164 |

### |Δ| ≥ 0.25 sensitivity (D0 heldout pairs)

| model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|
| ACRA_REF | 0.4864 | 0.6502 | 0.876 |
| GINE_REF | 0.4659 | 0.5985 | 0.876 |
| MORGAN_COUNT_KRR | 0.5425 | 0.7042 | 0.838 |
| PATH_COUNT_KRR | 0.5421 | 0.6970 | 0.835 |
| PATH_COUNT_RIDGE | 0.5255 | 0.6950 | 0.869 |
| PSC_POOL_BAG_DIAG | 0.5739 | 0.7255 | 0.845 |
| PSC_POOL_BAG_PAIR | 0.5372 | 0.6943 | 0.876 |
| PSC_SUM_BAG_PAIR | 0.5512 | 0.7130 | 0.892 |

Counts: {"components_all": 458, "direction_items_removed": 3, "scope_counts": {"fit_fit": 2456, "fit_heldout": 350, "heldout_heldout": 413}, "unique_pairs": 3219, "unknown_fraction": {"atoms": 0.0, "bonds": 0.0, "n_atoms": 90514.0, "n_bonds": 98561.0}}

## Pair-level metrics, D1

### By matched-pair scope (unique unordered pairs, unweighted)

| scope | model | pairs scored | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|
| fit_fit | ACRA_REF | 2501 | 0.3401 | 0.4651 | 0.830 |
| fit_fit | GINE_REF | 2501 | 0.2846 | 0.3872 | 0.854 |
| fit_fit | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_fit | PATH_COUNT_KRR | 0 | – | – | – |
| fit_fit | PATH_COUNT_RIDGE | 0 | – | – | – |
| fit_fit | PSC_POOL_BAG_PAIR | 2501 | 0.4225 | 0.5667 | 0.790 |
| fit_fit | PSC_SUM_BAG_PAIR | 2501 | 0.4566 | 0.6157 | 0.772 |
| fit_heldout | ACRA_REF | 444 | 0.4260 | 0.5750 | 0.834 |
| fit_heldout | GINE_REF | 444 | 0.3943 | 0.5303 | 0.854 |
| fit_heldout | MORGAN_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PATH_COUNT_KRR | 0 | – | – | – |
| fit_heldout | PATH_COUNT_RIDGE | 0 | – | – | – |
| fit_heldout | PSC_POOL_BAG_PAIR | 444 | 0.5588 | 0.7458 | 0.761 |
| fit_heldout | PSC_SUM_BAG_PAIR | 444 | 0.5823 | 0.7689 | 0.733 |
| heldout_heldout | ACRA_REF | 274 | 0.3932 | 0.5433 | 0.797 |
| heldout_heldout | GINE_REF | 274 | 0.3455 | 0.4779 | 0.801 |
| heldout_heldout | MORGAN_COUNT_KRR | 274 | 0.4344 | 0.6070 | 0.751 |
| heldout_heldout | PATH_COUNT_KRR | 274 | 0.4064 | 0.5989 | 0.784 |
| heldout_heldout | PATH_COUNT_RIDGE | 274 | 0.4021 | 0.5814 | 0.792 |
| heldout_heldout | PSC_POOL_BAG_PAIR | 274 | 0.4374 | 0.6018 | 0.773 |
| heldout_heldout | PSC_SUM_BAG_PAIR | 274 | 0.4255 | 0.5823 | 0.783 |

### By support class (heldout × heldout, cut-row weighted)

| support class | pairs | weight | model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|---|---|---|
| T_seen | 153 | 136.3 | ACRA_REF | 0.3248 | 0.4439 | 0.856 |
| T_seen | 153 | 136.3 | GINE_REF | 0.3284 | 0.4465 | 0.846 |
| T_seen | 153 | 136.3 | MORGAN_COUNT_KRR | 0.3788 | 0.5301 | 0.826 |
| T_seen | 153 | 136.3 | PATH_COUNT_KRR | 0.3432 | 0.4889 | 0.846 |
| T_seen | 153 | 136.3 | PATH_COUNT_RIDGE | 0.3267 | 0.4587 | 0.846 |
| T_seen | 153 | 136.3 | PSC_POOL_BAG_PAIR | 0.3710 | 0.4939 | 0.815 |
| T_seen | 153 | 136.3 | PSC_SUM_BAG_PAIR | 0.3569 | 0.4793 | 0.832 |
| T_unseen_both_frags_unseen | 15 | 10.7 | ACRA_REF | 0.3930 | 0.6236 | 0.600 |
| T_unseen_both_frags_unseen | 15 | 10.7 | GINE_REF | 0.2623 | 0.4515 | 0.733 |
| T_unseen_both_frags_unseen | 15 | 10.7 | MORGAN_COUNT_KRR | 0.4116 | 0.6846 | 0.467 |
| T_unseen_both_frags_unseen | 15 | 10.7 | PATH_COUNT_KRR | 0.4177 | 0.7053 | 0.600 |
| T_unseen_both_frags_unseen | 15 | 10.7 | PATH_COUNT_RIDGE | 0.4102 | 0.6792 | 0.533 |
| T_unseen_both_frags_unseen | 15 | 10.7 | PSC_POOL_BAG_PAIR | 0.4007 | 0.7006 | 0.467 |
| T_unseen_both_frags_unseen | 15 | 10.7 | PSC_SUM_BAG_PAIR | 0.4565 | 0.7079 | 0.567 |
| T_unseen_frag_both_seen | 84 | 71.0 | ACRA_REF | 0.4194 | 0.5358 | 0.833 |
| T_unseen_frag_both_seen | 84 | 71.0 | GINE_REF | 0.3534 | 0.4722 | 0.815 |
| T_unseen_frag_both_seen | 84 | 71.0 | MORGAN_COUNT_KRR | 0.4303 | 0.6403 | 0.821 |
| T_unseen_frag_both_seen | 84 | 71.0 | PATH_COUNT_KRR | 0.4299 | 0.6282 | 0.821 |
| T_unseen_frag_both_seen | 84 | 71.0 | PATH_COUNT_RIDGE | 0.4282 | 0.6063 | 0.833 |
| T_unseen_frag_both_seen | 84 | 71.0 | PSC_POOL_BAG_PAIR | 0.4479 | 0.6084 | 0.851 |
| T_unseen_frag_both_seen | 84 | 71.0 | PSC_SUM_BAG_PAIR | 0.3999 | 0.5370 | 0.839 |
| T_unseen_one_frag_unseen | 68 | 56.0 | ACRA_REF | 0.5266 | 0.7235 | 0.672 |
| T_unseen_one_frag_unseen | 68 | 56.0 | GINE_REF | 0.3931 | 0.5562 | 0.716 |
| T_unseen_one_frag_unseen | 68 | 56.0 | MORGAN_COUNT_KRR | 0.5795 | 0.7139 | 0.567 |
| T_unseen_one_frag_unseen | 68 | 56.0 | PATH_COUNT_KRR | 0.5284 | 0.7605 | 0.642 |
| T_unseen_one_frag_unseen | 68 | 56.0 | PATH_COUNT_RIDGE | 0.5512 | 0.7666 | 0.642 |
| T_unseen_one_frag_unseen | 68 | 56.0 | PSC_POOL_BAG_PAIR | 0.5924 | 0.7821 | 0.619 |
| T_unseen_one_frag_unseen | 68 | 56.0 | PSC_SUM_BAG_PAIR | 0.6193 | 0.7974 | 0.612 |

### Novelty (bag_full) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2501 | 394 | ACRA_REF | 0.3401 | 0.4651 |
| eta=0 | 2501 | 394 | GINE_REF | 0.2846 | 0.3872 |
| eta=0 | 2501 | 394 | PSC_POOL_BAG_PAIR | 0.4225 | 0.5667 |
| eta=0 | 2501 | 394 | PSC_SUM_BAG_PAIR | 0.4566 | 0.6157 |

### Novelty (order_full) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2501 | 394 | ACRA_REF | 0.3401 | 0.4651 |
| eta=0 | 2501 | 394 | GINE_REF | 0.2846 | 0.3872 |
| eta=0 | 2501 | 394 | PSC_POOL_BAG_PAIR | 0.4225 | 0.5667 |
| eta=0 | 2501 | 394 | PSC_SUM_BAG_PAIR | 0.4566 | 0.6157 |

### Novelty (order_len3_L) — fit_fit, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| eta=0 | 2501 | 394 | ACRA_REF | 0.3401 | 0.4651 |
| eta=0 | 2501 | 394 | GINE_REF | 0.2846 | 0.3872 |
| eta=0 | 2501 | 394 | PSC_POOL_BAG_PAIR | 0.4225 | 0.5667 |
| eta=0 | 2501 | 394 | PSC_SUM_BAG_PAIR | 0.4566 | 0.6157 |

### Novelty (bag_full) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 36 | 15 | ACRA_REF | 0.6295 | 0.8574 |
| (0.25,0.5] | 36 | 15 | GINE_REF | 0.4826 | 0.6525 |
| (0.25,0.5] | 36 | 15 | MORGAN_COUNT_KRR | 0.6069 | 0.7998 |
| (0.25,0.5] | 36 | 15 | PATH_COUNT_KRR | 0.6559 | 0.9114 |
| (0.25,0.5] | 36 | 15 | PATH_COUNT_RIDGE | 0.6648 | 0.9434 |
| (0.25,0.5] | 36 | 15 | PSC_POOL_BAG_PAIR | 0.6308 | 0.8843 |
| (0.25,0.5] | 36 | 15 | PSC_SUM_BAG_PAIR | 0.6387 | 0.8628 |
| (0.5,1.0] | 7 | 5 | ACRA_REF | 0.2238 | 0.2800 |
| (0.5,1.0] | 7 | 5 | GINE_REF | 0.1612 | 0.1864 |
| (0.5,1.0] | 7 | 5 | MORGAN_COUNT_KRR | 0.1462 | 0.1609 |
| (0.5,1.0] | 7 | 5 | PATH_COUNT_KRR | 0.0747 | 0.0849 |
| (0.5,1.0] | 7 | 5 | PATH_COUNT_RIDGE | 0.2020 | 0.2904 |
| (0.5,1.0] | 7 | 5 | PSC_POOL_BAG_PAIR | 0.1339 | 0.1666 |
| (0.5,1.0] | 7 | 5 | PSC_SUM_BAG_PAIR | 0.2502 | 0.3292 |
| eta=0 | 120 | 39 | ACRA_REF | 0.2875 | 0.4018 |
| eta=0 | 120 | 39 | GINE_REF | 0.2954 | 0.4075 |
| eta=0 | 120 | 39 | MORGAN_COUNT_KRR | 0.3054 | 0.4348 |
| eta=0 | 120 | 39 | PATH_COUNT_KRR | 0.2972 | 0.4204 |
| eta=0 | 120 | 39 | PATH_COUNT_RIDGE | 0.2952 | 0.4083 |
| eta=0 | 120 | 39 | PSC_POOL_BAG_PAIR | 0.3628 | 0.4859 |
| eta=0 | 120 | 39 | PSC_SUM_BAG_PAIR | 0.3387 | 0.4619 |

### Novelty (order_full) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 51 | 18 | ACRA_REF | 0.6506 | 0.8470 |
| (0.25,0.5] | 51 | 18 | GINE_REF | 0.4870 | 0.6565 |
| (0.25,0.5] | 51 | 18 | MORGAN_COUNT_KRR | 0.6528 | 0.8445 |
| (0.25,0.5] | 51 | 18 | PATH_COUNT_KRR | 0.6363 | 0.8900 |
| (0.25,0.5] | 51 | 18 | PATH_COUNT_RIDGE | 0.5975 | 0.8698 |
| (0.25,0.5] | 51 | 18 | PSC_POOL_BAG_PAIR | 0.6292 | 0.8905 |
| (0.25,0.5] | 51 | 18 | PSC_SUM_BAG_PAIR | 0.6476 | 0.8725 |
| (0.5,1.0] | 9 | 6 | ACRA_REF | 0.2233 | 0.2791 |
| (0.5,1.0] | 9 | 6 | GINE_REF | 0.1532 | 0.1826 |
| (0.5,1.0] | 9 | 6 | MORGAN_COUNT_KRR | 0.1439 | 0.1671 |
| (0.5,1.0] | 9 | 6 | PATH_COUNT_KRR | 0.1325 | 0.1893 |
| (0.5,1.0] | 9 | 6 | PATH_COUNT_RIDGE | 0.2513 | 0.3263 |
| (0.5,1.0] | 9 | 6 | PSC_POOL_BAG_PAIR | 0.1567 | 0.1938 |
| (0.5,1.0] | 9 | 6 | PSC_SUM_BAG_PAIR | 0.2267 | 0.3012 |
| eta=0 | 101 | 30 | ACRA_REF | 0.2478 | 0.3246 |
| eta=0 | 101 | 30 | GINE_REF | 0.2804 | 0.3709 |
| eta=0 | 101 | 30 | MORGAN_COUNT_KRR | 0.2818 | 0.3755 |
| eta=0 | 101 | 30 | PATH_COUNT_KRR | 0.2673 | 0.3521 |
| eta=0 | 101 | 30 | PATH_COUNT_RIDGE | 0.2631 | 0.3392 |
| eta=0 | 101 | 30 | PSC_POOL_BAG_PAIR | 0.3325 | 0.3961 |
| eta=0 | 101 | 30 | PSC_SUM_BAG_PAIR | 0.3119 | 0.3873 |

### Novelty (order_len3_L) — heldout_heldout, D1

| bucket | pairs | components | model | Δ MAE | Δ RMSE |
|---|---|---|---|---|---|
| (0.25,0.5] | 27 | 10 | ACRA_REF | 0.6956 | 0.8785 |
| (0.25,0.5] | 27 | 10 | GINE_REF | 0.5014 | 0.6619 |
| (0.25,0.5] | 27 | 10 | MORGAN_COUNT_KRR | 0.6921 | 0.9163 |
| (0.25,0.5] | 27 | 10 | PATH_COUNT_KRR | 0.6566 | 0.9471 |
| (0.25,0.5] | 27 | 10 | PATH_COUNT_RIDGE | 0.6282 | 0.9484 |
| (0.25,0.5] | 27 | 10 | PSC_POOL_BAG_PAIR | 0.7263 | 1.0166 |
| (0.25,0.5] | 27 | 10 | PSC_SUM_BAG_PAIR | 0.7677 | 0.9895 |
| (0.5,1.0] | 11 | 5 | ACRA_REF | 0.2613 | 0.3345 |
| (0.5,1.0] | 11 | 5 | GINE_REF | 0.1612 | 0.1898 |
| (0.5,1.0] | 11 | 5 | MORGAN_COUNT_KRR | 0.1719 | 0.1879 |
| (0.5,1.0] | 11 | 5 | PATH_COUNT_KRR | 0.1197 | 0.1752 |
| (0.5,1.0] | 11 | 5 | PATH_COUNT_RIDGE | 0.2702 | 0.3316 |
| (0.5,1.0] | 11 | 5 | PSC_POOL_BAG_PAIR | 0.1730 | 0.2017 |
| (0.5,1.0] | 11 | 5 | PSC_SUM_BAG_PAIR | 0.2202 | 0.2848 |
| eta=0 | 190 | 51 | ACRA_REF | 0.3132 | 0.4062 |
| eta=0 | 190 | 51 | GINE_REF | 0.2975 | 0.3990 |
| eta=0 | 190 | 51 | MORGAN_COUNT_KRR | 0.3613 | 0.4969 |
| eta=0 | 190 | 51 | PATH_COUNT_KRR | 0.3373 | 0.4648 |
| eta=0 | 190 | 51 | PATH_COUNT_RIDGE | 0.3402 | 0.4523 |
| eta=0 | 190 | 51 | PSC_POOL_BAG_PAIR | 0.3667 | 0.4700 |
| eta=0 | 190 | 51 | PSC_SUM_BAG_PAIR | 0.3518 | 0.4574 |

### |Δ| ≥ 0.25 sensitivity (D1 heldout pairs)

| model | Δ MAE | Δ RMSE | direction acc. |
|---|---|---|---|
| ACRA_REF | 0.4451 | 0.6050 | 0.887 |
| GINE_REF | 0.4188 | 0.5523 | 0.903 |
| MORGAN_COUNT_KRR | 0.4831 | 0.6707 | 0.855 |
| PATH_COUNT_KRR | 0.4708 | 0.6749 | 0.860 |
| PATH_COUNT_RIDGE | 0.4516 | 0.6457 | 0.882 |
| PSC_POOL_BAG_PAIR | 0.5088 | 0.6712 | 0.866 |
| PSC_SUM_BAG_PAIR | 0.4952 | 0.6535 | 0.871 |

Counts: {"components_all": 458, "direction_items_removed": 5, "scope_counts": {"fit_fit": 2501, "fit_heldout": 444, "heldout_heldout": 274}, "unique_pairs": 3219, "unknown_fraction": {"atoms": 0.0, "bonds": 0.0, "n_atoms": 90514.0, "n_bonds": 98561.0}}

## Pair-residual decomposition (D0 heldout × heldout, component weights)

| model | E[(e_b−e_a)²] | E[e_a²] | E[e_b²] | E[e_a e_b] | corr(e_a,e_b) | identity err |
|---|---|---|---|---|---|---|
| ACRA_REF | 0.2679 | 0.3747 | 0.2718 | 0.1893 | 0.440 | 1.7e-16 |
| PSC_POOL_BAG_DIAG | 0.4001 | 0.5829 | 0.4565 | 0.3196 | 0.589 | 1.1e-16 |
| PSC_POOL_BAG_PAIR | 0.3280 | 0.5800 | 0.4316 | 0.3418 | 0.467 | 0.0e+00 |
| PSC_SUM_BAG_PAIR | 0.3077 | 0.6225 | 0.5041 | 0.4094 | 0.553 | 1.1e-16 |


## Molecule-size strata (D0 dev, seed 0) and worst-error sensitivity

| model | overall RMSE | MAE | Q1 | Q2 | Q3 | Q4 | top-5 removed |
|---|---|---|---|---|---|---|---|
| ACRA_REF | 0.5809 | 0.4481 | 0.6683 | 0.5928 | 0.5573 | 0.5362 | 0.5458 |
| COUNT_ONLY_RIDGE | 1.1614 | 0.9016 | 1.1849 | 1.1977 | 0.9664 | 1.2865 | 1.1174 |
| GINE_REF | 0.5216 | 0.3985 | 0.5591 | 0.5346 | 0.5088 | 0.5005 | 0.4904 |
| PSC_POOL_BAG_PAIR | 0.6933 | 0.5266 | 0.7525 | 0.7502 | 0.6346 | 0.6671 | 0.6444 |

Quartile edges (fit atom counts): [21.0, 27.0, 32.0]

| model (union of 15 worst removed) | RMSE |
|---|---|
| ACRA_REF | 0.5226 |
| COUNT_ONLY_RIDGE | 1.1053 |
| GINE_REF | 0.4790 |
| PSC_POOL_BAG_PAIR | 0.6267 |


## SUM edit ledger cases

| rule | a | b | true Δ | ACRA_B* Δ | eta | T seen | frags seen | ΔF_S | ΔF_C | ΔF_X |
|---|---|---|---|---|---|---|---|---|---|---|
| T unseen / fragments seen · largest ACRA error | 114 | 3789 | 0.2000 | -0.9213 | 0.000 | False | True | – | – | – |
| T unseen / fragments seen · largest ACRA error | 570 | 3184 | 1.3000 | 0.7265 | 0.009 | False | True | – | – | – |
| T unseen / fragments seen · smallest ACRA error | 1594 | 2299 | -0.6300 | -0.6201 | 0.200 | False | True | – | – | – |
| T unseen / fragments seen · smallest ACRA error | 2501 | 3184 | -0.4000 | -0.3803 | 0.000 | False | True | – | – | – |
| high eta · largest ACRA error | 181 | 1944 | -0.7600 | 0.8045 | 0.404 | True | False | – | – | – |
| high eta · smallest ACRA error | 1560 | 4135 | 0.4700 | 0.5313 | 0.278 | False | False | – | – | – |
| seen T · largest positive Δ | 53 | 3051 | 3.6100 | 2.7562 | 0.018 | True | True | – | – | – |
| seen T · most negative Δ | 815 | 3159 | -1.0200 | -0.4790 | 0.000 | True | True | – | – | – |

Ledger model: PSC_SUM_BAG_PAIR (champion PSC_POOL_BAG_PAIR); identity error 1.68e-06 (standardised units).

## Compute budget

| item | value |
|---|---|
| GPU-hours (all launched jobs) | 3.477 |
| GPU-hours (completed runs) | 3.462 |
| train-loop hours | 3.182 |
| train-loop fraction of GPU wall | 0.919 |
| CPU-hours (closed-form baselines) | 0.143 |
| budget | 16 GPU-hours |
