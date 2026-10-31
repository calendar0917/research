# Decision — LCRC Lipophilicity v3 chemistry-feature pilot

**Date:** 2026-10-31  
**Protocol:** `lcrc-lipo-chem-feature-pilot-v3`  
**Status:** **COMPLETE**  
**Freeze SHA-256:** `72f2e2142be7eedfe056bcbb948118d0d32ce73f3167fbb356847cf9acbfa14c`  
**Verdict:** **`NO_GLOBAL_DESCRIPTOR_GAIN`**  
**GPU:** 0.076225 h of 0.30 h; two unconditional A runs; conditional C not run.

## Result

Fold 1, seed 0, recipe R3. `ΔRMSE = RMSE(A-R3) − RMSE(v3)`; positive would favour the chemistry model.

| Model | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE | ΔRMSE vs A-R3 |
|---|---:|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.269630 | 0.740343 | 0.546694 | 0.000000 |
| A_Chem12 | 1 | 0 | 0.263954 | 0.749970 | 0.547924 | **−0.009628** |
| A_Chem13 | 1 | 0 | 0.255995 | 0.757425 | 0.551072 | **−0.017082** |
| C-R3 (v2 historical) | 1 | 0 | 0.299892 | 0.726491 | 0.538706 | reference |
| C_Chem12 | 1 | 0 | — | — | — | **not run** |

Both chemistry arms improved the training fit slightly but were worse on dev. The 95% paired bootstrap intervals for ΔRMSE are `[−0.029179, +0.009654]` (Chem12) and `[−0.043579, +0.006949]` (Chem13); MAE also moves negative. `MolLogP` on top of Chem12 changes RMSE by `−0.007455`, so the direct logP prior does not rescue the result.

## Decision rule applied

The pre-registered rule was:

- `GENERAL_CHEM_PRIOR_SUPPORTED`: Chem12 Δ ≥ 0.03, tail-robust, Chem13 Δ ≥ 0.03.
- `DIRECT_LOGP_PRIOR_ONLY`: Chem12 < 0.03, Chem13 Δ ≥ 0.03, tail-robust.
- `NO_GLOBAL_DESCRIPTOR_GAIN`: both < 0.03.
- `INCONCLUSIVE`: tail-dependent, MolLogP removes the gain, or RMSE/MAE conflict.

Both observed ΔRMSE values are negative and well below 0.03, and MAE agrees. The verdict is **`NO_GLOBAL_DESCRIPTOR_GAIN`**.

The conditional `C_Chem12` gate required `A_Chem12` ΔRMSE ≥ 0.03, ΔMAE > 0 and non-tail-dependent improvement. The observed values were `ΔRMSE = −0.009628`, `ΔMAE = −0.001230`; the gate was not met and `C_Chem12` was **not executed**. Running it would not test the pre-registered hypothesis and would spend budget without a trigger.

## Interpretation

Adding global TPSA/MolWt/HBA/HBD/MolLogP-type summaries to the frozen LCRC-A `rho` input did **not** improve Lipophilicity generalisation on Fold 1. The direction is slightly negative, and the post-hoc subgroup check shows the most chemistry-relevant extremes did not improve: low-MolLogP dev RMSE went from `0.9660` (A-R3) to `1.0087` (Chem12) / `1.0224` (Chem13), high-TPSA from `0.7968` to `0.8087` / `0.8279`, and NumHDonors ≥ 4 from `1.4076` to `1.5942` / `1.6426`. That subgroup check is descriptive only and did not drive the decision.

This closes the **global descriptor supplementation** route for LCRC-A under this protocol. It does not rule out atom-level/local chemical representations, architecture changes, or different supervision; those require a new hypothesis, protocol and freeze. No further seed, fold, SSL arm, descriptor combination, full-Fit retrain or Monitor evaluation is authorized by this decision.

## Artifacts

- `REPORT.md`
- `analysis/table_v3.csv`
- `analysis/results_v3.json`
- `analysis/paired_v3.json`
- `analysis/decision_v3.json`
- `analysis/subgroup_results_v3.json`
- `analysis/gpu_ledger_v3.json`
- `analysis/runs/F1_A_Chem12_R3_s0/`, `analysis/runs/F1_A_Chem13_R3_s0/`

The two GPU runs used host `res`, pool `res-gpu0`, A100-SXM4-40GB, driver `550.163.01`, torch `2.5.1+cu124`. Both were authorized against the freeze SHA above; the v2 immutability guard passed at launch.
