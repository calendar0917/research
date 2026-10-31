# LCRC Lipophilicity v3 — chemistry-feature pilot

**Status:** **COMPLETE** — two unconditional GPU runs; conditional `C_Chem12` not run (gate not met).  
**Protocol:** `lcrc-lipo-chem-feature-pilot-v3`  
**Protocol freeze SHA-256:** `72f2e2142be7eedfe056bcbb948118d0d32ce73f3167fbb356847cf9acbfa14c`  
**GPU authorization:** `analysis/gpu_authorization_v3.json`, pinned to that freeze, host `res`, pool `res-gpu0`, cap 0.30 A100 h.  
**Verdict:** **`NO_GLOBAL_DESCRIPTOR_GAIN`**  
**GPU ledger:** **0.0762 h / 0.30 h**, 2 charged jobs, 0.2238 h remaining.

## 1. Question

Does label-free global physicochemical information limit LCRC-A on Lipophilicity? This was an exploratory, single-fold pilot. The only supervised scope was the frozen v2 Fold-1 train/dev split of Fit2352; Monitor, Reserved and official valid/test labels were not read.

The v2 comparators are:

| Arm | v2 run | Dev RMSE | Dev MAE |
|---|---|---|---:|---:|
| A | `F1_A_sel_s0` | 0.7403427006133139 | 0.546694119695902 |
| C | `F1_C_sel_s0` | 0.7264913055478032 | 0.5387064120516623 |

## 2. What was added

`Chem12`, in the fixed order:

`TPSA, MolWt, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAromaticRings, NumAliphaticRings, NumSaturatedRings, RingCount, FractionCSP3, HeavyAtomCount, MolMR`

`Chem13` is exactly `Chem12` followed by `MolLogP`.

Only the `rho` global input is widened; local encoders, S2, `f`, atom51, bond13 and scalar9 are unchanged. Features come only from unlabeled canonical SMILES, are aligned by canonical-SMILES SHA-256 for all 2,352 Fit rows, and are normalised with Fold-1 training statistics only. The zero-extra forward is equivalent to the original model (max diff `0.0` for A arms, `1.19e-7` for C), and every new column receives finite, non-zero gradient.

Added parameters: `A_Chem12 +768`, `A_Chem13 +832`, `C_Chem12 +768` (0.31–0.34% of the original model).

## 3. Results

All rows are Fold 1, seed 0, recipe R3. `ΔRMSE` is `RMSE(historical same-architecture) − RMSE(v3)`; **positive would favour the chemistry model**.

| 模型 | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE | ΔRMSE vs 同架构原模型 |
|---|---:|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.269630 | **0.740343** | **0.546694** | 0.000000 (reference) |
| A_Chem12 | 1 | 0 | 0.263954 | 0.749970 | 0.547924 | **−0.009628** |
| A_Chem13 | 1 | 0 | 0.255995 | 0.757425 | 0.551072 | **−0.017082** |
| C-R3 (v2 historical) | 1 | 0 | 0.299892 | **0.726491** | **0.538706** | 0.000000 (reference) |
| C_Chem12 (conditional) | 1 | 0 | — | — | — | **not run** |

Both chemistry arms fit the training fold slightly better than A-R3 (0.26395 and 0.25599 vs 0.26963) but are worse on dev. This is the opposite of a useful prior: the extra global descriptors increased capacity and overfitting, not generalisation.

### Paired and bootstrap evidence

| Comparison | ΔRMSE | 95% bootstrap CI | ΔMAE | Molecule win rate |
|---|---:|---|---:|---:|
| A_Chem12 vs A-R3 | −0.009628 | [−0.029179, +0.009654] | −0.001230 | 50.5% |
| A_Chem13 vs A-R3 | −0.017082 | [−0.043579, +0.006949] | −0.004378 | 50.4% |

Both RMSE and MAE point in the same (negative) direction; the confidence intervals include zero but the point estimates are not material gains. Adding `MolLogP` on top of Chem12 changes RMSE by **−0.007455** (`A_Chem13 − A_Chem12`), so the direct logP prior does **not** rescue the result.

The pre-registered tail diagnostic is not a positive claim here: because both deltas are negative, the tail-stability flag is false by construction (`retained_fraction` is undefined for a non-positive full delta). The decision below uses the pre-registered 0.03 materiality rule, not a tail claim.

## 4. Why the gain did not appear — post-hoc subgroup check

The CPU phase had identified low-MolLogP, high-TPSA and high-HDonors molecules as error-concentrated. The table below asks whether the chemistry models improved exactly those groups. It is **post-hoc descriptive**, not used for the decision.

| Group | n | A-R3 RMSE | A_Chem12 RMSE | A_Chem13 RMSE | Δ A_Chem12 | Δ A_Chem13 |
|---|---:|---:|---:|---:|---:|---:|
| MolLogP ≤ 2.337 | 114 | 0.9660 | 1.0087 | 1.0224 | **−0.0427** | **−0.0564** |
| MolLogP (2.337, 3.266] | 234 | 0.6385 | 0.6486 | 0.6683 | −0.0101 | −0.0298 |
| MolLogP (3.266, 4.057] | 233 | 0.7130 | 0.7061 | 0.6997 | +0.0069 | +0.0133 |
| MolLogP > 4.057 | 203 | 0.7344 | 0.7374 | 0.7429 | −0.0029 | −0.0085 |
| TPSA > 97.54 | 206 | 0.7968 | 0.8087 | 0.8279 | **−0.0119** | **−0.0311** |
| NumHDonors ≥ 4 | 24 | 1.4076 | 1.5942 | 1.6426 | **−0.1866** | **−0.2350** |

The two most chemistry-relevant extremes — low MolLogP and high HDonors — became **worse**, not better. The only local improvements are in the middle MolLogP bin and some HDonors=1/2/3 bins; they do not offset the losses. This is consistent with the global descriptors acting as extra capacity rather than a targeted fix for the polar/low-logP failure mode.

Full per-group numbers: `analysis/subgroup_results_v3.json`.

## 5. Decision

**`NO_GLOBAL_DESCRIPTOR_GAIN`.** Neither Chem12 nor Chem13 reaches the pre-registered 0.03 RMSE materiality threshold; both point estimates are negative and MAE moves in the same direction. The conditional `C_Chem12` run was therefore **not executed**: its gate required `delta_rmse ≥ 0.03`, `delta_mae > 0`, and non-tail-dependent improvement from `A_Chem12`; the observed gate values were `delta_rmse = −0.009628`, `delta_mae = −0.001230`.

Under the frozen rules this closes the **global descriptor supplementation** route for LCRC-A on this fold: adding TPSA/MolLogP/HBA/HBD/MolWt-type summaries to `rho` did not help. It does **not** rule out atom-level or local chemical representations, architecture changes, or a different supervision regime; those would require a new protocol, new freeze and new authorization.

No further seed, fold, SSL arm, descriptor combination, full-Fit retrain or Monitor evaluation was run.

## 6. GPU accounting and regime

| Run | Arm | GPU | Wall / charged | Result |
|---|---|---|---:|---|
| `F1_A_Chem12_R3_s0` | `A_Chem12` | A100-SXM4-40GB | 124.0 s / 0.03667 h | dev RMSE 0.74997 |
| `F1_A_Chem13_R3_s0` | `A_Chem13` | A100-SXM4-40GB | 134.4 s / 0.03956 h | dev RMSE 0.75743 |

- Host `res`, node `a100-2`, pool `res-gpu0`, driver `550.163.01`, torch `2.5.1+cu124`, python `3.12.14`.
- `F1_A_Chem12_R3_s0` used `c1f9f83289af`; `F1_A_Chem13_R3_s0` used `feac117b0e8b`. Only the ledger file differed between those commits; the v3 code hashes are identical and both runs used the SHA-pinned authorization.
- Total charged: **0.076225 h**; cap **0.30 h**; remaining **0.223775 h**.
- Three infrastructure attempts failed before training (`v3-A12`, `v3-A12b`: missing v2 cache files on the remote; `v3-A13`: remote tracked ledger dirty after the first pull). They produced no `train.json`, are not charged, and are not part of the result. The successful reruns are the two entries above.
- The failed attempts did not consume a material GPU budget; the ledger is based only on completed run artifacts.

## 7. Integrity and artifacts

- v1 and v2 were not modified. The v2 guard checks 237 files/records, including both historical checkpoints and prediction vectors; it passed on the GPU host at each training launch.
- Monitor, Reserved and official valid/test labels were never read.
- `train_norm_fingerprint` for both runs is `33a563a7a181…9810`, matching the historical v2 fold-1 train norm; target mean/scale match `2.1474617346938776` / `1.2227341856846436`.
- Result table: `analysis/table_v3.csv`
- Machine-readable results: `analysis/results_v3.json`, `analysis/paired_v3.json`, `analysis/decision_v3.json`
- Post-hoc subgroup table: `analysis/subgroup_results_v3.json`
- GPU ledger: `analysis/gpu_ledger_v3.json`
- `rr` execution-regime provenance: `analysis/gpu_run_meta_v3.json`
- Run artifacts: `analysis/runs/F1_A_Chem12_R3_s0/`, `analysis/runs/F1_A_Chem13_R3_s0/`
- Frozen protocol and CPU audits: `analysis/protocol_freeze_v3.json`, `analysis/cpu_gates_v3.json`, `analysis/frozen_extras_v3.{json,npz}`

## 8. Reproduction

The frozen CPU phase can be re-verified with:

```bash
uv run --group lipo pytest -q tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/tests
uv run --group lipo python tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/code/run_v3.py verify
```

The two GPU runs are reproducible from the committed revision, the freeze above, and the authorization file; `run_v3.py analyze` regenerates the table, paired statistics and decision from the run artifacts.
