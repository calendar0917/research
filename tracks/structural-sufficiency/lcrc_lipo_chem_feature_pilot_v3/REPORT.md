# LCRC Lipophilicity v3 — chemistry-feature pilot

**Protocol:** `lcrc-lipo-chem-feature-pilot-v3`  
**Status:** **CPU phase complete. GPU not started.**  
**Protocol freeze SHA-256:** `72f2e2142be7eedfe056bcbb948118d0d32ce73f3167fbb356847cf9acbfa14c`  
**GPU authorization:** none. No GPU job was launched; `analysis/gpu_ledger_v3.json` is `NO_GPU_RUNS` (0.0000 h).

## 1. Question

Does label-free global physicochemical information limit the current LCRC models on Lipophilicity? This is an exploratory pilot, not a confirmatory study. The only supervised scope is the frozen v2 Fold-1 train/dev split of Fit2352. Monitor, Reserved and official valid/test labels were not read.

Frozen comparators (v2, Fold 1, seed 0, R3):

| Arm | v2 run | Dev RMSE | Dev MAE |
|---|---|---|---:|---:|
| A | `F1_A_sel_s0` | 0.7403427006133139 | 0.546694119695902 |
| C | `F1_C_sel_s0` | 0.7264913055478032 | 0.5387064120516623 |

Both historical runs were executed on `res` / `res-gpu0` with an A100-SXM4-40GB, torch 2.5.1+cu124. The v3 pilot is pinned to the same host/pool for regime comparability.

## 2. Exactly two feature versions

`Chem12`, in this fixed order:

`TPSA, MolWt, NumHAcceptors, NumHDonors, NumRotatableBonds, NumAromaticRings, NumAliphaticRings, NumSaturatedRings, RingCount, FractionCSP3, HeavyAtomCount, MolMR`

`Chem13` is exactly `Chem12` followed by `MolLogP`.

- Source: `rdkit.Chem.Descriptors.CalcMolDescriptors` (RDKit 2026.09.1), computed only from label-free canonical SMILES.
- Row identity: every one of the 2,352 Fit rows is aligned by canonical-SMILES SHA-256 to the frozen v2 descriptor cache and row manifest. Independent RDKit recomputation on 27 deterministic rows (column extrema plus random rows) matches all 13 cached values exactly.
- `Chem12` matrix SHA-256: `8f3dbf65b86fc483…`; `Chem13` matrix SHA-256: `89579fc77e5d64bb…` (full hashes in `analysis/chem_feature_audit_v3.json`).
- Non-finite policy: training-fold median per column, then training-fold mean/std; zero training variance uses scale 1.0. Fold-1 train (1,568 rows) only; the dev fold never contributes.

## 3. Augmentation and weight migration

Only the global input of `rho` is widened. The local encoder, S1, S2, `f`, the readout, atom51, bond13 and scalar9 keep their frozen semantics. `v2_training.train_job` is reused verbatim; only the model-builder dispatch is extended.

For the same seed, the original model is built first, then all same-shaped tensors are copied. In `rho.lins[0].weight`, the original 42 global columns are copied, the 12/13 new input columns are set to zero, and the original `mean_h` / `mean_h2` / `count` columns are shifted right. Shared tensors are bitwise identical to the original model.

| Arm | Original params | Augmented params | Added | Added fraction |
|---|---:|---:|---:|---:|
| `A_Chem12` | 247,106 | 247,874 | **+768** | 0.311% |
| `A_Chem13` | 247,106 | 247,938 | **+832** | 0.337% |
| `C_Chem12` | 245,305 | 246,073 | **+768** | 0.313% |

## 4. CPU audit results

All gates pass; `analysis/cpu_gates_v3.json` is `READY_FOR_GPU_AUTHORIZATION`.

| Audit | Result |
|---|---|
| Feature names/order | exact Chem12/Chem13 order; `Chem13[:12] == Chem12`; last column is `MolLogP` |
| Fold-1 alignment | 1,568 train / 784 dev; disjoint; covers Fit2352; every row canonical-hash aligned |
| Normalisation | fitted on Fold-1 train only; dev perturbation leaves median/mean/scale unchanged; original 42 means/scales unchanged |
| Zero-extra equivalence | A arms max abs output diff **0.0**; C arm **1.19e-7** (float32 GEMM), all within 1e-6 |
| New-column gradients | finite and non-zero for all three arms (768/832 gradient entries) |
| Denormalisation / raw RMSE | independent checkpoint replay matches stored predictions within 1.7e-6; stored RMSE/MAE match `train.json` and historical constants exactly |
| Historical artifacts | v1 guard ok; v2 guard **237 checks ok**; v2 code/config, 61 run records, checkpoints, predictions and 226-file manifest unchanged |
| Temporary training | 1-epoch CPU smoke in a temp directory; v2 loop reused; +768 params; no v1/v2 write |

The frozen GPU input is `analysis/frozen_extras_v3.npz` (147 KB, SHA-256 `fd73ccd7b4b4…`) plus `analysis/frozen_extras_v3.json`; it contains the exact train/dev matrices, positions, rows, spec hashes and 54-wide normalisation. The GPU host does not need RDKit.

## 5. Fold-1 dev descriptive error groups (CPU, A-R3)

These are descriptive only. No threshold, run or decision was changed after seeing them. Full A and C tables are in `analysis/descriptive_groups_v3.json`.

| Group | n | RMSE | MAE | Share of SSE |
|---|---:|---:|---:|---:|
| **TPSA** ≤ 58.36 | 213 | 0.7203 | 0.5707 | 25.7% |
| TPSA (58.36, 81.22] | 195 | 0.6861 | 0.5445 | 21.4% |
| TPSA (81.22, 97.54] | 170 | 0.7539 | 0.5550 | 22.5% |
| TPSA > 97.54 | 206 | 0.7968 | 0.5171 | **30.4%** |
| **MolLogP** ≤ 2.337 | 114 | **0.9660** | 0.6185 | 24.8% |
| MolLogP (2.337, 3.266] | 234 | 0.6385 | 0.4695 | 22.2% |
| MolLogP (3.266, 4.057] | 233 | 0.7130 | 0.5518 | 27.6% |
| MolLogP > 4.057 | 203 | 0.7344 | 0.5894 | 25.5% |
| **NumHDonors** 0 | 112 | 0.6054 | 0.4613 | 9.6% |
| NumHDonors 1 | 277 | 0.7531 | 0.6017 | 36.6% |
| NumHDonors 2 | 274 | 0.6892 | 0.5191 | 30.3% |
| NumHDonors 3 | 97 | 0.7449 | 0.5352 | 12.5% |
| NumHDonors ≥ 4 | 24 | **1.4076** | 0.6726 | **11.1%** |
| **Formal charge** neutral | 784 | 0.7403 | 0.5467 | 100% |
| Formal charge positive/negative | 0 | — | — | — |

Cut points are Fold-1 training-fold quartiles. `NumHDonors` uses natural bins; `formal_charge` is degenerate on Fold-1 dev (all neutral; train has one +1), so it cannot explain dev errors and is reported as such. The descriptive pattern points to low-MolLogP / high-TPSA / hydrogen-bond-rich molecules as error-concentrated, but this is not evidence that the chemistry features will help.

## 6. Budget and pending GPU runs

Hard cap for this pilot: **0.30 A100 GPU-hours**. Anchor-based estimate for all three registered runs (including the conditional third): **0.1217 h**; 0.1783 h margin. Unconditional two runs: 0.0871 h.

| Pending run | Arm | Fold | Seed | Recipe | Estimate |
|---|---|---|---:|---:|---|---:|
| `F1_A_Chem12_R3_s0` | `A_Chem12` | 1 | 0 | R3 | 0.0396 h |
| `F1_A_Chem13_R3_s0` | `A_Chem13` | 1 | 0 | R3 | 0.0396 h |
| `F1_C_Chem12_R3_s0` (conditional) | `C_Chem12` | 1 | 0 | R3 | 0.0315 h |

No other seed, fold, recipe, SSL arm, descriptor combination, full-Fit retrain or Monitor evaluation is registered.

## 7. Result table

Only the historical rows exist at the CPU freeze. The v3 rows are written to `analysis/table_v3.csv` after the two authorized GPU runs.

| 模型 | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE | ΔRMSE vs 同架构原模型 |
|---|---:|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.26963 | 0.74034 | 0.54669 | 0.00000 (reference) |
| A_Chem12 | 1 | 0 | — | — | — | — |
| A_Chem13 | 1 | 0 | — | — | — | — |
| C-R3 (v2 historical) | 1 | 0 | 0.29989 | 0.72649 | 0.53871 | 0.00000 (reference) |
| C_Chem12 (conditional) | 1 | 0 | — | — | — | — |

## 8. Integrity

- v1/v2 were not modified; the v2 manifest check covers 237 files/records including the two historical checkpoints and prediction vectors.
- No Monitor, Reserved or official valid/test label was read at any point.
- All write-behaviour tests use temporary directories; no historical prediction is overwritten.
- Code/config/tests were committed before the freeze (`28cee21`). The freeze pins all v3 code and test hashes.

## 9. Required next step

The researcher must issue a GPU authorization pinned to the exact freeze SHA above before any GPU work. The runner refuses otherwise. After authorization:

```bash
rr run res v3-A12 --pool res-gpu0 --time 0:20:00 \
  --result tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/analysis/runs/F1_A_Chem12_R3_s0 \
  -- uv run --frozen python tracks/structural-sufficiency/lcrc_lipo_chem_feature_pilot_v3/code/run_v3.py \
     train --key F1_A_Chem12_R3_s0 --device cuda
# same for F1_A_Chem13_R3_s0
```

The conditional `F1_C_Chem12_R3_s0` is refused by the runner unless Chem12 improves historical A-R3 by ≥0.03 RMSE, MAE also improves, and the gain is not tail-dependent under the pre-registered top-2% rule.
