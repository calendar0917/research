# PROTOCOL — LCRC performance route (8 h, public122, seed 0)

Round / protocol ID: `lcrc-performance-route-8h-public122-s0-v1`
Track: `tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1`
Authorisation: user brief `LCRC_Performance_Route_8h_Agent_Prompt.txt` (2026-11-13).
Wall budget 7–8 h (hard 8 h), GPU soft 14 A100·h / hard 16 A100·h, max 64 new
trainings (excluding ≤2-epoch smokes), seed 0 only.

Everything in this file is **frozen before the first formal run**.  The machine
copy is `configs/experiment.yaml`; the code constants in `code/lcpr_common.py`
are authoritative.  A run that contradicts this file is a bug, not a result.

## 1. Questions

| ID | question |
|---|---|
| Q1 | How large is the current C1 gap to (a) a plain recipe screen, (b) a cheap capacity/budget increase, (c) an XGBoost/ECFP reference, (d) a real D-MPNN (chemprop v2.2.1)? |
| Q2 | Do the two dissection findings — the dropout/second-moment training shift of the second-order pooling, and the low-support generalisation of the S2 joint (J) pair term — convert into prediction gains under matched controls? |
| Q3 | Does a computational-task structural target (WC_LOGP) help as an *auxiliary supervision*, or is it just generic regularisation / extra optimisation load? |
| Q4 | Is any gain row-specific, Lipo-specific, or transferable to a second task? Which routes stop and which remain untested? |

This round does **not** redraw "the model relies on S2".  It must connect the
mechanism to real per-row prediction error and to controlled training gains.

## 2. Data

| dataset | task | split | sizes (train/valid/test) | role |
|---|---|---|---|---|
| Lipophilicity | regression | ChemBench `random122` | 3360/420/420 | primary diagnosis field |
| ESOL | regression | ChemBench `random122` | 902/113/113 | transfer 1 |
| BACE | binary | ChemBench `scaffold122` | 1210/151/152 | transfer 2 |
| FreeSolv | regression | ChemBench `random122` | 513/64/65 | extension 3 |
| ogbg-molhiv | binary | official OGB scaffold | 32901/4113/4113 | HIV stress test |

* Structure caches: the frozen, **label-free** `lipo_v1` caches of
  `lcrc_molnet_benchmark_pilot_v2/cache/<ds>_<split>.pt`, consumed read-only and
  hash-recorded (`analysis/prepare.json`, `analysis/cache_audit.json`).
* Split order is preserved and hashed per split; sets are never compared.
* Parameter updates use Train labels only.  Target normalisation, structure
  normalisation, support counts, auxiliary targets and the shuffle are fitted
  on Train only.
* **Every epoch** evaluates the full Train, Valid and Test splits in
  `model.eval()` / `no_grad`, with the Python/NumPy/Torch CPU+CUDA RNG state and
  the module mode saved and restored.  Test is therefore fully exposed; all Test
  numbers are `DEVELOPMENT_TEST / TEST_USED_FOR_ROUTE_SELECTION`.
* Checkpoints: best **Validation** only (strict improvement, ties keep the
  earlier epoch); last epoch also saved.  Test never selects an epoch but is
  allowed to select a *route* (recorded in `analysis/selection_trace.json`).

## 3. Model

`C1` = `NO_BOND_MEAN`, 228 674 trainable parameters: AF51 / BF13 / scalar9 /
global42, S1 joint, typed S2 (type/end/joint/closure/content), no `b_mean`.

* New C1-shaped arms start from the **same** `A0=FULL` seed-0 state through the
  frozen `remap_state("C1", full)`; extension modules are initialised from a
  private `fork_rng` stream, shared parameters are copied bit-exactly, and the
  model RNG is reset before the first update.  Verified per run
  (`shared_init_bitwise_equal`, `init_state_hash`).
* `A0` (247 106) is only used for the historical bridge.
* S1/S2 graph definitions are unchanged in every arm.

## 4. Recipes

`R0` = historical: AdamW(.9,.999), eps 1e-8, peak 3e-4 → min 1e-5 per-step
cosine, weight_decay 1e-3, clip 5, batch 128, 360 epochs, FP32, no
EMA/SWA/soup/early stop, dropout 0.10 on the final `h`.
`R1` = peak 1e-3 only.  `R2` = weight_decay 1e-2 only.  `R3` = both.
`R*` = the best-Validation recipe of the four C1 arms on Lipo (ties:
R0, R1, R2, R3 order), frozen before any Stage-B arm starts.
`LONG` = C1 at `R*` with 720 epochs and a 720-epoch cosine horizon (from
scratch; **not** the same first 360 epochs as `R*`).
Batch order: one frozen permutation table per dataset,
`torch.Generator().manual_seed(0 + 917_000)` drawn epoch by epoch (identical for
every arm of a dataset, never redrawn).

## 5. Candidate registry (fixed)

| arm | family | definition |
|---|---|---|
| `C1_R0..R3` | VANILLA | plain C1 under the four recipes |
| `WIDE_LOCAL` | WIDE | LocalMLP hidden 192→384, output 64; first layer keeps the 192 original rows + private Kaiming rows (new biases 0), second layer keeps the 192 original columns and zeroes the new ones, bias copied.  Step 0 equals C1 exactly. |
| `LONG` | LONG | C1, 720 epochs |
| `A0_R0`, `A0_CHEM8_CONTRIB_R0` | BRIDGE | historical A0 and the historical chem8+multicontrib combo, both at R0 |
| `K_COMBO` | K | C1 + chem8 (historical 8 atom channels) + multicontrib (`Q:64→64→8`, graph sum, `delta:[8+log1p n]→32→1`, last layer zero); `y = L + rho + delta` |
| `C1_CHEM8`, `C1_CONTRIB` | K | single-arm controls for the 2×2 (bought only if K shows signal) |
| `M_UNBIASED` | M | `d = m·h/(1-p)`; `f` uses `d`; `rho` uses `mean d`, `(1-p)·mean d²` **during training only**, eval unchanged |
| `M_CLEANPOOL` | M | `f` uses `d`; `rho` uses `mean h`, `mean h²` from the clean `h` (`mean(h*h)`, not `mean(h)²`) |
| `M_NODROP` | M | `p = 0` (ordinary regularisation-strength control) |
| `S_SELECT_K1` / `S_UNIFORM_K1` | S | J contraction with `κ = max(1, median_occupied c(key))` |
| `S_SELECT_K4` / `S_UNIFORM_K4` | S | same with `κ' = 4κ` |
| `A_LOGP`, `A_MW`, `A_SHUFFLE` | A | auxiliary heads `q_aux:64→64→1` (sum) + `rho_aux:171→64→1`, `L = L_primary + 0.1·MSE(aux, z_aux)`; targets WC_LOGP / MolWt / one Train-row permutation of WC_LOGP (private seed 90217) |
| `MP_CONTEXT`, `POOL_CONTEXT` | CTX | diagnostic learnable-context positive control (7.6), only if the trigger fires |
| `DMPNN_LR3E4`, `DMPNN_LR1E3` | DMPNN | chemprop v2.2.1 BondMessagePassing (hidden 300, depth 3, bias F, undirected F, ReLU, message dropout 0) + mean aggregation + FFN 300→300→1; batch 64, Adam, 100 epochs, 2-epoch linear warmup, exponential decay to 1e-4, weight_decay 0, clip 5, peaks 3e-4 and 1e-3 |

### 5.1 S-family rule (label-blind, Train structures only)

`key = (path_cat, path_end63)`, `c(key)` = number of **distinct Train molecules**
containing the key at least once.  `κ = max(1, median_{occupied keys} c)`.
Per root `r`: `a_r = mean_{paths rooted at r} c/(c+κ)`; a root with no path has
`a_r = 1`.
`S_SELECT`: `J_r = J_marg_r + a_r·(J_full_r − J_marg_r)`.
`S_UNIFORM`: the same formula with `a_r` replaced by the Train-global mean
`a_bar` (computed for that κ).  `J_marg` is the frozen B5 `marginal_bt2`
operator; `T/E/K/C` blocks keep their FULL aggregation.  The contraction
residual is **not** detached.  α = 1 must reproduce FULL and α = 0 must
reproduce B5 (fixture in `tests/test_lcpr_core.py`).

### 5.2 A-family rule

Aux heads are private-initialised, add no dropout, and reuse the **same** `h`
and the same dropout mask as the main head.  The main prediction never reads the
real WC_LOGP / MolWt value; the heads are dropped at inference.  Real WC_LOGP
metrics on Valid/Test are reported as a side observation, not as property
accuracy.  λ = 0.1 fixed, no search.

## 6. Selection rules (frozen)

* gain (regression) = `(RMSE_base − RMSE_arm)/RMSE_base`; (classification) =
  `AUC_arm − AUC_base`.  Every base is the same dataset/split/recipe/epoch
  budget.  Positive = improvement.
* Lipo clear signal: Test relative RMSE gain ≥ 1 %, Valid gain ≥ −0.5 %, and
  neither Valid nor Test MAE worse than its own base by more than 1 %.
  This is a round-local purchase rule, not significance or generality.
* Within a family: rank by `min(gain_Valid, gain_Test)`, then mean MAE gain,
  then fewer parameters, then fixed ID.  Keep the first of each family.
* Promote at most two different families.  If none has a clear signal, promote
  the single highest balanced-score arm as a *diagnostic transfer*, clearly
  labelled.  Test is used deliberately here and that is recorded.
* WIDE/LONG are kept as performance routes if they win; their different update
  counts and cost are listed separately in a Pareto table.  "Better than the
  small C1" is never written as "better than all vanilla C1".
* Best static arm = argmax over eligible arms of `min(Valid gain, Test gain)` on
  the C1@R* base, tie-broken by mean MAE gain, then GPU cost, parameters, ID.
* Transfer (7.2): per task, `R_task` = best-Validation recipe among
  `{C1@R0, C1@R*}`, frozen before any candidate runs.  Then the promoted arms
  run once each.  "Transferable signal" needs the Lipo clear signal **and**
  ESOL relative RMSE ≥ 1 % or BACE AUC ≥ +0.005, with compatible
  Valid/Test direction, and no obvious reversal on the other transfer task
  (regression worse by > 2 %, AUC worse by > 0.01).
* HIV (7.3): C1 base and the best static arm, same recipe (`R0` if the base is
  C1@R0 and the candidate recipe is frozen per its own definition — the pair
  must share one recipe), 160 epochs, cosine 160, no early stopping, full
  three-split eval every epoch.  Both jobs must be launchable together.
* 7.4 combination: at most one combination, only if two **new** families (M/S/A)
  each have a Lipo clear signal and at least one has a transfer signal.
* 7.5 capacity attribution: at most two extra paired runs on WIDE.
* 7.6 failure-diagnosis branch (mutually exclusive with 7.4, ≤ 4 runs): only if
  every new static family lacks a Lipo clear signal **and** the D-MPNN
  reference beats the best vanilla C1 by ≥ 3 % relative RMSE on both Valid and
  Test with no worse MAE.
* 7.7 remaining-budget order: missing matched controls → unadvanced families'
  best Lipo arm on ESOL/BACE → FreeSolv base+best pair → best static arm on
  ESOL WIDE pair.  No new seeds, no third auxiliary property, no extra κ/λ, no
  fourth hop, no attention/SSL/pretraining, no new random split.

## 7. Analysis contract

* Error ledger on the aligned best-Valid rows: e, AE, SE, ΔAE/ΔSE vs the frozen
  base; classification adds BCE, p and an all-negative per-row rank loss, split
  by class.
* Fixed structural covariates (never re-thresholded): atom count, formal charge
  / `fr_quatN`, `fr_COO`, `fr_phos_acid` counts, `maxTan_to_Train_reference`
  (Morgan, Train ≤ 4096 reference subsample by fixed row hash; Train queries
  exclude themselves), rooted-WL2 Train support, S2 key support, out-of-Train-
  range standardised `g`/scalars, existing family hits.
* Tail view: always all rows as the primary metric; additionally the fixed base’s
  largest-SE row, and top-1 % / top-5 %-SE exclusion (same base row set for
  every model).  A tail-only gain is labelled `TAIL_DOMINATED`.
* Paired bootstrap: 2000 replicates, seed 20261113.  Lipo reuses the existing
  `eval_component` where available; other regression tasks use scaffold clusters
  (empty scaffold → canonical SMILES); classification uses stratified
  positive/negative row bootstrap with a scaffold sensitivity check.
  Direction: positive = improvement (`RMSE_control − RMSE_candidate`,
  `AUC_candidate − AUC_control`), verified by a fixture.
  All intervals are conditional on one seed and on this round's
  recipe/candidate/Test screening: they are not independent confirmation.
* Mechanism panels: M — 256 fixed masks per row on the frozen clean `h`
  comparing `mean h`, `mean h²`, `rho` and the final prediction (MC mean and
  variance vs the eval forward); rows chosen by a fixed hash (Train 128,
  Valid 64, Test 64) plus named cases (row 2087 if present).
  S — α / key support / ΔJ / Δh / Δq / Δrho versus real error change.
  A — gradient norms and cosines at steps 1/100/1000/last (isolated
  `autograd.grad`, never added to `.grad`) plus aux R² on real targets.
  K — retrained 2×2 where bought.
* Cost accounting: GPU·h = wall seconds × GPUs for every job that touches a GPU
  (including eval, smoke, failed and cancelled jobs); pure train-kernel time is
  reported separately.  CPU work is listed separately.
* Confidence/practicality: same rows/split as each reference, with features,
  parameters, optimiser updates, preprocessing CPU time, cache size,
  train/eval cost and a batch-128, 10-batch-warmup, 5-repeat Test inference
  timing (median + range, with `cuda.synchronize`), measured only when a pool is
  idle.  C1 and D-MPNN differ in inputs and recipe: their comparison is a
  practical coordinate only.

## 8. Deliverables

`PROTOCOL.md`, `configs/experiment.yaml`, `code/`, `tests/`, `REPORT.md` (中文,
four questions first), `analysis/run_table.csv`, `epoch_metrics/`,
prediction list, `analysis/selection_trace.json`, budget/runtime,
`analysis/paired_comparisons.json`, `analysis/error_ledger.csv`,
`analysis/mechanism_summary.json`, four core figures.
Checkpoints and large caches stay git-ignored, kept locally **and** on `res`;
light results plus hashes and recovery commands are committed.
No paper draft, no submission, no leaderboard, no PR, no messaging.
