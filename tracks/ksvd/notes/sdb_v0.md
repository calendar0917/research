# SDB-v0 — Sparse Structural Dictionary Binding (analysis)

Protocol version `sdb_v0`. Study `zinc-context-gap`. Preregistration:
`notes/sdb_v0_preregistration.md` (+ Amendment A1, A2). Status: **stopped after
Stage 4 seed 0 by explicit user decision** ("no more seeds, do not re-run what
has already run, stop once the conclusion is clear").

Official ZINC **test was never loaded** in any stage. No historical reusable
baseline (M0/MB/MM soups, strict-static S0, TCCD, DTX) was retrained. All runs
were local CPU on the durable FSAR-R2-AR0 / strict-static caches.

---

## 1. The question

The multi-seed-validated FSAR-R2-AR0 object is the ZINC assignment

```
C_phi = sum_v (phi_v - phi_bar)(q_v - q_bar)^T ,   phi_v in R^65 (pure topology),
                                                    q_v  in R^28 (one-hot atom chemistry),
```

read by a bias-free linear `W` and shown (2-3 seeds) to beat matched marginal
and parameter-matched controls. SDB-v0 asks:

> Can the **structural-role axis** `phi_v` be replaced by a **reusable, sparse,
> task-coupled dictionary coordinate** (K=32, s=8, one shared dictionary `D`)
> without losing the object's main task information?

The frozen, non-negotiable config: `K=32`, `s=8`, `phi_v in R^65`,
`C_D = sum_v (alpha_v - alpha_bar)(q_v - q_bar)^T in R^{32x28}` **never
compressed**, dictionary input chemistry-free. Forbidden rescues (no K/s sweeps,
no LISTA, no attention, no new reader, no dense bypass, etc.) were honoured.

## 2. Pipeline and stages

| stage | type | cost | verdict |
|---|---|---|---|
| 0 sanity | data-free + synthetic controls | seconds | PASS |
| 1 label-free dictionary identity | no `y`; K-SVD fit | 298 s | PASS |
| 2 FSAR mechanism preservation | frozen `M0` + linear residual, 3 seeds | minutes | PASS |
| 3 task-coupled tied-IHT dictionary | frozen `M0`, seed 0 | 4 min | nominal PASS, marginal (see 5) |
| 4 strict-static S0 strong backbone | frozen S0, seed 0 | 41 s | **FAIL** |

### Stage 0 — sanity / controls

`sanity.json`: 10/10 data-free checks pass (`all_pass=true`) — batching
invariance, exact binding identity, `C_D` assignment sensitivity, chemistry
permutation, dictionary-input chemistry purity, exact sparsity, joint
permutation invariance, no official-test access, no raw/mixed bypass, node
relabel invariance. Synthetic ridge controls (`synthetic_controls.json`):
positive case (assignment-only signal present) binding MAE `0.0927` vs marginal
`0.5019` (pass); negative case (marginal only) `binding_advantage = -0.958`, so
no false positive.

### Stage 1 — label-free dictionary identity (no `y`)

Full K-SVD `D in R^{65x32}`, `s=8`, fit on **231,664** train atoms, 10 epochs,
298 s; `ksvd_final_fit_mse = 4.97e-4`.

| reference | E_phi dev | E_bind dev |
|---|---:|---:|
| ksvd (sparse, s=8) | 1.637e-05 | **3.454e-04** |
| ksvd dense (same D, s=K) | — | 1.602e-05 |
| random normalized | 0.6938 | 1.713 |
| pca (rank-32 affine) | 1.910e-08 | 5.304e-07 |

Health: **32/32** atoms used, exact sparsity 8, top-1 / top-8 coefficient mass
`0.803 / 1.0`, support entropy `0.210`. PCA is a near-lossless dense rank-32
reference (Amendment A1), so the meaningful contrast is sparse vs dense on the
**same** learned `D`: `E_bind(sparse)/E_bind(dense) = 21.6`. **PASS** on the
hard gates (`E_bind <= 0.10`, `< random`, `used >= 24`).

### Stage 2 — mechanism preservation vs the frozen `phi65` oracle (3 seeds)

Frozen `M0` Top-5 soup + a single linear assignment residual. `dict32` =
frozen `D` (65x32) + readout (32x28); `dense32` = learned `W` (65x32) +
readout — parameter matched.

| seed | M0 | phi65 | **dict32** | dict_dense32 | pca32 | dense32 | dense32_frozen | rand32 | recovery | shuffle_deg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.55379 | 0.494538 | **0.486293** | 0.491329 | 0.488480 | 0.495732 | 0.501665 | 0.497199 | 1.139 | 0.2368 |
| 1 | 0.53557 | 0.481283 | **0.471510** | 0.477209 | 0.475874 | 0.481869 | 0.483369 | 0.481945 | 1.180 | 0.2395 |
| 2 | 0.546893 | 0.490486 | **0.481535** | 0.487915 | 0.484218 | 0.489971 | 0.492226 | 0.490163 | 1.159 | 0.2598 |

Mean recovery vs the frozen `phi65` oracle = **1.159** (`>1`: the sparse dict32
branch is *better* than the full 65-D pure-topology `phi` axis); mean
assignment-shuffle degradation = **0.245**; `dict32` beats every control
(pca32, dense-coded `D`, learned dense, frozen dense, random). `phi65`
reproduces the durable `h5_frozen_m0.json` bit-exactly (protocol drift
`8.2e-09`). **PASS**.

Interpretation: the sparse dictionary coordinate is a *sufficient and
non-lossy* replacement for the structural-role axis — the validated assignment
survives the compression to `K=32, s=8`, and the compressed statistic is
causally used (large shuffle sensitivity).

### Stage 3 — task-coupled tied-IHT dictionary (seed 0)

`D` initialised at the frozen `D_KSVD`, learned by tied-IHT encoding with
`loss = L1(pred, y) + lambda * ||phi_v - D alpha_v||^2 / ||phi_v||^2`
(`lambda` calibrated once, detached; `lambda_rec = 49.4`).

| arm | held-out valid MAE |
|---|---:|
| frozen-D + linear readout | 0.498979 |
| task-coupled D | 0.489882 |
| delta | **-0.009097** (threshold 0.003) |

Verdict nominally `TASK_COUPLING_HELPS`, but **marginal and single-seed**:
`atom_movement = 5.07` (the dictionary moves far from the K-SVD init) and the
honest effect is ~3x smaller than an earlier, buggy estimate.

> **Correction discovered during Stage 4.** The first `train_task_coupled_dictionary`
> implementation evaluated its "valid MAE" over the **train** split (no held-out
> split), so its early-stopping/selection was on training error and its reported
> gain was inflated (`-0.030`). It was fixed to take an explicit held-out valid
> split *before* any durable record was written. The Stage-3 result committed
> here is the corrected one. Result files carry `git_commit = 00bf711` (HEAD at
> run time); the Stage-3 runs used `00bf711` plus the uncommitted validation-split
> fix (`sdb_v0.py` sha256 `0bc1faca…`). Stages 0-2 and the Stage-4 base/branch
> arms are unaffected by the fix.

Given the marginal corrected effect, Stage 3 is **not** promoted as a strong
positive.

### Stage 4 — strict-static S0 strong backbone (seed 0) — FAIL

Branch: `y_hat = y_hat_S0 + <W_D, C~_D>` with `W_D` a single trained 32x28
readout (896 params) over the train-RMS-scaled centered binding tensor from the
**frozen** Stage-1 `D_KSVD`. Matched dense control additionally trains the
`65x32` projection (2,976 params total).

**Amendment A2 (pre-run):** the historical strict-static seed-0 Top-5 *soup
member states* were never persisted (only the best `selection_state.pt` and the
soup's *valid predictions*), so the soup cannot be reproduced on the **train**
split. The branch is therefore trained and evaluated on the same frozen
**selection-state** base, and the material-gain gate is measured against that
base. The historical soup is reported as context.

| arm | valid MAE |
|---|---:|
| S0 base (selection state) | 0.145674 |
| S0 + DenseBinding | 0.144183 |
| **S0 + DictBinding** | **0.143294** |
| historical S0 Top-5 soup (context) | 0.140794 |

| gate | value | threshold | pass |
|---|---:|---:|---|
| gain vs base | 0.002381 | >= 0.003 | **FAIL** |
| Dict <= Dense + slack | -0.000890 | <= 0.002 | pass |
| assignment-shuffle degradation | 0.016131 | >= 0.02 | **FAIL** |

**Verdict FAIL** (2 of 3). The dictionary branch is not inert — it slightly
*beats* the parameter-matched dense control — but its absolute gain on the
strong backbone is below the materiality threshold and its within-molecule
assignment-shuffle sensitivity is below the mechanism threshold. It also lands
*worse* than the already-existing S0 Top-5 soup (`0.143294 > 0.140794`), so it
does not improve on the durable strong baseline.

## 3. Conclusion

**Question answer (two-part).**

1. **Replacement / reuse — supported.** The structural-role axis `phi_v` can be
   replaced by a reusable, sparse (`K=32, s=8`), chemistry-free dictionary
   coordinate with **no loss** of the object's main task information: the
   compressed statistic is label-free-faithful (Stage 1), causally used
   (Stage 2 shuffle), and on the frozen weak base it is *better* than the full
   65-D axis under matched controls (recovery 1.159, 3 seeds). This is the
   durable, reusable finding.

2. **Incremental task value on a strong backbone — not supported.** On the
   strict-static S0 backbone the same branch adds only a sub-threshold
   (`0.0024 < 0.003`) gain with sub-threshold shuffle sensitivity
   (`0.0161 < 0.02`) and does not beat the existing S0 soup (Stage 4 FAIL).

**Mechanistic reading.** On the weak frozen `M0` base the assignment statistic
is the dominant channel, so compressing it is free and task-coupling has room
(Stage 2/3). On the strong `S0` base the backbone already internalises the
topology↔chemistry assignment, leaving too little residual for a per-molecule
linear `C_D` readout; the residual branch is real but small. This matches the
track's recurring pattern (e.g. SDPK-v0: dictionary used, no absolute gain on a
strong backbone).

**What was NOT done (per stop decision / forbidden-rescue list):** no K or `s`
sweep, no LISTA/attention, no new reader, no dense bypass, no seed 1 for
Stage 3/4, no retraining of any historical baseline, no official test. The
implemented-but-unrun supplementary Stage-4 task-coupled arm (see below) was
deliberately left unrun.

## 4. Provenance

| item | value |
|---|---|
| stages 0-2, 4 code | HEAD `00bf711` |
| stage 3 code | `00bf711` + held-out-validation fix (`sdb_v0.py` `0bc1faca…`, `zinc_sdb_v0.py` `96613cef…`) |
| preregistration sha256 | `920f74b8…` (+ Amendments A1, A2) |
| device | local CPU (`torch` deterministic paths; no GPU run) |
| official test | never loaded |
| reused artifacts | FSAR cache `results/fsar_r2_ar0/cache/{train,valid}.pkl.gz` + `scaler.pt`; M0 soup `soup_states/r2ar0_m0_seed{0,1,2}_top5_soup.pt`; `h5_frozen_m0.json`; `zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt` |
| dictionary meta | `dictionary_meta.json` (`dict_seed 20260924`, 10 K-SVD epochs) |

Result files: `results/sdb_v0/{sanity,synthetic_controls,dictionary_meta,
stage1_label_free,stage2_mechanism,stage3_task_coupled,stage4_static_seed0}.json`
plus `STAGE1_REPORT.md`, `STAGE2_REPORT.md`.

## 5. Not run / next steps (do not execute without a new preregistration)

- Supplementary Stage-4 task-coupled arm (task-coupled `D` on the frozen S0
  base) is **implemented** in `zinc_sdb_v0.stage4` but was not run. Given the
  corrected Stage-3 effect (~`-0.009` on the weak base) and the sub-threshold
  frozen-`D` Stage-4 gain (`0.0024`), a material rescue on the strong backbone
  is not expected; it is a documented follow-up, not a licence for a sweep.
- If the strong-backbone question is revisited, it must (i) persist the S0 soup
  member states so the branch can be trained against the exact strong baseline,
  and (ii) buy matched controls separating "dictionary richness" from
  "assignment value".

## 6. Invariants

- No official ZINC test access in any stage (`official_test_loaded = false`
  everywhere).
- No historical baseline retrained.
- Dictionary input never contains chemistry; `C_D` never compressed.
- All gates, thresholds and the two amendments were frozen before their runs.
