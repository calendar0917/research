# LCRC support-indexed joint transfer v1 — protocol

Round id: `lcrc-support-joint-transfer-v1`.
Protocol frozen before the first formal training step. The machine-readable freeze is
[`analysis/protocol_freeze.json`](analysis/protocol_freeze.json)
(`freeze_hash = fdd2c15f7f47264bbc7173a14a4bf78ff50a3669ced65e9c7ede59420265f9e9`,
config SHA256 `0de19575a0330e43f86ba7678098aa47b784cbfa10957007cd2b2f301f1e34b6`).
Any change after this point is recorded as a deviation in [`REPORT.md`](REPORT.md),
not silently folded into the frozen protocol.

## 0. Question and scope

The round asks whether the **support-indexed S2 joint channel** from
`lcrc_support_joint_function_v0` transfers from the prior random-public122 Lipo
study to the **official OGB scaffold splits**, and whether the same change helps
the PCBA W2 and FIXED1 backbones. It is an implementation / transfer /
verification round, not an architecture search.

Fixed constraints:

* no new structural inputs, gates, message-passing layers, readouts, auxiliary
  losses, parameter sharing or architecture redesign;
* no distillation, ensembles, MoE, new seeds, or hyperparameter search;
* at most four new training units: `C1_JOINT`, `C1_ADD`, `W2_JOINT`,
  `FIXED1_JOINT`;
* existing backbones, initialisation streams, data order and recipes are reused;
* only the S2 joint channel is replaced. Every other computation must remain
  bitwise identical at initialisation.

The prior round's `r122_C1_s0` is **not** an official-split baseline and is not
used as one. The official Lipo baseline is arm A of
`lcrc_readout_context_pilot_small_v1` (C1 readout, official scaffold split).

## 1. Reused assets (read-only)

| role | asset | identity |
|---|---|---|
| Lipo C1 baseline | `lcrc_readout_context_pilot_small_v1/results/ogbg-mollipo/A` | 228,674 params; best epoch 275; Valid RMSE 0.7086243164; Test RMSE 0.7253920405; batch-order SHA256 `1a3a440f…b1618` |
| Lipo data/normalisation | `lcrc_readout_context_pilot_small_v1` `cr_data.build_bundle('ogbg-mollipo')` | official OGB scaffold split; target scaler mean 2.1621607145, std 1.2115398401, n 3360 |
| S2 support/table code | `lcrc_support_joint_function_v0/code/{sj_support,sj_model,sj_common}.py` | Omega, bipartite incidence, float64 SVD, `P = U Uᵀ`; JOINT (B) and ADDITIVE (C) formulas |
| PCBA W2 | `lcrc_pcba_backbone_scaling_v1/runs/W2` | 994,324 params; best epoch 24; Valid AP 0.2652733161; Test AP (brief) 0.2536164787 |
| PCBA FIXED1 | `direct_structural_computation_recurrence_v0/runs/FIXED1_seed0` | 1,128,468 params; best epoch 25; Valid AP 0.2758455214; Test AP 0.2570014322 |
| PCBA data | `lcrc_pcba_scale_pilot_v1/cache/pcba_v1` + frozen split JSON | official OGB scaffold split 350,343 / 43,793 / 43,793 |
| PCBA recipe / plan | `direct_structural_computation_recurrence_v0` (`dc_common`, `ds_train`, `ds_model`) | 40-epoch AdamW schedule, masked BCE, seed 0, plan SHA256 `06f43729…b6f6`, prefix-25 SHA256 `8c857251…9b40` |

The Lipo feature cache, row ids, target normalisation, initialisation and
recipe are inherited without regeneration. The PCBA feature cache is the same
163 GB memory-mapped cache; no re-featurisation is performed.

**Historical reference values are descriptive, not retrained:** Lipo GINE
Valid/Test 0.6091/0.6456; PCBA DYNAMIC 0.275067/0.262226. They are not used as
controls in the paired analyses.

## 2. S2 support, frozen on Train only

For each dataset, production S2 records define `p = path_cat`, `e = path_end63`.
`Omega` is the set of distinct `(p,e)` pairs occurring in the **official Train
split only**. Build:

* bipartite incidence `X ∈ {0,1}^{|Omega| × (n_p+n_e)}`, two unit entries per row;
* float64 SVD of `X`, rank `rank(X) = n_p + n_e - n_components`;
* `U` = left singular vectors with `s > 1e-10 s_max`, `P = U Uᵀ`.

Frozen supports:

| dataset | `n_omega` | `n_p` | `n_e` | `rank(X)` | joint dim | components | rectangles | Train/Valid/Test occurrence coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Lipo | 108 | 28 | 24 | 48 | 60 | 4 | 441 | 1.00000 / 0.99983 / 0.99950 |
| PCBA | 214 | 42 | 29 | 67 | 147 | 4 | 2,585 | 0.99994 / 0.99998 / 0.99999 |

Every `Omega` row has at least one Train occurrence. Valid/Test pairs outside
`Omega` use the v0 fallback: same-component marginal least squares when the two
types occur in one component, otherwise the type-equal mean over `Omega` rows of
the current table. The fallback creates **no new parameters**.

The support is rebuilt from Train alone and compared elementwise with the full
build in `jt_prepare.py`; equality of `omega_code` and `P` is a hard gate.
The split audit proves official CSV row ids equal the payload/cache order
elementwise (Lipo) and the frozen split JSON and raw index hashes (PCBA).

## 3. Four training units and the exact change

Let `h2` be the S2 joint-channel pre-activation inside the production local
trunk. In the original backbone:

```
J0(p,e) = joint_proj( path_u(p) * path_v(e) )        # includes bias
h2      = W_j · J0(p,e) + b_j
```

`J0` is materialised from the **bitwise-shared original C1/W2/FIXED1 state**.

**C1_JOINT / W2_JOINT / FIXED1_JOINT** (v0 B):

```
J(p,e) = J_table[ index_Omega(p,e) ]                  # index lookup
h2     = W_j · J(p,e) + b_j
```

`J_table` is initialised to the materialised `J0`. It is the only new tensor:
108×32 for Lipo (`stat_rank=32`), 214×64 for PCBA (`stat_rank=64`).

**C1_ADD** (v0 C, Lipo only):

```
J(p,e) = U · Theta,      Theta ∈ R^{rank(X) × 32}
h2     = W_j · U · Theta + b_j
```

`Theta` is initialised as `Uᵀ J0`, so `(I−P)J = 0` exactly in float64 at
initialisation. The additive arm is in the marginal subspace of the same
support; it is not a second free table.

All other modules, residuals, LayerNorms, bond projections, readouts, dropout,
initialisation streams and data order are unchanged. Each JOINT arm has a
bitwise-identical common parameter state with its original backbone. The
initial forward on Train-seen types agrees to numerical tolerance because
`J_table` starts as `J0`.

Parameter counts (constructed, not nominal):

| arm | total | joint-channel parameters | Train-supported joint parameters |
|---|---:|---:|---:|
| C1 (reused) | 228,674 | 0 | 0 |
| C1_JOINT | 215,714 | 3,456 (`J_table`) | 3,456 |
| C1_ADD | 213,794 | 1,536 (`Theta`) | 1,536 |
| W2_JOINT | 973,140 | 13,696 (`J_table`) | 13,696 |
| FIXED1_JOINT | 1,107,284 | 13,696 (`J_table`) | 13,696 |

The lower totals versus the originals are the removed `joint_proj` parameters,
not parameter sharing or compression of other modules.

## 4. Recipes

### Lipo (`C1_JOINT`, `C1_ADD`)

Inherited R3 recipe from the arm-A round:

* seed 0; official Train only; batch 128;
* 360 epochs, one `np.random.default_rng(0).permutation` of Train-local rows per epoch;
* AdamW, peak LR 3e-4, per-step cosine to 1e-5 over 360 × ceil(3360/128) steps, no warmup;
* weight decay 1e-3, gradient clip 5.0, dropout 0.10, no AMP/EMA/SWA/soup;
* MSE on the standardised target; full Train + Valid + Test evaluation every epoch;
* checkpoint by minimum Valid raw RMSE, strict improvement, earliest wins ties;
* the checkpoint's Test value is a development-test read, not used for selection.

The historical arm-A batch-order SHA256 and best epoch are replay gates.

### PCBA (`W2_JOINT`, `FIXED1_JOINT`)

Strict 25-epoch prefix of the historical 40-epoch DSCR schedule, never
recompressed:

* seed 0, official Train only, batch 128, original plan/order;
* AdamW, 1-epoch linear warmup 3e-5 → 3e-4, per-step cosine defined on the
  **40-epoch** horizon to 1e-5; at epoch 25 the LR is the historical prefix value;
* weight decay 1e-3, clip 5.0, dropout 0.10, masked BCE, no AMP/EMA/SWA/soup;
* full Train + Valid + Test evaluation every epoch;
* checkpoint by highest Valid macro AP within the 25-epoch prefix; strict
  improvement, earliest wins ties;
* Test is an exposed development comparison set; it is never used for
  checkpoint selection. Test role is recorded in every run.

`W2_ORIGINAL` and `FIXED1_ORIGINAL` are reused checkpoints. For a like-for-like
delta, they are also re-evaluated once through the same evaluator/device as the
new arms; the historical brief values are reported separately.

## 5. Verification gates (must pass before/after formal runs)

Pre-training, CPU/GPU as appropriate, written to
`analysis/checks_lipo.json` and `analysis/checks_pcba.json`:

1. **Split identity.** Official CSV row ids == payload == cache order
   elementwise (Lipo); official CSV == frozen JSON and raw index hashes (PCBA).
2. **Train-only support.** Rebuilt-from-Train `Omega` and `P` equal the frozen
   support; no Valid/Test pair enters `Omega`.
3. **Only S2 changed.** Common parameter state bitwise equal to the original
   backbone; parameter counts as above.
4. **Initial forward equivalence.** Original vs JOINT on a real Train batch:
   max abs difference ≤ 1e-5 (Dropout eval mode).
5. **ADD residual.** `max |(I−P)J|` ≈ 0 in float64 at init and after training
   (float32 model-path residual reported separately).
6. **FIXED1 two-stage formula.** `stage1`/`stage2` class methods and
   `layer2_reads_original` are unchanged; stage outputs agree on the same batch.
7. **Gradients.** No `None`/non-finite gradients; table/Theta tensors receive
   non-zero gradients; zero-gradient-at-init parameters caused by the inherited
   zeroed residual/readout outputs are listed and must receive gradients after
   one optimiser step. Table rows exercised by the probe are reported against
   Train support (every row has ≥ 1 Train occurrence).
8. **Invariance.** Atom renumbering (raw rebuild for Lipo; `permute_atoms` for
   PCBA), path-record order permutation, and batch regrouping (7-row chunks vs
   full batch) each change predictions by ≤ 1e-5.
9. **Checkpoint replay.** Historical A/W2/FIXED1 checkpoints evaluate to their
   recorded Valid metrics (≤ 1e-5); Test replay is compared at 5e-4 because
   ranking ties move AP at that scale across devices/pipelines. Each new best
   checkpoint is replayed after training with the same tolerance.
10. **Plan/recipe identity.** Lipo batch-order SHA256 equals `1a3a440f…`;
    PCBA plan prefix SHA256 equals `8c857251…`.

Unseen-pair fallback is exercised on the real Valid/Test unseen codes: values
must be finite and must not create parameters.

## 6. Evaluation and analysis plan

Primary metrics: Lipo raw RMSE (Å); PCBA official OGB macro AP (masked BCE
secondary). Report best-Valid checkpoint, final epoch, and last-20-epoch mean,
with same-epoch curves.

**Lipo:** `C1_JOINT − C1`, `C1_ADD − C1`, `C1_JOINT − C1_ADD` on Valid and Test.
Paired per-molecule absolute errors and RMSE; paired bootstrap over molecules
(2,000 replicates, seed 20261126) for RMSE difference and relative change.
Extreme-molecule and low-support sensitivity: worst-|error| quantiles, molecules
whose S2 codes are rare in Train, and per-code support strata. Reference effect
size (exploratory only): ≥ 3% relative Test RMSE improvement.

**PCBA:** `Δ_W2 = W2_JOINT − W2_ORIGINAL`,
`Δ_FIXED = FIXED1_JOINT − FIXED1_ORIGINAL`,
interaction `Δ_FIXED − Δ_W2`, on Valid and Test. Paired per-task AP deltas,
molecule-level paired bootstrap for macro AP (2,000 replicates), number of tasks
improved/degraded, task-support strata, and a common high-support subset
(≥ 20 positives in the relevant split) to check that an effect is not driven by
a few tasks. Reference effect size ≈ 0.005 AP on both splits; report the gap to
Test AP 0.28/0.29. Test is descriptive.

## 7. Interpretability (frozen scope)

Reuse the exact v0 decomposition for the JOINT table:

```
J_add = P J,        J_int = (I − P) J,        J = J_add + J_int
```

Report magnitudes, exact closure `||J − J_add − J_int||`, per-component and
per-type summaries, and the LocalMLP first-layer projection usage of the joint
columns (how much of the first-layer input weight acts on the joint channel).
One frozen `ADD_ONLY` intervention on the single best new model only: replace
its joint values by the marginal projection at inference and measure the
metric change. No new structural discovery, trees, dictionaries, or feature
search.

## 8. Budget, priority and stop rules

* GPU target 3.0 h, hard cap 4.0 h; wall-clock hard cap 3.5 h.
* Preferred GPU 1 on `res`, GPU 0 when free; all formal runs via `rr`.
* Timing preflight measures real training steps **and** full three-split
  evaluation per epoch; the projection includes every-epoch eval.
* Priority: Lipo `C1_JOINT` then `C1_ADD`; PCBA `FIXED1_JOINT` then
  `W2_JOINT`. An arm is never shortened to fake a complete comparison; if the
  hard cap is reached, remaining arms are reported as not run with measured
  cost.
* If only the small Lipo data benefits, label the result a local
  parameterisation improvement.
* If `FIXED1_JOINT` clearly gains, keep it as the next-stage backbone candidate.
* If both PCBA arms fail, stop typed-S2 table / marginal-decomposition work.
* No new models are trained with leftover budget.

## 9. Deliverables

`PROTOCOL.md`, frozen config, code, support/split audit, pre/post checks,
`runs/{arm}` artifacts (curve, checkpoints, predictions, status), main tables,
curves, decomposition and intervention results, `REPORT.md`, tests, and a budget
ledger. The report records reused assets, split identity, execution deviations,
incomplete items, and conclusion limits; it does not design an automatic v2.

## 10. Deviations after freeze

None at freeze time. Any later deviation (e.g. GPU reassignment, a failed arm,
a tolerance justified by a measured numerical effect) is listed in `REPORT.md`
with its timestamp and impact.
