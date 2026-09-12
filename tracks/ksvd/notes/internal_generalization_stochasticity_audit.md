# Internal Generalization & Training-Stochasticity Decomposition Audit

**Date:** 2026-09-12
**Scope:** training-dynamics / generalization audit (no architecture change)
**Module:** `tracks/ksvd/experiments/luyin16/zinc_internal_generalization_stochasticity_audit.py`
**Results:** `tracks/ksvd/results/internal_generalization_stochasticity_audit/`
**Tests:** `tracks/ksvd/tests/test_internal_generalization_stochasticity_audit.py` (18 pass)
**Official valid:** never loaded. **Official test:** never loaded.

**Final verdict: `CHECKPOINT-SELECTION NOISE DOMINANT` (mandate Case A).**

The 82,115-parameter compact-v4-smallhead seed spread seen on official valid does
**not** reproduce as a material difference on the independent internal probe:
`S_raw = 0.00382 < 0.004`, paired-bootstrap 95% CI `[-0.00210, +0.00984]`
(crosses zero, `P(diff>0)=0.894`). But both canonical runs show **material
selection regret** (mean `0.00338 >= 0.002`) and sharp raw-800 minima. The
apparent seed spread is dominated by checkpoint-selection noise, not by a stable
training-trajectory generalization difference.

---

## 1. Motivation

The optimized compact-v4 baseline was extended to the 82,115-param
`compact-v4-smallhead` (`notes/compact_v4_smallhead_e2e_compression.md`). On the
two development seeds it reached official-valid MAE

| seed | official-valid |
|---|---:|
| 0 | 0.145334 |
| 1 | 0.138059 |

i.e. a seed contrast of **0.007275**. Both numbers were produced by
`official train (10K) -> official valid` checkpoint selection. This stage asks
whether that contrast is a *real* training/generalization phenomenon or a
selection artefact, using an internal probe that never participates in
optimization or selection.

## 2. Why representation redesign was paused

The preceding audits removed the local-structure hypotheses one by one:

* P1 learned composer -> NO-GO; P2 relation refresh -> sub-threshold NO-GO;
* explicit cycle cells -> NO-GO; covariance / endpoint / triad witnesses -> NO-GO;
* attribute factorization -> NO-GO; FM head / function basis -> NO-GO;
* parameter allocation -> only a compression donor, no R-SUPPORTED recipient;
* stagewise representation collision audit -> **Case E** (no clear stagewise
  collision).

With no stable representation boundary identified, the remaining unexplained
observation is the *seed-dependent spread of the same 82K architecture*. That is
a training-dynamics question, not a structural-module question.

## 3. Why existing 10K-trained checkpoints cannot answer this question

The existing smallhead checkpoints were trained with **all 10,000 official-train
molecules**. The 800 selection set and the 2000 probe therefore already entered
`loss.backward()` / `optimizer.step()`. They cannot be used to measure
generalization to held-out data. This stage retrains **from scratch on 7200
molecules only**; the 800 is inference-selection only and the 2000 is opened
only after training finishes.

## 4. Internal 7200/800/2000 protocol

The audit reuses the frozen official-train deterministic hash split
(`split_seed = optimized-manifold-broad-state-screen-v1-20260919`):

```text
7200 = optimization train      (gradient updates only)
800  = checkpoint selection     (inference selection + early stopping only)
2000 = untouched internal probe (opened only after training completes)
```

`split_inventory.json` recomputes the assignment and verifies it is
index-for-index identical to the frozen manifest
(`assignment_sha256 = 7b6704d482defbe91765b5348df824f840ba1e22aea6ff7844b0abe83591f386`,
`all_match = true`). No new split was drawn.

## 5. Why official valid is excluded

Official valid has been used for architecture development in every previous
stage. Using it here for anything (selection, diagnostics, decision, subgroup or
sanity) would re-contaminate the environment. The audit therefore never loads
official valid. The internal 7200/800/2000 split of official train is the only
data used.

## 6. Seed factorization: initialization vs trajectory

Two independent seeds are defined:

* **I (initialization seed)** controls only trainable parameter initialization;
* **T (trajectory seed)** controls the DataLoader shuffle generator **and** the
  global training RNG. This matters because the compact-v4 backbone contains
  Dropout modules (`patch_encoder`, `global_encoder`, `relation_encoder`,
  `pair_encoder`, `center_update`), so dropout is a genuine training-time
  stochastic operation and is assigned to T.

`seed_factorization_lock.json` records the separation preconditions:

* model construction depends only on I and completes before the trajectory RNG
  is reset from T, so *same I, different T* gives identical initial tensors;
* DataLoader order depends only on T through its dedicated generator, so *same
  T, different I* gives the identical first-epoch order fingerprint.

`I0`/`I1` reproduce the historical compact-v4-smallhead shared initial tensors
exactly: the recomputed shared-tensor SHA-256 matches
`initialization_match_seed0.json` / `initialization_match_seed1.json`
(`max_abs_diff = 0.0` in the historical files).

## 7. Canonical training protocol

Identical to compact-v4-smallhead, unchanged:

```text
optimizer = Adam
lr = 1e-3
weight_decay = 1e-5
batch_size = 128
max_epochs = 240
patience = 40
scheduler = none
loss = L1
gradient_clip_norm = 5.0
checkpoint selection = best 800-selection MAE
```

7200 molecules only: 57 optimizer steps per epoch.

| run | I | T | trained epochs | optimizer steps | raw best epoch | raw select-800 | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0T0 | 0 | 0 | 240 | 13,680 | 232 | 0.152610 | 864 s |
| I1T1 | 1 | 1 | 213 | 12,141 | 173 | 0.154226 | 768 s |

I0T0 reached the horizon (`early_stopped = false`); I1T1 early-stopped at epoch
213 (best 173). This is recorded as a limitation, not repaired (no horizon
extension is authorized).

## 8. Probe information firewall

A runtime `InformationFirewallError` guard replaces the official valid/test
extraction entry points with raising stubs. `train_run` contains no reference to
`data["probe"]`; only `data["train"]` (gradient) and `data["select"]` (inference
selection) are touched. The probe is unlocked only by `offline_trajectory`.
`integrity_gates.json` records 16/16 passing gates (T1-T16; T17-T20 are Stage-2
only and were not run).

**Representation boundary, stated explicitly.** The compact-v4 tokenizer
vocabulary and the patch/topology standardizers are *target-independent* and are
fit once on the full official train, exactly as in the canonical architecture, to
preserve the exact 82,115-param fingerprint and the historical initialization
continuity. No probe *target* is ever seen during training, the probe is never
used for gradient updates or selection, and the probe MAE is only reconstructed
after training completes. This is a transductive, label-free preprocessing
choice; it is the only place where probe *inputs* influence the fixed
representation. It is identical for both runs, so it cannot bias the I/T
comparison.

## 9. Epoch snapshot protocol

Every trained epoch writes `snapshots/<run_id>/epoch_XXX.pt` (model weights
only). `checkpoint_manifest_<run_id>.json` records the per-epoch snapshot path,
its SHA-256, the select-800 MAE and the training loss. Snapshots contain no
probe-derived state.

## 10. Raw checkpoint selection

`e_raw = argmin_e MAE_800(e)`, earliest tie. This is the checkpoint the
canonical rule would actually deploy:

| run | e_raw | MAE_800(e_raw) |
|---|---:|---:|
| I0T0 | 232 | 0.152610 |
| I1T1 | 173 | 0.154226 |

## 11. Smoothed selection diagnostic

Pre-registered centred 5-epoch moving average of `MAE_800`, restricted to epochs
with a full window; `e_smooth = argmin_e mean` (earliest tie). It is a
retrospective diagnostic only and does **not** change the training rule.

| run | e_smooth | probe@e_smooth | smooth gain (raw - smooth) |
|---|---:|---:|---:|
| I0T0 | 217 | 0.137923 | **-0.006244** |
| I1T1 | 182 | 0.135462 | **-0.007605** |

Mean smooth gain `-0.006925`. In this environment the 5-epoch smoothed selector
was **worse** on the probe for both runs, so the pre-registered stronger
selection-noise-support condition is **not** met. Smoothing alone is not a fix.

## 12. Offline probe reconstruction

After both runs finished, every snapshot was reloaded and evaluated in eval mode
on 7200 (exact MAE), 800 and 2000. `offline_trajectory_<run>.csv` holds the three
curves; the 2000 curve is post-training reconstructed and was never visible
during training. Per-molecule probe absolute errors at `e_raw` are saved for the
paired bootstrap.

## 13. Independent-probe seed spread

| run | probe@e_raw |
|---|---:|
| I0T0 | 0.131679 |
| I1T1 | 0.127857 |

`S_raw = |0.131679 - 0.127857| = 0.003821`, below the pre-registered material
gate `0.004`. For reference, the historical official-valid contrast was
`0.007275`; the independent probe shrinks the contrast by ~47%. The probe-oracle
spread is `S_oracle = 0.005683` — a reminder that *per-trajectory best* probe
performance still differs somewhat, but the deployable raw-selected difference
does not.

## 14. Selection regret

`R_sel = probe@e_raw - probe@e_oracle`:

| run | e_oracle | probe@e_oracle | R_sel |
|---|---:|---:|---:|
| I0T0 | 223 | 0.129228 | **0.002451** |
| I1T1 | 197 | 0.123545 | **0.004312** |

Mean `R_sel = 0.003381 >= 0.002` -> the checkpoint-selection noise criterion is
triggered. Both runs have a real regret: the 800-selection missed the probe-best
epoch by 9 and 24 epochs respectively, costing 0.0025 / 0.0043 MAE.

## 15. Checkpoint sharpness

The raw-800 minima are isolated dips, not broad basins:

| run | W5 (median +/-5) | W10 (median +/-10) | epochs within +0.001 | within +0.002 |
|---|---:|---:|---:|---:|
| I0T0 | 0.005464 | 0.009838 | 2 | 6 |
| I1T1 | 0.009059 | 0.009059 | 3 | 4 |

A handful of adjacent epochs are effectively tied, and a single lucky dip at
`e_raw` wins the argmin. This is the mechanical origin of the selection noise.

## 16. Select / probe curve agreement

Despite the noise, the 800 curve is a faithful *directional* tracker of the
independent probe:

| run | Pearson | Spearman |
|---|---:|---:|
| I0T0 | 0.9972 | 0.9871 |
| I1T1 | 0.9988 | 0.9913 |

The problem is not that 800 is uninformative; it is that its epoch-to-epoch noise
is large enough that its argmin is displaced relative to the probe argmin.

## 17. Generalization-gap dynamics

`G(e) = MAE_2000(e) - MAE_7200^eval(e)`:

| run | raw | oracle | last |
|---|---:|---:|---:|
| I0T0 | 0.07878 | 0.07437 | 0.05789 |
| I1T1 | 0.07451 | 0.07119 | 0.04716 |

The gap is large, but gap size alone is **not** treated as overfitting proof
(mandate section 33). Late-phase train-eval MAE actually *rose* (I0T0:
0.0529 at epoch 232 -> 0.0981 at 240), so the late degradation is not a clean
"train improves while probe worsens" overfit pattern.

## 18. Overfit signature

Pre-registered material overfit signature: both runs satisfy
`Delta_train_after >= 0.005` **and** `Delta_probe_after >= 0.003`, where

* `Delta_train_after = MAE_7200(e_oracle) - MAE_7200(e_last)`,
* `Delta_probe_after = MAE_2000(e_last) - MAE_2000(e_oracle)`.

| run | Delta_train_after | Delta_probe_after | material |
|---|---:|---:|---|
| I0T0 | -0.043251 | +0.026772 | false |
| I1T1 | -0.077541 | +0.053514 | false |

`Delta_train_after` is strongly negative: training did **not** keep improving
after the probe-best epoch. The probe did degrade afterwards, but the joint
signature fails, so there is **no** material regularization/overfit signal under
the pre-registered test.

## 19. Stage 1 decision

```text
S_raw = 0.003821 < 0.004            -> not material
paired bootstrap 95% CI [-0.00210,+0.00984] crosses 0
mean selection regret = 0.003381 >= 0.002 -> material selection noise
MATERIAL_OVERFIT_SIGNATURE = false
smooth selector = worse on both runs
```

Verdict: **`CHECKPOINT-SELECTION NOISE SIGNAL`** (final mandate Case A:
`CHECKPOINT-SELECTION NOISE DOMINANT`). The Stage 2 factorial is **not**
purchased, because the independent-probe material stochasticity gate is not met.

## 20. Conditional 2x2 factorial

Not run. The Stage-1 trigger (`S_raw >= 0.004` and paired-bootstrap CI excluding
zero) fails, so `I0T1` and `I1T0` were not bought. No placeholder files were
created.

## 21. Initialization effect

Not estimated (Stage 2 not triggered). For reference, the seed contrast present
in the historical valid numbers does not survive as a material independent-probe
contrast.

## 22. Trajectory effect

Not estimated (Stage 2 not triggered). Note that the two canonical runs differ
in both I and T, so the observed sub-threshold probe difference cannot be
attributed to either factor from Stage 1 alone.

## 23. Interaction

Not estimated (Stage 2 not triggered).

## 24. Raw-vs-oracle factor interpretation

Not applicable: the factorial was not run. Descriptively, `S_oracle = 0.005683`
is larger than `S_raw = 0.003821`, which is consistent with part of the apparent
seed difference being a property of *which epoch got selected* rather than of the
deployable selected model.

## 25. What is and is not proven

**Proven (within this internal environment):**

* The apparent 82K seed spread does **not** reproduce as a material
  independent-probe difference (`S_raw < 0.004`, bootstrap CI crosses zero).
* Both canonical runs have material selection regret (`mean 0.00338 >= 0.002`)
  and sharp raw-800 minima.
* The 800 curve tracks the probe direction but its epoch-noise displaces the
  argmin.
* The pre-registered material overfit signature does not fire.
* The smoothed selector did not help here.

**Not proven / not claimed:**

* that initialization or trajectory stochasticity is zero;
* that a larger seed panel would not reveal a real difference;
* that the 7200-trained model is comparable to the 10K-trained model
  (different train size and different evaluation sets — see section 26);
* that any stabilization method (EMA/SWA/averaging) would help.

## 26. Probe reuse limitations

The 2000 probe was used for this mechanism audit. After this stage it must **not**
be treated as a normal HPO selection set. Future intervention design must not
tune hyperparameters against this probe. Any future intervention must first
pre-register a single candidate and then decide between a *new* internal
cross-validation protocol or a single official-valid confirmation. Repeated
trials on this exact 2000 set are forbidden.

## 27. Next hypothesis authorization

Only one family is authorized: **`checkpoint_selection_stabilization`**. The
audit explicitly does **not** authorize EMA/SWA, checkpoint averaging, snapshot
ensembles, regularization search, optimizer-family search or another structural
module. The natural design space is a *pre-registered selection rule* tested on
a fresh internal split (e.g. a multi-epoch/tie-band rule, or an unbiased
held-out selection estimate), not a change to the training objective or
architecture. No such experiment is implemented here.

## 28. Final verdict

**`CHECKPOINT-SELECTION NOISE DOMINANT` (Case A).**

The 82,115-param compact-v4-smallhead seed spread observed on official valid is
dominated by checkpoint-selection noise. On a genuinely independent internal
probe the two canonical runs differ by `0.00382` MAE with a bootstrap CI that
crosses zero, while both runs lose `0.0025-0.0043` MAE simply because the noisy
800-selection argmin lands away from the probe optimum. Official valid and
official test remain completely outside this audit.

---

## Q1-Q20

* **Q1.** `7200 optimization_train / 800 checkpoint_selection / 2000 internal_probe`;
  assignment SHA-256 `7b6704d4...`; matches the frozen manifest (`all_match=true`).
* **Q2.** I0T0: 240 epochs / 13,680 steps. I1T1: 213 epochs / 12,141 steps.
* **Q3.** I0T0 raw best epoch **232**.
* **Q4.** I1T1 raw best epoch **173**.
* **Q5.** 800 best MAE: I0T0 **0.152610**, I1T1 **0.154226**.
* **Q6.** Independent-2000 MAE at the raw checkpoint: I0T0 **0.131679**,
  I1T1 **0.127857**.
* **Q7.** `S_raw = 0.003821` (< 0.004 material gate).
* **Q8.** Paired bootstrap (B=2000, fixed seed): MAE difference `0.003821`,
  95% CI `[-0.002099, +0.009841]`, CI crosses zero, `P(diff>0)=0.894`
  -> does **not** support a material seed difference.
* **Q9.** Probe-oracle: I0T0 epoch 223 MAE **0.129228**; I1T1 epoch 197 MAE
  **0.123545**.
* **Q10.** Selection regret: I0T0 **0.002451**, I1T1 **0.004312**.
* **Q11.** Mean selection regret **0.003381 >= 0.002** -> material.
* **Q12.** 5-epoch smooth selector: I0T0 epoch 217 probe 0.137923 (gain
  -0.006244); I1T1 epoch 182 probe 0.135462 (gain -0.007605). It was **worse**
  than raw on the probe for both runs.
* **Q13.** Raw minima are isolated dips: I0T0 W5 0.005464 / W10 0.009838, 2
  epochs within +0.001; I1T1 W5 0.009059 / W10 0.009059, 3 epochs within +0.001.
* **Q14.** 800-vs-2000 correlation: I0T0 Pearson 0.9972 / Spearman 0.9871;
  I1T1 Pearson 0.9988 / Spearman 0.9913.
* **Q15.** `MATERIAL_OVERFIT_SIGNATURE = false` (train also degraded after the
  oracle epoch).
* **Q16.** The 2x2 factorial was **not** triggered (`S_raw < 0.004`, bootstrap CI
  crosses zero).
* **Q17.** `E_I`, `E_T`, `E_IT`: **not estimated** (factorial not run).
* **Q18.** The observed spread is best attributed to **checkpoint-selection
  noise**, not to a stable initialization/trajectory effect measurable on this
  probe.
* **Q19.** Authorized next family: **`checkpoint_selection_stabilization`**.
* **Q20.** Final case: **`CHECKPOINT-SELECTION NOISE DOMINANT` (Case A)`**.
