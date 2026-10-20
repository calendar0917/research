# REPORT — zinc-exact-structure-residual-v1

**Status: CLOSED — negative; the pre-registered phase-2 rule did not trigger.**

Ten formal trajectories (B, E@0.1, E@1.0, H@0.1, H@1.0 × seeds 0/1), 2.208
A100-hours, no official-test access. Both residual arms are substantially worse
than the frozen baseline on the exposed valid split. The parent-sharing prior
consistently reduces the damage relative to independent residuals, but it does
not come close to the baseline. Phase 2 was **not** purchased.

Round provenance:

| stage | commit | job id | wall |
|---|---|---|---|
| protocol freeze | `8df7986` | `esr-freeze-c645aed5` | 7 s |
| prep (CPU) | `8df7986` | `esr-prep-429aa0ca` | 41 s (35 s compute) |
| gates (CPU) | `8df7986` | `esr-gates-a20412ac` | 35 s |
| smoke (GPU) | `8df7986` | `esr-smoke-0fc2907f` | 15 s |
| seed 0 chain (GPU 0) | `8df7986` | `esr-p1-gpu0-91323d9d` | 64m20s |
| seed 1 chain (GPU 1) | `8df7986` | `esr-p1-gpu1-7c09d55c` | 69m55s |
| prediction freeze | `8df7986` | `esr-freeze-preds-a626b735` | 6 s |
| score | `ce7d64a` | `esr-score-p1-03059273` | 8 s |
| mechanism / groups / ledger | `ce7d64a` | `esr-analyze-p1-f0dece3e` | 10 s |

The first attempt at `esr-freeze` (`b71bc5d1`, commit `abbd97d`) failed because
the git-ignored `valid_y_read_once.npy` is not present on the remote checkout;
see §10. All numbers below come from the runs at `8df7986` and the frozen
predictions; no artifact was retrained or replaced.

---

## 1. Question

On the mature P3 system (BJOC-Null + train-only masked SSL + R3, 413,505
parameters), does adding an **exact local structure residual** — a per-type
correction indexed by the atom's rooted attributed induced r2/r1 patch — improve
held-out generalisation, and does a **real r2→r1 parent-sharing prior** beat an
independent per-type residual at matched capacity?

Primary metric: fixed last-5-epoch parameter-mean soup, raw-y MAE on the official
valid 1,000. `delta = error(candidate) − error(B)`; negative favours the
candidate. The valid split is an exposed development set; official test was
never constructed or loaded.

## 2. Answers (pre-registered questions)

1. **Does the exact residual improve generalisation?** No. Every E/H run at both
   `eta` levels and both seeds is worse than B, by +0.0148 to +0.0310 valid MAE
   (seed mean). All four phase-2 checks fail in the harmful direction.
2. **Does parent-child sharing beat independent residuals?** Yes, consistently:
   `H − E = −0.00435` (eta 0.1) and `−0.00454` (eta 1.0) seed-mean, with the same
   sign for all four (seed, eta) pairs and bootstrap CIs excluding zero. But both
   arms lose to B by 0.015–0.031, so this is harm reduction, not incremental
   value.
3. **Does the branch participate in prediction?** Yes, and far more than the
   intended "small correction" regime. Setting the trained `beta` to zero at
   inference (eta = 1.0) raises valid MAE from 0.090–0.102 to 0.465–0.518 and train MAE from
   0.0031 to 0.44–0.47. The jointly trained network routes the prediction through
   the type lookup.
4. **Where is the damage?** It is broad, not confined to OOV or rare-type
   molecules: ordinary, tail, large-ring, all fallback-burden groups and all
   size terciles degrade. The largest absolute degradations are on tail
   molecules and on molecules with partial r2 fallback.
5. **Cost / reproducibility.** 2.208 GPU-hours for the whole round (27.6% of the
   8 h cap), 35 s CPU preprocessing, 10/10 runs and predictions hash-frozen,
   all 10 gates passed before freeze, no test access.

## 3. Primary table

Valid MAE (official valid 1,000, fixed soup), per seed and seed mean:

| config | seed 0 | seed 1 | seed mean | Δ vs B | per-seed Δ |
|---|---:|---:|---:|---:|---:|
| B | 0.0815892 | 0.0778594 | **0.0797243** | — | — |
| E@0.1 | 0.1143073 | 0.1071823 | 0.1107448 | +0.0310205 | +0.0327201, +0.0293209 |
| E@1.0 | 0.1018374 | 0.0963046 | 0.0990710 | +0.0193466 | +0.0202482, +0.0184462 |
| H@0.1 | 0.1103749 | 0.1024062 | 0.1063906 | +0.0266663 | +0.0287857, +0.0245468 |
| H@1.0 | 0.0994897 | 0.0895631 | 0.0945264 | +0.0148021 | +0.0179005, +0.0117037 |

Paired molecule bootstrap (5,000 resamples, seed 0), candidate − B:

| contrast | mean Δ | 95% CI | fraction of molecules improved |
|---|---:|---|---:|
| E@0.1 − B | +0.031020 | [+0.022165, +0.042790] | 0.414 |
| E@1.0 − B | +0.019347 | [+0.011905, +0.028791] | 0.441 |
| H@0.1 − B | +0.026666 | [+0.018517, +0.037319] | 0.423 |
| H@1.0 − B | +0.014802 | [+0.007574, +0.023891] | 0.467 |
| **H@0.1 − E@0.1** | **−0.004354** | **[−0.006328, −0.002389]** | 0.550 |
| **H@1.0 − E@1.0** | **−0.004545** | **[−0.006498, −0.002652]** | 0.581 |

The bootstrap describes molecule-level uncertainty for these two seeds only; it
is not an independent-data confirmation.

Train (last-5 soup) MAE:

| config | seed 0 | seed 1 |
|---|---:|---:|
| B | 0.001875 | 0.001711 |
| E@0.1 | 0.002792 | 0.002463 |
| E@1.0 | 0.003009 | 0.002598 |
| H@0.1 | 0.002679 | 0.002537 |
| H@1.0 | 0.002950 | 0.002569 |

Residual runs are *worse on train as well*, so this is not a "train-only
improvement that failed to transfer". Both splits degrade; the valid degradation
is roughly an order of magnitude larger in absolute terms.

### 3.1 Decision checks (frozen rule)

| check | E@0.1 | E@1.0 | H@0.1 | H@1.0 |
|---|---|---|---|---|
| both seeds improve | no (+0.0327, +0.0293) | no (+0.0202, +0.0184) | no (+0.0288, +0.0245) | no (+0.0179, +0.0117) |
| seed-mean gain ≥ 0.003 | no (−0.0310) | no (−0.0193) | no (−0.0267) | no (−0.0148) |
| ordinary region not reversed | no (+0.0252) | no (+0.0154) | no (+0.0220) | no (+0.0114) |
| drop-B-worst-10 not reversed | no (+0.0262) | no (+0.0156) | no (+0.0220) | no (+0.0113) |
| **trigger** | **no** | **no** | **no** | **no** |

No family triggered; `decision.json → phase2_plan.authorized = false`. No third
or fourth seed, no extra `eta`, rank, radius or module search was run.

## 4. How to read the negative result

* **Not "structure information is useless".** The branch is active, receives
  gradient, changes predictions materially, and the two `eta` levels are clearly
  separated. The result is about *this residual parameterisation and prior at
  this vocabulary granularity*, not about the informational value of local
  structure.
* **Not a train-only win.** Train soup MAE is worse for every E/H run
  (0.0025–0.0030 vs 0.0017–0.0019), and valid is much worse.
* **A shortcut, not a small additive bias.** The intended regime was
  `h_new = h_P3 + a[t]` with `a` small and the frozen P3 stack carrying the
  prediction. What happened is the opposite: the stack learned to rely on `a`.
  Zeroing `a` destroys both train and valid predictions (§5.2), and the reader
  shifts weight mass toward the `a_sum` input block (§5.4). The per-type lookup
  fits the 10k train molecules but is a rule that does not transfer to new
  molecules, even when the individual types are well supported.
* **Hierarchy helps only relatively.** `H` beats `E` by ~0.0045 with a stable
  sign, which is consistent with the parent–child prior doing what it was
  designed to do. It does not rescue the approach: `H@1.0` is still +0.0148
  behind B, i.e. 19% higher valid error.

The closest pre-registered mapping is "H beats E but does not match B"; the
protocol's `H ≈ B` clause is not met (0.0945 vs 0.0797), so this is **not**
reported as "avoids lookup damage with no incremental value". It is reported as:
the residual construction is net harmful here, and sharing reduces but does not
remove the harm.

## 5. Mechanism evidence (post-hoc, not part of the frozen protocol)

### 5.1 Prior and correction scale

| run | prior `R` (seed 0/1) | mean type norm | mean atom norm on valid | depth-1 mean norm |
|---|---|---:|---:|---:|
| B | — | — | 0 | — |
| E@0.1 | 0.01036 / 0.01081 | 0.433 / 0.462 | 0.452 / 0.505 | **0.000 / 0.000** |
| E@1.0 | 0.00145 / 0.00163 | 0.165 / 0.181 | 0.189 / 0.226 | **0.000 / 0.000** |
| H@0.1 | 0.01194 / 0.01279 | 0.357 / 0.388 | 0.376 / 0.430 | 0.209 / 0.235 |
| H@1.0 | 0.00156 / 0.00162 | 0.135 / 0.145 | 0.157 / 0.180 | 0.090 / 0.104 |

`lambda = 1.157879` at `eta = 1.0`, `0.115788` at `eta = 0.1`. The prior does
bite: `R` falls 5–7× from eta 0.1 to eta 1.0, and mean correction norms fall
~2.6×. `eta = 1.0` is consistently *less harmful* than `eta = 0.1`
(+0.010–0.015 valid MAE), i.e. the learned correction content is net negative
and stronger shrinkage improves it.

**The `E` arm's depth-1 rows are exactly zero.** Every train atom has a retained
r2 type (train r2 coverage is 100%), so no training atom ever falls back to
r1; the r1 rows receive no data gradient and the prior pulls them to zero. `H`
trains them through their r2 children. This mechanical asymmetry explains part
of `H > E`, but not all of it: `H@1.0` is also better than `E@1.0` on molecules
with **no** r2 fallback at all (0.06183 vs 0.06415) and on molecules with no
low-support atoms (0.04026 vs 0.04519).

### 5.2 Zero-correction dependence probe

Full model vs the same soup with `beta = 0` (labels are the exposed valid labels;
this is a dependence probe, not a model-selection metric):

| run | valid full | valid zero | train full (2k) | train zero (2k) |
|---|---:|---:|---:|---:|
| E@1.0 | 0.1018 / 0.0963 | 0.5054 / 0.5179 | 0.00308 | 0.4722 |
| H@1.0 | 0.0995 / 0.0896 | 0.4670 / 0.4649 | 0.00327 | 0.4366 |

At `eta = 0.1` the same probe gives 0.543–0.581 (E) and 0.543–0.570 (H) on valid.
The trained function is not "P3 plus a small correction"; the correction is the
main carrier of the prediction on both splits.

### 5.3 Residual-scale sweep (seed 0, `eta = 1.0`)

Valid MAE when the trained `beta` is multiplied by `s`:

| s | 0 | 0.1 | 0.25 | 0.5 | **1.0** | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| E@1.0 | 0.5054 | 0.4560 | 0.3839 | 0.2701 | **0.1018** | 0.2511 |
| H@1.0 | 0.4670 | 0.4223 | 0.3570 | 0.2536 | **0.0995** | 0.2326 |

The sharp minimum exactly at the trained scale, with both halving and 1.5×
blowing up, is the signature of a co-adapted cancellation: the shared stack and
the lookup are tuned as a pair. It is not the profile of a small additive bias
on top of an intact predictor.

### 5.4 Where the reader looks

First reader layer (`321→256`) weight mass by input block, seed 0:

| model | z | a_sum | a_sq | g |
|---|---:|---:|---:|---:|
| B | 0.151 | 0.289 | 0.257 | 0.303 |
| E@1.0 | 0.100 | 0.368 | 0.332 | 0.200 |
| H@1.0 | 0.102 | 0.387 | 0.317 | 0.195 |

The residual contributes 14.9% (E) / 11.7% (H) of the norm of `Σ_v h_v` on
valid (p90 28.5% / 22.6%), and the reader moves weight mass from `z`/`g` toward
`a_sum`/`a_sq`. This is consistent with the branch becoming a primary input
rather than a correction.

## 6. Pre-declared slices

Valid MAE by molecule group (seed mean; `n` is molecules):

| group | n | B | E@0.1 | E@1.0 | H@0.1 | H@1.0 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary (`y > -3`) | 935 | 0.06889 | 0.09413 | 0.08430 | 0.09091 | 0.08027 |
| tail (`y ≤ -3`) | 65 | 0.23557 | 0.34971 | 0.31158 | 0.32903 | 0.29953 |
| large ring (≥6) | 940 | 0.07934 | 0.10914 | 0.09768 | 0.10558 | 0.09376 |
| fallback none | 686 | 0.05559 | 0.07014 | 0.06415 | 0.06834 | 0.06183 |
| fallback partial | 290 | 0.11572 | 0.17831 | 0.15863 | 0.17064 | 0.15139 |
| fallback ≥0.25 | 24 | 0.33454 | 0.45482 | 0.37740 | 0.41775 | 0.34208 |
| low-support none | 180 | 0.03717 | 0.05077 | 0.04519 | 0.04828 | 0.04026 |
| low-support partial | 815 | 0.08858 | 0.12353 | 0.11075 | 0.11880 | 0.10649 |
| low-support majority | 5 | 0.16797 | 0.18648 | 0.13452 | 0.17531 | 0.09858 |
| size small (≤21) | 352 | 0.08723 | 0.11891 | 0.10484 | 0.11070 | 0.09638 |
| size medium (22–25) | 366 | 0.06920 | 0.10790 | 0.09729 | 0.10351 | 0.09327 |
| size large (≥26) | 282 | 0.08401 | 0.10425 | 0.09418 | 0.10475 | 0.09385 |

No pre-declared group improves under either family. H beats E in every group
at eta 1.0 (largest on tail −0.0121, fallback ≥ 0.25 −0.0353, size-small
−0.0085) and in every group but `size large` at eta 0.1 (+0.0005, negligible).
The 5-molecule `low-support majority` group has a large H−E (−0.036 at eta 1.0)
but is not interpreted at n = 5.

## 7. Vocabulary, support, coverage

* tokenizer `typed_tokenizer_v2_corrected`; r2 radius 2, r1 radius 1; r0 fixed
  zero correction.
* `K = 15,730` = 512 r1 + 15,218 r2; no truncation (`dropped_r2 = 0`), ancestor
  closure holds for every r2 type. Parameter cost 63,304; total **476,809**
  (cap 500,000; max types under the cap 21,527).
* r2 support (distinct train molecules): 6,784 types support 1, 4,139 at 2–4,
  1,762 at 5–9, 1,890 at 10–49, 491 at 50–199, 152 at ≥200; quantiles
  `[1, 1, 2, 5, 18, 41, 199, 5296]` (q0/q25/q50/q75/q90/q95/q99/q100).
* r1 support: quantiles `[1, 2, 9, 98, 551, 1451, 5127, 8808]`.
* train: 231,664 atoms, r2 retained 100%, zero-correction 0%, 15,218 distinct
  types used, 12.97% of atoms have support < 10.
* valid: 23,083 atoms, r2 retained 97.01%, **r2 OOV 2.92%**, zero correction
  0.069% (16 atoms), 4,243 distinct types used, 10.87% of atoms have support
  < 10.
* `sy = 1.472189` (train median 0.441003); `Sh2` = 16.386387 (seed 0),
  17.768767 (seed 1), computed from the frozen train-only SSL encoders.

## 8. Cost

| item | value |
|---|---:|
| formal GPU training (10 runs) | 7,940.5 s |
| prediction forwards | 7.4 s |
| smoke timing probe | 1.5 s |
| **phase-1 GPU total** | **2.208 h** (8 h cap; 27.6%) |
| wall per GPU | 64m20s (gpu0), 69m55s (gpu1) |
| CPU prep | 35.0 s (24.3 s patch keys) |
| CPU score + analysis | 18 s |
| CPU gates + freeze + pred freeze | 48 s |
| SSL training | 0 s (reused frozen train-only checkpoints) |

Smoke projected 3.171 h for phase 1 (cap 4.0, margin 1.20 + 300 s); actual was
2.208 h. No retry was charged to the GPU beyond the failed pre-freeze stage,
which ran on CPU.

## 9. Checks and provenance

* All 10 gates passed (`gates.json → all_pass: true`), including: zero residual
  reproduces B bit-identically (max abs 0.0); E/H share identical P3 weights,
  vocabulary, rank, parameter count and batch order; `beta` has a step-1
  gradient (0.0350), `P` has zero step-1 and non-zero step-2 gradient
  (0.00147); permutation invariance (5.96e-8); direct parent recomputation
  (191 checks) and ancestor closure; optimizer weight-decay split exactly
  {1e-5, 0}; frozen input hashes.
* Batch order matches the frozen R3 per-seed hash (`matches_frozen_R3: true`
  in every `train.json`).
* Predictions were hash-frozen (`prediction_freeze.json`,
  `frozen_before_valid_read: true`) before any valid metric was computed. The
  freeze file records all 10 prediction hashes; `metrics.json` records the label
  hash.
* Valid-label provenance: the single existing read from
  `bjoc_recipe_tuning_v1`, sha256 `8f6ee636…`, read UTC 2026-10-13, reused
  read-only. On the remote checkout the original ignored `.npy` is absent, so
  the array was re-materialised from the official raw valid split and
  **hash-checked byte-identical** against the committed lock before use
  (`metrics.json → valid_labels.reconstructed_from_official_raw_valid: true`).
  No metric was computed before the prediction freeze. Official test was never
  loaded (`official_test_loaded: false` everywhere).
* Baseline reproduction: in-round B seed mean 0.079724 vs the historical frozen
  R3 seed mean 0.079844 (Δ 0.00012); per seed 0.081589/0.077859 vs
  0.079101/0.080586. Prediction correlation with the historical R3 is 0.9990–
  0.9996, mean absolute prediction difference 0.032/0.038. This is the expected
  run-to-run spread of the same recipe (GPU reduction order, 22,680 Adam steps,
  sharp final loss); the primary contrasts are therefore computed against the
  **in-round B**, trained under the identical harness, not against the
  historical number.

## 10. Deviations, bugs, amendments

1. **Missing ignored label file on the remote** (`esr-freeze` at `abbd97d`
   failed). The `.npy` copy is git-ignored and had never been transferred. The
   frozen label array was reconstructed from the official raw valid split and
   accepted only because its sha256 equals the committed single-read lock
   (`8f6ee636…`). This is a byte-identical re-materialisation of an
   already-recorded read, not a new scientific read; it is recorded in
   `metrics.json`.
2. **Frozen runner's `score`/`analyze` could not map registered config names to
   run keys.** `E@0.1` in the config maps to the on-disk key `E_eta0.1`; the
   frozen orchestration used the config string directly, and read
   `drop_reference_worst_k` from the wrong level of the decision block. The
   frozen training/verification code was not edited (its hashes still verify).
   Scoring and analysis were done by `code/esr_score.py`, a separate post-freeze
   module that imports and reuses the frozen helpers (`_err`, `_contrast`,
   `verify_freeze`, `valid_labels`, `model_for_run`). The first two `esr-score`
   attempts (`8f390d42`, `8f201880`) failed on these two plumbing bugs and are
   kept in `rr jobs`.
3. **`train.json → prior.mean_R_per_step` is 0.0 for every residual run.** The
   training loop accumulated the per-step prior in `epoch_prior` but serialised
   a different variable. This is a logging-only field: it is never used for the
   loss, the optimizer, the soup, prediction or selection. The report uses
   (a) the exact prior recomputed from the frozen soup (`mechanism.json →
   prior_value`) and (b) the 20-epoch logged curve values. The frozen runner was
   not edited; see `AMENDMENT.md`.
4. **Post-hoc diagnostics** (`code/esr_diag.py`, `analysis/diagnostics.json`)
   were added after the decision. They only read frozen artifacts
   (soups, predictions, cached types) and are labelled as not part of the
   protocol. They do not enter any decision.

## 11. Non-claims

This is a two-seed development result on one exposed valid split. It is not an
official-test number, not a blind evaluation, not a confirmation, and not a
paper-novelty claim. It does not show that local structure is uninformative, and
it does not establish that no exact-structure residual could help; it shows that
*this* per-type additive residual with *this* prior, jointly trained at this
granularity, is net harmful on this split, and that real parent–child sharing
reduces but does not remove the harm. No further strength, rank, radius or
module search was performed, by design.

## 12. Reproduce

```bash
# CPU: patch types, train-only vocabulary, 10 gates (no valid labels)
uv run python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py --device cpu prep
uv run python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py --device cpu gates
# remote: freeze, five runs per GPU, freeze predictions, score
rr run res esr-freeze --pool res-local --gpus 0 --time 0:10:00 \
  --result tracks/structural-sufficiency/zinc_exact_structure_residual_v1 \
  -- python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py --device cpu freeze
rr run res esr-p1-gpu0 --pool res-gpu0 --gpus 1 --time 2:30:00 \
  --result tracks/structural-sufficiency/zinc_exact_structure_residual_v1/runs \
  --result tracks/structural-sufficiency/zinc_exact_structure_residual_v1/predictions \
  -- python tracks/structural-sufficiency/zinc_exact_structure_residual_v1/code/esr_runner.py --device cuda \
     chain --seed 0 --configs B,E_eta0.1,E_eta1,H_eta0.1,H_eta1
# seed 1 on res-gpu1, then:
python .../esr_runner.py freeze-predictions
python .../esr_score.py score --phase 1
python .../esr_score.py analyze
```

Artifacts: `runs/<key>_s<seed>/{train.json,soup.pt,curve.csv}`,
`predictions/valid_<key>_s<seed>.npz` + hash freeze, `analysis/{metrics,contrasts,
decision,mechanism,groups,diagnostics}.json`, `gpu_ledger.json`,
`run_manifest.tsv`, `curves/*.csv`.
