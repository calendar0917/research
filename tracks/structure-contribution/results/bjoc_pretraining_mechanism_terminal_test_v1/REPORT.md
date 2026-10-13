# REPORT — cscl-bjoc-pretraining-mechanism-terminal-test-v1

Protocol `cscl-bjoc-pretraining-mechanism-terminal-test-v1` · track `tracks/structure-contribution`
· date 2026-10-13 · frozen decision **`PRETRAINING_TEST_GAIN_REPLICATES`** (exposed official test)
· J-reset verdict **`J_RESET_NEUTRAL_VALID`** · target band 0.06–0.07 **not reached** (gap 0.003957).

This round did three things: (A) replayed the frozen B/P checkpoints and re-audited the
masked-pretraining legality; (B) located the mechanism on train/valid and settled the one
pre-registered development ablation (J-reset, scratch junction encoder, official valid only);
(C) scored the frozen B0/B1/P0/P1 models on the official test with **exactly one label read**,
after every prediction was written label-blind and hashed. The official test was already
exposed in an earlier round, so the test column below is an *exposed-test terminal comparison
of frozen models*, not a blind confirmation; no model, seed or hyper-parameter was chosen with
test data.

## 1. Official-test single-model results (primary table)

Official PyG ZINC `subset=True` test split, 1000 molecules, raw-y, single-model MAE
(seed-mean = arithmetic mean of the two single-model MAEs; predictions were never averaged).

| model | init | seed 0 MAE | seed 1 MAE | seed-mean MAE | seed-mean RMSE |
|---|---|---:|---:|---:|---:|
| B (scratch) | BJOC-Null scratch | 0.094243 | 0.095010 | **0.094627** | 0.176108 |
| P (masked-pretrained) | BJOC-Null + masked-structure SSL | 0.074825 | 0.073088 | **0.073957** | 0.149588 |
| **P − B (paired, molecule bootstrap 5000)** | | **−0.019418** | **−0.021922** | **−0.020670** | **−0.026519** |

Paired-molecule 95% CI for the seed-mean delta: **[−0.027157, −0.014712]** (upper bound < 0);
647/1000 molecules have a better P−B absolute error; relative reduction vs B **21.84%**.

Secondary official-test numbers (seed-mean): median abs P 0.042324 vs B 0.059299;
p90 0.146752 vs 0.184169; p95 0.226041 vs 0.255824; p99 0.623403 vs 0.642614;
max 2.2203 vs 2.5750; worst-molecule error share 3.0% (P) vs 2.7% (B).
Subgroups: ordinary `y > −3` (n=923) P 0.066750 vs B 0.089535 (**−0.022785**);
tail `y ≤ −3` (n=77) P 0.160342 vs B 0.155654 (**+0.004688**, signed mean +0.0483 vs +0.0349).
Removing the single worst row (402): **−0.022160**; top-1/5/10 removed: −0.018968 / −0.019602 /
−0.019648. y cross-check against the historical KSVD test labels: max abs diff **0.0**.

Reference (valid, not test): B seed-mean 0.099202, P seed-mean 0.083549, P−B −0.015653
(CI95 [−0.021051, −0.009904]).  The valid number is a development-set score and is **not**
a test score.

## 2. The six questions

**Q1. What is demonstrated about the source of the masked-structure pretraining gain, and
what is only hypothesis?**
*Demonstrated (facts).* (i) The gain is real on held-out raw y under a same-round matched
recipe: train10k fit P−B −0.0165 (ordinary −0.017318), official valid P−B −0.015653
(ordinary −0.017391), exposed test P−B −0.020670 (ordinary −0.022785). (ii) It is a
single-model effect at both seeds — every per-seed delta is negative in train/valid/test —
so it is not an ensemble or seed-selection artifact. (iii) It survives removing the single
worst molecule on valid and test. (iv) It is not pure optimisation speed: the previous
round's 50%-extra-update scratch control C remained worse on the 8k split, and P also fits
the train set better (fit MAE 0.0269 vs B 0.0434). (v) It is not encoder preservation via a
reduced LR: Q (encoder LR 3e-4) was decisively harmful (+0.019484 vs P on valid). (vi) The
pretext tasks are far above their marginal priors (atom acc ≈ 0.93, bond acc ≈ 0.995), all
three encoders move during SSL and move again during fine-tuning, so the pretrained state
acts as a prior, not a frozen extractor.
*Hypothesis (not proven).* That the causal route is "restoring masked atom/bond/junction
classes from the molecule's own structure forces a local-structure representation that is a
better starting point for the raw-y optimisation". Phase B shows the gain is concentrated in
ordinary molecules and slightly *negative* on the valid tail; which pretraining object
(atom vs bond vs junction) carries the gain is not identified — J-reset tests only the
junction coarse component; no learned feature is identified with a named chemical rule; and
no claim is made that the method fixes extreme-molecule generalisation.
*Mechanism context (Phase B).* Frozen `TAIL_GENERALISATION_GAP_SUPPORTED`: the pretraining
gain is a bulk/ordinary gain (train ordinary −0.017318, valid −0.017391) plus a valid-only
tail regression (valid tail +0.009341) concentrated in two rows (753 +1.30, 172 +0.68;
excluding both, tail delta −0.0218; 40/65 tail rows improve). The tail failures are
systematic magnitude *under*-prediction (signed dominates |error|, corr 0.995), the failing
extremes sit farther from train in a label-free composition/topology space (nearest-train
distance 2.37 vs 1.62), and their label neighbourhood is uninformative (5-NN median-y
correlation −0.154 tail vs 0.815 ordinary) — a coverage/statistics signature, not capacity.

**Q2. Is the pretraining legal (leakage-free)?** Yes — `LEGALITY_CONFIRMED`. The driver and
configuration hashes match `training_manifest.json`; the inherited audits are clean:
mask tokens free on all 10000 molecules (`MASK_TOKEN_SLOTS_FREE`), 200/200 atom and 200/200
bond hidden-truth swaps produce identical objects (`MASK_SWAP_INVARIANT`), and 100/100
masked-rebuild cases were shown leak-free (`MASKED_REBUILD_LEAK_FREE`), 10/10 correctness
gates. No validation/test structure, no target, no extra label entered the pretraining; the
masked task reconstructs only fields already present in the molecule's own graph, and the
gain was measured against a same-round matched scratch baseline. The replay audit also
confirms the frozen predictions are the artefact hashed last round (bit-identical,
maxdiff 0.0; `REPLAY_CONFIRMED`).

**Q3. Official-test MAE of the frozen B/P models, per seed and seed-mean?** See the primary
table: B0 0.094243, B1 0.095010, B seed-mean 0.094627; P0 0.074825, P1 0.073088, P seed-mean
**0.073957**. Paired seed-mean P−B = **−0.020670** (CI95 [−0.027157, −0.014712]); per-seed
deltas −0.019418 (seed 0) and −0.021922 (seed 1). Frozen decision rule checked item by item:
seed-mean < 0 ✓, both per-seed < 0 ✓, CI95 upper < 0 ✓, ordinary < 0 ✓, all-rows-minus-worst
< 0 ✓, no provenance/label-source failure ✓ → **`PRETRAINING_TEST_GAIN_REPLICATES`**.
Target band 0.06–0.07: **not reached** — P seed-mean 0.073957 is 0.003957 above the 0.07
upper edge (best single model P1 0.073088 is 0.003088 above). This statement concerns the
exposed-test version only and must not be called a new blind confirmation.

**Q4. Does the official-valid gain transfer to the exposed official test, including
ordinary/tail/RMSE?** Direction and significance transfer for the bulk, not for the tail or
RMSE signature. Valid P−B −0.015653 (CI [−0.021051, −0.009904]) → test −0.020670
(CI [−0.027157, −0.014712]); gain retained fraction 1.32 on the seed-mean. Ordinary molecules
agree (valid −0.017391 → test −0.022785). The tail does *not* replicate the valid
regression's magnitude: valid +0.009341 → test +0.004688 (n=77), with test signed tail error
only +0.048 (P) vs +0.035 (B); the valid-tail damage was dominated by the two extreme rows,
which are not the same failure on test. RMSE *reverses* direction: P is worse on valid
(+0.015246) but better on test (−0.026519). Overall: the pretraining gain is a robust
bulk/ordinary effect that is neutral-to-slightly-negative on tail molecules, with tail
behaviour noisy across splits; it is not a tail fix.

**Q5. What is the incremental value of J-reset (junction pretraining initialisation)?**
`J_RESET_NEUTRAL_VALID` — no measurable incremental value; if anything negative. J0 0.092369,
J1 0.089962, seed-mean 0.091165 vs P 0.083549: J−P **+0.007616** (CI95 [−0.000202, +0.019367],
per-seed +0.007584/+0.007649, better fraction 0.468). The CI marginally includes zero, so the
pre-registered verdict is NEUTRAL, but the point estimate is worse at both seeds, driven by
the tail (J−P tail +0.1115; ordinary +0.0004; top-10-removed +0.0027). J is still better than
scratch B (−0.008037 [−0.017354, +0.005552]), consistent with the atom/bond pretraining
carrying the gain. Conclusion: restoring the junction encoder to scratch destroys part of the
joint pretrained advantage on the tail; the complete pretrained encoder state is retained in
all future recipes, and J-reset is not a route.

**Q6. Next best experiment toward 0.06–0.07?** Pre-registered *before* the test read in
`next_predictive_hypothesis.md`: **Shared-Type Residual Composition (STRC)** — keep the full
pretrained BJOC-Null forward and the frozen P recipe, and add a zero-initialised, ≤ ~500
parameter shared per-atom-type (plus a small frozen junction-context vocabulary) additive
residual channel `ŷ = f_θ(x) + s1·Σ α[type_i] + s2·Σ β[context_j]`, trained jointly under the
same raw-y L1. Rationale from Phase B: the residual failure is magnitude shrinkage on rare
label-free structural contexts while the same model fits 716 train-tail molecules well, i.e.
a sample-sharing/statistics problem, not capacity; a shared type table gives every rare type
the sample size of all its occurrences. This is neither capacity (≤0.1% new parameters) nor
output calibration (no `a(counts)·ŷ + b(counts)`; the term uses raw type counts only), and no
tail-weighted loss/target transform/extra label is used. CPU pre-check: zero-init identity
(forward equals P exactly), gradient activity, type-count linearity, and a synthetic
group-contribution identifiability test. First GPU design: ≤2 seeds, one A100 per seed,
matched recipe; **purchase gate** seed-mean official-valid MAE ≤ P − 0.004 with both seed
deltas < 0, ordinary ≤ 0, tail < 0, RMSE ≤ P + 0.005 and stable α across seeds; gate failure
stops the route permanently. The terminal-test result did not choose this direction.

## 3. Discipline and provenance

* Test predictions were written **label-blind** behind `LabelGuard`; the label-storage
  sentinel check (labels temporarily zeroed / set to 999) leaves the object hashes unchanged
  (`330f636e…` before/after), so molecule objects cannot see labels. Relabelling and batch
  composition invariance hold at float32 epsilon (≤4.8e-7), checkpoint replay is exact (0.0).
  Freeze: combined prediction hash `5925b7eea0a317a1…`, test-object hash `6b0fd9e0d6bf2fa1…`,
  row-order hash `702746827e553786…`, structure fingerprint `75668e7838f6efcd…`, frozen in
  `terminal_test_freeze.json` before the read.
* The single read is recorded in `test_read.lock` (2026-10-13T01:01:33Z, y sha `3afb4be3…`,
  cross-check vs the historical KSVD labels max abs 0.0). No training, selection, calibration
  or architecture change happened after the read, and none will.
* Budget: J-reset 2 seeds × 240 epochs on one A100 each (wall 528 s / 463 s, ≈0.28–0.31
  GPU-h of the 0.8 cap); everything else CPU. Wall clock ≈25 min of the 6.5 h budget.
* Replay: frozen B/P checkpoints re-evaluated on official valid are bit-identical to last
  round (`REPLAY_CONFIRMED`, maxdiff 0.0 → B0 0.100172, B1 0.098232, P0 0.084786, P1 0.082313).

## 4. Deliverables

`model_reuse_audit.json`, `pretraining_legality_review.json`, `mechanism_evidence.json`,
`train_valid_tail_analysis.json`, `junction_reset_gate.json`, `train_J_seed{0,1}.json`,
`soup_J_seed{0,1}.pt`, `junction_reset_valid_metrics.json`, `jreset_prediction_freeze.json`,
`next_predictive_hypothesis.md`, `terminal_test_protocol.json`, `terminal_test_freeze.json`,
`test_read.lock`, `official_test_metrics.json`, `official_test_paired_bootstrap.json`,
`valid_test_comparison.json`, `scientific_decision.json`, `test_y_read_once.npy`,
`predictions/test_{B0,B1,P0,P1}.npz`, `analysis/*.npz` (replay predictions, train predictions).

Decision: **`PRETRAINING_TEST_GAIN_REPLICATES`** on the exposed official test; **target band
0.06–0.07 not reached** (gap 0.003957); J-reset **`J_RESET_NEUTRAL_VALID`**; next route STRC
awaits authorization.
