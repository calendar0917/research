# cscl-bjoc-pretraining-mechanism-terminal-test-v1 — design / pre-registration note

Status: **COMPLETED 2026-10-13.**  Phase A/B/C and the J-reset ablation ran;
Phase D/E scored the frozen models on the exposed official test with a single
label read.  Decisions: `REPLAY_CONFIRMED`, `LEGALITY_CONFIRMED`,
`TAIL_GENERALISATION_GAP_SUPPORTED`, `PURCHASE_JRESET`,
`J_RESET_NEUTRAL_VALID`, `PRETRAINING_TEST_GAIN_REPLICATES`.
Official test: B seed-mean 0.094627, P seed-mean **0.073957**, P−B −0.020670
(CI95 [−0.027157, −0.014712]); target band 0.06–0.07 not reached (gap
0.003957).  Report: `results/bjoc_pretraining_mechanism_terminal_test_v1/REPORT.md`.
Protocol: `configs/bjoc_pretraining_mechanism_terminal_test_v1.yaml`.

## Questions

1. Why does masked-structure pretraining improve raw-y prediction, and which
   parts of the explanation are evidence-backed vs hypotheses? (Phase B)
2. Does the junction encoder's pretrained initialisation have incremental
   value?  (Phase C, J-reset)
3. Does the full-10k official-valid gain (`P−B = −0.015653`) transfer to the
   exposed official test?  (Phase D/E, terminal score of B0/B1/P0/P1)

## Phase A — frozen models and legality

* `reuse-audit`: all four soup checkpoints exist, hash-match
  `prediction_freeze.json`, are 413,505-parameter BJOC-Null models with no
  forbidden modules and no pretext heads; the CPU replay of the official valid
  predictions is bit-identical (`max|dpred| = 0.0`) and reproduces the
  historical seed-mean MAE `B 0.099202` / `P 0.083549` exactly.  B/P share the
  per-seed init, composer/reader init and batch order; P's encoder state is the
  frozen SSL state; the normaliser is fit on official train 10000 only.
* `legality`: current code hashes equal the frozen
  `training_manifest.json` code hashes (training code version correspondence);
  the inherited audits (`MASK_TOKEN_SLOTS_FREE`, `MASK_SWAP_INVARIANT`
  200/200 atom + 200/200 bond, `MASKED_REBUILD_LEAK_FREE` 100/100,
  correctness gates 10/10, consistency all-pass) hold; the pretraining
  function reads only `x/edge_index/edge_attr` of official train and the
  supervised y comes from official train raw y only.

## Phase B — mechanism evidence and tail diagnosis

* Evidence-backed: P improves 8k locked-dev and full10k official-valid raw-y
  MAE on both seeds, improves fit MAE, and survives the extra-update scratch
  control C on the 8k split; Q (encoder-LR 3e-4) is harmful.
* Interpretation: the SSL state is a better *initialisation* for the raw-y
  optimisation, not a frozen representation — fine-tuning moves the encoders
  strongly again.
* Not proven: which pretext object matters, any chemical identification of the
  representation, junction-initialisation necessity (Phase C), extreme-tail
  generalisation, robustness to a new molecule distribution.
* Tail diagnosis (frozen B/P, seed-mean of single-model metrics):
  train tail `y<=-3` n=716, `P−B = −0.005891`; valid tail n=65,
  `P−B = +0.009341`; ordinary molecules improve on both splits
  (−0.017318 train / −0.017391 valid).  The valid tail regression is
  concentrated in a few molecules (rows 753 and 172 carry +1.98 of the
  +0.607 total tail sum; tail delta excluding them is −0.0218).  Verdict
  `TAIL_GENERALISATION_GAP_SUPPORTED`: P fits train extremes but does not
  transfer to valid extremes.

## Phase C — the single development ablation (J-reset)

Gate `PURCHASE_JRESET`: scratch init reproducible and equal to the recorded
B/P init hash; SSL encoder files re-hash to the recorded state; the hybrid
differs from the P encoder state only inside the junction encoder (112,896
junction parameters; atom/bond differences = 0); composer/reader equal the P
init; 413,505 parameters; relabelling invariance exact; batch consistency
1e-7; junction sensitivity 0.48.  Estimated two-seed cost 0.31 GPU-h.

Recipe identical to P: official train 10000, raw-y L1, Adam 1e-3 / wd 1e-5 /
batch 128 / clip 5 / fp32 / 240 epochs / last-5 soup / same batch order.
Evaluated on official valid only; **never** brought to official test.

## Phase D/E — terminal exposed-test evaluation

Only B0/B1/P0/P1.  All predictions are written label-blind on CPU behind a
LabelGuard, with label-storage mutation, relabelling, batch-composition and
checkpoint-reload checks; hashes (checkpoints, test objects, row order,
predictions, code/config) are frozen in `terminal_test_freeze.json` before the
single official-test label read (`test_read.lock`).  Official test is an
exposed instrument; this is a terminal exposed-test comparison, not a first
blind test.

## Phase F — next predictive hypothesis

Drafted from Phase B/C before the test read.  The test outcome may re-scope
confidence but may not choose the next direction.

## Results (after the single test read)

* J-reset: J0 0.092369 / J1 0.089962, seed-mean 0.091165 vs P 0.083549;
  J−P +0.007616 (CI95 [−0.000202, +0.019367]), both seeds +0.0076, tail
  +0.111456, ordinary +0.000397, top-10-removed +0.002729.  Verdict
  `J_RESET_NEUTRAL_VALID` (directional evidence against scratch junction
  init on the tail; complete pretrained state retained).
* Terminal test (exposed, single read 2026-10-13T01:01:33Z, y cross-check 0.0):
  B0 0.094243 / B1 0.095010 (mean 0.094627); P0 0.074825 / P1 0.073088
  (mean 0.073957); P−B −0.020670 (CI95 [−0.027157, −0.014712]), ordinary
  −0.022785 (n=923), tail +0.004688 (n=77), RMSE −0.026519, without worst row
  −0.022160.  Decision `PRETRAINING_TEST_GAIN_REPLICATES`; the valid-tail
  penalty did not reproduce at the same magnitude (test tail +0.0047).
* Next hypothesis (drafted before the read): Shared-Type Residual Composition
  (STRC) in `next_predictive_hypothesis.md`; not selected by test results;
  awaits authorization.
