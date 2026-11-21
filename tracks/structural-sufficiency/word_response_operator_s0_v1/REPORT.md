# REPORT — word response operator, s0 (ogbg-mollipo, seed 0)

Protocol: `PROTOCOL.md`, config `configs/wo_s0.json`, protocol id
`word-response-operator-s0-v1`. Final commit: the commit that adds this file
(`analysis/run_manifest.json` holds per-unit provenance). Decision: **RED →
`NO_EVIDENCE_AT_THIS_PILOT`**. H holdout was enabled and evaluated once after the
candidate list was frozen.

## Direct answers

1. **Scores (official Valid RMSE/MAE, raw label units, one seed).**
   FULL 0.968 / 0.715 · LOCAL 0.976 / 0.724 · ATOMIC 1.002 / 0.748 ·
   ADD_WORD 1.073 / 0.810 · FP_KRR 0.875 / 0.687 · GINE_RAW 0.635 / 0.481.
   The new operator does **not** beat the scalar-word reference, the independent
   per-word variant, or the atomic control by the pre-registered margin: it beats
   ADD_WORD (+0.105) and ATOMIC (+0.0335, both > delta 0.0182) but only edges
   LOCAL by +0.0079 (< delta), and it is 0.334 behind the best strong reference
   (band 0.0608). Neither GREEN nor YELLOW: **RED**.
2. **Where does the (small) word-operator advantage come from?** ATOMIC is
   parameter-matched to FULL (13 817 vs 13 621, +1.4%), so the +0.0335 over the
   atomic response is not a parameter-count effect. But LOCAL is the same model
   with the cross-word off-diagonals removed and is statistically indistinguishable
   from FULL (0.0079, inside delta), so the gain cannot be attributed to
   *composing* boundary functions through real ports; it comes from word-level
   factorisation and word-specific parameters (θ, ψ, PairMLP), which LOCAL also
   has. The large distance to KRR/GINE (0.13–0.33 RMSE) is a property of the
   family and the linear readout (`ŷ = β0+βᵀz+Σα_hF_h`), not of the word
   partition: Train RMSE stops at 0.72 (FULL) while GINE reaches 0.18.
3. **Additional value of the multi-port function: no evidence.** FULL ≈ LOCAL on
   Valid and on the H pair deltas (FULL 0.463 vs LOCAL 0.451 MAE, i.e. LOCAL is
   marginally better). The conditional PORT_DIAG arm was never unlocked because
   the GREEN gate did not trigger. Whether the AMIGA/graph-kernel/QEq difference
   enters a *useful* computation remains untested here — the exportable
   `(c,S,h)` objects are correct but their coupling does not change predictions
   measurably.
4. **H label holdout.** 629 molecules, 47 components, 105 frozen evaluable pairs,
   58 transforms. Whole-molecule RMSE: GINE 0.593, FP_KRR 0.765, FULL 0.837,
   LOCAL 0.845, ATOMIC 0.872, ADD 0.931. Pair deltas (Δpred vs Δy): GINE
   MAE 0.300 / r 0.833 / direction 0.857; LOCAL 0.451 / 0.623 / 0.790; FULL
   0.463 / 0.597 / 0.771; ATOMIC 0.477 / 0.574 / 0.762. Per-transform TABLE:
   GINE 0.272, LOCAL 0.389, FULL 0.398, ATOMIC 0.408. The operator beats ADD_WORD
   and ATOMIC on whole-molecule error and is behind both strong references; on
   *pair deltas* it is slightly behind LOCAL. 500 component-bootstrap replicates
   (FULL 0.463 [0.383, 0.537]) are reported as a spread over components, not as a
   mechanism probability.
5. **Next step: stop this version.** No pre-registered condition for extending
   (seeds, channels, vocabulary, non-Gaussian family) is met. The limiting factor
   identified here is the response family + linear readout, not the word
   partition; a future round would have to change the function family or the
   readout before spending more compute on word discovery.

## Main table (Valid, best-Valid checkpoint)

| arm | params | epochs (best) | RMSE | MAE | wall s (1 GPU) |
|---|---:|---:|---:|---:|---:|
| FULL_WORD_OP | 13 621 | 80 (63) | **0.9682** | 0.7148 | 691 |
| LOCAL_WORD_OP | 13 621 | 80 (58) | 0.9760 | 0.7243 | 230 |
| ATOMIC_OP_CAP (w=96) | 13 817 | 80 (51) | 1.0017 | 0.7478 | 627 |
| ADD_WORD | 188 coefs | – | 1.0733 | 0.8098 | <1 (CPU) |
| FP_KRR | 2 731 duals | – | 0.8750 | 0.6873 | 4 (CPU) |
| GINE_RAW_REF | 387 974 | 240 (143) | 0.6346 | 0.4814 | 205 |

delta = 0.015·1.2159 = 0.0182, band = 0.05·1.2159 = 0.0608; `min(KRR,GINE)+band`
= 0.695. FULL−LOCAL +0.0079, FULL−ATOMIC +0.0335, FULL−ADD_WORD +0.105,
FULL−best strong +0.334.

All operator arms are the frozen recipe (seed 0, batch 64, AdamW 2e-3/1e-4,
5-epoch warmup, cosine to 2e-5, clip 5, FP32, TF32 off), 80 epochs because the
frozen timing rule projected 42.5 min for 160 epochs (> 18 min). FULL and LOCAL
started from a bit-identical state hash (`a2b564ddb3b7…`); Cholesky jitter count
was 0 in every run. GINE kept its own 240-epoch recipe and is not a
compute-matched comparison.

## Structure and support (label-free)

* 148 frozen word types, 6-atom maximum, all 9 atom / 3 bond columns, salts and
  components kept; replay of the frozen partition reproduces the previous round's
  `word_tmpl`, `word_atoms`, `word_n_atoms`, `word_bonds` and pointers exactly
  for all 3 360 Train + 420 Valid molecules.
* Distinct types present: Fit 144, Valid 137, H 139. Unknown-type instance rate
  0.0 in all three splits (the vocabulary saw all official Train structures, so
  this is not a generalisation statement).
* Boundary atom share 0.517 (Fit) / 0.523 (Valid) / 0.521 (H); crossing bonds per
  molecule 1–10+. Word sizes 1–6, ports 0–5.
* Fit/H core overlap: 576 Fit core keys, evaluable H pairs exclude all of them by
  construction. Murcko: 188 of 424 H scaffolds also occur in Fit (52.9 % of H
  molecules) — the split is a **pair-component** holdout, not a scaffold
  holdout.
* Fit-only scaler mean/std 2.1624/1.2159, `n0` = 27.06, `s_w = sqrt(df/(df+20))`;
  188 count columns kept after zero-variance removal.

## Engineering and algebra checks (`analysis/checks.json`, all pass)

* 16 representative molecules (no-boundary, one-port, multi-port, multi-component,
  many-crossing-bond): maps, 9/3 columns, internal + crossing bonds, batch
  offsets, and every crossing bond lands on its recorded boundary slot.
* FULL atom-space solve vs per-word boundary assembly: FP64 max |Δ| 7.1e-15,
  FP32 3.8e-6 (LOCAL 3.6e-15/3.9e-6, PORT_DIAG 7.1e-15/3.6e-6).
* Batch split/reversal: 1.4e-17 (FP64).
* Merge two connected words before eliminating vs per-word elimination: 3.6e-15.
* Bit-identical init for FULL/LOCAL/PORT_DIAG; ATOMIC width 96 chosen from the
  frozen list before any metric (13 817 vs 13 621 params, +1.4%).
* 32 Fit molecules / 60 steps: loss 1.6555 → 1.0157, no NaN, finite gradients
  (θ 1.1e-3, ψ 1.9e-3, PairMLP 2.9e-2, α 1.57, β 63.4). Double-precision central
  finite differences: α rel 3.2e-11, PairMLP-bias 9.4e-8, θ 3.4e-3 (gradient
  2.3e-9), ψ 3.7e-4 (7.4e-10) — analytic and numeric agree to 1e-11 absolute.

## H holdout (single unlock, after `analysis/candidates_frozen.json`)

Model weights, prediction files and the decision were committed before
`unlock-h`; H labels were then read once. No training, selection or scaling had
touched them. 629 molecules / 47 components / 105 pairs / 58 transforms.

| arm | H RMSE | H MAE | pair ΔMAE | pair r | direction | TABLE MAE |
|---|---:|---:|---:|---:|---:|---:|
| GINE_RAW_REF | 0.593 | 0.427 | 0.300 | 0.833 | 0.857 | 0.272 |
| FP_KRR | 0.765 | 0.572 | – | – | – | – |
| FULL_WORD_OP | 0.837 | 0.649 | 0.463 | 0.597 | 0.771 | 0.398 |
| LOCAL_WORD_OP | 0.845 | 0.653 | 0.451 | 0.623 | 0.790 | 0.389 |
| ATOMIC_OP_CAP | 0.872 | 0.679 | 0.477 | 0.574 | 0.762 | 0.408 |
| ADD_WORD | 0.931 | 0.733 | – | – | – | – |

Note the H ranking flips relative to Valid for the references (FP_KRR degrades
from 0.875 to 0.765 relative to the operator arms — i.e. the operator family does
*not* transfer better), and the mean predicted pair delta is roughly twice the
true one (FULL 0.127 vs 0.058) for every operator arm: the model over-reacts on
novel cores.

## Structural cases (`analysis/cases.json`, channel 0)

Two multi-port Valid molecules chosen before any label was read (rows 3 and 5:
10 words / 11 crossing bonds, 8 words / 8 crossing bonds; predicted 0.687 vs
label 1.93, predicted −0.933 vs label −0.03) and the best / worst Valid molecule
for FULL-vs-LOCAL absolute error (rows 307 and 215). Each record carries the real
ports, per-word `(c,h,S)` and the FULL prediction. Example word: 2 atoms, 1 port,
c = 0.184, h = [−1.748], diag S = [0.900]. The `(c,S,h)` export reproduces the
whole-molecule prediction exactly (equivalence check above); this is functional
correctness, not task evidence.

## Deviations and incidents

1. **ATOMIC width error (recorded, resources counted).** The first ATOMIC run used
   the default width 32 (5 241 params, Valid 0.9938, 611 s). The frozen width is
   96; the run was repeated with 13 817 params (Valid 1.0017, 627 s) and only the
   width-96 run enters the table. `wo_run.py train --arm ATOMIC_OP_CAP` now pins
   the width to `analysis/width_selection.json`.
2. **Timing vs steady state.** The frozen timing measured 12.8 s/epoch for LOCAL
   (projection 42.5 min for 160 epochs) and selected 80 epochs; the actual runs
   were 2.9–8.6 s/epoch. The frozen rule was not re-run after seeing this; 80
   epochs stayed. Both operator arms were still improving slowly at 80 epochs
   (Valid 0.9682 @63, 0.9709 @80 for FULL), which makes the reported scores, if
   anything, pessimistic; per protocol no extra epochs were added.
3. **H label placeholder bug.** `data/wo_struct_h.npz` carries a zero-valued
   placeholder `labels` array. An early `holdout` would have scored against
   zeros; `load_all` now ignores it and only the one-shot `wo_h_labels.npz`
   counts. The first `refs --ref add/krr` H numbers (computed against zeros) were
   discarded and regenerated; the reported H numbers come from the regenerated
   files.
4. **Murcko statistics** were computed on the local CPU (RDKit is not installed on
   `res`); every other number is from `res`.
5. **Execution regime.** The four formal neural units ran on `res`
   (`a100-2`, 2×A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, one GPU per
   job). ADD_WORD and FP_KRR are deterministic CPU computations on the local
   machine (ridge solve, 2 731×2 731 Tanimoto kernel); they are algebraically
   exact on either machine and are not compared as a neural regime. Structure
   cache construction ran on both (`content_sha256` identical).
6. **Not tested.** Atom renumbering invariance was not tested as a separate probe
   (the construction is permutation-invariant by indexing, and batch split/merge
   invariance were tested); no PORT_DIAG run (gate not reached); no re-tokenisation
   test (explicitly out of scope).
7. **Resource accounting.** Formal units: FULL, LOCAL, ATOMIC, GINE = 4 of 7;
   plus timing (1 GPU unit, 142 s) and the mis-specified ATOMIC run. Total
   ≈ 0.70 GPU·h (2 506 s of GPU work), well inside 4 GPU·h and 4 h wall. No unit
   failed, nothing was cancelled.

## What would make this candidate worth another round

Only if a pre-registered change to the *response family or readout* (e.g. a
learned non-linear functional of the per-channel `F_h`, or more channels with
shared coupling) closed most of the 0.33 gap to GINE on Valid while LOCAL stayed
inside delta. Under the current family, more seeds, a wider vocabulary or extra
channels would only resample the same RED conclusion, so they are not proposed.
