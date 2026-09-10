# Frozen conditional readout sufficiency audit (compact-v4-hinge, ZINC)

Status: **NO-GO** (screening). Stage 0 measurement repaired and gated; Stage 1
seed-0 primary screening is a pre-registered *clear negative*; the second
frozen backbone seed was **not** spent. No new backbone was trained and the
main model was not modified.

Code: `tracks/ksvd/experiments/luyin16/zinc_frozen_readout_sufficiency.py`
Tests: `tracks/ksvd/tests/test_frozen_state_export_v2.py`
Results: `tracks/ksvd/results/frozen_readout_sufficiency/`

This is a **representation / readout sufficiency diagnostic**, not an
architecture benchmark. It answers exactly one question:

> Given the already-frozen compact-v4-hinge states and the existing 302D graph
> vector R, is there extra *signed* predictive signal that R does not expose
> but a tiny permutation-invariant set readout can use?

---

## 1. Motivation

The compact-v4-hinge candidate pools each molecule into one fixed 302D graph
vector `R = [unary moments ; pair-bucket moments ; global encoder ; topology
hinge]` and a single MLP head. A moment map is not injective: two multisets
can share count / mean / second moment while differing in max, multimodality
or joint distribution. The question is whether that non-injectivity is
*task-relevant* — i.e. whether the frozen pre-pooling set states `{h'_i}` and
`{q_ij}` carry signed predictive information that `R` does not expose.

A sufficient condition for "no cheap post-state gain" is a **parameter-matched
control**: a residual head of equal size reading only `R` (B1) versus a
residual head reading `[R ; set summary]` (S). The mechanism metric is

    ΔR = MAE(B1) - MAE(S).

`Δ0 = MAE(B0) - MAE(S)` only says a residual head helped; `ΔR` says the
*frozen set states* added value beyond an equally sized readout of `R`.

Mathematical sanity example (motivation only, **not** task evidence): the
multisets `{1, 5, 5}` and `{0, 5, 6}` both have count 3, mean 11/3 and second
moment 17, but different max and support. So the moment map is not injective.
Task-relevant evidence in this audit comes only from the frozen conditional
adapter performance below.

## 2. Why previous state evidence needed correction

An external review of the OOF difficulty audit found two measurement bugs in
`zinc_oof_difficulty_audit._forward_capture_fold`:

1. **Pair grouping (Q1).** The code did
   `pair_ids = batch.pair_index[0]; pair_boundaries = np.flatnonzero(np.diff(pair_ids)) + 1`.
   `pair_index[0]` is the *source patch index within a molecule*, not a graph
   id. Because `pair_sources` is piecewise-constant increasing, `np.diff`
   creates one group per source value, so pairs were grouped by source node,
   never by molecule. Correct graph id is `batch.batch[batch.pair_index[0]]`.
2. **`head_input` (Q3).** The hook named `head_input` was registered on
   `model.head[0]` and captured its **output** (a 64D hidden activation), not
   the input. The true pre-head representation is the *input* to
   `model.head[0]`.

Neither bug affected model predictions; only derived per-molecule state
statistics were invalid. The impact record is in
`results/frozen_readout_sufficiency/legacy_evidence_impact_record.json`
(predictions / MAE / rarity / disagreement / ensemble = **unaffected**;
`state_head_input_norm` interpretation = **requires caveat**;
`state_pair_norm_*` / `n_pairs` / `states_only` predictor = **requires
re-check**). Historical results are not deleted or re-run wholesale.

## 3. Export repair

New export module + version `frozen_state_export_v2_corrected`
(`results/frozen_readout_sufficiency/state_exports/`). Per holdout molecule it
stores `{h'_i}`, `{q_ij}`, bucket ids `b_ij`, local pair endpoints, the true
pre-head `R`, and `yhat_0`. The cache fingerprint includes export version,
tokenizer version (`typed_tokenizer_v1_historical`, the historical performance
candidate), tokenizer fingerprint, config fingerprint, vocabulary fingerprint,
split fingerprint, checkpoint SHA-256, backbone seed and fold. `load_export`
refuses any legacy `export_version` (Test 9).

Verified shapes (Q4, re-confirmed from the real model, not hard-coded blindly):
`patch_hidden = 48`, `pair_hidden = 16`, 5 distance buckets, head hidden
`64 → 32`, so

    unary moments     2*48 + 1        =  97
    pair moments      5*(2*16 + 1)    = 165
    global encoder                     =  32
    topology hinge output              =   8
    pre-head R                         = 302
    model.head[0] output               =  64   (the old, mislabelled capture)

The corrected export is produced **only** from the frozen OOF checkpoints'
`state.pt`; no backbone is trained and official valid/test are never loaded
(only `_load_zinc(root, "train")`, for the permutation check).

## 4. Integrity gates (Gate 0, HARD GATE)

`export_integrity_report.json` — **all gates pass on all 5 folds** (Q2):

| gate | meaning | result |
|------|---------|--------|
| G0.1 pair graph grouping | expected `n_g(n_g-1)/2` == exported, per graph | PASS |
| G0.2 pair endpoint identity | source graph id == target graph id | PASS |
| G0.3 pair coverage | each unordered pair row consumed exactly once, no self-loops | PASS |
| G0.4 reconstruct R | moments from `h'_i`,`q_ij`,`b_ij` ≈ true R (max abs 9.2e-5) | PASS |
| G0.5 reconstruct prediction | reconstructed R through frozen head ≈ `yhat_0` (max abs 1.9e-6) | PASS |
| G0.6 batch invariance | batch 128 vs 97 + reversed order (R/ŷ/means; max 4.8e-7) | PASS |
| G0.7 node permutation | relabelled nodes: R max 2.9e-6, ŷ max 1.2e-7 | PASS |

Q5/Q6: yes — the exported set states reconstruct both `R` and the frozen
prediction to numerical tolerance. Gate-0 failure would have been a
measurement error, not a readout NO-GO, and would have STOPPED the audit.

## 5. OOF adapter protocol

For each outer fold the backbone never trained on the 2000 held-out
molecules. Those molecules are split by a **pre-registered, target-independent
molecule-ID hash** (`sha256("frozen-readout-sufficiency-v2-20260910|id")`,
sorted) into 1200 adapter-fit / 400 adapter-selection / 400
adapter-evaluation. The same split is used for every model and control, and
would be reused for the second backbone seed. Adapters never touch official
valid/test. Splits/thresholds: `fold_split_manifest.json`.

Frozen: tokenizer/vocab, standardizers, embeddings, patch/pair encoders,
centre update, global/topology encoders, graph head — all `eval()`.

## 6. Budgeted staging policy

* Stage 0 = export repair + Gate 0 only, no adapter training.
* Stage 1 = 1 frozen OOF backbone seed × 5 folds × 1 deterministic adapter
  init (10 tiny adapters, ~4.1k params each).
* Stage 2 = second frozen OOF backbone seed, **only if** Stage 1 advances.
* B2 = joint-structure control, **only if** Set beats B1 meaningfully.
* No 4-seed backbone training, no optimizer / width / LR sweeps.

Pre-registered decisions (Section 22–32 of the mandate) are implemented
verbatim in `_decide_stage1` / `_final_decision`.

## 7. R-only control (B1)

`R → 13 → 13 → 1` MLP, `yhat = yhat_0 + rho_R(R)`, 4135 params. Inputs
standardized on the adapter-fit 1200 only. Adam(1e-3), full-batch, 400 epochs,
best selection checkpoint (patience 50). Fixed for B1 and S alike.

## 8. Set adapter (S)

Shared `phi_h: 48 → 16 → 8`; shared-across-buckets `phi_q: 16 → 16 → 8`;
`s = [mean_i phi_h(h'_i) ; mean_{bucket b} phi_q(q_ij) for b=1..5]` (8 + 5×8 =
48); `rho: 350 → 8 → 1`; `yhat = yhat_0 + rho([R ; s])`, 4145 params. Empty
buckets contribute zero. No attention, no new graph features, no change to the
distance-bucket definition. Parameter match vs B1 = **0.24%**.

Viability (no v6-style collapse): gradients nonzero, output non-constant, and
prediction genuinely depends on the summary — min sensitivity
`max|f(R,s) − f(R,0)| = 0.096`, prediction std ≈ 1.7–1.9.

## 9. Stage 1 results

Backbone seed 0, adapter init 0. Evaluation MAE on each fold's 400
adapter-evaluation molecules:

| fold | B0 | B1 (R-only) | S (set) | Δ0 = B0−S | ΔR = B1−S |
| ---: | -: | --------: | ------: | --------: | --------: |
| 0 | 0.224365 | 0.217165 | 0.224556 | −0.000191 | −0.007391 |
| 1 | 0.159687 | 0.151446 | 0.154212 | +0.005475 | −0.002766 |
| 2 | 0.263058 | 0.267327 | 0.264346 | −0.001288 | +0.002981 |
| 3 | 0.153514 | 0.153222 | 0.151367 | +0.002147 | +0.001854 |
| 4 | 0.165176 | 0.163337 | 0.166962 | −0.001786 | −0.003625 |
| mean | 0.193160 | 0.190499 | 0.192289 | **+0.000871** | **−0.001789** |
| median | — | — | — | −0.000191 | −0.002766 |

`mean Δ0 = +0.00087`, `mean ΔR = −0.00179`; Δ0 positive in **2/5** folds, ΔR
positive in **2/5** folds. Stratified (by outer fold, molecule-level) paired
bootstrap over 2000 molecules:

* Δ0 mean +0.00087, 95% CI [−0.00277, +0.00450], P(>0) = 0.68
* ΔR mean −0.00179, 95% CI [−0.00507, +0.00153], P(>0) = 0.14 (Q9)

So a small nonlinear **R-only** residual head does help slightly
(MAE_B0 − MAE_B1 = **+0.00266** pooled), but the **set summary adds nothing
over it** — in fact the set adapter is worse than the equally sized R-only
head on the selection split in 4/5 folds, so the deficit is optimization /
generalization, not eval noise (Q7, Q8, Q10).

Convergence check: re-training fold 0 and fold 2 with 2000 epochs (patience
200) gives **bit-identical** selection and evaluation MAE, confirming the
negative is not an under-training artifact.

## 10. Replication decision

Stage 1 satisfies the pre-registered **CLEAR NO-GO**:

* `mean Δ0 = +0.00087 < +0.001`
* `mean ΔR = −0.00179 ≤ +0.001`
* Set does not improve on B1 in ≥4/5 folds (2/5)

(Q11) Therefore the second frozen backbone seed is **not** spent, Stage 1b
(extra adapter init) is not triggered, and B2 is not run. Q12 (second-seed
replication) is not applicable.

## 11. Stage 2 results if executed

Not executed (Stage 1 clear NO-GO). No `stage2_fold_results.csv` /
`pooled_backbone_results.csv` / `final_bootstrap.json` are produced.

## 12. Joint-structure control if executed

Not executed. B2 (per-channel marginal-preserving row shuffle + retrain) is
gated on Set already beating B1; it did not. No `joint_structure_control.csv`
/ Figure 3.

## 13. What is and is not proven

Proven (within this OOF setup):

* The corrected export is measurably correct (all Gate-0 checks pass, Q1–Q6).
* A tiny nonlinear residual head on the existing 302D `R` yields a small,
  fold-inconsistent pooled gain over the frozen model (≈ +0.0027 MAE).
* A parameter-matched readout that additionally sees the frozen
  `{h'_i}/{q_ij}` set states does **not** beat the R-only head (ΔR ≈ −0.0018,
  95% CI straddling 0 from below).

Not proven:

* That moment pooling is *sufficient* (a small set adapter could still be
  under-parameterized or wrongly parameterized).
* That the patch paradigm has reached its ceiling, or that higher-order
  (triplet / centre-incidence) structure is useless — this audit only rules
  out a **cheap low-capacity set-summary extension** at this seed and budget.
* That the frozen states contain *no* information (only that a 48D
  mean-pooled shared-MLP summary does not expose usable extra signed signal
  beyond `R` at matched capacity).

## 14. Final decision

**NO-GO — cheap post-state readout extension** (`final_decision.json`).

Claim (pre-registered NO-GO wording):

> No useful incremental signal was found from a low-capacity nonlinear set
> summary of the existing frozen v4 states beyond an equally sized readout of
> the existing graph vector.

This is a screening NO-GO; it does **not** license "pooling is sufficient".
Per the mandate, do **not** open gated pooling / DeepSets variants / max
pooling / Set Transformer / attention readout next. The next research question
becomes pair endpoint association / centre-incidence compression — but only
after a real task-relevant witness diagnostic.

---

## Core questions

| # | question | answer |
|---|----------|--------|
| Q1 | where was the OOF pair grouping wrong? | `np.diff(batch.pair_index[0])` groups by source patch index, not graph membership; fix = `batch.batch[batch.pair_index[0]]` |
| Q2 | does corrected grouping pass all coverage/invariance gates? | yes, 7/7 gate families pass on 5/5 folds |
| Q3 | what was the old `head_input`? | the 64D output of `model.head[0]` |
| Q4 | true pre-head R? | 302D = unary(97)+pair moments(165)+global(32)+topology(8) |
| Q5 | reconstruct R from exported states? | yes, max abs 9.2e-5 |
| Q6 | reconstruct prediction? | yes, max abs 1.9e-6 |
| Q7 | R-only residual gain? | MAE_B0−MAE_B1 mean +0.00266 (2/5 folds positive for Δ0) |
| Q8 | Set adapter gain? | MAE_B0−MAE_S mean +0.00087 (2/5) |
| Q9 | ΔR = MAE(B1)−MAE(S)? | **−0.00179** (95% CI [−0.00507, +0.00153]) |
| Q10 | consistent across 5 folds? | no — ΔR positive in 2/5 folds |
| Q11 | second backbone seed needed? | no — Stage 1 clear NO-GO |
| Q12 | did seed 2 replicate? | not run |
| Q13 | common-input bulk safe? | mixed (per-fold Set bulk Δ up to +0.0050); not the deciding gate |
| Q14 | B2 mechanism (marginal vs joint)? | not run (Set never beat B1) |
| Q15 | final verdict? | **NO-GO** |

## Outputs

`results/frozen_readout_sufficiency/`:
`checkpoint_inventory.json`, `export_integrity_report.json`,
`fold_split_manifest.json`, `stage_export.json`, `stage1_fold_results.csv`,
`stage1_bootstrap.json`, `stage1_decision.json`, `final_decision.json`,
`legacy_evidence_impact_record.json`, `state_exports/*.npz`,
`figures/figure1_per_fold_delta.png`, `figures/figure2_mae_comparison.png`.

Figures are limited to 2 (Figure 3 is B2-only and B2 was not run).
