# Phase 1b corrigendum — interpretation limits of the frozen result

Status: **new corrigendum, written before Phase 1c GPU results**  
Date: 2026-10-19  
Scope: interpretation only. The frozen Phase 1b protocol, run JSONs,
`results/phase1b/REPORT.md`, `analysis_phase1b.json` and `run_manifest.tsv` are
**not** rewritten. Where this file conflicts with the Phase 1b report's prose,
this file is authoritative for Phase 1c interpretation.

Upstream: `notes/phase1b_protocol.md` (sha256 `25bfff0c…`),
`code/phase1b/{ssrc_models,ssrc_rewire,ssrc_train}.py`,
`results/phase1b/REPORT.md`, `results/phase1b/decision_log.md`.

---

## 1. `R1 -> M1` changed two things, not one

`R1` and `M1` do **not** differ only by a node-state update.

- `R1` readout: `concat(mean node token, mean unique-edge relation)`.
- `M1` readout: mean over the **updated node states** only; the direct edge
  readout is gone.

Therefore `Gain_update = MAE(R1) - MAE(M1)` mixes:

1. the effect of one node-state update, and
2. the replacement of R1's direct edge readout by a pooled-node readout.

The Phase 1b report's sentence that this is the effect of "the update, not
iteration" is too strong. `M1` is a different readout form, not just `R1` plus
an update. Phase 1c does not repeat this contrast as a mechanism test. Any
future update test must keep the R1 direct edge readout (`M_keep_edge`), which
is why that arm is specified but not automatically bought.

## 2. `M1 -> M2` is parameter-identical but not compute-identical

`M2` reuses the exact `M1` parameters and applies the tied relation/update
module twice per forward pass. It is parameter-identical, but it is **not**
training-compute-identical:

- `M1` at 300 epochs was undertrained (best epoch 290/300; no early stop).
- `M2` received two relation applications per epoch.
- The Phase A `M1 -> M2` gain (+0.0234 ZINC) is therefore not a clean
  iteration effect.

The B2 control (`M1_long`, 285 epochs, 713.8 GPU-s vs M2 705.6 GPU-s) closed
the gap: `M1_long` 0.2495 vs `M2` 0.2522, Δ = −0.0027, CI crossing zero.
This is evidence that **this particular second-round gain is not separated
from optimization/compute budget**.

## 3. `M1_long ≈ M2` is a local null, not a general claim

The matched-compute result says:

> At this scale (ZINC/Lipo), this recipe, this depth, and one extra tied
> round, the observed M2 advantage is not distinguishable from a longer
> optimization schedule.

It does **not** say:

- message passing is unnecessary in general;
- iteration can never help at larger depth, larger data, or on long-range
  tasks;
- all iterative relational computation is useless.

The Phase 1b report's headline "iteration is not the mechanism" should be read
as "the second tied round is not separated from compute in this harness". The
Phase 1c report repeats this narrower wording.

## 4. `G_ref` is a six-layer local GINE, not a global-attention model

`G_ref` is `ModelGRef` in `code/phase1b/ssrc_models.py`:

- 6 residual `GINEConv` layers on the raw atom graph,
- hidden 128, FFN 128, dropout 0.1,
- readout `concat(sum node, mean node)`.

It has a **6-hop local receptive field**, not a global attention token and not
an all-pairs interaction. The ZINC gap between the patch arms and `G_ref` is
therefore confounded with:

- depth / receptive field,
- parameter count (439,559 vs 94,659 for R1),
- readout form (sum+mean vs mean-only),
- optimization of a larger model.

It is **not** evidence that global attention is the missing ingredient, and it
is not a GraphGPS/GRIT/ESA+ comparison.

## 5. `S0/R1` and `G_ref` use different graph readouts

- `S0`: mean over patch tokens.
- `R1`: `concat(mean node token, mean unique-edge relation)`; both are means.
- `G_ref`: `concat(sum node, mean node)`.

A mean-only readout is invariant to token multiplicity: given the same patch
*mixture*, a molecule with twice as many identical patches gets the same
representation. `G_ref`'s sum channel is not. This is a genuine readout-class
difference and a candidate explanation for part of the ZINC gap. Phase 1b did
not control it. Phase 1c's `S_cap_size` and `R1_size` add node `sum`, node
`mean` and explicit size features under a matched-capacity readout.

## 6. The Lipo task difference is exploratory

Lipo shows no meaningful relation gain (S0−R1 = +0.0089, CI crosses zero,
below the negligible band) while ZINC does (+0.0571 seed 0, +0.0992 seed 1).
With one split per task, a different target scale, a different molecule-size
regime, and a different split construction, this is a **clue**, not a
confirmed task law. It does not establish "local structure suffices on Lipo"
or "relations matter on ZINC" as general statements.

## 7. What Phase 1b did establish, narrowly

Within the frozen harness and the internal train-only Monitor:

1. A one-shot, endpoint-swap-invariant relation on true root edges improved
   over the S0 local-patch mean-readout model on ZINC (two seeds, positive
   paired sign; seed-0 CI excluded zero).
2. A degree/type/count-preserving broken adjacency was worse than true R1 by
   +0.0706 and worse than S0. This isolates **correspondence given the R1
   architecture**; it does not isolate correspondence from R1's extra capacity,
   because the broken control keeps R1's 94,659 parameters.
3. The extra tied round's apparent gain did not survive the matched-compute
   B2 control.
4. The ZINC gap to the 6-layer GINE reference is real and large; the Lipo gap
   is absent. Both are internal Monitor statements, not official-test numbers.

## 8. What Phase 1b did not establish

- That R1's gain is due to relational information rather than +66% parameters.
- That R1's gain is due to endpoint pairing rather than edge-type statistics,
  extra depth, or a different readout.
- That the ZINC gap is due to missing global attention, missing sum readout,
  or missing depth individually.
- That Lipo and ZINC have a task-level difference in the value of relations.
- That `E_2` is task-information sufficient or insufficient in the Bayes-risk
  sense.

Phase 1c is designed to attack the first three confounds, not to confirm the
Phase 1b route by adding modules.
