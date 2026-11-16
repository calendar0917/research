# DECISION — lcrc-readout-context-pilot-small-v1

Status: DEVELOPMENT feasibility pilot, one training seed, two small official tasks.
Test numbers are DEVELOPMENT_TEST.

## Recorded axes

| task | axis | label |
|---|---|---|
| ogbg-molesol | retention | PERFORMANCE_LOSS_THIS_RECIPE |
| ogbg-molesol | context | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-molesol | gine | WITHIN_5PCT_OF_THIS_GINE (A) |
| ogbg-molesol | trace | TRACE_PASS |
| ogbg-mollipo | retention | PRESERVED_POINT_ESTIMATE / HORIZON_CONSISTENT |
| ogbg-mollipo | context | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-mollipo | gine | BEHIND_THIS_GINE |
| ogbg-mollipo | trace | TRACE_PASS |

## Decision

1. **The explicit readout is implementable and traceable.** Every Val/Test prediction of arms B and C was rebuilt from exported per-molecule and per-root terms within the pre-declared tolerance, the fixed-input intervention was reproduced from the ledger, the `a_k/c_k` scale degeneracy was closed exactly, and the selected checkpoints replay bit-for-bit. This part does not depend on the numbers being good.
2. **The additive readout (C) is not performance-neutral against the unchanged C1 head (A)** on at least one task at the pre-declared 5% tolerance. In this recipe the C1 pooling head still buys measurable accuracy that additive roots + global single-variable terms do not recover.
3. **The context increment is not stable across surfaces.** C vs B changes sign between Val and Test and/or between the selected state and the last-20 horizon (ogbg-molesol: last20_test +21.35%, last20_valid -8.19%, selected_test +20.50%, selected_valid -8.67%; ogbg-mollipo: last20_test -2.06%, last20_valid +8.53%, selected_test -1.56%, selected_valid +8.20%). No context gain is claimed: at this scale the two frozen fields and this 2-factor functional form do not produce a reproducible improvement, even though the learned interaction responds measurably to the inputs (section 6).
4. **Arm(s) A land within the pre-declared 5% band of the frozen GINE** on both splits of ogbg-molesol. Because G uses a different input path and a stronger learning-rate preset, this is a distance marker, not an attribution, and it is not a competitiveness claim for the round.
5. **The additive arms are behind the frozen GINE reference on ogbg-mollipo**. With a different input path and a stronger learning-rate preset in G, this is again a distance marker, not an attribution.

## What happens next (and what does not)

* The round stops here: two tasks, one seed, no re-run with a friendlier configuration.
* A second seed or a wider readout is **not** justified by these numbers; a seed sweep would answer a
  different question (stability) and is out of scope for this pilot.
* If the project wants to pursue the context question, the next round must be declared in advance with
  its own protocol: e.g. a fixed-form interaction *and* a frozen-`h` baseline that separates 'the number
  went in' from 'the structure is read differently', on a task where a context field is known to matter.
* The reusable part of this round is the **machinery**: the shared-trunk construction, the budget-matched
  readouts, the ledger/intervention/scale-equivalence gates and the analysis driver.

Commit `8b6269877609` on `lcrc-readout-context-pilot-small-v1`; freeze hash `5a6d48aaef012aac…`.
