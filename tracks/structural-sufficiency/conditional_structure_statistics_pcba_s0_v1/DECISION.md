# DECISION — css-pcba-s0-v1

**Round**: conditional structure statistics (per-occurrence vs post-aggregation task
conditioning), `ogbg-molpcba`, seed 0.
**Verdict: STOP.** Do not continue this architecture line as a performance direction;
the hypothesis it was built to test is **not supported** on the tasks where it can be
measured. The round is closed with a negative/mixed result, S not purchased, and
≈ 6.7 GPU-hours of the hard budget left deliberately unused.

All Test numbers are `DEVELOPMENT_TEST`.

## The decision rule I applied in advance

The round existed to answer one comparison: E (condition every occurrence) vs L
(condition once after aggregation), identical in parameters (831,088/831,088), initial
state, batch order, evaluation and 40-epoch budget. S (EARLY_SHARED) was to be bought
only if the frozen rule fired: `d40 ≥ 0.002 or Val_E(best) ≥ 0.268273`, **and**
`Val_E(best) ≥ 0.262273`, **and** the S cost fits the GPU budget.

## What happened

| | E | L |
| --- | --- | --- |
| best Val AP | 0.246329 (ep 36) | 0.242804 (ep 31) |
| Test AP at that state | 0.240453 | 0.232267 |
| epoch-40 Val / Test | 0.245035 / 0.240328 | 0.240905 / 0.232187 |

* Aggregate difference: **+0.0035 Val / +0.0082 Test** in favour of E, positive in all
  four comparisons.
* Well-supported tasks only (≥ 20 positives in Val and Test): **−0.0035 Val**
  (p = 0.081) and **+0.0002 Test** (interval [−0.0038, +0.0044]). The aggregate
  advantage is produced by ~19 sparse tasks with 4–30 positives, where AP moves by
  0.05–0.34 for a single mis-ranked molecule.
* Per-task sign counts are 61/66 (Val) and 62/65 (Test): no consistent per-task gain,
  median Δ ≈ −0.0005.
* Both arms sit ≈ 0.013 Test AP below the historical W2 anchor (0.253617).

## Why this is a STOP and not a "promising direction"

1. **The effect is not where the measurement is reliable.** A +0.008 Test macro AP that
   becomes +0.0002 when the sparse tail is removed is not evidence for a mechanism; it
   is evidence that AP on tasks with four positives is nearly a coin flip.
2. **No replicated resolution.** One seed, one budget, two arms of identical capacity.
   A single-seed difference of 0.0035 Val AP between identical-capacity models is
   inside what a re-run could plausibly reverse; nothing here justifies buying a second
   seed to find out.
3. **The subject is not competitive.** Even granting the aggregate sign, the model is
   0.013 Test AP behind the historical W2 subject. Improving a mechanism question on an
   architecture that is behind the existing baseline is not a good use of the next
   10 GPU-hours.
4. **The mechanism is confirmed, the benefit is not.** The query is really used and
   really task-specific (permuting task-query rows changes predictions; removing the
   molecular context collapses the gate spread from 0.26 to 0.16 and returns the gate
   mean to ≈ 1), and E is ~1.8× more task-query-sensitive than L. So the negative
   result is informative: making the conditioning more expressive changes what the
   model computes without making it rank better.
5. **S is not worth its 4.6 GPU-hours.** The trigger fired (`d40 = +0.0041`) but the
   floor did not (`0.2463 < 0.2623`). Running the shared-query ablation now would tell
   us how much of a non-effect the task query explains.

## What would have changed the decision (recorded in advance, not hindsight)

* `Val_E(best) ≥ 0.2623` (near the historical anchor) **and** a positive E−L difference
  on the well-supported tasks → S purchased, and a second seed proposed in a new round.
* A well-supported-task difference ≥ +0.005 Val with the same sign on Test → the
  hypothesis would be alive regardless of the absolute level.
* Neither happened.

## What is worth keeping from this round

* A working, gated, cheap harness for per-occurrence structure statistics
  (S1 directed occurrences + full non-backtracking S2 index, per-occurrence gating,
  root statistics readout), with 14 correctness gates and a 4.6-GPU-hour 40-epoch run.
* The measured fact that per-occurrence conditioning *changes* the computation
  substantially (1.8× task-query sensitivity, gate spread 0.26 vs 0.25 with different
  structure) **without** changing ranking quality on well-supported tasks.
* A reusable negative control for future "condition earlier" claims on PCBA: any
  claimed gain must be shown on the ≥ 20-positive subset, not on the macro average.

## Explicitly not done

No second seed, no longer horizon, no hyper-parameter search, no ZINC/MolHIV/PCQM
transfer, no new descriptors, no distillation from LCRC outputs, no retraining of the
historical baselines, no S arm, no test-set selection of any kind. Idle budget was left
idle.
