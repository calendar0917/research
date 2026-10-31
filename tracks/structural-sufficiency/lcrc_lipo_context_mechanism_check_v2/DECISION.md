# DECISION — `lcrc-lipo-context-mechanism-check-v2`

## Decisions

| dimension | decision |
| --- | --- |
| `PERFORMANCE_SIGNAL` | **`ABSENT`** |
| `ROLE_SPECIFICITY_SIGNAL` | **`ABSENT`** |

All Test420 metrics below are **`EXPOSED_TEST_EXPLORATORY`**. No blind-test or
confirmatory claim is made.

## Proposition 1 — repeatable predictive value of context conditioning

The frozen rule requires all four BASE→CONTEXT RMSE gains
(`RMSE(BASE) − RMSE(CONTEXT)`, positive = CONTEXT better) to be strictly
positive: seed 0 and seed 1 on both Validation420 and Test420.

| seed | split | gain | 95 % paired-bootstrap interval |
| ---: | --- | ---: | --- |
| 0 | Validation420 | +0.011829 | [−0.016638, +0.012054, +0.039801] |
| 0 | Test420 `EXPOSED_TEST_EXPLORATORY` | +0.032256 | [+0.001666, +0.031715, +0.065042] |
| 1 | Validation420 | +0.005858 | [−0.025003, +0.005304, +0.037265] |
| 1 | Test420 `EXPOSED_TEST_EXPLORATORY` | −0.002741 | [−0.028259, −0.002463, +0.021262] |

Only 3 of 4 cells are positive; seed 1 reverses sign between Validation and
Test. The rule therefore returns **`ABSENT`**.

The only interval excluding zero is seed 0 on exposed Test. It is not
repeatable: seed 1's Test gain is negative, and the seed-to-seed Test-gain
change (`0.034997`) is larger in magnitude than seed 0's apparent Test gain
(`0.032256`). The seed-1 Validation gain is also tail-carried (non-tail 415
RMSE `0.543590 → 0.545203`, gain `−0.001613`). Molecule bootstrap intervals
describe within-model sampling, not training-seed variance; with two seeds the
latter is only described, not estimated.

**Conclusion:** no repeatable predictive value of CONTEXT conditioning is
established under this protocol.

## Proposition 2 — chemical-role grouping vs whole-graph context

`ROLEBLIND_s0` is the parameter-matched control (`m_all` = per-molecule
all-atom mean; `m_het = m_ch = m_all`; same 255,458 parameters and bit-identical
initialisation as CONTEXT_s0). Positive `ROLEBLIND advantage` below is
`RMSE(CONTEXT) − RMSE(ROLEBLIND)`.

| split | CONTEXT RMSE | ROLEBLIND RMSE | ROLEBLIND advantage | interval |
| --- | ---: | ---: | ---: | --- |
| Validation420 | 0.568587 | 0.534232 | +0.034355 | [+0.006146, +0.034563, +0.062389] |
| Test420 `EXPOSED_TEST_EXPLORATORY` | 0.544731 | 0.537335 | +0.007396 | [−0.015334, +0.007752, +0.029403] |

ROLEBLIND is better on both splits in point estimate; the Validation interval
is entirely in its favour, and the Test interval contains zero but its median
also favours ROLEBLIND. CONTEXT never beats ROLEBLIND. The rule requires both
CONTEXT-better intervals to be entirely above zero, so the result is
**`ABSENT`**.

The CONTEXT gain therefore cannot be attributed to chemical-role grouping.
The all-atom mean is not information-matched (it also encodes hetero-atom
fraction/composition), so this experiment establishes lack of role-specificity
support, not why the all-atom summary performs better. This conclusion is
independent of the performance decision.

## Scope

* Two training seeds (0, 1), one frozen random42 split, Validation420 and
  exposed Test420.
* Three new one-shot trainings: `BASE_s1`, `CONTEXT_s1`, `ROLEBLIND_s0`;
  no retraining of historical `BASE_s0`/`CONTEXT_s0` or PNA.
* No early stopping, no Validation-based checkpoint selection, last-five-epoch
  soup, no tuning after Test/Validation access.
* Total GPU spend 0.3125 / 0.45 A100 GPU·h (both pools summed); no failures.
* No follow-up experiment or new model is proposed. All Test metrics remain
  `EXPOSED_TEST_EXPLORATORY`.
