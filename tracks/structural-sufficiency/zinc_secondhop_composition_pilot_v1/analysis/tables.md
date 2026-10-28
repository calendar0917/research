# Second-hop composition pilot — valid tables

## Stage 1 (seed 0)

| model | full | drop row172 | drop worst10 | ordinary | r2-unseen ord | seen ord | tail | best ep | W20 | T100 | train MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_s0 | 0.076582 | 0.070076 | 0.062517 | 0.063545 | 0.130915 | 0.042195 | 0.264113 | 476 | 0.077333 | 0.077359 | 0.005268 |
| Real_s0 | 0.078976 | 0.069955 | 0.062419 | 0.062500 | 0.125420 | 0.042561 | 0.315978 | 368 | 0.079500 | 0.079534 | 0.012544 |
| Null_s0 | 0.082897 | 0.072317 | 0.065149 | 0.063813 | 0.129254 | 0.043075 | 0.357420 | 522 | 0.083033 | 0.083056 | 0.002387 |

## Stage-1 paired contrasts (negative = first is better)

| contrast | full | drop row172 | drop worst10 | CI95 | win rate | max row share | r2-unseen ord | seen ord |
|---|---|---|---|---|---|---|---|---|
| Real_minus_Null | -0.003921 | -0.002362 | -0.002730 | [-0.010734, +0.002133] | 0.489 | 0.424 | -0.003834 | -0.000514 |
| Real_minus_A | +0.002395 | -0.000121 | -0.000099 | [-0.004085, +0.010414] | 0.519 | 1.051 | -0.005495 | +0.000366 |
| Null_minus_A | +0.006316 | +0.002241 | +0.002631 | [-0.001280, +0.017168] | 0.508 | 0.646 | -0.001661 | +0.000880 |

## Stage-1 purchase gate

* c0 no-break: False (max new full 0.082897 vs limit 0.081582)
* signal A (vs A): False []
* signal B (Real vs Null): True [{'variant': 'full', 'delta': -0.003920920737540398, 'delta_full': -0.003920920737540398}]
* signal C (r2-unseen ordinary): False (r2-unseen ord -0.003834, seen ord -0.000514)
* **buy stage 2: False**

## Mechanism (frozen best checkpoints)

| run | W_new/W_old | g_new/g_old @step1 | C-switch mean|Δpred| | C-switch ΔMAE | h PR |
|---|---|---|---|---|---|
| Real_s0 | 0.5892 | 0.1967 | 0.037056 | +0.013022 | 4.31 |
| Null_s0 | 0.6269 | 0.1949 | 0.032177 | +0.010173 | 4.99 |

Rows are the official valid split (1,000 molecules). `A_s0`/`A_s1` are the frozen LCRC-A predictions; the paired comparison uses the stored per-molecule predictions, not re-training.
