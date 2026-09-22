# DTX-v0 -- no-ring dictionary x generic topology cross: results summary

Question: can an **aligned** joint statistic of learned dictionary assignments and *generic, untyped, label-free* topology roles beat a matched marginal-only control, without ever reading an explicit ring / cycle context?

Official ZINC **test was never loaded** in any stage.

## Primary metric — fixed Top-5 weight soup (official-valid MAE)

| arm | best MAE | best epoch | soup MAE | soup members | params | wall s |
|---|---|---|---|---|---|---|
| MARGINAL (control) | 0.148671 | 238 | 0.144330 | [232, 233, 235, 236, 238] | 60442 | 1353.4 |
| ALIGNED (cross) | 0.154001 | 232 | 0.150357 | [216, 224, 227, 231, 232] | 60442 | 1673.0 |

## Outcome

* Delta_align (soup) = -0.006027
* Delta_align (best) = -0.005330
* case: **D**
* verdict: **NO_ALIGNED_CROSS_SIGNAL**
* base regression confound (M soup > 0.145): False
* seed 1 authorized: False

## Alignment-use interventions (Arm A)

| intervention | mean |dpred| | max |dpred| | valid MAE | clear |
|---|---|---|---|---|
| alignment_removal | 0.064944 | 0.326489 | 0.168766 | True |
| topology_shuffle | 0.087158 | 0.480084 | 0.184156 | True |
* repeat-forward noise floor (max |dpred|): 0.000e+00

## Dictionary / topology diagnostics

* MARGINAL: effective atoms 2.47 / argmax-used 51 / top-8 mass 0.979 / tau 0.0823
* ALIGNED: effective atoms 2.88 / argmax-used 41 / top-8 mass 0.969 / tau 0.1029

## Budget

* full training runs used: 2 / 2
* seed 1 purchased: False
* sweeps / HPO: none
* official test accessed = false

