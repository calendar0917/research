# Frozen-model sensitivity tables

Per-molecule changes are in `pred_per_molecule.csv`; every legal prediction is included as JSON in the `*_reps_json` columns.

## Group summary

| group | n | n alt | model | mean max |Δ| | median max |Δ| | mean range | MAE orig | MAE all-rep mean | ΔMAE |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| all | 1000 | 827 | p3b | 0.028896 | 0.010373 | 0.029665 | 0.076224 | 0.080411 | +0.004187 |
| all | 1000 | 827 | t1 | 0.030246 | 0.010708 | 0.030877 | 0.079088 | 0.081918 | +0.002830 |
| seen_ordinary | 710 | 619 | p3b | 0.026606 | 0.013688 | 0.027324 | 0.047468 | 0.051123 | +0.003655 |
| seen_ordinary | 710 | 619 | t1 | 0.029854 | 0.014797 | 0.030331 | 0.050322 | 0.053514 | +0.003192 |
| seen_tail | 39 | 16 | p3b | 0.084410 | 0.000000 | 0.084410 | 0.116284 | 0.151853 | +0.035569 |
| seen_tail | 39 | 16 | t1 | 0.037208 | 0.000000 | 0.037316 | 0.172494 | 0.184378 | +0.011884 |
| unseen_ordinary | 225 | 181 | p3b | 0.026800 | 0.002595 | 0.027951 | 0.133940 | 0.135282 | +0.001342 |
| unseen_ordinary | 225 | 181 | t1 | 0.030537 | 0.003798 | 0.031745 | 0.132872 | 0.133904 | +0.001032 |
| unseen_tail | 26 | 11 | p3b | 0.026300 | 0.000000 | 0.026300 | 0.301922 | 0.298186 | -0.003736 |
| unseen_tail | 26 | 11 | t1 | 0.027997 | 0.000000 | 0.028644 | 0.259057 | 0.253985 | -0.005072 |

## Support-gain subgroups

| subgroup | n | model | mean max |Δ| | median max |Δ| | MAE orig | ΔMAE (all-rep mean) |
|---|---:|---|---:|---:|---:|---:|
| unseen_ordinary_all | 225 | p3b | 0.026800 | 0.002595 | 0.133940 | +0.001342 |
| unseen_ordinary_all | 225 | t1 | 0.030537 | 0.003798 | 0.132872 | +0.001032 |
| unseen_ordinary_rescued_any | 26 | p3b | 0.032344 | 0.022199 | 0.108837 | -0.000042 |
| unseen_ordinary_rescued_any | 26 | t1 | 0.051972 | 0.030018 | 0.105590 | -0.004916 |
| unseen_ordinary_no_rescue | 199 | p3b | 0.026075 | 0.000040 | 0.137219 | +0.001523 |
| unseen_ordinary_no_rescue | 199 | t1 | 0.027737 | 0.000592 | 0.136436 | +0.001809 |
| union_full_support_gain | 14 | p3b | 0.026202 | 0.020006 | 0.080877 | -0.001853 |
| union_full_support_gain | 14 | t1 | 0.039259 | 0.030105 | 0.073233 | -0.005604 |
| rep_full_support_gain | 9 | p3b | 0.034267 | 0.032296 | 0.064991 | -0.005098 |
| rep_full_support_gain | 9 | t1 | 0.051166 | 0.032348 | 0.063844 | -0.003777 |
| arom_full_support_gain | 12 | p3b | 0.030088 | 0.029035 | 0.080992 | -0.001051 |
| arom_full_support_gain | 12 | t1 | 0.048248 | 0.031733 | 0.077445 | -0.000193 |
| rescued_but_insensitive | 7 | p3b | 0.001418 | 0.000000 | 0.091355 | -0.000709 |
| rescued_but_insensitive | 7 | t1 | 0.000081 | 0.000078 | 0.077393 | -0.000008 |

## Error / sensitivity relation

| model | subset | n | mean |error| | mean max |Δ| | ratio | Pearson | n |Δ|>|err| |
|---|---|---:|---:|---:|---:|---:|---:|
| p3b | all | 1000 | 0.076224 | 0.028896 | 0.379 | 0.315 | 277 |
| p3b | unseen_ordinary | 225 | 0.133940 | 0.026800 | 0.200 | 0.116 | 41 |
| t1 | all | 1000 | 0.079088 | 0.030246 | 0.382 | 0.421 | 285 |
| t1 | unseen_ordinary | 225 | 0.132872 | 0.030537 | 0.230 | 0.179 | 39 |
