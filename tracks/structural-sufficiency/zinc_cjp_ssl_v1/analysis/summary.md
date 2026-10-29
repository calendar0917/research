# CJP-SSL v1 — valid-only summary

Primary metric: official valid 1,000 raw-y MAE (lower is better).
Official test was never loaded.

## Supervised arms

| arm | seed | best MAE | best ep | W20 | T100 | final | params | ordinary | tail | r2-unseen ord | -row172 | -A-worst10 | train@best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-REPLAY | 0 | 0.076116 | 419 | 0.077020 | 0.077050 | 0.077062 | 237519 | 0.064499 | 0.243227 | 0.134269 | 0.070165 | 0.064141 | 0.009362 |
| A-REPLAY | 1 | 0.082301 | 473 | 0.082456 | 0.082496 | 0.082495 | 237519 | 0.063508 | 0.352639 | 0.131125 | 0.070559 | 0.063842 | 0.004985 |
| MARG-SSL | 0 | 0.088419 | 442 | 0.089060 | 0.089093 | 0.089041 | 237519 | 0.064215 | 0.436581 | 0.135339 | 0.077750 | 0.071266 | 0.007308 |
| JOINT-SSL | 0 | 0.083514 | 505 | 0.083763 | 0.083804 | 0.083792 | 237519 | 0.066209 | 0.332450 | 0.145599 | 0.072517 | 0.065234 | 0.005171 |

## Paired contrasts (delta = first - second; negative = first better)

| contrast | seed | full delta | CI95 | ordinary | r2-unseen ord | -row172 | first-better frac | median delta |
|---|---|---|---|---|---|---|---|---|
| MARG-SSL - A-REPLAY | 0 | 0.012303 | [+0.001117, +0.026505] | -0.000284 | 0.001070 | 0.007585 | 0.491 | 0.000539 |
| MARG-SSL - A-REPLAY | mean | 0.012303 | [+0.001117, +0.026505] | -0.000284 | 0.001070 | 0.007585 | | |
| JOINT-SSL - MARG-SSL | 0 | -0.004905 | [-0.013398, +0.001987] | 0.001993 | 0.010260 | -0.005233 | 0.511 | -0.000605 |
| JOINT-SSL - MARG-SSL | mean | -0.004905 | [-0.013398, +0.001987] | 0.001993 | 0.010260 | -0.005233 | | |
| JOINT-SSL - A-REPLAY | 0 | 0.007398 | [-0.001981, +0.020554] | 0.001710 | 0.011330 | 0.002352 | 0.522 | -0.001895 |
| JOINT-SSL - A-REPLAY | mean | 0.007398 | [-0.001981, +0.020554] | 0.001710 | 0.011330 | 0.002352 | | |

## Pretext diagnostics

| run | updates | diag CE before | diag CE after | class prior CE | cond prior CE | joint interaction RMS | eff. rank before | eff. rank after | atom mask frac | bond mask frac | paths/mol | swap delta | shuffle delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MARG-SSL_s0 | 7560 | 4.069545 | 0.318218 | 1.610220 | 1.419779 | 4.382170 | 22.852 | 29.412 | 0.1294 | 0.1202 | 2.996 | 3.581393 | 8.211188 |
| JOINT-SSL_s0 | 7560 | 4.058932 | 0.297191 | 1.610220 | 1.419779 | 3.209279 | 22.852 | 29.491 | 0.1294 | 0.1202 | 2.996 | 3.859315 | 7.607191 |

## Same-samples check across arms

```json
{
 "0": {
  "masked_batch_sha256_equal": true,
  "first_batch_query_sha256_equal": true,
  "split_equal": true,
  "mask_coverage_equal": true
 }
}
```

