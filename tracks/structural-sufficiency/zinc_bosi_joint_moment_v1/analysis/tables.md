# BOSI-JM v1 — frozen official-valid tables

written 2026-10-26T23:12:49+0800 · n_valid = 1000 · official test never loaded

## Main table (raw-y MAE)

| run | arm | seed | params | GPU-h | best_valid | best_ep | W20 | T100 | final_valid | train@best | train@final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_s0 | B | 0 | 473889 | 0.373 | 0.082405 | 439 | 0.083203 | 0.083241 | 0.083242 | 0.008536 | 0.000593 |
| JM_s0 | JM | 0 | 475425 | 0.548 | 0.084088 | 377 | 0.085327 | 0.085420 | 0.085394 | 0.019174 | 0.000476 |

## Paired contrasts (negative = first arm better)

| contrast | ckpt | ΔMAE | 95% CI | CI excludes 0 | first better frac | Δ drop-top10 | ordinary | tail | unseen | rings≥4 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| JM_s0 − B_s0 | best | +0.001683 | [-0.006670, +0.013801] | False | 0.504 | -0.002447 | -0.001518 | +0.047716 | -0.015418 | -0.011602 |
| JM_s0 − B_s0 | final | +0.002151 | [-0.005688, +0.013703] | False | 0.511 | -0.002293 | -0.003151 | +0.078417 | -0.010608 | -0.008913 |

## Group sizes

* `overall`: 1000
* `ordinary_y_gt_-3`: 935
* `tail_y_le_-3`: 65
* `radius2_all_supported`: 384
* `radius2_low_support`: 365
* `radius2_any_unseen`: 251
* `rings_ge4`: 213

## Decision (pre-registered)

```
{
  "rule": {
    "A": "JM_s0.best_valid <= 0.075",
    "B": "(JM_s0.best_valid - B_s0.best_valid) <= -0.003 and JM_s0.best_valid <= 0.078",
    "guards": [
      "ordinary group (y > -3) shows no clear regression",
      "W20 shows no large regression in the opposite direction"
    ]
  },
  "stage": "seed0",
  "JM_s0_best_valid": 0.08408815455710282,
  "B_s0_best_valid": 0.0824054759553983,
  "delta": 0.0016826786017045264,
  "condition_A": false,
  "condition_B": false,
  "ordinary_delta": -0.0015175277307470216,
  "W20_delta": 0.0021242533111944795,
  "guards_ok": false,
  "buy_seed1": false
}
```

