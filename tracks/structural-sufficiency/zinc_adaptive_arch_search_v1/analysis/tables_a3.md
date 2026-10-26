# AAS v1 stage a3 — official-valid tables

written 2026-10-27T03:01:29+0800 · n_valid=1000 · official test never loaded

## Main table (raw-y MAE)

| run | params | GPU-h | best_valid | best_ep | W20 | T100 | final_valid | train@best | train@final | median AE | P90 AE | top10 share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0M0_s0 | 488321 | 0.516 | 0.088275 | 338 | 0.091337 | 0.091625 | 0.091570 | 0.019148 | 0.000557 | 0.042086 | 0.163707 | 0.228 |
| A0M0_s1 | 488321 | 0.457 | 0.081282 | 414 | 0.082312 | 0.082405 | 0.082426 | 0.011377 | 0.000756 | 0.040720 | 0.148675 | 0.238 |
| A0_s0 | 488321 | 0.581 | 0.079656 | 390 | 0.081408 | 0.081465 | 0.081466 | 0.013494 | 0.000460 | 0.039326 | 0.148206 | 0.206 |
| A0_s1 | 488321 | 0.556 | 0.079361 | 469 | 0.080537 | 0.080584 | 0.080634 | 0.007544 | 0.000542 | 0.033321 | 0.151824 | 0.242 |
| B_s0 | 473889 | 0.373 | 0.082405 | 439 | 0.083203 | 0.083241 | 0.083242 | 0.008536 | 0.000593 | 0.037000 | 0.164440 | 0.234 |

## Paired contrasts (negative = first arm better)

| contrast | ckpt | ΔMAE | 95% CI | excludes 0 | first better | Δ drop-top10 | ordinary | tail | all_sup | low_sup | unseen | rings≥4 | worst10 | drop172 | drop top-1 contributor |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0M0_s0 − A0_s0 | best | +0.008619 | [+0.000576, +0.018718] | True | 0.449 | +0.005444 | +0.005468 | +0.053956 | +0.000931 | +0.018659 | +0.005782 | +0.016494 | +0.327309 | +0.004911 | +0.004911 (row 172) |
| A0M0_s0 − A0_s0 | final | +0.010104 | [+0.001882, +0.022087] | True | 0.427 | +0.005869 | +0.006312 | +0.064652 | +0.002277 | +0.020971 | +0.006278 | +0.012552 | +0.478909 | +0.005429 | +0.005429 (row 172) |
| A0M0_s1 − A0_s1 | best | +0.001921 | [-0.002504, +0.006267] | False | 0.461 | +0.001675 | +0.003015 | -0.013821 | +0.001000 | -0.000735 | +0.007192 | +0.002721 | -0.060811 | +0.002770 | +0.002770 (row 172) |
| A0M0_s1 − A0_s1 | final | +0.001792 | [-0.003016, +0.006477] | False | 0.456 | +0.000875 | +0.002410 | -0.007100 | +0.000494 | -0.002112 | +0.009455 | +0.003840 | +0.001254 | +0.002927 | +0.002927 (row 172) |

## Group sizes

* `overall`: 1000
* `ordinary_y_gt_-3`: 935
* `tail_y_le_-3`: 65
* `radius2_all_supported`: 384
* `radius2_low_support`: 365
* `radius2_any_unseen`: 251
* `rings_ge4`: 213
* `reference_worst10`: 10
* `reference_complement_990`: 990
* `drop_row172`: 999

## Decision (pre-registered)

```
{
  "rules": {
    "DEEPEN_A": "A1 - B_s0 <= -0.003 and A1 - A0 <= -0.001",
    "NONLINEAR_ONLY": "A1 - B_s0 <= -0.003 and |A1 - A0| < 0.001",
    "STOP_A": "A1 - B_s0 >= +0.005, or (W20 and T100 both >= B_s0 + 0.003)",
    "AMBIGUOUS": "otherwise: at most one confirmation purchase",
    "guard": "ordinary group (y > -3) must not regress by more than 0.002"
  },
  "paired_by_seed": {
    "A0_minus_B": {
      "0": {
        "delta_mae_best": -0.0027494255703641103,
        "ci95": [
          -0.013161499659717081,
          0.004401163101010023
        ],
        "excludes_zero": false,
        "delta_drop_row172": 0.0013543610416541065,
        "ordinary": -0.00044009638788367815,
        "rings_ge4": -0.005098930149046188,
        "unseen": -0.0019727187357338302,
        "tag_x": "A0_s0",
        "tag_y": "B_s0"
      }
    },
    "A0M0_minus_A0": {
      "0": {
        "delta_mae_best": 0.008619322208571248,
        "ci95": [
          0.0005794383929402101,
          0.018764680947735896
        ],
        "excludes_zero": true,
        "delta_drop_row172": 0.004911034581348397,
        "ordinary": 0.005467569539049531,
        "rings_ge4": 0.016493828206414908,
        "unseen": 0.005781905851753585,
        "tag_x": "A0M0_s0",
        "tag_y": "A0_s0"
      },
      "1": {
        "delta_mae_best": 0.0019208705248311162,
        "ci95": [
          -0.002619490990974009,
          0.0063253412526566525
        ],
        "excludes_zero": false,
        "delta_drop_row172": 0.002770457999432111,
        "ordinary": 0.0030152203375722634,
        "rings_ge4": 0.002720820630065432,
        "unseen": 0.007192012071015825,
        "tag_x": "A0M0_s1",
        "tag_y": "A0_s1"
      }
    }
  },
  "same_arm_seed_spread": {
    "A0": {
      "n": 2,
      "values": [
        0.07965605334291467,
        0.07936091785127064
      ],
      "range": 0.00029513549164403297,
      "mean": 0.07950848559709266
    },
    "A0M0": {
      "n": 2,
      "values": [
        0.08827537189325085,
        0.08128177988802782
      ],
      "range": 0.006993592005223037,
      "mean": 0.08477857589063933
    }
  },
  "moment_ablation": {
    "rules": {
      "MOMENT_REDUNDANT": "mean(A0M0 - A0) <= +0.001",
      "MOMENT_ESSENTIAL": "mean(A0M0 - A0) >= +0.003",
      "AMBIGUOUS": "otherwise"
    },
    "per_seed": {
      "0": {
        "delta_mae_best": 0.008619322208571248,
        "ci95": [
          0.0005794383929402101,
          0.018764680947735896
        ],
        "excludes_zero": true,
        "delta_drop_row172": 0.004911034581348397,
        "ordinary": 0.005467569539049531,
        "rings_ge4": 0.016493828206414908,
        "unseen": 0.005781905851753585,
        "tag_x": "A0M0_s0",
        "tag_y": "A0_s0"
      },
      "1": {
        "delta_mae_best": 0.0019208705248311162,
        "ci95": [
          -0.002619490990974009,
          0.0063253412526566525
        ],
        "excludes_zero": false,
        "delta_drop_row172": 0.002770457999432111,
        "ordinary": 0.0030152203375722634,
        "rings_ge4": 0.002720820630065432,
        "unseen": 0.007192012071015825,
        "tag_x": "A0M0_s1",
        "tag_y": "A0_s1"
      }
    },
    "mean_delta": 0.005270096366701182,
    "n_seeds": 2,
    "n_seeds_worse": 2,
    "verdict": "MOMENT_ESSENTIAL"
  },
  "verdict": "MOMENT_ESSENTIAL",
  "contradiction_verdicts": {
    "A0_minus_B": "NO_EFFECT_CI_INCLUDES_ZERO",
    "A0M0_minus_A0": "MIXED"
  }
}
```

