# zinc_shared_factor_v1

| arm | seed | params | best valid | best ep | final valid | best20 median | tail100 median | s/epoch | GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SF-Dense | 0 | 449601 | 0.081252 | 339 | 0.082611 | 0.082610 | 0.082645 | 2.37 | 0.411 |
| SF-SharedState | 0 | 451777 | 0.077045 | 384 | 0.078846 | 0.078646 | 0.078800 | 3.58 | 0.620 |

## contrasts (best checkpoint, per molecule)

| pair | dMAE | 95% CI (molecules) | frac improved | median dAE |
| --- | --- | --- | --- | --- |
| T_minus_D_s0 | -0.004207 | [-0.008219, -0.000284] | 0.526 | -0.002131 |
| T_minus_N_s0 | n/a | n/a | n/a | n/a |
| N_minus_D_s0 | n/a | n/a | n/a | n/a |
| T_minus_D_s1 | n/a | n/a | n/a | n/a |
| T_minus_N_s1 | n/a | n/a | n/a | n/a |
| N_minus_D_s1 | n/a | n/a | n/a | n/a |

## groups (valid, radius-2 x ordinary)

| cell | n | D_s0 | T_s0 |
| --- | --- | --- | --- |
| seen_ordinary | 710 | 0.0510 | 0.0487 |
| seen_tail | 39 | 0.1605 | 0.1472 |
| unseen_ordinary | 225 | 0.1359 | 0.1339 |
| unseen_tail | 26 | 0.3149 | 0.2537 |

## stage gates

```json
{
 "A_s0": {
  "available": true,
  "checks": {
   "margin": {
    "T": 0.07704540812963387,
    "D": 0.08125194833794376,
    "required": "T <= D - 0.003",
    "pass": true
   },
   "absolute": {
    "T": 0.07704540812963387,
    "required": "T <= 0.075",
    "pass": false
   },
   "window_median": {
    "available": true,
    "metric": "best20_window_median",
    "candidate": 0.07864591038221261,
    "baseline": 0.08261014082463225,
    "delta": -0.003964230442419642,
    "supports_candidate": true,
    "required": "T < D (20-epoch sliding median)",
    "pass": true
   },
   "tail100_median": {
    "available": true,
    "metric": "tail100_median",
    "candidate": 0.0788000472559943,
    "baseline": 0.08264509877184173,
    "delta": -0.0038450515158474363,
    "supports_candidate": true,
    "required": "T < D (tail-100 median)",
    "pass": true
   }
  },
  "pass": false,
  "would_satisfy_B_direction": true,
  "B_executed": false,
  "B_reason": "stage A failed, so stage B/C were not launched (DESIGN.md \u00a77)"
 },
 "A_s1": {
  "available": false
 },
 "C": {
  "available": false,
  "reason": "N runs not executed (stage B not repeated)"
 }
}
```
