# BOSI statistics-only SSL v1 — frozen tables

## Locked MAE (one read, n=1000)

| run | init | seed | ckpt | Locked MAE | RMSE |
|---|---|---:|---|---:|---:|
| B_s0/best_monitor | historical | 0 | best_monitor | 0.082449 | 0.2837 |
| B_s0/final | historical | 0 | final | 0.081969 | 0.2838 |
| B_s1/best_monitor | historical | 1 | best_monitor | 0.081491 | 0.2760 |
| B_s1/final | historical | 1 | final | 0.080795 | 0.2750 |
| S_s0/best_monitor | ssl | 0 | best_monitor | 0.087756 | 0.2747 |
| S_s0/final | ssl | 0 | final | 0.084910 | 0.2758 |

## Paired contrasts (negative = first better)

| contrast | ΔMAE | CI95 | better frac | top-10 removed | ordinary | any-unseen |
|---|---:|---|---:|---:|---:|---:|
| S_s0 - B_s0 | +0.005307 | [+0.000996, +0.009452] | 0.469 | +0.005875 | +0.005491 | +0.009791 |

## Decision: `NO_GAIN`

stopped at the pre-registered seed-0 Monitor gate; a single seed cannot satisfy the two-seed rule

- `pilot_atom_beats_prior`: true
- `pilot_bond_beats_prior`: true
- `pilot_atom_uses_partner_stats`: true
- `pilot_bond_uses_partner_stats`: true
- `ssl_seeds_run`: [0]
- `monitor_deltas`: {"S_s0": 0.008184483449906119}
- `seed0_monitor_gate`: false
