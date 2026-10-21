# DBT — main table (valid = development/selection, test = descriptive)

| model | params | best valid | best ep | train@best | test MAE | updates | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| P3B_s0 | 413,505 | 0.076223754 | 377 | 0.014680 | 0.069864841 | 47,400 | completed | 0.30 |
| L_s0 | 494,193 | 0.078537152 | 489 | 0.005510 | 0.066266555 | 47,400 | completed | 0.35 |
| A_s0 | 488,601 | 0.082189201 | 632 | 0.011543 | 0.080528948 | 188,400 | completed | 2.72 |
| A_SSL_s0 | 488,601 | 0.079876146 | 439 | 0.017018 | 0.084753448 | 141,300 | completed | 1.84 |
| A_attn_s0 | 488,889 | 0.082637432 | 519 | 0.012976 | 0.079930412 | 141,300 | completed | 2.02 |
| DBT_s0 | 452,523 | 0.077768402 | 949 | 0.015964 | 0.069401494 | 235,500 | completed | 3.69 |

Historical rows are reused unchanged from `zinc_arch_exploration_v3/analysis/{main_table,test_reporting}.json`; they are not retrained and their test numbers are not recomputed.  The new row's test MAE is one descriptive read of its frozen best-valid checkpoint.

Best epoch 949, lr at best 9.205e-05, valid in the `lr>1e-4` band 0.079030852 @614, mean epoch 8.85s.

Test − valid for DBT: -0.008367 (test RMSE 0.1534, median AE 0.0360, p95 AE 0.2235).

Valid worst molecule: row 172 with AE 7.1262 = 9.2 % of the valid MAE (valid MAE without it 0.070712873; 4 molecules have AE > 1).

Descriptive paired valid bootstrap (same resamples):

| ref | MAE ref | mean Δ | 95 % CI | DBT better on | ref AE at worst row | mean Δ without worst |
|---|---:|---:|---|---:|---:|---:|
| P3B_s0 | 0.076223754 | +0.001545 | [-0.011511, +0.018792] | 52.5 % | 1.5630 | -0.004023 |
| L_s0 | 0.078537152 | -0.000769 | [-0.011279, +0.012395] | 50.8 % | 6.0993 | -0.001797 |

Descriptive only; the worst molecule is not used for selection or as a headline.

Descriptive paired test bootstrap (10,000 resamples, Δ = AE(DBT) − AE(ref)):

| ref | MAE ref | MAE DBT | mean Δ | 95 % CI | DBT better on |
|---|---:|---:|---:|---|---:|
| P3B_s0 | 0.069864841 | 0.069401494 | -0.000463 | [-0.005308, +0.004362] | 50.3 % |
| L_s0 | 0.066266555 | 0.069401494 | +0.003135 | [-0.002272, +0.008404] | 47.5 % |

Single seed; the intervals are descriptive, not confirmatory.
