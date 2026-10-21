# DBT — main table (valid = development/selection, test = descriptive)

| model | params | best valid | best ep | train@best | test MAE | updates | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| P3B_s0 | 413,505 | 0.076223754 | 377 | 0.014680 | 0.069864841 | 47,400 | completed | 0.30 |
| L_s0 | 494,193 | 0.078537152 | 489 | 0.005510 | 0.066266555 | 47,400 | completed | 0.35 |
| A_s0 | 488,601 | 0.082189201 | 632 | 0.011543 | 0.080528948 | 188,400 | completed | 2.72 |
| A_SSL_s0 | 488,601 | 0.079876146 | 439 | 0.017018 | 0.084753448 | 141,300 | completed | 1.84 |
| A_attn_s0 | 488,889 | 0.082637432 | 519 | 0.012976 | 0.079930412 | 141,300 | completed | 2.02 |
| DBT_s0 | 452,523 | 0.077768402 | 949 | 0.015964 | n/a | 235,500 | completed | 3.69 |

Historical rows are reused unchanged from `zinc_arch_exploration_v3/analysis/{main_table,test_reporting}.json`; they are not retrained and their test numbers are not recomputed.  The new row's test MAE is one descriptive read of its frozen best-valid checkpoint.

Best epoch 949, lr at best 9.205e-05, valid in the `lr>1e-4` band 0.079030852 @614, mean epoch 8.85s.
