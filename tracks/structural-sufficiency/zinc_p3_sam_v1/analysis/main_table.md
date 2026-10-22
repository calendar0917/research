# zinc_p3_sam_v1 — main table

| arm | params | best valid | ep | eval-train@best | final valid | test MAE | RMSE | p95 AE | max AE | updates | grads | wall s | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C P3-Control** | 413,505 | 0.079588262 | 431 | 0.010286 | 0.081668123 | **0.072688588** | 0.165587 | 0.209016 | 2.504708 | 47,400 | 47,400 | 1169 | 0.325 |
| **S P3-SAM** | 413,505 | 0.118310817 | 299 | 0.041025 | 0.140518076 | **0.100376384** | 0.176018 | 0.337065 | 2.035403 | 47,400 | 94,800 | 2516 | 0.699 |

Same-round contrast (molecule-paired, 10,000 bootstrap resamples):

- valid S−C = +0.038723, 95% CI [+0.023093, +0.062618]
- test S−C = +0.027688, 95% CI [+0.018752, +0.036656]; molecule-level fraction with AE(S) < AE(C) = 0.339
- valid C−P3B = +0.003365; test C−P3B = +0.002824

Historical rows are reused reference predictions (not retrained) and only for orientation; the C vs S contrast is the same-round evidence.

| reference | params | valid | test | source |
|---|---:|---:|---:|---|
| P3B_s0 | 413,505 | 0.076223754 | 0.069864841 | reference_preds_test.csv (recomputed MAE) |
| L_s0 | 494,193 | 0.078537152 | 0.066266555 | reference_preds_test.csv (recomputed MAE) |
| DBT_s0 | 452,523 | 0.077768402 | 0.069401494 | reference_preds_test.csv (recomputed MAE) |
| A_joint_BF_s0 | 447,505 | 0.076731364 | 0.070067461 | zinc_structural_object_tokens_v1 frozen REPORT/analysis; no prediction rows reused |
