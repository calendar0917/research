| model | params | init | best valid | ep | eval-train@best | final valid | test MAE | epochs | updates | GPU-h |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3B_s0 | 413,505 | train-only masked SSL | 0.076223754 | 377 | — | — | 0.069864841 | 600 | 47,400 | 0.3000 |
| L_s0 | 494,193 | train-only masked SSL | 0.078537152 | 489 | 0.005510 | — | 0.066266555 | 600 | 47,400 | 0.3500 |
| DBT_s0 | 452,523 | train-only masked SSL | 0.077768402 | 949 | 0.015964 | 0.081314 | 0.069401494 | 1500 | 235,500 | 3.6918 |
| old P3-BF_s0 | 447,505 | supervised P3B best.pt | 0.076223756 | 0 | 0.014680 | 0.078363 | 0.069864822 | 100 | 7,900 | 0.0645 |
| A_joint_BF_s0 | 447,505 | train-only masked SSL atom/bond/junction only | 0.076731364 | 375 | 0.015666 | 0.078218 | 0.070067461 | 600 | 47,400 | 0.3640 |
| B_object_tokens_s0 | 470,469 | train-only masked SSL atom encoder only | 0.135830103 | 230 | 0.051184 | 0.149754 | 0.108306215 | 490 | 38,710 | 0.5183 |
