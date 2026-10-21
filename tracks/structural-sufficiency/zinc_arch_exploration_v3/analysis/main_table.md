| model | params | best valid | best ep | eval-train@best | final valid | updates | SSL | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| P3B_s0 | 413,505 | 0.076223754 | 377 | 0.014680 | 0.078178808 | 47,400 | P3 masked SSL encoders | completed | 0.3014 |
| L_s0 | 494,193 | 0.078537152 | 489 | 0.005510 | 0.079076417 | 47,400 | P3 masked SSL encoders | completed | 0.3452 |
| C_s0 | 494,193 | 0.082367432 | 482 | 0.004906 | 0.082965206 | 47,400 | P3 masked SSL encoders | completed | 0.3999 |
| A_s0 | 488,601 | 0.082189201 | 632 | 0.011543 | 0.083806124 | 188,400 | P3 masked SSL encoders | completed | 2.7151 |
| B_s0 | 448,623 | 0.101720000 | 332 | nan | n/a (truncated) | 128,269 | P3 masked SSL encoders | CANCELLED_BY_BUDGET | 2.3005 |
| A_SSL_s0 | 488,601 | 0.079876146 | 439 | 0.017018 | 0.084783518 | 141,300 | P3 masked SSL encoders + native A-SSL pretrain | completed | 1.8394 |
| A_attn_s0 | 488,889 | 0.082637432 | 519 | 0.012976 | 0.085459559 | 141,300 | P3 masked SSL encoders | completed | 2.0216 |
