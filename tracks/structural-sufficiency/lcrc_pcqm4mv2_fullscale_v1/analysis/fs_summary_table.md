# Full-Train comparison (Validation MAE, eV)

| model | Train size | Parameters | Best Valid MAE | Best epoch | Valid@40 | Monitor@40 | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2-1M historical | 1,000,000 | 939,079 | 0.132311 | 40 | 0.132311 | 0.105891 | 5.162 |
| W3-1M historical | 1,000,000 | 2,038,167 | 0.127809 | 39 | 0.127875 | 0.097213 | 3.447 |
| W2-FULL | 3,378,606 | 939,079 | 0.128821 | 18 | 0.135414 | 0.117057 | 11.787 |
| W3-FULL | 3,378,606 | 2,038,167 | 0.118775 | 35 | 0.123822 | 0.102832 | 11.628 |

W3−W2 at FULL: best -0.010046 eV, @40 -0.011592 eV (-8.56 %).
1M→FULL: W2 +0.003103 eV, W3 -0.004053 eV (at epoch 40).
