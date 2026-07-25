# molhiv structure-only probe

Protocol: `molhiv-struct-probe-v0` · n_used=2000

Degree test AUC: **0.5366**

| pool | valid AUC | test AUC | dim |
|------|-----------|----------|-----|
| mean | 0.5751 | 0.8737 | 16 |
| max | 0.5662 | 0.8914 | 16 |
| attn | 0.5263 | 0.6717 | 35 |

## Gate

- P1 pass if best pool **test AUC > degree** and not random (~0.5).
- Then run dual-channel (`run_molhiv_dual.py`) for P2.

Elapsed: 11.66s
