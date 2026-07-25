# molhiv structure-only probe

Protocol: `molhiv-struct-probe-v0` · n_used=8000

Degree test AUC: **0.4832**

| pool | valid AUC | test AUC | dim |
|------|-----------|----------|-----|
| mean | 0.5494 | 0.5259 | 16 |
| max | 0.5710 | 0.6114 | 16 |
| attn | 0.4899 | 0.6293 | 35 |

## Gate

- P1 pass if best pool **test AUC > degree** and not random (~0.5).
- Then run dual-channel (`run_molhiv_dual.py`) for P2.

Elapsed: 36.72s
