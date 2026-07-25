# Pool ablation (synthetic C4)

Protocol: `pool-ablation-v0`

Degree baseline: **87.5%**

| pool | Acc % | dim |
|------|-------|-----|
| mean | 92.5 ± 1.6 | 12 |
| max | 92.5 ± 4.2 | 12 |
| attn | 92.0 ± 2.9 | 27 |
| rich+mean | 94.0 ± 3.7 | 132 |
| rich+attn | 93.0 ± 3.7 | 159 |

## Interpretation

- Compare **attn** vs **mean** on same D; gain → keep MIL for molhiv P1.
- No claim on molhiv; synthetic mechanism only.

Elapsed: 1.4s
