# Fusion ablations (Xu global epoch)

Dataset: **MUTAG** · `xu-gin-fusion-v0`

| variant | Acc % @ e* | e* | per-fold-max mean % |
|---------|------------|----|---------------------|
| gin_only | 89.4 ± 5.8 | 114 | 94.6 |
| concat | 90.4 ± 5.3 | 53 | 95.2 |
| residual | 89.3 ± 5.4 | 13 | 93.0 |
| gate | 89.4 ± 5.8 | 47 | 95.7 |

## Fusion definitions

- **gin_only**: attributes only
- **concat**: `x' = [x; s_v]`
- **residual**: after each GINConv, `h = h + W s_v` (W bias-free)
- **gate**: `h = h + σ(g) W s_v`, `g = Linear([h; s_v])`

Structure: node-level B0 |coef| (12-d), shared D train-only, z-score.

