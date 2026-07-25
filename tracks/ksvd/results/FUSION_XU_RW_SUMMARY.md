# Fusion ablations (Xu global epoch)

Dataset: **MUTAG** · `xu-gin-fusion-v0`

Patch: `{'mode': 'RW', 'p': 0.5, 'q': 2.0, 'walk_length': 6, 'max_nodes': 8, 'num_walks': 5, 'pool': 'mean_max', 'how_selected': 'B0: S={v}∪N(v). RW: r node2vec walks from v; each → induced G[S]; x_i=OMP(D,y_i); s_v=pool_i(x_i) (mean/max/mean_max).', 'p_q_selection': 'Not grid-searched on MUTAG. Default multi-walk run uses p=0.5,q=2 (local bias).', 'recon_note': 'recon comparable only within same patch family; not B0 vs RW ranking.'}`

| variant | Acc % @ e* | e* | per-fold-max mean % |
|---------|------------|----|---------------------|
| gin_only (ref) | 89.4 ± 5.8 | — | — |
| gate | 89.3 ± 7.6 | 168 | 93.1 |

## Fusion definitions

- **gin_only**: attributes only
- **concat**: `x' = [x; s_v]`
- **residual**: after each GINConv, `h = h + W s_v` (W bias-free)
- **gate**: `h = h + σ(g) W s_v`, `g = Linear([h; s_v])`

## How subgraph is chosen

B0: S={v}∪N(v). RW: r node2vec walks from v; each → induced G[S]; x_i=OMP(D,y_i); s_v=pool_i(x_i) (mean/max/mean_max).

## p, q selection

Not grid-searched on MUTAG. Default multi-walk run uses p=0.5,q=2 (local bias).

