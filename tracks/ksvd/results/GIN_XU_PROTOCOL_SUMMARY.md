# GIN ± node structure under **Xu GIN paper epoch rule**

Dataset: **MUTAG**

Protocol: `xu-gin-paper-epoch-v0`

single global epoch = argmax_e mean_fold Acc(e); report mean±std of folds at e*

HPs (fixed for relative compare): `{'epochs': 350, 'hidden': 32, 'batch_size': 32, 'dropout': 0.5, 'lr': 0.01, 'lr_decay': '×0.5 every 50 epochs', 'layers_mp': 4}`

Paper also grids hidden∈{16,32}, batch∈{32,128}, dropout∈{0,0.5}; we fix one setting.

| variant | Acc % @ global e* | e* | per-fold-max mean % (old) |
|---------|-------------------|----|---------------------------|
| gin_only | 89.4 ± 5.8 | 114 | 94.6 |
| gin_node_B0 | 90.4 ± 5.3 | 53 | 95.2 |
| gin_node_RW | 84.6 ± 9.6 | 165 | 92.6 |

## Literature anchor (external)

- Xu GIN MUTAG: **89.4 ± 5.6 (Xu Table; their full HP grid + code)**
- anchor only if same protocol family; not same implementation

## Protocol difference vs our previous 94.6%

| | Previous (node_struct_gin) | This run (Xu) |
|--|------------------------------|---------------|
| epoch pick | **per-fold** max held-out | **one global** e* on mean curve |
| hidden | 64 (paper: social) | **32** (paper: bio) |
| lr schedule | none | ×0.5 / 50 ep |

Same-protocol relative compare: gin_only vs gin_node_* in the table above.

