# Graph-level RW sampling — solidification

Protocol: `graph-level-solid-v0`

## 1. Default algorithm

**Name:** CoverageRW-KSVD-Readout

- 1. Seeds: degree_stratified (not all nodes)
- 2. Each seed: node2vec walk (p,q), soft decay on traversed edges
- 3. Cap nodes → induced G[S] as patch
- 4. Stop if traj_edge_cover>=target or max_walks
- 5. Never hard-delete edges
- 6. Train: stack patches → KSVD shared D
- 7. Per graph: encode patches with D → energy/rich readout → s_G
- 8. Downstream: LR on s_G (synthetic graph classification)

Config: `{'p': 0.5, 'q': 2.0, 'L': 8, 'm': 8, 'max_walks': 12, 'edge_decay': 0.7, 'seed_policy': 'degree_stratified', 'cover_target': 0.95, 'n_atoms': 12, 'T': 3}`

## 2. Process metrics (many graphs)

| tag | n | cover | repeat | walks | mean|S| |
|-----|---|-------|--------|-------|---------|
| c4_set_coverage | 200 | 1.000±0.000 | 2.181±0.852 | 5.7 | 6.65 |
| c4_set_B0 | 200 | 1.000±0.000 | 0.999±0.011 | 14.0 | 3.00 |
| tri_set_coverage | 160 | 0.999±0.007 | 2.518±1.030 | 7.1 | 6.32 |
| tri_set_B0 | 160 | 1.000±0.000 | 1.092±0.093 | 16.0 | 3.00 |
| ring_family_coverage | 30 | 0.976±0.026 | 1.176±0.491 | 7.6 | 7.92 |
| ring_family_B0 | 30 | 1.000±0.000 | 1.000±0.000 | 22.0 | 3.27 |

## 3. Closed loop Acc (C4 vs C8, shared D, 5-fold)

Degree baseline: **82.0%**

| mode | Acc % |
|------|-------|
| B0 | 82.0 ± 5.3 |
| coverage | 91.0 ± 5.4 |

### Triangle task (control)

| mode | Acc % |
|------|-------|
| B0 | 100.0 ± 0.0 |
| coverage | 79.4 ± 8.1 |

## Interpretation

- **Process:** coverage mode should use fewer walks than all-node B0 while keeping high cover; lower repeat is good **for graph-level corpus**.
- **Closed loop:** coverage should beat B0 on C4 if sampling+KSVD capture multi-hop rings.
- Triangle task may be less RF-sensitive; report honestly.
- This does **not** claim MUTAG/GIN gains; graph-level only.

