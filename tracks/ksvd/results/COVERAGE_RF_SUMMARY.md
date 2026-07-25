# Coverage-driven RW + receptive-field curve (process metrics)

Protocol: `process-coverage-rf-v0` · hard_delete=False

## A. Coverage (synthetic ring+chords)

| name | walks | traj cover | traj repeat | mean|S| |
|------|-------|------------|-------------|---------|
| all_nodes_no_decay | 24 | 1.000 | 5.143 | 7.96 |
| random_decay | 11 | 1.000 | 1.750 | 8.00 |
| deg_strat_decay | 8 | 1.000 | 1.000 | 8.00 |
| uncovered_decay | 8 | 1.000 | 1.036 | 8.00 |
| uncovered_no_decay | 10 | 1.000 | 1.607 | 7.90 |

B0 (every node star): n_sg=24 induced_cover=1.0 induced_repeat=1.0

## B. RF curve — fraction of C4-class graphs with ≥1 patch containing C4

| setting | c4_hit | mean|S| |
|---------|--------|---------|
| B0_1hop | 0.000 | 3.00 |
| RW_L4_m4_w8 | 0.325 | 3.87 |
| RW_L6_m6_w8 | 0.887 | 5.28 |
| RW_L8_m8_w8 | 0.988 | 6.42 |
| RW_L12_m10_w8 | 1.000 | 7.67 |
| RW_L8_m8_w4 | 0.887 | 6.38 |

## C. luyin10 checklist update

| ID | Status after this suite |
|----|-------------------------|
| R4 少重复 | process metrics + soft decay + uncovered seeds |
| R5 非全点 | degree_stratified / uncovered / budget max_walks |
| R6 不硬删 | invariant hard_delete=False |
| R1 感受野 | RF curve c4_hit vs L,m |

Downstream Acc not claimed here.

