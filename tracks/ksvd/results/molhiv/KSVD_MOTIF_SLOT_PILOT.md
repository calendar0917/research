# KSVD motif-slot scaffold pilot

## Protocol

- `n=8000`, only OGB official-train is used for development.
- Three Bemis–Murcko scaffold folds (`seed=20260726`); scaffold groups do not overlap across inner train/valid.
- **Each fold has its own dictionary cache fit only on that fold's inner-train graphs.** The held-out scaffold fold is encoded but never used for KSVD/PCA/random dictionary fitting.
- Official-valid and official-test evaluations: **0**.
- Model: one 3-layer GINE-JK backbone, then after GINE layer 1 one atom→dictionary-slot→atom transport.
- KSVD configuration: radius-2 patch, `D=32`, OMP `T=3`, motif bottleneck rank 16, zero-init residual gate, gate scale 0.25, token branch warm-up 10 epochs.
- Trainable parameters: GINE `37,382`; motif-slot `42,359` (+4,977, +13.3%).

## Seed-0 matched controls across three scaffold folds

| Model | Fold 0 | Fold 1 | Fold 2 | Mean | Δ vs GINE | Wins |
|---|---:|---:|---:|---:|---:|---:|
| GINE-JK | 0.75225 | 0.63354 | 0.78157 | 0.72245 | — | — |
| **KSVD motif-slot** | **0.75902** | 0.64343 | **0.78112** | **0.72785** | **+0.00540** | 2/3 |
| PCA motif-slot | 0.75579 | **0.64488** | 0.77548 | 0.72538 | +0.00293 | 2/3 |
| Random-patch motif-slot | 0.75490 | 0.64475 | 0.76175 | 0.72047 | -0.00198 | 2/3 |
| Graph-wise shuffled KSVD IDs | 0.74203 | 0.63833 | 0.77764 | 0.71933 | -0.00312 | 1/3 |
| KSVD transport without atom ID | 0.75368 | 0.64198 | 0.76396 | 0.71987 | -0.00258 | 2/3 |

KSVD beats shuffled-ID and no-ID controls on all three folds. Its mean margins are +0.00852 and +0.00798 respectively. This is the first evidence in this line that persistent dictionary-atom identity contributes beyond generic extra message passing. KSVD also beats PCA on 2/3 folds, with a smaller mean margin of +0.00247.

## Frozen-config 3-fold × 3-seed confirmation

| Fold | Seed | GINE | KSVD motif-slot | Δ |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.75225 | 0.75902 | +0.00677 |
| 1 | 0 | 0.63354 | 0.64343 | +0.00989 |
| 2 | 0 | 0.78157 | 0.78112 | -0.00045 |
| 0 | 1 | 0.72594 | 0.73145 | +0.00551 |
| 1 | 1 | 0.64771 | 0.66085 | +0.01314 |
| 2 | 1 | 0.76229 | 0.76483 | +0.00254 |
| 0 | 2 | 0.74510 | 0.74149 | -0.00361 |
| 1 | 2 | 0.64244 | 0.64444 | +0.00200 |
| 2 | 2 | 0.76594 | 0.76342 | -0.00252 |

Aggregate:

- GINE mean: **0.71742**
- KSVD motif-slot mean: **0.72112**
- Paired mean delta: **+0.00370**
- Wins: **6/9**
- Delta standard deviation: **0.00563**

This passes the predeclared 6/9 win criterion, but misses the stronger `mean delta >= +0.005` confirmation criterion. Therefore P0 is **promising but not yet sufficient** for an official-valid/test run.

## Runtime result

The original dense transport meets the small-data smoke target and was about 1.44× GINE in the seed-0 three-fold batch, but longer confirmation runs showed scheduling-dependent overhead. A sparse `T=3` implementation reduced fold-0 runtime to 23.3 s, but changed floating-point accumulation/training trajectory and dropped AUC to 0.74917. It is retained as an explicit `--motif-slot-backend sparse` experimental option; the reported results use the reproducible dense backend.

## Decision and next breakthrough

Do **not** touch official-valid/test yet. The graph-global slot creates a message shortcut between every occurrence of the same atom inside a molecule. The next clean KSVD-native step is P1: split each `(graph, dictionary atom)` into spatially connected motif occurrences, and perform atom↔occurrence transport. This preserves self-learned KSVD identities while preventing unrelated distant occurrences from being merged. Required controls remain PCA, random patch, shuffled IDs, and no-ID occurrence transport.
