# E2E-DictEnv-A1 — report

Verdict: **REPRESENTATION_NOT_QUALIFIED**

## Gate 0
* correctness all passed: True
* assignment semantics passed: True
* continuity: REAL code-space AUC 0.5344, x-space AUC 0.5606, INDEP code-space AUC 0.7056 (pool 1500, near 800, threshold 0.7)

## Parameter accounting

* TOPO 97487, INDEP 109263, REAL 109263 (INDEP == REAL exactly)

### G0.3 continuity strata (frozen gate is `REAL` code-space)

| arm | x-space AUC | code-space AUC | graded-tail x AUC | graded-tail code AUC |
|---|---|---|---|---|
| TOPO | 0.4669 | 0.4325 | 0.5463 | 0.5706 |
| INDEP | 0.6375 | 0.7056 | 0.7819 | 0.8375 |
| REAL | 0.5606 | 0.5344 | 0.6756 | 0.6588 |

Post-hoc stratum diagnostic (not the gate): the frozen near stratum is entirely equal-size (mean patch size 5.35, range 3-9), WL cosine >= 1.000000. Sampled-pair Spearman(WL cosine, -distance): REAL x 0.317 / code 0.259; chemistry-blind TOPO x 0.518. Isomorphic control distance: REAL x 4.66e-09.

## Dictionary health (official train, exact OMP)

| arm | dim | rec(fit) | rec(holdout) | random/holdout | used | effective | top1 mass | train-valid rho |
|---|---|---|---|---|---|---|---|---|
| TOPO | 65 | 1.364e-05 | 1.410e-05 | 49286.4 | 32/32 | 4.03 | 0.679 | 0.9989 |
| INDEP | 433 | 2.122e-02 | 2.109e-02 | 45.2 | 32/32 | 18.59 | 0.149 | 0.9989 |
| REAL | 433 | 1.964e-01 | 1.971e-01 | 4.8 | 31/32 | 19.99 | 0.134 | 0.9993 |

## Stage 2 — not reached (Gate 0 stop)

## Anchors

* historic P2-ABS H1 official-valid soup: 0.123548629
* official ZINC test was never loaded in this round.
