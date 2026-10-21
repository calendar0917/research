# Erratum — zinc_generalization_audit_v1

Date: 2026-10-21. Scope: derivative statistics, report text, and wording only.
No prediction, checkpoint, label, split, cache, or historical training result was
modified or rebuilt. Official test was not read.

## 1. Top-case path-unseen mass

The top-case table in `REPORT.md` §2.10 had copied split-local pattern IDs
(`0.607` for row 917, `0.701` for row 172, etc.). The correct content-key values
already present in `analysis/per_molecule.csv` and `analysis/top_cases.json` are:

| row | old shown | corrected path-unseen mass |
|---:|---:|---:|
| 917 | 0.607 | 0.000 |
| 172 | 0.701 | 0.000 |
| 753 | 0.605 | 0.007 |
| 659 | 0.519 | 0.011 |
| 260 | 0.531 | 0.039 |
| 31 | 0.493 | 0.011 |
| 447 | 0.395 | 0.140 |
| 224 | 0.485 | 0.000 |
| 678 | 0.579 | 0.000 |
| 375 | 0.607 | 0.066 |

The per-molecule table and `top_cases.json` were already on the correct
content-key basis; only the report table was stale.

## 2. Partial Spearman with ties

The old helper ranked with a double `argsort`, which splits tied values by row
order, and residualised on `n_atoms` only. The corrected helper uses
`scipy.stats.rankdata(method="average")` and `numpy.linalg.lstsq` residuals of
both average-ranked vectors on an intercept plus average-ranked controls.

Independent recomputation from `analysis/per_molecule.csv`:

| statistic (radius-2 unseen atom fraction vs P3B AE) | value |
|---|---:|
| ordinary Spearman | 0.3136804055 |
| partial, controlling `n_atoms` | 0.3011701829 |
| partial, controlling `n_atoms` + `cycle_rank` | 0.2956178151 |

A fixed row permutation reproduces all three values to `1.1e-16`
(`analysis/erratum_checks.json`; `pass=true`). `analysis/coverage_summary.json`
now records both partial versions and the row-order check. The report's old
`0.301` "controlling n_atoms + cycle_rank" value was the size-only corrected
number; the size+cycle value is 0.29562.

## 3. Radius-2 signature wording

"Radius-2 signature" means the code's two nested neighbourhood rounds: central
atom type + incident bond types + the one-ring signature of each neighbour. It
is not the exact radius-2 induced subgraph, and it does not claim to record ring
closure or node coincidence completely.

## 4. H2 / H3 wording

- H2 (sample/structure support): **not rejected**. The 99.6% path-occurrence
  coverage argues against a simple global-label-shortage story, but it cannot
  exclude finite supervision/compositional support: only molecule-level `y`
  exists, with no independent label per pattern. This remains an open
  explanation.
- H3 (Null-grouping interface dilution): **the hypothesis tested this round, not
  a confirmed main cause**. The audit evidence is consistent with it, but the
  frozen-P3B Null→organized replacement (0.0762→0.1213) only shows that an
  untrained replacement is bad; it does not prove a parallel channel helps.

`analysis/decision.json` and the corresponding report sections were narrowed
accordingly. The original predictions, checkpoints, labels, and historical
training scores are untouched.
