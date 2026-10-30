# ERRATUM — Stage IV radius-2 support vocabulary

Target: `zinc_compositional_research_tree_v1` Stage IV (train10k, official valid1k). Predictions and checkpoints are unchanged; only the radius-2 support grouping is recomputed.

## What was wrong

Stage IV was trained on the complete official train10k, but the report grouped valid molecules with a radius-2 environment vocabulary built from Fit8k only. The correct support vocabulary for that stage is the complete train10k structure.

* old vocabulary: Fit8k, 11015 signatures
* new vocabulary: train10k, 12108 signatures

## Group counts

| group | old (Fit8k vocab) | new (train10k vocab) |
|---|---:|---:|
| full | 1000 | 1000 |
| ordinary | 935 | 935 |
| unseen_ordinary | 267 | 225 |
| seen_ordinary | 668 | 710 |
| tail | 65 | 65 |
| any_unseen | 297 | 251 |

Rows that change side: any_unseen: 46 old-only, 0 new-only; unseen_ordinary: 42 old-only, 0 new-only; seen_ordinary: 0 old-only, 42 new-only.

## REAL − MEAN MAE (negative = REAL better)

| seed | group | n old | Δ old | n new | Δ new | Δ new 95% CI | win new |
|---:|---|---:|---:|---:|---:|---|---:|
| 0 | full | 1000 | -0.016733 | 1000 | -0.016733 | [-0.043570, -0.000542] | 0.518 |
| 0 | ordinary | 935 | -0.001773 | 935 | -0.001773 | [-0.006254, +0.002653] | 0.509 |
| 0 | unseen_ordinary | 267 | -0.000703 | 225 | -0.002998 | [-0.016662, +0.010513] | 0.547 |
| 0 | seen_ordinary | 668 | -0.002200 | 710 | -0.001384 | [-0.005293, +0.002386] | 0.497 |
| 0 | tail | 65 | -0.231935 | 65 | -0.231935 | [-0.628409, -0.006272] | 0.646 |
| 0 | any_unseen | 297 | -0.008476 | 251 | -0.011784 | [-0.036818, +0.007889] | 0.542 |
| 1 | full | 1000 | -0.019705 | 1000 | -0.019705 | [-0.039175, -0.007567] | 0.556 |
| 1 | ordinary | 935 | -0.009608 | 935 | -0.009608 | [-0.014271, -0.004894] | 0.553 |
| 1 | unseen_ordinary | 267 | -0.012320 | 225 | -0.014535 | [-0.028392, -0.000721] | 0.556 |
| 1 | seen_ordinary | 668 | -0.008524 | 710 | -0.008047 | [-0.012691, -0.003594] | 0.552 |
| 1 | tail | 65 | -0.164939 | 65 | -0.164939 | [-0.439509, -0.008663] | 0.600 |
| 1 | any_unseen | 297 | -0.014593 | 251 | -0.015990 | [-0.030851, -0.001084] | 0.546 |

## Effect of the correction

* The correction makes the unseen set **smaller and more strictly unseen**: 251 vs 297 any-unseen rows and 225 vs 267 unseen-ordinary rows.
* For seed 0 the unseen-ordinary REAL advantage strengthens from -0.000703 to -0.002998 (still not reaching the −0.004 purchase threshold and the CI still crosses 0).
* For seed 1 the unseen-ordinary advantage strengthens from -0.012320 to -0.014535.
* The recorded Stage IV decision is **not rewritten**. It was based on the old grouping; this erratum only reports what the correct grouping shows. The qualitative reading in the Stage IV report (tail effect unambiguous, ordinary/unseen effect seed-dependent) is unchanged.

Old numbers remain in `zinc_compositional_research_tree_v1/analysis/contrasts.json`; the old predictions and checkpoints were not modified.

Machine-readable: `/home/calendar/code/research/tracks/structural-sufficiency/zinc_lcrc_hierarchical_composition_v1/analysis/erratum_stage4_support.json`.
