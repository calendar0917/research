# Phase 0 — edit support and differential-signal gate

- Round: `zinc-edit-conditioned-local-contribution-v1`
- Created: `2026-10-29T22:02:31+0800`
- Verdict: **`PASS_COMPOSITION_SIGNAL_ONLY`**
- Data: official ZINC subset train-10k only; valid loaded = `False`, test loaded = `False`

## 1. Strict edit-pair support

Construction: bridge cuts only; the larger side is the preserved body; the smaller side is one pendant edit region; the body graph must be isomorphic with atom/bond labels and the attachment port; `|sub_i| + |sub_j| < min(|body_i|, |body_j|)`; `|sub| <= 12`; RDKit reconstruction failures are rejected; exact graph isomorphism is verified for every accepted body.  Labels are never read during pair construction.

| criterion | required | observed | pass |
|---|---:|---:|:---:|
| accepted strict edit pairs | ≥300 | 350 | yes |
| distinct preserved bodies | ≥100 | 317 | yes |
| edit classes in ≥3 bodies | ≥15 | 29 | yes |

- Independent molecules: 621 / 10,000 (6.21%).
- Distinct bodies: 317; edit classes: 190; max bodies per class: 24.
- Largest connected pair-graph component: 7 molecules; components: 285.
- Pair degree: mean 1.1272141706924315, max 5, p90 2.0, Gini 0.10055670577409707; endpoints / 10,000 = 0.07.  Pair training therefore reweights a small, mildly repeated subset, not the whole train set.
- Ring composition: both substituents acyclic in 37 pairs; at least one substituent contains a ring in 313 pairs; both bodies contain a ring in 350 pairs.
- Edit-region size: median 13.0 (range 2–22); preserved-body size: median 16.0 (range 9–28).  These are one-pendant-region edits, not only tiny functional-group swaps.
- Strict full-substituent-pair classes: 350 total, 0 recurring in ≥3 bodies.  The gate class is therefore deliberately the *local* radius-1 typed signature, not the full substituent pair.

## 2. Differential signal

Train-only labels; 5-fold GroupKFold by preserved body.  Features are signed low-order structural differences (atom-type counts, bond-type counts, size, ring/aromatic counts).  For a strict single-edit pair the preserved body cancels, so these are exactly the edit-region differences.

| model | held-out Δy MAE | rel. improvement vs zero |
|---|---:|---:|
| predict zero | 0.885882 | 0.0000 |
| **low-order structural difference (primary)** | 0.346950 | +0.6084 |
| radius-1 edit-class only | 0.966043 | -0.0905 |
| combined (low-order + true edit class, stress test) | 0.384284 | +0.5662 |

Edit-class permutation control (100 permutations): mean 0.376430, min 0.358892, 5th pct 0.361946.  The primary low-order model (0.346950) beats every control.  The combined true-class model (0.384284) is worse than 81/100 controls (one-sided empirical p = 0.812).  The categorical edit-class identity therefore does **not** add a transferable signal; the recurring local classes exist, but their identity is not the predictive object.

## 3. Gate decision

- Support pass: `True`
- Primary low-order ≥10% improvement: `True` (+0.6084)
- Primary beats permutation mean: `True`
- Class-identity component supported: `False`
- Verdict: **`PASS_COMPOSITION_SIGNAL_ONLY`**

Interpretation: the data contain a strong cross-body transfer signal, but it is additive composition signal, not a categorical edit-type effect.  That is exactly the signal an additive local base head can consume, so the architecture pilot is licensed; the pilot may not be described as evidence for edit-class identity transfer.
