# Stable real-prototype MIL + low-capacity occurrence residual (2026-07-29)

## 1. Question and evaluation boundary

The experiment asks whether the spatial occurrence shape of frozen real
prototypes supplies a small signal beyond the previously frozen
three-prototype-bank MIL ensemble.

Development is restricted to the 8,000-graph subset and its 6,400
official-train graphs:

- three official-train-only Bemis--Murcko scaffold folds;
- fold-specific raw radius-2 descriptor + PCA64 caches;
- official valid/test latent rows remain exactly zero;
- no official-valid/test labels or predictions are evaluated;
- fixed training epochs; no held-out epoch selection.

The official valid split is deliberately reserved for a small number of
candidates that pass the internal scaffold-fold gate. This is not because
valid cannot be used for tuning. It is the next-level selection set, whereas
the internal folds support broad and inexpensive development. Test remains a
terminal evaluation after the valid-selected configuration is frozen.

## 2. Frozen-member residual

Implementation:

- `code/run_molhiv_prototype_occurrence_residual.py`

For each frozen graph-balanced real-prototype MIL member:

1. load its already saved fit/held-out logits; the 46,658-parameter base model
   is never retrained or changed;
2. reconstruct and hash-check its exact 32 observed-patch prototypes;
3. split each prototype's positive top-3 node support into connected molecular
   components;
4. compute six statistics per prototype: component count, active-node count,
   largest component size, singleton fraction, total coefficient mass, and
   maximum component mass;
5. train a 4,706-parameter one-hidden-layer residual with a zero-initialized
   scalar gate and a maximum logit correction of 0.25.

Controls:

- `real`: persistent prototype identity retained;
- `shuffled`: prototype rows independently permuted in every graph;
- `no_id`: rows sorted by occurrence shape, removing persistent identity.

### 2.1 Three-bank ensemble by scaffold fold

| Fold | Frozen base | Real identity | Shuffled identity | No identity |
|---:|---:|---:|---:|---:|
| 0 | 0.748232 | 0.750928 | 0.749500 | 0.749593 |
| 1 | 0.677951 | 0.680003 | 0.680300 | 0.681043 |
| 2 | 0.758473 | 0.760028 | 0.759901 | 0.760602 |
| **Mean** | **0.728219** | **0.730320** | **0.729900** | **0.730412** |
| **Gain** | — | **+0.002101** | **+0.001681** | **+0.002194** |

At the individual-member level, the real residual improves all 9/9
fold-bank pairs, but its mean gain is only +0.002148. The no-identity control
is marginally stronger than the real-identity branch. Therefore the stable
weak signal is better interpreted as generic connected-support morphology,
not evidence that persistent prototype identity is essential.

## 3. Multi-bank consensus residual

Implementation:

- `code/run_molhiv_multibank_occurrence_residual.py`

This version first forms the frozen three-bank probability ensemble. Within
each bank, prototype rows are canonicalized by occurrence shape without
retaining identity; the three canonical feature matrices are then averaged.
Only one residual is trained on top of the frozen ensemble.

Controls:

- `consensus_no_id`: canonicalized multi-bank occurrence shape;
- `consensus_shuffled`: graph-wise shuffled rows averaged across banks;
- `global_stats`: only prototype-wise means/maxima (386 parameters).

| Fold | Frozen base | Consensus no-ID | Shuffled | Global stats |
|---:|---:|---:|---:|---:|
| 0 | 0.748232 | 0.749856 | 0.752098 | 0.750015 |
| 1 | 0.677951 | 0.679225 | 0.681212 | 0.678744 |
| 2 | 0.758473 | 0.760901 | 0.757211 | 0.759257 |
| **Mean** | **0.728219** | **0.729994** | **0.730174** | **0.729339** |
| **Gain** | — | **+0.001775** | **+0.001955** | **+0.001120** |

The consensus branch improves all 3 folds but is not stronger than its
shuffled control on average. Its learned correction is genuinely small:
held-out correction standard deviation is about 0.045--0.051 logits, and the
largest absolute correction is about 0.066--0.070.

## 4. Predeclared decision

Promotion requirements were:

1. mean scaffold-fold gain at least +0.005;
2. at least 2/3 fold wins;
3. real/canonical occurrence structure stronger than shuffled and no-identity
   controls;
4. no fold-level mean degradation.

The route passes consistency but fails effect size and mechanism specificity:

- best relevant mean gain is only about +0.0022;
- identity-free or shuffled controls are as strong as, or stronger than, the
  intended branch;
- multi-bank consensus does not amplify the signal.

**Decision: do not advance this configuration to official valid or test.**
This protects official valid from being consumed by a candidate that did not
pass its train-only gate. The result does establish that connected-support
morphology is a real but very weak side signal; it is not sufficient as the
next main optimization direction.

## 5. Consequence for train/valid/test use

The practical protocol from this point is:

1. broad architecture and mechanism search on official-train internal folds;
2. promote only a small, predeclared candidate set to official valid;
3. use official valid for finite hyperparameter/model selection;
4. freeze the selected configuration and evaluate test once.

Thus official valid should be used more than test, but less than internal
folds. Previous work treated it unusually conservatively; future candidates
that pass an internal gate should use it explicitly for limited selection.
