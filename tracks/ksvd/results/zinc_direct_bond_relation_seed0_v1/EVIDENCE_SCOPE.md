# EVIDENCE_SCOPE — `zinc_direct_bond_relation_seed0_v1`

What this single O/T pair (plus a read-only use of the source `M` soup) can and cannot establish.

## This round can say

1. Whether exposing the direct-bond type as a primitive per-pair input (T) improves the internal
   chemical residual `g = y − c` relative to its zero-matched control O (same 15-D relation,
   four zero columns), under the inherited source recipe, on the frozen fresh fold, seed 0, on
   the frozen dev endpoints, with the paired row bootstrap and the three-item calibrated gate.
2. Whether beta is exactly the train cache's `pair_relation[:,19:23]` and is exactly re-derived
   from raw train-only `edge_index`/`edge_attr` (all 10,000 graphs, three bond classes, adjacent
   pairs 249,279), one-hot and zero for non-adjacent, batched/ordered consistently, and that the
   beta path is isolated from environments (`E` invariant to a beta mutation) and composed once.
3. Whether T's new relation columns receive non-zero task gradient while O's are exactly zero,
   and whether replay of the raw soup reproduces stored predictions within `1e-5`.
4. Descriptive dev error movement by k-group (k=0/-1/-2/≤-3), the dominant contribution, and
   worst-row deletion sensitivity. No mechanism attribution is claimed.

## This round cannot say

* It cannot establish training robustness: one seed, one raw soup per arm; the row bootstrap
  covers row-level uncertainty only, never initialisation/training randomness.
* It cannot establish "direct-bond information is useless", "the bottleneck is here", or that no
  architecture can use this signal. Beta is a label-free input feature, not a capacity or
  representation argument.
* It cannot attribute a calibrated shift to "better raw prediction" vs a bias offset: each arm's
  sole offset is the fit median, so calibration can move without a raw improvement.
* It cannot measure counterfactuals (other schedules, seeds, folds, bond featureisations, or
  official-valid/test); it does not validate a deployable ZINC-y prediction.
* It cannot revise the source round's conclusions about the local tuple interface, the
  dictionary/MLP question, ring/cycle tails, or luyin19's structural-property claims.

## Decision use

If T passes the frozen calibrated gate (G0 cal and overall cal gains both ≥0.003, G0 CI lower
>0) with CIs inside ±0.003, the round supports a single frozen confirmation; otherwise
`DIRECTIONAL_NOT_CONFIRMED` is recorded and no confirmation is bought. The optional
beta-to-graph-marginal forward diagnostic is permitted **only** after a gate pass; it is a label
intervention on T's soup only, never a trained marginal control. `g` is an internal `y − c`
residual, not official ZINC-`y`.
