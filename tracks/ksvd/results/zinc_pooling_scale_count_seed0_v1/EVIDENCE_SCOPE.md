# EVIDENCE_SCOPE — `zinc_pooling_scale_count_seed0_v1`

What one paired C/N experiment plus the read-only source `M` comparison can and cannot establish.

## This round can say

1. Whether count-normalised mean pooling (`N`) beats the sum/count parametrisation (`C`) under
   identical information, parameters, init, schedule and RNG streams, and whether either beats
   the read-only source `M` reference on the frozen dev endpoints, under the frozen five-item
   gate and paired row bootstrap.
2. Whether the six count coordinates are actually visible and non-constant at runtime (not just
   flag changes), whether the `N` moments are the actual per-graph/per-bucket sums divided by
   the actual counts, and whether the pre-reader block scales changed relative to `C`.
3. Whether a dev gain, if any, co-moves with graph size in the fixed fit-quartile bins, and
   whether it survives dropping the single worst mean-error dev row.
4. Mechanism health of the round arms (block RMS at epoch 0/40/240, reader-block gradient norms
   from normal training, pair-count/moment-scale relation at init).

## This round cannot say

* It cannot establish training robustness: one seed, one soup per arm; the dev bootstrap covers
  row-level uncertainty only, never initialisation/training randomness.
* It cannot establish "input information is insufficient", "all aggregation forms are useless"
  or "the bottleneck is located". C and N carry the same information (moments are recoverable
  from each other); this is a scale/function-accessibility comparison, not a new-information or
  added-capacity test.
* It cannot attribute a gain to "only better numerical optimisation": `C→M` additionally exposes
  the count interface directly and `N→M` changes aggregation scale relative to the source
  parametrisation, so any increment is an interface package, not isolated optimisation.
* It cannot measure counterfactuals at other pooling scales (no scan), other folds, other seeds
  or official-valid/test; it does not validate a deployable `y` prediction.
* It cannot revise the source round's conclusions about the local tuple interface, the
  dictionary/MLP question, ring/cycle tails or luyin19's structural-property claims.

## Decision use

If `N` beats both `C` and `M` at the frozen bar, the round supports buying one frozen
confirmation round later; no official-valid/test read happens in this round. If only `C→M` (or
`N→M`) passes, the count-accessibility part is retained as a candidate and the normalisation
part is not bought. `NO_CANDIDATE` closes this small aggregation variant as specified; it does
not close pooling research in general.
