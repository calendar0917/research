# DECISION — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Scientific status: COMPLETED

Two formal trajectories (SUM, COMP), one seed (0), matched recipe and full-train data/labels
(10,000 rows). The comparison is valid: no dev-driven configuration selection occurred, no
scores were read from failed fragments.

## Verdict

| gate | result |
|---|---|
| CHEM_CONFIRMED | ✅ True |
| DEPLOY_CONFIRMED | ✅ True |
| `y_cal < 0.09` benchmark marker | ❌ Not met (SUM 0.1169, COMP 0.1122) |

**Conclusion:** This full-train confirmation round retains the chemistry signal — COMP improves
both `g` and deployable `y` on the frozen official-valid — but the overall `y`-MAE remains above
the 0.09 benchmark. The chemistry + deployment reference is promoted to the new full-train
baseline, but the absolute `y` bar is a marker for a follow-up, not closed in this round.

## Evidence disposition

* **Train/soup states:** both arms' init/last/raw_soup saved. Q init/soup/last saved.
* **Train replay:** raw and component predictions saved per arm; `b_g` and `b_y` computed from
  train-only medians.
* **Valid:** one frozen read (1000 rows), predictions saved in `valid_frozen_predictions.npz`,
  summary in `valid_summary.json`.
* **Bootstrap:** 1000 draws, seed `20261009`, shared indices, G0/overall per endpoint.
* **Identities:** float64 row-wise deploy identity checks to 3.8e-7 / 4.1e-7; `g = ell + s`
  identity to 4.4e-16; `y = g + c` identity to 0.0.

## Next responsibility

1. Close the `y_cal < 0.09` gap. The `s` component is the largest remaining gap (COMP valid
   L_s = 0.086 vs L_ell = 0.079); a dedicated `s` modeling step is the natural next question.
2. Any follow-up `s` design must keep total `g`/`y` as the final judgment to avoid sacrificing
   the favourable `comp_ell`/`comp_s` cancellation.
3. A follow-up should also test whether the `y` gain survives a different valid/test split, since
   official-valid is reused.

## What this round does NOT purchase

* A new model architecture / capacity sweep
* A different loss coefficient or gradient normalisation
* A cycle-head redesign or Q retraining
* A third body arm
* A seed-1 replication
* official-test evaluation
* A post-hoc rescue by deleting valid rows
