# Decision — BondAnchoredTriple-JointTune-v1

- run id: `20260930-154827-2dc792ad`
- verdict: **JOINT_TUNE_SCREEN_NO_STRONG_SIGNAL**
- `M_joint_soup = 0.123159514`, `M_start = 0.123002916`, `Delta_vs_start = +0.000156598`, `M_parent = 0.123704928` (background)

The run is complete and valid but `M_joint_soup > 0.120`.  This closes the combination candidate: with the shared dictionary coordinates fixed and the representation + composition jointly adapted for 40 epochs, there is no strong validation signal.  No rescue (lr, unfreeze scope, initialisation, normalisation, horizon, seed 1) is authorised; a future step needs a new pre-registration.

## Forbidden without a new pre-registration

- seed 1/2/3 of this candidate, or a rerun with changed lr / unfreeze scope / initialisation / normalisation / soup window / horizon;
- matched control, M0 retraining, shuffle, ablation, branch-off, mechanism arm or dead-node binding rescue purchased off this result;
- official test access (never loaded in this round);
- treating this single performance candidate as evidence about the triple operator's independent effect or the irreplaceability of the three-environment mechanism / dictionary coordinates.
