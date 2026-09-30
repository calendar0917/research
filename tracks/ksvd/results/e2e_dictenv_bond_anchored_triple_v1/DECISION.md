# Decision — BondAnchoredTriple-v1

- run id: `20260930-143250-4edd72a2`
- verdict: **FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL**
- `M_soup = 0.123002916`, `M_parent = 0.123704928`, `Delta = -0.000702012`

The run is complete and valid but `M_soup > 0.120`.  This closes the round: there is no strong 80-epoch signal for the frozen three-environment static composition.  No rescue (width, initialisation, shuffle, seed 1, extended horizon) is authorised; a future step needs a new pre-registration.

## Forbidden without a new pre-registration

- seed 1/2/3 of this candidate, or a rerun with changed width/init/normalisation;
- any third-environment/code/atom/semantic shuffle, mechanism or ablation arm;
- a 320-epoch or otherwise extended horizon, or a changed 0.120 gate;
- official test access (never loaded in this round).
