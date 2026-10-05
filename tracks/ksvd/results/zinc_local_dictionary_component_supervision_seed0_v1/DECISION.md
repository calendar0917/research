# DECISION — zinc_local_dictionary_component_supervision_seed0_v1

**Verdict: `DIRECTIONAL_NOT_CONFIRMED` — the direction favours the matched MLP;
the fixed local IHT dictionary recipe is closed in this interface.**

1. The dictionary-candidate gate failed all five conditions (G0 cal gain
   −0.002045 CI [−0.005525, +0.001502]; overall cal −0.001138 CI [−0.004850,
   +0.002301]; G0 raw −0.008939 CI [−0.012561, −0.005426]; overall raw
   −0.007946 CI [−0.011362, −0.004266]). The raw deficits are statistically
   unambiguous; the calibrated deficits are directional with CIs crossing zero.
   Sensitivity (drop M's worst dev row) keeps the direction (G0 cal gain
   −0.001411). The swapped-role M gate passes only its raw conditions, failing
   the +0.003 cal thresholds — so neither `DICTIONARY_CANDIDATE` nor
   `MLP_SUPPORTED_IN_THIS_INTERFACE` (nor `LOCAL_EQUIVALENCE`) is warranted.
2. The dictionary mechanism is alive (8-sparse codes, 61/64 atoms, drift, live
   task gradients, injection RMS 0.415) — this is a genuine performance
   negative, not `MECHANISM_FAILED`.
3. **Stop**: no rescue, no second seed, no sparsity/IHT-step/width/loss tuning,
   no marginal-control arm, no new fold; the fixed local IHT tuple dictionary
   recipe is closed under this interface (same J incidence, same skeleton,
   same COMP supervision, matched parameters). No full-train/official-valid
   confirmation run (that purchase required D to win).
4. **Keep**: the COMP + matched-local-MLP recipe (M_COMP) as the working
   internal reference for the body line (best internal dev G0 cal 0.088062 —
   an internal-fold chemistry-component score only). The difference vs the
   dictionary is concentrated in the s (residual chemistry) component
   (−0.006832 dev raw), slightly offset by ell (+0.000811).
5. **Next question** (luyin19's actual subject, not this interface): how the
   structure/semantics fusion itself is built — structure-attribute crossing —
   and the body error channel; carried under a fresh preregistration with
   M_COMP-style local encoder as the matched reference and a train-only
   purchase gate of the same strength. No dev-driven tuning of either arm.

## NOT claimed

- Component supervision "revived" the dictionary, or any supervision×encoder
  interaction (would need the g-supervised pair under the identical new fold —
  not run here).
- All dictionaries worse than all MLPs, or anything beyond this fixed
  125-D-tuple/J-incidence/seed-0/single-fold interface.
- J vs marginal aggregation (both arms use J; the frozen J→I switch is
  converged-state sensitivity only).
- Any official-valid/test y-MAE claim (internal fold g-component only).
