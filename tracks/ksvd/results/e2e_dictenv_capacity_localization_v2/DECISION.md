# DECISION — `e2e_dictenv_capacity_localization_v2`

Date: 2026-09-29 (CPU-only; official ZINC test never loaded)
Preregistration: `notes/e2e_dictenv_capacity_localization_v2_preregistration.md`
(sha256 `c5004e3d2aee…`, implementation freeze commit `dccc149`, harness fix
`cc11654`)
Evidence: `REPORT.md`, `notes/e2e_dictenv_capacity_localization_v2_analysis.md`,
`analysis_tables.md`, `summary.json`, `preflight.json`,
`architecture_fingerprints.json`, `parameter_budget.json`, `init_audit.json`,
`calibration/*`, `screening/decision.json`, `screening/screening_summary.csv`,
`screening/*/result.json` + `curve.csv` + `soup.json`, `full/NOT_RUN.json`,
`mechanism/probes.json`.

## Decision

```text
1. WARM-ADAPTATION PROTOCOL VALIDATED (Phase A).  M0 from the frozen
   CSSD-q1 soup with fresh Adam(lr = 1e-4), 20 epochs:
     soup(1-20) 0.128984   <= 0.1320   (C1 pass)
     mean(16-20) 0.130331  <= 0.1350   (C2 pass)
     epoch20 0.129672, late slope -1.55e-4/epoch   (C3 pass)
   Deltas vs M_start = 0.130028: best -0.000356, soup -0.001044,
   last-5 +0.000303.  The low-rate continuation does not degrade the
   trained representation - it slowly improves it.  The screen is
   therefore authorised, and the v1 failure is attributed to the
   fresh-Adam lr = 1e-3 restart.

2. NO WINNER -> LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED (Phase B stop).
   Top-5 soup over epochs 21-40 (base lr 1e-4 / new-capacity lr 1e-3):
     M0 0.128723 | F 0.128613 (delta -0.000110) | R 0.129422 (+0.000699)
     | G 0.128879 (+0.000156);  last-10: M0 0.130721 | F 0.130971
     (+0.000250) | R 0.133506 (+0.002785) | G 0.130604 (-0.000116).
   S1 (matched delta <= -0.003), S2 (absolute soup <= 0.1270) and S3
   (last-10 delta <= -0.003) fail for all three candidates; S4 (real
   branch usage) passes for all three.  passing = [], winner = None.

3. NO FULL RUN BOUGHT.  full/NOT_RUN.json written; no 320-epoch seed-0
   trajectory; no mechanism probes (they require a frozen winner soup);
   no combination candidate; no second LR protocol; no seeds 1/2.
   Budget spent: 180 / 500 epochs (20 calibration + 4 x 40 screen).

4. CAP-BASE STAYS CSSD-q1 (97 727 params, seed-0 soup 0.130028).  F
   (+29 568), R (+34 400) and G (+30 336) remain screened-only and
   unauthorised for training.  FINAL-CLEAN sparse (0.128499) is still a
   historical reference only; the best screen soup (F 0.128613) is still
   +0.000114 above it.

5. SIMPLE LOCAL WIDENING IS NOT SUPPORTED AS THE ROUTE OFF THE PLATEAU,
   with a stated boundary.  The measured protocol floor (two M0 runs of
   the same configuration differ by -0.00060 at the Top-5-mean level, mean
   |valid delta| 0.00121 per epoch) puts F/G at the floor and R 2-5x above
   it as a small regression; the frozen 3e-3 gate effect size is ~5x the
   floor and no direction reaches a third of it.  Boundary: 40-epoch warm
   adaptation, one frozen size per direction, single seed; from-scratch
   320-epoch training of these blocks was never authorised by the gate and
   remains unspent, not answered.

6. NEXT ROUND: controlled iterative environment composition, as a NEW
   preregistration and not implemented here:
     E_i^(0) = dictionary-semantic environment
     m_i^(l) = Agg_j psi(E_i^(l), E_j^(l), r_ij)
     E_i^(l+1) = E_i^(l) + eta_l m_i^(l)     at most l = 0, 1 (two stages)
   The question is whether the 0.13 plateau comes from one-shot static
   composition itself rather than from any single block's width.  Any such
   round must reuse the validated low-rate/differential adapter and state
   its gate effect size relative to the measured floor.
```

## Because

* The calibration gate passed on every condition with margin, and the control
  *improved* the checkpoint over 20 epochs, so the screen ran in a stable
  baseline neighbourhood — exactly what v1 lacked.
* All three candidates satisfied the residual-augmentation contract: zero
  residual reproduces CAP-BASE exactly, step-0 mean shift 0.00080 / 0.00059 /
  0.00113 (v1: 0.01053 / 0.01948 / 0.00113), finite non-zero gradients on
  every new module, and v1-identical architectures and parameter counts
  (shape fingerprints recorded in the preregistration).
* S4 passed for all three arms with healthy per-epoch new-group gradient and
  update norms (F 0.19/0.011-0.024, R 7.4/1.2-1.4, G 0.054/0.003-0.02), so
  the null result is not a dead-initialisation artefact: the capacity was
  trainable and was trained.
* The frozen gate was two-sided (matched control S1/S3 plus absolute anchor
  S2) precisely so that a candidate cannot pass by comparison against a
  drifting control; here the control does not drift, and no candidate clears
  either side.

## Alternatives rejected

* **Buying the full 320-epoch run for the best screen arm (F, soup
  0.128613).**  Rejected: S1/S2/S3 all fail; the frozen §8 rule authorises a
  full run only on a gate pass, and buying one on the lowest raw soup is
  post-hoc selection on a difference (1.1e-4) below the measured floor.
* **Re-running the screen at another learning rate, horizon or concurrency.**
  Rejected: forbidden as a rescue after seeing results (§10); the single
  differential protocol was the round's only authorised adapter, and the
  validated calibration already shows the ordering is not an adapter artefact.
* **Reading R's regression as "pair composition is definitely bad".**
  Rejected: R is worse than M0 by +0.0007 soup and +0.0028 last-10 at a
  10x new-parameter rate; this shows the tested two-block residual perturbs
  the frozen pair state, not that pair composition capacity is useless.
* **Reading F/G's neutrality as "fusion/readout capacity is provably
  unnecessary".**  Rejected: the tested objects are one frozen size each,
  trained for 40 epochs by warm adaptation on a converged checkpoint; the
  claim is scoped to that regime.
* **Combining F+G or F+R to increase the effect.**  Rejected: combination
  candidates are forbidden this round, and two sub-floor effects do not
  compose into a preregistered claim.
* **Comparing the screen numbers against v1's numbers.**  Rejected: v1 was
  run in a degradation regime; no v1 screening number is carried forward, and
  the v1 late-window "stabilisation" ordering is explicitly retired.
* **Rolling the base back to FINAL-CLEAN sparse.**  Rejected: historical
  reference only; CSSD-q1 is the frozen base and no rollback is authorised.

## Revisit if

A new preregistration tests compositional (multi-stage) environment
contextualisation with the residual-zero contract and the validated
low-rate/differential adapter of this round, and states its gate effect size
relative to the `~6e-4` soup-level floor measured here.  The frozen F/R/G
blocks may be reused as *components* of a stage only if the new round
preregisters that exact composition and its parameter budget; they are not
authorised for training as standalone full runs by this decision.
