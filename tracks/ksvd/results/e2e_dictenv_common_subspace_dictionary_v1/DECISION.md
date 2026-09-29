# DECISION — `e2e_dictenv_common_subspace_dictionary_v1`

Date: 2026-09-29 (CPU-only; official ZINC test never loaded)
Preregistration: `notes/e2e_dictenv_common_subspace_dictionary_v1_preregistration.md`
(sha256 `8e722afbc757…`, implementation freeze commit `5f0f284`)
Evidence: `REPORT.md`, `notes/e2e_dictenv_common_subspace_dictionary_v1_analysis.md`,
`analysis_tables.md`, `summary.json`, `common_subspace.json`,
`zero_training/selection.json`, `training/epoch40_gate.json`,
`training/final.json`, `reusable_structure/*`.

## Decision

```text
1. KEEP FINAL-CLEAN UNCHANGED as the line's reference model
   (C6 + paired node/edge binding + sparse tied-IHT K=32/s=8/IHT-10 +
   full relation).  CSSD q1 is adopted only as a supported *exploratory*
   variant: single seed, task-neutral MAE (0.130028 vs 0.128499), and its
   representation distinctness is not established.

2. THE COMMON-SUBSPACE HYPOTHESIS IS SUPPORTED FOR DICTIONARY STRUCTURE.
   The zero-training probe fires on 3/3 reused dictionaries (q1 selected by
   the frozen A/B/C rule), the epoch-40 gate passes, and the trained
   dictionary is healthier on every frozen metric: 32/32 active atoms,
   N_eff 14.52 -> 21.39, top-5 4.80 -> 3.01, no atom above 0.95, coherence
   max 0.987 -> 0.780, weighted Spec 0.207 -> 0.258, 2.7x more atoms per
   graph.  The old universal triplet was exactly the common-direction atom
   set; atom 6 becomes the most specialized atom (rate 0.010, Spec 0.965).

3. NO ARCHITECTURE ADOPTION, NO FURTHER TRAINING FROM THIS ROUND.
   The remaining gap to the last-round specificity question is a
   matched-capacity control, which this round explicitly did not buy.

4. NEXT ROUND SHAPE (only authorisable by a new preregistration):
   CSSD q1 vs DenseTied at the same 97727 parameters, 3 seeds, identical
   soup protocol, with the usage-structure metrics (effective atoms per
   graph, min coverage, coherence max, top-5) preregistered as a gate.
```

## Because

* **Selection was representation-only and it fired cleanly.**  On all three
  reused C6 soup dictionaries with an identical `Dbar` and only the
  descriptor changed, the q1 residual variant lowered the triplet
  6/24/27 from 1.000 to 0.03–0.40 (condition A on 3/3), dropped top-5 by
  1.06–1.87 (condition B on 3/3) and kept 78.5 % of the centred structural
  variation (condition C).  No target and no valid MAE entered selection.
* **The epoch-40 gate passed on the live model** (`CSSD_CONTINUE`): 0 atoms
  above 0.95 (reference 3), top-5 2.746, `N_eff` 22.73, weighted-Spec ratio
  1.320, `||∂L/∂D|| = 6.6e-2`, projected column min 0.317, and
  `max |Uᵀ D̄⊥| = 4.2e-16`.  Training therefore continued to 320 epochs in
  the same process and optimizer state.
* **The task cost is neutral.**  Soup valid MAE 0.130028 vs the frozen
  FINAL-CLEAN sparse seed-0 soup 0.128499: `+0.001529`, inside the frozen
  neutral band, for +240 parameters (0.25 %).  No claim of task improvement
  is made.
* **Concentration was the common direction, not a coder artefact.**  The
  three and only three atoms with `|cos(atom, u1)| > 0.5` were the universal
  triplet (0.965 / 0.941 / 0.819).  After separation the CSSD dictionary has
  `max |cos(atom, u1)| = 8e-8`; atom 6 loses its DC role and re-specializes
  (Spec 0.965 at rate 0.010) while atom 19 takes over as a moderate
  generalist (0.796, Spec 0.080).
* **Graph-level reuse changed materially and replicates across splits.**
  Effective atoms per graph 6.14 → 16.69 (train) and 6.15 → 16.66 (valid);
  top share 0.278 → 0.133; atoms ≥ 5 % mass 4.01 → 7.67; min atom coverage
  126 graphs vs five never-active RAW atoms.  This is the strongest single
  piece of evidence for a "reusable factor vocabulary" reading.
* **Claim B is not excluded, so the decision stays conservative.**  Task MAE
  is unchanged, the top-5 SMD feature-profile diversity is unchanged
  (Jaccard 0.136 over 32 CSSD atoms vs 0.125 over 27 active RAW atoms), and
  the RAW and CSSD sparse codes remain mutually ~0.97–0.98 linearly
  recoverable.  What improved is *how the dictionary is used*, not a
  demonstrated new information channel.
* **All guardrails held.**  CPU only, one new seed-0 trajectory (1 → 40 →
  320), no IHT-30/100/OMP training, no K/s/λ sweep, no penalty, no new
  feature/relation/fusion, no q3+, no seed 1/2, no DenseTied full run,
  official test never loaded, `docs/luyin/luyin19.txt` untouched, focused
  tests 24/24 (including the bit-equivalence of the training loop with the
  frozen audit loop).

## Alternatives rejected

* **Adopt CSSD q1 as the new FINAL-CLEAN.**  Rejected: task-neutral on one
  seed, +240 parameters, and no matched dense control; the structural gain
  does not by itself justify replacing the reference model.
* **Train CSSD seed 1/2 or a second q to firm up the gain.**  Rejected: the
  round is frozen at one trajectory; seed noise is exactly what the round
  cannot answer, and a new round must price it.
* **Buy a DenseTied run now "to compare like for like".**  Rejected: a
  matched dense run is a second training process (and the historic 0.125563
  is not matched); it belongs to the next preregistration.
* **Report the MAE as an improvement because the reconstruction term is
  tiny.**  Rejected: the soup is +0.0015 worse than the reference and the
  task metric is the only task evidence; the reconstruction term is a
  diagnostic, not a win.
* **Read the usage-structure gain as establishing a new representation.**
  Rejected: the recovery directions (0.9704 / 0.9813) and the unchanged
  profile diversity leave Claim B open.
* **Drop the nine zero descriptors / 65→56 now.**  Rejected: layout frozen;
  recorded only.

## Revisit if

A new preregistration is written with a matched-capacity design: CSSD q1
(97727 params) vs DenseTied at the same parameter count, ≥ 3 seeds, the same
soup protocol, and the usage-structure metrics (effective atoms per graph,
minimum graph coverage, coherence max, top-5 share, atoms above 0.95) as
frozen gates.  Secondary: cross-seed Hungarian alignment of the CSSD atoms to
test whether the new rare specialists replicate; an ablation that removes
`c1` from the binding while keeping it in the reconstruction term to separate
"extra width" from "extra information"; and a q2 arm only if a new
representation-level reason appears (q2 keeps too little centred variation
and is not authorised by this round).  None of these is started by this
decision.
