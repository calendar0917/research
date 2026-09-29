# DECISION — `e2e_dictenv_capacity_localization_v1`

Date: 2026-09-29 (CPU-only; official ZINC test never loaded)
Preregistration: `notes/e2e_dictenv_capacity_localization_v1_preregistration.md`
(sha256 `f5c4a8d2e489…`, implementation freeze commit `becc6f6`)
Evidence: `REPORT.md`, `notes/e2e_dictenv_capacity_localization_v1_analysis.md`,
`analysis_tables.md`, `summary.json`, `preflight.json`, `init_audit.json`,
`screening/decision.json`, `screening/screening_summary.csv`,
`screening/*/result.json` + `curve.csv` + `soup.json`, `full/NOT_RUN.json`,
`mechanism/probes.json`.

## Decision

```text
1. NO_CLEAR_CAPACITY_LOCALIZATION — the round cannot localise the current
   expressive bottleneck among structure-semantic fusion (F), static
   environment composition (R) and invariant graph readout (G).
   Frozen primary metric (Top-5 soup over epochs 21-40):
     M0 0.129662 | F 0.129464 (Δ -0.000198) | R 0.129168 (Δ -0.000494)
     | G 0.129663 (Δ +0.000002).
   Gate (Δsoup <= -0.004 AND Δlast10 <= -0.003) fired for nobody
   (passing = []).  Secondary Δlast10: F -0.004266, R -0.004466,
   G -0.006732 — all pass, none decisive.  Frozen branch 1 applies.

2. NO FULL RUN BOUGHT.  full/NOT_RUN.json written; no 320-epoch seed-0
   trajectory; no 320-epoch budget spent (160/480 epochs used: 4 x 40).
   No post-full mechanism probes (they require a frozen winner soup).
   No seed 1/2, no F+R / F+G / R+G combination, no post-hoc rescue.

3. CAP-BASE STAYS CSSD-q1 (97 727 params, seed-0 soup 0.130028).  The
   candidates were screened, not adopted: F (+29 568), R (+34 400),
   G (+30 336) added parameters all remain unauthorised for training this
   round.  FINAL-CLEAN sparse (0.128499) remains historical reference only.

4. THE LIMITING INGREDIENT IS THE SCREEN, NOT (DEMONSTRABLY) THE CAPACITY.
   Post-hoc, clearly labelled: the fresh-Adam warm restart degrades every
   arm — the control included — from step-0 0.130028 to best 0.135621
   (epoch 21) and 0.156383 (epoch 40); the control's own soup is -0.00037
   from its own start, and the candidate Δsoup values sit at/below that
   noise floor.  The only systematic separation is in the late window
   (31-40), where all three capacities slow the drift by -0.0043…-0.0067
   (largest for G) without improving best valid accuracy (all >= 0.1356).
   This round therefore records a *stabilisation hypothesis*, not a
   bottleneck claim.

5. NEXT ROUND SHAPE (only authorisable by a new preregistration; the
   frozen candidates themselves are not the problem):
   (a) repair the screen first — from-scratch epoch-40 checkpoint gate in
       the style of the CSSD round, or a low-rate warm adaptation
       (lr 1e-4) whose M0 noise floor is measured and preregistered, and
       require an effect size above that floor;
   (b) then re-ask the same question with the frozen F/R/G architectures;
   (c) optionally test the stabilisation hypothesis directly (does added
       capacity slow late-epoch valid drift under an overfitting regime?)
       as a separate preregistered question, never as a bottleneck claim.
```

## Because

* Every candidate triggered the frozen engineering checks: bit-identical
  warm load, finite strictly-positive step-0 gradients (18/18, 20/20, 8/8),
  step-0 MAE shift ≤ +0.0011 (far under the +0.03 bound), budget ratio
  `1.1634 ≤ 1.5`, all totals inside the preferred 120 k–160 k band.
* The gate condition that failed is the primary one, by 8–20×: the soup
  deltas are −0.0002 (F), −0.0005 (R), +0.000002 (G) against a required
  −0.004, while the control's own soup-vs-start difference is −0.0004 —
  i.e. the measured effect is within protocol noise.
* The frozen preregistration forbids any threshold, width, rank, seed,
  epoch or tie-break change after the first screening process started; the
  `STRONG_CAPACITY_SIGNAL` upgrade and the tie-break were never reachable.

## Alternatives rejected

* **Buying the full run anyway** for the best `Δlast10` (G, −0.0067).
  Rejected: §8 requires both gate conditions; a full run is authorised only
  when the gate fires.  Doing it would be a post-hoc selection on the
  non-frozen secondary metric.
* **Re-running the screen with a lower lr or fewer/more epochs.**  Rejected:
  a protocol change after seeing results is exactly the forbidden rescue.
  It belongs to a new preregistration (see decision 5a).
* **Combining F+R or F+G.**  Rejected: explicitly forbidden this round
  (§12), and no candidate was even authorised.
* **Returning to FINAL-CLEAN sparse (0.128499).**  Rejected: historical
  reference only; CSSD-q1 is the frozen base and the representation change
  was already supported at representation level.
* **Reporting the Δlast10 improvements as capacity localisation.**
  Rejected: the frozen primary metric shows no effect; the secondary one is
  a late-window drift difference in an overfitting regime and cannot
  distinguish "more capacity" from "different noise trajectory".
* **Reading G's branch as "readout is not the bottleneck"** because its
  gradient is tiny (0.001).  Rejected: the tiny gradient is an
  initialisation/optimisation property of the near-zero-init summaries in a
  40-epoch warm restart, not evidence about the readout bottleneck.

## Revisit if

A new preregistration provides a discriminative screen (decision 5a);
specifically: measure the M0 soup-vs-start floor and the epoch-to-epoch
valid spread over the screening window under the chosen adapter, and set the
gate effect size above that floor before running anything.  The same three
architectures (exact F/R/G definitions and parameter budgets from this
preregistration) may be reused verbatim in that new round.  The
stabilisation hypothesis (added capacity slows late-epoch valid drift) may
be preregistered as a secondary, explicitly non-bottleneck question with a
slower/longer adaptation where drift is the measured phenomenon.
