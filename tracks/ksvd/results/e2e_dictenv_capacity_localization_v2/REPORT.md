# REPORT — `e2e_dictenv_capacity_localization_v2`

Repaired Capacity Localization round on the frozen ZINC dictionary line.
CPU only (16 cores / 27 GiB, 4 threads per process), official ZINC test never
loaded.  Base: `CAP-BASE = CSSD-q1`, seed-0 Top-5 soup valid MAE
`M_start = 0.130028`.

Preregistration:
`tracks/ksvd/notes/e2e_dictenv_capacity_localization_v2_preregistration.md`
(sha256 `c5004e3d2aee…`); implementation freeze commit `dccc149`, harness fix
`cc11654`; results `tracks/ksvd/results/e2e_dictenv_capacity_localization_v2/`;
analysis note `tracks/ksvd/notes/e2e_dictenv_capacity_localization_v2_analysis.md`.

---

## 0. What was actually run

| phase | arms | epochs | result |
|---|---|---|---|
| A — calibration | M0, fresh Adam `lr=1e-4` | 20 | `WARM_ADAPTATION_PROTOCOL_VALIDATED` |
| B — screen | M0 / F / R / G, base `1e-4` + new `1e-3` | 4 × 40 | all three candidates `NO_CAPACITY_SIGNAL` |
| C — full run | — | 0 | not authorised: `LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED` |

180 of the 500 authorised training epochs were spent (20 + 160); no from-scratch
run was bought, no mechanism probe was run, no seed 1/2, no combination
candidate, no second learning rate.  Training wall clock 3855 s (~64 min);
R arm dominates (2008 s alone under 3-way concurrency).

Frozen candidates are v1's, unchanged in architecture (same names, shapes,
counts; shape fingerprints in `architecture_fingerprints.json`): F multi-rank
structure-semantic fusion (`H=4`, `d_h=24`, +29 568), R two residual FiLM pair
blocks (hidden 256, +34 400), G gated DeepSets summaries (dim 192, +30 336).
Only the residual-projection scale changed so that the step-0 prediction shift
is ≤ 0.002 and zeroing the residual reproduces CAP-BASE exactly.

## 1. Answers to the preregistered questions

**Q1 — Does `lr = 1e-4` M0 hold the CSSD soup basin?**
Yes.  Calibration: soup (1–20) `0.128984`, mean(16–20) `0.130331`, epoch 20
`0.129672`, late slope `-1.55e-4`/epoch; all three frozen conditions pass.  The
control does not just hold the basin, it slowly improves it (`Δsoup =
-0.001044` vs `M_start`; train MAE 0.0780 → 0.0755).

**Q2 — Was the v1 failure mainly the `lr = 1e-3` warm restart?**
Yes.  Same checkpoint, same seed, same data order: at `1e-3` epoch 1 valid was
`0.1435` and the best over 40 epochs `0.1356`; at `1e-4` epoch 1 is `0.130852`
and the best over 20 epochs is `0.129672`.  v1's degradation (control included)
was the adapter, so its `NO_CLEAR_CAPACITY_LOCALIZATION` and its late-window
"stabilisation" ordering carry no capacity information.

**Q3 — Under the stable protocol, who produced a real absolute improvement
rather than only slowing drift?**
Nobody.  Relative to the matched M0 soup: F `-0.00011`, R `+0.00070`,
G `+0.00016`.  F and G are at or below the measured run-to-run floor
(`~6e-4` at the soup level), R is a small real regression (last-10 `+0.0028`,
window mean 0.132906 vs 0.130802).  The only entity that improves on the frozen
checkpoint is M0 itself (`-0.001305` soup), i.e. the low-rate base adaptation.

**Q4 — Does any candidate both beat matched M0 by ≥ 0.003 and reach ≤ 0.1270?**
No.  S1 fails for all (`-0.00011`, `+0.00070`, `+0.00016`); S2 fails for all
(best candidate soup F `0.128613`, `+0.0026` above the absolute gate).  S3 fails
for all; S4 passes for all (branches demonstrably trained).

**Q5 — Did a short warm signal reproduce from scratch?**
Not applicable: no winner, so the full run was not purchased.

**Q6 — Does a winner still depend on dictionary / semantic binding / relation?**
Not applicable: no frozen winner checkpoint exists, so P1–P5 were not run
(`mechanism/probes.json` = `run: false`).

**Q7 — Does this rule out simple local widening?**
Yes, for the tested regime, and with a stated boundary.  Under a validated
protocol whose own soup-level reproducibility floor is `6e-4`, one extra
bilinear-rank block (F), two extra pair residual blocks (R) and one extra
invariant readout summary (G) move the soup by `|Δ| ≤ 7e-4` and never reach
`0.1270`; the gate's `-0.003` effect size is ~5× the floor and none of the
directions approaches even a third of it.  Boundary: this is a 40-epoch
warm-adaptation screen at one frozen size per direction, single seed; it does
not test from-scratch 320-epoch training of those blocks (the gate, by design,
never authorised it), nor any other width/rank/depth.

**Q8 — Should the next step be controlled iterative environment composition?**
Yes.  With the adapter repaired and each single-block widening direction
measured as neutral or mildly harmful, the remaining unexamined hypothesis is
the one-shot *static* composition itself.  Recorded only, not implemented this
round:

```text
E_i^(0)   = dictionary-semantic environment
m_i^(l)   = Agg_j psi(E_i^(l), E_j^(l), r_ij)
E_i^(l+1) = E_i^(l) + eta_l m_i^(l)          at most l = 0, 1
```

## 2. Calibration (Phase A, 20 epochs)

| epoch | train MAE | valid MAE | rec (optimised term) | total loss | base grad norm | base update norm |
|---|---|---|---|---|---|---|
| 1 | 0.078033 | 0.130852 | 0.9761 | 0.07936 | 2.698 | 7.05e-3 |
| 5 | 0.077194 | 0.130416 | 0.9761 | 0.09236 | 2.603 | 5.82e-3 |
| 10 | 0.076347 | 0.130261 | 0.9761 | 0.07750 | 2.273 | 6.01e-3 |
| 15 | 0.075690 | 0.132418 | 0.9761 | 0.07704 | 2.591 | 6.14e-3 |
| 20 | 0.075481 | 0.129672 | 0.9761 | 0.07738 | 2.176 | 5.62e-3 |

`soup(1–20) 0.128984` (members 8, 10, 11, 12, 20); `best 0.129672 @20`;
`last5 0.130331`; `late_slope -1.55e-4`.  Verdict
`WARM_ADAPTATION_PROTOCOL_VALIDATED`; C1/C2/C3 all pass.

## 3. Initialisation audit

| candidate | step0 MAE | Δ MAE | mean shift | median shift | max shift | residual-zero identity | new-branch grads |
|---|---|---|---|---|---|---|---|
| M0 | 0.130028 | 0.000000 | 0.00000 | 0.00000 | 0.00000 | exact | — |
| F | 0.130073 | +0.000045 | 0.00080 | 0.00078 | 0.00389 | exact | 18/18 > 0 |
| R | 0.129986 | −0.000042 | 0.00059 | 0.00050 | 0.00378 | exact | 20/20 > 0 |
| G | 0.129941 | −0.000087 | 0.00113 | 0.00114 | 0.00300 | exact | 9/9 > 0 |

Hard bound (`mean shift ≤ 0.002`) satisfied by all; preferred bound (`≤ 0.001`)
satisfied by F and R, G at `0.00113`.  v1 shifts were 0.01053 (F) and 0.01948
(R) — 13× and 33× larger.

## 4. Screen (Phase B, 40 epochs, base lr 1e-4 / new lr 1e-3)

| candidate | params | step0 MAE | best valid | soup (21–40) | Δ vs M0 | Δ vs start | last-10 | Δlast10 vs M0 | branch grad / update | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 97 727 | 0.130028 | 0.128668 @6 | 0.128723 | 0.000000 | −0.001305 | 0.130721 | 0.000000 | — | — | — | — | — |
| F | 127 295 | 0.130073 | 0.129047 @16 | 0.128613 | −0.000110 | −0.001415 | 0.130971 | +0.000250 | 0.019 / 0.006 | ✗ | ✗ | ✗ | ✓ |
| R | 132 127 | 0.129986 | 0.129948 @28 | 0.129422 | +0.000699 | −0.000606 | 0.133506 | +0.002785 | 0.740 / 0.004 | ✗ | ✗ | ✗ | ✓ |
| G | 128 063 | 0.129941 | 0.128998 @6 | 0.128879 | +0.000156 | −0.001149 | 0.130604 | −0.000116 | 0.005 / 0.008 | ✗ | ✗ | ✗ | ✓ |

(`branch grad` = mean over epochs of the new-group gradient norm; R's maximum
was 7.4/1.2–1.4 across the window.)

Frozen gate: S1 `Δsoup ≤ −0.003`, S2 `soup ≤ 0.1270`, S3 `Δlast10 ≤ −0.003`,
S4 finite non-zero branch gradient and update.  Passing: `[]`; winner `None`;
frozen branch 1 applies (`LOCAL_CAPACITY_AUGMENTATION_NOT_SUPPORTED`),
`full/NOT_RUN.json` written.

## 5. Protocol noise floor (measured inside the round)

The M0 calibration run and the M0 screening arm are the same configuration, run
under different CPU concurrency.  Over epochs 1–20 their valid MAE differs by
`mean 0.00121 / median 0.00082 / max 0.00490`, and the Top-5-mean over 1–20 by
`-0.00060`.  No dropout exists and the RNG streams are equal by test, so this is
CPU thread-scheduling float noise, and it sets the resolution of the round: the
soup-level floor is `~6e-4` and the frozen `-0.003` effect size is ~5× it.
F/G deltas sit at or under that floor; R's `+0.00070` soup / `+0.0028` last-10
exceed it.

## 6. Interpretation

* The v1 adapter is repaired: `lr = 1e-4` (and a 10×-higher rate for the new
  capacity parameters only) keeps the trained representation intact while
  still updating it.
* Under that adapter, single-block local widening does not buy accuracy on this
  base: fusion rank, pair-composition depth and invariant readout width are all
  neutral (F, G) or mildly harmful (R) at the frozen sizes; the branches are
  demonstrably live (S4), so this is not an initialisation artefact.
* The honest reading is a scope statement, not a verdict on the mentor's
  direction: `structure-semantic fusion capacity`, `pair-composition capacity`
  and `invariant readout capacity` are each *not the one missing block* that
  lifts this model off the 0.128–0.130 plateau when added alone for 40 epochs.
* The remaining untested hypothesis is compositional: whether one-shot static
  composition (each environment computed once, then pooled) is itself the
  ceiling.  That is the next round's question; this round does not implement it.
* `CAP-BASE` stays `CSSD-q1` (0.130028); the candidates remain screened-only.
  FINAL-CLEAN sparse (0.128499) is still a historical reference only
  (Δ of the best screen arm vs it: `+0.00011` for F — i.e. even the best
  candidate does not reach the historical reference).

## 7. Provenance / hygiene

* Preregistration sha256 `c5004e3d2aee…`; the preregistration file was not
  edited after the first training process.
* CAP-BASE checkpoint sha256 `5fc41ab4…`, subspace sha256 `36636ce9…`, soup MAE
  verified `0.130028` (`|Δ| < 5e-7`).
* `official_test_loaded = false` in every payload; `CUDA_VISIBLE_DEVICES=""`;
  `torch.device("cpu")`; no SSH/remote compute.
* 59/59 focused tests (v2) and 40/40 v1 regression tests pass.
* `docs/luyin/luyin19.txt` untouched.
* Artifacts: `preflight.json`, `parameter_budget.json`,
  `architecture_fingerprints.json`, `init_audit.json`,
  `calibration/{m0/curve.csv,m0/soup.json,m0/result.json,calibration_summary.json,calibration_decision.json}`,
  `screening/{m0,fusion,relation,readout}/…`, `screening/screening_summary.csv`,
  `screening/decision.json`, `full/NOT_RUN.json`,
  `mechanism/probes.json`, `analysis_tables.md`, `summary.json`.
