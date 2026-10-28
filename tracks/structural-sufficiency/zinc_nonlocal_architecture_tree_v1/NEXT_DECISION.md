# NEXT_DECISION — zinc non-local architecture tree v1

**Decision: `STOP`.**

Do not promote, do not buy a seed-1 replication, do not buy a mechanism control, and do not read the
official test for any arm in this round. All three pre-registered hypotheses are RED on the matched
official-valid protocol, and none is within the 0.003 purchase band.

## Evidence (seed 0, matched valid-1000, frozen LCRC-A)

| arm | valid best | Δ vs A_s0 | ordinary Δ | r2-unseen-ord Δ | verdict |
|---|---:|---:|---:|---:|---|
| SCE-Real | 0.082364 | +0.005782 | +0.000459 | +0.002798 | RED |
| GEF-Real | 0.082361 | +0.005780 | +0.004359 | +0.011721 | RED |
| JSSP→A | 0.083658 | +0.007077 | +0.001970 | +0.004565 | RED |

Reference A_s0: valid 0.076582 @476, historical test 0.066782 (already exposed; no new test read).
The two best new arms (SCE/GEF) are +0.0058 worse; SCE's only negative group is seen-ordinary
(−0.00028), while its unseen-ordinary and ordinary means are worse. GEF is worse in every group. Removing valid:0172 or A's fixed worst-10 does not
change the sign. All three arms are active and non-redundant (SCE term R² 0.29, std 0.061; GEF term
R² 0.44, std 0.107; JSSP 7,560 updates, exact historical init hash), so the failure is not an
inactive or collapsed branch.

## Why stop rather than refine

* A seed-1 replication cannot rescue a +0.0058/+0.0071 point deficit under the frozen rule; it would
  only spend ~1.8 GPU·h to estimate the same negative effect.
* The three mechanisms are mathematically unrelated yet converge to the same 0.082–0.084 plateau,
  while A reaches 0.0766. Adding a late non-local scalar channel or shifting the SSL task both cost
  roughly the same amount of late generalisation. There is no evidence that a different λ, rank,
  width, mask ratio or epoch count would reverse the sign; tuning those would be an unregistered
  search and is forbidden.
* No mechanism control was purchased because there is no performance sign to explain. A control
  would not change the promotion decision.
* Official test was never loaded; the test read gate is not met (no candidate with two seeds and a
  consistent ≥0.003 same-seed valid gain). The historical A test number is context only.

## Budget / discipline

2.08 GPU·h used (SCE 0.5947, GEF 0.6738, JSSP pretrain 0.4518, JSSP downstream 0.3562, timing
0.0020) of a 14 soft / 16 hard cap; 11.92 GPU·h unspent. Wall ≈3.5 h of the 7–8 h target. Test never
read; no hyperparameter sweep; no `rr` bypass; at most two concurrent formal jobs. The unused budget
is deliberately returned rather than spent on a direction with no signal.

## What would change this decision

Only a new, pre-registered mechanism class with a concrete reason to improve the **ordinary and
radius-2-unseen** groups at the same time — not a re-parameterisation of SCE/GEF/JSSP and not a
re-tune. Examples would need to address the late-training capacity/optimisation cost directly
(e.g. a parameter-matched local-only change or a training-regime intervention with its own frozen
control), and would require fresh CPU gates plus a matched A reference before any GPU run. Absent
that, the correct action is to preserve the negative result and stop.

## Artifacts

`REPORT.md` (full protocol, tables, probes, seven answers), `DECISION_TREE.md` (chronological
verdicts), `analysis/final_table.{json,md}`, `analysis/probe_trained_{sce,gef}.json`,
`analysis/jssp_pretrain_s0.json`. Runs: `runs/sce-real_s0`, `runs/gef-real_s0`,
`runs/jssp_base-real_s0` (gitignored). Reference: `zinc_local_relation_composition_v1/runs/lcrcA_s0`.
