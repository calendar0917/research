# DECISION — LCRC-HC3-v1

## Verdict

**STOP at Stage A. Do not purchase Stage B (seed 1). Do not purchase Stage C
(official train10k / valid). Do not tune, widen, change path length, change the
loss, or add a rescue run.**

This is the pre-registered stop rule, applied to the actual Stage A result.

## Why

Stage A (Fit8k / Monitor1k, seed 0) failed four of the six numbered purchase
gate groups:

| gate | required | observed | result |
|---|---:|---:|---|
| Δ(H−B) full | ≤ −0.004 | **−0.001025** | FAIL |
| Δ(H−F) full | ≤ −0.003 | **+0.000377** | FAIL |
| Δ(H−B) ordinary | ≤ −0.002 | **−0.001612** | FAIL |
| Δ(H−B) unseen-ordinary | ≤ −0.004 | −0.004731 | PASS pointwise, one-row-sensitive |
| Δ(H−B) on Monitor \ fixed W10 | ≤ −0.002 | −0.003326 | PASS |
| W10 share of net AE reduction | ≤ 0.50 | −2.213 | PASS because H is worse on W10 |
| W20(H) − W20(B) | ≤ −0.004 | **−0.001235** | FAIL |
| W20(H) − W20(F) | ≤ −0.003 | **+0.000316** | FAIL |
| T100(H) − T100(F) | < 0 | **+0.000322** | FAIL |
| final(H) − final(F) | ≤ 0 | **+0.000305** | FAIL |

The two primary full-MAE gates fail by large margins, and H is **worse** than
the same-information flat encoder F on the full slice. The only support gate
that passes pointwise is not robust: removing molecule 374 (Monitor position
39) turns Δ(H−B) unseen-ordinary from −0.004731 into **−0.000385**. That row
alone contributes more than the entire full H−B gain; without it H is
**+0.000154 worse than B** and **+0.002163 worse than F** on full Monitor
(post-hoc sensitivity; the pre-registered gates already fail, so this is
explanatory, not a new gate).

## Scientific reading

* **“Hierarchical composition is more effective” is not supported.** H does not
  beat the nearly capacity-matched, same-information flat encoder F. The H−F
  comparison is the clean test of the hierarchy, and its point estimate favors
  F (+0.000377 full; +0.002163 excluding one row).
* **“Mature single-model performance improved” is not claimed.** Stage C was
  not run, so there is no train10k/valid H/F result, no DCC-L comparison, and no
  basis for a mature-performance claim.
* **“Three-bond information broadly helps” is not established.** The nominal
  H−B full gain is −0.001025, below the −0.004 gate, and is one-row-driven.
  H vs B is 512 wins / 488 losses across Monitor rows; the full and ordinary
  confidence intervals include zero.
* The implementation itself is supported by the CPU gates: enumeration,
  chords, centre-bond ownership, reversal/renumbering/batch invariance, shared
  inputs, exact frozen initialisation, bitwise step-0 equality, the prescribed
  zero-init gradient path, label/test boundary, recipe, optimizer,
  checkpoint/resume and parameter counts.

## What would have been required to continue

Not a different result on the same data, and not a post-hoc slice. The frozen
protocol required, before seed 1:

* H at least 0.004 better than B and 0.003 better than F on full Monitor;
* H at least 0.002 better than B on ordinary rows;
* the same directions in `W20`, `T100` and final, with no sign reversal.

Those conditions were not met. Seed 1 and Stage C are therefore not purchased.

## Cost and status

* Stage A cost: **1.911 A100·h** (B 0.487, H 0.691, F 0.730, timing ~0.004),
  below the 6 h soft / 8 h hard caps.
* Stage B: not run.
* Stage C: not run.
* Official test: never opened.
* Task 0 Stage IV support-vocabulary erratum: completed (no GPU), additive,
  historical decision not rewritten.

## Artifacts

* `REPORT.md` — full results, support strata, paired per-molecule comparisons,
  curves, tail/penalty descriptive split, provenance and budget.
* `analysis/stageA_s0.json`, `analysis/stageA_s0.md` — Stage A machine-readable
  and human-readable results.
* `analysis/per_molecule_stageA_s0.csv` — every Monitor row with predictions,
  AE, paired deltas and flags.
* `analysis/gates/cpu_gates.json` — 15/15 CPU gate groups pass.
* `ERRATUM_stage4_support.md`, `analysis/erratum_stage4_support.json` — Task 0.
* `TREE_LEDGER.md` — chronological execution, deviation and budget record.

**Final decision: stop and report the negative result as-is.**
