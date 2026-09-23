# PEC-C1 — decision

Round **PEC-C1** · study `zinc-context-gap` · protocol `pec_c1`.
Remote commit `bf8b5a628d99eedaf52f0b6e95e79cf583c4ba7b` (clean), A100-SXM4-40GB,
seed 0, official valid 1,000, fixed 240 epochs.

**`official_test_loaded = false`** · only `train` / `val` ever read ·
no historical baseline retrained.

> **PEC-v0 Gate 1 remains a historical frozen FAIL. PEC-C1 does not
> retroactively pass or recalibrate it.**

---

## Frozen verdict

```
PURE_ENV_COMPOSITION_ABSOLUTE_WEAK
seed1_authorized = false
```

| | value |
|---|---|
| `M_CK` frozen-SparseDict soup | **0.160971** |
| `M_CD` trainable-DenseRole soup | **0.151767** |
| `Δ_dict = M_CD − M_CK` | **−0.009204** |
| `min(M_CK, M_CD)` | **0.151767** → band `> 0.145` |

Cases fired: `A_absolute_weak` and `B_dense_dominates_sparse`.
Precedence `A > B > D > C > E` ⇒ **Case A**. No seed 1.

## Answers

| question | answer |
|---|---|
| **Q1** absolute viability | **`> 0.145`** — weak. `0.151767` best of the two arms, `+0.0210` worse than the strict-static S0 seed-0 soup (`0.140794`), orientation only. |
| **Q2** sparse dictionary vs parameter-matched dense | **worse** by exactly `+0.009204` MAE (CK 0.160971 vs CD 0.151767). |
| **Q3** mechanism integrity | **intact and decisive** — chemistry-placement shuffle `+1.576`, neutral dictionary `+1.967`, composition relation shuffle `+1.393`; no MP / recurrence / mixed bypass. |
| **Q4** next decision | `PURE_ENV_COMPOSITION_ABSOLUTE_WEAK`, `seed1_authorized = false`. |

## Decision

**STOP.** The round is closed with a clean negative on absolute capacity.

## Because

1. `min(M_CK, M_CD) = 0.151767 > 0.145` fires the pre-registered Case A. The
   pure no-MP environment→static-composition architecture does **not** reach the
   viability band in the full official-train → official-valid regime.
2. The deficit is real in both directions, descriptively split at the best epoch
   (S0 seed-0 train-at-best `0.075434`, gap `0.070241`): CK `+0.023924` =
   fit `+0.015014` (63 %) + gap `+0.008911` (37 %); CD `+0.014108` =
   fit `+0.006084` (43 %) + gap `+0.008023` (57 %). The pure class cannot even
   fit the training set as well as the strict-static backbone.
3. The fixed 240-epoch horizon cannot change the verdict: CK's minimum is at
   epoch 210 and its tail is flat-to-worse; CD's minimum is at 235 and its
   entire last 48 epochs move within `0.003`, while leaving the weak band would
   need `−0.0148`.
4. Case B also fires: a **frozen** K-SVD sparse structural-role dictionary is
   `0.009204` MAE worse than a parameter-matched **trainable** dense role map
   (CK was handicapped by design: its 416 role parameters are frozen, drift
   exactly `0.0`, while CD's 416 are trained). This closes the dictionary route
   **in its frozen form only**; a task-coupled dictionary is not estimated here.
5. The mechanisms are not at fault. Chemistry placement, dictionary dependence
   and composition-relation sensitivity are all strongly load-bearing at full
   scale, so the failure is capacity, not inertness, collapse, or a dead branch.

## Alternatives rejected

* **Buy seed 1** — rejected: neither pre-registered trigger fired. Not
  triggered by "CK is only 0.009 behind", a flattering best checkpoint, a noisy
  curve, or good dictionary usage statistics.
* **Rescue with `K`, `s`, epochs, reader width, recurrence or a threshold
  change** — rejected: explicitly forbidden, and none is needed to explain the
  result.
* **Relax/re-run PEC-v0 Gate 1** — rejected: Gate 1 has no bearing on this
  round, remains a historical frozen FAIL, and is not re-optimised.
* **Run a task-coupled dictionary now** — rejected: the pre-registration makes
  task coupling conditional on a stable absolute pure architecture, which this
  round shows we do not have.
* **Claim `ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT` as the verdict** —
  rejected as the *primary* verdict: Case A outranks it. Case B is recorded
  inside the report, with the frozen-dictionary confound stated.

## Revisit if

* A **new pre-registration** proposes a structurally different route to
  absolute capacity that keeps the purity contract (e.g. a different
  environment-formation function class), **or**
* a `PEC-C2 task-coupling` round is proposed on a *different* architecture that
  first demonstrates absolute viability, **or**
* the research direction is re-scoped away from ZINC absolute MAE and toward
  the mechanistic claims (environment binding / static composition), which this
  round supports decisively and which the strict-static band is not needed for.

## Durable artifacts

`notes/pec_c1_preregistration.md`, `notes/pec_c1_implementation.md`,
`notes/pec_c1_analysis.md`,
`results/pec_c1/{REPORT.md,DECISION.md,decision.json,pec_c1_decision.json}`,
`results/pec_c1/{CK,CD}_seed0{,_curve.csv,_best.pt,_top5.pt}.json`,
`results/pec_c1/dicts/`, `record(s)` for this round, `STATE.yaml`.
