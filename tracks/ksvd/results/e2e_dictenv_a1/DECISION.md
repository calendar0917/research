# E2E-DictEnv-A1 — decision

**Round** E2E-DictEnv-A1 · study `zinc-context-gap` · protocol `e2e_dictenv_a1` ·
branch `exp/e2e-dictenv-a1-attributed-dictionary` · preregistration `bd3700a` ·
implementation `1ac429f` + `37324eb` + `17017125` · formal-run commit `37324eb` ·
remote NVIDIA A100-SXM4-40GB, **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`; GPU0
foreign-owned all round) · run tag `a1-full` (pid `3802491`).

`official_test_loaded = false` in every artifact · no sweep · seed 0 only.

---

## Frozen verdicts

| gate | frozen verdict |
|---|---|
| Gate 0.1 — correctness (19 gates `G0a`…`G0s`) | **PASS** |
| Gate 0.2 — assignment semantics | **PASS** (REAL sensitive 3.62, INDEP invariant 0.0) |
| Gate 0 — dictionary health (3 arms) | **PASS** (30–32/32 atoms, unit-norm, usage Spearman 0.999) |
| **Gate 0.3 — continuity, REAL code-space AUC ≥ 0.70** | **FAIL (0.534375)** |
| Stage 1 — OMP screen | **NOT RUN** |
| Stage 2 — IHT qualification | **NOT RUN** |
| Stage 3 — E2E `G_attr ≥ 0.003` | **NOT RUN** |
| Stage 4 — dense-tied conditional | **NOT RUN** |
| seed 1 (conditional on `G_attr ≥ 0.003`) | **NOT AUTHORISED** |

**Round verdict: `REPRESENTATION_NOT_QUALIFIED`.**

## Decision

**Stop the round at Gate 0. Publish `REPRESENTATION_NOT_QUALIFIED`, spend no
Stage-1/2/3/4 budget, buy no seed 1, and do not reinterpret the failure as a
result about ATTR-REAL vs ATTR-INDEP.**

## Because

1. The pre-registered stop rule for Gate 0.3 is unconditional: `REAL` code-space
   AUC must be ≥ 0.70, measured 0.534375. The gate is frozen before
   implementation and is reported verbatim. No rescue is permitted.
2. The primary scientific question (`G_attr ≥ 0.003`, ATTR-REAL vs ATTR-INDEP)
   therefore remains **unanswered**; the round's contribution is a qualified
   object set (both 433-D coordinates pass correctness/assignment/health and are
   parameter-identical at 109 263) plus a refuted measurement criterion.
3. The failure is attributable to the frozen criterion's population, not
   demonstrated to be a defect of the object: the near stratum is 100 %
   size-equal/root-category-equal/molecule-disjoint with mean size 5.35 in a pool
   whose median size is 6 and in which only 1.07 % of patches have ≥ 10 atoms
   (`gate0_stratum_diagnostics.json`). For such patches attributed-WL identity is
   almost implied by (size, root category), so the criterion ends up rewarding
   representations that *forget* residual detail: INDEP 0.7056 > REAL 0.5344,
   while the chemistry-blind TOPO arm is worst (0.4325). The same criterion
   already failed on the TCCD-v0 object (0.4675), i.e. it has now failed twice on
   objects built from unrelated principles.
4. The REAL object's own K-SVD fit is 3× harder than INDEP's at identical
   parameters (0.5174 vs 0.1701), independently confirming that the real
   structure↔attribute pairing retains heterogeneous detail that the marginal
   statistic averages away — a property the binary WL-identity criterion
   penalises by construction.

## Alternatives rejected

* **Reinterpret 0.5344 as evidence that REAL is a worse representation than
  INDEP** — rejected: the criterion's validity is contradicted by the stratum
  diagnostics (TinyPatch/AUC premise), and INDEP's advantage is exactly what an
  assignment-independent statistic should show on a WL-histogram criterion.
* **Re-run Gate 0.3 with a bigger patch-size floor, a different statistic or a
  relaxed threshold to unlock Stages 1–4** — rejected: that is an
  unpreregistered rescue of a failed frozen gate; it must be a *new*
  preregistration.
* **Proceed to Stage 1/2/3 “for information”, treating G0.3 as diagnostic** —
  rejected: the frozen decision order makes G0.3 a round-stopping gate; running
  it anyway and reporting it would be a silent protocol violation, and the
  stage budget is a finite scientific resource.
* **Buy seed 1** — rejected: authorised only after `G_attr ≥ 0.003`, which was
  never measured.
* **Touch the official test split** — rejected: forbidden by the frozen brief;
  the round stopped before any predictor existed.

## Revisit if

* A new preregistration replaces the continuity criterion as proposed in
  `notes/e2e_dictenv_a1_analysis.md` §5: `≥ 10`-atom equal-size molecule-disjoint
  population, graded Spearman criterion with a paired REAL-vs-INDEP contrast,
  scale-invariant (angular or norm-matched) representation distance, and a
  in-gate sanity check that the statistic separates the chemistry-blind TOPO arm
  from an attribute arm (it does on graded pairs: 0.128 vs 0.287); **or**
* a new preregistration keeps the binary continuity gate but calibrates its
  threshold and population on frozen artifacts so that a FAIL is interpretable;
  **or**
* the primary question is re-asked with the Gate-0.3 criterion reported as a
  diagnostic instead of a round-stopping prerequisite.

## Spent budget

One formal run (tag `a1-full`, 1 h 02 m 55 s): cache 58 s, three dictionaries
(TOPO reuse; INDEP 1 687 s; REAL 1 727 s), six OMP caches ≈ 102 s, Gate-0
audits. No Stage-1/2/3/4 compute, no GPU training epochs, no seed 1.

## Evidence

`results/e2e_dictenv_a1/`: `REPORT.md`, `decision.json`, `correctness.json`,
`assignment_semantics.json`, `dictionary_meta_{topo,indep,real}.json`,
`dictionary_health{,_topo,_indep,_real}.json`, `parameter_accounting.json`,
`scaler_meta.json`, `cache_meta_{train,valid}.json`, `continuity_audit.json`,
`continuity_posthoc.json`, `gate0_stratum_diagnostics.json`,
`notes/e2e_dictenv_a1_{preregistration,prior_artifact_audit,implementation,analysis}.md`,
`code/analyze_e2e_dictenv_a1_gate0_stratum.py`.
