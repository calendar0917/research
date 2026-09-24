# E2E-DictEnv-A1 — analysis (Gate-0 stop)

Protocol `e2e_dictenv_a1`, branch `exp/e2e-dictenv-a1-attributed-dictionary`.
Preregistration frozen at `bd3700a`, implementation at `1ac429f` + `37324eb` +
`17017125` (report-stage fix), formal run commit `37324eb`, remote
NVIDIA A100-SXM4-40GB via `CUDA_VISIBLE_DEVICES=1` (GPU0 foreign-owned all
round), official ZINC **test never loaded**.

## 1. Verdict

**REPRESENTATION_NOT_QUALIFIED** (pre-registered Gate-0 stop).

Every Gate-0 component passed except the frozen continuity criterion G0.3:

| Gate-0 component | Result |
|---|---|
| correctness, 19 gates `G0a`…`G0s` | PASS (`correctness.json`, `all_passed: true`) |
| assignment semantics | PASS: REAL sensitive (`max |ΔJ| = 3.62`), INDEP invariant (`0.0`), marginals preserved (`drift 0.0`), `phi` drift `0.0` |
| dictionary health (all three arms) | PASS: 30–32/32 atoms used, unit-norm columns, `train/valid` usage Spearman 0.999 |
| **G0.3 continuity (`REAL` code-space AUC ≥ 0.70)** | **FAIL: 0.534375** |

The pre-registered stop rule is unconditional ("A FAIL stops the round"), so
Stage 1 (OMP screen), Stage 2 (IHT qualification), Stage 3 (E2E) and Stage 4
(dense-tied) were **never run**, and the round's primary question
(`G_attr ≥ 0.003` for ATTR-REAL vs ATTR-INDEP) is **not answered**.

## 2. What the formal run measured

Run: tag `a1-full`, pid `3802491`, command `... zinc_e2e_dictenv_a1 all --device cuda`,
log `~/.research_runs/a1-full.log`, wall clock 1 h 02 m 55 s (01:47:28 → 02:50:23
remote time; cache → Gate-0 stop).
All object caches are bit-identical to the local smoke
(`phi` train `f8641dab…`, `joint_v` `206f4958…`; valid `33db50b8…`), and the
`TOPO` dictionary reproduces the audited SDB-v0 sha `b0c5da98…` — the frozen
objects and the frozen reuse path are exact.

| stage | evidence | key numbers |
|---|---|---|
| cache | `cache_meta_{train,valid}.json` | 10 000 mol / 231 664 nodes (53.1 s), 1 000 mol / 23 083 nodes (5.2 s), 0 zero-edge patches, 231 664 occurrence joins checked |
| scaler | `scaler_meta.json` | train-only, no mean subtraction; masked zero-RMS coordinates S 9 / V 86 (REAL) and 9 / 77 (INDEP) / E 26 (REAL) and 22 (INDEP) |
| dictionaries | `dictionary_meta_*.json` | TOPO reuse (sha `b0c5da98…`); INDEP fit 1 687.3 s, final fit MSE 0.17009, sha `400821ee…`; REAL fit 1 726.5 s, final fit MSE 0.51739, sha `c1cafb08…` |
| OMP codes | `[omp]` lines | 6 caches, 31.9–32.3 s (train) / 3.2 s (valid) each |
| health | `dictionary_health.json` | rec(holdout) TOPO 1.41e-05, INDEP 2.11e-02, REAL 1.97e-01; effective atoms 4.03 / 18.59 / 19.99; top-1 mass 0.679 / 0.149 / 0.134 |
| accounting | `parameter_accounting.json` | TOPO 97 487, INDEP 109 263, REAL 109 263, INDEP ≡ REAL exactly, all in budget |

Continuity audit (`continuity_audit.json`, seed 20260933, 1 500-patch
attributed-WL pool, 3 WL rounds, 800 near / 800 matched-control pairs):

| arm | x-space AUC | code-space AUC | graded-tail x | graded-tail code |
|---|---|---|---|---|
| TOPO | 0.466875 | 0.432500 | 0.54625 | 0.570625 |
| INDEP | 0.637500 | 0.705625 | 0.781875 | 0.837500 |
| **REAL (gate)** | 0.560625 | **0.534375** | 0.675625 | 0.658750 |

## 3. Diagnostic — the frozen stratum, not the object, is what failed

`gate0_stratum_diagnostics.json` (local, `code/analyze_e2e_dictenv_a1_gate0_stratum.py`,
label-free, train-only, 3.4 s) rebuilds the identical pool/strata (near-pair WL
cosine min `0.9999999999979995` and mean `0.9999999999980003` reproduce the
remote values to the last digit) and characterises them:

* the pool is dominated by **tiny patches**: mean size 6.07, median 6, and only
  **1.07 %** of pool patches have ≥ 10 atoms;
* the near stratum is 100 % size-equal, 100 % root-category-equal, 100 %
  molecule-disjoint, mean size 5.35 (range 3–9);
* the frozen control is a genuine, harder-negative control (WL cosine mean
  0.895, min 0.566) and is matched on (size, root category) — the audit's
  *design intent* is intact;
* 2 511 of 1 117 515 distinct-key pool pairs exceed cosine 0.9999 (no exact ties),
  so the near-800 is not a tie artefact, and the isomorphic control is exact
  (same slot key, cosine ≈ 1.0, mean patch size 5.24).

The measured failure mode is the opposite of the metric's premise:

| arm | near code distance | control code distance | near code norm | control code norm |
|---|---|---|---|---|
| TOPO | 1.862 | 1.148 | 5.002 | 4.976 |
| INDEP | 1.551 | 2.209 | 1.666 | 1.700 |
| REAL | 3.917 | 3.087 | 2.739 | 2.100 |

For REAL, attributed-WL-identical patches are **farther** apart in both spaces
than (size, root-category)-matched random patches (`x`: 1.295 vs 1.141). The
same holds under a scale-invariant angular distance (REAL paired AUC 0.531,
INDEP 0.736, TOPO 0.481) and when restricting to norm-matched pairs (REAL 0.527,
n = 187), so this is not a code-norm artefact. Within equal-size,
molecule-disjoint sampled pairs, the *graded* relation is nonetheless positive
for both attribute arms: Spearman(WL cosine, −x-distance) REAL 0.287, INDEP 0.281,
and chemistry-blind TOPO only 0.128 — i.e. the attribute channels do track graded
chemistry, while the *binary* WL-identity stratum does not reward them.

The paired statistic reproduces locally as 0.53625 (remote 0.534375); the 3 of
1 600 flips are float32 norm-rounding between machines. The direction of every
conclusion is unchanged.

Interpretation. A 3–6 atom rooted patch carries almost no WL content beyond
(size, root type), so "WL-identical" is a nearly vacuous predicate there and the
residual variation is dominated by rare, atypical attribute detail that the
REAL object legitimately encodes. The frozen criterion therefore measures, on
this population, whether the representation *forgets* detail — which is why the
chemistry-marginal INDEP object (WL-cosine AUC 0.7056) outranks the real-pairing
REAL object, and why the chemistry-blind TOPO object is worst. The round's own
K-SVD fits show the same asymmetry in a second, independent way: the REAL object
is 3× harder to represent sparsely (final fit MSE 0.5174 vs 0.1701 for the
parameter-identical INDEP object), consistent with the real pairing retaining
heterogeneous interaction detail that the assignment-independent marginal
statistic averages away.

Comparison with the prior round's identical machinery on the TCCD-v0 object
(`tccd_v0` audit, code AUC 0.4675 on a 714-D canonical coordinate): the same
criterion has now failed twice, at 0.47 and 0.53, on objects built from
different principles. That is evidence about the *criterion*, not about the two
objects.

## 4. Non-claims

This round does **not** claim:

* anything about `G_attr` for ATTR-REAL vs ATTR-INDEP, the round's primary
  question — Stages 1–4 never ran;
* that ATTR-REAL is worse than ATTR-INDEP as a representation — the only
  measured difference is on a criterion whose validity is refuted by §3;
* anything about absolute MAE, sparse-vs-dense specificity, or mechanism
  (zero / node-shuffle / edge-shuffle / combined) for A1;
* any test-split result — the official ZINC test split was never loaded
  (`official_test_loaded: false` in every artifact).

The objects themselves are qualified and reusable: both A1 coordinates pass the
assignment-semantics, invariance and health gates, the REAL/INDEP parameter
counts are exactly equal (109 263), and the frozen TOPO dictionary is
bit-identical to SDB-v0.

## 5. Proposal for the next pre-registration (not authorised by this round)

No rescue of A1 is permitted: the verdict is frozen. A future round should fix
the *criterion* before spending GPU time, and should not gate its primary
question behind it twice. Concretely:

1. **Population**: restrict the continuity audit to patches with ≥ 10 atoms
   (≥ 10 % of the pool, or sample deliberately at typical size) where attributed
   WL content is informative; report the tiny-patch regime separately as a
   descriptive stratum.
2. **Statistic**: replace the binary near-vs-control AUC with a *graded*
   criterion on equal-size, molecule-disjoint pairs (Spearman between attributed-WL
   similarity and representation similarity) plus a **paired REAL-vs-INDEP
   contrast** on the same pairs (the scientific question is the difference, not
   an absolute 0.70 bar), and use a scale-invariant (angular or norm-matched)
   representation distance.
3. **Sanity checks inside the gate**: the criterion must separate the
   chemistry-blind TOPO arm from an attribute arm on graded pairs (it does:
   0.128 vs 0.287) — a gate that cannot make this distinction is not informative
   for a structure↔attribute question.
4. **Ordering**: keep correctness/assignment/health as hard gates, but make the
   continuity criterion a *reported diagnostic* for the primary Stage-3 contrast
   rather than a round-stopping prerequisite, or (if it must stop the round)
   calibrate its threshold on the frozen objects at a size/power that makes a
   FAIL interpretable.

## 6. Closure

Round closed at Gate 0 with frozen verdict `REPRESENTATION_NOT_QUALIFIED`;
Stage 1/2/3/4 budget unspent; seed 1 not authorised (the seed-1 rule was
conditional on `G_attr ≥ 0.003`); no K/S/IHT/dictionary/reader/λ/horizon change
authorised; official test untouched. Evidence in
`results/e2e_dictenv_a1/` (`correctness.json`, `assignment_semantics.json`,
`dictionary_{meta,health}_*.json`, `parameter_accounting.json`,
`continuity_audit.json`, `continuity_posthoc.json`,
`gate0_stratum_diagnostics.json`, `decision.json`, `REPORT.md`).
