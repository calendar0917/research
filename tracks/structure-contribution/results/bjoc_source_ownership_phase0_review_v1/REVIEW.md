# REVIEW — independent re-analysis of `bjoc_source_ownership_v1` Phase 0

Read-only review.  Original namespace untouched; original verdict
`SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES` is **not** retroactively changed.

**Review verdict: `SOURCE_SIGNAL_CONFIRMED_WITH_CORRECTED_STATISTICS`.**
Two quantitative corrections change *how* the Phase-0 claims may be worded
(the matched-control advantage is a **magnitude** advantage, not a signed
directional one; the correct statistical unit is the 256 molecules, not the 512
molecule×seed rows), and one documentation correction is due (the 0C sign
statistic subtracts the pooled median in the implementation).  None of them
overturns the claim.

## Reproduction

The matched-control recomputation rebuilt every per-junction response from the
frozen sample, R3 soups, RNG convention (`70000000 + 100000·seed + row`) and
matching rule: `top4_sets` mismatches 0/512, `top4_response` max diff 0.0,
`random_response` max diff 0.0, `matched_control_gap` max diff **0.0**,
coverage/dev-exact rows mismatched 0/512.  Pooled means, bootstrap CIs and sign
p-values reproduce the stored history **bit-exactly** (`matched signed
+0.010941047963362113 [0.0026161586675416573, 0.01924214848949174], p=0.00199`;
`IG−dev +0.006652427`, etc.).

## 1. Signed vs absolute matched-control semantics

The frozen matched gap `Δ_top − mean_c Δ_c` is **signed**; the baseline
contrasts (`top4_response`) are **|Δ|** means.  Recomputed both from the same
intervention responses:

| unit | signed gap | absolute gap |
|---|---|---|
| pooled 366 molecule×seed, mean [CI95] | +0.01094 [0.00262, 0.01924] | **+0.03582 [0.03008, 0.04212]** |
| positive units | 211/366 (57.7%) | **299/366 (81.7%)** |
| per seed 0 / seed 1 mean | +0.01122 [0.00149, 0.02102] / +0.01065 [−0.00376, 0.02440] | +0.03477 [0.02869, 0.04134] / +0.03695 [0.02736, 0.04744] |
| conservative per-molecule (n=217), mean [CI95] | +0.01094 [0.00127, 0.02080] | **+0.03582 [0.02940, 0.04257]** |
| conservative sign test | 122/217, p=0.039 | **183/217, p=3.5e-26** |

Pooled top-junction |Δ| = 0.0743 vs matched controls 0.0385 (**1.93×**).
Direction consistency between the signed and absolute per-molecule gaps is only
**54.6%** (200/366, p=0.042) — the signed advantage is weak and does not imply
an absolute-sensitivity advantage of the same form; the reverse reading is the
correct one: the **magnitude** advantage is strong, while the signed direction
is noisy.  Coverage is unchanged: junction coverage **0.3125** (seed 0 0.342,
seed 1 0.283; structural upper bound 0.62), dev-decile-exact **0.0734**.

## 2. Statistical unit (256 molecules, each seen twice)

All contrasts were re-evaluated per seed, with a row-cluster bootstrap (a drawn
molecule contributes both seeds), and with the conservative per-molecule unit.

| contrast (|Δ| responses unless noted) | per-seed means | row-unit mean [CI95] | rows + / n | sign p |
|---|---|---|---|---|
| IG − dev | +0.00677 / +0.00653 | **+0.00665 [0.00467, 0.00864]** | 176/256 | 9.4e-10 |
| IG − rare | +0.03315 / +0.03279 | **+0.03297 [0.02809, 0.03789]** | 219/256 | 6.2e-33 |
| IG − random | +0.04851 / +0.05224 | **+0.05038 [0.04526, 0.05586]** | 246/256 | 2.5e-60 |
| matched, signed | +0.01122 / +0.01065 | +0.01094 [0.00127, 0.02080] | 122/217 | 0.039 |
| matched, absolute | +0.03477 / +0.03695 | **+0.03582 [0.02940, 0.04257]** | 183/217 | 3.5e-26 |

Both seeds are individually positive for every contrast (the seed-1 signed
matched CI is the only one crossing zero), so model-repeatability is not the
source of the pooled significance.  The cluster/row unit widens CIs modestly
(e.g. IG−dev lower bound 0.00484 → 0.00467) and leaves every substantive
conclusion intact; the only contrast whose significance is unit-sensitive is
the **signed** matched gap (p=0.039 row-unit, seed-1 CI crosses 0) — another
reason not to lead with it.

## 3. Phase-0C null and sign-statistic semantics

* Implementation (audited): `s_j = sign(v_j − median(v))`, `v` = pooled
  within-molecule-centered IG, pooled median `2.7116e-4` (**not** zero), class
  agreement weighted over classes with `sample_count ≥ 30`.
  Protocol text said `sign(centered IG_j)`.  The two differ marginally:
  `sign_stat` 0.48259 (implemented) vs 0.48099 (reported semantics); both sit far
  above their null p95 (0.059 / 0.060) and both give p at the 0.001 floor.  The
  documentation should be corrected in future rounds; the historical number is
  not silently changed.
* Original null permutes each seed independently within a molecule, which
  **breaks the cross-seed pairing of the same junction**.  A corrected null
  applies one permutation per molecule jointly to both seeds (junction order is
  verified identical across seeds for all 256 rows).  Observed statistics are
  unchanged by construction; the null widens slightly
  (enrichment p95 2.89 → 3.28; sign p95 0.059 → 0.074) and both p-values remain
  at the 0.001 floor.  Per seed: enrichment 3.87 / 4.78, sign 0.451 / 0.507,
  p=0.001 each.  The 1000-draw floor means the p-values cannot resolve below
  0.001; the effect-size gap to the null p95 is the informative quantity.

## 4. Corrected evidence judgment

* **IG beyond the simple encoder-deviation baseline** — confirmed with the
  corrected unit: row-cluster `IG−dev +0.00665 [0.00467, 0.00864]`, 176/256
  molecules, both seeds positive.  The same holds against rarity and random
  subsets, and IG is not a re-selection of the dev ranking (top-4 Jaccard 0.29).
* **Absolute advantage under matchable local structure** — confirmed:
  `+0.03582 [0.02940, 0.04257]`, 183/217 molecules, controls matched on
  signature/degree/ring.  The claim must be phrased as **~2× larger |Δ|
  response of |IG| top-4 junctions than structurally matched controls on the
  natural-replacement intervention**; it is not a claim about signed direction
  (54.6% agreement).  Coverage is 31% of top-4 junctions (dev-decile exact 7%),
  so the statement is conditional on the matchable subset.
* **Signature enrichment and sign regularity** — confirmed under the corrected
  joint null and per seed (enrichment 6.45 pooled / 3.87 / 4.78; sign
  ~0.48 / 0.51), with the caveat that the sign statistic's documented formula
  needs the median note.

## 5. Answer to the one-page question

**What survives:** (a) `|IG|` carries source-importance ranking signal beyond
encoder anomaly / rarity / random selection under a cluster-robust unit;
(b) within structurally matchable junctions the top-|IG| junctions have about
twice the absolute natural-replacement sensitivity of matched controls;
(c) the signature classes are enriched and sign-regular in both seeds, also
under a null that preserves cross-seed pairing.

**What must be re-worded:** the matched-control result is a magnitude result,
not a signed-direction result; pooled 512-unit tests overstate precision modestly
relative to the 256-molecule unit; the 0C sign statistic subtracts the pooled
median (2.7e-4) rather than using zero.

**Is the chemistry/next step worth it?** Yes — with the caveats kept in view:
the signal is encoded-source importance in a frozen nonlocal grouping, the
matched analysis covers ~31% of top junctions (structural bound 62%), and
Phase 2 already showed the sum-ownership relational form does not convert the
signal into predictive gain.  The next round should characterize the enriched
signature classes (ring/bond-type contexts) and test their **cross-molecule
reuse** against these corrected units; it should not treat the signed matched
gap as evidence of directional sensitivity, and should use per-molecule units
in any confirmation design.

Artifacts: `PROTOCOL.md` (pre-declared plan), `review_matched_records.json`
(512 recomputed units with per-junction responses + reproduction check),
`review_results.json`, `review_decision.json`.
