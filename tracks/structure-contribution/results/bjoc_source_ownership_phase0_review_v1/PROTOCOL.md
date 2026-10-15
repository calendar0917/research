# Review plan — independent re-analysis of `bjoc_source_ownership_v1` Phase 0

Scope: **read-only re-analysis** of the committed Phase-0 results.  No model
training, no GPU, no supervised labels, no official test / Locked access, no
re-selection of structures, models, samples or thresholds.  The original
namespace (`results/bjoc_source_ownership_v1/`) is never modified; this
directory holds the corrected formulas, recomputed values, statistical units
and the review verdict.

Original frozen numbers under review (from
`results/bjoc_source_ownership_v1/phase0_results.json`):
`IG−dev +0.00665 [0.00484, 0.00856]`, `IG−rare +0.03297`, `IG−random +0.05038`,
`matched signed gap +0.010941 [0.002616, 0.019242] (n=366 molecule×seed units,
coverage 0.3125)`, `0C enrichment 6.4528 (p=0.001)`, `sign_stat 0.48259
(p=0.001)`, verdict `SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES`.

## R1 — signed vs absolute matched-control semantics

The frozen intervention returns **signed** responses
`Δ_j = pred(w_j ← natural replacement) − pred_nat`.  The historical matched gap
aggregates `Δ_top − mean_c Δ_ctrl`, i.e. a signed directional contrast, not a
magnitude contrast.

Recompute from the frozen sample/models/RNG/matching rule:

```
signed gap  g_sig(j) = Δ_j − mean_{c∈C(j)} Δ_c            (historical)
absolute gap g_abs(j) = |Δ_j| − mean_{c∈C(j)} |Δ_c|       (new)
per-(seed,row) value  = mean over matched top-4 junctions j of g(j)
```

Report: pooled reconstruction check against the stored `matched_control_gap`
(must be exact); per-seed and per-molecule signed and absolute gaps; direction
consistency `sign(g_sig) == sign(g_abs)`; coverage and dev-decile-exact rate.
Absolute versions for the dev / rare / random contrasts are reported as
secondary evidence on the same recomputed fits.

## R2 — statistical unit (256 molecules observed twice)

The historical pooled tests treat the 512 molecule×seed units as independent.
Re-analysis adds:

* **per seed** (256 molecules each): paired bootstrap over molecules + one-sided
  exact sign test;
* **cluster bootstrap** over the 256 molecule identities (a drawn molecule
  contributes both of its seeds together; a molecule with both seeds contributes
  two values);
* **conservative row unit**: `d̄_row = mean over available seeds` of the
  per-(seed,row) contrast; bootstrap and exact sign test over rows (and a
  both-seeds-complete variant);
* reproduction of the historical pooled-number bootstrap for comparison.

Bootstrap n = 5000, rng seed 0 (as the original); all CIs are 2.5–97.5
percentiles; sign tests are one-sided exact binomial.

## R3 — Phase-0C permutation null and sign statistic semantics

* Implementation semantics (audited from the frozen code):
  `s_j = sign(v_j − median(v))` with `v` = pooled within-molecule-centered IG
  (`ig_centered`); class agreement = `|mean_j s_j|`, weighted by class sample
  size over classes with `sample_count ≥ 30`.
* Reported semantics (PROTOCOL.md): `s_j = sign(centered IG_j)`.
  Both are recomputed and reported; `median(v) = 2.7116e-4` (non-zero).
* Original null: independent within-`(seed, row)` permutation per seed, which
  **breaks the same-molecule cross-seed pairing**.  Corrected null: one
  permutation per molecule applied **jointly** to both seeds' junction positions
  (same `j` order across seeds is verified), preserving the cross-seed
  dependence of every junction.  Enrichment and sign statistics are recomputed
  under both nulls (1000 draws; original rng 20261015, corrected rng 20261016).

## R4 — pre-declared review decision rule

* **P1 (IG beyond encoder-deviation baseline)**: conservative per-row
  `IG−dev` mean > 0, row-bootstrap CI95 lower > 0, one-sided row sign
  `p < 0.01`.
* **P2 (absolute matched advantage under matchable structure)**: conservative
  per-row absolute matched gap > 0, row-bootstrap CI95 lower > 0, one-sided row
  sign `p < 0.05`.
* **P3 (0C under the corrected unit/statistic)**: enrichment joint-permutation
  `p < 0.05` **and** reported-semantics (`sign(centered IG)`) sign statistic
  joint-permutation `p < 0.05`.

Verdict:

```
SOURCE_SIGNAL_CONFIRMED_WITH_CORRECTED_STATISTICS  iff P1 ∧ P2 ∧ P3
SOURCE_SIGNAL_INCONCLUSIVE                         iff ¬P1 ∨ (¬P2 ∧ ¬P3)
SOURCE_SIGNAL_PARTIALLY_SUPPORTED                  otherwise
```

These are review labels only; the original preregistered verdict is not
retroactively changed.
