# cscl-bjoc-signature-conditional-value-v1 — does the fixed Junction signature add information beyond DEV?

**Frozen decision (evaluated on the merged molecule-unit top-4 response contrasts): `INCONCLUSIVE`.**
Gates: `A1 ✓ A2 ✗ A3 ✓ A4 ✓ A5 ✓ A6 ✓ A7 ✗ A8 ✓ | B1 ✗ B2 ✗ B3 ✗ B4 ✗ B5 ✗`.
The point estimate is positive (B−A = +0.00053) but the molecule-unit CI includes 0,
only 46.4 % of molecules are better, the discovery cross-fit is non-positive, the
k=1/k=2 selections are significantly worse, and the specified context pair adds
nothing.  There is **no confirmed independent and stable value** of the fixed
two-bond Junction signature beyond DEV.

## 0. Scientific judgment (read this first)

1. **SIG beyond DEV: not confirmed.** The frozen calibration adds +0.00053 in
   mean top-4 natural-replacement response over DEV-only (0.09140 vs 0.09087),
   95 % CI [−0.00030, +0.00136] over 1000 molecules, less than half the
   molecules improve (464/1000, sign-test one-sided p = 0.99).  Both seeds are
   positive in point estimate but both CIs cross zero.  The in-domain
   molecule-disjoint cross-fit is slightly negative (−0.00024).  At the most
   selective k the residual table is clearly harmful (k=1: −0.00747,
   CI [−0.01098, −0.00406]; k=2: −0.00203, CI [−0.00377, −0.00032]).
2. **Context: no.** `center_degree` + `endpoint_ring_count` do not add value:
   C−B = −0.00043, CI [−0.00092, +0.00006], negative in both seeds
   (−0.00015 / −0.00070), and the context increment does not beat its
   pseudo-control (+0.00008, CI [−0.00043, +0.00059]).  The context pair adds
   vocabulary complexity without measurable gain.
3. **Scale and stability.** Even taken at face value, the B−A effect recovers
   only **7.3 % of the DEV→oracle remaining response gap** (gap = 0.00720) and
   0.6 % of the DEV level itself.  It is not concentrated in one class
   (top driver class share 14.5 %; excluding the top driver still +0.00030) and
   it survives on the scaffold-unseen subset in point estimate (+0.00049), but
   it is not stable across k, is negative on small molecules (−0.00109), and
   does not reproduce in the discovery cross-fit.  This is below the bar for
   continuing the vocabulary as a core explanation object.
4. **Decision.** The fixed two-bond **Junction Signature should not be
   developed further as the core explanation object**.  The defensible reading
   of this round is: under a frozen model, a fixed two-bond signature + fixed
   context + fixed shrinkage calibration carries at most a tiny, statistically
   unsupported increment over the encoder deviation DEV.  The frozen label is
   `INCONCLUSIVE` rather than a positive result; the route is closed with
   evidence, not extended (no new vocabulary, no new radius, no new context
   search).  Research resources should return to representation-learning
   questions.

**What this does not say.** It does not say that DEV is a complete explanation,
that no structural information exists beyond DEV, or that structure types cannot
carry predictive information in another representation.  It refutes only this
fixed vocabulary, this fixed context pair and this fixed calibration rule on
this frozen model.  Official valid is a historically exposed development set,
not a fresh blind test.

---

## 1. Frozen design and reuse (no model inference this round)

* Proposition: conditional on per-Junction DEV, do SIG and two supported
  context variables still carry independent, stable, transferable information
  about the frozen model's internal predictive sensitivity?
* Per-Junction DEV, |IG|, signatures, context and the natural-encoding
  replacement responses `delta_j` are **reused** from the frozen
  `bjoc_signature_transfer_v1` raw artifacts (they were replay-verified against
  Phase 0 bit-exactly).  No checkpoint, soup or GPU is used; no property label
  is read; official test and the internal Locked split are never touched.
* **Reproduction gate R** recomputed the published DEV / SIG / ORACLE / RANDOM
  per-molecule top-4 metrics from the stored arrays: `max|diff| = 0.0`
  (tolerance 1e-9), so the reuse is exactly the historical data.
* Within-molecule percentiles `d_j` (DEV) and `q_j` (|IG|) use the unchanged
  average-rank rule; residual `r_j = q_j − d_j`.
* Arms: `A = d_j`; `B = d_j + R'_τ`; `C = d_j + R'_τ + R'_(τ,κ)`,
  `κ = (center_degree, endpoint_ring_count)`.
* Discovery fit is molecule-capped with seeds as repeats: `R'_τ = n/(n+20)·R_τ`
  for `n_τ ≥ 20` molecules, else 0; `R'_(τ,κ)` uses the same cap with
  `n ≥ 10`, else 0.  The complete table, supports and hashes are in
  `residual_tables.json` (58 signature units at floor, 387 context units,
  127 of them at floor).

## 2. Discovery fit and in-domain cross-fit

* 58 candidate signatures (support 20–233 molecules; 29 positive / 29 negative
  shrunk residuals, range [−0.296, +0.138]).
* Molecule-disjoint two-fold check (even/odd sorted diagnostic rows; tables
  estimated on the other fold, evaluated with |Δ|):
  - pooled `B−A = −0.00024 [−0.00141, +0.00089]`,
  - pooled `C−B = +0.00025 [−0.00077, +0.00127]`,
  - fold A `B−A = −0.00047`, fold B `B−A = −0.00001`.
  The type residual does not generalise across discovery molecules: gate A7
  fails.

## 3. Primary validation result (1000 official-valid structures, 2 seeds, labels never read)

Mean top-4 |Δ| (molecule unit, two seeds averaged per molecule):

| arm | mean | vs A |
| --- | --- | --- |
| A (DEV-only) | 0.09087 | — |
| B (DEV+SIG) | 0.09140 | **+0.00053 [−0.00030, +0.00136]** |
| C (DEV+SIG+context) | 0.09097 | +0.00010 [−0.00079, +0.00100] |
| SIG historical (no DEV) | 0.05430 | −0.03657 |
| RANDOM historical | 0.04483 | −0.04604 |
| ORACLE | 0.09807 | +0.00720 |

* `C−B = −0.00043 [−0.00092, +0.00006]` (fraction positive 0.298).
* Per seed `B−A`: seed 0 `+0.00060 [−0.00031, +0.00148]`,
  seed 1 `+0.00046 [−0.00074, +0.00162]` — same sign, both CIs include 0.
* Gain distribution: 130 molecules unchanged, 464 better, 406 worse; median
  gain exactly 0; positive/negative subgroups are near-symmetric in magnitude
  (+0.0099 / −0.0100 mean).  The positive pooled mean comes from a slightly
  thicker positive tail, not from a broad majority.
* k-sensitivity of `B−A` (pooled, bootstrap CI):
  `k=1 −0.00747 [−0.01098, −0.00406]`, `k=2 −0.00203 [−0.00377, −0.00032]`,
  `k=4 +0.00053 [−0.00030, +0.00136]`, `k=8 +0.00026 [−0.00005, +0.00059]`.
  The residual table re-orders the top of each molecule and damages the single
  most sensitive Junction while giving a small k=4/k=8 gain.
* Ranking quality (per-molecule Spearman vs |IG|, mean ±):
  A 0.637/0.634, B 0.652/0.655, C 0.657/0.660 (seeds 0/1);
  top-4 oracle overlap: A 0.597/0.569, B 0.615/0.591, C 0.618/0.584.
  B/C improve the average ranking slightly, but this does not translate into a
  supported top-4 response gain.

## 4. Controls

* **Conditional pseudo-control** (16 replicates; residuals permuted inside
  (seed, molecule, DEV-quartile) strata, keeping all signature/context labels
  and the DEV marginal): mean pseudo-B gain over A across replicates
  ≈ −0.00018; real B beats the mean pseudo-B by **+0.00068 [0.00002, 0.00133]**
  (gate A6 passes).  This is the only positive robustness signal: the real
  table is better than a within-DEV-stratum shuffle of itself.  It does not
  establish the *level* of the gain: the pseudo arm is a noisy control, not a
  zero-information arm, and the differential is ~1 % of the per-molecule
  response.
* **Concentration** (gate A5 passes): the top positive-gain driver class
  (`C` center, two single C–C bonds, center in ring, 1 ring bond; support 184)
  accounts for 14.5 % of positive gain; excluding it the mean gain is +0.00030;
  88 driver classes appear.  The effect is not a one-class artifact.
* **Subsets**:
  * `scaffold_unseen_discovery` (961): `B−A = +0.00049 [−0.00038, +0.00137]`
    (direction preserved, gate A4 passes),
  * `scaffold_not_in_train10k` (905): `+0.00069 [−0.00018, +0.00156]`
    (reported separately from the previous item; never conflated),
  * size terciles: low `−0.00109 [−0.00245, +0.00023]`,
    mid `+0.00145 [−0.00003, +0.00292]`, high `+0.00126 [−0.00026, +0.00280]`.
    The small-molecule regression is one reason the pooled result is fragile.

## 5. Coverage, context support, fallback

Junction-weighted over 34 520 validation Junctions:

* candidate signature (discovery support ≥ 20): **86.10 %**,
* below-floor discovery signature (support 1–19, bonus 0): **13.19 %**,
* never seen in discovery (bonus 0): **0.72 %** (type-level: 91 of 306
  distinct validation types, 29.7 % of types).
* Arm-B top-4 selections: 75.33 % candidate / 22.45 % below-floor / 2.22 %
  never-seen.  Arm-A top-4 selections: 77.44 / 20.40 / 2.16.  The residual
  bonus does not increase candidate coverage of the selected set.
* Arm-C top-4 selections: 95.8 % carry a known `(τ,κ)` unit, 77.4 % carry a
  floor-passing context unit; only 127/387 context units clear the floor 10.

Coverage is high, so the null is **not** a coverage artifact: the fixed
signature is known for the overwhelming majority of Junctions and still fails
to add supported value.  Naming note: `metrics.json` stores
`junction_fraction_below_floor` / `junction_fraction_never_seen` as type-level
counters (the below-floor counter is vacuous because the discovery "known" set
used there contains all 225 discovery units).  The junction-weighted values
above come from the post-freeze descriptive audit
`post_freeze_descriptive_coverage.json`; no score, gate or decision depends on
them.

## 6. Answers to the four required questions

1. **Does SIG carry independent, transferable importance information beyond
   DEV?**  Not confirmed.  The point estimate is positive but not
   distinguishable from zero at the molecule unit; it does not reproduce in
   the discovery cross-fit; it is harmful at k=1/k=2.  The fixed two-bond
   signature is essentially redundant with DEV for this frozen model.
2. **Do center degree and endpoint ring environment add substance or only
   vocabulary complexity?**  Only complexity.  Every context gate fails and
   the increment is negative on the primary mean and in both seeds.
3. **If there is an increment, how large, over which structures, and is it
   stable enough to justify the next structural-explanation step?**  At most
   ~7 % of the DEV→oracle remaining gap, ~0.6 % of the DEV level, no
   concentration in one class, but statistically unsupported, seed-uncertain,
   k-unstable and negative on small molecules.  Not enough to justify a
   continuation round of the same kind.
4. **Should the fixed Junction Signature explanation-algorithm route be
   formally stopped in favour of high-performance molecular representation
   learning?**  Yes — under this round's pre-registered evidence the fixed
   vocabulary, fixed context and fixed calibration have no confirmed
   conditional value, and the research decision is to stop developing them as
   the core explanation object and return to representation-learning
   questions.  This is a stop decision about this object class, not a claim
   that all structure-type information is replaceable by DEV.
5. **Did this analysis change our judgment about "structure-type priors +
   context modulation"?**  Yes, partially and negatively: the hypothesis that
   a fixed two-bond signature plus the two supported context variables can be
   a reusable cross-molecule structural importance prior is now unsupported on
   this frozen model.  Continuing that exact hypothesis would require a new
   pre-registration with a materially different object (e.g. a learned local
   representation), not another vocabulary or context search — and no such
   round is started or recommended automatically here.

## 7. Limitations

* Official valid is a historically exposed development set; this is a
  controlled validation on an existing development domain, not blind testing.
* All per-Junction quantities are predictor-internal; no chemical causality,
  chemical naming or whole-molecule explanation is claimed.  `|Δ|` is the
  frozen model's response to a natural-encoding replacement, not ground-truth
  importance.
* The conditional pseudo-control preserves per-DEV-quartile residual
  multifrequency and destroys signature–residual association, but it is not a
  full confound control and its noise level is not zero.
* The decision vocabulary has the pre-registered asymmetry: a positive point
  estimate with a CI crossing 0 is `INCONCLUSIVE`, so this report does **not**
  upgrade the effect; neither does it use the failure to re-open vocabulary,
  radius or context searches.
* Only the two pre-fixed context variables were tested; no claim is made about
  other structural contexts.

## 8. Decision and authorization

* Frozen verdict: `INCONCLUSIVE` → **no confirmed conditional value**.
* The fixed two-bond Junction Signature line is closed as a candidate core
  explanation object.  No new vocabulary, radius, motif, context variable,
  weight, shrinkage or gate change is authorized by this round.
* The only legitimate follow-ups would be a new pre-registration on a
  materially different object class (e.g. a learned, permutation-invariant
  local representation evaluated as an explanation prior) or a return to
  representation learning.  Nothing is started here.

## 9. Reproducibility, integrity and artifacts

* Freeze: `protocol_freeze.json` (git revision `ef72b47`, config/code/test
  hashes, input hashes, environment).  A recorded pre-validation amendment
  documents the loader-key fix and the re-freeze before the validation
  comparison (`PROTOCOL.md` §10).
* Commands (CPU only, no GPU, no model inference):
  `verify-inputs → freeze → prepare → residuals → pseudo → crossfit → analyze → finalize`
  via `uv run python tracks/structure-contribution/code/run_bjoc_signature_conditional_value_v1.py`.
* Recorded deviations: the analyze loader fix above; the coverage-name audit
  is post-freeze and descriptive only.
* Historical integrity: `historical_integrity.json` re-hashes the frozen prior
  artifacts against the previous round's `finalize.json` and the freeze-time
  input audit; the reproduction gate proves the reused arrays are the
  historical ones.  `finalize.json` records the final artifact hashes.
* Unit tests: 18 fast tests (`tests/test_bjoc_signature_conditional_value_v1.py`)
  cover the conditioning math, molecule cap, shrinkage floors, context
  residual, pseudo-control invariants, arm scores/ties, molecule-unit
  statistics, the four-label decision tree and the label/freeze discipline.
* Deliverables: `PROTOCOL.md`, `metrics.json`, `validation_analysis.json`,
  `residual_tables.json`, `discovery_junctions.npz`,
  `validation_signatures.npz`, `pseudo_tables.npz`, `crossfit.json`,
  `reproduction_check.json`, `row_metrics_v2.npz`, `final_decision.json`,
  `post_freeze_descriptive_coverage.json`, `finalize.json`, and this report.
