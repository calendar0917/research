# cscl-bjoc-signature-conditional-value-v1 — Preregistered Protocol

**Status:** PREREGISTERED before any validation-set comparison of this round is
computed.  All method, statistic and decision rules below are frozen (see
`protocol_freeze.json` for the config/code hashes and the git revision).

**Track:** `tracks/structure-contribution` · **Namespace:** `bjoc_signature_conditional_value_v1`
**Compute:** CPU only.  No model inference is run this round.

**Single proposition under test (the only one):**

> Given the frozen model's per-Junction encoder deviation DEV, does the existing
> two-bond Junction signature SIG, together with the two already-supported
> context variables (center_degree, endpoint_ring_count), still carry
> **independent, stable and transferable** information about the frozen
> predictor's internal predictive sensitivity?

**Explicitly not under test:** chemical causality, chemical contribution values,
whole-molecule explanation, new models/GNNs, motif or radius search, vocabulary
expansion, R3 re-tuning, supervised property labels.  A null result refutes only
the fixed vocabulary, the fixed context pair and the fixed calibration below —
it can never prove that DEV is a complete explanation, nor that no structural
information exists beyond DEV.

---

## 1. Frozen evidence base (read-only reuse)

Per-Junction quantities are **reused**, not recomputed by inference, from the
frozen previous round (`results/bjoc_signature_transfer_v1/`), whose provenance
was replay-verified against Phase 0 (bit-exact IG, top-k sets and natural
replacement responses over 256×2 diagnostic molecules):

* `dev_j` = `||w_j − mean_G(w)||₂` (frozen Phase-0 encoder deviation),
* `ig_abs_j` = `|IG_j|`,
* `delta_j` = `pred(w_j ← w_ref[j mod n_ref]) − pred(natural)` with α = 1.0
  (natural-encoding replacement response),
* `signature τ_j` = `(center_type, sorted(endpoint_types), sorted(bond_types),
  center_in_ring, n_ring_bonds)`,
* stored candidate code `sig_code`, degree and endpoint-ring context.

The reuse is guarded by **reproduction gate R**: this round recomputes the
published per-molecule DEV / SIG / ORACLE / RANDOM top-4 response metrics from
the stored arrays and requires `max|diff| ≤ 1e-9` against
`row_metrics.npz`.  Failure aborts the round.  No frozen model, checkpoint or
soup is loaded; no `objects_test` and no Locked split is ever touched.

---

## 2. Junction-level quantities

For a molecule G with `n_j` junctions:

* `d_j` = within-molecule average-rank percentile of `dev_j`,
* `q_j` = within-molecule average-rank percentile of `|ig_abs_j|`,
* percentile rule: `(rank − 1)/(n_j − 1)` with ties averaged; `n_j = 1 → 0.5`
  (identical to the previous round),
* residual: `r_j = q_j − d_j`,
* context: `κ_j = (center_degree, endpoint_ring_count)`, endpoint ring count ∈ {0,1,2}.

`q_j` is used **only** to fit the discovery tables and to score ranking quality;
it never selects validation junctions.

---

## 3. Arms (frozen)

* **Arm A (DEV-only):** `score_A(j) = d_j`.  A is the most important control.
* **Arm B (DEV + signature residual):** `score_B(j) = d_j + R'_τ(j)`.
* **Arm C (DEV + signature + fixed context):**
  `score_C(j) = d_j + R'_τ(j) + R'_(τ,κ)(j)`.

Arm A top-4 selection is identical to the historical DEV ranking (monotone
transform).  Ties break by ascending Junction index (stable argsort).  Unknown /
below-floor signatures contribute bonus 0.

### 3.1 Discovery fit (molecule-capped, seeds are repeats)

From the 256 Phase-0 diagnostic train molecules × 2 frozen R3 seeds:

* per `(seed, molecule, key)`: mean residual of that key's Junctions;
* per molecule: mean over the seeds containing the key;
* `R_τ` = mean over molecules (each molecule one unit; repeated Junctions cannot
  dominate);
* `n_τ` = number of distinct discovery molecules containing τ;
* **shrinkage:** `R'_τ = n_τ/(n_τ+20) × R_τ` for `n_τ ≥ 20`, else `R'_τ = 0`
  (below-floor signatures excluded from Arm B).

Context units are defined for **every** signature present in the discovery set
(including below-floor signatures; their `R'_τ = 0`):

* `r'_j = r_j − R'_τ(j)`;
* same molecule-capped aggregation for `R_(τ,κ)`; `n_(τ,κ)` = distinct
  discovery molecules containing the cell;
* **shrinkage:** `R'_(τ,κ) = n/(n+20) × R_(τ,κ)` for `n ≥ 10`, else 0.

No class selection, feature search, weight tuning or threshold change is
allowed.  The Arm B/C tables are functions only of the discovery junction sets
the builder receives.

---

## 4. Conditional pseudo-control (pre-fixed)

For each replicate `p ∈ {0..15}`:

* within each `(seed, molecule)`, residuals are permuted among Junctions inside
  the same DEV percentile quartile bin `min(⌊4 d_j⌋, 3)`;
* every Junction keeps its own `d_j`, signature, degree and endpoint ring count;
* tables `R'^p_τ`, `R'^p_(τ,κ)` are estimated with the identical molecule-capped
  procedure and identical floors/shrinkage;
* RNG: `numpy default_rng(SeedSequence([810000000, p, seed, row]))`.

**Preserved:** the per-(seed, molecule, DEV-quartile) residual multiset (hence
the marginal residual distribution and its coarse DEV dependence), the molecule
and two-seed pairing, all signature/context labels.
**Destroyed:** the association between signature identity (and τ,κ cell) and the
residual magnitude inside a DEV stratum.
**Limitation:** this is not a full confound control; continuous DEV values,
higher-order class interactions and molecule-level structure are not preserved.

Scoring uses the same A/B/C formula on validation with the pseudo tables.

---

## 5. Discovery-internal cross-fitting (anti-memorisation, gate A7)

Diagnostic rows sorted ascending: even index → fold A, odd index → fold B (same
split as the previous round).  Each fold's tables are estimated on the **other**
fold only and used to rank the held-out fold; every molecule is evaluated
exactly once.  The |delta| response is the stored natural-replacement response.
This is a diagnostic, not a parameter-selection device.

---

## 6. Validation

* Data: the 1000 official-valid structures.
* Exposure: official valid is a **historically exposed development set**, not a
  fresh blind test; this is stated in every output.
* No prediction labels are read anywhere.  Official test is never loaded; the
  internal Locked split is never read.
* Primary metric: for each molecule-seed, rank Junctions by the arm score and
  take the mean `|delta_j|` of the top-4: `A@4`, `B@4`, `C@4`.
* Primary contrasts: **B − A** (core conditional-value test), **C − A**,
  **C − B** (context increment).
* Secondary: Spearman of the score vs `|IG|` (per molecule, mean over
  molecules), top-4 overlap with the oracle ranking, k ∈ {1,2,4,8} sensitivity,
  historical SIG / ORACLE / RANDOM references, subsets, sizes, scaffolds.

### 6.1 Statistical unit

The independent unit is the **molecule**, never the Junction and never
molecule×seed.

* per seed: report the contrast separately (means);
* merged inference: per molecule average the two seeds first; bootstrap
  (5000 resamples, seed 0) over molecule identities; exact one-sided sign test;
* report mean difference, 95% CI, fraction of positive molecules;
* no Junction-level significance claims.

### 6.2 Subsets

* `scaffold_unseen_discovery`: Murcko scaffold absent from the 256 discovery
  molecules (pre-fixed direction check for A4);
* `scaffold_not_in_train10k`: scaffold absent from the supervised official-train
  10k (reported separately from the previous item, never conflated);
* `no_exact_dup_train`, molecule-size terciles.

---

## 7. Decision gates (exact; frozen before any validation contrast)

**Result A conditions (`SIGNATURE_HAS_CONDITIONAL_VALUE`)**

| gate | condition |
| --- | --- |
| A1 | mean(B − A) > 0 |
| A2 | molecule-bootstrap CI95 lower of (B − A) > 0 |
| A3 | per-seed mean(B − A) ≥ 0 for both seeds |
| A4 | scaffold_unseen_discovery subset mean(B − A) > 0 |
| A5 | mean gain excluding the top driver class > 0 AND top-1 driver positive-gain share < 0.5 |
| A6 | mean(B − meanₚ pseudoBₚ) > 0 AND CI95 lower > 0 |
| A7 | discovery cross-fit mean(B − A) > 0 |
| A8 | (B − A) / (ORACLE − A) ≥ 0.05 (practical recovery of the DEV→oracle gap) |

**Result B conditions (`CONTEXT_ADDS_VALUE_BUT_SIGNATURE_WEAK`)**

| gate | condition |
| --- | --- |
| B1 | mean(C − B) > 0 |
| B2 | molecule-bootstrap CI95 lower of (C − B) > 0 |
| B3 | per-seed mean(C − B) ≥ 0 for both seeds |
| B4 | mean((C − B) − (pseudoC − pseudoB)) > 0 AND CI95 lower > 0 |
| B5 | (C − B) / (ORACLE − B) ≥ 0.05 |

**Label logic (frozen):**

1. A1..A8 → `SIGNATURE_HAS_CONDITIONAL_VALUE`.
2. else B1..B5 and not (A1∧A2∧A3) → `CONTEXT_ADDS_VALUE_BUT_SIGNATURE_WEAK`.
3. else a positive point estimate whose CI crosses 0, or a statistically
   reliable B−A that passes A4/A5/A6/A7 but fails only the practical A8 →
   `INCONCLUSIVE`.
4. else → `NO_CONFIRMED_CONDITIONAL_VALUE` (includes: non-positive estimates, or
   a statistically reliable B−A that fails a robustness gate A4/A5/A6/A7, i.e.
   the gain cannot be attributed to stable shared type information).

No post-hoc threshold, weight, shrinkage or context change is permitted after
any validation contrast is seen.  A weak statistical positive without a
practically meaningful ranking gain (A8/B5) is reported as such and is never
upgraded to Result A.

---

## 8. Required outputs

* discovery residual tables with support counts and hashes (`residual_tables.json`),
* frozen protocol/code hashes (`protocol_freeze.json`) and final artifact hashes (`finalize.json`),
* paired A/B/C results per seed and molecule-clustered uncertainty (`metrics.json`, `validation_analysis.json`),
* conditional pseudo-control tables and contrasts,
* context support / fallback / unknown-type coverage,
* a proof that historical artifacts were not modified (`historical_integrity.json`, prior artifact hashes),
* fast unit tests for the conditioning math, pseudo-control, statistics unit and decision rule.

---

## 9. Discipline

* Model completely frozen; no training, no inference, no GPU.
* No supervised property labels, no official test, no Locked split.
* Discovery tables use only the 256 diagnostic train molecules; validation data
  never enters any fit, floor, shrinkage or threshold.
* Historical reports, checkpoints and results are read-only.
* The decision vocabulary and gates above are frozen before the first
  validation contrast; any implementation-stage amendment must be recorded here
  **before** validation statistics are computed.

## 10. Amendments

*None before the validation comparison.  Any amendment added after this line
must state the time, the changed rule, and why no validation statistic had been
computed yet.*
