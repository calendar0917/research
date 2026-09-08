# ZINC Long-Cycle / Target-Decomposition Audit (compact-v2 information-gap follow-up)

**Status:** FINAL — 2026-09-08
**Questions answered:** Q1–Q12 · Tables A–F · Figures 1–6
**Decision:** **GO — GLOBAL TOPOLOGY** (with benchmark-artifact caveat)
**Hypothesis verdict:** **CONFIRMED** — the extreme negative ZINC-subset targets are driven by the
**long-cycle penalty term in the benchmark target definition** (a graph-global topology term),
NOT by a generic heavy-tail regression failure of compact-v2.
**Primary recommendation:** one experiment — compact permutation-invariant global cycle-scale
channel with a monotone-capable head (ceiling ≈ +0.0117 MAE on valid, i.e. 0.184 → 0.172).

---

## 0. Executive summary

| Fact | Value | Status |
|---|---|---|
| Target formula source | GVAE (Kusner et al. 2017) `generate_latent_features_and_targets.py`: `y = z(logP) + z(SA) + z(cycle)`, `cycle = -max(0, max(nx.cycle_basis(...)) - 6)` | **VERIFIED** (script + constants + replication) |
| Per-molecule label reproduction | 11,981 / 12,000 (99.84%) exact | VERIFIED |
| Deep tail (y < −10) caused by cycle term | 12/12 molecules (100%) have a negative label cycle value | VERIFIED |
| y < −6 caused by cycle term | 110/132 (83.3%) | VERIFIED |
| y < −4 caused by cycle term | 222/461 (48.2%) — the rest are genuinely heavy-tailed logP/SA molecules | VERIFIED |
| Residual ↔ missing cycle term | valid Pearson **0.755** (residual ≈ (1−λ)·cycle_component) | VERIFIED |
| Model learns the penalty partially | λ(severity): 0.99 → 0.89 (excess 1) → 0.46 (excess 2) → 0.001 (excess ≥4) | VERIFIED |
| The term's exact value is ORDER-DEPENDENT | 34/655 molecules basis-unstable under random orders (8.8% of extremes); 19/12,000 label vs GVAE-order mismatches | VERIFIED (artifact) |
| Best model-reachable correction | isotonic on invariant mean-max-basis: **+0.0117** (strong) | VERIFIED (oracle) |
| Best target-knowing correction | isotonic on exact (order-dependent) cycle component: **+0.0247** (very strong) | VERIFIED (oracle) |
| Artifact share of the recoverable signal | 1 − 0.0117/0.0247 = **53%** (not reachable by any equivariant model) | VERIFIED |

---

## 1. Scope, files, protocol

* Audit code: `tracks/ksvd/experiments/luyin16/zinc_long_cycle_audit.py` (stages
  `provenance / match / verify / refine / audit / perm / oracle / decision`) and
  `zinc_long_cycle_summary.py` (Tables A–F).
* Artifacts: `tracks/ksvd/results/zinc_long_cycle_audit/` (per-split audit CSVs,
  `label_effective_cycle.csv`, `exact_longest_all.csv`, `randomized_maxbasis.csv`,
  `oracle_results.csv`, `decision_record.json`, `figures/`).
* Protocol `zinc-context-gap`, seed 0. **No model, data, loss, or optimizer was modified.**
* Test split: descriptive only (composition/prevalence); no fits, no selection.
* The 13 residual probes of the Information Gap Audit were **not** re-run (out of scope here).

### Key definitions (exact wording)
* `label_effective_cycle_snapped`: per-molecule cycle value *actually used by the label
  generator* — reverse-engineered from `y − (z(logP) + z(SA))` and snapped to the integer lattice
  (p95 snap error 0.0016).
* `candidate_normalized_cycle_component`: the GVAE-formula component replicated with community
  constants; used as **candidate** until the label-implied value was derived.
* `max_basis_cycle_length` (stored order): `nx.cycle_basis` on the graph as stored in the PyG
  pickle. **NOT** the longest simple cycle (see §11).

---

## 2. Target provenance (Q2)

* **VERIFIED.** Data = PyG `ZINC(subset=True)`; `y = mol['logP_SA_cycle_normalized']` comes from a
  pre-computed pickle (`molecules.zip` feo9qle74kg48gy → benchmarking-gnns split indexes).
* Upstream formula **located and read** in GVAE (Kusner et al. 2017):
  `generate_latent_features_and_targets.py` — `logP = Descriptors.MolLogP`,
  `SA = −sascorer.calculateScore`, `cycle = −max(0, max(nx.cycle_basis(M))−6)`,
  normalized by **full-dataset** mean/std, then summed. (JTNN `optimize.py` checked: MolLogP −
  sascorer only, no cycle term. GVAE atom-type dictionary checked: unused for target.)
* Replication: 12,000/12,000 subset molecules matched to SMILES (pynauty isomorphism) and the
  GVAE pipeline re-run; the cycle lattice is exactly {0, −1, −2, −4, −5, −6, −12} in the 12k
  (full set includes −3, −7…−18). Constants check: logP mean rel. diff 1.1e-05, SA std 0.4%,
  cycle mean 1.1%, cycle std 1.5% (fitted σ_cycle = 0.2883 vs community 0.2860).
* The community constants are therefore confirmed as the *normalization* of the GVAE generation
  pool (~249,456 molecules); the label itself is fixed per molecule.

---

## 3. Tables

### Table A — Extreme target decomposition (label-implied cycle values)
`results/zinc_long_cycle_audit/table_a_extreme_decomposition.csv` (top-50 by target + all y<−4).

| molecule_id | split | y | label cycle | cycle comp. | y without cycle | stored basis | exact longest simple | RDKit max ring |
|---|---|---|---|---|---|---|---|---|
| train:2210 | train | **−42.04** | −12 | −41.62 | **−0.42** | 18 | **26** | 18 |
| train:1760 | train | −21.91 | −5 | −17.34 | −4.57 | 13 | 17 | 6 |
| train:1424 | train | −20.78 | −6 | −20.81 | +0.03 | 12 | 12 | 6 |
| valid:0172 | valid | **−20.34** | −6 | −20.81 | **+0.47** | 12 | 12 | 6 |
| train:3776 | train | −20.26 | **−6** | −20.81 | +0.55 | **6** | 12 | 6 |
| train:2347 | train | −11.57 | **−4** | −13.87 | +2.31 | **6** | 10 | 6 |
| valid:0214 | valid | −10.31 | −2 | −6.94 | −3.37 | 8 | 12 | 6 |
| test:0670 | test | −9.68 | −2 | −6.94 | −2.74 | 8 | 8 | 6 |

**Reading:** every molecule with y < −10 has a normal-range "chemical" remainder once the label
cycle term is subtracted (train:2210's −42.04 is a −0.42 molecule + a −41.62 cycle term).
Two molecules (**train:3776, train:2347**) have stored-order basis max = 6 → **no penalty in the
stored order**, yet the label penalizes them (−6, −4): the value used by the label is a function
of the *generator's* node order, not of the graph alone.

### Table B — Validation error by label-implied cycle severity
`table_b_valid_error_by_label_excess.csv` (n=1000; label-based excess; pred/resid from the
selection model, cross-checked against the Information Gap Audit table, max diff 1e-9).

| group | n | target mean | pred mean | resid mean | MAE | MAE mass share | λ (learned) |
|---|---|---|---|---|---|---|---|
| excess = 0 | 965 | 0.194 | 0.213 | −0.018 | 0.1391 | 0.729 | (0.99) |
| excess = 1 | 30 | −3.26 | −2.88 | −0.376 | 0.4662 | 0.076 | 0.892 |
| excess = 2 | 4 | −9.48 | −5.70 | −3.780 | 3.7796 | 0.082 | 0.455 |
| excess ≥ 4 | 1 | −20.34 | +0.45 | −20.79 | 20.7936 | 0.113 | 0.001 |

**35 molecules (3.5%) carry 27.1% of the validation MAE mass**, and a single molecule
(valid:0172, excess 6) carries 11.3%. Cycle-free validation MAE = **0.1391**.

### Table C — Permutation (order-)sensitivity of the basis statistic
`table_c_basis_instability.csv`: 655 molecules (387 extreme-tail + 268 uniform sample) × 100
random node-insertion orders.

| statistic | tested | unstable | stable fraction |
|---|---|---|---|
| `nx.cycle_basis` max (stored order replicated) | 655 | 34 (all y<−4; 8.8% of extremes) | 0.948 |
| `nx.minimum_cycle_basis` max | 20 × 20 perms | 0 | 1.0 |
| exact longest simple cycle | 5 × 10 perms | 0 | 1.0 |

The 34 unstable molecules are **all** in y < −4 (0 unstable among non-extremes): ordering affects
exactly the molecules where the penalty matters.

### Table D — Cycle-statistic comparison (correlations, 12k or valid)
`table_d_cycle_statistic_comparison.csv`.

| statistic (class) | corr(y) | corr(|resid|, valid) | permutation-invariant |
|---|---|---|---|
| max basis (stored order) [candidate] | −0.114 | +0.454 | NO (proven) |
| GVAE-replicated cycle score [candidate] | +0.517 | −0.763 | NO (order-dependent) |
| label-implied cycle value [decomposition] | +0.515 | −0.763 | n/a (the label) |
| max min-cycle-basis length | −0.035 | +0.078 | YES |
| total min-cycle-basis length | +0.126 | +0.059 | YES |
| exact longest simple cycle (all 12k) | **+0.001** | +0.139 | YES |
| RDKit max ring size | −0.035 | +0.078 | YES |
| mean max-basis over 10 random orders | −0.107 | +0.378 | YES |
| frac random orders with max basis ≥ 7 | **−0.432** | +0.271 | YES |

Key: the *label's* cycle term correlates with the residual at −0.763 (the residual essentially
*is* the missing term), while **no permutation-invariant statistic exceeds 0.38**; the exact
longest simple cycle is essentially uncorrelated with y over the full 12k (because 1,624 fused
ring systems have longest ≥ 10 without any penalty — naphthalene-like).

### Table E — Oracle results (train-OOF residuals only; valid evaluate-only)
`table_e_oracle.csv` (full set also in `oracle_results.csv`).

| method | corrected valid MAE | ΔMAE | verdict |
|---|---|---|---|
| baseline | 0.1842 | — | — |
| linear probe: cycle component | 0.2017 | −0.0175 | fails |
| linear probe: label-implied cycle | 0.2029 | −0.0188 | fails |
| linear probe: excess basis length | 0.2014 | −0.0173 | fails |
| binned-median: excess buckets | 0.1825 | +0.0016 | weak |
| **isotonic: cycle component (exact label term)** | **0.1595** | **+0.0247** | **very strong** |
| isotonic: longest simple cycle (invariant) | 0.1929 | −0.0088 | fails |
| isotonic: max min-basis (invariant) | 0.1825 | +0.0017 | weak |
| isotonic: RDKit max ring size (invariant) | 0.1825 | +0.0017 | weak |
| **isotonic: mean max-basis over random orders (invariant)** | **0.1725** | **+0.0117** | **strong** |
| isotonic: frac random orders ≥ 7 (invariant) | 0.1900 | −0.0058 | fails |
| poly-2: 3 invariant stats | 0.2000 | −0.0158 | fails |
| GBR: 3 invariant stats | 0.1826 | +0.0016 | weak |
| GBR: 5 randomized-basis invariant stats | 0.1846 | −0.0005 | fails |
| TARGET-DEFINITION subtraction (full cycle term) | 0.3155 | −0.1313 | not a model (see §8) |

**Reading:** (i) the correction must be *monotone-nonlinear*, not linear (every linear probe
hurts; the error is a saturation step, not a linear trend); (ii) the best *model-reachable*
(invariant) oracle is **+0.0117**, about half of the +0.0247 that exact target knowledge gives —
the other half is order-dependence (artifact) that no equivariant model can express.

### Table F — Distribution with/without the label cycle term
`table_f_distribution_stats.csv`.

| split | original min | removed min | kurtosis orig → removed | skew orig → removed |
|---|---|---|---|---|
| train | −42.04 | −7.74 | 24.39 → 0.53 | −2.32 → −0.84 |
| valid | −20.34 | −6.29 | 12.62 → **0.24** | −2.07 → −0.72 |
| test | −9.68 | −8.32 | 1.71 → 0.72 | −1.13 → −0.84 |

Removing the label cycle term **removes the heavy tail**: kurtosis drops by ≥1 order of
magnitude; valid y < −10 count 2 → 0; y < −6 count 11 → 2.

---

## 4. Figures

1. `fig1_target_vs_basis_and_longest.png` — y vs stored-order basis max (a) and vs exact
   longest simple cycle (b), all splits; shows the invariant panel's fused-ring confusion.
2. `fig2_target_vs_cycle_component_label.png` — y vs label-implied cycle component (3 splits).
3. `fig3_valid_residual_vs_cycle_component_label.png` — valid residual vs cycle component; the
   y = x line shows residual ≈ missing cycle term.
4. `fig4_valid_mae_by_excess_label.png` — valid MAE by label excess group (log-ish scale masses).
5. `fig5_target_distribution_removed_label.png` — original vs cycle-removed distributions.
6. `fig6_basis_instability.png` — basis max vs permutation (100 orders) on the 655 tested
   molecules; unstable ones flagged; inset min-basis/longest (stable).

---

## 5. Q1–Q12

**Q1. Where do the extreme negative targets come from?**
From the benchmark target's long-cycle term. y < −10: 12/12 (100%) have a negative label cycle
value (11/12 ≤ −2); y < −6: 83.3%; y < −4: 48.2% (the remainder are genuinely low logP/SA
molecules — a separate, milder heavy tail; test:0449 y = −8.32, cycle 0). Deepest examples:
train:2210 = −0.42 chemistry + −41.62 cycle; valid:0172 = +0.47 chemistry + −20.81 cycle.

**Q2. What is the provenance of the target formula?**
VERIFIED as GVAE/Kusner et al. `generate_latent_features_and_targets.py` (z-scored logP + SA +
cycle, full-pool constants). Community constants reproduce the data to 1e-5–1.5%. The formula is
NOT a local "ring" property: it is a **graph-global** `nx.cycle_basis` max.

**Q3. What cycle value does each molecule carry?**
Label-implied lattice {…, −12, −6, −5, −4, −2, −1, 0}; 459/12,000 (3.8%) molecules penalized;
11,981/12,000 exact reproduction (19 order-sensitive exceptions, incl. train:3776
stored 6/label −6 and train:2347 stored 6/label −4).

**Q4. Is the "basis-cycle" statistic the longest cycle?**
No. Exact longest simple cycles on the tail: train:2210 → 26 (label penalized basis-18),
train:1760 → 17 (basis 13), valid:0229 → 14 (label excess 1!), valid:0403 → 15 (excess 1).
The label's *value* is a particular basis's max — between the min-basis max and the true
longest simple cycle, and it varies with node ordering.

**Q5. How much of the validation error comes from this term?**
The residual ≈ (1−λ)·cycle_component (Pearson 0.755; corr(|resid|, comp) = −0.763). Valid
cycle molecules (3.5%) → 27.1% of MAE mass; excess ≥ 2 (5 molecules) → 19.5%; one molecule
(valid:0172) → 11.3%. Cycle-free valid MAE = 0.1391.

**Q6. Has the model learned the penalty at all?**
Partially — at small severity only: λ = 0.89 at excess 1 (predictions track −3.26 targets to
−2.88), λ = 0.46 at excess 2, λ ≈ 0.00 at excess ≥ 4 (predicts +0.45 for the −20.34 target).
The effective prediction floor is ≈ −9.3; severity beyond what the local receptive field can
see is invisible to it.

**Q7. Is the global shrinkage driven by long-cycle samples?**
Yes. Shrinkage slopes: no-cycle 0.989 (essentially no shrinkage), cycle 0.211, all 0.838. The
16%-global compression is entirely attributable to the 3.5% long-cycle molecules.

**Q8. Is the basis statistic permutation-stable?**
No for the basis max (34/655 tested unstable; 8.8% of 387 extremes; 0/268 non-extremes), yes for
`minimum_cycle_basis` max (20×20) and exact longest simple cycle (5×10).

**Q9. Does the ordering ambiguity affect the label itself?**
Yes, and materially: 19/12,000 molecules' label value differs from our GVAE-order replication
(order-sensitive), and two of the top-15 deepest molecules (train:3776, train:2347) have
stored-order basis 6 → no penalty under that ordering yet label −6/−4. The penalty is a property
of (graph, generator order), not of the graph.

**Q10. Do invariant cycle statistics explain the tail?**
Only weakly. Full-data correlations: exact longest simple cycle vs y = 0.001 (1,624 fused
no-penalty confounders), min-basis max = −0.035, mean random-basis max = −0.107, frac(≥7) =
−0.432. Residual correlation: best invariant = 0.378 (mean random-basis) vs 0.763 for the label
term. The *global ring scale* is real (longest cycles 8–26 on the tail) — the *exact value* is
not an invariant function of the graph.

**Q11. What does the simple global-cycle oracle buy?**
Best invariant oracle: isotonic on mean max-basis over random orders — **ΔMAE +0.0117**
(0.1842 → 0.1725, 6.4% relative; "strong" by the pre-registered threshold ≥0.008). The
target-exact (order-dependent) oracle: **+0.0247** (very strong, ≥0.015). The gap (53% of the
recoverable signal) is unreachable by any permutation-equivariant model. **Any linear head on a
cycle feature hurts (−0.0175)**; the correction shape is a monotone saturation step.

**Q12. Valid/test differences?**
Composition: excess ≥ 1 present in 3.72% (train) / 3.5% (valid) / 5.2% (test); but severity is
what differs — max excess 12/6/**2**; test has **no** molecule with excess ≥ 3 (its min −9.68 is
an excess-2 molecule). The valid MAE 0.184 vs test 0.135 gap is dominated by the presence of a
single excess-6 sample in valid (11% of valid MAE mass) and its absence in test; the cycle-free
distribution is otherwise similar (test kurtosis 1.7 already). Test was descriptive-only.

---

## 6. Decision (GO/NO-GO) — the single decision

**GO — GLOBAL TOPOLOGY** (per the pre-registered §32 rule set: mechanism verified at y<−6
(0.833 ≥ 0.8) with the residual ratio (0.139×3 < 20.79); best *model-reachable* oracle ΔMAE
+0.0117 ≥ 0.008; pre-registered artifact-rule trigger 8.8% < 0.3).

**The original hypothesis is CONFIRMED with one refinement:** the extreme tail is a target-
definition long-cycle mechanism, but the exact term is order-dependent (artifact). For the
track's purposes the actionable consequence is two-sided:

1. **Primary recommendation (the one experiment):** add a compact **permutation-invariant
   global cycle-scale channel** (e.g. exact longest-simple-cycle summary + ring-size spectrum,
   or the randomized-basis-mean estimator) into the compact-v2 graph head via a
   **monotone-capable** readout (isotonic-style/step or log-scaled units); train-only, valid
   evaluation, protocol `zinc-context-gap` seed 0; expected ceiling **+0.0117** (0.1842 →
   0.1725, "strong"). Expect the gains to concentrate on long-cycle molecules (the 3.5%).
   **Do not** expose a linear cycle-basis feature (linear oracle −0.0175; binned +0.0016).
2. **Artifact caveat (not the primary recommendation):** 53% of the recoverable error is tied
   to the order-dependence of the label value; if the topology experiment lands as NO-GO, the
   fallback is a **benchmark target redefinition** (invariant cycle penalty, e.g. canonical-order
   basis or randomized-mean basis; re-run of the terminal evaluations), which unlocks the
   remaining +0.013 — and makes predictions reproducible across dataset reloads.

**Explicitly NOT recommended:** generic heavy-tail regression fixes (weighted L1, quantile
regression, tail reweighting — the tail is not a sampling imbalance, it is a defined term),
CIN/cell message passing, and new patch/ring embeddings (v3 NO-GO stands; local ring context is
not this statistic).

---

## 7. Corrections to earlier notes (addenda)

See the appended addendum sections in:
* `tracks/ksvd/notes/compact_v2_information_gap_audit.md` (addendum 2026-09-08)
* `tracks/ksvd/notes/compact_hybrid_v2_budget_reallocation.md` (addendum 2026-09-08)
* `tracks/ksvd/notes/compact_v3_context_conditioned_patch_representation.md` (addendum
  2026-09-08)

Summary of corrections:
1. "Heavy-tail regression collapse" → **"target-definition long-cycle mechanism with a
   learning-saturation component"**: the 16% global shrinkage (ŷ ≈ 0.07+0.838y) is caused by
   3.5% long-cycle samples with λ-collapse, not by an L1 mean-shift on a genuinely heavy-tailed
   regression problem. The tail is *defined into* the targets.
2. L1 is **conditional-median** regression (not "mean regression"); with a saturated head and
   rare extreme targets, MAE-optimal predictions understate severity — consistent with the
   observed −9.3 floor.
3. The old residual-cycle probe (Spearman ≈ 0.15; 16-D cycle-stats probe R² = +0.0002, NO-GO)
   remains NO-GO **for linear probes**; it is superseded for the *nonlinear monotone* case by
   the isotonic evidence (+0.0117 invariant / +0.0247 exact). The 16-D probe did not test the
   monotone shape and used local statistics, not the global cycle scale.
4. v3 NO-GO is not overturned; it is reinterpreted: local ring-context conditioning (coarse/
   typed ring descriptors) ≠ global long-cycle statistic. The v3 result is consistent with the
   audit — those are different quantities; only the latter carries the missing term (see Q10).
