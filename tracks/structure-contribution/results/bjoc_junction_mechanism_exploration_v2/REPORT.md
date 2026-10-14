# cscl-bjoc-junction-mechanism-exploration-v2 — REPORT

Track: `tracks/structure-contribution` · Protocol: `cscl-bjoc-junction-mechanism-exploration-v2`
Frozen commit: `f25de479ac94` · Split: D (Fit 8000 / Monitor 1000 / Locked 1000)

This round had two parts, both pre-registered in
`configs/bjoc_junction_mechanism_exploration_v2.yaml` before any new score was produced:

* **Part I (CPU, frozen predictions):** error-complementarity analysis of the already frozen
  split-D Monitor predictions `A0/A1/A2` seed 0 from `bjoc_junction_sufficiency_audit_v1`.
  No retraining of A0/A1/A2 seed 0.
* **Part II (GPU, 2.0 GPU-h cap):** three pre-declared new arms on the frozen BJOC backbone —
  `C = J_COUNT_ONLY`, `D = J_DIRECT_READOUT`, `E = J_TYPE_CONDITIONED_POOL` — plus A1 seed 1 as
  the same-seed control of D. No GNN / Transformer / attention / iterative message passing; same
  encoders, SSL, recipe, and per-seed batch order as P3.

## 1. Data, reuse and label discipline

Every reused artifact was hash-verified before training and again before scoring
(`input_audit.json`, `protocol_freeze.json`):

| artifact | sha256 (12) | status |
|---|---|---|
| split-D Fit rows | `9468656affcc` | match |
| split-D Monitor rows | `d196bc75ce89` | match |
| split-D Locked rows | `f3525ae88211` | match (never read) |
| SSL `pretrain_D_s0` / `s1` | `8da281fac33b` / `2e06d2f5067b` | match |
| P3 soups seed 0 / 1 | `13a78a632fb9` / `13c5e8124007` | match |
| A0 / A1 soups seed 0 | `849244a07e28` / `ee6c9729a167` | match |
| frozen A0/A1/A2 Monitor preds | `4ef6777aa640` / `5964b5c925ea` / `152791eea263` | match |
| Monitor `y` (junction-audit lock) | `bbcf6faf4ab1` | match |

Discipline actually exercised:

* **Monitor labels were read exactly once**, after all new predictions were frozen and
  hash-manifested; `monitor_read.lock` records the read and the manifest sha.
* **Locked labels were never read; official valid/test were never loaded** (asserted in
  `monitor_scores.json` and `scientific_decision.json`).
* Split D and its Monitor are **historically exposed exploratory development data**; all
  statements are within-split paired arm contrasts, not blind validation.
* Training reused the per-seed P3 scratch init, batch order, and Fit-only masked-SSL encoder;
  each new `train_*.json` records `p3_matched_checks == all True` (init full/group/encoder hash,
  encoder source, parameter audit, batch order, recipe, shared-parameter hash).
* Deviations recorded: (i) the config's gradient gate was implemented as a two-phase witness
  (at-init + after one optimizer step) because the D zero-init readout structurally blocks the
  first Junction gradient; (ii) remote `freeze` verified only training-critical sources, while
  the full local audit (including analysis-only files) is green; (iii) seed-1 binaries missing
  from the local checkout were restored from the prior `rr pull` bundle
  `tat-score-monitor-20261014-022445-aef26be5` and hash-verified; (iv) `A0:0` fit/wall metadata
  in the score table comes from the historical junction-audit train record.

## 2. CPU correctness gates (19/19 green, `cpu_correctness_gates.json`)

All new formulas were validated on synthetic objects and on a real Fit batch before training:

* A0/A1/A2/C/E = 413,505 params; **D = 462,657** confirmed via `named_parameters()` (reader
  first layer 256×513; +192 columns).
* E at init is tensor-identical to A2; E moment identities
  `Σ_e W_e = 2 Σ_j w_j`, `Σ_e V_e = 2 Σ_j w_j²` hold to ≤ 2.6e-7 relative error; the vectorised
  bucketing equals an independent Python loop to ≤ 5.7e-6.
* C: `W = V = 0`, `q = log1p(n_e)` exactly; the Junction encoder is structurally disconnected
  from the supervised loss (disclosed, effective active capacity < A2).
* D: new reader columns exactly zero, old columns tensor-identical to A0, init forward equals
  A0 to 6e-8 (float32 reduction order; exact in exact arithmetic); after one Adam step the
  Junction encoder gradient is non-zero (0.092) and the new columns receive gradient.
* Label-blind type-conditioned witness: two object-level structures with identical A2
  moments/n_e/bond-type histogram but different type co-occurrence — A2 blind (ΔW = ΔV = 0),
  E separates them (ΔW = 2.0). Type relabel consistency, junction-row permutation, relabel/swap/
  batch invariances, save/load replay, empty edge cases, no forbidden modules, no-y forward.
* `pytest tests/test_bjoc_junction_mechanism_exploration_v2.py` — **20/20 passed**.

## 3. Part II — training and results

Seven new trajectories, one per pre-declared `(arm, seed)`; seed 1 was bought for completeness
only, never selected on seed-0 outcomes (A0 seed 1 does not exist and is reported as NOT_RUN).

| arm:seed | MAE | RMSE | ordinary (y>−3, n=929) | tail (y≤−3, n=71) | fit MAE | params |
|---|---|---|---|---|---|---|
| A0:0 | 0.121514 | 0.326862 | 0.102599 | 0.369005 | 0.002741 | 413505 |
| A1:0 | 0.106239 | 0.517173 | 0.082460 | 0.417385 | 0.001434 | 413505 |
| A2:0 | 0.103960 | 0.493170 | 0.081137 | 0.402585 | 0.001082 | 413505 |
| C:0 | 0.125444 | 0.404440 | 0.101577 | 0.437721 | 0.002814 | 413505 |
| D:0 | **0.099480** | 0.418778 | 0.080976 | **0.341594** | 0.001860 | 462657 |
| E:0 | 0.104061 | 0.500447 | **0.079113** | 0.430501 | 0.001014 | 413505 |
| A1:1 | 0.099541 | 0.438709 | 0.076525 | 0.400685 | 0.001448 | 413505 |
| A2:1 | 0.104642 | 0.512340 | 0.079695 | 0.431061 | 0.001077 | 413505 |
| C:1 | 0.129983 | 0.513103 | 0.102677 | 0.487266 | 0.002395 | 413505 |
| D:1 | 0.100665 | 0.489263 | 0.077560 | 0.402991 | 0.001592 | 462657 |
| E:1 | 0.101182 | 0.497865 | 0.078020 | 0.404246 | 0.001086 | 413505 |

Pre-declared contrasts (paired molecule bootstrap, 5000 resamples, seed-averaged where both
seeds exist):

| contrast | Δ MAE | CI95 | ordinary Δ | tail Δ | token |
|---|---|---|---|---|---|
| `C−A0` s0 | +0.003929 | [−0.002801, +0.012035] | −0.001022 | +0.068715 | |
| `C−A2` s01 | **+0.023412** | **[+0.014641, +0.032549]** | +0.021711 | +0.045670 | `COUNT_ONLY_INSUFFICIENT` |
| `D−A1` s01 | −0.002818 | [−0.006791, +0.001298] | −0.000225 | −0.036742 | `J_DIRECT_READOUT_NO_GAIN` |
| `D−A2` s01 | −0.004229 | [−0.009318, +0.000431] | −0.001148 | −0.044530 | |
| `E−A2` s01 | −0.001679 | [−0.004045, +0.000592] | −0.001850 | +0.000551 | `J_TYPE_POOL_NO_GAIN` |
| `E−A1` s01 | −0.000268 | [−0.003357, +0.002866] | −0.000926 | +0.008339 | |
| `D−A0` s0 | −0.022035 | [−0.031667, −0.010226] | −0.021624 | −0.027411 | |
| `E−A0` s0 | −0.017453 | [−0.030450, −0.000028] | −0.023487 | +0.061496 | |

Per-seed deltas (the seed spread is the story):

* `D−A1`: seed 0 **−0.006760**, seed 1 **+0.001125**.
* `E−A2`: seed 0 **+0.000102**, seed 1 **−0.003460** (seed-1 CI upper −0.000279).
* `C−A2`: seed 0 +0.021484, seed 1 +0.025341 (consistent).
* A1 itself moves 0.106239 → 0.099541 between seeds; A2 moves 0.103960 → 0.104642.

**Decision tokens (frozen rules, no post-hoc tuning):**

* `C = COUNT_ONLY_INSUFFICIENT` — the deterministic per-bond count is clearly not enough:
  +0.0234 MAE vs A2, CI excluding 0 (+0.0146, +0.0325), consistent across both seeds. C stays
  near A0 on ordinary molecules (0.1016/0.1027 vs A0 0.1026) but is the worst arm on the tail
  (0.4377/0.4873). Junction *content* (W, V), not just the count, matters — mostly for extreme
  molecules.
* `D = J_DIRECT_READOUT_NO_GAIN` — D−A1 seed mean −0.0028 with CI95 crossing zero and opposite
  signs across seeds. Descriptively D is the best single model on seed 0 (0.099480) and its
  gain is tail-specific (tail −0.076 on seed 0, pooled −0.037), while `D−A2` is −0.0042 with CI
  [−0.0093, +0.0004]. With two seeds, a ~0.003 effect cannot be separated from seed noise.
* `E = J_TYPE_POOL_NO_GAIN` — E−A2 seed mean −0.0017 with CI crossing zero; seed 0 is +0.0001,
  seed 1 is −0.0035. E is the best-fitting arm on Fit (0.00101/0.00109, better than P3's
  0.00108) and improves ordinary molecules in both seeds (0.0791/0.0780 vs A2 0.0811/0.0797),
  but seed 0 pays for it on the tail (+0.0279, CI [+0.0101, +0.0466]) and is worse than A2 on
  tail in both seeds. The pre-registered `J_TYPE_POOL_FIT_ONLY` token was not triggered because
  its rule requires an `E−A2` Monitor delta ≥ 0; the observed pattern is "better Fit, one-sided
  ordinary improvement, tail trade-off".

Descriptive tail/ordinary subgroup contrasts are recorded separately in
`tail_ordinary_contrasts.json` (post-hoc, no decision rule attached).

## 4. Part I — error complementarity of the frozen A0/A1/A2 predictions

Full tables in `ERROR_COMPLEMENTARITY.md`; this section answers the three pre-declared questions.

**(1) Is A2's RMSE (square-error) disadvantage driven by a few extreme molecules? — Yes.**
A2 has lower MAE than A0 (0.103960 vs 0.121514) but higher RMSE (0.493170 vs 0.326862). The
single molecule `row 514` (y = −15.327; A2 pred −1.038 vs A0 pred −7.977) contributes **96.0 %**
of all positive SSE increments and **110.1 %** of the net SSE change of `A2−A0`. Remove it and
A2's RMSE drops to **0.1977 < 0.2299** (A0). Removing the worst 3/5/10/20 ΔSE molecules keeps
A2 strictly better (0.1955/0.1944/0.1862/0.1829 vs 0.2296/0.2298/0.2247/0.2247). The mirror
pair `A1−A0` shows the identical pattern (top-1 share 95.7 %, same row 514; RMSE without it
0.1999 < 0.2299). The `A2−A1` pair has no dominant molecule (top-1 positive share 8.3 %, net
share −2.1 %) and A2 is better after any top-k removal — A1 and A2 share their failure modes
(absolute-error Pearson 0.990), A0 is qualitatively different (0.897).

**(2) Are the molecules where A0 is clearly better structurally rarer? — Slightly, but not the
explanation.** Label-blind descriptors on Fit-only standardisation, Monitor novelty quartiles:
A0-clearly-better molecules (ΔAE > 0.05, n = 137) have median nearest-Fit distance 3.030 vs
2.636 for A2-clearly-better (n = 301), and 53/137 sit in the most novel quartile Q4. But A2
still has lower mean AE in **every** quartile (ΔAE: Q1 −0.0157, Q2 −0.0298, Q3 −0.0206,
Q4 −0.0041), and the A0-better fraction rises only mildly (0.364 → 0.440 → 0.416). Structural
rarity modifies the size of A2's advantage (smallest in Q4), it does not reverse it. Rarity and
label tail are coupled (40/71 tail molecules live in Q4; y vs novelty: Pearson −0.369), so
"novel" and "extreme-y" cannot be cleanly separated in a 1000-molecule Monitor; the structure
strata support exploratory readings only.

**(3) Do A0/A2 error patterns still differ within similar structural novelty? — Yes.** The
within-quartile tables show A0-better fractions of 0.320–0.440, i.e. substantial two-sided
disagreement everywhere, while A1/A2 track each other closely inside every stratum (e.g. Q2:
0.0620 vs 0.0585). Since A1 ≈ A2 (same per-bond broadcast with/without normalization) and both
differ from A0, the structural layer that distinguishes A0 is the composer topology itself, not
normalization and not novelty. This is the Part-I motivation for Part II: if the "where/how" of
the Junction path is to matter at all, it should show up in exactly these molecule groups.

## 5. Synthesis — what limits the current BJOC result

The round's four-way question was: (a) Junction content, (b) entry point, (c) organisation, or
(d) coverage of extreme molecules?

* **(a) Content is necessary.** Count-only C is the worst arm on the tail and significantly worse
  than A2 overall (+0.0234, CI [+0.0146, +0.0325]); A0 (no Junction W at all beyond the base
  composer) also has the best RMSE and the best tail after D. The raw count is not the signal;
  the junction moment content is.
* **(b) Entry point is plausible but unproven.** D's direct readout gives the best single-seed
  score and a tail-consistent direction (D−A2 negative in both seeds: −0.0045/−0.0040), but
  `D−A1` flips sign across seeds and its CI crosses zero. Two seeds cannot resolve ~0.003.
* **(c) Type-conditioned organisation is not the bottleneck as implemented.** E ≈ A2 overall,
  helps ordinary molecules, and trades that away on the tail in seed 0.
* **(d) Extreme molecules dominate the remaining error.** Tail MAE is 3.4–4.9× ordinary MAE for
  every arm; the RMSE reversal between A0 and A2 is one molecule plus the tail; C's failure is a
  tail failure; D's only consistent gain is tail. The most defensible next hypothesis is that
  the current limit is the model's behaviour on extreme targets, not the Junction readout path.
  (This is a recommendation, not a result of this round.)

## 6. Deliverables and integrity

Deliverables in `results/bjoc_junction_mechanism_exploration_v2/`:

* Reports: `REPORT.md`, `ERROR_COMPLEMENTARITY.md`, `ARCHITECTURE_COMPARISON.md`,
  `ADVISOR_BRIEF.md` (Chinese).
* Manifests / records: `prediction_manifest.json` (11 predictions), `checkpoint_manifest.json`
  (7 new soups + reused hashes), `parameter_gradient_audit.json` (init/trained, both seeds),
  `cpu_correctness_gates.json` (19/19), `gpu_budget_ledger.json`, `protocol_freeze.json`,
  `input_audit.json`, `monitor_scores.json`, `monitor_read.lock`, `monitor_y.npy`,
  `scientific_decision.json`, `tail_ordinary_contrasts.json`, `error_complementarity.json`,
  plus the 7 `train_*.json` and soup files.
* Code: `code/bjoc_junction_mechanism_models_v2.py`, `code/run_bjoc_junction_mechanism_exploration_v2.py`,
  `configs/bjoc_junction_mechanism_exploration_v2.yaml`,
  `tests/test_bjoc_junction_mechanism_exploration_v2.py`.

Budget: **1.2061 GPU-h** for 7 trajectories (limit 2.0) on 2× A100; no other GPU use.

Claims explicitly not made (`scientific_decision.json`): no causal attribution from forward
sensitivity; no claim that a bootstrap crossing zero proves no difference; the type witness is
an object-level counterexample, not a molecule or a real ZINC collision; no blind/independent
validation claim (split-D Monitor is exposed); no official valid/test or Locked read.
