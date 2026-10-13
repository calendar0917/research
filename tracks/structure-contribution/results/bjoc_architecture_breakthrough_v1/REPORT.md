# BJOC architecture breakthrough v1 — final report

* Protocol: `configs/bjoc_architecture_breakthrough_v1.yaml` (`cscl-bjoc-architecture-breakthrough-v1`), frozen 2026-10-13 (commit `96ab00d`, remote prep on `res`).
* Track: `tracks/structure-contribution`.
* Status: **complete**. Frozen decisions: **`GLRC_NO_GAIN`**, **`DLP_NO_GAIN`**. The round stops here: no third architecture, no rank/bucket/width search, no replication purchase, no full-10k refit.
* GPU spend: **1.777 of 6.0 GPU-hours** (`gpu_budget_ledger.json`).

## 1. Data access and provenance

* The official test split was never loaded, referenced, or scored by any code path in this round (`official_test_loaded: false` in every artifact).
* Official validation is an exposed development split; it was not loaded in this round at all (the conditional full-10k refit was not purchased).
* Internal splits C (primary) and D (replication) are group-aware re-partitions of the official train 10 000. Per-split overlap with the historical A/B splits is computed and stored in `split_C.json`; these molecules have appeared in historical experiments, so the splits are **not independent external tests** and only within-split matched contrasts are interpreted.
* The previous round's preprocessing-time Monitor label read is disclosed in `results/bjoc_performance_mechanism_long_v1/ERRATUM_label_access.md`. This round fixes the defect structurally: `ab.LabelGuard` makes Monitor/Locked `.y` unreachable during preprocessing, SSL, training and prediction (positive-control test included); the object-build alignment check uses Fit rows only (`alignment_rows_scope: "Fit rows only"`, `alignment_max_abs_diff: 0.0`).
* Label reads: exactly one declared Monitor read for split C (`monitor_read_C.lock`, 2026-10-13T13:13:49Z, y sha `5720bd1a…`, recorded `prediction_manifest_sha256` proving all 10 prediction files were frozen first) and exactly one declared Locked read (`locked_read_C.lock`, 2026-10-13T13:35:17Z, y sha `2dabe97a…`). Two declared Monitor scoring rounds are recorded in `monitor_rounds_C.json` (GLRC, then the pre-registered conditional DLP purchase). The Locked read happened only after both decisions and the stop were fixed.

## 2. Preregistered designs and controls

**GLRC** (Stage A, 413 539 params, +34): `W'_b = W_null_b + tanh(θ0 + θᵀg_G)·(W_local_b − W_null_b)`, likewise `V`. `g_G` is the existing Fit-only standardised 33-dim global structure vector. Zero-init → `λ(G) ≡ 0`, initial forward bit-identical to BJOC-Null. Conservation `Σ_b ΔW_b = Σ_b ΔV_b = 0` per molecule.

**DLP** (Stage B, conditional, 441 153 params, +27 648): `h_i = P a_i` (rank 32), `p_k(G)` = mean of `h_i ⊙ h_j` over unordered pairs at shortest-path distance d=3, d=4, d≥5, appended to the frozen reader input with zero-init new columns. No iterative message passing.

Both candidates were trained with the frozen R3 recipe, the **same Fit-only masked-SSL encoder per seed as their matched P3 baseline**, the same batch order, and on the same GPU device per seed. The training-loop equivalence test proves `ab.train_supervised_arch` with `model_kind="base"` reproduces `bp.train_supervised` soups and batch orders bitwise. Recorded init parity for seed 0: P3 and GLRC share `init_full_sha256 d06d5e16…`, `init_encoder_sha256 91af980a…`, and batch hash `d142e307…` (`full_matches: true`).

## 3. CPU correctness gates (before any GPU trajectory)

Both gates passed on `res` and were re-checked locally (`cpu_gate_glrc.json`, `cpu_gate_dlp.json`):

* **GLRC**: exact identity at λ=0 (base parameter hash equal, forward diff ≤1e-5), gate gradient active, active-vs-λ=0 prediction difference nonzero, conservation `ΣdW ≤3.9e-6`, ownership counterexample (identical Null marginal statistics, local residual difference 3.1e-2, different active prediction), atom/bond/junction/row/batch invariances ≤1.2e-7, 413 539 params, no forbidden modules.
* **DLP**: brute-force pair-construction cross-check, information witness (same first/second moments, different pair features), zero-init identity with old reader columns copied, non-degenerate features, reader-new-column gradient at step 0 and pair-projection gradient from step 1, prediction change, recoverability probe R² median 0.723 ≤ 0.90, invariances ≤1.5e-5, 441 153 params, no forbidden modules.

## 4. Monitor results (split C, 1000 molecules, single read, seed-paired)

Baseline P3 seed-mean Monitor MAE **0.081907** (per-seed 0.080237 / 0.083578). Gate (frozen): Δ ≤ −0.004, both seeds negative, bootstrap CI upper < 0, ordinary not worse, top-10-removed negative, RMSE within +0.005, no single-molecule dominance.

| candidate | seeds | seed-mean MAE | Δ vs P3 | paired CI95 | per-seed Δ | ordinary Δ | tail Δ | RMSE Δ | top-10 Δ | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| GLRC | 0, 1 | 0.083154 | **+0.001247** | [−0.001145, +0.003644] | +0.001670 / +0.000824 | +0.001311 | +0.000487 | +0.006671 | +0.000672 | **FAIL** (6/7) |
| DLP | 0 | 0.083348 | **+0.002204** | [−0.001134, +0.005518] | +0.002204 | +0.002855 | −0.005590 | +0.000408 | +0.002334 | **FAIL** (4/7) |

* `GLRC_NO_GAIN`: both seeds are *worse*, the CI is positive-crossing, and the degradation is broad (ordinary and top-10-removed both positive; RMSE worse by 0.0067).
* `DLP_NO_GAIN`: seed 0 is worse overall; the only positive signal is the tail (Δ −0.0056, mixed with ordinary +0.0029).
* Per the frozen purchase rules: ORGANIZED was **not** purchased (GLRC failed), DLP seed 1 was **not** purchased (seed-0 gate failed), split-D replication was **not** purchased (no candidate passed a primary gate), the full-10k refit was **not** purchased. DLP seed 0 itself was the pre-registered conditional purchase triggered by the GLRC failure. No search of any kind followed the read.

## 5. Locked final (single read, after the stop was fixed)

| arm | seeds | locked MAE | median abs | p90 abs | RMSE |
|---|---|---|---|---|---|
| P3 | 0, 1 | 0.104454 | 0.049514 | 0.172122 | 0.589199 |
| GLRC | 0, 1 | 0.103430 | 0.050496 | 0.175816 | 0.586377 |
| DLP | 0 | 0.104691 | 0.047475 | 0.185094 | 0.579022 |

Paired contrasts vs P3 (locked):

* **GLRC**: Δ **−0.001024**, CI95 [−0.003390, +0.001288], per-seed −0.000730 / −0.001318; ordinary +0.000349, tail −0.017253, RMSE −0.002822, top-10 −0.000914, 51.3% molecules better.
* **DLP** (seed 0): Δ **−0.000348**, CI95 [−0.003984, +0.003597]; ordinary +0.000149, tail −0.006220, RMSE −0.003657, top-10 −0.001369, 50.8% molecules better.

**Monitor/Locked contradiction (disclosed):** GLRC is worse than P3 on every Monitor seed but better on every Locked seed; DLP agrees in sign with GLRC on both zones. Both effects are ≤0.0022 in magnitude, two orders below the amount needed for the promotion gate (0.004) and far below the historical masked-SSL gain (≈0.012). The frozen rule is: disclose and stop, never retune on Locked. No decision was changed after the Locked read, and no candidate is promoted.

## 6. Mechanism reading (`analysis_C.json`)

* The GLRC gate is genuinely active, not dead: λ(std) ≈ 0.40 (seed 0) / 0.18 (seed 1) over 2000 held-out molecules, spanning [−0.97, +1.00]; the learned gate direction is dominated by ring-system and size descriptors (`ring_size8plus_frac`, `log1p_n_atoms`, `log1p_n_junctions`, `n_ring_systems`, …), and the top directions **differ between seeds** (seed 1: `bond_type3_frac` negative, `ring_size8plus_frac` positive) — the gate direction is not seed-identifiable at this noise level.
* Both candidates move held-out predictions by ~0.035–0.041 mean absolute, i.e. the channels are consumed, but neither reduces error. `corr(Δpred, λ)` is only 0.02–0.08, so the per-molecule gate value does not translate one-to-one into the output shift.
* On Locked (seed 0, descriptive), GLRC's ΔMAE is mildly negative in the higher-λ quartiles (−0.0018 in q3 vs +0.0008 in q0) and roughly uniform across junction-count halves — consistent with a real but tiny direction that the Monitor noise dominates. DLP's tail is the only consistently positive subgroup, but it is offset by the ordinary-molecule degradation on Monitor, and it does not reproduce as a gate-level effect.
* Confound control: the comparison is capacity-matched to the architecture question (P3 vs GLRC differ by exactly the 34 gate parameters; DLP buys +27 648 parameters *and* the distance information, so a DLP gain would not have been attributable to information alone — it showed none anyway).

## 7. Answers to the paper-facing questions

**(A) Which architecture, magnitude, and attribution.** Neither. GLRC is +0.00125 (Monitor) / −0.00102 (Locked) and DLP is +0.00220 (Monitor) / −0.00035 (Locked, seed 0); all CIs cross zero, signs are not stable across zones, and every seed is on the wrong side of the −0.004 promotion threshold. The conditional ORGANIZED contrast was never purchased because the frozen rule ties it to a GLRC pass, so the question "is GLRC explained by a fully-Organized incidence readout?" was not answered by experiment; it is explicitly **not decided** rather than answered in the negative. Which signals are weak: the per-molecule global-conditioned gate (λ) and the distance-conditioned pair moments.

**(B) Provenance of the effect.** Both channels are real and active (CPU-gated information witnesses, nonzero gradients, gate spread over held-out molecules, prediction shifts ~0.04), and the comparison isolates the architecture question from optimization (shared init, encoder, batch order). The effect is not a capacity artifact for GLRC (capacity-matched) and not single-molecule dominance (fraction ≤0.026). The evidence of *useful* local-ownership or non-local-distance signal is essentially zero at this scale; the small locked-side tail/quartile movements are consistent with noise plus a weak direction that the Monitor contrast does not confirm.

**(C) The 0.059 tier.** Adding either principled missing channel — true junction–bond ownership with an adaptive gate, or explicit d=3/4/≥5 pair products — does not move the system toward the 0.059 tier. Together with the historical observations that the masked-SSL gain (~0.012) is the only robust architecture-adjacent lever found so far, the working hypothesis is that the remaining error is dominated by label/data-limited generalization (per-molecule noise and finite 8000-row Fit fits), not by missing incidence or distance features of the Null-composer family. This round cannot prove that hypothesis: the internal splits reuse historically trained molecules, and the frozen best model's official-test 0.069822 is a different-data endpoint.

**(D) Evidence strength and external validation.** For a paper claim, this round is a **well-controlled negative result**: two pre-registered one-shot architecture hypotheses, matched baselines, exact init/order parity, CPU information gates, one-shot Monitor/Locked reads, and a clean stop rule. It establishes that these specific 34- and 27 648-parameter additions do not beat BJOC-Null at 8k Fit rows under R3. It does not establish a population-level claim, and it does not validate on a second molecule task. Recommendation for the next round: do not iterate the ownership/distance feature family; first decide the target level of evidence (label-noise floor, external unlabelled data, or a second molecular endpoint such as a different property/task) and preregister a matched-budget comparison against the frozen P3 system on that endpoint.

## 8. Round decisions and artifacts

* Decisions: `GLRC_NO_GAIN`, `DLP_NO_GAIN`; stop per preregistration. Not purchased: ORGANIZED, DLP seed 1, split D, full-10k refit, matched-capacity DLP control.
* Key artifacts: `protocol_freeze.json`, `input_audit.json`, `split_C.json`, `objects_manifest_C.json`, `prep_fit_C.json`, `stage_a_audit.json`, `cpu_gate_glrc.json`, `cpu_gate_dlp.json`, `pretrain_C_s{0,1}.json`, `train_{P3,GLRC}_C_s{0,1}.json`, `train_DLP_C_s0.json`, `prediction_manifest_C.json`, `monitor_read_C.lock`, `monitor_rounds_C.json`, `monitor_{GLRC,DLP}_C.json`, `locked_read_C.lock`, `locked_manifest_C.json`, `locked_final_C.json`, `analysis_C.json`, `gpu_budget_ledger.json`, `post_freeze_amendments.json`.
* Post-freeze amendment (disclosed in `post_freeze_amendments.json`): the locked finalizer was fixed to read each arm's seed set from the frozen manifest, because the pre-registered conditional purchase left DLP at one seed. The fix changed no model, prediction, gate, threshold or selection; `ssl`/`arm`/`full10k-train` still verify the frozen code hash and were not rerun.
* The historical full-10k P3 official-test 0.069822 was used as background only and never as a selection signal.
