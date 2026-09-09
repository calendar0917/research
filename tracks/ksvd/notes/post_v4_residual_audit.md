# Post-v4 Residual Audit (ZINC-12k, frozen compact-v4-hinge)

**Status:** FINAL — 2026-09-10
**Questions answered:** Q1–Q14 · probe families A–G · Figures 1–6
**Decision:** **NO CLEAR SECONDARY SIGNAL** (no fitted non-oracle probe reaches even the Weak band; see §37)
**Mode:** diagnosis-only. v4-hinge is frozen as the benchmark baseline; **no model was trained, no v5 was designed, and the official test split was never loaded.**

---

## 0. Executive summary

| Fact | Value | Status |
|---|---|---|
| Frozen baseline | compact-v4-hinge, seeds 0–3, serial runs 20260909-194445/200320/201918/203516 | VERIFIED (valid MAE bit-equals `best_mae`, 0.1700656/0.1631665/0.1741492/0.1704209; epochs 53/48/56/60; 99,613 params) |
| Frozen-inference reproduction | 4/4 seeds reproduce recorded run predictions **bit-exactly** (max abs diff 0.0) from cached records + selection-state checkpoints | VERIFIED |
| Subgroup residual mass (valid, ensemble) | A (excess 0, n=965): MAE 0.1186 · B (excess 1, n=30): 0.1847 · C (excess ≥2, n=5): 6.10 | — |
| Residual ↔ z_cycle (all valid) | Pearson ≈ 0.72, 4/4 seeds — the *already-accounted-for* long-cycle mechanism (Group A z_cycle ≈ 0 by definition) | CONFIRMED (context, not new signal) |
| Group A signed bias vs z_SA | none: per-seed Pearson −0.027…+0.068, ≤2/4 unstable | VERIFIED |
| Group A **difficulty** vs z_SA | monotone, 4/4 seeds: MAE 0.173 (low-SA quintile) → 0.072 (high-SA quintile); Spearman(ens) −0.359; partial given z_logP −0.314 | VERIFIED (variance side only) |
| Group A signed bias vs z_logP | weak positive, 4/4 seeds: Pearson mean ≈ +0.052; quintile drift −0.032 → +0.043 (top bin 4/4 seeds positive) | VERIFIED (small; not correctable) |
| Group A difficulty vs z_logP | ~0 after controlling z_SA (partial −0.03); the logP difficulty gradient is an SA proxy | VERIFIED |
| Rarity (rare≤5 ratio) | abs-error Spearman 0.28–0.32/seed (4/4), ens 0.41; **survives** component control (partial +0.32); signed side nil (≤3/4, ens −0.05) | VERIFIED (variance side only) |
| Composition lead | "N H1 +" (atom-type 8, protonated N) fraction: signed Spearman +0.109 ens (4/4; per-seed +0.01…+0.16), **partial +0.18 given z_SA+z_logP** — only composition signal surviving component control; present in 135/965 Group-A molecules (Δ mean residual +0.054 vs absent) | VERIFIED (small, sparse) |
| Exact patch/relation/unary hash probes (7) | ΔMAE −0.0006…−0.0010, R² ≤ 0 | NO-GO |
| Continuous pair h_i⊙h_j (240-D, 4 seeds) | ΔMAE −0.0006 ens, all seeds ≤ +0.0004 | NO-GO |
| 1D valid-internal-CV probes (19 incl. composition) | best R² +0.017 (cycle_rank isotonic, ΔMAE −0.0005); best **positive** ΔMAE across all fitted probes +0.00007 | NO-GO (gate 0.003) |
| Frozen-state statistics (Probe G) | no stat |Spearman| ≥ 0.21 with sign stability across seeds | NO-GO |
| Systematic difficulty (not seed noise) | Group A ensemble MAE 0.1186 = 86% of mean per-seed MAE 0.1374; Spearman(mean_abs, residual_std across seeds) 0.556 | VERIFIED |

Bottom line: the residual variance inside ordinary Group A is **systematic, structured, and mostly irreducible for a fixed-objective patch model** — it concentrates on the low-z_SA axis and on rare-patch molecules (heteroscedastic *difficulty*, no signed bias), while signed bias is limited to (i) a small logP-side under-prediction of the top quintile and (ii) a sparse N-H1+ composition hint. No graph-observable feature family that a v5 could consume exposes a second-layer structural signal large enough to clear the pre-registered gates.

---

## 1. Scope, frozen objects, protocol

* Audit module: `tracks/ksvd/experiments/luyin16/zinc_post_v4_residual_audit.py`
  (stages `residuals → master → attribution → probes → pairs → states → state_probes → cross → summary → decision → figures`).
* Artifacts: `tracks/ksvd/results/post_v4_residual_audit/` (stage markers, per-molecule CSV, probe/cross/summary tables, decision record, figures).
* Protocol `zinc-context-gap`, code state 0a61b23 + this module. Freeze constraints honored: v4-hinge runs, topology features, loss/optimizer/LR/epochs/head, and the frozen Group definitions (A = label-excess 0, B = 1, C = ≥2) were **not** touched; no new model trained; **test never loaded** (`load_test=False` everywhere; the audit reads train records only for vocabulary/frequency context, exactly as the frozen runs did).
* All residual analysis uses the **validation** split. There is no OOF-train residual under the freeze (would require 5-fold retraining), so every fitted probe is labelled **VALID-INTERNAL-CV (descriptive diagnostics only)** — per pre-registered fallback, these numbers never enter GO claims; GO thresholds apply to them *as if* OOF only because they were gated at 0.003+, far below any benefit of OOF vs valid-internal-CV.

### Frozen run inventory
| Seed | Run id | valid MAE (bit-verified = best_mae) | epochs | checkpoint |
|---|---|---|---|---|
| 0 | 20260909-194445-182c7021 | 0.1700656 | 53 | `legacy_full_result_selection_state.pt` |
| 1 | 20260909-200320-34b347bf | 0.1631665 | 48 | same |
| 2 | 20260909-201918-45fbe48d | 0.1741492 | 56 | same |
| 3 | 20260909-203516-04e62a28 | 0.1704209 | 60 | same |

Group A per-seed valid MAE: 0.1378 / 0.1323 / 0.1389 / 0.1408; 4-seed ensemble (mean prediction) MAE 0.1186.

## 2. Reused frozen artifacts (provenance)

* `results/zinc_long_cycle_audit/label_effective_cycle.csv` → y (target bit-matches run targets on valid), raw logP / SA, label-implied cycle, frozen `z_cycle` (`label_normalized_cycle_component`), frozen subgroup excess. Cycle-order mismatches 0 on valid; reconstruction error p50 0.0019 / p95 0.0059 / max 0.0416 (Group A max 0.0091). Reconstruction constants VERIFIED: z_logP μ=2.4570953 σ=1.4351123, z_SA μ=−3.1922245 σ=0.8321043, z_cycle μ=−0.00013341 σ=0.2883163 (y = z_logP + z_SA + z_cycle in the benchmark label generator).
* `results/information_gap_audit/width/` per-molecule 136-col feature CSV (aligned by molecule_id) — rarity, size, graph complexity, atom/bond composition (atom-type bins from the graphdeeplearning `atom_dict`, verified against `data/ZINC/raw/atom_dict.pickle`; 0 C, 1 O, 2 N, 3 F, 4 C H1, 5 S, 6 Cl, 7 O −, **8 N H1 +**, 9 Br, 10 N H3 +, 11 N H2 +, 12 N +, 15 I, 16 P …).
* `results/information_gap_audit/hashed_features.npz` (2048-D md5-hash unary/pair/relation log1p + presence; vocab 6785 incl. OOV, radius 2) — bit-identical to rebuilds from cached `graph_records_{train,valid}.pkl.gz`.
* New frozen-inference cache: `results/post_v4_residual_audit/cache/v4_records_*.pkl.gz` (rebuilt with `_load_zinc` + `matrices_for_split(…, "hinge")` + mirrored `PatchPathModel`); forward passes under the frozen selection checkpoints reproduce the recorded predictions bit-exactly (gating performed by the audit itself).

## 3. Residual / master table (§4–§5)

`validation_per_molecule_residuals.csv` (all 1,000 valid molecules): target, per-seed prediction/residual/abs error, `mean_prediction`, `mean_residual`, `mean_abs_error`, `residual_std_across_seeds`, subgroup. `validation_master_table.csv` adds the frozen y-components and all probe features.

## 4. Target-component attribution (oracles, Group A) (§6–§11)

Attribution is correlation/binning only; target components are **oracles — never future model features**.

| Component | signed residual (ens) | consistency | abs error (ens) | consistency | reading |
|---|---|---|---|---|---|
| z_SA | Pearson +0.005 (seeds −0.027…+0.068) | ≤2/4 unstable | Spearman −0.359 | 4/4 same sign | pure *difficulty* axis; no signed bias |
| z_logP | Pearson +0.058 | 4/4 same sign | Spearman −0.186 | 4/4 same sign | weak positive signed drift; abs side ≈ SA proxy (partial given z_SA −0.03) |
| z_cycle | n/a in A (no variation); all-valid Pearson ≈ 0.72 | 4/4 | — | — | the known long-cycle mechanism (v4 already moves it; B/C improved +0.0092 test MAE vs v2) |

Quintile facts (Group A, ensemble): z_SA bins → MAE 0.173/0.151/0.104/0.093/0.072 (monotone, 4/4 seeds), signed means +0.012/−0.047/−0.004/−0.006/+0.014 (no monotone bias). z_logP bins → signed means −0.002/−0.032/−0.028/−0.012/**+0.043** (top bin positive 4/4 seeds: +0.039/+0.054/+0.040/+0.040), MAE 0.139/0.134/0.108/0.106/0.106 (flat). The difficulty gradient is **not** a y-level effect: in the two-way median grid, MAE is 0.149–0.161 whenever z_SA is low (target means −1.69 and +0.43!) vs 0.077–0.096 whenever z_SA is high.

## 5. Bias vs heteroscedastic-variance decomposition

* Signed-residual distribution (Group A, ens): mean −0.0062, median +0.0035, std 0.193, skew +0.13, kurtosis 12.9 → near-symmetric with a long tail; mean absolute 0.1186 vs median absolute 0.072 → the MAE mass is tail-driven.
* Errors are **systematic across seeds**, not seed luck: ensemble MAE / mean per-seed MAE = 0.1186/0.1374 = 86%; Spearman(mean_abs_error, residual_std_across_seeds) = 0.556.
* Therefore "z_SA difficulty" and "rarity difficulty" are *stable per-molecule difficulty* (heteroscedastic variance), not random noise patterns, and *not* correctable signed bias: per-seed isotonic corrections on z_SA give ΔMAE −0.0008 and on rarity −0.0017…−0.0019 (everything below the Weak band). This is the honest wording: **no fitted probe exposes a correctable bias; the structure is conditional variance.** (The L1 head is *not* called "mean regression"; the phenomenon is systematic under-representation of extreme-magnitude targets in the tail, i.e. heavy-tailed conditional distributions, which L1 is entitled to by design.)

## 6. Ordinary-only probes A–G (§12–§22) — all on Group A, 4 seeds

Thresholds (§21): Strong ≥0.008 & ≥3/4 seeds; Candidate ≥0.005; Weak 0.003–0.005; NO-GO <0.003. (All ΔMAE below are valid-internal-CV deltas; a non-oracle fitted probe would additionally need ≥3/4 same-sign seeds to be credible.)

| Family | Evidence | Verdict |
|---|---|---|
| A rarity/OOV | abs-error Spearman 0.28–0.32/seed rare≤5 (4/4), ens 0.41 (oov 0.33); signed ≤3/4 and ≈ −0.05 ens | NO-GO as bias; confirmed *variance* signal (1D CV ΔMAE −0.0017) |
| B graph complexity | abs-error negative 4/4 (diameter −0.21/−0.18/−0.21/−0.17, ens −0.27); signed ≤4/4 but |ρ|≤0.14 & partial ≈ 0 after component control | NO-GO (1D CV ΔMAE −0.0015) |
| C atom/bond composition | signed leads: N H1 + (+0.109 ens, 4/4; per-seed +0.131/+0.011/+0.064/+0.161), N H3 + (−0.096, 4/4), I (−0.088, 4/4 but 4 molecules), double-bond fraction (−0.020, 3/4); abs leads ≈ component proxies | NO-GO as correction; one surviving signed hint (N H1 +, partial +0.175 given both components; 1D CV ΔMAE −0.0011) |
| D exact patch-pair hash (2048-D, unary/pair/relation/presence/log1p) | R² ≤ 0; ΔMAE −0.0006…−0.0010 | NO-GO |
| E relation-conditioned pairs | not stronger than pairs (ΔMAE ≈ −0.0006) | NO-GO |
| F continuous pair h_i⊙h_j (240-D frozen-state, 5-bucket def) | ridge 5-fold: ens ΔMAE −0.0006, R² −0.003; per-seed −0.0004…+0.0004 | NO-GO |
| G frozen-state statistics (norms/deltas/centroid, per seed + ensemble) | best single |Spearman| 0.206 (unified_norm seed 3), no stat stable across seeds at useful magnitude | NO-GO |

Representation-state summary CSV per seed (`state_summary_seed{s}.csv`) gated: predictions bit-exact and patch/pair counts equal to the master table.

## 7. Cross / partial analysis (§23–§27)

Partial Spearman, Group A, ensemble (controls in parentheses):

| relation | partial | uncontrolled | meaning |
|---|---|---|---|
| abs error ↔ z_SA (given z_logP) | −0.314 | −0.359 | SA difficulty is independent of logP |
| abs error ↔ z_logP (given z_SA) | −0.032 | −0.186 | logP difficulty ≈ SA proxy |
| abs error ↔ rarity (given z_SA+z_logP) | +0.317 | +0.409 | rarity difficulty independent of components |
| abs error ↔ z_SA (given rarity) | −0.248 | −0.359 | SA difficulty partly independent of rarity |
| abs error ↔ diameter (given z_SA+z_logP) | −0.191 | −0.271 | small residual size link (mostly rarity proxy: given rarity −0.158) |
| signed residual ↔ N H1+ (given z_SA+z_logP) | **+0.175** | +0.106 | only composition signal surviving control |
| signed residual ↔ z_SA (given z_logP) | +0.028 | +0.034 | no SA-side bias |
| signed residual ↔ z_logP (given z_SA) | +0.052 | +0.069 | logP-side drift survives control |

Group-A intercorrelation context: z_logP↔z_SA Spearman +0.445; z_SA↔rarity −0.365; diameter↔num_nodes +0.81. Target quintiles (§ distribution): bottom quintile mean signed −0.010 / MAE 0.153; mid (−1.24…+0.06) MAE 0.172 (hardest band); top quintile +0.031 / 0.077 — again a difficulty picture, not a y-trend bias.

Error distribution & top-30 (`groupA_error_distributions.csv`, `top30_groupA_molecules.csv`): Group A residuals are heavy-tailed within A (kurtosis 13–29/seed), yet the top-30 worst molecules (MAE 0.4–1.6) share the expected profile — low–mid targets, low z_SA, rare-patch rich (rare≤5 ratio 0.06–0.40), larger graphs — with both signs represented; no new structure appears under manual inspection (Fig. 6).

## 8. Bottleneck ranking (§36)

1. **Heteroscedastic conditional-variance difficulty on the low-z_SA axis** (every z_logP level; 4/4 seeds; partial −0.31). Variance side only — no model feature consumed it in this audit as a correctable bias.
2. **Rare-patch difficulty** (independent of components; partial +0.32; 4/4 seeds) — a training-coverage effect; in-train rarity difficulty already documented in the v2 information-gap audit.
3. **Small signed drifts**: z_logP top-quintile under-prediction (~+0.04, 4/4 seeds) and N-H1+ composition hint (+0.05–0.06 mean-residual shift on 135 molecules). Both far below any correction gate and only partly graph-observable.
4. Long-cycle tail B/C: still the largest *per-molecule* errors (C: MAE 6.10 on 5 molecules) — but v4 already addressed it (+0.0092 mean test paired); this audit adds nothing actionable there beyond the known order-dependence artifact documented in the long-cycle audit.

## 9. Decision (§37) — decision tree walked for Group A

Every decision-tree branch was evaluated against the pre-registered gates. Outcome: **NO CLEAR SECONDARY SIGNAL** — no non-oracle fitted probe clears the Weak 0.003 band on any seed (best positive ΔMAE across all 29 fitted rows ≈ +0.0001; best R² +0.017 on the *non-monotone* cycle_rank isotonic, ΔMAE −0.0005). The tree's GO branches (structural-representation GO: need consistent ≥0.005 structural signal; objective/calibration GO: need ≥0.003 non-oracle structural Δ or a reproducible signed bias ≥0.003) are **all** unmet; "no clear secondary signal" is therefore the mandated outcome, not an abstention. Supporting oracle-side heterogeneity (SA/rarity difficulty) is real but sits outside every GO branch because it is variance-side only and no observable proxy fixes it.

### Q1–Q14 (condensed; full text in `decision_record.json`)

| Q | Answer |
|---|---|
| Q1 where is residual mass | Group A still carries most MAE mass by count; largest per-molecule errors remain B/C (C ens MAE 6.10, n=5) vs A 0.119 |
| Q2 Group A error distribution | mean −0.0062, median +0.0035, std 0.193, skew +0.13, kurt 12.9 (ens); heavy-tailed but symmetric; MAE is tail-driven |
| Q3 z_SA attribution | difficulty-only: abs −0.359 (4/4; partial −0.31); signed nil (≤2/4) |
| Q4 z_logP attribution | weak signed +0.052 mean Pearson (4/4; partial +0.052); abs −0.186 (4/4) but ≈ SA proxy (partial −0.03) |
| Q5 strongest component | none for signed residual; difficulty axis = z_SA (with z_logP as proxy), signed drift = z_logP side |
| Q6 rarity | yes for abs (0.28–0.32/seed, 4/4; partial +0.32) — variance only; fitted ΔMAE ≤ −0.002 |
| Q7 branching/complexity | abs-negative 4/4 (diameter ens −0.27); ≈ proxy of components/rarity; no signed structure after control |
| Q8 composition | N H1+ (135/965 molecules): signed +0.109 ens (4/4), partial +0.18 — survives; fitted ΔMAE −0.0011; multiple-comparison caveat over 18 features |
| Q9 exact pairs | no (ΔMAE ≈ −0.001, R²≤0; same as v2) |
| Q10 relation-conditioned | no (not stronger than pairs) |
| Q11 continuous pair h_i⊙h_j | no (240-D, ens ΔMAE −0.0006; all seeds ≤ +0.0004) |
| Q12 3/4+ findings | (1) SA-axis difficulty 4/4; (2) rarity difficulty 4/4; (3) logP signed drift 4/4 (small); (4) state statistics: none |
| Q13 most credible bottleneck | systematic heteroscedastic difficulty on low-z_SA + rare-patch molecules; not a correctable bias; B/C tail remains a separate severity issue |
| Q14 v5 recommendation | **NO CLEAR SECONDARY SIGNAL**; if a v5 is ever opened it must be an objective/calibration (heteroscedastic- or density-aware) study — explicitly not another representation channel |

### Figures (results/post_v4_residual_audit/figures/)
fig1 Group A mean residual vs z_SA (binned, flat ≈ 0); fig2 mean residual vs z_logP (binned, top-bin drift); fig3 abs residual vs rarity (monotone increase); fig4 abs residual vs branching (weak decrease); fig5 best non-oracle 1D probes predicted-vs-true residual (cycle_rank isotonic, + rare_le5 alternate — scatter is a flat noise band, no diagonal structure); fig6 top-30 feature heatmap (robust-normalized).

## 10. Reproducibility

```
python -m tracks.ksvd.experiments.luyin16.zinc_post_v4_residual_audit \
    residuals master attribution probes pairs states state_probes cross summary decision figures
```
Stages are idempotent via `stage_*.json` markers (use `--force` to rerun). Records cache in `results/post_v4_residual_audit/cache/` (~2×58 s first build). The audit self-gates: predictions must equal recorded run predictions bit-exactly; patch/pair counts must equal master counts; test split never loads.

## 11. Follow-ups (none pre-committed)

* Recorded as revisit conditions only: (i) if an N-H1+-conditioned correction ever exceeds +0.003 on an OOF-train residual under a *new* training protocol (permitted only when the freeze lifts); (ii) if a heteroscedastic/objective study becomes permissible, the z_SA-axis and rarity-axis difficulty documented here is its target list; (iii) long-cycle C remains a severity/order-artifact issue, not a residual-audit one.
