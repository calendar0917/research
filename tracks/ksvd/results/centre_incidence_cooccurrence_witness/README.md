# Centre-incidence co-occurrence witness audit — output manifest

**Frozen-feature witness diagnostic** on the frozen compact-v4-hinge OOF states
(official TRAIN molecules only; official valid/test never loaded). It asks whether the
**cross-channel relation co-occurrence** that the current per-centre mean/std compression
of incident pair states provably discards — the off-diagonal population covariance
`c_{i,b} = offdiag Cov(q_ij)` ∈ R^120 per (centre, distance bucket) — carries stable,
task-relevant signed predictive value beyond the existing 302D graph vector `R` **and**
beyond a matched marginal-control pathway built from the centre statistics the model
already consumes.

No backbone is trained; compact-v4 is unchanged; the pair encoder is unchanged; there is
no second centre update; no attention and no higher-order GNN.

Full scientific note: `tracks/ksvd/notes/centre_incidence_cooccurrence_witness_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_centre_incidence_cooccurrence_witness.py`
(stages `inventory → spec → export → gate0 → splits → ambiguity → stage1 → stage1b →
stage2 → surrogate → figures → decision → all`).
Tests: `tracks/ksvd/tests/test_centre_incidence_cooccurrence_witness.py` (12 tests, all pass).

CSV/JSON/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## v4 state export extension

New cache version `frozen_state_export_v4_centre_incidence`, built from the *same* frozen
OOF checkpoints (nothing retrained). The existing v3 export (`u_i`, `q_ij`, pair indices,
buckets, raw relation, `R`, `yhat_0`) was **sufficient** to reconstruct the centre
incidence assignment; the v4 export adds the centre-path fields that make the Gate-0
reconstruction gates directly checkable:

- `patch_states_pre` — the 48D patch state entering `center_update` (also the input to the
  pair projection);
- `patch_states_post` — `patch + center_update([patch ; centre_context])`;
- `center_context` — the true forward 165D per-centre context (5 buckets × [mean 16 ;
  std 16 ; log1p count 1]).

Cache fingerprint includes export version, tokenizer version, config fingerprint,
vocabulary fingerprint, split fingerprint, checkpoint SHA-256, backbone seed, fold, and
the two deterministic random-projection fingerprints.

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, state `.pt` SHA-256, params, oof_mae); seeds [0,1] complete |
| `centre_feature_spec.json` | real centre-update computation graph + centre summary definition + covariance/marginal witness spec |
| `centre_export_integrity_report.json` | Gate-0 checks G0.1–G0.8 on 5/5 folds + pair-state sparsity diagnostic; `all_passed: true` |
| `fold_split_manifest.json` | per-fold target-independent molecule-ID-hash split (1200 fit / 400 selection / 400 evaluation, *identical* to the frozen-readout audit) + common-input bulk threshold |
| `stage_export.json` | v4 export summary + fingerprints |
| `representation_ambiguity.json` | target-free marginal↔covariance ambiguity (k=16 nearest neighbours in `m`-space); descriptive only, NOT a performance gate |
| `adapter_depth_sensitivity.json` | fixed, non-swept R-only depth sensitivity (1 vs 2 hidden layers at matched budget) quantifying the depth confound |
| `stage1_fold_results.csv` | per-fold `B0 / B1(matched-budget R-only) / B1_hist(historical R-only) / B2(marginal control) / E(covariance witness)` MAE, `ΔR`, `ΔR_hist`, `ΔM`, bulk safety, params, viability |
| `stage1_bootstrap.json` | stratified molecule-level paired bootstrap for `ΔR`, `ΔR_hist`, `ΔM` + aggregates |
| `final_decision.json` | final verdict + criteria + case + claim wording + scope caveat |
| `state_exports/frozen_state_export_v4_centre_incidence_fold{0-4}_seed0.npz` | per-molecule v4 states + fingerprint |
| `figures/figure1_representation_ambiguity.png`, `figures/figure2_per_fold_delta.png` | marginal-vs-covariance ambiguity; per-fold `ΔR`/`ΔM` (Figure 3 is Stage-2/surrogate-only and neither ran) |

`stage2_fold_results.csv`, `pooled_results.csv`, `final_bootstrap.json` and
`channel_shuffle_control.csv` are **not** produced: Stage 1 was a pre-registered CLEAR
NO-GO, so the second backbone seed, the second init, and the surrogate control were not
spent.

## Centre-incidence computation graph (verified from the real code)

- each unordered pair row contributes `q_ij` **identically to both endpoint centres**
  (`torch.cat([source, target])` duplication in `_pool_pairs_to_centres`);
- `bucket = min(max(distance,1),5) − 1`, 5 buckets for distances 1,2,3,4,5+;
- per (centre, bucket): `mean(q)` (16), `std(q)` (16, population: `E[q²]−mean²` clamped ≥0,
  `sqrt(var+1e-8)`, masked to 0 when the bucket is empty), `log1p(count)` (1) ⇒ 33D,
  5 × 33 = **165D**;
- centre update: `patch ← patch + MLP([patch ; 165D])` with
  `MLP = Linear(213→60) → LayerNorm → ReLU → Dropout → Linear(60→48)`, final projection
  zero-initialised — a **residual** connection; the unary readout is recomputed afterwards.

## Key numbers (frozen OOF backbone seed 0, 5 folds, 1 adapter init)

- Gate 0: **all pass 5/5** — pair grouping; each unordered pair consumed exactly once to
  **each** endpoint (self-loops 0, duplicates 0, `2·n_pairs` incidence); bucket identity
  consistent with the relation one-hot *and* `log1p(distance)`; reconstructed 165D context
  max `3.98e-4` (mean `9.09e-9`, entirely float32 variance cancellation in near-degenerate
  cells); reconstructed `h'_i` max `5.74e-6`; reconstructed `R` max `1.14e-5`;
  reconstructed prediction max `9.5e-7`; incident-pair row-order invariance `5.4e-19`;
  batch invariance `R 0.0`, `yhat_0 4.8e-7`, centre context `0.0`.
- Pair-state sparsity (frozen representation, not an export artifact — independently
  reproduced from the v3 export): `q_ij` has 86–97% exactly-zero entries; only 3–29% of
  cells with `n ≥ 2` have a nonzero covariance row.
- Representation ambiguity (target-free, `k = 16`, informative cells): `m`-space nearest
  neighbours are **0.282×** as far in covariance space as random cells;
  `Pearson(‖Δm‖,‖Δc‖) = 0.60`, `Spearman = 0.68`; normalized conditional
  `Var(c | 16-NN m) = 0.285`. Mild-to-moderate ambiguity — present, but not the dominant
  structure, and **not** a GO gate.
- Adapter-evaluation MAE mean over folds:
  **B0 0.19316 / B1 matched-budget R-only 0.20248 / B1_hist historical R-only 0.19050 /
  B2 marginal control 0.20019 / E covariance witness 0.20156**.
- `mean ΔR = MAE(B1) − MAE(E) = +0.000922` (95% CI `[−0.00447, +0.00619]`, `P(>0)=0.64`,
  3/5 folds positive) — below the `+0.0005`…`+0.002` borderline band.
- **`mean ΔM = MAE(B2) − MAE(E) = −0.001361`** (95% CI `[−0.00569, +0.00279]`,
  `P(>0)=0.27`, **2/5** folds positive) — the decisive number: the covariance witness is
  **worse than the matched marginal control**.
- `mean ΔR_hist = MAE(B1_hist) − MAE(E) = −0.011056` (**0/5** folds positive,
  CI `[−0.01610, −0.00612]`).
- Adaptors are non-collapsed (witness sensitivity min `0.601`, prediction std `1.84`), and
  E is worse than B2 on the selection split in **5/5** folds — so this is not evaluation
  noise. E is also worse than B2 on the common-input bulk (max `+0.0077`, gate ≤ 0.002,
  FAIL), i.e. the covariance pathway is slightly harmful rather than merely uninformative.
- Decision: **NO-GO** — `Case B + Case C`
  (`records/decisions/decision-centre-incidence-cooccurrence-witness-nogo-20260912.yaml`).
  The second frozen backbone seed, Stage-1b init, and channel-shuffle surrogate were
  **not** spent. Do **not** build a covariance architecture, do **not** enlarge or learn
  the projection, do **not** use full covariance, and do **not** reach for attention or a
  relation-set Transformer.

## Documented confound (strengthening, not rescuing, the negative)

The pre-registered matched-budget 1-layer R-only head (`302 → 14 → 1`, 4,257 params)
underperforms B0 here, unlike the historical 2-layer head. A fixed, non-swept depth
sensitivity (`R → 14 → 1` vs `R → 13 → 13 → 1` at matched budget, 5/5 folds) measures the
extra layer at **+0.01198 mean MAE** — the same order as the whole `ΔM` scale. So the
historical-head comparison is depth-confounded and is reported as robustness evidence
only. The **primary** mechanism test (E vs B2) is depth-matched *and* parameter-matched
(E and B2 share the exact architecture, `382 → 11 → 1`, 4,225 params; B1 differs by
+0.75%), so it is unaffected; and it is unambiguously non-positive. The confound cuts
*against* the negative, i.e. the conclusion is conservative.
