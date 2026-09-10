# Pair endpoint association witness audit — output manifest

**Frozen-feature witness diagnostic** on the frozen compact-v4-hinge OOF states
(official TRAIN molecules only; official valid/test never loaded). It asks whether the
cross-coordinate endpoint association that the current symmetric pair map provably
discards — `a_ij = offdiag_{k<l}(u_i u_j^T + u_j u_i^T) ∈ R^120` — carries stable,
task-relevant signed predictive value beyond the existing 302D graph vector `R`, and
beyond a matched diagonal/current-information control. No backbone is trained, no pair
encoder / `q_ij` / feature is modified, no new pair module is trained.

Full scientific note: `tracks/ksvd/notes/pair_endpoint_association_witness_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_pair_endpoint_association_witness.py`
(stages `inventory → spec → export → gate0 → splits → ambiguity → stage1 → stage1b →
stage2 → figures → decision → all`).
Tests: `tracks/ksvd/tests/test_pair_endpoint_association_witness.py` (10 tests, all pass).

CSV/JSON/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## v3 state export extension

New cache version `frozen_state_export_v3_pair_endpoint`, built from the *same* frozen
OOF checkpoints (nothing retrained). It extends the already-corrected v2 export with:

- `projected_patch_state` — `u_i = P(h_i) ∈ R^16`, the **pre-centre-update** patch
  state (the exact tensor that enters pair feature construction);
- `pair_relation` — the raw 23D relation descriptor per pair.

Gate 0 additionally checks that the exported `u_i` == the forward pair projection and
that the reconstructed pair encoder input == the true forward pair input (both max 0.0).

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, state `.pt` SHA-256, params, oof_mae); seeds [0,1] complete |
| `pair_feature_spec.json` | real pair computation graph constants + relation layout + feature spec |
| `export_integrity_report.json` | Gate-0 checks G0.1–G0.8 on 5/5 folds; `all_passed: true` |
| `fold_split_manifest.json` | per-fold target-independent molecule-ID-hash split (1200 fit / 400 selection / 400 evaluation) + common-input bulk threshold |
| `stage_export.json` | v3 export summary + fingerprints (incl. random-projection fingerprint) |
| `representation_ambiguity.json` | target-free `c`-vs-`a` ambiguity (same-bucket and relation-restricted); descriptive only |
| `stage1_fold_results.csv` | per-fold `B0 / B1(R-only) / B2(diag) / E(endpoint)` MAE, `ΔR`, `Δdiag`, bulk safety, params, viability |
| `stage1_bootstrap.json` | stratified molecule-level paired bootstrap for `ΔR` and `Δdiag` + aggregates |
| `final_decision.json` | final verdict + criteria + claim wording + scope caveat |
| `state_exports/frozen_state_export_v3_pair_endpoint_fold{0-4}_seed0.npz` | per-molecule v3 states (`h'_i`, `u_i`, `q_ij`, buckets, raw relation, true `R`, `yhat_0`) + fingerprint |
| `figures/figure1_representation_ambiguity.png`, `figures/figure2_per_fold_delta.png` | `c`-vs-`a` ambiguity; per-fold `ΔR`/`Δdiag` (Figure 3 is Stage-2-only and Stage 2 was not run) |

## Key numbers (frozen OOF backbone seed 0, 5 folds, 1 adapter init)

- Gate 0: **all pass** — `u_i` == forward pair projection `0.0`; reconstructed pair
  input == forward pair input `0.0`; reconstructed `q_ij` `0.0`; R reconstruction
  `1.14e-5`; prediction reconstruction `9.5e-7`; batch invariance `0.0` / `4.8e-7`;
  node-permutation witness `1.24e-7`.
- Representation ambiguity (target-free): same-bucket c-nearest `a`-distance is
  **0.317×** random (relation-restricted `0.495×`); `Pearson(‖Δc‖,‖Δa‖) = 0.927`;
  close-c/far-a fraction `0.0`; normalized conditional `Var(a | 16-NN c) = 0.252`.
- Adapter-evaluation MAE mean: **B0 0.19316 / B1 (R-only, 4135 params) 0.19050 /
  B2 (diagonal control, 4177 params) 0.19475 / E (endpoint witness, 4177 params) 0.19749**.
- `mean ΔR = MAE(B1) − MAE(E) = −0.006994` (95% CI `[−0.01039, −0.00359]`, `P(>0)=0.0001`,
  **0/5** folds positive); `mean Δdiag = MAE(B2) − MAE(E) = −0.002741` (**0/5** non-negative).
- E is worse than B1 on the selection split in 5/5 folds (optimization/generalization,
  not eval noise); common-input bulk degradation mean `+0.0050`, max `+0.0112`
  (> 0.002 gate ⇒ FAIL); endpoint adapter non-collapsed (sensitivity min `0.338`).
- Decision: **NO-GO — off-diagonal endpoint cross-coordinate association (Case C)**; the
  second frozen backbone seed and the Stage-1b init were not spent
  (`records/decisions/decision-pair-endpoint-association-witness-nogo-20260911.yaml`).
  Do **not** learn the projection, sweep rank, feed the full 120D outer product, or open
  attention / triplet networks / a low-rank symmetric bilinear pair encoder. This does
  **not** prove the pair encoder is globally wrong nor that a full outer product is
  optimal.
