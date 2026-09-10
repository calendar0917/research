# Frozen conditional readout sufficiency audit — output manifest

Representation/readout **sufficiency diagnostic** on the frozen compact-v4-hinge OOF
states (official TRAIN molecules only; official valid/test never loaded). It asks whether
a tiny permutation-invariant set readout of the pre-pooling states `{h'_i}`/`{q_ij}` adds
signed predictive value beyond a parameter-matched readout of the existing 302D graph
vector `R`. No backbone is trained; the main model is not modified.

Full scientific note: `tracks/ksvd/notes/frozen_conditional_readout_sufficiency_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_frozen_readout_sufficiency.py`
(stages `inventory → export → gate0 → splits → stage1 → all`).
Tests: `tracks/ksvd/tests/test_frozen_state_export_v2.py` (12 tests, all pass).

CSV/JSON/NPZ/PNG artifacts below are generated outputs (git-ignored by policy); this
README, the note, and the claim/decision YAML records are the tracked scientific record.

## Measurement repair (Stage 0)

Two historical export bugs are repaired and gated:

1. pair rows must be grouped by `batch.batch[batch.pair_index[0]]` (graph membership),
   not by `np.diff(batch.pair_index[0])` (source patch index);
2. the true pre-head representation is the **input** to `model.head[0]`, i.e.
   `R ∈ R^302 = unary(97) + pair-bucket moments(165) + global(32) + topology(8)`, not the
   64D output of `head[0]`.

## Files

| File | Contents |
|---|---|
| `checkpoint_inventory.json` | frozen OOF checkpoint inventory (fold × seed, state `.pt` SHA-256, params, oof_mae); 10/10 present, complete seeds [0,1] |
| `export_integrity_report.json` | Gate-0 checks G0.1–G0.7 on 5/5 folds; `all_passed: true` |
| `fold_split_manifest.json` | per-fold target-independent molecule-ID-hash split (1200 fit / 400 selection / 400 evaluation) + common-input bulk threshold |
| `stage_export.json` | corrected export summary + fingerprints |
| `stage1_fold_results.csv` | per-fold `B0 / B1(R-only) / S(set)` MAE, `Δ0`, `ΔR`, bulk safety, params, viability |
| `stage1_bootstrap.json` | stratified molecule-level paired bootstrap for `Δ0` and `ΔR` + aggregates |
| `stage1_decision.json` | Stage-1 verdict (pre-registered) |
| `final_decision.json` | final verdict + criteria + claim wording + scope caveat |
| `legacy_evidence_impact_record.json` | evidence-status update for historical audits touched by the two bugs |
| `state_exports/frozen_state_export_v2_corrected_fold{0-4}_seed0.npz` | per-molecule corrected states (`h'_i`, `q_ij`, buckets, true `R`, `yhat_0`) + fingerprint |
| `figures/figure1_per_fold_delta.png`, `figures/figure2_mae_comparison.png` | per-fold `ΔR`/`Δ0`; B0 vs B1 vs S (Figure 3 is B2-only and B2 was not run) |

## Key numbers (frozen OOF backbone seed 0, 5 folds)

- Gate 0: **all pass** — R reconstruction max abs `9.2e-5`, prediction reconstruction
  `1.9e-6`, batch invariance `4.8e-7`, node permutation `2.9e-6`.
- Adapter-evaluation MAE mean: **B0 0.19316 / B1 (R-only, 4135 params) 0.19050 /
  S (set, 4145 params) 0.19229**.
- `mean Δ0 = MAE(B0) − MAE(S) = +0.00087`; `mean ΔR = MAE(B1) − MAE(S) = −0.00179`
  (positive in 2/5 folds); molecule-level stratified 95% CI `[−0.00507, +0.00153]`.
- The set adapter is worse than the matched R-only head on the selection split in 4/5
  folds; 400 vs 2000 training epochs are bit-identical (not under-training).
- Decision: **NO-GO — cheap post-state readout extension**; the second frozen backbone
  seed was not spent (`records/decisions/decision-frozen-readout-sufficiency-nogo-20260910.yaml`).
  This does **not** prove moment pooling is sufficient and does **not** close the patch
  paradigm — it rules out a cheap low-capacity set-summary extension at this budget.
