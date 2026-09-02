# T readout ablation and staged-search decision

Protocol: `mentor-t-readout-ablation-k64-s8-official-valid-v1`.

This study uses official train/validation only; official test was not evaluated.
The 624D typed-slot object remains a local proxy for the unknown mentor schema.

## Main attribution

K-SVD reduced mean graph reconstruction error from `0.4163` to `0.3212`, but
the reconstructed graph readout discarded task-relevant information:

| T view | dimension | fixed valid ROC-AUC | 8-trial Optuna valid ROC-AUC |
|---|---:|---:|---:|
| raw `mean(Y)` | 624 | 0.7014 | 0.6742 |
| reconstruction `mean(DX)` | 624 | 0.6687 | 0.6567 |
| residual `mean(abs(Y-DX))` | 624 | 0.7116 | 0.7068 |
| rich sparse-code summary | 640 | 0.6920 | 0.6741 |
| absolute code mean | 64 | 0.6732 | 0.6758 |

The result does not support the current `mean(DX)` construction as the
mentor's `recon_typed[624]`. Surface dimension is not effective information
dimension: with `K=64`, `mean(DX)=D mean(X)` lies in at most a 64D subspace.

## Final surviving fusion candidate

The 8-trial random inner-CV screen retained only `S+T_raw`. A 12-trial matched
comparison then gave:

| view | inner CV | official valid | valid std |
|---|---:|---:|---:|
| S | 0.7657 | **0.7848** | 0.0039 |
| S + T_raw | **0.7671** | 0.7689 | 0.0057 |

`S+T_raw-S = -0.0159` on official validation, with `0/5` seed wins. Thus the
small random-inner improvement did not transfer to the official scaffold
split. The present typed-slot T schema is closed; it should only be reopened
for a materially different patch/schema or the mentor's exact upstream code.

## Compute-budget rule for further exploration

New feature ideas use a three-stage gate:

1. **Screen**: frozen official-train features, three scaffold-aware inner folds,
   three search trials, no official validation and no multi-seed fit.
2. **Promote**: for a fused candidate, require inner delta at least `-0.005`
   versus S and wins on at least two of three folds. T-only views are diagnostic
   and do not receive full tuning merely because their standalone AUC is nonzero.
3. **Confirm**: only promoted mechanisms receive 12 Optuna trials and one
   official-validation multi-seed evaluation. Failed mechanisms are not rescued
   by wider K/T/Beam or classifier scans.

The scaffold screen is an elimination device, not a substitute for official
validation. In this study it reduced ten candidates to one, although that final
candidate still failed the official split.

Screening entry point:

```bash
uv run python -m tracks.ksvd.experiments.luyin16.tune_xgb_fused_proxy \
  --features <frozen_train_valid.npz> \
  --result <screen.json> \
  --trials 3 \
  --folds-file tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz \
  --screen-only \
  --baseline-view s \
  --promotion-delta=-0.005 \
  --minimum-fold-wins 2 \
  --views s,<candidate-fusions>
```

Primary outputs:

- `mentor_t_readout_ablation_k64_s8_official_valid/summary.json`
- `mentor_t_readout_ablation_k64_s8_official_valid/xgb_optuna_t_views.json`
- `mentor_t_readout_ablation_k64_s8_official_valid/xgb_optuna_t_raw_survivor_12.json`
- `mentor_t_readout_ablation_k64_s8_official_valid/xgb_scaffold_screen_t_raw.json`
