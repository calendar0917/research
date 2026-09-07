# Compact-Hybrid-v2 training-horizon audit (60 -> 120 epoch ceiling)

Single-variable audit of whether the original 60-epoch ceiling truncated the
validation optimum of Compact-Hybrid-v2.  Only `model.epochs` changed (60 ->
120); patience-12 early stopping, seed 0, optimizer/LR rule (no scheduler),
and the whole model/representation are identical to Compact-v2.  Selection
only; test stays blocked.

- config: `tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_horizon120.yaml`
- run: `tracks/ksvd/runs/2026/09/07/20260907-201522-e0d1d0a3/` (screen, test_access=blocked)
- automated invariance gate: `tracks/ksvd/tests/test_zinc_patch_path_horizon_audit.py`
  (config leaf-diff: only `model.epochs` + protocol_id/output paths differ;
  model built from both configs has identical parameter counts/blocks/dims)

## Result

- STOP_REASON = EARLY_STOPPING at epoch 68 (best 56 + 12 stale epochs)
- best valid MAE = 0.184158 (best epoch 56) — bit-identical to the 60-epoch
  Compact-v2 run (epochs 1-60 valid MAE identical, max diff 0.0)
- epochs 57-68 after the optimum are all strictly worse (0.1853-0.3159);
  train L1 keeps decreasing (0.140691 at epoch 60) without validation
  improvement — the classical train-loss-is-not-evidence pattern
- params 98,549; budget PASS; vocab 6785, valid coverage 0.9879565047870728
  (identical to Compact-v2)

## Verdict

HORIZON_LIMITED = NO.  The 60-epoch ceiling did not truncate the validation
optimum: removing the ceiling did not move the best epoch and did not improve
best valid MAE; the unchanged patience-12 rule stopped training at epoch 68
by itself.  No follow-up (180/240 epochs, scheduler, seeds, test) was run.
