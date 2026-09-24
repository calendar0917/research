# FEC-S1 — Shared Local Environment Replacement (seed 0)

Round `fec_s1`; study `zinc-context-gap`; protocol `fec_s1_v1`.
Official ZINC **test was never loaded**.

## Primary result

* Top-5 soup official-valid MAE: **0.130422** (case **A**)
* best-checkpoint valid MAE: **0.136783** @ epoch 238
* soup members: [218, 219, 223, 238, 239]
* soup member MAEs: [0.136942670997174, 0.13735506682930282, 0.13938668785238406, 0.1367825070246472, 0.14001875323796412]
* epochs run: 240 (early_stopped=False)
* wall clock: 774.9 s
* peak GPU memory: 139.69775390625 MB

## Frozen verdict

```
FEC_S1_SHARED_REPLACEMENT_STRONG
```

`seed1_authorized = true`; seed 1 was **not** executed.

## Parameter accounting

* released lookup: typed 36420 + parent 256 = **36676**
* adapter hidden `H* = 214`, adapter params **36618** (rel. err 0.158 %)
* FEC-S1 total params **66170** vs S0 66228 (Δ -58)
* `typed_embedding is None` = True, `parent_embedding is None` = True

## Correctness gates

* G0_adapter_input: True
* G0_descriptor_identity: True
* G10_official_test_blocker: True
* G1_no_lookup_params: True
* G2_token_poisoning: True
* G2b_vocabulary_independence: True
* G3_adapter_gradient: True
* G4_G5_G6_G7_static_contract: True
* G5_adapter_environment_freeze: True
* G7_relabel_invariance: True
* G8_downstream_identity: True
* G9_parameter_fairness: True

## Baseline guard (read-only replay)

* replay valid MAE: 0.14567434728989612
* |Δ vs recorded|: 3.9488076974958375e-09

## Mechanism (report only)

* token invariance after training: {'device': 'cpu', 'invariant': True, 'invariant_within_rerun_noise': True, 'max_abs_pred_diff': 0.0, 'noise_floor_rerun_max_abs_diff': 0.0}
* adapter zeroing: {'frac_shift_gt_1e-6': 1.0, 'max_abs_prediction_shift': 1.7219340205192566, 'mean_abs_prediction_shift': 0.46315548503398896, 'shared_channel_inert': False, 'valid_mae_on': 0.13678250572824618, 'valid_mae_zeroed': 0.49851077648909997}
* effective rank: {'n': 23083, 'n_singular_values': 24, 'participation_ratio': 6.203909756927913, 'stable_rank': 3.20758454357272, 'top_singular_fraction': 0.3117610732985279}

## References

* historical S0 seed0 soup (provenance only): 0.140794 (members [125, 142, 159, 164, 167]; states not persisted)
* historical S0 seed0 selection checkpoint: 0.14567435123870381

run commit `341be8ffa6e1a7393340aa44ec7bbb911d7af99d`; official_test_loaded = False
