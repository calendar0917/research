# METHOD_CONTRACT — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Contract: two matched body arms + one shared Q, full-train confirmation

### Arm recipe (identical for SUM and COMP)

* Model: `build_arm_mj(payload, kappa_M)` with `ComponentReader` (39→2 half-split output head).
  Total parameters: **297,539**. The original 39→1 head (297,539 params) becomes 2×40=80 params,
  net +1 param (40 bias slots vs 1).
* Seed: 0. Adam(lr=1e-3, weight_decay=1e-5). Grad clip 5. Batch 128. FP32. No AMP/DDP.
* Epochs: 240. Steps per epoch: ceil(10000/128) = 79. Total: 18,960.
* Schedule: `build_schedule(10000, 240, 101)` — new 10k schedule, hash frozen.
* Soup: epochs {236, 237, 238, 239, 240}, parameter mean of the 5 members.

### Loss

* `L_g = MAE(hat_ell + hat_s, g)`
* `L_ell = MAE(hat_ell, ell)`
* `L_s = MAE(hat_s, s)`

| arm | loss |
|---|---|
| SUM | `L_g` |
| COMP | `L_g + 0.5 × (L_ell + L_s)` |

Coefficient `0.5` is fixed. SUM logs component losses detached but never uses them.

### Q head (shared deploy accessory, fixed recipe)

* Structure: `Linear(25, 64) → SiLU → Linear(64, 32) → SiLU → Linear(32, 1)`.
  **3,777 parameters**.
* Input: `topology25` after the frozen full-train prep.
* Target: `c = (k - mu_cycle) / sigma_cycle` (frozen train constants).
* Seed: 0. Adam(lr=1e-3, weight_decay=1e-5). Grad clip 5. Batch 128. FP32. 300 epochs (23,700 steps).
* Independent generator: `torch.Generator().manual_seed(20261003)`.
* Last-layer init: `weight=0`, `bias=median(c_train)`.
* Soup: epochs {296, 297, 298, 299, 300}.

### Calibration wrappers (frozen train-only)

```
b_g_a = median(g_train - h_a_train)           # g endpoint bias
b_y_a = median(y_train - h_a_train - q_raw_train)  # y endpoint bias
g_cal_a = h_a + b_g_a
y_cal_a = h_a + q_raw + b_y_a
```

One `b_g` and one `b_y` per arm. No stacking, no second Q bias.

### Data provenance

* Train: 10,000 positional official-train rows (canonical order). GVAE properties from
  `zinc_long_cycle_audit/cache/gvae_full_properties.npz`. `y`/`gid` from
  `zinc_dictionary_real_data_handoff/train.npz`.
* Valid: `encoded_valid.pt` + `env_valid.pt` + valid cycle audit CSV with `smi_line` indices.
* official-test: never loaded, predicted, or scored.

### Standardizer / kappa refit (full-train)

* Body feature standardizers: refit on all 10,000 train rows (invert all-train cache, refit).
* `phi` scaler + `kappa_M`: refit on all 10,000 realized train roots (KAPPA_SEED=20261004,
  sample cap 8192).
* Target decomposition constants: refit on all 10,000 rows.
* `MU_LOGP = 2.4570953396190123` fixed.

### Evaluation protocol

* Valid is loaded exactly once, after the frozen_eval_manifest is written.
* Valid diagnostics (`y, g, c, k, ell, s`) computed with frozen train constants.
* Valid `q_raw` from the frozen Q soup on valid `topology25`.
* Bootstrap: 1000 draws, seed 20261009, shared indices within each endpoint, G0/overall.
* Gates: see PROTOCOL §8.
