# Compact-v4 recurrent pair--centre — frozen cell A official-test closure (deterministic A100)

Date: 2026-09-14
Protocol: `compact_v4_recurrent_pair_centre_capacity_test_closure_v1`
Code commit: `4f1ccde`
Source experiment: `compact_v4_recurrent_pair_centre_capacity_decomposition_v1` (commit `481a7fa`)
Official ZINC test: loaded **exactly once**, only after the pre-registered freeze.

## Question

The deterministic-A100 2x2 capacity decomposition selected **cell A** as the
ZINC configuration (`NEITHER_AXIS`; best 2-seed fixed Top-5 soup valid MAE
`0.126368` at the smallest parameter count). Cell A is validation-optimal,
parameter-minimal and architecturally frozen. This workstream closes the
official test for that frozen cell **without any further architecture
selection**: no new architecture, no H112/H128, no optimizer sweep, no
test-driven tuning, and no re-selection of the Top-5 soup.

## A.1 Freeze record

Written by the `freeze` stage while the official test was still unloaded
(`test_loaded_at_freeze_time: false`, `test_status: "not yet loaded"`). The
freeze record and the seed0/seed1 checkpoint hashes are in
`results/compact_v4_recurrent_pair_centre_capacity_test_closure/architecture_freeze.json`.

```text
selected_architecture = cell A
h_dim = 64
q_dim = 16
patch_encoder_hidden = 64
global_encoder_hidden = 32
T = 2 (weight-tied refresh)
params = 85,763
selection_metric = 2-seed fixed Top-5 soup valid MAE
selection_value = 0.1263680279762484
seeds = [0, 1]
execution_regime = deterministic A100
validation soup per seed = 0.124704 / 0.128032
validation raw  per seed = 0.129710 / 0.131415
best epochs = 161 / 153
Top-5 epochs s0 = [161, 176, 180, 189, 186]
Top-5 epochs s1 = [153, 192, 187, 157, 169]
soup rule = K=5, lowest valid MAE, earliest-epoch ties, equal-weight parameter mean, no k/weight search
test not loaded at freeze time
```

## A.2 Official test (n = 1000)

### Raw (best-valid selection checkpoint -> test MAE)

| seed | test MAE |
|---:|---:|
| 0 | 0.110585 |
| 1 | 0.108137 |
| **mean ± std** | **0.109361 ± 0.001731** |

### Fixed Top-5 soup (unchanged k / checkpoint set / equal-weight average)

| seed | test MAE |
|---:|---:|
| 0 | 0.108624 |
| 1 | 0.104809 |
| **mean ± std** | **0.106717 ± 0.002698** |

### Diagnostic only (NOT a single model)

| estimator | test MAE |
|---|---:|
| 2-seed equal-weight **raw** prediction ensemble | 0.100670 |
| 2-seed equal-weight **soup** prediction ensemble | 0.098641 |

The 2-seed ensembles are diagnostics; the frozen single-model results are the
raw and soup rows above.

## A.3 Distance to `<0.10`

| estimator | distance to 0.10 |
|---|---:|
| raw single model | **+0.009361** |
| soup single model | **+0.006717** |
| diagnostic raw 2-seed ensemble | +0.000670 |
| diagnostic soup 2-seed ensemble | −0.001359 |

A single frozen 85,763-parameter cell-A model does **not** reach `<0.10`
(raw or soup). Only the diagnostic 2-seed prediction ensembles touch it, and
the soup ensemble crosses it by a small margin; neither is a single-model
result.

## A.4 Regime-aware comparison with historical CPU results

The historical H48/H96 numbers were produced on CPU under a different
execution regime; the comparison is context only, never a strict architecture
claim. Cell A is bit-identical in architecture to the historical H64
(85,763 params) but is *labelled* A here because it was frozen as the winner of
the 2x2 decomposition.

| model | regime | params | valid soup 2-seed | test raw mean | test soup mean |
|---|---|---:|---:|---:|---:|
| H48 | CPU (historical) | 82,115 | 0.134644 | 0.114951 | — |
| **cell A (== H64)** | **deterministic A100** | **85,763** | **0.126368** | **0.109361 ± 0.001731** | **0.106717 ± 0.002698** |
| H96 | CPU (historical) | 103,219 | 0.123241 | 0.107958 ± 0.000249 | 0.104990 ± 0.002373 |

| comparison | delta (cell A − historical) |
|---|---:|
| GPU cell A raw vs CPU H48 raw | **−0.005590** (cell A better) |
| GPU cell A raw vs CPU H96 raw | **+0.001403** (cell A worse, sub-noise) |
| GPU cell A soup vs CPU H96 soup | **+0.001727** (cell A worse, sub-noise) |

### Answers

1. **Did the GPU valid improvement transfer to test?** In the matched GPU
   regime there was no valid improvement to transfer: A→D moved the 2-seed soup
   by only `−0.001247` (H96 slightly worse), and the 2x2 main effects were both
   `|·| < 0.001`. On test, cell A (the H64-equivalent) lands between the
   historical CPU H48 and CPU H96 checkpoints and is statistically
   indistinguishable from CPU H96 (`+0.0014` raw, `+0.0017` soup, both below
   the ~0.002 GPU per-seed spread scale). The honest reading is a **regime
   shift**, not evidence that width-driven valid capacity transfers.
2. **Does cell A approach / reach `<0.10`?** No as a single model
   (raw +0.009361, soup +0.006717). Yes only as a 2-seed diagnostic ensemble
   (raw +0.000670, soup −0.001359).
3. **Most trustworthy ZINC single-model result under the deterministic GPU
   regime:** the frozen cell-A **fixed Top-5 soup = `0.106717`** (single
   85,763-parameter model); raw selection checkpoint = `0.109361`.

## Verdict

* Cell A is validated as the deterministic-GPU ZINC configuration and its
  official test is now closed at soup `0.106717` / raw `0.109361`.
* It does not reach `<0.10` as a single model. The 2-seed ensemble is a
  diagnostic, not a deployed model.
* The GPU regime closes the historical CPU H64→H96 gap at a smaller parameter
  budget, but the `<0.002` differences to the historical CPU H96 checkpoint are
  not interpreted as architecture effects.
* **No architecture change is made after opening test.** No H112/H128, no
  q-width, no reallocation, no optimizer/regularization rescue, no re-opening
  of the Top-5 rule.

Stop rule honoured.

## Files

- `experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_capacity_test_closure.py`
- `tests/ksvd/tests/test_compact_v4_recurrent_pair_centre_capacity_test_closure.py` (9 pass)
- `results/compact_v4_recurrent_pair_centre_capacity_test_closure/`:
  `parameter_accounting.json`, `sanity.json`, `architecture_freeze.json`,
  `official_test_unlock.json`, `official_test_results.json`, `report.json`
