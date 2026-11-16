# LCRC explicit-context readout — small-dataset feasibility pilot

**Round** `lcrc-readout-context-pilot-small-v1` · **status** DEVELOPMENT · every Test number below is
`DEVELOPMENT_TEST` · **branch** `lcrc-readout-context-pilot-small-v1` on top of design basis
`9b3dba098915` · **formal runs executed at** `ee72310d9290` ·
**freeze hash** `5a6d48aaef012aac…` · **generated** 2026-11-16T23:13:08

> This is a feasibility pilot on two small tasks with **one** training seed.
> The Test split was evaluated every epoch by explicit user authorisation
> (`test_policy=always`). Test never selected a checkpoint, a configuration or a
> number that appears in a verdict. No result here is a benchmark submission, a
> significance claim, or evidence about a second seed.

## 1. Identity of what was trained

| task | Train/Val/Test | A readout | B readout | C readout | M active | data audit |
|---|---|---|---|---|---|---|
| ogbg-molesol | 902/113/113 | 15298 | 15296 (H=207) | 15330 (H=200) | 34 | yes |
| ogbg-mollipo | 3360/420/420 | 15298 | 15326 (H=206) | 15290 (H=198) | 36 | yes |

All three arms share **one** local trunk (bitwise the C1 local path) and differ only in the
readout. `M` is the number of raw global columns whose Train span exceeds 1e-8; it is a
property of the frozen cache, not a choice. `H` is fixed so the readout parameter count lands
within one hidden-width step of the 15298-parameter C1 head.

Frozen caches (SHA-256 of the real bytes, re-verified on the compute host):

* `ogbg_molesol_sq-pilot-cache-v1.npz` — `3b7a8e16ca9c7ab2…`
* `ogbg_molesol_sq-pilot-lipo-raw-v1.npz` — `a3ab9f7e22798a8f…`
* `ogbg_mollipo_sq-pilot-cache-v1.npz` — `fe002287fa9c5812…`
* `ogbg_mollipo_sq-pilot-lipo-raw-v1.npz` — `3005aa55e766d83a…`

Shared-trunk initialisation (identical in every arm, reproduced in a fresh process):

* ogbg-molesol: trunk `1d075977ddb49ffe…`
* ogbg-mollipo: trunk `30262df9366f83c3…`

Frozen recipe (one for all arms): seed 0, 360 epochs, AdamW peak 0.0003 -> min 1e-05, wd 0.001, batch 128, test_policy `always`, selection = minimum Val RMSE only.

## 2. Main table (raw target units)

| task | arm | origin | epoch | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | last20 Val | last20 Test | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | A | THIS_ROUND | 248 | 0.3489 | 0.8753 | 0.8545 | 0.6595 | 0.6695 | 0.9149 | 0.8890 | 228674 |
| ogbg-molesol | B | THIS_ROUND | 167 | 0.3966 | 0.9184 | 1.3853 | 0.7042 | 0.8721 | 0.9529 | 1.4546 | 228672 |
| ogbg-molesol | C | THIS_ROUND | 127 | 0.4876 | 0.9980 | 1.1012 | 0.7471 | 0.8094 | 1.0310 | 1.1440 | 228706 |
| ogbg-molesol | G | REUSED_FROZEN_REFERENCE | 179 | 0.3299 | 0.9353 | 0.9354 | 0.7095 | 0.7288 | 0.9936 | 0.9473 | 377094 |
| ogbg-mollipo | A | THIS_ROUND | 275 | 0.2639 | 0.7086 | 0.7254 | 0.5123 | 0.5292 | 0.7137 | 0.7401 | 228674 |
| ogbg-mollipo | B | THIS_ROUND | 116 | 0.4052 | 0.7589 | 0.7290 | 0.5402 | 0.5466 | 0.8087 | 0.7491 | 228702 |
| ogbg-mollipo | C | THIS_ROUND | 133 | 0.3912 | 0.6967 | 0.7404 | 0.5136 | 0.5539 | 0.7397 | 0.7644 | 228666 |
| ogbg-mollipo | G | REUSED_FROZEN_REFERENCE | 168 | 0.2691 | 0.6091 | 0.6456 | 0.4624 | 0.4847 | 0.6337 | 0.6690 | 377350 |

* `epoch` is the Val-selected epoch (minimum Val RMSE); the Test column is that same frozen
  state and is `DEVELOPMENT_TEST`.
* `last20` is the arithmetic mean of the last 20 epochs of the same trajectory — a curve
  summary, not an ensemble and not a second model.
* `G` is the frozen GINE of `sq-pilot-small-v1`, reused unchanged (`REUSED_FROZEN_REFERENCE`, no
  retraining here); it has a different input path and a stronger learning-rate preset
  (`1e-3` vs the frozen `3e-4` of this round), so it is a performance coordinate only.

### 2.1 Val -> Test transfer gap of the same frozen state

The selected state is a Val-RMSE minimum, so this table asks how much of that Val number
survives on Test. With 113 Val molecules (molesol) a small Val improvement is noisy.

| task | arm | Val RMSE | Test RMSE | Test - Val | Test / Val | last20 Test - Val | last20 Test / Val |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | A | 0.8753 | 0.8545 | -0.0208 | 0.976 | -0.0259 | 0.972 |
| ogbg-molesol | B | 0.9184 | 1.3853 | 0.4669 | 1.508 | 0.5016 | 1.526 |
| ogbg-molesol | C | 0.9980 | 1.1012 | 0.1032 | 1.103 | 0.1130 | 1.110 |
| ogbg-molesol | G | 0.9353 | 0.9354 | 0.0001 | 1.000 | -0.0463 | 0.953 |
| ogbg-mollipo | A | 0.7086 | 0.7254 | 0.0168 | 1.024 | 0.0264 | 1.037 |
| ogbg-mollipo | B | 0.7589 | 0.7290 | -0.0298 | 0.961 | -0.0597 | 0.926 |
| ogbg-mollipo | C | 0.6967 | 0.7404 | 0.0437 | 1.063 | 0.0247 | 1.033 |
| ogbg-mollipo | G | 0.6091 | 0.6456 | 0.0365 | 1.060 | 0.0353 | 1.056 |

The largest selected-state gap among A/B/C is ogbg-molesol/B (Test = 1.51x Val). A gap of this size means the Val-selected epoch of that arm is not a reliable performance coordinate on this task; it is reported rather than smoothed away, and no cross-split average is used anywhere in this report.


## 3. Paired contrasts (same frozen states)

| task | contrast | Δ Val RMSE | ratio Val | 95% CI (Val) | Δ Test RMSE | ratio Test | 95% CI (Test) | Δ last20 Val | Δ last20 Test |
|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | B-A | 0.0431 | 1.049 | [-0.054, 0.151] | 0.5308 | 1.621 | [0.073, 1.096] | 0.0380 | 0.5655 |
| ogbg-molesol | C-A | 0.1227 | 1.140 | [0.022, 0.243] | 0.2467 | 1.289 | [0.044, 0.480] | 0.1161 | 0.2550 |
| ogbg-molesol | C-B | 0.0797 | 1.087 | [-0.057, 0.242] | -0.2840 | 0.795 | [-0.647, 0.068] | 0.0780 | -0.3105 |
| ogbg-molesol | A-G | -0.0600 | 0.936 | [-0.160, 0.042] | -0.0809 | 0.913 | [-0.206, 0.036] | -0.0787 | -0.0583 |
| ogbg-molesol | C-G | 0.0627 | 1.067 | [-0.083, 0.205] | 0.1658 | 1.177 | [-0.060, 0.421] | 0.0374 | 0.1967 |
| ogbg-mollipo | B-A | 0.0503 | 1.071 | [-0.014, 0.112] | 0.0037 | 1.005 | [-0.052, 0.062] | 0.0950 | 0.0089 |
| ogbg-mollipo | C-A | -0.0119 | 0.983 | [-0.055, 0.034] | 0.0150 | 1.021 | [-0.034, 0.064] | 0.0260 | 0.0243 |
| ogbg-mollipo | C-B | -0.0622 | 0.918 | [-0.141, 0.005] | 0.0113 | 1.016 | [-0.024, 0.043] | -0.0690 | 0.0154 |
| ogbg-mollipo | A-G | 0.0996 | 1.163 | [0.021, 0.181] | 0.0798 | 1.124 | [0.032, 0.129] | 0.0800 | 0.0711 |
| ogbg-mollipo | C-G | 0.0876 | 1.144 | [0.025, 0.150] | 0.0948 | 1.147 | [0.043, 0.144] | 0.1060 | 0.0955 |

Δ = left − right (negative favours the left arm). Ratios are left / right. The intervals are
paired molecule bootstrap percentiles (2000 replicates, RNG 1703, one resample shared by every
model of a split). They describe uncertainty **conditional on the single-seed selected state**:
they do not measure training-seed variance, they do not repair Val selection, and they do not
undo the development exposure of Test.

Leave-one-row-out sensitivity of those deltas (does a single molecule decide the sign?):

| task | split | contrast | Δ RMSE (full) | Δ range over dropping one row | #rows that flip the sign | largest single-row shift |
|---|---|---|---|---|---|---|
| ogbg-molesol | valid | B-A | 0.0431 | [0.0207, 0.0606] | 0 | 0.0224 |
| ogbg-molesol | valid | C-A | 0.1227 | [0.0718, 0.1365] | 0 | 0.0509 |
| ogbg-molesol | valid | C-B | 0.0797 | [0.0175, 0.0992] | 0 | 0.0622 |
| ogbg-molesol | test | B-A | 0.5308 | [0.1930, 0.5565] | 0 | 0.3377 |
| ogbg-molesol | test | C-A | 0.2467 | [0.1397, 0.2662] | 0 | 0.1070 |
| ogbg-molesol | test | C-B | -0.2840 | [-0.3298, -0.0533] | 0 | 0.2307 |
| ogbg-mollipo | valid | B-A | 0.0503 | [0.0184, 0.0591] | 0 | 0.0319 |
| ogbg-mollipo | valid | C-A | -0.0119 | [-0.0195, 0.0003] | 1 | 0.0122 |
| ogbg-mollipo | valid | C-B | -0.0622 | [-0.0663, -0.0210] | 0 | 0.0412 |
| ogbg-mollipo | test | B-A | 0.0037 | [-0.0056, 0.0161] | 7 | 0.0125 |
| ogbg-mollipo | test | C-A | 0.0150 | [0.0051, 0.0272] | 0 | 0.0122 |
| ogbg-mollipo | test | C-B | 0.0113 | [0.0066, 0.0157] | 0 | 0.0047 |

_Dropped numbers are diagnostics only; the official score is always the full split._


## 4. Preflight gates (before any formal epoch)

| task | #gates | all pass | #failed |
|---|---|---|---|
| ogbg-molesol | 13 | yes | 0 |
| ogbg-mollipo | 13 | yes | 0 |

Gates executed: `armA_equivalence`, `checkpoint_reload`, `data`, `dropout_shared`, `gine_reference_reuse`, `gradients`, `invariance`, `ledger_closure`, `parameters`, `roundtrip_and_zero_readout`, `scale_equivalence`, `shared_init`, `single_variable_functions`.

| gate | ogbg-molesol | ogbg-mollipo |
|---|---|---|
| armA_equivalence | pass | pass |
| checkpoint_reload | pass | pass |
| data | pass | pass |
| dropout_shared | pass | pass |
| gine_reference_reuse | pass | pass |
| gradients | pass | pass |
| invariance | pass | pass |
| ledger_closure | pass | pass |
| parameters | pass | pass |
| roundtrip_and_zero_readout | pass | pass |
| scale_equivalence | pass | pass |
| shared_init | pass | pass |
| single_variable_functions | pass | pass |


## 5. Trace: can every number be taken apart again?

The readout is only worth its explicitness if the exported terms reproduce the prediction.
Tolerance in every row is the pre-declared `tol(x) = 1e-5 + 1e-6·max(1,|x|)` in raw target units.

### 5.1 Prediction reconstruction from the exported components

| model | split | rows | max abs error | tolerance | verdict |
|---|---|---|---|---|---|
| ogbg-molesol/B | valid | 113 | 2.69e-07 | 2.03e-05 | pass |
| ogbg-molesol/B | test | 113 | 4.70e-07 | 1.91e-05 | pass |
| ogbg-molesol/C | valid | 113 | 5.48e-07 | 1.97e-05 | pass |
| ogbg-molesol/C | test | 113 | 4.86e-07 | 1.93e-05 | pass |
| ogbg-mollipo/B | valid | 420 | 1.97e-07 | 1.72e-05 | pass |
| ogbg-mollipo/B | test | 420 | 2.43e-07 | 1.56e-05 | pass |
| ogbg-mollipo/C | valid | 420 | 2.71e-07 | 1.51e-05 | pass |
| ogbg-mollipo/C | test | 420 | 2.02e-07 | 1.51e-05 | pass |

The account is `pred_raw = mu + sigma·(intercept + local terms + global terms + interactions)`
with `mu, sigma` the frozen Train target mean/std; only the intercept carries `mu`.

### 5.2 Reloading the selected checkpoint

| task | arm | max abs Δ Val | max abs Δ Test | state hash | verdict |
|---|---|---|---|---|---|
| ogbg-molesol | A | 0.000002 | 0.000001 | ac89f8ba9c0a | pass |
| ogbg-molesol | B | 0.000002 | 0.000002 | d182cb7ca5da | pass |
| ogbg-molesol | C | 0.000003 | 0.000002 | 10b21544b72f | pass |
| ogbg-mollipo | A | 0.000002 | 0.000002 | 3b72ef730698 | pass |
| ogbg-mollipo | B | 0.000004 | 0.000001 | 8f0f56a3f8b5 | pass |
| ogbg-mollipo | C | 0.000003 | 0.000002 | 7bf8802725c0 | pass |

### 5.3 Fixed-input intervention

One typed-S2 occurrence is removed from a Val molecule (raw graph, S1 codes, scalars and the
global vector unchanged), and the *change* in the prediction is rebuilt from the change in the
ledger row that owns that occurrence:

| task | molecules | removed | #failures | verdict |
|---|---|---|---|---|
| ogbg-molesol | 32 | the molecule's first typed-S2 record (cache path index of its atom 0) | 0 | pass |
| ogbg-mollipo | 32 | the molecule's first typed-S2 record (cache path index of its atom 0) | 0 | pass |

In every case the other roots' base terms and every global term are unchanged to `<= 1e-6`, and
in arm C the context factor `a_k` is unchanged (the context vector did not move). The original
C1 head also responds — through `rho` (mean h / mean h² / n) — which is recorded, not scored as
an implementation error: it is exactly the entanglement this round is about.

### 5.4 Context response with the local state held fixed

For each field the frozen intervention moves the context from its **observed** Train-standardised
value to the Train Q75 and keeps `h` fixed; the closed form
`Δ = Δu_k + Δa_k · S_k` is then compared to an independent full forward pass.

| task | context field | RMS Δu (main effect) | RMS Δ interaction | #failures | verdict |
|---|---|---|---|---|---|
| ogbg-molesol | log1p_n_atoms | 0.0834 | 0.4709 | 0 | pass |
| ogbg-molesol | ring_atom_fraction | 0.1645 | 0.0688 | 0 | pass |
| ogbg-mollipo | log1p_n_atoms | 0.2636 | 0.2111 | 0 | pass |
| ogbg-mollipo | ring_atom_fraction | 0.1919 | 0.1838 | 0 | pass |

`Δu` is the *single-variable global* response and `Δinteraction` is the local-state-weighted
response. They are separate columns on purpose: merging them would hide the difference between
'the number went in' and 'the number changed how the structure is read'.

### 5.5 Scale equivalence

`a_k ×2, c_k ÷2` leaves every prediction and every per-root interaction unchanged:

| task | Val rows | max abs Δ pred | a ratio | verdict |
|---|---|---|---|---|
| ogbg-molesol | 113 | 0.00 | 2.0000–2.0000 | pass |
| ogbg-mollipo | 420 | 0.00 | 2.0000–2.0000 | pass |

So an individual `a_k` or `c_k` norm is **not** an importance measure. This closes the scale
degeneracy only; it does not resolve every attribution non-uniqueness.

### 5.6 Verdict

| task | label |
|---|---|
| ogbg-molesol | TRACE_PASS |
| ogbg-mollipo | TRACE_PASS |

`TRACE_PASS` means the bookkeeping closes. It is **not** proof that a per-root number is a
chemical fragment (root windows overlap), nor that the attribution is unique.


## 6. Is the interaction real, or a constant?

Reference decomposition of the interaction term at molecule level,
`I_k = a_k · [n·c̄_k + Σ_v (c_k(v) − c̄_k)]`, with `c̄_k` the node-equal-weight mean over Train
roots of the selected model. This is a reference decomposition, not a refit; RMS parts can exceed
the whole through cancellation, so they are not exclusive error shares.

| task | split | field | RMS interaction (raw) | constant part | root-deviation part | median root std/RMS | #1-root molecules |
|---|---|---|---|---|---|---|---|
| ogbg-molesol | valid | log1p_n_atoms | 2.1770 | 2.5750 | 1.2043 | 0.660 | 0 |
| ogbg-molesol | valid | ring_atom_fraction | 0.2456 | 0.0634 | 0.2044 | 0.869 | 0 |
| ogbg-molesol | test | log1p_n_atoms | 2.7540 | 2.8402 | 1.1726 | 0.636 | 0 |
| ogbg-molesol | test | ring_atom_fraction | 0.2793 | 0.0660 | 0.2401 | 0.897 | 0 |
| ogbg-mollipo | valid | log1p_n_atoms | 0.2290 | 0.0206 | 0.2352 | 0.990 | 0 |
| ogbg-mollipo | valid | ring_atom_fraction | 0.2768 | 0.2058 | 0.1308 | 0.932 | 0 |
| ogbg-mollipo | test | log1p_n_atoms | 0.1895 | 0.0168 | 0.1873 | 0.987 | 0 |
| ogbg-mollipo | test | ring_atom_fraction | 0.3075 | 0.2164 | 0.1528 | 0.918 | 0 |

Pre-declared read-out: an interaction counts as *resolved* only when its RMS is at least 1% of
the Train target standard deviation on Val, Test **and** the Q25→Q75 intervention; and it is
*root-specific* only when the median per-molecule root std/RMS (root-specific spread, not a
coefficient of variation) is above 0.05 — below that the interaction is nearly a constant multiplier.

| task | field | RMS (Val) | RMS (intervention) | median root std/RMS | response | root specificity |
|---|---|---|---|---|---|---|
| ogbg-molesol | log1p_n_atoms | 1.0534 | 0.4709 | 0.660 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-molesol | ring_atom_fraction | 0.1188 | 0.0688 | 0.869 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-mollipo | log1p_n_atoms | 0.1890 | 0.2111 | 0.990 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |
| ogbg-mollipo | ring_atom_fraction | 0.2285 | 0.1838 | 0.932 | RESOLVED_FUNCTIONAL_RESPONSE | NONUNIFORM_ROOT_RESPONSE |


## 7. The four pre-declared axes

The axes are reported separately and never collapsed into one score.

| task | axis | label |
|---|---|---|
| ogbg-molesol | retention | PERFORMANCE_LOSS_THIS_RECIPE |
| ogbg-molesol | context | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-molesol | gine | WITHIN_5PCT_OF_THIS_GINE (A) |
| ogbg-molesol | trace | TRACE_PASS |
| ogbg-mollipo | retention | PRESERVED_POINT_ESTIMATE / HORIZON_CONSISTENT |
| ogbg-mollipo | context | SPLIT_OR_SELECTION_SENSITIVE |
| ogbg-mollipo | gine | BEHIND_THIS_GINE |
| ogbg-mollipo | trace | TRACE_PASS |

### ogbg-molesol

* **Retention (C vs A)**, tolerance 1.05×: Val ratio 1.140, Test ratio 1.289; last-20 ratios 1.127 / 1.287.
* **Context increment (C vs B)**, threshold +2.00%:

| surface | relative gain C over B |
|---|---|
| last20_test | +21.35% |
| last20_valid | -8.19% |
| selected_test | +20.50% |
| selected_valid | -8.67% |

  above threshold on: selected_test, last20_test; below zero on: selected_valid, last20_valid; sign disagrees between Val and Test: yes.

* **GINE competitiveness (each arm vs the reused G)**:

| arm | Val RMSE | Test RMSE | Val / G | Test / G | within 5% both |
|---|---|---|---|---|---|
| A | 0.8753 | 0.8545 | 0.936 | 0.913 | yes |
| B | 0.9184 | 1.3853 | 0.982 | 1.481 | no |
| C | 0.9980 | 1.1012 | 1.067 | 1.177 | no |

  G is a strong performance coordinate with a different input path and a stronger learning-rate preset (1e-3 vs the frozen 3e-4); no gap is attributable to a single operator and no SOTA claim is made in either direction

### ogbg-mollipo

* **Retention (C vs A)**, tolerance 1.05×: Val ratio 0.983, Test ratio 1.021; last-20 ratios 1.036 / 1.033.
* **Context increment (C vs B)**, threshold +2.00%:

| surface | relative gain C over B |
|---|---|
| last20_test | -2.06% |
| last20_valid | +8.53% |
| selected_test | -1.56% |
| selected_valid | +8.20% |

  above threshold on: selected_valid, last20_valid; below zero on: selected_test, last20_test; sign disagrees between Val and Test: yes.

* **GINE competitiveness (each arm vs the reused G)**:

| arm | Val RMSE | Test RMSE | Val / G | Test / G | within 5% both |
|---|---|---|---|---|---|
| A | 0.7086 | 0.7254 | 1.163 | 1.124 | no |
| B | 0.7589 | 0.7290 | 1.246 | 1.129 | no |
| C | 0.6967 | 0.7404 | 1.144 | 1.147 | no |

  G is a strong performance coordinate with a different input path and a stronger learning-rate preset (1e-3 vs the frozen 3e-4); no gap is attributable to a single operator and no SOTA claim is made in either direction


## 8. Does arm A reproduce the frozen C1 trajectory?

Arm A is the unchanged C1 head, so its trajectory must match the frozen `L` run of
`sq-pilot-small-v1` up to the one difference the prompt froze: the batch order (`rng(0)`
permutation here, `shuffle_offset=917000` there).

| task | same peak lr | same epochs | L epoch | A epoch | L Val | A Val | Δ Val | Δ Test | Δ last20 Val |
|---|---|---|---|---|---|---|---|---|---|
| ogbg-molesol | True | True | 248 | 248 | 0.8686 | 0.8753 | +0.77% | -0.30% | +0.84% |
| ogbg-mollipo | True | True | 275 | 275 | 0.7052 | 0.7086 | +0.49% | +1.61% | +0.29% |

Agreement at the same epoch with a ~1% gap is the expected signature of an identical recipe with a
different shuffle convention; it is not a new experiment and is not evidence about seeds.


## 9. What this round does not show

* **One seed.** Every contrast is a single trajectory per arm. Nothing here estimates run-to-run
  variance; the paired bootstrap describes molecules *given* the frozen states, not training noise.
* **Val selection is optimistic in both arms.** Each arm's reported number is its best epoch on 113
  (molesol) / 420 (mollipo) Val molecules. The comparison A/B/C is fair among themselves (same
  selection rule, same split) but no absolute number is an unbiased estimate of generalisation.
* **Test is `DEVELOPMENT_TEST`.** It was looked at every epoch and is reported next to the selected state.
  Any reader who used it to choose anything would invalidate it; this round did not.
* **G is a coordinate, not a control.** The frozen GINE has a different input path and a stronger
  learning-rate preset; a gap is not attributable to a single operator.
* **No chemical claim.** A root row is a *root-local input environment*; root windows overlap and are
  not independent fragments, and the last-word attribution on internal paths is not unique.
* **No architecture search.** The readout forms were fixed in code before the runs; a third context
  field, a wider `a_k`, or a different `H` would be a different round.
* **Two small tasks.** No ZINC, no PCBA, no public-122, no MoleculeNet benchmark claim.

## 10. Reproducing this report

```bash
# local identity + gates (CPU, ~1 min)
uv run pytest -q tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/tests
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
  --purpose "prepare" --set stage=prepare

# formal training: one worker per GPU, one dataset each
rr deploy res
rr run res lcrc-rc-molesol --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/analysis \
  -- bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
# ... same with --pool res-gpu1 --set dataset=ogbg-mollipo
rr pull lcrc-rc-molesol && rr pull lcrc-rc-mollipo

# CPU-only analysis: tables, ledgers, interventions, figures, verdicts, this report
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
  --purpose "aggregate" --set stage=aggregate
```

Provenance: round commit `8b6269877609` on `lcrc-readout-context-pilot-small-v1`,
design basis `9b3dba098915`, execution regime recorded per run by
`rr` (host `res`, A100-SXM4-40GB, torch 2.5.1+cu124), analysis on CPU.

## 11. Execution provenance

Read from the control-plane run records (`rr status --json`), not typed by hand:

| experiment | pool | node | driver | torch | commit | state | exit | started | completed |
|---|---|---|---|---|---|---|---|---|---|
| lcrc-rc-molesol | res-gpu0 | a100-2 | 550.163.01 | 2.5.1+cu124 | ee72310d9290 | completed | 0 | 2026-11-16T22:31:39+08:00 | 2026-11-16T22:39:35+08:00 |
| lcrc-rc-mollipo | res-gpu1 | a100-2 | 550.163.01 | 2.5.1+cu124 | ee72310d9290 | completed | 0 | 2026-11-16T22:31:51+08:00 | 2026-11-16T23:03:29+08:00 |

The analysis stage is CPU-only and reads only the pulled artefacts; it never touches the GPU host.

### 11.1 Compute actually spent

| experiment | job wall (s) | job wall (A100-h) | GPU |
|---|---|---|---|
| lcrc-rc-molesol | 476 | 0.132 | res-gpu0 |
| lcrc-rc-mollipo | 1898 | 0.527 | res-gpu1 |

Total formal compute: **0.659 A100-h** of job wall across the two workers (one GPU each, one training process per GPU, no fan-out), plus one short GPU smoke run and CPU-only prepare/aggregate stages.

Per-arm trainer-measured time is in each `results/<task>/<arm>/trajectory.json` (`train_seconds`, `eval_seconds_total`).

