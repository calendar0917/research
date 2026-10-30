# PROTOCOL — LCRC-A Lipophilicity screen v1

- **Protocol id:** `lcrc-lipo-screen-v1`
- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Config:** `configs/lcrc_lipo_screen_v1.yaml` (sha256 see `analysis/protocol_freeze.json`)
- **Config hash:** `b724a521438c31c356f36157ba7a52bbf408da327f47ac2243f3b09c146b6533`
- **Protocol freeze record:** `analysis/protocol_freeze.json`, sha256 `87ec475107121d7b48cf9ae1e2664bb27bba972c39a543ebf24ea7885dbed9f4`
- **Status at write time:** `FROZEN_PRE_GPU` / round status `READY_FOR_GPU_AUTHORIZATION`
- **GPU authorization:** `NOT_AUTHORIZED`. No GPU process was started in this round. `analysis/gpu_authorization.json` does not exist.
- **Implementation commit at freeze:** `c3202fee7d9a` (the freeze artifacts themselves are committed in the following freeze commit).

This file is the pre-registered design. It is frozen before any formal model sees a
Monitor label. The config and every code/test file are hashed into
`analysis/protocol_freeze.json`; `run_screen.py train` refuses to run if the hashes
drift or if an authorization file is missing/invalid.

---

## 1. Question and scope

Three application-level questions on the same Fit/Monitor split:

1. Does scratch LCRC-A learn the Lipophilicity regression task at all?
2. Does the explicit two-hop S2 channel add a stable increment over an otherwise
   identical S2-off ablation (arm B)?
3. How does LCRC-A compare with a standard GINE (arm C) and a frozen ECFP4+XGBoost
   baseline (arm D) on the same data and metric?

This round is not a SOTA attempt, not an architecture search, and not an SSL study.
No new model mechanism is introduced. The only new code is data/training/evaluation
plumbing around the already-audited `lipo_v1` LCRC-A.

**Not answerable in this pre-GPU state:** questions 1–5 and 7 of the final report
require formal A/B/C predictions on Monitor504. This protocol only fixes *how* they
will be answered. No Monitor label has been read or scored.

---

## 2. Data and supervision boundary

### 2.1 Frozen split

`tracks/structural-sufficiency/results/phase1a/split_phase1a.json`
(sha256 `9fa96d49efd94cd2ff381922f83a07f963d832064c75469657b4ce5f68c332b6`):

| role | n | row sha256 |
|---|---:|---|
| Fit | 2352 | `c474e38cdecf10071965527409901403ec8b670b96fe4a3c42452531dd0210e9` |
| Monitor | 504 | `fe81cc7904a6da9378f5bf5ae7452ac71629adad83a2a98b917f87299a85f39f` |
| Reserved | 504 | `bc28c5d37c2fe2fbe39c01d5bf2ab9c56630eee2d0a4454a108d150ab2f42d86` |

Roles are disjoint. The split is not re-drawn, and there is no patient split
reselection.

### 2.2 Labels

The only label artifact is
`tracks/structural-sufficiency/results/phase1a/labels_fit_monitor.json`
(sha256 `0260a3909eec30900df646bc760e65360315f4a1398c7092c7c72d04a4e404f7`).
It contains 2856 rows (Fit 2352 + Monitor 504) and certifies
`reserved_labels_present=false` and `official_valid_test_labels_present=false`.
The artifact-level row hash recomputes to
`2ccf20f5293b16dacc5fbaa4542d38ea4bbf2dc3eea68e20f75197ba0fbd8493`.

Label access is implemented as two separate entry points:

- `code/fit_labels.py` scans the JSON text without `json.load`, materialises **only**
  the 2352 requested Fit values, and counts the 504 Monitor values as skipped
  without converting them. Audit: `label_values_constructed=2352`,
  `label_values_skipped=504`, `unrequested_values_constructed=0`.
- `code/monitor_labels.py` refuses to open the artifact unless
  `analysis/predictions/MANIFEST.json` and `MANIFEST.sha256` both exist, verify the
  frozen registry of all 12 predictions, and match every file hash.

No training module imports `monitor_labels` (enforced by a static import scan in
gate G1). `code/screen_training.py`, `screen_data.py`, `screen_models.py`,
`screen_predict.py` and `baseline_ecfp_xgb.py` contain no Monitor evaluation path.

### 2.3 Target standardisation

Computed from **Fit labels only**:

- mean `2.157857142857143`
- scale (population std) `1.2241982660328445`
- Fit label-vector sha256 `2f71c3e236c42cd147b2d519690cad3a33ce08a666c697e1e9bb8c124681969d`

Saved as `analysis/norm_target.json` (sha256
`50c5ad9da285b15233dc0f26704cfb5136ac4bbe5a97b4bf21734d8f1f9a0748`). Training uses
`z = (y - mean) / scale`; predictions are denormalised back to raw `exp`/logD units.
Monitor, Reserved and official valid/test values are never used for standardisation.

### 2.4 Structures

The frozen first-round artifacts are reused read-only:

| artifact | sha256 | content fingerprint |
|---|---|---|
| `feature_spec_lipo.json` | `a65c67f9…4206c` | spec hash `7270f767…a515ee` |
| `norm_lipo.json` | `084c129c…94d77` | `9c54f31ca26b6bc4b303958dfb465f16512fb6c576d23817aff2117d872d4286` |
| `cache/lipo_fit.pt` | `907509ae…00eb` | `0a73407fe3b20bb5158b521bcd8dd94bac69d0857ec5c5bd31314c378e9990d9` |
| `cache/lipo_monitor.pt` | `2abd1aef…8b54` | `db48a9c237d628025185f76497a024dac35537c77413c511b1d76fb3721919b8` |

The reserved cache (`lipo_reserved.pt`) is never loaded. Caches carry no labels
(`labels_present=false`), and their molecule-id order equals the split row order.

Feature schema: atom 51, bond 13, global 42, atom scalars 9; S1 joint codes and S2
typed two-hop paths (`n_nbr_codes`, `n_path_codes` from the frozen spec). The S2
cache stores path categories, endpoints and closure flags; no labels or targets.

---

## 3. Arms

### Arm A — LCRC-A Full

`lipo_v1` LCRC-A from the first round, rebuilt from source, **no SSL weights**.
247,106 parameters. All frozen mechanisms are preserved: AtomEncoder, BondEncoder,
S1 joint neighbourhood statistics, S2 typed two-hop paths, two-hop endpoint content,
closure relation, LocalMLP + LayerNorm, per-atom contribution sum, and
`rho(global42, mean h, mean h², log1p n_atoms)`.

### Arm B — LCRC-A S2-off

Same class, same config, same seed, same tensors, same nominal parameter count as A.
A forward pre-hook on the first LocalMLP linear layer zeroes the concatenated S2
block of its input:

```
[a_v 96 | b_sum 96 | b_mean 96 | S1joint 32 | S2path 104 | S2content 96 | scalars 9] = 529
                                     zeroed: [320:520) = 104 + 96
```

The hook is active in **both training and eval**. Nothing in the frozen model source
is modified, and the hook does not change the state dict.

**Disclosure:** B has the same nominal 247,106 parameters but only **176,514 effective
parameters** (71.43%); 70,592 parameters are dead (51,392 in S2 modules plus a
19,200-column block of the first LocalMLP weight). A vs B is therefore **not an
information-matched control**, and any difference cannot be attributed solely to
"the S2 information". It is an ablation of the S2 input channel under an unequal
effective-capacity budget. This is reported, not corrected.

### Arm C — GINE + global42

Standard 3-layer `GINEConv`, hidden width 170, inner MLP `[Linear, ReLU, Linear]`,
`train_eps=True`, bidirectional real physical-bond edges, LayerNorm residual updates.
Atom51/bond13 features are the same `lipo_v1` features; the adjacency is the real
molecular bond graph, not the S1/S2 code tables. The readout has the same form as A:

```
y = sum_v f(h_v) + rho(global42, mean h, mean h², log1p n_atoms)
```

245,305 parameters (Δ = −1,801, −0.73% vs A). The width was chosen by a deterministic
even-integer search over 128…192 minimising `|N_gine − 247106|`; no performance was
consulted. The full search table is in `analysis/parameters.json`. GINE is a practical
performance reference, not an information-matched mechanism control.

### Arm D — ECFP4 + XGBoost

Frozen BJOC Lipophilicity baseline config: Morgan radius 2, 2048 bits, no feature
invariants; XGBoost `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `min_child_weight=1`,
`tree_method=hist`, `n_jobs=-1`; random states 0/1/2, primary 0. Refit on Fit2352 and
scored on Monitor504 only. No hyperparameter search, no reuse of the earlier BJOC
train3360/valid420 numbers. The baseline is CPU-only and runs locally because RDKit
is not installed on the remote GPU hosts.

### Run registry (12 predictions)

`A_s0..A_s2`, `B_s0..B_s2`, `C_s0..C_s2`, `D_r0..D_r2`. At freeze time, the three
`D_r*` prediction files exist (label-blind); the nine neural predictions are pending
GPU authorization. No partial prediction manifest is written.

---

## 4. Frozen training recipe

| item | value |
|---|---|
| seeds | 0, 1, 2 |
| supervision | Fit 2352 only, scratch, no SSL |
| optimizer | AdamW |
| peak LR | 1e-3 |
| minimum LR | 1e-5 |
| schedule | cosine per optimizer step |
| weight decay | 1e-5 |
| batch size | 128 |
| epochs | 360 |
| steps/epoch | ceil(2352/128) = 19 |
| total updates | 6840 per run |
| gradient clipping | 5.0 |
| dropout | 0 |
| loss | MSE on Fit-standardised target |
| checkpoint | elementwise float mean of the last 5 epoch-end state dicts (epochs 356–360) |
| early stopping / Monitor selection | none |
| arm-specific epochs | none |
| shuffle offset | 917000 (separate from the init seed) |

A and B share an identical initial state hash for each seed. All state-dict entries
are floating point, so the same parameter-averaging rule is safe for A, B and C
(checked by `soup_safety_report`). Formal training refuses any override of these
constants.

GPU training is not assumed bitwise reproducible: the frozen LCRC-A forward uses
`index_add_` atomic accumulation. Same-seed run-to-run variation is disclosed in the
report rather than post-processed.

---

## 5. Evaluation order and access control

1. All 12 predictions are produced **label-blind** and written as
   `analysis/predictions/<key>.npz` plus a JSON sidecar (rows, hash, model metadata,
   `monitor_labels_read=false`).
2. `run_screen.py freeze-predictions` verifies every file hash, the Monitor row order
   and the config hash, then writes `MANIFEST.json` and `MANIFEST.sha256`.
3. Only then may `evaluate` read Monitor504 labels. `monitor_labels.py` re-verifies
   the manifest and lock before opening the label artifact; it returns only the
   Monitor vector and an audit record.
4. Evaluation writes `monitor_metrics.json`, `monitor_paired.json` and
   `monitor_diagnostics.json`. Nothing in evaluation feeds back into training or
   checkpoint choice.

Runtime `access_guard.py` installs a `sys.addaudithook` deny-list for the raw CSV,
official split, `labels.npy`, `labels_meta.json` and the reserved cache, and gates the
single Fit/Monitor label artifact behind a one-shot grant. The guard is armed by both
the training and baseline entry points; denied attempts are recorded in the audit.

Static import scan: only `evaluate.py` (and the CPU gate module) may import
`monitor_labels.py`. Training, data, model, prediction and baseline modules do not.

---

## 6. Metrics and frozen decision rules

**Primary:** raw-scale RMSE, `sqrt(mean((pred - y)²))` in logD units.

**Secondary:** raw MAE; per-seed and mean RMSE/MAE; paired per-molecule
MSE/MAE differences; percentile bootstrap CIs; tail diagnostics; A's
`contrib`/`rho` decomposition.

Uncertainty is reported in two separate buckets:

- **Molecule resampling:** percentile bootstrap over the 504 Monitor molecules,
  2000 resamples, seed 0. Used for paired ΔRMSE CIs.
- **Training seed:** the three per-seed values, their mean, spread and range. Never
  replaced by a best-seed number; `best_seed_minus_mean_rmse` is reported only as an
  optimism diagnostic.

Comparisons are always paired at the molecule level, with `ΔRMSE = RMSE(A) − RMSE(B)`
(negative favours A).

**Materiality threshold:** `0.03` logD RMSE. Classification:

| ΔRMSE and CI | label |
|---|---|
| Δ ≤ −0.03 and CI_high < 0 | `A_MATERIALLY_BETTER` |
| Δ ≤ +0.03 and CI_high ≤ +0.03 | `COMPETITIVE` |
| Δ > +0.03 and CI_low > 0 | `A_MATERIALLY_WORSE` |
| otherwise | `INCONCLUSIVE` |

**S2 increment (A vs B), over the three seeds:**

| condition | label |
|---|---|
| all three per-seed Δ < 0, seed-mean CI_high < 0, mean Δ ≤ −0.03 | `CLEAR_INCREMENT` |
| all three per-seed Δ < 0, seed-mean CI_high < 0 | `SUGGESTIVE_NOT_MATERIAL` |
| otherwise | `NO_CLEAR_INCREMENT` |

**Round case:**

- `CASE_1_S2_INCREMENT_SUPPORTED`: A learned the task (mean RMSE below the Fit-mean
  constant predictor **and** each seed below it), S2 is `CLEAR_INCREMENT`, and A is not
  materially worse than C or D.
- `CASE_2_LCRC_COMPETITIVE_NO_CLEAR_S2_INCREMENT`: A learned and is not materially
  worse than C or D, but S2 is not clear.
- `CASE_3_A_WEAKER_OR_DID_NOT_LEARN`: otherwise.

Additional diagnostics are descriptive and do not change the case: top 1/2/5% error
share, RMSE after removing the worst 2%/5% molecules, sign-flip checks for A-vs-B/C
on the tail, per-seed/molecule variance decomposition, and A's `contrib + rho` vs
prediction identity (`max |contrib + rho − pred|`). Contribution magnitudes are not
treated as chemical causal attribution.

---

## 7. CPU gates (pre-GPU)

All gates are implemented in `code/screen_gates.py`; the evidence JSON is
`analysis/cpu_gates.json` (sha256
`aa29fb09c0135f2946e6b0e5c8dd6bef7ed30a156e303e779fb0a3e3d0b79185`), final status
`READY_FOR_GPU_AUTHORIZATION`.

| gate | check | result |
|---|---|---|
| G0 | frozen spec/norm/split/cache hashes and row order, no labels in caches, no reserved cache | PASS |
| G1 | Fit row/label alignment (2352 constructed, 504 skipped), Monitor unreachable before freeze, import boundary, label-injection cannot change batches | PASS |
| G2 | A/B identical init; B zeroes exactly `[320:520)`; A uses S2 and receives S2 gradients; B receives zero S2 gradients; shared paths bit-identical | PASS |
| G3 | GINE uses real adjacency and bond features; permutation invariance 5.96e-8; batch isolation 2.38e-7; finite gradients everywhere | PASS |
| G4 | batch order hash deterministic; save/restore reproduces the next step bitwise (diff 0.0); soup = manual mean; formal overrides rejected; GPU blocked | PASS |
| G5 | CPU smoke on 256 Fit molecules × 12 epochs for A/B/C; train MSE falls; no Monitor access | PASS |
| G6 | ECFP4 deterministic; row manifest verified for 2856 rows; XGBoost refit bitwise identical (diff 0.0) | PASS |
| G7 | metric/CI/classification unit checks; `evaluate()` refuses before the prediction freeze; partial predictions are label-blind | PASS |
| G8 | deny-list blocks CSV/official split/reserved cache; label grant works once and only once | PASS |

In addition, `tests/` contains 53 pytest checks covering the same surface
(`uv run --group lipo python -m pytest tests`).

---

## 8. Budget

Anchor: one measured A100 run of the same recipe family
(`bjoc_cross_task_lipophilicity_v1/train_P_s0.json`, 419,713 parameters, 3360 rows,
9720 updates, 235.96 s → 0.02428 s/update).

| run | updates | estimate |
|---|---:|---:|
| A, per seed | 6840 | 166 s = 0.046 GPU-h |
| B, per seed | 6840 | 166 s = 0.046 GPU-h |
| C, per seed | 6840 | 332 s = 0.092 GPU-h (2× factor) |
| 9 neural runs | 61,560 | 0.553 GPU-h |
| + 20% overhead and reserves | | **0.734 GPU-h** |
| Arm D (local CPU) | 3 XGBoost fits | 0 GPU-h |

Frozen caps: internal hard cap **1.5 A100 GPU-hours**, researcher-suggested external
cap **4.0 A100 GPU-hours**. The estimate fits the internal cap. The 4-hour figure is a
cap, not an authorization. Detail: `analysis/budget_estimate.json` (sha256
`d7148f13b85f341f0dcb389e4a7232afce136491bedd016d46870d48855d0536`).

---

## 9. Freeze and authorization

`analysis/protocol_freeze.json` records: config hash, hashes of all code/test files,
frozen artifact hashes, recipe constants, decision constants, the 12-key prediction
registry, Fit target statistics, `gpu_authorized=false`, `gpu_hours_used=0.0`, and the
implementation git commit at freeze time. `git_dirty_tracked` covers source files only
(`code/`, `tests/`, `configs/`); `git_dirty_worktree` records the full tracked worktree.
Formal runs require the full worktree to be clean, not just the source.

To run formal training, the researcher must create
`analysis/gpu_authorization.json` with at least:

```json
{
  "authorized": true,
  "protocol_id": "lcrc-lipo-screen-v1",
  "protocol_freeze_sha256": "<sha256 of analysis/protocol_freeze.json>",
  "max_gpu_hours": 1.5,
  "authorized_runs": ["A_s0", "A_s1", "A_s2", "B_s0", "B_s1", "B_s2", "C_s0", "C_s1", "C_s2"],
  "researcher_note": "<explicit authorization text>"
}
```

`run_screen.py train` additionally requires a clean committed worktree and verifies
that no code/config hash has drifted. Missing, invalid, over-cap or run-mismatched
authorization is a hard `PermissionError`; there is no bypass flag.

---

## 10. Pre-GPU implementation record (no protocol change)

These corrections were made during implementation, before any formal run and before
the final freeze. None was chosen by looking at Monitor labels or performance.

1. **S2-off slice:** an initial `start = 3*atom_out + 2*rank` produced a 465-wide
   input; corrected to `start = 3*atom_out + stat_rank = 320`, block
   `[320, 520)`, total input 529.
2. **GINE edge orientation:** `GINEConv` requires `[2, E]`; the cached bond array is
   `[E, 2]`, so it is transposed before duplicating directions.
3. **GINE parameter count:** the reference is 245,305 (not 245,302) at hidden 170;
   the config, code and parameter table were corrected before freeze.
4. **RDKit generator API:** `useFeatures` is no longer accepted by the installed
   RDKit 2026.09.1 `GetMorganGenerator`; the frozen `use_features=false` case uses the
   plain generator, and a feature-invariant generator is attached if the flag is ever
   true. The frozen baseline is unaffected.
5. **Soup identity tolerance:** averaging five identical float32 states can differ by
   ≤1 ulp; the gate checks `<1e-6` for that identity case, while the soup-vs-manual
   equality remains bitwise.

Arm D predictions for `D_r0..D_r2` were produced locally **after** the protocol freeze
and **without reading Monitor labels**; they are not scored. Their sidecars certify
`monitor_labels_read=false`. No prediction manifest exists, so the evaluation entry
point remains locked.

---

## 11. Non-claims

- This round has **no** Lipophilicity performance result. CPU gates validate
  implementation, not predictive validity.
- Monitor504 is a development set that SSRC used before; it is not a fresh blind test.
- Reserved and official valid/test labels were never read and are not scored.
- Arm B is not information-matched, and arm C is not an information-matched or
  parameter-identical control.
- The CPU smoke losses cannot rank the arms.
- No causal or chemical-faithfulness claim follows from `contrib`/`rho` magnitudes.
- GPU results, if authorized later, will be reported in raw units with all seeds and
  the frozen decision rule; no post-hoc seed or checkpoint selection is permitted.
