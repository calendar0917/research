# AMENDMENT-001 — Fit batch index semantics (lcrc-lipo-screen-v1)

- **Amendment id:** `amendment-001-fit-batch-index`
- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Date:** 2026-10-30
- **Type:** implementation correction (code + gate), **not** a scientific redesign
- **Status:** `AMENDMENT_CPU_VERIFIED` / `READY_FOR_REAUTHORIZATION` (GPU not restarted)
- **Previous freeze:** `analysis/protocol_freeze.json`
  sha256 `87ec475107121d7b48cf9ae1e2664bb27bba972c39a543ebf24ea7885dbed9f4`,
  git commit `c3202fee7d9a` (freeze artifacts committed in `5a597b1`).
  Its byte content remains retrievable with
  `git show e09c627:tracks/structural-sufficiency/lcrc_lipo_screen_v1/analysis/protocol_freeze.json`
- **Authorization under the previous freeze:** `analysis/gpu_authorization.json`
  sha256 `48cb9a28ff36ec6765bcbc05d72391f5018e123a1ae8cef11d89196e44a3f30f`
  (commit `4d8f136`), 9 runs, 1.5 A100 GPU-hours
- **Failed GPU stage record (preserved):** `analysis/gpu_stage_halt.json`
  sha256 `111275dcc2037f19526fcb2c041a3e2e2868955e66755c065d52301fad72b113`;
  0.0139 GPU-hours used; 0/9 runs completed; no labels read

This file records the amendment required by `DECISION.md` §4/§7. It is an
engineering record: **the scientific design, recipe, data, metrics and decision rules
are unchanged.** No GPU run is started by this amendment.

---

## 1. Failure that triggered the amendment

Two formal runs (`A_s0`, `A_s1`) on `res` (A100-SXM4-40GB) exited 1 after 16–17 s at the
first optimizer batch:

```
screen_training.py:165  batch = D.to_device(D.collate(caches["fit"]["cache"], cache_rows), device)
screen_data.py:133      return prep_data.collate(cache, rows)
lcrc_data.py:658        raise ValueError("row index outside cache")
```

The run ledger, timings and tracebacks are in `analysis/gpu_stage_halt.json` and in git
history (`e09c627`).

## 2. Root cause

Three index spaces were conflated:

| space | range | meaning |
|---|---|---|
| `global_row_id` | 0…4199 | original Lipophilicity dataset row id (frozen split + label key) |
| `fit_cache_pos` | 0…2351 | position in the frozen Fit cache; also the row of `y_fit` |
| `shuffle_pos` | 0…n_subset−1 | position inside the shuffled training subset |

The old loop did:

```python
cache_rows = [train_rows[int(i)] for i in rows]        # shuffle_pos -> global_row_id
batch = D.collate(cache, cache_rows)                   # global_row_id used as cache_pos
y = y_fit[[train_row_index[r] for r in cache_rows]]    # label looked up by global_row_id
```

so the input molecule was the one at cache position `g = global_row_id`, while the label
was the one at the *shuffle* position. Measured on the real frozen Fit split:

| quantity (epoch 1, seed 0) | value |
|---|---:|
| out-of-range slots (`g ≥ 2352`) | **1162 / 2352**, in 19/19 batches |
| first batch out-of-range slots | 66 / 128 |
| in-range slots with a wrong molecule/label pair | **1185 / 1190** |
| cache positions where `molecule_id == position` | 2 / 2352 |

All nine planned runs (A/B/C × s0/s1/s2) use the same loop, so the defect was
arm/seed-independent. The CPU gates missed it because G5 smoke used only the first 256
Fit rows (global ids ≲ 460), which never hit the range check and do not raise on a
mislabelled pair; no gate exercised the full 2352-row epoch or asserted
`(molecule id, label row id)` pairing.

## 3. Code change (minimal)

No data, split, feature, model, recipe or decision constant was touched.

### `code/screen_data.py` — explicit index interface (new)

- `FitBatchPlan` (frozen dataclass): `shuffle_positions`, `cache_positions`,
  `global_row_ids`, `n_subset`.
- `fit_training_positions(n_subset)`: the subset's cache positions (`0..n-1`).
- `plan_fit_batch(shuffle_positions, *, train_positions, cache, fit_global_rows)`:
  translates shuffle positions → cache positions → global ids (provenance only).
  Rejects tensors, non-integers, duplicates, out-of-subset positions and any cache
  order that differs from the frozen split.
- `assemble_fit_batch(cache, y_fit, plan)`: **requires a `FitBatchPlan`**; a raw index
  list (e.g. dataset row ids) raises `TypeError`. Collates cache positions and takes
  `y_fit[cache_positions]`, so the input molecule and its target always share the same
  cache position.
- `fit_epoch_plans(seed, epoch, n_subset, cache, fit_global_rows)`: the plans for one
  epoch of the same shuffle stream the trainer consumes.
- `fit_batch_fingerprints(plans, y_fit)`: order-preserving SHA-256 of positions,
  global ids and targets plus coverage/uniqueness flags.

### `code/screen_training.py` — uses the interface

| before | after |
|---|---|
| `cache_rows = train_rows[shuffle_pos]` (global id) | `plan = plan_fit_batch(shuffle_pos, …)` |
| `collate(cache, cache_rows)` (wrong space) | `assemble_fit_batch(cache, y_fit, plan)` |
| label via `train_row_index[global_id]` | label = `y_fit[cache_pos]` from the same plan |
| no per-batch provenance | `batch_alignment.per_epoch[]` fingerprints recorded for every epoch |

`train_row_index` (the mixing vector) was removed. The formal run and the CPU smoke share
this single path. `fit_rows_sha256` is now computed from the subset's global ids derived
from cache positions (identical values, explicit direction).

### `code/screen_gates.py`

- New **G9 `full_fit_batch_alignment`** (below).
- G4's authorization check no longer depends on the authorization file being absent: it
  probes temporary authorization files (missing / wrong freeze sha / over cap / unlisted
  run / empty note must be rejected; a valid one accepted) and reports whether the real
  file matches the current freeze.
- G5 smoke now asserts per-epoch alignment coverage and records it.
- `run_all` final status is `READY_FOR_REAUTHORIZATION`.

### `code/run_screen.py`

- Status reports `gpu_authorization_matches_current_freeze`, prior GPU hours and the
  `READY_FOR_REAUTHORIZATION` / `AUTHORIZED_UNDER_CURRENT_FREEZE` / `BLOCKED` state.
- `cpu-gates` accepts the new final status.

### `code/screen_common.py`

- Freeze record gains an `amendment` block (previous freeze sha, halt record, reason,
  "design changed: false", new gate, prior GPU hours). The frozen recipe block is
  unchanged.

### `tests/`

- `tests/test_training.py::test_cuda_training_requires_matching_authorization` replaces
  the state-dependent "authorization file must be absent" assertion with a
  state-independent probe of the guard.
- `tests/test_fit_alignment.py` (new): full-Fit epoch coverage/alignment for seeds 0/1/2
  against an independent `global_row_id → label` map, rejection of dataset row ids at
  the cache-position interface, and an executed-path vs gate-path fingerprint check.

## 4. New gate G9 — full-Fit batch alignment

`gate_full_fit_alignment()` uses the real frozen Fit2352 cache and labels and, for
**each** seed 0/1/2, builds **all 19 batches of a full epoch**:

- every batch goes through `plan_fit_batch` + `assemble_fit_batch` (the executed path);
- 2352 cache positions, each exactly once, all in `0…2351`;
- the last batch has 48 molecules and collates normally;
- for every molecule: `cache_positions[p]`, `cache.molecule_ids[p]` and the frozen
  split row agree; the target equals `label_by_global[global_row_id]` from an
  **independently scanned** label map (artifact scanned in reversed row-id order) and
  equals `y_fit[cache_pos]`;
- batch input length equals target length;
- the positions fingerprint equals `batch_order_hash(seed, epochs=1)` (gate path ==
  frozen order function);
- **sensitivity:** the pre-amendment mapping is recomputed from the same plans and must
  show the original failure (out-of-range slots > 0 and label mismatches > 0), so the
  gate provably fails on the old code;
- **executed path:** a real 1-epoch, 256-row smoke through `train_one` must reproduce the
  gate's per-epoch `cache_positions`, `global_row_ids` and `targets` fingerprints.

### Result (from `analysis/cpu_gates.json`, generated before the freeze regeneration)

| seed | batches | last | positions | ids > 2351 | old out-of-range | old label mismatch | = `batch_order_hash` |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |
| 1 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |
| 2 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |

Positions hashes: seed 0 `2b27df0b…`, seed 1 `dc9d5b40…`, seed 2 `cd7e0927…`.
Executed-path record (256-row smoke, seed 0): positions `5027456e…`, ids `86dc3dc2…`,
targets `19d2e869…` — identical to the gate recomputation.

## 5. Re-run CPU evidence

### Original CPU gates (all ten)

`analysis/cpu_gates.json`: G0–G9 all PASS, final status `READY_FOR_REAUTHORIZATION`.
G4 (numerics/soup/save-restore/authorization guard), G5 (smoke), G6 (ECFP4/XGB),
G7 (evaluation order), G8 (access control) unchanged in intent; G9 is new.

### CPU smoke (regenerated, real training path)

`analysis/smoke.json` (sha in the appendix). 256 Fit molecules, 12 epochs, 24 updates,
CPU only; each run also records full per-epoch alignment.

| arm | train MSE first → last | Fit soup RMSE (raw) | soup hash | alignment |
|---|---:|---:|---|---|
| A | 151.104 → 0.932 | 1.1146 | `fc00c30663ef…` | 12/12 epochs covered |
| B | 113.537 → 0.825 | 1.1424 | `46fa926e75a2…` | 12/12 epochs covered |
| C | 423.958 → 0.827 | 1.1577 | `fb4caef4a9bd…` | 12/12 epochs covered |

No non-finite loss, gradient or optimizer step; `grad_norm_max` and clip events are in
`analysis/smoke.json`. These are **Fit-only engineering checks**, not a model ranking.

### Previous smoke is invalidated, not deleted

`analysis/smoke_invalidated_pre_amendment.json` is a byte-identical copy of the
pre-amendment `smoke.json` (sha256
`fef92f3d0badfbbc7af5aca8f8b22c8f7399aa88604d8cad86b9c4b5b420fd4a`). Its loss curves
came from the defective assembly and are **not** valid training evidence. The new
`smoke.json` records `supersedes.status = INVALIDATED_PRE_AMENDMENT` and the old sha.

### pytest

The full suite is run after the freeze regeneration; results are in the appendix.

## 6. Scientific design: unchanged

Verified unchanged by this amendment:

- Arms: LCRC-A Full, LCRC-A S2-off, GINE + global42, ECFP4 + XGBoost.
- Seeds 0/1/2; 360 epochs; 19 steps/epoch; 6840 updates; batch 128.
- AdamW, peak LR 1e-3 → 1e-5 per-step cosine, weight decay 1e-5, grad clip 5.0,
  dropout 0, MSE on Fit-standardised targets.
- Checkpoint = elementwise mean of the last five epoch-end state dicts; no early
  stopping, no Monitor/seed selection.
- Fit2352 / Monitor504 / Reserved504 split, FeatureSpec, caches, Fit-only
  standardisation.
- Evaluation: raw-scale RMSE (primary), MAE, paired molecule bootstrap (2000, seed 0),
  materiality 0.03, and the `CLEAR_INCREMENT` / `CASE_1/2/3` rules.

The freeze's `recipe`, `decision_constants`, `expected_*` hashes and
`prediction_registry` blocks are byte-identical to v1. Only code hashes, the new gate and
the amendment metadata differ.

## 7. Status and next step

- Round status after the CPU re-verification: **`AMENDMENT_CPU_VERIFIED` /
  `READY_FOR_REAUTHORIZATION`**.
- The historical authorization remains on disk as an execution record. Because it pins
  the v1 freeze sha, it **cannot** authorize runs under the amended freeze
  (`require_gpu_authorization` compares the live freeze sha).
- No GPU job was launched, no new authorization was created, no hash check was bypassed,
  and the failed `A_s0`/`A_s1` jobs were not resumed.
- A new explicit researcher authorization pinned to the amended freeze is required
  before the nine formal runs may be attempted.

---

## Appendix — final verification (written after the freeze regeneration)

- **New freeze:** `analysis/protocol_freeze.json`
  sha256 `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8`,
  generated at commit `131937b` (recorded `git_dirty_worktree=true` only because the
  generated analysis artifacts were rewritten by the freeze step; `git_dirty_tracked`
  is false and the code hashes match `131937b`).
- **Recorded design unchanged:** `recipe`, `decision_constants`, expected artifact hashes
  and the 12-key prediction registry are identical to v1. `gpu_authorized=false` and
  `gpu_hours_used_this_freeze=0.0`; `gpu_hours_used_before_amendment=0.0139`.
- **CPU gates:** `analysis/cpu_gates.json`
  sha256 `7e47ae2bc14f5b7ea03a3f4fd7a8d81ffe4aecc31df07d24d8cd4425cf420da3`;
  G0–G9 all PASS; final `READY_FOR_REAUTHORIZATION`; ≈10 s. G4 records
  `real_authorization_matches_current_freeze=false` (v1 authorization pins the old
  freeze and is refused).
- **pytest:** `59 passed in 3.4 s` (10 files), including `tests/test_fit_alignment.py`.
- **Smoke:** `analysis/smoke.json`
  sha256 `eb2869a952cefe07dbf850a04a937326c109e693c8a59e46d4520121c7e78756`;
  A 151.104→0.932, B 113.537→0.825, C 423.958→0.827; all finite; 12/12 epochs aligned.
- **Invalidated old smoke:** `analysis/smoke_invalidated_pre_amendment.json`
  sha256 `fef92f3d0badfbbc7af5aca8f8b22c8f7399aa88604d8cad86b9c4b5b420fd4a`
  (byte-identical to the v1 file).
- **Other regenerated artifacts:** `analysis/parameters.json`
  `cef8946700c4e9d4c3311147428349192ac6f8dfcae2eb54c353e03e5f87b646`;
  `analysis/budget_estimate.json`
  `b19037db9b2b141885ca8ddfb7405dc5ba751b4598f40797aa18389aba6a475f`;
  `analysis/norm_target.json` unchanged
  `50c5ad9da285b15233dc0f26704cfb5136ac4bbe5a97b4bf21734d8f1f9a0748`.
- **Historical records untouched:** `analysis/gpu_authorization.json`
  `48cb9a28…`, `analysis/gpu_stage_halt.json` `111275dc…`.
- **GPU:** no job launched, no authorization created, no hash check bypassed. Total GPU
  hours remain 0.0139 (v1 halt) of the 1.5 h cap.
- **Remaining blocker:** a new researcher authorization pinned to `7fffba42…`.

