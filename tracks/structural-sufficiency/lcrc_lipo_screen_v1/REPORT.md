# REPORT — LCRC-A Lipophilicity screen v1

- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Final status:** `AMENDMENT_CPU_VERIFIED` / `READY_FOR_REAUTHORIZATION`
- **Scientific result:** **none.** No formal training completed, no Monitor prediction
  exists, no label was read, no `CASE_1/2/3` can be assigned.
- **GPU:** 0 GPU-seconds under the current (amended) freeze. The pre-amendment stage
  used 0.0139 A100 GPU-hours before halting (preserved in
  `analysis/gpu_stage_halt.json`). The historical authorization is **not** reusable.
- **Protocol freeze (amended):** `analysis/protocol_freeze.json`
  sha256 `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8`
  (previous v1 freeze: `87ec475107121d7b48cf9ae1e2664bb27bba972c39a543ebf24ea7885dbed9f4`)
- **Amendment record:** [`AMENDMENT-001-fit-batch-index.md`](AMENDMENT-001-fit-batch-index.md)
- **CPU gates:** 10/10 PASS, final status `READY_FOR_REAUTHORIZATION`
  (`analysis/cpu_gates.json`)
- **Tests:** 59 passed (10 test files)

This report has two parts: the halted v1 GPU attempt (§1), and the index fix plus CPU
re-verification that followed (§2–§6). The scientific questions remain unanswered; the
package is now engineering-ready for a new authorization, not authorized.

---

## 1. The halted v1 GPU stage (history, preserved)

The v1 package was authorized for 1.5 A100 GPU-hours. It was deployed with `rr` and run
on `res` (2×A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124). `A_s0` and `A_s1`
both exited 1 after 16–17 s at the first optimizer batch:

```
screen_training.py:165 -> screen_data.py:133 -> lcrc_data.py:658
ValueError: row index outside cache
```

Full ledger: `analysis/gpu_stage_halt.json` (sha256 `111275dc…`), commit `e09c627`.
Zero of nine runs completed; no checkpoint, prediction, manifest or label access.

### Root cause

Three index spaces were conflated: `global_row_id` (0…4199), `fit_cache_pos` (0…2351)
and `shuffle_pos` (0…n_subset−1). The old loop passed a global row id to `collate()` as
if it were a cache position, while looking the label up by the same global id:

| epoch 1, seed 0 (real frozen Fit split) | value |
|---|---:|
| out-of-range slots (`g ≥ 2352`) | **1162 / 2352**, 19/19 batches |
| first batch out-of-range | 66 / 128 |
| in-range slots with the wrong molecule/label pair | **1185 / 1190** |
| cache positions where `molecule_id == position` | 2 / 2352 |

All nine runs shared the loop. The CPU gates missed it because G5 smoke used the first
256 Fit rows (global ids ≲ 460) and never exercised the full-row range or asserted
pairing.

## 2. The amendment — minimal index fix

`code/screen_data.py` now exposes one explicit interface (see AMENDMENT-001 §3):

- `FitBatchPlan` — `shuffle_positions`, `cache_positions`, `global_row_ids`, `n_subset`;
- `plan_fit_batch()` — shuffle position → cache position → global id (provenance only),
  rejecting tensors, non-integers, duplicates, out-of-subset indices and any cache order
  that differs from the frozen split;
- `assemble_fit_batch()` — **requires a plan** (raw lists raise `TypeError`), collates
  cache positions and takes `y_fit[cache_positions]`;
- `fit_epoch_plans()`, `fit_batch_fingerprints()` — replay and fingerprint an epoch.

`code/screen_training.py` uses this path for both formal training and CPU smoke:
`targets = y_fit[cache_positions]` and inputs `collate(cache, cache_positions)` now share
one index. `train_row_index` (the bug vector) was removed. Each epoch records
`batch_alignment.per_epoch[]` with position/global-id/target hashes and coverage flags.

| index | before | after |
|---|---|---|
| input molecule | cache position = global row id | cache position = shuffle position |
| target | `y_fit[shuffle_pos]` (wrong molecule) | `y_fit[cache_pos]` (same molecule) |
| global row id | used as a position | identity checks and provenance only |

## 3. New gate G9 — full-Fit batch alignment

`gate_full_fit_alignment()` uses the real frozen Fit2352 cache and labels; for each seed
0/1/2 it builds **all 19 batches of one full epoch** through the executed path and
checks: 2352 positions each exactly once and in `0…2351`; last batch 48; per-molecule
`cache_positions[p] == cache.molecule_ids[p] == split row` and target equal to an
**independently scanned** `global_row_id → label` map; batch/target length equality; and
positions fingerprint equal to `batch_order_hash(seed, epochs=1)`.

It also recomputes the pre-amendment mapping from the same plans and requires it to fail
(sensitivity), and runs a real 1-epoch 256-row smoke through `train_one` whose recorded
fingerprints must equal the gate's recomputation (executed path == gate path).

| seed | batches | last | positions | ids > 2351 | old out-of-range | old label mismatch | = order hash |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |
| 1 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |
| 2 | 19 | 48 | 2352 | 1162 | 1162 | 1185 | yes |

Position hashes: seed 0 `2b27df0b…`, seed 1 `dc9d5b40…`, seed 2 `cd7e0927…`.
Executed path (256-row smoke): positions `5027456e…`, ids `86dc3dc2…`, targets
`19d2e869…` — identical to the recomputation. Gate time ≈1.2 s.

## 4. CPU re-verification

### Gates (10/10 PASS)

`analysis/cpu_gates.json`: G0 frozen artifacts, G1 label alignment/isolation, G2 LCRC/S2
ablation, G3 GINE, G4 numerics/soup/save-restore/**authorization guard**, G5 smoke,
G6 ECFP4/XGB, G7 evaluation order, G8 access control, **G9 full-Fit alignment**. Total
≈10 s. G4 now probes the authorization mechanism with temporary files (missing, wrong
freeze sha, over cap, unlisted run, empty note rejected; valid accepted) instead of
assuming the file is absent, and records
`real_authorization_matches_current_freeze=false` — the v1 authorization pins the old
freeze and is refused.

### pytest

59 passed in 3.4 s (10 files), including the new `tests/test_fit_alignment.py`
(seed 0/1/2 full-epoch alignment, interface rejections, executed-path fingerprints) and
the rewritten authorization test.

### Regenerated CPU smoke (real path, 256 Fit molecules × 12 epochs = 24 updates)

| arm | train MSE first → last | best | Fit soup RMSE (raw) | soup hash | clip | non-finite |
|---|---:|---:|---:|---|---:|---:|
| A | 151.104 → 0.932 | 0.879 | 1.1146 | `fc00c30663ef…` | 24 | 0 |
| B | 113.537 → 0.825 | 0.825 | 1.1424 | `46fa926e75a2…` | 24 | 0 |
| C | 423.958 → 0.827 | 0.827 | 1.1577 | `fb4caef4a9bd…` | 24 | 0 |

Every epoch of every arm reports 100% subset coverage, unique in-range positions, and a
matching batch-order hash (`bbfffc463583…` for seed 0 / 256 / 12 epochs). All losses,
gradients and optimizer steps are finite. These are Fit-only engineering checks and
**cannot rank the arms**.

### The old smoke is invalidated, not deleted

`analysis/smoke_invalidated_pre_amendment.json` is byte-identical to the old
`smoke.json` (sha256 `fef92f3d…`). It was produced by the defective assembly, so its
curves are not training evidence; the new `smoke.json` records
`supersedes.status = INVALIDATED_PRE_AMENDMENT` and the old sha.

## 5. Scientific design unchanged

The amended freeze keeps the v1 `recipe`, `decision_constants`, frozen artifact hashes
and 12-key prediction registry byte-identical: LCRC-A / S2-off / GINE / ECFP4+XGB,
seeds 0/1/2, 360 epochs, 19 steps/epoch, 6840 updates, batch 128, AdamW 1e-3→1e-5,
wd 1e-5, clip 5.0, dropout 0, MSE on Fit-standardised targets, last-5 soup, no early
stopping or seed selection, RMSE/MAE, 2000-resample bootstrap (seed 0), materiality
0.03, and the `CLEAR_INCREMENT` / `CASE_1/2/3` rules. Only code hashes, the new gate and
amendment metadata differ.

## 6. Required questions — still unanswered

| # | question | status |
|---|---|---|
| 1 | Did scratch LCRC-A learn the task? | **no data** (0/9 formal runs) |
| 2 | Is the S2 increment over B stable? | **no data** |
| 3 | Is A competitive with GINE and ECFP4? | **no data** |
| 4 | Seed / molecule sensitivity? | **no data** |
| 5 | Do `contrib` and `rho` both contribute? | **no data** (implementation only) |
| 6 | Train-vs-Monitor gap? | **no data** |
| 7 | CASE_1 / CASE_2 / CASE_3? | **not assignable** |

No RMSE/MAE, per-seed table, bootstrap interval, tail share or diagnostic is estimated
here.

## 7. Budget and provenance

| item | GPU hours |
|---|---:|
| v1 halted stage (`lcrc-timing-a`, `lcrc-formal-A-s0`, `lcrc-formal-A-s1`) | 0.0139 |
| current amended freeze | 0.0 |
| remaining of the 1.5 h cap | 1.486 |

The v1 authorization (`48cb9a28…`) is kept as an execution record but cannot authorize
the amended freeze: `require_gpu_authorization` compares the pinned sha with the live
freeze and refuses. No GPU job was launched and no hash check was bypassed during this
amendment.

## 8. Status and next step

- Round status: **`AMENDMENT_CPU_VERIFIED` / `READY_FOR_REAUTHORIZATION`**.
- Blocking item: a **new explicit researcher authorization** pinned to the amended
  freeze `7fffba42…` is required; nothing else is.
- When re-authorized: deploy the committed revision with `rr`, run the nine frozen runs,
  produce nine label-blind predictions plus the existing `D_r0..D_r2`, run
  `freeze-predictions`, then exactly one `evaluate`.
- Not authorized and not started: SSL, AQSOL or other tasks, extra ablations, Reserved
  labels, official valid/test, any recipe change or retry of the failed v1 jobs.

## 9. Artifact index (amended)

| artifact | sha256 |
|---|---|
| `analysis/protocol_freeze.json` | `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8` |
| `analysis/cpu_gates.json` | `7e47ae2bc14f5b7ea03a3f4fd7a8d81ffe4aecc31df07d24d8cd4425cf420da3` |
| `analysis/smoke.json` | `eb2869a952cefe07dbf850a04a937326c109e693c8a59e46d4520121c7e78756` |
| `analysis/smoke_invalidated_pre_amendment.json` | `fef92f3d0badfbbc7af5aca8f8b22c8f7399aa88604d8cad86b9c4b5b420fd4a` |
| `analysis/parameters.json` | `cef8946700c4e9d4c3311147428349192ac6f8dfcae2eb54c353e03e5f87b646` |
| `analysis/budget_estimate.json` | `b19037db9b2b141885ca8ddfb7405dc5ba751b4598f40797aa18389aba6a475f` |
| `analysis/norm_target.json` | `50c5ad9da285b15233dc0f26704cfb5136ac4bbe5a97b4bf21734d8f1f9a0748` |
| `analysis/gpu_authorization.json` (v1, historical) | `48cb9a28ff36ec6765bcbc05d72391f5018e123a1ae8cef11d89196e44a3f30f` |
| `analysis/gpu_stage_halt.json` | `111275dcc2037f19526fcb2c041a3e2e2868955e66755c065d52301fad72b113` |
| `analysis/predictions/D_r0.npz` | `e2a18b9be03ed9e1da96614e9337abf2da577e14079d1a43958d295f736696b2` |
| `analysis/predictions/D_r1.npz` | `d2b42ebcc4a024d7e0dafa12e2c7672eb2650797361ba19a4e1704dc2c9be6c1` |
| `analysis/predictions/D_r2.npz` | `af4831f7dada243cde31f098e23dcfda4edc02e4582929dbd36ced9e49f46b78` |

## 10. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_lipo_screen_v1
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py cpu-gates
uv run --project /home/calendar/code/research --group lipo python -m pytest tests -q
uv run --project /home/calendar/code/research --group lipo python code/run_screen.py status
```

Expected: `READY_FOR_REAUTHORIZATION`, 10/10 gates PASS, 59 tests pass, and
`state = READY_FOR_REAUTHORIZATION` with
`gpu_authorization_matches_current_freeze = false`.
