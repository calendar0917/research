# REPORT — LCRC-A Lipophilicity screen v1

- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Final status:** `HALTED_IMPLEMENTATION_DEFECT` (formal GPU stage attempted, stopped
  on the first optimizer-step batch)
- **GPU:** authorized for 1.5 A100 GPU-hours and used for **0.014 GPU-hours** (50 s of
  A100 wall time across three jobs). No training run completed. No Monitor prediction
  was produced.
- **Monitor/Reserved/official labels:** not read, not scored. No prediction manifest
  exists, so `monitor_labels.py` / `evaluate.py` were never invoked.
- **Protocol freeze:** `analysis/protocol_freeze.json`
  (sha256 `87ec475107121d7b48cf9ae1e2664bb27bba972c39a543ebf24ea7885dbed9f4`),
  unchanged. No frozen code, config, test or artifact was edited.
- **Halt evidence:** `analysis/gpu_stage_halt.json`
- **CPU gates:** 9/9 PASS (`analysis/cpu_gates.json`) — **but** they do not cover the
  full-Fit training batch path; see §2.3.
- **Tests now:** 52 passed, 1 failed. The single failure is
  `tests/test_training.py::test_cuda_training_requires_authorization`, which asserts the
  *pre-GPU absence* of `analysis/gpu_authorization.json`; it is a state assertion that
  the researcher's authorization supersedes. The test was **not** edited (that would
  break the frozen code hashes).

This report contains **no Lipophilicity performance result**. It records an attempted
formal execution that failed on a code defect before any label was read. The scientific
questions of the round remain unanswered.

---

## 1. What was executed

Order of operations actually performed on 2026-10-30 (UTC+8):

1. Read `AGENT.md`, `PROTOCOL.md`, `DECISION.md`, `README.md`,
   `analysis/protocol_freeze.json`; verified the frozen config/code/artifact hashes and
   that no formal `analysis/runs/*/train.json` existed locally.
2. Searched remote state with `rr jobs`. Predating exploratory runs from 2026-10-28
   (`lcrc-A-s0`, `lcrc-R-s0`, `lcrc-N-s0`, commits `42c01a3c` / `d2d3743b` / `b9367655`)
   exist but are **not** this protocol: they used 600+ epochs with per-epoch valid
   best-checkpoint selection. Nothing was reused or continued.
3. Created and committed `analysis/gpu_authorization.json` pinning the freeze sha256,
   the nine runs and `max_gpu_hours=1.5`.
4. Deployed the committed revision `4d8f13660000` with `rr deploy res-2` and
   `rr deploy res`. No manual SSH launch was used.
5. `rr` preflight on `res-2` (CPU pool) verified the freeze, the authorization, the
   clean worktree, and rebuilt both structural caches from the frozen raw artifact with
   matching fingerprints (`0a73407f…`, `db48a9c2…`).
6. `res-2` GPU pool `res2-cu124` was saturated by unrelated unlimited-time jobs
   (`c05`/`c06`; three submissions stayed `PENDING` on `Resources`/`Priority` for
   ~18 min). They were cancelled **before starting** (0 GPU-seconds) and execution moved
   to the healthy A100 host `res` (2× A100-SXM4-40GB, driver 550.163.01,
   `torch 2.5.1+cu124`, `cuda_available=True`), still through `rr`.
7. Budget calibration on `res-gpu0` (arm A, batch 128, 13 optimizer steps):
   **0.01485 s/update** → 101.6 s extrapolated per 6840-update run. The frozen recipe
   was estimated to fit comfortably inside 1.5 h.
8. Two formal runs launched concurrently: `A_s0` on `res-gpu0`, `A_s1` on `res-gpu1`.
   Both exited with code 1 after 16–17 s at the **first batch of epoch 1**.

### Run ledger (all through `rr`)

| experiment | run id | host/pool | state | exit | seconds |
|---|---|---|---|---|---:|
| `lcrc-preflight` | `lcrc-preflight-20261030-225323-8d02828d` | res-2 / res2-cpu | completed | 0 | 42 |
| `lcrc-timing-a` | `lcrc-timing-a-20261030-231346-6dce3bb8` | res / res-gpu0 | completed | 0 | 17 |
| `lcrc-formal-A-s0` | `lcrc-formal-A-s0-20261030-231545-30e34cc8` | res / res-gpu0 | **failed** | 1 | 16 |
| `lcrc-formal-A-s1` | `lcrc-formal-A-s1-20261030-231557-3aa197e8` | res / res-gpu1 | **failed** | 1 | 17 |
| `lcrc-A-s0/s1/s2` | `…-20261030-2255…` | res-2 / res2-cu124 | cancelled pre-start | – | 0 |

GPU-hours charged: `50 s = 0.0139 h` of the 1.5 h cap (0.93%). Budget is not the
binding constraint.

---

## 2. Failure

### 2.1 Symptom

```
File "…/lcrc_lipo_screen_v1/code/screen_training.py", line 165, in train_one
    batch = D.to_device(D.collate(caches["fit"]["cache"], cache_rows), device)
File "…/lcrc_cross_task_preparation_v1/code/lcrc_data.py", line 658, in collate
    raise ValueError("row index outside cache")
ValueError: row index outside cache
```

The same traceback occurred for seed 0 and seed 1, on two different GPUs, at the first
optimizer step. No `train.json`, no `soup.pt`, no prediction and no manifest were
written.

### 2.2 Root cause (deterministic, reproducible on CPU)

`splits_phase1a["fit"]` holds **global molecule row ids** `[0, 1, 3, 4, 7, …, 4199]`,
and the Fit cache stores its 2352 molecules in exactly that order. `collate()` indexes
**cache positions** `0…2351`. The training loop mixes the two:

- `screen_training.py:164` `cache_rows = [train_rows[int(i)] for i in rows]` turns the
  shuffled **position** `i` into a **global id** `g`;
- line 165 passes `g` to `collate` as if it were a cache position;
- line 167 looks the label up by `g`, i.e. it pairs the graph at cache position `g`
  with the label of global row `g`.

Consequences on the frozen Fit split:

| quantity | value |
|---|---|
| first batch (epoch 1) slots with global id ≥ 2352 | **66 / 128** (max id 4197) |
| epoch-1 slots out of range | **1162 / 2352**, in **19/19** batches |
| cache positions where molecule id equals position | **2 / 2352** |
| in-range epoch-1 slots whose label belongs to a different molecule | **1188 / 1190** |

So the formal loop cannot survive epoch 1: every batch contains at least one index that
is out of range. Even if the range were larger, the (molecule, label) pairing would be
wrong for essentially every row. All nine planned runs (A/B/C × s0/s1/s2) share this
loop, so the defect is arm- and seed-independent.

### 2.3 Why the CPU gates did not catch it

G5 smoke trained on `fit_limit=256`, and the G4 numerics checks used 128-row batches
taken from the beginning of the Fit list. Their global ids are all < 2352 (roughly
0…460), so the range assertion never fired, and a mislabelled pair does not raise — the
loss still falls. No gate asserted full-Fit `(molecule id, label row id)` pairing or
exercised a batch containing a global id ≥ 2352. Gate G1 verified the *label file*
alignment (`fit_labels` returns values in split order), which is correct; the defect is
in the trainer's position/id translation.

### 2.4 What was deliberately not done

- The frozen code was **not** patched, even though the fix is small (collate by cache
  position; derive the label index from the position). A patch changes code hashes,
  invalidates `analysis/protocol_freeze.json`, and requires a new authorization.
- No run was retried with altered data, seeds, epochs, batch size or loss.
- No Monitor prediction, no prediction freeze, no `evaluate`, no label read.
- No result was invented to fill the pre-registered tables.

The minimal repair and the additional gate it needs are written down (not applied) in
`analysis/gpu_stage_halt.json` → `required_next_step`.

---

## 3. What remains true from the pre-GPU package

These were verified before the halt and are unaffected by it (they are implementation
and data-discipline facts, not performance evidence):

- Frozen assets: split `9fa96d49…`, labels `0260a390…`, raw `8416837e…`, row manifest
  `2c638e48…`, spec `a65c67f9…`, norm `084c129c…`, caches `907509ae…`/`2abd1aef…`,
  content fingerprints `0a73407f…`/`db48a9c2…`; all matched locally **and** after
  deterministic rebuild on the remote checkout.
- Fit-only target standardisation: mean `2.157857142857143`, scale `1.2241982660328445`,
  Fit label-vector sha256 `2f71c3e2…`; Monitor/Reserved never used.
- A/B share an identical init state per seed; B's S2 block is exactly zero at LocalMLP
  input columns `[320:520)` in train and eval; A receives non-zero S2 gradients, B
  exactly zero (G2).
- GINE uses the real molecular adjacency with permutation invariance 5.96e-8 (G3);
  ECFP4 rows align and XGBoost refits bitwise (G6); the runtime guard blocks the CSV,
  official split and reserved cache (G8).
- Arm D's three label-blind Monitor predictions exist and are still **unscored**.

What is **not** established: that the trainer is correct on the full Fit set. The
gates' `READY_FOR_GPU_AUTHORIZATION` verdict is therefore too strong; it should have
required a full-Fit first-epoch batch-integration gate.

---

## 4. The seven required questions

All seven are **NOT ANSWERED — GPU stage halted**, not "deferred pending budget":

| # | question | status | evidence |
|---|---|---|---|
| 1 | Did scratch LCRC-A learn the task? | **no data** | 0/9 trainings completed |
| 2 | Is the S2 increment over B stable? | **no data** | no A/B predictions |
| 3 | Is A competitive with GINE and ECFP4? | **no data** | no A/B/C predictions; D unscored |
| 4 | Seed / molecule sensitivity? | **no data** | no per-seed results |
| 5 | Do `contrib` and `rho` both contribute? | **no data** (implementation only, G2) | no A prediction file |
| 6 | Train-vs-Monitor gap? | **no data** | no Fit RMSE from a completed run, no Monitor score |
| 7 | CASE_1 / CASE_2 / CASE_3? | **not assignable** | decision rule requires Monitor predictions |

The required summary numbers (RMSE/MAE for A/B/C/D, per-seed values, bootstrap
intervals, tail shares, `contrib`/`rho` scales, case label) **do not exist** and are not
estimated here.

---

## 5. Budget accounting

| item | GPU-seconds | note |
|---|---:|---|
| `lcrc-timing-a` | 17 | calibration only, 13 steps |
| `lcrc-formal-A-s0` | 16 | failed at first batch |
| `lcrc-formal-A-s1` | 17 | failed at first batch |
| three `res-2` submissions | 0 | cancelled while pending |
| **total** | **50 s = 0.0139 h** | cap 1.5 h, remaining ≈ 1.486 h |

No GPU-hour overrun occurred. The frozen estimate (0.734 h) was plausible: measured
0.01485 s/update implies ≈102 s per run and ≈0.34 h for nine runs plus predictions.

---

## 6. Deviations and limitations

1. **This round has no scientific result.** The pre-registered questions are unanswered.
2. **The halt is a code defect, not a data or budget problem.** Frozen data, splits and
   labels were hash-verified; the GPU budget was sufficient.
3. **Gates were incomplete for the formal path.** G5 used a 256-row prefix, so the
   full-Fit index range and label pairing were never exercised. This is disclosed as a
   gate-design failure, not as an excuse.
4. **Execution host changed from `res-2` to `res`.** Planned runs on `res-2` never
   started (cluster GPUs saturated by unrelated unlimited-time jobs) and were cancelled
   at 0 GPU-seconds; the two attempted runs and the calibration ran on `res`
   (A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124). Both are A100 regimes, but
   the regime is recorded per run and no cross-regime comparison is made.
5. **The authorization file remains in place** (`authorized=true`, 9 runs, 1.5 h). It
   does not by itself make the round runnable: any rerun under the frozen hashes will
   fail identically. It is kept as the authorization record.
6. **One pre-GPU test now fails** (`test_cuda_training_requires_authorization`), by
   design: it asserts the absence of the authorization file. It was not edited.
7. **Predictions from 2026-10-28 exploratory runs are not used**, and the old recipe's
   best-epoch/valid selection is not part of this protocol.
8. **No early stopping, no seed selection, no threshold change** was applied — there was
   nothing to select.

---

## 7. What a future amended round must contain

Only a new, explicitly pre-registered amendment can continue this work:

1. Fix the Fit batching: collate by cache position and index the label by the position
   (or by `train_row_index[train_rows[i]]`), not by the global row id.
2. Add a full-Fit first-epoch gate: 19 batches over all 2352 rows, asserting no
   out-of-range index and that every `(molecule id, label row id)` pair equals the split
   order; re-run the soup/order checks after the fix.
3. Regenerate `analysis/protocol_freeze.json` (new hashes) and obtain a **new** explicit
   GPU authorization; the current authorization should not be reused for changed code.
4. Then run the nine frozen runs, the label-blind predictions, the prediction freeze and
   exactly one evaluation, under the pre-registered decision rules.

This report does not authorize any of that, and does not start SSL, AQSOL, extra
ablations or test-set evaluation.

---

## 8. Evidence index

| artifact | sha256 |
|---|---|
| `analysis/protocol_freeze.json` | `87ec475107121d7b48cf9ae1e2664bb27bba972c39a543ebf24ea7885dbed9f4` |
| `analysis/gpu_authorization.json` | `48cb9a28ff36ec6765bcbc05d72391f5018e123a1ae8cef11d89196e44a3f30f` |
| `analysis/cpu_gates.json` | `aa29fb09c0135f2946e6b0e5c8dd6bef7ed30a156e303e779fb0a3e3d0b79185` |
| `analysis/norm_target.json` | `50c5ad9da285b15233dc0f26704cfb5136ac4bbe5a97b4bf21734d8f1f9a0748` |
| `analysis/parameters.json` | `db1763d6dccfb123ce7cfced08126121ce8a550cd0741da90b8154c262df9388` |
| `analysis/budget_estimate.json` | `d7148f13b85f341f0dcb389e4a7232afce136491bedd016d46870d48855d0536` |
| `analysis/smoke.json` | `fef92f3d0badfbbc7af5aca8f8b22c8f7399aa88604d8cad86b9c4b5b420fd4a` |
| `analysis/predictions/D_r0.npz` | `e2a18b9be03ed9e1da96614e9337abf2da577e14079d1a43958d295f736696b2` |
| `analysis/predictions/D_r1.npz` | `d2b42ebcc4a024d7e0dafa12e2c7672eb2650797361ba19a4e1704dc2c9be6c1` |
| `analysis/predictions/D_r2.npz` | `af4831f7dada243cde31f098e23dcfda4edc02e4582929dbd36ced9e49f46b78` |
| `analysis/gpu_stage_halt.json` | `111275dcc2037f19526fcb2c041a3e2e2868955e66755c065d52301fad72b113` |

`analysis/protocol_freeze.json` hashes of all 12 code modules, 9 test files and the
config are unchanged; `PROTOCOL.md` was not edited after the freeze.

### Remote run logs (reproducible with `rr logs <experiment>`)

```bash
rr logs lcrc-preflight
rr logs lcrc-timing-a
rr logs lcrc-formal-A-s0 --stderr
rr logs lcrc-formal-A-s1 --stderr
rr status lcrc-formal-A-s0
```
