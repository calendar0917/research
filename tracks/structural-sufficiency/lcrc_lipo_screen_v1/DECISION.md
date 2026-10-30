# DECISION — LCRC-A Lipophilicity screen v1

- **Date:** 2026-10-30
- **Round status:** `AMENDMENT_CPU_VERIFIED` / `READY_FOR_REAUTHORIZATION`
- **Scientific decision:** **DEFERRED — no verdict on any hypothesis.**
- **GPU:** 0 GPU-seconds under the amended freeze; 0.0139 A100 GPU-hours were used by
  the halted v1 stage. No new authorization exists.
- **Monitor/Reserved/official labels:** not read, not scored.

## 1. Decisions taken

1. **The v1 GPU stage stays halted.** Two runs failed at the first batch on a real code
   defect (global dataset row ids passed to a cache-position collator). That record is
   preserved (`analysis/gpu_stage_halt.json`, commit `e09c627`); no v1 job was resumed.
2. **The index defect is fixed as a recorded implementation amendment**, with no change
   to the scientific design: `AMENDMENT-001-fit-batch-index.md`, new gate G9, amended
   freeze `7fffba42…`, commit `131937b` plus the freeze commit.
3. **CPU re-verification is complete**: 10/10 gates PASS, 59 tests pass, new smoke on the
   real training path, full-Fit alignment proven for seeds 0/1/2 against an independent
   label map, and the pre-amendment mapping is shown to fail the same gate.
4. **No new GPU authorization is created here.** The v1 authorization
   (`48cb9a28…`) remains as a historical execution record and is refused against the
   amended freeze. The next formal stage requires an explicit new researcher
   authorization pinned to freeze `7fffba42…`.

No scientific decision is taken: there is no RMSE, no seed table, no paired comparison
and no `CASE_1` / `CASE_2` / `CASE_3`. The pre-registered interpretation matrix
(PROTOCOL §6) is unchanged and still applicable only to a completed formal run.

## 2. What the amendment did and did not change

**Changed (engineering only):**

| item | change |
|---|---|
| Fit batch index semantics | `shuffle_pos → cache_pos` for inputs and targets; `global_row_id` only for identity/provenance |
| new interface | `FitBatchPlan`, `plan_fit_batch`, `assemble_fit_batch`, `fit_epoch_plans`, `fit_batch_fingerprints` |
| trainer | uses the interface; records per-epoch alignment fingerprints; no `train_row_index` |
| new gate | G9 `full_fit_batch_alignment` (real 2352 rows × 19 batches × seeds 0/1/2) |
| G4 | state-independent authorization probes; reports whether the real file matches the live freeze |
| tests | new `tests/test_fit_alignment.py`; authorization test rewritten to be state-independent |
| status | `BLOCKED`/halt → `AMENDMENT_CPU_VERIFIED` → `READY_FOR_REAUTHORIZATION` |

**Unchanged:** arms, models, seeds, split, FeatureSpec, caches, Fit-only
standardisation, optimizer/LR/schedule/weight decay/clip/dropout/loss, 360 epochs,
batch 128, 6840 updates, last-5 soup, no early stopping or seed selection, RMSE/MAE,
bootstrap settings, materiality threshold, decision rules, prediction registry.

The amended freeze keeps the v1 `recipe`, `decision_constants`, expected artifact hashes
and 12-key registry identical; only `code_hashes`, the amendment block and the new gate
differ.

## 3. Evidence

| evidence | result |
|---|---|
| `analysis/cpu_gates.json` | G0–G9 all PASS; final `READY_FOR_REAUTHORIZATION`; ≈10 s |
| G9 seeds 0/1/2 | 19 batches, 48-molecule tail, 2352 positions each once, 1162 ids > 2351, old path 1162 out-of-range / 1185 mislabelled, gate path = `batch_order_hash` |
| G9 executed path | real 1-epoch 256-row smoke fingerprints equal the gate recomputation |
| `tests/` | 59 passed |
| `analysis/smoke.json` | A 151.104→0.932, B 113.537→0.825, C 423.958→0.827; finite losses/grads; 100% per-epoch coverage |
| old smoke | preserved byte-identical as `smoke_invalidated_pre_amendment.json`, marked invalid |
| authorization | v1 file rejected against the amended freeze (`does not match the current protocol freeze`) |

## 4. Budget

- Frozen internal hard cap: **1.5 A100 GPU-hours**.
- Used before the amendment: **0.0139 h** (three jobs on `res`; three pending `res-2`
  submissions cancelled before start, 0 GPU-s).
- Used under the amended freeze: **0.0 h**.
- Remaining: **≈1.486 h**. No GPU job was launched and no budget was spent by this
  amendment.

## 5. Authorization contract going forward

A formal rerun requires, in this order:

1. explicit new authorization pinned to `analysis/protocol_freeze.json`
   sha256 `7fffba423df449fae09688f29e1d4d33cdd5df2e51c4a15ae5986031470818b8`;
2. a clean committed worktree, `rr deploy`, and the nine frozen runs (A/B/C × s0/s1/s2)
   under the unchanged recipe;
3. nine label-blind predictions plus `D_r0..D_r2`, then `freeze-predictions`;
4. exactly one `evaluate` after the freeze, with the frozen metrics and decision rules.

The v1 authorization must not be reused, edited in place or treated as covering the
amended code. No `--force`-style bypass exists.

## 6. Invalidation conditions (unchanged, plus amendment-specific)

The round remains without a performance claim because 0/9 formal runs exist. In addition:

- editing the recipe, data, seeds, thresholds or models under the name of "index fix"
  would invalidate the amendment — nothing of the sort was done;
- the old smoke must not be cited as training evidence (it is marked invalid);
- the old GPU failure and its record must not be deleted or overwritten;
- a formal run under the amended freeze without a new matching authorization is a
  protocol violation.

Not triggered: Monitor/Reserved/official label access, best-seed or checkpoint
selection, metric substitution, hash bypass.

## 7. Next step gate

Proceed **only** to the reauthorization decision. Do not start SSL, AQSOL or other
tasks, extra ablations, Reserved evaluation or official valid/test evaluation. Do not
retry the failed v1 jobs. The round is engineering-ready, not scientifically decided.
