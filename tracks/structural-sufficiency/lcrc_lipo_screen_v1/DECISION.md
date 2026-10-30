# DECISION — LCRC-A Lipophilicity screen v1

- **Date:** 2026-10-30
- **Round status:** `HALTED_IMPLEMENTATION_DEFECT`
- **Scientific decision:** **NO VERDICT.** No hypothesis was tested; no Monitor label
  was read.
- **GPU:** authorized (1.5 h cap); **0.0139 A100 GPU-hours used**; two formal runs
  attempted and failed at the first batch; seven runs never started.
- **Monitor/Reserved/official labels:** not read, not scored.

## 1. Decision

The formal GPU stage of the frozen `lcrc-lipo-screen-v1` protocol was attempted under
the researcher's authorization and **halted at the first optimizer step** because the
frozen trainer passes global Fit row ids into a collator that indexes cache positions.
Every batch of epoch 1 contains at least one out-of-range index
(1162/2352 slots, 19/19 batches; the first batch alone has 66/128), and the in-range
slots pair each input molecule with another molecule's label (1188/1190). All nine
planned runs use the same loop and would fail identically.

The procedural decision is therefore:

> **Stop. Do not patch the frozen code in place. Preserve the frozen revision and the
> failed-run evidence. Report the defect and require a recorded amendment (fix + new
> full-Fit gate + regenerated freeze + new authorization) before any formal run.**

No scientific decision is taken. There is no RMSE, no seed table, no paired comparison,
and no `CASE_1` / `CASE_2` / `CASE_3`. Assigning one would be fabrication.

The pre-registered post-GPU interpretation matrix (PROTOCOL §6) remains the rule for a
future successful run; it is **not applicable** now.

## 2. What was and was not done

| item | status |
|---|---|
| Read AGENT / PROTOCOL / DECISION / README / freeze record | done |
| Verify frozen hashes locally and on the remote checkout | done, all match |
| Check for pre-existing formal runs | none for this protocol (2026-10-28 runs use a different loop) |
| Create `analysis/gpu_authorization.json` | done, committed (`48cb9a28…`) |
| Deploy with `rr` (res-2, then res) | done; no manual SSH launch |
| Nine neural trainings | **0/9 completed** (A_s0, A_s1 failed; 7 not started) |
| Nine label-blind Monitor predictions | **none produced** |
| Verify `D_r0..D_r2` | done: 504 frozen rows, `monitor_labels_read=false`, unscored |
| `freeze-predictions` | **not run** (only 3/12 predictions exist) |
| `evaluate` / Monitor labels | **not run / not opened** |
| CASE classification | **not assignable** |

Budget: 50 s GPU (0.0139 h) of the 1.5 h cap. The budget was never the limiting factor;
the measured 0.01485 s/update implies ≈0.34 h for the nine runs.

## 3. Root cause and evidence

Full record: `analysis/gpu_stage_halt.json`.

- Traceback: `screen_training.py:165` → `screen_data.py:133` → `lcrc_data.py:658`
  `ValueError: row index outside cache`.
- `splits_phase1a["fit"]` = global ids in `[0, 4199]`; cache has 2352 positions.
  `train_one` converts position → global id and feeds that id to `collate()` as a
  position, while looking the label up by the same global id.
- CPU gates missed it because G5 smoke used `fit_limit=256` and G4 used 128-row prefixes
  whose ids are < 2352; no gate exercised the full-Fit first epoch or asserted
  `(molecule id, label row id)` pairing.

No frozen file was modified. The proposed one-line-class fix is written down in the halt
record but **was not applied**, because editing frozen code changes the hashes and
requires a new freeze and a new authorization; the execution order forbids patching
frozen configuration to rescue an experiment.

## 4. Authorization contract — consumed by this halt

The authorization file remains in the repository as an execution record
(`authorized=true`, nine runs, `max_gpu_hours=1.5`, freeze sha `87ec4751…`). It must
**not** be reused to run changed code. A continuation requires:

1. a recorded amendment that fixes the Fit batching;
2. a new gate that runs the full 2352-row Fit epoch (19 batches) and asserts index range
   and `(molecule id, label) `alignment, in addition to the existing soup/order checks;
3. regeneration of `analysis/protocol_freeze.json` with new code hashes;
4. a new explicit GPU authorization pinned to the new freeze sha.

Until then, `run_screen.py train` will keep refusing or failing; no seed, arm or epoch
may be changed to work around it.

## 5. Budget rules (unchanged)

- Frozen internal hard cap: **1.5 A100 GPU-hours**; used **0.0139 h**.
- Researcher-suggested external cap: 4.0 h (a cap, not a target).
- Every launch, including failed/cancelled ones, is charged and recorded:
  `lcrc-timing-a` 17 s, `lcrc-formal-A-s0` 16 s, `lcrc-formal-A-s1` 17 s, three
  `res-2` submissions cancelled before start (0 s).
- Arm D remains local CPU, 0 GPU-hours.

## 6. Invalidation conditions

This round is reported as **halted and incomplete**, which is itself an invalidation of
any performance claim. Specifically:

- 0 of 9 required trainings exist, and there is no recorded acceptable failure that
  permits dropping one (all nine are blocked by the same defect);
- no prediction manifest exists, so no Monitor label may be read;
- no metric was computed, so no decision threshold was applied or changed;
- no code/config hash differs from the freeze; `PROTOCOL.md` is unchanged.

Not triggered: Monitor/Reserved/official labels read early, best-seed or checkpoint
selection, metric substitution.

## 7. Next round gate

A future round is justified **only** by a corrected, separately pre-registered package
with the full-Fit alignment gate. Nothing else is authorized here:

- no SSL, no AQSOL or other task, no extra ablations, no architecture search;
- no Reserved labels, no official valid/test;
- no reuse of the 2026-10-28 exploratory runs (different recipe with valid-based best
  epoch selection);
- no "quick fix" rerun under the current frozen hashes.

This document ends the round. Stop here.
