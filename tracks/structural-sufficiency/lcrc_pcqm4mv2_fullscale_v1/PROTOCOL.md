# PROTOCOL — PCQM4Mv2 full-Train W2/W3 standard experiment (v1)

Round id: `lcrc-pcqm4mv2-fullscale-v1`
Status: **frozen before the first full-cache molecule is parsed and before the
first formal training step.** §9 is an append-only amendment log; nothing else
may change after the first formal step.

Two questions, fixed by the execution instruction:

* **Q1 (capacity at the official Train size).** With the historical strict
  no-iterative-message-passing LCRC recipe, what Validation MAE does the
  official 3,378,606-molecule Train split give at width 2× and 3×?
* **Q2 (width contrast at full data).** Does the small W3-over-W2 advantage seen
  on the 1M subset (`−0.004436 eV` at epoch 40) survive, grow or disappear when
  the same recipe is trained on the full official Train split?

Exactly two formal runs, both from scratch:

| Run ID | Width | Parameters | Train molecules | Epochs | Seed |
|---|---:|---:|---:|---:|---:|
| `PCQM_W2_FULL` | 2× | 939,079 | 3,378,606 | 40 | 0 |
| `PCQM_W3_FULL` | 3× | 2,038,167 | 3,378,606 | 40 | 0 |

No W1, W4, DYNAMIC, extra seed, architecture change, loss search, batch-size
search or unplanned long run. One final Test-dev inference with the single
selected model.

## 1. Inherited, read-only ingredients

| item | source | reused as |
|---|---|---|
| frozen `lipo_v1` feature spec / graph / adapter | `lcrc_pcqm4mv2_transfer_pilot_v1/{code,analysis}` | unchanged |
| `pcqm_train1m` cache (1,000,000 rows) | same `cache/` | read-only **prefix control** for the rebuilt full cache |
| `pcqm_valid` cache (73,545 rows) | same `cache/` | read-only, unchanged |
| `pcqm_labels_trainvalid.npy`, `pcqm_monitor_positions.npy` | same `cache/` | read-only |
| feature + target normalisation (`lipo_v1`, Train-300K constants) | same `analysis/` | unchanged, identical for both widths |
| `pcqm_permutation.npy` (3,378,606 official Train row ids, seed 42) | same `cache/` | full Train cache order |
| model builder, init rule, `train_step`, official evaluator | `pq_model`, `pq_train`, `ew_model` | unchanged |
| **E1 vectorised collate** (selected in the previous round) | `lcrc_pcqm4mv2_efficiency_width_v1/code/ew_collate.py` | formal training path |

No data is re-downloaded, no historical file is modified, and the historical
1M W2/W3 checkpoints are never retrained or overwritten. E2/E3 are not retried.

## 2. Data identity

Use the installed `ogb` dataset and the official `get_idx_split()` exactly as
shipped. Record split sizes, row-id hashes and file SHA256 for `data.csv.gz`
and `split_dict.pt`. The known one-row difference between the public web page
total and the installed data version is recorded, not "fixed": no row is
dropped or added to force agreement.

* Official Train: 3,378,606 rows.
* Validation: 73,545 rows (read-only cache).
* Test-dev: 147,037 rows (accessed **only** in the final inference stage).
* Test-challenge: 147,432 rows (never accessed in this round).

Full Train cache order is `pcqm_permutation.npy` (the seed-42 permutation of
the official Train rows already used to build the 1M subset). Required checks:

1. it is a permutation of exactly the official Train row ids (no duplicate, no
   missing, set equality);
2. its first 1,000,000 entries equal `pcqm_subset_rows.npy` and the row ids of
   the historical `PCQM_W2_1M` / `PCQM_W3_1M` runs;
3. it has zero overlap with Validation, Test-dev and Test-challenge rows.

Both widths use the same row set and the same per-epoch batch-order rule
(`numpy.random.default_rng(0)`, one fresh permutation of the 3,378,606 cache
positions per epoch), with the 40-epoch permutation hashes recorded at start.

## 3. Feature cache

`SMILES → mb2_rdkit → MolecularGraph → lipo_v1 LCRC cache`, unchanged; no 3D,
no fingerprints, no descriptor additions, no UNK-rule change.

The full cache is a **reliable full rebuild** of all 3,378,606 molecules in the
frozen permutation order (the "rebuild" branch of the instruction, not the
prefix-injection optimisation). The existing 1M cache is used as an independent
**control**: after the merge, every field of the new cache's first 1M rows must
be bitwise equal to the historical 1M cache (pointer offsets, value arrays,
`gids`, per-row SMILES), and collated batches from the prefix must be identical.
If any check fails, the result is `BLOCKED_CACHE_PREFIX_MISMATCH` and training
does not start.

Build requirements: shard build in chunks (4,000 rows) with the frozen adapter;
record strict vs `unsanitized_fallback` parse counts, every fallback row and
every failure row. No silent row drop: a single failed row blocks the build.
The Validation cache is the historical read-only cache.

**Test-dev features are not built, read or inspected until both formal runs
have completed and a single model has been selected (§7).**

## 4. Normalisation

Feature normalisation (atom scalars + global features) and target mean/std are
the historical Train-300K constants loaded read-only from the parent round;
`z = (gap_eV − mean) / std` and the inverse are unchanged. Both widths use the
same constants. Full-Train statistics are **not** recomputed, so the comparison
with the 1M runs is not contaminated by a normalisation change.

## 5. Frozen training recipe (both runs)

Seed 0, batch size 128, 40 epochs, AdamW `(β1, β2)=(0.9,0.999)`, `eps=1e-8`,
weight decay `1e-3`, dropout `0.10`, peak LR `3e-4`, one-epoch linear warmup
`3e-5 → 3e-4`, per-step cosine to `1e-5` over the complete horizon, FP32 with
TF32 off, gradient clip norm `5.0`, L1 on the standardised target, fresh
initialisation, no early stopping, no extra data or pretraining.

Horizon: `ceil(3,378,606 / 128) = 26,396` steps/epoch × 40 =
**1,055,840 optimizer steps**. The scheduler is built for this horizon and the
1M horizon (312,520 steps) is never reused.

W2 and W3 differ only in width. Same data order, optimizer rule, evaluation
protocol and budget. After every epoch: full official Validation MAE (eV), the
fixed 20,000-row Train-monitor MAE (eV), mean Train L1(z), LR, step index,
times, throughput, memory, and best-checkpoint update (minimum Validation MAE,
ties resolved in favour of the earlier epoch). Each run writes
`checkpoint_best_valid.pt`, `checkpoint_epoch40.pt`, `checkpoint_last.pt`,
`train_curve.csv`, `run.json`, `rng_state.pt`.

## 6. Budget and resources

* Target total: **≤ 26 A100 GPU-hours**; hard ceiling **38 GPU-hours**, including
  preflight, both formal runs, final inference and the failure reserve.
* Revised per-run projection is recorded before launch from a measured
  throughput preflight on the real full cache. If only one complete run fits,
  finish W3, stop, and report W2 as missing; an incomplete run is never
  presented as a comparison.
* Preferred layout: `res` GPU0 = `PCQM_W2_FULL`, GPU1 = `PCQM_W3_FULL`; both
  GPUs are checked live before launch. If co-tenancy reduces total efficiency
  materially or threatens RAM/stability, the runs are serialised and the reason
  is recorded; no double-GPU speed-up is claimed without the measurement.
* GPU parallelism is preceded by per-GPU and simultaneous real-cache short
  benchmarks. Record molecules/s per GPU, CPU cores/utilisation, RSS, host RAM,
  GPU utilisation/memory and page-cache state.
* Epochs, batch size, LR, precision and evaluation frequency are never reduced
  to fit the budget. Safe epoch-boundary resume is allowed (full optimizer, RNG
  and scheduler state); a machine failure or exhausted budget is reported
  honestly with the actual progress.
* Disk/RAM headroom is checked before the build (~20 GB cache plus shards and a
  temporary merge copy).

## 7. Model selection and Test-dev inference

Only after both runs complete: select the single final model by lowest **best
Validation MAE**; tie → fewer parameters. Freeze model identity, checkpoint
SHA256, preprocessing code and normalisation.

Then build the official `test-dev` features from the official SMILES in the
original `split['test-dev']` row order, run one inference pass, restore eV with
the frozen inverse normalisation, and write
`y_pred_pcqm4m-v2_test-dev.npz` through
`PCQM4Mv2Evaluator.save_test_submission(input_dict={"y_pred": ...},
dir_path=..., mode="test-dev")`.

No Test labels are read or inferred; no weights/normalisation/selection change.
Verify independently: shape `(147037,)`, finite, dtype, SHA256, row-order hash,
and row-id alignment. Report preprocessing, inference and total wall time on a
single CPU+GPU machine and confirm the official 4-hour inference limit.
**No local Test-dev MAE is computed or reported.**

Submission boundary: prepare the reproducible code, README, short technical
report and metadata; check the current official OGB-LSC submission
instructions. **Nothing is uploaded and no leaderboard submission is made
without explicit further user authorisation.** If not submitted, mark
`SUBMISSION_READY_NOT_SCORED`; if submitted, mark the Test MAE
`PENDING_OFFICIAL_EVALUATION` until the official score is public.

## 8. Gates, decision rules, out of scope

Correctness gates before formal training (both widths where applicable):
analytic == built parameter counts; E1 == frozen collation field-by-field on
real full-cache batches; full cache row alignment; 1M prefix bitwise equality;
no Validation/Test row in any optimizer batch; reproducible initialisation and
RNG; one atom encoding + one bond encoding + one S1/S2 local call and no
iterative message passing in the forward; finite loss/gradient/update; full
checkpoint save→restore→continue; official MAE == independent NumPy MAE; frozen
normalisation constants; SHA256 of code and data identity.

Decision rules (resource allocation, not significance tests):

* FULL materially better than the corresponding 1M result → data scale still
  buys accuracy; report the absolute change.
* W3 still better than W2 at FULL → report the absolute change, late-epoch
  consistency and the parameter cost; no "best width" claim from one seed.
* W3 ≈ W2 at FULL → keep the smaller W2 as the efficiency reference.
* FULL still clearly above 0.11–0.12 eV → acknowledge the distance to strong 2D
  graph models; no automatic new architecture search.

Out of scope: W1/W4, dynamic variants, extra seeds, ensembles, architecture,
loss or batch-size search, additional epochs, test-challenge, hidden-label
guessing, external data, any leaderboard upload without authorisation.

## 9. Amendments (append-only)

- 2026-11-11 — protocol written and frozen before any full-scale parse or
  training step. Ingredients, full-row plan, rebuild policy, recipe, horizon,
  budget, gates, model-selection and inference rules come from the execution
  instruction. No model step has been taken at this commit.
