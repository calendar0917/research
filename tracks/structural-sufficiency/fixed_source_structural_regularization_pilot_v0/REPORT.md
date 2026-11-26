# FSRP v0 — FIXED1 structural-unit learning-constraint short validation

Round: `fixed-source-structural-regularization-pilot-v0`
Seed: 0 · PCBA official Train/Valid · epoch 25 → 33 continuation · Test not accessed.

**Verdict: `NO_CLEAR_GAIN`**
Matched-continuation Valid AP improves materially (`STRUCT_DROP - BASE_CONT` =
**+0.00419**, paired task bootstrap 95% CI [0.00174, 0.00656]), but the gain over
the **frozen epoch-25 common start** is only **+0.00172** (95% CI
[-0.00114, 0.00455]), below the pre-registered +0.003 practical threshold.
Endpoint, last-3 and high-support task AP are all better than `BASE_CONT`.

---

## 1. Question and design

Does randomly masking real structural computation units **during training only**
improve standard graph-level prediction, while FIXED1 inference computation stays
complete?

| arm | training computation |
|---|---|
| `BASE_CONT` | original FIXED1 continuation, no constraint |
| `STRUCT_DROP` | FIXED1 + structural-unit message dropout, `q = 0.10` |

`STRUCT_DROP` draws one Bernoulli mask per **physical undirected bond per training
batch** with an independent `torch.Generator` (`mask_seed = 20261126`), expands it
as `cat([keep, keep])` in the exact `directed_bonds()` order, uses the same
`keep_directed` tensor in both FIXED1 blocks, and scales kept messages by
`1/(1-q)`. Eval mode delegates to the inherited FIXED1 code path (no mask, no
scaling). No new parameter, loss, head, degree/descriptor change or `post_local`
change. The model's training RNG is not advanced by the mask draw.

Both arms restore the *same* FIXED1 seed-0 epoch-25 state: model, AdamW moments,
RNG, global step 68,450, the original 40-epoch order plan and the original
40-epoch warmup+cosine LR schedule (not rescaled). The continuation horizon
25 → 33 is allowed only in this round's isolated trainer; no historical file is
modified.

## 2. Necessary checks (all passed)

`analysis/checks.json` — C1–C8 and T1 all PASS.

| check | result |
|---|---|
| C1 same weights / optimizer / RNG / LR / plan | YES; state fp and optimizer fp identical; LR at restore 1.0358628017e-4, first continued step 1.0358229138e-4 |
| C2 initial eval logits / Valid AP equal | ΔAP = 1.36e-6 (CUDA atomic-add re-score noise) |
| C3 q=0 forward + gradients = FIXED1 | **CPU exact 0** for logits, loss and all 1,128,468 gradient entries; GPU max logit Δ 9.5e-6 (atomic order only) |
| C4 same physical mask, both directions, both blocks | YES (identical tensor object/data pointer; 3,617 physical bonds → 7,234 directed edges) |
| C5 realised drop fraction ≈ 10% | 0.099324 over 40 batches (3σ binomial gate); exact scatter reference Δ = 0 |
| C6 eval has no mask, equals FIXED1 | mask draws unchanged; CPU logit Δ = 0; GPU Δ 9.5e-6 |
| C7 parameters | both arms 1,128,468 |
| C8 complete resume + Test isolation | epoch 25 / step 68,450 / optimizer + RNG + plan restored; 0 finite Test labels visible; Test never scored or loaded |

Base checkpoint actually used:
`direct_structural_computation_recurrence_v0/runs/FIXED1_seed0/checkpoint_last.pt`
(sha256 `520e32a1…b6fa9`, `last_epoch_resumable`, epoch 25, step 68,450,
recorded Valid AP 0.2758455214).

## 3. Curves (complete Valid every epoch; complete Train at 25 / 29 / 33)

| epoch | BASE Valid AP | STRUCT Valid AP | Δ AP | BASE Valid BCE | STRUCT Valid BCE | BASE Train AP | STRUCT Train AP | mask dropped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.275848 | 0.275845 | -0.000002 | 0.0463769 | 0.0463769 | 0.599570 | 0.599579 | – |
| 26 | 0.272653 | 0.275235 | +0.002582 | 0.0465566 | 0.0463584 | – | – | 0.10003 |
| 27 | 0.273336 | 0.276506 | +0.003170 | 0.0465366 | 0.0462254 | – | – | 0.10014 |
| 28 | 0.272956 | 0.276348 | +0.003392 | 0.0467379 | 0.0463812 | – | – | 0.10005 |
| 29 | 0.272559 | 0.276227 | +0.003668 | 0.0468569 | 0.0463878 | 0.626365 | 0.618362 | 0.09997 |
| 30 | 0.273208 | 0.275845 | +0.002637 | 0.0470419 | 0.0465215 | – | – | 0.09978 |
| 31 | 0.272492 | 0.276808 | +0.004316 | 0.0470881 | 0.0465087 | – | – | 0.10011 |
| 32 | 0.273379 | **0.277578** | +0.004198 | 0.0472915 | 0.0467337 | – | – | 0.09991 |
| 33 | 0.272753 | 0.276899 | +0.004146 | 0.0473797 | 0.0467440 | 0.642637 | 0.632375 | 0.09984 |

`BASE_CONT` continuation decayed below the epoch-25 start from epoch 26 onward.
`STRUCT_DROP` stayed within ~0.001 of the start and exceeded it at six of the
eight epochs (27, 28, 29, 31, 32, 33). The mask fraction is 0.09998 ± 0.00011
across the 8 epochs.

## 4. Main table

| metric | BASE_CONT | STRUCT_DROP | Δ (STRUCT − BASE) |
|---|---:|---:|---:|
| best epoch | 32 | 32 | – |
| **best Valid AP** | 0.2733803 | **0.2775724** | **+0.0041921** |
| best Valid BCE | 0.0472915 | 0.0467337 | −0.0005578 |
| endpoint (33) Valid AP | 0.2727529 | 0.2768986 | +0.0041457 |
| last-3 mean Valid AP (31–33) | 0.2728747 | 0.2770948 | +0.0042202 |
| Train AP @25 / 29 / 33 | 0.599570 / 0.626365 / 0.642637 | 0.599579 / 0.618362 / 0.632375 | +0.000008 / −0.008004 / −0.010263 |
| Train BCE @25 / 29 / 33 | 0.0242222 / 0.0232680 / 0.0226316 | 0.0242222 / 0.0235678 / 0.0230347 | 0.0000000 / +0.0002999 / +0.0004031 |
| best-checkpoint Train AP / BCE | 0.639658 / 0.0227787 | 0.630122 / 0.0231460 | −0.009536 / +0.000367 |
| high-support Valid AP (best) | 0.297022 | 0.301422 | +0.004400 |
| high-support Valid AP (endpoint) | 0.297226 | 0.300830 | +0.003604 |
| Train–Valid gap, same best ckpt, 127 common tasks | 0.366361 | 0.352634 | −0.013727 |
| parameters | 1,128,468 | 1,128,468 | 0 |
| GPU hours (formal run) | 0.4149 | 0.3495 | – |

High-support = Valid positive ≥ 20 **and** negative ≥ 20 → 107 tasks. The gap is
the mean per-task AP difference over the same 127 commonly scorable tasks, not a
substitution for the official macro AP.

## 5. Paired task comparison and sensitivity

Paired bootstrap over the 127 tasks (B = 10,000, seed 20261126).

| comparison | mean Δ | 95% CI | P(Δ>0) | tasks + / − | drop-top-3 mean Δ |
|---|---:|---|---:|---:|---:|
| **best, STRUCT − BASE** | **+0.0041921** | [0.0017404, 0.0065603] | 0.9995 | 97 / 30 | +0.0041546 |
| endpoint 33, STRUCT − BASE | +0.0041457 | [0.0023859, 0.0061667] | 1.0000 | 94 / 33 | +0.0029614 |
| last-3 mean, STRUCT − BASE | +0.0042202 | [0.0025190, 0.0060671] | 1.0000 | 96 / 31 | +0.0038781 |
| best, STRUCT vs epoch-25 start | +0.0017247 | [−0.0011377, 0.0045469] | 0.8792 | 75 / 52 | +0.0013429 |
| best, BASE vs epoch-25 start | −0.0024674 | [−0.0057045, 0.0006995] | 0.0632 | 56 / 71 | −0.0020079 |

Top-3 removed for the primary comparison: tasks 124 (−0.0913), 28 (+0.0556),
61 (+0.0530); the remaining 124 tasks still average +0.004155. The matched-arm
advantage is therefore not carried by three outlier tasks.

## 6. Fit versus generalization

The two directions clearly diverged:

* **Train fit fell**: STRUCT Train AP is lower at every measured continuation
  point (29: −0.00800; 33: −0.01026; best checkpoint: −0.00954) and Train BCE is
  higher (+0.00030 / +0.00040 / +0.00037).
* **Valid prediction improved relative to the matched continuation**: best Valid
  AP +0.00419, Valid BCE −0.00056, endpoint +0.00415, last-3 +0.00422,
  high-support +0.00440.
* The same-checkpoint Train–Valid gap narrowed from 0.3664 to 0.3526 over the
  same 127 tasks — the reduction is driven by the lower Train AP, not by a Valid
  decline (Valid AP rose).

So this is **not** `FIT_REDUCTION_WITHOUT_GAIN` in the literal matched-arm sense:
there is a real matched-continuation Valid gain. It is also **not**
`STRUCTURAL_REGULARIZATION_PROMISING`, because the gain over the epoch-25 frozen
start misses the pre-registered +0.003 criterion and its CI includes zero.
`BASE_CONT`'s own 8-epoch continuation lost −0.00247 AP against the start, so
most of the matched-arm advantage is avoidance of continuation decay, not an
absolute improvement over the starting model.

## 7. Answers to the three report questions

1. **Did the regularization bring a gain beyond normal continuation?**
   Yes against the matched continuation: +0.00419 best (CI excludes 0),
   +0.00415 endpoint, +0.00422 last-3, +0.00440 high-support, 97/30 tasks
   improved/degraded, robust to dropping the three largest |Δ| tasks. But no
   against the frozen epoch-25 start at the pre-registered practical threshold:
   +0.00172 with CI [−0.00114, +0.00455].
2. **Did Train fit and Valid prediction move in different directions?**
   Yes. Train AP fell by ~0.008–0.010 and Train BCE rose, while Valid AP rose by
   ~0.004 relative to `BASE_CONT` and Valid BCE fell. The Train–Valid gap
   narrowed by 0.0137 on common tasks.
3. **Is structural-level learning constraint worth continuing?**
   Under the pre-registered rule, **no** (`NO_CLEAR_GAIN`): the absolute gain over
   the common start is not established at the required +0.003, the CI crosses
   zero, and this is a single seed / single q / 8-epoch pilot without a plain
   message-dropout control. Per the protocol this line stops here: no increase in
   q, no longer training, no extra seeds or arms, and no next round is started.
   The matched-continuation signal is consistent enough to be recorded as a
   hypothesis, not a result.

## 8. Execution deviations

1. **Base checkpoint variant.** `checkpoint_last.pt` was available and used
   (preferred). The local `runs/` tree is not tracked; the exact file was
   recovered from the earlier pulled run and verified byte-for-byte
   (sha256 `520e32a1…b6fa9`, epoch 25, step 68,450, complete resume state).
   `checkpoint_epoch25.pt` / `checkpoint_best.pt` were not needed.
2. **Horizon.** The historical trainer's `HORIZONS` tuple rejects 33; this
   round's isolated trainer continues 25 → 33 while keeping the original
   40-epoch schedule and its absolute global-step indexing. No schedule
   rescaling, no restart.
3. **Deployment.** The remote `res` checkout carried untracked result files from
   the preceding fsdp-v0 round, which collided with that round's later local
   commits during a fast-forward deploy. A temporary branch `fsrp-v0`, based on
   the remote's current commit `5528f83b`, was used to deploy only the new round
   files (executed commit `90aeaea456d8`). No remote tracked file was edited and
   no history was rewritten. The branch is merged back into `main` when the
   results are committed.
4. **Re-scoring noise.** The epoch-25 baseline re-scored on GPU is
   0.2758477 (`BASE_CONT`) / 0.2758454 (`STRUCT_DROP`) versus the recorded
   0.2758455; differences of ~2e-6 are the documented CUDA atomic-add
   non-determinism. The C3/C6 exactness gates are therefore evaluated on CPU
   (exact 0) and separately reported for GPU (≤1e-5).
5. **GPU contention.** GPU 0 was slower (~150 s/epoch) than GPU 1 (~122 s/epoch)
   due to co-tenant load; both arms still ran in parallel and completed. The
   timing preflight projected 1,394 s per arm; actual formal wall was 1,494 s
   (BASE) and 1,258 s (STRUCT).

No Test file or Test label was loaded, no Test metric was computed, and no
results were selected using Test. Only seed 0, only q = 0.10 was run.

## 9. Resource consumption

* GPU hours (formal, both arms): 0.4149 + 0.3495 = **0.7644**; plus checks/timing
  (~0.01) and failed check attempts (~0.02) → **≈0.79 GPU-hour**, within the
  0.9 target and 1.1 hard limit.
* Wall clock: implementation + local unit tests ≈7 min; deploy and checks
  (including two failed check attempts) ≈7 min; parallel formal training
  24.9 min (longest arm); analysis/report ≈11 min → **≈54 min**, inside the
  75-min hard limit.
* Both arms were run in parallel on the two idle A100-40GB cards of `res`
  (`res-gpu0`, `res-gpu1`); no queue wait, no preemption.

## 10. Artifacts

* Protocol: `PROTOCOL.md`; fixed config `configs/round.json`.
* Checks/timing: `analysis/checks.json`, `analysis/preflight_timing.json`.
* Curves and tables: `analysis/epoch_curves.csv`, `analysis/main_table.csv`,
  `analysis/report_tables.md`.
* Paired comparison and summary: `analysis/paired_comparisons.json`,
  `analysis/summary.json`.
* Checkpoint hashes: `analysis/checkpoint_manifest.json`,
  `results/manifest.json`; run ledger `results/ledger.csv`.
* Lightweight run records: `results/runs/{BASE_CONT,STRUCT_DROP}_seed0/`
  (full checkpoints remain in the gitignored `runs/` tree).
* Tests: `tests/test_struct_drop.py` (6 CPU unit tests, all pass).

No next round or follow-up experiment is started by this report.
