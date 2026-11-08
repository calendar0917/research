# REPORT — LCRC-S3-SP v1: root-conditioned selective three-hop path residual

Round id: `lcrc-pcba-s3-selective-v1`
Protocol: `PROTOCOL.md` (frozen before the first training step, commit `6c170e64`)
Training commit: `79bf2304` · Freeze/score commit: `5ec90109`
Test role: **`EXPOSED_TEST_PERFORMANCE_BENCHMARK`** — the MolPCBA official Test
split is *not* a blind generalisation estimate.
Decision: **`NO_CLEAR_GAIN`**

---

## 1. Answer

| model | parameters | best Valid epoch | **Best Valid AP** | AP@40 | **Test AP** | Test BCE | GPU-h | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 (2026-11-06) | 994,324 | 24 | 0.265273316 | 0.261866918 | 0.253616479 | 0.048584 | 1.3949 | 124.9 |
| S3 (2026-11-07) | 1,018,532 | 29 | 0.271489040 | 0.267721100 | 0.250633382 | 0.048756 | 1.8479 | 165.6 |
| **S3-SP (this round)** | **1,036,420** | **29** | **0.266722209** | 0.265085321 | **0.252549103** | 0.049162 | 1.9018 | 170.4 |

| contrast | Validation | Test (exposed) |
|---|---:|---:|
| SP − S3 | **−0.004766831** | **+0.001915721** |
| SP − W2 | +0.001448893 | −0.001067376 |
| S3 − W2 | +0.006215774 | −0.002983097 |

Three facts, in order of importance:

1. **The selective residual did not improve Validation.** Best Valid AP
   `0.2667222` vs S3 `0.2714890`: 0.0048 *worse*, i.e. −1.76 % relative. It is
   not a tuning accident: the two curves track each other within ±0.0069 in every
   epoch (SP ahead in 13 of 40, by at most +0.0057 at epoch 14), but the second
   half is consistently negative (mean −0.00292 over epochs 21–40 versus +0.00036
   over epochs 1–20), so the final gap is a trend rather than one bad epoch.
2. **On the exposed Test split it is nominally ahead of S3 by +0.001916**, which
   is below the pre-registered `+0.002` threshold and is spread over 62 up /
   65 down per-task flips (see §6). By the frozen decision table this is
   `NO_CLEAR_GAIN`, not a gain.
3. **The S3 result itself does not transfer to Test.** S3 is the best of the
   three on Validation and the *worst* of the three on Test
   (`0.250633` vs W2 `0.253616`). Its Valid→Test drop is −0.02086 versus
   −0.01166 for W2. Any "explicit three-hop paths help" claim built on
   Validation AP alone does not survive this Test measurement.

The engineering work target (`Valid ≥ 0.2745` **and** `Test ≥ TestAP(S3)+0.003`)
fails on both clauses: `0.2667 < 0.2745` and `+0.00192 < +0.003`.

## 2. What was added, and evidence that it is real

```
S3-SP = frozen W2 + frozen S3 + root-conditioned 2-head attention over the same
        S3 paths, applied as a zero-initialised residual on the S3 uniform mean
```

* query: `Linear(192→48)` on the root's **static** atom embedding `a`, reshaped
  to 2 heads × 24;
* key: `Linear(48→48)` on the S3 path embedding `z`, 2 heads × 24; value = `z`
  (no extra projection);
* score `= q·k/√24`, **per-root** softmax over the root's own paths, then
  `attn = Σ_p w_p z_p` per root and head;
* residual `= Linear(48→128)(attn − mean_z)` with `out` initialised to all
  zeros, forced to 0 for roots with no path;
* added after the unchanged S3 residual: `h = drop(h_raw + r_s3 + r_sel)`.

17,888 new parameters (q 9,264 / k 2,352 / out 6,272), 1.76 % of the S3 branch,
1,036,420 total (exactly `sum(p.numel())`).

Pre-training audits (`analysis/init_audit.json`, `analysis/architecture_audit.json`,
run on `res-gpu1`) — all pass, and they are not formalities:

| property | measured |
|---|---|
| W2 / S3 tensors bitwise equal to history | base `6a3df38f…`, S3 `a850ac85…` |
| epoch-0 logits vs W2 / vs S3 | max abs diff `2.9e-6` (tolerance `2e-5`, CUDA) |
| stage calls | atom 1 / bond 1 / local 1 / dropout 1 / S3-encode 1 / S3-out 1 / selective 1 |
| per-root weight sums | max error `3.6e-7` over 8.4 M root-heads |
| single-path root | `attn − mean = 0` exactly, residual 0 |
| empty root (`counts == 0`) | residual 0, finite output |
| atom renumbering | `|Δlogit| ≤ 2e-5` (float32 scatter order) |
| cross-molecule paths | 0 |
| query source | static `a` only; no `h`, no propagated state |
| gradient flow | step 1 `out.*`; by step 3 `q.*`, `k.*` finite and non-zero |
| iterative message passing | none (each operator called exactly once) |

The branch was **used**, not ignored: the selective term is 48.7 % of `‖h_raw‖`
after epoch 1 and 62.5 % at epoch 40 (mean 61.0 %), attention entropy ratio falls
from `0.292` to `0.171` of the uniform maximum, and the share of near-one-hot
roots rises from `2.6 %` to `17.1 %`. So the negative result is *not* "the
addition did nothing"; the model learned a strongly non-uniform per-root path
weighting and still lost ~0.005 Validation AP.

## 3. Training (one trajectory, frozen recipe)

| item | value |
|---|---|
| host / pool | `res` (`a100-2`) / `res-gpu1`, `CUDA_VISIBLE_DEVICES=1` |
| GPU | NVIDIA A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124 |
| run id | `sp-SP-20261108-131055-5deba912` |
| epochs / steps | 40 / 109,520 (`2,738 × 40`) |
| batch plan | `06f43729…b6f6`, prefix `1a24dfca…5f58` (equal to W2/S3) |
| AdamW | lr 3e-4 peak, 1-epoch warmup from 3e-5, per-step cosine to 1e-5, wd 1e-3, β (0.9, 0.999), ε 1e-8 |
| dropout / clip / precision | 0.10 / 5.0 / FP32 (14 clip events, no NaN/Inf) |
| loss | inherited masked BCE over observed entries |
| epoch-0 AP | `0.029062988` (W2 `0.029063332`, S3 `0.029063052`) |
| epoch 40 | train BCE `0.025763`, Valid BCE `0.046465`, Valid AP `0.265085` |
| train AP@40 | `0.595033` (S3 `0.590940`, W2 `0.581875`) — the train/Valid gap grew slightly |
| compute | `1.9018` GPU-h, `170.36` s/epoch mean, 17.22 steps/s, peak 239 MB alloc / 449 MB reserved |

`analysis/training_parity.json`: all 27 checks pass (recipe, plan, prefix, LR
endpoints, dropout, clip, precision, masked BCE, no early stopping, 40/40
evaluated epochs, no Test access, epoch-0 AP parity with W2 *and* S3, identical
Train+Valid cache identity).

Checkpoints: `runs/SP/checkpoint_best.pt` (epoch 29,
`f4633004…e10c8`), `checkpoint_epoch40.pt` (`f55b83eb…d2bf2`),
`checkpoint_last.pt` (`47f9a112…6f2e3`).

## 4. Selection (Validation only)

```
best_epoch = argmax(valid_AP[1..40]) = 29   (no tie)
```

`analysis/selected_checkpoint.json` was written *before* any Test label existed
on the scoring machine and declares `test_read=false`,
`selection_uses_test=false`. Batch plan, recipe and selection rule are identical
to the W2 and S3 rounds; the SP epoch happens to coincide with S3's.

## 5. Test evaluation (freeze → score, labels kept local)

1. `sp-test-freeze` (job `sp-test-freeze-20261108-150945-86303911`, 45.1 s GPU)
   restored the three Validation-selected checkpoints, replayed Validation AP
   (measured vs recorded: W2 `0.265273244/0.265273316`, S3
   `0.271486502/0.271489040`, SP `0.266722607/0.266722209`), predicted the
   43,793 official Test rows from the **label-free** Test structure cache and
   sealed everything (manifest seal `b0084029956963f5…`).
2. `sp-score` ran **locally**, where the official labels live
   (`data/ogb/ogbg_molpcba/raw/graph-label.csv.gz`); the remote host has only the
   derived Train+Valid label artifact, which is exactly the separation we want.
   The frozen manifest hashes were re-verified before the label file was opened.
3. Scoring: OGB official `Evaluator("ogbg-molpcba")` AP **and** an independent
   per-task sklearn AP; they agree to `5.6e-17`. 127 of 128 tasks are scoreable
   on both splits; Test observed entries 3,376,672 with 56,698 positives
   (1.679 %).

## 6. Where the Test difference comes from

Per-task Test deltas, SP − S3 over the 127 scored tasks:

| stratum (Test positives) | tasks | improved | degraded | mean Δ | median Δ |
|---|---:|---:|---:|---:|---:|
| ≥ 20 | 108 | 50 | 58 | **−0.0000770** | −0.0004625 |
| 5–19 | 18 | 12 | 6 | +0.0248532 | +0.0214577 |
| < 5 | 1 | 0 | 1 | −0.1957466 | −0.1957466 |
| all | 127 | 62 | 65 | +0.0019157 | −0.0000372 |

* On the 108 tasks with ≥ 20 positives — the only stratum whose AP is stable —
  SP is *not* better than S3 (`50/58`, mean −0.00008).
* The whole net +0.0019 comes from small-support tasks (mean +0.0249 on 18
  tasks) where AP moves in large jumps; the same stratum has the worst individual
  losses (−0.183).
* The top-10 tasks account for 50.4 % of the total positive gain, and the single
  largest positive flip (+0.227) is larger than the net gain by two orders of
  magnitude. This is a wash, not a transferable improvement.
* On Validation the picture is the same: ≥ 20 positives, 107 tasks, 54 up / 53
  down, mean −0.0011.

## 7. Valid → Test transfer

| model | Best Valid | Test | drop |
|---|---:|---:|---:|
| W2 | 0.265273 | 0.253616 | −0.011657 |
| S3 | 0.271489 | 0.250633 | **−0.020856** |
| SP | 0.266722 | 0.252549 | −0.014173 |

Validation ranking (S3 > SP > W2) and Test ranking (W2 > SP > S3) are **reversed
at both ends**. The two models that exploit explicit three-hop structure lose
more on Test than the plain W2 trunk, i.e. on this split the structural residual's
Validation advantage is split-specific rather than a generalisation gain. With a
single seed and 127 tasks, a ±0.002 Test difference is well inside seed noise;
nothing here supports a claim that either path model generalises better.

## 8. Decision (per the pre-registered table)

```
Validation gain  −0.004766831  <  +0.002   -> not improved
Test gain        +0.001915721  <  +0.002   -> not improved
label: NO_CLEAR_GAIN
```

* Do **not** adopt S3-SP as the main backbone. S3 remains the best
  Validation-selected checkpoint, but its Test advantage over W2 is also absent
  (−0.0030), so the honest position is that the "explicit paths" question is
  *unresolved on Test* rather than answered.
* `further_rounds_authorised = false`. No extra heads, no longer training, no
  second architecture, no seed 1, no COUNT/ROOT ablation, no new scaffold split
  is opened by this result.
* If a follow-up is ever authorised, the only defensible target is the
  Valid→Test gap (why S3 loses 0.021 on Test while W2 loses 0.012), **not** more
  attention capacity on top of S3 — the capacity experiment has now been run and
  it did not help.

## 9. Cost

| item | value |
|---|---:|
| new parameters | 17,888 (+1.76 % over S3) |
| selective-branch FLOPs | `0.503 ×` S3 branch (830 atoms, 3,210 paths, measured formula) |
| epoch time | `170.36 s` = `1.029 ×` S3, `1.364 ×` W2 |
| training GPU-hours | `1.9018` |
| Test freeze (3 checkpoints, replay + prediction) | `0.0125` GPU-h (45.1 s) |
| audits + smoke | `0.0097` GPU-h |
| **total this round** | **`1.9240` GPU-h** (target ≤ 3, cap ≤ 5) |
| CPU preprocessing | Test structure cache 43,793 rows / 4,859,378 paths in `22.8 s` |

## 10. Provenance

| step | job | commit | wall |
|---|---|---|---|
| GPU audit (init + architecture) | `sp-audit-20261108-130537-730cd425` | `98ac4fd2` | 24 s |
| Test structure cache (CPU, `res-local`) | `sp-test-cache-20261108-130550-a7279125` | `98ac4fd2` | 51 s |
| SP training, 40 epochs | `sp-SP-20261108-131055-5deba912` | `79bf2304` | 1 h 54 min |
| Test freeze | `sp-test-freeze-20261108-150945-86303911` | `5ec90109` | 56 s |
| Test score (labels) | local | `5ec90109` | 4 s |

`run.json` records `git_state.dirty=true` because the remote worktree carried two
**pre-existing untracked** files from the S3 round
(`tracks/.../lcrc_pcba_s3_v1/analysis/{architecture,cost}_failed.json`); the
tracked tree was exactly at `79bf2304`. No tracked file was modified on the
remote.

## 11. Artifacts

```
analysis/init_audit.json              parameters, init parity, epoch-0 parity, gradients, roundtrip
analysis/architecture_audit.json      13 structural/semantic checks + FLOP model
analysis/training_parity.json         27 recipe/plan/parity checks
analysis/selected_checkpoint.json     Validation-only selection (test_read=false)
analysis/test_prediction_manifest.json sealed logits + Validation replay
analysis/test_scores.json             official + independent Test AP, per-task AP
analysis/per_task_valid_test.csv      per-task Valid/Test AP and deltas
analysis/performance_comparison.json  all models, strata, cost, attention diagnostics
analysis/decision.json                NO_CLEAR_GAIN with the pre-registered thresholds
analysis/gpu_ledger.json              1.9240 GPU-h, budget compliance
analysis/test_cache_build.json        label-free Test cache invariants
analysis/test_predictions/*.npy       frozen Test logits (W2, S3, SP)
```

`runs/SP/` (checkpoints, `train_curve.csv`, `run.json`, `per_task_ap.csv`,
`train_eval_epoch40.json`) stays local / git-ignored, like every other
`lcrc-pcba` round.
