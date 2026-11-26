# FSRP-SCRATCH v1 — structural-unit regularization from training step 0

Round: `fixed-source-structural-regularization-scratch-v1`
Seed 0 · PCBA official Train/Valid · 25 epochs of the frozen 40-epoch recipe ·
Test never loaded or scored · single GPU (`res-gpu1`, A100-SXM4-40GB), one training unit.

**Verdict: `FIT_REDUCTION_WITHOUT_GAIN`** — training fit is unambiguously lower at every
measured stage, and Valid prediction is *not* improved: best Valid AP
**0.276997** (epoch 19) vs historical FIXED1 **0.275847** = **+0.00115** (paired task
bootstrap 95% CI [−0.00417, +0.00640], 64 tasks better / 63 worse — indistinguishable from
zero), while the endpoint (−0.00217), the last-5 mean (−0.00085), the three phase means
(1–12: −0.00367, 13–20: −0.00087, 21–25: −0.00085) and 19 of 25 epoch-level comparisons
are *below* FIXED1. The pre-registered +0.003 advance threshold is not met.

---

## Answers to the four required questions

**1. Does applying the structural constraint from the start beat the original FIXED1?**
**No.** Best Valid AP is +0.00115 above FIXED1 with a CI that contains zero, and every
other endpoint (endpoint −0.00217, last-5 −0.00085, high-support −0.00041 at the best
checkpoint, +0.00039 at the endpoint) is flat to slightly negative. The highest single
epoch of the whole run (epoch 19, 0.27700) is +0.00544 above FIXED1 *at that same epoch*
but is not sustained (0.27356 at epoch 20, 0.27032 at epoch 21) and does not carry the
endpoint. Against the pre-registered rule the answer is **no advance**.

**2. Is the gain larger than in the late-phase continuation experiment?**
**No — it is smaller and differently shaped.** That round reported +0.00419 over its
matched no-mask continuation, but only +0.00172 over the frozen epoch-25 starting model,
with 97/127 tasks improved. Here the comparison against the *from-scratch* no-mask model
at the identical 25-epoch horizon is +0.00115 at best, with 64/127 tasks improved and a
worse endpoint. Both absolute numbers sit below +0.003; this one is the smaller of the
two and is the only one accompanied by a *negative* endpoint and last-5 (the late-phase
experiment had positive endpoint/last-5 against its matched continuation). Applying the
mask for all 25 epochs did not convert the late-phase matched-arm signal into an absolute
performance advance.

**3. Does the effect come from late-phase fit suppression, or from a favourable change of
the whole training trajectory?** Neither description fits. Observationally:

* the early phase is **worse**, not merely slower to start: mean Valid AP over epochs
  1–12 is −0.00367 below FIXED1, and only 2 of those 12 epochs are better; the landmark
  epochs are equal at AP 0.15 and 0.20 (both epoch 2 / 4) but the masked run reaches
  AP 0.25 two epochs later (10 vs 8) and AP 0.27 one epoch later (18 vs 17);
* the deficit then **shrinks** (13–20: −0.00087; 21–25: −0.00085) but never turns into a
  systematic advantage (2 of 8 and 2 of 5 epochs better, respectively);
* Train fit is lower at **every** measured stage (Train AP −0.00685 / −0.01113 / −0.01225
  at epochs 12 / 20 / 25; Train BCE +0.00038 / +0.00049 / +0.00044), i.e. the fit
  reduction is present from early training and grows, while Valid does not improve;
* the same-stage (epoch-25 vs epoch-25) Train−Valid gap is −0.0101 narrower
  (0.3136 vs 0.3237), and that narrowing comes from the lower Train AP, not from a
  higher Valid AP.

So the observed pattern is a **uniformly suppressed fit with a slightly depressed Valid
trajectory plus one favourable fluctuation at epoch 19** — not late-phase fit suppression
buying a late advantage, and not a favourable reshaping of the whole trajectory. This is a
description of the curves, not a causal claim about the mechanism.

**4. Should the next stage be entered?**
**No.** The pre-registered advance threshold (+0.003 with no reverse movement) is missed,
the sign of the endpoint/last-5 change is negative, and the fit reduction has no matching
generalization benefit. Per the protocol the structural-dropout line stops here: no q
search, no mask schedule, no longer training, no extra seeds/arms.

---

## 1. What was run

| item | value |
|---|---|
| arm | `STRUCT_DROP_SCRATCH` (only arm) |
| architecture | unmodified DSCR **FIXED1** (inherited read-only) |
| initialisation | official FIXED1 seed-0 constructor state, **no checkpoint loaded** |
| parameters | 1,128,468 (audited per tensor group) |
| structural dropout | q = 0.10, one Bernoulli draw per real physical undirected bond per training batch, both directions and both FIXED1 blocks share the mask, inverted-dropout scaling 1/(1−q), independent generator seed 20261126, **active from training step 0** |
| epochs | 25 (steps 0…68,449 of the frozen 40-epoch schedule; cosine **not** rescaled) |
| recipe | batch 128, AdamW from scratch (β 0.9/0.999, ε 1e-8, wd 1e-3), masked BCE, clip 5, fp32, h0 dropout 0.10 |
| order plan | frozen seed-0 plan, global `06f4372…`, 25-epoch prefix `8c85725…` (identical to FIXED1) |
| evaluation | full official Valid every epoch; full official Train at 12 / 20 / 25 and at the best checkpoint |
| selection | highest full Valid macro AP over epochs 1–25 (strict improvement) |
| executed commit | `011a1dc67382` (`res`, branch `fss-v1`) |

The initial state is bitwise the historical FIXED1 seed-0 construction
(`initial_state_sha256 3d313189…50e5c`, new-module state `f7c24ee4…1177`, 71 state
tensors, 65 parameter tensors, identical buffers) and its epoch-0 Valid AP is
0.02906323 / BCE 1.939159 (the untrained W2 readout, as expected from the zeroed `fc2`
residuals).

## 2. Correctness gates — all 13 passed before training

`analysis/checks.json` (`all_passed: true`), evaluated on the run GPU in the same
execution unit, 24.8 s, gate result written *before* the trainer started.

| gate | result |
|---|---|
| C1 initial parameters **and buffers** = FIXED1 seed-0 constructor | bitwise equal; fingerprint `3d313189…` matches the historical record; 71 state tensors |
| C2 initial eval logits = FIXED1 | exact 0 on 256 real Valid molecules |
| C3 optimizer truly from scratch | 0 AdamW state entries, LR at construction 3e-5, step-0 scheduled LR 3.099e-5, one param group |
| C4 mask active from the first training forward | exactly one draw per training forward, 0 draws in eval; q = 0.10 |
| C5 mask does not advance the model RNG | global torch RNG state after a masked training forward is **exactly** the state after the FIXED1 forward; only the mask generator advances |
| C6 25-epoch plan = historical FIXED1 | global and 25-epoch prefix hashes match; 2,738 steps × 128 rows/epoch, each epoch a full Train permutation with 0 Valid/Test rows |
| C7 eval arithmetic = FIXED1 | exact 0 logits difference (init and residual-perturbed states), 0 mask draws |
| C8 Test isolation | 0 finite Test labels visible; no Test entry point referenced in any round module |
| C0a q = 0 reproduces FIXED1 | logits Δ = 0.0, loss Δ = 0.0, gradients over all 1,128,468 entries Δ ≤ 1.9e-9 (max relative 2.3e-9, float32 reduction order only) |
| C0b one physical-bond mask | same tensor object in both blocks and both directions, mask length = directed edges |
| C0c realised drop fraction | 0.098606 mean over the 30-step probe / 0.100009 mean over the run (per-epoch 0.09972–0.10041 across 68,450 steps), inside the 3σ binomial band |
| T1 timing preflight | 0.0409 s/step → 111.9 s/epoch → projected 52.6 min / 0.877 GPU-h; 41.2 GB GPU free |

Note on the q > 0 gate: the official FIXED1 constructor zeroes both blocks' `fc2`, so at
initialisation the residual branch is identically zero and *no* message mask can change
the logits. The "q = 0.10 changes the training forward" part of C0a is therefore evaluated
on residual-perturbed copies of the same state (max logit change 0.4136), and the
"q = 0 equals FIXED1" part is exact on both the initial and the perturbed state.

## 3. Complete Valid curve (official full-task macro AP)

Full 25-epoch table: `analysis/epoch_curves.csv` / `analysis/report_tables.md`.
Reference = historical FIXED1 seed-0, same frozen recipe, same order plan, 25 epochs.

| epoch | FSS Valid AP | FIXED1 Valid AP | Δ AP | FSS Valid BCE | FIXED1 Valid BCE | FSS Train AP | FIXED1 Train AP | dropped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.098782 | 0.099220 | −0.000438 | 0.057004 | 0.056958 | – | 0.100013 | 0.10006 |
| 2 | 0.151751 | 0.152945 | −0.001194 | 0.053055 | 0.053047 | – | 0.168436 | 0.10014 |
| 3 | 0.181380 | 0.181348 | +0.000032 | 0.049941 | 0.050036 | – | 0.218792 | 0.10004 |
| 4 | 0.202383 | 0.206824 | −0.004441 | 0.048882 | 0.048787 | – | 0.256419 | 0.10000 |
| 5 | 0.217572 | 0.221697 | −0.004125 | 0.048438 | 0.048127 | – | 0.290616 | 0.09978 |
| 6 | 0.232761 | 0.237459 | −0.004697 | 0.046918 | 0.046822 | – | 0.326384 | 0.10010 |
| 7 | 0.238801 | 0.244614 | −0.005813 | 0.046598 | 0.046583 | – | 0.349853 | 0.09992 |
| 8 | 0.247413 | 0.252096 | −0.004683 | 0.045699 | 0.045906 | – | 0.378506 | 0.09986 |
| 9 | 0.247438 | 0.249202 | −0.001763 | 0.045951 | 0.046013 | – | 0.389908 | 0.10010 |
| 10 | 0.255011 | 0.253577 | +0.001434 | 0.045277 | 0.045211 | – | 0.417370 | 0.09994 |
| 11 | 0.259290 | 0.266199 | −0.006908 | 0.045509 | 0.045429 | – | 0.431587 | 0.10005 |
| 12 | 0.258600 | 0.269992 | −0.011392 | 0.045494 | 0.045383 | 0.441070 | 0.447919 | 0.09994 |
| 13 | 0.260939 | 0.263579 | −0.002640 | 0.045329 | 0.045301 | – | 0.463189 | 0.09998 |
| 14 | 0.265492 | 0.267615 | −0.002123 | 0.045162 | 0.045205 | – | 0.478081 | 0.10000 |
| 15 | 0.266881 | 0.268190 | −0.001309 | 0.045222 | 0.045235 | – | 0.493117 | 0.09995 |
| 16 | 0.267017 | 0.269260 | −0.002242 | 0.045263 | 0.045368 | – | 0.508083 | 0.10005 |
| 17 | 0.266927 | 0.270694 | −0.003767 | 0.045387 | 0.045461 | – | 0.518633 | 0.10008 |
| 18 | 0.270969 | 0.271269 | −0.000301 | 0.045306 | 0.045586 | – | 0.530946 | 0.09983 |
| 19 | **0.277000** | 0.271558 | **+0.005441** | 0.045295 | 0.045701 | – | 0.542645 | 0.10041 |
| 20 | 0.273564 | 0.273559 | +0.000005 | 0.045206 | 0.045639 | 0.543046 | 0.554179 | 0.10011 |
| 21 | 0.270323 | 0.273017 | −0.002693 | 0.045669 | 0.045786 | – | 0.561738 | 0.10004 |
| 22 | 0.271512 | 0.270976 | +0.000536 | 0.045679 | 0.045963 | – | 0.571841 | 0.10001 |
| 23 | 0.272653 | 0.273738 | −0.001084 | 0.045866 | 0.046117 | – | 0.581325 | 0.09972 |
| 24 | 0.274398 | 0.273216 | +0.001181 | 0.045799 | 0.046182 | – | 0.591137 | 0.10006 |
| 25 | 0.273678 | 0.275846 | −0.002167 | 0.045917 | 0.046377 | 0.587290 | 0.599536 | 0.10004 |

Validation BCE is lower than FIXED1 on **15 of 25** epochs — on the last **12 consecutive**
epochs (14–25) and only 3 of the first 13. So the one axis on which the masked model is
consistently ahead (probability calibration) turns favourable only after the mid phase,
while its AP stays slightly below.

## 4. Main table (`analysis/main_table.csv`)

| metric | FIXED1 | STRUCT_DROP_SCRATCH | Δ |
|---|---:|---:|---:|
| best epoch | 25 | 19 | – |
| **best Valid AP** | 0.275847 | **0.276997** | **+0.001150** |
| best Valid BCE | 0.046377 | 0.045295 | −0.001082 |
| endpoint (25) Valid AP | 0.275846 | 0.273678 | −0.002167 |
| endpoint (25) Valid BCE | 0.046377 | 0.045917 | −0.000460 |
| last-5 mean Valid AP (21–25) | 0.273358 | 0.272513 | −0.000846 |
| last-5 mean Valid BCE | 0.046085 | 0.045786 | −0.000299 |
| Train AP @12 / @20 / @25 | 0.447919 / 0.554179 / 0.599536 | 0.441070 / 0.543046 / 0.587290 | −0.006849 / −0.011133 / **−0.012246** |
| Train BCE @12 / @20 / @25 | 0.029790 / 0.025892 / 0.024222 | 0.030171 / 0.026382 / 0.024660 | +0.000382 / +0.000489 / +0.000438 |
| best-checkpoint Train AP | 0.599532 (ep25) | 0.532051 (ep19) | −0.067482 † |
| high-support Valid AP, best / endpoint / last-5 | 0.298487 / 0.298487 / 0.298487 | 0.298076 / 0.298882 / 0.298487 | −0.000410 / +0.000395 / +0.000001 |
| Train−Valid gap, same stage (epoch 25) | 0.323690 | 0.313612 | −0.010078 |
| parameters | 1,128,468 | 1,128,468 | 0 |
| mask dropped fraction | – | 0.100009 | – |
| formal GPU hours | 1.230773 ‡ | 0.917379 | −0.313394 |

† not like-for-like: this run's best checkpoint is epoch 19, the historical best is epoch
25. The same-stage comparison (epoch 25 vs epoch 25) is the Train AP/BCE rows above.
‡ the historical run scored full Train+Valid+**Test** every epoch; this round scores full
Valid every epoch and full Train at 12/20/25, so GPU hours are not a like-for-like cost.

High-support = Valid positive ≥ 20 **and** negative ≥ 20 → 107 tasks.

## 5. Phase comparison (epochs 1–12 / 13–20 / 21–25)

| phase | FSS mean AP | FIXED1 mean AP | Δ | FSS mean Valid BCE | FIXED1 mean Valid BCE | Δ BCE | epochs better |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–12 | 0.215932 | 0.219598 | −0.003666 | 0.048230 | 0.048192 | +0.000039 | 2 / 12 |
| 13–20 | 0.268599 | 0.269465 | −0.000867 | 0.045271 | 0.045437 | −0.000166 | 2 / 8 |
| 21–25 | 0.272513 | 0.273358 | −0.000846 | 0.045786 | 0.046085 | −0.000299 | 2 / 5 |

The masked run is behind in every phase. The deficit is largest early and shrinks by a
factor of ~4 towards the end, but the late phase is still negative: there is no phase in
which the constraint pays for itself.

## 6. Paired task comparison and sensitivity

Paired bootstrap over the 127 scorable Valid tasks, B = 10,000, seed 20261126
(`analysis/paired_comparisons.json`).

| comparison | mean Δ | 95% CI | P(Δ>0) | tasks + / − | drop-top-3 mean Δ |
|---|---:|---|---:|---:|---:|
| **best (FSS ep19 vs FIXED1 ep25)** | **+0.001150** | [−0.004171, +0.006396] | 0.668 | 64 / 63 | +0.000875 |
| endpoint (ep25 vs ep25) | −0.002169 | [−0.008385, +0.003687] | 0.236 | 67 / 60 | −0.001176 |
| last-5 mean (21–25) | −0.000846 | [−0.005580, +0.003655] | 0.354 | 56 / 71 | +0.000052 |

All three intervals straddle zero and the improved/degraded counts are near 50/50; removing
the three largest-|Δ| tasks does not change any sign except shrinking the best-checkpoint
delta to +0.000875. This is a null result on task-level structure, not a small robust gain.

## 7. Fit versus generalization

* **Fit is lower, consistently**: Train AP −0.00685 (epoch 12), −0.01113 (epoch 20),
  −0.01225 (epoch 25); Train BCE +0.00038/+0.00049/+0.00044. The gap between the arms
  widens monotonically with training, exactly the signature of a training-time constraint.
* **Valid does not gain**: best +0.00115 (CI crosses 0), endpoint −0.00217, last-5 −0.00085,
  phases all negative, 19/25 epochs below FIXED1.
* **The same-stage Train−Valid gap narrows** (0.3237 → 0.3136 at epoch 25) but only because
  Train AP fell; Valid AP did not rise.
* **BCE vs AP**: Valid BCE is lower on the 12 epochs 14–25 (mean −0.000299 over 21–25) but
  higher on 10 of the first 13, i.e. the masked model's probabilities end up slightly
  better calibrated while its ranking of positives is not better. A calibration gain that
  arrives with the fit deficit rather than before it does not support a
  "regularization bought generalization" reading.

That combination is the pre-registered `FIT_REDUCTION_WITHOUT_GAIN` signature.

## 8. Decision

**`FIT_REDUCTION_WITHOUT_GAIN`** (runner-up reading: `NO_CLEAR_GAIN`).

Pre-registered rule as applied:

1. `ADVANCE` requires best ≥ FIXED1 + 0.003 with endpoint/last-5/high-support ≥ −0.001 →
   best is +0.00115 **and** the no-reverse condition fails (endpoint −0.00217). Not met.
2. `WEAK_REGULARIZATION_SIGNAL` requires a late-phase or high-support positive signal
   ≥ +0.001 → late phase −0.00085, high-support best −0.00041, high-support last-5 +0.000001.
   Not met.
3. `FIT_REDUCTION_WITHOUT_GAIN` → same-stage Train AP lower (−0.01225) while best Valid AP
   stays below the practical threshold. **This is the label.**

Consequences per protocol: the structural-dropout line stops. No q search, no mask
schedule, no longer training, no extra seeds, no plain-message-dropout control to explain a
null result. Single seed, single q: exploration decision only.

## 9. Mask and resource consumption

* **Mask**: 68,450 draws = exactly one per training step, 0 draws in eval; mean dropped
  fraction **0.100009** (per-epoch 0.09972 – 0.10041, i.e. ±0.0004 around q = 0.10); the
  same tensor object was used by both directions and both FIXED1 blocks in every epoch;
  mask-RNG state saved in every checkpoint (fingerprint recorded in `run.json`).
* **Formal run**: 3,302.6 s (55.0 min) wall on `res-gpu1` → **0.9174 GPU-hours**, 25 epochs
  at 117.1 s train + 7.03 s Valid eval per epoch, 309.7 s total evaluation, GPU peak
  allocation 430.4 MB (41.2 GB free on the card), 13 gradient-clip events.
* **Whole round**: 0.9174 (formal) + 0.0203 (gpu0 probe) + 0.0181 (gpu1 probe)
  + 0.0069 (gate unit) = **≈ 0.956 GPU-hours** (target 1.0–1.2, hard cap 1.3 — inside the
  cap and just below the target band because the projection of 0.877 h was accurate).
* **Wall clock**: round start ≈ 14:47 CST, formal unit launched 15:14:43, completed
  16:10:19, report finished ≈ 16:25 → **≈ 98 min, i.e. above the 75-minute hard cap**
  (see deviation 2). No queue wait at any point.

## 10. Execution deviations

1. **Deploy path.** The `res` checkout still carried untracked pulled result files from the
   previous round, which collide with a fast-forward. As in the previous round, the commit
   was deployed through a temporary branch (`fss-v1`) based on the remote's current commit
   `90aeaea4`; the branch is merged back into `main` when these results are committed. No
   remote tracked file was edited and no history was rewritten.
2. **Wall-clock cap exceeded (documented, not hidden).** The GPU-hour budget was respected
   (0.956 h), but the round took ≈ 98 min instead of ≤ 75 min. Cause: implementation plus
   four gate iterations (a NaN-tolerant historical reader; moving the probe model to the
   target device; discovering that the zeroed `fc2` residual makes the q > 0 mask
   unobservable at initialisation, which required residual-perturbed copies for C0a/C7;
   two over-strict assertions in the plan and Test-isolation scans) consumed ≈ 27 min
   before the formal unit could start, and the 25-epoch run itself is ≈ 55 min at
   0.041 s/step. The task was **not** extended for queueing (there was none) and no
   training length was shortened to fit: the full pre-registered 25 epochs were run.
   The preflight gate that guards the run was passed against the declared 3,300 s run
   allowance (projected 3,158 s / 52.6 min); that gate does not cover development time.
3. **Gate iterations on GPU.** Two probe jobs failed on the code defects above (fixed and
   re-run), and `res-gpu0` was rejected after its probe measured 0.0729 s/step with only
   19.7 GB free (projected 90 min / 1.50 GPU-h). Only `res-gpu1` hosted the formal unit.
4. **Mask observability at initialisation.** Documented in §2: because the FIXED1
   constructor zeroes `fc2` in both blocks, the message mask cannot change the initial
   logits by construction. The q = 0 / eval exactness gates use residual-perturbed copies;
   the "initial logits equal FIXED1" gate (C2) is exact on the unperturbed initialisation.
5. **Best-checkpoint Train metrics are not like-for-like.** This run's best epoch is 19,
   FIXED1's is 25. The main table reports both the (misleading) best-checkpoint pair and
   the same-stage epoch-25 pair; only the latter is used in the fit argument.
6. **Re-score noise.** Re-scoring the saved best checkpoint reproduces the recorded Valid
   AP to −2.4e-6 (CUDA atomic-add ordering in the second block's scatter).
7. **Guards.** The trainer's soft (3,900 s) and hard (4,200 s) wall guards were armed and
   did not fire (training took 3,303 s); no epoch was truncated.

## 11. Artifacts

* Protocol: `PROTOCOL.md`; fixed config `configs/round.json`.
* Implementation: `code/ss_common.py`, `ss_model.py`, `ss_train.py`, `ss_checks.py`,
  `ss_analyze.py`, `ss_manifest.py`, `ss_formal.py`; tests `tests/test_struct_drop_scratch.py`
  (8 CPU tests, all pass).
* Gates and timing: `analysis/checks.json` (13/13 passed), `analysis/preflight_timing.json`.
* Curves and tables: `analysis/epoch_curves.csv`, `analysis/phase_table.csv`,
  `analysis/main_table.csv`, `analysis/report_tables.md`.
* Comparisons and summary: `analysis/paired_comparisons.json`, `analysis/summary.json`.
* Checkpoints: `analysis/checkpoint_manifest.json` and `results/manifest.json`
  (best ep19 `6e964ca8…`, endpoint ep25 `7afadf51…`, last ep25 `71f42d22…`, sha256 recorded).
* Lightweight run records: `results/runs/STRUCT_DROP_SCRATCH_seed0/`
  (`run.json`, `train_curve.csv` sha `084102f2…`, `per_task_ap.csv` sha `e3ddf353…`,
  `initial_eval.json`, `best_checkpoint_eval.json`, `training_order.json`); the full
  checkpoints stay in the git-ignored `runs/` tree. Ledger: `results/ledger.csv`.

No Test file or Test label was loaded, no Test metric was computed, and no result was
selected using Test. Only seed 0 and only q = 0.10 were run. No next round is started.
