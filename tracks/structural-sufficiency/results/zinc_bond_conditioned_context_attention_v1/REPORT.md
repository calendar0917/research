# SSRC side round — zinc bond-conditioned context attention v1

**Does replacing the frozen BJOC-Null bond context moments with a learned per-bond
attention allocation help official-valid MAE, and is any gain specific to the
per-bond conditioning rather than to capacity or global reweighting?**

| field | value |
|---|---|
| protocol_id | `zinc-bond-conditioned-context-attention-v1` |
| status | **COMPLETE** |
| primary label | **`D_ATTENTION_ACTIVE_NO_USEFUL_GAIN`** |
| performance axis | C **worse** than P3 by 0.007452 MAE (CI95 [−0.013741, −0.002427], both seeds) |
| conditioning axis | C−G = −0.001410 (CI95 [−0.003766, +0.000924]); conditioning is **not** better than the graph marginal |
| capacity axis | C−F = −0.007983 (CI95 [−0.015924, −0.001993]); the loss is not a capacity effect |
| honest one-liner | the attention is numerically alive but its effect on valid error is **negative** |
| run commit | `a6b51bbb2207053a24e277c02afd4036adbf8b07` |
| GPU budget | **1.450 / 4.0 A100-hours** (36.2%), six formal runs, no failed or repeated GPU job |
| official test | **never loaded** (no test code path; AST-gated) |
| valid labels | **read once**, after all six predictions and the mechanism record were frozen |
| protocol freeze | `a6b51bb` (PROTOCOL + config + code + 22 CPU tests committed together) |

---

## 0. TL;DR

1. Five stages ran in the frozen order: audit/gates/FLOPs → smoke → six formal
   runs → label-free mechanism → a single valid score. P3 was reused, not
   retrained: the execution regime matched (A100-SXM4-40GB, driver 550.163.01,
   torch 2.5.1+cu124), and the frozen soups replay the historical valid arrays
   **exactly** (MAE 0.07910149778419873 / 0.08058639967715135, diff 0.0).
2. Valid MAE (seed-mean of single models, no ensembling): **P3 0.079844**,
   **F 0.079313**, **G 0.085886**, **C 0.087296**.
3. C is **worse** than P3 by 0.00745 (9.3% relative), on both seeds
   (−0.00838 / −0.00653), with a paired CI that excludes zero and a negative
   sign in every robustness set. G is worse by 0.00604 with the same sign
   pattern. F is statistically indistinguishable from P3 (+0.00053, CI includes 0).
4. The conditioning is **not** the useful axis: C−G = −0.00141 (CI includes 0),
   C−F = −0.00798 (CI excludes 0). The loss comes from replacing the Null
   moments with a learned reweighting at all, not from per-bond vs per-graph.
5. Mechanism: the attention is clearly **not inert** — D = 0.310/0.340,
   normalised entropy 0.952/0.958, forced-uniform changes the prediction by
   0.093/0.079 on average — but the *same-weight* C→G change is only
   0.0056/0.0083. Most of the movement is the learned global reweighting,
   and that reweighting generalises worse than the frozen uniform pooling.
6. Cost was low (1.45 GPU-h). The negative result is therefore cheap and
   informative: within this design the frozen Null moments are a better
   inductive bias than the learned alternative, and there is no case for
   promoting C (or G) as a follow-up candidate.

---

## 1. What was asked, what was run

The question is narrow and pre-registered in `PROTOCOL.md`: on the **same**
frozen P3 encoder, **same** 6,144 extra parameters, **same** R3 recipe and
**same** initialisation (exactly P3 at step 0), compare three placements of the
new capacity against the frozen P3 soups:

* **C** — per-bond conditional attention over the bond's own junctions;
* **G** — the same attention scores marginalised to one distribution per
  molecule (the minimal "global reweighting" control);
* **F** — an FFN on the composer input, context untouched (capacity control).

Six formal runs: C0/G0/F0 on `res-gpu0`, C1/G1/F1 on `res-gpu1`, one process per
GPU, reversed seed order, 28,440 updates each, soup of the last 5 epochs. No
early stopping, no valid-based selection, FP32, no AMP.

The frozen P3 reference was reused under **Branch A** of the protocol: the
regime audit matched, the historical valid prediction arrays were present, and
the label-free replay reproduced them exactly. No P3 retrain was charged.

### 1.1 Freeze and provenance checks (all passed)

* All frozen input, source and soup SHA256s match the config.
* Rebuilt initial state, group state and encoder state match the frozen R3
  records for both seeds; the C/G/F models are bit-identical to each other on
  the shared state and exactly P3-function at step 0 (prediction ≤ 5e-5,
  W/V ≤ 1e-5 relative).
* Batch-order hashes equal the frozen R3 hashes (`3c964a71…` seed 0,
  `8710751a…` seed 1).
* CPU gates G1–G9 all pass, including toy moment conservation, symmetry,
  cross-molecule isolation, multi-step gradients, C/G shared-attention
  marginalisation, parity with the frozen R3 training loop (state diff 0.0,
  identical batch order and soup MAE), label isolation (AST scan) and the
  module/parameter audit.
* Parameter count is 419,649 for F/G/C (413,505 + 6,144) and 413,505 for P3.

---

## 2. Results

### 2.1 Official-valid MAE (raw y; development split, not a blind test)

| arm | seed 0 | seed 1 | seed mean | Δ vs P3 | paired 95% CI on Δ |
|---|---|---|---|---|---|
| P3 (frozen) | 0.07910150 | 0.08058640 | **0.07984395** | — | — |
| F | 0.07905061 | 0.07957461 | **0.07931261** | +0.000531 | [−0.002023, +0.003224] |
| G | 0.08518237 | 0.08658986 | **0.08588612** | −0.006042 | [−0.012479, −0.000990] |
| C | 0.08747699 | 0.08711518 | **0.08729609** | −0.007452 | [−0.013741, −0.002427] |

Positive Δ means the arm is better than P3. The bootstrap is molecule-level,
5,000 resamples, seed 20261019. MSE deltas agree in sign for G and C
(C−P3 MSE CI [−0.1039, −0.0010], G−P3 [−0.0992, −0.0009]); F's MSE CI includes 0.

### 2.2 Specificity contrasts

| contrast | seed 0 | seed 1 | mean | CI95 |
|---|---|---|---|---|
| C − G | −0.002295 | −0.000525 | **−0.001410** | [−0.003766, +0.000924] |
| C − F | −0.008426 | −0.007541 | **−0.007983** | [−0.015924, −0.001993] |
| G − F | −0.006132 | −0.007015 | **−0.006574** | [−0.014517, −0.000749] |

C is not better than the graph marginal (the point estimate is negative and the
CI includes 0), and it is clearly worse than the capacity control. The negative
effect is therefore attached to **using learned junction weights in the bond
context**, not to the extra parameters and not to the per-bond axis.

### 2.3 Robustness (all descriptive; the label uses the pre-registered set)

C−P3 gain:

| set | n | gain |
|---|---|---|
| all | 1000 | −0.007452 |
| ordinary (y > −3) | 935 | −0.004338 |
| tail (y ≤ −3) | 65 | −0.052253 |
| drop P3-worst-10 | 990 | −0.004671 |
| without row 172 | 999 | −0.005157 |
| cycle-penalty complement | 964 | −0.007014 |

C is worse in every set, so the sign is robust even though the magnitude varies.
The tail (65 high-magnitude molecules) is hit hardest: C 0.3012 vs P3 0.2489
MAE there, while F is slightly *better* than P3 on the tail (0.2418). Row 172
(y = −20.34) is the single largest P3 error (max AE 5.98); it is reported, not
silently removed — removing it lowers every arm's MAE and does not change any
ordering.

Subgroup MAE (seed-mean):

| subgroup | n | P3 | F | G | C |
|---|---|---|---|---|---|
| all | 1000 | 0.079844 | 0.079313 | 0.085886 | 0.087296 |
| ordinary | 935 | 0.068090 | 0.068017 | 0.070833 | 0.072427 |
| tail | 65 | 0.248923 | 0.241789 | 0.302422 | 0.301176 |
| cycle-penalty proxy | 36 | 0.229828 | 0.235111 | 0.245083 | 0.249017 |

`better_fraction` for C−P3 is 0.452 (C helps on 45.2% of molecules), so this is
not a handful of catastrophic rows driving the mean; the max single-molecule
share of the ΔMAE is 7.2%.

### 2.4 Training health (per run, identical recipe)

| run | updates | soup fit MAE | step ms | wall s | grad check | clip events |
|---|---|---|---|---|---|---|
| F0 | 28,440 | 0.001956 | 24.5 | 703 | U≠0, T=0→≠0 | 12,659 |
| F1 | 28,440 | 0.001734 | 26.8 | 779 | ok | 14,165 |
| G0 | 28,440 | 0.001673 | 30.3 | 870 | Q≠0, K=0→≠0 | 13,252 |
| G1 | 28,440 | 0.001412 | 32.4 | 937 | ok | 14,229 |
| C0 | 28,440 | 0.001603 | 30.4 | 874 | Q≠0, K=0→≠0 | 13,086 |
| C1 | 28,440 | 0.001210 | 31.9 | 922 | ok | 13,962 |

All runs used the frozen 360 epochs / 28,440 updates, the frozen batch order,
cosine LR (max deviation 4e-17), and produced finite losses and gradients. The
attention arms **fit the train set slightly better** than F (soup fit 0.0012–
0.0017 vs 0.0017–0.0020) and still generalise worse — this is not an
optimisation failure or an underfitting story.

---

## 3. Mechanism diagnostics (label-free, frozen before the label read)

Computed from the frozen C and G soups on the 1,000 valid structures, no labels.

| statistic | seed 0 | seed 1 | reading |
|---|---|---|---|
| D (C) | 0.3097 | 0.3401 | learned allocation is far from the graph marginal |
| normalised entropy | 0.9525 | 0.9581 | mildly non-uniform (uniform = 1.0) |
| mean max attention weight | 0.0790 | 0.0751 | vs 0.0291 uniform baseline |
| W-content dispersion, C | 0.121 | 0.139 | per-bond contexts differ |
| W-content dispersion, G | 2.4e-9 | 2.4e-9 | constant per molecule by construction |
| mean |Δpred| C vs uniform | 0.0929 | 0.0786 | forcing uniform changes predictions a lot |
| mean |Δpred| G vs uniform | 0.0953 | 0.0840 | same for the marginal arm |
| mean |Δpred| C vs G (same weights) | 0.0056 | 0.0083 | the per-bond axis itself moves little |

The same-weight C→G moment conservation is exact up to float32 noise
(W/V max relative error ≤ 2.3e-7; see the post-hoc note in §7). Inertness is
**false** for both seeds: D ≫ 0.02, entropy < 0.99, and the forced-uniform
prediction change ≫ 5e-4.

Interpretation (response magnitude, not performance or chemistry): the trained
attention spends its movement on the **learned reweighting of junction content**
(which G also does), and only a small part on the **per-bond allocation**.
Both reweightings lower valid performance relative to the frozen uniform
moments, with G and C losing almost the same amount. That is consistent with
the numerical result and with the fact that the frozen mean pooling is a strong
inductive bias for this encoder.

---

## 4. Cost

| item | value |
|---|---|
| analytic GEMM FLOPs vs P3 (train batch 128) | F ×1.0197, G/C ×1.0235 |
| attention pair work | 3.70M score MACs + 22.20M moment MACs, negligible vs 2.0 GFLOPs |
| smoke projection with 1.20 margin | 1.263 GPU-h |
| actual GPU-pool wall time (all jobs) | **1.450 GPU-h** (36.2% of the 4.0 cap) |
| formal training only | 5,085 s = 1.413 GPU-h |
| per formal run | 703–937 s |
| failed / repeated GPU jobs | 0 |

The only non-training GPU jobs were prep (21 s) and mechanism (8 s). The budget
was never at risk; the cap did not influence any decision.

---

## 5. Verdict against the pre-registered thresholds

| label | result |
|---|---|
| `PERFORMANCE_PROMISING` (C) | **no** — gain is negative, not ≥ 0.004 |
| `PERFORMANCE_PROMISING` (G) | **no** |
| `PERFORMANCE_PROMISING` (F) | **no** — gain 0.00053 < 0.004, CI includes 0, fails drop-worst-10 |
| `CONDITIONING_PROMISING` | **no** — C−G is −0.00141, not ≥ 0.002 |
| `CAPACITY_CLAIM` (C > F) | **no** — C is worse than F, CI excludes 0 |
| attention inert | **no** — D/entropy/prediction change all far past the inert bounds |
| primary decision label | **`D_ATTENTION_ACTIVE_NO_USEFUL_GAIN`** |

No threshold was changed after seeing the numbers. This is a clean
pre-registered negative result, not an inconclusive one: the sign is
consistent across seeds, robustness sets and both attention arms.

---

## 6. The five questions

**1) Best real valid number vs P3, per seed / mean / CI / robustness.**
F is the best non-P3 arm at 0.079313 vs P3 0.079844 (+0.00053), but this is
statistically indistinguishable from zero (CI [−0.00202, +0.00322]) and does not
survive dropping the P3-worst-10. C is 0.087296, i.e. **worse** by 0.00745
(CI [−0.01374, −0.00243]) and worse on both seeds and in every robustness set.
The best real valid number produced by this round is therefore F's 0.079313,
and it is not an improvement that can be claimed.

**2) C vs G and vs F, with attribution.**
C−G = −0.00141 (CI [−0.00377, +0.00092]): the per-bond conditioning is not
better than a per-molecule marginal computed from the same scores, and the point
estimate is negative. C−F = −0.00798 (CI [−0.01592, −0.00199]): the loss is not
a capacity effect, because the same 6,144 parameters on the composer input are
neutral. Attribution: putting learned junction weights into the bond context
(the C *and* G mechanism) is what hurts; the additional per-bond axis adds
nothing measurable.

**3) Is C a better follow-up candidate than P3, and at what cost?**
No. C is worse by 0.0075 MAE (9.3% relative), significantly so, with the same
sign in both seeds and every robustness set; G is worse by 0.0060. The round
cost 1.45 A100-hours (36% of the cap), so the negative answer is cheap. F, the
capacity control, is the only arm with a non-negative point estimate, and its
effect is too small and too fragile to promote.

**4) What is still missing for a publishable innovation, and one next step.**
Missing: (a) a positive, specific effect — none exists here; (b) confirmation on
data not reused across many prior rounds (this valid split has been the
development endpoint of the whole track, so even a positive number would need a
blind confirmation); (c) a mechanism-level claim about *when* conditioning
helps (junction-set size, graph size, sparsity), which this round does not
support; (d) a clear separation from prior edge-conditioned / adaptive
aggregation work (SAT, GPM, ESA, GRIT, SACA, DGT, Cardinality Preservation).
The one next step I would recommend is **not** another attention variant on the
same axis: keep the frozen Null moments and test a **zero-initialised, strongly
regularised correction** (or condition the reader instead of the moments), so
that the model can only deviate from P3 where the data pays for it. The present
result says full replacement destroys a useful inductive bias; a gated,
shrinkage-style correction is the natural way to test whether *any* learned
deviation is worth it.

**5) Which checks were completed / not completed, and why.**
Completed: frozen-input and source hashes; init/encoder/batch-order provenance;
bit-exact shared weights and exact P3-function at step 0; CPU gates G1–G9;
FLOPs; smoke budget projection; six formal runs; label-free mechanism; a
single-read valid score with the P3 replay reproduced exactly; robustness,
tails, subgroup and extreme-row reporting; GPU ledger and run manifest.
Not done, by design: official test (never loaded; no code path), early
stopping / valid-based selection, more than two seeds, a same-round P3 retrain
(Branch A matched, so none was needed), and any post-hoc per-molecule bucket
analysis that would require a second valid-label read. One **defect** is
disclosed in §7.

---

## 7. Defects, deviations and execution notes

1. **Mechanism record defect (disclosed).** `cmd_mechanism` initialised its
   C→G moment-conservation accumulator inside the per-mode loop, so the frozen
   `mechanism_this_round.json` stores `null` for
   `moment_conservation_C_vs_G_same_weights`. The decision does not use this
   field. It was recovered post hoc from the same frozen C soups with the same
   `BM.moment_conservation` code and is stored in
   `posthoc_moment_conservation.json` (label-free; W/V max relative error
   ≤ 2.3e-7 for both seeds). No frozen artefact was rewritten.
2. **Deployment note (infrastructure only).** The remote checkout carried
   untracked Phase 2b `test_read.lock` / `valid_read.lock` copies that blocked
   the Git bundle merge. They were copied to `/home/hxy/bcca_lock_backup/` on
   the host and removed from the checkout via an `rr` job; no tracked file was
   edited remotely and no Phase 2b artefact was modified in the repository.
3. **Single read respected.** `valid_read.lock` records `read_count = 1`,
   `reread = false`; the combined prediction freeze was written at 11:40:58Z,
   before the label read at 11:41:00Z; the mechanism hash in the lock and in
   `valid_metrics.json` matches the frozen record.
4. **No P3 retrain.** The regime audit matched Branch A, and the replay
   reproduced the historical valid MAE with diff 0.0 (well inside 1e-5).
5. **No post-hoc selection.** Nothing was re-scored, no threshold was moved, no
   arm/seed was dropped.

---

## 8. Artefact index

Frozen artefacts (all in this directory unless noted):

```
PROTOCOL.md                          frozen protocol (commit a6b51bb)
../../configs/zinc_bond_conditioned_context_attention_v1.yaml   frozen config
p3_reuse_audit.json                  hashes, init/encoder/batch-order provenance, replay
cpu_gates_this_round.json            G1–G9 with details
flops_this_round.json                analytic GEMM + pair counts
smoke_this_round.json                2-epoch timing + 1.263 GPU-h projection
train_{F,G,C}_s{0,1}.json            per-run records (hashes, gradients, snapshots)
valid_freeze_{F,G,C}_s{0,1}.json     per-prediction freeze records
valid_prediction_freeze.json         combined prediction freeze (pre-label)
mechanism_this_round.json            label-free diagnostics + inertness
valid_read.lock                      single-read marker (read_count = 1)
valid_metrics.json                   per-arm metrics, contrasts, subgroups, tails
valid_decision.json                  thresholds, labels, primary label
gpu_ledger.json                      every GPU-pool job + per-run ledger
run_manifest.tsv                     per-run soup/prediction hashes and seed MAE
posthoc_moment_conservation.json     post-hoc recovery of the dropped field
```

Soups (`soup_*.pt`) and prediction arrays (`valid_pred_*.npz`,
`mech_pred_*.npz`) are kept on the compute host and are not committed, matching
the Phase 2b convention; their SHA256s are recorded in the train/freeze records
and in `run_manifest.tsv`. The code for this round lives in
`tracks/structural-sufficiency/code/zinc_bcca_v1/` and is covered by 22 CPU
tests (`uv run pytest -q tracks/structural-sufficiency/code/zinc_bcca_v1/tests`).

---

## 9. Bottom line

Within the frozen BJOC object abstraction, replacing the uniform Null bond
moments with a learned attention allocation **does not help**: it costs
0.0060–0.0075 MAE against the frozen P3 reference, on both seeds, in every
robustness set, while the same parameter budget on the composer input is
neutral. The per-bond conditional axis is not the useful one — the graph
marginal loses the same amount. The attention is numerically alive and uses its
freedom, but the freedom is spent in a direction that generalises worse than
the frozen uniform pooling. This branch should not be promoted; the cheapest
next test is a zero-initialised, regularised correction on top of the Null
moments rather than a replacement of them.
