# REPORT — LCRC-PW v1: one-shot pair-token attention on the frozen W2 atom state

Round id: `lcrc-pcba-pw-v1`
Protocol: [`PROTOCOL.md`](PROTOCOL.md) (frozen before the first formal step, commit `db4a33ab`;
one pre-training amendment, commit `365af2d2`, §6 below)
Training, freeze and score commit: `8e1856a2` (the freeze manifest records the same commit; its
`git_state.dirty_files` are only the untracked analysis artefacts of this round)
Test role: **`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`** — MolPCBA Test was exposed by an
earlier round in this track, so nothing here is a blind generalisation estimate. Selection is
best-official-Validation AP everywhere; Test is an engineering comparison only.
Seed: **0 only**, one model, 40 epochs, no tuning, no second seed, no PW-v2.

Every number below is read back from the committed artefacts:
`analysis/preflight.json`, `analysis/pair_index_build.json`, `analysis/profile.json`,
`analysis/gpu_schedule.json`, `analysis/test_prediction_manifest.json`,
`analysis/test_scores.json`, `analysis/per_task_pw_test.csv`, `analysis/query_entropy.json`,
`analysis/comparison.json`, `runs/PW/run.json`, `runs/PW/train_curve.csv`.

**One-line result.** PW **does not beat** the existing LCRC models. Official Test AP is
**0.250914** (best-Valid) / **0.249187** (epoch 40), i.e. **−0.00270 vs W2**, **−0.00503 vs
RC-REAL** and −0.00058 vs RC-BLIND → decision label **`PW_REGRESSION`**. Validation AP is
*identical* to W2 (0.265173 vs 0.265273), so the deficit is Test-only. Decomposed per task, it
is also **not** where the metric is stable: on the 108 well-supported tasks (Test positives
≥ 20) PW is exactly flat vs W2 (**+0.000034** mean, 57/108 improved, Σ +0.0037), while the
−0.0027 macro deficit is produced by 5 sparse tasks (support 4–39) whose deltas sum to −0.537.
The mechanism is visible in the diagnostics: the branch's 16 queries never specialised
(**normalised attention entropy 0.99997**, effective pairs 390 of 391, max weight 0.0036) and
the residual was driven to **0.8% of the base-logit RMS** by epoch 5 and stayed there, so PW
degenerated into "W2 + a small MLP on the mean pair token". Nothing was retrained, no
hyperparameter was touched, and **no PW-v2 was started** — results are handed back as they are.

## 1. Five-model comparison (same pipeline, same 43,793 Test rows)

| model | params | best-Valid epoch | Valid AP (best) | Valid AP@40 | **Test AP (best-Valid)** | Test AP@40 | Test BCE | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LCRC-W2 | 994,324 | 24 | 0.265273 | 0.261867 | **0.253616** | 0.250985 | 0.048584 | 1.3949 |
| **LCRC-PW** | **1,135,092** | 29 | 0.265173 | 0.263618 | **0.250914** | 0.249187 | 0.048991 | 1.8156 |
| LCRC-RC-REAL | 1,051,022 | 33 | 0.267716 | 0.266950 | **0.255941** | 0.256166 | 0.049581 | 1.9186 |
| LCRC-RC-BLIND | 1,051,022 | 29 | 0.264195 | 0.261850 | 0.251493 | 0.251336 | 0.049058 | 1.9685 |

| contrast (best-Valid) | Δ Test AP | relative | Δ Valid AP | improved / degraded tasks |
|---|---:|---:|---:|---:|
| PW − W2 | **−0.002702** | −1.065% | −0.000101 | 68 / 59 |
| PW − RC-REAL | **−0.005027** | −1.964% | −0.002543 | 60 / 67 |
| PW − RC-BLIND | −0.000579 | −0.230% | +0.000978 | 50 / 77 |
| PW@40 − W2@40 | −0.001798 | −0.716% | +0.001749 | 63 / 64 |
| PW@40 − RC-REAL@40 | −0.006979 | −2.725% | −0.003332 | — |

PW adds **140,768 parameters (+14.2%)** and costs **+30.2% GPU time**; the strongest existing
LCRC (RC-REAL) remains 0.0050 Test AP ahead. PW is ahead of nobody on Test; on Validation it
sits between RC-BLIND and W2 to within 0.001.

## 2. Where the Test deficit comes from (per-task, best-Valid)

| Test-positive stratum | tasks | mean Δ (PW − W2) | median | improved | Σ Δ |
|---|---:|---:|---:|---:|---:|
| ≥ 20 (well supported) | 108 | **+0.000034** | +0.000592 | 57/108 | +0.003721 |
| 5–19 | 18 | **−0.010581** | +0.002597 | 11/18 | −0.190457 |
| < 5 | 1 | −0.156374 | — | 0/1 | −0.156374 |
| all scored | 127 | −0.002702 | +0.000423 | 68/127 | −0.343110 |

| quantity | value |
|---|---:|
| Σ positive mass / Σ negative mass | +1.1038 / −1.4469 |
| five largest positive deltas (share of positive mass) | 62 +0.1074, 117 +0.0694, 9 +0.0638, 109 +0.0496, 13 +0.0391 → **29.8%** |
| five largest &#124;Δ&#124; tasks | 31 **−0.3388** (support 6), 57 **−0.1564** (4), 62 +0.1074 (11), 28 −0.0756 (19), 70 −0.0739 (39) |
| net of those five | **−0.5373** = 157% of the net deficit |
| macro mean without those five | **+0.001591** (122 tasks) |

Two things are true at once and both matter: **(i)** on the 108 tasks where Test AP is stable,
PW is indistinguishable from W2 (mean +0.00003, 57/108 improved, gross ±0.78 AP);
**(ii)** the macro number moves −0.0027 because 5 sparse tasks (support 4–39) swing by up to
−0.34 AP. Task 31 (6 positives) alone contributes −0.3388/127 = −0.00267 to the macro mean,
i.e. 99% of the −0.002702 deficit; drop the five largest movers and PW would be +0.0016 *ahead*
of W2. A single seed cannot separate a systematic degradation from sparse-task variance —
what it can say is that the 108 well-supported tasks (85% of the scored tasks, and the part of
the metric that is stable at this support) did not move at all.

Against RC-REAL the same pattern holds (ge20 −0.000893, 53/108 improved; sparse tasks
−0.0145 mean), so the RC round's gain is neither reproduced nor reversed by PW; PW simply
does not add the relation signal RC-REAL had.

## 3. Train / Validation / Test gap and cost

| model | Train AP@40 | Valid AP@40 | Train − Valid | Valid(best) − Test(best) | params | GPU-hour |
|---|---:|---:|---:|---:|---:|---:|
| W2 | 0.581875 | 0.261867 | 0.320008 | +0.011657 | 994,324 | 1.3949 |
| PW | 0.583813 | 0.263618 | 0.320195 | **+0.014259** | 1,135,092 | 1.8156 |
| RC-REAL | 0.604281 | 0.266950 | 0.337331 | +0.011775 | 1,051,022 | 1.9186 |

PW does not overfit *more* in the train/valid sense (+0.0019 Train AP at the same
Train − Valid gap), but its Validation→Test drop is the largest of the three (+0.0143 vs
+0.0117), consistent with the sparse-task sensitivity above.

Cost: 40 epochs in **1.8156 GPU-hours** (6,536 s wall: 6,106 s training steps, 339 s
Validation, 51 s final Train evaluation, the rest diagnostics and checkpointing), peak GPU
allocation **0.87 GiB**, `step_p50` 52–58 ms during training vs 26.95 ms for W2 in the same
profile harness.
Pre-training profile predicted 1.957 GPU-h (raw 1.303 × 1.502 empirical factor calibrated on
W2's real wall clock), so the run came in **7% under** its own calibrated projection and well
under the 3.0 h hard cap. Inference cost per Test row is +45.7% in step time (profile p50
39.26 ms vs 26.95 ms for W2), all of it in the pair gather + token projections.

## 4. The 16-query attention entropy (inference-only diagnostic)

`analysis/query_entropy.json`, official Validation rows (43,793 molecules, 17,085,366 legal
pairs), best-Valid checkpoint (epoch 29), no retraining, no query re-initialisation.

| quantity | value |
|---|---:|
| mean normalised entropy over the 16 queries (H / log n_pairs) | **0.999967** |
| min / max query normalised entropy | 0.999930 / 0.999995 |
| mean entropy nats (query 0) vs uniform log n | 5.792521 vs 5.792689 |
| mean effective pairs `exp(H)` | **390.07** (mean legal pairs per molecule: 390.14) |
| mean maximum attention weight | **0.00360** |
| `all_queries_near_uniform` (≥ 0.95) | **true** |
| `any_query_collapsed` (≤ 0.5) | false |

All sixteen queries are uniform to 4 significant figures: the softmax never learned to prefer
any distance bucket or any pair, so `z_k` is (numerically) the same mean-over-pairs vector for
every `k`, the 768-dim "multi-query" representation is effectively 48 redundant copies of one
mean token, and the readout is a 2-layer MLP on that mean. This is not a sampling artefact —
no pair is dropped, the segment softmax is exact, and the entropy is flat from the first
epoch's diagnostics (`attn_max_mean` 0.0194 → 0.0188 across all 40 epochs).

## 5. Why the branch degenerated (observation, no new experiment)

From `runs/PW/train_curve.csv`:

| epoch | train BCE | Valid AP | residual RMS / base RMS | branch grad norm |
|---:|---:|---:|---:|---:|
| 1 | 0.06132 | 0.09687 | **0.4388** | 0.0519 |
| 3 | 0.04157 | 0.17525 | 0.0764 | 0.0674 |
| 5 | 0.03783 | 0.21936 | **0.0075** | 0.0116 |
| 10 | 0.03358 | 0.24682 | 0.0067 | 0.0129 |
| 29 (best) | 0.02734 | **0.26517** | 0.0081 | 0.0046 |
| 40 | 0.02621 | 0.26362 | 0.0077 | 0.0038 |

The shared optimizer shrank the residual by ~58× in four epochs (it is the fastest way to
reduce loss while the trunk is still learning), and once it is ~0.8% of the base logits the
branch's gradient is ~0.004 — small but non-zero, so it keeps moving; it just never becomes
useful. Combined with uniform attention, the branch had no mechanism to specialise: with a
zero-initialised output layer the only gradient reaching the pair trunk at step 0 is through
`fc2`, and by the time the trunk sees it the residual has already been suppressed. This is a
statement about *this* architecture/recipe, not a proof that pair interaction is useless — the
RC-REAL result in §1 (+0.0023 vs W2) shows a graph-level relation term can move the metric.

## 6. Pre-training gates (all recorded before the first step)

`analysis/preflight.json` — **11/11 pass on `cuda`**, run at commit `365af2d2`:

| gate | evidence |
|---|---|
| W2 sub-state bitwise equal to the historical seed-0 construction | 0 differing tensors, `base_state_sha256` match |
| inventory / parameter arithmetic | 140,768 new, 1,135,092 total, last Linear exactly zero |
| zero-init forward == W2 | residual exactly 0.0, logits bitwise equal on CPU |
| atom renumbering invariance | relative 2.3e-7 (≤1e-4) |
| pair token `i↔j` symmetry | token **bitwise** equal (max diff 0.0); logits 3.8e-6 on CUDA |
| no cross-graph / batch-mate pairs | worst relative 2.9e-7 with per-molecule re-collation |
| buckets vs bonds, no cross-component pairs | 876 sampled rows / 293,617 pairs vs pure-Python BFS: 0 mismatches; 2,000 rows vs the RC distance cache: 0 mismatches |
| segment softmax | row-sum error 1.4e-6, empty molecules exactly 0, weights non-negative |
| gradient path | all new modules non-zero after 3 steps; `key.bias` 3.8e-14 (exact softmax degeneracy, reported not hidden) |
| brute force (float64, per-pair loop) vs batched | relative ≤1e-5 on token/attention/z/residual |
| no-legal-pair graphs | residual exactly 0.0, logits equal W2 |
| Train-only labels / plan | Test rows all NaN, official vs independent AP 2.2e-16, order plan SHA == W2's `06f43729…` |

**Amendment (commit `365af2d2`, still before any training step).** Gate 4 originally required
the post-softmax logits to be *bitwise* invariant under the `i↔j` swap. The CUDA run showed a
3.8e-6 difference: `index_add`/`scatter_reduce` are atomic reductions, so the summation order
of `z` changes between two otherwise identical calls. The definition-level property (the pair
token) remains exact on every device; the logit part now uses 1e-4 on CUDA and stays exact on
CPU. The tolerance is recorded in `PROTOCOL.md` §5.

Pair index: `analysis/pair_index_build.json` — 437,929 rows, **152,489,351 legal pairs**,
217,129 cross-component pairs dropped, 6 bucket totals **identical** to the RC round's
independent cache, `labels_read: false`, 188.7 CPU-seconds, no BFS at training time.

## 7. What this round shows, and what is handed back

* **Shows:** with W2's exact trunk, data, order and recipe, a one-shot 140,768-parameter
  pair-token attention readout (16 queries, distance-binned pair tokens, exact shortest-path
  index, no message passing, no iteration) **does not improve** MolPCBA Test AP: −0.00270 vs
  W2, −0.00503 vs RC-REAL. On the 108 well-supported tasks it is exactly flat; the deficit is
  concentrated in 5 sparse tasks. The diagnostic reason is unambiguous: uniform attention +
  a residual suppressed to 0.8% of base RMS = a branch that cannot express any relation.
* **Does not show:** that pairwise structure is useless (PW implemented one specific readout;
  the RC round's graph-level relation term did move the metric by +0.0023), nor anything about
  seeds (single seed, no repeats), nor a blind Test estimate (Test is exposed).
* **Handing back:** per instruction, no PW-v2, no multi-seed, no extra hyperparameter search,
  no ablation was run. If the researcher wants another attempt, the two concrete handles this
  round produced are (a) *when* the zero-initialised tail is opened to the trunk
  (a delayed/larger-`fc2` warm-up would let the pair trunk see gradient before the residual is
  suppressed) and (b) *why* the queries stay uniform (an additive 16-query readout with no
  per-query bias/normalisation has no pressure to decorrelate). Both are design questions, not
  implementation fixes, so they are left to the researcher.
