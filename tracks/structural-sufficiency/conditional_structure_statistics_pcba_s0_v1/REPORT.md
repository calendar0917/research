# REPORT — css-pcba-s0-v1

**Round**: conditional structure statistics on `ogbg-molpcba`, seed 0
**Question**: with identical structure objects, encoder, parameter count (831,088),
initial state and 40-epoch budget, does conditioning **each raw occurrence** on the
molecule's task context (`E`, EARLY_TASK) beat applying the same conditioning **once
after** the root aggregation (`L`, LATE_TASK)?
**Status**: both primary arms complete, 109,520 updates each, all correctness gates
green, **S NOT_PURCHASED** by the pre-registered rule.
**Every Test number here is `DEVELOPMENT_TEST`** (`test_policy: always`, authorised
for this round). Test never selected a checkpoint.

---

## 1. Headline

| arm | params | best ep | best Val AP | **Test AP @best** | ep40 Val AP | ep40 Test AP | last-5 Val | last-5 Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **E** EARLY_TASK | 831,088 | 36 | **0.246329** | **0.240453** | 0.245035 | 0.240328 | 0.245428 | 0.240317 |
| **L** LATE_TASK | 831,088 | 31 | 0.242804 | 0.232267 | 0.240905 | 0.232187 | 0.240940 | 0.232127 |
| difference | 0 | — | +0.003525 | **+0.008186** | +0.004130 | **+0.008141** | +0.004488 | +0.008190 |

Every one of the four E−L comparisons is positive, on Validation **and** on Test, in
the best-Validation state, the epoch-40 state and the last-5 mean. That is the only
sense in which "E wins": the sign is consistent.

It does **not** survive the decomposition that this round pre-specified (section 4):
the whole difference lives in the sparse tail. On the 107/108 tasks with ≥ 20
positives in both Validation and Test, E is **−0.0035 on Validation** (resampling
p = 0.081) and **+0.0002 on Test** (95 % task-resampling interval
[−0.0038, +0.0044], p = 0.54). Nothing there.

Both arms also land ≈ 0.013 Test AP **below** the historical LCRC W2 anchor
(0.253617), so the subject is not a performance candidate either.

## 2. Training and evaluation integrity

* Frozen seed-0 order plan, both hashes re-verified
  (`06f437299fa4…` global, `1a24dfca5489…` consumed-40 prefix); 2,738 updates/epoch,
  40/40 epochs, `steps_completed = 109,520` for both arms.
* Recipe inherited unchanged: AdamW (3e-4 cosine, 3e-5 warmup, wd 1e-3), clip 5.0,
  fp32/no TF32, batch 128, masked observed-entry BCE, dropout 0.10 on `h` only from
  generator seed 230,000. Zero gradient-clip events, max observed gradient norm 0.76,
  no non-finite value in any of the 40 × 3 evaluation rows.
* Full Train (350,343) + Valid (43,793) + Test (43,793) evaluation after **every**
  epoch, official OGB AP (`Evaluator('ogbg_molpcba')`), independently reproduced with
  sklearn to 3.5e-18 during preflight.
* Both arms ran the same committed revision `d5d3b85b`, same host `a100-2`, same
  driver 550.163.01 / torch 2.5.1+cu124 (one execution regime), one process per GPU.
* `q0 = qt = Wc = 0` at initialisation, so all three arms start as the same function
  (checked at 9.5e-07 on 8,192 rows); the initial-function evaluation is cached once
  and not re-spent per arm.
* CPU + GPU gates (14) all pass: S1 = 2 × bonds, per-root S2 closed-form count,
  middle/bond-id alignment, closure, reorder + renumber invariance (2.4e-07 / 4.2e-07),
  degenerate cases finite, non-exchange witness E = tanh(1)/√2 vs L = 0, task-chunk
  16/8/4 equivalence, Test-label isolation, gradient reachability, resume equivalence
  (2.95e-07), full-split evaluation audit.
* Data pipeline: collation overlapped with the GPU step; the loss sequence is bitwise
  identical to the synchronous path (asserted in `tests/test_css_round.py`).

## 3. Where the difference comes from — and what it is not

Paired per-task AP, E − L, epoch-40 state (127 commonly scored tasks):

| task subset | n | mean Δ Val AP | mean Δ Test AP |
| --- | --- | --- | --- |
| all common tasks | 127 | **+0.00413** | **+0.00814** |
| well supported (≥ 20 pos. in Val and Test) | 107 / 108 | **−0.00345** | **+0.00020** |
| sparse remainder | 19–20 | +0.0447 | +0.0533 |

Task-resampling bootstrap (1,000 draws, seed 20261201, fixed models):

| subset | Val Δ | 95 % interval | P(Δ>0) | Test Δ | 95 % interval | P(Δ>0) |
| --- | --- | --- | --- | --- | --- | --- |
| all 127 | +0.00413 | [−0.0025, +0.0109] | 0.877 | +0.00814 | [+0.0014, +0.0163] | 0.994 |
| well supported | −0.00345 | [−0.0087, +0.0015] | 0.081 | +0.00020 | [−0.0038, +0.0044] | 0.540 |

Sign counts are near coin-flips: 61 tasks improve / 66 degrade on Validation, 62/65 on
Test; median Δ is ≈ −0.0005 / −0.0004. The sparse tasks carry the mean because one
mis-ranked positive on a 4-positive task moves AP by 0.1–0.3: the largest single
contributions are task 72 (+0.234 Val, 4 positives) and task 57 (+0.344 Test, 4
positives) against task 17 (−0.138 Val, 48 positives) and task 9 (−0.067 Test, 32
positives). This is estimator variance, not a structural effect.

Calibration points the same way: E's Training BCE is marginally lower (0.03083 vs
0.03092) while L's Validation BCE is marginally lower (0.04582 vs 0.04605). With the
same parameter count and the same data, the two arms are learning the same thing to
within noise.

## 4. Mechanism: does the task query actually do anything?

Read-only diagnostics on the frozen best-Validation state, first 4,096 official
Validation rows (`rows_sha256 2d23d976…`), no gradient step, no checkpoint written:

| measurement | E | L |
| --- | --- | --- |
| task-query norm | 0.271 | 0.342 |
| between-task query spread | 0.0619 | 0.0732 |
| gate mean / std | 0.964 / 0.259 | 0.980 / 0.254 |
| permuting the 128 query rows: mean abs logit change | 0.233 (prob 0.0041) | 0.129 (prob 0.0021) |
| switching the molecular context term off: mean abs logit change | 0.386 (prob 0.0071) | 0.308 (prob 0.0055) |
| gate mean without the context term (task-independent reference) | 0.997 | 1.012 |
| state hash before/after the checks | unchanged | unchanged |

The query is genuinely used and genuinely task-specific in both arms (permuting the
query rows changes predictions; removing the molecular context collapses the gate
spread from 0.26 to 0.16 and pushes the gate mean back to ≈ 1). E is ~1.8× more
sensitive to *which* task query is applied than L, exactly as the per-occurrence
formula predicts — the conditioning is not decorative. It simply does not convert
into ranking quality on tasks where ranking quality is measurable.

## 5. S arm: NOT_PURCHASED

Pre-registered rule, evaluated after both primary arms:

| condition | value | verdict |
| --- | --- | --- |
| `d40 = Val_E(ep40) − Val_L(ep40) ≥ 0.002` **or** `Val_E(best) ≥ 0.268273` | d40 = **+0.004130** | ✅ trigger fires |
| `Val_E(best) ≥ 0.262273` (floor) | **0.246329** | ❌ **fails** |
| used + 1.2 × projected S ≤ 16 GPU-h | 9.14 + 1.2 × 4.57 = 14.63 | ✅ |

**`NOT_PURCHASED` — reason: `E best-Val 0.246329 < 0.262273`.** The trigger fires, but
a subject that is 0.016 Val AP below the historical W2 anchor is not a position from
which a 128×16-parameter ablation is worth 4.6 GPU-hours of the remaining budget. No
idle budget was filled with another model, another seed or a longer horizon.

## 6. Cost (measured, not projected)

| stage | GPU-hours | wall |
| --- | --- | --- |
| GPU preflight (14 gates, incl. 2-epoch resume check + full 3-split eval) | 0.16 | 10.4 min |
| cost probe (300 real steps, prefetched pipeline) | 0.02 | 1.0 min |
| E, 40 epochs | 4.57 | 4 h 35 min |
| L, 40 epochs | 4.56 | 4 h 34 min |
| **total** | **≈ 9.3** | **≈ 5.2 h** (soft 6 h / hard 10 h; GPU soft 12 h / hard 16 h) |

Per epoch: 411 s (273 s of 2,738 updates at 110.8 ms/step, ~118 s for the three full
evaluations). Peak reserved memory 3.50 GB (E) / 3.10 GB (L) on a 40 GB A100. Collation
(median 37.5 ms/batch) is fully hidden behind the GPU step (mean wait 0.2 ms).

## 7. Position against the historical anchors

Cited for performance position only — different architecture, single seed, and never
an equal-parameter control of this subject.

| model | best Val AP | Test AP (selected state) |
| --- | --- | --- |
| W2 (historical) | 0.265273 | 0.253617 |
| W2-DYNAMIC (historical) | 0.275067 | 0.262226 |
| LCRC_CPC (historical) | 0.265944 | — |
| LCRC_RC_REAL (historical) | 0.267718 | — |
| **E (this round)** | 0.246329 | 0.240453 |
| **L (this round)** | 0.242804 | 0.232267 |

The gap to W2 is −0.019 Val / −0.013 Test for E. Part of that is the different
readout (graph statistics instead of the historical additive atom head) and part is
that this subject is deliberately narrow; either way it is large enough that the round
cannot be read as "a competitive model that happens to answer a mechanism question".

## 8. What is established, and what is not

Established (this protocol, this budget, seed 0):

1. Under identical parameters, initial state and order, per-occurrence conditioning
   gives a consistently positive but small aggregate Val/Test AP difference
   (+0.0035 … +0.0082) over post-aggregation conditioning.
2. That difference is **not** a well-supported-task effect: on the 107/108 tasks with
   ≥ 20 positives it is −0.0035 (Val) / +0.0002 (Test), with intervals spanning zero.
3. Both arms use the task query substantively (query permutation changes predictions;
   removing the molecular context collapses the gate spread), and E is ~1.8× more
   task-query-sensitive than L.
4. The subject as specified is ≈ 0.013 Test AP below the historical W2 position.

Not established / not claimed:

* No training-variance claim: one seed, no replication, no best-of-N. A single-seed
  +0.004 Val difference between two identical-capacity arms is not a resolved effect.
* No molecule-level confidence interval; the bootstrap resamples tasks only.
* No statement about other datasets, other budgets, other readouts, or about
  per-occurrence conditioning in general.
* No causal claim from the mechanism checks — they measure usage, not benefit.
* No S arm, therefore no estimate of how much of E's behaviour is the task query
  rather than the molecular-background term.

## 9. Artefacts

* `protocols/css-pcba-s0-v1.yaml`, `studies/css-pcba-s0-v1.yaml`, `PROTOCOL.md` — the
  frozen rules; `code/css_common.py` holds the constants every run hashes.
* `analysis/analysis.json` — main table, paired per-task differences (4 comparisons),
  both bootstraps, mechanism checks, S gate, label-view audit, commit.
* `runs/EARLY_TASK`, `runs/LATE_TASK` (pulled from `res`, not committed): `run.json`,
  `train_curve.csv` (40 epochs × 3 splits), `best_eval.json`, `epoch40_eval.json`,
  `per_task_ap.csv`, `checkpoint_best.pt`, `checkpoint_epoch40.pt`,
  `checkpoint_last.pt`, `predictions_{best,epoch40}_{train,valid,test}.npz`.
* `analysis/preflight.json` — every gate with its measured value.
* Reproduce: `bash run_remote.sh --set stage=analyze` (analysis only, CPU);
  `--set stage=train --set arm=E` replays the arm from the committed revision.
