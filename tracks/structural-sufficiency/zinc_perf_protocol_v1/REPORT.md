# REPORT — zinc-perf-protocol-v1

| field | value |
|---|---|
| protocol | `zinc-perf-protocol-v1` (new performance protocol) |
| round status | **closed at the development conclusion; no terminal test scoring** |
| data | official PyG ZINC `subset=True`, 10,000 / 1,000 / 1,000 |
| metric | raw-`y` MAE, lower is better |
| primary selector | per-seed single **best-valid** checkpoint (ties → earlier epoch) |
| hardware | `res` = a100-2, 2× A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124 |
| training commits | P3 corrected: `f847a2a`; ZSRT screen: `67af22d` (code-equivalent for ZSRT) |
| artifacts | `runs/<run>/{best.pt,final.pt,last5/,predictions/,train.json,curve.csv}` (git-ignored; pull with `rr pull`) |
| figures | `analysis/curves.png` (real curves); `analysis/main_table.json`; `analysis/round_decision.json` |

## 0. Answers

**Q1 — where is P3 under normal best-valid selection?** P3 best-valid is
**0.079105** (seed 0, epoch 272) and **0.076668** (seed 1, epoch 267), mean
**0.077886 ± 0.001723** (sample std, n=2). The old last-5 soup gives
0.080907 / 0.077923, mean 0.079415. Best-valid selection therefore improves the
P3 reference by **0.001529** over soup (0.001579 over the epoch-360 final).
The old soup estimate 0.079844 was ~0.0020 pessimistic about P3. This is a
selector effect, not a correction of the old soup measurement: the new soup
mean (0.079415) is close to the historical soup mean (0.079844). The
best-vs-soup gain is a within-run paired contrast and is comparable to the
cross-run soup reproducibility band (~0.0027), so 0.0015 is a modest selector
gain, not a new performance level. **No limited
extension is warranted:** both best epochs are <320 and the terminal 320→360
gains are −0.000104 (s0) and +0.000290 (s1), so the pre-declared continuation
rule is false for both seeds. P3 is closed at 360 epochs.

**Q2 — did the new backbone beat P3?** **No.** The best of the four ZSRT
configs is `C_wd_high`, seed-0 best-valid **0.103363** (epoch 315), which is
**+0.024258** worse than P3 seed 0 (0.079105), i.e. +30.7% relative and ~14×
the P3 two-seed sample std. All four configs are worse than P3 by more than the
pre-declared 0.005 stop band. The RRWP ablation `D_no_rrwp` is much worse
(0.185165), so the structural relation input is doing real work; the failure is
sample efficiency of the full-attention composition at 28,440 updates, not a
broken RRWP. The valid tail of C is flat after epoch 315 while train MAE keeps
falling, so this is not a “still improving at the cap” budget truncation.
Per the stop rule, no ZSRT seeds 1/2 were purchased, no freeze was written, and
the terminal test was **not** scored (the audit-only test-scale read is
disclosed in §8).

## 1. What the new protocol changed

1. **Primary number.** The per-seed epoch-end checkpoint with the lowest
   official-valid MAE is the result. Valid may be evaluated every epoch and
   used for early stopping / configuration choice. The last-5 soup is only a
   secondary P3 diagnostic; it is not the selector.
2. **Data.** Official PyG ZINC `subset=True` 10k/1k/1k, raw-`y` MAE. The old
   internal 8k/1k numbers are not mixed in. The audit cross-checks the frozen
   P3 object cache and historical valid predictions (SHA256-verified).
3. **P3 reference.** Frozen R3 recipe only (batch 128, Adam 1e-3→1e-5 per-step
   cosine over 28,440 updates, wd 1e-5, clip 5.0, shuffle offset 101), plus
   per-epoch valid, best/final/last-5 checkpoints, and eval-mode train
   snapshots. No retuning; a pre-declared continuation rule was written before
   training and evaluated after epoch 360.
4. **New family.** One borrowed-backbone family, ZSRT: typed atom/bond
   embeddings + deterministic RRWP k-step pair/node encodings + 8 full-attention
   GritTransformer layers with edge-state updates and degree scaling + sum
   pooling + MLP. It does not read the P3 321-D reader input and has no local
   message passing. GRIT borrowing is disclosed; the family is not claimed as a
   novel mechanism. Four pre-declared screen configs (`A_main`, `B_lr_low`,
   `C_wd_high`, `D_no_rrwp`), one screening seed, selection by the complete
   valid curve / best-valid.
5. **Test.** One-shot terminal read only after a freeze, and only if a usable
   config exists. This round stopped before that point, so test remains unread.

## 2. Main table

All runs: batch 128, 360/360 epochs, 28,440 updates, raw-`y` L1. “Best” is
`best_valid`; “final” is epoch 360; “soup” is the last-5 mean. Test was not
read for any row.

| model | config | seed | params | pretrain source | batch / max→actual epochs / updates | best epoch | best-valid MAE | final valid | soup valid | test | GPU-h |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|
| P3 | R3 | 0 | 413,505 | frozen train-only masked SSL | 128 / 360→360 / 28,440 | 272 | **0.079105** | 0.080947 | 0.080907 | not read | 0.201 |
| P3 | R3 | 1 | 413,505 | frozen train-only masked SSL | 128 / 360→360 / 28,440 | 267 | **0.076668** | 0.077984 | 0.077923 | not read | 0.214 |
| P3 | R3 | mean | 413,505 | — | — | — | **0.077886 ± 0.001723** | 0.079465 | 0.079415 | not read | 0.414 |
| ZSRT | A_main | 0 | 380,161 | none | 128 / 360→360 / 28,440 | 292 | 0.112037 | 0.113828 | 0.115774 | not read | 0.954 |
| ZSRT | B_lr_low | 0 | 380,161 | none | 128 / 360→360 / 28,440 | 359 | 0.148439 | 0.149571 | 0.151376 | not read | 0.933 |
| ZSRT | C_wd_high | 0 | 380,161 | none | 128 / 360→360 / 28,440 | 315 | **0.103363** | 0.104748 | 0.106622 | not read | 1.007 |
| ZSRT | D_no_rrwp | 0 | 377,473 | none | 128 / 360→360 / 28,440 | 344 | 0.185165 | 0.186120 | 0.189851 | not read | 0.825 |

P3 mean ± sample std uses the two seeds; with n=2 the std is descriptive only.
ZSRT has only the screening seed because the stop rule fired; it is **not** a
seed-mean claim.

ZSRT screen (seed 0), with the fixed-object train MAE at the selected
checkpoint:

| config | lr | wd | RRWP | best-valid | best epoch | Δ vs P3 s0 | train MAE at best |
|---|---:|---:|---|---:|---:|---:|---:|
| A_main | 1e-3 | 1e-5 | on | 0.112037 | 292 | +0.032932 | 0.047232 |
| B_lr_low | 3e-4 | 1e-5 | on | 0.148439 | 359 | +0.069334 | 0.048694 |
| C_wd_high | 1e-3 | 1e-4 | on | **0.103363** | 315 | **+0.024258** | 0.031421 |
| D_no_rrwp | 1e-3 | 1e-5 | off | 0.185165 | 344 | +0.106060 | 0.064621 |

C is the best ZSRT config by valid, but it is 0.0243 worse than P3 seed 0. The
0.005 stop band is exceeded by every config; the RRWP control is 0.0818 worse
than C, so removing the relation input is clearly harmful for this backbone.

## 3. P3 calibration and the old reference

Fixed-object comparison (all valid numbers are eval-mode official valid;
`train` is full-10k eval-mode MAE of the fixed checkpoint):

| model | seed | best valid (epoch) | final valid | soup valid | train@best | train@final | train@soup |
|---|---:|---:|---:|---:|---:|---:|---:|
| P3 | 0 | **0.079105** (272) | 0.080947 | 0.080907 | 0.011261 | 0.002270 | 0.001918 |
| P3 | 1 | **0.076668** (267) | 0.077984 | 0.077923 | 0.011949 | 0.002111 | 0.001784 |
| ZSRT C | 0 | **0.103363** (315) | 0.104748 | 0.106622 | 0.031421 | 0.031156 | 0.035377 |

The P3 best-valid checkpoint is **not** the lowest-train checkpoint: the final
and soup fit train ~5–7× better (0.0018–0.0023 vs 0.0113–0.0119) but are
0.0013–0.0018 worse on valid. That is the expected shape under normal
checkpoint selection and is the direct evidence that terminal train descent is
not a valid signal.

Continuation decision (written before training; `analysis/p3_continue_decision.json`):

| seed | best epoch | valid@320 | valid@360 | terminal gain | best epoch ≥320 | gain ≥0.001 | trigger |
|---:|---:|---:|---:|---:|---|---|---|
| 0 | 272 | 0.080842 | 0.080947 | −0.000104 | no | no | **no** |
| 1 | 267 | 0.078274 | 0.077984 | +0.000290 | no | no | **no** |

No continuation was run. Historical frozen soup values were 0.079102 (s0) and
0.080586 (s1); the new-protocol soups are 0.080907 / 0.077923. The ≤0.0027
cross-run soup difference is the known CUDA-nondeterminism band; it is not
used as a result. The reported P3 primary is the new best-valid pair above.

A first P3 calibration used `floor(10000/128)=78` steps/epoch, which reached
the cosine floor 360 updates early. It was corrected to `ceil()=79` / 28,440
and both seeds were rerun; the first-pass runs are archived under
`runs/_archive/` and are not reported. The corrected run is the one in the
table.

## 4. ZSRT: design and outcome

The computation graph, in one line: typed atom/bond embeddings +
deterministic RRWP `k=21` node/pair encodings → 8 full-attention
GritTransformer layers (edge-state update, edge enhancement, degree-scaled
residual, BatchNorm, FFN) → additive graph pooling → MLP. It keeps atom/bond
types, connectivity and bond multiplicity until attention mixes them; it does
not use P3's hand-built junction/incidence summary or its 321-D reader vector.

Why the screen failed, by evidence:

* **Not a selector bug.** C is the argmin of the seed-0 best-valid curve; the
  selector never loads test. A/B/C/D were run with the same seed, data order,
  batch size, epoch budget and update count.
* **Not an RRWP bug.** D (RRWP off) is 0.185165, 0.0818 worse than C. The
  relation encoding is necessary for this backbone.
* **Not overfitting in the parameter-count sense.** C's train MAE at its best
  checkpoint is 0.0314, versus P3's 0.0113; the model is still underfitting
  train relative to P3 at the same 28,440 updates, yet generalizes worse.
* **A real sample-efficiency gap.** C's valid curve improves from 0.1206 at
  epoch 260 to 0.1034 at 315, then is flat/slightly worse (0.1045 at 320,
  0.1042 at 340, 0.1047 at 360) while train MAE keeps falling. More epochs of
  the same recipe are not guaranteed to close 0.0243; the terminal train
  descent is the audit-`592d766` non-signal.
* **Budget context, not an excuse.** GRIT's published ZINC recipe is 2,000
  epochs × 313 updates = 626,000 updates, ~20.5 A100-h/seed, ~22× our
  28,440-update budget. This round deliberately bought a matched 360-epoch /
  28,440-update opportunity (and spent ~5× P3's wall-clock on it); at that
  opportunity the family is worse.

## 5. Real train/valid curves

Selected points; the full 360-point curves are `runs/<run>/curve.csv` and the
figure is `analysis/curves.png`.

* `valid` — epoch-end model in `eval()`, official valid 1,000.
* `online train` — within-epoch running mean of **pre-update** predictions over
  all 10k train rows. This is not an eval-mode number.
* `snapshot train` — epoch-end model in `eval()`, fixed first 2,000 official
  train rows, every 20 epochs.

| epoch | valid | online train | snapshot train |
|---:|---:|---:|---:|
| **P3 s0** | | | |
| 1 | 0.532751 | 0.710443 | 0.536234 |
| 60 | 0.129258 | 0.089450 | 0.097057 |
| 140 | 0.100149 | 0.048439 | 0.059226 |
| 220 | 0.082862 | 0.024810 | 0.026791 |
| 260 | 0.081494 | 0.016046 | 0.013440 |
| 300 | 0.081118 | 0.006865 | 0.009546 |
| 320 | 0.080842 | 0.004360 | 0.005318 |
| 360 | 0.080947 | 0.002253 | 0.002978 |
| **P3 s1** | | | |
| 1 | 0.444278 | 0.720198 | 0.428789 |
| 60 | 0.126070 | 0.089167 | 0.086298 |
| 140 | 0.095071 | 0.048502 | 0.051048 |
| 220 | 0.082593 | 0.024472 | 0.025471 |
| 260 | 0.078787 | 0.014506 | 0.014529 |
| 300 | 0.077653 | 0.007546 | 0.008180 |
| 320 | 0.078274 | 0.004437 | 0.005097 |
| 360 | 0.077984 | 0.002035 | 0.002183 |
| **ZSRT C s0** | | | |
| 1 | 1.116234 | 1.428566 | 1.141815 |
| 60 | 0.246889 | 0.222044 | 0.211927 |
| 140 | 0.218593 | 0.159257 | 0.182148 |
| 220 | 0.130857 | 0.102172 | 0.073429 |
| 260 | 0.120620 | 0.085000 | 0.057844 |
| 300 | 0.106718 | 0.072592 | 0.037429 |
| 320 | 0.104469 | 0.067733 | 0.033563 |
| 360 | 0.104748 | 0.065955 | 0.031062 |

Interpretation boundaries (from audit `592d766`, kept explicit): terminal
train-MAE descent is not a valid improvement; `online train` is a pre-update
within-epoch running mean, not the eval-mode epoch-end train MAE; the last-5
soup is one controlled same-protocol diagnostic, not proof that a family
peaked; and the 20-epoch snapshot grid is not read as a monotone mechanism.

Reading the three columns together:

* P3's online train MAE falls to 0.0020–0.0023 at epoch 360 while valid is
  0.0780–0.0809 and essentially flat after epoch ~270. The online number is
  lower than the final eval-mode full-train MAE (0.0021–0.0023) because it is a
  different, pre-update running quantity; neither is a valid-selection signal.
* ZSRT C's online train falls to 0.0660 and its 2k snapshot to 0.0311, while
  valid stays ~0.104–0.105 after epoch 300. Its best-valid epoch (315) is
  before the final epoch, and the terminal train improvement does not produce
  a valid improvement.
* The soup has the lowest full-train MAE for P3 (0.0018–0.0019) but is not the
  best-valid object; this is exactly the old-protocol trap the new selector
  removes.

## 6. Budget ledger

| item | GPU-h |
|---|---:|
| P3 seed 0 (360 epochs) | 0.201 |
| P3 seed 1 (360 epochs) | 0.214 |
| ZSRT A_main seed 0 | 0.954 |
| ZSRT B_lr_low seed 0 | 0.933 |
| ZSRT C_wd_high seed 0 | 1.007 |
| ZSRT D_no_rrwp seed 0 | 0.825 |
| smoke runs (P3 + ZSRT batch 128/256) | 0.005 |
| **total measured** | **4.138** |
| cap | 8.000 |
| unused | 3.862 |

The measured `gpu_hours` field is the training-loop wall; end-of-run fixed
evaluations add <0.01 h per run and are not separately billed. The round did
not approach the cap; the limiting factor was the pre-declared stop rule, not
GPU budget.

## 7. Failure taxonomy

| class | finding |
|---|---|
| implementation bug | **Found and fixed:** P3's first calibration used `floor(10000/128)=78` steps/epoch, 360 updates short of the frozen 28,440-update cosine. Corrected to `ceil()=79`; both seeds rerun; old runs archived. No bug found in ZSRT, data, RRWP cache, selector, or metric. P3 fixed-object valid MAE replays to <1e-7 and all P3 input hashes match. |
| budget | The paper GRIT recipe (2,000 epochs / 626k updates, ~20.5 A100-h/seed) was not purchased. The 360-epoch / 28,440-update opportunity cost 0.95–1.01 h per ZSRT run and 0.20–0.21 h per P3 run. ZSRT is underfit at this budget (train MAE 0.031–0.049 for A/B/C and 0.065 for D, vs P3 0.011), but its valid tail is flat, so “just add epochs” is not supported by the curve; the architecture's sample efficiency is the issue. |
| optimization / selection | A_main (lr 1e-3) beats B_lr_low (3e-4) by 0.0364; C_wd_high (wd 1e-4) beats A by 0.0087; best-valid selection chose C. The screen moved the result but could not close a 0.024 gap. No test was used; only the screening seed exists. |
| generalization | The decisive gap: C 0.103363 vs P3 0.079105 (+0.024258, ~14× the P3 seed std). RRWP removal costs 0.0818, so structural relations matter; the full-attention composition over RRWP is not competitive at 28,440 updates. This is a real performance shortfall of this family/budget, not a data leak or metric artifact. |

## 8. Test status and historical exposure

* `zpp_score.py` was **not** run. `predictions/test_read.lock` and
  `analysis/test_scores.json` do not exist. No test prediction or test MAE was
  computed this round.
* The mandatory pre-training data audit did read the official test split once
  to verify its size (1,000) and label scale/hash, as required by the brief.
  It produced only aggregate audit statistics; the test split was never used in
  a loss, selector, hyper-parameter choice, or model comparison. This is
  recorded as an erratum to PROTOCOL §5; the wording “test labels are read only
  in terminal scoring” was too strong.
* Historical P3 test MAE **0.069304 / 0.070340** comes from the earlier
  phase2b round, where the test split was already exposed. It is disclosed for
  context only; it is not a new-protocol result and was not re-read here.
* Therefore the main table has no test column values. The test split remains
  available for a future, separately pre-registered round.

## 9. Reproduction and resume

Shortest faithful reproduction (local commit `f847a2a`; remote via `rr` only):

```bash
# local: verify the frozen data and CPU tests
uv run python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_data.py audit
uv run pytest -q tracks/structural-sufficiency/zinc_perf_protocol_v1/code/tests

# remote: one process per GPU
rr deploy res
rr run res zpp-p3-s0 --pool res-gpu0 --gpus 1 --time 1:00:00 \
  --result tracks/structural-sufficiency/zinc_perf_protocol_v1/runs \
  -- python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_p3.py --device cuda:0 train --seed 0
rr run res zpp-p3-s1 --pool res-gpu1 --gpus 1 --time 1:00:00 \
  --result tracks/structural-sufficiency/zinc_perf_protocol_v1/runs \
  -- python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_p3.py --device cuda:0 train --seed 1

# four-config screen (run two at a time)
rr run res zpp-zsrt-C-s0 --pool res-gpu0 --gpus 1 --time 2:00:00 \
  --result tracks/structural-sufficiency/zinc_perf_protocol_v1/runs \
  -- python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_zsrt.py --device cuda:0 train --config C_wd_high --seed 0
# A_main / B_lr_low / D_no_rrwp analogously

uv run python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_p3.py guard
uv run python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_report.py
uv run python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_plot.py
```

Continuation: the P3 entry point exists (`zpp_p3.py continue --seed N`) and is
the pre-declared 360 + low-LR path, but it was **not exercised** because the
guard returned `trigger: false` for both seeds. ZSRT `final.pt` stores
`model`, `optimizer`, generator state, epoch, step, best-valid and the full
curve; no ZSRT continuation loader is wired, because resuming the finished
cosine would be a new schedule and a new pre-registration. The `best.pt`,
`final.pt`, `last5/` and `predictions/valid_*.npz` artifacts are present on the
remote checkout and were pulled to the local `runs/` directory.

Retrieve a run's artifacts with:

```bash
rr pull <job-name>          # writes .rr/pulled/<run-id>/result/...
```

## 10. Next decision

**Do not extend ZSRT and do not open its test.** The evidence says the family
is not sample-efficient at the P3-matched 28,440-update budget: C's valid
plateaus at 0.103–0.105 after epoch 300 while train MAE keeps falling, and the
RRWP control shows the relation input is not the problem. Buying seeds 1/2 or
a 720-epoch schedule would spend the remaining 3.86 GPU-h without a
pre-registered reason to expect a 0.024 improvement.

The next round should pre-register **one path-level / compositional family**,
not another side module on P3 and not a longer ZSRT run. The concrete candidate
is the GPM-style direction already scoped in `DESIGN.md`: sampled walk
**path patterns** quantised through a VQ codebook, with a path-level encoder
and a 360-epoch / ~28k-update budget, four seed-0 configs, and the same
best-valid selector. It differs from ZSRT in computation graph (paths, not all
pairs), from P3 in composition (learned paths, not fixed incidence sums), and
from the failed STA/BCCA routes (no P3 reader substrate). If the team instead
wants to falsify the budget hypothesis first, the minimum purchase is one
720-epoch C run (~2.0 GPU-h, seed 0); the flat valid tail predicts it will not
close the gap, so it should be labelled a schedule test, not a performance bet.

## 11. Compliance notes

* Train-only backward passes; standardisation/vocab/SSL are train-only; RRWP is
  label-free. P3 frozen inputs are SHA256-verified in each `train.json`.
* Valid labels were used for selection, as explicitly authorised. Test labels
  were read once by the data audit for scale verification; no test prediction
  or test MAE was produced. See §8 and the PROTOCOL erratum.
* ZSRT is 380,161 parameters (D: 377,473), below the 500k cap; P3 is the
  413,505-parameter frozen reference. No external pretraining for ZSRT.
* The ZSRT screen is seed-0 only because the stop rule fired; the report does
  not present a seed mean or a test number for it.
* The old rounds' numbers and access logs were not rewritten; the historical
  test exposure is disclosed rather than hidden.
