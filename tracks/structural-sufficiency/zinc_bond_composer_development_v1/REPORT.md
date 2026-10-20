# REPORT — zinc bond-composer development v1

**Protocol:** `zinc-bond-composer-development-v1` (internal development probe).
**Status:** completed; negative development decision.
**Decision label:** `B2_DIRECTIONAL_BUT_BELOW_DEV_LINE`.
**Bottom line:** B2 is better than both B0 and B1 on the fixed soup, but its
B2-vs-B0 Monitor gain is 0.001775 MAE, below the pre-agreed 0.003 line. There is
no practical single-seed development signal. Stop this candidate; do not add
seeds, modules or budget inside this round.

## 1. What was tested

The full original P3 reader (321→256→256→1) was retained. Only the shared
per-bond composer output `c_e` was modified by one pre-LN block and a zero-init
output projection, then aggregated as `sum(c'_e)` instead of `sum(c_e)`.

| model | branch | params |
|---|---|---:|
| B0 | original P3, no branch | 413,505 |
| B1 | per-bond FFN capacity control (`gamma ⊙ FFN_extra`) | 441,825 |
| B2 | per-bond 4-head key self-attention (head dim 12) | 441,825 |

All three used the frozen phase1b Fit-8000 / Monitor-1000 split, the phase1e
Fit-only objects and the seed-0 Fit-only masked-SSL encoder. Same rows, same
R3 recipe: batch 128, raw-y L1, Adam 1e-3 with per-step cosine to 1e-5,
weight decay 1e-5, clip 5, 360 epochs = 22,680 updates, last-5-epoch parameter
mean soup. No new SSL, no 10k soup initialiser, no early stopping, no official
valid/test or Reserved read.

## 2. Integrity checks

`prep` and `gates` passed before training; local pytest 4/4 and ruff clean.

* Frozen hashes matched: split `fbca5fd5…`, Fit rows `7f66a23e…`, Monitor rows
  `dc6cc4e9…`, objects `bfbf9653…` / `7a718506…`, SSL `0d045e60…`,
  standardiser `b00add…`.
* Canonical P3 post-SSL trunk `fe1b6df…` is bit-identical across B0/B1/B2;
  B1/B2 common branch modules hash `81cffcc8…`; reader first weight is `[256,321]`.
* B1/B2 parameter groups: input 4,656 + block 18,960 + output 4,704 = 28,320;
  B1 `FFN_extra` 9,360 + `gamma` 48 = B2 MHA 9,408.
* G1–G7 all passed: zero-branch outputs equal B0 exactly (max abs 0.0);
  atom-renumber / bond-order / batch-composition invariance max abs 1.14e-5,
  9.54e-6 and 0.0; empty-bond outputs finite; step-1 output-projection gradient
  present and step-2 branch-internal gradients non-zero; Monitor labels only in
  evaluation; batch-order SHA-256 equals frozen P3
  `d142e307…` for all three formal runs.

## 3. Fixed-soup results

Primary metric: raw-y MAE on Monitor-1000, fixed last-5 soup. Fit MAE is shown
for completeness.

| model | params | soup Fit | soup Monitor | Monitor RMSE | best Monitor (epoch) | step mean | process wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 413,505 | 0.001093 | **0.088506** | 0.270283 | 0.088411 (300) | 24.86 ms | 584.4 s |
| B1 | 441,825 | 0.001445 | 0.089253 | 0.269294 | 0.089060 (320) | 34.81 ms | 825.5 s |
| B2 | 441,825 | 0.001205 | **0.086731** | 0.266930 | 0.086556 (300) | 30.81 ms | 720.3 s |

Contrast sign convention: `MAE(left) − MAE(right)`; positive means the right
model is better.

| contrast | all Monitor | ordinary `y > -3` (n=918) | drop B0-worst-10 (n=990) |
|---|---:|---:|---:|
| B0 − B1 | −0.000747 | −0.000682 | −0.001017 |
| B0 − B2 | **+0.001775** | +0.002442 | +0.001386 |
| B1 − B2 | +0.002522 | +0.003125 | +0.002403 |

B1, the equal-parameter FFN capacity control, is slightly worse than B0 on all
three slices. B2 is directionally better than B0 and better than B1, and both
pre-declared robustness slices are positive for B2. But the B2-vs-B0 gain is
0.001775, below 0.003. The best-epoch observation gives B0−B2 = 0.001855, also
below the line, so this is not a soup-versus-best-epoch selection artefact.

## 4. Decision

Pre-agreed development follow-up signal:

| condition | result |
|---|---|
| B2 vs B0 Monitor gain ≥ 0.003 | **false** (0.001775) |
| B2 better than B1 | true (0.002522) |
| ordinary-region gain positive | true (0.002442) |
| drop-B0-worst-10 gain positive | true (0.001386) |
| **worth follow-up** | **false** |

The attention-specific B1−B2 contrast is positive, but the absolute B2−B0 gain
is too small for this round's practical line. This is not evidence that
attention is generally useless, and it is not a novelty claim. It is also not a
reason to promote B2, run more seeds, or tune heads/depth/initialisation in this
round. The correct next action is to stop this candidate.

## 5. Budget and deviations

| item | seconds | hours |
|---|---:|---:|
| 2-epoch smoke (initial gate) | 6.0 | — |
| 5-epoch timing probe | 7.9 | — |
| B0 formal | 584.4 | — |
| B1 formal | 825.5 | — |
| B2 formal | 720.3 | — |
| **process-wall total** | **2,144.2** | **0.5956** |
| conservative `rr` scheduler wall | 2,184 | 0.6067 |

Cap: 0.75 A100-hours. Remaining: 0.1544 h (process wall) or 0.1433 h
(conservative scheduler wall). Evaluation every 20 epochs and final soup
evaluation are included; there were no failed formal runs or retries. Both
smoke attempts are counted.

**Gate deviation (disclosed).** The original 2-epoch smoke projected 0.7567 h
and returned `STOP_BUDGET` (24 s over). Because 16 updates are dominated by
warm-up, one timing-only 5-epoch probe was run on the same rows and models with
the same 1.20 margin + 300 s reserve; it projected 0.7483 h and passed. No
architecture, data, recipe, epoch count or cap was changed. Both records are
retained (`smoke.json`, `smoke_probe_5ep.json`); the deviation is documented in
`TIMING_AMENDMENT.md`. A strict reading of the first record would have stopped
before formal training.

**Wall-clock correction.** B0/B1 recorded a negative `wall_seconds` because the
first two runs mixed `time.perf_counter()` with `time.time()`. The correct UTC
start/finish timestamps in the same records are used in the analysis and ledger;
training, parameters and predictions are unaffected. B2 used the fixed clock.
See `wall_clock_correction.json`.

## 6. Boundary

Single seed, repeatedly-used Monitor-1000 development set, no blind or
independent confirmation, no official valid/test, no Reserved labels. This round
does not reopen S/T/R screening, does not run GRIT, and does not modify any
frozen phase1b/phase1e/phase2a product. The result only decides whether this
particular bond-composer candidate deserves further verification; it does not.
