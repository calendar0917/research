# Recipe tuning v1 — note

**Status: PREREGISTERED 2026-10-13 (frozen before any GPU run).**
Protocol: `cscl-bjoc-recipe-tuning-v1`
Config: `configs/bjoc_recipe_tuning_v1.yaml`
Driver: `code/run_bjoc_recipe_tuning_v1.py` · Tests: `tests/test_bjoc_recipe_tuning_v1.py`

## Question

Holding the validated masked-pretrained BJOC-Null P completely fixed except
for the supervision-phase recipe, do the five pre-declared recipes change
official-valid raw-y MAE relative to R0 on the same seed and split?

| ID | Epochs | LR schedule | Weight decay |
|---|---:|---|---:|
| R0 | 240 | constant 1e-3 | 1e-5 |
| R1 | 240 | cosine 1e-3 → 1e-5 (per step, no warmup) | 1e-5 |
| R2 | 360 | constant 1e-3 | 1e-5 |
| R3 | 360 | cosine 1e-3 → 1e-5 (per step, no warmup) | 1e-5 |
| R4 | 240 | constant 1e-3 | 1e-4 |

R0 is the historical P and is **never retrained**: its frozen checkpoint and
frozen official-valid prediction file are replayed and hash-verified.  R1–R4
are single-variable perturbations of the same P state.

## Hypotheses

* H1 — the fixed constant LR limits late optimisation; cosine decay helps.
* H2 — 240 supervised epochs under-train; 360 epochs help.
* H3 — cosine decay and longer training interact.
* H4 — 10x weight decay improves generalisation at fixed length.
* Null — none of R1–R4 robustly improves; `NO_ROBUST_RECIPE_GAIN` is an
  allowed and reportable outcome.

## What is held fixed (matching constraints)

* BJOC-Null architecture, 413,505 parameters, no new module (audited).
* Same full-10k masked-SSL encoder state, byte-for-byte
  (`pretrain_encoder_s{0,1}.pt`; the pretraining round is not rerun).
* Same scratch init for composer/reader and same scratch encoder init as the
  recorded P seed (hash equality with `train_P_seed{s}.json` and with the
  frozen config).
* Same official-train 10,000 rows and same raw-y labels; same batch size 128,
  raw-y L1, grad clip 5, fp32, no dropout, no AMP, no encoder/head LR split.
* Same per-seed batch order: the first 240 epochs of every candidate have the
  same `torch.Generator(seed + 101)` draws as P (prefix hash checked against
  an independent recomputation and against P's recorded batch-order hash).
* Same last-5-epoch parameter soup — always the final five epochs of the
  candidate's own length (356–360 for 360 epochs), never a reused window.
* No target transform, calibration, clipping, tail weighting or ensemble.

## Screening (seed 0, official valid)

Official valid is an exposed development set.  Each candidate is compared
paired-per-molecule against R0 on the same seed 0.  Gate:

1. seed-0 valid MAE delta ≤ −0.003;
2. ordinary (y > −3) MAE delta ≤ 0;
3. delta excluding `valid:0172` ≤ −0.002 (the gain is not carried by the one
   extreme row);
4. numerics/audit gates pass (finite loss and gradients, parameter / init /
   batch-order audits).

If several candidates pass, the winner is the lowest seed-0 valid MAE; ties
go to fewer supervision updates.  If none passes, the GPU search stops and no
seed 1 is purchased.  No sixth candidate, no re-runs to chase a number.

## Confirmation (unique winner, seed 1)

One seed-1 training of the selected candidate only, compared against the
frozen P seed 1.  Final gate: both seed deltas < 0, seed-mean delta ≤ −0.003,
ordinary seed-mean delta ≤ 0, top-10-removed delta < 0, extreme-row-excluded
delta < 0, no numerical/evaluation anomaly.  `RECIPE_GAIN_CANDIDATE` requires
the paired molecule bootstrap CI95 upper bound < 0; a CI crossing zero is
`WEAK_OR_INCONCLUSIVE`, never a confirmed gain.  Because the candidate was
chosen on the same exposed valid set, the seed-1 check and its CI are **not**
an independent blind test.

## Data / label discipline

* Training reads only official-train raw y and structures.
* Prediction forwards read only official-valid structures (label-blind).
* All candidate prediction files are hashed and frozen before the single
  official-valid label read in `score-screening`; `valid_read.lock` records
  the read.
* Official test is never loaded in any form, including for selection.
* No post-read training or retuning; the analysis stage is descriptive.

## Budget

≤ 2.5 GPU-hours on `res` pool `res-gpu1` (GPU1 only).  R0 is replay-only;
exactly one seed-0 training per R1–R4 and at most one seed-1 training for the
pre-registered winner.

## Progress log

* 2026-10-13 — preregistration written; local CPU prep passed
  (input audit all-ok, R0 replay max|Δpred| = 0.0 on both seeds), fast test
  suite 19 passed, local 1-epoch CPU smoke for R1 (cosine) and R4 (wd 1e-4)
  passed (init/encoder/batch-order hashes exact, LR schedule deviation
  ≤ 4e-19).  No GPU run yet.
* 2026-10-13 — **pre-GPU amendment (before any training/GPU trajectory).**
  The remote compute checkout does not carry the git-ignored frozen P valid
  prediction `npz` files (the same infrastructure fact the STRC round
  recorded).  The R0 replay check now verifies a frozen, device- and
  serialisation-independent SHA-256 of the regenerated float32 prediction
  **array** (`valid_P_pred_s{0,1}_array_sha256` in the config), while the npz
  file hash and `max|Δpred|` remain an optional local cross-check when the
  file is present.  Training-input hashes are unchanged and still blocking;
  no candidate, threshold, gate or analysis changed.  Remote `prep` is then
  re-run on `res` (GPU1 pool) before any training.
