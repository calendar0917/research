# PROTOCOL — zinc-perf-protocol-v1

| field | value |
|---|---|
| protocol id | `zinc-perf-protocol-v1` |
| track | `structural-sufficiency` (new, independent round directory) |
| namespace | `tracks/structural-sufficiency/zinc_perf_protocol_v1/` |
| status | `FROZEN_BEFORE_FORMAL_TRAJECTORY` |
| data | official PyG ZINC `subset=True`: train 10,000 / valid 1,000 / test 1,000 |
| metric | raw-`y` MAE (lower is better); no target transform |
| primary selector | per-seed single **best-valid** checkpoint |
| test policy | one-shot terminal read after architecture/config/seed freeze |
| budget | 8.0 A100-hours total (both GPUs summed), reference cap 1.5 h, reserve 0.5 h |

This round replaces the previous *interpretability-first* protocol. It is a
performance-development protocol with an explicit validation budget. It is not
a novelty claim for any structural statistic, message passing, or attention.

## 0. What changed relative to the old rounds

1. The primary model is the **single epoch-end checkpoint with the lowest
   official-valid MAE**, selected independently per seed; ties go to the
   earlier epoch. The last-5 parameter soup is retained only as a secondary
   diagnostic for the old P3 reference and is never the primary number.
2. Official valid may be evaluated every epoch and used for early stopping,
   learning-rate/regularisation/configuration choice. This is an explicit,
   recorded use of valid. The old `valid-read-once` discipline is not reused,
   and old access logs are not rewritten.
3. Raw-`y` MAE is computed on the official 10k/1k/1k ZINC-12k split. Numbers
   from the old internal 8k/1k split are never mixed into the primary table.
4. The new backbone is allowed its own recipe (optimizer, LR, batch, warmup,
   weight decay, dropout, epochs, patience). Fairness is judged by a declared
   compute opportunity, not by identical epochs or identical parameter counts.

Interpretation boundaries retained from audit `592d766`:

* terminal train-MAE descent does not imply valid is still improving;
* the within-epoch running train MAE (pre-update predictions) is **not** the
  eval-mode train MAE of the epoch-end fixed model;
* the last-5 rule is one controlled same-protocol contrast, not proof that
  every family reached its own best generalisation epoch;
* curve shapes at 20-epoch sampling are not read as optimisation mechanisms.

## 1. Data discipline

* Graphs and labels come only from `torch_geometric.datasets.ZINC(root=data/ZINC,
  subset=True, split=...)`. Splits are the official PyG splits; no re-splitting,
  no ZINC-Full, no external molecules, no extra property labels, no target
  decomposition, no `logP`/SA formula inputs.
* Atom/bond categories and label scale are audited from the three official
  splits and recorded in `analysis/data_audit.json` before any GPU run.
* RRWP structural encodings are deterministic, label-free functions of the raw
  graph. They are precomputed once and cached; no parameter is fitted on valid
  or test.
* P3 calibration reuses the frozen train-only masked-SSL encoders and the
  frozen object caches (`objects_train10k.pt`, `objects_valid.pt`) built from
  the same official split. Their SHA256 values are verified before training.
* Valid labels are read from the official valid split in this round. Test labels
  are read only in the terminal scoring step (§5).

## 2. Training and validation mechanics

* Only train rows are back-propagated. No valid/test row enters a loss.
* Per epoch we record: `valid_mae` (epoch-end model, `model.eval()`,
  `torch.no_grad()`), `online_train_mae` (within-epoch running mean of
  **pre-update** predictions), `lr`, wall seconds, and (at every 20th epoch)
  `snapshot_train_mae` (eval-mode MAE on the fixed first 2,000 official train
  rows).
* `best_train_mae` and `final_train_mae` are full-10k eval-mode measurements of
  the fixed best/final checkpoints. The online train MAE is never compared
  against these as if it were the same quantity.
* Validation runs inside `torch.random.fork_rng` (CPU + current CUDA device),
  so it cannot advance the training RNG stream, batch order, scheduler or
  optimizer. Validation uses no dropout and no BN updates.
* Checkpoint selection is `argmin(valid_mae)`; equal scores keep the earlier
  epoch. The selector never builds or reads test data.

## 3. P3 reference calibration

* Model: frozen P3 = `BJOC-Null` + train-only masked SSL + R3
  (413,505 parameters). Initialisation, batch 128, Adam, LR 1e-3 → 1e-5
  per-step cosine over 28,440 updates, weight decay 1e-5, gradient clip 5,
  shuffle offset 101, and batch order follow the frozen R3 recipe.
* Seeds 0 and 1 only. The only additions are per-epoch validation, checkpoint
  saving, soup/last-5 retention, and the eval-mode train snapshots.
* Extract from each trajectory: best-valid single, epoch-360 final, and the
  original last-5 soup. Compare valid MAE and eval-mode train MAE of these
  fixed objects. Historical valid arrays are used only as an alignment sanity
  check; they are never presented as new-protocol best-valid results.

### 3.1 Pre-declared continuation rule

After epoch 360, continue the *same* trajectory for at most 360 additional
epochs **only if both** conditions hold for that seed:

1. the best-valid epoch is `>= 320`, and
2. `valid_mae(epoch 360) <= valid_mae(epoch 320) - 0.001` (a meaningful
   terminal improvement, not train-MAE descent).

If triggered: restore the epoch-360 `final.pt` (Adam moments + epoch),
**freeze the LR at 1e-5**, disable the finished cosine schedule, and stop after
60 epochs without a new best. This is recorded as `360 + low-LR continuation`,
never as a fresh 720-epoch cosine run. The original 360-epoch best/final/soup
artifacts remain separate.

If a seed does not trigger, no continuation is run and no extra seed/width/LR
search is performed.

## 4. New backbone family (ZSRT)

The family is pre-registered in `DESIGN.md` and `configs/zsrt_screen.yaml`.
It is a borrowed-backbone design (GRIT-style RRWP relation input + full
attention), with an explicit statement of what is and is not new. It keeps
explicit atom and bond types, exact connectivity and multiplicities as inputs
until attention integrates them, and performs no iterative local message
passing. It does **not** read the frozen P3 321-D reader input.

* Four pre-declared screen configurations, one screening seed (0):
  `A_main`, `B_lr_low`, `C_wd_high`, `D_no_rrwp` (a structural-relation
  control on the same backbone).
* Selection is by the complete valid curve and best-valid MAE at seed 0.
  Differences below 0.0005 are treated as ties and resolved in favour of
  `A_main`; test, tail subgroups and train loss are not selection criteria.
* If a variant is still improving at the 360-epoch cap, that is recorded as
  budget-limited, not as architecture failure. No extra LR/epoch search is
  opened after seeing valid.
* The selected configuration is then run for seeds 1 and 2 (the screening seed
  participates in the final set and is declared). Primary result is the
  per-seed best-valid MAE and mean ± sample std.
* A gain of ~0.001 or a difference comparable to seed spread is reported as a
  small signal, not a breakthrough.
* Standard parameter budget: <= 500,000 total trainable parameters, counting
  embeddings and any frozen pretraining. No external pretraining is used for
  ZSRT (train-only data only).

## 5. Terminal test evaluation

After the selected ZSRT configuration and the final seed set are frozen in
`analysis/protocol_freeze.json` (checkpoint SHA256s included), the official
test split is evaluated once for each frozen P3 seed and each frozen ZSRT seed.
The test read is a reporting step for already-selected checkpoints: it does not
change architecture, hyper-parameters, epochs, soup rule, or seeds, and no new
seed is added afterwards.

The official ZINC test split has been exposed in earlier rounds. This round
discloses that history; the terminal read is **not** a blind test. Test MAE is
reported per seed and as mean ± sample std, and is never used to choose the
winner between P3 and ZSRT.

## 6. Stop conditions

* P3 is closed after the declared 360-epoch calibration (plus a continuation
  only if §3.1 triggers). It is not tuned further.
* ZSRT stops after the four-config screen, or after the screen plus seeds 1/2
  if a usable config exists.
* If every ZSRT screen config is worse than P3 by more than 0.005 best-valid
  MAE with no improving tail, the round stops at the development conclusion and
  the test step is not opened for ZSRT.
* No automatic follow-up architecture ("another residual/lookup/attention
  block") is authorised by this document. The next decision is stated in the
  report and requires a new pre-registration.
