# PROTOCOL — LCRC PCBA continuation-recipe round v1

Round id: `lcrc-pcba-continuation-recipe-v1`
Dataset: `ogbg-molpcba` (official OGB scaffold split, 128 tasks, official macro AP)
Seed: `0` only
Code root: `tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/`

This protocol is frozen before the first formal continuation run.  No arm,
learning-rate endpoint, evaluation epoch, selection rule or promotion rule may
be changed after seeing Validation AP.  The official Test targets are not read
until every training run, the Stage-2 promotion decision and the final
Validation selection have been written and sealed.

## 1. Question and scope

The primary question is whether the three existing LCRC PCBA models still have
usable performance gains after their 40-epoch cosine schedule, and whether the
40-epoch cosine endpoint (`1e-5`) is limiting further optimization.  The
secondary question is the official Test performance of the frozen models.

This round is **not** an architecture search, does not add seeds, does not
change the model, and does not train a fresh 80-epoch model.  Recipe R is a
learning-rate restart from an already trained epoch-40 checkpoint, not a
correct 80-epoch from-scratch recipe.

## 2. Frozen source checkpoints

All paths are relative to the repository root.  Every continuation run starts
from exactly one of these epoch-40 checkpoints.

| Model | Source checkpoint | Historical Valid AP | SHA256 |
|---|---|---|---|
| A (attention pooling, no edge MP) | `tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_A/checkpoint_epoch40.pt` | 0.24635609609821457 | `f110935a6055f3c087f9d0b40580506a512cdeb09f6af67a56bde787cc93a437` |
| G (graph context, no bond MP) | `tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/runs/phaseC_seed0_G/checkpoint_epoch40.pt` | 0.2508481135161762 | `e363e3729629f5799f3e2e74ae091ec21e273565e542d939a6bef9bab16849bb` |
| V (bond MP + graph context) | `tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/runs/phase3_seed0_V/checkpoint_epoch40.pt` | 0.25777232477872586 | `2d75e4449cc1be669c779fa11b6d9842923965654c880290c430c64e698939a0` |

Frozen C1 reference (Test only, never continued):

| Model | Source checkpoint | Historical Valid AP | SHA256 |
|---|---|---|---|
| C1 | `tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/runs/phase3_seed0_C/checkpoint_epoch40.pt` | 0.23194477365845528 | `43ff839505eebcae58cf0a3789f1ee519a958fb1456d779a813c8ca487ea0f4e` |

A/G are `NMPModel` checkpoints and V is an `SGModel` checkpoint.  They are
never mixed.  The source run.json files must report `COMPLETED`, the state
dicts must load bitwise, the AdamW moments and 109,520 per-parameter steps must
restore exactly, and the epoch-40 Validation AP must replay within `1e-4`.

Frozen data identities (inherited from `lcrc_pcba_structural_gain_sprint_v1`):

- feature spec `7270f767…a515ee`
- Train-only normalization `6ce50f98…293fb`
- split partition `c872bd63…7bf4f`
- label artifact `5c3b5fbe…27d01` (all official Test rows NaN)
- cache manifest `caad3e90…59035`

Official Test targets are unreachable from the training loader.  The loader
refuses any finite label on an official Test row.

## 3. Learning-rate recipes

Both recipes start from the same epoch-40 checkpoint of the same model and
restore the same model parameters, AdamW first/second moments, step count,
weight decay, betas and eps.  The optimizer is not reset.  The only intended
difference is the learning rate as a function of the global optimizer step.

### Recipe L — low fixed learning rate

For every step `s >= 109520`:

```
LR_L(s) = 1e-5
```

### Recipe R — warm learning-rate restart

Step indices are zero-based and counted over all 40 prior epochs, so
`S40 = 109520`, `S60 = 164280`, `S80 = 219040`, and one epoch is 2,738 steps.

- Epoch 41, steps `109520..112257`:

```
LR_R(s) = 1e-5 + (5e-5 - 1e-5) * (s - 109520) / (2738 - 1)
```

  The first step is exactly `1e-5`; the last step of epoch 41 is exactly `5e-5`.

- Epochs 42–80, steps `112258..219039`:

```
frac = (s - 112258) / (219040 - 112258)
LR_R(s) = 1e-5 + (5e-5 - 1e-5) * 0.5 * (1 + cos(pi * frac))
```

  The first epoch-42 step is `5e-5`; the last epoch-80 training step is
  `1.0000000009e-5`; the schedule endpoint at step 219040 is exactly `1e-5`.

The schedule is a deterministic function of the global optimizer step and is
never modified after viewing Validation AP.  At epoch 60, Recipe R is still at
about `3.08e-5`; both recipes reach `1e-5` only at epoch 80.

## 4. Fixed training configuration

| Parameter | Value |
|---|---|
| Seed | 0 |
| Batch size | 128 |
| Optimizer | AdamW, restored moments |
| Betas | (0.9, 0.999) |
| Eps | 1e-8 |
| Weight decay | 1e-3 |
| Dropout | 0.10 (original model definition) |
| Precision | FP32 |
| Loss | observed-entry masked BCE |
| Gradient clip | 5.0 |
| Data | complete official Train split |
| Main metric | official Validation macro AP |
| Architecture | unchanged |
| Early stopping | none |

Prohibited: re-initialising the model, resetting AdamW, positive-class
weighting or focal loss, changing S1/S2, widening the model, adding message
passing or attention modules, changing batch size or split, or modifying a
historical checkpoint.

## 5. New epoch 41–80 batch order

The historical 40-epoch plan need not define orders after epoch 40.  This round
therefore pre-generates a new order plan for epochs 41–80:

- dedicated CPU `torch.Generator().manual_seed(0)`;
- 80 `torch.randperm(len(train_rows))` draws; epochs 1–40 are unused, epochs
  41–80 are used and renumbered;
- the plan is independent of the global torch/dropout RNG;
- L and R for the same model share the identical plan and plan SHA256;
- every epoch is a complete permutation of the official Train rows;
- no Validation or Test row appears;
- the plan summary and isolation evidence are written before training.

This is not claimed to be a bitwise recovery of a hypothetical original
80-epoch trajectory.  The pre-generated plan is frozen for both L and R.

## 6. Stage 0 — correctness gates

Before formal training, for each of A, G and V:

1. Source checkpoint exists, SHA256 matches, file is complete.
2. State dict loads bitwise into the matching model class and parameter count.
3. AdamW parameter ids, state tensors, hyperparameters and step count restore.
4. Epoch-40 Validation AP replays within `1e-4`.
5. Before the first optimizer step, L and R have identical model parameters and
   optimizer moments.
6. L and R use the same epoch-41..80 batch order and the same first batch.
7. L and R restore the same RNG state; the dropout stream starts identically.
8. The new order plan does not advance the global RNG and its SHA is recorded.
9. Every plan epoch is a Train-row permutation; no Test row is present.
10. With the same LR, a paired first step produces the same parameters and
    optimizer state (CPU bitwise where possible; audited float32 reduction
    tolerance otherwise).
11. Feature cache, labels, normalization, split and official AP implementation
    match the frozen identities.
12. Loss, gradients and logits are finite in the smoke and first-step probes.
13. The first-step loss is reproducible under the same restored RNG and batch.
14. The full training code commit is recorded.

A short GPU smoke (20 steps per model) measures step time, peak memory and
projected GPU-hours.  If any gate fails, formal training does not start.

## 7. Stage 1 — epoch 40 to 60

Formal matrix, six runs:

| Model | Recipe L | Recipe R |
|---|---|---|
| A | A-L | A-R |
| G | G-L | G-R |
| V | V-L | V-R |

Each run adds 20 epochs.  At most two GPUs run concurrently.

Fixed evaluation points: `epoch40 / epoch50 / epoch60`.  Epoch 40 must be a
replay.  Epoch 50 is a diagnostic.  Epoch 60 is the fixed primary comparison
for the two recipes.  A better-looking intermediate epoch is never substituted
for the primary comparison.

For every model and recipe record: official Valid AP, per-task Valid AP, Train
masked BCE, Valid masked BCE, epoch-60 Train AP, Train−Valid gap, per-epoch LR,
gradient norm/clipping, step time, GPU-hours and a complete epoch-60
checkpoint.

Primary within-model comparisons:

```
D_L  = AP_L60 - AP40
D_R  = AP_R60 - AP40
D_RL = AP_R60 - AP_L60
```

Pre-registered labels:

- `RESTART_PROMISING`: `D_RL >= 0.003` and R improves at least 70 of the 127
  scorable tasks relative to L.
- `LOW_LR_PROMISING`: `D_RL <= -0.003` and L improves at least 70 scorable
  tasks relative to R.
- `NO_CLEAR_RECIPE_WINNER`: neither condition holds, but at least one recipe
  improves by `>= 0.002` over epoch 40.
- `NO_CONTINUATION_GAIN`: both recipes fail to improve by `0.002` over epoch 40.

These are exploratory screening labels, not significance tests.

## 8. Stage 2 — conditional epoch 60 to 80

For each model, a recipe is eligible only if:

- `AP60 >= AP40 + 0.002`;
- `AP60 >= AP50 - 0.001`;
- all correctness checks pass and the Stage-1 run is complete.

If any recipe is eligible, the model may continue for 20 more epochs from its
complete epoch-60 checkpoint (model, AdamW, schedule, RNG, consumed steps and
the frozen epoch-61..80 order plan).  Among eligible recipes, choose the one
with the higher epoch-60 AP; a difference below `1e-4` selects L.  If the
higher-AP60 recipe failed the late-stability gate while the other recipe
passed, the passing recipe is selected.  This interpretation is fixed here,
before any Stage-1 result exists.  No model may run more than one recipe in
Stage 2.

Fixed evaluation points: `epoch70 / epoch80`; report the full
`AP40/50/60/70/80` trajectory.  Convergence labels:

- `LATE_GAIN`: `AP80 - AP60 >= 0.003`.
- `SMALL_LATE_GAIN`: `0 < AP80 - AP60 < 0.003`.
- `PLATEAU_OR_REGRESSION`: `AP80 <= AP60`.

The last-ten-epoch delta is `AP80 - AP70`.  No round may continue to epoch 100.
A model stopped at epoch 60 is never presented as an epoch-80 result.

## 9. Validation-frozen final selection

After all training and Stage-2 decisions, and before reading any official Test
label, select one final model for each of A, G and V from:

- the original epoch-40 checkpoint;
- Recipe L epoch-60 checkpoint;
- Recipe R epoch-60 checkpoint;
- the selected recipe's epoch-80 checkpoint, only if Stage 2 ran.

Epoch 50/70 checkpoints are not final candidates.  Selection uses official
Validation macro AP:

1. highest Validation AP;
2. if within `1e-4`, choose the candidate trained for fewer epochs;
3. if the epoch is the same and the difference is within `1e-4`, choose L;
4. never mix parameters across models;
5. never use Test.

The fixed epoch-60 recipe comparison and the final Validation-selected result
are reported separately and never merged into one conclusion.

## 10. One-time official Test evaluation

The Test split is opened only after all training, Stage-2 promotion and
Validation selection are frozen.

Prediction models, fixed in advance:

1. C1 epoch 40;
2. A epoch 40;
3. G epoch 40;
4. V epoch 40;
5. A final Validation-selected checkpoint;
6. G final Validation-selected checkpoint;
7. V final Validation-selected checkpoint.

If a final model equals its epoch-40 model, it is recorded as an alias and is
not scored twice.  No additional seed, ensemble or Test-chosen epoch may be
added.

Freeze procedure before reading Test labels:

1. replay Validation AP for every final model;
2. verify checkpoint, cache and model identities;
3. generate logits for the official Test rows;
4. save row ids, logits, shape, dtype and SHA256;
5. seal a prediction manifest with a canonical SHA256;
6. only then read `data/ogb/ogbg_molpcba/raw/graph-label.csv.gz` and verify
   that its Train/Valid blocks equal the frozen label artifact.

Scoring uses `ogb.graphproppred.Evaluator("ogbg-molpcba")` and an independent
per-task `sklearn.average_precision_score` recomputation with NaN masking.
Test results are descriptive external evaluation only.  They cannot change a
model, recipe, epoch or selection, and the Test-best epoch is not reported as
the "best model".

Prior PCBA rounds in this repository did not read official Test targets.  This
round records that history; after this round the Test split is exposed, and
future model iteration on the same Test must disclose this exposure rather than
claim a fresh blind confirmation.

## 11. Budget and failure handling

- Host: `res`; pools: `res-gpu0`, `res-gpu1`.
- Target GPU: `<= 7` A100 GPU-hours; hard limit `<= 8`.
- Target wall clock: `<= 5 h`; hard limit `<= 6 h`.
- At most two GPUs concurrently.
- Failures, smoke and Test inference count against the budget.
- All formal runs start from a traceable Git commit via `rr doctor/deploy/run`.
- No direct SSH/nohup, no Test-based early stopping, no extra seeds.
- If a stage cannot be completed within budget, it is stopped and an incomplete
  group is never presented as a formal paired comparison.

## 12. Deliverables

`PROTOCOL.md`, `README.md`, `code/`, `tests/`, `REPORT.md`, and:

- `analysis/source_checkpoint_audit.json`
- `analysis/paired_initialization_audit.json`
- `analysis/lr_schedule_audit.json`
- `analysis/epoch60_recipe_comparison.json`
- `analysis/epoch80_continuation.json` (present even if Stage 2 is not run)
- `analysis/validation_selection.json`
- `analysis/test_prediction_freeze.json`
- `analysis/test_results.json`
- `analysis/per_task_valid_test.csv`
- `analysis/gpu_ledger.json`
- `analysis/run_registry.json`

Each run directory contains its curve, Validation curve, fixed-epoch reports,
complete checkpoint, optimizer state, RNG state, per-task AP and resource log.
Large prediction/checkpoint files remain Git-ignored.
