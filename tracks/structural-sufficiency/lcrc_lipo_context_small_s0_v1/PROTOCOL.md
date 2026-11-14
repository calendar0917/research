# PROTOCOL — lcrc-lipo-context-small-s0-v1

Small-data pilot on **ogbg-mollipo** (Lipophilicity). Five context arms on the
frozen C1 backbone plus one ready-made D-MPNN reference, single seed, official
scaffold split. Purpose: a cheap decision on whether *local-state encoding plus
neighbourhood state exchange* buys a prediction increment worth continuing, and
whether it repairs a locatable structural failure. Nothing here is PCBA
evidence and nothing here is a model-selection exercise.

Frozen before any formal run; the file is committed together with the code.

## 1. Data and split (locked)

* Dataset `ogbg-mollipo` from OGB; archive `lipophilicity.zip`
  sha256 `c58d285a55af5d383483a8c9167030e2beea94041ddbf69c39d2a193548a25d0`.
* Split read **verbatim** from `GraphPropPredDataset('ogbg-mollipo').get_idx_split()`:
  Train 3360 / Valid 420 / Test 420, mutually disjoint, union 4200
  (`split_v1.json`, sha256 recorded in `data/manifest.json`).
* No `random_split`, no re-derived scaffold split. `Lipop.random122.ind.pkl`
  (ChemBench random split) is **not** the official split and is not used here.
  Old random122 checkpoints are background only.
* Row identity is frozen by `data/labels_v1.npz` (4200 labels),
  `data/split_v1.json` and `data/raw_attributes_v1.json.gz`
  (label-free graph attributes). Feature construction never reads labels;
  `adapters.lipo_row_to_graph` rejects label-carrying rows.
* Target: raw logD. Normalisation uses Train only (`train_mean=2.16216`,
  `train_scale=1.21154`); no Valid/Test statistic enters any fit.
* D-MPNN features (`data/dmpnn_features_v1.npz`, 72 atom / 14 bond dims) are
  generated once, locally, with RDKit and committed; the cluster has no RDKit.

## 2. Arms

| arm | state passed into the layer-2 message | source hops |
|---|---|---|
| **B** | none (frozen C1 backbone + readout) | 0 |
| **S** | the target node's own layer-1 state (no bond term) | 0 |
| **R1** | `h0` of the neighbour (the R1 recipe) | 1 |
| **R2** | fixed extended source `c_j = (h0_j + Σ_{u∈N(j)} h0_u) / (1+deg_j)` | 2 |
| **D2** | the neighbour's **learned** layer-1 state `h1` | 2 |
| **G** | D-MPNN reference (chemprop v2.2.1, hidden 300, depth 3) | – |

All context arms share one base: the frozen C1 (`NO_BOND_MEAN`, 51/301 input
encoders, 128/96 widths, dropout 0.10) with `y = Σ_v f(h_v) + ρ(·)`. Two context
layers, width 64, private CPU generator `seed=17`, and a **zero-initialised**
`fc2` so that every arm's epoch-0 prediction is bit-identical to B
(max |Δ| = 0 measured). Parameters: B 228674, S/R1/R2/D2 262978
(+34304 each: 896 bond projection, 2 × 16704 message blocks), G 318301.

## 3. Recipes (frozen, reused unchanged)

* C1 family: the frozen R3 recipe — AdamW, peak 3e-4, floor 1e-5, wd 1e-3,
  batch 128, dropout 0.10, clip 5.0, 360 epochs, per-step cosine over the
  planned epoch budget, `shuffle_offset = 917000`.
* Shared epoch plan: `torch.Generator(train_seed + 917000)`, one `randperm`
  per epoch, written to `epoch_plan.npz`; every arm consumes the same plan.
* G: the recorded E4-corrected D-MPNN schedule — Adam, batch 64, 100 epochs,
  init 1e-4, peak 1e-3, floor 1e-4, wd 0, clip 5.0, 2 warmup epochs,
  exponential decay (Noam warmup + exponential floor, `LambdaLR`).
* One seed only: `seed = 0`.

## 4. Evaluation and selection

* Every epoch evaluates **all three splits in full** (Train/Valid/Test),
  under `inference_mode`, with RNG snapshot/restore and independent iterators,
  so evaluation never perturbs the training stream (verified bitwise in
  preflight `d`/`e`).
* Checkpoint selection is **Validation-only**: lowest Valid RMSE, strict
  improvement, ties keep the earlier epoch. Test is never consulted for
  selection.
* Test is a *development-evaluation* split for this round: the round is
  explicitly authorised to read it, and each read is appended to
  `analysis/test_access_log.json` (stage, UTC time, purpose, n rows, rule).
  No historical freeze is invented and no global guard is modified.
* The late window is defined per arm as the last 20 epochs (G: last 10) and
  reported as the mean of the per-epoch metrics — not a prediction ensemble.

## 5. Pre-registered judgements

Relative improvement `rel = (RMSE_control − RMSE_candidate) / RMSE_control`.

An arm pair is **PROMISING_THIS_TASK** only if *all* hold:

1. Test RMSE rel ≥ 2% at the best-Val point,
2. Valid RMSE rel ≥ 2% at the best-Val point,
3. MAE not worse by more than 1% on Valid and on Test,
4. last-epoch Test rel > 0 and late-window Test rel > 0,
5. the Test gain survives deleting the control's worst 1% absolute errors,
6. the layer-2 source channel is demonstrably used (preflight `c`).

Otherwise exactly one of: `FIT_ONLY` (Train gain only), `TAIL_SPECIFIC`
(gain concentrated in the tail), `SELECTION_OR_SPLIT_SENSITIVE` (signs of
Valid/Test/last/late disagree), `NO_CLEAR_INCREMENT_THIS_TASK`.

Supporting numbers, always reported: MAE, per-molecule absolute-error
difference (mean/median, win rate), tail sensitivity at 1% and 5%, and a
500-replicate Murcko-scaffold cluster bootstrap (Exp(1) weights shared by all
models in a pair, paired deltas, seed 20261114). The interval describes sample
uncertainty of fixed models; it is *not* training-seed variance.

Arm-level label (mechanical, in this order):
`LEARNED_CONTEXT_CANDIDATE_ON_LIPO` (D2 vs S and D2 vs R2 both promising and
D2 vs R2 separated) → `FIXED_CONTEXT_COMPETITIVE_ON_LIPO` (a neighbour arm
promising, D2 not separated from R2) → `NO_SEPARATED_NEIGHBOR_GAIN_ON_LIPO`
(D2 not separated from S) → `WEAK_CHANNEL_THIS_CONFIG`. "Separated" = the
paired Test interval excludes 0 **and** the relative improvement reaches 2%.

## 6. Structural groups (defined before prediction, selected on Validation)

Environment support = RDKit Morgan count fingerprint, radius 3 with redundant
environments; Train document frequency with leave-one-molecule-out.

* **H1** — a root atom whose radius-2 environment has Train support 0.
* **H2** — radius-2 support ≥ 5 while every radius-3 environment of that root
  has support 0.
* **H3** — atoms of the SMARTS `[NX3]-[CX3](=[OX1])` (amide-like fragments).

A root without a valid radius-2 environment is reported as `MISSING`, never
silently counted as unseen. Group MAE gap is size-standardised (Train heavy-atom
tertiles, Train marginal bin weights), estimable only with ≥ 40 group and ≥ 40
non-group molecules and ≥ 2 common bins with ≥ 10 molecules per side. The group
is chosen on **B's best-Val point** (largest estimable gap ≥ 0.05 logD) and
frozen before Test is read; if nothing qualifies the honest label is
`NO_CLEAR_FAILURE_GROUP`. All groups are reported for every arm with counts,
cluster intervals and paired D2-vs-control gap deltas.

## 7. Source interventions (frozen D2, `best` and `last` checkpoints)

On the complete Valid and Test splits: `NORMAL`; `REUSE_ALL` (layer-2 source =
`h0` for every edge, R1 semantics); `FIXED_ALL` (source = fixed `c_j`, R2
semantics). If a failure group was frozen, additionally `REUSE_TARGET`
(replace the source only for edges entering the group's roots) and
`REUSE_CONTROL` (the same for degree-matched control roots *inside the same
molecules*, selected by a fixed SHA-256 order that never looks at predictions
or labels). The identity `Δpred = Δcontrib + Δρ` is checked numerically.
These are internal edits of a frozen prediction function; they are never
reported as a retraining gain.

## 8. Budget and stop rule

Soft 1.5 GPU-hours, hard 2.5 GPU-hours for the round. Only `seed=0`, only the
official split, only the six arms above: no PCBA, ZINC, multi-seed, multi-split,
pretraining or additional dataset work. If the total projected cost exceeds the
hard limit the round stops and reports what exists.

## 9. Reproduction

`code/lp_runner.py` compares the frozen constants against the YAML config on
every invocation and refuses to run on drift. `lp_preflight.py` sections a–f
must pass before training; `tests/test_lp_round.py` re-checks the data identity,
the shared initial state, the parameter budget, the epoch-0 equivalence, the
source reach and the shared epoch plan.
