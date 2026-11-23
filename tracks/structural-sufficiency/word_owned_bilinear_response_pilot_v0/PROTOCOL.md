# WOBR-v0 — Word-Owned Bilinear Response pilot

Status: **frozen protocol**. This document, `configs/wobr.json`, and the
committed code are the experiment definition. One model, one seed, no search,
no controls, no WOBR-v1. The only role of this round is to implement and run
this protocol, verify engineering correctness, and report what happened.

## 1. Question and scope

Can a whole-molecule lipophilicity predictor be built from **word-owned
responses** rather than a pooled molecular state?

For every atom `v`:

1. a rooted radius-2 sentence produces a query `h_v`;
2. a small set of connected induced atom supports containing `v` are candidate
   words; each word has its own embedding `w_(v,S)`;
3. a root-wise sparsemax selects a convex mixture of those words,
   `z_v = sum_S alpha_(v,S) w_(v,S)`;
4. `z_v` enters a strictly linear self term and a bilinear endpoint interaction
   on every real physical bond.

`h_v` is only a selection query. It never enters the final prediction
directly. There is no pooled hidden state, no readout MLP, no extra branch.

This is a **pilot**, not a claim that words are causal or that the model has
solved structural sufficiency. The test split is a development test
(`DEVELOPMENT_TEST`) and is reported with that label.

## 2. Inputs and provenance

| object | identity |
|---|---|
| raw cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz` |
| raw cache SHA-256 | `fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c` |
| official split | `train_idx` 3360 / `valid_idx` 420 / `test_idx` 420 |
| scaffold map | `tracks/structural-sufficiency/acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json`, SHA-256 `60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e` |
| copied GWIC sources | byte pins in `code/wobr_common.py::GWIC_SOURCE_PINS` |
| seed | `0` |
| test policy | `DEVELOPMENT_TEST` (evaluable every epoch; never used for gradients, standardization, checkpoint selection, or early stopping) |

The cache is opened with `allow_pickle=False`. Structure arrays are separated
from labels and split ids. The target is the raw whole-molecule label.
Standardization uses **official Train labels only**, mean and population
standard deviation (`ddof=0`). All reported RMSE/MAE are in raw label units.

All 9 atom categorical fields and all 3 bond categorical fields are used. The
embedding tables have the full OGB legal vocabulary. Category support is
computed from official Train only; unsupported rows (including composite path
and neighbour codes) are zeroed after construction. Because training batches
contain only Train molecules, those rows receive no supervision.

Physical undirected bonds are stored once per cache (`bond_u < bond_v`).

## 3. Sentence and candidate words

### 3.1 Rooted sentence

For each atom `v`, the sentence is the **full radius-2 induced ego graph**
`R2(v)`: all atoms at graph distance 0, 1, or 2, and every real induced bond
among them. No truncation, sampling, or top-k. The sentence encoder is
non-iterative and LCRC-style:

- S1 content: atom embeddings, incident bond sums/means, neighbour codes;
- typed non-backtracking 2-paths `v -> w -> u` with `u != v`, including the
  middle atom type, both bond types, end atom type, closure flag, and end-atom
  content;
- complete R2 bond sums/means and distance buckets;
- distance-1 and distance-2 atom sums/means;
- local atom scalars and R2 graph scalars.

The result is `h_v` in `R^96`. It is used only as the candidate-selection
query.

### 3.2 Candidate words

For each root `v`, candidates are all **connected induced supports** `S` with

```
v in S,   1 <= |S| <= 4
```

Each support occurs exactly once per root. There is no hard top-k, no sampling,
and no truncation. The only structural change relative to GWIC is the addition
of the singleton `{v}`.

For a candidate word:

- **internal bonds** are all real bonds with both endpoints in `S`;
- for the singleton, the internal bond list is empty;
- **boundary bonds** are all real bonds with one endpoint in `S` and the other
  in `R2(v) \ S`;
- the 25-d structural scalar vector has the GWIC layout:
  - `[0]` size / 4
  - `[1]` internal bond count / 6
  - `[2]` cycle rank / 3
  - `[3]` root degree inside the word / 3 (0 for the singleton)
  - `[4:7]` induced internal pairwise-distance distribution for distances 1,2,3
  - `[7:10]` root-distance distribution for distances 1,2,3
  - `[10]` boundary count / sum of full-molecule degrees over `S`
  - `[11]` `log1p(boundary count)`
  - `[12:25]` boundary bond-type fractions over the 13 mixed-radix
    `(bond_type, stereo, conjugation)` categories

**Necessary implementation fix.** The parent GWIC helper used the root-distance
of the right-hand atom of each pair when filling `[4:7]`; that value depended on
the slot order of the support and broke atom-relabelling invariance. WOBR
computes true shortest-path distances inside the induced support for `[4:7]`.
For the singleton all pairwise entries are zero, so its scalars are unchanged.
This deviation is recorded in the report.

Word embedding `w_(v,S)` uses root atom content, support atom sums/means/stds,
internal bond sums/means/stds, joint endpoint-bond features, and the 25-d
scalars. Word encoder width is 160; output is 64-d, followed by LayerNorm
inside the word encoder only.

## 4. Selection

For each root independently, scores are

```
score_(v,S) = ScoreMLP(concat(w_(v,S), h_v)),
```

with score hidden width 96. Sparsemax at temperature 1 gives

```
alpha_(v,S) = max(score_(v,S) - tau_v, 0),   sum_S alpha_(v,S) = 1.
```

`tau_v` is the per-root sparsemax threshold. Implementation is vectorised on
padded `[n_roots, max_candidates]` tensors, with padding masked to a neutral
value during the sort; no per-candidate Python loop is used in selection.

Then

```
z_v = sum_S alpha_(v,S) w_(v,S)
```

is a **strictly linear** mixture. There is no LayerNorm, MLP, dropout, or other
post-mix transformation on `z_v`.

## 5. Prediction

With all quantities on the standardized label scale:

```
t_v       = U(z_v)                              U: 64 -> 16, no bias
lambda_e  = tanh(BondInteraction(b_e))          b_e: raw 24-d bond embedding
Q_e       = sum_r lambda_e[r] * t_u[r] * t_v[r] / sqrt(16)
yhat_std  = beta
            + c_n * log1p(n_atoms)
            + c_m * log1p(n_physical_bonds)
            + sum_v dot(a, z_v)
            + sum_e Q_e
```

`BondInteraction` is `24 -> 32 (SiLU) -> 16 (tanh)`, applied to the raw 24-d
bond embedding. `a` is a learnable 64-d vector. There is no other MLP, branch,
pooled state, R2 readout, or direct `h_v` term.

Dimensions are fixed: atom 48, bond 24, sentence 96, word 64, response 16,
path rank 24, path end 16, path close 8, neighbour 16, backbone hidden 192,
word hidden 160, score hidden 96, dropout 0.1.

## 6. Initialization

- Conventional layers and embeddings: PyTorch defaults.
- GWIC sentence path gate: ones.
- `a ~ Normal(0, 0.02)`.
- `U`: Xavier uniform.
- Last `BondInteraction` layer: weight `Normal(0, 0.02)`, bias 0.
- `beta = c_n = c_m = 0`.
- Seed 0; the full initial state fingerprint and per-module hashes are recorded
  in `run.json`.

The interaction is not initialized all-zero. Category-support zeroing happens
after this initialization and is recorded separately.

## 7. Training recipe

Single model, seed 0, no hyperparameter search.

| item | value |
|---|---|
| optimizer | AdamW |
| peak lr | 3e-4 |
| minimum lr | 1e-5 |
| weight decay | 1e-4 |
| batch | 64 molecules |
| warmup | 5 epochs |
| schedule | cosine |
| max epochs | 240 |
| min epochs | 80 |
| early-stop patience | 45 epochs without Valid improvement |
| gradient clip | 5.0 |
| loss | standardized Train-label MSE only |
| checkpoint | minimum official Valid RMSE |
| precision | float32, no AMP |
| auxiliary loss / distillation | none |
| per-epoch evaluation | full Train, Valid, Test RMSE and MAE |

Test values are logged every epoch as `DEVELOPMENT_TEST`; they are never used
for gradients, standardization, checkpoint selection, or early stopping.

## 8. Pre-training checks (must pass before training)

The following are implemented in `code/wobr_checks.py` and written to
`analysis/checks.json`:

1. candidates are real connected induced supports; every root has exactly one
   singleton; internal and boundary bonds are exact;
2. atom renumbering, edge reversal, bond ordering, and candidate ordering do
   not change the float64 prediction by more than `1e-8`;
3. sparsemax matches two independent NumPy references to `1e-8`, is
   nonnegative, normalized, can produce exact zeros, and padding is zero;
4. two synthetic graphs with the same atom/bond counts but different
   connections produce different bilinear bond terms;
5. the exact contribution identity holds in float64 on a small graph
   (`<=1e-10`) and in float32 on a real batch (`<=1e-4`), and per-bond
   word-pair expansion equals the direct `z_u,z_v` bilinear term;
6. batched and single-molecule inference agree (`<=1e-5`);
7. word encoder, score MLP, `a`, `U`, and bond interaction have finite nonzero
   gradients on a real batch;
8. seed 0 reproduces the same initial state and the frozen explicit
   initialization values;
9. data SHA, official split, Train-only scaler, raw-unit restoration, full OGB
   embedding widths, and unsupported-row zeroing are correct;
10. isolated atoms and disconnected components are preserved and never cross
    components;
11. complete candidate totals, per-root maximum, and real-batch peak resources
    are recorded.

The formal checks must use all 4200 rows. `--max-rows` is a local smoke aid
only and is recorded as such.

## 9. Compute budget and preflight

All remote execution goes through `rr`; no manual SSH or `sbatch`. Preferred
pool: `res-gpu0`, fallback `res-gpu1`.

Before the formal run:

1. run the full checks and a real 2-epoch preflight with full Train/Valid/Test
   evaluation, recording seconds per epoch and peak VRAM;
2. project 240-epoch GPU hours plus the actual preflight cost;
3. formal GPU budget: **2.0 GPU-hours**; total including preflight: **2.5
   GPU-hours**.

If the projection exceeds budget, the only permitted change is an equivalent
vectorisation improvement. No candidate cut, batch change, model change, or
hyperparameter change is allowed.

## 10. Post-training analysis

One forward pass of the best-Valid checkpoint over all 4200 rows. No retraining
and no checkpoint switching.

Recorded:

- nonzero candidates per root, top-1 weight, alpha mass by word size;
- `z_v` variance and effective dimension (participation ratio, entropy rank,
  top eigenvalue share, near-constant fraction);
- RMS of self and bilinear terms, cancellation ratios, nonzero fraction of bond
  terms;
- contribution validity by word size, split, and singleton vs multi-atom;
- exact rooted-word canonicalization: root fixed in slot 0, exhaustive
  permutation of at most three non-root atoms, full 9 atom + 3 bond fields,
  exact certificate bytes (no hash-collision IDs);
- word reuse: exact word classes with at least 20 Train molecules and at least
  3 Train scaffolds qualify; top 20 by Train mean absolute contribution.
  Valid/Test labels are never used to rank words.

Contribution decomposition, in standardized units:

```
c_(v,S) = alpha_(v,S) * ( dot(a, w_(v,S))
          + 0.5 * sum_{u adjacent v} Q_(v,u)(w_(v,S), z_u) )
```

Real physical bonds are split half per endpoint. The same `alpha`, `w`, `U`,
and `lambda` are used; nothing is detached. The implementation checks

```
sum_(v,S) c_(v,S) = yhat_std - beta - c_n*log1p(n_atoms)
                    - c_m*log1p(n_physical_bonds)
```

and checks that the per-bond word-pair expansion equals the direct
`z_u,z_v` bilinear term. Contributions are reported in both standardized and
raw units (`raw = standardized * Train label std`).

These contributions are **model decompositions**, not counterfactual effects,
causal attributions, or statements that a word causes the label. They describe
how this trained function allocates its prediction.

## 11. Verdicts

Predictive verdict from official Valid RMSE:

| Valid RMSE | verdict |
|---|---|
| `<= 0.65` | `PREDICTIVE_CANDIDATE` |
| `(0.65, 0.70]` | `WEAK_PREDICTIVE_SIGNAL` |
| `> 0.70` | `NO_COMPETITIVE_SIGNAL` |

Word-evidence labels:

- `PREDICTIVE_BUT_WORD_EVIDENCE_WEAK` — prediction is competitive but multi-atom
  words add little non-singleton mass, contribution, or cross-molecule reuse;
- `STOP_THIS_IMPLEMENTATION` — no competitive predictive signal;
- `CONTINUE_DISCUSSION` — competitive prediction and non-trivial multi-atom word
  evidence.

The word-evidence rule is frozen before the run: if Valid RMSE is `> 0.70`,
label `STOP_THIS_IMPLEMENTATION`; otherwise compute (a) Train multi-atom alpha
mass, `1 - mass(size 1)`, and (b) Train multi-atom absolute-contribution share,
`sum_{size 2..4} sum|c| / sum_all sum|c|`. If either is below `0.10`, label
`PREDICTIVE_BUT_WORD_EVIDENCE_WEAK`; otherwise label `CONTINUE_DISCUSSION`.
These are evidence gates, not causal claims.

The historical GINE reference range `0.61–0.64` is descriptive only. It comes
from a different recipe and is not a matched control.

## 12. Artifacts

```
PROTOCOL.md
configs/wobr.json
code/                       implementation and checks
tests/                      CPU unit tests
analysis/
  checks.json
  candidate_coverage.json
  preflight_gpu.json
  word_selection.json
  word_contributions.json
  word_reuse.json
  word_types.csv
  high_contribution_words.csv
  analysis_summary.json
  curves.json
results/seed0/
  metrics.csv
  curves.csv
  best.pt
  last.pt
  predictions_best.npz
  contributions_best.npz
  contributions_best.csv
  run.json
  train.log
REPORT.md
```

`git status` is checked first. Source, config, and protocol are committed before
the formal run. Results are committed and pushed after the run. No
`reset --hard`, `clean -fd`, or force push.

### Reproduction entry points

```bash
# local CPU unit tests
uv run pytest -q tracks/structural-sufficiency/word_owned_bilinear_response_pilot_v0/tests

# pre-training checks (formal: no --max-rows)
uv run python tracks/structural-sufficiency/word_owned_bilinear_response_pilot_v0/code/wobr_checks.py

# two-epoch preflight
uv run python tracks/structural-sufficiency/word_owned_bilinear_response_pilot_v0/code/wobr_train.py \
  --run-id preflight_seed0 --epochs 2

# formal run
uv run python tracks/structural-sufficiency/word_owned_bilinear_response_pilot_v0/code/wobr_train.py \
  --run-id seed0

# budget projection and post-training analysis
uv run python .../code/wobr_preflight.py
uv run python .../code/wobr_analyze.py
```

Remote execution uses `rr deploy` / `rr run` / `rr pull` on the committed
revision. The exact host, pool, commit, torch/CUDA versions, timing, and peak
VRAM are recorded in `run.json` and `analysis/preflight_gpu.json`.