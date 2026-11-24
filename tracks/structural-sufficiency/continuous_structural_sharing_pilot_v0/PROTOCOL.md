# CSS v0 — Continuous Structural Sharing (pilot protocol)

**Protocol id:** `css-v0-continuous-structural-sharing-pilot`
**Dataset:** `ogbg-mollipo`, official scaffold split 3360 / 420 / 420
**Seed:** 0
**Primary metric:** whole-graph RMSE in original label units
**Test role:** `DEVELOPMENT_TEST` (never selects a checkpoint, never changes a configuration)

Frozen before any formal training. Everything below is fixed; no mid-round
change is permitted after timing or after seeing validation performance.

## 1. Question and scope

Does continuous sharing over **real reference structures** improve whole-graph
generalisation and sample efficiency for a non-message-passing graph property
predictor?

Four training units only:

| unit | Fit | branch input |
|---|---|---|
| `DIRECT_100` | all 3360 official Train | `z_v` |
| `SHARE_100` | all 3360 official Train | `q_v` |
| `DIRECT_50` | 1680 hash-selected Train rows | `z_v` |
| `SHARE_50` | same 1680 rows | `q_v` |

The 50% subset is the 1680 official Train rows with the smallest
`sha256("css-v0-50pct/<row_id>")`; the rule reads row ids only and is
label-free. Both sizes use exactly the same official Valid (420) and Test (420).

Out of scope: structure-replacement response, contribution decomposition, state
algebra, discrete structure words, sparse evidence selection, message passing,
transformers, teachers, distillation, pretraining, extra chemical labels, extra
seeds, extra K / tau / SVD dimensions, other data fractions or datasets.

## 2. Reused frozen artefacts

* raw graph cache `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`
  (sha256 pinned; contains all 4200 official labels);
* official split adapter `structural_evidence_selection_pilot_v0/code/ses_data.py`
  (byte-pinned): 3360/420/420, label-vault cross-check;
* R2 predictor `graph_word_incidence_composition_pilot_v0/code/gwic_model.py`
  (byte-pinned): `GwicModel(use_words=False)`, 416,177 parameters;
* complete-molecule struct + collation `ses_structure.py` / `ses_tensor.py`
  (byte-pinned).

No frozen file is modified.

## 3. Fixed structure descriptor

For every real root `v`, the complete radius-2 induced ego graph `R2(v)` gives
five sparse blocks:

1. `ROOT` — one-hot of the root's 9 official OGB categorical fields;
2. `S1` — `(atomic_number, bond_type)` joint counts over real neighbours;
3. `S2` — `(middle_atomic_number, first_bond_type, second_bond_type,
   end_atomic_number, closure)` joint counts over real non-backtracking
   two-hop paths `v-w-u`, `u != v`;
4. `EGO_ATOM` — per root distance 0/1/2 category counts of each of the 9 atom
   fields;
5. `EGO_BOND` — per unordered endpoint-distance pair category counts of each of
   the 3 bond fields over the real induced `R2(v)` bonds.

Real multiplicities are kept. Each block is `sqrt(count)` then L2-normalised;
empty blocks stay zero; the five blocks are concatenated and divided by
`sqrt(5)`. Original atom ids / molecule ids / labels / task responses are never
features, no edge or path is invented, and no extra connected subgraph of size
3–6 is enumerated.

Per training size (Fit only): frozen feature vocabulary, then
`TruncatedSVD(n_components=64, n_iter=5, random_state=0)`, then L2 normalisation
of every projected root descriptor → `z_v ∈ R^64` (zero-safe). Valid/Test use
only the frozen vocabulary and SVD; unseen keys are ignored and their fraction
recorded. Fewer than 64 effective dimensions would be zero-padded and reported.
The projection dimension is never chosen by performance.

## 4. Real references and continuous weights

Per training size, `K = 64` references are selected from **Fit roots only** by
fixed seed-0 k-means++ D² sampling: real existing roots, no Lloyd update, no
synthetic centroids, no duplicate projected vector. `P = [p_1 … p_64]`.

```
d_j² = ||z_v - p_j||²
tau  = max(1e-6, median_Fit(d_8th²))          # 8th nearest squared distance
q_v  = softmax(-d² / tau)
```

`tau` is Fit-only. Nothing about the references, temperature or metric is
learned. `z_v` and `q_v` are precomputed; no top-k, no hard assignment, no
sparsity loss.

## 5. Two strictly matched predictors

The frozen R2/SENTENCE_BASE trunk is unchanged except for a 32-dimensional
branch at the backbone-MLP input:

```
r_v = Linear(64, 32)(normalize(t_v))
h_v = BackboneNorm(BackboneMLP(concat(x_v, r_v)))
normalize(t) = (t - mean_Fit(t)) / max(global_RMS_Fit, 1e-6)
t_v = z_v      (DIRECT)
t_v = q_v      (SHARE)
```

DIRECT and SHARE have identical parameter count (424,401), bitwise identical
shared initialisation, the same training molecule order / batch plan, the same
Fit-only label scaler, R2 normalisers and category support masks, and the same
shared graph readout. The only difference is `z_v` versus `q_v`.

## 6. Training recipe

AdamW, peak LR 1e-3, min LR 1e-5, weight decay 1e-3, batch 64 molecules,
5 warmup epochs then per-step cosine, dropout 0.1, gradient clip 5,
standardised-label MSE. No early stopping. Full Train/Valid/Test evaluation
every epoch in original units; the 50% arms additionally split the 1680 Fit
molecules from the 1680 held-out Train molecules. Best checkpoint by official
Valid RMSE (ties keep the earlier epoch). All parameters train from scratch; no
historical checkpoint is loaded.

## 7. Budget and horizon (decided from throughput only)

Two waves of two parallel A100 runs (`res`, one GPU per run): wave 1 = 100%
DIRECT/SHARE, wave 2 = 50% DIRECT/SHARE. Target ≤ 10 min per run, hard ≤ 15 min,
total ≤ 0.5 GPU·h, CPU preparation ≤ 3 min.

Before formal training, measure throughput with 3-epoch timing runs of
`DIRECT_100` and `DIRECT_50` (full three-split evaluation each epoch). Use the
mean per-epoch time of epochs 2–3, including evaluation, plus the measured
non-training overhead. Choose the largest common horizon `H ∈ {120, 90, 60}`
such that, with a 25% timing margin,

```
1.25 × H × t_100 ≤ 600 s           (per-run target)
1.25 × total_projected_GPU_seconds ≤ 1800 s   (0.5 GPU·h)
```

and the same per-run bound for the 50% runs. If 60 epochs is not feasible,
report a budget block and stop; never change the model or reduce the evaluation
scope. The horizon is never changed after seeing validation performance.

## 8. Pre-training checks (all must pass)

1. official row ids/splits; 50% Fit is a true 1680-row subset of 100% Fit;
2. every reference and every fitted statistic is reproduced from the
   corresponding Fit alone;
3. atom renumbering and bond-order changes leave `z_v`, `q_v` and the final
   prediction unchanged;
4. `q_v` finite, non-negative, rows sum to one;
5. identical structure descriptors give identical reference weights;
6. DIRECT/SHARE parameter counts, initial state and batch plans identical;
7. the new branch has a finite non-zero gradient and the original trunk updates.

Reference-weight degeneration is audited (mean/max entropy, effective number,
mean max weight, mean reference distribution, nearest-reference distance, spread
over Fit roots). No temperature search: if the frozen design itself were
degenerate, report and stop.

## 9. Evaluation and decision rule

Primary comparison `gain = RMSE(DIRECT) − RMSE(SHARE)` (positive = SHARE
better) on Valid and on Test, with 1000 molecule-level paired bootstrap
replicates (seed 0).

* A `PROMISING_FULL_DATA_GAIN`: SHARE_100 Valid gain ≥ 0.02, Test same
  direction, paired interval/training curves supportive.
* B `PROMISING_LOW_DATA_GAIN`: SHARE_50 Valid gain ≥ 0.02, Test same direction,
  no clear 100% regression.
* C `NO_USEFUL_SHARING_GAIN`: no useful gain at either size.
* D `SPLIT_CONFLICT_OR_BUDGET_LIMITED`: Valid/Test directions conflict, or
  training is clearly unconverged / budget-limited.

The 0.02 threshold is a planning reference, not a statistical law; one seed and
a molecule-level bootstrap do not represent training randomness. A positive
result here does not yet separate real references from an arbitrary fixed
nonlinear feature map; that control is future work.

## 10. Outputs

`configs/css.json`, `PROTOCOL.md`, `code/`, `tests/`, `analysis/checks.json`,
four-arm `curves_all.csv`, `main_table.csv`, best checkpoints and per-molecule
predictions, `reference_analysis.json`, `REPORT.md`.

Mechanism analysis runs on the best-Valid SHARE checkpoints only: reference
usage, eight fixed-hash Valid local structures with their top-3 references and
real atom/bond supports, and near/mid/far molecule groups by Valid
mean nearest-reference distance (groups defined without labels or errors).
Reference weights are never interpreted as atom contributions or chemical
causal importance.
