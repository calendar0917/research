# Pre-registration: TU graph-classification channel necessity (tu_channel_necessity_v1)

Written **before** any TU candidate training. Frozen revision: the commit that
adds `tracks/ksvd/experiments/luyin16/tu_patch_path_pooling.py`,
`tu_channel_necessity.py`, `protocols/tu-graph-classification.yaml` and this
file. Protocol: `tu-graph-classification`.

## Why a new benchmark track

The ZINC and MolHIV rounds answered the same question twice on molecular
graphs: the molecule-dependent local patch-token channel (the exact
rooted-patch certificate lookup) is **redundant** — a null token loses nothing
and a single trainable constant recovers whatever shared offset mattered
(`decision-local-token-null-20260919`,
`decision-molhiv-local-token-null-20260918`). TU is the natural third setting:
non-molecular, discrete node/edge labels (or none at all), binary
classification, 10x–100x fewer graphs.

This is a **new benchmark track, not a drop-in**. TU has no official split, no
molecular chemistry, and a different task; the port keeps the *scientific
question* (is the local token channel necessary, and is the compact pair–centre
backbone competitive at all?) while re-implementing the data pipeline,
splitting and metric.

## Question

1. **Feasibility / competitiveness.** Does the compact recurrent pair–centre
   backbone train and reach non-trivial accuracy on TU graph classification?
2. **Channel necessity.** On that backbone, is the graph-dependent local
   patch-token channel necessary, or does the fixed downstream backbone carry
   almost all predictive power?

(2) is a channel-necessity / backbone ablation. The parameter drop is the
experimental variable; no parameter re-investment, no architecture search.

## Model (self-contained port)

Faithful to the ZINC/MolHIV compact pair–centre family, with TU-sized widths:

* per-node rooted radius-2 patch descriptor (distance-binned node-type
  histogram, edge-type histogram, degree stats, patch-size stats),
  standardized;
* local token channel, width 16 — the ablated slot;
* radius-1 parent certificate token, width 8;
* all unordered patch-centre pairs with a relation vector
  (distance one-hot + log-distance, node-overlap, boundary-overlap, adjacency
  edge type);
* `T = 2` weight-tied recurrent pair→centre refresh
  (`patch += center_update([patch, centre_context])`);
* readout: patch moments (total, squared, log-count, per distance bucket),
  pair moments per distance bucket, global context (n, m, density, degree
  mean/max/std, type widths);
* MLP head to `n_classes`.

Widths: `patch_hidden=64`, `pair_hidden=16`, `token_width=16`,
`parent_width=8`, `center_hidden=60`, `dropout=0.05`. Training: Adam lr 1e-3,
wd 1e-5, batch 64, cross-entropy, grad clip 5, 150 epochs, patience 30 on valid
accuracy, Top-5 equal-weight soup, best-valid raw checkpoint.

**Known degeneracy (reported, not hidden).** IMDB-BINARY has no node or edge
features; there the certificate is built from clipped degree only, so the
typed channel carries far less molecule-independent information than on
MUTAG/PROTEINS. That is an expected outcome of the port, not a bug.

## Conditions (only the local token changes)

| condition | local token | generator params |
|---|---|---|
| `typed_lookup` | learned row per exact rooted-patch certificate | `V_token x 16` |
| `null` | exact zero 16-D per patch, no generator | 0 |
| `constant` | one trainable graph-wide 16-vector | 16 |

Everything else is identical; the patch-encoder input width is unchanged.

## Strict split protocol

TU ships no canonical split, so the split is generated here and frozen before
any training (`protocols/tu-graph-classification.yaml`):

* outer `StratifiedKFold(k, shuffle=True, random_state=20260918)`;
* the test set is the held-out fold; the remaining `k-1` folds are split once
  into train (80 %) / valid (20 %), `random_state = 20260918 + 1000*(fold+1)`;
* invariants checked and recorded: pairwise disjoint within a fold, every graph
  tested exactly once, identical splits for every condition;
* the frozen index lists and a SHA-256 fingerprint are written to
  `splits_<dataset>_k<k>.json` **before** training; every run record repeats the
  fingerprint;
* vocabularies and standardizers are fit on the **train fold only**; unseen
  certificate → learned OOV row.

## Test discipline

The test fold is read **once** per run, after the best-valid checkpoint and the
fixed Top-5 soup are frozen. It never feeds training, early stopping, soup
selection or any hyperparameter choice. Both `raw` and `soup` are reported
without cherry-picking. `official_test_loaded: false` is written because no
official split is used; the held-out fold is the terminal read.

## Plan and pre-registered gates

**Stage 1 — 3-fold screen (this pre-registration).** Datasets MUTAG, PROTEINS,
IMDB-BINARY; conditions `typed_lookup`, `null`, `constant`; model seed 0; `k=3`.
A majority-class baseline (train-fold majority applied to the test fold) is
recorded for context.

For each dataset let `D = mean_fold(typed_soup_test) - mean_fold(null_soup_test)`
(positive = the typed channel helps).

* `D <= 0.02` → **STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED**.
* `0.02 < D <= 0.05` → **MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT**.
* `0.05 < D <= 0.10` → **SUBSTANTIAL_LOCAL_TOKEN_CHANNEL_CONTRIBUTES**.
* `D > 0.10` → **STOP** for that dataset; report and analyse.

Bands are coarse on purpose: with test folds of 62–371 graphs a single fold's
accuracy moves by 0.002–0.016 per graph, so differences below ~0.02 are inside
split noise. This is a **feasibility screen**, not a headline benchmark result.

**Stage 2 — 10-fold, only if Stage 1 is judged worthwhile.** Same fixed model
and splits generator with `k=10`; a GIN baseline and a report of mean ± std over
folds. Stage 2 is not authorised by this document; it is authorised only after
Stage 1 is recorded in `STATE.yaml`.

## Forbidden in this round

Official/extra test splits (none exist, none are invented); multi-seed upfront;
parameter re-investment; backbone widening; new structural features
(cycles, RRWP, pair networks, attention); hyperparameter/optimizer sweeps;
per-dataset tuning; ad-hoc rescue; editing remote tracked files; treating a
frozen knockout as retrained causality. 10-fold before Stage 1 is recorded.
