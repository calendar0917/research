# TU graph-classification channel necessity (tu_channel_necessity_v1)

Result note for the pre-registration
[`tu_channel_necessity_preregistration.md`](tu_channel_necessity_preregistration.md)
and protocol [`protocols/tu-graph-classification.yaml`](../protocols/tu-graph-classification.yaml).
Written **after** the 3-fold screen. Frozen revision
`15849603dd627bb60899bdc7e6a85cd94eef7ba0`; remote A100-SXM4-40GB, GPU 0
(shared with an unrelated 35.8 GB co-tenant, never touched); `official_test_loaded: false`
everywhere (TU has no official split; the held-out fold is the terminal read).

## 1. What was asked

1. Feasibility: does the compact recurrent pair–centre backbone port to TU
   graph classification and train to non-trivial accuracy?
2. Necessity: is the graph-dependent local patch-token channel (the exact
   rooted-patch certificate lookup) necessary, or does the fixed downstream
   backbone carry almost all predictive power?

This is a new benchmark track, not a molecular experiment: new data pipeline,
new strict split generator, new classification head and accuracy metric. The
scientific question is carried over from the ZINC/MolHIV rounds.

## 2. Setup

Self-contained port (`experiments/luyin16/tu_patch_path_pooling.py`): rooted
radius-2 patch descriptor, switchable local token of width 16, radius-1 parent
token, all patch-centre pairs with a distance/overlap/adjacency relation, `T=2`
weight-tied recurrent pair→centre refresh, patch/pair moment readout + global
context, MLP classification head. Widths `patch_hidden=64`, `pair_hidden=16`,
`center_hidden=60`, `dropout=0.05`; Adam lr 1e-3, wd 1e-5, batch 64,
cross-entropy, grad clip 5, 150 epochs, patience 30, fixed equal-weight Top-5
soup + best-valid raw checkpoint.

Strict split protocol: generated once per dataset and frozen before training.
Outer `StratifiedKFold(k=3, shuffle=True, random_state=20260918)`; the test set
is the held-out fold; the remaining folds are split once into train 80 % /
valid 20 %. All three conditions share the identical splits; vocabularies and
standardizers are fit on the train fold only; unseen certificate → learned OOV
row. Invariants verified and recorded: pairwise disjoint within a fold, every
graph tested exactly once.

| dataset | graphs | classes | types (node/edge) | split fingerprint (first 16) | majority baseline |
|---|---|---|---|---|---|
| MUTAG | 188 | 2 | 7 / 4 | `eada5c9b3eb70644` | 0.665 |
| PROTEINS | 1113 | 2 | 3 / 1 | `d991fb7c4013d659` | 0.596 |
| IMDB-BINARY | 1000 | 2 | 16 (degree) / 1 | `f1f7a5cde2cd0848` | 0.499 |

## 3. Results (3-fold, seed 0)

`D = mean_fold(typed_soup) − mean_fold(null_soup)`, positive = the typed
channel helps. Pre-registered STRONG band `D <= 0.02`.

| dataset | condition | params | valid soup | test raw | test soup | D_soup | D_raw |
|---|---|---|---|---|---|---|---|
| MUTAG | typed_lookup | 82,902 | 0.9472 | 0.8565 ± 0.0309 | 0.8777 ± 0.0178 | **+0.0054** | −0.0265 |
| MUTAG | null | 79,558 | 0.9338 | 0.8830 ± 0.0241 | 0.8723 ± 0.0012 | | |
| MUTAG | constant | 79,574 | 0.9338 | 0.8776 ± 0.0245 | 0.8670 ± 0.0086 | | |
| PROTEINS | typed_lookup | 147,966 | 0.7897 | 0.7430 ± 0.0127 | 0.7475 ± 0.0102 | **−0.0135** | −0.0099 |
| PROTEINS | null | 80,142 | 0.8009 | 0.7529 ± 0.0136 | 0.7610 ± 0.0243 | | |
| PROTEINS | constant | 80,158 | 0.8009 | 0.7493 ± 0.0093 | 0.7502 ± 0.0189 | | |
| IMDB-BINARY | typed_lookup | 124,190 | 0.7662 | 0.7090 ± 0.0177 | 0.7040 ± 0.0191 | **−0.0100** | −0.0010 |
| IMDB-BINARY | null | 91,022 | 0.7662 | 0.7100 ± 0.0110 | 0.7140 ± 0.0020 | | |
| IMDB-BINARY | constant | 91,038 | 0.7637 | 0.7230 ± 0.0076 | 0.7170 ± 0.0101 | | |

Per-fold detail and the parameter/vocabulary accounting are in
`results/tu_channel_necessity/runs/*.json` and `report_k3.json`.

### Parameter accounting of the ablated channel

| dataset | typed total | null total | channel params | share | typed vocab (incl. OOV) |
|---|---|---|---|---|---|
| MUTAG | 82,902 | 79,558 | 3,344 | 4.0 % | 209 |
| PROTEINS | 147,966 | 80,142 | 67,824 | 45.8 % | 4,239 |
| IMDB-BINARY | 124,190 | 91,022 | 33,168 | 26.7 % | 2,073 |

## 4. Verdict

**STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED on all three datasets.**

On every TU dataset `D_soup` lies inside ±0.014, far inside the pre-registered
STRONG band (`D <= 0.02`). The typed certificate lookup is entirely redundant:

* PROTEINS — deleting 67,824 params (45.8 % of the model) *improves* soup
  accuracy by 0.0135 and raw by 0.0099.
* IMDB-BINARY — deleting 33,168 params (26.7 %) improves soup by 0.0100.
* MUTAG — deleting 3,344 params (4.0 %) costs +0.0054 soup but *gains* 0.0265
  raw in the other direction; the sign flips between raw and soup, i.e. inside
  split noise (test folds are 63 graphs, one graph = 0.016).

This is the third independent replication of the ZINC/MolHIV pattern
(`decision-local-token-null-20260919`,
`decision-molhiv-local-token-null-20260918`), now on non-molecular discrete
graph classification.

**Pattern 2 (constant recovers the channel) does not replicate here either.**
Constant is at or below null on every dataset (MUTAG 0.8670 vs 0.8723,
PROTEINS 0.7502 vs 0.7610, IMDB-BINARY 0.7170 vs 0.7140), consistent with the
valid-vs-test retraction in `decision-local-token-null-test-read-20260918`.
Only "the channel is redundant" survives.

### Feasibility / competitiveness (secondary)

The backbone is competitive with commonly cited GIN-class numbers at 3 folds:
MUTAG soup 0.878 (typed/null), PROTEINS 0.747–0.761, IMDB-BINARY 0.704–0.717,
all well above the majority baselines (0.665 / 0.596 / 0.499). Runs are cheap
(MUTAG ~35–44 epochs, PROTEINS ~34–64, IMDB-BINARY ~41–82; whole 27-run screen
well under an hour on one shared GPU).

### Known degeneracy

IMDB-BINARY has no node/edge features, so its certificate is built from
clipped degree only; as pre-registered, the typed channel is correspondingly
small (26.7 %) and its removal is neutral-to-positive. This is expected, not a
bug.

## 5. Scope and limits

* Single model seed, 3 folds, test folds of 62–371 graphs (MUTAG 63/63/62,
  PROTEINS 371×3, IMDB-BINARY 334/333/333). This is a pre-registered
  **screen**, not a statistical claim.
* The typed channel is small on MUTAG (4.0 %), so MUTAG is weak evidence about
  channel necessity; PROTEINS and IMDB-BINARY carry the informative parameter
  drops and they are both non-positive.
* No GIN baseline was run at 3 folds (deferred by the pre-registration to the
  10-fold stage); the majority baseline plus literature numbers are context
  only.

## 6. What was NOT done

No official/extra test split, no multi-seed, no parameter re-investment, no
backbone widening, no new structural features, no hyperparameter sweep, no
per-dataset tuning. The 10-fold stage and the GIN baseline are not authorised
by this round.

## 7. Next action (not auto-authorised)

If the TU track is continued, the single pre-registered next step is Stage 2:
the same fixed model and split generator at `k=10`, plus a GIN baseline, so the
backbone-competitiveness question (the actually interesting one on TU) is
answered with mean ± std instead of a 3-fold screen. Channel necessity is
already answered and needs no further TU compute.
