# Pre-registration: MolHIV local-token channel necessity (molhiv_local_token_null_v1)

Written **before** any MolHIV candidate training. Frozen revision: the commit
that adds `tracks/ksvd/experiments/luyin16/molhiv_local_token_null.py`.

## Motivation (transfer of the ZINC result)

On the ZINC strong mixed backbone the whole molecule-dependent local 16-D
patch-token family (35-36k params, 41.6-42.5 % of the model) turned out to be
redundant: B-Null (`e_patch = 0`, no generator, 49,343 params) reached Top-5
soup valid MAE 0.123028 and Constant-16 (one trainable graph-wide 16-vector,
49,359 params) reached 0.120515, against the matched seed0 references
B-Full 0.119818 / A0 0.124704 / B-Bag 0.127382
(`decision-local-token-null-20260919`). The slot's residual value was a shared
trainable offset, not molecule-dependent information.

MolHIV is the natural transfer target because the same channel dominates the
parameter budget there: the frozen `molhiv_recurrent_pair_centre` model has
1,076,589 params, of which the dense typed-certificate lookup
`typed_embedding [26233, 32]` alone is 839,456 (78.0 %), plus the radius-1
`parent_embedding [103, 16]` 1,648; the fixed backbone is 235,485 (21.9 %).
See `decision-zinc-cell-a-test-closure-and-molhiv-parameter-attribution-20260914`.

## Question

Is the molecule-dependent local 32-D patch token (the dense exact-certificate
lookup) actually necessary for the MolHIV recurrent pair--centre backbone, or
does the fixed downstream backbone carry almost all predictive power?

This is a **channel-necessity / backbone ablation**, not an architecture search
and not a parameter re-investment. The parameter drop is the experimental
variable.

## Conditions

| condition | local token | generator params | total params |
|---|---|---|---|
| typed (reference) | `typed_embedding(data.typed_token)`, width 32 | 839,456 | 1,076,589 |
| B-Null | exact zero 32-D per patch, no generator | 0 | **237,133** |
| Constant-32 | one trainable graph-wide 32-vector | 32 | **237,165** |

Everything else (patch encoder, parent embedding, pair projection, relation
encoder, distance gate, pair encoder, centre update, global encoder, head,
T=2 weight tying, loss, optimizer, batch size, epochs, patience, soup rule) is
identical to the frozen MolHIV reference model. The patch-encoder input width
is unchanged in all three conditions.

## Protocol

* `protocol_id: molhiv-cross-scaffold-interaction`, `test_policy: terminal`.
* **The official MolHIV test split is never loaded in this round.** The
  candidate runner has no `freeze`/`test` stage and writes
  `official_test_loaded: false` everywhere.
* Official train / official valid only; vocabulary, standardizers and the
  train-side mean token are fit on official train only.
* Frozen training regime (identical to the reference model):
  Adam lr 1e-3, wd 1e-5, batch 128, `BCEWithLogitsLoss` unweighted, grad clip
  5.0, 240 epochs, patience 40 on official valid ROC-AUC, no scheduler.
* Selection: `raw` = single best-valid-AUC checkpoint; `soup` = fixed
  equal-weight parameter average of the 5 highest-valid-AUC checkpoints
  (ties -> earliest epoch). No k/weight search.
* Seed 0 only, exactly one architecture at a time, as in the ZINC round.

## Pre-registered gates (seed 0, official valid ROC-AUC soup)

Matched seed0 typed reference: **raw 0.814239, soup 0.809505**.
(Seed1 typed: raw 0.849865, soup 0.851092.)

Let `D = typed_soup_seed0 - null_soup_seed0` (positive = null worse).

* `D <= 0.005` -> **STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED**; authorise the
  Constant-32 stage.
* `0.005 < D <= 0.020` -> **MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT**;
  authorise the Constant-32 stage.
* `0.020 < D <= 0.050` -> **SUBSTANTIAL_LOCAL_TOKEN_CHANNEL_CONTRIBUTES**;
  do not auto-authorise Constant-32; analyse and report.
* `D > 0.050` -> **STOP**; the local-token channel is required at this budget.

The bands are deliberately coarse because the two existing typed seeds differ
by 0.0416 AUC soup (0.809505 vs 0.851092): a single-seed delta below ~0.02
cannot be resolved against that spread, and this round is explicitly a
**screening** result, never a causal claim.

## Interpretation boundary (declared now)

* A STRONG/MILD result means "the dense certificate lookup is not necessary to
  preserve this backbone's accuracy at this budget" — a parameter-efficiency
  and redundancy statement. It is **not** a claim that local structural
  identity is uninformative for MolHIV, nor a SOTA claim.
* A SUBSTANTIAL/STOP result means the lookup carries accuracy this backbone
  cannot recover from its other channels at this budget.
* Frozen knockout on MolHIV is **not** run in this round (unlike ZINC Stage A)
  to save budget; the axis is retrained necessity.
* MolHIV valid is noisy at this scale; the round reports point estimates with
  the seed-spread context and buys no extra seeds.

## Forbidden in this round

Official test; multi-seed upfront; re-investing the released 787k/839k params;
widening the backbone; new structural features/radius/RRWP/cycles/pair network/
attention; hyperparameter or optimizer sweep; ad-hoc rescue after a bad
B-Null; remote tracked-code editing; treating a single-seed delta as a
statistical claim; changing the frozen reference model's code path.
