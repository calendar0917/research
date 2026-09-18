# MolHIV B-Null frozen small-head sufficiency probe — pre-registration

Protocol: `molhiv_bnull_small_head_probe_v1`
Protocol id: `molhiv-cross-scaffold-interaction`
Branch: `exp/bnull-sab-path-ablation`

Committed **before** any representation extraction or head training.

## 0. Reference (read from the repo)

* Frozen model: **MolHIV B-Null seed0 Top-5 soup**
  (`results/molhiv_local_token_null/soup_null_seed0.pt`).
* Recorded official-valid soup ROC-AUC `0.840847`
  (`run_null_seed0.json`); raw `0.843168 @ep24`.
* Parameters 237,133 (exact zero 32-D local patch token, no local-token
  generator).

## 1. Question

On the frozen pre-head graph representation `R(G)`, does one fixed ~14k small
head recover essentially the same official-valid ROC-AUC as a fresh head with
the current architecture? This positions whether a future end-to-end compact
MolHIV model (graph-head compression) is worth pursuing at all.

## 2. Frozen representation

`R` is the exact input to the current graph head:
`[unary node moments ; distance-conditioned pair moments ; global encoder]`
= `2·96 + 1` (unary) `+ 6·(2·16 + 1)` (pair, 6 distance buckets) `+ 32`
(global) = **423**. Dimensions are confirmed from the real head
(`head[0].in_features == 423`), not assumed.

Extraction:
* official **train** (32,901) and official **valid** (4,113) only;
* backbone frozen (`model.eval()`, `torch.no_grad()`), no gradients;
* `R` captured by a forward pre-hook on the head and verified by re-applying the
  frozen head to `R` and comparing with the model logits;
* hashes recorded for `R`/targets and for the frozen soup checkpoint;
* **no test representation is extracted and the official test is never loaded**.

## 3. H_refit — current-head refit control

Fresh head with the **exact current MolHIV head architecture**
`Linear(423,192) → LayerNorm → ReLU → Dropout(0.05) → Linear(192,96) → ReLU →
Linear(96,1)` (100,417 params) trained only on frozen `R`.

Protocol: BCEWithLogitsLoss, Adam lr 1e-3, weight decay 1e-5, batch 128, max 240
epochs, patience 40, official-valid ROC-AUC selection, fixed equal-weight Top-5
soup, seed0.

Gate: if `H_refit_soup_AUC < original_BNull_soup_AUC − 0.010` ⇒
`HEAD_PROBE_INVALID`; stop small-head architecture conclusions and analyse why.

## 4. H_small32 — the single small head

Exactly one pre-registered head, no width sweep:

`Linear(423,32) → LayerNorm(32) → ReLU → Dropout(0.05) → Linear(32,16) → ReLU →
Linear(16,1)` (14,177 params).

Same frozen `R`, same loss/optimizer/schedule/selection/soup/seed.

## 5. Gate

`ΔAUC = H_refit_soup_AUC − H_small32_soup_AUC`:

* `Δ ≤ 0.005` ⇒ `STRONG_SMALL_HEAD_SUFFICIENCY` (worth a future end-to-end
  compact MolHIV);
* `0.005 < Δ ≤ 0.015` ⇒ `MILD_SMALL_HEAD_SIGNAL`;
* `0.015 < Δ ≤ 0.020` ⇒ `BORDERLINE_SMALL_HEAD_SIGNAL`;
* `Δ > 0.020` ⇒ `SMALL_HEAD_NO_GO`.

`H_refit` is the matched frozen-representation reference, never the joint
training AUC.

## 6. Non-goals

No backbone training, no end-to-end compact MolHIV, no head width/dropout/lr
sweep, no second small head, no official test, no seed sweep beyond seed0.
