# PROTOCOL — zinc structural-token attention v1

| field | value |
|---|---|
| protocol id | `zinc-structural-token-attention-v1` |
| round directory | `tracks/structural-sufficiency/results/zinc_structural_token_attention_v1/` |
| code | `tracks/structural-sufficiency/code/zinc_sta_v1/` |
| config | `tracks/structural-sufficiency/configs/zinc_structural_token_attention_v1.yaml` |
| status at freeze | `FROZEN_PRE_GPU` |
| kind | exploratory object-token attention feasibility screen |
| relation to Phase 2b / C-G | independent side round; no Phase 2b or C/G file is modified; not Phase 2c |
| official test | never loaded, in any stage, by any code path in this round |
| valid labels | single read at the `score` stage only, after all six predictions and the mechanism record are frozen |
| main reference | frozen historical P3 (`BJOC-Null`, R3, full-10k masked SSL) |

This document is frozen **before** the first GPU second of this round and **before** any new
official-valid score. It is an exploratory screen on an already-exposed development split. It is
not a blind test, not a novelty claim, and not a GRIT/ESA+ reproduction.

---

## 1. Question

On top of the frozen P3 explicit object representation (atoms, physical bonds, junctions), does a
**complete cross-object representation interaction followed by aggregation** beat the current P3
readout? Is the **topological relation between objects** more useful than content similarity alone?

Three arms separate the confounds:

* **S** — same token blocks, but `MIX(x) = gamma * FFN_extra(x)`. No new cross-token interaction.
  This is the capacity / readout control. It still retains P3's own Null context path, so it must
  not be described as "completely without global information".
* **T** — ordinary 4-head object self-attention. All valid token pairs in the same molecule are
  visible; no positional encoding, no atom numbering.
* **R** — T plus an additive structural bias per layer and head:
  `Btype[type_i,type_j,h] + Bdist[dist_bin(i,j),h] + Boverlap[overlap(i,j),h]`.
  R is **not** a sparse neighbour mask; it still performs full within-molecule object attention.
  No GCN/GINE/local graph convolution is added.

The comparison is always against the frozen P3 reference, and additionally T vs S, R vs S and R vs T.

---

## 2. Data and common starting point

* Only supervised data: official PyG ZINC `subset=True` train, rows 0..9999, raw-y L1.
* Only SSL structures: the same official train 10,000. No new unlabelled molecule, external
  structure, pretrained model, target component, teacher or ensemble is introduced.
* For each seed 0/1, every arm starts from the **same P3 R3 supervised start**:
  1. `bp.build_bjoc_null(seed)` with the frozen train-10k standardiser;
  2. load the seed-matched frozen full-10k masked-SSL encoder (`pretrain_encoder_s{seed}.pt`);
  3. copy `atom_encoder`, `bond_encoder`, `junction_encoder`, `group_composer` and buffers
     tensor-by-tensor into the new model;
  4. initialise the new modules from dedicated per-module RNGs; the global RNG is restored.
* The completed P3 soup is **never** used to initialise a new arm.
* Parameter *provenance* is fixed, but every copied module **participates in optimisation**.
  Nothing is a frozen training weight.

---

## 3. Representation and reader

### 3.1 P3 bypass

Keep the original P3 Null grouping and composer code path. For each molecule:

```
z_P3 = [ sum_e c_e (96), sum_i a_i (96), sum_i a_i^2 (96), g_33 (33) ]   -> 321
```

This is the original P3 reader input, not a frozen P3 prediction. It is kept as a bypass; no
performance preservation is assumed.

### 3.2 Object tokens

```
H0 = concat_rows(a, b, w) + learned_type_embedding[A/B/J]        (96)
```

Token supports are `A={i}`, `B={u,v}`, `J={i,j,k}` from the original legal object indices.
There is no atom index, no sequence/position encoding, and no truncation. Attention is strictly
within one molecule.

### 3.3 Two independent blocks

Two distinct module instances, each with its own parameters:

```
U      = H + MIX(LN1(H))
Hnext  = U + FFN(LN2(U))
```

* `LN` affine=True, eps=1e-5.
* `FFN`: 96 -> 192 -> 96, SiLU, both layers biased, dropout 0.
* Attention dropout 0; no final LayerNorm; no residual scaling.
* Padding exists only for batch storage. It is never an attention key and never enters a
  statistic; padding rows are zeroed after each block.

`MIX`:

* **S**: `gamma ⊙ FFN_extra(x)`, `FFN_extra` 96->192->96 + SiLU, `gamma` 96-d initialised to 1.
  Each MIX has 37,248 parameters, exactly the same count as the 4-head MHA.
* **T**: standard 4-head Q/K/V/output projection MHA, head dim 24, all projections biased.
  `logits = q_i · k_j / sqrt(24)`, all valid pairs in the molecule visible.
* **R**: T plus the zero-initialised structural bias above. Distance and overlap are deterministic
  functions of the existing undirected atom graph; no new data is introduced.

### 3.4 Final aggregation

```
Z = concat( z_P3(321),
            sum_A H2(96), sum_B H2(96), sum_J H2(96),
            log1p(N_A), log1p(N_B), log1p(N_J) )              -> 612
y_hat = reader(Z),  reader = 612 -> 96 -> 96 -> 1, SiLU, no dropout/BN
```

Type-wise sums only; no softmax/mean pooling replacement. An absent type contributes an exact
zero vector. The bypass and type-wise sums retain composition/count channels; this does not make
them optimal or causally interpretable.

---

## 4. Parameters

| item | count |
|---|---:|
| P3 non-reader modules (atom/bond/junction/composer) | 265,024 |
| P3 old reader (removed, not retained) | 148,481 |
| each new block | 74,784 |
| type embedding | 288 |
| new reader | 68,257 |
| **S** | **483,137** |
| **T** | **483,137** |
| **R** | **483,289** (152 relation-bias parameters) |

S/T are nominally parameter-matched. R/T differ by exactly 152. Equal parameter counts do not
guarantee equal effective capacity or optimisation; this boundary is carried into the report.

---

## 5. Required checks (lightweight)

Run on CPU before the formal matrix and recorded in `cpu_gates_this_round.json`:

1. Parameter counts and group inventory; no old reader retained.
2. The P3 bypass equals the original P3 reader input under the same parameters (exact / float32
   tolerance), using the original Null W/V code path.
3. Atom renumbering, bond endpoint swap and junction endpoint swap do not change eval
   predictions (float32 tolerance).
4. Token row permutation with synchronised relation matrices does not change type-wise sums.
5. Batch composition does not change predictions; padding and another molecule cannot affect a
   molecule's prediction; empty bond/junction synthetic boundaries compute finite outputs.
6. R distance/overlap bins cross-checked by hand on a small path graph; a disconnected synthetic
   case returns the `disconnected` bin.
7. T and R with zero relation bias have identical step-0 outputs; R bias tables are zero at init.
8. One- and two-update gradients are finite; the new branches participate; R bias receives a
   non-zero gradient and is non-zero after training.
9. Batch-order hashes match the frozen R3 hashes for both seeds.
10. Frozen P3 soup hashes are present; the full label-free P3 valid replay lives in the audit and
    must agree with the historical arrays to <= 5e-5.

No full historical audit, new Ridge probe or collision certificate is run. Label-free sensitivity
is not treated as a benefit.

---

## 6. Training

Exactly six formal runs: S/T/R x seeds 0/1.

| item | value |
|---|---|
| train rows | official train 10,000 |
| loss | raw-y L1 |
| batch | 128 |
| optimiser | Adam, lr 1e-3, per-step cosine -> 1e-5, wd 1e-5 |
| clip | 5.0 |
| epochs / updates | 360 / 28,440 |
| soup | mean of the last 5 epoch parameter states |
| early stopping / valid selection | none |
| precision | FP32, no AMP, no `torch.compile` |
| shuffle | `torch.Generator().manual_seed(seed + 101)`; R3 batch-order hash is a hard assertion |

Schedule:

| GPU | seed | arm order |
|---|---|---|
| `res-gpu0` | 0 | S -> T -> R |
| `res-gpu1` | 1 | R -> T -> S |

One training process per GPU. Seeds are fixed per GPU; arm order is reversed to decorrelate load
drift from arm identity.

Smoke: two epochs on a 1,024-row train prefix for S/T/R, temporary models, no valid labels, no
architecture selection. The formal matrix starts only if
`1.20 * projected_formal + smoke_wall + 300 s <= 4.0 A100-hours`. If the projection does not fit,
the round stops and reports the implementation and cost projection.

Timing-estimator amendment (before any formal training; architecture, recipe and
compute unchanged): the first smoke run used the raw mean over all 16 updates and
projected 4.72 GPU-h (`smoke_this_round_v1_mean_failed.json`). The first update of
each arm includes CUDA/library warm-up and is not representative of a 28,440-update
run. The frozen projection therefore uses the mean over updates 3..16 of each arm
(drop the first two per arm), with the same 1.20 margin, smoke wall and 300 s
reserve. This amendment changes only the timing estimator, not the model, data,
epochs, batch size, optimiser, scoring rule or cap. Both smoke records and the
amendment note are kept.

### 6.1 P3 reuse

The P3 reference is reused, not retrained, only under Branch A: the execution regime must match
the recorded P3 regime (A100-SXM4-40GB, driver 550.163.01, torch 2.5.1+cu124, CUDA 12.4, Python
3.12.14), the frozen soups must rebuild from the historical init/encoder, and the label-free valid
replay must agree with the historical arrays to <= 5e-5. If the regime does not match, Branch B
would require two same-round P3 R3 runs; if the budget cannot contain them, the round stops
rather than comparing to a mismatched baseline.

---

## 7. Freeze and score

Order is fixed:

1. `prep` — audit, CPU gates, FLOPs.
2. `smoke` — timing only; no valid labels.
3. Six formal trainings; each writes `train_{arm}_s{seed}.json` and `soup_{arm}_s{seed}.pt`.
4. `predict-all` — six single-model valid predictions, each file/array hashed and frozen; no labels.
5. `mechanism` — label-free response statistics on the first 128 valid **structures**; no labels.
6. `freeze` — combined pre-score hash record.
7. `score` — one valid-label read, only after the pre-score freeze verifies. A second read is
   refused unless `--recover` is explicitly used and recorded as a deviation.

The valid split is the historical development split and is already exposed. This round records
its own accesses in `valid_access_log.json` and does not claim that the whole research program has
only one valid access. Official test is never loaded, parsed or predicted.

---

## 8. Metrics and decision

Primary metric: per-seed raw-y MAE, then the mean of the two seed MAEs. No prediction ensemble.
`gain = MAE(reference) - MAE(candidate)`; positive is better.

Paired bootstrap: 5,000 resamples of molecules, seed 20261019, both seeds resampled synchronously,
95% percentile CI. The CI describes sample fluctuation for these two training instances only; it
does not cover all training randomness.

Robustness sets (descriptive, pre-registered): ordinary (`y > -3`), drop the 10 molecules with the
worst mean P3 error, and drop original valid row 172.

The practical line for an arm against P3 is **all** of:

* mean gain >= 0.003;
* both seed gains > 0;
* ordinary gain > 0;
* drop-P3-worst-10 gain > 0;
* paired 95% CI lower bound > 0.

0.003 is this round's budget-dependent decision line, not a universal constant.

Consistent positive evidence for a specificity contrast is: both seed gains > 0 and paired CI
lower bound > 0. A non-significant contrast is written as "no detected gain", never as
equivalence.

Fixed contrasts: S/T/R vs P3, T vs S, R vs S, R vs T.

Decision labels:

* `R_STRUCTURAL_RELATION_CANDIDATE` — R passes the practical line and is consistently positive
  over S and T. Supports further study of whether structural relations improve composition; not
  a novelty or causal claim.
* `T_OBJECT_ATTENTION_NO_EXTRA_STRUCTURE` — T passes the practical line and is consistently
  positive over S, while R is not consistently better than T. Supports ordinary object interaction,
  not an extra structural-bias effect.
* `S_CAPACITY_READOUT_DOMINATES` — S passes the practical line while T/R show no consistent extra
  advantage over S. The gain is more likely readout/capacity reallocation; do not claim attention.
* `R_STRUCTURE_ADDS_OVER_S_NOT_T` / `T_OBJECT_ATTENTION_NO_GAIN_OVER_S` — descriptive fallbacks.
* `NO_OBJECT_ATTENTION_PRACTICAL_GAIN` — no arm reaches the practical line. Close this
  object-attention architecture screen; do not add layers, heads, relation buckets or seeds.
* `INCONCLUSIVE_MIXED_CONTRASTS` — at least one practical pass but mixed specificity.

---

## 9. Cost reporting and claims

Report parameters, actual training wall time, prediction cost, projected/analytic FLOPs, and the
GPU ledger. Equal training steps do not imply equal compute; this round is not called strictly
compute-matched. No extra seed, dropout/LR/depth search or attention residual scaling is run.

No claim is made that this round beats literature GRIT/ESA+; no official-test evaluation and no
full mature-model reproduction is attempted. "Structure tokens + attention" itself is prior art
(SAT, TokenGT, GPM, ESA, GRIT and related work); the only question here is whether the explicit
object-token readout and its structural relation bias are useful on this frozen P3 substrate.

### Access disclosure

Before this protocol was frozen, a local environment probe called the frozen valid-label loader
once to check shape/hash plumbing and printed the shape and three values. It was not used for any
architecture, arm, threshold or selection decision, and no model prediction was read. This is
recorded honestly in `valid_access_log.json`; the formal score still uses one concentrated read.
