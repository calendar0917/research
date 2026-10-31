# PROTOCOL — `lcrc-lipo-composition-discovery-v1`

Exploratory architecture discovery on the frozen Lipophilicity random42 split.
Validation420 is the Stage-I screening metric.  Test420 is read only for the
candidates selected before Test is opened and is always reported as
**`EXPOSED_TEST_EXPLORATORY`**.

| field | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-composition-discovery-v1` |
| Experiment root | `tracks/structural-sufficiency/lcrc_lipo_composition_discovery_v1/` |
| Split | Train3360 / Validation420 / Test420, split seed 42 |
| Training seed | 0 for Stage I; seed 1 only for selected Stage-II arms |
| Test access | exposed exploratory, never used for training or selection |
| Host / pools | `res`, `res-gpu0` and `res-gpu1`, GPU budget 14 A100 GPU·h |
| Primary metric | Validation420 RMSE on raw logD |
| Checkpoint rule | average of the last five epoch-end state dicts; no early stopping |

## 1. Data

Frozen read-only inputs:

* `lcrc_lipo_esa_random42_v1/data/split_esa_random42_v1.json`
  SHA256 `1834414a…33b4`
* `lcrc_lipo_esa_random42_v1/data/labels_train_valid_v1.json`
  SHA256 `1da26935…09ca`
* `data/lipophilicity/Lipophilicity.csv`
  SHA256 `aed41590…bacd`
* LCRC cache content fingerprint `23ec00be…8e03`, label-free.

Train/Validation/Test row hashes are those of the frozen split.  Target
standardisation is fitted on Train3360 only (`mean 2.2081011904761905`,
`scale 1.1906281212426577`); structure normalisation is the frozen Train3360
fit.  The B-family distance buckets are pure structure (unweighted shortest
paths) and contain no labels.

## 2. Stage I arms (seed 0)

Eight arms, all with the frozen LCRC backbone, S1/S2 local content, `f`, `rho`
and the frozen training recipe:

| arm | definition |
| --- | --- |
| `g2l_all` | ROLEBLIND four layers; correction applied to **all** atoms |
| `g2l_loo` | ROLEBLIND modules; context is the per-atom leave-one-out mean; hetero atoms only |
| `g2l_local_cap` | ROLEBLIND modules; `p_i=tanh(W_het h_i)`, `c_i=tanh(W_ch h_i)`; hetero atoms only |
| `g2l_scalar` | original `h`; scalar hetero-atom correction from the all-atom context |
| `dist_true` | per-atom D34/D5 context from real shortest-path buckets |
| `dist_blind` | identical parameters/masks; bucket content replaced by the leave-one-out all-atom mean |
| `path_bind` | decomposed S2 role embedding bound to the true path-endpoint content: `mean(v*r)` |
| `path_unbound` | identical parameters; two marginals: `mean(v)*mean(r)` |

`rho` always reads the original `h`.  `f` reads the corrected representation
except in `g2l_scalar`, where the correction is an explicit graph-level sum of
per-hetero-atom scalars.  No new message passing, no iterative update, no
full-graph attention and no extra pre-training are used.

## 3. Training recipe (all arms)

AdamW (`beta 0.9/0.999`, `eps 1e-8`), batch 128, 360 epochs, 27 steps/epoch
(9,720 updates), peak LR `3e-4` -> `1e-5` per-step cosine, weight decay `1e-3`,
gradient clip 5.0, MSE on Train3360-standardised logD, dropout 0.10 on the final
atom-level local representation (`DropoutAtLocal` attached to `model.local`),
last-five-epoch soup.  The global RNG is reseeded after model construction; the
shuffle generator seed is `seed + 917000`.  Shared BASE parameters are
bit-identical to the same-seed BASE initialisation.

## 4. Correctness gates before CUDA

All gates in `code/dc_gates.py` must pass and are frozen with the code:

1. frozen split/label/cache/normalisation hashes;
2. Train-only target scaler;
3. shared BASE initialisation and parameter counts; mechanism pairs are
   parameter- and initialisation-identical;
4. zeroing the correction output projection exactly recovers BASE;
5. D34/D5 independently recomputed from shortest paths; self, distance <=2 and
   unreachable excluded; the two buckets are disjoint;
6. S2 path-category decoding and independent path enumeration agree;
7. PATH-BIND/PATH-UNBOUND agree for one path and for constant role/content,
   and differ for mixed multi-path roots;
8. batch-composition/order and cross-molecule isolation;
9. atom-renumbering and path-instance-order invariance;
10. finite non-zero gradients and parameter updates for every new tensor;
11. `rho` reads original `h`; `f` reads `h_new`;
12. dropout masks in train mode and is the identity in eval;
13. frozen BASE_s0 / ROLEBLIND_s0 reproduce their Validation420 RMSE;
14. frozen-checkpoint inference changes no weight;
15. metric/denormalisation arithmetic and a static training-code Test-scope scan.

## 5. Stage II selection (resource gate, not significance)

After Stage I, at most two candidates may be promoted, preferably from different
families.  A candidate is eligible if at least one holds:

* Validation420 RMSE improves on ROLEBLIND_s0 by about `0.008` or more and MAE
  does not clearly worsen; or
* the true arm beats its matched control (`path_bind` vs `path_unbound`,
  `dist_true` vs `dist_blind`) by about `0.008` on Validation420 while its
  absolute RMSE is not clearly behind ROLEBLIND.

For a promoted candidate run seed 1; if it is a true arm, run its matched
control at seed 1 too.  `ROLEBLIND_s1` is run to pair with the existing
`BASE_s1`.  No seed search and no best-seed selection.

## 6. Stage III/IV (conditional)

Only if a candidate is positive at both seeds without a catastrophic regression
may the official Lipophilicity scaffold 3360/420/420 split be used, with
Train-only scalers refit on that split.  Test numbers from that split are a
development-distribution shift check on the same 4,200 molecules, not an
external independent test.  PATH-SELECT (single root-conditioned softmax over
paths) is attempted only if PATH-BIND shows both a performance and a binding
advantage; it is not attributed as an attention mechanism without a separate
capacity control.

## 7. Reporting

Every Test number is `EXPOSED_TEST_EXPLORATORY`.  For key comparisons report
paired RMSE/MAE/SSE differences, the fraction of molecules improved, descriptive
paired-molecule bootstrap intervals, the five fixed tail rows and their removal,
scaffold seen/unseen relative to Train3360, molecule-size/TPSA groups, and the
mechanism-usage statistics.  `PERFORMANCE` and `RELATIONAL_SPECIFICITY` are
reported as independent dimensions.  No causal chemical claim is made from
mechanism execution alone.
