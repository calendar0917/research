# Pre-registration — ZINC E2E-DictEnv-Hier-Relation-v1 (HIER-RELATION, seed 0, local CPU)

Round: `zinc-e2e-dictenv-hier-relation-v1`; runner
`zinc_e2e_dictenv_hier_relation_v1`; protocol `zinc-context-gap`
(`test_policy: terminal`, non-terminal runs must not touch official test).
Frozen before the formal prepare/screen runs; the code and this note are
committed before training starts (commit recorded in §8).

Write-then-follow: nothing below may be changed after the endpoint is seen.

## 1. Question

The frozen Joint709 dictionary produced no gain and its coordinate was inert at
the endpoint (the multiplicative binding collapsed to the float32 denormal floor,
so zero/shuffle probes were exactly 0.0). This round changes **how the
dictionary enters the prediction**, in a single model:

```
local environment -> learnable node dictionary (all 128 coordinates)
                  -> within-molecule, same-atom-type shared environment mu
                     + individual deviation delta
                  -> static combination (one layer, no write-back)
                  -> learnable sparse relation dictionary (228 x 64, l0 = 8)
                  -> one graph readout -> one head
```

The relation object keeps, in one tensor, (a) the shared environment of the two
endpoints, (b) the deviation of each endpoint from its shared environment, and
(c) their combination. All three go into the *same* relation dictionary and the
*same* graph readout; there are never two independently trained models and the
predictions are never averaged.

This round screens **absolute performance only**. It buys **no** matched-training
control (no dense/PCA arm, no random/shuffled dictionary arm) and **no** seed 1.

## 2. Frozen inputs (re-used read-only, verified at run time)

| object | identity |
|---|---|
| official train / valid | 10 000 / 1 000 molecules (PyG ZINC `subset=True`) |
| joint scaler | `results/e2e_dictenv_joint709_absolute_v1/joint_standardizers.json`, sha256 `0ea418fa…193060` |
| joint cache (train) | `cache/corr_joint_train.pt`, 231 664 x 709, shipped-file sha256 `b7da4437…aa87b` |
| joint cache (valid) | `cache/corr_joint_valid.pt`, 23 083 x 709, shipped-file sha256 `224adf5a…6e85` |
| common subspace | `…/common_subspace.json`, sha256 `36636ce9…8f6c24`, `q1` `[65,1]`, rms 5.082852828320509 |
| correspondence cache | `…/e2e_dictenv_rolecorr_v1/cache/corr_raw_{train,valid}.pt` (node/edge sha256 match the Joint709 record) |
| env cache | `…/e2e_dictenv_p1/cache/env_{train,valid}.pt` |

**No K-SVD / PCA object is refit** (`stage_joint_objects` and the old
`prepare` chains are never called). The frozen Joint709 prepare is re-used.

### 2.1 Provenance nuance found this round (recorded, not repaired)

The upstream `cache_meta_joint_{split}.json` field `sha256_f32` is the hash of the
**float64** array *before* the float32 serialisation (`build_joint_cache` hashes
the float64 `scaled` array and then saves `torch.float32`). It therefore does not
reproduce as a hash of the shipped file. This round records it as
`sha256_f64_recorded_upstream` and verifies content instead by recomputing
`apply_joint_scaler(struct_residual, Sem108, corr)` in float64 for a molecule
prefix of both splits (tolerance `1e-5`; measured `9.0e-7` train / `4.5e-7` valid
in the authoring run). The shipped-file float32 hashes above are the identity
used by the downstream stages.

## 3. Model (one model; no message passing, no write-back)

* Node input `x_i = [Joint709_i ; common1_i ; size2_i]`, `d_x = 712`
  (`common1 = (dict_phi @ U)/rms` from the frozen common subspace,
  `size2 = anchor[:, 60:62]`). The 709 joint block re-uses the frozen three-block
  scale; the appended `common` (1) and `size` (2) blocks get their own train-only
  per-column RMS + block-energy rule so each block's train mean squared norm is 1.
  Only fixed zero-RMS columns may be masked (joint block: 0 masked; relation:
  1 masked column, the absent bond category). `d_x`, mask and statistics recorded.
* Node dictionary `D_N [712, 128]` trainable, `D̄_N` = column-normalised,
  `z_i = x_i @ D̄_N`, `x̂_i = z_i @ D̄_Nᵀ`. **All 128 coordinates are kept**
  (no top-k), and every local input goes through `D̄_N` (no raw bypass).
* Initialisation: top-128 eigenvectors of the **uncentred train second moment**
  over at most 32 768 deterministically sampled train nodes (`numpy` float64
  accumulation, sign fixed by largest-|component| positive). If the effective
  rank is below 128, deterministic normalised supplements (seed 0) are used and
  recorded; K is not changed, no K-SVD runs, no PCA arm is added.
* Grouping: per graph and atom category (`dict_atom`, 28 classes),
  `mu_i` = group mean of `z`, `delta_i = z_i - mu_i`. This is a side read for the
  relation object; the node readout always reads the original `z_i`.
* Fixed projection `P [128, 32]`, i.i.d. `±1/sqrt(32)`, own CPU generator seed
  `20261001`, registered as a non-trainable buffer. All 128 coordinates are used;
  no learnable endpoint MLP.
* Relation object, one per **unique undirected physical key** (`u < v`), only once:
  `S(a,b) = [a+b ; |a-b| ; a*b]`,
  `r_uv = [S(m_u,m_v) (96) ; S(d_u,d_v) (96) ; m_u*d_v + m_v*d_u (32) ; onehot(bond,4) (4)]`,
  with `m = mu @ P`, `d = delta @ P`. Endpoint-swap invariant, 228 dims.
* Relation scaler: four blocks, train-only per-column RMS + block energy, fit on
  **all** train physical edges under the **initial** `D_N` and the fixed `P`, then
  frozen (never refit during training, at checkpoints or on valid). Relation input
  block-energy drift during training is recorded.
* Relation dictionary `D_R [228, 64]` trainable; `β_uv = tied-IHT(r_scaled, D̄_R,
  s=8, steps=10)` (the repository-validated `e2e_dictenv_v0.tied_iht_codes` step
  size / hard-threshold), `r̂ = β @ D̄_Rᵀ`; gradients flow into both dictionaries
  (the relation object is never detached). Initialisation: top-64 eigenvectors of
  the train second moment of the *scaled* relation objects over a deterministic
  32 768-edge sample under the same initial `D_N`/`P`; same supplement rule.
* Graph readout: `sum/mean/population std` of the 128 node coordinates (384) and
  of the 64 relation coordinates (192), `log1p(nodes)`, `log1p(unique edges)` (2),
  `topology_features(25) -> Linear(16) -> ReLU -> Linear(8)` (8) = **586**.
  `std = sqrt(var + 1e-8)`; edge-empty graphs give an exactly zero relation
  pooling block (no NaN). One head: `586 -> 64 -> SiLU -> Dropout(0.05) ->
  32 -> SiLU -> Dropout(0.05) -> 1`.
* Trainable parameters = `128 * d_x + 54 825` = **145 961** at `d_x = 712`
  (`P` and the scaler gains are buffers, 5 039 elements).
* Structurally forbidden: message passing, GNN/Transformer/attention, recursive
  updates, any relation -> node write-back, stacked relation layers (depth 1),
  raw local input bypass, older model heads, Sem108 bypass, φ65 reconstruction
  interface, `pair_index` as adjacency.

## 4. Loss and training protocol

```
loss = MAE + 0.05 * R_N + 0.02 * R_R
R_N = mean(||x - x̂||²) / (mean(||x||²) + 1e-12)
R_R = mean(||r_scaled - r̂||²) / (mean(||r_scaled||²) + 1e-12)   (0 when the batch has no edge)
```

Real inputs and real reconstructions only; no zero-reconstruction placeholder and
no φ65 term. Auxiliary-loss gradients propagate; additionally the **MAE-only**
dictionary gradient is measured on a fixed probe (see §6) to show the dictionary
is not driven by the reconstruction terms alone.

Protocol: local CPU, `torch_threads = 8`, seed 0, official train 10 000 / valid
1 000, batch 128, Adam (LR `1e-3`, weight decay `1e-5`), gradient clip 5.0,
**320 epochs, fixed LR, no scheduler**, official valid evaluated every epoch,
Top-5 checkpoints by valid MAE averaged into a **soup** (whole state dict; the
buffers must be identical and are averaged only for parameters), no
valid-tuned soup weights, no early stop. The soup is one model, not an ensemble.
The run starts from the saved initialisation state and seed 0; correctness and
smoke use independent model instances and the formal training never warm-starts
from them. Resumable every 10 epochs (optimiser / epoch / RNG state saved).

The official ZINC **test** split is never loaded (the stage module has no code
path that can open it; the control plane records `test_access: blocked`).

## 5. Acceptance gates before the 320-epoch run

1. 712-D input, actual mask, 228-D relation object and 586-D readout widths.
2. Every undirected physical key counted exactly once with the right type and
   graph ownership (dedup from `env_bond_u/v/type`, spot-checked against raw ZINC).
3. Node relabelling, endpoint swap and batch composition leave the prediction
   unchanged within a fixed `1e-5` tolerance.
4. Grouping never crosses graphs; singleton groups have exactly `delta = 0`.
5. Re-pairing relation endpoints never changes `z`, `x̂`, node pooling or `mu`.
6. Both dictionaries receive finite non-zero MAE-only gradients and their
   normalised directions actually move under a MAE-only Adam step.
7. Reconstruction formulas match an independent hand recomputation; relation code
   `l0 <= 8`.
8. zero/shuffle interventions change only the target branch (counts, topology and
   the non-target pooling stay identical).
9. The test blocker is effective (no `env_test` cache, no test split reference).

All gates must pass before training; a failure is an implementation bug and is
repaired (never worked around by weakening a gate).

## 6. Activity diagnostics (fixed train probe, no parameter update)

On a fixed probe of the first 256 train molecules, at epochs
`0, 5, 20, 80, 160, 240, 320`, recorded as diagnostics only (never a gate):
MAE-only gradient norms into both dictionaries (raw parameters and normalised
directions); node/relation code norms and variances; both real relative
reconstruction errors; the four relation input block energies; prediction RMS
change after zeroing the relation codes, the node pooling, the relation `delta`
and after two in-group endpoint permutations (seeds 11 / 22). The probe does not
update parameters, does not touch the training data order, the optimizer state or
the RNG (gradient measurement uses `torch.autograd.grad` on a scratch copy in
eval mode). Training is never stopped early because of the valid curve.

Endpoint interventions (on the soup, and the best checkpoint for reference) are
reported as MAE delta and prediction-RMS change on official valid. Pre-registered
sensitivity indicator: prediction change `RMS > 1e-4`. Sensitivity is *not* a
performance claim; and "the intervention hurts MAE" only means the channel
carries prediction inside this model — it does not show that the channel is
better than a matched-training control, nor that sparse coding beats dense.

If the node path is alive and the absolute performance is good while the relation
path does not carry prediction, the two statements are reported **separately**:
"the node-dictionary model has promising absolute performance" and "the relation
dictionary's value was not demonstrated in this round".

Note (frozen, not tuned): a zeroed branch contributes exactly `sqrt(1e-8) = 1e-4`
in the std slots of its pooling block, which is the pre-registered sensitivity
floor.

## 7. Decision rule (absolute performance only)

`soup_valid_mae` (official valid, soup state):

| MAE | judgement |
|---|---|
| <= 0.115 | strong signal, recommend follow-up confirmation |
| <= 0.120 | promising, recommend follow-up confirmation |
| <= 0.1233 | borderline signal, no automatic follow-up |
| > 0.1233 | stop the current configuration |

The band and the mechanism judgement are reported separately. Historical numbers
(A/B/C, Sem108, Joint709, the legacy RPD 0.11903) are background only; this round
makes no matched-gain and no significance claim.

Budget: 4 h wall clock total, one seed-0 candidate, no control arm, no seed 1, no
K/s/width/LR/lambda/horizon sweep, no second model. If 320 epochs do not fit the
remaining budget the run is delivered as `INCOMPLETE_BUDGET` with a resumable
state — the horizon is not shortened and then declared a failure.

Inertness rule: if a channel is inert at the endpoint, only one bounded (<= 10
min) localisation pass is allowed; implementation bugs may be fixed,
initialisation/regularisation/structure/loss changes are new candidates and are
**not** run in this round.

## 8. Provenance to record

Revision (authoring commit), protocol hash, split fingerprint, the frozen input
hashes of §2, `P` sha256, both scaler JSONs, the init-state sha256, the soup/best
state sha256, environment (torch, threads, device, RSS), per-epoch wall clock, and
the promotion ids of the prepare and screen runs. Large checkpoints stay out of
Git.
