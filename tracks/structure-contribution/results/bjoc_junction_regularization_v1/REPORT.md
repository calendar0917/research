# cscl-bjoc-junction-regularization-v1 — REPORT

One predictive intervention, frozen before training: elementwise inverted dropout on the learned junction embedding before the null incidence grouping (R0 `p=0.0`, R1 `p=0.10`), everything else identical.  Official valid and official test were never loaded.

## 0. Decision

**`JUNCTION_DROPOUT_HURTS_GENERALIZATION`**

R1 was materially no better than R0 on at least one held-out readout and not favourable on either; the frozen regularisation did not buy generalisation in this regime (monitor Δ +0.01056, locked Δ +0.01010, ordinary molecules Δ +0.00871, top-10-removed locked Δ +0.01036)

## 1. Table 1 — training and generalisation

| Split | Seed | R0 fit | R1 fit | R0 heldout (locked) | R1 heldout (locked) | R0 monitor | R1 monitor |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0.033786 | 0.060690 | 0.137930 | 0.152903 | 0.098953 | 0.114460 |
| A | 1 | 0.036416 | 0.050343 | 0.148650 | 0.153880 | 0.105001 | 0.110613 |

## 2. Table 2 — paired improvement (Δ = R1 − R0; negative favours R1)

| Split | Seed | Monitor Δ [CI95] | Locked Δ [CI95] | Ordinary locked Δ | Low-y tail Δ | Better-molecule frac | Max-row removal Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 0 | +0.015507 [+0.009339, +0.022074] | +0.014973 [+0.007233, +0.024285] | +0.008715 | +0.028809 | 0.425 | n/a |
| A | 1 | +0.005613 [-0.000407, +0.011472] | +0.005230 [-0.001297, +0.011318] | +0.008715 | +0.028809 | 0.443 | n/a |
| A | mean | +0.010560 [+0.006709, +0.014528] | +0.010101 [+0.005413, +0.014749] | +0.008715 | +0.028809 | 0.401 | +0.010358 |

cross-seed (A): R0 locked MAE per seed ['0.137930', '0.148650'] (spread 0.010720); R1 ['0.152903', '0.153880'] (spread 0.000977); Δ per seed ['+0.014973', '+0.005230'], Δ difference +0.009744.

## 3. Table 3 — mechanism (frozen checkpoints, eval-mode counterfactuals)

| Quantity | R0 | R1 |
|---|---:|---:|
| mean abs(w), fit sample | 0.06260 | 0.09100 |
| fraction abs(w) < 1e-3, fit sample | 0.03663 | 0.02562 |
| monitor MAE after junction_zero | 0.35490 | 0.31457 (Δ -0.04033) |
| monitor MAE after junction_within_molecule_permutation | 0.10198 | 0.11254 (Δ +0.01056) |
| monitor MAE after junction_cross_molecule_shuffle | 0.42419 | 0.35887 (Δ -0.06531) |

Mechanism reading: dropout did move the frozen system in the intended direction. R1's junction embeddings are larger (0.0910 vs 0.0626 mean |w|) and less frequently near zero (0.0256 vs 0.0366), and cross-molecule replacement of junction content damages R1's monitor predictions less (-0.06531) — the trained model relies less on the molecule-specific junction multiset. But the price is paid in fit: R1's soup fit is 0.0555 vs R0's 0.0351, and the held-out readouts are worse. The within-molecule permutation is a numerical no-op (|Δpred| ~ 4e-8, grouping conservation error ~2e-7), confirming that the null architecture sees the junction channel only through permutation-invariant multiset sums: 'which junction' is not recoverable from this predictor, only the multiset of embeddings. The regulariser therefore flattens/shifts embeddings, not junction selection. The seed spread of R1's locked MAE is smaller than R0's (0.00098 vs 0.01072): dropout stabilised the trajectory but toward a worse optimum.

## 4. Phase A — historical audits (zero training)

Monitor vs locked of the completed representation round: y≤−3 fraction 0.076 vs 0.080; MAE 0.14067 vs 0.09606; top-10 error mass share 0.345 vs 0.094.
N0/N1 full-10k curves: soup fit 0.05210 / 0.03952; junction-encoder gradients live on both frozen checkpoints: True / True.

## 5. Pre-registered promotion gate

Training budget used: 4 formal trajectories (split A only), 1498 s of train wall on one A100 (≈0.42 GPU-h). Split B was **not** trained: the frozen gate requires a purchase on split A, and all five substantive checks failed; the stop rule forbids any further trajectory or p search after a failed gate. `split_b.json` (the frozen split definition) exists, but no split-B metrics, checkpoints or predictions exist.

Split A: `JDROP_NO_REPLICATION_PURCHASE`

```json
{
  "both_seed_locked_delta_non_positive": false,
  "both_seed_monitor_delta_negative": false,
  "dropout_and_correctness_gates_pass": true,
  "locked_seed_mean_delta_le_-0.003": false,
  "monitor_seed_mean_delta_le_-0.006": false,
  "not_driven_by_single_extreme_row": false,
  "ordinary_molecules_not_worse": false
}
```

## 6. Next-stage CSCL decision (no training this round)

**NEXT_STAGE_DECISION: `PROPOSE_IDENTIFIABLE_JUNCTION_CONTRIBUTION_MODEL` (rank 1).** Two mutually exclusive readings of this round's result are considered. (a) *Regularisation failure*: junction dropout is simply not a useful regulariser here (fit cost 0.020, held-out cost 0.010). (b) *Mechanism mismatch*: the frozen null predictor cannot express per-junction contributions at all — its junction block is a per-molecule multiset average — so a dropout regulariser is forced to operate on the wrong object and necessarily trades fit for a shifted but not better solution. Reading (b) is favoured by the conservation audit (within-molecule permutation is a no-op) and by the observation that the intended mechanism change did occur (larger |w|, smaller cross-molecule-shuffle damage, smaller seed spread) while held-out error rose. Both readings agree on the next stage: do not tune dropout further; instead build the object that the CSCL question actually needs — an identifiable, ownership-preserving per-junction contribution model — and require it to pay only a pre-registered predictive-cost band.

**Proposition (identifiable structural contributions from the learned junction system).**

1. *Contribution object*: a typed junction object (centre atom environment + ordered pair of incident bonds) carries a scalar task contribution ν_j = f_θ(shared junction type, context descriptor) with an explicit global bias b and atom/bond base terms (α_i, β_uv). The unit of attribution is the junction instance with its true bond ownership preserved.
2. *Shared types and context*: junction types are canonical local colourings built from the existing label-blind raw features; context enters through pre-frozen, centred descriptors of the two incident bonds and the local neighbourhood, so a type's baseline contribution is comparable across molecules.
3. *Additive terms and necessary interactions*: pred = b + Σ_i α_i + Σ_uv β_uv + Σ_j ν_j + Σ_{jk} g(τ_j, τ_k, relation) for junctions sharing a bond; no higher-order learned interaction beyond the pre-registered first-order closure term.
4. *Completeness identity*: the exported per-object contributions must sum to the prediction to numerical tolerance on every molecule (conservation gate), with UNK buckets for types unseen in the fit split.
5. *Duplicate counting*: every junction, atom and bond is assigned to exactly one contribution object; a junction touching two bonds is one object, not two.
6. *Gauge*: context and relation descriptors are centred on fit rows per type, so the type mean lives in the additive term and the interaction term has zero fit-domain mean; shared-type contributions are identified up to the declared centring convention, which must be exported with every attribution.
7. *Synthetic validation*: before touching ZINC, verify identifiability on a synthetic task with known junction contributions and a co-occurrence confound, including recovery of the true interaction sign and a false-interaction rate measurement.
8. *Predictive cost*: compare the decomposable model against the current opaque reader at matched parameters and recipe; the decomposition is only a candidate if the loss penalty stays inside a pre-registered band.
9. *Seed stability*: contribution vectors for the same junction type must correlate across seeds above a pre-registered threshold, and the sign of the top contributions must be stable.
10. *Boundary*: these are prediction attributions, not chemical causal effects; the round can only claim that the model's computation is faithfully decomposed.

Important constraint discovered this round: the frozen BJOC-Null uses a graph-level junction mean, so no per-junction attribution can be read out of its weights or predictions. Any identifiable-contribution model must re-introduce true junction→bond ownership and be re-evaluated for predictive performance; its contribution scores cannot be validated by the current Null predictor.

## 7. Artifacts

* `dropout_correctness_gates.json`
* `global_norm_a.json`
* `global_norm_b.json`
* `historical_error_audit.json`
* `historical_training_audit.json`
* `junction_regularization_mechanism.json`
* `objects_summary.json`
* `parameter_and_init_audit.json`
* `prediction_manifest_A.json`
* `protocol_freeze_manifest.json`
* `scientific_decision.json`
* `split_a.json`
* `split_a_label_access_audit.json`
* `split_a_metrics.json`
* `split_a_paired_bootstrap.json`
* `split_a_promotion_gate.json`
* `split_b.json`
* `split_crosstab.json`
* `train_R0_A_s0.json`
* `train_R0_A_s1.json`
* `train_R1_A_s0.json`
* `train_R1_A_s1.json`
* `train_manifest.json`

