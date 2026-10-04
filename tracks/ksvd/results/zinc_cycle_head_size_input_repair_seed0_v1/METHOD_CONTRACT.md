# METHOD_CONTRACT — zinc-cycle-head-size-input-repair-seed0-v1

Frozen before any head training (2026-10-04).  Single change under test: the
independent cycle head receives the actual graph size (N, E) in addition to the
frozen `T25`.  No chemistry-branch change, no dictionary/ISTA change, no
re-weighting, no input/width/epoch/optimizer search, no extra seed, no
ensemble/combination.

## Frozen sources (verified by hash in `identity_checks.json`)

| object | path | hash |
|---|---|---|
| T25 input, all 10000 | `zinc_frozen_chemistry_learned_cycle_v1/T25_all.npz` | array `dc2e1516…` |
| fit/dev split | same file | `165e87ef…` / `fb8b7806…` |
| frozen O raw soup | `zinc_full_cycle_target_decomposition_v1/O_seed0_raw_soup_state.pt` | file `61d4aeb…` |
| frozen O predictions | `…/O_seed0_predictions.npz` | file `5cee5782…` |
| c/g/k decomposition | `…/target_decomposition.npz` (train-only) | `decomp_sha256 02d44958…` |
| prep blob (8000-fit) | `zinc_joint_dictionary_decision_v1/prep/fold_objects.npz` | file `968e82dd…` |
| actual train graphs | `zinc_dictionary_real_data_handoff/train.npz` | `official_test_loaded=False`, `split=train` |

`c`, `g`, `k` are label-derived cycle targets (`c = (snapped − mu)/sigma` on the
node-order-dependent cycle basis).  They are used only as supervision and
evaluation labels; no inference path receives them.  Their use is extra
label-derived supervision and is disclosed in the REPORT.

## inputs

* `T25` is the existing Full `topology25` model input after the 8000-fit prep
  (unchanged columns and order; `prep topology vs T25 maxdiff = 0.0`).
* `N` = `num_nodes` of the actual graph (isolated nodes included).
* `E` = number of unique undirected pairs `{min(u,v), max(u,v)}` of the actual
  graph (NOT the number of PyG `edge_index` columns).
* scaler: unweighted graph-level mean/std over the 8000 fit rows, float64, std
  floor 1e-6, frozen for fit and dev: `N: mean 23.18275, std 4.488552`; `E:
  mean 24.9435, std 5.288602`.
* `X25 = T25`; `X27 = [T25, (N−mean)/std, (E−mean)/std]`, float32.
* forbidden in N/E: atom/bond labels, `y`, `c`, `k`, molecule id, SMILES,
  precomputed cycle targets.  N/E are recomputed from the graph at deploy time
  by `deploy_wrapper.size2_from_graph`; stable ids are used only for row
  alignment.

## exact-class semantics

Exact float32 equality (numpy `==`, no tolerance, no neighbour search); NaN is
forbidden and signed zeros are checked.  A class is "conflicting" when it has
>= 2 rows and > 1 distinct `c`.  The fit-wide class-median output cost by group
is descriptive (`sum |c − class_median|` split by group / group n); it is **not**
called an irreducible lower bound.

## heads

```
z = Linear_T25(25, 64) + gate * Linear_size(2, 64, bias=False)
q = Linear(32, 1)(SiLU(Linear(64, 32)(SiLU(z))))
```

* `Q0`: gate = 0 (T25 effective only; `W_S` kept in the optimizer, exactly
  zero throughout).  `QNE`: gate = 1.
* 3905 parameters per arm (3777 original + 128 `W_S`).
* init: the original seed-0 fresh 25-dim `Linear/SiLU` stack, last layer
  `weight=0`, `bias=median_fit(c)=0.0004627120960246`; `W_S = 0`.  Verified
  tensor-equal to the published `cpu_P_U_head_init_state.pt`.
* recipe: Adam lr 1e-3, coupled wd 1e-5, global clip 5, batch 128, unweighted
  mean `L1(q, c)`, 300 epochs (18900 steps/arm), FP32 CPU 8 threads, no
  scheduler/early-stop/AMP/DDP.  Schedule: `torch.Generator(seed=20261003)`,
  one `randperm(8000)` per epoch, precomputed once and shared by both arms
  (`schedule_hash d836d9b3…`).  Final model: mean of epoch-end states 296–300.
* engineer smoke (4 fresh steps, discarded): forward, real backward and the
  zero-init path; `W_S` has zero gradient at step 1 (last layer `W=0`), nonzero
  from step 2 (`w_s_grad_step2 ≈ 2.4e-4`), final norm 0.617.
* `P_raw = h_raw + q`, `b_P = median_fit(y − P_raw)` once per arm,
  `P_cal = P_raw + b_P`.  `h` (frozen `O_seed0` raw soup) is never updated.

## gates (fixed)

* Structure gate (Phase A, fit only): identity/count invariants pass;
  `global_min_l1(X25) − global_min_l1(X27) ≥ 0.001`; at least one original
  `k<=-3` fit row from an X25 conflict class has X27 class span `≤ 1e-10`.
  Failure → `INPUT_REPAIR_NOT_SUFFICIENT`, no head is bought.
* Performance gate (Phase B, dev): overall cal gain ≥ 0.003 with paired
  bootstrap 95% CI lower > 0; overall raw gain > 0; G0 cal worsening ≤ 0.001;
  identity/no-label/single-calibration/replay contracts pass.  Bootstrap 1000
  draws seed 20261004, main CI uniform over all dev rows with same indices for
  both arms, G0 within G0 rows; severity-stratified CI is secondary only.
* Fit-improvement marker (descriptive, not a performance gate): original 5 fit
  `k<=-3` rows q-MAE drop vs Q0 ≥ 25%.
