# METHOD_CONTRACT — `zinc_cycle_prototype_transfer_cpu_v1`

## 1. Frozen inputs (read-only, identities verified)

| object | file | role |
|---|---|---|
| full-train targets | `zinc_component_supervision_fulltrain_confirmation_seed0_v1/full_train_targets.npz` | `y/g/c/k/ell/s`, constants, `mu_logP` |
| full-train prep | `.../full_train_prep.npz` | 10000-row standardizers (no refit) |
| COMP body | `.../COMP_raw_soup_state.pt` | frozen `h_raw` |
| Q cycle head | `.../Q_raw_soup_state.pt` | frozen `q_raw` |
| valid cache | `.../valid_frozen_predictions.npz` | reused `h_raw`, `q_raw`, `y/g/c/k` |
| biases | `.../calibration.json`, `.../frozen_eval_manifest.json` | `b_y_COMP`, `b_g_COMP` |

Source execution commit `947d2837c4936ef1276fcbbf0e9189e7d44ace23`.

## 2. Key

* `dim = 25`, `dtype = float32`.
* Encoding: contiguous little-endian float32 of the actual `Q` model input (`topology_features`
  after the frozen 10000-row full-train prep and `topo_fit.transform(topo_all.inverse(.))`).
* Normalisation: `-0` → `+0`; NaN/Inf forbidden.
* Equality: exact float32 equivalence (`np.any(a != b)`), no tolerance, no grid, no neighbour.
* The full key vector is stored (`class_key_vec`, `T25_cache.npz`), not only a hash; a collision
  check asserts `#unique byte keys == #classes`.
* No `N`/`E`, no atom/bond label, no SMILES/graph-ID, no `k/c/y`, no cycle-snap value, no valid
  label enters the key.

## 3. Train-only prototype table

* Built solely from 10000 official-train rows' T25, `c_train`, `k_train`.
* For every exact class `K`: `n_K`, member stable IDs, `unique(k)`, `k_min/k_max`,
  `median(c)`, `min(c)`, `max(c)`, `span(c)`, T25 vector, key.
* `k` is the frozen source integer label, verified finite and integer-valued.
* A class is *consistent* iff `unique(k) == 1`. Prototype value `v_K = median(c_train[K])`.
* Singletons count as consistent classes and are used; no support threshold, no shrinkage.
* `c = (k − mu_cycle)/sigma_cycle` is verified to full float precision; the consistent-class
  `c` span must lie at that precision, otherwise the round stops.

## 4. Inference rule

```text
if input_key in train_table and train_table[input_key].k_is_unanimous:
    q_H = train_table[input_key].median_c
else:
    q_H = frozen_Q(T25)
```

Routing labels: `CONSISTENT_HIT`, `TRAIN_CONFLICT_FALLBACK`, `UNSEEN_FALLBACK`. Routing depends
only on the input key and the train table. True valid `k`/`c` are read only after freeze, for
scoring.

## 5. Calibration

* Body `h` and Q weights unchanged.
* `y_B_raw = h_raw + q_B_raw`, `b_B = b_y_COMP` (source frozen), `y_B_cal = y_B_raw + b_B`.
* `y_H_raw = h_raw + q_H_raw`, `b_H = median(y_train − h_raw_train − q_H_raw_train)`,
  `y_H_cal = y_H_raw + b_H`.
* Diagnostic only: `y_H_fixedB = y_H_raw + b_B`. Not a candidate, not selected between biases.
* No `b_g` stack, no Q offset, no valid median, no raw/cal ranking.

## 6. Metrics and gate

* `gain = MAE(B) − MAE(H)`, positive = H improves.
* `PROTOTYPE_DEPLOY_SUPPORT`: overall cal `y` gain ≥ 0.003; cal `y` paired 95% CI lower > 0;
  overall raw `y` gain > 0; G0 cal `y` worsening ≤ 0.001.
* Paired bootstrap 1000 draws, seed `20261010`, shared indices. Witnesses checked.
* Markers: `SINGLE_ROW_DOMINATED` (largest positive-gain row ≥ 50% of positive sum, or gain
  vanishes after removing B's worst row); `TARGETED_REPAIR_ONLY` (points 1/3/4 hold, CI 2 fails).
* `y_cal < 0.09` is a separate marker only.

## 7. Computation

* CPU FP32, ≤ 8 threads. No optimizer, no backward, no new training, no GPU, no remote job.
* Training-side new fitting limited to exact class grouping, per-class `c` median, train-only
  scalar bias.
* official-valid evaluated once, unconditionally, after the frozen package was written.
  official-test never instantiated.
