# EXECUTION — `zinc_cycle_prototype_transfer_cpu_v1`

## Environment

* Local CPU only, FP32, `torch.set_num_threads(8)`; no GPU, no remote job, no `rr` call.
* Python 3.12 `.venv`, `numpy 2.1.3`, `torch 2.5.1+cu124` (CPU kernels used).
* No optimizer, no backward, no training step.

## Phase order (all completed)

1. `source_identity.py` — read the frozen source round read-only and verified:
   * `y = g + c`, `g = ell + s` identities;
   * published train/valid COMP `y_cal`/`g_cal` MAE reproduced to 1e−9;
   * frozen `Q_raw_soup_state.pt` replayed on freshly built current T25 reproduces the
     cached `q_raw` exactly on both train (10,000) and valid (1,000) rows;
   * SHA-256 match against `frozen_eval_manifest.json` / `manifest.json`.
   Result: `source_identity.json`, `all_source_identity_ok = true`.

2. `freeze_prototype.py` — train-only:
   * built exact T25 float32 equivalence classes (355) of the 10,000 train rows;
   * verified the key is collision-free; `-0` absent, no NaN/Inf;
   * verified `c = (k − mu_cycle)/sigma_cycle` at full precision and consistent-class `c`
     span 0.0;
   * built the consistent/conflict member tables and the train-only prototype;
   * computed `b_H = median(y_train − h_raw_train − q_H_raw_train)`;
   * saved `H_model_package.pt`, `prototype_table.npz`, `T25_cache.npz`,
     `frozen_eval_manifest.json`, `budget.json`, and the train tables.
   No valid prediction was produced in this phase (`official_valid_loaded_in_freeze: false`).

3. `evaluate_valid.py` — single unconditional official-valid evaluation after freeze:
   * reloaded `prototype_table.npz` independently of the freeze process;
   * re-ran the frozen COMP body and frozen Q on the reused valid cache;
   * produced B/H/route/per-row/group/routing/coverage/tail/sensitivity/cancellation tables,
     paired bootstrap, gate and markers; wrote `heldout_access.json`.

4. `deploy_wrapper.py` — standalone single-file reloadable wrapper; reproduces the frozen valid
   `y_cal` MAE.

5. `replay_checks.py` — wrapper-vs-cache replay on valid and train, query-label independence,
   key determinism/reload, order restoration, shape guard. `REPLAY_OK: true`.

6. `analyze.py` — consolidated `analysis.json`.

7. `make_manifest.py` — `manifest.json`.

## Runtime

| phase | wall clock |
|---|---|
| source identity | 8.7 s |
| freeze | 8.1 s |
| valid evaluation | 7.7 s |
| wrapper / replay | ≈ 20 s |
| analysis / manifest | < 1 s |

Total agent wall clock well inside the 45 min target; no new compute step after minute 45.

## Deviations

* None. No gate, threshold, key, rule or bias was changed after seeing valid.
* `route_and_value` is a pure key lookup; NaN/Inf rows are handled by routing to
  `UNSEEN_FALLBACK` (they are not "matched") rather than raising — this is a documented
  behaviour of the frozen wrapper, not a tolerance change.

## Missing items

* None of the required deliverables is missing.
* The old discrete `topo_raw` key was deliberately not used (not needed: the exact model-input
  key is directly available and consistent).
