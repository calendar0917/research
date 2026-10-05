# METHOD_CONTRACT — `zinc_cycle_level_transfer_terminal_test_seed0_v1`

## 1. Frozen inputs (read-only; identities verified in phase `identity`)

| object | file | role |
|---|---|---|
| SUM body | `zinc_component_supervision_fulltrain_confirmation_seed0_v1/SUM_raw_soup_state.pt` | frozen `SUM_h_raw` (297,539 params) |
| COMP body | `.../COMP_raw_soup_state.pt` | frozen `COMP_h_raw` (297,539 params) |
| Q cycle head | `.../Q_raw_soup_state.pt` | frozen `q_raw` (3,777 params) |
| prep / payload / targets | `.../full_train_prep.npz`, `full_train_payload.npz`, `full_train_targets.npz` | frozen 10k standardisation, tuple payload, y/g/c/k/ell/s + constants |
| biases | `.../calibration.json` | `b_y_SUM = −0.01399996131658554`, `b_y_COMP = −0.011630002409219742`, `b_g_*` |
| valid cache | `.../valid_frozen_predictions.npz` | replayed `h_raw`/`q_raw`/labels for the exposed valid |
| prototype package | `zinc_cycle_prototype_transfer_cpu_v1/prototype_table.npz`, `H_model_package.pt`, `T25_cache.npz` | exact T25 key table, consistent-class rule, `b_H = −0.011446799464432368` |
| test labels | `zinc_long_cycle_audit/test_cycle_audit_label.csv` + `cache/gvae_full_properties.npz` | official test `y`; `k`/`c`/`g`/`ell`/`s` formed at scoring with frozen train constants |

Source execution commit `947d2837c4936ef1276fcbbf0e9189e7d44ace23`; prototype round commit `5a510d9`.

## 2. T25 key and class folds (train only)

* Key: exact float32 equivalence of the current Q model input (25-dim, contiguous
  little-endian, `−0→+0`, NaN/Inf forbidden) — identical to the prototype round's verified
  key. Train T25 sha256 `0ccfad54a7c429ac7c705e29aa38528fa66f08ecaeb9d22e6111a9ce810bfd64`.
* 355 classes; class→fold by support-descending, tie by canonical key bytes, greedy to the
  currently-lightest fold, tie to the lowest fold index; label-blind; 3 folds; conflict
  classes stay whole. Held ∩ fit = ∅ per fold.

## 3. Vocabulary and level constants

* `vocab = sorted(unique integer k over all 10000 train rows) = [-12,-6,-5,-4,-2,-1,0]`,
  `K = 7`, fixed for all folds and `D_full`. Class index = position in this list.
* `c_level(k) = (k − mu_cycle)/sigma_cycle` with the frozen full-train constants
  (`mu_cycle = −0.00014482659782137728`, `sigma_cycle = 0.28844036744667567`).
* Decoder (only): softmax over the 7 level logits → levels sorted by `c_level` ascending →
  first cumulative ≥ 0.5 → output that `c_level`.

## 4. Heads (exactly one R/D recipe)

* Hidden stack `25→64→32`, SiLU, built by the existing Q seed-0 untrained construction rule
  (`torch.manual_seed(0)` + same `Sequential`); R/D hidden init identical item-by-item;
  trained-Q hidden weights never reused.
* `R`: `32→1`, last weight 0, bias `median(c_fit_fold)`; loss `L1(q, c)`; params 3777.
* `D`: `32→7`, last weight 0, bias 0; loss `F.cross_entropy(logits, class_index)`; params
  3976 = 3777 + 33·6.
* Training: CPU FP32, seed 0, batch 128, 300 epochs, `torch.optim.Adam(lr=1e-3,
  weight_decay=1e-5)` (coupled), grad clip 5, soup = mean state of epochs 296–300. Fold `j`
  generator `20261003 + j`; R/D same fold share the fit order and batch stream;
  construction RNG isolated. `D_full` (purchased only): generator `20261003`, 10000 rows,
  23700 steps.
* Per fold: OOF `c`-MAE, k accuracy + confusion, per-k group n/errors, missing fit levels.
  All six saved before any gate score.

## 5. Purchase gate (pre-fixed)

Pooled OOF over all 10,000 rows, `gain_c = MAE(R_OOF) − MAE(D_OOF)`:
(1) overall ≥ 0.003; (2) k=−2 ≥ 0.25; (3) k=0 worsening ≤ 0.001; (4) k=−1 worsening ≤
0.05. Non-estimable group → NO_BUY. Descriptive CI: 1000 class-level resamples, seed
20261011, shared across R/D, rows-of-class kept whole.

## 6. Candidate C (purchased only)

```text
q_C = class_median_c        if key in table and consistent      (== H)
q_C = frozen_Q(T25)         elif key in table (conflict)        (== H)
q_C = decode(D_full(T25))   else                                 (new head)
y_C_cal = h_COMP_raw + q_C + b_H
```

Train predictions equal H's by construction; `b_H` is the only deploy bias. Seen-route
Δpred vs H = 0 within 1e−5.

## 7. Terminal roster (frozen before any this-round test prediction/metric)

SUM_Q / B / H (+ C only if purchased), per §8 of PROTOCOL. Single terminal execution:
generate all valid (replay) and test predictions first, then statistics. Paired bootstrap
1000, seed 20261012, shared indices; witnesses identical→0, swap→mirror, shift→inside.
Valid−test gaps per system. No post-test selection of any kind.

## 8. Metrics

MAE (y units) raw/cal; `g` raw/cal alongside where labels exist; gain = reference −
candidate (positive = improvement); routing/group contributions add back exactly;
`triangle_gap = mean(|e_g| + |e_c| − |e_g + e_c|) ≥ 0` (this round's sign convention; the
prototype round's negative values used the opposite ordering — clarified in ERRATA);
float64 identities 1e−9; prediction tolerance 1e−5; state/hash exact.
