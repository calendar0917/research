# EXECUTION — zinc-frozen-chemistry-learned-cycle-v1

Wall clock: first tool call `2026-10-03 07:55:43 UTC`; all compute stopped
`2026-10-03 08:13:45 UTC` (~18 min active, budget 60 min).  The remaining time
was used only for analysis, replay and delivery.

## Resource regime

* Device: local CPU only.  `torch.set_num_threads(8)`; process `nproc = 16`,
  capped at 8 threads.  No GPU job, no new Full training, no remote job, no
  `rr deploy/run`.
* Two cycle heads trained sequentially (not in parallel); each 300 epochs =
  18,900 steps, ~10.0 s.  Full round compute `~27 s` (data load + prep + identity
  + smoke + 2 heads + wrapper/replay).
* `budget.json`: `gpu_jobs = 0`, `new_full_training = 0`,
  `official_valid_loaded = false`, `official_test_loaded = false`.

## Steps

| # | step | artifact | result |
|---|---|---|---|
| 0 | branch `task/zinc-frozen-chemistry-learned-cycle-v1` from `87894a2` | — | isolated |
| 1 | freeze protocol + runner, commit before fitting | `e391550` | frozen |
| 2 | load `encoded_train.pt` + `env_train.pt` (train-only), `apply_prep_train_only`, build `T25` | `T25_all.npz` | `(10000,25)`, `sha256 dc2e1516…` |
| 3 | identity: raw-soup forward vs cached `h_raw`; oracle replay; split/prep hashes | `identity_checks.json` | `all_ok = true` |
| 4 | smoke 4 steps (head updates, frozen O unchanged), discarded | `smoke_checks.json` | `ok = true` |
| 5 | formal head seed 0, then seed 1, frozen recipe, soup 296–300 | `P_seed{0,1}_*.pt/json`, `P_seed{0,1}_predictions.npz`, `deploy_bundle_seed{0,1}.pt` | done |
| 6 | wrapper / label-invariance / soup replay | `wrapper_checks.json` | pass |
| 7 | fixed analysis + bootstrap + gate | `analysis_summary.json`, `gate.json`, `bootstrap.json`, tables, `severe_rows.csv` | gate met |
| 8 | manifest + hashes | `input_manifest.json`, `manifest.json` | 29 files, frozen artifacts unchanged |

## Identity / provenance evidence

* `fit_idx`/`dev_idx` SHA-256 equal `165e87ef…` / `fb8b7806…`; dev strata
  `1926 / 65 / 9`; prep blob `968e82dd…`.
* `T25` is the actual frozen Full model input from the frozen runner's train-only
  path (no new graph descriptor; no SMILES/atom/bond/label/group/molecule id).
* `h_raw` forward vs cache `9.5e-7 / 1.4e-6` (`<= 2e-6`); frozen O state hash
  `e759f906…` / `cbdb6bd2…` unchanged before/after and equal to the old manifest;
  `h_raw + b_O + c` reproduces the released dev MAE exactly
  (`0.1016261613 / 0.0989988937`).
* All frozen `O_seed*` / `Y_seed*` files are byte-identical to the old manifest
  (`frozen_artifacts_unchanged = true`).

## Deploy / no-leak evidence

* `deploy_wrapper.py` `forward(batch, topo25)` = `frozen_h(x) + q(T) + b_P`;
  wrapper vs cached `8.5e-7 / 1.4e-6`; repeat bit-identical; label permutation
  delta exactly `0`.
* Head soup reloaded and replayed over all 10000 `T25` rows: `max_abs = 0`.
* One fit-only `b_P` per seed; `q` never separately calibrated; no dev offset /
  coefficient / threshold / gating; no oracle routing.

## Access status

* Official ZINC **test**: never instantiated, never loaded, never evaluated.
* Official ZINC **valid**: never loaded, never re-read, never re-evaluated.
* All data come from the existing 8000/2000 split inside official train.

## Git

* Isolated branch `task/zinc-frozen-chemistry-learned-cycle-v1`.
* Frozen protocol + runner committed at `e391550` **before** formal fitting.
* Results/analysis/reports committed on the task branch.
* Not pushed, not merged; no history rewrite; old result dirs untouched
  (only the new `results/zinc_frozen_chemistry_learned_cycle_v1/` added).

## Missing / not done (by design)

* No third seed; no head-width / `lambda` / WD / loss / optimizer search; no
  Full fine-tune; no auxiliary loss; no new graph descriptor; no joint tuning.
* No official-valid / test; no 10000-row confirmation; no remote training job.
* The rare extreme cycle tail (`k<=-3`) is not fit/transferred; the fit-severe
  descriptive reference `<= 1.0` is not met.  This is reported as a boundary, not
  resolved, and not repaired with additional local heads.