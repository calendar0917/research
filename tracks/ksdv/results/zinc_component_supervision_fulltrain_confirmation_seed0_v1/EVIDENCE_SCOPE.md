# EVIDENCE_SCOPE — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## What is in scope (evidence for this round)

* **Train-only prep, targets, payload:** refit on all 10,000 official-train rows. The old
  8,000-fold standardizers, old kappa, old schedule, and the mixed 12k (train+valid) constants
  are **not** reused as this round's train-only objects.
* **Both body arms (SUM, COMP):** identical init, schedule, recipe. Only the loss differs.
* **Shared Q head:** trained on full-train topology25 + c; both arms use the same Q.
* **Calibration:** one `b_g` and one `b_y` per arm, computed train-only.
* **official-valid (1000 rows):** one frozen read, after the frozen_eval_manifest. Diagnostics
  computed with frozen train constants.
* **Bootstrap / gates:** 1000 draws, 4 metrics per endpoint (G0/overall × raw/cal).

## What is explicitly out of scope

* **official-test:** never loaded, instantiated, predicted, or scored.
* **Seed replication:** single seed (0). No seed 1.
* **Model selection:** no dev-driven choice of epoch, soup, or coefficient. Soup epochs are
  fixed by recipe.
* **Hyperparameter scan:** coefficient 0.5 is fixed.
* **New architecture:** same M skeleton, same ComponentReader (39→2).
* **Dictionary revival / loss scan / cycle-rescue training.**

## Data reuse disclosure

* **official-valid** is reused from prior research rounds (it appears in prior experiment
  results). This is a confirmation round, not a novel evaluation.
* **official-train** (10,000 rows) is the training set. The old internal dev2000 is now inside
  the training set.
* **GVAE property cache** (`gvae_full_properties.npz`) is reused as the property table, read
  only by `smi_line` index.

## Evidence artifacts

All artifacts are in this directory. Key files:

| file | purpose |
|---|---|
| `frozen_eval_manifest.json` | frozen train-only objects, model hashes, bootstrap/gate definition |
| `heldout_access.json` | timestamped record of the one valid read |
| `SUM_meta.json`, `COMP_meta.json`, `Q_meta.json` | recipe, schedule, allocation, curves |
| `SUM_raw_soup_state.pt`, `COMP_raw_soup_state.pt`, `Q_raw_soup_state.pt` | trained model weights |
| `valid_frozen_predictions.npz` | all valid predictions (h, ell, s, q, g, y, cal/raw) |
| `analysis.json` | full gates, gains, component/cycle diagnostics |
| `main_table.csv`, `group_table.csv`, `component_table.csv` | tabular results |
