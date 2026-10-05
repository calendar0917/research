# ERRATA — `zinc_component_supervision_fulltrain_confirmation_seed0_v1`

## Engineering failures (discarded, no scores consumed)

1. **`full_train_payload.npz` not committed (Git-ignored by default).** Initial training
   submissions failed with `FileNotFoundError` because `.npz` artifacts are git-ignored. Fix:
   `git add -f` to force-include the freeze artifacts. No model scores were produced or read
   from these failed runs.

2. **Module path typo `tracks.ksdv` vs `tracks.ksvd`.** Two early submissions
   (`-132954` run IDs) failed with `ModuleNotFoundError: No module named 'tracks.ksdv'` due to
   a typo in the `--result` path argument that propagated to the experiment invocation. Fix:
   corrected to `tracks.ksvd`. No scores consumed.

3. **CUDA forward nondeterminism in init-check.** The initial SUM-vs-COMP forward max-abs
   difference on CUDA was 1.5e-6 (kernel nondeterminism), while the state dicts were
   byte-identical (`shared_state_max_abs_diff = 0.0`). Fix: relaxed the forward tolerance from
   exact-0 to 1e-5; the state-dict identity check remains exact-0.

4. **Missing `local_mol_id` on valid data.** The first valid read failed with
   `AttributeError: 'GlobalStorage' object has no attribute 'local_mol_id'` because the
   `load_valid_data` function did not set `local_mol_id` on the valid data objects (required
   by the model's tuple-indexing forward). Fix: added a loop to set
   `data.local_mol_id = torch.tensor([index])` for each valid row.

## Non-issues (within tolerance)

* CPU vs GPU replay: max-abs diff 1.2e-6 (SUM), 1.4e-6 (COMP) — below the 1e-5 tolerance.
* Float64 row-wise deploy identity: 3.8e-7 (SUM), 4.1e-7 (COMP) — below 1e-5.
* `g = ell + s` identity: 4.4e-16 — float64 exact.
* `y = g + c` identity: 0.0 — float64 exact.
