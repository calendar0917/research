# EVIDENCE_SCOPE — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

What one new-fold pair of trajectories can and cannot establish. This round replaces, for the
B-vs-M_J question, the old fold with a pre-fixed new fold; it is not a new model family and not an
independent test.

## This round can say

1. Whether the historical B→M_J chemical-component (`g = y − c`) gain keeps its sign, magnitude
   and uncertainty band **on one new pre-fixed partition of the same official-train rows**,
   under the frozen gate. The new dev rows were trained on by the old models and the old/new fits
   overlap; the result measures partition stability of the training-side decomposition, not
   generalisation to unseen chemistry.
2. Whether both arms' fit/dev, raw/cal and per-k-group behaviour is consistent with the old-fold
   direction; fit/dev movement, bias and per-group contributions are disclosed.
3. That all fit-dependent objects for the new models (target constants, body standardizers, tuple
   phi scaler, kappa sample) were derived from new-fit rows only, and that the actual training
   graph stream changed (global-ID stream hash, position schedule unchanged, `|new_fit∩old_fit| =
   6000`).
4. Whether the M_J local channel is active after training (drift, task gradients, code RMS, dead
   dims, injection RMS, zero-ablation dependence) — mechanism health only.

## This round cannot say

* It cannot call the new dev a "never-touched independent test": the old B and M_J trained on all
  2000 new-dev rows (they lie inside the old fit). New-fit overlap with the old fit is 6000/8000,
  and the canonical-group cross check shows 1 shared group (1 fit row, 1 dev row) — reported, not
  used to re-choose the fold.
* It cannot attribute any gain to the correspondence relation, to structural information, to a
  sparse dictionary or to information exclusivity. M_J adds 29,888 local parameters plus an
  optimisation path; the comparison is "B skeleton vs B skeleton + whole local interface", so
  added capacity/optimisation remain confounds.
* It cannot pool old-fold and new-fold results into a larger confirmation CI; they share rows and
  the new fold was not independent.
* It cannot establish "MLP ≥ dictionary" or "the dictionary form is unneeded". See ERRATA.
* It does not test ring/cycle relief, long-cycle tails, message passing, global structure,
  multi-seed variance, the incidence-permuted control, official-valid/test, or the 10k
  confirmation.
* A `g-MAE < 0.09` (if reached) is an internal chemical-component diagnostic, not a deployable
  `y` score; no cycle head is trained and no true `c` is added back for a deployment claim.

## Decision use

The new fold is the only decision endpoint, and there is only one paired comparison. Any result
buys no third trajectory. `PERFORMANCE_REPLICATED` keeps M_J as a working reference for the
chemical component while keeping the dictionary leg and luyin19's unrealised structural-property
claims open. `DIRECTIONAL_NOT_CONFIRMED`/`NOT_REPLICATED` freezes this local interface and stops
encoder/fold/seed thresholds investment; it does not prove the interface is permanently useless
nor that the remaining error necessarily lacks information.
