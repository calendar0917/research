# DECISION — zinc_full_nonlinear_probe_v1

**Decision: stop the probe here; buy no next pilot this round.**

Evidence (`nonlinear_probe.json`, run `20261002-200428-11b0b325`):

* A `[R]` two-seed mean calibrated valid gain vs parent = **−0.00653** (both seeds negative).
* B `[R,z]` = **−0.00839**; B−A = **−0.00186**; B−C = **−0.00030** — no node-information gain.
* Heads fit train *better* than the parent (raw train MAE 0.0325/0.0272 vs 0.04540), so the
  negative result is **not** `PROBE_UNDERPOWERED`; it is a generalisation/regularisation gap.

Scope of closure: **only** these three frozen configs on the frozen Full R. It does **not**
close R sufficiency, node information, or nonlinear readouts in general.

Secondary findings:
* The valid `y<−3` "tail shrink" (slope 0.287) is almost entirely the single id172 row:
  excluding it, slope = 0.962 vs train 1.034. Absolute tail MAE stays elevated (0.219 vs 0.066).
* Dead node weights (`W_A_S/W_A_C/node_encoder.0.weight`) are identical denormals in
  `raw`, `final`, and `soup` states → **not** an averaging-cancellation artefact. 5 soup
  members unavailable (not saved).
* No same-protocol paired valid/test summary exists in context → the "valid ≈ test+0.02"
  observation is **unverified**; no fixed valid→test conversion used.

Not done (boundary): official test never instantiated; no backbone training; no node revival.