# FEC-S0 — DECISION

```
FEC_S0_LOCAL_FACTORIZATION_BLOCKED
```

Stop.  Strict-static S0 is **not** already a pure
environment → read-only-composition model: its handcrafted channels factor
exactly into role × primitive bindings, but its dominant learned local channel
(`typed_token`/`parent_token`) is an aliased per-configuration token memory.

## Evidence summary

* `patch_cont` (146), `pair_relation` (23), `global_context` (62) reconstruct
  from raw primitives **bit-identically** (raw and standardized), with explicit
  per-block provenance.
* `typed_token`/`parent_token` certificates reconstruct exactly from the
  explicit rooted typed incidence binding, and vocabulary lookup reproduces
  every cached id — but the certificate is a **non-injective alias** and the
  table owns independent parameters per key.
* The full forward is bit-identical (`h`, `q`, `R`, prediction all `max_abs =
  0.0`) and reproduces the recorded best-checkpoint valid MAE to `1.34e-8`.
* Static purity holds: no MP / recurrence / pair→centre / context writeback.
* Soup `0.140794` not reproduced: member states were never persisted.

## What is closed

* The claim "historical strict-static S0 is already an explicit factored
  environment-composition model" is **blocked at the local token** and must not
  be asserted without qualification.
* No rescue: the blocking path is not deleted, zeroed, or replaced; no weight
  is retrained; no dictionary is introduced.

## What is established (durable)

* The mixed local/relation/global *handcrafted tensors* factor exactly; the
  coarse chemistry was not the obstruction (consistent with PEC-I1).
* A precise, quantitative reason the PEC line removed the typed token:
  55 % of S0's parameters sit in an aliased per-key local table whose alias
  buckets mix 66.3 % of patch occurrences across distinct root chemistry.

## Not this round (proposal only)

`FEC-D1 — Baseline-Preserving Sparse Structural Role Refinement`:
`r_new = r_coarse + g·r_dict`, with `g=0 ⇒ FEC-D1 ≡ FEC-S0`.  **Not
implemented, not trained.**  Dictionary work stays closed until a
baseline-preserving coordinate is defined on the verified
`role × primitive` channels.

## Restrictions honoured

Zero training / HPO / dictionary / new performance run; official test never
loaded; only the historical checkpoint and read-only historical artifacts
consumed.
