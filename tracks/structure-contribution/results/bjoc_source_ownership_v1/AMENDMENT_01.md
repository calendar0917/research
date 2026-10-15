# AMENDMENT 01 — implementation repair + matched-control coverage semantics

Recorded **before** the first full Phase-0 execution and before any monitor or
official-valid label is read.  No Phase-0/Phase-2 result exists yet; the only
runs so far are: unit tests, CPU correctness gates, P3/valid reference replays,
and an 8-molecule Phase-0 *smoke* run whose output was discarded.

## A1.1 Aggregation bug (implementation, no protocol meaning)

`phase0` compared `IG_minus_random` through the same `top4_response` dict used
for `dev`/`rare`, but random subsets are stored as 16 repeated draws per
molecule (`random_response`), not as one top-4 set.  The smoke run aborted with
`KeyError: 'random'`.  Fix: `IG_minus_random` is now its own contrast
(bootstrap + sign test over the 16-draw molecule means).  The frozen protocol
already defined random as "uniform k-subset draw; 16 draws per molecule".

## A1.2 Matched-control coverage gate

Preregistered statement: `P0_MATCHED` required `coverage >= 0.8` (fraction of
IG top-4 junctions with a matched control) in addition to
`mean gap > 0, CI95 lower > 0`.

Measured structural fact, using only signatures/degrees of the frozen
diagnostic set and **no model scores**:

* same-molecule controls matched on `(signature, center degree, ring)` exist
  for an upper bound of **0.620** of diagnostic junctions (0.660 for
  signature-only), because 519 signature classes are spread over ~35 junctions
  per molecule and the top-4 exclusion removes additional candidates.

The 0.8 coverage gate is therefore unattainable by construction, not by model
behaviour.  Amended semantics, chosen before the real Phase-0 run:

* `P0_MATCHED = (mean gap > 0) and (CI95 lower > 0)` over the junctions that do
  have controls; the paired bootstrap and the one-sided exact sign test already
  price in the matched-junction count `n`;
* coverage, matched-junction count, and the fraction of matched junctions whose
  dev-decile was exactly matched are **reported as diagnostics** (no silent
  threshold);
* the Fallback order is unchanged: dev-decile-exact pool first, then
  `(signature, degree, ring)`, same molecule only.

Amendment recorded here, in `configs/bjoc_source_ownership_v1.yaml`, and in
`PROTOCOL.md`.  The protocol freeze is re-issued after this amendment
(`protocol_freeze.json` hash changes; the git commit of this file is the
before/after boundary).
