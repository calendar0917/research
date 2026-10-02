# DECISION — zinc_full_bottleneck_audit_v1

`BOTTLENECK_CATEGORY = GENERALISATION_TAIL_COVERAGE` (best supported),
with `NODE_BINDING_DEAD` confirmed and `PROBE_NEGATIVE_AT_TWO_LOCATIONS`.

* Node-side structure--semantics binding is identically zero in the current
  Full soup (`W_A_S/W_A_C/node_encoder.0.weight` denormals; `node_out` constant).
* Task latent dictionary code is 87% dense, not sparse; its moments were already
  closed by CODE and were not re-tested.
* `[1,H2]` + pre-fusion-interface moments and `[1,H2]` + node-joint moments both
  fail the 0.003 calibrated-valid purchase gate, with near-zero same-width
  controls.  Do NOT buy a fusion-widening, node-binding-repair, or task-code
  module run on this evidence.
* Error is dominated by the extreme-negative tail and id172; train tail fits
  well, valid tail regresses to the mean -> generalisation/coverage, not a
  demonstrated representation ceiling.

PRIMARY NEXT EVIDENCE (no training): frozen-R tail recoverability on a
train-internal fold, to separate "readable but the current reader cannot read
it" from "no generalisable tail signal in R".

Boundaries: frozen-representation head split (backbone saw all train labels);
exploratory official valid; the `[1,H2]` convex head is underpowered for
nonlinear information (H2 effective rank ~2, 78% dead units).
