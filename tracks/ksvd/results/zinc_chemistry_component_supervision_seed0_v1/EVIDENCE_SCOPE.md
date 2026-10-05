# EVIDENCE_SCOPE — `zinc_chemistry_component_supervision_seed0_v1`

## In scope (this round)

* The matched-pair internal-dev comparison SUM vs COMP on the frozen fresh fold.
* The component fit/dev diagnostics of the COMP model only.
* Init-identity, label-permutation, save-reload, CPU/GPU replay, bootstrap witness checks.

## Out of scope

* Official-valid / official-test. Never loaded, instantiated, predicted or scored.
* Any trained state from the original g-only M model beyond its frozen init (used as the
  construction reference only).
* The `zinc_direct_bond_relation_seed0_v1` model; read only as a closed prior arm.
* The local-tuple / pooling / direct-bond variants closed out by this round (no further
  interface appends).

## Comparability

Same `protocol_version`, same fold fingerprints, same skeleton/init, same schedule hash
(`7b11a529584a8bdaa8e2f17a8c8413cdc6677714ae855339b5984e9fe6938e65`), same regime
(driver 525.85.12 / torch 2.5.1+cu124 / CUDA 12.4 / Python 3.12.14 / FP32). The two arms are
matched for all of these.
