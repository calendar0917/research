# DECISION — `zinc_local_tuple_dictionary_vs_mlp_seed0_v1`

## Final class

**Primary encoder comparison: `INCONCLUSIVE`** (directional M_J advantage, not separable).
**Performance vs B: `PERFORMANCE_SIGNAL` not passed** (4/5 frozen conditions; G0 calibrated CI
includes zero).
**Lineage/access: valid** (`official_valid_loaded=false`, `official_test_loaded=false`; fit-only
targets and calibration; the fixed formal trajectory completed 15,120/15,120 steps at the
deployed commit `9af8e92713b9`).

Round complete: exactly one new formal trajectory (M_J, seed 0, 240 epochs), D_J and B reused
read-only, one fold, one soup, no seed/config/encoder search, no second arm, no rescue run, no
official-valid/test, no 10k confirmation.

## What was decided

1. **The dictionary form is not required by this interface at this budget.** A same-parameter,
   same-input, same-injection per-tuple `Silu(Linear)` projection is directionally better than the
   IHT dictionary on all four dev endpoints (G0 cal `+0.002200`, overall cal `+0.002526`, G0 raw
   `+0.002245`, overall raw `+0.002684`) and worse on fit (`0.030235` vs `0.028155`). Every paired
   CI crosses zero and the frozen `MLP_LOCAL_SUPPORT` gate is not met; `LOCAL_EQUIVALENCE` is also
   not established (CIs wider than ±0.003). **Keep the interface, freeze this encoder-family
   question; do not declare either encoder the winner.**
2. **No performance candidate is bought.** M_J achieves dev cal G0 `0.098216` / overall
   `0.099381` and raw G0 `0.100434` / overall `0.101414` — the best numbers so far on this
   chemical-component dev, improving on B and D_J directionally (B raw gains significant), but not
   below `0.09` and not through the frozen calibrated gate. No official-valid/test read.
3. **Mechanism is healthy, not failed.** Both encoders are active; zeroing or mean-replacing the
   local code destroys performance (dependence), while the real-vs-marginal operator switch moves
   M_J ~13× less than D_J (`0.000260` vs `0.003382` mean|Δpred|). The four-grid `G = O + W`
   decomposition (W ≈ 96% of the finite difference) is reported as descriptive only.
4. **This closes the specific configuration**: fixed capacity, fixed input, fixed injection
   location, seed 0, this fold, this dev. It does not close local dictionaries in general, the
   J/I correspondence question, ring chemistry, or message-passing variants.

## What is preserved, what is next

* Preserved: the local tuple interface and its verified production path, the D_J/B references and
  hashes, the fit-only target/scaler lineage.
* The remaining uncertainty is one-seed optimisation/dropout noise vs a genuine G0-specific
  advantage of the MLP code, plus the (still unconfirmed) J>I correspondence effect.
* Next spend (design only, not executed): the pre-registered **incidence-permuted marginal
  control** and/or **≥3-seed paired replication on an independent fold**, with the same
  calibration, bootstrap and `0.003` gate plus equivalence band. Not another encoder-family sweep
  on this repeatedly used dev.

## Boundary

This is an exploratory single-seed result on a repeatedly used 2000-row dev fold of the
official-train split; magnitudes are small (≈0.002–0.008) and the calibrated threshold is not
passed. `INCONCLUSIVE` here means "not separated", not "no effect" and not "equivalent".
