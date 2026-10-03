# DECISION — zinc-full-decomposition-valid-test-confirmation-v1 (2026-10-03)

Pre-registered interpretation branch table (protocol §8, applied verbatim; the
test split was evaluated **only after** the frozen manifest and is reported
unconditionally):

| observation | result | branch |
|---|---|---|
| valid gate met? | **FAIL** — both valid gains negative; mean `−0.00237 < 0.003` | gate failed |
| test signal? | both seeds positive; `G0` preserved | — |
| valid gate not met but test improves | matches the "valid gate not met, test improvement" row in the protocol | report both |

**Decision: INVALID signal for promoting `P` as a baseline.** The fixed method
did **not** reproduce the +0.0109 inner-dev signal on the official-valid split,
so the decomposition route is **not** validated for this scale/interface.  The
official-test numbers are a real observation (positive, both seeds) but they
come from a single frozen configuration and cannot, by this round's policy, be
used to change the gate or author further tuning.

## What this does and does not license

* **Licences:** an honest report of the full-10k / official-valid / official-test
  results for the fixed decomposition recipe, including the valid-gate failure.
* **Does not licence:** promoting `P` to a new baseline, claiming the 0.09 target
  is reached on valid, claiming dictionary specificity, a third seed, any recipe
  change, or any further test use of the official-test split.

## Next-step evidence still required

1. Whether the rare extreme cycle tail (`k≤−3`, `c` down to `−41.6`) is learnable
   from the topology25 input at all — the fixed `Q` is near-zero there
   (`valid:0172`, `q≈−0.17` vs `c≈−20.8`) while the chemistry branch `H` is
   correct, so the tail is the single dominant error source.
2. Stability of the valid→test disagreement (negative on valid, positive on
   test): a larger held-out agreement set, or an independent confirmation under an
   explicit protocol, before any "method works" claim.
3. Whether the bulk (`k=0`) gain that appears on test/generalisation can be
   decoupled from the tail collapse — i.e. the signal is real on the head but
   fragile on the tail, which is not resolved here.

## Recommended next action (not executed this round)

Do **not** search from the current checkpoint.  Run a dedicated, explicitly
protocol-registered round that trains `Q` (under the same frozen `H`) to directly
attack the `k≤−3` rows — e.g. severity-stratified sampling of the cycle target
— and register in advance that reaching a target on official-test alone is
insufficient (the valid gate still applies).  If the tail remains unlearnable
from topology25 even with targeted training, that is the discriminator that kills
the topology25-only cycle-readout sub-hypothesis.

## Route status

The decomposition route is **not closed**: the chemistry branch `h(x)`
generalises, but the independent learned cycle head does **not** recover its
valid portion of the oracle gain at full scale — the inner-dev `+0.0109`
did not reproduce on official-valid.  The chemistry-branch sub-hypothesis
(`g = y − c` learnable from this Full representation) remains open; the
cycle-readout-from-topology25 sub-hypothesis is the one put on hold by this
round.