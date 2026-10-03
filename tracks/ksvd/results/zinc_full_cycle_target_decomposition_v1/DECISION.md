# DECISION — zinc-full-cycle-target-decomposition-v1

## Result

Pre-registered route signal **met**:

| condition | seed 0 | seed 1 | mean | met |
|---|---:|---:|---:|:--:|
| cal total gain > 0 | +0.024053 | +0.021341 | +0.022697 | ✅ |
| mean cal gain ≥ 0.003 | | | +0.022697 | ✅ |
| `G0` MAE worsen ≤ 0.001 | −0.001863 | −0.005335 | −0.003599 | ✅ |

Frozen table, top-down: **branch B1 — "total signal met, mean `G0` gain ≥ 0.002
and both seeds improve `G0`"** (mean `G0` MAE gain `+0.003599`; both seeds
positive).  Within the same table, branch B2 ("`G0` preserved, gain mainly from
severe") is *also factually present* in the contribution split — the severe
group is `78.7 %` of the equal-weight mean gain — but the frozen table is
applied top-down, so **B1 is the selected branch** and the severe share is
reported as a contribution, not as a probability.

## Decision

**Continue the decomposition-supervision line, but only as a bounded,
existence-focused test — do not treat `O` as a deployable gain.**

What the evidence supports:

* Removing the rare cycle target from the shared Full's training task changes
  the *bulk* too: `G0` (`k=0`, 1926 rows, where `c ≈ +0.00046` is a tiny
  constant) improves by `+0.0036` mean, with the mean bootstrap interval
  `[0.00074, 0.00658]`; the raw/cal directions agree.  This component is
  **not** oracle-assisted.
* The large gain is the severe tail: `O` shrinks the 9 severe rows from
  `MAE 4.98 / 3.57` to `0.27 / 0.33`.  That gain is **oracle** — it requires
  adding back the true, label-derived `c`, which is node-order dependent and
  not readable from `x`.  The same holds, in part, for the `k=-1` group.
* The remaining chemical target is **not solved**: `O`'s fit/dev `g`-MAE gap is
  still `~0.069` (dev `~0.100`, fit `~0.031`).  De-cycling reduces but does not
  remove the generalisation gap.

What this does **not** support:

* It is not a claim that the cycle tail causes all of the Full's trouble, nor a
  unique attribution to "tail interference": the fixed `λ` means removing the
  cycle target also changes the task/reconstruction gradient balance.
* `O` is not SOTA, not deployable, and not a realisable upper bound.
* No conclusion that the shared dictionary is information-limited.

## Single next action (design only — not run this round)

**Mechanism.**  Keep exactly the canonical Full, the same split, the same input
interface `x` (structural dictionary, Sem108, static pair composer,
`topology25 → 8`; no new inputs, no message passing, no transformer).  Add one
**training-side auxiliary branch** supervised on the cycle component `c` (the
label-derived oracle, used only as an auxiliary label), while the main head is
supervised on `y`.  At inference the auxiliary branch is dropped (or its output
is a diagnostic head); **true `c` must never enter inference**.  This is the
frozen B1 design: "supervise the chemical / cycle branches separately under the
same Full information interface; `c` only as a training-side auxiliary label,
never at inference."

**Discriminating purpose.**  Decide whether the large severe-tail error is
*learnable from the existing inputs* (then a real de-cycling route exists and
the bulk `G0` gain might be kept while the severe penalty is modelled), or
whether the cycle term is only an oracle offset (then the severe gain is
permanently non-realisable and the route should stop at the small bulk effect).
The outcome is read as: (a) the auxiliary branch's held-out cycle-component MAE
and its effect on the main-head `G0` / non-severe MAE; (b) whether the main-head
`G0` gain measured here survives when the true `c` is replaced by a predicted
one.

## Stop / not authorised

No `λ`, WD, head-width, support-set, coverage, node-amplitude or coord
perturbation search; no third seed; no 10000-row confirmation; no official-valid
or official-test evaluation; no D-wide, dense baseline or fusion replacement.
Both seeds are already inside the four delivered trajectories, so there is no
seed-0 purchase gate.

## Revisit if

* the auxiliary cycle branch cannot beat a constant on held-out molecules, or
* the predicted-`c` version destroys the `G0` gain, or
* a later equal-information structural–semantic fusion control shows the bulk
  effect was a fusion artefact.

Any of these closes the decomposition-supervision route at "small bulk effect
only".