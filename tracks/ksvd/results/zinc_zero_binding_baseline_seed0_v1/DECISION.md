# DECISION — zinc-zero-binding-baseline-seed0-v1

Single seed (0), one new 240-epoch trajectory (`N0`), train-inner dev only.

## 1. Design decision

**Keep the compressed `S_M` deployment model as the working reference, and
treat the explicit structural binding as a module whose increment still has to
be demonstrated.  Do not buy a second seed, do not scan amplitudes / weight
decay / capacity, do not build another rescue trajectory.**

Reasons:

1. Phase A proved the reference is real: the trained `S_M` prediction is
   exactly independent of the structural binding path over all 8000 fit +
   2000 dev rows, and an algebraically equivalent model with the task
   dictionary, Sem108, size2, topology and static relations retained
   reproduces it (`max|diff| = 1.9e-6`).  This becomes the clean "no
   structural binding in inference" reference for future interface work.
2. N0 (binding fixed to zero from the first optimizer step) is
   **inconclusive at the pre-registered threshold**: G0 cal difference
   `−0.002613` with CI `[−0.006588, +0.001113]`, while the overall cal endpoint
   worsens by `+0.004569` (raw consistent).  This is neither a decisive
   training-role signal nor an equivalence result.
3. The uncertainty is about *training-time* effect, not about the inference
   result.  The result to carry forward is the Phase A simplification; the
   unresolved item is whether removing the early binding signal changes
   learning, which cannot be resolved by adding searches in this round.
4. `S_M`'s original relative G0 candidate status vs `S_J` is untouched and is
   reproduced by this round's pipeline (`+0.004981`,
   `[+0.000740, +0.009020]`).  Its mechanism remains "trained binding-off".

## 2. Mapping to the pre-registered interpretation table

| observation (primary G0 cal) | row | reading |
|---|---|---|
| point `−0.002613`, CI `[−0.006588, +0.001113]`, overall cal worsening `+0.004569`, raw same direction | "其他 / 不确定" | No decisive early-path harm (needs point ≤ −0.003 **and** CI upper < 0), no equivalence (CI not inside ±0.003, overall worsens > 0.001).  Keep the numbers and the boundary; do not read a CI crossing zero as equivalence. |

## 3. What this round closes

* **Closed (inference):** "the final `S_M` prediction still needs the
  structure-binding path." It does not: slots are exactly zero on all
  10,000 rows, coord/D substitutions change the prediction by `0.0`, and the
  reduced model is equivalent within `1.9e-6`.
* **Closed (artifact):** "the simplification cannot be deployed
  independently." It can: two checkpoints, one wrapper class, 267,611
  parameters, explicit input schema, deterministic replay script.
* **Advanced (training):** `N0` is a usable single-seed observation (exact
  init/stream match, zero binding grads, alive reconstruction), but the
  training-role question is **not closed** at the pre-registered effect size.

## 4. What remains unknown (not to be bought here)

* the sign and size of the early-path training effect at the population level
  (single seed, reused 2000-row dev; row CIs do not include training-seed
  uncertainty);
* whether the `k=−1` worsening of N0 (65 rows) is real or noise;
* whether an *alive but better-controlled* binding interface has a
  discriminative gain over the compressed Sem108 + task-dictionary +
  static-relation reference.  Any future interface must demonstrate that
  increment against this reference; parameters/search menus are out of scope.

## 5. Final answers

1. **Is the S_M final prediction really free of the structural binding path,
   and is the compressed deployment valid?**
   Yes.  Over the full 8000 fit + 2000 dev rows the node and edge slots are
   exactly zero (absmax/RMS/rowvar 0.0) and the encoder outputs are row
   constants; zeroing/replacing the structural code or perturbing `D` leaves
   the prediction bit-identical on the check batch.  The exported model
   (fusion 446 -> 110-D `[Sem108; size2]`, task dictionary and all static
   structure retained) matches the full model with `max|diff| = 1.907e-6`,
   passes label/order/independent-reload checks, has 267,611 deployment
   parameters, and its replay script passes for both S_M and N0.
2. **Does fixing both slots to zero from the first step keep or lose S_M's
   G0 performance?**
   It loses a small, CI-inconclusive amount on the primary endpoint:
   `gain_N0_vs_SM = −0.002613` G0 cal `[−0.006588, +0.001113]`, raw
   `−0.002765`; overall cal worsens by `+0.004569` with CI
   `[−0.009696, −0.000316]` (raw similar).  Paired validity holds
   (init/stream match, shared bootstrap indices, witnesses pass).  Verdict:
   **inconclusive** — neither a decisive early-path benefit/harm nor
   equivalence.
3. **What reference did we establish, and what can we not claim?**
   Established: the compressed Sem108 + task-dictionary + static-relation
   model is an exact inference-equivalent of trained S_M and a valid
   comparison reference; the matched S_M/S_J pipeline is reproduced.
   Not claimable: that the structural dictionary or dictionaries in general
   are useless (task dictionary, Sem108 shell statistics, topology and static
   relations all remain); that N0 proves an early-training binding role (the
   early path co-determines `D`'s trajectory, reconstruction/clip coupling
   and constant-bias learning); that any single-seed/dev result is robust.
4. **Next design decision?**
   Keep the compressed reference; the next dictionary interface must
   demonstrate a **discriminative gain over exactly this reference**
   (Sem108 + task dictionary + static relations, no slot binding) and define
   its information responsibility up front.  No parameter search, no
   amplitude/weight-decay scan, no seed-2 purchase in this round; the only
   item left unresolved is the training-time role documented above.
