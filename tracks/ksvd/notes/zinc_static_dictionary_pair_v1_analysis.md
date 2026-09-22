# Analysis — ZINC strict-static dictionary-pair v1 confirmation

Round: **ZINC-strict-static-dictionary-pair-v1-confirmation**.
Preregistration: `tracks/ksvd/notes/zinc_static_dictionary_pair_v1_preregistration.md`
(frozen at commit `1cb4fc7`).
Implementation: commit `1cb4fc7eec47cbc474cd14d3fff6a880bb1c7477`
(remote A100-SXM4-40GB, GPUs 0 and 1).
Status: one paired seed-1 confirmation; round closed.

## 0. What the round asked

> Does the seed-0 dictionary-specific signal replicate on seed 1 **after**
> removing the unequal-checkpoint-opportunity / late-training confound?

The architecture contract was inherited unchanged.  No pair kernel, no sweep,
no rescue, no HPO, no seed 2/3.  Official ZINC **test was never loaded**
(`official_test_loaded = false` on every artifact).

## 1. Phase A — zero-training common-horizon audit (before any seed-1 run)

Read from the durable seed-0 curve CSVs, not hardcoded:

```text
H_common = min(epochs_run) = min(204, 210, 240) = 204
```

| arm | epochs run | best full | best_common (<=204) | best epoch | mean best-5 (<=204) |
|---|---|---|---|---|---|
| S0 | 204 | 0.145674 | 0.145674 | 164 | 0.146560 |
| S-Dense | 210 | 0.145958 | 0.145958 | 170 | 0.148202 |
| S-Dict | 240 | 0.136050 | **0.139009** | 193 | 0.140259 |

```text
M0c = 0.145674   MDc = 0.145958   MKc = 0.139009
base_gain_common = M0c - MKc = +0.006666   (gate >= 0.004)
dict_gain_common = MDc - MKc = +0.006949   (gate >= 0.002)
```

**A1 — `COMMON_HORIZON_SIGNAL_SURVIVES`.**  Even restricted to the shared 204
epochs, S-Dict beats both S0 and the matched dense control.  The seed-0
direction was therefore **not** created by S-Dict's extra 36 epochs.

Restricted Top-5 *weight soup* (epoch `<= H_common`) is **unavailable from
saved states**: the v0 runner persisted only the single best-checkpoint
`selection_state.pt` per arm; per-epoch Top-5 snapshots were an in-memory cache.
This diagnostic was **not** faked and was **not** replaced by the best-5
validation-MAE mean.  Only curve statistics are reported for Phase A.

Late-horizon dependence: S0 and S-Dense do not improve after their best epoch,
while S-Dict's full best (`0.136050 @225`) is `0.002959` below its common-horizon
best (`0.139009 @193`).  Therefore

```text
(0.009624 - 0.006666) / 0.009624 = 30.7 %
```

of the seed-0 primary `S0 - Dict` margin came from epochs 205–240; the remaining
~69 % is already present at `H_common`.

## 2. Seed-1 execution

Exactly three full-training runs, one per arm, each **forced to 240 epochs**
(no early termination), with a shadow early-stopping tracker (patience 40)
running alongside.  Because SSE/GPU, split, tokenizer, topology, head,
optimizer, LR, wd, batch, shuffle-seed convention and parameter matching are
inherited unchanged, and all 40 shared tensors are bit-identical
(`max_abs_diff == 0.0`), the only intended difference is the residual branch.

| arm | device | equal-horizon best | epoch | Top-5 soup | run wall |
|---|---|---|---|---|---|
| S0 | GPU0 | 0.139389 | 233 | 0.136423 | 984 s |
| S-Dict | GPU1 | 0.138020 | 240 | 0.134697 | 1111 s |
| S-Dense | GPU0 | 0.145830 | 164 | 0.141608 | 1014 s |

The shadow tracker never stopped S0 or S-Dict (both kept improving inside any
40-epoch window), and stopped S-Dense at epoch 204.  Consequently the
equal-horizon and shadow views are **identical** for S0 and S-Dict, and differ
only trivially for S-Dense (its best was at epoch 164, inside both windows).

## 3. Seed-1 primary — equal-horizon 240-epoch view

```text
M0_1 = S0    = 0.139389
MD_1 = Dense = 0.145830
MK_1 = Dict  = 0.138020

Gbase_1 = M0_1 - MK_1 = +0.001369     (gate >= 0.004 -> NOT met)
Gdict_1 = MD_1 - MK_1 = +0.007811     (gate >= 0.002 -> met)
```

Corroborating equal-horizon Top-5 soup:

```text
S0 0.136423 / Dense 0.141608 / Dict 0.134697
Gbase_1 soup = +0.001726     Gdict_1 soup = +0.006912
```

Both gains are positive on both the best-checkpoint and the soup metric, but the
baseline gate is not reached.  Pre-registered verdict: **B2 —
`SEED1_DIRECTIONAL_REPLICATION`** (direction replicates; the strong double gate
does not fire).  The shadow view gives the same numbers and the same case, so
`equal_horizon` and `shadow` agree in direction.

## 4. Critical mechanism finding — the seed-1 dictionary is inert

The arm-level ordering hides a decisive mechanism fact.  At the seed-1 best
checkpoint the residual **scale** `gamma` has decayed to essentially zero:

| arm | gamma init | gamma final | residual ablation mean shift | max shift |
|---|---|---|---|---|
| S0 | 0.100 | 0.100 | n/a | n/a |
| S-Dense | 0.100 | **9.5e-05** | 9.3e-07 | 9.5e-06 |
| S-Dict | 0.100 | **3.75e-04** | 1.02e-05 | 1.02e-04 |

`gamma` decays monotonically from step 0 and is ~0 from epoch ~140 onward
(S-Dict: 0.0492 @20, 0.0021 @100, ~0 @180–220, 0.0004 @240; S-Dense behaves the
same).  The inference ablation confirms it directly: forcing the dictionary
residual off changes predictions by `1.0e-05` on average versus `0.3044` in
seed 0.  `mean_residual_norm` is `3.3e-08`.

In other words, **the seed-1 `S-Dict` best checkpoint is effectively the
strict-static base network**, not a working dictionary.  The `S-Dict` run only
differs from the other arms because the branch was transiently active early in
training (gamma ~0.1 for ~60 epochs) and perturbed the base optimisation
trajectory.

Dictionary diagnostics confirm the degenerate regime:

| diagnostic | seed 0 (best) | seed 1 (best) |
|---|---|---|
| active atoms | 64 / 64 | 64 / 64 |
| argmax-used atoms | 12 | 5 |
| effective atom count | 8.13 | 58.45 |
| mean assignment entropy | 2.095 | 4.068 (near max ln 64 = 4.159) |
| top-8 assignment mass | 0.845 | 0.239 |
| max average assignment mass | 0.348 | 0.031 |
| dictionary coherence mean / max | 0.453 / 1.000 | 0.205 / 0.775 |
| tau init -> final | 0.200 -> 0.210 | 0.200 -> 0.547 |
| gamma init -> final | 0.100 -> 0.181 | 0.100 -> 0.0004 |
| residual ablation mean shift | 0.3044 | 1.02e-05 |

Seed 1 drove the assignment distribution toward near-uniform and the residual
scale to zero — a *high-temperature / zero-scale* collapse.  The atoms are not
dead in the count sense, but the residual does no work.

## 5. Two-seed synthesis

### 5.1 Original-protocol paired summary (seed-0 original + seed-1 shadow)

```text
seed0  S0 - Dict (best)   = +0.009625
seed1  S0 - Dict (best)   = +0.001369     mean = +0.005497
seed0  Dense - Dict (best)= +0.009908
seed1  Dense - Dict (best)= +0.007811     mean = +0.008859

seed0  S0 - Dict (soup)   = +0.006965
seed1  S0 - Dict (soup)   = +0.001726     mean = +0.004346
seed0  Dense - Dict (soup)= +0.008831
seed1  Dense - Dict (soup)= +0.006912     mean = +0.007871
```

Sign consistency across the two seeds holds for all four pairings.

### 5.2 Equal-opportunity evidence (reported separately, never averaged)

```text
seed 0 common horizon <= 204 : M0c 0.145674  MDc 0.145958  MKc 0.139009
seed 1 full horizon    1..240: M0_1 0.139389 MD_1 0.145830 MK_1 0.138020
```

These two horizons are different experiments and are **not** averaged into one
benchmark.  Seed 0 was **not** retrained.

### 5.3 Seed-to-seed movement of each arm

| arm | seed 0 best | seed 1 best | change |
|---|---|---|---|
| S0 | 0.145674 | 0.139389 | **-0.006285** |
| S-Dense | 0.145958 | 0.145830 | -0.000128 |
| S-Dict | 0.136050 | 0.138020 | +0.001970 |

The baseline `S0` moved by `0.0063` between seeds — about `4.6x` the seed-1
`S0 - Dict` margin (`0.00137`).  The seed-1 "dict beats S0" margin is therefore
**inside S0's own seed-to-seed variability**.  The `Dense - Dict` ordering is the
only pairing whose gap is comfortably larger than the seed movement of the
control arm, and even that is not mechanism-attributable in seed 1 because the
dictionary residual is inert.

## 6. Pre-registered verdict and honest reading

```text
case: B2_SEED1_DIRECTIONAL_REPLICATION
```

* **Arm-level**: S-Dict is better than S0 on both the equal-horizon and the
  shadow view, on both best-checkpoint and soup; and it is clearly better than
  the parameter-matched S-Dense control on both views.  The *direction* of the
  seed-0 ordering replicates.
* **Mechanism-level**: it does **not** replicate.  In seed 1 the dictionary
  residual is inert at the best checkpoint (`gamma ~ 4e-4`, ablation shift
  `1e-5`), whereas in seed 0 it was strongly load-bearing (`gamma 0.181`,
  ablation shift `0.304`).  The seed-1 `S-Dict` endpoint is the base network.
* **Strength**: the strong double gate (`Gbase_1 >= 0.004`) does not fire; the
  seed-1 `S0 - Dict` margin is below both the gate and the baseline's own
  seed-to-seed movement.

Therefore the correct reading is: *the dictionary-specific signal did **not**
robustly replicate; only a weak, subthreshold arm-level direction survived, and
that direction is not explained by a live dictionary in seed 1.*  The
dictionary mechanism is **seed-unstable** and cannot be treated as a confirmed
foundation for the occurrence-level pair kernel.

### Caveats

1. The per-epoch official-valid curve is very noisy; best-checkpoint is a max
   statistic.  Seed-1 S0's best is at epoch 233 (`valid 0.1633` at epoch 240),
   and seed-1 S-Dict's best is at the final epoch 240.
2. `gamma` decay is a genuine training-dynamics outcome, not a diagnostic
   artefact: the residual-disable ablation is computed by re-running the
   forward pass with the branch disabled, and the two matched branches decay
   independently yet similarly.
3. No statistical significance is claimed; two seeds is a paired confirmation,
   not a distribution.

## 7. Strict-static contract (seed 1)

All gates pass for every seed-1 arm on real valid batches:

```text
center_context == False / center_update is None      : true (all arms)
forward with _pool_pairs_to_centres raising          : succeeds
pair encoder calls per forward                       : 1
relation encoder calls per forward                   : 1
h_i bit-identical under pair-relation mutation       : true (max abs diff 0.0)
prediction changes under relation mutation           : true
pair-order invariance                                : <= 1.9e-6
dictionary / Wq / Wv gradients alive at step 0       : true
```

## 8. Direct answers to the round's questions

1. **Seed-0 signal under a common horizon?** Yes — A1
   `COMMON_HORIZON_SIGNAL_SURVIVES` (`base_gain_common +0.006666`,
   `dict_gain_common +0.006949`).
2. **How much of the seed-0 advantage depends on epochs 205–240?** ~30.7 % of
   the primary `S0 - Dict` margin (0.002959 of 0.009624).  S-Dict improves over
   that window; S0/S-Dense do not.
3. **Seed-1 equal-240 S0?** best `0.139389 @233`, soup `0.136423`.
4. **Seed-1 equal-240 S-Dict?** best `0.138020 @240`, soup `0.134697`.
5. **S-Dense seed-1?** best `0.145830 @164`, soup `0.141608` (bought because the
   provisional check showed a positive but sub-threshold direction).
6. **Seed-1 `S0 - Dict`?** `+0.001369` (best), `+0.001726` (soup).
7. **Seed-1 `Dense - Dict`?** `+0.007811` (best), `+0.006912` (soup).
8. **Shadow vs equal-horizon?** Identical for S0 and S-Dict (shadow never
   stopped either); S-Dense stopped at 204 with its best at 164, so the two
   views coincide there too.  Same case (B2) in both views.
9. **Does the dictionary-specific signal replicate across seeds?** Arm-level
   direction: yes, with sign consistency, but strongly attenuated for
   `S0 - Dict` and sub-gate.  Mechanism-level: **no** — the seed-1 dictionary is
   inert.
10. **Is the dictionary still load-bearing?** **No in seed 1** (`gamma 0.000375`,
    ablation mean/max shift `1.0e-05` / `1.0e-04`); yes in seed 0 (`gamma 0.181`,
    ablation `0.304` / `0.983`).  The mechanism is seed-unstable.
11. **Strict-static contract still satisfied?** Yes, all gates pass for all
    seed-1 arms.
12. **2 or 3 full runs?** 3 new full-training runs.
13. **Wall clock?** GPU wall `3109.4 s` (51.8 min) across the three runs;
    elapsed wall for the formal seed-1 phase `2181 s` (36.4 min), of which
    S0 ∥ S-Dict ran concurrently.
14. **Official test loaded?** No — never, in any stage.
15. **Enough evidence to authorize the next round's occurrence-level
    dictionary-conditioned pair kernel?** **No, not as a confirmatory round.**
    The dictionary mechanism did not replicate in seed 1 (gamma→0, inert
    residual), so the pair kernel would be built on an unstable foundation.  The
    next round should first resolve residual-dictionary liveness/stability —
    e.g. a preregistered study of the `gamma` dynamics (why the residual scale
    collapses and whether it can be kept alive under the same strict-static
    contract), with matched controls and mandatory liveness (gamma + ablation)
    diagnostics.  Only if the dictionary is demonstrably live and
    seed-stable should the occurrence-level dictionary-conditioned pair kernel
    be registered as a confirmatory experiment.

## 9. Artifacts

* `results/zinc_static_dictionary_pair_v1_confirmation/common_horizon_audit.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/provisional.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/analysis.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/parameter_audit.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/initialization_match.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/integrity_gates.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/smoke.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/runs/{s0,dense,dict}_seed1.json`
* `results/zinc_static_dictionary_pair_v1_confirmation/curves/{s0,dense,dict}_seed1_curve.csv`
* `results/zinc_static_dictionary_pair_v1_confirmation/RESULTS_SUMMARY.md`
