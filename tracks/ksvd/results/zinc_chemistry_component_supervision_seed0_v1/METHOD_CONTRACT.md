# METHOD_CONTRACT — `zinc_chemistry_component_supervision_seed0_v1`

Frozen before the two formal trajectories. Implementation map and the exact checks that must hold.

## Files

* runner: `tracks/ksvd/experiments/luyin16/zinc_chemistry_component_supervision_seed0_v1.py`
* sources: `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/` (read-only)
* outputs: `tracks/ksvd/results/zinc_chemistry_component_supervision_seed0_v1/`

## Reader replacement (single architectural change)

`mlpmod.build_arm_mj(payload, kappa_M)` builds the unchanged fresh-fold M skeleton. Its
`reader.net[-1]` linear `39 → 1` is replaced by a `ComponentReader` whose last linear is `39 → 2`
with `weight[i] = original_weight/2`, `bias[i] = original_bias/2`. Construction of the new layer is
wrapped in `torch.random.fork_rng(devices=[])`, so it consumes no shared RNG. The two linear
outputs sum to one linear output, so the total-`g` function class is unchanged; only the
supervision / optimisation geometry and the identifiable components change.

* `ComponentReader.forward` returns the `(n,)` total; `ComponentReader.components()` returns the
  `(n, 2)` split produced by the same forward.
* `evaluate_state_components` runs one body pass per batch and asserts `sum.shape == (batch,)` and
  `components.shape == (batch, 2)`. A `(n,2)` tensor is never compared to a `(n,)` label and never
  `.view(-1)`-flattened.

## Parameter audit

`total = 297,539 = 297,499 + 39 + 1`; base body `184,667 + 40 = 184,707`; bridge `82,944`;
local tuple `29,888`; new reader output `2×39 + 2 = 80`. SUM and COMP have identical state key sets
and identical initial tensors (max abs diff 0).

## Loss

* SUM: `L = L_g`.
* COMP: `L = L_g + 0.5·(L_ell + L_s)`.
* `L_ell`/`L_s` use `components[:,0]`/`components[:,1]` and the fit-row `ell`/`s` tensors indexed by
  the same batch indices as `batch.y`. SUM's `L_ell`/`L_s` are detached and never in its backward.

## Targets cache

`component_targets.npz`: `ell, s, s_SA, epsilon, g, y, c, k, gid, constants, constant_names, mu_logP`.
`g` equals the frozen fresh-fold `g`; `s = g − ell`; `epsilon = s − s_SA`. Identity checks in
`component_labels.json` (`g = ell + s` max abs `4.4e-16`, `y = g + c` max abs `0.0`).

## Required checks (all green before formal training)

1. `--targets`: identity, finiteness, epsilon stats, per-k summaries, SA-interpretation flag.
2. `--init-check`: SUM/COMP identical init, split-half identity, initial SUM vs original M ≤ 1e-5,
   parameter audit, frozen schedule hash.
3. `--smoke --arm SUM|COMP`: real loss/backward; aux only in COMP; label permutation leaves the
   forward **and** both components unchanged; save/reload state hash and forward identical.
4. Training: standard forward `(batch,)`, cached components `(batch, 2)`; data-shuffle/global-ID
   streams and the frozen schedule hash recorded.
5. Save/replay: init/last/raw_soup states, curve, metadata, fit/dev component and sum raw
   predictions, fit-median bias; single-file CPU/GPU replay diff ≤ 1e-5.
6. Allocation probe (UUID/PCI/node/Slurm/CUDA-visible) recorded inside each task process.

## Analysis

Paired bootstrap (1000, seed 20261008, shared indices), same-prediction-zero and swap-mirror
witnesses, gate conditions, group contribution identities, COMP component diagnostics,
cancellation identities, fixed worst-SUM-row sensitivity, replay. `official_valid_loaded=False`,
`official_test_loaded=False` in every artifact.
