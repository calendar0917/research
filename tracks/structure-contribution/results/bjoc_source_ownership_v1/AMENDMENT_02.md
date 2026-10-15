# AMENDMENT 02 — sham-arm grouping vectorisation (performance only)

Recorded after the seed-0 N/O arms finished (both at wall ≈ 9.5/10.5 min,
encoder/init hashes matching) and while the seed-0 S arm was still running.

## What happened

The first S run (`own-s0-S-20261015-125951-f507fd49`) reached ~21 epochs at
**16.1 s/epoch** (N/O: 1.6 s/epoch), i.e. ≈ 96 min projected for 360 epochs,
which would have violated the preregistered 1.5 GPU-hour cap.  It was cancelled
(exit 143) **before any soup was written**; no result from it exists.

Cause: `_group_shuffled` looped over every molecule in the batch with small
per-graph tensor ops (sort, repeat_interleave, hash, scatter), which is
latency-bound on CUDA.

## Change (semantics preserved exactly)

`_group_shuffled` now performs the identical draw fully batched:

* canonical per-molecule junction order = stable sort by key, then stable sort
  by molecule id (exactly lexicographic `(molecule, key)` with original row as
  the tie-break);
* per-bond slot expansion `repeat_interleave` over `n_e`, within-bond counter
  `k`, hash `(mol_key*A) ^ (bkey*B) ^ (k*C) & MASK`, `r = hash % n_j(mol)`,
  scatter-add of `wcan[start(mol)+r]` into `W` and its square into `V`.

No constant, draw rule, key definition, count, or format changed.  A new unit
test (`test_sham_vectorized_grouping_matches_per_graph_reference`) reproduces
the old per-graph loop independently and asserts bit-exact equality of
`W, V, n_e` on a synthetic batch (25/25 tests pass).

## Consequences for the arm comparison

* N/O soups are unaffected: their grouping path is frozen upstream code
  (`src.BJOC.group_to_bonds` NULL/ORGANIZED) and was untouched.  They were
  trained from the same scratch init `d06d5e16…`, the same recorded SSL encoder
  and the same batch order hash `d142e307…`.
* S is retrained from scratch (same seed, SSL init, recipe) with the new
  grouping.  The frozen freeze is re-issued; the comparison stays within one
  protocol because only the implementation speed of the sham changed, and the
  equivalence test pins the semantics.

## A2.1 Trained-soup mechanism completeness (analysis code, pre-mechanism-run)

Before the mechanism audit ran, `cmd_mechanism` was extended to evaluate the two
frozen mechanism-gate items directly on the trained soups instead of relying on
the init-time CPU-gate evidence:

* per-molecule `c` invariance under a random `p_global` override
  (`torch.equal(f0["c"], f_glob["c"])`, all arms/seeds/molecules);
* for O, editing one junction's incidence on 3 real molecules and asserting
  `c_e` is bit-identical outside the touched bonds.

These are assertions of already-frozen gate properties (config `gates_seed0.mechanism`),
not new statistics.  No protocol quantity, gate threshold or decision changes.

## A2.2 valid-secondary seed availability (analysis code, post-seed-0 decision)

`cmd_valid_secondary` assumed both seeds existed for every arm.  Seed 1 was not
purchased (seed-0 gates failed), so the command now uses every seed file that
exists and records `arm_seeds_available`.  No selection, threshold or gate uses
the official-valid set; it remains descriptive only.
