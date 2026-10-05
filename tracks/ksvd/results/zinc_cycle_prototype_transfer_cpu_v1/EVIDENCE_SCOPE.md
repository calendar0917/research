# EVIDENCE_SCOPE — `zinc_cycle_prototype_transfer_cpu_v1`

## What this round is

A single-rule, CPU-only, **no-training** test of one fixed deployable cycle prototype lookup
(`H`) built from exact train T25 classes, compared to the frozen COMP+Q reference (`B`) on the
**reused and already-exposed** official-valid split (1000 rows).

## What this round is not

* **Not** an independent confirmation. official-valid was used by earlier research and by last
  round; it is a reused exposed validation split.
* **Not** a chemistry improvement. Body `h` and `Q` weights are unchanged; the `g` prediction is
  identical to last round and only its identity is replayed.
* **Not** a proof of a learned shared structure–attribute dictionary. It is a topology-class
  prototype / lookup reference.
* **Not** a multi-seed or training-randomness replication. One seed (0), one candidate.
* **Not** an official-test read. official-test is never instantiated, predicted or scored.
* **Not** the earlier "correct the total residual per topology class" idea: this uses the
  already-separated, train-supervised cycle target `c` directly.

## Reused exposed validation

Explicit disclosure: this round **reuses the exposed official-valid**. The new candidate `H` is
evaluated once, unconditionally, after the frozen package and eval manifest were written
(`frozen_eval_manifest.json`). The main result uses all 1000 rows; no row deletion.

## Evidence boundaries

* The CI covers only this fixed model on these 1000 rows. It excludes training-seed and
  selection-history uncertainty.
* `SINGLE_ROW_DOMINATED = true`: the entire measured gain is one row (`valid:0172`, k=−6); the
  other 999 rows are net flat/slightly worse. The gate passes points 1/3/4 but fails CI point 2,
  matching `TARGETED_REPAIR_ONLY`.
* Unseen/uncovered rows: the 12 `UNSEEN_FALLBACK` rows receive no candidate information by
  construction; their large Q error is *not* addressed. This is "no coverage", not "covered but
  not learned".
* Whether T25 is invariant to node renumbering is **not** re-derived here; the round inherits the
  source implementation and does not infer whole-graph permutation invariance from the lookup.
* The oracle (`y = h + c + b_B`) reads scoring labels only, never enters deployment/calibration/
  gate, and is not a reachable upper bound.

## Held-out access

* official-valid: **read** (reused; see `heldout_access.json`).
* official-test: **not read**.
