# ERRATA — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

Corrections to the interpretation of the previous local-encoder round
(`zinc_local_tuple_dictionary_vs_mlp_seed0_v1`), issued with this new-fold replication round.
No historical file is modified; this is the current reading.

## E1. "MLP at least matches the dictionary" / "the dictionary form is not needed" was stronger than the statistical evidence

The previous report's one-line mechanism summary said the same-capacity SiLU projection "at least
matches (directionally beats)" the IHT dictionary and "the dictionary form is not required by/for
the interface here". The frozen statistics do not support that phrasing: M_J−D_J is positive in
point estimate on all four dev endpoints but **every paired 95% CI crosses zero**, and the frozen
`MLP_LOCAL_SUPPORT` gate failed on both the `≥ +0.003` point threshold and the CI condition;
`LOCAL_EQUIVALENCE` also failed (CIs wider than ±0.003). The evidence therefore does not establish
either an MLP advantage or an MLP–dictionary equivalence.

**Correct statement.** On the old fold, no advantage of that local IHT dictionary over the fixed
nonlinear same-capacity control was detected; the win/loss/equivalence of the two encoders remains
unresolved. "No detected advantage" is not "equivalent" and not "the dictionary form is
scientifically unnecessary".

## E2. Stopping the encoder search is a budget decision, not a closed scientific question

The previous round's decision to freeze the dictionary-vs-MLP encoder comparison was taken under a
fixed compute/seed/fold budget. It does not prove that dictionary encoders, correspondence
relations, or sparse structure are irrelevant, and it does not close luyin19's transferable
structure–property hypothesis. This round is likewise a single fixed comparison on a new partition
and buys no further encoder, seed or fold.

## E3. The previous B→M_J "performance signal" was directional only

The old M_J−B calibrated G0 gain `+0.003659` did not pass the frozen five-condition gate because
the G0 calibrated CI lower bound was `−0.000438` (overall cal and both raw gains were positive).
This round treats the old number strictly as a read-only reference direction, not as a confirmed
effect and not as a basis for pooling.

## E4. New-fold naming

The new 8000/2000 split is a model-internal re-partition of the same 10000 official-train rows. It
is **not** an official-valid/test result and must not be described as an independent or
never-seen test. The 2000 new-dev rows were in the old fit; the old models trained on them.
