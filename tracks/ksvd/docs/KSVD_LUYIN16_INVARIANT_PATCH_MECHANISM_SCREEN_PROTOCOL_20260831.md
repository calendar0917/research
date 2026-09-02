# luyin16 invariant patch mechanism screen

Protocol: `luyin16-molhiv-invariant-patch-mechanism-screen-v1`

## Purpose

The historical all-center radius-2 proxy cannot be used for further
RAW/INIT/FINAL attribution because its adjacency coordinates depend on node-ID
tie breaks, its topology and chemistry blocks use different node scopes after
the eight-node cap, and `%16` merges observed atom categories.

This screen replaces that object with a deliberately compact statistical
object that is invariant by construction:

- every atom is a center and every radius-2 ego node/edge is retained;
- topology uses root-shell/induced-degree histograms, shell-pair edge
  histograms, and small scalar graph statistics;
- attributes use direct (collision-free) OGB atomic-number and bond-type
  categories, conditioned on root shells;
- no coordinate is a node slot and no node ID enters the feature definition.

It is a mechanism screen, not a performance optimization or an exact
replication of the mentor's unknown 69/624D schema.

## Stages and stopping rules

1. **Object audit.** On 128 official-validation molecules, run three random
   relabelings. Topology, attribute, and joint readouts must have maximum drift
   at most `1e-12`. The audit is label-free and never evaluates official test.
2. **RAW screen.** On each of three frozen official-train-only scaffold folds,
   use deterministic stratified caps of 6000 train and 3000 validation graphs.
   Compare fixed-XGBoost `S`, local topology, local attributes, local joint, and
   their `S+local` fusions. No tuning is performed.
3. **K-SVD attribution.** Run only if `S+joint RAW` beats `S` by at least
   `0.003` mean AUC and wins at least two folds. Each fold then learns its own
   INIT and FINAL dictionary from that fold's training graphs only. Compare
   matched `RAW/INIT/FINAL` readouts and reconstruction errors.

The K-SVD update passes only if FINAL beats INIT by at least `0.003` mean AUC
and wins at least two folds. Better reconstruction alone is not a task pass.

## Boundary

- official validation is used only for the label-free relabel audit;
- all supervised scores use official-train scaffold folds;
- official test is neither encoded nor evaluated;
- no Optuna, K/T sweep, classifier sweep, or ZINC run is authorized here.
