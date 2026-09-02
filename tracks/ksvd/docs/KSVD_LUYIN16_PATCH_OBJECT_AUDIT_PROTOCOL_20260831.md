# luyin16 MolHIV patch-object audit protocol

Protocol: `luyin16-molhiv-patch-object-audit-v1`

## Question

Before any further XGBoost or K-SVD search, determine whether the current
all-center radius-2 object is a defensible graph representation:

1. Does random node relabeling leave the graph readout and frozen-model
   prediction unchanged?
2. Do the adjacency, atom histogram, and bond histogram describe the same
   retained node set when a radius-2 ego graph exceeds eight nodes?
3. Does the current modulo-based atom/bond encoding merge categories that
   actually occur in MolHIV?

## Scope

- Data: official MolHIV validation only for the relabel audit; official test is
  neither loaded into the feature cache nor evaluated.
- Sample: 256 stratified validation graphs, five deterministic relabelings per
  graph.
- Representation: the frozen all-center radius-2, 52D patch object and 12-block
  coordinate-wise readout.
- Prediction audit: one frozen `S+R_raw` XGBoost parameter set, trained on the
  already frozen official-train feature cache. No parameter selection occurs.
- Scope audit: compare the historical full-ego atom/bond histograms with a
  matched retained-node version while leaving all other operations unchanged.

## Gates

- Relabel invariance passes only when maximum feature drift is at most `1e-8`
  and maximum prediction drift is at most `1e-6`.
- Object scope passes only when topology and attributes use the same retained
  node/edge set, or truncation is absent.
- Category encoding passes only when no observed raw category collision is
  introduced by the configured modulo binning.

Any failed gate blocks the fold-level RAW/INIT/FINAL attribution until the
object definition is corrected. This audit does not optimize performance.
