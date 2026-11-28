# Extracted additive functions

* `hiv_functions.json` — 7 MolHIV structures selected from Train support and
  the learned functions only. `f_tree`, `f_nn` and `f_linear` are all on the
  same raw input grid and reference point (structures: 0; size terms: Train
  median). Values are the zero-baseline contribution to the raw margin.
* `hiv_local_explanations.json` — exact per-molecule decomposition of three
  Valid molecules for the better additive model.
* `zinc_functions.json` — size terms, native explicit-subgraph columns (family
  identity only; canonical keys are not persisted) and WL count columns.

All contributions reconstruct the stored model margin; see `closure` blocks.
