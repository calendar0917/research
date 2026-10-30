# Structural Binding Pretraining — CPU Feasibility Gate (v1)

## Question

Within the frozen LCRC native two-hop representation, does the **actual**
midpoint–terminal correspondence contain a signal that is learnable from
structure alone and is **not** recoverable from the same marginal statistics?

This is a CPU-only feasibility gate.  It does not train a property predictor,
does not design a new architecture, and does not access official valid/test.

## Data and frozen encoding

- Source: official PyG ZINC `subset=True`, **train split only**, 10,000 molecules.
- Properties (`y`) are never read.  Official valid/test are never loaded.
- Frozen LCRC builder and arrays: `tracks/structural-sufficiency/zinc_local_relation_composition_v1/`.
- Frozen binding cache: `cache/lc_train.pt` (content SHA-256 recorded in
  `analysis/results.json`).  It contains 231,664 atom roots, 692,716 ordered
  non-backtracking two-hop paths, and 249,279 bonds.
- The LCRC arrays were independently rebuilt from the raw train graphs and
  compared bit-for-bit with the frozen cache (`lc_array_rebuild.all_equal`).
- A root is a centre atom `v`.  A path is `v -> w -> u`, with midpoint `w`,
  terminal `u`, and bond types `bt1`, `bt2`.
- `mid = type(w) * 9 + (bt1-1)*3 + (bt2-1)` (189 codes).
- `end = type(u)` (21 codes).
- The real root is the multiset of `(mid, end)` pairs over its paths.  All
  models see only the root's path multiset, not atom indices or global IDs.

## Real pairing and controls

For each root with `n >= 2` paths, a **valid standard permutation fake** is a
uniform random permutation of the terminal types across the `n` paths that is
not the identity.  It keeps the multiset of `end` types and the path count
exactly, and changes only the pairing.  Roots for which every permutation is
the identity are excluded from the standard evaluation.

A **compatibility-preserving hard fake** is a uniformly random 2-swap of the
terminal types of two paths `i, j` such that:

- `mid_i != mid_j` and `end_i != end_j`;
- the swapped terminal type is locally allowed at midpoint `i` and at
  midpoint `j`.

Local allowance is a deterministic bitmask computed from the midpoint atom's
neighbour atom types (excluding the root), keyed by the relevant bond type.
It uses only atom type and bond type.  The hard fake preserves the end-type
marginal, the path count, and per-path local compatibility, but changes the
pairing.  The hard root set is small; its coverage and composition are
reported rather than hidden.

An earlier debug version selected the *first* valid 2-swap.  That rule had a
systematic pair-frequency bias (the first swap tended to create rarer pairs).
The final run uses the uniform selector above; only the final run is reported.

## Models compared

All learned heads have the same feature vocabulary and are trained on the same
real/fake pair classification task with balanced labels.

- `binding`: pair head with per-path terminal embedding and midpoint/path
  context.  19,195 parameters.
- `marginal`: same parameter count and architecture, but the per-path terminal
  embedding is replaced by the root-level weighted mean of terminal-type
  counts.  Real and fake inputs are therefore bit-identical, so this head is
  an exact 0.5 control (verified: max score difference 0.0 on every fold/seed).
- `linear_compat`: linear head on 189 x 21 joint pair one-hot counts plus five
  simple compatibility/interaction counts.  3,975 parameters.
- `linear_pair_only`: the same linear head without the compatibility features.
  3,970 parameters.
- `binding_compat`: the binding head plus the explicit compatibility bit
  (diagnostic ablation).
- `frequency`: a no-training add-one-smoothed table of real `(mid, end)` pairs
  counted in the training molecules; a candidate pairing is scored by summed
  log frequency.  This is the simple atom-type + bond-type baseline.
- `context_frequency` (post-hoc simple-statistic check): a no-training
  Dirichlet-backoff table conditioned on
  `(root_deg, root_ring, root_mring, mid_deg, mid_ring, closure)`, with
  `alpha = 20` backoff to the global pair frequency.  This is the strongest
  simple degree/ring/type baseline; it is reported separately and is not part
  of the original 7-point gate checklist.
- `oracle_compat`: a no-training deterministic rule that scores a candidate
  path by local compatibility.  This is the simple structural rule.

## Splits and training

- 5 folds by a typed Weisfeiler–Lehman fingerprint of each molecule.  Repeated
  structures stay in the same fold; no duplicate fingerprint crosses folds.
- Each fold uses 3 folds for training, 1 for development, 1 for test.  The
  development fold is reserved; this gate uses fixed 10 epochs and does not
  select models on it.
- All roots of a molecule are in exactly one fold.  Train/test molecules are
  disjoint.
- Optimization: AdamW, learning rate 2e-3, weight decay 1e-4, batch size 8192,
  10 epochs.  Two model seeds (0, 1) for the binding/marginal/linear heads;
  the diagnostic/control heads use seed 0.
- Reported pair accuracy is the fraction of roots on which the real pairing
  scores strictly above the fake (ties count 0.5).  Confidence intervals are
  95% bootstrap intervals resampling **molecules**, not roots.

## Novel-context groups

For each test root, keys and pair types seen in the training molecules are
recorded.  A root is `unseen_key` if any of its structural keys is absent from
the training molecules, and `unseen_pair` if any of its `(mid, end)` pair
types is absent from the training molecules.  These groups are reported
separately for the binding and linear heads.  They are natural (not
adversarial) novelty and are sparse.

## Leakage and invariance controls

- Independent LCRC array rebuild must equal the frozen cache bit-for-bit.
- Random atom renumbering must preserve the canonical root-signature multiset.
- Random path storage permutation must leave model outputs unchanged.
- Marginal head real/fake scores must be identical (tie rate 1.0).
- Random-label control: train the binding head on within-root random labels;
  evaluate on (a) clean test pairs and (b) test pairs whose labels are also
  randomly permuted.  The proper permuted-label test must return to 0.5.
- Untrained-head control: report the initialization-only bias on clean and
  permuted test pairs.
- No atom row index, global atom ID, path array order, or property label is an
  input.

## Gate criterion

The hard control is decisive because it removes the local compatibility and
end-marginal explanations.  `BINDING_SIGNAL_PRESENT` is allowed only if all of
the following hold on the hard control:

1. binding accuracy is above chance (95% CI lower bound > 0.5);
2. binding beats the simple pair-frequency table (paired molecule-bootstrap
   95% CI of the difference excludes 0);
3. binding beats the simple linear pair-only head (same paired criterion);
4. both binding seeds are above chance;
5. the seed gap is < 0.05;
6. every test fold is above 0.5;
7. at least 300 hard roots are available.

Otherwise the gate returns `NO_NONTRIVIAL_BINDING_SIGNAL`.  The hard set is
also structurally narrow; even a positive result there would need replication
on a broader root population before a GPU pretraining experiment.

## Post-hoc degree/ring/type check

The task explicitly asks whether the result can be explained by simple node
degree, atom type, bond type, and ring flags.  After the final run, a
no-training context-conditioned pair-frequency table was added to test this
directly.  It uses the features above and is reported in `REPORT.md`; it is a
diagnostic, not a change to the 7-point gate.  Its result reinforces the gate:
the binding head is not better than this simple statistic on the hard control.
