# KSVD prototype occurrence-graph screen (2026-07-28)

## 1. Research question

This screen tests whether the historical integration

> three-layer supervised atom-level GINE + an optional KSVD side channel

should remain the default architecture. The alternative gives the two parts
non-overlapping roles:

1. frozen local prototypes define sparse node assignments;
2. every radius-1 connected component of one prototype's support becomes an
   explicit occurrence node;
3. occurrence edges encode shared original atoms and adjacency through an
   original molecular bond;
4. a graph classifier uses either occurrence MIL alone or one occurrence-level
   GINE layer;
5. **no supervised message passing is performed on the original atom graph**.

This is a cleaner test of whether interactions among already-defined local
motifs help, because a strong original-node GINE cannot relearn and bypass the
radius-2 prototype representation.

Important qualification: the local 64-d input in this screen is still produced
by a frozen, label-free two-layer masked-context GINE. Thus the supervised
architecture is original-node-GNN-free, but this screen alone is not yet a
fully GNN-free representation test.

## 2. Strict protocol

- Dataset: 8,000-graph MolHIV development subset.
- Data allowed for selection/fitting: official-train only.
- Outer evaluation: the existing three Bemis-Murcko scaffold folds.
- Dictionary and SSL context fitting: fold-fit graphs only.
- Epoch policy: fixed 30 epochs and one outer-fold evaluation.
- Official-valid evaluations: **0**.
- Official-test evaluations: **0**.
- Screen seed: 0.
- Batch size: 192 for all controls in this runner.

The matched attribution control uses random real fold-fit latent patches under
exactly the same top-3 occurrence construction as KSVD.

## 3. Occurrence graph audit

Three-fold average statistics over the 6,400 official-train graphs are:

| Family | occurrences / graph | directed edges / graph | singleton fraction | mean occurrence size |
|---|---:|---:|---:|---:|
| random real prototypes | 51.24 | 409.95 | 75.60% | 1.484 |
| KSVD directions | 46.67 | 357.95 | 72.48% | 1.630 |

Both families produce many small occurrence nodes. KSVD gives fewer, slightly
larger connected occurrences. The interaction graph is therefore nontrivial,
but most occurrences are still singletons.

## 4. Three-fold seed-0 results

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| random node MIL | 0.7478 | 0.5860 | 0.7730 | **0.7023** |
| random occurrence MIL | 0.7164 | 0.6799 | 0.7621 | **0.7195** |
| random occurrence GINE | 0.7444 | 0.6416 | 0.7612 | **0.7157** |
| random occurrence gated GINE | 0.7123 | 0.5867 | 0.7527 | **0.6839** |
| KSVD occurrence MIL | 0.7469 | 0.6190 | 0.7453 | **0.7038** |
| KSVD occurrence GINE | 0.6887 | 0.6343 | 0.7624 | **0.6951** |
| KSVD occurrence gated GINE | 0.7174 | 0.6253 | 0.7533 | **0.6987** |
| KSVD GINE, shuffled prototype ID | 0.6611 | 0.6254 | 0.6980 | **0.6615** |
| KSVD GINE, no prototype ID | 0.6631 | 0.6278 | 0.6994 | **0.6634** |

## 5. What the occurrence abstraction contributes

Random occurrence MIL minus random node MIL was:

- fold 0: -0.0314;
- fold 1: +0.0939;
- fold 2: -0.0109;
- mean: **+0.0172**, 1/3 wins.

This is not fold-stable, but it shows that grouping atom assignments into
connected motif occurrences can remove a large failure on fold 1. The
occurrence abstraction itself is therefore meaningful even though it is not a
uniform improvement.

## 6. Does one-layer occurrence GINE help?

### Direct residual GINE

Random occurrence GINE minus random occurrence MIL:

- fold deltas: +0.0281, -0.0383, -0.0009;
- mean: **-0.0037**;
- wins: 1/3.

KSVD occurrence GINE minus KSVD occurrence MIL:

- fold deltas: -0.0583, +0.0153, +0.0170;
- mean: **-0.0087**;
- wins: 2/3, with a severe fold-0 collapse.

The direct message-passing update is too aggressive. It sometimes extracts
useful interaction signal, but its mean result is worse than performing no
message passing among occurrences.

### Zero-initialized scalar-gated GINE

A second implementation starts exactly from the occurrence-MIL model and adds

\[
h' = h + 0.25\tanh(g)\,\mathrm{Dropout}(\mathrm{GINE}(h)), \qquad g_0=0.
\]

Random gated GINE minus random occurrence MIL:

- fold deltas: -0.0040, -0.0931, -0.0093;
- mean: **-0.0355**;
- wins: 0/3.

KSVD gated GINE minus KSVD occurrence MIL:

- fold deltas: -0.0295, +0.0063, +0.0080;
- mean: **-0.0051**;
- wins: 2/3, but fold 0 still collapses.

Final `tanh(g)` values were:

- random: -0.0214, -0.0093, -0.0051;
- KSVD: +0.0023, -0.0017, +0.0018.

The KSVD gates remain extremely close to zero. This is useful diagnostic
behavior: optimization itself largely concludes that the GINE update should
be suppressed. The gate reduces the ungated fold-0 damage from -0.0583 to
-0.0295, but it cannot turn interaction message passing into an improvement.
The random gates move more, in the harmful direction, and fold 1 collapses.

The preregistered gate required mean gain >= +0.005, at least 2/3 wins, and no
severe single-fold collapse. Neither family passes; no multi-seed confirmation
is justified.

## 7. Is persistent KSVD identity meaningful?

For the ungated KSVD occurrence GINE:

- persistent ID minus shuffled ID: **+0.0336 mean**, 3/3 wins;
- persistent ID minus no ID: **+0.0317 mean**, 3/3 wins.

So dictionary identity is not interchangeable bookkeeping. The model does use
stable KSVD atom identity. However, this does **not** establish a KSVD advantage:

- KSVD occurrence MIL minus matched random occurrence MIL: **-0.0157 mean**,
  1/3 wins;
- ungated KSVD occurrence GINE minus matched random occurrence GINE:
  **-0.0206 mean**, 1/3 wins.

The gated KSVD model beats the gated random model by +0.0147 mean only because
the random gated model is severely degraded; it is not evidence of a positive
KSVD interaction gain, since KSVD gated GINE also loses to its own no-GINE MIL
baseline.

## 8. Architecture decision

The current evidence supports the following answer.

1. **A three-layer supervised original-node GINE is reasonable as a strong
   baseline, but not as the natural center of the KSVD method.** Its receptive
   field overlaps the radius-2 dictionary input and gives it a direct path to
   bypass the dictionary.
2. **Removing that GINE is viable.** Prototype MIL and occurrence MIL remain
   competitive classifiers with zero supervised original-graph message
   passing.
3. **Replacing it with a one-layer occurrence GINE is not yet better.** Both
   direct and zero-initialized gated variants fail against occurrence MIL.
4. **The useful new object is the occurrence, not GINE itself.** Connected
   occurrences improve stability in at least one difficult fold and expose
   interpretable motif identity, but interaction should not be forced through
   unrestricted message passing.
5. **Current results still do not support a KSVD-specific advantage over real
   local prototypes.** KSVD identity carries signal, yet matched random real
   prototypes remain stronger on average.

Therefore stop adding GINE gates/layers. The next independent test should
remove even the frozen SSL-GINE representation and run real/KSVD prototype MIL
on raw permutation-invariant radius-2 descriptors after a train-only PCA or
whitening transform. That directly tests whether the GNN foundation is needed
at all, rather than continuing to optimize the same message-passing family.

## 9. Artifacts

Code:

- `code/run_molhiv_prototype_occurrence_graph.py`
- `code/summarize_molhiv_prototype_occurrence_graph.py`

Summary:

- `results/molhiv/prototype_occurrence_graph_scaffold3_seed0_summary.json`

Fold reports:

- `results/molhiv/prototype_occurrence_graph_fold{0,1,2}_seed0.json`
- `results/molhiv/prototype_occurrence_gated_fold{0,1,2}_seed0.json`
