# MolHIV typed Beam8 explicit-incidence pilot protocol

> Date: 2026-08-15  
> Scope: OGB `ogbg-molhiv` official-train only  
> Status: completed; Beam8-specific topology route closed after matched controls

## Question

Does keeping every Beam8 patch as an explicit entity, with role-conditioned
atom--patch incidence and typed patch-chain relations, retain scaffold-held-out
classification information that is lost by patch mean/max or graph BAG readout?

The first pilot is dictionary free. KSVD, PCA, and random dictionaries are not
introduced until the raw Beam8 topology passes its matched controls.

## Representation

- Beam8 geometry: `s=8`, target overlap `o=2`, retained beam `8`.
- Coverage checkpoint: `EDGE100`. The raw continuous Beam cover is completed
  until every molecular bond occurs inside at least one patch. Each completion
  patch is an independent chain segment; it participates in atom--patch
  incidence but never creates a synthetic Beam transition.
- Canonicalization colors use the complete OGB categorical atom-feature row.
- Typed edges use the complete OGB categorical bond-feature tuple.
- A patch stores canonical atom slots, typed internal bonds, coverage position,
  and its original atoms.
- An atom--patch incidence edge stores canonical slot, center status, previous
  overlap status, and following overlap status.
- Consecutive patches store the exact shared-atom slot correspondence in both
  directions.

The checkpoint choice follows a 96-molecule official-train-only geometry
audit. BASE covered `74.26%` of bonds on average. FAIR95 raised this to `99.61%`
with `6.11` patches per molecule, while EDGE100 required only `6.24` patches
and guaranteed complete coverage. The marginal cost of EDGE100 was therefore
small enough to remove the residual coverage confound entirely.

The model performs one atom GINE layer, scatters those atom states into their
canonical patch slots, optionally transfers states through the exact shared-slot
chain correspondence, and sends slot-conditioned messages back to atoms before
the remaining GINE layers. Patch pooling is retained only for the historical
`node_meanmax` and `bag` controls. The final residual projection is zero
initialized, so every candidate initially equals the matched GINE function.

## Matched variants

- `gine`: original atom/bond GINE only.
- `node_meanmax`: learned patch states are reduced to per-atom mean/max before
  fusion, reproducing the coarse historical interface.
- `incidence_no_chain`: role-conditioned per-incidence messages, no patch chain.
- `true_chain`: true typed Beam8 chain followed by explicit incidence fusion.
- `patch_shuffled`: patch content is permuted within each molecule/component.
- `chain_shuffled`: chain endpoints are permuted while retaining the chain
  degree distribution.
- `bag`: graph-mean patch state broadcast to all atoms.

All variants use the same atom backbone, parameter initialization, optimizer,
data order, epoch count, and final graph readout. Unused branches remain
dormant rather than changing baseline capacity.

## Development protocol

1. Use the existing Bemis--Murcko scaffold cache aligned to the requested
   MolHIV subset.
2. Encode and train only graphs in the selected official-train fold partition.
3. Run a small fixed-epoch smoke first; no held-out-driven epoch selection.
4. Record BASE/FAIR95/EDGE100 coverage and completion-patch overhead before
   interpreting model scores.
5. Fast screen: three scaffold folds, model seed 0.
6. Confirmation, only after a positive fast screen: three folds by three model
   seeds.
7. Report ROC-AUC and average precision because MolHIV is strongly imbalanced.

## Promotion gate

The raw topology advances only if `true_chain`:

- improves mean scaffold-held-out ROC-AUC over `gine` by at least `+0.003` in
  the fast screen and wins at least two of three folds;
- exceeds `node_meanmax`, `bag`, `patch_shuffled`, and `chain_shuffled` in mean;
- does not obtain an opposite material result in average precision;
- preserves exact required relabel invariants in the data audit.

The confirmation gate is mean `true_chain - gine >= +0.005`, at least six of
nine wins, and positive Beam8-specific matched-control differences. Failure
stops dictionary fitting and larger fusion models.

## Boundary

- This pilot tests Beam8 topology, not KSVD-specific value.
- OGB official-valid/test are historically exposed elsewhere in the repository
  and are not valid model-selection or new terminal-claim targets.
- A positive result requires later confirmation on a pre-registered external
  molecular split or untouched benchmark.
