# MolHIV Beam8 route closure

> **Superseded for node-level bond-endpoint models.** This document still closes
> the chain / ordinary incidence / overlap routes described below, but later
> full-fold experiments reopened a distinct typed bond-endpoint route. See
> `KSVD_MOLHIV_BEAM8_NODE_ENDPOINT_ROUTE_20260815.md`.

> Date: 2026-08-15  
> Scope: OGB `ogbg-molhiv`, official-train scaffold development only  
> Decision: close the Beam8-specific topology/KSVD route; retain unordered patch augmentation as a separate hypothesis

## Final representation tested

- Complete OGB categorical atom and bond features.
- Typed-canonical Beam8 slots.
- EDGE100 completion with no synthetic completion-chain edges.
- First-layer GINE atom states scattered into canonical slots.
- Exact shared-slot chain transfer before patch pooling.
- Slot-conditioned patch-to-atom residual fusion.
- Explicit base/completion roles and separately normalized incidence channels.

The final matched controls include graph BAG, patch-content shuffle,
degree-preserving chain-endpoint shuffle, and a stricter control that preserves
the true chain endpoints and overlap marginals while shuffling only shared-slot
bindings.

## Coverage result

Across the full 6400-graph official-train scaffold fold-0 partition:

- mean atoms: 25.336;
- mean EDGE100 patches: 6.318;
- mean completion patches: 2.307 (38.7% of patch tokens);
- mean original Beam chain edges: 2.931 undirected;
- edge, node, and incident-edge coverage: 1.0.

Coverage is therefore no longer a confound in the final comparison.

## Development sequence

### 1. Small slot-level pilot

The initial 512/512, seed-0 pilot appeared negative before completion roles
were made explicit. After adding a patch-role embedding and base/completion
separation, one seed showed a `true_chain - GINE` ROC-AUC improvement of
`+0.0234`. The true chain exceeded endpoint shuffle by `+0.0087` and
mapping-only shuffle by `+0.0035` in that run.

This was treated as a screen, not a conclusion.

### 2. Three folds by three seeds

The data subset was fixed with `limit_seed=0`; only model seed changed.

For `incidence_split` across the nine runs:

- mean ROC-AUC delta versus GINE: `-0.0598`, 3/9 wins;
- mean AP delta versus GINE: `-0.0048`, 3/9 wins;
- mean ROC-AUC delta versus matched shuffled split: `-0.0074`, 4/9 wins;
- mean AP delta versus matched shuffled split: `+0.0199`, 6/9 wins;
- mean ROC-AUC delta versus BAG: `-0.0668`, 2/9 wins.

The seed-0 gain did not reproduce. Aligned and shuffled incidence usually moved
together, and BAG was substantially more stable.

### 3. Residual regularization and class balancing

Fusion scales of 0.25 and 0.10 with branch dropout produced occasional large
ROC-AUC gains for split incidence or BAG, but the AP direction reversed and the
true chain lost to mapping shuffle. Fit-fold automatic positive weighting also
failed to produce consistent aligned-incidence or true-chain gains.

These checks rule out the simplest explanations based on an over-strong
residual branch or unweighted BCE collapse.

### 4. Full scaffold fold confirmation

The final run removed the 512/512 limit. It used 4108 fit graphs (152 positive)
and 2292 held-out graphs (88 positive), fixed 10 epochs, hidden width 64, and
model seed 0.

| Variant | Held-out ROC-AUC | Held-out AP | Delta AUC vs GINE |
|---|---:|---:|---:|
| `gine` | 0.6993 | 0.1458 | 0.0000 |
| `incidence_split` | 0.7194 | 0.2121 | +0.0201 |
| `incidence_split_shuffled` | 0.7262 | 0.2054 | +0.0269 |
| `true_chain` | 0.7068 | 0.1753 | +0.0075 |
| `mapping_shuffled` | 0.7203 | 0.1798 | +0.0210 |
| `bag` | 0.6939 | 0.1655 | -0.0054 |

The decisive matched differences are:

- aligned split minus shuffled split: `-0.0068` ROC-AUC, `+0.0066` AP;
- true chain minus mapping-shuffled chain: `-0.0136` ROC-AUC, `-0.0045` AP.

The branch is active rather than dead: the aligned split fusion norm is 1.501,
the true-chain fusion norm is 1.632, and the true-chain output norm is 1.109.

## Conclusion

1. Returning to OGB-MolHIV was useful. It exposed the original 74% coverage
   problem and provided enough scaffold-shift discrimination to reject weak
   Beam8 claims.
2. Node-level canonical incidence can improve over GINE in individual runs,
   but the improvement does not depend reliably on correct patch alignment.
3. Exact Beam chain correspondence does not outperform a control with identical
   endpoints and overlap marginals but shuffled slot binding.
4. The evidence supports, at most, a generic multi-patch molecular augmentation
   hypothesis. It does not support Beam8 continuity, canonical chain order, or
   KSVD as the source of value.

Therefore:

- do not fit KSVD dictionaries for this route;
- do not run official-valid or official-test evaluation;
- do not spend additional runs on larger Beam-chain models without a new
  falsifiable mechanism;
- if molecular work continues, move the unordered patch branch into a separate
  project and compare it against standard molecular augmentations such as
  virtual-node GINE, stronger message-passing backbones, and fixed RDKit/global
  descriptor fusion under the same official-train scaffold protocol.

No official-valid or official-test graph was encoded or evaluated.
