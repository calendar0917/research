# MolHIV Beam8 explicit-incidence update

> Date: 2026-08-15  
> Scope: `ogbg-molhiv` official-train scaffold development only  
> Decision: keep OGB, stop KSVD promotion, revise the raw topology interface first

This intermediate update is superseded by
`KSVD_MOLHIV_BEAM8_ROUTE_CLOSURE_20260815.md`, which includes explicit
base/completion roles, mapping-only shuffle, three folds by three seeds, and a
full scaffold-fold confirmation.

## What changed

- Added BASE, FAIR95, and EDGE100 coverage checkpoints.
- Completed patches are typed-canonical, participate in atom--patch incidence,
  and are isolated from the original continuous Beam chain.
- Moved the neural interface to node granularity: first-layer GINE atom states
  are scattered into canonical patch slots, exact shared-slot chain messages
  are applied before pooling, and slot-conditioned messages return to atoms.
- Kept matched GINE, mean/max, BAG, patch-shuffle, and chain-shuffle controls.

## Coverage audit

The audit used 96 molecules sampled only from scaffold fold 0's official-train
fit/held-out partitions.

| Checkpoint | Mean patches | Mean edge coverage | Fully covered graphs |
|---|---:|---:|---:|
| BASE | 3.833 | 0.7426 | 0.0104 |
| FAIR95 | 6.115 | 0.9961 | 0.8750 |
| EDGE100 | 6.240 | 1.0000 | 1.0000 |

EDGE100 costs only 0.125 patches per graph beyond FAIR95, so it is the new
default. It also preserves the original mean Beam-chain edge count (2.781
undirected edges per graph), confirming that completion did not create fake
chain transitions.

## Slot-level pilot

Configuration: scaffold fold 0, 512 fit and 512 held-out molecules, model seed
0, 10 fixed epochs, hidden width 64, EDGE100. The fit side contained 19
positives and the held-out side 20 positives.

| Variant | Held-out ROC-AUC | Held-out AP | ROC-AUC delta vs GINE |
|---|---:|---:|---:|
| `gine` | 0.6857 | 0.1058 | 0.0000 |
| `node_meanmax` | 0.5883 | 0.1469 | -0.0974 |
| `incidence_no_chain` | 0.6235 | 0.1330 | -0.0622 |
| `true_chain` | 0.6159 | 0.1160 | -0.0698 |
| `patch_shuffled` | 0.6233 | 0.1567 | -0.0624 |
| `chain_shuffled` | 0.6276 | 0.1376 | -0.0580 |
| `bag` | 0.6115 | 0.1293 | -0.0742 |

The raw explicit topology does not pass the promotion gate on this pilot.
`true_chain` is below GINE, below `incidence_no_chain`, and slightly below the
degree-preserving shuffled-chain control. `patch_shuffled` is essentially tied
with aligned incidence in ROC-AUC and is higher in AP. The AP movements are not
enough to override the ROC-AUC result with only 20 held-out positives.

The result is not caused by a dead residual branch: after training, fusion
weight norms are 1.10--1.65 for the non-GINE variants and the true-chain output
norm is 0.597.

## Interpretation

1. OGB-MolHIV is more diagnostic than the small TU screens: it exposed a large
   BASE coverage deficit and rejects the current Beam relation signal under
   scaffold shift.
2. Moving from patch mean/max to true slot-level incidence is necessary for a
   valid test, but it is not sufficient. The aligned chain has not demonstrated
   value over its shuffled control.
3. EDGE100 completion patches are 39.3% of patch tokens in the 1024-molecule
   pilot, while only 51.6% of tokens are eligible for a real Beam chain. Mixing
   these roles in one fusion channel is the clearest remaining representation
   confound.
4. The large fusion norms and the `node_meanmax` fit/held-out gap
   (0.9350/0.5883 ROC-AUC) indicate overfitting rather than an optimization
   failure to activate the branch.

## Next experiments, in order

1. Separate original Beam patches from EDGE100 completion patches in the model:
   add an explicit patch-role embedding and independently normalized base and
   completion incidence channels. Include base-only and completion-only
   ablations.
2. Constrain the residual branch with a learned scalar gate, lower branch
   learning rate, and stronger branch dropout. The goal is to prevent the patch
   path from replacing a better GINE representation in ten epochs.
3. Test whether chain correspondence helps only locally: transfer aligned
   shared-slot states but fuse only into atoms that occur in both neighboring
   patches. Compare against a mapping-shuffled control with identical endpoints.
4. If and only if a revised raw variant beats GINE, BAG, patch shuffle, and
   chain shuffle on all three development folds, run three seeds and then test
   dictionary features. Do not fit KSVD before this gate.

No official-valid or official-test graph was encoded or evaluated.
