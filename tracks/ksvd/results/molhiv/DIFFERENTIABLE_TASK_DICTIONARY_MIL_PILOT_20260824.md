# MolHIV differentiable task-aware dictionary MIL pilot

## Purpose

Test whether a dictionary initialized by ordinary reconstruction K-SVD can be
rotated by the graph-label objective while retaining a sparse local-code
interpretation. This is a mechanism pilot, not an official-valid/test result.

Each graph is represented by up to eight chemical RW patches. The frozen K-SVD
baseline uses OMP codes and max absolute activation per atom. The task-aware
branch jointly updates the dictionary, soft-threshold sparse code and graph
MIL head with:

```text
graph BCE + 0.2 * patch reconstruction + 0.05 * atom incoherence
```

All fitting is restricted to 5,120 inner-train graphs from an 8,000-graph
official-train subset; each result uses a separate 1,280-graph inner holdout.
Official valid/test are not loaded.

## Results

| inner seed | raw patch max | frozen KSVD OMP maxabs | task-aware dictionary maxabs | task−KSVD | task−raw |
|---:|---:|---:|---:|---:|---:|
| 20260824 | 0.6312 | 0.6221 | **0.6558** | +0.0337 | +0.0247 |
| 20260825 | **0.6794** | 0.6316 | 0.6741 | +0.0424 | -0.0053 |
| 20260826 | 0.6095 | 0.6402 | **0.6903** | +0.0502 | +0.0808 |
| mean | 0.6400 | 0.6313 | **0.6734** | **+0.0421** | **+0.0334** |

The task-aware branch beats frozen ordinary KSVD in all three inner splits. It
also beats the raw max-patch readout in two of three splits. The raw baseline
has much larger split variance, so the raw comparison is not yet conclusive.

## Interpretation

This is the first MolHIV result in this route where the task objective updates
the dictionary itself rather than merely rebalancing the patch pool or
selecting task-scored prototypes. The result supports the narrower claim that
task-aware dictionary learning can preserve or recover graph-label signal that
reconstruction-only KSVD loses.

It does not yet establish a full MolHIV improvement over GINE/CIN. The next
gate should be a fixed multi-scaffold, multi-seed confirmation with the same
dictionary family, followed by an atom-occurrence MIL readout and matched
ordinary KSVD/PCA/random controls. Only after that should a GINE or relation
Transformer be introduced.

## Fixed-split model-seed confirmation

The inner split `20260824` was then frozen and only the dictionary/optimizer
seed was changed:

| model seed | raw patch max | frozen KSVD OMP maxabs | task-aware dictionary maxabs | task−KSVD |
|---:|---:|---:|---:|---:|
| 0 | 0.6312 | 0.6454 | **0.7094** | +0.0640 |
| 1 | 0.6312 | 0.6181 | **0.7127** | +0.0946 |
| 2 | 0.6312 | 0.6260 | **0.7004** | +0.0744 |
| mean | 0.6312 | 0.6299 | **0.7075** | **+0.0777** |

The task-aware branch wins all three model seeds. This is still an internal
subset pilot, but it makes the signal less likely to be caused solely by one
split or one dictionary initialization. The next formal gate should move the
same fixed configuration to multiple scaffold folds and include PCA/random
matched controls in the same graph-level MIL protocol.

## Matched dense projection control

The same fixed split and model seeds were rerun with the soft threshold
removed. This gives a dense supervised projection with the same initialization,
graph MIL readout, reconstruction term and incoherence term.

| model seed | sparse task dictionary | dense task projection | sparse−dense |
|---:|---:|---:|---:|
| 0 | 0.7094 | 0.6958 | +0.0137 |
| 1 | 0.7127 | **0.7187** | -0.0059 |
| 2 | 0.7004 | **0.7087** | -0.0083 |
| mean | 0.7075 | **0.7077** | -0.0002 |

This changes the interpretation substantially: the stable pilot signal is
**task-aware representation learning beyond frozen ordinary KSVD**, but it is
not yet evidence that sparse coding itself is superior to a dense task-aware
projection. The next gate must compare sparse and dense task-aware branches
across scaffold folds before claiming a KSVD-specific advantage.

## Scaffold-fold confirmation (model seed 0)

The same pilot was then run on the three existing 8,000-graph scaffold folds.

| fold | raw patch max | frozen KSVD | sparse task-aware | dense task-aware |
|---:|---:|---:|---:|---:|
| 0 | 0.6523 | 0.6247 | 0.6338 | 0.6504 |
| 1 | 0.5516 | 0.5629 | 0.6200 | 0.6324 |
| 2 | 0.6307 | 0.5862 | **0.6991** | 0.6734 |
| mean | 0.6116 | 0.5913 | 0.6510 | **0.6521** |

The task-aware branches beat frozen ordinary KSVD in all three scaffold
folds. Sparse task-aware beats dense task-aware only in fold 2; the dense
control is marginally higher in the pooled mean. Thus the robust result is
task-aware adaptation, not a demonstrated sparse-KSVD-specific advantage.
