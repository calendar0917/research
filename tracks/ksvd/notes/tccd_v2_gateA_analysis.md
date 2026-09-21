# TCCD-v2 — Gate A Analysis

Pre-registration: `notes/tccd_v2_preregistration.md`.
Implementation / formal Gate A commit: `69a985a4ea3eb184c96c8ddad3857c4e87ee1dda`.
Pre-registration commit: `6773866e7d25b08acebf6ca505ae8e588e505ccf`.
Device: remote A100 **GPU1**. Seed: `0`. Split: TCCD-v1 internal split
seed `20260922`, 8,000 train / 2,000 dev; 231,664 train-record patches.
Official test: never loaded. K-SVD refit: **NO**. OMP/IHT: **NO**.
Raw reconstruction loss: **NO**.

## Matched architecture

Dense and Prototype arms share the same `714 -> 64` task-learned linear local
encoder initialization and optimizer family. Prototype arms use `K=64`
shared soft prototypes, cosine assignment, one global trainable temperature
`tau = 0.05 + 0.95*sigmoid(a)`, and only assignment-derived BAG / `C^T R C`
features. No continuous latent bypass is present.

Relations are exactly the TCCD-v1 set: overlap, three native bond relations,
and global relative position. Reader and training protocol are the same
lightweight vectorized TCCD-v1 protocol. The regularizers are calibrated once
on the first batch and frozen; each initial contribution is 5% of initial task
loss.

## Results

| arm | best internal-dev MAE | Top-5 soup MAE |
|---|---:|---:|
| Dense-REL | 0.410789 | 0.371536 |
| Prototype-BAG | 0.354108 | 0.345701 |
| Prototype-REL | **0.286244** | **0.266352** |
| Prototype-REL-SHUFFLE | 0.838816 | evaluation-only |

The prior TCCD-v1 DENSE result was local CPU, so the matched Dense-REL arm was
rerun on GPU1 rather than mixed across execution regimes. Its historical CPU
reference was best 0.403933 / soup 0.387409.

Composition gain:

    Delta_comp = 0.838816 - 0.286244 = 0.552572

The preregistered minimum was `0.010`; Gate A **PASS** decisively.

Prototype bottleneck gap:

    Delta_proto = 0.286244 - 0.410789 = -0.124546

The preregistered strong-pass condition was `Delta_proto <= 0`; this is a
**STRONG PASS**. Prototype-REL is better than the matched GPU1 Dense-REL on the
best-checkpoint decision metric. The soup result is also better by `0.105185`.

## Calibration and runtime

Prototype-REL first-batch calibration: initial task `1.595759`, local entropy
`3.924510`, balance KL `0.227467`; `lambda_local=0.020331`,
`lambda_balance=0.350767`; each initial contribution `0.079788`.

Wall times: Dense-REL `214.1 s`, Prototype-BAG `369.3 s`, Prototype-REL
`1119.3 s`. Peak GPU memory was about `50.7 MB` for Dense/REL. The run was
vectorized and did not repeat patch extraction or relation construction.

## Decision

Gate A allows vocabulary quality analysis and the absolute gate. No paired seed
was needed. The result directly supports assignment-sensitive prediction in a
shared learned prototype vocabulary, beyond both BAG and shuffled assignment.
