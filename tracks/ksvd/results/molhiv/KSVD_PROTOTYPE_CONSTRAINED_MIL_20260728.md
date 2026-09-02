# Prototype-constrained task-aware MIL screen (2026-07-28)

## Question

Can the current three-layer supervised GINE backbone be replaced by a dictionary/prototype-primary graph architecture, and should the dictionary atoms themselves be matched to the downstream task?

## Pilot

- Candidate bank: 256 **real fold-fit node context latents**.
- Selected atoms: 32 frozen real prototypes.
- Task-aware selector: graph-level candidate occurrence features + logistic ranking + diversity-constrained MMR.
- Downstream classifier: positive top-3 prototype occurrences, atom identity embeddings, attention/mean/max MIL pooling.
- Supervised message passing: **0 layers**.
- Important qualification: the context latent still comes from a frozen, label-free two-layer masked-context GINE.
- Protocol: official-train only; three Bemis-Murcko scaffold folds; fixed 30 epochs; no official-valid/test access.

## Seed-0 results

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| random 32/256 | 0.7736 | 0.6507 | 0.7686 | **0.7310** |
| farthest-point 32 | 0.6944 | 0.5893 | 0.7819 | **0.6885** |
| task-aware real prototypes | 0.7063 | 0.6537 | 0.7567 | **0.7056** |
| label-shuffled selector | 0.7138 | 0.7169 | 0.7814 | **0.7373** |
| cached random patches | 0.6990 | 0.6548 | 0.7553 | **0.7030** |
| KSVD directions | 0.7387 | 0.5967 | 0.7333 | **0.6896** |

Paired findings:

- task-aware − random: **-0.0254**, 1/3 wins;
- task-aware − label-shuffled: **-0.0318**, 0/3 wins;
- random − KSVD: **+0.0414**, 3/3 wins.

The selector fit AUC was 0.918–0.926 with true labels, but even shuffled labels reached 0.830–0.871. This is direct evidence that a high-dimensional one-split task selector can fit label noise and scaffold-specific accidents.

## Decision

The preregistered promotion gate failed. Do not run a multi-seed confirmation of this naive task-aware selector and do not add a shallow GNN on top of it.

The useful architecture conclusion is narrower:

1. A three-layer supervised GINE is **not required** to obtain a competitive graph classifier from local occurrence features; random real-prototype MIL reached 0.7310 mean in this seed-0 screen.
2. However, direct task matching of atom identity is not supported: random and label-shuffled selections beat the true-label selector.
3. The task signal should primarily enter at the **occurrence/readout/interactions** level, while prototype identity remains real, unsupervised, or selected only through nested scaffold-stability criteria.
4. Current evidence still does not establish a KSVD-specific advantage: KSVD directions scored 0.6896 in the matched cosine-occurrence architecture.

## Architectural implication

Use the strong three-layer GINE only as a baseline, not as the unquestioned center of the method. The next high-value architecture is an occurrence graph or bipartite atom↔occurrence model: real local prototypes define motif occurrences, and a small one-layer edge-aware network models interactions among occurrences. This separates roles and avoids making GINE relearn the same radius-2 neighborhood already encoded by the dictionary branch.

A fully GNN-free claim still requires replacing the frozen SSL-GINE latent with the raw 848-d permutation-invariant radius-2 descriptor plus a train-only projection.
