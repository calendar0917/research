# KSVD task-adapted dictionary-primary MIL screen (2026-07-28)

## 1. Question

This screen addresses three connected questions from the current KSVD discussion:

1. Is the present node-level patch representation artificially limited by a node-count cap?
2. Can the dictionary be made task/person-matched rather than purely unsupervised?
3. Is the current KSVD + strong supervised GINE integration masking a better KSVD-primary architecture?

The patch-limit correction remains: `centered_ego_vector()` uses the complete radius-2 ego set. Its `max_nodes=8` argument scales/clips fixed descriptor coordinates; it does not truncate the node-level ego set. The graph-level CoverageRW route has a real `max_nodes=8` walk-patch cap, but that is not the current node-token path.

## 2. Implemented formulation

A focused runner was added instead of extending the already large GINE fusion runner:

- fold-specific 64-d masked-context SSL latent `z_v` for each official-train node;
- 32-atom dictionary initialized from KSVD, random training patches, or PCA;
- cached OMP code as a warm start;
- differentiable exact-top-3 iterative hard-thresholding updates;
- graph-label task loss plus reconstruction and dictionary-anchor losses;
- center-atom embeddings;
- signed/absolute dictionary-occurrence embeddings;
- attention + mean + max MIL/DeepSets graph pooling;
- **zero supervised message-passing layers**.

Objective:

\[
L=L_{\text{task}}+0.1L_{\text{rec}}+0.1L_{\text{anchor}}.
\]

Graph labels supervise only graph outputs; molecule labels are not copied to individual patches.

Important qualification: the supervised classifier is GNN-free, but the 64-d context input comes from a frozen, label-free two-layer masked-context GINE. Therefore this experiment removes the strong supervised three-layer GINE baseline, but is not yet a fully GNN-free raw-patch test.

## 3. Leakage policy and preregistered gate

- Dataset: the 8,000-graph development subset.
- Selection data: official train only.
- Outer split: three existing Bemis-Murcko scaffold folds.
- Dictionary/SSL fitting: fold-inner graphs only.
- Evaluation: once after fixed epoch 30.
- Official-valid evaluations: 0.
- Official-test evaluations: 0.

Promotion required:

- adapted KSVD mean gain over frozen KSVD >= 0.005;
- wins on at least 2/3 folds (later robustness audit: at least 6/9 fold-seed cells);
- beats a matched adapted/random control;
- no reconstruction collapse.

## 4. Seed-0 screen

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| frozen KSVD | 0.7089 | 0.6355 | 0.7780 | 0.7075 |
| adapted KSVD | 0.7232 | 0.6356 | 0.7831 | 0.7139 |
| adapted random patch | 0.7378 | 0.6633 | 0.7847 | 0.7286 |
| adapted KSVD, no reconstruction | 0.7087 | 0.6254 | 0.7718 | 0.7020 |

Seed 0 alone looked encouraging against frozen KSVD:

- mean delta: **+0.00646**;
- 3/3 nominal wins;
- removing reconstruction lost about 0.012 mean AUC relative to adapted KSVD.

But adapted random beat adapted KSVD on all three folds by **+0.01469** mean. Therefore the KSVD-specific claim already failed the matched initialization control.

## 5. Attribution controls

Seed-0 frozen controls:

| Control | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| frozen random patch | 0.7368 | 0.6753 | 0.7840 | 0.7321 |
| frozen PCA | 0.7069 | 0.6503 | 0.7669 | 0.7080 |
| adapted PCA | 0.7048 | 0.6282 | 0.7540 | 0.6957 |

Thus random-patch performance was not created by task adaptation: the frozen random dictionary was already strongest. PCA adaptation was harmful in all three folds.

## 6. Three-fold, three-seed robustness

Primary robust controls:

| Control | Seed 0 fold mean | Seed 1 fold mean | Seed 2 fold mean | Grand mean (9 cells) |
|---|---:|---:|---:|---:|
| frozen KSVD | 0.7075 | 0.7027 | 0.7163 | **0.7088** |
| adapted KSVD | 0.7139 | 0.6927 | 0.7220 | **0.7095** |
| frozen random patch | 0.7321 | 0.7105 | 0.7240 | **0.7222** |

Paired conclusions:

- adapted KSVD - frozen KSVD: **+0.00071** grand mean, 5/9 wins;
- frozen random - frozen KSVD: **+0.01339** grand mean, 8/9 wins;
- adapted KSVD - frozen random: **-0.01268** grand mean, 3/9 wins.

The seed-0 KSVD adaptation gain was not robust. The preregistered promotion gate fails, so this formulation should **not** be promoted to a shallow edge-GNN integration yet.

## 7. Geometry/usage audit

Simple diversity explanations were checked and did not explain the random-prototype win:

- KSVD node-usage effective atoms: 28.58 / 32;
- random-patch node-usage effective atoms: 27.16 / 32;
- KSVD mean absolute coherence: 0.331;
- random-patch mean absolute coherence: 0.356.

So random was neither more usage-balanced nor less coherent.

Prototype audit gives a more plausible distinction:

- random atoms are actual observed fold-fit context prototypes: mean nearest signed patch cosine about 0.997;
- KSVD atoms are synthetic reconstruction directions: mean nearest signed patch cosine about 0.845;
- KSVD covers arbitrary patches better (mean best absolute atom cosine 0.910 vs random 0.881), consistent with better reconstruction;
- nevertheless random prototypes classify better in occurrence-level MIL.

This supports a new hypothesis: **for graph-label MIL, atom prototype fidelity/task identity may matter more than global reconstruction coverage.** KSVD can build better reconstructors while blurring the discrete local identities useful to the graph task.

## 8. Decision on GNN/GINE integration

1. The current KSVD + three-layer GINE remains a valid strong baseline/control.
2. It should not be treated as the only reason KSVD fails to add value: after removing the supervised GINE, robust task adaptation still did not establish a KSVD advantage.
3. Adding a one-layer edge-aware GNN now would confound the unresolved dictionary issue. The fixed promotion rule therefore stops that branch.
4. A better next primary architecture is more likely to be **prototype-constrained task-aware MIL** than another GINE gate/fusion variant.

Recommended next experiment:

- choose atoms from real fold-fit patches (medoid/candidate selection rather than free synthetic columns);
- use graph labels to select/reweight prototypes through graph-level MIL;
- retain a coverage or reconstruction regularizer, but do not allow it to dominate prototype identity;
- compare against frozen random prototypes, KSVD, PCA, and label-shuffled selection;
- only after a prototype-constrained method beats frozen random robustly, add a one-layer edge-aware GNN.

A separate fully GNN-free audit should replace the frozen SSL-GINE latent with the existing raw 848-d permutation-invariant radius-2 descriptor plus a train-only linear/autoencoding projection. That experiment is needed before claiming that all GNN foundations are unnecessary.

## 9. Artifacts

New code:

- `code/extract_molhiv_ssl_latents.py`
- `code/run_molhiv_task_adapted_dictionary.py`
- `code/summarize_molhiv_task_adapted_dictionary.py`
- `code/summarize_molhiv_task_adapted_robustness.py`

Main summaries/audits:

- `results/molhiv/taskadapt_dictionary_primary_mil_scaffold3_seed0_summary.json`
- `results/molhiv/taskadapt_dictionary_primary_mil_3fold3seed_summary.json`
- `results/molhiv/taskadapt_dictionary_usage_geometry_audit.json`
- `results/molhiv/taskadapt_dictionary_prototype_audit.json`

The three latent archives contain only official-train encodings; official-valid/test rows are exactly zero.
