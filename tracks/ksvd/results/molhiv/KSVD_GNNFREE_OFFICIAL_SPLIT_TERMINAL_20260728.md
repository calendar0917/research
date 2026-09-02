# KSVD / GNN-free raw-patch MIL: official-split terminal evaluation

Date: 2026-07-28

## Scope and frozen protocol

This is a terminal evaluation on the **8,000-graph stratified development subset**, while preserving/remapping OGB's official train/valid/test membership. It is **not** a run on the complete 41,127-graph ogbg-molhiv dataset.

- official train: 6,400 graphs
- official valid: 800 graphs
- official test: 800 graphs
- representation: complete radius-2 permutation-invariant raw patch descriptor -> train-fit PCA64 -> unit normalization
- dictionary: 32 KSVD directions, sparsity 3; PCA and KSVD both fit only on official-train patches
- supervised models: 30 fixed epochs on official train, with no epoch selection
- seeds: 0, 1, 2
- predeclared aggregation: arithmetic mean of the three seeds' positive-class probabilities
- official valid and test were each evaluated once per frozen seed/model
- no post-test tuning or rerun is permitted under this protocol

The repository contains earlier experiments that had already inspected official test, so this should be described as a controlled frozen terminal comparison, **not** as an untouched-test claim.

## Results (ROC-AUC)

| Model | Valid seeds | Valid mean +/- sd | Valid ensemble | Test seeds | Test mean +/- sd | Test ensemble |
|---|---|---:|---:|---|---:|---:|
| GNN-free random real-prototype node MIL | .6685, .6929, .7287 | .6967 +/- .0303 | .7116 | .5918, .7055, .7318 | .6764 +/- .0744 | .7006 |
| GNN-free KSVD-direction node MIL | .7003, .5907, .6885 | .6599 +/- .0602 | .6774 | .6748, .6358, .6499 | .6535 +/- .0197 | .6777 |
| 3-layer original-node GINE | .6974, .7210, .7285 | .7156 +/- .0162 | **.7463** | .6838, .7302, .7141 | .7094 +/- .0236 | **.7238** |

Ensemble deltas:

- random MIL - GINE: valid `-0.0347`, test `-0.0231`
- KSVD MIL - GINE: valid `-0.0689`, test `-0.0460`
- KSVD MIL - random MIL: valid `-0.0342`, test `-0.0229`

## Interpretation

1. **The scaffold-development win did not transfer to the official held-out splits.** On both official valid and official test, the fixed 3-layer GINE ensemble is best.
2. **Random real prototypes remain stronger than KSVD directions at the ensemble level**, matching the earlier conclusion that persistent dictionary identity is useful but current reconstruction-oriented KSVD does not produce the best task-facing prototype geometry.
3. **The GNN-free route is not dead, but it is not yet a replacement for GINE.** Random-prototype MIL reaches `.7006` test AUC versus `.7238` for GINE, a gap of `.0231`; however its seed variance is high, especially because seed 0 collapses to `.5918` test AUC.
4. **KSVD is more stable on test than random prototypes but consistently lower.** Its test seed standard deviation is `.0197`, versus `.0744` for random MIL, while its ensemble AUC is only `.6777`.
5. The immediate research problem is therefore not “add more GINE to KSVD,” but to improve prototype selection/stability and task alignment without assigning graph labels naively to local patches.

## Leakage / alignment audit

- official split overlaps: train-valid `0`, train-test `0`, valid-test `0`
- dictionary fit graph count: `6,400`
- dictionary fit-index SHA256 equals official-train-index SHA256: `bc9befa7cb2ffb16f33ae741b8a5ecbbce1b7b653115d5a42b7ecf1b313d1e1a`
- PCA64/KSVD source cache encoded no valid/test rows during fitting
- valid/test raw-patch latents were generated only after PCA/KSVD were frozen
- all train/valid/test latent rows are present and unit-normalized
- model parameter counts match the development implementations:
  - each GNN-free node MIL: `46,658`
  - GINE: `37,382`

## Artifacts

- `results/molhiv/node_tokens_n8000_a32_t3_pca64_officialtrainfit.npz`
- `results/molhiv/rawpatch_pca64_latents_n8000_officialtrainfit_all.npz`
- `results/molhiv/rawpatch_pca64_official_terminal_seed{0,1,2}.json`
- `results/molhiv/fixed_epoch30_gine_official_terminal_seed{0,1,2}.json`
- `results/molhiv/rawpatch_pca64_official_terminal_3seed_summary.json`
