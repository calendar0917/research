# BondAnchoredTriple-JointTune-v1 — joint representation + composition fine-tune
- run id: `20260930-154827-2dc792ad`
- verdict: **JOINT_TUNE_SCREEN_NO_STRONG_SIGNAL**
- soup valid MAE `M_joint_soup = 0.123159514`
- online hot start `M_start = 0.123002916` (expected 0.123002916)
- `Delta_vs_start = +0.000156598` (overall joint adaptation)
- parent replay `M_parent = 0.123704928` (background only)
- best valid `0.122973156` @ epoch 6
- last-10-epochs valid mean `0.124253387`
- final train MAE `0.046758975`
- soup members `[27, 25, 40, 38, 22]` / member valid `[0.123404, 0.123408, 0.12343, 0.123519, 0.123659]`
- 40 epochs completed=True, wall 405.3s (10.13s/epoch, first epoch 9.5s, estimated remaining after epoch 1 371s), peak RSS 2743.9 MB
- device: `cpu`, 8 threads (local execution this round; no remote A100 work)

## Scope
- one candidate, seed 0, fixed 40 epochs, batch 128, new Adam lr 1e-4, coupled wd 1e-5, clip 5 over exactly the unfrozen whitelist, graph-level L1, Top-5 soup over epochs 21-40;
- the shared dictionary coordinates (`D`, `U`, `common_rms`), the semantic interface, the C6 rule, `W_A_S`/`W_A_C`, `node_encoder`, `global_encoder`, `topology_encoder` and the six BAT-v1 normalisation buffers are frozen;
- the frozen feature extractor stays in eval mode (original backend dropout off); the dynamic chain is recomputed online with autograd every step (no cached `E`/`p_ij`/`z_old` training input);
- no M0 retraining, matched control, shuffle, ablation, branch-off, mechanism arm, extra seed, learning-rate search, early stop or horizon extension; official test never instantiated.

## Frozen sources
- Sem108 backbone `tracks/ksvd/results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt` sha256 `6ec0fdef824d2c93973eaba1847c1173b9d939c75276476caf672d8010a74f89`, canonical state `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a` (49 keys, 97709 params, old Reader 4135)
- BAT-v1 M1 soup `tracks/ksvd/results/e2e_dictenv_bond_anchored_triple_v1/checkpoints/BAT-v1-seed0_soup_full_state.pt` sha256 `0f2b170cfaa28b8df25d925906389a21a93b494ab120e0e35dc1247549f05a55`, canonical state `f898a595cbc66b890495a2faa52f44ee26916c2af6fd5828842f9999de675a5b` (16 keys), members `[50, 57, 43, 49, 71]` valid `[0.124576, 0.125264, 0.125437, 0.125595, 0.125666]`; buffers match `standardizers.json`: True
- source checkpoint files unchanged before/after: `True`

## Parameter accounting (measured)
- trainable `82805` params / `34` tensors; frozen `20952` params; total registered `103757`; removed old 302-D Reader `4135` params (not registered)
  - `F`: 5216 params
  - `distance_gate`: 80 params
  - `edge_binding`: 4944 params
  - `edge_encoder`: 3920 params
  - `fusion`: 56478 params
  - `pair_encoder`: 5328 params
  - `pair_projection`: 768 params
  - `reader`: 4967 params
  - `relation_encoder`: 1104 params
- frozen subset unchanged before/after: `True` (params sha `7bdc8ddceef9d0f8`, buffers sha `8dc1691c5dc424f2`, fixed coordinates sha `6a89796b8adb5fe6`)

## Correctness
- `batch_mapping`: True
- `hot_start`: True
- `mode_forward`: True
- `one_batch_gradient`: True
- `online_vs_cached`: True
- `parameter_accounting`: True
- `structure_audit_train`: True
- `structure_audit_valid`: True
- online hot start `0.123002916414232` vs `0.123002916414232` (abs diff 0.000e+00, tolerance 1.0e-06)

## Curve (every 5th epoch)

| epoch | train MAE | valid MAE | s/epoch |
|---:|---:|---:|---:|
| 1 | 0.057079 | 0.123307 | 9.51 |
| 5 | 0.053785 | 0.123290 | 10.11 |
| 10 | 0.051637 | 0.126127 | 10.14 |
| 15 | 0.050295 | 0.123575 | 9.71 |
| 20 | 0.049336 | 0.123853 | 10.33 |
| 25 | 0.048328 | 0.123408 | 9.71 |
| 30 | 0.047708 | 0.124101 | 10.69 |
| 35 | 0.046751 | 0.124619 | 10.22 |
| 40 | 0.046759 | 0.123430 | 10.51 |

## Interpretation boundary
Single-candidate 40/40-epoch joint-adaptation performance screen with fixed dictionary coordinates.  Delta_vs_start describes this round's overall joint adaptation; it is not the triple operator's independent effect and cannot establish that the three-environment mechanism or the dictionary coordinates are irreplaceable.  No M0 retraining, matched control, shuffle, ablation, mechanism arm, extra seed or added training; the official test was never read.
