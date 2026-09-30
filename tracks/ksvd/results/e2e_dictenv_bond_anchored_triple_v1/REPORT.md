# BondAnchoredTriple-v1 — mainline performance screen
- run id: `20260930-143250-4edd72a2`
- verdict: **FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL**
- soup valid MAE `M_soup = 0.123002916`
- parent replay valid MAE `M_parent = 0.123704928` (historical 0.123704928)
- `Delta = M_soup - M_parent = -0.000702012` (unmatched screen difference)
- best valid `0.123930523` @ epoch 30
- last-20-epochs valid mean `0.127611312`
- final train MAE `0.060109160`
- soup members `[50, 57, 43, 49, 71]` / member valid `[0.124576, 0.125264, 0.125437, 0.125595, 0.125666]`
- 80 epochs completed=True, wall 330.9s (4.14s/epoch), peak RSS 3517.2 MB
- device: `cpu` (remote A100 unavailable this round: NVML/driver `GPU0000:4B:00.0: Unknown Error`)

## Scope
- one candidate, seed 0, fixed 80 epochs, batch 128, Adam lr 1e-3, wd 1e-5, clip 5, Top-5 soup over epochs 41-80;
- the parent `CSSD-Sem108 + C6` soup is frozen and never retrained; `p_ij` and `z_old` are cached under `no_grad` and the training model contains no parent parameter;
- no third-environment shuffle, code/atom/semantic shuffle, mechanism or ablation arm; no second seed; no hyper-parameter search; no 320-epoch extension; official test never instantiated.

## Frozen parent
- checkpoint `tracks/ksvd/results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt` sha256 `6ec0fdef824d2c93973eaba1847c1173b9d939c75276476caf672d8010a74f89`
- canonical state sha256 `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a` (49 keys)
- params 97709 (reader 4135), pair width 16, env width 48, reader input 302
- mask `C6 (unary=count, pair=count, relation=path_count, global=atom_histogram+bond_histogram)`
- parent state unchanged before/after training: `True`

## Candidate
- F 48->64->32: 5216 params; new Reader 4967 params (+832 vs the replaced parent Reader); trainable 10183; full model 103757
- triple objects: one per real bond anchor `(i, j)` (`i < j`) times every third node `k`, `m*(n-2)` per graph, zero summary for degenerate graphs
- `z3 = [mean(t); mean(t*t)]` over that graph's triples; fixed train-only standardizers for `p`, `z_old` and the initial `z3`

## Correctness
- `batch_pooling`: True
- `cache_alignment`: True
- `endpoint_swap`: True
- `hand_pooling`: True
- `one_batch_gradient`: True
- `parameter_accounting`: True
- `parent_freeze`: True
- `parent_replay`: True
- `triple_audit_train`: True
- `triple_audit_valid`: True

## Curve (every 10th epoch)

| epoch | train MAE | valid MAE | s/epoch |
|---:|---:|---:|---:|
| 1 | 0.673171 | 0.359444 | 3.90 |
| 10 | 0.083883 | 0.133600 | 9.25 |
| 20 | 0.072821 | 0.124984 | 3.92 |
| 30 | 0.066583 | 0.123931 | 3.97 |
| 40 | 0.066209 | 0.126399 | 4.06 |
| 50 | 0.064112 | 0.124576 | 4.08 |
| 60 | 0.063761 | 0.135224 | 4.15 |
| 70 | 0.061766 | 0.131174 | 4.20 |
| 80 | 0.060109 | 0.127332 | 4.08 |

This is a performance screen, not a matched control.  Delta is the difference against the historical, unmatched parent soup; it is not the treatment effect of the triple operator (the Reader is re-initialised, the inputs are standardised and the 80-epoch budget differs).
