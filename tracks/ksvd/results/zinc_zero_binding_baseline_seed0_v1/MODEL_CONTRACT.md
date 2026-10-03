# MODEL_CONTRACT — zinc-zero-binding-baseline-seed0-v1

Three **different objects** are involved in this round.  They are never to be
confused:

| object | what it is | provenance |
|---|---|---|
| **old S_M** | the source arm from `zinc-structure-semantic-factorial-seed0-v1` (sparse tied-IHT + independent-pairing aggregation), 240 epochs, fresh seed-0 init | source branch `task/zinc-structure-semantic-factorial-seed0-v1`, HEAD `4083ee38d878…`, trained at `118e2481362f`; raw soup state `S_M_raw_soup_state.pt`, SHA `2def4cb64c13…` |
| **compressed S_M** (`S_M_deploy`) | an algebraically equivalent *deployment* re-expression of old S_M with the structural slot path removed | produced this round from the released S_M soup; CPU replay |
| **N0** | a **new** 240-epoch trajectory, identical recipe to S_M except both slot tensors are multiplied by zero after aggregation and before the encoders, from the first optimizer step | this round, one run, seed 0, same init/stream as S_M |

* compressed S_M inherits old S_M's performance; it is **not** a new training
  result and **not** trained from zero.
* N0 is **not** the same-weight model as old S_M; only its initialisation,
  data stream and recipe are matched.
* Neither compressed S_M nor N0 is "dictionary-free": both retain the task
  dictionary `D_L/V_L`, Sem108 (which already carries shell-conditioned
  semantic statistics), size2, unit/topology/global encoders and the static
  relation path.

## 1. Old S_M — measured constant binding path

Evidence: `phaseA_precheck.json` (all 8000 fit + 2000 dev rows, eval mode,
CPU float32).

* node slots and edge slots measured **exactly zero** at every batch:
  `absmax = 0.0`, `RMS = 0.0`, within-batch row variation `0.0`, for fit and dev.
* the two slot encoders therefore receive the all-zero tensor on every
  molecule: `node_encoder_out_rowvar = 0.0`, `edge_encoder_out_rowvar = 0.0`
  (the encoder outputs are bias-determined constants, identical across rows).
* their contributions to the fusion are exactly the constants
  `c_slot = [node_encoder(0); edge_encoder(0)]` (336-D).
* the structural code `coord` enters the masked forward **only** through
  `environments_masked -> slots`.  Substituting the coord with zeros or with a
  fixed finite vector, and perturbing `D`, changes the prediction by `0.0`
  (bit-identical) on the fixed check batch.
* `S_M_raw_soup_state.pt` hash matches the source manifest
  (`2def4cb64c1350a60348fb6fc7ce88a07bffe563367ac7e9666ee6f7dff5c5b8`);
  CPU replay of fit matches the released GPU `fit_raw` to `5.7e-6` and the
  released `predictions.csv` rows exactly.

## 2. Compression algebra

Original first fusion operator (`fusion.0`, `Linear(446, 342)`):

```
h = W_sem @ x_sem + W_slot @ c_slot + b_fusion
```

with `x_sem = [Sem108(108); size2(2)]` (110-D) and `c_slot` the measured
constant 336-D slot block.  Export:

```
W_reduced        = W_sem                     # [342, 110]
b_fusion_reduced = b_fusion + W_slot @ c_slot
```

All later operations (`SiLU`, `fusion.2`, task dictionary `D_L/V_L`,
unary/pair/relation/global/topology/reader) are kept unchanged; the output
still gets the **original S_M train bias** (`b = -0.0004399…`, externally
added, exactly as before — the fusion-bias fold and the scalar calibration are
different objects, never double-counted).

Deleted from the deployed model: `D`, `W_A_S`, `W_A_C`, `W_E_S`, `W_E_C`,
`node_encoder`, `edge_encoder`, buffers `U`, `common_rms`, `kappa`, the IHT
coding and the binding projections.  `phi65` is no longer an input.

Measured equivalence (full fit + dev, same device/dtype, eval mode):

| quantity | value |
|---|---|
| `max abs(pred_raw_deploy − pred_raw_S_M)` fit / dev | `1.907e-06` / `1.907e-06` |
| cal/dev MAE difference | `+1.3e-08` |
| label shuffle on fixed batch | `0.0` |
| graph-order (ID-restored) | `2.4e-07` |
| independent re-load max abs | `0.0` |

Tolerance is `1e-5` (FP32 summation order only; algebraically the map is
identical).  This is **not** claimed bit-identical.

## 3. N0 intervention

Inserted in the production environment path (sparse code + independent-pairing
aggregation), after bucket aggregation, before the encoders:

```python
node_slots = node_slots * 0.0
edge_slots = edge_slots * 0.0
```

Both the common-coordinate and the residual-code contributions are zeroed
(the whole slot tensor).  The multiply keeps the autograd graph: upstream
binding parameters get exact zero gradient tensors (measured:
`W_A_S/W_A_C/W_E_S/W_E_C` grads all exactly 0), `D` keeps its reconstruction
gradient (measured `0.073`), and every one of the **408,651** trainable
parameters is still in the Adam parameter set (smoke: 408,651 with grad).
Loss, optimiser (Adam coupled L2, lr 1e-3, wd 1e-5), global clip 5.0 over all
parameters, batch 128, 240 epochs, soup 236–240 are unchanged from S_M.

## 4. Parameter and input accounting

| | old S_M (training) | compressed S_M / N0 (deployment) |
|---|---:|---:|
| trainable parameters | 408,651 | n/a (inference) |
| deployment parameters | n/a | **267,611** (measured) |
| removed deployment parameters | — | **141,040** = 114,912 (fusion first layer, 342×336) + 2,080 (`D`) + 5,856 (node binding) + 4,944 (edge binding) + 9,328 (node encoder) + 3,920 (edge encoder) |

Retained dictionary/statistical inputs (both deployment models):

* task dictionary `D_L [144, 288]`, `V_L [288, 144]` (+ `rho`, thresholds);
* Sem108 = atom-shell `3×28` + bond-shell `6×4` from `patch_cont[:, 0:108]`
  (shell-conditioned, kept);
* size2 = `anchor[:, 60:62]`;
* static relations (`pair_relation` groups), distance gate buckets,
  unit/pair moments, global context (C6: chemistry histograms zeroed),
  topology25.

Removed inference input: `dict_phi` (phi65).  All other batch fields are the
original ones.

## 5. Deployment input schema and replay

Model class: `DeployFull` in
`tracks/ksvd/experiments/luyin16/zinc_zero_binding_baseline_seed0_v1.py`.
Minimum batch fields read by `DeployFull.forward(data, mask=C6_MASK)`:

```
batch, patch_cont, anchor, pair_index, pair_relation, pair_bucket,
global_context, topology_features        # y is never read
```

`dict_phi` may be present but is ignored (`DeployFull.code` returns a zero
placeholder).  Checkpoints: `S_M_deploy_state.pt`, `N0_deploy_state.pt`
(state dicts, load with `build_deploy_model(state)`); deterministic replay
script: `replay_deploy.py`.
