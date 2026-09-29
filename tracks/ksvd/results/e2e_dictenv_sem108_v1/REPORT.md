# e2e_dictenv_sem108_v1 — CSSD-Sem108 (Shell-Resolved Primitive Semantic Interface)

CPU only · official test never loaded · single seed-0 trajectory.

## A. Provenance

- formal-run commit `f836b04f7354a892c62b9e5f8ff682add7340a03`
- analysis / report commit `c841c6d5ad13a862d2e5379f7a2756258976c96a`
- device `cpu` · threads `8`
- `official_test_loaded = false`

## B. Historical references (read-only; never rerun; unmatched)

- CSSD-q1-seed0: `0.130028`
- CSSD-q1-zero-training-selection: n/a
- FEC-S1-sparse-seed0: n/a
- FINAL-CLEAN-C6-seed0: `0.128499`
- P1-sparse-seed0: `0.131975`
- RNDB-seed0: `0.133117`
- T1-tuned-seed0: `0.125765`
- training-protocol-audit-control-lr1e3: `0.127428`
- training-protocol-audit-low-lr1e4: `0.126818`

## C. Exact architecture

```
parent:  anchor62 -> anchor_encoder(62->32->32) -> concat with 3x48 node / 6x32 edge -> fusion(368->128->48)
Sem108:  patch_cont[:, 0:108] = atom_shell(3x28=84) | bond_shell(6x4=24)
candidate: [Sem108 ; size2](110) -> concat with 3x48 node / 6x32 edge -> fusion(446->114->48)
no RNDB, no message passing, no new fine structural descriptor
```

## D. Parameter budget

- parent `97727`, candidate `97709` (delta `-18`), relative `0.00018` · fusion hidden `114`

## E. Zero-training prior audit

- decision `PROCEED`
- T1 block audit: `T1_BLOCK_AUDIT_UNAVAILABLE`

## F. Training

- epochs `320` · wall `1916.8 s` · seed `0`
- best valid `0.131313` @ 309 · Top-5 members [238, 255, 257, 282, 309] · soup `0.123705`

## G. Performance interpretation

- `M_S = 0.123705` → band **SEM108_WITHIN_EXISTING_PERFORMANCE_BAND**
- distance to the pre-registered promising threshold (0.123): `+0.000705`
- `G_hist = 0.006323` vs historical CSSD-q1 `0.130028` — historical, unmatched

## H. Semantic mechanism

- mean-fill Sem108: `M_sem0=1.430745` `G_sem0=1.307040`
- atom block: `G_atom=1.097573` · bond block: `G_bond=0.480401`
- Sem108 row shuffle: `G_sem_shuffle=0.493259` [0.481105, 0.505354] → `strong`

## I. Dictionary mechanism

- zero-alpha diagnostic: `M_dict0=0.190254` `G_dict0=0.066549`
- node correspondence `G_node=0.000000` · edge `G_edge=0.038245` · `G_corr=0.038245` → `clear_incremental`
- health: active 32/32, effective 20.71, top1 share 0.823, recon 0.97696
- binding health (soup): `|W_A_S|max=7.050e-38`, `|W_A_C|max=7.048e-38`, `|W_E_S|max=7.241e-01`, `|W_E_C|max=5.792e-01`
- **Node-binding collapse (observation).** `W_A_S` / `W_A_C` fell to float32 denormals during training, so node slots are identically zero and the node assignment shuffle is a no-op (`G_node = 0.0`). The path is alive at the 8-epoch smoke state and the implementation is bit-identical to the parent at init, so this is a learned redundancy collapse (Sem108 already carries the atom chemistry), not a defect. `G_corr = max(G_node, G_edge)` is therefore driven entirely by the edge correspondence.

## J. Verdict

**Case D — DICTIONARY_INCREMENT_RETAINED_BUT_NO_NEW_TASK_BAND**

> Node-side dictionary correspondence is exactly zero (G_node = 0.0), and the frozen soup state confirms why: W_A_S / W_A_C collapsed to float32 denormals (absmax 7.050e-38 / 7.048e-38) during training, so node slots are identically zero and the node assignment shuffle is a no-op. The path is alive in the 8-epoch smoke state, so this is a learned redundancy collapse (the direct Sem108 interface already carries the atom chemistry), not a coding defect. The pre-registered G_corr = max(G_node, G_edge) therefore reflects the edge correspondence only.

## K. T1 attribution limitation

- `T1_BLOCK_AUDIT_UNAVAILABLE`: exact T1 selected/soup state tensors are not durable (git-ignored and deleted)

