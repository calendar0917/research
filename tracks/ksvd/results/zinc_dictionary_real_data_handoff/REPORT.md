# ZINC dictionary real-data handoff (frozen Full reader)

**Scope.** Read-only export of a *small real ZINC* package for a downstream
agent to run lightweight candidate checks against the frozen Full soup reader.
No training, no refit, no HPO, no seed1, no official test, no push of the
scientific record.

## Provenance

* Repo revision: `e651d5ad0c71d9e87554194ab508f836faf126c8`
* Parent checkpoint: `tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`
  (`sha256 17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`),
  identical to this round's `R` cache.
* Split fingerprint: `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`
* Official order rule: row `i` of every per-split array is official split
  position `i` (PyG ZINC subset processed order); `ids = 0..N-1`.
* `official_test_loaded = false`; `frozen_backbone = true`; no training performed.

## Deliverables (`tracks/ksvd/results/zinc_dictionary_real_data_handoff/`)

Large `*.npz` / `*.zip` are git-ignored local evidence; `manifest.json` and
`accept_summary.json` are also local (git-ignored) evidence.

| file | bytes | notes |
| --- | --- | --- |
| `package_zinc_handoff.zip` | 22,157,951 | combined single package (< 31 MiB) |
| `package_A_reader_R.zip` | 21,647,957 | reader + train/valid `R` |
| `package_B_graphs_topology.zip` | 21,713,973 | graphs + topology + predictions |
| `train.npz` | 19,760,552 | full per-split bundle |
| `valid.npz` | 1,976,447 | full per-split bundle |
| `reader.npz` | 440,797 | reader `814->39->39->1` + topology encoder + `D_L`/`V_L` |
| `manifest.json` | — | revisions, SHAs, shapes/dtypes, caveats |
| `accept_summary.json` | — | acceptance gate results |

`R` is stored `float32` (verified lossless against the `float64` cache). No
pickle/object arrays anywhere. `canonical_group_id` (verified semantic-complete
key `pynauty_certificate+coloured_incidence_sequence`) is present on **train
only**; no verified canonical cache exists for official valid.

## Acceptance (all passed)

| check | result |
| --- | --- |
| `N_train` / `N_valid` / `R` width | 10000 / 1000 / 814 |
| checkpoint sha matches this round's cache | true |
| independent reader replay max abs diff | train `5.14e-06`, valid `9.43e-07` (<= 1e-5) |
| valid MAE vs `0.1191540920053958` | diff `0.0` (<= 1e-6) |
| valid id 172 | `y=-20.340498`, `p_base=-1.388554` |
| train min label | `-42.036564` |
| `topo_model_input` vs `R[:, -8:]` (encoder replay) | train `7.30e-06`, valid `1.73e-06` |
| graph arrays | `u<v`, all endpoints in-graph, bond types consistent, no self-loops |
| y / topo order alignment | exact (`0.0` vs encoded and V4 record caches) |
| source checkpoint / caches before vs after | all SHA-256 unchanged |
| wall clock | 14.3 s |

## Caveats

* The Full backbone has already seen all official train labels; a train head
  split is **not** an end-to-end OOF estimate.
* The official valid split has been reused repeatedly and is only an
  exploratory screen.
* `topo_raw` (unstandardized `raw_vector('hinge')`) and `topo_model_input`
  (standardized tensor fed to `topology_encoder`) are deliberately separate.
* No known Full-model OOF resource was identified (not scanned).

Exporter: `tracks/ksvd/experiments/luyin16/zinc_dictionary_real_data_handoff.py`.
