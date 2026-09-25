# E2E-DictEnv-Purify-v0 — implementation

Round `e2e_dictenv_purify_v0` · study `zinc-context-gap`.
Architecture audit: [`e2e_dictenv_purify_v0_architecture_audit.md`](e2e_dictenv_purify_v0_architecture_audit.md).
Pre-registration: [`e2e_dictenv_purify_v0_preregistration.md`](e2e_dictenv_purify_v0_preregistration.md).

## 1. Code layout

```text
tracks/ksvd/experiments/luyin16/e2e_dictenv_purify_v0.py    four-layer model, both arms
tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_purify_v0.py  matched runner (GPU1 only)
tracks/ksvd/tests/test_e2e_dictenv_purify_v0.py             targeted CPU tests
```

One code path serves both arms.  The flavour is a config field
(`PurifyConfig.flavor in {"reference", "purified"}`) and the only places it is
read are the two well-named objects that must differ:

* `AttributedLocalMeasure` — constant channel rows and their routing,
* `StaticComposer.local_anchor` — 62-D anchor vs its 2-D size slice.

There is no `if purified: ... else: ...` scattered through the composition,
pair, global, topology or reader code; those modules are shared verbatim.

## 2. The four layers

| layer | object | responsibility | math |
|---|---|---|---|
| 1 | `StructuralBasis` | structure defines the basis | `Dbar = colnorm(D)`, `alpha = IHT_{s=8,10}(Dbar, phi)`, `hat_phi = alpha Dbar^T`, `L_rec = ||phi-hat_phi||^2/(||phi||^2+eps)` |
| 2 | `AttributedLocalMeasure` | attributes value the basis | `u_v = ((beta_v W_A_S) ⊙ (q_v W_A_C))/sqrt(96)` per shell; `u_uv = ((gamma_uv W_E_S) ⊙ (b_uv W_E_C))/sqrt(48)` per shellpair |
| 3 | `StaticComposer` | static composition builds the molecule | H1 slot decoders + `368->128->48` fusion, pair projection, 15-D pure-topology relation, distance gate, pair encoder, unary/pair pooling, global 62->32->32, topology 25->16->8 |
| 4 | `PropertyReader` | final prediction only | `GenericReader(302, (13,13))` |

Frozen semantics are unchanged: no message passing, no recurrence, no attention,
no pair→centre, no handcrafted shell×chemistry histogram, no dropout/normalization
touched, no pair-chemistry relation.

### 2.1 Global context split into two named objects

```text
GlobalStructuralInvariant          global_context[0:30]    pure topology (short+long)
GlobalZerothOrderAttributeMeasure  global_context[30:62]   whole-graph atom marginal 28 + bond marginal 4
```

Both blocks are kept this round.  Their encoder stays the single fused
`62->32->32` MLP: splitting it would break Stage-0 equivalence and would
confound the single purification principle.  The split is semantic (named
slices, explicit documentation, explicit entries in `information_flow.json`),
not a capacity change.

## 3. Stage 0 — functional equivalence (PASSED)

The refactor must reproduce the legacy implementation exactly.

```text
legacy:   e2e_dictenv_p2_abs.P2Model (H1 config, 97,487 params)
refactor: PurifyV0Model(PurifyConfig(flavor="reference"))
mapping:  e2e_dictenv_purify_v0.load_p2_state / p2_key_to_purify
```

Comparison points and measured deviation, on real cached official-valid
molecules with the **trained H1 soup checkpoint** loaded
(`results/e2e_dictenv_p2_abs/states/H1_soup_state.pt`):

| point | max abs diff |
|---|---:|
| alpha (`coord`) | 0.0 |
| node slots | 0.0 |
| edge slots | 0.0 |
| anchor | 0.0 |
| pair value | 0.0 |
| unary | 0.0 |
| relation readout | 0.0 |
| global out | 0.0 |
| topology out | 0.0 |
| unified | 0.0 |
| environment `E` | 0.0 |
| **prediction** | **0.0** |

Requirement was `<= 1e-6`; the refactor is bit-identical.  On CUDA the same
comparison is limited by `index_add_` atomic reduction order (prediction
4.77e-07, pooled read-outs up to 5.7e-06); amendment A1 makes that explicit and
gates CUDA against the implementation's own measured rerun noise floor while
keeping the deterministic CPU path bit-identical.  Artifact:
`results/e2e_dictenv_purify_v0/semantic_refactor_equivalence.json`
(`bit_identical: true`, `passed: true`).  The deterministic-init comparison is
also bit-identical, and the model initialization is bit-identical to
`e2e_dictenv_p2_abs.build_model` for the H1 config (same RNG stream, same
parameter creation order).

## 4. The purified candidate (the one authorized change)

```text
beta_v    = [1 ; alpha_v]                                   (33)  node structural basis
gamma_uv  = [1 ; g_uv],  g_uv = [au+av; |au-av|; au o av]   (97)  edge structural basis
anchor    = data.anchor[:, 60:62]  (2-D standardized size, reference stats slice)
```

Invariants of the constant channel, enforced by construction:

```text
fixed value 1.0; not a dictionary atom; not part of alpha (alpha stays R^32);
not reconstructed; not counted toward the top-8 sparsity constraint;
no parameters of its own (it is a column of ones built in the forward).
```

The two constant rows (`W_A_S[0]`, `W_E_S[0]`) are the only candidate-only
parameters; they are drawn from a separate deterministic generator at the
standard `kaiming_uniform_(a=sqrt(5))` scale, and never displace a shared
tensor.

### 4.1 Parameter ledger (exact, verified against the built models)

| module | reference | purified | delta |
|---|---:|---:|---:|
| `StructuralBasis.dictionary` | 2,080 | 2,080 | 0 |
| `AttributedLocalMeasure.node_binding` | 5,760 | 5,856 | +96 |
| `AttributedLocalMeasure.edge_binding` | 4,800 | 4,848 | +48 |
| `StaticComposer.node_slot_decoder` | 9,328 | 9,328 | 0 |
| `StaticComposer.edge_slot_decoder` | 3,920 | 3,920 | 0 |
| `StaticComposer.anchor_decoder` | 3,072 | 1,152 | −1,920 |
| `StaticComposer.fusion_decoder` | 53,424 | 53,424 | 0 |
| `StaticComposer.pair_projection` | 768 | 768 | 0 |
| `StaticComposer.relation_encoder` | 1,104 | 1,104 | 0 |
| `StaticComposer.distance_gate` | 80 | 80 | 0 |
| `StaticComposer.pair_encoder` | 5,328 | 5,328 | 0 |
| `StaticComposer.global_encoder` | 3,136 | 3,136 | 0 |
| `StaticComposer.topology_encoder` | 552 | 552 | 0 |
| `PropertyReader.reader` | 4,135 | 4,135 | 0 |
| **total** | **97,487** | **95,711** | **−1,776 (−1.822 %)** |

Layer totals: `StructuralBasis` 2,080; `AttributedLocalMeasure`
10,560 → 10,704; `StaticComposer` 80,712 → 78,792; `PropertyReader` 4,135.
The parameter rule (`purified <= reference`) holds with margin.

## 5. Matched initialization

`build_matched_pair` builds the reference under `torch.manual_seed(seed)`, then
copies **every same-shape shared tensor** into the candidate, and finally

* gives the candidate anchor encoder the reference weights of the two size
  coordinates (`W[:, 60:62]`, reference bias),
* draws the two constant rows deterministically from separate generators.

Verified: `max_abs_diff == 0.0` over all 48 shared tensors, the dictionary and
reader are shared verbatim, and repeated constructions give bit-identical
constant rows.  Reported by `shared_init_report()` and asserted by the targeted
tests.

## 6. Runner protocol

```text
stages: audit equiv smoke train mechanism health ablate paired decision report all
```

* `resolve_device("cuda")` returns `cuda:1` without a mask and `cuda:0` only when
  `CUDA_VISIBLE_DEVICES=1`; `CUDA_VISIBLE_DEVICES=0` and multi-GPU masks raise.
  There is no GPU0 fallback and no DDP path.
* `load_split` raises `PermissionError` for `test`; every artifact records
  `official_test_loaded: false`.
* Training: 320 epochs, Adam lr 1e-3, wd 1e-5, batch 128, clip 5, seed 0/1,
  `lambda_rec = 33.95873017865987`, fixed Top-5 soup on official valid,
  `official train 10000 / official valid 1000`.
* Interventions are driven by a single `Interventions` object; shuffle indices
  travel on the graphs (`env_occ_coord_node`, `env_bond_u_shuffled`,
  `env_bond_v_shuffled`) so the shared collate applies the per-graph offsets.
* Every stage is resumable: an existing artifact short-circuits the stage.

## 7. Targeted tests

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_purify_v0.py
```

24 tests, all passing locally, covering the frozen pre-registration list:
semantic-refactor equivalence (synthetic + real checkpoint), constant channel
not part of alpha, not counted toward sparsity, `max ||alpha||_0 <= 8`,
dictionary reconstruction definition, chemistry relabel invariance, node
permutation / graph relabel invariance, edge endpoint symmetry, local anchor
chemistry absence, size-only structural scalar slot, global chemistry and
global topology untouched, pair relation untouched, reader untouched, matched
init identity + deterministic constant rows, parameter ledger, official-test
blocker, GPU device explicitness, intervention semantics, and non-zero
gradients to `D`, the atom/bond valuation matrices and both constant rows.

## 8. Reproduction

```bash
# local (CPU)
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 audit
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 equiv
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_purify_v0.py

# remote (GPU1 only)
bash scripts/launch_remote.sh 1 purify-all python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 all --device cuda
```
