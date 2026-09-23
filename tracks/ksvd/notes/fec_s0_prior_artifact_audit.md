# FEC-S0 — prior-artifact audit / Stage-0 exact S0 access audit

Round **FEC-S0** · study `zinc-context-gap` · protocol `fec_s0`.
Pre-registration: [`fec_s0_preregistration.md`](fec_s0_preregistration.md).
Written after code tracing + a runtime reachability trace, before any
equivalence verdict.

Lineage HEAD: `d4de88e` (clean worktree).  Official test **never** loaded.

This note modifies no historical record.  PEC-I1's frozen verdict stays
`STATIC_POOLING_NOT_PRIMARY_GAP`.

---

## 0. Why a fresh access audit

The brief forbids reusing the B-Full / compact-v4 path table: the strict-static
S0 (round `ZINC-strict-static-dictionary-pair-v0`) has a *different* reachable
path set because it forces `center_context=False` and uses the `hit`-free
small raw reader.  The table below is obtained by tracing the **exact** S0
builder.

## 1. Exact S0 artifacts (identity / provenance only)

| item | value |
|---|---|
| model builder | `tracks/ksvd/experiments/luyin16/zinc_static_dictionary_pair.py::build_s0(0)` |
| class | `StrictStaticPairModel(residual_mode="none")` subclass of `zpp.PatchPathModel` |
| base kwargs | `zinc_static_dictionary_pair.base_model_kwargs()` reading `configs/luyin16/zinc_compact_v4_topology_hinge.yaml` (`center_context` forced `False`) |
| forward | `zinc_patch_path_pooling.PatchPathModel.encode` / `.forward` (no override) |
| checkpoint | `results/zinc_static_dictionary_pair/states/s0_seed0_selection_state.pt` (sha256 `8d81f129…8fcf03`) |
| encoded inputs | `results/zinc_static_dictionary_pair/cache/encoded_{train,valid}.pt` |
| preprocessing | `zpp._phase_data` → `_encode_records` (train-only vocab + standardizers) |
| collate | `zpp._make_loader` (batch 128, `train_shuffle_seed_offset=91011`, `eval_shuffle_seed_offset=91012`) |
| reader | `GenericReader(302, (13, 13))` |
| params / runtime width | `66,228` / `302D` |
| recorded best valid | `0.14567435123870381` @ epoch 164 |
| recorded Top-5 soup | `0.140794` (members `[125,142,159,164,167]`) |
| recorded run | `results/zinc_static_dictionary_pair/runs/s0_seed0.json` |
| raw data | PyG ZINC `subset=True` official train 10 000 / valid 1 000 |

The soup **member states were never persisted** (only the best selection state
was saved).  This round therefore runs the full equivalence audit on the
available best-checkpoint state and treats the soup value as recorded
provenance only.

Checkpoint load is exact: `load_state_dict` reports no missing and no
unexpected keys.  On CPU the best-checkpoint valid MAE recomputes to
`0.1456743378872634` (recorded `0.14567435123870381`; `|Δ| = 1.34e-8`), and
recomputed predictions differ from the recorded GPU predictions by at most
`2.86e-6` (expected float32 device reduction-order difference).

## 2. Active parameter blocks (measured)

| module | params |
|---|---:|
| `typed_embedding` (`_HybridEmbedding`, 6785 rows: 768 full + 6017 rank-4) | 36 420 |
| `parent_embedding` (32 rows, rank-4 hybrid) | 256 |
| `patch_encoder` (`ResidualPatchEncoder`, base `_MLPBlock(170→64→48)` + `gamma`) | 14 193 |
| `pair_projection` (`Linear(48→16, bias=False)`) | 768 |
| `relation_encoder` (`_MLPBlock(23→32→16)`) | 1 360 |
| `distance_gate` (`Embedding(5,16)`) | 80 |
| `pair_encoder` (`_MLPBlock(64→64→16)`) | 5 328 |
| `global_encoder` (`_MLPBlock(62→32→32)`) | 3 136 |
| `topology_encoder` (`Linear(25→16)→ReLU→Linear(16→8)`) | 552 |
| `head` (`GenericReader(302→13→13→1)`) | 4 135 |
| **total** | **66 228** |

`center_update` is `None`; `structural_encoder`, `attribute_encoder`,
`context_patch_projection`, `structural_context_embedding` are all `None`.

## 3. Exact reachable-path table

Runtime trace confirms: one call each of `patch_encoder`, `pair_encoder`,
`relation_encoder` per forward; no `_pool_pairs_to_centres`.

| path | width | raw sources | chemistry? | topology? | mixed? | reaches prediction? | FEC classification |
|---|---:|---|---|---|---|---|---|
| `patch_cont` | 146 | `_shell_descriptor`: atom_shell (3×28) + bond_shell (6×4) + root_atom (28) + incident (4) + scalars (6) | yes (4 of 5 blocks) | yes (roles + scalars) | yes | **yes** | `FACTORIZED_SHARED` |
| `typed_token` | 16 | `pynauty` certificate (v1 historical) of the rooted typed incidence graph | yes (joint atom+bond) | yes (rooted incidence) | yes | **yes** | `EXPLICIT_BINDING_LOOKUP` (non-injective alias) |
| `parent_token` | 8 | same certificate at radius 1 | yes | yes | yes | **yes** | `EXPLICIT_BINDING_LOOKUP` |
| `pair_relation` | 23 | untyped shortest-path topology (15) + path bond composition (4) + adjacent bond one-hot (4) | yes (8) | yes (15) | yes (explicit concat) | **yes** | `FACTORIZED_SHARED` |
| `pair_bucket` | 1 | shortest-path distance bucket | no | yes | no | **yes** | `PURE_TOPOLOGY` |
| `global_context` | 62 | 30-D pure topology + 28-D atom marginal + 4-D bond marginal | yes (marginals) | yes | yes (explicit concat) | **yes** | `FACTORIZED_SHARED` |
| `topology_features` | 25 | exact simple-cycle spectrum + hinge basis of the untyped graph | no | yes | no | **yes** | `PURE_TOPOLOGY` |
| `pair_index` | 2×E | unordered centre pairs | no | no | no | yes (routing) | `PURE_TOPOLOGY` |

## 4. Dormant / not-reachable paths (code-proven)

| path | reason |
|---|---|
| `patch_context` | `context_radius=0` → width 0 |
| `structural_token`, `structural_coarse` | `structural_context_mode="none"` |
| `attribute_*` | `attribute_mode="none"` |
| `center_update` | `center_context=False` → `center_update is None` |
| `direct_token_readout` | disabled |
| `_pool_pairs_to_centres` | never called (monkeypatch-to-raise test passes) |

## 5. The single blocking-risk path

The historical alias tokenizer
(`typed_patch_tokenizer.DEFAULT_TYPED_TOKENIZER_VERSION =
"typed_tokenizer_v1_historical"`) is `bytes(pynauty.certificate(incidence))`.
Per the tokenizer module's own docstring, this certificate is the canonical
adjacency matrix and **does not encode the vertex coloring**, so it is a
strictly coarser invariant than the rooted typed patch: distinct rooted typed
bindings can share one token id.  It is therefore an *aliased,
non-injective* categorical key, and the `typed_embedding` table owns
independent parameters for each key (36 420 params).  This is the one path
that is **reconstructible but not a shared role × primitive factorization**;
the equivalence analysis quantifies its aliasing on the audit split.

## 6. Stage-0 conclusion

```
reachable predictive paths = { patch_cont(146), typed_token, parent_token,
                               pair_relation(23), pair_bucket, global_context(62),
                               topology_features(25) }
no message passing / recurrence / pair->centre / context writeback
one opaque-token candidate: the aliased historical typed/parent certificate
```

Proceed to Stage A/B/C factorization and the numerical equivalence audit.
