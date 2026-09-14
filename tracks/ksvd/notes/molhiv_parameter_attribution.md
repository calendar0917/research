# MolHIV current-model parameter attribution + vocabulary audit (zero training)

Date: 2026-09-14
Protocol: `molhiv_parameter_attribution_v1`
Code commit: `4f1ccde`
Model audited: the frozen `molhiv_recurrent_pair_centre_v1` port (H96, T=2)
Official MolHIV test: opened once previously in the frozen transfer run; used
here **read-only** for vocabulary diagnostics. **No training, no architecture
search, no tokenizer change, no cutoff decision, no embedding compression.**

## B.1 Exact total and category breakdown (`named_parameters()`, seed-independent)

Model vocabulary is the actual training-time fit on official train only:
typed `26,232` (+1 OOV) x 32, parent `102` (+1 OOV) x 16.

```text
total params (real)              = 1,076,589
sum(named_parameters)            = 1,076,589   (exact match)
sum(category_breakdown)          = 1,076,589   (exact match)
other_uncategorized              = 0
```

| category | params | % total | tensors |
|---|---:|---:|---|
| identity_storage.certificate_patch_token | 839,456 | 77.97% | `typed_embedding.weight` [26233, 32] |
| patch_encoder | 90,336 | 8.39% | `patch_encoder.layers.{0,1,4}.{weight,bias}` |
| readout_projection | 81,408 | 7.56% | `head.0.{weight,bias}` (192 x 423) |
| center_update | 23,676 | 2.20% | `center_update.{0,1,4}.{weight,bias}` |
| prediction_head.head_hidden | 18,528 | 1.72% | `head.4.{weight,bias}` |
| global_encoder | 12,128 | 1.13% | `global_encoder.layers.{0,1,4}.*` |
| pair_encoder | 5,328 | 0.49% | `pair_encoder.layers.{0,1,4}.*` |
| relation_related_modules.relation_encoder | 1,968 | 0.18% | `relation_encoder.layers.{0,1,4}.*` |
| identity_storage.radius1_parent_certificate | 1,648 | 0.15% | `parent_embedding.weight` [103, 16] |
| pair_projection | 1,536 | 0.14% | `pair_projection.weight` [16, 96] |
| prediction_head.head_layer_norm | 384 | 0.04% | `head.1.{weight,bias}` |
| prediction_head.head_output | 97 | 0.01% | `head.6.{weight,bias}` |
| learned_embedding_tables.fixed.distance_bucket | 96 | 0.01% | `distance_gate.weight` [6, 16] |

Requested categories with **no dedicated module** in this architecture:

* **atom / node encoding**: no atom embedding table; atom identity is a
  continuous one-hot shell histogram inside `patch_cont` and inside the
  `global_context` atom histogram (consumed by the encoders' first layers).
* **bond / edge encoding**: no bond embedding table; bond identity is a
  continuous one-hot shell histogram inside `patch_cont`, `global_context`, and
  the pair-relation descriptor.
* **topology encoder**: absent; the topology branch exists only in the ZINC
  hinge variants.

Secondary feature-block attribution of the first encoder layers (must **not** be
added to the primary table):

| patch_encoder.layers.0 input block | params |
|---|---:|
| atom/node shell columns | 66,816 |
| bond/edge shell columns | 8,736 |
| shell scalar columns | 576 |
| typed-token columns | 3,072 |
| parent-token columns | 1,536 |
| layer bias + second layer | 9,600 |
| **patch_encoder total** | **90,336** |

| global_encoder.layers.0 input block | params |
|---|---:|
| global short columns | 720 |
| global long columns | 720 |
| atom histogram columns | 8,352 |
| bond histogram columns | 624 |
| layer bias + second layer | 1,712 |
| **global_encoder total** | **12,128** |

## B.2 Largest 20 tensors

| # | name | shape | numel | % total |
|---:|---|---|---:|---:|
| 1 | `typed_embedding.weight` | [26233, 32] | 839,456 | 77.97% |
| 2 | `head.0.weight` | [192, 423] | 81,216 | 7.54% |
| 3 | `patch_encoder.layers.0.weight` | [96, 841] | 80,736 | 7.50% |
| 4 | `head.4.weight` | [96, 192] | 18,432 | 1.71% |
| 5 | `center_update.0.weight` | [60, 294] | 17,640 | 1.64% |
| 6 | `global_encoder.layers.0.weight` | [48, 217] | 10,416 | 0.97% |
| 7 | `patch_encoder.layers.4.weight` | [96, 96] | 9,216 | 0.86% |
| 8 | `center_update.4.weight` | [96, 60] | 5,760 | 0.54% |
| 9 | `pair_encoder.layers.0.weight` | [64, 64] | 4,096 | 0.38% |
| 10 | `parent_embedding.weight` | [103, 16] | 1,648 | 0.15% |
| 11 | `global_encoder.layers.4.weight` | [32, 48] | 1,536 | 0.14% |
| 12 | `pair_projection.weight` | [16, 96] | 1,536 | 0.14% |
| 13 | `relation_encoder.layers.0.weight` | [32, 42] | 1,344 | 0.12% |
| 14 | `pair_encoder.layers.4.weight` | [16, 64] | 1,024 | 0.10% |
| 15 | `relation_encoder.layers.4.weight` | [16, 32] | 512 | 0.05% |
| 16 | `head.1.weight` | [192] | 192 | 0.02% |
| 17 | `head.0.bias` | [192] | 192 | 0.02% |
| 18 | `head.1.bias` | [192] | 192 | 0.02% |
| 19 | `head.4.bias` | [96] | 96 | 0.01% |
| 20 | `center_update.0.bias` | [60] | 60 | 0.01% |

The only vocabulary-sized lookups are `[num_tokens, embedding_dim]`:
`typed_embedding.weight` [26233, 32] and `parent_embedding.weight` [103, 16].
There are no `[num_certificates, d]` / `[num_patches, d]` tables beyond these
(no separate patch-identity table). The single largest tensor is
`typed_embedding.weight`, at 77.97% of the model.

## B.3 Fixed-size vs dataset-dependent parameters

| class | params | fraction |
|---|---:|---:|
| **dataset-dependent** (identity storage: typed + parent lookups) | **841,104** | **78.13%** |
| fixed-size (everything else) | 235,485 | 21.87% |

The near-1M total is **predominantly identity storage**, not backbone capacity.
Dense `nn.Embedding` is used for both vocabularies (no frequency-adaptive
factorization).

## B.4 Vocabulary accounting (real pipeline, official train fit)

Pipeline: `molhiv_patch_path_pooling` exact-rooted record cache (no approximate
re-tokenizer). Train molecules: **32,901**; patch occurrences: **830,936**.

### Train identities

| | typed certificate | radius-1 parent certificate |
|---|---:|---:|
| unique train types | 26,232 | 102 |
| embedding dim | 32 | 16 |
| identity params (with OOV) | 839,456 | 1,648 |
| singleton types | 10,309 (39.3% of types, 1.24% of occurrences) | 24 |
| rare <= 2 types | 15,046 (57.4% of types, 2.38% of occurrences) | 30 |
| rare <= 5 types | 19,547 (74.5% of types, 4.41% of occurrences) | — |
| max frequency | 53,077 | 341,824 |
| median frequency | 2 | 13 |
| coverage of top-50 / 100 / 250 / 500 / 1000 | 52.7% / 60.3% / 71.0% / 77.8% / **83.8%** | ~100% |

### Official valid / test coverage by the train vocabulary

| split | field | OOV occurrence frac | known type frac | molecules with any OOV | per-molecule OOV mean | p90 |
|---|---|---:|---:|---:|---:|---:|
| valid | typed | 7.37% | 54.25% | 59.18% | 8.31% | 23.08% |
| test | typed | 6.54% | 56.25% | 55.36% | 7.88% | 22.22% |
| valid | parent | 0.042% | 72.32% | 0.90% | 0.042% | 0 |
| test | parent | 0.031% | 80.49% | 0.29% | 0.038% | 0 |

**Valid vs test distribution shift:** none material. The typed OOV occurrence
difference is 0.83 percentage points and the per-molecule OOV mean difference is
0.43 points; parent is near-zero for both. Roughly **55-59% of valid and test
molecules contain at least one unseen typed certificate**, and ~6.5-7.4% of all
valid/test patch occurrences are OOV. So valid and test are drawn from very
similar vocabulary-coverage conditions; the valid→test AUC gap is **not**
explained by a valid/test vocabulary shift.

## B.5 Parameter growth law + pure top-K accounting

```text
identity_params(V_typed, V_parent) =
    (V_typed + 1) * token_width + (V_parent + 1) * parent_width
    = (V_typed + 1) * 32 + (V_parent + 1) * 16
```

Current: `V_typed = 26,232`, `V_parent = 102` ->
typed 839,456 + parent 1,648 = **841,104** identity params; fixed backbone
235,485; total **1,076,589**.

Pure accounting only (no model built, no training, **cutoffs neither recommended
nor ranked**, not chosen on valid/test performance). Parent vocabulary held
fixed:

| top-K typed kept | identity params | theoretical total | saved vs current |
|---:|---:|---:|---:|
| 50 | 3,280 | **238,765** | 837,824 |
| 100 | 4,880 | **240,365** | 836,224 |
| 250 | 9,680 | **245,165** | 831,424 |
| 500 | 17,680 | **253,165** | 823,424 |
| 1000 | 33,680 | **269,165** | 807,424 |

The parameter count collapses from ~1.08M to ~0.24-0.27M purely on identity
accounting; the fixed backbone (235,485) becomes the floor.

## B.6 ZINC vs MolHIV parameter structure

Built from the frozen ZINC builders (`cd.build_cell`); identity = typed +
parent certificate embeddings; everything else is fixed.

| model | total | identity storage | identity % | fixed backbone | fixed % |
|---|---:|---:|---:|---:|---:|
| ZINC cell A (85,763) | 85,763 | 36,676 | 42.8% | 49,087 | 57.2% |
| ZINC cell D / H96 (103,219) | 103,219 | 36,676 | 35.5% | 66,543 | 64.5% |
| **MolHIV H96 port** | **1,076,589** | **841,104** | **78.1%** | **235,485** | **21.9%** |

ZINC lookup detail: typed `_HybridEmbedding` (frequent dense + rare low-rank)
for 6,784 types at output width 16 -> 36,420; parent hybrid width 8 -> 256.
MolHIV uses a plain dense `nn.Embedding` for 26,232 types at width 32 -> 839,456.

**Why MolHIV is ~1M while ZINC is ~86k/103k.** Both effects contribute:

1. **Identity storage (dominant, ~805k of the gap):** ~3.9x more typed
   identities (26,232 vs 6,784), **2x the embedding width** (32 vs 16), and a
   **dense** table instead of ZINC's frequency-adaptive hybrid
   (full + low-rank). Dense-x2-width-x3.9-types alone is ~7.7x; the missing
   factorization multiplies that further.
2. **Fixed backbone (~186k of the gap, ~3.5-4.8x larger):** larger input
   descriptors (shell width 793 vs 146; global width 217), token width 32 vs 16,
   and the H96/D-sized centre; ZINC additionally offsets with a small topology
   encoder that MolHIV does not have.

So the primary cause is **dataset-dependent identity storage**, not a hidden
extra backbone; the backbone is also larger, but identity storage alone is
78.1% of the model.

## B.7 Optional zero-training rarity x prediction diagnostic

Read-only forward pass of the two frozen raw checkpoints on the already-opened
valid/test splits (no training). 2-seed ensemble test AUC reproduced exactly
(`0.7800478959` vs the recorded `0.7800478959`). Subgroup AUCs are descriptive,
not causal, and did not inform any architecture choice.

### Test (4,113 molecules, 130 positives, overall AUC 0.780)

| bucket | n | positives | AUC |
|---|---:|---:|---:|
| per-molecule OOV fraction == 0 | 1,836 | 80 | **0.837** |
| OOV fraction (0, 0.1] | 1,059 | 29 | 0.761 |
| OOV fraction (0.1, 0.3] | 1,027 | 18 | **0.654** |
| OOV fraction > 0.3 | 191 | 3 | skipped (<10 positives) |
| min train frequency == 0 (any OOV) | 2,277 | 50 | **0.691** |
| min train frequency in (1, 5] | 502 | 13 | 0.670 |
| min train frequency > 5 | 1,067 | 61 | **0.894** |
| mean train frequency tercile: low | 1,371 | 50 | 0.869 |
| mean train frequency tercile: mid | 1,371 | 59 | 0.769 |
| mean train frequency tercile: high | 1,371 | 21 | **0.579** |

### Valid (4,113 molecules, 81 positives, overall 2-seed ensemble AUC 0.854)

| bucket | n | positives | AUC |
|---|---:|---:|---:|
| OOV fraction == 0 | 1,679 | 39 | 0.935 |
| OOV fraction (0, 0.1] | 1,141 | 26 | 0.848 |
| OOV fraction (0.1, 0.3] | 1,094 | 15 | 0.675 |
| min train frequency == 0 | 2,434 | 42 | 0.790 |
| min train frequency > 5 | 1,002 | 28 | 0.951 |

### Reading

* **OOV presence is directionally associated with worse ranking:** on test,
  fully-covered molecules rank at AUC 0.837 versus 0.691 for molecules with any
  unseen certificate, and molecules with 10-30% OOV patches fall to 0.654.
  Subgroup positive counts are adequate for the main OOV-zero / OOV-present
  split, and the pattern is not driven by a valid/test vocabulary shift (B.4).
* **But the continuous mean-frequency tercile reverses** (high-familiarity
  tercile AUC 0.579 vs low 0.869). That tercile is heavily confounded
  (high-familiarity molecules have only 21/1371 = 1.5% positive rate versus
  3.6-4.3% for the other terciles), so it is not clean evidence of a monotone
  rarity effect.
* Conclusion: the evidence for an "identity memorization / scaffold
  generalization failure" mechanism is **partial and confounded, not causal**.
  It is a hypothesis to test with a dedicated diagnostic, not a basis for
  choosing a compact architecture.

## Candidate compact design directions (for later discussion only)

Not implemented, not ranked, not selected:

1. **Frequency-capped identity vocabulary** — keep the top-K typed certificates
   plus an OOV row (e.g. combine with frequency-aware OOV handling). Pure
   accounting shows this alone would bring the total to ~0.24-0.27M.
2. **Hashed / shared low-rank identity embeddings** — mirror ZINC's
   frequency-adaptive hybrid (dense for frequent identities, low-rank /
   factorized or hashed for rare ones) instead of a dense `[26233, 32]` table.
3. **Compositional identity encoder** — replace per-certificate exact rows with
   an encoder over the certificate's structural content (root atom + orbit/role
   descriptors) so rare / unseen identities are composed rather than memorized.

The next decision (which direction, if any) is deferred until this attribution is
consumed.

## Files

- `experiments/luyin16/molhiv_parameter_attribution.py`
- `tests/ksvd/tests/test_molhiv_parameter_attribution.py` (9 pass)
- `results/molhiv_parameter_attribution/`:
  `parameter_breakdown.json`, `vocabulary_audit.json`, `growth_law.json`,
  `zinc_comparison.json`, `rarity_prediction_diagnostic.json`, `report.json`
