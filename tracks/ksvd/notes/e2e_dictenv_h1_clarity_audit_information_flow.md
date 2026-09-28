# E2E-DictEnv H1 — Phase A information-flow inventory

Round: `e2e_dictenv_h1_clarity_audit` (CPU-only information-flow / mechanism audit of the
H1 / P2-ABS main line on ZINC).
Preregistration: [`e2e_dictenv_h1_clarity_audit_preregistration.md`](./e2e_dictenv_h1_clarity_audit_preregistration.md).
Machine-readable inventory: `results/e2e_dictenv_h1_clarity_audit/information_inventory.json`
(produced by `zinc_e2e_dictenv_h1_clarity_audit.py inventory`, no training, no GPU).

This note records **where every coordinate that reaches the reader actually comes from**.
Every claim below is a check on the generating code, not an inference from behaviour; the
`provenance_checks` block of the JSON re-derives each slice from the producer functions on
64 real molecules and confirms it is affine-identical to the tensor the forward pass uses.

## 1. Reader input composition (302 D)

Verified against `e2e_dictenv_v0.py` line 523 (`torch.cat([unary, relation_readout, graph_hidden, topology], dim=1)`) and the
`self.reader = GenericReader(97 + 5*(2*16+1) + 32 + 8)` construction at line 418:

```
reader input = 97 + 165 + 32 + 8 = 302
             = unary pool of the local dictionary env (97)
             + distance-bucketed pair readout (165)
             + global encoder output (32)
             + topology encoder output (8)
```

| block | dim | producer | reaches the reader |
|---|---|---|---|
| `unary` | 97 = 2·48+1 | `pool_moments(E)`: `[mean(E), mean(E²), log1p(count)]`; `E` = `env_mlp(cat[m_v, m_e, env_scalars])` (86→329→48, SiLU), and `m_v`/`m_e` are the trilinear contractions of the node slots `(alpha_v·W_A^S)*(q_v·W_A^C)` / edge slots with `W_R` (which consumes the anchor coordinates) | **dictionary-mediated** (α is the frozen-dictionary IHT code, `N6`; the α↔atom slot is `N1`) |
| `relation_readout` | 165 = 5·(2·16+1) | `pool_pair_moments(pair_value)`: for each of the 5 distance buckets, `[mean, mean², log1p(count)]` of the encoded pair value | bypass (with a hard-wired distance partition) |
| `graph_hidden` | 32 | `global_encoder(global_context[62])` | bypass |
| `topology` | 8 | `topology_encoder(topology_features[25])` | bypass |

`unary` is the only dictionary-mediated channel. Everything else reaches the reader directly
(`is_handcrafted_statistic = 21`, `bypasses_dictionary = 19`,
`bypasses_local_structure_semantic_binding = 19` out of 42 inventoried entries). Note the
second-order structure: `relation_readout` is *bucketed by distance*, so the reader sees the
pair statistics already partitioned by the distance one-hot — the relation block never has to
be the only distance carrier.

## 2. `global_context` is 62 D, not 60 D

Verified slice layout (`zinc_long_range_proxy.global_feature_views`):

| group | slice | dim | content |
|---|---|---|---|
| `structure_short` | `[0, 15)` | 15 | short-range rooted topology summaries |
| `structure_long` | `[15, 30)` | 15 | long-range / global topology summaries |
| `atom_histogram` | `[30, 58)` | 28 | graph-level atom-type histogram (chemistry marginal) |
| `bond_histogram` | `[58, 62)` | 4 | graph-level bond-type histogram (chemistry marginal) |

So `global_context = concat([structure_short(15), structure_long(15), atom_hist(28), bond_hist(4)])`
and the **graph-level chemistry marginal is 32 of the 62 coordinates**. 17 of the 62
coordinates are constant on the audit sample (e.g. hydrogens never appear), which is why the
provenance check reports `min_abs_corr_informative` over the non-constant coordinates;
all four groups reconstruct with `min_abs_corr = 1.000000` (worst deviation 3e-15).

## 3. `anchor` is 62 D and always present

| group | slice | dim |
|---|---|---|
| `root` (root atom identity one-hot) | `[0, 28)` | 28 |
| `atom_mass` (patch atom-type histogram) | `[28, 56)` | 28 |
| `bond_mass` (patch bond-type histogram) | `[56, 60)` | 4 |
| `size` (patch size scalars) | `[60, 62)` | 2 |

`atom_mass + bond_mass` are the **patch-level chemistry marginals**; `root` is a graph-context
chemistry identity; `size` is a pure size statistic. The anchor is concatenated into the
reader input unconditionally — it is not gated by the dictionary.

## 4. `pair_relation` is built as 23 D and consumed as 15 D

The producer emits a 23-D block; the forward pass keeps
`raw[0:14] + raw[18:19]` (indices `0,1,…,13,18`) and discards the rest. Verified layout of the
**used** 15 D:

| group | slice in used vector | dim | content |
|---|---|---|---|
| `distance` | `[0, 6)` | 6 | distance one-hot bucket + distance scalars (`bucket`, `log1p(dist)`, …) |
| `overlap` | `[6, 11)` | 5 | patch-overlap / intersection summaries |
| `boundary` | `[11, 14)` | 3 | boundary-overlap indicators + centre containment |
| `path_count` | `[14, 15)` | 1 | `log1p` of the number of simple paths (the coordinate that was at raw index 18) |

Per-coordinate checks all pass: distance one-hot rows sum to 1, the bucket equals the argmax
one-hot, `log_distance` matches the bucket, boundary indicators are binary, and the adjacent
bond-type one-hot agrees with the graph. **The discarded raw coordinates (`14..17`) never reach
the model**, so any intervention on them is a no-op by construction.

## 5. Pooled readout blocks

The pair readout is **nested**: each of the 5 distance buckets contributes its own
`[mean, mean², log1p(count)]` triple, and every intervention below is applied *inside every
bucket* rather than only to the concatenation.

| pool | within-bucket slice | dim |
|---|---|---|
| `unary.first` | `[0, 48)` | 48 |
| `unary.second` | `[48, 96)` | 48 |
| `unary.count` | `[96, 97)` | 1 |
| `pair.first` | `[0, 16)` of each 33-wide bucket block | 16 × 5 |
| `pair.second` | `[16, 32)` of each bucket block | 16 × 5 |
| `pair.count` | `[32, 33)` of each bucket block | 1 × 5 |

`first` = mean pooling of the per-node / per-pair vectors, `second` = mean of squared
vectors, `count` = `log1p` of the bucket member count. The readout mirrors v0
exactly, which is what lets the masked model be bit-identical to the parent at `mask=None`
(verified in the baseline replay).

## 6. Bypass classification used by Phase B

An entry is a **dictionary bypass** if it reaches the reader without passing through the
frozen dictionary code, and a **binding bypass** if it does not require the α↔atom / role↔bond
correspondence. The 19 non-dictionary bypasses are:

`anchor_root_identity`, `anchor_patch_atom_mass`, `anchor_patch_bond_mass`, `anchor_size`,
`pair_relation_raw`, `pair_relation_{distance,overlap,boundary,path_count}`, `pair_bucket`,
`global_{structure_short,structure_long,atom_histogram,bond_histogram}`, `topology_features_25`,
`unary_count`, `pair_count`, `graph_hidden`, `topology_hidden`.

Phase B measures each of these; `information_flow` verdicts are in
`results/e2e_dictenv_h1_clarity_audit/REPORT.md` (§2–§5).

## 7. Probe methodology (important)

Phase B uses four probe families, because the obvious one is wrong:

1. `zero` — set the block to 0 (external input blocks) or replace the pooled value by 0
   (internal masks). Kept for comparability, but it is an **out-of-distribution input
   corruption**, not clean information removal.
2. `fill` — replace external input blocks by their per-coordinate training-set mean
   (distribution-matched first moment). 27 rows.
3. row-shuffle — replace each molecule's (or pair's) block by another molecule's (pair's)
   block, preserving every marginal exactly. `graph_shuffle` (5 rows: global62 groups,
   topology25), `readout_shuffle` (9 rows: pooled blocks), `relation_shuffle` (5 rows: the four
   used relation groups + all).

The zero probe can **understate** (R5 all-zero Δ = +0.164 vs R3 single-group Δ = +0.734 —
removing more caused less damage) and **overstate** (P1 unary second moment: zero +1.005 vs
shuffle +0.369) the true dependence. Verdicts in the report therefore always quote the
distribution-preserving probe when one exists, and flag `zero_probe_inflated` rows.
