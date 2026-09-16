# MolHIV shared, vocabulary-free B-Bag patch encoder (parallel transfer)

Date: 2026-09-16
Protocol: `molhiv_shared_bag_patch_encoder_v1`
Formal code commit: `fb172dd` (base `132f100`)
Branch: `exp/molhiv-shared-bag-patch-encoder` (parallel to the ZINC/BCE
`exp/binding-composition-encoder-zinc` line; not merged)
Isolation: local worktree `../research-molhiv-bbag`, remote worktree
`/home/hxy/cy/research-molhiv-bbag`; remote default checkout
`/home/hxy/cy/research` (GPU0/BCE) was never switched.
GPU provenance: **GPU1 only**; GPU0 (pid 3885391, ~35 GiB) untouched.
Official test: loaded once, after `architecture_freeze.json`; no change after
test.

## Question (one sentence)

> Replace MolHIV's vocabulary-sized exact rooted-patch identity embedding with a
> single shared DeepSets-style encoder over OGB atom/root/distance and bond
> primitives, while leaving the relational downstream unchanged.

Background: the frozen MolHIV H96 recurrent pair--centre port
(`molhiv_recurrent_pair_centre_v1`) has `1,076,589` parameters, of which
`839,456` (77.97 %) are one dense `typed_embedding.weight` of shape
`[26233, 32]` — a learned row per exact rooted typed certificate seen in the
official training split. On ZINC the analogous lookup was removable with a
shared connectivity encoder (`shared_structural_patch_encoder`, 84,495 params,
near-rank-1) and with its connectivity-free ablation B-Bag. This experiment
asks whether the same substitution transfers to a second, scaffold-split
molecular benchmark.

## What was replaced (single variable)

Only the per-patch token source:

```text
before: token_p = typed_embedding[certificate_id_p]                (R^32)
after:  token_p = MolhivSharedBagPatchEncoder(patch primitives)    (R^32)
```

Unchanged (inherited verbatim from the frozen MolHIV H96 RPC port): radius-2
rooted patch definition, `patch_cont` shell descriptor (793-D), radius-1 parent
embedding, shortest-path-conditioned pair relation, `pair_projection` /
`relation_encoder` / `distance_gate` / `pair_encoder`, `T=2` weight-tied
recurrent pair--centre, unary + pair-moment + global readout, prediction head,
unweighted `BCEWithLogitsLoss`, Adam lr 1e-3 / wd 1e-5 / batch 128 / 240 epochs
/ patience 40 / clip 5.0, official OGB scaffold split, and the
best-valid + fixed Top-5 soup selection rule with **no train+valid refit**.

The exact typed vocabulary is never fit and never enters the forward. The
radius-1 parent lookup is kept (1,648 params — not a parameter-explosion
source).

## B-Bag encoder (one pre-registered architecture; no sweep)

OGB atoms are multi-field categorical, so each field gets its own fixed
**schema-sized** embedding (sizes from `ATOM_FEATURE_DIMS = (119,5,12,12,10,6,6,2,2)`
and `BOND_FEATURE_DIMS = (5,6,2)`, never from the observed certificate count):

```text
atom_base(v) = mean_k atom_emb_k(atom_field_k(v))                 in R^48
z_v          = node_mlp(atom_base(v) + root_emb(root_v)
                        + dist_emb(root_distance_v))              in R^48
bond_base(e) = mean_k bond_emb_k(bond_field_k(e))                 in R^24
g_e          = bond_mlp(bond_base(e))                             in R^24
token_p      = fusion([z_root ; mean_v z_v ; std_v z_v ;
                       mean_e g_e ; std_e g_e])                   in R^32
```

Widths: `node_dim=48, edge_dim=24, node_hidden=96, bond_hidden=48,
fusion_hidden=104, output_dim=32` (matching the old token width so the patch
encoder input width is unchanged), `n_distance_bins=3` (radius 2). The "mean"
over fields is implemented as a `1/sqrt(K)`-scaled sum.

The forward reads `struct_atom_fields`, `struct_root`, `struct_dist`,
`struct_patch` (node→patch grouping), `struct_bond_fields` and
`struct_edge_patch` (bond→patch grouping) only. It **never** reads
`struct_src` / `struct_dst`: no message is propagated along a real edge, so two
patches with identical primitive multisets but different connectivity receive
exactly the same token (the definitional B-Bag property).

## Preprocessing / cache

The existing exact-rooted record cache
(`results/luyin16/unified_relational_patch_molhiv_exact_rooted/record_cache`,
6.5 GB, linked into the isolated worktree) supplies certificates, shell
descriptors, pair relations and global context. A **new, target-free,
label-independent, schema-versioned** structural cache
(`molhiv-shared-bag-structural-cache-v1`) supplies the B-Bag primitives;
patch `p` corresponds to centre `graph.nodes[p]`, matching the record cache
one-to-one (verified on train/valid prefixes before freeze).

| split | graphs | patches | patch-nodes | patch-bonds | build | cache size |
|---|---:|---:|---:|---:|---:|---:|
| train | 32,901 | 830,936 | 5,130,112 | 4,412,152 | 19.0 s | 174.7 MB |
| valid | 4,113 | 114,300 | 729,396 | 637,826 | 2.8 s | 24.8 MB |
| test | 4,113 | 103,927 | 661,915 | 577,455 | 2.5 s | 22.6 MB |

`official_test_labels_loaded=false`; the pre-freeze patch-order audit touched
only train/valid. No exact certificate id, no target, and no label is stored in
the structural cache.

## Parameter accounting (first key result)

`sum(named_parameters)` cross-checked against the block breakdown: exact.

| block | params | % new |
|---|---:|---:|
| atom-field embeddings (9 fields × 48) | 8,352 | 2.97 % |
| root + distance embeddings | 240 | 0.09 % |
| bond-field embeddings (3 fields × 24) | 312 | 0.11 % |
| bag node MLP (48→96→48) | 9,360 | 3.33 % |
| bag bond MLP (24→48→24) | 2,376 | 0.85 % |
| bag fusion (192→104→32) | 23,432 | 8.33 % |
| **B-Bag encoder total** | **44,072** | **15.67 %** |
| exact typed token embedding | **0** | 0 % |
| radius-1 parent embedding | 1,648 | 0.59 % |
| patch encoder | 90,336 | 32.13 % |
| pair / relation modules | 8,928 | 3.17 % |
| recurrent centre update | 23,676 | 8.42 % |
| global encoder | 12,128 | 4.31 % |
| prediction head | 100,417 | 35.71 % |
| **total** | **281,205** | 100 % |

```text
old total                 1,076,589
new total                   281,205      (-73.880 %)
old typed identity          839,456  ->  0
old exact identity (total)  841,104  ->  1,648   (parent only)
```

There is no `[num_exact_patch_types, D]` tensor anywhere: the largest tensor is
`head.0.weight [192,423]` (81,216). The new total is well under the 0.40 M
guard.

## Correctness tests

`tracks/ksvd/tests/test_molhiv_shared_bag_patch_encoder.py` (15 pass; 22 with
the frozen RPC tests, locally and on the remote GPU node):

* **A** node permutation invariance; **B** edge-order invariance;
* **C** connectivity-free witness — same primitive multisets, different
  adjacency (endpoints present but unread, and removed entirely) → identical
  token;
* **D/E** every one of the 9 atom fields and 3 bond fields moves the token;
  root position and root-distance move the token; field embeddings are exactly
  schema-sized; out-of-range indices raise;
* **F** no `typed_embedding` attribute/state, no tensor ≥ 100 k, total ≤ 0.40 M,
  exact identity = parent table only;
* **G** batch invariance (single molecule vs the same molecule inside a batch);
* **H** finite/non-zero gradients on atom/bond embeddings, node MLP, bond MLP,
  fusion;
* downstream equivalence: when the B-Bag token is replaced by the old typed
  lookup, the model output is bit-equivalent (< 1e-6) to the frozen RPC model;
* custom collate offsets patch groupings across a batch.

Remote `sanity` (11/11 checks pass, `--device cuda`): no typed embedding,
no vocabulary-sized tensor, params below guard, output width 32, parent
preserved, `T=2/q=16`, forward finite, gradients healthy, batch invariance,
connectivity-free token.

## GPU1 smoke

`smoke --device cuda` (2,048 train molecules): 20 steps, no NaN/Inf/OOM,
gradients alive; peak GPU **0.356 GB**; **760.7 graphs/s** (5.94 steps/s);
valid ROC-AUC already 0.672 after 20 steps.

## Results (frozen protocol, official validation selection, no refit)

| seed | best ep | epochs | mean ep | peak GPU | raw valid | soup valid |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 51 | 30.05 s | 0.799 GB | **0.823648** | 0.808091 |
| 1 | 15 | 55 | 51.12 s | 0.830 GB | **0.828802** | 0.827516 |
| **2-seed mean** | | | | | **0.826225** | **0.817803** |

Matched frozen reference (`molhiv_recurrent_pair_centre_v1`, 1,076,589 params,
same split/protocol/device regime): raw valid 0.814239 / 0.849865 (mean
0.832052), soup valid 0.809505 / 0.851092 (mean 0.830299).

Valid deltas: raw mean **−0.005827**, soup mean **−0.012496**; per-seed signs
flip (seed0 raw **+0.009409**, seed1 raw **−0.021063**). seed1's mean epoch
time (51.1 s vs seed0's 30.1 s) reflects GPU1 being shared with a concurrent
BCE run, not an architecture difference.

## Frozen official test (opened once after freeze)

| seed | raw test | soup test |
|---|---:|---:|
| 0 | 0.769918 | 0.779067 |
| 1 | 0.746828 | 0.776477 |
| **mean** | **0.758373** (−0.004231) | **0.777772** (+0.015677) |
| 2-seed ensemble (diagnostic) | 0.777680 (−0.002368) | **0.787817** (+0.011848) |

Reference (RPC, same protocol): raw test mean 0.762605 (ensemble 0.780048),
soup test mean 0.762095 (ensemble 0.775969).

Context (different protocols; not a like-for-like rank): prior
`MOLHIV_PATCH_PATH_POOLING_20260904` 0.7852; CIN-small 0.801; GPS 0.788;
GIN 0.756. The user's external ~0.80 target is **not reached** by this frozen
variant: the single-model soup is 0.7778 and the (diagnostic) 2-seed soup
ensemble 0.7878, i.e. 0.0122 short.

## The two questions

**Q1 — is the parameter explosion solved? YES.**
No exact typed vocabulary table exists; the only dataset-dependent identity
storage left is the 1,648-param radius-1 parent lookup (0.59 % of the model).
Total params drop from 1,076,589 to 281,205 (−73.880 %) with the relational
downstream bit-identically preserved. There is no tensor whose size grows with
the number of exact rooted patches.

**Q2 — did predictive transfer succeed? PARTIAL / preserved, aggregator-dependent.**
* The frozen official-test **raw** checkpoint is statistically tied with the
  baseline (mean −0.0042; 2-seed ensemble −0.0024).
* The frozen official-test **Top-5 soup** is **better** than the baseline
  (mean +0.0157; 2-seed ensemble +0.0118), and 0.7878 is the best MolHIV
  number in this line so far (above the prior 0.7852 port and the old 0.7800
  ensemble).
* On the official **validation** 2-seed mean the variant is mildly *lower*
  (raw −0.0058, soup −0.0125) with a per-seed sign flip.

Because the valid→test direction reverses between the raw and soup rules, the
defensible claim is *no meaningful loss / mildly better on official test*, not
a transfer advantage. The 4,113-graph / 81-positive valid split makes valid
selection noisy, so a −0.006 to −0.012 valid difference at 2 seeds is not
separable from selection noise. There is no evidence that the removed exact
identity was carrying irreplaceable predictive signal.

## Discipline / stop rules honoured

* GPU1 only; GPU0 untouched; no DDP; no cross-GPU; no unknown process killed.
* No test access before freeze; test loaded once; no change after test.
* Exactly one architecture; no width/depth/round/pooling/radius/head/loss
  sweep; no class weighting; no refit; no B-Full/ASB/Binding-Composition/
  attention added to MolHIV.
* Both seeds were run as pre-registered (no early stop on a "mediocre" seed0).
* Results live in a separate directory (`results/molhiv_shared_bag_patch_encoder/`)
  and a separate branch; the BCE/GPU0 line was not touched or merged.

## Is it worth migrating the Binding Composition method to MolHIV?

Not on this evidence. The **connectivity-free** shared compositional patch
representation already transfers with no meaningful loss at 74 % fewer
parameters, and on ZINC the connectivity-aware B-Full encoder did not beat its
connectivity-free B-Bag ablation. Adding connectivity-aware Binding Composition
to MolHIV would not be motivated by a demonstrated gap that B-Bag fails to
close. If a future step wants to chase the ~0.80 test target, the honest lever
is more seeds / a less noisy selection rule, not migrating the binding cell.

## Files

- `experiments/luyin16/molhiv_shared_bag_patch_encoder.py`
- `tests/ksvd/tests/test_molhiv_shared_bag_patch_encoder.py` (15 pass)
- `results/molhiv_shared_bag_patch_encoder/`: `data_sanity.json`,
  `preprocess.json`, `parameter_accounting.json`, `sanity.json`, `smoke.json`,
  `run_seed{0,1}.json`, `raw_state_seed*.pt`, `soup_state_seed*.pt`,
  `architecture_freeze.json`, `official_test_unlock.json`,
  `official_test_results.json`, `report.json`, `curves/`, `structural_cache/`
