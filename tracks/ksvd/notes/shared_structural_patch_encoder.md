# Shared Structural Patch Encoder (vocab-free) — ZINC Cell A

**Date:** 2026-09-15
**Branch:** `exp/shared-structural-patch-encoder`
**Commit (formal runs):** `c9574bda1a00e834dbf91161704c1d58d1ee8966` (base code `3b3ff676`)
**Run tags:** `sspe-s0` (GPU0), `sspe-s1` (GPU1)
**Remote:** `res` = `hxy@a100-2`, repo `/home/hxy/cy/research`
**Verdict:** **STRONG_SUCCESS** — 2-seed Top-5 soup valid mean **0.118972** vs Cell A **0.126368** (Δ = **−0.007396**, both seeds same direction), at **84,495** params, vocab-free.

---

## 1. Question

Can the vocab-sized typed certificate→ID→learned-embedding lookup be removed
entirely and replaced by a **shared encoder** that reads each rooted typed
radius-2 patch's **internal connectivity + semantic attributes**, while
**preserving or improving** the frozen deterministic-A100 ZINC Cell A baseline?

Cell A (frozen reference): `h=64, q=16, T=2, patch_hidden=64, global_hidden=32`,
**85,763 params**, 2-seed fixed Top-5 soup valid **0.126368**. Official test
**never loaded** in this study.

## 2. Pre-registered architecture (exactly one)

Exactly one architecture was registered and run; no width/depth sweep.

- **Node primitive** (per patch-node): `atom_embedding(28,48)` +
  `root_embedding(2,48)` (root flag) + `distance_embedding(3,48)` (rooted
  distance bucket). No certificate ID, no learned token dictionary.
- **Edge primitive**: `bond_embedding(4,24)`.
- **2 untied edge-aware message-passing rounds.** Round *r*:
  `m = G_r([state[src], bond_embed])`, `agg = scatter_mean(m, dst)`,
  `state = state + F_r([state, agg])` (`G_r`/`F_r` are plain MLPs).
- **Permutation-invariant pooling**: `[root_state, mean_nodes, std_nodes]`
  → fusion MLP → `e_struct ∈ R^16`.
- Widths: `node_dim=48, edge_dim=24, hidden_dim=48, rounds=2,
  include_std_pool=True, output_dim=16` → **35,152 params**, replacing the
  released typed lookup (**36,420 params**).
- **Preserved unchanged:** radius-2 patch definition, relation system, pair
  descriptors, `T=2` recurrent pair-centre, parent embedding, global/topology
  channels, R width (334), graph head, training protocol.
- **Forbidden and absent:** attention/Transformer, learned token dictionary,
  certificate-ID lookup, certificate bytes as input, exact identity residual,
  MolHIV training, ZINC official test.

Node/edge primitives are exactly the three atom-side attributes and the one
bond attribute; OOV patches are naturally encodable because representation is
computed from connectivity, not from a vocabulary index.

### 2.1 How this differs from earlier closed lines

| Closed line | What it kept | What this study does instead |
|---|---|---|
| Historical **v6** attribute-role factorization | pooled `(type, role)` primitives **independently**, kept a coarse certificate lookup | propagates **along real patch edges** and **fully removes** the typed lookup |
| **Compositional patch sharing** (KNN/rare-OOV) | borrowed from frequent frozen embeddings (still a vocabulary) | reconstructs each patch from scratch; no neighbors, no donor rows |
| **Shared Structural Base + Exact Identity Residual** | kept an exact-identity residual | no identity residual at all |
| **SBCI** low-rank relational composition | factorized the relation/centre family | leaves relation/centre family untouched; only replaces the patch embedding source |
| **Corrected exact-token lookup** | a (corrected) vocabulary table | no table; representation width is fixed and independent of vocabulary |

## 3. Training protocol

Deterministic-A100 regime (`torch.use_deterministic_algorithms(True)`), seed0
and seed1 in parallel on two A100-40GB GPUs. Adam `lr=1e-3`, `wd=1e-5`,
batch 128, `max_epochs=240`, `patience=40`, scheduler none, L1 loss, gradient
clip 5.0, fixed equal-weight Top-5 checkpoint soup. Train shuffle offset
91011, eval offset 91012.

The candidate is built by mirroring
`capacity_decomposition.build_cell("A", seed)` (temporarily relaxing
`rec.Q_DIM`/`shead.R_DIM` to 64/334) so all shared tensors are copied
**bit-exactly** and the `GenericReader` head is re-drawn from seed 0, exactly
as in Cell A. The only difference is the patch representation.

### 3.1 Determinism / reproducibility smoke

A short 6-epoch `repro` stage was run in parallel on both A100s. The two runs
were **bit-identical**: same valid curve, same best valid `0.30481370162963867`,
same selection-state SHA-256 `91f1a615b5f8dc01985b14a0d7ca834feb4e6f1bd9165b8c531248831528f276`.

### 3.2 Structural graph cache (remote preprocess)

`shared_structural_patch_graphs_v2`, official train only (valid at selection):
10,000 train molecules / **231,664 patches** / **1,418,500 patch-nodes** /
**2,465,688 directed edges**, mean 6.10 nodes/patch, max 14; valid 1,000
molecules / 23,083 patches; 89.8 s; `official_test_loaded=false`.

## 4. Tests (all pass)

`sanity` (remote, 14/14): node relabel invariance; atom-type change moves the
representation; bond-type change moves it; root-position change moves it;
same rooted typed graph under permutation gives the same representation
(edge-order invariance in unit tests); unseen certificate needs no vocabulary
row (`no_certificate_dependence` — zeroing `typed_token` changes nothing);
**no vocab-sized `typed_embedding.weight`**; parent path preserved;
`q=16/h=64/T=2`; recurrent weight tying; forward finite; backward finite and
nonzero; total params in 80k–90k.

Unit test file `tracks/ksvd/tests/test_shared_structural_patch_encoder.py`
(11 tests) plus `test_compact_v4_recurrent_pair_centre.py`,
`..._capacity_decomposition.py`, `test_compact_v4_cell.py`, `test_core_api.py`
→ **72 passed**.

## 5. Parameter accounting

| Block | Cell A | Candidate |
|---|---:|---:|
| typed token embedding (hybrid lookup) | 36,420 | **0** |
| shared structural patch encoder | 0 | **35,152** |
| parent token embedding | 256 | 256 |
| patch / pair / relation / centre / global / topology / head | 49,087 | 49,087 |
| **Total** | **85,763** | **84,495** |

`candidate_has_typed_embedding = false`; no `typed_embedding` state keys.

## 6. Results (valid, seed0+seed1)

| seed | raw best valid | best epoch | train@best | Top-5 soup | wall | peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.125324 | 231 | 0.106296 | **0.119818** | 5503.8 s | 0.407 GB |
| 1 | 0.124394 | 197 | 0.092201 | **0.118126** | 2998.5 s | 0.410 GB |

- **2-seed Top-5 soup mean = 0.118972** (raw mean 0.124859).
- Cell A reference = 0.126368.
- **Δ = −0.007396**, per-seed Δ = −0.004886 / −0.009906, **same direction**.
- Decision logic: Δ ≤ −0.002 and same direction → **STRONG_SUCCESS**.

(seed0 ran longer because GPU0 was shared with another user's 34.8 GiB process;
seed1 ran alone on GPU1. Memory stayed ≈0.4 GB for this model — no OOM risk.)

## 7. Diagnostics

Effective rank (entropy-based) / participation ratio / top singular fraction,
measured over 5,864 valid patches per seed:

| quantity | seed0 | seed1 |
|---|---|---|
| `e_struct` (shared structural encoder output, R^16) | rank 1.107, PR 1.043, top 0.979 | rank 1.202, PR 1.094, top 0.955 |
| patch `h0` (post patch-encoder, R^64) | rank 18.39, PR 8.22, top 0.301 | rank 19.30, PR 7.91, top 0.320 |

**Caveat / honest reading.** The shared encoder's per-patch output is
**nearly rank-1** across patches (one dominant direction, ~96–98 % of energy),
i.e. it behaves close to a learned per-patch bias. The recurrent patch
representation `h0` keeps substantially more rank. Therefore the observed gain
should **not** be attributed to rich structural connectivity being richly
exploited; the defensible claim is narrower: *the categorical vocabulary
memory can be removed and replaced by a small shared connectivity encoder
without hurting (here improving) Cell A performance*. Sensitivity tests
confirm the encoder is not constant (atom/bond/root changes move `e_struct`),
but its cross-patch variation is low-rank. This is a property of the trained
solution, not a protocol violation.

## 8. MolHIV parameter projection (accounting only, no training)

Using `results/molhiv_parameter_attribution/parameter_breakdown.json`
(total **1,076,589**):

- removed typed table `[26233,32]` = **839,456** params;
- shared structural encoder with `output_dim=32` (to keep the MolHIV patch
  encoder input width unchanged) = **35,936** params;
- **projected total = 1,076,589 − 839,456 + 35,936 = 273,069 params**.

No MolHIV training, no performance claim.

## 9. Stop rules honoured

No depth/width sweep; no residual exact identity; no hash table; no KNN
sharing; no corrected exact lookup; no MolHIV training; no ZINC official test.
A single bug in the decision reader (wrong JSON key `best_epoch`) was found and
fixed (`c9574bda`) before evaluation; it changed no training result.

## 10. Provenance & artifacts

- Code: `structural_patch_encoder.py`, new model file
  `zinc_shared_structural_patch_encoder.py`, modifications to
  `zinc_patch_path_pooling.py` and `zinc_compact_v4_recurrent_pair_centre.py`,
  tests `test_shared_structural_patch_encoder.py`.
- Remote results pulled to
  `tracks/ksvd/results/shared_structural_patch_encoder/` (curves, selection +
  Top-5 states, soups, `runs/*.json`, `soup_sspe_seed*.json`,
  `diagnostics.json`, `parameter_accounting.json`, `molhiv_projection.json`,
  `sanity.json`, `decision.json`, `report.json`, `repro_gpu{0,1}_seed0.json`,
  cache).
- Local reproduction of `decide` gives the same **STRONG_SUCCESS** verdict.

## 11. Consequence / next boundary

This study **closes** the question "can the typed lookup be removed by a shared
connectivity encoder without loss?" with a **yes, with a measured gain**, but
flags that the encoder is near-collapsed. It does **not** authorize:
- architecture search around this encoder (width/rounds/pooling),
- claiming a structural-inductive-bias advantage,
- opening the ZINC official test,
- MolHIV training.

A legitimate next question (not run) is whether the near-collapse can be
avoided (e.g. by a better readout or an auxiliary structural objective) and
whether that converts the gain into genuinely structure-attributable benefit.
