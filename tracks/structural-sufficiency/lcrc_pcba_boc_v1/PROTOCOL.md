# LCRC-BOC v1 — protocol

Round: `lcrc-pcba-boc-v1`
Track: `structural-sufficiency`
Repository: `calendar0917/research`
Status: frozen before the formal 40-epoch runs.

## 1. Question

Can a **one-shot, learnable joint composition** of

1. bond-centred local instances with full real structural correspondence
   (`P_uv = G[{u,v} ∪ N(u) ∪ N(v)]`), and
2. ring-system instances (connected components of the non-bridge subgraph),

injected into the **atom main representation** (`h0_i`), improve the official
OGB `ogbg-molpcba` scaffold-split AP of LCRC-W2?

This is **not** a `W2_logits + residual` experiment.  No object encoder may read
`h0` or `h`, there is no iteration, no message passing and no multi-layer
attention over objects.  `h_i` is produced once by one fusion MLP.

Test is already exposed in this repository.  The Test role is
`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`: a descriptive engineering comparison,
never a blind estimate and never a model-selection signal.

## 2. Inheritance (read-only)

Only these historical implementations are the basis:

| file | use |
|---|---|
| `lcrc_pcba_backbone_scaling_v1/code/bs_model.py` | W2 trunk/readout, model/optimizer construction |
| `lcrc_pcba_backbone_scaling_v1/code/bs_common.py` | data identities, recipe, official evaluator, order plan |
| `lcrc_pcba_backbone_scaling_v1/code/bs_train.py` | exact `train_step`, checkpoint/IO helpers |
| `lcrc_tu_quick_screen_v1/code/tu_model.py` | `TULCRC` / `MLP` |
| `lcrc_pcba_s3_selective_v1/code/sp_test.py` | freeze/score discipline |
| `lcrc_pcba_continuation_recipe_v1/code/cc_data.py` | official Test label gate |

No historical file is edited.  The W2 trunk is **initialised** from the frozen
historical seed-0 untrained state (bitwise checked), not from trained W2
weights.  Training is from scratch for both arms.

## 3. Data and structural cache

* Dataset: official OGB MolPCBA, 437,929 molecules, 128 tasks.
* Split: official OGB scaffold split; Train 350,343 / Valid 43,793 / Test 43,793.
* Features: frozen Train-only normalisation inherited from `bs_common` /
  `pc_data`; no refitting anywhere.
* Training order: the frozen W2 seed-0 order plan
  `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`,
  batch size 128, 2,738 steps/epoch.
* Labels: the training artifact masks official Test targets.  The trainer never
  opens `raw/graph-label.csv.gz`.

The structure cache is **static, label-free** and built by
`code/boc_structures.py`:

* one BOC object per physical bond (12,309,186 objects);
* 34,369,144 side-neighbour entries;
* 34,610,219 same-side / cross-side pair entries;
* 1,100,845 ring systems, 7,343,607 ring-atom memberships,
  23,424,393 intra-system pairs, 2,386,191 external substituents;
* 1,645,218,822 bytes on disk (29 packed arrays);
* bridge detection on the original bond graph (iterative Tarjan), **no SSSR**;
* ring systems are connected components of the non-bridge subgraph;
* exact intra-system shortest-path distances, bucketed
  `1, 2, 3, 4, 5–6, ≥7` (6 buckets × 16-d embedding);
* no truncation or sampling of large systems;
* all training batches assembled with vectorised gather/scatter; no per-molecule
  Python loops in the forward path.

The cache is deterministic apart from metadata timestamps.  Its `cache_sha256`
covers the 29 arrays only.  `labels_read=false` and `label_artifact_opened=false`
are asserted by `load_cache`.

## 4. Architecture

Base: `BOCModel(C.BSM.BSModel)` with the frozen W2 width dict
(atom 192 / hidden 128 / bond 192), 128 outputs.

### 4.1 One computation pass

```
a, b, scal = W2._encode(batch)        # exactly once
h0          = W2.local(a, b, scal)    # exactly once
x_i         = SiLU(Linear(192,64)(a))
y_e         = SiLU(Linear(192,64)(b))
z_b         = bond_object(x,y,static structure)   # reads only x/y/structure
m_i         = Σ_{b∋i} z_b / sqrt(deg_i)
r_i         = ring_system_object(...)            # FULL; exact zeros in BOND
fusion_in   = concat(h0_i [128], m_i [64], r_i [32])   # 224
h_i         = LayerNorm(Linear(256,128)(SiLU(Linear(224,256)(fusion_in))))
logits      = W2.post_local(W2.drop(h_i))
```

* One `_encode`, one `local`, one fusion MLP.
* Object encoders never receive `h0`/`h`, never iterate, never message-pass.
* No zero-initialised output layer; fusion is standard random init.
* `m_i` uses `index_add` over incident bonds and `sqrt(deg)` normalisation.
* `BOND` instantiates identical ring modules but bypasses them; its ring content
  is exact zeros, and `full(ring_zero=True)` must equal `BOND` bitwise/near-bitwise.

### 4.2 Bond object (192 → 64)

For physical bond `(u,v)`:

* side sets `L = N(u)\{v}`, `R = N(v)\{u}`;
* side-neighbour `t` MLP `385 → 96 → 64`, input
  `[x_u, x_v, x_w, y_uv, y_uw, y_vw, is_side_R]`
  (385 = 6·64 + 1; `x` and `y` are both 64-d);
* query `q_uv` MLP `192 → 96 → 64`, `[x_u, x_v, y_uv]`;
* same-side fork `p_same` `257 → 96 → 64`, `[t_w, t_z, y_wz, is_R]`
  (skip the pair if `wz` is the centre bond);
* cross-side fork `p_cross` `257 → 96 → 64`, `[t_w, t_z, y_wz, y_uv]` with
  `w ∈ L`, `z ∈ R`, `w ≠ z`, skip `wz = uv`;
* normalised sums `s_L`, `s_R`, `c_LL`, `c_RR`, `c_LR` (each 64-d);
* final feature vector 389:
  `[q(64), s_L(64), s_R(64), c_LL(64), c_RR(64), c_LR(64), log1p|L|, log1p|R|,
  log1p|LL|, log1p|RR|, log1p|LR|]`;
* `bond_mlp` `389 → 128 → 64` + LayerNorm(64);
* compute once for `u→v` and once for `v→u` with the same parameters, then
  average the two 64-d outputs.  The reverse orientation swaps the side counts
  (`|L'|=|R|`, `|R'|=|L|`, `|LL'|=|RR|`, `|RR'|=|LL|`) as well as the sums.

### 4.3 Ring system (→ 32)

For each connected component `C` of the non-bridge subgraph:

* ring atom feature `[x_i, t_i]` projected 128 → 64, where `t_i` is the sum of
  projected features of external substituents (one-hop atoms outside `C`);
* intra-system exact distance buckets 0–5 (6 × 16-d);
* pair feature `[t_i+t_j (64), |t_i−t_j| (64), t_i⊙t_j (64), y_ij (64),
  dist_emb (16), is_bonded (1)]` = 273 → `ring_pair_mlp` 96 → 32; mean over
  unordered pairs;
* per-system concat `[mean_pair(32), mean_node(64), n_atoms, n_pairs,
  n_external]` (99) → `ring_out` 64 → 32 + LayerNorm(32);
* `r_i = r_C` on ring atoms, exactly zero elsewhere;
* no truncation/sampling.

### 4.4 Parameter budget

`W2 994,324 + bond 212,960 + ring 54,560 + fusion 90,752 = 1,352,596`.
Band 1.2M–1.6M.  Both arms have the same parameter count and state-dict keys.
The run stops if >1.6M.

## 5. Audits before formal training

CPU structural audit (`code/boc_audit.py structural`):

1. one object per physical bond and all source bonds represented;
2. atom/bond/neighbour/pair/ring pointer totals;
3. side counts `nL,nR,nLL,nRR,nLR` consistent with the blocks;
4. exact independent Python re-enumeration on 2,000 spread rows
   (neighbour entries, pair entries, ring systems, external substituents);
5. ring-system membership, pair completeness and distance buckets;
6. non-bridge / bridge correctness via independent bridge-by-removal;
7. label-free proof (`labels_read=false`, no label array);
8. full-cache hash and metadata identity.

Small-batch GPU audit (`code/boc_audit.py model`) on real rows:

9. parameter count and W2 bitwise initialisation;
10. centre-swap invariance of `z_b` (both endpoint orders averaged);
11. same-side neighbour-order invariance;
12. atom-renumbering invariance of logits;
13. two-molecule isolation (no cross-molecule leakage);
14. BOND ring content exactly zero and `full(ring_zero=True) == BOND`;
15. one-pass call counts (`atom_encoder=1`, `bond_encoder=1`, `local=1`,
    `fusion=1`, `q_mlp=2`, no ring calls in BOND);
16. finite forward/backward; bond/fusion/ring/W2 gradients as expected.

Formal training refuses to start unless both audit JSONs exist, pass, and bind
the same `cache_sha256` as the loaded cache.

Unit tests in `tests/` cover the same identities plus hand-built graphs
(chain, triangle, four-cycle with chord, benzene ortho/meta/para, fused
naphthalene, bridge-connected rings, brute-force pair enumeration, cache
roundtrip).

## 6. Training recipe

Frozen, identical for both arms:

* seed 0; 40 epochs; batch 128; 2,738 steps/epoch;
* fp32; AdamW `wd=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`;
* 1-epoch linear warmup `3e-5 → 3e-4`, per-step cosine to `1e-5`;
* dropout 0.10; gradient clip 5.0; masked BCE; official AP on Validation;
* same Train batch order as W2;
* best-Valid checkpoint saved on strict `>` AP (earliest wins ties), plus the
  fixed epoch-40 checkpoint, last resumable checkpoint, curves, per-task AP,
  channel RMS, gradient L2/nonzero fractions, GPU peak memory, step times,
  RNG hashes, environment.

GPU scheduling on `res` via `rr`:

* GPU1 = FULL (`res-gpu1`);
* GPU0 = BOND only when no external job is present and it cannot block FULL;
* never disturb another user's job; re-check `nvidia-smi` immediately before
  launch;
* budget ≤ ~3 A100 GPU-hours/arm; stop and report if >4.

## 7. Test freeze and score

After both formal runs:

1. build `analysis/selected_checkpoint.json` from Validation only;
2. `boc_test.py freeze` replays Validation AP for each checkpoint (tolerance
   `1e-4`), writes official Test **logits** for best-Valid and epoch-40 of each
   BOC arm, plus W2 and RC-REAL best/epoch-40 same-pipeline replays;
3. seal all checkpoint/logit hashes and Test row order;
4. only then `boc_test.py score` calls
   `cc_data.read_official_test_labels` and scores with
   `ogb.graphproppred.Evaluator("ogbg-molpcba")` plus independent per-task AP.

No Test label is read by training, profiling, auditing, cache building or the
freeze.  No Test score is used to select a checkpoint or an arm.

## 8. Comparison and decision

Primary comparison: official Test AP of `BOC-FULL best-Valid` vs
`W2 best-Valid` and `RC-REAL best-Valid` on the same pipeline.

Reference table:

| model | params | best epoch | best Valid AP | Valid AP @40 | Test AP best | Test AP @40 | GPU h |
|---|---:|---:|---:|---:|---:|---:|---:|
| W2 | 994,324 | 24 | 0.265273 | 0.261867 | 0.253616 | 0.250985 | 1.3949 |
| RC-REAL | 1,051,022 | 33 | 0.267718 | 0.266950 | 0.255941 | 0.256166 | 1.9186 |

Decision bands (engineering, not significance tests), applied to the
`BOC-FULL` official Test AP:

* `≥ 0.270` — strong evidence;
* `0.265 – 0.270` — meaningful improvement;
* `0.258 – 0.265` — small improvement;
* `≤ 0.258` — no evidence.

Also reported: per-task deltas and win/loss counts, support strata
(`<5`, `5–19`, `≥20` Test positives), channel norms/gradients, BOND-vs-FULL
zeroing sensitivity, inference cost, instance counts and cache size.  At most
two arms; stop after reporting.

## 9. Pre-registered implementation corrections

Found and fixed before any formal run:

* pair indices in `collate_boc` were offset by the molecule's neighbour-block
  start instead of the bond's block start; unit tests caught it.  The cache
  remains valid (it stores bond-local pair indices).
* the reverse bond orientation used the forward side counts in the 389-d
  feature vector; the protocol requires swapped counts for `z_vu`.  Fixed and
  covered by the centre-swap invariance test.

Neither correction changes the frozen architecture or recipe; both are recorded
in the README/report.
