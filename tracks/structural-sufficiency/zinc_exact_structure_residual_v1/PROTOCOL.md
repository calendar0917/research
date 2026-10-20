# PROTOCOL — zinc-exact-structure-residual-v1

| field | value |
|---|---|
| protocol id | `zinc-exact-structure-residual-v1` |
| track | `structural-sufficiency` (new, independent round directory) |
| code | `.../zinc_exact_structure_residual_v1/code/` |
| config | `.../configs/zinc_exact_structure_residual_v1.yaml` |
| status at freeze | `FROZEN_BEFORE_FORMAL_TRAJECTORY` |
| baseline | frozen P3 = BJOC-Null + train-only masked SSL + R3 (413,505 params) |
| data | official ZINC `subset=True` train 10,000 / valid 1,000 |
| official test | **never constructed, loaded or evaluated** |
| valid set | exposed development set; labels are the single existing read of `bjoc_recipe_tuning_v1` (sha256 `8f6ee636…`), reused read-only, no second read |
| budget | 8 A100-hours total, both GPUs summed; phase-1 target ≤ 4 h |

This is a bounded development round, not a confirmation and not a novelty claim.
It asks a single statistical question and stops on a pre-registered line.

## 1. Question

Does adding an **exact local structural residual** to the mature P3 system improve
held-out generalisation, and is a **real parent/child type-sharing prior** more
effective than an independent per-type residual?

The proposition under test is *statistical sharing of exact structure*, not a new
attention module. The round does **not** claim that attention or any other block
is the bottleneck, and it does not search over modules, ranks, radii or depths.

## 2. Data discipline

* Train: the 10,000 rows of the official PyG ZINC `subset=True` train split, in
  their official order. Valid: the official 1,000 rows.
* Every vocabulary item, support count, scale (`sy`, `Sh2`) and SSL checkpoint
  uses **train only**. Invalid/valid structures are only encoded at inference.
* The model reads exactly the frozen `MolObjects` built from official raw `x`,
  `edge_index`, `edge_attr`. No target decomposition, no `logP`/SA formula, no
  oracle cycle penalty, no extra chemical descriptor.
* The exact-structure branch exists **only in the supervised phase**. The masked
  SSL task continues to run on the frozen P3 encoder without any type identity;
  no unmasked structure identity is exposed to SSL.
* Valid labels are read once, in a previous round, and are reused here for a
  bounded configuration choice. Official test is never touched. No "blind set"
  is claimed.

## 3. Exact structure representation

### 3.1 Objects

For every atom `v` of a raw molecule:

* `r2(v)`: rooted attributed **induced** patch — all atoms at graph distance ≤ 2
  from `v`, the induced subgraph on them, **every** bond in the induced
  subgraph, with atom types and bond types as colours plus the root flag and
  root-distance class. This is the frozen `typed_tokenizer_v2_corrected`
  construction (`build_colored_incidence`; complete invariant = canonical
  adjacency + canonical semantic colour sequence).
* `r1(v)`: the same construction with radius 1. It is exactly the r2 patch
  truncated to distance ≤ 1 and re-canonicalised; it is **not** a string prefix,
  hash distance or approximate collision rule.
* `r0`: the raw atom type. Its correction is **fixed to 0** and it is not a
  learnable type.

### 3.2 Vocabulary (train only)

* support `n_t` = number of **distinct training molecules** containing type `t`
  (not atom occurrences).
* retained set = all 512 r1 types (ancestors and fallback targets) + all 15,218
  r2 types. `K = 15,730`. Ancestor closure holds for every r2 type.
* no truncation was needed: residual parameters = `96×4 + 4×K = 63,304`;
  total inference parameters = **476,809 ≤ 500,000**. `r0` costs nothing.
* the truncation rule (not triggered) is deterministic: r2 types in
  `(-support, key-bytes)` order, with the mandatory r1 ancestor's cost charged
  to the child; no additional frequency gate or arbitrary rarity threshold is
  applied.
* support histogram (r2): 6,784 types with support 1, 4,139 with 2–4, 1,762
  with 5–9, 1,890 with 10–49, 491 with 50–199, 152 with ≥ 200. Median 2,
  q90 18, q99 199, max 5,296.
* on valid, 2.92% of atoms have an unseen r2 type and therefore receive the
  retained r1 correction; 0% have neither r2 nor r1 (zero correction).
  An unseen r2 is **not** claimed to have learned its exact identity.

## 4. Model

```
h_v = a_v^P3 + a[t*(v)],     a[t] = P @ beta[t]
```

* `P ∈ R^{96×4}` (the actual `atom_out` is 96), orthonormal non-zero columns,
  sign-fixed, deterministic per seed; `beta ∈ R^{K×4}`, initialised to 0.
* the residual is added to the atom embedding **before** the unchanged bond,
  junction, composer and `321→256→256→1` reader stack. Every P3 parameter is
  retained and jointly trained with the original R3 recipe. The reader is not
  shrunk to make room.
* `t*(v)` = deepest retained type: r2 → r1 → r0 (zero correction).
* `B` = original P3, 413,505 params, no residual and no prior.
* `E` = independent residual, `H` = parent-shared residual. `E`/`H` use the
  same vocabulary, rank, parameter count, initial weights (bit-identical) and
  data order. Only the prior differs.

## 5. Prior / regulariser

With `N = 10,000`, `K = 15,730`:

```
R_E = mean_t ||a[t]||² / (depth(t) · Sh2)
R_H = mean_t ||a[t] − a[parent(t)]||² / Sh2        (a[r0] = 0)
lambda = eta · sy · K / (2N)
loss = batch molecule-mean raw-y L1 + lambda · R_arm
```

* `Sh2` = mean squared norm of the SSL-initialised train atom embeddings,
  computed once per seed before training and frozen: seed 0 = 16.386387,
  seed 1 = 17.768767 (a strictly positive lower bound is enforced).
* `sy` = mean absolute deviation of the train labels from the train median =
  1.472189 (frozen).
* `eta ∈ {0.1, 1.0}` only. These are dimensionally explicit initial screening
  strengths, not a claim of an optimal range. They are frozen before the first
  formal valid metric is read.
* the prior is evaluated on the **complete vocabulary at every optimizer step**,
  including types that only learn through the prior. It is not restricted to
  the current batch and it is not reweighted by occurrence count. `lambda` is
  not divided by batch size.
* `P` and `beta` have optimizer weight decay **0**; the original P3 parameter
  group keeps weight decay `1e-5`. The prior acts on `P @ beta`, not on `beta`
  alone.
* `E`'s depth factor matches the prior marginal scale of each depth under a
  fixed `P`; it prevents a looser deep prior from being misread as a sharing
  benefit.

## 6. Recipe (inherited from frozen P3/R3)

| item | value |
|---|---|
| loss | raw-y L1, batch molecule mean |
| batch / steps | 128 / 79 per epoch |
| epochs / updates | 360 / 22,680 |
| optimiser | Adam, lr 1e-3, per-step cosine → 1e-5 |
| weight decay | 1e-5 (P3), 0 (P, beta) |
| grad clip | 5.0 |
| soup | parameter mean of the last 5 epoch states |
| selection | no early stopping, no best-valid epoch; the fixed soup is the result |
| precision | FP32, no AMP, no `torch.compile` |
| batch order | `torch.randperm` with `Generator(seed + 101)`, one permutation per epoch; seeds 0/1 must match the frozen R3 hash |

Same-seed models share the same SSL starting point:
`bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{0,1}.pt` (train-10k-only
masked SSL). Seeds 2/3, if purchased, train the **same frozen SSL function and
recipe** and the checkpoint is written under this round's `ssl_pretrain/`.

## 7. Pre-training checks (`gates`)

1. parameter budget and vocabulary counts;
2. ancestor closure, direct r2→r1 parent recomputation on real molecules, and
   deepest-retained-type/OOV fallback rule;
3. zero residual reproduces B bit-identically, and E/H initial weights match;
4. `beta` has gradient at step 1, `P` has zero gradient at step 1 and non-zero
   gradient at step 2, prior is exactly 0 at initialisation;
5. atom-restricted relabelling leaves the patch keys, per-atom types and
   prediction unchanged (max abs ≤ 1e-4);
6. every vocabulary key is a train type and its stored support equals the
   recomputed distinct-molecule support; no valid label is read before freeze;
7. batch order equals the frozen R3 hash for seeds 0/1;
8. optimizer weight-decay split is exactly {1e-5, 0};
9. module scope is the frozen P3 stack plus one per-type linear map;
10. retention/truncation accounting.

## 8. Phase-1 run matrix (pre-registered, all reported)

| GPU | seed | configs (serial) |
|---|---|---|
| `res-gpu0` | 0 | `B`, `E@0.1`, `E@1.0`, `H@0.1`, `H@1.0` |
| `res-gpu1` | 1 | `B`, `E@0.1`, `E@1.0`, `H@0.1`, `H@1.0` |

No `r3`, attention, extra SSL loss, ensemble, new readout, target decomposition
or auxiliary task is added.

## 9. Primary table and decision rule

Primary metric: fixed last-5-soup raw-y MAE on official valid 1,000, per seed and
seed-mean. Contrast convention: `delta = error(candidate) − error(B)`, so
negative favours the candidate.

A family (`E` or `H`) is **continued to phase 2** only if, for one of the two
declared `eta` values, relative to `B`:

1. both seeds improve (per-seed delta < 0);
2. seed-mean valid MAE gain ≥ **0.003**;
3. the ordinary region (`y > -3`) gain is not reversed;
4. the slice with B's worst 10 molecules removed is not reversed.

If a family triggers, freeze the triggering `eta` with the largest seed-mean
gain. Then, inside the remaining budget, add seeds 2 and 3 for `B` and each
triggered family. 0.003 is a development-budget line, not a scientific zero
threshold. If nothing triggers, stop: no further strength, rank, radius or
module search.

Pre-declared descriptive slices (thresholds fixed before any valid-label read):
low-support type = train support < 10 molecules; molecule groups by fraction of
low-support root atoms (none / partial / majority ≥ 0.5) and by r2-fallback
burden (none / partial / ≥ 0.25); ordinary (`y > -3`) vs tail; large-ring
(ring size ≥ 6); n-atom terciles from train (21 / 25). Molecules are the
evaluation unit. Paired bootstrap (5,000 resamples, seed 0) describes
molecule-level uncertainty for the given seeds only; it does not replace
multi-seed or independent-data confirmation.

## 10. Reporting

Every seed and every configuration is reported, including failures. The report
contains: train/valid MAE, B−E, B−H, E−H paired differences, both eta levels,
parameter counts, training time, preprocessing time, the GPU ledger, support and
fallback statistics, effective correction norms, prior values, branch gradients
and a zero-correction dependence probe.

Interpretation mapping (fixed in advance):

* E and H both beat B: exact residual is useful; hierarchy is not yet necessary.
* H beats B and H beats E: support continuing the shared design.
* H beats E but H ≈ B: mainly avoids lookup damage; no incremental value.
* train-only improvement: the design has not solved generalisation; stop.
* branch inactive or the two prior levels indistinguishable: report
  insufficient treatment strength; do not write "structure information is
  useless".

## 11. Budget

8 A100-hours total (sum over both GPUs), phase-1 target ≤ 4 h. Includes smoke,
all formal runs, prediction, retries, and (if purchased) the two SSL runs for
seeds 2/3. The `smoke` stage projects the phase-1 matrix with a 1.20 margin plus
a 300 s reserve and refuses to proceed if it exceeds the phase-1 cap. If the
phase-2 projection would exceed 8 h, no extra run is purchased. GPU and CPU
preprocessing time are recorded separately.

## 12. Non-claims

This round does not demonstrate interpretable chemical contribution, does not
establish paper-level novelty, does not produce an official-test number, and
does not select a model on test. It is a two-or-four-seed development result on
one exposed valid split.
