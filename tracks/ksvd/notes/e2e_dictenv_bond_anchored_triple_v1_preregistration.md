# Pre-registration — `e2e_dictenv_bond_anchored_triple_v1`

Round: **BondAnchoredTriple-v1** (mainline performance screen), study
`zinc-context-gap`, track `ksvd`, runner
`zinc_bond_anchored_triple_v1_mainline`.

This is a **single-candidate 80-epoch performance screen** on top of the
**frozen** `CSSD-Sem108 + C6` parent soup.  It does **not** retrain the parent,
does **not** run any shuffle / mechanism / ablation arm, does **not** use a
second seed, and never reads the official test split.  It answers exactly one
question and then stops at a report:

> With the parent dictionary / environment / pair encoders frozen, does keeping
> the **common-endpoint correspondence of three pair tokens** before graph-level
> aggregation — one environment per real bond anchor and one environment per
> third node — provide a validation-MAE signal worth a matched follow-up?

This is a **screen**, not a causal comparison.  Any performance difference is
`M1_soup − M_parent` against the **historical parent**, not an estimate of the
triple operator's contribution (the M1 Reader is retrained, the inputs are
standardized and the budget differs).

Frozen **before** any caching run or training.  Nothing below is tuned after
seeing results.  A changed pre-registration voids the round.

---

## 0. Frozen parent

* Checkpoint:
  `tracks/ksvd/results/e2e_dictenv_sem108_v1/checkpoints/SEM108-seed0_soup_state.pt`.
* Canonical state hash (`audit.state_sha256` over sorted keys, recorded and
  re-verified at load time):
  `7a721984c5579dd76f9bcaff13a2055b8efba71da553eefb3645d94207f5050a`
  (full hash is read from the artifact, never completed from the prefix).
* Source run: `tracks/ksvd/results/e2e_dictenv_sem108_v1/` (runner commit
  `f836b04`, analysis commit `c841c6d`), historical official-valid soup MAE
  `0.123704927947314` under the C6 mask.
* Frozen parent configuration: `cm.H1_CONFIG` (`h1`, `d_e=48`, `K=32`, `s=8`,
  IHT-10, `lambda=33.95873017865987`, horizon 320), common subspace `q1`
  from `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json`,
  dictionary `sdb32` sha
  `b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd`.
* Training/selection mask: `cm.C6_MASK` (canonical form; equal to the recorded
  clarity-audit C6 set):
  `global_zero_groups=("atom_histogram","bond_histogram")`,
  `unary_zero_blocks=("count",)`, `pair_zero_blocks=("count",)`,
  `relation_zero_groups=("path_count",)`.
* The parent is loaded with `sem.build_sem108_model(D, seed=0, q1)` +
  `load_state_dict(soup)`.  49 state keys, 97,709 parameters, pair-token width
  16, environment width 48, original Reader input 302, Reader hidden `(13,13)`.
  These are re-measured from the instance at preflight; any mismatch aborts the
  round (no silent width change).
* Parent replay gate: the parent's official-valid MAE, recomputed on CPU with
  the C6 mask and the frozen evaluation loader, must satisfy
  `|replay − 0.123704927947314| <= 1e-7`.  The measured value is recorded
  alongside the historical value.  A CPU/GPU discrepancy is located before any
  training; the round does not proceed on an unexplained replay gap.
* **No parent parameter is ever updated.**  All parent parameters are set
  `requires_grad_(False)`; feature extraction runs under `eval()` +
  `torch.no_grad()`.  The training model contains **no parent parameters**
  (its forward consumes only detached cached tensors), so parent gradients
  cannot accumulate and cannot enter the optimizer or the gradient-clipping
  set.  Parent state hashes are recorded before and after the training run.

## 1. Data and splits

* Official ZINC train 10,000 / official valid 1,000, the existing repository
  split and target handling (`p1run.load_split`, encoded + environment caches,
  order = official split order).  Batch size 128.
* The official **test** split is never instantiated, loaded or evaluated
  (`official_test_loaded = false`); the control plane runs non-terminal with
  `test_access = blocked`.
* Graph order inside each split is the cache order; targets are the
  per-graph `y` of the same objects.  Cache and target alignment is checked
  (replay of cached `z_old` through the original Reader must reproduce the
  direct parent forward).

## 2. Frozen feature cache

For every graph in each split, one parent forward (eval, `no_grad`, C6 mask)
captures and stores:

1. `p_ij` — the 16-D `pair_value` the parent feeds to pair pooling, once per
   **unordered** environment pair (`pair_cache` row order, `i < j`), i.e. the
   output of the frozen `pair_encoder` under the C6 relation mask.
2. `z_old` — the full 302-D `unified` Reader input
   (`[unary(97); pair moments by distance bucket(165); graph hidden(32);
   topology(8)]`) with the original unary / bucketed pair moments / graph
   hidden / topology behaviour and the C6 mask.
3. Per-graph bookkeeping: node count, unordered pair count, pair-bucket array
   (to extract the unique real-key anchors), graph target `y`.

Cache store (per split, one artifact):

* `pair_tokens [P,16] float32`, `z_old [G,302] float32`, `y [G] float32`,
  `n_nodes [G]`, `pair_ptr [G+1]`, `pair_bucket [P] int64`.
* Provenance payload: parent checkpoint **full** sha256, canonical parent
  state sha256, C6 mask signature, extraction code version (module sha256 +
  cache format version), split name, graph order policy, git commit.
* The cache is reused only when the whole provenance payload matches; any
  mismatch forces a rebuild (no silent reuse across parent/code changes).

Extraction is batched (batch size 128) and split back per graph by the pair
pointer; every cached value is re-verified on a subset against a fresh
per-graph extraction.  Because the frozen parent forward is float32, a
batched and a per-graph forward can differ by rounding only (verified
bitwise-stable within each path); the alignment check therefore allows
`<= 2e-6` for pair tokens and `<= 1e-5` for `z_old`, requires targets to be
bit-identical, and requires the replayed Reader prediction of the cached
`z_old` to match the direct parent forward to `<= 1e-6`.

## 3. Triple objects

For each graph with `n` nodes and `m` unique real bonds (unordered keys,
`i < j`):

* anchors = cache rows with `pair_bucket == 0` (verified to be exactly the
  unique real bonds of that graph; the `env_bond_*` occurrence list is **not**
  used as an anchor list).
* for each anchor `(i,j)` and every `k not in {i,j}` (any distance, no
  adjacency restriction) one triple `(i,j;k)` using the cache rows of the
  unordered pairs `(i,j)`, `(i,k)`, `(j,k)`.
* each real bond is used with exactly **one** endpoint order `i < j`; the
  directed edge entries are never treated as two anchors.
* per-graph object count is exactly `m*(n-2)`; graphs with `n < 3` or `m == 0`
  contribute the zero 64-D summary (no triples).
* no sampling, no truncation of large graphs.  Triple rows live in the global
  per-split pair-token matrix; batching adds **pair-row** offsets (never node
  offsets) and every triple row of graph `g` must fall inside
  `[pair_ptr[g], pair_ptr[g+1])` (no cross-graph reference).

## 4. Architecture (exactly as locked)

Training-only path on top of the frozen parent (new parameters only):

```python
F = Sequential(Linear(48, 64, bias=True), SiLU(), Linear(64, 32, bias=True))  # 5,216 params

p_bar = (p − mu_p) / scale_p

a = cat([p_bar_ij, p_bar_ik, p_bar_jk], dim=-1)
b = cat([p_bar_ij, p_bar_jk, p_bar_ik], dim=-1)
t = 0.5 * (F(a) + F(b))                       # shared F, both orderings

z3 = cat([mean_over_tuples(t), mean_over_tuples(t*t)], dim=-1)   # 64-D, per graph
z3_bar = (z3 − mu_3) / scale_3
z_old_bar = (z_old − mu_old) / scale_old

Reader = Linear(302 + 64, 13) -> ReLU -> Linear(13, 13) -> ReLU -> Linear(13, 1)
prediction = Reader(cat([z_old_bar, z3_bar], dim=-1))
```

* F has no dropout, gate, trainable scale, residual chain or extra norm;
  both orderings call the **same** F module.  PyTorch default Linear
  initialisation (no tiny/zero output init).
* `mean`, not `sum`; no tuple-count feature; no node/pair state update; no
  message passing, recurrence or attention.
* z3 is zero for graphs with no triples; the same fixed `mu_3 / scale_3` is
  applied (`(0 − mu_3) / scale_3`).
* Parameter accounting (measured, must match): F 5,216; new Reader 4,967
  (old Reader 4,135, so +832); trainable 10,183; full model
  `97,709 − 4,135 + 10,183 = 103,757`.  A mismatch is investigated, never
  silently accepted or patched by changing widths.
* No parent prediction is fed in, and no small-scale residual from the parent
  output is attached.

## 5. Initialisation and normalisation

* Model RNG: `torch.manual_seed(0)` immediately before constructing F then the
  Reader (construction order F → Reader).  Data order RNG is independent
  (`torch.Generator().manual_seed(0 + TRAIN_SHUFFLE_OFFSET)` for the shuffled
  train loader), so statistics computation cannot perturb either.
* Train-only statistics, float64 accumulation, float32 model:
  1. `mu_p, std_p` over **all** official-train pair tokens (`p_ij`),
     per channel.
  2. `mu_old, std_old` over official-train `z_old` (302 channels).
  3. initialise F; compute the official-train raw `z3` under `no_grad` with
     the above `p_bar`; take `mu_3_init, std_3_init` over all train graphs
     (empty-triple graphs enter as zero vectors).
* All statistics are non-trainable buffers, fixed for training and inference;
  they are never re-estimated when F updates.  The valid split contributes to
  no statistic.
* Channel rule: `x_bar = (x − mu) / scale` with `scale = std` for
  `std >= 1e-6`, and `scale = 1` for near-constant channels
  (`std < 1e-6`).  The near-constant count and channel indices are recorded for
  all three blocks.
* No valid mean/variance/label is ever used.  Soup never averages or
  re-estimates the statistics (they are identical buffers in every member).

## 6. Training protocol (one candidate, one seed, fixed budget)

* Seed 0; exactly **80 epochs**; no warm-up/pilot optimisation and no
  early stopping by valid; no scheduler, dropout or learning-rate search.
* Batch 128 graphs; train order reshuffled per epoch by the loader generator
  seeded `0 + 91011`; valid order fixed (split order, batches of 128).
* Loss: graph-level `L1(prediction, y)` (the frozen parent task term).  The
  parent reconstruction term is constant w.r.t. the new parameters, so it is
  not used.
* Optimizer: `torch.optim.Adam`, `lr = 1e-3`, `weight_decay = 1e-5` (coupled
  L2), parameters = F + new Reader only (identity-checked).
* Gradient clipping: global norm 5.0 over **exactly the optimiser's parameter
  list** (the frozen parent is not in that list and accumulates no gradient).
* Per epoch record train MAE, valid MAE and wall time; epoch 1 time estimates
  the remaining budget.  No automatic extension to 320 epochs: the round stops
  at 80 regardless of outcome.

## 7. Soup and metrics

* Save the trainable-parameter checkpoints of the **top 5 epochs of epochs
  41–80 by valid MAE** (ties: earlier epoch).
* Soup = arithmetic mean of the F and Reader parameters of those 5
  checkpoints, loaded onto the same frozen parent and the identical fixed
  buffers.
* Primary metric `M_soup` = soup official-valid MAE.
* Also reported: parent replay MAE `M_parent` (recorded historical value
  `0.123704927947314`), `Delta = M_soup − M_parent`, overall best valid MAE and
  epoch, member epochs/values, mean valid MAE over the last 20 epochs, final
  train MAE, wall time, s/epoch, peak RSS, all correctness checks and parent
  state hashes before/after.
* `Delta` is a difference against the **historical, unmatched** parent; it is
  not a matched-control treatment effect.

## 8. Stop rule and verdict

* `M_soup <= 0.120` → `FROZEN_TRIPLE_SCREEN_PROMISING` (report only; worth a
  matched follow-up, no automatic extra training).
* Otherwise, if the run is complete and valid →
  `FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL` (round closed; no rescue).
* Repair/cache/correctness failure, numerical failure or an incomplete 80-epoch
  run → the corresponding `invalid`/`incomplete` state; **no performance
  verdict**, no shortened-horizon substitute primary metric.
* The 0.120 gate is never relaxed after seeing results (not even for a
  `0.000x` miss).

## 9. Required checks (only those needed for this round)

1. Parent replay within `1e-7`; graph/target order and cache alignment.
2. Unique-anchor count == the graph's unique real bonds; triple count
   `m*(n-2)`; no out-of-range row; no cross-graph triple row; sample checked
   against brute-force Python enumeration.
3. Endpoint swap `i <-> j` leaves `t` unchanged; batched graph pooling equals
   per-graph pooling within float tolerance.
4. Hand-computed mean / second moment on a tiny example, including the
   empty-triple case.
5. One training batch: finite loss/output/gradients; F and Reader weights
   receive non-zero task gradients (record grad norms, input/output RMS).
6. Parent parameters `requires_grad_(False)` and `.grad is None`; the
   optimiser and clipping sets are exactly F + Reader; parent state hashes
   identical before/after training.

## 10. Explicitly out of scope

* No third-environment correspondence shuffle, code shuffle, atom shuffle,
  semantic shuffle or any mechanism/ablation arm.
* No M0 retraining, no JointBond/diagnostic checkpoint as parent, no
  weight-decay contrast, no denormal rescue, no second seed, no hyper-parameter
  search, no second architecture, no 320-epoch extension.
* No official test access of any kind.

## 11. Reporting

Implementation commit, result commit, run ID, device (A100 if available,
otherwise the repository's CPU regime with the reason recorded), full parent
checkpoint hash, cache provenance, parameter accounting, all metrics above,
the verdict, and an explicit statement of what was and was not run.
