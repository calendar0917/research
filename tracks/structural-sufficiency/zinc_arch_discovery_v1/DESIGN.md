# DESIGN — zinc-arch-discovery-v1

**Status:** implementation + timing; formal seed-0 runs follow the measured
schedule. This is an architecture-discovery round, not a GRIT reproduction and
not a mechanism claim. It replaces the old long-run GRIT-baseline plan for this
task. No official test prediction/MAE is computed.

## 0. Fixed boundary

* Data: official `torch_geometric.datasets.ZINC(root=data/ZINC, subset=True)`,
  train 10,000 / valid 1,000. Raw-`y` L1. Labels only from train/valid.
* Primary selector: per-seed single **best-valid** epoch-end checkpoint; ties
  keep the earlier epoch. Valid is evaluated every epoch and used for
  development. Test is the terminal split and is **not** read in this round.
* Parameter cap: <= 500,000 trainable parameters, including any SSL encoder.
  Every model records its actual count and optimizer coverage.
* No local recursive message passing. Allowed: MLPs, GRU over sampled paths,
  static relation tensors, global self-attention. Static graph statistics are
  precomputed from the raw atom/bond graph; no hidden-state neighbour updates
  are used to create them.
* Existing label-free P3 features (`atom_feat`, `bond_feat`, `junc_feat`,
  `global33`) are reused where useful. Every new statistic is a deterministic
  function of the input graph, never of `y`, SA/logP components, or target-
  derived labels. The new cache is independent from the P3 cache and is built
  with an atomic single-writer prep job; training jobs only read it.
* No VQ. P0/P1 use a fixed sampled walk bank; they claim only relabelling-
  distribution equivariance, not exact invariance under one finite sample.
* No new SSL pretraining. T loads the existing train-only masked-SSL P3
  encoders (seed-matched) and then trains them jointly with the new readout.

## 1. Six candidates

| id | representation | encoder / composition | readout | params (actual) | SSL |
|---|---|---|---:|---:|---|
| **R0** | rooted balls r=1/2/3 per atom: root P3 atom descriptor, exact distance-shell atom-type counts, cumulative ball joint counts over unordered endpoint-type pair x bond type (231x3=693), 11 degree/ring/branch scalars | shared 2-layer MLP (96), concat+residual scale fusion | sum, mean, second moment, count, atom/bond composition, global33; residual MLP | 307,809 | none |
| **R1** | same R0 atom tokens and cache | 2 pre-LN global attention layers, d=96, 4 heads, FFN x2; bias = shortest-path bucket + sparse two-hop `(middle atom, two bond types)` 189-way relation | same count-aware readout | 458,961 | none |
| **T** | frozen P3 atom/bond/junction objects and static incidence | shared P3 encoders; low-rank rank-16 atom-bond and bond-junction association tensors `sum (Ua) outer (Vb)` / `sum (Pb) outer (Qw)`; no Null grouping, no composer, no softmax reallocation | first-order sums/means per object type, incidence counts, association tensors, global33 | 462,081 | P3 masked SSL (seed 0/1), jointly trained |
| **P0** | fixed bank of 128 sampled walks/molecule (32 each length 2/4/6/8), atom types, bond types, anonymous first-visit pattern; reversal-canonical; train samples 8 per length (32 total), valid uses all 128 | shared atom/bond/anon embeddings + 1-layer GRU (96) over semantic+anonymous tokens; padding is packed out | length-grouped sum/mean/second moment, pattern count, full atom/bond composition, global33 | 367,361 | none |
| **P1** | same P0 bank and local GRU | P0 + one pre-LN global pattern-attention layer (d=96, 4 heads, FFN x2), no VQ | same readout | 442,145 | none |
| **B** | graph-invariant biconnected blocks + bridges + isolated atoms; per block joint atom/bond counts, size/cyclomatic/ring/articulation statistics | shared block MLP (96) + one pre-LN block attention layer with block-cut-tree distance and block-overlap bias; no recursion on the block-cut tree | sum/mean/second moment of blocks, block count, full-molecule atom/bond composition once, global33 | 338,805 | none |

All six are distinct computation graphs. R0 is a static readout over richer
rooted statistics, R1 adds shallow global composition, T redoes the object
association statistics, P0/P1 are path-pattern models, and B is a ring/block
view. None is a renamed local MP layer.

## 2. Feature definitions (fixed before training)

* **Root joint counts**: for each root `v` and radius `r`, count every
  undirected physical bond whose two endpoints are within distance `r`.
  Category = unordered endpoint atom-type pair (21x22/2 = 231) x bond type
  (single/double/triple = 3). This is a true joint histogram; it is not two
  concatenated marginals. Counts are raw integers (not log1p). Distance-shell
  atom-type counts are exact shell counts; for radius `r`, shells > r are
  zeroed. The 11 scalars use log1p for sizes/counts and fractions elsewhere;
  scalar standardization is fit on train only.
* **Two-hop relation** (R1): `C[u,v,a,b1,b2] = sum_w 1(atom(w)=a)
  1(bond(u,w)=b1) 1(bond(w,v)=b2)`, 21x3x3=189 categories, stored sparsely
  per molecule and added to the attention bias after a per-head linear map.
  Shortest-path distance is bucketed 0..8. No dense all-pairs histogram is
  stored.
* **T associations**: `C_AB = sum_(v,e incidence) (U a_v) outer (V b_e)` and
  `C_BJ = sum_(e,j incidence) (P b_e) outer (Q w_j)`, rank 16. Both are divided
  by their incidence count (mean outer product); the counts are kept explicitly
  as `log1p` channels. No softmax over objects.
* **Walks**: a walk may revisit atoms (including immediate backtracking); it is
  not restricted to simple paths. Lengths are 2/4/6/8 edges. Each sampled walk
  and its reverse are compared lexicographically and the canonical orientation
  is stored; this gives exact reversal symmetry. The anonymous pattern maps
  each atom occurrence to its first-visit position. The bank seed and valid
  rule are fixed (`20261021 + split_offset`). Train selection is 8 per length,
  valid uses all 32 per length.
* **Blocks**: `nx.biconnected_components` plus bridges (2-atom blocks) and
  isolated atoms. Block-cut tree nodes are blocks + articulation atoms;
  distance buckets and overlap counts are computed statically. Ring statistics
  reuse P3's label-free ring/min-ring flags. Articulation atoms are not
  double-counted in the molecule-level composition readout.
* **Global composition**: full atom-type counts (21) and bond-type counts (3)
  are read once per molecule. `global33` is standardized with the frozen
  train-only P3 prep mean/scale when present; otherwise it is fit on our train
  cache.

## 3. Recipes

| id | optimizer | LR -> min | WD | warmup | batch | planned epochs | valid batch | target |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R0 | Adam | 1e-3 -> 1e-5 | 1e-5 | 500 | 128 | 600 | 256 | ~0.27 GPU-h |
| R1 | AdamW | 8e-4 -> 1e-5 | 1e-4 | 500 | 128 | 600 | 256 | ~0.91 GPU-h |
| T | AdamW | 1e-3 -> 1e-5 | 1e-5 | 500 | 128 | 600 | 256 | ~0.36 GPU-h |
| P0 | AdamW | 1e-3 -> 1e-5 | 1e-5 | 500 | 128 | 600 | 16 | ~0.45 GPU-h |
| P1 | AdamW | 8e-4 -> 1e-5 | 1e-4 | 500 | 128 | 600 | 8 | ~0.57 GPU-h |
| B | AdamW | 1e-3 -> 1e-5 | 1e-4 | 500 | 128 | 600 | 256 | ~0.41 GPU-h |

The per-step cosine horizon is fixed at the start of each formal run. The
2-epoch timing job measured 1.65 s/epoch (R0), 5.46 s/epoch (R1), 2.18 s/epoch
(T), 2.69 s/epoch (P0), 3.44 s/epoch (P1), and 2.48 s/epoch (B) including full
valid; the pre-declared 600-epoch schedule is ~0.27/0.91/0.36/0.45/0.57/0.41
GPU-h respectively, i.e. about 0.3-0.9 GPU-h per candidate. The horizon is not
reset or compressed mid-run. `steps/epoch = ceil(10000/128)
= 79`, so the last batch is retained (16 rows). Valid is full 1,000 every epoch
for all candidates; P0/P1 use the full 128-walk bank at valid time and the
cost is included in the measured epoch time.

## 3b. Pre-declared follow-up correction: P0/P1 sampled-count leak

The original P0/P1 run exposed a readout-scale mismatch, not a representation
failure.  P0 s0 reached best-valid 0.625814 at epoch 70 (train@best 0.608447),
and P1 s0 reached 0.729133 at epoch 223 (train@best 0.697384).  On the same 128
valid molecules, the P1 best checkpoint scored MAE 0.3382 when given the
32-walk training sample and 0.6911 when given the 128-walk fixed valid bank;
on 128 train molecules it scored 0.1908 (32 walks) vs 0.6941 (128 walks).  The
reader consumed a raw path sum and `log1p(number of sampled walks)`.  Those are
split indicators under the declared 32-train / 128-valid sampling, not molecular
features.  The model learned the train-side scale and failed when the valid bank
supplied four times as many walks.

Correction (only this readout change; representation, bank, attention, recipe
and 600-epoch schedule unchanged): remove the raw sum and raw sampled count from
the reader; keep per-length mean and second moment, both divided by the actual
sampled count, plus the full-molecule atom/bond composition, molecule size, and
global33.  Corrected parameter counts are P0 281,121 and P1 355,905.  The two
corrected seed-0 runs are the pre-declared follow-up; the original P0/P1 runs
remain in the result table and are not deleted.  This is a disclosed bug fix and
its compute is billed below.

## 3c. Pre-declared follow-up correction: T weight decay

T is the best new candidate (0.091375 at epoch 426), but it fits the training
objects more tightly than P3 while generalizing worse: T train@best 0.009922,
final online train 0.000535, versus P3 s0 train@best 0.011261 and best-valid
0.079105.  This is a plausible regularization mismatch, not evidence that the
association representation is exhausted.  One pre-declared correction: raise T
weight decay from 1e-5 to 1e-4 (10x), keep the rank-16 tensor, P3 SSL encoders,
readout, optimizer, LR schedule and 600-epoch horizon fixed, seed 0.  No LR/WD
grid and no seed-1 repeat in this round.  If this single correction does not
improve best-valid, stop and report T as the best new candidate but still below
the P3 reference; do not add another module or family.

## 4. Checks before formal runs

* Data labels/splits aligned to official PyG; frozen P3 object cache verified
  by count/atom/bond/label SHA; cache audit checks joint mass, block joint
  mass, walk lengths and reversal canonicalization.
* Each candidate: forward on a real batch, finite scalar, nonzero gradient for
  every parameter tensor, parameter count <= 500k, optimizer covers all
  parameters.
* R1/B: cross-graph attention isolation by block-diagonal mask; padding does
  not enter readout.
* P0/P1: packed GRU removes padding; valid uses the fixed bank; train selection
  is per-length stratified.
* Static-statistic relabelling invariance is checked on synthetic graphs.
* No test split construction in any formal path.

## 5. Budget and stop

Cap: 10 A100-hours total (both GPUs summed), with six seed-0 candidates at
~1 h each and up to ~2 h for one or two evidence-driven follow-ups. Two GPUs
only reduce wall-clock; GPU-hours add. This round does not need to spend the
full cap. The first-round stopping rule is: run all six seed-0 candidates for
their measured schedule unless a numerical error or the hard cap stops one;
then use at most two follow-up purchases (seed 1 of a clear leader, a
budget-boundary continuation, or one pre-declared recipe correction). Do not
open a seventh family or a full GRIT reproduction.

`<= 0.075` best-valid is a development target, not a universal significance
threshold. A single-seed win is not a stable improvement claim.

## 6. Post-round terminal test addendum (outside the pre-registered round)

After the six candidates and the two pre-declared follow-ups were complete and
the round report was closed, the user requested a test generalization check.
The test split was then constructed and read **once**, after all checkpoints and
selection rules were frozen. No model, recipe, epoch, or checkpoint was changed
from test feedback. The original round boundary ("no test during development")
was therefore preserved; the late read is reported as a terminal descriptive
evaluation in `REPORT.md` §5b and `analysis/test_terminal.json`. P3's frozen
checkpoints were evaluated directly on the same test objects and labels
(`analysis/p3_test_direct.json`), not retrained. Original P0/P1 were excluded
from test scoring because their sampled-count readout is invalid and their old
reader shape is not loadable by the corrected model.
