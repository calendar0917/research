# cscl-task-adaptive-structure-units-v1 — task-adaptive molecular structure units

One-round protocol: learn **structural unit boundaries end-to-end from the task
loss** on top of frozen Atom/Bond encoders (masked-SSL init from
`bjoc_task_adaptive_ssl_v1/pretrain_D_s0.pt`), and ask whether the learned
partition is useful, connected, non-duplicated and reusable.

## Layout

- Prototype: `code/task_adaptive_structure_units_v1.py`
- Driver:   `code/run_task_adaptive_structure_units_v1.py`
- Results:  `results/cscl_task_adaptive_structure_units_v1/`

## Part A — canonical physical bonds and base blocks

Raw graphs are canonicalised to unique undirected physical bonds
`(u<v, type)` before any topology: the v0 double-count bug (each directed copy
counted separately) is gone.  Base blocks are the **bridges + 2-edge-connected
components** of the physical bond graph (NetworkX), with an independent
reference implementation (`cscl_units`-style DFS + brute-force edge-removal
reference) and a path-product membership cross-check.

ZINC split D applicability (label free): fit has **10.85 base blocks and 9.85
bridge boundaries per molecule on average**, 99.99% of molecules have at least
one learnable boundary, 78.5% of blocks are single atoms and 21.5% are ring
systems; SSSR ring bonds are never bridges (0 overlap in 9,000 molecules).
Boundary learning is therefore applicable; the GPU stage is not skipped.

## Part B — learned boundaries

Bridge merge scores `p_e = sigmoid(s_theta([h_u+h_v, |h_u-h_v|, h_u*h_v, b_e]))`
read only base-block summaries (`sum`` pooling of atom/bond embeddings; no
message passing).  Hard 0.5 partition with a straight-through surrogate, so the
raw-y L1 loss reaches the scorer.  Units are merged connected components of the
block tree (connected by construction, never duplicated).

Four arms share seed, initialisation, batch order and recipe (R3, 360 epochs,
batch 128, Adam 1e-3, cosine to 1e-5, wd 1e-5, clip 5, raw-y L1, last-5 soup):

| arm | partition | readout |
| --- | --- | --- |
| F-A | all bridges cut (fixed) | `y = b + Σ α(U) + Σ γ(R)` (additive) |
| L-A | learned merge gates | additive |
| F-O | fixed | opaque reader (capacity matched to the additive heads) |
| L-O | learned | opaque |

## Part C — CPU feasibility before GPU

- `cpu_gates.json`: structural correctness, path-product membership, additive
  decomposition identity, invariance to atom relabelling and bond order,
  scorer gradient activity, parameter inventory / forbidden modules, capacity
  match of the readout contrast, smoke reproducibility.
- `synthetic_s1.json`: direct boundary supervision learns held-out bridge
  accuracy **0.914** and exact partition **0.732**.
- `synthetic_s2.json`: y-only, known-truth grouping task with oracle-partition
  and all-merge controls (3 seeds).  Grouping helps (oracle **1.56** vs fixed
  **2.17**); the learned partition beats fixed on average (**2.07**) but is
  unstable and collapses to all-merge in 2/3 seeds — a known non-identifiability
  result that we keep rather than tune away.  Oracle-partition alpha/gamma
  direction recovery is positive (Pearson 0.64/0.53 for the true unit/bridge
  terms) but partial.

## Part D — frozen GPU pass and evidence

8 runs (`res-2`, Slurm, cu124 pool): 4 arms × 2 seeds.  Monitor (split D, 1000
molecules) predictions are frozen and hashed **before** the single label read;
the locked side is never touched.  H1 gate: seed-mean monitor ΔMAE of
L-A vs F-A ≤ **-0.01**, both seeds negative, bootstrap CI upper < 0, ordinary
and top-10-removed deltas not worse, RMSE within +0.02, no single-row dominance.
Readout contrasts (F-O vs F-A, L-O vs L-A, L-O vs F-O) answer the explicit
decomposition vs opaque readout question; BJOC-Null monitor MAE 0.104301 is the
read-only reference.

After scoring, label-blind audits quantify partition degeneracy, exact-isomorph
unit reuse (WL buckets + NetworkX exact isomorphism, H'/α consistency within
float noise), and the additive decomposition identity on real data.
