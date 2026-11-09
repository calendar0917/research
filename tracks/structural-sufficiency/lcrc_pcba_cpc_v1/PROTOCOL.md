# PROTOCOL — LCRC-CPC v1

Round: `lcrc-pcba-cpc-v1`
Status: **frozen before the first formal training step.**  Any post-freeze
incident is recorded in `REPORT.md` under "Incidents and deviations"; this file
is not silently edited after the run.

## 0. Scope and one-sentence question

> With LCRC-W2's local encoding and readout unchanged, can a one-shot, rooted,
> two-bond path conditioning of already-learned atom states — a nonlinear joint
> computation over the `h0` of root/middle/endpoint atoms and their bonds —
> recover part of W2-DYNAMIC's gain **without iterative hidden-state
> propagation**?

Exactly **one** model is trained formally: `LCRC-CPC`, seed 0, 40 epochs.  This
is a fixed-mechanism boundary experiment, not a PathNN/GNN reproduction, not a
tuning search, and not a capacity-matched architecture comparison.  No second
CPC form, STATIC arm, MP arm, other path length, attention variant, additional
seed, external GNN baseline, new objective, pretraining or augmentation is
allowed.

Historical references (read-only; Test values are exposed engineering
comparisons, never a blind estimate and never a selection signal):

| Model | Parameters | Best Valid AP | Best-Valid Test AP |
|---|---:|---:|---:|
| W2 | 994,324 | 0.2652733161 | 0.2536168632 |
| RC-REAL | 1,051,022 | 0.2677177024 | 0.2559409168 |
| BOC-FULL | 1,352,596 | 0.269950 | 0.254142 |
| W2-DYNAMIC | 1,128,468 | 0.2750671763 | 0.2622260490 |

## 1. Inherited, read-only implementations

* `lcrc_pcba_backbone_scaling_v1/code/bs_model.py` — `BSModel` (W2 at scale 2),
  the historical W2 seed-0 constructor, `permute_atoms`, `make_optimizer`,
  `set_lrs`, checkpoint-state helpers.
* `lcrc_pcba_backbone_scaling_v1/code/bs_common.py` — W2 widths, frozen
  scratch recipe, frozen 40-epoch batch-order plan, data identities, provenance.
* `lcrc_pcba_scale_pilot_v1/code/pc_data.py` — frozen PCBA cache access and
  batch collate semantics.
* `lcrc_pcba_s3_v1/code/s3_paths.py` / `s3_common.py` — historical S2 path
  representation and the path-construction conventions used for cross-checking.
* `lcrc_pcba_structural_gain_sprint_v1/code/sg_data.py` — official task, split,
  NaN-masked Train/Valid label artifact, official AP.
* `lcrc_pcba_boc_v1/code/boc_model.py` — BOC path-object encoding reference
  only; no BOC code is executed by this model.
* `lcrc_pcba_pw_v1/REPORT.md` — pair-attention degeneration lesson; CPC uses an
  independent sigmoid gate, never a cross-path softmax.
* `lcrc_pcba_dynamic_v1/code/dy_model.py` / `REPORT.md` and
  `analysis/test_prediction_manifest.json` — the iterative comparison and its
  frozen Test logits.
* `lcrc_pcba_continuation_recipe_v1/code/cc_data.py` — the only gated reader of
  official Test labels, after the prediction freeze.

## 2. Model: `LCRC-CPC`

### 2.1 Main path

Construct the **untrained** historical W2 seed-0 model.  Do not load a trained
W2 checkpoint.  Keep every W2 computation unchanged:

```python
a, b, scal = base._encode(batch)          # once
h0 = base.drop(base.local(batch, a, b, scal))  # local and drop once

delta = CPC(h0, b, path_index)            # once, reads only this h0/b
h_cpc = h0 + delta

logits = base.post_local(h_cpc, batch)    # inherited W2 readout, once
```

`h0, delta, h_cpc` are `[N_atoms,128]`; `b` is the W2 bond-encoder output
`[N_bonds,192]`; `delta` is `[N_atoms,128]`.  No path reads `h_cpc`; no updated
atom state is ever written back and re-read.

### 2.2 Path instances

A path is a rooted simple two-bond path `i--j--k`:

* `i` root, `j` middle, `k` endpoint;
* `(i,j)` and `(j,k)` are real physical bonds;
* `i,j,k` are pairwise distinct;
* `(i,j,k)` and `(k,j,i)` are different instances because the update target
  differs.

For every root `i`, enumerate `j ∈ N(i)`, then `k ∈ N(j) \ {i}`.  Keep every
legal path, all multiplicities of different middles reaching the same endpoint,
triangle paths, and both directions.  No sampling, truncation, shortest-path
dedup, virtual node or artificial edge.

The cache stores `cpc_ptr`, `cpc_middle`, `cpc_end`, `cpc_bond_ij`,
`cpc_bond_jk`; root is implicit in the CSR pointer and materialised exactly as
`path_atom` during collate.  Per-molecule `row_atom_ptr`, `row_bond_ptr` and
per-row path/atom/bond counts make the molecule-local offsets explicit.

### 2.3 Reuse of the frozen S2 structure

Do not rebuild molecules from RDKit.  Start from the frozen PCBA cache fields
`atom_ptr`, `bond_ptr`, `bond_index`, `path_ptr`, `path_end`.  The original S2
count identity \( \#S2(i)=\sum_{j\in N(i)}(\deg(j)-1) \) is re-read from the
cache, never hard-coded.

The new enumerator is checked independently against a slow Python reference and
against every molecule of the frozen cache:

* every root's CPC count equals the original `path_ptr` S2 count;
* the total is 34,369,144 for the full 437,929-row cache (actual value read);
* endpoint multisets match the original `path_end` per root (order may differ);
* both physical bond IDs really connect `(i,j)` and `(j,k)`;
* all three atoms belong to exactly one molecule;
* no cross-molecule gather/scatter.

If a reliable bond-ID correspondence cannot be established, the round stops
with `STRUCTURE_MAPPING_INVALID`; no guessing.

Full-cache identity (label-free, structure only):

* rows: 437,929; paths: 34,369,144; size: 658,409,864 bytes;
* enumeration schema: `rooted-two-bond-simple-path-v1`;
* row IDs sha256: `85b8d38b10293ddd5f431c6feae3aa9c22ec56f19a3a2367c4da2c170d9c1642`;
* per-field checksums and cache `meta.json` sha256 are recorded in
  `analysis/structure_cache.json` and `analysis/preflight.json`;
* source cache sha256 `caad3e90…`, feature-spec hash `7270f767…`, split
  partition sha256 `c872bd63…`, label artifact sha256 `5c3b5fbe…`.
* The Test structure cache may be built without labels; Test labels stay NaN.

### 2.4 Input projections

Compute once per atom:

```python
u = LayerNorm(128)(h0)
r = SiLU(W_root(u))     # Linear(128,32)
m = SiLU(W_middle(u))   # Linear(128,32)
d = SiLU(W_end(u))      # Linear(128,32)
```

Three independent linear maps, one shared LayerNorm.  Compute once per real
bond:

```python
e = SiLU(W_bond(b))     # Linear(192,16)
```

`b` is the learned W2 bond-encoder output, not a raw feature pipeline.  All
projections happen before path gathering.

### 2.5 Joint path token

For each `p=(i,j,k)`:

```python
x_p = concat(
    r[i], m[j], d[k],
    r[i] * d[k],
    m[j] * d[k],
    e[bond_ij_id],
    e[bond_jk_id],
)                       # [N_paths,192]

z_p = SiLU(
    Linear(96,64)(
        SiLU(Linear(192,96)(x_p))
    )
)                       # [N_paths,64]
```

The elementwise products are required; pure additive marginals are not the
mechanism.  One fixed token MLP, no extra layers or candidate widths.

### 2.6 Root-conditioned sigmoid gate

```python
gate_p = sigmoid(
    Linear(96,1)(concat(r[i], z_p))
)                       # [N_paths,1]
```

Independent per-path gate; no softmax across paths, no attention heads, no
top-k truncation.  The gate is not required to be non-uniform for the model to
use joint interactions; gate and token statistics are reported separately.

### 2.7 One-shot root aggregation

```python
count[i] = number_of_paths_rooted_at_i
context[i] = sum_p(gate_p * z_p) / max(count[i], 1)
log_count = log1p(count)                      # [N_atoms,1]
delta = Linear(65,128)(concat(context, log_count))
delta[i] = 0 exactly where count[i] == 0      # final bias cannot leak
h_cpc = h0 + delta
```

Aggregation executes once.  There is no `h1 = CPC(h0); h2 = CPC(h1)`, and no
middle/endpoint write-back followed by a second propagation.

### 2.8 Parameter budget (must match exactly)

| Component | Parameters |
|---|---:|
| Shared LayerNorm(128) | 256 |
| 3 × Linear(128,32) | 12,384 |
| Linear(192,16) bond projection | 3,088 |
| Path Linear(192,96) | 18,528 |
| Path Linear(96,64) | 6,208 |
| Gate Linear(96,1) | 97 |
| Output Linear(65,128) | 8,448 |
| **CPC new** | **49,009** |
| W2 original | 994,324 |
| **total** | **1,043,333** |

Per-tensor audit must match.  Widths are never adjusted to fit a budget.

### 2.9 Initialisation

* W2 part: historical seed-0 constructor state, bitwise per tensor.
* CPC modules: constructed inside `torch.random.fork_rng`; parameters
  re-initialised from a private generator `seed*1_000_003 + 1001`; the W2 public
  RNG is not advanced.
* Role projections, bond projection, path MLP and gate use the historical
  uniform-linear / ones-zero LayerNorm rule.
* `Output Linear(65,128)` weight and bias start at zero.
* Therefore epoch-0 `CPC logits == historical untrained W2 logits` (tolerance
  `1e-4`; CPU is checked exactly).
* All W2 and CPC parameters are optimised jointly from step 1; nothing is
  frozen.  Zero output means only the output layer receives gradient at step 1;
  steps 1/3/10/100 are audited for every new tensor.

## 3. Correctness and mechanism gates (before formal training)

1. Path enumeration on the 9 required small graphs (single atom, single bond,
   ³-chain, ⁴-chain, triangle, square, branched tree, multi-path ring, K4)
   matches the brute-force reference, including multiplicities and reversed
   directions.
2. Full-cache per-root count, endpoint multiset, bond endpoints and
   molecule membership match the frozen S2 representation.
3. Atom renumbering and physical-bond direction reversal leave logits invariant
   (1e-4).
4. Two molecules in one batch are isolated; perturbing one changes only its
   logits.
5. Zero-path atoms have exactly zero `delta`.
6. W2 part bitwise identical to the seed-0 constructor; total parameters exactly
   1,043,333; task order unchanged.
7. Non-additive pairing witness: two path sets with identical root/middle/end
   marginals, path counts and bond features, differing only in `(middle,
   endpoint)` pairing, must be equal under an additive reference and **different**
   under the real CPC token/aggregation with a deterministic non-degenerate
   weight draw.  The difference must come from the pairing, not counts or
   renumbering.
8. Non-recursive dependency audit with real hooks/counters: `_encode`, `local`,
   `drop`, `CPC`, `f`, `rho`, readout and `post_local` each called exactly once;
   CPC reads the same `h0`/`b`; `h_cpc` enters `post_local` once; changing a
   remote endpoint's `h0` changes the root `delta` directly.
9. Pairing shuffle stays within root/molecule; the shuffle changes logits.
10. Gradients finite; every new tensor receives a non-zero gradient by step 100.
11. Masked BCE, NaN mask, official AP and 128-task order match the inherited
    implementation.
12. CPU/CUDA outputs agree within a reasonable tolerance; all test graphs finite.

Failure of structure, pairing witness, non-recursive dependency or data identity
blocks formal training.  The model definition and witness conditions are never
silently changed to make a gate pass.

## 4. Single formal training run

* `ogbg-molpcba`, frozen cache, official scaffold split.
* Train 350,343 / Valid 43,793 / Test 43,793; 128 tasks; seed 0; 40 epochs.
* Batch 128; AdamW; weight decay 1e-3; betas (0.9,0.999); eps 1e-8.
* 1-epoch linear warmup 3e-5 → 3e-4, then per-step cosine 3e-4 → 1e-5 over the
  40-epoch horizon (109,520 updates).
* Dropout 0.10 (inherited W2 `drop`), gradient clip 5.0, FP32, masked BCE.
* The historical W2 seed-0 40-epoch batch-order plan:
  `global_plan_sha256 = 06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6`,
  consumed-prefix `1a24dfca…`, 2,738 steps/epoch.
* One parameter group; W2 and CPC share the same LR.  No freeze phase, no
  group-specific LR, no task weights.
* Official Validation AP every epoch; save `checkpoint_best.pt` (strict argmax,
  earliest wins ties), `checkpoint_epoch40.pt`, and a resumable
  `checkpoint_last.pt`.
* Record per-epoch Train BCE, Valid AP/BCE, step timings, GPU memory, gradient
  norms of every CPC module, CPC module weight norms, RNG hashes; full
  per-task Validation AP; final-Train AP at epoch 40.
* Not permitted: changing data, features, mask, loss, optimizer, LR schedule,
  task weights, readout, epochs or selection rule in response to intermediate
  Validation AP.

## 5. Branch usage and post-training mechanism diagnostics

Every epoch, on one fixed 32-row official-Validation diagnostic batch in eval
mode:

```text
R = RMS(delta) / RMS(h0)
```

plus path-token RMS, context RMS, gate mean/std/quantiles, per-root gate
dispersion, zero-path atom fraction, mean/p95/max paths per atom, finite
fractions, and CPC module gradient/weight norms.  All 18 new tensors are
tracked; no giant path-token artifact is saved.

After training, on the frozen best-Validation checkpoint, Validation only:

* real CPC;
* `delta` forced to zero;
* middle–endpoint pairing shuffled within each root, holding root indices,
  path counts, bond multiset and all inputs fixed; re-verify no cross-molecule
  pairing.

These interventions create out-of-distribution inputs.  Their AP drops measure
sensitivity to the trained CPC path; they are **not** a causal estimate of CPC's
independent gain over W2 and are never used for checkpoint selection.

## 6. GPU budget and stop rules

* Host `res`, one idle A100 (`res-gpu1`), one job at a time.
* Target ≤ 2.8 A100 GPU-hours; hard cap 3.5.  Structure cache target ~1 GiB.
* A CPU enumeration/cache profile and a real 100-batch GPU forward/backward,
  memory and validation-timing profile precede formal training.
* Extra gates, mechanism inference and Test inference count against the ledger.
* If the projection exceeds the hard cap, only numerically equivalent
  engineering optimisation is attempted; otherwise the round reports
  `RESOURCE_STOP`.
* Forbidden shortcuts: fewer epochs, sampled/truncated paths, fewer training
  rows, width changes, second seed, architecture changes.
* No automatic "if AP is low, change architecture and rerun" permission.  The
  round ends after the single formal run, freeze and report.

## 7. Selection, Test freeze and Test scoring

The official Test split is already exposed.  Role:
`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`.  It is never a model-selection signal
and never a blind generalisation estimate.

1. Train 40 epochs; select by official Validation AP (earliest wins ties).
2. Freeze the best-Validation checkpoint; retain the epoch-40 checkpoint.
3. Verify checkpoint SHA256, parameter audit and Validation replay.
4. Test inference with **labels still NaN**; save logits, row IDs and SHA256.
5. Seal all prediction hashes before reading any Test label.
6. Only then call `cc_data.read_official_test_labels` through the inherited
   gate; verify the Train/Valid label alignment and Test row identity.
7. Score every model with `ogb.graphproppred.Evaluator("ogbg-molpcba")` plus an
   independent per-task sklearn AP recomputation.
8. Reuse the frozen historical W2 / RC-REAL / W2-DYNAMIC logits and re-score
   them with the same code path.

No Test-driven change to model, checkpoint, gate, threshold, seed or training
plan is permitted.

## 8. Primary metrics and decision rule

Report:

* official full-127-task macro AP;
* fixed epoch-40 Validation AP and best-epoch Validation AP;
* the common high-support set (≥20 Validation **and** Test positives; 106 tasks);
* CPC − W2 and CPC − W2-DYNAMIC per-task differences;
* improved/degraded task counts;
* Train/Validation gap;
* GPU cost and parameter count.

Recovery ratios use the historical best checkpoints:

```text
R_valid = (Valid_AP_CPC - Valid_AP_W2) / (Valid_AP_DYNAMIC - Valid_AP_W2)
R_test  = (Test_AP_CPC  - Test_AP_W2)  / (Test_AP_DYNAMIC  - Test_AP_W2)
```

with `Valid_AP_W2=0.265273`, `Valid_AP_DYNAMIC=0.275067`,
`Test_AP_W2=0.253617`, `Test_AP_DYNAMIC=0.262226`.  The 75% targets are
`Valid ≥ 0.27262` and `Test ≥ 0.26007`; they are never redefined after seeing
the result.

Pre-registered labels:

* **`CPC_MATCHES_DYNAMIC_RANGE`** — `R_valid ≥ 1.0` and `R_test ≥ 1.0` and
  `≥70/106` high-support tasks improve vs W2, with the CPC branch demonstrably
  active.
* **`CPC_RECOVERS_MOST_DYNAMIC_GAIN`** — `R_valid ≥ 0.75`, `R_test ≥ 0.75`,
  and `≥70/106` high-support tasks improve.
* **`CPC_PARTIAL_GAIN`** — not "most", but CPC is above W2 on both Validation and
  exposed Test, at least `53/106` high-support tasks improve, and the
  high-support mean difference vs W2 is positive (or the Validation/Test
  direction is inconsistent).  Report the distribution; run no extra experiment.
* **`CPC_NO_CLEAR_GAIN`** — no clear, broad gain, or gains dominated by sparse
  tasks.
* **`CPC_MECHANISM_INVALID`** — structure/index, pairing, gradient, non-recursive
  mechanism or data-identity audit failed.

No label is a statistical-significance claim.  Near-boundary results are
reported with exact numbers and uncertainty rather than forced into a binary
success claim.
