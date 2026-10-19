# PROTOCOL — zinc bond-conditioned context attention v1

| field | value |
|---|---|
| protocol id | `zinc-bond-conditioned-context-attention-v1` |
| round directory | `tracks/structural-sufficiency/results/zinc_bond_conditioned_context_attention_v1/` |
| code | `tracks/structural-sufficiency/code/zinc_bcca_v1/` |
| config | `tracks/structural-sufficiency/configs/zinc_bond_conditioned_context_attention_v1.yaml` |
| status at freeze | `FROZEN_PRE_GPU` |
| kind | attention feasibility + mechanism screen (not a confirmatory round) |
| relation to Phase 2b | independent side round under the same structural-sufficiency question; **not** Phase 2c; no Phase 2b file is modified |
| official test | never loaded, in any stage, by any code path in this round |
| valid labels | single read, at the `score` stage only, after every prediction and diagnostic is frozen |

This document is frozen **before** the first GPU second of this round and **before** any new
official-valid score. It is written to be checkable against the code: every numeric constant
below also appears in the frozen YAML, and the CPU gates assert the agreement.

---

## 1. Question

The frozen P3 object encoder (`BJOC-Null`, 413,505 parameters) reads each physical bond's
local context as two **Null-grouped moments** over the junctions attached to that bond:

```
W_e = n_e * mean_{j in J_e} w_j          V_e = n_e * mean_{j in J_e} w_j^2
```

`mean` is uniform. Every bond in a molecule therefore sees the same *pooling rule*, and the
same junction contributes to every incident bond with the same weight. The question of this
round is narrow:

> On top of the **same frozen P3 encoder and the same R3 recipe**, does replacing that uniform
> within-bond pooling by a **learned, per-bond conditional allocation** over the bond's own
> junctions produce an official-valid MAE gain that is large enough, consistent enough across
> seeds, and specific enough to the conditioning (rather than to added parameters or to a
> global reweighting) to justify further work on adaptive object composition?

Three arms separate the confounds. All three add exactly 6,144 parameters and start from a
bit-exact copy of the same P3 weights, so any difference is attributable to *where the new
capacity sits*, not to how much capacity exists.

* **C** — per-bond conditional attention over the bond's own junctions.
* **G** — the *same* attention scores, marginalised into one distribution per molecule and
  shared by all bonds of that molecule (the "global reweighting" control).
* **F** — no change to context at all; the 6,144 parameters are an FFN on the composer input
  (the "capacity" control).

The P3 reference is the frozen historical pair of soups, not a retrain, unless the execution
regime audit fails (§6.5).

### 1.1 Pre-registered propositions

| id | proposition | falsified if |
|---|---|---|
| H-perf | C improves official-valid MAE over P3 by a round-relevant margin | §8 thresholds not met |
| H-cond | C beats G, i.e. conditioning on the bond matters beyond reweighting junctions | C−G gain < 0.002 mean, or not positive on both seeds, or paired CI lower bound ≤ 0 |
| H-cap | C beats F, i.e. the gain is not generic capacity | C−F not positive on both seeds, or paired CI lower bound ≤ 0 |
| H-inert | the attention actually moves (not numerically dead) | D, entropy or prediction-change criterion in §10 says inert |

H-perf is the primary axis. H-cond and H-cap are **specificity** axes: without them a positive
H-perf is read as "more capacity helped", not as "conditioning helped".

---

## 2. Frozen substrate and lineage

P3 is the `Null` object-composition arm of the closed Phase 2b round: `BJOC-Null`, trained with
recipe R3 from a seed-matched **full-10k masked-SSL** encoder, soup of the last 5 epochs. The
historical official-valid MAE of the frozen soups is

| seed | official-valid MAE |
|---|---|
| 0 | 0.07910149778419873 |
| 1 | 0.08058639967715135 |
| mean | 0.07984394873067505 |

Frozen inputs (SHA256 verified by the `audit` stage, values in the YAML):

* `objects_train10k.pt` `14326a7a…`, `objects_valid.pt` `61fca36a…`
* `prep_train10k.json` `d0fb86a3…`, `encoded_train.pt` `10d76495…`
* `pretrain_encoder_s0.pt` `006fabc1…`, `pretrain_encoder_s1.pt` `ff8d6474…`
* `soup_R3_seed0.pt` `a021d36e…`, `soup_R3_seed1.pt` `b622824a…`
* frozen model/object sources `structural_relation_composer_v1` `1c83a97e…`,
  `structural_relation_objects_v1` `8f4ba44d…`, `bjoc_masked_pretraining_v1` `f27dbfe3…`,
  `phase2a_models` `f74c9691…`

The audit recomputes every hash, rebuilds P3 for both seeds from scratch, checks that the
rebuilt initial state, group state and encoder state match the frozen R3 records, **replays the
frozen soups on the 1,000 valid molecules without reading a label**, and requires the replay
predictions to agree with the historical arrays `valid_R30.npz` / `valid_R31.npz` to ≤ 5e-5.
The replay itself is frozen as an artefact (`p3_valid_replay.npz`); the historical MAE is
recomputed from it and must reproduce the table above to ≤ 1e-5 at score time.

Nothing in this round re-tunes P3, its recipe, its encoder or its soup selection.

---

## 3. Definitions

For a molecule `g` with atom features, bond features and a junction set `J_g`:

* `E_g` physical bonds; each bond `e = (u, v)`.
* `J_e ⊂ J_g` the junctions incident to `e`; `junc_bond` is the incidence list.
* `n_e = deg(u) + deg(v) - 2 = 2 |J_e|`. This is the frozen P3 normaliser, **not** `|J_e|`:
  each junction is counted once from each endpoint, so `n_e` is the number of endpoint
  incidences. The implementation checks `n_e == 2 * |J_e|` on the frozen data as a gate.
* Null pooling: `W_e = n_e * (1/|J_e|) Σ_{j∈J_e} w_j`, `V_e = n_e * (1/|J_e|) Σ_{j∈J_e} w_j^2`.
  Both are invariant to how many junctions a bond has; `n_e` only rescales.

The composer input for bond `e` is the frozen five-slot vector
`b_e = [h_u, h_v, e_uv, W_e, V_e]` (dim 96 each after the frozen projections, 321 total
before the 32-d reader). This round only ever changes which `W_e, V_e` are written into the
frozen `NULL` slots, except for F which changes the composer's `b` slot only.

---

## 4. Arms

### 4.1 Shared attention module (C and G only)

```
x_e = LN(b_e)              LN(96, eps=1e-5, elementwise_affine=False)
q_e = Q x_e                Q: 96 -> 32, no bias
k_j = K LN(w_j)            K: 96 -> 32, no bias, w_j the junction embedding
s_ej = q_e . k_j / sqrt(32)    one head, temperature sqrt(32)
A_e  = softmax_{j in J_e} s_ej  (segment softmax; strictly within the bond's own molecule)
```

* The softmax is a **segment softmax over `J_e` only**. A bond never attends to another
  molecule, and (by construction) only to junctions incident to itself. Implemented with a
  dense `[n_bonds, n_junctions]` block-diagonal mask built from `junc_bond`, masked fill
  `-inf`, then `softmax` and `masked_select`, or an equivalent segment reduction; the audit
  uses the dense form and the training path uses the segment form, and G6 checks they agree.
* Empty junction set or zero bonds returns exact zeros, no NaN (§G4 of the CPU gates).
* Dropout 0, one head, no biases anywhere.

### 4.2 C — bond-conditional context

```
W_e = n_e * Σ_{j∈J_e} A_ej w_j          V_e = n_e * Σ_{j∈J_e} A_ej w_j^2
```

Replaces the two `NULL` slots; everything else in the encoder and composer is frozen P3.

### 4.3 G — global marginal context (control for "any reweighting")

Same `A` from the same module and weights, marginalised **per molecule**:

```
p_j = ( Σ_{e ∈ E_g} n_e A_ej ) / ( Σ_{e ∈ E_g} n_e ),   j ∈ J_g
W_e = n_e * Σ_{j∈J_g} p_j w_j          V_e = n_e * Σ_{j∈J_g} p_j w_j^2
```

Every bond in a molecule therefore receives the same context. This is the *minimal* way to
keep the learned scores while removing the per-bond conditioning; it is not a constant across
the dataset, and it is computed from the same `A` (not from a separate parameter set). G is
the correct control for "is it the per-bond allocation, or just a learned reweighting?".

### 4.4 F — composer-input FFN (control for capacity)

```
b_e^F = b_e + U SiLU(T LN(b_e))         T: 32 x 96, U: 96 x 32
```

`W_e, V_e` remain the frozen P3 Null moments. This is the same 6,144-parameter budget placed
where it cannot condition anything on the junction set; it can only transform the composer
input non-linearly.

### 4.5 Why G and F are the right controls

* C vs G differ **only** in the marginalisation axis (per-bond vs per-molecule) with identical
  parameters, identical `A`, identical training data. A C−G gain cannot be explained by extra
  capacity, extra depth, or a different attention mechanism.
* C vs F differ **only** in where the new capacity sits (context vs composer input). A C−F gain
  cannot be explained by "any 6k parameters help".
* P3 vs all three is the reference for whether *any* of this is worth the cost.

---

## 5. Parameters and initialisation

| arm | new parameters | total | relative |
|---|---|---|---|
| P3 | 0 | 413,505 | — |
| F | 6,144 | 419,649 | +1.4858% |
| G | 6,144 | 419,649 | +1.4858% |
| C | 6,144 | 419,649 | +1.4858% |

* `Q` 3,072 + `K` 3,072 = 6,144 for C/G; `T` 3,072 + `U` 3,072 for F. No biases, no extra
  reader channel, no extra depth, no gate, no positional / distance / ring / ownership bias.
* Build order per arm and seed:
  1. `bp.build_bjoc_null(seed)` (the frozen constructor) and load the seed-matched full-10k
     masked-SSL encoder;
  2. hash the initial full state, group state and encoder state; compare with the frozen R3
     records;
  3. save the global RNG state, construct the extension modules, restore the global RNG state
     (so the extension cannot perturb the shared init or the batch-order stream);
  4. `copy_shared_` every shared tensor from the base, tensor by tensor;
  5. initialise the extension: `Q = 0`, `U = 0`; `K` and `T` are the **same** draw
     `uniform(-1/sqrt(96), +1/sqrt(96))` from `torch.Generator().manual_seed(seed + 9119)`;
     `F_T` is placed from that same matrix, not redrawn.
* With `Q = 0` and `U = 0`, every arm is an **exact P3 function at step 0** (checked to ≤ 5e-5
  absolute on the prediction, ≤ 1e-5 relative on W/V). Training therefore starts on the P3
  solution, not on a random perturbation of it; differences at the end are not init noise.
* `K` is nonzero at step 0 so the score path is well-defined; the first update gives `K` zero
  gradient (softmax of a constant score has zero derivative w.r.t. the scores up to the
  row-sum-zero structure), and `K` becomes nonzero from step 2. This is expected and gated
  (G5), not a bug.
* No P3 soup / no warm start from Phase 2b finalists; the encoder is the historical full-10k
  masked-SSL checkpoint for that seed, exactly as in R3.

---

## 6. Data, recipe, execution

### 6.1 Data

* Official PyG ZINC `subset=True`: **10,000 train**, **1,000 valid**.
* The valid split is the historical development split used for every prior BJOC number. It is
  **not** a blind test and no claim of test-set generalisation is made. It is used here only
  as a screening endpoint under the single-read rule.
* Official test: not loaded, not parsed, not predicted. No stage in the runner has a test
  code path (AST-scanned by G8).
* Target standardisation is the frozen train-only scaler; loss is raw-y L1 (the historical R3
  objective). Cycle penalty is a **reporting-only proxy** for the internal monitor
  (`max cycle-basis length > 6`); it is not an input, not a loss term and not used for
  selection.

### 6.2 Recipe R3 (verbatim, no early stopping)

| item | value |
|---|---|
| epochs | 360 |
| batch size | 128 |
| steps / epoch | 79 (last batch 16) |
| updates | 28,440 |
| optimiser | Adam, lr 1e-3, step-cosine to 1e-5 |
| weight decay | 1e-5 |
| grad clip | 5.0 |
| loss | raw-y L1 |
| soup | mean of the last 5 epoch states |
| batch order | `torch.Generator().manual_seed(seed + 101)`, one permutation per epoch |
| early stopping | none |
| valid-based selection | none |
| precision | FP32, no AMP, no `torch.compile` |

The batch-order hash for both seeds is recomputed from the generator and must equal the frozen
R3 hash (`3c964a71…` for seed 0, `8710751a…` for seed 1) before any training starts. The
training loop is parity-tested against the frozen `bjoc_perfmech_long_v1.train_supervised`
on a synthetic slice: identical state after 2 updates (≤ 1e-4), identical batch-order hash,
identical update count, soup-fit MAE within 1e-5 (G7).

### 6.3 Run matrix and GPU discipline

| GPU | order |
|---|---|
| `res-gpu0` | F0 → G0 → C0 |
| `res-gpu1` | C1 → G1 → F1 |

* Seeds are reversed across GPUs so that arm order is not confounded with load drift.
* At most one training process per GPU; all remote execution goes through `rr` (no hand-written
  SSH / `nohup`), host `res` (not `res-2`).
* Every run is a separate `rr run … --pool res-gpuN` invocation; the runner refuses to rewrite
  an existing artefact, so a rerun requires an explicit new attempt directory and is charged
  to the budget.

### 6.4 Budget

* Hard cap: **4.0 A100 GPU-hours**, summed over both GPUs, including smoke, failures, reruns
  and any P3 baseline retraining that §6.5 would require.
* A 2-epoch train-only smoke on 1,024 real training molecules measures per-update time for
  F, G and C. Projection:

  `projected = 2 * (t_F + t_G + t_C) * 28,440`

  and the matrix may start only if `1.20 * projected ≤ 4.0 h` (the 1.20 covers soup,
  validation-free diagnostics and process overhead). If not, the round stops and reports the
  projection; no arm is trained.
* Smoke runs are budget-charged. The smoke record includes the observed seconds and the
  projection; the ledger is appended after every `rr` job.

### 6.5 P3 provenance branches

The audit compares the observed execution regime (GPU name, driver, torch, CUDA, Python)
with the regime recorded for the frozen P3 soups.

* **Branch A (expected)** — regime matches. The historical soups and their historical valid
  predictions are reused as the P3 reference; the replay must agree with the historical arrays
  to ≤ 5e-5, and the recomputed historical MAE must reproduce the §2 table to ≤ 1e-5.
* **Branch B (regime mismatch)** — two same-round P3 baselines (seeds 0, 1) must be trained
  with R3 **before** the valid label read, and the P3 reference becomes those soups. This is
  budget-charged and may by itself exhaust the cap; in that case the round reports the
  blocked state rather than training fewer arms or dropping seeds.

The audit record states which branch applies (`regime.p3_baseline_required`). The runner
refuses to train a P3 baseline unless Branch B is recorded.

---

## 7. Freeze discipline and the label gate

The valid labels are the scarce resource. The order is fixed:

1. `prep` — audit (hashes, replay, provenance) → CPU gates → FLOPs.
2. `smoke` — 2-epoch timing on real training molecules; projection vs cap.
3. 6 formal trainings (`F0,G0,C0,C1,G1,F1`), each writing `train_<arm>_s<seed>.json` and
   `soup_<arm>_s<seed>.pt`.
4. `predict-valid` for each of the 6 (model reloaded from the soup, prediction written to
   `predictions/valid_<arm>_s<seed>.npz`, then frozen with `valid_freeze_<arm>_s<seed>.json`).
5. `mechanism` — label-free diagnostics: C/G forward passes (same weights), C in `uniform`
   mode, G in `uniform` mode, D / entropy / max weight, W/V dispersion, C→G moment
   conservation, forced-uniform prediction change, inertness flags. Written to
   `mechanism_this_round.json` + `predictions/mech_pred_*.npz`.
6. **Label gate** — a single `score` invocation:
   * verifies the mechanism record exists and every frozen prediction file + hash is present;
   * reads `valid_y_read_once.npy` **once** (creates `valid_read.lock`; a second `score`
     invocation refuses to run unless `score --recover` is passed, which writes
     `score_recovery_deviation.json` and is reported as a deviation);
   * recomputes the P3 historical MAE from the frozen replay and checks ≤ 1e-5 agreement;
   * writes `valid_metrics.json` and `valid_decision.json`.

Before step 6 no valid label array is opened by this round's code. G8 AST-scans the train,
predict, mechanism and smoke stage functions to prove that none of them can call a valid- or
test-label loader, and the label gate is exercised in the gate stage to prove it raises.

No re-scoring, no "just one more look", no seed/arm/architecture changes after step 6. Any
bug found after the label read is reported as a post-hoc deviation, not silently fixed.

---

## 8. Endpoints, statistics and thresholds

**Primary endpoint**: official-valid MAE of the frozen soup, computed per seed and then
averaged over the two seeds (`mean of per-seed MAEs`, **no prediction ensembling** across seeds
or arms). MSE and the tail breakdown are secondary.

**Paired statistics**: for a contrast (e.g. C−P3) the per-molecule absolute-error difference is
bootstrapped over molecules (5,000 resamples, seed 20261019, percentile 95% CI). The reported
`mean_gain` is positive when the first arm has lower error. Per-seed gains are reported
separately and both are required for a "promising" label.

**Robustness set**: (a) all 1,000 molecules ("ordinary"); (b) the 1,000 minus the 10 molecules
with the worst mean P3 error ("drop P3-worst-10"). The extreme valid row 172 is reported
explicitly but never removed silently.

**Pre-registered thresholds** (round-internal, not universal constants):

| label | condition |
|---|---|
| `PERFORMANCE_PROMISING` (for arm X) | X−P3 mean gain ≥ 0.004 **and** positive on both seeds **and** paired 95% CI lower bound > 0 **and** the gain survives both robustness sets |
| `CONDITIONING_PROMISING` | C−G mean gain ≥ 0.002 **and** positive on both seeds **and** paired 95% CI lower bound > 0 (the C−G robustness split is reported descriptively, not required for the label) |
| `CAPACITY_CLAIM` (C better than F) | C−F positive on both seeds **and** paired 95% CI lower bound > 0; no magnitude threshold |

A gain smaller than 1e-5, or one that changes sign between the ordinary and robust sets, is
reported with its margin and is **not** called robust. Thresholds are not changed after seeing
the numbers.

---

## 9. Decision labels (multi-axis; first label is primary)

The `score` stage emits labels from the fixed code path, not by hand:

| label | meaning |
|---|---|
| `A_CONDITIONAL_ALLOCATION_PROMISING` | C passes H-perf **and** H-cond **and** H-cap |
| `B_GLOBAL_REWEIGHTING_PROMISING` | G passes H-perf, but conditioning/capacity specificity does not |
| `C_CAPACITY_GAIN_DOMINATES` | F passes H-perf, but conditioning/capacity specificity does not |
| `D_ATTENTION_ACTIVE_NO_USEFUL_GAIN` | attention is not inert, but no arm passes H-perf |
| `E_ATTENTION_NUMERICALLY_INERT` | both C and G are inert by §10 |
| `F_INCONCLUSIVE` | seed sign conflict, or no other label applies |

The report answers the task's five questions in this order: best real valid number vs P3 (per
seed / mean / CI / robustness); C vs G and vs F with attribution; whether C is a better
follow-up candidate than P3 and at what cost; what is still missing for a publishable claim
and one next step; which checks were completed and which were not, and why.

---

## 10. Mechanism diagnostics (label-free, before the label gate)

All diagnostics are computed from the frozen C and G soups, with no valid label and no test
data. They describe **response magnitude**, never performance, chemistry or faithfulness.

* `D = Σ_e n_e ||A_e − p||_1 / Σ_e n_e`, per molecule and molecule-weighted, where `p` is the
  graph marginal of the same `A`. `D = 0` means the conditional allocation is indistinguishable
  from the global reweighting.
* Normalised attention entropy `H(A_e) / log(max(|J_e|, 2))`, mean over bonds; 1 = uniform.
* Mean maximum attention weight per bond; `1/|J_e|` = uniform.
* Across-bond dispersion of `W_e / n_e` and `V_e / n_e` within a molecule (std / mean of the
  vector norm). The G control must be ≈ 0 (its content is constant per molecule by
  construction); C must be larger for the attention to be doing anything.
* C→G same-weight checks: moment conservation relative error, and the prediction change when C
  is forced to `uniform` and when C is evaluated in `G` mode with its own weights (and G in
  `uniform` mode). This is the label-free version of "does the allocation matter at all?".
* Inertness criteria (frozen): `D ≤ 0.02` **and** mean normalised entropy ≥ 0.99 **and**
  |Δprediction| ≤ 5e-4 under forced uniform ⇒ `attention_inert = true` for that arm. Inertness
  is a statement about the trained weights, not about the arm's design.

The mechanism record is written and hashed before the label gate opens. If an arm is inert,
the report says so and does not interpret its MAE as evidence about conditioning.

---

## 11. FLOPs and cost accounting

`flops` reports analytic GEMM FLOPs on a real train batch of 128 molecules plus analytic pair
counts (score MACs, weighted first/second-moment MACs, marginal reduction adds). The CPU wall
clock in that record is indicative only and is **not** a GPU estimate; the GPU projection comes
from `smoke`. Expected order: F ≈ 1.02×, G/C ≈ 1.024× P3 GEMM FLOPs; the attention pair work
is negligible next to the junction encoder.

The GPU ledger records, per `rr` job: stage, arm/seed, pool, start/end UTC, wall seconds, and
the artifact hashes produced. The budget line is the sum of wall seconds over both pools,
converted at the pool's billed rate, and is compared with the 4.0 h cap after every job.

---

## 12. Deviations and amendments

* The PROTOCOL, the YAML and the decision code are committed before the first GPU run. Any
  change before the label gate is an **amendment**: appended here with a UTC timestamp, the
  reason, and the affected artefacts; the old config hash is recorded.
* Any change after the label gate is **post hoc**: reported in the REPORT under a separate
  heading and excluded from the pre-registered decision.
* The only built-in recovery is `score --recover` after a crash that consumed the read but
  wrote no decision; it increments `read_count`, writes `score_recovery_deviation.json` and
  must be disclosed in the REPORT.
* Empty at freeze.

---

## 13. Claims policy and prior art

This round does **not** claim a new attention mechanism. Bond-conditioned / edge-conditioned
attention and adaptive neighbourhood aggregation have prior art, including (non-exhaustive)
SAT, GPM, ESA, GRIT, SACA, DGT and the Cardinality-Preservation line of work. The contribution
under test is narrow: *within the frozen BJOC object abstraction, is per-bond conditional
allocation over the bond's own junctions better than the frozen Null moments, better than a
global marginal reweighting, and better than the same parameter budget spent on the composer
input?*

The round reuses P3's existing deterministic structural features as-is. It does **not** add
RWSE, ring encodings or any new structural feature, and the report must not repeat the
Phase 2b "no ring features" misstatement: the frozen BJOC objects already contain deterministic
ring/degree/pair-distance structure, and this round changes none of it.

No claim about official-test generalisation is made, because official test is never touched.
No claim about chemistry, interpretability or faithfulness is made from the attention weights;
`D` and the entropy are response-magnitude diagnostics only.

---

## 14. Artefact manifest (written by the round)

```
p3_reuse_audit.json                 frozen-input / init / provenance audit
cpu_gates_this_round.json           G1–G9
flops_this_round.json               analytic FLOPs
smoke_this_round.json               2-epoch timing + budget projection
train_{F,G,C}_s{0,1}.json           per-run record (hashes, timings, grad probes, soup)
soup_{F,G,C}_s{0,1}.pt              frozen soups
predictions/valid_{F,G,C}_s{0,1}.npz
valid_freeze_{F,G,C}_s{0,1}.json    prediction hashes
predictions/p3_valid_replay.npz     label-free P3 replay (Branch A)
mechanism_this_round.json           label-free diagnostics + inertness
predictions/mech_pred_C_s0_{C,G,uniform}.npz   (and s1; G likewise)
valid_read.lock                     single-read marker
valid_metrics.json                  per-arm metrics, contrasts, robustness, tails
valid_decision.json                 thresholds, labels, primary label
gpu_ledger.json                     rr job ledger
REPORT.md                           final write-up
```
