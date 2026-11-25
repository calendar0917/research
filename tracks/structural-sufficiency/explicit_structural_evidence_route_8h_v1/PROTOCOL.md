# PROTOCOL — explicit structural evidence route (8 h round)

Round: `explicit-structural-evidence-route-8h-v1` · T0 `2026-11-25T16:04:18Z`
Wall-clock hard stop `T0+8h` (`2026-11-26T00:04Z`) · GPU soft cap 14 h, hard cap 16 h.

This file freezes the scientific design *before* the first formal GPU step. It
is written together with `configs/experiment.json`,
`configs/run_matrix.json`, `analysis/schedule.json` and recorded in
`analysis/protocol_freeze.json` (SHA-256 of each file plus of every code module).
No arm, width, token function, aggregation, recipe, evaluation epoch or
promotion rule may change after seeing a Validation score.

---

## 1. Question

Can a **non-recursive** graph prediction function that computes directly on
explicit structural objects — local rooted structures and their shortest-path
relations — be *predictively competitive* and at the same time keep an exactly
reconstructable prediction account?

There is no iterative message passing, no distilled teacher, no ensemble, no
EMA/SWA/soup, no extra supervision, and no new literature hypothesis in this
round. Candidate models start from random initialisation; historical checkpoints
are read-only performance references.

Sub-questions: (Q1) direct structural prediction, (Q2) nonlinear evidence
aggregation, (Q3) task-projection position, (Q4) relation interaction placement,
(Q5) performance/interpretability relationship, (Q6) contribution regularities.

## 2. Data (frozen splits, no regeneration)

| dataset | split | Train / Valid / Test | primary metric |
|---|---|---|---|
| `ogbg-mollipo` | official OGB scaffold | 3360 / 420 / 420 | raw RMSE |
| `ogbg-molesol` | official OGB scaffold | 902 / 113 / 113 | raw RMSE |
| Tox21 | frozen ChemBench split of `lcrc_predictive_trace_benchmark_v1` (via `lcrc_molnet_benchmark_pilot_v2`) | 6264 / 783 / (frozen) | observed-label macro ROC-AUC (12 endpoints) |
| `ogbg-molpcba` | official OGB scaffold | 350343 / 43793 / 43793 | official OGB macro AP |

Inherited caches/labels, read-only:

* Lipo: `lcrc_lipo_context_small_s0_v1/cache/mollipo_{train,valid,test}_v1.pt`,
  `data/labels_v1.npz`, `data/split_v1.json`;
* ESOL: `lcrc_explicit_semantic_learning_8h_s0_v1/data/esol_{raw,labels,split}_v1*`
  (label-free raw artifact, built once with RDKit; GPU hosts have no RDKit);
* Tox21: `lcrc_molnet_benchmark_pilot_v2/cache/Tox21_{train,valid,test}.pt`;
* PCBA: `lcrc_pcba_scale_pilot_v1/cache/pcba_v1`, split/labels/order plan through
  `sg_data.load_task`;

Data isolation: structural normalisation, target scaler and the two auxiliary
global statistics (`log1p n_atoms`, `log1p n_relation_pairs`) are fitted on
**Train rows only**. Training views expose `train`/`valid` only; Test is touched
exactly once by `code/esr_score_test.py` after every decision is frozen.
**PCBA Test cannot be scored**: the frozen label artifact is NaN on every Test
row and the loader refuses to continue otherwise. No PCBA Test number is
produced; historical Test AP values are quoted as references only.

All Test numbers of this round carry the label
`EXPOSED_TEST_ENGINEERING_COMPARISON_SET` (historical rounds already read them).

## 3. Shared structure encoder (`h0`)

Reused unchanged from LCRC-A / `BSModel` (`lcrc_pcba_backbone_scaling_v1`):

```
a, b, scal = base._encode(batch)
h0 = base.drop(base.local(batch, a, b, scal))       # exactly one call per forward
```

* small datasets: `config_for_scale(1)` (`width=64`, full width table inherited);
* PCBA: `config_for_scale(2)` (W2 widths, `width=128`);
* trunk initialisation: the frozen `FULL -> C1` remap of the same scale, seed 0;
* no arm reads an updated node state; `h0` is immutable for the whole forward.

`BASE` = the original `BSModel` (S1/S2 local content + `f` + `rho` + the four-head
attention readout) trained with the same recipe on the same data; on PCBA the
historical W2 checkpoint is reused read-only, not retrained. Every new arm
**deletes** `f`, `rho` and `readout`; no unused read-head parameter remains.

## 4. Relation instances and relation features

```
P(G) = {(i,j) : i != j, 1 <= shortest_path_distance(i,j) <= 2}
```

Each unordered pair exactly once; no sampling, no truncation, no atom-id
selection, no cross-molecule or cross-component pairs, no multi-path double
counting. `D=4` is used only by the conditional `BEST_EXPLICIT_RANGE4` arm.

Relation vector (19 dims): distance one-hot over `SELF,1,2,3,4` (5) +
`log1p(#common neighbours)` (1) + the **cached raw 13-d bond features** of the
direct physical bond (zeros when not directly bonded). Pair indices, distance
and common-neighbour counts are cached once (`cache/pairs/...`, label-free,
memory-mapped); the training loop never runs a per-molecule Python forward.

Local and relation tokens (`K=48` small, `K=64` PCBA):

```
u_i  = LN(Linear(2K,K)(SiLU(Linear(d,2K)(h0_i))))
x_ij = [u_i+u_j, u_i*u_j, |u_i-u_j|, r_ij]
v_ij = LN(Linear(2K,K)(SiLU(Linear(3K+19,2K)(x_ij))))
```

Both are shared by all arms; `U`/`V`/global-linear initialisation is *bitwise
identical* across arms (private generator, fixed module order — checked in
`analysis/checks.json`).

## 5. Evidence aggregation (frozen)

`tau = 1`, no temperature search.

```
Mean(s) = sum(s)/m
LME(s)  = tau * log(mean(exp(s/tau)))          (stable segment log-sum-exp)
D_pos   = LME(s) - Mean(s) >= 0
D_neg   = LME(-s) + Mean(s) >= 0
R(s)    = pi_mean*Mean(s) + pi_pos*LME(s) - pi_neg*LME(-s)
pi      = softmax(theta),  init [0.50, 0.25, 0.25]
A(s)    = sqrt(m) * R(s)                       (all new models)
A_add(s)= sum(s)/sqrt(m)                       (ADD control)
```

Empty token set → 0; single token → `R(s)=s` exactly. No count-scaling search.
`LME(s)` is **not** a positive contribution and the mixing weights are **not**
additive attributions.

Global term (identical in all explicit arms):

```
c_t(G) = w_t^T g(G) + b_t
g(G)   = [frozen 42-d global descriptors, standardised,
          standardised log1p(n_atoms), standardised log1p(n_relation_pairs)]
```

No pooled hidden state, attention output or extra graph MLP enters `g(G)`.

## 6. Arms (frozen)

| arm | token streams | aggregation | task projection |
|---|---|---|---|
| `BASE` | LCRC-A `f`/`rho`/attention | unchanged | unchanged |
| `SREN-ADD` | local + relation | `A_add` | before aggregation (task read head per stream) |
| `SREN-EVID` | local + relation | `A` (per-task theta per stream) | before aggregation |
| `SREN-BANK` | local + relation | `A` per channel (theta per channel per stream) | after aggregation (task-linear) |
| `SREN-COMPOUND` | local + composite root token `w_i` | `A` per channel | after aggregation |
| `HIDDEN` | BANK channels | same channel pooling | hidden MLP `2K -> H -> T` replacing the task-linear map |
| `UNARY` | local + one SELF token per root (same relation MLP) | BANK aggregation | BANK-style task-linear |
| `SREN-BANK-FLEX` (D3) | BANK channels | mean / positive gap / negative gap separately | six task-linear heads |
| `REL-BLIND` (D4) | as BEST_EXPLICIT | as BEST | relation vector replaced by the molecule's pair-mean relation vector |
| `BEST_EXPLICIT_RANGE4` (D5) | as BEST_EXPLICIT | as BEST | pair set extended to `d in {1,2,3,4}` |

`COMPOUND` (per root): `C_i = sum_{j in P(i)} v_ij / sqrt(max(deg_P(i),1))`,
`w_i = LN(u_i + Linear(2K,K)(SiLU(Linear(2K+1,2K)([u_i, C_i, log1p deg_P(i)]))))`,
`logit = c(G) + q_W[t]^T A_channel(w)`. It has one reduction of pair tokens into
the root object but no recursion, no feedback into `u_i`, no adjacency update.

`HIDDEN` width `H` is chosen deterministically from `{K/2, K, 2K}` to minimise
the active-total-parameter difference to `BANK` (ties → smaller `H`), and the
audit records total/read-head parameter and compute differences. If the total
difference exceeds 5 % the arm is marked `APPROX_CAPACITY_CONTROL`.

`ADD` and `EVID` share all parameters and the bitwise identical initialisation;
the only differences are the two theta tensors and the aggregation function.

## 7. Training protocol

Small datasets (Lipo, Tox21, ESOL) — inherited R3:

* seed 0, batch 128, 360 epochs, AdamW, peak LR `3e-4`, per-step cosine to
  `1e-5`, wd `1e-3`, betas `(0.9,0.999)`, eps `1e-8`, clip 5, dropout 0.10 on
  `h0`, FP32, no warmup, no early stopping, no extra loss;
* Lipo/ESOL: MSE on standardised targets; Tox21: observed-label masked BCE with
  NaN kept as missing;
* one frozen batch order per dataset (generator `seed+917000`), shared by all arms;
* full Validation every epoch; best checkpoint = lowest RMSE / highest macro
  ROC-AUC (ties → lower BCE, then earlier epoch);
* full Train evaluation at the end (diagnostic only).

PCBA — inherited 40-epoch recipe:

* seed 0, batch 128, official Train order plan of seed 0, 1-epoch linear warmup
  `3e-5 -> 3e-4`, then per-step cosine to `1e-5` **over the 40-epoch horizon**
  (never compressed), wd `1e-3`, clip 5, dropout 0.10, FP32, masked BCE;
* formal comparison horizon = the common **25-epoch prefix** (historical W2 /
  FIXED1 prefix points are read from their curves, not retrained);
* Validation AP every epoch; full-Train AP at epochs 1, 10, 25 (and 40 when run);
* best-Validation checkpoint inside the horizon; extension to 40 epochs resumes
  model/optimizer/scheduler/RNG/plan position from `checkpoint_last`, never from
  `checkpoint_best`.

One seed, no seed averaging, no seed-variance claim. All bootstrap intervals are
sample/task uncertainty on a fixed trained model.

## 8. Phases, budgets, triggers

* **A** (≤0.7 GPU-h): data identity, Train-only normalisers, pair index vs
  independent BFS, invariance/renumbering, degenerate graphs, shared init,
  aggregation reference, empty/single/constant identities, ADD commutation,
  EVID/BANK non-commutation witness, symmetric pair witness, one `h0` call,
  gradient reach, accounting reconstruction, checkpoint replay, no Test read.
  Then throughput preflight and protocol freeze.
* **B** (~3-4 GPU-h): Lipo (7 arms), Tox21 (7 arms), ESOL (BASE, HIDDEN, the best
  explicit arm on each of Lipo/Tox21; one run if identical).
* **C** (~7 GPU-h): PCBA `ADD, EVID, BANK, HIDDEN, COMPOUND`, 25-epoch prefix.
* **D** (2-3 GPU-h, conditional): D1 `UNARY_MATCHED` (if BEST_EXPLICIT gains
  ≥0.005 AP over same-horizon W2 or reaches Level 2), D2 paired 40-epoch
  extension (Level 1 + still improving; pairs EVID→ADD, BANK→HIDDEN,
  COMPOUND→BANK, ADD→HIDDEN), D3 `BANK-FLEX` (HIDDEN ≥ BEST_EXPLICIT + 0.005 and
  HIDDEN Level 1 and no implementation confound), D4 `REL-BLIND`, D5 `RANGE4`.
  Priority order D1 > D2 > D3 > D4 > D5.

`BEST_EXPLICIT` = highest PCBA Validation AP among ADD/EVID/BANK/COMPOUND; if the
gap is ≤0.001 AP prefer fewer parameters / simpler computation in the order
ADD, EVID, BANK, COMPOUND.

Performance levels (exploratory, not significance tests): L1 = +0.005 AP vs the
same-horizon W2 reference; L2 = within 0.003 AP of historical FIXED1/DYNAMIC;
L3 ≥ 0.280 Valid AP; L4 ≥ 0.290 Valid AP.

Budget rule before each launch: `remaining_wall > 1.15 * projected_job_time +
1.0 h analysis reserve`, and `remaining_GPU_hours > 1.15 * projected_job_time`.
Degrade order: D5, D4, non-core ESOL, untriggered D branches, PCBA COMPOUND,
then (only before the first formal PCBA step) all PCBA core arms to 20 epochs.

## 9. Evaluation and reporting

* Regression: raw RMSE (primary), MAE, Train–Valid gap, paired molecule
  bootstrap interval, per-molecule |error| improvement fraction, largest-error
  sensitivity, best and late-horizon values.
* Classification: official macro AP (PCBA) or frozen macro ROC-AUC (Tox21),
  masked BCE, per-task metrics, improved/degraded task counts, median task
  delta, task bootstrap, high-support stratum
  (`valid positives >= 20 and valid negatives >= 20`) *in addition to* the full
  metric.
* Interpretability (separate section, never a substitute for performance):
  exact prediction accounting (`ADD`, `EVID` entropy decomposition, `BANK`
  channel contributions, `COMPOUND` root contributions, `FLEX` mean/gap terms);
  stream participation and intervention diagnostics; object-level zeroing and
  `h0`-zeroing; frozen-checkpoint ablations; structural rule support across
  Train molecules/scaffolds and Valid reuse; up to 8 real molecule cases.

## 10. Registered deviations from the brief

1. PCBA Test scoring is impossible (frozen label artifact NaN on Test); only
   Validation/historical Test references are reported.
2. The PCBA pair cache is one global array addressed by official row id, not one
   file per split (same objects, no sampling).
3. `UNARY` uses the BANK prediction formula for the small datasets as specified;
   on PCBA the conditional `UNARY_MATCHED` inherits BEST_EXPLICIT's aggregation
   and task heads.
4. `HIDDEN` deletes the task-linear heads entirely (they are unused once the
   hidden MLP replaces them); its parameter match is reported against BANK.
5. Training processes cannot decrypt a single-npz label file per split; they
   request `train`/`valid` only and never index `test` (statically checked).
