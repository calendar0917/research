# Composition Objective Pilot v1 — Report

## The five questions

1. **Cut-loss variance.** How much of the current cut-wise loss is cut variance
   rather than graph-level error, and is that variance structured?
2. **Graph-level supervision.** Does replacing the cut-wise main loss with
   graph-level supervision improve Valid whole-molecule prediction?
3. **Real replacement prediction.** Does it improve real cross-molecule
   replacement prediction, measured separately as whole-molecule δ and direct
   cut δ?
4. **Local context support.** Do small local contexts around the C-side
   attachment have enough cross-molecule reuse to be a better object than the
   complete C context?
5. **Follow-up direction.** Which follow-up direction does the evidence support:
   local-context object, graph-level objective, larger interaction branch, or
   something else?

## Short answers

1. **Cut variance is material but not the whole loss.** On Train, `Var_cut` is
   **40.3% / 19.5% / 30.4%** of `L_cut` for the parent FACTOR, CSOI_ADD and
   CSOI_GLOBAL arms; on Valid it is **24.7% / 13.8% / 16.7%**. The identity
   `L_cut = L_graph + Var_cut` holds per molecule to numerical precision. The
   variance is structured: it is larger for small C/S contexts in the parent,
   rises on Valid for the parent, and is only weakly-to-moderately associated
   with graph error.
2. **No.** All three graph-loss arms are point-worse on Valid whole-molecule
   RMSE. `F_GRAPH` is clearly worse than its parent (`+0.1001` RMSE,
   95% CI `[+0.0554, +0.1462]`; all intervals here are new − parent and
   positive means worse). `B_GRAPH` (`+0.0371`) and `C_GRAPH` (`+0.0412`) are
   also point-worse, but their single-seed paired CIs include zero. None
   improves Valid RMSE.
3. **No evidence of improvement.** On 434 Train→Valid pair comparisons, direct
   cut δ MAE is `0.544 → 0.572` (A→F_GRAPH), `0.470 → 0.472`
   (B→B_GRAPH), and `0.562 → 0.596` (C→C_GRAPH). Whole-molecule δ shows the
   same pattern. The 15 Valid→Valid pairs are descriptive only and unstable.
4. **Yes for radius-1/2 local contexts.** Of Valid parses, **95.7%** (r1) and
   **82.3%** (r2) have a C-side local context seen in Train; **80.4%** and
   **64.5%** have the exact (context, S-type) pair seen. For the complete C
   context the corresponding rates are **5.8%** and **0.0%**. Small contexts
   are reused across molecules; the complete C object is nearly unique.
5. **The evidence supports a local-context follow-up, not graph-level
   supervision.** The graph-level objective is harmful in this pilot and does
   not improve pair deltas. The local-context census is the only strong positive
   structural signal. The same evidence says the objective must retain a
   cut-calibration/variance term: removing it causes cancellation, not a better
   representation. No further training was done.

---

## 1. What was run

The frozen protocol is `PROTOCOL.md` (this directory). Inputs were reused
read-only from the parent and CSOI rounds; no historical file or checkpoint was
modified. The dataset is Lipo with the fixed official split
**3360 Train / 420 Valid / 420 Test**. Test labels and Test prediction arrays
were never read.

Three arms were trained for 160 epochs, seed 0, on the same data, enumeration,
model family, initialization, optimizer and CUTRESP auxiliary loss as their
parents. The **only** change was the main loss:

```python
# frozen parent/CSOI cut-wise objective (reused arms)
sq = (f_cut - y[owner]) ** 2
per_mol = scatter_mean(sq, owner)
loss_main = per_mol[:n_main].mean()

# new graph-level objective (F_GRAPH, B_GRAPH, C_GRAPH)
loss_main = mean((f_bar[:n_main] - y[:n_main]) ** 2)
loss = loss_main + 0.25 * loss_cutresp_aux
```

`f_bar` is exactly the existing `scatter_mean(f_cut, owner)` molecule
prediction; no head or architecture changed.

| new arm | family / model | parent arm | run id | best epoch |
|---|---|---|---|---:|
| `F_GRAPH` | parent `FACTOR` | `A_TRC_FACTOR_CUTRESP` | `F_GRAPH_LIPO_R0` | 31 |
| `B_GRAPH` | CSOI `CSOI_ADD` | `B_CSOI_ADD` | `B_GRAPH_LIPO_R0` | 147 |
| `C_GRAPH` | CSOI `CSOI_GLOBAL` | `C_CSOI_GLOBAL` | `C_GRAPH_LIPO_R0` | 101 |

The old arms A/B/C and the external `GINE_RAW_REF` reference are reused from
the prior rounds. Best-by-Valid was used for every arm; Test was never used for
selection. Every new manifest records
`main_loss = graph_level_mean_squared_error` and the graph-level expression;
the engineering test asserts this for all three arms. Each `curves.csv` records
Train/Valid molecule RMSE and MAE, cut-level and graph-level MSE, mean cut
variance, cut span, cancellation/mixed-sign fractions, auxiliary pair loss,
gradient norms, and epoch time.

**Execution.** Formal runs used the remote `res` host (`a100-2`, process
backend) with `res-gpu0` / `res-gpu1`, 2×A100-40GB, driver `550.163.01`,
torch `2.5.1+cu124`, CUDA 12.4, from commit `83a020c`. Total new training:
**1.006 A100 GPU-hours** (F 0.194, B 0.235, C 0.577), under the 2.0 GPU-hour
budget. Wall times were 11.7 / 14.1 / 34.6 minutes; epoch times about 4.4 / 5.3
/ 13.0 seconds. Exact `rr` run ids, commands and hashes are in
`results/provenance.json` and `results/provenance/`.

---

## 2. Q1 — How much of `L_cut` is variance, and is it structured?

### 2.1 The identity

For a molecule `G` with parse/cut set `P(G)`, owner labels `y`, cut responses
`f_c`, and `f_bar = mean_c f_c`:

```text
L_cut(G)  = mean_{c in P(G)} (f_c - y_G)^2
L_graph(G)= (f_bar - y_G)^2
Var_cut(G)= mean_{c in P(G)} (f_c - f_bar)^2
L_cut     = L_graph + Var_cut
```

This is the exact current objective: `csoi_train.py:291-294` and
`trc_train.py:296-299` compute the cut-wise mean; `csoi_model.py:434-435` and
`trc_model.py:369-370` compute `f_bar` by `scatter_mean(f_cut, owner)`. The
numbers below are from the frozen best checkpoints (A epoch 100, B 147, C 135)
on the official Train and Valid splits. Per-molecule identity residuals are at
most `2.3e-15`; see `analysis/checks.json` and
`analysis/loss_decomposition.json`.

### 2.2 Aggregate decomposition

Standardised units. `Var share = Var_cut / L_cut`.

| arm | split | `L_cut` | `L_graph` | `Var_cut` | Var share | `Var_cut` std |
|---|---|---:|---:|---:|---:|---:|
| A parent FACTOR | Train | 0.0826 | 0.0493 | 0.0333 | **0.403** | 0.0227 |
| A parent FACTOR | Valid | 0.6689 | 0.5035 | 0.1654 | **0.247** | 0.1127 |
| B CSOI_ADD | Train | 0.4084 | 0.3287 | 0.0797 | **0.195** | 0.0543 |
| B CSOI_ADD | Valid | 0.6005 | 0.5179 | 0.0826 | **0.138** | 0.0563 |
| C CSOI_GLOBAL | Train | 0.1936 | 0.1347 | 0.0589 | **0.304** | 0.0401 |
| C CSOI_GLOBAL | Valid | 0.6478 | 0.5397 | 0.1082 | **0.167** | 0.0737 |

`D_CSOI_REL_EXTRA` is similar (Var share 0.270 Train / 0.183 Valid). The
variance term is therefore not negligible: on Train it is about one-fifth to
two-fifths of the objective, and on Valid it is about one-seventh to one-quarter.
The parent's Train fit is especially variance-heavy; on Valid the parent's graph
error grows faster than its cut variance, so the relative variance share falls
even though absolute `Var_cut` rises from 0.033 to 0.165.

### 2.3 Cut geometry, cancellation and association with error

| arm | split | mean cut span | cancellation share | mixed-sign fraction | Pearson `Var` vs `|graph error|` |
|---|---|---:|---:|---:|---:|
| A | Train | 0.517 | 0.568 | 0.161 | 0.319 |
| A | Valid | 1.062 | 0.435 | 0.402 | 0.174 |
| B | Train | 0.805 | 0.400 | 0.286 | 0.137 |
| B | Valid | 0.787 | 0.323 | 0.283 | 0.073 |
| C | Train | 0.678 | 0.493 | 0.222 | 0.233 |
| C | Valid | 0.866 | 0.363 | 0.319 | 0.163 |

- Span is the mean within-molecule `max_c f_c - min_c f_c`, standardised.
- Cancellation share is the mean per-molecule `Var_cut / (L_graph + Var_cut)`.
- Mixed-sign fraction is the fraction of molecules whose cuts contain both
  positive and negative responses.

Cut counts per molecule: Train mean **8.11**, median 8, p90 13, max 55; Valid
mean **7.48**, median 7, p90 12, max 50. So there is ample opportunity for
within-molecule cancellation: the typical molecule has 7–8 cuts, and the parent
already has both positive and negative cut responses in 16% of Train molecules
and 40% of Valid molecules.

The variance is **structured, not uniform**:

- In the parent's Train split the Var share rises from 0.317 for `S=1` to 0.511
  for `S=5-8`, and from 0.326 for `C>=33` to 0.676 for `C=5-8` (the latter cell
  has only 40 molecules). On Valid, absolute `Var_cut` is largest for the small
  `C=5-16` contexts (~0.13) and lower for `C>=33` (~0.097), though the small
  cells contain only 5 and 70 molecules.
- The association between `Var_cut` and `|f_bar - y|` is weak-to-moderate:
  Pearson 0.319/0.174 for A, 0.137/0.073 for B, 0.233/0.163 for C
  (Train/Valid). Variance is not simply a proxy for graph error; it is a
  separate, molecule-dependent dispersion term.

### 2.4 What the old `pair_metrics.json` delta actually was

The old delta was the **whole-molecule delta**, not the direct cut-response
delta:

```text
whole_molecule_delta = ŷ(G_b) - ŷ(G_a) = f_bar(G_b) - f_bar(G_a)
direct_cut_delta     = f(C, S_b) - f(C, S_a)   # shared C and cut bond
```

The recomputation matches the old `pair_metrics.json` to `0.0` absolute
difference for delta MAE and delta RMSE in every arm/scope; see
`analysis/checks.json`. The two definitions are materially different:

| arm | Train×Train whole / direct MAE | Train×Valid whole / direct MAE | Valid×Valid whole / direct MAE |
|---|---:|---:|---:|
| A | 0.227 / 0.145 | 0.477 / 0.544 | 0.647 / 0.800 |
| B | 0.408 / 0.391 | 0.450 / 0.470 | 0.626 / 0.623 |
| C | 0.312 / 0.228 | 0.548 / 0.562 | 0.593 / 0.742 |

The old pair metric therefore mixed the desired direct composition response
with all other cuts' contributions and cancellation. It must not be interpreted
as a direct `f(C,S_b)-f(C,S_a)` result.

---

## 3. Q2 — Does graph-level supervision improve Valid prediction?

### 3.1 Valid and Train results

All values are original label units. `Δ` is `new − parent` (positive = worse);
the CI is a paired bootstrap over the 420 Valid molecules.

| arm | best epoch | Valid RMSE | Δ vs parent | 95% CI | Train RMSE | Valid MAE | Valid `Var_cut` std | Valid graph MSE std | Valid cancel |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| A parent | 100 | **0.7096** | — | — | 0.2220 | 0.5393 | 0.1127 | 0.3430 | 0.4350 |
| `F_GRAPH` | 31 | **0.8097** | **+0.1001** | `[+0.0554, +0.1462]` | 0.4948 | 0.6332 | 1.3813 | 0.4467 | 0.7268 |
| B parent | 147 | **0.7196** | — | — | 0.5733 | 0.5631 | 0.0563 | 0.3528 | 0.3231 |
| `B_GRAPH` | 147 | **0.7567** | +0.0371 | `[-0.0098, +0.0772]` | 0.5768 | 0.6026 | 0.2423 | 0.3901 | 0.5115 |
| C parent | 135 | **0.7346** | — | — | 0.3670 | 0.5646 | 0.0737 | 0.3677 | 0.3625 |
| `C_GRAPH` | 101 | **0.7758** | +0.0412 | `[-0.0184, +0.1033]` | 0.4053 | 0.5988 | 1.0103 | 0.4100 | 0.6976 |
| GINE reference | 128 | **0.6406** | — | — | 0.2946 | 0.4674 | — | — | — |

The direction is consistent: **all three graph-loss arms are worse on Valid**.
For the parent architecture the degradation is clear and the CI excludes zero.
For B and C the point estimates are worse by 0.037 and 0.041, but a single-seed
paired CI over 420 molecules does not resolve them; they are not evidence of
improvement either.

The graph loss did **not** improve graph-level prediction at the selected
checkpoints:

| arm | Train graph MSE (best) | Valid graph MSE (best) | Train `Var_cut` | Valid `Var_cut` |
|---|---:|---:|---:|---:|
| A parent | 0.0336 | 0.3430 | 0.0227 | 0.1127 |
| `F_GRAPH` | 0.1668 | 0.4467 | 1.4527 | 1.3813 |
| B parent | 0.2239 | 0.3528 | 0.0543 | 0.0563 |
| `B_GRAPH` | 0.2267 | 0.3901 | 0.2823 | 0.2423 |
| C parent | 0.0918 | 0.3677 | 0.0401 | 0.0737 |
| `C_GRAPH` | 0.1119 | 0.4100 | 1.0992 | 1.0103 |

For `F_GRAPH` the best-by-Valid checkpoint is epoch 31, not the end. By epoch
160 its Train graph MSE falls to **0.0096** while Valid graph MSE rises to
**0.4814** and Valid RMSE to **0.8406**: the graph objective is being fit by
cancellation patterns that do not transfer. The early-stopped epoch-31 model is
still worse than the parent on both Valid graph MSE and Valid RMSE.

### 3.2 Mechanism: cancellation, not branch death

Removing the cut-wise term increases cut dispersion by an order of magnitude on
Valid:

- `F_GRAPH`: `Var_cut` **0.1127 → 1.3813** (≈12.3×), cancellation
  0.435 → 0.727, mixed-sign 0.402 → 0.824, cut-MSE 0.456 → 1.828.
- `B_GRAPH`: `Var_cut` **0.0563 → 0.2423** (≈4.3×), cancellation
  0.323 → 0.512, mixed-sign 0.283 → 0.526, cut-MSE 0.409 → 0.632.
- `C_GRAPH`: `Var_cut` **0.0737 → 1.0103** (≈13.7×), cancellation
  0.363 → 0.698, mixed-sign 0.319 → 0.776, cut-MSE 0.441 → 1.420.

At the best checkpoints, `Var_cut` is 75.6% of `L_cut` for `F_GRAPH` and 71.1%
for `C_GRAPH` (Valid), versus 24.7% and 16.7% for their parents. This is the
signature of a null space: the molecule mean is constrained, but individual cut
responses can grow in opposite directions.

The interaction branch did not shut down; it became **more** active:

| diagnostic (frozen best checkpoint) | old C | `C_GRAPH` | old A | `F_GRAPH` |
|---|---:|---:|---:|---:|
| interaction contribution `mean_abs` | 0.607 | **0.885** | 0.402 | **0.795** |
| attention effective tokens, p50 (all heads) | 26.8 | **30.8** | — | — |
| interaction gradient norm at best epoch | 0.792 | **1.012** | — | — |
| attention entropy, p50 (nats) | 3.064 | 2.870 | — | — |

`B_GRAPH` is additive by construction, so its interaction contribution and
interaction gradient are exactly zero; its cut variance still rises 4.3×. This
isolates the effect: even without an interaction branch, removing the cut-wise
penalty lets the additive model spread cut responses. Adding or enlarging an
interaction branch is therefore not the missing ingredient in this pilot.

The gradient snapshots tell the same story: `F_GRAPH` total gradient norm falls
from 9.75 (first epoch) to 2.63 (best epoch 31), while `B_GRAPH` rises from 0.54
to 9.17 and `C_GRAPH` from 1.04 to 1.84. The optimization is not failing to
move; it is moving into cancelling cut configurations.

**The variance rise is not interpreted as improved representation.** It is the
opposite: the graph-only objective removes the term that kept cut responses
calibrated, and the resulting dispersion is associated with worse Valid
whole-molecule error.

### 3.3 All pre-registered paired contrasts

`Δ = RMSE(first) − RMSE(second)`; negative means the first arm is better.

| contrast | Δ RMSE | 95% paired CI | reading |
|---|---:|---|---|
| A → `F_GRAPH` | **−0.1001** | `[−0.1462, −0.0554]` | A better; clear degradation |
| B → `B_GRAPH` | −0.0371 | `[−0.0772, +0.0098]` | B point-better; unresolved |
| C → `C_GRAPH` | −0.0412 | `[−0.1033, +0.0184]` | C point-better; unresolved |
| `B_GRAPH` → `C_GRAPH` | −0.0191 | `[−0.0817, +0.0410]` | B point-better; unresolved |
| `F_GRAPH` → `C_GRAPH` | +0.0339 | `[−0.0198, +0.0889]` | C point-better; unresolved |
| `C_GRAPH` → GINE | **+0.1352** | `[+0.0530, +0.2150]` | GINE better by 0.135 |

The graph objective does not narrow the GINE gap; it widens it from 0.094
(`C_CSOI_GLOBAL` − GINE) to 0.135 (`C_GRAPH` − GINE). The 0.02 RMSE threshold
is exploratory only; the single-seed CI does not cover training randomness.

---

## 4. Q3 — Does it improve real replacement prediction?

### 4.1 Definitions and scopes

Two deltas are reported separately and never mixed:

- **Whole-molecule δ**: `ŷ(G_b) − ŷ(G_a)`, the old `pair_metrics.json`
  definition.
- **Direct cut δ**: `f(C, S_b) − f(C, S_a)` at the shared C-side context and cut
  bond, using the model's cut-response head.

The pair directory has 5770 real matched pairs. After excluding pairs that touch
Test, the scopes are **4646 Train×Train**, **434 Train×Valid**, and **15
Valid×Valid**. No synthetic edits were made. The 434 cross pairs come from 107
Valid molecules; those are the cluster units for the cross-pair bootstrap. The
15 Valid×Valid pairs form one transform component, so their cluster interval is
degenerate and is reported descriptively only.

### 4.2 Train→Valid cross pairs (n = 434, 107 Valid-molecule clusters)

| arm | whole MAE | whole RMSE | whole dir. acc. | direct MAE | direct RMSE | direct dir. acc. |
|---|---:|---:|---:|---:|---:|---:|
| A parent | 0.477 | 0.629 | 0.811 | 0.544 | 0.762 | 0.781 |
| `F_GRAPH` | 0.563 | 0.714 | 0.742 | 0.572 | 0.743 | 0.777 |
| B parent | 0.450 | 0.602 | 0.823 | 0.470 | 0.623 | 0.800 |
| `B_GRAPH` | 0.451 | 0.581 | 0.795 | 0.472 | 0.604 | 0.813 |
| C parent | 0.548 | 0.766 | 0.760 | 0.562 | 0.782 | 0.779 |
| `C_GRAPH` | 0.561 | 0.714 | 0.749 | 0.596 | 0.773 | 0.760 |
| GINE reference | **0.407** | **0.537** | **0.830** | — | — | — |

The graph-loss variants do not improve the direct cross-molecule response:
A→`F_GRAPH` direct MAE 0.544→0.572; B→`B_GRAPH` 0.470→0.472; C→`C_GRAPH`
0.562→0.596. Whole-molecule δ shows the same direction. Direct direction accuracy is
0.813 vs 0.800 for B→`B_GRAPH` but 0.760 vs 0.779 for C→`C_GRAPH`, so it is not
consistently better; MAE is not better in either case, and the differences are
within the single-seed uncertainty. GINE remains the strongest reference on
both whole δ MAE and RMSE.

### 4.3 Valid→Valid pairs (n = 15, one component)

| arm | whole MAE | whole RMSE | whole dir. acc. | direct MAE | direct RMSE | direct dir. acc. |
|---|---:|---:|---:|---:|---:|---:|
| A parent | 0.647 | 0.765 | 0.600 | 0.800 | 0.869 | 0.667 |
| `F_GRAPH` | 0.573 | 0.703 | 0.867 | 0.620 | 0.784 | 0.667 |
| B parent | 0.626 | 0.770 | 0.667 | 0.623 | 0.760 | 0.733 |
| `B_GRAPH` | 0.577 | 0.715 | 0.733 | 0.662 | 0.763 | 0.800 |
| C parent | 0.593 | 0.758 | 0.667 | 0.742 | 0.914 | 0.733 |
| `C_GRAPH` | **0.518** | **0.644** | **0.867** | 0.734 | 0.901 | **0.533** |
| GINE reference | 0.459 | 0.545 | 0.667 | — | — | — |

There are a few favourable point estimates here (`C_GRAPH` whole MAE 0.518 vs
0.593; `F_GRAPH` direct MAE 0.620 vs A 0.800), but with 15 pairs in one
component they are not stable evidence. `C_GRAPH`'s direct direction accuracy
falls to 0.533 while its whole-molecule accuracy rises to 0.867, which is
exactly the kind of small-sample split that should not be over-read. GINE still
has the best whole-molecule Valid-only MAE (0.459).

### 4.4 Train×Train in-sample pairs (n = 4646, 458 pair components)

| arm | whole MAE | whole RMSE | whole dir. acc. | direct MAE | direct RMSE | direct dir. acc. |
|---|---:|---:|---:|---:|---:|---:|
| A parent | 0.227 | 0.304 | 0.883 | 0.145 | 0.203 | 0.919 |
| `F_GRAPH` | 0.405 | 0.545 | 0.803 | 0.360 | 0.510 | 0.825 |
| B parent | 0.408 | 0.556 | 0.796 | 0.391 | 0.551 | 0.812 |
| `B_GRAPH` | 0.414 | 0.563 | 0.797 | 0.382 | 0.545 | 0.817 |
| C parent | 0.312 | 0.415 | 0.832 | 0.228 | 0.319 | 0.868 |
| `C_GRAPH` | 0.356 | 0.469 | 0.810 | 0.310 | 0.431 | 0.836 |
| GINE reference | 0.276 | 0.378 | 0.851 | — | — | — |

These are in-sample and are not used as evidence of generalization. The
cluster intervals resample 458 pair components, not individual pairs; full
intervals are in `analysis/pair_delta_comparison.json`.

**Answer to Q3:** the graph-level objective does not improve real replacement
prediction on the only cross-molecule scope with usable uncertainty. The old
`pair_metrics.json` whole-molecule metric and the direct cut metric can disagree
in both directions; both are now reported separately.

---

## 5. Q4 — Local-context support

### 5.1 Definition and identity

For each real parse, the C-side attachment atom is the endpoint of the cut bond
that lies in C. The local context is the induced subgraph of C-side atoms within
graph distance `r` of that atom (`r = 1, 2, 3`), or the complete C object.
Identities are canonical and numbering-independent, using the parent
`canonical_key` encoding (colour-encoded raw attributes plus a port marker at
the C-side endpoint). The primary key is structure + cut-bond 3-tuple; a
structure-only variant is also computed.

All 30251 complete-C keys reconstruct exactly, and all 60 renumbering checks
pass (`analysis/checks.json`). Test parses are excluded. Train has 27157 real
parses across 3353 molecules; Valid has 3120 parses across 419 molecules.

### 5.2 Reuse census

Primary key with cut bond. "Seen" means the key occurs at least once in Train.
"Joint" means the exact (context, S-type) pair occurs in Train.

| radius | Train distinct contexts | Valid seen parses | Valid seen unique | Valid seen molecule-any | Valid joint (context,S) | Train contexts in ≥2 molecules | max Train molecules/context |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 1,135 | **95.7%** | 79.2% | 99.8% | **80.4%** | 58.8% | 1,154 |
| r2 | 4,026 | **82.3%** | 64.2% | 97.1% | **64.5%** | 45.2% | 577 |
| r3 | 7,580 | 65.5% | 50.8% | 91.6% | 50.1% | 34.5% | 527 |
| complete C | 22,614 | **5.8%** | 6.0% | 25.5% | **0.0%** | 5.7% | 14 |

The complete-C row has an important nuance: 25.5% of Valid molecules have at
least one parse whose complete C context was seen somewhere in Train, and 4.3%
of parses have both the context and the S-type seen separately, but **no Valid
parse has the exact (context, S) pair seen**. The joint object is effectively
unsupported. For r1, by contrast, 80.4% of Valid parses have the exact pair
seen, and the most common r1 context occurs in 1,154 Train molecules.

The structure-only variants are nearly identical (r1: 96.0% seen, 80.4% joint;
r2: 83.6%, 64.5%; r3: 66.4%, 50.1%; complete C: 6.1%, 0.0%), so the result is
not an artifact of including the cut-bond suffix. The complete-C cross-check
reproduces the prior support census (`valid_C_seen_rate` 0.0576 vs 0.0580 on
real parses only).

**Answer to Q4:** radius-1 and radius-2 C-side contexts have substantial
cross-molecule reuse, including reuse with the same S-type. The complete C
context does not. This makes a local context a plausible object for a follow-up,
but this is a support census, not a trained-model result; reuse does not by
itself prove learnability.

---

## 6. Q5 — Which follow-up direction?

The evidence points to **local-context object**, with a **retained
cut-calibration/variance term**, not to the graph-level objective and not to
simply enlarging the interaction branch.

- **Graph-level objective:** rejected as the next step for this setting. It is
  point-worse in all three architectures, clearly worse for the parent, and it
  increases cancellation and cut dispersion by 4–14×. It also does not improve
  direct or whole replacement deltas on Train→Valid pairs.
- **Larger interaction branch:** not supported as the primary lever. The
  interaction branch is already active and broad in `C_GRAPH` (interaction
  contribution 0.885, attention effective tokens 30.8, interaction gradient
  1.01), yet the arm is worse. `B_GRAPH` has no interaction branch and still
  suffers the same dispersion problem. The bottleneck is the objective's
  calibration, not branch capacity.
- **Local-context object:** the strongest positive evidence. Radius-1/2 contexts
  are reused across molecules and across S-types, while complete C is nearly
  unique and has zero exact joint support on Valid. The next question should be
  whether conditioning the cut response on a local C-side context improves
  cross-molecule composition response while preserving within-molecule
  calibration.
- **Something else:** the objective itself must keep a variance/calibration
  term. The graph-only arms show that if cut responses are free to cancel, the
  model will use that freedom and Valid whole-molecule error gets worse. The
  pair evaluation should continue to report whole-molecule and direct cut δ
  separately; the old whole-molecule-only metric is not a direct response
  metric.

This is a direction, not a new architecture. Per the protocol, no further model
was designed or trained.

---

## 7. Engineering and provenance

- **Tests:** `12 passed` in 14.2 s at the end of the round. They cover
  checkpoint reproduction, per-molecule loss identity, graph-loss expression,
  pair-delta code paths, Train/Valid-only `RunData`, CPU smoke of all three
  arms, and the absence of Test keys in new predictions.
- **Gate:** `analysis/checks.json` has `all_ok: true` for the loss identity, old
  `pair_metrics` provenance, source audit, local-context identity, and the CPU
  smoke.
- **Data:** cache content digest
  `46439b91dfd760255be4c53256f12687959d203b6740e55c0213d66476168825`;
  scaler mean `2.1621607`, std `1.2115398`; 3360/420/420 molecules. Initial
  module hashes match the CSOI manifest. Old checkpoint Valid predictions
  reproduce the stored arrays within `2e-6`.
- **Test policy:** only `train_*` and `valid_*` arrays were materialised from
  legacy npz files; Test arrays were never indexed. New prediction files contain
  only Train/Valid keys.
- **Provenance:** `results/provenance.json` records the local commit, config and
  protocol hashes, cache digest, `rr` run ids, pools, commands, environment, and
  SHA-256/size for every checkpoint, prediction file, curve and manifest.
  `results/provenance/rr_status_*.json` are the raw scheduler records.
- **Figure:** `results/figures/training_curves.png` shows Valid RMSE, Valid cut
  variance and Valid graph MSE against epoch for the new arms and the reused
  parents where the metric exists.

### Deviations

1. **Post-run diagnostic fix (no training effect).** The in-run
   `mechanism_diagnostics.json` for `F_GRAPH` failed because the frozen-weight
   zeroing shim assigned to `model.q` / `model.r` was a plain callable, not an
   `nn.Module`. Training, loss, checkpoint selection and predictions were
   unaffected. The shim was fixed in `code/cpo_train.py`; diagnostics were
   recomputed for all six old/new arms at their frozen best checkpoints in
   `analysis/mechanism_diagnostics.json`.
2. **Remote smoke run.** A 2-step smoke run (16 Train / 8 Valid molecules) was
   used on `res-gpu0` before the formal runs. It is not an arm, is not in
   `results/`, and is not charged to the 1.006 GPU-hour total.
3. **No formal recipe deviations.** All three formal runs used seed 0, 160
   epochs, the frozen parent/CSOI recipe, and best-by-Valid selection.

### Limitations

Single seed 0; the paired bootstrap resamples the 420 Valid molecules and does
not cover training-seed randomness. The `~0.02` RMSE threshold is exploratory,
not a confirmation threshold. The B/C graph-loss contrasts are unresolved, not
positive. The 15 Valid×Valid pairs are one transform component and cannot
support a stable interval. The local-context result is a support census, not a
trained local-context model. The graph-loss intervention changes only the main
loss; it does not test other ways of regularising cut responses.

---

## 8. Artifacts

| path | contents |
|---|---|
| `PROTOCOL.md` | frozen protocol and deviations log |
| `REPORT.md` | this report |
| `code/` | audit, trainer, local-context, checks, pair comparison, analysis, provenance, plot |
| `tests/` | 12 engineering tests |
| `analysis/loss_decomposition.json` | per-molecule decomposition, strata, quantiles, old pair provenance |
| `analysis/local_context_support.json` | radius census, identity checks, structure-only variants |
| `analysis/pair_delta_comparison.json` | whole vs direct δ for old/new arms, scopes, cluster bootstrap |
| `analysis/mechanism_diagnostics.json` | frozen best-checkpoint interaction/attention/gradient diagnostics |
| `analysis/checks.json` | engineering gate |
| `results/main_table.csv` | main arm table |
| `results/summary.json` | machine-readable summary, contrasts, support, mechanism, execution |
| `results/provenance.json` | execution and artifact hashes |
| `results/<RUN_ID>/state_best.pt` | best-by-Valid checkpoints for F/B/C_GRAPH |
| `results/<RUN_ID>/state_final.pt` | final-epoch checkpoints for F/B/C_GRAPH |
| `results/<RUN_ID>/predictions_best.npz` | Train/Valid predictions only |
| `results/<RUN_ID>/curves.csv` | per-epoch Task-3 metrics |
| `results/figures/training_curves.png` | training curves |

## 9. Reproduction

Formal runs were launched from a clean commit with `rr` (host `res`):

```bash
rr doctor res --refresh
rr deploy res

rr run res CPO_C_GRAPH --pool res-gpu0 \
  --result tracks/structural-sufficiency/composition_objective_pilot_v1/results/C_GRAPH_LIPO_R0 \
  -- python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_train.py \
  --run-id C_GRAPH_LIPO_R0 --family csoi --arm CSOI_GLOBAL --dataset lipo \
  --epochs 160 --seed 0 --device cuda

rr run res CPO_B_GRAPH --pool res-gpu1 \
  --result tracks/structural-sufficiency/composition_objective_pilot_v1/results/B_GRAPH_LIPO_R0 \
  -- python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_train.py \
  --run-id B_GRAPH_LIPO_R0 --family csoi --arm CSOI_ADD --dataset lipo \
  --epochs 160 --seed 0 --device cuda

rr run res CPO_F_GRAPH --pool res-gpu1 \
  --result tracks/structural-sufficiency/composition_objective_pilot_v1/results/F_GRAPH_LIPO_R0 \
  -- python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_train.py \
  --run-id F_GRAPH_LIPO_R0 --family parent --arm FACTOR --dataset lipo \
  --epochs 160 --seed 0 --device cuda
```

Local analyses (CPU; run from the repository root):

```bash
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_audit.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_local_context.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_checks.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_mechanism.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_pair_compare.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_analyze.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_provenance.py
.venv/bin/python tracks/structural-sufficiency/composition_objective_pilot_v1/code/cpo_plot.py
.venv/bin/python -m pytest -q tracks/structural-sufficiency/composition_objective_pilot_v1/tests
```

The formal training commit is `83a020c`. The post-run diagnostic/analysis code
is the later working-tree state recorded in `results/provenance.json`; it does
not change the training path or any reported prediction.

---

*End of report. Execution stopped after the three arms, the pre-registered
diagnostics, the pair evaluation and the local-context census.*