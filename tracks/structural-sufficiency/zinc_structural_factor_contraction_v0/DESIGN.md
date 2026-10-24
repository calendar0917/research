# DESIGN — zinc_structural_factor_contraction_v0 (SFC-v0)

Track `structural-sufficiency`. New directory, new protocol id `sfc-v0-cpu-audit`.
This round is a **CPU-only mathematical / computational feasibility audit**. It is
**not** a ZINC MAE round, not a hyper-parameter search, and not a new main line.

Frozen before any code runs. Written 2026-10-25, before the first measurement.

## 0. What is being audited

A candidate operator, **SFC** (Structural Factor Contraction): a shared
two-state graph-factor system whose exact partition function is used as a
structural interaction descriptor.

Given a typed graph `G = (V, E, a, b)` with atom categories `a_v` and bond types
`b_e ∈ {1,2,3}`, and `C` channels:

```
q_c(s | a)         = softmax_s A[a, c, s]                         (a shared node factor)
K_{b,c}(s, t)      = exp( tanh( S[b, c, s, t] ) )                  (a shared edge factor)
Z_c(G; theta)      = sum_{s in {0,1}^V} prod_v q_c(s_v | a_v) prod_{e=(u,v)} K_{b_e,c}(s_u, s_v)
logZ_c(G; theta)   = log Z_c(G; theta)                             (exact, log domain)
```

`A[28, C, 2]` and `S[3, C, 2, 2]` are shared across every graph, atom and
position. There is no node-id / whole-pattern embedding. `Z` is computed by
**exact variable elimination** (bucket elimination over a tree decomposition),
not by message passing.

Two derived scalars are used as the "interaction" readout:

```
H_c(G) = logZ_c(G) - sum_v logZ_c({v})        "centred interaction"
T_c(G) = H_c(G) - sum_{e} H_c({e})            "connected interaction" (tree => 0)
```

`H` is the object the task text calls "中心化交互量". The audit tests both.

**Honest labelling.** Variable elimination is *exact graph inference*. It is not
zero-graph propagation and it is not a GNN layer. Every number reported here is
an exact log-partition function up to float64 rounding, and the report says so.

## 1. Round structure

| Task | Question | Where |
|---|---|---|
| **T1** implementation correctness | Is the exact `logZ` right, invariant, differentiable, and structurally discriminating? | `code/sfc_core.py`, `code/sfc_checks.py` |
| **T2** real-graph cost | What does exact elimination cost on the 10,000 official ZINC-subset **train** structures, and what would `K=8/16` forward+backward cost? | `code/sfc_cost.py` |
| **T3** synthetic sample efficiency | Does the SFC interaction beat (a) composition statistics and (b) an equal-information one-shot pair/junction model on two non-SFC-generatable rules? | `code/sfc_synth.py` |
| verdict | one of the three allowed labels | `REPORT.md` |

## 2. T1 — correctness contract (all must pass before T2/T3 count)

The reference is an **independent** implementation: dense enumeration of all
`2^|V|` joint states in float64, no shared code with the elimination path except
the factor tables themselves.

1. `logZ` from elimination equals dense enumeration on many random typed graphs
   (`n ∈ [1, 8]`, cycles, multi-component, isolated nodes), `max|Δ| ≤ 1e-10`.
2. Atom relabelling: a random permutation of the vertex labels leaves every
   `logZ_c` unchanged (`≤ 1e-12`).
3. Gradients: analytic `d logZ / d theta` (autograd) equals a float64 central
   finite difference of the **dense enumeration** reference for `A` and `S`
   (`rel err ≤ 1e-6`). Also checked against the elimination path.
4. Zero coupling: if every `K ≡ 1` (i.e. `tanh(S) = 0`), then `H_c = 0` exactly
   (up to float64), for every graph.
5. Trees: with **no node field** (`q ≡ 1/2`, i.e. `A ≡ 0`) the interaction of a
   tree is exactly `0`, while a graph with one cycle has `≠ 0`. Checked on
   matched pairs (same `n`, same edge count, same degree sequence where possible).
6. Correspondence sensitivity: two graphs with **identical** atom-category
   multiset and identical bond-type multiset but different real connections give
   different `H_c` (and different `T_c`) at a fixed non-degenerate parameter
   point.
7. Elimination-order invariance: every valid elimination order gives the same
   `logZ` (to `1e-10`); the reported width depends on the order, the value does not.
8. Determinism: same input, same output, bit-identical on repeat.

If any of 1–7 fails, the round stops and reports the failure. No T3 number may
be quoted if T1 is not green.

## 3. T2 — cost on real structures (structures only, no labels)

Source: `data/ZINC/raw/train.pickle` + `data/ZINC/raw/train.index`, read through
a **label-blocking mapping** that raises on `logP_SA_cycle_normalized`. Only
`num_atom`, `atom_type`, `bond_type` are read. Official valid/test are never
loaded. The PyG `train.pt` tensor cache is *not* used (it materialises `y`).

For all 10,000 train graphs, per graph, with the **min-fill** elimination order
and the best-of-{min-degree, min-fill, min-fill-tiebreak} as a reported upper
bound on width quality:

* elimination **treewidth proxy** = max intermediate factor scope size `− 1`;
* max intermediate factor size in floats for `C` channels (`2^w · C`);
* peak bytes for the elimination (sum of live buckets, measured, not modelled);
* wall time per graph at `C=1`, and the same elimination replayed at `C=8`, `C=16`;
* per-graph forward time and backward time (autograd through the same elimination)
  for `K = 8` and `K = 16` channels.

Reported as **distributions** (median, p90, p95, p99, max) and as
**total + worst-graph**, never as a mean that hides the tail. Also reported:
the fraction of the 10k that a stated wall-clock budget can cover, and the exact
count of graphs whose single-graph cost exceeds that budget.

If the full 10k cannot be processed within the declared budget, the report says
**BLOCKED** with the measured per-graph distribution and the projected total.
It does not subsample silently, does not truncate molecules, and does not call an
approximation exact.

## 4. T3 — synthetic sample-efficiency test

Two **independent** tasks. Neither target rule is a function of an SFC partition
function: task A is a *parity of a path cover* over a bond-type-coloured spanning
structure; task B is a *bottleneck/articulation* rule. Both are stated formally in
`code/sfc_synth.py` and are verifiable by brute force on small graphs.

Both tasks are built on **synthetic graphs with the same atom/bond category
statistics as ZINC train** (category frequencies matched, degree distribution
matched, `n` matched), but with **randomised real connection patterns** so that a
composition-only model cannot solve them by counting.

The generator is split into a **seen-combination** set and an **unseen-combination**
set: the unseen set uses atom-category pairs and bond-type pairs that never
co-occur in the seen set, at matched marginals. This is the distribution where
sample efficiency is actually measured.

Three arms, equal budget (same number of trainable parameters within ±5%, same
optimiser, same number of gradient steps, same seeds):

| arm | input |
|---|---|
| **C** composition | atom-category counts + bond-type counts (+ degree histogram), nonlinear MLP readout |
| **P** pair/junction | shared one-shot pair and junction statistics read from the *same raw graph* (typed induced pair counts, typed 2-hop junction counts, typed wedge counts) — same information as SFC's local factors, no contraction |
| **S** SFC | `H_c`, `T_c` for `K` channels + the same composition channel as C |

Reported: train-distribution error, unseen-combination error, **two independent
initialisations** per arm, and measured wall-clock / FLOP-equivalent cost.
Rule: SFC may only be called a sample-efficiency win if it beats **P** (not just
C) on the unseen-combination set, in both seeds, with a margin larger than the
seed spread. Beating only C is reported as "composition-only advantage" and does
**not** count.

## 5. Forbidden in this round

* no official valid/test load, no `y` read anywhere (label-blocking mapping);
* no ZINC target decomposition (logP / SAS / cycle components);
* no PPGT reproduction, no A100, no GPU run, no Slurm job;
* no modification of any historical checkpoint, config, result or report;
* no claim that synthetic separability implies real-task performance;
* no "approximate = exact" wording anywhere.

## 6. Deliverables

`REPORT.md` (short), `analysis/t1_correctness.json`, `analysis/t2_cost.json`,
`analysis/t3_synth.json`, the frozen code, and exactly one verdict:

* `COMPUTATION_INFEASIBLE`
* `COMPUTABLE_BUT_NO_SAMPLE_EFFICIENCY_EVIDENCE`
* `PROMISING_FOR_ZINC_PILOT`

The verdict does **not** start the next training round. It is an input to a human
decision.
