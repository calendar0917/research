# Compact-v4-cell Minimal Falsification Experiment

> **Question.** Does promoting cycles from *graph-level descriptors* to
> *persistent, identity-bearing explicit structural objects* (cycle cells) with
> explicit lower<->cycle incidence improve compact-v4 under the frozen
> optimized protocol?
>
> **Verdict.** **Decision Case A — CELL REPRESENTATION NO-GO (under the locked
> protocol).** True-cell seed0 reached official-valid MAE **0.147462** vs the
> optimized compact-v4 seed0 reference **0.146420**, i.e.
> `Delta_arch0 = -0.001042` — no improvement (a noise-scale negative point
> estimate; the v4 seed-to-seed spread is ≈0.010), far below the
> pre-registered `+0.004` architecture gate. The cell branch was genuinely
> alive (zeroing its summary changes predictions by max|Δ| = 1.954) and no
> optimization pathology was observed. Per the pre-registered budget rule the
> broken-incidence control and seed1 were **not purchased**. Official test was
> **never loaded**.
>
> Code: `tracks/ksvd/experiments/luyin16/zinc_compact_v4_cell.py`.
> Tests: `tracks/ksvd/tests/test_compact_v4_cell.py` (21 pass).
> Results: `tracks/ksvd/results/compact_v4_cell/` (+ `answers_q1_q20.json`).

---

## 1. Motivation

This is the **single** representation-family hypothesis authorised by the
Same-Budget Structural Computation Gap Analysis
(`notes/structural_computation_gap_analysis.md`,
`results/structural_computation_gap_analysis/top1_hypothesis.json`,
`.../minimal_falsification_plan.json`). Every earlier local route terminated
NO-GO by adding a *statistic / readout over already-frozen states*. The gap
analysis changed the question to: *what structural computation family does
compact-v4 use before R, and what does a same-scale higher-order reference
(CIN-small) spend its parameters on instead?*

## 2. Evidence chain leading to this hypothesis

* Optimized compact-v4 (valid 0.146420 seed0; 0.149332 seed1) is a strong,
  real baseline; the 60-epoch protocol was under-trained
  (`notes/compact_v4_training_sufficiency.md`).
* The global topology hinge survives fair optimization and *widens*
  (`Delta_topology = +0.019846`, `notes/optimized_baseline_critical_revalidation.md`).
* The optimized-manifold broad frozen-state screen is INCONCLUSIVE
  (`notes/optimized_manifold_broad_state_screen.md`): a parameter-matched
  R-only reader is as good as a broad frozen-state reader.
* CIN-small is 2-layer cellular message passing over **induced-cycle 2-cells**
  with persistent cell states and explicit incidence; its No-Rings ablation is
  0.174 vs 0.094. compact-v4 has **no per-cycle object**: cycle information
  exists only as a graph-level 25D topology vector.

## 3. Why this is not another missing-statistic experiment

The earlier NO-GOs added static statistics/readers to frozen states. This
experiment changes the **computation graph**: a new object class (cycle cells)
with a persistent learned state, explicit incidence, and a small fixed number
of learned incidence-aware updates — from training start, not post-hoc.

## 4. Architecture lock

`results/compact_v4_cell/architecture_lock.json` was written **before any
full training** and was not modified afterwards. It fixes: cycle family, max
cycle length, cell dim, incidence, exact two-round equations, parameter
sharing, final pooling, `R` dimension, graph head, total params, protocol and
all thresholds.

Base pathway preserved (unchanged): patch radius 2; historical aliased
rooted-topology tokenizer; patch encoder (`146+16+8 -> 48`); complete-pair
encoder + relation encoder; centre aggregation; one-shot centre update; unary
moments 97 + pair moments 165 + global 32; **v4 global topology hinge branch**
(25->16->8); `R_original = 302`. The global topology branch is untouched.

## 5. Cycle construction

`results/compact_v4_cell/cycle_construction_lock.json`. Cycle family =
**induced (chordless) simple cycles**, enumerated with the repository's exact
min-node DFS (`structural_context._find_cycles`, the same enumerator behind the
topology channel) and filtered to chordless. `max_cycle_len = 8` is
pre-registered and **not swept**. Rationale (target-independent): `k=8` is a
strict superset of the plan's `k=6` option, covers all observed ZINC induced
ring sizes (3-8), cuts only a negligible tail (induced cycles longer than 8
appear in <0.1% of graphs), and is saturated at 8 on the valid split. No choice
used the target, residuals or valid performance.

Valid-split statistics: mean 2.786 cells/molecule, median 3, max 10; sizes
`{3:82, 4:9, 5:867, 6:1786, 7:30, 8:12}`; 1 graph with no cycle, 75 with one,
924 with multiple; mean 15.64 incidences/graph.

> **Construct-definition note (honesty).** The gap-analysis prose calls the
> existing enumerator "bounded induced-cycle enumeration"; the repository
> function `_find_cycles` actually enumerates *all simple* cycles up to the
> bound. We therefore implement the plan's literal wording (induced/chordless)
> by filtering chords, and record the discrepancy here rather than silently
> reinterpreting it.

## 6. Structural objects

* Lower object: a **patch-centred structural object** `p_i` (48D). The patch
  front-end builds exactly one radius-2 ego-net per atom, in node order, so
  patch index == centre atom index.
* Higher object: an **explicit cycle cell** `c_k` (48D), initialised as
  `CellInit(mean_{i in boundary(c)} h'_i)` from the mature v4 patch states
  `h'_i` (post centre-update). Each cell keeps an independent state.

## 7. Incidence definition

`I_PC`: patch `i` is incident to cell `c` iff atom `i` is a vertex of cycle
`c`. This is the cell boundary. **No `I_QC`** is used: the minimal plan does
not include a relation<->cell incidence, so no complete-pair relation `q_ij`
is treated as a cycle boundary relation. The cell boundary is its atoms.

## 8. Two-round computation

Exactly `T = 2` rounds (`CELL_ROUNDS = 2`), no depth sweep. One round =
`lower->cell` then `cell->lower`; parameters shared across the two rounds:

```
agg_c = mean_{i in boundary(c)} p_i^{r-1}
c_c^r = LayerNorm_c( c_c^{r-1} + W_c · relu(agg_c) )
agg_i = mean_{c containing i} c_c^r
p_i^r = p_i^{r-1} + alive_i · W_p · relu(agg_i)      # alive_i ∈ {0,1}
```

`CellInit` is applied once. The update outputs are zero-initialised
(`W_c`, `W_p`), so the branch starts as a patch-state no-op and cannot inject
bias into cycle-free patches.

## 9. Structural persistence

The two rounds operate on a branch-local copy of the patch states, so the v4
pathway (`R_original`) is never modified. Each `c_k` keeps its identity across
both rounds; cell identity is **not** pooled into a graph vector until both
pre-registered rounds complete. The cell branch sees the *mature* lower-order
states (post centre-update), i.e. it builds on the existing structural
computation rather than replacing it.

## 10. Parameter budget

`results/compact_v4_cell/parameter_audit.json` (instantiated model, not an
estimate):

| component | params |
|---|---:|
| original compact-v4 | 99,613 |
| cell initialization (`Linear(48,48)+LN+ReLU`) | 2,448 |
| incidence operators (`W_c`, `LN_c`, `W_p`) | 4,800 |
| head expansion (302->399, first Linear) | 6,208 |
| **total** | **113,069** |

Increment `+13,456` (`+13.5%`). The target range 108-112K is not attainable
without violating the locked `d_cell = 48` and the mandatory 302->399 head
expansion (which alone costs 6,208); the design is comfortably below the hard
ceiling 120K. No width was chosen by performance.

## 11. Compute budget

`results/compact_v4_cell/compute_audit.json`. Operator application counts per
molecule: `cell_init` x1, `cell_update` x2, `patch_update` x2, lower
aggregation x3 (init + 2 rounds), cell aggregation x2. Cycle enumeration is
target-independent graph structure; preprocessing of train+valid records +
cells took ~99 s (one-off, cached). The optimized protocol trains 185 epochs
(14,615 optimizer steps) at ~1072 s wall.

## 12. Stage 1 — true-cell seed0

Single run, frozen protocol (Adam lr 1e-3, wd 1e-5, batch 128, max 240,
patience 40, no scheduler, L1, best official-valid). No search.

| run | valid MAE | best epoch | epochs run | params |
|---|---:|---:|---:|---:|
| optimized compact-v4 seed0 (reference, reused) | **0.146420** | 169 | 209 | 99,613 |
| **compact-v4-cell true seed0** | **0.147462** | 145 | 185 | 113,069 |

`result: results/compact_v4_cell/stage1_true_seed0.json` (+ per-epoch curve with
`train_mae`, `valid_mae`, `lr`, `optimizer_steps`, `checkpoint_selected`,
`cell_grad_norm`, `cell_state_norm`, `cell_summary_norm`).

## 13. Architecture gate

`Delta_arch0 = 0.146420 - 0.147462 = -0.001042`. Required: `>= +0.004`.
**FAIL** (the point estimate is indistinguishable from zero; the v4 seed-to-seed
spread over four seeds is ≈0.010, so a -0.001 single-seed difference is not a
meaningful degradation). No optimization pathology: no NaN, best epoch 145/240
(not boundary-pinned), train loss decreasing monotonically, early stop fired
normally. The cell branch is alive (see §18). Common-input bulk safe
(§17). Per the pre-registered rule this is **Case A — CELL REPRESENTATION
NO-GO**; the broken-incidence control and seed1 were **not purchased**.

## 14. Broken-incidence control design (designed, NOT run)

Designed per `minimal_falsification_plan` and the task mandate: deterministic
**within-molecule degree-preserving bipartite edge swaps** on the patch<->cell
incidence, preserving per-cell boundary count (cell size), per-patch cell
degree, cell count and total incidence count, changing only *which* patch is
incident to *which* cell. A manifest
(`broken_incidence_manifest_seed0.json`) would have been fixed before training.
Unit tests for the control are implemented and pass
(`tests/test_compact_v4_cell.py`, tests 11-19), but **no broken full-training
run was executed** because the architecture gate failed. No placeholder result
file was created.

Honest caveat recorded now: because `cell_init` aggregates the boundary patch
states, a broken-incidence cell's initialization value necessarily changes as
a function of the (rewired) membership. The control can preserve *degrees and
marginals* exactly (cell count, per-cell size multiset, per-patch degree,
total incidence count, lower-state multiset), but not the exact per-cell init
value — that is inherent to the mechanism under test.

## 15. Mechanism gate (not reached)

`Delta_inc0 = MAE(broken seed0) - MAE(true seed0) >= +0.0025` and
`Delta_arch0 >= +0.004` are both required. Not evaluated: Stage 2 was not
authorised. No broken run, no mechanism claim.

## 16. Seed1 conditional replication (not run)

Not authorised (seed0 architecture gate failed). No `stage3_*` artifacts were
created.

## 17. Common-input bulk

`results/compact_v4_cell/common_input_bulk.json`. The target-independent bulk
definition reuses the repository's `rare_le5_ratio` (fraction of patches whose
typed token has train frequency <= 5) below the 80th percentile of train
(threshold 0.0714, **756** valid molecules — exactly matching the optimized
broad-state screen). Re-running the reused optimized-v4 seed0 checkpoint on the
same pipeline reproduces valid MAE 0.14642022556537995 exactly.

| | v4 | true-cell | diff (true - v4) |
|---|---:|---:|---:|
| full valid MAE | 0.146420 | 0.147462 | **+0.001042** |
| common-input bulk MAE | 0.094721 | 0.096483 | **+0.001762** |

Both within the `+0.002` bulk gate (bulk safe), but in the *wrong direction*:
the cell branch slightly degrades the easy bulk.

## 18. Numerical / mechanism sanity

* **Branch-alive check** (inference-only): zeroing the cell summary at the
  trained checkpoint changes predictions by `max|Δ| = 1.954` (>> 1e-3). The
  branch is genuinely used, so the negative is a real negative, not a dormant
  or collapsed branch.
* `cell_grad_norm` is non-zero every epoch (≈0.05-0.32); cell-state norms grow
  from ≈6.97 to ≈7.61 over training — the branch is being optimised.
* `results/compact_v4_cell/mechanism_diagnostics.json` (trained checkpoint,
  official valid): `cell_init_norm = 4.52`; cell-state norm by round
  `[7.32, 7.49]`; lower-state update norm by round `[3.84, 3.71]`; cell
  summary norm `7.60`. The two incidence rounds measurably move the branch-local
  lower states, so the incidence-aware computation is genuinely executed (it
  simply does not improve the graph-level prediction).
* **Implementation-integrity note (recorded):** the first Stage-1 attempt used
  `torch_geometric.loader.DataLoader(collate_fn=...)`, which silently overrode
  the custom collate, so the cycle-cell incidence tensors never reached the
  model (cell state identically zero; the run merely reproduced v4). This was
  detected from the all-zero `cell_state_norm` curve, fixed by using
  `torch.utils.data.DataLoader`, guarded by an explicit runtime check, and
  covered by test `test_10b_cell_loader_carries_incidence`. The invalid run
  was discarded and re-run; only the corrected run is reported.

## 19. What is and is not proven

**Proven (within seed0, locked protocol):**

* A minimal explicit persistent cycle-cell branch (`d_cell=48`, `T=2`,
  113,069 params, `R=399`) is implementable, permutation-invariant,
  deterministic and numerically healthy; the original compact-v4 pathway is
  bit-identical inside the hybrid model (G0.1 max diff 0.0).
* Under the frozen optimized protocol this branch does **not** improve
  compact-v4 seed0 valid MAE: `Delta_arch0 = -0.001042` (gate +0.004).

**Not proven:**

* That explicit cycle cells have no value in general. This is a statement about
  *this minimal architecture* under *this training protocol*, not a
  mathematical impossibility.
* Anything about the incidence mechanism — the broken control was not run
  because the architecture gain was insufficient to justify it.
* Any official-test number — test was never loaded.

## 20. Method-philosophy compatibility

The design respects the constraints: explicit structural objects with clear
meaning; rank-specific shared operators; permutation invariance; **exactly 2**
pre-registered incidence rounds; no attention/Transformer; no generic node
message passing; no per-cycle-length MLPs; global topology branch preserved.
No depth escalation was attempted. If a gain had required many/iterative
rounds, the decision would have flipped to Case G (method-philosophy tradeoff);
that situation did not arise — the branch simply did not help at `T=2`.

## 21. Official-test lock

Official test was **never loaded** in any stage, including Stage 0, the
extraction cache, the training run and every diagnostic. Only official
train + valid were used.

## 22. Final verdict

**Decision Case A — CELL REPRESENTATION NO-GO.** True-cell seed0 did not reach
the `+0.004` architecture gate (it did not improve at all); the broken-incidence control was therefore not
purchased. The budget principle ("do not pay for the control unless the
architecture clears `+0.004`") was honoured exactly: **one** new full-training
run was spent.

`results/compact_v4_cell/final_decision.json`.

---

## Q1-Q20

| # | answer |
|---|---|
| Q1 | Induced (chordless) cycles, length 3..8, exact min-node DFS + chordless filter, canonical sorted tuples, target-independent. |
| Q2 | mean 2.786 / median 3 / max 10 cells per valid molecule (27,808 cells over 10k train). |
| Q3 | d_cell = 48. |
| Q4 | T = 2 rounds; one round = lower->cell then cell->lower; equations in §8. |
| Q5 | I_PC = atom(patch)⊂cycle membership; lower->cell = mean boundary patch state, cell->lower = mean incident cell state; no I_QC. |
| Q6 | R_new = 399 = 302 + 97. |
| Q7 | 113,069 params. |
| Q8 | +13,456 (+13.5%). |
| Q9 | true-cell seed0 best valid 0.147462, best epoch 145. |
| Q10 | Delta_arch0 = -0.001042. |
| Q11 | No. |
| Q12 | Not run (control not purchased). |
| Q13 | N/A. |
| Q14 | N/A (designed marginals listed in §14). |
| Q15 | N/A (0% of runs executed). |
| Q16 | Not reached; architecture gate failed first. |
| Q17 | No. |
| Q18 | Not run. |
| Q19 | No. |
| Q20 | Case A. |
