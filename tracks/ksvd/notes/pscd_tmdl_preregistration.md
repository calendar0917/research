# Pre-registration — PSCD-TMDL-v0: Task + MDL Motif Discovery Diagnostic

Round name: **PSCD-TMDL-v0**. Written **before** the formal remote run, following
the `remote-research-runner` skill (local code → remote A100 → local analysis).
It does not modify any historical record.

Official ZINC `test` is **never** loaded, instantiated or referenced. `y` is used
only by the internal Ridge selector and the Stage-B readers. Splits are the
repository's canonical PyG ZINC `subset=True`: **train 10 000 / valid 1 000**.

---

## 0. The single question

PSCD-SC-v0 showed that a *strong* port-operator reader on the frozen
**frequency** dictionary does not follow the raw graph (valid 0.263 vs raw
0.136) while fitting train better — a frequency-induced abstraction ceiling.
This round changes **only the dictionary-selection rule** and asks:

> If reader, ports, vocabulary size and motif size are frozen, does changing
> only the dictionary-selection principle from frequency to task+MDL materially
> improve held-out prediction?

$$
\Delta_{\rm dictionary} = MAE_{\rm FREQ} - MAE_{\rm TM}.
$$

## 1. What this round does NOT re-litigate / forbids

* No change to the reader architecture (PSCD-SC-v0 `StrongPortOperator` reused
  verbatim), no port / vocabulary-budget / motif-size sweep, no K sweep.
* No overlapping motifs, no differentiable/Gumbel tokenizer, no new port
  representation, no GW/FGW, no computational-compression objective.
* No handcrafted chemistry in motif identity.
* Official test never loaded. Official valid never used to choose a dictionary
  merge.

## 2. Data discipline (frozen before the run)

* Canonical train = 10 000, official valid = 1 000.
* Deterministic split of the canonical train (**seed 0**, fixed permutation):
  **9 000 dictionary-discovery** + **1 000 internal-monitor**.
* Indices are written to `results/pscd_tmdl/split_indices.json` and reused.
* **Every** merge choice uses only the 9 000 discovery graphs. The 1 000
  internal-monitor graphs participate in **no** merge decision; they are used
  only for the post-hoc overfitting diagnostic.
* The official valid split is touched only after the dictionary is fully frozen
  and the Stage-B reader has finished training.

## 3. Common initial state and frozen limits

* Initial partition `P_G^(0) = {{v} : v in V_G}` (singleton atoms), identical for
  every arm.
* `max motif size = 8`, **exactly 64 merge steps**, identical candidate
  enumeration for every arm.
* Motif identity = exact attributed graph canonical key (atom category + typed
  internal bonds, `pynauty`), no chemistry names.

## 4. Arms

| arm | selection rule | enters Stage B |
|---|---|---|
| **F (freq)** | `c* = argmax_c f_occ(c)` (standard graph-BPE), same 9 000 discovery split | yes |
| **TM** | MDL/reuse feasibility, task residual selects among feasible | yes |
| **task-only** | task residual alone (no MDL), shortlist top-256 by graph support | no (cheap diagnostic) |

All three start from the same singleton state and use the same candidate
enumeration; only the selection rule differs.

## 5. Candidate enumeration (every step)

For every adjacent occurrence pair `(o_i, o_j)` with `|o_i| + |o_j| <= 8`,
`c = G[V(o_i) ∪ V(o_j)]`. Candidate type = exact canonical attributed graph.
On discovery-9000 we record `f_occ(c)` (candidate-instance count) and
`f_graph(c)` (number of graphs with >= 1 instance). Candidates whose key is
already in the dictionary are skipped.

## 6. MDL / reuse eligibility (TM)

`c` is **eligible** iff

$$
f_{\rm graph}(c) \ge 45 \quad\text{and}\quad \Delta L_{\rm MDL}(c) < 0 .
$$

If more than 32 are eligible, the TM step keeps the **top-32 by MDL gain**
(`-ΔL`). If 32 or fewer, all are kept.

### 6a. First-order MDL proxy (explicit implementation choice)

The specification asks to reuse the PSCD-v0 `mdl_proxy`. That proxy is a
*corpus-level empirical-entropy* code: it assigns ~0 bits to the dominant atom
symbols and the unseen-symbol length (12 bits) to a brand-new mesoscale motif.
Used directly as a **per-merge** objective it can never express a positive merge
gain for the common atom pairs that BPE actually selects (verified analytically
and empirically: 0/36 eligible at step 0). For a *first-order merge* delta we
therefore use a transparent **two-part code in the BPE/two-part tradition**:

* dictionary cost of adding `c` (size prior + atom/bond codes + internal-pair
  bits — same decomposition as `mdl_proxy`'s `dict_cost`);
* corpus cost = uniform token-symbol code `log2(K+1)` per occurrence + empirical
  edge/port code (`bond_lp`, `port_lp`) computed from the **current** discovery
  codes.

$$
\Delta L_{\rm MDL}(c) = \big[L_{\rm dict}(D\cup\{c\}) - L_{\rm dict}(D)\big]
+ \sum_{G \text{ touched}} \big[L_{\rm graph}(\text{after}) - L_{\rm graph}(\text{before})\big].
$$

Only graphs the merge touches contribute, so the delta is computed
incrementally. The corpus-level empirical-entropy proxy (`mdl_proxy`) is still
reported in Stage A as the **compression measurement** (`ratio_total_over_atom`).

**Documented contingency.** If at some step *no* candidate satisfies `ΔL < 0`
(observed rarely at smoke scale), the step falls back to the top-32 by MDL gain
among `f_graph >= 45` and the step is flagged `forced_fallback=true`. This
guarantees the pre-registered 64 steps without a hyper-parameter sweep.

## 7. Task selector (cheap, deterministic, no sweep)

* Current dictionary gives each discovery graph a motif-count feature
  `x_G[k] = # occurrences of current motif/atom type k`.
* Fit **Ridge regression, alpha = 1** (fixed), on discovery-9000 predicting `y`.
* Residual `r_G = y_G - ŷ_G`.
* For candidate `c`, `z_c(G) = # legal occurrences of c in the current
  partition`; standardise over discovery-9000:
  `z̃_c = (z_c - mean) / (std + eps)`.
* `S_task(c) = | (1/N) Σ_G r_G z̃_c(G) |`.

## 8. Merge selection + tie-breaks

TM: `c* = argmax_{c in eligible} S_task(c)`.
task-only: `c* = argmax S_task(c)` over the top-256-by-graph-support shortlist.

If top candidates are within **1 % relative** of the max `S_task`, tie-break
deterministically by: (1) larger MDL gain (TM only), (2) larger graph support,
(3) larger occurrence frequency, (4) canonical motif key. Internal-monitor /
official-valid MAE are never consulted to choose a candidate.

## 9. Feedback loop

After applying the selected merge to the discovery partitions (and monitor
partitions), the motif-count features, Ridge fit, residual and next-step scores
are all recomputed. Thus `D^(t) -> task residual -> D^(t+1)` — discrete
task-coupled dictionary learning (not differentiable end-to-end).

## 10. Internal monitor policy

Every 8 merge steps (8, 16, ..., 64) we record discovery Ridge MAE and
internal-monitor Ridge MAE. **No stopping / selection uses the monitor.** The
run is fixed at 64 steps. Persistent discovery-improves-monitor-worsens is
recorded explicitly as **dictionary-selection overfitting**.

## 11. Stage A comparison (§13–15)

For all three dictionaries report: exact Decode reconstruction (must be 100 %);
discovery + monitor Ridge MAE; reuse (graph support, occurrence frequency,
support distribution, number of low-support motifs); compression (motif/atom
ratio, `mdl_proxy` ratio); boundary complexity (active ports, port/size ratio);
motif-size distribution; atom/bond category composition.

**Dictionary similarity (F vs TM):** identical positional merge rules, shared
rule keys, identical final motif types, Jaccard of motif types, motif-size
distribution, support distribution, mean boundary/port complexity, top-20
task-selected motifs (by `S_task`) and top-20 frequency motifs (no functional
labels assigned).

## 12. Stage B — decisive strong-reader test

Entered whenever the TM dictionary is exact, permutation-invariant and completes
64 steps without catastrophic fragmentation — regardless of the cheap monitor
numbers.

* Frozen dictionaries `D_FREQ`, `D_TM`.
* Reader = PSCD-SC-v0 `StrongPortOperator` **unchanged** (`d=64`, `L_outer=4`,
  `L_motif=2`, `L_intra=2`, shared state maps, alternating inter/intra-motif
  operator).
* Identical architecture / hidden size / depth / optimizer / budget /
  checkpoint rule for both arms; canonical B-Full protocol (Adam lr 1e-3,
  wd 1e-5, batch 128, clip 5, L1, max 240 epochs, patience 40, Top-5 soup).
* Reader training uses the **full 10 000** canonical train graphs (the 1 000
  internal-monitor graphs rejoin predictor training); evaluation on the official
  valid 1 000. Official test never loaded.
* Seed **0** for the primary comparison. If `|Δ| < 0.02`, run seed 1 to judge
  noise; if `Δ > 0.04`, seed 0 suffices for direction.

## 13. Gates (frozen)

* **Strong positive** — `Δ_dictionary >= 0.04`, `MAE_TM <= 0.22`, reuse healthy,
  MDL/compression not collapsed, train-valid gap smaller than the freq arm
  ⇒ *task supervision materially improves the learned graph dictionary*.
* **Moderate positive** — `0.02 <= Δ_dictionary < 0.04` with consistent seeds
  ⇒ *task coupling has real but insufficient signal*.
* **No useful signal** — `|Δ_dictionary| < 0.02` or TM worse, and no stable
  held-out improvement in the cheap probes ⇒ *simple task+MDL motif selection
  does not resolve the PSCD abstraction gap*.
* **Overfit failure** — discovery selector keeps improving, monitor keeps
  worsening, final official valid clearly worse than freq ⇒ *task-driven motif
  discovery overfits structural selection*.
* **MDL-importance criterion** — if task-only is better on discovery but worse
  on monitor / lower support / worse compression than TM, this supports
  *task relevance alone is insufficient; reuse/compression regularisation is
  necessary*.

## 14. Final conclusion first line (choose exactly one)

> **Task supervision materially improves the reusable compositional dictionary.**
> / **Task coupling provides a modest but real dictionary-selection signal.** /
> **Task-driven motif selection overfits and does not generalize.** /
> **Simple task+MDL motif selection does not resolve the PSCD abstraction gap.** /
> **Inconclusive due to implementation/optimization failure.**

## 15. Next decision (choose exactly one)

`formalize alternating task-driven dictionary learning` /
`run one reader-gradient dictionary-selection diagnostic` /
`strengthen dictionary regularization before further task coupling` /
`reconsider dictionary-as-primary representation` /
`fix experiment integrity before interpretation`.

This round does **not** implement the next stage.

## 16. Claim discipline

If positive, the accurate statement is **task-coupled discrete graph dictionary
learning**, not fully differentiable end-to-end dictionary learning. The
classical task-driven dictionary correspondence is
`G --D--> C_G --f--> y` with `y -> motif selection`.

## 17. Provenance

Code `tracks/ksvd/code/run_pscd_tmdl_dictionary.py`; tests
`tracks/ksvd/tests/test_pscd_tmdl.py`; results (git-ignored)
`tracks/ksvd/results/pscd_tmdl/`. Reader protocol and `StrongPortOperator` reused
from `run_pscd_strong_reader.py` (PSCD-SC-v0), which is not modified.
