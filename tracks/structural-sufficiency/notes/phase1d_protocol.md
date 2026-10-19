# SSRC Phase 1d protocol — is non-additive endpoint pairing necessary, and can capacity close the ZINC gap?

Protocol ID: `ssrc-phase1d-zinc-v1`  
Date: 2026-10-19  
Status: **FROZEN_PRE_GPU** (Phase A CPU audit is complete; Phase B/C GPU
decisions below are frozen before the first Phase 1d GPU trajectory)  
Upstream: `results/phase1c/REPORT.md`, `notes/phase1c_protocol.md`,
`notes/phase1b_corrigendum.md`, `configs/phase1c.yaml`  
Frozen code: `code/phase1d/` on top of the Phase 1b/1c modules (extension
hooks only; frozen Phase 1b/1c arms are not retrained or edited).

> This is an exploratory, single-seed-first screening round. It is allowed to
> end in `INCIDENCE_STATISTICS_SUFFICIENT_FOR_CURRENT_GAIN`,
> `REPRESENTATION_OR_INDUCTIVE_BIAS_GAP`, or `INSUFFICIENT_EVIDENCE`. No result
> here establishes a general law about message passing, graph neural networks,
> or chemistry. The official ZINC valid/test splits and the internal Reserved
> split are never read.

---

## 0. Why this round exists

Phase 1c established that true endpoint correspondence beats an
endpoint-marginal ablation at matched parameters, modules, FLOPs and readout
(`Gain_pair` +0.0682 at seed 0, +0.0309 at seed 1), and beats a
parameter-matched no-adjacency model (`Gain_over_capacity` +0.0529 at seed 0).
But Phase 1c did **not** establish that the useful computation is a genuinely
non-additive binary edge interaction. `R1_size` uses

\[
\phi(h_u,h_v,e_{uv}) = \mathrm{MLP}\big([h_u+h_v,\ |h_u-h_v|,\ h_u\odot h_v,\ e_{uv}]\big),
\]

which contains joint endpoint products and differences inside one edge. An
additive node--key statistic can encode the same local node, bond-type and
degree information without a joint endpoint nonlinearity. Phase 1c also left a
large absolute gap to the 6-layer local GINE reference `G_ref` (0.2366 vs
0.1653 MAE at the best R1_size seed).

This round asks two bounded questions:

- **B (mechanism).** At matched parameters, modules, readout, size features,
  labels and optimization, does the true-endpoint non-additive relation
  `phi(h_u,h_v,e_uv)` beat a strictly additive node--incidence statistic
  `psi(h_u,e_uv)+psi(h_v,e_uv)`?
- **C (absolute capability).** If one non-iterative candidate still has
  independent value, does scaling only its hidden width to about `G_ref`
  capacity (≈440K trainable parameters) substantially close the gap, at
  acceptable compute?

---

## 1. Phase A — CPU descriptive audit (completed before this protocol freeze)

Phase A used only frozen Phase 1b/1c Monitor predictions, the Fit/Monitor label
vault, and structure-only graph/patch features. No model was retrained. No
Reserved or official valid/test label was read. Artifacts:

- `code/phase1d/phase1d_features.py` →
  `results/phase1d/structural_features.npz` (structure-only)
- `code/phase1d/analyze_phase1d_cpu.py` →
  `results/phase1d/analysis_phase1d_cpu.json`

The audit is **descriptive**. It does not select an architecture, does not
turn a correlation into a causal mechanism, and does not contain a mechanical
"stop SSRC if a variable explains <0.02 MAE" rule. The old Phase 1c stopping
rule is void.

### 1.1 Fixed per-molecule differences

For Monitor molecule \(i\), with absolute error \(a_i(h)=|f_h(x_i)-y_i|\):

\[
\begin{aligned}
d_{\rm pair}(i) &= a_i(R1\_marginal\_size)-a_i(R1\_size),\\
d_{\rm capacity}(i) &= a_i(S\_cap\_size)-a_i(R1\_size),\\
d_{\rm gap}(i) &= a_i(R1\_size)-a_i(G\_ref).
\end{aligned}
\]

Positive means the second model is better. `d_gap` positive means R1_size is
worse than `G_ref`. Predictions are reindexed by the frozen Monitor molecule
ids; row alignment is asserted, not assumed.

Reported for each difference: MAE/RMSE, median AE, p90 AE, per-molecule win
rate, quantiles of \(d\), positive/negative fraction, the contribution of the
largest 1%/5%/10% \(|d|\) samples to the total mean difference, and diagnostic
truncation sensitivity (drop-top-|d| and winsorised/clipped versions).
Truncated values are **diagnostic only**; the untruncated MAE remains the
performance metric.

### 1.2 Fixed structural variables

Structure-only variables, computed for all 10,000 molecules from the frozen
cache (no labels, no external properties):

- size/topology: `n_atoms`, `n_bonds`, `rings_cyclomatic`, `rings_sssr`,
  `max_ring_size`, `ring_fusion_bonds`, `ring_shared_atoms`,
  `ring_bond_fraction`, `frac_ring_atoms`, `large_cycles_gt6` (SSSR rings with
  >6 atoms; a structure-only proxy for the target's `cycles` term),
  `mean_degree`, `max_degree`, `min_degree`, `frac_degree1`, `diameter`,
  `mean_pair_distance`, `frac_pairs_distance_gt4`;
- bond composition: `single_bonds`, `double_bonds`, `triple_bonds`,
  `double_bond_fraction`, `kekule6_ring_count` (6-rings with three double and
  three single ring bonds), `bond_type_entropy`;
- atom composition: `atom_type_entropy`;
- E2 patch size/repetition: `patch_size_mean`, `patch_size_max`,
  `patch_size_std`, `patch_whole_fraction`, `patch_dup_rate` (exact pynauty
  rooted-patch key duplicates), `patch_key_entropy`, `patch_max_multiplicity`,
  `patch_unique_count`, `mean_pool_bits`;
- root-vs-pool representation: `root_vs_patch_tv_mean` (total-variation
  distance between the root atom-type one-hot and the patch mean atom-type
  distribution), `patch_atom_entropy_mean`, `first_shell_mean`,
  `first_shell_max`, `second_shell_mean`.

The ZINC-12k cache has no explicit aromatic bond type: the 10k subset uses raw
bond values 1/2/3 (single/double/triple) and benzene-like rings are
Kekulé-encoded as alternating single/double bonds. `kekule6_ring_count` is the
structure-only aromatic-ring proxy. This is a data-encoding fact, not a model
input.

### 1.3 Fixed groups and interpretation rules

Groups are fixed from Fit-only structure distributions (integer edges:
`n_atoms` [0,21,24,27,∞); `rings_sssr` [0,2,3,4,∞); `max_ring_size`
[0,6,7,∞); `large_cycles_gt6` [0,1,2,∞); `kekule6_ring_count` [0,1,2,∞);
`ring_fusion_bonds`/`ring_shared_atoms` [0,1,3,∞); `diameter`
[0,11,13,15,∞); patch/residual quartile edges [0,5.94,6.10,6.27,∞),
[0,0.05,0.11,0.17,∞), [0,0.33,0.39,0.44,∞), [0,0.459,0.523,0.569,∞)).
For every bin the report states group size, fraction, model MAE, mean \(d\),
the weighted contribution `fraction × mean(d)`, and win rates. No small-bin
conditional MAE is promoted to a primary bottleneck without its weighted
contribution.

Phase A findings are used to **interpret** B/C, not to add modules:

- The relation/capacity gains are not broad: the 40 Monitor molecules with
  `large_cycles_gt6 ≥ 1` (4.0%, equivalently `max_ring_size ≥ 7`) carry more
  than the entire positive `d_pair`/`d_capacity` mean. On the 96% complement,
  `d_pair = −0.0107` (CI −0.030…+0.007) and `d_capacity = −0.0280`
  (CI −0.047…−0.012): R1_size is slightly worse than the marginal/capacity
  controls there.
- The `G_ref` gap is broader: the same 4% tail contributes 0.0289 of the
  0.0713 MAE gap, but the complement still has a positive gap of +0.0442
  (CI +0.027…+0.064).
- The target is `normalized(logP − SAS − cycles)`, where `cycles` counts rings
  with >6 atoms. The large-ring subgroup is therefore label-mechanism-coupled;
  a large-ring improvement is not by itself evidence for a general relational
  mechanism.
- Per-molecule seed consistency is weak: `d_pair` seed0 vs seed1 Spearman
  0.337, sign agreement 0.606, top-decile Jaccard 0.198. The aggregate gain
  replicates, the molecule-level attribution does not.

These facts make the B question sharper: **is the large-ring/tail rescue
specific to non-additive endpoint pairing, or can additive node--incidence
statistics reproduce it?**

---

## 2. Phase B — `R_incidence` mechanism control

### 2.1 Model definition (frozen)

`R_incidence` uses the **bitwise-identical** Phase 1b/1c E2 patch trunk,
the same true root adjacency, the same atom/bond features, the same size
sidecar \(s(G)\), the same size-aware node readout, the same edge-readout
module, and the same regression head as `R1_size`.

Only the edge relation function changes. For every physical undirected bond
\(\{u,v\}\) (the unique copy with `src < dst`):

\[
r_{uv} = \psi(h_u,e_{uv}) + \psi(h_v,e_{uv}),
\qquad
\psi(h,e)=\mathrm{Linear}_{h+e\to r}(\mathrm{SiLU}(\cdot)) \to \mathrm{Linear}_{r\to h},
\]

with one shared `psi` for both endpoint slots. The edge aggregate is the
mean and sum of \(r_{uv}\) over the true physical bonds, exactly the same
aggregation rule as `R1_size`. The node readout receives the same
`mean`/`sum` patch-token pools and the same \(s(G)\).

There is **no** \(h_u\odot h_v\), no \(|h_u-h_v|\), no
\(\mathrm{MLP}(h_u+h_v)\), and no other joint-endpoint nonlinearity inside an
edge. Because

\[
\sum_{\{u,v\}\in E} r_{uv}
= \sum_{v\in V}\sum_{e\ni v}\psi(h_v,e),
\]

the edge sum is a sum of node--incidence contributions; the edge mean is the
same quantity divided by the (preserved) edge count. This is the intended
"node--key statistics only" control.

### 2.2 Capacity and compute matching

Frozen base widths: hidden 64, token FFN 128, `R1_size` relation hidden 128,
edge-readout hidden 64. `R_incidence` uses `incidence_rel_hidden = 251`.
Predicted parameter counts from the frozen builders:

| Arm | Params | Relation params | Forward FLOPs (batch 64) |
|---|---:|---:|---:|
| `R1_size` | 115,977 | 33,472 | 646,843,008 |
| `R_incidence` | 115,952 | 33,447 | measured in CPU gate; expected ≈ R1_size |

The −25 parameters (−0.02%) are reported, not hidden. `R1_size` applies `phi`
to both directed copies of every physical bond; `R_incidence` applies the two
endpoint terms once per physical bond. Both therefore perform two relation-MLP
applications per physical bond, which is the intended compute match. Actual
forward FLOPs and step time are measured on CPU before GPU and recorded for
the GPU run. If the measured FLOPs or step time differ materially, the
comparison is reported as capacity/FLOP-approximate, not as a pure causal
intervention.

### 2.3 Information access and equality checks

`R_incidence` is registered with the same `RELATIONAL_SIZE_KEYS` batch guard
as `R1_size`: same patch features, same `root_edge_index`/`root_edge_attr`,
same `root_batch`, same `mol_size_feat`, same `y`/bookkeeping keys. It has the
same `SharedTrunk`, `SizeAwareNodeReadout`, `EdgeReadout` and
`RegressionHead` classes and state-dict shapes. It is trained with the same
frozen recipe: L1 loss, AdamW 1e-3, weight decay 1e-5, cosine `T_max=300`,
`eta_min=1e-5`, batch 64, 300 epochs, Monitor early stopping patience 30,
`min_delta=1e-4`, grad clip 5.0, dropout 0.1, seed 0, same `default_rng([seed,
epoch])` data order. No post-result tuning.

### 2.4 GPU run and endpoints

Phase B trains exactly one trajectory: `R_incidence`, ZINC Fit 8000 / Monitor
1000, seed 0. The frozen `R1_size` seed-0 JSON is reused; it is not retrained.

Primary endpoint:

\[
\mathrm{Gain}_{\rm joint}
= \mathrm{MAE}(R\_incidence) - \mathrm{MAE}(R1\_size).
\]

Positive means the non-additive true-endpoint `R1_size` is better. Report
Monitor MAE, RMSE, per-molecule win rate, median/p90 AE, paired bootstrap CI
(5,000 molecule resamples), and the pre-specified Phase A subgroup readout
(`large_cycles_gt6 ≥ 1` vs complement). The subgroup is secondary and
descriptive; it does not replace the overall endpoint.

Bands (same as Phase 1b/1c ZINC): negligible `<0.01`, weak `0.01–0.02`,
meaningful `≥0.02`. A "clear" direction requires the paired CI to exclude zero
and the RMSE sign to agree.

### 2.5 Pre-registered B decision

- **`PAIRWISE_INTERACTION_PROMISING`** — `Gain_joint ≥ 0.02`, CI95 lower > 0,
  RMSE sign agrees. Keep "non-additive endpoint interaction" as a candidate
  mechanism. If the gain is concentrated in the large-ring subgroup, say so;
  do not generalize it to all chemistry.
- **`INCIDENCE_STATISTICS_SUFFICIENT_FOR_CURRENT_GAIN`** — `Gain_joint < 0.02`
  in absolute value (including a negative value), regardless of whether the
  CI excludes zero, **unless** the point estimate is in the weak band and the
  CI lower is > 0, in which case report `INSUFFICIENT_EVIDENCE` and do not
  raise the pairwise claim. In the sufficient case, lower the Phase 1c claim:
  the current gain is explainable by additive node--incidence statistics
  (plus the local patch trunk and nonlinear graph readout), not by a proven
  non-additive pairwise interaction. A negative `Gain_joint` (incidence
  better) strengthens this conclusion and is also reported as an efficiency
  advantage.
- If `R1_size` is better but the effect is weak (0.01–0.02) or the CI crosses
  zero: **`INSUFFICIENT_EVIDENCE`**; do not buy a second seed to rescue it.
  Phase C selection follows the rule in §3.

No second seed is run in this round. A molecule bootstrap is not training-seed
replication.

---

## 3. Phase C — one bounded capacity-scaling test

### 3.1 Candidate selection (frozen)

- If Phase B yields `PAIRWISE_INTERACTION_PROMISING`, scale **`R1_size`**.
- Otherwise scale **`R_incidence`**.

This rule is fixed before the Phase B GPU run. It does not select a model by
Monitor performance after the fact; it selects the candidate whose mechanism
still has independent support, and otherwise scales the more parsimonious
additive candidate.

### 3.2 Frozen capacity configurations

Only hidden width is scaled; depth, radius, local layers, token-FFN blocks,
readout rule, loss, schedule and data are unchanged. Derived widths scale
proportionally with hidden (`ffn_hidden = 2*hidden`,
`rel_hidden = 2*hidden`, `edge_readout_hidden = hidden`, and for
`R_incidence`, `incidence_rel_hidden = 2*hidden`). These widths were computed
from parameter counting only, not from a performance grid.

| Candidate | hidden | ffn/rel/incidence-rel | edge-readout | Params | vs G_ref (439,559) |
|---|---:|---:|---:|---:|---:|
| `R1_size_cap` (if B says pairwise) | 128 | 256 / 256 / — | 128 | 453,129 | +3.09% |
| `R_incidence_cap` (else) | 136 | 272 / 272 / 272 | 136 | 436,841 | −0.62% |

Only **one** of these configurations is run, seed 0, once. No width grid, no
second seed, no LR/dropout/schedule change.

### 3.3 Endpoints and decision

Let `base` be the seed-0 candidate from Phase B (`R1_size` or
`R_incidence`), and `cap` its capacity-scaled version.

\[
\mathrm{Gain}_{\rm cap} = \mathrm{MAE}(base)-\mathrm{MAE}(cap),
\qquad
\mathrm{Gap}_{\rm cap} = \mathrm{MAE}(cap)-\mathrm{MAE}(G\_ref).
\]

Positive `Gain_cap` means capacity helps; positive `Gap_cap` means the scaled
candidate is still worse than `G_ref`. Report MAE/RMSE, paired bootstrap CI,
win rate, tail errors, the large-ring subgroup, parameters, measured forward
FLOPs, median step time, peak GPU memory, GPU seconds, best epoch/epochs run,
train MAE, and train--Monitor gap. Compare all of these to the frozen `G_ref`
run and to the unscaled candidate.

**Convergence gate.** The capacity run is "converged under the frozen recipe"
only if early stopping actually fires before the 300-epoch budget (i.e. the
run does not end at the budget edge and the best epoch is not the last epoch).
If the run is still improving at the budget edge, report
**`OPTIMIZATION_UNRESOLVED`** and do not interpret a null as a capability
ceiling.

Pre-registered labels:

- **`CAPACITY_SCALING_PROMISING`** — `Gain_cap ≥ 0.02`, CI95 lower > 0, RMSE
  sign agrees, and training converged. If `Gap_cap` is still ≥0.02, also
  report `REPRESENTATION_OR_INDUCTIVE_BIAS_GAP` (capacity helped but did not
  make the method competitive). If `Gap_cap < 0.02` and compute is not
  prohibitive, state that the method is approaching the reference.
- **`REPRESENTATION_OR_INDUCTIVE_BIAS_GAP`** — `Gain_cap < 0.01` in absolute
  value (or the CI crosses zero) with a converged run. Capacity is not the
  main bottleneck; the next question is patch-token representation, local
  computation range, or inductive bias, not more width.
- **`OPTIMIZATION_UNRESOLVED`** — the convergence gate fails. Report it
  honestly; do not call it evidence for or against capacity.
- If `Gain_cap ≥ 0.02` but the FLOPs or step time is more than ~2× `G_ref`,
  record an explicit **efficiency obstacle**: the method may be capacity-
  limited but not competitive at matched compute.

A second seed, matched-capacity non-relational controls, and extra tasks are
**not** run in this round even if capacity is promising. They require a new
pre-registration.

---

## 4. Budget and run matrix

| Phase | Runs | Cap (GPU-h) | Trigger |
|---|---|---:|---|
| P0 | CPU gates/tests + one short GPU smoke | 0.03 | always |
| B | `R_incidence` seed 0 | 0.30 | after frozen commit |
| C | one capacity run (candidate fixed by §3.1) seed 0 | 0.80 | after B analysis and only if C is informative under §3 |
| Reserve | failure/smoke | 0.20 | — |
| **Total** | | **≤2.00** | hard cap |

Unused budget is allowed. A run that cannot finish within the remaining budget
is not started. All GPU spend is recorded in
`results/phase1d/run_manifest.tsv` with the `rr` run id, host/pool, commit,
GPU seconds, job occupancy, best epoch and output hash.

---

## 5. CPU gates and tests (must pass before GPU)

`code/phase1d/phase1d_gates.py` writes
`results/phase1d/cpu_gates_phase1d.json`. `tests/test_phase1d.py` and the
frozen Phase 1b/1c tests must pass. Required checks:

1. Frozen Phase 1b split/label/cache hashes unchanged; label vault retains
   9,000 Fit+Monitor labels, 0 Reserved, 0 official valid/test.
2. `R_incidence` uses exactly the `R1_size` batch keys and the same
   trunk/readout/head classes; no endpoint-marginal or graph-only key leak.
3. Node renumbering (arbitrary permutation) leaves `R_incidence` output and
   edge aggregate unchanged.
4. Reversing every undirected edge direction leaves output and edge aggregate
   unchanged.
5. A valid rewiring that preserves each node's attribute and incident
   bond-type multiset (and edge count) leaves the `R_incidence` **edge
   aggregate** unchanged; the same rewiring changes `R1_size`'s edge
   aggregate (different endpoint pairing).
6. `R1_size` can produce different outputs for two different endpoint pairings
   of the same node/bond multiset; this is the positive control for the
   mechanism contrast.
7. Parameter count: `R_incidence` within ±1% of `R1_size`; capacity candidate
   within ±5% of `G_ref` 439,559. Forward FLOPs and CPU step time measured for
   both base arms and the capacity candidate.
8. Forward/backward gradients are finite and nonzero for the trunk,
   `psi`/`phi`, edge readout, node readout and head; the incidence relation
   gradient is live for both endpoint slots.
9. `R_incidence` has no joint-endpoint nonlinearity: the CPU gate inspects the
   relation input width (hidden+edge_dim, not 3*hidden+edge_dim) and confirms
   the module is the additive `psi(h,e)` form.
10. Data isolation: no official valid/test or internal Reserved labels are
    instantiated; all run JSONs record `official_valid_test_labels_read=False`,
    `internal_reserved_labels_read=False`, `reserved_labels_retained=0`.
11. Frozen Phase 1b/1c anchors unchanged (`S0 0.317303`, `R1 0.260161`,
    `M2 0.252220`, `G_ref 0.165290`); Phase 1d does not retrain them.
12. The Phase A structural feature table is structure-only and matches the
    frozen cache/split hashes.

CPU smoke: first 128 Fit / 64 Monitor, 2 epochs, for `R_incidence` and the
chosen capacity candidate. Smoke numbers are not results.

---

## 6. Prohibitions

- No new global token, global attention, GPS/GRIT/ESA+ implementation.
- No radius 2/3/4 grid, no extra message-passing layer, no iterative update.
- No reopening the closed BJOC recipe search.
- No post-Monitor LR/dropout/width/schedule changes.
- No official ZINC valid/test or internal Reserved read for selection.
- No retraining of frozen Phase 1b/1c arms just to increase run counts.
- No second seed or extra capacity configuration in this round.
- No claim that a structural correlation is a causal mechanism or a new
  algorithm; `R_incidence` is a control, not a proposed method.

---

## 7. Artifacts and reproduce

```bash
# CPU: structure-only features (all 10k rows)
uv run python tracks/structural-sufficiency/code/phase1d/phase1d_features.py --rows all

# CPU: Phase A audit
uv run python tracks/structural-sufficiency/code/phase1d/analyze_phase1d_cpu.py

# CPU: gates + tests
uv run python tracks/structural-sufficiency/code/phase1d/phase1d_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1d/tests/test_phase1d.py
uv run pytest -q tracks/structural-sufficiency/code/phase1c/tests/test_phase1c.py
uv run pytest -q tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py

# GPU: Phase B (through rr; one process job, one GPU)
rr deploy res
rr run res ssrc-p1d-zinc-incidence-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/runs/phase1d \
  -- python tracks/structural-sufficiency/code/phase1d/run_phase1d.py \
     train --dataset zinc --model R_incidence --seed 0 --device cuda

# GPU: Phase C (model/width fixed by the frozen B decision rule)
rr run res ssrc-p1d-zinc-capacity-s0 --pool res-gpu1 \
  --result tracks/structural-sufficiency/runs/phase1d \
  -- python tracks/structural-sufficiency/code/phase1d/run_phase1d.py \
     train --dataset zinc --model R1_size_cap --seed 0 --device cuda
# or --model R_incidence_cap
```

The exact remote run IDs, commits, GPU seconds, occupancy, best epochs and
output hashes are recorded in `results/phase1d/run_manifest.tsv`.

---

## 8. Non-claims

1. `R_incidence` is not "the same information" as `R1_size`: it removes the
   joint endpoint pairing but keeps the true root incidence, bond types,
   patch tokens and size features.
2. The incidence model's edge aggregate is permutation-invariant and
   rewiring-invariant under the stated local-multiset-preserving rewiring; this
   is a property of the additive form, not evidence that it is chemically
   meaningful.
3. A single seed and a Monitor bootstrap do not establish formal significance
   or training-seed robustness.
4. The large-ring subgroup is partly a target-label mechanism (the target
   includes `-cycles`); a gain there is not a general chemical or relational
   claim.
5. `G_ref` is a 6-layer local GINE, not global attention and not a paper
   reproduction.
6. Internal train-only Monitor numbers are not official ZINC test numbers.
7. No result proves \(E_2\) task-information sufficiency/insufficiency in the
   Bayes-risk sense.
