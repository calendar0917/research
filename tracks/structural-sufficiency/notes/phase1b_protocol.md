# SSRC Phase 1b protocol — S0 / R1 / M1 / M2 / G_ref

Protocol IDs: `ssrc-phase1b-zinc-v1`, `ssrc-phase1b-lipo-v1`  
Date: 2026-10-19 (frozen before any Phase 1b GPU trajectory)  
Status: **FROZEN_PRE_GPU**  
Upstream: `notes/research_core.md`, `notes/phase0_corrigendum.md`,
`notes/phase1a_protocol.md`, `results/phase1a/REPORT.md`

> This is an exploratory, single-seed-per-arm screening round, not a
> confirmatory test. It is allowed to end in `INSUFFICIENT_EVIDENCE` or
> `PERFORMANCE_GAP_DOMINATES`. No result here can establish that message
> passing is generally necessary or unnecessary.

---

## 1. The question

Phase 1a compared two models that both saw only the \(E_2\) patch multiset and
found that adding cross-patch self-attention (X) did not help; S was at least
as good. That result says nothing about **real root-to-root connections**,
because neither arm saw the original graph adjacency.

Phase 1b gives the arms different *information access* in a controlled ladder:

| Arm | Local \(E_2\) patch encoder + token FFN | True root graph edges / bond types | Relational computation |
|---|---|---|---|
| **S0** | yes | **no** | none (mean pool) |
| **R1** | yes | **yes** | one-shot, endpoint-swap-invariant edge relation; node + edge readout; **no node-state update** |
| **M1** | yes | **yes** | the same relation function aggregated into **one** node-state update; mean pool |
| **M2** | yes | **yes** | the **same weight-tied** relation/update module applied **twice**; mean pool |
| **G_ref** | no | yes (raw atom graph) | strong 6-layer residual GINE + FFN blocks; sum+mean readout |

The four SSRC arms share exactly the same local radius-2 patch encoder and the
same two residual per-token FFN blocks. For a fixed seed their shared trunk
initialisation is bitwise identical (CPU gate). S0/R1/M1/M2 differ only in the
root-graph computation. M1 and M2 are parameter-identical; M2 applies the same
module twice, so the M1→M2 contrast isolates iteration from capacity.

### 1.1 Endpoints

For each task, using Monitor MAE (lower is better):

\[
\begin{aligned}
\mathrm{Gain}_{\rm relation} &= \mathrm{MAE}(S0)-\mathrm{MAE}(R1) &&(+:\ R1\text{ better})\\
\mathrm{Gain}_{\rm update} &= \mathrm{MAE}(R1)-\mathrm{MAE}(M1) &&(+:\ M1\text{ better})\\
\mathrm{Gain}_{\rm iteration} &= \mathrm{MAE}(M1)-\mathrm{MAE}(M2) &&(+:\ \text{two rounds better})\\
\mathrm{Gap}(h) &= \mathrm{MAE}(h)-\mathrm{MAE}(G_{\rm ref}) &&(+:\ h\text{ worse})
\end{aligned}
\]

`Gain_relation` is **not** a pure computation-mechanism effect: R1 sees edges
that S0 does not. `Gain_update` and `Gain_iteration` are relational-computation
contrasts under the same edge access; only `Gain_iteration` is parameter-matched
by construction. `Gap` is an application-performance distance, not a causal
decomposition.

### 1.2 Practical-significance bands (pre-registered)

| Task | negligible | weak / unresolved | potentially meaningful |
|---|---:|---:|---:|
| Lipophilicity MAE | \(<0.02\) | \(0.02\)–\(0.04\) | \(\ge 0.04\) |
| ZINC MAE | \(<0.01\) | \(0.01\)–\(0.02\) | \(\ge 0.02\) |

A "clear" direction also requires the same sign on both tasks, or a
task-specific mechanism argument written before the second seed is bought.
These are exploratory bands, not significance or equivalence tests.

---

## 2. Data and splits

### 2.1 ZINC-12k

- Official PyG `ZINC(root=data/ZINC, subset=True, split="train")`; 10,000 rows.
- **Official valid and test are never instantiated or read.** The code path
  only constructs the train split.
- Internal split reuses the repository-frozen TCCD-v0 permutation:
  ```python
  perm = np.random.default_rng(20260922).permutation(10000)
  fit      = perm[:8000]
  dev      = perm[8000:]
  monitor  = dev[:1000]      # permutation order preserved
  reserved = dev[1000:]
  ```
  CPU gate `zinc_split_matches_tccd_v0` verifies the first 8,000 and the
  remaining 2,000 are exactly `run_tccd_v0.internal_split(10000)`.
- Target: PyG `logP_SA_cycle_normalized` (benchmark-normalized penalized logP).
  No additional standardization; mean \(\approx 0.015\), std \(\approx 2.011\)
  over official train. Literature ZINC MAE numbers are on the same normalized
  target but use different splits and training budgets; they are **not**
  directly rankable against internal Monitor numbers.
- Raw fingerprints (recorded by `prep`):
  - `data/ZINC/raw/train.pickle` sha256 `9c76df5a…`
  - `data/ZINC/raw/train.index` sha256 `575d6acd…`
  - `data/ZINC/subset/processed/train.pt` sha256 `e0f1837b…`
- Fit/Monitor labels only in `results/phase1b/zinc_labels_fit_monitor.json`;
  Reserved labels are not written and are never scored.

### 2.2 Lipophilicity

- Reuse the Phase 1a frozen DeepChem 2.8.0 scaffold split:
  Fit/Monitor/Reserved = 2352/504/504; scaffold overlap 0.
  Split file sha256 `9fa96d49…`; row hashes match `STATE.yaml` exactly.
- Reuse the Phase 1a Fit/Monitor-only label artifact
  `results/phase1a/labels_fit_monitor.json` sha256 `0260a390…`.
- Official valid/test and internal Reserved are never read or scored.
- Metric: raw-scale MAE and RMSE. Units are logD; **not** comparable in
  absolute value to ZINC's normalized penalized logP.

### 2.3 Historical blind disclosure

Phase 0 used all 3,360 official-train Lipophilicity labels for LOO diagnostics,
and historical ZINC experiments used the official train labels. The internal
Reserved rows are therefore not "blind from the beginning of the project".
Phase 1b discipline is: no Reserved label is loaded or scored, and the label
vault audit records `reserved_labels_retained=0`.

---

## 3. Inputs and information discipline

Common raw information for S0/R1/M1/M2:

- One exact rooted induced radius-2 patch per atom (root first; remaining nodes
  ordered by (distance, atom index)); node feature = raw atom feature block +
  root flag + 3-way distance one-hot; edge feature = raw bond feature block.
- No atom index, no patch-to-patch incidence, no global descriptor, no
  ring/distance statistics, no RDKit-recomputed target component.

Root graph access for R1/M1/M2:

- The original molecular graph restricted to root atoms, with both directed
  records per physical bond and the raw bond feature.
- Dense storage order is atom order but all operators are permutation
  equivariant and pooling is permutation invariant; a relabelling gate checks
  output invariance.
- R1 counts each physical undirected bond once in the edge readout
  (`src < dst`). M1/M2 aggregate both directions in the node update, the
  standard undirected message-passing convention. Both protocols are explicit.

G_ref sees only the raw atom graph (atom/bond features + adjacency), **not** the
E2 patch representation. It is an application reference, not an information
control.

Leakage guards (CPU gates):

- S0 batch keys contain no root edges; R1/M require them; G_ref contains no
  patch fields.
- Permutation invariance, batch-composition invariance, endpoint-swap
  invariance, relation/update gradient liveness.
- Patch tensors are bitwise identical to Phase 1a for real Lipophilicity
  molecules, and S0 is bitwise the Phase 1a `ModelS` after a weight-name map.
- The cache contains no labels.

---

## 4. Training recipe (all arms, both tasks)

| Item | Value |
|---|---|
| loss | L1 on raw target (no extra standardization) |
| optimizer | AdamW, lr \(10^{-3}\), weight decay \(10^{-5}\) |
| schedule | cosine, \(T_{\max}\) = epochs, \(\eta_{\min}=10^{-5}\) |
| batch | 64 |
| ZINC epochs | 300, early stop Monitor MAE patience 30, min_delta \(10^{-4}\) |
| Lipophilicity epochs | 200, patience 30 |
| grad clip | 5.0 |
| dropout | 0.1 |
| seeds | 0 (Phase A); second seed only under the Phase D rule |
| data order | same permutation `default_rng([seed, epoch])` for all arms |
| model selection | best Monitor MAE; Monitor also used for early stopping |
| soup / ensemble / SSL / external data | **none** |

Monitor is used for early stopping and arm comparison; this is an exploratory
screen, not a held-out confirmation. Reserved is untouched.

### 4.1 Capacity and cost (CPU-measured; reported per run)

Lipophilicity: S0 59,011; R1 97,731; M1 = M2 110,211; G_ref 447,751.
ZINC: S0 56,963; R1 94,659; M1 = M2 107,139; G_ref 439,559.

FLOPs and step time are measured on a fixed batch for every run. R1/M1/M2 are
not parameter-matched to S0; this is disclosed, and a capacity/shuffled-edge
control is a Phase B purchase only if R1 shows a meaningful gain.

---

## 5. Phase A run matrix

10 trajectories, seed 0:

| Task | Arms |
|---|---|
| Lipophilicity (Fit 2352, Monitor 504) | S0, R1, M1, M2, G_ref |
| ZINC (Fit 8000, Monitor 1000) | S0, R1, M1, M2, G_ref |

No official valid/test, no Lipo Reserved, no ZINC Reserved.

Before launch, a GPU smoke on each task measures step time and peak memory and
confirms that all five arms execute and produce finite Monitor MAE.

---

## 6. Decision logic (pre-registered)

The first read is the Phase A table. We do **not** automatically run all
branches.

1. **Sanity failure** (any arm not better than the training-mean constant
   predictor, flat/non-finite training curve, or label/guard failure):
   debug implementation/optimization first; do not interpret the mechanism.
2. **B1 — one-shot relation signal.** R1 beats S0 by at least the "potentially
   meaningful" band on at least one task and has the same sign on the other,
   while M2 does not beat M1 by that band. Purchase **one** control: R1 with a
   controlled broken root adjacency (same edge count and edge-type multiset;
   degree-preserving where possible) on the task with the larger gain. If the
   broken-edge R1 is close to true-edge R1, downgrade "true relational
   correspondence" and treat the gain as a generic edge channel.
3. **B2 — iteration signal.** M2 beats M1 by at least the band on at least one
   task with the same sign on the other. Purchase **one** control: M1 trained
   for the same number of relation rounds/FLOPs as M2 (e.g. 2× epochs) on the
   larger-gain task. If the longer M1 closes the gap, iteration is not
   separated from extra computation.
4. **B3 — all arms close, G_ref competitive.** No meaningful S0/R1/M
   difference. Do not declare "structure is sufficient"; report the
   non-identification and, if the cost table supports it, consider a static
   efficiency claim. Stop relational-architecture expansion if no testable
   condition emerges.
5. **B4 — all candidates clearly worse than G_ref.** Run a performance-gap
   audit (fit, inputs, receptive field, preprocessing, capacity, optimization)
   rather than adding modules. Report `PERFORMANCE_GAP_DOMINATES` if the gap is
   large and unexplained by a single fixable factor.
6. **B5 — tasks disagree.** Check splits, target scale, input fields, training
   curves and label noise before proposing a task-conditioned hypothesis. A
   task hypothesis must predict a cheap diagnostic on existing train-only data
   before more GPU is bought.

Phase C (at most two evidence-backed improvements) starts only if a concrete
bottleneck is identified and a matched control can decide its cause. No
hyper-parameter search, no SSL objective search, no junction ownership, no
fixed signatures, no arbitrary distance channels, no larger recipe sweep.

Phase D buys a second seed only for a candidate that passed a gate. It is a
paired confirmation against the most important control, not a 5-seed paper
claim. If run-to-run noise is comparable to the effect, report
`INSUFFICIENT_EVIDENCE` rather than a confirmed gain.

---

## 7. Resource plan and stop conditions

| Phase | GPU-hour cap | Purpose |
|---|---:|---|
| A | 2.0 | anchors + one-seed S0/R1/M1/M2/G_ref |
| B | 2.0 | one controlled mechanism contrast |
| C | 2.0 | at most two evidence-backed fixes |
| D | 1.5 | second seed / paired confirmation |
| reserve | 0.5 | smoke, failure recovery |
| **total** | **8.0** | hard cap |

Wall-clock cap: 8 hours from the start of this round. If the remaining window
cannot complete a scientifically interpretable contrast, do not start it.
Unused budget is not a failure.

Stop immediately if:

- all arms fail the constant-predictor sanity check after a correct debug;
- the only proposed next step is another architecture variant without a
  pre-written discriminating prediction;
- the candidate is far behind G_ref and no single evidence-backed bottleneck
  can be named;
- the result is dominated by run-to-run noise that a second seed cannot
  resolve within the remaining budget.

---

## 8. Non-claims

1. Strong structural encoders reducing message-passing need is not new
   (Eijkelboom et al. 2023; GSN, GraphGPS, GRIT, CSA, ESA+, MoSE/SPSE).
2. R1's edge relation and M's message passing have many precedents; the
   candidate contribution is the **controlled decomposition** of access,
   one-shot composition, state update and iteration, not the operators.
3. A single split, one seed and one recipe cannot establish a general
   task-conditioned law.
4. No claim that message passing is generally necessary or unnecessary.
5. No claim that E2 is information-sufficient or insufficient for either task;
   only the observed predictive behavior of these finite models.
6. ZINC internal Monitor numbers are not official-test or literature numbers;
   Lipophilicity MAE/RMSE are not comparable in absolute units to ZINC.

---

## 9. Reproduce

```bash
# 1) structures + frozen split + Fit/Monitor labels + patch cache (CPU)
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py prep --dataset zinc
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py prep --dataset lipo

# 2) CPU gates (23/23)
uv run python tracks/structural-sufficiency/code/phase1b/phase1b_gates.py

# 3) CPU smoke (one arm)
uv run python tracks/structural-sufficiency/code/phase1b/run_phase1b.py train \
  --dataset lipo --model S0 --seed 0 --device cpu --smoke --epochs 2 --patience 1

# 4) GPU trajectory (through rr; one job per arm)
rr deploy res
rr run res ssrc-p1b-zinc-S0-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/runs/phase1b \
  -- python tracks/structural-sufficiency/code/phase1b/run_phase1b.py \
     train --dataset zinc --model S0 --seed 0 --device cuda
```

Frozen artifacts:

| Item | Value |
|---|---|
| ZINC split | `results/phase1b/split_zinc_phase1b.json` sha256 `fbca5fd5…`; row hashes Fit `8351cbee…`, Monitor `2e4801c6…`, Reserved `cf1e8c6c…` |
| ZINC labels | `results/phase1b/zinc_labels_fit_monitor.json` sha256 `631dae5f…` |
| Lipo split | `results/phase1a/split_phase1a.json` sha256 `9fa96d49…` |
| Lipo labels | `results/phase1a/labels_fit_monitor.json` sha256 `0260a390…` |
| CPU gates | `results/phase1b/cpu_gates_phase1b.json` 23/23 PASS |
