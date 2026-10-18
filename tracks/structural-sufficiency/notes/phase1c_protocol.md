# SSRC Phase 1c protocol — relation correspondence, capacity, and readout

Protocol ID: `ssrc-phase1c-zinc-v1`  
Date: 2026-10-19 (frozen before any Phase 1c GPU trajectory)  
Status: **FROZEN_PRE_GPU**  
Upstream: `results/phase1b/REPORT.md`,
`notes/phase1b_corrigendum.md`, `notes/phase1b_protocol.md`,
`configs/phase1b.yaml`  
Frozen code: `code/phase1c/` on top of `code/phase1b/` (backward-compatible
extension hooks only; Phase 1b arms keep their original construction path).

> This is an exploratory, single-seed-first screening round, not a
> confirmatory test. It is allowed to end in `NO_ENDPOINT_CORRESPONDENCE_GAIN`,
> `CAPACITY_READOUT_EXPLAINS`, or `PERFORMANCE_GAP_DOMINATES`. No result here
> establishes that message passing is generally necessary or unnecessary.

---

## 1. Why this round exists

Phase 1b's `PURSUE_RELATIONAL_SUFFICIENCY` was an exploratory judgment, not a
confirmed law. Its three unresolved confounds are:

1. **Capacity/readout.** R1 has 94,659 parameters vs S0's 56,963. R1's readout
   is `concat(mean node, mean edge)`; S0's is mean-only. The broken-adjacency
   control kept R1's architecture, so it did not decompose capacity.
2. **Broken adjacency is not a surgical correspondence null.** It mixed
   endpoint errors, bond-feature/endpoint inconsistency, and global topology
   destruction.
3. **R1/M1 do not share a readout.** R1 uses a direct edge readout; M1 pools
   updated nodes. R1→M1 is not a pure node-update contrast.

Phase 1c asks three minimal questions:

- **Q1 (readout).** Does an additively count-aware, size-aware graph readout
  close part of the ZINC gap without changing the local patch encoder?
- **Q2 (correspondence).** At identical parameters, modules, readout and
  optimization, does true endpoint correspondence beat the same relation
  branch fed only the molecule-level mean token and the true edge-type
  multiset?
- **Q3 (capacity).** At matched parameters and the same readout rule, is the
  true-edge relation model better than a no-root-adjacency model?

This round does **not** implement R1-G, global tokens, deeper trunks, new
structural encodings, SSL, or a search over widths/depths/LRs.

---

## 2. Data and frozen artifacts

Only the official ZINC-12k **train** split is used, through the frozen Phase 1b
internal split. Official valid/test and internal Reserved are never
instantiated or scored.

| Item | Value |
|---|---|
| Source | PyG `ZINC(root=data/ZINC, subset=True, split="train")`, 10,000 rows |
| Target | `logP_SA_cycle_normalized` (benchmark-normalized; no extra standardization) |
| Internal split | Fit 8,000 / Monitor 1,000 / Reserved 1,000; `default_rng(20260922).permutation(10000)` |
| Split file | `results/phase1b/split_zinc_phase1b.json` sha256 `fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49` |
| Row hashes | Fit `8351cbee…`, Monitor `2e4801c6…`, Reserved `cf1e8c6c…` |
| Labels | `results/phase1b/zinc_labels_fit_monitor.json` sha256 `631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78` (9,000 Fit+Monitor labels; Reserved absent) |
| Structure cache | `results/phase1b/cache/zinc_phase1b.pt` sha256 `76c17492…` |
| Size sidecar | `results/phase1c/zinc_size_features.pt` sha256 `8a20858e…`; structure-only; no labels |
| Label vault audit | Fit 8,000, Monitor 1,000, Reserved 1,000; `reserved_labels_retained=0`; `official_valid_test_labels_retained=0` |

**Size features (explicit, structure-only, no penalized-logP components):**

\[
s(G)=\left[\frac{n_{\text{atoms}}}{10},\ \frac{n_{\text{bonds}}}{10},\
n_{\text{bonds}}-n_{\text{atoms}}+n_{\text{components}}\right].
\]

All 10,000 ZINC molecules are connected (CPU gate), so the third coordinate is
the cyclomatic ring number. `n_bonds` and `rings` are count information, not
adjacency; `rings` is redundant given `n_atoms`, `n_bonds` and connectivity.
The sidecar records formula, connectivity, hashes and counts.

---

## 3. Models (frozen architecture)

All three arms use the **bitwise-identical Phase 1b shared trunk**:

- 2-layer GINE inside each rooted radius-2 patch, hidden 64, edge dim 4;
- patch-node mean → one token per root atom;
- 2 residual per-token FFN blocks (hidden 64, FFN 128, SiLU, dropout 0.1).

The Phase 1b patch encoder, patch cache, atom/bond features and distance
encoding are unchanged.

### 3.1 `S_cap_size` (no root adjacency)

- Node pools: `mean` and `sum` over patch tokens.
- Size input: `s(G)` above.
- `SizeAwareNodeReadout`: `LayerNorm(2*64+3=131)` →
  `Linear(131,64)` → SiLU → Dropout 0.1 → `Linear(64,64)`.
- Two real residual graph FFNs: `LayerNorm(64)` →
  `Linear(64,177)` → SiLU → Dropout → `Linear(177,64)` → residual.
- Head: `LayerNorm(64)` → `Linear(64,64)` → SiLU → Dropout → `Linear(64,1)`.

No root edges are present in its batch. It uses only patch tokens plus the
three declared size counts. The two graph FFNs are the capacity-matching
budget, not frozen or dead parameters; their gradients are checked.

### 3.2 `R1_size` (true endpoint relation)

- Same trunk, same `SizeAwareNodeReadout`, same head.
- `RelationMLP`: input `[h_u+h_v, |h_u-h_v|, h_u*h_v, e_uv]` (3·64+4=196) →
  128 → SiLU → Dropout 0.1 → 64. Exactly the Phase 1b `RelationMLP`.
- Edge aggregate over the **true root edges**, each physical undirected bond
  counted once (`src < dst`):
  `edge_mean`, `edge_sum` of the per-edge relation vectors.
- `EdgeReadout`: `LayerNorm(128)` → `Linear(128,64)` → SiLU → Dropout →
  `Linear(64,64)`.
- Graph vector: `SizeAwareNodeReadout(mean, sum, s(G)) + EdgeReadout(edge_mean,
  edge_sum)`, then the head.
- No node-state update.

### 3.3 `R1_marginal_size` (endpoint-correspondence ablation)

- **Exactly the same modules, parameter shapes, initialization order, readout,
  and optimization as `R1_size`.** The same `RelationMLP` object and the same
  `EdgeReadout`.
- It receives the E2 patch tokens and the true per-edge bond attributes, but
  **no endpoint indices and no root-node ids at all**. Its batch keys are
  `MARGINAL_SIZE_KEYS` (patch keys + `root_edge_attr`, `root_edge_mol`,
  `root_edge_unique`, `mol_size_feat`).
- For every directed edge, both endpoint slots are the molecule's global mean
  node token: `h_global = node_mean[edge_mol]`; the relation input is
  `[2h_global, 0, h_global*h_global, e_uv]`.
- Edge aggregate is still mean/sum over one copy of each physical bond, so it
  retains edge count and the exact bond-type multiset.
- This removes **which two root tokens are attached to the edge**. It is an
  endpoint-correspondence ablation, not a full graph-information match and
  **not a chemical causal experiment**. It does not preserve global topology;
  it simply does not use endpoint identity/pairing.

### 3.4 Optional `M_keep_edge` (only under the follow-up gate)

- Keeps the R1 direct edge readout (mean/sum of `r` from the pre-update trunk
  tokens) and the size-aware node readout.
- Adds one tied node-update round:
  `LayerNorm(h + Dropout(MLP(mean incident relation)))`.
- It is implemented and CPU-tested but **not** in the seed-0 first round.
- Matched-compute comparison: `R1_size_long` with **322 epochs** (CPU forward
  FLOPs ratio 694,913,664 / 646,843,008 = 1.0743) vs `M_keep_edge` at 300
  epochs. Actual GPU seconds and relation applications are reported; if the
  wall-clock ratio differs materially, the comparison is reported as
  approximate, not silently re-tuned.

### 3.5 Capacity and cost (CPU-measured, frozen before GPU)

| Arm | Params | Forward FLOPs, batch 64 | Relative to S0 |
|---|---:|---:|---:|
| S0 (Phase 1b anchor) | 56,963 | 434,141,824 | 1.000 |
| R1 (Phase 1b anchor) | 94,659 | 644,196,992 | 1.484 |
| **S_cap_size** | **115,883** | **441,539,200** | **1.017** |
| **R1_size** | **115,977** | **646,843,008** | **1.490** |
| **R1_marginal_size** | **115,977** | **646,843,008** | **1.490** |
| G_ref (Phase 1b anchor) | 439,559 | 1,185,134,336 | 2.730 |
| M_keep_edge (optional) | 132,681 | 694,913,664 | 1.601 |

- `S_cap_size` vs `R1_size`: **94 parameters** difference (0.08%). The match is
  reported, not claimed perfect. The unavoidable compute difference is real:
  `R1_size` is 1.465× `S_cap_size` forward FLOPs because it runs the relation
  MLP per edge; `S_cap_size` runs two pooled graph FFNs.
- `R1_size` and `R1_marginal_size` are parameter-identical (115,977) and share
  every module/parameter shape. The only difference is endpoint access.

---

## 4. Training recipe (identical to Phase 1b, frozen)

| Item | Value |
|---|---|
| Loss | L1 on the raw target |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-5 |
| Schedule | cosine, `T_max=300`, `eta_min=1e-5` |
| Batch | 64 |
| Epochs | 300 |
| Early stop | Monitor MAE, patience 30, `min_delta=1e-4` |
| Grad clip | 5.0 |
| Dropout | 0.1 |
| Seed | 0 first; second seed only under §7 |
| Data order | same `default_rng([seed, epoch])` for all arms |
| Model selection | best Monitor MAE; Monitor also used for early stopping |
| Soup / ensemble / SSL / external data | none |

No post-result changes to width, dropout, LR, readout or batch size.

---

## 5. Run matrix and budget

| Phase | Runs | Cap (GPU-h) | Trigger |
|---|---|---:|---|
| P0 | CPU gates + CPU smoke + one short GPU smoke | 0.05 | always |
| A | `S_cap_size`, `R1_size`, `R1_marginal_size`, seed 0 | 0.60 | after frozen commit |
| B | `R1_size`, `R1_marginal_size`, seed 1 | 0.40 | only if §7 seed gate passes |
| C | `M_keep_edge` seed 0 + `R1_size_long` (322 ep) seed 0 | 0.50 | only if §7 follow-up gate passes |
| Reserve | failure/smoke | 0.45 | — |
| **Total** | | **2.00** | hard cap |

Unused budget is not a failure. Do not start a phase that cannot finish within
the remaining 2.0 h. All GPU spend is recorded in
`results/phase1c/run_manifest.tsv` (run-level GPU seconds and job occupancy).

---

## 6. Endpoints (pre-registered)

Using Monitor MAE (lower is better); positive means the right-hand arm is
better:

\[
\begin{aligned}
\mathrm{Gain}_{\rm pair} &= \mathrm{MAE}(R1\_marginal\_size) - \mathrm{MAE}(R1\_size)\\
\mathrm{Gain}_{\rm over\_capacity} &= \mathrm{MAE}(S\_cap\_size) - \mathrm{MAE}(R1\_size)\\
\mathrm{Gain}_{\rm readout\_vs\_R1} &= \mathrm{MAE}(R1) - \mathrm{MAE}(R1\_size)\\
\mathrm{Gain}_{\rm readout\_vs\_S0} &= \mathrm{MAE}(S0) - \mathrm{MAE}(S\_cap\_size)\\
\mathrm{Gap}(h) &= \mathrm{MAE}(h) - \mathrm{MAE}(G\_ref)
\end{aligned}
\]

`R1`, `S0`, `G_ref` are the frozen Phase 1b seed-0 anchors; they are not
retrained. `Gain_pair` is the primary correspondence endpoint.
`Gain_over_capacity` is the primary capacity/readout endpoint.

Uncertainty: paired bootstrap over Monitor molecules, 5,000 resamples, 95%
percentile CI, per-comparison (not family-wise). Single-seed and bootstrap
results are screening evidence, not formal significance claims.

**Bands (same as Phase 1b ZINC):**

| | negligible | weak / unresolved | meaningful |
|---|---:|---:|---:|
| ZINC Monitor MAE | <0.01 | 0.01–0.02 | ≥0.02 |

A direction is "clear" only if the paired CI excludes zero and the RMSE sign
agrees. A result in the weak band is reported as `INSUFFICIENT_EVIDENCE`, not
rounded up.

---

## 7. Decision logic (pre-registered)

Read the three seed-0 runs first. Do not automatically buy B or C.

### 7.1 Seed-0 labels

- **`CORRESPONDENCE_SURVIVES_CAPACITY`** — `Gain_pair ≥ 0.02` with CI95 lower
  > 0 **and** `Gain_over_capacity ≥ 0.02` with CI95 lower > 0. Endpoint
  correspondence has value beyond both the edge-type marginal and the
  matched-capacity no-adjacency model.
- **`CORRESPONDENCE_VS_MARGINAL_ONLY`** — `Gain_pair ≥ 0.02` with CI>0, but
  `Gain_over_capacity < 0.02` (S_cap_size is weak/negligible relative to
  R1_size). Correspondence beats the marginal control, but the relational
  version has no clear application advantage over the capacity/readout model.
- **`CAPACITY_READOUT_EXPLAINS`** — `Gain_over_capacity < 0.01` (S_cap_size
  matches or beats R1_size) **and** `Gain_pair < 0.02`. The matched-capacity
  readout model removes the earlier R1 advantage; lower the relational
  sufficiency claim.
- **`READOUT_BOTTLENECK_DOMINATES`** — both `Gain_readout_vs_R1 ≥ 0.02` and
  `Gain_over_capacity` is in the negligible band `|·| < 0.02`. The common
  sum/mean/size readout, not the relation branch, is the main axis.
- **`NO_ENDPOINT_CORRESPONDENCE_GAIN`** — `Gain_pair < 0.01` (the marginal
  model matches or beats R1_size). Stop the correspondence claim for this
  harness.
- **`ABSOLUTE_GAP_PERSISTS`** — the best of the three remains at least 0.02
  MAE worse than frozen `G_ref`. Report absolute performance insufficiency;
  do not call the method competitive.
- **`INSUFFICIENT_EVIDENCE`** — any primary endpoint in the weak band
  (0.01–0.02) or CI crossing zero, or MAE/RMSE signs disagree.

These labels are not mutually exclusive with `ABSOLUTE_GAP_PERSISTS`; the
report states both the mechanism label and the absolute-gap label.

### 7.2 Seed-1 gate (Phase B)

Buy **only** `R1_size` and `R1_marginal_size` at seed 1 if **all** hold:

1. seed-0 `Gain_pair ≥ 0.02` (meaningful);
2. seed-0 `Gain_pair` CI95 lower > 0;
3. `Gain_over_capacity ≥ −0.01` (R1_size is not meaningfully worse than the
   matched-capacity S_cap_size);
4. the seed-0 pair gap exceeds the Phase 1b R1 run-to-run spread (0.0031 MAE).

If the pair gain is not meaningful, or S_cap_size clearly dominates R1_size,
**do not buy seed 1**; end the one-shot correspondence claim for this round.

If seed 1 is bought: report both seeds, the paired sign, the mean, and the
per-seed run-to-run spread. Do not pool with Phase 1b R1/S0.

### 7.3 Follow-up `M_keep_edge` gate (Phase C)

Buy `M_keep_edge` + `R1_size_long` **only if all** hold:

1. seed-0 and seed-1 `Gain_pair` have the same positive sign and the two-seed
   mean is ≥0.02;
2. `R1_size`'s gap to `G_ref` is still ≥0.02 (large);
3. the remaining budget can finish both runs.

Then test:

\[
\mathrm{Gain}_{\rm update\_keep\_edge} =
\mathrm{MAE}(R1\_size\_long) - \mathrm{MAE}(M\_keep\_edge)
\]

with the same 0.02 meaningful band. Positive with CI>0 means one update helps
under the **same direct edge readout** and matched compute. This is the only
new mechanism diagnosis allowed in this round. It does not repeat the Phase 1b
R1→M1 readout change.

If the pair gain disappears at seed 1, or the absolute gap is small, **do not
buy `M_keep_edge`**. Report the stop.

---

## 8. What this round will not do

- No new global attention token, no R1-G, no GPS/GRIT/ESA+ implementation.
- No deeper residual trunk, no new structural encoding family, no distance
  channel, no SSL, no ensemble, no external data.
- No official ZINC valid/test or internal Reserved scoring.
- No hyperparameter/width/depth/LR/dropout search after seeing results.
- No claim that adding FFN layers increases structural receptive field.
- No claim that `G_ref` is a global-attention model.
- No claim that `R1_marginal_size` is a chemical causal intervention.
- No claim of SOTA or leaderboard comparability from internal Monitor values.

---

## 9. CPU gates and tests (completed before freeze)

`phase1c_gates.py`: **17/17 PASS** (0.5 s), saved to
`results/phase1c/cpu_gates_phase1c.json`.

Key checks:

- frozen split/label/cache hashes exactly match Phase 1b;
- label vault retains 9,000 Fit+Monitor labels, Reserved 0, official valid/test 0;
- size sidecar matches cache counts and the cyclomatic formula; all 10,000 ZINC
  molecules connected;
- parameter counts: S_cap_size 115,883; R1_size = R1_marginal_size 115,977;
  relative difference 0.08%;
- forward FLOPs measured for all arms;
- shared trunk initialization identical across the three arms;
- `R1_size` and `R1_marginal_size` have bitwise-identical initialization and
  parameter shapes at the same seed;
- permutation invariance and batch-composition invariance;
- S_cap_size guard rejects root edges; marginal guard rejects endpoint keys;
- physical-bond-type permutation changes R1_size (Δ≈1.5e-4) but not
  R1_marginal_size (Δ≈4e-8);
- S0 cannot separate G from a disjoint duplicate (Δ≈4e-8), while all three
  size-aware arms can (Δ≈1e-2);
- forward/gradient liveness for graph FFNs, relation MLP and edge readout;
- Phase 1b frozen anchors unchanged.

`tests/test_phase1c.py`: **11/11 passed** (1.7 s).
`tests/test_phase1b.py`: **7/7 passed** (3.2 s) after the extension hooks.

CPU smoke (first 128 Fit / 64 Monitor, 2 epochs) completed for S_cap_size,
R1_size and R1_marginal_size with finite Monitor MAE and the correct label
vault audit. Smoke numbers are not results.

---

## 10. Artifacts and reproduce

```bash
# CPU: size sidecar
uv run python tracks/structural-sufficiency/code/phase1c/run_phase1c.py prep --dataset zinc

# CPU: gates and tests
uv run python tracks/structural-sufficiency/code/phase1c/phase1c_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1c/tests/test_phase1c.py
uv run pytest -q tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py

# CPU smoke (one arm)
uv run python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train \
  --dataset zinc --model S_cap_size --seed 0 --device cpu --smoke --epochs 2 --patience 1 \
  --out-dir tracks/structural-sufficiency/runs/phase1c_smoke

# GPU (through rr; one job per arm)
rr deploy res
rr run res ssrc-p1c-zinc-Scap-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py \
     train --dataset zinc --model S_cap_size --seed 0 --device cuda
rr run res ssrc-p1c-zinc-R1size-s0 --pool res-gpu1 \
  --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py \
     train --dataset zinc --model R1_size --seed 0 --device cuda
rr run res ssrc-p1c-zinc-R1marg-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py \
     train --dataset zinc --model R1_marginal_size --seed 0 --device cuda
```

The exact remote run IDs, commits, GPU seconds, job occupancy, best epochs and
output hashes are recorded in `results/phase1c/run_manifest.tsv`.

---

## 11. Non-claims (repeat)

1. `S_cap_size` is not "the same information" as R1: it has no adjacency, but
   it does receive graph counts and extra pooled-graph FFN capacity.
2. `R1_marginal_size` is an endpoint-correspondence ablation, not a chemical
   causal experiment and not a full graph-information match.
3. `R1_size` vs `S_cap_size` is matched in parameters and node/size readout,
   not in forward FLOPs or per-edge computation.
4. `G_ref` is a 6-layer local GINE with sum+mean readout, not global attention.
5. Internal train-only Monitor numbers are not official ZINC test numbers.
6. A single seed and a Monitor bootstrap do not establish a formal
   significance claim.
7. No result here proves \(E_2\) is task-information sufficient/insufficient
   in the Bayes-risk sense.
