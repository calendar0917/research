# Phase 1d — Incidence control and bounded capacity scaling (ZINC-12k)

| Field | Value |
|---|---|
| Protocol | `ssrc-phase1d-zinc-v1` |
| Freeze / run commit | `6e26ca9` (`ssrc: freeze phase1d incidence control + bounded capacity scaling`) |
| Status | COMPLETE — 3/3 GPU runs exit 0 (one smoke, two full); 19/19 CPU gates; 25/25 tests across Phase 1b–1d. |
| Primary B decision | `PAIRWISE_INTERACTION_PROMISING` — **qualified: large-ring-tail-specific**. |
| Primary C decision | `CAPACITY_SCALING_PROMISING` + `REPRESENTATION_OR_INDUCTIVE_BIAS_GAP` — **qualified: the remaining gap is concentrated in the same large-ring tail**. |
| Efficiency note | `EFFICIENCY_OBSTACLE` (2.10× G_ref forward FLOPs). |
| GPU spend | 0.464 h run-level / 0.471 h job occupancy (cap 2.00 h; no failed jobs). |

---

## 0. One-paragraph summary

Phase 1d asked whether the Phase 1c endpoint-pairing gain is really a non-additive pairwise interaction, and whether more width closes the remaining gap to the 6-layer local GINE reference `G_ref`. The additive node–incidence control `R_incidence` (115,952 params; 645.9M FLOPs) is **worse than `R1_size` overall by +0.0431 MAE** (0.2797 vs 0.2366; 95% CI +0.0121…+0.0763), with RMSE also favoring `R1_size` (0.8012 vs 0.6069). That would ordinarily support the pairwise-interaction claim. The subgroup readout changes the interpretation: on the **96% complement without a ring larger than six atoms**, `R_incidence` is slightly **better** than `R1_size` (−0.0281 MAE, CI −0.0460…−0.0117); on the **4% large-ring subgroup (n = 40)**, `R1_size` is much better (+1.751 MAE, CI +1.303…+2.201). The top 1% of molecules carries 81% of the overall gain; dropping that 1% leaves +0.0082, inside the negligible band. The same tail dominates the capacity result: `R1_size_cap` (453,129 params) improves the unscaled candidate by **+0.0431 MAE** (CI +0.0192…+0.0688) and converges, but its overall gap to `G_ref` remains **+0.0282** (CI +0.0071…+0.0509). On the 96% complement that gap is only **+0.0174**, while on the 4% tail it is **+0.286**. The defensible conclusion is narrower than “pairwise interactions are generally necessary” and narrower than “capacity is the bottleneck”: **additive incidence statistics are sufficient on ordinary ZINC molecules; non-additive endpoint pairing and extra width both buy a tail-specific rescue on large-ring molecules, and neither closes that tail gap to G_ref.**

---

## 1. Phase A — CPU descriptive audit (completed before the GPU freeze)

Phase A used only frozen Phase 1b/1c Monitor predictions and the frozen Fit/Monitor labels. It did not read Reserved or official valid/test labels, and it did not select a model. Artifacts: `results/phase1d/analysis_phase1d_cpu.json`, `structural_features.npz`, `structural_features_manifest.json`.

### 1.1 Overall frozen errors

| Model | Monitor MAE | Monitor RMSE | Note |
|---|---:|---:|---|
| `G_ref` seed 0 | **0.1653** | 0.4482 | 6-layer local GINE reference |
| `R1_size` seed 0 | **0.2366** | 0.6069 | best non-iterative relation |
| `R1_size` seed 1 | 0.2810 | 0.7689 | same model, different seed |
| `R1_marginal_size` seed 0 | 0.3047 | — | endpoint-correspondence ablation |
| `S_cap_size` seed 0 | 0.2895 | — | parameter-matched no-edge control |

The three Phase 1c contrasts are positive at seed 0: `d_pair = +0.0682`, `d_capacity = +0.0529`, `d_gap = +0.0713` (positive means `R1_size` is better than the respective control).

### 1.2 The gains are concentrated in a small, label-coupled tail

The structure-only binary split (`large_cycles_gt6 >= 1`, equivalently `max_ring_size >= 7`) contains **40/1,000 molecules (4%)**. It was fixed before Phase 1d GPU work and is used as a secondary readout, not as a model-selection variable.

| Group | n | `R1_size` MAE | `R1_marginal` MAE | `G_ref` MAE | `d_pair` mean [CI] | `d_capacity` mean [CI] |
|---|---:|---:|---:|---:|---:|---:|
| No ring >6 | 960 | 0.1741 | 0.1634 | 0.1299 | **−0.0107** [−0.0304, +0.0067] | **−0.0280** [−0.0468, −0.0121] |
| Ring >6 | 40 | 1.7364 | 3.6971 | 1.0145 | **+1.9607** [+1.5035, +2.3942] | **+1.9942** [+1.5467, +2.4669] |

On the 96% complement, `R1_size` is **not** better than the marginal or capacity controls; the additive/endpoint-marginal models are slightly better. The entire positive mean `d_pair` and `d_capacity` is carried by the 4% tail. The `G_ref` gap is broader but still tail-heavy: complement gap **+0.0442** [+0.0273, +0.0642]; the tail contributes **0.0289** of the overall 0.0713 gap.

Top-tail sensitivity of `d_pair` seed 0: top 1% contributes 0.0290 (42.6% of the mean), top 5% contributes 0.0570 (83.6%), and dropping the top 5% leaves **+0.0118** (negligible band). For `d_capacity`: top 5% contributes 0.0564 (106.7% of the mean), and dropping the top 5% gives **−0.0037**. For `d_gap`: top 5% contributes 0.0543 (76.1%), and dropping it leaves **+0.0179**.

### 1.3 Seed consistency and structural correlates are weak

Per-molecule `d_pair` seed 0 vs seed 1: Spearman **0.337**, sign agreement **0.606**, top-decile Jaccard **0.198**. The same-model `R1_size` seed 0 vs seed 1 difference is itself large (`−0.0445` mean, CI `[−0.0753, −0.0155]`), so the seed-0 tail advantage is not a stable per-molecule ranking.

Fixed structure-only correlates (size, rings, fusion/shared atoms, large cycles, Kekulé six-rings, diameter, bond composition, patch size/repetition, root–patch TV distance, shell sizes) are generally weak: the largest absolute Spearman correlation with any of `d_pair`/`d_capacity`/`d_gap` is about **0.09**. No continuous variable explains ≥0.02 MAE. The old “stop SSRC if a structural variable explains <0.02” rule remains void, and no model was selected from these correlations. The only subgroup used downstream is the pre-specified binary large-ring split.

---

## 2. Phase B — additive node–incidence control

### 2.1 Design

`R_incidence` replaces the joint endpoint relation with a shared additive term on each true physical bond:

\[
r_{uv} = \psi(h_u,e_{uv}) + \psi(h_v,e_{uv}),
\]

with the same E2 patch trunk, root adjacency, node features, size-aware readout, edge mean/sum aggregate, loss, schedule and data as `R1_size`. Each undirected bond contributes one edge readout; direction is handled by the symmetric sum. CPU gates verify node-permutation invariance, edge-direction invariance, local-multiset-preserving rewiring invariance, shared-module use, no joint endpoint input, and parameter/FLOP matching.

| | `R1_size` | `R_incidence` | Difference (incidence − R1) |
|---|---:|---:|---:|
| Parameters | 115,977 | 115,952 | −25 (−0.02%) |
| Forward FLOPs (batch 64) | 646,843,008 | 645,911,200 | −0.14% |
| Median step time (ms) | 20.53 | 19.96 | −2.8% |
| GPU seconds (seed 0) | 694.2 | 614.0 | −11.6% |
| Best epoch / epochs run | 225 / 256 | 165 / 196 | both early-stopped |

The parameter and compute match is close enough for a mechanism contrast, not for a pure causal claim.

### 2.2 Overall result

| Endpoint | `R1_size` | `R_incidence` | `Gain_joint` = MAE(`R_incidence`) − MAE(`R1_size`) |
|---|---:|---:|---:|
| Monitor MAE | **0.2366** | 0.2797 | **+0.0431** [95% CI +0.0121, +0.0763] |
| Monitor RMSE | **0.6069** | 0.8012 | +0.1943 |
| Median AE | 0.1002 | 0.0989 | incidence slightly better |
| p90 AE | 0.3806 | 0.3628 | incidence better |
| Mean signed error | +0.1022 | +0.1612 | both positive |
| Per-molecule wins | 47.8% | 52.2% | incidence slightly better |

By the frozen rule, `Gain_joint ≥ 0.02`, CI lower > 0, and the RMSE sign agrees, so Phase B is **`PAIRWISE_INTERACTION_PROMISING`**. But the gain is not distributed across chemistry.

### 2.3 The gain is a large-ring tail effect

| Group | n | `R1_size` MAE | `R_incidence` MAE | Difference (`R1` − `incidence`; positive = `R1` better) |
|---|---:|---:|---:|---:|
| No ring >6 | 960 | 0.1741 | **0.1460** | **−0.0281** [−0.0460, −0.0117] |
| Ring >6 | 40 | **1.7364** | 3.4876 | **+1.7512** [+1.3033, +2.2013] |

On the 96% complement, the additive incidence model is **better** and cheaper. On the 4% tail, the joint pairing is much better; the incidence model essentially collapses (3.49 MAE vs 1.74). Top-tail sensitivity confirms this: the top 1% contributes 0.0350 (81.2% of the overall gain), dropping the top 1% leaves **+0.0082** (negligible), and dropping the top 5% leaves **−0.0036**.

Training diagnostics point in the same direction: `R_incidence` has train MAE 0.2399 vs `R1_size` 0.1510 and a larger train–Monitor gap, so this is not a case where the additive model is simply under-optimized on the complement. It fits ordinary molecules well enough to beat `R1_size` there; it lacks the joint endpoint term needed for the large-ring tail.

**Mechanism interpretation.** Non-additive endpoint pairing is supported **as a tail-specific mechanism**, not as a general requirement on ZINC. The Phase 1c claim should be lowered accordingly: on ordinary molecules, additive node–incidence statistics plus the patch trunk and nonlinear readout already explain the gain; the residual joint-pairing advantage is concentrated where the target’s `-cycles` term and large-ring structure make the local endpoint relation harder to summarize additively.

---

## 3. Phase C — one bounded capacity-scaling test

### 3.1 Selection and configuration

The frozen rule selects `R1_size` after `PAIRWISE_INTERACTION_PROMISING`. Only hidden width was scaled:

| | `R1_size` | `R1_size_cap` | `G_ref` |
|---|---:|---:|---:|
| hidden | 64 | 128 | — |
| local layers / token-FFN blocks | 2 / 2 | 2 / 2 | 6 GINE layers |
| parameters | 115,977 | **453,129** | 439,559 |
| params vs G_ref | 0.264× | **1.031×** | 1.000× |
| forward FLOPs (batch 64) | 646.8M | **2,491.9M** | 1,185.1M |
| FLOPs vs G_ref | 0.546× | **2.103×** | 1.000× |
| median step time (ms) | 20.53 | 22.22 | 29.99 |
| peak allocated GPU memory (MB) | 72.7 | 130.7 | 81.6 |
| GPU seconds | 694.2 | 1,056.5 | 1,051.5 |

The capacity run early-stopped at epoch **258** after **289/300** epochs, with train MAE 0.1054 and Monitor MAE 0.1935. The convergence gate passes.

### 3.2 Capacity helps, but does not close the overall gap

| Endpoint | `R1_size` | `R1_size_cap` | `G_ref` |
|---|---:|---:|---:|
| Monitor MAE | 0.2366 | **0.1935** | **0.1653** |
| Monitor RMSE | 0.6069 | **0.5215** | **0.4482** |
| p90 AE | 0.3806 | 0.3223 | 0.2900 |

`Gain_cap = MAE(R1_size) − MAE(R1_size_cap) = **+0.0431**` [CI +0.0192, +0.0688], RMSE also favors capacity, and the run converged. This is **`CAPACITY_SCALING_PROMISING`**.

The gap to `G_ref` is still `+0.0282` [CI +0.0071, +0.0509], so the protocol also requires **`REPRESENTATION_OR_INDUCTIVE_BIAS_GAP`**. The subgroup decomposition shows where that gap lives:

| Group | n | `R1_size` | `R1_size_cap` | `G_ref` | Capacity gain [CI] | Gap cap−G_ref |
|---|---:|---:|---:|---:|---:|---:|
| No ring >6 | 960 | 0.1741 | **0.1473** | **0.1299** | **+0.0267** [+0.0117, +0.0443] | **+0.0174** |
| Ring >6 | 40 | 1.7364 | **1.3008** | **1.0145** | **+0.4356** [+0.0090, +0.8850] | **+0.2862** |

On the 96% complement, capacity closes most of the gap (0.0174, just below the 0.02 meaningful band). On the 4% tail, capacity improves the model but leaves a 0.286 gap; that tail contributes 0.0114 of the 0.0282 overall gap. The tail capacity CI is wide because n = 40 and a few molecules dominate.

### 3.3 Efficiency obstacle

Parameter count is matched (+3.1% vs `G_ref`), and median step time is actually lower than `G_ref` (22.2 vs 30.0 ms), but forward FLOPs are **2.10×** `G_ref` and peak allocated memory is **1.60×**. The protocol’s efficiency criterion is FLOPs or step time >~2×; the FLOPs criterion is met, so the result carries an explicit **`EFFICIENCY_OBSTACLE`**. The lower step time is an architecture/runtime effect, not evidence of a compute advantage; do not describe `R1_size_cap` as competitive at matched compute.

**Interpretation.** Capacity is a real lever on ordinary molecules: with 3.1% more parameters than `G_ref`, the capacity-scaled relation is within 0.0174 MAE on the 96% complement. It is not the main bottleneck for the large-ring tail, where a 3.9× parameter increase still leaves a 0.286 gap. That pattern is consistent with a representation/inductive-bias limitation (or with the target’s ring/cycle label mechanism), not with “just add width.”

---

## 4. What is established, and what is not

### Descriptive facts

- On the frozen ZINC Monitor split, `R1_size` beats `R_incidence` overall by 0.0431 MAE, but `R_incidence` beats `R1_size` on the 96% of molecules with no ring larger than six atoms by 0.0281 MAE.
- The overall pairwise gain is tail-driven: 40 molecules (4%) carry more than the full positive mean; top 1% contributes 81%; dropping top 1% leaves +0.0082.
- `R1_size_cap` improves the unscaled model by 0.0431 MAE and converges; its overall gap to `G_ref` is 0.0282, of which the large-ring tail contributes 0.0114.
- On the 96% complement, the capacity model is within 0.0174 MAE of `G_ref`; on the 4% tail it is 0.286 MAE worse.
- Structural correlations with per-molecule gains are weak (max |Spearman| ≈ 0.09), and per-molecule seed consistency is weak (Spearman 0.337, sign agreement 0.606).

### Mechanism contrast

- The additive incidence control is a strong, slightly cheaper model on ordinary molecules. This **lowers** the broad Phase 1c claim that a non-additive pairwise interaction is generally necessary for the observed gain.
- The joint endpoint pairing is supported only as a **large-ring rescue**: it prevents the incidence model’s 3.49 MAE tail failure and brings it to 1.74, still above `G_ref`’s 1.01.
- Capacity scaling and joint pairing both help the tail, but neither closes it. The remaining tail gap is therefore not identified as a width problem or as a pure pairwise-interaction problem.

### Absolute performance

- Best non-iterative Phase 1d model: `R1_size_cap` at **0.1935** Monitor MAE / 0.5215 RMSE. `G_ref` remains better at **0.1653** / 0.4482.
- `R1_size_cap` uses 453,129 params (1.031× `G_ref`) but 2.10× forward FLOPs and 1.60× peak memory. It is not a compute-matched win.
- These are internal train-only Monitor numbers, not official ZINC test numbers; no SOTA claim.

### Novelty

- `R_incidence` is a diagnostic control, not a proposed method. The contribution is the **narrowing**: additive incidence statistics are sufficient on ordinary molecules, while the non-additive pairing and capacity gains are tail-specific.
- No new architecture, global token, attention mechanism, radius/depth grid, or iterative update was introduced.
- The result is a mechanism/qualification result, not a new algorithm or a new state of the art.

### Evidence limits

- One seed per new arm. Molecule-bootstrap CIs are not training-seed CIs; the `R1_size` seed-0 vs seed-1 difference is itself 0.0445 MAE.
- The 4% tail is n = 40; its capacity CI is wide and a few molecules dominate. `G_ref` also has 1.01 MAE there, and the target includes a `-cycles` term, so this subgroup is partly label-mechanism-coupled.
- The large-ring split was pre-specified for B/C readout but was discovered descriptively in Phase A; the mechanism interpretation remains post-hoc.
- `G_ref` is a 6-layer local GINE, not global attention and not a paper reproduction. “Capacity” here means width only, at one fixed configuration.
- The parameter match is close, but FLOPs and memory are not matched; the efficiency comparison is architecture-dependent.
- No official valid/test or internal Reserved label was read, retained, or scored. `run_validation.json` is PASS for both full runs; the smoke run is separate and not used for decisions.

---

## 5. Decision and one next step

**Final Phase 1d labels**

- Phase B: **`PAIRWISE_INTERACTION_PROMISING`** — qualified as **large-ring-tail-specific**; not a general chemistry claim.
- Phase C: **`CAPACITY_SCALING_PROMISING`** + **`REPRESENTATION_OR_INDUCTIVE_BIAS_GAP`** — the remaining gap is tail-concentrated; ordinary molecules are within 0.0174 MAE of `G_ref`.
- Efficiency: **`EFFICIENCY_OBSTACLE`** (2.10× forward FLOPs vs `G_ref`).
- Track status: **`PHASE1D_COMPLETE`**.

**One next-step decision:** do **not** run another width, seed, depth, or global-token arm. The next action is a **CPU-only large-ring predictability audit** on the frozen predictions and structure-only features: quantify whether the 40-molecule tail has any patch-observable headroom, whether its errors are driven by cycle-size/ring-fusion features, and whether a ring-aware local token could plausibly help without degrading the 96% complement. Only if that audit shows a pre-specified ≥0.02 MAE opportunity should one ring-aware diagnostic be pre-registered for a later round; it must be a single local representation change, not a width/depth/radius grid, and it must stop if the complement degrades or the tail gain is <0.02 / CI crosses zero. No new GPU work is justified by Phase 1d alone.

This is deliberately conservative: the broad “non-additive pairwise interaction is necessary” claim is **not** supported; the broad “capacity is the bottleneck” claim is **not** supported; the surviving claim is the narrower tail-specific one.

---

## 6. Artifacts, validation, and reproduce

**Data isolation.** Fit 8,000 / Monitor 1,000 / Reserved 1,000. The two full runs retain 9,000 Fit+Monitor labels and **0** Reserved / official valid/test labels. `results/phase1d/run_validation.json`: **PASS**, 2/2 full runs, identical Monitor molecule order.

**Artifacts.**

| Artifact | Path |
|---|---|
| Protocol | `notes/phase1d_protocol.md` |
| Config | `configs/phase1d.yaml` |
| Phase A features | `results/phase1d/structural_features.npz` (+ manifest) |
| Phase A audit | `results/phase1d/analysis_phase1d_cpu.json` |
| CPU gates | `results/phase1d/cpu_gates_phase1d.json` (19/19 PASS) |
| GPU analysis | `results/phase1d/analysis_phase1d.json` |
| Run validation | `results/phase1d/run_validation.json` (PASS) |
| Run manifest | `results/phase1d/run_manifest.tsv` |
| Code | `code/phase1d/` |
| Runs | `runs/phase1d/` (git-ignored raw Monitor predictions) |

**GPU runs.**

| Phase | Experiment | Model | GPU s | Job s | Best epoch / run | Monitor MAE |
|---|---|---|---:|---:|---:|---:|
| smoke | `ssrc-p1d-smoke-zinc` | `R_incidence` | 0.6 | 7 | 1 / 2 | 1.1503 (smoke subset) |
| B | `ssrc-p1d-zinc-incidence-s0` | `R_incidence` | 614.0 | 622 | 165 / 196 | 0.2797 |
| C | `ssrc-p1d-zinc-R1cap-s0` | `R1_size_cap` | 1,056.5 | 1,065 | 258 / 289 | 0.1935 |

Run-level total: **1,671.1 s = 0.4642 h**. Job occupancy: **1,694 s = 0.4706 h**. Cap: 2.00 h. No failed jobs.

**Reproduce (CPU).**

```bash
uv run python tracks/structural-sufficiency/code/phase1d/phase1d_features.py --rows all
uv run python tracks/structural-sufficiency/code/phase1d/analyze_phase1d_cpu.py
uv run python tracks/structural-sufficiency/code/phase1d/phase1d_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1d/tests/test_phase1d.py
uv run pytest -q tracks/structural-sufficiency/code/phase1c/tests/test_phase1c.py
uv run pytest -q tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py
uv run python tracks/structural-sufficiency/code/phase1d/validate_phase1d_runs.py
uv run python tracks/structural-sufficiency/code/phase1d/analyze_phase1d.py
uv run python tracks/structural-sufficiency/code/phase1d/make_run_manifest.py
```

**Reproduce (GPU, already run).**

```bash
rr deploy res
rr run res ssrc-p1d-zinc-incidence-s0 --pool res-gpu0 \
  --result tracks/structural-sufficiency/runs/phase1d \
  -- python tracks/structural-sufficiency/code/phase1d/run_phase1d.py \
     train --dataset zinc --model R_incidence --seed 0 --device cuda
rr run res ssrc-p1d-zinc-R1cap-s0 --pool res-gpu1 \
  --result tracks/structural-sufficiency/runs/phase1d \
  -- python tracks/structural-sufficiency/code/phase1d/run_phase1d.py \
     train --dataset zinc --model R1_size_cap --seed 0 --device cuda
```

---

## 7. Non-claims

1. `R_incidence` is not “the same information” as `R1_size`: it removes the joint endpoint pairing but keeps true root incidence, bond types, patch tokens and size features.
2. The incidence model’s edge aggregate invariance is a property of the additive form, not evidence that it is chemically meaningful.
3. A single seed and a molecule bootstrap do not establish formal significance or training-seed robustness.
4. The large-ring subgroup is partly a target-label mechanism (the target includes `-cycles`); a gain there is not a general chemical or relational claim.
5. `G_ref` is a 6-layer local GINE, not global attention and not a paper reproduction.
6. Internal train-only Monitor numbers are not official ZINC test numbers.
7. No result proves E2 task-information sufficiency or insufficiency in the Bayes-risk sense.
8. The capacity arm tests width only, at one parameter-matched configuration; it does not rule out depth, receptive field, or a different local representation as the tail bottleneck.
