# Phase 1c — Readout, capacity, and endpoint-correspondence controls (ZINC-12k)

**Protocol:** `ssrc-phase1c-zinc-v1`  
**Freeze / run commit:** `f57aec8` (`ssrc: freeze phase1c readout/capacity/correspondence controls + CPU gates`)  
**Status:** COMPLETE — 10/10 GPU runs exit 0; 17/17 CPU gates; 18/18 tests.  
**Primary decision:** `CORRESPONDENCE_SURVIVES_CAPACITY`  
**Secondary decision:** `ABSOLUTE_GAP_PERSISTS`  
**Route status:** `PURSUE_RELATIONAL_SUFFICIENCY` is **conditionally supported and narrowed**, not confirmed as sufficient. Endpoint correspondence is a real signal beyond readout/capacity/edge-type marginals, but it does not close the absolute gap to the local GINE reference.

---

## 0. One-paragraph summary

Phase 1b left three confounds: R1 differed from S0 in capacity and readout, the broken-adjacency control mixed several failures, and R1→M1 changed the graph readout. Phase 1c resolves them with three minimal, parameter-controlled arms on the frozen ZINC-12k Fit/Monitor split (8,000/1,000; Reserved never scored): `S_cap_size` (no root edges, size-aware sum+mean readout, real graph-FFN capacity), `R1_size` (true endpoint `phi(h_u,h_v,e_uv)` plus the same size-aware readout and edge-relation aggregate), and `R1_marginal_size` (identical modules/parameters/FLOPs/readout, same true bond-type multiset and edge count, but both endpoint tokens replaced by the molecule global mean). At seed 0, `R1_size` beats `R1_marginal_size` by **+0.0682 MAE** (95% CI +0.0350…+0.1036) and `S_cap_size` by **+0.0529** (+0.0196…+0.0890). Seed 1 reproduces the correspondence gain in the same direction (**+0.0309**, +0.0135…+0.0501; two-seed mean **+0.0495**). The original R1−S0 direction therefore survives parameter, readout, and edge-type-marginal controls. It is not the whole story: the best R1_size seed is still **+0.0713 MAE worse than G_ref** (0.2366 vs 0.1653; 43% of G_ref MAE), and the one pre-registered extra-update diagnostic did **not** show a clean gain. The remaining bottleneck looks like base-model/representation capability, not readout, and not a failure of endpoint correspondence.

---

## 1. What Phase 1b actually established, and what Phase 1c tested

The frozen corrigendum (`notes/phase1b_corrigendum.md`) records the limits that motivated this round:

| Phase 1b issue | Consequence | Phase 1c response |
|---|---|---|
| R1→M1 changed both the node update **and** the graph-level readout (mean-only → concat node/edge) | The R1/M1 contrast cannot isolate the update | Keep the direct edge readout fixed and add exactly one update (`M_keep_edge`) only after the primary gate |
| M1→M2 tied parameters but not compute | The “iteration” gain is not mechanism-identified | Do not rerun the M1/M2 contrast; test a compute-aware one-update diagnostic with an explicit confound check |
| M1_long ≈ M2 is a local null | Not a general claim that depth/iteration is useless | Report it as local; do not add deeper trunks |
| G_ref is a 6-layer **local GINE**, not global attention | The G_ref gap is not evidence for a global token | Keep G_ref as a local, higher-capacity reference; do not add attention |
| S0/R1 use mean-only readout; G_ref uses sum+mean | Readout class is confounded with the relation comparison | Add a parameter-matched size-aware sum+mean readout to both sides |
| Lipo difference was exploratory | Not a confirmed cross-task effect | Do not use Lipo for Phase 1c decisions |

Phase 1c therefore asks a narrower question than “is relational sufficiency true?”:

> With the same E2 patch input, the same readout rule, matched parameter capacity, and the same true bond-type multiset, does **endpoint correspondence** still add predictive value; and how much of the remaining gap is readout/capacity rather than relation?

---

## 2. CPU-first diagnosis of the Phase 1b bottleneck

All CPU diagnostics used only official-train Fit/Monitor labels and frozen Phase 1b predictions. Reserved and official valid/test were never read. Artifact: `results/phase1c/readout_diagnostics.json`.

### 2.1 The Monitor is hard but not size-dominated

- n = 1,000; constant baseline MAE = **1.5196**; linear size-only baseline MAE = **1.4473**.
- Errors are only weakly size-correlated: corr(|error|, n_atoms) = **+0.017** (S0), **+0.002** (R1), **+0.031** (G_ref).
- Large-minus-small bias is small (≈0.05 for S0/R1; ≈0.09 for G_ref). Size is not the main error axis.
- Error rises with ring count: S0 MAE ≈0.225 (0–1 rings), 0.228 (2), 0.339 (3), 0.455 (4–6). G_ref keeps this flatter (0.160, 0.124, 0.165, 0.232). Ring-rich molecules are the hard subset, but they are only 218/1,000 molecules.

### 2.2 Readout probe: count/sum-style readout looked much better in a linear probe

A ridge probe on a coarse, label-free local-patch key (root type, root degree, typed neighbour multiset, root ring closure, second-shell count; 1,103 keys on Fit; 2.8% Monitor OOV), fit on Fit and evaluated on Monitor:

| Probe readout | Monitor MAE | Monitor RMSE |
|---|---:|---:|
| mean only | 0.7174 | 1.1163 |
| counts / sum | **0.4868** | 0.8800 |
| mean + size | 0.7058 | 1.1032 |
| counts + size | **0.4821** | 0.8742 |

This motivated the size-aware sum+mean readout, but it **overstates** the expected end-to-end gain: it is a linear readout on frozen coarse keys, not the full model. The actual S0→S_cap_size gain is only **+0.0278 MAE** (Section 4), not ~0.23.

### 2.3 The E2 patch multiset contains size, but mean pooling discards it by construction

- Audit: 6,951 patches over 300 molecules; patch nodes 3–12; root degree visible in **100%** of patches; first-shell local degree positive in **100%**.
- Number of E2 tokens = n_atoms. Mean pooling is invariant to token multiplicity and therefore discards graph size. This is a real readout defect, not a hypothesis.
- Patches encode root and first-shell degree; distance-2 degrees are only partially visible. The patch key vocabulary is coarse (1,103 Fit keys), so patch tokens may also lose bond-level detail. This is a plausible base-model limitation, not established by the probe.

### 2.4 Optimization vs representation

- S0 and R1 both early-stop before 300 epochs (230 and 191), with best epochs 199 and 160. The original runs were not budget-limited.
- R1 has a larger train–Monitor gap (0.086) than S0 (0.056), and a lower train MAE; it is fitting more, not underfitting.
- G_ref has 439,559 parameters and a **smaller** train–Monitor gap (0.050) than R1, with much lower train and Monitor MAE. The G_ref advantage is not just “more parameters”; it has a better inductive bias/representation.
- Phase 1c therefore tests whether a matched-capacity, matched-readout relation model can recover part of that advantage, and whether an extra update helps.

---

## 3. Phase 1c controls

### 3.1 Information access (enforced by gates)

| Arm | E2 patches | True root edges | Endpoint tokens | Edge-type multiset | Edge count | Size features | Readout |
|---|---|---|---|---|---|---|---|
| `S_cap_size` | yes | **no** | — | — | — | yes | node mean + sum + size, graph FFN |
| `R1_size` | yes | yes | true `(h_u, h_v)` | yes | yes | yes | same node readout + edge mean/sum |
| `R1_marginal_size` | yes | no endpoint index | global mean token for both endpoints | yes | yes | yes | **identical** to `R1_size` |

`R1_marginal_size` is an **endpoint-correspondence ablation**, not a chemical causal experiment. It preserves topology-independent edge statistics (true bond-type multiset, edge count) and size, but destroys which endpoint pair carries which bond. The guard checks are in `phase1c_gates.py`; `S_cap_size` is rejected if any `root_edge_*` key appears, and `R1_marginal_size` is rejected if `root_edge_index` appears.

### 3.2 Capacity and compute

| Model | Params | Forward FLOPs (batch 64) | Params vs R1_size | FLOPs vs R1_size |
|---|---:|---:|---:|---:|
| S0 (Phase 1b anchor) | 56,963 | 434,141,824 | 0.491× | 0.671× |
| R1 (Phase 1b anchor) | 94,659 | 644,196,992 | 0.816× | 0.996× |
| `S_cap_size` | 115,883 | 441,539,200 | **0.999×** | **0.683×** |
| `R1_size` | 115,977 | 646,843,008 | 1.000× | 1.000× |
| `R1_marginal_size` | 115,977 | 646,843,008 | 1.000× | 1.000× |
| `M_keep_edge` (diagnostic only) | 132,681 | 694,913,664 | 1.144× | 1.074× |
| G_ref (Phase 1b anchor) | 439,559 | 1,185,134,336 | 3.790× | 1.832× |

**Important limitation:** `S_cap_size` is **parameter-matched**, not FLOP-matched. Its graph FFN is wider (177 hidden, 2 blocks) to match R1_size’s parameters, but R1_size’s relation/edge branch costs 46.5% more forward FLOPs. `Gain_over_capacity` is therefore a parameter-capacity control, not a pure compute control. The `R1_size` vs `R1_marginal_size` comparison is both parameter- and FLOP-identical, which is why it is the primary correspondence endpoint.

### 3.3 Data isolation and tests

- All 7 seed-0/seed-1/M_keep/R1_long run JSONs: 8,000 Fit / 1,000 Monitor / 1,000 Reserved **not scored**; label vault retains 9,000 Fit+Monitor labels, **0 Reserved**, **0 official valid/test**.
- Monitor molecule order identical across all runs.
- Protocol, config, CPU-gate, size-feature, split, label, and cache hashes all match the frozen Phase 1b/Phase 1c values (`run_validation.json`, status PASS).
- CPU gates: **17/17 PASS** in 0.4 s (`cpu_gates_phase1c.json`).
- Tests: **18 passed** (11 Phase 1c + 7 Phase 1b), including permutation invariance, batch-composition invariance, forward/gradient liveness, endpoint-ablation sensitivity, size-readout duplicate-graph separation, and frozen Phase 1b anchor equality.

---

## 4. GPU results

All runs: frozen recipe (L1, AdamW 1e-3, cosine T_max = planned epochs, eta_min 1e-5, batch 64, 300 epochs, patience 30, min_delta 1e-4, grad clip 5, dropout 0.1). Seed 0 first; seed 1 bought only for the pre-registered pair after the gate passed; `M_keep_edge` bought only after the follow-up gate passed.

| Model | Seed | Params | FLOPs (M) | Monitor MAE | Monitor RMSE | Train MAE | Best / run epoch | GPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 (Phase 1b) | 0 | 56,963 | 434.1 | 0.3173 | 0.8303 | 0.2617 | 199 / 230 | 394.9 |
| R1 (Phase 1b) | 0 | 94,659 | 644.2 | 0.2602 | 0.6175 | 0.1743 | 160 / 191 | 337.0 |
| G_ref (Phase 1b) | 0 | 439,559 | 1,185.1 | **0.1653** | **0.4482** | 0.1155 | 250 / 281 | 1,051.5 |
| `S_cap_size` | 0 | 115,883 | 441.5 | 0.2895 | 0.8333 | 0.2453 | 225 / 256 | 503.1 |
| `R1_size` | 0 | 115,977 | 646.8 | **0.2366** | **0.6069** | 0.1510 | 225 / 256 | 694.2 |
| `R1_marginal_size` | 0 | 115,977 | 646.8 | 0.3047 | 0.8304 | 0.2525 | 134 / 165 | 507.9 |
| `R1_size` | 1 | 115,977 | 646.8 | 0.2810 | 0.7689 | 0.2283 | 134 / 165 | 381.8 |
| `R1_marginal_size` | 1 | 115,977 | 646.8 | 0.3119 | 0.8279 | 0.2678 | 122 / 153 | 428.7 |
| `M_keep_edge` (diagnostic) | 0 | 132,681 | 694.9 | 0.2370 | 0.6184 | 0.1761 | 141 / 172 | 579.1 |
| `R1_size` long (diagnostic) | 0 | 115,977 | 646.8 | 0.2961 | 0.7931 | 0.2489 | 83 / 114 (planned 322) | 433.5 |

Phase 1b GPU seconds are from its run manifest and are **not** part of the Phase 1c budget; Phase 1c times are from the current manifest. G_ref’s 1,051.5 s is its Phase 1b run.

### 4.1 Primary endpoints (seed 0)

Sign convention: **positive Δ = the right-hand / second model is better**.

| Endpoint | Δ MAE | 95% paired bootstrap CI | Δ RMSE | RMSE agrees? |
|---|---:|---:|---:|---|
| `Gain_pair` = MAE(marginal) − MAE(R1_size) | **+0.0682** | +0.0350 … +0.1036 | +0.2235 | yes |
| `Gain_over_capacity` = MAE(S_cap) − MAE(R1_size) | **+0.0529** | +0.0196 … +0.0890 | +0.2264 | yes |
| MAE(marginal) − MAE(S_cap) | −0.0153 | −0.0229 … −0.0077 | +0.0029 | marginal RMSE slightly lower, MAE worse |
| MAE(S0) − MAE(S_cap) | +0.0278 | +0.0178 … +0.0378 | **−0.0030** | **no** (S_cap RMSE slightly worse) |
| MAE(R1) − MAE(R1_size) | +0.0236 | +0.0010 … +0.0462 | +0.0105 | yes |
| MAE(R1) − MAE(S_cap) | −0.0293 | −0.0605 … −0.0008 | −0.2158 | R1 better on both |
| MAE(R1_size) − MAE(G_ref) | +0.0713 (R1 worse) | +0.0420 … +0.1012 | +0.1587 | G_ref better |

Bands were frozen before GPU: negligible < 0.01, meaningful ≥ 0.02. Both primary gains are above the meaningful band with CIs excluding zero; both are therefore not readout/capacity/marginal artifacts at the pre-registered level.

### 4.2 Seed replication

| Seed | R1_size | R1_marginal_size | `Gain_pair` | 95% CI | RMSE agrees? |
|---:|---:|---:|---:|---:|---|
| 0 | 0.2366 | 0.3047 | +0.0682 | +0.0350 … +0.1036 | yes |
| 1 | 0.2810 | 0.3119 | +0.0309 | +0.0135 … +0.0501 | yes |
| mean | 0.2588 | 0.3083 | **+0.0495** | — | both positive |

The seed-1 pair gain is about half the seed-0 gain, and the seed-1 R1_size is 0.2810 (close to original R1 seed 0, 0.2602). The direction is consistent, but with two seeds the effect could be near the lower seed-1 value. Single-seed + bootstrap is screening, not a precise effect estimate; the two-seed mean is the honest point estimate.

### 4.3 Distributional read: the gain is in the extreme tail, not a uniform shift

| Comparison | Win rate of second model (lower abs error) | Median AE (first → second) | p90 AE (first → second) |
|---|---:|---:|---:|
| R1_size vs marginal (seed 0) | 53.1% | 0.1169 → 0.1002 | 0.435 → 0.381 |
| R1_size vs marginal (seed 1) | 54.7% | 0.1326 → 0.1130 | 0.406 → 0.392 |
| R1_size vs S_cap (seed 0) | **49.2%** | 0.1031 → 0.1002 | 0.363 → 0.381 |
| S_cap vs S0 | 57.8% | 0.1288 → 0.1031 | 0.456 → 0.363 |
| R1_size vs R1 | 57.3% | 0.1235 → 0.1002 | 0.430 → 0.381 |
| G_ref vs R1_size | 57.1% (G_ref better) | 0.1002 → 0.0840 | 0.381 → 0.290 |

Two different effects are visible:

- **S_cap_size** improves the bulk (median and p90) but **not the extreme tail**: its RMSE is slightly worse than S0 (0.8333 vs 0.8303). The size-aware readout + graph FFN capacity helps typical molecules, but the worst outliers remain.
- **R1_size** improves mean MAE and RMSE substantially, but it beats S_cap on only **49.2%** of molecules; the mean gain is driven by a minority of high-error molecules. The relation branch is not a uniform improvement; it is a tail/hard-molecule effect. This also explains the large RMSE gain (+0.2235) relative to the mean gain (+0.0529).

---

## 5. Answers to the seven questions

### Q1. Which Phase 1b conclusions are strengthened, narrowed, or refuted?

**Strengthened — endpoint correspondence is real.** The Phase 1b S0→R1 direction (0.3173 → 0.2602) survives a parameter-matched readout control and a FLOP-identical edge-type-marginal ablation. In Phase 1c the clean correspondence contrast is `R1_size` vs `R1_marginal_size`: +0.0682 at seed 0 and +0.0309 at seed 1, same sign, two-seed mean +0.0495. This is stronger evidence than Phase 1b because the only difference between the two arms is whether the relation sees true endpoint pairs.

**Narrowed — the original R1−S0 gap is not a pure relation effect.** Both Phase 1b arms used mean-only readout and had different capacity. A size-aware sum+mean readout with matched graph-FFN capacity improves **both** sides by a similar amount: S0→S_cap +0.0278, R1→R1_size +0.0236. The residual `R1_size − S_cap` gap is +0.0529, essentially the original R1−S0 gap (0.0571). So the relation effect is not explained away, but the original gap size includes a readout/capacity component that Phase 1b did not control.

**Narrowed / effectively refuted for this architecture — “one update helps.”** Phase 1b’s R1→M1 contrast (0.2602 → 0.2756, i.e. M1 worse) changed the readout, so it could not identify an update effect. Phase 1c’s `M_keep_edge` keeps the direct edge readout and adds exactly one residual node update. Under the same 300-epoch recipe it is **0.2370 vs R1_size 0.2366** (Δ = −0.0004; 95% CI −0.0218…+0.0210; RMSE slightly worse), despite +14.4% parameters and +7.4% FLOPs per epoch. There is no evidence that this one update helps. This does not prove that depth/receptive field is irrelevant in general; it does mean the Phase 1b M1 result should not be read as an update mechanism gain.

**Still unestablished — tied-round iteration.** Phase 1b’s M1→M2 difference was parameter- but not compute-identical, and M1_long ≈ M2 was a local null. Phase 1c does not rerun M2 and provides no positive update evidence. The local null remains local. The R1_long diagnostic is a methodological warning: stretching the cosine schedule to 322 epochs changed the optimization problem and early-stopped at epoch 83/114 with train MAE 0.2489 vs 0.1510 for the 300-epoch R1_size. A “compute-matched” longer run is not automatically a clean control.

**Strengthened caution — G_ref is not global attention.** G_ref is a 6-layer local GINE with a 6-hop receptive field, 439,559 parameters, and sum+mean readout. The Phase 1c gap to G_ref persists (Section Q4); it is a local, deeper/higher-capacity reference, not evidence for a global attention token.

**Quantified — readout class matters, but is secondary.** The size-aware readout improves S0 by +0.0278 MAE; it is a real defect fix (mean pooling discards size by construction). But it does not explain the relation gap, and for S_cap it does not improve RMSE. The readout is a secondary axis, not the dominant bottleneck.

**Unchanged — Lipo is exploratory.** Phase 1c is ZINC-only; no cross-task claim is made.

**Refined — broken adjacency.** The Phase 1b rewire control destroyed topology and bond semantics as well as endpoint correspondence. Phase 1c’s marginal arm is a cleaner correspondence-only ablation: it preserves edge-type multiset, edge count, size, modules, parameters, FLOPs, and readout, and still loses to R1_size. This is consistent with the rewire result but isolates the correspondence component. It is not a chemical causal experiment and does not claim that any specific bond rearrangement is chemically meaningful.

### Q2. Does the graph-level readout explain part of the ZINC gap?

**Yes, a small part — but it is not the main gap.**

- `S_cap_size` vs S0: **+0.0278 MAE** (CI +0.0178…+0.0378), but RMSE **−0.0030** (slightly worse). The control bundles two changes: size-aware sum+mean readout and a wider graph FFN to match parameters. The improvement cannot be attributed to readout alone. It helps the bulk (median 0.1288→0.1031; p90 0.456→0.363) but not the extreme tail.
- `R1_size` vs R1: **+0.0236 MAE** (CI +0.0010…+0.0462), RMSE +0.0105. The same readout/capacity bundle gives a similar gain on the relation side.
- The **relation gap remains**: `R1_size − S_cap` = **+0.0529** (CI +0.0196…+0.0890), almost the original R1−S0 gap (0.0571). Readout/capacity moves both sides up; it does not close the relation difference.
- The CPU ridge probe suggested count/sum readout could be much better on a linear probe (0.4868 vs 0.7174 MAE), but the end-to-end gain is only ~0.028. The probe measured a different object (linear readout on coarse frozen keys) and overstates the achievable effect.

**Conclusion:** readout is a real, fixable defect, but it explains only ~0.024–0.028 MAE of the Phase 1b gap, not the relation effect and not the absolute gap to G_ref. The capacity and readout changes are bundled in `S_cap_size`; a pure readout decomposition would need a separate control, which was not run.

### Q3. Does endpoint correspondence survive parameter and edge-type controls?

**Yes, within the limits of the controls.**

- Primary: `Gain_pair` = MAE(marginal) − MAE(R1_size) = **+0.0682** at seed 0 (CI +0.0350…+0.1036) and **+0.0309** at seed 1 (CI +0.0135…+0.0501); two-seed mean **+0.0495**; RMSE agrees at both seeds.
- `R1_marginal_size` has the **same modules, parameters (115,977), forward FLOPs (646,843,008), readout, size features, true bond-type multiset, and edge count** as `R1_size`. The only change is that both endpoint tokens are replaced by the molecule global mean. It is worse than R1_size, and also worse than `S_cap_size` by 0.0153 MAE. So the relation branch’s value is not from edge-type statistics alone, and not from simply adding parameters/FLOPs; the branch needs true endpoint correspondence to be useful.
- The marginal model also fits training worse (train MAE 0.2525 vs S_cap 0.2453 and R1_size 0.1510). It is not merely overfitting; the uninformative endpoint tokens make the branch harder to optimize and less useful.
- Parameter-capacity control: `Gain_over_capacity` = **+0.0529** (CI +0.0196…+0.0890), RMSE +0.2264. Caveat: S_cap is parameter-matched but **not FLOP-matched** (0.683× R1_size FLOPs). The marginal comparison is the cleaner, FLOP-identical control.
- The original R1 (94,659 params, mean readout) still beats S_cap by **+0.0293 MAE** (CI +0.0008…+0.0605), despite S_cap having 21% more parameters and the “better” size-aware readout. This is supportive but cross-model; it is not the primary control.

**Limits:** S_cap has one seed; the pair has two. The seed-1 pair effect is about half the seed-0 effect (0.0309 vs 0.0682), so the two-seed mean +0.0495 has a wide seed-level range. The marginal ablation removes correspondence but preserves edge-type counts; it is not a chemical causal experiment and does not test whether specific bond types are chemically meaningful.

### Q4. How large is the MAE gap to G_ref?

| Candidate | Seed | MAE | G_ref MAE | Gap (candidate − G_ref) | 95% CI |
|---|---:|---:|---:|---:|---:|
| `R1_size` | 0 | 0.2366 | 0.1653 | **+0.0713** | +0.0420 … +0.1012 |
| `R1_size` | 1 | 0.2810 | 0.1653 | **+0.1157** | — |
| `R1_size` two-seed mean | — | 0.2588 | 0.1653 | **+0.0935** | — |
| `S_cap_size` | 0 | 0.2895 | 0.1653 | **+0.1242** | +0.0877 … +0.1645 |
| `R1_marginal_size` | 0 | 0.3047 | 0.1653 | **+0.1394** | +0.1027 … +0.1796 |

The best R1_size seed is still 43% worse than G_ref in MAE (0.0713/0.1653); the seed-1 gap is 70% worse. On the molecule level, G_ref has lower absolute error on 57.1% of Monitor molecules, with median 0.0840 vs 0.1002 and p90 0.290 vs 0.381. The gap is broad, not a single-bin artifact.

G_ref is a 6-layer local GINE (439,559 params, 1,185.1M forward FLOPs, 1.83× R1_size), not global attention. Its advantage is consistent with deeper local propagation, atom/bond-level tokens, and/or a better message-passing form; Phase 1c cannot separate these. No official ZINC test was read, so this is an internal-Monitor statement only.

### Q5. Is there still a gain from one update under the same direct-edge readout and compute budget?

**The pre-registered matched-compute comparison is optimization-confounded; the same-recipe comparison shows no gain.**

Pre-registered `M_keep_edge` diagnostic:

- `M_keep_edge`: 0.2370 MAE, 0.6184 RMSE, 141/172 epochs, 132,681 params, 694.9M FLOPs.
- `R1_size_long` (planned 322 epochs to match M_keep’s FLOPs): 0.2961 MAE, 0.7931 RMSE, **83/114** epochs, train MAE **0.2489** (vs 0.1510 for the 300-epoch R1_size).
- Pre-registered Δ = MAE(R1_long) − MAE(M_keep) = **+0.0591** (CI +0.0328…+0.0874), RMSE +0.1747. This would label `UPDATE_HELPS_WITH_DIRECT_EDGE_READOUT` if taken at face value, but the R1_long arm failed to optimize: its train MAE is 0.098 higher than R1_size, and its actual FLOP ratio vs M_keep is **1.62×**, not the planned 1.074×. The longer cosine schedule is a different optimization problem, so the pre-registered contrast is **`MATCHED_COMPUTE_ARM_OPTIMIZATION_CONFOUNDED`**, not evidence for the update.

Same-recipe comparison (both 300 epochs, frozen LR schedule):

- `M_keep_edge` 0.2370 vs `R1_size` 0.2366; Δ = **−0.0004** (CI −0.0218…+0.0210), RMSE −0.0114 (M_keep slightly worse), win rate 50.8%.
- M_keep has **+16,704 parameters (+14.4%)** and **+7.4% FLOPs per epoch**. If extra capacity/compute alone helped, it should win; it does not. This is a conservative negative: under this frozen recipe, one extra generic update with the direct edge readout preserved is at best neutral and slightly worse.

Limits: the update is a residual MLP on concatenated endpoint messages, not a GINE layer; it is one update, not “depth” in general. The node readout uses the updated node state while the edge readout stays pre-update; that is the intended “keep the direct edge readout” design. A different update form or schedule could behave differently. The result does **not** prove that deeper/receptive-field changes are useless; it shows that this particular “more of the same” update is not the missing ingredient.

### Q6. Is the bottleneck more like a real scientific bottleneck or base-model capability?

**The evidence favors a base-model/representation capability gap, not a readout bottleneck and not a failure of endpoint correspondence.**

- Endpoint correspondence is real: `Gain_pair` is positive at both seeds and survives a FLOP-identical marginal control. The scientific claim “true endpoint relations carry predictive signal” is supported.
- Readout/capacity is real but secondary: it improves both sides by ~0.024–0.028 MAE; it does not close the relation gap (+0.0529) or the G_ref gap (+0.0713 best seed).
- One extra generic update is neutral/slightly worse. Simply adding depth in the current patch-level relation architecture is not a free win.
- G_ref is a **local** 6-layer GINE on atom/bond tokens, not a global attention model. Its advantage is therefore more plausibly due to token granularity, 6-hop receptive field, edge-conditioned message passing, or capacity/optimization, not to a global token. The patch abstraction (3–12 nodes per patch, 1,103 coarse keys, partial distance-2 visibility) is a plausible information bottleneck, but Phase 1c did not test atom-level tokens or patch-detail ablations.

**Defensible statement:** the remaining gap is a **base-model capability/architecture gap** (or an unresolved representation gap), not a readout defect and not evidence that endpoint correspondence is irrelevant. The data do not establish which capability is missing. The “scientific bottleneck” framing (relational sufficiency) is not refuted; it is insufficient by itself to close the performance gap.

### Q7. If only one next step is allowed, what should it be, and what is the stop condition?

**Do one CPU-only residual/representation audit before buying any new GPU mechanism.** Use the frozen R1_size and G_ref per-molecule predictions (and the existing Monitor structures) to decompose

`|err_R1_size| − |err_G_ref|`

by patch-observable variables: n_atoms, n_bonds, ring count, degree moments, bond-type counts, root/first-shell degree, patch size, number of repeated coarse patch keys, and patch-key entropy. Compare the same decomposition for S_cap and R1_marginal to see which residual components are relation-specific.

The audit should answer one falsifiable question: **is the remaining gap concentrated in molecules whose E2 patch tokens are ambiguous or information-poor (large patches, repeated coarse keys, high ring count), or is it uniform/noise-like?**

**Stop condition:**

- If **no** patch-observable variable explains **≥0.02 MAE** of the R1_size−G_ref gap, or the residual is uniform/noise-like, **stop the SSRC route** and report the absolute-gap negative. Do not add global tokens, deeper trunks, new structural encodings/distance channels, or SSL.
- If a specific patch-observable variable explains ≥0.02 MAE, pre-register **one** compute-matched diagnostic that changes only that variable. Stop if its gain is <0.02 MAE or its CI crosses zero. Do not run a mechanism sweep.

Why this and not another GPU arm: the one-update result already argues against “more of the same”; the readout/capacity control is done; the remaining uncertainty is whether the patch representation itself is the bottleneck. The audit is cheap, uses frozen predictions, respects the forbidden list, and prevents buying an architecture without a specific hypothesis.

A lower-priority robustness check is `S_cap_size` seed 1 to firm up the single-seed capacity control. It is **not** part of the pre-registered seed gate and should not be run before the residual audit; the pair result already has two seeds, while the absolute gap is the larger open question.

---

## 6. What is established, and what is not

### Established (within this task/split/recipe)

1. True endpoint correspondence adds predictive value beyond patch-only pooling: `Gain_pair` > 0 at both seeds, two-seed mean +0.0495, and the marginal control is identical in modules/params/FLOPs/readout/edge-type multiset.
2. The original R1−S0 direction is not a readout artifact: R1_size still beats S_cap by +0.0529 with a parameter-matched, size-aware readout on both sides.
3. A size-aware sum+mean readout with matched graph-FFN capacity improves both S0 and R1 by ~0.024–0.028 MAE; mean-only readout was a real defect.
4. The best R1_size seed remains +0.0713 MAE worse than a local 6-layer GINE (G_ref; 439,559 params, 1.83× FLOPs). The absolute gap persists.
5. One extra generic update with the direct edge readout preserved is neutral/slightly worse under the same 300-epoch recipe; the pre-registered longer-schedule comparison is optimization-confounded.

### Not established

- That endpoint correspondence is sufficient to close the gap to G_ref, or that it is the dominant factor in absolute ZINC performance.
- That the remaining gap is specifically due to atom-level tokens, depth, receptive field, or GINE’s edge-conditioned message form; Phase 1c did not isolate these.
- That the one-update null generalizes to other update forms, depths, or schedules.
- That the seed-0 pair gain (+0.0682) is the true effect size; seed 1 gives +0.0309, and the two-seed mean is +0.0495 with only two seeds.
- That `S_cap_size`’s single-seed result is stable; the capacity control is parameter-matched but not FLOP-matched and has one seed.
- Any official ZINC test, SOTA, or cross-task (Lipo) claim.
- Any chemical causal claim from `R1_marginal_size`; it is an endpoint-correspondence ablation, not a chemical intervention.

---

## 7. GPU budget accounting

From `results/phase1c/run_manifest.tsv` (rr status job times + run JSON hashes):

| Phase | Runs | Train-loop GPU s | GPU h |
|---|---|---:|---:|
| Smoke (3 arms, 1 epoch) | 3 | 1.9 | 0.0005 |
| A: S_cap, R1_size, R1_marginal (seed 0) | 3 | 1,705.2 | 0.4737 |
| B: R1_size, R1_marginal (seed 1) | 2 | 810.5 | 0.2251 |
| C: M_keep_edge, R1_size_long | 2 | 1,012.6 | 0.2813 |
| **Total run-level** | **10** | **3,530.3** | **0.9806** |
| Job occupancy (includes ~9 s/job overhead) | 8 jobs | 3,609.0 | **1.0025** |

- Cap: **2.000 GPU-h**. Remaining: **1.0194 GPU-h** run-level / **0.9975 GPU-h** occupancy.
- Phase A cap 0.60 h → used 0.474 h. Phase C cap 0.50 h → used 0.281 h.
- All jobs exit 0; no failed-job GPU time. No post-hoc width/dropout/LR/readout changes.
- Seed 1 was bought only after the four pre-registered gate conditions passed. `M_keep_edge` was bought only after the two-seed pair sign/mean and the R1_size−G_ref gap gate passed. `S_cap_size` seed 1 was **not** bought because it is not in the pre-registered gate.

---

## 8. Reproduction and artifacts

### Commands (local / CPU)

```bash
uv run python tracks/structural-sufficiency/code/phase1c/prep_size_features.py  # if rebuilding size sidecar
uv run python tracks/structural-sufficiency/code/phase1c/phase1c_gates.py
uv run pytest -q tracks/structural-sufficiency/code/phase1c/tests/test_phase1c.py \
  tracks/structural-sufficiency/code/phase1b/tests/test_phase1b.py
uv run python tracks/structural-sufficiency/code/phase1c/diagnose_zinc_readout.py
uv run python tracks/structural-sufficiency/code/phase1c/analyze_phase1c.py
uv run python tracks/structural-sufficiency/code/phase1c/validate_phase1c_runs.py
uv run python tracks/structural-sufficiency/code/phase1c/make_run_manifest.py
```

### GPU commands (rr, A100; all ran at commit `f57aec8`)

```bash
rr run res ssrc-p1c-zinc-Scap-s0 --pool res-gpu0 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model S_cap_size --seed 0 --device cuda
rr run res ssrc-p1c-zinc-R1size-s0 --pool res-gpu1 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model R1_size --seed 0 --device cuda
rr run res ssrc-p1c-zinc-R1marg-s0 --pool res-gpu0 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model R1_marginal_size --seed 0 --device cuda
# seed 1 pair, only after the pre-registered gate
rr run res ssrc-p1c-zinc-R1size-s1 --pool res-gpu0 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model R1_size --seed 1 --device cuda
rr run res ssrc-p1c-zinc-R1marg-s1 --pool res-gpu1 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model R1_marginal_size --seed 1 --device cuda
# follow-up diagnostic, only after the two-seed gate
rr run res ssrc-p1c-zinc-Mkeep-s0 --pool res-gpu0 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model M_keep_edge --seed 0 --device cuda
rr run res ssrc-p1c-zinc-R1long-s0 --pool res-gpu1 --result tracks/structural-sufficiency/runs/phase1c \
  -- python tracks/structural-sufficiency/code/phase1c/run_phase1c.py train --dataset zinc --model R1_size --seed 0 --device cuda --epochs 322 --label long
```

### Frozen hashes

| Artifact | SHA-256 |
|---|---|
| `notes/phase1c_protocol.md` | `545e58037386fde248a84b9ad104cbb7dab8ccd0f961a0392fdb933a14547ca5` |
| `notes/phase1b_corrigendum.md` | `4947d8cb4f34beffef164a3dac40edc8f41d981b32f4fdb00b773239648ad13d` |
| `configs/phase1c.yaml` | `60c50f369983d97a7dec9c90f9f35d5c2497f5a40fdd8a07fce38c046bddeb82` |
| `results/phase1c/cpu_gates_phase1c.json` | `aa91573e138f8910b92f96d31235ccb67a571320fd8edf2b0a0a14272ff75c4a` |
| `results/phase1c/zinc_size_features.pt` | `8a20858e8b660f2b7befee0ba36e17a339667393aa4d40cabb04a05c5e6fd434` |
| `results/phase1b/split_zinc_phase1b.json` | `fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49` |
| `results/phase1b/zinc_labels_fit_monitor.json` | `631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78` |
| ZINC cache | `76c1749277f1c4019f8b94b5a3a9bc7b9f83abb8b11b7feed60449f47a9ed84c` |
| `results/phase1c/analysis_phase1c.json` | `3d2dc898c27f46e61f4187165fd337945858bae08dad6c0f2d475d0535bc9683` |
| `results/phase1c/run_manifest.tsv` | `f0c95dc747112b770a1bbdcdcfbe59fadf62b209c8cacba59913ef0eb2f156a1` |
| `results/phase1c/run_validation.json` | `cceaa9caa115bcdaec1a3415c61e5f3e73478361323976388836a61b3deeaa30` |

### Artifact notes

- The frozen `readout_diagnostics.json` has a Python-3.12 serialization artifact in one ring-bin value; it does not affect any conclusion. New analysis/validation/manifest artifacts use standard JSON.
- `run_validation.json` checks protocol/config/gate/size/split/label/cache hashes, label-vault isolation, monitor order, and per-arm information access across all 7 seed-0/seed-1/M_keep/R1_long runs.
- The Phase 1c run JSONs retain per-molecule Monitor predictions and labels; the report tables are derived from those.

---

## 9. Bottom line

`PURSUE_RELATIONAL_SUFFICIENCY` should be treated as **conditionally supported**, not confirmed as a sufficient route:

- The **core scientific claim** — true endpoint correspondence carries predictive value beyond patch-only pooling, readout capacity, and edge-type marginals — is **strengthened** and survives the Phase 1c controls.
- The **performance claim** — that this relation is enough to close the gap to a strong local GINE — is **not supported**. The best R1_size seed is still +0.0713 MAE worse than G_ref; the two-seed mean gap is +0.0935.
- The **iteration/update claim** remains unsupported: the one clean update diagnostic is neutral/slightly worse under the same recipe, and the pre-registered longer-schedule comparison is optimization-confounded.
- The next move is a **CPU residual/representation audit**, not another mechanism. If the audit cannot attribute ≥0.02 MAE of the remaining gap to a patch-observable variable, stop the route and report the absolute-gap negative.
