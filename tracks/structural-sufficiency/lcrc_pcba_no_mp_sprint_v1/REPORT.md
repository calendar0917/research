# REPORT — LCRC PCBA no-edge-MP sprint v1

Round: `lcrc-pcba-no-mp-sprint-v1`  
Host: `res` (`a100-2`, 2 × A100-SXM4-40GB), driver 550.163.01, torch
2.5.1+cu124, CUDA 12.4, FP32. Same execution regime as the historical
structural-gain Phase-3 runs.  
Formal runs: 2026-11-05 23:35 – 2026-11-06 04:33 UTC.  
Status: **complete**; pre-registered labels
**`STATIC_POSITIVE_BUT_BELOW_V`** and **`EDGE_MP_BUNDLE_RETAINS_ADVANTAGE`**.
`GLOBAL_COMMUNICATION_PROMISING` was **not** triggered. No Test target was read
or scored.

## Result in one paragraph

All three no-bond-MP additions improve over the historical C1 control on both
seeds, and the gains cover most of the 127 scored PCBA tasks. The strict static
route is best represented by **A (four-head attention readout)** at a two-seed
mean of **0.245279** (0.246356 / 0.244203), +0.015854 over the paired
historical C, but 0.013083 below the historical V mean and below 0.250 on both
seeds. **G (graph-context feedback)** is the best no-MP arm overall:
two-seed mean **0.248040** (0.250848 / 0.245231), 0.010323 below V and +0.018614
over C. V remains clearly ahead of every new arm by the pre-registered 0.008
margin. Attention weights are permutation-correct but unstable across seeds
(atom-weight Pearson 0.09–0.30), so they are not treated as stable chemical
attributions. Total GPU use, including preflight and failed retries: **8.277
GPU-hours**; wall clock ≈5.6 h,
inside the 7.5 h target and 8 h hard wall.

## 1. Question and frozen design

Can explicit S1/S2 local statistics plus a capacity-matched static computation
or graph-level attention readout approach the previous round's V **without
iterative hidden-state message passing along real chemical bonds**?

| Arm | Added computation | Reads bonds? | New params |
|---|---|---|---:|
| S | two per-atom residual FFN blocks, 64→160→64 | no | 41,664 |
| A | four permutation-invariant graph-level attention pools + residual readout; `h` unchanged | no | 42,460 |
| G | two blocks with per-molecule mean-context broadcast | no | 41,728 |
| V | historical bond-message blocks + graph context (reference only) | yes | 42,624 |

All arms start from the same frozen C1 initialisation, use the same 40-epoch
Train order, masked observed-entry BCE, dropout 0.10, AdamW, batch 128,
1-epoch warmup `3e-5 → 3e-4`, then per-step cosine to `1e-5` over 109,520
updates, gradient clip 5.0, and no early stopping. The primary comparison is
the fixed epoch 40. Historical C/V were not retrained.

## 2. Correctness and comparability gates

All gates passed before the first formal epoch:

* CPU tests 17/17; GPU preflight on both A100s for S/A/G.
* Parameter counts: C 245,184; S 286,848; A 287,644; G 286,912; V 287,808.
  New-parameter gaps to V: S 2.25%, A 0.38%, G 2.10% (all ≤3%).
* Base C1 init fingerprints for seed 0 and seed 1 equal the historical Phase-3
  values bitwise (`1b1f1ad3…`, `3af93edb…`).
* 40-epoch order plans equal the historical Phase-3 plans
  (`06f43729…`, `0af6ec6a…`) including the consumed-40 prefix hashes.
* Recipe fields and numerical LR schedule equal the previous Phase-3 recipe;
  Train-only normalisation, split, labels and cache hashes equal the frozen
  identities.
* Zero-residual identity at initialisation: S/A/G reproduce C1 logits
  (GPU max |Δ| ≤ 2e-5, tolerance 1e-4). CUDA `index_add` is not bitwise stable;
  the audited tolerance is used.
* Perturbing `bond_index`/`bond_feat` leaves each added branch unchanged
  (S/G bitwise; A within 1e-5 because its pooled sums use CUDA `index_add`).
* S has no inter-atom aggregation; A normalises within each molecule and does
  not leak across molecules; G's context does not leak across molecules.
* Atom-order permutation changes S/G per-atom outputs only by the permutation;
  A's per-atom weights follow the permutation and its graph readout is
  invariant.
* New parameters receive effective gradients after three real updates. The
  four head output biases in A have structurally zero gradient because a
  constant score shift does not change the per-graph softmax; this is recorded
  and is not a training failure.
* No non-finite loss, gradient or logit in any formal run. No Test row was
  read.

`analysis/architecture_audit.json`, `analysis/information_access_audit.json`
and `analysis/initialization_parity.json` contain the machine-readable audits.

## 3. Phase B — seed 0, epochs 1–20

| Arm | AP10 | AP20 | AP20 − AP10 |
|---|---:|---:|---:|
| S | 0.207231 | 0.230899 | +0.023667 |
| A | 0.221849 | 0.238289 | +0.016439 |
| G | 0.217189 | 0.242068 | +0.024879 |

The pre-registered early-stop rule (all three with AP20 < 0.220 **and**
AP20 − AP10 < 0.002) was not close. The round continued to Phase C.

## 4. Phase C — seed 0 to epoch 40

Fixed comparison epoch 40; no best-epoch substitution.

| Arm | AP40 | Train AP | Train BCE | Valid BCE | Tasks improved vs C | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.238164 | 0.394877 | 0.031243 | 0.046149 | 82/127 | 0.558 |
| A | 0.246356 | 0.419657 | 0.030858 | 0.045773 | 93/127 | 0.663 |
| G | **0.250848** | 0.410716 | 0.030522 | 0.045822 | 99/127 | 0.762 |
| C (hist.) | 0.231945 | 0.361277 | 0.032682 | 0.046317 | — | — |
| V (hist.) | 0.257772 | 0.428159 | 0.029773 | 0.045398 | — | — |

`NO_STRONG_SEED0_SIGNAL` (all three AP40 < 0.242) was not triggered because G
reached 0.250848. Phase D was therefore executed for all three arms.

## 5. Phase D — independent seed 1, 40 epochs from scratch

| Arm | AP10 | AP20 | AP30 | AP40 | Two-seed mean | Gain vs C (s0 / s1) | Gap to V mean | GPU-h (s1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S | 0.209567 | 0.227712 | 0.237176 | 0.239145 | **0.238654** | +0.006219 / +0.012239 | −0.019708 | 1.463 |
| A | 0.219373 | 0.239793 | 0.245663 | 0.244203 | **0.245279** | +0.014411 / +0.017297 | −0.013083 | 1.373 |
| G | 0.217107 | 0.235535 | 0.244461 | 0.245231 | **0.248040** | +0.018903 / +0.018325 | −0.010323 | 1.218 |

Historical paired controls: C seed 0 0.231945, seed 1 0.226906. Historical V:
0.257772 / 0.258952, mean 0.258362.

Per-task breadth (at epoch 40):

| Arm / seed | Tasks improved vs C | Mean AP delta vs C | Tasks improved vs V | Mean AP delta vs V |
|---|---:|---:|---:|---:|
| S / 0 | 82/127 | +0.006219 | 38/127 | −0.019609 |
| S / 1 | 92/127 | +0.012239 | 31/127 | −0.019808 |
| A / 0 | 93/127 | +0.014411 | 40/127 | −0.011416 |
| A / 1 | 101/127 | +0.017297 | 34/127 | −0.014750 |
| G / 0 | 99/127 | +0.018903 | 49/127 | −0.006924 |
| G / 1 | 100/127 | +0.018325 | 31/127 | −0.013722 |
| G / two-seed mean | 105/127 | +0.018614 | 34/127 | −0.010323 |

All 128 tasks have ≥20 observed Train positives; the usual "sparse task"
stratum is empty on PCBA Train, so the gains are not confined to a few
low-support tasks.

## 6. The seven required questions

### 6.1 What Validation AP did S/A/G reach?

| Arm | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| S | 0.238164 | 0.239145 | 0.238654 |
| A | 0.246356 | 0.244203 | 0.245279 |
| G | 0.250848 | 0.245231 | 0.248040 |

### 6.2 Which strict no-iterative-MP arm is best, and does it reach 0.25?

A is the better strict static arm: two-seed mean 0.245279 vs S 0.238654. It does
**not** reach 0.250 on either seed (0.246356, 0.244203). The best single run in
the strict static route is A seed 0 = 0.246356.

### 6.3 What is the two-seed gap between S/A and V?

V mean = 0.258362.

* A: 0.258362 − 0.245279 = **0.013083** below V.
* S: 0.258362 − 0.238654 = **0.019708** below V.

Paired per seed: A is 0.011416 (seed 0) and 0.014750 (seed 1) below V; S is
0.019609 and 0.019808 below V.

### 6.4 Does G show that non-bond propagation is competitive?

G is the best no-MP arm: mean 0.248040, **0.010323 below V**, and it beats A by
0.002761. It closes about 64% of V's mean gain over C
(`(0.248040−0.229425)/(0.258362−0.229425)`), descriptively.

The pre-registered `GLOBAL_COMMUNICATION_PROMISING` rule required both
(a) |G mean − V mean| ≤0.008 and (b) G mean − best S/A mean ≥0.008. Neither is
met (0.010323 and 0.002761), so the label is **not** triggered. The honest
statement is narrower: graph-level context feedback without chemical-bond
messages is the strongest no-MP variant here, but it does not yet establish a
statistically or practically equivalent alternative to V.

### 6.5 Do the improvements cover most PCBA tasks rather than a sparse subset?

Yes for the improvement over C: 93–101/127 tasks for A, 99–100/127 for G,
82–92/127 for S, with positive mean and median per-task deltas. Against V,
however, the new arms improve only 21–49/127 tasks; V remains better on the
majority of tasks. So the claim "broadly better than C1" is supported, while
"broadly matches V" is not.

### 6.6 What is the relation between parameters, FLOPs, speed and gain?

* Parameters are deliberately capacity-matched: new parameters are 41,664
  (S), 42,460 (A), 41,728 (G) versus V's 42,624 (≤2.25% difference).
* Measured forward FLOPs on a fixed 4-molecule/83-atom batch:
  total S 39.91 M, A 34.76 M, G 38.62 M; added-module FLOPs S 6.80 M (17.0%),
  G 5.51 M (14.3%), A 1.65 M (4.75%).
* Step p50 across formal runs: S 35.2–48.6 ms, A 42.2–44.0 ms, G 39.0–51.4 ms;
  preflight on quiet GPUs was 41.1 / 44.8 / 40.6 ms. Co-tenancy changed
  throughput by ~10–20%, so speed differences are **not** attributed to the
  mechanisms.
* More added FLOPs did not buy more AP: S has the most added FLOPs but the
  lowest AP; A has the fewest added FLOPs but is the best strict static arm;
  G is best overall at intermediate added FLOPs. With one architecture per
  mechanism and no within-mechanism FLOPs sweep, this is a descriptive
  observation, not an efficiency law.

### 6.7 Which limited, testable claim is supported, and which stronger claims remain unsupported?

**Supported (limited):** under the frozen C1/S1/S2 representation, identical
40-epoch scratch recipe and PCBA Validation, all three no-bond-MP additions
produce broad, two-seed-positive gains over C1. G (graph-context feedback) is
the strongest no-MP variant at 0.248040 mean, but V's bond-MP bundle still
leads by 0.010323 mean AP.

**Not supported:** that edge message passing is theoretically necessary; that
S1/S2 statistics are proven insufficient; that V's advantage is entirely
caused by MP2; that attention weights identify chemical causes; that G is
"no inter-node communication" (it uses graph-level context); or that the
current evidence proves equivalence to V.

## 7. Attention diagnostics (Arm A only, descriptive)

Fixed epoch-40 checkpoints, 64 deterministically spaced Validation molecules,
no checkpoint selection based on interpretability:

* Normalised mean entropy per head: 0.708, 0.491, 0.524, 0.424.
* Cross-seed atom-weight Pearson per head: 0.296, 0.086, 0.301, 0.172;
  top-1 agreement: 0.219, 0.094, 0.234, 0.156. The heads are not stable across
  independent seeds.
* Permutation check: max |Δweight| = 2.98e-7; weights follow atom permutation.
* Mean pooled-vector L2 norm per head: 4.93, 6.32, 6.24, 6.82.
* Top-weight atoms are mostly C/N (aromatic and non-aromatic), degree 2–3,
  roughly half in rings. These are descriptive counts on 64 molecules, **not**
  causal chemical attributions; the four heads are not 128-task-specific
  attribution maps.

Artifacts: `analysis/attention_diagnostic.json`,
`analysis/attention_pooled.npz`.

## 8. Facts, inferences, open questions

### Measured facts

* All formal runs completed; no non-finite events; total 8.277 GPU-hours
  (9 completed training jobs plus preflight/retry entries).
* S/A/G improve over C on both seeds; best new arm G = 0.248040 mean.
* Best strict static arm A = 0.245279 mean; both seeds below 0.250.
* V = 0.258362 mean remains above every new arm by ≥0.010323.
* G is 0.002761 above A, below the pre-registered 0.008 separation.
* Gains vs C are broad across tasks; gains vs V are not.
* Attention heads are permutation-correct but seed-unstable.

### Inferences and competing explanations

* The V advantage is consistent with either (i) real bond-message propagation
  adding information beyond the frozen S1/S2 state, or (ii) extra optimisation
  capacity/inductive bias in the bundle rather than MP specifically. This round
  separates "bond MP" from "static/graph-global computation" but does not
  isolate MP2 from graph-context feedback inside V.
* G's advantage over A/S suggests that global context feedback is a useful
  non-bond channel. It does not prove that graph-global communication is
  sufficient in general; one dataset, one width, one recipe.
* A achieves more AP with fewer added FLOPs than S, so raw capacity/FLOPs do
  not order the results. The mechanism and optimisation path matter.
* Low cross-seed attention agreement cautions against reading individual head
  patterns as learned chemical motifs.

### Open questions

* Would a FLOPs-matched or width-matched within-mechanism sweep change the
  ordering of S/A/G?
* Is V's advantage mostly from bond messages or from the graph-context feedback
  that V also contains? A future V-minus-context arm would test this.
* Are the seed-to-seed differences in G (0.250848 vs 0.245231) ordinary
  optimisation variance or a real stability difference from A/S?
* Do these conclusions transfer to Lipophilicity/ESOL or to a different
  readout? This round deliberately did not run those datasets.
* Can attention readouts be made seed-stable (e.g., by sharing or constraining
  heads) without changing the frozen architecture? That is a new experiment,
  not a post-hoc change to this round.

## 9. Pre-registered labels and stop rules

| Rule | Outcome |
|---|---|
| Phase B early stop (all AP20 <0.220 and AP20−AP10 <0.002) | not triggered |
| Phase C stop (all AP40 <0.242) | not triggered (G 0.250848) |
| `STATIC_CLOSE_TO_V` | not met (A mean 0.245279; both seeds <0.250; gap 0.013083) |
| `STATIC_POSITIVE_BUT_BELOW_V` | **assigned** (A and S both seeds ≥+0.005 over C) |
| `GLOBAL_COMMUNICATION_PROMISING` | not triggered (gap 0.010323 >0.008; G−A 0.002761 <0.008) |
| `EDGE_MP_BUNDLE_RETAINS_ADVANTAGE` | **assigned** (V mean exceeds every new mean by ≥0.008) |
| `SECOND_SEED_NOT_RUN_RESOURCE_LIMIT` | not applicable; Phase D complete |
| `INCOMPLETE` | not applicable |

These are resource/research-direction labels, not significance tests.

## 10. Reproducibility and artifacts

* Frozen protocol: `PROTOCOL.md`; design and usage: `README.md`.
* Code: `code/nmp_common.py`, `nmp_model.py`, `nmp_data.py`,
  `nmp_checkpoint.py`, `nmp_train.py`, `nmp_preflight.py`, `nmp_audit.py`,
  `nmp_analyze.py`, `nmp_attention.py`.
* Tests: `tests/test_nmp_model.py` (17 passed).
* Per-run outputs under `runs/phaseB_seed0_*`, `runs/phaseC_seed0_*`,
  `runs/phaseD_seed1_*`: curve, fixed-epoch Validation report, per-task AP,
  checkpoint, optimizer/RNG state, order plan, Git commit and stop reason.
* Analysis: `analysis/architecture_audit.json`,
  `analysis/information_access_audit.json`,
  `analysis/initialization_parity.json`,
  `analysis/phase_b_seed0_screen.json`,
  `analysis/phase_c_seed0_full.json`,
  `analysis/phase_d_seed1_confirmation.json`,
  `analysis/per_task_comparison.csv`, `analysis/per_task_summary.json`,
  `analysis/attention_diagnostic.json`, `analysis/gpu_ledger.json`,
  `analysis/run_registry.json`, `analysis/failed_gpu_attempts.json`.
* GPU ledger: 8.277 h total, 6.723 h below the 15 h target; hard limit not
  approached. Wall clock ≈5.6 h, below the 7.5 h target.
* Official Test: never evaluated; all run records carry
  `test_evaluated: false`.
