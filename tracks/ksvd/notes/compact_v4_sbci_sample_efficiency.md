# Compact-v4 Shared-Basis Compositional Interaction (SBCI) — Sample-Efficiency Falsification

**Track:** `ksvd` · **Protocol:** `compact_v4_sbci_v1`
**Module:** `tracks/ksvd/experiments/luyin16/zinc_compact_v4_sbci.py`
**Results:** `tracks/ksvd/results/compact_v4_sbci/`
**Tests:** `tracks/ksvd/tests/test_compact_v4_sbci.py`
**Official valid:** never loaded. **Official test:** never loaded.

**Verdict: Case A — `SBCI MINIMAL SHARED-BASIS FACTORIZATION NO-GO`.**

The single authorised seed0 run (N=3600) reaches SOUP probe MAE
**0.182924** vs the reused `compact-v4-smallhead` baseline **0.176633**
(`delta_SE = -0.006291`, RAW `-0.006215`, `DDR = -0.128`). The candidate is
*not* more sample-efficient; it is slightly worse, with the two-seed
replication, N=7200 capacity check, N=1800 and seed2/3 all correctly not
purchased.

---

## 1. Motivation

The `compact-v4-smallhead` family (82,115 params) is strongly data-limited over
1800–7200 unique training molecules: each data doubling buys roughly `0.05`
probe MAE (`+0.049116` seed0, `+0.051234` seed1 at 3600→7200). A gain of that
size is far larger than the ~0.02 architecture-gap scale, so the natural next
question is not "where is information lost?" but "why is the same information
hard to learn from few examples?".

## 2. Why further localization audits were stopped

The information-bottleneck line was already closed:

* `raw_graph_patch_sufficiency` — pre-neural patch system has `LB(P3)=0`
  hard aliasing and better target geometry than generic raw references
  (`Delta_pre=-0.0878`).
* `stagewise_representation_collision` — no stage shows a stable target-relevant
  geometry degradation; patch/pair encoders even create geometry.
* `sample_efficiency_gain_localization` — Case H, no target-independent support
  family (token / relation / neighbour density) is a robust per-molecule
  predictor of the doubling gain; primary well-covered stress test underpowered.

Adding another structural statistic is therefore not justified. The remaining
scientifically justified move is a **function-class** intervention.

## 3. Strong data-scaling evidence

Reused verbatim from the frozen audit (never retrained):

| N | seed0 SOUP | seed1 SOUP |
|---:|---:|---:|
| 1800 | 0.225184 | 0.240017 |
| 3600 | 0.176633 | 0.175501 |
| 7200 | 0.127517 | 0.124268 |

`G_36→72 = +0.049116 / +0.051234`; mean `+0.050175`, paired bootstrap lower
`+0.043256 > 0`. This is the scale any architecture must beat to be interesting.

## 4. Why basic coverage was insufficient

The localization audit found token/relation/NN support associations only in the
tail, with out-of-sample `R^2 <= 0.004`, and the powered coarse-key
well-covered subset still retained 60–72% of the gain. Basic structural
coverage therefore does not explain the scaling gain; the hypothesis had to be
about *how the same information is parameterised*.

## 5. Function-sharing hypothesis

> Structural effects should be expressed through reusable low-rank functional
> coordinates shared across molecules.

Concretely: endpoint identities, relations and contexts currently form fairly
free context-specific latent functions. Restricting them to a **small shared
functional basis** `z_i = phi(h_i)` with **factorized relational composition**
should reduce the number of examples needed to generalise.

## 6. Difference from capacity expansion

SBCI is deliberately **smaller** (62,045 vs 82,115 params, `-20,070`). No module
was added for capacity; every removed compact-v4 module is deleted, not kept as
a residual bypass. A smaller constrained function class that performed better
would have been stronger evidence than an equal-parameter improvement.

## 7. Retained compact-v4 components

Exactly unchanged, and initialised bit-identically: historical rooted
tokenization; token embedding (36,420); parent embedding (256); patch
construction and 146D shell / pair / topology descriptors; patch encoder
(14,192); complete patch-pair enumeration; the 5 distance buckets; the
deterministic 23D relation descriptor; the global encoder (3,136); the topology
encoder (552); loss / optimizer / data protocol.

## 8. Removed original relation/centre pathway

Not instantiated at all: original pair projection `h -> u` (768);
original `q` encoder (`relation_encoder`, 1,360); fixed `q` moment pathway;
original 213D centre-update MLP (`center_update`, 15,888); original `h^1`
centre state; original 302D `R`; original `302 -> 13 -> 13 -> 1` small head
(4,135). The original 21,633-param full head is also absent. Integrity gates
T12/T13/T14 prove none of these attributes exist on the candidate.

## 9. Shared functional basis

`K = 16` fixed (no sweep). `z_i = phi(h_i) in R^16` with
`Linear(48,32) -> ReLU -> Linear(32,16)`, no LayerNorm, dropout, residual or
output activation. Every patch type, molecule and context uses the same `phi`;
`z_i` is a signed learned coordinate, not a chemical basis.

## 10. Relation-only modulation

`r_ij in R^23` is the deterministic relation descriptor only
(`relation_input_inventory.json`): distance-bucket one-hot (5) + log1p distance
(1) + patch overlap (5) + boundary overlap (3) + path bond composition (4) +
log1p path count (1) + adjacent bond (4). No learned endpoint state enters
`psi`. `a_ij = psi(r_ij)`, `g_ij = 1 + tanh(a_ij) in (0,2)^16`, with
`Linear(23,32) -> ReLU -> Linear(32,16)`. No attention/softmax/bucket MLP.

## 11. Factorized pair interaction

`p_ij = z_i * z_j * g_ij in R^16`. This is the **only** pair representation;
no `[z_i, z_j, z_i - z_j, z_i*z_j]` concatenation, bilinear matrix, rank-R
tensor, post-product MLP or additive pair branch. Gate T8 verifies the exact
factorisation; T9 verifies endpoint-swap symmetry.

## 12. Bucket composition

The five canonical distance buckets are reused exactly. For centre `i` and
bucket `b`, `P_{i,b} = {p_ij : j in N_b(i)}` is reduced to population mean,
population std and `log(1+count)`, giving `16+16+1 = 33` per bucket and
`A_i in R^165`. Empty buckets are exactly `mean=0`, `std=0`, `log=0`
(gate T11).

## 13. Low-order centre composition

`delta_i = W_c A_i + b_c` with a single `Linear(165,16)` — no hidden layer,
activation, LayerNorm or dropout — and `c_i = z_i + delta_i`. This is the
deliberate function-class restriction: the hypothesis is about shared
factorised pair effects, not another learned composer.

## 14. Graph aggregation

Three fixed invariant channels over `{c_i}`: mean (`C_mu`), population std
(`C_sigma`), and sqrt-normalised sum `C_s = n^{-1/2} sum_i c_i` (a
size-sensitive channel that does not grow linearly with `n`). Concatenated,
`C in R^48`. No task-specific global feature was added.

## 15. Parameter accounting

| block | params |
|---|---:|
| identity/token storage (typed + parent) | 36,676 |
| patch encoder | 14,192 |
| basis encoder `phi` | 2,096 |
| relation modulator `psi` | 1,296 |
| centre composer | 2,656 |
| global branch | 3,136 |
| topology branch | 552 |
| final head | 1,441 |
| **candidate total** | **62,045** |

Baseline `compact-v4-smallhead` = **82,115**; `delta_params = -20,070`; hard cap
respected. The released 45,057 parameters of removed modules were not
reallocated.

## 16. Initialization fairness

Canonical baseline is instantiated first and all shared tensors are snapshotted;
the SBCI model is then built; SBCI-only modules are re-initialised under the
deterministic sub-seed `sha256("SBCI-v1|seed") mod (2^31-1)`; finally every
shared tensor is copied back. Both seeds: **20 shared tensors**, `max_abs_diff
= 0.0`, shared-state SHA-256 match. Gates F0.1/F0.2 additionally verify the
retained patch/global/topology encoder outputs are bit-identical at
initialisation. No warm start, no distillation, no frozen backbone.

## 17. N3600 protocol

Exact compute-matched reuse of the sample-efficiency audit: `D3600` nested
target-independent subset (salt `compact-v4-sample-efficiency-nested-subset-v1-20260912`,
dataset hash `3be9f7e1…`), 800 selection, 2000 probe, official valid/test never
loaded. Adam lr `1e-3`, wd `1e-5`, batch 128, L1, no scheduler, gradient clip
5.0, `max_optimizer_steps=13,680`, `selection_eval_interval=57`,
`patience=40` selection evaluations, earliest-step tie-break.

## 18. Data-Doubling Recovery Ratio

For seed `s`: `G_s = B_{36,s} - B_{72,s}` (baseline doubling gain),
`delta_SE,s = B_{36,s} - C_{36,s}` (candidate vs baseline at N=3600), and
`DDR_s = delta_SE,s / G_s` — the fraction of the observed 3600→7200
data-doubling benefit substituted by the architecture.

## 19. Seed0 result

| quantity | value |
|---|---:|
| SBCI SOUP probe MAE | **0.182924** |
| SBCI RAW probe MAE | 0.187545 |
| baseline SOUP | 0.176633 |
| baseline RAW | 0.181330 |
| `delta_SE` SOUP | **-0.006291** |
| `delta_SE` RAW | -0.006215 |
| `DDR_0` | **-0.128** |
| best 800-selection MAE | 0.215511 @ step 3534 |
| run length | 5,814 steps / 102 evals / 200.5 effective epochs |
| wall clock | 210.7 s |

Paired bootstrap over the 2,000 probe molecules (B=2000, seed 20261007):
SOUP mean `-0.006291`, 95% CI `[-0.012760, -0.000176]`; RAW mean
`-0.006215`, 95% CI `[-0.013078, +0.000696]`. The candidate is significantly
*worse* under the primary SOUP estimator.

## 20. Conditional seed1 replication

Not purchased. The pre-registered seed0 strong-GO gate required
`delta_SE >= +0.015` with positive bootstrap support and positive RAW; the
observed `delta_SE = -0.006291` is a clean NO-GO by a wide margin, so no
seed1 run exists.

## 21. Conditional N7200 capacity check

Not run. It is only authorised after a replicated N3600 sample-efficiency GO.

## 22. RAW/SOUP consistency

RAW and SOUP agree in direction (both negative) and differ by `7.6e-5`. No
estimator ambiguity: the candidate is worse on both channels.

## 23. Compute cost

Parameter count `62,045` vs `82,115` (`-24.4%`). A 128-molecule forward pass is
`7.66 ms` vs `19.43 ms` baseline (`-60.6%`) because the pair encoder and the
213D centre-update MLP are gone; peak RSS is comparable (`~2.0 GB`). The
candidate is cheaper, not merely smaller. Performance and compute claims are
reported separately.

## 24. Relation to P1/P2/cell NO-GOs

P1 kept the current `q`/centre family and added a learned DeepSets-style
residual composer on top of the fixed per-(centre,bucket) statistics (NO-GO).
P2 kept the pair encoder and recomputed `q` after centre context (NO-GO).
`compact-v4-cell` added explicit persistent cycle cells (NO-GO). SBCI does
none of these: it **deletes** the free relation/centre function family and
replaces it with a shared basis times a relation-only gate. It is not a P1
rescue, not a staleness repair, and not a P2 variant.

## 25. Relation to the old compositional-sharing NO-GO

The old `compositional_patch_sharing_oracle` replaced rare/OOV **exact token
embeddings** with structurally matched frequent embeddings while keeping the
entire q/centre/302D-R/large-head pipeline and doing no training; every variant
regressed (KNN8 `-0.010882`, etc.) and structural KNN was no better than random
donors or the frequent mean. It changed an *input embedding*, with frozen
weights, not the *function class*. SBCI instead removes the free relation and
centre latent functions entirely, constrains pair effects to
`z_i*z_j*g(r_ij)`, and jointly retrains from scratch. The two experiments share
the intuition "share structure across molecules" but differ in every
mechanically relevant respect: object of sharing (token embedding vs functional
coordinates), training regime (frozen inference vs from scratch), and function
class (unchanged vs strictly restricted). The old NO-GO therefore does not
predict this result, and this NO-GO does not reopen the embedding-sharing
route.

## 26. What is and is not proven

**Proven (within seed0, official-train development environment):**

* a 62,045-param SBCI with exact baseline shared init and no bypass is
  trainable and stable (no NaN, all branch gradients alive every eval);
* the pair factorisation, endpoint symmetry, bucket semantics, permutation
  invariance and `R=88D` are exactly as pre-registered;
* it does not improve N=3600 probe MAE (`delta_SE = -0.0063`), and is
  significantly worse under the primary SOUP estimator.

**Not proven / not claimed:**

* that "compositional inductive bias is disproven" — only this minimal
  shared-basis factorisation is closed;
* anything about N=7200, seed1, or official valid/test;
* that K=16 / width / aggregation / relation form would rescue the idea (all
  explicitly forbidden here).

## 27. Sample-efficiency interpretation

This minimal low-rank reusable functional factorization does **not** recover a
meaningful fraction of the observed data-doubling benefit. The direction is
negative rather than merely sub-threshold, which is consistent with the
constraint being too strong at the tested scale: removing the free relation and
centre function families costs more at N=3600 than the shared basis buys back.
Per the pre-registered rules this is a clean scientific NO-GO, not an
optimization failure.

## 28. Official-valid/test lock

Official valid was never loaded. Official test was never loaded, even though
the decision is negative. Integrity gates T20/T21 and the module-level firewall
in the reused audit enforce this; no test MAE, prediction, subgroup or
checkpoint exists.

## 29. Final verdict

**`SBCI MINIMAL SHARED-BASIS FACTORIZATION NO-GO`.**

```
baseline N3600 seed0 SOUP   0.176633
SBCI     N3600 seed0 SOUP   0.182924
delta_SE (SOUP)            -0.006291    RAW -0.006215
DDR                       -0.128
bootstrap 95% CI           [-0.012760, -0.000176]
best 800 MAE                0.215511 @ step 3534 (not boundary-pinned)
seed1 / N7200 / N1800 / seed2/3   NOT run
```

No rescue. Do not change K, add a residual q/centre branch, add attention, or
widen the basis on this evidence. A future re-opening would need new
independent evidence.

---

## Q1–Q20

* **Q1** SBCI total = **62,045** params (≤ 82,115 cap).
* **Q2** Retained baseline modules: typed/parent embeddings, patch encoder,
  global encoder, topology encoder.
* **Q3** Removed / not instantiated: pair projection, relation encoder,
  distance gate, pair encoder, centre update, original head.
* **Q4** Relation input = 23D deterministic relation-only descriptor
  (no learned endpoint state).
* **Q5** `phi = Linear(48,32) -> ReLU -> Linear(32,16)`.
* **Q6** `psi = Linear(23,32) -> ReLU -> Linear(32,16)`; `g = 1 + tanh(a)`.
* **Q7** `p_ij = z_i * z_j * g_ij` (verified exactly).
* **Q8** `Linear(165,16)`; `c_i = z_i + delta_i`.
* **Q9** `R_SBCI` = **88D** = `[C(48), G(32), T(8)]`.
* **Q10** Shared init exact: 20 tensors, `max_abs_diff = 0.0`, hash match.
* **Q11** SBCI N3600 seed0 SOUP = **0.182924**.
* **Q12** `delta_SE,0` = **-0.006291** (SOUP), `-0.006215` (RAW).
* **Q13** `DDR_0` = **-0.128**.
* **Q14** SOUP 95% CI `[-0.012760, -0.000176]`.
* **Q15** N3600 seed1 **not purchased**.
* **Q16** n/a (seed1 not run).
* **Q17** Replicated GO = **false**.
* **Q18** N7200 **not purchased**.
* **Q19** n/a (N7200 not run).
* **Q20** Final case = **`SBCI_MINIMAL_FACTORIZATION_NO_GO` (Case A)**.

---

## Artifact map

* Locks/inventory: `architecture_lock.json`, `baseline_inventory.json`,
  `sbci_parameter_ledger.json`, `relation_input_inventory.json`,
  `initialization_lock.json`, `integrity_gates.json`, `branch_alive_tests.json`,
  `compute_audit.json`.
* Stage 1: `stage_run_N3600_I0T0.json`, `checkpoint_manifest_N3600_I0T0.json`,
  `soup_construction_N3600_I0T0.json`, `stage1_N3600_I0T0.json`,
  `stage1_bootstrap.json`, `stage1_decision.json`, `representation_diagnostics.json`,
  `probe_abs_err_{raw,soup}_N3600_I0T0.npy`, `curves/N3600_I0T0.csv`.
* Decision: `final_decision.json`, `answers_q1_q20.json`.
* Figures: `figures/figure1_architecture.png`,
  `figure2_N3600_baseline_vs_sbci.png`, `figure3_DDR.png`.
