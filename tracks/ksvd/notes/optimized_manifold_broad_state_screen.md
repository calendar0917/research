# Optimized-Manifold Broad Frozen-State Sufficiency Screen

Status: **INCONCLUSIVE — DO NOT BUILD OPTIMIZED OOF** (mandate Case C). The
primary adapter init was borderline; the mandated second adapter init flipped
seed 1 negative and the combined point estimate fell to `+0.00030`, i.e. below
the pre-registered CLEAR-NO-GO line `+0.0005`. No optimized OOF backbone is
authorised and the local witness tree is **not** reopened. No backbone was
trained; official test was never loaded.

Code: `tracks/ksvd/experiments/luyin16/zinc_optimized_manifold_broad_state_screen.py`
Tests: `tracks/ksvd/tests/test_optimized_manifold_broad_state_screen.py` (13 pass)
Results: `tracks/ksvd/results/optimized_manifold_broad_state_screen/`
Fingerprint version: `optimized_frozen_state_export_v1`

The one question:

> On the already fully-trained **optimized** compact-v4-hinge latent manifold,
> do the frozen node/patch/pair states (`h'_i`, `q_ij`) still carry a
> *low-complexity* increment of predictive signal beyond the final 302D graph
> vector `R`?

---

## 1. Motivation

Every local witness in the track (pair-endpoint association, centre-incidence
co-occurrence, triadic binding, function-basis accessibility, conditional
readout sufficiency, head-family) terminated NO-GO on the **historical
60-epoch** compact-v4-hinge manifold. The 60-epoch protocol was then shown to
be **materially under-trained** (`notes/compact_v4_training_sufficiency.md`):
extending the unchanged regime to 240/40 improved 4-seed valid
`0.169451 → 0.147201` and frozen test `0.136885 → 0.125179`.

Because the optimized representation **drifts** from the historical one
(normalized L2 drift ≈ 0.314, cosine ≈ 0.950; `representation_drift.json`), the
historical NO-GO stays a valid statement *about the old manifold* but cannot be
upgraded for free to the optimized manifold. This screen re-asks the *broadest*
possible question on the optimized manifold **before** paying for any expensive
optimized OOF reconstruction.

## 2. Why historical under-training changes the interpretation

A stronger baseline can change which discarded statistic becomes
task-relevant. The historical broad audit was run on a manifold that had not
converged; its negative result could in principle be an artifact of a weak
frozen backbone. The honest re-screening therefore re-runs the *same*
diagnostic architecture (only the backbone manifold and the evaluation protocol
change) rather than re-running each local witness.

## 3. What remains valid from historical audits

* The historical broad audit **still stands for the historical manifold**:
  corrected frozen-readout audit (`notes/frozen_conditional_readout_sufficiency_audit.md`)
  gave `B1 R-only ≈ 0.19050`, `E set/state ≈ 0.19229`, `ΔR = MAE(B1) − MAE(E)
  ≈ −0.00179` (2/5 folds positive) → NO-GO.
* The *measurement repairs* (pair grouping by `batch.batch[pair_index[0]]`;
  true pre-head `R` = 302D input to `head[0]`) remain valid and are reused
  verbatim.
* What is **not** automatically valid is the transfer of that NO-GO to the
  optimized manifold.

## 4. Why a broad gate comes before local revalidation

Local witnesses (endpoint / centre covariance / triad) are expensive to
re-run and were already negative once. The mandate is to first ask: *is there
any broad low-complexity exploitable state signal at all?* If not, there is no
reason to re-open the local witness tree on the optimized manifold.

## 5. Canonical-screening caveat (READ FIRST)

The two reused optimized checkpoints were **valid-selected**. Evaluating
adapters on official valid is therefore a **canonical-checkpoint screening
diagnostic**, not clean OOF mechanism evidence. Its only purpose is to decide
whether optimized OOF reconstruction is worth the compute. Official test is
never loaded; a positive result would only authorise a two-fold optimized OOF
*pilot*, never a five-fold spend directly.

## 6. Optimized checkpoint inventory

| seed | run | valid MAE | best epoch | params | state sha256 (prefix) |
|---:|---|---:|---:|---:|---|
| 0 | `Pstar_A2_long_seed0` | **0.14642023** | 169 | 99,613 | `60b7d297…` |
| 1 | `Pstar_A2_long_seed1` | **0.14933226** | 104 | 99,613 | `93bf4ec2…` |

Protocol (frozen): Adam, lr `1e-3`, wd `1e-5`, batch `128`, max_epochs `240`,
patience `40`, no scheduler, L1, best-valid, single-stage. Config sha256
`b9205b05…`. No backbone was trained in this stage (`_train_phase` never
called; Tests 1).

## 7. State export

`optimized_frozen_state_export_v1` (new version; historical 60-epoch caches are
refused by `load_export`). Captured with the *committed* corrected forward
hooks (`zinc_frozen_readout_sufficiency._capture_states`): true pre-head `R`
(`head[0]` *input*), updated patch/centre states `h'_i = patch_pre + centre_update`
(`model.center_update` hook), frozen pair states `q_ij = pair_encoder(...)`,
plus buckets and graph-local pair endpoints. Fingerprint fields: export
version, model commit, optimized protocol lock hash, backbone seed, checkpoint
sha256, config sha256, tokenizer version + fingerprint, vocabulary fingerprint,
topology fingerprint, split fingerprint, expected valid/best-epoch/params.

Dependency note: the export uses the **committed** forward hooks
(`_capture_states`); it does **not** depend on the uncommitted pure `encode()`
refactor in `zinc_patch_path_pooling.py`. No model numerics were touched.

## 8. Integrity gates (Gate 0 — all PASS, both seeds)

| gate | meaning | result |
|---|---|---|
| G0.1 | true pre-head `R` dimension | **302** |
| G0.2 | checkpoint/config/tokenizer/topology fingerprint | PASS (config sha + checkpoint sha re-checked) |
| G0.3a | stored/exported `yhat_0` == independent hook-free forward | max diff **0.0** |
| G0.3b | valid prediction anchor == optimized checkpoint valid MAE | abs err **0.0** (both seeds) |
| G0.4 | reconstruct `R` from `h'_i`/`q_ij`/buckets | max 6.1e-5 / 8.4e-5 |
| G0.5 | reconstructed `R` through frozen head == `yhat_0` | max 1.9e-6 / 1.4e-6 |
| G0.6/0.7 | pair counts `n(n-1)/2`; patch grouping | PASS |
| G0.9/0.10 | pair endpoints belong to the same molecule | PASS |
| G0.11/0.12 | every unordered pair consumed once, no self-loops | PASS |
| G0.6 inv | batch size 128 vs 97 + reversed order | R 0.0, ŷ 4.8e-7 |
| G0.8 inv | node relabelling changes neither `R` nor summary | R 2.3e-5, summary 1.8e-7 |

Any failure would be `INVALID / STOP`; none occurred.

## 9. Historical reader definitions (reused unchanged)

* **B0** — frozen optimized checkpoint prediction. No training.
* **B1** — parameter-matched R-only residual reader: `R → 13 → 13 → 1`,
  `yhat = yhat_0 + ρ_R(R)`, **4135** params.
* **E** — historical broad frozen-state reader: shared `φ_h: 48→16→8`,
  shared-across-buckets `φ_q: 16→16→8`,
  `s = [mean_i φ_h(h'_i); mean_{b} φ_q(q_ij)]` (8 + 5×8 = 48),
  `ρ: 350→8→1`, `yhat = yhat_0 + ρ([R; s])`, **4145** params.

Parameter mismatch **0.24 %** (≤ 2 %). Both readers inherit the historical
preprocessing (fit-only standardisation), L1/Adam(lr 1e-3, wd 0), full-batch,
400 epochs, patience 50, best selection-checkpoint, deterministic init. No
attention / Transformer / covariance / learned pair binding / new projection
sweep was introduced.

## 10. R-only control (B1)

B1 is the decisive control: the metric of interest is
`Δ_state = MAE(B1) − MAE(E)`. Beating B0 is *not* sufficient — a frozen head is
under-adapted to the final `R`, so a residual R-only head already beats B0
substantially (see §13).

## 11. Broad state reader (E)

E sees `R` plus the historical permutation-invariant mean summaries of the
frozen `h'_i` and `q_ij`. No new state inputs were added (no covariance, no
triad, no endpoint outer product, no raw descriptors, no corrected-token
features, no target components). This is an *exact broad state re-check*.

## 12. Adapter protocol

Official-train-only deterministic molecule-id hash split
(`optimized-manifold-broad-state-screen-v1-20260919`):
**7200 adapter-fit / 800 adapter-selection / 2000 train-probe**. No identical
whole-train manifest existed (the existing 7200/800/2000 manifests are the
per-fold OOF nested splits), so a fresh stable split was written to
`split_manifest.json`. `official valid` is touched once, after the adapter
checkpoint is locked, and never for training / selection / thresholds /
architecture.

## 13. Seed 0

Primary init (adapter seed 0), official valid (1000 molecules):

| reader | valid MAE |
|---|---:|
| B0 frozen optimized | 0.146420 |
| B1 R-only (4135) | 0.143495 |
| E broad state (4145) | 0.143125 |

`Δ_state(seed0) = +0.000370`; `Δ0 = B0 − E = +0.003296`. Paired molecule
bootstrap over the 1000 valid molecules: mean +0.00037, 95 % CI
[−0.00142, +0.00215], P(>0)=0.657. The bootstrap is an uncertainty
description only; **backbone-selection dependence remains** (the checkpoints
were valid-selected), and n=2 backbones is not a significance sample.

## 14. Seed 1

| reader | valid MAE |
|---|---:|
| B0 frozen optimized | 0.149332 |
| B1 R-only (4135) | 0.146420 |
| E broad state (4145) | 0.145270 |

`Δ_state(seed1) = +0.001150`; `Δ0 = B0 − E = +0.004062`. Bootstrap: mean
+0.00115, 95 % CI [−0.00087, +0.00318], P(>0)=0.864 (same descriptive caveat).

## 15. Bulk safety

Common-input bulk = valid molecules whose **target-independent** train-derived
`rare_le5_ratio` is below the 80th percentile of the adapter-fit molecules
(threshold 0.0714, 756 bulk valid molecules). `MAE(E) − MAE(B1)` in the bulk:
seed0 **+0.00041**, seed1 **−0.00028** → both ≤ +0.002. Bulk safe.

## 16. Adapter dependency (collapse check)

E genuinely consumes the injected states (mandate §38 / Test 12):

| seed | state-zero `max|Δ|` | within-molecule state permutation `max|Δ|` |
|---:|---:|---:|
| 0 | 0.1208 | 0.0265 |
| 1 | 0.0576 | 0.0338 |

Both far above the 1e-3 aliveness floor; prediction std ≈ 1.85. The state
branch is **not** collapsed, so the negative result is a real negative and not
an invalid diagnostic. (An extra per-inverse-variance note: E also beats B0 by
≈ +0.003–0.004, so the reader is learning something; it just does not add over
R.)

## 17. Historical vs optimized qualitative comparison

| audit | manifold | B1 R-only | E / set | Δ_state sign | verdict |
|---|---|---:|---:|---|---|
| historical corrected broad | under-trained 60-ep | 0.19050 | 0.19229 | **negative** | NO-GO |
| optimized broad (this) | optimized 240/40 | 0.14349 / 0.14642 | 0.14313 / 0.14527 | **weakly positive but non-robust** | NO-GO |

Absolute MAEs are **not** comparable (different backbone, protocol, split). The
qualitative conclusion is the same: a parameter-matched R-only reader is
essentially as good as the broad state reader.

## 18. OOF authorization decision

Primary init gives mean `Δ_state = +0.00076`, both seeds positive → not a clear
NO-GO but also far below the +0.003 advance requirement → **BORDERLINE**. Per
mandate §35 the second adapter init (adapter seed 1, same frozen backbones, all
else fixed) was run:

| seed | init 0 Δ_state | init 1 Δ_state | combined (mean of 2 inits) |
|---:|---:|---:|---:|
| 0 | +0.000370 | +0.000179 | +0.000274 |
| 1 | +0.001150 | **−0.000513** | +0.000318 |

Second init **flips seed 1 negative** and the two inits disagree in sign, so
the signal is non-robust (mandate §36 / Decision Case C) →
**INCONCLUSIVE — DO NOT BUILD OPTIMIZED OOF**. The combined point estimate
`+0.00030` is in addition below the §33 CLEAR-NO-GO line `+0.0005`; under either
reading the budget consequence is identical.

**Decision: DO NOT rebuild the old witness tree.** No optimized OOF pilot, no
endpoint/covariance/triad revalidation, no second backbone seed, no extra
adapter init. The historical NO-GOs are not reopened.

## 19. What is and is not proven

**Proven (within this screen):**

* The optimized manifold export is measurably correct (Gate 0 all pass; valid
  anchor reproduced exactly).
* A refit R-only residual head beats the frozen optimized head on valid
  (seed0 +0.0033, seed1 +0.0041) — a *late-readout adaptation* effect, not a
  state effect.
* A parameter-matched reader of the frozen `{h'_i}`/`{q_ij}` states adds only a
  weak, non-robust increment over that R-only reader (combined mean
  `+0.00030`, second-init sign flip).

**Not proven:**

* That no information exists beyond `R` information-theoretically.
* That moment pooling is sufficient, or that the patch paradigm has reached
  its ceiling.
* That a *different* (higher-capacity, learned) parameterisation of the states
  would also fail — only the historical low-complexity broad reader fails.

## 20. Final verdict

**`INCONCLUSIVE_DO_NOT_BUILD_OOF`** (mandate Case C). The optimized latent
manifold provides **no broad screening support** for frozen internal states
containing readily exploitable information beyond the final graph
representation `R`: the primary init was borderline (`+0.00076`), the mandated
second init was sign-unstable (seed 1 flipped to `−0.00051`), and the combined
estimate `+0.00030` sits below the CLEAR-NO-GO line. This is not a positive
result — it simply does not reach the advance bar, so the expensive OOF
reconstruction is not authorised. Bottleneck map:

```
historical state audit       NO-GO
optimized broad screen       INCONCLUSIVE (no OOF)
```

This terminates mass revalidation of local frozen witnesses (endpoint,
centre covariance, triad). The next discussion is representation-family /
structural-computation redesign — **not** re-running the historical experiment
set.

---

## Required questions Q1–Q15

| # | answer |
|---|--------|
| Q1 | optimized compact-v4-hinge seed0 (valid 0.146420, epoch 169) and seed1 (valid 0.149332, epoch 104) |
| Q2 | true pre-head R = 302D = unary 97 + pair 165 + global 32 + topology 8 |
| Q3 | updated patch/centre states `h'_i` (48D) and frozen pair states `q_ij` (16D) |
| Q4 | historical corrected broad audit: B1 0.19050, set/state 0.19229, ΔR ≈ −0.00179 → NO-GO |
| Q5 | seed0 valid B0 0.14642 / B1 0.14349 / E 0.14312 (init 0) |
| Q6 | seed0 Δ_state = **+0.000370** (init 0); +0.000179 (init 1) |
| Q7 | seed1 valid B0 0.14933 / B1 0.14642 / E 0.14527 (init 0) |
| Q8 | seed1 Δ_state = **+0.001150** (init 0); −0.000513 (init 1) |
| Q9 | mean Δ_state = +0.00076 (init 0); +0.00030 (combined) |
| Q10 | sign consistent on init 0 (both positive); **not** consistent after init 1 |
| Q11 | common-input bulk safe (seed0 +0.00041, seed1 −0.00028; gate ≤ +0.002) |
| Q12 | yes — E depends on the injected states (zero/permutation sensitivity 0.026–0.121) |
| Q13 | second adapter init **was** required (borderline) and was run; seed1 flipped negative |
| Q14 | optimized OOF pilot authorised: **NO** |
| Q15 | **INCONCLUSIVE_DO_NOT_BUILD_OOF** |

## Outputs

`results/optimized_manifold_broad_state_screen/`:
`checkpoint_inventory.json`, `export_integrity.json`, `export_fingerprints.json`,
`adapter_protocol_lock.json`, `split_manifest.json`, `seed0_results.json`,
`seed1_results.json`, `valid_summary.csv`, `train_probe_summary.csv`,
`common_input_bulk.json`, `adapter_dependency.json`, `final_decision.json`,
`second_init_results.csv`, `state_exports/*.npz`,
`figures/figure1_valid_mae.png`, `figures/figure2_delta_state.png`.

No optimized OOF results were produced (no backbone training permitted).
