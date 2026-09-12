# Compact-v4 Small-Head End-to-End Compression Confirmation

**Date:** 2026-09-12
**Scope:** one from-scratch compressed architecture, two seeds, official test locked
**Module:** `tracks/ksvd/experiments/luyin16/zinc_compact_v4_smallhead_e2e.py`
**Results:** `tracks/ksvd/results/compact_v4_smallhead_e2e/`
**Tests:** `tracks/ksvd/tests/test_compact_v4_smallhead_e2e.py` (15 pass)
**Official test:** never loaded.

**Verdict: `REPLICATED_NON_INFERIOR_COMPRESSION_WITH_FAVORABLE_POINT_ESTIMATES`**

The integrated compact-v4 graph representation `R` (302D) does **not** require
its 21,633-parameter graph head for successful end-to-end learning. Replacing
it with the pre-audited fixed 4,135-parameter raw small head removes **17,498**
parameters (17.57% of the whole model, 80.89% of the head) and is
validation-non-inferior on both development seeds -- with favourable point
estimates.

---

## 1. Motivation

The Parameter Allocation & Representation Leverage audit
(`notes/parameter_allocation_representation_leverage.md`) found exactly one
undisputed donor class in the ~100K compact-v4 budget: the late-readout graph
head. Every previous *additive* module returned a clean or sub-threshold NO-GO
(P1 composer, P2 refresh, cycle cells, covariance, triad, endpoint association,
attribute factorization, FM head, function basis). The audit's own conclusion
was a **compression-only opportunity**, not a "spend the freed parameters"
opportunity.

This stage executes the compression half of that conclusion under the exact
frozen optimized protocol.

## 2. Why parameter reallocation was not authorized

The audit identified a strong donor (graph head) but **no R-SUPPORTED
representation-forming recipient**. Pair path / centre update were R-DISFAVORED
by P1/P2/covariance/triad; higher-order persistence was R-DISFAVORED by
compact-v4-cell; the identity lookup was L-NO (already a frequency-adaptive
hybrid with negative SVD saving); topology/global are positive and out of
scope. Reallocating the freed parameters to any of them would have re-opened an
already-closed hypothesis. This experiment therefore changes the head only and
releases the rest **unused**.

## 3. Frozen optimized-manifold donor evidence

On the two already-trained optimized backbones (frozen `R`, refit small head;
historical `Sraw`/`Hsmall` protocol: raw `R`, direct `yhat=f(R)`, L1,
Adam(lr 1e-3, wd 0), minibatch 512, horizon 800, head_seed 0):

| seed | H0 valid | small-head valid | Δ | params saved |
|---|---:|---:|---:|---:|
| 0 | 0.146420 | 0.143208 | **−0.003213** | 17,498 (80.9%) |
| 1 | 0.149332 | 0.147754 | **−0.001578** | 17,498 (80.9%) |

Mean Δ = −0.002396. This labelled the head **H-STRONG**.

## 4. Why frozen-head reducibility is not enough

The donor evidence has the form

```
already-trained backbone -> frozen R -> refit small head
```

This is a *readout-fitting* statement. The large head may still have played a
role during **joint representation learning**: the historical graph-head refit
audit already showed the jointly-trained head was under-adapted to the final
`R` (late-readout adaptation). The actual question is

> can the large head be deleted during representation co-training?

so the test must be `random init -> small-head model -> full joint end-to-end
optimization`. This stage runs exactly that; it does **not** reuse the frozen
head-refit numbers as its result.

## 5. Exact baseline architecture

Canonical optimized compact-v4-hinge: **99,613** parameters, `R` = **302D**
(unary 97 + pair 165 + global 32 + topology 8). The graph head is

```
Linear(302,64) -> LayerNorm(64) -> ReLU -> Dropout(0.05)
-> Linear(64,32) -> ReLU -> Linear(32,1)
```

= 21,633 params (21,505 linear + 128 LayerNorm). The reused seed0 baseline
anchor reproduces official valid **0.14642022556537995** exactly from the
existing checkpoint (no retraining).

## 6. Exact small-head architecture

The audited historical `Sraw` / `Hsmall` reader
(`zinc_graph_head_function_family.GenericReader`), reused verbatim:

```
Linear(302,13) -> ReLU -> Linear(13,13) -> ReLU -> Linear(13,1)
```

* raw `R` input (no standardisation, no fit-only transform);
* direct `yhat = f(R)` (no residual over `yhat_0`);
* no LayerNorm, no dropout;
* head init under the historical convention `head_seed = 0`;
* **no** width / depth / activation / normalisation / dropout search.

## 7. Parameter accounting

| model | total | head | non-head |
|---|---:|---:|---:|
| compact-v4 (baseline) | 99,613 | 21,633 | 77,980 |
| compact-v4-smallhead | **82,115** | **4,135** | 77,980 |

Δparams = −17,498. Relative total reduction **17.566%**; relative head
reduction **80.886%**. Mechanical account: `82,115 = 99,613 − 21,633 + 4,135`.

## 8. Architecture lock

`architecture_lock.json` records the baseline fingerprint, the exact small-head
layers/activation/normalisation/dropout, `R` = 302D, the small-head and total
params, the inherited training protocol, seed policy, non-inferiority and
replication thresholds, and the official-test lock. It was written **before**
any training and was not modified afterwards.

## 9. Initialization matching

`initialization_match_seed0.json` / `initialization_match_seed1.json`:

* deterministic canonical baseline instantiated first, all shared tensors
  snapshotted;
* small-head model instantiated under the same seed, then **every shared tensor
  copied back exactly**;
* the small head is then re-initialised independently under `head_seed = 0`.

Result, both seeds: **40 shared tensors, `max_abs_diff = 0.0`, hash match
`true`**. The seed1 shared hash differs from seed0, so seed1 does **not** reuse
seed0 weights. This is a normal from-scratch initialisation -- never a
trained-checkpoint warm start.

## 10. End-to-end training protocol

Identical to the canonical optimized compact-v4 baseline, unchanged:

```
Adam, lr = 1e-3, weight_decay = 1e-5, batch_size = 128,
max_epochs = 240, patience = 40, scheduler = none,
loss = L1, gradient_clip_norm = 5.0,
checkpoint_selection = best official-valid MAE, single stage
```

The training loop mirrors `zpp._train_phase` (same `torch.manual_seed` seeding,
same data-loader shuffle seed offsets `seed+91011` / `seed+91012`, same
`clip_grad_norm_(5.0)`, same best-valid selection). No head-specific LR, no
warmup, no scheduler, no longer horizon.

Integrity gate G0.9 confirms the protocol is byte-for-byte the canonical one;
G0.8 confirms pre-head `R` is **bit-identical** at initialisation (max abs diff
0.0 on a real 128-molecule valid batch).

## 11. Seed0 result

| metric | value |
|---|---:|
| small-head seed0 valid | **0.145334** |
| baseline seed0 valid | 0.146420 |
| best epoch | 176 / 240 |
| epochs run | 216 (early stop) |
| optimizer steps | 17,064 |
| wall clock | 1,028 s |
| params | 82,115 |

## 12. Non-inferiority gate

* `D0 = S0 − B0 = 0.145334 − 0.146420 = −0.001086`
* seed0 gate `D0 <= +0.002` -> **PASS** (indeed D0 < 0).
* No optimization ambiguity: no NaN, head and backbone gradients alive every
  epoch, best epoch 176/240 is not boundary-pinned.

Stage 1 verdict: **`SEED0_NON_INFERIOR`**, authorising seed1.

## 13. Training dynamics

`curves/smallhead_seed0_curve.csv` and `..._seed1_curve.csv` record `epoch,
train_mae, valid_mae, lr, optimizer_steps, checkpoint_selected, head_grad_norm,
backbone_grad_norm, head_weight_norm` every epoch. Both runs converge normally:
train MAE decreases monotonically, official valid reaches a clear minimum well
before the horizon (epoch 176 and 143), early stopping fires after patience 40,
and neither the head nor the backbone gradient collapses. Diagnostics were
computed after `backward()` and did not touch the RNG.

## 14. Common-input bulk

Target-independent bulk = valid molecules whose train-derived rare-patch
fraction (`rare_le5_ratio`) is below the train 80th percentile (756/1000).

| seed | v4 full valid | small-head full valid | Δ full | v4 bulk | small-head bulk | Δ bulk |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.146420 | 0.145334 | −0.001086 | 0.094721 | 0.086754 | **−0.007967** |
| 1 | 0.149332 | 0.138059 | −0.011273 | 0.093790 | 0.086961 | **−0.006829** |

Bulk is safely **improved** on both seeds (caution threshold +0.003). No
architecture was re-tuned on the basis of bulk.

## 15. Compute / memory reduction

`compute_audit.json`: 128-molecule forward pass, baseline 20.18 ms -> small-head
21.05 ms (**+4.3%**, within timing noise; the backbone dominates). The 17.57%
parameter reduction is **not** a compute reduction and is not reported as one.
Training wall clock ≈ 1,028 s / 895 s for seed0 / seed1. Parameter compression
and actual compute compression are reported separately.

## 16. Seed1 conditional replication

Purchased only because `D0 <= +0.002`. Seed1 used the **exact** frozen protocol
with `seed = 1`; shared initial tensors exact-match canonical baseline seed1
(40 tensors, max diff 0.0); the seed1 small head is initialised independently
under `head_seed = 0`; seed0 weights are not reused.

## 17. Two-seed result

| seed | baseline valid | small-head valid | D | best epoch | params |
|---|---:|---:|---:|---:|---:|
| 0 | 0.146420 | 0.145334 | −0.001086 | 176 | 82,115 |
| 1 | 0.149332 | 0.138059 | −0.011273 | 143 | 82,115 |

* both `D_s <= +0.002` -> **per-seed pass**;
* `mean(D0, D1) = −0.006180 <= +0.0015` -> **mean pass**;
* both `D_s < 0` -> favourable point estimates.

Verdict: **`REPLICATED_NON_INFERIOR_COMPRESSION_WITH_FAVORABLE_POINT_ESTIMATES`**.
Two development seeds are **not** enough to claim the small head is a
performance improvement; the primary claim remains compression non-inferiority.

## 18. Representation drift (descriptive)

`representation_drift.json` (seed0 selected checkpoints, official valid,
n=1000): normalized L2 drift mean **1.294**, mean cosine **0.172**; block norms
shift (unary 36.4 -> 48.6, pair 62.1 -> 51.5, global 6.29 -> 3.24, topology
2.27 -> 1.83). This is **expected and not a gate**: because the head co-trains,
the small head induces a different learned representation. The claim is
end-to-end parameter compression, not representation preservation. The
historical optimized-v4 training itself already drifted (normalized L2 0.314,
cosine 0.950) relative to the historical protocol.

## 19. What is and is not proven

**Proven (within seeds 0/1, official train/valid only):**

* a from-scratch compact-v4 model with a fixed 4,135-param raw small head
  (82,115 total) is validation-non-inferior to the 99,613-param optimized
  baseline on both development seeds, with favourable point estimates;
* the released 17,498 parameters are genuinely redundant *end to end*, not
  merely redundant for a frozen readout;
* all non-head representation-forming modules are structurally unchanged and
  start from exactly matching initial tensors;
* pre-head `R` is bit-identical at initialization.

**Not proven / not claimed:**

* that the small head is a *superior* architecture (two seeds; not a
  performance GO);
* that official test is non-inferior (test never loaded);
* that the freed 17,498 parameters should be reallocated;
* that the frozen-head donor evidence generalises to other tokenizers /
  representations / datasets.

## 20. Implication for future baseline

If retained, the new architecture core becomes **compact-v4-smallhead, 82,115
parameters** (backbone identical, head 4,135). It is a cleaner and cheaper
baseline. It is *not* a licence to immediately spend the released budget: a
future, independently pre-registered, R-SUPPORTED representation hypothesis may
use part of the freed budget while staying under the original ~100K ceiling.

## 21. Why released parameters were not reallocated

Hard rule of this stage: compression only. The 17,498 freed parameters were not
given to the patch encoder, pair encoder, embeddings, centre update or topology
branch, and no new module was added. Reallocation would mix two hypotheses
(compression and reallocation) and re-open closed branches.

## 22. Official-test lock

Official test was never loaded at any stage. No test MAE, test prediction, test
subgroup, test-selected checkpoint or test-based decision exists. Even after the
two-seed compression success, a final benchmark would require a separate,
explicitly locked stage.

## 23. Final verdict

**`REPLICATED_NON_INFERIOR_COMPRESSION_WITH_FAVORABLE_POINT_ESTIMATES`.**

The optimized compact-v4 graph representation does not require the 21.6K
graph head for successful end-to-end learning. A fixed 4.1K raw small head
removes 17,498 parameters (17.57% whole model, 80.89% head) while remaining
validation-non-inferior on both seeds. Official test remains locked; the
freed parameters remain unused.

---

## Q1–Q18

* **Q1** baseline total **99,613**; baseline head **21,633**.
* **Q2** `302 -> 13 -> 13 -> 1`, raw `R`, direct `yhat=f(R)`, ReLU, no
  LayerNorm, no dropout, no residual.
* **Q3** small head **4,135**.
* **Q4** total **82,115**.
* **Q5** −17,498 absolute; **17.566%** of the whole model.
* **Q6** **80.886%** of the head.
* **Q7** yes -- 40 shared tensors, max abs diff 0.0, hash match true.
* **Q8** seed0 best valid **0.145334** at epoch **176**.
* **Q9** `D0 = −0.001086`.
* **Q10** yes, `D0 <= +0.002`.
* **Q11** no optimization pathology (no NaN, gradients alive, best epoch not
  boundary-pinned).
* **Q12** forward 20.18 -> 21.05 ms (+4.3%); no assumed speedup; training
  1,028 s / 895 s; 82,115 vs 99,613 params.
* **Q13** yes, seed1 purchased (D0 passed).
* **Q14** seed1 small-head valid **0.138059**.
* **Q15** `D1 = −0.011273`.
* **Q16** mean degradation `−0.006180` (a favourable point estimate, i.e. mean
  improvement).
* **Q17** yes -- replicated non-inferiority.
* **Q18** **`REPLICATED_NON_INFERIOR_COMPRESSION_WITH_FAVORABLE_POINT_ESTIMATES`**.
