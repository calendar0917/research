# Compact-v4 Parameter Allocation & Representation Leverage Audit

**Date:** 2026-09-12
**Scope:** analysis-only / zero-full-backbone-training
**Artifacts:** `tracks/ksvd/results/parameter_allocation_representation_leverage/`
**Tests:** `tracks/ksvd/tests/test_parameter_allocation_representation_leverage.py` (15 pass)
**Official test:** never loaded.

---

## 1. Motivation

Every recent compact-v4 study added a *local* module (P1 composer, P2 refresh,
cycle cells, covariance, triad, endpoint association, attribute factorization,
FM head, function basis) and every one returned a clean or sub-threshold NO-GO.
This stage stops asking *what else can be added* and asks a different question:

> What is the current ~100K parameter budget actually **doing**, and is any of
> it reallocatable?

The parameter count conflates three different kinds of computation:

* **identity-specific storage** — token lookup tables;
* **shared structural computation** — operators applied to many structural
  objects;
* **late readout** — the graph head after the representation `R` is formed.

The audit separates them using the real instantiated optimized checkpoints
(seed0 valid 0.146420, seed1 valid 0.149332; 99,613 params; `R` = 302D), reusing
the deterministic official-train hash split (7200 fit / 800 selection / 2000
train-probe) already established by the optimized-manifold broad-state screen.

## 2. Why local architecture patching was stopped

Three consecutive local repairs failed their pre-registered gates:

* **P1** learned relation-to-centre composer: seed0 valid 0.147229 vs 0.146420
  (Δ = −0.000809).
* **compact-v4-cell** explicit persistent cycle cells: seed0 valid 0.147462
  (Δ = −0.001042).
* **P2** one-shot relation refresh: seed0 valid 0.144457 (sub-threshold
  +0.001963 at +61% compute).

Alongside the frozen-state witnesses (endpoint association, centre covariance,
triadic binding, frozen readout, broad-state) these close a large family of
"add one more local statistic/ordering" hypotheses. The remaining question is
budget-level, not patch-level.

## 3. Scope and zero-training discipline

* 0 full-training seeds. Reuses the two existing optimized checkpoints.
* No new architecture, no attention/Transformer, no CIN-like hierarchy, no
  hyperparameter search.
* Allowed: inference, instrumentation, SVD, frozen-`R` export, tiny head fit.
* `seed0` and `seed1` only; official test never loaded.
* The small-head refit is the only thing trained, and it is a *screening* device
  on a valid-selected backbone — not a clean independent compression proof.

## 4. Exact parameter ledger

Recomputed from the real instantiated model (`parameter_ledger.csv`,
`parameter_class_summary.json`), not from any earlier hard-coded number.
Sum = **99,613** exactly; tensor names are unique (no double counting).

| component | params | % total | class |
|---|---:|---:|---|
| typed_token_embedding (hybrid) | 36,420 | 36.56 | identity_storage |
| parent_token_embedding | 256 | 0.26 | identity_storage |
| patch_encoder | 14,192 | 14.25 | shared_operator |
| center_update | 15,888 | 15.95 | shared_operator |
| pair_encoder | 5,328 | 5.35 | shared_operator |
| relation_encoder | 1,360 | 1.37 | shared_operator |
| pair_projection | 768 | 0.77 | shared_operator |
| distance_gate | 80 | 0.08 | shared_operator |
| global_encoder | 3,136 | 3.15 | global_structural_prior |
| topology_encoder | 552 | 0.55 | global_structural_prior |
| graph_head | 21,633 | 21.72 | late_readout |
| **TOTAL** | **99,613** | **100** | |

Class summary: identity_storage 36,676 (36.82%), shared_operator 37,616
(37.76%), late_readout 21,633 (21.72%), global_structural_prior 3,688 (3.70%).

## 5. Storage vs computation parameters

The identity lookup is **not** a plain `V × d` table. `_HybridEmbedding` stores
768 frequent tokens as free 16-D rows and 6,017 rare tokens as shared rank-4
factorized codes (`rare.embedding` 6017×4 plus a 16×4 projection). So the
lookup is already a **frequency-adaptive** storage scheme: 12,288 + 24,068 + 64
= 36,420. Frequent/rare structure is built in, not a candidate to add.

## 6. Identity lookup utilisation

`token_frequency_rows.csv` records, per vocab row, the train occurrence count,
number of molecules containing it, relative frequency, embedding norm, row
mean/std, seed0/seed1 row cosine, and valid-input occurrence (inputs only). The
frequency counts come from official-train `typed_token` inputs only; no target
is read. `stage_frequency` / `stage_embedding_spectrum` source is asserted
target-free by test 5.

## 7. Token frequency concentration

Locked log bins (not re-fit to results):

| bin | rows | lookup params | occurrences |
|---|---:|---:|---:|
| 0 | 1 | 16 | 0 |
| 1 | 2,658 | 10,632 | 2,658 |
| 2–5 | 2,087 | 8,348 | 6,085 |
| 6–20 | 1,187 | 4,748 | 12,618 |
| 21–100 | 585 | 8,340 | 26,208 |
| >100 | 267 | 4,272 | 184,095 |

Occurrence is highly concentrated: the top **10%** of rows cover **89.0%** of
token occurrences, top 25% cover 95.3%, top 50% cover 98.2%. Conversely 80% /
90% / 95% of occurrences need only 280 / 770 / 1,616 rows. **Rare ≠ useless**:
the low-frequency rows are not automatically waste, and this audit does not
authorize frequency pruning.

## 8. Embedding spectral structure

Effective 16-D table (all 6,785 rows materialized via the factorized rare block):

| seed | stable rank | entropy rank | r95 | r99 | r999 |
|---|---:|---:|---:|---:|---:|
| 0 | 1.99 | 4.66 | 9 | 15 | 16 |
| 1 | 2.94 | 6.84 | 12 | 16 | 16 |

The **full 768×16 block** is genuinely full-rank: stable rank 11.16, entropy
rank 15.50, r99 = 16. The **rare 6,017 rows** are exactly rank ≤ 4 (r99 = 4):
already maximally factorized. The combined table reaches 99% energy only at
rank 15–16.

## 9. Frozen low-rank lookup screen

Truncated SVD rank comes **only** from 99% / 99.9% Frobenius energy
(`embedding_lowrank_accounting.csv`):

| variant | rank | original | factorized | savings | hypothetical total |
|---|---:|---:|---:|---:|---:|
| r99 | 15 | 36,420 | 102,015 | **−65,595** | 165,208 |
| r999 | 16 | 36,420 | 108,816 | **−72,396** | 172,009 |

Because the rare rows are already rank-4, a global rank-r factorization is
*larger* than the current hybrid. No positive hypothetical saving exists at the
pre-registered ranks.

Frozen perturbation (no training, non-lookup params untouched):

| seed | r99 drift mean / median / p90 / max | r99 R-drift | r99 valid Δ | r999 valid Δ |
|---|---:|---:|---:|---:|
| 0 | 0.0119 / 0.0095 / 0.0247 / 0.1079 | 0.0176 (cos 0.9998) | +0.00033 | 0.0 |
| 1 | 0.0 (r99 = 16) | 0.0 | 0.0 | 0.0 |

The "full" effective table reproduces the direct forward to 1.4e-6 (`yhat`) /
1.9e-5 (`R`), so the instrumentation is faithful. r999 is the identity here.

## 10. Graph-head parameter allocation

`graph_head` = 302 → 64 → 32 → 1 with `LayerNorm(64)` + dropout 0.05, 21,633
params (21.72%). It is the second-largest region and the largest single
*late readout* region.

## 11. Optimized-manifold small-head screen

One fixed small raw head (mechanically reused `Sraw`/`Hsmall`):
302 → 13 → 13 → 1, **4,135** params, direct `ŷ = f(R)`, **raw R**, no
LayerNorm, no residual over `ŷ₀`; L1 / Adam(lr 1e-3, wd 0) / minibatch 512 /
horizon 800 / head_seed 0 / best-selection checkpoint; fit/selection on the
official-train internal 7200/800 split; official valid evaluated once after the
head checkpoint is locked. No width/depth/activation/LayerNorm/dropout search.

| seed | H0 valid | small-head valid | Δ = small − H0 | params saved |
|---|---:|---:|---:|---:|
| 0 | 0.146420 | 0.143208 | **−0.003213** | 17,498 (80.9%) |
| 1 | 0.149332 | 0.147754 | **−0.001578** | 17,498 (80.9%) |

Mean Δ = −0.002396. Both seeds satisfy Δ ≤ +0.002, mean ≤ +0.001, reduction
≥ 60% → **HEAD DONOR — STRONGLY SUPPORTED FOR REALLOCATION SCREENING**.
This is screening evidence on a valid-selected backbone, not a proof that an
end-to-end model can remove those parameters; the historical
graph-head-refit late-adaptation result is the mechanistic reading (the
jointly-trained head was under-adapted to the final `R`).

## 12. Shared operator reuse

`operator_reuse.csv` (2000-molecule train-probe; descriptive, not a score):

| component | params | applications/molecule | object |
|---|---:|---:|---|
| patch_encoder | 14,192 | 23.05 | each patch |
| pair_projection / relation_encoder / distance_gate / pair_encoder | 768 / 1,360 / 80 / 5,328 | 264.29 | each pair |
| center_update | 15,888 | 23.05 | each centre |
| global_encoder / topology_encoder / head | 3,136 / 552 / 21,633 | 1.0 | per molecule |

`applications / parameters` is descriptive only — it distinguishes storage from
reusable computation, and is **not** an efficiency or optimality metric.

## 13. Representation bandwidth / activation rank

Participation ratio `r_PR = (Σλ)² / Σλ²` on the train-probe (seed0; seed1
similar):

| state | dim | `r_PR` | `r_PR/d` |
|---|---:|---:|---:|
| h0 (patch) | 48 | 7.56 | 0.157 |
| u (pair projection) | 16 | 2.91 | 0.182 |
| q (pair state) | 16 | 4.45 | 0.278 |
| R_unary | 97 | 6.41 | 0.066 |
| R_pair | 165 | 2.62 | 0.016 |
| R_global | 32 | 2.91 | 0.091 |
| R_topology | 8 | 1.13 | 0.142 |
| R | 302 | 3.20 | 0.011 |

The representation uses a small fraction of its nominal dimensionality. Per the
pre-registered rule this is **bandwidth evidence only**: low rank does not prove
a dimension should be reduced, high rank does not prove it should be increased.

## 14. Local gradient sensitivity

Descriptive only (`local_gradient_sensitivity.json`). Highest gradient RMS
(seed0): pair_encoder 1.9e-2 > pair_projection 1.7e-2 > head 9.3e-3 >
patch_encoder 7.4e-3; lowest: typed_embedding 2.6e-4. Per the pre-registered
caveat, gradient magnitude at a trained checkpoint is **not** a
capacity-importance metric and is not used to rank donors/recipients.

## 15. Existing positive evidence

* Optimized global topology channel: `v2` 0.166266 vs `v4` 0.146420,
  Δ = +0.019846 (A-STRONG) → **keep topology**.
* Optimized training protocol (240 ep / patience 40) is the frozen baseline.
* Historical aliased rooted-topology token remains the performance-token
  representation; the corrected tokenizer regresses (+0.004429).
* Graph-head refit late-adaptation: the original head is under-adapted.

## 16. Existing negative evidence

* P1 learned composer NO-GO; P2 one-shot refresh sub-threshold (+0.00196, +61%
  compute); compact-v4-cell NO-GO (−0.001042).
* Endpoint association, centre-incidence covariance, triadic binding, frozen
  readout, broad frozen-state, FM head, additive function basis: all NO-GO or
  INCONCLUSIVE.
* Compositional patch sharing / corrected tokenizer / attribute factorization:
  NO-GO.

## 17. Donor candidates

Only two classes were eligible: identity lookup storage and graph head.

* **Graph head — H-STRONG.** Small raw head improves valid on both seeds while
  removing 17,498 params (80.9%).
* **Identity lookup — L-NO.** Despite 36.8% of params, the lookup is already a
  frequency-adaptive hybrid (768 full + 6,017 rank-4); the full block is
  genuinely rank-16; a global rank-15/16 factorization has **negative** saving
  and r99/r999 prediction drift/valid change is negligible. There is no
  evidence-supported simple compression donor here, and rare-token pruning is
  explicitly not authorized.

## 18. Recipient candidates

Recipient must be **representation formation**, not a new head/descriptor/
statistic, and must survive the existing NO-GOs.

* patch_encoder (shared h0): **R-UNCLEAR** — shared operator, but low `h0`
  `r_PR/d` is bandwidth evidence only and no untested representation *principle*
  is isolated by internal evidence.
* pair path / centre_update: **R-DISFAVORED** — P1/P2/covariance/triad/endpoint
  NO-GOs equivalently close this mechanism space.
* higher-order structural persistence: **R-DISFAVORED** — compact-v4-cell
  NO-GO.
* topology/global prior: out of scope (positive; not representation formation).

No region reaches **R-SUPPORTED**.

## 19. Why P1/P2/cell are not being reopened

P1 closed learned pre-pool relation→centre composition; P2 closed one-shot
relation refresh (numerical staleness is real but fixing it is not valuable
enough); compact-v4-cell closed explicit persistent higher-order objects. Any
new recipient that touches pair/centre composition or cycle-object persistence
would be an equivalent re-run and is forbidden by the audit rules. This is a
hard constraint on recipient selection, not a suggestion.

## 20. Evidence matrix

`allocation_evidence_matrix.csv` lists region / params / role / reuse /
compressibility evidence / historical evidence / donor evidence / recipient
evidence / confounds for the lookup, patch encoder, pair path, centre update,
topology, global encoder and graph head. No arbitrary weighted score is used;
only qualitative labels and pre-registered gates.

## 21. Top-1 budget-reallocation hypothesis or no hypothesis

**NO ACTIONABLE REALLOCATION HYPOTHESIS** for a *performance* architecture.

`top1_reallocation_hypothesis.json` is emitted with `authorized = false` and
`no_fabricated_candidate = true`: a strongly-supported donor exists (graph
head) but no representation-forming recipient is R-SUPPORTED.

## 22. Parameter budget for next experiment

No next full-training experiment is authorized in this stage. The head donor
would free roughly 17.5K (a future head budget of ~4–8K is plausible), but
without a justified recipient the correct future action is a *compactness /
compression* experiment, not a "spend the freed parameters" experiment. If such
an experiment is ever authorized it must be budget-neutral (95K–105K, target
≈99.6K, hard ceiling 105K), seed0 only, optimized protocol, official test
locked.

## 23. What is and is not proven

**Proven (screening):**

* The ledger is exact and the identity lookup is a frequency-adaptive hybrid.
* The graph head is a strong donor candidate on two optimized backbones.
* The identity lookup is not a simple SVD-compression donor.
* The representation states are narrow in participation-ratio terms.

**Not proven:**

* That an end-to-end model with a small head is as good as compact-v4 (frozen
  screening only; backbones are valid-selected).
* That the lookup cannot be improved by a *structured* (non-post-hoc) scheme.
* That low activation rank means capacity should be reduced or increased.
* That gradient magnitude implies importance.

## 24. Final verdict

# COMPRESSION-ONLY OPPORTUNITY (Case C)

A donor exists (graph head, H-STRONG) but no scientifically justified
representation-formation recipient is supported by internal evidence. No larger
middle network is authorized. Official test remains completely locked.

---

## Q1–Q20

* **Q1** 99,613 = identity_storage 36,676 (36.82%) + shared_operator 37,616
  (37.76%) + late_readout 21,633 (21.72%) + global_structural_prior 3,688
  (3.70%).
* **Q2** identity lookup = 36,676 (36.82%): typed 36,420 + parent 256.
* **Q3** graph head = 21,633 (21.72%).
* **Q4** shared structural operators = 37,616 (37.76%): patch 14,192, pair path
  7,536, centre 15,888.
* **Q5** low-frequency rows (≤20): 1 + 2,658 + 2,087 + 1,187 = 5,933 rows.
* **Q6** lookup params serving low-frequency rows: 16 + 10,632 + 8,348 + 4,748 =
  23,744 (≈65% of the typed lookup).
* **Q7** yes, highly concentrated: top-10% rows cover 89.0% of occurrences.
* **Q8** stable rank 1.99 / 2.94 (seed0/seed1); entropy rank 4.66 / 6.84.
* **Q9** r99/r999 = 15/16 (seed0), 16/16 (seed1).
* **Q10** hypothetical saving is negative; global factorization is larger than
  the current hybrid.
* **Q11** r99 drift small/none; valid Δ +0.00033 (seed0) / 0 (seed1); r999 is
  the identity.
* **Q12** seed0 small head 0.143208 vs H0 0.146420 → Δ −0.003213 (better).
* **Q13** seed1 small head 0.147754 vs H0 0.149332 → Δ −0.001578 (better).
* **Q14** yes — H-STRONG donor.
* **Q15** no — L-NO; already frequency-adaptive, no positive saving.
* **Q16** states use a small fraction of their dimensions (h0 0.157, u 0.182,
  q 0.278, R_pair 0.016); bandwidth evidence only.
* **Q17** pair_projection / relation_encoder / distance_gate / pair_encoder
  have the highest object reuse (264.3/molecule); descriptive, not a score.
* **Q18** P1 closes learned pre-pool composition; P2 closes relation refresh;
  cell closes explicit higher-order objects; endpoint/covariance/triad close
  pair/centre statistics.
* **Q19** no clear donor+recipient pair: head donor yes, recipient no.
* **Q20** **COMPRESSION-ONLY OPPORTUNITY**.
