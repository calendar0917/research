# Optimized-Baseline Critical Revalidation Audit

> **Question.** After the historical 60-epoch protocol was shown to be
> under-trained (`decision-compact-v4-training-sufficiency-20260917`, optimized
> compact-v4-hinge valid **0.147201 ± 0.003697**), which *historical
> architecture conclusions still survive when they are re-checked against the
> stronger optimized baseline?*
>
> **Verdict.** Two old claims were reopened (only the two that are both central
> to the method and plausibly sensitive to the training horizon):
>
> * **Part A -- TOPOLOGY STRONGLY SURVIVES SEED0.** Optimized compact-v2 valid
>   **0.166266** vs optimized compact-v4-hinge **0.146420**,
>   **Δtopo = +0.019846** (≥ +0.006, and *larger* than the historical
>   +0.014093). The global topology hinge gain is **not** a historical
>   under-training artifact; keep topology as core architecture. Seed1 not
>   purchased.
> * **Part B -- CORRECTED TOKENIZER CLEAR NO-GO.** Optimized corrected-token v4
>   valid **0.150849** vs optimized historical-token v4 **0.146420**,
>   corrected is still worse by **+0.004429** (≥ +0.004). Longer training
>   *partially* relieves the historical corrected degradation
>   (+0.008093 → +0.004429) but does **not** rescue it below the no-go line.
>   Seed1 not purchased.
>
> Exactly **2 new full-training runs** were spent (one per branch, seed 0).
> **Official test was never loaded** in this stage.

Module: `experiments/luyin16/zinc_optimized_baseline_critical_revalidation.py`
Results: `results/optimized_baseline_critical_revalidation/`
Tests: `tests/test_optimized_baseline_critical_revalidation.py` (10 tests)
Configs: `configs/luyin16/zinc_optimized_v2_historical.yaml`,
`configs/luyin16/zinc_optimized_corrected_v4.yaml`

---

## 1. Motivation

Every architecture / representation / statistic route in the track terminated
NO-GO, and then the compact-v4 training-sufficiency audit showed the *entire
historical comparison baseline was under-trained*: the canonical 60-epoch
protocol reached valid 0.170066 on seed 0, while the unchanged regime at 240
epochs reached 0.146420. Every historical NO-GO was therefore obtained against
a weaker baseline and on a **not-yet-optimised representation manifold**
(normalized R drift 0.314, cosine 0.950).

The natural temptation is to re-run all of history. The rule adopted here
(and in the task prompt) is the opposite:

> Do not rerun history indiscriminately after discovering an under-trained
> baseline; revalidate only the historical conclusions that are **central to
> the method** and **plausibly sensitive to the training horizon**.

## 2. Historical undertraining correction

The frozen optimized protocol (single stage, from scratch):

```
optimizer      = Adam
lr             = 1e-3
weight_decay   = 1e-5
batch_size     = 128
max_epochs     = 240
patience       = 40
scheduler      = none
loss           = L1 / MAE
checkpoint     = best official-valid
```

Optimized compact-v4-hinge (historical tokenizer) reference, reused here
without retraining: seed 0 valid **0.146420** (best epoch 169), seed 1 valid
**0.149332** (best epoch 104); 4-seed optimized valid **0.147201 ± 0.003697**.

## 3. Why only two old claims are reopened

* **Question A -- global topology hinge.** The topology channel is the main
  architecture GO of the compact family (historical +0.0092 paired test, 4/4
  seeds). If optimized-v2 catches optimized-v4, the historical topology GO
  must be reinterpreted. This determines whether compact-v4's main
  architectural gain is real.
* **Question B -- corrected typed tokenizer.** The historical tokenizer was
  later shown **not** to be a strict typed-isomorphism canonical key; the
  corrected tokenizer (~6.8k → ~15.2k types, more sparsity/OOV) performed
  slightly worse under 60 epochs. Because correctness and training difficulty
  were previously entangled, the corrected variant gets exactly one optimized
  seed.

Everything else (radius-3, local ring conditioning, v5 quantile, v6
factorization, endpoint association, centre covariance, triadic binding, FM,
head-family, 277k) is **not** re-run: a stronger baseline should *reduce*
unnecessary experimentation, not trigger a full historical rerun.

## 4. Compute-budget policy

* Default budget: **2 new full-training runs** — `optimized-v2 seed0` and
  `optimized-corrected-v4 seed0`.
* seed1 is bought **only** when the seed0 decision gate creates genuine
  decision uncertainty (or threatens a core claim).
* seed2 / seed3 are **forbidden** in this exploratory revalidation stage.
* Hard cap: 4 new runs (seed0+seed1 per branch) if both gates fire. Only 2
  were spent.
* **Official test is never loaded.** Only official train + valid are used for
  training, checkpoint selection and every research decision.
* Runs execute **serially** (one training process at a time), deterministic CPU
  policy (`torch.set_num_threads(4)`, fixed global torch RNG, environment
  fingerprint recorded).

## 5. Optimized training protocol (inherited, not re-tuned)

All new variants inherit the frozen optimized protocol exactly (§2). The only
variables that move are the ones under test:

* Part A: the global topology channel (`topology_mode = none` vs `hinge`) —
  **the** architecture variable.
* Part B: the typed tokenizer version (`typed_tokenizer_v1_historical` vs
  `typed_tokenizer_v2_corrected`).

Part A additionally pins the tokenizer version **explicitly**
(`typed_tokenizer_v1_historical`) instead of relying on the module default; the
cache fingerprint records the version, and a cache written with a different
version is refused. All other compact-v2 architecture leaves (patch
definition, descriptor widths, embedding dims/rank, pair path, centre update,
pooling, global branch, graph head, parameter-budget rules) are byte-for-byte
the historical definitions.

If a variant's best epoch had pinned the 240-epoch horizon, the audit would
record `horizon_boundary_warning = true` rather than auto-extending the
horizon (this is not an HPO stage). Both new runs early-stopped well inside the
horizon (best epochs 166 and 146), so no boundary warning fired.

## 6. Part A — v2 vs v4

Metric: `Δtopo = MAE_valid(v2) − MAE_valid(v4)` (positive ⇒ v4 topology better).

| seed | optimized v2 valid | optimized v4 valid | Δtopo = v2 − v4 |
| ---: | -----------------: | -----------------: | --------------: |
| 0 | 0.166266 | 0.146420 | **+0.019846** |

Only seed 0 was run (A-STRONG gate met).

* optimized-v2 best epoch **166** (206 epochs run, early stop), 98,549 params.
* optimized-v4 reference best epoch 169, 99,613 params.
* Historical 60-epoch context: v2 0.184158 / v4 0.170066 ⇒ Δtopo +0.014093.
  Fair optimization *increases* the topology advantage to +0.019846.

Decision gate: `Δtopo ≥ +0.006` ⇒ **TOPOLOGY ADVANTAGE STRONGLY SURVIVES SEED0**.
No seed1 purchased (the topology claim already had 4/4 historical directional
gains; seed 0 was enough to show under-training did not eat the gain).

## 7. Topology interpretation

Both architectures improve substantially under fair optimization
(optimized-v2 gains +0.017892 over its 60-epoch value; optimized-v4 gains
+0.023645). The **gap widens** rather than closes. Therefore:

> The compact-v4 global topology hinge is a genuine architecture component, not
> an artifact of an under-trained baseline. **KEEP TOPOLOGY AS CORE
> ARCHITECTURE COMPONENT.**

Because the A-STRONG branch fired, seed1 is deliberately reserved for a future
paper-core confirmation rather than spent here.

## 8. Part B — historical vs corrected tokenizer

Metric: `Δcorrect = MAE_valid(historical) − MAE_valid(corrected)`
(positive ⇒ corrected better); `corrected − historical > 0` ⇒ historical better.

| seed | historical-token v4 | corrected-token v4 | Δcorrect = hist − corrected |
| ---: | ------------------: | -----------------: | --------------------------: |
| 0 | 0.146420 | 0.150849 | **−0.004429** |

Only seed 0 was run (CLEAR-NO-GO gate met).

* optimized corrected-v4 best epoch **146** (186 epochs run, early stop),
  107,201 params.
* historical-token optimized-v4 reference: 99,613 params.
* Parameter budget reported truthfully: corrected vocab adds embedding rows
  (typed vocab 8,193 vs 6,785; parent 513 vs 32), giving **+7,588 params**. No
  rank/dim was compressed to force a 100k match.

Decision gate: corrected is worse by `+0.004429 ≥ +0.004` ⇒ **CORRECTED CLEAR
NO-GO**. No seed1 purchased.

## 9. Correctness/performance distinction

Longer training substantially helps the corrected representation itself
(0.178158 → 0.150849, a +0.027309 gain), so under-training accounted for
roughly half of the historical corrected-vs-historical degradation
(+0.008093 → +0.004429). But it does **not** rescue corrected below the
historical token; the degradation survives fair optimization at +0.004429.

Independently of Part B performance, the correctness fact is preserved:

> the historical tokenizer is **not** a complete typed canonical invariant key.
> If the final performance model keeps using it, the paper must not call it an
> "exact typed patch token"; use an accurate label such as **historical aliased
> rooted-topology token**.

The default result is therefore: keep the historical aliased representation for
the performance model (with accurate terminology), retain the corrected
tokenizer as the semantically-correct method, and record a
**PERFORMANCE/CORRECTNESS TRADEOFF REMAINS**, rather than rolling back
correctness.

## 10. Which historical NO-GOs remain untouched

Not re-run in this stage (and not invalidated by it):

* radius-3 patch path
* local ring conditioning / v3 context conditioning
* compact-v5 multi-quantile objective
* compact-v6 topology–attribute factorization
* pair endpoint association witness
* centre-incidence co-occurrence / covariance witness
* triadic relation binding witness
* function-basis accessibility / FM head family
* graph-head refit family
* 277k large model

These remain **valid for the historical regime** but were obtained on the
under-trained manifold; they are framed as "describe the not-yet-optimised
manifold" unless and until revalidated.

## 11. What requires future revalidation

The optimized representation drifts materially, so the *decisive representation
diagnostics* (currently all NO-GO) describe the historical manifold. Before any
new architecture family, a **cheap broad frozen-state diagnostic** on the
already-existing optimized-v4 seed0 checkpoint is the right next step; only if
that broad gate turns clearly positive is it worth building the expensive
optimized OOF models. The 277k model is only worth re-baselining if the paper
needs a strict parameter-efficiency claim.

## 12. Compute cost

`compute_accounting.csv` / `compute_accounting.json`.

| variant | seed | topo | tokenizer | epochs | optimizer steps | wall (s) | params |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| v2_hist | 0 | none | historical | 206 | 16,274 | 992.2 | 98,549 |
| v4_corr | 0 | hinge | corrected | 186 | 14,694 | 914.3 | 107,201 |

Total: **2 new full-training runs**, 30,968 optimizer steps, ~1,906 s wall on 4
CPU threads. Peak memory was not re-profiled (no extra runs). Reference
optimized-v4 checkpoints were reused, not retrained.

## 13. Final updated evidence map

| claim | historical (60 ep) | optimized (240 ep) | verdict |
| --- | ---: | ---: | --- |
| v2 vs v4 topology (Δtopo) | +0.014093 (4 seeds, historical protocol) | +0.019846 (seed0) | **TOPOLOGY GO SURVIVES (stronger)** |
| corrected vs historical token (degradation) | +0.008093 (4 seeds) | +0.004429 (seed0) | **CORRECTED NO-GO SURVIVES (relieved, not rescued)** |
| optimized compact-v4 baseline | 0.170066 | 0.146420 | new comparison baseline |
| optimized v2 baseline | 0.184158 | 0.166266 | — |
| optimized corrected-v4 baseline | 0.178158 | 0.150849 | — |

## 14. Next research recommendation

**Do the cheap broad optimized-manifold frozen-state diagnostic next** (using
the existing optimized-v4 seed0 checkpoint), before any new architecture stage,
277k rebaseline, or expensive optimized OOF models.

Reasoning:

* Part A confirms topology is real architecture value, so the *next* question is
  no longer "is topology real?" but "what does the optimized representation
  still miss?" — which is precisely what the historical NO-GO diagnostics
  measured, but on the wrong manifold.
* This step is cheap (frozen state, no training), and it is the pre-registered
  gate that decides whether the expensive optimized OOF reconstruction is
  worth building.
* A 277k fair rebaseline is only justified if the paper needs the
  compact-100k-beats-277k parameter-efficiency claim, which is not this stage's
  question.
* New architecture design should wait until the broad diagnostic says which (if
  any) channel is newly relevant under the optimized manifold.

---

## Appendix — protocol integrity (Stage 0)

Verified from durable records before training:

* optimized-v4 seed0 reference record
  `results/compact_v4_training_sufficiency/runs/Pstar_A2_long_seed0.json`
  (valid 0.146420, epoch 169, 99,613 params) and its checkpoint, commit
  `586542e1c03b8c4517a20b1ac24520f4e7a3ffed`;
* optimized-v4 tokenizer fingerprint `typed_tokenizer_v1_historical` matches the
  sufficiency `final_training_protocol_lock.json`;
* historical configs: compact-v2 valid 0.184158 / historical v4
  0.170066 / corrected v4 0.178158, all 60 epochs;
* Part A config differs from historical compact-v2 only by protocol fields +
  the explicit tokenizer pin; Part B config differs from historical
  corrected-v4 only by protocol fields.

Outputs: `protocol_lock.json`, `checkpoint_inventory.json`,
`part_a_v2_seed0.json`, `part_a_decision.json`, `part_b_corrected_seed0.json`,
`part_b_decision.json`, `final_decision.json`, `compute_accounting.csv`
(+ `compute_accounting.json`), `curves/`, `states/`, `cache/`.
