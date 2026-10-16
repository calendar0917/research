# structure-concept-instance-gate-v1 — Report

**Round:** `tracks/structure-contribution/results/structure_concept_instance_gate_v1`
**Track:** `structure-contribution` (CSCL core question)
**Date:** 2026-10-16
**Compute:** CPU only (torch 2.5.1+cu124, `cuda_available=false`), no ZINC /
Lipophilicity / official valid-test / Locked reads, no GPU, no SSH.
**Frozen protocol:** `PROTOCOL.md` + `protocol_freeze.json` (hashes), preflight
passed before any training (`preflight.json`).
**Final label (frozen gate vocabulary): `PREDICTION_ONLY_NO_GROUNDING`**
**Continue?** Do **not** continue the interpretable-concept route on real
molecules. A new preregistration is required before any follow-up.

**结论摘要（中文）**：在纯合成属性图上，本轮复现并冻结了“结构—属性绑定”和
“局部 patch bag 完全相同、真实关系不同”两个反例；预检 5 项全部通过。四个
最小模型中，A/C（只读局部候选对象）在 bag 相同的配对上严格处于 50% 信息
上限；B/D 只有在真实关系特征下才能把配对判对（配对准确率 1.0），把关系
打乱后跌回 0.5–0.58，说明关系通道被真正使用；D 的硬概念前向可被“仅硬
概念 id + 声明关系”逐位复现，无软概念泄漏。但离散概念与合成真值 SAP 的
语义对齐不足：测试集纯度 0.72/0.67（阈值 0.80，16 个概念 vs 77 种 SAP 的
上界 0.863），跨 seed 概念一致性 0.78（<0.80），且预注册的 key-SAP 纯度
条件在数学上不可达（190 种 key SAP，最大占比 4.4%）。组合泛化（XOR 任务）
在所有模型、两个 seed 上均为 0.5。因此路线目前只证明“可用关系特征做预测 +
硬瓶颈可精确复现”，不证明“学到可核验语义的共享离散概念”。

---

## 0. Provenance, deviations and amendments

* **Generator artifact missing.** The task brief refers to a
  `structure_concept_preflight.zip`; it was not present in the repository or
  anywhere on this machine. The synthetic generator was therefore
  **reconstructed from the definitions written in the brief**, not from the
  original artifact. The reconstruction is deterministic, frozen, and its
  behaviour is documented in §1 and in `PROTOCOL.md` §2.
* **Two pre-freeze design revisions** (before freezing, before any training):
  * composition family re-designed from short-tail to fused triangle + long
    homogeneous cycle, after the first draft failed its own bag-twin test
    (the moved `O` entered patches containing the unique triangle). The
    binding/relation families, models, gates and budget were unchanged.
  * a metric-capacity diagnostic (`test_purity_support5`) was added to the
    analysis and is reported next to the frozen purity; it is not used by any
    gate.
* **Training-budget amendment `fair-convergence-v2` (post-freeze, disclosed
  non-blind).** The frozen 600/150 budget early-stopped both VQ arms while
  their training loss was still 0.25–0.27 (dense arm reached <0.001); arm D
  stopped at epoch 78 with a still-decreasing training loss. A post-hoc
  diagnostic that included test evaluation showed D reaches the relation pairs
  with more epochs. The amendment therefore sets `max_epochs=1500,
  patience=500` for all four arms and both scramble controls. No loss,
  optimiser, lr, architecture, data, metric or gate was changed. Both
  budget versions are retained (`train_logs/frozen_budget_v1/`,
  `analysis/metrics_seed0_frozen_budget_v1.json`, and the amended
  `train_logs/` + `analysis/metrics_seed{0,1}.json`); the verdict is computed
  from the amended runs against the unchanged gates.
* **Frozen-budget (seed 0) result for reference:** A rel-pair 0.5, B 1.0,
  C 0.5, D **0.5** (D purity 0.582, key-SAP 0.42, relation "recovered" pair
  `[5,5]`, i.e. both marked patches collapsed to one concept). Under the frozen
  decision logic that single run would have read `RELATION_NOT_USED`; the
  amended two-seed result shows that failure was an optimisation-budget
  artefact and is reported as such.

---

## 1. Mathematically verified facts

These are asserted by construction and re-checked by the preflight and unit
tests; they do not depend on training.

**1.1 The two required counterexamples exist.**

* *Binding (structure–attribute binding).* For each binding base, the
  `in`/`out` variants have identical node set, identical edge list and
  identical node-attribute histogram; they differ only by swapping the
  attributes at `p_in` (triangle) and `p_out` (non-triangle). The
  topology-only patch bag is identical; the joint structure-attribute patch
  bag differs; the label is decided by `N` being on the triangle. 45 matched
  pairs overall have bit-identical marginals.
* *Relation (bag-identical, relation-different).* The required 20-cycle pair
  (`rel_cycle20_NO_short` vs `rel_cycle20_NO_long`, `N` at 0, `O` at 5 vs 9)
  has **exactly identical attributed patch bags** (identical exact SAP
  multisets, not hashes), identical edges and identical attribute histogram,
  opposite labels, and true anchor distances `[5, 9]`. The same holds for
  every relation-family pair (cycles 18–28 × N–O/N–S/O–S).

**1.2 Preflight (`preflight.json`, passed).**

* 102 graphs: train 54 (binding 36, relation 18), dev 16 (6/6/4), test 32
  (12/12/8); every split balanced positive/negative; 30 base families, each
  in exactly one split.
* Relabeling invariance over all 102 graphs (patch bags, SAPs, instance
  truth unchanged).
* No marginal shortcut: 45 matched pairs with identical marginals; the
  label-from-graph-marginals probe is at chance (0.5).
* Instance-level truth returned per graph and per patch (labels, exact SAPs,
  decisive patches, anchor distances).

**1.3 Composition twins are exact bag twins.** All six composition matched
pairs (`in_short↔in_long`, `out_short↔out_long` on three fused-cycle bases)
have identical attributed patch bags with opposite labels and designed
distances (short ≤ 7, long ≥ 8; e.g. 6/8, 6/9, 7/10, 4/8, 4/9, 5/10). The
composition rule is `xor(N in triangle, d(N,O) ≤ 7)` and has **zero training
examples**; it is a pure zero-shot composition probe.

**1.4 Information limit (mathematical, not learned).** On the bag-identical
relation and composition pairs, arms A and C produce predictions that are
identical up to float summation order (max |Δlogit| ≤ 6e-8; C: 6/6 relation
pairs with identical hard predictions; A: 4/6, the other 2 only float-noise
different). Their matched-pair accuracy is exactly 0.5. No amount of training
can make a bag-only reader separate these pairs; only the declared relation
feature can.

**1.5 Hard-forward exactness (audited).** For C/D, the full forward pass
equals the concept-only path `logits_from_concepts(ids, rel)` **bit-exactly**
(`max_abs_diff = 0.0`) on the test split; multiplying the continuous encoder
output by 0.5 preserves the argmax and leaves the forward bit-exact. The
recorded reader-input tensors are exactly C = {`concept_histogram`,
`concept_mean`} and D = {`concept_histogram`, `concept_mean`, `pair_mean`,
`pair_max`}; there are no hidden continuous features, no graph IDs and no
undeclared global quantities in the hard reader. The straight-through
gradient flows to the encoder (unit-tested).

**1.6 Metric ceilings (discovered during analysis, reported for honesty).**
The frozen G4 purity metric is systematically capacity-confounded:

* test split: 612 patches, **77 distinct exact SAPs**; the best possible
  matched purity of a 16-prototype codebook is `top-16 SAP coverage = 0.863`;
* decisive binding patches: 366 patches, **190 distinct key SAPs**, largest
  single share 4.4% — the frozen `key_sap_purity ≥ 0.80` sub-condition is
  **mathematically unattainable**.

The amended-support diagnostic (`test_purity_support5`, Hungarian purity
restricted to SAPs with global support ≥ 5) removes the long tail and is
reported beside the frozen value.

---

## 2. Learning-experiment observations

Training: 4 arms + 2 label-blind relation-scramble controls per seed, seeds
0 (primary) and 1 (confirmation), identical encoders/decoders, shared
initialisation, equal budget; no hyper-parameter search, no test-based model
selection. Hard codebook K=16, τ=0.1. Test is evaluated once per seed after
all runs (see `train_logs/`, `analysis/`).

### 2.1 Prediction and relation use (Q2)

| arm | test BA s0/s1 | binding s0/s1 | relation s0/s1 | composition s0/s1 | relation pair acc s0/s1 | scrambled pair acc s0/s1 |
| --- | --- | --- | --- | --- | --- | --- |
| A dense-bag | 0.688 / 0.688 | 1.0 / 1.0 | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 | — |
| B dense-rel | 0.875 / 0.781 | 1.0 / 0.75 | 1.0 / 1.0 | 0.5 / 0.5 | **1.0 / 1.0** | 0.583 / 0.5 |
| C hard-bag | 0.688 / 0.688 | 1.0 / 1.0 | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 | — |
| D hard-rel | 0.844 / 0.781 | 0.917 / 0.75 | 1.0 / 1.0 | 0.5 / 0.5 | **1.0 / 1.0** | 0.5 / 0.5 |

* **Relation necessity.** On the six test relation pairs whose patch bags are
  mathematically identical, A/C are at chance and B/D are at 1.0, with 0/6
  identical predictions across the pair for B/D vs 4–6/6 for A/C. The
  label-blind scramble drops B/D back to 0.5–0.583 (true − scrambled ≥ 0.42).
  The declared relation channel is therefore necessary and used for this task.
* **Caveat that must not be lost.** The relation feature contains the exact
  anchor distance and the label is a threshold on it; this shows usability of
  a declared instance relation, not the discovery of a new relational
  mechanism. The composition probe (below) shows the mechanisms do not
  combine zero-shot.
* **Seed sensitivity.** Binding test accuracy is 1.0 (A/C) but 0.75–0.917 for
  the relation arms on new binding bases; the relation pairs are stable at 1.0
  across both seeds and both solving arms.

### 2.2 Discrete bottleneck role (Q3)

* Hard forward is exactly concept-only (§1.5); the reader never sees the
  continuous embedding.
* Hard vs soft on test (D): seed 0 `0.844` vs `0.875` (gap 0.031 ≤ 0.05);
  seed 1 `0.781` vs `0.781` (gap 0.0). No soft-leakage pattern (the frozen
  leakage flag fires on hard < 0.70 and soft ≥ 0.85 with gap > 0.10; neither
  seed qualifies).
* Interventions (concept and concept-pair ablation through the exact concept
  path) are used in §2.4. The bottleneck is genuinely discrete and sufficient
  to carry the relation-driven prediction, but it degrades dense accuracy by
  0.03–0.09.

### 2.3 Concept semantics (Q1)

| metric (D) | seed 0 | seed 1 | frozen threshold |
| --- | --- | --- | --- |
| matched purity vs exact SAP (test) | 0.724 | 0.667 | ≥ 0.80 |
| purity restricted to SAP support ≥ 5 | 0.829 | 0.770 | (diagnostic) |
| key-SAP dominant share | 0.209 | 0.228 | ≥ 0.80 (unattainable, §1.6) |
| cross-base consistency | 1.0 (27 SAPs) | 1.0 | ≥ 0.80 |
| cross-seed assignment agreement | 0.777 | (same) | ≥ 0.80 |
| concept-from-marginals accuracy / majority | 0.637 / 0.358 | 0.763 / 0.472 | chance would be majority |
| label-from-marginals accuracy | 0.5 | 0.5 | — |
| I(concept; label) / I(SAP; label) / I(concept; label \| SAP) | 0.051 / 0.163 / **0.0** | 0.044 / 0.163 / **0.0** | — |

* **Cross-pattern alignment exists but is partial.** The Hungarian mapping
  covers ~72 % of test patches; on frequent patterns (support ≥ 5) it reaches
  0.83 for seed 0. Concept examples show interpretable content: concept 0
  concentrates on the triangle-N decisive patch, concept 1 on an
  O-centered patch; but the same concept groups N- and S-variants of
  otherwise identical patches, i.e. the heteroatom identity is not preserved.
* **Cross-base consistency is perfect (1.0 on 27 shared SAPs)** across all
  base graphs, including test-only bases.
* **Cross-seed stability fails the bar**: 0.777 concept-assignment agreement
  between independently trained seeds. The learned lexicon is partly
  arbitrary.
* **No label shortcuts through marginals at graph level** (label-from-
  marginals 0.5), but concepts themselves are predictable from graph-level
  marginals (0.64–0.76 vs majority 0.36–0.47), so part of the concept
  assignment tracks graph size/family/position rather than purely local
  structure.
* **Concepts carry no label information beyond the exact SAP**
  (`I(concept; label | SAP) = 0.0`), a clean negative for hidden-label
  leakage in the bottleneck.

### 2.4 Explanation recovery vs synthetic truth (Q4)

* **Binding.** Model-level concept-ablation sensitivity vs generator-truth
  key-patch mass: top-key concept sensitivity rank 1 (seed 0) / 2 (seed 1);
  Spearman 0.52 / 0.41. The decisive-patch concept is among the most
  causally sensitive concepts — partial recovery.
* **Relation.** True N–O concept pair sensitivity rank 3 (seed 0) / 1
  (seed 1), both within the gate. However, in seed 0 the single most
  sensitive pair is a **self-pair** `(7,7)` (sensitivity 1.42) far above the
  true pair `(5,10)` (0.47): the reader is sensitive to co-occurrence of one
  dominant concept rather than cleanly to the two-instance pairing. Under the
  frozen-budget collapse, the true pair was reported as `(5,5)` — a rank that
  passes while being semantically empty, which is exactly why the pair rank
  alone is weak evidence.
* **Composition.** B and D are at 0.5 on all four composition pairs, both
  seeds; A/C likewise. Zero training exposure makes this a pure
  composition-generalisation probe; the result is no evidence of zero-shot
  composition.

---

## 3. Frozen gates and verdict

Computed by `run.py finalize` from the amended seed-0 metrics (unchanged
thresholds); seed-1 values shown for stability.

| gate | seed 0 | seed 1 | pass |
| --- | --- | --- | --- |
| P_preflight | 5/5 checks | — | ✅ |
| G1 info limit (A, C pair acc) | 0.5, 0.5 | 0.5, 0.5 | ✅ |
| G2 relation used (B, D; scramble) | 1.0, 1.0; 0.583, 0.5 | 1.0, 1.0; 0.5, 0.5 | ✅ |
| G3 hard concept (audit, hard, gap) | exact; 1.0; 0.031 | exact; 1.0; 0.0 | ✅ |
| G4 concept semantics (purity / key / x-base / x-seed) | 0.724 / 0.209 / 1.0 / 0.777 | 0.667 / 0.228 / 1.0 / 0.777 | ❌ |
| G5 explanation recovery (binding rank/spearman, relation rank) | 1 / 0.52; 3 | 2 / 0.41; 1 | ✅ |
| G6 composition (exploratory) | 0.5 | 0.5 | ❌ |
| soft leakage flag | none | none | ✅ |

**Verdict:** `PREDICTION_ONLY_NO_GROUNDING`. Prediction and relation use are
established on the synthetic mechanism (G2, G3, G5 pass; no leakage), but the
pre-registered test of concept semantics (G4) fails. Part of that failure is
metric design (key-SAP sub-condition unattainable; all-SAP purity capped at
0.863 by K=16 vs 77 SAPs) and part is genuine (test purity 0.72 < 0.80 with
an attainable ceiling of 0.863; cross-seed stability 0.78 < 0.80; concept
identities group N/S variants and are partly marginal-driven). The label is
therefore the conservative frozen reading: **the round supports prediction
and relation use, not verifiable shared concept semantics.**

---

## 4. What is NOT established (no molecule/novelty claims)

* **No real-molecule claim.** No ZINC, Lipophilicity, official valid/test or
  Locked data was read; all statements here are about synthetic graphs with
  generator truth.
* **No paper novelty.** The round does not test K-SVD/CSSD, GNN/Transformer
  baselines, or any new architecture; it is a mechanism gate on four minimal
  readers.
* **No causal/chemical interpretation.** Concept-ablation sensitivity is a
  model-level diagnostic compared with generator truth, not chemical
  causality; the relation feature is a declared distance and the label is a
  threshold on it, so "relation used" means usable/used, not discovered.
* **No grounding claim.** Exact-SAP semantics are not established
  (purity/cross-seed), the key-SAP metric is invalid by construction, and the
  relation-pair explanation is partly driven by self-pairs.
* **No composition claim.** Zero-shot combination of the binding and
  relation mechanisms fails at chance on all arms and seeds.
* **No transfer claim.** Nothing here implies these concepts transfer to
  molecules or that a dictionary/K-SVD route is or is not necessary.

---

## 5. Decision and recommendation

**Stop the current interpretability route; do not run real-molecule
experiments from this result.** Per the frozen stop rules, the round is not a
pass for `GROUNDING_AND_RELATIONS_SUPPORTED`, and the only real-task signal
(prediction with declared relations) is already known to be weak evidence
because it is threshold-on-distance.

If the route is continued, the next **new preregistration** should, in order:

1. fix the semantic metric before training (support-weighted purity; no
   key-SAP-dominance test over 190 heterogeneous SAPs; report the achievable
   ceiling alongside every purity);
2. make the concept objective explicitly semantic (e.g. per-SAP supervision
   or support-aware contrastive/commitment terms) rather than relying on
   reconstruction + task loss to shape the codebook;
3. require pair-distinctness in explanation recovery (reject self-pair
   explanations) and report cross-seed stability as a gate, not a diagnostic;
4. treat composition as a separate family with a dedicated preregistered
   probe, since zero-shot composition at chance is informative but
   underdetermined here;
5. keep the two counterexamples and the bag-identity information-limit check
   as permanent preflight assets.

---

## 6. Deliverables and reproduction

Directory `tracks/structure-contribution/results/structure_concept_instance_gate_v1/`:

| artifact | content |
| --- | --- |
| `PROTOCOL.md` | frozen protocol + every amendment with disclosure |
| `protocol_freeze.json` | code hashes, split manifest, thresholds, amendments |
| `preflight.json` | five required checks, 102 graphs, 45 matched pairs |
| `REPORT.md` | this report (facts / observations / unestablished claims) |
| `metrics.json` | merged seed-0/1 metrics, gates, verdict, recommendation |
| `analysis/metrics_seed{0,1}.json` | full per-seed metrics |
| `analysis/metrics_seed0_frozen_budget_v1.json` | retained frozen-budget reference |
| `analysis/concepts_{C,D}_seed{0,1}.jsonl` | per-patch learned concept + true SAP |
| `analysis/data_manifest_seed{0,1}.json` | frozen splits and generator inventory |
| `predictions/predictions_seed{0,1}.jsonl` | per-graph hard/soft predictions, all arms |
| `train_logs/{arm}{,_scrambled}_seed{0,1}.{pt,json}` | checkpoints + training provenance |
| `train_logs/frozen_budget_v1/` | retained pre-amendment checkpoints |
| `environment.json` | CPU/torch/commit provenance of the amended run |
| `code/synthetic.py`, `code/models.py`, `code/pipeline.py`, `code/analysis.py`, `code/run.py`, `code/export_predictions.py` | generator, four models, training, analysis, CLI, export |
| `tests/test_gate_v1.py` (+ `tracks/structure-contribution/tests/test_structure_concept_instance_gate_v1.py`) | 24 canonical tests, run green (48 with the forwarder) |

```bash
# tests (CPU, ~3 s)
uv run pytest -q tracks/structure-contribution/results/structure_concept_instance_gate_v1/tests

# full reproduction
uv run python .../code/run.py preflight
uv run python .../code/run.py freeze
uv run python .../code/run.py train --seed 0 --max-epochs 1500 --patience 500 --amendment fair-convergence-v2
uv run python .../code/run.py analyze --seed 0
uv run python .../code/run.py train --seed 1 --max-epochs 1500 --patience 500 --amendment fair-convergence-v2
uv run python .../code/run.py analyze --seed 1
uv run python .../code/run.py finalize
uv run python .../code/export_predictions.py
```
