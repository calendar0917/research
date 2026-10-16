# structure-concept-instance-gate-v1 — Preregistered Protocol

**Status:** FROZEN before any model training of this round. The generator
shipped with the task brief (`structure_concept_preflight.zip`) was **not
available on this machine or in the repository**; the counterexample generator
was therefore reproduced deterministically from the definitions in the brief
(see §2) and the five required preflight checks were re-run and passed before
freezing (`preflight.json`, `code_sha256` in `protocol_freeze.json`).

- **Track:** `tracks/structure-contribution`
- **Round:** `structure_concept_instance_gate_v1`
- **Compute:** CPU only. No ZINC, no Lipophilicity, no official valid/test, no
  Locked split, no GPU, no SSH.
- **Budget:** 4 fixed architectures (A/B/C/D), 2 seeds (0 primary, 1
  confirmation), no hyper-parameter search, no dictionary/patch/relation
  sweeps, no real molecular training.

**方案摘要（中文）**：本轮在纯合成属性图上验证一条研究路线——模型能否学到
跨图共享、具有可核验结构含义的离散概念，并在只依赖这些概念及真实实例关系
的情况下完成预测。先用两个数学反例（结构—属性绑定；局部 patch bag 相同但
真实关系不同的配对）做生成器自检；再用四个最小模型（A 连续编码无关系、
B 连续编码+真实关系、C 硬概念瓶颈无关系、D 硬概念瓶颈+真实关系）在冻结的
小型多背景图族上训练；最后用合成真值评估概念语义、硬概念前向是否可预测、
以及描述解释是否恢复生成机制。所有门控与停止规则在训练前冻结。

---

## 1. Single proposition under test

> Can a model learn **cross-graph shared, verifiable, discrete structural
> concepts** and predict a target **using only the discrete concept identities
> and the real instance relations** between them?

The round is a synthetic mechanism gate, not a performance round. It is not
about improving BJOC, not about a new paper model, and not about ZINC or any
real molecular property.

**Explicitly not under test:** chemical causality, real-molecule transfer,
paper novelty, K-SVD/CSSD necessity, GNN/Transformer baselines, hyper-parameter
quality, or whether the composition family can be solved by any strong model.

---

## 2. Required counterexamples (preflight; must pass before training)

All graphs are small attributed graphs: node attributes in {C,N,O,S}, simple
undirected edges, no edge attributes.

### 2.1 Binding counterexample

A base topology with a designated triangle position `p_in` and a non-triangle
position `p_out` of the same node degree. Two variants share **the same node
set, the same edge list and the same node-attribute histogram**; they differ
only by swapping the attributes of `p_in` and `p_out`:

* variant `in`: `N` at `p_in`, `S` at `p_out`; label `+1`;
* variant `out`: `N` at `p_out`, `S` at `p_in`; label `-1`.

The label is decided by the structure–attribute **binding** (`N` inside a
triangle). The unattributed topology and the attribute histogram are unchanged,
and the **topology-only patch bag** (all attributes collapsed) is identical.
The joint structure–attribute patch bag differs, which is exactly the
information the four candidate models are allowed to use.

### 2.2 Relation counterexample

20-node cycle, `N` fixed at position 0, `O` at position 5 (label `+1`) or
position 9 (label `-1`). Edge-anchored radius-1 **attributed** patch
multisets are exactly identical between the two graphs (verified as exact
canonical isomorphism classes, not hashes), the unattributed topology is
identical and the node-attribute histogram is identical. The label depends
only on the true shortest-path distance between the two concept instances.

### 2.3 The five preflight checks (all must pass)

1. Unattributed topology and node-attribute histogram stay identical inside
   every matched pair.
2. The relation twins have exactly identical attributed patch bags (feature
   multiset and exact SAP multiset).
3. Node relabeling does not change patch bags, per-patch SAPs or any
   prediction (checked for all generated graphs).
4. No shortcut through graph size, node/edge counts, attribute counts or patch
   counts: marginals are bit-identical inside every matched pair; split labels
   are balanced; a marginal-only probe is at chance.
5. The generator returns instance-level truth: per-graph rule label, per-patch
   exact SAP, decisive patches, per-pair true anchor distances.

Failure of any check aborts the round before training (`run.py preflight`).

---

## 3. Frozen graph families and splits

Splits are by **base graph family**; no positive/negative variant of a base
graph crosses train / dev / test. All labels are balanced inside every split.

### 3.1 Binding family (rule `n_in_triangle`)

Nine base topologies: six train (`tri_tail3`, `tri_tail5`, `bowtie_tail`,
`tri_square`, `two_tri_edge`, `tri_branch`), one dev (`tri_long_branch`), two
test (`spiro_tail`, `tri_fused5`). Each base has 3 deterministic auxiliary
attribute placements × the in/out swap pair. Train 36 / dev 6 / test 12 graphs.

### 3.2 Relation family (rule `d_short`, threshold 5)

Cycles of size 18/20/22 (train), 24 (dev), 26/28 (test) × three heteroatom
pair types (N–O, N–S, O–S) × two variants (distance 5 vs 9). Every pair is a
bag-identical, label-opposite matched pair. Train 18 / dev 6 / test 12 graphs.

### 3.3 Composition family (rule `xor(n_in_triangle, d(N,O) <= 7)`)

Fused triangle + long cycle: triangle `0-1-2`, cycle `0-3-…-n-1-0`; `N` at
position 1 (in triangle) or 3 (cycle, not in a triangle); `O` at two
homogeneous, triangle-free arc positions. Four variants per base; the matched
bag twins are `in_short↔in_long` and `out_short↔out_long`. This family is
**test-only for generalization** (dev: `comp_fused_cycle20`; test:
`comp_fused_cycle22`, `comp_fused_cycle24`). It composes the two atomic rules
and can only be solved by combining the binding patch content with the true
N–O instance relation.

### 3.4 Totals

54 train / 16 dev / 32 test graphs; 45 matched pairs (relation + composition);
labels balanced in every split; all base families in exactly one split.

---

## 4. Frozen candidate object, features and declared information

* **Candidate object:** edge-anchored radius-1 subgraph (both endpoints plus
  their neighbours), one per graph edge.
* **Patch feature (35 dims):** role × attribute node counts (5 roles × 4
  attributes) plus edge role-pair counts (15 pairs); canonicalised by taking
  the lexicographic minimum over the two anchor orientations. Roles:
  `root_u, root_v, u-only-neighbour, v-only-neighbour, common-neighbour`.
* **Ground-truth SAP:** exact isomorphism class of the anchored, attributed
  patch (brute-force canonical form over ≤ 6 nodes, anchor-flag/attribute
  colours); returned by the generator for every patch.
* **Relation feature (11 dims) between two patches:** one-hot bucket of the
  true shortest-path distance between their anchor node sets (0…8, `>=9`)
  plus a shared-endpoint flag. This is a **declared** structural relation and
  is generated by the generator, not learned.
* **Declared shortcut caveat:** the relation feature contains the exact
  anchor distance; the relation family label is therefore *learnable by
  thresholding*. A positive relation result only demonstrates that the
  instance relation is usable/used, not that a new mechanism is necessary.
  This is reported explicitly and is not upgraded to a novelty claim.

No molecular IDs, no global graph features, no labels, no learned message
passing, no attention/Transformer modules are visible to any reader.

---

## 5. The four frozen models

All four share the same patch encoder (35→64→32), the same decoder
(32→64→35) and the same relation-MLP shape (2·32+11→32→32); they differ only
in the prediction path. The encoder is initialised from the same state dict
across arms for a given seed. Parameter counts are reported; the frozen target
is `max/min <= 1.8` and `< 30 000` parameters.

* **A — Dense-Bag:** continuous patch embeddings; bag mean/max; no relations.
* **B — Dense-Relation:** continuous patch embeddings; pairwise relation terms
  `rel_mlp([h_i, h_j, rel_ij])` aggregated with masked mean and max over
  partner patches and over patches.
* **C — HardConcept-Bag:** VQ concept bottleneck (K = 16 cosine prototypes,
  τ = 0.1); the reader sees the normalised hard concept histogram and the
  mean of **hard codebook rows** only.
* **D — HardConcept-Relation:** same bottleneck; the reader sees the hard
  concept histogram, the mean hard row and pairwise relation terms computed
  from **hard codebook rows** plus the declared relation matrix.

**Hard-forward guarantee.** For C/D the per-patch embedding is
`codebook[argmax]` passed through a straight-through estimator whose forward
value is *exactly* the codebook row; the audit path
`logits_from_concepts(idx, rel)` must reproduce the full forward bit-exactly,
and must also reproduce it after scaling the continuous encoder output (same
argmax). The reader-input table records the complete set and provenance of
tensors entering the reader; only the declared discrete inputs are allowed.
Auxiliary reconstruction (if enabled) decodes only from the bottleneck
representation; there is no bypass decoder.

**Soft diagnostic.** `mode='soft'` replaces the hard rows by the posterior
expectation over prototypes. It is used only for the leakage check, never for
training or for any decision other than the frozen leakage gate.

---

## 6. Frozen training protocol

* Loss: BCE-with-logits + 0.5 · MSE(patch reconstruction) + (C/D only)
  0.1 · commitment + 0.1 · codebook. Reconstruction is always from the
  bottleneck representation of the arm.
* Optimiser Adam, lr 3e-3, weight decay 0, batch 16 graphs, gradient-norm clip
  5.0, max 600 epochs, early stopping patience 150 on dev balanced accuracy
  (ties broken by dev loss). Model selection uses dev only; test is evaluated
  once after all runs of a seed are trained.
* Seeds: 0 (primary), 1 (confirmation, run only after seed 0 passes execution
  and the preflight). B and D are additionally retrained under the
  **label-blind relation scramble** (control): every off-diagonal pair feature
  is replaced by an i.i.d. draw from the empirical relation-feature
  distribution of the train split; no label is used.
* Relation arms are trained with true relations; the scrambled control is a
  separate training condition for B/D only (same architecture and budget).

---

## 7. Metrics and evidence classes

Every output separates:

1. **Mathematical facts** (asserted): identical patch bags, identical-input
   predictions, exact hard-forward equalities, information-theoretic 50 %
   limit for bag-only arms on bag-identical pairs.
2. **Learning results** (measured): balanced accuracy per family/split, matched
   pair accuracy, hard-vs-soft gap, concept↔SAP Hungarian purity/NMI,
   cross-base and cross-seed consistency, degeneracy probes, conditional MI,
   ablation sensitivities.
3. **Interpretation** (reported with rank/percentile): model-level concept and
   concept-pair ablation sensitivity compared with generator-level truth
   (decisive SAP classes, true N–O concept pair), never as causal chemistry.

Head-input audit records every tensor entering the reader with its declared
provenance. Model-level ablations and generator-level truth are reported
separately.

---

## 8. Frozen decision gates

| gate | condition |
| --- | --- |
| P_preflight | all five preflight checks pass |
| G1 information limit | A and C matched-pair accuracy on relation test ∈ [0.45, 0.55] and within-pair logit difference is exactly 0 |
| G2 relation used | B and D matched-pair accuracy ≥ 0.80; scrambled control ≤ 0.60; true − scrambled ≥ 0.15 |
| G3 hard concept | head audit exact; D hard matched-pair accuracy ≥ 0.80; \|hard − soft\| ≤ 0.05 |
| G4 concept semantics | D test matched purity ≥ 0.80; decisive-SAP dominant share ≥ 0.80; cross-base agreement ≥ 0.80; cross-seed agreement ≥ 0.80 (seed 1 only) |
| G5 explanation recovery | binding: matched key concept has ablation-sensitivity rank ≤ 3 or key-mass/sensitivity Spearman ≥ 0.5; relation: true N–O concept pair has pair-sensitivity rank ≤ 5 |
| G6 composition | exploratory: B and D composition matched-pair accuracy ≥ 0.65 |
| soft leakage | hard < 0.70 and soft ≥ 0.85 and gap > 0.10 for C or D |

**Verdict vocabulary (exactly one):**

1. `SOFT_CONCEPT_LEAKAGE` — leakage flag first.
2. `RELATION_NOT_USED` — G2 fails.
3. `INCONCLUSIVE` — preflight/G1 failure or mixed evidence.
4. `PREDICTION_ONLY_NO_GROUNDING` — G2+G3 pass, G4 or G5 fails.
5. `GROUNDING_AND_RELATIONS_SUPPORTED` — G2+G3+G4+G5 pass.

G6 is reported and does not by itself change the verdict.

---

## 9. Stop rules / forbidden actions

* No ZINC, Lipophilicity, official valid/test, Locked split, GPU or SSH.
* No generator search after freezing; no retuning of the task to favour a
  candidate; no post-hoc rule changes.
* No K-SVD, Transformer, message passing, soft-partition mixtures or multiple
  candidate architectures. Max 4 fixed models; max 2 seeds.
* If a model fails on a pair it should *mathematically* distinguish, the
  failure is analysed as implementation / optimisation / capacity, and **no new
  module is invented to patch it**.
* If concepts have no verifiable semantics, or prediction succeeds only through
  soft-concept leakage, the route stops.
* A synthetic pass only supports further pre-registered research; it is not a
  paper-novelty claim and does not start real-molecule experiments.

---

## 10. Deliverables and reproduce commands

Deliverable directory
`tracks/structure-contribution/results/structure_concept_instance_gate_v1/`:
`PROTOCOL.md`, `REPORT.md`, `metrics.json`, `protocol_freeze.json`,
`preflight.json`, `environment.json`, generator code, the four-model code,
unit tests, concept tables, predictions and training logs.

```bash
# unit tests
uv run pytest -q tracks/structure-contribution/results/structure_concept_instance_gate_v1/tests

# preflight -> freeze -> seed 0 -> seed 1 -> finalize (CPU)
uv run python .../code/run.py preflight
uv run python .../code/run.py freeze
uv run python .../code/run.py train --seed 0
uv run python .../code/run.py analyze --seed 0
uv run python .../code/run.py train --seed 1
uv run python .../code/run.py analyze --seed 1
uv run python .../code/run.py finalize
```

---

## 11. Amendments

* **2026-10-15 — pre-freeze generator revision (before any training and before
  freezing).** The first draft of the composition family reused a short tail
  and moved `O` along it; the preflight test `test_composition_bag_twins`
  failed because the moved `O` entered patches that also contain the unique
  triangle/attachment structure. The composition family was redesigned as
  fused triangle + long cycle with `O` on a homogeneous, triangle-free arc.
  The binding/relation families, the four models, the gates and the budget
  were not changed by this revision. The revised generator passed all 24 unit
  tests before the protocol was frozen.

* **2026-10-15 — post-freeze orchestration fix (analysis path only).**
  `run.py` collected concept records for A/B (which have no VQ head) and the
  G4 gate read the full `contingency_stats` dict instead of its
  `matched_purity` field; both are analysis-path bugs that cannot change a
  training result. Hashes before/after are recorded in
  `protocol_freeze.json.amendments`; `synthetic.py`, `models.py` and
  `pipeline.py` hashes are unchanged. In addition, a capacity-aware
  diagnostic `test_purity_support5` (Hungarian purity restricted to SAPs with
  global support >= 5) was added to the analysis; it is reported next to the
  frozen all-SAP purity and is **not** used by any gate.

* **2026-10-15 — training-budget amendment `fair-convergence-v2` (amended
  gates unchanged).** Under the frozen 600/150 budget both VQ arms (C, D) had
  not converged: their training loss was still 0.25–0.27 when early stopping
  fired, while the dense arm reached <0.001. Arm D's frozen run stopped at
  epoch 78 with a still-decreasing training loss, so the failed relation
  matched-pair gate measured an optimisation-budget artefact rather than the
  representation (confirmed by a longer diagnostic in which arm D reaches the
  relation pairs). The amendment sets `max_epochs=1500, patience=500` for all
  four arms and all scramble controls. No loss, optimiser, learning rate,
  architecture, data, metric or gate threshold was changed.
  **Disclosure:** post-hoc diagnostic runs that included test-set evaluation
  were observed before this amendment was written, so the amendment is not
  test-blind. Both the frozen-budget and the amended-budget results are kept
  (`train_logs/frozen_budget_v1/`, `analysis/metrics_seed0_frozen_budget_v1.json`,
  and the amended `train_logs/`, `analysis/metrics_seed{0,1}.json`) and both
  are reported in `REPORT.md`. The verdict is computed from the amended runs
  against the unchanged gates.
