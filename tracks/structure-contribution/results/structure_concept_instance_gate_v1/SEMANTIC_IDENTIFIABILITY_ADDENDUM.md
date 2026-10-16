# SEMANTIC_IDENTIFIABILITY_ADDENDUM

**Round:** `structure-concept-instance-gate-v1` — read-only audit of the
semantic evidence. **Date:** 2026-10-16.
**Nature:** no training, no GPU, no molecule labels, no changes to
`REPORT.md`, `PROTOCOL.md`, `metrics.json` or any checkpoint. The frozen
verdict (`PREDICTION_ONLY_NO_GROUNDING`) is **not** re-adjudicated here; this
document corrects the *interpretation* of the semantic metrics and adds a
role-level evaluation from the frozen artifacts.

**Inputs (all frozen):** `analysis/concepts_{C,D}_seed{0,1}.jsonl`,
`analysis/metrics_seed{0,1}.json`, `predictions/predictions_seed{0,1}.jsonl`,
and the deterministic generator (`code/synthetic.py#all_graphs`).
**New output:** `analysis/semantic_identifiability_audit.json`, produced by
`code/semantic_identifiability_audit.py` (CPU, ~2.5 s, no model forward
training).

---

## 0. Corrections / retractions of earlier claims

1. **Retract:** "the frozen `key_sap_purity ≥ 0.80` sub-condition is
   mathematically unattainable because there are 190 key SAPs." Wrong.
   `key_sap_purity()` never matches SAPs to concepts. It filters patches whose
   SAP string is in the key set and returns the share of those patches
   assigned to the single most frequent concept. A constant assignment scores
   **dominant_share = 1.0** (verified with the real function on the frozen
   records); a perfect exact-SAP-identity model scores **0.118**; the trained
   models score 0.209 (D seed 0), 0.228 (D seed 1), 0.308/0.306 (C). The
   metric rewards concentration and has no recall term, so it cannot certify
   SAP fidelity and should not be used as a semantic gate without an
   anti-collapse companion.
2. **Retract:** "the best possible matched purity of a 16-prototype codebook
   is 0.863 (top-16 test-SAP coverage)." The 0.863 figure was computed with
   test-support peeking. Selecting the 16 SAPs on **train** support and
   evaluating with the frozen function gives **0.814** on test. The frozen
   0.80 threshold was therefore attainable by an exact-SAP-faithful K=16
   codebook, but it selects *toward exact-SAP fidelity*, not toward role
   abstraction.
3. **Contextualise, do not count as evidence:** `cross_base_consistency = 1.0`
   and `I(concept; label | SAP) = 0` are deterministic consequences of
   `concept = argmax(encoder(feature))` and `feature = f(SAP)`. Verified: all
   215 SAPs in the four concept files map to a single concept (0 exceptions
   in C/D × seeds 0/1). The conditional-MI estimator is a per-SAP weighted
   average, so it is zero by construction whenever a concept is constant
   inside a SAP. Neither number is independent evidence of cross-base
   semantic generalisation.
4. **Weak evidence, not valid evidence:** the relation "explanation recovery"
   rank. For both C seeds every pair-ablation sensitivity is 0.0 (C has no
   pair path), so its "true pair rank" is arbitrary; for D seed 0 the most
   sensitive pair is a **self-pair** `(7,7)` (sensitivity 1.42) versus the
   true N–O pair `(5,10)` (0.47); for D seed 1 the true pair is rank 1 but the
   effect size is tiny (0.081) and self-pairs are comparable. A rank ≤ 5 is
   therefore **not** evidence that the model recovered the *instance*
   relation.
5. **Correct the composition wording.** Composition has 0 train graphs but
   **4 dev composition graphs** (25 % of the 16-graph dev metric) that drive
   early stopping and best-checkpoint selection in `train_arm`. D seed 0
   scores 0.75 on dev composition and 0.5 on test composition; the test result
   is therefore "no training exposure, but selection-exposed", not a pure
   zero-shot confirmation.
6. **Metric-capacity note.** The Hungarian `matched_purity` requires a
   near-injective concept ↔ exact-SAP assignment: a *perfect* structural-role
   abstraction scores **0.654** under it (vs 1.0 under role-level evaluation),
   while a train-chosen 16-SAP-faithful model scores **0.814**; the constant
   collapse floor is 0.556. The metric ranks exact-SAP fidelity above role
   abstraction and cannot be used to decide whether role-level semantics were
   learned.

---

## 1. Feature identifiability (what the encoder can even see)

Recomputed from the deterministic generator + frozen patch records:
215 exact SAPs, 209 distinct 35-dim features. Only **6 features merge two
SAPs** (30 patch instances, 2.0 % of 1512). Separately, **9 feature groups
occur in both ring and chain global contexts** (142 instances, 9.4 %), e.g.
`AOd2|ASd2|nCd1|nCd1;E[…(0,2),(1,3),(2,3)]` appears both on a cycle and in a
chain — a radius-1 patch cannot reveal a ring of size > 3.

Consequences for the audit: exact-SAP collisions can affect at most ~2 % of
patches; global ring membership is genuinely unobservable for ~9 % and must
not be scored as semantic learning; element identity and local closure are
fully observable (element-feature ambiguity fraction = 0.0; triangle closure
is inside the patch for 84 of 88 single-atom triangle patches, only 4 are
"triangle not visible"). The model's element-identity failure below is
therefore **not** input-limited.

---

## 2. Role-level evaluation (train-only mapping, held-out bases)

**Definitions.** For each patch computed from generator truth: each special
atom gets `(element, global_ring ∈ {triangle, ring, chain}, local_closure ∈
{triangle, local_ring, no_local_closure}, position ∈ {anchor, common,
neighbor})`. Evaluated taxonomies (all reported, none selected on test):
`role_element` (multiset of special elements, coarse statistic);
`single_atom_element` (N/O/S for patches with exactly one special atom, 500 of
1512 = 33 %); `single_atom_local` (element + locally observable closure +
position); `single_atom_global` (adds unobservable ring/chain);
`single_ring_triangle_visible` (triangle closure vs everything else).
The concept → role map is learned **only from the train split** (per concept,
majority role, deterministic tie-break), then applied unchanged to dev/test.
Both purity (accuracy/purity per concept) and per-role recall are reported.

### 2.1 Element identity: not recovered (the main negative)

| D arm, test | seed 0 | seed 1 | baseline (train-majority) | feature lookup ceiling |
| --- | --- | --- | --- | --- |
| `single_atom_element` accuracy | 0.537 | 0.495 | 0.437 | 0.789 |
| per-role recall N/O/S | 0.98 / **0.00** / 0.54 | 0.88 / **0.00** / 0.54 | – | – |
| MI(concept; element) vs MI(feature; element), nats | 0.461 vs 1.055 | 0.333 vs 1.055 | – | – |

The prototypes merge elements inside the same structural slot. Held-out
element composition of the concepts that absorb the O patches (purity =
share of the concept's test patches matching its **train-majority** element,
so near-ties appear as low purity):
D seed 0 concept 10 = N 25 + O 34 (train majority N, 26:24; purity 0.42),
concept 5 = O 34 + S 17 (train majority S, 26:24; purity 0.33),
concept 0 = N 32 + S 16 (train tie 26:26; purity 0.67);
D seed 1 concept 1 = N 20 + O 34 + S 16 (purity 0.29),
concept 3 = N 37 + O 34 (purity 0.52).
The code is partially element-informative but is **not** a set of
element-specific structural roles; the `O` recall 0 is the hard face of these
near-50/50 mixtures.

### 2.2 Local triangle closure: partially recovered

| D arm, test | seed 0 | seed 1 | baseline |
| --- | --- | --- | --- |
| `single_ring_triangle_visible` accuracy | 0.937 | 0.916 | 0.832 |
| recall triangle-closure / rest | 0.625 / 1.000 | 0.750 / 0.949 | – |
| MI(concept; label) vs MI(feature; label) | 0.341 vs 0.413 | 0.271 vs 0.413 | – |

Triangle-sensitive concepts are consistent within seed (concept purities
0.92–1.00 for the not-triangle concepts; 1.00 for concept 9 seed 0 and
concept 0 seed 1) and the triangle closure is locally visible, so this is a
real, if partial, structural-attribute role. It is exactly the signal the
binding task needs, which is why graph-level binding accuracy is 1.0 for the
bag-only arms even though element-specific prototypes are absent.

### 2.3 Local role (element + closure + position): weak, above baseline

| D arm, test | seed 0 | seed 1 | baseline |
| --- | --- | --- | --- |
| `single_atom_local` accuracy | 0.432 | 0.389 | 0.132 |
| macro recall over roles | 0.406 | 0.385 | – |
| MI(concept; role) vs MI(feature; role) | 1.430 vs 2.130 | 1.189 vs 2.130 | – |
| per-base model / baseline on held-out bases | comp 0.37/0.11, cycles 0.25–0.75/0.00–0.25, `tri_fused5` 0.60/0.10, `spiro_tail` 0.00/0.00 | comp 0.26/0.11, cycles 0.25–0.75, `tri_fused5` 0.60/0.10, `spiro_tail` 0.00/0.00 | – |

Role-level dominant-concept agreement across held-out bases is 0.93 (both
seeds) — the same role tends to receive the same prototype family — but the
per-prototype held-out purity ranges from 0.00 to 1.00 (median ≈ 0.4) and
the element axis is merged, so the above-baseline accuracy reflects
slot/position structure, not a clean role lexicon. Cross-seed lexicon
agreement remains 0.777 (frozen metric).

### 2.4 Global ring/chian identity: mostly unidentifiable

`single_atom_global` ≈ `single_atom_local` (0.521/0.432 seed 0; 0.389/0.389
seed 1) and 378 single-atom patches lie on rings of size > 3, whose closure
is not inside the patch. The ring/chain component of the "global" role must
be excluded from semantic scoring (input-limited); the remaining component is
covered by §2.2–2.3.

### 2.5 Summary diagnosis per the requested categories

* **Prototypes genuinely recover a structural role:** yes, but only the
  locally visible triangle-closure/position component, and partially.
* **Only coarse element statistics recovered:** the opposite: element
  identity is the axis that is *most* degraded (O recall 0, train-majority
  element purity 0.29–0.67 for the merging concepts), while slot/closure
  structure is better; the code is structurally coarse, not element-coarse.
* **Only prediction, no provable semantics:** close to the truth for the
  tested wording. The bag arms reach 1.0 on binding and the relation arms
  1.0 with declared relations, while no element-specific role is verifiable.
* **Unidentifiable due to insufficient input:** only for ring membership
  (§2.4) and ≤2 % SAP collisions; not a valid excuse for element identity
  (feature lookup 0.789, ambiguity 0.0).
* **No training was performed for this audit; no new evidence was fabricated.**

---

## 3. Answers

**Q1. What is wrong with / weak in the current metrics?**

* `key_sap_purity` measures concept concentration over patches whose SAP
  string is in a 190-SAP key set (474 instances per model file, not only the
  366 decisive patches), with no SAP matching and no recall. Max is 1.0 by
  constant collapse; the exact-SAP oracle scores 0.118; the previous
  "190 SAPs → 0.80 unattainable" claim is retracted.
* Hungarian `matched_purity` requires near one-to-one concept ↔ exact-SAP
  matching; a perfect role model scores 0.654, a train-chosen 16-SAP-exact
  model 0.814, a constant 0.556. It is a fidelity-to-exact-SAP metric, the
  wrong target for role-level semantics.
* The 0.863 "ceiling" was test-peeked; the honest train-selected value is
  0.814, so the G4 threshold was attainable but only for exact-SAP fidelity.
* `cross_base_consistency = 1.0` and `I(concept;label|SAP)=0` are
  deterministic by construction (215/215 SAPs → one concept); they carry no
  independent semantic information.
* The relation explanation rank is not valid evidence: C has no pair path at
  all (all sensitivities 0), and D seed 0's top pair is a self-pair with 3×
  the true pair's sensitivity.
* The composition test is not selection-blind: 4 dev composition graphs
  influence checkpoint choice; D seed 0 dev 0.75 vs test 0.5.

**Q2. Has the current model learned verifiable basic structure–attribute
roles? How strong is the evidence?**

Partially, and the evidence is asymmetric. The prototypes verifiably recover
the locally visible **triangle-closure** distinction (test accuracy
0.937/0.916 vs 0.832 baseline; triangle recall 0.625/0.75; concept purities
up to 1.0) and a weak **anchor/neighbor position** signal (local-role
accuracy 0.432/0.389 vs 0.132 baseline; 0.25–0.75 per held-out base), but
they do **not** represent **element identity**: O patches are absorbed by
concepts whose train majority is N or S, per-concept element purity is
0.29–0.52 for those concepts, MI(concept; element) is 0.33–0.46 nats versus
the 1.055 nats available in the input, and O per-role recall is 0.0. The frozen G4
numbers therefore did not measure this either way; the corrected role audit
does, and it says: **structure-slot semantics partially present,
element-level structural–attribute semantics absent, cross-seed lexicon
agreement below the frozen 0.8 bar (0.777).** The failure is objective- and
allocation-driven (16 prototypes shared across all patch types, 702/1512
C-only patches, no pressure to preserve element identity), not caused by
missing input information.

**Q3. Next step: independent semantic constraint, or continuous structure
representations and drop the forced hard bottleneck?**

The evidence supports **dropping the hard bottleneck as the semantic
carrier**. Continuous structure representations plus the explicit relation
channel already carry the predictive result (dense B reaches 1.0 on the
relation pairs; the hard D matches it only with declared relations and adds
no predictive advantage), while the discrete code demonstrably destroys
element identity even though the input contains it. If "verifiable shared
concepts" is itself the research deliverable, then the only defensible next
study is an **independent semantic constraint** (role-preserving objective
with purity/recall and anti-collapse terms and pre-registered role-level
gates) — but that is a *new question*, not a continuation of the current
objective; it should not be run with the invalid metrics above.

**Q4. Is there enough reason to buy the next training? — NO-GO.**

No-go for training under the current formulation. The blocking issues are
metric validity (now corrected here) and the objective/target decision in Q3;
another run of the same task+reconstruction VQ would produce concepts that
are again free to merge element identities, and the old gates cannot
adjudicate it. A future GO is justified only after a preregistration that
(1) states whether the target is prediction or element-specific roles,
(2) defines role-level purity **and** recall with class-frequency and
feature-lookup baselines plus an explicit identifiability accounting
(triangle closure ∈ input, ring membership ∉ input, SAP collisions ~2 %),
(3) removes the deterministic/tautological metrics from the evidence list,
and (4) fixes the explanation test to reject self-pair and zero-sensitivity
explanations. Then: one seed for execution, one confirmation — nothing more.

---

## 4. Reproduction

```bash
uv run python tracks/structure-contribution/results/structure_concept_instance_gate_v1/code/semantic_identifiability_audit.py
# -> analysis/semantic_identifiability_audit.json  (read-only w.r.t. all frozen artifacts)
```

The audit reads only frozen concept records, frozen metrics/predictions and
the deterministic generator; it trains no model and writes only the new JSON.
It does not alter the frozen verdict, the protocol, the metrics, or any
checkpoint.
