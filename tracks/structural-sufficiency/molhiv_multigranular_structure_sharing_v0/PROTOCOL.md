# PROTOCOL — MolHIV 多粒度显式结构发现与预测价值研究 v0

Round id: `molhiv-multigranular-structure-sharing-v0`
Directory: `tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0/`
Seed: `0` everywhere (single seed by design). Dataset: `ogbg-molhiv`, official
scaffold Train/Valid/Test split only. No architecture search, no extra datasets.

This document pins the protocol that was **executed**. The filters, caps, C grid,
arm list, thresholds and the Phase-E trigger rule were fixed by the round's task
text before any model was fitted, and none of them were changed after seeing a
result. Two timing facts, stated up front because they matter for reading the
record:

* Phases A/B/D (CPU) ran before this file existed. They follow the task text
  verbatim; their machine-readable configs are in `analysis/*.json` and
  `configs/protocol.json`.
* This file was committed **before Phase E training started** (the only GPU
  stage), together with the frozen Phase-E input bundle.

---

## 1. Question and scope

Primary metric: official MolHIV ROC-AUC. Secondary: average precision (sklearn
`average_precision_score`). Question: how do the *discriminative granularity*,
*statistical support* and *knowledge sharing* of explicit graph structure affect
generalisation?

Explicitly out of scope: hard tokenisation / graph partitioning, KSVD, structure
response losses, new aggregation operators, extra seeds, extra structure scales,
architecture search, other datasets or tasks.

Evidence classes used in `REPORT.md`:

| class | meaning |
|---|---|
| `EXACT_MATCH` | same data, features, model, recipe, evaluation — usable as a strict control |
| `SAME_SPLIT_REFERENCE` | same split, different recipe |
| `HISTORICAL_REFERENCE` | earlier numbers, not a strict ablation |

## 2. Data and evaluation

Source: raw `ogbg-molhiv` (`data/ogb/ogbg_molhiv/raw/*.csv.gz`), official scaffold
index files, `mapping/mol.csv.gz` SMILES. Graph index = global 0-based row id.

Pre-flight audit (`analysis/data_audit.json`, produced by `msds_data.audit`):

1. splits pairwise disjoint and their union covers all 41127 graphs;
2. sizes 32901 / 4113 / 4113;
3. graph index ↔ label ↔ SMILES one-to-one; no molecule dropped;
4. all 9 raw OGB atom columns and 3 raw bond columns read;
5. physical bonds are counted once: the raw `edge.csv.gz` stores each bond once
   (mixed order), the PyG-processed view doubles it — deduplicating the doubled
   `edge_index` reproduces our bond set exactly for all 41127 graphs
   (`bonds.pyg_view_doubled_check.dedup_equals_physical_bonds = true`);
6. historical baseline identity: the frozen XGBoost baseline's Train/Valid row
   ids, and its stored per-molecule Valid/Test predictions, are row-identical to
   the official split (`historical_identity.* = true`).

Metrics: ROC-AUC computed with sklearn **and** cross-checked against
`ogb.graphproppred.Evaluator(name='ogbg-molhiv')` for every reported model
(`ogb_evaluator` blocks, `abs_diff = 0`). AP via sklearn. All Test results are
labelled `EXPOSED_DEVELOPMENT_TEST`: Test is observed every epoch by explicit
authorisation, never used for design, C selection, checkpoint choice or the
Phase-E trigger.

Train-only fitting rule: vocabulary, support weights, Morgan/count
standardisation, class weights and all model selection use Train only.

## 3. Phase A — structure discovery

Subgraph objects (each instance = the real node set **plus all internal bonds**;
instances may overlap; no tokenisation, no partition):

| family | definition |
|---|---|
| `size-2`, `size-3`, `size-4` | **every** connected induced 2/3/4-node subgraph |
| `cycle-5`, `cycle-6` | every **chordless (induced) cycle** of length 5/6 |

`cycle-5/6` is a restricted family: it is *not* all 5/6-node connected induced
subgraphs. Exhaustive 5/6-node enumeration was deliberately excluded (resource
bound); the ring families are the extra structure type this round adds. No family
was ever removed after seeing predictions.

Granularities (both keep the real internal adjacency):

* `COARSE`: atom columns `atomic_num`, `formal_charge`, `aromatic`
  (raw OGB atom columns 0, 3, 7) + `bond_type` (raw bond column 0).
* `FULL`: all 9 raw OGB atom columns + all 3 raw bond columns.

`FULL` means "identity under the complete OGB attribute tuple", not complete
physicochemical or stereochemical equivalence.

Canonicalisation (`msds_enum.canonical_key`): attribute-and-bond-aware 1-WL
refinement gives a canonically ordered colour partition; the key is the minimum,
over all node orderings consistent with that partition, of the serialised
(node attributes in order, sorted internal bonds with bond attributes). Because
the serialised object *is* a fully labelled attribute graph, the key is a
complete invariant, and the candidate set of orderings is isomorphism-invariant.

The COARSE parent `p(f)` of every FULL type `f` is obtained by **projecting the
instance's attributes and re-canonicalising at COARSE granularity** — never by
string prefixes or id arithmetic (`full_to_coarse.conflicts = 0`,
`types_without_parent = 0` in `analysis/phaseA_run.json`).

Bemis–Murcko scaffold (RDKit, `includeChirality=False`) defines scaffold groups.
The empty scaffold of acyclic molecules is **one** group (`""`), kept, never
dropped; therefore "number of scaffolds" is a support proxy and **not** a count of
statistically independent samples. 1582 molecules share the acyclic group; 7
SMILES fail scaffold perception and are kept under explicit sentinel strings.

Frozen vocabulary filters (Train statistics only, five families separately):

| granularity | min distinct Train molecules | min Train scaffolds | cap / family |
|---|---:|---:|---:|
| COARSE | 20 | 5 | 256 |
| FULL | 5 | 3 | 1024 |

FULL additionally requires its COARSE parent to be retained. Ranking: Train
document frequency desc, then Train scaffold frequency desc, then canonical key
asc. Fewer than the cap -> keep all eligible candidates. Positive/negative
support is recorded for analysis only and never selects a type.

Frozen result: 754 COARSE and 3652 FULL columns (`analysis/vocabulary.json`,
`analysis/structure_registry_*.csv`). Valid/Test were encoded only after the
vocabulary was frozen and cannot add a type.

## 4. Phase B — fixed representations, linear arms

* `M`: 2048-bit binary Morgan (radius 2, `GetMorganFingerprintAsBitVect` —
  binary, *not* count-folded) + `log1p(n_atoms)`, `log1p(n_bonds)` z-scored with
  Train mean/std. No scaling of the bit columns.
* `C`, `F`: instance counts scaled by `count / sqrt(max(n_atoms, 1))`, fixed for
  every arm.
* Invariant: for each COARSE parent, the retained FULL children satisfy
  `sum_f F_f <= C_p` per molecule (`analysis/vocab_verification.json`).

Model: one additive logit `z = b + M·w + C·u + F·v`, fitted with sklearn
`LogisticRegression(solver='liblinear', penalty='l2', class_weight='balanced',
max_iter=2000, tol=1e-6)`; a non-converged fit may be re-fitted once with
`max_iter=5000` (same objective, same C). Candidate `C ∈ {0.03, 0.3, 3.0}` shared
by all arms; every arm reports all three and selects by **highest Valid ROC-AUC,
ties to the smaller C**.

| Arm | input | regularisation / structure relation |
|---|---|---|
| B0 | M | plain L2 |
| B1 | M+C | plain L2 |
| BF | M+F | plain L2 |
| B2 | M+C+F | plain L2 |
| B2-PNULL | M+C_null+F | plain L2, shuffled parent assignment |
| B2-U | M+C+F/√mean(a) | uniform full-block rescaling |
| B3 | M+C+F/√a_f | true-support-weighted FULL penalty |
| B3-NULL | M+C+F/√a_null | within-family shuffled support weights |

`a_f = clip(sqrt(m_med / m_f), 0.25, 4.0)` with `m_f` = distinct Train scaffolds
of FULL type `f` and `m_med` = median over selected FULL types. The weighted
penalty `||w||² + ||u||² + Σ a_f v_f²` is implemented by the reparameterisation
`F'_f = F_f / sqrt(a_f)` with plain L2; the identity is unit-tested
(`tests/test_structure_core.py::test_weighted_penalty_reparameterisation_matches_direct_objective`).

`B2-PNULL`: `R_p = C_p − Σ_{f: parent(f)=p} F_f` (verified non-negative), then
parent labels of the selected FULL types are permuted **within structural family**
with `numpy default_rng(0)`, preserving the per-parent child count and `R_p`, and
`C_null_p = R_p + Σ_{f: shuffled_parent(f)=p} F_f`. Per-molecule total mass is
preserved; column totals are not, by construction. The realised change fraction is
recorded (`parent_shuffle.fraction_changed`).

`B3-NULL` permutes `a_f` within family with `default_rng(0)`.

Core contrasts, all reported on Valid and Test for AUC **and** AP, positive =
first-mentioned arm: `B1−B0`, `B2−B1`, `B2−BF`, `B2−B2-PNULL`, `B3−B2`,
`B3−B2-U`, `B3−B3-NULL`.

## 5. Phase C — structure ↔ task analyses (no new models)

1. FULL types bucketed by Train scaffold frequency (quartiles of `m_f`): type
   count, instance coverage, Valid/Test occurrence, effective |coefficient|
   distribution, and the B2 → B3 coefficient change; a check that the support
   penalty actually shrinks low-support coefficients.
2. Twelve COARSE families chosen by Train-only criteria (4 high-frequency
   multi-child, 4 with large Train prediction variation of the FULL correction,
   4 with strongly differing child support), each with real molecule matches and
   at least two different contexts.
3. Per-molecule subgroup analysis by the share of low-support / unseen FULL
   structures (Train-defined boundaries), with `INSUFFICIENT_SUPPORT` rather than
   fabricated AUC when a group lacks both classes.
4. Paired **scaffold** bootstrap (1000 resamples, seed 0, whole scaffold groups)
   of the AUC/AP differences of the main contrasts; the number of valid resamples
   is reported. This quantifies sampling uncertainty of fixed predictions only.

Logistic coefficients are read as fitted parameters of a chosen parameterisation,
never as causal chemical contributions; co-occurrence with other structures is
checked before any "this correction is about structure X" statement.

## 6. Phase D — traditional strong baseline

Historical baseline: `tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2`
(ECFP4 2048 + 217 RDKit descriptors + XGBoost, frozen config). It is **not
retrained for reporting**. The script rebuilds its feature matrix and Train-only
preprocessing from the frozen cache and must reproduce the recorded
`median_sha256`, `keep_sha256`, `n_used_columns = 2258`, `features_sha256`; a
single verification refit must then reproduce the stored per-molecule Valid/Test
predictions bit-for-bit (`historical_identity.evidence_type = EXACT_MATCH`).

One new model `XGB_STRUCT = historical 2258-column block (unchanged) + C + F`
with the historical recipe (`n_estimators=500, max_depth=5, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.8, reg_lambda=5, min_child_weight=1,
tree_method=hist, random_state=0, device=cpu`), no new descriptors, no new
tuning.

Dense input is mandatory: with a scipy-sparse DMatrix and `missing=NaN`, xgboost
treats *un-stored* zeros as missing (`DMatrix.num_nonmissing()` equals the stored
nnz, and a same-model dense-vs-sparse prediction test changes 96% of rows). The
historical recipe is dense float32 anyway, so the combined design is dense and
every explicit zero stays the value 0; after Train-median imputation no cell is
NaN. No sparse missing encoding is used.

Paired comparison against the historical per-molecule predictions on the same
rows (scaffold bootstrap as in §5.4).

## 7. Phase E — conditional plain-GINE verification

Start condition (resource trigger, not a significance test): `B1`, `B2` or `B3`
improves Valid AUC over `B0` by ≥ 0.005, **or** `XGB_STRUCT` improves Valid AUC
over the verified historical XGBoost by ≥ 0.005. Input selection: if
`XGB_STRUCT` hit the trigger use C+F; otherwise the better of `B1`/`B2` on Valid
decides C vs C+F; the extra `G-STRUCT-W` arm runs only if `B3` beats `B2`,
`B2-U` **and** `B3-NULL` with a smallest Valid gap ≥ 0.005.

Model: plain GINE, OGB atom/bond features, 3 layers, hidden 64, global mean
pooling, scalar graph logit, plus zero-initialised graph-level linear branches:

`z = z_GINE + z_M + z_structure` (branch bias = 0; branches regularised with an
explicit `0.5·lambda·||w||²`, `lambda = 1/(0.3·n_train)`, never AdamW decay).

Recipe: seed 0, 30 epochs, batch 128, AdamW lr 1e-3, backbone weight decay 1e-4,
dropout 0, class-balanced BCE (Train-derived, normalised mean weight 1), gradient
clip 5, identical backbone initialisation and batch order across arms, full
Train/Valid/Test evaluation every epoch. Checkpoint = highest Valid ROC-AUC.
`G-STRUCT-W` applies `a_f` to the FULL columns by the same `1/sqrt(a_f)` feature
reparameterisation. No attention, motif GNN, port propagation or response loss.

## 8. Budgets and stop rules

Wall clock hard limit 4 h; GPU hard limit 4 A100 GPU-hours, at most three new
neural units; CPU structure preparation ≤ 60 min; CPU linear experiments
≤ 45 min; new XGBoost ≤ 30 min. No extra seeds, no vocabulary growth, no extra
structure scales, no recipe changes driven by Test, no fallback to other
datasets. Engineering errors may be fixed, with the affected runs recorded in
`REPORT.md`.

## 9. Engineering deviations recorded in this round

| # | what happened | treatment |
|---|---|---|
| 1 | First brute-force cross-check was launched on a 222-atom molecule (`C(222,6)`) and had to be killed | reference check restricted to `n_atoms ≤ 18`; the exhaustive check is now a bounded unit test |
| 2 | Induced-cycle enumerator initially missed cycles whose root is not the closing vertex (root chord was rejected as a chord) | fixed before any statistic was produced; `analysis/phaseA_run.json` hash `4f6e70…` is the post-fix artefact, and it is reproduced bit-identically on a rerun |
| 3 | First `C_null` check compared columns instead of per-molecule mass | check corrected; column totals are documented as *not* preserved by design |
| 4 | xgboost sparse input silently treated un-stored zeros as missing | measured explicitly, Phase D switched to dense (see §6) |

## 10. Reproduction

```bash
cd tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0
PYTHONPATH=code python -m msds_phaseA --workers 10 --chunk 250   # ~80 s
PYTHONPATH=code python -m run_phaseA_vocab                        # ~10 s
PYTHONPATH=code python -m run_phaseB_linear                       # ~135 s
PYTHONPATH=code python -m build_phaseE_bundle                     # ~10 s
PYTHONPATH=code python -m run_phaseD_xgb                          # ~250 s
PYTHONPATH=code python -m run_phaseC_analysis                     # ~60 s
# GPU (through rr, committed revision):
rr run res <tag> --result <round>/runs -- python -m run_phaseE_gine --arm G-STRUCT-U --epochs 30
uv run pytest -q tracks/structural-sufficiency/molhiv_multigranular_structure_sharing_v0/tests
```
