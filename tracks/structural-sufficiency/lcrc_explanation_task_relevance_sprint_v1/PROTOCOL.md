# PROTOCOL — LCRC explanation task relevance + GINE control (sprint v1)

Round / protocol ID: `lcrc-explanation-task-relevance-sprint-v1`
Directory: `tracks/structural-sufficiency/lcrc_explanation_task_relevance_sprint_v1/`

Status: **rules frozen before the first formal GPU worker**. Only observed
support counts, implementation notes, and engineering-fix notes may be appended
after freezing. No scientific rule below is changed after seeing a result.

This round implements, verifies, executes and reports a single closed loop. It
does **not** modify the LCRC architecture, add a random seed, train a new
explainer, re-optimise the previous `effect_ratio`, search new motifs, or expand
the dataset.

---

## 0. Responsibility and scientific boundary

The question is whether LCRC's automatically discovered connected structures
carry explanatory evidence that is **not merely its internal masking score**, and
whether that evidence is better than the already-trained GINE `G0` under the
same discovery procedure.

Two independent main lines:

* **A — real Lipophilicity labels**: use the official `ogbg-mollipo` split and
  the 823 already verified matched molecular pairs with their atom mappings.
  Report prediction-change quality and structural localisation against the
  real edit region.
* **B — WC_LOGP generated contributions**: use the existing WC_LOGP oracle and
  already-trained LCRC/GINE WC_LOGP checkpoints. Report prediction-change
  quality and whether discovered structures capture the known atom-contribution
  changes.

The WC_LOGP per-atom values are a **specified generating decomposition** of a
computed property. They are not a causal truth of experimental logD. The real
edit region is a structural localisation target, not a chemical causal truth.

MOLWT is retained only as the previous round's background control; no new MOLWT
training or attribution experiment is run.

---

## 1. Frozen source assets and identity

All source files are copied into this round's `analysis/` or
`results/checkpoints/` and re-hashed. Source identity is recorded in
`analysis/data_identity.json`.

| asset | source | frozen identity |
|---|---|---|
| LCRC L (Lipophilicity, recipe Rf) | `gnn_route_clarification_8h_v1` stage 1 | checkpoint SHA-256 + state hash + selected epoch |
| GINE G0 (Lipophilicity, recipe Rf) | `gnn_route_clarification_8h_v1` stage 1 | checkpoint SHA-256 + state hash + selected epoch |
| LCRC L (WC_LOGP oracle) | `gnn_route_clarification_8h_v1` oracle | checkpoint SHA-256 + state hash + selected epoch |
| GINE T (WC_LOGP oracle) | `gnn_route_clarification_8h_v1` oracle | checkpoint SHA-256 + state hash + selected epoch |
| pair structures / atom mappings | `gnn_route_clarification_8h_v1` `analysis/pair_catalog.jsonl.gz`, `row_atom_mapping.jsonl.gz` | SHA-256 |
| WC_LOGP atom contributions | `gnn_route_clarification_8h_v1` `analysis/oracle_contrib/ogbg-mollipo.npz` | SHA-256 |

Data are the official OGB `ogbg-mollipo` graph cache and raw attributes already
used by the source rounds. The split is the official scaffold split
Train/Valid/Test = 3360/420/420, verified against the OGB split CSV row indices
(shipped under `data/ogb/ogbg_mollipo/split/scaffold/`) and against the frozen
cache's `train_idx` / `valid_idx` / `test_idx`. The cache SHA-256 table is the
one in `lcrc_readout_context_pilot_small_v1/code/cr_common.py`.

If a necessary checkpoint is missing, malformed, or has a different hash/state
than recorded, the affected method stops with `CHECKPOINT_UNAVAILABLE`. It is
never replaced by random initialisation, another architecture, or a retrained
model. Other independent lines continue.

### 1.1 Frozen models

* `L` = original LCRC-C1, 228,674 parameters, `NO_BOND_MEAN`, recipe Rf,
  selected only on its own Valid RMSE.
* `G0` = original 5-layer GINE, 377,350 parameters, recipe Rf, selected only on
  its own Valid RMSE.
* `L_WC` / `T_WC` = the same two architectures trained with the same frozen
  recipe on the WC_LOGP target; `T_WC` is the GINE oracle model.

The experimental Lipophilicity models are never used as WC_LOGP models, and the
WC_LOGP models are never used as experimental-label models. Test is a
`DEVELOPMENT_TEST`: it is fully reported but was already observed in the source
round and is not an untouched final blind test.

---

## 2. P0 identity gate

Before any explanation result:

1. verify cache bytes, split sizes, split disjointness/coverage, and exact split
   row indices against the OGB split CSV;
2. verify labels by row against the frozen cache and the source-round stored
   predictions;
3. verify the 823 pair catalogue, its split strata, all row ids, and the
   identity atom mapping on every endpoint (element / aromaticity / formal
   charge / total H / bond set / stereo);
4. verify every pair's atom partition `core ∪ variable = all atoms` and that the
   stored core mapping is label- and bond-consistent;
5. verify every WC_LOGP graph contribution closes
   `sum_v c_v == WC_LOGP(molecule)` and every pair's contribution difference
   closes to `WC_LOGP(B) - WC_LOGP(A)` in the stored float precision;
6. verify each checkpoint by SHA-256, state hash, architecture/parameter count
   and strict state-dict load; replay its stored Train/Valid/Test predictions
   within the frozen raw tolerance;
7. build the edit region independently from structure only (Section 4).

P0 failure stops the affected dataset/model, not the whole round.

### 2.1 Observed support (appended after P0; rules unchanged)

The following counts are produced by `xr_prepare` from the actual catalogue and
rechecked in `analysis/pair_support.json`:

| stratum | pairs | shared-endpoint components |
|---|---:|---:|
| Train–Train | 791 | 231 total across all strata (see JSON for per-stratum) |
| Train–Valid | 5 | see JSON |
| Train–Test | 27 | see JSON |
| both endpoints non-Train (`H`) | 0 | 0 |
| all | 823 | 231 |

The `H` stratum has **NO_SUPPORT**. No non-Train–Train result is generalised.
These numbers are not hard-coded into the analysis: they are counted and the
per-stratum component counts are written to `analysis/pair_support.json`.

---

## 3. Unified automatic connected-subgraph discovery

Candidate discovery is run **independently for every molecule endpoint**. It is
never told the matched partner, the real label, the edit region, or the WC_LOGP
contribution.

### 3.1 LCRC gate

The frozen `CONDITIONAL_S1_S2_EVIDENCE_MASK` from
`lcrc_auto_subgraph_faithfulness_pilot_v1` is used unchanged:

* one gate `m_e ∈ [0,1]` per deduplicated physical bond;
* S1: the bond's contribution to both endpoint `b_sum` encodings and its S1
  neighbour-code embedding are multiplied by `m_e`;
* S2: each non-backtracking two-path occurrence `(root → middle → end)` with
  physical bonds `e1,e2` has weight `m_e1 * m_e2` on its whole contribution;
* `atom_feat`, `atom_scalar`, `global`, atom counts, encodings, and all model
  parameters are untouched;
* all-ones gates must reproduce the frozen model exactly within
  `tol_raw(x) = 1e-5 + 1e-6 * max(1,|x|)`.

### 3.2 GINE gate

For `G0` / `T_WC`, one shared gate `m_e` per physical bond multiplies the GINE
message term of that bond in **both directed copies** and in **every** original
message-passing layer. The gate is applied to the message after the GINE
`relu(h_src + edge_attr)` and before neighbourhood aggregation. No weight is
changed and nothing is retrained.

All-ones gates must reproduce the unmodified GINE prediction bit-for-bit for
the tested molecules (raw tolerance above). Finite differences use central
differences with `eps = 1e-3`; a sampled bond passes when
`|fd - grad| <= 1e-2 + 1e-2 * max(1, |fd|)`. Atom renumbering and physical-bond
renumbering must leave predictions and gradients equivalent after mapping back.

The two models' gate gradients are **not** the same unit of measurement: LCRC's
gate acts on explicit S1/S2 evidence, GINE's on its message tensor. Absolute
masked-effect values are never compared between architectures. The cross-model
comparison uses external structural localisation metrics only.

### 3.3 Frozen discovery score and search

For a molecule with physical bonds `e ∈ {0..n_bonds-1}`:

```
g_e = d(pred_raw) / d(m_e)          evaluated once at all m_e = 1
A(S) = |sum_{e in S} g_e| / sqrt(|S|)
```

Search:

* candidate bond sets are connected subgraphs of real physical bonds, size
  `k = 2..6` (size 1 is **not** a candidate, by the inherited protocol);
* `k=2`: enumerate every pair of bonds sharing an endpoint;
* `k=3..6`: from the top-12 candidates of the previous level, add one adjacent
  physical bond; keep at most beam width 12 per level;
* rank key: score descending, fewer bonds, then sorted physical-bond-id tuple;
* Top-1 is primary; Top-3 is supplementary with Jaccard ≤ 0.5 between selected
  candidates;
* the full candidate log is written; Top-3 is never re-picked after seeing a
  mask effect, real label, edit region or oracle contribution.

The same procedure, ranges, beam width, tie-break and RNG are used for LCRC and
GINE. Only the gradient field differs.

### 3.4 Local-output heuristic baseline

The existing LCRC per-atom local output `c_v = f(h_v)` is scored as

```
B(S) = sum_{v in atoms(S)} |c_v| / sqrt(|S|)
```

with the same search. It is a cheap direct comparison heuristic, not a causal
per-atom truth. It is not recomputed for GINE (GINE has no equivalent frozen
per-atom local readout); it is reported as the LCRC local baseline.

### 3.5 Random connected controls

For each molecule and its Top-1 candidate size `k`:

* generate up to 16 random connected subgraphs from the same molecule, exactly
  `k` physical bonds, excluding the Top-1 set;
* fixed RNG seed `20261117`, derived per `(molecule, control kind)`;
* deterministic growth; at most 512 attempts; accepted in generation order;
* if fewer than 8 valid controls are found, record `INSUFFICIENT_RANDOM` for
  that molecule rather than inventing controls.

The primary control is **same-size random** for both models. For LCRC only, a
supplementary control additionally matches `n_s2_affected` within 25% of the
Top-1 value. This S2-dose match is a structure-specific quantity of LCRC's gate
and is never attributed to GINE.

---

## 4. Real structural edit region

For a verified pair `(A,B)` with core mapping `π`, variable atom sets
`V_A`, `V_B`, attachment core atoms `a_A`, `a_B`:

* **Variable fragment A**: `V_A` and every bond incident to a `V_A` atom.
* **Variable fragment B**: `V_B` and every bond incident to a `V_B` atom.
* **Attachment boundary A**: `a_A` plus the bond(s) connecting `a_A` to `V_A`.
* **Attachment boundary B**: `a_B` plus the bond(s) connecting `a_B` to `V_B`.
* **Bond-type changes**: any mapped core–core bond whose bond type differs
  between A and B; both endpoints and the bond are added to both edit regions.
* **Edit region A** = variable fragment A ∪ attachment boundary A ∪ changed
  mapped bonds. **Edit region B** is defined analogously.

The edit region is determined only by structure and the verified mapping. It is
never selected by label magnitude, prediction difference, gradient, candidate
score, or WC_LOGP contribution.

Pairs whose atom partition, core mapping, attachment, or bond mapping cannot be
established uniquely and consistently are marked `NOT_EVALUABLE` with a reason.
They are not silently dropped and no favourable alternative mapping is chosen.
If a molecule has non-trivial core automorphisms, the pre-existing identity
mapping is kept and the edit/contribution quantities are checked for invariance
under the admissible automorphisms; a genuine dependence makes the pair
`NOT_EVALUABLE`.

---

## 5. Main A — real Lipophilicity labels

For every evaluable pair, record:

* official labels `y_a`, `y_b`, `Δy = y_b - y_a`;
* LCRC and GINE predictions `ŷ_a`, `ŷ_b`, `Δŷ`;
* pair error `|Δŷ - Δy|`;
* direction correctness for `|Δy| >= 0.1` (ties/zero prediction count wrong);
* per-endpoint edit region and candidate metrics (Section 6).

Full-molecule Valid/Test RMSE is recomputed from each checkpoint on the official
split and reported separately from pair change quality. The two models'
prediction accuracy is checked for comparability. Test numbers are
`DEVELOPMENT_TEST`.

### 5.1 Structural localisation metrics

For each endpoint and candidate `S`:

* candidate atoms `atoms(S)` and candidate bonds `S`;
* atom overlap, atom precision `|S_atoms ∩ E_atoms| / |S_atoms|`, atom recall
  over the edit atoms, atom Jaccard;
* bond overlap, bond precision, bond recall, bond Jaccard;
* whether `S` hits the attachment core atom, the attachment bond, or either
  (`hit_attachment_boundary`);
* candidate bond count.

The same metrics are computed for every same-size random control and for the
local heuristic Top-1. The **localisation advantage** is
`metric(Top1) - mean(metric(random same-size))` at the molecule level. The
primary localisation claim is not "the highlighted set equals the edit region";
it is whether Top-1 is enriched for the edit region relative to matched random
connected structures.

For LCRC, the `n_s2_affected`-matched random control is reported separately; it
is never described as an equal perturbation dose for GINE.

### 5.2 Prediction-change association

Report independently:

* Valid/Test RMSE for LCRC and GINE;
* `|Δŷ - Δy|` on the same pairs (component-weighted and pair-equal summaries);
* direction accuracy for `|Δy| >= 0.1`;
* localisation metrics for direction-correct vs direction-wrong pairs, and for
  prediction-accurate vs prediction-inaccurate pairs.

Pre-declared prediction strata for the descriptive split:

* `direction_correct`: `sign(Δŷ) == sign(Δy)` when `|Δy| >= 0.1`;
* `pair_accurate`: `|Δŷ - Δy| <= 0.25` (logD units), with a median split
  reported as a secondary, explicitly data-dependent descriptive cut.

Prediction and localisation are never combined into one uninterpretable total
score.

---

## 6. Main B — WC_LOGP known generated contributions

Use the same 823 pairs and the same atom mappings. The WC_LOGP contribution
space is the one in `analysis/ogbg-mollipo.npz` (`WC_LOGP_atom`, `atom_ptr`).

For mapped core atom `v` in A and `π(v)` in B:

```
δc_v = c_B(π(v)) - c_A(v)
```

For atoms only in B: `δc_v = +c_B(v)`. For atoms only in A:
`δc_v = -c_A(v)`. Every atom belongs to exactly one of these groups.

Gate: `sum(δc) == WC_LOGP(B) - WC_LOGP(A)` within
`1e-8 + 1e-8 * max(1,|Δ|)` in stored float64 precision (the source round's
closure used 1e-6 for graph contributions; the pair difference is tighter).
The number of non-closing pairs is reported, and any non-closing pair is
`NOT_EVALUABLE` for oracle localisation.

For each endpoint:

```
mass_capture(S) = sum_{v in atoms(S)} |δc_v| / sum_{v in endpoint} |δc_v|
```

only when the denominator is non-zero; otherwise the endpoint is missing and
counted. Report:

* `oracle_mass_capture(Top1)`;
* median mass capture of same-size random controls;
* `oracle_enrichment = mass_capture(Top1) - median(mass_capture(random))`;
* the same for the LCRC local heuristic;
* for LCRC and GINE oracle models, the full Valid/Test WC_LOGP RMSE, pair
  `|Δpred - Δc|`, and pair sign accuracy on non-zero `Δc`;
* the fraction of `sum |δc|` on added/deleted variable atoms vs mapped core
  atoms;
* mass capture for prediction-accurate vs prediction-inaccurate pairs, using
  `|Δpred - Δc| <= 0.1` (WC_LOGP units) as the pre-declared fixed cut and a
  median split as secondary descriptive evidence.

The known contribution is a generating-rule decomposition; a high capture is
not presented as proof of a chemical mechanism.

---

## 7. Statistics and anti-pseudo-replication

Strata are always reported separately:

1. `Train–Train`;
2. at least one endpoint non-Train;
3. both endpoints non-Train (`H`).

`H` has zero pairs in the frozen catalogue: report `NO_SUPPORT`; do not
fabricate a stratum result.

The 823 pairs are not 823 independent samples. Confidence intervals use a
**shared-endpoint connected-component bootstrap**:

* unit = connected component of molecules linked by the frozen pairs;
* fixed seed `20261117`, 2000 replicates;
* resample components with replacement, keep all pairs of a selected
  component, recompute the metric; pairs may be duplicated through their
  component;
* percentile 2.5/97.5 interval;
* report the number of independent components next to every stratum estimate;
* non-Train strata with fewer than 30 components are marked `LOW_SUPPORT`.

The same resampling is used for LCRC, GINE, random controls, local heuristic and
their paired differences. Full-split RMSE intervals use a molecule-level
bootstrap with the same seed, separately labelled.

Test is a development split in this round; it is not called an untouched final
blind test. No result is used to re-select candidates, change the search range,
or change the discovery score.

---

## 8. Execution and budget

* Host `res`, two idle A100 cards: `res-gpu0` = Main A (real labels),
  `res-gpu1` = Main B (WC_LOGP).
* One process per card; at most 2 GPU·h per worker budget (hard), soft 1 GPU·h.
* Reuse existing checkpoints. No new training.
* Molecule-by-molecule forward/backward; flush JSONL incrementally; batch only
  predictions; no all-pair molecule tensor retained in memory.
* CPU-heavy preparation, pair assembly, bootstrap, figures and report run
  locally or through a CPU-only command; GPU workers only run the frozen-model
  explanation and write one small record per endpoint.
* All remote execution goes through `rr` (`rr doctor`, `rr deploy`, `rr run`,
  `rr jobs`, `rr status`, `rr logs`, `rr pull`) with a committed revision.
* If a new training would be required for a line, do not start it; mark
  `CHECKPOINT_UNAVAILABLE` and continue the independent line.

---

## 9. P1 gates

Run and store in `analysis/model_replay.json` / gate tables:

1. **LCRC replay**: the old pilot pipeline reproduces stored candidates and
   masked effects on the pilot model; the new route L checkpoint reproduces its
   stored predictions; all-ones gated forward equals the frozen arm.
2. **GINE all-ones**: gated forward equals unmodified G0/T_WC within tolerance.
3. **GINE finite difference**: central-difference gate gradients agree on
   sampled molecules/bonds.
4. **LCRC finite difference**: same check on the S1/S2 mask.
5. **Renumbering**: atom and physical-bond renumbering equivalence for both
   models.
6. **WC_LOGP closure**: graph and pair contribution conservation.
7. **Edit region**: structure/mapping consistency, partition, attachment.
8. **Label-blind discovery**: search functions take no label/edit/oracle input;
   a shuffled-label caller test leaves candidates unchanged.

Any failed gate stops the affected method. It does not get a substitute method
under the same name. Other independent lines continue.

---

## 10. P2 formal evaluation and P3 decisions

After P1, run the two formal workers with the frozen code and candidates.
No re-selection on Valid/Test. The aggregate produces:

* `analysis/measured_pair_results.csv`
* `analysis/oracle_localization_results.csv`
* `analysis/model_comparison.csv`
* `analysis/decision.json`
* `analysis/run_manifest.json`
* `REPORT.md`, `DECISION.md`, figures.

Decision labels (multiple labels may co-occur):

* `CONTINUE_STRUCTURAL_EXPLANATION` — prediction accurate, oracle localization
  reliable, and real-label pair localization supported;
* `EXPLANATION_METHOD_OR_REPRESENTATION_BOTTLENECK` — prediction accurate but
  oracle localization not better than controls;
* `CHEMICAL_RELEVANCE_UNRESOLVED` — oracle localization reliable but real-label
  pair evidence insufficient;
* `PREDICTION_CONFOUNDED` — prediction poor enough that the localization
  comparison is materially affected;
* `NO_UNIQUE_LCRC_ADVANTAGE_SHOWN` — LCRC localises, but not better than GINE.

Pre-declared operational thresholds (chosen before the run, not tuned):

* `prediction_accurate` = both models' full Valid/Test RMSE ≤ 0.80 logD and both
  direction accuracies on `|Δy|>=0.1` ≥ 0.75;
* `oracle_localization_reliable` = for at least one model, the component
  bootstrap 95% lower bound of `oracle_enrichment` is > 0 and the point estimate
  is ≥ 0.02 mass-capture units;
* `real_label_localization_supported` = for at least one model and at least one
  of atom-Jaccard advantage, bond-recall advantage, attachment-boundary hit-rate
  advantage, the component-bootstrap 95% lower bound is > 0;
* `no_unique_lcrc_advantage` = LCRC and GINE are both evaluable, LCRC's
  localisation advantage point estimate is not clearly above GINE's (the paired
  LCRC−GINE interval includes 0 or is negative).

These thresholds are descriptive decision aids, not a new benchmark. If the
support is too low, the correct label is `CHEMICAL_RELEVANCE_UNRESOLVED` or
`NO_SUPPORT` for that stratum, not a forced positive.

---

## 11. Explicitly forbidden

* new model training, new seeds, architecture changes, new explainer heads;
* changing the candidate size range, beam width, score, tie-break or controls
  after seeing Valid/Test;
* selecting candidates by real label, edit region, or oracle contribution;
* using experimental-label predictions as WC_LOGP predictions;
* comparing LCRC and GINE gate magnitudes as one dose unit;
* re-running or extending the previous S1/S2 `effect_ratio` optimisation;
* expanding the chemical matching rule to manufacture more non-Train pairs;
* claiming the edit region is a causal chemical truth or the WC_LOGP
  decomposition is an experimental mechanism.
