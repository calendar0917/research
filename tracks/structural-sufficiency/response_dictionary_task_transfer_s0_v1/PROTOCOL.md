# PROTOCOL — frozen RESP/ADD/JOINT + K=16 dictionaries on real Lipo edits (no training)

Status: **frozen before any label or frozen-model prediction is connected to the
new catalogue.** Base commit `e0a6386122c4407c1cb974b1c434bb13927030a7`.
Round directory `tracks/structural-sufficiency/response_dictionary_task_transfer_s0_v1/`.

This round does not train, fine-tune, re-select a checkpoint, change a view,
aggregate, fallback, dictionary, alpha, projector, split or edit definition.
It asks one question: with the official split fixed and every prediction
function frozen, does the existing RESP carry an extrapolable signal about
**real structural edit changes**, and how much of that signal the response
dictionary preserves.

## 1. Read-only sources and frozen objects

Read-only source rounds:

* `response_dictionary_small_s0_v1/` — Lipo RESP best-Val epoch 63 and ADD
  best-Val epoch 91; parent fragment cache, splits and fragment identity code.
* `response_dictionary_lipo_followup_s0_v1/` — JOINT best-Val epoch 85; the
  K=16 dictionary construction.
* `response_dictionary_explanation_check_s0_v1/` — HP_CERTIFIED full-`F`
  per-molecule predictions of `D_RESPONSE`, `D_LATENT`, `D_FREQ`.
* `lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv` and
  `pair_mappings.json` — the old 823-pair catalogue, used **only** for a
  structural intersection/absence report.

Frozen objects:

| object | source |
|---|---|
| RESP | seed 0, best-Val epoch 63, full saved per-molecule `F(G)` |
| ADD | seed 0, best-Val epoch 91, full saved per-molecule `F(G)` |
| JOINT | seed 0, best-Val epoch 85, full saved per-molecule `F(G)` |
| dictionaries | K=16 `D_RESPONSE` / `D_LATENT` / `D_FREQ`, **HP_CERTIFIED**: certified row where available, frozen FW64 alpha otherwise |
| data | official OGB Lipo cache `fe002287…`, fragment cache `ba6e9a41…`, official 3360/420/420 rows |
| identity | official row id; RDKit canonical isomeric SMILES is a cross-check only |

The old catalogue's own `split_a/split_b` columns are never used. Splits and
labels are re-derived from the official row id. The follow-up's 331 stored-split
mismatches are re-counted as an identity check, not reused.

Before any new number is accepted, full-molecule Train/Valid/Test RMSE/MAE of
all seven frozen functions are replayed from the saved predictions and compared
with the committed old reports to `1e-5` on Valid/Test. If this identity check
fails, the run stops; numbers are not forced to match. No old C1/GINE position
reference is added to the pair comparison and no old baseline is retrained.

## 2. One new edit catalogue (label-free)

### 2.1 Molecule scope

All 4200 real Lipo rows are read. The edit evaluation includes only single
connected components with no explicit-H node. No desalting, deduplication,
stereo removal or graph rewriting is performed. A molecule that carries
stereochemistry is allowed; **all** original OGB atom and bond columns are
carried into S, C, the ports and the cut bond.

Official row 1646 (valid) is the only multi-component molecule and is excluded
from edits. No explicit-H molecule exists. These facts are recorded explicitly;
nothing is silently degraded.

### 2.2 Cut and size rules

For every eligible molecule, enumerate **all** bridge edges of the physical OGB
bond graph. A legal cut is a bridge with OGB bond type `SINGLE` (`bond_feat[:,0]
== 0`). The bridge property supplies "non-ring"; the OGB column `bond_feat[:,2]`
is `is_conjugated`, not `is_in_ring` (see OGB
`bond_to_feature_vector`). The parent helper's docstring/comment that reads it as
ring membership is not treated as an OGB field definition. The literal parent
filter's effect is recorded as an audit count only; it is not a second
catalogue.

For each legal cut both orientations are considered. The variable side `S` must
have 1–8 heavy atoms; the complement `C` must have at least 10 heavy atoms and
at least 75% of the parent molecule's heavy atoms. A molecule can contribute
several cuts; the model's fixed ≤4 selected views are irrelevant to this
enumeration.

### 2.3 Exact core identity

`C` is induced with the parent round's `rd_fragment._induce` on the **whole**
OGB attributed graph. Its identity is the exact coloured-incidence-graph
canonical serialization: all nine atom columns, a binary port colour, the
auxiliary bond vertices with all three bond columns, and the SHA-256 of that
serialization. Two cores are equal only if that exact certificate, the port and
the three cut-bond columns are equal. WL/hash buckets are not used as identity
in this round: every candidate receives the exact certificate. RDKit
substructure or SMILES equality is never used as the match.

Each candidate is also checked by the parent `verify_reassembly`: `S + C + the
cut bond` must restore the original attributed graph, the cut bond attributes
must be unchanged, and the two ports must be the cut endpoints. A failure aborts
the catalogue; it is never silently dropped.

### 2.4 One definition per unordered molecule pair

For every unordered real-molecule pair that has at least one exact shared
`(C, port, cut-bond)` candidate, keep exactly one definition by:

```
(max core_heavy, core canonical key, sorted(S_A key, S_B key),
 sorted(cut_edge_A, cut_edge_B))
```

A pair with identical complete canonical isomeric molecules or identical `S`
native identities is not an edit. The direction is fixed by the **S native key**
(the exact parent fragment key, lexicographically smaller = A) and is
consistent across different cores; `delta = value_B - value_A`.

`transformation_id = ordered(S_A native key, S_B native key, cut-bond
type/stereo/conjugated)`.

No label or prediction is read in this stage. The catalogue files are written,
hashed and committed **before** the label/prediction connection.

### 2.5 Comparison with the old 823 pairs

The old catalogue is rebuilt from `pair_mappings.json` using the same OGB graph
and the same exact `C` certificate. Each old pair is classified as:

* `EXACT_SAME_DEFINITION` — same unordered row pair and same exact
  `(C, port, cut-bond)` identity under this round's definition;
* `PRESENT_DIFFERENT_DEFINITION` — same row pair, but the new catalogue chose a
  different exact core (larger core, or the old core is not exact in full OGB
  attributes);
* `ABSENT` — the row pair is not in the new catalogue (old core is not exact in
  full OGB attributes or the old cut does not meet this round's rule).

New pairs are reported separately as `new_rowpair`; they are stratified by
`max |S| ∈ {1–4, 5–8}` and by whether both molecules passed the old
(no `@`, `/`, `\`) stereo eligibility. Old pairs that change definition are
reported as `redefined_old_rowpair`. These strata describe provenance only; none
is an optional primary endpoint. The new catalogue is **not** forced to contain
the old one.

## 3. Independent support and strictness

Components merge rows with identical complete canonical isomeric SMILES,
endpoints of every pair, and all endpoints of pairs sharing the exact matched
`core_id`. Original rows are retained; no duplicate row is deleted. A molecule
that appears in many pairs is one unit, not many independent experiments.

For each official split combination report: pairs, distinct molecules,
distinct non-Train molecules, Test molecules, pair-bearing components, cores and
transformation ids. Fixed sets:

* `E_T` — at least one Test endpoint (Train–Test, Valid–Test, Test–Test); this
  is "edit response involving a held-out molecule", not new-core *or* both-ends
  held-out generalisation.
* `E_HH_T` — `E_T` with neither endpoint in Train.
* `E_STRICT` — `E_HH_T` where the exact matched `core_id` appears in **no legal
  Train cut** under this same definition, and the same `transformation_id`
  appears in a Train–Train pair in another component. Train-seen is computed
  from all legal Train cuts, not only paired molecules and not only the fixed
  ≤4 views. `E_STRICT_C_ONLY` uses only the C graph key (not the cut-bond
  columns) as a descriptive variant.
* `E_STRICT` may be empty; an empty set is reported as empty. No looser set
  inherits the name.

Two coverage fields are recorded but never change the evaluation subset:

* whether both endpoints have a fixed ≤4 selected view with the same exact `C`
  and cut bond (`shared_selected_core_view`);
* whether each `S` belongs to the existing dictionary's 702 Train S-word
  candidate inventory from the fixed selected Train views. This is not the set
  of all legal Train cuts.

The old catalogue's held-out support (27 Train–Test pairs, 7 pair-bearing
components, 7 Test molecules, 0 both-non-Train pairs) is the reference for
"how much was added"; the new counts use only the new components.

## 4. Frozen predictions, references and labels

For every pair and every frozen model `M`:

```
delta_M = F_M(G_B) - F_M(G_A)
delta_y = y_B - y_A
```

`F_M` is the actual complete saved molecule prediction (all fixed views or the
existing fallback). A single-view `s(C,S)` is never substituted for `F`. No
best epoch is re-selected.

Two cheap label references, no new neural network:

* `ZERO`: `delta_pred = 0`.
* `TRAIN_TRANSFORM_MEAN`: for the pair's `transformation_id`, use only
  Train–Train pairs. Average `delta_y` within each source component, then
  average over source components, **excluding the evaluated pair's entire
  component**. If no source component remains, the reference is `NA`; it is
  never filled with 0. Comparisons involving it use exactly the same supported
  pair/component subset for every model, with coverage and independent counts
  reported. No Valid/Test label enters the table, and there is no ridge,
  parameter or tuning.

Reported metrics, pair-equal and component-equal (within-component MAE, then
equal weight per component): pair-equal MAE/RMSE for all models on the whole
catalogue, each split cell, `E_T`, `E_HH_T`, `E_STRICT`; component-equal MAE on
the same pairs. MAE is compared only with MAE, RMSE only with RMSE, on the same
pairs, weights and units.

On the `TRAIN_TRANSFORM_MEAN`-supported subset, list RESP, JOINT, ADD, the three
dictionaries, ZERO and the table mean together. This subset is for discussing
cross-environment transformation transfer; its numbers are not compared with
the full catalogue as if they were the same population.

Paired differences:

* RESP vs JOINT, ADD and the table mean; gain = reference error − RESP error;
* `D_RESPONSE` vs native RESP, `D_LATENT`, `D_FREQ`; gain = reference error −
  `D_RESPONSE` error.

Dictionary delta fidelity `RMSE(delta_D - delta_RESP)` and label error
`RMSE/MAE(delta_D - delta_y)` are separate tables. Better fidelity is not
better labels.

Uncertainty: 2000 paired component bootstrap draws, NumPy seed 1703, the same
draws for every model in a scope. Pair-equal metrics resample components and
keep all pairs of sampled components; component-equal metrics average sampled
component means. CI is the 2.5/97.5 percentile; fewer than 2 effective
components → `NA`. The interval conditions on the existing training state and is
not a cross-seed interval. Small component/molecule counts are reported as
limitations; no automatic "enough pairs proves a law" label is defined.

Influence: for the `E_T` main comparisons, remove one Test molecule and every
`E_T` pair containing it, recompute the gain, and report the range and any sign
reversal. `E_HH_T` / `E_STRICT` show their actual support; correlated pairs are
not used to manufacture precision.

All Test numbers are `DEVELOPMENT_TEST`. Test is evaluated only; it selects no
checkpoint, dictionary, alpha, subset, transformation or reference.

## 5. Explanation boundary

This round adds no single-view word accounting, panel, alpha-uniqueness or
multi-view closure. A shared offline `C` does not imply that the fixed ≤4
selected views of both molecules contain that `C`; the shared-selected-view
count is a coverage field only. A label result from the full frozen function is
not automatically an explanation by a specific word account. No `C ⊕ D` is
synthesised and no prototype is called an implementable replacement. The
`702 → 136` statement remains an S-word-table compression, not a whole-network
or inference-cost compression.

## 6. Stop rules and budget

CPU only, 0 GPU, 0 A100·h. Target wall ≤ 15 min, hard stop 45 min. Reuse the
committed prediction/cache files; if a required frozen prediction or cache is
missing, record the missing object and stop rather than start a GPU job or
training. No new training, optimizer step, seed, dataset, LR/K/rank/panel sweep,
no relaxation of the edit definitions, no multi-bond/MCS/approximate matching,
no change to the official split or any frozen component. The structure
catalogue is hashed before labels; the evaluation reads that frozen hash. If the
full-molecule replay identity check fails, stop instead of forcing numbers. A
small or empty `E_STRICT` is a valid final result; the round ends and returns
the data-task / paper-positioning decision to the researcher. No further round
is started automatically.
