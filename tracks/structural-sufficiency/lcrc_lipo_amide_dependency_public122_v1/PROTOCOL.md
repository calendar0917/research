# PROTOCOL — LCRC Lipophilicity amide-link dependence (frozen model, masked inputs)

Round / protocol ID: `lcrc-lipo-amide-dependency-public122-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_amide_dependency_public122_v1/`
Design revision: `6f70d05fd883fb3ac66bc786c62739416710054a`
Written before the first formal inference run. The scientific definitions below
(motif, populations, views, scopes, statistics, bootstrap, influence rules,
decision rule) are frozen; only implementation-mapping and observed-support
notes may be appended afterwards.

---

## 0. Question and boundary

The question is *prediction dependence under masking*, not performance:

> Does the already-trained LCRC-C1 Lipophilicity prediction depend on one
> connected structural element with clear chemical meaning — the amide
> `C(=O)–N` link — and is that dependence distinct from a matched single-bond
> cut at the same centre?

Everything here is **zero-training**: no optimiser is created, no backward pass,
no probe, no new model, no new seed, no dataset, no split change, no feature
change. The original complete-molecule Test RMSE is *not* expected to improve; the
frozen weights stay bit-identical.

Masking the link is an information-occlusion intervention on the model input. It
is **not** a chemical reaction, not a new compound with an experimental label, and
it does not identify unique atom contributions or real chemical causality.

Explicitly out of scope this round: PNA probes, PCQM4Mv2, R/W/N arms, LOCAL3,
DUP2, NULL3, fourth-hop experiments, auxiliary supervision, hyper-parameter
search, and re-running the two previous negative results.

---

## 1. Frozen data and split

ChemBench Lipophilicity, official published split `random122`; 4200 molecules,
Train/Valid/Test = 3360/420/420, published row ids and row order kept.

| file | SHA256 |
|---|---|
| `Lipop.csv.gz` | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |
| `Lipop.random122.ind.pkl` | `bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d` |
| `raw_attributes_v1.json.gz` | `1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be` |

Upstream revision `0b10d4c1ac72b6929503694a5f60446824736391`. Row-index bytes:

* Train `4e8d42a95c0b705f43f932b3bbb265277c76a7aa65684b0e5e6ca028580aaf96`
* Valid `31e3f28f31da69249f70922126f19795920bc1e32cc05d5e29e27507bd5aa365`
* Test  `a4d92ad939bf268c00ff69afe77aa5d58c19a6f809b6a09f6dadc51cd648ec93`

The raw attribute artifact is label-free (both directed records per physical bond,
raw `atom_class` / `atom_charge` / `atom_h` / `edge_class` / `edge_stereo`). Test
labels may be read; there is no extra approval gate.

## 2. Frozen model

Ordinary-MSE arm `A_s0`, LCRC-C1 (`NO_BOND_MEAN`), 228 674 parameters, best epoch
244:

* `tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt`
* checkpoint `bdd670e8df5ee03ba28f2f4898e7fd137991a9fd23e9add91c08d8b6e0a1d6a7`
* state hash `7d32221e0c3f482a16e2e170b7df1d4753ff4d7783cc2b5a3386b183a7941e36`

Frozen reference metrics: Valid RMSE `0.7025358944353126`, Test RMSE
`0.5761674771952362`, Test MAE `0.4280564144318241`. The frozen `lipo_v1` spec
(hash `7270f767…`), Train-only feature normalisation and target mean/scale are
reused exactly; nothing is refitted. Reproduction tolerance `1e-5` on RMSE/MAE and,
because per-molecule predictions exist, on the maximum per-molecule prediction
difference. A larger difference triggers a model/split/denormalisation/feature/
precision investigation — never a tuning fix. A missing checkpoint/norm is
reported as `MISSING_BASELINE` and stops the round.

Reused read-only entry points: `lcrc_lipo_model_dissection_v1/code/{dissect_model,
dissect_data}.py`, `lcrc_cross_task_preparation_v1/code/{adapters,molecular_graph,
lcrc_data,feature_spec}.py`, `lcrc_molnet_benchmark_pilot_v1/code/{mb_rdkit,
mb_raw_artifact}.py`.

## 3. Target structure (structure only, label- and prediction-blind)

Four distinct heavy atoms `acyl_c – c(=O) – n`, from the raw attribute graph:

1. `c`: `atom_class = 2` (C, non-aromatic), `atom_charge = 2` (neutral),
   `atom_h = 0`, heavy-atom degree 3;
2. `o`: `atom_class = 9`, `atom_charge = 2`, `atom_h = 0`, degree 1, `c–o`
   DOUBLE;
3. `n`: `atom_class = 7` (non-aromatic N), `atom_charge = 2`, every incident
   original bond SINGLE, `degree(n) + atom_h(n) = 3`, `c–n` SINGLE and a bridge;
4. `acyl_c`: `atom_class ∈ {2, 3}`, `c–acyl_c` SINGLE and a bridge;
5. the three neighbours of `c` are exactly `{o, n, acyl_c}`.

Bridges are defined on the original undirected physical graph (Tarjan, cross-checked
with `networkx.bridges`). A ring atom has at least one incident non-bridge bond.
This is a *restricted* amide definition, not "all amides". `n` may itself sit in
another ring; the two cut bonds must not be ring bonds.

Every match is kept as `(row_id, c, o, n, acyl_c)`; there is no per-molecule
instance choice and no effect-based instance choice.

### Populations (fixed before any prediction)

| population | rule | Train mol/inst | Valid mol/inst | Test mol/inst |
|---|---|---|---|---|
| `ALL` | molecules with ≥ 1 instance; all instances used | 1438 / 1657 | 194 / 234 | 185 / 208 |
| `MATCHED` (main) | only instances with `degree(n) = degree(acyl_c)` **and** identical ring-atom flags | 343 / 354 | 52 / 54 | 42 / 43 |
| `BALANCED` | only instances with `|n_side − acyl_side| ≤ 2` heavy atoms | 172 mol | 28 mol | 23 mol |
| `R` | instances with ≥ 1 remote control candidate | — | 49 mol | 43 mol |

Inside a population, a molecule contributes only its qualifying instances; no
non-qualifying instance of the same molecule is mixed in. These counts are
structural checks: a mismatch is treated as an implementation/index bug and fixed
*before* any formal analysis; if the source data genuinely disagreed, the
deviation would be reported and the analysis stopped, never patched by changing
the definition.

## 4. Main intervention: cut one physical bond and rebuild the input

For every target instance three views:

* `O` — original graph;
* `T` — delete only the `c–n` physical bond;
* `K` — delete only the `c–acyl_c` physical bond.

`T` destroys the `C(=O)–N` connection; `K` keeps it and destroys the neighbouring
`C–C` connection. Both are one-bond, same-molecule, same-centre, same-bond-type
(single, bridge) deletions, and each increases the component count by exactly one.
`MATCHED` additionally matches the degree and ring status of the far end. This
control still does not remove every positional/fragment difference; that boundary
is reported.

Construction rules (frozen): copy the label-free raw row; delete **both** directed
records of the physical bond with their `edge_class`/`edge_stereo`; update
`n_components` on the copied row; then use the frozen raw-graph adapter and the
frozen `lipo_v1` feature builder to rebuild AF51/BF13, scalars9, S1/S2 and
global42. Atoms, `atom_class`, `atom_charge`, `atom_h` and untouched bonds are
preserved; no H is added, no atom deleted, no cap written, no re-sanitisation, no
MASK/UNK type. The original SMILES is metadata only; the model input is never
re-parsed from it. `y`/`exp` never enters the raw graph; labels join by `row_id`
only.

### Two fixed scopes (both reported, never chosen afterwards)

* `FULL` (primary): every input block, including global42, is recomputed from the
  cut graph.
* `G_FIXED` (localisation diagnostic): local inputs and atom scalars from the cut
  graph, but the **un-normalised** `global42` restored to that molecule's `O`
  global42 before the frozen normalisation.

`FULL` is the whole encoding pipeline's response; `G_FIXED` asks whether the
conclusion depends on global statistics changing with the cut. If they disagree,
only scope sensitivity is reported — no claim that one branch alone "causes" the
effect.

## 5. Two bounded supplementary controls (still zero training)

**A. Size-balanced subset.** For every `ALL` instance compute the heavy-atom count
on the `n` side after cutting `c–n` and on the `acyl_c` side after cutting
`c–acyl_c`; keep instances with `|difference| ≤ 2` as a fixed `BALANCED` subset.
Report the overlap with `MATCHED`.

**B. Remote same-element/same-bond-type control `R`.** Only where an `ALL`
instance has a usable remote bond: `q` neutral non-aromatic C (class 2), `r`
neutral non-aromatic N (class 7), single, bridge, all bonds at `r` single,
`degree(r) + atom_h(r) = 3`, `{q,r}` disjoint from `{c,o,n}`, inside the same
original connected component, and `r` matching `n` exactly in degree and
`atom_h`. Cost tuple `(|deg(q)−deg(c)|, |atom_h(q)−atom_h(c)|, |s(q–r)−s(c–n)|)`
with `s` the heavy-atom count of the smaller side of the original component after
that cut. **All** minimal-cost ties are kept; each tie is one separate view; the
loss is averaged over ties inside the instance. Candidates may themselves be
amides and are then reported as "another amide", not as a non-amide control.
`R` checks element/bond-type and positional differences only; its population
differs from `MATCHED`, so the two populations' means are never compared to
prove structural specificity. The strict `degree(q) = degree(c)` subset is
described separately without adding rules.

## 6. Statistics (answer the within-molecule dependence first)

Main population: `MATCHED` Test (primary) with `MATCHED` Valid as a direction
check (not for selecting scope/motif/checkpoint). Train is used only for
structural coverage, as the frozen normalisation source, and for the original
input-range check; no Train masked view is run.

Per molecule `m` with qualifying instances `j`:

```
L0_m = |f(O_m)   − y_m|
LT_m = mean_j |f(T_mj) − y_m|
LK_m = mean_j |f(K_mj) − y_m|
LR_m = mean_j mean_cand |f(R_mjc) − y_m|

B    = mean_m (LT_m − L0_m)     masking cost of the target link
D    = mean_m (LT_m − LK_m)     extra cost of the target link over the matched cut
B_R  = mean_m (LR_m − L0_m)
D_R  = mean_m (LT_m − LR_m)
```

`B > 0` means the mask lowers the prediction quality for the original molecule's
label; `D > 0` means the target connection carries more information under this
protocol. Masked loss is **not** the error of a new compound, and no sign
direction (amide raises/lowers logD) is assumed.

Molecules are equally weighted, instances equally weighted inside a molecule,
tied remote candidates equally inside an instance. Losses are formed per view and
then averaged (never averaged predictions followed by a loss). RMSE uses the same
weights on squared errors. MAE, RMSE, mean absolute and mean signed prediction
movement are all reported, for `O`/`T`/`K`/`R`, in raw logD (one denormalisation
only).

**Uncertainty.** Cluster = original-molecule Bemis–Murcko scaffold (canonical SMILES
without stereo; empty scaffold → the molecule's own non-isomeric canonical
SMILES, so acyclic molecules are not merged into one fake cluster). Paired
cluster bootstrap, 1000 replicates, seed `20261114`; whole clusters are resampled
with replacement, all their molecules are kept with molecule-equal weights, and
one resample is shared by every view/scope of a population. Report percentile 95 %
CI, molecule count, instance count, cluster count and maximum cluster share. The
CI describes sampling uncertainty of the fixed model only, not training-seed
uncertainty; a small cluster count is reported as exploratory, never "fixed" by
treating instances as independent.

**Fixed influence checks.** (1) remove the single molecule with the largest
`|D_m|` (ties → smallest `row_id`) and recompute `B`,`D`; (2) remove the
`ceil(0.05 n)` largest-`|D_m|` molecules and recompute; (3) always report that
row's `y`, `O`/`T`/`K` predictions, all its instances and its structure, and keep
it in the main result; (4) report the fraction and maximum magnitude of frozen
normalised global/scalar values outside the full Train per-dimension min/max for
`T`/`K` (no clipping, no refitting). Removal rules are fixed; "outlier" is never
an extra deletion criterion.

## 7. Real legal molecule pairs (low-support side evidence, non-blocking)

Reuse the deduplicated 823 unordered MMP row pairs of the previous round verbatim
(no re-mining, no expansion). Select only by structure: `|n_target(a) −
n_target(b)| = 1`, where `n_target` counts `ALL` target instances (not
`MATCHED`). Direction: more target instances minus fewer. Pre-registered support:
`P_train` 38 pairs / 17 eval components, `E_T` 21 pairs / 8 components, `E_HH`
3 pairs / 3 components (`E_T` contains Train–Test and therefore lists endpoint
splits; `E_HH` is `LOW_SUPPORT`).

Report, on the legal original molecules, `Δy`, `Δf`, `|Δf − Δy|` and the fixed
zero-change baseline `|Δy|`; `P_train` descriptive, `E_T` supplementary with
pair-equal and component-equal MAE. All `E_T` pairs are listed, no
chemically-pleasant subset. No pair loss training, no transformation-family
baseline. This only says whether the frozen model is compatible with the observed
property differences of real pairs; each pair may also change other fragment
information, so `Δy` is never attributed to the amide alone.

## 8. Mandatory implementation checks (not a 20–30 item gate pile)

1. Zero-cut re-encoding equals the original cache arrays byte for byte; the
   original Test metrics and per-molecule predictions are reproduced (1e-5).
2. Every `T`/`K`/`R` view removes exactly one physical bond (both directed
   records), preserves all atom attributes, increases the component count by one,
   and changes no unrelated edge; the `o–c–n` two-bond path disappears in `T`,
   survives in `K`; the corresponding S2 records are tracked. Neighbourhood
   feature values are not required to be unchanged in `K`.
3. The existing feature/cache validators run (`S1` degree and physical bond count,
   `S2` path validity, index/shape, finiteness).
4. `G_FIXED` restores only the original global42; the new encoding is proven to be
   fed to `forward` (view bond count = original − 1, atom count unchanged);
   predictions are deduplicated by `(row_id, cut set, scope)` while instance
   weights are kept.
5. Fixed atom-relabelling permutation on the 3 smallest `MATCHED` Test rows:
   mapped keys/instances, predictions and aggregates unchanged (1e-5); no extra
   model seed.
6. Small numerical fixtures: raw logD ↔ standardised conversion, pair direction,
   molecule/instance/tie weights, `B`/`D` sign, whole-cluster bootstrap sampling;
   state/norm hashes identical before and after inference.

## 9. Execution, budget, stop conditions

Host `res` only (`res-gpu0`/`res-gpu1`, at most one inference process per GPU;
`res-2` unused). GPU budget planned ≤ 0.1 GPU·h, hard cap 0.5 GPU·h; CPU
preprocessing/analysis ≤ 30 min. Execution via `rr doctor/deploy/run/jobs/status/
logs/pull` on a committed revision; remote compute never edits sources. Record job
id, revision, device and wall time.

Code, config and this protocol are committed before the formal run. Only clear
implementation errors may be fixed and re-run, with the reason recorded; results
never change the motif, the controls, the scopes, the populations, the seed or the
thresholds. When the prescribed inference/analysis is done the round stops — no
next model change, no second motif, no cross-dataset extension.

## 10. Deliverables and conclusion boundary

Directory: `PROTOCOL.md`, `REPORT.md`, `configs/experiment.yaml`, `code/`,
`tests/`, `analysis/` with at least `inputs_manifest.json`, `motif_catalog.csv`,
`view_manifest.csv`, `per_view_predictions.csv`, `results.csv`,
`uncertainty.json`, `natural_pairs.csv`, `runtime.json` (plus the molecule
catalog, `R` candidates, checks, tables and the three case figures). Every view row
traces `row_id`, split, instance, atom/bond ids, scope, population, control cost,
original label and original prediction. Large binaries stay out of git with a
recoverable path.

Main table: `MATCHED` Valid/Test × `FULL`/`G_FIXED` × `O`/`T`/`K` MAE, RMSE, `B`,
`D` with CIs. Second table: all supplementary populations with support counts.

Three original-molecule figures: `D` near the median, maximum `D`, minimum `D`
(deduplicated). Original structure with the target `C–N` and control `C–C` bonds
and atom indices marked, plus `y`, `O`/`T`/`K` and the contrast. SMILES drawing
must first be proven to match the raw graph; masked graphs are never drawn as if
they were measured compounds. No new atom-contribution heat-map; no neural-branch
load-bearing argument replaces the structural control.

Report answers first, in Chinese: what connected structure was intervened on and
how the LCRC prediction moved; whether the target link carries more information
than the matched cut, and whether that is sensitive to scope and single molecules;
what the real pairs can and cannot support. Then the fixed decision rule:

* "consistent amide-link prediction dependence under this masking protocol" requires
  main `FULL` `B` and `D` 95 % CI lower bounds > 0, Valid point estimates in the
  same direction, `G_FIXED` point estimates in the same direction, and `D > 0`
  under both fixed influence checks — and even then the supplementary controls,
  the input-distribution shift and the natural-pair low support are reported
  alongside, without upgrading to a general chemical mechanism, cross-seed
  stability or causal effect.
* main `B` or `D` point estimate ≤ 0 → "this protocol does not support the
  structural-dependence claim" (not "the model never uses the amide").
* positive point estimate with an interval crossing zero → "directional
  observation, evidence unresolved".
* contradictions across scope/Valid/influence → "conclusion sensitive"; no
  favourable subset replaces the main conclusion.

Finally: route suggestions only (not executed), plus 新增训练 = 0, the actual
number of inference passes/views, GPU·h, and the remaining uncertainty.
