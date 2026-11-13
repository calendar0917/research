# PROTOCOL — LCRC Lipophilicity connected amide rewiring (frozen model, zero training)

Round / protocol ID: `lcrc-lipo-amide-connected-rewire-public122-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_amide_connected_rewire_public122_v1/`
Previous protocol: `lcrc-lipo-amide-dependency-public122-v1` (executed revision `a0850d89`)
Design revision: `0084daff` (repository commit on which this brief is based)

Written before the first formal inference run. The scientific definitions below
(target motif, donor rule, T/K construction, invariants, scopes, populations,
range rule, statistics, bootstrap, influence rules, decision rule) are frozen.
Only implementation-mapping notes and observed support counts may be appended
afterwards.

---

## 0. Question

The previous round masked bonds: every cut view was **disconnected**, so the
frozen encoder received graphs that are far outside the training input
distribution (`n_components` and `frac_unreachable_pairs` moved by hundreds of
Train SD), and the huge `FULL`-scope numbers could not be read as chemistry.

This round asks a narrower, better-controlled question:

> With atom composition, every atom's heavy-atom degree, the total bond-type
> counts, the rings and the connectivity all held **identical**, does breaking the
> amide connection pattern (versus keeping it, with the same remote donor bond
> rewired) produce a *selective* response of the frozen LCRC-C1 model - and is that
> response task-associated?

Everything is **zero-training**: no optimiser, no backward pass, no probe, no new
model, no new seed, no new dataset, no split change, no feature change. The frozen
weights stay bit-identical. This is not a search for unique atom/bond
contributions, not a chemical-causality estimate, and it is not required to
improve complete-molecule Test RMSE.

Out of scope: PCQM, PNA, ESOL, FreeSolv, other motifs, new training runs, and any
"try another functional group until something is positive" loop.

## 1. Frozen data and split

ChemBench Lipophilicity, official published split `random122`; 4200 molecules,
Train/Valid/Test = 3360/420/420, published row ids and row order kept.

| file | SHA256 |
|---|---|
| `Lipop.csv.gz` | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |
| `Lipop.random122.ind.pkl` | `bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d` |
| `raw_attributes_v1.json.gz` | `1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be` |

Upstream revision `0b10d4c1ac72b6929503694a5f60446824736391`. Row-index bytes:
Train `4e8d42a9…`, Valid `31e3f28f…`, Test `a4d92ad9…` (full hashes in the config).
The previous round's row ids, split manifest, raw attributes, feature spec, frozen
normalisation and checkpoint are reused verbatim; no new split, no refit. Test
labels may be read; there is no extra approval gate.

Raw-graph connected-component census (structure only, recorded before predictions):

* Train: **3359** single-component molecules and **1** four-component molecule
  (row 1646); Valid/Test: 420/420 single-component.
* Hence `n_components` and `frac_unreachable_pairs` are **near-constant** in Train
  (tiny but non-zero variance) - they are *not* identically zero. The previous
  round's wording is corrected in its own REPORT; no stored number changes.

## 2. Frozen model

Ordinary-MSE arm `A_s0`, LCRC-C1 (`NO_BOND_MEAN`), 228 674 parameters, best epoch
244:

* `tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt`
* checkpoint `bdd670e8df5ee03ba28f2f4898e7fd137991a9fd23e9add91c08d8b6e0a1d6a7`
* state hash `7d32221e0c3f482a16e2e170b7df1d4753ff4d7783cc2b5a3386b183a7941e36`

Frozen reference metrics: Test RMSE `0.5761674771952362`, Test MAE
`0.4280564144318241`; reproduction tolerance `1e-5` on the metrics and on the
maximum per-molecule prediction difference against the previous round's stored
predictions. `lipo_v1` spec hash `7270f767…`, Train-only normalisation and target
mean/scale reused exactly (never refitted). A missing checkpoint/cache/norm is
reported as `MISSING_BASELINE` and stops the round; the missing baseline is never
repaired by training.

Every forward pass runs under `eval()` + `torch.inference_mode()`; the prediction
is denormalised to raw logD exactly once. State and normalisation fingerprints are
re-checked after the run.

## 3. Target structure (label- and prediction-blind)

The restricted amide definition of the previous round is kept unchanged: four
distinct heavy atoms `acyl_c - c(=O) - n`, from the raw attribute graph -

1. `c`: `atom_class = 2` (non-aromatic C), `atom_charge = 2` (neutral), `atom_h = 0`,
   heavy-atom degree 3;
2. `o`: `atom_class = 9`, charge 2, `atom_h = 0`, degree 1, `c-o` DOUBLE;
3. `n`: `atom_class = 7`, charge 2, every incident original bond SINGLE,
   `degree(n) + atom_h(n) = 3`, `c-n` SINGLE **and** a bridge;
4. `acyl_c`: `atom_class ∈ {2, 3}`, `c-acyl_c` SINGLE **and** a bridge;
5. the three neighbours of `c` are exactly `{o, n, acyl_c}`.

Bridges are Tarjan bridges of the original undirected physical graph (4200/4200
cross-checked with `networkx.bridges` in the previous round and re-checked here).
A ring atom has at least one incident non-bridge bond. This is a *restricted*
amide, not "all amides".

**New in this round (a new protocol, not a re-run of the previous 42-molecule
confirmation):** the main analysis no longer requires `degree(n) = degree(acyl_c)`.
That old `MATCHED` requirement survives only as a *pre-marked supplementary
subset*.

## 4. Donor bonds and the T/K rewiring (structure only)

Only molecules whose **original** graph has `n_components = 1` are used. For every
target instance, every remote donor bond `(u, v)` in the same component is
enumerated with, and only with:

* `u`, `v` both neutral `atom_class = 2` (non-aromatic carbon);
* `u-v` an original SINGLE bond, a bridge, `edge_stereo = 0` (no E/Z code);
* `{u, v}` disjoint from `{c, o, n, acyl_c}`;
* neither `u` nor `v` is a carbonyl carbon (no double bond to a class-9 O);
* no ranking, no distance filter, no label/prediction/effect-based selection.

For each `(instance, donor)` pair, two views are built:

* `T` (destroy the pattern): delete `(c, n)` and `(u, v)`; try both crossings
  `[(c,u), (n,v)]` and `[(c,v), (n,u)]`; keep the one that leaves the graph
  connected and creates no duplicate bond/self-loop. Exactly one crossing must be
  legal; otherwise the pair is rejected and the reason recorded (never chosen by
  prediction).
* `K` (keep the pattern): delete `(c, acyl_c)` and the **same** `(u, v)`; try
  `[(c,u), (acyl_c,v)]` and `[(c,v), (acyl_c,u)]` under the same legality rule.

Added bonds are plain SINGLE bonds with `edge_stereo = 0`, written in both
directions together with their attributes. Only pairs where **both** T and K exist
for the **same** donor qualify; a T from one donor and a K from another never form
a pair.

View identity (frozen, and different from the previous round's cut-set-only key):

```
view_key = "<row_id>:R<a-b>.<c-d>:A<e-f>.<g-h>"       # row id, removed set, added set
```

The scope (`FULL`, `G_FIXED`) is an additional column of every prediction row, so
`(view_key, scope)` is the prediction key.

### 4.1 Structural invariants that must be verified (not assumed)

1. Whole graph remains a single connected component; no self-loop, no duplicate
   physical bond; both directed records of each added bond exist.
2. Heavy atoms, `atom_class` / `atom_charge` / `atom_h`, per-atom heavy degree, the
   total bond count and every bond-type total are unchanged; the removed set and
   the added set are exactly the requested ones.
3. Every original ring bond is unchanged and still a ring bond; the added bonds are
   bridges; ring-atom flags and minimum ring sizes are unchanged.
4. `T` loses the `o-c-n` two-bond pattern and `n_instances` drops by exactly one;
   `K` keeps `o-c-n` and `n_instances` is unchanged (the third neighbour of `c`
   becomes a fresh non-aromatic carbon carrying the acyl role).
5. Every view removes exactly two and adds exactly two physical bonds (both
   directions), i.e. the edit is a real graph edit and never only a statistics
   column change.
6. Rebuilding with the frozen `lipo_v1` adapter must leave **every atom's AF51
   feature byte-identical to the original view** (composition, degree, raw atom
   attributes, incident bond-class fractions, rings). If it does not, the
   implementation is investigated before any prediction - this control failure may
   not be ignored. `K` is not a chemically equivalent negative control: it tests
   "connected rewiring that keeps the amide pattern", not "all C-N effects
   excluded".

Independent structural pre-counts (original graph only; label- and prediction-blind):

| split | molecules | instances | `(instance, donor)` pairs |
|---|---|---|---|
| Train | 870 | 1021 | 2506 |
| Valid | 121 | 150 | 332 |
| Test | 105 | 120 | 271 |

Previous-round `MATCHED` instances that own a donor (supplementary subset):
Valid 35 molecules / 36 instances, Test 18 molecules / 18 instances. A mismatch is
an implementation/index bug and is fixed before any analysis; reductions caused by
*extra* legality checks are listed per pair with their reason.

## 5. Chemistry and input validity (before any prediction)

1. **SMILES -> raw atom mapping.** `smiles_to_raw_row` (the frozen vocabulary
   function that produced the artifact) must reproduce the artifact row exactly; if
   the atom order differed, a typed / bond-typed graph isomorphism supplies the
   mapping. No mapping -> the molecule is rejected.
2. **Rewired RDKit copy.** The same two deletions and two additions are applied to
   a copy of the RDKit molecule and `Chem.SanitizeMol` must succeed. Heavy atoms,
   per-atom formal-charge bucket, per-atom H-count bucket, aromatic flags, the bond
   set with bond-type codes and the component count must match the edited raw
   graph. Failures are recorded as rejection reasons; no H is added, nothing is
   capped, nothing is chemically "repaired" to make a pair pass. Valence legality
   does not mean the rewired graph is a real compound or has been measured.
3. **Input range.** Per-dimension min/max of the **raw** `global42` over the whole
   Train split (from the frozen Train cache, tolerance `1e-6`) define the frozen
   marginal range. A pair is `RANGE_SUPPORTED` iff `O`, `T` **and** `K` are all
   inside it. Being per-dimension inside the range is **not** a proof of joint
   in-distribution membership; the report must not claim that.

Main analysis uses `RANGE_SUPPORTED` pairs only: filter donors -> average inside an
instance -> average inside a molecule. All structurally and chemically legal donors
form a fixed supplementary population; out-of-range views are listed separately
(dimension, value, Train min/max, excess, count) and never replace the main result.
Quantiles of the frozen-normalised `scalar9`/`global42` and the deviation from the
Train range are reported without clipping and without refitting.

All targets, donors, rejection reasons, range flags, populations and raw hashes are
written and frozen **before** any prediction; no re-selection after seeing
predictions. If the Test split has no qualifying molecule, the round reports
`NO_SUPPORTED_COHORT` and stops the main inference (no relaxing of the range, no
new structure).

## 6. Two fixed scopes (both reported)

* `FULL` (primary): every input block including `global42` recomputed from the
  rewired, still-connected graph.
* `G_FIXED` (input-path localisation diagnostic): local inputs and atom scalars
  from the rewired graph, but the molecule's own original raw `global42` restored
  before the frozen normalisation.

`O` (original graph) views are predicted too; for an `O` view the two scopes are
mathematically identical, so that identity is a tolerance gate. `n_components = 1`
and `frac_unreachable_pairs = 0` must hold identically for `O`, `T` and `K`; no new
placeholder, embedding or "disconnected" branch is used.

## 7. Readout: selective response vs reference deviation (kept separate)

For original molecule `m`, target instance `j`, the same donor `d`:

```
p0 = f(O),  pT = f(T),  pK = f(K)            (raw logD, one denormalisation)

J_m = mean_j mean_d ( |pT-p0| - |pK-p0| )     extra prediction movement
D_m = mean_j mean_d ( |pT-y_original| - |pK-y_original| )
B_m = mean_j mean_d ( |pT-y_original| - |p0-y_original| )
```

`J` measures the response only and does not prove the response is *right*. `D` and
`B` are computational-fidelity contrasts against the **original molecule's**
experimental value; the rewired graph is a different (unmeasured) graph, so `y` is
never its label and `|pT - y|` is never "the new compound's accuracy". No sign of
`logD` change is assumed.

Main estimate: Test `RANGE_SUPPORTED` `mean(D_m)`, `mean(J_m)`, with `mean(B_m)`
alongside; Valid reported with the same definitions as a direction check.
Molecules are equally weighted, instances equally weighted inside a molecule,
qualifying donors equally weighted inside an instance. Losses/movements are formed
per view first, then averaged; predictions are never averaged before the loss.
Every pair's `p0/pT/pK`, signed movement, reference losses and graph difference are
written out.

**Uncertainty.** Cluster = the original molecule's non-isomeric Bemis-Murcko
scaffold, reused verbatim from the previous round's catalog (empty scaffold -> the
molecule's own non-isomeric canonical SMILES). Paired cluster bootstrap, 1000
replicates, seed `20261114`; whole clusters are resampled with replacement, and one
resample is shared by both scopes of a population. Report percentile 95 % CIs,
molecule / instance / donor / cluster counts and the maximum cluster share. Sample
size is counted in molecules and clusters, never in donor views; the CI describes
sampling uncertainty of this one frozen checkpoint only.

**Fixed influence checks.** Rank molecules by main `FULL` `|D_m|` (ties by
`row_id`); (1) drop the single largest, (2) drop the `ceil(0.05 n)` largest, and
recompute `D`, `J`, `B` each time. The `G_FIXED` scope gets its own identical
checks and is reported separately and clearly labelled. The previous-round
`MATCHED` subset and the all-legal-donor population are pre-defined supplements
only.

No new MMP mining and no retraining: any real-compound side evidence is quoted
from the clarified previous round (21 `E_T` pairs / 8 components, `E_HH` 3 pairs,
`LOW_SUPPORT`).

## 8. Mandatory implementation checks (bounded, not a gate pile)

1. Zero-rewire re-encoding equals the frozen structure cache byte for byte; the
   original Valid/Test metrics and per-molecule predictions are reproduced (`1e-5`).
2. Every `T`/`K` view satisfies §4.1 items 1-6 (structure + AF51 identity) **on the
   exact tensors fed to the model**; component count and
   `frac_unreachable_pairs` identical across `O`/`T`/`K`; the view keys contain
   row id, removed set and added set; duplicate/self-loop crossings are rejected.
3. The frozen feature validator runs (`S1` degree and physical bond count, `S2`
   path validity, index/shape checks, finiteness).
4. RDKit mapping, sanitisation and attribute equality per pair (§5).
5. Numerical fixtures: raw logD <-> standardised conversion, molecule / instance /
   donor weighting, `J`/`D`/`B` signs, the original-label reference, and
   whole-cluster bootstrap sampling (shared resample, cluster grouping).
6. State and normalisation hashes identical before and after inference;
   `optimizer_created = false`, `backward_calls = 0`.

## 9. Conclusion rule (validity first, then effect)

Answer first: do the structural invariants, the chemistry validation and the input
range checks hold? The main conclusion covers only the population that actually
qualifies.

The sentence

> "under this frozen LCRC and this paired rewiring protocol, breaking the amide
> pattern produces a larger prediction movement and a larger reference deviation
> than keeping it"

may be written only if all of:

1. main `FULL` Test `D` **and** `J` 95 % CI lower bounds > 0,
2. main `FULL` Test `B` 95 % CI lower bound > 0,
3. Valid `D`/`J`/`B` point estimates point the same way,
4. both fixed influence checks still give `D > 0` and `J > 0`.

Even then it does not mean the new molecules are predicted worse, that the amide
has a real causal contribution, or that the amide matters more than other groups.

`G_FIXED` Test `D`/`J` (and Valid) are reported separately; only if `G_FIXED` also
supports the direction may one add "the conditional local-input intervention is
compatible with that direction". If `FULL` supports but `G_FIXED` does not, the
report distinguishes the whole-model response from the local-statistics path and
does not attribute the whole-model effect to S2.

CI crossing zero -> "unresolved"; reversed point estimate -> "this control does not
support the selectivity"; contradictions across scope/Valid/influence -> "conclusion
sensitive". A non-significant result is never "no effect", a model response is never
a chemical mechanism, and the previous round's formal rule pass is never merged with
this round's scientific conclusion.

After this round the amide-intervention line stops: the outcome is either a limited
case-based observation or a documented boundary of interpretation. No automatic
switch to another functional group.

## 10. Execution, budget, deliverables

Host `res` only (`res-gpu0` / `res-gpu1`, one A100 is enough, no forced
parallelism, `res-2` unused, no other jobs touched). Execution through
`rr doctor/deploy/run/jobs/status/logs/pull` on a committed revision; the remote
never edits sources. GPU budget planned ≤ 0.1 GPU·h, hard cap 0.3 GPU·h; CPU
preprocessing/analysis ≤ 30 min. Only clear implementation errors may be fixed and
re-run, with the reason recorded; a weak result is never a reason to rerun or
extend. Code, config and this protocol are committed before the formal run; all
eligibility and range manifests are frozen before prediction and inference reads
only those fixed lists.

Deliverables in this directory: `PROTOCOL.md`, `REPORT.md`,
`configs/experiment.yaml`, `code/`, `tests/`, `analysis/` with input hashes, the
instance/donor/rejection/range manifests, per-view predictions, per-pair and
per-molecule summaries, CI/influence outputs, run records, and three paired
original/rewired case figures (chosen by main `FULL` `D` near the median, the
maximum and the minimum; deduplicated; both scopes listed; the figures state that
the rewired graphs have no experimental label and that `y` is the original
molecule's value used only as a computational reference).

`REPORT.md` answers first, in Chinese: which intervention problem this round fixed;
whether the frozen model still responds selectively to the target pattern when
composition/AF51 are unchanged; and whether the result supports a structural
interpretation and what is still missing. It also states the difference between the
original-label reference and a real chemical label, the actual main/supplementary
coverage, the zero-training statement, and the device/version/GPU·h.
