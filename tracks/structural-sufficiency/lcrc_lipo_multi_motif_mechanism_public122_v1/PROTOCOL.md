# PROTOCOL — LCRC/Lipo multi-class connection-motif selectivity, parameter dependence, internal path

Round / protocol ID: `lcrc-lipo-multi-motif-mechanism-public122-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_multi_motif_mechanism_public122_v1/`
Previous protocol: `lcrc-lipo-amide-connected-rewire-public122-v1` (executed revision `64bb1c9f`)
Design revision: `773373a0`
Supersedes (unexecuted plan): "amide singleton integration only" — **never executed**; nothing
of that plan was run, and the completed integration / amide evidence is reused, not deleted.

Written before the first formal inference run. The scientific definitions below (six-family
panel, donor rule, T/K0 construction, invariants, scopes, populations, range rule, dose
caliper, parameter states, rC/E, Möbius decomposition, statistics, bootstrap, multiplicity,
influence rules, decision rule) are **frozen**. Only implementation-mapping notes, observed
support counts and the aromatic-coverage pre-registration note (§3.3) may be appended.

---

## 0. Question, and the four propositions this round must separate

The executed amide pilot supports, under the specified rewiring controls, a *selective*
response of the frozen LCRC-C1 model to the amide connection pattern. On its own it cannot
support a general chemical-structure explanation. This round keeps that result and asks four
**separately reportable** questions:

1. **Across families** — does the selective response hold for every member of a *pre-registered*
   six-cell structural panel (three connection environments × N/O), or only for some?
2. **Beyond the statistical perturbation** — does it survive a structure-only *dose calibration*
   that matches the amount of input-statistics change instead of only the physical bond count?
3. **Parameter dependence** — is it a property of the *trained* weights, or already present in the
   verified training-start initialisation?
4. **Internal path** — through which input block (first-order S1 aggregation, second-order S2
   path aggregation, atom scalars) and which interactions does the response mainly flow?

**Zero new training.** No optimiser, no backward pass, no probe training, no new seed or variant,
no label-shuffle training, no dataset swap, no `lr_dispatch.py`. The only parameter states are the
frozen ordinary-MSE `A_s0` checkpoint and **one** same-architecture untrained-weight control (the
verified training-start initialisation). That initial state is an *inference diagnostic*, not a
new-training replicate and not a random-architecture population test.

## 1. Frozen data and split

ChemBench Lipophilicity, published split `random122`; 4200 molecules, Train/Valid/Test =
3360/420/420, published row ids and row order kept.

| file | SHA256 |
|---|---|
| `Lipop.csv.gz` | `34280f0adde234bcf477bd98878c9918c9ea51ed859c9efdc31917c6e5ecd758` |
| `Lipop.random122.ind.pkl` | `bf7acb177e080caf2be9ee220f479f8ea91e0039ff4867a7281753a2fce24a7d` |
| `raw_attributes_v1.json.gz` | `1127607de45e341c89ffc67db85714117831ec84b0845c75aaa11005489c51be` |

Upstream revision `0b10d4c1ac72b6929503694a5f60446824736391`. Row-index bytes: Train
`4e8d42a9…`, Valid `31e3f28f…`, Test `a4d92ad9…`. Source revision, checkpoint, input manifest,
feature normalisation and target mean/scale are reused verbatim from
`lcrc_lipo_amide_connected_rewire_public122_v1` (formal source revision `64bb1c9f`). All three
data SHA256 and the split identity are re-verified in Stage A; nothing is re-split or re-fitted.

Train connected-component census (structure only, recorded before predictions): **3359**
single-component + **1** four-component (row 1646); Valid/Test 420/420 single-component. The
four-component Train molecule is excluded from the rewiring queue.

## 2. Frozen model and the two parameter states

Ordinary-MSE arm `A_s0`, LCRC-C1 (`NO_BOND_MEAN`), 228 674 parameters, best epoch 244.

* checkpoint `tracks/structural-sufficiency/lcrc_lipo_edit_supervision_public122_s0_v1/runs/A_s0/best.pt`
* checkpoint SHA256 `bdd670e8df5ee03ba28f2f4898e7fd137991a9fd23e9add91c08d8b6e0a1d6a7`
* state hash **TRAINED** `7d32221e0c3f482a16e2e170b7df1d4753ff4d7783cc2b5a3386b183a7941e36`
* spec hash `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee`
* reference Test RMSE `0.5761674771952362`, MAE `0.4280564144318241`; reproduction tolerance `1e-5`

**INIT** is rebuilt as the *actual* common initial state of arm `A`:
`build_full_state(spec, train_norm, seed=0)` followed by the original `remap_state` mapping to the
C1 layout, and must reproduce the checkpoint's recorded initial hashes exactly — verified:
init state hash `aac1e061176a93bd12b0e6a3f79e80eced2ff93f84eb72391dc819be59ebb83c`, full init state
hash `666b88d7d0a0f18b55235756ed34b8bfa38263c3bb166fc551db008e2fd36a4f`. Hence the INIT label is
`INIT_TRAINING_START` (not `…_UNVERIFIED`). Both states run in `eval()` under
`torch.inference_mode`, weights and normalisation bit-identical before and after inference.

## 3. Six-family panel (frozen, structure only)

### 3.1 Definitions

| family | environment | target connection |
|---|---|---|
| `amide` | carbonyl | restricted amide `C(=O)–N` |
| `ester` | carbonyl | restricted ester `C(=O)–O–C` |
| `sulfonamide` | sulfonyl | restricted sulfonamide `S(=O)2–N` |
| `sulfonate` | sulfonyl | restricted sulfonate `S(=O)2–O–C` |
| `arylamine` | aromatic | restricted arylamine `Ar–N–C_alkyl` |
| `aryl_ether` | aromatic | restricted aryl ether `Ar–O–C_alkyl` |

Raw encoding: non-aromatic C = 2, aromatic C = 3, non-aromatic N = 7, O = 9, S = 12; neutral charge
bucket = 2; edge class 0 = single, 1 = double; single bond stereo = 0.

### 3.2 Two strictly separate catalogues

* **family key / global count** — `T` must make the target key disappear and drop the family count
  by exactly 1; `K0` must keep the target key and leave the count unchanged. This is the count
  used in every check below.
* **main instances** — the *rewirable* subset: same constraints plus "target bond is a bridge and
  a usable carbon control bond exists". Nothing here reads a label, prediction or fitted quantity.

### 3.3 Pre-registration note on donor scope (no definition changed after any effect)

This round uniformly allows **neutral class-2 *and* class-3 (aromatic) carbon** donor bridges
`u–v` of a single bond. The amide pilot used only class-2 carbons. That change was made *before any
prediction* in this round. Consequence, recorded in Stage A: for the carbonyl and sulfonyl
families the frozen (unified) donor rule reproduces the pre-registered instance/pair counts
exactly; for the **aromatic** families the pre-registered counts had been generated with the
stricter *legacy class-2-only* donor restriction, so the frozen unified rule yields **larger**
aromatic coverage (e.g. arylamine Train 240/255/653 instances/pairs vs the legacy 116/125/293;
aryl-ether `test` 90/113/237 vs 58/70/112). Under the legacy rule aryl-ether matches exactly and
arylamine matches 11 of 12 cells. Both censuses are stored
(`inputs_manifest.json: family_census`, `family_census_legacy_class2_donor`,
`precount_reconciliation`). Nothing is re-selected on the basis of any observed effect.

## 4. Interventions T / K0

For each (target instance, control `k`, donor bridge `u–v`):

* **T** deletes `(center, target)` and `(u, v)` and inserts the **unique** crossing that keeps a
  single connected component (target pattern destroyed).
* **K0** deletes `(center, k)` and the same `(u, v)` and inserts its unique connected crossing
  (target pattern kept).
* Added bonds: single, stereo 0, both directed records.

Invariants (each verified on the exact tensor fed to the model): O / T / K0 single-component;
unreachable-pair fraction 0; no duplicate physical bond or self-loop; atoms, bond count,
bond-type totals unchanged; every atom's AF51 encoding **byte-identical** to O; exactly 2 bonds
deleted and 2 added; T's target key disappears with count − 1; K0's target key holds with count
unchanged; RDKit valence/chemistry legality re-checked without repair or capping.

Stage-A funnel (all rejection reasons stored): 2197 candidate pairs → 1810 structurally and
chemically accepted (0 chemistry rejections, 840 O-graph identity mappings) → the frozen
calibration pool over the same molecules adds 539 accepted T–C candidate links.

## 5. Scope, range support, populations

* **FULL** — the whole new graph is re-encoded, global42 recomputed.
* **G_FIXED** — the local input is taken from the new graph while global42 is restored to the
  molecule's own O value. **Main selectivity and internal-path analysis use G_FIXED**; FULL is
  reported alongside as the full-model response check. Both are always reported.

`RANGE_SUPPORTED` = per-dimension within the raw-Train global42 min/max ± 1e-6, requiring O **and**
the paired T/K0 to be inside. Main rewiring queue = RANGE_SUPPORTED; every structurally and
chemically legal view is a fixed supplement (out-of-range views listed with dimension/value/excess).
Per-dimension membership is **not** a joint in-distribution claim.

Formal rewiring inference is restricted to Valid/Test. Train is used only as a structure
catalogue, for the original-range definition, scaffold "seen-in-Train" marks, and the real-pair
baseline — **no Train rewiring inference**.

## 6. Competing explanation "the model just sees a bigger perturbation" → dose calibration

Physically deleting 2 / adding 2 does **not** mean the model sees the same statistical change.
Weight-, label- and prediction-free doses are computed from the rebuilt input tables alone:

| dose | definition |
|---|---|
| `d1n` | Σ over roots of the L1 distance between per-root multisets of `nbr_code` (typed-neighbour S1) |
| `d1b` | Σ over roots of the L1 distance between per-root multisets of `(AF51(neighbour) bytes, BF13(edge) bytes)` |
| `d2j` | per-root L1 of the joint `(path_cat, path_end63)` counts |
| `d2c` | per-root L1 of the `(path_cat, AF51(endpoint) bytes)` content counts |
| `d2k` | per-root L1 of the `path_closure` marginal |
| `d3` | Σ over atoms of the Train-standardised scalar9 L1 norm |
| `dr` | number of roots with any S1 / S2 / scalar change |

AF51 is atom-wise fixed; `b_sum` generally changes and is recorded separately from the typed
neighbourhood (`d1b`).

**Fixed intensity-calibration control C** (does not replace K0): pool all six families' already
generated *lawful* T/K0 views of the **same molecule**; for a focal T the candidates are those
views of that molecule that keep the family global count unchanged, still hold the target key, and
are structurally/chemically/range legal. Both T and C delete 2 / add 2; the caliper requires
`d1n`, `d1b`, `d2k` **exactly equal**, and `d2j`, `d2c`, `d3`, `dr` either all zero or all nonzero
with a T/C ratio inside `[0.8, 1.25]`. Every caliper-satisfying C is used with equal weight
(identical views deduplicated); C selection never reads a prediction or a label.

If a family has no matching C, the competing explanation is **NOT excluded** for it; no
regression extrapolation and no "no sample = zero effect" is allowed.

## 7. Parameter dependence: rC, rK, E

`aT = |pT − p0|`, `aK = |pK0 − p0|`, `aC = mean over C of |pC − p0|` (raw logD);
`rC = (aT − aC)/(aT + aC)`, `rK = (aT − aK)/(aT + aK)`; a denominator ≤ 1e-12 gives 0 and the
zero-denominator count is reported per family and state (guard: the fraction of pairs with a
degenerate denominator and whether the INIT state is constant on the real views).

**Main parameter-dependence quantity** `E = rC_TRAINED − rC_INIT` on the *identical* views, C set,
weights and queue, with raw companions `J_C = aT − aC` and `J_K = aT − aK`. Per family we report
`rC_TRAINED`, `rC_INIT`, `E`, `J_C`; the `K0` analysis (`rK`) is listed too but never substitutes
for the calibrated evidence. The C pool is frozen and shared between the two states.

## 8. Internal path: full three-factor Möbius decomposition (TRAINED only)

The actual `local_input` entering `LocalMLP` is split into three frozen groups, determined from the
model's own `active_cols` / source (never hard-coded):

* **G1** all first-order aggregation blocks (`b_sum`, typed-neighbour; C1 has no `b_mean`; `s1_joint`)
* **G2** all S2 blocks (`s2_type`, `s2_end`, `s2_joint`, `s2_closure`, `s2_content`)
* **G3** the nine atom-scalar columns

AF51 is always taken from O. For every real T / K0 / C view each of the 8 subsets `S ⊆ {1,2,3}` is
injected through a read-only `forward_pre_hook` on `LocalMLP`: blocks in `S` from the view, all
others from O; weights, LayerNorm, readout, pooling, the global block (fixed at O's value) and
node count untouched; dropout disabled. `S = ∅` must reproduce O and `S = {1,2,3}` must reproduce
the real G_FIXED prediction (both ≤ 1e-5 — verified: max |S=∅ − O| = 6.9e-7,
max |S=123 − G_FIXED| = 1.2e-7 over 3617 views).

`F_V(S) = pV(S) − p0`, `H_V(S) = |pV(S) − p0|`, `L_V(S) = |pV(S) − y_original| − |p0 − y_original|`;
Möbius `γ(U) = Σ_{S⊆U} (−1)^{|U|−|S|} value(S)` gives 3 first-order, 3 second-order and 1
third-order term whose sum equals the full-input change (raw residual ≤ 1e-5). Differences
`T − K0` and `T − mean C` sum to `J`; `L`-space differences sum to `D`. All eight identities are
re-verified in Stage C (observed residuals ≤ 1.1e-6).

## 9. Task relevance, real labels, extrapolation

`J_K`/`J_C` are label-free response differences; `D_K`/`D_C` are deviation differences against the
**original** molecule's measured label; `B` is T vs O deviation. `J` does not use `y`; `D` and `B`
are not "true prediction errors of the new graphs" — the rewired graphs are not compounds with
measured values.

Real-pair evidence reuses the previous round's **823 unique unordered MMP pairs** (no new mining).
Per family, pairs with `|Δ family count| = 1` are selected (direction: more minus fewer), then a
fixed cleaner subset with the other five family counts unchanged. Both populations are reported;
no cherry-picking. Groups: Train–Train; has-Test-with-Train-endpoint; both-non-Train-with-Test;
Valid–Valid. Reported: `Δf/Δy`, MAE, sign agreement, zero-change baseline, and the Train–Train
same-`class_id` mean `Δy` baseline with coverage. Uncertainty is by the original `eval_component`
(not by pair count); `LOW_SUPPORT` is kept rather than dropped. Main families are also
stratified by "scaffold seen in Train" (structure only, pre-defined).

## 10. Statistics

* **Weights:** molecule-equal; instance-equal inside a molecule; control-equal inside an instance;
  donor-equal inside a control; C-equal inside a focal T. Losses/movements are always averaged per
  view first; predictions are never averaged before the loss. Units: raw logD.
* **Primary estimand:** Test split, calibration queue, G_FIXED. **Valid** is a pre-registered
  direction/range check only; Valid and Test are never merged to pad the main sample.
* **Cluster bootstrap:** original non-stereo Bemis–Murcko scaffold clusters (empty scaffold → full
  canonical non-isomeric SMILES), 5000 replicates, seed `20261114`, whole clusters with
  replacement; **one shared multiplicity per split** across family / scope / parameter state /
  mechanism term.
* **Multiplicity:** 6 families × 2 quantities (`rC_TRAINED`, `E`) = **12 items**; each item reports
  a 95% descriptive interval and a Bonferroni simultaneous interval (quantiles `0.05/24` and
  `1 − 0.05/24`). A no-sample item is `NA` — never negative, and it does not reduce the multiplier.
* **Fixed influence check:** per family, delete the top-1 and the `ceil(5%·n)` molecules by the main
  calibrated TRAINED `|J_C,m|` (ties by `row_id`) and recompute `rC`, `E`, `J_C`. The raw `K0`
  queue is reported separately on its own `|J_K,m|` ranking; the two checks never substitute.

## 11. Execution, budget, stop conditions

* **Stage A** (local CPU, before any prediction): verify source identity, build the six-family
  catalogue, generate T/K0, RDKit legality, input range, original doses, the C pool/matching
  coverage, scaffold labels, fixed parameter hashes, and freeze all configs/protocol. Then commit
  and freeze.
* **Stage B** (compute): **one** fixed-matrix inference through `rr` on host `res`
  (`rr doctor/deploy/run/jobs/status/logs/pull`, committed revision, read-only manifests, no manual
  `ssh`/`nohup`): `res-gpu0` runs TRAINED FULL/G_FIXED + the eight-cell internal path; `res-gpu1`
  runs INIT G_FIXED on the actual views. One process per card, `res-2` unused, other tasks
  untouched. GPU budget: planned ≤ 0.2 GPU·h (soft 0.5, hard 1.0). CPU prep: soft 30 min, hard
  90 min.
* **Stage C** (local CPU): shared bootstrap, natural pairs, mechanism tables and figures, report.

Stop immediately and report honestly on: frozen input / checkpoint / manifest mismatch
(`MISSING_BASELINE`); unverifiable initial state; AF51 or structural invariant failure;
reproduction difference > 1e-5; caliper re-verification failure; eight-cell endpoint identity
> 1e-5; no supported Test cohort for a family (`NA`, not a negative).

## 12. Decision rule and allowed conclusions

A family is **supported** only if, on the Test calibration queue with G_FIXED, both
`rC_TRAINED` and `E` have 95% CI lower bounds above zero (and the Valid point estimates point the
same way). Partial patterns are reported as such: `rC > 0` with `E` not above zero means
"present already without the trained parameters"; `rC < 0` means "the calibrated response does not
favour the target pattern"; `NA` means "the competing dose explanation is not excluded (no
caliper-matching control)".

Explicitly allowed conclusions include: some families hold and some do not; the original amide
selectivity is explained by dose or by initialisation; the response flows mainly through S1 and not
S2; or a multi-family, training-related local-pattern response. Never claimed: new-molecule
accuracy, causal chemical effect, a motif ranking, or joint in-distribution membership. The round
stops after this report.

---

## Appendix A — observed Stage-A support (appended after the freeze; no definition changed)

Rewiring support (molecules / instances / pairs), main queue:

| family | Valid | Test |
|---|---|---|
| amide | 174 / 211 / 650 | 164 / 182 / 533 |
| ester | 6 / 7 / 18 | 8 / 10 / 46 |
| sulfonamide | 36 / 36 / 143 | 30 / 31 / 112 |
| sulfonate | 0 / 0 / 0 | 0 / 0 / 0 |
| arylamine | 15 / 16 / 32 | 17 / 17 / 41 |
| aryl_ether | 62 / 74 / 123 | 58 / 70 / 112 |

Dose-calipered calibration queue (molecules / pairs):

| family | Valid | Test |
|---|---|---|
| amide | 68 / 134 | 62 / 110 |
| ester | 2 / 4 | 4 / 7 |
| sulfonamide | 17 / 39 | 13 / 29 |
| sulfonate | 0 / 0 | 0 / 0 |
| arylamine | 3 / 6 | 4 / 4 |
| aryl_ether | 5 / 8 | 8 / 10 |

Calibration coverage is a *structure-only* property and is **low** for the aromatic and ester
families (≈ 9–26 % of the main queue), which is reported as a limitation rather than repaired.

Ester and the two aromatic families therefore carry 4–8 molecules in the Test calibration queue:
their intervals are wide by construction and are reported as `LOW_SUPPORT`, not dropped.
