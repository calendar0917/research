# ZINC Kekulé equivalence: training support and frozen-model sensitivity (v1)

Scope: **data analysis + frozen-model inference only**. No training, no checkpoint
selection, no official test access. Official PyG ZINC subset `train` (10,000) and
`valid` (1,000) only. The 225 unseen+ordinary valid molecules are the fixed old
group from `zinc_restricted_adaptation_v1`; it was reproduced, not redefined.

## Bottom line

**A. Do representation differences clearly fragment training support?**
There is a real, verified effect, but it is **small and not the main source** of
the old radius-2 fragmentation. In the fixed 225 unseen+ordinary group the
unseen atom fraction moves from **8.817% (original)** to **8.491%** under the
verified Kekulé-orbit quotient (`rep`), **9.021%** under naive all-aromatic
tokenisation (`arom`), and **8.226%** under joint statistics over all legal
representations (`union`). Full support goes **0/225 → 9 / 12 / 14**. So
**211/225 still have at least one unseen local key after joint canonicalisation**.
Most of the 225 is genuine local-environment novelty, not a Kekulé artefact.

**B. Do existing predictions clearly depend on the representation difference?**
For a subset, yes, but not uniformly. On the 181/225 molecules that have a legal
alternative, mean max |Δ| is **0.0268 (P3B seed0)** and **0.0305 (T1 seed1)**;
median is only **0.0026 / 0.0038**. 46 / 55 of the 225 exceed 0.05 (25% / 30% of
the 181 with alternatives). The largest case, row 278, moves by **0.412 (P3B)** and
**0.340 (T1)** with **no support rescue**. Conversely, 7 support-rescued rows have
essentially zero prediction change. The 44/225 molecules with only one legal
Kekulé form are unchanged by construction, not evidence of invariance.

**C. Do both phenomena explain the main error of current unseen ordinary molecules?**
**No.** Sensitivity is about **20–23%** of the mean absolute error of the 225
(0.0268–0.0305 vs 0.1329–0.1339), the correlation between per-molecule error and
sensitivity is weak (**0.116 / 0.179**), and only 41 / 39 of 225 have sensitivity
larger than their original error. Mechanically averaging the fixed model over all
legal representations makes MAE slightly **worse** on the 225
(+0.00134 P3B, +0.00103 T1; fixed diagnostic, not a new model). Support rescue
affects only 26/225 molecules (14 reach full support); several rescued molecules
have unchanged predictions.

**D. Is a training comparison “ordinary aromatization vs joint equivalent-representation
statistics” worth doing?**
**No, not now; low priority.** Ordinary aromaticisation is not a valid equivalence:
it merges 929 train local classes, 227 of which contain more than one witness
component, and 55 merge atoms with different total H (the dictionary H gap). The
verified orbit quotient is much safer (686 merge classes, only 1 over-merge class),
but its support gain is small; joint statistics gain 29 atom occurrences and 14
full molecules out of 225. There is no evidence that a full training comparison
would address the main error, and this round provides no performance gain to
chase. Do not expand the resonance scope or add a training experiment from this
signal.

---

## 1. Data, grouping, and reconstruction

* Commit used for the inference run: `cfc5cc2` (artifact and code). Local final
  report commit is later; the inference command and run id are in
  `analysis/rr_run.json`.
* Old grouping reproduced from
  `tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis/reference_preds_valid.csv`:
  251 unseen, **225 unseen+ordinary**, 26 unseen+tail, 710 seen+ordinary,
  39 seen+tail. Independent radius-2 content-key reimplementation reproduces
  `r2_unseen_atom_frac` with max absolute difference **4.76e-10**.
* Official test split was never loaded. No labels or prediction errors were used
  to generate representations.
* RDKit fixed at **2026.09.1**. Aromaticity is perceived on the **whole molecule**
  before local environments are cut.
* The existing
  `bjoc_mmp_feasibility_v1.build_rdkit_mol_from_graph` was checked against the
  new reconstruction on all 10,000 train + 1,000 valid graphs: **0 failures,
  0 canonical-SMILES mismatches, 0 total-H mismatches**. It is usable for
  constitution, but it does not make neutral `[nH]` explicit; H is inferred by
  RDKit.

### H-state gap and applicability

The committed atom dictionary cannot encode neutral N/O/S H. Using
**dictionary-explicit H only** for aromatic capacity enumeration is not safe:

| split | molecules with inferred aromatic H | explicit-H capacity outside {0,1} | auditable | no solution | original missing | H-changing candidates |
|---|---:|---:|---:|---:|---:|---:|
| valid | 911 | 907 | 4 | 3 | 1 | 1 |
| train | 9,156 | 9,112 | 44 | 42 | 2 | 2 |

Therefore the pipeline does **not** enumerate with dictionary-only H. It rebuilds
the molecule with RDKit, freezes the inferred **total H**, charge, element and
degree, and verifies every candidate against that frozen state. The explicit-H
audit is diagnostic only; its candidates are never used. No molecule in this
dataset needed the `unsupported` path (0 valid, 0 train), but the code retains
the original prediction and full denominator if verification fails.

## 2. Verified legal representations

For each molecule:

* start from the official PyG graph;
* identify aromatic atoms/bonds on the full RDKit molecule;
* enumerate only single/double reassignments of aromatic bonds legal under the
  fixed atom valences (Kekulé matching);
* include the original representation first, then up to 8 total;
* re-sanitize each candidate and require identical:
  atom correspondence, element, charge, total H, degree, connectivity,
  aromatic atom set, aromatic bond set, canonical aromatic SMILES, no radicals,
  and unchanged non-aromatic bond orders.

Candidate generation never sees labels or model errors.

| split | exhaustive | no alternative | truncated | unsupported |
|---|---:|---:|---:|---:|
| valid (1,000) | 825 | 173 | 2 | 0 |
| train (10,000) | 8,300 | 1,683 | 17 | 0 |

Truncated means the true count is at least 9 and only the first 8 (including the
original) were used; it is flagged in the per-molecule files. An independent
brute-force matching count on all 999 valid and 9,968 train molecules with at
most 24 aromatic edges found **0 count mismatches** against the enumerator.
The 2 valid truncated molecules have 23 and 28 aromatic edges; the largest train aromatic system has 32.

## 3. Training-support change

### 3.1 Train-only vocabulary and merge/split accounting

Vocabularies are built only from official train molecules. `rep` is the
lexicographic minimum key over the verified legal Kekulé orbit of the **same atom
occurrence** (a representation quotient). `arom` replaces every verified aromatic
bond with one aromatic token (ordinary aromaticisation). `union` is the union of
all verified representation keys (joint statistics), not a quotient.

| vocabulary | train types | change vs orig |
|---|---:|---:|
| orig | 12,108 | — |
| rep (orbit quotient) | 11,744 | −364 (−3.0%) |
| arom (ordinary aromaticisation) | 13,195 | +1,087 (+9.0%) |
| union (joint statistics) | 13,168 | +1,060 (+8.8%) |

Train atom occurrences: 231,664; **50,865 (22.0%)** have more than one legal local
key. Key accounting:

| statistic | rep quotient | ordinary arom |
|---|---:|---:|
| merge classes | 686 | 929 |
| extra original keys merged | 699 | 1,075 |
| original keys with >1 image (context split) | 316 (2.6%) | 1,458 (12.0%) |
| classes with >1 witness component | 1 | 227 |
| singleton (unwitnessed) keys in such classes | 2 | 380 |
| classes merging different total H | — | 55 |

Interpretation:

* **`rep` eliminates representation differences of the same local environment.**
  The 686 merge classes are overwhelmingly witnessed inside a single molecule’s
  legal orbit; only **one** class has two disconnected witness components (2 keys).
  That one is reported as an explicit over-merge example below.
* **`arom` merges two different local types**, not merely Kekulé variants.
  227 classes have disconnected witness components, and 55 classes merge atoms
  with different total H. Ordinary aromaticisation is therefore **not** a safe
  representation quotient for this dataset.

### 3.2 Valid groups

Unseen atom occurrences and full support (all local keys seen in train):

| group | n | atoms | scheme | unseen atoms | pooled unseen frac | mean per-mol frac | fully supported |
|---|---:|---:|---|---:|---:|---:|---:|
| all | 1,000 | 23,083 | orig | 506 | 2.192% | 2.461% | 749 |
| all | 1,000 | 23,083 | rep | 495 | 2.144% | 2.404% | 753 |
| all | 1,000 | 23,083 | arom | 556 | 2.409% | 2.671% | 735 |
| all | 1,000 | 23,083 | union | 477 | 2.066% | 2.330% | 763 |
| seen_ordinary | 710 | 16,818 | orig | 0 | 0.000% | 0.000% | 710 |
| seen_ordinary | 710 | 16,818 | rep | 5 | 0.030% | 0.028% | 705 |
| seen_ordinary | 710 | 16,818 | arom | 35 | 0.208% | 0.196% | 684 |
| seen_ordinary | 710 | 16,818 | union | 0 | 0.000% | 0.000% | 710 |
| seen_tail | 39 | 834 | orig | 0 | 0.000% | 0.000% | 39 |
| seen_tail | 39 | 834 | rep | 0 | 0.000% | 0.000% | 39 |
| seen_tail | 39 | 834 | arom | 0 | 0.000% | 0.000% | 39 |
| seen_tail | 39 | 834 | union | 0 | 0.000% | 0.000% | 39 |
| unseen_ordinary | 225 | 4,911 | orig | 433 | 8.817% | 9.218% | 0 |
| unseen_ordinary | 225 | 4,911 | rep | 417 | 8.491% | 8.878% | 9 |
| unseen_ordinary | 225 | 4,911 | arom | 443 | 9.021% | 9.387% | 12 |
| unseen_ordinary | 225 | 4,911 | union | 404 | 8.226% | 8.638% | 14 |
| unseen_tail | 26 | 520 | orig | 73 | 14.038% | 14.878% | 0 |
| unseen_tail | 26 | 520 | rep | 73 | 14.038% | 14.878% | 0 |
| unseen_tail | 26 | 520 | arom | 78 | 15.000% | 16.147% | 0 |
| unseen_tail | 26 | 520 | union | 73 | 14.038% | 14.878% | 0 |

Rescued / lost atom occurrences relative to original:

| group | rescued rep | rescued arom | rescued union | lost rep | lost arom |
|---|---:|---:|---:|---:|---:|
| all | 19 | 34 | 29 | 8 | 84 |
| seen_ordinary | 0 | 0 | 0 | 5 | 35 |
| seen_tail | 0 | 0 | 0 | 0 | 0 |
| unseen_ordinary | 19 | 34 | 29 | 3 | 44 |
| unseen_tail | 0 | 0 | 0 | 0 | 5 |

Key points:

* For the 225: rep changes 433 → 417 unseen atoms (19 rescued, 3 lost), union
  433 → 404 (29 rescued, 0 lost), arom 433 → 443 (34 rescued, 44 lost).
* Full support after joint statistics is **14/225**, not 225/225. The old grouping
  remains mostly genuine unseen local structure.
* Canonicalisation is not monotonic: rep loses full support on 5 seen+ordinary
  molecules, and arom loses it on 26 (35 atom occurrences). `union` has no losses
  because it includes all representations.
* 44/225 have no legal alternative; 0/225 are truncated; 0/225 are unsupported.
  The 44 cannot be helped by this mechanism and are unchanged by construction.

### 3.3 Checkable examples

* **Same environment eliminated (witnessed rep merge).** Train row 0, atoms 3 and 7
  are two atoms in one molecule whose original keys are
  `C(S:C(S:C,D:C),D:C(S:C,S:O,D:C))` and
  `C(S:C(S:C,S:O,D:C),D:C(S:C,D:C))`. They merge to the same canonical key
  `C(S:C(S:C,S:O,D:C),D:C(S:C,D:C))`; the orbit witness is inside row 0. This is a
  genuine Kekulé representation difference of the same local environment.
* **Actual over-merge by the rep quotient (one class only).** Train row 9141 atom 21
  and row 292 atom 5 are both carbon with total H 0, but their original keys differ
  in which adjacent N carries the double bond. They merge to the same rep key with
  **two disconnected witness components**. This is the only rep over-merge class
  (2 singleton keys), so it is a caveat, not a dominant effect.
* **Ordinary aromaticisation over-merges distinct N states.** Aromatic key
  `N(A:C(A:C,A:N),A:C(A:C,A:N))` contains row 2287 atom 12 (**total H 1**) and
  row 1 atom 18 (**total H 0**). Both have dictionary atom word `N` (dictionary H 0),
  so the H state is only available from RDKit. Their original keys are different
  and they are in different witness components. This is both a local-type merge
  and a direct consequence of the neutral `[nH]` dictionary gap.
* **Context split.** Original key
  `C(S:C(S:C,D:C),D:C(S:C,S:O,D:C))` occurs at row 0 atom 3 and row 50 atom 9 and
  maps to two different aromatic keys because the surrounding aromatic system
  differs. This is why `arom` has 1,458 context-split keys, not a clean quotient.
* **Rescued atom.** Valid row 55 atom 9 is an unseen carbon in the 225. Its original
  local bonds are C6 single, N10 double, C18 single; alternate rep 1 has N10 single,
  C18 double. The original key is unseen; rep 1’s key is in the union vocabulary.
  This is the sort of local representation change the old key cannot see.

## 4. Frozen-model sensitivity

### 4.1 Fixed models and controls

* **P3B seed0**: `zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/best.pt` (epoch 377,
  413,505 parameters; recorded valid MAE 0.07622375).
* **T1 seed1**: `zinc_shared_factor_v1/runs/SF_T_s1/best.pt` (epoch 363,
  451,777 parameters; recorded valid MAE 0.07908751).
* No checkpoint reselection. No training. Official test not loaded.
* Run: `res`, pool `res-gpu0`, one A100-40GB, `torch 2.5.1+cu124`, run id
  `kekule-equiv-infer-20261022-225734-18a1823d`, 6.2 s inference.
* **Original replay**: max |Δ| vs stored valid predictions **1.91e-6** for both
  models; MAE 0.07622375 / 0.07908751, matching recorded values.
* **Feature rebuild**: for every one of the 827 molecules with an alternative,
  rebuilding from the raw graph changed **bond** and **junction** features
  (rep 1: 14,084 bond cells, 26,060 junction cells); **atom** and **global**
  features changed in **0** cells. So the test is not only a bond-input swap;
  all relevant derived features were rebuilt, and the only features that depend
  on Kekulé order are bond/junction features, as expected.
* **Atom-renumbering control**: 24 rows with an alternative, rep 1, random atom
  permutation. Max |Δ| **2.38e-7 (P3B)** and **2.38e-7 (T1)**. No permutation
  artefact.

### 4.2 Per-molecule and group results

All 1,000 rows are in `analysis/pred_per_molecule.csv`, with every legal
prediction in `p3b_reps_json` / `t1_reps_json`, plus original, mean, range, std,
max/mean absolute change and errors. Group summary:

| group | n | n alt | model | mean max |Δ| | median max |Δ| | mean range | MAE orig | ΔMAE (fixed all-rep mean) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| all | 1,000 | 827 | P3B | 0.0289 | 0.0104 | 0.0297 | 0.0762 | +0.0042 |
| all | 1,000 | 827 | T1 | 0.0302 | 0.0107 | 0.0309 | 0.0791 | +0.0028 |
| seen_ordinary | 710 | 619 | P3B | 0.0266 | 0.0137 | 0.0273 | 0.0475 | +0.0037 |
| seen_ordinary | 710 | 619 | T1 | 0.0299 | 0.0148 | 0.0303 | 0.0503 | +0.0032 |
| seen_tail | 39 | 16 | P3B | 0.0844 | 0.0000 | 0.0844 | 0.1163 | +0.0356 |
| seen_tail | 39 | 16 | T1 | 0.0372 | 0.0000 | 0.0373 | 0.1725 | +0.0119 |
| unseen_ordinary | 225 | 181 | P3B | 0.0268 | 0.0026 | 0.0280 | 0.1339 | +0.0013 |
| unseen_ordinary | 225 | 181 | T1 | 0.0305 | 0.0038 | 0.0317 | 0.1329 | +0.0010 |
| unseen_tail | 26 | 11 | P3B | 0.0263 | 0.0000 | 0.0263 | 0.3019 | −0.0037 |
| unseen_tail | 26 | 11 | T1 | 0.0280 | 0.0000 | 0.0286 | 0.2591 | −0.0051 |

The seen_tail mean is dominated by one outlier, row 172 (P3B change 2.62,
T1 0.98; P3B predictions −18.777 → −16.159). Its median change is 0. It is not
an unseen+ordinary case; do not use the seen_tail mean as a typical effect.

Counts of per-molecule max change:

| group | model | >0.01 | >0.05 | >0.10 |
|---|---|---:|---:|---:|
| all (1,000) | P3B | 501 | 204 | 47 |
| all (1,000) | T1 | 508 | 236 | 68 |
| unseen_ordinary (225) | P3B | 102 | 46 | 14 |
| unseen_ordinary (225) | T1 | 106 | 55 | 20 |

For the 181/225 with alternatives, the >0.05 counts are 46/181 (25.4%, P3B) and
55/181 (30.4%, T1). The 44 no-alternative molecules contribute exact zeros.

### 4.3 Support-gain subgroup

| subgroup | n | model | mean max |Δ| | median max |Δ| | MAE orig | ΔMAE (fixed mean) |
|---|---:|---|---:|---:|---:|---:|
| unseen_ordinary, any rescue | 26 | P3B | 0.0323 | 0.0222 | 0.1088 | −0.0000 |
| unseen_ordinary, any rescue | 26 | T1 | 0.0520 | 0.0300 | 0.1056 | −0.0049 |
| full support gain, union | 14 | P3B | 0.0262 | 0.0200 | 0.0809 | −0.0019 |
| full support gain, union | 14 | T1 | 0.0393 | 0.0301 | 0.0732 | −0.0056 |
| full support gain, rep | 9 | P3B | 0.0343 | 0.0323 | 0.0650 | −0.0051 |
| full support gain, rep | 9 | T1 | 0.0512 | 0.0323 | 0.0638 | −0.0038 |
| rescued but insensitive | 7 | P3B | 0.0014 | 0.0000 | 0.0914 | −0.0007 |
| rescued but insensitive | 7 | T1 | 0.0001 | 0.0001 | 0.0774 | −0.0000 |

The “rescued but insensitive” rows have support rescue but max |Δ| < 0.01 for
both models; examples: rows 382, 449, 473, 529. Row 382 gains full union support
but P3B predicts 1.314 for both representations and T1 predicts 1.419 for both
(T1 change ~1e-4). Row 529 gains full rep support with P3B change 0.0099 and T1
change 0.0002. Coverage improvement is therefore **not** a performance gain.

### 4.4 Relation to the unseen-ordinary error

| model | subset | n | mean |error| | mean max |Δ| | ratio | Pearson | n with |Δ|>|error| |
|---|---|---:|---:|---:|---:|---:|---:|
| P3B | all | 1,000 | 0.0762 | 0.0289 | 0.379 | 0.315 | 277 |
| P3B | unseen_ordinary | 225 | 0.1339 | 0.0268 | 0.200 | 0.116 | 41 |
| T1 | all | 1,000 | 0.0791 | 0.0302 | 0.382 | 0.421 | 285 |
| T1 | unseen_ordinary | 225 | 0.1329 | 0.0305 | 0.230 | 0.179 | 39 |

The fixed all-representation mean is a diagnostic only; labels are not used to
choose a representation. It increases MAE overall and on the 225 (see table
above), so there is no evidence here for a simple representation-averaging
fusion scheme. Any use of the per-representation predictions for selection would
be a new method and is out of scope.

### 4.5 Checkable per-molecule examples

* **Large sensitivity without support rescue.** Row 278, unseen+ordinary, 4
  legal reps, the same four local keys remain unseen under every scheme. P3B:
  `[0.032, −0.360, −0.380, 0.035]`, max change **0.412**; T1:
  `[−0.167, −0.434, −0.507, −0.112]`, max change **0.340**. The original is not
  obviously privileged; this is model instability under a legal re-representation,
  not a coverage effect.
* **Sensitivity with support gain.** Row 69, unseen+ordinary, 4 reps, union full
  support gain. P3B `[1.580, 1.513, 1.552, 1.539]` (max 0.068); T1
  `[1.570, 1.450, 1.526, 1.498]` (max 0.120). Row 751 (3 reps, full rep/union
  gain): P3B max 0.033, T1 max 0.125.
* **Support gain but no prediction change.** Row 382 (full union gain):
  P3B `[1.314, 1.314]`, T1 `[1.419, 1.419]`. Row 529 (full rep gain):
  P3B `[1.325, 1.334]`, T1 `[1.339, 1.339]`. These are the clearest counterexamples
  to “better coverage implies better prediction”.
* **Outlier.** Row 172, seen+tail, 2 reps: P3B `[−18.777, −16.159]`, T1
  `[−16.887, −15.908]`. The huge P3B change dominates the seen_tail mean; it is
  reported for completeness, not as a typical effect.

## 5. Answers A–D

**A. Fragmentation.** Verified Kekulé differences do fragment the old radius-2
support, but only mildly. The orbit quotient removes 16 net unseen atom
occurrences in the 225 (19 rescued, 3 lost), and joint statistics remove 29
(29 rescued, 0 lost); full support rises from 0 to 9/14 out of 225. That leaves
211/225 still fragmented. The old “unseen” label is therefore mostly capturing
real local chemistry, not an artefact of which Kekulé form was written down.
Naive all-aromatic tokenisation is worse (443 unseen atoms, 44 lost) because it
merges distinct local environments.

**B. Prediction dependence.** The frozen models are not representation-invariant.
Typical changes are small (median 0.003–0.004 in the 225), but a meaningful
minority moves by 0.05–0.4; the largest unseen+ordinary case moves by 0.41 (P3B)
and 0.34 (T1). The dependence is molecule-specific, not a constant offset, and
is not confined to molecules whose support changes. The 44 no-alternative
molecules are unchanged by construction and say nothing about invariance.

**C. Main error.** Not supported. The sensitivity is about one-fifth of the
unseen+ordinary MAE, weakly correlated with error, and averaging representations
does not improve MAE. Support rescue is limited to 26/225 molecules and 14 full
rescues; several rescued molecules have zero or near-zero prediction change.
There are high-error molecules with zero sensitivity and high-sensitivity
molecules with no support rescue (row 278). Representation choice is a secondary
fluctuation, not the main error mechanism.

**D. Training comparison.** Not worth a full round now. The safe object is the
verified orbit quotient, not ordinary aromaticisation. Ordinary aromaticisation
over-merges 227 multi-component classes and 55 H-disagreement classes, so a
training comparison built on it would be chemically invalid. The verified
quotient/joint statistics produce only a small support gain and no demonstrated
error reduction. If this is ever revisited, it should be a narrowly scoped,
pre-registered test on the 14/225 full-gain subgroup, not a broad resonance
expansion; this round does not justify it.

## 6. Reproduction

Local (RDKit 2026.09.1, CPU):

```bash
uv run python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_prep.py
uv run python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_support.py
uv run python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_recon_check.py
uv run python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_enum_check.py
```

Remote frozen inference (no RDKit needed; uses the committed artifact):

```bash
rr deploy res
rr run res kekule-equiv-infer --pool res-gpu0 \
  --result tracks/structural-sufficiency/zinc_kekule_equiv_v1/analysis/preds_valid.npz \
  --result tracks/structural-sufficiency/zinc_kekule_equiv_v1/analysis/infer_checks.json \
  -- python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_infer.py \
       --device cuda:0 \
       --out tracks/structural-sufficiency/zinc_kekule_equiv_v1/analysis/preds_valid.npz \
       --checks tracks/structural-sufficiency/zinc_kekule_equiv_v1/analysis/infer_checks.json
rr pull kekule-equiv-infer
```

Then copy the pulled results into `analysis/` if using a non-default `rr pull`
destination and run:

```bash
uv run python tracks/structural-sufficiency/zinc_kekule_equiv_v1/code/ke_analyze.py
```

The inference run used commit `cfc5cc2`, host `res`, pool `res-gpu0`, one
A100-40GB, torch 2.5.1+cu124, run id
`kekule-equiv-infer-20261022-225734-18a1823d`. `analysis/rr_run.json` records the
execution regime. `analysis/infer_checks.json` records replay, feature-rebuild
and permutation checks.

## 7. Files

* `code/ke_graph.py` — PyG loading, old radius-2 key, model B/F cache and frozen
  inference helpers.
* `code/ke_chem.py` — RDKit reconstruction, Kekulé enumeration, strict candidate
  verification, H audit.
* `code/ke_prep.py` — builds the committed valid representation artifact.
* `code/ke_support.py` — train-only vocabularies and per-molecule support.
* `code/ke_infer.py` — remote-friendly frozen P3B/T1 inference over all reps.
* `code/ke_analyze.py` — per-molecule prediction diagnostics and group summaries.
* `code/ke_recon_check.py` — comparison with the existing BJOC reconstruction.
* `code/ke_enum_check.py` — independent brute-force completeness check of the
  Kekulé matching enumerator.
* `analysis/valid_reps.npz` — verified valid representations (force-tracked;
  `*.npz` is otherwise ignored).
* `analysis/support_per_molecule.csv` — per-molecule support under orig/rep/arom/union.
* `analysis/pred_per_molecule.csv` — per-molecule predictions for every legal rep,
  original/mean/range/std/changes/errors.
* `analysis/support_summary.json`, `analysis/pred_sensitivity_summary.json` —
  machine-readable summaries.
* `analysis/infer_checks.json`, `analysis/rr_run.json`,
  `analysis/reconstruction_check.json`, `analysis/enum_check.json` — controls and provenance.

## 8. Limitations

* Only 8 representations per molecule; 2 valid and 17 train molecules are
  truncated (true count ≥9). Their support and sensitivity may be underestimated.
* 173 valid molecules (44 in the 225) have no legal alternative; zeros there are
  structural, not evidence of invariance.
* The analysis is restricted to radius 2, following the old grouping. Larger
  radii could change the support picture and were not explored.
* The all-representation mean is a diagnostic, not a model. No label was used to
  select a representation.
* Row 172 is a single large outlier in the seen+tail group; group means should be
  read alongside medians.
