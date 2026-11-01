# FEATURE_GAP_AUDIT — what LCRC-A can and cannot see

Round `lcrc-taskaware-representation-pilot-v1`, workstream **P0** (no training).
Artifacts: `analysis/p0_information_flow.json`, `analysis/feature_coverage.json`,
`analysis/feature_collision_tests.json`, `analysis/collision_label_floor.json`,
`analysis/feature_gap_matrix.csv`.
Code: `code/ta_audit.py`, `code/ta_coverage.py`, `code/ta_collision.py`,
`code/ta_floor.py`. Frozen spec: `lipo_v1` (`atom_feat` 51, `bond_feat` 13,
`atom_scalar` 9, `global` 42, `nbr_code` 80, `path_code` 400).

The question is not "does the model perform well". It is: *which chemical
information physically reaches the encoder, which reaches it only implicitly,
which is destroyed before the encoder, and which could never be there?* Every
claim below is backed by an executable artifact, not by reading comments.

---

## 1. Method: three kinds of evidence, three different strengths

| evidence | how it is produced | what it can prove | what it cannot |
| --- | --- | --- | --- |
| **structural** | replay the real path for one molecule and instrument the batch (`ta_audit`) | which fields exist, which keys the forward actually reads, which fields are dead | nothing about learned importance |
| **constructive** | build two molecules that differ only in one piece of chemistry and compare tensors byte-for-byte (`ta_collision`) | a *provable* collision: the input cannot distinguish them | that any real error is caused by it |
| **statistical** | coverage over every molecule of all seven datasets + real-data twin counting (`ta_coverage`, `ta_collision`, `ta_floor`) | how common a structure is, and an upper bound on the error it can explain | causal attribution |

Two deliberately conservative rules were used.

1. A field counts as **wired** only if perturbing one of its columns changes the
   atom representation `h` or the output `y` (68 per-field probes, 73 column
   perturbations, all on a randomly initialised model: this is a graph test, not
   an importance test).
2. A gap counts as **proven** only if two chemically different molecules produce
   identical input tensors *and* the frozen checkpoint returns the identical
   output. Frequency of the pattern is then measured separately, and the error it
   can explain is bounded (`collision_label_floor.json`).

## 2. The path, and the one surprise in it

```
SMILES ──► raw artifact (labels-agnostic npz: atom_class, charge, h, degree,
           ring flags, bond_class, edge_stereo, …)   [chirality NEVER read here]
        ──► per-molecule tensors (atom_feat, atom_scalar, bond_feat, global,
           nbr_code, path_cat/path_end/path_end63/path_closure)
        ──► batch (collate: atom_offset, bond_index shift, path_end globalised)
        ──► LCRC-A: S1 neighbour codes ─┐
                     S2 path codes ─────┼─► shared encoder ─► h ─► f (per-atom) ─┐
                     global (42) ───────┘                          ρ (graph) ───┴─► head ─► ŷ
```

The forward reads exactly **14** tensor keys: `atom_feat, atom_scalar,
bond_feat, bond_index, atom_batch, n_graphs, nbr_atom, nbr_code, path_atom,
path_cat, path_end, path_end63, path_closure, global`.

`atom_type` (int per atom) and `bond_type` (int per bond) are present in every
batch but **never read**. They are documentation tensors: the same information
enters through `atom_feat[:, :16]` (the element one-hot) and through the S1/S2
codes. Anyone reasoning about the model from the batch keys would over-count the
representation by two tensors.

**S1/S2 are joint codes, not marginals.** Both tests pass
(`p0_information_flow.json → joint_codes`):

* two graphs with an *identical multiset* of atom elements and bond classes but a
  different connectivity get S1 codes `[10,46]` vs `[11,45]` — the code encodes
  the neighbour relation, not the atom alphabet;
* swapping the two ends of two S2 path records leaves every additive term
  unchanged but changes the joint term by 2.64 (max abs). Consequence: the
  middle↔end **pairing** of a length-3 path is representable *only* through the
  rank-32 multiplicative term; it is not in any additive S2 code. Two molecules
  that differ only in which end carries which decoration are therefore
  distinguishable but only through one rank-limited interaction.

**No dead field.** All 68 declared per-field entries respond
(`fields_with_no_response: []`). Global fields reach `y` only through `ρ`, as
designed.

## 3. The 24 information items

Statuses: **EXPLICIT** 11 · **INDIRECT** 4 · **EXTRA_COMPUTE** 4 ·
**LOST_IN_PREPROCESSING** 2 · **NOT_DETERMINABLE** 3.
(`EXPLICIT` = a named field carries it; `INDIRECT` = expressible from named
fields but never named; `EXTRA_COMPUTE` = a deterministic rule over the structure
would produce it, but nothing computes it; `LOST` = destroyed before the encoder;
`NOT_DETERMINABLE` = not in 2D constitution at all.)

### 3.1 EXPLICIT (11)

| item | carrier |
| --- | --- |
| element identity | `atom_feat[:, :16]` one-hot (13 elements + aromatic C/N/O/S variants + UNK), S1/S2 codes, `global.distinct_atom_types` |
| aromaticity | 5 aromatic element classes, `bond_class=AROMATIC`, `global.aromatic_atom_fraction` |
| formal charge | `atom_feat` charge one-hot (−2…+2, UNK), `global` charge aggregates |
| hydrogen count | `atom_feat` h-count one-hot (0…4+), `global` mean/max |
| heavy-atom degree | `atom_feat` degree one-hot, `atom_scalar.log1p_degree` |
| bond order | `bond_feat.bond_class` (SINGLE/DOUBLE/TRIPLE/AROMATIC/UNK) |
| ring membership/size/fusion | `in_ring, min_ring_size, ring_bond_count, ring_bond_fraction, is_fused` |
| global topology | the 42 global fields (size, bonds, junctions, cyclomatic rank, components, …) |
| formal charge state / protonation | the SMILES ionic form itself |
| double-bond E/Z | `bond_feat[stereo]` (NONE/E/Z/OTHER/UNK) from `edge_stereo` |
| multi-component salt | `global.n_components`, `frac_unreachable_pairs` |

### 3.2 INDIRECT (4) — present only as a pattern the network must learn

`bond_conjugation`, `valence_hybridisation`, `local_polarity/electronegativity`,
`ring_system_aromaticity`. Each is *derivable* from element + bond order + ring
flags, but nothing names it; the encoder has to build the concept from
`atom_feat`/`bond_feat`/S1/S2. This is the normal cost of an end-to-end
representation and is not a defect — it is the reason the budget below matters.

### 3.3 EXTRA_COMPUTE (4) — cheap, deterministic, and **absent**

| item | why it is not there | what it would take |
| --- | --- | --- |
| HBD / HBA counts | never computed | Lipinski SMARTS over the parsed mol |
| TPSA | never computed | Ertl fragment table over N/O/S environments |
| molecular weight | never computed | element composition is present, so MW is a fixed linear functional of what the model already sees |
| logP / logD prior | never computed | Crippen atom typing (or Wildman) |

These four are the *only* items in the audit that are simultaneously
(a) high-need for several tasks, (b) computed by the classical baseline for free,
and (c) recoverable exactly, deterministically, without touching a label.

### 3.4 LOST_IN_PREPROCESSING (2) — provably destroyed

| item | proof |
| --- | --- |
| **tetrahedral chirality** | the raw adapter never calls `GetChiralTag`/`AssignStereochemistry`; the artifact has no chirality field, so `atom_feat` has no chirality block and no bond-level chirality relation. Constructive test: the BACE enantiomer pair `…[C@@]3…` / `…[C@]3…` (CIP `R` vs `S`) produces **byte-identical** raw artifacts, identical `atom_feat / atom_scalar / bond_feat / nbr_code / path_*` and identical model input, and the frozen `BACE_s0` checkpoint returns the same number to the last bit (`8.033543586730957`) for both. |
| isotope label | same story; no task in this suite resolves isotopes, so it is recorded and ignored |

This is the *only* proven representation gap in the whole audit: a chemist
looking at the input cannot tell the two molecules apart, and neither can the
model.

### 3.5 NOT_DETERMINABLE (3) — no preprocessing change could fix them

`three_dimensional_conformation`, `tautomer/protomer choice`,
`target_specific_binding_pose`. All three are properties of the *measurement
context*, not of the 2D constitution. They are listed so that the round does not
mistake a representation limit for a modelling limit: for FreeSolv (hydration),
BACE (stereospecific inhibition) and HIV (binding pose) these are HIGH-need and
unavailable by construction.

## 4. Structural coverage over the seven datasets (all molecules, no labels)

| dataset | n | UNK atoms | charged mol | chiral mol (centres) | E/Z mol | aromatic mol | HBD>0 | HBA>0 | MW | TPSA | rot. bonds | multi-comp. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESOL | 1128 | 0 | 5.2 % | **0.00 %** (0) | 0.53 % | 58.7 % | 0.44 | 0.73 | 203 | 34.9 | 2.2 | 0.00 % |
| Lipophilicity | 4200 | 1 | 2.4 % | 26.83 % (1934) | 1.90 % | 98.9 % | 0.86 | 1.00 | 383 | 78.8 | 5.2 | 0.02 % |
| FreeSolv | 642 | 0 | 5.9 % | 7.63 % (86) | 1.71 % | 41.6 % | 0.29 | 0.67 | 139 | 20.4 | 1.6 | 0.00 % |
| BACE | 1513 | 0 | 55.9 % | **25.31 %** (802) | 3.64 % | 99.0 % | 1.00 | 1.00 | 479 | 95.2 | 7.7 | 0.00 % |
| BBBP | 2039 | 52 | 5.7 % | **31.24 %** (2920) | 7.26 % | 79.5 % | 0.74 | 0.98 | 344 | 70.7 | 4.2 | 5.15 % |
| Tox21 | 7831 | 139 | 13.2 % | 16.88 % (6494) | 5.94 % | 64.3 % | 0.59 | 0.93 | 276 | 59.6 | 4.3 | 3.13 % |
| ogbg-molhiv | 41127 | 1102 | 12.8 % | **0.00 %** (0) | 0.00 % | 84.8 % | 0.66 | 0.99 | 370 | 82.8 | 4.7 | 7.51 % |

Read carefully, this table already constrains the narrative:

* **Chirality is common exactly where the collision is provable** (BACE 25 %,
  BBBP 31 %, Lipophilicity 27 %) **and absent where it cannot matter** (ESOL 0 %,
  HIV 0 %: the OGB SMILES in this build carry no stereo at all).
* **Charged species are common in BACE (56 %)** — the second-highest need item
  after size — and the spec does carry formal charge explicitly.
* **Multi-component records are real** (BBBP 5.2 %, Tox21 3.1 %, HIV 7.5 % up to
  9 fragments): the encoder sees disconnected graphs, which is handled
  (`n_components`, `frac_unreachable_pairs`) but is a data-hygiene issue rather
  than a representation one.
* UNK atom rows are ≤ 0.16 % of atoms everywhere except BBBP's 52 rows; unknown
  element classes cannot explain the residual error.

Two "headroom" observations from the same artifact: per dataset only **23–59 of
the 400 path codes** and **21–37 of 80 neighbour codes** are used, and
**39–49 of the 51** `atom_feat` columns are non-constant. The alphabet is not
saturated, so "the vocabulary is too small" is not a supportable claim; the
missing items are the four EXTRA_COMPUTE descriptors and chirality.

## 5. Real-data collisions: how often does the same input carry two different molecules?

For each dataset, molecules were grouped by constitution (element + bond graph
with stereo removed); every group containing ≥2 members is a stereo-isoform
family. Each intra-group pair was then pushed through the *real* path and the
model input compared tensors-of-tensors.

| dataset | pairs | pairs with identical model input | Train pairs | Train pairs with **different** label |
| --- | --- | --- | --- | --- |
| Lipophilicity | 115 | **114** | 75 | 68 (mean \|Δ logD\| 0.15, median 0.10, max 1.55) |
| BACE | 32 | **30** | 32 | 18 (56 %) |
| BBBP | 23 | **21** | 23 | 4 (19 %) |
| Tox21 | 139 | **102** | 85 | 19 (22 %) |
| FreeSolv | 3 | 1 | 1 | 1 |
| ESOL | 2 | 0 | 1 | 0 |
| ogbg-molhiv | 0 | 0 | 0 | 0 |

The 2 pairs that *are* distinguishable in each of BACE/BBBP are the E/Z cases:
the double-bond label survives in `bond_feat`, the tetrahedral label does not.

**How much error can this explain?** For a colliding pair with Train labels
differing by δ, the best constant prediction loses δ/2 on each member, so the
Train RMSE floor from collisions alone is
`sqrt(Σ δ²/2 / n_train)` (`analysis/collision_label_floor.json`):

| dataset | floor from collisions |
| --- | --- |
| Lipophilicity | **0.0279** |
| BACE (binary, disagreement rate) | 57 % of pairs disagree |
| BBBP (binary) | 19 % |
| Tox21 (multitask) | 22 % |

Lipophilicity's Train RMSE is 0.28 and its Test RMSE 0.57–0.74; a 0.028 floor
means **the chirality collision cannot be the main source of Lipophilicity
error**, even though it is common and provable. For BACE the label-disagreement
rate is high but the number of affected Train rows is 60/1210 = 5 %, which caps
the achievable AUC gain from fixing chirality at the few-point level. This is the
single most useful quantitative result of P0: it converts "we found a real
representation gap" into "we found a real representation gap of *bounded size*".

## 6. Task–chemistry needs, and where the needs land

Judgement matrix (HIGH/MEDIUM/LOW/NONE) in `analysis/feature_gap_matrix.csv`,
one row per item, with the measured status and carrier in the same row so that
judgement and measurement can never be confused. Aggregated:

| where the high needs land | items | consequence |
| --- | --- | --- |
| **EXPLICIT already** | element identity, aromaticity, bond order, H count, global topology, formal charge, ring membership, protonation state, E/Z (partly) | this block is *not* the gap; it is why LCRC-A is competitive at all |
| **EXTRA_COMPUTE, absent** (4 items × several tasks) | TPSA (HIGH: ESOL, Lipo, FreeSolv, BBBP), HBD/HBA (HIGH: ESOL, Lipo, FreeSolv, BBBP), MW (HIGH: Lipo, BACE, BBBP, HIV), logP prior (HIGH: ESOL, Lipo, BBBP) | the largest *count* of high-need missing information is physicochemical, deterministic and label-free |
| **LOST, proven** (1 item × 1 task) | tetrahedral chirality (HIGH: BACE; MEDIUM: Tox21, HIV) | provable, bounded (see §5) |
| **NOT_DETERMINABLE** | 3D conformation (HIGH: FreeSolv, BACE, HIV), tautomer (HIGH: FreeSolv), binding pose (HIGH: BACE, Tox21, HIV) | a hard ceiling that no representation work in this repo can lift |

Per-dataset sufficiency verdict (input side only):

* **ESOL** — sufficient in kind (no chirality, few charges, MW 203): the
  remaining error is small-molecule solvation physics, where TPSA/HBD/logP priors
  are the natural missing summary, not a structural gap.
* **Lipophilicity** — sufficient in kind but missing the three descriptor priors
  that essentially *are* the target's physical determinants (logP, TPSA, HBD/HBA);
  chirality is present but provably worth ≤ 0.028 RMSE.
* **FreeSolv** — not sufficient, and not fixable: hydration needs 3D conformers
  and tautomer populations that the 2D record does not contain. This dataset is a
  data-limit benchmark, not a representation benchmark.
* **BACE** — the only dataset with BOTH a high-need lost item (chirality) and a
  high-need non-determinable item (pose). It is also the dataset where LCRC-A
  already beats the descriptor baseline, so the bound on chirality matters here.
* **BBBP** — high needs are all EXTRA_COMPUTE (TPSA, MW, HBD, logP); no lost
  information that matters; the observed problem is *split* (valid↔test
  inversion), not structure (see P2 in `REPORT.md`).
* **Tox21** — reactivity/substructure alerts are only INDIRECT; the label matrix
  is sparse and the assay panel heterogeneous. The interesting lever is the
  objective, not the input (see P3a).
* **ogbg-molhiv** — no stereo at all in this build, high needs are pose/aromatic;
  the binding constraint is class imbalance (3.74 % positives), i.e. the
  objective, not the input (see P3b).

## 7. What this audit does *not* claim

* It does not claim that any dataset's error is caused by a collision. The only
  causal statement is a *bound*: a collision cannot explain more than the floor
  in §5.
* It does not measure learned importance. The sensitivity probes prove wiring,
  not reliance.
* It does not evaluate whether the encoder *would* use a feature if it were
  added. That is next round's job, with a matched control.
* The ECDF-style collision scan is capped at 400 groups/dataset for the pair
  enumeration; `n_groups_capped_at` is recorded per dataset, so the pair counts in
  §5 are lower bounds for Tox21/HIV-sized datasets and exact for the small ones.

## 8. Conclusion in one line

The frozen input is **chemically sufficient in kind for most of what these tasks
ask, with three exceptions**: tetrahedral chirality is *provably* destroyed
(bounded consequence), four physicochemical descriptors with high task-need are
*never computed* despite being free and deterministic, and 3D/tautomer/pose are
*nothing the input can ever contain*. Everything else labelled "missing" in
practice is a training objective, split or sample-size problem wearing a
representation costume.
