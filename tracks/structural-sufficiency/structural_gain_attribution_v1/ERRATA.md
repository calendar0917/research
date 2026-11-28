# ERRATA — MSDS-v0 display layer (chemical attribute decoding)

Round: `structural-gain-attribution-v1` (Phase 0.1).  **This file corrects the
*interpretation layer* of `molhiv_multigranular_structure_sharing_v0` only.**  No
frozen artefact of MSDS-v0 is rewritten; every hash below is unchanged.

## 1. What was wrong

`msds_enum.atom_label` / `bond_label` (MSDS-v0) read raw **OGB category indices**
as if they were physical atomic numbers, formal charges, hybridizations and bond
orders.  Correct decoding uses OGB `allowable_features`:

| field | MSDS-v0 read | correct |
|---|---|---|
| `atomic_num` | `_ELEMENTS[index]`, i.e. element with atomic number *index* | `possible_atomic_num_list[index]` = element with atomic number *index+1* |
| `formal_charge` | the index itself | `[-5..5,'misc'][index]` (index 5 = 0) |
| `hybridization` | shifted ad-hoc table | `['SP','SP2','SP3','SP3D','SP3D2','misc'][index]` |
| `bond_type` | shifted ad-hoc table (1 = SINGLE ...) | `['SINGLE','DOUBLE','TRIPLE','AROMATIC','misc'][index]` |
| `chirality`, `stereo`, `conjugated`, `num_h`, `num_radical_e`, `degree`, `in_ring` | mislabelled or ignored | `allowable_features` tables (`degree` is OGB's `GetTotalDegree()`, i.e. heavy-atom degree **plus** implicit H) |

The example named in the round brief is reproduced exactly: raw `atomic_num` index **5** was displayed as **B**, while it is actually **C** (atomic number 6).

## 2. Scale of the display error

| registry | columns | columns with ≥1 wrong label | node-label slots | wrong node slots |
|---|---:|---:|---:|---:|
| COARSE | 754 | 754 (100.0%) | 2739 | 2739 (100.0%) |
| FULL | 3652 | 3652 (100.0%) | 13622 | 13622 (100.0%) |

Retained-type counts by true element (corrected):

| registry | element counts |
|---|---|
| COARSE | C:1598, N:564, O:287, S:175, P:39, Cl:19, F:16, Br:9, Sn:9, Si:7, Se:5, Co:5, I:4, B:1, Ni:1 |
| FULL | C:9942, N:2109, O:1010, S:364, P:86, Cl:26, F:25, Si:18, Br:16, Sn:8, I:7, Se:4, Co:4, B:2, Ni:1 |

## 3. Impact scope: display only

* Static call-graph scan of all MSDS-v0 modules: `describe_key` / `atom_label` /
  `bond_label` are called only from the registry writer (`run_phaseA_vocab.py`), the
  case-family writer (`run_phaseC_analysis.py`) and the figure titles.  `key_to_string` only serialises the raw integer canonical tuple.  `on_model_input_path` = `false` for every helper.
* The frozen vocabulary was **re-derived from raw integer indices** and is identical to
  the frozen one: coarse_selection_identical=True, full_selection_identical=True, full_parent_identical=True.
* The frozen design matrices were rebuilt in this round from the enumeration cache and
  match `design_CF.npz` with max abs diff 0 (see `bundle/sav1_inputs_meta.json`), so the
  model inputs of C/F are unchanged.
* Frozen artefact hashes are unchanged: phaseA_instances matches the value recorded in
  `analysis/phaseA_run.json`, `design_CF.npz` and `vocab.npz` match the values recorded
  in MSDS-v0 `analysis/vocabulary.json`, and the historical feature cache matches the
  frozen manifest hash.

**Conclusion:** the mislabelled chemistry never entered a model input, a vocabulary
decision, a support statistic or a prediction.  Affected MSDS-v0 statements are the
structural *names* in `structure_registry_*.csv`, `case_families.{csv,json}` and the
figure titles (for example a carbon atom was written `B`, and `SINGLE` bonds were
written as `DOUBLE`).  Any chemical reading of those tables must use
`analysis/registry_v1/` instead.  MSDS-v0's REPORT/PROTOCOL text is left untouched.

## 4. Independent RDKit check (all 41 127 molecules)

* atoms checked: 1048955, bonds checked: 1129451,
  SMILES parse failures: 7.
* On rows where the stored OGB index equals what OGB's own encoder computes from the
  SMILES re-parse, the corrected decode agrees with RDKit for **every** field:
  `{}` atom mismatches,
  `{}` bond mismatches.
* The remaining index disagreements are **dataset-level SMILES ↔ raw-graph drift**
  (the raw graph attributes were featurised from the original SDF, not from the
  `mapping/mol.csv.gz` SMILES): 0 molecules, {'atomic_num': 0, 'chirality': 0, 'degree': 3, 'formal_charge': 0, 'num_h': 3, 'num_radical_e': 25, 'hybridization': 94, 'aromatic': 262, 'in_ring': 0} atoms, {'conjugated': 429, 'bond_type': 418} bonds.  This is
  a property of ogbg-molhiv, not of the decoder, and it is why the cross-check is
  conditioned on index reproducibility.

