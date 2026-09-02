# MolHIV CIN × Beam8 full-official-train internal-fold pilot (2026-08-16)

## Question

The 8,000-graph audit produced internal-fold CIN AUCs around .62--.73, while
published CIN results on `ogbg-molhiv` are around .80.  This pilot tests whether
the gap is mainly data/protocol scale and whether the most consistent Beam8
mechanism, bond-state auxiliary supervision, still adds value when all 32,901
official-train molecules are available.

## Leakage boundary

The existing full-data Bemis--Murcko archive was re-audited before training:

- official train: 32,901 graphs;
- official valid: 4,113 graphs;
- official test: 4,113 graphs;
- fold fit/heldout overlap: zero;
- every fold exactly partitions official train;
- fold/official-valid/test intersection: zero.

The experiment records:

- `official_valid_evaluations = 0`
- `official_test_evaluations = 0`

No official-valid or official-test graph was converted to a CIN--Beam8 training
item or evaluated by a model.

## Configuration

- fold: full-data internal scaffold fold 1;
- fit: 23,150 graphs / 904 positives;
- heldout: 9,751 graphs / 328 positives;
- CIN-small-compatible model: 2 layers, hidden 48, dropout .5;
- batch size 128, learning rate `1e-4`, 150 epochs;
- Beam auxiliary weight .1;
- seed 0, CPU;
- variants initialized with an identical CIN core:
  - CIN;
  - aligned bond-state Beam auxiliary;
  - shuffled-patch-target auxiliary.

The auxiliary variants use Beam8 only as label-free training supervision.
Inference receives the same atom/bond/ring complex as CIN.

## Results

| variant | fit AUC / AP | heldout AUC / AP | delta vs CIN |
|---|---:|---:|---:|
| CIN | .9176 / .5412 | **.7684 / .1862** | -- |
| aligned Beam auxiliary | .9293 / .5658 | .7500 / .1910 | -.0185 / +.0047 |
| shuffled auxiliary | .9224 / .5525 | .7599 / **.1985** | -.0086 / +.0123 |

Aligned minus shuffled is:

- AUC: `-.0099`
- AP: `-.0076`

## Interpretation

### The low 8k CIN score was primarily a scale/protocol effect

Moving from roughly 4.2k fit molecules in the localized fold to 23.2k raises
fold-1 CIN AUC from about .624 to .768.  This is much closer to the published
MolHIV CIN range.  It is still an internal scaffold heldout result, not an
official-test reproduction.

### Auxiliary regularization changes the ranking tradeoff, but Beam alignment is not the gain

Both auxiliary variants improve AP over CIN while reducing AUC.  They move
more positives toward the top of the ranking but damage the full positive--
negative pair ordering.  The shuffled target is better than the correctly
aligned target on both metrics, so the AP gain cannot be attributed to Beam8
patch organization.

### The 8k bond-auxiliary promotion does not survive this full-data pilot

On the localized 8k audit, aligned bond auxiliary was the most broadly
consistent mechanism across six fold/seed cells.  On full-data fold 1, correct
alignment is actively worse than shuffled and neither auxiliary variant beats
CIN AUC.  A second full-data seed/fold could measure variance, but this result
is already a negative promotion gate: no official-valid/test evaluation is
justified.

## Decision

- Keep full-data CIN as the proper development baseline.
- Do not promote current Beam8 bond auxiliary as a MolHIV improvement.
- Do not spend official-valid/test budget on this mechanism.
- If Beam8 continues on MolHIV, the next experiment must target the observed
  AUC/AP conflict and demonstrate aligned > shuffled on internal full-data
  folds.  Another scalar auxiliary-weight sweep alone is not sufficient.

Artifact:

- `tracks/ksvd/results/molhiv/cin_beam8_aux_fulltrain_fold1_seed0_20260816.json`

