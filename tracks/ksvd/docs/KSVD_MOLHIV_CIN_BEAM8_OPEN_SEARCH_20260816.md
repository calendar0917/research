# MolHIV CIN × Beam8 open-search audit (2026-08-16)

## Scope and evaluation boundary

This audit broadens Beam8 beyond the previously closed direct atom-residual
route.  Every reported experiment uses only the 6,400 official-train graphs in
the localized 8,000-graph MolHIV subset and evaluates an internal
Bemis--Murcko scaffold fold.  No official-valid or official-test graph is
encoded or evaluated by a model in this audit.

All 2026-08-16 result files were checked and record:

- `official_valid_evaluations = 0`
- `official_test_evaluations = 0`

The approximately 0.80 CIN number from the CWN paper is therefore not the
baseline for these tables.  The official `cwn-molhiv-small` configuration does
indeed use two layers, hidden size 48, mean cell readout, dropout 0.5 and 150
epochs, as this compatible implementation does.  The decisive protocol
difference is that the paper model trains on the full official training set
(about 33k molecules) and reports the official benchmark split, while this
audit trains on roughly 4.1k--4.5k molecules per internal fold and tests
cross-scaffold inside a 6.4k subset.

## Mechanisms tested

### Direct atom residual (closed earlier)

Beam patch states were scattered to their incident atoms after the first CIN
layer.  Correct patch binding did not reliably beat shuffled/no-patch controls.
The route remains rejected; see
`KSVD_MOLHIV_CIN_BEAM8_INCREMENT_AUDIT_20260815.md`.

### Bond-cell injection

Each patch-internal bond occurrence is mapped explicitly to the corresponding
CIN 1-cell.  A patch-conditioned message is injected after layer 1, then the
second CIN layer propagates it to atoms and rings.  This is more chemically
aligned than atom averaging but still aggregates repeated patch occurrences at
the same bond.

Across seed 0 and all three folds:

| fold | aligned AUC / AP | shuffled | no-patch |
|---:|---:|---:|---:|
| 0 | .6917 / .1220 | .6971 / .1360 | .6977 / .1289 |
| 1 | .6270 / .0555 | .6228 / .0581 | .6254 / .0556 |
| 2 | .7634 / .2753 | .7408 / .2210 | .7599 / .2483 |

Fold 2 was repeated for three seeds.  Aligned beat shuffled AUC in 3/3 seeds
(mean delta about +.0108) and no-patch AUC in 3/3 (mean delta about +.0054).
Fold 0 was also repeated for three seeds and did not reproduce a correct-binding
advantage.  Bond-cell injection therefore contains a real but
scaffold-conditional signal, not a cross-scaffold improvement.

### Beam auxiliary supervision on bond states

Beam is used only during training.  First-layer CIN bond states must reconstruct
patch position, the canonical slot-pair multiset and completion role; inference
uses the unchanged CIN graph.

Across two seeds and three folds, aligned auxiliary supervision beat shuffled
in AUC in 5/6 cells (mean +.0061) and matched CIN in 5/6 (mean +.0097).  AP gains
were less alignment-specific: 3/6 wins against shuffled (mean +.0018), but 4/6
wins against matched CIN (mean +.0118).

This is the broadest mechanism found so far.  Most of its value is best
interpreted as structured multitask regularization; the aligned-vs-shuffled gap
shows a smaller additional Beam-specific component.

## Paired graph and structural-subgroup audit

Prediction-bearing seed-1 files were produced for fold 0 and fold 2, with a
second fold-0 seed for stability.  The read-only analyzer reconstructs only
label-free descriptors: molecule size, cycle/ring density, Beam patch count,
completion fraction, patches per atom/bond, overlap density, and several
ring-boundary--patch multiplicity measures.

Key paired results for aligned bond injection:

| fold/seed | comparison | AUC delta | AP delta |
|---|---|---:|---:|
| 0/1 | vs shuffled | -.0103 | -.0194 |
| 0/1 | vs no-patch | -.0087 | -.0118 |
| 0/2 | vs shuffled | -.0002 | -.0115 |
| 0/2 | vs no-patch | +.0012 | -.0178 |
| 2/1 | vs shuffled | +.0005 | -.0135 |
| 2/1 | vs no-patch | +.0097 | +.0218 |

The fold-2 aligned-vs-no-patch gain remains positive across low, middle and
high tertiles of patch count, completion fraction and ring density.  Fold 0's
loss is similarly broad.  Descriptor/error correlations are small and change
sign across folds.  In particular, ring density has no stable direction that
separates the useful and harmful regimes.

Conclusion: there is no defensible low-capacity structure gate from the tested
descriptors.  Selecting one from heldout subgroup labels would be a post-hoc
rule, so no conditional gate is promoted.

Artifacts:

- `tracks/ksvd/code/analyze_molhiv_cin_beam8_subgroups.py`
- `tracks/ksvd/results/molhiv/cin_beam8_bond_subgroup_audit_20260816.json`

## Node-level auxiliary route

The next mechanism moves the more stable auxiliary idea to the requested node
granularity.  After the first CIN layer, each atom state reconstructs the mean
Beam patch-position vector over its incidences, slot distribution, center /
previous-overlap / next-overlap roles, completion incidence and log incidence
count.  Beam is again absent at inference.

The shuffled control permutes patch-level position and completion attributes
inside each component while preserving atom--patch degree, slot and local role
marginals.  A combined `multiaux` variant averages this node objective with the
bond-state auxiliary objective.

Seed-0 cross-fold results are:

| fold | matched CIN | node aligned | node shuffled | aligned - CIN | aligned - shuffled |
|---:|---:|---:|---:|---:|---:|
| 0 | .6989 / .1303 | **.7326 / .1652** | .7220 / .1224 | +.0337 / +.0349 | +.0107 / +.0428 |
| 1 | .6242 / .0513 | .6002 / .0547 | .6079 / .0521 | -.0240 / +.0034 | -.0077 / +.0026 |
| 2 | .7348 / .1900 | **.7450 / .2632** | .7399 / .2744 | +.0102 / +.0732 | +.0051 / -.0112 |

The node objective beats matched CIN AUC on 2/3 folds and AP on 3/3, but
correct alignment beats shuffled AUC and AP on only 2/3 folds.  Fold 0 is the
clearest alignment-specific result in the open search; fold 1 reverses the AUC
effect and fold 2 reverses the AP effect.  The mechanism is therefore
promising but still scaffold-sensitive.

Naively averaging node and bond auxiliary losses is harmful: on fold 0 the
aligned multiaux model reaches only .7051/.1224, while its shuffled control is
.7216/.1333.  Equal-weight multiaux is rejected as gradient-conflicted rather
than complementary supervision.

Fold-1 diagnosis did not repair the negative transfer:

| fold-1 variant | AUC | AP |
|---|---:|---:|
| full node aux, weight .10 | .6002 | .0547 |
| full node aux, weight .03 | .6094 | .0601 |
| patch position/completion only, aligned | .6075 | .0528 |
| patch position/completion only, shuffled | **.6255** | .0565 |
| local slot/role/count only | .6111 | **.0621** |
| full node aux, decay .10 to zero by epoch 50 | .5961 | .0549 |
| same decay, shuffled | **.6271** | **.0637** |

Lower weight helps but does not recover CIN AUC.  More decisively, the
patch-only shuffled control recovers to approximately CIN level while correct
patch organization remains harmful.  Thus fold 1 is not explained by one bad
local statistic: its task gradient conflicts with aligned Beam patch
organization itself.  Early-only auxiliary supervision is worse: even after
100 epochs of label-only fine-tuning, the aligned trajectory does not recover,
while the shuffled trajectory slightly exceeds matched CIN.  Simple constant
weight, lower weight, target ablation, equal-weight multiaux and linear
annealing are therefore all closed as generic repairs.

## Current decision map

- Direct atom residual: reject.
- Bond-cell inference-time injection: retain only as evidence of a
  scaffold-conditional alignment signal; do not promote as the main model.
- Descriptor gate for bond injection: reject with current evidence.
- Bond-state Beam auxiliary: retain as the strongest broadly consistent route.
- Node-level Beam auxiliary: retain as a stronger node-granularity signal, but
  do not promote as a universal replacement because its task gain and
  alignment advantage remain scaffold-sensitive.
- Bond + node multiaux: reject in its equal-weight form.
- Node-aux weight/target/schedule repair: reject the simple variants tested;
  any continuation needs explicit gradient-conflict handling rather than
  another scalar weight.
- Full official-train internal-fold pilot: completed on fold 1.  CIN rises to
  .7684 AUC, while aligned bond auxiliary falls to .7500 and loses to shuffled
  auxiliary (.7599).  See
  `KSVD_MOLHIV_CIN_BEAM8_FULLTRAIN_PILOT_20260816.md`; no official-valid/test
  evaluation is justified.

## Multi-cover pretraining terminal update

An 8-seed audit on 1000 official-train molecules confirms that Beam8 cover
diversity is real: 96.4% of eligible molecules change cover and the mean
number of distinct covers is 6.27/8.  Multi-cover therefore was not rejected
as a duplicate-sampling artifact.

The first masked-chemistry pilot exposed a protocol leak: complete atom/bond
attributes participated in attributed canonicalization while the target's
canonical slot remained visible.  Those runs are archived only as leakage
diagnostics.  In the repaired protocol, cover construction and slot IDs use
binary topology only.

Across three internal scaffold folds, topology-only multi-Beam improves over
single Beam and shuffled patch binding, but loses to matched 4-cover random
BFS on every fold.  Mean heldout CE / field accuracy are .5748/.7895 for
multi-Beam and .4713/.8123 for random BFS.  The Beam-minus-random effects are
-.1036 CE reduction and -.0227 accuracy gain.

Decision: generic multi-cover SSL remains viable, but the current Beam-specific
pretraining route is rejected before CIN transfer.  See
`KSVD_MOLHIV_BEAM8_MULTICOVER_SSL_20260816.md`.

## Remaining-routes terminal audit

The topology-only SSL checkpoints were subsequently transferred into matched
CIN models by copying only atom/bond chemistry embeddings.  Beam SSL averages
+.0003 AUC / -.0022 AP versus CIN across three folds; shuffled Beam SSL is
better on average, so the effect is not alignment-specific.  A balanced-BFS
candidate-pool sampler improves masked reconstruction on all folds, but its
CIN AUC deltas (+.0122, +.0135, -.0207) remain scaffold-sensitive.

Finally, continuous masked-context embeddings were evaluated as raw features,
PCA projections, random sparse dictionaries and KSVD codes.  Mean AUCs are
.6217, .5299, .5907 and .5691 respectively; KSVD's lower reconstruction error
does not translate to scaffold-heldout performance.

The current Beam8/KSVD performance search is therefore closed.  Generic
multi-cover SSL, balanced cover engineering and prototype interpretation
remain research side routes.  See
`KSVD_MOLHIV_REMAINING_ROUTES_TERMINAL_20260816.md`.
