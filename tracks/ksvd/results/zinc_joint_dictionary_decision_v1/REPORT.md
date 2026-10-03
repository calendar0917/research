# REPORT — zinc-joint-dictionary-decision-v1

Step start `2026-10-03 00:48:19 UTC`; compute cutoff `06:03 UTC`.  All results
below are reproducible from the committed sources + the force-added prep blob;
the official ZINC **test** split was never instantiated, loaded or evaluated in
any code path (`official_test_loaded: false` is emitted by every artifact).

## 0. Headline

1. **The severe long-cycle failures are not a topology25 key ambiguity.**  The
   raw 25-D and the standardised 25-D topology keys induce the *identical*
   partition of the 10000 train rows; the query rows that fail have **no**
   exact train match, and the one train row whose topology key is bit-identical
   to `valid:0172` is nonetheless predicted 21 MAE away.  Exact-class median
   lookup transfers perfectly on the bulk (penalty 0/−1) but cannot cover the
   severe rows.
2. **Moving the dictionary to a fixed local description does not improve
   generalisation.**  On the fixed 2000-row molecule-level dev split (seed 0,
   240 epochs, matched recipe) the calibrated MAE is `F 0.118018`,
   `B 0.128883`, `D 0.124505`.  The C3 gate fails at the first clause:
   `D − F total cal gain = −0.006487` (needs `≥ +0.003`).
3. The sparse dictionary is nevertheless a **working, correctly-implemented
   mechanism**: it beats the equal-capacity plain MLP on the same input
   (`D − B = +0.004378`) and reconstructs X175 at 1.26 % relative error.  The
   loss is not in the encoder; it is that the fixed X175 + small decoder is a
   weaker function class than F's learned local environment, and the deficit is
   concentrated in the bulk G0 rows.

Gate result: **FAIL → seed 1 not bought, Stage D not run.**

## 1. Setup, provenance and guardrails

* Local branch `task/zinc-joint-dictionary-decision-v1`; remote `res-2`
  (`res2-cu124`, A100-PCIE-40GB, driver 525.85.12, torch 2.5.1+cu124).
* Total remote GPU time ≈ **0.95 GPU-hours** (F 1363 s, B 720 s, D 941 s,
  smokes/probes ≈ 300 s), well under the 6 GPU-hour ceiling.
* Recipe identical to the canonical Full run: Adam lr 1e-3, coupled wd 1e-5,
  batch 128, clip 5, FP32, no scheduler / AMP / DDP, fresh init, fixed last-5
  soup (epochs 236–240), fit-only median-residual bias.
* Fold-internal discipline: every train-fit object (the four standardisers, the
  sdb32 dictionary, the common subspace, the B1 X175 normaliser, the D
  dictionary and `lambda_rec`) was fit **on the 8000 fit molecules only**.  A
  remote `zjd-verify` job recomputed X175 from the shipped blob and matched the
  local `prep_meta.json` hashes bit-for-bit for fit / all-train / valid.
* No message passing, no transformer/attention, no pair→centre update.  The
  local environment, shared dictionary and distance relations are preserved.

### 1.1 Split

`canonical_group_id`, seed 20261003, 20 % dev, stratified by raw cycle penalty:

| stratum | rows | dev rows |
|---|---|---|
| penalty 0 | 9628 | 1926 |
| penalty −1 | 325 | 65 |
| penalty ≤ −2 | 47 | 9 |
| **total** | **10000** | **2000** |

No group straddles fit/dev; no group has inconsistent penalty.

## 2. Stage A — is the severe-cycle failure a topology25 ambiguity?  (Q1: no)

### A1 — identity / accounting (re-play of `N0_s0`)

Replaying the published soup state reproduces `raw = 0.11073645`,
`cal = 0.11120612` (published `0.11120613`, Δ ≈ 1e-8), train bias
`−0.01204258`.  The per-row re-accounting sums to the published total exactly.

Severity strata of the 1000 valid rows:

| stratum | n | MAE | contribution |
|---|---|---|---|
| penalty 0 | 965 | 0.0871300 | 0.0840804 |
| penalty −1 | 30 | 0.1547958 | 0.0046439 |
| penalty ≤ −2 | 5 | 4.4963655 | 0.0224818 |
| total | 1000 | 0.1112061 | 0.1112061 |

The frozen row-set partition `G0 / G1_ex172 / G172` is a *different* partition
of the same rows (0.0840804 / 0.0084035 / 0.0187222).  The overnight REPORT §5
printed the severity numbers under the row-set labels — see `ERRATA.md` E1.

### A2 — exact topology25 classes

* `topo_raw` and `topo_model_input` induce the **identical** partition
  (`raw_nrm_class_partition_identical = true`, 355 classes; standardiser affine
  residual ≤ 6.5e-7).  A standardised key cannot separate rows the raw key
  merges, so "the model sees a normalised key" is not a source of confusion.
* Train penalty is pure in **350/355** classes; 5 classes are mixed (13 rows,
  e.g. `[-5,0]`, `[-6,0]`, `[-2,0]`), none of which contains a query row.
* Of 70 query × key-space pairs: 56 `MATCHED`, 14 `NO_EXACT_MATCH`.  The 7
  valid ring graphs **214, 229, 249, 403, 478, 804, 935** have no exact train
  key; three of them (214, 249, 935) are among the severe five.
* **Decisive pair:** `train:3776` and `valid:0172` have bit-identical raw *and*
  normalised topology keys, both penalty −6, class size 1 — yet the control
  predicts them 21 MAE apart.

### A3 — train-only probe (molecule-level, 2-fold, canonical group)

Input = raw `topo_raw` 25-D, target = raw train cycle penalty.

| probe | OOF MAE | valid MAE | valid G0 exact | valid severe exact |
|---|---|---|---|---|
| exact-class median lookup | 0.00236 (97.6 % covered) | 0.0 (98.8 % covered) | 0.995 | 0.40 |
| ExtraTrees (256/leaf2) | 0.00880 | 0.00690 | 0.990 | 0.20 |

**Verdict (Q1).**  Penalty 0/−1 topology signal is strongly learnable and
transfers; the severe ≤ −2 stratum is *not* recoverable from topology alone
(ET valid MAE 0.885 on those rows), and the failing rows are precisely the
no-exact-match rows.  The severe failures are therefore **not** caused by
topology25 equivalence-class confusion; they are a small set of rows whose
target is not a function of the local topology key at all.

**T25 ambiguity's share of the valid error budget is exactly zero.**  The only
ambiguous objects are the 5 mixed train classes (13 of 10000 train rows); **no
valid row** lands in a mixed class
(`cycle_input_classes.csv` → `n_class_penalty_mixed = 0` for all 70 query ×
key-space pairs), and every failing valid ring row is a *no-exact-match* row,
not an ambiguous-match row.  Ambiguity can therefore account for none of the
0.111206 published valid error; the severe rows' 0.022482 contribution comes
from the model failing on rows the topology key simply does not determine.

## 3. Stage B — the fixed prototype and its availability

`X175 = [raw φ65 ; raw Sem108 ; raw size2]` (one row per root), B1-normalised on
the fit molecules (per-column mean/std, floor 1e-3; each block divided by its
standardised block RMS).  B/D replace the **entire** old local-environment
module (structural binding, fusion, post-fusion bridge); pair projection,
relation/distance, unary/pair moments, C6 mask, global/topology branches and the
reader (`R=814`) are shared and initialise to the same tensors.

| arm | local module | local params | loss |
|---|---|---|---|
| F | canonical learned environment | 311,338 | `L1 + H1_LAMBDA·recon` |
| B | `175→256→SiLU→144` MLP | 82,064 | `L1` |
| D | 175×256 dictionary (K=256, s=64, 10 IHT) → `256→144` | 81,808 | `L1 + λ·relative-recon` |

B3 checks (`availability_checks.json`, `mechanism_ok = true`):

* dims X175 175, E 144, reader 814; shared backend hash `a2b15342…` identical
  for F/B/D; batch-composition invariance max |Δ| = 0;
* D reconstruction: fit **0.0126**, held-out **0.0126** (blocks φ 0.023,
  sem 0.081, size 0.0011) — far inside the 0.30 / 0.50 thresholds;
* α active (`max|α| = 3.20`, mean column std 0.056); task and rec gradients
  reach the dictionary (0.152) and value matrix (0.449); outputs finite.
* `lambda_rec = 20.0048` (4-batch one-shot calibration; rec/task gradient ratio
  ≈ 5.0e-3).

## 4. Stage C — seed 0, 240 epochs, fixed dev

| arm | dev raw MAE | **dev cal MAE** | fit cal MAE | bias | wall |
|---|---|---|---|---|---|
| F | 0.116365 | **0.118018** | 0.040495 | −0.013531 | 1363 s |
| B | 0.127340 | **0.128883** | 0.043627 | −0.019516 | 720 s |
| D | 0.124460 | **0.124505** | 0.049061 | −0.006411 | 941 s |

### 4.1 C3 gate (severity strata on the 2000-row dev)

| check | required | measured | verdict |
|---|---|---|---|
| D − F total cal gain | ≥ 0.003 | **−0.006487** | FAIL |
| D − F G0 cal gain | ≥ 0.002 | **−0.006099** | FAIL |
| D − F severe-contribution worsening | ≤ 0.001 | −0.000181 | pass |
| D − B severe-contribution worsening | ≤ 0.001 | −0.001788 | pass |
| mechanism available | true | true | pass |

**The gate fails at the total and G0 clauses.**  D is worse than F, and the
deficit is in the bulk group (penalty 0: F 0.10048 vs D 0.10658 vs B 0.11036).

Seeded paired bootstrap on the 1926 G0 rows (2000 resamples):

| contrast | mean error delta | 95 % CI | D-win rate |
|---|---|---|---|
| D − F | **+0.00610** | [+0.00147, +0.01053] | 47.3 % |
| D − B | −0.00379 | [−0.00848, +0.00090] | 51.8 % |
| B − F | +0.00989 | [+0.00580, +0.01426] | 46.6 % |

So D is **significantly worse than F**, and its edge over B is positive but not
significant.  The severe group (9 rows, penalties `[−2,−2,−12,−2,−2,−5,−2,−2,−2]`,
targets down to −42) contributes 0.0156–0.0174 and is dominated by two rows;
D's small advantage there is not evidence.

## 5. Where the prototype loses (failure localisation)

The five candidate links, adjudicated against the evidence:

| link | verdict |
|---|---|
| **encoding / reconstruction** | **ruled out** — D recovers X175 at 1.26 % rel. error, held-out = fit; α is sparse and active. The dictionary is not the bottleneck. |
| **external distribution shift** | **ruled out** — dev is a held-out subset of the same train distribution. |
| **input** | **contributing** — X175 is a fixed *subset* of the channels F's learned environment consumes; F additionally binds atom/bond category identity and the remaining patch/anchor channels. |
| **omitted correspondences** | **contributing** — the fixed description flattens the environment into one vector and drops the learned atom↔centre / bond↔centre bindings that F's `W_A_S/W_A_C/W_E_S/W_E_C` provide. |
| **task learning** | **the measured deficit** — the fixed decoder fits fit-data (train 0.078) but generalises ~0.0065 cal MAE worse; the gap is in the function class, not the optimiser. |

The one positive micro-signal: at **equal input and near-equal capacity**, the
sparse dictionary (D) numerically beats the plain MLP (B) by +0.0044, i.e. the
dictionary mechanism itself is a good decoder.  But D and B share X175 and
81–82 k local parameters, while F carries 311 k learned local parameters and
more channels.  The most parsimonious reading is that the fixed X175 + small
decoder is **capacity/information-limited** relative to F, and that the sparse
code cannot bridge a gap created upstream of it.

## 6. What was NOT bought (and why)

* **Seed 1** — C3 failed its total/G0 clauses by a wide margin; buying a paired
  seed could not move a −0.0065 deficit past a +0.003 threshold.
* **Stage D** (full-10000 refit, F/D × seed 0/1) — same reason.
* No node-amplitude / weight-decay / reader-WD / ISTA-step / K,s,λ / warm-tail /
  topology-widening arm was bought; those configurations are closed.

## 7. Artifacts

* `cycle_input_audit.py`, `cycle_input_decision.json`,
  `cycle_input_classes.csv`, `cycle_class_members.csv`,
  `cycle_probe_predictions.csv` — Stage A.
* `availability_checks.json`, `prep/fold_objects.npz`, `prep/*.json` — Stage B
  prep and availability.
* `F_seed0.json`, `B_seed0.json`, `D_seed0.json`, `gate_seed0.json` — Stage C.
* `ERRATA.md`, `DECISION.md`, `EXECUTION.md` — corrections, decision, provenance.
* Code: `tracks/ksvd/experiments/luyin16/zinc_joint_dictionary_decision_v1.py`;
  protocol `tracks/ksvd/protocols/zinc-joint-dictionary-decision-v1.yaml`.