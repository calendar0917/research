# REPORT — LCRC-C1 structural-confusion audit and a minimal 3-hop static addition

Round: `lcrc-lipo-s3-locality-public122-s0-v1`
Directory: `tracks/structural-sufficiency/lcrc_lipo_s3_locality_public122_s0_v1/`
Training commit: `311306ecfd2cdad27b784a329f2f1fe513fe978b` (host `res`, A100-SXM4-40GB, torch 2.5.1+cu124)
Formal trainings this round: **2** (LOCAL3, DUP2). NULL3 **not purchased** by the fixed rule.
GPU cost: **0.2014 h** recorded by `runtime.json` for the two formal trainings plus the
smoke; ≈**0.207 h** including the short check/intervention jobs (soft cap 1.0 h, hard cap 2.0 h).

---

## 1. Answers to the two propositions

**H_rep — supported.**
The five frozen pairs are confirmed, at the level of the model's own forward, to be
non-isomorphic connected structures that receive the *same* C1 input: identical
per-root sufficient-signature multiset, identical `global42`, and identical reused-A
predictions (|Δpred| ≤ 3e-7 ≪ 2e-5). A single typed exact-distance-3 statistic
separates exactly four of the five; the fifth (1467/2674) remains identical, as
designed; and the same-capacity control built from S2 end-marginals (U2) separates
none. Labels: `C1_INPUT_ALIAS_VERIFIED`, `S3_INPUT_SEPARATION_VERIFIED`.
This is a representation-boundary result for this C1/`lipo_v1` configuration only.

**H_pred — not supported.**
LOCAL3 is slightly *worse* than the reused ordinary C1 baseline on the primary
metric (full Test420 raw RMSE 0.58882 vs 0.57617, gain −0.0127, paired component
bootstrap 95% CI [−0.0475, +0.0161]), and slightly *better* on Valid (0.68451 vs
0.70254, gain +0.0180, CI [−0.0286, +0.0582]). The two splits disagree in sign and
both intervals cross zero. Labels: `NO_PREDICTIVE_GAIN_THIS_CONFIG` (primary
metric) together with `PERFORMANCE_UNRESOLVED_THIS_RUN` (direction inconsistency).
No increment over existing information is established either: LOCAL3 beats DUP2 on
Test only through DUP2's single catastrophic molecule (§7), with a CI that spans
zero, so `NEW_INFORMATION_INCREMENT_UNRESOLVED` would be the strongest statement
and it is not upgraded.

**Mechanism.** `ROOT_ALIGNMENT` was **not purchased**: the fixed Stage-II rule fails
at its first condition (LOCAL3 Test RMSE < A). The trained LOCAL3 clearly *uses* its
correction branch (disabling it moves Test RMSE from 0.589 to 0.806), yet it barely
separates the four input-separable pairs (|Δpred| ≤ 0.021, i.e. 0.5 %–3 % of the
label gap), so the case-level record is `FEATURE_AVAILABLE_NOT_USED` — the
information is in the input, but this training did not convert it into a case-level
response.

---

## 2. Main results (raw logD units, 4200 molecules)

| arm | best epoch | params | Train RMSE | Valid RMSE | Test RMSE | Test MAE | GPU·h |
|---|---|---|---|---|---|---|---|
| A (reused, not retrained) | 244 | 228674 | 0.28825 | 0.70254 | **0.57617** | **0.42806** | 0.0866 (previous round) |
| LOCAL3 (`h+ψ(h,s3_v)`) | 210 | 233378 | 0.31207 | **0.68451** | 0.58882 | 0.44218 | 0.0943 |
| DUP2 (`h+ψ(h,s2_v)`) | 260 | 233378 | **0.27416** | 0.68941 | 0.77090 | 0.43854 | 0.1032 |
| NULL3 (mean field) | — | — | — | — | — | — | not run |

Best checkpoints are chosen on Valid420 raw RMSE only (strict improvement, ties keep
the earlier epoch); Test never selects an epoch. `best.pt` reload reproduces the
recorded Valid value to 7.5e-10 (LOCAL3) and 2.9e-08 (DUP2); the curve argmin equals
the recorded `best_epoch` (210, 260). The last shared epoch 360 (§4) is reported for
progress inspection only.

Both new conditions reproduce the frozen plumbing exactly: base init state hash
`aac1e061…`, one shared 4704-parameter corrector initialisation, 233378 = 228674 +
4704 parameters, batch-order sha256 `b10278f960300fef…` equal to the baseline
manifest, 10476 dropout calls in 360 epochs (the same count as the baseline A run),
and an **exact epoch-1 dropout-RNG match** against the reused baseline trace on the
same device (`rng_match = True` for both arms). Gradient clipping fires on ~50 % of
steps, as in the baseline.

The saved raw-unit predictions were re-checked independently: `train/valid/test_rows`
equal the published split row ids in order, the stored labels equal the source labels,
and a NumPy recomputation of RMSE from `predictions_best.npz` equals the `train.json`
metrics and `results.csv` to <1e-12 for both arms (raw conversion applied exactly
once). Source-file, split and feature-spec hashes for all of the above are listed in
`PROTOCOL.md` §1.

## 3. Paired comparisons (label-free components, 1000 replicates, seed 20261114)

`gain = metric(baseline) − metric(candidate)`; positive means the candidate is
better. Components = the previous round's `eval_component` merged with this round's
identical-full-input groups (3626 components; 2954/412/407 per Train/Valid/Test).

| comparison | split | gain | 95 % CI | reading |
|---|---|---|---|---|
| LOCAL3 vs A | Test RMSE | **−0.01265** | [−0.04753, +0.01605] | LOCAL3 worse, not resolved |
| LOCAL3 vs A | Valid RMSE | +0.01803 | [−0.02862, +0.05818] | LOCAL3 better, not resolved |
| LOCAL3 vs A | Train RMSE | +0.02382 | [+0.01759, +0.03019] | LOCAL3 fits the training set worse |
| LOCAL3 vs DUP2 | Test RMSE | +0.18207 | [−0.04063, +0.47994] | driven by one DUP2 molecule (§7) |
| LOCAL3 vs DUP2 | Valid RMSE | +0.00490 | [−0.03641, +0.03673] | flat |
| LOCAL3 vs DUP2 | Train RMSE | −0.03792 | [−0.04370, −0.03233] | DUP2 fits the training set better |

The intervals are conditional on these three trained networks and say nothing about
training-seed repeatability. Crossing zero is not equivalence, and comparing Train
RMSE at different `best_epoch` values (210 vs 244 vs 260) is not evidence of an
"unavoidable objective conflict".

Per-molecule error correlation between A and LOCAL3 on Test is 0.884, and LOCAL3 is
closer to the label on 201 of 420 Test rows — i.e. no systematic redistribution.

## 4. Fixed sensitivities

| check | result |
|---|---|
| remove A's largest Test squared error (row 3986, SE 4.735) | Test RMSE A 0.56697, LOCAL3 0.58097, DUP2 0.76453 → LOCAL3 vs A **−0.01400**, LOCAL3 vs DUP2 +0.18356 |
| remove A's largest Valid squared error (row 2087, SE 85.48) | Valid RMSE A 0.53919, DUP2 0.53212, LOCAL3 0.54561 → LOCAL3 vs A **−0.00642**, LOCAL3 vs DUP2 −0.01348 |
| re-evaluate at the shared final epoch 360 | Test LOCAL3 0.61238, DUP2 0.78478; Valid LOCAL3 0.70924, DUP2 0.70371; Train LOCAL3 0.24510, DUP2 0.25149 |

No influence check flips the conclusion in LOCAL3's favour: the Test gain versus A
stays negative and the Valid gain versus DUP2 turns negative. Row 2087 (`y = 2.5`) is
a Valid molecule on which all three models fail badly (A −6.75, LOCAL3 −5.99,
DUP2 −6.50); it is not specific to this round.

## 5. The five frozen pairs

### 5.1 Input-level audit (real forward of the reused A checkpoint)

| pair | split | y_A / y_B | pred_A / pred_B | sig. multiset | global42 | raw graph | U3 sep. | U2 sep. | min SSE |
|---|---|---|---|---|---|---|---|---|---|
| 294/1665 | train/train | 1.74 / 2.20 | 1.6965513 / 1.6965516 | equal | equal | non-iso | **yes** | no | 0.1058 |
| 748/913 | train/train | 2.92 / 2.92 | 2.8214473 / 2.8214471 | equal | equal | non-iso | **yes** | no | 0.0000 |
| 963/4186 | train/**test** | −0.03 / 0.83 | 0.3711850 / 0.3711850 | equal | equal | non-iso | **yes** | no | 0.3698 |
| 1467/2674 | train/train | 3.90 / 3.80 | 3.5971546 / 3.5971546 | equal | equal | non-iso | **no** | no | 0.0050 |
| 2102/3823 | train/train | 0.98 / 1.06 | 1.0478598 / 1.0478599 | equal | equal | non-iso | **yes** | no | 0.0032 |

`min SSE` = Σ_g Σ_i (y_i − mean_g y)² for the identical-input group: this is the
fitting constraint this sample imposes on a model with this input, not irreducible
noise. Summed over all 86 identical-full-input groups it is 3.3048, against A's total
SSE of 625.9 over the three splits (0.53 %) — the conflation is real but small, and
the five cases are already-exposed mechanism examples, not an evaluation set.

Root-level block-by-block signature correspondence was verified for all roots of each
pair (32 / 25 / 31 / 37 / 31 roots). Forward identity: identical signatures give
identical local-input and `h` multisets (max |Δ| 1.2e-06 from float32 reduction
order). Of the 102 group members compared against their group representatives, 97 are
raw-attribute-graph isomorphic (stereo writing) and 5 are connectivity/attribute
non-isomorphic; one extra local-only group (837, 1200) is separated by `global42` and
is therefore *not* a full-input alias. No additional full-input alias pair exists
beyond the five.

### 5.2 What the U3 difference actually is

Because U2 is a function of the existing S2 end-marginal, the pairs can only differ
where the *distance-3* shell differs. Both separating classes of 294/1665, for
example, sit on the chain nitrogens: at exactly three bonds from root atom 3 (N),

* 294: N(3)–C(4)–C(5)–**C(6)**, N(3)–C(2)–C(1)–O(0), N(3)–C(2)–C(1)–C_ar(21)
  → U3 = {C:1, C_aromatic:1, O:1}
* 1665: N(3)–C(4)–C(5)–**S(6)**, and the same two outer paths
  → U3 = {C_aromatic:1, O:1, S:1}

and symmetrically at root atom 10. This is exactly the "heteroatom position along the
chain" difference visible in the two SMILES (`CNCCCSCCNCC` vs `CNCCSCCCNCC`). 963/4186
is the same pattern (the distance-3 end of one chain is C versus O), 2102/3823 the
same (C versus S versus O). 1467/2674 has **zero** signature classes with a differing
U3 multiset (30 classes each, all matched) — the sixth atom sits outside the 3-hop
neighbourhood of every root whose local signature is identical, so this pair is
invisible to U3 by construction.

### 5.3 Trained responses on the pairs (`as_trained`, raw units)

| pair | Δlabel | Δpred A | Δpred LOCAL3 | Δpred DUP2 | LOCAL3 response / Δlabel |
|---|---|---|---|---|---|
| 294/1665 | +0.46 | −0.000000 | −0.002557 | 0.000000 | 0.6 % (right direction) |
| 748/913 | 0.00 | −0.000000 | +0.020715 | 0.000000 | — (no label gap to fit) |
| 963/4186 | +0.86 | +0.000000 | −0.003924 | +0.000001 | 0.5 % (wrong direction) |
| 1467/2674 | −0.10 | −0.000000 | **0.000000** | −0.000000 | 0 (input still identical) |
| 2102/3823 | +0.08 | +0.000000 | +0.002437 | −0.000001 | 3.0 % (right direction) |

The input-level repair does not become a case-level repair: two of the four separable
pairs move in the correct direction but by ≤3 % of the label gap, one moves the wrong
way, and the U3-identical pair stays bit-identical. `FEATURE_AVAILABLE_NOT_USED` is the
accurate record; no claim of a repaired confounder is made.

## 6. Mechanism: what the branch is doing

Model-computation interventions on the trained LOCAL3 weights (one fixed weight vector;
not chemical edits and not causal effects):

| variant | Train | Valid | Test | prediction movement vs `as_trained` (Test RMSE / mean abs) |
|---|---|---|---|---|
| as_trained (LOCAL3 operator) | 0.31207 | 0.68451 | 0.58882 | — |
| correction_off | 0.64747 | 0.87173 | 0.80558 | 0.660 / 0.549 |
| NULL3 operator with LOCAL3 weights | 0.35459 | 0.69001 | 0.59305 | 0.186 / 0.121 |

Switching the branch off costs 0.217 Test RMSE, so the correction is load-bearing. Swapping
in the exact NULL3 mean field with LOCAL3's own weights and normalisation leaves the
aggregate metric close (0.593 vs 0.589) while moving individual predictions by 0.12 on
average — the per-root correspondence matters locally but does not show up as an
aggregate advantage here.

Correction magnitude and gradient flow (LOCAL3, last recorded batch / diagnostic steps):

* `‖ψ‖` mean 1.451 vs `‖h‖` mean 7.949 (18 %), max 8.60, on the final batch recorded by
  the training unit; over the whole Test split the intervention run measures
  `‖ψ‖` mean 1.315. DUP2 on its final recorded batch: 0.950 vs 7.947 (12 %).
* step 1: `‖∇W2‖ = 89.2`, `‖∇b2‖ = 65.5`, `‖∇W_h‖ = ‖∇W_s‖ = ‖∇b1‖ = 0` — the zero output
  projection blocks the feature path for exactly one step, as designed.
* step 28: `‖∇W_s‖ = 0.0269` and `‖∇W_h‖ = 0.0529` (non-zero after the first update);
  step 271: 0.0438 / 0.4039. Nothing is accidentally detached or frozen.

## 7. Descriptive: DUP2 is dominated by one molecule

Test row **508** (`y = 2.60`) is predicted as A 4.354, LOCAL3 0.287, DUP2 −8.209. That
single row contributes 116.8 of DUP2's 249.6 Test SSE; without it DUP2's Test RMSE
would be 0.563, i.e. better than both A and LOCAL3. Its U2 vector is unremarkable
(max 5 paths from one root; 37th of 420 Test molecules), so this is an optimisation
instability rather than an input-scale artefact. Consequently the LOCAL3-vs-DUP2 Test
"gain" of +0.182 is one molecule, not a distributional advantage; it is reported as a
point estimate with a CI that spans zero, and nothing is concluded from it. Neither
the protocol's fixed sensitivity (which removes *A's* worst row) nor this observation
changes the pre-registered decision.

## 8. Stage II: NULL3 not purchased

Fixed purchase rule, evaluated in raw units after Stage I:

| condition | value | pass |
|---|---|---|
| LOCAL3 Test RMSE < A | 0.58882 vs 0.57617 | **fail** |
| LOCAL3 Test RMSE < DUP2 | 0.58882 vs 0.77090 | pass |
| LOCAL3 Valid RMSE < A | 0.68451 vs 0.70254 | pass |
| after removing A's worst Test row, gain > 0 vs A and DUP2 | vs A −0.01400 | **fail** |

→ `NOT_PURCHASED_BY_FIXED_RULE`. NULL3 was not trained and is not reported as a
zero effect. For the record, the static degeneracy accounting that a NULL3 run would
have used: per split, 32.8 % (Test) / 31.4 % (Valid) / 32.8 % (Train) of
(molecule × atom_type) cells are degenerate (singleton, or all `s3` identical): Test 728/2219,
Valid 705/2243, Train 5862/17879 cells, i.e. 7.8 % / 7.6 % / 7.8 % of atoms sit in
such cells and NULL3 would coincide with LOCAL3 there.

## 9. Execution and budget

| stage | job | host/pool | commit | outcome |
|---|---|---|---|---|
| caches + features | `s3-prep` | res / res-gpu0 | `54da9a2f` | 3 caches rebuilt, fingerprints equal to the baseline, batch order equal |
| 2-epoch smoke (LOCAL3+DUP2) | `s3-smoke` | res / res-gpu0 | `54da9a2f` | 0.97 s / 0.88 s per epoch, 95 MB peak, RNG match |
| GPU operator/init checks | `s3-gpuchecks` | res / res-gpu1 | `4ac9e3c7`…`311306ec` | NULL3 optimised = naive (fwd 4e-07, grad 9e-04), shortcut differs by 0.43, zero-correction = C1 |
| Stage I LOCAL3 | `s3-local3` | res / res-gpu0 | `311306ec` | 335.8 s, exit 0 |
| Stage I DUP2 | `s3-dup2` | res / res-gpu1 | `311306ec` | 367.6 s, exit 0 |
| interventions | `s3-intervene` | res / res-gpu0 | `311306ec` | CPU-free GPU re-evaluation of the trained LOCAL3 |

GPU·h charged from process wall time: LOCAL3 0.0943, DUP2 0.1032, smoke 0.0039 →
`runtime.json` records **0.2014 h**; the three short check/intervention jobs add
≈0.006 h of wall time, so the round's real total is ≈**0.207 h**. Both figures are
inside the 1.0 h soft cap and the 2.0 h hard cap. One round job per card, one parallel round, no retries were needed and no
other task on `res` was touched. `res-2` was not used.

## 10. What this supports, and what it does not

Supported:

* In this C1/`lipo_v1` configuration, a fixed set of five structurally distinct
  molecules receives an identical model input and identical A prediction; the
  per-root signature introduced here is a sufficient description of the local
  readout and the conflation is therefore a property of the representation, not of
  the weights.
* One typed exact-distance-3 statistic separates four of those five at the input
  level and provably cannot separate the fifth; the S2 end-marginal control
  separates none.
* Adding that statistic as a zero-initialised correction changes generalisation only
  marginally, and in this single seed the primary-metric point estimate is worse than
  the reused baseline while Valid is better. The input-level repair did not become a
  case-level or aggregate predictive gain.

Not supported (and not claimed): unique atom contributions or real chemical
causality; that 5 pairs represent the population (the identical-input groups account
for 3.30 of 625.9 total SSE); cross-seed or cross-dataset stability; that a per-root
3-hop binding is either necessary or sufficient in general; that NULL3 would have
scored zero (it was not run); that DUP2's Test number is informative (one molecule);
that the mean-field operator swap is an "equivalence" of the two conditions.

## 11. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_lipo_s3_locality_public122_s0_v1
uv run --group lipo python code/prepare.py          # caches, U3/U2, batch order
uv run --group lipo python code/p0.py               # alias audit + A replication
uv run python -m pytest -q tests/                   # 21 key checks
rr doctor res && rr deploy res
rr run res s3-smoke --pool res-gpu0 --gpus 1 --cpus 8 --result <dir>/runs -- \
  uv run --group lipo python <dir>/code/train.py --smoke --device cuda --out-dir <dir>/runs/smoke
rr run res s3-local3 --pool res-gpu0 --gpus 1 --cpus 8 --result <dir>/runs -- \
  uv run --group lipo python <dir>/code/train.py --arm LOCAL3 --device cuda
rr run res s3-dup2   --pool res-gpu1 --gpus 1 --cpus 8 --result <dir>/runs -- \
  uv run --group lipo python <dir>/code/train.py --arm DUP2 --device cuda
rr run res s3-intervene --pool res-gpu0 --gpus 1 --result <dir>/analysis -- \
  uv run --group lipo python <dir>/code/intervene.py --device cuda
uv run --group lipo python code/evaluate.py         # results/comparisons/cases/runtime
```

`<dir>` = `tracks/structural-sufficiency/lcrc_lipo_s3_locality_public122_s0_v1`.
`runs/` and `cache/` are git-ignored (checkpoints, curves, predictions stay local;
`analysis/` JSON/CSV and this report are the committed evidence).

### Implementation notes (no reported number was produced before these fixes)

* `code/evaluate.py`: comparison keys were built as `"A_vs_LOCAL3"` while the decision
  block looked up `"LOCAL3_vs_A"`, so the performance decision could never fire; the
  gain sign convention is now `metric(baseline) − metric(candidate)` with the key
  `"<new>_vs_<baseline>"`. Fixed before any result was read.
* `code/intervene.py`: standardised predictions were scored against raw labels
  (double unit error, ~1.9× RMSE); the raw conversion is now applied once. Fixed
  before the reported intervention table.
* the zero-correction identity is exact on CPU and holds to 9.5e-07 on GPU (two module
  instances may select different cuBLAS kernels); the gate is 2e-6.
* `prepare.py` rewrites two tracked manifests (`created_utc`); after the remote prep
  run they were re-committed from the run's own bytes so the remote tree stayed clean.
* the old edit-supervision round's `train_diagnostics` pair-field unit mixing is not
  used anywhere in this round; per-molecule raw-unit predictions are the reference.
