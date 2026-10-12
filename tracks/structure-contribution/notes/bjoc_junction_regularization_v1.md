# bjoc-junction-regularization-v1 — design and pre-registration record

Round `cscl-bjoc-junction-regularization-v1` (track `structure-contribution`).
Pre-registration: `configs/bjoc_junction_regularization_v1.yaml` (committed before any formal
trajectory; the split row hashes, recipe, gate thresholds and trajectory plan are frozen there).
Driver: `code/run_bjoc_junction_regularization_v1.py`. Model: `code/bjoc_junction_dropout_v1.py`.
Tests: `tests/test_bjoc_junction_regularization_v1.py` (25 tests, all pass).

## Question

The previous round showed that adding typed structural channels improves fit while hurting
held-out molecules, and that removing learned junction content at matched capacity costs
+0.0198 locked-dev MAE (CI excludes zero). The remaining question is not "more structure" but
"less reliance on accidental training co-occurrence":

> Can one fixed training-time regularisation of the learned junction representation improve
> BJOC prediction on molecules it never saw, without changing the inference architecture, the
> supervision, or the structural inputs?

## The single intervention

Exactly one predictive variable is changed — elementwise inverted dropout on the learned
junction embedding `w`, applied **before** the static incidence grouping:

```python
w = self.encode_junctions(a, b, batch)
w_used = F.dropout(w, p=self.junction_dropout_p, training=self.training)
W, V, n_b = self.group_to_bonds(w_used, batch, mode)
```

* R0 `p=0.0` (bit-identical to the frozen BJOC-Null; `F.dropout(p=0)` is the identity and
  consumes no RNG).
* R1 `p=0.10`.
* No other probability, location, schedule or feature is authorised.
* Dropout precedes both moments, so it perturbs `mean(w)` and `mean(w**2)` — this is **not** a
  faithful simulation of deleting individual junction objects.
* Inference semantics: `model.eval()` gives `w_used == w`; models sharing a weight state are
  prediction-identical regardless of `p`. No inference capacity or input information changes.

Both arms must have exactly **413,505** parameters (the published BJOC backbone). The
extended 432,705-parameter `ExtendedBJOC` of the previous round and the TypedPathRing feature
cache are not read anywhere in this round.

## RNG boundary (recorded, not overclaimed)

* Same seed ⇒ same initial state hash for R0/R1 (verified in `parameter_and_init_audit.json`
  and re-verified per trajectory in `train_manifest.json`).
* Data order comes from an independent `torch.Generator(seed+101)`; the full 240-epoch batch
  index stream is hashed per trajectory (`batch_order_sha256`); R0/R1 must match, as must the
  total step count.
* R1's dropout consumes global PyTorch RNG during training; the whole global stream is
  therefore *not* bit-identical between arms. Global RNG-state hashes at epochs 1/40/120/240
  are recorded per trajectory for auditability. The paired quantities are the initial weights
  and the data order, not every stochastic op.

## Data (frozen before training)

Two new canonical-SMILES grouped internal splits, salted **a priori** (no salt search):

| split | salt | fit | monitor | locked |
| --- | --- | --- | --- | --- |
| A — discovery | `cscl-bjoc-jdrop-v1-discovery` | 8000 | 1000 | 1000 |
| B — replication | `cscl-bjoc-jdrop-v1-replication` | 8000 | 1000 | 1000 |

Row-set SHA256s are frozen in `configs/bjoc_junction_regularization_v1.yaml` and verified by
the driver and tests. Duplicate SMILES groups never cross boundaries (4 groups / 8 rows).

**Cross-split interpretation boundary.** Both splits come from the same official-train 10000
rows — they are not independent datasets. Observed row cross-table: B fit × A = 6419 fit /
778 monitor / 803 locked; B monitor × A = 800 / 110 / 90; B locked × A = 781 / 112 / 107.
Split B therefore replicates fit-set / partition sensitivity, not external data; its
locked-locked overlap with A is 107/1000. The salts were fixed a priori and the overlap was
**not** optimised.

**Label boundary.** Training reads only fit + monitor y. Locked-dev y is read exactly once per
split inside `score`, after every prediction file is hashed. `decide` reads metric JSONs only.
Split B's locked labels cannot be read in any split-A decision stage. No validation label ever
enters a scaler, a feature or a training loss.

## Recipe (identical for every arm)

Adam lr 1e-3, weight decay 1e-5, batch 128, grad clip 5, 240 epochs, FP32, no AMP, no
scheduler, raw-y L1, last-5 parameter soup (epochs 236-240), shuffle seed = seed+101. Global
features standardised on each split's fit rows only (objects cache read-only).

## Trajectories

| stage | plan | count |
| --- | --- | --- |
| split A | R0/A/s0, R1/A/s0, R0/A/s1, R1/A/s1 | 4 |
| split B (conditional) | R0/B/s0, R1/B/s0, R0/B/s1, R1/B/s1 | 4 |

Maximum 8 trajectories, 1 GPU, ≤ 2 GPU-hours, ≤ 7 h 45 m wall with ≥ 35 min reserved.

## Pre-registered split-A purchase gate (`JDROP_REPLICATION_PURCHASE`)

1. seed-mean monitor Δ = R1 − R0 ≤ −0.006;
2. both seeds' monitor Δ < 0;
3. both seeds' locked-dev Δ ≤ 0 **and** seed-mean locked Δ ≤ −0.003;
4. ordinary molecules (`y > −3`) seed-mean locked Δ ≤ 0;
5. not single-row driven: top-10-removed seed-mean locked Δ < 0 and each arm's worst-row error
   mass share < 0.05;
6. all dropout/correctness gates pass.

These are GPU-purchase rules, not significance definitions. A favourable point estimate with a
bootstrap interval crossing zero is recorded as directional with its uncertainty, and the
cross-seed spread is always reported separately (molecule bootstrap does not cover training
randomness).

## Phase A evidence used to justify the intervention (zero training)

* **Monitor-vs-locked difficulty of the completed representation round** is dominated by a few
  extreme low-y rows, not by a harder distribution: monitor y≤−3 fraction 0.076 vs locked 0.080;
  monitor top-10 error-mass share 0.345 vs locked 0.094; removing the top-10 rows moves monitor
  MAE 0.1407 → 0.0931, next to locked 0.0961; the monitor structural strata (atoms 23.1 vs 23.4,
  junctions 34.5 vs 34.9, diameter 12.4 vs 12.6) are comparable. This audit was used only to
  understand the old instrument; the new splits were fixed independently by salt.
* **N0/N1 full-10k trajectories** descend continuously to epoch 240 (soup fit 0.05210 / 0.03952),
  the last-5 soup is a small, well-defined change from the last epoch, and the junction encoder
  receives non-zero gradients on both frozen checkpoints (grad norms 1.36 / 0.79) — a
  training-time dropout on the junction embedding is an engineering-meaningful, testable knob.
* **Junction necessity (reused):** locked J−B = +0.019825, CI95 [+0.013324, +0.026108], band
  `JUNCTION_NECESSITY_SUPPORTED` — under that round's matched capacity and fixed recipe,
  removing learned junction content degraded held-out predictions systematically. This is not
  a claim that the junction construction is mathematically irreplaceable.

## Correctness gates before GPU (all 20 pass)

R0 ≡ frozen backbone forward (bitwise), both arms 413,505 parameters, identical init for
R0/R1, eval prediction identity under shared weights, R1 train-mode dropout non-zero and RNG
consuming, R0 train-mode identity and RNG-neutral, dropout only on `w`, eval preserves all
junctions, live junction-encoder backprop, both moments consume `w_used`, permutation
invariance, batch consistency, no new trainable parameters, single dropout call / no hidden
bypass, object cache unchanged, label independence, no NaN/Inf, save/load replay, and
official valid/test not referenced.

## Discipline

* `official valid` and `official test` are never loaded in any form (gate 20 scans the driver
  source; no prediction file of those splits exists in this round's results).
* The exposed historical official-test number 0.092233 is not used for any decision.
* No dropout-probability search, no location search, no architecture change, no new features,
  no distance-bucket change, no g/ell/s/c/k supervision, no extra trajectories.
