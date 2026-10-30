# REPORT — LCRC-A Lipophilicity screen v1 (pre-GPU)

- **Round:** `tracks/structural-sufficiency/lcrc_lipo_screen_v1/`
- **Final status:** `READY_FOR_GPU_AUTHORIZATION`
- **GPU:** not authorized, not used. `torch.cuda.is_available()` was `False`; no GPU
  training, prediction or hyperparameter run has been executed.
- **Monitor/Reserved/official labels:** not read, not scored.
- **Protocol freeze:** `analysis/protocol_freeze.json`
  (sha256 `b0b1bf08adc1e759be42bee1b28165230a66fd7e7d86f3aa2e601af56e68ed1c`)
- **CPU gates:** 9/9 PASS, final status `READY_FOR_GPU_AUTHORIZATION`
  (`analysis/cpu_gates.json`, sha256 `23028cac…7cbb`)
- **Tests:** 53 passed

This report is deliberately a **pre-GPU engineering report**. It contains no
Lipophilicity performance result because no formal A/B/C model has been trained and no
Monitor label has been read. Questions that require Monitor predictions are marked
`DEFERRED` with the exact frozen rule that will answer them.

---

## 1. What was done

1. Re-verified the first-round frozen assets: split, labels artifact, raw structures,
   `lipo_v1` FeatureSpec, Fit-only normalisation, and the Fit/Monitor caches. All
   hashes and row hashes match the first-round audit; caches contain no labels.
2. Implemented the full second-round pipeline without modifying first-round assets:
   selective Fit label loading, a Monitor entry point that is locked until a prediction
   manifest exists, runtime file-access denial, LCRC-A / S2-off / GINE builders, the
   frozen trainer, label-blind prediction writer, ECFP4+XGBoost baseline, prediction
   freezer and the frozen evaluator.
3. Wrote and ran 9 CPU gates and 53 pytest checks covering data isolation, model
   equivalence/ablation behaviour, GINE adjacency, optimizer/soup determinism, the
   baseline, the evaluation order and the runtime guard.
4. Ran three independent CPU smoke trainings (A, B, C; 256 Fit molecules, 12 epochs,
   24 updates each) to prove the loss actually decreases and no gradient path is dead.
5. Froze the protocol, config, code and test hashes, the parameter table and the
   budget estimate.
6. Produced arm D's three label-blind Monitor predictions locally (RDKit is unavailable
   on the GPU hosts). They are **not scored**; no prediction manifest exists.

## 2. CPU gate results

All evidence and timings are in `analysis/cpu_gates.json`.

| gate | result | key evidence |
|---|---|---|
| G0 frozen artifacts | PASS | spec `7270f767…`, norm `9c54f31c…`, caches `0a73407f…` / `db48a9c2…`; no labels; reserved cache not loaded |
| G1 label isolation | PASS | 2352 Fit values constructed, 504 Monitor skipped, 0 unrequested; row hash `2ccf20f5…`; Monitor blocked before freeze; no training module imports `monitor_labels` |
| G2 LCRC/S2-off | PASS | A/B identical init hash per seed; B exactly zero at LocalMLP columns `[320:520)`; A output changes 4.358 under S2 perturbation, B changes 0.0; A S2 grads > 0, B S2 grads exactly 0 |
| G3 GINE adjacency | PASS | 245,305 params; permutation invariance 5.96e-8; batch isolation 2.38e-7; rewiring Δ 0.0041; no-message-passing Δ 1.106; finite non-zero grads |
| G4 training numerics | PASS | 19 steps/epoch, 6840 updates; batch hash `1aeefe0f…`; save/restore next step diff 0.0; soup = manual mean; overrides rejected; CUDA blocked |
| G5 CPU smoke | PASS | A 146.13 → 1.118 MSE; B 95.05 → 1.133; C 412.65 → 1.225 (24 updates, 256 Fit rows); no Monitor access |
| G6 ECFP4+XGB | PASS | 2856 canonical SMILES verified; XGBoost refit bitwise identical (max diff 0.0) |
| G7 evaluation order | PASS | metric/CI/classification unit checks; `evaluate()` raises before the freeze; partial D predictions certified label-blind |
| G8 access control | PASS | CSV, official split, reserved cache denied; label grant works exactly once |

The CPU gates validate implementation and data discipline. They say nothing about
predictive quality.

## 3. Parameters and optimization steps

From `analysis/parameters.json`:

| arm | model | parameters | effective parameters | updates |
|---|---|---:|---:|---:|
| A | LCRC-A Full | 247,106 | 247,106 | 6840 |
| B | LCRC-A S2-off | 247,106 | **176,514** (71.43%) | 6840 |
| C | GINE + global42 | 245,305 | 245,305 | 6840 |
| D | ECFP4 + XGBoost | 500 trees × 3 seeds | n/a | n/a |

- GINE layers = 3; even-width search over 128…192 selected hidden 170 by
  `min |N_gine − 247106|`; no performance was consulted. The table
  (`hidden → parameters`) is in `analysis/parameters.json`.
- B's 70,592 dead parameters consist of the S2 modules (51,392) and the first-LocalMLP
  weight columns feeding the zeroed slice (19,200). A vs B is therefore an ablation of
  the S2 input channel under unequal effective capacity, not an information-matched
  control.
- All three neural arms use the same seeds, data order offset, optimizer steps, loss and
  soup rule.

## 4. CPU smoke (not a scientific result)

`analysis/smoke.json` (sha256 `d67839ce…`) records every smoke run:

| arm | run id | updates | train MSE first → last | ratio | wall |
|---|---|---:|---:|---:|---:|
| A | `smoke_A_s0_n256_e12` | 24 | 146.133 → 1.118 | 0.0077 | 0.73 s |
| B | `smoke_B_s0_n256_e12` | 24 | 95.049 → 1.133 | 0.0119 | 0.74 s |
| C | `smoke_C_s0_n256_e12` | 24 | 412.652 → 1.225 | 0.0030 | 0.75 s |

All three curves are monotone in trend after the first few updates; 23–24 of 24 steps
hit the gradient clip, which is expected for a randomly initialised sum-readout model
at peak LR and is not a failure. `batch_order_matches=true`, `nonfinite_events=0`, and
`soup` was produced for each run. These losses use Fit targets only and **cannot rank
the arms**; no Monitor score was computed.

## 5. Arm D predictions (label-blind, unscored)

Three files `analysis/predictions/D_r{0,1,2}.npz` were produced locally:

- 504 rows in the frozen Monitor order; 500 boosted trees; no hyperparameter search.
- `monitor_labels_read=false` in every sidecar; no manifest/lock was written.
- Prediction mean/std: `2.221/0.715`, `2.217/0.715`, `2.214/0.713` (raw logD scale).
- Fingerprint hashes: Fit `376095a2…`, Monitor `2d7b045c…`; all 2856 canonical SMILES
  were checked against the frozen row manifest.
- They remain unscored. Scoring would require the A/B/C predictions and a completed
  prediction freeze.

## 6. Budget estimate

`analysis/budget_estimate.json`: anchor 235.96 s / 9720 updates on an A100
(0.02428 s/update); A/B ≈166 s per run, C ≈332 s (2× factor); nine neural runs
≈0.553 GPU-h; +20% overhead and 0.07 h reserves → **0.734 GPU-h** estimated, versus a
frozen internal hard cap of **1.5 h** and a researcher-suggested external cap of
**4.0 h**. No GPU hours were used. The 4-hour figure is a cap, not an authorization.

---

## 7. Answers to the seven required questions

### Q1. Did LCRC-A actually learn the Lipophilicity supervised task?

**Status: DEFERRED.** No formal A/B/C training has run, and Monitor labels are sealed.

What is known: the training path optimizes (smoke loss falls 100× on 256 Fit
molecules), gradients reach every live parameter, and the checkpoint/soup path is
deterministic on CPU. That is an implementation check, not evidence of generalization.

The frozen answer rule: A "learned" iff its mean Monitor RMSE is below the Fit-mean
constant predictor **and** each of seeds 0/1/2 is below that constant. The constant
baseline is computed inside `evaluate.py` from the Fit mean.

### Q2. Is the increment over S2-off stable?

**Status: DEFERRED.** Nine neural predictions and the prediction freeze are required.

The frozen answer rule: compute per-seed `ΔRMSE = RMSE(A) − RMSE(B)`, the paired
molecule bootstrap CI of the seed-mean Δ, then classify
`CLEAR_INCREMENT` / `SUGGESTIVE_NOT_MATERIAL` / `NO_CLEAR_INCREMENT` (PROTOCOL §6).
B's lower effective parameter count is reported alongside; a difference cannot be
attributed purely to S2 information.

### Q3. How does it compare with GINE and ECFP4?

**Status: DEFERRED.** No predictions to compare.

The frozen rule compares A with C and with D on the paired molecule bootstrap
(`A_MATERIALLY_BETTER` / `COMPETITIVE` / `A_MATERIALLY_WORSE` / `INCONCLUSIVE`),
materiality 0.03. GINE is a practical reference, not an information-matched control;
D is refit on Fit2352, never the previous BJOC train3360/valid420 numbers.

### Q4. Is performance sensitive to seeds or a few molecules?

**Status: DEFERRED**, but the measurement is defined. The evaluator reports all three
per-seed RMSEs, their mean/spread/range, the seed-mean paired bootstrap CI, a
within-seed vs between-molecule variance decomposition, top 1/2/5% error shares,
RMSE after dropping the worst 2/5%, and sign-flip checks on the tail. Best-seed
optimism is reported only as `best_seed_minus_mean_rmse`, never used for selection.

### Q5. Do the local sum and `rho` both participate in prediction?

**Status: IMPLEMENTATION VERIFIED; Monitor diagnostics DEFERRED.**

- Arm A's forward returns `contrib = sum_v f(h_v)` and `rho(...)`; the prediction is
  their sum. Prediction files for A include both terms and a
  `contrib + rho − pred` identity check.
- G2 confirms `f` and `rho` both receive non-zero gradients, and A's S2 parameters
  receive non-zero gradients while B's receive exactly zero.
- No A prediction exists yet, so the empirical `contrib`/`rho` scale and variation on
  Monitor504 cannot be reported. Any such numbers will be descriptive, not causal.

### Q6. What do these results support, and what do they not support?

**Supported now (engineering):**

- The frozen data isolation works: Fit labels align to rows, Monitor is unreachable
  from the training path, and the Reserved/official paths are denied.
- A and B share an identical initial state per seed; B's S2 block is exactly zero in
  train and eval while A uses it; the rest of the network is bit-identical.
- GINE messages along the real physical adjacency and is permutation-invariant to
  numerical precision; ECFP4 rows align to the frozen manifest and XGBoost refits
  deterministically.
- The optimizer-step schedule, save/restore and last-5 soup are reproducible on CPU,
  and the formal recipe cannot be silently overridden.
- The budget estimate fits the frozen hard cap.

**Not supported by this report:**

- Any claim that LCRC-A, S2, GINE or ECFP4 is better or worse on Lipophilicity.
- Any claim that Monitor will confirm the smoke loss descent.
- Any claim that S2 is the "core contribution" or that `contrib`/`rho` are chemically
  faithful.
- Any claim that this round is a fresh blind test (Monitor was used historically).
- Any inference from the CPU smoke loss values to model ranking.

### Q7. Is it worth entering SSL, mechanism ablation or independent-data validation?

**Status: NOT DECIDED.** A decision requires the frozen Monitor results. `DECISION.md`
records the evidence gate and the interpretation matrix; it deliberately does not
pre-commit to a next round. No next round has been started.

---

## 8. Limitations and disclosures

1. **No formal result.** Nothing in this report estimates Lipophilicity performance.
2. **Small evaluation set.** Monitor is 504 molecules; the molecule bootstrap describes
   sample uncertainty, not training-seed uncertainty. Both are reported separately.
3. **Three seeds only**, chosen before the run. Seed spread will be reported as-is.
4. **Monitor is a development set.** SSRC used it in earlier rounds; it is not fresh or
   fully blind. Reserved is not used at all, and official valid/test are never read.
5. **B is not information-matched.** Same nominal parameters, 71.43% effective.
6. **C is not information- or parameter-matched.** 245,305 vs 247,106 params; a
   different encoder family by design.
7. **GPU reproducibility.** LCRC-A uses `index_add_` atomic accumulation; same-seed GPU
   runs are not assumed bitwise identical. This will be disclosed, not averaged away.
8. **Pre-freeze implementation fixes** (S2 slice offset, GINE edge orientation, GINE
   reference count, RDKit generator API) were made before the freeze and before any
   Monitor value was read; see PROTOCOL §10. No protocol constant changed after freeze.
9. **`contrib`/`rho` diagnostics are descriptive**, not causal/chemical attribution.

## 9. Reproduction

See `README.md`. The complete pre-GPU state is reproducible with:

```bash
python code/run_screen.py verify-data
python code/run_screen.py freeze-protocol
python code/run_screen.py cpu-gates
python -m pytest tests -q
python code/run_screen.py baseline-ecfp
```

Expected: `READY_FOR_GPU_AUTHORIZATION`, 9/9 gates PASS, 53 tests pass. Formal
`train`/`predict-nn` CUDA commands are blocked until an authorization file exists.

## 10. Artifact index

| artifact | sha256 |
|---|---|
| `analysis/protocol_freeze.json` | `b0b1bf08adc1e759be42bee1b28165230a66fd7e7d86f3aa2e601af56e68ed1c` |
| `analysis/norm_target.json` | `50c5ad9da285b15233dc0f26704cfb5136ac4bbe5a97b4bf21734d8f1f9a0748` |
| `analysis/cpu_gates.json` | `23028cacbc7344681c1c65af0d02c3f12c019e9e17ec424bfdd6578031407cbb` |
| `analysis/parameters.json` | `e55d521adeb12a48ddc097c9bb9c9d5f8f9c076b1dd1fbf8b8e1e5468e37bf7c` |
| `analysis/budget_estimate.json` | `a9e6083e508bf8617f29b3092e42290a160aafa4726d6942798acd46fbe2ccae` |
| `analysis/smoke.json` | `d67839ceb87608a834f10c67c747fd21030b6aef391abe1292f92db4b80cdf70` |
| `analysis/data_audit.json` | `f240a661bc6d5396d1df89ab922e3f37bc1c281d322d2b1a1e0221368c25eb79` |
| `analysis/predictions/D_r0.npz` | `e2a18b9be03ed9e1da96614e9337abf2da577e14079d1a43958d295f736696b2` |
| `analysis/predictions/D_r1.npz` | `d2b42ebcc4a024d7e0dafa12e2c7672eb2650797361ba19a4e1704dc2c9be6c1` |
| `analysis/predictions/D_r2.npz` | `af4831f7dada243cde31f098e23dcfda4edc02e4582929dbd36ced9e49f46b78` |

Hashes of every code and test file are recorded in `analysis/protocol_freeze.json`
(22 entries: 12 code modules, 9 test files, config).
