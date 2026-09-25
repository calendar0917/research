# E2E-DictEnv-A2-Lite — analysis (minimum paired frozen-OMP screen)

Round `E2E-DictEnv-A2-Lite` (protocol `e2e_dictenv_a2_lite`), the
user-requested compute-budget continuation of the truncated round
`E2E-DictEnv-A2`.  Rules frozen *before* execution in
`tracks/ksvd/notes/e2e_dictenv_a2_compute_amendment.md` (rules commit
`457c276`; boundary record `02bc3f5`; run commit `4f184dd`).
Remote physical **GPU1 only** (`CUDA_VISIBLE_DEVICES=1`, one CUDA process at a
time), seed 0, official ZINC **test never loaded**.

Frozen verdict: **`NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D`**

---

## 1. Provenance chain (nothing re-derived)

`lite_verify.json` (3.0 s, `passed = true`) attaches this round to the
identity-verified parent artifacts *by reference*:

| link | value |
|---|---|
| parent identity | `results/e2e_dictenv_a2/artifact_identity.json`, sha256 `1b967746…`, `all_passed = true`, 26/26 entries, `git_commit 24d528635fab49c082115d90592cb7d5938eeb37`, `preregistration_commit 1813f53` |
| referenced entries required and re-checked | `cache_train_phi`, `cache_valid_phi`, `scaler_real`, `scaler_indep`, `dictionary_{REAL,INDEP}`, `omp_{REAL,INDEP}_{train,valid}`, `matched_init_{REAL,INDEP}` — all `passed = true` |
| producer freeze | sha256 of `e2e_dictenv_a2.py`, `zinc_e2e_dictenv_a2.py` and the frozen A2 preregistration all match their pins |
| dictionaries (live) | REAL `c1cafb086662fb0753d52987fc321b1369d4f586164dde7960a3bed775d4b809`, INDEP `400821ee5105050eb34600a0fb4cb8040f983daa1e773738dc9e2774036ff32d` |
| frozen OMP codes (bytes) | `omp_REAL_train` / `omp_REAL_valid` / `omp_INDEP_train` / `omp_INDEP_valid`, shapes `[231664, 32]` / `[23083, 32]`, file sha256 recorded; **not recomputed** |

No A1 cache, scaler, dictionary or OMP code was rebuilt in this round.

Every arm artifact records `horizon = 160`, `frozen_dictionary = true`, the
arm's frozen dictionary sha, `official_test_loaded = false`, and
`protocol_version = e2e_dictenv_a2_lite`.

## 2. Screen — frozen exact-OMP, 160 epochs, matched protocol

Same H1 trainer as the parent round (seed 0, identical init, batch order
`SEED + 91011` / `+ 91012`, identical optimizer/LR/weight-decay/clip, λ
33.95873017865987, Top-5 soup rule, frozen dictionaries and codes).

| arm | soup valid MAE | best valid MAE (epoch) | soup members | wall |
|---|---:|---|---|---|
| `ATTR-REAL-OMP` | **0.15437116196932038** | 0.16085957256139954 (158) | [149, 150, 153, 155, 158] | 1204.5 s |
| `ATTR-INDEP-OMP` | **0.15993232336913935** | 0.16623578738694778 (145) | [130, 145, 146, 156, 157] | 1303.4 s |

```
G_pair_screen = MAE(INDEP-OMP) - MAE(REAL-OMP) = +0.005561161399818965   (REAL better)
```

Late window (epochs 121–160, paired):

| statistic | value | frozen bar |
|---|---:|---|
| positive fraction of `delta(e) = MAE_INDEP(e) - MAE_REAL(e)` | 0.65 | ≥ 0.75 ⇒ **failed** |
| mean delta | +0.003818 | > 0 ⇒ passed |
| best-epoch delta in window | +0.005376 | > 0 ⇒ passed |
| `direction_stable` | **false** | |
| strong-positive (≥ 0.006 **and** stable) | **false** | |
| decision | `PCA_DIAGNOSTIC_AUTHORISED` (cheap dense diagnostic) | |

Whole-screen context: the paired per-epoch delta is positive in 57.5 % of the
160 epochs with mean +0.002038 — i.e. the REAL advantage is small and noisy
epoch-by-epoch, and it is carried mostly by the soup (mean of the five best
late epochs) rather than by a stable per-epoch ordering.

## 3. Cheap dense rank-32 diagnostic (frozen train-only PCA)

`ATTR-{REAL,INDEP}-PCA32`: `D = Vᵀ` of the train-only rank-32 PCA of the same
normalised 433-D object (delegated to `sdb_v0.fit_pca_rank` through
`e2e_dictenv_a2.fit_dense_rank`), centred codes attached exactly as the frozen
OMP codes are, tied decoder as in the frozen model, 160 epochs, same trainer.

| arm | soup valid MAE | best valid MAE (epoch) | soup members | wall (fit + 2 arms) |
|---|---:|---|---|---|
| `ATTR-REAL-PCA32` | **0.15774420121958246** | 0.1609946350780665 (153) | [145, 151, 152, 153, 154] | 2801.3 s |
| `ATTR-INDEP-PCA32` | **0.15633950045978418** | 0.16135399029304973 (153) | [143, 144, 145, 147, 153] | — |

```
G_pair_PCA = MAE(INDEP-PCA32) - MAE(REAL-PCA32) = -0.0014047007597982886   (INDEP marginally better)
                                                   frozen bar +0.003 ⇒ failed
```

Reconstruction asymmetry (context, not a gate): the frozen OMP code
reconstructs REAL's object far worse than INDEP's (`train_rec`
`1.965e-01` vs `2.119e-02`), while the rank-32 PCA is the better *affine*
reconstruction of REAL (`valid_affine_normalized_rec` `0.1119` vs the sparse
`0.197`) and of INDEP (`0.0163` vs `0.0212`).  The tied convention the frozen
model actually evaluates is mean-free (`valid_tied_normalized_rec` REAL
`0.5862`, INDEP `0.4935`), which is why the dense arms train with a large and
roughly constant reconstruction term; both dense arms carry it equally, so the
paired comparison stays fair, but dense absolute levels are **not** comparable
to the sparse levels.

## 4. Frozen verdict and the boundary case (read this before quoting the label)

The amendment's rule maps this evidence to
**`NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D`**: the sparse screen did not clear
the 0.006 strong bar with a stable late direction, and the dense control did not
clear the 0.003 bar either.

Two honest qualifications, both recorded here rather than folded into the label:

1. **The sparse screen was positive, just below the amendment's strong bar.**
   `G_pair_screen = +0.00556` is above the *parent* round's `MATERIAL = 0.003`
   gate yet below the amendment's own `0.006` strong bar, and the late-window
   agreement (0.65) missed the frozen 0.75 stability bar.  The frozen rule
   therefore did **not** declare `PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION`.
2. **Neither sparse arm had converged at epoch 160.**  REAL's best epoch is
   158/160 (last-epoch value 0.16626), INDEP's 145/160 (0.17270); mean late
   (121–160) values are 0.17690 vs 0.18071.  A 320-epoch run is therefore not
   guaranteed to keep this ranking, and the frozen parent Stage-1 gate
   (`G_pair_OMP` at the frozen 320-epoch horizon) remains **unanswered** — the
   parent round was truncated after its TOPO arm and never ran its REAL/INDEP
   arms.

The dense result also rules out the alternative story: the REAL advantage is
**not** recoverable by a rank-32 linear compression (dense arms tied; soup gap
−0.0014 favouring INDEP, best-epoch gap +0.0004 favouring REAL), so
`ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK` is not supported either.
Notably, REAL's dense arm is *worse* than its sparse arm (0.15774 vs 0.15437)
despite the denser code reconstructing its object better — the sparse REAL code
carries task information that the top-32 PCA directions do not, in this
protocol.

## 5. Independent local recomputation

`uv run python tracks/ksvd/code/analyze_e2e_dictenv_a2_lite.py --write`
recomputes, from the pulled artifacts only, both gaps, the paired late-window
direction, the frozen label mapping and every provenance flag, and compares them
with the remote decision records: **0 errors** (`analysis.json`).  Values match
to 1e-12 (the printed gaps are bit-identical).

## 6. What was *not* run (deferred, not cancelled)

TOPO-OMP predictor baseline; IHT-10/30/100/200 coder qualification;
task-coupled E2E sparse dictionary (E0/E1/E2); REAL→INDEP code-pairing-removal
mechanism; node-only / edge-only interventions; DenseTied specificity control;
seed 1; official test; K64; s12; any architecture or hyper-parameter sweep.
Cost of the round actually executed: 42.5 min (screen) + 46.7 min (dense) +
3 s (verify) ≈ **94 min of GPU1 wall clock**, two arms each per step, no sweeps.

## 7. Options the user owns (nothing started)

* **Buy the frozen A2 Stage-1 answer**: the two remaining 320-epoch frozen-OMP
  arms (REAL, INDEP) ≈ 1.4 h of GPU1 at the measured 7.1–8.1 s/epoch, giving
  the parent round's pre-registered `G_pair_OMP ≥ 0.003` verdict and TOPO's
  already-complete 320-epoch arm as the secondary read.
* **Buy a second seed of the 160-epoch screen** to estimate screen noise before
  spending on 320 epochs — this needs a new pre-registration (the amendment
  defers seed 1).
* **Stop here**: this round produced no confirmed attributed-code-formation
  signal at 32-D, so no capacity study (K/s), coder qualification or E2E
  dictionary work is authorised by these numbers alone.

## 8. Reproduction

```bash
# remote (GPU1 only), commit 4f184dd
bash scripts/launch_remote.sh 1 a2lite-all python -m \
  tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_a2_lite all --device cuda

# local
bash scripts/pull_results.sh tracks/ksvd/results/e2e_dictenv_a2_lite
uv run python tracks/ksvd/code/analyze_e2e_dictenv_a2_lite.py --write
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_a2_lite.py
```

Official ZINC test data was never loaded; `official_test_loaded = false` in
every artifact.
