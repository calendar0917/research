# REPORT — `lcrc-lipo-esa-random42-v1`

**Status:** `COMPLETED_SINGLE_SEED_BENCHMARK`

Four pre-registered models, one fixed split (random42), one training seed (0),
one terminal Test evaluation. All four models completed and were scored.

| identifier | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-esa-random42-v1` |
| `PROTOCOL.md` SHA256 | `707fcf63580862d2c6fea233840f926a7f82d99bab0783b9ee9c952493babac1` |
| `configs/benchmark_v1.yaml` SHA256 | `0d8d58eae2351294f09da21056b4d193e94cefccc8b79b94ec54ca247a2d8402` |
| canonical config SHA256 | `0bb3990f17ebe8be9ba8c43ef4f1efe3a33fe7aadc834272777f449c5d8e9c7f` |
| `analysis/pre_test_manifest.json` SHA256 | `5ebdfdaa5151476b641df5b065beb1616df1cb9cb59b73258bd6f1cea69976db` |
| frozen-model commit | `91955ee` (`r42 v1: freeze all four models before the single Test evaluation`) |
| GPU spend | **0.2960 / 6.0 A100 GPU-hours** (4.9 %) |
| Split | `torch.utils.data.random_split(range(4200), [3360,420,420], seed 42)` |
| Test evaluation | one pass only; lock `analysis/test_read.lock`; a second invocation raises `PermissionError` |

**This is a single-seed benchmark. Every comparison below is exploratory.** No
significance claim is made, no second seed exists, and no number from the
literature is quoted or mixed in.

---

## 1. Headline table (all metrics in raw logD)

| Model | Seed | Params | Train RMSE | Valid RMSE | Test RMSE | Test MAE | Test R² | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LCRC-A-R3-scratch | 0 | 247,106 | 0.1968 | 0.5890 | **0.5840** | 0.3917 | 0.7864 | 0.0919 |
| PNA-native | 0 | 7,767,349 | 0.1572 | 0.5540 | **0.5289** | 0.3738 | 0.8248 | 0.1334 |
| ESA-native | 0 | 10,351,105 | 0.3045 | 0.5729 | **0.5989** | 0.4269 | 0.7753 | 0.0671 |
| XGB-P3 | 0 | 800 trees / 19,473 leaves | 0.2871 | 0.6456 | **0.6570** | 0.4838 | 0.7296 | 0.0000 |

Reference points on the same Test420: constant-mean predictor RMSE **1.2635**,
Validation420 constant-mean RMSE **1.2294**.

XGB-P3 ran on CPU only: GPU-h = 0 exactly; its cost is **3.92 s** fit +
0.019 s inference (4.18 s total, one thread-efficient `hist` fit, `n_jobs=4`).

Epochs actually run: LCRC 360 (fixed, soup of the last 5 epoch-end states),
PNA 114 (early stop; best epoch 84), ESA 60 (early stop; best epoch 30),
XGB 800 boosting rounds.

---

## 2. Eight quantitative findings

All intervals are 95 % percentile bootstrap intervals from
`analysis/secondary_analysis.json` (10,000 paired resamples of the 420 Test
molecules, seed 0). They describe the frozen predictions only.

1. **Ranking on Test:** PNA-native 0.5289 < LCRC-A-R3-scratch 0.5840 <
   ESA-native 0.5989 < XGB-P3 0.6570. All four beat the constant baseline
   (1.2635) by 48–58 %.

2. **RMSE with uncertainty (Test420):** PNA-native 0.5289 [0.4598, 0.6047];
   LCRC 0.5840 [0.4959, 0.6780]; ESA-native 0.5989 [0.5244, 0.6849];
   XGB-P3 0.6570 [0.5924, 0.7258]. The intervals overlap for the three learned
   models, so the ordering among them is not resolved by one seed.

3. **PNA − LCRC = −0.0551 RMSE** [−0.1138, +0.0129], P(PNA worse) = 0.052.
   The attention-based model is ahead on the point estimate, but the paired
   interval still crosses zero: at seed 0 the two are not separable.

4. **ESA − PNA = +0.0700 RMSE** [+0.0261, +0.1129], P(ESA worse) = 0.0005. This
   is the one gap whose paired interval excludes zero: the edge-set-attention
   model is *worse* than PNA here, despite 33 % more parameters and a validation
   RMSE that looked competitive (0.5729 vs 0.5540).

5. **Validation did not rank Test correctly.** ESA-native had the second-best
   Validation RMSE (0.5729 vs PNA 0.5540, LCRC 0.5890) but the third-best Test
   RMSE (0.5989 vs 0.5840), i.e. it degraded by +0.0260 from Validation to Test
   while LCRC improved by −0.0050 and PNA by −0.0252. With 420 Validation
   molecules, a 0.02 RMSE difference is inside noise, so Validation-based model
   choice is not trustworthy at this sample size.

6. **LCRC vs the descriptor baseline:** LCRC − XGB = −0.0730 [−0.1542, +0.0042],
   P(LCRC worse) = 0.032; PNA − XGB = −0.1281 [−0.1928, −0.0703],
   P(PNA worse) = 0.0000. PNA beats the tuned XGB baseline by a margin that
   survives the bootstrap; LCRC does not quite.

7. **The models make correlated errors.** Squared-error correlations on Test:
   PNA–ESA 0.840, LCRC–ESA 0.791, LCRC–PNA 0.722, ESA–XGB 0.620, PNA–XGB 0.611,
   LCRC–XGB 0.539. Per-molecule "closest model" shares are LCRC 30.7 %,
   PNA 27.1 %, ESA 23.1 %, XGB 19.0 %, i.e. no single model dominates
   everywhere; the graph models agree much more with each other than with XGB.

8. **Generalisation gap grows with capacity, not with performance.** Train →
   Validation RMSE: PNA +0.397, LCRC +0.392, XGB +0.359, ESA +0.268. ESA
   overfits least and still generalises worst, so under-fitting (not
   over-fitting) is what limits it at 360-epoch/60-epoch budgets; LCRC's 247 k
   parameters reach within 0.055 RMSE of a 7.8 M-parameter PNA.

Error by molecule size (tertiles of heavy-atom count incl. explicit H, cut at
43 / 56 atoms) is in `analysis/secondary_analysis.json`: only LCRC improves
monotonically with size (MAE 0.411 → 0.348), XGB is flat (0.480 → 0.492) and
PNA degrades slightly (0.366 → 0.384).

---

## 3. What was run, and where

| item | value |
| --- | --- |
| Host / backend | `res` (`a100-2`), process backend via `rr` |
| GPU | 2 × NVIDIA A100-SXM4-40GB, driver 550.163.01 |
| torch | 2.5.1+cu124, CUDA 12.4 |
| precision | bf16 autocast (PNA, ESA); fp32 (LCRC, XGB) |
| data host | structures frozen from the committed, hash-verified artifact |
| Test inference | frozen checkpoints, fp32, Batches, `analysis/pre_test_manifest.json` |

Per-run GPU wall time charged (full process wall time, including data loading
and validation): LCRC 330.8 s, PNA 480.3 s, ESA 241.6 s, CPU smokes 8.1 s, plus
one failed ESA smoke charged 5 s from launcher timestamps → **0.2960 GPU-h**.
No retry, no re-run, no second seed. See `analysis/gpu_ledger.json`.

---

## 4. Test discipline evidence

1. All four models finished training and Validation reporting
   (`analysis/metrics_validation.json`, recomputed from the frozen
   `valid_predictions.npz` of each run, matching every reported value to 1e-9).
2. Checkpoints and Test predictions were frozen and committed at `91955ee`,
   together with `analysis/pre_test_manifest.json`
   (SHA256 `5ebdfdaa…`, containing the checkpoint hash, prediction hash, split
   hash, config hash and charged GPU-hours of every model).
3. The single scoring pass ran once against that manifest hash and wrote
   `analysis/metrics_test.json` + `analysis/test_read.lock`.
4. A second invocation is refused:
   `PermissionError: Test has already been scored once
   (…/analysis/test_read.lock exists)`.
5. After the lock, `code/r42_analysis.py` re-read the Test labels **only** to
   compute the bootstrap intervals in §2. It verifies every prediction file
   against the manifest first, writes `status: POST_LOCK_DESCRIPTIVE`, and
   cannot alter `metrics_test.json`.

---

## 5. Deviations, and things a reader should know

1. **Single seed, exploratory.** No seed variance, no seed selection, no
   hyperparameter search. Any ranking claim is a point estimate.
2. **random42 is not a blind Test set for this repository.** Earlier rounds
   (`lcrc_lipo_screen_v1`, `…_v2`, `…_v3`, `…_v4`,
   `bjoc_cross_task_lipophilicity_v1`) already touched the same 4,200 molecules
   in other splits and, in some cases, on this very split. No checkpoint,
   prediction or hyperparameter was reused here, but the Test number must not
   be read as a never-seen generalisation estimate.
3. **Upstream runtime adaptations** (vendored files byte-identical to
   `861a0b12…`, listed in `analysis/implementation_manifest.json`):
   `pytorch_lightning.LightningModule → torch.nn.Module`,
   `bitsandbytes.AdamW8bit` never constructed (**the formal optimiser is
   `torch.optim.AdamW`**), `flash_attn.swiglu → F.silu(x) * y`, `admin_torch`
   OmegaResidual reproduced verbatim, `xformers` unavailable so the upstream
   `torch` SDPA backend is used (bool-mask semantics corrected to match the
   xformers additive-bias convention, proven by
   `tests/test_mask_and_training.py`), and mask-helper devices derived from
   `edge_index` instead of hard-coded `cuda:0`.
4. **Four ESA parameters receive no gradient in this configuration**
   (`mlp_norm.ln.{weight,bias}`, `st_fast.decoder.0.norm.ln.{weight,bias}`).
   This is upstream behaviour for edge-attention + PMA; it is recorded in
   `analysis/cpu_gates.json` and was not "fixed" in this round.
5. **The GPU host has no RDKit**, so the Chemprop featurisation import is
   optional there; the pre-built, hash-verified structure artifact is used
   instead. Row identity was verified against the committed row manifest on the
   GPU host and against the CSV on the archival host (`analysis/prep.json`).
6. **XGB-P3 was trained and frozen on the CPU host** (4.6 s) because it needs
   RDKit; it is reported with GPU-h = 0 as required.
7. **One GPU smoke attempt failed** (missing RDKit, 5 s) before model
   construction. It produced no envelope, so `r42_ledger.MANUAL_CHARGES` charges
   it explicitly; the ledger total includes it.
8. **LCRC soup** averages the last 5 epoch-end state dicts by the frozen rule;
   it records no "best epoch" because that rule has no selection step.

---

## 6. Reproduction

```bash
cd tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1
uv run python code/run_v1.py freeze-protocol
uv run python code/run_v1.py audit          # 8 checks PASS
uv run python code/run_v1.py gates          # 10 checks PASS
uv run python -m pytest tests -q            # 41 passed
uv run python code/run_v1.py implementation

# on the GPU host (deployed through rr)
uv run python code/run_v1.py prep
uv run python code/run_v1.py train lcrc --device cuda --gpu-budget-check
uv run python code/run_v1.py train pna  --device cuda --gpu-budget-check
uv run python code/run_v1.py train esa  --device cuda --gpu-budget-check

# back on the archival host
uv run python code/run_v1.py train xgb --device cpu
uv run python code/run_v1.py ledger
uv run python code/run_v1.py manifest
uv run python code/run_v1.py score --manifest-sha256 <pre_test_manifest sha256>
uv run python code/run_v1.py secondary      # post-lock, descriptive only
```

---

## 7. Files

```
PROTOCOL.md                                   frozen protocol (sha256 above)
configs/benchmark_v1.yaml                     frozen constants
data/split_esa_random42_v1.json               random42 indices + hashes
data/labels_train_valid_v1.json               Train/Valid labels only (Test absent)
data/structures_esa_random42_v1.npz(+_meta)   4200 Chemprop graphs, hash-verified
code/r42_*.py, code/run_v1.py                 implementation + CLI
code/upstream/                                vendored upstream @861a0b12 (byte-identical)
tests/                                        41 tests
analysis/data_audit.json                      8/8 PASS
analysis/cpu_gates.json                       10/10 PASS
analysis/implementation_manifest.json         what was implemented, incl. stubs/patches
analysis/prep.json                            GPU-host cache build + row verification
analysis/metrics_validation.json              Train/Valid metrics per model
analysis/pre_test_manifest.json               the freeze (sha256 above)
analysis/gpu_ledger.json                      0.2960 h charged, per-run detail
analysis/metrics_test.json                    the single Test result
analysis/secondary_analysis.json              bootstrap CIs, deltas, error breakdown
analysis/prediction_manifest.json             frozen Test prediction hashes
checkpoints/                                  frozen weights (LCRC/PNA/ESA/XGB)
predictions/                                  frozen Test420 predictions per model
analysis/test_read.lock                       one-shot Test lock
```

## 8. Bottom line

On the frozen random42 split, at one seed: **PNA-native is the strongest of the
four (Test RMSE 0.5289, R² 0.8248)**, the 247 k-parameter LCRC model is second
(0.5840) and not separable from it at this sample size, the 10.4 M-parameter
edge-set-attention model is third (0.5989) and significantly worse than PNA,
and the tuned descriptor baseline is last (0.6570). Validation ranking did not
transfer to Test for ESA vs LCRC. With 420 Test molecules and a single seed,
only the PNA-vs-ESA and PNA-vs-XGB gaps have paired intervals that exclude zero.
