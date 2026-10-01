# E2E-DictEnv-RoleCorr-Increment-v2 — pre-registration (frozen)

Round: **E2E-DictEnv-RoleCorr-Increment-v2** (`e2e_dictenv_rolecorr_increment_v2`).
Protocol: `zinc-context-gap` (study `zinc-context-gap`), track `tracks/ksvd`.
Write-then-follow: nothing below may change after the first formal run.

任务来源（中文摘要）：旧的 RoleCorr-v1 用“结构16 + 对应关系16”替换了
“结构32”，负结果混合了“新增对应信息”和“删除一半结构容量”两件事。本轮
保留完整结构字典 K32/s8，在同一宽度 49 的接口上**追加**一个 16 维冻结块，
构建 A/B/C 三臂，回答“对应关系在完整结构预算之上是否还有增量”；
若路线一没有前景且预算允许，再检验“联合编码对象”（结构残差 + Sem108 +
对应关系）是否有性能前景。

---

## 0. Environment status (recorded at freeze time)

This pre-registration was frozen while the remote compute host could not
provide CUDA: the physical GPU0 (`0000:4b:00.0`) is in a driver-level failed
state (`nvidia-smi` reports `Unknown Error`; `cuInit(0)` returns 999
`CUDA_ERROR_UNKNOWN`; torch reports `cuda_available=False, device_count=0`),
which makes the healthy GPU1 unreachable through CUDA even with
`CUDA_VISIBLE_DEVICES=1`.  `nvidia-smi -i 1` alone still shows the A100 healthy.
No formal run has been executed; when the driver is restored, the run must
still follow the device rules below (physical GPU1 as logical `cuda:0`).

---

## 1. Single question

**Primary (route 1).** Holding the full frozen structural dictionary
(K32/s8), the common-1 coordinate, the Sem108 interface, the C6 mask, the
static composition, the reader, the optimiser and the 320-epoch budget
identical, does appending the frozen role↔attribute correspondence coordinate
(16 sparse codes of the frozen RoleCorr object) improve official-valid Top-5
soup MAE over a matched **appended structural** coordinate of the same width?

**Auxiliary.** Is any increment specific to sparse dictionary coding of the
correspondence, or is it recovered by a dense train-fitted PCA16 of the same
object?

**Conditional (route 2, only if the route-1 gate misses).** Does replacing the
residual coordinate with one shared frozen dictionary over the joint
structural-residual + Sem108 + correspondence representation have performance
prospect at the same 49-wide interface, versus a fixed PCA48 control?

This is a frozen-coordinate screening round, not a new architecture search.

---

## 2. Frozen reused objects (read-only, SHA-verified at run time)

| object | definition | frozen identity |
|---|---|---|
| common subspace | frozen CSSD-q1 `U ∈ R^{65×1}` | `common_subspace.json` file SHA-256 `36636ce9…8f6c24` |
| base structural dictionary | frozen `D_SDB ∈ R^{65×32}` (SDB-v0 K-SVD, `s=8`) | `results/sdb_v0/dictionary.pt` |
| extra structural dictionary | frozen RoleCorr `D_S ∈ R^{65×16}`, `s=4`, fit on the common-1 train residual | `dictionary_struct.pt` SHA-256 `57494647…a0a660bdd` |
| correspondence dictionary | frozen RoleCorr `D_C ∈ R^{536×16}`, `s=4`, fit on the train-scaled `C` | `dictionary_corr.pt` SHA-256 `8d21e24b…7b1d94da` |
| correspondence scaler | frozen RoleCorr node/edge block scaler | `standardizers.json`; verified by refitting from the raw cache (must match bit-for-bit) |
| raw correspondence cache | frozen 536-D object `C` (node 392 / edge 144) | train/valid node+edge SHA-256 pinned in the runner |
| Sem108 interface / C6 mask / backend / reader | frozen `SEM108Model` parent | code-level reuse |

Nothing is refit except: (route 1) the PCA16 control, and (route 2) the joint
scaler / K48 dictionary / PCA48.  Official valid is never used to fit any
scaler, dictionary or PCA.

---

## 3. Frozen coordinate and arms (route 1)

```
coordinate  z = [ c~ (1) ; alpha_base (32) ; block (16) ]   in R^49
c~          = (phi · U) / common_rms                        (frozen CSSD-q1)
alpha_base  = IHT10( colnorm((I - UUᵀ) D_SDB), (I - UUᵀ) phi )  exact top-8
```

Appended block per arm (all frozen except the PCA):

| arm | block (16) | source |
|---|---|---|
| A `EXTRA-STRUCT` | `IHT10( colnorm((I - UUᵀ) D_S), r )`, exact top-4, `r = (I - UUᵀ) phi` | frozen RoleCorr `D_S` |
| B `CORR-ADD` | `IHT10( colnorm(D_C), C_scaled )`, exact top-4 | frozen RoleCorr scaler + `D_C` |
| C `CORR-PCA-ADD` | `(C_scaled − mean) · componentsᵀ` | affine PCA16 fit on official train (this round) |

Rules:

* all three arms share the identical base 32-coordinate and common coordinate;
* any extra magnitude normalisation is the frozen RoleCorr scaler (B) or the
  train-fitted PCA (C); the two-layer train-RMS + zero-RMS-mask rule is
  fixed and identical across arms where applicable;
* the full SDB K32/s8 dictionary is **not** reduced or refit.

### 3.1 Binding extension (frozen, deterministic)

The historical widened binding `W_A_S [33, 48]` / `W_E_S [99, 48]` is copied
into `[49, 48]` / `[147, 48]` with the 16 new rows **zero-initialised**
(node rows 33…48; per edge block, rows `block*49+33 … block*49+48`).  No
other parameter is touched, so all arms share a bit-identical readout /
binding initialisation; only `D` and `D_block` differ.

### 3.2 Frozen (forbidden) changes

No message passing, GNN, Transformer, attention, recurrence or multi-round
state propagation.  No dictionary fine-tuning, no `lambda_rec` change (the
frozen dictionaries make the reconstruction term gradient-inert), no
Sem108/backend/reader change, no width other than the necessary 33→49 binding
extension, no LR / lambda / horizon / fusion / readout / relation / pooling
change, no canonical adjacency flattening.

---

## 4. Training protocol (frozen, inherited)

* data: official ZINC train 10000 / official valid 1000; **official test never
  loaded** (`test_policy: terminal`; non-terminal control-plane modes enforce
  `test_access: blocked`).
* seed 0; epochs 320; batch 128; Adam `lr=1e-3`, `weight_decay=1e-5`,
  grad clip 5.0; loss task L1 (+ frozen-dictionary reconstruction diagnostic);
  no early stop; Top-5 soup by valid MAE.
* data order: frozen Sem108 loader seeds (`seed + TRAIN_SHUFFLE_OFFSET` /
  `seed + EVAL_SHUFFLE_OFFSET`); arms A/B/C run serially in one invocation,
  identical loaders, optimiser schedule and soup rule.
* **device**: explicit `runtime.device`; the formal round uses the remote
  physical **GPU1** only (`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`), never
  GPU0.  `device: cpu` preserves the historical regime.  Device, GPU name,
  GPU UUID, torch/CUDA version, commit and split fingerprint are recorded.
* resumability: an epoch-level `*_resume.pt` (model + optimizer + RNG + curve +
  soup bookkeeping) is written every 10 epochs; a relaunch continues from it and
  records `resumed_at_epoch` / `data_order_restart: true` (a resumed run is not
  bit-identical to an uninterrupted one and may only be used after an
  infrastructure interruption).  A truncated run reports `completed: false` and
  is never recorded as a scientific failure.

---

## 5. Correctness gates (all before training; any FAIL ⇒ STOP)

* **G0 geometry** — coordinate width 49; slices `common [0,1)`, `base [1,33)`,
  `block [33,49)`; binding shapes `W_A_S [49, 96]`, `W_E_S [147, 48]`.
* **G1 base identity** — `alpha_base` is bit-identical to the frozen CSSD
  K32/s8 path; the common coordinate matches the frozen projection.
* **G2 appended-block identity** — arm A's block is bit-identical to the
  RoleCorr-v1 `structural_codes`; arm B's block is bit-identical to the
  RoleCorr-v1 `corr_codes`.
* **G3 binding extension** — the first 33 rows of `W_A_S` / `W_E_S` are
  bit-identical to the parent widened binding; the 16 appended rows are exactly
  zero at initialisation.
* **G4 shared initialisation** — every non-dictionary parameter (48 shared
  tensors) is bit-identical across A/B/C; trainable parameter counts are equal
  (frozen dictionaries excluded) and total parameter counts are reported.
* **G5 purity** — zeroing the appended block changes only columns 33…48;
  zeroing the base structural block changes only columns 1…32.
* **G6 sparsity / freezing** — exact `l0 ≤ 8` (base), `l0 ≤ 4` (A/B block);
  `D`, `D_S`, `D_C` are frozen (`requires_grad=False`); the PCA control is a
  fixed linear map.
* **G7 wiring / direction** — perturbing the correspondence input changes only
  the appended block coordinate; a deterministic non-zero extension-binding
  pattern makes the prediction respond to zeroing the appended coordinate
  (the shared zero initialisation makes the untrained model insensitive by
  design, so the gate probes the wiring, not the untrained readout).
* **G8 official-test blocker** — every payload asserts
  `official_test_loaded = false`; the test split is never instantiated.

Plus a device smoke (8 epochs × 1024 train / 512 valid) per arm: finite loss,
non-zero gradients on the readout, the bindings and the **appended binding
rows**, no gradient on frozen dictionaries.

---

## 6. Frozen probes (soup states, no retraining)

| probe | content |
|---|---|
| `M_A`, `M_B`, `M_C` | soup valid MAE per arm |
| `G_block_zero(arm)` | zero the appended block only |
| `G_block_shuffle(arm)` | row-shuffle the appended block within each molecule (5 seeds `11/22/33/44/55`), distribution-preserving |
| `G_object_shuffle(B)` | evaluate B with the within-group attribute-permuted correspondence object (frozen RoleCorr seeds `101/202/303/404/505`, same scaler/dictionary) |
| `G_base_zero(arm)` | zero the base structural block (diagnostic) |
| paired per-molecule | A–B, A–C, C–B absolute-error differences on official valid |
| code usage | active/effective atoms, top1 share, exact `l0` for base and block |

---

## 7. Frozen screening gate and decision (route 1)

```
improvement = M_A - M_B          # absolute soup-MAE improvement of B over A
PROCEED iff improvement >= 0.003
```

* If **not met**: verdict `INCREMENT_NO_MATERIAL_GAIN`; STOP route 1.  No
  K/s/LR/horizon/width rescue and no seed purchase.
* If met but the appended block is not load-bearing
  (`G_block_zero(B) < 0.003` **or** `G_block_shuffle(B) < 0.003`): verdict
  `INCREMENT_COORDINATE_NOT_LOAD_BEARING`; STOP.
* If met and load-bearing and `M_B <= M_C`: verdict
  `INCREMENT_SUPPORTED_SPARSE_SPECIFIC` — correspondence has an increment on
  top of the full structural budget and the sparse dictionary is not worse
  than the dense PCA16 control.  A paired-seed confirmation requires its own
  pre-registration.
* If met and load-bearing but `M_B > M_C`: verdict
  `INCREMENT_SUPPORTED_NOT_SPARSE_SPECIFIC` — report *increment supported,
  sparse-dictionary advantage not supported*.

Continue standard (frozen): a seed-0 soup improvement of ~≥0.003 over the
matched control with an actually-used channel is required before buying paired
seeds; small boundary differences are `uncertain` and are never reported as a
breakthrough.

---

## 8. Route 2 (conditional; only entered after a route-1 gate miss)

Frozen definition (implemented in the core module; its training/inference
stages are only enabled after route 1 misses the gate **and** the round is
explicitly extended within budget; otherwise it is recorded as skipped, never
half-run):

* joint input = train-only scale-balanced concatenation of
  structural residual (65) ‖ `Sem108` (108) ‖ `C` (536) = **709** dims;
  each block gets per-coordinate train-RMS scaling with a zero-RMS mask, then
  block-energy equalisation `w_b = 1/sqrt(E_train[‖block_b‖²]+ε)` so the three
  blocks carry equal mean squared energy;
* coordinate `[ c~ (1) ; IHT12(colnorm(D_joint48), joint_scaled) (48) ]`,
  width 49, one frozen K-SVD dictionary (K48/s12, 10 epochs, seed 20260924,
  full train rows);
* dense control `[ c~ ; PCA48(joint_scaled) ]` on the identical input;
* no canonical adjacency flattening; end-to-end dictionary updates are out of
  scope for this screening round.

Route-2 gate (only if entered): absolute soup-MAE improvement of the sparse
arm over the PCA48 control `>= 0.003` plus a load-bearing block; otherwise
verdict `JOINT_ENCODING_NO_MATERIAL_GAIN` and stop.

---

## 9. Stop discipline and reporting

* One route-1 invocation: cache → verify → pca16 → correctness → smoke →
  train A/B/C → interventions → analysis, all on one device.
* No seed 1, no horizon/width/lr/K/s change, no official test, no reuse of
  historical numbers as if they were matched.
* Every output records `protocol_version`, `git_commit`, `device`, GPU
  identity (when CUDA), `seed`, `official_test_loaded=false`, data/split
  fingerprints via the control plane, parameter counts (trainable and total),
  wall time, seconds/epoch and peak GPU memory.
* Deliverables: this pre-registration, the core module
  `e2e_dictenv_rolecorr_increment_v2.py`, the stage runner
  `zinc_e2e_dictenv_rolecorr_increment_v2.py`, the control-plane runner +
  config, focused CPU tests, the result directory (curves, checkpoints,
  probes, per-molecule errors), `REPORT.md`, `DECISION.md`, plus a STATE /
  claim / decision update when a real run exists.
