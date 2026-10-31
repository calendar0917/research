# LCRC-Random42 v2 — structural-mechanism and parameter-efficiency minimal controls

**Protocol ID:** `lcrc-lipo-random42-controls-v2`  
**Base commit:** `16c2c271a746a72cb699b1d7798679c3b17f5fac`  
**Formal training commit:** `b8495e96c318` (protocol freeze `849669ca…`)  
**Final evaluation commit:** see `git log -1` for this directory.  
**Scope:** Train3360 / Validation420 only. **No Test label, Test prediction, Test metric, Test bootstrap or Test error analysis was read.**

This report is descriptive. It does **not** claim that a mechanism is proven. It records two new seed-0 controls, their validation numbers, mechanical gates, inference timing and cost.

## 1. Executive summary

| Model | Role | Params (nominal / effective) | Validation RMSE | Validation MAE | Validation R² |
|---|---|---:|---:|---:|---:|
| LCRC-A-R3-scratch | historical reference, read-only | 247,106 / 247,106 | **0.589007** | 0.420814 | 0.770450 |
| LCRC-S2off-capacity-matched | new structural control | 285,449 / **247,049** | 0.659794 | 0.518401 | 0.711959 |
| PNA-native | historical reference, read-only | 7,767,349 / 7,767,349 | **0.554034** | 0.400718 | 0.796900 |
| PNA-small (width 45) | new parameter-efficiency control | **256,041** / 256,041 | 0.597499 | 0.438204 | 0.763783 |

Key observations, stated without causal language:

- The S2-off control has **57 fewer effective parameters** than LCRC-A (−0.0231%) but is **0.070787 RMSE worse** on Validation420. The paired bootstrap interval is entirely above zero (descriptive 95% interval [+0.018051, +0.127056]); S2-off is worse on 259/420 molecules and better on 161/420.
- The PNA-small control is **3.62% larger** than LCRC-A and **0.008492 RMSE worse** than LCRC-A, while being **30.3× smaller** than PNA-native and **0.043465 RMSE worse** than PNA-native.
- The S2-off result is a negative control result: under this recipe/split/seed, the S1-context replacement does not recover LCRC-A's validation accuracy at matched effective capacity.
- All reported numbers come from the new runs or the hash-verified historical artifacts; nothing was selected by Validation score for the new models beyond the pre-declared checkpoint rules.

## 2. Frozen reuse and data isolation

All reuse is byte-level and hash-verified.

| Artifact | SHA-256 |
|---|---|
| Lipophilicity CSV | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` |
| Split file | `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4` |
| Train3360 row-index hash | `0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee` |
| Validation420 row-index hash | `8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44` |
| Test420 row-index hash (integrity only; never used) | `4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b` |
| Train/Validation labels file | `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca` |
| Train label vector | `85a4c7c559f4b7fd58ad78b1b05d5a4160e0f60c351b7fa842eb0354aa47abd8` |
| Validation label vector | `09bc1a1e1756bbd09dd17a5fc77a1b3c33e7010aef55fc352a4bba1d9822636f` |
| Structures archive | `a098d27fa11c5d718d2f0a373636cfc9eeac9604ea5db8ae50a36d05fe035a2c` |
| LCRC train-only normalisation fingerprint | `4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26` |
| Target scaler | mean `2.2081011904761905`, scale `1.1906281212426577` |

- The split and labels files were copied/read, not regenerated. The LCRC cache was materialised byte-for-byte from the v1 cache and its content plus train-norm fingerprints were checked.
- The target scaler is fitted on Train3360 labels only. Validation labels are used only for loss/checkpoint selection and final descriptive metrics.
- Static and runtime guards reject Test artifacts; `data_audit.json`, `reuse_audit.json` and the prediction-scope gate record `test_labels_read: false` and `test_predictions_read: false`.
- Historical checkpoints and Validation predictions were hash-verified and re-scored; historical models were **not** retrained.

Historical artifacts used for the table:

| Model | Checkpoint SHA-256 | Validation prediction SHA-256 |
|---|---|---|
| LCRC-A-R3-scratch | `eca1486102faed79ed4c233332644d8767dc76193adc16523fc115bb33a0e683` | `7f9ad3919d57755d5adc08b99f1209b18ab6629c14eb182592a858183d88e894` |
| PNA-native | `3eebd089a6db49f504bc7de769ce73cbafe55d575028523560577b74e139524b` | `735df74915d43803b63323e09d85089c6c2d284e04acffcf242eb587ae1ac905` |

## 3. Experiment A — capacity-matched S2-off

The historical `v2_models.LCRCCapacityMatched` class was imported unchanged. The S2 slot is zeroed in train and inference; the replacement branch consumes only `[a, S1_sum, S1_mean, S1joint]`.

| Audit | Result |
|---|---|
| Nominal parameters | 285,449 |
| Structurally dead LocalMLP S2 input columns | 38,400 |
| Effective trainable parameters | **247,049** |
| LCRC-A reference | 247,106 |
| Effective difference | −57 (−0.0231%) |
| Replacement branch | 70,535 parameters |
| Shared initialization vs LCRC-A | 27/27 shared entries bitwise identical |
| New message passing / attention | none |
| Typed-S2 inputs used | false |

Mechanical gates:

- **Typed-S2 invariance:** CPU gate is exact. On CUDA, repeated identical forwards themselves differ by up to 2.4e-7 because `index_add_` uses atomics; masked and perturbed S2 inputs stay within the same run-to-run noise (pre: 6.0e-8; post-soup: 3.3e-7; tolerance 1e-6). Restricted batches that try to access typed-S2 keys raise. This is a CUDA determinism limitation, not evidence of S2 use.
- **Gradient placement:** all four replacement parameter tensors receive non-zero gradients; the 38,400 dead S2 input columns have maximum absolute gradient **exactly 0.0**; live LocalMLP columns receive gradients (max abs 17.56 on the gate batch).
- **Recipe:** AdamW, peak LR 3e-4 → min LR 1e-5 per-step cosine, weight decay 1e-3, batch 128, 360 epochs, 27 steps/epoch, 9,720 optimizer steps, grad clip 5.0, last-5 epoch soup (members 356–360), no early stopping.
- **Dropout:** v1 configured `dropout: 0.10` but its executed LCRC loop never applied it. Both the historical arm and this control therefore use the **executed** recipe, `dropout_applied: 0.0`. This is recorded in `train.json` to avoid a recipe confound.
- **Checkpoint:** `soup_last_5`; Validation predictions are from the soup state.

New artifacts:

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `3c18288188319f8cc2aba66f27d668468095233643b63826590002686a592720` |
| Validation predictions | `41693f78ee0b9bf26316d1f5749b84334785d517530a7ce618dc56106cd9334b` |
| Train predictions | `870a828356b0d335be1add0b0c80d52964fbda4734fe2085deb17578574c3c95` |

## 4. Experiment B — PNA-small

Only `gnn_intermediate_dim == output_node_dim` was changed. The PNA-native stack, split, native Chemprop input, four layers, aggregators (`mean/min/max/std`), scalers (`identity/amplification/attenuation`), five towers, AdamW LR 1e-4, weight decay 1e-10, batch 128, ReduceLROnPlateau(0.5, 15), patience 30, max 360 epochs, best-Validation-loss checkpoint, seed 0 and no pretraining were retained.

**Width rule (CPU only, before any training):** instantiate widths 20, 25, …, 100, count real parameters, choose minimum `|params − 247,106|`. No Validation score was read. The search was frozen before training.

| Width | Parameters | Absolute difference | Relative difference |
|---:|---:|---:|---:|
| 40 | 204,061 | 43,045 | 17.42% |
| **45** | **256,041** | **8,935** | **3.6159%** |
| 50 | 313,921 | 66,815 | 27.04% |

Selected width: **45**. Status: **`TIGHTLY_SIZE_MATCHED`** (relative difference 3.6159% < 15%). `labels_read: false`, `validation_scores_read: false`, `cuda_available: false` at search time.

Training/checkpoint audit:

- 256,041 parameters; all 120 parameter tensors receive gradients on the gate batch; 253,295/256,041 individual elements are non-zero on that batch.
- Best epoch **179** by standardized Validation loss; ran **209** epochs (early stop patience 30); **5,643** optimizer steps.
- Checkpoint rule: `best_validation_loss`.
- PNA degree histogram is Train3360-only and matches the frozen v1 tensor SHA-256 `56bc94c1186a1a471b9af7885313ed914f51469f3e40fa99266a20c65721fc40`.

New artifacts:

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `634fa20f65540432a1c6d2ff4f0ebf1063235412a02d399b4a2e8e314c451f71` |
| Validation predictions | `dd9ca29c3bb7417f9aad4a2ab44673dd87128fb3010daf0435db14148a12dfb3` |
| Train predictions | `e62d2d054e4a237e292d31db3511e0a2cd310b1352bba0c9d0b153eb2924d94e` |

## 5. Main Validation420 table

All metrics are raw logD, recomputed from the saved molecule-level predictions. No Test number appears here.

| Model | Seed | Params nominal | Params effective | Width | Checkpoint rule | Best epoch | Epochs | Optimizer steps | Train RMSE | Valid RMSE | Valid MAE | Valid R² |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| LCRC-A-R3-scratch | 0 | 247,106 | 247,106 | — | last-5 soup | — | 360 | 9,720 | 0.196846 | 0.589007 | 0.420814 | 0.770450 |
| LCRC-S2off-capacity-matched | 0 | 285,449 | 247,049 | — | last-5 soup | — | 360 | 9,720 | 0.445373 | 0.659794 | 0.518401 | 0.711959 |
| PNA-native | 0 | 7,767,349 | 7,767,349 | 256 | best valid loss | 84 | 114 | 3,078 | 0.157223 | 0.554034 | 0.400718 | 0.796900 |
| PNA-small | 0 | 256,041 | 256,041 | 45 | best valid loss | 179 | 209 | 5,643 | 0.401510 | 0.597499 | 0.438204 | 0.763783 |

Historical GPU hours shown in `validation_table.json` are from the v1 ledger and are **not** charged against this round's cap: LCRC-A 0.091893 h, PNA-native 0.133414 h.

## 6. Paired bootstrap — LCRC-A vs S2-off

- Comparison: `S2-off − LCRC-A`; positive means S2-off has larger error.
- Unit: Validation420 molecules, paired.
- Resamples: **2,000**; seed: **0**; generator `numpy.random.default_rng(0)`.
- Status: **`DESCRIPTIVE_ONLY`**; single seed; no significance claim.

| Quantity | Observed | Bootstrap mean | 95% percentile interval | P(delta > 0) |
|---|---:|---:|---:|---:|
| ΔRMSE | +0.070787 | +0.071867 | [+0.018051, +0.127056] | 0.995 |
| ΔMAE | +0.097586 | +0.097932 | [+0.059404, +0.135195] | 1.000 |

Molecule-level counts: S2-off better 161, worse 259, ties 0. Squared-error correlation 0.5803.

These intervals describe this split/seed/recipe only. They do not establish a general mechanism and do not estimate Test performance.

## 7. Same-A100 inference benchmark

Device: one NVIDIA A100-SXM4-40GB, 40 GB, CUDA 12.4, torch 2.5.1+cu124. Batch size 128 (four Validation420 batches), 3 warm-up passes, 10 timed repeats. Preprocessing and forward are timed separately:

- **Preprocessing:** collate / `Batch` construction + host-to-device transfer.
- **Forward:** model forward only; no loss, no backward.
- **Peak memory:** `torch.cuda.max_memory_allocated` after resetting at the start of each model's benchmark block (includes resident model and inference tensors).

| Model | Precision | Preprocessing (s / 420) | Forward (s / 420) | End-to-end (s / 420) | Peak allocated |
|---|---|---:|---:|---:|---:|
| LCRC-A-R3-scratch | fp32 | 0.03158 ± 0.00007 | 0.00436 ± 0.00001 | 0.03594 ± 0.00007 | 55.32 MB |
| LCRC-S2off-capacity-matched | fp32 | 0.03156 ± 0.00008 | 0.00422 ± 0.00018 | 0.03579 ± 0.00019 | 55.26 MB |
| PNA-native | bf16 | 0.01850 ± 0.00046 | 0.11020 ± 0.00044 | 0.12870 ± 0.00054 | 962.09 MB |
| PNA-small | bf16 | 0.01801 ± 0.00032 | 0.03697 ± 0.00004 | 0.05498 ± 0.00031 | 188.41 MB |

Checkpoint reproduction during the benchmark:

- LCRC/S2-off (fp32) reproduce the saved Validation predictions to **2.0e-6** max absolute difference; RMSE difference 0.0.
- PNA arms use native bf16. Repeated bf16 inference is not bitwise reproducible because PNA's `std`/scatter aggregators use reduction atomics. PNA-native max difference 0.0372 (RMSE difference 0.000693); PNA-small max difference 0.2186 (RMSE difference 0.002695). The benchmark records these differences and requires metric agreement within 5e-3 RMSE; the table itself uses the saved predictions, not the benchmark pass.

## 8. GPU ledger and budget

Charging rule: full wall time of every process that touched CUDA, including data loading, smoke runs, failed attempts, benchmark and the diagnostic. CPU-only work is recorded at zero GPU-hours.

| Item | GPU hours |
|---|---:|
| Formal S2-off (`s2off_s0`) | 0.063692 |
| Formal PNA-small (`pna_small_s45_s0`) | 0.118113 |
| Smoke attempts (3 S2-off + 3 PNA-small) | ~0.0026 |
| Benchmark attempts (5 failed + 1 successful) | ~0.0055 |
| PNA reproduction diagnostic | 0.001944 |
| **Total charged** | **0.191911** |
| Hard cap | 0.500000 |
| Remaining | 0.308089 |

Ledger status: **within cap**, 15 entries, 8 failed attempts recorded (3 smoke failures + 5 benchmark failures). The failed attempts were plumbing/determinism issues, not performance retries; no formal training run was repeated.

Failed-attempt summary (all charged):

1. S2-off smoke: CPU/GPU device mismatch in `shared_init_audit`; fixed by auditing on CPU then moving the model.
2. PNA smoke: `mse_loss` outside autocast mixed bf16 and fp32; fixed by keeping the loss inside autocast.
3. S2-off smoke: typed-S2 invariance required exact CUDA equality; fixed to require invariance within run-to-run atomic noise.
4. Benchmark: `torch.cuda.set_device(torch.device("cuda"))` rejected; fixed to an indexed device.
5. Benchmark: LCRC predictions were compared in standardized space to raw saved predictions; fixed by denormalising.
6. Benchmark: prediction files under `analysis/runs/` were ignored by the repository-wide `*.npz` rule; force-tracked.
7. Benchmark: PNA bf16 per-molecule reproduction was stricter than bf16 allows; relaxed to a metric check plus bounded per-molecule difference.
8. Benchmark: PNA-small bf16 scatter noise exceeded the first relaxed bound; final bound is 0.5 raw logD with a 5e-3 RMSE agreement requirement.

## 9. Protocol provenance and post-freeze changes

- The protocol document is unchanged; its internal SHA-256 is `35ae8a0b952bf099cf9e678ad6dcfa93563d3eee7e8759d3d6c5ff875a3cebeb`.
- Formal training ran at commit `b8495e96c318` with protocol-freeze file SHA-256 `849669ca86145489a7e14a8eb3ff119b743289d332ed7e6fad10b8486c5b1874`.
- The final freeze file SHA-256 is `4a4c8bd77d1d750036895d8af9f52e92446138d4c9176671893b012014ac7f11` (recorded in `analysis/protocol_sha256.json` as `4a4c8bd7…`).
- **Training/data/gate code hashes are identical between the formal-training freeze and the final freeze**: `c2_train.py`, `c2_models.py`, `c2_data.py`, `c2_common.py`, `c2_gates.py`, `run_v2c.py`. Only `c2_eval.py` changed after training, and only inside `run_inference_benchmark` / `_benchmark_entry` (CUDA device indexing, denormalisation for the reproduction check, bf16 tolerance). The validation table and paired bootstrap functions were not changed.
- The S2-off model class is the historical `v2_models.LCRCCapacityMatched` imported unchanged. The only code-level adaptations are: (i) build/audit on CPU then move to the requested device, preserving the bitwise shared-init audit; (ii) a CUDA-noise-aware invariance assertion. Neither changes parameters, forward semantics, or the training recipe.
- The PNA-small training loop keeps the loss inside autocast, matching the historical v1 loop; this was necessary for bf16.

## 10. Remaining risks and limits

- **Single seed (0).** No seed variance, no seed-level confidence interval, no hyperparameter search, no performance retry. The paired bootstrap is molecule-level and descriptive only.
- **S2-off is a negative control result.** It shows that this S1-context replacement, at 57 fewer effective parameters, is worse than LCRC-A on this split/seed. It does not by itself identify which part of LCRC-A causes the difference.
- **Nominal vs effective capacity.** S2-off has 285,449 nominal parameters and 247,049 effective parameters; the 38,400 dead S2 input columns still exist in the optimizer state and receive weight decay (with exactly zero gradient). The capacity accounting follows the historical v2 implementation and its fixed 200→192 residual.
- **CUDA atomic non-determinism.** S2-off output invariance and PNA bf16 predictions are not bitwise reproducible on this GPU. Gates use a 1e-6 noise floor for S2-off and metric-level agreement for PNA benchmark reproduction.
- **PNA-small is not exactly 247,106 parameters.** It is 256,041 (+3.62%), the closest width on the pre-declared 5-parameter grid; the task's 15% rule labels it tightly size-matched.
- **Benchmark scope.** Timing covers Validation420 only, at batch 128, on one A100-SXM4-40GB. Preprocessing includes Python collate and host-to-device copy; it is not a raw CSV-parsing benchmark. Peak memory is allocator-observed and excludes other processes.
- **Historical timing provenance.** Historical GPU hours and recorded inference times come from the v1 ledger, not from a re-run in this round. The new four-model benchmark was run in this round on the same GPU model.
- **No Test access.** No Test labels, predictions, metrics, bootstrap or molecule-level errors were opened. The Test index hash appears only as an integrity check of the frozen split file.

## 11. Deliverables and reproduction

| Item | Path |
|---|---|
| Protocol | `PROTOCOL.md` |
| Config | `configs/controls_v2.yaml` |
| Code | `code/` |
| CPU gates | `analysis/cpu_gates.json` |
| Width search | `analysis/pna_width_search.json` |
| Protocol freeze | `analysis/protocol_freeze.json`, `analysis/protocol_sha256.json` |
| Reuse audit | `analysis/reuse_audit.json`, `analysis/data_audit.json` |
| New checkpoints | `checkpoints/lcrc_s2off_seed0.pt`, `checkpoints/pna_small_w45_seed0.pt` |
| New Validation predictions | `predictions/lcrc_s2off_valid420.npz`, `predictions/pna_small_w45_valid420.npz` |
| Train predictions | `predictions/lcrc_s2off_train3360.npz`, `predictions/pna_small_w45_train3360.npz` |
| Run manifests | `analysis/runs/s2off_s0/`, `analysis/runs/pna_small_s45_s0/` |
| Validation table | `analysis/validation_table.json` |
| Paired bootstrap | `analysis/paired_bootstrap.json` |
| Inference benchmark | `analysis/inference_benchmark.json` |
| GPU ledger | `analysis/gpu_ledger.json` |
| PNA reproduction diagnostic | `analysis/pna_small_repro_diag.json` |

Key commands (from the repository root):

```bash
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py gates
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py freeze
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py width-search
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py preflight
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py train s2off --device cuda
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py train pna_small --device cuda
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py benchmark --batch-size 128 --warmup 3 --repeats 10
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py evaluate
uv run python tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/code/run_v2c.py ledger
uv run pytest -q tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/tests
```

At the final verification, the unit suite passed **14/14** and the CPU gate suite passed **12/12** with zero GPU use.
