# PROTOCOL — `lcrc-lipo-random42-controls-v2`

Frozen protocol for two new single-seed controls on the random42 split:
one capacity-matched **S2-off** LCRC arm and one **PNA-small** arm.  This round
does **not** evaluate Test.  It reuses, byte-for-byte, the random42 v1
Train3360 / Validation420 indices, labels, raw CSV and upstream structure
artifact, and treats the two historical random42 v1 models as read-only
references.

| field | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-random42-controls-v2` |
| Track | `tracks/structural-sufficiency/` |
| Experiment root | `tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2/` |
| Baseline commit | `16c2c271a746a72cb699b1d7798679c3b17f5fac` |
| Dataset | MoleculeNet Lipophilicity, 4,200 molecules, target `exp` (logD) |
| Split | `torch.utils.data.random_split(range(4200), [3360,420,420], generator=torch.Generator().manual_seed(42))` |
| Training seed | `0` only |
| New models | `LCRC-S2off-capacity-matched`, `PNA-small` |
| GPU cap | **0.5 A100 GPU-hours** (all smoke, training and benchmark time) |
| Test access | **forbidden**: no Test label, Test prediction, Test metric or Test bootstrap |
| Evaluation scope | Train3360 and Validation420 only, raw logD |

Everything below is frozen before the first formal training run.  The
machine-readable copy is `configs/controls_v2.yaml`; the freeze file is
`analysis/protocol_freeze.json` with an SHA256 lock.

---

## 1. Frozen data reuse

The following files are consumed **in place** from the random42 v1 track and
hash-verified before any model is built:

| artifact | path | SHA256 |
| --- | --- | --- |
| split file | `lcrc_lipo_esa_random42_v1/data/split_esa_random42_v1.json` | `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4` |
| labels | `lcrc_lipo_esa_random42_v1/data/labels_train_valid_v1.json` | `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca` |
| structures | `lcrc_lipo_esa_random42_v1/data/structures_esa_random42_v1.npz` | `a098d27fa11c5d718d2f0a373636cfc9eeac9604ea5db8ae50a36d05fe035a2c` |
| structure meta | `lcrc_lipo_esa_random42_v1/data/structures_esa_random42_v1_meta.json` | `e3866a483f56d1cc8291122219cffd0949e167673e5bc1455f1cc697643e0a39` |
| raw CSV | `data/lipophilicity/Lipophilicity.csv` | `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd` |

Index-vector hashes:

| split | n | SHA256 of the `int64` index vector |
| --- | ---: | --- |
| Train | 3360 | `0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee` |
| Validation | 420 | `8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44` |
| Test | 420 | `4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b` |

The Test index is reproduced only to prove that the frozen split is unchanged.
No Test label is loaded and no Test structure is materialised into a batch.

**Target normalisation.** Train3360-only `StandardScaler` convention
(population standard deviation, `ddof = 0`):
`mean = 2.2081011904761905`, `scale = 1.1906281212426577`.
Validation labels never enter the fit.  Metrics are always denormalised back to
raw logD before RMSE/MAE/R² are computed.

**LCRC structural normalisation.** The `lipo_v1` Train3360-only normalisation
fingerprint must equal the frozen v1 value
`4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26`.
The LCRC structure cache is copied from v1 when present (or rebuilt from the
committed label-free raw-graph artifact) into this round's `cache/`; the v1
directory is never written.

---

## 2. Historical references (read-only)

The two historical random42 v1 models are not retrained.  Their checkpoints,
Validation predictions and manifests are hash-verified, and their Validation
metrics are recomputed from the frozen prediction files.

| model | parameters | Validation RMSE seed 0 | checkpoint SHA256 | Validation prediction SHA256 |
| --- | ---: | ---: | --- | --- |
| `LCRC-A-R3-scratch` | 247,106 | 0.5890072081930064 | `eca1486102faed79ed4c233332644d8767dc76193adc16523fc115bb33a0e683` | `7f9ad3919d57755d5adc08b99f1209b18ab6629c14eb182592a858183d88e894` |
| `PNA-native` | 7,767,349 | 0.5540342318996931 | `3eebd089a6db49f504bc7de769ce73cbafe55d575028523560577b74e139524b` | `735df74915d43803b63323e09d85089c6c2d284e04acffcf242eb587ae1ac905` |

The frozen v1 Validation report
(`analysis/metrics_validation.json`, SHA256
`c540f9ac89a391ea13c8235b7b14cb7faf10ae2a9d60616d2e4716659c576fdd`)
is cross-checked at 1e-12.  Historical Train predictions are also verified
against their manifests and recomputed, but they are not used for selection.

---

## 3. Experiment A — capacity-matched S2-off

**Implementation.** The historical, already-validated class
`lcrc_lipo_mechanism_generalization_ssl_v2/code/v2_models.py:LCRCCapacityMatched`
(SHA256 `00b3d3c3657cb7320affd79d8829110ad8d34ca8327191781882f225853c2988`) is
imported **unchanged**.  No new replacement architecture is designed.

The class keeps the LCRC-A trunk and the original 529-wide LocalMLP.  The
200-wide typed-S2 input slot is zeroed in both training and inference, so the
corresponding 38,400 first-layer input columns are structurally dead.  A
two-layer MLP consumes only the existing 320-wide S1/local context
`[a, S1_sum, S1_mean, S1joint]` and injects its 200-wide output through a
parameter-free fixed residual into the 192-wide LocalMLP hidden state.

Capacity ledger:

| quantity | value |
| --- | ---: |
| LCRC-A parameters | 247,106 |
| S2-off nominal parameters | 285,449 |
| dead typed-S2 input columns | 38,400 |
| **S2-off effective trainable parameters** | **247,049** |
| effective delta vs LCRC-A | −57 (−0.023 %) |
| replacement parameters | 70,535 |
| typed-S2 tensors read | none |
| new message passing / attention | none |

**Executed recipe.** AdamW `lr 3e-4 -> 1e-5` per-step cosine, weight decay
`1e-3`, batch 128, 360 epochs, gradient clip 5.0, MSE on the standardised
target, average of the last five epoch-end `state_dict`s, no early stopping, no
SSL.  The random42 v1 LCRC loop did **not** wire the configured `dropout=0.10`
into its forward pass.  This control mirrors that executed historical recipe
exactly; retrofitting dropout to only one arm would confound the contrast.  The
configured and applied values are both recorded in the run manifest.

**Mechanical gates before training.**
1. Same-seed shared initialisation is bitwise identical to a fresh LCRC-A for
   all 27 shared state entries, including the dead S2 input columns.
2. Replacing the batch with a `RestrictedBatch` that raises on every typed-S2
   key leaves the output bitwise unchanged.
3. Randomly perturbing `path_cat`, `path_end63`, `path_end` and `path_closure`
   leaves the output bitwise unchanged.
4. One real-batch backward pass gives every replacement parameter a nonzero
   gradient, gives the dead `lins[0].weight[:, 320:520]` block exactly zero
   gradient, and gives the live local columns a nonzero gradient.

---

## 4. Experiment B — PNA-small

**Architecture.** Upstream `gnn/graph_models.py: Estimator(conv_type="PNA")`
with the same native Chemprop one-hot input, explicit hydrogens, 4 PNA layers,
aggregators `mean/min/max/std`, scalers
`identity/amplification/attenuation`, 5 towers, residual connections, BN
(BatchRenorm) and Mish.  Only `gnn_intermediate_dim` and `output_node_dim` are
changed, and they are kept equal.

**Width rule (executed before training, CPU only).** Instantiate widths
`20, 25, 30, ..., 100` on CPU, count the actual total parameters of the full
upstream `Estimator`, and select the width with the smallest absolute
difference from 247,106.  No Validation label or score is read.  If the
relative difference exceeds 15 %, the selected row is labelled
`NOT_TIGHTLY_SIZE_MATCHED` and is still retained as the closest configuration;
it may not be called a strict parameter match.  The result is frozen in
`analysis/pna_width_search.json` before any formal training and cannot be
changed in response to model performance.

Selected result: **width 45**, **256,041 parameters**, absolute difference
**8,935**, relative difference **0.036159 (3.62 %)**, label
`TIGHTLY_SIZE_MATCHED`.

**Training.** Same as PNA-native v1: AdamW, LR `1e-4`, weight decay `1e-10`,
batch 128, gradient clip 0.5, bf16 autocast, maximum 360 epochs, early stopping
patience 30, `ReduceLROnPlateau(factor=0.5, patience=15)`, best Validation-loss
checkpoint, seed 0, no pretraining.  The PNA degree histogram is computed from
Train3360 only and must match the frozen v1 histogram tensor SHA256
`56bc94c1186a1a471b9af7885313ed914f51469f3e40fa99266a20c65721fc40`.

Only Train3360 and Validation420 structure objects are materialised.  No Test
structure is batched and no Test label is loaded.

---

## 5. Training and resource rules

* Exactly one training seed (`0`) per new model.  No seed sweep, no seed
  selection, no second seed.
* No hyperparameter search.  Every recipe value is frozen in
  `configs/controls_v2.yaml` and in `analysis/protocol_freeze.json`.
* Two formal training runs only.  Smoke runs are small and are charged if they
  touch CUDA.
* The GPU cap is **0.5 A100 GPU-hours**, charged as the full wall time of every
  process that touched CUDA, including data loading, smoke tests, the inference
  benchmark and any failed attempt.  CPU-only work is charged 0 GPU-hours and
  recorded separately.
* If the ledger reaches 0.5 GPU-hours, stop and report the completed part.
* No architecture change, no additional seed, no extra hyperparameter, and no
  performance retry after seeing a result.

---

## 6. Evaluation

Only Train3360 and Validation420 are evaluated.  Primary metrics: RMSE on raw
logD; also MAE and R².  The evaluator is the frozen v1 module
`r42_metrics.py`, cross-checked against NumPy and scikit-learn in the CPU gates.

Additional required records:

* nominal and effective trainable parameter counts;
* checkpoint rule and best epoch (or soup members);
* optimizer steps, GPU wall time, peak allocated GPU memory;
* repeated Validation420 inference on the same A100, same batch size
  (128) and same warmup (3 passes), with **preprocessing time and model-forward
  time measured separately** (10 timed repeats).  Preprocessing is
  collate/Batch construction plus host-to-device transfer; forward is the model
  call only.
* paired Validation420 error changes for LCRC-A vs S2-off, with a descriptive
  paired bootstrap fixed at 2,000 resamples and seed 0.  The interval is
  descriptive only; no significance or mechanism claim is made.

---

## 7. Test discipline

* No Test label is loaded.
* No Test prediction is produced, read or re-scored.
* No Test bootstrap or Test molecule-level error is computed.
* No historical Test metric, pre-test manifest, post-lock analysis or Test
  prediction file is opened.
* A static source guard rejects forbidden Test-artifact references; a runtime
  guard rejects Test-named prediction artifacts.
* The final table contains **Validation only**.  Train metrics are reported
  separately for completeness, not as a comparison target.

---

## 8. Pre-registered failure modes

| condition | consequence |
| --- | --- |
| any frozen data/history hash mismatch | abort before model construction |
| any CPU gate fails | abort before GPU use |
| config/code/width/gates change after freeze | abort before GPU use |
| GPU ledger ≥ 0.5 h | abort before the next GPU run |
| S2-off output depends on typed-S2 input | abort |
| replacement parameter has no gradient | abort |
| dead S2 input columns receive gradient | abort |
| PNA width changed after freeze | abort |
| any Test artifact access | abort |

---

## 9. Deliverables

```
configs/controls_v2.yaml
PROTOCOL.md
code/                 (c2_*.py, run_v2c.py)
tests/                (reuse, capacity, width, label isolation, metrics)
analysis/protocol_freeze.json (+ protocol_sha256.json)
analysis/reuse_audit.json
analysis/data_audit.json
analysis/pna_width_search.json
analysis/cpu_gates.json
analysis/gpu_ledger.json
analysis/runs/s2off_s0/            (checkpoint, train.json, predictions)
analysis/runs/pna_small_s45_s0/    (checkpoint, train.json, predictions)
analysis/validation_table.json
analysis/paired_bootstrap.json
analysis/inference_benchmark.json
checkpoints/          (two new checkpoints)
predictions/          (two new Validation prediction files)
REPORT.md
```

No Test prediction or Test metric is part of the deliverables.
