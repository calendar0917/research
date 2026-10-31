# PROTOCOL — `lcrc-lipo-esa-random42-v1`

Frozen protocol for one round of the *structural-sufficiency* line: four fixed
models, one fixed split, one training seed, one terminal Test evaluation.

| field | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-esa-random42-v1` |
| Track | `tracks/structural-sufficiency/` |
| Experiment root | `tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1/` |
| Created | 2026-02-05 |
| Split seed | `42` (fixed by the task statement) |
| Training seed | `0` (single seed, no seed selection) |
| Dataset | MoleculeNet `Lipophilicity` (4,200 molecules, target `exp` in logD) |
| Upstream code | `davidbuterez/edge-set-attention` @ `861a0b12d445ae80cc69b3948af623b4331321d4` |
| GPU budget | 6.0 A100 GPU-hours, hard cap (all smoke tests and retries included) |
| Terminal status on success | `COMPLETED_SINGLE_SEED_BENCHMARK` |

Everything below is *fixed before any formal run* and is enforced in code
(`code/r42_common.py:validate_config`, `code/r42_audit.py`,
`code/r42_gates.py`, `tests/`). The YAML in `configs/benchmark_v1.yaml` is the
machine-readable copy of the same constants.

---

## 1. Data

**Raw file.** `data/lipophilicity/Lipophilicity.csv`, SHA256
`aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`.
Columns: `CMPD_CHEMBLID`, `exp`, `smiles`. 4,200 rows, original file order kept,
no deduplication, no filtering, no re-canonicalisation.

**Multi-component molecule.** Original zero-based row `1646`
(`O.O.O.CC(=O)O[C@@]...`, CHEMBL3545252, `exp = 3.4`) contains more than one
fragment. It is *kept* in the dataset and in whatever split the frozen indices
assign it to. No special-casing, no fragment-merging, no drop.

**Split (frozen).**

```python
torch.utils.data.random_split(range(4200), [3360, 420, 420],
                              generator=torch.Generator().manual_seed(42))
```

The index lists are frozen in `data/split_esa_random42_v1.json` and are
byte-reproducible from the rule above (`code/r42_data.py:compute_split`).

| split | n | SHA256 of the `int64` index vector | first five rows |
| --- | --- | --- | --- |
| Train | 3360 | `0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee` | `3942, 3415, 1426, 2365, 3462` |
| Validation | 420 | `8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44` | `501, 1649, 3579, 2873, 106` |
| Test | 420 | `4bc449106a64833973b8e8d3d16097bf5d2906c788d7dfcdf02a4b3a30ffa49b` | `967, 3557, 3492, 462, 2258` |

The three splits are disjoint and their union is exactly `range(4200)`.

**Target normalisation.** `sklearn.preprocessing.StandardScaler` convention
(population standard deviation, `ddof = 0`) fitted **only** on the 3,360 Train
labels: `mean = 2.2081011904761905`, `scale = 1.1906281212426577`.
Training uses this standardised target; every reported metric is denormalised
back to raw logD. Validation and Test labels never enter the fit.

**Feature normalisation (LCRC only).** The LCRC graph normalisation statistics
are fitted on Train3360 structures only. PNA and ESA use the upstream raw
Chemprop features (no extra normalisation beyond the models' own layers). XGB
preprocessing (variance filter, median imputation) is fitted on Train3360 only.

**Structures.** Both graph models consume the same upstream Chemprop
featurisation of the *same* 4,200 molecules with explicit hydrogens
(`Chem.AddHs`): 79 atom features, 13 bond features, 203,750 atoms, 428,162
directed edges, `max_node = 45`, `max_edge_global = 438`. The materialised
artifact is `data/structures_esa_random42_v1.npz`
(SHA256 `a098d27f...`), verified against re-derivation.

---

## 2. Models (all fixed, no search)

### 2.1 `LCRC-A-R3-scratch`

Local LCRC model from the frozen preparation track (`lcrc_model.LCRC` with the
`lipo_v1` graph spec: atom/bond/path content channels and a local-content
aggregator). 247,106 parameters, **no attention, no junction/relation term**.

| | |
| --- | --- |
| Optimiser | AdamW, `lr 3e-4 → 1e-5`, per-step cosine schedule |
| Weight decay | `1e-3` |
| Dropout | `0.10` |
| Gradient clip | `5.0` |
| Batch size | 128 |
| Epochs | 360 (fixed; no early stopping) |
| Loss | MSE on standardised logD |
| Checkpoint | average of the **last 5 epoch-end `state_dict`s** |
| Self-supervised / external pretraining | none |

### 2.2 `PNA-native`

Upstream `gnn/graph_models.py: Estimator(conv_type="PNA")`, i.e. the original
PNA stack with residual connections, `BN` (BatchRenorm) and `Mish`.

| | |
| --- | --- |
| Layers | 4 PNA conv blocks, hidden 256 → 255 (nearest multiple of 5) |
| Aggregators / scalers / towers | `mean, min, max, std` / `identity, amplification, attenuation` / 5 |
| Degree histogram | computed from **Train3360 graphs only**, saved in the upstream layout (`saved_PNA_degrees/degrees.pt`) |
| Optimiser | AdamW, `lr 1e-4`, `weight_decay 1e-10` |
| Gradient clip | `0.5` |
| Batch size | 128 |
| Epochs | max 360, early stop patience 30 |
| LR schedule | `ReduceLROnPlateau(factor=0.5, patience=15)` |
| Precision | bf16 autocast (formal runs) |
| Checkpoint | **best Validation loss** |

7,767,349 parameters.

### 2.3 `ESA-native`

Upstream `esa/models.py: Estimator` over `esa/masked_layers.py`, Edge Set
Attention applied **on edges**, no positional encoding, no external
pretraining, no pretrained weights.

| | |
| --- | --- |
| Layer stack | `S, S, M, M, M, S, P` (7 layers) |
| Width / heads | 256 × 7 layers / 8 heads |
| Graph readout | PMA with 32 seeds → `graph_dim = 512` |
| MLP | `gated_mlp`, 2 layers, hidden 512, LN, post-norm |
| Masked attention backend | upstream **`torch`** path (`scaled_dot_product_attention`), bool mask semantics verified equal to the xformers additive-bias convention |
| `num_max_items` | `nearest_multiple_of_8(438 + 1) = 440` directed-edge slots |
| Optimiser | AdamW, `lr 1e-4`, `weight_decay 1e-10` |
| Gradient clip | `0.5` |
| Batch size | 128 |
| Epochs | max 360, early stop patience 30 |
| LR schedule | `ReduceLROnPlateau(factor=0.5, patience=15)` |
| Precision | bf16 autocast (formal runs) |
| Checkpoint | **best Validation loss** |

10,351,105 parameters. Four parameters
(`mlp_norm.ln.{weight,bias}`, `st_fast.decoder.0.norm.ln.{weight,bias}`) are
unreachable in this upstream edge-attention + PMA configuration and therefore
receive no gradient; this is recorded in `analysis/cpu_gates.json` and is
upstream behaviour, not a change made in this round.

### 2.4 `XGB-P3`

XGBoost 2.1.3 on the frozen v2 `P2` RDKit descriptor list (217 sorted names),
variance filter and median imputation fitted on Train3360.

| | |
| --- | --- |
| Trees | `n_estimators = 800`, `max_depth = 5`, `min_child_weight = 1` |
| Learning rate | `0.03` |
| Sampling | `subsample = 0.8`, `colsample_bytree = 0.8` |
| Regularisation | `reg_lambda = 5.0` |
| Tree method | `hist` |
| Seed / threads | `random_state = 0`, `n_jobs = 4` |
| GPU | none (CPU only; GPU-hours contribution = 0) |

---

## 3. Training rules

* One training seed: `0`. No seed sweep, no seed selection, **no second seed**.
* No hyperparameter search of any kind: every value above is fixed in
  `configs/benchmark_v1.yaml` and hard-checked against `r42_common.py`.
* No self-supervised or auxiliary objectives; no external pretraining; no
  additional features, modules, or ensembling beyond the checkpoint rules.
* Validation is used only for early stopping / best-checkpoint selection (PNA,
  ESA) and for reporting; never for gradient updates.
* Test labels are never read during training. The label artifact
  `data/labels_train_valid_v1.json` contains Train and Validation only and the
  accessor raises on `"test"`.

---

## 4. Metrics

Primary metric: **RMSE** on raw logD. Also reported: MAE and R² (raw logD).
All three are computed by an independent module (`code/r42_metrics.py`) and
cross-checked against `sklearn.metrics` and a NumPy reference in the unit tests.
Validation metrics are computed once per model from the frozen checkpoint;
Test metrics once, terminal, after the freeze described in §5.

Reporting rules:

* the comparison table reports Train (3360), Validation (420) and Test (420)
  RMSE plus Test MAE and Test R², parameter counts and GPU-hours;
* XGB is reported with `GPU-h = 0` and its CPU wall time separately;
* the round is a **single-seed** benchmark and is labelled *exploratory*; no
  significance test is claimed and no literature number is mixed in.

---

## 5. Test discipline (one shot)

Fixed order:

1. all four models are trained and their Training/Validation metrics recorded;
2. checkpoints and Test predictions are frozen into
   `checkpoints/` and `predictions/`, with per-file SHA256;
3. `analysis/pre_test_manifest.json` is written, listing every model, its
   checkpoint hash, its frozen Test prediction hash, the split hash, the
   configuration hash and the charged GPU-hours; its SHA256 is published;
4. `analysis/metrics_test.json` and `analysis/prediction_manifest.json` are
   produced by a single scoring pass guarded by `analysis/test_read.lock`,
   which refuses to run a second time or against a changed manifest.

Consequences, enforced in `code/r42_score.py`:

* exactly one scoring pass; a second invocation fails;
* no retraining, no checkpoint re-selection, no metric hunting after Test;
* if a model cannot be completed, its absence is recorded in the manifest's
  `missing_models` list and the round ends with the partial table clearly
  marked instead of silently dropping a row.

---

## 6. Budget

The hard cap is **6.0 A100 GPU-hours**, charged as the full wall time of every
process that touched CUDA (data loading, smoke tests, failed attempts and
retries included). CPU-only work (all audits, gates, unit tests, XGB) is
charged 0 GPU-hours but its CPU wall time is recorded separately. There is no
automatic extension; exceeding the cap aborts the round before a new run.

`analysis/gpu_ledger.json` is rebuilt from the per-run `envelope.json` files and
is the single source of truth for the charged hours.

---

## 7. Pre-registered failure modes

| condition | consequence |
| --- | --- |
| CSV / split / structure hash mismatch | abort before training |
| Any CPU gate fails | abort before GPU use |
| Unit test fails | abort before GPU use |
| GPU ledger ≥ 6.0 h | abort before the next run |
| Test labels reachable during training | abort |
| Second Test scoring attempt | abort |

Upstream adaptations that were unavoidable on the available environment are
declared up front and listed in `analysis/implementation_manifest.json`:
`pytorch_lightning` → `torch.nn.Module` runtime stub, `bitsandbytes.AdamW8bit`
→ `torch.optim.AdamW` (the formal optimiser), `flash_attn.swiglu` →
`F.silu(x) * y`, `admin_torch` OmegaResidual reproduced verbatim, `xformers`
unavailable so the upstream `torch` attention backend is used everywhere, and
device derivation from `edge_index` so that `cuda:0` behaviour is preserved on
CPU gates. Vendored upstream files are byte-identical to the pinned commit.

---

## 8. History and no-reuse declaration

The following frozen results are used **read-only** for context, never for
initialisation, checkpoint reuse, seed selection or model choice:
`lcrc_lipo_screen_v1`, `lcrc_lipo_mechanism_generalization_ssl_v2`,
`lcrc_lipo_chem_feature_pilot_v3`, `lcrc_lipo_local_chem_pilot_v4`,
`tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1`.

Because those earlier rounds already touched these same 4,200 molecules in
other splits, the random42 Test set is **not** a never-seen blind set in the
strict sense. This is disclosed in REPORT.md; no Test number in this round is
presented as a blind generalisation estimate.

---

## 9. Deliverables

```
configs/benchmark_v1.yaml
data/split_esa_random42_v1.json
data/labels_train_valid_v1.json
data/structures_esa_random42_v1.npz (+ _meta.json)
code/            (r42_*.py, vendored upstream/)
tests/           (split, label isolation, metrics, mask semantics, model fidelity)
analysis/data_audit.json
analysis/implementation_manifest.json
analysis/cpu_gates.json
analysis/gpu_ledger.json
analysis/pre_test_manifest.json (+ .sha256)
analysis/prediction_manifest.json
analysis/metrics_validation.json
analysis/metrics_test.json
checkpoints/
predictions/
PROTOCOL.md
REPORT.md
```

---

## 10. Frozen hashes

| artifact | SHA256 |
| --- | --- |
| `configs/benchmark_v1.yaml` | recorded in `analysis/protocol_freeze.json` |
| canonical config JSON | recorded in `analysis/protocol_freeze.json` |
| `data/split_esa_random42_v1.json` | recorded in `analysis/protocol_freeze.json` |
| `data/labels_train_valid_v1.json` | recorded in `analysis/protocol_freeze.json` |
| `data/structures_esa_random42_v1.npz` | `a098d27f...` (full value in `analysis/data_audit.json`) |
| `PROTOCOL.md` | this file; digest published in `analysis/protocol_sha256.json` and in REPORT.md |

This document may not be edited after the first formal training run. Any
post-hoc deviation is listed in REPORT.md, not patched in here.
