# PROTOCOL — `lcrc-lipo-contextual-contrib-v1`

Frozen protocol for one matched two-arm comparison on the frozen random42 split.
Everything below is fixed **before any formal training run** and is enforced in
code (`code/x1_common.py:validate_config`, `code/x1_gates.py`, `tests/`).

| field | value |
| --- | --- |
| Protocol ID | `lcrc-lipo-contextual-contrib-v1` |
| Track | `tracks/structural-sufficiency/` |
| Experiment root | `tracks/structural-sufficiency/lcrc_lipo_contextual_contrib_v1/` |
| Data source (read-only) | `tracks/structural-sufficiency/lcrc_lipo_esa_random42_v1/` |
| Split seed | `42` |
| Training seed | `0` (single seed, no seed selection) |
| Split | Train3360 / Validation420 only |
| Test access | forbidden: no Test label, prediction, metric or per-molecule Test information |
| Arms | `BASE`, `CONTEXT` |
| GPU budget | **0.45 A100 GPU·h hard cap**, all CUDA smoke and retries included |
| Host / pools | `res`; BASE → `res-gpu0`, CONTEXT → `res-gpu1`, in-process device `cuda:0` |

---

## 1. Data (frozen, read-only)

Read (hash-verified, never written):

* `lcrc_lipo_esa_random42_v1/data/split_esa_random42_v1.json`
  SHA256 `1834414a6da73e9d2c5942ad157a345f08653110acfc4b9fef83c3064bc233b4`
* `lcrc_lipo_esa_random42_v1/data/labels_train_valid_v1.json`
  SHA256 `1da269356e0e89974a7388d1e430d7e810a0ab1056be5cd747a1c8ed3d2009ca`

| split | n | row-index SHA256 (`int64`) | first five rows |
| --- | ---: | --- | --- |
| Train | 3360 | `0df18c72753974d502f8bede1cd2aa5c0ebdc1df450b8871fffdb1f67d2648ee` | `3942, 3415, 1426, 2365, 3462` |
| Validation | 420 | `8f9c35171ac2125e678d2ad44d87b293719bb9806a528810ac271e9aa39f9b44` | `501, 1649, 3579, 2873, 106` |

The label artifact is certified `test_labels_included: false`; its accessor
raises on `"test"`. Target normalisation is `StandardScaler`-style with
population std (`ddof = 0`) fitted **only on Train3360**:
`mean = 2.2081011904761905`, `scale = 1.1906281212426577`. Every reported metric
is in raw logD.

Structures are the frozen `lipo_v1` LCRC cache (content fingerprint
`23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03`,
`labels_present: false`). LCRC feature normalisation is fitted on Train3360
structures only; fingerprint `4d6e5bf456ff9d2217eb11a2c4a29d3f7cbbcb95665bf56e62b93c3ae93d0e26`.

The frozen split JSON is read as a whole, but **no Test row value is used**;
only the declared Test size is checked. No Test artifact path is ever opened
(runtime `open` audit, gate `test_isolation`).

---

## 2. Models

### 2.1 BASE

The original `lcrc_model.LCRC` from
`lcrc_cross_task_preparation_v1/code/lcrc_model.py`
(SHA256 `d5af76cb9b5998e6bfa068afd28c483c93dde579cfb85344ed928582b05c6d37`),
`lipo_v1` schema, width 64, 247,106 parameters, unchanged: atom encoder, bond
encoder, S1/S2 local content encoder, atom-contribution MLP
`f: 64 -> 64 -> 1`, global correction `rho`.

### 2.2 CONTEXT

CONTEXT keeps every BASE module and adds exactly four `Linear` layers.  Let
`h_i in R^64` be the output of `_encode(batch)` for atom `i` of molecule `g`
(after the dropout hook described in §3).

Atom roles are decided by the frozen `lipo_v1` `atom_type` index and
`spec.atom_type_metadata["element"]`:

* `hetero`: element in `{N, O, S, P}`;
* `carbon_halogen`: element in `{C, F, Cl, Br, I}`;
* all other elements do not participate in the role aggregation.

These are **fixed element classes only**.  They are *not* claimed to be strict
HBA/HBD or physical hydrophobicity labels.

For each molecule `g` in the batch:

```
m_het[g] = mean_{i in hetero(g)}          h_i      (exact zero if the set is empty)
m_ch[g]  = mean_{i in carbon_halogen(g)}  h_i      (exact zero if the set is empty)

a_i    = tanh(W_atom h_i)
p_g    = tanh(W_het  m_het[g])
c_g    = tanh(W_ch   m_ch[g])
change_i = W_out(a_i * p_g * c_g)

h_new_i = h_i + change_i   if i in hetero(g)
h_new_i = h_i              otherwise
```

with fixed `d = 32` and exactly four layers:

| layer | shape | initialisation |
| --- | --- | --- |
| `W_atom` | `Linear(64, 32)` | PyTorch default |
| `W_het` | `Linear(64, 32)` | PyTorch default |
| `W_ch` | `Linear(64, 32)` | PyTorch default |
| `W_out` | `Linear(32, 64)` | weight `Normal(0, 1e-3)`, bias `0` |

Final readout:

```
y = sum_i f(h_new_i) + rho(G, mean_i h_i, mean_i h_i^2, log1p(n_atoms))
```

`f` is the original full `64 -> 64 -> 1` nonlinear MLP and is not replaced or
reduced.  `rho` uses the **original** `h`, never `h_new`.  There is no separate
molecule-level residual, no prototype, no attention, no message passing, no
extra descriptor, no SSL/auxiliary loss.

Shared initialisation: BASE and CONTEXT are built at seed 0; CONTEXT then
overwrites every state entry that also exists in BASE with BASE's value, so all
shared parameters and buffers are bit-identical (re-verified in the gates).

New parameter count: `3 * (64*32 + 32) + (32*64 + 64) = 8,352`.
CONTEXT total: `247,106 + 8,352 = 255,458`.

---

## 3. Training recipe (identical for both arms)

* AdamW, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `1e-3`;
* peak LR `3e-4` → minimum LR `1e-5`, per-step cosine schedule;
* batch size `128`, 360 epochs, 27 optimizer steps per epoch, 9,720 total steps;
* gradient clipping `5.0`;
* loss: MSE on the Train3360-standardised target;
* checkpoint: average of the last **5 epoch-end `state_dict`s**, no early
  stopping, no checkpoint selection on Validation;
* dropout: the frozen `v2_models.DropoutAtLocal` (imported unchanged,
  SHA256 `00b3d3c3657cb7320affd79d8829110ad8d34ca8327191781882f225853c2988`)
  attached to `model.local`, `p = 0.10`, **actually executed during training**
  for both arms.  The global torch RNGs are reseeded after model construction,
  so both arms see the same dropout mask sequence.

---

## 4. Pre-GPU correctness gates (12)

No CUDA work before all twelve pass on CPU:

1. random42 Train/Validation rows, sizes, order, hashes match the frozen file.
2. No Test label/prediction/metric/per-molecule artifact is opened (runtime
   `open` audit + static scan + Test-label accessor raises).
3. BASE equals the original `lcrc_model.LCRC` at the same seed/input (state
   dict and forward exactly equal).
4. CONTEXT shared initialisation is bit-identical to BASE, tensor by tensor.
5. `W_out` forced to zero makes CONTEXT equal same-weight BASE within `1e-5`.
6. `hetero` / `carbon_halogen` masks match `spec.atom_type_metadata`.
7. Atom renumbering within molecules and molecule reordering within a batch do
   not change predictions.
8. Per-molecule context does not mix across molecules in a batch.
9. All four new `Linear` layers receive finite non-zero gradients after two
   real optimizer steps (and every new tensor is updated).
10. `rho` reads only the original `h`, never `h_new` (captured `rho` input is
    exactly the reconstruction from `h_original`, while the `h_new`
    reconstruction differs).
11. CPU two-batch forward/backward/optimizer step succeeds for both arms.
12. Per-molecule denormalisation and RMSE/MAE/R² recompute correctly.

Supplementary: dropout actually active in train mode and inactive in eval
mode; parameter counts; role coverage.

Results are written to `analysis/cpu_gates.json` (`ok: true`, `12/12`).

---

## 5. Metrics and fixed decision rule

Primary metric: Validation420 RMSE in raw logD.

```
gain = RMSE(BASE) - RMSE(CONTEXT)        # positive = CONTEXT better
```

Reported: Train and Validation RMSE/MAE/R²; parameter counts; total SSE
difference; number of Validation molecules whose absolute error improves;
mean magnitude of the learned correction; `mean |f(h_new) - f(h_original)|`;
hetero atom counts and molecule coverage of the two roles.

Pre-registered exploratory rule:

* `PROMISING_EXPLORATORY` iff `gain >= 0.020` **and**
  `MAE(CONTEXT) - MAE(BASE) <= 0.010`;
* otherwise `NO_CLEAR_GAIN`.

Tail description uses only the five row IDs pre-identified by
`lcrc_lipo_random42_gap_audit_v1`:
`{1630, 2924, 822, 1153, 3634}`.  No new tail search is performed and no
subgroup boundary is re-optimised.

Pre-registered tail rule:

```
KNOWN_TAIL_CONCENTRATED iff net SSE improvement > 0 and
    (share of net SSE improvement from the five rows >= 0.50
     or gain after removing the five rows <= 0)
```

If the rule fires, the report must not claim independent generalisation of the
gain.  Subgroups are copied exactly from the gap audit: scaffold-seen/unseen,
heavy-atom quartiles, TPSA quartiles, MolLogP quartiles.

The historical random42 LCRC Validation RMSE `0.589007` is a **descriptive
reference only**: the historical training loop configured dropout 0.10 but never
executed it, so neither arm of this round is a matched replica of that run.

---

## 6. GPU budget and run plan

Hard cap **0.45 A100 GPU·h**, charged as the full rr-measured process lifetime
of every CUDA process (smoke, training, validation, failures/retries).  Both
parallel pools are summed.  There is no extension.

| pool | arm | command |
| --- | --- | --- |
| `res-gpu0` | BASE | `rr run res base-s0 --pool res-gpu0 -- ... train --arm base --device cuda --pool res-gpu0` |
| `res-gpu1` | CONTEXT | `rr run res context-s0 --pool res-gpu1 -- ... train --arm context --device cuda --pool res-gpu1` |

One CUDA smoke (both arms, short) precedes the two formal runs; the two formal
runs execute in parallel, one per GPU.  Only this seed, no extra configuration,
no extra fold, no additional model.

`analysis/gpu_ledger.json` is the single source of truth and records host, GPU
model, driver, CUDA, torch, training commit, wall time, GPU time, steps and
gradient-clip events.

---

## 7. Abort conditions

* any data/split/label/cache/normalisation hash mismatch;
* any CPU gate or unit test failure;
* a changed config or code hash after the protocol freeze;
* an arm launched on the wrong pool;
* GPU ledger at or above the cap before a new run;
* any Test label / Test prediction access.

---

## 8. Deliverables

```
configs/contextual_contrib_v1.yaml
code/  (x1_*.py, run_x1.py)
tests/
analysis/data_audit.json
analysis/cpu_gates.json
analysis/protocol_freeze.json (+ protocol_sha256.json)
analysis/smoke_cpu/smoke.json
analysis/smoke_cuda/smoke.json
analysis/runs/base_s0/{checkpoint.pt,train.json,curve.csv,diagnostics.json,train_predictions.npz,valid_predictions.npz,envelope.json}
analysis/runs/context_s0/{...}
analysis/rr_status/*.json
analysis/gpu_ledger.json
analysis/metrics_validation.json
analysis/comparison.json
analysis/context_usage.json
analysis/tail_audit.json
analysis/subgroups.json
analysis/decision.json
analysis/per_molecule_valid.csv
REPORT.md
DECISION.md
```

This document may not be edited after the protocol freeze.  Any post-hoc
deviation is listed in `REPORT.md`, not patched here.
