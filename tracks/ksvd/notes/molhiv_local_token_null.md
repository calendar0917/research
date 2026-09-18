# MolHIV local-token channel necessity

Protocol: `molhiv_local_token_null_v1` (pre-registration:
`notes/molhiv_local_token_null_preregistration.md`, committed before any
candidate training). Protocol id `molhiv-cross-scaffold-interaction`,
`test_policy: terminal`. **The official test split was never scored**; all
candidate runs record `official_test_loaded: false`.
Training revision **`e163ac6`** (data_sanity re-recorded at `ccac6e7`), GPU 1
(A100-SXM4-40GB; GPU 0 held an unrelated 35.8 GB / 94 % task that was never
touched).

## Question

The frozen MolHIV recurrent pair--centre model is 1,076,589 params, of which
the dense exact-certificate lookup `typed_embedding [26233, 32]` is 839,456
(78.0 %) and the radius-1 `parent_embedding [103, 16]` is 1,648; the fixed
backbone is 235,485. Is that molecule-dependent local 32-D patch token
actually necessary, or does the fixed downstream backbone carry almost all
predictive power?

## Conditions (only the local-token channel changes)

| condition | local token | generator params | total params | drop |
|---|---|---|---|---|
| typed (reference) | `typed_embedding(token)`, width 32 | 839,456 | **1,076,589** | — |
| B-Null | exact zero 32-D per patch, no generator | 0 | **237,133** | 839,456 (77.97 %) |
| Constant-32 | one trainable graph-wide 32-vector | 32 | **237,165** | 839,424 (77.97 %) |

The patch-encoder input width (841 = 809 shell + 32 token + 16 parent) and
every other module, loss, optimizer, epoch/patience budget and the fixed
equal-weight Top-5 soup rule are identical to the frozen reference.

## Results (official valid ROC-AUC, higher better)

Frozen typed reference (seed 0): raw 0.814239, **soup 0.809505**, best ep 16,
56 epochs. (Seed 1 typed: raw 0.849865, soup 0.851092 — i.e. the two existing
typed seeds already span 0.0416 AUC soup.)

| candidate | params | best ep | epochs | wall | raw AUC | **soup AUC** | soup − typed s0 |
|---|---|---|---|---|---|---|---|
| B-Null seed0 | 237,133 | 24 | 64 | 1357 s | 0.843168 | **0.840847** | **−0.031342** |
| Constant-32 seed0 | 237,165 | 40 | 80 | 1670 s | 0.856445 | **0.860768** | **−0.051263** |

Top-5 soup epochs/values: Null `[24, 23, 35, 22, 40]` =
`[0.8432, 0.8425, 0.8410, 0.8378, 0.8363]`; Constant `[40, 76, 77, 75, 22]` =
`[0.8564, 0.8564, 0.8515, 0.8504, 0.8497]`.

Pre-registered gate on `D = typed_soup_s0 − null_soup`:
`D = −0.031342 <= 0.005` → **STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED**,
`authorize_constant: true` (Constant was then run and reported).

## Mechanism witness (post-training soup)

* Null: total 237,133; `typed_embedding` not instantiated; token exactly zero
  on a real valid batch (3,227 patches, mean norm 0, max abs 0); patch-encoder
  input width 841 unchanged; all 9 remaining backbone groups receive non-zero
  task gradient (parent 1.9e-3 … pair_encoder 3.8e-2); forward finite
  (logits std 1.81); `official_test_loaded: false`.
* Constant: total 237,165; no typed embedding; learned graph-wide vector norm
  0.01287, max |value| 0.00569, per-dim std 0.00212; the constant itself
  receives non-zero gradient (2.5e-4); all 10 groups non-zero; forward finite
  (logits std 3.20); `official_test_loaded: false`.

## Interpretation

**Pattern 1 + Pattern 2, replicated on MolHIV and larger.** Deleting the entire
dense certificate lookup (78 % of the model) does not hurt: B-Null seed0 soup
0.840847 sits **above** the matched typed seed0 (0.809505, +0.0313) and inside
the typed two-seed band (0.8095 – 0.8511). Replacing the deleted lookup with a
single 32-parameter graph-wide vector (Constant-32) reaches 0.860768, **above
both typed seeds** (+0.0513 vs s0, +0.0097 vs s1). The local 32-D slot's
residual value is a shared trainable offset, not molecule-dependent identity —
exactly the ZINC finding (`decision-local-token-null-20260919`, where
Constant-16 matched B-Full within +0.0007).

This is also the training-side counterpart of the identity audit
(`claim-identity-incremental-information-audit-20260915`): the exact certificate
identity adds no molecule-level distinguishing power on ZINC, and here the
839k-param table that stores it is empirically unnecessary.

### Honest limits

* **Single seed per candidate.** MolHIV is noisy at this scale: the two frozen
  typed seeds differ by 0.0416 soup, and a MolHIV run is only ~20 s/epoch, so
  the pre-registered bands were deliberately coarse and this round is a
  **screen**, not a statistical claim. The point estimates are at least as good
  as the reference but the constant-32 > typed-seed1 margin (0.0097) is inside
  the observed seed spread.
* The typed seed0 reference early-stopped at epoch 16 while the candidates ran
  64/80 epochs under the identical patience rule; the comparison is the
  pre-registered fixed-soup rule, not a matched-epoch curve.
* No official test was scored and no parameter re-investment was attempted;
  the 839k released parameters are simply deleted.

## What was not done

No official test; no multi-seed upfront; no re-investment of the released
839k/839k params; no backbone widening; no new structural feature/radius/RRWP/
cycles/pair network/attention; no hyperparameter or optimizer sweep; no rescue
after a good B-Null; no remote tracked-code editing; frozen MolHIV reference
code path unchanged (default `patch_representation="typed_lookup"` is
bit-identical).

## Artefacts

`results/molhiv_local_token_null/`:
`parameter_accounting.json`, `data_sanity.json`, `smoke_{null,constant}.json`,
`run_{null,constant}_seed0.json`, `witness_{null,constant}_seed0.json`,
`decision.json`, `report.json`, `curves/`, `snapshots/`.
Code: `experiments/luyin16/molhiv_local_token_null.py` (+ the `null`/`constant`
option in `molhiv_patch_path_pooling.py` and `molhiv_recurrent_pair_centre.py`);
tests `tests/test_molhiv_local_token_null.py` (13 pass, with
`test_molhiv_patch_path_center_context.py` and
`test_molhiv_recurrent_pair_centre.py`: 20 pass).
