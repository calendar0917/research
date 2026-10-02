# Pre-registration — zinc_graph_dictionary_readout_v1

Round: `zinc_graph_dictionary_readout_v1` (Workstream Z, ZINC, local CPU only).
Study: `zinc-context-gap`. Protocol: `zinc-context-gap` (`test_policy: terminal`;
all non-terminal runs have `test_access=blocked`).

This note was frozen before the real ZINC fit/evaluate stages. It fixes the
parent checkpoint, the feature blocks, the dictionary rule, the three `lambda`
values, the solver precision/budget and the purchase gates. The negative result
below therefore excludes only this pre-registered configuration.

## 1. Question

Given the **already trained** Full `m=3` seed-0 parameter soup, can its 814-D
graph-level reader input `R` support a *fixed* graph-level prototype dictionary
plus a certified convex L2-regularised MAE readout that is clearly better than
the **same** frozen Full model's own MLP readout on the identical 1000 official
valid molecules? No backbone retraining, no new object, no architecture/HPO
search.

## 2. Frozen parent

- Checkpoint: `tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt`
- Checkpoint SHA-256 (recorded): `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`
- Exact parameters: `408651`; all parameters `requires_grad_(False)`.
- Backbone state (reader excluded) SHA-256: `ad7e1baacce1d7e075301a622d17df55d5508e6d4b6dd5abbfcf8a135417f2e7`
- Published in-protocol soup valid MAE: `0.1191540920053958`.
- Split fingerprint (zinc-context-gap): `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`.
- Only the existing official train (`10000`) and valid (`1000`) caches are
  loaded. The official test split is never instantiated.

## 3. Captured representation

A `register_forward_pre_hook` on `model.reader` captures the **actual tensor
that reaches the reader**; no first-hidden-layer proxy, no hand rebuild. The
Full reader input is exactly 814 dims in four blocks:

| block | width | note |
|---|---|---|
| unary `[sum(E), sum(E^2), count]` | 289 | C6 zeroes `count` |
| 5 static-pair buckets `[sum, sum^2, count]` | 485 | C6 zeroes `count` |
| global | 32 | C6 zeroes histograms |
| topology | 8 | |

`C6_MASK` is passed on every forward. Frozen-backbone replay of the captured
`R` through the original reader must match the saved per-molecule prediction to
`<= 2e-6`; the replayed official valid MAE must match `0.1191540920053958` to
`<= 2e-6`.

## 4. Frozen dictionary and head

The vendored package
`tracks/ksvd/experiments/luyin16/graph_dictionary_readout/v1_20261002/`
(`prototype_dictionary.py`, `run_cached_head.py`) is used unchanged.

- `R` is transformed by column-wise `asinh`, then standardised with mean/std fit
  only on head-fit rows; zero-variance columns are dropped; each block floor is
  `0.05 x` the median active std in that block; the four blocks are balanced by
  active dimension and non-empty block count. C6 zero slots stay off.
- `K = 256` shared prototypes, seed `20261002`, drawn without replacement from
  fit rows (no label/residual-based selection, no ID feature).
- Bandwidth is the median squared distance over 4096 fixed-seed random fit
  pairs (`sigma^2 = median(||z_a - z_b||^2)`), never scanned.
- Gaussian kernel `k(z, C) = exp(-||z - C||^2 / (2 sigma^2))`,
  `W = k(C, C)`, `psi = k(z, C) W^{-1/2}` with the code's fixed spectrum
  truncation (duplicate prototypes handled by a consistent pseudo-inverse).
- Direct prediction `median(y_fit) + [1, psi] w`; the constant atom is also
  L2-penalised; objective `mean|Aw - (y - median)| + (lambda/2) ||w||^2`.
- `lambda` grid fixed at `{1e-5, 1e-4, 1e-3}`. Head dev = every 10th canonical
  group (canonical certificate keys from the typed-cycle probe cache); the
  remaining groups are head fit. `lambda` is selected by head-dev MAE, ties to
  the **larger** `lambda`. On the selected `lambda` the scaler, prototypes,
  bandwidth and head are fit **once** on all 10000 train rows.

## 5. Solver certificate

ADMM with a valid dual lower bound. Every tried `lambda` and the final fit must
report `status=CONVERGED` and `primal-dual gap <= 1e-6`; single-fit cap
180 s (algorithm settings from the package). Any unmet certificate is
`INCOMPLETE_SOLVER`, never a scientific negative.

## 6. One paired screen and purchase gates

The frozen head is opened exactly once on the same 1000 official valid
molecules. Continue (positive signal) only if **all** hold:

1. absolute MAE improvement over the replayed same-checkpoint Full H0 readout
   `>= 0.006`;
2. new MAE `<= 0.113`;
3. at least four of five fixed-ID bins (sorted ID, stride 5) improve.

Otherwise `GRAPH_DICTIONARY_READOUT_STOP`; no K increase, no bandwidth scan, no
normaliser/object swap, no backbone training, and no new full neural run is
purchased. Deployment acceptance is assembled and run only after a positive
signal. The valid split has been reused historically and selected the Full
soup, so any positive is exploratory only, never an independent confirmation.

## 7. Provenance note

The bundle's synthetic NumPy/functional acceptance was re-run locally (all
gates passed); the provider's PyTorch deployment scaffold was **not** executed
by the provider and is re-validated here by a train-only real mini-batch gate.
