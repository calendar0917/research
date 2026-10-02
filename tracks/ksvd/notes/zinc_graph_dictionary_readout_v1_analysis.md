# Analysis — zinc_graph_dictionary_readout_v1

Round: `zinc_graph_dictionary_readout_v1`, local CPU only, seed 0, official
ZINC train 10000 / valid 1000, **official test never instantiated**.

## 1. Result

Single paired screen on the frozen official valid split:

| readout (same frozen Full backbone) | official-valid MAE |
|---|---|
| original Full `814 -> 39 -> 39 -> 1` H0 readout (replayed) | **0.1191540920053958** |
| fixed graph-level prototype dictionary readout (K=256, `lambda=1e-5`) | **0.2482236020672727** |

- absolute gain `-0.1290695100618769` (gate `>= +0.006` failed);
- new MAE `0.248224` (gate `<= 0.113` failed);
- five fixed-ID bins gains `[-0.129204, -0.112836, -0.133628, -0.094671, -0.175008]`,
  `0/5` improved (gate `>= 4/5` failed);
- per-molecule improvement fraction `0.259`; median gain `-0.081157`.

Verdict: **`GRAPH_DICTIONARY_READOUT_STOP`**. No frozen gate was met, so the
deployment single-unit was not assembled and a new full neural training run is
**not** purchased this round.

## 2. What was verified before the screen

The result is not an integration or numeric artefact. All replay gates passed:

- captured `R` replayed through the original reader: max per-molecule delta
  `0.0` (train) / `0.0` (valid) against the saved `p_base`;
- replayed official valid MAE `0.1191540920053958` vs published
  `0.1191540920053958`, delta `0.0`;
- 814-D block geometry `289 + 485 + 32 + 8 = 814`; hook fires exactly once per
  batch; single-graph vs batched and reversed-order replay within `<= 4.8e-7`;
- C6 mask applied on every forward, so the C6 zero slots (`unary count`,
  `pair count`, `path_count`) stay zero-variance and are dropped by the scaler;
- canonical train duplicate groups attached in official row order (order
  verified by the full 10000-row label vector; 9997 unique groups);
- the frozen backbone hash is unchanged across export/fit/evaluate
  (`ad7e1baa...`).

Head fit (train only; the valid cache is not accepted by the fit stage):

- head-dev MAE: `1e-5 -> 0.266476`, `1e-4 -> 0.267175`, `1e-3 -> 0.301175`;
- selected `lambda = 1e-5`; final solver `CONVERGED`, gap `9.47e-7`,
  `660` iterations, `0.45 s`; head train MAE `0.221953`.

The final full-train head MAE `0.222` is already about twice the frozen Full's
own valid error, while the frozen Full's **train** MAE is `0.0454`. That gap is
the direct signature of the result: a 257-coefficient fixed-kernel prototype
readout underfits the 814-D representation that a trained 814-39-39-1 MLP
(≈33k readout parameters, and the whole frozen body upstream) fits.

## 3. Deployment scaffold re-acceptance

The provider's `torch_readout_scaffold.py` was never executed in the provider
environment. It was re-validated here by a **train-only** real mini-batch gate
(128 official-train graphs), explicitly **not** a deployment acceptance and it
never touches the official valid split:

- `GraphPrototypeReadout` vs the cached NumPy folded head: max `1.92e-07`
  (`<= 1e-5`), MAE delta `4.56e-09` (`<= 2e-6`);
- exactly one backbone forward; kernel computed in float64 and returned in the
  original float32 dtype; prototype/scaler tensors are buffers; no online fit;
- `state_dict` save/reload replays bit-for-bit (`0.0`); the official-test
  blocker raises.

Because the screen is negative, full deployment acceptance is
`NOT_RUN_NO_POSITIVE_SCREEN` and no deployed unit is claimed.

## 4. Interpretation and limits

- This is a clean, certified negative for **one fixed** configuration:
  Gaussian kernel on train-selected prototypes, Nyström whitening, direct MAE +
  L2 head, `K=256`, `lambda in {1e-5,1e-4,1e-3}`. It does not prove the 814-D
  representation is theoretically sufficient or insufficient, and it does not
  refute other readout families.
- It never re-uses the old weak `0.36 -> 0.33` ridge proxy as evidence, and it
  does not treat any inactivated-ablation delta as a gain.
- The official valid split has been reused across many historical rounds and
  selected the Full soup, so this is an **exploratory screen**, not an
  independent confirmation or a multi-seed conclusion.
- Per the frozen rule, the correct action is to seal the round: no K increase,
  no bandwidth scan, no normaliser or object swap, no backbone training, and no
  new full neural training run.

## 5. Evidence discipline

- Provider bundle: synthetic NumPy/functional acceptance only (re-run locally;
  all gates passed). No ZINC or PyTorch execution by the provider.
- This round: real Full replay, real frozen train-only head fit, one paired
  official-valid screen, and a train-only scaffold mini-batch gate.
- `official_test_loaded = false` in every payload.