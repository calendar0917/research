# REPORT — E2E-DictEnv-Hier-Relation-v1 (seed 0, local CPU)

Round: `zinc-e2e-dictenv-hier-relation-v1` (`e2e_dictenv_hier_relation_v1`); protocol `zinc-context-gap`;
authoring revision `cc4cf83dbce3c40977c36ff5125a47c73afd525c`; device CPU
(8 threads); official ZINC **test never loaded**.

## 0. Decision

**`HIERREL_STOP`** — soup valid MAE **0.182343**
(best 0.193210 @ epoch 316) falls in the
pre-registered **`stop`** interval (threshold inf).
Mechanism (reported separately): relation dictionary carries prediction.

## 1. Endpoint

| quantity | value |
|---|---|
| soup valid MAE | 0.182343094 |
| soup members (Top-5 by valid MAE) | [291, 294, 313, 316, 320] |
| member valid MAE | [0.193753, 0.195304, 0.194067, 0.19321, 0.195031] |
| best single epoch | 0.193209686 @ 316 |
| valid first / last / tail-20 mean | 0.574420 / 0.195031 / 0.202198 |
| soup train MAE (official train 10000) | 0.091181 |
| train–valid gap | -0.091162 |
| epochs run | 320 / 320 (completed=True) |
| seconds / epoch, wall clock | 4.128 s, 1320.9 s |
| peak RSS | 1.77 GB |
| trainable parameters | 145961 (expected 145961) |
| buffer elements (P + scalers) | 5039 |

## 2. Section-checked architecture

* 712-D node input `[Joint709 (709) ; common1 ; size2]`, frozen joint scaler reused
  bit-identically (joint block sha256 verified against the Joint709 cache).
* node dictionary `712 x 128`,
  all 128 coordinates kept, column-normalised,
  no top-k, no message passing, no node write-back.
* relation object 228 dims = S(mu) 96 + S(delta) 96 + cross 32 + bond one-hot 4;
  one undirected physical edge counted once.
* relation dictionary 228 x 64,
  tied-IHT s=8, steps=10.
* readout 586 =
  384 node (sum/mean/std) + 192 relation (sum/mean/std) + 2 log-counts + 8 topology.
* single head 586 -> 64 -> 32 -> 1 (`SiLU`, dropout 0.05).

## 3. Frozen train-only preparation

| item | value |
|---|---|
| P seed / shape / sha256 | 20261001 / [128, 32] / `e22f997853d75c70…` |
| node dict init | 32768 sampled train rows, effective rank 128, supplements 0 |
| node dict init sha256 | `08f24b380f930b6e…` |
| relation scaler fit rows (edges) | 249279 |
| relation scaler scaled block energy | {'bond': 0.9999999999996672, 'delta': 0.9999999999999896, 'mu': 0.9999999999999893, 'mu_delta': 0.9999999999999685} |
| relation dict init | 32768 sampled train edges, effective rank 64, supplements 0 |
| relation dict init sha256 | `55c31d86bfb568e4…` |
| init state sha256 | `926cddc6bd3520e2…` |
| soup / best state sha256 | `757596b2b64c85ae…` / `769ce9f888ec1504…` |

Statistics are fit once under the initial node dictionary and the fixed `P`, then
frozen: no training-time, checkpoint-time or validation-time refit.

## 4. Reconstruction, code usage and endpoint interventions

| quantity | train (soup) | valid (soup) |
|---|---|---|
| node relative reconstruction `R_N` | 0.078679 | 0.071091 |
| relation relative reconstruction `R_R` | 0.267536 | 0.266298 |

Code usage (official valid):
node code norm mean 1.8464, variance
0.0244, active coordinate
rate 0.9891;
relation code norm mean 1.0630,
`l0` mean 8.00,
top-1 share 0.4482,
effective atoms 16.92 /
64.

Endpoint interventions on the **soup** state (official valid 1000; pre-registered
`RMS > 1e-04` = clear sensitivity,
which is not by itself a performance claim):

| intervention | valid MAE | ΔMAE | pred RMS | ΔRMS | mean abs Δpred | sensitive |
|---|---|---|---|---|---|---|
| `permute_endpoints_11` | 0.388458 | +0.206115 | 1.7536 | -0.1183 | 0.3353 | yes |
| `permute_endpoints_22` | 0.382102 | +0.199758 | 1.7635 | -0.1085 | 0.3213 | yes |
| `zero_beta` | 0.655395 | +0.473052 | 1.5966 | -0.2754 | 0.6119 | yes |
| `zero_delta_rel` | 0.412621 | +0.230278 | 1.7539 | -0.1181 | 0.3539 | yes |
| `zero_node_pool` | 0.827275 | +0.644932 | 1.2566 | -0.6154 | 0.7882 | yes |

Mechanism flags: node channel load-bearing
**True**, delta channel
**True**, relation code channel
**True**.

## 5. Activity probe (fixed train probe, no parameter update)

| epoch | MAE-only grad Dbar_N | MAE-only grad Dbar_R | z norm | beta norm | beta l0 | R_N | R_R | ΔMAE zero_beta | ΔMAE zero_node_pool | ΔMAE zero_delta |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.693e-01 | 3.716e-02 | 1.8437 | 1.3375 | 8.00 | 0.3411 | 0.4021 | +0.02034 | -0.01761 | +0.00889 |
| 5 | 2.309e+00 | 4.601e-01 | 1.8669 | 1.3895 | 8.00 | 0.4222 | 0.5749 | +0.18035 | +0.74207 | +0.18249 |
| 20 | 6.238e-01 | 1.565e-01 | 1.8490 | 1.1859 | 8.00 | 0.4160 | 0.3266 | +0.25524 | +0.83052 | +0.18525 |
| 80 | 4.484e+00 | 6.458e-01 | 1.8438 | 1.1241 | 8.00 | 0.3323 | 0.2308 | +0.31777 | +0.73326 | +0.18237 |
| 160 | 1.041e+01 | 1.343e+00 | 1.8569 | 1.0972 | 8.00 | 0.2031 | 0.1930 | +0.35751 | +0.66650 | +0.16014 |
| 240 | 7.298e+00 | 8.975e-01 | 1.8561 | 1.0807 | 8.00 | 0.1695 | 0.1812 | +0.41625 | +0.66450 | +0.21982 |
| 320 | 3.121e+00 | 4.038e-01 | 1.8659 | 1.0938 | 8.00 | 0.1417 | 0.1818 | +0.49601 | +0.72847 | +0.25465 |

## 6. Provenance and reproduction

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_hier_relation_v1.py
uv run research run zinc_e2e_dictenv_hier_relation_v1 \
  --study zinc-context-gap --mode scratch \
  --purpose "Hierarchical static relation dictionary train-only preparation, CPU" \
  --set runtime.device=cpu --set model.stage=prepare
uv run research run zinc_e2e_dictenv_hier_relation_v1 \
  --study zinc-context-gap --mode screen \
  --purpose "Single-model hierarchical relation dictionary absolute-performance screen, CPU" \
  --set runtime.device=cpu --set model.stage=screen
```
