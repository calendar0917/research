# Experiment ledger — zinc_performance_search_v1

Every launched GPU job appears here, including failures. Costs are the job's
own `wall_seconds` converted to GPU-hours (one A100 per job), or the run
interval for jobs that died before writing `train.json` (marked *est*).

Round clock: start **2026-10-23 16:06 UTC**, hard deadline **2026-10-24 00:06 UTC**.
Host `res`, 2 x A100-40GB (pools `res-gpu0`, `res-gpu1`), torch 2.5.1+cu124,
PyG 2.6.1. All runs: official PyG `ZINC-subset`, 10,000 / 1,000 / 1,000,
original split, raw-y MAE, single model, <=500k parameters.

## 1. Training runs

| # | run id | body | recipe | seed | epochs / updates | params | best-valid (epoch) | final valid | wall (s) | GPU-h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `zps-A_R1_ema_s0` (attempt 1, `ed4b279`) | A P3Null | R1 + EMA | 0 | 600 / 47,400 | 413,505 | — | — | ~1560 (est) | 0.44 | **FAILED** at final save (EMA state missing `global_mean`/`global_scale`); 600 epochs burned |
| 2 | `zps-B_R1_ema_s0` (attempt 1, `ed4b279`) | B DCC-L | R1 + EMA | 0 | 600 / 47,400 | 494,193 | — | — | ~1560 (est) | 0.44 | **FAILED** same bug |
| 3 | `zps-A_R1_ema_s0` (`5ffc2ac`) | A P3Null | R1 + EMA | 0 | 600 / 47,400 | 413,505 | **0.0875697** @433 | 0.0886532 | 1128 | 0.313 | complete; **EMA branch invalid** (constant 1.60004) -> online only |
| 4 | `zps-B_R1_ema_s0` (`5ffc2ac`) | B DCC-L | R1 + EMA | 0 | 600 / 47,400 | 494,193 | **0.0792465** @437 | 0.0806209 | 1498 | 0.416 | complete; EMA branch invalid -> online only |
| 5 | `zps-A_R0_s1` | A P3Null | R0 | 1 | 600 / 47,400 | 413,505 | 0.0822762 @470 | 0.0830813 | 1296 | 0.360 | complete |
| 6 | `zps-B_R0_s1` | B DCC-L | R0 | 1 | 600 / 47,400 | 494,193 | 0.0738369 @424 | 0.0751832 | 1447 | 0.402 | complete |
| 7 | `zps-dbtc-s0` | C-sub DBT | native, 600 ep | 0 | 600 / 94,200 | 452,523 | 0.0772600 @449 | 0.0791624 | 5313 | 1.476 | complete |
| 8 | `zps-A_R0L_s0` | A P3Null | R0L (1200 ep) | 0 | 1200 / 94,800 | 413,505 | 0.0766988 @557 | 0.0823147 | 2579 | 0.717 | complete |
| 9 | `zps-dbtc-s1` (attempt 1) | C-sub DBT | native, 600 ep | 1 | 600 / 94,200 | 452,523 | — | — | ~330 | 0.09 | **CANCELLED** after 5 min: wrapper wrote `search.json`/`train.json` into the seed-0 run directory |
| 10 | `zps-dbtc-s1` (relaunch) | C-sub DBT | native, 600 ep | 1 | 600 / 94,200 | 452,523 | see §5 | | | | running |

Reused, verified, not re-run (identical code path, same data caches, same SSL
encoder sha256 `006fabc1...`; verified in this round by re-reading their
`train.json` + `predictions/valid_best.npz`):

| run | body | recipe | seed | params | best-valid (epoch) | final valid | wall (s) | GPU-h | source |
|---|---|---|---|---|---|---|---|---|---|
| `xBssl_P3N_s0` | A P3Null | R0 | 0 | 413,505 | **0.0762238** @377 | 0.0781788 | 1085 | 0.301 | `zinc_recipe_trunk_v2/runs/xBssl_P3N_s0` |
| `L_s0` | B DCC-L | R0 | 0 | 494,193 | **0.0785371** @489 | 0.0790764 | 1243 | 0.345 | `zinc_dual_channel_composer_v1/runs/L_s0` |
| `dbt_s0` | DBT (reference) | native, 1500 ep | 0 | 452,523 | **0.0777684** @949 | 0.0813138 | 13,290 | 3.692 | `zinc_directed_bond_transformer_v1/runs/dbt_s0` |

## 2. Timing / engineering jobs (no model claimed)

| job | purpose | result | GPU-h |
|---|---|---|---|
| `zps-dense-timing` (failed, `d6bb814`) | dense GRIT harness smoke | harness bug, no numbers | ~0.02 |
| `zps-dense-timing` (`b067616`) | dense GRIT epoch cost, bs 32/64/128 | 34.8 / 20.95 / 14.84 s per epoch (contended) | ~0.05 |
| `zps-dense-timing2` (`e01785f`) | clean re-measure bs 32/64 | **30.97 / 18.65 s per epoch** -> 2000 ep = 17.2 / 10.4 h | ~0.05 |
| local CPU checks | dense-vs-sparse parity (forward 1.7e-6, grad 9.6e-5 rel), RRWP vs frozen cache (2.4e-7) | written to `analysis/dense_parity.json` | 0 |

## 3. Cost accounting

| item | GPU-h |
|---|---|
| failed R1 attempts (2) | 0.88 |
| A-R1 + B-R1 (seed 0) | 0.73 |
| timing / smoke | 0.12 |
| **subtotal at 17:30 UTC** | **1.73** |
| running: A-R0 s1, DBT-C0 | +? |
| hard cap | 16.0 |
| soft target | 15.0 |

## 4. Declared deviations

1. **EMA checkpoint policy dropped.** The trainer's EMA evaluation copied the
   freshly-built model *into* the EMA buffer instead of the reverse (argument
   order in the `zip` copy), so every EMA evaluation reported the untrained
   init (valid 1.60004 for 600 epochs). Fixed in `ad_train.py` after the fact
   (`ada...` commit) but **no EMA number from this round is used**; A-R1/B-R1
   are ranked on their online branch only and their `*_ema` artifacts are
   marked invalid.
2. **Arm C = GRIT is excluded on measured cost.** The audited dense port
   (473,473 params, 10 layers / 64 wide / 8 heads, RRWP 1..20) runs at
   96 ms/step (bs 32) and the vendored sparse runtime at ~9.5 s/epoch
   (8 layers, bs 128); 10 layers x 1500 epochs is >=4 h, over the 2 h/run
   ceiling. Fallback body used instead: native DBT at 600 epochs / 94,200
   updates (declared in `runs/dbtc_/dbt_s0/search.json`).
3. DBT horizon shortened from 1500 to 600 epochs; batch stays native 64.
4. **Provenance repair.** The first `zps-dbtc-s1` launch used a wrapper whose
   declaration path was hard-coded to the seed-0 directory; it overwrote
   `runs/dbtc_/dbt_s0/search.json` before being cancelled. That file was
   restored byte-for-byte from the copy pulled at 18:51 UTC (seed 0,
   `declared_at_utc` 2026-10-23T17:21:05Z) and the seed-0 `train.json` /
   `curve.csv` were verified unchanged (sha256 identical on host and local).
   The wrapper now writes `dbt_s{seed}`; the relaunch is seed 1.
