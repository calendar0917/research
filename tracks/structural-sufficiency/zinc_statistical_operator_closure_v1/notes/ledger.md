# GPU ledger — SOC-v1

Every job is listed here, including smoke, failed and retried jobs. A100·h is
card-hours from the run's recorded `gpu_hours` (or wall × cards for multi-card).

| # | date | host/pool | job | arm | seed | mode | status | A100·h | notes |
|---|---|---|---|---|---:|---|---|---:|---|
| 0 | 2026-10-03 | local CPU | CPU gates | R/T/CTRL | — | gate | planned | 0.000 | no GPU |
| 1 | 2026-10-03 | res-gpu0 | smoke 64 updates | R | 0 | screen | planned | ≤0.05 | code-path check |
| 2 | 2026-10-03 | res-gpu0/1 | screen R | R | 0,1 | fit8k | planned | ≤1.2 | 2 cards |
| 3 | 2026-10-03 | res-gpu0/1 | screen T | T | 0,1 | fit8k | planned | ≤1.2 | 2 cards |
| 4 | conditional | res-gpu0/1 | control | CTRL | 0,1 | fit8k | conditional | ≤1.2 | only if a SOC arm passes |
| 5 | conditional | res-gpu0/1 | full 10k | winner+CTRL | 0,1 | 10k | conditional | ≤2.4 | only if screen passes |
| 6 | conditional | res-gpu0 | test score | winner | 0,1 | terminal | conditional | ≤0.05 | once, after full gate |

Hard cap: 16.0 A100·h. Soft plan: ≤6.0. Stop at screen failure.

## Actual entries

*(filled as jobs are launched and completed)*

## 2026-10-28 (UTC+8) — executed

| # | time | pool | experiment | arm | seed | mode | status | A100·h | run_id |
|---|---|---|---|---|---:|---|---|---:|---|
| 1 | 01:14 | res-gpu0 | soc-smoke-r0 (64 upd) | R | 0 | screen | completed | 0.0011 | `soc-smoke-r0-20261028-011415-9314a812` |
| 2 | 01:16 | res-gpu0 | soc-screen-R-s0 (47,400 upd) | R | 0 | fit8k | running | — | `soc-screen-R-s0-20261028-011613-0fe46447` |
| 3 | 01:16 | res-gpu1 | soc-screen-R-s1 (47,400 upd) | R | 1 | fit8k | running | — | `soc-screen-R-s1-20261028-011625-7e2d5210` |

Infrastructure note: first `rr deploy res` failed because the remote checkout had
an **untracked** copy of
`zinc_counterfactual_ownership_composer_v1/analysis/profile_s0.json` that the new
commit adds. The remote copy was sha256-identical to the tracked local file
(`3b4719c4…`), so the duplicate was removed and the deploy succeeded
(`331fab569441 → 9e7fbae097ad`). No result file was lost.

### R screen result (completed)

| seed | SOC R Monitor best | frozen L Monitor | Δ (SOC−L) | SOC valid best | L valid |
|---|---:|---:|---:|---:|---:|
| 0 | 0.124011 (ep 571) | 0.123151 | +0.000861 | 0.090063 | 0.089873 |
| 1 | 0.126506 (ep 580) | 0.127772 | −0.001266 | 0.097404 | 0.095470 |
| mean | 0.125259 | 0.125461 | **−0.000203** | 0.093734 | 0.092672 |

Paired bootstrap (1,000 Monitor rows, 10,000 reps): Δ seed-mean
−0.000203, 95% CI [−0.002516, +0.002076], P(Δ<0) 0.584, win fraction 0.501.
Gate: seed mean 0.125259 > 0.1215 → **SCREEN_FAIL**. R A100·h 0.678 + 0.772 =
1.450. Final closure gates: s0 [0.00565, 0.00715, 0.00611], s1
[0.00517, 0.00682, 0.00485] — essentially unchanged from the 0.006693
initialisation, i.e. the optimizer suppressed the operator.

| 4 | 02:12 | res-gpu0 | soc-screen-T-s0 | T | 0 | fit8k | launched | — | pending |
| 5 | 02:12 | res-gpu1 | soc-screen-T-s1 | T | 1 | fit8k | launched | — | pending |

### T screen result (completed)

| seed | SOC T Monitor best | frozen L Monitor | Δ (SOC−L) | SOC valid | L valid |
|---|---:|---:|---:|---:|---:|
| 0 | 0.126867 (ep 599) | 0.123151 | +0.003716 | 0.092100 | 0.089873 |
| 1 | 0.128599 (ep 583) | 0.127772 | +0.000827 | 0.090502 | 0.095470 |
| mean | 0.127733 | 0.125461 | **+0.002271** | 0.091301 | 0.092672 |

Paired bootstrap seed-mean: +0.002271, 95% CI [−0.000270, +0.005042].
Gate **FAIL**. T A100·h 0.680 + 0.792 = 1.472.

### Mechanism diagnostics (CPU, no GPU)

* Final gates: R s0 [0.00565, 0.00715, 0.00611], R s1 [0.00517, 0.00682,
  0.00485]; T s0 [0.00577, 0.00762, 0.00663], T s1 [0.00495, 0.00709, 0.00532].
* Closure off / trained gate / best forced gate (Monitor):
  R s0 0.125746 / 0.124062 / 0.123695 (g=0.02);
  R s1 0.126693 / 0.126521 / 0.126521;
  T s0 0.128021 / 0.126895 / 0.125831 (g=0.02);
  T s1 0.128612 / 0.128603 / 0.128603.
* R s0 train-2k vs forced gate: 0.010264 (g=1e-4), 0.008469 (g=0.006693,
  minimum), 0.015423 (g=0.02), 0.055502 (g=0.1). The gate is at the
  train-optimal amplitude, not a cold-start victim.
* Total round: 2.9237 A100·h. Official test never loaded.

### Closed jobs

* `soc-screen-T-s0-20261028-020617-78f7cbab` completed, 0.6804 A100·h.
* `soc-screen-T-s1-20261028-020630-586e2d96` completed, 0.7918 A100·h.
* No CTRL, no full-10k, no test scoring: the screen gate failed.
