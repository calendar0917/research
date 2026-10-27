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
