# RUN LEDGER — zinc-statistical-program-search-v1

Clock: round start 2026-10-25 23:55 CST. Hard wall 2026-10-26 07:55 CST.
GPU budget: 16.0 A100·h hard / 14.0 soft (sum over both cards, incl. timing,
failures, evaluation). No official test at any point.

| # | time (CST) | item | where | status | GPU·h |
|---|---|---|---|---|---:|
| 0 | 23:55-00:20 | local context read, CPU checks, prep, CPU smoke | local | done | 0.00 |
| 1 | 00:21 | remote prep (Q record view, Fit-only constants) | res CPU | done (8.5 s) | 0.00 |
| 2 | 00:22 | 2-epoch timing: QF 2.08/2.49 s per epoch, FC 2.87/2.32 s | res gpu0/gpu1 | done; projections QF 0.38, FC 0.43 GPU·h | 0.01 |
| 3 | 00:26-00:35 | QF/FC seed-0 runs **cancelled at ~epoch 130** | res gpu0/gpu1 | **implementation error**: the reused trainer passes its own train/valid cache to one collate closure; my closure had bound the train split, so valid labels were scored against train features. Runs invalid, artifacts deleted, no usable result. | 0.31 |
| 4 | 00:36 | fix + new CPU check (one closure must be correct for both splits; cache-vs-reference exactly 0.0) | local | done | 0.00 |
| 5 | 00:36 | QF seed 0 (600 ep) on gpu0, FC seed 0 (600 ep) on gpu1 relaunched | res | running | — |
| 6 | 01:02 | QF seed 0 complete: best 0.110569@308, W20 0.112376, T100 0.112406, 248,361 params, 0.427 GPU·h | res gpu0 | completed, full 600 epochs | 0.427 |
| 7 | 01:00 | FC seed 0 complete: best 0.101948@337, W20 0.103530, T100 0.103601, 205,506 params, 0.393 GPU·h | res gpu1 | completed, full 600 epochs | 0.393 |
| 8 | 01:05-01:15 | local analysis: summary/W20/T100, paired per-molecule vs frozen DCC-L s0, frozen-weight mechanism probes (forward only, CPU) | local | done, 0 A100 | 0.00 |
| — | — | **total A100·h** | | of 16 hard / 14 soft; 14.86 unused; official test never read | **1.14** |
| 9 | 01:15-01:35 | REPORT.md + NEXT_DECISION.md + artifacts manifest, STATE.yaml pointer, commit `deef180` | local | done | 0.00 |
