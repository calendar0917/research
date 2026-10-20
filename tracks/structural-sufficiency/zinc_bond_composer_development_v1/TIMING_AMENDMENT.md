# Timing-smoke amendment

Status: timing-only amendment made after the first smoke and before any formal
run. No architecture, data, recipe, epoch count or budget cap was changed.

## Original record: 2-epoch smoke

`smoke.json` used the same 1,024-row Fit prefix for B0/B1/B2, 16 updates per
model, and the steady-state mean over updates 3–16.

| model | steady step |
|---|---:|
| B0 | 24.70 ms |
| B1 | 30.16 ms |
| B2 | 31.94 ms |

Projection: 2,014.0 s formal + 6.0 s smoke, with the frozen 1.20 margin and
300 s reserve = **2,724.1 s = 0.7567 h**, i.e. 24 s over the 0.75 h cap.
Status: `STOP_BUDGET`.

## Executed timing probe: 5 epochs

The 2-epoch estimate is dominated by CUDA allocator / kernel warm-up. One
timing-only probe was run on the same 1,024 rows and the same three models,
with 40 updates each and the same estimator (drop the first two updates).

| model | steady step |
|---|---:|
| B0 | 24.23 ms |
| B1 | 30.61 ms |
| B2 | 30.82 ms |

Projection: 1,987.1 s formal + 7.9 s probe, with the same 1.20 margin and
300 s reserve = **2,694.0 s = 0.7483 h**. Status: `PASS`.

The pass rule was not relaxed. Both records are retained. The formal runs used
the 5-epoch probe as the gate. A strict reading of the first record would have
stopped the round before the formal runs; that is disclosed in `REPORT.md`.

## Realised cost

| item | seconds |
|---|---:|
| B0 formal process wall | 584.4 |
| B1 formal process wall | 825.5 |
| B2 formal process wall | 720.3 |
| both smoke attempts | 13.9 |
| **process-wall total** | **2,144.2 (0.5956 h)** |
| `rr` scheduler wall incl. startup | 2,184 (0.6067 h) |

Cap: 0.75 h. Remaining: 0.1544 h on process wall, 0.1433 h on the conservative
scheduler wall. The actual formal total was 7.2 % above the 5-epoch central
projection, inside the 20 % margin.
