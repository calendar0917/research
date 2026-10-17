# P3-Slim v1 — reader-width capacity test

**Protocol:** `cscl-p3-slim-v1`; **split:** U (salt `cscl-p3-slim-v1`); **decision:**
`SLIM_NO_USEFUL_GAIN` (Monitor stage; Locked unread).

## Question

Holding the P3 atom/bond/junction encoders, Null composer, Fit-only SSL, data, raw-y L1,
Adam/cosine recipe, minibatch order and 22,680 updates fixed, does shrinking the graph-level
reader from `321→256→256→1` to `321→128→128→1` stably improve held-out MAE?

## Design

- P3: 413,505 params; reader 148,481.
- P3-Slim: 322,881 params; reader 57,857; −90,624 params (Reader −61.0%, total −21.9%).
- Same seed: canonical P3 built first; Slim reader from independent fixed seed `seed+10000`;
  then all four common modules and buffers copied bit-for-bit from canonical P3.
- Same Fit-only SSL encoder, same 22,680-update batch order; common init hashes, encoder hashes,
  batch-order hashes and update counts asserted equal.

## Result (Monitor, n=1,000)

| | P3 | P3Slim | Δ |
|---|---:|---:|---:|
| seed-mean MAE | 0.088836 | 0.084633 | −0.004203 |
| per-seed MAE | 0.084178 / 0.093493 | 0.085367 / 0.083900 | +0.001188 / −0.009594 |
| paired CI | — | — | [−0.014906, +0.002712] |
| RMSE | 0.374088 | 0.229635 | −0.144453 |
| ordinary MAE | 0.070833 | 0.069989 | −0.000844 |
| tail MAE (n=74) | 0.314111 | 0.267880 | −0.046231 |
| top-10-removed Δ | — | — | +0.000771 |
| seed spread | 0.009315 | 0.001467 | −0.007848 |
| Fit MAE mean | 0.001174 | 0.001311 | +0.000137 |

Gate: M1 pass, M2 fail, M3 fail, M4 pass, M5 fail, M6 pass, M7 pass → **4/7, fail**.
The mean gain is not seed-stable and is concentrated in tail/extreme errors; ordinary molecules
improve only 0.000844, and top-10-removed is slightly worse.

## Cost

Training wall 1,189.3 s (P3) vs 1,185.4 s (Slim), essentially 1.00×; Reader is a small part of
total compute. Total GPU ledger 1.6045 / 2.0 h including smoke, SSL, training, prediction and
failed-retry accounting. Prediction jobs for both arms and both zones took about 9–10 s/seed.

## Interpretation boundary

This is a negative result for this one fixed `256→128` Reader intervention. It does not prove
that Reader capacity is irrelevant, nor that Reader overfitting is the causal bottleneck. The
Fit–Monitor gap narrowed but that is not success. No Locked/full10k/test was purchased; no
width/seed/optimizer rescue. See `results/p3_slim_v1/REPORT.md` and `DEVIATIONS.md`.
