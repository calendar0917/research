# Decision — E2E-DictEnv-Joint709-Absolute-v1

- candidate: `JOINT-SPARSE` (one candidate, seed 0, 320 epochs)
- soup valid MAE: **0.132442119** (best 0.135177988 @ 320)
- band: `stop` — stop: the candidate did not enter a better absolute interval
- official train soup MAE 0.056583716 (gap -0.075858403)

The candidate did not enter a better absolute interval (> 0.1233): the frozen joint dictionary route is stopped for now. This is an absolute-performance screen only — it is not evidence about any control arm.

Endpoint caveat (recorded with the result): at the soup state the coordinate binding is
denormal-zero (all four binding matrices ≈ 7e-38), so the zero/shuffle probes are exactly
0.0 and this MAE reflects the frozen protocol with an inert dictionary coordinate, not the
joint dictionary route's ceiling. The band rule fires `stop` regardless. See the analysis
note §3.

## Not claimed by this round

- No increment over any control (no control arm was run: no PCA48, no random / shuffled / no-dictionary training).
- No sparse-dictionary-vs-PCA or joint-vs-separate causal statement.
- No statistical-significance claim; single seed.
- No terminal (official-test) statement.

## Forbidden without a new pre-registration

- seed 1, any K / s / K-SVD-epoch / scaler / width / horizon / lr change;
- unfreezing or end-to-end fine-tuning the dictionary;
- touching the official ZINC test split;
- re-using these numbers as a matched comparison against historical A/B/C runs.
