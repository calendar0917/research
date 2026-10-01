# Decision — E2E-DictEnv-Hier-Relation-v1

- candidate: `HIER-RELATION` (one candidate, seed 0, 320 epochs, local CPU)
- soup valid MAE: **0.182343094** (best 0.193209686 @ 316)
- band: `stop` — stop the current configuration
- official train soup MAE 0.091181269 (gap -0.091162)
- official test: never loaded

Mechanism (separate statement, not a band change): relation dictionary carries prediction
(node channel load-bearing = True, delta channel
= True, relation code channel =
True).

## Not claimed by this round

- no matched-training control (no dense/PCA arm, no random/shuffled-dictionary arm);
- no sparse-vs-dense or dictionary-vs-no-dictionary causal statement;
- no statistical-significance claim (single seed);
- no terminal (official-test) statement;
- no statement about any previous round's architecture (A/B/C, Sem108, Joint709 are
  background only, and this round's numbers are not a matched comparison to them).

## Forbidden without a new pre-registration

- seed 1, any width / K / s / epoch / LR / lambda change;
- touching the official ZINC test split;
- re-using these numbers as a matched comparison against the older runs.
