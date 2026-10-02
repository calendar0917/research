# zinc_full_bottleneck_audit_v1 — REPORT

Read-only mechanism + error-budget audit of the frozen Full soup, plus one
frozen conditional head probe.  Official test never instantiated.
Parent: `SCALE-FULL-seed0_soup_state.pt` sha256 `17f5fcc3…574eb`.

## What was run

| artifact | script | content |
|---|---|---|
| `audit_full.json`, `valid_per_graph.csv` | `audit_full.py` | soup replay (raw/calibrated train/valid), id172, branch scales, dictionaries, R/H2 spectra |
| `phase_c_stats.json` | `stats_phase_c.py` | error budget by train-fit target/size, coverage, id%5 groups, top-15 |
| `probe_features.npz`, `probe_features_meta.json` | `extract_probe_features.py` | frozen H2 / pre-fusion-interface moments / node-joint moments |
| `phase_d_probe.json` | `probe_phase_d.py` | `[1,H2]` baseline + P1/P2 + same-width controls (MAE+L2 ADMM) |
| run `20261002-175703-469df2f8` | control plane | probe stage metrics |

## Headline results

* Replay exact: raw valid MAE `0.1191540920053958`; calibrated `0.11506585458567133`
  (bias `-0.03428781`, `r = y - pred`, train-median); eval-mode train MAE `0.0453967`.
* **Node branch dead**: `W_A_S`, `W_A_C`, `node_encoder.0.weight` are float32
  denormals; `node_slot` zero-fraction 1.0; `node_out` constant across rows.
* **Task dictionary dense**: mean nonzero `250.89/288` (87.1%); reconstruction
  relative error `0.0588`.
* **Error budget (calibrated valid)**: id172 contributes `0.018918` (16.4% of the
  absolute-error sum); MAE excluding id172 `0.096244`; top1/5/10% shares
  25.3/39.8/50.6%; central-90% MAE `0.063195`. Train-fit lowest decile valid MAE
  `0.3812` (30.8% of the sum) vs train `0.0576`.
* **Conditional probes**: `[1,H2]` baseline `0.115024`; P1 pre-fusion-interface
  moments `-0.000272`; P2 node-joint moments `-0.000131`;
  same-width controls `+3.6e-7 / +4.5e-7`. Neither passes the `0.003` gate.

## Verdict

No purchasable feature block at the two tested compression locations.  The
best-supported residual-error category is **generalisation / tail coverage**;
the node-binding collapse is a confirmed mechanism deviation with no
demonstrated performance upside (probe negative, likely redundant with Sem108).
No new training is bought.  Next cheapest discriminating evidence: a frozen-R
tail recoverability check on a train-internal fold.

See `tracks/ksvd/notes/zinc_full_bottleneck_audit_v1_analysis.md` for the full
audit and the five direct answers.
