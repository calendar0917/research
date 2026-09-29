# DECISION — `e2e_dictenv_training_protocol_audit_v1`

Date: 2026-09-29 (CPU only; official ZINC test never loaded)
Preregistration: `notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md`
Evidence: `comparison.json`, `REPORT.md`, `analysis_tables.md`,
`shared_prefix/prefix_summary.json`, `fork/fork_integrity.json`,
`control_lr1e3/result.json`, `low_lr1e4/result.json`.

## Decision

```text
TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK

G_schedule = +0.000610 (+1.0x the ~0.0006 CPU soup floor)
M_control(lr=1e-3 tail) = 0.127428
M_low_lr(lr=1e-4 tail)  = 0.126818
canonical_protocol      = 320@1e-3
horizon_may_still_be_binding = False
```

## Because

- the shared prefix and the fork are exact: epoch-280 valid MAE `0.139555` (`PREFIX_HEALTHY`), prediction max |Δ| `0.0`, only the learning rate differs after the fork, and all 40 tail batch-order hashes match;
- the primary comparison is therefore a matched, single-variable LR experiment;
- the dictionary stays healthy on both arms (CONTROL N_eff 20.28, LOW-LR N_eff 20.28, epoch-280 N_eff 20.76);
- no official test was read; the decision is based on official valid only.

## Next

Close the training-protocol-as-primary-bottleneck hypothesis. The next round is a
new architecture study — a Dictionary-Conditioned Semantic Operator
(`e_v = sum_k alpha_vk f_k(q_v)`, shared dictionary roles driving a transferable
structure→semantics map) — still with **no message passing and no environment
update**; no further optimizer/scheduler search.

