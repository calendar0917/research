# compact_v4_relation_refresh — P2 one-shot relation refresh

Minimal falsification of relation-state staleness in optimized compact-v4.

**Verdict: Decision Case B — ONE-SHOT RELATION REFRESH NO-GO** (sub-threshold
positive). P2 seed0 valid **0.144457** vs reused optimized-v4 seed0 **0.146420**
→ `Δ_P2,0 = +0.001963 < +0.004`. Exactly **0** added parameters. Seed1 **not**
purchased. Official test **never loaded**.

| file | contents |
|---|---|
| `baseline_inventory.json` | reused optimized-v4 checkpoint fingerprint / config / protocol |
| `staleness_probe_manifest.json` | reused 2000-molecule official-train train-probe (target-independent hash) |
| `staleness_audit_lock.json` | frozen Stage A definitions, metrics, stop gate |
| `staleness_integrity.json` | Stage A gates A0.1–A0.10 (all pass) |
| `relation_drift.json` | D1 (normalized L2) + D2 (cosine), incl. robust degenerate-row handling |
| `pair_summary_drift.json` | graph pair-summary drift (per graph and per bucket) |
| `prediction_shift.json` | mechanical `|y_refresh − y0|` (true `y` unused) |
| `stageA_decision.json` | near-identity gate → ADVANCE |
| `architecture_lock.json` | frozen P2 architecture (shared tensors, refresh = 1, R = 302D) |
| `parameter_audit.json` | 99,613 vs 99,613, Δ = 0 |
| `initialization_match.json` | all 99,613 initial params identical to baseline |
| `compute_audit.json` | +61.3% forward wall-clock at 0 added params |
| `stageB_p2_seed0.json` | seed0 run (best valid 0.144457, epoch 156, 196 epochs) |
| `stageB_decision.json` | architecture gate decision (Case B) |
| `common_input_bulk.json` | target-independent bulk safety (safe, −0.00099) |
| `mechanism_diagnostics.json` | refresh dependency (on 0.1445 / off 0.1751) |
| `final_decision.json` | Case B |
| `answers_q1_q20.json` | Q1–Q20 |
| `curves/p2_seed0_curve.csv` | per-epoch train/valid MAE, q0/q1 norms, cosine, drifts, grads |
| `states/p2_seed0_selection_state.pt` | selected P2 seed0 checkpoint (99,613 params) |
| `figures/` | architecture, staleness audit, valid MAE |

See `tracks/ksvd/notes/compact_v4_relation_refresh.md` (23 sections + Q1–Q20) and
`tracks/ksvd/tests/test_compact_v4_relation_refresh.py` (24 pass).
