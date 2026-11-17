# lcrc-relation-role-preserving-pilot-v1

Does the **frozen C1 model** tolerate its 200-d S2 record being split into two
role groups (same topology block vs. crossing a topology-block boundary) and
rescaled by two trainable per-channel gammas — and does a *real* structural role
beat an identical-capacity, label-free type-hash split of exactly the same
records?

The previous round (`lcrc_relation_interface_routing_pilot_v1`) replaced the C1
readout with block/interface tokens and failed its retention gate. This round
therefore changes **nothing** about the C1 readout: `sum_v f(h_v) + rho([g,
mean h, mean h², log1p n])` is used verbatim.

## Design in one paragraph

Atom encoder, bond encoder, S1, typed-S2, LocalMLP, LayerNorm, dropout, `f`,
`rho`, the 433-d local-input layout and the readout are the frozen C1 modules.
The only change is inside the S2 column block: the per-atom sum of the 200-d S2
records is computed twice — once over the *internal* records, once over the
*cross-boundary* records — and recombined as
`Q = gamma_internal ⊙ Q_internal + gamma_cross ⊙ Q_cross` with two 200-d
vectors initialised exactly to 1. With `gamma = 1` the model is bitwise the C1
model. `RR_CONTROL` uses the identical architecture, parameter count, initial
state and recipe, but assigns each occurrence by a frozen
`blake2b(salt|path_cat|path_end63|path_closure)` hash (one class per typed-S2
record, no atom index, no bond id, no row id, no label, no oracle).

Parameter delta over C1: exactly 400 (2 × 200) for both models.

## Layout

```
code/            rp_* modules (common, data, models, train, checks, analysis, report, runner)
configs/         frozen.yaml (engineering knobs only)
protocols/       the comparability protocol (rules, thresholds, test policy)
studies/         the study declaration (question, candidates, open questions)
cache/           the frozen CONTROL role table (content-addressed, sha256 in the manifest)
analysis/        gates, group audit, budgets, predictions, bootstrap, gamma, oracle, decision
results/         per-unit trajectories, curves, checkpoints, predictions (checkpoints not committed)
tests/           fast invariant tests (no 360-epoch trajectory)
run_remote.sh    control-plane entry point (four units, two GPUs)
```

## Reproduce

```bash
# 1. freeze / re-verify the reuse and the CONTROL table (CPU, ~12 s)
bash run_remote.sh --purpose "prepare" --set stage=prepare
# 2. all correctness gates before any training (CPU, ~5 s)
bash run_remote.sh --purpose "pre-training gates" --set stage=checks
# 3. the four formal units (one seed, 360 epochs each)
bash run_remote.sh --purpose "RR_REAL lipo"     --set stage=train --set task=lipo    --set model=RR_REAL
bash run_remote.sh --purpose "RR_REAL wc_logp"  --set stage=train --set task=wc_logp --set model=RR_REAL
bash run_remote.sh --purpose "RR_CONTROL lipo"  --set stage=train --set task=lipo    --set model=RR_CONTROL
bash run_remote.sh --purpose "RR_CONTROL wc"    --set stage=train --set task=wc_logp --set model=RR_CONTROL
# 4. post-training gates, then analysis + reports
bash run_remote.sh --purpose "post-training gates" --set stage=checks
bash run_remote.sh --purpose "aggregate" --set stage=aggregate
```

Local tests: `uv run pytest tracks/structural-sufficiency/lcrc_relation_role_preserving_pilot_v1/tests -q`
(~18 s, no training).

## Read-only reuse (verified by SHA-256 *and* by rebuild)

| source round | what is reused |
|---|---|
| `lcrc_readout_context_pilot_small_v1` | the C1 atom/bond/S1/S2/LocalMLP modules, the C1 readout, the canonical initial state, the frozen scalers |
| `lcrc_relation_interface_routing_pilot_v1` | the frozen topology partition, the S2 physical-key routing cache (`path_cross`, `path_owner1/2`, block membership, block bond lists) and the S2 type codes |
| `lcrc_cross_task_preparation_v1`, `lcrc_lipo_model_dissection_v1`, `structural-query-learning` | the frozen graph/LCRC feature builders and the C1 local operator |
| `gnn_route_clarification_8h_v1` | the read-only C1(Rf) and GINE baseline rows of `analysis/tables/stage1.csv` |

Nothing from any of these rounds is retrained, re-selected or re-evaluated here.

## Result (seed 0, official split, four units, DEVELOPMENT)

| unit | best-Valid epoch | Valid RMSE | Test RMSE (DEVELOPMENT_TEST) |
|---|---|---|---|
| C1(Rf), read-only baseline | 124 | 0.7069 | 0.7293 |
| RR_REAL · Lipophilicity | 148 | 0.7071 (1.0004× C1) | 0.7610 |
| RR_CONTROL · Lipophilicity | 115 | 0.6945 (0.9825× C1) | 0.7069 |
| C1(R3), read-only baseline (recipe differs) | 237 | 0.0626 | 0.0781 |
| RR_REAL · WC_LOGP | 340 | 0.1179 | 0.0966 |
| RR_CONTROL · WC_LOGP | 220 | 0.1092 | 0.1049 |

* Both models **recover** the frozen C1 prediction: +400 parameters and two
  per-channel gates cost nothing measurable (REAL is 0.0003 RMSE above C1).
* The real topology role does **not** beat the label-free type hash:
  Lipophilicity paired bootstrap `RMSE(REAL) − RMSE(CONTROL)` is
  +0.0126 [-0.0328, +0.0572] on Valid and +0.0541 [+0.0104, +0.1036] on Test,
  i.e. if anything the CONTROL split is better on Test (P(REAL better) = 0.005).
* WC_LOGP oracle enrichment is ~0 for both models (mean +0.002…+0.003, ~48–53 %
  matched molecules positive, CI spanning 0) and the paired REAL−CONTROL
  difference is −0.0006 [-0.0023, +0.0010] (Valid) / +0.0001 [−0.0011, +0.0013]
  (Test): no localisation advantage.
* The learned gammas do move (|γ−1| ≈ 0.07–0.09 per channel) but do not buy
  anything: re-running the frozen models with both gammas at 1 gives
  0.7015 (REAL) / 0.6940 (CONTROL) Valid RMSE, i.e. equal or slightly better.
* The two models' native block-mask responses are nearly the same map
  (per-molecule Spearman 0.90, same arg-max block 89 %).

**Verdict: `ONLY_C1_READOUT_RECOVERY`** — keeping the C1 readout restores the C1
prediction for both groupings, and the real structural role provides no
pre-registered-strength prediction or explanation increment over an arbitrary
label-free split of the same 200-d records. Per the frozen protocol the round
stops here: no extra seed, no search, no new architecture.

See `DECISION.md` (verdict + flags) and `REPORT.md` (every table generated from
`analysis/*.json|csv`, with the gate list, the group sizes, the gamma ablations,
the oracle controls and the mask-response comparison).
