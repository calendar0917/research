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

## Verdict

See `DECISION.md` (verdict + flags) and `REPORT.md` (every table regenerated
from `analysis/*.json|csv`).
