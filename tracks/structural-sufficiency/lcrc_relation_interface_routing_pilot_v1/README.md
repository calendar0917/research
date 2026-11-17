# lcrc relation–interface routing pilot (v1)

Feasibility pilot: can LCRC S1/S2 occurrences be **routed** to frozen topology
blocks and their interfaces without losing predictive power, and does an
interface need context from both endpoint blocks?

Status: **running**. Numbers appear in `REPORT.md` / `DECISION.md` once the four
formal units finish; nothing is typed by hand — `REPORT.md` is generated from
`analysis/*.json|csv` by `code/rr_report.py`.

## Layout

| path | what |
| --- | --- |
| `PROTOCOL.md` | the frozen rules of this round (partition, routing, models, recipe, decision scales) |
| `configs/frozen.yaml` | engineering knobs only |
| `protocols/`, `studies/` | comparability protocol + study declaration |
| `code/rr_common.py` | identity, paths, frozen constants, IO |
| `code/rr_partition.py` | frozen topology partition (bridges / cycle blocks / bridge chains / isolated atoms) |
| `code/rr_routing.py` | S1 + S2 routing to blocks and interfaces, conservation |
| `code/rr_models.py` | R1 / R2 (shared C1 trunk, block tokens, interface tokens, rho_g) |
| `code/rr_data.py` | frozen caches + routing + targets + Train-only scalers |
| `code/rr_train.py` | Rf recipe, every-epoch Train/Valid/Test, resume, ledgers |
| `code/rr_prepare.py` | CPU: routing caches, support gate, parameter budget, protocol freeze |
| `code/rr_checks.py` | the twelve correctness gates |
| `code/rr_analysis.py` | predictions, bootstrap, block gating, oracle localisation, decision |
| `code/rr_report.py` | `REPORT.md` / `DECISION.md` from the artefacts |
| `cache/` | the single frozen topology-routing cache (CPU, SHA-256 pinned) |
| `analysis/` | every number reported by `REPORT.md` |
| `results/` | the four training units (checkpoints and predictions are pulled, not committed) |
| `tests/` | fast invariants |

## Reproduce

```bash
bash tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/run_remote.sh --set stage=prepare
bash tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/run_remote.sh --set stage=checks
# res-gpu0 (R1) and res-gpu1 (R2) in parallel, one job per training unit
bash .../run_remote.sh --set stage=train --set task=lipo    --set model=R1
bash .../run_remote.sh --set stage=train --set task=wc_logp --set model=R1
bash .../run_remote.sh --set stage=train --set task=lipo    --set model=R2
bash .../run_remote.sh --set stage=train --set task=wc_logp --set model=R2
bash .../run_remote.sh --set stage=aggregate
```

Only the four `train` units need a GPU; `prepare`, `checks` and `aggregate` are
CPU-only and read the committed caches.
