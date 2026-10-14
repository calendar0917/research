# cscl-task-adaptive-structure-units-v1 runbook

Single frozen pass, four arms, two seeds:

```
python tracks/structure-contribution/code/run_task_adaptive_structure_units_v1.py verify-inputs
python ... audit-blocks
python ... cpu-gates          # S1 + S2 + forward/invariance/gradient gates
python ... freeze
# remote (res, GPU):
python ... train --arm F-A --seed 0
python ... train --arm F-A --seed 1
... 8 runs total ...
python ... predict --arm F-A --seed 0   # monitor only, no labels
... 8 predictions ...
python ... freeze-predictions
python ... score-monitor      # single monitor label read
python ... audit-partition    # label blind
python ... audit-contribution
python ... audit-sharing
python ... finalize
```

`audit-*` and `finalize` must run before any interpretation of monitor scores.
