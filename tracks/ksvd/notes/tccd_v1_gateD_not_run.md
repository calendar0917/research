# TCCD-v1 Gate D — Not Run

Gate D was **not run**. The preregistered Gate C dictionary-uniqueness test failed decisively:

* TASK-D best internal-dev MAE: `0.938099`.
* DENSE best internal-dev MAE: `0.403933`.
* TASK-D worse than DENSE by `0.534167`, versus the allowed margin `0.005`.

Per the frozen decision tree, full-data TASK-D training and absolute comparison
against the canonical GPU1 baseline were not authorized. Official test remained
unopened. The canonical GPU1 baseline `0.119818` is therefore not compared to
TCCD-v1 in this round.
