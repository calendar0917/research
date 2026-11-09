# lcrc-pcba-dynamic-v1 — W2-DYNAMIC

Diagnostic boundary experiment on the LCRC "no iterative message passing"
constraint for OGB MolPCBA (official scaffold split).

**Question.** After the existing LCRC-W2 one-shot S1/S2 chemical-structure
encoding, do two layers of real chemical-bond message passing that read the
neighbours' *learned hidden states* meaningfully improve the official MolPCBA
scaffold-split predictive performance?

**One model, one run.** `W2-DYNAMIC`, seed 0, 40 epochs, frozen W2 seed-0
initialisation plus two `MessageFFN(128→256→128)` layers driven by a shared
`Linear(13,128)` bond projection. 1,128,468 parameters (= W2 994,324 +
132,352 + 1,792). No STATIC control, no W2 retraining, no extra seed, no
hyper-parameter tuning.

Outcome
-------

| | W2 | RC-REAL | **W2-DYNAMIC** |
|---|---:|---:|---:|
| Parameters | 994,324 | 1,051,022 | **1,128,468** |
| Best Valid AP | 0.265273 | 0.267718 | **0.275067** (ep 25) |
| Test AP at best-Valid | 0.253617 | 0.255941 | **0.262226** |
| Valid@40 / Test@40 | 0.261867 / 0.250986 | 0.266950 / 0.256167 | 0.268791 / 0.260136 |
| GPU-hours | 1.3949 | 1.9186 | 1.7340 |

Δ Test = **+0.008609** vs W2 and **+0.006285** vs RC-REAL; 76/108 high-support
Test tasks improve (mean +0.010866); the single <5-positive task reduces the
macro by −0.0012; the dynamic branch is used (R1 0.645 / R2 0.529 / Rtotal 0.939,
all 14 new tensors receive gradient from step 3, `message_off` −0.1601 and
within-molecule `src` shuffle −0.0993 Validation AP on the frozen checkpoint).

Pre-registered decision: **`POSSIBLE_DYNAMIC_SIGNAL`** — the holistic ΔTest of
+0.0086 misses the +0.010 `STRONG_DYNAMIC_SIGNAL` line while every other STRONG
condition holds. See `REPORT.md` for the interpretation boundaries (no
capacity-matched STATIC control; the effect is not separated from added
parameters or from the wider receptive field).

Files
-----

| Path | Contents |
|---|---|
| `PROTOCOL.md` | pre-registration: model, gates, recipe, budget, decision rules, interpretation boundaries |
| `code/dy_common.py` | frozen constants, inherited imports, historical references, freeze seal |
| `code/dy_model.py` | `W2DynamicModel` (W2 + 2 directed-bond message layers), initialisation, parameter audit, usage metrics |
| `code/dy_audit.py` | pre-training correctness/mechanism gates + gradient audit (steps 1/3/10/100) + 100-batch cost profile |
| `code/dy_train.py` | the single formal 40-epoch run (frozen W2 order plan, per-epoch R1/R2/Rtotal, gradient statistics) |
| `code/dy_mechanism.py` | post-training Validation-only `message-off` / `message-shuffle` interventions |
| `code/dy_test.py` | Test-logit freeze (`freeze`) and the post-freeze official scoring pass (`score`) |
| `code/dy_analyze.py` | local comparison table, support strata, per-task structure, robustness decomposition, decision label |
| `tests/` | 24 CPU correctness tests |
| `analysis/` | preflight, mechanism, freeze manifest, Test scores, per-task deltas, comparison, report tables |
| `REPORT.md` | observed / interpretation / decision |

Order of execution (exactly once, in this order)
------------------------------------------------

```bash
# 0. local CPU gates
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/tests

# 1. GPU pre-training gates + gradient audit + cost profile (no training artefact)
rr run res dyn-audit --pool res-gpu1 --gpus 1 --result tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/analysis -- \
  timeout 2400s uv run python -u tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_audit.py --device cuda

# 2. the single formal training run (40 epochs, seed 0)
rr run res dyn-train --pool res-gpu1 --gpus 1 --result tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/runs -- \
  timeout 12600s uv run python -u tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_train.py \
  --out-dir tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/runs/W2_DYNAMIC \
  --soft-seconds 14400 --hard-seconds 16200

# 3. post-training Validation-only mechanism intervention (frozen checkpoints)
rr run res dyn-mechanism --pool res-gpu1 --gpus 1 --result tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/analysis -- \
  timeout 1800s uv run python -u tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_mechanism.py --device cuda

# 4. freeze Test logits (no Test label is opened)
rr run res dyn-freeze --pool res-gpu1 --gpus 1 --result tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/analysis -- \
  timeout 1800s uv run python -u tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_test.py freeze --device cuda

# 5. only now: score against the official Test labels
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_test.py score

# 6. local analysis + decision
uv run python tracks/structural-sufficiency/lcrc_pcba_dynamic_v1/code/dy_analyze.py
```

Inherited, read-only
--------------------

`lcrc_pcba_backbone_scaling_v1` (W2 model/constructor/recipe/data identity),
`lcrc_pcba_structural_gain_sprint_v1` (`MessageFFN` formula and the
private-generator initialisation rule), `lcrc_pcba_boc_v1` (frozen W2 /
RC-REAL Test logits and the freeze/seal pattern),
`lcrc_pcba_continuation_recipe_v1` (the only gated official-Test-label reader).

Test discipline
---------------

The official Test split is already exposed
(`EXPOSED_TEST_ENGINEERING_COMPARISON_SET`). During training only Train and
Validation labels exist. Test logits are written and sealed before
`dy_test.py score` reads the official Test labels; W2 and RC-REAL are scored
from their already frozen BOC logits with the identical metric code. Nothing is
retrained, reselected or tuned after any Test number is seen.
