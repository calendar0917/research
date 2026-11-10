# LCRC-W3 regularization diagnostic v1

Round `lcrc-pcba-w3-regularization-diagnostic-v1` (track: `structural-sufficiency`).

One question: historical **LCRC-A-W3** (2,120,844 parameters) on MolPCBA raised
Train AP from 0.4197 (A0) to 0.7055 while Validation AP stalled (0.2619 W2 →
0.2590 W3). Is that at least partly a regularization recipe that never adapted
to the model size?

Two and only two new runs, everything else frozen at the historical W3 recipe
(seed 0, 40 epochs, batch 128, AdamW betas (0.9, 0.999) eps 1e-8, peak LR 3e-4,
1-epoch warmup from 3e-5, per-step cosine to 1e-5, clip 5.0, FP32, masked BCE,
the historical seed-0 batch order, no early stopping):

| Run | Dropout | AdamW weight_decay |
|---|---:|---:|
| `W3_WD10` | 0.10 | **0.010** |
| `W3_DO20` | **0.20** | 0.001 |

Every epoch trains on the full Train split and then evaluates **Train,
Validation and Test in full** (official OGB macro AP, masked BCE, per-task AP,
support counts). The primary checkpoint is the maximum Validation AP (ties →
earlier epoch). Test is scored every epoch by design and labelled
`EXPOSED_TEST_EXPLORATORY`; it never selects anything.

## Layout

```
PROTOCOL.md                  frozen pre-registration (§12.1 pins the label arithmetic)
code/rd_common.py            runs, thresholds, historical numbers, inherited imports
code/rd_data.py              three-split evaluation + the isolated official Test label reader
code/rd_model.py             W3 constructor + dropout/weight-decay interventions and probes
code/rd_train.py             the 40-epoch trainer (three-split eval, checkpoints, run.json)
code/rd_audit.py             data/model/test-isolation/smoke/historical-test audits
code/rd_compare.py           comparison, per-task deltas, curves, labels, figures
tests/                       38 CPU tests (frozen definition, model, data, train/compare)
analysis/                    committed JSON/CSV evidence + figures
runs/W3_WD10, runs/W3_DO20   run artifacts (git-ignored, pulled from the GPU host)
```

Inherited **read-only**: `lcrc_pcba_backbone_scaling_v1/code/bs_{common,model,train}.py`
(the historical W3 constructor, width audit and training step),
`sg_common`/`sg_data` (recipe, data, order plan, official OGB evaluator),
`lr_data`, `pc_metrics`, `pc_model`, `pf_pipeline`, `tu_model`, `nmp_model`.

## Commands

```bash
# 0) CPU correctness suite (~30 s)
uv run --no-sync pytest -q tracks/structural-sufficiency/lcrc_pcba_w3_regularization_diagnostic_v1/tests

# 1) CPU audits (host-independent): frozen data, both models, Test isolation,
#    and the single read-only historical-W3 Test fill
uv run --no-sync python <round>/code/rd_audit.py \
  --modes data,model,test-isolation,historical-test --device cpu --out-dir <round>/analysis

# 2) GPU smoke: 10 real training steps + one full three-split evaluation,
#    timings and the GPU-hour projection (writes evaluation_cost_probe.json)
rr run res rd-smoke --pool res-gpu1 --result <round>/runs/audit -- \
  bash -c "uv run --no-sync python <round>/code/rd_audit.py --modes smoke --device cuda --out-dir <round>/runs/audit"

# 3) the two formal runs (GPU0 was externally saturated, so both ran serially on GPU1)
rr run res rd-w3-wd10 --pool res-gpu1 --result <round>/runs/W3_WD10 -- \
  bash -c "uv run --no-sync python <round>/code/rd_train.py --run W3_WD10 --seed 0 --device cuda \
           --out-dir <round>/runs/W3_WD10 --soft-seconds 16000 --hard-seconds 18000"
rr run res rd-w3-do20 --pool res-gpu1 --result <round>/runs/W3_DO20 -- \
  bash -c "uv run --no-sync python <round>/code/rd_train.py --run W3_DO20 --seed 0 --device cuda \
           --out-dir <round>/runs/W3_DO20 --soft-seconds 16000 --hard-seconds 18000"

# 4) pull and compare (local)
rr pull rd-w3-wd10 && rr pull rd-w3-do20
uv run --no-sync python <round>/code/rd_compare.py
```

A failed run resumes from `checkpoint_last.pt`
(`--resume <round>/runs/<run>/checkpoint_last.pt`); a completed 40-epoch run is
never rerun.

## Result

Both runs completed 40/40 epochs with full three-split evaluation every epoch
(`W3_WD10` 2.18 GPU-h, `W3_DO20` 1.96 GPU-h; 4.17 GPU-h total including the
smoke, target was <= 10).

| Model | Best epoch | Valid AP best | Valid AP@40 | Train AP@40 | Test AP@best | Label |
|---|---:|---:|---:|---:|---:|---|
| W3 historical | 25 | 0.263670 | 0.259004 | 0.705498 | 0.251822 | — |
| W3_WD10 | 19 | 0.264135 (+0.00047) | 0.259214 (+0.00021) | 0.694325 | 0.254055 | `NO_CLEAR_GAIN` |
| W3_DO20 | 19 | 0.263498 (-0.00017) | 0.260978 (+0.00197) | 0.663519 | 0.257691 | `NO_CLEAR_GAIN` |

See [`REPORT.md`](REPORT.md) for the full analysis (Test is exploratory only).
