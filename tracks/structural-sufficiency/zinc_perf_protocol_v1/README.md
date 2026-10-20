# zinc_perf_protocol_v1

New performance protocol for ZINC-12k (official PyG `subset=True`, raw-y MAE)
after the audit at `592d766`.

* `PROTOCOL.md` — the new selection/validation/test protocol (best-valid single
  checkpoint, per-epoch valid, one-shot terminal test).
* `DESIGN.md` — bounded GRIT/GPM check, the ZSRT computation graph, and the
  differences from P3 / STA / BCCA.
* `configs/` — frozen protocol and ZSRT screen.
* `code/` — P3 calibration, ZSRT training/eval, freeze, terminal scoring,
  table builder, light CPU tests.
* `analysis/` — data audit, continuation decision, round decision, main
  table and curves (small, tracked); freeze/test scores only if a future round
  opens them.
* `cache/`, `runs/`, `predictions/*.npz` — generated, git-ignored.

## Round status

Closed at the development conclusion on 2026-10-20. P3 calibration completed
(two seeds, best-valid mean 0.077886); the four-config ZSRT screen completed
(best: `C_wd_high`, seed-0 best-valid 0.103363 vs P3 seed-0 0.079105). The
pre-declared stop rule was met, so no ZSRT seeds 1/2, no freeze, and no
terminal test read was purchased. See `REPORT.md` and
`analysis/round_decision.json`.

## Commands

Local CPU audit (official splits + frozen P3 cache cross-check):

```bash
uv run python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_data.py audit
uv run pytest -q tracks/structural-sufficiency/zinc_perf_protocol_v1/code/tests
```

P3 calibration (one process per GPU; each run is 360 epochs of the frozen R3
recipe with per-epoch validation and best-valid checkpointing):

```bash
rr run res zpp-p3-s0 --pool res-gpu0 --gpus 1 --time 1:00:00 \
  --result tracks/structural-sufficiency/zinc_perf_protocol_v1/runs \
  -- python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_p3.py --device cuda:0 train --seed 0
rr run res zpp-p3-s1 --pool res-gpu1 --gpus 1 --time 1:00:00 \
  --result tracks/structural-sufficiency/zinc_perf_protocol_v1/runs \
  -- python tracks/structural-sufficiency/zinc_perf_protocol_v1/code/zpp_p3.py --device cuda:0 train --seed 1
```

Continuation is only run if the pre-declared rule triggers:

```bash
python .../zpp_p3.py guard
# if triggered for a seed:
rr run ... zpp-p3-s0-cont ... -- zpp_p3.py --device cuda:0 continue --seed 0
```

ZSRT screen (seed 0) and confirmation seeds:

```bash
rr run res zpp-zsrt-A-s0 --pool res-gpu0 --gpus 1 --time 2:00:00 \
  --result .../runs -- python .../zpp_zsrt.py --device cuda:0 train --config A_main --seed 0
# B/C/D analogously; add seeds 1/2 only if the pre-declared stop rule did not fire
```

Freeze and terminal test (single read) are intentionally **not** part of this
closed round. The entry points remain for a future, separately authorised round:

```bash
python .../zpp_freeze.py freeze --config C_wd_high --seeds 0
python .../zpp_freeze.py verify
rr run res zpp-score --pool res-gpu0 --gpus 1 --time 1:00:00 \
  --result .../analysis -- python .../zpp_score.py --device cuda:0 score
python .../zpp_report.py --variants A_main,B_lr_low,C_wd_high,D_no_rrwp
```

Every training log is `runs/.../train.json` + `curve.csv`; the primary number
is `best_valid.valid_mae`, never the soup and never a test-selected epoch.
