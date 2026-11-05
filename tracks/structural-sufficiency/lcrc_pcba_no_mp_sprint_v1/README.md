# LCRC PCBA no-edge-MP sprint v1

Pre-registered test of whether explicit S1/S2 local statistics plus a
capacity-matched **static** computation (S), an **attention readout** (A), or
**graph-context feedback** (G) can approach the previously trained bond-MP
reference **V** on `ogbg-molpcba` Without iterative message passing along real
chemical bonds.

See [`PROTOCOL.md`](PROTOCOL.md) for the frozen arms, correctness gates,
training recipe, stopping rules and interpretation labels. The historical C/V
runs are references only and are never retrained here.

## Layout

```
code/
  nmp_common.py       frozen constants, arms, schedule, historical references
  nmp_data.py         thin re-export of the audited PCBA loader / PinnedStager
  nmp_model.py        S / A / G modules on the frozen LCRC-C1 base
  nmp_checkpoint.py   complete continuation checkpoints and strict restore
  nmp_train.py        Phase B (epochs 1-20), C (21-40), D (seed-1 40 epochs)
  nmp_preflight.py    GPU identity/access/gradient/speed/FLOPs preflight
  nmp_audit.py        architecture + information-access + parity audits
  nmp_analyze.py      phase reports, per-task CSV, ledger, decisions
  nmp_attention.py    post-hoc Arm-A attention diagnostics only
tests/                CPU correctness / parity tests
analysis/             audit and phase JSON/CSV deliverables
runs/                 per-run outputs (git-ignored, pulled from `res`)
```

## Local checks

```bash
cd /home/calendar/code/research
uv run pytest -q tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/tests
uv run python tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/code/nmp_audit.py --require-preflight
```

## Formal GPU runs (`res`, pools `res-gpu0`/`res-gpu1`)

All formal commands are launched from a committed revision via `rr`:

```bash
rr doctor res
rr deploy res --pool res-gpu0   # and res-gpu1
rr run res nmp-preflight-gpu0 --pool res-gpu0 --result <round>/analysis \
  -- python tracks/structural-sufficiency/lcrc_pcba_no_mp_sprint_v1/code/nmp_preflight.py \
     --device cuda --arms S,G --out <round>/analysis/preflight_gpu0.json
rr run res nmp-preflight-gpu1 --pool res-gpu1 --result <round>/analysis \
  -- python .../nmp_preflight.py --device cuda --arms A --out .../preflight_gpu1.json
```

Phase B (seed 0, epochs 1–20; two arms on gpu0/gpu1, then G on gpu0):

```bash
rr run res nmp-pb-s0-S --pool res-gpu0 --result <round>/runs/phaseB_seed0_S \
  -- python .../nmp_train.py --phase B --arm S --seed 0 --device cuda \
     --out-dir <round>/runs/phaseB_seed0_S --soft-seconds 9000 --hard-seconds 10800
```

Phase C resumes from `runs/phaseB_seed0_<arm>/checkpoint_epoch20.pt`; Phase D
trains seed 1 from scratch with the same recipe.

`rr pull <experiment> --dest <local round dir>` brings back the declared
result directory. Checkpoints and curves are git-ignored; analysis JSON/CSV are
tracked.

## Non-negotiable invariants

- Official Test targets are never read or scored.
- No arm reads `bond_index`/`bond_feat` in its added modules; the frozen S1/S2
  encoder may still use the graph to build `h`.
- All arms share the C1 initialisation, 40-epoch order plan, masked BCE,
  dropout, optimizer and per-step cosine schedule of the previous Phase 3.
- The primary comparison is the fixed epoch 40; no best-epoch promotion.
- G is reported separately from the strict no-MP S/A route.
