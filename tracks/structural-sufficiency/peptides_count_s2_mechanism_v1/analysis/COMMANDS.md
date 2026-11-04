# Executed commands (round `peptides-count-s2-mechanism-v1`)

All commands were run from `/home/calendar/code/research` unless noted. Remote
execution used only `rr` (host `res`, process backend, pools `res-gpu0` /
`res-gpu1`, one training job per GPU). `$ROUND` abbreviates
`tracks/structural-sufficiency/peptides_count_s2_mechanism_v1`.

## Recon / environment

```bash
git status --short
uv run research context
rr doctor res
```

`rr doctor res` (before formal runs): 10 ok / 0 warn / 0 fail; 2× A100-SXM4-40GB;
torch 2.5.1+cu124; pools `res-gpu0` and `res-gpu1` ready.

## P0 features

```bash
cd $ROUND
PYTHONPATH=code uv run python code/pc_p0_features.py
```

Builds `analysis/features_{train,valid}.npz`, `raw_{train,valid}.npz`,
`features_{train,valid}.json`, and `p0_features_manifest.json`. Train
4,492,788 paths / Valid 949,954 paths; 64.2 s.

## ECFP reproduction (isolated CPU venv)

```bash
cd $ROUND
PYTHONPATH=code /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python code/pc_p0_ecfp.py
```

Result: 72.9760% vs historical 72.98%; delta −0.0040 pp; `reproduction_ok=true`.

## P0 fixed LightGBM / MLP

```bash
cd $ROUND
PYTHONPATH=code /home/calendar/.cache/research-venvs/peptides-ecfp/bin/python code/pc_p0_lgbm.py
PYTHONPATH=code uv run python code/pc_p0_mlp.py
```

## CPU correctness gates and unit tests

```bash
cd $ROUND
PYTHONPATH=code uv run python code/pc_gates.py
PYTHONPATH=code uv run pytest -q tests
```

`correctness_gates.json` PASS; 6 unit tests pass.

## Commit / deploy

```bash
cd /home/calendar/code/research
git add $ROUND
git commit -m "Add Peptides-func S2 count mechanism v1 protocol, code, P0 artifacts, gates"
git add -f $ROUND/analysis/features_train.npz $ROUND/analysis/features_valid.npz
git commit -m "Stage frozen S2 count matrices for remote P1 runs"
rr deploy res
```

Deployment initially failed because the remote had untracked files in
`lcrc_peptides_weighted_bce_v1/runs/weighted_bce_seed0/` identical to local
tracked files. They were moved (not deleted) to
`/home/hxy/cy/rr_untracked_backup/lcrc_peptides_weighted_bce_v1_runs_20261104/`;
deployment then succeeded.

After the CUDA init-gate fix:

```bash
git add $ROUND/code/pc_p1_train.py $ROUND/PROTOCOL.md
git commit -m "Gate zero-init branch on exact branch delta; record CUDA base-forward roundoff"
rr deploy res
```

## GPU engineering smoke (A2, 2 epochs)

```bash
cd /home/calendar/code/research
rr run res pcs2-smoke-a2-gpu3 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 00:20:00 \
  --result $ROUND/runs/smoke_A2_gpu \
  -- uv run python $ROUND/code/pc_p1_train.py --arm A2 --seed 0 --device cuda \
     --smoke-epochs 2 --out-dir $ROUND/runs/smoke_A2_gpu --max-seconds 900
rr pull pcs2-smoke-a2-gpu3
```

Smoke passed all gates; reload delta 6.5e−6; recorded in `analysis/gpu_smoke.json`.
(Earlier failed smokes were `pcs2-smoke-a2-gpu` and `pcs2-smoke-a2-gpu2`; see
REPORT §8.)

## Formal P1a / P1b seed 0

```bash
cd /home/calendar/code/research
rr run res pcs2-p1a-480-s0 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 02:30:00 \
  --result $ROUND/runs/p1a_480_s0 \
  -- uv run python $ROUND/code/pc_p1_train.py --arm P1A --seed 0 --device cuda \
     --out-dir $ROUND/runs/p1a_480_s0 --max-seconds 8400

rr run res pcs2-a0-s0 --pool res-gpu1 --gpus 1 --cpus 4 --mem 32G --time 01:30:00 \
  --result $ROUND/runs/a0_s0 \
  -- uv run python $ROUND/code/pc_p1_train.py --arm A0 --seed 0 --device cuda \
     --out-dir $ROUND/runs/a0_s0 --max-seconds 4500

rr run res pcs2-a1-s0 --pool res-gpu1 --gpus 1 --cpus 4 --mem 32G --time 01:30:00 \
  --result $ROUND/runs/a1_s0 \
  -- uv run python $ROUND/code/pc_p1_train.py --arm A1 --seed 0 --device cuda \
     --out-dir $ROUND/runs/a1_s0 --max-seconds 4500

rr run res pcs2-a2-s0 --pool res-gpu0 --gpus 1 --cpus 4 --mem 32G --time 01:30:00 \
  --result $ROUND/runs/a2_s0 \
  -- uv run python $ROUND/code/pc_p1_train.py --arm A2 --seed 0 --device cuda \
     --out-dir $ROUND/runs/a2_s0 --max-seconds 4500

rr run res pcs2-a3-s0 --pool res-gpu1 --gpus 1 --cpus 4 --mem 32G --time 01:30:00 \
  --result $ROUND/runs/a3_s0 \
  -- uv run python $ROUND/code/pc_p1_train.py --arm A3 --seed 0 --device cuda \
     --out-dir $ROUND/runs/a3_s0 --max-seconds 4500
```

Each job was pulled with `rr pull <experiment>` and the result tree copied from
`.rr/pulled/<run-id>/result/$ROUND/runs/<name>/` to `$ROUND/runs/<name>/`.

## Aggregation / final consistency

```bash
cd $ROUND
PYTHONPATH=code uv run python code/pc_aggregate.py
PYTHONPATH=code uv run python code/pc_final_checks.py
PYTHONPATH=code uv run pytest -q tests
```

`pc_aggregate.py` writes `analysis/p1_summary.json`,
`analysis/results_machine.json`, and `analysis/REPORT_TABLES.md`.
`pc_final_checks.py` writes `analysis/formal_consistency.json` (133 checks PASS).

## Checksums

```bash
cd $ROUND
sha256sum PROTOCOL.md REPORT.md configs/protocol.json code/*.py tests/*.py \
  analysis/*.json analysis/features_train.npz analysis/features_valid.npz \
  runs/*/{run.json,curve.csv,selection.json,checkpoint.pt,predictions.npz} \
  > /tmp/pcs2_sha256.txt
```

The machine-readable version is `analysis/checksums.json`; it includes only
files that exist in the final local worktree.
