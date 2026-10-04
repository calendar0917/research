# EXECUTION — zinc-chemistry-dictionary-vs-mlp-seed0-v1

Command/decision trail.  Result dir:
`tracks/ksvd/results/zinc_chemistry_dictionary_vs_mlp_seed0_v1/`.
Branch: `task/zinc-chemistry-dictionary-vs-mlp-seed0-v1` (isolated, created in
place from main; no push, no merge).

## 1. Timeline (CST)

| time | event |
| --- | --- |
| 11:38:05 | round start (wall-clock budget begins) |
| 11:39-11:41 | recon: previous round protocol, runner, label sources, luyin19 |
| 11:42-11:44 | exploratory fit-only constant refit; reproduced old stage_refine exactly; 0 k-label changes |
| 11:44-11:45 | new runner written; local `--phase-a` (first attempt failed on the k-count check dict shape -> fixed, no score seen) |
| 11:46:11 | local engineering smoke launched; `all_ok=True` in 14.8 s |
| 11:47:33 | `--phase-a` refresh (env_train manifest path fix) |
| 11:49:39 | Phase A complete: target provenance, anchors, init identity, input manifest |
| ~11:51 | `PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `protocol.json` frozen |
| 11:52 | commit `8bf8b4e` (pre-registration + fit-only targets + runner + local smoke) |
| 11:52:xx | `rr deploy res-2 --pool res2-cu124` -> `8bf8b4eae3b2` |
| 11:53:xx | remote smoke `zcdm-smoke` on c05 A100: **failed** on the exact-zero GPU target-not-read check (GPU forward float noise 4.77e-07), engineering fix only |
| 11:55 | commit `7c0224a` (tolerance 1e-5 for a float-noise check; no science change); redeploy |
| 11:54:32 / 11:54:48 | remote smoke `zcdm-smoke2` on c05: `exit_code=0`, `all_ok=True` |
| 11:56:30 / 11:56:45 | formal arms `zcdm-d` (Slurm 55898) and `zcdm-m` (55899) submitted, both on c05 |
| 12:04:48 | `zcdm-m` completed, exit 0, 573.4 s trained, replay 1.43e-06 |
| 12:10:09 | `zcdm-d` completed, exit 0, 908.7 s trained, replay 1.19e-06 |
| 12:12 / 12:15 | `rr pull zcdm-m` / `rr pull zcdm-d` materialised into the local result dir |
| 12:15:44 | local `--analyze` started (CPU, <= 8 threads) |
| 12:18:07 | analysis done: gate `INCONCLUSIVE`, relief `RELIEF_UNCONFIRMED`; `--replay` all_ok |
| 12:18:46 | `--figures`; `--mechanism` finished |
| ~12:25 | `--budget`, `--manifest`, final commit; round close |

## 2. Commands actually used

```bash
git switch -c task/zinc-chemistry-dictionary-vs-mlp-seed0-v1

# Phase A (local CPU, <= 8 threads)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_chemistry_dictionary_vs_mlp_seed0_v1 --phase-a

# local engineering smoke (states discarded)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_chemistry_dictionary_vs_mlp_seed0_v1 --smoke --out /tmp/zcdm_smoke

# remote preflight + formal arms
rr doctor res-2
rr deploy res-2 --pool res2-cu124
rr run res-2 zcdm-smoke2 --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 00:30:00 \
  --result tracks/ksvd/results/zinc_chemistry_dictionary_vs_mlp_seed0_v1 \
  -- python -m tracks.ksvd.experiments.luyin16.zinc_chemistry_dictionary_vs_mlp_seed0_v1 --smoke --device cuda \
     --out tracks/ksvd/results/zinc_chemistry_dictionary_vs_mlp_seed0_v1
for arm in D M; do rr run res-2 zcdm-${arm,,} --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 02:00:00 \
  --result tracks/ksvd/results/zinc_chemistry_dictionary_vs_mlp_seed0_v1 \
  -- python -m tracks.ksvd.experiments.luyin16.zinc_chemistry_dictionary_vs_mlp_seed0_v1 --arm ${arm} --device cuda \
     --out tracks/ksvd/results/zinc_chemistry_dictionary_vs_mlp_seed0_v1; done

# pull + local analysis (CPU)
rr pull zcdm-d ; rr pull zcdm-m
PYTHONPATH=. uv run python -m ... --analyze ; --replay ; --mechanism ; --figures ; --budget ; --manifest
```

## 3. Verification actually performed

* Phase A: fresh `build_arm` reproduces the saved untrained init states
  item-for-item (36/36 tensors per arm, 0 mismatch) and the published shared
  hash `5a8d8ff0…`; initial D/M functions differ (max abs 0.4010).
* Historical anchors recomputed exactly (max |delta| = 0.0) for
  `D_y 0.10429955195479323` / `M_y 0.10330495342758773` on new-dev G0 cal MAE,
  old biases, fold hashes and schedule hash.
* Target: `k` unchanged vs the old decomposition (0/10000 rows); `c` refit
  difference max `0.03365`; `y = g + c` exact; fit/dev k counts frozen.
* Remote smoke (c05, A100, driver 525.85.12, torch 2.5.1+cu124):
  `all_ok=True`; shared body hash, both consumer checks, `L1(g)` loss
  semantics, endpoint offsets, optimizer updates, bridge perturbation,
  schedule match.
* Remote smoke (c05, A100, driver 525.85.12, torch 2.5.1+cu124):
  `all_ok=True`; shared body hash, both consumer checks, `L1(g)` loss
  semantics, endpoint offsets, optimizer updates, bridge perturbation,
  schedule match.  Formal arms: 15,120/15,120 steps, `completed`, schedule /
  data-stream `7b11a529…`, fold hashes frozen, parameter audit 267,611, CPU/GPU
  replay `1.19e-06` (D) / `1.43e-06` (M) at training and `9.54e-07` /
  `1.19e-06` on independent reload (threshold `1e-5`).
* Result: new-dev G0 cal g-MAE `D_g 0.103614` / `M_g 0.101875`,
  `G_g = -0.001739` CI `[-0.005604, +0.002716]`; overall cal gain `-0.001769`
  CI `[-0.005804, +0.002341]`; raw gains favour `D_g` (`+0.002262` G0,
  `+0.002122` overall); fit cal favours `M_g` (`-0.004011`, CI separated).
  Bootstrap self-tests all pass; sensitivity (drop `train:8052`) keeps the
  category; `G_y` reproduced the old round exactly.

## 4. Deviations, failures, recovery

* Phase A k-count check compared aggregate `k<=-2` against a 4-key dict; fixed
  before any training and before any new score was read.
* First remote smoke failed on `target_not_read` because the GPU forward is
  float-noisy at ~5e-7 between two calls; the check is now a documented
  tolerance (`1e-5`) instead of exact zero.  This is an engineering-only fix;
  the training revision after this fix is `7c0224a` and both formal arms use
  it.
* No hyper-parameter change, no third arm, no second seed, no dev-based
  selection, no official-valid/test read.

## 5. Budget ledger

Wall-clock GPU: `D_g 908.7 s` + `M_g 573.4 s` (training) plus two 1-GPU smoke
jobs (`42 s + 42 s`) = `1621 s = 0.450 GPU-h`; training-only `0.4117 GPU-h`
(limit 1.0); at most 2 GPUs at once (limit 2).  See `budget.json`.

## 6. Post-round closure

* `rr jobs`: all four this-round jobs are terminal (`zcdm-smoke` failed on the
  pre-fix float-noise check, `zcdm-smoke2`/`zcdm-d`/`zcdm-m` exit 0); no
  running or queued self-created job remains.
* See `budget.json` for the GPU-hour ledger and `manifest.json` for hashes.
* The branch is left isolated (no push, no merge); old result directories are
  byte-preserved.
