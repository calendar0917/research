# EXECUTION — zinc-task-dictionary-and-cycle-witness-seed0-v1

Everything below is the actual command/decision trail.  Result dir:
`tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/`.
Branch: `task/zinc-task-dictionary-and-cycle-witness-seed0-v1` (isolated,
created in place from main `7668597`; no push, no merge).

## 1. Timeline and revisions

| time (CST) | event |
| --- | --- |
| 09:16:50 | round start (wall-clock budget begins) |
| ~09:18-09:26 | recon, runner + CPU witness module written, smoke run passed locally |
| `77c40dd` | commit: main runner + CPU witness module (training revision, remote `77c40dda9d5b`) |
| `2e8cc80` | commit: `PROTOCOL.md`, `METHOD_CONTRACT.md`, `protocol.json`, `jm_flip_replay.py`, CPU evidence |
| 09:27:09/09:27:28 | Slurm jobs `55894` (`ztdw-d`) and `55895` (`ztdw-m`) start on `c05` |
| 09:37:55 | `ztdw-m` completes (exit 0, 600.7 s trained) |
| 09:42:41 | `ztdw-d` completes (exit 0, 903.2 s trained) |
| 09:29:58-09:33:14 | local CPU witness outputs written (bounds -> witnesses -> renumber -> J/M flip) |
| 09:48:17 | `rr pull` snapshots materialised into the local result dir |
| 09:51:40 | refreshed `smoke_checks.json` |
| 09:54:28 | `analysis.json` + tables (re-run after analysis-only fixes) |
| ~10:00 | docs (`REPORT.md`, `DECISION.md`, `ERRATA.md`, `EXECUTION.md`), manifest, final commit |

Deployed revision for both GPU jobs: `77c40dda9d5b730a64ea653b63b053a11460993d`,
`git_dirty = false`, `git_diff_hash = e3b0c44298fc1c14` (empty diff).  The
local analysis-side edits made after the launch never touched `train_arm`,
`build_arm`, the fold, the schedule or the gate thresholds.

## 2. Commands actually used

```bash
# branch (in place, from main HEAD)
git switch -c task/zinc-task-dictionary-and-cycle-witness-seed0-v1

# local CPU witness (train-only; <= 8 threads)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_cycle_witness_seed0_v1 --stage bounds
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_cycle_witness_seed0_v1 --stage witnesses
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_cycle_witness_seed0_v1 --stage renumber
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/jm_flip_replay.py

# engineering smoke (all_ok=true; re-run into a temp out_dir so the results
# dir stays free of smoke states, then copied smoke_checks.json in)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_task_dictionary_and_cycle_witness_seed0_v1 --smoke

# remote deploy + the two formal arms (Slurm/A100, 1 GPU each)
rr deploy res-2 --pool res2-cu124
rr run res-2 ztdw-d --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 02:00:00 \
  --result tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1 \
  -- python -m tracks.ksvd.experiments.luyin16.zinc_task_dictionary_and_cycle_witness_seed0_v1 --arm D --device cuda
rr run res-2 ztdw-m --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 02:00:00 \
  --result tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1 \
  -- python -m tracks.ksvd.experiments.luyin16.zinc_task_dictionary_and_cycle_witness_seed0_v1 --arm M --device cuda
rr status ztdw-d ; rr status ztdw-m
rr logs ztdw-d ; rr logs ztdw-m

# pull snapshots (D finished later, so its snapshot contains both arms)
rr pull ztdw-m ; rr pull ztdw-d
cp -f .rr/pulled/<run>/result/tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/* \
      tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/

# local analysis (CPU, <= 8 threads)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_task_dictionary_and_cycle_witness_seed0_v1 --analyze

# manifest (hashes code, docs and every result file; run last)
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_task_dictionary_and_cycle_witness_seed0_v1/make_manifest.py
```

Hardware/software regime (recorded by `rr`): `res-2`, Slurm, pool
`res2-cu124`, node `c05`, NVIDIA A100-PCIE-40GB (`CUDA_VISIBLE_DEVICES=0`),
driver `525.85.12`, torch `2.5.1+cu124` (CUDA 12.4), python 3.12.14, FP32,
no AMP, no DDP, `torch.set_num_threads(4)` inside GPU jobs.

## 3. Verification performed

* `smoke_checks.json`: `all_ok = true` — parameter audit 184,667 + 82,944 =
  267,611 per arm; both bridge variants feed the consumed aux `E`; `y` cannot
  change a prediction; grouped endpoints equal per-molecule endpoints;
  optimizer steps change weights; bridge perturbation moves the prediction;
  identical tiny-training schedule.  No official split loaded.
* `D_init_state.pt` vs `M_init_state.pt`: 34 shared (non-bridge) tensors, zero
  mismatch, `shared_init_sha256 = 5a8d8ff0...` (same as the smoke hash).
* Both runs: `15,120/15,120` steps, `stopped_reason = completed`, schedule AND
  data-stream hash `7b11a529...`, 128-batch CPU/GPU replay `1.67e-06`,
  `contract_errors = []`.
* CPU witness: `global_min_l1` and old `5.54946` reproduced exactly
  (`checks.{global_min_l1_matches_old_overall, group_kle3_matches_old_5_549}`
  both true); byte-classes = exact classes = 327; no signed-zero ambiguity.
* No official-valid/test open anywhere (`official_*_loaded = false` in
  `smoke_checks.json`, both `{D,M}_meta.json` and `analysis.json`).

## 4. Deviations, failures, recovery

* Two analysis-only defects found after training (no result was read from them):
  (a) `paired_bootstrap_gain` stratified indexing mixed full-dev masks with
  G0-subset arrays -> `IndexError`; fixed by bootstrapping the subset arrays
  directly; (b) the `exists` label for "overall improvement >= 0.003" was
  tested on the signed gain; fixed to `abs(point) >= Delta` with an explicit
  `favored` field.  Both are analysis-side only; training revision and trained
  states unchanged.
* No GPU job failed, none was cancelled, no re-run of a different
  configuration.  The D arm is ~1.5x slower than M (ISTA unroll); both fit
  comfortably in the 2 h Slurm limit.
* No third arm, no second seed, no hyper-parameter search, no dev-based
  selection, no official-valid rescue attempt.

## 5. Budget ledger (`budget.json`)

* Trained GPU wall: `903.2 s` (D) + `600.7 s` (M) = `1503.9 s = 0.418
  GPU-hours` (limit 1.2), max 2 parallel GPUs (limit 2).
* CPU witness + analysis: ~4 min witness stages + ~16 min smoke/analyze
  re-runs, <= 8 threads (limit 45 min).
* Round wall clock: 09:16:50 -> ~10:00 CST, inside the 180 min budget; compute
  stopped well before the 150 min cutoff.

## 6. Shutdown / hygiene

* `rr status ztdw-d` / `rr status ztdw-m`: `state = completed`, `exit_code =
  0`; no pending or running self-created jobs remain.
* Old result directories byte-preserved; this round only adds the new result
  dir and the two committed source files.
* Branch left isolated: no push, no merge into main.

## Addendum — post-round merge (operator instruction)

The round was executed, audited and committed on its isolated branch under the
original "no push / no merge" constraint.  After the round closed, the
operator instructed "merge to main, commit, push".  On 2026-10-04 10:00:53 CST
`task/zinc-task-dictionary-and-cycle-witness-seed0-v1` was merged into `main`
with `--no-ff` (merge commit `7fd0e28`) and pushed to `origin/main`.  No result
file, model state, prediction or metric changed; the research decision is
unaffected (no write-back into the model pipeline) — this is a code/record
merge only.  `manifest.json` was refreshed after this addendum.
