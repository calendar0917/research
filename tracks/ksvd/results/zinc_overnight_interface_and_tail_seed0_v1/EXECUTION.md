# EXECUTION — zinc-overnight-interface-and-tail-seed0-v1

Autonomous single session on branch `task/zinc-overnight-interface-and-tail-
seed0-v1` (base `8d765d0`, 2026-10-04 00:03:31 +0800), isolated: no push, no
merge.  All times local (CST, +0800).  Data boundary: official-valid was never
read in this round; official-test was never read; only `encoded_train.pt` /
`env_train.pt` and frozen prep artifacts were loaded for fitting.

## 1. Timeline and commits

| time | event |
|---|---|
| 00:46 | session start (first tool call), reconnaissance of anchors/contracts |
| 00:59:59 | `fce07ac` runner + protocol/phase-0 committed |
| 01:04:13 | `bfffdae` force-add Phase-0 evidence (results `*.json` are git-ignored); `rr deploy res-2 --pool res2-cu124` -> `bfffdaeebe16` |
| 01:04:43–01:05:46 | 5 Phase-1 Slurm jobs submitted (`zoni-p1-r-cs/sm/sj/dm/dj`); first submission round had failed before any training (missing `interface_stats.json`), fixed and resubmitted |
| ~01:14–01:39 | Phase-1 jobs completed (max 2 concurrent GPUs, node c05) |
| 01:26:24 | `f53ab69` CPU tail branch + analysis + frozen-eval scripts |
| 01:28:54 | `05414fe` PROTOCOL.md + METHOD_CONTRACT.md |
| 01:39:11 | `b7234cf` Phase-1 results + `stage2_decision.json` (T = R_DM) |
| 01:40:38–01:41:19 | Phase-2 jobs submitted (`zoni-p2-a0`, `zoni-p2-c`, `zoni-p2-t`; T queued behind GPU quota) |
| ~01:58–02:14 | Phase-2 completed (`A0` 1029 s, `C` 1025 s, `T` 863 s) |
| ~02:16–02:24 | pull, replay, mechanism, Stage-1/2 analysis, CPU replay, decision files |
| 02:25 | documentation and manifest written; all self-created jobs completed |
| < 07:46 | new-compute cutoff not approached (all training finished by ~02:14) |

Remote deployment ids: `bfffdaeebe16` (first round of compute), `b7234cfc1561`
(Phase 2).  The git worktree stayed clean of untracked compute only via
`git add -f` (results `*.json/*.pt/*.npz` are git-ignored in this repo).

## 2. Commands actually used

```bash
# local phase 0 / smoke (CPU, 8 threads)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_overnight_interface_and_tail_seed0_v1 --mode phase0
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_overnight_interface_and_tail_seed0_v1 --mode smoke --max-steps 2

# remote deploy
rr deploy res-2 --pool res2-cu124

# Phase 1 (x5 arms), Slurm/A100
rr run res-2 zoni-p1-r-<arm> --pool res2-cu124 --gpus 1 --time 04:00:00 \
  --result tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1 \
  -- python -m tracks.ksvd.experiments.luyin16.zinc_overnight_interface_and_tail_seed0_v1 \
     --mode phase1 --arm <R_CS|R_SM|R_SJ|R_DM|R_DJ> --device cuda --epochs 160

# Phase 2 (x3 arms), same wrapper with --mode phase2 --arm <A0|C|T> --epochs 240
# T is selected by stage2_decision.json (dense, marginal); A0/C automatically dense.

# pull snapshots, then materialise the shared result path
rr pull zoni-p1-r-dj      # snapshot dir contains all five arms (shared FS)
cp -f .rr/pulled/<run>/result/tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/* <results>/
rr pull zoni-p2-t
cp -f .rr/pulled/<run>/result/tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/* <results>/

# local analysis / replay (CPU <= 8 threads)
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/analyze.py --stage 1
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/analyze.py --stage 2
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/analyze.py --replay
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/analyze.py --mechanism
PYTHONPATH=. uv run python tracks/ksvd/results/zinc_overnight_interface_and_tail_seed0_v1/replay_cpu.py

# CPU tail training (local, 4 threads)
PYTHONPATH=. uv run python -m tracks.ksvd.experiments.luyin16.zinc_overnight_interface_and_tail_seed0_v1_cpu --stage 8k

# final validation entry (committed as standby for a future passing round;
# deliberately NOT invoked: no gate passed)
# PYTHONPATH=. uv run python .../final_eval.py --mode freeze
# PYTHONPATH=. uv run python .../final_eval.py --mode heldout
```

Hardware/software regime: `res-2` (Slurm), pool `res2-cu124`, node `c05`,
NVIDIA A100-PCIE-40GB, driver `525.85.12`, torch `2.5.1+cu124`, FP32, no
AMP, no DDP, `torch.set_num_threads(4)` inside GPU jobs.

## 3. Verification performed

* Phase 0 (`phase0_checks.json`): split hashes, dev strata, parameter audit
  (395,763 = 267,611 + 2,080 + 126,072), zero-init identity
  (`9.5e-07` fit / `1.9e-06` dev), `J != M` on real batches, `M` invariant to
  category permutation / `J` changes, `n=1 J=M`, `n=0` zero, label-shuffle
  invariance, adapter-only phase-1 optimizer with dead step-1 upstream gradient
  and live phase-2 `D` reconstruction gradient.
* Smoke: shared adapter init across arms, frozen parent byte-identical after
  smoke steps, `rec > 0` in Phase 2.
* All formal runs: shared schedule stream per phase, soup epochs 156–160 and
  236–240, 10080/15120 steps, true final states saved.
* `replay_check.json`: every Phase-1/Phase-2 soup rebuilt from its state dict
  reproduces the saved fit/dev predictions with max abs diff `<= 4.8e-06`;
  `B` replay `<= 1.9e-06`.
* `cpu_replay_check.json`: `P_U`/`P_B` heads rebuilt from soup states
  reproduce `q_fit/q_dev`, `b_P` and dev cal MAE exactly (`0.0`); `P_U`
  reproduces the released `P_seed0` exactly.
* Group contributions sum to the overall endpoint exactly for every arm in
  both Phase-1 and Phase-2 tables (partition `k=0/-1/-2/<=-3`).
* Budget/`nvidia` concurrency: max 2 concurrent jobs observed (Slurm quota);
  all 8 jobs `completed`; none cancelled/requeued after starting.

## 4. Deviations, failures, recovery

1. **First Phase-1 launch failed before training** (5 submissions): remote
   checkout lacked `interface_stats.json` because `tracks/*/results/**/*.json`
   is git-ignored.  Fix: `git add -f` the Phase-0 evidence (`bfffdae`), redeploy,
   resubmit.  The failed jobs executed zero training steps and are not counted
   as trajectories.
2. **CPU tail batch scheduling bug** in the first implementation (fresh
   `randperm` per batch).  Fixed before any result was used; `P_U` then
   reproduced the released `P_seed0` exactly.  The buggy run is not reported.
3. **Phase-2 overwrite risk**: `train_arm` writes the same `{arm}_*` names for
   Phase 2 and Phase 3.  Since no Phase 3 was bought, current files are the
   Phase-2 states; they were additionally archived as `phase2_{arm}_*` for
   unambiguous provenance.
4. No protocol change after inspecting results: `T` stayed `R_DM` (dense,
   marginal) even though `C` beat `T` in Phase 2; the pre-registered gate
   closed the round.

## 5. Budget ledger (`budget.json`)

| job | phase | run id | wall s | GPU-h |
|---|---|---:|---:|---:|
| zoni-p1-r-cs | 1 | 5879a41b | 532.1 | 0.1478 |
| zoni-p1-r-sm | 1 | 40ad6584 | 628.6 | 0.1746 |
| zoni-p1-r-sj | 1 | 19466411 | 559.8 | 0.1555 |
| zoni-p1-r-dm | 1 | cafa78b4 | 578.7 | 0.1608 |
| zoni-p1-r-dj | 1 | fcc8aaac | 517.8 | 0.1438 |
| zoni-p2-a0 | 2 | c6176091 | 1028.6 | 0.2857 |
| zoni-p2-c | 2 | 44015836 | 1024.9 | 0.2847 |
| zoni-p2-t | 2 | 57f4f9af | 862.7 | 0.2396 |
| **total** | | | **5733.2** | **1.5926** |

* GPU-hour cap `6` -> used `1.5926` (26.5%).  Max concurrent GPUs `2` -> `2`.
  Formal trajectories `8` (`5 + 3`) <= `10`.  Seed 0 only.  New training
  stopped at ~02:14, well before the 07:46 cutoff.
* Local CPU: `set_num_threads(8)` in analysis/phase-0, `4` in the CPU tail
  runner/replay; CPU tail training 19.2 s total, analysis/replay minutes;
  no other heavy local jobs.

## 6. Shutdown / hygiene

* `rr jobs` at delivery time: all 8 experiment entries `completed`; no
  pending/running self-created jobs.  (Check performed immediately before the
  final commit.)
* No writes to official-valid/test caches; `official_valid_loaded=false`,
  `official_test_loaded=false` is asserted in every result JSON.
* No push, no merge, no writes outside the isolated branch and the results
  directory.
