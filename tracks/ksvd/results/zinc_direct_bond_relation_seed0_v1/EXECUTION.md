# EXECUTION — `zinc_direct_bond_relation_seed0_v1`

Round first work began 2026-10-05 08:03 CST (branch creation); formal allocations were 2026-10-05. Branch `task/zinc-direct-bond-relation-seed0-v1`, training/protocol commit `a33cb81590cc40346c04da584ca27eff8ef5a695`. No push, no merge, main untouched.

## 1. Freeze and local checks

| item | execution | outcome |
|---|---|---|
| source audit | source module `zinc_local_tuple_fresh_fold_replication_seed0_v1`; source train revision `a5400de`, fresh fold prereg `2e4ee3f` | both ancestry verified |
| fold/prep/payload | source `fresh_{fold,targets,tuple_payload,prep}.npz`; copied and source hashes checked | fit/dev/schedule/gid hashes frozen |
| beta provenance | load train-only encoded/env cache + raw PyG ZINC train edges; derive/cross-check all 10,000 graphs | exact; 249,279 adjacent pairs, category mass `[0,185060,63548,671]`; no valid/test loaded |
| init and relation path | source M untrained init, expanded 15→19; input cache and runtime forward checked | both arms 297,883 params, same init hash; source/O/T prediction max diff 0.0; actual relation width 19 |
| batching/isolation/gradient | 8 graph batch/order witness; real L1 backward; hooks and beta mutation | batch-single max 3.20e-7; reorder max 9.69e-8; O new-column grad exactly 0; T new-column grad 0.1857; E invariant; env/pair-composer called once/forward |
| recipe/smoke | schedule and construction/train RNG hash witnesses; 2-step fit-only CPU smoke (states discarded) | all pass; schedule `7b11a529…`, build RNG `a2e8a8ab…`, train RNG `1ccf1725…` |
| tests | `uv run pytest -q -m 'not slow' tracks/ksvd/tests` | 1,840 passed, 224 deselected, 1 failed: unrelated pre-existing `test_git_state_captures_commit` assumes a clean worktree and asserts a dirty tree has a nonempty diff hash. At this point the new runner/docs/results were intentionally uncommitted. No source/test changes were made to hide the environment-dependent failure. |

## 2. Remote deployment and jobs

* `rr doctor res-2 --refresh`: passed required checks; pool `res2-cu124` provisioned on c05/c06, both driver 525.85.12; Python 3.12.14, torch 2.5.1+cu124/CUDA 12.4. (The unrelated `res2-cu118-all` pool remains unprovisioned.)
* `rr deploy res-2 --pool res2-cu124`: deployed frozen commit `a33cb81590cc`; source/remote worktrees clean and matching.
* Initial submissions O (`55999`) and T (`56000`) were inadvertently both placed on one c05 GPU. They were cancelled early (O had reached about epoch 140; T about epoch 40); their partial weights/results were not used. These jobs are terminal `cancelled`, and their existence is recorded as a launch deviation, not as formal trajectories.
* Clean formal jobs: O `zinc-direct-bond-O-20261005-094104-60ad0bf5`, Slurm `56001`; T `zinc-direct-bond-T-20261005-094418-f11c30b2`, Slurm `56002`. Both exit 0, each 1 GPU, each 4 CPU threads, 32G, 95-minute Slurm limit. Slurm scheduled both sequentially on c05, **not concurrently**; no oversubscription. Runtime inventories prove distinct physical A100 PCI devices / UUIDs, driver 525.85.12:

| arm | node | GPU UUID | PCI | start (CST) | complete (CST) | training wall | steps |
|---|---|---|---|---|---|---:|---:|
| O | c05 | `GPU-c7067be6-0979-516e-45c4-27f69628e1df` | `00000000:1D:00.0` | 09:39:04 | 09:49:31 | 586.6 s | 15,120/15,120 |
| T | c05 | `GPU-0c6f2aa8-cd1b-242d-4f94-052642143c9d` | `00000000:25:00.0` | 09:42:17 | 09:54:10 | 673.8 s | 15,120/15,120 |

Both runtime reports: Python 3.12.14, torch 2.5.1+cu124, CUDA 12.4, A100-PCIE-40GB. `CUDA_VISIBLE_DEVICES=0` in each is allocation-local; UUID/PCI values identify the distinct physical cards. Both reported one allocated GPU; `SLURM_JOB_GPUS` was 2 and 3 respectively. `rr status` confirms both jobs terminal/completed/exit 0; no pending or running self-launched jobs remain.

## 3. Training integrity and analysis

* Both init state hashes match each other (`9678be159fdc…`); source M keys copied exactly, four appended relation columns exactly zero. Both use source build/train RNG streams and exact position schedule and global graph-ID stream `69187f13…`.
* Runtime probes: O new-column task gradients are exactly zero at epochs 1/40/120/240; T new-column gradient norms are `0.02398 / 0.00835 / 0.00198 / 0.00435` respectively. T therefore receives and retains task signal in the beta path; this is implementation/learning evidence, not a generalization result.
* Pulled results using `rr pull` into `.rr/pulled/.../result/` and copied only the declared O/T artifacts to the results directory. Analysis/bootstrap/replay/manifest ran locally after both formal jobs were terminal. Replay (128 fit/dev rows per arm) max errors: O fit `9.54e-7`, O dev `1.43e-6`, T fit `1.67e-6`, T dev `1.91e-6` (all ≤`1e-5`).
* Runtime input contains `pair_beta` explicitly and has 19-wide relation; path-bond-mean excluded. No beta scaler, extra categories, auxiliary loss, or write-back.

## 4. Resource accounting and final status

`budget.json` is the resource record. Formal allocation use is `586.6 + 673.8 = 1,260.4` GPU-seconds = **0.3501 GPU-hours** (rounding record conservatively reports `0.3538` GPU-h). This is within ≤0.8 GPU-h; two distinct GPU UUIDs used, at most one concurrently; each process 4 threads. Round start to final job completion ~111 minutes, below 120 minutes; no new compute after minute 90. CPU smoke/prechecks used at most 8 local threads and no GPU allocation.

All formal jobs are terminal; both formal runs passed integrity and replay requirements. Gate is negative-direction unconfirmed, not implementation failure. No additional compute is authorized or scheduled. Official valid/test never loaded.

## 5. Reproduction commands

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --prepare
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --pre-checks
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --smoke --device cpu --smoke-steps 2
# In the frozen RR deployment, one job per arm:
rr run res-2 zinc-direct-bond-O --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 95 --result tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/O_meta.json --result tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/O_raw_predictions.npz -- python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --out tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1 --train --arm O --device cuda
rr run res-2 zinc-direct-bond-T --pool res2-cu124 --gpus 1 --cpus 4 --mem 32G --time 95 --result tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/T_meta.json --result tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1/T_raw_predictions.npz -- python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --out tracks/ksvd/results/zinc_direct_bond_relation_seed0_v1 --train --arm T --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.zinc_direct_bond_relation_seed0_v1 --analyze --replay --manifest
```
