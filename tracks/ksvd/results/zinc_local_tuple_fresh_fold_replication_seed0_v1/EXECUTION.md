# EXECUTION — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

Round start (first tool call): **2026-10-04 20:39 CST**. Budget: ≤150 min wall, compute stop at
135 min, ≤1.0 GPU-h (smoke/failures/recovery included), ≤2 concurrent GPUs, local CPU ≤8 threads
per child process. Branch `task/zinc-local-tuple-fresh-fold-replication-seed0-v1`; the two formal
trajectories ran on deployed revision **`2e4ee3f70b14`** (`git_dirty=false`).

## 1. Pre-work (local CPU, ≤8 threads)

| step | command | result |
|---|---|---|
| phase A | `--phase-a` | 6.5 s; fold/targets/payload/prep built; all checks below pass |
| operator checks | `--operator-checks` | all_ok=true; model vs reference max abs `0.0`; batch-vs-single concat `0.0` |
| smoke B | `--smoke --arm B --device cpu` | 3 fit-only steps, `all_ok=true`; states discarded |
| smoke M | `--smoke --arm M --device cpu` | 3 fit-only steps, `all_ok=true`; states discarded |

Key pre-run evidence (`fresh_manifest.json`, `init_identity.json`, `operator_path_checks.json`,
`smoke_checks_{B,M}.json`):

* Fold: new fit `2a21cb8771f6…`, new dev `270ab4126b0f…` (match the frozen hashes); sizes
  8000/2000, disjoint, cover 10000, `new_dev ∩ old_dev = ∅`, `new_dev ⊂ old_fit`,
  `|new_fit ∩ old_fit| = 6000`; 1 shared canonical group (1 fit row, 1 dev row).
* Label lineage: reconstructed train-only raw fields reproduce the historical saved constants
  (`sigma_logP` Δ `6.7e-16`, others 0.0) and old `k`/`c` exactly on the old fit mask; the mixed
  12k verification CSV is never opened; dev raw-field shuffle leaves the new constants bit-identical.
* New constants (new fit only): `sigma_logP=1.434428173759835`, `sigma_SA=0.8327498022638992`,
  `mu_SA=-3.1924626828735625`, `sigma_cycle=0.2885551506645267`, `mu_cycle=-1.3447189184664423e-05`;
  dev k groups `1935/56/7/2`.
* Prep: all four `Std.fit` arrays equal the direct new-fit recomputation (max abs `0.0`); the
  all-train blob constants are inversion-only.
* Payload: phi scaler and 8192-root kappa sample are new-fit-only; `kappa_D=2.0631041526794434`,
  `kappa_M=2.00007850651312`; structure arrays match the env cache (`node_sizes`, `root_atom`).
* Init/stream: 36 shared tensors byte-identical (`max Δ 0.0`); `A_raw = D_loc_init.T`;
  `W_loc ≡ 0`; post-construction RNG equal; both training RNG starts equal; schedule hash
  `7b11a529…` frozen.
* M smoke: `W_loc` task grad nonzero at step 1; `A_raw` grad 0 at step 1 and nonzero after the
  first `W_loc` update; tuple path executes (6325 tuples / 128 molecules); label permutation and
  save/reload forward exact.

## 2. Formal trajectories (two arms, `res-2`, Slurm, pool `res2-cu124`, run in parallel)

| field | B | M |
|---|---|---|
| experiment / run_id | `fresh-fold-B` / `fresh-fold-B-20261004-212016-9e354e31` | `fresh-fold-M` / `fresh-fold-M-20261004-212054-6e1f1ee1` |
| Slurm job / node | `55941` / `c05` | `55942` / `c05` |
| GPU request | `gres/gpu:1`, 4 cpu, 32G | `gres/gpu:1`, 4 cpu, 32G |
| GPU | A100-PCIE-40GB (1x) | A100-PCIE-40GB (1x) |
| driver / torch / CUDA / python | `525.85.12` / `2.5.1+cu124` / 12.4 / 3.12.14 | same |
| commit | `2e4ee3f70b14` (clean) | `2e4ee3f70b14` (clean) |
| start / end (wall) | 21:18:17 / 21:29:08 (623.3 s) | 21:18:56 / 21:30:06 (642.4 s) |
| steps | 15,120 / 15,120, exit 0 | 15,120 / 15,120, exit 0 |
| schedule / position stream / gid stream | `7b11a529…` / `7b11a529…` / `69187f13fba5def8…` | same / same / same |
| training RNG start | `1ccf17250133dec5…` | same |
| calibration bias (fit median) | `−0.02291946699922634` | `+0.013943578191542905` |
| replay (CPU vs GPU, 128 fit rows) | `2.15e-06` | `2.15e-06` |

**GPU allocation evidence.** Both jobs ran concurrently on c05, so the per-process
`CUDA_VISIBLE_DEVICES=0` in each run meta is device numbering inside the allocation, not evidence
of the same physical card. `scontrol show node c05` during both runs (probe run
`fresh-fold-alloc`, Slurm job `55943`, captured in `gpu_allocation_probe.txt`) shows
`Gres=gpu:4`, `CfgTRES=...,gres/gpu=4` and `AllocTRES=cpu=16,gres/gpu=4` while both jobs held
`TRES=...,gres/gpu=1` / `TresPerNode=gres:gpu:1`: Slurm had allocated all four physical GPUs
(two to our jobs, two to other users), i.e. two distinct physical GPUs for the paired arms.
`rr jobs` at round end: `fresh-fold-B`, `fresh-fold-M`, `fresh-fold-alloc` all completed; nothing
running or pending.

## 3. Local analysis (CPU)

`--analyze` on the pulled prediction npz files: two fit-median biases, raw/cal MAE, per-k group
table and contribution identities, 1000-draw paired bootstrap (seed `20261005`, shared endpoint
indices), bootstrap witnesses, pre-fixed drop-worst-row sensitivity, mechanism zero-ablation,
soup reload replay. Frozen classification fired **`DIRECTIONAL_NOT_CONFIRMED`**, mechanism
`HEALTHY`, replay `all_ok=true` (max abs vs stored `1.4e-6`/`9.5e-7`).

## 4. Incidents and post-run changes

* No formal-training failure, no recovery, no failed GPU attempt.
* One **analysis-only** follow-up code change after the formal runs: added `per_graph_fit.csv`
  emission in `analyze` (`git diff 2e4ee3f70b14` is confined to the `analyze` function, 21 added
  lines; `build_fresh_fold`, targets, payload, prep, `train_arm`, `run_smoke`, `operator_checks`
  unchanged). It does not alter any reported number or model state.
* Smoke states were discarded; no smoke checkpoint entered either formal init or stream.

## 5. Budget

* GPU: B 623.3 s + M 642.4 s = 1265.7 s = **0.352 GPU-h** (limit 1.0); the allocation probe used
  the CPU pool (0 GPU). Peak GPU concurrency 2.
* Wall at analysis end: ~65 min; compute (all GPU work) stopped at 21:30, ~51 min into the round.
* CPU: all local children `OMP_NUM_THREADS=8` / `torch.set_num_threads(8 or 4)`.
* Smoke: 2 fit-only local CPU smokes (3 steps each); states discarded.

## 6. Artifacts

Under `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/`, hashes in
`manifest.json`: PROTOCOL/METHOD_CONTRACT/EVIDENCE_SCOPE/ERRATA/REPORT/DECISION/EXECUTION;
`fresh_{fold,targets,tuple_payload,prep}` + meta + `fresh_manifest.json` + `input_manifest.json`;
`init_identity.json`, `operator_path_checks.json`, `smoke_checks_{B,M}.json`,
`gpu_allocation_probe.txt`, `fresh-fold-{B,M}_run_meta.json`; per-arm states/curve/meta/raw
predictions; `analysis.json`, `bootstrap.json`, `gains.json`, `gate.json`, `main_table.csv`,
`group_table.csv`, `per_graph_{fit,dev}.csv`, `mechanism_health.json`, `replay_checks.json`,
`budget.json`. Old results were read-only by hash/stable id; no old report or state was edited.
