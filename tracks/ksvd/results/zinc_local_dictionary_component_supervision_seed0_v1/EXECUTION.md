# EXECUTION — zinc_local_dictionary_component_supervision_seed0_v1

## Timeline (wall clock, ~90 min total)

* Round start (luyin19 + sources read, ~25 min exploration).
* `22:05` local: build-objects (fold/phi/kappa/prep refit, 7.1 s), init-check
  (all pass), smoke run 1/2 (4 steps/arm, CPU, discarded).
* `22:14` first training pair submitted (jobs 56046/56047, commit af6368f).
* `22:23-22:26` both first-pair jobs failed at the post-training JSON write
  (delivery-layer bug; all epochs + state saves complete; no dev score existed).
  One-line fix (write_json list serialization), committed d1fe445, redeployed.
* `22:30` formal pair re-submitted (jobs 56048/56049) — 62 min mark, inside the
  70-minute new-training window.
* `22:40/22:42` M_COMP (690.1 s) and D_COMP (826.7 s) completed, exit 0.
* `22:44` artifacts pulled; eval roster frozen and committed BEFORE any dev
  prediction (83ad447).
* `22:46` first dev-eval crashed pre-metric (missing argument); fixed
  deterministically; dev-eval + mechanism-health completed (~22:50).

## Formal runs (the comparison pair)

| arm | wall_s | steps | GPU UUID | job | node | driver | torch/cuda/python |
|---|---|---|---|---|---|---|---|
| D_COMP | 826.7 | 15,120 | GPU-ba05543b-868e-8605-9dec-a8e4f2a82bc3 | 56048 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |
| M_COMP | 690.1 | 15,120 | GPU-afab53a3-3153-cf22-9c18-43d754ad0fac | 56049 | c05 | 525.85.12 | 2.5.1+cu124 / 3.12.14 |

Both ran FP32, no AMP/DDP, same commit d1fe445d8c2a, same node/driver/regime, in
parallel on two separate one-GPU Slurm allocations (distinct UUIDs; both showing
CUDA_VISIBLE_DEVICES=0 is the in-allocation index). Position schedule + actual
global gid streams identical across arms (verified in eval_roster.json).

## Budget

* GPU: 4 jobs ≈ 0.78 allocation-GPU-hours (failed fragments 56046/56047 ≈ 0.35 h
  + formal 0.230 + 0.192 h) ≤ 1.0 h. Failed fragments counted and disclosed.
* CPU: 8 threads local (prep/build/eval), 4 threads/arm remote.
* Wall: ~90 min (≤120 target; no new training after the 70-min mark; all
  compute finished ~85 min).

## Recovery / restarts

* Jobs 56046/56047: both arms completed all 15,120 epochs and saved
  init/last/soup/checkpoints/fit-predictions, then crashed writing the curve
  JSON (`dict(list)` in write_json). No dev score existed or was read; data and
  recipe unchanged; one from-scratch engineering restart per arm (the allowed
  budget) was used on the fixed commit; discarded fragments did not select
  models (states were overwritten by the identical deterministic re-runs).
* Dev-eval attempt 1: crash before any metric was computed/printed/written
  (raw dev predictions saved only); deterministic one-argument fix; re-run
  after the roster freeze. No training, calibration or selection occurred after
  any dev score.

## Science vs execution status

* Science: COMPLETED — two formal seed-0 trajectories, matched recipe, fold,
  data and labels; no dev-driven configuration selection; roster frozen before
  any dev prediction.
* Execution: CLEAN after two disclosed engineering failures (delivery-layer
  JSON bug; analysis-layer missing argument), each fixed deterministically,
  each before any dev score was seen; both formal runs completed inside budget.

## Artifacts

results/zinc_local_dictionary_component_supervision_seed0_v1/: PROTOCOL.md,
METHOD_CONTRACT.md, REPORT.md, DECISION.md, EXECUTION.md, new_objects_manifest.json
(+ new_fold/new_targets/new_tuple_payload/new_prep npz, new_kappa.json),
init_identity.json, smoke_runs.json, eval_roster.json, dev_eval.json,
mechanism_health.json, per-arm init/last/raw_soup/epoch{1,40,120,240} states,
curves, probe logs, metas, fit/dev predictions (sums + components), and the
run module tracks/ksvd/experiments/luyin16/
zinc_local_dictionary_component_supervision_seed0_v1.py.
