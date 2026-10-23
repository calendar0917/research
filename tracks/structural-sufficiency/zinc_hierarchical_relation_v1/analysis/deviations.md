# Deviations log

- 2026-10-23 (pre-formal): added `prep --manifest` so the remote cache build can
  write its manifest under the git-ignored `cache/` instead of modifying the
  tracked `analysis/prep.json`. No model, data contract, recipe, gate or budget
  changed. This is plumbing for a clean remote worktree.
- 2026-10-23 (pre-formal): GPU smoke failed twice at
  `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))` with
  `RuntimeError: Invalid device argument` before any CUDA context existed.
  Added an explicit `torch.cuda.init()` before the reset in train/profile/smoke
  (three call sites). No model, data, recipe, gate or budget changed; the two
  failed smoke attempts cost ~20 s wall each and are charged to the budget.
- 2026-10-23 (pre-formal): after the CUDA init fix, GPU smoke (1024 train
  molecules, 30 steps, both arms) completed at ~0.054 s/step and 0.23/0.31 GB
  peak; GPU profile (128-molecule batch) gave A 0.032 s / B 0.037 s on gpu0.
  Two failed smoke attempts plus prep/smoke/profile consumed well under
  0.1 A100-h; the 4-hour round budget is intact.
- 2026-10-23 (post-run, analysis only): `score` stored a relative `run_dir`
  when invoked with `--run ../runs/...`, so `compare` resolved it outside the
  repo and failed. `cmd_score` now resolves the run directory and `cmp_compare`
  resolves both paths robustly. No training, prediction, metric or gate value
  changed; scores are recomputed from the same frozen `valid_best.npz`.
