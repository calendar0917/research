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
