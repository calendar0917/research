# Deviations log

- 2026-10-23 (pre-formal): added `prep --manifest` so the remote cache build can
  write its manifest under the git-ignored `cache/` instead of modifying the
  tracked `analysis/prep.json`. No model, data contract, recipe, gate or budget
  changed. This is plumbing for a clean remote worktree.
