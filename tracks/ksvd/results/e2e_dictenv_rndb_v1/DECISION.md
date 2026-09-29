# e2e_dictenv_rndb_v1 — decision

**Case F — RNDB_NO_GO_TASK_LEVEL_MECHANISM_STRONGLY_SUPPORTED**

- `M_R = 0.133117` (RNDB_NO_USEFUL_TASK_GAIN)
- `G_hist = -0.003089` vs historical CSSD-q1 `0.130028` (unmatched)
- `G_psi = 0.190679` (gate 0.003) — psi strongly load-bearing
- `G_dict = 0.288827` (gate 0.010) — dictionary strongly load-bearing
- seed 1 authorized for a *future* round: `False` (not executed this round)
- durable mechanism finding: `True` (future-round hypothesis only: `True`)
- no post-hoc rescue, no matched baseline rerun, official test never loaded

> Boundary outcome. The literal Case F condition is `M_R > 0.126` AND `no strong new mechanism evidence`; the mechanism evidence here is very strong (G_psi=0.190679 >= 0.003, G_dict=0.288827 >= 0.010, psi alive), so the conjunction does not hold. Case C would require `M_R <= 0.126`. No pre-registered case enumerates `M_R > 0.126` WITH strong new mechanism evidence. The dominant condition (`M_R > 0.126`, band P3, no useful task gain) forces the task-level no-go; the mechanism result is recorded rather than folded into a case. This boundary was not enumerated in the pre-registered table.
