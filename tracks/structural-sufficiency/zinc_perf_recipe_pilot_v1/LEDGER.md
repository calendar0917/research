# ZPR-v1 — run ledger

All formal work ran on `res` (2×A100-40GB, process backend), one GPU per job.
GPU-hours are charged as job wall-clock × 1 GPU and include container/import overhead.

## Jobs

| job | run id | pool | commit | wall | GPU·h | exit |
|---|---|---|---|---|---|---|
| smoke | `zpr-smoke-20261029-154512-eedd6d0e` | res-gpu0 | `1f7ada7` | 00:00:32 | 0.0089 | 0 |
| G0 L1 control (s0) | `zpr-G0-L1-s0-20261029-154950-d697a6c0` | res-gpu0 | `1f7ada7` | 00:22:14 | 0.3706 | 0 |
| G1 SmoothL1 (s0) | `zpr-G1-SL1-s0-20261029-155056-f3a41992` | res-gpu1 | `1f7ada7` | 00:27:44 | 0.4622 | 0 |
| divergence control | `zpr-divergence-20261029-194505-2f24c33c` | res-gpu0 | `22ccbb6` | 00:01:18 | 0.0217 | 0 |
| terminal test read | `zpr-test-read-20261029-195516-41378554` | res-gpu0 | `fe5d60f` | 00:00:15 | 0.0042 | 0 |
| infra: clear untracked `analysis/smoke.json` | `zpr-fix-untracked-20261029-194259-756d4a30` | res-local | `1f7ada7` | 00:00:08 | 0 | 0 |
| infra: restore `analysis/decision.json` | `zpr-fix-decision-20261029-…` | res-local | `fe5d60f` | 00:00:08 | 0 | 0 |

Charged total: **0.868 A100·h** of the 1.5 cap (trainer-measured 0.368 + 0.459,
plus the controls). No job was cancelled, re-run, or wasted on a failure other
than the two CPU-only deploy-hygiene jobs above (both 8 s, 0 GPU).

Note on the two infra jobs: the first `rr deploy` after the formal runs was
blocked by an untracked `analysis/smoke.json` in the remote checkout (Git refuses
to overwrite untracked working-tree files), and the test-read script had edited
the tracked `analysis/decision.json` on the remote. Both were resolved without
touching a GPU and without changing any scientific artifact; the code was then
changed so remote jobs write only under the Git-ignored `runs/` tree.

## Checkpoint identities

| checkpoint | sha256 | valid MAE (replay) | params |
|---|---|---|---|
| `runs/G0_s0/best.pt` | `f75ad5e946d0c9ca…` | 0.0792349 | 237,519 |
| `runs/G0_s0/ema_best.pt` | `c3f7c15d3afba605…` | 0.0796640 | 237,519 |
| `runs/G1_s0/best.pt` | `347a65f13e8d509f…` | 0.0822147 | 237,519 |
| `runs/G1_s0/ema_best.pt` | `57f4f190ad0cc0dc…` | 0.0824656 | 237,519 |

The four valid MAEs were recomputed from the checkpoint files before the test
labels were read (`analysis/test_read.json` → `checkpoints[*].abs_diff` < 1e-6).

## Order of operations (provenance)

1. protocol frozen (`PROTOCOL.md`) → 2. CPU gates (`analysis/cpu_checks.json`,
   36 unit tests) → 3. GPU smoke (`analysis/smoke.json`) → 4. two formal arms
   (`1f7ada7`) → 5. decision frozen on valid only (`analysis/decision.json`,
   `fe5d60f`) → 6. divergence control for the C4 restatement → 7. **one**
   terminal test read (`fe5d60f`) → 8. report.

Nothing after step 5 changed a checkpoint, a hyper-parameter, an epoch, or the
verdict.
