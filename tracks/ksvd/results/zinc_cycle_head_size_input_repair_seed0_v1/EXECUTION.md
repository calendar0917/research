# EXECUTION — zinc-cycle-head-size-input-repair-seed0-v1

## Environment

* Local CPU only, `torch.set_num_threads(8)`, no CUDA (`torch.cuda.is_available()`
  not used), no remote job (`rr` not invoked).
* `uv run` environment: torch 2.5.1+cu124 (CPU), numpy 2.1.3, networkx present.
* Branch: `task/zinc-cycle-head-size-input-repair-seed0-v1` (isolated, no
  push/merge).  Old result dirs and checkpoints byte-unchanged.

## Commands actually run

```bash
# from the repository root
uv run python -m tracks.ksvd.experiments.luyin16.zinc_cycle_head_size_input_repair_seed0_v1 --stage A
# -> Phase A + N/E graph audit + structure gate; passed=True (reduction 0.003902)

uv run python -m tracks.ksvd.experiments.luyin16.zinc_cycle_head_size_input_repair_seed0_v1 --stage all
# -> Phase A again (identical) + Phase B (Q0/QNE training, evaluation, wrapper)
# -> final_class=INPUT_REPAIRED_NOT_FIT, ~43-48 s

uv run python tracks/ksvd/results/zinc_cycle_head_size_input_repair_seed0_v1/analyze.py
# -> descriptive.json: identity, movement, leave-one-row-out sensitivity

uv run python tracks/ksvd/results/zinc_cycle_head_size_input_repair_seed0_v1/make_manifest.py
# -> manifest.json over all committed result files
```

## Observed run facts

* `--stage all` prints: T25/fold/O/prep hashes all match; 10000-graph audit
  `self_loops=0 dup=0 rev=0 nx_bad=0 renumber_bad=0`; structure gate
  `passed=True old_classes=327 new_classes=1214`; h re-forward
  `fit_max=1.907e-06 dev_max=1.907e-06 fixed=9.537e-07 ok=True`;
  smoke `ok=True w_s_grad_step1=0.0`; `Q0 steps=18900` (~11-13 s);
  `QNE steps=18900` (~11-13 s, `w_s_final_norm=0.61666`);
  wrapper `ok=True cached_diff=9.537e-07 size_diff=0.0`;
  `final_class=INPUT_REPAIRED_NOT_FIT`.
* Engineering defect found and fixed during the round **before any conclusion**:
  the native-25 replay comparison subtracted a `(n,1)` Sequential output from a
  `(n,)` head output, broadcasting to `(n,n)`.  Fixed with `.view(-1)`; the
  corrected replay is `max|diff| = 0.0` for both fit and dev.  This did not
  touch training, the frozen anchors, or any reported gate.
* No dev-dependent decision was made before `structure_gate.json` was written;
  dev coverage and all head results were computed afterwards.

## Data boundaries

Only official-train objects were opened (`T25_all.npz`,
`O_seed0_raw_soup_state.pt`, `O_seed0_predictions.npz`,
`target_decomposition.npz`, `fold_objects.npz`, handoff `train.npz` with
`official_test_loaded=False`).  No official-valid/test file or old row cache
was loaded, predicted or scored.  No `load_split("train")` call and no 12k
cycle re-audit.

## Addendum — post-round merge (operator instruction)

The round was executed, audited and committed on its isolated branch under the
original "no push / no merge" constraint.  After the round closed, the
operator instructed "merge to main, then commit and push".  On 2026-10-04
11:04:12 CST `task/zinc-cycle-head-size-input-repair-seed0-v1` was merged into
`main` with `--no-ff` (merge commit `2ecbdf7`, amended once to drop a stray
`__pycache__` file) and pushed to `origin/main`.  No result file, model state,
prediction or metric changed; the research decision is unaffected (no
write-back into the model pipeline) — this is a code/record merge only.
`manifest.json` was refreshed after this addendum.
