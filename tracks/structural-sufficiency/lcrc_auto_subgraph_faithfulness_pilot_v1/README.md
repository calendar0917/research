# lcrc-auto-subgraph-faithfulness-pilot-v1

Automatic connected-subgraph explanation of the **frozen LCRC-C1** on two official
small OGB molecular tasks, verified by the frozen model's own forward response.

The question is not "is this a nice picture" but:

1. can the model's *own* sensitivity find a connected atom–bond subgraph with **no**
   pre-specified functional group or SMARTS pattern;
2. does re-running the frozen computation with those bonds' explicit structural
   evidence masked actually change the prediction more than a same-size,
   same-path-count random control;
3. is that response carried by S1, S2, their interaction, or the nonlinear
   full-graph readout `rho`;
4. and is the gradient-based discovery at least directionally consistent with the
   real masked response.

Full rules: [`PROTOCOL.md`](PROTOCOL.md). Results: [`REPORT.md`](REPORT.md),
verdict: [`DECISION.md`](DECISION.md).

## What is reused (read-only) and what is new

| reuse (never modified) | use |
|---|---|
| `lcrc_readout_context_pilot_small_v1/code` | the frozen arm-A module, the checkpoint loader, the split caches, the target scaler |
| `lcrc_readout_context_pilot_small_v1/results/<dataset>/A` | `checkpoint_best_val.pt`, `trajectory.json`, `best_predictions.npz` |
| `lcrc_cross_task_preparation_v1/code` | `molecular_graph.enumerate_s1` / `enumerate_s2`, `lcrc_data.collate` |
| `lcrc_lipo_model_dissection_v1/code` | the frozen `LocalContent('C1')` column layout |
| `tracks/structural-query-learning/{code,data}` | the label-free graph caches |

New in this round: `code/as_*.py` (gating, occurrence→bond mapping, gradient
discovery, beam search, matched-random controls, interventions, aggregation,
figures, report) and `tests/test_as_round.py`.

**No arm-A training happens here.** The 360-epoch trajectory of the source round
is verified (checkpoint/trajectory/prediction SHA-256, 360/360 three-split
evaluations, recomputed minimum-Val epoch) and reused.

## Reproducing it

Local (CPU) correctness tests — these load the frozen caches and checkpoint, so
they are *not* fresh-clone-safe:

```bash
uv run pytest -q tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/tests
```

Formal runs, one worker per GPU through the control plane:

```bash
# preflight (identity + ENGINEERING_GATE, no explanation)
bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
  --purpose "preflight" --set stage=preflight --set dataset=all

# 8-molecule engineering smoke per dataset
bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
  --purpose "smoke" --set stage=smoke --set dataset=ogbg-molesol

# formal workers
rr run res lcrc-as-molesol --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/results \
  -- bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
rr run res lcrc-as-mollipo --pool res-gpu1 \
  --result tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/results \
  -- bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
     --purpose "formal: ogbg-mollipo" --set stage=dataset_worker --set dataset=ogbg-mollipo

rr jobs
rr status lcrc-as-molesol
rr logs lcrc-as-mollipo --stderr
rr pull lcrc-as-molesol
rr pull lcrc-as-mollipo
```

Then the CPU-only aggregate stage. It recomputes every table, bootstrap interval,
figure and verdict **from the pulled per-molecule records** and writes
`REPORT.md` / `DECISION.md` / `run_manifest.json`; it never touches a GPU and
never changes a weight:

```bash
bash tracks/structural-sufficiency/lcrc_auto_subgraph_faithfulness_pilot_v1/run_remote.sh \
  --purpose "aggregate" --set stage=aggregate
```

`run_remote.sh` exports the control-plane opt-in for you (`RESEARCH_TRACK_ROOT`,
`RESEARCH_EXTRA_RUNNER_PATHS`, `RESEARCH_EXTRA_RUNNERS=as_runner`), so the same
script works locally and on `res`.

## Artefacts

```
analysis/preflight.json                 data/cache/checkpoint identity + every engineering gate
analysis/training_source.json           what was reused, 360-epoch coverage, selected epoch, hashes
analysis/protocol_freeze.json           frozen constants + freeze_hash + code hashes
analysis/main_table.csv                 per dataset x split main effects
analysis/architecture_diagnostics.csv   rho_share / cancellation / S1-S2 / coverage
analysis/molecule_table.csv             per-molecule detail (replayable)
analysis/bootstrap.json                 molecule-level bootstrap CIs
analysis/failure_cases.json             high-gradient / high-local-output but low-response cases
analysis/decision.json                  frozen verdict computation
analysis/figures/                       highlighted molecules (8 Valid + 8 Test + 4 diagnostics per dataset)
results/<dataset>/samples.json          selected official row ids + sha256 per split
results/<dataset>/preflight.json        worker-side gate record
results/<dataset>/<split>/molecules.json  per-molecule discovery, interventions, controls
results/<dataset>/candidates.npz        the complete candidate search log
results/<dataset>/budget.json           GPU seconds actually spent
```

## Rules this round keeps

* one reused checkpoint, one seed; no retraining, no best-of-N, no second seed
* no new explanation network, no extra message passing / attention / readout
* the intervention is `CONDITIONAL_S1_S2_EVIDENCE_MASK`, never a chemical bond or
  group deletion, and never evidence that "LCRC did not learn structure"
* the search space is connected 2–6 bond subgraphs only; nothing is claimed beyond it
* Test is `DEVELOPMENT_TEST`: evaluated and reported, never used to select anything
* frozen sibling tracks are read-only

## Engineering notes

* RDKit's drawing backend on this machine needs `libXrender`/`libXext`/`libexpat`
  that are not on the default loader path; `as_figures._draw2d()` resolves them
  from the local nix store on demand and records every preloaded path in
  `analysis/figures/index.json`. If the backend cannot be resolved, the affected
  molecules are recorded as `rendered: false` with the reason instead of failing
  the aggregate stage — the numerical results never depend on a figure.
* The figure stage renders a bounded set (8 Valid + 8 Test + 4 diagnostics +
  2 distribution panels per dataset). The several hundred raw molecules are not
  committed and can be regenerated from `results/<dataset>/`.
