# lcrc-readout-context-pilot-small-v1

Small-dataset feasibility pilot for an **explicit context readout**: can a
budget-matched additive head that consumes two named global context fields
(size, ring-atom fraction) be trained on the existing C1 / LCRC local encoder on
`ogbg-molesol` and `ogbg-mollipo`, and can every prediction be reconstructed from
the terms it exports?

This is a *feasibility* round on two small tasks with **one** training seed.  It
is not a benchmark submission and makes no significance claim.

## Design basis

* commit: `9b3dba098915b0cdd800767f61b523681ee9e25d` (branch `main`)
* experiment branch: `lcrc-readout-context-pilot-small-v1`
* frozen numeric caches and the C1 / GINE code paths are **read-only reuse**:
  * `tracks/structural-query-learning/{code,data,results/sq-pilot-small-v1}`
  * `tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code`
  * `tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code`

The four caches this round consumes are verified against a frozen SHA-256 table
of their **real file bytes** at every stage, on every machine:

| file | rows |
| --- | --- |
| `ogbg_molesol_sq-pilot-cache-v1.npz` | 1128 (902 / 113 / 113) |
| `ogbg_mollipo_sq-pilot-cache-v1.npz` | 4200 (3360 / 420 / 420) |
| `ogbg_molesol_sq-pilot-lipo-raw-v1.npz` | 1128 |
| `ogbg_mollipo_sq-pilot-lipo-raw-v1.npz` | 4200 |

## The three arms

All three share **one** local trunk — bitwise the C1 local path, loaded from the
same canonical initial state — and differ only in the readout.

| arm | readout `P(H)` | form |
| --- | --- | --- |
| **A** | 15298 | `sum_v f(h_v) + rho([z_g, mean h, mean h^2, log1p n])` — the frozen C1 head, unchanged |
| **B** | `66H + 48M + 2` | `b_global + sum_v c0(h_v) + sum_j u_j(z_j)` over the `M` active global fields |
| **C** | `68H + 48M + 98` | B `+ a_size(z) * sum_v c_size(h_v) + a_ring(z) * sum_v c_ring(h_v)` |

`M` is not chosen: a raw global column is *active* when its Train span exceeds
`1e-8` (M = 34 for molesol, 36 for mollipo).  `H` is chosen so the nominal
readout parameter count lands within one hidden-width step of the 15298-parameter
C1 head (molesol: B 207 / 15296, C 200 / 15330; mollipo: B 206 / 15326,
C 198 / 15290).  Exact equality is impossible for integer `H`, and the frozen
`analysis/protocol_freeze.json` records the assignment.

Two context fields only, fixed in advance: `log1p_n_atoms` and
`ring_atom_fraction`.  Each new single-variable family is

```
q(z) = w2 . [SiLU(w1 z + b1) - SiLU(b1)]        # 48 params, no output bias
```

so `q` is a genuine single-variable function whose response cannot be confounded
with a change in `h`.

## Recipe (frozen, identical for A / B / C)

Seed 0; 360 epochs; AdamW peak `3e-4` -> min `1e-5` with a per-step cosine over
`360 * ceil(N_train/128)` steps; `wd 1e-3`; batch 128 (last partial batch kept);
MSE on the standardised target; grad-norm clip 5; FP32; no warmup / EMA / SWA /
soup / augmentation / resampling / distillation / frozen trunk.

Every epoch evaluates **Train, Val and Test in full**, in raw target units,
`test_policy=always`.  Checkpoint selection is **minimum Val RMSE only**
(strictly smaller replaces, ties keep the earlier epoch); epoch 0 is an
initialisation diagnostic and can never be selected.  Test is a
`DEVELOPMENT_TEST` column: it is reported next to the Val-selected state, never
used to choose one.

## What the round delivers

* `analysis/protocol_freeze.json` — every frozen constant + hashes at prepare time
* `analysis/dataset_audit.json` — split disjointness/cover, occurrence counts, row-id hashes
* `analysis/checks_<dataset>.json` — 13 preflight gates per dataset (see below)
* `results/<dataset>/<arm>/trajectory.json` — the full per-epoch curve, selection record, hashes
* `results/<dataset>/<arm>/checkpoint_best_val.pt` — the Val-selected state
* `analysis/tables/main_table.csv`, `contrasts.csv`, `bootstrap.csv`, `drop_one.csv`
* `analysis/ledgers/<dataset>_<arm>_{molecules,atoms}.csv|npz` — raw-unit account of every Val/Test molecule and root
* `analysis/{interventions,context_response,interaction_diagnostics,equivalence_scale}.json`
* `analysis/figures/*.png` — curves, learned context functions, three fixed display cases
* `analysis/verdicts.json`, `REPORT.md`, `DECISION.md`

## Preflight gates (all must pass before training)

Per dataset: data identity/occurrence audit; parameter budget and formula
consistency; **arm A bitwise equal to the frozen `DissectLCRC('C1')`**; bitwise
identical trunk initialisation across the three arms (also reproduced in a fresh
subprocess); exactly one dropout call per forward; readout zeroing returns the
trunk's prediction; gradients finite and reachable for every parameter;
invariance to eval batching, molecule isolation and atom renumbering; ledger
closure; exact `a_k x2 / c_k /2` scale equivalence; checkpoint reload equality;
and a sanity re-read of the frozen GINE reference.  Plus three
config-independent checks of the single-variable family.

## Running it

```bash
# one worker per GPU; each does verify -> checks -> A -> B -> C
rr run res lcrc-rc-molesol --pool res-gpu0 \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results \
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/analysis \
  -- bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
# ... same for mollipo on --pool res-gpu1
```

Then pull the two runs and run the CPU-only aggregate stage locally:

```bash
uv run research run lcrc-readout-context-pilot-small-v1 --study lcrc-readout-context-pilot-small-v1 \
  --mode screen --set stage=aggregate     # with RESEARCH_TRACK_ROOT/EXTRA_* as in run_remote.sh
```

## Rules this round keeps

* one training seed, no repeat for better numbers, no best-of-N
* no architecture / learning-rate / epoch search; one recipe for all arms
* no new loss, no distillation, no SSL, no extra labels, no MMP mining, no molecule generation
* no literature search; the readout forms were fixed in code before the runs
* frozen reuse is read-only: nothing under `tracks/structural-query-learning` or the two
  `lcrc_*` reuse rounds is modified
* two small tasks only; stop after them
