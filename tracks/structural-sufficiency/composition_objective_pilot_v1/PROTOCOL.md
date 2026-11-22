# Composition Objective Pilot v1 — protocol

Protocol id: `composition-objective-pilot-v1`
Round dir: `tracks/structural-sufficiency/composition_objective_pilot_v1/`
Dataset: `ogbg-mollipo` (official scaffold split), seed `0`.
Status: **frozen before any formal training run**. Append-only deviations go to §12.

This round asks one engineering/optimisation question, not a new architecture
question:

> Does replacing the current cut-wise main loss by a graph-level main loss
> (`f_bar` vs `y`) change whole-molecule Valid performance, cut dispersion,
> and real matched-pair replacement prediction, when every other component of
> the parent and CSOI pilots is held fixed?

No new architecture, no extra arm, no hyperparameter search, no additional seed,
no additional dataset, and no official Test label access are allowed.

---

## 1. Read-only inputs

The following are reused **read-only** and are never modified:

- `tracks/structural-sufficiency/task_response_composition_s0_v1/` — parent
  round: cut enumeration, parse table, attribute tables, scaffold split,
  train-only scaler, train-only pair directory, `PhiEncoder`, `ResponseModel`
  (`FACTOR` / `ADD` / `GLOBAL`), optimizer/schedule, GINE reference.
- `tracks/structural-sufficiency/contextual_structural_object_interaction_pilot_v1/`
  — CSOI round: `FrameEncoder`, token view, cross-attention, cut response head,
  `CSOIModel`, interface recipe, and the existing B/C checkpoints.
- The prepared structural cache `csoi_cache_lipo.npz` (or the parent's
  `trc_cache_lipo.npz`, which has the same content digest). Identity is checked
  by the existing content digest
  `46439b91dfd760255be4c53256f12687959d203b6740e55c0213d66476168825`;
  every run records the digest it actually loaded.

No parent/CSOI file, checkpoint, result or protocol is written to. The new
round has its own `code/`, `tests/`, `analysis/`, `results/` and `configs/`.

---

## 2. Question 1 — audit of the current objective

The audit is CPU-only and is completed before any training.

For a molecule `G` with parses `c = 1..n_G`, cut predictions `f_c`,
target `y` (standardised units), and `f_bar = (1/n_G) Σ_c f_c`:

```text
L_cut   = (1/n_G) Σ_c (f_c - y)^2
L_graph = (f_bar - y)^2
Var_cut = (1/n_G) Σ_c (f_c - f_bar)^2
```

The algebraic identity `L_cut = L_graph + Var_cut` is verified numerically on
both Train and Valid, per molecule, from the **best existing checkpoints**
(A `TRC_FACTOR_CUTRESP`, B `CSOI_ADD`, C `CSOI_GLOBAL`; D `CSOI_REL` is audited
as an extra existing checkpoint only). The audit records:

- mean / median / p90 / p99 of `Var_cut` and `L_graph`;
- per-molecule cut count and cut-prediction range;
- the fraction `Var_cut / L_cut` and `L_graph / L_cut` (the direct measure of
  how much the cut-wise objective is actually penalising dispersion);
- association between `Var_cut` and whole-graph error `|f_bar - y|`
  (Pearson and Spearman), overall and by split;
- the same quantities stratified by S size and C size;
- Train versus Valid differences.

The audit also establishes, by source inspection and by executed arithmetic,
which delta the existing `pair_metrics.json` used:

- **whole-molecule delta**: `ŷ(G_b) - ŷ(G_a)`, where `ŷ` is the molecule-level
  prediction `f_bar` (plus scaler);
- **direct cut delta**: `f(C, S_b) - f(C, S_a)` at the same context and cut
  bond, i.e. the quantity the `CUTRESP` auxiliary is trained on.

The two are never mixed in tables or labels. The old `pair_metrics.json` is
shown to be the **whole-molecule** variant.

---

## 3. Question 2 — three fixed training arms (seed 0)

Only the main loss changes. Everything else is inherited verbatim from the
corresponding existing family implementation.

| ID | model family | main loss |
|---|---|---|
| `F_GRAPH` | parent `ResponseModel` arm `FACTOR` (same as `TRC_FACTOR_CUTRESP`) | graph-level |
| `B_GRAPH` | CSOI `CSOIModel` arm `CSOI_ADD` | graph-level |
| `C_GRAPH` | CSOI `CSOIModel` arm `CSOI_GLOBAL` | graph-level |

New main loss, for the first `n_main` molecules of the batch:

```python
loss_main = mean((f_bar[:n_main] - y[:n_main]) ** 2)
loss = loss_main + 0.25 * loss_cutresp_aux
```

The auxiliary term is unchanged: it is the same train-only `CUTRESP` pair
sampler, the same `pair_cut_delta` implementation, the same `tau2` floor and the
same weight `0.25`. No auxiliary loss is added, removed, reweighted or
recomputed.

Frozen and unchanged:

- data, official scaffold split, target scaler, pair directory;
- cut/parse enumeration, rooted-object/frame construction;
- `FrameEncoder`, token view, attention, relation branch (where present);
- cut response head, additive branch, graph aggregation `f_bar`;
- `CUTRESP` auxiliary supervision;
- AdamW, peak LR `1e-3`, weight decay `1e-4`, gradient clip `5.0`;
- 5-epoch linear warmup, cosine decay to `1e-5`;
- batch `64` molecules + `16` auxiliary pairs;
- deterministic per-module initialisation (seed 0);
- best-by-Valid-whole-molecule-RMSE checkpoint selection;
- horizon **160 epochs**, seed `0`;
- no Test evaluation.

The recipe is the shared parent/CSOI recipe. If any inherited default differs
from the brief, the frozen parent/CSOI value wins and the difference is listed
in §12. No new recipe is searched.

Per epoch, each run records at least:

- Train/Valid molecule RMSE and MAE (original label units);
- Train/Valid cut-level MSE (`mean_c (f_c - y)^2`) and graph-level MSE
  (`mean_G (f_bar - y)^2`), in standardised and original units;
- Train/Valid mean `Var_cut`, cut span, and cancellation share
  `Var_cut / (L_graph + Var_cut)` per molecule;
- auxiliary pair loss (training batches);
- total gradient norm and per-module-group gradient norms before clipping;
- epoch wall time.

The interaction branch is checked after training: gradient norms, the
interaction contribution `f - f_additive` (CSOI) / bilinear `q·r` term (parent),
and the fraction of molecules with positive/negative cut predictions. A rise in
cut variance is **not** read as improved representation.

---

## 4. Question 3 — real matched replacement evaluation

The existing train-only real pair directory is reused. No synthetic chemistry
edit is constructed. Pairs touching official Test are dropped without reading
their labels.

Two deltas are evaluated and reported under distinct names:

A. **whole-molecule delta**: `ŷ(G_b) - ŷ(G_a)` (best checkpoint `f_bar`).
B. **direct cut delta**: `f(C, S_b) - f(C, S_a)` at the shared C and cut bond
   (best checkpoint cut head / pair head).

Scopes: Train×Train, Train×Valid, Valid×Valid. Metrics: δMAE, δRMSE, direction
accuracy, Pearson, bias, and valid pair count.

- Valid×Valid has ≈15 pairs and is descriptive only.
- Train×Valid pairs are not independent. Uncertainty is described by a
  **cluster bootstrap over the Valid molecule** (or over pair components where
  no Valid molecule is present), not by an i.i.d. pair bootstrap.
- Train×Train is in-sample for every arm and is labelled as such.

---

## 5. Question 4 — local context support (CPU only)

For every real bridge parse (`parse_kind == 0`, `C` non-empty), let `c0` be the
C-side attachment atom. On the original attributed molecular graph, construct
the C-side context at:

- `radius 1`, `radius 2`, `radius 3`: atoms of `C` within graph distance
  `≤ r` from `c0`, induced subgraph;
- `complete C`: the full `C = V(G) \ S` object used by the model.

Each object gets a canonical, atom-numbering-independent identity using the
parent's colour-encoded graph encoding (RDKit canonical SMILES over attribute
colours plus a port marker at `c0`) and, as the primary key, the raw cut-bond
3-tuple appended exactly as the model's context key does. A structure-only
variant without the cut bond is also reported.

For each radius, Train/Valid report:

- number of distinct Train contexts and valid contexts;
- Valid context seen rate (parse-weighted and molecule-weighted);
- distribution of context size (atoms), and number of distinct contexts;
- cross-molecule reuse: number of distinct Train molecules / parses containing
  each seen Valid context (median, p90, fraction seen in ≥2 and ≥5 Train
  molecules);
- joint `(context, S type)` support: number of distinct Train pairs and Valid
  joint seen rate, where `S type` is the existing canonical S fragment key;
- the same quantities for `complete C` as baseline.

This analysis creates no model and trains nothing.

---

## 6. Evaluation and interpretation

Primary metric: **Lipo Valid whole-molecule RMSE** at the best-by-Valid epoch.

Reused coordinates (read-only): A `TRC_FACTOR_CUTRESP` 0.7096, B `CSOI_ADD`
0.7196, C `CSOI_GLOBAL` 0.7346, E `GINE_RAW_REF` 0.6406 (the exact values are
re-read from the parent/CSOI prediction files, not typed from this document).

Pre-registered paired comparisons, paired bootstrap over the 420 Valid
molecules (2000 resamples, seed 0):

1. A vs `F_GRAPH`
2. B vs `B_GRAPH`
3. C vs `C_GRAPH`
4. `B_GRAPH` vs `C_GRAPH`
5. `F_GRAPH` vs `C_GRAPH`
6. `C_GRAPH` vs `GINE_RAW_REF`

Positive delta means the second arm is better. Single seed: the interval covers
only Valid-molecule sampling, **not** training-seed randomness. An interval that
crosses zero is reported as "not resolved", not as "no effect".

An improvement of about **0.02 RMSE** is treated as an exploratory signal worth
following up, not as a formal confirmation threshold. The report also checks:

- whether only Train error improved;
- whether cut variance increased;
- whether whole-molecule and direct-cut pair deltas moved together;
- whether the absolute gap to the GINE reference narrowed.

---

## 7. Execution and budget

- ≤ **2 A100 GPU-hours**, ≤ 2 GPUs in parallel, seed 0 only.
- CPU first: audit, local-context support, engineering tests, all analysis.
- Formal training via `rr deploy res` + `rr run res` with pools
  `res-gpu0` / `res-gpu1`. No manual SSH, `sbatch` or `nohup`.
- Every run writes its own `results/<run_id>/` with `manifest.json`,
  `curves.csv`, `predictions_best.npz`, `predictions_final.npz`,
  `state_best.pt`, `state_final.pt`, `train.log`.
- If hardware is unavailable, local CUDA may be used only with an explicit
  execution-regime note; remote formal runs are the default.
- A run may be restarted from scratch at most once for an infrastructure
  failure, same commit and same recipe.

---

## 8. Test and safety

- Official Test labels are never read. Training views expose Train + Valid only.
- Old Test metrics are never used for model selection or any decision.
- Old experiments, checkpoints and results are not overwritten.
- No ESOL, no PCBA, no new model, no extra arm, no multi-seed.
- Existing user Git changes are not reset.

---

## 9. Deliverables

`PROTOCOL.md` (this file), `REPORT.md`, code and tests, three trained arms with
checkpoints, and:

- `results/main_table.csv`
- `results/summary.json`
- `analysis/loss_decomposition.json`
- `analysis/local_context_support.json`
- `analysis/pair_delta_comparison.json`
- training curves and execution provenance

The final `REPORT.md` begins with the five required questions and gives
evidence plus residual uncertainty for each.

---

## 10. Engineering gate (must pass before formal training)

`tests/` and `code/cpo_checks.py` must establish:

1. exact re-load of A/B/C best checkpoints and reproduction of their stored
   Valid predictions within a documented CPU/GPU tolerance;
2. the per-molecule identity `L_cut = L_graph + Var_cut` on real data;
3. the new graph-loss expression equals an independent manual computation;
4. gradient flow into the correct module groups for all three models;
5. pair-direct delta and whole-molecule delta are computed by different code
   paths and are labelled separately;
6. a CPU smoke run of each family completes finite steps, writes a checkpoint,
   and leaves model/eval semantics unchanged;
7. no Test array is loaded by the training data view.

---

## 11. Frozen recipe values (inherited)

AdamW, peak LR `1e-3`, min LR `1e-5`, weight decay `1e-4`, clip `5.0`,
5 warmup epochs, cosine schedule, batch 64 + 16, `CUTRESP` weight `0.25`,
`tau2` floor `0.25`, pair pool weights 0.8/0.2, sampler seed 0, eval batch
32 (CSOI) / 64 (parent), seed 0, horizon 160.
`dropout_fragment = 0.1`, `dropout_global = 0.1` (parent defaults).

---

## 12. Deviations log (append only)

- **2026-11-22 — post-run diagnostic fix (no training effect).** The in-run
  `mechanism_diagnostics.json` for `F_GRAPH` failed because the frozen-weight
  diagnostic assigned a plain callable zeroing shim to `model.q` / `model.r`;
  PyTorch requires an `nn.Module` there. Training, the loss, checkpoint
  selection, and all predictions were unaffected. The shim was made an
  `nn.Module` in `code/cpo_train.py`, and the diagnostic was recomputed for all
  six old/new arms at their frozen best checkpoints in
  `analysis/mechanism_diagnostics.json`.
- **2026-11-22 — remote smoke run.** A 2-step `cpo-remote-smoke` run (16 Train /
  8 Valid molecules, `res-gpu0`) was executed to verify the remote environment
  before the formal runs. It is not an arm, its outputs are not in `results/`,
  and its time is not charged to the 1.006 A100 GPU-hour training total.
- **2026-11-22 — no formal recipe deviations.** All three formal arms used seed
  0, 160 epochs, the frozen parent/CSOI recipe values, and best-by-Valid
  selection. Test labels and Test prediction arrays were never read.