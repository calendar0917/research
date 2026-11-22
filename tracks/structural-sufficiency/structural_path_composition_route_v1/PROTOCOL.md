# PSC: structural path composition route — protocol

Round: `structural_path_composition_route_v1`
Version: 2026-11-22/v1 (frozen before formal training)
Task: ogbg-mollipo whole-molecule lipophilicity regression. Primary metric:
whole-molecule RMSE in original label units. `Δ = metric(reference) − metric(candidate)`,
positive = candidate better.

This protocol was written before any formal PSC/ACRA/GINE run of this round.
Later additions are append-only sections at the end (timing, deviations).

## 1. Question and scope

Can structural scores shared along concrete connection paths transfer the
regularities of seen atoms/bonds and local combinations to unseen path words,
unseen exact substitutions and unseen molecules, while improving whole-molecule
prediction?

The round answers Q1–Q5 of the brief with a complete 2×2×2 PSC factorial
(input handling × readout × supervision target), the ACRA-v2 reference with and
without the same pair term, a raw-feature GINE anchor, four CPU baselines, a
second scaffold split and a single frozen official-Valid review.

Out of scope (prohibited by the brief): GNN/MPNN/Graph-Transformer/attention
inside PSC, pretrained molecular models, learnable fragment-id embeddings,
independent pair heads, Test access, Valid-based tuning.

## 2. Data and label isolation

| item | value |
|---|---|
| raw structure cache | `tracks/structural-query-learning/data/ogbg_mollipo_sq-pilot-cache-v1.npz`, sha256 `fe002287…c7c` |
| Train labels | extracted once from `attachment_conditioned_response_algebra_pilot_v1/results/ACRA_REL_S0/predictions_best_train.npz`, members `row_id` + `label` only → `data/labels_official_train.npz` |
| Valid labels | `…/predictions_best.npz`, members `row_id` + `label`, opened **once** by `psc_final_eval.py` after `analysis/final_eval_lock.json` |
| structure members read | whitelist `node_ptr, bond_ptr, atom_feat, bond_feat, bond_u, bond_v, degree, min_ring, in_cycle, graph_n_*, smiles, row_ids, train_idx, valid_idx` |
| Test | never used: no structure frequency, vocabulary, scaler, training, prediction or evaluation |

The training object only exposes finite labels on the 3360 official Train rows;
Valid labels are `NaN` there and every statistic asserts membership in the fit
rows. Matched pairs come from the frozen TRC cache (label-free members only).

## 3. Splits

* **F0/D0** — the existing frozen scaffold-disjoint Train-sub 2856 / Dev 504
  (`split_train_sub_dev.json`, file sha256 `675289f6…e845`, seed 0).
* **F1/D1** — second scaffold split of official Train built at the start of this
  round with the same algorithm, seed **20261122**, target 504 held out
  (`data/split_train_sub_dev_d1.json`). Bemis–Murcko `includeChirality=False`,
  acyclic molecules keyed by their own SMILES, sorted keys → `default_rng(seed)
  .permutation` → greedy whole-group assignment. No label is read. This is a
  **split-robustness replication**, not a fresh blind test.
* **FULL** — all 3360 official Train rows; official Valid scored once at the end.

Statistics (label scaler, `c_l`, category support, path vocabulary, pair graph,
kernel hyper-parameters) are recomputed from the current fit rows only.

## 4. Path objects and cache

`L = 5` maximum path length. Every simple (vertex-distinct) undirected path of
length `l = 0..L` is enumerated; a length-0 path is one atom; the reverse of a
geometric path is the same path and is enumerated once (canonical orientation
`min(vertex tuple, reversed)`); ring paths are all enumerated, not replaced by a
shortest path. The stored tokens are the raw 9-column atom rows and 3-column
bond rows; no recomputed degree/hydrogen/slice statistic is added.

The cache key is the lexicographic minimum of the full raw token sequence
(atoms block then bonds block) and its reverse. It is a **cache key only**: it is
not a learnable id, and no frequency statistic is attached to it.

Cache (`data/psc_path_cache_lipo.npz`, schema `psc-path-cache-v1`) stores per
molecule the unique canonical keys with their within-molecule multiplicity, the
true geometric path counts per length, and the padded ORDER/BAG slot tensors for
each global key. Computed values (encoder outputs) are never cached. A cache is
only reused when schema, `L`, raw sha256, preprocessing code sha256 and row set
all match. The cache covers official Train + Valid **structure** only; Test rows
are not in it.

*Batch rule*: entries are grouped per molecule, deduplicated by the ORDER
canonical key inside the batch, encoded once, and scattered by
`multiplicity × c_l`. BAG changes only the input tensor; different ORDER keys
are never merged after sorting.

`c_l = 1 / max(1, mean_{G in fit} |P_l(G)|)` — one constant per length, a
property of the whole fit set.

## 5. Model

Shared encoder `E`: input → 256 → 128 → 64 (SiLU + Dropout 0.1 after the first
two linear layers, SiLU after the last). Each atom categorical field has a 32-d
embedding; the nine are summed and divided by √9. Each bond field has a 16-d
embedding; the three are summed and divided by √3. Padded slots stay exactly
zero. Input dimension 289 for `L=5` (`6·32 + 5·16 + 11 masks + 6 length
one-hot`). Raw categories unseen in the fit rows are mapped to a zeroed
embedding row.

Shared scalar head `h`: `Linear(64,64,bias=False) → SiLU →
Linear(64,1,bias=False)`. The only intercept is `beta`, initialised at 0; the
last `h` weight is initialised with std 1e-3 so the initial prediction is the
standardised label mean and the encoder already receives gradient. No
BatchNorm and no graph-size-dependent normalisation.

```
SUM : F(G) = beta + Σ_l c_l Σ_p 0.5 [h(E(x_p)) + h(E(rev_p))]
POOL: z(G) =      Σ_l c_l Σ_p 0.5 [ E(x_p)  +  E(rev_p) ];  F = beta + h(z)
```

`rev` reverses only the valid atom/bond slots. SUM applies `h` before the sum,
POOL after it; parameters, shapes and initialisation are identical. BAG sorts
the 9-field atom rows and the 3-field bond rows independently inside the same
path, keeping length, masks and multiplicity; therefore `BAG(rev) = BAG` and
the second direction is not recomputed (exact, not an approximation).

All arms of one seed share the identical encoder/head/embedding initialisation
(single explicit init function, saved `shared_init` hashes).

## 6. Supervision

`e_i = F(G_i) − y_i_std`.

```
MOLECULE : L = mean_{main batch} e_i^2
PAIR     : L = L_mol + 0.25 · Σ_(a,b) w_ab (e_b − e_a)^2
DIAG     : L = L_mol + 0.25 · Σ_(a,b) w_ab (e_a^2 + e_b^2) = L_mol + 0.25·Σ_i d_i e_i^2
```

Unique unordered matched pairs with both endpoints in the current fit rows;
`w_ab = 1/(K·m_c)` over connected components of the unique-pair graph
(`Σ w = 1`). Minibatch: uniform shuffle, main batch 64 molecules, 32 extra pairs
per step sampled uniformly over components then uniformly over that component's
unique edges (with replacement). One forward per unique molecule of
`main ∪ endpoints`; `L_mol` is averaged on the main batch only. PAIR and DIAG use
the same sampler seed/sequence and the same main-batch order. Same-molecule
predictions are shared inside a step.

Optimiser: AdamW, peak LR 1e-3, weight decay 1e-4, clip 5, 5 warmup epochs, then
cosine to 1e-5, **160 epochs**. No extra regularisers, no support gate, no path
contribution L2.

ACRA keeps the parent objective and regularisers; `ACRA_PAIR` adds only
`0.25·L_pair`. The parent site/relation terms are evaluated on the molecules and
events of the uniform main minibatch only (verified numerically against the
parent loss with no auxiliary endpoints).

GINE uses the frozen native raw-feature recipe (5 layers, hidden 128, batch 128,
weight decay 1e-5, 160 epochs) and is **not** parameter/FLOP matched. Its seeds
are set before the factory call.

## 7. Arm matrix and selection

Stage 1, seed 0 then seeds 1–2 for all eleven arms:

| id | input | readout | target |
|---|---|---|---|
| PSC_SUM | ORDER | SUM | MOLECULE |
| PSC_SUM_PAIR | ORDER | SUM | PAIR |
| PSC_SUM_BAG | BAG | SUM | MOLECULE |
| PSC_SUM_BAG_PAIR | BAG | SUM | PAIR |
| PSC_POOL | ORDER | POOL | MOLECULE |
| PSC_POOL_PAIR | ORDER | POOL | PAIR |
| PSC_POOL_BAG | BAG | POOL | MOLECULE |
| PSC_POOL_BAG_PAIR | BAG | POOL | PAIR |
| ACRA_REF | — | ACRA parent | parent |
| ACRA_PAIR | — | ACRA parent | parent + PAIR |
| GINE_REF | graph | GINE | MOLECULE |

Checkpoint selection: best whole-molecule RMSE on the held-out side of D0
(whole molecules only; never pair Δ, never a novelty bucket, never MAE).

Champion lock: over the eight PSC arms take `r_min` of the three-seed mean D0
RMSE, form `S = {m : mean ≤ r_min + 0.005}`, and pick inside `S` by priority
SUM → MOLECULE → BAG → arm id. `ACRA_B*` uses the same rule over
`ACRA_REF`/`ACRA_PAIR` with tie priority REF. `O*` flips only ORDER↔BAG,
`R*` flips only SUM↔POOL, `M*` is `O*` when the champion is SUM and `R*` when it
is POOL. If the champion uses PAIR, a DIAG twin (same input/readout) is trained
for seeds 0/1/2 before D1 starts; DIAG never re-selects the champion.

D1: for `C*, O*, R*, ACRA_B*, GINE_REF`, seeds 0/1, each trained from scratch on
F1 statistics for `E_m = median D0 best epoch` epochs with the original 160-epoch
cosine horizon (stops early; horizon is not redefined). D1 metrics are computed
after training only, from the final epoch.

FULL: `C*, ACRA_B*, GINE_REF`, seeds 0/1/2, `E_m` epochs, all statistics refit
on 3360 rows, `state_final` only. CPU baselines refit with D0-selected
hyper-parameters. `M*` full-Train is added only if budget remains and only if
decided before Valid labels are opened.

## 8. Evaluation

Primary: whole-molecule RMSE (original units) on D0, D1 and the frozen official
Valid; secondary: MAE, seed SD, parameter count, inference time, device hours.
Uncertainty: paired molecule bootstrap (2000 reps, seed 0) where the same
molecule resampling index is applied to every model and every seed; scaffold
cluster bootstrap for the three main comparisons only.

Matched-pair analysis: three scopes (fit×fit, fit×heldout, heldout×heldout) on
unique unordered pairs, direction fixed by row_id order, Δ=0 direction items
removed (count reported, tolerance 1e-8), plus a pre-fixed |Δ| ≥ 0.25
sensitivity table. The old cut-row-weighted table is kept as an appendix. The
novelty attribution table keeps cut rows but renormalises each unique pair's
cut-row weights to 1. Pair bootstrap resamples connected components of the full
pair graph (built on fit ∪ heldout), never individual rows.

Support classes per held-out pair: T seen; T unseen with both fragments seen;
T unseen with exactly one fragment unseen; T unseen with both fragments unseen,
where the T identity is the fixed lexicographic order of the two rooted-fragment
canonical keys plus the cut-bond attribute id, and "fragment seen" means the
exact rooted fragment occurs in any legal cut event of the fit molecules.

Path-novelty `eta(a,b)` = unscaled ORDER path-count mass on keys absent from the
fit vocabulary, divided by total absolute count change; full length and lengths
3..L; buckets `{0, (0,0.25], (0.25,0.5], (0.5,1]`; N/A when the denominator is 0.
For BAG arms the actual BAG-input novelty is reported separately, plus the raw
categorical UNK rate per bucket.

SUM ledger: in standardised space, `F(Gb) − F(Ga) = Σ_l c_l Σ_t (n_t(b) − n_t(a))
· s_θ(t)`; multiply by the fit label SD for original units. The bridge
decomposition `F(G) = beta + F0(S) + F0(C) + X(S,C;b)` is claimed only for real
bridge pairs whose context fragment is raw-identical; non-bridge graphs are
trained normally without that decomposition.

## 9. Engineering checks (must pass before formal training)

`code/psc_checks.py` writes `analysis/checks.json`:

1. length-0 count = atoms, length-1 = bonds, enumeration equals an independent
   directed-DFS enumeration (all molecules including rings);
2. canonical keys invariant under random atom renumbering;
3. reversal touches only valid slots, padding stays exactly zero, batch
   prediction equals single-molecule prediction, explicit occurrence loop equals
   the multiplicity ledger;
4. ORDER distinguishes a sequence from its reverse; BAG is exactly invariant to
   member permutation;
5. SUM difference ledger and bridge partition (recomputed per molecule);
6. `PAIR = DIAG − 2·cross`, unique-pair dedupe, component sampler unbiasedness,
   `Σw = 1`, no held-out endpoint in the training pair graph;
7. scaler / category support / vocabulary only from fit rows (mutating held-out
   labels does not move them; held-out-only categories are masked);
8. identical shared init across arms of a seed, finite forward/backward, no
   zero-gradient parameter, 32-molecule overfit smoke;
9. short jobs on both `res-gpu0` and `res-gpu1` with real device/version/timing
   and a working `rr pull` path.

## 10. Budget and degradation

Hard deadline T0+8h; ≤ 16 GPU-hours total (`wall × allocated GPUs`), CPU time
reported separately. One job per GPU, no DDP, no stacking. Degradation order is
fixed in the brief §11: cancel extensions and M* full-Train; then D1 seed 1/2;
then the predefined REDUCED_MATRIX (all eleven seed-0 arms first, then three
seeds only for the SUM-family and POOL-family pick plus ACRA_REF, ACRA_PAIR and
GINE). No per-arm epoch shortening; any change of `L`, input or horizon must
happen before formal numbers and applies uniformly.

Fixed extension eligibility (only time/resource triggered, never by results):
D1 seed 2; M* full-Train seeds 0/1/2 if decided before Valid labels are opened;
ESOL structure-transfer stress test; data-size curve. All must finish before
T0+6:30 with 45 min of wrap-up reserved.

## 11. Deliverables

`PROTOCOL.md`, `REPORT.md`, `code/`, `configs/`, `data/` (split + label-source
records; large caches are regenerable), `results/main_table.csv`,
`results/summary.json`, per-run `manifest.json`/`curves.csv`/checkpoints,
`analysis/{comparison,support_novelty,unique_pair_metrics,
pair_residual_decomposition,sum_edit_ledger,selection_lock,final_eval_lock,
checks,compute_budget}` and at most four core figures. Every planned arm —
including failures, cancellations and budget skips — appears in the main table.