# PROTOCOL — SSRC Research Tree v1 (CMA / SPC)

Status: **frozen before any GPU run** (created 2026-10-30, `git_commit` recorded in
`analysis/frozen_baseline.json`). Round directory:
`tracks/structural-sufficiency/zinc_compositional_research_tree_v1/`.

This file is the pre-registration. The scientific design is given by the task
(`SSRC Research Tree v1`); this document only records (a) the fixed
implementation-level resolutions needed to make the design executable, (b) the
frozen evaluation definitions, and (c) the stage/budget/stop rules. Nothing here
may be changed after the first formal run; engineering fixes are recorded in
`TREE_LEDGER.md` with the affected runs.

## 1. Fixed shared skeleton (task §1)

```
a_v      = atom_encoder(raw108)                                  108 -> 128 -> 96
b_e      = bond_encoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bond7])    295 -> 160 -> 96
g_base   = [sum a, sum a^2, sum b, sum b^2, global33,
            log1p(n_atoms), log1p(n_bonds)]                      419
y_base   = rho_base(g_base)                          419 -> 128 -> 64 -> 1  (SiLU)
y_hat    = y_base + sum_anchor rho_event(anchor_state, event, event^2, log1p(count))
rho_event: 225 -> 128 -> 1, SiLU hidden, **output layer zero-initialised**
```

* `atom_encoder` / `bond_encoder` are initialised from the frozen train-only SSL
  checkpoints `bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s{seed}.pt`
  (`sha256` in `analysis/gates/cpu_gates.json :: g10_data_boundary`), and are
  fine-tuned end-to-end.
* No P3 / LCRC / DCC composer, reader or supervised weights are inherited.
  Nothing recursive: no learned atom/bond hidden-state message passing.
* `global33` is standardised with **Fit-row-only** mean/std (the frozen
  historical preprocessing; recomputed from all 10k train rows for Stage IV).
  `sum a, sum a^2, sum b, sum b^2` and the two `log1p` counts are used raw.
* The shared modules are constructed first under a seeded
  `torch.manual_seed(seed)` RNG, so **the shared parameters are bit-identical
  across families at the same seed** (stronger than the “matched within family”
  requirement). No dropout anywhere.
* Consequence of the zero-initialised event output: at step 0 both arms of a
  family predict exactly `rho_base(g_base)` and are numerically identical.
  Convergence of the event path is checked by the CPU gradient gate.

Deployment parameter counts (from `analysis/gates/parameter_table.json`):

| family | rho_base | rho_event | atom enc | bond enc | family modules | total |
|---|---:|---:|---:|---:|---:|---:|
| CMA | 62,081 | 29,057 | 26,336 | 62,816 | F2 51,216 + F3 30,496 | **262,002** |
| SPC | 62,081 | 29,057 | 26,336 | 62,816 | Fport 32,688 + Fpair 29,344 | **242,322** |

Both ≤ 500,000.

## 2. CMA (task §2) — implementation resolutions

* **Directed enumeration.** For every physical bond `e={j,k}` *both* centre
  orientations `(j,k)` and `(k,j)` are enumerated, and for each orientation all
  `i in N(j)\{k}`, `l in N(k)\{j}` with `i != l`. This yields exactly
  `2 * n_e` directed records per bond with **no index-based direction choice**.
  `n_e` (undirected path count) is `directed_count / 2`, which is exact.
* `type(u,w) ∈ {0 no edge, 1/2/3 bond type + 1}`; the chord types stored per
  record are `(type(i,k), type(j,l), type(i,l))`; all three are used.
* `z_ijk = 0.5 [F2(i,j,k) + F2(k,j,i)]`, `F2: 484 -> 96 -> 48`. `z` is indexed by
  `(centre, min(a,b), max(a,b))`, so reversal maps `zL ↔ zR` and leaves
  `r` and the centre bond `b_jk` unchanged (verified numerically in gate 2).
* `u = [zL+zR, |zL-zR|, zL*zR, b_jk, r]`, `r = [oh_ik+oh_jl, |oh_ik-oh_jl|, oh_il]`
  (12 dims), `F3: 252 -> 96 -> 64`.
* `m_e = sum_oriented F3(u) / (2 sqrt(1+n_e))`; `n_e = 0` ⇒ `m_e ≡ 0` exactly.
* **Events are every physical bond**, including `n_e = 0` bonds:
  `rho_event([b_e, m_e, m_e^2, log1p(n_e)])`.
* **CMA-MEAN (the only matched NULL).** Means are taken over all *directed*
  records of the same centre bond: `mean(zL)`, `mean(zR)`, `mean(r)`, then one
  `F3` call on `[mzL+mzR, |mzL-mzR|, mzL*mzR, b_e, mr]`, scaled by
  `n_e / sqrt(1+n_e)` — the exact algebraic counterpart of REAL's
  `(2 n_e · mean) / (2 sqrt(1+n_e))`.
  **Honest statement of what the NULL loses:** because the enumeration is
  reversal-symmetric, `mean(zL) = mean(zR)` *exactly*, so the `|zL-zR|` block of
  the NULL input is identically zero. REAL vs MEAN therefore differ in (i) the
  per-path joint correspondence between the left and right fragments and
  (ii) the nonlinearity of `F3` (Jensen gap). This is not a same-information
  linear control. It sees identical fragment multisets, identical centre bonds,
  identical counts and identical induced-chord marginals (`mean(r)`), and it
  receives the same `rho_event` parameters.
* No third CMA arm.

## 3. SPC (task §3) — implementation resolutions

* Articulation vertices from a deterministic Tarjan DFS over sorted adjacency;
  the gate compares the result against `networkx.articulation_points` and the
  components against `networkx.connected_components(G - v)`.
* For cut vertex `v`, the components `C_vj` of `G - v` are ordered by their
  smallest node (an arbitrary but fixed order; gate 4 verifies that permuting
  branch rows does not change any prediction).
* One branch = one component `C_vj`. Branch statistics:
  * `n_atoms_branch = |C_vj|`, atom content sum over `u ∈ C_vj`;
  * `n_bonds_branch = #internal bonds of C_vj + #bonds from C_vj to v`,
    bond content sum over exactly those bonds;
  * cyclomatic rank of the induced subgraph on `C_vj ∪ {v}` = `E' - V' + 1`.
  With the task's own dims this is `96 + 96 + 2 + 1 = 195`; `S_vj` (195) is
  concatenated with `a_v` (96) → `Fport: 291 -> 96 -> 48`.
* `q_ij = [p_i+p_j, |p_i-p_j|, p_i*p_j, a_v]`, `Fpair: 240 -> 96 -> 64`,
  `t_v = sum_{i<j} Fpair(q_ij) / sqrt(1+n_pairs)`,
  `n_pairs = k_v (k_v-1)/2`.
* Event: `rho_event([a_v, t_v, t_v^2, log1p(n_pairs)])`, summed over cut
  vertices. A molecule without cut vertices contributes **exactly zero**
  (`rho_event` is not called at all), so it is predicted from `rho_base` only.
* **SPC-MEAN (the only matched NULL):** `pbar_v = mean_j port_vj`,
  `t_v_null = n_pairs * Fpair([2 pbar_v, 0, pbar_v^2, a_v]) / sqrt(1+n_pairs)`.
  Same parameters, same branch summaries, same cut vertices, same counts.
* No extra structure blocks, no attention, no recurrent state, no learned
  routing.

## 4. Data boundary

* Official PyG ZINC `subset=True`: **train 10k labels only** for Stages I–III.
  Stage IV additionally uses the official **valid 1k** for selection.
  `ad_test.pt` is never opened; `xs_common.load_cache` refuses the test split and
  gate 10 statically checks that no module outside the boundary module mentions
  it.
* Internal development split: the frozen `bjoc_predictive_bottleneck_v1`
  split A rows (Fit8k 8000 / Monitor 1000 / Locked-dev 1000), verified against
  `split_a.json` row hashes on every load, plus the frozen label hashes.
* **Disclosure**: the reused train-only SSL encoders were pre-trained on the
  *complete* official train10k structures (unlabelled). Internal held-out
  evaluation (Monitor / Locked) is therefore **not fully inductive-blind in the
  structural-pretraining sense**. This is reported in `REPORT.md`.
* Locked labels are read exactly once, after checkpoints are frozen, and only
  if the Stage III Monitor conditions are met.
* No prediction path may read a label-derived quantity (logP / SA / cycle
  penalty). Gate 9 whitelists the forward keys and shows that mutating the label
  vector does not change any prediction.

## 5. Frozen supervised recipe (task §5)

raw-`y` L1, AdamW, peak LR 1e-3 → min 1e-5 per-update cosine, 500-update
warmup, weight decay 1e-5, gradient clip 5, batch 128, **47,400 optimizer
updates**, no dropout, no EMA/soup/distillation, full Monitor every epoch, strict
minimum Monitor MAE checkpoint (first minimum wins). Fit8k and train10k use the
same 47,400 updates. Batch order from `torch.Generator().manual_seed(seed+101)`;
both arms of a pair share it. CUDA non-determinism is allowed and recorded
(GPU model / driver / torch in `train.json`).

## 6. Frozen evaluation definitions

Reported on the best-Monitor checkpoint (and `final`), on the Monitor rows:

| name | definition |
|---|---|
| `full` | all 1000 Monitor rows |
| `ordinary` | `y > -3` (outside the artificial cycle penalty); n=926 |
| `tail` | `y <= -3`; n=74 |
| `any_unseen` | ≥1 atom whose exact typed radius-2 environment is absent from Fit8k; n=317 |
| `unseen_ordinary` | `any_unseen & ordinary`; n=271 |
| `seen_ordinary` | `~any_unseen & ordinary`; n=655 |
| `minus_row172` | drop the row at **fixed position 172** of the frozen Monitor list (row 1656; identity by row number, never by `y`); n=999 |
| `minus_fixed_worst10` | drop the 10 Monitor rows with the largest \|error\| of the frozen DCC-L Fit8k seed-0 best-Monitor checkpoint (fixed before any run of this round); n=990 |
| curve | `best`, best-20-epoch rolling median, tail-100-epoch median, `final` |
| `eval_train_mae` | MAE on the Fit8k rows in `eval()` mode |

Historical same-split reference (same machine/driver/torch regime):
**DCC-L Fit8k Monitor MAE 0.123151 (s0) / 0.127772 (s1)**, recomputed from the
frozen prediction files as `0.1231509 / 0.1277716`.

## 7. Stages, order, budget, stop rules

| Stage | content | cap (GPU·h) |
|---|---|---:|
| 0 | CPU gates 1–12 + structure caches | 0 |
| I | CMA-REAL / CMA-MEAN seed 0 (2-epoch timing first) | 4.0 |
| II | SPC-REAL / SPC-MEAN seed 0 (only if CPU feasibility passed) | 4.0 |
| III | seed 1 pair of the promoted family | 3.0 |
| IV | full official train10k REAL/NULL seed 0 pair | 3.0 |
| reserve | failure re-runs / condition checks | 2.0 |
| **total** | | **16.0 hard, 14.0 soft** |

* Stop rule (task §4): SPC stops at CPU with
  `INSUFFICIENT_SEPARATOR_COVERAGE` if <60% of Fit8k molecules have a cut
  vertex. **Measured: 100.0%** → proceed.
* Cost rule (task §6): measure 2 epochs (126 updates), extrapolate linearly to
  47,400 updates; if the projected pair exceeds 4.0 GPU·h, do not start formal
  training and record `CMA_COST_EXCEEDS_BRANCH_CAP` (resp. `SPC_...`).
* Early abort allowed only for hard failures (NaN, non-finite gradients,
  persistent no-gradient, impossible-to-finish). A lagging early valid point is
  not a reason to change anything.
* Stage III promotion (all must hold, seed 0, Monitor):
  1. `REAL − NULL` full ≤ **−0.004**
  2. `REAL − NULL` ordinary ≤ **0**
  3. `REAL − NULL` unseen-ordinary ≤ **−0.004**
  4. `REAL` full ≤ seed-matched DCC-L Monitor + **0.010**
  5. the 20-epoch rolling-median result has the same sign as `best`
  Then freeze both checkpoints and read Locked1k **once**; if REAL is not better
  than NULL on Locked, the family does not advance.
* If both families pass, advance only the lower-Monitor family; ties (<0.004)
  broken by the larger unseen-ordinary matched gain, then CMA.
* `MECHANISM_ONLY_SIGNAL` if the matched full gain ≥ 0.010 with consistent
  ordinary / unseen-ordinary / Locked direction but the absolute level is not
  competitive; no full-10k purchase.
* Stage IV only if both Stage III seeds agree in sign and full **and**
  unseen-ordinary improve repeatably. Findings that must be reported: REAL vs its
  own NULL; REAL vs mature historical single models; breadth across ordinary /
  unseen-ordinary / no-manual-penalty groups; dependence on the penalty tail,
  row172 or the fixed worst-10.
* No unplanned seeds, widths, losses, schedules or architectures. Unspent budget
  is not reallocated.

## 8. Deliverables

`REPORT.md`, `DECISION.md`, `TREE_LEDGER.md`, `analysis/main_table.csv`,
`analysis/slices.csv`, `analysis/budget.json`, `analysis/contrasts.json`,
`analysis/frozen_baseline.json`, `analysis/gates/*` (CPU gates, parameter table,
coverage), plus every run's `train.json` (git SHA, checkpoint sha256, start/end
time, GPU hours, regime).
