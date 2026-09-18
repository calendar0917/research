# B-Null S/A/B path ablation (ZINC) — pre-registration

Protocol: `bnull_sab_path_ablation_v1`
Branch: `exp/bnull-sab-path-ablation`
Study: `zinc-context-gap`
Regime: deterministic A100, ZINC-12K official train/valid, **official test never
loaded**.

This file is committed **before** any Stage A evaluation or Stage B training.

## 0. Reference (read from the repo, not from memory)

* Frozen model: **B-Null seed0 Top-5 soup**, `results/local_token_null/`
  (`soup_states/lt_null_seed0_top5_soup.pt`).
* Recorded official-valid Top-5 soup MAE: `0.1230275` (`soup_lt_null_seed0.json`).
  Stage A recomputes it under the intervention harness and reports the residual.
* Parameters: **49,343** (local 16-D patch token constant zero, no local-token
  generator; `patch_cont` + parent typed context + pair relation + T=2 recurrent
  pair–centre + global branch + topology branch + small head all retained).
* Every delta below is `intervention − original`; positive = deleting the path
  makes valid MAE worse.

## 1. Scientific question

B-Null already proved that a molecule-dependent local 16-D identifier is not
required. Which of the **remaining** information paths and computation
mechanisms carry the performance? Families:

* **S — structure**: distance, overlap, boundary, path count, shell masses,
  size/cycle/degree scalars, topology channel.
* **A — attribute marginals**: per-patch / per-molecule atom- and bond-type
  composition.
* **B — structure–attribute binding**: which attribute sits at which structural
  role (shell / shell-pair / root / radius-1 parent / pair path / adjacent pair).
* **C — composition**: `P`, `Q`, pair→centre aggregation, `U`, T=2 feedback.

Every ablation measures the **incremental value of one path with the others
kept**. Nothing is a Shapley attribution; deltas are never summed.

## 2. Exact feature map (read off the real code)

Source of truth: `zinc_patch_path_pooling.py` constants and
`_shell_descriptor` / `_pair_relation`.

### 2.1 `patch_cont` (`SHELL_WIDTH = 146`, standardized on train)

| block | columns | family | content |
|---|---|---|---|
| atom_shell shell 0/1/2 | 0–27 / 28–55 / 56–83 | A / B | atom-type histogram per distance shell, normalized by patch node count |
| bond_shell (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) | 84–87, 88–91, 92–95, 96–99, 100–103, 104–107 | A / B | bond-type histogram per shell-pair, normalized by patch edge count |
| root_atom | 108–135 | B | centre atom one-hot (root role ↔ atom type) |
| incident_bonds | 136–139 | B | centre incident bond proportions (root role ↔ bond type) |
| scalars | 140–145 | S | log1p(n_nodes), log1p(n_edges), radius-boundary fraction, cycle_rank/n, centre degree/4, mean degree/4 |

Standardization is per-column train-fit `(x − mean)/scale`, `scale<1e-6 → 1`.

### 2.2 `pair_relation` (`RELATION_WIDTH = 23`, **not** standardized)

| block | columns | family | content |
|---|---|---|---|
| distance_onehot | 0–4 | S | `min(d,5)-1` one-hot |
| log_distance | 5 | S | `log1p(d)` |
| overlap | 6–10 | S | patch intersection / union / min / max, size diff |
| boundary | 11–13 | S | boundary overlap, centre containment |
| path_bond_mean | 14–17 | **B** | mean bond-type composition over shortest paths |
| log_path_count | 18 | S | `log1p(#shortest paths)` |
| adjacent_bond | 19–22 | S/B | adjacent bond-type one-hot for d=1 pairs; block mass is a pure-S adjacency indicator, the type is **B** |

### 2.3 Other inputs

* `global_context` 62-D, standardized, S + graph-level A marginals — **never
  touched** by any intervention.
* `topology_features` (hinge mode), pure-S — **never touched**.
* `parent_token` → `parent_embedding` (32 rows × 8-D): radius-1 typed joint
  context (parent-B).
* local 16-D patch token: exact zero in B-Null.
* Recurrent computation C: `P = pair_projection`, `Q = pair_encoder`,
  `gate = 1 + tanh(distance_gate(bucket))`, distance-conditioned pair→centre
  mean/std/log-count pool, `U = center_update`, T=2 weight-tied.

## 3. Stage A — frozen screen (eval only)

All interventions keep the output shape and reuse the frozen soup weights.
No retraining, no re-initialisation.

* **A1 parent-null.** Force `parent_embedding(parent_token)` to `zeros_like` its
  own output; everything else untouched. measures radius-1 typed parent joint
  context (parent-B).
* **A2 pair-B marginalized.** Pure-S pair columns are copied bit-identically.
  `path_bond_mean` is replaced by the molecule's graph-level bond-type marginal
  (exact, read off the adjacent-bond block: every edge is one distance-1 pair).
  `adjacent_bond` becomes `(adjacency mass) × graph bond marginal`, preserving
  the pure-S adjacency indicator while destroying the bond-type assignment.
* **A3 patch-B marginalized.** `atom_shell[s,a] → shell_mass[s]·patch_atom_marginal[a]`,
  `bond_shell[p,b] → shell_pair_mass[p]·patch_bond_marginal[b]`, `root_atom →
  patch_atom_marginal`, `incident_bonds → patch_bond_marginal`. Row sums (shell
  / shell-pair mass) and column sums (patch atom / bond marginal) are preserved
  exactly; the six pure-S scalars stay bit-identical. The **original** train-fit
  patch standardizer is reused so all preserved columns remain bit-identical
  (no refit-on-control confound). No new feature family is created.
* **A4 T=1 frozen recurrence.** Same weights, exactly one pair→centre round
  (`h0 → q0 → h1 → readout`), via the model's own `encode_original`.

Integrity requirements (all must pass): intervention changes the target field;
non-target fields bit-identical; permutation/endpoint symmetry unaffected;
no NaN; deterministic predictions; official test never loaded; before/after
tensor examples saved.

## 4. Stage A interpretation bands (mechanism language, not significance)

`Δ = intervention − original`:
`|Δ| < 0.003` NO_CLEAR_INCREMENTAL_SIGNAL; `+0.003…+0.010` small-to-moderate;
`+0.010…+0.030` material; `> +0.030` very important. Negative ⇒ candidate point
estimate better, no superiority claim from one seed.

## 5. Stage B — at most 3 retrained ZINC seed0 tickets

Default tickets, all on the identical B-Null backbone (49,343 params):

* **N1 NoParent** — `parent_embedding` weights zeroed and frozen, parent slot an
  exact zero; patch-encoder input shape and parameter count unchanged; nothing
  re-invested.
* **N2 Pair-B-Marginal** — exactly the A2 control representation from scratch
  (patch standardizer unchanged because `pair_relation` is raw).
* **N3 Patch-B-Marginal** — exactly the A3 transformed raw patch representation
  from scratch; the patch standardizer is refit on the marginalized train split
  through the canonical `_phase_data` path (the representation is the variable).

Protocol: seed0 only; Adam lr 1e-3, wd 1e-5, batch 128, max 240 epochs,
patience 40, L1, grad clip 5, fixed equal-weight Top-5 valid soup, official
valid only. Forbidden: width/optimizer/dropout/lr sweeps, extra features, radius
change, attention, parameter re-investment, seed1, official test.

### Pre-registered ticket-selection rule

Default = N1/N2/N3. **Exception**: if the frozen T=1 delta
`Δ_T1 ≥ 0.020` **and** at least one of the N1/N2/N3 frozen effects has
`|Δ| < 0.003`, replace the training ticket with the smallest frozen `|Δ|` by a
**T=1 retrain**. Total full-training tickets stay ≤ 3; no fourth run.

## 6. Stage B interpretation bands

`Δ = candidate soup MAE − B-Null soup MAE`: `|Δ|<0.003` no clear incremental
signal; `+0.003…+0.010` small-to-moderate; `+0.010…+0.030` material;
`>+0.030` very important. Single seed + possibly concurrent execution ⇒ tiny
effects are screens, not architecture claims.

## 7. Execution

GPU 1 only, `CUDA_VISIBLE_DEVICES=1`. GPU 0 is never used. At most two
independent candidate trainings share GPU 1, only after single-process GPU smoke
and a short non-formal concurrency throughput check; otherwise sequential. Each
process logs separately and all launched jobs are waited on.

## 8. Non-goals

No official ZINC test, no seed1, no 4th full run, no parameter re-investment,
no new architecture/feature/radius/RRWP/LapPE/attention, no head/optimizer
sweeps, no post-hoc new ablation from valid results.
