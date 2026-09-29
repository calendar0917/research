# Pre-registration — `e2e_dictenv_dictionary_coder_audit_v1`

Round: **Dictionary–Coder Disentanglement** on the closed ZINC
dictionary-environment line.  Protocol id:
`e2e_dictenv_dictionary_coder_audit_v1`.

**Frozen before any diagnostic is run.**  The file hash and the commit are
recorded in `results/e2e_dictenv_dictionary_coder_audit_v1/preregistration_snapshot.json`
and `preflight.json`.  No threshold, metric definition, split or gate rule in
this document may be edited after the first Stage-A process starts.

This round is **CPU-only** (`CUDA_VISIBLE_DEVICES=""`, `device = cpu`), never
uses SSH / the remote runner, and **never loads the official ZINC test split**
(`official_test_loaded = false` in every artifact).

---

## 0. Frozen context (not re-opened)

The previous round (`e2e_dictenv_clean_mechanism_v1`) froze

```text
FINAL-CLEAN = C6
              + paired node binding
              + paired edge binding
              + sparse tied-IHT (K=32, s=8, IHT-10)
              + full distance+overlap+boundary relation
```

with these closed findings: C6 safe across 3 CPU seeds; graph-level chemistry
histogram shortcut removed; node and edge per-occurrence correspondence kept;
distance-only relation rejected; anchor chemistry and first/second moments
retained.  Sparse dictionary specificity was **not** established
(`G_dict = DenseTied − Sparse = −0.002936`, matched soups `0.125563` vs
`0.128499`), and the sparse code is strongly concentrated
(27–28/32 atoms active, `N_eff ≈ 13.7–15.3`, top-5 atoms carry ≈4.4–5.0 of the
8 activations per row, one atom active in almost every row).

This round does **not** touch the architecture, the fusion, the relation, the
features or the training protocol.  It only **audits the existing
representations**.

## 1. Primary question

> Why is sparse dictionary specificity not established?  Is it (1) the IHT-10
> coder being under-converged, (2) the learned dictionary being intrinsically
> concentrated, (3) the sparse code being a near-linear re-parameterisation of
> the dense tied coordinate, or (4) sparsification genuinely losing
> downstream-useful information?

No predictor is trained unless the frozen gate in §8 fires.

## 2. Reused checkpoints (no new training before the gate)

All `D` tensors are read from the **soup** state dict key `"D"` of the closed
rounds:

| tag | path | soup valid MAE |
|---|---|---|
| C6 seed 0 | `results/e2e_dictenv_h1_clarity_audit/matched_cpu/C6_e320_soup_state.pt` | 0.12849851670576026 |
| C6 seed 1 | `results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed1_soup_state.pt` | 0.13050547152006767 |
| C6 seed 2 | `results/e2e_dictenv_clean_mechanism_v1/stage_a_cpu_baseline/C6_seed2_soup_state.pt` | 0.12226534780760995 |
| DenseTied seed 0 (secondary context) | `results/e2e_dictenv_clean_mechanism_v1/stage_f_dictionary_specificity/FINAL-CLEAN-DENSE-TIED_seed0_soup_state.pt` | 0.1255625270641758 |

The audit dictionary is **always** the column-normalised `Dbar`
(`tccd_v0.normalize_columns`, float64; `v0.normalized_dictionary` in torch),
i.e. exactly the operator the model uses (`reconstruct = coord @ Dbarᵀ`).
Raw column norms are recorded (they are not 1 after training in either
family) but never used directly.

Splits: official ZINC **train** (10 000 molecules, 231 664 nodes) and official
**valid** (1 000 molecules, 23 083 nodes) from the frozen cache
`results/e2e_dictenv_p1/cache/env_train.pt` / `env_valid.pt`.  The dictionary
input space is the **raw float32 phi65 stored in that cache** — there is no
scaler, no mean subtraction and no whitening between phi65 and the dictionary.
The mean / PCA / structural descriptors of §5–§6 are therefore computed in the
same space the dictionary receives.

## 3. Coder variants (Stage A, all on the same frozen `Dbar`)

| coder | definition |
|---|---|
| `iht10` | `v0.tied_iht_codes(Dbar, phi, s=8, steps=10)` (the trained operator) |
| `iht30` | same, `steps=30` |
| `iht100` | same, `steps=100` |
| `omp` | `tccd_v0.omp_codes(Dbar, phi, s=8)` — exact OMP reference, never a training candidate |
| `dense_tied` | `phi @ Dbar` — the frozen P1 DenseTied control definition |

The IHT variants run in float32 exactly as the model does; reconstruction
metrics are computed in float64.  OMP runs through the frozen `tccd_v0`
implementation.  **Every coder in a given seed uses one identical `Dbar`** —
separating dictionary effects from coder effects.

## 4. Stage A metrics (per seed × split × coder)

Reconstruction (with `EPS = v0.EPS`):

```text
recon_frobenius      = ||phi - A Dbar^T||_F / ||phi||_F
recon_mean_row_sq    = mean_v [ ||phi_v - a_v Dbar^T||_2^2 / (||phi_v||_2^2 + EPS) ]
row_l2               = ||phi_v - a_v Dbar^T||_2 / sqrt(||phi_v||_2^2 + EPS)   (mean/median/p90/p95)
```

Concentration of the activation frequency `f_j = #{v : j ∈ supp(α_v)} / N`:
`active_atoms`, `effective_atoms = exp(−Σ p_j log p_j)` with `p_j = f_j/Σf`,
`top1/top3/top5/top8_share`, `max_activation_rate`, `usage_entropy`, Gini of
`f` (`gini(x) = Σ_i Σ_j |x_i−x_j| / (2 n² mean(x))`; 0 when `mean(x)=0`).

Per-row coefficient geometry (`|α|`): L1, L2, `max|α|`, `top1/L1`, `top3/L1`,
each reported as mean/median/p90/p95.

Support / coefficient agreement against the OMP reference (per row):

```text
J(v)      = |S_coder(v) ∩ S_OMP(v)| / |S_coder(v) ∪ S_OMP(v)|     (mean/median/p10/p90)
exact     = fraction of rows with S_coder(v) == S_OMP(v)
top_s_intersection = |S_coder ∩ S_OMP|                             (mean/median/p10/p90)
cosine    = rowwise cosine of the full 32-D codes                  (mean/median)
pearson   = rowwise Pearson of the full 32-D codes                 (finite rows)
norm_l2   = ||α_coder − α_OMP||_2 / (||α_OMP||_2 + EPS)            (mean)
```

Stage-A classification labels (non-exclusive; a seed may carry more than one):

* `A1_CODER_SUPPORT_ARTIFACT`: IHT-30 vs IHT-10 improves the OMP support
  Jaccard by ≥ 0.05 **and** improves concentration (`N_eff` +1.0 or top-5 share
  −0.10).
* `A2_CODER_AMPLITUDE_BOTTLENECK`: reconstruction improves substantially
  (`recon_frobenius` ≤ 0.5×) while neither support nor concentration improves.
* `A3_INTRINSIC_CONCENTRATION`: OMP itself is concentrated relative to IHT-10
  (`N_eff_OMP ≤ N_eff_IHT10 + 1.0` and `top5_OMP ≥ top5_IHT10 − 0.10`).
* `NONE`: none of the above.

## 5. Stage B — dominant atom / common direction

For each seed and coder in {IHT-10, IHT-30, OMP}, on the **train** split:

* dominant atom `j* = argmax_j f_j`;
* mean direction `mu = (1/N) Σ phi_v` in the raw dictionary-input space, and
  `|cos(d_j, mu)|` for all 32 atoms (absolute value: atom/coefficient sign
  symmetry);
* train-only PCA (mean-centred SVD) of the dictionary inputs, `PC1..PC3`, the
  explained-variance ratios and `|cos(d_j, PC_k)|` for all atoms;
* Spearman correlation between `|α_{v,j*}|` and each frozen structural
  descriptor of §6 (all 32 descriptors are reported, not just the top ones).

## 6. Stage C — atom structural specialisation

`z(v)` is the fixed, named, topology-only descriptor vector recovered from the
verified phi65 provenance (`fsar_r2_ar0.phi_for_center` / `fsar_v2._explicit_basis_for_patch`).
No new chemistry, no target `y`:

```text
patch_nodes, patch_edges, root_induced_degree, root_neighbour_shell1/2,
root_walk1/2/3, shell_pop1/2, shell_frac1/2, mean/std log1p induced degree,
mean log1p neighbour-by-shell1/2, mean/std log1p walks,
edge_frac_shellpair_{00,01,02,11,12,22},
mean/std edge log1p{degree sum, |degree diff|, common neighbours}
```

(32 coordinates; exact index provenance in the module's
`STRUCTURAL_DESCRIPTORS` table.)

For coder in {IHT-10, OMP} (primary) and IHT-30 (auxiliary):

* activation population: `j ∈ supp(α_v)`;
* stricter top-response population: the top 10 % of the non-zero `|α_{vj}|` of
  that atom (no target);
* SMD profiles: mean/std are fit on **train only** and frozen for valid:

```text
SMD_{j,k} = ( E[z_k | j active] − E_train[z_k] ) / std_train(z_k)
Spec_j    = sqrt( (1/p) Σ_k SMD_{j,k}^2 )
```

Reported per atom: activation rate (train/valid), `Spec_j` (active and
top-response), top-5 absolute and top-5 signed SMD features, and the
train→valid profile cosine.  Aggregates: median `Spec`, usage-weighted `Spec`,
median profile cosine, and the same for dominant and rare atoms.

## 7. Stage D — cross-seed dictionary stability

For each pair `0↔1`, `0↔2`, `1↔2`:

* `C_ij = |d_i^T d_j|` (permutation + sign symmetric);
* Hungarian matching (`scipy.optimize.linear_sum_assignment`) maximising the
  total `|cos|`;
* report mean / median / min / p10 matched `|cos|`, `#{≥0.90}`, `#{≥0.95}`;
* for matched atoms compare activation frequency difference and the IHT-10
  train SMD profile cosine.

Frozen verdict (raw distributions are always reported as well):

* `STABLE_VOCABULARY`: mean matched `|cos| ≥ 0.90` **and** `#{≥0.90} ≥ 24`
  **and** median matched profile cosine `≥ 0.80`;
* `PARTIAL_VOCABULARY`: mean `|cos| ≥ 0.70` **or** `#{≥0.90} ≥ 16`, **and**
  median profile cosine `≥ 0.50`;
* `UNSTABLE_BASIS`: otherwise.

## 8. Stage E — sparse ↔ dense recoverability (same frozen `D`)

Primary representations, all with the seed's own `Dbar`:
`sparse_iht10`, `sparse_omp`, `sparse_iht30`, `dense_tied = phi @ Dbar`.
For the six ordered pairs (dense↔iht10, dense↔omp, dense↔iht30):

* OLS `y ≈ [x, 1] W` fit on **train only** (`np.linalg.lstsq`, no ridge, no
  hyper-parameter sweep), evaluated on valid;
* metrics: overall `R²` (target mean taken from train), per-dimension `R²`
  distribution, `||Y−Ŷ||_F/||Y||_F`, mean per-sample cosine, linear CKA.

Secondary (seed 0 only, contextual): the same dense↔iht10 recoverability using
the trained DenseTied control's own dictionary.

Interpretation (frozen): high both-way `R²` → sparse is largely a linear
re-parameterisation of dense; high dense→sparse with low sparse→dense →
sparsification loses dense information; neither high while DenseTied
downstream is better → the representations genuinely differ and the sparse one
shows no task advantage.

## 9. Stage F — the IHT-30 training gate (frozen)

Default: **DO NOT TRAIN**.  The single authorised new run is FINAL-CLEAN with
IHT-30, seed 0, and only if, on the **valid** split:

* **Condition 1** (per seed): `recon_frobenius(IHT30) ≤ 0.5 × recon_frobenius(IHT10)`
  **and** the absolute drop `≥ 1e-4`.
* **Condition 2** (per seed): at least one of
  * support: `mean Jaccard(IHT30, OMP) ≥ mean Jaccard(IHT10, OMP) + 0.05`;
  * concentration: `N_eff +1.0` **or** top-5 share `−0.10` **or**
    max activation rate `−0.10`;
  * coefficient geometry: mean `top1/L1` share `−0.05` **or** mean code cosine
    to OMP `+0.05`.
* **Condition 3**: conditions 1 and 2 hold jointly for **≥ 2 of the 3**
  dictionaries.

If the gate fires, the run must be exactly
`FINAL-CLEAN + IHT-30, seed 0`, with C6 mask, K=32, s=8, the same phi65, the
same binding, relation, reader, lambda (`33.95873017865987`), Adam
(`1e-3` / `wd 1e-5`), batch 128, clip 5.0, 320 epochs, Top-5 soup and the same
initialisation / data-order protocol.  **The only changed quantity is the IHT
step count 10 → 30.**  IHT-10 seed 0 is never retrained; it is the paired
historical reference.

Single-trajectory rule: one seed-0 IHT-30 trajectory (observable as epoch-20,
40, …, 320 checkpoints of that one run).  A 20-epoch sanity screen of the same
seed and protocol is permitted (and bit-identical to the prefix of the formal
run) before the formal 320-epoch run; no second formal run is allowed.

## 10. Epoch-20 sanity gate (only if the gate fires)

Proceed unless a numerical failure is observed: non-finite loss / dictionary /
gradient, representation collapse, or a catastrophic MAE that does not improve.
A temporary MAE difference of 0.003–0.005 is explicitly **not** a stop reason.

## 11. Final interpretation rules (only if the gate fires)

* `MAE(IHT30) < MAE(DenseTied) − 0.003` → at most “single-seed evidence
  supporting sparse specificity”; never “established across seeds”.
* `MAE(IHT30) ≈ MAE(DenseTied)` → specificity remains unestablished.
* DenseTied still better → the bottleneck is not only the IHT iteration count;
  no IHT-100 training search in this round.

## 12. Forbidden in this round

IHT-100 training, OMP training, K=64, any s/λ sweep, mean-centred or residual
dictionary implementation, new phi descriptors / chemistry, A1 attributed
dictionary, new node/edge fusion, new relation, shell experiments, new
MLP width/depth, official test, new seed-1/seed-2 training, a second IHT-30
run, or any post-hoc threshold change.  The next-round recommendation may only
be recorded, not executed.

## 13. Official-test blocker and CPU guard

`official_test_loaded = false` in every payload written by the module and the
runner; the module contains no official-test code path and the guard
`official_test_blocker` raises on any payload that does not carry `False`.
`cpu_only_guard` raises unless `device.type == "cpu"`.  The runner only reads
`env_train.pt` / `env_valid.pt`.

## 14. Focused tests (`tests/test_e2e_dictenv_dictionary_coder_audit_v1.py`)

IHT-10 reproduces the frozen coder exactly; IHT-30/100 preserve the exact
top-s; OMP exact-s contract and l2 optimality; the same-D audit truly uses one
identical `Dbar`; support Jaccard / activation frequency / effective-atom /
Gini / entropy toy checks; dictionary matching under permutation and sign flip;
mean direction in the dictionary input space; train-only PCA; structural
profile train normalisation with valid using the frozen train statistics;
structural descriptors follow the phi65 provenance; train-only linear
recoverability; CPU-only guard; official-test blocker; frozen gate thresholds;
`IHTStepModel(iht_steps=10)` bit-identical to `CleanMechModel`.

## 15. Stop condition

Stop after: (1) the 3-dictionary same-D coder audit, (2) dominant-atom
diagnostics, (3) atom specialisation, (4) cross-seed stability, (5) sparse↔dense
recoverability, (6) the gate decision, (7) the single seed-0 IHT-30 run iff the
gate fired, (8) REPORT / DECISION / records / STATE.  No further exploration
even if CPU budget remains.

## 16. Deliverables

`results/e2e_dictenv_dictionary_coder_audit_v1/` with `preflight.json`,
`preregistration_snapshot.json`, `coder_geometry/seed{0,1,2}/summary.json`,
`dominant_atom/seed{0,1,2}.json`, `atom_specialization/seed{0,1,2}.json`,
`cross_seed_stability/summary.json` + `dictionary_stability.csv`,
`sparse_dense_recoverability/summary.json` + `recoverability.csv`, `gate.json`,
`analysis_tables.md`, `summary.json`, and (only if the gate fired)
`iht30_training/`; plus the committed `REPORT.md` / `DECISION.md`, one claim
record, one decision record and the `STATE.yaml` entry.
