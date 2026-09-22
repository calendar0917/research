# Pre-registration — TCCD-v2 GRAD-CONT: ordinal local-geometry, no scalar target

Round name: **TCCD-v2 GRAD-CONT** (a single intervention on top of frozen TCCD-v2
Prototype-REL). Study: `zinc-context-gap`. Protocol version:
`tccd_v2_gradcont`.

This is a **single-arm, single-seed diagnostic**. It does not amend TCCD-v2, does
not retrain BASE or CHEM-CONT, and never loads the official test or the official
valid split.

## 0. Why this round

The CHEM-CONT round showed the `714 → 64` encoder can reorganize the local
geometry, but that a **binary** "L1-identical = positive vs `c1 ≥ 2` = negative"
contrast only relocates the cliff: the Exact→VeryNear cliff shrank while the
radius-1 boundary sharpened (VeryNear→Moderate AUC `0.909`,
HardNegative Z cosine `−0.338`, MAE `0.28624 → 0.30206`, Outcome D).

This round removes the binary definition entirely and asks one question:

> If we only require **"a chemically closer environment must be closer in latent
> Z"**, without ever fixing an absolute similarity value, does Z form a genuinely
> graded local chemical geometry instead of a partition with a moved boundary?

## 1. Frozen BASE and CHEM-CONT — NOT RERUN

All existing numbers are read from frozen artifacts; no BASE/CHEM-CONT training,
no re-audit of their frozen JSONs.

| item | value | source |
|---|---|---|
| BASE checkpoint | `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt` | frozen |
| BASE internal-dev best MAE | `0.2862437069416046` | `results/tccd_v2/gateA_seed0.json` |
| BASE internal-dev Top-5 soup | `0.26635152101516724` | `results/tccd_v2/gateA_seed0.json` |
| CHEM-CONT checkpoint | `tracks/ksvd/results/tccd_v2_chemcont/prototype_chemcont_seed0_best.pt` | frozen |
| CHEM-CONT internal-dev best MAE | `0.3020593225955963` | `results/tccd_v2_chemcont/train_seed0.json` |
| CHEM-CONT Top-5 soup | `0.275267094373703` | `results/tccd_v2_chemcont/train_seed0.json` |

Frozen BASE/CHEM-CONT pair-uniform smoothness baselines (read-only, same pair
set, same audit code as this round):

| metric (internal-dev, uniform-over-pairs) | BASE | CHEM-CONT |
|---|---:|---:|
| Z max adjacent-bin jump across slices | `0.4054` | `0.5936` |
| Z max largest-boundary share | `0.8922` | `1.0321` |
| Z ordinal accuracy (all held-out triplets) | `0.6991` | `0.7164` |

## 2. Local chemistry and the ordinal relation (frozen definitions)

Decoding is structure-only, lifted from the frozen continuity audit and shared
by training and audit through `tracks/ksvd/code/tccd_v2_local_chem.py`.

For a patch `i`:

* `d1` = radius-1 edit count
  `½‖s1atom_i − s1atom_j‖₁ + ½‖rbond_i − rbond_j‖₁`;
* `d2` = peripheral / radius-2 edit count
  `|n2_i − n2_j| + ½‖s2atom_i − s2atom_j‖₁ + ½‖attach_i − attach_j‖₁ + ½‖s2bond_i − s2bond_j‖₁`.

**Pareto partial order (no scalar, no shell weights):** for anchor `i` and
candidates `j`, `k`,

    j ≺ k   iff   d1(i,j) ≤ d1(i,k)  and  d2(i,j) ≤ d2(i,k)
                   and at least one of the two is strict.

Declared `j` **closer** than `k`. If `j` has smaller `d1` but larger `d2`, the
pair is **incomparable** and is never used.

**Exact pairs are excluded:** any pair with equal canonical radius-2 key or with
`d1 = d2 = 0` is not an anchor–partner or anchor–candidate relation.

## 3. Triplet eligibility (frozen)

A training/dev ordinal triplet `(anchor i, closer j, farther k)` must satisfy:

* `j ≺ k` strictly under the Pareto order above (checked on the true counts);
* `graph(i) ≠ graph(j)`, `graph(i) ≠ graph(k)`, `graph(j) ≠ graph(k)`
  (cross-molecule, all three);
* `root(i) = root(j) = root(k)` and `degree(i) = degree(j) = degree(k)`;
* `|n_atoms(x) − n_atoms(i)| ≤ 2` and `|n_atoms(j) − n_atoms(k)| ≤ 2`;
* not exact (canonical-key equality / zero edits excluded on every relation);
* no target label `y` is read anywhere.

Candidate pool per anchor: the union of a uniform sample of the
`(root, degree)` group (≤ 384) and a sample of the radius-1-identical group
(≤ 96), cross-molecule. Bins are `d1 ∈ {0,1,2,3,≥4}` and `d2 ∈ {0,…,5,≥6}`. The
farther bin is bounded to `d1 ≤ 3`, `d2 ≤ 5`; valid Pareto bin pairs are consumed
nearest-farther-first, at most `4` triplets per anchor. Seed `20260922`.

## 4. Phase 0 coverage result (zero-training, run before training)

Train split: 8000 graphs, 185538 patches, 489 radius-1 keys, 13800 canonical
keys. Held-out internal dev: 2000 graphs, 46126 patches.

| statistic | train | dev |
|---|---:|---:|
| anchors total | 185538 | 46126 |
| eligible anchors (≥1 legal triplet) | 185488 | 46107 |
| eligible fraction | **0.99973** | **0.99959** |
| ordinal triplets | **741774** | 184332 |
| unmatched fraction | 0.00027 | 0.00041 |
| graph coverage | 1.000 | 1.000 |
| cross-molecule anchor↔closer | 1.000 | 1.000 |
| cross-molecule closer↔farther | 1.000 | 1.000 |
| comparison types (same_d1 / same_d2 / both) | 650603 / 82052 / 9119 | 159917 / 21049 / 3366 |

Most common train comparisons: `(0,1)≺(0,2)` 590168; `(0,1)≺(1,1)` 70172;
`(0,2)≺(0,3)` 24011; `(1,0)≺(1,1)` 18652; `(0,1)≺(1,2)` 7053. Candidate bin
histogram covers a dense near→medium→far ladder (`d1` bins 0–4, `d2` bins 0–6
all populated).

**Coverage stop rule (frozen):** proceed only if
`eligible_fraction ≥ 0.50` **and** `n_triplets ≥ 10000` **and**
`cross_molecule_anchor_closer ≥ 0.99`. **Result: PASS, proceed to training.**

## 5. Only new arm: GRAD-CONT (seed 0)

Architecture / protocol are byte-for-byte the TCCD-v2 Prototype-REL path: same
714-D records, same frozen `714 → 64` linear encoder init, same `K = 64` cosine
prototypes, same temperature parameterization, same `BAG + CᵀRC` reader, same
Adam/batch/lr/wd/clip, same ≤240 epochs / patience 40 / Top-5 soup, same internal
split seed `20260922` (8000/2000), same seed 0, same stopping. Prototypes,
composition, relations, reader and optimizer are unchanged. `L_grad` acts on `z`
only.

### Loss (frozen)

    L_grad = mean softplus( cos(z_i, z_far) − cos(z_i, z_close) )

over the training triplets `close ≺ far`. No projector, no margin, no distance-gap
weighting, no shell weighting, no scalar chemical kernel, no absolute cosine
target.

### λ calibration (frozen, no sweep)

On the first fixed training batch (first 32 training graphs), detached:

    λ_grad = 0.05 · L_task_initial / max(L_grad_initial, 1e-8)

frozen for the whole run. `L_total = L_task + λ_local·L_local + λ_balance·L_balance
+ λ_grad·L_grad`. λ_grad is **not** retuned after seeing dev MAE or geometry.

### Sanity gates before the formal run

`gate0` must pass: triplet legality (Pareto strict, root/degree match, size
tolerance, cross-molecule, exact excluded), no dev triplet, construction
invariant to scrambled `y`, non-zero `L_grad` gradient to `W`, zero gradient to
prototypes, `λ=0` forward/task identity against `tccd_v2.train_model`,
permutation invariance. Official test never loaded.

## 6. Metrics (frozen)

On the held-out internal dev split, one shared pair set (uniform-over-pairs,
matching the frozen audit weighting) and one shared held-out triplet set
(`dev_ordinal_triplets.npz`, 184332 triplets):

1. **Geometry curve** by `(d1, d2)` bin: N, mean/median Z cosine, mean/median C
   cosine, top-1 prototype agreement, for BASE / CHEM-CONT / GRAD-CONT.
2. **Fixed-axis slices** (Z and C): fix `d1` sweep `d2`; fix `d2` sweep `d1`;
   plus the audit-comparable `l1-identical` (sweep `d2`) and radius-1 (sweep `c1`)
   sub-curves.
3. **Adjacent-bin jumps**: per slice, `J_t = |mean_cos(t+1) − mean_cos(t)|`;
   report median, max, location; and **largest-boundary share** =
   largest adjacent drop / total near-to-far drop. Aggregate across slices.
4. **Ordinal accuracy** `P(cos(i,close) > cos(i,far))` (ties 0.5), overall and
   stratified into `same_d1` / `same_d2` / `both`, with molecule-cluster
   bootstrap CIs and paired deltas `GRAD-CONT − baseline`.
5. **Guardrail**: random-pair Z cosine, far-bin Z cosine, minimum bin mean Z
   cosine, number of bins (N ≥ 200) with negative mean Z cosine. (No
   HardNegative should be pushed to anomalous negative cosine.)
6. **Z and C reported separately**; top-1 prototype agreement reported. C is not
   trained or repaired this round.
7. **Task guardrail**: internal-dev best MAE and Top-5 soup only. MAE is never
   used as a tuning signal.

## 7. Pre-registered outcomes (frozen before the formal run)

Operational thresholds use the frozen baselines in §1.

* `graded_Z` = all of:
  * (a) dev ordinal accuracy (all) exceeds `max(BASE, CHEM-CONT)` by `≥ +0.03`
    with paired molecule-cluster bootstrap CI excluding 0;
  * (b) Z `max_abs_jump ≤ 0.445` (25 % below the worst baseline `0.5936`) **and**
    Z `max_largest_boundary_share ≤ 0.774` (25 % below `1.0321`);
  * (c) at least 3 of the 6 Z fixed-axis slices are monotone non-increasing;
  * (d) no bin with N ≥ 200 has mean Z cosine `< −0.10`.
* `mae_ok` = (best-checkpoint MAE − BASE ≤ `0.010`) **or**
  (Top-5 soup − BASE ≤ `0.010`).

Classification priority:

* **A — truly graded**: `graded_Z` and `mae_ok`.
  → Z already forms task-compatible graded local chemical geometry; next step is
  `Z → C` dictionary coding.
* **C — graded but task cost**: `graded_Z` and not `mae_ok`.
  → the structural ordering and task-relevant locality conflict; study what is a
  task-compatible local boundary. No λ rescue.
* **B — ordering learned, still step-like**: (a) holds but `graded_Z` fails
  through (b)/(c)/(d).
  → the model learned the rule boundary, not the chemical manifold; do not start
  dictionary coding; reconsider local similarity/environment definition.
* **D — ordinal loss barely changes Z**: (a) fails and (b) fails.
  → reconsider the representation limit (only with clear evidence; Z was already
  shown highly plastic).
* otherwise `INCONCLUSIVE`.

## 8. Stop rule

One arm, seed 0, one λ_grad. No seed-1 rescue. No λ / margin / distance-weight /
K / temperature / reader / relation / radius sweep, no projector, no GNN, no
message passing, no Transformer, no prototype or composition redesign, no
official test, no official-valid architecture selection. A favourable MAE does
not authorize further runs.

**Frozen before the formal GRAD-CONT run.** Commit: see the commit that adds
this file. Phase 0 coverage (`coverage_seed0.json`) was produced at the same
commit.
