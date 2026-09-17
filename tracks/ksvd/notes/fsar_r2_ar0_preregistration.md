# FSAR-R2-AR0 pre-registration — fixed radius-2 explicit pure-topology coordinates + assignment residual

Branch `exp/fsar-r2-ar0-zinc`, created from the FSAR route-recheck verdict HEAD
`4bf7ff3`.  Official ZINC test is **never** loaded (`official_test_loaded =
false` in every JSON; the preprocessor only reads `train` / `val`).

This is a self-contained round.  It is **not** a continuation of FSAR-v1/v2 or
of the route recheck, and it does not chase ZINC SOTA.  It answers exactly one
question.

## 0. Question

> Under a fixed radius-2 explicit pure-topology coordinate, a strong
> structure/attribute marginal baseline and one identical training protocol,
> does the real "node structural-role ↔ node attribute assignment" provide a
> stable predictive increment over the marginals **and** over a matched
> marginal-capacity control?

Explicitly **not** this round: chasing SOTA, extending the radius, motif search,
fitting the historical `B-full` hidden state, or reusing `B-full` / `B-bag` as a
pure-topology teacher.  `B-full` reads atom/root/distance/bond primitives and
runs two rounds of edge-aware message passing; it is not a pure-topology `S`
teacher and is only recorded as an external mixed-model reference.

## 1. Information definitions

A molecule is `G = (T, X, E_attr)` with `T` the unlabelled topology, `X` the
node attributes (28 padded categories; 21 observed) and `E_attr` the undirected
edge attributes (4 padded categories; 1..3 observed).

```
S  : per-node rooted radius-2 pure-topology coordinate phi_v (65-D).
A  : whole-graph node/edge attribute multiset only (64-D).
B  : the assignment of node attribute to structural role, read only as the
     centred cross-statistic C (65 x 28) against the explicit S coordinates.
```

No aligned/mixed bypass exists: `C` (or `P`) reach the prediction only through
a bias-free linear functional of the normalised statistic.

## 2. `S`: fixed radius-2 explicit pure-topology coordinate (no primitive search)

Reused **unchanged** from `fsar_v2._explicit_basis_for_patch`:

* 11-D node basis: root indicator (1); restrained shell indicators (3); induced
  degree (1); shell-resolved neighbour counts (3); rooted walk coordinates
  `(A^k)_{vu}` for `k = 1, 2, 3` (3).
* 15-D undirected edge basis: unordered endpoint shell pair (6); degree sum /
  degree difference / common neighbours (3); shell-neighbour sum and difference
  (6).

For every **original** node `v` (one row per node, never a patch copy):

```
phi_v = [ root_basis(11),
          mean_node_basis(11), std_node_basis(11),
          mean_edge_basis(15), std_edge_basis(15),
          log1p(|V_{B2(v)}|), log1p(|E_{B2(v)}|) ]   in R^65
```

`|E_{B2(v)}|` counts every induced **undirected** edge exactly once.  No
chemistry enters `phi_v`; no C4/C5, homomorphism, RRWP or other new primitive is
added this round.  Radius stays 2.

S encoder (fixed, no width sweep): `65 -> 64 -> 64`, SiLU.  Whole-graph

```
z_S = [ sum_v u_v , mean_v u_v , log1p(n) , log1p(m) ]   in R^130
```

## 3. `A`: whole-graph attribute marginal (no structural position)

```
a(G) = [ log1p(atom_counts)(28), atom_freq(28),
         log1p(bond_counts)(4),  bond_freq(4) ]   in R^64
```

Undirected bonds are counted once.  `A` may not contain root identity,
shell-conditioned histograms, typed patch histograms, typed WL tokens, or any
joint atom-type × structural-position statistic.

## 4. `M0`: strong marginal baseline

```
yhat_0 = F0([z_S, a])        # 194 -> 64 -> 32 -> 1, SiLU, no architecture sweep
```

Nonlinear fusion of `S` and `A` is allowed (it is still marginal, not an
assignment).

## 5. `B`: strict assignment residual

With `q_v = onehot(x_v) in R^28`:

```
J = sum_v phi_v q_v^T
P = (1/n) (sum_v phi_v) (sum_v q_v)^T
C = J - P = sum_v (phi_v - phibar)(q_v - qbar)^T      # sum-centred, no /(n-1)
```

Train-only per-coordinate RMS scaling (no dataset-mean subtraction):

```
D_C = sqrt(E_train[C^2] + eps)
C~  = C / D_C      (coordinates with raw train RMS <= 1e-9 are masked to zero)
```

The "effective coordinate" mask and counts are recorded.  Final model:

```
yhat_B = F0([z_S, a]) + <W_B, C~>          # W_B in R^{65 x 28}, no bias
```

Hard requirements: `W_B` has no bias; there is no MLP / activation / square /
other nonlinearity after `B`; no other aligned feature path exists; `W_B` is
zero-initialised so that step-0 prediction is exactly `M0`; step-0 / early-step
task gradient on `W_B` must be non-zero.

For any fixed parameters and a uniform random permutation `pi` of the node
attributes, `E_pi[C(T, pi X)] = 0` exactly, hence `E_pi[<W_B, C(T, pi X)>] = 0`.
This is the property the round's interpretability rests on, and the reason the
**sum-centred** `C` (not `C/(n-1)`) is used.

## 6. `MM`: matched marginal-capacity control

```
P~  = P / D_P                              # train-only RMS, same safe handling
yhat_M = F0([z_S, a]) + <W_M, P~>          # W_M in R^{65 x 28}, no bias, zero-init
```

`W_M` has the same shape / init principle / architecture context as `W_B`, and
shares the **identical** `S`/`A`/`F0` module objects' definitions and seeds.
Both nominal and active (non-zero-gradient) parameter counts are reported.

The formal comparison is exactly three models: `M0`, `MB`, `MM`.  No extra grid.

## 7. Pre-registered correctness tests (`tests/test_fsar_r2_ar0.py`, 17 tests)

1. node relabel invariance of `M0`/`MB`/`MM`
2. `S` chemistry purity
3. `A` assignment invariance
4. `P` assignment invariance
5. `C` assignment sensitivity
6. exact permutation expectation `mean_pi C = 0`
7. scalar `B` exact expectation `mean_pi <W_B, C> = 0`
8. batched statistics equal the numpy reference
9. no patch-copy inconsistency (permutation on original nodes only)
10. undirected bond counting
11. train-only scaler estimation (no dataset mean)
12. gradient viability of `W_B` / `W_M`
13. no mixed bypass (C/P enter only through the bias-free linear term)
14. `M0`/`MB`/`MM` share the base architecture bit-for-bit
15. zero-init `MB`/`MM` predicts exactly `M0`
16. parameter counts
17. assignment-only synthetic positive control

## 8. Two synthetic controls (must pass before ZINC)

* **Positive (assignment-only).**  Fixed 4-node path, fixed multiset 1 N + 3 C,
  label = whether N sits at an endpoint or an internal node.  `S` marginal and
  `A` marginal are identical across examples; only the assignment differs.
  Requirement: `M0`/`MM` cannot use the assignment, `MB` learns it, `W_B` does
  not collapse.
* **Negative (marginal-only).**  Label depends only on the topology marginal and
  the attribute counts; attribute placement is then randomised within each
  molecule.  Requirement: no stable, repeatable `MB` advantage over `M0`/`MM`.

If the positive control fails, **no formal ZINC run is launched**; an
implementation / scale / optimisation problem is diagnosed instead.

## 9. Cache and cheap offline audit

One versioned cache (`results/fsar_r2_ar0/cache/`) holds per-node `phi_v`,
`atom_idx`, graph-level `A`, `n_nodes`, `n_edges`, the label, and the
train/valid split identity; plus the train-only scaler.  `J`/`P`/`C` are
reconstructible from `phi` and `atom_idx`.  Official test is never loaded.

The cheap audit (`audit` stage) reports feature variance, zero coordinates,
`C`/`P` RMS and effective-coordinate counts, rank / redundancy, NaN/Inf,
the permutation property, effective dimensionality and the molecule-size
correlation.  No large feature search and no historical-`B-full`-driven feature
selection.

## 10. Training protocol (inherited verbatim, one protocol for all models)

Adam, `lr = 1e-3`, `weight_decay = 1e-5`, batch = 128, L1 loss, grad clip 5.0,
no scheduler, `max_epochs = 240`, `patience = 40`, checkpoint selection on
official validation, fixed equal-weight Top-5 soup.  No protocol change is made
for `MB` or `MM`.  Recorded per run: total and active parameters, gradient
norms, `||W_B||` / `||W_M||`, branch output mean/std, wall time, peak GPU
memory, best epoch, Top-5 epochs, soup MAE.

## 11. Short runs before the formal round

`M0`, `MB`, `MM` each get one short run (max 60 epochs).  This is a
trainability / implementation smoke, not a scientific result; small MAE
differences do not select a model.  Formal runs start only if all three are
finite with normal learning curves, the `MB`/`MM` branch receives real task
gradient, and there is no collapse / NaN / constant-output bug.

## 12. Formal budget and GPU policy

First formal round: `3 models x 2 paired seeds = 6 runs` at seeds `0, 1`.
The round prefers two independent jobs in parallel.  GPU policy: never touch or
co-tenant an unknown process; only a genuinely free GPU is used, and the
occupancy is re-checked immediately before each launch.  No DDP.  Each job is
one process with its own log / provenance; all launched jobs are waited on
inside the agent's shell call (no detached-and-forgotten work).

## 13. Seed-2 gate (automatic)

With `delta_base = MAE(M0) - MAE(MB)` and
`delta_capacity = MAE(MM) - MAE(MB)`, seed 2 is authorised only if **both**
paired seeds 0 and 1 satisfy `delta_base > 0` **and** `delta_capacity > 0`
(equivalently `MB` beats both `M0` and `MM` on both seeds).  If the direction
flips, no more seeds are bought, no width is changed, no radius is changed and
no motif is added; the round goes to a durable verdict.

## 14. Evaluation-only assignment shuffle

On the trained `MB` soup, each valid molecule's topology and full attribute rows
are kept fixed and the attribute assignment is permuted on the **original**
nodes (the permutation is then reflected in every derived statistic).  `A` and
`S` must be exactly unchanged; `C` must change.  Reported separately: actual
prediction, permutation-mean prediction, actual MAE, per-permutation MAE
distribution and mean, and `MAE(E[yhat_pi])` vs `E[MAE(yhat_pi)]` (not
confused).  This is a mechanism diagnostic only; it does not replace the
independently trained `M0`/`MM` controls.

## 15. Forbidden this round

No official ZINC test; no radius sweep; no width/depth sweep; no new primitive
(C4/C5, RRWP, homomorphism); no attention/Transformer; no distillation; no
auxiliary loss; no adaptive gate; no validation-driven feature selection; no
training `S` from historical `B-full` hidden states; no treating `B-full` as a
pure-topology upper bound; no temporary reintroduction of mixed descriptors; no
seed hunting; no optimizer sweep; no checkpoint cherry-picking beyond the frozen
Top-5 rule.  Only genuine correctness bugs may be fixed; a mere performance
deficit does not authorise an unregistered model extension.

## 16. Interpretation bands (pre-registered)

* `MB` stably beats both `M0` and `MM` on 2 or 3 seeds:
  "under the current radius-2 explicit topology coordinates, the real node
  structural-role ↔ atom-attribute assignment provides a stable predictive
  increment beyond the structure/attribute marginals and a matched
  marginal-capacity control."  No generalisation to "binding matters" or
  "FSAR beats GNNs".
* `MB` unstable or not better than `MM`:
  "the current node-level linear assignment residual is not stably supported
  under this explicit basis / training budget."  Never "structure-attribute
  binding does not exist".
* Synthetic positive passes but ZINC does not support `MB`: a valid negative
  result; the next candidate is a pair-level assignment residual, which is
  **not** implemented this round.
* Synthetic positive fails: an implementation / optimisation / scale problem,
  not a scientific refutation.

## 17. Local pre-flight (recorded before the remote deploy)

* `tests/test_fsar_r2_ar0.py`: **17/17 pass** (5.2 s, CPU).
* `sanity` stage: **17/17 checks pass**, `all_pass = true`.
* `synthetic_controls.json`: positive **pass** (`MB` valid MAE mean `0.0216` vs
  `M0 0.5653` / `MM 0.5247`, `||W_B|| ~ 0.056`); negative **pass** (`M0 0.0069`,
  `MB 0.0169`, `MM 0.0247`; no repeated `MB` advantage).
* `parameter_accounting.json` (exact):

| model | S encoder | base F0 | binding linear | total | active path |
|---|---:|---:|---:|---:|---:|
| `M0` | 8,384 | 14,593 | 0 | **22,977** | 22,977 |
| `MB` | 8,384 | 14,593 | 1,820 | **24,797** | 24,797 |
| `MM` | 8,384 | 14,593 | 1,820 | **24,797** | 24,797 |

  `M0`/`MB`/`MM` share bit-identical `S`/`F0` tensors at the same seed;
  `dataset_dependent_vocabulary_params = 0`.

* `feature_audit.json` (500-molecule sample):
  * `phi` (65-D): `phi` effective rank `9.11`, 11 zero-std coordinates,
    no NaN/Inf;
  * `A` (64-D): effective rank `16.69`, 24 zero-std coordinates (unused padded
    categories / rare types), no NaN/Inf;
  * `C` `65 x 28`: effective coordinates `1109 / 1820`, cross-molecule
    effective rank `1.40` (top singular fraction `0.89`) — `C` is close to
    rank-1 across molecules, a descriptive diagnostic;
  * `P` `65 x 28`: effective coordinates `1151 / 1820`;
  * train-only scalers: `C_rms max 0.963`, `P_rms max 32.47`,
    `dataset_mean_subtracted = false`;
  * `pearson(||C||, n) = 0.363`.
* `preprocess`: 10,000 train / 1,000 valid, 33 s, official test not loaded.
* Local CPU training smoke (`MB`, 2 epochs): finite, decreasing
  (`train 0.982 -> valid 0.764`), branch norm `0.435`, no NaN.

## 18. Execution waves

```
short  GPU(available): M0 60ep, MB 60ep, MM 60ep   (trainability smoke)
wave 1 GPU: M0-s0, MB-s0
wave 2 GPU: MM-s0, M0-s1
wave 3 GPU: MB-s1, MM-s1
wave 4 (only if the seed-2 gate fires): M0-s2, MB-s2, MM-s2
```

Two independent jobs may run concurrently on one genuinely free GPU; GPU 0 was
observed occupied by an unknown third-party process at pre-registration time
and is never co-tenanted.

## 19. Files

* model / features: `experiments/luyin16/fsar_r2_ar0.py`
* runner: `experiments/luyin16/zinc_fsar_r2_ar0.py`
* tests: `tests/test_fsar_r2_ar0.py`
* results: `results/fsar_r2_ar0/` (gitignored)
* durable note: `notes/fsar_r2_ar0.md` (written after the pulled results)

## 20. Durable outputs

After the formal results are pulled back and analysed locally: write the
complete note, update claims/decisions, update `STATE.yaml` if appropriate, and
commit + push.  Every scientific conclusion is based only on pulled formal
results.
