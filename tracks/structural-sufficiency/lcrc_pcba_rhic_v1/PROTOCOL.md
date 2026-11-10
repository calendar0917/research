# PROTOCOL — LCRC-RHIC v1

Round: `lcrc-pcba-rhic-v1`
Status: **frozen before the first formal training step.** Any post-freeze
incident is recorded in `REPORT.md` under "Incidents and deviations"; this file
is not silently edited after the run.

## 0. Scope and one-sentence question

> With LCRC-W2's local encoding and readout unchanged, can CPC-v1's rooted
> two-bond path tokens be **hierarchically grouped by first-hop branch** — one
> within-branch saturated mean and one across-branch saturated mean, both
> computed from pairwise co-occurrence of *distinct* elements — recover more of
> W2-DYNAMIC's gain than a flat sum over paths, still **without any iterative
> hidden-state propagation**?

Exactly **one** model is trained formally: `LCRC-RHIC`, seed 0, 40 epochs. This
is a fixed-mechanism boundary experiment, not a PathNN/GNN reproduction, not a
tuning search, and not a capacity-matched comparison. No second form, no second
seed, no path-length or attention variant, no extra loss, no pretraining, no
augmentation, and no retraining of W2 / RC-REAL / CPC / W2-DYNAMIC is allowed.

Historical references (read-only; Test values are exposed engineering
comparisons, never a blind estimate and never a selection signal):

| Model | Parameters | Best Valid AP | Best-Valid Test AP | Test AP @40 |
|---|---:|---:|---:|---:|
| W2 | 994,324 | 0.2652733161 | 0.2536168632 | 0.2509858 |
| RC-REAL | 1,051,022 | 0.2677177024 | 0.2559409168 | 0.2561672 |
| LCRC-CPC | 1,043,333 | 0.2659443284 | 0.2623560107 | 0.2614923 |
| W2-DYNAMIC | 1,128,468 | 0.2750671763 | 0.2622260490 | 0.2601363 |

## 1. Inherited, read-only implementations

* `lcrc_pcba_backbone_scaling_v1/code/bs_model.py` — `BSModel` (W2 at scale 2),
  the historical W2 seed-0 constructor and state helpers.
* `lcrc_pcba_backbone_scaling_v1/code/bs_common.py` — W2 widths, frozen
  scratch recipe, frozen 40-epoch batch-order plan, data identities, official
  Evaluator, W2 / RC-REAL / W2-DYNAMIC numbers, provenance.
* `lcrc_pcba_cpc_v1/code/cpc_paths.py` — the audited rooted two-bond path cache
  and its vectorised collate. **Reused verbatim; not rebuilt.**
* `lcrc_pcba_cpc_v1/code/cpc_model.py` — the audited CPC path-token encoder and
  state comparison helpers. The token is reused verbatim as `rhic.token`.
* `lcrc_pcba_continuation_recipe_v1/code/cc_data.py` — the only gated reader of
  official Test labels, used strictly after the Test prediction freeze.
* `lcrc_pcba_cpc_v1/analysis/decision.json` — the frozen 106-task high-support
  set (official Valid and Test positives >= 20 each) and the 22 low-support
  complement. `rhic_common` re-reads and verifies this list; it is never
  recomputed from Test labels before the freeze.

No inherited module is edited. No W2 / RC-REAL / CPC / W2-DYNAMIC parameter is
retrained, fine-tuned or reinitialised.

## 2. Model: `LCRC-RHIC`

### 2.1 Main path

Construct the **untrained** historical W2 seed-0 model. Keep every W2
computation unchanged and call each module exactly once:

```python
a, b, scal = base._encode(batch)                # once
h0 = base.drop(base.local(batch, a, b, scal))   # local and drop once

delta = RHIC(h0, b, path_index, bond_index)     # once, reads only this h0/b
h = h0 + delta

logits = base.post_local(h, batch)              # inherited W2 readout, once
```

`h0, delta, h ∈ R^{N_atoms×128}`; `b` is the W2 bond-encoder output
`R^{N_bonds×192}`. RHIC never reads `h`, never writes back, and there is no
iteration: the only sequence is `h0 → delta → h0+delta → readout`.

### 2.2 Paths and branches (reused cache, new runtime grouping)

A path is a rooted simple two-bond path `i--j--k`: root `i`, middle `j`,
endpoint `k`, with `i != j`, `j != k`, `i != k`. Enumeration is complete over
every root, every `j ∈ N(i)`, every `k ∈ N(j) \ {i}`, including triangles. The
cache is the frozen CPC cache:

* fields: `cpc_ptr`, `cpc_middle`, `cpc_end`, `cpc_bond_ij`, `cpc_bond_jk`,
  `cpc_path_atom` (implicit root);
* identity: 437,929 rows, **34,369,144 paths**, 658,409,864 bytes,
  `rows_sha256 85b8d38b…`, `cache_sha256 caad3e90…`;
* the cache contains structures and bond tables only — no labels of any split.

A **directed first-hop branch** is the ordered pair `(root i, middle j)`. Its id
is derived at runtime, with no new cache:

```
branch_id = 2 * physical_bond_id + direction
direction = 0 if bond_index[physical_bond_id] == (i, j) in storage order
            1 otherwise
```

Thus `(i,j)` and `(j,i)` are different branches. A branch with `n_b = 0` paths
does not exist, is never encoded, and is never counted at its root. A root with
`d_i = 0` branches receives `delta_i = 0` exactly.

### 2.3 Token (CPC v1, verbatim) and the two-level composition

For every path `p = (i,j,k)`:

```
r = W_r h0[i],  m = W_m h0[j],  e = W_e h0[j->k bond],  d = W_d b[i->j bond]
u_p = [r_i, m_j, e_ij, d_jk, path features]
z_p = LayerNorm(u_p)
v_p = gate_p * path2(SiLU(path1(z_p)))          # 64-d, gate_p in (0,1)
```

`branch1: Linear(209→96)`, `branch2: Linear(96→64)`; `root1: Linear(162→192)`,
`root2: Linear(192→128)`; all hidden activations SiLU, no LayerNorm in the new
MLPs (the token already has its LayerNorm).

**Level 1 — inside one branch `b = (i,j)`**, over its `n_b` distinct path
tokens `v_p ∈ R^64`:

```
mu_b    = (1/n_b) Σ_p v_p
D_b     = Σ_p (v_p − mu_b)²
q_b     = mu_b² − D_b / (n_b (n_b − 1))          ∈ R^64
        = (S_b² − T_b) / (n_b (n_b − 1)),  S_b = Σ v_p, T_b = Σ v_p²
q_b     = 0 exactly when n_b < 2
c_b     = [r_i (32), m_j (32), e_ij (16), mu_b (64), q_b (64), log1p(n_b) (1)]   # 209
beta_b  = branch2(SiLU(branch1(c_b)))            ∈ R^64
```

`q_b` is the mean of `v_p ⊙ v_q` over **distinct ordered pairs** `p ≠ q`
(`n_b(n_b−1)` pairs); self-products `v_p²` are excluded. The implementation
never forms the pair matrix: `S² − T = n(n−1)mu² − D`, and `D` is accumulated
directly from the deviations for numerical stability.

**Level 2 — across the `d_i` branches of one root `i`**, over distinct branch
values `beta_b`:

```
mean_branch_i = (1/d_i) Σ_b beta_b
D_i           = Σ_b (beta_b − mean_branch_i)²
cross_branch_i= mean_branch_i² − D_i / (d_i (d_i − 1))    ∈ R^64
cross_branch_i= 0 exactly when d_i < 2
g_i           = [r_i (32), mean_branch_i (64), cross_branch_i (64),
                 log1p(path_count_i) (1), log1p(d_i) (1)]                  # 162
delta_i       = root2(SiLU(root1(g_i)))            ∈ R^128
delta_i       = 0 exactly when d_i = 0
```

`h = h0 + delta` is the only write-back; `post_local` then runs its inherited
computation once. There is no cross-root exchange, no path-to-path message
passing, and no second RHIC invocation.

### 2.4 Exact parameter budget

| Group | Tensors | Parameters |
|---|---|---:|
| W2 base (atom/bond encoders, local, f, rho) | inherited | 848,096 |
| W2 readout (`post_local`) | inherited | 146,228 |
| **W2 subtotal** | | **994,324** |
| Token: LayerNorm(128) | 256 | |
| Token: role projections 3×(128·32+32) | 12,384 | |
| Token: bond projection 192·16+16 | 3,088 | |
| Token: `path1` 192·96+96 | 18,528 | |
| Token: `path2` 96·64+64 | 6,208 | |
| Token: gate 96·1+1 | 97 | |
| **Token subtotal** | | **40,561** |
| Branch: `branch1` 209·96+96 | 20,160 | |
| Branch: `branch2` 96·64+64 | 6,208 | |
| **Branch subtotal** | | **26,368** |
| Root: `root1` 162·192+192 | 31,296 | |
| Root: `root2` 192·128+128 | 24,704 | |
| **Root subtotal** | | **56,000** |
| **RHIC subtotal** | | **122,929** |
| **Total** | | **1,117,253** |

### 2.5 Initialisation and the epoch-0 identity

* The W2 seed-0 state is loaded **bitwise** from
  `bs_model.build_model("W2", seed=0)`; its trunk fingerprint must equal the
  recorded `5f7101a8…`.
* The RHIC modules are constructed inside `torch.random.fork_rng(devices=[])`
  and initialised from a private `torch.Generator` with seed
  `0·1_000_003 + 2002`; the global torch RNG is not advanced.
* `root2.weight` and `root2.bias` are **zeroed**, so at epoch 0 `delta ≡ 0`,
  `h ≡ h0`, and the RHIC logits equal the untrained W2 logits.
* Required checks: exact equality on CPU (`max |Δlogit| = 0`), and
  `max |Δlogit| <= 1e-4` on the CUDA preflight; epoch-0 Valid AP must be within
  `1e-4` of the historical untrained W2 endpoint.

## 3. Data protocol

* Dataset: official `ogbg-molpcba` through the inherited loader — 350,343
  Train / 43,793 Valid / 43,793 Test molecules, 128 tasks, official scaffold
  split. **No new split is created.**
* Labels: the frozen NaN-masked Train/Valid artifact. Official Test targets are
  NaN during training and analysis and are opened exactly once, after the Test
  logits are sealed, through `cc_data.read_official_test_labels`.
* Normalisation: the frozen W2 Train-only normalisation; recomputed nowhere.
* Batch order: the historical W2 seed-0 40-epoch plan
  (`global_plan_sha256`, `consumed40_prefix_sha256` verified at start), batch
  size 128, 2,738 steps/epoch, 109,520 steps in total.

## 4. Training recipe (fixed)

| Item | Value |
|---|---|
| Optimizer | AdamW, one parameter group, same LR for W2 and RHIC |
| Weight decay | 1e-3 |
| Betas / eps | (0.9, 0.999) / 1e-8 |
| LR schedule | 1-epoch linear warmup 3e-5 → 3e-4, then per-step cosine 3e-4 → 1e-5 over 109,520 steps |
| Dropout | 0.10 (inherited W2 dropout, unchanged) |
| Gradient clip | 5.0 (global norm) |
| Precision | FP32 |
| Loss | masked BCE over observed labels (NaN masked) |
| Epochs | 40 |
| Seed | model/init seed 0; batch-order seed 0 |
| Eval | official Validation AP and BCE after **every** epoch |
| Budget | target ≤ 3.0 A100 GPU-hours, hard cap 4.0 |

No early stopping: every epoch 1..40 is trained and evaluated, the last epoch
is the pre-registered `@40` endpoint, and the best-Validation checkpoint is
selected by `argmax Valid AP` (earliest wins ties).

## 5. Pre-training audits (blocking)

All of the following must pass **before** the formal run; outputs are committed
under `analysis/`.

**A. Structure invariances and cache identity** (`rhic_audit.py --mode structure`)

* cache identity: 437,929 rows, 34,369,144 paths, `rows_sha256`, `cache_sha256`,
  field hashes and byte count equal to the CPC round; no labels present;
* full-cache branch audit over all paths: exactly one direction per path, the
  middle is the other endpoint of the first bond, the second bond connects
  `middle` to `end`, `branch_id ∈ [0, 2·n_bonds)`, and `(root, middle)` is
  bijective with the branch id;
* atom-permutation, bond-storage-direction, bond-id-order and path-order
  invariance; molecule isolation;
* 11 hand-checked small graphs (single atom, zero-path bond, chains, triangle,
  square, branched tree, multi-path ring, K4, …): independent Python
  enumeration of branches, counts, `q_b` and `cross_branch` zeros;
* an independent NumPy float64 recomputation of `mu_b`, `q_b`, `mean_branch`
  and `cross_branch` on a real batch (max abs diff ≤ 1e-6);
* the **grouping witness**: graphs A = `{x,x},{y,y}` and B = `{x,y},{x,y}` have
  the same multiset of tokens, hence the same CPC global mean, but different
  branch partitions. The audit must show identical CPC mean
  (`< 1e-6`) and materially different RHIC branch `q_b` and root delta.
* Train structure coverage on the frozen Train atoms (9,001,316): branch count,
  path count, fraction of roots with ≥ 2 branches, fraction with a multi-path
  branch, multi-path branch fraction, molecule-level coverage.

**B. One-shot propagation** (`--mode preflight`, architecture gate)

* hooks count exactly one call to atom encoder, bond encoder, `local`, `drop`,
  RHIC, `f`, `rho`, readout, and `post_local`;
* RHIC receives exactly the `drop(local(...))` tensor and the bond-encoder
  output; `post_local` receives exactly `h0 + delta`;
* no re-reading of the updated state; a remote endpoint atom's `h0` change
  propagates to a root delta (path read), but never through an updated atom
  state.

**C. Gradients** (preflight, 100-batch GPU profile)

* exactly 110 steps (10 warm-up, 100 measured) on the formal plan;
* after step 3, 10 and 100 **every one of the 24 new tensors** has a non-zero
  gradient; at step 1 the zeroed `root2` keeps the gate closed and only its own
  weight/bias move;
* step-time p50/p90, peak memory, and projected 40-epoch GPU-hours ≤ 4.0.

**D. Test isolation**

The training and analysis scripts assert that no official Test target is
finite; the checkpoint payload records `test_evaluated: false`.

**E. Numerical parity**

CPU `max |Δlogit| = 0` at epoch 0; the CUDA preflight requires ≤ 1e-4 against
the untrained W2, and CPU↔CUDA RHIC logits within 1e-4.

## 6. Test flow (one shot, after training)

1. Select the checkpoint by **official Validation AP only**.
2. Freeze two RHIC prediction files: best-Valid and epoch-40. Each is produced
   from the label-free structure cache, hashed (SHA-256), shape-checked
   (43,793 × 128), finite-checked, and sealed together with the frozen
   reference logits (W2, RC-REAL, CPC, W2-DYNAMIC) in
   `analysis/test_prediction_manifest.json`.
3. Replay official Validation AP from both checkpoints; require
   `|replay − recorded| <= 1e-4`.
4. Only then open official Test labels through
   `cc_data.read_official_test_labels(task, freeze_path=...)` and run exactly
   one scoring pass.
5. Score every model with the OGB official
   `Evaluator("ogbg-molpcba")` **and** an independent sklearn per-task AP;
   require the two aggregates to agree within 1e-6.

No Test number is ever used to choose a checkpoint, epoch, hyper-parameter or
architecture.

## 7. Mechanism diagnostics (Validation only, frozen best checkpoint)

Three eval-mode passes over the full official Validation split:

* `real` — the frozen model;
* `delta_zero` — `h = h0` (residual removed, W2 readout unchanged);
* `branch_assignment_shuffle` — within every root, the assignment of its paths
  to branch ids is permuted (seed 0), preserving the path-token multiset, the
  root membership, the branch count `d_i` and the per-root branch sizes.

Reported: Valid AP/BCE, mean/max |Δlogit|, mean |Δp|, per-task wins/losses,
full-Validation `R = RMS(delta)/RMS(h0)`, RMS of `mu_b`, `q_b`, branch value,
`mean_branch`, `cross_branch`, `z`, `v`, gate statistics, branch/path
coverage, and the non-zero fractions of `q_b` and `cross_branch`.

`MECHANISM_INVALID` is declared if any of: the trained residual is exactly
zero or `R < 0.005`; `R > 2`; active-branch fraction ≤ 0.5; multi-path branch
fraction ≤ 0.1; `q_b` non-zero fraction ≤ 0.1; `cross_branch` non-zero fraction
≤ 0.1; the `real` vs `delta_zero` mean |Δlogit| ≤ 1e-6; or the `real` vs
shuffled-grouping mean |Δlogit| ≤ 1e-6.

## 8. Decision rule (pre-registered)

All APs are official macro APs over the **127 scorable Test tasks** (task 45 is
unscorable). The **106-task high-support set** is fixed from official Valid and
Test positives ≥ 20 each; the 22 low-support tasks are reported separately and
never presented as the scorable population.

* `PERFORMANCE_ADVANCE`: Valid best ≥ 0.280 **and** Test best ≥ 0.270 **and**
  high-support mean ΔAP vs CPC ≥ +0.003 **and** ≥ 70/106 high-support tasks
  improved.
* `PARTIAL_ADVANCE`: mechanism valid **and** either
  (Valid ≥ 0.273 and Test ≥ 0.265 with high-support Δ ≥ 0), or both splits
  improve on CPC by ≥ 0.002 with high-support Δ ≥ +0.002.
* `REGRESSION`: Valid < 0.262944 and Test < 0.259356, or high-support mean ΔAP
  vs CPC < −0.003.
* `MECHANISM_INVALID`: any mechanism gate in §7 fails, regardless of AP.
* `NO_CLEAR_ADVANCE`: none of the above.

## 9. Reporting requirements

`REPORT.md` (Chinese research report) must answer:

1. Did RHIC beat W2 / RC-REAL / CPC / W2-DYNAMIC on official Validation and Test?
2. Did it reach the pre-registered targets (Valid 0.280, Test 0.270)?
3. Did the hierarchical branch composition add anything over flat CPC?
4. Is the gain, if any, broad across the 106 high-support tasks or concentrated?
5. What did the mechanism diagnostics show (real vs delta-zero vs shuffled
   grouping)?
6. What did it cost (parameters, GPU-hours, BCE/AP trade-offs), and was the
   one-shot constraint respected?
7. What is the decision label, and what does it imply for the structural
   sufficiency hypothesis?

The mandatory comparison table contains, for W2, RC-REAL, CPC, W2-DYNAMIC and
RHIC: parameters, Valid best, Valid @40, Test best-Valid, Test @40, Train AP@40,
Train BCE, Valid BCE and GPU-hours. The report also gives the full 127-task
macro and the 106-task high-support means for every model, plus the 22
low-support tasks as a separate, clearly marked diagnostic set.

## 10. Deviations

Any deviation (interrupted run, resumed checkpoint, GPU change, unexpected
diagnostic failure) is appended to `REPORT.md` with UTC timestamps and the
affected hashes. The formal run is repeated only if the run itself did not
complete 40 epochs and its full Validation curve.
