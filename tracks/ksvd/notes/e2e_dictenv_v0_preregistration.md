# E2E-DictEnv-v0 — pre-registration (frozen before any implementation run)

Round **E2E-DictEnv-v0** · protocol `e2e_dictenv_v0` · study `zinc-context-gap`.
Prior-artifact audit: [`e2e_dictenv_v0_prior_artifact_audit.md`](e2e_dictenv_v0_prior_artifact_audit.md).

This document is the frozen specification.  It is written **before** any
implementation, cache build, training or GPU use.  Any deviation requires a
written amendment recorded here before the affected run.

---

## 1. Scientific hypothesis (the only hypothesis)

> A sparse, shared, task-coupled pure-topology structural dictionary
> (`phi65 -> D -> alpha32`) can be the **only fine-grained learned structural
> coordinate** inside local chemical environment formation; environments are
> constructed by binding that role code to primitive chemistry at
> root-relative shells; the frozen environment is then statically composed
> (read-only pairs, global marginal statistics) into a property prediction —
> and this sparse dictionary-core architecture is **specifically** better than
> the parameter-identical dense tied coordinate `z = phi @ P`.

The computation path that must really run:

```text
phi_v^65 -> D -> alpha_v^32 -> (alpha_v, q_v, shell_iv) -> E_i -> static composition -> y
```

Explicitly **not**:

```text
strong dense environment + dictionary correction      (FEC-D1, failed)
dictionary as message-passing transition              (GSCN, closed)
dictionary as whole-environment compression          (TCCD, closed)
```

## 2. Hard purity contract (enforced)

```text
NO message passing
NO recurrence
NO pair->centre
NO relation refresh
NO attention update
NO context writeback
NO typed-token lookup
NO parent-token lookup
NO FEC-S1 146->214->24 adapter
NO B-Full/B-Null hidden state
NO graph-level dictionary residual
NO dictionary pair-kernel side channel
NO 146-D patch_cont tensor anywhere in the model
```

Once `E_i` is formed it is frozen: no pair information may produce `E_i'`.
The only allowed cross-root computation is `c_ij = F(E_i, E_j, rho_ij)` once,
then direct graph pooling.

## 3. Input information discipline

* **Dictionary input** `phi_v in R^65`: the audited FSAR-R2-AR0 pure-topology
  node coordinate, produced by the existing implementation
  (`fsar_r2_ar0.build_phi`), identical to the SDB/FSAR artifact
  (`tracks/ksvd/results/fsar_r2_ar0/cache/`).  Pure topology only: no atom
  category, no bond category, no target, no learned hidden state.  For each
  molecular node `v`, `phi_v` is computed once.
* **Root-relative placement**: `s_iv = dist(i,v) in {0,1,2}` from BFS on the
  unlabelled graph.  `alpha_v` is node-centric (identical in every root's
  environment); `shell_iv` is role-relative and may differ per (i, v).

## 4. End-to-end sparse dictionary (frozen, no sweep)

* `K = 32`, `s = 8`, IHT steps `= 10`.  No sweep of K, s, steps.
* `D in R^{65x32}`; **initialised from the SDB-v0 frozen K-SVD artifact**
  `tracks/ksvd/results/sdb_v0/dictionary.pt` (65×32; sha256 begins
  `925d573a5808...`), verified exactly (SHA / shape / orientation) before any
  run.
* **Column normalization**: `Dbar_k = D_k / max(||D_k||_2, eps)` used for
  encoding and reconstruction in both arms.
* **Tied-IHT** (reuse repository implementation semantics):
  `C^{t+1} = H_s[ C^t + eta (phi - C^t Dbar^T) Dbar ]`, `||alpha_v||_0 <= 8`,
  10 iterations.  No LISTA, no softmax prototypes, no soft threshold, no
  learned encoder separate from D.  The same D is used for encoding,
  reconstruction and task prediction.
* **Task-coupling**: `D.requires_grad = true`; the property loss must reach
  `D` (`||grad_D L_task|| > 0` tested with the reconstruction term switched
  off).  Report `||Dbar_final - Dbar_init||_F` after training.  D must not be
  "technically trainable but effectively frozen".

## 5. Structural reconstruction anchor

* `hat_phi_v = alpha_v Dbar^T`; `L_rec = E_v ||phi_v - hat_phi_v||^2 / (||phi_v||^2 + eps)`.
* Total loss `L = MAE(hat y, y) + lambda_rec * L_rec`.

## 6. lambda_rec — one calibration, frozen (no sweep)

On the fixed official-train calibration batch (first 512 train molecules,
seed fixed, no valid, no test), at the Sparse arm's initialisation:

```text
lambda_rec = L_task^init / (L_rec^init + eps)
```

This single value is used **unchanged for both arms** (Sparse and DenseTied).
The two arms' objective weights must be identical; no comfort-calibration for
the dense control.  The value is recorded.

## 7. Local atom environment (the dictionary is the core)

No `patch_cont 146D` MLP, no FEC-S1 adapter.  The environment is formed from
sparse code × chemistry:

* `r_v = [1; alpha_v] in R^33` — dim 0 is a constant 1: an explicit
  **chemistry marginal anchor** (mass/count preservation), not a dictionary
  atom.
* Primitive chemistry `q_v in R^28`: ZINC atom category one-hot, never
  topology-conditioned (no degree/shell/neighbour/graph-state input before the
  dictionary role is read).
* Fixed latent rank `r_atom = 64`; learn `W_R in R^{33x64}`, `W_C in R^{28x64}`,
  `S in R^{3x64}` (root-relative shell embedding).  For occurrence (i, v):

```text
u_iv = (r_v W_R) * (q_v W_C) * S_{s_iv} / sqrt(64)
m_i^V = sum_{v in P_i} u_iv   in R^64     (SUM, never MEAN; no LayerNorm)
```

`m_i^V` contains (a) the DC component `1 x q_v x shell` (coarse
shell-conditioned chemistry marginal) and (b) the dictionary component
`alpha_v x q_v x shell` (sparse structural-subrole × chemistry assignment).
No 84-D atom_shell bypass, no 28-D root-atom dense bypass, no DenseRole/PCA
structural branch.

## 8. Bond environment (no edge dictionary this round)

* Primitive bond category `b_e in R^4`; root-relative 6-class shellpair
  `p_ie in {0..5}` (ordered SHELL_PAIRS (0,0),(0,1),(0,2),(1,1),(1,2),(2,2));
  `r_bond = 16`; learn `W_B in R^{4x16}`, `P in R^{6x16}`:

```text
m_i^E = sum_{e in P_i} ( (b_e W_B) * P_{p_ie} ) / sqrt(16)   in R^16    (SUM)
```

## 9. Local pure-topology scalars

`t_i in R^6` — the exact historical FEC/S0 six pure-topology local scalars
(log1p patch size, log1p patch edges, shell-2 boundary fraction,
cycle_rank/n, root degree/4, mean patch degree/4), taken as the last six
coordinates of the historical standardized `patch_cont` (exact semantics of
the S0/FEC-S1 descriptor).  Fixed coarse anchors, not a learned fine structural
representation.  No additional handcrafted topology descriptors.

## 10. Environment decoder

```text
z_i = [m_i^V (64); m_i^E (16); t_i (6)]  in R^86
E_i = SiLU-MLP : 86 -> 329 -> 48        in R^48
```

No extra patch_cont encoder, no local token, no dense structural adapter.

## 11. Static composition backend (exact FEC-S1 backend, frozen)

| component | params | specification |
|---|---:|---|
| pair projection | 768 | `W_P: Linear(48, 16, bias=False)`, `u_i = W_P E_i` |
| relation encoder | 1360 | `_MLPBlock(23, 32, 16)` |
| distance gate | 80 | `Embedding(5, 16)` |
| pair encoder | 5328 | `_MLPBlock(64, 64, 16)`, applied **once** per pair |
| global encoder | 3136 | `_MLPBlock(62, 32, 32)` |
| topology encoder | 552 | `Linear(25,16) ReLU Linear(16,8)` |
| reader | 4135 | `GenericReader(302, (13,13))` |
| **subtotal** | **15359** | |

Pair representation (computed once per pair, never written back into E):

```text
[ u_i + u_j ; |u_i - u_j| ; u_i * u_j * gate(d_ij) ; r_ij ]
r_ij in R^23 : pure-topology relation + path bond composition + adjacent bond type
```

### Graph readout

```text
unary  : sum(E), sum(E^2), log1p(n)                    = 97
pair   : 5 distance buckets x [sum(q), sum(q^2), log1p(n_d)] = 165
global : 62 -> 32 -> 32   (30D topology + 28D atom + 4D bond zeroth-order) = 32
topology: 25 -> 16 -> 8                                 = 8
R      : 302 ; reader 302 -> 13 -> 13 -> 1
```

## 12. Parameter identity (hard gate G10)

```text
Sparse dictionary   65*32               = 2 080
W_R + W_C + S                           = 4 096
W_B + P                                 =   160
environment MLP 86->329->48             = 44 463
local total                             = 50 799
backend                                 = 15 359
TOTAL                                   = 66 158
```

FEC-S1 historical: 66 170.  Difference −12 (0.018%).  Must be re-derived by
code.  Any mismatch: STOP, locate; do not fix by changing widths.

## 13. Matched control: DenseTied-32 (the only control this round)

* NO PCA control (FEC-D1 already tested PCA; the question here is sparse tied
  coding vs dense tied coordinate).
* `P in R^{65x32}` (2,080 params), `P_init = exact same SDB K-SVD matrix as
  D_init`, identical column normalization.  Encoding
  `z_v = phi_v Pbar` (no IHT, no top-s, no threshold, no sparsity);
  reconstruction `hat_phi_v = z_v Pbar^T`; `r_v = [1; z_v]`; identical
  `L_task + lambda_rec * L_rec`.

## 14. Two formal arms — everything matched

```text
S — SparseDictEnv : phi -> tied-IHT(D), exact top-8
D — DenseTiedEnv  : phi -> phi @ P
```

Both: **total params 66,158; same seed (0); same initial shared weights;
same D/P initialization; same data order; same optimizer (Adam 1e-3, wd 1e-5,
batch 128, clip 5); same lambda_rec; same backend; same reader; same chemistry
inputs; same structural input phi65.**  G11: all shared parameter tensors
bit-identical at step 0; Sparse D and Dense P bit-identical initial values.

Protocol: `OPTIMIZED_PROTOCOL` inherited — Adam, lr 1e-3, wd 1e-5, batch 128,
L1, grad clip 5, no scheduler, 240 epochs, no early termination, fixed Top-5
soup.  Seed 0.  Official train 10 000 / official valid 1 000; official test
**never** (test loader raise-guarded).

## 15. Correctness gates G0..G12 (all must pass before any formal run)

| gate | requirement |
|---|---|
| G0 | `phi65` bit-identical to the FSAR/SDB frozen artifact (max_abs = 0) |
| G1 | chemistry purity: relabelling atom/bond categories, topology fixed ⇒ phi, Sparse alpha, Dense z unchanged |
| G2 | exact sparsity: `l0(alpha_v) <= 8`; near-1 fraction of nodes with exactly-8 support |
| G3 | gradient to dictionary: real train batch, MAE alone (rec term off) ⇒ `grad_D != 0` |
| G4 | tied reconstruction: perturbing D/P changes both code and reconstruction; no independent learned encoder exists |
| G5 | no local dense bypass: zeroing the dictionary coordinate leaves only DC chemistry marginal + bond + 6 scalars in E; no hidden fine structural state |
| G6 | environment freeze: mutating pair relation leaves E_i bit-identical |
| G7 | no pair→centre: monkeypatching any centre-update/pool-to-centre path to raise; forward succeeds |
| G8 | once-only composition: pair encoder executes exactly its frozen expected count per forward; no second pass |
| G9 | relabel invariance: graph node relabel ⇒ prediction invariant within tolerance |
| G10 | parameter identity: Sparse == Dense == 66,158 (or audited code value) |
| G11 | initialization matching: all shared tensors bit-identical at step 0; D == P init |
| G12 | test blocker: any official test split access raises |

Any failure: STOP (no workaround).

## 16. Stage 1 — train-only mechanism smoke (no small-data performance)

Fixed: first 512 train molecules, 3 epochs max, Sparse arm only.  Official
valid not read.  Checks (loss finite; train MAE decreases; rec finite;
`grad_D != 0`; dictionary not immediately collapsed; alpha still sparse; atom
usage not all-zero/single-atom; environment variance nonzero) plus the hard
mechanism gate:

```text
>= 24 / 32 atoms active ; no single atom > 50 % of nonzero assignments ;
effective atom count >= 8 ; environment effective rank > 1
```

Failure ⇒ `E2E_DICTENV_MECHANISM_COLLAPSED`, STOP.  No tuning of lambda/K/s
at this stage.

## 17. Formal training

After Gate 0 PASS and Stage-1 PASS, immediately full data: train 10 000,
valid 1 000, seed 0, protocol §14, both arms.  No LR/wd/K/s/IHT/rank/width/
epoch sweeps, no scheduler.

## 18. Primary metrics and gates

Both arms report: best valid MAE, best epoch, Top-5 soup valid MAE, train MAE
at best, train minimum, reconstruction train/valid, dictionary/P movement,
wall time, peak GPU memory.

```text
M_S = SparseDictEnv soup      M_D = DenseTiedEnv soup      G_sparse = M_D − M_S
historical anchors (context only): FEC-S1 seed0 soup 0.130422, best 0.136783, S0 soup 0.140794
report M_S − 0.130422, but never treat FEC-S1 as a matched comparison
```

### Mechanism interventions (evaluation-only, trained Sparse soup)

* **A — dictionary-coordinate neutralization**: keep DC `1`, chemistry, bond,
  six scalars, backend; set `alpha_v -> 0` (`r_v = [1; 0...0]`).  `M_zero`;
  `G_dict-use = M_zero − M_S`; **required ≥ 0.010**.
* **B — assignment shuffle**: within each (root, shell), permute `alpha_v`
  against `q_v` (node set, shell, alpha multiset, q multiset, DC marginal,
  bond branch, global context unchanged); 5 fixed permutations; mean `M_shuffle`;
  `G_assign = M_shuffle − M_S`; **required ≥ 0.010**.
* **C — dictionary health** (Sparse soup/final): active atoms train & valid
  (≥ 24/32), effective atom count ≥ 8, support entropy, top-1/top-8 support
  mass, coefficient magnitude distribution, per-atom molecule coverage,
  train-valid usage Spearman, dictionary effective rank, coherence,
  `D` movement from K-SVD init (> numerical noise), reconstruction error,
  task gradient to D nonzero.

## 19. Performance bands (Sparse absolute)

```text
Strong            : M_S <= 0.135
Viable but weaker : 0.135 < M_S <= 0.145
Weak              : M_S > 0.145
```

## 20. Dictionary-specific gate (the core gate)

```text
G_sparse = M_D − M_S  >=  0.003   (required)
```

Without it: no Sparse absolute result may be interpreted as dictionary-specific
success.

## 21. Frozen verdict table (precedence order)

| case | condition | verdict |
|---|---|---|
| A | `M_S <= 0.135` and `G_sparse >= 0.003` and `G_dict-use >= 0.010` and `G_assign >= 0.010` and health PASS | `E2E_DICTENV_STRONG_DICTIONARY_CORE_SUPPORTED` |
| B | `0.135 < M_S <= 0.145` and `G_sparse >= 0.003` and zero/shuffle/health PASS | `E2E_DICTENV_DICTIONARY_SPECIFIC_BUT_ABSOLUTE_WEAK` |
| C | `M_S <= 0.145` and `G_sparse < 0.003` | `E2E_DICTENV_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC` |
| D | `M_S > 0.145` | `E2E_DICTENV_ABSOLUTE_WEAK` |
| E | Sparse better than Dense but zero/shuffle/health FAIL | `E2E_DICTENV_GAIN_NOT_DICTIONARY_MEDIATED` |
| S1 | Stage-1 mechanism gate FAIL | `E2E_DICTENV_MECHANISM_COLLAPSED` |

Case A does **not** authorize seed 1 or any control automatically; it only
records that a *new* round (e.g. E2E-DictEnv-C1, a ~2080-param generic
`65->21->32` SiLU structural encoder at the same placement) is now
scientifically justified under a new pre-registration.  Generic control is
defined for the future, not implemented this round.

## 22. Forbidden rescues (all cases)

```text
K=64 / K=16 / s=4 / s=16 / more IHT steps / LISTA / soft dictionary /
multiple dictionaries / edge dictionary / per-shell dictionaries /
deeper environment decoder / attention / LayerNorm rescue /
task-specific lambda sweep / extra epochs / reader enlargement /
FEC-S1 residual / PCA side branch / message passing / recurrence /
seed 1
```

DenseTied is not a "bad baseline": if `DenseTied < Sparse`, the correct reading
is that the structural-role × chemistry architecture may be right while
sparsity is not the winning parameterization.  Absolute success without
specificity is still "dictionary not specific".  Mechanism and performance are
separated in every report.

## 23. Durable artifacts (minimum set)

```text
tracks/ksvd/notes/e2e_dictenv_v0_{prior_artifact_audit,preregistration,implementation,analysis}.md
tracks/ksvd/results/e2e_dictenv_v0/
    artifact_identity.json  correctness.json  parameter_accounting.json
    lambda_calibration.json  smoke_gate.json
    sparse_seed0.json  dense_tied_seed0.json
    sparse_curve.csv  dense_tied_curve.csv
    dictionary_health.json  mechanism_zero.json  mechanism_shuffle.json
    states/  REPORT.md  DECISION.md  decision.json
```

JSON/PT/CSV follow the existing repo-wide gitignore convention (no force-add).
Plus claim YAML, decision YAML and `STATE.yaml` update.

## 24. Final report must answer Q1..Q8

1. Architecture purity (no MP/recurrence/writeback/FEC-S1 bypass)?
2. Dictionary centrality (`E`'s fine structure only from `IHT(D, phi)`)?
3. End-to-end coupling (task gradient to D; movement from K-SVD init)?
4. Structural anchoring (still sparse / reusable / reconstructive /
   non-collapsed)?
5. Absolute band of `M_S`?
6. `M_D − M_S >= 0.003`?
7. zero-code `Δ >= 0.010`; assignment shuffle `Δ >= 0.010`?
8. Verdict from the six allowed verdicts?
