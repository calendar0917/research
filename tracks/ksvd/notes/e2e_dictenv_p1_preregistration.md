# E2E-DictEnv-P1 — pre-registration (frozen before any implementation run)

Round **E2E-DictEnv-P1** · protocol `e2e_dictenv_p1` · study `zinc-context-gap`.
Prior-artifact audit: [`e2e_dictenv_p1_prior_artifact_audit.md`](e2e_dictenv_p1_prior_artifact_audit.md).

This is the frozen specification.  It is written **before** any P1
implementation, cache build, training or GPU use.  Any deviation requires a
written amendment recorded here before the affected run.

---

## 1. The single scientific question

> If every fine local structure–chemistry assignment must pass through one
> shared, task-coupled sparse structural dictionary, and all handcrafted
> structure×chemistry statistics are removed, can the model still reach a
> competitive ZINC predictive performance and show that the dictionary is the
> load-bearing organizer of the local chemical environment?

T1 already produced a strong absolute result (`M_S = 0.125765`) but with the
146-D FEC-S0 coarse descriptor (`patch_cont`) routed directly into the
environment decoder, so the dictionary became auxiliary and the pre-registered
assignment-shuffle gate failed.  P1 deliberately gives that up and tests the
clean hypothesis.

---

## 2. Claim discipline

Only this claim may be supported:

> A task-coupled sparse structural dictionary is the load-bearing organizer of
> local chemical context: primitive chemistry enters directly only as
> zeroth-order identity/mass, while local structure–chemistry assignment must
> pass through dictionary structural roles.

The claim "the dictionary helps a strong handcrafted environment" is **not**
testable here and may not be reported.

---

## 3. Hard architectural purity (forbidden tensors and paths)

None of the following may enter the local environment network:

```text
patch_cont (146-D)
atom_shell (84-D) / bond_shell (24-D) / any shell x chemistry histogram
root-conditioned atom-type or bond-type histograms
typed_token / parent_token
FEC-S1 146 -> 214 -> 24 adapter
PCA32 / DenseRole / FEC-D1 residual side branches
B-Null / B-Full local hidden state
```

Source-level gates enforce this: the model module must never name
`patch_cont`, `atom_shell` or `bond_shell` (`G5`), and the forward must be
bit-identical when `patch_cont` is poisoned with NaN.

---

## 4. Strict no-message-passing

```text
NO message passing
NO recurrence
NO pair->centre
NO context writeback
NO attention
NO second environment pass
```

For every root `i`, `E_i` is formed once from root-local primitives and then
frozen.  The pair composer may read only `E_i`, `E_j`, `rho_ij`; it may never
produce `E_i'`.  Enforced by `G9` (environment frozen under pair mutation),
`G10` (pair→centre monkeypatch raises, forward still succeeds) and `G11`
(pair/relation/env encoders called exactly once).

---

## 5. Dictionary substrate and core (frozen)

```text
phi_v        = exact FSAR-R2-AR0 R^65 pure-topology coordinate (reused)
K            = 32
s            = 8
IHT steps    = 10  (tied unrolled IHT, exact top-s)
D_init       = exact SDB-v0 K-SVD artifact (65x32, sha256 925d573a5808...)
Dbar_k       = D_k / max(||D_k||_2, eps)
alpha_v      = IHT_{s=8}(Dbar, phi_v)
hat_phi_v    = alpha_v Dbar^T
D            requires_grad = true
```

No K/s/IHT/LISTA/dictionary-count/edge-dictionary/per-shell-dictionary change
is authorized.  `||alpha_v||_0 ≤ 8` and `||∂L_task/∂D|| > 0` on a real MAE
backward are hard gates (`G2`, `G3`).

---

## 6. Reconstruction anchor (frozen)

```text
L_rec = E_v ||phi_v - hat_phi_v||^2 / (||phi_v||^2 + eps)
L     = L_MAE + lambda_rec * L_rec
lambda_rec = 33.95873017865987            (frozen T1 winner; no sweep)
```

---

## 7. Local primitive anchor `p_i in R^62` (zeroth order only)

```text
root identity    q_i                        28
patch atom mass  sum_{v in P_i} q_v         28   (NOT grouped by shell)
patch bond mass  sum_{e in E_i} b_e          4   (NOT grouped by shellpair)
size             [log1p|V_i|, log1p|E_i|]    2
------------------------------------------------
p_i                                          62
```

Official-train-only per-coordinate standardizer.  Gate `G6` recomputes `p_i`
from raw primitives and matches the stored value, and confirms no
shell-conditioned chemistry is present.

---

## 8. Node dictionary–chemistry binding (frozen)

```text
d_A = 96
W_A^S in R^{32x96}, W_A^C in R^{28x96}       (bias-free)
u_v^A = ( (alpha_v W_A^S) o (q_v W_A^C) ) / sqrt(96)
A_{i,s} = sum_{v : s_iv = s} u_v^A  in R^96
A_i = [A_{i,0}; A_{i,1}; A_{i,2}]   in R^288
```

No extra MLP, no attention, no early shell mixing.  Shell is a routing index
only.  Gate `G7` confirms edge-role symmetry; node shuffling is intervention 2.

---

## 9. Edge dictionary-role × bond-chemistry binding (frozen)

```text
d_E = 48
g_uv = [alpha_u + alpha_v ; |alpha_u - alpha_v| ; alpha_u o alpha_v]  in R^96
W_E^S in R^{96x48}, W_E^C in R^{4x48}        (bias-free)
u_uv^E = ( (g_uv W_E^S) o (b_uv W_E^C) ) / sqrt(48)
E_{i,p} = sum_{e : p_i(e) = p} u_uv^E  in R^48     (6 shellpair slots)
E_i^edge = [E_{i,0}; ... ; E_{i,5}]   in R^288
```

The historical `bond_shell` block is deleted.  Shellpair is a routing index
only.

---

## 10. Local environment input and decoder (frozen)

```text
z_i = [ p_i^62 ; A_i^288 ; E_i^{edge,288} ]   in R^638
E_i = W_2 SiLU(W_1 z_i)                       638 -> 102 -> 48
```

No LayerNorm, no dropout, no residual local bypass, no extra layer, no
attention.

---

## 11. Pair relation and composition (frozen)

Only the pure-topology relation is used:

```text
P1_RELATION_INDICES = 0:14 and 18            (15-D)
relation encoder = _MLPBlock(15, 32, 16)
```

Pair composition (unchanged static FEC-S1 family):

```text
u_i = W_P E_i in R^16        (bias-free)
inputs = [u_i+u_j ; |u_i-u_j| ; u_i o u_j o gate(d_ij) ; relation]
pair encoder = _MLPBlock(64, 64, 16)
distance gate = Embedding(5,16)
```

Pair chemistry may enter **only** through the edge environment binding
`(alpha_u, alpha_v) × bond type`.  `G8` confirms the pair relation never reads
`path_bond_mean` / `adjacent_bond_type`.

---

## 12. Global / topology / reader (frozen, unchanged)

```text
global context = 30-D pure topology + 28-D whole-molecule atom marginal
                 + 4-D whole-molecule bond marginal        (62 -> 32 -> 32)
graph topology = 25 -> 16 -> 8
reader         = GenericReader(302, (13, 13))
```

These are zeroth-order whole-molecule composition/mass statistics and pure
topology; they do not encode local structural assignment and may not write back
to `E_i`.

---

## 13. Single architecture candidate

Exactly one architecture, `P1_PRIMITIVE_ONLY_DICT_CORE`, no sweep:

```text
local params    82762
backend params  15103
whole model     97865        (hard budget 90000..105000)
```

`G13` checks the actual parameter count equals the audit and lies in the
budget.  If out of budget: STOP.

---

## 14. Stage 0 correctness gates (all PASS before any formal run)

```text
G0  phi65 identity vs fresh build_phi (bit-identical)
G1  dictionary identity: shape / file sha256 / tensor bytes / column norms
G2  exact top-s: max l0 <= 8, exact-8 fraction >= 0.99
G3  real MAE backward with reconstruction OFF gives ||grad_D|| > 0
G4  chemistry purity: chemistry relabelling leaves phi/alpha/incidence topology
    unchanged
G5  forbidden local descriptors absent: AST + NaN-poisoned patch_cont forward
    is bit-identical
G6  primitive anchor: recompute from raw primitives; no shell-conditioned
    chemistry
G7  edge role symmetry g_uv == g_vu
G8  pair-relation chemistry blocker: 14:18 + 19:23 randomisation leaves the
    sliced relation identical and the prediction bit-identical; no
    path_bond_mean / adjacent_bond_type name in the model
G9  environment freeze: E bit-identical under pair-relation mutation
G10 no pair->centre: monkeypatched raise, forward succeeds
G11 once-only composition: pair/relation/env encoders called exactly once
G12 relabel invariance: node relabelling leaves the prediction invariant
G13 parameter budget: 90000 <= actual == accounted <= 105000
G14 official test blocker: test access raises; encoded cache unflagged
```

Any FAIL: STOP.

---

## 15. Stage 1 train-only smoke (mechanism only)

```text
first 512 official-train molecules, seed 0, 3 epochs, valid never read
```

Required:

```text
loss finite; MAE finite and decreasing; reconstruction finite
task gradient -> D nonzero
exact top-8
active atom retention >= 80% of initial
effective atom count >= 8
max atom share <= 0.50
environment effective rank > 1
no NaN/Inf
node dictionary-binding gradients nonzero
edge dictionary-binding gradients nonzero
```

If the edge branch is zero-gradient: STOP; no post-hoc direct bond feature.

---

## 16. Formal Stage 2 — Sparse seed 0 (single main run)

```text
official train 10000 / official valid 1000
seed 0, 240 epochs, Adam, lr 1e-3, weight_decay 1e-5, batch 128, grad clip 5
no scheduler
L1 + 33.958730 * reconstruction
fixed Top-5 equal-weight soup over official-valid MAE
```

No tuning of lr/wd/batch/epochs/lambda/K/s/IHT/width.  Test forbidden.
`M_S` = the fixed Top-5 soup official-valid MAE.

---

## 17. Sparse seed-0 bands

```text
strong   M_S <= 0.135
viable   0.135 < M_S <= 0.145
weak     M_S > 0.145
```

P1 does not have to beat T1's `0.125765` (T1 used the removed bypass).

---

## 18. Mechanism interventions (evaluation only, on the Sparse soup)

### 18.1 complete dictionary zero

`alpha_v -> 0` for **both** the node and edge branches; anchor, global context,
pure topology and reader kept.  `M_zero`, `G_zero = M_zero - M_S`.

### 18.2 node assignment shuffle

Within every `(root, shell)`, permute `alpha_v` relative to `q_v`; `alpha`
multiset, `q` multiset, shell, anchor, edge branch, global context preserved;
5 fixed permutations.  `M_node`, `G_node = M_node - M_S`.

### 18.3 edge assignment shuffle

Within every `(root, shellpair)`, permute the structural edge-role endpoints
`g_uv` relative to `b_uv`; the `g` multiset and bond-type multiset preserved;
5 fixed permutations.  `M_edge`, `G_edge = M_edge - M_S`.  **Report only.**

### 18.4 combined shuffle

Both of the above simultaneously (node seed and edge seed fixed per
permutation).  `M_all`, `G_all = M_all - M_S`.

---

## 19. Dictionary health (final Sparse soup)

Report active atoms (train/valid), effective atom count, support entropy, max
support share, coefficient magnitudes, usage Spearman train-valid, effective
rank, coherence, `D` movement from K-SVD init, train/valid reconstruction and
the task gradient to `D`.  Required:

```text
active retention >= 80% initial
effective atom count >= 8
max support share <= 0.50
exact top-8
task gradient nonzero
D movement > numerical noise
valid reconstruction <= 0.01
```

---

## 20. Sparse seed-0 GO gate (all must hold)

```text
M_S <= 0.145
G_zero >= 0.030
G_node >= 0.010
G_all  >= 0.015
dictionary health PASS
```

If any fails: STOP.  No primitive-anchor change and no `patch_cont` rescue.

---

## 21. DenseTied matched control (conditional)

If GO: train `DenseTied-P1`, structurally identical, the only difference being

```text
Sparse: alpha_v = IHT(D, phi_v)
Dense : z_v     = phi_v Dbar
```

with `P_init == D_init`, same column normalization, same tied reconstruction,
same λ, same atom/edge binding, same decoder, same pair composer, same
global/readout and same parameter count.

Seed-0 dictionary-specificity:

```text
G_specific^(0) = M_D^(0) - M_S^(0) >= 0.003
```

If it fails: verdict `P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC`,
STOP.  No seed 1.

---

## 22. Seed-1 confirmation (conditional)

If seed-0 specificity PASS: freeze the config and run paired Sparse seed 1 and
DenseTied seed 1 with identical batches, init conventions, horizon, optimizer,
λ, architecture.  Define `G_specific^(1) = M_D^(1) - M_S^(1)`.

---

## 23. Specificity conclusion

```text
replicated   G_specific^(0) >= 0.003 AND G_specific^(1) >= 0.003
directional  both > 0 but one < 0.003
unstable     sign flip  ->  DICTIONARY_SPECIFICITY_UNSTABLE
```

Seed 0 may not override a seed-1 sign flip.

---

## 24. Final verdicts (only these)

| id | condition |
|---|---|
| `P1_STRONG_CLEAN_DICTIONARY_CORE_SUPPORTED` | `M_S <= 0.135`, zero/node/all gates, health, both specificities ≥ 0.003 |
| `P1_VIABLE_CLEAN_DICTIONARY_CORE_SUPPORTED` | same but `0.135 < M_S <= 0.145` |
| `P1_CLEAN_CORE_MECHANISM_SUPPORTED_ABSOLUTE_WEAK` | mechanism + health strong, `M_S > 0.145` |
| `P1_CLEAN_ENVIRONMENT_VIABLE_BUT_DICTIONARY_NOT_SPECIFIC` | absolute+mechanism ok, matched dense not worse by ≥ 0.003 |
| `P1_DICTIONARY_NOT_LOAD_BEARING` | `G_zero < 0.030` or `G_node < 0.010` or `G_all < 0.015` |
| `P1_MECHANISM_COLLAPSED` | health / task coupling itself fails |
| `P1_DICTIONARY_SPECIFICITY_UNSTABLE` | seed-0/seed-1 specificity sign flip |

Precedence: mechanism collapse → not load-bearing → absolute weak → not
specific → strong/viable.

---

## 25. Test policy

Official test has been opened by historical rounds and is **not**
project-pristine.  It must not be used for architecture design, GO/STOP, seed
selection, checkpoint selection or any hyper-parameter change.  Only after the
Sparse/Dense seed-0 pair, the authorized seed-1 pair, all mechanisms and the
frozen configuration are recorded may a single dedicated terminal stage read
the test once and report.  If P1 fails before the GO gate, the test is **not**
opened.

---

## 26. Forbidden rescues (regardless of how close)

```text
NO patch_cont ; NO atom_shell ; NO bond_shell ; NO direct incident-bond
histogram ; NO pair bond-chemistry relation ; NO local GNN ; NO MP ;
NO recurrence ; NO attention ; NO K/s sweep ; NO lambda sweep ;
NO width sweep ; NO extra epochs ; NO new dictionary ; NO edge dictionary ;
NO T1 residual branch
```

---

## 27. Durable artifacts

```text
notes/e2e_dictenv_p1_{prior_artifact_audit,preregistration,implementation,
                      analysis,test_read}.md
results/e2e_dictenv_p1/
    artifact_identity.json  parameter_accounting.json  correctness.json
    smoke_gate.json  sparse_seed0.json  sparse_seed0_curve.csv
    dictionary_health.json
    mechanism_zero.json  mechanism_node_shuffle.json
    mechanism_edge_shuffle.json  mechanism_all_shuffle.json
    gate_sparse_seed0.json
    dense_seed0.json            (conditional)
    specificity_seed0.json
    sparse_seed1.json  dense_seed1.json  specificity_seed1.json (conditional)
    architecture_freeze.json
    official_test_unlock.json  official_test_results.json   (conditional)
    REPORT.md  DECISION.md
plus claim YAML, decision YAML and the STATE.yaml update.
```
