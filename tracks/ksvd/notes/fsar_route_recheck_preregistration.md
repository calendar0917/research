# FSAR route-recheck pre-registration — clean binding null + explicit local structure

Branch `exp/fsar-route-recheck-binding-explicit-zinc`, created from the FSAR-v2
verdict HEAD `6df1f98`.  Official ZINC test is never loaded
(`official_test_loaded = false` in every JSON produced by this round).

This is **not** FSAR-v3 and **not** a continuation/optimization of FSAR-v2.  It
re-poses two questions that FSAR-v1/v2 left entangled with other failures
(binding vs generic capacity; explicit vs latent local structure) and fixes the
conceptual/control defects those rounds exposed.  Everything below is frozen
*before* any formal run is launched.

## 0. Questions

```
Q1  Is the FSAR-v1 SAB advantage stable across seeds?
Q2  Under a genuinely operator-matched, active-capacity-matched null,
    does aligned binding still carry an increment?
Q3  Can the latent topology-GNN local S be replaced by an explicit local
    structural basis while keeping most of the performance?
```

The three questions are independent; Wave 2 runs one job for Q2 and one for Q3
in parallel by design.

## 1. Route corrections (what this round changes in the *interpretation*)

### 1.1 FSAR-v1 is the current factorized main baseline

```
FSAR-v1 seed0 soup:  A 0.157917   SA 0.151617   SAB 0.130913   SAM 0.134182
matched seed0 refs :  B-Bag 0.127382   A2 0.121694   B-Full 0.119818
```

FSAR-v1 is **not** discarded because FSAR-v2 degraded.  FSAR-v1 `A` and `SAB`
remain the reference factorized architecture of this round.

### 1.2 The correct conclusion from FSAR-v2 `A_exact`

Do **not** claim "the complete attribute marginal is not the bottleneck".
Record instead:

> An exact raw categorical marginal followed by the tested count-MLP interface
> did not improve FSAR and degraded the whole nested family.

```
raw marginal collision-free  !=  learned A representation lossless / optimally usable
```

`A_exact` is **not** retrained this round (§25).

### 1.3 Current A is not topology-free A

Every FSAR mode shares a topology-only pair relation `R_S` and a topology-only
global hinge channel `G_S`.  Therefore all reporting below uses the honest
conditional notation

```
A     | R_S, G_S
A+S   | R_S, G_S
A+S+B | R_S, G_S
```

and never "pure attribute-only model".  Any "A is weak" reading is conditional
on `R_S, G_S`.

## 2. The three explicit scales of topology already present

| scale | symbol | existing content | nature |
|---|---|---|---|
| global | `G_S` | `topology_features`: cycle rank, exact cycle spectrum, longest simple cycle, minimum cycle-basis statistics, fixed hinge functions | explicit |
| pair | `R_S` | graph distance, distance bucket, patch overlap/relative size, boundary overlap, shortest-path count | explicit |
| local | `S_v` | FSAR-v1 topology-only GNN role state | **latent** |

Q3 replaces *only* the local term: `S_v^latent -> S_v^explicit`, holding `R_S`
and `G_S` bit-identical.

## 3. A is frozen at FSAR-v1

All new implicit/explicit comparisons use the FSAR-v1 `A`:

```
A_v = [ A_v^self , A_v^ctx ]
A_self = learned centre atom semantics (E_atom(x_v), rooted)
A_ctx  = learned permutation-invariant atom/bond context marginals (mean/std)
```

`A_exact`, the raw-count MLP and the 64-D exact marginal are **forbidden** this
round (§25, §34), so Q3 does not inherit the FSAR-v2 A failure.

## 4. Aligned binding (frozen, unchanged from FSAR-v1)

Node side, with `s_i` the structural role coordinate and `a_i` the learned
attribute semantic:

```
p_i = U_S(s_i - mean s)      q_i = U_A(a_i - mean a)
z_i = p_i ⊙ q_i
m_align = mean_i z_i         σ_align = std_i z_i
```

Edge side analogous (`edge_role_mlp`, `edge_attribute_mlp`, endpoint
projections).  Final:

```
B_align = F_B[ m_n, σ_n, m_e, σ_e ]
```

For the explicit route `s_i` is the read-only explicit coordinate `φ_i` and
`U_S -> U_φ` (node) / `edge_role_mlp(φ^E)` (edge); the chemistry side `q_i` is
the **same** FSAR-v1 learned entity semantics (§18).

## 5. Assignment-free analytical null `B_indep`

Given the same `p_i`, `q_i` (hence the same projections and MLPs), the null uses
only the assignment-independent moments.  Because `mean(p)=mean(q)=0` exactly,

```
E_π[ mean_i (p_i ⊙ q_{π(i)}) ]        = 0
E_π[ mean_i (p_i^2 ⊙ q_{π(i)}^2) ]    = mean_i(p_i^2) ⊙ mean_i(q_i^2)
```

so

```
m_indep    = 0
σ_indep    = sqrt( mean(p^2) ⊙ mean(q^2) + ε )
B_indep    = F_B[ 0, σ_n^indep, 0, σ_e^indep ]
```

Identically for the explicit route with `φ` in place of `s`.

### 5.1 Exact operator / active-parameter matching

`B_indep` reuses the *same module objects* as `B_align`:

```
atom_embedding, bond_embedding, atom_mlp, bond_mlp,
structural role encoder / explicit basis,
node_role_projection, node_attribute_projection,
edge_role_mlp, edge_attribute_mlp,
edge_role_projection, edge_attribute_projection,
binding_fuse
```

Same parameters, same active parameters, same projection dimensions, same
second-order multiplicative machinery.  The only missing information is *which
attribute is paired with which structural role*.  No extra MLP is created for
the null.

### 5.2 Pre-declared structural note on the null's first-moment block

`m_indep ≡ 0` **by the theorem above**, not by choice: any assignment-independent
first moment of the centred product is exactly zero.  Consequently the
`binding_fuse` input columns that multiply the `m` blocks are structurally
zero-gradient in the null, and only there.

This is pre-registered and is **not** the FSAR-v2 failure mode (an entire unused
semantic module outside the prediction path).  Under the null:

* every B *module* (7 modules above) receives a nonzero task gradient through
  `σ_indep`;
* every atom/bond embedding, atom/bond MLP, `node_init`, relation core and head
  module receives a nonzero task gradient;
* the ONLY zero-gradient parameters are the exactly-two `binding_fuse[0]`
  weight column blocks `[:, 0:B]` and `[:, 2B:3B]` (2·B·hidden elements), which
  is the analytical first-moment channel being set to its independent value 0.

Accordingly `active_prediction_path_params` is defined **at module
granularity** (parameters in modules that receive a nonzero task gradient), and
`B_align` and `B_indep` must match on it exactly.  The exact zero-gradient
element count is also reported for transparency.  This definition directly
addresses the FSAR-v2 defect (total params matched but whole modules unused).

## 6. New modes

| mode | local S | B | meaning |
|---|---|---|---|
| `A` | none | none | FSAR-v1 |
| `SA` | latent | none | FSAR-v1 |
| `SAB` | latent | `B_align` | FSAR-v1 aligned |
| `SAM` | latent | MLP capacity control | FSAR-v1 historical control |
| `SABI` | latent | `B_indep` | implicit assignment-free null |
| `SAE` | explicit | none | explicit local S |
| `SABE` | explicit | `B_align^explicit` | explicit S + aligned |
| `SABEI` | explicit | `B_indep^explicit` | explicit S + assignment-free null |

Everything else (node init, topology-only pair relation `R_S`, `T=2` recurrence,
topology hinge `G_S`, pooling, head, optimizer, schedule) is identical across
all modes.

## 7. Explicit local structural basis (chemistry-free, no message passing)

Node coordinate `φ^node_{vu}` (11-D, `log1p` on integer counts):

```
1[u=v]; shell one-hot d(v,u)=0,1,2; log1p(indeg_{W_v}(u));
log1p(|N(u) ∩ shell_j(v)|) j=0,1,2; log1p((A)_{vu}), log1p((A^2)_{vu}), log1p((A^3)_{vu})
```

Edge coordinate `φ^E_{v,e}` (15-D):

```
one-hot unordered endpoint shell pair (6);
log1p(deg-sum); log1p|deg-diff|; log1p(common neighbours);
log1p(shell-neighbour sum_j), log1p|shell-neighbour diff_j| for j=0,1,2
```

No chemistry is read (test `explicit_basis_chemistry_purity`); the explicit path
contains **no adjacency message passing** (AST test + `message is None`).

```
S_v^explicit = F_S( φ_root , mean φ_n , std φ_n , mean φ_e , std φ_e )
             : Linear(63,32) -> SiLU -> Linear(32,32)      -> 32-D
```

Same 32-D output width as the latent `S`, so node-init widths are comparable.

## 8. Forbidden this round

```
A width sweep; A_exact repair; new DeepSets variants; Transformer; attention;
adaptive motif; BCE; support selection; path-level B2; radius sweep; optimizer
sweep; distillation; teacher; auxiliary binding loss.
```

No `A_exact` / `SA_exact` / `SAB_exact` / `SAM_exact` formal training (FSAR-v2
outputs stay as negative/interface evidence).  No A-gate may block `SAE`/`SABE`.

## 9. Frozen interventions are diagnostics, not causal estimates

`B -> 0`, attribute-assignment shuffle and `S -> 0` remain available for
`SAB` / `SABE`, but the report must write:

> These are dependency / reliance diagnostics, not additive causal estimates of
> MAE contribution.

The binding-specific evidence is the **retrained matched** comparison
`B_align` vs `B_indep` on paired seeds.

## 10. Pre-registered gates / verdict bands

* operator/active gate: `#params(B_align) == #params(B_indep)`; all 7 B modules
  and every embedding/MLP/init/relation/head module receive nonzero gradient in
  both aligned and independent conditions.  Otherwise **STOP — matched control
  invalid**.
* correctness: same-marginal invariance
  `B_indep(P_1) == B_indep(P_2)` to tolerance for same structure / root /
  attribute multiset with different assignment; aligned sensitivity
  `B_align(P_1) != B_align(P_2)`; shared chemistry tensor and shared structural
  basis tensor across align/null; full-output relabel invariance.
* binding verdict (implicit), per seed: `Δ_B^latent = MAE(SABI) - MAE(SAB)`.
  Claimable only if both seeds have `Δ_B^latent > 0` and the mean `>= 0.001`.
* explicit binding verdict: `Δ_B^explicit = MAE(SABEI) - MAE(SABE)`, direction
  stable.
* explicit-S cost bands vs latent `SAB`:
  * near parity `SABE <= SAB_latent + 0.002`
  * moderate cost `+0.002 < gap <= +0.005`
  * large cost `gap > 0.005` — at most "current fixed explicit basis is
    insufficient to replace latent topology-GNN roles"; **never** "explicit
    structure is falsified".
* historical continuity: seed1 `SAB_legacy < SAM_legacy` replicates or not
  (`SAM` is a historical control only; final binding claims use
  `B_align` vs `B_indep`).

No linear decomposition of a gain into "X % is binding".  `A` is not required to
reach `B-Bag`: `B-Bag` already jointly mixes atom type with a root-relative
role/distance inside a nonlinear node representation, so it is not a pure
marginal comparator.

## 11. Parameter reporting

For every mode: `total params`, `trainable params`,
`active_prediction_path_params` (module-granularity, §5.2), `A params`,
`local-S params`, `B params`, `relation core`, `global topology encoder`,
`head`.  `B_align` and `B_indep` must have identical `total params` and identical
`active_prediction_path_params`.

## 12. Gradient audit (pre-training smoke, blocking)

For aligned and independent null separately:

```
atom embedding grad > 0; atom MLP grad > 0;
bond embedding / MLP grad > 0 when bonds present;
structural projection grad > 0; attribute projection grad > 0;
binding fuse grad > 0; node init grad > 0;
relation grad > 0; head grad > 0
```

If a null's semantic/projection module is inactive -> STOP, matched control
invalid.

## 13. Structural explicitness diagnostics (descriptive only)

For explicit-S: raw basis coordinate mean/std, nonzero frequency, range,
correlation/redundancy matrix; for the learned first projection: column norm per
explicit coordinate.  Purpose: answer "which explicit coordinates does the model
actually use?" — descriptive attribution, **not** a causal claim.

## 14. Execution protocol (frozen)

Training protocol inherited verbatim from the frozen optimized compact-v4 cell:
Adam, lr 1e-3, weight decay 1e-5, batch 128, grad clip 5.0, max 240 epochs,
patience 40, no scheduler, single stage, best-official-valid checkpoint, fixed
equal-weight Top-5 soup, `torch.use_deterministic_algorithms(True)`, single ZINC
L1/MAE task loss, seeds 0 and 1.  Official ZINC test never loaded.

Waves (all architecture implemented, tested and committed before Wave 1):

```
Wave 1  GPU0: FSAR-v1 SAB_legacy seed1      GPU1: FSAR-v1 SAM_legacy seed1
Wave 2  GPU0: SABI seed0                    GPU1: SAE seed0
Wave 3  GPU0: SABI seed1                    GPU1: SABE seed0
Wave 4 (conditional)  GPU0: SABEI seed0     GPU1: SABE seed1
        if SABE_seed0 <= SAB_implicit_seed0 + 0.005  or  SABE < SAE - 0.001
Wave 5 (conditional)  GPU0: SABEI seed1     GPU1: SAE seed1
        if Wave4 SABE_seed0 < SABEI_seed0 - 0.001 and the explicit binding
        intervention behaves normally; not bought if explicit seed0 is clearly
        > implicit SAB + 0.005 with no special mechanism advantage
```

Wave 1 uses the *unmodified* FSAR-v1 runner (`zinc_fsar.py`, modes `SAB`/`SAM`,
seed 1) so the historical architecture is bit-for-bit.  Waves 2-5 use the new
route-recheck runner.

GPU rule: never touch or co-tenant an unknown process; use only a genuinely free
GPU.  If only one GPU is free, the wave's two jobs run sequentially on it.
Multi-GPU parallelism is authorized only for our own independent jobs.

## 15. Files

* code: `experiments/luyin16/fsar_route_recheck.py`,
  `experiments/luyin16/zinc_fsar_route_recheck.py`
* tests: `tests/test_fsar_route_recheck.py`
* results: `results/fsar_route_recheck/` (gitignored)
* reused without copy: `fsar.py` (`A`, latent `S`, core, head, protocol),
  `fsar_v2.py` explicit-basis utilities, `zinc_fsar.py` (Wave 1),
  `zinc_fsar_v2.py` frozen eval helpers.

## 16. Local pre-flight recorded before the remote deploy

All architecture, tests and gates were implemented and committed **before** any
formal run.  Local CPU pre-flight (this commit):

* `tests/test_fsar_route_recheck.py`: 15/15 pass; inherited suites
  `test_fsar.py` + `test_fsar_v2.py` still 22/22 (37/37 together).
* bit-for-bit FSAR-v1 equivalence: `PatchPathFSARRouteModel(mode="SA"|"SAB"|"SAM")`
  has an identical state dict (every tensor `torch.equal`) and identical forward
  to `fsar.PatchPathFSARModel` at the same seed.
* parameter table (total / B block):

| mode | total | A | local S | B | M |
|---|---:|---:|---:|---:|---:|
| A | 56,617 | 5,264 | 0 | 0 | 0 |
| SA | 77,609 | 5,264 | 16,864 | 0 | 0 |
| SAB | 95,209 | 5,264 | 16,864 | 13,504 | 0 |
| SAM | 95,153 | 5,264 | 16,864 | 0 | 9,352 |
| SABI | 95,209 | 5,264 | 16,864 | 13,504 | 0 |
| SAE | 63,817 | 5,264 | 3,104 | 0 | 0 |
| SABE | 79,177 | 5,264 | 3,104 | 11,264 | 0 |
| SABEI | 79,177 | 5,264 | 3,104 | 11,264 | 0 |

* `#params(SAB) == #params(SABI)`, `#params(SABE) == #params(SABEI)`, equal
  `B` blocks, equal latent/explicit local-S blocks, equal relation core.
* gradient audit (`gradient_audit --all-modes`, 2 real optimizer steps then a
  fresh backward): every module active in every mode;
  `active_prediction_path_params == total params` for all modes
  (SAB 95,209 == SABI 95,209; SABE 79,177 == SABEI 79,177).  The null's only
  zero-gradient elements are the two `binding_fuse[0]` first-moment column
  blocks (2,048 elements for `b_dim = b_hidden = 32`), exactly as pre-declared
  in section 5.2.
* CPU smoke for `SAB`, `SABI`, `SAE`, `SABE`, `SABEI`: finite decreasing loss,
  non-constant output, all A/S/B channels alive, all modules active.

## 17. GPU availability observed at pre-registration time

| device | state | policy |
|---|---|---|
| GPU 0 | unknown task, ~35.8 GB, 97 % util | never touched, never co-tenanted |
| GPU 1 | a small unrelated active process (GSN reproduction, ~0.56 GB, ~13 %) | see below |

Because GPU 0 is occupied, only GPU 1 is available to this round.  GPU 1 also
hosts a small unrelated active process, so this round **records the co-tenant
explicitly** rather than treating the card as empty.  Policy actually used:

* GPU 0 is never touched;
* each wave's two independent jobs run as two separate `launch_remote.sh`
  processes on GPU 1, matching the FSAR-v2 execution regime and the brief's
  parallel intent (Wave 2 explicitly asks for `SABI` and `SAE` in parallel);
* the co-tenant memory / utilisation is re-checked immediately before each
  launch (abort and report if it grows beyond ~5 GB);
* all runs are deterministic, one process per job, no DDP.

## 18. Durable conclusion will be split into four independent answers

```
A. Historical replication : does FSAR-v1 SAB < SAM replicate at seed1?
B. Clean binding evidence : is SAB < SABI stable on paired seeds?
C. Structural explicitness : what does SABE cost vs latent SAB?
D. Explicit binding       : if run, is SABE < SABEI?
```

No single combined verdict.
