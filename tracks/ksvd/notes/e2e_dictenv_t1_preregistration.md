# E2E-DictEnv-T1 — pre-registration (frozen before any implementation run)

Round **E2E-DictEnv-T1** · protocol `e2e_dictenv_t1` · study `zinc-context-gap`.
Prior-artifact audit: [`e2e_dictenv_t1_prior_artifact_audit.md`](e2e_dictenv_t1_prior_artifact_audit.md).

This is the frozen specification.  It is written **before** any implementation,
cache build, training or GPU use.  Any deviation requires a written amendment
recorded here before the affected run.

---

## 1. Scientific question (the only question)

> The E2E-DictEnv-v0 sparse dictionary core is confirmed load-bearing,
> assignment-mediated, healthy and 2.15x better than its matched dense tied
> coordinate, but its absolute band is weak (`M_S` = 0.145508).  Is that ceiling
> caused by a **weak dictionary core**, or by an **unnecessarily compressed
> dictionary -> environment interface** and an **over-constrained
> reconstruction/task balance**?

T1 changes only:

```text
A. dictionary -> environment interface      (Stage A: 3 candidates)
B. reconstruction/task trade-off            (Stage B: 2 lambda arms)
C. training horizon                         (Stage C: conditional, <= 1 run)
```

Nothing else.  This is **not** an open-ended HPO.

---

## 2. Test-status discipline (frozen)

```text
official valid = DEVELOPMENT / TUNING set in E2E-DictEnv-T1
                 (architecture, lambda, horizon, checkpoints and soup are
                  selected on official-valid MAE)
official test  = TERMINAL REPORTING ONLY
official test is NOT project-wide pristine
                 (unrelated historical rounds already opened it; see the audit)
```

Absolute prohibitions, enforced in code and in this document:

```text
no test result -> architecture change
no test result -> lambda change
no test result -> horizon change
no test result -> checkpoint selection
no test result -> seed selection
no test result -> soup member selection
```

The test is loaded exactly once, after the freeze, by a dedicated
`terminal_test` stage (§15).

---

## 3. Frozen invariants (every candidate, every stage)

```text
phi_v  = exact audited R65 pure-topology coordinate (FSAR-R2-AR0)
K      = 32
s      = 8
IHT    = 10 unrolled tied-IHT steps
D_init = exact SDB-v0 K-SVD artifact (65x32, sha256 925d573a5808...)
         (never refit; column-normalized Dbar for encode + reconstruct)
tied encode/reconstruct, exact top-8, task-coupled D (grad reaches D)
no message passing, no recurrence, no pair->centre, no relation refresh,
no attention, no context writeback, no typed/parent lookup,
no edge dictionary, no multiple dictionaries, no per-shell dictionary,
no PCA/dense learned topology encoder, no FEC-S1 146->214->24 adapter,
no B-Full local encoder

environment width  = 48
static pair composer = exact FEC-S1 strict-static backend (15 359 params)
global branch      = FEC-S1 (62 -> 32 -> 32)
topology branch    = FEC-S1 (25 -> 16 -> 8)
reader             = GenericReader(302, (13, 13))
optimizer family   = Adam, lr 1e-3, wd 1e-5, batch 128, clip 5
scheduler          = none
batch size         = 128
seed (tuning)      = 0
checkpoint protocol= best official-valid MAE + fixed equal-weight Top-5 soup
```

Every candidate uses the identical backend tensors and semantics as v0.

---

## 4. Explicit coarse chemistry (new, allowed, bounded)

T1 exposes the exact **FEC-S0** factorized, train-fit-standardized 146-D
descriptor

```text
x_i^coarse in R^146 =
    atom_shell      84   (3 shells x 28 atom categories)
  + bond_shell      24   (6 shellpairs x 4 bond categories)
  + root_atom       28
  + incident_bonds   4
  + topology scalars 6   (log1p n, log1p m, shell2 boundary frac,
                          cycle_rank/n, root degree/4, mean degree/4)
  ------------------------------------------------
  total            146
```

built by the **FEC-S0 bit-identical construction + train scaler**
(`fec_s0_factorization.FactorizedFeatureTransform`, verified `max_abs = 0.0`).

Interpretation, fixed in advance:

> The 146-D coarse descriptor is **not** a learned fine structural bypass.  It
> contains only primitive chemistry, coarse rooted shell/shellpair routing and
> six fixed topology scalars.  The only **learned fine** structural coordinate
> is still

```text
alpha_v = IHT(D, phi_v).
```

The descriptor is read from `data.patch_cont` (the encoded 146-D tensor whose
FEC-S0 identity is re-checked by gate G0b).  `patch_cont` may **not** be fed to
any learned module other than the final environment decoder input described
below; no learned encoder may be interposed on it.

### 4.1 Relationship to v0's bond branch (recorded design resolution)

v0 formed the local environment input as
`[m_V(64); m_E(16); t_i(6)]` with a separate rank-16 bond marginal `m_E`.  T1's
environment decoder input is the spec's exact `[x_i^146 ; m_i^D]`; the v0
separate 16-D bond marginal is **removed**, because bond chemistry is already
carried by `x_i^146`'s `bond_shell` (24) and `incident_bonds` (4) blocks.  This
is exactly what the round spec's parameter algebra encodes (all three candidate
audits omit the 160 bond parameters and the decoder input widths are
210/290/338 = 146 + dict width).  It is therefore a designed part of the T1
interface, not an accident.

---

## 5. DenseTied control (unchanged definition)

```text
z_v = phi_v @ Pbar      (no IHT, no top-s, no threshold, no sparsity)
```

`P in R^{65x32}`, `P_init = D_init` (same matrix), identical column
normalization, identical reconstruction target, identical architecture and
hyper-parameters.  Used **only** at the frozen seed-1 confirmation (§13) and at
the terminal test (§15).  DenseTied is never re-tuned.

---

## 6. Stage A — interface architecture search (3 candidates, seed 0)

Common: official train 10 000 / official valid 1 000, seed 0, 240 epochs, fixed
Top-5 soup, `lambda_rec = lambda_0 = 135.834928` (the v0 frozen value, reused —
no recalibration).

Notation: `K = 32`, `q_v in R^28` atom one-hot, `s_iv in {0,1,2}` root shell.

### A1 — `A1_COARSE146_POOLED64`

```text
W_R in R^{32x64},  W_C in R^{28x64},  S in R^{3x64}
u_iv  = (alpha_v W_R) o (q_v W_C) o S_{s_iv} / sqrt(64)
m_i^D = sum_v u_iv                       in R^64      (SUM)
z_i   = [ x_i^146 ; m_i^D ]              in R^210
E_i   = SiLU MLP 210 -> 172 -> 48
```

Parameter audit (re-derived by code; hard gate G10):

```text
dictionary  2080
binding     4032   (W_R 2048 + W_C 1792 + S 192)
decoder    44596   (210*172+172 + 172*48+48)
local      50708
backend    15359
--------------------
total      66067    (FEC-S1 66170 -> delta -103, -0.16%)
```

### A2 — `A2_COARSE146_SLOT48`

```text
W_R in R^{32x48},  W_C in R^{28x48}      (no shell embedding)
m_{i,s}^D = sum_{v: s_iv = s} (alpha_v W_R) o (q_v W_C) / sqrt(48)
m_i^D     = [ m_{i,0} ; m_{i,1} ; m_{i,2} ]   in R^144
z_i       = [ x_i^146 ; m_i^D ]               in R^290
E_i       = SiLU MLP 290 -> 135 -> 48
```

Parameter audit:

```text
dictionary  2080
binding     2880   (W_R 1536 + W_C 1344)
decoder    45813   (290*135+135 + 135*48+48)
local      50773
backend    15359
--------------------
total      66132    (FEC-S1 66170 -> delta -38, -0.057%)
```

### A3 — `A3_COARSE146_SLOT64`

Same as A2 with `R = 64`:

```text
W_R in R^{32x64},  W_C in R^{28x64}
m_{i,s}^D in R^64,  m_i^D = [m_{i,0};m_{i,1};m_{i,2}] in R^192
z_i = [ x_i^146 ; m_i^D ]  in R^338
E_i = SiLU MLP 338 -> 116 -> 48
```

Parameter audit:

```text
dictionary  2080
binding     3840   (W_R 2048 + W_C 1792)
decoder    44940   (338*116+116 + 116*48+48)
local      50860
backend    15359
--------------------
total      66219    (FEC-S1 66170 -> delta +49, +0.074%)
```

Any mismatch between code and this algebra: STOP and locate; do not change widths
to "get close".

### A0 (reference, not re-trained)

```text
A0 = E2E-DictEnv-v0 SparseDictEnv, M_S = 0.145508  (reused durable result)
```

---

## 7. Stage-A correctness gates (all PASS before any formal Stage-A run)

```text
G0a  phi65 identity PASS (fresh build_phi vs FSAR cache vs env cache)
G0b  patch_cont146 identity PASS (FEC-S0 factorized train-fit scaler)
G1   chemistry purity of phi/alpha PASS (relabelling chemistry leaves
     phi, alpha, z unchanged)
G2   exact top-8 PASS (max l0 <= 8; exact-8 fraction >= 0.99)
G3   task gradient -> D nonzero PASS
G4   tied reconstruction PASS (perturb D changes code and reconstruction;
     no independent learned encoder)
G5   no dense fine-topology bypass PASS (alpha -> 0 removes the only learned
     fine coordinate; patch_cont enters only the decoder input)
G6   environment frozen under pair-relation mutation PASS (E bit-identical)
G7   no pair->centre PASS
G8   once-only composition PASS (pair/relation/env encoders exactly 1 call)
G9   relabel invariance PASS
G10  parameter identity PASS (Sparse == Dense == the candidate audit value)
G11  initialization matching PASS (shared tensors bit-identical at step 0;
     D == P init)
G12  official test blocked PASS
```

Shared backend tensors and semantics must be identical to v0.

---

## 8. Stage-A selection

Compare A0/A1/A2/A3 on the **primary** metric:

```text
fixed Top-5 soup official-valid MAE
```

Report for each candidate: best valid MAE, train MAE at best, best epoch,
dictionary health, reconstruction, wall time, parameter audit.  Only
health-PASS candidates participate.

Select

```text
A* = argmin_A MAE_valid,soup
```

(ties broken by lower best-valid MAE, then earlier best epoch).  This is an
explicit valid-set tuning decision; the report must not call it confirmation.

Interpretation rules (registered):

* if A2/A3 clearly beat A1 -> preserving root-relative shell localization into
  the nonlinear decoder is supported;
* if A1 ~ A2 ~ A3 -> coarse-information exposure matters more than delayed shell
  mixing;
* if no candidate beats A0 by >= 0.002 -> record
  `INTERFACE_UPGRADE_NO_MATERIAL_GAIN` (Stage B may still proceed, but the
  architecture search itself produced no material gain).

---

## 9. Stage B — reconstruction-weight tuning (2 arms)

Fix `A*`.  Do **not** re-search the architecture.  `B0` = the Stage-A winner run
at `lambda_0 = 135.834928` (reused).  Two new arms:

```text
B1 = 0.50 * lambda_0 = 67.917464
B2 = 0.25 * lambda_0 = 33.958732
```

Same architecture, seed 0, 240 epochs, Top-5 soup.  Forbidden:
`lambda = 0`, any other value, per-candidate recalibration.

### 9.1 Stage-B health condition (a lambda arm is eligible only if all hold)

```text
exact top-8
>= 80% of the initial active-atom coverage retained
effective atom count >= 8
no atom > 50% support
task gradient -> D nonzero
environment effective rank > 1
train/valid dictionary usage stable
normalized valid reconstruction <= 0.01
```

`0.01` is a structural-anchor integrity ceiling, not a performance target.

### 9.2 Stage-B selection

Among `B0/B1/B2`, choose the lowest eligible

```text
fixed Top-5 soup official-valid MAE  ->  lambda*
```

Report: train MAE improvement, valid MAE improvement, reconstruction
degradation, D movement, effective vocabulary.

---

## 10. Stage C — conditional horizon extension (<= 1 run)

Trigger **only if** the final Stage-B winner has

```text
best_epoch >= 220   OR   any Top-5 soup member epoch >= 230
```

Then run one extra Sparse run with identical architecture / lambda / seed /
optimizer / data order and `max_epochs = 320` (no scheduler change).  Compare
the 240-epoch and 320-epoch valid soups; adopt 320 **only if**

```text
MAE_240 - MAE_320 >= 0.001
```

Otherwise keep 240.  No 400/500-epoch run is authorized.

---

## 11. Explicit tuning budget

```text
Stage A: 3 new full Sparse runs
Stage B: 2 new full Sparse runs
Stage C: <= 1 new full Sparse run
--------------------------------------
maximum 6 new full Sparse runs  (A0 reused, never re-run)
```

After the budget is spent, architecture tuning stops regardless of the result.

---

## 12. Forbidden during the tuning phase

```text
K, s, IHT steps, dictionary initialization, different dictionary,
edge dictionary, multiple/per-shell dictionaries, attention,
message passing, recurrence, pair-composer redesign, global-branch redesign,
reader width/depth, optimizer family, learning rate, weight decay,
batch size, activation search
```

---

## 13. Final configuration freeze + paired seed-1 confirmation

After Stage A/B/C, choose `architecture*`, `lambda*`, `horizon*` and write

```text
results/e2e_dictenv_t1/architecture_freeze.json
```

containing: config id, architecture source hash (sha256 of the module), commit,
parameter count, `K`, `s`, IHT steps, `D` init sha256, `lambda`, horizon, seed-0
winner state hashes, Top-5 soup member epochs, Top-5 soup tensor hashes,
official-valid metrics, and

```text
official_test_loaded_at_freeze_time = false FOR THIS ROUND
```

(The `false` only asserts that this round's freeze happened before this round's
test load; it does not claim project-wide test purity, since unrelated historical
rounds already opened the test.)

Then, and only then, run the paired confirmation at **seed 1**:

```text
final Sparse
final DenseTied     (same architecture / init-knobs / data order / optimizer /
                     lambda / horizon / soup protocol; z = phi @ Pbar)
```

DenseTied may not re-search its own architecture or lambda.  Valid is still used
for checkpoint and soup selection at seed 1, but **not** to modify the config.

---

## 14. Reporting layers (must not be merged)

```text
Development result : valid-guided tuning (seed 0 Sparse tuning ceiling estimate)
Confirmation result: frozen-config seed-1 Sparse vs DenseTied
Terminal test result: frozen states on official test (reporting only)
```

Primary dictionary-specific confirmation:

```text
G_sparse^seed1 = M_Dense^seed1 - M_Sparse^seed1
```

Reproduce: if `G_sparse^seed1 >= 0.003` ->
`dictionary-specific effect replicates under the frozen tuned architecture`;
otherwise -> tuning improved Sparse absolute performance but dictionary-specific
superiority is unstable.  The large seed-0 v0 gap may **not** substitute for the
seed-1 result.

---

## 15. Terminal official test (one load, reporting only)

Only after all of the following exist:

```text
all tuning complete
architecture_freeze.json written and committed
seed-0 selected/soup states frozen (hashes recorded)
seed-1 Sparse/Dense selected/soup states frozen (hashes recorded)
all valid metrics recorded
no further architecture/HPO authorised
```

write `results/e2e_dictenv_t1/official_test_unlock.json` with

```text
user_authorised_test_read = true
purpose                   = terminal reporting only
project_wide_pristine     = false
reason                    = historical unrelated ZINC test reads already exist
architecture_frozen_before_this_rounds_test_read = true
test_will_not_affect_any_model_config_checkpoint_decision = true
```

and then, in a single dedicated `terminal_test` process, load the test split
exactly once (`_load_zinc(..., "test")`, via the canonical
`ztraining.extract_test_records` + `ztraining.build_encoded`) and evaluate the
pre-frozen objects:

```text
final Sparse seed0 soup
final Sparse seed1 soup
final DenseTied seed1 soup
v0 Sparse seed0 soup      (only if the exact historical state is persisted+SHA;
                           never rebuilt)
FEC-S1 seed0 soup         (only if its member states are exactly recoverable;
                           otherwise SKIP; never retrained)
```

Report test MAE, mean prediction, prediction std; for Sparse also zero-code test
MAE and 5x assignment-shuffle test MAE (mechanism generalization diagnostics).
No gate may trigger a new run after seeing test.  The round ends either way.

---

## 16. Mechanism interventions (frozen final Sparse states)

### zero-code

```text
alpha_v -> 0 ; coarse146 / chemistry / backend kept
G_dict-use = M_zero - M_S     target >= 0.010
```

### assignment shuffle

within each `(root, shell)`, shuffle `alpha_v <-> q_v`, preserving the alpha
multiset, q multiset, shell, coarse146, bond content, global context;
5 fixed permutations; report the mean

```text
G_assign = M_shuffle - M_S    target >= 0.010
```

---

## 17. Tuning success bands (tuning seed-0 valid soup)

```text
Breakthrough : M_S <= 0.130
Strong       : 0.130 < M_S <= 0.135
Viable       : 0.135 < M_S <= 0.145
Weak         : M_S > 0.145
```

These describe the tuning result only; the final scientific conclusion also uses
the frozen seed-1 pair and the mechanism interventions.

---

## 18. Final verdict categories (precedence order)

| case | condition | verdict |
|---|---|---|
| 1 | seed0 tuned Sparse viable/strong (`<= 0.135` for STRONG prefix, `<= 0.145` for VIABLE) and seed1 Sparse viable and `G_sparse^seed1 >= 0.003` and zero/shuffle/health PASS | `E2E_DICTENV_TUNED_STRONG_AND_DICTIONARY_SPECIFIC` (if `<= 0.135`) / `E2E_DICTENV_TUNED_VIABLE_AND_DICTIONARY_SPECIFIC` (if `<= 0.145`) |
| 2 | absolute tuning clearly improves but `G_sparse^seed1 < 0.003` | `E2E_DICTENV_TUNED_PERFORMANCE_GAIN_DICTIONARY_SPECIFIC_UNSTABLE` |
| 3 | best tuned result improves on v0 `0.145508` by `< 0.002` | `E2E_DICTENV_TUNING_NO_MATERIAL_ABSOLUTE_GAIN` |
| 4 | performance may improve but health / zero-code / assignment-shuffle core mechanism fails | `E2E_DICTENV_TUNED_MECHANISM_LOST` |

Precedence: mechanism loss (4) first, then no-material-gain (3), then
specificity-unstable (2), then 1.

---

## 19. Durable artifacts

```text
tracks/ksvd/notes/e2e_dictenv_t1_{prior_artifact_audit,preregistration,
    implementation,analysis,test_read}.md

tracks/ksvd/results/e2e_dictenv_t1/
    parameter_audit.json
    correctness.json
    stage_a_a1.json  stage_a_a2.json  stage_a_a3.json
    stage_b_lambda050.json  stage_b_lambda025.json
    stage_c_horizon320.json          # only if triggered
    tuning_decision.json
    final_sparse_seed1.json
    final_dense_seed1.json
    mechanism_zero.json  mechanism_shuffle.json
    dictionary_health.json
    architecture_freeze.json
    official_test_unlock.json
    official_test_results.json
    REPORT.md  DECISION.md

plus claim YAML, decision YAML and the STATE.yaml update.
```

JSON/PT/CSV under `results/` follow the repo-wide gitignore convention (no
force-add).

---

## 20. Stop rule

If after the maximum budget the best tuned Sparse remains ~0.145 and no
interface/lambda/horizon change produces a material gain, the round accepts:

> the dictionary mechanism is established, but the current dictionary-first
> environment class has a material absolute-performance ceiling.

No rescue to K/s search, no 400/500-epoch run, no reader/attention/MP change is
authorized by this round.
