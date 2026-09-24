# E2E-DictEnv-T1 — prior-artifact audit (round 1, before any code)

Round name: **E2E-DictEnv-T1** (*End-to-End Sparse Dictionary-Core Chemical
Environment, tuning round*).  Protocol id: `e2e_dictenv_t1`.  Study
`zinc-context-gap`.  Written **before** any implementation, cache build, training
or GPU use.

Local audit base:

```text
git rev-parse --show-toplevel        = /home/calendar/code/research
git rev-parse HEAD                    = b233eac314d34b1148248c84cc56e53be01ca04a
git log -1 --oneline                  = b233eac record(e2e-dictenv-v0): E2E_DICTENV_ABSOLUTE_WEAK
untracked (local)                     = tracks/ksvd/results/fec_s1/  (git-ignored working evidence)
remote (res) branch=main commit=eeeb6b34f641260a0373a3daeda1286a855717d7 state=clean
```

This note modifies no historical record.  It exists to fix the round's
scientific boundary: T1 is **not** a re-run of E2E-DictEnv-v0 and **not** a K/s
search.  It absorbs the v0 result and asks the next question.

---

## 0. The round's single question

> The E2E-DictEnv-v0 sparse dictionary core was confirmed **load-bearing,
> assignment-mediated, healthy, and sharply better than its matched dense tied
> coordinate** (`G_sparse` = +0.168), yet it landed at
> `M_S` = 0.145508 — the *weak* absolute band, +0.015086 behind the FEC-S1
> static anchor.  Is that ceiling caused by a weak dictionary core, or by an
> unnecessarily compressed **dictionary -> environment interface** and an
> over-constrained **reconstruction/task balance**?

T1 therefore tunes **only**:

```text
A. dictionary -> environment interface   (Stage A, 3 candidates)
B. reconstruction/task trade-off         (Stage B, 2 lambda arms)
C. training horizon                      (Stage C, conditional, <= 1 run)
```

and forbids K/s/IHT/dictionary-count/attention/MP/recurrence/reader/optimizer
search.

---

## 1. Test-status audit (mandatory first)

The repository has **already opened the official ZINC test** in unrelated
historical rounds.  It is therefore **not project-wide pristine**.

Recorded terminal reads (non-exhaustive):

| artifact | split | note |
|---|---|---|
| `notes/local_token_null_test_read.md` | ZINC official test (n=1000) | B-Null / Constant-16 post-hoc terminal read |
| `records/decisions/decision-local-token-null-test-read-20260918.yaml` | ZINC + MolHIV | authorised one-shot test read, both unlock files spent |
| `notes/compact_v4_recurrent_pair_centre_capacity_test_closure.md` | ZINC official test | cell-A typed closure (soup 0.106717) |
| `notes/structural_encoder_mechanism_decomposition_test_closure.md` | ZINC official test | mechanism-decomposition closure |
| `notes/molhiv_bnull_small_head_test_read.md` | MolHIV official test | terminal read |

Consequences, fixed for T1:

```text
official valid = DEVELOPMENT / TUNING set in E2E-DictEnv-T1
                 (architecture, lambda and horizon are selected on it)
official test  = TERMINAL REPORTING ONLY
official test is NOT project-wide pristine
official test is loaded AT MOST ONCE, after the full freeze
no test result may change any architecture / lambda / horizon /
checkpoint / seed / soup decision
```

This matches the repo guardrail (`STATE.yaml`):

> "official ZINC and MolHIV tests are SPENT and no longer pristine; they are
> reporting-only, must never drive selection, and re-opening requires a new
> preregistration + explicit authorization".

T1 has that preregistration (`e2e_dictenv_t1_preregistration.md`) and the
user's explicit authorization recorded in §24 of the round spec.  The round
still **must not** describe the test as an unbiased untouched benchmark.

---

## 2. v0 — the result being built on (positive mechanism, weak absolute)

`notes/e2e_dictenv_v0_analysis.md`, `results/e2e_dictenv_v0/`:

```text
E2E-DictEnv-v0 formal commit eeeb6b34, remote A100-SXM4-40GB, seed 0,
official train 10 000 / official valid 1 000, official test never loaded.

SparseDictEnv soup valid MAE  M_S = 0.145508   (best 0.150982 @ 228)
DenseTiedEnv  soup valid MAE  M_D = 0.313047
G_sparse   = +0.167539   (gate >= 0.003)  PASS
M_zero     = 1.000347    G_dict-use = +0.854839 (gate >= 0.010) PASS
M_shuffle  = 0.162797    G_assign   = +0.017289 (gate >= 0.010) PASS
dictionary health PASS: 27/32 active, effective 13.89, exact top-8,
                        usage Spearman 0.998, effective rank 10.83,
                        D movement 5.94, rec 9.7e-5
frozen verdict: E2E_DICTENV_ABSOLUTE_WEAK  (case D: M_S > 0.145)
```

`A0 = E2E-DictEnv-v0 SparseDictEnv` is reused as the tuning reference.  T1
**does not** retrain `A0`.

### 2.1 v0's dictionary -> environment interface (the thing T1 changes)

```text
coord = IHT(D, phi65)              # alpha_v in R^32, exact top-8
q_v   = atom one-hot               # 28
r_v   = [1 ; alpha_v]              # 33  (explicit chemistry-marginal anchor)
u_iv  = (r_v W_R) o (q_v W_C) o S_{s_iv} / sqrt(64)
m_V   = sum_v u_iv    in R^64      # SUM
w_ie  = (b_e W_B) o P_{p_ie} / sqrt(16)
m_E   = sum_e w_ie    in R^16      # SUM
z_i   = [m_V(64) ; m_E(16) ; t_i(6)]      # 86
E_i   = SiLU MLP 86 -> 329 -> 48
```

The **coarse chemistry** in v0 is therefore: (a) the atom one-hot `q_v` bound
through the rank-64 latent, (b) the bond one-hot `b_e` bound through a separate
rank-16 branch, and (c) the six pure-topology scalars `t_i`.  There is **no**
146-D `patch_cont` in the model.

---

## 3. FEC-S0 / FEC-S1 — the exact coarse descriptor T1 is allowed to use

* **FEC-S0** (`notes/fec_s0_equivalence_analysis.md`) factorized the historical
  146-D `patch_cont` into explicit role x primitive sums and proved the
  reconstruction is **bit-identical** to the historical encoded tensor
  (`local_descriptor_equivalence.json`, `descriptor identity max_abs = 0.0`),
  with the **train-fit** `Standardizer`.
* **FEC-S1** (`notes/fec_s1_analysis.md`) reused exactly that descriptor as the
  input of a parameter-matched shared adapter `146 -> 214 -> 24` and reached
  seed-0 soup **0.130422** (best 0.136783 @ 238) — the strongest frozen static
  anchor, and it verified `adapter_input == batch.patch_cont` bit-identically.

T1 is allowed to expose that **exact** 146-D coarse descriptor
(atom_shell 84 + bond_shell 24 + root_atom 28 + incident_bonds 4 + six topology
scalars = 146) to the environment decoder.  It is **not** a learned fine
structural encoder: its construction and train scaler are frozen, and the only
learned fine structural coordinate remains `alpha_v = IHT(D, phi_v)`.

Forbidden in T1 (round spec §4): the FEC-S1 `146 -> 214 -> 24` adapter, typed
lookup, parent lookup, a PCA structural coordinate, a dense learned topology
encoder, or the B-Full local encoder.

---

## 4. FEC-S1 / FEC-D1 / SDB / TCCD — mandatory prior conclusions

### 4.1 FEC-S1 (strong static line; context only)

```text
FEC-S1 seed-0 soup = 0.130422, best 0.136783 @ 238
train MAE at best  = 0.098696, train minimum 0.094507
```

T1 seed-0 `train MAE at best` for v0 was 0.115179.  The train-valid gaps are
comparable (v0 0.0358, FEC-S1 0.0381).  The working diagnosis is therefore
**representation / fitting capacity**, not over-fitting.  FEC-S1 is a historical
anchor for the *band*, never a matched architecture comparison.

### 4.2 FEC-D1 (localized binding supported, dictionary NOT specific)

```text
Dict32 localized binding soup = 0.131537
PCA32 localized binding soup  = 0.130231
shuffle                       = 0.144669
```

Localization + chemistry binding into a frozen strong environment helps
(+0.0052) and is assignment-mediated (+0.0131), but a sparse K32/s8 dictionary
is not better than a matched dense PCA32 of the same `phi65` **at that
placement**.  T1 does not repeat FEC-D1: T1 keeps the dictionary as the **only
fine structural coordinate** (as in v0) and changes only the interface and the
reconstruction anchor.  v0 already showed the matched *dense tied* coordinate
(`z = phi @ Dbar`, same matrix) is 2.15x worse, so the FEC-D1 PCA refutation does
not transfer to this placement.

### 4.3 SDB-v0 (validated dictionary substrate; reused read-only)

* `phi_v in R^65` (FSAR-R2-AR0), `K=32`, `s=8`, exact top-8 tied-IHT.
* Stage 1: 32/32 atoms used, exact sparsity 8; K-SVD fit `E_bind` sparse 3.45e-4
  vs dense 1.60e-5 on the same `D` (sparse/dense ratio 21.6).
* Stage 2 (3 seeds): recovery 1.159 vs the frozen `phi65` oracle; shuffle
  degradation 0.245.
* Stage 3: task-coupled `D` beats frozen `D` by only 0.009097 (single seed).
* Stage 4: on the strong S0 backbone the branch supplies only +0.002381
  (< 0.003 material gate).

T1 reuses the frozen K-SVD dictionary
`results/sdb_v0/dictionary.pt` (65x32, sha256
`925d573a58083c3c20f36980dec9bba77dff592b30ea1abf11098ca3a7b7487a`) and the
exact `K=32 / s=8 / 10 IHT steps` semantics.  It never refits, never sweeps.

### 4.4 TCCD

Closed: task-learned soft prototypes can be healthy and assignment-sensitive but
a prototype/whole-attributed-patch representation is an insufficient standalone
predictor (TCCD-v7 normalized-moment standalone costs 0.133754 MAE).  T1 does
**not** resurrect soft prototypes; the dictionary stays exact-top-8 tied-IHT over
pure topology.

---

## 5. Other closed lines (dedupe)

| prior | object | why not repeated |
|---|---|---|
| GSCN-v0 | dictionary as message-passing transition | closed; T1 has no message passing |
| SDPK / SRDA / DTX | dictionary in the pair kernel / relation algebra / aligned graph cross | closed; T1 has no pair-side dictionary |
| FEC-D0 | within-shell residual subrole basis | closed (no held-out subrole signal) |
| PEC family | pure static composition interface | closed; T1 keeps the FEC-S1 strict-static backend |
| TCCD family | soft prototype / normalized-moment predictor | closed |
| FEC-S1 adapter | `146 -> 214 -> 24` learned local encoder | forbidden in T1 |

---

## 6. What is genuinely new in T1

1. **Explicit coarse chemistry next to the sparse dictionary.**  v0 forced the
   DC chemistry marginal and the dictionary assignment through *the same*
   factorized path (`r_v = [1; alpha_v]`).  T1 exposes the exact FEC-S0
   bit-identical 146-D coarse descriptor to the environment decoder while
   keeping `alpha = IHT(D, phi)` as the only *learned fine structural*
   coordinate.  This is the "coarse chemistry + sparse role code" interface the
   round spec asks for.
2. **Interface geometry search, not a mechanism search.**  The three candidates
   differ only in whether the dictionary x chemistry occurrence vector is
   projected to a pooled 64-D code (A1, delay shell mixing) or kept
   shell-localized per slot (A2/A3), and in the slot width.  K/s/IHT/D-init are
   frozen.
3. **Reconstruction-anchor tuning.**  Two registered lambda arms (0.5x, 0.25x of
   the frozen v0 `lambda_0`) test whether v0's reconstruction anchor is
   over-constraining task adaptation.
4. **Conditional horizon extension** with an explicit, pre-registered trigger
   (`best_epoch >= 220` or a Top-5 soup member at `epoch >= 230`), one extra
   run maximum, 320 epochs, no scheduler change.

---

## 7. Dedupe verdict

**No equivalent implementation exists.**  The closest objects:

| prior | decisive difference from T1 |
|---|---|
| FEC-S1 | learned `146 -> 214 -> 24` adapter, no dictionary, no sparse role code |
| v0 (A0) | dictionary x chemistry through one factorized path with a DC anchor; no explicit coarse 146-D; separate 16-D bond marginal; `86 -> 329 -> 48` decoder |
| FEC-D1 | dictionary/PCA as a residual branch on a frozen strong environment |
| SDB Stage 4 | graph-level linear residual on a strong S0 |

T1's only scientific content is the valid-guided selection among A1/A2/A3, then
the `lambda` arms, then the conditional horizon, followed by a **frozen** seed-1
Sparse vs DenseTied confirmation and a terminal test report.  Any clear negative
stops the tuning with no rescue.

---

## 8. Proceed / stop prior

* Proceed with a **mixed prior**: v0 proves the *mechanism* (`G_sparse` = +0.168,
  `G_dict-use` = +0.855, `G_assign` = +0.017, health PASS), but every historical
  *absolute* anchor (v0 0.1455, SDB-4, FEC-D1) says the sparse-dictionary-first
  environment class has not yet beaten the strong static line.  T1 is the first
  round that gives the dictionary core a fair coarse-chemistry interface.
* If A2/A3 + a reduced reconstruction anchor push `M_S` into `0.13x` while
  Sparse still beats DenseTied and zero/shuffle remain strong, that is the
  strongest evidence to date that a sparse end-to-end structural dictionary can
  be the predictive core of a no-message-passing molecular environment.
* If the tuning budget is exhausted at ~0.145, the correct conclusion is:
  *dictionary mechanism established, but the current dictionary-first
  environment class has a material absolute-performance ceiling.*  No unbounded
  rescue.
* K/s/IHT tuning is **not** authorized this round; it is a separate scientific
  question (structural vocabulary capacity / sparsity trade-off).
